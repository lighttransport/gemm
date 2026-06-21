/*
 * vlm_runner.c - CLI for the a64fx Qwen3-VL vision encoder runner.
 *
 * Usage:
 *   vlm_runner <model.gguf> <mmproj.gguf> <image> [options]
 *
 * Options:
 *   --dtype fp32|bf16|fp16    (default: fp32; bf16/fp16 fall back to fp32 in M1)
 *   --threads N
 *   --dump DIR                 write per-stage VLMD tensor dumps to DIR
 *   --bench N                  re-encode N times, report median
 *   --image-size S             longer-side target (default 384)
 *   --prompt "..."             accepted; ignored (vision only)
 *   --no-prof                  silence profiler if enabled
 *
 * model.gguf is required positional to match cpu/vlm/test_vision.c's CLI
 * shape, but is otherwise unused by this runner.
 */

#define GGUF_LOADER_IMPLEMENTATION
#include "../../../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../../common/ggml_dequant.h"

#include "../../../common/qtensor_utils.h"   /* qtensor struct */

#define VISION_ENCODER_IMPLEMENTATION
#include "../../../common/vision_encoder.h"

#define IMAGE_UTILS_IMPLEMENTATION
#include "../../../common/image_utils.h"

#include "vit_a64fx.h"
#include "vlm_parallel.h"
#include "tensor_dump.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

/* ── small helpers ── */

static double mono_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int parse_dtype(const char *s) {
    if (!strcasecmp(s, "fp32")) return VIT_DTYPE_FP32;
    if (!strcasecmp(s, "bf16")) return VIT_DTYPE_BF16;
    if (!strcasecmp(s, "fp16")) return VIT_DTYPE_FP16;
    return -1;
}

static const char *dtype_name(int dt) {
    switch (dt) {
        case VIT_DTYPE_FP32: return "fp32";
        case VIT_DTYPE_BF16: return "bf16";
        case VIT_DTYPE_FP16: return "fp16";
        default:             return "?";
    }
}

static int arg_is_image(const char *s) {
    if (!s) return 0;
    int n = (int)strlen(s);
    if (n < 4) return 0;
    const char *e4 = s + n - 4;
    const char *e5 = (n >= 5) ? s + n - 5 : NULL;
    if (!strcasecmp(e4, ".png")) return 1;
    if (!strcasecmp(e4, ".jpg")) return 1;
    if (!strcasecmp(e4, ".bmp")) return 1;
    if (!strcasecmp(e4, ".ppm")) return 1;
    if (e5 && !strcasecmp(e5, ".jpeg")) return 1;
    return 0;
}

static uint8_t *make_checkerboard(int w, int h, int cell) {
    uint8_t *img = (uint8_t *)malloc((size_t)w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int c = ((x / cell) + (y / cell)) & 1;
            uint8_t v = c ? 255 : 0;
            size_t i = ((size_t)y * w + x) * 3;
            img[i + 0] = v; img[i + 1] = v; img[i + 2] = v;
        }
    }
    return img;
}

static int dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ── usage / main ── */

static void usage(const char *p) {
    fprintf(stderr,
        "Usage: %s <model.gguf> <mmproj.gguf> [image] [options]\n"
        "Options:\n"
        "  --dtype fp32|bf16|fp16   (default fp32)\n"
        "  --threads N\n"
        "  --dump DIR\n"
        "  --bench N                (re-encode N times)\n"
        "  --image-size S           (longer-side target, default 384)\n"
        "  --prompt \"...\"\n"
        "  --no-prof\n", p);
}

int main(int argc, char **argv) {
    /* The encoder issues ~250 fork/join OMP regions per encode. With the
     * default passive wait policy, idle workers sleep between regions and
     * pay a cross-CMG futex wakeup each time — this was the "24→48T wall"
     * (Task #15): 48T barely beat 24T. Spinning workers (active) fixes it
     * (~150→~220 tok/s at 48T). setenv with overwrite=0 so an explicit
     * OMP_WAIT_POLICY in the environment still wins. Must run before the
     * OMP runtime initialises (first parallel region). */
    setenv("OMP_WAIT_POLICY", "active", 0);
    setenv("OMP_PROC_BIND",   "close",  0);
    setenv("OMP_PLACES",      "cores",  0);

    if (argc < 3) { usage(argv[0]); return 1; }

    const char *model_path = argv[1];   /* required, unused */
    const char *mmproj_path = argv[2];
    const char *image_path  = NULL;
    const char *user_prompt = "Explain the image";
    const char *dump_dir    = NULL;
    int  dtype       = VIT_DTYPE_FP32;
    int  threads     = 0;
    int  bench       = 1;
    int  img_size    = 384;
    int  enable_prof = 1;
    (void)model_path; (void)user_prompt;

    for (int i = 3; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--dtype") && i + 1 < argc) {
            int d = parse_dtype(argv[++i]);
            if (d < 0) { fprintf(stderr, "unknown dtype '%s'\n", argv[i]); return 1; }
            dtype = d;
        } else if (!strcmp(a, "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (!strcmp(a, "--dump") && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (!strcmp(a, "--bench") && i + 1 < argc) {
            bench = atoi(argv[++i]);
            if (bench < 1) bench = 1;
        } else if (!strcmp(a, "--image-size") && i + 1 < argc) {
            img_size = atoi(argv[++i]);
        } else if (!strcmp(a, "--prompt") && i + 1 < argc) {
            user_prompt = argv[++i];
        } else if (!strcmp(a, "--no-prof")) {
            enable_prof = 0;
        } else if (arg_is_image(a)) {
            image_path = a;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            usage(argv[0]);
            return 1;
        }
    }

    fprintf(stderr, "vlm_runner: dtype=%s threads=%d bench=%d image_size=%d\n",
            dtype_name(dtype), threads, bench, img_size);
    if (dump_dir) fprintf(stderr, "vlm_runner: dump dir = %s\n", dump_dir);

    /* ── load mmproj ── */
    fprintf(stderr, "Loading mmproj: %s\n", mmproj_path);
    gguf_context *g = gguf_open(mmproj_path, 1);
    if (!g) { fprintf(stderr, "failed to open mmproj\n"); return 1; }
    vision_model *vm = vision_load(g);
    if (!vm) { fprintf(stderr, "failed to load vision model\n"); return 1; }

    /* ── load or synth image ── */
    int img_w = img_size, img_h = img_size;
    int grid = vm->patch_size * vm->spatial_merge;
    uint8_t *img_rgb = NULL;
    if (image_path) {
        int sw = 0, sh = 0;
        uint8_t *src = img_load(image_path, &sw, &sh);
        if (!src) { fprintf(stderr, "failed to load image '%s'\n", image_path); return 1; }
        int long_side = (sw > sh) ? sw : sh;
        float s = (float)img_size / (float)long_side;
        int dw = (int)(sw * s); if (dw < grid) dw = grid;
        int dh = (int)(sh * s); if (dh < grid) dh = grid;
        dw = (dw / grid) * grid;
        dh = (dh / grid) * grid;
        fprintf(stderr, "loaded %dx%d → resize %dx%d (grid %d)\n", sw, sh, dw, dh, grid);
        img_rgb = img_resize_ac(src, sw, sh, dw, dh);
        free(src);
        img_w = dw; img_h = dh;
    } else {
        if (img_w % grid != 0) img_w = (img_w / grid) * grid;
        if (img_h % grid != 0) img_h = (img_h / grid) * grid;
        if (img_w < grid) img_w = grid;
        if (img_h < grid) img_h = grid;
        fprintf(stderr, "synthesizing %dx%d checkerboard\n", img_w, img_h);
        img_rgb = make_checkerboard(img_w, img_h, 64);
    }

    fprintf(stderr, "normalizing...\n");
    float *img_norm = vision_normalize_image(vm, img_rgb, img_w, img_h);
    free(img_rgb);

    /* ── thread pool ── */
    vlm_pool *pool = vlm_pool_init(threads);
    if (!pool) { fprintf(stderr, "failed to init pool\n"); return 1; }
    fprintf(stderr, "thread pool size = %d\n", vlm_pool_size(pool));

    /* ── dump writer ── */
    vlmd_writer writer;
    if (vlmd_writer_open(&writer, dump_dir) != 0) return 1;

    /* ── build weight cache (one-time dequant + transpose; BF16 if requested) ── */
    double t_cache0 = mono_sec();
    struct vit_a64fx_cache *cache = vit_a64fx_cache_build(vm, dtype);
    fprintf(stderr, "weight cache built in %.3f s (storage=%s)\n",
            mono_sec() - t_cache0, dtype_name(dtype));

    /* Optional NUMA replication across CMGs (M6 B3.2). */
    {
        const char *e = getenv("VLM_NUMA");
        int n_cmgs = (e && *e) ? atoi(e) : 0;
        if (n_cmgs > 0) {
            double t0 = mono_sec();
            if (vit_a64fx_cache_replicate(cache, n_cmgs) != 0) {
                fprintf(stderr, "vit_a64fx_cache_replicate(%d) failed\n", n_cmgs);
                return 1;
            }
            fprintf(stderr, "cache replicated across %d CMGs in %.3f s\n",
                    n_cmgs, mono_sec() - t0);
        }
    }

    /* ── encode (with optional benching) ── */
    vit_a64fx_opts opts = {
        .dtype = dtype,
        .pool  = pool,
        .dump  = writer.enabled ? &writer : NULL,
        .enable_prof = enable_prof,
        .cache = cache,
    };

    double *times = (double *)calloc(bench, sizeof(double));
    float *embd = NULL;
    int n_merged = 0, embd_dim = 0;
    for (int it = 0; it < bench; it++) {
        if (it > 0 && embd) { free(embd); embd = NULL; }
        /* only dump on first iter to keep dumps deterministic and cheap */
        opts.dump = (it == 0 && writer.enabled) ? &writer : NULL;
        double t0 = mono_sec();
        embd = vit_a64fx_encode(vm, img_norm, img_w, img_h, &opts,
                                &n_merged, &embd_dim);
        double t1 = mono_sec();
        times[it] = t1 - t0;
        if (!embd) { fprintf(stderr, "encode failed (iter %d)\n", it); return 1; }
        fprintf(stderr, "iter %d/%d: %.3f s\n", it + 1, bench, times[it]);
    }
    free(img_norm);
    vlmd_writer_close(&writer);

    /* ── stats over runs ── */
    qsort(times, bench, sizeof(double), dcmp);
    double tmin = times[0];
    double tmax = times[bench - 1];
    double tmed = (bench & 1) ? times[bench / 2]
                              : 0.5 * (times[bench / 2 - 1] + times[bench / 2]);

    /* ── embedding summary ── */
    {
        float vmin = embd[0], vmax = embd[0], norm = 0.0f;
        size_t total = (size_t)n_merged * embd_dim;
        for (size_t i = 0; i < total; i++) {
            if (embd[i] < vmin) vmin = embd[i];
            if (embd[i] > vmax) vmax = embd[i];
            norm += embd[i] * embd[i];
        }
        fprintf(stderr, "\n=== Result ===\n");
        fprintf(stderr, "tokens=%d  dim=%d  min=%.4f  max=%.4f  norm=%.4f\n",
                n_merged, embd_dim, vmin, vmax, sqrtf(norm));
        fprintf(stderr, "time: min=%.3f s  median=%.3f s  max=%.3f s  (%d iters)\n",
                tmin, tmed, tmax, bench);
        fprintf(stderr, "tokens/s (median): %.2f\n", n_merged / tmed);
    }

    free(times);
    free(embd);
    vit_a64fx_cache_free(cache);
    vlm_pool_free(pool);
    vision_free(vm);
    gguf_close(g);
    return 0;
}
