/*
 * test_hip_qimg.c - Qwen-Image end-to-end text-to-image test (HIP/RDNA4)
 *
 * Modes:
 *   --test-init    : Initialize HIP, compile kernels
 *   --test-dit     : Run single DiT step on GPU
 *   --test-vae     : VAE decode test
 *   --generate     : Full text-to-image pipeline (text encoder on CPU)
 *
 * Build:
 *   make
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION
#define QIMG_TEXT_ENCODER_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/gguf_loader.h"
#include "../../common/bpe_tokenizer.h"
#include "../../common/transformer.h"
#include "../../common/qwen_image_scheduler.h"
#include "../../common/stb_image_write.h"
#include "../rocew.h"
#include "../llm/hip_llm_runner.h"  /* must be before text_encoder.h for GPU path */
#include "../../common/qwen_image_text_encoder.h"
#include "hip_qimg_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- PRNG: Box-Muller ---- */
static uint64_t rng_state = 42;
static int rng_cached_valid = 0;
static float rng_cached = 0.0f;

static float randn(void) {
    if (rng_cached_valid) { rng_cached_valid = 0; return rng_cached; }
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    rng_cached = (float)(r * sin(theta));
    rng_cached_valid = 1;
    return (float)(r * cos(theta));
}

/* CHW float [-1,1] → HWC uint8 [0,255] */
static unsigned char *rgb_to_u8(const float *rgb, int h, int w) {
    unsigned char *u8 = (unsigned char *)malloc((size_t)h * w * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++) {
                float v = rgb[c * h * w + y * w + x];
                v = v * 0.5f + 0.5f;
                v = v < 0 ? 0 : (v > 1 ? 1 : v);
                u8[(y * w + x) * 3 + c] = (unsigned char)(v * 255.0f + 0.5f);
            }
    return u8;
}

static int ends_with_icase(const char *s, const char *suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    if (lf > ls) return 0;
    for (size_t i = 0; i < lf; i++) {
        char a = s[ls - lf + i], b = suf[i];
        if (a >= 'A' && a <= 'Z') a += 32;
        if (b >= 'A' && b <= 'Z') b += 32;
        if (a != b) return 0;
    }
    return 1;
}

static void save_image(const char *path, const float *rgb, int h, int w) {
    unsigned char *u8 = rgb_to_u8(rgb, h, w);
    int ok = 0;
    if (ends_with_icase(path, ".png")) {
        ok = stbi_write_png(path, w, h, 3, u8, w * 3);
    } else if (ends_with_icase(path, ".jpg") || ends_with_icase(path, ".jpeg")) {
        ok = stbi_write_jpg(path, w, h, 3, u8, 95);
    } else {
        FILE *fp = fopen(path, "wb");
        if (fp) {
            fprintf(fp, "P6\n%d %d\n255\n", w, h);
            fwrite(u8, 1, (size_t)h * w * 3, fp);
            fclose(fp);
            ok = 1;
        }
    }
    free(u8);
    if (ok) fprintf(stderr, "Saved %s (%dx%d)\n", path, w, h);
    else    fprintf(stderr, "Failed to write %s\n", path);
}

static void save_ppm(const char *path, const float *rgb, int h, int w) {
    save_image(path, rgb, h, w);
}

/* Load a raw F32 binary of known element count. */
static float *load_f32_bin(const char *path, size_t n_elems) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    size_t want = n_elems * sizeof(float);
    if ((size_t)sz != want) {
        fprintf(stderr, "%s: size %ld != expected %zu\n", path, sz, want);
        fclose(fp); return NULL;
    }
    float *buf = (float *)malloc(want);
    if (fread(buf, 1, want, fp) != want) { free(buf); fclose(fp); return NULL; }
    fclose(fp);
    return buf;
}

static int write_f32_bin(const char *path, const float *data, size_t n_elems) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot write %s\n", path);
        return -1;
    }
    size_t n = fwrite(data, sizeof(float), n_elems, fp);
    fclose(fp);
    if (n != n_elems) {
        fprintf(stderr, "Short write to %s (%zu/%zu)\n", path, n, n_elems);
        return -1;
    }
    return 0;
}

static void compare_f32_arrays(const char *label, const float *ref, const float *ours, size_t n) {
    double dot = 0.0, nr = 0.0, no = 0.0, mse = 0.0, mae = 0.0;
    float maxd = 0.0f, peak = 0.0f;
    size_t max_i = 0;
    for (size_t i = 0; i < n; i++) {
        double r = ref[i], o = ours[i];
        double d = fabs(r - o);
        if ((float)d > maxd) { maxd = (float)d; max_i = i; }
        if (fabs(ref[i]) > peak) peak = fabs(ref[i]);
        dot += r * o;
        nr += r * r;
        no += o * o;
        mse += d * d;
        mae += d;
    }
    mse /= (double)n;
    mae /= (double)n;
    double cos = dot / (sqrt(nr) * sqrt(no) + 1e-30);
    double psnr = 20.0 * log10((double)(peak > 1e-20f ? peak : 1e-20f))
                - 10.0 * log10(mse > 1e-30 ? mse : 1e-30);
    fprintf(stderr,
            "%s: n=%zu cos=%.6f mae=%.6f rmse=%.6f max=%.6f@%zu psnr_peak=%.2f dB\n",
            label, n, cos, mae, sqrt(mse), maxd, max_i, psnr);
}

/* Load text hidden states from raw F32 file (any token count, fixed dim). */
static float *load_txt_bin(const char *path, int *n_tok, int txt_dim) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz % (txt_dim * (int)sizeof(float)) != 0) {
        fprintf(stderr, "txt-bin %s size %ld not multiple of %d*%zu\n",
                path, sz, txt_dim, sizeof(float));
        fclose(fp); return NULL;
    }
    int n = (int)(sz / (txt_dim * (int)sizeof(float)));
    float *buf = (float *)malloc((size_t)n * txt_dim * sizeof(float));
    if (fread(buf, 1, sz, fp) != (size_t)sz) { free(buf); fclose(fp); return NULL; }
    fclose(fp);
    *n_tok = n;
    fprintf(stderr, "Loaded %s: %d tokens × %d dim\n", path, n, txt_dim);
    return buf;
}

int main(int argc, char **argv) {
    const char *dit_path = NULL;
    const char *vae_path = NULL;
    const char *enc_path = NULL;
    const char *prompt = "a red apple on a white table";
    const char *out_path = "hip_qimg_out.ppm";
    const char *mode = NULL;
    const char *init_bin_path = NULL;
    const char *txt_bin_path = NULL;
    const char *neg_txt_bin_path = NULL;
    const char *sigmas_bin_path = NULL;
    const char *dump_final_path = NULL;
    const char *dump_steps_prefix = NULL;
    const char *ref_final_path = NULL;
    const char *negative = " ";
    int skip_unstd = 0;   /* skip latent un-standardization (e.g. when init came pre-normalized) */
    int path_stats = 0;
    int mem_stats = 0;
    int quant_stats = 0;
    int quant_stats_max = 80;
    const char *fp8_fp8_allow = NULL;
    const char *fp8_fp8_deny = NULL;
    int fp8_fp8_block_min = -1;
    int fp8_fp8_block_max = -1;
    float fp8_quality_target_db = 0.0f;
    float fp8_act_scale_div = 0.0f;
    const char *fp8_act_scale_mode = NULL;
    int device_id = 0;
    int verbose = 1;
    int out_h = 256, out_w = 256, n_steps = 20;
    uint64_t seed = 42;
    float cfg_scale = 1.0f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--test-init")) mode = "init";
        else if (!strcmp(argv[i], "--test-dit")) mode = "dit";
        else if (!strcmp(argv[i], "--test-vae")) mode = "vae";
        else if (!strcmp(argv[i], "--generate")) mode = "gen";
        else if (!strcmp(argv[i], "--test-enc")) mode = "enc";
        else if (!strcmp(argv[i], "--dit") && i+1 < argc) dit_path = argv[++i];
        else if (!strcmp(argv[i], "--vae") && i+1 < argc) vae_path = argv[++i];
        else if (!strcmp(argv[i], "--enc") && i+1 < argc) enc_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--negative") && i+1 < argc) negative = argv[++i];
        else if (!strcmp(argv[i], "--cfg") && i+1 < argc) cfg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-scale") && i+1 < argc) cfg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--height") && i+1 < argc) out_h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--width") && i+1 < argc) out_w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "-d") && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else if (!strcmp(argv[i], "--init-bin") && i+1 < argc) init_bin_path = argv[++i];
        else if (!strcmp(argv[i], "--txt-bin") && i+1 < argc) txt_bin_path = argv[++i];
        else if (!strcmp(argv[i], "--neg-txt-bin") && i+1 < argc) neg_txt_bin_path = argv[++i];
        else if (!strcmp(argv[i], "--sigmas-bin") && i+1 < argc) sigmas_bin_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-final") && i+1 < argc) dump_final_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-steps-prefix") && i+1 < argc) dump_steps_prefix = argv[++i];
        else if (!strcmp(argv[i], "--ref-final") && i+1 < argc) ref_final_path = argv[++i];
        else if (!strcmp(argv[i], "--verify-final") && i+1 < argc) { mode = "gen"; ref_final_path = argv[++i]; }
        else if (!strcmp(argv[i], "--skip-unstd")) skip_unstd = 1;
        else if (!strcmp(argv[i], "--path-stats")) path_stats = 1;
        else if (!strcmp(argv[i], "--mem-stats")) mem_stats = 1;
        else if (!strcmp(argv[i], "--fp8-quant-stats")) quant_stats = 1;
        else if (!strcmp(argv[i], "--fp8-quant-stats-max") && i+1 < argc) quant_stats_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fp8-fp8-allow") && i+1 < argc) fp8_fp8_allow = argv[++i];
        else if (!strcmp(argv[i], "--fp8-fp8-deny") && i+1 < argc) fp8_fp8_deny = argv[++i];
        else if (!strcmp(argv[i], "--fp8-fp8-block-min") && i+1 < argc) fp8_fp8_block_min = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fp8-fp8-block-max") && i+1 < argc) fp8_fp8_block_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fp8-quality-target-db") && i+1 < argc) fp8_quality_target_db = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--fp8-act-scale-div") && i+1 < argc) fp8_act_scale_div = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--fp8-act-scale-mode") && i+1 < argc) fp8_act_scale_mode = argv[++i];
        else {
            fprintf(stderr, "Usage: %s --test-init|--test-dit|--test-vae|--generate\n"
                    "  --dit <st>  --vae <st>  --enc <gguf>  --prompt <text>\n"
                    "  --height <h>  --width <w>  --steps <n>  --seed <s>\n"
                    "  [--cfg <scale> --negative <text>] [--neg-txt-bin <bin>]\n"
                    "  [--dump-steps-prefix <pfx>] [--ref-final <bin>] [--path-stats] [--mem-stats]\n"
                    "  [--fp8-quant-stats] [--fp8-fp8-allow <labels>] [--fp8-fp8-deny <labels>]\n"
                    "  [--fp8-fp8-block-min <i>] [--fp8-fp8-block-max <i>]\n"
                    "  [--fp8-quality-target-db <db>]\n"
                    "  [--fp8-act-scale-div <x>] [--fp8-act-scale-mode perrow|comfy|clamp]\n"
                    "  [-o out.ppm] [-d dev] [-v]\n", argv[0]);
            return 1;
        }
    }

    if (path_stats) setenv("QIMG_PATH_STATS", "1", 1);
    if (mem_stats) setenv("QIMG_MEM_STATS", "1", 1);
    if (quant_stats) setenv("QIMG_FP8_QUANT_STATS", "1", 1);
    if (quant_stats && quant_stats_max > 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", quant_stats_max);
        setenv("QIMG_FP8_QUANT_STATS_MAX", buf, 1);
    }
    if (fp8_fp8_allow) setenv("QIMG_FP8_FP8_ALLOW", fp8_fp8_allow, 1);
    if (fp8_fp8_deny) setenv("QIMG_FP8_FP8_DENY", fp8_fp8_deny, 1);
    if (fp8_fp8_block_min >= 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", fp8_fp8_block_min);
        setenv("QIMG_FP8_FP8_BLOCK_MIN", buf, 1);
    }
    if (fp8_fp8_block_max >= 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", fp8_fp8_block_max);
        setenv("QIMG_FP8_FP8_BLOCK_MAX", buf, 1);
    }
    if (fp8_quality_target_db > 0.0f) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.8g", fp8_quality_target_db);
        setenv("QIMG_FP8_QUALITY_TARGET_DB", buf, 1);
    }
    if (fp8_act_scale_div > 0.0f) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.8g", fp8_act_scale_div);
        setenv("QIMG_FP8_ACT_SCALE_DIV", buf, 1);
    }
    if (fp8_act_scale_mode) setenv("QIMG_FP8_ACT_SCALE_MODE", fp8_act_scale_mode, 1);

    if (!mode) {
        fprintf(stderr, "Error: specify a mode (--test-init, --test-dit, --test-vae, --generate)\n");
        return 1;
    }

    int lat_h = out_h / 8, lat_w = out_w / 8;

    fprintf(stderr, "=== HIP Qwen-Image: %s ===\n", mode);

    /* For generate mode with GPU text encoder:
     * 1. Init qimg runner FIRST (compiles qimg kernels, gets module loaded)
     * 2. Load LLM, encode text, offload LLM weights
     * 3. Load DiT weights into already-compiled qimg module
     * This avoids the ROCm bug where hipModuleLoadData fails after
     * another module was loaded then unloaded in the same process. */
    int n_txt_precomputed = 0;
    float *txt_precomputed = NULL;
    int n_txt_neg_precomputed = 0;
    float *txt_neg_precomputed = NULL;
    int use_cfg = fabsf(cfg_scale - 1.0f) > 1e-6f;

    /* Init HIP + compile qimg kernels FIRST */
    hip_qimg_runner *r = hip_qimg_init(device_id, verbose);

    /* Now run GPU text encoder (LLM module loads as second module — this works) */
    if (r && !strcmp(mode, "gen") && enc_path &&
        (!txt_bin_path || (use_cfg && !neg_txt_bin_path))) {
        fprintf(stderr, "\n[1/3] Text conditioning (GPU)...\n");
        clock_t enc_t0 = clock();
        qimg_text_enc *enc = qimg_text_enc_load_gpu(enc_path, NULL, device_id);
        if (enc) {
            if (!txt_bin_path) {
                fprintf(stderr, "  Encoding: \"%s\"\n", prompt);
                txt_precomputed = qimg_text_enc_encode(enc, prompt, &n_txt_precomputed);
                if (txt_precomputed)
                    fprintf(stderr, "  Text hidden: [%d, 3584] (%.1fs)\n",
                            n_txt_precomputed, (double)(clock()-enc_t0)/CLOCKS_PER_SEC);
            }
            if (use_cfg && !neg_txt_bin_path) {
                fprintf(stderr, "  Encoding negative: \"%s\"\n", negative);
                txt_neg_precomputed = qimg_text_enc_encode(enc, negative, &n_txt_neg_precomputed);
                if (txt_neg_precomputed)
                    fprintf(stderr, "  Negative hidden: [%d, 3584] (%.1fs total)\n",
                            n_txt_neg_precomputed, (double)(clock()-enc_t0)/CLOCKS_PER_SEC);
            }
            qimg_text_enc_free(enc);
        }
        if (!txt_bin_path && !txt_precomputed)
            fprintf(stderr, "  GPU text encoder failed, will use CPU fallback\n");
    }
    if (!r) { fprintf(stderr, "Init failed\n"); return 1; }

    if (!strcmp(mode, "init")) { hip_qimg_free(r); return 0; }

    clock_t t0;

    /* Load DiT (deferred for gen/enc modes — GPU text encoder runs first) */
    if (strcmp(mode, "gen") != 0 && strcmp(mode, "enc") != 0) {
        if (!dit_path) {
            fprintf(stderr, "Error: --dit <path> is required\n");
            hip_qimg_free(r); return 1;
        }
        t0 = clock();
        if (hip_qimg_load_dit(r, dit_path) != 0) {
            hip_qimg_free(r); return 1;
        }
        fprintf(stderr, "DiT loaded in %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
    }

    /* ---- test-vae ---- */
    if (!strcmp(mode, "vae")) {
        if (!vae_path) {
            fprintf(stderr, "Error: --vae <path> is required for VAE test\n");
            hip_qimg_free(r); return 1;
        }
        hip_qimg_load_vae(r, vae_path);

        /* Generate random latent for testing */
        rng_state = seed;
        float *latent = (float *)malloc((size_t)16 * lat_h * lat_w * sizeof(float));
        for (int i = 0; i < 16 * lat_h * lat_w; i++) latent[i] = randn() * 0.5f;

        float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
        t0 = clock();
        hip_qimg_vae_decode(r, latent, lat_h, lat_w, rgb);
        fprintf(stderr, "VAE decode: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

        save_ppm("hip_qimg_vae_test.ppm", rgb, out_h, out_w);
        free(latent); free(rgb);
        hip_qimg_free(r); return 0;
    }

    /* ---- test-enc: compare GPU vs CPU text encoder ---- */
    if (!strcmp(mode, "enc")) {
        if (!enc_path) {
            fprintf(stderr, "Error: --enc required\n");
            hip_qimg_free(r); return 1;
        }
        int dim = 3584;

        /* GPU (runs first while full VRAM is available — no DiT loaded yet) */
        fprintf(stderr, "\n=== GPU Text Encoder ===\n");
        int n_gpu = 0;
        float *gpu_h = NULL;
        {
            qimg_text_enc *enc = qimg_text_enc_load_gpu(enc_path, NULL, device_id);
            if (enc) {
                hip_llm_set_debug((hip_llm_runner *)enc->model, 1);
                gpu_h = qimg_text_enc_encode(enc, prompt, &n_gpu);
                qimg_text_enc_free(enc);
            }
        }

        /* CPU */
        fprintf(stderr, "\n=== CPU Text Encoder ===\n");
        int n_cpu = 0;
        float *cpu_h = NULL;
        {
            qimg_text_enc *enc = qimg_text_enc_load(enc_path);
            if (enc) {
                cpu_h = qimg_text_enc_encode(enc, prompt, &n_cpu);
                qimg_text_enc_free(enc);
            }
        }

        fprintf(stderr, "\n=== Comparison ===\n");
        fprintf(stderr, "GPU: %d tokens, CPU: %d tokens\n", n_gpu, n_cpu);
        if (gpu_h && cpu_h) {
            int n = n_gpu < n_cpu ? n_gpu : n_cpu;
            for (int t = 0; t < n && t < 5; t++) {
                float *g = gpu_h + t * dim, *c = cpu_h + t * dim;
                float max_d = 0, gn = 0, cn = 0;
                for (int d = 0; d < dim; d++) {
                    float diff = fabsf(g[d] - c[d]);
                    if (diff > max_d) max_d = diff;
                    gn += g[d]*g[d]; cn += c[d]*c[d];
                }
                fprintf(stderr, "  tok %d: gpu_norm=%.2f cpu_norm=%.2f max_diff=%.4f "
                        "gpu[0..3]=[%.4f,%.4f,%.4f,%.4f] cpu[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
                        t, sqrtf(gn), sqrtf(cn), max_d,
                        g[0],g[1],g[2],g[3], c[0],c[1],c[2],c[3]);
            }
        }
        free(gpu_h); free(cpu_h);
        hip_qimg_free(r); return 0;
    }

    /* ---- test-dit: single step ---- */
    if (!strcmp(mode, "dit")) {
        int ps = 2, hp = lat_h/ps, wp = lat_w/ps, n_img = hp*wp, n_txt = 7;
        int in_ch = 64, txt_dim = 3584;
        rng_state = seed;
        float *img = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *txt = (float *)malloc((size_t)n_txt * txt_dim * sizeof(float));
        float *out = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        for (int i = 0; i < n_img*in_ch; i++) img[i] = randn() * 0.1f;
        for (int i = 0; i < n_txt*txt_dim; i++) txt[i] = randn() * 0.1f;

        t0 = clock();
        hip_qimg_dit_step(r, img, n_img, txt, n_txt, 500.0f, out);
        fprintf(stderr, "DiT step: %.2fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

        float mn=out[0], mx=out[0], sm=0;
        int nn = n_img*in_ch, nc=0;
        for (int i = 0; i < nn; i++) {
            if (out[i] != out[i]) nc++;
            else { if (out[i]<mn) mn=out[i]; if (out[i]>mx) mx=out[i]; sm+=out[i]; }
        }
        fprintf(stderr, "Output: min=%.4f max=%.4f mean=%.4f nan=%d/%d\n",
                mn, mx, sm/(nn-nc), nc, nn);
        free(img); free(txt); free(out);
        hip_qimg_free(r); return 0;
    }

    /* ---- generate: full pipeline ---- */
    if (!strcmp(mode, "gen")) {
        /* Load DiT now (after GPU text encoder freed VRAM) */
        if (dit_path) {
            clock_t t0_dit = clock();
            if (hip_qimg_load_dit(r, dit_path) != 0) {
                hip_qimg_free(r); return 1;
            }
            fprintf(stderr, "DiT loaded in %.1fs\n", (double)(clock()-t0_dit)/CLOCKS_PER_SEC);
        }

        int ps = 2;
        int hp = lat_h / ps, wp = lat_w / ps;
        int n_img = hp * wp;
        int in_ch = 64, lat_ch = 16;
        int txt_dim = 3584;

        /* 1. Text encoder — pinned --txt-bin > GPU encoder > CPU fallback > random */
        int n_txt = 0;
        float *txt_tokens = NULL;
        int n_txt_neg = 0;
        float *txt_tokens_neg = NULL;

        if (txt_bin_path) {
            txt_tokens = load_txt_bin(txt_bin_path, &n_txt, txt_dim);
            if (!txt_tokens) { hip_qimg_free(r); return 1; }
        } else if (txt_precomputed) {
            txt_tokens = txt_precomputed;
            n_txt = n_txt_precomputed;
            txt_precomputed = NULL;
        } else if (enc_path) {
            fprintf(stderr, "\n[1/3] Text conditioning (CPU fallback)...\n");
            fprintf(stderr, "  Loading text encoder: %s\n", enc_path);
            clock_t enc_t0 = clock();
            qimg_text_enc *enc = qimg_text_enc_load(enc_path);
            if (enc) {
                fprintf(stderr, "  Encoding: \"%s\"\n", prompt);
                txt_tokens = qimg_text_enc_encode(enc, prompt, &n_txt);
                if (txt_tokens)
                    fprintf(stderr, "  Text hidden: [%d, %d] (%.1fs)\n",
                            n_txt, txt_dim,
                            (double)(clock()-enc_t0)/CLOCKS_PER_SEC);
                if (use_cfg && !neg_txt_bin_path) {
                    fprintf(stderr, "  Encoding negative: \"%s\"\n", negative);
                    txt_tokens_neg = qimg_text_enc_encode(enc, negative, &n_txt_neg);
                    if (txt_tokens_neg)
                        fprintf(stderr, "  Negative hidden: [%d, %d] (%.1fs total)\n",
                                n_txt_neg, txt_dim,
                                (double)(clock()-enc_t0)/CLOCKS_PER_SEC);
                }
                qimg_text_enc_free(enc);
            }
        }
        if (!txt_tokens) {
            fprintf(stderr, "  No text encoder -- using random conditioning\n");
            n_txt = 7;
            rng_state = seed + 1;
            txt_tokens = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
            for (int i = 0; i < n_txt * txt_dim; i++) txt_tokens[i] = randn() * 0.1f;
        }
        if (use_cfg) {
            if (neg_txt_bin_path) {
                txt_tokens_neg = load_txt_bin(neg_txt_bin_path, &n_txt_neg, txt_dim);
                if (!txt_tokens_neg) { hip_qimg_free(r); return 1; }
            } else if (txt_neg_precomputed) {
                txt_tokens_neg = txt_neg_precomputed;
                n_txt_neg = n_txt_neg_precomputed;
                txt_neg_precomputed = NULL;
            } else if (enc_path) {
                fprintf(stderr, "\n[1/3] Negative text conditioning (CPU fallback)...\n");
                qimg_text_enc *enc = qimg_text_enc_load(enc_path);
                if (enc) {
                    txt_tokens_neg = qimg_text_enc_encode(enc, negative, &n_txt_neg);
                    qimg_text_enc_free(enc);
                }
            }
            if (!txt_tokens_neg) {
                fprintf(stderr, "  No negative text encoder output -- using blank random-free conditioning\n");
                n_txt_neg = 1;
                txt_tokens_neg = (float *)calloc((size_t)n_txt_neg * txt_dim, sizeof(float));
            }
            fprintf(stderr, "  CFG enabled: scale=%.3f cond_tokens=%d uncond_tokens=%d\n",
                    cfg_scale, n_txt, n_txt_neg);
        }

        /* 2. Initial latent: pinned --init-bin (already packed) > random+patchify */
        fprintf(stderr, "\n[2/3] Denoising (%dx%d, %d steps)...\n", out_w, out_h, n_steps);
        float *latent = (float *)malloc((size_t)lat_ch * lat_h * lat_w * sizeof(float));
        float *img_tokens = (float *)malloc((size_t)n_img * in_ch * sizeof(float));

        if (init_bin_path) {
            float *loaded = load_f32_bin(init_bin_path, (size_t)n_img * in_ch);
            if (!loaded) { hip_qimg_free(r); return 1; }
            memcpy(img_tokens, loaded, (size_t)n_img * in_ch * sizeof(float));
            free(loaded);
            fprintf(stderr, "  loaded pinned init latent from %s\n", init_bin_path);
        } else {
            rng_state = seed;
            for (int i = 0; i < lat_ch * lat_h * lat_w; i++) latent[i] = randn();
            for (int tok = 0; tok < n_img; tok++) {
                int py = tok / wp, px = tok % wp;
                int idx = 0;
                for (int c = 0; c < lat_ch; c++)
                    for (int dy = 0; dy < ps; dy++)
                        for (int dx = 0; dx < ps; dx++)
                            img_tokens[tok * in_ch + idx++] =
                                latent[c * lat_h * lat_w + (py*ps+dy) * lat_w + (px*ps+dx)];
            }
        }

        /* FlowMatch schedule: pinned --sigmas-bin > scheduler default */
        qimg_scheduler sched;
        int seq_len = n_img + n_txt;
        qimg_sched_init(&sched);
        if (sigmas_bin_path) {
            float *loaded = load_f32_bin(sigmas_bin_path, (size_t)(n_steps + 1));
            if (!loaded) { hip_qimg_free(r); return 1; }
            for (int i = 0; i <= n_steps; i++) sched.sigmas[i] = loaded[i];
            for (int i = 0; i < n_steps; i++) sched.dt[i] = sched.sigmas[i+1] - sched.sigmas[i];
            sched.n_steps = n_steps;
            free(loaded);
            fprintf(stderr, "  loaded pinned sigmas from %s\n", sigmas_bin_path);
        } else {
            qimg_sched_set_timesteps(&sched, n_steps, seq_len);
        }

        float *vel = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *vel_uncond = use_cfg ? (float *)malloc((size_t)n_img * in_ch * sizeof(float)) : NULL;

        fprintf(stderr, "Generating %dx%d image with %d steps...\n", out_w, out_h, n_steps);
        t0 = clock();

        for (int step = 0; step < n_steps; step++) {
            float t = sched.sigmas[step] * 1000.0f;
            float dt = sched.dt[step];

            if (use_cfg) {
                hip_qimg_dit_step(r, img_tokens, n_img, txt_tokens_neg, n_txt_neg, t, vel_uncond);
                hip_qimg_dit_step(r, img_tokens, n_img, txt_tokens, n_txt, t, vel);
                for (int i = 0; i < n_img * in_ch; i++)
                    vel[i] = vel_uncond[i] + cfg_scale * (vel[i] - vel_uncond[i]);
            } else {
                hip_qimg_dit_step(r, img_tokens, n_img, txt_tokens, n_txt, t, vel);
            }

            /* Euler step: img_tokens += dt * vel */
            for (int i = 0; i < n_img * in_ch; i++)
                img_tokens[i] += dt * vel[i];

            if (dump_steps_prefix) {
                char step_path[1024];
                snprintf(step_path, sizeof(step_path), "%s_step%03d.bin",
                         dump_steps_prefix, step);
                if (write_f32_bin(step_path, img_tokens, (size_t)n_img * in_ch) == 0)
                    fprintf(stderr, "  dumped %s\n", step_path);
            }

            fprintf(stderr, "  step %d/%d: t=%.1f dt=%.4f\n", step+1, n_steps, t, dt);
        }

        fprintf(stderr, "Denoising done in %.1fs (%.2fs/step)\n",
                (double)(clock()-t0)/CLOCKS_PER_SEC,
                (double)(clock()-t0)/CLOCKS_PER_SEC/n_steps);

        if (dump_final_path) {
            if (write_f32_bin(dump_final_path, img_tokens, (size_t)n_img * in_ch) == 0)
                fprintf(stderr, "  dumped final packed latent to %s\n", dump_final_path);
        }
        if (ref_final_path) {
            float *ref_final = load_f32_bin(ref_final_path, (size_t)n_img * in_ch);
            if (ref_final) {
                compare_f32_arrays("Final packed latent", ref_final, img_tokens,
                                   (size_t)n_img * in_ch);
                free(ref_final);
            }
        }

        /* Unpatchify → latent */
        for (int tok = 0; tok < n_img; tok++) {
            int py = tok / wp, px = tok % wp;
            int idx = 0;
            for (int c = 0; c < lat_ch; c++)
                for (int dy = 0; dy < ps; dy++)
                    for (int dx = 0; dx < ps; dx++)
                        latent[c * lat_h * lat_w + (py*ps+dy) * lat_w + (px*ps+dx)] =
                            img_tokens[tok * in_ch + idx++];
        }

        /* Un-standardize latent: DiT normalized space → VAE natural space.
         * Values from diffusers autoencoder_kl_qwenimage.py config.
         * Skip when --skip-unstd is set (e.g. when --init-bin pre-normalized). */
        if (!skip_unstd) {
            static const float lm[16] = {-0.7571f,-0.7089f,-0.9113f,0.1075f,-0.1745f,0.9653f,-0.1517f,1.5508f,
                                          0.4134f,-0.0715f,0.5517f,-0.3632f,-0.1922f,-0.9497f,0.2503f,-0.2921f};
            static const float ls[16] = {2.8184f,1.4541f,2.3275f,2.6558f,1.2196f,1.7708f,2.6052f,2.0743f,
                                          3.2687f,2.1526f,2.8652f,1.5579f,1.6382f,1.1253f,2.8251f,1.9160f};
            for (int c = 0; c < lat_ch; c++)
                for (int i = 0; i < lat_h * lat_w; i++)
                    latent[c * lat_h * lat_w + i] = latent[c * lat_h * lat_w + i] * ls[c] + lm[c];
        }

        /* 3. VAE decode */
        fprintf(stderr, "\n[3/3] VAE decode...\n");
        if (vae_path) {
            hip_qimg_load_vae(r, vae_path);
            float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
            t0 = clock();
            hip_qimg_vae_decode(r, latent, lat_h, lat_w, rgb);
            fprintf(stderr, "VAE decode: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
            save_ppm(out_path, rgb, out_h, out_w);
            free(rgb);
        } else {
            fprintf(stderr, "No VAE specified, skipping decode\n");
        }

        free(latent); free(img_tokens); free(txt_tokens); free(txt_tokens_neg);
        free(vel); free(vel_uncond);
        hip_qimg_free(r); return 0;
    }

    hip_qimg_free(r);
    return 0;
}
