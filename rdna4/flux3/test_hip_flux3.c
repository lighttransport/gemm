/* SPDX-License-Identifier: MIT */
#include "../flux2/hip_flux2_runner.h"
#include "../rocew.h"
#include "hip_flux3_runner.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
static int integer(const char *s, int min, int max) {
    char *e;
    long v = strtol(s, &e, 10);
    if (!*s || *e || v < min || v > max) {
        fprintf(stderr, "invalid integer: %s\n", s);
        exit(2);
    }
    return (int)v;
}
static float time_value(const char *s) {
    char *end;
    float v = strtof(s, &end);
    if (!*s || *end || !isfinite(v) || v < 0 || v > 1) {
        fprintf(stderr, "timestep must be in [0,1]: %s\n", s);
        exit(2);
    }
    return v;
}
static float *read_raw(const char *path, size_t n) {
    struct stat st;
    if (!path || stat(path, &st) || (uint64_t)st.st_size != (uint64_t)n * 4) {
        fprintf(stderr, "expected %zu F32 values: %s\n", n, path ? path : "(missing)");
        return NULL;
    }
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    float *p = malloc(n * 4);
    if (!p) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(p, 4, n, f);
    fclose(f);
    if (got != n) {
        free(p);
        return NULL;
    }
    for (size_t i = 0; i < n; i++)
        if (!isfinite(p[i])) {
            fprintf(stderr, "nonfinite input: %s\n", path);
            free(p);
            return NULL;
        }
    return p;
}
static int write_raw(const char *path, const float *p, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    int rc = fwrite(p, 4, n, f) == n ? 0 : -1;
    if (fclose(f))
        rc = -1;
    return rc;
}
static float *ids(int h, int w) {
    float *p = calloc((size_t)h * w * 4, 4);
    if (p)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                p[(y * w + x) * 4 + 1] = y * .5f;
                p[(y * w + x) * 4 + 2] = x * .5f;
            }
    return p;
}
static uint64_t rng_state;
static float uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)(((rng_state * 2685821657736338717ULL) >> 40) + .5) / 16777216.f;
}
static float bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u;
    memcpy(&x, &u, 4);
    return x;
}
static float gaussian(void) {
    return sqrtf(-2 * logf(uniform())) * cosf(6.283185307179586f * uniform());
}
static float schedule(int i, int steps) {
    float t = 1.f - (float)i / steps;
    return 6.93f * t / (1.f + 5.93f * t);
}
static void usage(const char *app) {
    fprintf(stderr,
            "Usage: %s --text features.f32 --ntxt N [options]\n"
            "  --model-dir DIR  default /mnt/nvme02/model/flux3-klein\n"
            "  --dit FILE       BF16 or per-row FP8 safetensors\n"
            "  --vae FILE       diffusers VAE checkpoint\n"
            "  --width W --height H  multiples of 16 (default 256)\n"
            "  --steps N        default 16, distilled flow-map schedule alpha=6.93\n"
            "  --seed N --device N --reserve-mib N (default 1024)\n"
            "  --latent FILE    initial packed F32 tokens, otherwise seeded native noise\n"
            "  --condition FILE --condition-ids FILE --ncond N  prepared reference tokens\n"
            "  --step T TARGET  single velocity evaluation, requires --latent\n"
            "  --dump FILE      velocity (--step) or final packed latent\n"
            "  --vae-wmma 0|1   F32 (default) or BF16 VAE convolutions\n"
            "  --no-vae         keep packed latent only; --output image.ppm\n"
            "  --vae-only --latent FILE --dump-rgb FILE   decode packed latent only\n"
            "  --trace-dir DIR  diagnostic activation dumps to an existing directory\n"
            "  --selftest       test native BF16 and row-scaled FP8 GEMMs\n",
            app);
}
static int decode_vae(const char *vae, const float *image, int width, int height, int device,
                      int verbose, int vae_wmma, int reserve, const char *output,
                      const char *dump_rgb) {
    int ph = height / 16, pw = width / 16;
    size_t count = (size_t)ph * pw * 128;
    const char *skip = getenv("FLUX2_SKIP_VAE_BN");
    if (skip && strcmp(skip, "0") && strcmp(skip, "false")) {
        fprintf(stderr,
                "flux3: unset FLUX2_SKIP_VAE_BN; packed-channel normalization is required\n");
        return -1;
    }
    // Unpatch WITHOUT BN; the reused decoder applies packed-channel BN once.
    float *latent = malloc(count * 4), *rgb = malloc((size_t)width * height * 3 * 4);
    if (!latent || !rgb) {
        free(latent);
        free(rgb);
        return -1;
    }
    int lh = height / 8, lw = width / 8;
    for (int y = 0; y < ph; y++)
        for (int x = 0; x < pw; x++)
            for (int c = 0; c < 32; c++)
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                        latent[((size_t)c * lh + 2 * y + dy) * lw + 2 * x + dx] =
                            image[((size_t)y * pw + x) * 128 + c * 4 + dy * 2 + dx];
    hip_flux2_runner *v = hip_flux2_init(device, verbose);
    size_t avail = 0, total = 0;
    size_t needed = ((size_t)1024 << 20) + (size_t)width * height * 128 * 4 * 12;
    if (!v || hipMemGetInfo(&avail, &total) != hipSuccess ||
        avail < needed + (size_t)reserve * 1048576) {
        fprintf(stderr,
                "flux3: VAE needs conservative %.0f MiB plus %d MiB reserve; %.0f MiB free\n",
                needed / 1048576., reserve, avail / 1048576.);
        if (v)
            hip_flux2_free(v);
        free(latent);
        free(rgb);
        return -1;
    }
    int ok = v && hip_flux2_set_vae_wmma(v, vae_wmma) == 0 && hip_flux2_load_vae(v, vae) == 0 &&
             hip_flux2_vae_decode(v, latent, lh, lw, rgb) == 0;
    if (v)
        hip_flux2_free(v);
    free(latent);
    if (!ok) {
        free(rgb);
        return -1;
    }
    if (dump_rgb && write_raw(dump_rgb, rgb, (size_t)width * height * 3)) {
        free(rgb);
        return -1;
    }
    FILE *f = fopen(output, "wb");
    if (!f) {
        free(rgb);
        return -1;
    }
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    int valid = 1;
    for (int p = 0; p < width * height; p++)
        for (int c = 0; c < 3; c++) {
            float z = rgb[(size_t)c * width * height + p];
            if (!isfinite(z))
                valid = 0;
            unsigned char b = (unsigned char)lrintf(fminf(255, fmaxf(0, (z + 1) * 127.5f)));
            if (fwrite(&b, 1, 1, f) != 1)
                valid = 0;
        }
    if (fclose(f))
        valid = 0;
    free(rgb);
    if (!valid)
        return -1;
    fprintf(stderr, "flux3: wrote %s\n", output);
    return 0;
}
int main(int argc, char **argv) {
    const char *dir = "/mnt/nvme02/model/flux3-klein", *dit = NULL, *vae = NULL, *textfile = NULL,
               *latfile = NULL, *condfile = NULL, *condidsfile = NULL, *dump = NULL,
               *dump_rgb = NULL, *trace_dir = NULL, *output = "flux3.ppm";
    int width = 256, height = 256, steps = 16, seed = 42, device = 0, reserve = 1024, nt = 0,
        nc = 0, selftest = 0, one = 0, no_vae = 0, vae_only = 0, vae_wmma = 0, verbose = 1;
    float t = 1, target = 0;
#define ARG(name) !strcmp(argv[i], name)
#define VALUE() ((i + 1 < argc) ? argv[++i] : (usage(argv[0]), exit(2), (char *)NULL))
    for (int i = 1; i < argc; i++) {
        if (ARG("--trace-dir"))
            trace_dir = VALUE();
        else if (ARG("--vae-only"))
            vae_only = 1;
        else if (ARG("--vae-wmma"))
            vae_wmma = integer(VALUE(), 0, 1);
        else if (ARG("--dump-rgb"))
            dump_rgb = VALUE();
        else if (ARG("--model-dir"))
            dir = VALUE();
        else if (ARG("--dit"))
            dit = VALUE();
        else if (ARG("--vae"))
            vae = VALUE();
        else if (ARG("--text"))
            textfile = VALUE();
        else if (ARG("--ntxt"))
            nt = integer(VALUE(), 1, 4096);
        else if (ARG("--width"))
            width = integer(VALUE(), 16, 2048);
        else if (ARG("--height"))
            height = integer(VALUE(), 16, 2048);
        else if (ARG("--steps"))
            steps = integer(VALUE(), 1, 1000);
        else if (ARG("--seed"))
            seed = integer(VALUE(), 0, INT_MAX);
        else if (ARG("--device"))
            device = integer(VALUE(), 0, 127);
        else if (ARG("--reserve-mib"))
            reserve = integer(VALUE(), 0, 65536);
        else if (ARG("--latent"))
            latfile = VALUE();
        else if (ARG("--condition"))
            condfile = VALUE();
        else if (ARG("--condition-ids"))
            condidsfile = VALUE();
        else if (ARG("--ncond"))
            nc = integer(VALUE(), 1, 16384);
        else if (ARG("--dump"))
            dump = VALUE();
        else if (ARG("--output"))
            output = VALUE();
        else if (ARG("--step")) {
            one = 1;
            t = time_value(VALUE());
            target = time_value(VALUE());
        } else if (ARG("--no-vae"))
            no_vae = 1;
        else if (ARG("--selftest"))
            selftest = 1;
        else if (ARG("--verbose"))
            verbose = 2;
        else if (ARG("--help")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!selftest && !vae_only &&
        (!textfile || !nt || width % 16 || height % 16 || (one && !latfile) ||
         (!!nc != !!condfile) || (!!nc != !!condidsfile))) {
        usage(argv[0]);
        return 2;
    }
    if (!getenv("TMPDIR")) {
        mkdir("tmp", 0775);
        setenv("TMPDIR", "tmp", 0);
    }
    if (vae_only) {
        if (!latfile || width % 16 || height % 16) {
            usage(argv[0]);
            return 2;
        }
        char path[4096];
        snprintf(path, sizeof(path), "%s/vae/diffusion_pytorch_model.safetensors", dir);
        float *packed = read_raw(latfile, (size_t)(width / 16) * (height / 16) * 128);
        if (!packed)
            return 1;
        int result = decode_vae(vae ? vae : path, packed, width, height, device, verbose, vae_wmma,
                                reserve, output, dump_rgb);
        free(packed);
        return result ? 1 : 0;
    }
    hip_flux3_runner *r = hip_flux3_init(device, reserve, verbose);
    if (!r)
        return 1;
    if (selftest) {
        int rc = hip_flux3_selftest(r);
        hip_flux3_free(r);
        return rc ? 1 : 0;
    }
    char ditpath[4096], vaepath[4096];
    snprintf(ditpath, sizeof(ditpath), "%s/variants/fp8r/model.safetensors", dir);
    snprintf(vaepath, sizeof(vaepath), "%s/vae/diffusion_pytorch_model.safetensors", dir);
    if (!dit)
        dit = ditpath;
    if (!vae)
        vae = vaepath;
    int ph = height / 16, pw = width / 16, ni = ph * pw, rc = 1;
    size_t count = (size_t)ni * 128;
    float *text = read_raw(textfile, (size_t)nt * 20480), *ti = calloc((size_t)nt * 4, 4),
          *imids = ids(ph, pw), *cond = NULL, *ci = NULL, *image = NULL,
          *velocity = malloc(count * 4);
    if (nc) {
        cond = read_raw(condfile, (size_t)nc * 128);
        ci = read_raw(condidsfile, (size_t)nc * 4);
    }
    image = latfile ? read_raw(latfile, count) : malloc(count * 4);
    if (!text || !ti || !imids || !image || !velocity || (nc && (!cond || !ci)))
        goto done;
    for (int i = 0; i < nt; i++)
        ti[i * 4 + 3] = i;
    if (!latfile) {
        rng_state = (unsigned)seed + 1;
        for (size_t j = 0; j < count; j++)
            image[j] = bf16_round(gaussian());
    }
    hip_flux3_set_trace(r, trace_dir);
    if (hip_flux3_load(r, dit))
        goto done;
    for (int i = 0; i < (one ? 1 : steps); i++) {
        float cur = one ? t : schedule(i, steps), next = one ? target : schedule(i + 1, steps);
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        if (hip_flux3_step(r, image, imids, ni, cond, ci, nc, text, ti, nt, cur, next, velocity))
            goto done;
        clock_gettime(CLOCK_MONOTONIC, &b);
        fprintf(stderr, "flux3: step %d/%d %.6f -> %.6f, %.3f s\n", i + 1, one ? 1 : steps, cur,
                next, b.tv_sec - a.tv_sec + (b.tv_nsec - a.tv_nsec) * 1e-9);
        if (!one)
            for (size_t j = 0; j < count; j++)
                image[j] += bf16_round((next - cur) * velocity[j]);
    }
    if (dump && write_raw(dump, one ? velocity : image, count)) {
        fprintf(stderr, "could not write %s\n", dump);
        goto done;
    }
    hip_flux3_free(r);
    r = NULL;
    if (!one && !no_vae &&
        decode_vae(vae, image, width, height, device, verbose, vae_wmma, reserve, output, dump_rgb))
        goto done;
    rc = 0;
done:
    hip_flux3_free(r);
    free(text);
    free(ti);
    free(imids);
    free(cond);
    free(ci);
    free(image);
    free(velocity);
    return rc;
}
