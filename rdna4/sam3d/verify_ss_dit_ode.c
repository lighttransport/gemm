/*
 * verify_ss_dit_ode — Phase 2c.14 GPU shortcut-ODE diff vs CPU host.
 *
 * Loads the SS Flow DiT checkpoint, runs the full shortcut ODE (default
 * 2 steps) on identical random cond + seed via:
 *   (a) the runner's `hip_sam3d_run_ss_dit` (now GPU-driven)
 *   (b) the static host `sam3d_cpu_ss_dit_run_ode`
 * and diffs the resulting [8,16,16,16] NCDHW SHAPE latent.
 *
 * Usage:
 *   ./verify_ss_dit_ode --safetensors-dir DIR
 *                       [--steps 2] [--nc 2740] [--threshold 5e-3] [-v]
 */

#include "hip_sam3d_runner.h"
#include "sam3d_cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float urand(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return (float)((*s) >> 8) / (float)(1u << 24);
}
static float max_abs(const float *a, const float *b, size_t n, double *mean) {
    double sum = 0.0; float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float dif = fabsf(a[i] - b[i]);
        if (dif > mx) mx = dif;
        sum += dif;
    }
    if (mean) *mean = sum / (n > 0 ? n : 1);
    return mx;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL;
    int N_c = 2740, steps = 2, verbose = 0;
    float threshold = 5e-3f, cfg_scale = 2.0f;
    uint64_t seed = 42;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(a, "--nc")        && i+1 < argc) N_c = atoi(argv[++i]);
        else if (!strcmp(a, "--steps")     && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--cfg")       && i+1 < argc) cfg_scale = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--seed")      && i+1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "-v"))                        verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (!sft_dir) {
        fprintf(stderr, "usage: %s --safetensors-dir DIR [--steps N] [--nc N] [-v]\n", argv[0]);
        return 2;
    }

    /* Build a deterministic random cond. */
    int cond_ch = 1024;  /* SS DiT cond channels — fixed by ckpt geometry. */
    size_t cond_n = (size_t)N_c * cond_ch;
    float *cond = (float *)malloc(cond_n * sizeof(float));
    uint32_t rng = 0xCA75u;
    for (size_t k = 0; k < cond_n; k++) cond[k] = urand(&rng) * 2.f - 1.f;

    /* === GPU ODE (runner) === */
    hip_sam3d_config cfg = {0};
    cfg.safetensors_dir = sft_dir;
    cfg.verbose         = verbose;
    cfg.precision       = "fp16";
    cfg.seed            = seed;
    cfg.ss_steps        = steps;
    cfg.cfg_scale       = cfg_scale;
    hip_sam3d_ctx *ctx = hip_sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "hip_sam3d_create failed\n"); free(cond); return 3; }
    if (hip_sam3d_debug_override_cond(ctx, cond, N_c, cond_ch) != 0) {
        fprintf(stderr, "override_cond failed\n");
        hip_sam3d_destroy(ctx); free(cond); return 4;
    }
    fprintf(stderr, "[verify_ss_dit_ode] running GPU ODE (steps=%d)...\n", steps);
    if (hip_sam3d_run_ss_dit(ctx) != 0) {
        fprintf(stderr, "GPU run_ss_dit failed\n");
        hip_sam3d_destroy(ctx); free(cond); return 5;
    }
    int dims[4] = {0};
    int n_lat_dims = 4;
    int gpu_numel = 8 * 16 * 16 * 16;
    float *gpu_lat = (float *)malloc((size_t)gpu_numel * sizeof(float));
    if (hip_sam3d_get_ss_latent(ctx, gpu_lat, dims) != 0) {
        fprintf(stderr, "get_ss_latent failed\n");
        free(gpu_lat); hip_sam3d_destroy(ctx); free(cond); return 6;
    }
    (void)n_lat_dims;

    /* === CPU ODE (reference) === */
    fprintf(stderr, "[verify_ss_dit_ode] running CPU ODE (reference)...\n");
    sam3d_cpu_ss_dit *w = sam3d_cpu_ss_dit_load(sft_dir);
    if (!w) {
        fprintf(stderr, "cpu ss_dit load failed\n");
        free(gpu_lat); hip_sam3d_destroy(ctx); free(cond); return 7;
    }
    float *cpu_lat = (float *)malloc((size_t)gpu_numel * sizeof(float));
    if (sam3d_cpu_ss_dit_run_ode(w, cond, N_c, steps, seed, cfg_scale,
                                 /*nthr*/ 32, cpu_lat) != 0) {
        fprintf(stderr, "cpu run_ode failed\n");
        free(gpu_lat); free(cpu_lat); sam3d_cpu_ss_dit_free(w);
        hip_sam3d_destroy(ctx); free(cond); return 8;
    }
    sam3d_cpu_ss_dit_free(w);

    double mean = 0.0;
    float mx = max_abs(gpu_lat, cpu_lat, (size_t)gpu_numel, &mean);
    int ok = (mx <= threshold);
    fprintf(stderr,
        "[verify_ss_dit_ode] steps=%d N_c=%d  ss_latent[8,16,16,16]  "
        "max_abs=%.4g (mean %.4g)  %s (threshold %.1g)\n",
        steps, N_c, (double)mx, mean,
        ok ? "OK" : "FAIL", (double)threshold);

    free(gpu_lat); free(cpu_lat); free(cond);
    hip_sam3d_destroy(ctx);
    return ok ? 0 : 9;
}
