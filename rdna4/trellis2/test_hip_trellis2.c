/*
 * test_hip_trellis2.c - TRELLIS.2 Stage 1 HIP/ROCm test harness (RDNA4)
 *
 * Usage:
 *   ./test_hip_trellis2 --dit <st> --decoder <st> --features <npy> [options]
 *
 * Modes:
 *   --full           Full 12-step sampling + decode → occupancy.npy
 *   --dit-only       Single DiT step (use --noise <npy> or random)
 *   --decode-only    Decoder only (from --latent <npy>)
 *   --dump-blocks    Save hip_block{00-29}_hidden.npy for layer-by-layer compare
 *   --verify-step N  Run N steps and save hip_latent_stepN.npy per step
 *
 * Occupancy output is a [64,64,64] float32 .npy (logits).
 * Use cpu/trellis2/test_trellis2 mesh to convert to .obj.
 *
 * Build: make
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "hip_trellis2_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ======================================================================== */
/* NumPy .npy I/O                                                           */
/* ======================================================================== */

static void write_npy_f32(const char *path, const float *data,
                          const int *dims, int ndim) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    char shape_str[256];
    int slen = 0;
    slen += snprintf(shape_str + slen, sizeof(shape_str) - slen, "(");
    size_t n_elem = 1;
    for (int i = 0; i < ndim; i++) {
        slen += snprintf(shape_str + slen, sizeof(shape_str) - slen, "%d,", dims[i]);
        n_elem *= (size_t)dims[i];
    }
    slen += snprintf(shape_str + slen, sizeof(shape_str) - slen, ")");
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape_str);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), n_elem, f);
    fclose(f);
    fprintf(stderr, "Wrote %s (", path);
    for (int i = 0; i < ndim; i++) fprintf(stderr, "%s%d", i ? "x" : "", dims[i]);
    fprintf(stderr, ", float32)\n");
}

static float *read_npy_f32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot read %s\n", path); return NULL; }
    fseek(f, 8, SEEK_SET);
    uint16_t header_len;
    if (fread(&header_len, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *header = (char *)malloc(header_len + 1);
    if (fread(header, 1, header_len, f) != (size_t)header_len) {
        free(header); fclose(f); return NULL;
    }
    header[header_len] = '\0';
    int fortran_order = (strstr(header, "'fortran_order': True") != NULL);
    *ndim = 0;
    char *sp = strstr(header, "shape");
    if (sp) {
        sp = strchr(sp, '(');
        if (sp) {
            sp++;
            while (*sp && *sp != ')') {
                while (*sp == ' ' || *sp == ',') sp++;
                if (*sp == ')') break;
                dims[*ndim] = (int)strtol(sp, &sp, 10);
                (*ndim)++;
                if (*ndim >= 8) break;
            }
        }
    }
    size_t n_elem = 1;
    for (int i = 0; i < *ndim; i++) n_elem *= (size_t)dims[i];
    float *data = (float *)malloc(n_elem * sizeof(float));
    size_t got = fread(data, sizeof(float), n_elem, f);
    fclose(f);
    free(header);
    if (got != n_elem)
        fprintf(stderr, "Warning: read %zu of %zu elements from %s\n", got, n_elem, path);
    if (fortran_order && *ndim >= 2) {
        /* Transpose from Fortran to C layout. Only handle 2D for simplicity. */
        if (*ndim == 2) {
            float *t = (float *)malloc(n_elem * sizeof(float));
            for (int i = 0; i < dims[0]; i++)
                for (int j = 0; j < dims[1]; j++)
                    t[i*dims[1]+j] = data[j*dims[0]+i];
            free(data);
            data = t;
            fprintf(stderr, "Note: converted %s from fortran_order to C layout\n", path);
        } else {
            fprintf(stderr, "ERROR: %s has fortran_order=True with ndim=%d — not supported\n",
                    path, *ndim);
        }
    }
    fprintf(stderr, "Read %s: (", path);
    for (int i = 0; i < *ndim; i++) fprintf(stderr, "%s%d", i ? ", " : "", dims[i]);
    fprintf(stderr, "), %zu elems\n", n_elem);
    return data;
}

/* ======================================================================== */
/* PRNG (Box-Muller) for random noise generation                           */
/* ======================================================================== */

/* xoshiro256** RNG — matches CPU trellis2_stage1.h and CUDA test_cuda_trellis2.c */
typedef struct { uint64_t s[4]; } rng_t;
static rng_t rng_g;

static uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64-k)); }
static uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s; uint64_t result = rotl64(s[1]*5,7)*9;
    uint64_t t = s[1]<<17; s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=t; s[3]=rotl64(s[3],45);
    return result;
}

static void rng_seed(uint64_t seed) {
    rng_g.s[0] = seed; rng_g.s[1] = seed ^ 0x9E3779B97F4A7C15ULL;
    rng_g.s[2] = seed ^ 0x6C62272E07BB0142ULL; rng_g.s[3] = seed ^ 0xBF58476D1CE4E5B9ULL;
    for (int i = 0; i < 8; i++) rng_next(&rng_g);
}

static float randn(void) {
    double u1 = ((double)(rng_next(&rng_g)>>11)+0.5)/(double)(1ULL<<53);
    double u2 = ((double)(rng_next(&rng_g)>>11)+0.5)/(double)(1ULL<<53);
    return (float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2));
}

/* ======================================================================== */
/* Helpers                                                                  */
/* ======================================================================== */

static void print_stats(const char *label, const float *a, int n) {
    if (n <= 0) return;
    float mn = a[0], mx = a[0]; double s = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] < mn) mn = a[i];
        if (a[i] > mx) mx = a[i];
        s += a[i];
    }
    fprintf(stderr, "  %s: n=%d min=%.4f max=%.4f mean=%.6f\n",
            label, n, mn, mx, s / n);
    fprintf(stderr, "    first8: [");
    for (int i = 0; i < 8 && i < n; i++) fprintf(stderr, "%.4f%s", a[i], i < 7 && i < n-1 ? ", " : "");
    fprintf(stderr, "]\n");
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Euler step: x = x - dt * velocity  (flow matching sign convention) */
static void euler_step(float *x, const float *v, float dt, int n) {
    for (int i = 0; i < n; i++) x[i] -= dt * v[i];
}

/* CFG rescale: compute x0 from v, combine with std-ratio matching, then
 * recover the effective velocity for the Euler step.
 * cfg_scale=7.5, rescale=0.7, sigma_min=1e-5 */
static void cfg_rescale_step(float *x, const float *v_cond, const float *v_uncond,
                              float t_cur, float dt, int n,
                              float cfg_scale, float rescale_w, float sigma_min) {
    float *v_cfg = (float *)malloc(n * sizeof(float));
    float *x0_cond = (float *)malloc(n * sizeof(float));
    float *x0_cfg  = (float *)malloc(n * sizeof(float));
    float coeff = sigma_min + (1.0f - sigma_min) * t_cur;

    for (int i = 0; i < n; i++)
        v_cfg[i] = cfg_scale * v_cond[i] + (1.0f - cfg_scale) * v_uncond[i];

    for (int i = 0; i < n; i++) {
        x0_cond[i] = (1.0f - sigma_min) * x[i] - coeff * v_cond[i];
        x0_cfg[i]  = (1.0f - sigma_min) * x[i] - coeff * v_cfg[i];
    }

    /* std-ratio matching: x0 = rescale * (x0_cfg * std_cond/std_cfg) + (1-rescale) * x0_cfg */
    double sum_c = 0, sum2_c = 0, sum_g = 0, sum2_g = 0;
    for (int i = 0; i < n; i++) {
        sum_c += x0_cond[i]; sum2_c += (double)x0_cond[i] * x0_cond[i];
        sum_g += x0_cfg[i];  sum2_g += (double)x0_cfg[i]  * x0_cfg[i];
    }
    double mean_c = sum_c / n, mean_g = sum_g / n;
    double var_c = sum2_c / n - mean_c * mean_c;
    double var_g = sum2_g / n - mean_g * mean_g;
    float std_ratio = (var_g > 1e-12) ? (float)(sqrt(var_c) / sqrt(var_g)) : 1.0f;

    for (int i = 0; i < n; i++) {
        float x0 = rescale_w * (x0_cfg[i] * std_ratio) + (1.0f - rescale_w) * x0_cfg[i];
        float v_eff = ((1.0f - sigma_min) * x[i] - x0) / coeff;
        x[i] -= dt * v_eff;
    }
    free(v_cfg); free(x0_cond); free(x0_cfg);
}

/* ======================================================================== */
/* Main                                                                     */
/* ======================================================================== */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "TRELLIS.2 Stage 1 HIP/ROCm Test Harness\n"
        "========================================\n"
        "Usage: %s [options]\n"
        "\n"
        "Modes (pick one):\n"
        "  --full           Full 12-step sampling + decode -> occupancy.npy\n"
        "  --dit-only       Single DiT step (use --noise or random)\n"
        "  --decode-only    Decoder only (requires --latent)\n"
        "  --dump-blocks    Dump all 30 block hidden states for verification\n"
        "  --verify-step N  Run N steps, save hip_latent_stepN.npy per step\n"
        "\n"
        "Required:\n"
        "  --dit    <path>   Stage 1 DiT safetensors\n"
        "  --decoder <path>  Decoder safetensors\n"
        "  --features <npy>  DINOv3 features [1029, 1024] (required for dit modes)\n"
        "\n"
        "Optional:\n"
        "  --noise <npy>     Initial noise [4096, 8] (default: random)\n"
        "  --latent <npy>    Latent [8,16,16,16] (for --decode-only)\n"
        "  --seed N          Random seed (default: 42)\n"
        "  --steps N         Number of Euler steps (default: 12)\n"
        "  --output-dir <d>  Directory for output files (default: .)\n"
        "  -d N              HIP device ID (default: 0)\n"
        "  -v                Verbose output\n",
        prog);
}

int main(int argc, char **argv) {
    /* --- Parse args --- */
    const char *dit_path     = NULL;
    const char *dec_path     = NULL;
    const char *feat_path    = NULL;
    const char *noise_path   = NULL;
    const char *latent_path  = NULL;
    const char *output_dir   = ".";
    int device_id   = 0;
    int verbose     = 0;
    int n_steps     = 12;
    uint64_t seed   = 42;
    int verify_n    = -1;
    int no_cfg      = 0;
    float dit_t     = 0.5f;

    /* Mode flags */
    int mode_full        = 0;
    int mode_dit_only    = 0;
    int mode_decode_only = 0;
    int mode_dump_blocks = 0;
    int mode_verify_step = 0;
    int mode_dump_b0     = 0;

    if (argc < 2) { print_usage(argv[0]); return 1; }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dit")        && i+1 < argc) dit_path    = argv[++i];
        else if (!strcmp(argv[i], "--decoder") && i+1 < argc) dec_path  = argv[++i];
        else if (!strcmp(argv[i], "--features") && i+1 < argc) feat_path = argv[++i];
        else if (!strcmp(argv[i], "--noise")   && i+1 < argc) noise_path = argv[++i];
        else if (!strcmp(argv[i], "--latent")  && i+1 < argc) latent_path = argv[++i];
        else if (!strcmp(argv[i], "--output-dir") && i+1 < argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "--seed")    && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--steps")   && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d")        && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v"))  verbose = 1;
        else if (!strcmp(argv[i], "--no-cfg")) no_cfg = 1;
        else if (!strcmp(argv[i], "--t") && i+1 < argc) dit_t = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--full"))          mode_full = 1;
        else if (!strcmp(argv[i], "--dit-only"))      mode_dit_only = 1;
        else if (!strcmp(argv[i], "--decode-only"))   mode_decode_only = 1;
        else if (!strcmp(argv[i], "--dump-blocks"))   mode_dump_blocks = 1;
        else if (!strcmp(argv[i], "--dump-b0"))       mode_dump_b0 = 1;
        else if (!strcmp(argv[i], "--verify-step") && i+1 < argc) {
            mode_verify_step = 1;
            verify_n = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]); return 0;
        }
    }

    /* Default mode: full */
    if (!mode_full && !mode_dit_only && !mode_decode_only &&
        !mode_dump_blocks && !mode_verify_step && !mode_dump_b0) {
        mode_full = 1;
    }

    /* Validate */
    int need_dit     = mode_full || mode_dit_only || mode_dump_blocks || mode_verify_step || mode_dump_b0;
    int need_decoder = mode_full || mode_decode_only || mode_verify_step;

    if (need_dit && !dit_path) {
        fprintf(stderr, "Error: --dit required for this mode\n"); return 1;
    }
    if (need_decoder && !dec_path) {
        fprintf(stderr, "Error: --decoder required for this mode\n"); return 1;
    }
    if (need_dit && !feat_path) {
        fprintf(stderr, "Error: --features required for DiT modes\n"); return 1;
    }
    if (mode_decode_only && !latent_path) {
        fprintf(stderr, "Error: --latent required for --decode-only\n"); return 1;
    }

    /* Shapes */
    const int N_TOK  = 4096;
    const int IN_CH  = 8;
    const int GRID   = 16;
    const int N_COND = 1029;
    const int COND_DIM = 1024;
    const int DIT_DIM  = 1536;
    const int OCC_N    = 64 * 64 * 64;

    /* --- Init HIP --- */
    fprintf(stderr, "Init HIP device %d...\n", device_id);
    hip_trellis2_runner *r = hip_trellis2_init(device_id, verbose);
    if (!r) { fprintf(stderr, "HIP init failed\n"); return 1; }

    /* --- Load weights --- */
    if (need_dit) {
        fprintf(stderr, "Loading DiT: %s\n", dit_path);
        double t0 = now_ms();
        if (hip_trellis2_load_dit(r, dit_path) != 0) {
            fprintf(stderr, "DiT load failed\n"); hip_trellis2_free(r); return 1;
        }
        fprintf(stderr, "  DiT loaded in %.1f ms\n", now_ms() - t0);
    }
    if (need_decoder) {
        fprintf(stderr, "Loading decoder: %s\n", dec_path);
        double t0 = now_ms();
        if (hip_trellis2_load_decoder(r, dec_path) != 0) {
            fprintf(stderr, "Decoder load failed\n"); hip_trellis2_free(r); return 1;
        }
        fprintf(stderr, "  Decoder loaded in %.1f ms\n", now_ms() - t0);
    }

    /* --- Load features --- */
    float *features = NULL;
    if (feat_path) {
        int ndim, dims[8];
        features = read_npy_f32(feat_path, &ndim, dims);
        if (!features) { hip_trellis2_free(r); return 1; }
        if (ndim != 2 || dims[0] != N_COND || dims[1] != COND_DIM) {
            fprintf(stderr, "Warning: features shape unexpected: [%d,%d] (expected [%d,%d])\n",
                    ndim >= 1 ? dims[0] : 0, ndim >= 2 ? dims[1] : 0, N_COND, COND_DIM);
        }
        print_stats("features", features, N_COND * COND_DIM);
    }

    /* --- Load / generate noise --- */
    float *noise = NULL;
    if (need_dit) {
        noise = (float *)malloc((size_t)N_TOK * IN_CH * sizeof(float));
        if (noise_path) {
            int ndim, dims[8];
            float *loaded = read_npy_f32(noise_path, &ndim, dims);
            if (!loaded) { free(noise); free(features); hip_trellis2_free(r); return 1; }
            memcpy(noise, loaded, (size_t)N_TOK * IN_CH * sizeof(float));
            free(loaded);
        } else {
            rng_seed(seed);
            for (int i = 0; i < N_TOK * IN_CH; i++) noise[i] = randn();
            fprintf(stderr, "Generated random noise (seed=%llu)\n", (unsigned long long)seed);
        }
        print_stats("noise", noise, N_TOK * IN_CH);
    }

    /* Output path helper */
    char outpath[1024];
    #define OUTPATH(fmt, ...) \
        snprintf(outpath, sizeof(outpath), "%s/" fmt, output_dir, ##__VA_ARGS__)

    /* ================================================================== */
    /* Mode: --dit-only                                                    */
    /* ================================================================== */
    if (mode_dit_only) {
        fprintf(stderr, "\n=== Mode: dit-only (t=%.4f) ===\n", dit_t);
        float *vel = (float *)calloc((size_t)N_TOK * IN_CH, sizeof(float));

        double t0 = now_ms();
        if (hip_trellis2_dit_step(r, noise, features, dit_t, vel) != 0) {
            fprintf(stderr, "DiT step failed\n");
            free(vel); goto cleanup;
        }
        fprintf(stderr, "DiT step: %.1f ms\n", now_ms() - t0);
        print_stats("velocity", vel, N_TOK * IN_CH);

        OUTPATH("hip_velocity.npy");
        int dims[2] = {N_TOK, IN_CH};
        write_npy_f32(outpath, vel, dims, 2);
        free(vel);
    }

    /* ================================================================== */
    /* Mode: --dump-blocks                                                 */
    /* ================================================================== */
    if (mode_dump_blocks) {
        fprintf(stderr, "\n=== Mode: dump-blocks (t=%.4f, all %d blocks) ===\n", dit_t, 30);
        float *hidden = (float *)malloc((size_t)N_TOK * DIT_DIM * sizeof(float));
        if (!hidden) { fprintf(stderr, "OOM\n"); goto cleanup; }

        int ok = 1;
        for (int blk = 0; blk < 30; blk++) {
            double t0 = now_ms();
            if (hip_trellis2_dump_block(r, noise, features, dit_t, blk, hidden) != 0) {
                fprintf(stderr, "dump_block %d failed\n", blk);
                ok = 0; break;
            }
            fprintf(stderr, "Block %02d: %.1f ms  ", blk, now_ms() - t0);
            print_stats("hidden", hidden, N_TOK * DIT_DIM);

            OUTPATH("hip_block%02d_hidden.npy", blk);
            int dims[2] = {N_TOK, DIT_DIM};
            write_npy_f32(outpath, hidden, dims, 2);

            /* Invalidate KV cache between dump calls (features unchanged, but re-upload) */
            /* No need — KV cache is valid for same features */
        }
        free(hidden);
        if (!ok) goto cleanup;
        fprintf(stderr, "Saved hip_block{00-29}_hidden.npy to %s/\n", output_dir);
        fprintf(stderr, "\nCompare with:\n");
        fprintf(stderr, "  cd ref/trellis2\n");
        fprintf(stderr, "  for N in $(seq -f '%%02g' 0 29); do\n");
        fprintf(stderr, "    python make_comparison.py ref_block${N}_hidden.npy %s/hip_block${N}_hidden.npy\n",
                output_dir);
        fprintf(stderr, "  done\n");
    }

    /* ================================================================== */
    /* Mode: --dump-b0 (block 0 detailed intermediates)                    */
    /* ================================================================== */
    if (mode_dump_b0) {
        fprintf(stderr, "\n=== Mode: dump-b0 (t=%.4f) ===\n", dit_t);
        size_t N = (size_t)N_TOK * DIT_DIM;
        hip_trellis2_b0_dbg dbg = {0};
        dbg.input_embed= (float *)malloc(N * sizeof(float));
        dbg.mod        = (float *)malloc(6 * DIT_DIM * sizeof(float));
        dbg.ln_h_sa    = (float *)malloc(N * sizeof(float));
        dbg.q_post     = (float *)malloc(N * sizeof(float));
        dbg.k_post     = (float *)malloc(N * sizeof(float));
        dbg.v          = (float *)malloc(N * sizeof(float));
        dbg.sa_proj    = (float *)malloc(N * sizeof(float));
        dbg.h_post_sa  = (float *)malloc(N * sizeof(float));
        dbg.ca_proj    = (float *)malloc(N * sizeof(float));
        dbg.h_post_ca  = (float *)malloc(N * sizeof(float));
        dbg.ln_h_mlp   = (float *)malloc(N * sizeof(float));
        dbg.mlp_proj   = (float *)malloc(N * sizeof(float));
        dbg.h_post_mlp = (float *)malloc(N * sizeof(float));

        if (hip_trellis2_dump_b0_detail(r, noise, features, dit_t, &dbg) != 0) {
            fprintf(stderr, "dump_b0_detail failed\n");
        } else {
            int d1[1] = {6 * DIT_DIM};
            int d2[2] = {N_TOK, DIT_DIM};
            OUTPATH("hip_b0_input_embed.npy"); write_npy_f32(outpath, dbg.input_embed, d2, 2);
            OUTPATH("hip_b0_mod.npy");        write_npy_f32(outpath, dbg.mod, d1, 1);
            OUTPATH("hip_b0_ln_h_sa.npy");    write_npy_f32(outpath, dbg.ln_h_sa,   d2, 2);
            OUTPATH("hip_b0_q_post.npy");     write_npy_f32(outpath, dbg.q_post,    d2, 2);
            OUTPATH("hip_b0_k_post.npy");     write_npy_f32(outpath, dbg.k_post,    d2, 2);
            OUTPATH("hip_b0_v.npy");          write_npy_f32(outpath, dbg.v,         d2, 2);
            OUTPATH("hip_b0_sa_proj.npy");    write_npy_f32(outpath, dbg.sa_proj,   d2, 2);
            OUTPATH("hip_b0_h_post_sa.npy");  write_npy_f32(outpath, dbg.h_post_sa, d2, 2);
            OUTPATH("hip_b0_ca_proj.npy");    write_npy_f32(outpath, dbg.ca_proj,   d2, 2);
            OUTPATH("hip_b0_h_post_ca.npy");  write_npy_f32(outpath, dbg.h_post_ca, d2, 2);
            OUTPATH("hip_b0_ln_h_mlp.npy");   write_npy_f32(outpath, dbg.ln_h_mlp,  d2, 2);
            OUTPATH("hip_b0_mlp_proj.npy");   write_npy_f32(outpath, dbg.mlp_proj,  d2, 2);
            OUTPATH("hip_b0_h_post_mlp.npy"); write_npy_f32(outpath, dbg.h_post_mlp,d2, 2);
            fprintf(stderr, "Saved hip_b0_*.npy to %s/\n", output_dir);
        }
        free(dbg.input_embed);
        free(dbg.mod); free(dbg.ln_h_sa); free(dbg.q_post); free(dbg.k_post);
        free(dbg.v); free(dbg.sa_proj); free(dbg.h_post_sa);
        free(dbg.ca_proj); free(dbg.h_post_ca); free(dbg.ln_h_mlp);
        free(dbg.mlp_proj); free(dbg.h_post_mlp);
    }

    /* ================================================================== */
    /* Mode: --decode-only                                                 */
    /* ================================================================== */
    if (mode_decode_only) {
        fprintf(stderr, "\n=== Mode: decode-only ===\n");
        int ndim, dims[8];
        float *latent = read_npy_f32(latent_path, &ndim, dims);
        if (!latent) goto cleanup;

        float *occupancy = (float *)calloc(OCC_N, sizeof(float));
        double t0 = now_ms();
        if (hip_trellis2_decode(r, latent, occupancy) != 0) {
            fprintf(stderr, "Decode failed\n");
            free(latent); free(occupancy); goto cleanup;
        }
        fprintf(stderr, "Decode: %.1f ms\n", now_ms() - t0);
        print_stats("occupancy", occupancy, OCC_N);

        int occ_count = 0;
        for (int i = 0; i < OCC_N; i++) if (occupancy[i] > 0.0f) occ_count++;
        fprintf(stderr, "  Voxels > 0: %d / %d (%.2f%%)\n",
                occ_count, OCC_N, 100.0f * occ_count / OCC_N);

        OUTPATH("hip_occupancy.npy");
        int out_dims[3] = {64, 64, 64};
        write_npy_f32(outpath, occupancy, out_dims, 3);

        free(latent); free(occupancy);
    }

    /* ================================================================== */
    /* Mode: --verify-step N (run N steps, save per-step latent)          */
    /* ================================================================== */
    if (mode_verify_step) {
        fprintf(stderr, "\n=== Mode: verify-step %d ===\n", verify_n);
        if (verify_n < 1 || verify_n > n_steps) {
            fprintf(stderr, "verify-step N must be in [1, %d]\n", n_steps);
            goto cleanup;
        }

        /* Euler timestep schedule: t = 1 to 0 in n_steps */
        float *x = (float *)malloc((size_t)N_TOK * IN_CH * sizeof(float));
        float *vel = (float *)malloc((size_t)N_TOK * IN_CH * sizeof(float));
        memcpy(x, noise, (size_t)N_TOK * IN_CH * sizeof(float));

        for (int step = 0; step < verify_n; step++) {
            float t_cur  = 1.0f - (float)step / (float)n_steps;
            float t_next = 1.0f - (float)(step + 1) / (float)n_steps;
            float t_hip  = t_cur;  /* runner scales by 1000 internally */
            float dt     = t_cur - t_next;

            double t0 = now_ms();
            if (hip_trellis2_dit_step(r, x, features, t_hip, vel) != 0) {
                fprintf(stderr, "Step %d failed\n", step);
                free(x); free(vel); goto cleanup;
            }
            euler_step(x, vel, dt, N_TOK * IN_CH);
            fprintf(stderr, "Step %02d (t=%.4f -> %.4f): %.1f ms\n",
                    step, t_cur, t_next, now_ms() - t0);

            /* Save per-step latent [16, 16, 16, 8] (tok-major; transpose to [8,16,16,16] for PyTorch compare) */
            OUTPATH("hip_latent_step%02d.npy", step);
            int dims[4] = {GRID, GRID, GRID, IN_CH};
            write_npy_f32(outpath, x, dims, 4);
        }
        free(x); free(vel);
    }

    /* ================================================================== */
    /* Mode: --full (full 12-step sampling + decode)                      */
    /* ================================================================== */
    if (mode_full) {
        fprintf(stderr, "\n=== Mode: full (%d steps) ===\n", n_steps);

        float *x   = (float *)malloc((size_t)N_TOK * IN_CH * sizeof(float));
        float *vel = (float *)malloc((size_t)N_TOK * IN_CH * sizeof(float));
        memcpy(x, noise, (size_t)N_TOK * IN_CH * sizeof(float));

        /* Rescaled timestep schedule: t_seq = rescale_t * t / (1 + (rescale_t-1)*t)
         * with rescale_t = 5.0 (from pipeline.json) */
        float *t_seq = (float *)malloc((n_steps + 1) * sizeof(float));
        for (int i = 0; i <= n_steps; i++) {
            float t_lin = 1.0f - (float)i / (float)n_steps;
            t_seq[i] = 5.0f * t_lin / (1.0f + 4.0f * t_lin);
        }

        /* CFG parameters */
        float cfg_scale = 7.5f, cfg_rescale = 0.7f, sigma_min = 1e-5f;
        float cfg_interval_lo = 0.6f, cfg_interval_hi = 1.0f;
        float *neg_cond = (float *)calloc((size_t)N_COND * COND_DIM, sizeof(float));
        float *vel_uncond = (float *)malloc((size_t)N_TOK * IN_CH * sizeof(float));

        double t_total = 0;
        for (int step = 0; step < n_steps; step++) {
            float t_cur  = t_seq[step];
            float t_next = t_seq[step + 1];
            float t_hip  = t_cur;  /* runner scales by 1000 internally */
            float dt     = t_cur - t_next;
            int use_cfg  = (!no_cfg) && (t_cur >= cfg_interval_lo && t_cur <= cfg_interval_hi);

            double t0 = now_ms();
            if (hip_trellis2_dit_step(r, x, features, t_hip, vel) != 0) {
                fprintf(stderr, "DiT step %d failed\n", step);
                free(x); free(vel); free(neg_cond); free(vel_uncond); goto cleanup;
            }
            if (use_cfg) {
                hip_trellis2_invalidate_kv(r);
                if (hip_trellis2_dit_step(r, x, neg_cond, t_hip, vel_uncond) != 0) {
                    fprintf(stderr, "DiT uncond step %d failed\n", step);
                    free(x); free(vel); free(neg_cond); free(vel_uncond); goto cleanup;
                }
                hip_trellis2_invalidate_kv(r);
            }
            double step_ms = now_ms() - t0;
            t_total += step_ms;
            if (use_cfg) {
                cfg_rescale_step(x, vel, vel_uncond, t_cur, dt, N_TOK * IN_CH,
                                 cfg_scale, cfg_rescale, sigma_min);
            } else {
                euler_step(x, vel, dt, N_TOK * IN_CH);
            }
            {
                int nn = N_TOK * IN_CH; double s2 = 0, vmn = 0;
                for (int _i = 0; _i < nn; _i++) { vmn += vel[_i]; }
                vmn /= nn;
                for (int _i = 0; _i < nn; _i++) { double d = vel[_i]-vmn; s2 += d*d; }
                fprintf(stderr, "Step %02d/%02d (t=%.3f->%.3f): %.1f ms  vel_std=%.4f%s\n",
                        step + 1, n_steps, t_cur, t_next, step_ms, (float)sqrt(s2/nn),
                        use_cfg ? " [CFG]" : "");
            }
            /* Save per-step latent */
            OUTPATH("hip_full_step%02d.npy", step);
            { int dims[4] = {GRID, GRID, GRID, IN_CH};
              write_npy_f32(outpath, x, dims, 4); }
        }
        free(neg_cond); free(vel_uncond); free(t_seq);
        fprintf(stderr, "Total DiT: %.1f ms (%.1f ms/step)\n",
                t_total, t_total / n_steps);
        print_stats("final latent", x, N_TOK * IN_CH);

        /* Save latent */
        {
            OUTPATH("hip_latent.npy");
            int dims[4] = {GRID, GRID, GRID, IN_CH};
            write_npy_f32(outpath, x, dims, 4);
        }

        /* Decode: DiT x is token-major (N_TOK, IN_CH); decoder wants (IN_CH, D,H,W). */
        fprintf(stderr, "\nDecoding...\n");
        float *occupancy = (float *)calloc(OCC_N, sizeof(float));
        float *lat_cf = (float *)malloc((size_t)N_TOK * IN_CH * sizeof(float));
        for (int s = 0; s < N_TOK; s++)
            for (int c = 0; c < IN_CH; c++)
                lat_cf[(size_t)c * N_TOK + s] = x[(size_t)s * IN_CH + c];
        double t0 = now_ms();
        if (hip_trellis2_decode(r, lat_cf, occupancy) != 0) {
            free(lat_cf);
            fprintf(stderr, "Decode failed\n");
            free(x); free(vel); free(occupancy); goto cleanup;
        }
        fprintf(stderr, "Decode: %.1f ms\n", now_ms() - t0);
        print_stats("occupancy", occupancy, OCC_N);

        int occ_count = 0;
        for (int i = 0; i < OCC_N; i++) if (occupancy[i] > 0.0f) occ_count++;
        fprintf(stderr, "Voxels > 0: %d / %d (%.2f%%)\n",
                occ_count, OCC_N, 100.0f * occ_count / OCC_N);

        OUTPATH("hip_occupancy.npy");
        {
            int dims[3] = {64, 64, 64};
            write_npy_f32(outpath, occupancy, dims, 3);
        }

        fprintf(stderr, "\nTo convert to mesh:\n");
        fprintf(stderr, "  cd cpu/trellis2 && ./test_trellis2 mesh %s/hip_occupancy.npy -o hip_mesh.obj\n",
                output_dir);

        free(lat_cf);
        free(x); free(vel); free(occupancy);
    }

cleanup:
    if (features) free(features);
    if (noise)    free(noise);
    hip_trellis2_free(r);
    return 0;
}
