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
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define T2_PBR_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/stb_image_write.h"
#include "../../common/trellis2_pbr.h"
#include "hip_trellis2_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

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
        "  --mesh           After --full, run SLAT DiT + shape_dec -> OBJ mesh\n"
        "\n"
        "Required:\n"
        "  --dit    <path>   Stage 1 DiT safetensors\n"
        "  --decoder <path>  Decoder safetensors\n"
        "  --features <npy>  DINOv3 features [1029, 1024] (required for dit modes)\n"
        "\n"
        "--mesh extras:\n"
        "  --slat-dit <path>   Shape SLAT DiT safetensors\n"
        "  --shape-dec <path>  shape_dec safetensors\n"
        "  --manifest <json>   manifest.json with shape_slat_normalization\n"
        "  --save-mesh <obj>   Output mesh path\n"
        "  --slat-steps N      SLAT Euler steps (default 12)\n"
        "  --tex               After --mesh, run tex DiT + tex_dec -> textured OBJ\n"
        "  --tex-dit <path>    Tex DiT safetensors\n"
        "  --tex-dec <path>    Tex decoder safetensors\n"
        "  --tex-steps N       Tex Euler steps (default 12)\n"
        "  --tex-res N         PBR texture resolution (default 1024)\n"
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

/* ======================================================================== */
/* End-to-end mesh path                                                     */
/* ======================================================================== */

/* Parse 32-element float array under a JSON key like
 * "shape_slat_normalization" or "tex_slat_normalization".
 * Returns 0 on success, fills mean[32] and std[32]. */
static int parse_slat_norm(const char *manifest_path, const char *root_key,
                           float *mean, float *std) {
    FILE *f = fopen(manifest_path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    buf[sz] = '\0'; fclose(f);

    char quoted[128];
    snprintf(quoted, sizeof(quoted), "\"%s\"", root_key);
    char *root = strstr(buf, quoted);
    if (!root) { free(buf); return -2; }
    for (int pass = 0; pass < 2; pass++) {
        const char *key = pass ? "\"std\"" : "\"mean\"";
        char *k = strstr(root, key);
        if (!k) { free(buf); return -3; }
        char *lb = strchr(k, '['); if (!lb) { free(buf); return -3; }
        char *p = lb + 1;
        for (int i = 0; i < 32; i++) {
            while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t') p++;
            char *end;
            float v = strtof(p, &end);
            if (end == p) { free(buf); return -4; }
            (pass ? std : mean)[i] = v;
            p = end;
        }
    }
    free(buf);
    return 0;
}

/* Back-compat wrapper for the shape SLAT call site. */
static int parse_shape_norm(const char *manifest_path, float *mean, float *std) {
    return parse_slat_norm(manifest_path, "shape_slat_normalization", mean, std);
}

/* Tiny .npy reader for int32 (no fortran_order support — used for coords). */
static int32_t *read_npy_i32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot read %s\n", path); return NULL; }
    fseek(f, 8, SEEK_SET);
    uint16_t hl; if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *h = (char *)malloc(hl + 1);
    if (fread(h, 1, hl, f) != (size_t)hl) { free(h); fclose(f); return NULL; }
    h[hl] = '\0';
    *ndim = 0;
    char *sp = strstr(h, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') {
            while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break;
            dims[*ndim] = (int)strtol(sp, &sp, 10);
            (*ndim)++;
            if (*ndim >= 8) break;
        } } }
    size_t ne = 1; for (int i = 0; i < *ndim; i++) ne *= (size_t)dims[i];
    int32_t *data = (int32_t *)malloc(ne * sizeof(int32_t));
    size_t got = fread(data, sizeof(int32_t), ne, f);
    fclose(f); free(h);
    if (got != ne) fprintf(stderr, "Warning: i32 read %zu of %zu\n", got, ne);
    return data;
}

/* Static OBJ writer (vertices + triangles, 1-based indexing). */
static int write_obj(const char *path, const float *verts, int n_verts,
                     const int *tris, int n_tris) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return -1; }
    for (int i = 0; i < n_verts; i++)
        fprintf(f, "v %.6f %.6f %.6f\n", verts[i*3+0], verts[i*3+1], verts[i*3+2]);
    for (int i = 0; i < n_tris; i++)
        fprintf(f, "f %d %d %d\n", tris[i*3+0]+1, tris[i*3+1]+1, tris[i*3+2]+1);
    fclose(f);
    return 0;
}

/* ======================================================================== */

int main(int argc, char **argv) {
    /* --- Parse args --- */
    const char *dit_path     = NULL;
    const char *dec_path     = NULL;
    const char *slat_path    = NULL;
    const char *shape_dec_path = NULL;
    const char *manifest_path = NULL;
    const char *save_mesh_path = NULL;
    const char *tex_dit_path   = NULL;
    const char *tex_dec_path   = NULL;
    int   tex_steps            = 12;
    int   tex_res              = 1024;
    const char *feat_path    = NULL;
    const char *noise_path   = NULL;
    const char *latent_path  = NULL;
    const char *output_dir   = ".";
    int device_id   = 0;
    int verbose     = 0;
    int n_steps     = 12;
    int slat_steps  = 12;
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
    int mode_mesh        = 0;
    int mode_tex         = 0;
    int mode_shape_only  = 0;
    const char *shape_feats_npy = NULL;
    const char *shape_coords_npy = NULL;

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
        else if (!strcmp(argv[i], "--slat-dit") && i+1 < argc) slat_path = argv[++i];
        else if (!strcmp(argv[i], "--shape-dec") && i+1 < argc) shape_dec_path = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i+1 < argc) manifest_path = argv[++i];
        else if (!strcmp(argv[i], "--save-mesh") && i+1 < argc) save_mesh_path = argv[++i];
        else if (!strcmp(argv[i], "--mesh"))           mode_mesh = 1;
        else if (!strcmp(argv[i], "--tex"))            mode_tex = 1;
        else if (!strcmp(argv[i], "--tex-dit") && i+1 < argc) tex_dit_path = argv[++i];
        else if (!strcmp(argv[i], "--tex-dec") && i+1 < argc) tex_dec_path = argv[++i];
        else if (!strcmp(argv[i], "--tex-steps") && i+1 < argc) tex_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tex-res") && i+1 < argc) tex_res = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slat-steps") && i+1 < argc) slat_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shape-only"))    mode_shape_only = 1;
        else if (!strcmp(argv[i], "--shape-feats-npy") && i+1 < argc) shape_feats_npy = argv[++i];
        else if (!strcmp(argv[i], "--shape-coords-npy") && i+1 < argc) shape_coords_npy = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]); return 0;
        }
    }

    /* Default mode: full */
    if (!mode_full && !mode_dit_only && !mode_decode_only &&
        !mode_dump_blocks && !mode_verify_step && !mode_dump_b0 && !mode_shape_only) {
        mode_full = 1;
    }

    /* On RDNA4 (gfx1201, ROCm 7.2.2) the power manager keeps mclk pinned at
     * level 0 (96 MHz) during tex_dec's sparse_conv3d kernels — the HIP async
     * queue never registers as compute-busy. AMD_SERIALIZE_KERNEL=3 forces a
     * sync after each launch (mclk ramps to level 5 = 1258 MHz). For the e2e
     * pipeline, the gap between shape_dec and tex_dec (tex DiT inference +
     * load) lets mclk drop again; HSA_ENABLE_SDMA=0 prevents that drop. Both
     * are needed for tex_dec to stay at 38× speedup through the full run.
     * The HIP runtime caches both env vars at init, so setenv() from main()
     * is too late — re-exec with them set in the environment instead. */
    if (mode_tex &&
        (!getenv("AMD_SERIALIZE_KERNEL") || !getenv("HSA_ENABLE_SDMA"))) {
        fprintf(stderr, "T2-HIP: re-exec with AMD_SERIALIZE_KERNEL=3 HSA_ENABLE_SDMA=0 "
                        "for tex_dec mclk ramp\n");
        if (!getenv("AMD_SERIALIZE_KERNEL")) setenv("AMD_SERIALIZE_KERNEL", "3", 1);
        if (!getenv("HSA_ENABLE_SDMA"))      setenv("HSA_ENABLE_SDMA",      "0", 1);
        execv(argv[0], argv);
        perror("execv");  /* falls through if execv fails — log + continue */
    }

    /* Shape-only fast path: load shape_dec, run on user-provided feats+coords,
     * write OBJ, exit. Bypasses DiT / decoder / SLAT entirely. */
    if (mode_shape_only) {
        if (!shape_dec_path || !shape_feats_npy || !shape_coords_npy || !save_mesh_path) {
            fprintf(stderr, "Error: --shape-only requires --shape-dec, --shape-feats-npy, "
                            "--shape-coords-npy, --save-mesh\n"); return 1;
        }
        hip_trellis2_runner *r = hip_trellis2_init(device_id, verbose);
        if (!r) { fprintf(stderr, "init failed\n"); return 1; }
        if (hip_trellis2_load_shape_dec(r, shape_dec_path) != 0) {
            fprintf(stderr, "load_shape_dec failed\n"); hip_trellis2_free(r); return 1;
        }
        int fnd = 0, fdims[8] = {0};
        float *feats = read_npy_f32(shape_feats_npy, &fnd, fdims);
        int cnd = 0, cdims[8] = {0};
        int32_t *coords = read_npy_i32(shape_coords_npy, &cnd, cdims);
        if (!feats || !coords) { fprintf(stderr, "bad feats/coords\n"); return 1; }
        int N = fdims[0], C = fdims[1];
        if (cnd != 2 || cdims[0] != N || cdims[1] != 4) {
            fprintf(stderr, "coords shape mismatch: expect [N=%d,4]\n", N); return 1;
        }
        fprintf(stderr, "shape-only: N=%d C=%d -> shape_dec\n", N, C);
        hip_trellis2_mesh mesh = {0};
        double t0 = now_ms();
        if (hip_trellis2_run_shape_dec(r, feats, coords, N, C, &mesh) != 0) {
            fprintf(stderr, "run_shape_dec failed\n"); return 1;
        }
        fprintf(stderr, "shape_dec: %.1f ms, mesh: %d verts, %d tris\n",
                now_ms() - t0, mesh.n_verts, mesh.n_tris);
        if (write_obj(save_mesh_path, mesh.vertices, mesh.n_verts,
                      mesh.triangles, mesh.n_tris) == 0)
            fprintf(stderr, "Wrote %s\n", save_mesh_path);
        hip_trellis2_shape_dec_mesh_free(&mesh);
        free(feats); free(coords);
        hip_trellis2_free(r);
        return 0;
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
    if (mode_mesh) {
        if (!mode_full) {
            fprintf(stderr, "Error: --mesh requires --full\n"); return 1;
        }
        if (!slat_path || !shape_dec_path || !manifest_path || !save_mesh_path) {
            fprintf(stderr, "Error: --mesh requires --slat-dit, --shape-dec, "
                            "--manifest, --save-mesh\n"); return 1;
        }
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
            /* Reference noise dump is [1,8,16,16,16] CDHW; the runner expects
             * [N_TOK=4096, IN_CH=8] DHWC tok-major. Transpose if needed.
             * Accept either [1,C,D,H,W] (5D, batch=1), [C,D,H,W] (4D) — both CDHW —
             * or the already-flat [N_TOK,IN_CH] / [4096,8] tok-major shape. */
            int is_cdhw = (ndim == 5 && dims[0] == 1 && dims[1] == IN_CH &&
                           dims[2] * dims[3] * dims[4] == N_TOK) ||
                          (ndim == 4 && dims[0] == IN_CH &&
                           dims[1] * dims[2] * dims[3] == N_TOK);
            int is_flat = (ndim == 2 && dims[0] == N_TOK && dims[1] == IN_CH);
            if (is_cdhw) {
                int D, H, W;
                if (ndim == 5) { D = dims[2]; H = dims[3]; W = dims[4]; }
                else           { D = dims[1]; H = dims[2]; W = dims[3]; }
                for (int d = 0; d < D; d++)
                    for (int h = 0; h < H; h++)
                        for (int w = 0; w < W; w++)
                            for (int c = 0; c < IN_CH; c++)
                                noise[((d*H + h)*W + w)*IN_CH + c] =
                                    loaded[((c*D + d)*H + h)*W + w];
                fprintf(stderr, "noise: CDHW [%d,%d,%d,%d] -> DHWC tok-major [%d,%d]\n",
                        IN_CH, D, H, W, N_TOK, IN_CH);
            } else if (is_flat) {
                memcpy(noise, loaded, (size_t)N_TOK * IN_CH * sizeof(float));
            } else {
                fprintf(stderr, "noise: unexpected shape ndim=%d dims=[%d,%d,%d,%d,%d]\n",
                        ndim, dims[0], dims[1], dims[2], dims[3], dims[4]);
                free(loaded); free(noise); free(features);
                hip_trellis2_free(r); return 1;
            }
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
        free(x); free(vel);

        /* ============================================================== */
        /* End-to-end: occupancy -> SLAT sampler -> shape_dec -> mesh.obj */
        /* ============================================================== */
        if (mode_mesh) {
            fprintf(stderr, "\n=== Stage 2: SLAT DiT + shape_dec -> mesh ===\n");

            /* Pipeline post-process: occ_64 = (logits > 0); occ_32 = max_pool3d(2,2)
             * > 0.5; coords = argwhere(occ_32) at resolution 32. SLAT DiT was
             * trained on coords in [0,32). */
            uint8_t *occ32 = (uint8_t *)calloc(32*32*32, 1);
            for (int z = 0; z < 64; z++)
                for (int y = 0; y < 64; y++)
                    for (int xi = 0; xi < 64; xi++)
                        if (occupancy[z*64*64 + y*64 + xi] > 0.0f) {
                            occ32[(z>>1)*32*32 + (y>>1)*32 + (xi>>1)] = 1;
                        }
            int N_sparse = 0;
            for (int i = 0; i < 32*32*32; i++) if (occ32[i]) N_sparse++;
            if (N_sparse <= 0) {
                fprintf(stderr, "  No occupied voxels — aborting mesh path\n");
                free(occ32); free(occupancy); goto cleanup;
            }
            int32_t *sparse_coords = (int32_t *)malloc((size_t)N_sparse * 4 * sizeof(int32_t));
            int idx = 0;
            /* dump_rocm uses argwhere() which iterates in (b,c,z,y,x) C order.
             * After the [:, [0,2,3,4]] selection that becomes (b,z,y,x). Match. */
            for (int z = 0; z < 32; z++)
                for (int y = 0; y < 32; y++)
                    for (int xi = 0; xi < 32; xi++) {
                        if (occ32[z*32*32 + y*32 + xi]) {
                            sparse_coords[idx*4+0] = 0;
                            sparse_coords[idx*4+1] = z;
                            sparse_coords[idx*4+2] = y;
                            sparse_coords[idx*4+3] = xi;
                            idx++;
                        }
                    }
            fprintf(stderr, "  Sparse voxels (32^3 max-pool): %d\n", N_sparse);
            free(occ32); free(occupancy);

            /* Free SS DiT + SS decoder before loading SLAT — SLAT's 1.3B F32
             * weights (~5 GB) on top of SS DiT (~5 GB) + decoder pushes a
             * 16 GB GPU into HMM page-fault thrashing. Unloading drops peak
             * VRAM from ~17 GB to ~11 GB, eliminating the swap that made
             * SLAT scalar attention ~600× slower than its compute bound. */
            fprintf(stderr, "  Freeing SS DiT + decoder to free VRAM for SLAT...\n");
            hip_trellis2_unload_dit(r);
            hip_trellis2_unload_decoder(r);

            /* Load SLAT DiT + shape_dec. */
            fprintf(stderr, "  Loading SLAT DiT: %s\n", slat_path);
            if (hip_trellis2_load_slat_dit(r, slat_path) != 0) {
                fprintf(stderr, "  SLAT DiT load failed\n");
                free(sparse_coords); goto cleanup;
            }
            fprintf(stderr, "  Loading shape_dec: %s\n", shape_dec_path);
            if (hip_trellis2_load_shape_dec(r, shape_dec_path) != 0) {
                fprintf(stderr, "  shape_dec load failed\n");
                free(sparse_coords); goto cleanup;
            }

            /* Parse normalization constants. */
            float slat_mean[32], slat_std[32];
            if (parse_shape_norm(manifest_path, slat_mean, slat_std) != 0) {
                fprintf(stderr, "  Failed to parse shape_slat_normalization "
                                "from %s\n", manifest_path);
                free(sparse_coords); goto cleanup;
            }
            fprintf(stderr, "  slat_norm: mean[0..3]=[%.4f %.4f %.4f] std[0..3]=[%.4f %.4f %.4f]\n",
                    slat_mean[0], slat_mean[1], slat_mean[2],
                    slat_std[0], slat_std[1], slat_std[2]);

            /* Initial sparse noise [N, 32], seed+1 to match cuda convention. */
            const int SLAT_CH = 32;
            float *s2_x = (float *)malloc((size_t)N_sparse * SLAT_CH * sizeof(float));
            rng_seed(seed + 1);
            for (int i = 0; i < N_sparse * SLAT_CH; i++) s2_x[i] = randn();

            /* CFG zero-cond buffer. */
            float *neg_cond = (float *)calloc((size_t)N_COND * COND_DIM, sizeof(float));
            float *v_cond   = (float *)malloc((size_t)N_sparse * SLAT_CH * sizeof(float));
            float *v_uncond = (float *)malloc((size_t)N_sparse * SLAT_CH * sizeof(float));

            const float s2_rescale_t = 3.0f;
            const float s2_cfg       = 7.5f;
            const float s2_rescale   = 0.7f;
            const float s2_sigma_min = 1e-5f;

            double t_total = 0;
            for (int step = 0; step < slat_steps; step++) {
                float t_lin_a = 1.0f - (float)step       / (float)slat_steps;
                float t_lin_b = 1.0f - (float)(step + 1) / (float)slat_steps;
                float t_cur  = s2_rescale_t * t_lin_a / (1.0f + (s2_rescale_t-1.0f)*t_lin_a);
                float t_next = s2_rescale_t * t_lin_b / (1.0f + (s2_rescale_t-1.0f)*t_lin_b);
                float dt     = t_cur - t_next;
                int use_cfg  = (!no_cfg) && (t_cur >= 0.6f && t_cur <= 1.0f);

                double t0 = now_ms();
                hip_trellis2_invalidate_slat_kv(r);
                if (hip_trellis2_slat_dit_step(r, s2_x, sparse_coords, N_sparse,
                                               t_cur, features, N_COND, v_cond) != 0) {
                    fprintf(stderr, "  SLAT step %d cond failed\n", step);
                    free(s2_x); free(neg_cond); free(v_cond); free(v_uncond);
                    free(sparse_coords); goto cleanup;
                }
                if (use_cfg) {
                    hip_trellis2_invalidate_slat_kv(r);
                    if (hip_trellis2_slat_dit_step(r, s2_x, sparse_coords, N_sparse,
                                                   t_cur, neg_cond, N_COND, v_uncond) != 0) {
                        fprintf(stderr, "  SLAT step %d uncond failed\n", step);
                        free(s2_x); free(neg_cond); free(v_cond); free(v_uncond);
                        free(sparse_coords); goto cleanup;
                    }
                }
                double step_ms = now_ms() - t0;
                t_total += step_ms;

                int n = N_sparse * SLAT_CH;
                if (use_cfg) {
                    /* CFG combine + rescale (mirrors cuda/test_cuda_trellis2). */
                    float coeff = s2_sigma_min + (1.0f - s2_sigma_min) * t_cur;
                    float one_m_sm = 1.0f - s2_sigma_min;
                    double sum_pos = 0, sum2_pos = 0, sum_cfg = 0, sum2_cfg = 0;
                    for (int i = 0; i < n; i++) {
                        float v_cfg = s2_cfg * v_cond[i] + (1.0f - s2_cfg) * v_uncond[i];
                        v_uncond[i] = v_cfg; /* repurpose as pred_v */
                        float x0p = one_m_sm * s2_x[i] - coeff * v_cond[i];
                        float x0c = one_m_sm * s2_x[i] - coeff * v_cfg;
                        sum_pos += x0p; sum2_pos += (double)x0p * x0p;
                        sum_cfg += x0c; sum2_cfg += (double)x0c * x0c;
                    }
                    double n_d = (double)n;
                    double std_pos = sqrt((sum2_pos - sum_pos*sum_pos/n_d) / (n_d - 1.0));
                    double std_cfg_v = sqrt((sum2_cfg - sum_cfg*sum_cfg/n_d) / (n_d - 1.0));
                    float ratio = (std_cfg_v > 1e-8) ? (float)(std_pos / std_cfg_v) : 1.0f;
                    float sc = s2_rescale * ratio + (1.0f - s2_rescale);
                    for (int i = 0; i < n; i++) {
                        float x0c = one_m_sm * s2_x[i] - coeff * v_uncond[i];
                        float pred = (one_m_sm * s2_x[i] - sc * x0c) / coeff;
                        s2_x[i] -= dt * pred;
                    }
                } else {
                    for (int i = 0; i < n; i++) s2_x[i] -= dt * v_cond[i];
                }
                fprintf(stderr, "  step %02d/%02d t=%.4f->%.4f %s %.1f ms\n",
                        step+1, slat_steps, t_cur, t_next,
                        use_cfg ? "CFG" : "noG", step_ms);
            }
            fprintf(stderr, "  SLAT total: %.1f ms (%.1f ms/step)\n",
                    t_total, t_total / slat_steps);
            free(neg_cond); free(v_cond); free(v_uncond);

            /* Denormalize: x = x * std + mean. */
            for (int i = 0; i < N_sparse; i++)
                for (int c = 0; c < SLAT_CH; c++)
                    s2_x[i*SLAT_CH + c] = s2_x[i*SLAT_CH + c] * slat_std[c] + slat_mean[c];
            print_stats("denorm slat", s2_x, N_sparse * SLAT_CH);

            /* Dump denorm feats + coords so shape_dec can be bisected via
             * --shape-only on these exact inputs. */
            {
                char outpath[1024]; int d2[2] = { N_sparse, SLAT_CH };
                snprintf(outpath, sizeof(outpath), "%s/hip_shape_denorm_feats.npy", output_dir);
                write_npy_f32(outpath, s2_x, d2, 2);
                fprintf(stderr, "  Wrote denorm feats -> %s\n", outpath);
                /* coords are int32 [N,4] — write npy manually */
                snprintf(outpath, sizeof(outpath), "%s/hip_shape_coords.npy", output_dir);
                FILE *f = fopen(outpath, "wb");
                if (f) {
                    fwrite("\x93NUMPY", 1, 6, f);
                    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
                    char body[256]; int n = snprintf(body, sizeof(body),
                        "{'descr': '<i4', 'fortran_order': False, 'shape': (%d, 4), }", N_sparse);
                    int total = 10 + n + 1;
                    int pad = ((total + 63) / 64) * 64 - total;
                    uint16_t hl = (uint16_t)(n + pad + 1);
                    fwrite(&hl, 2, 1, f);
                    fwrite(body, 1, (size_t)n, f);
                    for (int p = 0; p < pad; p++) fputc(' ', f);
                    fputc('\n', f);
                    fwrite(sparse_coords, sizeof(int32_t), (size_t)N_sparse * 4, f);
                    fclose(f);
                    fprintf(stderr, "  Wrote coords -> %s\n", outpath);
                }
            }

            /* Run shape_dec -> mesh. */
            fprintf(stderr, "  Running shape_dec...\n");
            double t_dec = now_ms();
            hip_trellis2_mesh mesh = {0};
            if (hip_trellis2_run_shape_dec(r, s2_x, sparse_coords,
                                            N_sparse, SLAT_CH, &mesh) != 0) {
                fprintf(stderr, "  shape_dec failed\n");
                free(s2_x); free(sparse_coords); goto cleanup;
            }
            fprintf(stderr, "  shape_dec: %.1f ms, mesh: %d verts, %d tris\n",
                    now_ms() - t_dec, mesh.n_verts, mesh.n_tris);

            if (write_obj(save_mesh_path, mesh.vertices, mesh.n_verts,
                          mesh.triangles, mesh.n_tris) == 0) {
                fprintf(stderr, "  Wrote mesh -> %s\n", save_mesh_path);
            }

            /* ============================================================ */
            /* Stage 3 — tex DiT + tex_dec → PBR textured OBJ                */
            /* ============================================================ */
            if (mode_tex) {
                if (!tex_dit_path || !tex_dec_path) {
                    fprintf(stderr, "  --tex requires --tex-dit and --tex-dec\n");
                } else {
                    fprintf(stderr, "\n=== Stage 3: Tex DiT + tex_dec -> textured OBJ ===\n");

                    /* tex_slat_normalization (separate from shape's). */
                    float tex_mean[32], tex_std[32];
                    if (parse_slat_norm(manifest_path, "tex_slat_normalization",
                                        tex_mean, tex_std) != 0) {
                        fprintf(stderr, "  Failed to parse tex_slat_normalization from %s\n",
                                manifest_path);
                        goto tex_done;
                    }

                    /* shape_for_tex = (denorm_slat - mean) / std (i.e., recover the
                     * raw normalized shape latent that tex DiT was trained on). */
                    float *shape_for_tex = (float *)malloc((size_t)N_sparse * 32 * sizeof(float));
                    for (int i = 0; i < N_sparse; i++)
                        for (int c = 0; c < 32; c++)
                            shape_for_tex[i*32 + c] =
                                (s2_x[i*32 + c] - slat_mean[c]) / slat_std[c];

                    /* Free SS DiT + SS decoder + shape SLAT DiT weights to make
                     * room for tex DiT + tex_dec. On a 16 GB GPU, leaving all of
                     * stage-1/2 resident OOMs the tex_dec forward scratch (kernel
                     * launch errors with hipErrorIllegalAddress). */
                    fprintf(stderr, "  Freeing SS DiT + SS decoder + shape SLAT DiT to make room...\n");
                    hip_trellis2_unload_dit(r);
                    hip_trellis2_unload_decoder(r);
                    hip_trellis2_unload_slat_dit(r);

                    /* Load tex DiT. */
                    fprintf(stderr, "  Loading tex DiT: %s\n", tex_dit_path);
                    if (hip_trellis2_load_tex_dit(r, tex_dit_path) != 0) {
                        fprintf(stderr, "  tex DiT load failed\n");
                        free(shape_for_tex); goto tex_done;
                    }

                    /* tex_x = cat(noise[32], shape_for_tex[32]) per voxel — only the
                     * noise half evolves; the shape half stays as fixed conditioning. */
                    float *tex_x = (float *)malloc((size_t)N_sparse * 64 * sizeof(float));
                    rng_seed(seed + 2);
                    for (int i = 0; i < N_sparse; i++) {
                        for (int c = 0; c < 32; c++) tex_x[i*64 + c]      = randn();
                        for (int c = 0; c < 32; c++) tex_x[i*64 + 32 + c] = shape_for_tex[i*32 + c];
                    }

                    /* Tex SLAT sampler: pipeline.json gives guidance_strength=1.0,
                     * guidance_rescale=0.0, rescale_t=3.0 — at strength=1.0 the CFG
                     * combine collapses to v_cond, so the uncond pass is skipped. */
                    const float t3_rescale_t = 3.0f;
                    const float t3_sigma_min = 1e-5f;

                    float *v_tex = (float *)malloc((size_t)N_sparse * 32 * sizeof(float));
                    double t_total_tex = 0;
                    for (int step = 0; step < tex_steps; step++) {
                        float ta = 1.0f - (float)step       / (float)tex_steps;
                        float tb = 1.0f - (float)(step + 1) / (float)tex_steps;
                        float t_cur  = t3_rescale_t * ta / (1.0f + (t3_rescale_t-1.0f)*ta);
                        float t_next = t3_rescale_t * tb / (1.0f + (t3_rescale_t-1.0f)*tb);
                        float dt     = t_cur - t_next;

                        double t0 = now_ms();
                        hip_trellis2_invalidate_tex_kv(r);
                        if (hip_trellis2_tex_dit_step(r, tex_x, sparse_coords, N_sparse,
                                                      t_cur, features, N_COND, v_tex) != 0) {
                            fprintf(stderr, "  tex step %d failed\n", step);
                            free(v_tex); free(tex_x); free(shape_for_tex);
                            goto tex_done;
                        }
                        double step_ms = now_ms() - t0;
                        t_total_tex += step_ms;

                        /* Euler step on noise half only; refresh shape half (already
                         * equal to shape_for_tex; no-op but keeps invariant explicit). */
                        for (int i = 0; i < N_sparse; i++) {
                            for (int c = 0; c < 32; c++)
                                tex_x[i*64 + c] -= dt * v_tex[i*32 + c];
                        }
                        (void)t3_sigma_min;
                        fprintf(stderr, "  tex step %02d/%02d t=%.4f->%.4f %.1f ms\n",
                                step+1, tex_steps, t_cur, t_next, step_ms);
                    }
                    fprintf(stderr, "  Tex SLAT total: %.1f ms (%.1f ms/step)\n",
                            t_total_tex, t_total_tex / tex_steps);
                    free(v_tex); free(shape_for_tex);

                    /* Denormalize tex SLat: tex = noise_half * std + mean. */
                    float *tex_slat = (float *)malloc((size_t)N_sparse * 32 * sizeof(float));
                    for (int i = 0; i < N_sparse; i++)
                        for (int c = 0; c < 32; c++)
                            tex_slat[i*32 + c] = tex_x[i*64 + c] * tex_std[c] + tex_mean[c];
                    free(tex_x);
                    print_stats("denorm tex_slat", tex_slat, N_sparse * 32);

                    /* Unload tex DiT, load tex_dec. */
                    fprintf(stderr, "  Loading tex_dec: %s\n", tex_dec_path);
                    if (hip_trellis2_load_tex_dec(r, tex_dec_path) != 0) {
                        fprintf(stderr, "  tex_dec load failed\n");
                        free(tex_slat); goto tex_done;
                    }

                    /* Run tex_dec → [N_dense, 6] feats + dense coords. */
                    double t_tdec = now_ms();
                    float *tex_feats = NULL; int32_t *tex_coords = NULL; int N_tex = 0;
                    if (hip_trellis2_run_tex_dec(r, tex_slat, sparse_coords,
                                                  N_sparse, 32,
                                                  &tex_feats, &tex_coords, &N_tex) != 0) {
                        fprintf(stderr, "  tex_dec forward failed\n");
                        free(tex_slat); goto tex_done;
                    }
                    fprintf(stderr, "  tex_dec: %.1f ms, N_tex=%d\n",
                            now_ms() - t_tdec, N_tex);
                    free(tex_slat);

                    /* DBG: minimal tex_coords zero-count check */
                    {
                        int zeroes = 0;
                        for (int i = 0; i < N_tex; i++) if (tex_coords[i*4+1]==0 && tex_coords[i*4+2]==0 && tex_coords[i*4+3]==0) zeroes++;
                        if (zeroes > N_tex/2)
                            fprintf(stderr, "  WARN: %d/%d voxels have coord=(0,0,0) — fast-path corruption\n", zeroes, N_tex);
                    }

                    /* PBR baking. */
                    int max_c = 0;
                    for (int i = 0; i < N_tex; i++)
                        for (int j = 1; j <= 3; j++)
                            if (tex_coords[i*4+j] > max_c) max_c = tex_coords[i*4+j];
                    int pbr_res = max_c + 1;
                    fprintf(stderr, "  PBR baking: voxel_res=%d, tex_res=%d\n",
                            pbr_res, tex_res); fflush(stderr);

                    t2_pbr_field pbr = t2_pbr_from_decoder(tex_feats, tex_coords,
                                                          N_tex, pbr_res);
                    fprintf(stderr, "  pbr field built (N=%d res=%d hash_cap=%d)\n",
                            pbr.N, pbr.resolution, pbr.hash_cap); fflush(stderr);
                    t2_pbr_attr *colors = (t2_pbr_attr *)malloc(
                        (size_t)mesh.n_verts * sizeof(t2_pbr_attr));
                    /* hip_trellis2_run_shape_dec() applies a CPU-axis X/Z swap on
                     * mesh.vertices before returning. tex_coords are still in the
                     * native voxel order, so undo the swap into a temp buffer for
                     * the PBR sampler. */
                    float *pbr_verts = (float *)malloc((size_t)mesh.n_verts * 3 * sizeof(float));
                    for (int vi = 0; vi < mesh.n_verts; vi++) {
                        pbr_verts[vi*3+0] = mesh.vertices[vi*3+2];
                        pbr_verts[vi*3+1] = mesh.vertices[vi*3+1];
                        pbr_verts[vi*3+2] = mesh.vertices[vi*3+0];
                    }
                    t2_pbr_sample_vertices(&pbr, pbr_verts, mesh.n_verts, colors);
                    free(pbr_verts);
                    fprintf(stderr, "  sampled %d verts; colors[0]=(%.3f,%.3f,%.3f) mr=(%.3f,%.3f)\n",
                            mesh.n_verts, colors[0].r, colors[0].g, colors[0].b,
                            colors[0].metallic, colors[0].roughness); fflush(stderr);

                    /* Strip extension from save_mesh_path for base. */
                    char base[1024];
                    snprintf(base, sizeof(base), "%s", save_mesh_path);
                    char *dot = strrchr(base, '.');
                    if (dot) *dot = '\0';
                    /* Use vertex-colored OBJ (per-vertex RGB) — much simpler than
                     * the textured atlas path, which is O(n_charts*n_tris) and brittle
                     * for sparse-voxel meshes. Vertex colors are enough to verify the
                     * tex_dec pipeline produces sensible output. */
                    char vc_path[1100];
                    snprintf(vc_path, sizeof(vc_path), "%s_colored.obj", base);
                    fprintf(stderr, "  writing vertex-colored OBJ -> %s\n", vc_path); fflush(stderr);
                    t2_pbr_write_colored_obj(vc_path, mesh.vertices, mesh.triangles,
                                             mesh.n_verts, mesh.n_tris, colors);

                    free(colors);
                    t2_pbr_free(&pbr);
                    free(tex_feats); free(tex_coords);
                    hip_trellis2_unload_tex_dec(r);
                }
            }
            tex_done:
            hip_trellis2_shape_dec_mesh_free(&mesh);
            free(s2_x); free(sparse_coords);
        } else {
            free(occupancy);
        }
    }

cleanup:
    if (features) free(features);
    if (noise)    free(noise);
    hip_trellis2_free(r);
    return 0;
}
