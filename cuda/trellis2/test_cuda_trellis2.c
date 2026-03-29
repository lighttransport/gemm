/*
 * test_cuda_trellis2.c - Test CUDA TRELLIS.2 pipeline
 *
 * Stage 1 only:
 *   ./test_cuda_trellis2 <stage1.st> <decoder.st> <features.npy> [options]
 *
 * Stage 1 + Stage 2 (shape generation):
 *   ./test_cuda_trellis2 <stage1.st> <decoder.st> <features.npy> --stage2 <stage2.st> \
 *       --shape-dec <shape_dec.st> [options]
 *
 * Runs Stage 1 DiT + decoder on GPU. If --stage2 is given, also runs Stage 2
 * flow sampling on GPU + shape decoder on CPU + FDG mesh extraction.
 */

#include "cuda_trellis2_runner.h"
#define MARCHING_CUBES_IMPLEMENTATION
#include "../../common/marching_cubes.h"

/* Shape decoder + FDG mesh (for Stage 2).
 * safetensors.h is already compiled in cuda_trellis2_runner.c,
 * so we only include headers here, not implementations. */
/* ggml_dequant, safetensors, sparse3d are compiled in cuda_trellis2_runner.c */
#include "../../common/ggml_dequant.h"
#include "../../common/safetensors.h"
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"
#define T2_FDG_MESH_IMPLEMENTATION
#include "../../common/trellis2_fdg_mesh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ---- .npy reader ---- */
static float *read_npy_f32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot read %s\n", path); return NULL; }
    fseek(f, 8, SEEK_SET);
    uint16_t header_len;
    if (fread(&header_len, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *header = (char *)malloc(header_len + 1);
    if (fread(header, 1, header_len, f) != header_len) { free(header); fclose(f); return NULL; }
    header[header_len] = '\0';
    *ndim = 0;
    char *sp = strstr(header, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') { while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break; dims[*ndim] = (int)strtol(sp, &sp, 10); (*ndim)++; if (*ndim >= 8) break; }}}
    size_t n = 1; for (int i = 0; i < *ndim; i++) n *= (size_t)dims[i];
    float *data = (float *)malloc(n * sizeof(float));
    fread(data, sizeof(float), n, f);
    fclose(f); free(header);
    fprintf(stderr, "Read %s: (", path);
    for (int i = 0; i < *ndim; i++) fprintf(stderr, "%s%d", i?",":"", dims[i]);
    fprintf(stderr, ") = %zu elements\n", n);
    return data;
}

/* ---- .npy writer ---- */
static void write_npy_f32(const char *path, const float *data, const int *dims, int ndim) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char shape[256]; int sl = 0;
    sl += snprintf(shape+sl, sizeof(shape)-sl, "(");
    size_t n = 1;
    for (int i = 0; i < ndim; i++) { sl += snprintf(shape+sl, sizeof(shape)-sl, "%d,", dims[i]); n *= dims[i]; }
    sl += snprintf(shape+sl, sizeof(shape)-sl, ")");
    char hdr[512];
    int hl = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape);
    int total = 10 + hl + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t hlen = (uint16_t)(hl + pad + 1);
    fwrite(&hlen, 2, 1, f); fwrite(hdr, 1, hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f); fputc('\n', f);
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "Wrote %s\n", path);
}

/* Simple xoshiro256** */
static uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
typedef struct { uint64_t s[4]; } rng_state;
static uint64_t rng_next(rng_state *r) {
    uint64_t *s = r->s; uint64_t result = rotl64(s[1]*5,7)*9;
    uint64_t t = s[1]<<17; s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=t; s[3]=rotl64(s[3],45);
    return result;
}
static float rng_randn(rng_state *r) {
    double u1 = ((double)(rng_next(r)>>11)+0.5)/(double)(1ULL<<53);
    double u2 = ((double)(rng_next(r)>>11)+0.5)/(double)(1ULL<<53);
    return (float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2));
}
static float rescale_t(float t, float rt) { return t*rt/(1.0f+(rt-1.0f)*t); }

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <stage1.st> <decoder.st> <features.npy> [options]\n\n"
                "Options:\n"
                "  -s <seed>        Random seed (default: 42)\n"
                "  -n <steps>       Euler steps (default: 12)\n"
                "  -g <cfg_scale>   CFG guidance scale (default: 7.5)\n"
                "  -o <output.obj>  Output mesh path (default: output.obj)\n"
                "  --grid <N>       Marching cubes grid resolution (default: 64, try 32 for lighter mesh)\n"
                "  --threshold <t>  Occupancy threshold (default: 0.0)\n"
                "  --npy <path>     Also save latent as .npy\n"
                "  --noise <path>   Load noise from .npy instead of generating\n"
                "  --occ <path>     Save occupancy grid as .npy\n"
                "\nStage 2 (shape generation):\n"
                "  --stage2 <path>      Stage 2 flow model .safetensors\n"
                "  --shape-dec <path>   Shape decoder .safetensors\n"
                "  --s2-steps <N>       Stage 2 Euler steps (default: 12)\n"
                "  --s2-cfg <scale>     Stage 2 CFG scale (default: 7.5)\n"
                "  --s2-npy <path>      Save Stage 2 latent as .npy\n"
                "  -t <threads>         CPU threads for shape decoder (default: 4)\n"
                , argv[0]);
        return 1;
    }
    const char *stage1_path = argv[1];
    const char *decoder_path = argv[2];
    const char *features_path = argv[3];
    uint32_t seed = 42;
    int n_steps = 12;
    float cfg_scale = 7.5f;
    float rescale = 5.0f;
    const char *obj_path = "output.obj";
    const char *npy_path = NULL;
    int mc_grid = 64;
    float mc_threshold = 0.0f;
    const char *noise_path = NULL;
    const char *occ_npy_path = NULL;
    const char *stage2_path = NULL;
    const char *shape_dec_path = NULL;
    int s2_steps = 12;
    float s2_cfg = 7.5f;
    const char *s2_npy_path = NULL;
    int n_threads = 4;

    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "-s") && i+1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g") && i+1 < argc) cfg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i+1 < argc) obj_path = argv[++i];
        else if (!strcmp(argv[i], "--npy") && i+1 < argc) npy_path = argv[++i];
        else if (!strcmp(argv[i], "--occ") && i+1 < argc) occ_npy_path = argv[++i];
        else if (!strcmp(argv[i], "--grid") && i+1 < argc) mc_grid = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) mc_threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--noise") && i+1 < argc) noise_path = argv[++i];
        else if (!strcmp(argv[i], "--stage2") && i+1 < argc) stage2_path = argv[++i];
        else if (!strcmp(argv[i], "--shape-dec") && i+1 < argc) shape_dec_path = argv[++i];
        else if (!strcmp(argv[i], "--s2-steps") && i+1 < argc) s2_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--s2-cfg") && i+1 < argc) s2_cfg = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--s2-npy") && i+1 < argc) s2_npy_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
    }

    /* Load features */
    int ndim, dims[8];
    float *features = read_npy_f32(features_path, &ndim, dims);
    if (!features) return 1;
    int n_cond = dims[0];
    fprintf(stderr, "Conditioning: %d tokens, %d dim\n", n_cond, ndim >= 2 ? dims[1] : 0);

    /* Init CUDA runner */
    fprintf(stderr, "\n=== Initializing CUDA runner ===\n");
    cuda_trellis2_runner *r = cuda_trellis2_init(0, 1);
    if (!r) { free(features); return 1; }

    /* Load weights (DINOv3 skipped — we already have features) */
    fprintf(stderr, "\n=== Loading weights ===\n");
    if (cuda_trellis2_load_weights(r, NULL, stage1_path, decoder_path) != 0) {
        cuda_trellis2_free(r); free(features); return 1;
    }
    if (stage2_path) {
        if (cuda_trellis2_load_stage2(r, stage2_path) != 0) {
            cuda_trellis2_free(r); free(features); return 1;
        }
    }

    /* Generate or load initial noise */
    int N = 4096, C = 8;
    float *x;
    if (noise_path) {
        int nd2, dd2[8];
        x = read_npy_f32(noise_path, &nd2, dd2);
        if (!x) { cuda_trellis2_free(r); return 1; }
        fprintf(stderr, "Loaded noise from %s\n", noise_path);
    } else {
        x = (float *)malloc((size_t)N * C * sizeof(float));
        rng_state rng = {{seed, seed ^ 0x9E3779B97F4A7C15ULL,
                           seed ^ 0x6C62272E07BB0142ULL, seed ^ 0xBF58476D1CE4E5B9ULL}};
        for (int i = 0; i < 8; i++) rng_next(&rng);
        for (int i = 0; i < N * C; i++) x[i] = rng_randn(&rng);
    }

    /* Upload conditioning features */
    fprintf(stderr, "\n=== Stage 1: Flow Sampling (%d steps, cfg=%.1f) ===\n",
            n_steps, cfg_scale);

    float *v_cond = (float *)malloc((size_t)N * C * sizeof(float));
    float *v_uncond = (float *)malloc((size_t)N * C * sizeof(float));
    float *zeros_cond = (float *)calloc((size_t)n_cond * 1024, sizeof(float));

    struct timespec t0_ts; clock_gettime(CLOCK_MONOTONIC, &t0_ts);
    double t0 = t0_ts.tv_sec * 1000.0 + t0_ts.tv_nsec / 1e6;

    for (int step = 0; step < n_steps; step++) {
        float t_start = 1.0f - (float)step / (float)n_steps;
        float t_end = 1.0f - (float)(step + 1) / (float)n_steps;
        float t_cur = rescale_t(t_start, rescale);
        float t_next = rescale_t(t_end, rescale);
        float dt = t_next - t_cur;

        struct timespec step_ts; clock_gettime(CLOCK_MONOTONIC, &step_ts);
        double step_t0 = step_ts.tv_sec * 1000.0 + step_ts.tv_nsec / 1e6;

        /* Guidance interval: only apply CFG when t_cur in [0.6, 1.0] */
        int apply_cfg = (t_cur >= 0.6f && t_cur <= 1.0f && cfg_scale != 1.0f);
        float sigma_min = 1e-5f;
        float cfg_rescale_val = 0.7f;

        if (apply_cfg) {
            cuda_trellis2_run_dit(r, x, t_cur, features, v_cond);
            cuda_trellis2_run_dit(r, x, t_cur, zeros_cond, v_uncond);

            /* Save original v_cond for CFG rescale (x0_pos uses cond prediction) */
            /* CFG: pred_v = cfg * v_cond + (1-cfg) * v_uncond */
            float *pred_v = v_uncond;  /* reuse v_uncond buffer for combined */
            for (int i = 0; i < N * C; i++)
                pred_v[i] = cfg_scale * v_cond[i] + (1.0f - cfg_scale) * v_uncond[i];

            /* CFG rescale: match official _pred_to_xstart / std / _xstart_to_pred */
            if (cfg_rescale_val > 0.0f) {
                float sm = sigma_min;
                float tc = sm + (1.0f - sm) * t_cur;  /* t_cur is already rescaled */
                float one_m_sm = 1.0f - sm;
                /* x0_pos = (1-sm)*x - tc * v_cond */
                /* x0_cfg = (1-sm)*x - tc * pred_v */
                /* Compute std (torch.std: mean-subtracted, Bessel corrected) */
                double sum_pos = 0, sum_cfg = 0;
                double sum2_pos = 0, sum2_cfg = 0;
                for (int i = 0; i < N * C; i++) {
                    float x0p = one_m_sm * x[i] - tc * v_cond[i];
                    float x0c = one_m_sm * x[i] - tc * pred_v[i];
                    sum_pos += x0p; sum2_pos += (double)x0p * x0p;
                    sum_cfg += x0c; sum2_cfg += (double)x0c * x0c;
                }
                double n = (double)(N * C);
                double std_pos = sqrt((sum2_pos - sum_pos*sum_pos/n) / (n - 1.0));
                double std_cfg = sqrt((sum2_cfg - sum_cfg*sum_cfg/n) / (n - 1.0));
                float ratio = (std_cfg > 1e-8) ? (float)(std_pos / std_cfg) : 1.0f;
                float sc = cfg_rescale_val * ratio + (1.0f - cfg_rescale_val);
                /* x0_rescaled = x0_cfg * (std_pos/std_cfg) */
                /* x0_final = cfg_rescale * x0_rescaled + (1-cfg_rescale) * x0_cfg = sc * x0_cfg */
                /* pred_final = ((1-sm)*x - x0_final) / tc */
                for (int i = 0; i < N * C; i++) {
                    float x0c = one_m_sm * x[i] - tc * pred_v[i];
                    pred_v[i] = (one_m_sm * x[i] - sc * x0c) / tc;
                }
            }

            /* Euler step: x -= (t_cur - t_next) * pred_v */
            for (int i = 0; i < N * C; i++)
                x[i] -= (t_cur - t_next) * pred_v[i];
        } else {
            /* No CFG — conditioned only */
            cuda_trellis2_run_dit(r, x, t_cur, features, v_cond);
            /* Debug: check for NaN/explosion */
            { float mx = 0; for (int i = 0; i < N*C; i++) { float a = v_cond[i]; if (a!=a || (a>1e10f || a<-1e10f)) { fprintf(stderr, "  WARNING: v_cond[%d]=%.4g\n", i, a); break; } if (a>mx || a<-mx) mx = (a>0?a:-a); } fprintf(stderr, "  v_cond max_abs=%.4f x[:4]=%.4f %.4f %.4f %.4f\n", mx, x[0],x[1],x[2],x[3]); }
            for (int i = 0; i < N * C; i++)
                x[i] -= (t_cur - t_next) * v_cond[i];
        }

        clock_gettime(CLOCK_MONOTONIC, &step_ts);
        double step_t1 = step_ts.tv_sec * 1000.0 + step_ts.tv_nsec / 1e6;
        fprintf(stderr, "  step %d/%d  t=%.4f->%.4f  %s  %.1f ms\n",
                step+1, n_steps, t_cur, t_next,
                apply_cfg ? "CFG" : "noG", step_t1 - step_t0);
    }

    free(zeros_cond); free(v_cond); free(v_uncond);

    /* Save latent */
    if (npy_path) {
        int d[4] = {C, 16, 16, 16};
        write_npy_f32(npy_path, x, d, 4);
    }

    /* Decode */
    fprintf(stderr, "\n=== Structure Decoder ===\n");
    float *occupancy = (float *)malloc(64 * 64 * 64 * sizeof(float));
    cuda_trellis2_run_decoder(r, x, occupancy);
    free(x);

    struct timespec t1_ts; clock_gettime(CLOCK_MONOTONIC, &t1_ts);
    double t1 = t1_ts.tv_sec * 1000.0 + t1_ts.tv_nsec / 1e6;
    fprintf(stderr, "\nStage 1 GPU time: %.1f s\n", (t1 - t0) / 1000.0);

    /* Occupancy stats */
    float mn = occupancy[0], mx_occ = occupancy[0];
    double sum = 0;
    int occ_count = 0;
    for (int i = 0; i < 64*64*64; i++) {
        if (occupancy[i] < mn) mn = occupancy[i];
        if (occupancy[i] > mx_occ) mx_occ = occupancy[i];
        sum += occupancy[i];
        if (occupancy[i] > 0) occ_count++;
    }
    fprintf(stderr, "Occupancy: min=%.2f max=%.2f mean=%.4f\n", mn, mx_occ, sum/(64*64*64));
    fprintf(stderr, "Occupied (logit>0): %d / %d (%.1f%%)\n",
            occ_count, 64*64*64, 100.0f * occ_count / (64*64*64));

    if (occ_npy_path) {
        int occ_dims[3] = {64, 64, 64};
        write_npy_f32(occ_npy_path, occupancy, occ_dims, 3);
    }

    /* ================================================================ */
    /* Stage 2: Shape generation (if --stage2 provided)                 */
    /* ================================================================ */
    if (stage2_path && shape_dec_path) {
        fprintf(stderr, "\n=== Stage 2: Shape Flow Sampling (%d steps, cfg=%.1f) ===\n",
                s2_steps, s2_cfg);

        /* Extract sparse coords from occupancy */
        int N_sparse = occ_count;
        int32_t *sparse_coords = (int32_t *)malloc((size_t)N_sparse * 4 * sizeof(int32_t));
        int idx = 0;
        for (int z = 0; z < 64; z++)
            for (int y = 0; y < 64; y++)
                for (int xi = 0; xi < 64; xi++)
                    if (occupancy[z * 64 * 64 + y * 64 + xi] > 0.0f) {
                        sparse_coords[idx * 4 + 0] = 0;   /* batch */
                        sparse_coords[idx * 4 + 1] = z;
                        sparse_coords[idx * 4 + 2] = y;
                        sparse_coords[idx * 4 + 3] = xi;
                        idx++;
                    }
        fprintf(stderr, "Sparse voxels: %d\n", N_sparse);

        /* Generate noise [N, 32] for Stage 2 */
        int s2_ch = 32;
        float *s2_x = (float *)malloc((size_t)N_sparse * s2_ch * sizeof(float));
        rng_state s2_rng = {{seed + 1, (seed + 1) ^ 0x9E3779B97F4A7C15ULL,
                             (seed + 1) ^ 0x6C62272E07BB0142ULL, (seed + 1) ^ 0xBF58476D1CE4E5B9ULL}};
        for (int i = 0; i < 8; i++) rng_next(&s2_rng);
        for (int i = 0; i < N_sparse * s2_ch; i++) s2_x[i] = rng_randn(&s2_rng);

        /* Stage 2 Euler sampling with CFG */
        float *s2_v_cond = (float *)malloc((size_t)N_sparse * s2_ch * sizeof(float));
        float *s2_v_uncond = (float *)malloc((size_t)N_sparse * s2_ch * sizeof(float));
        float *s2_zeros_cond = (float *)calloc((size_t)n_cond * 1024, sizeof(float));
        float s2_rescale_t = 3.0f;  /* Stage 2 uses rescale_t=3.0 */
        float s2_sigma_min = 1e-5f;
        float s2_cfg_rescale = 0.7f;

        struct timespec s2_t0_ts; clock_gettime(CLOCK_MONOTONIC, &s2_t0_ts);
        double s2_t0 = s2_t0_ts.tv_sec * 1000.0 + s2_t0_ts.tv_nsec / 1e6;

        for (int step = 0; step < s2_steps; step++) {
            float t_start = 1.0f - (float)step / (float)s2_steps;
            float t_end = 1.0f - (float)(step + 1) / (float)s2_steps;
            float t_cur = rescale_t(t_start, s2_rescale_t);
            float t_next = rescale_t(t_end, s2_rescale_t);

            struct timespec step_ts; clock_gettime(CLOCK_MONOTONIC, &step_ts);
            double step_t0x = step_ts.tv_sec * 1000.0 + step_ts.tv_nsec / 1e6;

            int apply_cfg = (t_cur >= 0.6f && t_cur <= 1.0f && s2_cfg != 1.0f);

            if (apply_cfg) {
                cuda_trellis2_run_stage2_dit(r, s2_x, t_cur, features, sparse_coords, N_sparse, s2_v_cond);
                cuda_trellis2_run_stage2_dit(r, s2_x, t_cur, s2_zeros_cond, sparse_coords, N_sparse, s2_v_uncond);

                /* CFG combine */
                float *pred_v = s2_v_uncond;
                for (int i = 0; i < N_sparse * s2_ch; i++)
                    pred_v[i] = s2_cfg * s2_v_cond[i] + (1.0f - s2_cfg) * s2_v_uncond[i];

                /* CFG rescale */
                if (s2_cfg_rescale > 0.0f) {
                    float sm = s2_sigma_min;
                    float tc = sm + (1.0f - sm) * t_cur;
                    float one_m_sm = 1.0f - sm;
                    double sum_pos = 0, sum_cfg_v = 0, sum2_pos = 0, sum2_cfg_v = 0;
                    for (int i = 0; i < N_sparse * s2_ch; i++) {
                        float x0p = one_m_sm * s2_x[i] - tc * s2_v_cond[i];
                        float x0c = one_m_sm * s2_x[i] - tc * pred_v[i];
                        sum_pos += x0p; sum2_pos += (double)x0p * x0p;
                        sum_cfg_v += x0c; sum2_cfg_v += (double)x0c * x0c;
                    }
                    double n_d = (double)(N_sparse * s2_ch);
                    double std_pos = sqrt((sum2_pos - sum_pos * sum_pos / n_d) / (n_d - 1.0));
                    double std_cfg_v = sqrt((sum2_cfg_v - sum_cfg_v * sum_cfg_v / n_d) / (n_d - 1.0));
                    float ratio = (std_cfg_v > 1e-8) ? (float)(std_pos / std_cfg_v) : 1.0f;
                    float sc = s2_cfg_rescale * ratio + (1.0f - s2_cfg_rescale);
                    for (int i = 0; i < N_sparse * s2_ch; i++) {
                        float x0c = one_m_sm * s2_x[i] - tc * pred_v[i];
                        pred_v[i] = (one_m_sm * s2_x[i] - sc * x0c) / tc;
                    }
                }

                for (int i = 0; i < N_sparse * s2_ch; i++)
                    s2_x[i] -= (t_cur - t_next) * pred_v[i];
            } else {
                cuda_trellis2_run_stage2_dit(r, s2_x, t_cur, features, sparse_coords, N_sparse, s2_v_cond);
                for (int i = 0; i < N_sparse * s2_ch; i++)
                    s2_x[i] -= (t_cur - t_next) * s2_v_cond[i];
            }

            clock_gettime(CLOCK_MONOTONIC, &step_ts);
            double step_t1x = step_ts.tv_sec * 1000.0 + step_ts.tv_nsec / 1e6;
            fprintf(stderr, "  step %d/%d  t=%.4f->%.4f  %s  %.1f ms  std=%.4f\n",
                    step + 1, s2_steps, t_cur, t_next,
                    apply_cfg ? "CFG" : "noG", step_t1x - step_t0x,
                    0.0f /* TODO: compute std */);
        }

        free(s2_v_cond); free(s2_v_uncond); free(s2_zeros_cond);

        struct timespec s2_t1_ts; clock_gettime(CLOCK_MONOTONIC, &s2_t1_ts);
        double s2_t1 = s2_t1_ts.tv_sec * 1000.0 + s2_t1_ts.tv_nsec / 1e6;
        fprintf(stderr, "Stage 2 GPU time: %.1f s\n", (s2_t1 - s2_t0) / 1000.0);

        /* Save Stage 2 latent (raw, before denormalization) */
        if (s2_npy_path) {
            int s2d[2] = {N_sparse, s2_ch};
            write_npy_f32(s2_npy_path, s2_x, s2d, 2);
        }

        /* TODO: Denormalize using pipeline.json shape_slat_normalization mean/std.
         * For now, skip denormalization — the shape decoder should still produce
         * meaningful geometry from the normalized latent. */
        fprintf(stderr, "\n=== Shape Decoder (CPU, %d threads) ===\n", n_threads);

        /* Load shape decoder */
        t2_shape_dec *dec = t2_shape_dec_load(shape_dec_path);
        if (!dec) {
            fprintf(stderr, "Failed to load shape decoder\n");
            free(s2_x); free(sparse_coords); free(occupancy);
            cuda_trellis2_free(r); free(features);
            return 1;
        }

        /* Create sparse tensor for shape decoder */
        sp3d_tensor *slat = sp3d_create(sparse_coords, s2_x, N_sparse, s2_ch, 1);
        free(s2_x); free(sparse_coords);

        /* Run shape decoder */
        t2_shape_dec_result result = t2_shape_dec_forward(dec, slat, n_threads);
        fprintf(stderr, "Shape decoder output: N=%d\n", result.N);

        /* Post-process: sigmoid on vertex offsets, softplus on split_weight */
        for (int i = 0; i < result.N; i++) {
            float *f = result.feats + i * 7;
            for (int j = 0; j < 3; j++)
                f[j] = 1.0f / (1.0f + expf(-f[j]));
            f[6] = logf(1.0f + expf(f[6]));
        }

        /* Extract coords without batch dim: [N, 3] from [N, 4] */
        int32_t *coords3 = (int32_t *)malloc((size_t)result.N * 3 * sizeof(int32_t));
        for (int i = 0; i < result.N; i++) {
            coords3[i * 3 + 0] = result.coords[i * 4 + 1];  /* z */
            coords3[i * 3 + 1] = result.coords[i * 4 + 2];  /* y */
            coords3[i * 3 + 2] = result.coords[i * 4 + 3];  /* x */
        }

        /* FDG mesh extraction */
        float aabb[6] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
        int max_coord = 0;
        for (int i = 0; i < result.N * 3; i++)
            if (coords3[i] > max_coord) max_coord = coords3[i];
        float vs = (aabb[3] - aabb[0]) / (float)(max_coord + 1);

        fprintf(stderr, "\n=== FDG Mesh (voxel_size=%.4f, max_coord=%d) ===\n", vs, max_coord);
        t2_fdg_mesh fdg_mesh = t2_fdg_to_mesh(coords3, result.feats, result.N, vs, aabb);
        free(coords3);

        if (fdg_mesh.n_tris > 0) {
            t2_fdg_write_obj(obj_path, &fdg_mesh);
        }

        t2_fdg_mesh_free(&fdg_mesh);
        t2_shape_dec_result_free(&result);
        sp3d_free(slat);
        t2_shape_dec_free(dec);
    } else {
        /* Stage 1 only: marching cubes mesh export */
        fprintf(stderr, "\n=== Mesh Export (grid=%d, threshold=%.1f) ===\n", mc_grid, mc_threshold);

        float *mc_input = occupancy;
        int mc_n = 64;

        if (mc_grid > 0 && mc_grid < 64) {
            mc_n = mc_grid;
            float *ds = (float *)calloc((size_t)mc_n * mc_n * mc_n, sizeof(float));
            float scale2 = 64.0f / (float)mc_n;
            for (int z = 0; z < mc_n; z++) {
                for (int y = 0; y < mc_n; y++) {
                    for (int xi = 0; xi < mc_n; xi++) {
                        float fz = (z + 0.5f) * scale2 - 0.5f;
                        float fy = (y + 0.5f) * scale2 - 0.5f;
                        float fx = (xi + 0.5f) * scale2 - 0.5f;
                        int iz = (int)fz, iy = (int)fy, ix = (int)fx;
                        if (iz < 0) iz = 0; if (iz > 62) iz = 62;
                        if (iy < 0) iy = 0; if (iy > 62) iy = 62;
                        if (ix < 0) ix = 0; if (ix > 62) ix = 62;
                        float dz = fz - iz, dy = fy - iy, dx = fx - ix;
                        #define OCC(a,b,c) occupancy[(a)*64*64 + (b)*64 + (c)]
                        float v = OCC(iz,iy,ix)*(1-dz)*(1-dy)*(1-dx)
                                + OCC(iz,iy,ix+1)*(1-dz)*(1-dy)*dx
                                + OCC(iz,iy+1,ix)*(1-dz)*dy*(1-dx)
                                + OCC(iz,iy+1,ix+1)*(1-dz)*dy*dx
                                + OCC(iz+1,iy,ix)*dz*(1-dy)*(1-dx)
                                + OCC(iz+1,iy,ix+1)*dz*(1-dy)*dx
                                + OCC(iz+1,iy+1,ix)*dz*dy*(1-dx)
                                + OCC(iz+1,iy+1,ix+1)*dz*dy*dx;
                        #undef OCC
                        ds[z * mc_n * mc_n + y * mc_n + xi] = v;
                    }
                }
            }
            mc_input = ds;
            fprintf(stderr, "Downsampled 64^3 -> %d^3\n", mc_n);
        }

        float bounds[6] = {0, 0, 0, 1, 1, 1};
        mc_mesh mesh = mc_marching_cubes(mc_input, mc_n, mc_n, mc_n, mc_threshold, bounds);
        fprintf(stderr, "Marching cubes: %d vertices, %d triangles\n", mesh.n_verts, mesh.n_tris);

        if (mesh.n_tris > 0) {
            mc_write_obj(obj_path, &mesh);
            fprintf(stderr, "Wrote %s\n", obj_path);
        }
        mc_mesh_free(&mesh);
        if (mc_input != occupancy) free(mc_input);
    }

    free(occupancy);
    free(features);
    cuda_trellis2_free(r);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
