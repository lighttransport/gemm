/*
 * test_cuda_trellis2.c - Test CUDA TRELLIS.2 Stage 1 runner
 *
 * Usage:
 *   ./test_cuda_trellis2 <stage1.st> <decoder.st> <features.npy> [-s seed] [-o output.obj]
 *
 * Runs Stage 1 DiT + decoder on GPU, exports mesh via marching cubes.
 * DINOv3 features must be pre-computed on CPU (pass as .npy).
 */

#include "cuda_trellis2_runner.h"
#define MARCHING_CUBES_IMPLEMENTATION
#include "../../common/marching_cubes.h"

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
                "  --npy <path>     Also save latent as .npy\n", argv[0]);
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

    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "-s") && i+1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g") && i+1 < argc) cfg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i+1 < argc) obj_path = argv[++i];
        else if (!strcmp(argv[i], "--npy") && i+1 < argc) npy_path = argv[++i];
        else if (!strcmp(argv[i], "--grid") && i+1 < argc) mc_grid = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) mc_threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--noise") && i+1 < argc) noise_path = argv[++i];
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

            /* CFG rescale (std-ratio matching): x0_pos from v_cond, x0_cfg from pred_v */
            if (cfg_rescale_val > 0.0f) {
                float tc = sigma_min + (1.0f - sigma_min) * t_cur;
                float s2_pos = 0, s2_cfg = 0;
                for (int i = 0; i < N * C; i++) {
                    float x0p = (1.0f - sigma_min) * x[i] - tc * v_cond[i];  /* from conditioned */
                    float x0c = (1.0f - sigma_min) * x[i] - tc * pred_v[i];  /* from CFG combined */
                    s2_pos += x0p * x0p; s2_cfg += x0c * x0c;
                }
                float ratio = sqrtf(s2_pos / (s2_cfg + 1e-12f));
                float sc = cfg_rescale_val * ratio + (1.0f - cfg_rescale_val);
                for (int i = 0; i < N * C; i++) {
                    float x0c = (1.0f - sigma_min) * x[i] - tc * pred_v[i];
                    pred_v[i] = ((1.0f - sigma_min) * x[i] - sc * x0c) / tc;
                }
            }

            /* Euler step: x -= (t_cur - t_next) * pred_v */
            for (int i = 0; i < N * C; i++)
                x[i] -= (t_cur - t_next) * pred_v[i];
        } else {
            /* No CFG — conditioned only */
            cuda_trellis2_run_dit(r, x, t_cur, features, v_cond);
            for (int i = 0; i < N * C; i++)
                x[i] -= (t_cur - t_next) * v_cond[i];
        }

        clock_gettime(CLOCK_MONOTONIC, &step_ts);
        double step_t1 = step_ts.tv_sec * 1000.0 + step_ts.tv_nsec / 1e6;
        fprintf(stderr, "  step %d/%d  t=%.4f->%.4f  %s  %.1f ms\n",
                step+1, n_steps, t_cur, t_next,
                apply_cfg ? "CFG" : "noG", step_t1 - step_t0);
    }

    free(zeros_cond); free(v_cond); free(v_uncond); free(features);

    /* Save latent */
    if (npy_path) {
        int d[4] = {C, 16, 16, 16};
        write_npy_f32(npy_path, x, d, 4);
    }

    /* Decode */
    fprintf(stderr, "\n=== Decoder ===\n");
    float *occupancy = (float *)malloc(64 * 64 * 64 * sizeof(float));
    cuda_trellis2_run_decoder(r, x, occupancy);
    free(x);

    struct timespec t1_ts; clock_gettime(CLOCK_MONOTONIC, &t1_ts);
    double t1 = t1_ts.tv_sec * 1000.0 + t1_ts.tv_nsec / 1e6;
    fprintf(stderr, "\nTotal GPU time: %.1f s\n", (t1 - t0) / 1000.0);

    /* Stats */
    float mn = occupancy[0], mx = occupancy[0];
    double sum = 0;
    int occ_count = 0;
    for (int i = 0; i < 64*64*64; i++) {
        if (occupancy[i] < mn) mn = occupancy[i];
        if (occupancy[i] > mx) mx = occupancy[i];
        sum += occupancy[i];
        if (occupancy[i] > 0) occ_count++;
    }
    fprintf(stderr, "Occupancy: min=%.2f max=%.2f mean=%.4f\n", mn, mx, sum/(64*64*64));
    fprintf(stderr, "Occupied (logit>0): %d / %d (%.1f%%)\n",
            occ_count, 64*64*64, 100.0f * occ_count / (64*64*64));

    /* Marching cubes + OBJ export */
    fprintf(stderr, "\n=== Mesh Export (grid=%d, threshold=%.1f) ===\n", mc_grid, mc_threshold);

    float *mc_input = occupancy;
    int mc_n = 64;

    /* Downsample occupancy grid if requested */
    if (mc_grid > 0 && mc_grid < 64) {
        mc_n = mc_grid;
        float *ds = (float *)calloc((size_t)mc_n * mc_n * mc_n, sizeof(float));
        float scale = 64.0f / (float)mc_n;
        for (int z = 0; z < mc_n; z++) {
            for (int y = 0; y < mc_n; y++) {
                for (int x = 0; x < mc_n; x++) {
                    /* Trilinear interpolation from 64^3 */
                    float fz = (z + 0.5f) * scale - 0.5f;
                    float fy = (y + 0.5f) * scale - 0.5f;
                    float fx = (x + 0.5f) * scale - 0.5f;
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
                    ds[z * mc_n * mc_n + y * mc_n + x] = v;
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
    free(occupancy);
    cuda_trellis2_free(r);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
