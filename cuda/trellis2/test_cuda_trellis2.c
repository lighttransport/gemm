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
/* trellis2_shape_decoder.h compiled in cuda_trellis2_runner.c */
#include "../../common/trellis2_shape_decoder.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../common/stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../common/stb_image_write.h"
#define T2_PBR_IMPLEMENTATION
#include "../../common/trellis2_pbr.h"
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

/* Load image and preprocess for DINOv3: resize to 512x512, normalize to [3,512,512] CHW */
static float *load_image_for_dinov3(const char *path) {
    int w, h, c;
    uint8_t *img = stbi_load(path, &w, &h, &c, 3);
    if (!img) { fprintf(stderr, "Cannot load image: %s\n", path); return NULL; }
    fprintf(stderr, "Loaded image %s: %dx%d\n", path, w, h);

    /* Center crop to square */
    int sz = w < h ? w : h;
    int ox = (w - sz) / 2, oy = (h - sz) / 2;

    /* Resize to 512x512 */
    uint8_t *resized = (uint8_t *)malloc(512 * 512 * 3);
    stbir_resize_uint8_linear(img + (oy * w + ox) * 3, sz, sz, w * 3,
                               resized, 512, 512, 512 * 3, STBIR_RGB);
    stbi_image_free(img);

    /* Normalize: ImageNet mean/std, output CHW [3, 512, 512] */
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std_[3] = {0.229f, 0.224f, 0.225f};
    float *out = (float *)malloc(3 * 512 * 512 * sizeof(float));
    for (int ch = 0; ch < 3; ch++)
        for (int y = 0; y < 512; y++)
            for (int x = 0; x < 512; x++) {
                float v = (float)resized[(y * 512 + x) * 3 + ch] / 255.0f;
                out[ch * 512 * 512 + y * 512 + x] = (v - mean[ch]) / std_[ch];
            }
    free(resized);
    return out;
}

/* Normalization constants from pipeline.json */
static const float shape_slat_mean[32] = {0.781296f, 0.018091f, -0.495192f, -0.558457f, 1.060530f, 0.093252f, 1.518149f, -0.933218f, -0.732996f, 2.604095f, -0.118341f, -2.143904f, 0.495076f, -2.179512f, -2.130751f, -0.996944f, 0.261421f, -2.217463f, 1.260067f, -0.150213f, 3.790713f, 1.481266f, -1.046058f, -1.523667f, -0.059621f, 2.220780f, 1.621212f, 0.877230f, 0.567247f, -3.175944f, -3.186688f, 1.578665f};
static const float shape_slat_std[32]  = {5.972266f, 4.706852f, 5.445010f, 5.209927f, 5.320220f, 4.547237f, 5.020802f, 5.444004f, 5.226681f, 5.683095f, 4.831436f, 5.286469f, 5.652043f, 5.367606f, 5.525084f, 4.730578f, 4.805265f, 5.124013f, 5.530808f, 5.619001f, 5.103930f, 5.417670f, 5.269677f, 5.547194f, 5.634698f, 5.235274f, 6.110351f, 5.511298f, 6.237273f, 4.879207f, 5.347008f, 5.405691f};
static const float tex_slat_mean[32] = {3.501659f, 2.212398f, 2.226094f, 0.251093f, -0.026248f, -0.687364f, 0.439898f, -0.928075f, 0.029398f, -0.339596f, -0.869527f, 1.038479f, -0.972385f, 0.126042f, -1.129303f, 0.455149f, -1.209521f, 2.069067f, 0.544735f, 2.569128f, -0.323407f, 2.293000f, -1.925608f, -1.217717f, 1.213905f, 0.971588f, -0.023631f, 0.106750f, 2.021786f, 0.250524f, -0.662387f, -0.768862f};
static const float tex_slat_std[32]  = {2.665652f, 2.743913f, 2.765121f, 2.595319f, 3.037293f, 2.291316f, 2.144656f, 2.911822f, 2.969419f, 2.501689f, 2.154811f, 3.163343f, 2.621215f, 2.381943f, 3.186697f, 3.021588f, 2.295916f, 3.234985f, 3.233086f, 2.260140f, 2.874801f, 2.810596f, 3.292720f, 2.674999f, 2.680878f, 2.372054f, 2.451546f, 2.353556f, 2.995195f, 2.379849f, 2.786195f, 2.775190f};

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
                "  --image <path>   Input image (JPEG/PNG) — runs DINOv3 on GPU\n"
                "  --dinov3 <path>  DINOv3 weights .safetensors\n"
                "  --noise <path>   Load noise from .npy instead of generating\n"
                "  --occ <path>     Save occupancy grid as .npy\n"
                "\nStage 2 (shape generation):\n"
                "  --stage2 <path>      Stage 2 flow model .safetensors\n"
                "  --shape-dec <path>   Shape decoder .safetensors\n"
                "  --s2-steps <N>       Stage 2 Euler steps (default: 12)\n"
                "  --s2-cfg <scale>     Stage 2 CFG scale (default: 7.5)\n"
                "  --s2-npy <path>      Save Stage 2 latent as .npy\n"
                "  -t <threads>         CPU threads for shape decoder (default: 4)\n"
                "  --max-gpu-layers <N> Max DiT layers on GPU (0=all, default: 0)\n"
                "                       Use 1-10 to reduce VRAM at cost of speed\n"
                "\nStage 3 (texture generation):\n"
                "  --stage3 <path>      Stage 3 texture flow model .safetensors\n"
                "  --tex-dec <path>     Texture decoder .safetensors\n"
                "  --s3-steps <N>       Stage 3 Euler steps (default: 12)\n"
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
    int max_gpu_layers = 0;
    const char *image_path = NULL;
    const char *dinov3_path = NULL;
    const char *stage3_path = NULL;
    const char *tex_dec_path = NULL;
    int s3_steps = 12;

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
        else if (!strcmp(argv[i], "--max-gpu-layers") && i+1 < argc)
            max_gpu_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--image") && i+1 < argc) image_path = argv[++i];
        else if (!strcmp(argv[i], "--dinov3") && i+1 < argc) dinov3_path = argv[++i];
        else if (!strcmp(argv[i], "--stage3") && i+1 < argc) stage3_path = argv[++i];
        else if (!strcmp(argv[i], "--tex-dec") && i+1 < argc) tex_dec_path = argv[++i];
        else if (!strcmp(argv[i], "--s3-steps") && i+1 < argc) s3_steps = atoi(argv[++i]);
    }

    /* Init CUDA runner */
    fprintf(stderr, "\n=== Initializing CUDA runner ===\n");
    cuda_trellis2_runner *r = cuda_trellis2_init(0, 1);
    if (!r) return 1;

    /* Load or compute DINOv3 features */
    int ndim, dims[8];
    float *features = NULL;
    int n_cond = 1029;
    if (image_path) {
        /* Load image and run DINOv3 on GPU */
        if (!dinov3_path) {
            fprintf(stderr, "Error: --image requires --dinov3 <weights.st>\n");
            cuda_trellis2_free(r); return 1;
        }
        float *img_f32 = load_image_for_dinov3(image_path);
        if (!img_f32) { cuda_trellis2_free(r); return 1; }
        fprintf(stderr, "\n=== DINOv3 Encoding ===\n");
        if (cuda_trellis2_load_weights(r, dinov3_path, NULL, NULL) != 0) {
            free(img_f32); cuda_trellis2_free(r); return 1;
        }
        features = (float *)malloc((size_t)n_cond * 1024 * sizeof(float));
        cuda_trellis2_run_dinov3(r, img_f32, features);
        free(img_f32);
        fprintf(stderr, "DINOv3: [%d, 1024] features computed\n", n_cond);
    } else {
        /* Load pre-computed features from .npy */
        features = read_npy_f32(features_path, &ndim, dims);
        if (!features) { cuda_trellis2_free(r); return 1; }
        n_cond = dims[0];
        fprintf(stderr, "Conditioning: %d tokens, %d dim\n", n_cond, ndim >= 2 ? dims[1] : 0);
    }

    /* Load weights (DINOv3 skipped — we already have features) */
    fprintf(stderr, "\n=== Loading weights ===\n");
    if (max_gpu_layers > 0)
        cuda_trellis2_set_max_gpu_layers(r, max_gpu_layers);
    /* Load Stage 1 + decoder (DINOv3 may already be loaded via --image) */
    if (cuda_trellis2_load_weights(r, image_path ? NULL : NULL,
                                     stage1_path, decoder_path) != 0) {
        cuda_trellis2_free(r); free(features); return 1;
    }
    if (stage2_path) {
        if (cuda_trellis2_load_stage2(r, stage2_path) != 0) {
            cuda_trellis2_free(r); free(features); return 1;
        }
    }
    if (stage3_path) {
        if (cuda_trellis2_load_stage3(r, stage3_path) != 0) {
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

        /* Denormalize shape latent: slat = x_feats * std + mean */
        for (int i = 0; i < N_sparse; i++)
            for (int c = 0; c < s2_ch; c++)
                s2_x[i * s2_ch + c] = s2_x[i * s2_ch + c] * shape_slat_std[c] + shape_slat_mean[c];
        fprintf(stderr, "Shape latent denormalized\n");

        /* ============================================================ */
        /* Stage 3: Texture generation (if --stage3 provided)           */
        /* ============================================================ */
        float *tex_slat = NULL;
        if (stage3_path) {
            fprintf(stderr, "\n=== Stage 3: Texture Flow Sampling (%d steps, no CFG) ===\n", s3_steps);

            /* Normalize shape_slat for concat_cond */
            float *shape_norm = (float *)malloc((size_t)N_sparse * 32 * sizeof(float));
            for (int i = 0; i < N_sparse; i++)
                for (int c = 0; c < 32; c++)
                    shape_norm[i * 32 + c] = (s2_x[i * 32 + c] - shape_slat_mean[c]) / shape_slat_std[c];

            /* Generate noise [N, 32] */
            float *s3_noise = (float *)malloc((size_t)N_sparse * 32 * sizeof(float));
            rng_state s3_rng = {{seed + 2, (seed + 2) ^ 0x9E3779B97F4A7C15ULL,
                                 (seed + 2) ^ 0x6C62272E07BB0142ULL, (seed + 2) ^ 0xBF58476D1CE4E5B9ULL}};
            for (int i = 0; i < 8; i++) rng_next(&s3_rng);
            for (int i = 0; i < N_sparse * 32; i++) s3_noise[i] = rng_randn(&s3_rng);

            /* Concatenate [noise, shape_norm] -> [N, 64] */
            float *s3_xt = (float *)malloc((size_t)N_sparse * 64 * sizeof(float));
            float *s3_v = (float *)malloc((size_t)N_sparse * 32 * sizeof(float));
            float s3_rescale_t = 3.0f;

            struct timespec s3_t0_ts; clock_gettime(CLOCK_MONOTONIC, &s3_t0_ts);
            double s3_t0 = s3_t0_ts.tv_sec * 1000.0 + s3_t0_ts.tv_nsec / 1e6;

            for (int step = 0; step < s3_steps; step++) {
                float t_start = 1.0f - (float)step / (float)s3_steps;
                float t_end = 1.0f - (float)(step + 1) / (float)s3_steps;
                float t_cur = rescale_t(t_start, s3_rescale_t);
                float t_next = rescale_t(t_end, s3_rescale_t);

                /* Build x_t = [noise, shape_norm] */
                for (int i = 0; i < N_sparse; i++) {
                    memcpy(s3_xt + i * 64, s3_noise + i * 32, 32 * sizeof(float));
                    memcpy(s3_xt + i * 64 + 32, shape_norm + i * 32, 32 * sizeof(float));
                }

                struct timespec step_ts; clock_gettime(CLOCK_MONOTONIC, &step_ts);
                double step_t0x = step_ts.tv_sec * 1000.0 + step_ts.tv_nsec / 1e6;

                /* No CFG for Stage 3 (guidance_strength=1.0) */
                cuda_trellis2_run_stage3_dit(r, s3_xt, t_cur, features,
                                              sparse_coords, N_sparse, s3_v);

                /* Euler step: noise -= (t_cur - t_next) * v */
                for (int i = 0; i < N_sparse * 32; i++)
                    s3_noise[i] -= (t_cur - t_next) * s3_v[i];

                clock_gettime(CLOCK_MONOTONIC, &step_ts);
                double step_t1x = step_ts.tv_sec * 1000.0 + step_ts.tv_nsec / 1e6;
                fprintf(stderr, "  step %d/%d  t=%.4f->%.4f  %.1f ms\n",
                        step + 1, s3_steps, t_cur, t_next, step_t1x - step_t0x);
            }

            struct timespec s3_t1_ts; clock_gettime(CLOCK_MONOTONIC, &s3_t1_ts);
            double s3_t1 = s3_t1_ts.tv_sec * 1000.0 + s3_t1_ts.tv_nsec / 1e6;
            fprintf(stderr, "Stage 3 GPU time: %.1f s\n", (s3_t1 - s3_t0) / 1000.0);

            /* Denormalize texture latent */
            tex_slat = s3_noise;  /* reuse buffer */
            for (int i = 0; i < N_sparse; i++)
                for (int c = 0; c < 32; c++)
                    tex_slat[i * 32 + c] = tex_slat[i * 32 + c] * tex_slat_std[c] + tex_slat_mean[c];
            fprintf(stderr, "Texture latent denormalized\n");

            free(shape_norm); free(s3_xt); free(s3_v);
            /* tex_slat (= s3_noise) kept alive for texture decoder */
        }

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

        if (fdg_mesh.n_tris > 0) {
            /* Always write shape-only mesh first */
            {
                char shape_path[512];
                snprintf(shape_path, sizeof(shape_path), "%s_shape.obj", obj_path);
                t2_fdg_write_obj(shape_path, &fdg_mesh);
            }

            if (tex_slat && stage3_path && tex_dec_path) {
                /* Run texture decoder to get 6-channel PBR voxel field */
                fprintf(stderr, "\n=== Texture Decoder (CPU, %d threads) ===\n", n_threads);
                t2_shape_dec *tex_dec = t2_shape_dec_load(tex_dec_path);
                if (tex_dec) {
                    /* Create sparse tensor from texture latent + shape coords */
                    sp3d_tensor *tex_tensor = sp3d_create(sparse_coords, tex_slat,
                                                           N_sparse, 32, 1);
                    t2_shape_dec_result tex_result = t2_shape_dec_forward(tex_dec, tex_tensor, n_threads);
                    fprintf(stderr, "Texture decoder output: N=%d, C=%d\n",
                            tex_result.N, tex_result.C);

                    /* Scale decoder output: * 0.5 + 0.5 -> [0,1] */
                    /* Build PBR field from texture decoder output */
                    int max_c = 0;
                    for (int i = 0; i < tex_result.N; i++) {
                        for (int j = 1; j <= 3; j++)
                            if (tex_result.coords[i*4+j] > max_c)
                                max_c = tex_result.coords[i*4+j];
                    }
                    int tex_res = max_c + 1;
                    fprintf(stderr, "\n=== PBR Texture Baking (res=%d) ===\n", tex_res);

                    t2_pbr_field pbr = t2_pbr_from_decoder(
                        tex_result.feats, tex_result.coords, tex_result.N, tex_res);

                    /* Sample PBR at mesh vertices */
                    t2_pbr_attr *colors = (t2_pbr_attr *)malloc(
                        (size_t)fdg_mesh.n_verts * sizeof(t2_pbr_attr));
                    t2_pbr_sample_vertices(&pbr, fdg_mesh.vertices, fdg_mesh.n_verts, colors);

                    /* Write textured OBJ + MTL + texture maps */
                    /* Strip .obj extension for base path */
                    char base[512];
                    snprintf(base, sizeof(base), "%s", obj_path);
                    char *dot = strrchr(base, '.');
                    if (dot) *dot = '\0';
                    t2_pbr_write_textured_obj(base, fdg_mesh.vertices, fdg_mesh.triangles,
                                               fdg_mesh.n_verts, fdg_mesh.n_tris, colors, 1024);

                    free(colors);
                    t2_pbr_free(&pbr);
                    t2_shape_dec_result_free(&tex_result);
                    sp3d_free(tex_tensor);
                    t2_shape_dec_free(tex_dec);
                } else {
                    fprintf(stderr, "Failed to load texture decoder, writing shape-only mesh\n");
                    t2_fdg_write_obj(obj_path, &fdg_mesh);
                }
            } else {
                t2_fdg_write_obj(obj_path, &fdg_mesh);
            }
        }

        free(coords3);
        t2_fdg_mesh_free(&fdg_mesh);
        t2_shape_dec_result_free(&result);
        sp3d_free(slat);
        t2_shape_dec_free(dec);
        if (tex_slat) free(tex_slat);
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
