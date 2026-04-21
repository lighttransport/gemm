/*
 * test_hip_c2s.c - Validate HIP Channel-to-Spatial block against CPU.
 *
 * Chain: LN_affine -> SiLU -> sparse_conv3d (C_in->C_out*8) -> gather
 *      -> LN_noaffine -> SiLU -> sparse_conv3d on fine (C_out->C_out)
 *      -> residual_repeat(x_fine).
 *
 * Uses cache_scale2_c2s_{idx,subidx,x_coords}.npy for stage-0 subdivision.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "hip_tex_dec_kernels.h"

static float *read_npy_f32(const char *p, int *nd, int *dd) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET); uint16_t hl; fread(&hl, 2, 1, f);
    char *h = malloc(hl + 1); fread(h, 1, hl, f); h[hl] = 0; *nd = 0;
    char *sp = strstr(h, "shape"); sp = sp ? strchr(sp, '(') : NULL;
    if (sp) { sp++; while (*sp && *sp != ')') {
        while (*sp == ' ' || *sp == ',') sp++;
        if (*sp == ')') break;
        dd[(*nd)++] = (int)strtol(sp, &sp, 10);
    } }
    size_t n = 1; for (int i = 0; i < *nd; i++) n *= dd[i];
    float *d = malloc(n * sizeof(float)); fread(d, sizeof(float), n, f);
    fclose(f); free(h); return d;
}
static int32_t *read_npy_i32(const char *p, int *nd, int *dd) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET); uint16_t hl; fread(&hl, 2, 1, f);
    char *h = malloc(hl + 1); fread(h, 1, hl, f); h[hl] = 0; *nd = 0;
    char *sp = strstr(h, "shape"); sp = sp ? strchr(sp, '(') : NULL;
    if (sp) { sp++; while (*sp && *sp != ')') {
        while (*sp == ' ' || *sp == ',') sp++;
        if (*sp == ')') break;
        dd[(*nd)++] = (int)strtol(sp, &sp, 10);
    } }
    size_t n = 1; for (int i = 0; i < *nd; i++) n *= dd[i];
    int32_t *d = malloc(n * sizeof(int32_t)); fread(d, sizeof(int32_t), n, f);
    fclose(f); free(h); return d;
}
static int64_t *read_npy_i64(const char *p, int *nd, int *dd) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET); uint16_t hl; fread(&hl, 2, 1, f);
    char *h = malloc(hl + 1); fread(h, 1, hl, f); h[hl] = 0; *nd = 0;
    char *sp = strstr(h, "shape"); sp = sp ? strchr(sp, '(') : NULL;
    if (sp) { sp++; while (*sp && *sp != ')') {
        while (*sp == ' ' || *sp == ',') sp++;
        if (*sp == ')') break;
        dd[(*nd)++] = (int)strtol(sp, &sp, 10);
    } }
    size_t n = 1; for (int i = 0; i < *nd; i++) n *= dd[i];
    int64_t *d = malloc(n * sizeof(int64_t)); fread(d, sizeof(int64_t), n, f);
    fclose(f); free(h); return d;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <tex_dec.st> <tex_slat_feats.npy> <tex_slat_coords.npy>\n"
            "       [--cache <dir>] [--scale N] [--c2s-block-idx N]\n", argv[0]);
        return 1;
    }
    const char *st_path = argv[1];
    const char *cache_dir = "/tmp/tex_knight_r512";
    int scale = 2;
    int c2s_block = 3;  /* blocks.0.3 is stage-0 C2S for tex_dec (3 convnext + 1 c2s) */
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--cache") && i+1 < argc) cache_dir = argv[++i];
        else if (!strcmp(argv[i], "--scale") && i+1 < argc) scale = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--c2s-block-idx") && i+1 < argc) c2s_block = atoi(argv[++i]);
    }

    int fnd, fdd[8], cnd, cdd[8];
    float *slat = read_npy_f32(argv[2], &fnd, fdd);
    int N = fdd[0], slat_C = fnd >= 2 ? fdd[1] : 1;
    int32_t *coords = read_npy_i32(argv[3], &cnd, cdd);
    fprintf(stderr, "coarse: N=%d slat_C=%d\n", N, slat_C);

    /* Build [N, 1024] input deterministically. */
    const int C_in = 1024, C_out = 512;
    const int C_in8 = C_in / 8;
    const int Cexp = C_out * 8;
    float *feats = (float *)aligned_alloc(64, (size_t)N * C_in * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int c = 0; c < C_in; c++) {
            float v = slat[(size_t)i * slat_C + (c % slat_C)];
            feats[(size_t)i * C_in + c] = v * (1.0f + 0.01f * (c / slat_C));
        }
    free(slat);

    /* Load subdiv cache. */
    char path[512];
    int dn, dd2[8];
    snprintf(path, sizeof path, "%s/cache_scale%d_c2s_idx.npy", cache_dir, scale);
    int64_t *idx = read_npy_i64(path, &dn, dd2);
    int N_fine = dd2[0];
    snprintf(path, sizeof path, "%s/cache_scale%d_c2s_subidx.npy", cache_dir, scale);
    int64_t *subidx = read_npy_i64(path, &dn, dd2);
    snprintf(path, sizeof path, "%s/cache_scale%d_c2s_x_coords.npy", cache_dir, scale);
    int32_t *x_coords = read_npy_i32(path, &dn, dd2);
    fprintf(stderr, "fine: N_fine=%d\n", N_fine);

    /* Load C2S weights: blocks.0.<c2s_block>.{norm1,conv1,conv2}.{weight,bias} */
    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "open failed\n"); return 1; }
    char wn[128];
    #define LW(field, suffix) do { \
        snprintf(wn, sizeof wn, "blocks.0.%d.%s", c2s_block, suffix); \
        field = t2sd_load_f32(st, wn); \
        if (!field) { fprintf(stderr, "missing %s\n", wn); return 1; } \
    } while(0)
    float *norm1_w, *norm1_b, *conv1_w, *conv1_b, *conv2_w, *conv2_b;
    LW(norm1_w, "norm1.weight");
    LW(norm1_b, "norm1.bias");
    LW(conv1_w, "conv1.weight");
    LW(conv1_b, "conv1.bias");
    LW(conv2_w, "conv2.weight");
    LW(conv2_b, "conv2.bias");
    fprintf(stderr, "C2S weights loaded (C_in=%d, C_out=%d)\n", C_in, C_out);

    t2sd_c2s blk = {
        .norm1_w = norm1_w, .norm1_b = norm1_b,
        .conv1_w = conv1_w, .conv1_b = conv1_b,
        .conv2_w = conv2_w, .conv2_b = conv2_b,
        .to_subdiv_w = NULL, .to_subdiv_b = NULL,
        .C_in = C_in, .C_out = C_out,
    };
    t2_shape_dec_subdiv_stage guide = { idx, subidx, x_coords, N_fine };

    /* CPU reference. */
    sp3d_tensor *t = sp3d_create(coords, feats, N, C_in, 1);
    double tc = t2sd_time_ms();
    sp3d_tensor *t_fine = t2sd_c2s_forward(t, &blk, &guide, 16);
    tc = t2sd_time_ms() - tc;
    fprintf(stderr, "CPU C2S: %.1f ms, fine N=%d C=%d\n", tc, t_fine->N, t_fine->C);

    /* HIP. */
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != 0) return 1;
    hipSetDevice(0);
    hipModule_t mod;
    if (hip_compile_kernels(&mod, 0, hip_tex_dec_kernels_src, "tex_dec", 1, "HIP") <= 0) return 1;
    hipFunction_t fn_ins, fn_conv, fn_conv_tiled, fn_ln, fn_silu, fn_gather, fn_resrep;
    hipModuleGetFunction(&fn_ins,    mod, "hash_insert_kernel");
    hipModuleGetFunction(&fn_conv,   mod, "sparse_conv3d_f32");
    hipModuleGetFunction(&fn_conv_tiled, mod, "sparse_conv3d_tiled_f32");
    hipModuleGetFunction(&fn_ln,     mod, "t2_layernorm_f32");
    hipModuleGetFunction(&fn_silu,   mod, "t2_silu_f32");
    hipModuleGetFunction(&fn_gather, mod, "t2_c2s_gather_f32");
    hipModuleGetFunction(&fn_resrep, mod, "t2_residual_repeat_f32");

    /* Uploads. */
    void *d_coords = hip_upload_raw(coords, (size_t)N * 4 * sizeof(int32_t));
    void *d_feats  = hip_upload_raw(feats,  (size_t)N * C_in * sizeof(float));
    void *d_fcoords = hip_upload_raw(x_coords, (size_t)N_fine * 4 * sizeof(int32_t));
    void *d_idx    = hip_upload_raw(idx,    (size_t)N_fine * sizeof(int64_t));
    void *d_subidx = hip_upload_raw(subidx, (size_t)N_fine * sizeof(int64_t));
    void *d_n1w = hip_upload_raw(norm1_w, (size_t)C_in * sizeof(float));
    void *d_n1b = hip_upload_raw(norm1_b, (size_t)C_in * sizeof(float));
    void *d_c1w = hip_upload_raw(conv1_w, (size_t)Cexp * 27 * C_in * sizeof(float));
    void *d_c1b = hip_upload_raw(conv1_b, (size_t)Cexp * sizeof(float));
    void *d_c2w = hip_upload_raw(conv2_w, (size_t)C_out * 27 * C_out * sizeof(float));
    void *d_c2b = hip_upload_raw(conv2_b, (size_t)C_out * sizeof(float));

    /* Scratch. */
    void *d_normed = NULL;   hipMalloc(&d_normed,   (size_t)N * C_in * sizeof(float));
    void *d_expanded = NULL; hipMalloc(&d_expanded, (size_t)N * Cexp * sizeof(float));
    void *d_hfine = NULL;    hipMalloc(&d_hfine,    (size_t)N_fine * C_out * sizeof(float));
    void *d_xfine = NULL;    hipMalloc(&d_xfine,    (size_t)N_fine * C_in8 * sizeof(float));
    void *d_hnormed = NULL;  hipMalloc(&d_hnormed,  (size_t)N_fine * C_out * sizeof(float));
    void *d_out = NULL;      hipMalloc(&d_out,      (size_t)N_fine * C_out * sizeof(float));

    /* Coarse hash. */
    int cap = 1; while (cap < N * 2) cap <<= 1;
    int cap_mask = cap - 1;
    void *d_keys = NULL; hipMalloc(&d_keys, (size_t)cap * sizeof(uint64_t));
    void *d_vals = NULL; hipMalloc(&d_vals, (size_t)cap * sizeof(int32_t));
    hipMemset(d_keys, 0, (size_t)cap * sizeof(uint64_t));
    hipMemset(d_vals, 0xff, (size_t)cap * sizeof(int32_t));
    int N_arg = N;
    { void *a[] = {&d_keys, &d_vals, &cap_mask, &d_coords, &N_arg};
      hipModuleLaunchKernel(fn_ins, (N+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* Fine hash. */
    int cap_f = 1; while (cap_f < N_fine * 2) cap_f <<= 1;
    int cap_f_mask = cap_f - 1;
    void *d_fkeys = NULL; hipMalloc(&d_fkeys, (size_t)cap_f * sizeof(uint64_t));
    void *d_fvals = NULL; hipMalloc(&d_fvals, (size_t)cap_f * sizeof(int32_t));
    hipMemset(d_fkeys, 0, (size_t)cap_f * sizeof(uint64_t));
    hipMemset(d_fvals, 0xff, (size_t)cap_f * sizeof(int32_t));
    int Nf = N_fine;
    { void *a[] = {&d_fkeys, &d_fvals, &cap_f_mask, &d_fcoords, &Nf};
      hipModuleLaunchKernel(fn_ins, (Nf+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    hipEvent_t ev0, ev1; hipEventCreate(&ev0); hipEventCreate(&ev1);
    hipEventRecord(ev0, 0);

    /* 1. LN_affine(feats) -> normed */
    float eps = 1e-6f; int yes = 1, no = 0;
    { void *a[] = {&d_normed, &d_feats, &d_n1w, &d_n1b, (void*)&(int){C_in}, &eps, &yes, &yes};
      hipModuleLaunchKernel(fn_ln, N, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 2. SiLU(normed) */
    int n_coarse = N * C_in;
    { void *a[] = {&d_normed, &d_normed, &n_coarse};
      hipModuleLaunchKernel(fn_silu, (n_coarse+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 3. sparse_conv3d: C_in -> C_out*8 */
    int in_c = C_in, out_c = Cexp;
    { void *a[] = {&d_expanded, &d_normed, &d_coords, &d_c1w, &d_c1b,
                    &d_keys, &d_vals, &cap_mask, &in_c, &out_c};
      if (out_c % 64 == 0 && in_c % 32 == 0)
        hipModuleLaunchKernel(fn_conv_tiled, N, out_c / 64, 1, 64, 1, 1, 0, 0, a, NULL);
      else
        hipModuleLaunchKernel(fn_conv, N, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 4. Gather h_fine + x_fine from expanded + feats */
    int mx = C_out > C_in8 ? C_out : C_in8;
    int gy = (mx + 255) / 256;
    { void *a[] = {&d_hfine, &d_xfine, &d_expanded, &d_feats,
                    &d_idx, &d_subidx,
                    (void*)&(int){C_out}, (void*)&(int){C_in8}};
      hipModuleLaunchKernel(fn_gather, N_fine, gy, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 5. LN_noaffine(h_fine) -> h_normed */
    void *nullw = NULL;
    { void *a[] = {&d_hnormed, &d_hfine, &nullw, &nullw,
                    (void*)&(int){C_out}, &eps, &no, &no};
      hipModuleLaunchKernel(fn_ln, N_fine, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 6. SiLU(h_normed) */
    int n_fine = N_fine * C_out;
    { void *a[] = {&d_hnormed, &d_hnormed, &n_fine};
      hipModuleLaunchKernel(fn_silu, (n_fine+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 7. sparse_conv3d on fine: C_out -> C_out */
    in_c = C_out; out_c = C_out;
    { void *a[] = {&d_out, &d_hnormed, &d_fcoords, &d_c2w, &d_c2b,
                    &d_fkeys, &d_fvals, &cap_f_mask, &in_c, &out_c};
      if (out_c % 64 == 0 && in_c % 32 == 0)
        hipModuleLaunchKernel(fn_conv_tiled, N_fine, out_c / 64, 1, 64, 1, 1, 0, 0, a, NULL);
      else
        hipModuleLaunchKernel(fn_conv, N_fine, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 8. Residual: out[i, c] += x_fine[i, c / rep] */
    int N_arg_f = N_fine;
    { void *a[] = {&d_out, &d_xfine, &N_arg_f,
                    (void*)&(int){C_out}, (void*)&(int){C_in8}};
      hipModuleLaunchKernel(fn_resrep, N_fine, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    hipEventRecord(ev1, 0);
    hipEventSynchronize(ev1);
    float th = 0; hipEventElapsedTime(&th, ev0, ev1);
    fprintf(stderr, "HIP C2S: %.1f ms\n", th);

    float *hip_out = (float *)aligned_alloc(64, (size_t)N_fine * C_out * sizeof(float));
    hipMemcpy(hip_out, d_out, (size_t)N_fine * C_out * sizeof(float), hipMemcpyDeviceToHost);

    double sse = 0, sref = 0; float mxabs = 0;
    for (size_t i = 0; i < (size_t)N_fine * C_out; i++) {
        double dv = (double)hip_out[i] - t_fine->feats[i];
        sse += dv*dv; sref += (double)t_fine->feats[i] * t_fine->feats[i];
        float a = (float)fabs(dv); if (a > mxabs) mxabs = a;
    }
    double rel = sqrt(sse / (sref + 1e-30));
    fprintf(stderr, "rel_err=%.3e max_abs=%.3e speedup=%.1fx\n",
            rel, mxabs, tc / (th + 1e-6));
    int ok = rel < 5e-4;
    fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
