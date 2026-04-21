/*
 * test_hip_convnext.c - Validate HIP ConvNeXt block (conv + LN + MLP + residual)
 * against CPU reference t2sd_convnext_forward.
 *
 * Chains: sparse_conv3d_f32 -> t2_layernorm_f32 -> t2_linear_f32 (C->4C) ->
 *         t2_gelu_f32 -> t2_linear_f32 (4C->C) -> t2_add_f32 (residual).
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

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <tex_dec.safetensors> <feats.npy> <coords.npy>\n", argv[0]);
        return 1;
    }
    const char *st_path = argv[1];
    int fnd, fdd[8], cnd, cdd[8];
    float *slat = read_npy_f32(argv[2], &fnd, fdd);
    int N = fdd[0], slat_C = fnd >= 2 ? fdd[1] : 1;
    int32_t *coords = read_npy_i32(argv[3], &cnd, cdd);
    fprintf(stderr, "N=%d slat_C=%d coords=[%d,%d]\n", N, slat_C, cdd[0], cdd[1]);

    const int C = 1024;
    float *feats = (float *)aligned_alloc(64, (size_t)N * C * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int c = 0; c < C; c++) {
            float v = slat[(size_t)i * slat_C + (c % slat_C)];
            feats[(size_t)i * C + c] = v * (1.0f + 0.01f * (c / slat_C));
        }
    free(slat);

    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "open failed\n"); return 1; }
    t2sd_convnext blk = {0};
    blk.C = C;
    blk.conv_w = t2sd_load_f32(st, "blocks.0.0.conv.weight");
    blk.conv_b = t2sd_load_f32(st, "blocks.0.0.conv.bias");
    blk.norm_w = t2sd_load_f32(st, "blocks.0.0.norm.weight");
    blk.norm_b = t2sd_load_f32(st, "blocks.0.0.norm.bias");
    blk.mlp0_w = t2sd_load_f32(st, "blocks.0.0.mlp.0.weight");
    blk.mlp0_b = t2sd_load_f32(st, "blocks.0.0.mlp.0.bias");
    blk.mlp2_w = t2sd_load_f32(st, "blocks.0.0.mlp.2.weight");
    blk.mlp2_b = t2sd_load_f32(st, "blocks.0.0.mlp.2.bias");
    if (!blk.conv_w || !blk.norm_w || !blk.mlp0_w || !blk.mlp2_w) {
        fprintf(stderr, "weight load failed\n"); return 1;
    }

    /* CPU reference. */
    float *cpu_feats = (float *)aligned_alloc(64, (size_t)N * C * sizeof(float));
    memcpy(cpu_feats, feats, (size_t)N * C * sizeof(float));
    sp3d_tensor *t = sp3d_create(coords, cpu_feats, N, C, 1);
    double t_cpu = t2sd_time_ms();
    t2sd_convnext_forward(t->feats, N, &blk, t, 16);
    t_cpu = t2sd_time_ms() - t_cpu;
    fprintf(stderr, "CPU convnext: %.1f ms\n", t_cpu);

    /* HIP. */
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != 0) { fprintf(stderr, "rocewInit\n"); return 1; }
    hipSetDevice(0);
    hipModule_t mod;
    if (hip_compile_kernels(&mod, 0, hip_tex_dec_kernels_src, "tex_dec", 1, "HIP") <= 0) {
        fprintf(stderr, "compile fail\n"); return 1;
    }
    hipFunction_t fn_ins, fn_conv, fn_conv_tiled, fn_ln, fn_lin, fn_gelu, fn_add;
    hipModuleGetFunction(&fn_ins,  mod, "hash_insert_kernel");
    hipModuleGetFunction(&fn_conv, mod, "sparse_conv3d_f32");
    hipModuleGetFunction(&fn_conv_tiled, mod, "sparse_conv3d_tiled_f32");
    hipModuleGetFunction(&fn_ln,   mod, "t2_layernorm_f32");
    hipModuleGetFunction(&fn_lin,  mod, "t2_linear_f32");
    hipModuleGetFunction(&fn_gelu, mod, "t2_gelu_f32");
    hipModuleGetFunction(&fn_add,  mod, "t2_add_f32");

    int cap = 1; while (cap < N * 2) cap <<= 1;
    int cap_mask = cap - 1;
    void *d_coords = hip_upload_raw(coords, (size_t)N * 4 * sizeof(int32_t));
    void *d_feats  = hip_upload_raw(feats,  (size_t)N * C * sizeof(float));
    void *d_convw  = hip_upload_raw(blk.conv_w, (size_t)C * 27 * C * sizeof(float));
    void *d_convb  = hip_upload_raw(blk.conv_b, (size_t)C * sizeof(float));
    void *d_nw     = hip_upload_raw(blk.norm_w, (size_t)C * sizeof(float));
    void *d_nb     = hip_upload_raw(blk.norm_b, (size_t)C * sizeof(float));
    void *d_m0w    = hip_upload_raw(blk.mlp0_w, (size_t)4 * C * C * sizeof(float));
    void *d_m0b    = hip_upload_raw(blk.mlp0_b, (size_t)4 * C * sizeof(float));
    void *d_m2w    = hip_upload_raw(blk.mlp2_w, (size_t)C * 4 * C * sizeof(float));
    void *d_m2b    = hip_upload_raw(blk.mlp2_b, (size_t)C * sizeof(float));

    void *d_tmp = NULL; hipMalloc(&d_tmp, (size_t)N * C * sizeof(float));
    void *d_mlp = NULL; hipMalloc(&d_mlp, (size_t)N * 4 * C * sizeof(float));
    void *d_keys = NULL; hipMalloc(&d_keys, (size_t)cap * sizeof(uint64_t));
    void *d_vals = NULL; hipMalloc(&d_vals, (size_t)cap * sizeof(int32_t));
    hipMemset(d_keys, 0, (size_t)cap * sizeof(uint64_t));
    hipMemset(d_vals, 0xff, (size_t)cap * sizeof(int32_t));

    int N_arg = N;
    { void *a[] = {&d_keys, &d_vals, &cap_mask, &d_coords, &N_arg};
      hipModuleLaunchKernel(fn_ins, (N+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    hipEvent_t ev0, ev1; hipEventCreate(&ev0); hipEventCreate(&ev1);
    hipEventRecord(ev0, 0);

    /* 1. sparse_conv3d: d_tmp = conv(d_feats) */
    int in_C = C, out_C = C;
    { void *a[] = {&d_tmp, &d_feats, &d_coords, &d_convw, &d_convb,
                    &d_keys, &d_vals, &cap_mask, &in_C, &out_C};
      if (out_C % 64 == 0 && in_C % 32 == 0)
        hipModuleLaunchKernel(fn_conv_tiled, N, out_C / 64, 1, 64, 1, 1, 0, 0, a, NULL);
      else
        hipModuleLaunchKernel(fn_conv, N, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 2. layernorm: d_tmp = LN(d_tmp, norm_w, norm_b) */
    float eps = 1e-6f;
    int has_w = 1, has_b = 1;
    { void *a[] = {&d_tmp, &d_tmp, &d_nw, &d_nb, &in_C, &eps, &has_w, &has_b};
      hipModuleLaunchKernel(fn_ln, N, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 3. linear C->4C: d_mlp = d_tmp @ m0w^T + m0b */
    int fourC = 4 * C;
    int gx = (fourC + 15) / 16, gy = (N + 15) / 16;
    { void *a[] = {&d_mlp, &d_tmp, &d_m0w, &d_m0b, &N_arg, &in_C, &fourC};
      hipModuleLaunchKernel(fn_lin, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL); }

    /* 4. GELU on d_mlp */
    int n_mlp = N * 4 * C;
    { void *a[] = {&d_mlp, &n_mlp};
      hipModuleLaunchKernel(fn_gelu, (n_mlp+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    /* 5. linear 4C->C: d_tmp = d_mlp @ m2w^T + m2b */
    gx = (C + 15) / 16;
    { void *a[] = {&d_tmp, &d_mlp, &d_m2w, &d_m2b, &N_arg, &fourC, &out_C};
      hipModuleLaunchKernel(fn_lin, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL); }

    /* 6. residual: d_feats += d_tmp */
    int n_fc = N * C;
    { void *a[] = {&d_feats, &d_tmp, &n_fc};
      hipModuleLaunchKernel(fn_add, (n_fc+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); }

    hipEventRecord(ev1, 0);
    hipEventSynchronize(ev1);
    float t_hip = 0; hipEventElapsedTime(&t_hip, ev0, ev1);
    fprintf(stderr, "HIP convnext: %.1f ms\n", t_hip);

    float *hip_out = (float *)aligned_alloc(64, (size_t)N * C * sizeof(float));
    hipMemcpy(hip_out, d_feats, (size_t)N * C * sizeof(float), hipMemcpyDeviceToHost);

    double sse = 0, sref = 0; float mx = 0;
    for (size_t i = 0; i < (size_t)N * C; i++) {
        double dv = (double)hip_out[i] - t->feats[i];
        sse += dv*dv; sref += (double)t->feats[i] * t->feats[i];
        float a = (float)fabs(dv); if (a > mx) mx = a;
    }
    double rel = sqrt(sse / (sref + 1e-30));
    fprintf(stderr, "rel_err=%.3e max_abs=%.3e speedup=%.1fx\n",
            rel, mx, t_cpu / (t_hip + 1e-6));
    int ok = rel < 5e-4;
    fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
