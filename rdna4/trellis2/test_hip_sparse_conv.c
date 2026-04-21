/*
 * test_hip_sparse_conv.c - Stand-alone validator for HIP submanifold sparse
 * conv3d kernel (phase-2 scaffolding for TRELLIS.2 tex decoder port).
 *
 * Strategy: load tex_slat coords/feats + the stage-0 ConvNeXt conv weight
 * from tex_dec_next_dc_f16c32_fp16.safetensors, run both CPU
 * (t2sd_sparse_conv) and HIP (sparse_conv3d_f32) on the same input, and
 * report max/mean error. Validates: coord hash table, 27-neighbor gather,
 * weight layout [out_C, 27, in_C], and F32 FMA accuracy.
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

/* Minimal .npy readers. */
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

/* Reuse t2sd_load_f32 (declared static in trellis2_shape_decoder.h).
 * It dequantizes F32/F16/BF16 into a malloc'd float buffer. */

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <tex_dec.safetensors> <tex_slat_feats.npy> <tex_slat_coords.npy>\n",
                argv[0]);
        return 1;
    }
    const char *st_path = argv[1];
    const char *feats_path = argv[2];
    const char *coords_path = argv[3];

    /* Load coords + a feats input. For conv test we substitute feats with
     * random data (any C=1024 array) — we want kernel correctness, not
     * end-to-end numerical match, since the real path is layernorm(silu(...))
     * before conv. Using real tex_slat feats tiled up to 1024 would also
     * work; we just pick whatever has the right N. */
    int fnd, fdd[8];
    float *slat_feats = read_npy_f32(feats_path, &fnd, fdd);
    int N = fdd[0];
    int slat_C = fnd >= 2 ? fdd[1] : 1;
    fprintf(stderr, "slat_feats: N=%d C=%d\n", N, slat_C);

    int cnd, cdd[8];
    int32_t *coords = read_npy_i32(coords_path, &cnd, cdd);
    fprintf(stderr, "coords: [%d, %d]\n", cdd[0], cdd[1]);

    /* Build a [N, 1024] input by broadcasting tex_slat (32ch) 32× then perturbing —
     * deterministic so CPU and HIP see identical bytes. */
    const int C = 1024;
    float *feats = (float *)aligned_alloc(64, (size_t)N * C * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < C; c++) {
            float v = slat_feats[(size_t)i * slat_C + (c % slat_C)];
            v *= 1.0f + 0.01f * (c / slat_C);
            feats[(size_t)i * C + c] = v;
        }
    }
    free(slat_feats);

    /* Load first ConvNeXt block's conv weight: blocks.0.0.conv (1024->1024). */
    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "failed to open %s\n", st_path); return 1; }
    float *conv_w = t2sd_load_f32(st, "blocks.0.0.conv.weight");
    float *conv_b = t2sd_load_f32(st, "blocks.0.0.conv.bias");
    if (!conv_w) {
        fprintf(stderr, "failed to load blocks.0.0.conv.weight\n");
        safetensors_close(st); return 1;
    }
    fprintf(stderr, "loaded conv weight [%d, 27, %d]\n", C, C);

    /* --- CPU reference --- */
    sp3d_tensor *t = sp3d_create(coords, feats, N, C, 1);
    float *cpu_out = (float *)aligned_alloc(64, (size_t)N * C * sizeof(float));
    double t_cpu = t2sd_time_ms();
    t2sd_sparse_conv(cpu_out, t, conv_w, conv_b, C, C, 16);
    t_cpu = t2sd_time_ms() - t_cpu;
    fprintf(stderr, "CPU sparse_conv3d: %.1f ms\n", t_cpu);

    /* --- HIP path --- */
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != 0) {
        fprintf(stderr, "rocewInit failed\n"); return 1;
    }
    int device_id = 0;
    hipSetDevice(device_id);

    hipModule_t mod;
    if (hip_compile_kernels(&mod, device_id, hip_tex_dec_kernels_src,
                             "tex_dec", 1, "HIP-TEXDEC") <= 0) {
        fprintf(stderr, "HIP kernel compile failed\n"); return 1;
    }
    hipFunction_t fn_insert, fn_conv, fn_conv_tiled;
    hipModuleGetFunction(&fn_insert, mod, "hash_insert_kernel");
    hipModuleGetFunction(&fn_conv,   mod, "sparse_conv3d_f32");
    hipModuleGetFunction(&fn_conv_tiled, mod, "sparse_conv3d_tiled_f32");

    int cap = 1; while (cap < N * 2) cap <<= 1;
    int cap_mask = cap - 1;

    void *d_coords = hip_upload_raw(coords, (size_t)N * 4 * sizeof(int32_t));
    void *d_feats  = hip_upload_raw(feats,  (size_t)N * C * sizeof(float));
    void *d_w      = hip_upload_raw(conv_w, (size_t)C * 27 * C * sizeof(float));
    void *d_b      = hip_upload_raw(conv_b, (size_t)C * sizeof(float));
    void *d_out = NULL; hipMalloc(&d_out, (size_t)N * C * sizeof(float));
    void *d_keys = NULL; hipMalloc(&d_keys, (size_t)cap * sizeof(uint64_t));
    void *d_vals = NULL; hipMalloc(&d_vals, (size_t)cap * sizeof(int32_t));
    hipMemset(d_keys, 0, (size_t)cap * sizeof(uint64_t));
    hipMemset(d_vals, 0xff, (size_t)cap * sizeof(int32_t));

    /* Hash insert */
    int N_arg = N;
    void *args_ins[] = { &d_keys, &d_vals, &cap_mask, &d_coords, &N_arg };
    hipModuleLaunchKernel(fn_insert, (N + 255) / 256, 1, 1, 256, 1, 1,
                          0, 0, args_ins, NULL);
    hipDeviceSynchronize();

    /* Sparse conv */
    int in_C = C, out_C = C;
    void *args_conv[] = { &d_out, &d_feats, &d_coords, &d_w, &d_b,
                          &d_keys, &d_vals, &cap_mask, &in_C, &out_C };
    hipEvent_t ev0, ev1; hipEventCreate(&ev0); hipEventCreate(&ev1);
    hipEventRecord(ev0, 0);
    hipModuleLaunchKernel(fn_conv, N, 1, 1, 256, 1, 1, 0, 0, args_conv, NULL);
    hipEventRecord(ev1, 0);
    hipEventSynchronize(ev1);
    float t_hip = 0; hipEventElapsedTime(&t_hip, ev0, ev1);
    fprintf(stderr, "HIP sparse_conv3d (scalar): %.1f ms\n", t_hip);

    /* Tiled kernel */
    float t_hip_tiled = 0;
    int tiled_ok = (out_C % 64 == 0 && in_C % 32 == 0);
    if (tiled_ok) {
        hipMemset(d_out, 0, (size_t)N * C * sizeof(float));
        hipEventRecord(ev0, 0);
        hipModuleLaunchKernel(fn_conv_tiled, N, out_C / 64, 1, 64, 1, 1,
                              0, 0, args_conv, NULL);
        hipEventRecord(ev1, 0);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&t_hip_tiled, ev0, ev1);
        fprintf(stderr, "HIP sparse_conv3d (tiled):  %.1f ms  (%.2fx vs scalar)\n",
                t_hip_tiled, t_hip / (t_hip_tiled + 1e-6));
    }

    float *hip_out = (float *)aligned_alloc(64, (size_t)N * C * sizeof(float));
    hipMemcpy(hip_out, d_out, (size_t)N * C * sizeof(float), hipMemcpyDeviceToHost);

    /* Compare */
    double sum_sq_err = 0, sum_sq_ref = 0; float max_abs = 0;
    for (size_t i = 0; i < (size_t)N * C; i++) {
        double d = (double)hip_out[i] - cpu_out[i];
        sum_sq_err += d * d;
        sum_sq_ref += (double)cpu_out[i] * cpu_out[i];
        float a = (float)fabs(d);
        if (a > max_abs) max_abs = a;
    }
    double rel = sqrt(sum_sq_err / (sum_sq_ref + 1e-30));
    fprintf(stderr, "\nresult: N=%d C=%d  rel_err=%.3e  max_abs=%.3e  speedup=%.1fx\n",
            N, C, rel, max_abs, t_cpu / (t_hip + 1e-6));
    int ok = rel < 1e-4;
    fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");

    hipFree(d_coords); hipFree(d_feats); hipFree(d_w); hipFree(d_b);
    hipFree(d_out); hipFree(d_keys); hipFree(d_vals);
    hipModuleUnload(mod);
    free(hip_out); free(cpu_out); free(feats); free(coords);
    free(conv_w); free(conv_b);
    sp3d_free(t); safetensors_close(st);
    return ok ? 0 : 1;
}
