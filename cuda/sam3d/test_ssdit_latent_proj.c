/*
 * test_ssdit_latent_proj — Phase 2c.8 standalone microbench.
 *
 * Validates the SS Flow DiT latent_mapping input/output projections
 * end-to-end on-device against host references `ssdit_project_input`
 * and `ssdit_project_output` (sam3d_ss_flow_dit.h):
 *
 *   input :  z [T, in_ch]
 *            → gemm_f32_bias(input_w, input_b)        → [T, dim]
 *            → residual_add_f32(pos_emb [T, dim])     (in-place)
 *
 *   output:  h [T, dim]
 *            → layernorm_token_f32(affine=0)          → [T, dim]
 *            → gemm_f32_bias(out_w, out_b)            → [T, in_ch]
 *
 * pos_emb is pre-baked into the safetensors checkpoint; on-device we
 * just upload the [T, dim] table and reuse `residual_add_f32` (no
 * pos-add kernel needed). Output LN has no affine — `layernorm_token_f32`
 * already supports `affine=0`.
 *
 * Realistic-ish shape-stream dims: T=4096, in_ch=8, dim=1024. Random
 * weights ~ 1/sqrt(fan_in); threshold accounts for LN reduce + 2 gemms.
 *
 * Usage:
 *   ./test_ssdit_latent_proj [--T 4096] [--inch 8] [--dim 1024]
 *                            [--threshold 5e-5] [-v]
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
#include "cuda_sam3d_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float urand(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return (float)((*s) >> 8) / (float)(1u << 24);
}
static float max_abs(const float *a, const float *b, size_t n, double *mean_out) {
    double sum = 0.0; float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    if (mean_out) *mean_out = sum / (n > 0 ? n : 1);
    return mx;
}
static void hgemm(float *Y, const float *X, const float *W, const float *b,
                  int N, int D_out, int D_in) {
    for (int n = 0; n < N; n++)
        for (int d = 0; d < D_out; d++) {
            float acc = b ? b[d] : 0.0f;
            const float *xr = X + (size_t)n * D_in;
            const float *wr = W + (size_t)d * D_in;
            for (int k = 0; k < D_in; k++) acc += wr[k] * xr[k];
            Y[(size_t)n * D_out + d] = acc;
        }
}
static void hlayernorm_no_affine(float *out, const float *in, int T, int dim, float eps) {
    for (int t = 0; t < T; t++) {
        const float *x = in  + (size_t)t * dim;
        float       *y = out + (size_t)t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= (float)dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= (float)dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++) y[i] = (x[i] - mean) * inv;
    }
}

int main(int argc, char **argv)
{
    int   T         = 4096;
    int   in_ch     = 8;
    int   dim       = 1024;
    float threshold = 5e-5f;
    int   verbose   = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--T")         && i+1 < argc) T         = atoi(argv[++i]);
        else if (!strcmp(a, "--inch")      && i+1 < argc) in_ch     = atoi(argv[++i]);
        else if (!strcmp(a, "--dim")       && i+1 < argc) dim       = atoi(argv[++i]);
        else if (!strcmp(a, "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                        verbose   = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    uint32_t rng = 0xC0FFEEu;
    float *z       = (float *)malloc((size_t)T * in_ch * sizeof(float));
    float *iw      = (float *)malloc((size_t)dim * in_ch * sizeof(float));
    float *ib      = (float *)malloc((size_t)dim * sizeof(float));
    float *ow      = (float *)malloc((size_t)in_ch * dim * sizeof(float));
    float *ob      = (float *)malloc((size_t)in_ch * sizeof(float));
    float *pos_emb = (float *)malloc((size_t)T * dim * sizeof(float));
    /* Random h-input for the output projection — independent of input
     * projection's output, so output verify isn't dragged by input drift. */
    float *h_in    = (float *)malloc((size_t)T * dim * sizeof(float));
    for (int i = 0; i < T * in_ch;     i++) z[i]       = urand(&rng) * 2.0f - 1.0f;
    float si = 1.0f / sqrtf((float)in_ch);
    float so = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < dim * in_ch;   i++) iw[i]      = (urand(&rng) * 2.0f - 1.0f) * si;
    for (int i = 0; i < dim;           i++) ib[i]      = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < in_ch * dim;   i++) ow[i]      = (urand(&rng) * 2.0f - 1.0f) * so;
    for (int i = 0; i < in_ch;         i++) ob[i]      = (urand(&rng) * 2.0f - 1.0f) * 0.01f;
    for (int i = 0; i < T * dim;       i++) pos_emb[i] = (urand(&rng) * 2.0f - 1.0f) * 0.02f;
    for (int i = 0; i < T * dim;       i++) h_in[i]    = urand(&rng) * 2.0f - 1.0f;

    /* Host ref. */
    float *in_ref  = (float *)malloc((size_t)T * dim * sizeof(float));
    hgemm(in_ref, z, iw, ib, T, dim, in_ch);
    for (size_t i = 0; i < (size_t)T * dim; i++) in_ref[i] += pos_emb[i];

    float *ln_ref  = (float *)malloc((size_t)T * dim * sizeof(float));
    float *out_ref = (float *)malloc((size_t)T * in_ch * sizeof(float));
    float eps = 1e-6f;
    hlayernorm_no_affine(ln_ref, h_in, T, dim, eps);
    hgemm(out_ref, ln_ref, ow, ob, T, in_ch, dim);
    free(ln_ref);

    /* Device. */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    if (cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                           "sam3d_kernels.cu", verbose, "test_ssdit_latent_proj") < 0) return 4;
    CUfunction fn_gemm = 0, fn_add = 0, fn_ln = 0;
    if (cuModuleGetFunction(&fn_gemm, mod, "gemm_f32_bias")      != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_add,  mod, "residual_add_f32")   != CUDA_SUCCESS) return 4;
    if (cuModuleGetFunction(&fn_ln,   mod, "layernorm_token_f32") != CUDA_SUCCESS) return 4;

    CUdeviceptr d_z   = cu_upload_raw(z,       (size_t)T * in_ch * sizeof(float));
    CUdeviceptr d_iw  = cu_upload_raw(iw,      (size_t)dim * in_ch * sizeof(float));
    CUdeviceptr d_ib  = cu_upload_raw(ib,      (size_t)dim * sizeof(float));
    CUdeviceptr d_ow  = cu_upload_raw(ow,      (size_t)in_ch * dim * sizeof(float));
    CUdeviceptr d_ob  = cu_upload_raw(ob,      (size_t)in_ch * sizeof(float));
    CUdeviceptr d_pos = cu_upload_raw(pos_emb, (size_t)T * dim * sizeof(float));
    CUdeviceptr d_h   = cu_upload_raw(h_in,    (size_t)T * dim * sizeof(float));
    CUdeviceptr d_in_out = 0, d_ln = 0, d_out = 0;
    cuMemAlloc(&d_in_out, (size_t)T * dim * sizeof(float));
    cuMemAlloc(&d_ln,     (size_t)T * dim * sizeof(float));
    cuMemAlloc(&d_out,    (size_t)T * in_ch * sizeof(float));

    /* Input projection. */
    {
        unsigned gx = (T + 15) / 16, gy = (dim + 15) / 16;
        void *args[] = { &d_in_out, &d_z, &d_iw, &d_ib, &T, &in_ch, &dim };
        if (cuLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        int total = T * dim;
        unsigned grid = (unsigned)((total + 255) / 256);
        void *args[] = { &d_in_out, &d_pos, &total };
        if (cuLaunchKernel(fn_add, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    /* Output projection: LN(no affine) → gemm. */
    {
        unsigned threads = 256;
        size_t smem = 2 * threads * sizeof(float);
        int affine = 0;
        CUdeviceptr d_null = 0;
        void *args[] = { &d_ln, &d_h, &d_null, &d_null, &T, &dim, &eps, &affine };
        if (cuLaunchKernel(fn_ln, (unsigned)T, 1, 1, threads, 1, 1,
                           (unsigned)smem, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    {
        unsigned gx = (T + 15) / 16, gy = (in_ch + 15) / 16;
        void *args[] = { &d_out, &d_ln, &d_ow, &d_ob, &T, &dim, &in_ch };
        if (cuLaunchKernel(fn_gemm, gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return 5;
    }
    cuCtxSynchronize();

    float *in_dst  = (float *)malloc((size_t)T * dim * sizeof(float));
    float *out_dst = (float *)malloc((size_t)T * in_ch * sizeof(float));
    cuMemcpyDtoH(in_dst,  d_in_out, (size_t)T * dim * sizeof(float));
    cuMemcpyDtoH(out_dst, d_out,    (size_t)T * in_ch * sizeof(float));

    double mean_in = 0.0, mean_out = 0.0;
    float mx_in  = max_abs(in_dst,  in_ref,  (size_t)T * dim,   &mean_in);
    float mx_out = max_abs(out_dst, out_ref, (size_t)T * in_ch, &mean_out);
    int ok = (mx_in <= threshold) && (mx_out <= threshold);

    fprintf(stderr,
        "[test_ssdit_latent_proj] T=%d in_ch=%d dim=%d  "
        "input max_abs=%.4g (mean %.4g)  output max_abs=%.4g (mean %.4g)  %s (threshold %.1g)\n",
        T, in_ch, dim, (double)mx_in, mean_in, (double)mx_out, mean_out,
        ok ? "OK" : "FAIL", (double)threshold);

    free(z); free(iw); free(ib); free(ow); free(ob); free(pos_emb); free(h_in);
    free(in_ref); free(out_ref); free(in_dst); free(out_dst);
    cuMemFree(d_z); cuMemFree(d_iw); cuMemFree(d_ib);
    cuMemFree(d_ow); cuMemFree(d_ob); cuMemFree(d_pos); cuMemFree(d_h);
    cuMemFree(d_in_out); cuMemFree(d_ln); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return ok ? 0 : 9;
}
