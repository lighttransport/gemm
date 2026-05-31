/*
 * cuda_svdquant.h -- reusable CUDA helpers for SVDQuant residual paths.
 *
 * The INT8 path uses cuBLAS IMMA (s8 x s8 -> s32) over group-64 K slices, then
 * applies per-token/per-output scales into an f32 residual. The FP4 path stays
 * in cuda/fp4_w4a4.h; this header covers the common encode/decode pieces and
 * the INT8 MMA residual dispatch.
 */
#ifndef CUDA_SVDQUANT_H
#define CUDA_SVDQUANT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuew.h"
#include "../cublasew.h"

typedef struct {
    CUstream stream;
    CUmodule module;
    CUfunction k_zero_f32;
    CUfunction k_smooth_div;
    CUfunction k_unpack_int4_to_i8;
    CUfunction k_quant_act_int4_g64;
    CUfunction k_quant_act_int8_g64;
    CUfunction k_quant_weight_int8_g64;
    CUfunction k_accum_i32_group;
    CUdeviceptr d_i32;
    size_t i32_bytes;
} cuda_svdquant_ctx;

static const char *cuda_svdquant_kernel_src =
"extern \"C\" __global__ void sq_zero_f32(float *dst, int n) {\n"
"  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
"  if (i < n) dst[i] = 0.0f;\n"
"}\n"
"extern \"C\" __global__ void sq_smooth_div(const float *x, const float *lam,\n"
"                                           float *xr, int tok, int in) {\n"
"  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"  long n = (long)tok * in;\n"
"  if (i < n) xr[i] = x[i] / lam[i % in];\n"
"}\n"
"extern \"C\" __global__ void sq_unpack_int4_to_i8(const unsigned char *qint4,\n"
"                                                 signed char *wq,\n"
"                                                 int out, int in) {\n"
"  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"  long n = (long)out * (in >> 1);\n"
"  if (i >= n) return;\n"
"  unsigned char b = qint4[i];\n"
"  int o = (int)(i / (in >> 1));\n"
"  int p = (int)(i - (long)o * (in >> 1));\n"
"  int lo = b & 15; if (lo >= 8) lo -= 16;\n"
"  int hi = (b >> 4) & 15; if (hi >= 8) hi -= 16;\n"
"  long base = (long)o * in + p * 2;\n"
"  wq[base] = (signed char)lo;\n"
"  wq[base + 1] = (signed char)hi;\n"
"}\n"
"extern \"C\" __global__ void sq_quant_act_int4_g64(const float *xr,\n"
"                                                  signed char *xq,\n"
"                                                  float *xscale,\n"
"                                                  int tok, int in) {\n"
"  __shared__ float sm[128];\n"
"  int gid = (int)blockIdx.x;\n"
"  int ng = in >> 6;\n"
"  int t = gid / ng;\n"
"  int g = gid - t * ng;\n"
"  int tid = threadIdx.x;\n"
"  int k0 = g << 6;\n"
"  float a = 0.0f;\n"
"  if (tid < 64) a = fabsf(xr[(long)t * in + k0 + tid]);\n"
"  sm[tid] = a;\n"
"  __syncthreads();\n"
"  for (int s = 64; s > 0; s >>= 1) {\n"
"    if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);\n"
"    __syncthreads();\n"
"  }\n"
"  float sc = fmaxf(sm[0] * 0.14285714285714285f, 1.0e-12f);\n"
"  if (tid == 0) xscale[(long)t * ng + g] = sc;\n"
"  if (tid < 64) {\n"
"    float qf = rintf(xr[(long)t * in + k0 + tid] / sc);\n"
"    qf = fminf(7.0f, fmaxf(-7.0f, qf));\n"
"    xq[(long)t * in + k0 + tid] = (signed char)((int)qf);\n"
"  }\n"
"}\n"
"extern \"C\" __global__ void sq_quant_act_int8_g64(const float *xr,\n"
"                                                  signed char *xq,\n"
"                                                  float *xscale,\n"
"                                                  int tok, int in) {\n"
"  __shared__ float sm[128];\n"
"  int gid = (int)blockIdx.x;\n"
"  int ng = in >> 6;\n"
"  int t = gid / ng;\n"
"  int g = gid - t * ng;\n"
"  int tid = threadIdx.x;\n"
"  int k0 = g << 6;\n"
"  float a = 0.0f;\n"
"  if (tid < 64) a = fabsf(xr[(long)t * in + k0 + tid]);\n"
"  sm[tid] = a;\n"
"  __syncthreads();\n"
"  for (int s = 64; s > 0; s >>= 1) {\n"
"    if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);\n"
"    __syncthreads();\n"
"  }\n"
"  float sc = fmaxf(sm[0] * 0.007874015748031496f, 1.0e-12f);\n"
"  if (tid == 0) xscale[(long)t * ng + g] = sc;\n"
"  if (tid < 64) {\n"
"    float qf = rintf(xr[(long)t * in + k0 + tid] / sc);\n"
"    qf = fminf(127.0f, fmaxf(-127.0f, qf));\n"
"    xq[(long)t * in + k0 + tid] = (signed char)((int)qf);\n"
"  }\n"
"}\n"
"extern \"C\" __global__ void sq_quant_weight_int8_g64(const float *R,\n"
"                                                     signed char *wq,\n"
"                                                     float *wscale,\n"
"                                                     int out, int in) {\n"
"  __shared__ float sm[128];\n"
"  int gid = (int)blockIdx.x;\n"
"  int ng = in >> 6;\n"
"  int o = gid / ng;\n"
"  int g = gid - o * ng;\n"
"  int tid = threadIdx.x;\n"
"  int k0 = g << 6;\n"
"  float a = 0.0f;\n"
"  if (tid < 64) a = fabsf(R[(long)o * in + k0 + tid]);\n"
"  sm[tid] = a;\n"
"  __syncthreads();\n"
"  for (int s = 64; s > 0; s >>= 1) {\n"
"    if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);\n"
"    __syncthreads();\n"
"  }\n"
"  float sc = fmaxf(sm[0] * 0.007874015748031496f, 1.0e-12f);\n"
"  if (tid == 0) wscale[(long)o * ng + g] = sc;\n"
"  if (tid < 64) {\n"
"    float qf = rintf(R[(long)o * in + k0 + tid] / sc);\n"
"    qf = fminf(127.0f, fmaxf(-127.0f, qf));\n"
"    wq[(long)o * in + k0 + tid] = (signed char)((int)qf);\n"
"  }\n"
"}\n"
"extern \"C\" __global__ void sq_accum_i32_group(const int *acc,\n"
"                                               const float *xscale,\n"
"                                               const float *wscale,\n"
"                                               float *resid,\n"
"                                               int tok, int out, int ng,\n"
"                                               int group_idx) {\n"
"  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"  long n = (long)tok * out;\n"
"  if (i >= n) return;\n"
"  int t = (int)(i / out);\n"
"  int o = (int)(i - (long)t * out);\n"
"  resid[i] += (float)acc[i] * xscale[(long)t * ng + group_idx]\n"
"            * wscale[(long)o * ng + group_idx];\n"
"}\n";

#define SQ_CUDA_GROW(ptr, cap, need) do { \
    if ((need) > (cap)) { \
        if (ptr) cuMemFree(ptr); \
        if (cuMemAlloc(&(ptr), (need)) != CUDA_SUCCESS) { (ptr) = 0; (cap) = 0; return -1; } \
        (cap) = (need); \
    } \
} while (0)

static int cuda_svdquant_compile(cuda_svdquant_ctx *ctx, CUdevice dev,
                                 CUstream stream, int verbose) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->stream = stream;

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, cuda_svdquant_kernel_src, "cuda_svdquant.cu",
                           0, NULL, NULL) != NVRTC_SUCCESS) {
        fprintf(stderr, "cuda_svdquant: nvrtcCreateProgram failed\n");
        return -1;
    }

    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    char arch[64];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
    const char *opts[] = { arch };
    nvrtcResult cr = nvrtcCompileProgram(prog, 1, opts);
    if (cr != NVRTC_SUCCESS) {
        size_t ls = 0;
        nvrtcGetProgramLogSize(prog, &ls);
        char *lg = (char *)malloc(ls + 1);
        if (lg) {
            nvrtcGetProgramLog(prog, lg);
            lg[ls] = 0;
            fprintf(stderr, "cuda_svdquant: kernel compile FAILED:\n%s\n", lg);
            free(lg);
        }
        nvrtcDestroyProgram(&prog);
        return -1;
    }

    CUmodule m = NULL;
    size_t bs = 0;
    if (nvrtcGetCUBINSize && nvrtcGetCUBINSize(prog, &bs) == NVRTC_SUCCESS && bs > 0) {
        char *bin = (char *)malloc(bs);
        nvrtcGetCUBIN(prog, bin);
        nvrtcDestroyProgram(&prog);
        if (bin) {
            if (cuModuleLoadData(&m, bin) != CUDA_SUCCESS) m = NULL;
            free(bin);
        }
    } else {
        size_t ps = 0;
        nvrtcGetPTXSize(prog, &ps);
        char *ptx = (char *)malloc(ps);
        nvrtcGetPTX(prog, ptx);
        nvrtcDestroyProgram(&prog);
        if (ptx) {
            if (cuModuleLoadData(&m, ptx) != CUDA_SUCCESS) m = NULL;
            free(ptx);
        }
    }
    if (!m) {
        fprintf(stderr, "cuda_svdquant: cuModuleLoadData failed\n");
        return -1;
    }
    ctx->module = m;

#define SQ_GET_FUNC(name) do { \
    if (cuModuleGetFunction(&ctx->k_##name, m, "sq_" #name) != CUDA_SUCCESS) { \
        fprintf(stderr, "cuda_svdquant: missing kernel sq_" #name "\n"); \
        return -1; \
    } \
} while (0)
    SQ_GET_FUNC(zero_f32);
    SQ_GET_FUNC(smooth_div);
    SQ_GET_FUNC(unpack_int4_to_i8);
    SQ_GET_FUNC(quant_act_int4_g64);
    SQ_GET_FUNC(quant_act_int8_g64);
    SQ_GET_FUNC(quant_weight_int8_g64);
    SQ_GET_FUNC(accum_i32_group);
#undef SQ_GET_FUNC

    if (verbose) fprintf(stderr, "cuda_svdquant: kernels compiled for sm_%d%d\n", major, minor);
    return 0;
}

static void cuda_svdquant_free(cuda_svdquant_ctx *ctx) {
    if (ctx->d_i32) { cuMemFree(ctx->d_i32); ctx->d_i32 = 0; ctx->i32_bytes = 0; }
    if (ctx->module) { cuModuleUnload(ctx->module); ctx->module = 0; }
}

static int cuda_svdquant_zero_f32(cuda_svdquant_ctx *ctx, CUdeviceptr dst, int n) {
    void *args[] = { &dst, &n };
    CUresult e = cuLaunchKernel(ctx->k_zero_f32, (unsigned)((n + 255) / 256), 1, 1,
                                256, 1, 1, 0, ctx->stream, args, NULL);
    return e == CUDA_SUCCESS ? 0 : -1;
}

static int cuda_svdquant_smooth_div(cuda_svdquant_ctx *ctx, CUdeviceptr x,
                                    CUdeviceptr smooth, CUdeviceptr xr,
                                    int tok, int in) {
    int n = tok * in;
    void *args[] = { &x, &smooth, &xr, &tok, &in };
    CUresult e = cuLaunchKernel(ctx->k_smooth_div, (unsigned)((n + 255) / 256), 1, 1,
                                256, 1, 1, 0, ctx->stream, args, NULL);
    return e == CUDA_SUCCESS ? 0 : -1;
}

static int cuda_svdquant_unpack_int4_to_i8(cuda_svdquant_ctx *ctx,
                                           CUdeviceptr qint4, CUdeviceptr wq,
                                           int out, int in) {
    if ((in & 1) != 0) return -1;
    int n = out * (in >> 1);
    void *args[] = { &qint4, &wq, &out, &in };
    CUresult e = cuLaunchKernel(ctx->k_unpack_int4_to_i8,
                                (unsigned)((n + 255) / 256), 1, 1,
                                256, 1, 1, 0, ctx->stream, args, NULL);
    return e == CUDA_SUCCESS ? 0 : -1;
}

static int cuda_svdquant_quant_act_int4_g64(cuda_svdquant_ctx *ctx,
                                            CUdeviceptr xr, CUdeviceptr xq,
                                            CUdeviceptr xscale,
                                            int tok, int in) {
    if ((in & 63) != 0) return -1;
    int blocks = tok * (in >> 6);
    void *args[] = { &xr, &xq, &xscale, &tok, &in };
    CUresult e = cuLaunchKernel(ctx->k_quant_act_int4_g64,
                                (unsigned)blocks, 1, 1, 128, 1, 1,
                                0, ctx->stream, args, NULL);
    return e == CUDA_SUCCESS ? 0 : -1;
}

static int cuda_svdquant_quant_act_int8_g64(cuda_svdquant_ctx *ctx,
                                            CUdeviceptr xr, CUdeviceptr xq,
                                            CUdeviceptr xscale,
                                            int tok, int in) {
    if ((in & 63) != 0) return -1;
    int blocks = tok * (in >> 6);
    void *args[] = { &xr, &xq, &xscale, &tok, &in };
    CUresult e = cuLaunchKernel(ctx->k_quant_act_int8_g64,
                                (unsigned)blocks, 1, 1, 128, 1, 1,
                                0, ctx->stream, args, NULL);
    return e == CUDA_SUCCESS ? 0 : -1;
}

static int cuda_svdquant_quant_weight_int8_g64(cuda_svdquant_ctx *ctx,
                                               CUdeviceptr R, CUdeviceptr wq,
                                               CUdeviceptr wscale,
                                               int out, int in) {
    if ((in & 63) != 0) return -1;
    int blocks = out * (in >> 6);
    void *args[] = { &R, &wq, &wscale, &out, &in };
    CUresult e = cuLaunchKernel(ctx->k_quant_weight_int8_g64,
                                (unsigned)blocks, 1, 1, 128, 1, 1,
                                0, ctx->stream, args, NULL);
    return e == CUDA_SUCCESS ? 0 : -1;
}

static int cuda_svdquant_int8_residual_mma(cuda_svdquant_ctx *ctx,
                                           cublasew_context *blas,
                                           CUdeviceptr resid,
                                           CUdeviceptr wq,
                                           CUdeviceptr xq,
                                           CUdeviceptr wscale,
                                           CUdeviceptr xscale,
                                           int tok, int out, int in) {
    if ((in & 63) != 0 || (tok & 3) != 0 || (out & 3) != 0) return -1;
    int ng = in >> 6;
    int n = tok * out;
    SQ_CUDA_GROW(ctx->d_i32, ctx->i32_bytes, (size_t)n * sizeof(int));
    if (cuda_svdquant_zero_f32(ctx, resid, n) != 0) return -1;
    for (int g = 0; g < ng; g++) {
        CUdeviceptr wg = wq + (size_t)g * 64;
        CUdeviceptr xg = xq + (size_t)g * 64;
        if (cublasew_gemm_int8_s32_rowmajor_nt_strided(blas, ctx->d_i32,
                                                       wg, in, xg, in,
                                                       tok, out, 64) != 0) {
            return -1;
        }
        void *args[] = { &ctx->d_i32, &xscale, &wscale, &resid, &tok, &out, &ng, &g };
        CUresult e = cuLaunchKernel(ctx->k_accum_i32_group,
                                    (unsigned)((n + 255) / 256), 1, 1,
                                    256, 1, 1, 0, ctx->stream, args, NULL);
        if (e != CUDA_SUCCESS) return -1;
    }
    return 0;
}

#undef SQ_CUDA_GROW

#endif /* CUDA_SVDQUANT_H */
