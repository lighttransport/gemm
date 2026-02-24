/*
 * cuda_ppd_runner.c - CUDA Pixel-Perfect-Depth via NVRTC-compiled kernels
 *
 * Pipeline: DA2 semantic encoder → DiT diffusion (4 Euler steps) → depth
 * Compiles with plain gcc (no nvcc). Uses cuew for dynamic CUDA/NVRTC loading.
 */

#define _GNU_SOURCE
#define PTH_LOADER_IMPLEMENTATION
#include "../../common/pth_loader.h"
#include "cuda_ppd_runner.h"
#include "../cuew.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ======================================================================== */
/* CUDA kernel source (compiled at runtime via NVRTC)                       */
/* ======================================================================== */

static const char *cuda_ppd_kernel_source =
"typedef unsigned short half_raw;\n"
"__device__ __forceinline__ float half_to_float(half_raw h) {\n"
"    float f; asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h)); return f;\n"
"}\n"
"\n"
"extern \"C\" {\n"
"\n"
"/* ---- layernorm_f32 ---- */\n"
"__global__ void layernorm_f32(float *dst, const float *src, const float *w,\n"
"                               const float *b, int dim, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int tok = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    const float *x = src + tok * dim;\n"
"    float *y = dst + tok * dim;\n"
"    float s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) s += x[i];\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)dim;\n"
"    __syncthreads();\n"
"    s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) { float d = x[i] - mean; s += d*d; }\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)dim + eps);\n"
"    for (int i = tid; i < dim; i += nt)\n"
"        y[i] = (x[i] - mean) * inv * w[i] + b[i];\n"
"}\n"
"\n"
"/* ---- qk_norm_f32: per-head layernorm on strided QKV buffer ---- */\n"
"/* One thread per (token, head). Normalizes head_dim elements in-place. */\n"
"__global__ void qk_norm_f32(float *qkv, const float *w, const float *b,\n"
"                              int n_tok, int n_heads, int head_dim,\n"
"                              int stride, float eps) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * n_heads;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / n_heads;\n"
"    int h = idx % n_heads;\n"
"    float *base = qkv + tok * stride + h * head_dim;\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) sum += base[i];\n"
"    float mean = sum / (float)head_dim;\n"
"    float var_sum = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) { float d = base[i] - mean; var_sum += d*d; }\n"
"    float inv = rsqrtf(var_sum / (float)head_dim + eps);\n"
"    for (int i = 0; i < head_dim; i++)\n"
"        base[i] = (base[i] - mean) * inv * w[i] + b[i];\n"
"}\n"
"\n"
"/* ---- gemm_f16_f32: MMA m16n8k16 ---- */\n"
"#define GEMM_N_TILE 8\n"
"__global__ void gemm_f16_f32(float *Y, const half_raw *W, const float *X,\n"
"                              const float *bias,\n"
"                              int n_out, int n_in, int n_tok) {\n"
"    extern __shared__ float smem_x[];\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int out_base = blockIdx.x * 256 + warp_id * 64;\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane % 4;\n"
"    int tid = threadIdx.x;\n"
"    if (tok_base >= n_tok) return;\n"
"    float d0[GEMM_N_TILE], d1[GEMM_N_TILE], d2[GEMM_N_TILE], d3[GEMM_N_TILE];\n"
"#pragma unroll\n"
"    for (int i = 0; i < GEMM_N_TILE; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        int srow = tid / 8, scol = (tid % 8) * 2;\n"
"        int grow = tok_base + srow;\n"
"        if (grow < n_tok) {\n"
"            smem_x[srow * 16 + scol] = X[grow * n_in + k + scol];\n"
"            smem_x[srow * 16 + scol + 1] = X[grow * n_in + k + scol + 1];\n"
"        } else {\n"
"            smem_x[srow * 16 + scol] = 0.0f;\n"
"            smem_x[srow * 16 + scol + 1] = 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        unsigned int a0, a1, a2, a3;\n"
"        { float f0 = smem_x[gid * 16 + tid4 * 2];\n"
"          float f1 = smem_x[gid * 16 + tid4 * 2 + 1];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a0) : \"f\"(f0), \"f\"(f1)); }\n"
"        { float f0 = smem_x[gid * 16 + tid4 * 2 + 8];\n"
"          float f1 = smem_x[gid * 16 + tid4 * 2 + 9];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a1) : \"f\"(f0), \"f\"(f1)); }\n"
"        { float f0 = smem_x[(gid + 8) * 16 + tid4 * 2];\n"
"          float f1 = smem_x[(gid + 8) * 16 + tid4 * 2 + 1];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a2) : \"f\"(f0), \"f\"(f1)); }\n"
"        { float f0 = smem_x[(gid + 8) * 16 + tid4 * 2 + 8];\n"
"          float f1 = smem_x[(gid + 8) * 16 + tid4 * 2 + 9];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a3) : \"f\"(f0), \"f\"(f1)); }\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < GEMM_N_TILE; nt++) {\n"
"            int bc = out_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < n_out) {\n"
"                const half_raw *wp = W + (size_t)bc * n_in + k;\n"
"                b0 = *(const unsigned int *)(wp + tid4 * 2);\n"
"                b1 = *(const unsigned int *)(wp + tid4 * 2 + 8);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
"                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int yr0 = tok_base + gid;\n"
"    int yr1 = tok_base + gid + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < GEMM_N_TILE; nt++) {\n"
"        int yc0 = out_base + nt * 8 + tid4 * 2;\n"
"        int yc1 = yc0 + 1;\n"
"        float bv0 = (bias && yc0 < n_out) ? bias[yc0] : 0.0f;\n"
"        float bv1 = (bias && yc1 < n_out) ? bias[yc1] : 0.0f;\n"
"        if (yr0 < n_tok && yc0 < n_out) Y[yr0 * n_out + yc0] = d0[nt] + bv0;\n"
"        if (yr0 < n_tok && yc1 < n_out) Y[yr0 * n_out + yc1] = d1[nt] + bv1;\n"
"        if (yr1 < n_tok && yc0 < n_out) Y[yr1 * n_out + yc0] = d2[nt] + bv0;\n"
"        if (yr1 < n_tok && yc1 < n_out) Y[yr1 * n_out + yc1] = d3[nt] + bv1;\n"
"    }\n"
"}\n"
"\n"
"/* ---- gemm_fp8_f32: FP8 E4M3 MMA m16n8k32 (sm_89+) ---- */\n"
"#if __CUDA_ARCH__ >= 890\n"
"__global__ void gemm_fp8_f32(float *Y, const unsigned char *W, const float *X,\n"
"                              const float *bias, int n_out, int n_in, int n_tok) {\n"
"    extern __shared__ float smem_x[];\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int out_base = blockIdx.x * 256 + warp_id * 64;\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane % 4;\n"
"    int tid = threadIdx.x;\n"
"    if (tok_base >= n_tok) return;\n"
"    float d0[GEMM_N_TILE], d1[GEMM_N_TILE], d2[GEMM_N_TILE], d3[GEMM_N_TILE];\n"
"#pragma unroll\n"
"    for (int i = 0; i < GEMM_N_TILE; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"    for (int k = 0; k < n_in; k += 32) {\n"
"        int srow = tid / 8, scol = (tid % 8) * 4;\n"
"        int grow = tok_base + srow;\n"
"        if (grow < n_tok) {\n"
"            smem_x[srow * 32 + scol]     = X[grow * n_in + k + scol];\n"
"            smem_x[srow * 32 + scol + 1] = X[grow * n_in + k + scol + 1];\n"
"            smem_x[srow * 32 + scol + 2] = X[grow * n_in + k + scol + 2];\n"
"            smem_x[srow * 32 + scol + 3] = X[grow * n_in + k + scol + 3];\n"
"        } else {\n"
"            smem_x[srow * 32 + scol]     = 0.0f;\n"
"            smem_x[srow * 32 + scol + 1] = 0.0f;\n"
"            smem_x[srow * 32 + scol + 2] = 0.0f;\n"
"            smem_x[srow * 32 + scol + 3] = 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        unsigned int a0, a1, a2, a3;\n"
"        { unsigned short lo, hi;\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(lo) : \"f\"(smem_x[gid * 32 + tid4 * 4]), \"f\"(smem_x[gid * 32 + tid4 * 4 + 1]));\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(hi) : \"f\"(smem_x[gid * 32 + tid4 * 4 + 2]), \"f\"(smem_x[gid * 32 + tid4 * 4 + 3]));\n"
"          a0 = (unsigned int)lo | ((unsigned int)hi << 16); }\n"
"        { unsigned short lo, hi;\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(lo) : \"f\"(smem_x[gid * 32 + tid4 * 4 + 16]), \"f\"(smem_x[gid * 32 + tid4 * 4 + 17]));\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(hi) : \"f\"(smem_x[gid * 32 + tid4 * 4 + 18]), \"f\"(smem_x[gid * 32 + tid4 * 4 + 19]));\n"
"          a1 = (unsigned int)lo | ((unsigned int)hi << 16); }\n"
"        { unsigned short lo, hi;\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(lo) : \"f\"(smem_x[(gid + 8) * 32 + tid4 * 4]), \"f\"(smem_x[(gid + 8) * 32 + tid4 * 4 + 1]));\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(hi) : \"f\"(smem_x[(gid + 8) * 32 + tid4 * 4 + 2]), \"f\"(smem_x[(gid + 8) * 32 + tid4 * 4 + 3]));\n"
"          a2 = (unsigned int)lo | ((unsigned int)hi << 16); }\n"
"        { unsigned short lo, hi;\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(lo) : \"f\"(smem_x[(gid + 8) * 32 + tid4 * 4 + 16]), \"f\"(smem_x[(gid + 8) * 32 + tid4 * 4 + 17]));\n"
"          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(hi) : \"f\"(smem_x[(gid + 8) * 32 + tid4 * 4 + 18]), \"f\"(smem_x[(gid + 8) * 32 + tid4 * 4 + 19]));\n"
"          a3 = (unsigned int)lo | ((unsigned int)hi << 16); }\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < GEMM_N_TILE; nt++) {\n"
"            int bc = out_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < n_out) {\n"
"                const unsigned char *wp = W + (size_t)bc * n_in + k;\n"
"                b0 = *(const unsigned int *)(wp + tid4 * 4);\n"
"                b1 = *(const unsigned int *)(wp + tid4 * 4 + 16);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
"                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int yr0 = tok_base + gid;\n"
"    int yr1 = tok_base + gid + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < GEMM_N_TILE; nt++) {\n"
"        int yc0 = out_base + nt * 8 + tid4 * 2;\n"
"        int yc1 = yc0 + 1;\n"
"        float bv0 = (bias && yc0 < n_out) ? bias[yc0] : 0.0f;\n"
"        float bv1 = (bias && yc1 < n_out) ? bias[yc1] : 0.0f;\n"
"        if (yr0 < n_tok && yc0 < n_out) Y[yr0 * n_out + yc0] = d0[nt] + bv0;\n"
"        if (yr0 < n_tok && yc1 < n_out) Y[yr0 * n_out + yc1] = d1[nt] + bv1;\n"
"        if (yr1 < n_tok && yc0 < n_out) Y[yr1 * n_out + yc0] = d2[nt] + bv0;\n"
"        if (yr1 < n_tok && yc1 < n_out) Y[yr1 * n_out + yc1] = d3[nt] + bv1;\n"
"    }\n"
"}\n"
"#endif\n"
"\n"
"/* ---- add_bias_f32 ---- */\n"
"__global__ void add_bias_f32(float *Y, const float *bias, int n_out, int n_tok) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n_out * n_tok) Y[i] += bias[i % n_out];\n"
"}\n"
"\n"
"/* ---- attn_prefill_f32: TC FlashAttention, online softmax ---- */\n"
"__global__ void attn_prefill_f32(float *out, const float *qkv,\n"
"                                  const float *K_t, const float *V_t,\n"
"                                  int n_tok, int dim, int n_heads, int head_dim,\n"
"                                  float scale) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int qb = blockIdx.y * 64 + warp_id * 16;\n"
"    if (qb >= n_tok) return;\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4, tid4 = lane % 4;\n"
"    int dim3 = 3 * dim;\n"
"    const float *kt_h = K_t + h * n_tok * head_dim;\n"
"    const float *vt_h = V_t + h * n_tok * head_dim;\n"
"    int qi0 = qb + gid, qi1 = qb + gid + 8;\n"
"    unsigned int qa0[4], qa1[4], qa2[4], qa3[4];\n"
"#pragma unroll\n"
"    for (int kk = 0; kk < 64; kk += 16) {\n"
"        int ks = kk / 16;\n"
"        int dc = kk + tid4*2;\n"
"        { float f0=(qi0<n_tok)?qkv[qi0*dim3+h*head_dim+dc]:0, f1=(qi0<n_tok)?qkv[qi0*dim3+h*head_dim+dc+1]:0;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa0[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"        { float f0=(qi0<n_tok)?qkv[qi0*dim3+h*head_dim+dc+8]:0, f1=(qi0<n_tok)?qkv[qi0*dim3+h*head_dim+dc+9]:0;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa1[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"        { float f0=(qi1<n_tok)?qkv[qi1*dim3+h*head_dim+dc]:0, f1=(qi1<n_tok)?qkv[qi1*dim3+h*head_dim+dc+1]:0;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa2[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"        { float f0=(qi1<n_tok)?qkv[qi1*dim3+h*head_dim+dc+8]:0, f1=(qi1<n_tok)?qkv[qi1*dim3+h*head_dim+dc+9]:0;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa3[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"    }\n"
"    float m0 = -1e30f, l0 = 0.0f, m1 = -1e30f, l1 = 0.0f;\n"
"    float oc0[8]={0}, oc1[8]={0}, oc2[8]={0}, oc3[8]={0};\n"
"    for (int kv = 0; kv < n_tok; kv += 16) {\n"
"        float s0[2]={0,0}, s1[2]={0,0}, s2[2]={0,0}, s3[2]={0,0};\n"
"#pragma unroll\n"
"        for (int kk = 0; kk < 64; kk += 16) {\n"
"            int ks = kk / 16;\n"
"            unsigned int a0=qa0[ks], a1=qa1[ks], a2=qa2[ks], a3=qa3[ks];\n"
"            for (int nh = 0; nh < 2; nh++) {\n"
"                int ki = kv + nh*8 + gid;\n"
"                unsigned int b0=0, b1=0;\n"
"                if (ki < n_tok) {\n"
"                    const float *kp = kt_h + ki*head_dim + kk;\n"
"                    float kf0=kp[tid4*2], kf1=kp[tid4*2+1];\n"
"                    asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(b0):\"f\"(kf0),\"f\"(kf1));\n"
"                    float kf2=kp[tid4*2+8], kf3=kp[tid4*2+9];\n"
"                    asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(b1):\"f\"(kf2),\"f\"(kf3));\n"
"                }\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    :\"=f\"(s0[nh]),\"=f\"(s1[nh]),\"=f\"(s2[nh]),\"=f\"(s3[nh])\n"
"                    :\"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1),\n"
"                     \"f\"(s0[nh]),\"f\"(s1[nh]),\"f\"(s2[nh]),\"f\"(s3[nh]));\n"
"            }\n"
"        }\n"
"        s0[0]*=scale; s1[0]*=scale; s2[0]*=scale; s3[0]*=scale;\n"
"        s0[1]*=scale; s1[1]*=scale; s2[1]*=scale; s3[1]*=scale;\n"
"        { int c0=kv+tid4*2, c1=c0+1;\n"
"          if(c0>=n_tok){s0[0]=-1e30f;s2[0]=-1e30f;} if(c1>=n_tok){s1[0]=-1e30f;s3[0]=-1e30f;}\n"
"          if(c0+8>=n_tok){s0[1]=-1e30f;s2[1]=-1e30f;} if(c1+8>=n_tok){s1[1]=-1e30f;s3[1]=-1e30f;} }\n"
"        if(qi0>=n_tok){s0[0]=-1e30f;s1[0]=-1e30f;s0[1]=-1e30f;s1[1]=-1e30f;}\n"
"        if(qi1>=n_tok){s2[0]=-1e30f;s3[0]=-1e30f;s2[1]=-1e30f;s3[1]=-1e30f;}\n"
"        float mx0 = fmaxf(fmaxf(s0[0],s1[0]),fmaxf(s0[1],s1[1]));\n"
"        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 1));\n"
"        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, 2));\n"
"        float mn0 = fmaxf(m0, mx0);\n"
"        float al0 = expf(m0 - mn0);\n"
"        l0 *= al0; m0 = mn0;\n"
"        for (int c=0;c<8;c++) { oc0[c]*=al0; oc1[c]*=al0; }\n"
"        s0[0]=expf(s0[0]-mn0); s1[0]=expf(s1[0]-mn0);\n"
"        s0[1]=expf(s0[1]-mn0); s1[1]=expf(s1[1]-mn0);\n"
"        float rs0=s0[0]+s1[0]+s0[1]+s1[1];\n"
"        rs0+=__shfl_xor_sync(0xFFFFFFFF,rs0,1); rs0+=__shfl_xor_sync(0xFFFFFFFF,rs0,2);\n"
"        l0 += rs0;\n"
"        float mx1 = fmaxf(fmaxf(s2[0],s3[0]),fmaxf(s2[1],s3[1]));\n"
"        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 1));\n"
"        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, 2));\n"
"        float mn1 = fmaxf(m1, mx1);\n"
"        float al1 = expf(m1 - mn1);\n"
"        l1 *= al1; m1 = mn1;\n"
"        for (int c=0;c<8;c++) { oc2[c]*=al1; oc3[c]*=al1; }\n"
"        s2[0]=expf(s2[0]-mn1); s3[0]=expf(s3[0]-mn1);\n"
"        s2[1]=expf(s2[1]-mn1); s3[1]=expf(s3[1]-mn1);\n"
"        float rs1=s2[0]+s3[0]+s2[1]+s3[1];\n"
"        rs1+=__shfl_xor_sync(0xFFFFFFFF,rs1,1); rs1+=__shfl_xor_sync(0xFFFFFFFF,rs1,2);\n"
"        l1 += rs1;\n"
"        unsigned int pa0,pa1,pa2,pa3;\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa0):\"f\"(s0[0]),\"f\"(s1[0]));\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa1):\"f\"(s0[1]),\"f\"(s1[1]));\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa2):\"f\"(s2[0]),\"f\"(s3[0]));\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa3):\"f\"(s2[1]),\"f\"(s3[1]));\n"
"        for (int c = 0; c < 8; c++) {\n"
"            int vki0 = kv+tid4*2, vki1 = vki0+1, vki8 = vki0+8, vki9 = vki8+1;\n"
"            unsigned int vb0=0, vb1=0;\n"
"            if (vki1 < n_tok) {\n"
"                float vf0=vt_h[vki0*head_dim+c*8+gid], vf1=vt_h[vki1*head_dim+c*8+gid];\n"
"                asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(vb0):\"f\"(vf0),\"f\"(vf1));\n"
"            } else if (vki0 < n_tok) {\n"
"                float vf0=vt_h[vki0*head_dim+c*8+gid];\n"
"                asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(vb0):\"f\"(vf0),\"f\"(0.0f));\n"
"            }\n"
"            if (vki9 < n_tok) {\n"
"                float vf0=vt_h[vki8*head_dim+c*8+gid], vf1=vt_h[vki9*head_dim+c*8+gid];\n"
"                asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(vb1):\"f\"(vf0),\"f\"(vf1));\n"
"            } else if (vki8 < n_tok) {\n"
"                float vf0=vt_h[vki8*head_dim+c*8+gid];\n"
"                asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(vb1):\"f\"(vf0),\"f\"(0.0f));\n"
"            }\n"
"            asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                :\"=f\"(oc0[c]),\"=f\"(oc1[c]),\"=f\"(oc2[c]),\"=f\"(oc3[c])\n"
"                :\"r\"(pa0),\"r\"(pa1),\"r\"(pa2),\"r\"(pa3),\"r\"(vb0),\"r\"(vb1),\n"
"                 \"f\"(oc0[c]),\"f\"(oc1[c]),\"f\"(oc2[c]),\"f\"(oc3[c]));\n"
"        }\n"
"    }\n"
"    float il0 = (l0>0) ? 1.0f/l0 : 0.0f;\n"
"    float il1 = (l1>0) ? 1.0f/l1 : 0.0f;\n"
"    for (int c = 0; c < 8; c++) {\n"
"        int d0 = c*8+tid4*2, d1 = d0+1;\n"
"        if (qi0<n_tok) { out[qi0*dim+h*head_dim+d0]=oc0[c]*il0; out[qi0*dim+h*head_dim+d1]=oc1[c]*il0; }\n"
"        if (qi1<n_tok) { out[qi1*dim+h*head_dim+d0]=oc2[c]*il1; out[qi1*dim+h*head_dim+d1]=oc3[c]*il1; }\n"
"    }\n"
"}\n"
"\n"
"/* ---- scalar_attn_f32: scalar attention (no MMA), one block per (head, query) ---- */\n"
"__global__ void scalar_attn_f32(float *out, const float *qkv,\n"
"                                 const float *K_t, const float *V_t,\n"
"                                 int n_tok, int dim, int n_heads, int head_dim,\n"
"                                 float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qi = blockIdx.y;\n"
"    if (h >= n_heads || qi >= n_tok) return;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int dim3 = 3 * dim;\n"
"    const float *q = qkv + qi * dim3 + h * head_dim;\n"
"    const float *kt_h = K_t + h * n_tok * head_dim;\n"
"    const float *vt_h = V_t + h * n_tok * head_dim;\n"
"    extern __shared__ float sdata[];\n"
"    float *scores = sdata;\n"            /* [n_tok] */
"    float *rbuf = sdata + n_tok;\n"      /* [nt] for reduction */
"\n"
"    /* Phase 1: Q @ K^T */\n"
"    for (int ki = tid; ki < n_tok; ki += nt) {\n"
"        const float *k = kt_h + ki * head_dim;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) dot += q[d] * k[d];\n"
"        scores[ki] = dot * scale;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 2: Softmax — find max */\n"
"    float lmax = -1e30f;\n"
"    for (int ki = tid; ki < n_tok; ki += nt)\n"
"        lmax = fmaxf(lmax, scores[ki]);\n"
"    rbuf[tid] = lmax;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) rbuf[tid] = fmaxf(rbuf[tid], rbuf[tid+r]);\n"
"        __syncthreads();\n"
"    }\n"
"    float mx = rbuf[0];\n"
"    __syncthreads();\n"
"\n"
"    /* Exp + sum */\n"
"    float lsum = 0.0f;\n"
"    for (int ki = tid; ki < n_tok; ki += nt) {\n"
"        scores[ki] = expf(scores[ki] - mx);\n"
"        lsum += scores[ki];\n"
"    }\n"
"    rbuf[tid] = lsum;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) rbuf[tid] += rbuf[tid+r];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv_sum = (rbuf[0] > 0) ? 1.0f / rbuf[0] : 0.0f;\n"
"    for (int ki = tid; ki < n_tok; ki += nt)\n"
"        scores[ki] *= inv_sum;\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 3: Weighted sum of V */\n"
"    for (int d = tid; d < head_dim; d += nt) {\n"
"        float acc = 0.0f;\n"
"        for (int ki = 0; ki < n_tok; ki++)\n"
"            acc += scores[ki] * vt_h[ki * head_dim + d];\n"
"        out[qi * dim + h * head_dim + d] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* ---- kv_transpose ---- */\n"
"__global__ void kv_transpose(float *K_t, float *V_t, const float *qkv,\n"
"                              int n_tok, int dim, int n_heads, int head_dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * dim;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / dim;\n"
"    int hd_idx = idx % dim;\n"
"    int h = hd_idx / head_dim;\n"
"    int d = hd_idx % head_dim;\n"
"    int dim3 = 3 * dim;\n"
"    int dst_idx = h * n_tok * head_dim + tok * head_dim + d;\n"
"    K_t[dst_idx] = qkv[tok * dim3 + dim + hd_idx];\n"
"    V_t[dst_idx] = qkv[tok * dim3 + 2*dim + hd_idx];\n"
"}\n"
"\n"
"/* ---- gelu_f32 (tanh approx, same as GELU used in DINOv2 & DiT) ---- */\n"
"__global__ void gelu_f32(float *x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        float v = x[i];\n"
"        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v*v*v)));\n"
"    }\n"
"}\n"
"\n"
"/* ---- add_f32 ---- */\n"
"__global__ void add_f32(float *dst, const float *src, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) dst[i] += src[i];\n"
"}\n"
"\n"
"/* ---- resize_normalize: bilinear resize + ImageNet normalize ---- */\n"
"__global__ void resize_normalize(float *dst, const unsigned char *src,\n"
"                                  int src_w, int src_h, int dst_w, int dst_h,\n"
"                                  float mean0, float mean1, float mean2,\n"
"                                  float istd0, float istd1, float istd2) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = dst_h * dst_w;\n"
"    if (idx >= total) return;\n"
"    int oh = idx / dst_w, ow = idx % dst_w;\n"
"    float fy = (dst_h > 1) ? (float)oh * (src_h-1) / (dst_h-1) : 0.0f;\n"
"    float fx = (dst_w > 1) ? (float)ow * (src_w-1) / (dst_w-1) : 0.0f;\n"
"    int y0 = (int)fy, x0 = (int)fx;\n"
"    int y1 = (y0+1 < src_h) ? y0+1 : y0;\n"
"    int x1 = (x0+1 < src_w) ? x0+1 : x0;\n"
"    float dy = fy - y0, dx = fx - x0;\n"
"    float mean[3] = {mean0, mean1, mean2};\n"
"    float istd[3] = {istd0, istd1, istd2};\n"
"    for (int c = 0; c < 3; c++) {\n"
"        float v = (float)src[(y0*src_w+x0)*3+c] * (1-dy)*(1-dx)\n"
"                + (float)src[(y0*src_w+x1)*3+c] * (1-dy)*dx\n"
"                + (float)src[(y1*src_w+x0)*3+c] * dy*(1-dx)\n"
"                + (float)src[(y1*src_w+x1)*3+c] * dy*dx;\n"
"        dst[c * total + idx] = (v / 255.0f - mean[c]) * istd[c];\n"
"    }\n"
"}\n"
"\n"
"/* ---- patch_embed_conv2d ---- */\n"
"__global__ void patch_embed_conv2d(float *out, const float *img, const float *w,\n"
"                                    const float *bias, int gw, int dim, int ps,\n"
"                                    int img_w) {\n"
"    int patch = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    int tok = 1 + patch;\n"
"    for (int co = tid; co < dim; co += blockDim.x) {\n"
"        float sum = bias ? bias[co] : 0.0f;\n"
"        for (int ci = 0; ci < 3; ci++)\n"
"            for (int kh = 0; kh < ps; kh++)\n"
"                for (int kw = 0; kw < ps; kw++)\n"
"                    sum += w[((co*3+ci)*ps+kh)*ps+kw]\n"
"                         * img[ci * img_w * img_w + (py*ps+kh) * img_w + (px*ps+kw)];\n"
"        out[tok * dim + co] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ---- cls_pos_embed ---- */\n"
"__global__ void cls_pos_embed(float *hidden, const float *cls, const float *pos,\n"
"                               int n_tok, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx >= n_tok * dim) return;\n"
"    int t = idx / dim;\n"
"    if (t == 0)\n"
"        hidden[idx] = cls[idx] + pos[idx];\n"
"    else\n"
"        hidden[idx] += pos[idx];\n"
"}\n"
"\n"
"/* ---- bilinear_upsample_f32 ---- */\n"
"__global__ void bilinear_upsample_f32(float *dst, const float *src,\n"
"                                       int C, int Hi, int Wi, int Ho, int Wo) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = C * Ho * Wo;\n"
"    if (idx >= total) return;\n"
"    int c = idx / (Ho * Wo);\n"
"    int rem = idx % (Ho * Wo);\n"
"    int oh = rem / Wo, ow = rem % Wo;\n"
"    float fy = (Ho > 1) ? (float)oh * (Hi-1) / (Ho-1) : 0.0f;\n"
"    float fx = (Wo > 1) ? (float)ow * (Wi-1) / (Wo-1) : 0.0f;\n"
"    int y0 = (int)fy, x0 = (int)fx;\n"
"    int y1 = (y0+1 < Hi) ? y0+1 : y0;\n"
"    int x1 = (x0+1 < Wi) ? x0+1 : x0;\n"
"    float dy = fy - y0, dx = fx - x0;\n"
"    const float *s = src + c * Hi * Wi;\n"
"    dst[idx] = s[y0*Wi+x0]*(1-dy)*(1-dx) + s[y0*Wi+x1]*(1-dy)*dx\n"
"             + s[y1*Wi+x0]*dy*(1-dx) + s[y1*Wi+x1]*dy*dx;\n"
"}\n"
"\n"
"/* ======== DiT-specific kernels ======== */\n"
"\n"
"/* ---- adaLN_modulate: y = norm(x) * (1 + scale) + shift ---- */\n"
"/* shift, scale are per-token (from conditioning), each of size dim. */\n"
"__global__ void adaln_modulate_f32(float *dst, const float *src, const float *w,\n"
"                                    const float *b, const float *shift,\n"
"                                    const float *scale, int dim, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int tok = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    const float *x = src + tok * dim;\n"
"    float *y = dst + tok * dim;\n"
"    const float *sh = shift;  /* [dim], broadcast to all tokens */\n"
"    const float *sc = scale;  /* [dim], broadcast to all tokens */\n"
"    float s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) s += x[i];\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)dim;\n"
"    __syncthreads();\n"
"    s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) { float d = x[i] - mean; s += d*d; }\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)dim + eps);\n"
"    for (int i = tid; i < dim; i += nt)\n"
"        y[i] = ((x[i] - mean) * inv * w[i] + b[i]) * (1.0f + sc[i]) + sh[i];\n"
"}\n"
"\n"
"/* ---- gate_residual_add: dst += gate * src ---- */\n"
"/* gate is per-token [n_tok, dim] */\n"
"__global__ void gate_residual_add_f32(float *dst, const float *src,\n"
"                                       const float *gate, int dim, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        dst[i] += gate[i % dim] * src[i];  /* gate is [dim], broadcast */\n"
"    }\n"
"}\n"
"\n"
"/* ---- silu_f32 ---- */\n"
"__global__ void silu_f32(float *x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        float v = x[i];\n"
"        x[i] = v / (1.0f + expf(-v));\n"
"    }\n"
"}\n"
"\n"
"/* ---- rope_2d_dit_f32: 2D RoPE for DiT (separate Q and K buffers) ---- */\n"
"/* vec: [n_tok, n_heads * head_dim] (Q or K only, NOT interleaved QKV) */\n"
"/* pos_y, pos_x: [n_tok] integer grid positions */\n"
"__global__ void rope_2d_dit_f32(float *vec, const int *pos_y, const int *pos_x,\n"
"                                 int n_tok, int n_heads, int head_dim,\n"
"                                 int stride, float freq_base) {\n"
"    int t = blockIdx.x;\n"
"    if (t >= n_tok) return;\n"
"    int tid = threadIdx.x;\n"
"    int half = head_dim / 2;\n"
"    int quarter = half / 2;\n"
"    int total = n_heads * quarter;\n"
"    if (tid >= total) return;\n"
"    int h = tid / quarter;\n"
"    int j = tid % quarter;\n"
"    float py = (float)pos_y[t];\n"
"    float px = (float)pos_x[t];\n"
"    float *v = vec + t * stride + h * head_dim;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2*j) / (float)half);\n"
"    float ty = py * freq;\n"
"    float cy = cosf(ty), sy = sinf(ty);\n"
"    float v0y = v[j], v1y = v[j + quarter];\n"
"    v[j]           = v0y * cy - v1y * sy;\n"
"    v[j + quarter] = v0y * sy + v1y * cy;\n"
"    float tx = px * freq;\n"
"    float cx = cosf(tx), sx = sinf(tx);\n"
"    float v0x = v[half + j], v1x = v[half + j + quarter];\n"
"    v[half + j]           = v0x * cx - v1x * sx;\n"
"    v[half + j + quarter] = v0x * sx + v1x * cx;\n"
"}\n"
"\n"
"/* ---- unpatchify: reshape [n_tok, patch_size^2] → [1, H, W] ---- */\n"
"/* in: [n_tok, ps*ps], out: [1, gH*ps, gW*ps] */\n"
"__global__ void unpatchify_f32(float *dst, const float *src,\n"
"                                int gH, int gW, int ps) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int H = gH * ps, W = gW * ps;\n"
"    if (idx >= H * W) return;\n"
"    int oh = idx / W, ow = idx % W;\n"
"    int ph = oh / ps, pw = ow / ps;\n"
"    int kh = oh % ps, kw = ow % ps;\n"
"    int tok = ph * gW + pw;\n"
"    dst[idx] = src[tok * ps * ps + kh * ps + kw];\n"
"}\n"
"\n"
"/* ---- concat_4ch: cat([latent, cond], dim=1) for [1,H,W]+[3,H,W] → [4,H,W] ---- */\n"
"__global__ void concat_4ch_f32(float *dst, const float *latent, const float *cond,\n"
"                                int HW) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= 4 * HW) return;\n"
"    int ch = i / HW;\n"
"    int hw = i % HW;\n"
"    dst[i] = (ch == 0) ? latent[hw] : cond[(ch-1) * HW + hw];\n"
"}\n"
"\n"
"/* ---- dit_patch_embed_conv2d: Conv2d(4, dim, k=ps, s=ps) for DiT ---- */\n"
"/* Input: [4, H, W]. Output: [n_tok, dim]. No CLS token. */\n"
"__global__ void dit_patch_embed_conv2d(float *out, const float *img, const float *w,\n"
"                                        const float *bias, int gw, int dim, int ps,\n"
"                                        int img_h, int img_w) {\n"
"    int patch = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    for (int co = tid; co < dim; co += blockDim.x) {\n"
"        float sum = bias ? bias[co] : 0.0f;\n"
"        for (int ci = 0; ci < 4; ci++)\n"
"            for (int kh = 0; kh < ps; kh++)\n"
"                for (int kw = 0; kw < ps; kw++)\n"
"                    sum += w[((co*4+ci)*ps+kh)*ps+kw]\n"
"                         * img[ci * img_h * img_w + (py*ps+kh) * img_w + (px*ps+kw)];\n"
"        out[patch * dim + co] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ---- euler_step: latent = (1-s/T)*pred_x0 + (s/T)*pred_xT ---- */\n"
"/* pred = model output (velocity). Updates latent in-place. */\n"
"__global__ void euler_step_f32(float *latent, const float *pred,\n"
"                                float t_ratio, float s_ratio, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float xt = latent[i];\n"
"    float v = pred[i];\n"
"    float pred_x0 = xt - t_ratio * v;\n"
"    float pred_xT = xt + (1.0f - t_ratio) * v;\n"
"    latent[i] = (1.0f - s_ratio) * pred_x0 + s_ratio * pred_xT;\n"
"}\n"
"\n"
"/* ---- add_scalar_f32: dst[i] += scalar ---- */\n"
"__global__ void add_scalar_f32(float *dst, float scalar, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) dst[i] += scalar;\n"
"}\n"
"\n"
"/* ---- sub_scalar_f32: dst[i] = src[i] - scalar (for cond = img - 0.5) ---- */\n"
"__global__ void sub_scalar_chw_f32(float *dst, const float *src, float scalar, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) dst[i] = src[i] - scalar;\n"
"}\n"
"\n"
"/* ---- pixel_shuffle_2x_f32: [gH*gW, 4*dim] → [2*gH*2*gW, dim] depth-to-space ---- */\n"
"/* Each input token has 4*dim channels, reshaped into 2x2 spatial block of dim channels */\n"
"__global__ void pixel_shuffle_2x_f32(float *dst, const float *src,\n"
"                                       int gH, int gW, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int oH = gH * 2, oW = gW * 2;\n"
"    int total = oH * oW * dim;\n"
"    if (idx >= total) return;\n"
"    int tok_out = idx / dim;\n"
"    int d = idx % dim;\n"
"    int oh = tok_out / oW, ow = tok_out % oW;\n"
"    int ih = oh / 2, iw = ow / 2;\n"
"    int sub = (oh % 2) * 2 + (ow % 2);\n"
"    int tok_in = ih * gW + iw;\n"
"    dst[idx] = src[tok_in * 4 * dim + sub * dim + d];\n"
"}\n"
"\n"
"/* ---- concat_semantic_f32: cat(x, semantics) along dim for semantic fusion ---- */\n"
"/* x: [n_tok, dim], sem: [n_tok, dim] → dst: [n_tok, 2*dim] */\n"
"__global__ void concat_tokens_f32(float *dst, const float *a, const float *b,\n"
"                                    int n_tok, int dim_a, int dim_b) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total_dim = dim_a + dim_b;\n"
"    int total = n_tok * total_dim;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / total_dim;\n"
"    int d = idx % total_dim;\n"
"    dst[idx] = (d < dim_a) ? a[tok * dim_a + d] : b[tok * dim_b + (d - dim_a)];\n"
"}\n"
"\n"
"/* ---- scalar_gemm_f16_f32: simple scalar GEMM for debugging ---- */\n"
"__global__ void scalar_gemm_f16_f32(float *Y, const half_raw *W, const float *X,\n"
"                                     const float *bias,\n"
"                                     int n_out, int n_in, int n_tok) {\n"
"    int out = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int tok = blockIdx.y;\n"
"    if (out >= n_out || tok >= n_tok) return;\n"
"    float sum = 0.0f;\n"
"    for (int k = 0; k < n_in; k++) {\n"
"        half_raw h = W[out * n_in + k];\n"
"        sum += half_to_float(h) * X[tok * n_in + k];\n"
"    }\n"
"    if (bias) sum += bias[out];\n"
"    Y[tok * n_out + out] = sum;\n"
"}\n"
"\n"
"} /* extern C */\n"
;

/* ======================================================================== */
/* Error checking                                                           */
/* ======================================================================== */

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *es = "?"; cuGetErrorString(err, &es); \
        fprintf(stderr, "PPD CUDA error %s:%d: %s (%d)\n", __FILE__, __LINE__, es, (int)err); \
        return -1; \
    } \
} while(0)

#define CHECK_CU_NULL(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *es = "?"; cuGetErrorString(err, &es); \
        fprintf(stderr, "PPD CUDA error %s:%d: %s (%d)\n", __FILE__, __LINE__, es, (int)err); \
        return NULL; \
    } \
} while(0)

/* ======================================================================== */
/* Runner state                                                             */
/* ======================================================================== */

/* Per-block weights for DINOv2 ViT (DA2 semantic encoder) */
typedef struct {
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr attn_qkv_w, attn_qkv_b;
    CUdeviceptr attn_out_w, attn_out_b;
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr ffn_up_w, ffn_up_b;
    CUdeviceptr ffn_down_w, ffn_down_b;
    CUdeviceptr ls1_gamma, ls2_gamma;  /* LayerScale (DINOv2) */
    int qkv_rows, qkv_cols;
    int out_rows, out_cols;
    int ffn_up_rows, ffn_up_cols;
    int ffn_down_rows, ffn_down_cols;
} sem_layer;

/* Per-block weights for DiT */
typedef struct {
    CUdeviceptr ln1_w, ln1_b;             /* norm1 */
    CUdeviceptr ln2_w, ln2_b;             /* norm2 */
    CUdeviceptr attn_qkv_w, attn_qkv_b;  /* qkv (bias=True) */
    CUdeviceptr attn_q_norm_w, attn_q_norm_b; /* q_norm */
    CUdeviceptr attn_k_norm_w, attn_k_norm_b; /* k_norm */
    CUdeviceptr attn_out_w, attn_out_b;   /* proj */
    CUdeviceptr mlp_fc1_w, mlp_fc1_b;     /* fc1 */
    CUdeviceptr mlp_fc2_w, mlp_fc2_b;     /* fc2 */
    CUdeviceptr adaln_w, adaln_b;         /* adaLN_modulation: Linear(dim, 6*dim) */
    int qkv_rows, qkv_cols;
    int out_rows, out_cols;
    int fc1_rows, fc1_cols;
    int fc2_rows, fc2_cols;
} dit_layer;

struct cuda_ppd_runner {
    CUdevice device;
    CUcontext context;
    CUstream stream;
    int verbose;

    CUmodule module;
    /* Kernel function handles */
    CUfunction fn_layernorm_f32;
    CUfunction fn_gemm_f16_f32;
    CUfunction fn_gemm_fp8_f32;
    CUfunction fn_add_bias_f32;
    CUfunction fn_attn_prefill_f32;
    CUfunction fn_kv_transpose;
    CUfunction fn_gelu_f32;
    CUfunction fn_add_f32;
    CUfunction fn_resize_normalize;
    CUfunction fn_patch_embed_conv2d;
    CUfunction fn_cls_pos_embed;
    CUfunction fn_bilinear_upsample_f32;
    CUfunction fn_silu_f32;
    /* DiT-specific */
    CUfunction fn_adaln_modulate_f32;
    CUfunction fn_gate_residual_add_f32;
    CUfunction fn_rope_2d_dit_f32;
    CUfunction fn_unpatchify_f32;
    CUfunction fn_concat_4ch_f32;
    CUfunction fn_dit_patch_embed_conv2d;
    CUfunction fn_euler_step_f32;
    CUfunction fn_add_scalar_f32;
    CUfunction fn_sub_scalar_chw_f32;
    CUfunction fn_pixel_shuffle_2x_f32;
    CUfunction fn_concat_tokens_f32;
    CUfunction fn_qk_norm_f32;
    CUfunction fn_scalar_gemm_f16_f32;
    CUfunction fn_scalar_attn_f32;
    int use_fp8;

    /* ---- DA2 semantic encoder params ---- */
    int sem_n_blocks;       /* 24 */
    int sem_dim;            /* 1024 */
    int sem_n_heads;        /* 16 */
    int sem_head_dim;       /* 64 */
    int sem_ffn_hidden;     /* 4096 */
    int sem_patch_size;     /* 14 */
    float sem_ln_eps;
    CUdeviceptr sem_patch_embed_w, sem_patch_embed_b;
    CUdeviceptr sem_cls_token, sem_pos_embed;
    sem_layer *sem_layers;
    CUdeviceptr sem_norm_w, sem_norm_b;  /* final layernorm */
    float *sem_pos_embed_host;   /* Host F32 copy for interpolation */
    int sem_pos_embed_total;     /* 1 + n_patches_orig */
    int sem_pos_embed_gH, sem_pos_embed_gW; /* original grid dims */

    /* ---- DiT params ---- */
    int dit_n_blocks;       /* 24 */
    int dit_dim;            /* 1024 */
    int dit_n_heads;        /* 16 */
    int dit_head_dim;       /* 64 */
    int dit_ffn_hidden;     /* 4096 */
    int dit_patch_size;     /* 16 (input stride), 8 (output unpatchify) */
    float dit_ln_eps;
    float dit_rope_freq;    /* 100.0 */
    CUdeviceptr dit_x_embed_w, dit_x_embed_b;     /* PatchEmbed Conv2d(4,1024,16,16) */
    CUdeviceptr dit_t_mlp_w1, dit_t_mlp_b1;       /* TimestepEmbedder fc1(256,1024) */
    CUdeviceptr dit_t_mlp_w2, dit_t_mlp_b2;       /* TimestepEmbedder fc2(1024,1024) */
    CUdeviceptr dit_proj_fusion_w[3], dit_proj_fusion_b[3]; /* proj_fusion 3-layer MLP */
    CUdeviceptr dit_final_ln_w, dit_final_ln_b;    /* final_layer adaLN */
    CUdeviceptr dit_final_adaln_w, dit_final_adaln_b; /* final_layer modulation */
    CUdeviceptr dit_final_proj_w, dit_final_proj_b;    /* final_layer linear */
    dit_layer *dit_layers;
    CUdeviceptr dit_const_ones, dit_const_zeros; /* shared constant norm buffers */

    /* Reusable buffers */
    CUdeviceptr d_img_raw;
    size_t d_img_raw_cap;
    CUdeviceptr d_result;
    size_t d_result_cap;

    /* Scratch buffers */
    CUdeviceptr d_hidden, d_hidden2, d_ln_buf;
    CUdeviceptr d_qkv, d_attn_out;
    CUdeviceptr d_ffn_buf, d_proj_out;
    CUdeviceptr d_img_norm;       /* preprocessed image [3, H, W] */
    CUdeviceptr d_semantics;      /* [n_tok, 1024] semantic tokens from DA2 */
    CUdeviceptr d_latent;         /* [1, H, W] diffusion latent */
    CUdeviceptr d_cond;           /* [3, H, W] condition (image - 0.5) */
    CUdeviceptr d_dit_input;      /* [4, H, W] concat(latent, cond) */
    CUdeviceptr d_dit_hidden;     /* [n_tok, dim] DiT hidden state */
    CUdeviceptr d_dit_qkv;
    CUdeviceptr d_dit_attn_out;
    CUdeviceptr d_dit_ffn_buf;
    CUdeviceptr d_dit_proj_out;
    CUdeviceptr d_dit_ln_buf;
    CUdeviceptr d_dit_modulation; /* [n_tok, 6*dim] adaLN output */
    CUdeviceptr d_dit_t_embed;    /* [1, dim] timestep embedding */
    CUdeviceptr d_dit_pos_y, d_dit_pos_x; /* DiT 2D position indices */
    CUdeviceptr d_dit_concat;     /* scratch for semantic fusion concat */
    CUdeviceptr d_dit_pred;       /* [1, H, W] DiT prediction output */

    int loaded;
};

/* ======================================================================== */
/* NVRTC compilation                                                        */
/* ======================================================================== */

static int ppd_compile_kernels(cuda_ppd_runner *r) {
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, r->device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, r->device);
    int sm = major * 10 + minor;

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: compiling kernels for sm_%d ...\n", sm);

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, cuda_ppd_kernel_source, "ppd_kernels.cu", 0, NULL, NULL)
        != NVRTC_SUCCESS)
        return -1;

    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d", sm);
    const char *opts[] = { arch, "--use_fast_math" };
    nvrtcResult nres = nvrtcCompileProgram(prog, 2, opts);

    if (nres != NVRTC_SUCCESS) {
        size_t log_sz;
        nvrtcGetProgramLogSize(prog, &log_sz);
        if (log_sz > 1) {
            char *log = (char *)malloc(log_sz);
            nvrtcGetProgramLog(prog, log);
            fprintf(stderr, "cuda_ppd: NVRTC log:\n%s\n", log);
            free(log);
        }
        nvrtcDestroyProgram(&prog);
        return -1;
    }

    size_t ptx_sz;
    nvrtcGetPTXSize(prog, &ptx_sz);
    char *ptx = (char *)malloc(ptx_sz);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUresult err = cuModuleLoadDataEx(&r->module, ptx, 0, NULL, NULL);
    free(ptx);
    if (err != CUDA_SUCCESS) return -1;

#define GET_FN(name) do { \
    err = cuModuleGetFunction(&r->fn_##name, r->module, #name); \
    if (err != CUDA_SUCCESS) { \
        fprintf(stderr, "cuda_ppd: kernel '%s' not found\n", #name); return -1; } \
} while(0)

    GET_FN(layernorm_f32);
    GET_FN(gemm_f16_f32);
    GET_FN(add_bias_f32);
    GET_FN(attn_prefill_f32);
    GET_FN(kv_transpose);
    GET_FN(gelu_f32);
    GET_FN(add_f32);
    GET_FN(resize_normalize);
    GET_FN(patch_embed_conv2d);
    GET_FN(cls_pos_embed);
    GET_FN(bilinear_upsample_f32);
    GET_FN(silu_f32);
    /* DiT-specific */
    GET_FN(adaln_modulate_f32);
    GET_FN(gate_residual_add_f32);
    GET_FN(rope_2d_dit_f32);
    GET_FN(unpatchify_f32);
    GET_FN(concat_4ch_f32);
    GET_FN(dit_patch_embed_conv2d);
    GET_FN(euler_step_f32);
    GET_FN(add_scalar_f32);
    GET_FN(sub_scalar_chw_f32);
    GET_FN(pixel_shuffle_2x_f32);
    GET_FN(concat_tokens_f32);
    GET_FN(qk_norm_f32);
    GET_FN(scalar_gemm_f16_f32);
    GET_FN(scalar_attn_f32);

    /* FP8 is optional — disabled for PPD to maintain numerical precision
     * (diffusion models are more sensitive to quantization than single-pass models) */
    r->use_fp8 = 0;
    if (sm >= 89) {
        err = cuModuleGetFunction(&r->fn_gemm_fp8_f32, r->module, "gemm_fp8_f32");
        (void)err; /* loaded but not used */
    }
#undef GET_FN

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: kernels compiled OK\n");
    return 0;
}

/* ======================================================================== */
/* Tensor upload helpers                                                    */
/* ======================================================================== */

/* Upload F32 data → GPU F32 */
static CUdeviceptr upload_f32(CUstream stream, const float *data, size_t n) {
    CUdeviceptr d;
    cuMemAlloc(&d, n * sizeof(float));
    cuMemcpyHtoDAsync(d, data, n * sizeof(float), stream);
    return d;
}

/* Upload F16 data → GPU F16 (raw) */
static CUdeviceptr upload_f16_raw(CUstream stream, const void *data, size_t nbytes) {
    CUdeviceptr d;
    cuMemAlloc(&d, nbytes);
    cuMemcpyHtoDAsync(d, data, nbytes, stream);
    return d;
}

/* Upload BF16 data → GPU F32 (convert on CPU) */
static CUdeviceptr upload_bf16_as_f32(CUstream stream, const void *data, size_t n) {
    float *tmp = (float *)malloc(n * sizeof(float));
    const uint16_t *src = (const uint16_t *)data;
    for (size_t i = 0; i < n; i++) {
        uint32_t f32 = (uint32_t)src[i] << 16;
        memcpy(&tmp[i], &f32, 4);
    }
    CUdeviceptr d = upload_f32(stream, tmp, n);
    free(tmp);
    return d;
}

/* Upload any-dtype tensor → GPU F32 */
static CUdeviceptr upload_tensor_as_f32(CUstream stream, const void *data,
                                          const char *dtype, size_t n) {
    if (strcmp(dtype, "F32") == 0) {
        return upload_f32(stream, (const float *)data, n);
    } else if (strcmp(dtype, "BF16") == 0) {
        return upload_bf16_as_f32(stream, data, n);
    } else if (strcmp(dtype, "F16") == 0) {
        /* Convert F16 → F32 on CPU */
        float *tmp = (float *)malloc(n * sizeof(float));
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint16_t h = src[i];
            int s = (h >> 15) & 1;
            int e = (h >> 10) & 0x1f;
            int m = h & 0x3ff;
            float f;
            if (e == 0) f = (float)m * (1.0f / 16777216.0f);
            else if (e == 31) f = m ? NAN : INFINITY;
            else f = (1.0f + m / 1024.0f) * powf(2.0f, (float)(e - 15));
            tmp[i] = s ? -f : f;
        }
        CUdeviceptr d = upload_f32(stream, tmp, n);
        free(tmp);
        return d;
    }
    fprintf(stderr, "cuda_ppd: unsupported dtype %s for F32 upload\n", dtype);
    return 0;
}

/* Upload any-dtype tensor → GPU F16 */
static CUdeviceptr upload_tensor_as_f16(CUstream stream, const void *data,
                                          const char *dtype, size_t n) {
    if (strcmp(dtype, "F16") == 0) {
        return upload_f16_raw(stream, data, n * 2);
    } else if (strcmp(dtype, "F32") == 0) {
        /* Convert F32 → F16 on CPU using simple truncation */
        uint16_t *tmp = (uint16_t *)malloc(n * 2);
        const float *src = (const float *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t fb;
            memcpy(&fb, &src[i], 4);
            int s = (fb >> 31) & 1;
            int e = ((fb >> 23) & 0xff) - 127 + 15;
            int m = (fb >> 13) & 0x3ff;
            if (e <= 0) { e = 0; m = 0; }
            if (e > 30) { e = 31; m = 0; }
            tmp[i] = (uint16_t)((s << 15) | (e << 10) | m);
        }
        CUdeviceptr d = upload_f16_raw(stream, tmp, n * 2);
        free(tmp);
        return d;
    } else if (strcmp(dtype, "BF16") == 0) {
        /* BF16 → F32 → F16 */
        float *tmp32 = (float *)malloc(n * sizeof(float));
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t f32 = (uint32_t)src[i] << 16;
            memcpy(&tmp32[i], &f32, 4);
        }
        uint16_t *tmp16 = (uint16_t *)malloc(n * 2);
        for (size_t i = 0; i < n; i++) {
            uint32_t fb;
            memcpy(&fb, &tmp32[i], 4);
            int s = (fb >> 31) & 1;
            int e = ((fb >> 23) & 0xff) - 127 + 15;
            int m = (fb >> 13) & 0x3ff;
            if (e <= 0) { e = 0; m = 0; }
            if (e > 30) { e = 31; m = 0; }
            tmp16[i] = (uint16_t)((s << 15) | (e << 10) | m);
        }
        free(tmp32);
        CUdeviceptr d = upload_f16_raw(stream, tmp16, n * 2);
        free(tmp16);
        return d;
    }
    fprintf(stderr, "cuda_ppd: unsupported dtype %s for F16 upload\n", dtype);
    return 0;
}

/* Upload any-dtype tensor → GPU FP8 E4M3 */
static CUdeviceptr upload_tensor_as_fp8(CUstream stream, const void *data,
                                          const char *dtype, size_t n) {
    /* First convert to F32, then to FP8 E4M3 */
    float *tmp32 = NULL;
    if (strcmp(dtype, "F32") == 0) {
        tmp32 = (float *)malloc(n * sizeof(float));
        memcpy(tmp32, data, n * sizeof(float));
    } else if (strcmp(dtype, "BF16") == 0) {
        tmp32 = (float *)malloc(n * sizeof(float));
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t f32 = (uint32_t)src[i] << 16;
            memcpy(&tmp32[i], &f32, 4);
        }
    } else if (strcmp(dtype, "F16") == 0) {
        tmp32 = (float *)malloc(n * sizeof(float));
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint16_t h = src[i];
            int s = (h >> 15) & 1;
            int e = (h >> 10) & 0x1f;
            int m = h & 0x3ff;
            float f;
            if (e == 0) f = (float)m * (1.0f / 16777216.0f);
            else if (e == 31) f = m ? NAN : INFINITY;
            else f = (1.0f + m / 1024.0f) * powf(2.0f, (float)(e - 15));
            tmp32[i] = s ? -f : f;
        }
    }
    if (!tmp32) return 0;

    /* F32 → FP8 E4M3 */
    uint8_t *fp8 = (uint8_t *)malloc(n);
    for (size_t i = 0; i < n; i++) {
        float v = tmp32[i];
        uint32_t fb;
        memcpy(&fb, &v, 4);
        int sign = (fb >> 31) & 1;
        int exp = ((fb >> 23) & 0xff) - 127 + 7; /* FP8 E4M3 bias=7 */
        int mant = (fb >> 20) & 0x7;  /* 3-bit mantissa */
        if (exp <= 0) { exp = 0; mant = 0; }
        if (exp > 15) { exp = 15; mant = 6; } /* max non-inf for E4M3 */
        fp8[i] = (uint8_t)((sign << 7) | (exp << 3) | mant);
    }
    free(tmp32);
    CUdeviceptr d;
    cuMemAlloc(&d, n);
    cuMemcpyHtoDAsync(d, fp8, n, stream);
    free(fp8);
    return d;
}

/* ======================================================================== */
/* Weight loading from .pth                                                 */
/* ======================================================================== */

/* Helper: find tensor in pth, upload as F32 */
static CUdeviceptr pth_upload_f32(pth_context *pth, CUstream stream, const char *name) {
    int idx = pth_find(pth, name);
    if (idx < 0) return 0;
    size_t n = pth_nbytes(pth, idx) / pth_dtype_size(pth_dtype(pth, idx));
    return upload_tensor_as_f32(stream, pth_data(pth, idx), pth_dtype(pth, idx), n);
}

/* Helper: find tensor in pth, upload as F16 */
static CUdeviceptr pth_upload_f16(pth_context *pth, CUstream stream, const char *name) {
    int idx = pth_find(pth, name);
    if (idx < 0) return 0;
    size_t n = pth_nbytes(pth, idx) / pth_dtype_size(pth_dtype(pth, idx));
    return upload_tensor_as_f16(stream, pth_data(pth, idx), pth_dtype(pth, idx), n);
}

/* Helper: upload as FP8 or F16 depending on GPU capability */
static CUdeviceptr pth_upload_backbone(pth_context *pth, CUstream stream,
                                         const char *name, int use_fp8) {
    int idx = pth_find(pth, name);
    if (idx < 0) return 0;
    size_t n = pth_nbytes(pth, idx) / pth_dtype_size(pth_dtype(pth, idx));
    if (use_fp8)
        return upload_tensor_as_fp8(stream, pth_data(pth, idx), pth_dtype(pth, idx), n);
    else
        return upload_tensor_as_f16(stream, pth_data(pth, idx), pth_dtype(pth, idx), n);
}

/* Helper: upload a constant F32 buffer (all elements = val) */
static CUdeviceptr upload_constant_f32(CUstream stream, float val, int n) {
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) buf[i] = val;
    CUdeviceptr d;
    cuMemAlloc(&d, (size_t)n * sizeof(float));
    cuMemcpyHtoDAsync(d, buf, (size_t)n * sizeof(float), stream);
    free(buf);
    return d;
}

/* Load DA2 semantic encoder weights from .pth */
static int ppd_load_sem_weights(cuda_ppd_runner *r, const char *path) {
    pth_context *pth = pth_open(path);
    if (!pth) return -1;

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: loading DA2 semantic encoder from %s (%d tensors)\n",
                path, pth_count(pth));

    /* Detect model size from first QKV weight */
    int qkv_idx = pth_find(pth, "blocks.0.attn.qkv.weight");
    if (qkv_idx < 0) {
        /* Try with 'pretrained.' prefix (depth-anything-v2 format) */
        qkv_idx = pth_find(pth, "pretrained.blocks.0.attn.qkv.weight");
    }
    if (qkv_idx < 0) {
        fprintf(stderr, "cuda_ppd: cannot find QKV weight in %s\n", path);
        pth_close(pth);
        return -1;
    }

    /* Detect prefix */
    const char *prefix = "";
    if (strncmp(pth_name(pth, qkv_idx), "pretrained.", 11) == 0)
        prefix = "pretrained.";

    const uint64_t *qkv_shape = pth_shape(pth, qkv_idx);
    r->sem_dim = (int)(qkv_shape[1]); /* [3*dim, dim] → dim = shape[1] */
    r->sem_n_heads = r->sem_dim / 64; /* head_dim = 64 for DINOv2 */
    r->sem_head_dim = 64;
    r->sem_ffn_hidden = r->sem_dim * 4;
    r->sem_patch_size = 14;
    r->sem_ln_eps = 1e-6f;

    /* Count blocks */
    r->sem_n_blocks = 0;
    for (int i = 0; i < pth_count(pth); i++) {
        const char *n = pth_name(pth, i);
        const char *bp = strstr(n, "blocks.");
        if (bp) {
            int blk = atoi(bp + 7);
            if (blk + 1 > r->sem_n_blocks) r->sem_n_blocks = blk + 1;
        }
    }

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: DA2 dim=%d heads=%d blocks=%d\n",
                r->sem_dim, r->sem_n_heads, r->sem_n_blocks);

    r->sem_layers = (sem_layer *)calloc(r->sem_n_blocks, sizeof(sem_layer));

    /* Load per-block weights */
    for (int L = 0; L < r->sem_n_blocks; L++) {
        sem_layer *ly = &r->sem_layers[L];
        char buf[256];

        snprintf(buf, sizeof(buf), "%sblocks.%d.norm1.weight", prefix, L);
        ly->ln1_w = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.norm1.bias", prefix, L);
        ly->ln1_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.weight", prefix, L);
        ly->attn_qkv_w = pth_upload_backbone(pth, r->stream, buf, r->use_fp8);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.bias", prefix, L);
        ly->attn_qkv_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.weight", prefix, L);
        ly->attn_out_w = pth_upload_backbone(pth, r->stream, buf, r->use_fp8);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.bias", prefix, L);
        ly->attn_out_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.norm2.weight", prefix, L);
        ly->ln2_w = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.norm2.bias", prefix, L);
        ly->ln2_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.weight", prefix, L);
        ly->ffn_up_w = pth_upload_backbone(pth, r->stream, buf, r->use_fp8);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.bias", prefix, L);
        ly->ffn_up_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.weight", prefix, L);
        ly->ffn_down_w = pth_upload_backbone(pth, r->stream, buf, r->use_fp8);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.bias", prefix, L);
        ly->ffn_down_b = pth_upload_f32(pth, r->stream, buf);

        /* LayerScale (DINOv2 has these) */
        snprintf(buf, sizeof(buf), "%sblocks.%d.ls1.gamma", prefix, L);
        ly->ls1_gamma = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.ls2.gamma", prefix, L);
        ly->ls2_gamma = pth_upload_f32(pth, r->stream, buf);

        ly->qkv_rows = 3 * r->sem_dim;
        ly->qkv_cols = r->sem_dim;
        ly->out_rows = r->sem_dim;
        ly->out_cols = r->sem_dim;
        ly->ffn_up_rows = r->sem_ffn_hidden;
        ly->ffn_up_cols = r->sem_dim;
        ly->ffn_down_rows = r->sem_dim;
        ly->ffn_down_cols = r->sem_ffn_hidden;
    }

    /* Patch embedding, CLS token, position embedding */
    char buf[256];
    snprintf(buf, sizeof(buf), "%spatch_embed.proj.weight", prefix);
    r->sem_patch_embed_w = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%spatch_embed.proj.bias", prefix);
    r->sem_patch_embed_b = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%scls_token", prefix);
    r->sem_cls_token = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%spos_embed", prefix);
    r->sem_pos_embed = pth_upload_f32(pth, r->stream, buf);

    /* Store host copy of pos_embed for interpolation at predict time */
    {
        int pe_idx = pth_find(pth, buf);
        if (pe_idx >= 0) {
            const uint64_t *pe_shape = pth_shape(pth, pe_idx);
            int pe_ndims = pth_ndims(pth, pe_idx);
            /* Shape: [1, n_tokens, dim] or [n_tokens, dim] */
            int pe_n = (pe_ndims == 3) ? (int)pe_shape[1] : (int)pe_shape[0];
            r->sem_pos_embed_total = pe_n;
            int n_patches = pe_n - 1; /* subtract CLS token */
            int g = (int)sqrtf((float)n_patches);
            /* Verify square grid */
            if (g * g != n_patches) {
                fprintf(stderr, "cuda_ppd: warning: non-square pos_embed grid %d\n", n_patches);
                g = (int)sqrtf((float)n_patches); /* best guess */
            }
            r->sem_pos_embed_gH = g;
            r->sem_pos_embed_gW = g;
            /* Download GPU copy (already uploaded as F32) */
            cuStreamSynchronize(r->stream);
            size_t pe_bytes = (size_t)pe_n * r->sem_dim * sizeof(float);
            r->sem_pos_embed_host = (float *)malloc(pe_bytes);
            cuMemcpyDtoH(r->sem_pos_embed_host, r->sem_pos_embed, pe_bytes);
            if (r->verbose >= 1)
                fprintf(stderr, "cuda_ppd: pos_embed: %d tokens, grid %dx%d\n",
                        pe_n, g, g);
        }
    }

    /* Final layernorm */
    snprintf(buf, sizeof(buf), "%snorm.weight", prefix);
    r->sem_norm_w = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%snorm.bias", prefix);
    r->sem_norm_b = pth_upload_f32(pth, r->stream, buf);

    cuStreamSynchronize(r->stream);
    pth_close(pth);

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: DA2 weights loaded OK\n");
    return 0;
}

/* Load DiT weights from .pth */
static int ppd_load_dit_weights(cuda_ppd_runner *r, const char *path) {
    pth_context *pth = pth_open(path);
    if (!pth) return -1;

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: loading DiT weights from %s (%d tensors)\n",
                path, pth_count(pth));

    /* Set DiT hyperparameters (fixed for PPD) */
    r->dit_n_blocks = 24;
    r->dit_dim = 1024;
    r->dit_n_heads = 16;
    r->dit_head_dim = 64;
    r->dit_ffn_hidden = 4096;
    r->dit_patch_size = 16;
    r->dit_ln_eps = 1e-6f;
    r->dit_rope_freq = 100.0f;

    /* Detect prefix: PPD saves tensors with "dit." prefix */
    const char *dit_pfx = "";
    if (pth_find(pth, "dit.x_embedder.proj.weight") >= 0)
        dit_pfx = "dit.";
    else if (pth_find(pth, "x_embedder.proj.weight") < 0) {
        fprintf(stderr, "cuda_ppd: cannot find x_embedder.proj.weight in %s\n", path);
        pth_close(pth);
        return -1;
    }

    r->dit_layers = (dit_layer *)calloc(r->dit_n_blocks, sizeof(dit_layer));

    char buf[256];

    /* PatchEmbed: x_embedder.proj */
    snprintf(buf, sizeof(buf), "%sx_embedder.proj.weight", dit_pfx);
    r->dit_x_embed_w = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sx_embedder.proj.bias", dit_pfx);
    r->dit_x_embed_b = pth_upload_f32(pth, r->stream, buf);

    /* TimestepEmbedder: t_embedder.mlp */
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.0.weight", dit_pfx);
    r->dit_t_mlp_w1 = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.0.bias", dit_pfx);
    r->dit_t_mlp_b1 = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.2.weight", dit_pfx);
    r->dit_t_mlp_w2 = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%st_embedder.mlp.2.bias", dit_pfx);
    r->dit_t_mlp_b2 = pth_upload_f32(pth, r->stream, buf);

    /* Proj fusion (3 layers) */
    snprintf(buf, sizeof(buf), "%sproj_fusion.0.weight", dit_pfx);
    r->dit_proj_fusion_w[0] = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.0.bias", dit_pfx);
    r->dit_proj_fusion_b[0] = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.2.weight", dit_pfx);
    r->dit_proj_fusion_w[1] = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.2.bias", dit_pfx);
    r->dit_proj_fusion_b[1] = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.4.weight", dit_pfx);
    r->dit_proj_fusion_w[2] = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sproj_fusion.4.bias", dit_pfx);
    r->dit_proj_fusion_b[2] = pth_upload_f32(pth, r->stream, buf);

    /* Final layer */
    snprintf(buf, sizeof(buf), "%sfinal_layer.adaLN_modulation.1.weight", dit_pfx);
    r->dit_final_ln_w = pth_upload_f16(pth, r->stream, buf);  /* F16 for kl_gemm */
    snprintf(buf, sizeof(buf), "%sfinal_layer.adaLN_modulation.1.bias", dit_pfx);
    r->dit_final_ln_b = pth_upload_f32(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sfinal_layer.linear.weight", dit_pfx);
    r->dit_final_proj_w = pth_upload_f16(pth, r->stream, buf);
    snprintf(buf, sizeof(buf), "%sfinal_layer.linear.bias", dit_pfx);
    r->dit_final_proj_b = pth_upload_f32(pth, r->stream, buf);
    /* norm_final uses elementwise_affine=False → constant weights */
    r->dit_final_adaln_w = upload_constant_f32(r->stream, 1.0f, r->dit_dim);
    r->dit_final_adaln_b = upload_constant_f32(r->stream, 0.0f, r->dit_dim);

    /* Constant norm buffers for DiT blocks (elementwise_affine=False) */
    r->dit_const_ones = upload_constant_f32(r->stream, 1.0f, r->dit_dim);
    r->dit_const_zeros = upload_constant_f32(r->stream, 0.0f, r->dit_dim);

    /* Per-block DiT weights */
    for (int L = 0; L < r->dit_n_blocks; L++) {
        dit_layer *ly = &r->dit_layers[L];

        /* norm1/norm2: elementwise_affine=False, use constant buffers */
        ly->ln1_w = r->dit_const_ones;
        ly->ln1_b = r->dit_const_zeros;
        ly->ln2_w = r->dit_const_ones;
        ly->ln2_b = r->dit_const_zeros;

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.weight", dit_pfx, L);
        ly->attn_qkv_w = pth_upload_backbone(pth, r->stream, buf, r->use_fp8);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.qkv.bias", dit_pfx, L);
        ly->attn_qkv_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.q_norm.weight", dit_pfx, L);
        ly->attn_q_norm_w = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.q_norm.bias", dit_pfx, L);
        ly->attn_q_norm_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.k_norm.weight", dit_pfx, L);
        ly->attn_k_norm_w = pth_upload_f32(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.k_norm.bias", dit_pfx, L);
        ly->attn_k_norm_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.weight", dit_pfx, L);
        ly->attn_out_w = pth_upload_backbone(pth, r->stream, buf, r->use_fp8);
        snprintf(buf, sizeof(buf), "%sblocks.%d.attn.proj.bias", dit_pfx, L);
        ly->attn_out_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.weight", dit_pfx, L);
        ly->mlp_fc1_w = pth_upload_backbone(pth, r->stream, buf, r->use_fp8);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc1.bias", dit_pfx, L);
        ly->mlp_fc1_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.weight", dit_pfx, L);
        ly->mlp_fc2_w = pth_upload_backbone(pth, r->stream, buf, r->use_fp8);
        snprintf(buf, sizeof(buf), "%sblocks.%d.mlp.fc2.bias", dit_pfx, L);
        ly->mlp_fc2_b = pth_upload_f32(pth, r->stream, buf);

        snprintf(buf, sizeof(buf), "%sblocks.%d.adaLN_modulation.1.weight", dit_pfx, L);
        ly->adaln_w = pth_upload_f16(pth, r->stream, buf);
        snprintf(buf, sizeof(buf), "%sblocks.%d.adaLN_modulation.1.bias", dit_pfx, L);
        ly->adaln_b = pth_upload_f32(pth, r->stream, buf);

        ly->qkv_rows = 3 * r->dit_dim;
        ly->qkv_cols = r->dit_dim;
        ly->out_rows = r->dit_dim;
        ly->out_cols = r->dit_dim;
        ly->fc1_rows = r->dit_ffn_hidden;
        ly->fc1_cols = r->dit_dim;
        ly->fc2_rows = r->dit_dim;
        ly->fc2_cols = r->dit_ffn_hidden;
    }

    cuStreamSynchronize(r->stream);
    pth_close(pth);

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: DiT weights loaded OK\n");
    return 0;
}

/* ======================================================================== */
/* Public API: init, load, predict, free                                    */
/* ======================================================================== */

cuda_ppd_runner *cuda_ppd_init(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuda_ppd: cuew init failed\n");
        return NULL;
    }
    cuInit(0);

    cuda_ppd_runner *r = (cuda_ppd_runner *)calloc(1, sizeof(cuda_ppd_runner));
    r->verbose = verbose;

    CHECK_CU_NULL(cuDeviceGet(&r->device, device_id));
    CHECK_CU_NULL(cuCtxCreate(&r->context, 0, r->device));
    CHECK_CU_NULL(cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING));

    if (ppd_compile_kernels(r) != 0) {
        fprintf(stderr, "cuda_ppd: kernel compilation failed\n");
        free(r);
        return NULL;
    }

    return r;
}

int cuda_ppd_load_weights(cuda_ppd_runner *r,
                           const char *ppd_pth_path,
                           const char *sem_pth_path) {
    /* Load semantic encoder (DA2) */
    if (ppd_load_sem_weights(r, sem_pth_path) != 0) {
        fprintf(stderr, "cuda_ppd: failed to load semantic encoder weights\n");
        return -1;
    }

    /* Load DiT */
    if (ppd_load_dit_weights(r, ppd_pth_path) != 0) {
        fprintf(stderr, "cuda_ppd: failed to load DiT weights\n");
        return -1;
    }

    r->loaded = 1;
    return 0;
}

/* Kernel launch wrappers */
static void kl_layernorm(cuda_ppd_runner *r, CUdeviceptr dst, CUdeviceptr src,
                          CUdeviceptr w, CUdeviceptr b, int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    cuLaunchKernel(r->fn_layernorm_f32, (unsigned)n_tok, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static void kl_gemm(cuda_ppd_runner *r, CUdeviceptr Y, CUdeviceptr W_f16,
                     CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W_f16, &X, &bias, &n_out, &n_in, &n_tok};
    /* Use scalar GEMM for all sizes — MMA m16n8k16 produces wrong results
       on sm_120 (Blackwell) for this compilation unit. */
    unsigned gx = (unsigned)((n_out + 255) / 256);
    cuLaunchKernel(r->fn_scalar_gemm_f16_f32, gx, (unsigned)n_tok, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_gemm_fp8(cuda_ppd_runner *r, CUdeviceptr Y, CUdeviceptr W_fp8,
                          CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W_fp8, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 255) / 256);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    cuLaunchKernel(r->fn_gemm_fp8_f32, gx, gy, 1, 128, 1, 1,
                   16 * 32 * (unsigned)sizeof(float), r->stream, args, NULL);
}

static void kl_backbone_gemm(cuda_ppd_runner *r, CUdeviceptr Y, CUdeviceptr W,
                               CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    if (r->use_fp8)
        kl_gemm_fp8(r, Y, W, X, bias, n_out, n_in, n_tok);
    else
        kl_gemm(r, Y, W, X, bias, n_out, n_in, n_tok);
}

static void kl_kv_transpose(cuda_ppd_runner *r, CUdeviceptr K_t, CUdeviceptr V_t,
                              CUdeviceptr qkv, int n_tok, int dim, int n_heads, int head_dim) {
    int total = n_tok * dim;
    int grid = (total + 255) / 256;
    void *args[] = {&K_t, &V_t, &qkv, &n_tok, &dim, &n_heads, &head_dim};
    cuLaunchKernel(r->fn_kv_transpose, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_attn(cuda_ppd_runner *r, CUdeviceptr out, CUdeviceptr qkv,
                     CUdeviceptr K_t, CUdeviceptr V_t,
                     int n_tok, int dim, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    void *args[] = {&out, &qkv, &K_t, &V_t, &n_tok, &dim, &n_heads, &head_dim, &scale};
    /* Use scalar attention — MMA attn_prefill_f32 produces wrong results on sm_120 */
    unsigned smem = (unsigned)((n_tok + 256) * sizeof(float));
    cuLaunchKernel(r->fn_scalar_attn_f32,
                   (unsigned)n_heads, (unsigned)n_tok, 1, 256, 1, 1,
                   smem, r->stream, args, NULL);
}

static void kl_gelu(cuda_ppd_runner *r, CUdeviceptr x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    cuLaunchKernel(r->fn_gelu_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_add(cuda_ppd_runner *r, CUdeviceptr dst, CUdeviceptr src, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &n};
    cuLaunchKernel(r->fn_add_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_silu(cuda_ppd_runner *r, CUdeviceptr x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    cuLaunchKernel(r->fn_silu_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_qk_norm(cuda_ppd_runner *r, CUdeviceptr qkv, CUdeviceptr w, CUdeviceptr b,
                         int n_tok, int n_heads, int head_dim, int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    int grid = (total + 255) / 256;
    void *args[] = {&qkv, &w, &b, &n_tok, &n_heads, &head_dim, &stride, &eps};
    cuLaunchKernel(r->fn_qk_norm_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_rope_2d_dit(cuda_ppd_runner *r, CUdeviceptr vec, CUdeviceptr pos_y, CUdeviceptr pos_x,
                             int n_tok, int n_heads, int head_dim, int stride, float freq_base) {
    int quarter = head_dim / 4;
    int threads = n_heads * quarter;
    if (threads > 1024) threads = 1024;
    void *args[] = {&vec, &pos_y, &pos_x, &n_tok, &n_heads, &head_dim, &stride, &freq_base};
    cuLaunchKernel(r->fn_rope_2d_dit_f32, (unsigned)n_tok, 1, 1, (unsigned)threads, 1, 1,
                   0, r->stream, args, NULL);
}



static void kl_adaln_modulate(cuda_ppd_runner *r, CUdeviceptr dst, CUdeviceptr src,
                                CUdeviceptr w, CUdeviceptr b,
                                CUdeviceptr shift, CUdeviceptr scale,
                                int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &shift, &scale, &dim, &eps};
    cuLaunchKernel(r->fn_adaln_modulate_f32, (unsigned)n_tok, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static void kl_gate_residual_add(cuda_ppd_runner *r, CUdeviceptr dst, CUdeviceptr src,
                                   CUdeviceptr gate, int dim, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &gate, &dim, &n};
    cuLaunchKernel(r->fn_gate_residual_add_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_concat_4ch(cuda_ppd_runner *r, CUdeviceptr dst, CUdeviceptr latent,
                            CUdeviceptr cond, int HW) {
    int n = 4 * HW;
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &latent, &cond, &HW};
    cuLaunchKernel(r->fn_concat_4ch_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_unpatchify(cuda_ppd_runner *r, CUdeviceptr dst, CUdeviceptr src,
                            int gH, int gW, int ps) {
    int n = gH * ps * gW * ps;
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &gH, &gW, &ps};
    cuLaunchKernel(r->fn_unpatchify_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_euler_step(cuda_ppd_runner *r, CUdeviceptr latent, CUdeviceptr pred,
                            float t_ratio, float s_ratio, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&latent, &pred, &t_ratio, &s_ratio, &n};
    cuLaunchKernel(r->fn_euler_step_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_add_scalar(cuda_ppd_runner *r, CUdeviceptr dst, float scalar, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &scalar, &n};
    cuLaunchKernel(r->fn_add_scalar_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_concat_tokens(cuda_ppd_runner *r, CUdeviceptr dst, CUdeviceptr a, CUdeviceptr b,
                               int n_tok, int dim_a, int dim_b) {
    int n = n_tok * (dim_a + dim_b);
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &a, &b, &n_tok, &dim_a, &dim_b};
    cuLaunchKernel(r->fn_concat_tokens_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_pixel_shuffle_2x(cuda_ppd_runner *r, CUdeviceptr dst, CUdeviceptr src,
                                  int gH, int gW, int dim) {
    int n = gH * 2 * gW * 2 * dim;
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &gH, &gW, &dim};
    cuLaunchKernel(r->fn_pixel_shuffle_2x_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

/* ======================================================================== */
/* Helper functions for predict                                              */
/* ======================================================================== */

/* Bilinear interpolation of position embeddings on CPU */
/* patch_pe: [orig_gH * orig_gW, dim] → output: [new_gH * new_gW, dim] */
static float *interpolate_pos_embed(const float *patch_pe, int dim,
                                      int orig_gH, int orig_gW,
                                      int new_gH, int new_gW) {
    int new_n = new_gH * new_gW;
    float *out = (float *)malloc((size_t)new_n * dim * sizeof(float));
    for (int oh = 0; oh < new_gH; oh++) {
        for (int ow = 0; ow < new_gW; ow++) {
            float fy = (new_gH > 1) ? (float)oh * (orig_gH - 1) / (new_gH - 1) : 0;
            float fx = (new_gW > 1) ? (float)ow * (orig_gW - 1) / (new_gW - 1) : 0;
            int y0 = (int)fy, x0 = (int)fx;
            int y1 = (y0 + 1 < orig_gH) ? y0 + 1 : y0;
            int x1 = (x0 + 1 < orig_gW) ? x0 + 1 : x0;
            float dy = fy - y0, dx = fx - x0;
            int dst_tok = oh * new_gW + ow;
            for (int d = 0; d < dim; d++) {
                out[dst_tok * dim + d] =
                    patch_pe[(y0 * orig_gW + x0) * dim + d] * (1-dy) * (1-dx) +
                    patch_pe[(y0 * orig_gW + x1) * dim + d] * (1-dy) * dx +
                    patch_pe[(y1 * orig_gW + x0) * dim + d] * dy * (1-dx) +
                    patch_pe[(y1 * orig_gW + x1) * dim + d] * dy * dx;
            }
        }
    }
    return out;
}

/* Generate standard normal random values via Box-Muller */
static void generate_randn(float *buf, int n) {
    for (int i = 0; i < n - 1; i += 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        float r = sqrtf(-2.0f * logf(u1));
        buf[i] = r * cosf(2.0f * (float)M_PI * u2);
        buf[i + 1] = r * sinf(2.0f * (float)M_PI * u2);
    }
    if (n % 2) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)rand() / (float)RAND_MAX;
        buf[n - 1] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    }
}

/* Compute sinusoidal timestep embedding [256] for a given timestep */
static void compute_sinusoidal_embed(float *out, float t, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)(logf(10000.0f)) * (float)i / (float)half);
        float arg = t * freq;
        out[i] = cosf(arg);
        out[half + i] = sinf(arg);
    }
}

/* Generate 2D position indices for a grid */
static void generate_grid_pos(int *pos_y, int *pos_x, int gH, int gW) {
    for (int h = 0; h < gH; h++)
        for (int w = 0; w < gW; w++) {
            int idx = h * gW + w;
            pos_y[idx] = h;
            pos_x[idx] = w;
        }
}

/* ======================================================================== */
/* Predict: full inference pipeline                                          */
/* ======================================================================== */

ppd_result cuda_ppd_predict(cuda_ppd_runner *r, const uint8_t *rgb, int w, int h) {
    ppd_result res = {0};
    if (!r || !r->loaded) return res;

    struct timespec ts;
    double t0, t1;

    /* ── Step 1: Determine processing resolutions ── */
    int proc_h = ((h + 15) / 16) * 16;
    int proc_w = ((w + 15) / 16) * 16;

    /* DA2 semantic encoder resolution */
    int sem_ps = r->sem_patch_size; /* 14 */
    int sem_gH = proc_h / 16;      /* semantic grid height */
    int sem_gW = proc_w / 16;
    int sem_h = sem_gH * sem_ps;    /* DA2 input pixel height */
    int sem_w = sem_gW * sem_ps;
    int sem_np = sem_gH * sem_gW;   /* number of patches */
    int sem_nt = 1 + sem_np;        /* with CLS token */

    /* DiT resolutions */
    int dit_gH_lo = proc_h / 16, dit_gW_lo = proc_w / 16;
    int dit_nt_lo = dit_gH_lo * dit_gW_lo;
    int dit_gH_hi = proc_h / 8, dit_gW_hi = proc_w / 8;
    int dit_nt_hi = dit_gH_hi * dit_gW_hi;

    int sem_dim = r->sem_dim;    /* 1024 */
    int dit_dim = r->dit_dim;    /* 1024 */
    int HW = proc_h * proc_w;

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_ppd: predict %dx%d → proc %dx%d\n", w, h, proc_w, proc_h);
        fprintf(stderr, "cuda_ppd: DA2 grid %dx%d (%d patches), DiT lo %dx%d hi %dx%d\n",
                sem_gH, sem_gW, sem_np, dit_gH_lo, dit_gW_lo, dit_gH_hi, dit_gW_hi);
    }

    /* ── Step 2: Allocate scratch buffers ── */
    int max_nt = dit_nt_hi;  /* largest token count */
    if (sem_nt > max_nt) max_nt = sem_nt;

    /* Shared scratch (reused between DA2 and DiT stages) */
    CUdeviceptr d_hidden = 0, d_ln_buf = 0, d_qkv = 0, d_attn_out = 0;
    CUdeviceptr d_ffn_buf = 0, d_proj_out = 0;
    cuMemAlloc(&d_hidden,   (size_t)max_nt * sem_dim * sizeof(float));
    cuMemAlloc(&d_ln_buf,   (size_t)max_nt * sem_dim * sizeof(float));
    cuMemAlloc(&d_qkv,      (size_t)max_nt * 3 * sem_dim * sizeof(float));
    cuMemAlloc(&d_attn_out, (size_t)max_nt * sem_dim * sizeof(float));
    cuMemAlloc(&d_ffn_buf,  (size_t)max_nt * r->sem_ffn_hidden * sizeof(float));
    cuMemAlloc(&d_proj_out, (size_t)max_nt * sem_dim * sizeof(float));

    /* Image/condition buffers */
    CUdeviceptr d_img_norm = 0, d_cond = 0, d_latent = 0;
    CUdeviceptr d_dit_input = 0, d_dit_pred = 0;
    cuMemAlloc(&d_img_norm,  (size_t)3 * sem_h * sem_w * sizeof(float));
    cuMemAlloc(&d_cond,      (size_t)3 * HW * sizeof(float));
    cuMemAlloc(&d_latent,    (size_t)HW * sizeof(float));
    cuMemAlloc(&d_dit_input, (size_t)4 * HW * sizeof(float));
    cuMemAlloc(&d_dit_pred,  (size_t)HW * sizeof(float));

    /* DiT-specific buffers */
    CUdeviceptr d_semantics = 0;     /* [sem_np, sem_dim] */
    CUdeviceptr d_t_embed = 0;       /* [1, dit_dim] */
    CUdeviceptr d_t_embed_silu = 0;  /* [1, dit_dim] SiLU(t_embed) */
    CUdeviceptr d_modulation = 0;    /* [1, 6*dit_dim] */
    CUdeviceptr d_pos_y = 0, d_pos_x = 0; /* [max(dit_nt_lo, dit_nt_hi)] */
    CUdeviceptr d_concat = 0;        /* for semantic fusion: [dit_nt_lo, 2*dit_dim] */
    CUdeviceptr d_fusion_out = 0;    /* [dit_nt_lo, 4*dit_dim] (reuse d_ffn_buf if big enough) */
    cuMemAlloc(&d_semantics,    (size_t)sem_np * sem_dim * sizeof(float));
    cuMemAlloc(&d_t_embed,      (size_t)dit_dim * sizeof(float));
    cuMemAlloc(&d_t_embed_silu, (size_t)dit_dim * sizeof(float));
    cuMemAlloc(&d_modulation,   (size_t)6 * dit_dim * sizeof(float));
    CUdeviceptr d_sin_buf = 0;
    cuMemAlloc(&d_sin_buf,      256 * sizeof(float));
    cuMemAlloc(&d_pos_y,        (size_t)dit_nt_hi * sizeof(int));
    cuMemAlloc(&d_pos_x,        (size_t)dit_nt_hi * sizeof(int));
    cuMemAlloc(&d_concat,       (size_t)dit_nt_lo * 4 * dit_dim * sizeof(float)); /* 4*dim for intermediate fusion */
    /* d_fusion_out: [dit_nt_lo, 4*dit_dim]. Reuse d_ffn_buf if large enough. */
    int ffn_buf_elems = max_nt * r->sem_ffn_hidden;
    int fusion_elems = dit_nt_lo * 4 * dit_dim;
    if (fusion_elems > ffn_buf_elems) {
        cuMemAlloc(&d_fusion_out, (size_t)fusion_elems * sizeof(float));
    } else {
        d_fusion_out = d_ffn_buf; /* alias */
    }

    /* Upload raw RGB image to GPU */
    size_t img_bytes = (size_t)w * h * 3;
    if (img_bytes > r->d_img_raw_cap) {
        if (r->d_img_raw) cuMemFree(r->d_img_raw);
        cuMemAlloc(&r->d_img_raw, img_bytes);
        r->d_img_raw_cap = img_bytes;
    }
    cuMemcpyHtoDAsync(r->d_img_raw, rgb, img_bytes, r->stream);

    /* ══════════════════════════════════════════════════════════ */
    /*  Step 3: DA2 Semantic Encoder                             */
    /* ══════════════════════════════════════════════════════════ */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    /* 3a. Resize + ImageNet normalize → d_img_norm [3, sem_h, sem_w] */
    {
        int total = sem_h * sem_w;
        int grid = (total + 255) / 256;
        float m0 = 0.485f, m1 = 0.456f, m2 = 0.406f;
        float is0 = 1.0f/0.229f, is1 = 1.0f/0.224f, is2 = 1.0f/0.225f;
        int src_w = w, src_h = h, dst_w = sem_w, dst_h = sem_h;
        void *args[] = {&d_img_norm, &r->d_img_raw, &src_w, &src_h, &dst_w, &dst_h,
                        &m0, &m1, &m2, &is0, &is1, &is2};
        cuLaunchKernel(r->fn_resize_normalize, (unsigned)grid, 1, 1, 256, 1, 1,
                       0, r->stream, args, NULL);
    }

    /* 3b. Patch embedding → d_hidden [1+sem_np, sem_dim] */
    {
        void *args[] = {&d_hidden, &d_img_norm, &r->sem_patch_embed_w, &r->sem_patch_embed_b,
                        &sem_gW, &sem_dim, &sem_ps, &sem_w};
        cuLaunchKernel(r->fn_patch_embed_conv2d, (unsigned)sem_np, 1, 1, 256, 1, 1,
                       0, r->stream, args, NULL);
    }

    /* 3c. CLS token + position embedding (with interpolation if needed) */
    {
        CUdeviceptr d_pos = r->sem_pos_embed;
        CUdeviceptr d_pos_alloc = 0; /* for interpolated pos_embed */

        if (r->sem_pos_embed_host &&
            (sem_gH != r->sem_pos_embed_gH || sem_gW != r->sem_pos_embed_gW)) {
            /* Need interpolation */
            float *cls_pe = r->sem_pos_embed_host;                      /* [dim] */
            float *patch_pe = r->sem_pos_embed_host + r->sem_dim;       /* [orig_np, dim] */
            float *interp = interpolate_pos_embed(patch_pe, r->sem_dim,
                                                    r->sem_pos_embed_gH, r->sem_pos_embed_gW,
                                                    sem_gH, sem_gW);
            /* Build full pos_embed: CLS + interpolated patches */
            size_t pe_bytes = (size_t)sem_nt * r->sem_dim * sizeof(float);
            float *full_pe = (float *)malloc(pe_bytes);
            memcpy(full_pe, cls_pe, (size_t)r->sem_dim * sizeof(float));
            memcpy(full_pe + r->sem_dim, interp, (size_t)sem_np * r->sem_dim * sizeof(float));
            free(interp);

            cuMemAlloc(&d_pos_alloc, pe_bytes);
            cuMemcpyHtoDAsync(d_pos_alloc, full_pe, pe_bytes, r->stream);
            free(full_pe);
            d_pos = d_pos_alloc;

            if (r->verbose >= 2)
                fprintf(stderr, "cuda_ppd: interpolated pos_embed %dx%d → %dx%d\n",
                        r->sem_pos_embed_gH, r->sem_pos_embed_gW, sem_gH, sem_gW);
        }

        int total = sem_nt * sem_dim;
        int grid = (total + 255) / 256;
        void *args[] = {&d_hidden, &r->sem_cls_token, &d_pos, &sem_nt, &sem_dim};
        cuLaunchKernel(r->fn_cls_pos_embed, (unsigned)grid, 1, 1, 256, 1, 1,
                       0, r->stream, args, NULL);

        /* d_pos_alloc freed after backbone completes */
        if (d_pos_alloc) { cuStreamSynchronize(r->stream); cuMemFree(d_pos_alloc); }
    }

    /* 3d. DA2 backbone: 24 ViT blocks (no RoPE, no QK-norm, no LayerScale) */
    {
        int dim = sem_dim;
        int nt = sem_nt;
        int n_heads = r->sem_n_heads;
        int head_dim = r->sem_head_dim;

        for (int L = 0; L < r->sem_n_blocks; L++) {
            sem_layer *ly = &r->sem_layers[L];

            /* LN1 */
            kl_layernorm(r, d_ln_buf, d_hidden, ly->ln1_w, ly->ln1_b, nt, dim);

            /* QKV projection */
            kl_backbone_gemm(r, d_qkv, ly->attn_qkv_w, d_ln_buf, ly->attn_qkv_b,
                             ly->qkv_rows, ly->qkv_cols, nt);

            /* KV transpose + attention */
            {
                CUdeviceptr K_t = d_ffn_buf;
                CUdeviceptr V_t = d_ffn_buf + (size_t)nt * dim * sizeof(float);
                kl_kv_transpose(r, K_t, V_t, d_qkv, nt, dim, n_heads, head_dim);
                kl_attn(r, d_attn_out, d_qkv, K_t, V_t, nt, dim, n_heads, head_dim);
            }

            /* Output projection + LayerScale + residual */
            kl_backbone_gemm(r, d_proj_out, ly->attn_out_w, d_attn_out, ly->attn_out_b,
                             ly->out_rows, ly->out_cols, nt);
            if (ly->ls1_gamma)
                kl_gate_residual_add(r, d_hidden, d_proj_out, ly->ls1_gamma, dim, nt * dim);
            else
                kl_add(r, d_hidden, d_proj_out, nt * dim);

            /* LN2 */
            kl_layernorm(r, d_ln_buf, d_hidden, ly->ln2_w, ly->ln2_b, nt, dim);

            /* FFN: fc1 → GELU → fc2 + LayerScale + residual */
            kl_backbone_gemm(r, d_ffn_buf, ly->ffn_up_w, d_ln_buf, ly->ffn_up_b,
                             ly->ffn_up_rows, ly->ffn_up_cols, nt);
            kl_gelu(r, d_ffn_buf, nt * ly->ffn_up_rows);
            kl_backbone_gemm(r, d_proj_out, ly->ffn_down_w, d_ffn_buf, ly->ffn_down_b,
                             ly->ffn_down_rows, ly->ffn_down_cols, nt);
            if (ly->ls2_gamma)
                kl_gate_residual_add(r, d_hidden, d_proj_out, ly->ls2_gamma, dim, nt * dim);
            else
                kl_add(r, d_hidden, d_proj_out, nt * dim);

        }

        /* Final layernorm */
        kl_layernorm(r, d_hidden, d_hidden, r->sem_norm_w, r->sem_norm_b, nt, dim);
    }

    /* 3e. Extract patch tokens (skip CLS at index 0) → d_semantics [sem_np, sem_dim] */
    cuMemcpyDtoDAsync(d_semantics,
                       d_hidden + (size_t)sem_dim * sizeof(float), /* skip CLS */
                       (size_t)sem_np * sem_dim * sizeof(float), r->stream);

    cuStreamSynchronize(r->stream);
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: DA2 semantic encoder: %.1f ms (%d patches)\n",
                (t1-t0)*1000, sem_np);

    /* ══════════════════════════════════════════════════════════ */
    /*  Step 4: Prepare DiT inputs                               */
    /* ══════════════════════════════════════════════════════════ */

    /* 4a. Condition image: resize to [3, proc_h, proc_w] with values (pixel/255 - 0.5) */
    {
        int total = proc_h * proc_w;
        int grid = (total + 255) / 256;
        /* Use resize_normalize with mean=0.5, istd=1 to get (pixel/255 - 0.5) */
        float m0 = 0.5f, m1 = 0.5f, m2 = 0.5f;
        float is0 = 1.0f, is1 = 1.0f, is2 = 1.0f;
        int src_w = w, src_h = h, dst_w = proc_w, dst_h = proc_h;
        void *args[] = {&d_cond, &r->d_img_raw, &src_w, &src_h, &dst_w, &dst_h,
                        &m0, &m1, &m2, &is0, &is1, &is2};
        cuLaunchKernel(r->fn_resize_normalize, (unsigned)grid, 1, 1, 256, 1, 1,
                       0, r->stream, args, NULL);
    }

    /* 4b. Initialize latent ~ N(0, 1) */
    {
        float *noise = (float *)malloc((size_t)HW * sizeof(float));
        srand(42); /* fixed seed for reproducibility */
        generate_randn(noise, HW);
        cuMemcpyHtoDAsync(d_latent, noise, (size_t)HW * sizeof(float), r->stream);
        free(noise);
    }

    /* ══════════════════════════════════════════════════════════ */
    /*  Step 5: DiT diffusion — 4 Euler steps                   */
    /* ══════════════════════════════════════════════════════════ */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    float timesteps[] = {1000.0f, 750.0f, 500.0f, 250.0f};
    float T = 1000.0f;

    for (int step = 0; step < 4; step++) {
        float t_cur = timesteps[step];
        float t_next = (step < 3) ? timesteps[step + 1] : 0.0f;
        float t_ratio = t_cur / T;
        float s_ratio = t_next / T;

        if (r->verbose >= 2)
            fprintf(stderr, "cuda_ppd: Euler step %d: t=%.0f → s=%.0f\n", step, t_cur, t_next);

        /* 5a. Concat [latent, cond] → d_dit_input [4, proc_h, proc_w] */
        kl_concat_4ch(r, d_dit_input, d_latent, d_cond, HW);

        /* 5b. DiT PatchEmbed: Conv2d(4, dit_dim, k=16, s=16) → [dit_nt_lo, dit_dim] */
        {
            int dit_ps = 16; /* input patch stride */
            void *args[] = {&d_hidden, &d_dit_input, &r->dit_x_embed_w, &r->dit_x_embed_b,
                            &dit_gW_lo, &dit_dim, &dit_ps, &proc_h, &proc_w};
            cuLaunchKernel(r->fn_dit_patch_embed_conv2d, (unsigned)dit_nt_lo, 1, 1, 256, 1, 1,
                           0, r->stream, args, NULL);
        }

        /* 5c. Timestep embedding: sinusoidal(t) → fc1 → SiLU → fc2 → [1, dit_dim] */
        {
            float sin_embed[256];
            compute_sinusoidal_embed(sin_embed, t_cur, 256);
            cuMemcpyHtoDAsync(d_sin_buf, sin_embed, 256 * sizeof(float), r->stream);
            /* fc1: [1, 256] → [1, 1024] */
            kl_gemm(r, d_t_embed, r->dit_t_mlp_w1, d_sin_buf, r->dit_t_mlp_b1, dit_dim, 256, 1);
            /* SiLU */
            kl_silu(r, d_t_embed, dit_dim);
            /* fc2: [1, 1024] → [1, 1024] */
            kl_gemm(r, d_t_embed_silu, r->dit_t_mlp_w2, d_t_embed, r->dit_t_mlp_b2, dit_dim, dit_dim, 1);
            /* Pre-compute SiLU(t_embed) for adaLN conditioning */
            cuMemcpyDtoDAsync(d_t_embed, d_t_embed_silu, (size_t)dit_dim * sizeof(float), r->stream);
            kl_silu(r, d_t_embed_silu, dit_dim);
        }

        /* 5d. Upload low-res position indices for RoPE */
        {
            int *py = (int *)malloc((size_t)dit_nt_lo * sizeof(int));
            int *px = (int *)malloc((size_t)dit_nt_lo * sizeof(int));
            generate_grid_pos(py, px, dit_gH_lo, dit_gW_lo);
            cuMemcpyHtoDAsync(d_pos_y, py, (size_t)dit_nt_lo * sizeof(int), r->stream);
            cuMemcpyHtoDAsync(d_pos_x, px, (size_t)dit_nt_lo * sizeof(int), r->stream);
            free(py); free(px);
        }

        /* 5e. DiT blocks 0-11 (low-res stage, dit_nt_lo tokens) */
        {
            int dim = dit_dim;
            int nt = dit_nt_lo;
            int n_heads = r->dit_n_heads;
            int head_dim = r->dit_head_dim;
            int stride_3dim = 3 * dim;

            for (int L = 0; L < 12; L++) {
                dit_layer *ly = &r->dit_layers[L];

                /* adaLN modulation: Linear(SiLU(t_embed)) → [1, 6*dim] */
                kl_gemm(r, d_modulation, ly->adaln_w, d_t_embed_silu, ly->adaln_b,
                        6 * dim, dim, 1);

                /* Split modulation into 6 parts: shift_msa, scale_msa, gate_msa,
                   shift_mlp, scale_mlp, gate_mlp (each [dim]) */
                CUdeviceptr shift_msa = d_modulation;
                CUdeviceptr scale_msa = d_modulation + (size_t)1 * dim * sizeof(float);
                CUdeviceptr gate_msa  = d_modulation + (size_t)2 * dim * sizeof(float);
                CUdeviceptr shift_mlp = d_modulation + (size_t)3 * dim * sizeof(float);
                CUdeviceptr scale_mlp = d_modulation + (size_t)4 * dim * sizeof(float);
                CUdeviceptr gate_mlp  = d_modulation + (size_t)5 * dim * sizeof(float);

                /* adaLN + norm1 → d_ln_buf */
                kl_adaln_modulate(r, d_ln_buf, d_hidden, ly->ln1_w, ly->ln1_b,
                                  shift_msa, scale_msa, nt, dim);

                /* QKV projection */
                kl_backbone_gemm(r, d_qkv, ly->attn_qkv_w, d_ln_buf, ly->attn_qkv_b,
                                 ly->qkv_rows, ly->qkv_cols, nt);

                /* QK-norm on Q and K */
                kl_qk_norm(r, d_qkv, ly->attn_q_norm_w, ly->attn_q_norm_b,
                           nt, n_heads, head_dim, stride_3dim);
                {
                    CUdeviceptr k_base = d_qkv + (size_t)dim * sizeof(float);
                    kl_qk_norm(r, k_base, ly->attn_k_norm_w, ly->attn_k_norm_b,
                               nt, n_heads, head_dim, stride_3dim);
                }

                /* 2D RoPE on Q and K */
                kl_rope_2d_dit(r, d_qkv, d_pos_y, d_pos_x,
                               nt, n_heads, head_dim, stride_3dim, r->dit_rope_freq);
                {
                    CUdeviceptr k_base = d_qkv + (size_t)dim * sizeof(float);
                    kl_rope_2d_dit(r, k_base, d_pos_y, d_pos_x,
                                   nt, n_heads, head_dim, stride_3dim, r->dit_rope_freq);
                }

                /* KV transpose + attention */
                {
                    CUdeviceptr K_t = d_ffn_buf;
                    CUdeviceptr V_t = d_ffn_buf + (size_t)nt * dim * sizeof(float);
                    kl_kv_transpose(r, K_t, V_t, d_qkv, nt, dim, n_heads, head_dim);
                    kl_attn(r, d_attn_out, d_qkv, K_t, V_t, nt, dim, n_heads, head_dim);
                }

                /* Output projection */
                kl_backbone_gemm(r, d_proj_out, ly->attn_out_w, d_attn_out, ly->attn_out_b,
                                 ly->out_rows, ly->out_cols, nt);

                /* Gate + residual: hidden += gate_msa * proj_out */
                kl_gate_residual_add(r, d_hidden, d_proj_out, gate_msa, dim, nt * dim);

                /* adaLN + norm2 → d_ln_buf */
                kl_adaln_modulate(r, d_ln_buf, d_hidden, ly->ln2_w, ly->ln2_b,
                                  shift_mlp, scale_mlp, nt, dim);

                /* MLP: fc1 → GELU → fc2 */
                kl_backbone_gemm(r, d_ffn_buf, ly->mlp_fc1_w, d_ln_buf, ly->mlp_fc1_b,
                                 ly->fc1_rows, ly->fc1_cols, nt);
                kl_gelu(r, d_ffn_buf, nt * ly->fc1_rows);
                kl_backbone_gemm(r, d_proj_out, ly->mlp_fc2_w, d_ffn_buf, ly->mlp_fc2_b,
                                 ly->fc2_rows, ly->fc2_cols, nt);

                /* Gate + residual: hidden += gate_mlp * proj_out */
                kl_gate_residual_add(r, d_hidden, d_proj_out, gate_mlp, dim, nt * dim);
            }
        }

        /* 5f. Semantic fusion at midpoint */
        {
            int dim = dit_dim;
            int nt_lo = dit_nt_lo;

            /* Concat [hidden, semantics] → d_concat [nt_lo, 2*dim]
             * (DA2 output is already normalized by its final LayerNorm) */
            kl_concat_tokens(r, d_concat, d_hidden, d_semantics, nt_lo, dim, dim);

            /* proj_fusion: 3-layer MLP with SiLU
             * Layer 0: Linear(2*dim, 4*dim) */
            kl_gemm(r, d_fusion_out, r->dit_proj_fusion_w[0], d_concat, r->dit_proj_fusion_b[0],
                    4 * dim, 2 * dim, nt_lo);
            kl_silu(r, d_fusion_out, nt_lo * 4 * dim);
            /* Layer 2: Linear(4*dim, 4*dim) */
            kl_gemm(r, d_concat, r->dit_proj_fusion_w[1], d_fusion_out, r->dit_proj_fusion_b[1],
                    4 * dim, 4 * dim, nt_lo);
            kl_silu(r, d_concat, nt_lo * 4 * dim);
            /* Layer 4: Linear(4*dim, 4*dim) → output for pixel_shuffle */
            kl_gemm(r, d_fusion_out, r->dit_proj_fusion_w[2], d_concat, r->dit_proj_fusion_b[2],
                    4 * dim, 4 * dim, nt_lo);
            /* Pixel shuffle 2×2: [nt_lo, 4*dim] → [nt_hi, dim] */
            kl_pixel_shuffle_2x(r, d_hidden, d_fusion_out, dit_gH_lo, dit_gW_lo, dim);
        }
        /* 5g. Upload high-res position indices for RoPE */
        {
            int *py = (int *)malloc((size_t)dit_nt_hi * sizeof(int));
            int *px = (int *)malloc((size_t)dit_nt_hi * sizeof(int));
            generate_grid_pos(py, px, dit_gH_hi, dit_gW_hi);
            cuMemcpyHtoDAsync(d_pos_y, py, (size_t)dit_nt_hi * sizeof(int), r->stream);
            cuMemcpyHtoDAsync(d_pos_x, px, (size_t)dit_nt_hi * sizeof(int), r->stream);
            free(py); free(px);
        }

        /* 5h. DiT blocks 12-23 (high-res stage, dit_nt_hi tokens) */
        {
            int dim = dit_dim;
            int nt = dit_nt_hi;
            int n_heads = r->dit_n_heads;
            int head_dim = r->dit_head_dim;
            int stride_3dim = 3 * dim;

            for (int L = 12; L < 24; L++) {
                dit_layer *ly = &r->dit_layers[L];

                /* adaLN modulation */
                kl_gemm(r, d_modulation, ly->adaln_w, d_t_embed_silu, ly->adaln_b,
                        6 * dim, dim, 1);

                CUdeviceptr shift_msa = d_modulation;
                CUdeviceptr scale_msa = d_modulation + (size_t)1 * dim * sizeof(float);
                CUdeviceptr gate_msa  = d_modulation + (size_t)2 * dim * sizeof(float);
                CUdeviceptr shift_mlp = d_modulation + (size_t)3 * dim * sizeof(float);
                CUdeviceptr scale_mlp = d_modulation + (size_t)4 * dim * sizeof(float);
                CUdeviceptr gate_mlp  = d_modulation + (size_t)5 * dim * sizeof(float);

                /* adaLN + norm1 */
                kl_adaln_modulate(r, d_ln_buf, d_hidden, ly->ln1_w, ly->ln1_b,
                                  shift_msa, scale_msa, nt, dim);

                /* QKV + QK-norm + RoPE + attention */
                kl_backbone_gemm(r, d_qkv, ly->attn_qkv_w, d_ln_buf, ly->attn_qkv_b,
                                 ly->qkv_rows, ly->qkv_cols, nt);

                kl_qk_norm(r, d_qkv, ly->attn_q_norm_w, ly->attn_q_norm_b,
                           nt, n_heads, head_dim, stride_3dim);
                {
                    CUdeviceptr k_base = d_qkv + (size_t)dim * sizeof(float);
                    kl_qk_norm(r, k_base, ly->attn_k_norm_w, ly->attn_k_norm_b,
                               nt, n_heads, head_dim, stride_3dim);
                }

                kl_rope_2d_dit(r, d_qkv, d_pos_y, d_pos_x,
                               nt, n_heads, head_dim, stride_3dim, r->dit_rope_freq);
                {
                    CUdeviceptr k_base = d_qkv + (size_t)dim * sizeof(float);
                    kl_rope_2d_dit(r, k_base, d_pos_y, d_pos_x,
                                   nt, n_heads, head_dim, stride_3dim, r->dit_rope_freq);
                }

                {
                    CUdeviceptr K_t = d_ffn_buf;
                    CUdeviceptr V_t = d_ffn_buf + (size_t)nt * dim * sizeof(float);
                    kl_kv_transpose(r, K_t, V_t, d_qkv, nt, dim, n_heads, head_dim);
                    kl_attn(r, d_attn_out, d_qkv, K_t, V_t, nt, dim, n_heads, head_dim);
                }

                kl_backbone_gemm(r, d_proj_out, ly->attn_out_w, d_attn_out, ly->attn_out_b,
                                 ly->out_rows, ly->out_cols, nt);
                kl_gate_residual_add(r, d_hidden, d_proj_out, gate_msa, dim, nt * dim);

                /* adaLN + norm2 + MLP */
                kl_adaln_modulate(r, d_ln_buf, d_hidden, ly->ln2_w, ly->ln2_b,
                                  shift_mlp, scale_mlp, nt, dim);

                kl_backbone_gemm(r, d_ffn_buf, ly->mlp_fc1_w, d_ln_buf, ly->mlp_fc1_b,
                                 ly->fc1_rows, ly->fc1_cols, nt);
                kl_gelu(r, d_ffn_buf, nt * ly->fc1_rows);
                kl_backbone_gemm(r, d_proj_out, ly->mlp_fc2_w, d_ffn_buf, ly->mlp_fc2_b,
                                 ly->fc2_rows, ly->fc2_cols, nt);
                kl_gate_residual_add(r, d_hidden, d_proj_out, gate_mlp, dim, nt * dim);
            }
        }

        /* 5i. Final layer: adaLN + Linear(dim, 64) + unpatchify */
        {
            int dim = dit_dim;
            int nt = dit_nt_hi;
            int out_ps = 8; /* output patch size */

            /* Final adaLN modulation → [1, 2*dim] → shift, scale */
            kl_gemm(r, d_modulation, r->dit_final_ln_w, d_t_embed_silu, r->dit_final_ln_b,
                    2 * dim, dim, 1);
            CUdeviceptr final_shift = d_modulation;
            CUdeviceptr final_scale = d_modulation + (size_t)dim * sizeof(float);

            /* adaLN + norm_final → d_ln_buf */
            kl_adaln_modulate(r, d_ln_buf, d_hidden, r->dit_final_adaln_w, r->dit_final_adaln_b,
                              final_shift, final_scale, nt, dim);

            /* Linear(dim, ps*ps) → [nt, 64] */
            int out_dim = out_ps * out_ps; /* 64 */
            kl_gemm(r, d_proj_out, r->dit_final_proj_w, d_ln_buf, r->dit_final_proj_b,
                    out_dim, dim, nt);

            /* Unpatchify: [nt, 64] → [gH_hi*8, gW_hi*8] = [proc_h, proc_w] */
            kl_unpatchify(r, d_dit_pred, d_proj_out, dit_gH_hi, dit_gW_hi, out_ps);
        }

        /* 5j. Euler step: update latent */
        kl_euler_step(r, d_latent, d_dit_pred, t_ratio, s_ratio, HW);
    }

    cuStreamSynchronize(r->stream);
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: DiT diffusion (4 Euler steps): %.1f ms\n", (t1-t0)*1000);

    /* ══════════════════════════════════════════════════════════ */
    /*  Step 6: Output depth                                     */
    /* ══════════════════════════════════════════════════════════ */

    /* depth = latent + 0.5 */
    kl_add_scalar(r, d_latent, 0.5f, HW);

    /* Resize from proc_h × proc_w to original w × h */
    if (proc_h != h || proc_w != w) {
        CUdeviceptr d_depth_resized = 0;
        cuMemAlloc(&d_depth_resized, (size_t)w * h * sizeof(float));
        {
            int n = h * w;
            int grid = (n + 255) / 256;
            int C = 1, Hi = proc_h, Wi = proc_w, Ho = h, Wo = w;
            void *args[] = {&d_depth_resized, &d_latent, &C, &Hi, &Wi, &Ho, &Wo};
            cuLaunchKernel(r->fn_bilinear_upsample_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                           0, r->stream, args, NULL);
        }
        /* Download to host */
        res.width = w;
        res.height = h;
        res.depth = (float *)malloc((size_t)w * h * sizeof(float));
        cuMemcpyDtoH(res.depth, d_depth_resized, (size_t)w * h * sizeof(float));
        cuMemFree(d_depth_resized);
    } else {
        /* Download directly */
        res.width = w;
        res.height = h;
        res.depth = (float *)malloc((size_t)w * h * sizeof(float));
        cuMemcpyDtoH(res.depth, d_latent, (size_t)w * h * sizeof(float));
    }

    /* ── Cleanup scratch buffers ── */
    cuMemFree(d_hidden);
    cuMemFree(d_ln_buf);
    cuMemFree(d_qkv);
    cuMemFree(d_attn_out);
    cuMemFree(d_ffn_buf);
    cuMemFree(d_proj_out);
    cuMemFree(d_img_norm);
    cuMemFree(d_cond);
    cuMemFree(d_latent);
    cuMemFree(d_dit_input);
    cuMemFree(d_dit_pred);
    cuMemFree(d_semantics);
    cuMemFree(d_t_embed);
    cuMemFree(d_t_embed_silu);
    cuMemFree(d_modulation);
    cuMemFree(d_sin_buf);
    cuMemFree(d_pos_y);
    cuMemFree(d_pos_x);
    cuMemFree(d_concat);
    if (d_fusion_out != d_ffn_buf) cuMemFree(d_fusion_out);

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_ppd: predict done (%dx%d)\n", w, h);

    return res;
}

void cuda_ppd_free(cuda_ppd_runner *r) {
    if (!r) return;

    /* Free semantic encoder weights */
    if (r->sem_layers) {
        for (int i = 0; i < r->sem_n_blocks; i++) {
            sem_layer *ly = &r->sem_layers[i];
            if (ly->ln1_w) cuMemFree(ly->ln1_w);
            if (ly->ln1_b) cuMemFree(ly->ln1_b);
            if (ly->attn_qkv_w) cuMemFree(ly->attn_qkv_w);
            if (ly->attn_qkv_b) cuMemFree(ly->attn_qkv_b);
            if (ly->attn_out_w) cuMemFree(ly->attn_out_w);
            if (ly->attn_out_b) cuMemFree(ly->attn_out_b);
            if (ly->ln2_w) cuMemFree(ly->ln2_w);
            if (ly->ln2_b) cuMemFree(ly->ln2_b);
            if (ly->ffn_up_w) cuMemFree(ly->ffn_up_w);
            if (ly->ffn_up_b) cuMemFree(ly->ffn_up_b);
            if (ly->ffn_down_w) cuMemFree(ly->ffn_down_w);
            if (ly->ffn_down_b) cuMemFree(ly->ffn_down_b);
            if (ly->ls1_gamma) cuMemFree(ly->ls1_gamma);
            if (ly->ls2_gamma) cuMemFree(ly->ls2_gamma);
        }
        free(r->sem_layers);
    }
    if (r->sem_patch_embed_w) cuMemFree(r->sem_patch_embed_w);
    if (r->sem_patch_embed_b) cuMemFree(r->sem_patch_embed_b);
    if (r->sem_cls_token) cuMemFree(r->sem_cls_token);
    if (r->sem_pos_embed) cuMemFree(r->sem_pos_embed);
    if (r->sem_norm_w) cuMemFree(r->sem_norm_w);
    if (r->sem_norm_b) cuMemFree(r->sem_norm_b);
    free(r->sem_pos_embed_host);

    /* Free DiT weights */
    if (r->dit_const_ones) cuMemFree(r->dit_const_ones);
    if (r->dit_const_zeros) cuMemFree(r->dit_const_zeros);
    if (r->dit_layers) {
        for (int i = 0; i < r->dit_n_blocks; i++) {
            dit_layer *ly = &r->dit_layers[i];
            /* ln1_w/b, ln2_w/b are shared constant buffers, already freed above */
            if (ly->attn_qkv_w) cuMemFree(ly->attn_qkv_w);
            if (ly->attn_qkv_b) cuMemFree(ly->attn_qkv_b);
            if (ly->attn_q_norm_w) cuMemFree(ly->attn_q_norm_w);
            if (ly->attn_q_norm_b) cuMemFree(ly->attn_q_norm_b);
            if (ly->attn_k_norm_w) cuMemFree(ly->attn_k_norm_w);
            if (ly->attn_k_norm_b) cuMemFree(ly->attn_k_norm_b);
            if (ly->attn_out_w) cuMemFree(ly->attn_out_w);
            if (ly->attn_out_b) cuMemFree(ly->attn_out_b);
            if (ly->mlp_fc1_w) cuMemFree(ly->mlp_fc1_w);
            if (ly->mlp_fc1_b) cuMemFree(ly->mlp_fc1_b);
            if (ly->mlp_fc2_w) cuMemFree(ly->mlp_fc2_w);
            if (ly->mlp_fc2_b) cuMemFree(ly->mlp_fc2_b);
            if (ly->adaln_w) cuMemFree(ly->adaln_w);
            if (ly->adaln_b) cuMemFree(ly->adaln_b);
        }
        free(r->dit_layers);
    }
    if (r->dit_x_embed_w) cuMemFree(r->dit_x_embed_w);
    if (r->dit_x_embed_b) cuMemFree(r->dit_x_embed_b);
    if (r->dit_t_mlp_w1) cuMemFree(r->dit_t_mlp_w1);
    if (r->dit_t_mlp_b1) cuMemFree(r->dit_t_mlp_b1);
    if (r->dit_t_mlp_w2) cuMemFree(r->dit_t_mlp_w2);
    if (r->dit_t_mlp_b2) cuMemFree(r->dit_t_mlp_b2);
    for (int i = 0; i < 3; i++) {
        if (r->dit_proj_fusion_w[i]) cuMemFree(r->dit_proj_fusion_w[i]);
        if (r->dit_proj_fusion_b[i]) cuMemFree(r->dit_proj_fusion_b[i]);
    }
    if (r->dit_final_ln_w) cuMemFree(r->dit_final_ln_w);
    if (r->dit_final_ln_b) cuMemFree(r->dit_final_ln_b);
    if (r->dit_final_proj_w) cuMemFree(r->dit_final_proj_w);
    if (r->dit_final_proj_b) cuMemFree(r->dit_final_proj_b);
    if (r->dit_final_adaln_w) cuMemFree(r->dit_final_adaln_w);
    if (r->dit_final_adaln_b) cuMemFree(r->dit_final_adaln_b);

    /* Free scratch/reusable buffers */
    if (r->d_img_raw) cuMemFree(r->d_img_raw);
    if (r->d_result) cuMemFree(r->d_result);

    if (r->module) cuModuleUnload(r->module);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->context) cuCtxDestroy(r->context);

    free(r);
}
