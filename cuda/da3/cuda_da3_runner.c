/*
 * cuda_da3_runner.c - CUDA DA3 depth estimation via NVRTC-compiled kernels
 *
 * GPU-accelerated approach:
 *   - Preprocessing + patch embedding: CPU
 *   - 12 transformer blocks (backbone): GPU (F16 GEMM, fused attention)
 *   - DPT head (token processing, RefineNet fusion, output convs): GPU
 *
 * Compiles with plain gcc (no nvcc). Uses cuew for dynamic CUDA/NVRTC loading.
 * F16 weights on GPU, F32 compute. Single-stream sequential kernel launches.
 */

/* CPU inference library (for preprocessing) */
#define GGML_DEQUANT_IMPLEMENTATION
#define DEPTH_ANYTHING3_IMPLEMENTATION
#include "../../common/depth_anything3.h"

#include "cuda_da3_runner.h"
#include "../cuew.h"
#include "../cuda_kernels_common.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ======================================================================== */
/* DA3-specific CUDA kernels (compiled at runtime via NVRTC)                */
/* Shared kernels (layernorm, GEMM, attention, etc.) are in                 */
/* cuda_kernels_common.h. This string is concatenated after them.           */
/* ======================================================================== */

static const char *cuda_da3_specific_kernels =
"\n"
"/* ---- DA3: qk_layernorm_f32: per-head LN on Q/K with stride ---- */\n"
"/* stride = distance in floats between same element in consecutive tokens */\n"
"__global__ void qk_layernorm_f32(float *vec, const float *w, const float *b,\n"
"                                   int n_tok, int n_heads, int head_dim,\n"
"                                   int stride, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int idx = blockIdx.x;\n"
"    int t = idx / n_heads, h = idx % n_heads;\n"
"    if (t >= n_tok) return;\n"
"    int tid = threadIdx.x;\n"
"    float *v = vec + t * stride + h * head_dim;\n"
"    float val = (tid < head_dim) ? v[tid] : 0.0f;\n"
"    /* Mean */\n"
"    float s = val;\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid+r];\n"
"        __syncthreads();\n"
"    }\n"
"    float mean = sdata[0] / (float)head_dim;\n"
"    /* Var */\n"
"    float d = val - mean;\n"
"    s = d * d;\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid+r];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv = rsqrtf(sdata[0] / (float)head_dim + eps);\n"
"    if (tid < head_dim)\n"
"        v[tid] = (val - mean) * inv * w[tid] + b[tid];\n"
"}\n"
"\n"
"/* ---- 5. rope_2d_f32: 2D RoPE with stride ---- */\n"
"__global__ void rope_2d_f32(float *vec, const int *pos_y, const int *pos_x,\n"
"                             int n_tok, int n_heads, int head_dim,\n"
"                             int stride, float freq_base) {\n"
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
"    /* Y rotation: first half */\n"
"    float ty = py * freq;\n"
"    float cy = cosf(ty), sy = sinf(ty);\n"
"    float v0y = v[j], v1y = v[j + quarter];\n"
"    v[j]           = v0y * cy - v1y * sy;\n"
"    v[j + quarter] = v0y * sy + v1y * cy;\n"
"    /* X rotation: second half */\n"
"    float tx = px * freq;\n"
"    float cx = cosf(tx), sx = sinf(tx);\n"
"    float v0x = v[half + j], v1x = v[half + j + quarter];\n"
"    v[half + j]           = v0x * cx - v1x * sx;\n"
"    v[half + j + quarter] = v0x * sx + v1x * cx;\n"
"}\n"
"\n"
"/* ---- 7. swiglu_f32: dst = silu(gate) * up ---- */\n"
"__global__ void swiglu_f32(float *dst, const float *gate_up, int hidden, int n_tok) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = hidden * n_tok;\n"
"    if (i >= total) return;\n"
"    int t = i / hidden, j = i % hidden;\n"
"    const float *gu = gate_up + t * 2 * hidden;\n"
"    float g = gu[j];\n"
"    g = g / (1.0f + expf(-g));\n"
"    dst[t * hidden + j] = g * gu[j + hidden];\n"
"}\n"
"\n"
"/* ---- 9. layerscale_add_f32: hidden[i] += proj[i] * gamma[i%dim] ---- */\n"
"__global__ void layerscale_add_f32(float *hidden, const float *proj, const float *gamma,\n"
"                                    int dim, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) hidden[i] += proj[i] * gamma[i % dim];\n"
"}\n"
"\n"
"/* ---- 12. depth_activation ---- */\n"
"__global__ void depth_activation(float *out, int hw) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < hw) {\n"
"        out[i]      = expf(out[i]);\n"
"        out[i + hw] = expf(out[i + hw]) + 1.0f;\n"  /* expp1: exp(x) + 1, per DA3 source */
"    }\n"
"}\n"
"\n"
"/* ---- 14. conv2d_f32 ---- */\n"
"__global__ void conv2d_f32(float *dst, const float *src, const float *weight,\n"
"                            const float *bias, int H, int W, int Ci, int Co,\n"
"                            int kH, int kW, int stride, int pad) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int Ho = (H + 2*pad - kH) / stride + 1;\n"
"    int Wo = (W + 2*pad - kW) / stride + 1;\n"
"    int total = Co * Ho * Wo;\n"
"    if (idx >= total) return;\n"
"    int co = idx / (Ho * Wo);\n"
"    int rem = idx % (Ho * Wo);\n"
"    int oh = rem / Wo, ow = rem % Wo;\n"
"    float sum = bias ? bias[co] : 0.0f;\n"
"    for (int ci = 0; ci < Ci; ci++) {\n"
"        for (int kh = 0; kh < kH; kh++) {\n"
"            int ih = oh * stride - pad + kh;\n"
"            if (ih < 0 || ih >= H) continue;\n"
"            for (int kw = 0; kw < kW; kw++) {\n"
"                int iw = ow * stride - pad + kw;\n"
"                if (iw < 0 || iw >= W) continue;\n"
"                sum += weight[((co*Ci+ci)*kH+kh)*kW+kw] * src[ci*H*W + ih*W + iw];\n"
"            }\n"
"        }\n"
"    }\n"
"    dst[idx] = sum;\n"
"}\n"
"\n"
"/* ---- 15. dpt_cls_concat: extract patches + concat CLS ---- */\n"
"__global__ void dpt_cls_concat(float *dst, const float *src, int np, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = np * 2 * dim;\n"
"    if (idx >= total) return;\n"
"    int p = idx / (2 * dim);\n"
"    int j = idx % (2 * dim);\n"
"    dst[idx] = (j < dim) ? src[(1 + p) * dim + j] : src[j - dim];\n"
"}\n"
"\n"
"/* ---- 17. dpt_tok_to_chw: token-major [np,C] -> spatial CHW [C,H,W] ---- */\n"
"__global__ void dpt_tok_to_chw(float *dst, const float *src, int C, int gH, int gW) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = C * gH * gW;\n"
"    if (idx >= total) return;\n"
"    int c = idx / (gH * gW);\n"
"    int rem = idx % (gH * gW);\n"
"    int p = (rem / gW) * gW + (rem % gW);\n"
"    dst[idx] = src[p * C + c];\n"
"}\n"
"\n"
"/* ---- 22. deconv_scatter_f32: scatter GEMM output to spatial CHW for ConvTranspose2d ---- */\n"
"/* Input: Y[Hi*Wi, kH*kW*Co] (GEMM output). Output: dst[Co, Ho, Wo]. */\n"
"/* For stride-aligned ConvT (kH==stride_h, kW==stride_w): each output pixel has */\n"
"/* exactly 1 source, selected by (oh%stride, ow%stride). */\n"
"__global__ void deconv_scatter_f32(float *dst, const float *Y, const float *bias,\n"
"                                     int Co, int Hi, int Wi, int Ho, int Wo,\n"
"                                     int kH, int kW, int stride_h, int stride_w) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = Co * Ho * Wo;\n"
"    if (idx >= total) return;\n"
"    int co = idx / (Ho * Wo);\n"
"    int rem = idx % (Ho * Wo);\n"
"    int oh = rem / Wo, ow = rem % Wo;\n"
"    int kh = oh % stride_h, kw = ow % stride_w;\n"
"    int ih = oh / stride_h, iw = ow / stride_w;\n"
"    int group = kh * kW + kw;\n"
"    int pos = ih * Wi + iw;\n"
"    int n_groups = kH * kW;\n"
"    float v = Y[pos * (n_groups * Co) + group * Co + co];\n"
"    dst[idx] = v + (bias ? bias[co] : 0.0f);\n"
"}\n"
"\n"
"/* ---- 23. conv_gemm_f16_f32: Conv2d via implicit im2col + MMA GEMM ---- */\n"
"/* dst[Co,Ho,Wo] = W[Co,K] * im2col(src)[K,M] + bias[Co], K=Ci*kH*kW, M=Ho*Wo */\n"
"/* Grid: (ceil(Co/64), ceil(M/64)). 128 threads = 4 warps in M direction. */\n"
"/* Each warp: 16 spatial x 64 output channels (8 MMA). Uses smem for im2col tile. */\n"
"/* Requires K % 16 == 0. */\n"
"#define CONV_N_TILE 8\n"
"__global__ void conv_gemm_f16_f32(float *dst, const float *src, const half_raw *weight,\n"
"                                    const float *bias, int H, int W, int Ci, int Co,\n"
"                                    int kH, int kW, int stride, int pad) {\n"
"    extern __shared__ float smem[];\n"
"    int Ho = (H + 2*pad - kH) / stride + 1;\n"
"    int Wo = (W + 2*pad - kW) / stride + 1;\n"
"    int K = Ci * kH * kW;\n"
"    int M = Ho * Wo;\n"
"    int m_base = blockIdx.y * 64;\n"
"    int n_base = blockIdx.x * 64;\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane % 4;\n"
"    int tid = threadIdx.x;\n"
"    int kHkW = kH * kW;\n"
"\n"
"    float cd0[CONV_N_TILE], cd1[CONV_N_TILE], cd2[CONV_N_TILE], cd3[CONV_N_TILE];\n"
"#pragma unroll\n"
"    for (int i = 0; i < CONV_N_TILE; i++) { cd0[i]=0; cd1[i]=0; cd2[i]=0; cd3[i]=0; }\n"
"\n"
"    for (int k = 0; k < K; k += 16) {\n"
"        /* Cooperative im2col load: 128 threads load 64x16 tile into smem */\n"
"        for (int i = tid; i < 64 * 16; i += 128) {\n"
"            int srow = i / 16;\n"
"            int scol = i % 16;\n"
"            int mi = m_base + srow;\n"
"            int ki = k + scol;\n"
"            float val = 0.0f;\n"
"            if (mi < M && ki < K) {\n"
"                int ci = ki / kHkW;\n"
"                int rem = ki - ci * kHkW;\n"
"                int kh_i = rem / kW;\n"
"                int kw_i = rem - kh_i * kW;\n"
"                int oh = mi / Wo;\n"
"                int ow = mi - oh * Wo;\n"
"                int ih = oh * stride - pad + kh_i;\n"
"                int iw = ow * stride - pad + kw_i;\n"
"                if (ih >= 0 && ih < H && iw >= 0 && iw < W)\n"
"                    val = src[ci * H * W + ih * W + iw];\n"
"            }\n"
"            smem[srow * 16 + scol] = val;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* A fragment from smem (each warp reads its 16 M-rows) */\n"
"        int wm = warp_id * 16;\n"
"        unsigned int ca0, ca1, ca2, ca3;\n"
"        { float f0 = smem[(wm + gid) * 16 + tid4 * 2];\n"
"          float f1 = smem[(wm + gid) * 16 + tid4 * 2 + 1];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(ca0) : \"f\"(f0), \"f\"(f1)); }\n"
"        { float f0 = smem[(wm + gid) * 16 + tid4 * 2 + 8];\n"
"          float f1 = smem[(wm + gid) * 16 + tid4 * 2 + 9];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(ca1) : \"f\"(f0), \"f\"(f1)); }\n"
"        { float f0 = smem[(wm + gid + 8) * 16 + tid4 * 2];\n"
"          float f1 = smem[(wm + gid + 8) * 16 + tid4 * 2 + 1];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(ca2) : \"f\"(f0), \"f\"(f1)); }\n"
"        { float f0 = smem[(wm + gid + 8) * 16 + tid4 * 2 + 8];\n"
"          float f1 = smem[(wm + gid + 8) * 16 + tid4 * 2 + 9];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(ca3) : \"f\"(f0), \"f\"(f1)); }\n"
"\n"
"        /* 8 N-tiles, B from global weight (FP16) */\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < CONV_N_TILE; nt++) {\n"
"            int bc = n_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < Co) {\n"
"                const half_raw *wp = weight + (size_t)bc * K + k;\n"
"                b0 = *(const unsigned int *)(wp + tid4 * 2);\n"
"                b1 = *(const unsigned int *)(wp + tid4 * 2 + 8);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(cd0[nt]), \"=f\"(cd1[nt]), \"=f\"(cd2[nt]), \"=f\"(cd3[nt])\n"
"                : \"r\"(ca0), \"r\"(ca1), \"r\"(ca2), \"r\"(ca3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(cd0[nt]), \"f\"(cd1[nt]), \"f\"(cd2[nt]), \"f\"(cd3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    /* Write output in CHW layout: dst[co * M + mi] */\n"
"    int mr0 = m_base + warp_id * 16 + gid;\n"
"    int mr1 = mr0 + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < CONV_N_TILE; nt++) {\n"
"        int nc0 = n_base + nt * 8 + tid4 * 2;\n"
"        int nc1 = nc0 + 1;\n"
"        float bv0 = (bias && nc0 < Co) ? bias[nc0] : 0.0f;\n"
"        float bv1 = (bias && nc1 < Co) ? bias[nc1] : 0.0f;\n"
"        if (mr0 < M && nc0 < Co) dst[nc0 * M + mr0] = cd0[nt] + bv0;\n"
"        if (mr0 < M && nc1 < Co) dst[nc1 * M + mr0] = cd1[nt] + bv1;\n"
"        if (mr1 < M && nc0 < Co) dst[nc0 * M + mr1] = cd2[nt] + bv0;\n"
"        if (mr1 < M && nc1 < Co) dst[nc1 * M + mr1] = cd3[nt] + bv1;\n"
"    }\n"
"}\n"
"\n"
"/* ---- 24. groupnorm_f32: per-channel spatial normalization ---- */\n"
"/* For GroupNorm(G) where G==C: normalize each channel across H*W. */\n"
"/* Grid: (C, 1), threads: 256. */\n"
"__global__ void groupnorm_f32(float *dst, const float *src, const float *w,\n"
"                                const float *b, int C, int HW, float eps) {\n"
"    int c = blockIdx.x;\n"
"    if (c >= C) return;\n"
"    int tid = threadIdx.x;\n"
"    extern __shared__ float sdata[];\n"
"    float s = 0.0f;\n"
"    for (int i = tid; i < HW; i += blockDim.x)\n"
"        s += src[c * HW + i];\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)HW;\n"
"    s = 0.0f;\n"
"    for (int i = tid; i < HW; i += blockDim.x) {\n"
"        float d = src[c * HW + i] - mean; s += d * d; }\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)HW + eps);\n"
"    float sc = w ? w[c] : 1.0f;\n"
"    float bi = b ? b[c] : 0.0f;\n"
"    for (int i = tid; i < HW; i += blockDim.x)\n"
"        dst[c * HW + i] = (src[c * HW + i] - mean) * inv * sc + bi;\n"
"}\n"
"\n"
"/* Per-position channel LayerNorm: normalize across C channels at each (h,w). */\n"
"/* Data layout: [C, H, W] = src[c * HW + hw]. Grid: ceil(HW/256), threads: 256. */\n"
"__global__ void channel_layernorm_f32(float *dst, const float *src, const float *w,\n"
"                                       const float *b, int C, int HW, float eps) {\n"
"    int hw = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (hw >= HW) return;\n"
"    float mean = 0.0f;\n"
"    for (int c = 0; c < C; c++) mean += src[c * HW + hw];\n"
"    mean /= (float)C;\n"
"    float var = 0.0f;\n"
"    for (int c = 0; c < C; c++) { float d = src[c * HW + hw] - mean; var += d * d; }\n"
"    float inv = rsqrtf(var / (float)C + eps);\n"
"    for (int c = 0; c < C; c++)\n"
"        dst[c * HW + hw] = (src[c * HW + hw] - mean) * inv * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);\n"
"}\n"
"\n"
"} /* extern C */\n"
;

/* Error macros from shared header: CU_CHECK, CU_CHECK_NULL */
#define CHECK_CU CU_CHECK
#define CHECK_CU_NULL CU_CHECK_NULL

/* ======================================================================== */
/* Runner state                                                             */
/* ======================================================================== */

typedef struct {
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr attn_qkv_w, attn_qkv_b;
    CUdeviceptr attn_q_norm_w, attn_q_norm_b;
    CUdeviceptr attn_k_norm_w, attn_k_norm_b;
    CUdeviceptr attn_out_w, attn_out_b;
    CUdeviceptr ls1, ls2;
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr ffn_gate_up_w, ffn_gate_up_b;
    CUdeviceptr ffn_up_w, ffn_up_b;
    CUdeviceptr ffn_down_w, ffn_down_b;
    int has_qk_norm, has_swiglu;
    int qkv_rows, qkv_cols;
    int out_rows, out_cols;
    int ffn_gu_rows, ffn_up_rows, ffn_down_rows;
    int ffn_gu_cols, ffn_up_cols, ffn_down_cols;
} cuda_da3_layer;

typedef struct {
    CUdeviceptr norm_w, norm_b;              /* Head LayerNorm F32 */
    CUdeviceptr proj_w[4], proj_b[4];        /* 1x1 proj: w=F16, b=F32 */
    int proj_rows[4];                        /* oc_val per level */
    CUdeviceptr upsample_0_w, upsample_0_b;  /* ConvT 4x4 s4: w=FP16 transposed [kH*kW*Co,Ci] */
    CUdeviceptr upsample_1_w, upsample_1_b;  /* ConvT 2x2 s2: w=FP16 transposed [kH*kW*Co,Ci] */
    CUdeviceptr downsample_w, downsample_b;   /* Conv 3x3 s2: F16 */
    CUdeviceptr adapter_w[4];                 /* Conv 3x3 no-bias: F32 */
    CUdeviceptr fuse_out_w[4], fuse_out_b[4]; /* 1x1 out_conv: F32 */
    CUdeviceptr fuse_rcu1_c1_w[4], fuse_rcu1_c1_b[4];
    CUdeviceptr fuse_rcu1_c2_w[4], fuse_rcu1_c2_b[4];
    CUdeviceptr fuse_rcu2_c1_w[4], fuse_rcu2_c1_b[4];
    CUdeviceptr fuse_rcu2_c2_w[4], fuse_rcu2_c2_b[4];
    CUdeviceptr neck_w, neck_b;
    CUdeviceptr out_0_w, out_0_b;
    CUdeviceptr out_2_w, out_2_b;
    int has_rcu1[4], has_rcu2[4];
    int out_mid;  /* intermediate channels in output_conv2 (typically 32) */
} dpt_gpu_weights;

struct cuda_da3_runner {
    CUdevice device;
    CUcontext context;
    CUstream stream;
    int verbose;

    CUmodule module;
    CUfunction fn_layernorm_f32;
    CUfunction fn_gemm_f16_f32;
    CUfunction fn_add_bias_f32;
    CUfunction fn_qk_layernorm_f32;
    CUfunction fn_rope_2d_f32;
    CUfunction fn_attn_prefill_f32;
    CUfunction fn_swiglu_f32;
    CUfunction fn_gelu_f32;
    CUfunction fn_layerscale_add_f32;
    CUfunction fn_add_f32;
    CUfunction fn_relu_f32;
    CUfunction fn_depth_activation;
    CUfunction fn_bilinear_upsample_f32;
    CUfunction fn_conv2d_f32;
    CUfunction fn_dpt_cls_concat;
    CUfunction fn_dpt_tok_to_chw;
    CUfunction fn_kv_transpose;
    CUfunction fn_resize_normalize;
    CUfunction fn_patch_embed_conv2d;
    CUfunction fn_cls_pos_embed;
    CUfunction fn_conv_gemm_f16_f32;
    CUfunction fn_deconv_scatter_f32;
    CUfunction fn_gemm_fp8_f32;   /* FP8 E4M3 backbone GEMM (sm_89+) */
    CUfunction fn_groupnorm_f32;  /* GroupNorm for aux output convs */
    CUfunction fn_channel_layernorm_f32;  /* Per-position channel LayerNorm */
    CUfunction fn_silu_f32;       /* SiLU for CameraDec MLP */
    int use_fp8;                   /* 1 if FP8 MMA available */

    /* Model params */
    int n_blocks, dim, n_heads, head_dim, ffn_hidden;
    int patch_size, image_size, grid_h, grid_w, n_patches, n_tokens;
    float ln_eps;
    int rope_start, qk_norm_start;
    int feature_layers[4];
    int use_swiglu;
    int head_features, head_out_channels[4];
    float image_mean[3], image_std[3];

    /* GPU weights */
    CUdeviceptr d_patch_embed_w, d_patch_embed_b;
    CUdeviceptr d_cls_token, d_pos_embed;
    cuda_da3_layer *layers;

    /* Preprocessing buffers */
    CUdeviceptr d_img_norm;   /* [3, target_h, target_w] float */
    CUdeviceptr d_img_raw;    /* reusable raw RGB upload buffer */
    size_t d_img_raw_cap;     /* current capacity in bytes */
    CUdeviceptr d_result;     /* reusable final result buffer */
    size_t d_result_cap;

    /* Scratch buffers */
    CUdeviceptr d_hidden, d_hidden2, d_ln_buf, d_qkv, d_attn_out;
    CUdeviceptr d_ffn_buf, d_ffn_mid, d_proj_out;
    CUdeviceptr d_pos_y, d_pos_x;
    CUdeviceptr d_features[4]; /* saved backbone features */

    /* DPT head GPU weights */
    dpt_gpu_weights dpt_w;

    /* DPT scratch buffers */
    CUdeviceptr d_dpt_cat;       /* [np, 2*dim] or fusion scratch */
    CUdeviceptr d_dpt_ln;        /* [np, 2*dim] or fusion scratch */
    CUdeviceptr d_dpt_proj;      /* [np, max_oc] */
    CUdeviceptr d_dpt_chw;       /* [max_oc, gh, gw] */
    CUdeviceptr d_dpt_spatial[4]; /* per-level after resize */
    CUdeviceptr d_dpt_adapted[4]; /* per-level after adapter */
    CUdeviceptr d_dpt_fused;     /* [feat, max_h, max_w] for fusion */
    CUdeviceptr d_dpt_tmp;       /* scratch for RCU/bilinear */
    CUdeviceptr d_dpt_tmp2;      /* scratch for RCU conv mid */
    CUdeviceptr d_dpt_out;       /* [2, 148, 148] final output */

    /* CameraDec weights (Phase 1: pose estimation) */
    struct {
        CUdeviceptr backbone_norm_w, backbone_norm_b;   /* F32 */
        CUdeviceptr mlp_w[2], mlp_b[2];                /* F16, [dim*2, dim*2] */
        CUdeviceptr fc_t_w, fc_t_b;                    /* F32 [3, dim*2] (tiny, CPU matmul) */
        CUdeviceptr fc_qvec_w, fc_qvec_b;              /* F32 [4, dim*2] */
        CUdeviceptr fc_fov_w, fc_fov_b;                /* F32 [2, dim*2] */
        int mlp_dim;                                    /* dim*2 */
        int loaded;
    } cam_dec;

    /* CameraEnc weights (Phase 3: pose conditioning) */
    struct {
        CUdeviceptr fc1_w, fc1_b, fc2_w, fc2_b;        /* F16/FP8, pose MLP 9→dim→dim*2 */
        cuda_da3_layer *trunk;                          /* 4 transformer blocks */
        int n_trunk_blocks;
        CUdeviceptr trunk_norm_w, trunk_norm_b;         /* F32 */
        CUdeviceptr token_norm_w, token_norm_b;         /* F32 */
        int trunk_dim;                                  /* typically dim */
        int loaded;
    } cam_enc;

    /* DPT Aux Branch weights (Phase 2: rays + sky seg) */
    struct {
        /* Aux RefineNet (same structure as main fuse weights) */
        CUdeviceptr fuse_out_w[4], fuse_out_b[4];
        CUdeviceptr fuse_rcu1_c1_w[4], fuse_rcu1_c1_b[4];
        CUdeviceptr fuse_rcu1_c2_w[4], fuse_rcu1_c2_b[4];
        CUdeviceptr fuse_rcu2_c1_w[4], fuse_rcu2_c1_b[4];
        CUdeviceptr fuse_rcu2_c2_w[4], fuse_rcu2_c2_b[4];
        int has_rcu1[4], has_rcu2[4];
        /* Per-level output conv chains: output_conv1_aux (5 Conv2d each, F32 weights) */
        CUdeviceptr oc1_w[4][5], oc1_b[4][5];
        int oc1_ci[4][5], oc1_co[4][5];
        int oc1_count[4];
        /* output_conv2_aux: Conv2d(128,32,3) + GroupNorm(32) + Conv2d(32,7,1) */
        CUdeviceptr oc2_conv_w[4], oc2_conv_b[4];
        CUdeviceptr oc2_gn_w[4], oc2_gn_b[4];
        CUdeviceptr oc2_out_w[4], oc2_out_b[4];
        int loaded;
    } dpt_aux;

    /* DPT Aux scratch buffers */
    CUdeviceptr d_aux_out[4];        /* per-level 7-channel output [7, sp_h, sp_w] */
    CUdeviceptr d_aux_scratch;       /* scratch for aux output conv chains */

    /* GSDPT weights (Phase 4: 3D Gaussian estimation) */
    struct {
        dpt_gpu_weights dpt;                           /* standard DPT weights */
        CUdeviceptr merger_w[3], merger_b[3];          /* images_merger Conv2d, F32 weights */
        int merger_ci[3], merger_co[3];                /* 3→32→64→128 */
        int gs_out_channels;                           /* 38 */
        int loaded;
    } gsdpt;

    /* GSDPT scratch buffers */
    CUdeviceptr d_gs_merged;         /* images_merger output [128, mg_h, mg_w] */
    CUdeviceptr d_gs_out;            /* [38, fh, fw] gaussian output */
    int gs_merger_h, gs_merger_w;    /* merger output spatial dims */

    /* Nested metric model (Phase 6) */
    struct {
        int n_blocks, dim, n_heads, head_dim, ffn_hidden;
        int feature_layers[4], rope_start, qk_norm_start;
        int use_swiglu;
        CUdeviceptr d_patch_embed_w, d_patch_embed_b;
        CUdeviceptr d_cls_token, d_pos_embed;
        cuda_da3_layer *layers;
        dpt_gpu_weights dpt_w;
        CUdeviceptr d_features[4];
        int loaded;
    } metric;

    /* CPU model for preprocessing */
    da3_model *cpu_model;

    /* Host output */
    float *h_output;
    int loaded;
};

/* ======================================================================== */
/* NVRTC compilation                                                        */
/* ======================================================================== */

static int da3_compile_kernels(cuda_da3_runner *r) {
    /* Concatenate shared + DA3-specific kernel source */
    size_t len1 = strlen(cuda_kernels_common_src);
    size_t len2 = strlen(cuda_da3_specific_kernels);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, cuda_kernels_common_src, len1);
    memcpy(full_src + len1, cuda_da3_specific_kernels, len2 + 1);

    int sm = cu_compile_kernels(&r->module, r->device, full_src,
                                 "da3_kernels.cu", r->verbose, "cuda_da3");
    free(full_src);
    if (sm < 0) return -1;

    CUresult err;
#define GET_FN(name) do { \
    err = cuModuleGetFunction(&r->fn_##name, r->module, #name); \
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_da3: kernel '%s' not found\n", #name); return -1; } \
} while(0)

    /* Shared kernels (from cuda_kernels_common.h) */
    GET_FN(layernorm_f32);
    GET_FN(gemm_f16_f32);
    GET_FN(add_bias_f32);
    GET_FN(attn_prefill_f32);
    GET_FN(gelu_f32);
    GET_FN(add_f32);
    GET_FN(relu_f32);
    GET_FN(bilinear_upsample_f32);
    GET_FN(kv_transpose);
    GET_FN(resize_normalize);
    GET_FN(patch_embed_conv2d);
    GET_FN(cls_pos_embed);
    GET_FN(silu_f32);

    /* DA3-specific kernels */
    GET_FN(qk_layernorm_f32);
    GET_FN(rope_2d_f32);
    GET_FN(swiglu_f32);
    GET_FN(layerscale_add_f32);
    GET_FN(depth_activation);
    GET_FN(conv2d_f32);
    GET_FN(dpt_cls_concat);
    GET_FN(dpt_tok_to_chw);
    GET_FN(deconv_scatter_f32);
    GET_FN(conv_gemm_f16_f32);
    GET_FN(groupnorm_f32);
    GET_FN(channel_layernorm_f32);

    /* FP8 E4M3 GEMM: only available on sm_89+ (Ada/Blackwell) */
    r->use_fp8 = (sm >= 89);
    if (r->use_fp8) {
        GET_FN(gemm_fp8_f32);
    }

#undef GET_FN

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_da3: %d kernels compiled (fp8=%d)\n",
                r->use_fp8 ? 26 : 25, r->use_fp8);
    return 0;
}

/* ======================================================================== */
/* Upload tensor to GPU                                                     */
/* ======================================================================== */

#define upload_tensor_raw cu_upload_raw

static CUdeviceptr upload_tensor_f32(const qtensor *t) {
    if (!t->data) return 0;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else {
        memset(buf, 0, (size_t)n * sizeof(float));
    }
    CUdeviceptr d;
    if (cuMemAlloc(&d, (size_t)n * sizeof(float)) != CUDA_SUCCESS) { free(buf); return 0; }
    cuMemcpyHtoD(d, buf, (size_t)n * sizeof(float));
    free(buf);
    return d;
}

static CUdeviceptr upload_tensor_f16(const qtensor *t) {
    if (!t->data || t->type != GGML_TYPE_F16) return upload_tensor_f32(t);
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    return upload_tensor_raw(t->data, (size_t)n * 2);
}

/* FP32 -> FP16 conversion (from shared header) */
#define fp32_to_fp16_raw cu_f32_to_f16

/* Upload conv weight to GPU as FP16, padding K to next multiple of 16.
 * Weight layout: [Co, Ci*kH*kW] row-major. If K%16!=0, pads each row with zeros.
 * Returns padded K through *out_K_padded (or K if already aligned). */
/* Upload tensor to GPU as FP16 (converts F32 -> FP16 on CPU if needed) */
static CUdeviceptr upload_tensor_f32_as_f16(const qtensor *t) {
    if (!t->data) return 0;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    if (t->type == GGML_TYPE_F16)
        return upload_tensor_raw(t->data, (size_t)n * 2);
    uint16_t *buf = (uint16_t *)malloc((size_t)n * 2);
    if (t->type == GGML_TYPE_F32) {
        const float *src = (const float *)t->data;
        for (int i = 0; i < n; i++) buf[i] = fp32_to_fp16_raw(src[i]);
    } else {
        memset(buf, 0, (size_t)n * 2);
    }
    CUdeviceptr d;
    if (cuMemAlloc(&d, (size_t)n * 2) != CUDA_SUCCESS) { free(buf); return 0; }
    cuMemcpyHtoD(d, buf, (size_t)n * 2);
    free(buf);
    return d;
}

/* FP32 -> FP8 E4M3 conversion (from shared header) */
#define f32_to_fp8_e4m3 cu_f32_to_fp8_e4m3

/* Upload tensor to GPU as FP8 E4M3 (1 byte/element) */
static CUdeviceptr upload_tensor_fp8_e4m3(const qtensor *t) {
    if (!t->data) return 0;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = f32_to_fp8_e4m3(ggml_fp16_to_fp32(src[i]));
    } else if (t->type == GGML_TYPE_F32) {
        const float *src = (const float *)t->data;
        for (int i = 0; i < n; i++) buf[i] = f32_to_fp8_e4m3(src[i]);
    } else {
        memset(buf, 0, (size_t)n);
    }
    CUdeviceptr d;
    if (cuMemAlloc(&d, (size_t)n) != CUDA_SUCCESS) { free(buf); return 0; }
    cuMemcpyHtoD(d, buf, (size_t)n);
    free(buf);
    return d;
}

/* Upload ConvTranspose2d weight transposed for GEMM-based deconv.
 * Input layout: [Ci, Co, kH, kW] (PyTorch ConvTranspose2d).
 * Output layout: [kH*kW*Co, Ci] as FP16 (GEMM W matrix).
 * For stride-aligned deconvs where kH==stride and kW==stride. */
static CUdeviceptr upload_deconv_weight_f16(const qtensor *t, int Ci, int Co, int kH, int kW) {
    if (!t->data) return 0;
    int N = kH * kW * Co;  /* GEMM n_out */
    int K = Ci;             /* GEMM n_in */
    size_t total = (size_t)Ci * Co * kH * kW;
    uint16_t *buf = (uint16_t *)malloc((size_t)N * K * 2);

    /* Get FP32 source */
    float *f32 = NULL;
    int need_free = 0;
    if (t->type == GGML_TYPE_F32) {
        f32 = (float *)t->data;
    } else if (t->type == GGML_TYPE_F16) {
        f32 = (float *)malloc(total * sizeof(float));
        const uint16_t *src = (const uint16_t *)t->data;
        for (size_t i = 0; i < total; i++) f32[i] = ggml_fp16_to_fp32(src[i]);
        need_free = 1;
    } else {
        memset(buf, 0, (size_t)N * K * 2);
        CUdeviceptr d;
        if (cuMemAlloc(&d, (size_t)N * K * 2) != CUDA_SUCCESS) { free(buf); return 0; }
        cuMemcpyHtoD(d, buf, (size_t)N * K * 2);
        free(buf);
        return d;
    }

    /* Transpose [Ci, Co, kH, kW] -> [kH*kW*Co, Ci] as FP16 */
    for (int ci = 0; ci < Ci; ci++) {
        for (int co = 0; co < Co; co++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int src_idx = ci * (Co * kH * kW) + co * (kH * kW) + kh * kW + kw;
                    int dst_row = (kh * kW + kw) * Co + co;
                    int dst_idx = dst_row * Ci + ci;
                    buf[dst_idx] = fp32_to_fp16_raw(f32[src_idx]);
                }
            }
        }
    }

    if (need_free) free(f32);

    CUdeviceptr d;
    if (cuMemAlloc(&d, (size_t)N * K * 2) != CUDA_SUCCESS) { free(buf); return 0; }
    cuMemcpyHtoD(d, buf, (size_t)N * K * 2);
    free(buf);
    return d;
}

/* ======================================================================== */
/* Public API: init                                                         */
/* ======================================================================== */

cuda_da3_runner *cuda_da3_init(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuda_da3: cuew init failed (no CUDA/NVRTC?)\n");
        return NULL;
    }
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_da3: cuInit failed\n");
        return NULL;
    }

    cuda_da3_runner *r = (cuda_da3_runner *)calloc(1, sizeof(cuda_da3_runner));
    r->verbose = verbose;

    CHECK_CU_NULL(cuDeviceGet(&r->device, device_id));
    CHECK_CU_NULL(cuCtxCreate(&r->context, 0, r->device));
    CHECK_CU_NULL(cuStreamCreate(&r->stream, CU_STREAM_DEFAULT));

    if (da3_compile_kernels(r) != 0) {
        fprintf(stderr, "cuda_da3: kernel compilation failed\n");
        free(r);
        return NULL;
    }

    return r;
}

/* ======================================================================== */
/* Public API: load_weights                                                 */
/* ======================================================================== */

/* GGUF KV helpers */
static int da3g_int(const gguf_context *g, const char *k, int d) {
    int i = gguf_find_key(g, k);
    if (i < 0) return d;
    if (g->kv[i].type == GGUF_TYPE_UINT32) return (int)g->kv[i].value.u32;
    if (g->kv[i].type == GGUF_TYPE_INT32) return g->kv[i].value.i32;
    return d;
}
static float da3g_float(const gguf_context *g, const char *k, float d) {
    int i = gguf_find_key(g, k);
    if (i < 0) return d;
    if (g->kv[i].type == GGUF_TYPE_FLOAT32) return g->kv[i].value.f32;
    return d;
}

static qtensor da3g_tensor(const gguf_context *g, const char *name) {
    qtensor t = {0};
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name.str, name) == 0) {
            t.data = gguf_tensor_data(g, (int)i);
            t.type = g->tensors[i].type;
            t.n_dims = (int)g->tensors[i].n_dims;
            for (int d = 0; d < t.n_dims; d++) t.dims[d] = g->tensors[i].dims[d];
            t.n_cols = (int)g->tensors[i].dims[0];
            t.n_rows = (t.n_dims >= 2) ? (int)g->tensors[i].dims[1] : 1;
            break;
        }
    }
    return t;
}

int cuda_da3_load_weights(cuda_da3_runner *r, gguf_context *gguf) {
    /* Load hyperparameters */
    r->dim        = da3g_int(gguf, "da3.embed_dim", 384);
    r->n_heads    = da3g_int(gguf, "da3.n_heads", 6);
    r->head_dim   = da3g_int(gguf, "da3.head_dim", 64);
    r->n_blocks   = da3g_int(gguf, "da3.n_blocks", 12);
    r->ffn_hidden = da3g_int(gguf, "da3.ffn_hidden", 1024);
    r->patch_size = da3g_int(gguf, "da3.patch_size", 14);
    r->image_size = da3g_int(gguf, "da3.image_size", 518);
    r->ln_eps     = da3g_float(gguf, "da3.ln_eps", 1e-6f);
    r->rope_start = da3g_int(gguf, "da3.rope_start_layer", 4);
    r->qk_norm_start = da3g_int(gguf, "da3.qk_norm_start_layer", 4);
    r->head_features = da3g_int(gguf, "da3.head.features", 64);

    r->grid_h = r->image_size / r->patch_size;
    r->grid_w = r->grid_h;
    r->n_patches = r->grid_h * r->grid_w;
    r->n_tokens = r->n_patches + 1;

    /* Feature layers */
    int fl = gguf_find_key(gguf, "da3.feature_layers");
    if (fl >= 0 && gguf->kv[fl].type == GGUF_TYPE_ARRAY) {
        int32_t *a = (int32_t *)gguf->kv[fl].value.arr.data;
        for (int i = 0; i < 4; i++) r->feature_layers[i] = a[i];
    } else {
        r->feature_layers[0] = 5; r->feature_layers[1] = 7;
        r->feature_layers[2] = 9; r->feature_layers[3] = 11;
    }

    /* Head out_channels */
    int oc = gguf_find_key(gguf, "da3.head.out_channels");
    if (oc >= 0 && gguf->kv[oc].type == GGUF_TYPE_ARRAY) {
        int32_t *a = (int32_t *)gguf->kv[oc].value.arr.data;
        for (int i = 0; i < 4; i++) r->head_out_channels[i] = a[i];
    } else {
        r->head_out_channels[0] = 48; r->head_out_channels[1] = 96;
        r->head_out_channels[2] = 192; r->head_out_channels[3] = 384;
    }

    /* Image normalization */
    r->image_mean[0] = 0.485f; r->image_mean[1] = 0.456f; r->image_mean[2] = 0.406f;
    r->image_std[0] = 0.229f; r->image_std[1] = 0.224f; r->image_std[2] = 0.225f;

    /* Detect FFN type */
    r->use_swiglu = 0;
    { qtensor t = da3g_tensor(gguf, "da3.blk.0.ffn_gate_up.weight"); if (t.data) r->use_swiglu = 1; }

    /* Upload backbone embeddings (F32 on GPU) */
    {
        qtensor t;
        t = da3g_tensor(gguf, "da3.cls_token");     r->d_cls_token = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.pos_embed");      r->d_pos_embed = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.patch_embed.weight"); r->d_patch_embed_w = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.patch_embed.bias");   r->d_patch_embed_b = upload_tensor_f32(&t);
    }

    /* Upload transformer blocks */
    int nb = r->n_blocks;
    r->layers = (cuda_da3_layer *)calloc((size_t)nb, sizeof(cuda_da3_layer));
    for (int L = 0; L < nb; L++) {
        cuda_da3_layer *ly = &r->layers[L];
        char name[128];
        qtensor t;

#define LOAD_F32(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    t = da3g_tensor(gguf, name); ly->field = upload_tensor_f32(&t);
#define LOAD_F16(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    t = da3g_tensor(gguf, name); ly->field = upload_tensor_f16(&t);
#define LOAD_BACKBONE_W(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    t = da3g_tensor(gguf, name); \
    ly->field = r->use_fp8 ? upload_tensor_fp8_e4m3(&t) : upload_tensor_f16(&t);

        LOAD_F32(ln1_w, "da3.blk.%d.ln1.weight", L)
        LOAD_F32(ln1_b, "da3.blk.%d.ln1.bias", L)

        LOAD_BACKBONE_W(attn_qkv_w, "da3.blk.%d.attn_qkv.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.attn_qkv.weight", L);
        t = da3g_tensor(gguf, name);
        ly->qkv_rows = t.n_rows; ly->qkv_cols = t.n_cols;
        LOAD_F32(attn_qkv_b, "da3.blk.%d.attn_qkv.bias", L)

        LOAD_F32(attn_q_norm_w, "da3.blk.%d.attn_q_norm.weight", L)
        LOAD_F32(attn_q_norm_b, "da3.blk.%d.attn_q_norm.bias", L)
        LOAD_F32(attn_k_norm_w, "da3.blk.%d.attn_k_norm.weight", L)
        LOAD_F32(attn_k_norm_b, "da3.blk.%d.attn_k_norm.bias", L)
        ly->has_qk_norm = (ly->attn_q_norm_w != 0);

        LOAD_BACKBONE_W(attn_out_w, "da3.blk.%d.attn_out.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.attn_out.weight", L);
        t = da3g_tensor(gguf, name);
        ly->out_rows = t.n_rows; ly->out_cols = t.n_cols;
        LOAD_F32(attn_out_b, "da3.blk.%d.attn_out.bias", L)

        LOAD_F32(ls1, "da3.blk.%d.ls1", L)
        LOAD_F32(ls2, "da3.blk.%d.ls2", L)
        LOAD_F32(ln2_w, "da3.blk.%d.ln2.weight", L)
        LOAD_F32(ln2_b, "da3.blk.%d.ln2.bias", L)

        LOAD_BACKBONE_W(ffn_gate_up_w, "da3.blk.%d.ffn_gate_up.weight", L)
        if (ly->ffn_gate_up_w) {
            snprintf(name, sizeof(name), "da3.blk.%d.ffn_gate_up.weight", L);
            t = da3g_tensor(gguf, name);
            ly->ffn_gu_rows = t.n_rows; ly->ffn_gu_cols = t.n_cols;
            ly->has_swiglu = 1;
        }
        LOAD_F32(ffn_gate_up_b, "da3.blk.%d.ffn_gate_up.bias", L)

        LOAD_BACKBONE_W(ffn_up_w, "da3.blk.%d.ffn_up.weight", L)
        if (ly->ffn_up_w) {
            snprintf(name, sizeof(name), "da3.blk.%d.ffn_up.weight", L);
            t = da3g_tensor(gguf, name);
            ly->ffn_up_rows = t.n_rows; ly->ffn_up_cols = t.n_cols;
        }
        LOAD_F32(ffn_up_b, "da3.blk.%d.ffn_up.bias", L)

        LOAD_BACKBONE_W(ffn_down_w, "da3.blk.%d.ffn_down.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.ffn_down.weight", L);
        t = da3g_tensor(gguf, name);
        ly->ffn_down_rows = t.n_rows; ly->ffn_down_cols = t.n_cols;
        LOAD_F32(ffn_down_b, "da3.blk.%d.ffn_down.bias", L)

#undef LOAD_F32
#undef LOAD_F16
#undef LOAD_BACKBONE_W
    }

    /* Upload DPT head weights to GPU */
    {
        qtensor t;
        char name[128];
        dpt_gpu_weights *dw = &r->dpt_w;

        /* Head LayerNorm */
        t = da3g_tensor(gguf, "da3.head.norm.weight"); dw->norm_w = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.head.norm.bias");   dw->norm_b = upload_tensor_f32(&t);

        /* Projection layers: w as F16 for gemm_f16_f32, bias as F32 */
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.proj.%d.weight", i);
            t = da3g_tensor(gguf, name);
            dw->proj_w[i] = upload_tensor_f16(&t);
            dw->proj_rows[i] = t.n_rows;
            snprintf(name, sizeof(name), "da3.head.proj.%d.bias", i);
            t = da3g_tensor(gguf, name);
            dw->proj_b[i] = upload_tensor_f32(&t);
        }

        /* Spatial alignment: ConvT weights transposed to FP16 for GEMM-based deconv */
        {
            int oc0 = r->head_out_channels[0]; /* 48 */
            t = da3g_tensor(gguf, "da3.head.upsample_0.weight");
            dw->upsample_0_w = upload_deconv_weight_f16(&t, oc0, oc0, 4, 4);
            t = da3g_tensor(gguf, "da3.head.upsample_0.bias");
            dw->upsample_0_b = upload_tensor_f32(&t);

            int oc1 = r->head_out_channels[1]; /* 96 */
            t = da3g_tensor(gguf, "da3.head.upsample_1.weight");
            dw->upsample_1_w = upload_deconv_weight_f16(&t, oc1, oc1, 2, 2);
            t = da3g_tensor(gguf, "da3.head.upsample_1.bias");
            dw->upsample_1_b = upload_tensor_f32(&t);
        }
        t = da3g_tensor(gguf, "da3.head.downsample.weight");  dw->downsample_w = upload_tensor_f32_as_f16(&t);
        t = da3g_tensor(gguf, "da3.head.downsample.bias");    dw->downsample_b = upload_tensor_f32(&t);

        /* Adapter convolutions (FP16 for MMA conv, no bias) */
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.adapter.%d.weight", i);
            t = da3g_tensor(gguf, name);
            dw->adapter_w[i] = upload_tensor_f32_as_f16(&t);
        }

        /* RefineNet fusion blocks (weights FP16 for MMA conv, biases FP32) */
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_out_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_out_b[i] = upload_tensor_f32(&t);

            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu1_c1_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu1_c1_b[i] = upload_tensor_f32(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu1_c2_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu1_c2_b[i] = upload_tensor_f32(&t);
            dw->has_rcu1[i] = (dw->fuse_rcu1_c1_w[i] != 0);

            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu2_c1_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu2_c1_b[i] = upload_tensor_f32(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu2_c2_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu2_c2_b[i] = upload_tensor_f32(&t);
            dw->has_rcu2[i] = (dw->fuse_rcu2_c1_w[i] != 0);
        }

        /* Output convolutions (neck, out_0 FP16 for MMA; out_2 stays FP32, Co=2 too small) */
        t = da3g_tensor(gguf, "da3.head.neck.weight");  dw->neck_w  = upload_tensor_f32_as_f16(&t);
        t = da3g_tensor(gguf, "da3.head.neck.bias");    dw->neck_b  = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.head.out_0.weight"); dw->out_0_w = upload_tensor_f32_as_f16(&t);
        t = da3g_tensor(gguf, "da3.head.out_0.bias");   dw->out_0_b = upload_tensor_f32(&t);
        dw->out_mid = t.n_cols; /* bias [out_mid] → n_cols = intermediate channels */
        t = da3g_tensor(gguf, "da3.head.out_2.weight"); dw->out_2_w = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.head.out_2.bias");   dw->out_2_b = upload_tensor_f32(&t);

        if (r->verbose >= 1)
            fprintf(stderr, "cuda_da3: DPT head weights uploaded to GPU\n");
    }

    /* Allocate backbone scratch buffers */
    int nt = r->n_tokens;
    int dim = r->dim;
    int np = r->n_patches;
    int gh = r->grid_h;
    int max_ffn = r->use_swiglu ? 2 * r->ffn_hidden : 4 * dim;

    CHECK_CU(cuMemAlloc(&r->d_hidden,   (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_hidden2,  (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_ln_buf,   (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_qkv,      (size_t)nt * 3 * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_attn_out, (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_ffn_buf,  (size_t)nt * max_ffn * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_ffn_mid,  (size_t)nt * r->ffn_hidden * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_proj_out, (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_pos_y,    (size_t)nt * sizeof(int)));
    CHECK_CU(cuMemAlloc(&r->d_pos_x,    (size_t)nt * sizeof(int)));
    CHECK_CU(cuMemAlloc(&r->d_img_norm, (size_t)3 * r->image_size * r->image_size * sizeof(float)));

    for (int i = 0; i < 4; i++)
        CHECK_CU(cuMemAlloc(&r->d_features[i], (size_t)nt * dim * sizeof(float)));

    /* Allocate DPT head scratch buffers */
    {
        int feat = r->head_features;  /* 64 */
        int oc_max = r->head_out_channels[3]; /* 384 */
        /* Spatial dims after alignment: level 0 = (gh-1)*4+4, 1 = (gh-1)*2+2, 2 = gh, 3 = (gh+1)/2 */
        int sp_h[4], sp_w[4];
        sp_h[0] = sp_w[0] = (gh - 1) * 4 + 4; /* 148 */
        sp_h[1] = sp_w[1] = (gh - 1) * 2 + 2; /* 74 */
        sp_h[2] = sp_w[2] = gh;                /* 37 */
        sp_h[3] = sp_w[3] = (gh + 2 - 3) / 2 + 1; /* 19 */
        int max_hw = sp_h[0] * sp_w[0]; /* 148*148 = 21904 */

        /* Large scratch: max of token processing and fusion needs */
        size_t large_sz = (size_t)feat * max_hw; /* 64*148*148 = 1,401,856 */
        if (large_sz < (size_t)np * 2 * dim)
            large_sz = (size_t)np * 2 * dim; /* 1369*768 = 1,051,392 */

        CHECK_CU(cuMemAlloc(&r->d_dpt_cat,  large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_ln,   large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_proj, (size_t)np * oc_max * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_chw,  (size_t)oc_max * gh * gh * sizeof(float)));

        for (int i = 0; i < 4; i++) {
            int oc = r->head_out_channels[i];
            CHECK_CU(cuMemAlloc(&r->d_dpt_spatial[i],
                                 (size_t)oc * sp_h[i] * sp_w[i] * sizeof(float)));
            CHECK_CU(cuMemAlloc(&r->d_dpt_adapted[i],
                                 (size_t)feat * sp_h[i] * sp_w[i] * sizeof(float)));
        }

        CHECK_CU(cuMemAlloc(&r->d_dpt_fused, large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_tmp,   large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_tmp2,  large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_out,   (size_t)2 * max_hw * sizeof(float)));

        if (r->verbose >= 1)
            fprintf(stderr, "cuda_da3: DPT scratch buffers allocated (~%.1f MB)\n",
                    (float)(4 * large_sz + np * oc_max + oc_max * gh * gh + 2 * max_hw) * 4 / 1e6f);
    }

    /* Upload position arrays for RoPE */
    int *py = (int *)calloc((size_t)nt, sizeof(int));
    int *px = (int *)calloc((size_t)nt, sizeof(int));
    for (int p = 0; p < np; p++) {
        py[1 + p] = p / r->grid_w;
        px[1 + p] = p % r->grid_w;
    }
    cuMemcpyHtoD(r->d_pos_y, py, (size_t)nt * sizeof(int));
    cuMemcpyHtoD(r->d_pos_x, px, (size_t)nt * sizeof(int));
    free(py); free(px);

    r->h_output = (float *)malloc((size_t)nt * dim * sizeof(float));

    /* Load CPU model for preprocessing */
    r->cpu_model = da3_load(gguf);
    if (!r->cpu_model) {
        fprintf(stderr, "cuda_da3: warning: failed to load CPU model\n");
    }

    r->loaded = 1;

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_da3: loaded %d blocks, dim=%d, tokens=%d, swiglu=%d\n",
                nb, dim, nt, r->use_swiglu);
    return 0;
}

/* ======================================================================== */
/* Public API: load_safetensors (direct .safetensors loading, no GGUF)      */
/* ======================================================================== */

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

/* Name mapping entry: safetensors name suffix → GGUF name pattern */
typedef struct { const char *st_suffix; const char *gguf_name; } st_name_map;

/* Build a name mapping table from safetensors tensor names to GGUF names.
 * Returns allocated array of {st_name, gguf_name} pairs. Caller frees. */
static st_name_map *build_st_name_map(const st_context *st, int *out_count) {
    /* Detect prefixes */
    const char *bb_prefix = NULL;
    const char *hd_prefix = NULL;
    static const char *bb_candidates[] = {
        "model.backbone.pretrained.", "backbone.pretrained.",
        "backbone.", "pretrained.", "encoder.", NULL
    };
    static const char *hd_candidates[] = {
        "model.head.", "head.", "depth_head.", "dpt_head.", NULL
    };
    for (int c = 0; bb_candidates[c]; c++) {
        for (int i = 0; i < st->n_tensors; i++) {
            if (strstr(safetensors_name(st, i), "blocks.") &&
                strncmp(safetensors_name(st, i), bb_candidates[c], strlen(bb_candidates[c])) == 0) {
                bb_prefix = bb_candidates[c];
                goto bb_found;
            }
        }
    }
bb_found:
    for (int c = 0; hd_candidates[c]; c++) {
        for (int i = 0; i < st->n_tensors; i++) {
            if (strstr(safetensors_name(st, i), "projects.") &&
                strncmp(safetensors_name(st, i), hd_candidates[c], strlen(hd_candidates[c])) == 0) {
                hd_prefix = hd_candidates[c];
                goto hd_found;
            }
        }
    }
hd_found:

    /* Backbone block sub-key mapping */
    static const struct { const char *st; const char *gg; } blk_map[] = {
        {"norm1.weight",       "ln1.weight"},
        {"norm1.bias",         "ln1.bias"},
        {"attn.qkv.weight",   "attn_qkv.weight"},
        {"attn.qkv.bias",     "attn_qkv.bias"},
        {"attn.q_norm.weight", "attn_q_norm.weight"},
        {"attn.q_norm.bias",   "attn_q_norm.bias"},
        {"attn.k_norm.weight", "attn_k_norm.weight"},
        {"attn.k_norm.bias",   "attn_k_norm.bias"},
        {"attn.proj.weight",   "attn_out.weight"},
        {"attn.proj.bias",     "attn_out.bias"},
        {"ls1.gamma",          "ls1"},
        {"ls2.gamma",          "ls2"},
        {"norm2.weight",       "ln2.weight"},
        {"norm2.bias",         "ln2.bias"},
        {"mlp.w12.weight",     "ffn_gate_up.weight"},
        {"mlp.w12.bias",       "ffn_gate_up.bias"},
        {"mlp.w3.weight",      "ffn_down.weight"},
        {"mlp.w3.bias",        "ffn_down.bias"},
        {"mlp.fc1.weight",     "ffn_up.weight"},
        {"mlp.fc1.bias",       "ffn_up.bias"},
        {"mlp.fc2.weight",     "ffn_down.weight"},
        {"mlp.fc2.bias",       "ffn_down.bias"},
        {NULL, NULL}
    };

    /* RefineNet sub-key mapping */
    static const struct { const char *st; const char *gg; } rn_map[] = {
        {"out_conv.weight",            "out.weight"},
        {"out_conv.bias",              "out.bias"},
        {"resConfUnit1.conv1.weight",  "rcu1.conv1.weight"},
        {"resConfUnit1.conv1.bias",    "rcu1.conv1.bias"},
        {"resConfUnit1.conv2.weight",  "rcu1.conv2.weight"},
        {"resConfUnit1.conv2.bias",    "rcu1.conv2.bias"},
        {"resConfUnit2.conv1.weight",  "rcu2.conv1.weight"},
        {"resConfUnit2.conv1.bias",    "rcu2.conv1.bias"},
        {"resConfUnit2.conv2.weight",  "rcu2.conv2.weight"},
        {"resConfUnit2.conv2.bias",    "rcu2.conv2.bias"},
        {NULL, NULL}
    };

    /* Allocate mapping array (upper bound = n_tensors) */
    st_name_map *map = (st_name_map *)calloc((size_t)st->n_tensors, sizeof(st_name_map));
    int n = 0;

    /* CameraDec sub-key mapping */
    static const struct { const char *st; const char *gg; } cam_dec_map[] = {
        {"backbone.0.weight",  "cam_dec.mlp.0.weight"},
        {"backbone.0.bias",    "cam_dec.mlp.0.bias"},
        {"backbone.2.weight",  "cam_dec.mlp.2.weight"},
        {"backbone.2.bias",    "cam_dec.mlp.2.bias"},
        {"fc_t.weight",        "cam_dec.fc_t.weight"},
        {"fc_t.bias",          "cam_dec.fc_t.bias"},
        {"fc_qvec.weight",     "cam_dec.fc_qvec.weight"},
        {"fc_qvec.bias",       "cam_dec.fc_qvec.bias"},
        {"fc_fov.0.weight",    "cam_dec.fc_fov.weight"},
        {"fc_fov.0.bias",      "cam_dec.fc_fov.bias"},
        {NULL, NULL}
    };

    /* CameraEnc sub-key mapping */
    static const struct { const char *st; const char *gg; } cam_enc_pose_map[] = {
        {"pose_branch.fc1.weight", "cam_enc.fc1.weight"},
        {"pose_branch.fc1.bias",   "cam_enc.fc1.bias"},
        {"pose_branch.fc2.weight", "cam_enc.fc2.weight"},
        {"pose_branch.fc2.bias",   "cam_enc.fc2.bias"},
        {"trunk_norm.weight",      "cam_enc.trunk_norm.weight"},
        {"trunk_norm.bias",        "cam_enc.trunk_norm.bias"},
        {"token_norm.weight",      "cam_enc.token_norm.weight"},
        {"token_norm.bias",        "cam_enc.token_norm.bias"},
        {NULL, NULL}
    };

    for (int i = 0; i < st->n_tensors; i++) {
        const char *key = safetensors_name(st, i);
        char gguf_name[256] = {0};

        /* Backbone */
        if (bb_prefix && strncmp(key, bb_prefix, strlen(bb_prefix)) == 0) {
            const char *s = key + strlen(bb_prefix);
            if (strcmp(s, "cls_token") == 0) {
                strcpy(gguf_name, "da3.cls_token");
            } else if (strcmp(s, "pos_embed") == 0) {
                strcpy(gguf_name, "da3.pos_embed");
            } else if (strcmp(s, "patch_embed.proj.weight") == 0) {
                strcpy(gguf_name, "da3.patch_embed.weight");
            } else if (strcmp(s, "patch_embed.proj.bias") == 0) {
                strcpy(gguf_name, "da3.patch_embed.bias");
            } else if (strcmp(s, "norm.weight") == 0) {
                strcpy(gguf_name, "da3.backbone_norm.weight");
            } else if (strcmp(s, "norm.bias") == 0) {
                strcpy(gguf_name, "da3.backbone_norm.bias");
            } else if (strncmp(s, "blocks.", 7) == 0) {
                int L = 0;
                const char *rest = s + 7;
                while (*rest >= '0' && *rest <= '9') { L = L * 10 + (*rest - '0'); rest++; }
                if (*rest == '.') rest++;
                for (int j = 0; blk_map[j].st; j++) {
                    if (strcmp(rest, blk_map[j].st) == 0) {
                        snprintf(gguf_name, sizeof(gguf_name), "da3.blk.%d.%s", L, blk_map[j].gg);
                        break;
                    }
                }
            }
        }
        /* CameraDec: model.cam_dec.* */
        else if (strncmp(key, "model.cam_dec.", 14) == 0) {
            const char *s = key + 14;
            for (int j = 0; cam_dec_map[j].st; j++) {
                if (strcmp(s, cam_dec_map[j].st) == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "da3.%s", cam_dec_map[j].gg);
                    break;
                }
            }
        }
        /* CameraEnc: model.cam_enc.* */
        else if (strncmp(key, "model.cam_enc.", 14) == 0) {
            const char *s = key + 14;
            /* Check fixed mappings first */
            for (int j = 0; cam_enc_pose_map[j].st; j++) {
                if (strcmp(s, cam_enc_pose_map[j].st) == 0) {
                    snprintf(gguf_name, sizeof(gguf_name), "da3.%s", cam_enc_pose_map[j].gg);
                    break;
                }
            }
            /* Trunk transformer blocks: trunk.{0-3}.* */
            if (!gguf_name[0] && strncmp(s, "trunk.", 6) == 0) {
                int L = 0;
                const char *rest = s + 6;
                while (*rest >= '0' && *rest <= '9') { L = L * 10 + (*rest - '0'); rest++; }
                if (*rest == '.') rest++;
                for (int j = 0; blk_map[j].st; j++) {
                    if (strcmp(rest, blk_map[j].st) == 0) {
                        snprintf(gguf_name, sizeof(gguf_name), "da3.cam_enc.trunk.%d.%s", L, blk_map[j].gg);
                        break;
                    }
                }
            }
        }
        /* GSDPT: model.gs_head.* */
        else if (strncmp(key, "model.gs_head.", 14) == 0) {
            const char *s = key + 14;
            /* images_merger: Conv2d layers 0, 2, 4 */
            if (strncmp(s, "images_merger.", 14) == 0) {
                const char *ms = s + 14;
                int idx = -1;
                if (ms[0] >= '0' && ms[0] <= '9') {
                    idx = ms[0] - '0'; /* 0, 2, 4 → map to 0, 1, 2 */
                    if (idx == 2) idx = 1;
                    else if (idx == 4) idx = 2;
                    else if (idx != 0) idx = -1;
                }
                if (idx >= 0 && ms[1] == '.') {
                    const char *wb = ms + 2;
                    snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.merger.%d.%s", idx, wb);
                }
            }
            /* DPT sub-structure: same as main head */
            else if (strcmp(s, "norm.weight") == 0) {
                strcpy(gguf_name, "da3.gsdpt.head.norm.weight");
            } else if (strcmp(s, "norm.bias") == 0) {
                strcpy(gguf_name, "da3.gsdpt.head.norm.bias");
            } else if (strncmp(s, "projects.", 9) == 0) {
                int idx = s[9] - '0';
                const char *wb = s + 11;
                snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.proj.%d.%s", idx, wb);
            } else if (strncmp(s, "resize_layers.", 14) == 0) {
                int idx = s[14] - '0';
                const char *wb = s + 16;
                if (idx == 0) snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.upsample_0.%s", wb);
                else if (idx == 1) snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.upsample_1.%s", wb);
                else if (idx == 3) snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.downsample.%s", wb);
            } else if (strncmp(s, "scratch.", 8) == 0) {
                const char *ss = s + 8;
                /* layer{1-4}_rn */
                for (int li = 1; li <= 4; li++) {
                    char pfx[32];
                    snprintf(pfx, sizeof(pfx), "layer%d_rn.weight", li);
                    if (strcmp(ss, pfx) == 0) {
                        snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.adapter.%d.weight", li - 1);
                        break;
                    }
                }
                /* refinenet{1-4}.* */
                if (!gguf_name[0]) {
                    for (int ri = 1; ri <= 4; ri++) {
                        char pfx[32];
                        snprintf(pfx, sizeof(pfx), "refinenet%d.", ri);
                        size_t plen = strlen(pfx);
                        if (strncmp(ss, pfx, plen) == 0) {
                            const char *rn = ss + plen;
                            for (int j = 0; rn_map[j].st; j++) {
                                if (strcmp(rn, rn_map[j].st) == 0) {
                                    snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.fuse.%d.%s", ri - 1, rn_map[j].gg);
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                /* Output convolutions */
                if (!gguf_name[0]) {
                    static const struct { const char *st; const char *gg; } gsout_map[] = {
                        {"output_conv1.weight",   "da3.gsdpt.head.neck.weight"},
                        {"output_conv1.bias",     "da3.gsdpt.head.neck.bias"},
                        {"output_conv2.0.weight", "da3.gsdpt.head.out_0.weight"},
                        {"output_conv2.0.bias",   "da3.gsdpt.head.out_0.bias"},
                        {"output_conv2.2.weight", "da3.gsdpt.head.out_2.weight"},
                        {"output_conv2.2.bias",   "da3.gsdpt.head.out_2.bias"},
                        {NULL, NULL}
                    };
                    for (int j = 0; gsout_map[j].st; j++) {
                        if (strcmp(ss, gsout_map[j].st) == 0) {
                            strcpy(gguf_name, gsout_map[j].gg);
                            break;
                        }
                    }
                }
            }
        }
        /* Head (both main and aux) */
        else if (hd_prefix && strncmp(key, hd_prefix, strlen(hd_prefix)) == 0) {
            const char *s = key + strlen(hd_prefix);
            if (strcmp(s, "norm.weight") == 0) {
                strcpy(gguf_name, "da3.head.norm.weight");
            } else if (strcmp(s, "norm.bias") == 0) {
                strcpy(gguf_name, "da3.head.norm.bias");
            } else if (strncmp(s, "projects.", 9) == 0) {
                int idx = s[9] - '0';
                const char *wb = s + 11; /* skip "X." */
                snprintf(gguf_name, sizeof(gguf_name), "da3.head.proj.%d.%s", idx, wb);
            } else if (strncmp(s, "resize_layers.", 14) == 0) {
                int idx = s[14] - '0';
                const char *wb = s + 16;
                if (idx == 0) snprintf(gguf_name, sizeof(gguf_name), "da3.head.upsample_0.%s", wb);
                else if (idx == 1) snprintf(gguf_name, sizeof(gguf_name), "da3.head.upsample_1.%s", wb);
                else if (idx == 3) snprintf(gguf_name, sizeof(gguf_name), "da3.head.downsample.%s", wb);
            } else if (strncmp(s, "scratch.", 8) == 0) {
                const char *ss = s + 8;
                /* layer{1-4}_rn */
                for (int li = 1; li <= 4; li++) {
                    char pfx[32];
                    snprintf(pfx, sizeof(pfx), "layer%d_rn.weight", li);
                    if (strcmp(ss, pfx) == 0) {
                        snprintf(gguf_name, sizeof(gguf_name), "da3.head.adapter.%d.weight", li - 1);
                        break;
                    }
                }
                /* refinenet{1-4}.* (main, NOT _aux) */
                if (!gguf_name[0]) {
                    for (int ri = 1; ri <= 4; ri++) {
                        char pfx[32];
                        snprintf(pfx, sizeof(pfx), "refinenet%d.", ri);
                        size_t plen = strlen(pfx);
                        if (strncmp(ss, pfx, plen) == 0) {
                            const char *rn = ss + plen;
                            for (int j = 0; rn_map[j].st; j++) {
                                if (strcmp(rn, rn_map[j].st) == 0) {
                                    snprintf(gguf_name, sizeof(gguf_name), "da3.head.fuse.%d.%s", ri - 1, rn_map[j].gg);
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                /* refinenet{1-4}_aux.* (aux branch) */
                if (!gguf_name[0]) {
                    for (int ri = 1; ri <= 4; ri++) {
                        char pfx[32];
                        snprintf(pfx, sizeof(pfx), "refinenet%d_aux.", ri);
                        size_t plen = strlen(pfx);
                        if (strncmp(ss, pfx, plen) == 0) {
                            const char *rn = ss + plen;
                            for (int j = 0; rn_map[j].st; j++) {
                                if (strcmp(rn, rn_map[j].st) == 0) {
                                    snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_fuse.%d.%s", ri - 1, rn_map[j].gg);
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                /* output_conv1_aux.{0-3}.{0-4}.weight/bias → per-level conv chains */
                if (!gguf_name[0] && strncmp(ss, "output_conv1_aux.", 17) == 0) {
                    int level = ss[17] - '0';
                    if (level >= 0 && level < 4 && ss[18] == '.') {
                        int ci = ss[19] - '0';
                        if (ci >= 0 && ci <= 4 && ss[20] == '.') {
                            const char *wb = ss + 21;
                            snprintf(gguf_name, sizeof(gguf_name),
                                     "da3.head.aux_oc1.%d.%d.%s", level, ci, wb);
                        }
                    }
                }
                /* output_conv2_aux.{0-3}.{0|2|5}.weight/bias */
                if (!gguf_name[0] && strncmp(ss, "output_conv2_aux.", 17) == 0) {
                    int level = ss[17] - '0';
                    if (level >= 0 && level < 4 && ss[18] == '.') {
                        int si = ss[19] - '0';
                        if (ss[20] == '.') {
                            const char *wb = ss + 21;
                            if (si == 0) snprintf(gguf_name, sizeof(gguf_name),
                                                   "da3.head.aux_oc2.%d.conv.%s", level, wb);
                            else if (si == 2) snprintf(gguf_name, sizeof(gguf_name),
                                                        "da3.head.aux_oc2.%d.gn.%s", level, wb);
                            else if (si == 5) snprintf(gguf_name, sizeof(gguf_name),
                                                        "da3.head.aux_oc2.%d.out.%s", level, wb);
                        }
                    }
                }
                /* Main output convolutions (not _aux) */
                if (!gguf_name[0] && !strstr(ss, "_aux")) {
                    static const struct { const char *st; const char *gg; } out_map[] = {
                        {"output_conv1.weight",   "da3.head.neck.weight"},
                        {"output_conv1.bias",     "da3.head.neck.bias"},
                        {"output_conv2.0.weight", "da3.head.out_0.weight"},
                        {"output_conv2.0.bias",   "da3.head.out_0.bias"},
                        {"output_conv2.2.weight", "da3.head.out_2.weight"},
                        {"output_conv2.2.bias",   "da3.head.out_2.bias"},
                        {NULL, NULL}
                    };
                    for (int j = 0; out_map[j].st; j++) {
                        if (strcmp(ss, out_map[j].st) == 0) {
                            strcpy(gguf_name, out_map[j].gg);
                            break;
                        }
                    }
                }
            }
        }

        if (gguf_name[0]) {
            map[n].st_suffix = key;
            map[n].gguf_name = strdup(gguf_name);
            n++;
        }
    }

    *out_count = n;
    return map;
}

/* Lookup a tensor from safetensors using GGUF-style name via name mapping */
static qtensor da3s_tensor(const st_context *st, const st_name_map *map, int map_count,
                           const char *gguf_name) {
    qtensor t = {0};
    /* Find the safetensors name corresponding to this GGUF name */
    for (int m = 0; m < map_count; m++) {
        if (strcmp(map[m].gguf_name, gguf_name) != 0) continue;
        int si = safetensors_find(st, map[m].st_suffix);
        if (si < 0) break;
        t.data = safetensors_data(st, si);
        /* Safetensors F32 → GGML_TYPE_F32 (0) */
        const char *dt = safetensors_dtype(st, si);
        if (strcmp(dt, "F32") == 0) t.type = 0; /* GGML_TYPE_F32 */
        else if (strcmp(dt, "F16") == 0) t.type = 1; /* GGML_TYPE_F16 */
        else if (strcmp(dt, "BF16") == 0) t.type = 30; /* GGML_TYPE_BF16 */
        else t.type = 0;
        t.n_dims = safetensors_ndims(st, si);
        const uint64_t *shape = safetensors_shape(st, si);
        /* Safetensors stores shape in PyTorch order (outermost first).
         * GGUF stores dims reversed. For weight matrices:
         *   safetensors [out, in] → gguf dims [in, out] → n_cols=in, n_rows=out
         * But the upload code uses n_cols/n_rows from gguf dims[0]/dims[1],
         * so we reverse the shape to match GGUF convention. */
        for (int d = 0; d < t.n_dims; d++)
            t.dims[d] = shape[t.n_dims - 1 - d];
        t.n_cols = (int)t.dims[0];
        t.n_rows = (t.n_dims >= 2) ? (int)t.dims[1] : 1;
        break;
    }
    return t;
}

int cuda_da3_load_safetensors(cuda_da3_runner *r, const char *st_path, const char *config_path) {
    /* Open safetensors */
    st_context *st = safetensors_open(st_path);
    if (!st) return -1;

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_da3: safetensors: %d tensors\n", st->n_tensors);

    /* Build name mapping */
    int map_count = 0;
    st_name_map *map = build_st_name_map(st, &map_count);
    if (r->verbose >= 1)
        fprintf(stderr, "cuda_da3: mapped %d tensors\n", map_count);

    /* Detect model params from tensors */
    int embed_dim = 384, n_heads = 6, head_dim = 64, n_blocks = 12, ffn_hidden = 0;
    int head_features = 64;
    int head_oc[4] = {48, 96, 192, 384};
    int feature_layers[4] = {5, 7, 9, 11};
    int rope_start = 4, qknorm_start = 4;
    int has_swiglu = 0;

    /* Parse config.json if available */
    if (config_path) {
        FILE *f = fopen(config_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = (char *)malloc(sz);
            size_t nr = fread(buf, 1, sz, f);
            (void)nr;
            fclose(f);
            json_val *root = json_parse(buf, (int)sz);
            if (root) {
                json_val *cfg = json_obj_get(root, "config");
                if (cfg) {
                    json_val *head = json_obj_get(cfg, "head");
                    if (head) {
                        json_val *v = json_obj_get(head, "features");
                        if (v && v->type == JSON_NUMBER) head_features = (int)v->num;
                        json_val *oc = json_obj_get(head, "out_channels");
                        if (oc && oc->type == JSON_ARRAY)
                            for (int i = 0; i < 4 && i < oc->arr.count; i++)
                                head_oc[i] = (int)oc->arr.items[i].num;
                    }
                    json_val *net = json_obj_get(cfg, "net");
                    if (net) {
                        json_val *ol = json_obj_get(net, "out_layers");
                        if (ol && ol->type == JSON_ARRAY)
                            for (int i = 0; i < 4 && i < ol->arr.count; i++)
                                feature_layers[i] = (int)ol->arr.items[i].num;
                        json_val *rs = json_obj_get(net, "rope_start");
                        if (rs && rs->type == JSON_NUMBER) rope_start = (int)rs->num;
                        json_val *qs = json_obj_get(net, "qknorm_start");
                        if (qs && qs->type == JSON_NUMBER) qknorm_start = (int)qs->num;
                    }
                }
                json_free(root);
            }
            free(buf);
        }
    }

    /* Detect embed_dim from patch_embed weight shape */
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "patch_embed.proj.weight")) {
            embed_dim = (int)safetensors_shape(st, i)[0]; /* [out_ch, 3, 14, 14] */
            break;
        }
    }
    /* Detect n_heads from q_norm weight (any block - may not exist in block 0) */
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "attn.q_norm.weight") && !strstr(nm, "_aux")) {
            int hd = (int)safetensors_shape(st, i)[0];
            n_heads = embed_dim / hd;
            head_dim = hd;
            break;
        }
    }
    /* Detect FFN hidden & type */
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "blocks.0.mlp.w12.weight")) {
            ffn_hidden = (int)safetensors_shape(st, i)[0] / 2;
            has_swiglu = 1;
            break;
        }
        if (strstr(nm, "blocks.0.mlp.fc1.weight")) {
            ffn_hidden = (int)safetensors_shape(st, i)[0];
            break;
        }
    }
    /* Detect n_blocks */
    n_blocks = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *p = strstr(nm, "blocks.");
        if (p && !strstr(nm, "_aux")) {
            p += 7;
            int blk = 0;
            while (*p >= '0' && *p <= '9') { blk = blk * 10 + (*p - '0'); p++; }
            if (blk + 1 > n_blocks) n_blocks = blk + 1;
        }
    }

    /* Set model params */
    r->dim        = embed_dim;
    r->n_heads    = n_heads;
    r->head_dim   = head_dim;
    r->n_blocks   = n_blocks;
    r->ffn_hidden = ffn_hidden;
    r->patch_size = 14;
    r->image_size = 518;
    r->ln_eps     = 1e-6f;
    r->rope_start = rope_start;
    r->qk_norm_start = qknorm_start;
    r->head_features = head_features;
    r->use_swiglu = has_swiglu;

    r->grid_h = r->image_size / r->patch_size;
    r->grid_w = r->grid_h;
    r->n_patches = r->grid_h * r->grid_w;
    r->n_tokens = r->n_patches + 1;

    for (int i = 0; i < 4; i++) r->feature_layers[i] = feature_layers[i];
    for (int i = 0; i < 4; i++) r->head_out_channels[i] = head_oc[i];

    r->image_mean[0] = 0.485f; r->image_mean[1] = 0.456f; r->image_mean[2] = 0.406f;
    r->image_std[0] = 0.229f; r->image_std[1] = 0.224f; r->image_std[2] = 0.225f;

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_da3: model: dim=%d, n_heads=%d, head_dim=%d, n_blocks=%d, ffn=%d, swiglu=%d\n",
                embed_dim, n_heads, head_dim, n_blocks, ffn_hidden, has_swiglu);
        fprintf(stderr, "cuda_da3: head: feat=%d, oc=[%d,%d,%d,%d], feat_layers=[%d,%d,%d,%d], rope_start=%d, qknorm_start=%d\n",
                head_features, head_oc[0], head_oc[1], head_oc[2], head_oc[3],
                feature_layers[0], feature_layers[1], feature_layers[2], feature_layers[3],
                rope_start, qknorm_start);
    }

    /* Upload backbone embeddings (F32 on GPU) */
    {
        qtensor t;
        t = da3s_tensor(st, map, map_count, "da3.cls_token");     r->d_cls_token = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.pos_embed");     r->d_pos_embed = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.patch_embed.weight"); r->d_patch_embed_w = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.patch_embed.bias");   r->d_patch_embed_b = upload_tensor_f32(&t);
    }

    /* Upload transformer blocks */
    int nb = r->n_blocks;
    r->layers = (cuda_da3_layer *)calloc((size_t)nb, sizeof(cuda_da3_layer));
    for (int L = 0; L < nb; L++) {
        cuda_da3_layer *ly = &r->layers[L];
        char name[128];
        qtensor t;

#define ST_LOAD_F32(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    t = da3s_tensor(st, map, map_count, name); ly->field = upload_tensor_f32(&t);
#define ST_LOAD_BACKBONE_W(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    t = da3s_tensor(st, map, map_count, name); \
    ly->field = r->use_fp8 ? upload_tensor_fp8_e4m3(&t) : upload_tensor_f32_as_f16(&t);

        ST_LOAD_F32(ln1_w, "da3.blk.%d.ln1.weight", L)
        ST_LOAD_F32(ln1_b, "da3.blk.%d.ln1.bias", L)

        ST_LOAD_BACKBONE_W(attn_qkv_w, "da3.blk.%d.attn_qkv.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.attn_qkv.weight", L);
        t = da3s_tensor(st, map, map_count, name);
        ly->qkv_rows = t.n_rows; ly->qkv_cols = t.n_cols;
        ST_LOAD_F32(attn_qkv_b, "da3.blk.%d.attn_qkv.bias", L)

        ST_LOAD_F32(attn_q_norm_w, "da3.blk.%d.attn_q_norm.weight", L)
        ST_LOAD_F32(attn_q_norm_b, "da3.blk.%d.attn_q_norm.bias", L)
        ST_LOAD_F32(attn_k_norm_w, "da3.blk.%d.attn_k_norm.weight", L)
        ST_LOAD_F32(attn_k_norm_b, "da3.blk.%d.attn_k_norm.bias", L)
        ly->has_qk_norm = (ly->attn_q_norm_w != 0);

        ST_LOAD_BACKBONE_W(attn_out_w, "da3.blk.%d.attn_out.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.attn_out.weight", L);
        t = da3s_tensor(st, map, map_count, name);
        ly->out_rows = t.n_rows; ly->out_cols = t.n_cols;
        ST_LOAD_F32(attn_out_b, "da3.blk.%d.attn_out.bias", L)

        ST_LOAD_F32(ls1, "da3.blk.%d.ls1", L)
        ST_LOAD_F32(ls2, "da3.blk.%d.ls2", L)
        ST_LOAD_F32(ln2_w, "da3.blk.%d.ln2.weight", L)
        ST_LOAD_F32(ln2_b, "da3.blk.%d.ln2.bias", L)

        ST_LOAD_BACKBONE_W(ffn_gate_up_w, "da3.blk.%d.ffn_gate_up.weight", L)
        if (ly->ffn_gate_up_w) {
            snprintf(name, sizeof(name), "da3.blk.%d.ffn_gate_up.weight", L);
            t = da3s_tensor(st, map, map_count, name);
            ly->ffn_gu_rows = t.n_rows; ly->ffn_gu_cols = t.n_cols;
            ly->has_swiglu = 1;
        }
        ST_LOAD_F32(ffn_gate_up_b, "da3.blk.%d.ffn_gate_up.bias", L)

        ST_LOAD_BACKBONE_W(ffn_up_w, "da3.blk.%d.ffn_up.weight", L)
        if (ly->ffn_up_w) {
            snprintf(name, sizeof(name), "da3.blk.%d.ffn_up.weight", L);
            t = da3s_tensor(st, map, map_count, name);
            ly->ffn_up_rows = t.n_rows; ly->ffn_up_cols = t.n_cols;
        }
        ST_LOAD_F32(ffn_up_b, "da3.blk.%d.ffn_up.bias", L)

        ST_LOAD_BACKBONE_W(ffn_down_w, "da3.blk.%d.ffn_down.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.ffn_down.weight", L);
        t = da3s_tensor(st, map, map_count, name);
        ly->ffn_down_rows = t.n_rows; ly->ffn_down_cols = t.n_cols;
        ST_LOAD_F32(ffn_down_b, "da3.blk.%d.ffn_down.bias", L)

#undef ST_LOAD_F32
#undef ST_LOAD_BACKBONE_W
    }

    /* Upload DPT head weights to GPU */
    {
        qtensor t;
        char name[128];
        dpt_gpu_weights *dw = &r->dpt_w;

        t = da3s_tensor(st, map, map_count, "da3.head.norm.weight"); dw->norm_w = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.norm.bias");   dw->norm_b = upload_tensor_f32(&t);

        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.proj.%d.weight", i);
            t = da3s_tensor(st, map, map_count, name);
            dw->proj_w[i] = upload_tensor_f32_as_f16(&t);
            dw->proj_rows[i] = t.n_rows;
            snprintf(name, sizeof(name), "da3.head.proj.%d.bias", i);
            t = da3s_tensor(st, map, map_count, name);
            dw->proj_b[i] = upload_tensor_f32(&t);
        }

        {
            int oc0 = r->head_out_channels[0];
            t = da3s_tensor(st, map, map_count, "da3.head.upsample_0.weight");
            dw->upsample_0_w = upload_deconv_weight_f16(&t, oc0, oc0, 4, 4);
            t = da3s_tensor(st, map, map_count, "da3.head.upsample_0.bias");
            dw->upsample_0_b = upload_tensor_f32(&t);

            int oc1 = r->head_out_channels[1];
            t = da3s_tensor(st, map, map_count, "da3.head.upsample_1.weight");
            dw->upsample_1_w = upload_deconv_weight_f16(&t, oc1, oc1, 2, 2);
            t = da3s_tensor(st, map, map_count, "da3.head.upsample_1.bias");
            dw->upsample_1_b = upload_tensor_f32(&t);
        }
        t = da3s_tensor(st, map, map_count, "da3.head.downsample.weight");  dw->downsample_w = upload_tensor_f32_as_f16(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.downsample.bias");    dw->downsample_b = upload_tensor_f32(&t);

        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.adapter.%d.weight", i);
            t = da3s_tensor(st, map, map_count, name);
            dw->adapter_w[i] = upload_tensor_f32_as_f16(&t);
        }

        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.weight", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_out_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.bias", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_out_b[i] = upload_tensor_f32(&t);

            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.weight", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu1_c1_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.bias", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu1_c1_b[i] = upload_tensor_f32(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.weight", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu1_c2_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.bias", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu1_c2_b[i] = upload_tensor_f32(&t);
            dw->has_rcu1[i] = (dw->fuse_rcu1_c1_w[i] != 0);

            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.weight", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu2_c1_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.bias", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu2_c1_b[i] = upload_tensor_f32(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.weight", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu2_c2_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.bias", i);
            t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu2_c2_b[i] = upload_tensor_f32(&t);
            dw->has_rcu2[i] = (dw->fuse_rcu2_c1_w[i] != 0);
        }

        t = da3s_tensor(st, map, map_count, "da3.head.neck.weight");  dw->neck_w  = upload_tensor_f32_as_f16(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.neck.bias");    dw->neck_b  = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.out_0.weight"); dw->out_0_w = upload_tensor_f32_as_f16(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.out_0.bias");   dw->out_0_b = upload_tensor_f32(&t);
        dw->out_mid = t.n_cols; /* bias [out_mid] → n_cols = intermediate channels (32) */
        t = da3s_tensor(st, map, map_count, "da3.head.out_2.weight"); dw->out_2_w = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.out_2.bias");   dw->out_2_b = upload_tensor_f32(&t);

        if (r->verbose >= 1)
            fprintf(stderr, "cuda_da3: DPT head weights uploaded to GPU (out_mid=%d)\n", dw->out_mid);
    }

    /* Upload CameraDec weights (Phase 1) */
    {
        qtensor t;
        t = da3s_tensor(st, map, map_count, "da3.backbone_norm.weight");
        if (t.data) {
            r->cam_dec.backbone_norm_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.backbone_norm.bias");
            r->cam_dec.backbone_norm_b = upload_tensor_f32(&t);

            t = da3s_tensor(st, map, map_count, "da3.cam_dec.mlp.0.weight");
            r->cam_dec.mlp_w[0] = upload_tensor_f32_as_f16(&t);
            r->cam_dec.mlp_dim = t.n_rows;
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.mlp.0.bias");
            r->cam_dec.mlp_b[0] = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.mlp.2.weight");
            r->cam_dec.mlp_w[1] = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.mlp.2.bias");
            r->cam_dec.mlp_b[1] = upload_tensor_f32(&t);

            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_t.weight");
            r->cam_dec.fc_t_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_t.bias");
            r->cam_dec.fc_t_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_qvec.weight");
            r->cam_dec.fc_qvec_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_qvec.bias");
            r->cam_dec.fc_qvec_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_fov.weight");
            r->cam_dec.fc_fov_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_fov.bias");
            r->cam_dec.fc_fov_b = upload_tensor_f32(&t);
            r->cam_dec.loaded = 1;
            if (r->verbose >= 1)
                fprintf(stderr, "cuda_da3: CameraDec weights loaded (mlp_dim=%d)\n",
                        r->cam_dec.mlp_dim);
        }
    }

    /* Upload CameraEnc weights (Phase 3) */
    {
        qtensor t;
        char name[128];
        t = da3s_tensor(st, map, map_count, "da3.cam_enc.fc1.weight");
        if (t.data) {
            r->cam_enc.fc1_w = upload_tensor_f32_as_f16(&t);
            r->cam_enc.trunk_dim = t.n_rows;
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.fc1.bias");
            r->cam_enc.fc1_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.fc2.weight");
            r->cam_enc.fc2_w = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.fc2.bias");
            r->cam_enc.fc2_b = upload_tensor_f32(&t);

            t = da3s_tensor(st, map, map_count, "da3.cam_enc.trunk_norm.weight");
            r->cam_enc.trunk_norm_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.trunk_norm.bias");
            r->cam_enc.trunk_norm_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.token_norm.weight");
            r->cam_enc.token_norm_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.token_norm.bias");
            r->cam_enc.token_norm_b = upload_tensor_f32(&t);

            /* Count trunk blocks */
            r->cam_enc.n_trunk_blocks = 0;
            for (int L = 0; L < 8; L++) {
                snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.ln1.weight", L);
                t = da3s_tensor(st, map, map_count, name);
                if (!t.data) break;
                r->cam_enc.n_trunk_blocks = L + 1;
            }
            int ntb = r->cam_enc.n_trunk_blocks;
            if (ntb > 0) {
                r->cam_enc.trunk = (cuda_da3_layer *)calloc((size_t)ntb, sizeof(cuda_da3_layer));
                for (int L = 0; L < ntb; L++) {
                    cuda_da3_layer *ly = &r->cam_enc.trunk[L];

#define CE_F32(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    t = da3s_tensor(st, map, map_count, name); ly->field = upload_tensor_f32(&t);
#define CE_F16(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    t = da3s_tensor(st, map, map_count, name); ly->field = upload_tensor_f32_as_f16(&t);

                    CE_F32(ln1_w, "da3.cam_enc.trunk.%d.ln1.weight", L)
                    CE_F32(ln1_b, "da3.cam_enc.trunk.%d.ln1.bias", L)
                    CE_F16(attn_qkv_w, "da3.cam_enc.trunk.%d.attn_qkv.weight", L)
                    snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.attn_qkv.weight", L);
                    t = da3s_tensor(st, map, map_count, name);
                    ly->qkv_rows = t.n_rows; ly->qkv_cols = t.n_cols;
                    CE_F32(attn_qkv_b, "da3.cam_enc.trunk.%d.attn_qkv.bias", L)
                    CE_F16(attn_out_w, "da3.cam_enc.trunk.%d.attn_out.weight", L)
                    snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.attn_out.weight", L);
                    t = da3s_tensor(st, map, map_count, name);
                    ly->out_rows = t.n_rows; ly->out_cols = t.n_cols;
                    CE_F32(attn_out_b, "da3.cam_enc.trunk.%d.attn_out.bias", L)
                    CE_F32(ls1, "da3.cam_enc.trunk.%d.ls1", L)
                    CE_F32(ls2, "da3.cam_enc.trunk.%d.ls2", L)
                    CE_F32(ln2_w, "da3.cam_enc.trunk.%d.ln2.weight", L)
                    CE_F32(ln2_b, "da3.cam_enc.trunk.%d.ln2.bias", L)
                    CE_F16(ffn_up_w, "da3.cam_enc.trunk.%d.ffn_up.weight", L)
                    if (ly->ffn_up_w) {
                        snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.ffn_up.weight", L);
                        t = da3s_tensor(st, map, map_count, name);
                        ly->ffn_up_rows = t.n_rows; ly->ffn_up_cols = t.n_cols;
                    }
                    CE_F32(ffn_up_b, "da3.cam_enc.trunk.%d.ffn_up.bias", L)
                    CE_F16(ffn_down_w, "da3.cam_enc.trunk.%d.ffn_down.weight", L)
                    snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.ffn_down.weight", L);
                    t = da3s_tensor(st, map, map_count, name);
                    ly->ffn_down_rows = t.n_rows; ly->ffn_down_cols = t.n_cols;
                    CE_F32(ffn_down_b, "da3.cam_enc.trunk.%d.ffn_down.bias", L)
                    ly->has_qk_norm = 0;
                    ly->has_swiglu = 0;
#undef CE_F32
#undef CE_F16
                }
            }
            r->cam_enc.loaded = 1;
            if (r->verbose >= 1)
                fprintf(stderr, "cuda_da3: CameraEnc weights loaded (%d trunk blocks, dim=%d)\n",
                        ntb, r->cam_enc.trunk_dim);
        }
    }

    /* Upload DPT Aux Branch weights (Phase 2) */
    {
        qtensor t;
        char name[128];
        t = da3s_tensor(st, map, map_count, "da3.head.aux_fuse.0.out.weight");
        if (t.data) {
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.out.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_out_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.out.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_out_b[i] = upload_tensor_f32(&t);

                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv1.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_rcu1_c1_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv1.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_rcu1_c1_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv2.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_rcu1_c2_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv2.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_rcu1_c2_b[i] = upload_tensor_f32(&t);
                r->dpt_aux.has_rcu1[i] = (r->dpt_aux.fuse_rcu1_c1_w[i] != 0);

                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv1.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_rcu2_c1_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv1.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_rcu2_c1_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv2.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_rcu2_c2_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv2.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.fuse_rcu2_c2_b[i] = upload_tensor_f32(&t);
                r->dpt_aux.has_rcu2[i] = (r->dpt_aux.fuse_rcu2_c1_w[i] != 0);
            }

            /* Per-level output_conv1_aux (up to 5 Conv2d each) */
            for (int lv = 0; lv < 4; lv++) {
                r->dpt_aux.oc1_count[lv] = 0;
                for (int ci = 0; ci < 5; ci++) {
                    snprintf(name, sizeof(name), "da3.head.aux_oc1.%d.%d.weight", lv, ci);
                    t = da3s_tensor(st, map, map_count, name);
                    if (!t.data) break;
                    r->dpt_aux.oc1_w[lv][ci] = upload_tensor_f32(&t); /* F32: no activations → values exceed FP16 range */
                    /* 4D conv weight [C_out,C_in,kH,kW] reversed → [kW,kH,C_in,C_out]
                     * n_rows=kH (wrong!), dims[3]=C_out, dims[2]=C_in */
                    r->dpt_aux.oc1_co[lv][ci] = (t.n_dims == 4) ? (int)t.dims[3] : t.n_rows;
                    r->dpt_aux.oc1_ci[lv][ci] = (t.n_dims == 4) ? (int)t.dims[2] : (t.n_cols / 9);
                    snprintf(name, sizeof(name), "da3.head.aux_oc1.%d.%d.bias", lv, ci);
                    t = da3s_tensor(st, map, map_count, name);
                    r->dpt_aux.oc1_b[lv][ci] = upload_tensor_f32(&t);
                    r->dpt_aux.oc1_count[lv] = ci + 1;
                }
            }

            /* Per-level output_conv2_aux */
            for (int lv = 0; lv < 4; lv++) {
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.conv.weight", lv);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.oc2_conv_w[lv] = upload_tensor_f32(&t); /* F32: oc1 output exceeds FP16 range */
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.conv.bias", lv);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.oc2_conv_b[lv] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.gn.weight", lv);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.oc2_gn_w[lv] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.gn.bias", lv);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.oc2_gn_b[lv] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.out.weight", lv);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.oc2_out_w[lv] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.out.bias", lv);
                t = da3s_tensor(st, map, map_count, name);
                r->dpt_aux.oc2_out_b[lv] = upload_tensor_f32(&t);
            }
            r->dpt_aux.loaded = 1;
            if (r->verbose >= 1) {
                fprintf(stderr, "cuda_da3: DPT Aux Branch weights loaded\n");
                for (int lv = 0; lv < 4; lv++)
                    fprintf(stderr, "  aux oc1[%d]: %d layers, channels:",
                            lv, r->dpt_aux.oc1_count[lv]);
                for (int lv = 0; lv < 4; lv++) {
                    fprintf(stderr, "  aux oc1[%d]:", lv);
                    for (int ci = 0; ci < r->dpt_aux.oc1_count[lv]; ci++)
                        fprintf(stderr, " %d→%d", r->dpt_aux.oc1_ci[lv][ci],
                                r->dpt_aux.oc1_co[lv][ci]);
                    fprintf(stderr, "\n");
                }
            }
        }
    }

    /* Upload GSDPT weights (Phase 4) */
    {
        qtensor t;
        char name[128];
        t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.proj.0.weight");
        if (t.data) {
            dpt_gpu_weights *gdw = &r->gsdpt.dpt;

            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.norm.weight");
            gdw->norm_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.norm.bias");
            gdw->norm_b = upload_tensor_f32(&t);

            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.gsdpt.head.proj.%d.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->proj_w[i] = upload_tensor_f32_as_f16(&t);
                gdw->proj_rows[i] = t.n_rows;
                snprintf(name, sizeof(name), "da3.gsdpt.head.proj.%d.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->proj_b[i] = upload_tensor_f32(&t);
            }
            {
                int oc0 = r->head_out_channels[0];
                t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.upsample_0.weight");
                if (t.data) gdw->upsample_0_w = upload_deconv_weight_f16(&t, oc0, oc0, 4, 4);
                t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.upsample_0.bias");
                gdw->upsample_0_b = upload_tensor_f32(&t);
                int oc1 = r->head_out_channels[1];
                t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.upsample_1.weight");
                if (t.data) gdw->upsample_1_w = upload_deconv_weight_f16(&t, oc1, oc1, 2, 2);
                t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.upsample_1.bias");
                gdw->upsample_1_b = upload_tensor_f32(&t);
            }
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.downsample.weight");
            gdw->downsample_w = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.downsample.bias");
            gdw->downsample_b = upload_tensor_f32(&t);

            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.gsdpt.head.adapter.%d.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->adapter_w[i] = upload_tensor_f32_as_f16(&t);
            }
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.out.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_out_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.out.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_out_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv1.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_rcu1_c1_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv1.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_rcu1_c1_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv2.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_rcu1_c2_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv2.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_rcu1_c2_b[i] = upload_tensor_f32(&t);
                gdw->has_rcu1[i] = (gdw->fuse_rcu1_c1_w[i] != 0);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv1.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_rcu2_c1_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv1.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_rcu2_c1_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv2.weight", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_rcu2_c2_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv2.bias", i);
                t = da3s_tensor(st, map, map_count, name);
                gdw->fuse_rcu2_c2_b[i] = upload_tensor_f32(&t);
                gdw->has_rcu2[i] = (gdw->fuse_rcu2_c1_w[i] != 0);
            }
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.neck.weight");
            gdw->neck_w = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.neck.bias");
            gdw->neck_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.out_0.weight");
            gdw->out_0_w = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.out_0.bias");
            gdw->out_0_b = upload_tensor_f32(&t);
            gdw->out_mid = t.n_cols; /* bias [out_mid] → n_cols = 32 */
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.out_2.weight");
            gdw->out_2_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.out_2.bias");
            gdw->out_2_b = upload_tensor_f32(&t);
            /* Use bias (1D [out_ch]) for reliable channel count; weight is 4D [kW,kH,in,out] reversed */
            r->gsdpt.gs_out_channels = t.n_cols;

            /* Images merger */
            r->gsdpt.merger_ci[0] = 3;  r->gsdpt.merger_co[0] = 32;
            r->gsdpt.merger_ci[1] = 32; r->gsdpt.merger_co[1] = 64;
            r->gsdpt.merger_ci[2] = 64; r->gsdpt.merger_co[2] = 128;
            for (int mi = 0; mi < 3; mi++) {
                snprintf(name, sizeof(name), "da3.gsdpt.merger.%d.weight", mi);
                t = da3s_tensor(st, map, map_count, name);
                if (t.data) {
                    r->gsdpt.merger_w[mi] = upload_tensor_f32(&t); /* F32 for kl_conv2d (K may not be %16) */
                    /* 4D conv [C_out,C_in,kH,kW] reversed → [kW,kH,C_in,C_out] */
                    r->gsdpt.merger_co[mi] = (t.n_dims == 4) ? (int)t.dims[3] : t.n_rows;
                    snprintf(name, sizeof(name), "da3.gsdpt.merger.%d.bias", mi);
                    t = da3s_tensor(st, map, map_count, name);
                    r->gsdpt.merger_b[mi] = upload_tensor_f32(&t);
                }
            }
            r->gsdpt.loaded = 1;
            if (r->verbose >= 1) {
                fprintf(stderr, "cuda_da3: GSDPT weights loaded (out_ch=%d, out_mid=%d)\n",
                        r->gsdpt.gs_out_channels, gdw->out_mid);
                fprintf(stderr, "  merger channels: %d→%d→%d→%d\n",
                        3, r->gsdpt.merger_co[0], r->gsdpt.merger_co[1],
                        r->gsdpt.merger_co[2]);
            }
        }
    }

    /* Allocate backbone scratch buffers */
    int nt = r->n_tokens;
    int dim = r->dim;
    int np = r->n_patches;
    int gh = r->grid_h;
    int max_ffn = r->use_swiglu ? 2 * r->ffn_hidden : 4 * dim;

    CHECK_CU(cuMemAlloc(&r->d_hidden,   (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_hidden2,  (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_ln_buf,   (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_qkv,      (size_t)nt * 3 * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_attn_out, (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_ffn_buf,  (size_t)nt * max_ffn * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_ffn_mid,  (size_t)nt * r->ffn_hidden * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_proj_out, (size_t)nt * dim * sizeof(float)));
    CHECK_CU(cuMemAlloc(&r->d_pos_y,    (size_t)nt * sizeof(int)));
    CHECK_CU(cuMemAlloc(&r->d_pos_x,    (size_t)nt * sizeof(int)));
    CHECK_CU(cuMemAlloc(&r->d_img_norm, (size_t)3 * r->image_size * r->image_size * sizeof(float)));

    for (int i = 0; i < 4; i++)
        CHECK_CU(cuMemAlloc(&r->d_features[i], (size_t)nt * dim * sizeof(float)));

    /* Allocate DPT head scratch buffers */
    {
        int feat = r->head_features;
        int oc_max = r->head_out_channels[3];
        int sp_h[4], sp_w[4];
        sp_h[0] = sp_w[0] = (gh - 1) * 4 + 4;
        sp_h[1] = sp_w[1] = (gh - 1) * 2 + 2;
        sp_h[2] = sp_w[2] = gh;
        sp_h[3] = sp_w[3] = (gh + 2 - 3) / 2 + 1;
        int max_hw = sp_h[0] * sp_w[0];

        size_t large_sz = (size_t)feat * max_hw;
        if (large_sz < (size_t)np * 2 * dim)
            large_sz = (size_t)np * 2 * dim;

        CHECK_CU(cuMemAlloc(&r->d_dpt_cat,  large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_ln,   large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_proj, (size_t)np * oc_max * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_chw,  (size_t)oc_max * gh * gh * sizeof(float)));

        for (int i = 0; i < 4; i++) {
            int oc = r->head_out_channels[i];
            CHECK_CU(cuMemAlloc(&r->d_dpt_spatial[i],
                                 (size_t)oc * sp_h[i] * sp_w[i] * sizeof(float)));
            CHECK_CU(cuMemAlloc(&r->d_dpt_adapted[i],
                                 (size_t)feat * sp_h[i] * sp_w[i] * sizeof(float)));
        }

        CHECK_CU(cuMemAlloc(&r->d_dpt_fused, large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_tmp,   large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_tmp2,  large_sz * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_dpt_out,   (size_t)2 * max_hw * sizeof(float)));

        if (r->verbose >= 1)
            fprintf(stderr, "cuda_da3: DPT scratch buffers allocated (~%.1f MB)\n",
                    (float)(4 * large_sz + np * oc_max + oc_max * gh * gh + 2 * max_hw) * 4 / 1e6f);

        /* Aux DPT scratch: per-level 7-channel output + conv chain scratch
         * All output conv chains operate at level 0 spatial size (sp_h[0] x sp_w[0]) */
        if (r->dpt_aux.loaded) {
            for (int i = 0; i < 4; i++)
                CHECK_CU(cuMemAlloc(&r->d_aux_out[i],
                                     (size_t)7 * sp_h[0] * sp_w[0] * sizeof(float)));
            /* Scratch for output conv chains (max 256 channels * max_hw) */
            CHECK_CU(cuMemAlloc(&r->d_aux_scratch,
                                 (size_t)256 * max_hw * sizeof(float)));
            if (r->verbose >= 1)
                fprintf(stderr, "cuda_da3: DPT Aux scratch allocated\n");
        }

        /* GSDPT scratch: merger output + 38-channel output */
        if (r->gsdpt.loaded) {
            /* Merger output: 128 channels after 3 stride-2 convs on image_size.
             * Each stride-2 conv: out = (in + 2*1 - 3)/2 + 1
             * 518→259→130→65 for default image_size=518 */
            int mg_h = r->image_size, mg_w = r->image_size;
            for (int s = 0; s < 3; s++) {
                mg_h = (mg_h + 2 * 1 - 3) / 2 + 1;
                mg_w = (mg_w + 2 * 1 - 3) / 2 + 1;
            }
            r->gs_merger_h = mg_h; r->gs_merger_w = mg_w;
            CHECK_CU(cuMemAlloc(&r->d_gs_merged,
                                 (size_t)128 * mg_h * mg_w * sizeof(float)));
            int gs_oc = r->gsdpt.gs_out_channels;
            if (gs_oc < 2) gs_oc = 38;
            CHECK_CU(cuMemAlloc(&r->d_gs_out,
                                 (size_t)gs_oc * max_hw * sizeof(float)));
            if (r->verbose >= 1)
                fprintf(stderr, "cuda_da3: GSDPT scratch allocated (%d channels, merger=%dx%d)\n",
                        gs_oc, mg_h, mg_w);
        }
    }

    /* Upload position arrays for RoPE */
    int *py = (int *)calloc((size_t)nt, sizeof(int));
    int *px = (int *)calloc((size_t)nt, sizeof(int));
    for (int p = 0; p < np; p++) {
        py[1 + p] = p / r->grid_w;
        px[1 + p] = p % r->grid_w;
    }
    cuMemcpyHtoD(r->d_pos_y, py, (size_t)nt * sizeof(int));
    cuMemcpyHtoD(r->d_pos_x, px, (size_t)nt * sizeof(int));
    free(py); free(px);

    r->h_output = (float *)malloc((size_t)nt * dim * sizeof(float));
    r->cpu_model = NULL; /* No GGUF-based CPU model needed */
    r->loaded = 1;

    if (r->verbose >= 1) {
        fprintf(stderr, "cuda_da3: loaded %d blocks, dim=%d, tokens=%d, swiglu=%d (safetensors)\n",
                nb, dim, nt, r->use_swiglu);
        fprintf(stderr, "cuda_da3: modules: cam_dec=%d cam_enc=%d dpt_aux=%d gsdpt=%d\n",
                r->cam_dec.loaded, r->cam_enc.loaded, r->dpt_aux.loaded, r->gsdpt.loaded);
    }

    /* Free name mapping */
    for (int i = 0; i < map_count; i++) free((void *)map[i].gguf_name);
    free(map);

    /* Keep safetensors mmap'd - the data is used for zero-copy weight upload,
     * but all data has been copied to GPU at this point so we can close it */
    safetensors_close(st);

    return 0;
}

/* ======================================================================== */
/* Kernel launch helpers                                                    */
/* ======================================================================== */

static void kl_layernorm(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr src,
                          CUdeviceptr w, CUdeviceptr b, int n_tok, int dim) {
    float eps = r->ln_eps;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    cuLaunchKernel(r->fn_layernorm_f32, (unsigned)n_tok, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static void kl_gemm(cuda_da3_runner *r, CUdeviceptr Y, CUdeviceptr W_f16,
                     CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W_f16, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 255) / 256); /* 4 warps × 64 outputs = 256 per block */
    unsigned gy = (unsigned)((n_tok + 15) / 16);  /* 16 tokens per block (shared via smem) */
    cuLaunchKernel(r->fn_gemm_f16_f32, gx, gy, 1, 128, 1, 1,
                   16 * 16 * (unsigned)sizeof(float), r->stream, args, NULL);
}

static void kl_gemm_fp8(cuda_da3_runner *r, CUdeviceptr Y, CUdeviceptr W_fp8,
                          CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W_fp8, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 255) / 256);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    cuLaunchKernel(r->fn_gemm_fp8_f32, gx, gy, 1, 128, 1, 1,
                   16 * 32 * (unsigned)sizeof(float), r->stream, args, NULL);
}

/* Dispatch backbone GEMM: FP8 if available, else FP16 */
static void kl_backbone_gemm(cuda_da3_runner *r, CUdeviceptr Y, CUdeviceptr W,
                               CUdeviceptr X, CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    if (r->use_fp8)
        kl_gemm_fp8(r, Y, W, X, bias, n_out, n_in, n_tok);
    else
        kl_gemm(r, Y, W, X, bias, n_out, n_in, n_tok);
}

static void kl_qk_layernorm(cuda_da3_runner *r, CUdeviceptr vec, CUdeviceptr w,
                               CUdeviceptr b, int n_tok, int n_heads, int head_dim,
                               int stride) {
    if (!w) return;
    float eps = r->ln_eps;
    int grid = n_tok * n_heads;
    int threads = 64; /* >= head_dim=64, power of 2 */
    void *args[] = {&vec, &w, &b, &n_tok, &n_heads, &head_dim, &stride, &eps};
    cuLaunchKernel(r->fn_qk_layernorm_f32, (unsigned)grid, 1, 1,
                   (unsigned)threads, 1, 1, (unsigned)(threads * sizeof(float)),
                   r->stream, args, NULL);
}

static void kl_rope_2d(cuda_da3_runner *r, CUdeviceptr vec, CUdeviceptr pos_y,
                         CUdeviceptr pos_x, int n_tok, int n_heads, int head_dim,
                         int stride) {
    int quarter = head_dim / 4;
    int threads = n_heads * quarter;
    float freq_base = 10000.0f;
    void *args[] = {&vec, &pos_y, &pos_x, &n_tok, &n_heads, &head_dim, &stride, &freq_base};
    cuLaunchKernel(r->fn_rope_2d_f32, (unsigned)n_tok, 1, 1,
                   (unsigned)threads, 1, 1, 0, r->stream, args, NULL);
}

static void kl_kv_transpose(cuda_da3_runner *r, CUdeviceptr K_t, CUdeviceptr V_t,
                              CUdeviceptr qkv, int n_tok, int dim, int n_heads, int head_dim) {
    int total = n_tok * dim;
    int grid = (total + 255) / 256;
    void *args[] = {&K_t, &V_t, &qkv, &n_tok, &dim, &n_heads, &head_dim};
    cuLaunchKernel(r->fn_kv_transpose, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_attn_prefill(cuda_da3_runner *r, CUdeviceptr out, CUdeviceptr qkv,
                              CUdeviceptr K_t, CUdeviceptr V_t,
                              int n_tok, int dim, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    void *args[] = {&out, &qkv, &K_t, &V_t, &n_tok, &dim, &n_heads, &head_dim, &scale};
    unsigned gy = (unsigned)((n_tok + 63) / 64);
    cuLaunchKernel(r->fn_attn_prefill_f32,
                   (unsigned)n_heads, gy, 1,
                   128, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_swiglu(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr gate_up,
                        int hidden, int n_tok) {
    int total = hidden * n_tok;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &gate_up, &hidden, &n_tok};
    cuLaunchKernel(r->fn_swiglu_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_gelu(cuda_da3_runner *r, CUdeviceptr x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    cuLaunchKernel(r->fn_gelu_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_layerscale_add(cuda_da3_runner *r, CUdeviceptr hidden,
                                CUdeviceptr proj, CUdeviceptr gamma, int dim, int n) {
    if (!gamma) {
        /* No layerscale, just add */
        int grid = (n + 255) / 256;
        void *args[] = {&hidden, &proj, &n};
        cuLaunchKernel(r->fn_add_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                       0, r->stream, args, NULL);
        return;
    }
    int grid = (n + 255) / 256;
    void *args[] = {&hidden, &proj, &gamma, &dim, &n};
    cuLaunchKernel(r->fn_layerscale_add_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

/* ======================================================================== */
/* DPT head kernel launch helpers                                           */
/* ======================================================================== */

static void kl_cls_concat(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr src,
                            int np, int dim) {
    int total = np * 2 * dim;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &src, &np, &dim};
    cuLaunchKernel(r->fn_dpt_cls_concat, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_tok_to_chw(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr src,
                             int C, int gH, int gW) {
    int total = C * gH * gW;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &src, &C, &gH, &gW};
    cuLaunchKernel(r->fn_dpt_tok_to_chw, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_conv2d(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr src,
                        CUdeviceptr w, CUdeviceptr b, int H, int W,
                        int Ci, int Co, int kH, int kW, int stride, int pad) {
    int Ho = (H + 2 * pad - kH) / stride + 1;
    int Wo = (W + 2 * pad - kW) / stride + 1;
    int total = Co * Ho * Wo;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &src, &w, &b, &H, &W, &Ci, &Co, &kH, &kW, &stride, &pad};
    cuLaunchKernel(r->fn_conv2d_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

/* Conv2d via implicit im2col MMA GEMM. Weight must be FP16, K must be multiple of 16. */
static void kl_conv_gemm(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr src,
                           CUdeviceptr w_f16, CUdeviceptr bias, int H, int W,
                           int Ci, int Co, int kH, int kW, int stride, int pad) {
    int Ho = (H + 2 * pad - kH) / stride + 1;
    int Wo = (W + 2 * pad - kW) / stride + 1;
    int M = Ho * Wo;
    void *args[] = {&dst, &src, &w_f16, &bias, &H, &W, &Ci, &Co,
                    &kH, &kW, &stride, &pad};
    unsigned gx = (unsigned)((Co + 63) / 64);
    unsigned gy = (unsigned)((M + 63) / 64);
    cuLaunchKernel(r->fn_conv_gemm_f16_f32, gx, gy, 1, 128, 1, 1,
                   64 * 16 * (unsigned)sizeof(float), r->stream, args, NULL);
}

/* GEMM-based ConvTranspose2d for stride-aligned case (kH==stride, kW==stride).
 * W_f16 is pre-transposed [kH*kW*Co, Ci] FP16. X is [Hi*Wi, Ci] token-major.
 * Uses GEMM + scatter: GEMM -> scratch[Hi*Wi, kH*kW*Co], scatter -> dst[Co,Ho,Wo]. */
static void kl_deconv_gemm_scatter(cuda_da3_runner *r, CUdeviceptr dst,
                                     CUdeviceptr X, CUdeviceptr W_f16,
                                     CUdeviceptr bias, CUdeviceptr scratch,
                                     int Ci, int Co, int Hi, int Wi,
                                     int kH, int kW, int stride) {
    int N = kH * kW * Co;
    CUdeviceptr null_bias = 0;
    /* GEMM: scratch[Hi*Wi, kH*kW*Co] = W[kH*kW*Co, Ci] * X[Hi*Wi, Ci]^T */
    kl_gemm(r, scratch, W_f16, X, null_bias, N, Ci, Hi * Wi);
    /* Scatter: scratch -> dst[Co, Ho, Wo] + bias */
    int Ho = (Hi - 1) * stride + kH;
    int Wo = (Wi - 1) * stride + kW;
    int total = Co * Ho * Wo;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &scratch, &bias, &Co, &Hi, &Wi, &Ho, &Wo,
                    &kH, &kW, &stride, &stride};
    cuLaunchKernel(r->fn_deconv_scatter_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_bilinear(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr src,
                           int C, int Hi, int Wi, int Ho, int Wo) {
    int total = C * Ho * Wo;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &src, &C, &Hi, &Wi, &Ho, &Wo};
    cuLaunchKernel(r->fn_bilinear_upsample_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_relu_inplace(cuda_da3_runner *r, CUdeviceptr x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    cuLaunchKernel(r->fn_relu_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_add_inplace(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr src, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &n};
    cuLaunchKernel(r->fn_add_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_groupnorm(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr src,
                           CUdeviceptr w, CUdeviceptr b, int C, int HW) {
    float eps = 1e-5f;
    void *args[] = {&dst, &src, &w, &b, &C, &HW, &eps};
    cuLaunchKernel(r->fn_groupnorm_f32, (unsigned)C, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

/* Per-position channel LayerNorm: normalize C channels at each spatial position.
 * Data layout: [C, H, W], one thread per spatial position. */
static void kl_channel_layernorm(cuda_da3_runner *r, CUdeviceptr dst, CUdeviceptr src,
                                   CUdeviceptr w, CUdeviceptr b, int C, int HW) {
    float eps = 1e-5f;
    int grid = (HW + 255) / 256;
    void *args[] = {&dst, &src, &w, &b, &C, &HW, &eps};
    cuLaunchKernel(r->fn_channel_layernorm_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void kl_silu_inplace(cuda_da3_runner *r, CUdeviceptr x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    cuLaunchKernel(r->fn_silu_f32, (unsigned)grid, 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

/* RCU: relu -> conv3x3 -> relu -> conv3x3 + residual
 * Uses d_dpt_tmp and d_dpt_tmp2 as internal scratch.
 * Input x and output out must NOT alias d_dpt_tmp or d_dpt_tmp2.
 * c1w, c2w must be FP16 (for MMA conv_gemm kernel). */
static void kl_rcu(cuda_da3_runner *r, CUdeviceptr out, CUdeviceptr x,
                     CUdeviceptr c1w, CUdeviceptr c1b,
                     CUdeviceptr c2w, CUdeviceptr c2b,
                     int C, int H, int W) {
    int sz = C * H * W;
    /* tmp2 = relu(x) */
    cuMemcpyDtoDAsync(r->d_dpt_tmp2, x, (size_t)sz * sizeof(float), r->stream);
    kl_relu_inplace(r, r->d_dpt_tmp2, sz);
    /* tmp = conv3x3(tmp2) via MMA */
    int pad = 1, stride = 1, kS = 3;
    kl_conv_gemm(r, r->d_dpt_tmp, r->d_dpt_tmp2, c1w, c1b, H, W, C, C, kS, kS, stride, pad);
    /* relu(tmp) */
    kl_relu_inplace(r, r->d_dpt_tmp, sz);
    /* out = conv3x3(tmp) via MMA */
    kl_conv_gemm(r, out, r->d_dpt_tmp, c2w, c2b, H, W, C, C, kS, kS, stride, pad);
    /* out += x (residual) */
    kl_add_inplace(r, out, x, sz);
}

/* GPU RefineNet fusion block.
 * result_buf: output buffer (must be pre-sized to features*fH*fW).
 * IMPORTANT: deeper and result_buf may alias (both d_dpt_fused). We handle this
 * by consuming deeper (bilinear upsample) before overwriting result_buf.
 * Uses d_dpt_cat, d_dpt_ln as extra scratch (beyond d_dpt_tmp/tmp2 used by kl_rcu). */
static void gpu_refinenet(cuda_da3_runner *r, int stage, CUdeviceptr feat, int fH, int fW,
                            CUdeviceptr deeper, int dH, int dW, int features,
                            CUdeviceptr result_buf) {
    int sz = features * fH * fW;
    dpt_gpu_weights *dw = &r->dpt_w;

    /* If deeper exists: bilinear upsample BEFORE overwriting result_buf,
     * since deeper and result_buf may be the same buffer (d_dpt_fused). */
    if (deeper) {
        /* Bilinear upsample deeper to feat's spatial size -> d_dpt_cat */
        kl_bilinear(r, r->d_dpt_cat, deeper, features, dH, dW, fH, fW);
    }

    /* result_buf = copy of feat (safe now, deeper already consumed above) */
    cuMemcpyDtoDAsync(result_buf, feat, (size_t)sz * sizeof(float), r->stream);

    /* If deeper exists: optional RCU1 on upsampled + add to result */
    if (deeper) {
        if (dw->has_rcu1[stage]) {
            /* RCU1 on upsampled deeper -> d_dpt_ln (uses d_dpt_tmp/tmp2 internally) */
            kl_rcu(r, r->d_dpt_ln, r->d_dpt_cat,
                   dw->fuse_rcu1_c1_w[stage], dw->fuse_rcu1_c1_b[stage],
                   dw->fuse_rcu1_c2_w[stage], dw->fuse_rcu1_c2_b[stage],
                   features, fH, fW);
            kl_add_inplace(r, result_buf, r->d_dpt_ln, sz);
        } else {
            kl_add_inplace(r, result_buf, r->d_dpt_cat, sz);
        }
    }

    /* RCU2 (if weights exist) */
    if (dw->has_rcu2[stage]) {
        kl_rcu(r, r->d_dpt_cat, result_buf,
               dw->fuse_rcu2_c1_w[stage], dw->fuse_rcu2_c1_b[stage],
               dw->fuse_rcu2_c2_w[stage], dw->fuse_rcu2_c2_b[stage],
               features, fH, fW);
        cuMemcpyDtoDAsync(result_buf, r->d_dpt_cat, (size_t)sz * sizeof(float), r->stream);
    }

    /* out_conv: 1x1 convolution via MMA */
    kl_conv_gemm(r, r->d_dpt_cat, result_buf,
                  dw->fuse_out_w[stage], dw->fuse_out_b[stage],
                  fH, fW, features, features, 1, 1, 1, 0);
    cuMemcpyDtoDAsync(result_buf, r->d_dpt_cat, (size_t)sz * sizeof(float), r->stream);
}

/* Generalized RefineNet fusion that takes explicit weight pointers.
 * Used for both main DPT and aux branch. */
static void gpu_refinenet_w(cuda_da3_runner *r,
                              CUdeviceptr feat, int fH, int fW,
                              CUdeviceptr deeper, int dH, int dW, int features,
                              CUdeviceptr result_buf,
                              CUdeviceptr fuse_out_w, CUdeviceptr fuse_out_b,
                              CUdeviceptr rcu1_c1_w, CUdeviceptr rcu1_c1_b,
                              CUdeviceptr rcu1_c2_w, CUdeviceptr rcu1_c2_b,
                              CUdeviceptr rcu2_c1_w, CUdeviceptr rcu2_c1_b,
                              CUdeviceptr rcu2_c2_w, CUdeviceptr rcu2_c2_b,
                              int has_rcu1, int has_rcu2) {
    int sz = features * fH * fW;
    if (deeper) kl_bilinear(r, r->d_dpt_cat, deeper, features, dH, dW, fH, fW);
    cuMemcpyDtoDAsync(result_buf, feat, (size_t)sz * sizeof(float), r->stream);
    if (deeper) {
        if (has_rcu1) {
            kl_rcu(r, r->d_dpt_ln, r->d_dpt_cat,
                   rcu1_c1_w, rcu1_c1_b, rcu1_c2_w, rcu1_c2_b, features, fH, fW);
            kl_add_inplace(r, result_buf, r->d_dpt_ln, sz);
        } else {
            kl_add_inplace(r, result_buf, r->d_dpt_cat, sz);
        }
    }
    if (has_rcu2) {
        kl_rcu(r, r->d_dpt_cat, result_buf,
               rcu2_c1_w, rcu2_c1_b, rcu2_c2_w, rcu2_c2_b, features, fH, fW);
        cuMemcpyDtoDAsync(result_buf, r->d_dpt_cat, (size_t)sz * sizeof(float), r->stream);
    }
    kl_conv_gemm(r, r->d_dpt_cat, result_buf,
                  fuse_out_w, fuse_out_b, fH, fW, features, features, 1, 1, 1, 0);
    cuMemcpyDtoDAsync(result_buf, r->d_dpt_cat, (size_t)sz * sizeof(float), r->stream);
}

/* CameraDec: backbone_norm(CLS) → MLP → 3 linear heads → pose[9] (CPU) */
static void run_camera_dec(cuda_da3_runner *r, float *pose_out) {
    int dim = r->dim;
    int mlp_dim = r->cam_dec.mlp_dim;

    /* 1. LayerNorm on CLS token (token 0 of d_hidden) → d_ln_buf[0] */
    kl_layernorm(r, r->d_ln_buf, r->d_hidden,
                  r->cam_dec.backbone_norm_w, r->cam_dec.backbone_norm_b, 1, dim);

    /* 2. MLP layer 1: GEMM [mlp_dim, dim] × CLS[1, dim] + bias → d_attn_out[1, mlp_dim] */
    kl_gemm(r, r->d_attn_out, r->cam_dec.mlp_w[0], r->d_ln_buf,
             r->cam_dec.mlp_b[0], mlp_dim, dim, 1);
    kl_gelu(r, r->d_attn_out, mlp_dim);

    /* 3. MLP layer 2: GEMM [mlp_dim, mlp_dim] × hidden[1, mlp_dim] → d_proj_out[1, mlp_dim] */
    kl_gemm(r, r->d_proj_out, r->cam_dec.mlp_w[1], r->d_attn_out,
             r->cam_dec.mlp_b[1], mlp_dim, mlp_dim, 1);
    kl_gelu(r, r->d_proj_out, mlp_dim);

    /* 4. Download MLP output to CPU for tiny matmuls */
    float *h_mlp = (float *)malloc((size_t)mlp_dim * sizeof(float));
    cuStreamSynchronize(r->stream);
    cuMemcpyDtoH(h_mlp, r->d_proj_out, (size_t)mlp_dim * sizeof(float));

    /* Download head weights */
    float *h_fc_t_w = (float *)malloc(3 * mlp_dim * sizeof(float));
    float *h_fc_t_b = (float *)malloc(3 * sizeof(float));
    float *h_fc_q_w = (float *)malloc(4 * mlp_dim * sizeof(float));
    float *h_fc_q_b = (float *)malloc(4 * sizeof(float));
    float *h_fc_f_w = (float *)malloc(2 * mlp_dim * sizeof(float));
    float *h_fc_f_b = (float *)malloc(2 * sizeof(float));
    cuMemcpyDtoH(h_fc_t_w, r->cam_dec.fc_t_w, 3 * mlp_dim * sizeof(float));
    cuMemcpyDtoH(h_fc_t_b, r->cam_dec.fc_t_b, 3 * sizeof(float));
    cuMemcpyDtoH(h_fc_q_w, r->cam_dec.fc_qvec_w, 4 * mlp_dim * sizeof(float));
    cuMemcpyDtoH(h_fc_q_b, r->cam_dec.fc_qvec_b, 4 * sizeof(float));
    cuMemcpyDtoH(h_fc_f_w, r->cam_dec.fc_fov_w, 2 * mlp_dim * sizeof(float));
    cuMemcpyDtoH(h_fc_f_b, r->cam_dec.fc_fov_b, 2 * sizeof(float));

    /* CPU matmuls: pose = [t(3), qvec(4), fov(2)] */
    for (int o = 0; o < 3; o++) {
        float s = h_fc_t_b[o];
        for (int k = 0; k < mlp_dim; k++) s += h_fc_t_w[o * mlp_dim + k] * h_mlp[k];
        pose_out[o] = s;
    }
    for (int o = 0; o < 4; o++) {
        float s = h_fc_q_b[o];
        for (int k = 0; k < mlp_dim; k++) s += h_fc_q_w[o * mlp_dim + k] * h_mlp[k];
        pose_out[3 + o] = s;
    }
    for (int o = 0; o < 2; o++) {
        float s = h_fc_f_b[o];
        for (int k = 0; k < mlp_dim; k++) s += h_fc_f_w[o * mlp_dim + k] * h_mlp[k];
        pose_out[7 + o] = s;
    }

    free(h_mlp);
    free(h_fc_t_w); free(h_fc_t_b);
    free(h_fc_q_w); free(h_fc_q_b);
    free(h_fc_f_w); free(h_fc_f_b);
}

/* Aux DPT branch: run aux refinenet fusion + output conv chains → rays[6,H,W] + conf[H,W] + sky[H,W]
 * Reuses d_dpt_adapted[4] from main DPT token processing (shared features). */
static void run_aux_dpt(cuda_da3_runner *r, int *sp_h, int *sp_w, int features,
                          CUdeviceptr *d_aux_out) {
    int feat = features;

    /* Bottom-up aux RefineNet fusion (same structure as main but with aux weights) */
    CUdeviceptr aux_fused = r->d_dpt_fused; /* reuse after main DPT completes */

    /* Level 3 (deepest) */
    gpu_refinenet_w(r, r->d_dpt_adapted[3], sp_h[3], sp_w[3],
                     0, 0, 0, feat, aux_fused,
                     r->dpt_aux.fuse_out_w[3], r->dpt_aux.fuse_out_b[3],
                     r->dpt_aux.fuse_rcu1_c1_w[3], r->dpt_aux.fuse_rcu1_c1_b[3],
                     r->dpt_aux.fuse_rcu1_c2_w[3], r->dpt_aux.fuse_rcu1_c2_b[3],
                     r->dpt_aux.fuse_rcu2_c1_w[3], r->dpt_aux.fuse_rcu2_c1_b[3],
                     r->dpt_aux.fuse_rcu2_c2_w[3], r->dpt_aux.fuse_rcu2_c2_b[3],
                     r->dpt_aux.has_rcu1[3], r->dpt_aux.has_rcu2[3]);
    int fh = sp_h[3], fw = sp_w[3];

    /* Level 2 */
    gpu_refinenet_w(r, r->d_dpt_adapted[2], sp_h[2], sp_w[2],
                     aux_fused, fh, fw, feat, aux_fused,
                     r->dpt_aux.fuse_out_w[2], r->dpt_aux.fuse_out_b[2],
                     r->dpt_aux.fuse_rcu1_c1_w[2], r->dpt_aux.fuse_rcu1_c1_b[2],
                     r->dpt_aux.fuse_rcu1_c2_w[2], r->dpt_aux.fuse_rcu1_c2_b[2],
                     r->dpt_aux.fuse_rcu2_c1_w[2], r->dpt_aux.fuse_rcu2_c1_b[2],
                     r->dpt_aux.fuse_rcu2_c2_w[2], r->dpt_aux.fuse_rcu2_c2_b[2],
                     r->dpt_aux.has_rcu1[2], r->dpt_aux.has_rcu2[2]);
    fh = sp_h[2]; fw = sp_w[2];

    /* Level 1 */
    gpu_refinenet_w(r, r->d_dpt_adapted[1], sp_h[1], sp_w[1],
                     aux_fused, fh, fw, feat, aux_fused,
                     r->dpt_aux.fuse_out_w[1], r->dpt_aux.fuse_out_b[1],
                     r->dpt_aux.fuse_rcu1_c1_w[1], r->dpt_aux.fuse_rcu1_c1_b[1],
                     r->dpt_aux.fuse_rcu1_c2_w[1], r->dpt_aux.fuse_rcu1_c2_b[1],
                     r->dpt_aux.fuse_rcu2_c1_w[1], r->dpt_aux.fuse_rcu2_c1_b[1],
                     r->dpt_aux.fuse_rcu2_c2_w[1], r->dpt_aux.fuse_rcu2_c2_b[1],
                     r->dpt_aux.has_rcu1[1], r->dpt_aux.has_rcu2[1]);
    fh = sp_h[1]; fw = sp_w[1];

    /* Level 0 */
    gpu_refinenet_w(r, r->d_dpt_adapted[0], sp_h[0], sp_w[0],
                     aux_fused, fh, fw, feat, aux_fused,
                     r->dpt_aux.fuse_out_w[0], r->dpt_aux.fuse_out_b[0],
                     r->dpt_aux.fuse_rcu1_c1_w[0], r->dpt_aux.fuse_rcu1_c1_b[0],
                     r->dpt_aux.fuse_rcu1_c2_w[0], r->dpt_aux.fuse_rcu1_c2_b[0],
                     r->dpt_aux.fuse_rcu2_c1_w[0], r->dpt_aux.fuse_rcu2_c1_b[0],
                     r->dpt_aux.fuse_rcu2_c2_w[0], r->dpt_aux.fuse_rcu2_c2_b[0],
                     r->dpt_aux.has_rcu1[0], r->dpt_aux.has_rcu2[0]);

    /* Debug: check aux_fused values after refinenet */
    if (r->verbose >= 2) {
        cuStreamSynchronize(r->stream);
        float dbg[8];
        cuMemcpyDtoH(dbg, aux_fused, sizeof(dbg));
        fprintf(stderr, "  aux_fused[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
    }

    /* Only the LAST level (index 3, finest resolution) is used for output.
     * Python: fused_aux_pyr[-1] → output_conv2_aux[-1] → final output.
     * output_conv1_aux[3]: 5 Conv2d, no activations (dense indices .0-.4)
     * output_conv2_aux[3]: Conv2d(128,32,3) → LayerNorm(32) → ReLU → Conv2d(32,7,1) */
    {
        int lv = 3;  /* last level = finest */
        int oh = sp_h[0], ow = sp_w[0];

        /* output_conv1_aux[3]: 5 Conv2d chain (NO activations) */
        CUdeviceptr cur = r->d_aux_scratch;
        cuMemcpyDtoDAsync(cur, aux_fused, (size_t)feat * oh * ow * sizeof(float), r->stream);
        int ci = feat;
        for (int ci_idx = 0; ci_idx < r->dpt_aux.oc1_count[lv]; ci_idx++) {
            int co = r->dpt_aux.oc1_co[lv][ci_idx];
            CUdeviceptr dst_buf = (ci_idx % 2 == 0) ? r->d_dpt_tmp : r->d_aux_scratch;
            CUdeviceptr src_buf = (ci_idx % 2 == 0) ? r->d_aux_scratch : r->d_dpt_tmp;
            if (ci_idx == 0) src_buf = cur;
            kl_conv2d(r, dst_buf, src_buf,
                       r->dpt_aux.oc1_w[lv][ci_idx], r->dpt_aux.oc1_b[lv][ci_idx],
                       oh, ow, ci, co, 3, 3, 1, 1);
            if (r->verbose >= 2) {
                cuStreamSynchronize(r->stream);
                float dbg[4];
                cuMemcpyDtoH(dbg, dst_buf, sizeof(dbg));
                fprintf(stderr, "    oc1[3].%d (%d→%d): %.4e %.4e %.4e %.4e\n",
                        ci_idx, ci, co, dbg[0], dbg[1], dbg[2], dbg[3]);
            }
            ci = co;
            cur = dst_buf;
        }

        /* output_conv2_aux[3]: Conv2d(128,32,3) + LayerNorm(32) + ReLU + Conv2d(32,7,1) */
        CUdeviceptr ln_in = r->d_dpt_tmp2;
        kl_conv2d(r, ln_in, cur,
                   r->dpt_aux.oc2_conv_w[lv], r->dpt_aux.oc2_conv_b[lv],
                   oh, ow, ci, 32, 3, 3, 1, 1);
        if (r->verbose >= 2) {
            cuStreamSynchronize(r->stream);
            float dbg[8];
            cuMemcpyDtoH(dbg, ln_in, sizeof(dbg));
            fprintf(stderr, "  aux oc2 conv[0..7]: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
            /* Also check spatial variation: read middle spatial position */
            int mid = 32 * (oh * ow / 2);
            cuMemcpyDtoH(dbg, ln_in + (size_t)mid * sizeof(float), sizeof(dbg));
            fprintf(stderr, "  aux oc2 conv[mid]: %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }
        CUdeviceptr relu_src = ln_in;
        /* Use LayerNorm weights for this level (NULL = default init w=1, b=0).
         * Only level 0 has trained LayerNorm weights in safetensors;
         * levels 1-3 use default init (normalize without learned scale/bias). */
        CUdeviceptr ln_w = r->dpt_aux.oc2_gn_w[lv];  /* may be 0 (NULL) */
        CUdeviceptr ln_b = r->dpt_aux.oc2_gn_b[lv];  /* may be 0 (NULL) */
        {
            CUdeviceptr ln_out = r->d_aux_scratch;
            kl_channel_layernorm(r, ln_out, ln_in, ln_w, ln_b, 32, oh * ow);
            relu_src = ln_out;
            if (r->verbose >= 2) {
                cuStreamSynchronize(r->stream);
                float dbg[8];
                cuMemcpyDtoH(dbg, ln_out, sizeof(dbg));
                fprintf(stderr, "  aux oc2 ln[0..7]: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",
                        dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
                int mid = 32 * (oh * ow / 2);
                cuMemcpyDtoH(dbg, ln_out + (size_t)mid * sizeof(float), sizeof(dbg));
                fprintf(stderr, "  aux oc2 ln[mid]: %.4e %.4e %.4e %.4e\n",
                        dbg[0], dbg[1], dbg[2], dbg[3]);
            }
        }
        kl_relu_inplace(r, relu_src, 32 * oh * ow);
        if (r->verbose >= 2) {
            cuStreamSynchronize(r->stream);
            float dbg[8];
            cuMemcpyDtoH(dbg, relu_src, sizeof(dbg));
            fprintf(stderr, "  aux oc2 relu[0..7]: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
        }
        kl_conv2d(r, d_aux_out[0], relu_src,
                   r->dpt_aux.oc2_out_w[lv], r->dpt_aux.oc2_out_b[lv],
                   oh, ow, 32, 7, 1, 1, 1, 0);
        if (r->verbose >= 2) {
            cuStreamSynchronize(r->stream);
            float dbg[8];
            cuMemcpyDtoH(dbg, d_aux_out[0], sizeof(dbg));
            fprintf(stderr, "  aux final[0..7]: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
            int mid = 7 * (oh * ow / 2);
            cuMemcpyDtoH(dbg, d_aux_out[0] + (size_t)mid * sizeof(float), sizeof(dbg));
            fprintf(stderr, "  aux final[mid]: %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }
    }
}

/* ======================================================================== */
/* Public API: predict (GPU backbone + GPU DPT head)                        */
/* ======================================================================== */

da3_cuda_result cuda_da3_predict(cuda_da3_runner *r, const uint8_t *rgb,
                                   int img_w, int img_h) {
    da3_full_result full = cuda_da3_predict_full(r, rgb, img_w, img_h, DA3_OUTPUT_DEPTH, NULL);
    da3_cuda_result result = {0};
    result.depth = full.depth;
    result.confidence = full.confidence;
    result.width = full.width;
    result.height = full.height;
    /* Don't free full since we transferred depth/confidence ownership */
    free(full.rays); free(full.ray_confidence); free(full.sky_seg);
    free(full.gaussians); free(full.metric_depth);
    return result;
}

/* ======================================================================== */
/* Public API: predict_full (all output modalities)                         */
/* ======================================================================== */

da3_full_result cuda_da3_predict_full(cuda_da3_runner *r, const uint8_t *rgb,
                                        int img_w, int img_h, int output_flags,
                                        const float *pose_in) {
    da3_full_result result = {0};
    if (!r->loaded) return result;

    int dim = r->dim;
    int nt = r->n_tokens;
    int np = r->n_patches;
    int gh = r->grid_h, gw = r->grid_w;
    int ps = r->patch_size;
    int target_h = gh * ps, target_w = gw * ps;

    double t0, t1;
    struct timespec ts;

    /* ─── GPU: Preprocess + Patch Embed + CLS + PosEmbed ─── */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    /* Upload raw RGB to GPU (reuse pre-allocated buffer) */
    size_t img_bytes = (size_t)img_w * img_h * 3;
    if (img_bytes > r->d_img_raw_cap) {
        if (r->d_img_raw) cuMemFree(r->d_img_raw);
        cuMemAlloc(&r->d_img_raw, img_bytes);
        r->d_img_raw_cap = img_bytes;
    }
    CUdeviceptr d_img_raw = r->d_img_raw;
    cuMemcpyHtoD(d_img_raw, rgb, img_bytes);

    /* Resize + normalize on GPU -> d_img_norm [3, target_h, target_w] */
    {
        int total = target_h * target_w;
        int grid = (total + 255) / 256;
        float istd0 = 1.0f/r->image_std[0], istd1 = 1.0f/r->image_std[1], istd2 = 1.0f/r->image_std[2];
        float m0 = r->image_mean[0], m1 = r->image_mean[1], m2 = r->image_mean[2];
        void *args[] = {&r->d_img_norm, &d_img_raw, &img_w, &img_h, &target_w, &target_h,
                        &m0, &m1, &m2, &istd0, &istd1, &istd2};
        cuLaunchKernel(r->fn_resize_normalize, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }

    /* Patch embedding on GPU -> d_hidden[1..np] */
    {
        int img_dim = target_w;  /* = gw * ps */
        void *args[] = {&r->d_hidden, &r->d_img_norm, &r->d_patch_embed_w, &r->d_patch_embed_b,
                        &gw, &dim, &ps, &img_dim};
        cuLaunchKernel(r->fn_patch_embed_conv2d, (unsigned)np, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }

    /* CLS token + positional embedding on GPU */
    {
        int total = nt * dim;
        int grid = (total + 255) / 256;
        void *args[] = {&r->d_hidden, &r->d_cls_token, &r->d_pos_embed, &nt, &dim};
        cuLaunchKernel(r->fn_cls_pos_embed, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts);
    t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "cuda_da3: GPU preprocess+embed: %.1f ms\n", (t1-t0)*1000);

    /* ─── GPU: 12 Transformer Blocks ─── */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    int stride_3dim = 3 * dim; /* stride for Q/K in interleaved QKV buffer */

    /* Per-kernel-type profiling (verbose >= 2) */
    int do_prof = (r->verbose >= 2);
    double prof_ln = 0, prof_gemm = 0, prof_bias = 0, prof_qknorm = 0;
    double prof_rope = 0, prof_attn = 0, prof_act = 0, prof_ls = 0, prof_feat = 0;
    double pt0, pt1;

#define PROF_START() do { if (do_prof) { cuStreamSynchronize(r->stream); \
    clock_gettime(CLOCK_MONOTONIC, &ts); pt0 = ts.tv_sec + ts.tv_nsec * 1e-9; } } while(0)
#define PROF_END(acc) do { if (do_prof) { cuStreamSynchronize(r->stream); \
    clock_gettime(CLOCK_MONOTONIC, &ts); pt1 = ts.tv_sec + ts.tv_nsec * 1e-9; acc += pt1 - pt0; } } while(0)

    for (int L = 0; L < r->n_blocks; L++) {
        cuda_da3_layer *ly = &r->layers[L];

        /* 1. LayerNorm 1 */
        PROF_START();
        kl_layernorm(r, r->d_ln_buf, r->d_hidden, ly->ln1_w, ly->ln1_b, nt, dim);
        PROF_END(prof_ln);

        /* 2. QKV projection (bias fused) */
        PROF_START();
        kl_backbone_gemm(r, r->d_qkv, ly->attn_qkv_w, r->d_ln_buf, ly->attn_qkv_b,
                 ly->qkv_rows, ly->qkv_cols, nt);
        PROF_END(prof_gemm);

        /* 3. QK Normalization (stride=3*dim for interleaved QKV) */
        if (L >= r->qk_norm_start && ly->has_qk_norm) {
            PROF_START();
            /* Q starts at offset 0 */
            kl_qk_layernorm(r, r->d_qkv, ly->attn_q_norm_w, ly->attn_q_norm_b,
                             nt, r->n_heads, r->head_dim, stride_3dim);
            /* K starts at offset dim */
            CUdeviceptr k_base = r->d_qkv + (size_t)dim * sizeof(float);
            kl_qk_layernorm(r, k_base, ly->attn_k_norm_w, ly->attn_k_norm_b,
                             nt, r->n_heads, r->head_dim, stride_3dim);
            PROF_END(prof_qknorm);
        }

        /* 4. RoPE 2D (CLS at pos (0,0) -> identity rotation, safe to apply) */
        if (L >= r->rope_start) {
            PROF_START();
            /* Q: RoPE on all tokens */
            kl_rope_2d(r, r->d_qkv, r->d_pos_y, r->d_pos_x,
                        nt, r->n_heads, r->head_dim, stride_3dim);
            /* K: starts at offset dim */
            CUdeviceptr k_base = r->d_qkv + (size_t)dim * sizeof(float);
            kl_rope_2d(r, k_base, r->d_pos_y, r->d_pos_x,
                        nt, r->n_heads, r->head_dim, stride_3dim);
            PROF_END(prof_rope);
        }

        /* 5a. Transpose K,V to contiguous per-head layout (reuse d_ffn_buf) */
        {
            CUdeviceptr K_t = r->d_ffn_buf;
            CUdeviceptr V_t = r->d_ffn_buf + (size_t)nt * dim * sizeof(float);
            kl_kv_transpose(r, K_t, V_t, r->d_qkv, nt, dim, r->n_heads, r->head_dim);

            /* 5b. Multi-head attention (full sequence) */
            PROF_START();
            kl_attn_prefill(r, r->d_attn_out, r->d_qkv, K_t, V_t,
                             nt, dim, r->n_heads, r->head_dim);
            PROF_END(prof_attn);
        }

        /* 6. Output projection (bias fused) */
        PROF_START();
        kl_backbone_gemm(r, r->d_proj_out, ly->attn_out_w, r->d_attn_out, ly->attn_out_b,
                 ly->out_rows, ly->out_cols, nt);
        PROF_END(prof_gemm);

        /* 7. LayerScale 1 + residual */
        PROF_START();
        kl_layerscale_add(r, r->d_hidden, r->d_proj_out, ly->ls1, dim, nt * dim);
        PROF_END(prof_ls);

        /* 8. LayerNorm 2 */
        PROF_START();
        kl_layernorm(r, r->d_ln_buf, r->d_hidden, ly->ln2_w, ly->ln2_b, nt, dim);
        PROF_END(prof_ln);

        /* 9. FFN */
        if (ly->has_swiglu && ly->ffn_gate_up_w) {
            /* SwiGLU: gate_up projection (bias fused) */
            PROF_START();
            kl_backbone_gemm(r, r->d_ffn_buf, ly->ffn_gate_up_w, r->d_ln_buf, ly->ffn_gate_up_b,
                     ly->ffn_gu_rows, ly->ffn_gu_cols, nt);
            PROF_END(prof_gemm);
            /* SwiGLU activation -> d_ffn_mid */
            PROF_START();
            int hid = ly->ffn_gu_rows / 2;
            kl_swiglu(r, r->d_ffn_mid, r->d_ffn_buf, hid, nt);
            PROF_END(prof_act);
            /* Down projection (bias fused) */
            PROF_START();
            kl_backbone_gemm(r, r->d_proj_out, ly->ffn_down_w, r->d_ffn_mid, ly->ffn_down_b,
                     ly->ffn_down_rows, hid, nt);
            PROF_END(prof_gemm);
        } else if (ly->ffn_up_w) {
            /* GELU MLP (bias fused) */
            PROF_START();
            kl_backbone_gemm(r, r->d_ffn_buf, ly->ffn_up_w, r->d_ln_buf, ly->ffn_up_b,
                     ly->ffn_up_rows, ly->ffn_up_cols, nt);
            PROF_END(prof_gemm);
            PROF_START();
            kl_gelu(r, r->d_ffn_buf, nt * ly->ffn_up_rows);
            PROF_END(prof_act);
            PROF_START();
            kl_backbone_gemm(r, r->d_proj_out, ly->ffn_down_w, r->d_ffn_buf, ly->ffn_down_b,
                     ly->ffn_down_rows, ly->ffn_up_rows, nt);
            PROF_END(prof_gemm);
        }

        /* 10. LayerScale 2 + residual */
        PROF_START();
        kl_layerscale_add(r, r->d_hidden, r->d_proj_out, ly->ls2, dim, nt * dim);
        PROF_END(prof_ls);

        /* 11. Save features at specified layers */
        PROF_START();
        for (int fi = 0; fi < 4; fi++) {
            if (L == r->feature_layers[fi]) {
                cuMemcpyDtoDAsync(r->d_features[fi], r->d_hidden,
                                   (size_t)nt * dim * sizeof(float), r->stream);
            }
        }
        PROF_END(prof_feat);
    }

#undef PROF_START
#undef PROF_END

    /* Synchronize after backbone */
    cuStreamSynchronize(r->stream);

    clock_gettime(CLOCK_MONOTONIC, &ts);
    t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "cuda_da3: GPU backbone (%d blocks): %.1f ms\n",
                r->n_blocks, (t1-t0)*1000);
    if (do_prof) {
        double total = prof_ln + prof_gemm + prof_bias + prof_qknorm + prof_rope
                     + prof_attn + prof_act + prof_ls + prof_feat;
        fprintf(stderr, "  backbone profile (sync'd, 12 blocks total):\n");
        fprintf(stderr, "    layernorm:  %6.1f ms (%4.1f%%)\n", prof_ln*1000, prof_ln/total*100);
        fprintf(stderr, "    gemm:       %6.1f ms (%4.1f%%)\n", prof_gemm*1000, prof_gemm/total*100);
        fprintf(stderr, "    add_bias:   %6.1f ms (%4.1f%%)\n", prof_bias*1000, prof_bias/total*100);
        fprintf(stderr, "    qk_norm:    %6.1f ms (%4.1f%%)\n", prof_qknorm*1000, prof_qknorm/total*100);
        fprintf(stderr, "    rope:       %6.1f ms (%4.1f%%)\n", prof_rope*1000, prof_rope/total*100);
        fprintf(stderr, "    attention:  %6.1f ms (%4.1f%%)\n", prof_attn*1000, prof_attn/total*100);
        fprintf(stderr, "    activation: %6.1f ms (%4.1f%%)\n", prof_act*1000, prof_act/total*100);
        fprintf(stderr, "    layerscale: %6.1f ms (%4.1f%%)\n", prof_ls*1000, prof_ls/total*100);
        fprintf(stderr, "    feat_save:  %6.1f ms (%4.1f%%)\n", prof_feat*1000, prof_feat/total*100);
        fprintf(stderr, "    sum:        %6.1f ms (sync overhead: %.1f ms)\n",
                total*1000, ((t1-t0) - total)*1000);
    }

    /* Debug: check CUDA error after backbone */
    if (r->verbose >= 2) {
        CUresult err = cuCtxSynchronize();
        fprintf(stderr, "  CUDA after backbone: %d\n", (int)err);
        /* Dump hidden stats after backbone */
        int _nt = r->n_tokens, _dim = r->dim;
        float *_hbuf = (float *)malloc((size_t)_nt * _dim * sizeof(float));
        cuMemcpyDtoH(_hbuf, r->d_hidden, (size_t)_nt * _dim * sizeof(float));
        float _hmin = _hbuf[0], _hmax = _hbuf[0], _hsum = 0;
        for (int _i = 0; _i < _nt * _dim; _i++) {
            if (_hbuf[_i] < _hmin) _hmin = _hbuf[_i];
            if (_hbuf[_i] > _hmax) _hmax = _hbuf[_i];
            _hsum += _hbuf[_i];
        }
        fprintf(stderr, "  GPU hidden after backbone: min=%.4f max=%.4f mean=%.6f\n",
                _hmin, _hmax, _hsum / (_nt * _dim));
        fprintf(stderr, "  GPU hidden[tok1][0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                _hbuf[_dim], _hbuf[_dim+1], _hbuf[_dim+2], _hbuf[_dim+3],
                _hbuf[_dim+4], _hbuf[_dim+5], _hbuf[_dim+6], _hbuf[_dim+7]);
        free(_hbuf);
    }

    /* ─── CameraDec (Phase 1): pose estimation ─── */
    if (r->cam_dec.loaded && (output_flags & DA3_OUTPUT_POSE)) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double cd0 = ts.tv_sec + ts.tv_nsec * 1e-9;

        if (pose_in) {
            /* Use user-provided pose */
            memcpy(result.pose, pose_in, 9 * sizeof(float));
        } else {
            run_camera_dec(r, result.pose);
        }
        result.has_pose = 1;

        clock_gettime(CLOCK_MONOTONIC, &ts);
        double cd1 = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_da3: CameraDec: %.1f ms (pose=[%.3f,%.3f,%.3f, %.3f,%.3f,%.3f,%.3f, %.3f,%.3f])\n",
                    (cd1-cd0)*1000, result.pose[0], result.pose[1], result.pose[2],
                    result.pose[3], result.pose[4], result.pose[5], result.pose[6],
                    result.pose[7], result.pose[8]);
    }

    /* ─── GPU: DPT Head ─── */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    if (!r->dpt_w.proj_w[0]) {
        fprintf(stderr, "cuda_da3: no DPT head weights on GPU, returning empty result\n");
        return result;
    }

    dpt_gpu_weights *dw = &r->dpt_w;
    int feat = r->head_features;           /* 64 */
    int head_dim_in = dim * 2;             /* CLS concat: 768 */

    /* Spatial dimensions after alignment */
    int sp_h[4], sp_w[4];
    sp_h[0] = sp_w[0] = (gh - 1) * 4 + 4; /* 148 */
    sp_h[1] = sp_w[1] = (gh - 1) * 2 + 2; /* 74 */
    sp_h[2] = sp_w[2] = gh;                /* 37 */
    sp_h[3] = sp_w[3] = (gh + 2 - 3) / 2 + 1; /* 19 */

    /* DPT profiling accumulators */
    double dpt_tok = 0, dpt_refine = 0, dpt_output = 0, dpt_upsample = 0;

#define DPT_PROF_START() do { if (do_prof) { cuStreamSynchronize(r->stream); \
    clock_gettime(CLOCK_MONOTONIC, &ts); pt0 = ts.tv_sec + ts.tv_nsec * 1e-9; } } while(0)
#define DPT_PROF_END(acc) do { if (do_prof) { cuStreamSynchronize(r->stream); \
    clock_gettime(CLOCK_MONOTONIC, &ts); pt1 = ts.tv_sec + ts.tv_nsec * 1e-9; acc += pt1 - pt0; } } while(0)

    /* Token processing + projection for each feature level (on GPU) */
    DPT_PROF_START();
    for (int fi = 0; fi < 4; fi++) {
        int oc_val = r->head_out_channels[fi];

        /* 1. Extract patch tokens + concatenate CLS -> d_dpt_cat [np, 2*dim] */
        kl_cls_concat(r, r->d_dpt_cat, r->d_features[fi], np, dim);

        /* 2. Head LayerNorm -> d_dpt_ln [np, 2*dim] */
        if (dw->norm_w) {
            kl_layernorm(r, r->d_dpt_ln, r->d_dpt_cat, dw->norm_w, dw->norm_b,
                          np, head_dim_in);
        } else {
            cuMemcpyDtoDAsync(r->d_dpt_ln, r->d_dpt_cat,
                               (size_t)np * head_dim_in * sizeof(float), r->stream);
        }

        /* 3. 1x1 projection (GEMM F16) -> d_dpt_proj [np, oc_val] */
        kl_gemm(r, r->d_dpt_proj, dw->proj_w[fi], r->d_dpt_ln, dw->proj_b[fi],
                 oc_val, head_dim_in, np);

        /* 4-5. Spatial alignment + reshape */
        if (fi == 0) {
            /* GEMM-based ConvTranspose2d 4x4 stride 4 (skip tok_to_chw) */
            /* d_dpt_proj[np,oc_val] is already token-major = [Hi*Wi, Ci] */
            kl_deconv_gemm_scatter(r, r->d_dpt_spatial[0],
                                    r->d_dpt_proj, dw->upsample_0_w,
                                    dw->upsample_0_b, r->d_dpt_ln,
                                    oc_val, oc_val, gh, gw, 4, 4, 4);
        } else if (fi == 1) {
            /* GEMM-based ConvTranspose2d 2x2 stride 2 (skip tok_to_chw) */
            kl_deconv_gemm_scatter(r, r->d_dpt_spatial[1],
                                    r->d_dpt_proj, dw->upsample_1_w,
                                    dw->upsample_1_b, r->d_dpt_ln,
                                    oc_val, oc_val, gh, gw, 2, 2, 2);
        } else if (fi == 2) {
            /* Identity: reshape to CHW */
            kl_tok_to_chw(r, r->d_dpt_chw, r->d_dpt_proj, oc_val, gh, gw);
            cuMemcpyDtoDAsync(r->d_dpt_spatial[2], r->d_dpt_chw,
                               (size_t)oc_val * gh * gw * sizeof(float), r->stream);
        } else {
            /* Conv2d 3x3 stride 2 pad 1 via MMA: needs CHW layout */
            kl_tok_to_chw(r, r->d_dpt_chw, r->d_dpt_proj, oc_val, gh, gw);
            kl_conv_gemm(r, r->d_dpt_spatial[3], r->d_dpt_chw,
                          dw->downsample_w, dw->downsample_b,
                          gh, gw, oc_val, oc_val, 3, 3, 2, 1);
        }

        /* 6. Adapter conv 3x3 -> d_dpt_adapted[fi] [feat, sp_h, sp_w] via MMA */
        CUdeviceptr null_bias = 0;
        kl_conv_gemm(r, r->d_dpt_adapted[fi], r->d_dpt_spatial[fi],
                      dw->adapter_w[fi], null_bias,
                      sp_h[fi], sp_w[fi], oc_val, feat, 3, 3, 1, 1);
    }
    DPT_PROF_END(dpt_tok);

    /* Bottom-up RefineNet fusion (on GPU) */
    DPT_PROF_START();
    /* Level 3 (deepest, no deeper input) */
    gpu_refinenet(r, 3, r->d_dpt_adapted[3], sp_h[3], sp_w[3],
                  0, 0, 0, feat, r->d_dpt_fused);
    int fh = sp_h[3], fw = sp_w[3];

    /* Level 2 */
    gpu_refinenet(r, 2, r->d_dpt_adapted[2], sp_h[2], sp_w[2],
                  r->d_dpt_fused, fh, fw, feat, r->d_dpt_fused);
    fh = sp_h[2]; fw = sp_w[2];

    /* Level 1 */
    gpu_refinenet(r, 1, r->d_dpt_adapted[1], sp_h[1], sp_w[1],
                  r->d_dpt_fused, fh, fw, feat, r->d_dpt_fused);
    fh = sp_h[1]; fw = sp_w[1];

    /* Level 0 */
    gpu_refinenet(r, 0, r->d_dpt_adapted[0], sp_h[0], sp_w[0],
                  r->d_dpt_fused, fh, fw, feat, r->d_dpt_fused);
    fh = sp_h[0]; fw = sp_w[0];
    DPT_PROF_END(dpt_refine);

    /* Output convolutions (on GPU) */
    DPT_PROF_START();
    int feat_half = feat / 2;
    if (feat_half < 1) feat_half = 1;
    int out_mid = dw->out_mid > 0 ? dw->out_mid : feat_half; /* typically 32 */

    /* neck: Conv2d(feat, feat/2, 3, pad=1) + ReLU via MMA */
    kl_conv_gemm(r, r->d_dpt_tmp, r->d_dpt_fused,
                  dw->neck_w, dw->neck_b,
                  fh, fw, feat, feat_half, 3, 3, 1, 1);
    kl_relu_inplace(r, r->d_dpt_tmp, feat_half * fh * fw);

    /* out_0: Conv2d(feat/2, out_mid, 3, pad=1) + ReLU via MMA */
    kl_conv_gemm(r, r->d_dpt_tmp2, r->d_dpt_tmp,
                  dw->out_0_w, dw->out_0_b,
                  fh, fw, feat_half, out_mid, 3, 3, 1, 1);
    kl_relu_inplace(r, r->d_dpt_tmp2, out_mid * fh * fw);

    /* out_2: Conv2d(out_mid, 2, 1) */
    kl_conv2d(r, r->d_dpt_out, r->d_dpt_tmp2,
               dw->out_2_w, dw->out_2_b,
               fh, fw, out_mid, 2, 1, 1, 1, 0);

    /* depth_activation: exp(depth), sigmoid(confidence) */
    {
        int hw = fh * fw;
        int grid = (hw + 255) / 256;
        void *args[] = {&r->d_dpt_out, &hw};
        cuLaunchKernel(r->fn_depth_activation, (unsigned)grid, 1, 1, 256, 1, 1,
                       0, r->stream, args, NULL);
    }
    DPT_PROF_END(dpt_output);

    /* Bilinear upsample to original resolution on GPU */
    DPT_PROF_START();
    {
        size_t result_sz = (size_t)2 * img_h * img_w * sizeof(float);
        if (result_sz > r->d_result_cap) {
            if (r->d_result) cuMemFree(r->d_result);
            cuMemAlloc(&r->d_result, result_sz);
            r->d_result_cap = result_sz;
        }
        CUdeviceptr d_result = r->d_result;

        kl_bilinear(r, d_result, r->d_dpt_out, 2, fh, fw, img_h, img_w);

        /* Synchronize before downloading */
        cuStreamSynchronize(r->stream);

        /* Download result to host */
        float *h_result = (float *)malloc(result_sz);
        cuMemcpyDtoH(h_result, d_result, result_sz);

        result.width = img_w;
        result.height = img_h;
        result.depth = (float *)malloc((size_t)img_w * img_h * sizeof(float));
        result.confidence = (float *)malloc((size_t)img_w * img_h * sizeof(float));
        memcpy(result.depth, h_result, (size_t)img_w * img_h * sizeof(float));
        memcpy(result.confidence, h_result + img_h * img_w,
               (size_t)img_w * img_h * sizeof(float));
        free(h_result);
    }
    DPT_PROF_END(dpt_upsample);

#undef DPT_PROF_START
#undef DPT_PROF_END

    clock_gettime(CLOCK_MONOTONIC, &ts);
    t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "cuda_da3: GPU DPT head: %.1f ms\n", (t1-t0)*1000);
    if (r->verbose >= 2) {
        CUresult err = cuCtxSynchronize();
        fprintf(stderr, "  CUDA after DPT head: %d\n", (int)err);
    }
    if (do_prof) {
        fprintf(stderr, "  DPT profile (sync'd):\n");
        fprintf(stderr, "    tok_process: %5.1f ms\n", dpt_tok*1000);
        fprintf(stderr, "    refinenet:   %5.1f ms\n", dpt_refine*1000);
        fprintf(stderr, "    output_conv: %5.1f ms\n", dpt_output*1000);
        fprintf(stderr, "    final_up:    %5.1f ms\n", dpt_upsample*1000);
    }

    /* ─── Aux DPT (Phase 2): rays + sky segmentation ─── */
    if (r->dpt_aux.loaded && (output_flags & DA3_OUTPUT_RAYS)) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double ax0 = ts.tv_sec + ts.tv_nsec * 1e-9;

        /* Run aux branch (reuses d_dpt_adapted from main DPT token processing) */
        run_aux_dpt(r, sp_h, sp_w, feat, r->d_aux_out);

        /* Average 4 levels (all at level 0 spatial size) */
        /* Only last level (index 3) is used — single output in d_aux_out[0] */
        int oh = sp_h[0], ow = sp_w[0];
        int aux_hw = oh * ow;

        /* Bilinear upsample 7 channels to original resolution */
        int npix = img_w * img_h;
        result.rays = (float *)malloc((size_t)6 * npix * sizeof(float));
        result.ray_confidence = (float *)malloc((size_t)npix * sizeof(float));

        CUdeviceptr d_aux_full = r->d_dpt_tmp;
        kl_bilinear(r, d_aux_full, r->d_aux_out[0], 7, oh, ow, img_h, img_w);
        cuStreamSynchronize(r->stream);

        float *h_aux_full = (float *)malloc((size_t)7 * npix * sizeof(float));
        cuMemcpyDtoH(h_aux_full, d_aux_full, (size_t)7 * npix * sizeof(float));

        /* Split: rays = channels 0-5 (linear activation = identity),
         *        ray_confidence = channel 6 (expp1 = exp(x) + 1) */
        memcpy(result.rays, h_aux_full, (size_t)6 * npix * sizeof(float));
        /* Apply expp1 activation to ray confidence: exp(x) + 1 */
        {
            const float *src = h_aux_full + 6 * npix;
            for (int i = 0; i < npix; i++)
                result.ray_confidence[i] = expf(src[i]) + 1.0f;
        }
        /* Sky segmentation: not available from DualDPT (only from metric DPT branch) */
        result.sky_seg = NULL;
        free(h_aux_full);

        result.has_rays = 1;

        clock_gettime(CLOCK_MONOTONIC, &ts);
        double ax1 = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_da3: Aux DPT (rays+sky): %.1f ms\n", (ax1-ax0)*1000);
    }

    /* ─── GSDPT (Phase 4): 3D Gaussian estimation ─── */
    if (r->gsdpt.loaded && (output_flags & DA3_OUTPUT_GAUSSIANS)) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double gs0 = ts.tv_sec + ts.tv_nsec * 1e-9;

        dpt_gpu_weights *gdw = &r->gsdpt.dpt;
        int gs_oc = r->gsdpt.gs_out_channels;
        if (gs_oc < 2) gs_oc = 38;

        /* Run images_merger: 3 stride-2 Conv2d on normalized image → d_gs_merged
         * Input: d_img_norm [3, target_h, target_w] */
        {
            CUdeviceptr cur = r->d_img_norm;
            CUdeviceptr bufs[2] = {r->d_dpt_tmp, r->d_dpt_tmp2};
            int mh = target_h, mw = target_w;
            for (int mi = 0; mi < 3; mi++) {
                if (!r->gsdpt.merger_w[mi]) break;
                int mci = r->gsdpt.merger_ci[mi];
                int mco = r->gsdpt.merger_co[mi];
                int moh = (mh + 2 * 1 - 3) / 2 + 1;
                int mow = (mw + 2 * 1 - 3) / 2 + 1;
                CUdeviceptr dst = (mi == 2) ? r->d_gs_merged : bufs[mi % 2];
                kl_conv2d(r, dst, cur, r->gsdpt.merger_w[mi], r->gsdpt.merger_b[mi],
                           mh, mw, mci, mco, 3, 3, 2, 1);
                /* Activation at indices 1,3 only (not after last conv at index 4) */
                if (mi < 2)
                    kl_silu_inplace(r, dst, mco * moh * mow);
                cur = dst;
                mh = moh; mw = mow;
            }
        }

        /* GSDPT token processing + refinenet (same pipeline as main DPT) */
        /* Reuses d_features from backbone as input */

        /* Debug: check backbone features and GSDPT proj weights */
        if (r->verbose >= 2) {
            CUresult err = cuStreamSynchronize(r->stream);
            fprintf(stderr, "  CUDA stream sync before GSDPT: %d\n", (int)err);
            err = cuCtxSynchronize();
            fprintf(stderr, "  CUDA ctx sync before GSDPT: %d\n", (int)err);
            float dbg[4];
            err = cuMemcpyDtoH(dbg, r->d_features[0], sizeof(dbg));
            fprintf(stderr, "  d_features[0] CLS[0..3]: %.6f %.6f %.6f %.6f (err=%d, ptr=0x%llx)\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], (int)err,
                    (unsigned long long)r->d_features[0]);
            /* Check patch token 0 (offset = 1 * dim) */
            err = cuMemcpyDtoH(dbg, r->d_features[0] + (size_t)dim * sizeof(float),
                          sizeof(dbg));
            fprintf(stderr, "  d_features[0] patch0[0..3]: %.6f %.6f %.6f %.6f (err=%d)\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], (int)err);
            /* Also check d_dpt_cat after CLS concat for fi=0 */
            /* Check GSDPT proj weight (download first 8 FP16 values as raw bytes) */
            uint16_t wdbg[8];
            cuMemcpyDtoH(wdbg, gdw->proj_w[0], sizeof(wdbg));
            float wf[4];
            for (int i = 0; i < 4; i++) {
                uint32_t bits = wdbg[i];
                uint32_t sign = (bits >> 15) & 1;
                uint32_t exp  = (bits >> 10) & 0x1F;
                uint32_t mant = bits & 0x3FF;
                if (exp == 0) wf[i] = sign ? -0.0f : 0.0f;
                else if (exp == 31) wf[i] = sign ? -1.0f/0.0f : 1.0f/0.0f;
                else { uint32_t f = (sign<<31)|((exp+112)<<23)|(mant<<13); memcpy(&wf[i],&f,4); }
            }
            fprintf(stderr, "  gsdpt proj_w[0] fp16[0..3]: %.6f %.6f %.6f %.6f\n",
                    wf[0], wf[1], wf[2], wf[3]);
        }

        for (int fi = 0; fi < 4; fi++) {
            int oc_val = r->head_out_channels[fi];

            kl_cls_concat(r, r->d_dpt_cat, r->d_features[fi], np, dim);
            if (gdw->norm_w)
                kl_layernorm(r, r->d_dpt_ln, r->d_dpt_cat, gdw->norm_w, gdw->norm_b,
                              np, head_dim_in);
            else
                cuMemcpyDtoDAsync(r->d_dpt_ln, r->d_dpt_cat,
                                   (size_t)np * head_dim_in * sizeof(float), r->stream);

            kl_gemm(r, r->d_dpt_proj, gdw->proj_w[fi], r->d_dpt_ln, gdw->proj_b[fi],
                     oc_val, head_dim_in, np);

            /* Debug: trace GSDPT fi=0 token processing */
            if (r->verbose >= 2 && fi == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[4];
                cuMemcpyDtoH(dbg, r->d_dpt_cat, sizeof(dbg));
                fprintf(stderr, "  gsdpt cat[0][0..3]: %.6f %.6f %.6f %.6f\n",
                        dbg[0], dbg[1], dbg[2], dbg[3]);
                cuMemcpyDtoH(dbg, r->d_dpt_ln, sizeof(dbg));
                fprintf(stderr, "  gsdpt ln[0][0..3]: %.6f %.6f %.6f %.6f\n",
                        dbg[0], dbg[1], dbg[2], dbg[3]);
                cuMemcpyDtoH(dbg, r->d_dpt_proj, sizeof(dbg));
                fprintf(stderr, "  gsdpt proj[0][0..3]: %.6e %.6e %.6e %.6e (oc=%d)\n",
                        dbg[0], dbg[1], dbg[2], dbg[3], oc_val);
                /* Check proj bias */
                cuMemcpyDtoH(dbg, gdw->proj_b[0], sizeof(dbg));
                fprintf(stderr, "  gsdpt proj_b[0][0..3]: %.6f %.6f %.6f %.6f\n",
                        dbg[0], dbg[1], dbg[2], dbg[3]);
            }

            if (fi == 0)
                kl_deconv_gemm_scatter(r, r->d_dpt_spatial[0], r->d_dpt_proj,
                                        gdw->upsample_0_w, gdw->upsample_0_b, r->d_dpt_ln,
                                        oc_val, oc_val, gh, gw, 4, 4, 4);
            else if (fi == 1)
                kl_deconv_gemm_scatter(r, r->d_dpt_spatial[1], r->d_dpt_proj,
                                        gdw->upsample_1_w, gdw->upsample_1_b, r->d_dpt_ln,
                                        oc_val, oc_val, gh, gw, 2, 2, 2);
            else if (fi == 2) {
                kl_tok_to_chw(r, r->d_dpt_chw, r->d_dpt_proj, oc_val, gh, gw);
                cuMemcpyDtoDAsync(r->d_dpt_spatial[2], r->d_dpt_chw,
                                   (size_t)oc_val * gh * gw * sizeof(float), r->stream);
            } else {
                kl_tok_to_chw(r, r->d_dpt_chw, r->d_dpt_proj, oc_val, gh, gw);
                kl_conv_gemm(r, r->d_dpt_spatial[3], r->d_dpt_chw,
                              gdw->downsample_w, gdw->downsample_b,
                              gh, gw, oc_val, oc_val, 3, 3, 2, 1);
            }

            /* Debug: trace spatial output */
            if (r->verbose >= 2 && fi == 0) {
                cuStreamSynchronize(r->stream);
                float dbg[4];
                cuMemcpyDtoH(dbg, r->d_dpt_spatial[0], sizeof(dbg));
                fprintf(stderr, "  gsdpt spatial[0][0..3]: %.4f %.4f %.4f %.4f\n",
                        dbg[0], dbg[1], dbg[2], dbg[3]);
            }

            CUdeviceptr null_bias = 0;
            kl_conv_gemm(r, r->d_dpt_adapted[fi], r->d_dpt_spatial[fi],
                          gdw->adapter_w[fi], null_bias,
                          sp_h[fi], sp_w[fi], oc_val, feat, 3, 3, 1, 1);
        }

        /* TODO: Inject merger features at level 0 (add to d_dpt_adapted[0]) */
        /* For now, skip merger injection — just run standard DPT pipeline */

        /* Bottom-up RefineNet fusion with GSDPT weights */
        gpu_refinenet_w(r, r->d_dpt_adapted[3], sp_h[3], sp_w[3],
                        0, 0, 0, feat, r->d_dpt_fused,
                        gdw->fuse_out_w[3], gdw->fuse_out_b[3],
                        gdw->fuse_rcu1_c1_w[3], gdw->fuse_rcu1_c1_b[3],
                        gdw->fuse_rcu1_c2_w[3], gdw->fuse_rcu1_c2_b[3],
                        gdw->fuse_rcu2_c1_w[3], gdw->fuse_rcu2_c1_b[3],
                        gdw->fuse_rcu2_c2_w[3], gdw->fuse_rcu2_c2_b[3],
                        gdw->has_rcu1[3], gdw->has_rcu2[3]);
        int gs_fh = sp_h[3], gs_fw = sp_w[3];
        gpu_refinenet_w(r, r->d_dpt_adapted[2], sp_h[2], sp_w[2],
                        r->d_dpt_fused, gs_fh, gs_fw, feat, r->d_dpt_fused,
                        gdw->fuse_out_w[2], gdw->fuse_out_b[2],
                        gdw->fuse_rcu1_c1_w[2], gdw->fuse_rcu1_c1_b[2],
                        gdw->fuse_rcu1_c2_w[2], gdw->fuse_rcu1_c2_b[2],
                        gdw->fuse_rcu2_c1_w[2], gdw->fuse_rcu2_c1_b[2],
                        gdw->fuse_rcu2_c2_w[2], gdw->fuse_rcu2_c2_b[2],
                        gdw->has_rcu1[2], gdw->has_rcu2[2]);
        gs_fh = sp_h[2]; gs_fw = sp_w[2];
        gpu_refinenet_w(r, r->d_dpt_adapted[1], sp_h[1], sp_w[1],
                        r->d_dpt_fused, gs_fh, gs_fw, feat, r->d_dpt_fused,
                        gdw->fuse_out_w[1], gdw->fuse_out_b[1],
                        gdw->fuse_rcu1_c1_w[1], gdw->fuse_rcu1_c1_b[1],
                        gdw->fuse_rcu1_c2_w[1], gdw->fuse_rcu1_c2_b[1],
                        gdw->fuse_rcu2_c1_w[1], gdw->fuse_rcu2_c1_b[1],
                        gdw->fuse_rcu2_c2_w[1], gdw->fuse_rcu2_c2_b[1],
                        gdw->has_rcu1[1], gdw->has_rcu2[1]);
        gs_fh = sp_h[1]; gs_fw = sp_w[1];
        gpu_refinenet_w(r, r->d_dpt_adapted[0], sp_h[0], sp_w[0],
                        r->d_dpt_fused, gs_fh, gs_fw, feat, r->d_dpt_fused,
                        gdw->fuse_out_w[0], gdw->fuse_out_b[0],
                        gdw->fuse_rcu1_c1_w[0], gdw->fuse_rcu1_c1_b[0],
                        gdw->fuse_rcu1_c2_w[0], gdw->fuse_rcu1_c2_b[0],
                        gdw->fuse_rcu2_c1_w[0], gdw->fuse_rcu2_c1_b[0],
                        gdw->fuse_rcu2_c2_w[0], gdw->fuse_rcu2_c2_b[0],
                        gdw->has_rcu1[0], gdw->has_rcu2[0]);
        gs_fh = sp_h[0]; gs_fw = sp_w[0];

        /* Debug: check GSDPT d_dpt_adapted[0] and d_dpt_fused after refinenet */
        if (r->verbose >= 2) {
            cuStreamSynchronize(r->stream);
            float dbg[4];
            cuMemcpyDtoH(dbg, r->d_dpt_adapted[0], sizeof(dbg));
            fprintf(stderr, "  gsdpt adapted[0][0..3]: %.4f %.4f %.4f %.4f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            cuMemcpyDtoH(dbg, r->d_dpt_fused, sizeof(dbg));
            fprintf(stderr, "  gsdpt fused[0..3]: %.4f %.4f %.4f %.4f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }

        /* Output convolutions → gs_oc channels
         * output_conv1 (neck): Conv2d(256, 128, 3) — NO ReLU (single .0 index)
         * + inject upsampled merger features (element-wise add)
         * output_conv2: Conv2d(128, 32, 3) + ReLU + Conv2d(32, gs_oc, 1) */
        int gs_feat_half = feat / 2;
        if (gs_feat_half < 1) gs_feat_half = 1;
        int gs_out_mid = gdw->out_mid > 0 ? gdw->out_mid : gs_feat_half;

        /* output_conv1 (neck): Conv2d(256, 128, 3, pad=1), no activation */
        kl_conv_gemm(r, r->d_dpt_tmp, r->d_dpt_fused,
                      gdw->neck_w, gdw->neck_b,
                      gs_fh, gs_fw, feat, gs_feat_half, 3, 3, 1, 1);

        /* Inject merger features: upsample d_gs_merged [128, mg_h, mg_w] → [128, gs_fh, gs_fw]
         * then add to neck output */
        if (r->d_gs_merged && r->gs_merger_h > 0) {
            kl_bilinear(r, r->d_dpt_tmp2, r->d_gs_merged,
                         gs_feat_half, r->gs_merger_h, r->gs_merger_w, gs_fh, gs_fw);
            kl_add_inplace(r, r->d_dpt_tmp, r->d_dpt_tmp2, gs_feat_half * gs_fh * gs_fw);
        }

        /* output_conv2: Conv2d(128, 32, 3, pad=1) + ReLU + Conv2d(32, gs_oc, 1) */
        kl_conv_gemm(r, r->d_dpt_tmp2, r->d_dpt_tmp,
                      gdw->out_0_w, gdw->out_0_b,
                      gs_fh, gs_fw, gs_feat_half, gs_out_mid, 3, 3, 1, 1);
        kl_relu_inplace(r, r->d_dpt_tmp2, gs_out_mid * gs_fh * gs_fw);
        kl_conv2d(r, r->d_gs_out, r->d_dpt_tmp2,
                   gdw->out_2_w, gdw->out_2_b,
                   gs_fh, gs_fw, gs_out_mid, gs_oc, 1, 1, 1, 0);

        /* Download gaussians */
        cuStreamSynchronize(r->stream);

        /* Debug: check output conv intermediate and final */
        if (r->verbose >= 2) {
            float dbg[4];
            cuMemcpyDtoH(dbg, r->d_dpt_tmp, sizeof(dbg));
            fprintf(stderr, "  gsdpt neck_out[0..3]: %.4f %.4f %.4f %.4f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            cuMemcpyDtoH(dbg, r->d_gs_out, sizeof(dbg));
            fprintf(stderr, "  gsdpt gs_out[0..3]: %.4f %.4f %.4f %.4f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }

        /* Bilinear upsample each channel to output resolution */
        int npix = img_w * img_h;
        size_t gs_full_sz = (size_t)gs_oc * npix * sizeof(float);
        CUdeviceptr d_gs_full;
        cuMemAlloc(&d_gs_full, gs_full_sz);
        kl_bilinear(r, d_gs_full, r->d_gs_out, gs_oc, gs_fh, gs_fw, img_h, img_w);
        cuStreamSynchronize(r->stream);
        result.gaussians = (float *)malloc(gs_full_sz);
        cuMemcpyDtoH(result.gaussians, d_gs_full, gs_full_sz);
        cuMemFree(d_gs_full);

        result.has_gaussians = 1;

        clock_gettime(CLOCK_MONOTONIC, &ts);
        double gs1 = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_da3: GSDPT (%d channels): %.1f ms\n",
                    gs_oc, (gs1-gs0)*1000);
    }

    return result;
}

/* ======================================================================== */
/* Public API: free                                                         */
/* ======================================================================== */

void cuda_da3_free(cuda_da3_runner *r) {
    if (!r) return;

    /* Free per-layer weights */
    if (r->layers) {
        for (int L = 0; L < r->n_blocks; L++) {
            cuda_da3_layer *ly = &r->layers[L];
            if (ly->ln1_w) cuMemFree(ly->ln1_w);
            if (ly->ln1_b) cuMemFree(ly->ln1_b);
            if (ly->attn_qkv_w) cuMemFree(ly->attn_qkv_w);
            if (ly->attn_qkv_b) cuMemFree(ly->attn_qkv_b);
            if (ly->attn_q_norm_w) cuMemFree(ly->attn_q_norm_w);
            if (ly->attn_q_norm_b) cuMemFree(ly->attn_q_norm_b);
            if (ly->attn_k_norm_w) cuMemFree(ly->attn_k_norm_w);
            if (ly->attn_k_norm_b) cuMemFree(ly->attn_k_norm_b);
            if (ly->attn_out_w) cuMemFree(ly->attn_out_w);
            if (ly->attn_out_b) cuMemFree(ly->attn_out_b);
            if (ly->ls1) cuMemFree(ly->ls1);
            if (ly->ls2) cuMemFree(ly->ls2);
            if (ly->ln2_w) cuMemFree(ly->ln2_w);
            if (ly->ln2_b) cuMemFree(ly->ln2_b);
            if (ly->ffn_gate_up_w) cuMemFree(ly->ffn_gate_up_w);
            if (ly->ffn_gate_up_b) cuMemFree(ly->ffn_gate_up_b);
            if (ly->ffn_up_w) cuMemFree(ly->ffn_up_w);
            if (ly->ffn_up_b) cuMemFree(ly->ffn_up_b);
            if (ly->ffn_down_w) cuMemFree(ly->ffn_down_w);
            if (ly->ffn_down_b) cuMemFree(ly->ffn_down_b);
        }
        free(r->layers);
    }

    /* Free embeddings */
    if (r->d_cls_token) cuMemFree(r->d_cls_token);
    if (r->d_pos_embed) cuMemFree(r->d_pos_embed);
    if (r->d_patch_embed_w) cuMemFree(r->d_patch_embed_w);
    if (r->d_patch_embed_b) cuMemFree(r->d_patch_embed_b);

    /* Free preprocessing and result buffers */
    if (r->d_img_norm) cuMemFree(r->d_img_norm);
    if (r->d_img_raw) cuMemFree(r->d_img_raw);
    if (r->d_result) cuMemFree(r->d_result);

    /* Free backbone scratch */
    if (r->d_hidden) cuMemFree(r->d_hidden);
    if (r->d_hidden2) cuMemFree(r->d_hidden2);
    if (r->d_ln_buf) cuMemFree(r->d_ln_buf);
    if (r->d_qkv) cuMemFree(r->d_qkv);
    if (r->d_attn_out) cuMemFree(r->d_attn_out);
    if (r->d_ffn_buf) cuMemFree(r->d_ffn_buf);
    if (r->d_ffn_mid) cuMemFree(r->d_ffn_mid);
    if (r->d_proj_out) cuMemFree(r->d_proj_out);
    if (r->d_pos_y) cuMemFree(r->d_pos_y);
    if (r->d_pos_x) cuMemFree(r->d_pos_x);
    for (int i = 0; i < 4; i++)
        if (r->d_features[i]) cuMemFree(r->d_features[i]);

    /* Free DPT head weights */
    {
        dpt_gpu_weights *dw = &r->dpt_w;
        if (dw->norm_w) cuMemFree(dw->norm_w);
        if (dw->norm_b) cuMemFree(dw->norm_b);
        for (int i = 0; i < 4; i++) {
            if (dw->proj_w[i]) cuMemFree(dw->proj_w[i]);
            if (dw->proj_b[i]) cuMemFree(dw->proj_b[i]);
            if (dw->adapter_w[i]) cuMemFree(dw->adapter_w[i]);
            if (dw->fuse_out_w[i]) cuMemFree(dw->fuse_out_w[i]);
            if (dw->fuse_out_b[i]) cuMemFree(dw->fuse_out_b[i]);
            if (dw->fuse_rcu1_c1_w[i]) cuMemFree(dw->fuse_rcu1_c1_w[i]);
            if (dw->fuse_rcu1_c1_b[i]) cuMemFree(dw->fuse_rcu1_c1_b[i]);
            if (dw->fuse_rcu1_c2_w[i]) cuMemFree(dw->fuse_rcu1_c2_w[i]);
            if (dw->fuse_rcu1_c2_b[i]) cuMemFree(dw->fuse_rcu1_c2_b[i]);
            if (dw->fuse_rcu2_c1_w[i]) cuMemFree(dw->fuse_rcu2_c1_w[i]);
            if (dw->fuse_rcu2_c1_b[i]) cuMemFree(dw->fuse_rcu2_c1_b[i]);
            if (dw->fuse_rcu2_c2_w[i]) cuMemFree(dw->fuse_rcu2_c2_w[i]);
            if (dw->fuse_rcu2_c2_b[i]) cuMemFree(dw->fuse_rcu2_c2_b[i]);
        }
        if (dw->upsample_0_w) cuMemFree(dw->upsample_0_w);
        if (dw->upsample_0_b) cuMemFree(dw->upsample_0_b);
        if (dw->upsample_1_w) cuMemFree(dw->upsample_1_w);
        if (dw->upsample_1_b) cuMemFree(dw->upsample_1_b);
        if (dw->downsample_w) cuMemFree(dw->downsample_w);
        if (dw->downsample_b) cuMemFree(dw->downsample_b);
        if (dw->neck_w) cuMemFree(dw->neck_w);
        if (dw->neck_b) cuMemFree(dw->neck_b);
        if (dw->out_0_w) cuMemFree(dw->out_0_w);
        if (dw->out_0_b) cuMemFree(dw->out_0_b);
        if (dw->out_2_w) cuMemFree(dw->out_2_w);
        if (dw->out_2_b) cuMemFree(dw->out_2_b);
    }

    /* Free DPT scratch buffers */
    if (r->d_dpt_cat) cuMemFree(r->d_dpt_cat);
    if (r->d_dpt_ln) cuMemFree(r->d_dpt_ln);
    if (r->d_dpt_proj) cuMemFree(r->d_dpt_proj);
    if (r->d_dpt_chw) cuMemFree(r->d_dpt_chw);
    for (int i = 0; i < 4; i++) {
        if (r->d_dpt_spatial[i]) cuMemFree(r->d_dpt_spatial[i]);
        if (r->d_dpt_adapted[i]) cuMemFree(r->d_dpt_adapted[i]);
    }
    if (r->d_dpt_fused) cuMemFree(r->d_dpt_fused);
    if (r->d_dpt_tmp) cuMemFree(r->d_dpt_tmp);
    if (r->d_dpt_tmp2) cuMemFree(r->d_dpt_tmp2);
    if (r->d_dpt_out) cuMemFree(r->d_dpt_out);

    /* Free CameraDec weights */
    if (r->cam_dec.loaded) {
        if (r->cam_dec.backbone_norm_w) cuMemFree(r->cam_dec.backbone_norm_w);
        if (r->cam_dec.backbone_norm_b) cuMemFree(r->cam_dec.backbone_norm_b);
        for (int i = 0; i < 2; i++) {
            if (r->cam_dec.mlp_w[i]) cuMemFree(r->cam_dec.mlp_w[i]);
            if (r->cam_dec.mlp_b[i]) cuMemFree(r->cam_dec.mlp_b[i]);
        }
        if (r->cam_dec.fc_t_w) cuMemFree(r->cam_dec.fc_t_w);
        if (r->cam_dec.fc_t_b) cuMemFree(r->cam_dec.fc_t_b);
        if (r->cam_dec.fc_qvec_w) cuMemFree(r->cam_dec.fc_qvec_w);
        if (r->cam_dec.fc_qvec_b) cuMemFree(r->cam_dec.fc_qvec_b);
        if (r->cam_dec.fc_fov_w) cuMemFree(r->cam_dec.fc_fov_w);
        if (r->cam_dec.fc_fov_b) cuMemFree(r->cam_dec.fc_fov_b);
    }

    /* Free CameraEnc weights */
    if (r->cam_enc.loaded) {
        if (r->cam_enc.fc1_w) cuMemFree(r->cam_enc.fc1_w);
        if (r->cam_enc.fc1_b) cuMemFree(r->cam_enc.fc1_b);
        if (r->cam_enc.fc2_w) cuMemFree(r->cam_enc.fc2_w);
        if (r->cam_enc.fc2_b) cuMemFree(r->cam_enc.fc2_b);
        if (r->cam_enc.trunk_norm_w) cuMemFree(r->cam_enc.trunk_norm_w);
        if (r->cam_enc.trunk_norm_b) cuMemFree(r->cam_enc.trunk_norm_b);
        if (r->cam_enc.token_norm_w) cuMemFree(r->cam_enc.token_norm_w);
        if (r->cam_enc.token_norm_b) cuMemFree(r->cam_enc.token_norm_b);
        if (r->cam_enc.trunk) {
            for (int L = 0; L < r->cam_enc.n_trunk_blocks; L++) {
                cuda_da3_layer *ly = &r->cam_enc.trunk[L];
                if (ly->ln1_w) cuMemFree(ly->ln1_w);
                if (ly->ln1_b) cuMemFree(ly->ln1_b);
                if (ly->attn_qkv_w) cuMemFree(ly->attn_qkv_w);
                if (ly->attn_qkv_b) cuMemFree(ly->attn_qkv_b);
                if (ly->attn_out_w) cuMemFree(ly->attn_out_w);
                if (ly->attn_out_b) cuMemFree(ly->attn_out_b);
                if (ly->ls1) cuMemFree(ly->ls1);
                if (ly->ls2) cuMemFree(ly->ls2);
                if (ly->ln2_w) cuMemFree(ly->ln2_w);
                if (ly->ln2_b) cuMemFree(ly->ln2_b);
                if (ly->ffn_up_w) cuMemFree(ly->ffn_up_w);
                if (ly->ffn_up_b) cuMemFree(ly->ffn_up_b);
                if (ly->ffn_down_w) cuMemFree(ly->ffn_down_w);
                if (ly->ffn_down_b) cuMemFree(ly->ffn_down_b);
            }
            free(r->cam_enc.trunk);
        }
    }

    /* Free DPT Aux weights */
    if (r->dpt_aux.loaded) {
        for (int i = 0; i < 4; i++) {
            if (r->dpt_aux.fuse_out_w[i]) cuMemFree(r->dpt_aux.fuse_out_w[i]);
            if (r->dpt_aux.fuse_out_b[i]) cuMemFree(r->dpt_aux.fuse_out_b[i]);
            if (r->dpt_aux.fuse_rcu1_c1_w[i]) cuMemFree(r->dpt_aux.fuse_rcu1_c1_w[i]);
            if (r->dpt_aux.fuse_rcu1_c1_b[i]) cuMemFree(r->dpt_aux.fuse_rcu1_c1_b[i]);
            if (r->dpt_aux.fuse_rcu1_c2_w[i]) cuMemFree(r->dpt_aux.fuse_rcu1_c2_w[i]);
            if (r->dpt_aux.fuse_rcu1_c2_b[i]) cuMemFree(r->dpt_aux.fuse_rcu1_c2_b[i]);
            if (r->dpt_aux.fuse_rcu2_c1_w[i]) cuMemFree(r->dpt_aux.fuse_rcu2_c1_w[i]);
            if (r->dpt_aux.fuse_rcu2_c1_b[i]) cuMemFree(r->dpt_aux.fuse_rcu2_c1_b[i]);
            if (r->dpt_aux.fuse_rcu2_c2_w[i]) cuMemFree(r->dpt_aux.fuse_rcu2_c2_w[i]);
            if (r->dpt_aux.fuse_rcu2_c2_b[i]) cuMemFree(r->dpt_aux.fuse_rcu2_c2_b[i]);
            for (int j = 0; j < r->dpt_aux.oc1_count[i]; j++) {
                if (r->dpt_aux.oc1_w[i][j]) cuMemFree(r->dpt_aux.oc1_w[i][j]);
                if (r->dpt_aux.oc1_b[i][j]) cuMemFree(r->dpt_aux.oc1_b[i][j]);
            }
            if (r->dpt_aux.oc2_conv_w[i]) cuMemFree(r->dpt_aux.oc2_conv_w[i]);
            if (r->dpt_aux.oc2_conv_b[i]) cuMemFree(r->dpt_aux.oc2_conv_b[i]);
            if (r->dpt_aux.oc2_gn_w[i]) cuMemFree(r->dpt_aux.oc2_gn_w[i]);
            if (r->dpt_aux.oc2_gn_b[i]) cuMemFree(r->dpt_aux.oc2_gn_b[i]);
            if (r->dpt_aux.oc2_out_w[i]) cuMemFree(r->dpt_aux.oc2_out_w[i]);
            if (r->dpt_aux.oc2_out_b[i]) cuMemFree(r->dpt_aux.oc2_out_b[i]);
        }
    }

    /* Free DPT Aux scratch */
    for (int i = 0; i < 4; i++)
        if (r->d_aux_out[i]) cuMemFree(r->d_aux_out[i]);
    if (r->d_aux_scratch) cuMemFree(r->d_aux_scratch);

    /* Free GSDPT weights */
    if (r->gsdpt.loaded) {
        dpt_gpu_weights *gdw = &r->gsdpt.dpt;
        if (gdw->norm_w) cuMemFree(gdw->norm_w);
        if (gdw->norm_b) cuMemFree(gdw->norm_b);
        for (int i = 0; i < 4; i++) {
            if (gdw->proj_w[i]) cuMemFree(gdw->proj_w[i]);
            if (gdw->proj_b[i]) cuMemFree(gdw->proj_b[i]);
            if (gdw->adapter_w[i]) cuMemFree(gdw->adapter_w[i]);
            if (gdw->fuse_out_w[i]) cuMemFree(gdw->fuse_out_w[i]);
            if (gdw->fuse_out_b[i]) cuMemFree(gdw->fuse_out_b[i]);
            if (gdw->fuse_rcu1_c1_w[i]) cuMemFree(gdw->fuse_rcu1_c1_w[i]);
            if (gdw->fuse_rcu1_c1_b[i]) cuMemFree(gdw->fuse_rcu1_c1_b[i]);
            if (gdw->fuse_rcu1_c2_w[i]) cuMemFree(gdw->fuse_rcu1_c2_w[i]);
            if (gdw->fuse_rcu1_c2_b[i]) cuMemFree(gdw->fuse_rcu1_c2_b[i]);
            if (gdw->fuse_rcu2_c1_w[i]) cuMemFree(gdw->fuse_rcu2_c1_w[i]);
            if (gdw->fuse_rcu2_c1_b[i]) cuMemFree(gdw->fuse_rcu2_c1_b[i]);
            if (gdw->fuse_rcu2_c2_w[i]) cuMemFree(gdw->fuse_rcu2_c2_w[i]);
            if (gdw->fuse_rcu2_c2_b[i]) cuMemFree(gdw->fuse_rcu2_c2_b[i]);
        }
        if (gdw->upsample_0_w) cuMemFree(gdw->upsample_0_w);
        if (gdw->upsample_0_b) cuMemFree(gdw->upsample_0_b);
        if (gdw->upsample_1_w) cuMemFree(gdw->upsample_1_w);
        if (gdw->upsample_1_b) cuMemFree(gdw->upsample_1_b);
        if (gdw->downsample_w) cuMemFree(gdw->downsample_w);
        if (gdw->downsample_b) cuMemFree(gdw->downsample_b);
        if (gdw->neck_w) cuMemFree(gdw->neck_w);
        if (gdw->neck_b) cuMemFree(gdw->neck_b);
        if (gdw->out_0_w) cuMemFree(gdw->out_0_w);
        if (gdw->out_0_b) cuMemFree(gdw->out_0_b);
        if (gdw->out_2_w) cuMemFree(gdw->out_2_w);
        if (gdw->out_2_b) cuMemFree(gdw->out_2_b);
        for (int i = 0; i < 3; i++) {
            if (r->gsdpt.merger_w[i]) cuMemFree(r->gsdpt.merger_w[i]);
            if (r->gsdpt.merger_b[i]) cuMemFree(r->gsdpt.merger_b[i]);
        }
    }

    /* Free GSDPT scratch */
    if (r->d_gs_merged) cuMemFree(r->d_gs_merged);
    if (r->d_gs_out) cuMemFree(r->d_gs_out);

    /* Free nested metric model */
    if (r->metric.loaded) {
        if (r->metric.d_patch_embed_w) cuMemFree(r->metric.d_patch_embed_w);
        if (r->metric.d_patch_embed_b) cuMemFree(r->metric.d_patch_embed_b);
        if (r->metric.d_cls_token) cuMemFree(r->metric.d_cls_token);
        if (r->metric.d_pos_embed) cuMemFree(r->metric.d_pos_embed);
        if (r->metric.layers) {
            for (int L = 0; L < r->metric.n_blocks; L++) {
                cuda_da3_layer *ly = &r->metric.layers[L];
                if (ly->ln1_w) cuMemFree(ly->ln1_w);
                if (ly->ln1_b) cuMemFree(ly->ln1_b);
                if (ly->attn_qkv_w) cuMemFree(ly->attn_qkv_w);
                if (ly->attn_qkv_b) cuMemFree(ly->attn_qkv_b);
                if (ly->attn_q_norm_w) cuMemFree(ly->attn_q_norm_w);
                if (ly->attn_q_norm_b) cuMemFree(ly->attn_q_norm_b);
                if (ly->attn_k_norm_w) cuMemFree(ly->attn_k_norm_w);
                if (ly->attn_k_norm_b) cuMemFree(ly->attn_k_norm_b);
                if (ly->attn_out_w) cuMemFree(ly->attn_out_w);
                if (ly->attn_out_b) cuMemFree(ly->attn_out_b);
                if (ly->ls1) cuMemFree(ly->ls1);
                if (ly->ls2) cuMemFree(ly->ls2);
                if (ly->ln2_w) cuMemFree(ly->ln2_w);
                if (ly->ln2_b) cuMemFree(ly->ln2_b);
                if (ly->ffn_gate_up_w) cuMemFree(ly->ffn_gate_up_w);
                if (ly->ffn_gate_up_b) cuMemFree(ly->ffn_gate_up_b);
                if (ly->ffn_up_w) cuMemFree(ly->ffn_up_w);
                if (ly->ffn_up_b) cuMemFree(ly->ffn_up_b);
                if (ly->ffn_down_w) cuMemFree(ly->ffn_down_w);
                if (ly->ffn_down_b) cuMemFree(ly->ffn_down_b);
            }
            free(r->metric.layers);
        }
        dpt_gpu_weights *mdw = &r->metric.dpt_w;
        if (mdw->norm_w) cuMemFree(mdw->norm_w);
        if (mdw->norm_b) cuMemFree(mdw->norm_b);
        for (int i = 0; i < 4; i++) {
            if (mdw->proj_w[i]) cuMemFree(mdw->proj_w[i]);
            if (mdw->proj_b[i]) cuMemFree(mdw->proj_b[i]);
            if (mdw->adapter_w[i]) cuMemFree(mdw->adapter_w[i]);
            if (mdw->fuse_out_w[i]) cuMemFree(mdw->fuse_out_w[i]);
            if (mdw->fuse_out_b[i]) cuMemFree(mdw->fuse_out_b[i]);
            if (mdw->fuse_rcu1_c1_w[i]) cuMemFree(mdw->fuse_rcu1_c1_w[i]);
            if (mdw->fuse_rcu1_c1_b[i]) cuMemFree(mdw->fuse_rcu1_c1_b[i]);
            if (mdw->fuse_rcu1_c2_w[i]) cuMemFree(mdw->fuse_rcu1_c2_w[i]);
            if (mdw->fuse_rcu1_c2_b[i]) cuMemFree(mdw->fuse_rcu1_c2_b[i]);
            if (mdw->fuse_rcu2_c1_w[i]) cuMemFree(mdw->fuse_rcu2_c1_w[i]);
            if (mdw->fuse_rcu2_c1_b[i]) cuMemFree(mdw->fuse_rcu2_c1_b[i]);
            if (mdw->fuse_rcu2_c2_w[i]) cuMemFree(mdw->fuse_rcu2_c2_w[i]);
            if (mdw->fuse_rcu2_c2_b[i]) cuMemFree(mdw->fuse_rcu2_c2_b[i]);
            if (r->metric.d_features[i]) cuMemFree(r->metric.d_features[i]);
        }
        if (mdw->upsample_0_w) cuMemFree(mdw->upsample_0_w);
        if (mdw->upsample_0_b) cuMemFree(mdw->upsample_0_b);
        if (mdw->upsample_1_w) cuMemFree(mdw->upsample_1_w);
        if (mdw->upsample_1_b) cuMemFree(mdw->upsample_1_b);
        if (mdw->downsample_w) cuMemFree(mdw->downsample_w);
        if (mdw->downsample_b) cuMemFree(mdw->downsample_b);
        if (mdw->neck_w) cuMemFree(mdw->neck_w);
        if (mdw->neck_b) cuMemFree(mdw->neck_b);
        if (mdw->out_0_w) cuMemFree(mdw->out_0_w);
        if (mdw->out_0_b) cuMemFree(mdw->out_0_b);
        if (mdw->out_2_w) cuMemFree(mdw->out_2_w);
        if (mdw->out_2_b) cuMemFree(mdw->out_2_b);
    }

    free(r->h_output);

    /* Free CPU model */
    if (r->cpu_model) da3_free(r->cpu_model);

    if (r->module) cuModuleUnload(r->module);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->context) cuCtxDestroy(r->context);
    free(r);
}
