/*
 * hip_hy3d_kernels.h - HIP kernel source strings for Hunyuan3D-2.1
 *
 * Contains all HY3D-specific HIP kernels as a C string literal for
 * HIPRTC runtime compilation. Concatenated after hip_kernels_common_src
 * (which opens extern "C" {) and closes it at the end.
 *
 * Port of cuda_hy3d_kernels.h for AMD ROCm/HIP (RDNA4, gfx1201).
 * All 15 kernels are pure CUDA C with no PTX ASM, so they work
 * unchanged under HIP/HIPRTC.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef HIP_HY3D_KERNELS_H
#define HIP_HY3D_KERNELS_H

static const char hip_hy3d_specific_kernels[] =
"\n"

/* -------------------------------------------------------------------- */
/* gelu_exact_f32: exact GELU using erf (matches nn.GELU default).      */
/* nn.GELU() uses approximate='none': 0.5*x*(1+erf(x/sqrt(2))).         */
/* common/gelu_f32 uses tanh approximation — Hy3D DiT needs exact.      */
/* -------------------------------------------------------------------- */
"__global__ void gelu_exact_f32(float *x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        float v = x[i];\n"
"        x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));\n"
"    }\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* rms_norm_f32: per-head RMSNorm (for DiT QK normalization)            */
/* One thread per (token, head). Normalizes head_dim elements in-place. */
/* -------------------------------------------------------------------- */
"/* ---- rms_norm_f32: per-head RMSNorm (for QK normalization) ---- */\n"
"/* One thread per (token, head). Normalizes head_dim elements in-place. */\n"
"__global__ void rms_norm_f32(float *data, const float *w,\n"
"                              int n_tok, int n_heads, int head_dim,\n"
"                              int stride, float eps) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * n_heads;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / n_heads;\n"
"    int h = idx % n_heads;\n"
"    float *base = data + tok * stride + h * head_dim;\n"
"    float sum_sq = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) sum_sq += base[i] * base[i];\n"
"    float inv = rsqrtf(sum_sq / (float)head_dim + eps);\n"
"    for (int i = 0; i < head_dim; i++)\n"
"        base[i] = base[i] * inv * w[i];\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* qk_layernorm_f32: per-head LayerNorm (for ShapeVAE QK norm)          */
/* -------------------------------------------------------------------- */
"/* ---- qk_layernorm_f32: per-head LayerNorm (for ShapeVAE QK norm) ---- */\n"
"__global__ void qk_layernorm_f32(float *data, const float *w, const float *b,\n"
"                                   int n_tok, int n_heads, int head_dim,\n"
"                                   int stride, float eps) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * n_heads;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / n_heads;\n"
"    int h = idx % n_heads;\n"
"    float *base = data + tok * stride + h * head_dim;\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) sum += base[i];\n"
"    float mean = sum / (float)head_dim;\n"
"    float var_sum = 0.0f;\n"
"    for (int i = 0; i < head_dim; i++) { float d = base[i] - mean; var_sum += d*d; }\n"
"    float inv = rsqrtf(var_sum / (float)head_dim + eps);\n"
"    for (int i = 0; i < head_dim; i++)\n"
"        base[i] = (base[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* layerscale_add_f32: LayerScale residual for DINOv2                    */
/* dst[i] += src[i] * scale[i % dim]                                    */
/* -------------------------------------------------------------------- */
"/* ---- layerscale_add_f32: LayerScale residual for DINOv2 ---- */\n"
"/* dst[i] += src[i] * scale[i % dim] */\n"
"__global__ void layerscale_add_f32(float *dst, const float *src,\n"
"                                     const float *scale, int n, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    dst[i] += src[i] * scale[i % dim];\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* cross_attn_f32: cross-attention with different Q/KV lengths           */
/* One block per (head, query_token). Handles Q_len != KV_len.           */
/* -------------------------------------------------------------------- */
"/* ---- cross_attn_f32: cross-attention with different Q/KV lengths ---- */\n"
"/* One block per (head, query_token). Handles Q_len != KV_len.          */\n"
"__global__ void cross_attn_f32(float *out,\n"
"                                const float *Q, const float *K, const float *V,\n"
"                                int q_len, int kv_len, int dim,\n"
"                                int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qi = blockIdx.y;\n"
"    if (h >= n_heads || qi >= q_len) return;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    const float *q = Q + qi * dim + h * head_dim;\n"
"    extern __shared__ float sdata[];\n"
"    float *scores = sdata;\n"
"    float *rbuf = sdata + kv_len;\n"
"\n"
"    /* Phase 1: Q @ K^T */\n"
"    for (int ki = tid; ki < kv_len; ki += nt) {\n"
"        const float *k = K + ki * dim + h * head_dim;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) dot += q[d] * k[d];\n"
"        scores[ki] = dot * scale;\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 2: Softmax -- find max */\n"
"    float lmax = -1e30f;\n"
"    for (int ki = tid; ki < kv_len; ki += nt)\n"
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
"    for (int ki = tid; ki < kv_len; ki += nt) {\n"
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
"    for (int ki = tid; ki < kv_len; ki += nt)\n"
"        scores[ki] *= inv_sum;\n"
"    __syncthreads();\n"
"\n"
"    /* Phase 3: Weighted sum of V */\n"
"    for (int d = tid; d < head_dim; d += nt) {\n"
"        float acc = 0.0f;\n"
"        for (int ki = 0; ki < kv_len; ki++)\n"
"            acc += scores[ki] * V[ki * dim + h * head_dim + d];\n"
"        out[qi * dim + h * head_dim + d] = acc;\n"
"    }\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* fourier_embed_3d_f32: 3D coordinate Fourier embedding                 */
/* input: [N, 3], output: [N, out_dim] where out_dim = 3*(2*nf+1)       */
/* Layout: [x,y,z, sin(f0*x),...sin(fn*x), sin(f0*y),..., cos(...)...]   */
/* -------------------------------------------------------------------- */
"/* ---- fourier_embed_3d_f32: 3D coordinate Fourier embedding ---- */\n"
"/* input: [N, 3], output: [N, out_dim] where out_dim = 3*(2*nf+1)  */\n"
"/* Layout: [x,y,z, sin(f0*x),...sin(fn*x), sin(f0*y),..., cos(...)...] */\n"
"__global__ void fourier_embed_3d_f32(float *out, const float *coords,\n"
"                                      const float *freqs, int N,\n"
"                                      int num_freqs, int out_dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx >= N) return;\n"
"    const float *in = coords + idx * 3;\n"
"    float *o = out + idx * out_dim;\n"
"    int p = 0;\n"
"    /* Include raw input */\n"
"    o[p++] = in[0]; o[p++] = in[1]; o[p++] = in[2];\n"
"    /* Sin embeddings */\n"
"    for (int d = 0; d < 3; d++)\n"
"        for (int f = 0; f < num_freqs; f++)\n"
"            o[p++] = sinf(in[d] * freqs[f]);\n"
"    /* Cos embeddings */\n"
"    for (int d = 0; d < 3; d++)\n"
"        for (int f = 0; f < num_freqs; f++)\n"
"            o[p++] = cosf(in[d] * freqs[f]);\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* timestep_embed_f32: sinusoidal timestep embedding                     */
/* -------------------------------------------------------------------- */
"/* ---- timestep_embed_f32: sinusoidal timestep embedding ---- */\n"
"/* Matches PyTorch Timesteps class: no 1000x scaling. */\n"
"__global__ void timestep_embed_f32(float *out, float t, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int half = dim / 2;\n"
"    if (i >= half) return;\n"
"    float exponent = -logf(10000.0f) * (float)i / (float)half;\n"
"    float emb = expf(exponent) * t;\n"
"    out[i] = sinf(emb);\n"
"    out[half + i] = cosf(emb);\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* euler_step_f32: x_new = x - dt * v                                    */
/* -------------------------------------------------------------------- */
"/* ---- euler_step_f32: x_new = x - dt * v ---- */\n"
"__global__ void euler_step_f32(float *x, const float *v, float dt, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    x[i] = x[i] - dt * v[i];\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* cfg_combine_f32: out = uncond + scale * (cond - uncond)               */
/* -------------------------------------------------------------------- */
"/* ---- cfg_combine_f32: out = uncond + scale * (cond - uncond) ---- */\n"
"__global__ void cfg_combine_f32(float *out, const float *cond,\n"
"                                  const float *uncond,\n"
"                                  float scale, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    out[i] = uncond[i] + scale * (cond[i] - uncond[i]);\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* split_qkv_interleaved_f32: split [N, 3*W] with interleaved heads      */
/* Input:  [N, H, 3, HD] (3*W = H*3*HD)                                 */
/* Output: Q[N, W], K[N, W], V[N, W] (W = H*HD)                         */
/* -------------------------------------------------------------------- */
"/* ---- split_qkv_interleaved_f32: split [N, 3*W] interleaved heads ---- */\n"
"/* Input:  [N, H, 3, HD] (3*W = H*3*HD) */\n"
"/* Output: Q[N, W], K[N, W], V[N, W] (W = H*HD) */\n"
"__global__ void split_qkv_interleaved_f32(\n"
"    float *Q, float *K, float *V,\n"
"    const float *qkv, int N, int H, int HD) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int W = H * HD;\n"
"    int total = N * W;\n"
"    if (idx >= total) return;\n"
"    int n = idx / W;\n"
"    int rem = idx % W;\n"
"    int h = rem / HD;\n"
"    int d = rem % HD;\n"
"    int base = n * 3 * W + h * 3 * HD;\n"
"    Q[idx] = qkv[base + d];\n"
"    K[idx] = qkv[base + HD + d];\n"
"    V[idx] = qkv[base + 2*HD + d];\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* split_kv_interleaved_f32: split [M, 2*W] with interleaved heads       */
/* Input:  [M, H, 2, HD] (2*W = H*2*HD)                                 */
/* Output: K[M, W], V[M, W] (W = H*HD)                                  */
/* -------------------------------------------------------------------- */
"/* ---- split_kv_interleaved_f32: split [M, 2*W] interleaved heads ---- */\n"
"/* Input:  [M, H, 2, HD] (2*W = H*2*HD) */\n"
"/* Output: K[M, W], V[M, W] (W = H*HD) */\n"
"__global__ void split_kv_interleaved_f32(\n"
"    float *K, float *V,\n"
"    const float *kv, int M, int H, int HD) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int W = H * HD;\n"
"    int total = M * W;\n"
"    if (idx >= total) return;\n"
"    int m = idx / W;\n"
"    int rem = idx % W;\n"
"    int h = rem / HD;\n"
"    int d = rem % HD;\n"
"    int base = m * 2 * W + h * 2 * HD;\n"
"    K[idx] = kv[base + d];\n"
"    V[idx] = kv[base + HD + d];\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* broadcast_add_f32: dst[i] += src[i % dim]                             */
/* Broadcasts a [dim] bias vector across an [N, dim] matrix.             */
/* -------------------------------------------------------------------- */
"/* ---- broadcast_add_f32: dst[i] += src[i % dim] ---- */\n"
"/* Broadcasts a [dim] vector across [N, dim] matrix. */\n"
"__global__ void broadcast_add_f32(float *dst, const float *src,\n"
"                                    int n, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    dst[i] += src[i % dim];\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* concat_token_f32: prepend a token to a sequence                       */
/* Given token [dim] and seq [N, dim], output [(N+1), dim]               */
/* First row is token, remaining rows are seq.                           */
/* -------------------------------------------------------------------- */
"/* ---- concat_token_f32: prepend a token to a sequence ---- */\n"
"/* token [dim] + seq [N, dim] -> out [(N+1), dim], first row is token */\n"
"__global__ void concat_token_f32(float *out, const float *token,\n"
"                                   const float *seq,\n"
"                                   int N, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = (N + 1) * dim;\n"
"    if (i >= total) return;\n"
"    if (i < dim) {\n"
"        out[i] = token[i];\n"
"    } else {\n"
"        out[i] = seq[i - dim];\n"
"    }\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* moe_gate_f32: MoE gating with top-k selection                         */
/* -------------------------------------------------------------------- */
"/* ---- moe_gate_f32: MoE gating with top-k selection ---- */\n"
"/* logits [n_experts], outputs top_k indices + softmax weights. */\n"
"/* Single-threaded: n_experts is small (4-8). */\n"
"__global__ void moe_gate_f32(int *indices, float *weights,\n"
"                               const float *logits,\n"
"                               int n_experts, int top_k) {\n"
"    if (threadIdx.x != 0 || blockIdx.x != 0) return;\n"
"\n"
"    /* Find top-k indices by repeated argmax with masking */\n"
"    float used[32]; /* assume n_experts <= 32 */\n"
"    for (int e = 0; e < n_experts && e < 32; e++)\n"
"        used[e] = logits[e];\n"
"\n"
"    for (int k = 0; k < top_k; k++) {\n"
"        int best = -1;\n"
"        float best_val = -1e30f;\n"
"        for (int e = 0; e < n_experts; e++) {\n"
"            if (used[e] > best_val) {\n"
"                best_val = used[e];\n"
"                best = e;\n"
"            }\n"
"        }\n"
"        indices[k] = best;\n"
"        weights[k] = best_val;\n"
"        if (best >= 0) used[best] = -1e30f; /* mask out selected */\n"
"    }\n"
"\n"
"    /* Softmax over the selected top-k weights */\n"
"    float mx = -1e30f;\n"
"    for (int k = 0; k < top_k; k++)\n"
"        mx = fmaxf(mx, weights[k]);\n"
"    float sum = 0.0f;\n"
"    for (int k = 0; k < top_k; k++) {\n"
"        weights[k] = expf(weights[k] - mx);\n"
"        sum += weights[k];\n"
"    }\n"
"    float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;\n"
"    for (int k = 0; k < top_k; k++)\n"
"        weights[k] *= inv;\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* concat_f32: concatenate two [N, dim] tensors along last dimension     */
/* Given a [N, dim] and b [N, dim], output c [N, 2*dim]                  */
/* -------------------------------------------------------------------- */
"/* ---- concat_f32: concat two [N, dim] tensors along last dim ---- */\n"
"/* a [N, dim] + b [N, dim] -> c [N, 2*dim] */\n"
"__global__ void concat_f32(float *c, const float *a, const float *b,\n"
"                             int N, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = N * 2 * dim;\n"
"    if (i >= total) return;\n"
"    int row = i / (2 * dim);\n"
"    int col = i % (2 * dim);\n"
"    if (col < dim) {\n"
"        c[i] = a[row * dim + col];\n"
"    } else {\n"
"        c[i] = b[row * dim + (col - dim)];\n"
"    }\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* strip_first_token_f32: drop first token from sequence                  */
/* Copy src[1:, :] to dst. [N+1, dim] -> [N, dim]                        */
/* -------------------------------------------------------------------- */
"/* ---- strip_first_token_f32: drop first token ---- */\n"
"/* src [N+1, dim] -> dst [N, dim], skipping first row */\n"
"__global__ void strip_first_token_f32(float *dst, const float *src,\n"
"                                        int N, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = N * dim;\n"
"    if (i >= total) return;\n"
"    dst[i] = src[i + dim];\n"
"}\n"
"\n"

/* -------------------------------------------------------------------- */
/* gemm_f32_f32: pure F32 tiled GEMM (no F16 conversion)                 */
/* Same interface as gemm_tiled_f16_f32 but weights are float, not half.  */
/* Grid: (ceil(n_out/64), ceil(n_tok/16)), blockDim=(16,16)               */
/* -------------------------------------------------------------------- */
"/* ---- gemm_f32_f32: pure F32 tiled GEMM ---- */\n"
"/* Grid: (ceil(n_out/64), ceil(n_tok/16)), blockDim=(16,16) */\n"
"__global__ void gemm_f32_f32(float *Y, const float *W, const float *X,\n"
"                              const float *bias,\n"
"                              int n_out, int n_in, int n_tok) {\n"
"    __shared__ float smA[16][16];\n"
"    __shared__ float smB[16][16];\n"
"    int tx = threadIdx.x, ty = threadIdx.y;\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int out_base = blockIdx.x * 64;\n"
"    int row = tok_base + ty;\n"
"    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        smA[ty][tx] = (tok_base+ty < n_tok && k+tx < n_in)\n"
"                      ? X[(tok_base+ty)*n_in + k+tx] : 0.f;\n"
"        __syncthreads();\n"
"        { int w_out = out_base + tx;\n"
"          smB[ty][tx] = (w_out < n_out && k+ty < n_in)\n"
"                        ? W[(size_t)w_out*n_in + k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc0 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w_out = out_base+16+tx;\n"
"          smB[ty][tx] = (w_out < n_out && k+ty < n_in)\n"
"                        ? W[(size_t)w_out*n_in + k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc1 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w_out = out_base+32+tx;\n"
"          smB[ty][tx] = (w_out < n_out && k+ty < n_in)\n"
"                        ? W[(size_t)w_out*n_in + k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc2 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"        { int w_out = out_base+48+tx;\n"
"          smB[ty][tx] = (w_out < n_out && k+ty < n_in)\n"
"                        ? W[(size_t)w_out*n_in + k+ty] : 0.f; }\n"
"        __syncthreads();\n"
"        for (int i = 0; i < 16; i++) acc3 += smA[ty][i] * smB[i][tx];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < n_tok) {\n"
"        if (out_base+   tx < n_out) Y[row*n_out+out_base+   tx] = acc0 + (bias?bias[out_base+   tx]:0.f);\n"
"        if (out_base+16+tx < n_out) Y[row*n_out+out_base+16+tx] = acc1 + (bias?bias[out_base+16+tx]:0.f);\n"
"        if (out_base+32+tx < n_out) Y[row*n_out+out_base+32+tx] = acc2 + (bias?bias[out_base+32+tx]:0.f);\n"
"        if (out_base+48+tx < n_out) Y[row*n_out+out_base+48+tx] = acc3 + (bias?bias[out_base+48+tx]:0.f);\n"
"    }\n"
"}\n"
"\n"

/* ---- gemm_f16w_bf16a_wmma_t: F16-weight × F32-act GEMM via BF16 WMMA (gfx12) */
/* Layout matches gemm_tiled_f16_f32: Y[n_tok, n_out] = X[n_tok, n_in] * W[n_out, n_in]^T + bias.
 * W is F16 (half_raw); X/Y/bias are F32. Weights and activations are converted to
 * BF16 at SMEM load (F16 -> F32 via __half2float -> top-16-bits as BF16).
 * CTA tile 128x128, 256 threads = 8 waves in 2x4 (M x N). Each wave does 4x2 = 8
 * WMMA ops. Requires n_in % 16 == 0; n_out and n_tok are bounds-checked. */
"#if defined(__gfx1200__) || defined(__gfx1201__)\n"
"__global__ void gemm_f16w_bf16a_wmma_t(float *Y, const half_raw *W, const float *X,\n"
"                                         const float *bias, int n_out, int n_in, int n_tok) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane    = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx  = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ short smA[128*16];\n"
"    __shared__ short smB[128*16];\n"
"    typedef float float8 __attribute__((ext_vector_type(8)));\n"
"    typedef short bf16x8 __attribute__((ext_vector_type(8)));\n"
"    float8 cv00 = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};\n"
"    float8 cv01 = cv00, cv10 = cv00, cv11 = cv00, cv20 = cv00, cv21 = cv00, cv30 = cv00, cv31 = cv00;\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        for (int it = 0; it < 8; it++) {\n"
"            int e = tid * 8 + it;\n"
"            int er = e >> 4;\n"
"            int ek = e & 15;\n"
"            int row = cta_m0 + er;\n"
"            int kp  = k + ek;\n"
"            float xv = (row < n_tok && kp < n_in) ? X[(long)row * n_in + kp] : 0.f;\n"
"            unsigned int xbits; memcpy(&xbits, &xv, 4);\n"
"            smA[er * 16 + ek] = (short)(xbits >> 16);\n"
"        }\n"
"        for (int it = 0; it < 8; it++) {\n"
"            int e = tid * 8 + it;\n"
"            int er = e >> 4;\n"
"            int ek = e & 15;\n"
"            int col = cta_n0 + er;\n"
"            int kp  = k + ek;\n"
"            float wv = (col < n_out && kp < n_in)\n"
"                     ? half_to_float(W[(long)col * n_in + kp]) : 0.f;\n"
"            unsigned int wbits; memcpy(&wbits, &wv, 4);\n"
"            smB[er * 16 + ek] = (short)(wbits >> 16);\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base_row = wM * 64;\n"
"        int b_base_row = wN * 32;\n"
"        bf16x8 a0, a1, a2, a3, b0, b1;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            a0[i] = smA[(a_base_row + 0  + idx) * 16 + k_off + i];\n"
"            a1[i] = smA[(a_base_row + 16 + idx) * 16 + k_off + i];\n"
"            a2[i] = smA[(a_base_row + 32 + idx) * 16 + k_off + i];\n"
"            a3[i] = smA[(a_base_row + 48 + idx) * 16 + k_off + i];\n"
"            b0[i] = smB[(b_base_row + 0  + idx) * 16 + k_off + i];\n"
"            b1[i] = smB[(b_base_row + 16 + idx) * 16 + k_off + i];\n"
"        }\n"
"        cv00 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0, b0, cv00);\n"
"        cv01 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0, b1, cv01);\n"
"        cv10 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1, b0, cv10);\n"
"        cv11 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1, b1, cv11);\n"
"        cv20 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2, b0, cv20);\n"
"        cv21 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2, b1, cv21);\n"
"        cv30 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3, b0, cv30);\n"
"        cv31 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3, b1, cv31);\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8]       = {0,   0,  16, 16, 32, 32, 48, 48};\n"
"    int ns[8]       = {0,  16,   0, 16,  0, 16,  0, 16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        if (col >= n_out) continue;\n"
"        float bv = bias ? bias[col] : 0.f;\n"
"        float8 acc = *accs[t];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row < n_tok)\n"
"                Y[(long)row * n_out + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"#else\n"
"__global__ void gemm_f16w_bf16a_wmma_t(float *Y, const half_raw *W, const float *X,\n"
"                                         const float *bias, int n_out, int n_in, int n_tok) {\n"
"    /* non-gfx12 fallback: unused (host dispatches to gemm_tiled_f16_f32) */\n"
"}\n"
"#endif\n\n"

/* Close the extern "C" block opened by hip_kernels_common_src */
"} /* extern C */\n"
;

#endif /* HIP_HY3D_KERNELS_H */
