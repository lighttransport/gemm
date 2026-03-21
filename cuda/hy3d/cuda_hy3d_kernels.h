/*
 * cuda_hy3d_kernels.h - CUDA kernel source strings for Hunyuan3D-2.1
 *
 * Contains all HY3D-specific CUDA kernels as a C string literal for
 * NVRTC runtime compilation. Concatenated after cuda_kernels_common_src
 * (which opens extern "C" {) and closes it at the end.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef CUDA_HY3D_KERNELS_H
#define CUDA_HY3D_KERNELS_H

static const char cuda_hy3d_specific_kernels[] =
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
/*                                                                       */
/* Matches PyTorch Timesteps class exactly:                              */
/*   half_dim = dim / 2                                                  */
/*   exponent = -log(10000) * arange(0, half_dim) / half_dim             */
/*   emb = exp(exponent)                                                 */
/*   emb = timestep * emb   (NO 1000x scaling)                          */
/*   out = [sin(emb), cos(emb)]                                         */
/*                                                                       */
/* The timestep value is already a raw float (e.g. 0.5) as passed to    */
/* the DiT; the Timesteps class does NOT multiply by 1000.              */
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
/*                                                                       */
/* Computes softmax over expert logits and selects top-k experts.        */
/* logits: [n_experts] expert scores for a single token                  */
/* indices: [top_k] output -- selected expert indices                    */
/* weights: [top_k] output -- softmax weights for selected experts       */
/*                                                                       */
/* Single-threaded kernel (called once per token). For small n_experts   */
/* (typically 4-8) this is efficient enough.                             */
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
/* Used for skip-connection concat before linear projection.             */
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

/* Close the extern "C" block opened by cuda_kernels_common_src */
"} /* extern C */\n"
;

#endif /* CUDA_HY3D_KERNELS_H */
