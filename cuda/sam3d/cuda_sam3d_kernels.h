/* NVRTC kernel source strings for cuda/sam3d.
 *
 * Phase 1b.0: layernorm_token_f32 lands as the first reusable kernel —
 * shared by DINOv2 (per-token LN), CondFuser, SS-DiT, SLAT-DiT, and
 * SLAT-GS-decoder forward paths. Standalone-tested via
 * test_layernorm_token; not yet wired into any cuda_sam3d_run_* path.
 *
 * Subsequent phases append per-stage kernels — see
 * cuda/sam3d_body/cuda_sam3d_body_kernels.h and
 * cuda/trellis2/cuda_trellis2_kernels.h for templates.
 */

#ifndef CUDA_SAM3D_KERNELS_H_
#define CUDA_SAM3D_KERNELS_H_

static const char *cuda_sam3d_kernel_src =
"extern \"C\" __global__ void sam3d_sentinel(int *flag) {                   \n"
"    if (threadIdx.x == 0 && blockIdx.x == 0) flag[0] = 0xC0DE;             \n"
"}                                                                          \n"

/* layernorm_token_f32
 *
 * Per-token LayerNorm over the last (feature) dim:
 *   dst[t, c] = (src[t, c] - mean_t) * rsqrt(var_t + eps) * gamma[c] + beta[c]
 * gamma and beta may be NULL — pass them in via an `affine` flag (=1 to
 * apply, =0 to skip). When affine=0 the per-feature scale/shift are not
 * read, so callers may pass any pointer.
 *
 * One block per token; blockDim.x threads reduce-then-broadcast. Stride
 * over dim with `for c = tid; c < dim; c += blockDim.x`. Block size
 * should be a multiple of 32; 256 is a good default for dim ≤ 4096.
 *
 * Grid: (n_tokens,)        Block: (THREADS,)
 */
"extern \"C\" __global__ void layernorm_token_f32(\n"
"    float *dst, const float *src,\n"
"    const float *gamma, const float *beta,\n"
"    int n_tokens, int dim, float eps, int affine) {\n"
"    int t = blockIdx.x;\n"
"    if (t >= n_tokens) return;\n"
"    const float *row_in  = src + (size_t)t * dim;\n"
"    float       *row_out = dst + (size_t)t * dim;\n"
"\n"
"    extern __shared__ float smem[];   /* size = 2 * blockDim.x */\n"
"    float *s_sum = smem;\n"
"    float *s_sq  = smem + blockDim.x;\n"
"\n"
"    float lsum = 0.0f, lsq = 0.0f;\n"
"    for (int c = threadIdx.x; c < dim; c += blockDim.x) {\n"
"        float v = row_in[c];\n"
"        lsum += v;\n"
"        lsq  += v * v;\n"
"    }\n"
"    s_sum[threadIdx.x] = lsum;\n"
"    s_sq [threadIdx.x] = lsq;\n"
"    __syncthreads();\n"
"    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {\n"
"        if (threadIdx.x < s) {\n"
"            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];\n"
"            s_sq [threadIdx.x] += s_sq [threadIdx.x + s];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    float mean = s_sum[0] / (float)dim;\n"
"    float var  = s_sq [0] / (float)dim - mean * mean;\n"
"    float inv  = rsqrtf(var + eps);\n"
"\n"
"    if (affine) {\n"
"        for (int c = threadIdx.x; c < dim; c += blockDim.x) {\n"
"            row_out[c] = (row_in[c] - mean) * inv * gamma[c] + beta[c];\n"
"        }\n"
"    } else {\n"
"        for (int c = threadIdx.x; c < dim; c += blockDim.x) {\n"
"            row_out[c] = (row_in[c] - mean) * inv;\n"
"        }\n"
"    }\n"
"}\n"

/* modulated_ln_f32
 *
 * Fused AdaLN-modulated LayerNorm (no per-feature affine):
 *   y[t, c] = (x[t, c] - mean_t) * rsqrt(var_t + eps) * (1 + scale[c]) + shift[c]
 * shift, scale are [dim] vectors broadcast across all tokens — these are
 * the per-block AdaLN modulation params produced by `silu(t_emb) @ W_adaLN`
 * and sliced into msa_/mlp_ shift/scale (see Phase 2c.2). Pass shift=NULL
 * and scale=NULL to fall through to plain (no-affine, no-modulation) LN —
 * useful for callers that share this kernel across norm sites that don't
 * modulate.
 *
 * Grid: (n_tokens,)        Block: (THREADS,)  (256 default; mult of 32)
 * Smem: 2 * blockDim.x * sizeof(float)
 */
"extern \"C\" __global__ void modulated_ln_f32(\n"
"    float *dst, const float *src,\n"
"    const float *shift, const float *scale,\n"
"    int n_tokens, int dim, float eps) {\n"
"    int t = blockIdx.x;\n"
"    if (t >= n_tokens) return;\n"
"    const float *row_in  = src + (size_t)t * dim;\n"
"    float       *row_out = dst + (size_t)t * dim;\n"
"\n"
"    extern __shared__ float smem[];\n"
"    float *s_sum = smem;\n"
"    float *s_sq  = smem + blockDim.x;\n"
"\n"
"    float lsum = 0.0f, lsq = 0.0f;\n"
"    for (int c = threadIdx.x; c < dim; c += blockDim.x) {\n"
"        float v = row_in[c];\n"
"        lsum += v;\n"
"        lsq  += v * v;\n"
"    }\n"
"    s_sum[threadIdx.x] = lsum;\n"
"    s_sq [threadIdx.x] = lsq;\n"
"    __syncthreads();\n"
"    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {\n"
"        if (threadIdx.x < s) {\n"
"            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];\n"
"            s_sq [threadIdx.x] += s_sq [threadIdx.x + s];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    float mean = s_sum[0] / (float)dim;\n"
"    float var  = s_sq [0] / (float)dim - mean * mean;\n"
"    float inv  = rsqrtf(var + eps);\n"
"\n"
"    if (shift && scale) {\n"
"        for (int c = threadIdx.x; c < dim; c += blockDim.x) {\n"
"            float v = (row_in[c] - mean) * inv;\n"
"            row_out[c] = v * (1.0f + scale[c]) + shift[c];\n"
"        }\n"
"    } else {\n"
"        for (int c = threadIdx.x; c < dim; c += blockDim.x) {\n"
"            row_out[c] = (row_in[c] - mean) * inv;\n"
"        }\n"
"    }\n"
"}\n"

/* multi_head_rmsnorm_f32
 *
 * SS DiT's MultiHeadRMSNorm — NOT classic RMSNorm. Upstream uses
 * `F.normalize(x, dim=-1)` (L2 normalize over head_dim) and then
 * multiplies by gamma[h, c] AND sqrt(head_dim):
 *
 *   inv = 1 / (sqrt(sum_i x[i]^2) + 1e-12)
 *   y[i] = x[i] * inv * gamma[h, i] * sqrt(head_dim)
 *
 * In place; one block per (token, head) pair. The token is laid out as
 * [n_tokens, stride] with `stride >= n_heads * head_dim` — supports
 * running directly on packed QKV ([n_tokens, 3*n_heads*head_dim]) by
 * passing stride=3*n_heads*head_dim and head_off=0/1/2 * H*D inside the
 * caller's pointer.
 *
 * Grid: (n_heads, n_tokens)  Block: (THREADS,)  THREADS ≥ 32
 * Smem:  blockDim.x * sizeof(float)
 */
"extern \"C\" __global__ void multi_head_rmsnorm_f32(\n"
"    float *x, const float *gamma,\n"
"    int n_tokens, int n_heads, int head_dim, int stride) {\n"
"    int t = blockIdx.y;\n"
"    int h = blockIdx.x;\n"
"    if (t >= n_tokens || h >= n_heads) return;\n"
"    float       *row = x     + (size_t)t * stride + h * head_dim;\n"
"    const float *g   = gamma + (size_t)h * head_dim;\n"
"\n"
"    extern __shared__ float smem[];\n"
"    float lsq = 0.0f;\n"
"    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {\n"
"        float v = row[i];\n"
"        lsq += v * v;\n"
"    }\n"
"    smem[threadIdx.x] = lsq;\n"
"    __syncthreads();\n"
"    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {\n"
"        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv   = 1.0f / (sqrtf(smem[0]) + 1e-12f);\n"
"    float scale = sqrtf((float)head_dim);\n"
"    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {\n"
"        row[i] = row[i] * inv * g[i] * scale;\n"
"    }\n"
"}\n"

/* dinov2_patch_embed_f32
 *
 * DINOv2 patch embed: Conv2d(in=3, out=dim, kernel=ps, stride=ps,
 * padding=0). Equivalent to per-patch dot(weight, image_window) + bias.
 *
 *   img: [3, H, W] CHW f32, ImageNet-normalized; H = W = grid * ps.
 *   w:   [dim, 3, ps, ps] f32 (PyTorch Conv2d order).
 *   b:   [dim] f32 (may be NULL).
 *   out: [n_tokens_total, dim] f32. The kernel writes
 *        out[(base_tok + patch) * dim + co] for patch in [0, gh*gw).
 *        Pass base_tok=0 to write contiguous [n_patches, dim], or
 *        base_tok = 1 + n_register to write directly into the prepended
 *        token layout (matching sam3d_body's pattern).
 *
 * Grid: (n_patches,)        Block: (THREADS,)  (256 is a good default)
 */
"extern \"C\" __global__ void dinov2_patch_embed_f32(\n"
"    float *out, const float *img, const float *w, const float *b,\n"
"    int gw, int dim, int ps, int img_w, int base_tok) {\n"
"    int patch = blockIdx.x;\n"
"    int tid   = threadIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    int tok = base_tok + patch;\n"
"    for (int co = tid; co < dim; co += blockDim.x) {\n"
"        float sum = b ? b[co] : 0.0f;\n"
"        for (int ci = 0; ci < 3; ci++)\n"
"            for (int kh = 0; kh < ps; kh++)\n"
"                for (int kw = 0; kw < ps; kw++)\n"
"                    sum += w[((co * 3 + ci) * ps + kh) * ps + kw]\n"
"                         * img[ci * img_w * img_w + (py * ps + kh) * img_w + (px * ps + kw)];\n"
"        out[(size_t)tok * dim + co] = sum;\n"
"    }\n"
"}\n"

/* dinov2_prepend_cls_reg_f32
 *
 * Writes CLS into row 0 and the n_register register tokens into rows
 * 1..n_register of `out` (a [n_tokens, dim] activation buffer where
 * patches are already in rows n_register+1..). No pos_embed is added
 * here — see dinov2_add_pos_embed_f32 for that.
 *
 * `register_tokens` may be NULL iff n_register == 0.
 *
 * Grid: (1 + n_register,)   Block: (THREADS,)  one block per token.
 */
"extern \"C\" __global__ void dinov2_prepend_cls_reg_f32(\n"
"    float *out, const float *cls, const float *register_tokens,\n"
"    int n_register, int dim) {\n"
"    int row = blockIdx.x;\n"
"    if (row > n_register) return;\n"
"    float *dst = out + (size_t)row * dim;\n"
"    if (row == 0) {\n"
"        for (int c = threadIdx.x; c < dim; c += blockDim.x) dst[c] = cls[c];\n"
"    } else {\n"
"        const float *src = register_tokens + (size_t)(row - 1) * dim;\n"
"        for (int c = threadIdx.x; c < dim; c += blockDim.x) dst[c] = src[c];\n"
"    }\n"
"}\n"

/* dinov2_add_pos_embed_f32
 *
 * Adds the learned pos_embed to CLS (row 0) and to each patch row
 * (rows n_register+1..n_register+n_patches). Register tokens get NO
 * pos_embed. pos_embed has shape [1 + n_patches, dim] with layout
 *   pos_embed[0]      = CLS pos embed
 *   pos_embed[1 + p]  = patch p pos embed
 *
 * v1 assumes the safetensors-loaded pos_embed grid already matches the
 * runtime grid (518/14 = 37 → 1369 patches). Bicubic interpolation for
 * mismatched sizes is deferred to a separate pre-pass kernel.
 *
 * Grid: (1 + n_patches,)   Block: (THREADS,)
 */
"extern \"C\" __global__ void dinov2_add_pos_embed_f32(\n"
"    float *out, const float *pos_embed,\n"
"    int n_register, int n_patches, int dim) {\n"
"    int idx = blockIdx.x;\n"
"    if (idx > n_patches) return;\n"
"    int target_tok = (idx == 0) ? 0 : (n_register + idx);\n"
"    int pe_idx     = idx;  /* 0 → CLS pe, 1+p → patch p pe */\n"
"    float       *row = out       + (size_t)target_tok * dim;\n"
"    const float *pe  = pos_embed + (size_t)pe_idx     * dim;\n"
"    for (int c = threadIdx.x; c < dim; c += blockDim.x)\n"
"        row[c] += pe[c];\n"
"}\n"

/* gemm_f32_bias
 *
 *   Y(N, D_out) = X(N, D_in) @ W^T(D_out, D_in) + b(D_out)
 *
 * One thread per (n, d_out). Used by every linear layer in DINOv2 +
 * the upcoming DiT/decoder stages: QKV-fused projection, attn out_proj,
 * MLP fc1, MLP fc2. `b` may be NULL.
 *
 * Grid: (ceil(N/16), ceil(D_out/16))   Block: (16, 16).
 */
"extern \"C\" __global__ void gemm_f32_bias(\n"
"    float *Y, const float *X, const float *W, const float *b,\n"
"    int N, int D_in, int D_out) {\n"
"    int n = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int d = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (n >= N || d >= D_out) return;\n"
"    const float *xr = X + (size_t)n * D_in;\n"
"    const float *wr = W + (size_t)d * D_in;\n"
"    float acc = (b ? b[d] : 0.0f);\n"
"    for (int k = 0; k < D_in; k++) acc += wr[k] * xr[k];\n"
"    Y[(size_t)n * D_out + d] = acc;\n"
"}\n"

/* qkv_split_f32
 *
 * Splits a fused QKV row-major buffer [n, 3*dim] into three [n, dim]
 * row-major buffers Q, K, V (the layout sdpa_f32 expects). One thread
 * per output element.
 *
 * Grid: (ceil(n*dim/256),)   Block: (256,).
 */
"extern \"C\" __global__ void qkv_split_f32(\n"
"    float *Q, float *K, float *V, const float *qkv, int n, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n * dim;\n"
"    if (idx >= total) return;\n"
"    int t = idx / dim;\n"
"    int d = idx - t * dim;\n"
"    size_t row3 = (size_t)t * (size_t)(3 * dim);\n"
"    Q[idx] = qkv[row3 + (size_t)d];\n"
"    K[idx] = qkv[row3 + (size_t)dim + (size_t)d];\n"
"    V[idx] = qkv[row3 + (size_t)(2 * dim) + (size_t)d];\n"
"}\n"

/* kv_split_f32
 *
 * Splits a fused KV row-major buffer [n, 2*dim] (the output of
 * `xa_kv = Linear(cond, 2*dim)`) into two [n, dim] row-major buffers
 * K and V, which is the layout `sdpa_f32` consumes. CPU reference
 * `cpu_xattn_worker` reads the same interleaved layout directly via
 * pointer offsets per (ki, h); on GPU we materialize the split because
 * `sdpa_f32` hardcodes `E = H * D_h` as the per-row stride.
 *
 * Grid: (ceil(n*dim/256),)   Block: (256,).
 */
"extern \"C\" __global__ void kv_split_f32(\n"
"    float *K, float *V, const float *kv, int n, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n * dim;\n"
"    if (idx >= total) return;\n"
"    int t = idx / dim;\n"
"    int d = idx - t * dim;\n"
"    size_t row2 = (size_t)t * (size_t)(2 * dim);\n"
"    K[idx] = kv[row2 + (size_t)d];\n"
"    V[idx] = kv[row2 + (size_t)dim + (size_t)d];\n"
"}\n"

/* sdpa_f32 — scaled dot-product attention.
 *
 *   q (N_q, H*D_h), k (N_k, H*D_h), v (N_k, H*D_h), out (N_q, H*D_h).
 * One block per (n_q, h). 256 threads parallelize N_k for scores +
 * softmax, then D_h for the value-mixing output.
 *
 * Grid: (N_q, H)   Block: (256,)
 * Shmem: (256 + N_k) * sizeof(float).
 */
"extern \"C\" __global__ void sdpa_f32(\n"
"    float *out, const float *q, const float *k, const float *v,\n"
"    int N_q, int N_k, int H, int D_h, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int nq = blockIdx.x;\n"
"    int h  = blockIdx.y;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int E  = H * D_h;\n"
"    const float *qv = q + (size_t)nq * E + (size_t)h * D_h;\n"
"    float *s_red  = smem;\n"
"    float *scores = smem + nt;\n"
"    for (int nk = tid; nk < N_k; nk += nt) {\n"
"        const float *kv = k + (size_t)nk * E + (size_t)h * D_h;\n"
"        float s = 0.0f;\n"
"        for (int d = 0; d < D_h; d++) s += qv[d] * kv[d];\n"
"        scores[nk] = s * scale;\n"
"    }\n"
"    __syncthreads();\n"
"    float lmax = -1e38f;\n"
"    for (int nk = tid; nk < N_k; nk += nt)\n"
"        if (scores[nk] > lmax) lmax = scores[nk];\n"
"    s_red[tid] = lmax;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) {\n"
"            float a = s_red[tid], b = s_red[tid + r];\n"
"            s_red[tid] = (a > b) ? a : b;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    float gmax = s_red[0];\n"
"    __syncthreads();\n"
"    for (int nk = tid; nk < N_k; nk += nt)\n"
"        scores[nk] = expf(scores[nk] - gmax);\n"
"    __syncthreads();\n"
"    float lsum = 0.0f;\n"
"    for (int nk = tid; nk < N_k; nk += nt) lsum += scores[nk];\n"
"    s_red[tid] = lsum;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) s_red[tid] += s_red[tid + r];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv = 1.0f / s_red[0];\n"
"    __syncthreads();\n"
"    for (int d = tid; d < D_h; d += nt) {\n"
"        float acc = 0.0f;\n"
"        for (int nk = 0; nk < N_k; nk++)\n"
"            acc += scores[nk] * v[(size_t)nk * E + (size_t)h * D_h + d];\n"
"        out[(size_t)nq * E + (size_t)h * D_h + d] = acc * inv;\n"
"    }\n"
"}\n"

/* sdpa_batched_f32 — independent SDPA per batch element.
 *
 *   q (B, N_q, H*D_h), k (B, N_k, H*D_h), v (B, N_k, H*D_h),
 *   out (B, N_q, H*D_h).  Each batch attends only to its own keys —
 *   windows in PointPatchEmbed don't share tokens.
 *
 * Grid: (N_q, H, B)   Block: (256,)
 * Shmem: (256 + N_k) * sizeof(float).
 *
 * Layout-matched with sdpa_f32: identical scores/softmax/value reduce,
 * but q/k/v/out pointers are pre-offset by `b * N_kq * H * D_h` for the
 * relevant N_kq.
 */
"extern \"C\" __global__ void sdpa_batched_f32(\n"
"    float *out, const float *q, const float *k, const float *v,\n"
"    int N_q, int N_k, int H, int D_h, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int nq  = blockIdx.x;\n"
"    int h   = blockIdx.y;\n"
"    int b   = blockIdx.z;\n"
"    int tid = threadIdx.x;\n"
"    int nt  = blockDim.x;\n"
"    int E   = H * D_h;\n"
"    size_t q_off = (size_t)b * N_q * E + (size_t)nq * E + (size_t)h * D_h;\n"
"    size_t kv_base = (size_t)b * N_k * E;\n"
"    const float *qv = q + q_off;\n"
"    float *s_red  = smem;\n"
"    float *scores = smem + nt;\n"
"    for (int nk = tid; nk < N_k; nk += nt) {\n"
"        const float *kv = k + kv_base + (size_t)nk * E + (size_t)h * D_h;\n"
"        float s = 0.0f;\n"
"        for (int d = 0; d < D_h; d++) s += qv[d] * kv[d];\n"
"        scores[nk] = s * scale;\n"
"    }\n"
"    __syncthreads();\n"
"    float lmax = -1e38f;\n"
"    for (int nk = tid; nk < N_k; nk += nt)\n"
"        if (scores[nk] > lmax) lmax = scores[nk];\n"
"    s_red[tid] = lmax;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) {\n"
"            float a = s_red[tid], bb = s_red[tid + r];\n"
"            s_red[tid] = (a > bb) ? a : bb;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    float gmax = s_red[0];\n"
"    __syncthreads();\n"
"    for (int nk = tid; nk < N_k; nk += nt)\n"
"        scores[nk] = __expf(scores[nk] - gmax);\n"
"    __syncthreads();\n"
"    float lsum = 0.0f;\n"
"    for (int nk = tid; nk < N_k; nk += nt) lsum += scores[nk];\n"
"    s_red[tid] = lsum;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) {\n"
"        if (tid < r) s_red[tid] += s_red[tid + r];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv = 1.0f / s_red[0];\n"
"    __syncthreads();\n"
"    for (int d = tid; d < D_h; d += nt) {\n"
"        float acc = 0.0f;\n"
"        for (int nk = 0; nk < N_k; nk++)\n"
"            acc += scores[nk] * v[kv_base + (size_t)nk * E + (size_t)h * D_h + d];\n"
"        out[q_off + d] = acc * inv;\n"
"    }\n"
"}\n"

/* layerscale_add_f32
 *
 *   hidden[t, c] += proj[t, c] * gamma[c]
 *
 * The DINOv2 LayerScale + residual fusion: gamma is the per-channel
 * ls1.gamma / ls2.gamma vector ([dim]), proj is the attention/MLP
 * branch output ([n_tok, dim]), hidden is the residual stream that
 * gets accumulated in place. One thread per (t, c).
 *
 * Grid: (ceil(n_tok*dim/256),)   Block: (256,).
 */
"extern \"C\" __global__ void layerscale_add_f32(\n"
"    float *hidden, const float *proj, const float *gamma,\n"
"    int n_tok, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * dim;\n"
"    if (i >= total) return;\n"
"    hidden[i] += proj[i] * gamma[i - (i / dim) * dim];\n"
"}\n"

/* gated_residual_add_f32
 *
 *   x[t, c] += t_in[t, c] * gate[c]
 *
 * Per-feature-broadcast gated residual used by SS DiT MOT blocks:
 *   x_shape += sa_out * gate_msa
 *   x_shape += mlp_out * gate_mlp
 * `gate` is shaped [dim] (one of the 6 AdaLN modulation slices). One
 * thread per element; n_tokens × dim elements total.
 *
 * Grid: (ceil(n_tokens*dim/256),)   Block: (256,).
 */
"extern \"C\" __global__ void gated_residual_add_f32(\n"
"    float *x, const float *t_in, const float *gate, int n_tokens, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tokens * dim;\n"
"    if (idx >= total) return;\n"
"    int c = idx - (idx / dim) * dim;\n"
"    x[idx] += t_in[idx] * gate[c];\n"
"}\n"

/* residual_add_f32
 *
 *   hidden[i] += proj[i]
 *
 * Plain residual sum used by blocks without LayerScale (e.g. the PPE
 * single ViT block). One thread per element.
 *
 * Grid: (ceil(n/256),)   Block: (256,).
 */
"extern \"C\" __global__ void residual_add_f32(\n"
"    float *hidden, const float *proj, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    hidden[i] += proj[i];\n"
"}\n"

/* slat_concat_feats_f32
 *
 *   out[n,:] = concat(a[n,:], b[n,:])
 *
 * Row-wise feature concat used by the SLAT decoder-side skip joins.
 * One thread per output element.
 *
 * Grid: (ceil(N*(Ca+Cb)/256),)   Block: (256,).
 */
"extern \"C\" __global__ void slat_concat_feats_f32(\n"
"    float *out, const float *a, const float *b, int N, int Ca, int Cb) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int C = Ca + Cb;\n"
"    int total = N * C;\n"
"    if (i >= total) return;\n"
"    int n = i / C;\n"
"    int c = i - n * C;\n"
"    out[i] = (c < Ca) ? a[n * Ca + c] : b[n * Cb + (c - Ca)];\n"
"}\n"

/* slat_unnormalize8_f32
 *
 *   x[n,c] = x[n,c] * std[c] + mean[c], c in [0,8)
 *
 * Final SLAT latent un-normalization for facebook/sam-3d-objects. This
 * mirrors the fixed CPU stats used after the SLAT ODE loop.
 *
 * Grid: (ceil(N*8/256),)   Block: (256,).
 */
"extern \"C\" __global__ void slat_unnormalize8_f32(float *x, int N) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = N * 8;\n"
"    if (i >= total) return;\n"
"    int c = i & 7;\n"
"    const float mean[8] = { 0.12211431f, 0.37204156f, -1.26521907f, -2.05276058f,\n"
"                           -3.10432536f, -0.11294304f, -0.85146744f, 0.45506954f };\n"
"    const float stdv[8] = { 2.37326008f, 2.13174402f, 2.24139530f, 2.30589401f,\n"
"                           2.11918940f, 1.89695110f, 2.41684989f, 2.08374642f };\n"
"    x[i] = x[i] * stdv[c] + mean[c];\n"
"}\n"

/* gelu_inplace_f32
 *
 *   x[i] = x[i] * 0.5 * (1 + erf(x[i] / sqrt(2)))
 *
 * Exact GELU — torch nn.GELU default (no `approximate='tanh'`).
 * One thread per element; used between MLP fc1 and fc2.
 *
 * Grid: (ceil(n/256),)   Block: (256,).
 */
"extern \"C\" __global__ void gelu_inplace_f32(float *x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float v = x[i];\n"
"    x[i] = v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));\n"
"}\n"

/* gelu_tanh_inplace_f32
 *
 *   u    = sqrt(2/pi) * (x + 0.044715 * x^3)
 *   x[i] = 0.5 * x * (1 + tanh(u))
 *
 * tanh-approx GELU (`approximate='tanh'` in torch nn.GELU). Used by SS
 * Flow DiT MLP between fc1 and fc2 — `ssdit_gelu_tanh_inplace`
 * (sam3d_ss_flow_dit.h) bit-for-bit. The exact erf-based variant is
 * `gelu_inplace_f32` above.
 *
 * Grid: (ceil(n/256),)   Block: (256,).
 */
"extern \"C\" __global__ void gelu_tanh_inplace_f32(float *x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float v = x[i];\n"
"    float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);\n"
"    x[i] = 0.5f * v * (1.0f + tanhf(u));\n"
"}\n"

/* silu_mul_f32
 *
 *   out[i] = silu(a[i]) * b[i],  silu(x) = x * sigmoid(x) = x / (1 + e^-x)
 *
 * SwiGLU activation core used by the CondFuser projection FFN
 * (Llama3 feed-forward: w2( silu(w1(x)) * w3(x) )).  One thread per
 * element; out may alias a or b for in-place use.
 *
 * Grid: (ceil(n/256),)   Block: (256,).
 */
"extern \"C\" __global__ void silu_mul_f32(const float *a, const float *b,\n"
"                                          float *out, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float av = a[i];\n"
"    float sig = 1.0f / (1.0f + __expf(-av));\n"
"    out[i] = (av * sig) * b[i];\n"
"}\n"

/* ppe_linear3_invalid_f32
 *
 *   For each pixel p in 0..(S*S-1) with xyz = pmap[p, 0..2]:
 *     if any of xyz is non-finite (NaN/Inf): out[p, d] = inv_tok[d]
 *     else:                                  out[p, d] = b[d] + sum_k W[d, k] * xyz[k]
 *
 * Mirrors PointPatchEmbed step-2 in `sam3d_cond_fuser.h`: the linear is
 * applied per-pixel, and invalid pixels (NaN xyz from MoGe) are
 * replaced with the learned `invalid_xyz_token` *after* the linear.
 *
 *   pmap:    [S, S, 3]  f32 (last dim contiguous)
 *   W:       [D, 3]     f32 (D rows of the linear weight)
 *   b:       [D]        f32 (linear bias; NULL → no bias)
 *   inv_tok: [D]        f32
 *   out:     [S*S, D]   f32
 *
 * Launch: one block per pixel; threads stride over D. Block size 256 is
 * a good default for D <= 1024.
 *
 *   Grid: (S*S,)        Block: (THREADS,)
 */
"extern \"C\" __global__ void ppe_linear3_invalid_f32(\n"
"    float *out, const float *pmap, const float *W, const float *b,\n"
"    const float *inv_tok, int n_pix, int D) {\n"
"    int p = blockIdx.x;\n"
"    if (p >= n_pix) return;\n"
"    float x = pmap[(size_t)p * 3 + 0];\n"
"    float y = pmap[(size_t)p * 3 + 1];\n"
"    float z = pmap[(size_t)p * 3 + 2];\n"
"    int valid = isfinite(x) && isfinite(y) && isfinite(z);\n"
"    float *or_ = out + (size_t)p * D;\n"
"    if (!valid) {\n"
"        for (int d = threadIdx.x; d < D; d += blockDim.x) or_[d] = inv_tok[d];\n"
"    } else {\n"
"        for (int d = threadIdx.x; d < D; d += blockDim.x) {\n"
"            const float *wr = W + (size_t)d * 3;\n"
"            float v = (b ? b[d] : 0.0f);\n"
"            v += wr[0] * x + wr[1] * y + wr[2] * z;\n"
"            or_[d] = v;\n"
"        }\n"
"    }\n"
"}\n"

/* ppe_window_pack_f32
 *
 *   For each window (wy, wx) in [0, Np)² and token t in [0, 1 + P²):
 *     out[(wy*Np + wx) * (1 + P*P) + t, d] =
 *         (t == 0 ? cls[d]
 *                 : xflat[(wy*P + ((t-1)/P)) * S + (wx*P + ((t-1)%P)), d])
 *         + pew[t, d]
 *
 *  S = Np * P (e.g. 32 * 8 = 256). One block per (window, token); threads
 *  stride over D. Combines step-3 of `sam3d_cond_fuser.h`'s PointPatchEmbed
 *  (window reshape + CLS prepend + pos_embed_window add) into a single
 *  launch — one D2D pass instead of three.
 *
 *   xflat: [S*S, D]   f32
 *   cls:   [D]        f32
 *   pew:   [1+P*P, D] f32
 *   out:   [Np*Np, 1+P*P, D] f32
 *
 *   Grid:  (Np*Np * (1+P*P),)   Block: (THREADS,)
 */
"extern \"C\" __global__ void ppe_window_pack_f32(\n"
"    float *out, const float *xflat, const float *cls, const float *pew,\n"
"    int Np, int P, int D) {\n"
"    int WL = 1 + P * P;\n"
"    int S  = Np * P;\n"
"    int wt = blockIdx.x;            /* (window * WL) + token */\n"
"    int Nwin = Np * Np;\n"
"    if (wt >= Nwin * WL) return;\n"
"    int w = wt / WL;\n"
"    int t = wt - w * WL;\n"
"    int wy = w / Np;\n"
"    int wx = w - wy * Np;\n"
"    const float *src = (t == 0)\n"
"        ? cls\n"
"        : xflat + (size_t)((wy * P + ((t - 1) / P)) * S + (wx * P + ((t - 1) % P))) * D;\n"
"    const float *pe  = pew + (size_t)t * D;\n"
"    float       *dst = out + (size_t)wt * D;\n"
"    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d] + pe[d];\n"
"}\n"

/* ppe_cls_pos_extract_f32 — Steps 5+6 of PointPatchEmbed.
 *
 *   For each window w in 0..Nwin-1 with (wy, wx) = (w/Np, w%Np):
 *     out[w, d] = tokens[w * WL + 0, d] + pe[d, wy, wx]
 *
 * Gathers the per-window CLS row from the WL-strided window stack and
 * adds the (1, D, Np, Np) CHW pos_embed in the same pass. Skips the
 * bilinear-resize path: sam3d always runs at input_size 256 so the
 * pos_embed grid (32×32) already matches Np×Np.
 *
 *   tokens: [Nwin, WL, D]
 *   pe:     [D, Np, Np]   (CHW)
 *   out:    [Nwin, D]
 *
 *   Grid: (Nwin,)   Block: (THREADS,)   threads stride over D.
 */
"extern \"C\" __global__ void ppe_cls_pos_extract_f32(\n"
"    float *out, const float *tokens, const float *pe,\n"
"    int Np, int WL, int D) {\n"
"    int w = blockIdx.x;\n"
"    int Nwin = Np * Np;\n"
"    if (w >= Nwin) return;\n"
"    int wy = w / Np;\n"
"    int wx = w - wy * Np;\n"
"    const float *cls_row = tokens + (size_t)w * WL * D;\n"
"    float       *dst     = out    + (size_t)w * D;\n"
"    int Np2 = Np * Np;\n"
"    int pe_off = wy * Np + wx;\n"
"    for (int d = threadIdx.x; d < D; d += blockDim.x) {\n"
"        dst[d] = cls_row[d] + pe[(size_t)d * Np2 + pe_off];\n"
"    }\n"
"}\n"

/* timestep_embed_cossin_f32 — sinusoidal positional embed for one
 * scalar `t`, layout [cos(args); sin(args)] matching SAM-3D's
 * `ssdit_freq_embed`. Used by the SS Flow DiT t/d-embedder
 * (Phase 2c.0). One thread per pair index in [0, freq_dim/2).
 *
 *   Grid: (ceil((freq_dim/2)/256),)   Block: (256,).
 */
"extern \"C\" __global__ void timestep_embed_cossin_f32(\n"
"    float *out, float t, int freq_dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int half = freq_dim / 2;\n"
"    if (i >= half) return;\n"
"    float exponent = -logf(10000.0f) * (float)i / (float)half;\n"
"    float arg = __expf(exponent) * t;\n"
"    out[i]        = cosf(arg);\n"
"    out[half + i] = sinf(arg);\n"
"}\n"

/* silu_inplace_f32 — x[i] = x[i] * sigmoid(x[i]). One thread per element.
 *
 *   Grid: (ceil(n/256),)   Block: (256,).
 */
"extern \"C\" __global__ void silu_inplace_f32(float *x, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float v = x[i];\n"
"    x[i] = v / (1.0f + __expf(-v));\n"
"}\n"

/* channel_layernorm_3d_f32
 *
 * TRELLIS/SAM3D SS decoder "GroupNorm" sites are channel layernorm:
 * normalize over C independently for each spatial position in NCDHW.
 *
 *   dst[c, s] = (src[c, s] - mean_s) * rsqrt(var_s + eps) * gamma[c] + beta[c]
 *
 * Grid: (spatial,)   Block: 256.
 * Smem: 2 * blockDim.x * sizeof(float).
 */
"extern \"C\" __global__ void channel_layernorm_3d_f32(\n"
"    const float *src, float *dst, const float *gamma, const float *beta,\n"
"    int C, int spatial, float eps) {\n"
"    int sidx = blockIdx.x;\n"
"    if (sidx >= spatial) return;\n"
"    extern __shared__ float smem[];\n"
"    float *ss = smem;\n"
"    float *sq = smem + blockDim.x;\n"
"    float sum = 0.0f, sumsq = 0.0f;\n"
"    for (int c = threadIdx.x; c < C; c += blockDim.x) {\n"
"        float v = src[(size_t)c * spatial + sidx];\n"
"        sum += v;\n"
"        sumsq += v * v;\n"
"    }\n"
"    ss[threadIdx.x] = sum;\n"
"    sq[threadIdx.x] = sumsq;\n"
"    __syncthreads();\n"
"    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n"
"        if (threadIdx.x < stride) {\n"
"            ss[threadIdx.x] += ss[threadIdx.x + stride];\n"
"            sq[threadIdx.x] += sq[threadIdx.x + stride];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    float mean = ss[0] / (float)C;\n"
"    float var = sq[0] / (float)C - mean * mean;\n"
"    float inv = rsqrtf(var + eps);\n"
"    for (int c = threadIdx.x; c < C; c += blockDim.x) {\n"
"        float g = gamma ? gamma[c] : 1.0f;\n"
"        float b = beta  ? beta [c] : 0.0f;\n"
"        dst[(size_t)c * spatial + sidx] =\n"
"            (src[(size_t)c * spatial + sidx] - mean) * inv * g + b;\n"
"    }\n"
"}\n"

/* conv3d_k3_pad1_f32
 *
 * Dense Conv3d with kernel_size=3, padding=1, stride=1.
 * Weight layout: [Co, Ci, 3, 3, 3]. Input/output layout: [C, D, H, W].
 * One thread computes one output element and accumulates in CPU loop order.
 */
"extern \"C\" __global__ void conv3d_k3_pad1_f32(\n"
"    const float *src, float *dst, const float *weight, const float *bias,\n"
"    int Ci, int Co, int D, int H, int W) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int spatial = D * H * W;\n"
"    int total = Co * spatial;\n"
"    if (idx >= total) return;\n"
"    int s = idx % spatial;\n"
"    int co = idx / spatial;\n"
"    int z = s / (H * W);\n"
"    int rem = s - z * H * W;\n"
"    int y = rem / W;\n"
"    int x = rem - y * W;\n"
"    float acc = bias ? bias[co] : 0.0f;\n"
"    for (int ci = 0; ci < Ci; ci++) {\n"
"        const float *kern = weight + ((size_t)co * Ci + ci) * 27;\n"
"        const float *src_ci = src + (size_t)ci * spatial;\n"
"        for (int kd = 0; kd < 3; kd++) {\n"
"            int zz = z + kd - 1;\n"
"            if (zz < 0 || zz >= D) continue;\n"
"            for (int kh = 0; kh < 3; kh++) {\n"
"                int yy = y + kh - 1;\n"
"                if (yy < 0 || yy >= H) continue;\n"
"                for (int kw = 0; kw < 3; kw++) {\n"
"                    int xx = x + kw - 1;\n"
"                    if (xx < 0 || xx >= W) continue;\n"
"                    acc += kern[kd * 9 + kh * 3 + kw]\n"
"                         * src_ci[(zz * H + yy) * W + xx];\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    dst[idx] = acc;\n"
"}\n"

/* pixel_shuffle_3d_f32 — 3D pixel shuffle with upscale_factor=2.
 *
 * Rearranges [C*8, D, H, W] (NCDHW, batch=1) into [C, 2D, 2H, 2W].
 * Sub-channel layout matches torch / common/trellis2_ss_decoder.h:
 *     sub_ch(sd, sh, sw) = (sd*2 + sh)*2 + sw   for sd, sh, sw ∈ {0, 1}
 *     dst[c, 2d+sd, 2h+sh, 2w+sw] = src[c*8 + sub_ch, d, h, w]
 *
 * Pure data movement (zero arithmetic), so device output is bit-exact
 * vs the host reference. Used by the SS-VAE decoder upsample stages
 * 16³ → 32³ (C: 128) and 32³ → 64³ (C: 32).
 *
 * One thread per output element. Grid: (ceil(numel_out/256),).
 */
"extern \"C\" __global__ void pixel_shuffle_3d_f32(\n"
"    const float *src, float *dst,\n"
"    int C, int D, int H, int W) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int D2 = 2 * D, H2 = 2 * H, W2 = 2 * W;\n"
"    long long total = (long long)C * D2 * H2 * W2;\n"
"    if ((long long)idx >= total) return;\n"
"    int ow = idx % W2;\n"
"    int t1 = idx / W2;\n"
"    int oh = t1 % H2;\n"
"    int t2 = t1 / H2;\n"
"    int od = t2 % D2;\n"
"    int c  = t2 / D2;\n"
"    int sd = od & 1, d = od >> 1;\n"
"    int sh = oh & 1, h = oh >> 1;\n"
"    int sw = ow & 1, w = ow >> 1;\n"
"    int sub_ch = (sd * 2 + sh) * 2 + sw;\n"
"    int src_ch = c * 8 + sub_ch;\n"
"    long long src_idx =\n"
"        (long long)src_ch * D * H * W + (long long)d * H * W + (long long)h * W + w;\n"
"    dst[idx] = src[src_idx];\n"
"}\n"

/* slat_ape_add_f32
 *
 * Adds SLAT DiT absolute positional embedding to sparse token features.
 * Matches common/sam3d_slat_dit.h::slat_apply_ape:
 *   coord layout: coords[N,4] = (batch,z,y,x)
 *   freq_dim = dim / 3 / 2
 *   per-axis layout: [sin(j=0..F-1), cos(j=0..F-1)]
 *   channel tail after 3 * 2 * F is left unchanged.
 *
 * Grid: ceil(N * filled / 256), Block: 256.
 */
"extern \"C\" __global__ void slat_ape_add_f32(\n"
"    float *feats, const int *coords, int N, int dim) {\n"
"    int freq_dim = dim / 3 / 2;\n"
"    int per_axis = freq_dim * 2;\n"
"    int filled = per_axis * 3;\n"
"    long long total = (long long)N * filled;\n"
"    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (idx >= total) return;\n"
"    int c = (int)(idx % filled);\n"
"    int n = (int)(idx / filled);\n"
"    int axis = c / per_axis;\n"
"    int rem = c - axis * per_axis;\n"
"    int j = rem < freq_dim ? rem : rem - freq_dim;\n"
"    float coord = (float)coords[n * 4 + 1 + axis];\n"
"    float freq = expf(-logf(10000.0f) * (float)j / (float)freq_dim);\n"
"    float arg = coord * freq;\n"
"    float add = (rem < freq_dim) ? sinf(arg) : cosf(arg);\n"
"    feats[(size_t)n * dim + c] += add;\n"
"}\n"

/* slat_downsample2_mean_include_self_serial_f32
 *
 * Sparse downsample by factor 2 for SLAT input block 1. This matches
 * sp3d_downsample(..., factor=2, pool_mode=2):
 *   - coords are transformed as floor((z,y,x) / 2), batch unchanged
 *   - unique output coord order is first occurrence in input row order
 *   - features are summed per output coord, then divided by count + 1
 *     to match torch.scatter_reduce(mean, include_self=True) into zeros.
 *
 * Grid: (1,) Block: (1,). Kept serial to preserve deterministic coord
 * order while the surrounding SLAT ODE body is still CPU-backed.
 */
"extern \"C\" __global__ void slat_downsample2_mean_include_self_serial_f32(\n"
"    const int *coords, const float *feats, int N, int C,\n"
"    int *out_coords, float *out_feats, int *out_counts, int *out_N) {\n"
"    if (blockIdx.x != 0 || threadIdx.x != 0) return;\n"
"    for (int i = 0; i < N; i++) {\n"
"        out_counts[i] = 0;\n"
"        out_coords[i * 4 + 0] = 0;\n"
"        out_coords[i * 4 + 1] = 0;\n"
"        out_coords[i * 4 + 2] = 0;\n"
"        out_coords[i * 4 + 3] = 0;\n"
"    }\n"
"    for (long long i = 0; i < (long long)N * C; i++) out_feats[i] = 0.0f;\n"
"\n"
"    int M = 0;\n"
"    for (int i = 0; i < N; i++) {\n"
"        int b = coords[i * 4 + 0];\n"
"        int z = coords[i * 4 + 1] / 2;\n"
"        int y = coords[i * 4 + 2] / 2;\n"
"        int x = coords[i * 4 + 3] / 2;\n"
"        int oi = -1;\n"
"        for (int j = 0; j < M; j++) {\n"
"            if (out_coords[j * 4 + 0] == b && out_coords[j * 4 + 1] == z &&\n"
"                out_coords[j * 4 + 2] == y && out_coords[j * 4 + 3] == x) {\n"
"                oi = j;\n"
"                break;\n"
"            }\n"
"        }\n"
"        if (oi < 0) {\n"
"            oi = M++;\n"
"            out_coords[oi * 4 + 0] = b;\n"
"            out_coords[oi * 4 + 1] = z;\n"
"            out_coords[oi * 4 + 2] = y;\n"
"            out_coords[oi * 4 + 3] = x;\n"
"        }\n"
"        for (int c = 0; c < C; c++)\n"
"            out_feats[(size_t)oi * C + c] += feats[(size_t)i * C + c];\n"
"        out_counts[oi] += 1;\n"
"    }\n"
"    for (int i = 0; i < M; i++) {\n"
"        float inv = 1.0f / (float)(out_counts[i] + 1);\n"
"        for (int c = 0; c < C; c++) out_feats[(size_t)i * C + c] *= inv;\n"
"    }\n"
"    *out_N = M;\n"
"}\n"

/* slat_upsample2_nearest_f32
 *
 * Sparse nearest-neighbor upsample by factor 2 into caller-supplied
 * target coords. Matches sp3d_upsample(..., factor=2): for each target
 * coord, lookup source coord (b, z/2, y/2, x/2). If missing, output zeros.
 * Target row order is preserved exactly.
 *
 * Grid: ceil(target_N / 128), Block: 128. One thread owns one target row
 * and loops over C, so copied feature values are bit-exact.
 */
"extern \"C\" __global__ void slat_upsample2_nearest_f32(\n"
"    const int *src_coords, const float *src_feats, int src_N, int C,\n"
"    const int *target_coords, int target_N, float *out_feats) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= target_N) return;\n"
"    int b = target_coords[i * 4 + 0];\n"
"    int z = target_coords[i * 4 + 1] / 2;\n"
"    int y = target_coords[i * 4 + 2] / 2;\n"
"    int x = target_coords[i * 4 + 3] / 2;\n"
"    int src = -1;\n"
"    for (int j = 0; j < src_N; j++) {\n"
"        if (src_coords[j * 4 + 0] == b && src_coords[j * 4 + 1] == z &&\n"
"            src_coords[j * 4 + 2] == y && src_coords[j * 4 + 3] == x) {\n"
"            src = j;\n"
"            break;\n"
"        }\n"
"    }\n"
"    float *dst = out_feats + (size_t)i * C;\n"
"    if (src >= 0) {\n"
"        const float *src_row = src_feats + (size_t)src * C;\n"
"        for (int c = 0; c < C; c++) dst[c] = src_row[c];\n"
"    } else {\n"
"        for (int c = 0; c < C; c++) dst[c] = 0.0f;\n"
"    }\n"
"}\n"

/* slat_build_coord_index64_i32
 *
 * Builds a dense row-index grid for batch-0 sparse coords in a 64^3 domain.
 * Caller must initialize `index_grid` to -1 before launch.
 */
"extern \"C\" __global__ void slat_build_coord_index64_i32(\n"
"    const int *coords, int N, int *index_grid) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= N) return;\n"
"    int z = coords[i * 4 + 1];\n"
"    int y = coords[i * 4 + 2];\n"
"    int x = coords[i * 4 + 3];\n"
"    if ((unsigned)z < 64u && (unsigned)y < 64u && (unsigned)x < 64u)\n"
"        index_grid[(z * 64 + y) * 64 + x] = i;\n"
"}\n"

/* slat_submconv3x3_f32
 *
 * Submanifold sparse 3D convolution on fixed output coords. Matches
 * sp3d_conv3d_forward for kernel_size=3 and weight layout
 * [out_C, K3=27, in_C], where K3 order is dz-major, then dy, then dx
 * over {-1,0,1}. One thread computes one (voxel, out_channel), using
 * scalar accumulation in k/inner-channel order.
 */
"extern \"C\" __global__ void slat_submconv3x3_f32(\n"
"    const int *coords, const float *feats, const int *index_grid,\n"
"    const float *weight, const float *bias,\n"
"    int N, int in_C, int out_C, float *out) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = N * out_C;\n"
"    if (idx >= total) return;\n"
"    int oc = idx % out_C;\n"
"    int i = idx / out_C;\n"
"    int z0 = coords[i * 4 + 1];\n"
"    int y0 = coords[i * 4 + 2];\n"
"    int x0 = coords[i * 4 + 3];\n"
"    float acc = bias ? bias[oc] : 0.0f;\n"
"    for (int dz = -1; dz <= 1; dz++) {\n"
"        int z = z0 + dz;\n"
"        if ((unsigned)z >= 64u) continue;\n"
"        for (int dy = -1; dy <= 1; dy++) {\n"
"            int y = y0 + dy;\n"
"            if ((unsigned)y >= 64u) continue;\n"
"            for (int dx = -1; dx <= 1; dx++) {\n"
"                int x = x0 + dx;\n"
"                if ((unsigned)x >= 64u) continue;\n"
"                int src = index_grid[(z * 64 + y) * 64 + x];\n"
"                if (src < 0) continue;\n"
"                int k = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);\n"
"                const float *w = weight + ((size_t)oc * 27 + k) * in_C;\n"
"                const float *f = feats + (size_t)src * in_C;\n"
"                for (int ic = 0; ic < in_C; ic++) acc += w[ic] * f[ic];\n"
"            }\n"
"        }\n"
"    }\n"
"    out[(size_t)i * out_C + oc] = acc;\n"
"}\n"

/* slat_occ_argwhere_serial_f32
 *
 * Deterministic argwhere(occupancy > 0) for the SLAT-DiT sparse-token
 * bootstrap. The serial scan intentionally preserves the CPU reference
 * order: z-major, then y, then x, with coords stored as (batch,z,y,x).
 * This keeps downstream seeded noise and sparse ops byte-stable while
 * moving the prune boundary onto the CUDA runner side.
 *
 * Grid: (1,) Block: (1,). The input is only 64^3 in SAM3D, so this is
 * negligible next to the SLAT ODE fallback and avoids nondeterministic
 * atomic compaction order.
 */
"extern \"C\" __global__ void slat_occ_argwhere_serial_f32(\n"
"    const float *occ, int D, int H, int W, int *count, int *coords) {\n"
"    if (blockIdx.x != 0 || threadIdx.x != 0) return;\n"
"    int k = 0;\n"
"    for (int z = 0; z < D; z++) {\n"
"        for (int y = 0; y < H; y++) {\n"
"            for (int x = 0; x < W; x++) {\n"
"                int idx = (z * H + y) * W + x;\n"
"                if (occ[idx] > 0.0f) {\n"
"                    coords[k * 4 + 0] = 0;\n"
"                    coords[k * 4 + 1] = z;\n"
"                    coords[k * 4 + 2] = y;\n"
"                    coords[k * 4 + 3] = x;\n"
"                    k++;\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    *count = k;\n"
"}\n"

/* sam3d_gs_pack_ply_f32
 *
 * Fuses SLAT-GS to_representation and INRIA PLY layout packing.
 * Input feats are [N,448] from the GS decoder transformer. Output is
 * [N*num_gaussians, stride] with stride >= 17.
 */
"extern \"C\" __global__ void sam3d_gs_pack_ply_f32(\n"
"    float *out_ply, const int *coords, const float *feats,\n"
"    const float *offset_perturbation, int has_perturb,\n"
"    int N, int C, int G, int stride, int resolution,\n"
"    float voxel_size, float scaling_bias, float opacity_bias,\n"
"    int r_xyz0, int r_dc0, int r_scl0, int r_rot0, int r_op0,\n"
"    float lr_xyz, float lr_features_dc, float lr_scaling,\n"
"    float lr_rotation, float lr_opacity) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = N * G;\n"
"    if (idx >= total) return;\n"
"    int v = idx / G;\n"
"    int g = idx - v * G;\n"
"    const float *fp = feats + (size_t)v * C;\n"
"    float inv_res = 1.0f / (float)resolution;\n"
"    float *r = out_ply + (size_t)idx * stride;\n"
"    for (int a = 0; a < 3; a++) {\n"
"        float center = ((float)coords[v * 4 + 1 + a] + 0.5f) * inv_res;\n"
"        float off = fp[r_xyz0 + g * 3 + a] * lr_xyz;\n"
"        if (has_perturb) off += offset_perturbation[g * 3 + a];\n"
"        off = tanhf(off) * inv_res * 0.5f * voxel_size;\n"
"        r[a] = center + off;\n"
"    }\n"
"    r[3] = 0.0f; r[4] = 0.0f; r[5] = 0.0f;\n"
"    r[6] = fp[r_dc0 + g * 3 + 0] * lr_features_dc;\n"
"    r[7] = fp[r_dc0 + g * 3 + 1] * lr_features_dc;\n"
"    r[8] = fp[r_dc0 + g * 3 + 2] * lr_features_dc;\n"
"    r[9] = fp[r_op0 + g] * lr_opacity + opacity_bias;\n"
"    float inv_sp_sb = logf(expf(scaling_bias) - 1.0f);\n"
"    for (int a = 0; a < 3; a++) {\n"
"        float x = fp[r_scl0 + g * 3 + a] * lr_scaling + inv_sp_sb;\n"
"        float lsp = (x > 20.0f) ? logf(x) : ((x < -15.0f) ? x : logf(log1pf(expf(x))));\n"
"        r[10 + a] = lsp;\n"
"    }\n"
"    for (int a = 0; a < 4; a++) r[13 + a] = fp[r_rot0 + g * 4 + a] * lr_rotation;\n"
"}\n"

/* Gather/scatter helpers for SLAT-GS shifted-window attention. */
"extern \"C\" __global__ void sam3d_gs_gather_qkv_window_f32(\n"
"    float *Q, float *K, float *V, const float *qkv,\n"
"    const int *fwd, int start, int len, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = len * dim;\n"
"    if (idx >= total) return;\n"
"    int t = idx / dim;\n"
"    int d = idx - t * dim;\n"
"    int src = fwd[start + t];\n"
"    const float *row = qkv + (size_t)src * (3 * dim);\n"
"    Q[idx] = row[d];\n"
"    K[idx] = row[dim + d];\n"
"    V[idx] = row[2 * dim + d];\n"
"}\n"
"extern \"C\" __global__ void sam3d_gs_scatter_window_f32(\n"
"    float *out, const float *win, const int *fwd,\n"
"    int start, int len, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = len * dim;\n"
"    if (idx >= total) return;\n"
"    int t = idx / dim;\n"
"    int d = idx - t * dim;\n"
"    int dst = fwd[start + t];\n"
"    out[(size_t)dst * dim + d] = win[idx];\n"
"}\n"
;

#endif /* CUDA_SAM3D_KERNELS_H_ */
