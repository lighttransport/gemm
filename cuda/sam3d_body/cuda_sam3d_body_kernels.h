/* NVRTC kernel source for SAM 3D Body.
 *
 * Organization mirrors cuda/sam3.1/cuda_sam3_1_kernels.h: concatenate
 * __global__ source strings into a single const char array, compile
 * once via cu_compile_kernels() at create-time, cache the CUmodule.
 */

#ifndef CUDA_SAM3D_BODY_KERNELS_H_
#define CUDA_SAM3D_BODY_KERNELS_H_

/* The shared cuda_kernels_common_src opens an `extern "C" {` block that
 * this string is concatenated inside — kernels here must NOT redeclare
 * extern "C", and the final string must close the block with a `}`. */
static const char cuda_sam3d_body_kernels_src[] =

    /* Sentinel — used by the runner to confirm module-load success. */
    "__global__ void sam3d_body_sentinel(int *out) {\n"
    "    if (threadIdx.x == 0 && blockIdx.x == 0) *out = 1;\n"
    "}\n"

    /* patch_embed_sam3d
     *
     * DINOv3-H+ patch embed (3 → dim) with kernel size = patch size = ps,
     * stride = ps. Differs from the common patch_embed_conv2d by writing
     * to `out[base_tok + patch]` instead of hard-coded `1 + patch`, since
     * sam-3d-body prepends 1 CLS + 4 storage tokens.
     *
     * grid: (n_patches, 1, 1)   block: (256, 1, 1)
     */
    "__global__ void patch_embed_sam3d(float *out, const float *img, const float *w,\n"
    "                                  const float *bias, int gw, int dim, int ps,\n"
    "                                  int img_h, int img_w, int base_tok) {\n"
    "    int patch = blockIdx.x;\n"
    "    int tid = threadIdx.x;\n"
    "    int py = patch / gw, px = patch % gw;\n"
    "    int tok = base_tok + patch;\n"
    "    for (int co = tid; co < dim; co += blockDim.x) {\n"
    "        float sum = bias ? bias[co] : 0.0f;\n"
    "        for (int ci = 0; ci < 3; ci++)\n"
    "            for (int kh = 0; kh < ps; kh++)\n"
    "                for (int kw = 0; kw < ps; kw++)\n"
    "                    sum += w[((co*3+ci)*ps+kh)*ps+kw]\n"
    "                         * img[ci * img_h * img_w + (py*ps+kh) * img_w + (px*ps+kw)];\n"
    "        out[tok * dim + co] = sum;\n"
    "    }\n"
    "}\n"

    /* patch_im2col_sam3d
     *
     * Writes each non-overlapping DINOv3 patch as one row:
     *   cols[patch, ci*ps*ps + kh*ps + kw] = img[ci, py*ps+kh, px*ps+kw]
     * This lets the strict path use GEMM for patch embedding, closer to the
     * cuDNN/linearized reduction path used by PyTorch Conv2d.
     */
    "__global__ void patch_im2col_sam3d(float *cols, const float *img,\n"
    "                                  int gw, int ps, int img_h, int img_w) {\n"
    "    int K = 3 * ps * ps;\n"
    "    int k = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (k >= K) return;\n"
    "    int patch = blockIdx.y;\n"
    "    int ci = k / (ps * ps);\n"
    "    int r = k - ci * ps * ps;\n"
    "    int kh = r / ps;\n"
    "    int kw = r - kh * ps;\n"
    "    int py = patch / gw;\n"
    "    int px = patch - py * gw;\n"
    "    cols[(size_t)patch * K + k] = img[(size_t)ci * img_h * img_w + (py * ps + kh) * img_w + (px * ps + kw)];\n"
    "}\n"

    /* prepend_special_tokens
     *
     * Copies cls (dim,) into row 0 and storage (n_storage, dim) into rows
     * 1..1+n_storage of `out`. Patch tokens are written separately by
     * patch_embed_sam3d at base_tok = 1 + n_storage.
     *
     * grid: (ceil((1+S)*dim/256), 1, 1)   block: (256, 1, 1)
     */
    "__global__ void prepend_special_tokens(float *out, const float *cls,\n"
    "                                       const float *storage,\n"
    "                                       int n_storage, int dim) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = (1 + n_storage) * dim;\n"
    "    if (idx >= total) return;\n"
    "    int row = idx / dim;\n"
    "    int col = idx % dim;\n"
    "    if (row == 0)         out[idx] = cls[col];\n"
    "    else                  out[idx] = storage[(row - 1) * dim + col];\n"
    "}\n"

    /* bf16_round_inplace_f32
     *
     * Round a float buffer through bf16 precision, keeping f32 storage.
     * This is used for the explicit bf16 diagnostic path.
     */
    "__global__ void bf16_round_inplace_f32(float *x, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    unsigned int u = __float_as_uint(x[i]);\n"
    "    unsigned int lsb = (u >> 16) & 1u;\n"
    "    u += 0x7fffu + lsb;\n"
    "    u &= 0xffff0000u;\n"
    "    x[i] = __uint_as_float(u);\n"
    "}\n"

    /* rope_apply_qk_rh_sam3d
     *
     * In-place rotate-half RoPE applied to Q and K slots of a fused QKV
     * buffer (n_tok, 3*D). Skips the first `patch_start` tokens (CLS +
     * storage), and applies the rotation only to the next `n_patch`
     * tokens. cos/sin tables shape (n_patch, head_dim) — same layout
     * as common/dinov3.h: per-token vector [sy(rope_dim4), sx(rope_dim4),
     * sy(rope_dim4), sx(rope_dim4)].
     *
     * rotate_half([a, b]) = [-b, a]   where (a, b) split head_dim in half.
     *   v_new[j]      = v[j]      * c[j]      + (-v[hd/2+j]) * s[j]
     *   v_new[hd/2+j] = v[hd/2+j] * c[hd/2+j] + ( v[j])      * s[hd/2+j]
     *
     * Grid: (n_patch,)   Block: (heads,)   1 thread = 1 head.
     */
    "__global__ void rope_apply_qk_rh_sam3d(float *qkv, const float *cos_tbl,\n"
    "                                       const float *sin_tbl, int n_patch,\n"
    "                                       int patch_start, int heads,\n"
    "                                       int head_dim, int D) {\n"
    "    int p = blockIdx.x;\n"
    "    int h = threadIdx.x;\n"
    "    if (p >= n_patch || h >= heads) return;\n"
    "    int t = patch_start + p;\n"
    "    const float *c = cos_tbl + (size_t)p * head_dim;\n"
    "    const float *s = sin_tbl + (size_t)p * head_dim;\n"
    "    int half = head_dim / 2;\n"
    "    /* Q slot (offset 0*D), then K slot (offset 1*D). */\n"
    "    for (int qk = 0; qk < 2; qk++) {\n"
    "        float *v = qkv + (size_t)t * 3 * D + qk * D + h * head_dim;\n"
    "        for (int j = 0; j < half; j++) {\n"
    "            float a = v[j];\n"
    "            float b = v[half + j];\n"
    "            v[j]        = a * c[j]        + (-b) * s[j];\n"
    "            v[half + j] = b * c[half + j] +  a   * s[half + j];\n"
    "        }\n"
    "    }\n"
    "}\n"

    /* silu_mul_f32: gate := silu(gate) * up, in-place on gate. */
    "__global__ void silu_mul_f32(float *gate, const float *up, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    float g = gate[i];\n"
    "    gate[i] = (g / (1.0f + expf(-g))) * up[i];\n"
    "}\n"

    /* layerscale_add_f32: hidden[i] += proj[i] * gamma[i % dim]. */
    "__global__ void layerscale_add_f32(float *hidden, const float *proj,\n"
    "                                   const float *gamma, int n_tok, int dim) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = n_tok * dim;\n"
    "    if (i >= total) return;\n"
    "    hidden[i] += proj[i] * gamma[i % dim];\n"
    "}\n"

    /* ---- ray_cond_emb kernels (Step 5b) ----
     *
     * Input/output convention is CHW = (C, N=H*W) row-major: src[c*N + n].
     * This matches the CPU port (sam3d_body_ray_cond_emb_forward) and the
     * downstream CUDA decoder, which can flatten (1280, 32, 32) → (1024, 1280)
     * for cross-attention K/V via a transpose at consumption time.
     *
     * `preconv` is the concatenated (C_img + 99, N) buffer:
     *   rows [0..C_img)    : image_emb (1280, N) — copied in via hipMemcpyDtoD
     *   rows [C_img..+99)  : fourier features filled by ray_cond_fourier_chw_f32
     */

    /* ray_cond_fourier_chw_f32
     *
     * Fills `fp_chw` (99, N) with fourier positional encoding of input rays
     * at positions n=0..N-1. Layout per the CPU port:
     *   rows  0..2     : raw [x, y, z]
     *   rows  3..18    : sin(π·x·band[0..15])
     *   rows 19..34    : sin(π·y·band[0..15])
     *   rows 35..50    : sin(π·z·band[0..15])
     *   rows 51..66    : cos(π·x·band[0..15])
     *   rows 67..82    : cos(π·y·band[0..15])
     *   rows 83..98    : cos(π·z·band[0..15])
     * with band[b] = 1 + 31·b/(num_bands-1) for num_bands=16 → linspace(1, 32, 16).
     *
     * `rays_hwc` is (N, 3) — i.e., per-position 3 axis values consecutively.
     *
     * Grid: (ceil(N/256),)   Block: (256,).
     */
    "__global__ void ray_cond_fourier_chw_f32(float *fp_chw,\n"
    "                                         const float *rays_hwc,\n"
    "                                         int N, int num_bands) {\n"
    "    int n = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (n >= N) return;\n"
    "    float x = rays_hwc[n * 3 + 0];\n"
    "    float y = rays_hwc[n * 3 + 1];\n"
    "    float z = rays_hwc[n * 3 + 2];\n"
    "    fp_chw[(size_t)0 * N + n] = x;\n"
    "    fp_chw[(size_t)1 * N + n] = y;\n"
    "    fp_chw[(size_t)2 * N + n] = z;\n"
    "    int sin_base = 3;\n"
    "    int cos_base = 3 + 3 * num_bands;\n"
    "    float pi = 3.14159265358979323846f;\n"
    "    for (int b = 0; b < num_bands; b++) {\n"
    "        float band = (num_bands == 1)\n"
    "                     ? 1.0f\n"
    "                     : 1.0f + 31.0f * (float)b / (float)(num_bands - 1);\n"
    "        float vx = pi * x * band;\n"
    "        float vy = pi * y * band;\n"
    "        float vz = pi * z * band;\n"
    "        fp_chw[(size_t)(sin_base + 0 * num_bands + b) * N + n] = sinf(vx);\n"
    "        fp_chw[(size_t)(sin_base + 1 * num_bands + b) * N + n] = sinf(vy);\n"
    "        fp_chw[(size_t)(sin_base + 2 * num_bands + b) * N + n] = sinf(vz);\n"
    "        fp_chw[(size_t)(cos_base + 0 * num_bands + b) * N + n] = cosf(vx);\n"
    "        fp_chw[(size_t)(cos_base + 1 * num_bands + b) * N + n] = cosf(vy);\n"
    "        fp_chw[(size_t)(cos_base + 2 * num_bands + b) * N + n] = cosf(vz);\n"
    "    }\n"
    "}\n"

    /* encoder_tokens_to_preconv_nomask_f32
     *
     * Builds the image-token rows of ray_cond_emb's preconv buffer directly
     * from final encoder tokens: out[c, n] = tokens[n_prefix + n, c] + no_mask[c].
     */
    "__global__ void encoder_tokens_to_preconv_nomask_f32(float *out_chw,\n"
    "                                                     const float *tokens,\n"
    "                                                     const float *no_mask,\n"
    "                                                     int n_prefix,\n"
    "                                                     int N, int D) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = N * D;\n"
    "    if (idx >= total) return;\n"
    "    int n = idx / D;\n"
    "    int c = idx - n * D;\n"
    "    out_chw[(size_t)c * N + n] = tokens[(size_t)(n_prefix + n) * D + c] + no_mask[c];\n"
    "}\n"

    /* chw_to_tok_f32: out[n, c] = in[c, n]. */
    "__global__ void chw_to_tok_f32(float *out_tok,\n"
    "                               const float *in_chw,\n"
    "                               int N, int D) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = N * D;\n"
    "    if (idx >= total) return;\n"
    "    int n = idx / D;\n"
    "    int c = idx - n * D;\n"
    "    out_tok[(size_t)n * D + c] = in_chw[(size_t)c * N + n];\n"
    "}\n"

    /* conv1x1_chw_f32
     *
     * 1×1 conv with F32 weights. Y(C_out, N) = W(C_out, C_in) @ X(C_in, N).
     * Layout: row-major both for X and W. No bias.
     *
     * Grid: (ceil(N/256), C_out)   Block: (256,).
     */
    "__global__ void conv1x1_chw_f32(float *out, const float *W,\n"
    "                                const float *src,\n"
    "                                int C_out, int C_in, int N) {\n"
    "    int n = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int c = blockIdx.y;\n"
    "    if (n >= N || c >= C_out) return;\n"
    "    const float *Wrow = W + (size_t)c * C_in;\n"
    "    float acc = 0.0f;\n"
    "    for (int i = 0; i < C_in; i++)\n"
    "        acc += Wrow[i] * src[(size_t)i * N + n];\n"
    "    out[(size_t)c * N + n] = acc;\n"
    "}\n"

    /* layernorm_chw_f32
     *
     * Per-spatial LayerNorm over channels for (C, N) layout.
     *   mean[n] = mean over c of src[c, n]
     *   var[n]  = mean over c of (src[c, n] - mean[n])^2
     *   dst[c, n] = (src[c, n] - mean) / sqrt(var + eps) * gamma[c] + beta[c]
     *
     * Grid: (ceil(N/256),)   Block: (256,).
     */
    "__global__ void layernorm_chw_f32(float *dst, const float *src,\n"
    "                                  const float *gamma, const float *beta,\n"
    "                                  int C, int N, float eps) {\n"
    "    int n = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (n >= N) return;\n"
    "    float sum = 0.0f;\n"
    "    for (int c = 0; c < C; c++) sum += src[(size_t)c * N + n];\n"
    "    float mean = sum / (float)C;\n"
    "    float var = 0.0f;\n"
    "    for (int c = 0; c < C; c++) {\n"
    "        float d = src[(size_t)c * N + n] - mean;\n"
    "        var += d * d;\n"
    "    }\n"
    "    float inv = rsqrtf(var / (float)C + eps);\n"
    "    for (int c = 0; c < C; c++)\n"
    "        dst[(size_t)c * N + n] = (src[(size_t)c * N + n] - mean) * inv\n"
    "                                 * gamma[c] + beta[c];\n"
    "}\n"

    /* gemm_f32_bias
     *
     * Y(N, D_out) = X(N, D_in) @ W^T(D_out, D_in) + b(D_out).
     * One thread per (n, d_out). Grid 2D.
     *
     * Grid: (ceil(N/16), ceil(D_out/16))   Block: (16, 16).
     */
    "__global__ void gemm_f32_bias(float *Y, const float *X, const float *W,\n"
    "                              const float *b,\n"
    "                              int N, int D_in, int D_out) {\n"
    "    int n = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int d = blockIdx.y * blockDim.y + threadIdx.y;\n"
    "    if (n >= N || d >= D_out) return;\n"
    "    const float *xr = X + (size_t)n * D_in;\n"
    "    const float *wr = W + (size_t)d * D_in;\n"
    "    float acc = (b ? b[d] : 0.0f);\n"
    "    for (int k = 0; k < D_in; k++) acc += wr[k] * xr[k];\n"
    "    Y[(size_t)n * D_out + d] = acc;\n"
    "}\n"

    /* add_two_f32: out = a + b (element-wise). */
    "__global__ void add_two_f32(float *out, const float *a, const float *b,\n"
    "                            int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    out[i] = a[i] + b[i];\n"
    "}\n"

    /* add_inplace_f32: a += b. */
    "__global__ void add_inplace_f32(float *a, const float *b, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    a[i] += b[i];\n"
    "}\n"

    /* gelu_inplace_f32: exact GELU via erf — torch nn.GELU default. */
    "__global__ void gelu_inplace_f32(float *x, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= n) return;\n"
    "    float v = x[i];\n"
    "    x[i] = v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));\n"
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
    "__global__ void sdpa_f32(float *out, const float *q, const float *k,\n"
    "                         const float *v,\n"
    "                         int N_q, int N_k, int H, int D_h, float scale) {\n"
    "    extern __shared__ float smem[];\n"
    "    int nq = blockIdx.x;\n"
    "    int h  = blockIdx.y;\n"
    "    int tid = threadIdx.x;\n"
    "    int nt = blockDim.x;\n"
    "    int E  = H * D_h;\n"
    "    const float *qv = q + (size_t)nq * E + (size_t)h * D_h;\n"
    "    float *s_red  = smem;             /* [nt]    */\n"
    "    float *scores = smem + nt;        /* [N_k]  */\n"
    "    /* 1. scores[nk] = scale * Σ q[d] * k[nk, d] */\n"
    "    for (int nk = tid; nk < N_k; nk += nt) {\n"
    "        const float *kv = k + (size_t)nk * E + (size_t)h * D_h;\n"
    "        float s = 0.0f;\n"
    "        for (int d = 0; d < D_h; d++) s += qv[d] * kv[d];\n"
    "        scores[nk] = s * scale;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    /* 2. block-max */\n"
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
    "    /* 3. scores[nk] = exp(scores[nk] - max), sum */\n"
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
    "    /* 4. out[nq, h, d] = inv * Σ scores[nk] * v[nk, h, d] */\n"
    "    for (int d = tid; d < D_h; d += nt) {\n"
    "        float acc = 0.0f;\n"
    "        for (int nk = 0; nk < N_k; nk++)\n"
    "            acc += scores[nk] * v[(size_t)nk * E + (size_t)h * D_h + d];\n"
    "        out[(size_t)nq * E + (size_t)h * D_h + d] = acc * inv;\n"
    "    }\n"
    "}\n"

    /* linear_f32_bias
     *
     * Per-row matvec with bias. Y[d] = b[d] + Σ_k W[d, k] * x[k].
     * One thread per output channel; W is row-major (D_out, D_in).
     *
     * Grid: (ceil(D_out/256),)   Block: (256,).
     */
    "__global__ void linear_f32_bias(float *out, const float *W,\n"
    "                                const float *x, const float *b,\n"
    "                                int D_out, int D_in) {\n"
    "    int d = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (d >= D_out) return;\n"
    "    const float *wr = W + (size_t)d * D_in;\n"
    "    float acc = (b ? b[d] : 0.0f);\n"
    "    for (int k = 0; k < D_in; k++) acc += wr[k] * x[k];\n"
    "    out[d] = acc;\n"
    "}\n"

    /* add_bias_rows_f32
     *
     * Add a length-D bias vector to every row of a row-major (N, D) matrix.
     * Used after cuBLAS GEMM calls, whose wrapper intentionally does not
     * fuse bias.
     */
    "__global__ void add_bias_rows_f32(float *x, const float *b,\n"
    "                                  int N, int D) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = N * D;\n"
    "    if (idx >= total) return;\n"
    "    int d = idx - (idx / D) * D;\n"
    "    x[idx] += b ? b[d] : 0.0f;\n"
    "}\n"

    "__global__ void layernorm_welford_f32(float *dst, const float *src,\n"
    "                                      const float *w, const float *b,\n"
    "                                      int dim, float eps) {\n"
    "    extern __shared__ unsigned char smem_raw[];\n"
    "    float *smean = (float *)smem_raw;\n"
    "    float *sm2 = smean + blockDim.x;\n"
    "    int *sn = (int *)(sm2 + blockDim.x);\n"
    "    int tok = blockIdx.x;\n"
    "    int tid = threadIdx.x;\n"
    "    int nt = blockDim.x;\n"
    "    const float *x = src + (size_t)tok * dim;\n"
    "    float mean = 0.0f;\n"
    "    float m2 = 0.0f;\n"
    "    int n = 0;\n"
    "    for (int i = tid; i < dim; i += nt) {\n"
    "        float v = x[i];\n"
    "        n++;\n"
    "        float d = v - mean;\n"
    "        mean += d / (float)n;\n"
    "        m2 += d * (v - mean);\n"
    "    }\n"
    "    smean[tid] = mean;\n"
    "    sm2[tid] = m2;\n"
    "    sn[tid] = n;\n"
    "    __syncthreads();\n"
    "    for (int stride = nt >> 1; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) {\n"
    "            int nb = sn[tid + stride];\n"
    "            if (nb > 0) {\n"
    "                int na = sn[tid];\n"
    "                float ma = smean[tid];\n"
    "                float mb = smean[tid + stride];\n"
    "                float delta = mb - ma;\n"
    "                int nc = na + nb;\n"
    "                float f = (float)nb / (float)nc;\n"
    "                smean[tid] = ma + delta * f;\n"
    "                sm2[tid] += sm2[tid + stride] + delta * delta * (float)na * f;\n"
    "                sn[tid] = nc;\n"
    "            }\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    float mu = smean[0];\n"
    "    float inv = rsqrtf(sm2[0] / (float)dim + eps);\n"
    "    float *y = dst + (size_t)tok * dim;\n"
    "    for (int i = tid; i < dim; i += nt)\n"
    "        y[i] = (x[i] - mu) * inv * w[i] + b[i];\n"
    "}\n"

    "__global__ void layernorm_welford_warp_f32(float *dst, const float *src,\n"
    "                                           const float *w, const float *b,\n"
    "                                           int dim, float eps) {\n"
    "    extern __shared__ unsigned char smem_raw[];\n"
    "    int *shared_n = (int *)smem_raw;\n"
    "    float *shared_mv = (float *)(shared_n + 32);\n"
    "    int tok = blockIdx.x;\n"
    "    int tid = threadIdx.x;\n"
    "    int lane = tid & 31;\n"
    "    int warp = tid >> 5;\n"
    "    int nwarps = blockDim.x >> 5;\n"
    "    const float *x = src + (size_t)tok * dim;\n"
    "    float avg = 0.0f;\n"
    "    float var_n = 0.0f;\n"
    "    int n = 0;\n"
    "    for (int i = tid; i < dim; i += blockDim.x) {\n"
    "        float v = x[i];\n"
    "        float d1 = v - avg;\n"
    "        n++;\n"
    "        avg += d1 / (float)n;\n"
    "        var_n += d1 * (v - avg);\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int off = 1; off < 32; off <<= 1) {\n"
    "        float o_avg = __shfl_xor_sync(0xffffffffu, avg, off, 32);\n"
    "        float o_var = __shfl_xor_sync(0xffffffffu, var_n, off, 32);\n"
    "        int o_n = __shfl_xor_sync(0xffffffffu, n, off, 32);\n"
    "        float factor = 1.0f / fmaxf(1.0f, (float)(n + o_n));\n"
    "        float delta = avg - o_avg;\n"
    "        var_n += o_var + delta * delta * (float)n * (float)o_n * factor;\n"
    "        avg = ((float)n * avg + (float)o_n * o_avg) * factor;\n"
    "        n += o_n;\n"
    "    }\n"
    "    if (lane == 0) {\n"
    "        shared_n[warp] = n;\n"
    "        shared_mv[2 * warp] = avg;\n"
    "        shared_mv[2 * warp + 1] = var_n;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    if (tid < 32) {\n"
    "        n = (tid < nwarps) ? shared_n[tid] : 0;\n"
    "        avg = (tid < nwarps) ? shared_mv[2 * tid] : 0.0f;\n"
    "        var_n = (tid < nwarps) ? shared_mv[2 * tid + 1] : 0.0f;\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int off = 1; off < 32; off <<= 1) {\n"
    "        float o_avg = __shfl_xor_sync(0xffffffffu, avg, off, 32);\n"
    "        float o_var = __shfl_xor_sync(0xffffffffu, var_n, off, 32);\n"
    "        int o_n = __shfl_xor_sync(0xffffffffu, n, off, 32);\n"
    "        float factor = 1.0f / fmaxf(1.0f, (float)(n + o_n));\n"
    "        float delta = avg - o_avg;\n"
    "        var_n += o_var + delta * delta * (float)n * (float)o_n * factor;\n"
    "        avg = ((float)n * avg + (float)o_n * o_avg) * factor;\n"
    "        n += o_n;\n"
    "    }\n"
    "    if (tid == 0) {\n"
    "        shared_mv[0] = avg;\n"
    "        shared_mv[1] = rsqrtf(var_n / (float)dim + eps);\n"
    "    }\n"
    "    __syncthreads();\n"
    "    float mu = shared_mv[0];\n"
    "    float inv = shared_mv[1];\n"
    "    float *y = dst + (size_t)tok * dim;\n"
    "    for (int i = tid; i < dim; i += blockDim.x)\n"
    "        y[i] = (x[i] - mu) * inv * w[i] + b[i];\n"
    "}\n"

    "__global__ void layernorm_sqrtdiv_f32(float *dst, const float *src,\n"
    "                                      const float *w, const float *b,\n"
    "                                      int dim, float eps) {\n"
    "    extern __shared__ float sdata[];\n"
    "    int tok = blockIdx.x;\n"
    "    int tid = threadIdx.x;\n"
    "    int nt = blockDim.x;\n"
    "    const float *x = src + (size_t)tok * dim;\n"
    "    float *y = dst + (size_t)tok * dim;\n"
    "    float s = 0.0f;\n"
    "    for (int i = tid; i < dim; i += nt) s += x[i];\n"
    "    sdata[tid] = s;\n"
    "    __syncthreads();\n"
    "    for (int r = nt >> 1; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid + r]; __syncthreads(); }\n"
    "    float mean = sdata[0] / (float)dim;\n"
    "    __syncthreads();\n"
    "    s = 0.0f;\n"
    "    for (int i = tid; i < dim; i += nt) { float d = x[i] - mean; s += d * d; }\n"
    "    sdata[tid] = s;\n"
    "    __syncthreads();\n"
    "    for (int r = nt >> 1; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid + r]; __syncthreads(); }\n"
    "    float root, inv;\n"
    "    asm volatile (\"sqrt.rn.f32 %0, %1;\" : \"=f\"(root) : \"f\"(sdata[0] / (float)dim + eps));\n"
    "    asm volatile (\"div.rn.f32 %0, %1, %2;\" : \"=f\"(inv) : \"f\"(1.0f), \"f\"(root));\n"
    "    for (int i = tid; i < dim; i += nt)\n"
    "        y[i] = (x[i] - mean) * inv * w[i] + b[i];\n"
    "}\n"

    "struct __align__(16) sb_vec4_f32 { float val[4]; };\n"
    "struct sb_welford_ln_f32 { float mean; float sigma2; float count; };\n"
    "__device__ __forceinline__ sb_welford_ln_f32 sb_welford_online_f32(float val, sb_welford_ln_f32 a) {\n"
    "    float delta = val - a.mean;\n"
    "    float count = a.count + 1.0f;\n"
    "    float mean = a.mean + delta * (1.0f / count);\n"
    "    return { mean, a.sigma2 + delta * (val - mean), count };\n"
    "}\n"
    "__device__ __forceinline__ sb_welford_ln_f32 sb_welford_combine_f32(sb_welford_ln_f32 dataB, sb_welford_ln_f32 dataA) {\n"
    "    float delta = dataB.mean - dataA.mean;\n"
    "    float count = dataA.count + dataB.count;\n"
    "    if (count > 0.0f) {\n"
    "        float coef = 1.0f / count;\n"
    "        float nA = dataA.count * coef;\n"
    "        float nB = dataB.count * coef;\n"
    "        float mean = nA * dataA.mean + nB * dataB.mean;\n"
    "        float sigma2 = dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB;\n"
    "        return { mean, sigma2, count };\n"
    "    }\n"
    "    return { 0.0f, 0.0f, 0.0f };\n"
    "}\n"
    "__global__ void layernorm_torchvec_f32(float *dst, const float *src,\n"
    "                                      const float *w, const float *b,\n"
    "                                      int dim, float eps) {\n"
    "    extern __shared__ float buf[];\n"
    "    int row = blockIdx.x;\n"
    "    int thrx = threadIdx.x + threadIdx.y * blockDim.x;\n"
    "    int numx = blockDim.x * blockDim.y;\n"
    "    const sb_vec4_f32 *xv = (const sb_vec4_f32 *)(src + (size_t)row * dim);\n"
    "    int nvec = dim / 4;\n"
    "    sb_welford_ln_f32 wd = {0.0f, 0.0f, 0.0f};\n"
    "    for (int i = thrx; i < nvec; i += numx) {\n"
    "        sb_vec4_f32 data = xv[i];\n"
    "#pragma unroll\n"
    "        for (int ii = 0; ii < 4; ii++) wd = sb_welford_online_f32(data.val[ii], wd);\n"
    "    }\n"
    "    for (int off = 16; off > 0; off >>= 1) {\n"
    "        sb_welford_ln_f32 wdB = {\n"
    "            __shfl_down_sync(0xffffffffu, wd.mean, off, 32),\n"
    "            __shfl_down_sync(0xffffffffu, wd.sigma2, off, 32),\n"
    "            __shfl_down_sync(0xffffffffu, wd.count, off, 32) };\n"
    "        wd = sb_welford_combine_f32(wd, wdB);\n"
    "    }\n"
    "    if (blockDim.y > 1) {\n"
    "        float *meansigmabuf = buf;\n"
    "        float *countbuf = buf + blockDim.y;\n"
    "        for (int off = blockDim.y / 2; off > 0; off >>= 1) {\n"
    "            if (threadIdx.x == 0 && threadIdx.y >= off && threadIdx.y < 2 * off) {\n"
    "                int wrt_y = threadIdx.y - off;\n"
    "                meansigmabuf[2 * wrt_y] = wd.mean;\n"
    "                meansigmabuf[2 * wrt_y + 1] = wd.sigma2;\n"
    "                countbuf[wrt_y] = wd.count;\n"
    "            }\n"
    "            __syncthreads();\n"
    "            if (threadIdx.x == 0 && threadIdx.y < off) {\n"
    "                sb_welford_ln_f32 wdB = { meansigmabuf[2 * threadIdx.y], meansigmabuf[2 * threadIdx.y + 1], countbuf[threadIdx.y] };\n"
    "                wd = sb_welford_combine_f32(wd, wdB);\n"
    "            }\n"
    "            __syncthreads();\n"
    "        }\n"
    "        if (threadIdx.x == 0 && threadIdx.y == 0) {\n"
    "            meansigmabuf[0] = wd.mean;\n"
    "            meansigmabuf[1] = wd.sigma2 / (float)dim;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        wd.mean = meansigmabuf[0];\n"
    "        wd.sigma2 = meansigmabuf[1];\n"
    "    } else {\n"
    "        wd.mean = __shfl_sync(0xffffffffu, wd.mean, 0, 32);\n"
    "        wd.sigma2 = __shfl_sync(0xffffffffu, wd.sigma2, 0, 32) / (float)dim;\n"
    "    }\n"
    "    float rstd = rsqrtf(wd.sigma2 + eps);\n"
    "    const sb_vec4_f32 *wv = (const sb_vec4_f32 *)w;\n"
    "    const sb_vec4_f32 *bv = (const sb_vec4_f32 *)b;\n"
    "    sb_vec4_f32 *yv = (sb_vec4_f32 *)(dst + (size_t)row * dim);\n"
    "    for (int i = thrx; i < nvec; i += numx) {\n"
    "        sb_vec4_f32 data = xv[i];\n"
    "        sb_vec4_f32 gamma = wv[i];\n"
    "        sb_vec4_f32 beta = bv[i];\n"
    "        sb_vec4_f32 out;\n"
    "#pragma unroll\n"
    "        for (int ii = 0; ii < 4; ii++) out.val[ii] = gamma.val[ii] * (rstd * (data.val[ii] - wd.mean)) + beta.val[ii];\n"
    "        yv[i] = out;\n"
    "    }\n"
    "}\n"

    /* relu_inplace_f32: x = max(x, 0). 1D grid. */
    "__global__ void relu_inplace_f32(float *x, int n) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) { float v = x[i]; if (v < 0.0f) x[i] = 0.0f; }\n"
    "}\n"

    /* grid_sample_chw_f32 — bilinear, align_corners=False, zeros padding.
     *   src      (C, H, W) f32 — single channel-major image (CHW).
     *   gxy      (K, 2) f32 — sample points already in [-1, 1].
     *   invalid  (K) i32   — optional; if non-null and invalid[k], out[k]=0.
     *   out      (K, C) f32.
     * Grid: (K, ceil(C/256))   Block: (256,).
     */
    "__global__ void grid_sample_chw_f32(float *out, const float *src,\n"
    "                                    const float *gxy, const int *invalid,\n"
    "                                    int K, int C, int H, int W) {\n"
    "    int k = blockIdx.x;\n"
    "    int c = blockIdx.y * blockDim.x + threadIdx.x;\n"
    "    if (c >= C) return;\n"
    "    if (invalid && invalid[k]) { out[(size_t)k * C + c] = 0.0f; return; }\n"
    "    float gx = gxy[k*2 + 0];\n"
    "    float gy = gxy[k*2 + 1];\n"
    "    float xf = (gx + 1.0f) * (float)W * 0.5f - 0.5f;\n"
    "    float yf = (gy + 1.0f) * (float)H * 0.5f - 0.5f;\n"
    "    int x0 = (int)floorf(xf), x1 = x0 + 1;\n"
    "    int y0 = (int)floorf(yf), y1 = y0 + 1;\n"
    "    float ax = xf - (float)x0, ay = yf - (float)y0;\n"
    "    float w00 = (1.0f - ax) * (1.0f - ay);\n"
    "    float w10 = ax * (1.0f - ay);\n"
    "    float w01 = (1.0f - ax) * ay;\n"
    "    float w11 = ax * ay;\n"
    "    size_t HW = (size_t)H * (size_t)W;\n"
    "    size_t base = (size_t)c * HW;\n"
    "    float v = 0.0f;\n"
    "    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) v += w00 * src[base + (size_t)y0 * W + x0];\n"
    "    if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H) v += w10 * src[base + (size_t)y0 * W + x1];\n"
    "    if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H) v += w01 * src[base + (size_t)y1 * W + x0];\n"
    "    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) v += w11 * src[base + (size_t)y1 * W + x1];\n"
    "    out[(size_t)k * C + c] = v;\n"
    "}\n"

    /* kp_pelvis_norm_f32 — out[i] = kp3d[i] - 0.5*(kp3d[9*3+d] + kp3d[10*3+d]).
     * Length K*3. 1D grid. */
    "__global__ void kp_pelvis_norm_f32(float *out, const float *kp3d, int K) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (idx >= K * 3) return;\n"
    "    int d = idx % 3;\n"
    "    float pv = 0.5f * (kp3d[9*3 + d] + kp3d[10*3 + d]);\n"
    "    out[idx] = kp3d[idx] - pv;\n"
    "}\n"

    /* augment_overwrite_with_mask_f32 —
     *   augment[(start + k)*D + d] = invalid && invalid[k] ? 0 : posemb[k*D + d].
     * Grid: (K, ceil(D/256))   Block: (256,).
     * Pointer args first, scalar args last (matches __attribute__((packed))
     * struct layout — naturally-aligned 8-byte pointers grouped with no pads). */
    "__global__ void augment_overwrite_with_mask_f32(\n"
    "    float *augment, const float *posemb, const int *invalid,\n"
    "    int start_row, int K, int D) {\n"
    "    int k = blockIdx.x;\n"
    "    int d = blockIdx.y * blockDim.x + threadIdx.x;\n"
    "    if (d >= D) return;\n"
    "    float v = (invalid && invalid[k]) ? 0.0f : posemb[(size_t)k * D + d];\n"
    "    augment[(size_t)(start_row + k) * D + d] = v;\n"
    "    (void)K;\n"
    "}\n"

    /* ======================================================================
     * ViT-H/16 (vit_hmr_512_384) backbone kernels.
     *
     * Differences from the DINOv3-H+ path:
     *   - patch_embed conv has padding=2 (windows overlap by 4 px);
     *   - no CLS / register tokens — output is (n_patches, dim);
     *   - learned absolute pos_embed (1, n_patches+1, dim); slot 0 is added
     *     uniformly to every patch as a legacy CLS bias;
     *   - no RoPE, no LayerScale, GELU MLP (handled via the existing
     *     gelu_inplace_f32 + gemm_f32_bias / gemm_tiled_f16_f32 kernels);
     *   - head_dim = 80 (1280/16) — does not fit the FA_HEAD_DIM=64 slot.
     *
     * `patch_embed_pad2_f32`:
     *   Conv2d with kernel=stride=patch_size, padding=patch_pad. Output
     *   shape (n_patches, dim) — row-major, NO CLS/storage prefix.
     *   weight (dim, 3, ps, ps) f32; bias (dim,) f32.
     *
     *   Grid: (n_patches, 1, 1)   Block: (256, 1, 1)
     */
    "__global__ void patch_embed_pad2_f32(float *out, const float *img,\n"
    "                                     const float *w, const float *bias,\n"
    "                                     int gw, int gh, int dim, int ps,\n"
    "                                     int img_h, int img_w, int pad) {\n"
    "    int patch = blockIdx.x;\n"
    "    int tid   = threadIdx.x;\n"
    "    int py = patch / gw, px = patch % gw;\n"
    "    int ih_base = py * ps - pad;\n"
    "    int iw_base = px * ps - pad;\n"
    "    for (int co = tid; co < dim; co += blockDim.x) {\n"
    "        float sum = bias ? bias[co] : 0.0f;\n"
    "        for (int ci = 0; ci < 3; ci++) {\n"
    "            for (int kh = 0; kh < ps; kh++) {\n"
    "                int ih = ih_base + kh;\n"
    "                if (ih < 0 || ih >= img_h) continue;\n"
    "                for (int kw = 0; kw < ps; kw++) {\n"
    "                    int iw = iw_base + kw;\n"
    "                    if (iw < 0 || iw >= img_w) continue;\n"
    "                    sum += w[((co*3+ci)*ps+kh)*ps+kw]\n"
    "                         * img[(ci*img_h + ih)*img_w + iw];\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "        out[(size_t)patch * dim + co] = sum;\n"
    "    }\n"
    "    (void)gh;\n"
    "}\n"

    /* pos_embed_add_vith_f32 —
     *   hidden[t*dim + d] += pos_embed[(1+t)*dim + d] + pos_embed[0*dim + d];
     * Slot 0 is added uniformly to every patch (legacy CLS bias). pos_embed
     * shape (1, n_patches+1, dim) — flattened, slot 0 first.
     *
     * Grid: (ceil(n_patches*dim/256),)   Block: (256,).
     */
    "__global__ void pos_embed_add_vith_f32(float *hidden, const float *pos,\n"
    "                                       int n_patches, int dim) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = n_patches * dim;\n"
    "    if (idx >= total) return;\n"
    "    int t = idx / dim;\n"
    "    int d = idx - t * dim;\n"
    "    hidden[idx] += pos[(size_t)(1 + t) * dim + d] + pos[d];\n"
    "    (void)n_patches;\n"
    "}\n"

    /* qkv_split_f32 — split fused qkv (n, 3*dim) into three (n, dim)
     * row-major buffers Q, K, V. Used by ViT-H sdpa_f32 path which needs
     * separate Q/K/V buffers.
     *
     * Grid: (ceil(n*dim/256),)   Block: (256,).
     */
    "__global__ void qkv_split_f32(float *Q, float *K, float *V,\n"
    "                              const float *qkv, int n, int dim) {\n"
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

    /* qkv_transpose_heads_f32 — deinterleave Q/K/V from fused qkv into
     * per-head contiguous row-major buffers shaped (H, N, Dh).
     */
    "__global__ void qkv_transpose_heads_f32(float *Q_t, float *K_t, float *V_t,\n"
    "                                        const float *qkv,\n"
    "                                        int n_tok, int dim,\n"
    "                                        int n_heads, int head_dim,\n"
    "                                        float qk_scale) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = n_tok * dim;\n"
    "    if (idx >= total) return;\n"
    "    int tok = idx / dim;\n"
    "    int hd_idx = idx - tok * dim;\n"
    "    int h = hd_idx / head_dim;\n"
    "    int d = hd_idx - h * head_dim;\n"
    "    size_t dst = ((size_t)h * n_tok + tok) * head_dim + d;\n"
    "    size_t row3 = (size_t)tok * (size_t)(3 * dim);\n"
    "    Q_t[dst] = qkv[row3 + (size_t)hd_idx] * qk_scale;\n"
    "    K_t[dst] = qkv[row3 + (size_t)dim + (size_t)hd_idx] * qk_scale;\n"
    "    V_t[dst] = qkv[row3 + (size_t)(2 * dim) + (size_t)hd_idx];\n"
    "}\n"

    "__global__ void heads_to_interleaved_f32(float *out, const float *heads_buf,\n"
    "                                         int n_tok, int dim,\n"
    "                                         int n_heads, int head_dim) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = n_tok * dim;\n"
    "    if (idx >= total) return;\n"
    "    int tok = idx / dim;\n"
    "    int hd_idx = idx - tok * dim;\n"
    "    int h = hd_idx / head_dim;\n"
    "    int d = hd_idx - h * head_dim;\n"
    "    out[idx] = heads_buf[((size_t)h * n_tok + tok) * head_dim + d];\n"
    "    (void)n_heads;\n"
    "}\n"

    "__global__ void attn_prob_v_serial_f32(float *out, const float *prob,\n"
    "                                       const float *V_t,\n"
    "                                       int n_tok, int dim,\n"
    "                                       int n_heads, int head_dim) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = n_tok * dim;\n"
    "    if (idx >= total) return;\n"
    "    int tok = idx / dim;\n"
    "    int hd_idx = idx - tok * dim;\n"
    "    int h = hd_idx / head_dim;\n"
    "    int d = hd_idx - h * head_dim;\n"
    "    const float *p = prob + ((size_t)h * n_tok + tok) * n_tok;\n"
    "    const float *v = V_t + (size_t)h * n_tok * head_dim + d;\n"
    "    float acc = 0.0f;\n"
    "    for (int k = 0; k < n_tok; k++) acc += p[k] * v[(size_t)k * head_dim];\n"
    "    out[idx] = acc;\n"
    "    (void)n_heads;\n"
    "}\n"

    "__global__ void scale_rows_f32(float *x, int n, float scale) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i < n) x[i] *= scale;\n"
    "}\n"

    /* scale_softmax_rows_f32 — in-place softmax over each row of x.
     * Grid: (rows,) blockDim=(256,). x is row-major (rows, cols).
     */
    "__global__ void scale_softmax_rows_f32(float *x, int rows, int cols,\n"
    "                                       float scale) {\n"
    "    extern __shared__ float red[];\n"
    "    int row = blockIdx.x;\n"
    "    int tid = threadIdx.x;\n"
    "    int nt = blockDim.x;\n"
    "    if (row >= rows) return;\n"
    "    float *r = x + (size_t)row * cols;\n"
    "    float m = -1e30f;\n"
    "    for (int c = tid; c < cols; c += nt) if (r[c] > m) m = r[c];\n"
    "    red[tid] = m;\n"
    "    __syncthreads();\n"
    "    for (int s = nt >> 1; s > 0; s >>= 1) {\n"
    "        if (tid < s) { float v = red[tid + s]; if (v > red[tid]) red[tid] = v; }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    m = red[0] * scale;\n"
    "    float sum = 0.0f;\n"
    "    for (int c = tid; c < cols; c += nt) {\n"
    "        float e = expf(r[c] * scale - m);\n"
    "        r[c] = e;\n"
    "        sum += e;\n"
    "    }\n"
    "    red[tid] = sum;\n"
    "    __syncthreads();\n"
    "    for (int s = nt >> 1; s > 0; s >>= 1) {\n"
    "        if (tid < s) red[tid] += red[tid + s];\n"
    "        __syncthreads();\n"
    "    }\n"
    "    float inv = 1.0f / red[0];\n"
    "    for (int c = tid; c < cols; c += nt) r[c] *= inv;\n"
    "    (void)rows;\n"
    "}\n"

    /* softmax_warp_1024_rows_f32 — PyTorch-style persistent softmax for
     * rows with <= 1024 columns. Uses one warp per row and the same lane
     * traversal/reduction shape as ATen's softmax_warp_forward for f32.
     */
    "__global__ void softmax_warp_1024_rows_f32(float *x, int rows, int cols) {\n"
    "    int lane = threadIdx.x;\n"
    "    int warp_y = threadIdx.y;\n"
    "    int warps_per_block = blockDim.y;\n"
    "    int row = blockIdx.x * warps_per_block + warp_y;\n"
    "    if (row >= rows) return;\n"
    "    float *r = x + (size_t)row * cols;\n"
    "    float elem[32];\n"
    "    float m = -3.4028234663852886e38f;\n"
    "#pragma unroll\n"
    "    for (int it = 0; it < 32; it++) {\n"
    "        int c = lane + it * 32;\n"
    "        float v = (c < cols) ? r[c] : -3.4028234663852886e38f;\n"
    "        elem[it] = v;\n"
    "        m = m > v ? m : v;\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int off = 16; off > 0; off >>= 1) {\n"
    "        float b = __shfl_xor_sync(0xffffffffu, m, off, 32);\n"
    "        m = m > b ? m : b;\n"
    "    }\n"
    "    float sum = 0.0f;\n"
    "#pragma unroll\n"
    "    for (int it = 0; it < 32; it++) {\n"
    "        int c = lane + it * 32;\n"
    "        float e = (c < cols) ? expf(elem[it] - m) : 0.0f;\n"
    "        elem[it] = e;\n"
    "        sum += e;\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int off = 16; off > 0; off >>= 1)\n"
    "        sum += __shfl_xor_sync(0xffffffffu, sum, off, 32);\n"
    "#pragma unroll\n"
    "    for (int it = 0; it < 32; it++) {\n"
    "        int c = lane + it * 32;\n"
    "        if (c < cols) r[c] = elem[it] / sum;\n"
    "    }\n"
    "}\n"

    /* sdpa_qkv_t_f32 — strict DINOv3 attention path.
     *
     * Q is read from fused qkv (N, 3*D); K/V are read from the per-head
     * transposed buffers already produced for flash_attn_tiled_f32. This
     * materializes the full score row in shared memory and applies a standard
     * max/exp/sum softmax, closer to torch SDPA math than online tiled FA.
     *
     * Grid: (N, H)   Block: (256,)
     * Shmem: (blockDim.x + N) * sizeof(float).
     */
    "__global__ void sdpa_qkv_t_f32(float *out, const float *qkv,\n"
    "                               const float *K_t, const float *V_t,\n"
    "                               int N, int D, int H, int Dh,\n"
    "                               float scale) {\n"
    "    extern __shared__ float smem[];\n"
    "    int nq = blockIdx.x;\n"
    "    int h = blockIdx.y;\n"
    "    int tid = threadIdx.x;\n"
    "    int nt = blockDim.x;\n"
    "    float *red = smem;\n"
    "    float *scores = smem + nt;\n"
    "    const float *qv = qkv + (size_t)nq * (3 * D) + (size_t)h * Dh;\n"
    "    const float *kh = K_t + (size_t)h * N * Dh;\n"
    "    const float *vh = V_t + (size_t)h * N * Dh;\n"
    "    for (int nk = tid; nk < N; nk += nt) {\n"
    "        const float *kv = kh + (size_t)nk * Dh;\n"
    "        float s = 0.0f;\n"
    "        for (int d = 0; d < Dh; d++) s += qv[d] * kv[d];\n"
    "        scores[nk] = s * scale;\n"
    "    }\n"
    "    __syncthreads();\n"
    "    float lmax = -1e30f;\n"
    "    for (int nk = tid; nk < N; nk += nt)\n"
    "        if (scores[nk] > lmax) lmax = scores[nk];\n"
    "    red[tid] = lmax;\n"
    "    __syncthreads();\n"
    "    for (int r = nt >> 1; r > 0; r >>= 1) {\n"
    "        if (tid < r) { float b = red[tid + r]; if (b > red[tid]) red[tid] = b; }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    float m = red[0];\n"
    "    __syncthreads();\n"
    "    for (int nk = tid; nk < N; nk += nt) scores[nk] = expf(scores[nk] - m);\n"
    "    __syncthreads();\n"
    "    float lsum = 0.0f;\n"
    "    for (int nk = tid; nk < N; nk += nt) lsum += scores[nk];\n"
    "    red[tid] = lsum;\n"
    "    __syncthreads();\n"
    "    for (int r = nt >> 1; r > 0; r >>= 1) {\n"
    "        if (tid < r) red[tid] += red[tid + r];\n"
    "        __syncthreads();\n"
    "    }\n"
    "    float inv = 1.0f / red[0];\n"
    "    for (int d = tid; d < Dh; d += nt) {\n"
    "        float acc = 0.0f;\n"
    "        for (int nk = 0; nk < N; nk++)\n"
    "            acc += scores[nk] * vh[(size_t)nk * Dh + d];\n"
    "        out[(size_t)nq * D + (size_t)h * Dh + d] = acc * inv;\n"
    "    }\n"
    "}\n"

    /* mhr_blend_combine_f32 — MHR-on-GPU helper.
     *
     * Computes  out[i] = (base ? base[i] : 0) + sum_n coeffs[n] * vectors[n*V_d + i]
     * where i ∈ [0, V_d). Used by:
     *   - blend_shape:      base = base_shape (V*3,), vectors = (45, V*3)
     *   - face_expressions: base = NULL,             vectors = (72, V*3)
     * One thread per output. K (== N_basis) is small (45 / 72) so a serial
     * inner loop is fine; we spend most of the cost crossing global memory
     * for the (N_basis, V*3) matrix once per launch.
     *
     * Grid: (ceil(V_d / 256),)   Block: (256,).
     */
    "__global__ void mhr_blend_combine_f32(float *out, const float *coeffs,\n"
    "                                      const float *vectors,\n"
    "                                      const float *base,\n"
    "                                      int N_basis, int V_d) {\n"
    "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (i >= V_d) return;\n"
    "    float acc = base ? base[i] : 0.0f;\n"
    "    for (int n = 0; n < N_basis; n++)\n"
    "        acc += coeffs[n] * vectors[(size_t)n * V_d + i];\n"
    "    out[i] = acc;\n"
    "}\n"

    /* mhr_lbs_skin_f32 — MHR-on-GPU LBS / Stage 11 helper.
     *
     * Per-skin-entry scatter-add: out[v] += w * skel_transform_point(jstate[j], rv[v]).
     *   skel_state layout (8 floats): [tx, ty, tz,  qx, qy, qz, qw,  scale].
     *   transform_point(s, p) = s.t + rotate(normalize(s.q), s.scale * p).
     * Mirrors CPU s3dm_skel_transform_point + s3dm_quat_rot_vec_norm exactly
     * (FMA-equivalent ops; same float ordering) so per-vert numerics differ
     * only by atomicAdd reduction order. Output buffer must be zeroed before
     * launch.
     *
     *   jstate         : (J, 8) f32     joint state (already prepped on host)
     *   rest_verts     : (V, 3) f32
     *   skin_indices   : (K,)   i32     joint idx per skin entry
     *   vert_indices   : (K,)   i64     vert idx per skin entry (CPU type)
     *   skin_weights   : (K,)   f32
     *   out_verts      : (V, 3) f32     scatter-add target (zero on entry)
     *
     * Grid: (ceil(K / 256),)   Block: (256,).
     */
    "__global__ void mhr_lbs_skin_f32(float *out_verts,\n"
    "                                 const float *jstate,\n"
    "                                 const float *rest_verts,\n"
    "                                 const int *skin_indices,\n"
    "                                 const long long *vert_indices,\n"
    "                                 const float *skin_weights,\n"
    "                                 int K) {\n"
    "    int k = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (k >= K) return;\n"
    "    int j = skin_indices[k];\n"
    "    int v = (int)vert_indices[k];\n"
    "    float w = skin_weights[k];\n"
    "    const float *s = jstate + (size_t)j * 8;\n"
    "    const float *p = rest_verts + (size_t)v * 3;\n"
    "    /* normalize(q) */\n"
    "    float qx = s[3], qy = s[4], qz = s[5], qw = s[6];\n"
    "    float n = sqrtf(qx*qx + qy*qy + qz*qz + qw*qw);\n"
    "    if (n < 1e-12f) n = 1e-12f;\n"
    "    float inv = 1.0f / n;\n"
    "    qx *= inv; qy *= inv; qz *= inv; qw *= inv;\n"
    "    /* sp = scale * p */\n"
    "    float sc = s[7];\n"
    "    float spx = sc * p[0], spy = sc * p[1], spz = sc * p[2];\n"
    "    /* rotate(q, sp): v + 2*(r*(axis x v) + axis x (axis x v)) */\n"
    "    float avx = qy*spz - qz*spy;\n"
    "    float avy = qz*spx - qx*spz;\n"
    "    float avz = qx*spy - qy*spx;\n"
    "    float aavx = qy*avz - qz*avy;\n"
    "    float aavy = qz*avx - qx*avz;\n"
    "    float aavz = qx*avy - qy*avx;\n"
    "    float rx = spx + 2.0f * (avx * qw + aavx);\n"
    "    float ry = spy + 2.0f * (avy * qw + aavy);\n"
    "    float rz = spz + 2.0f * (avz * qw + aavz);\n"
    "    float tx = s[0] + rx;\n"
    "    float ty = s[1] + ry;\n"
    "    float tz = s[2] + rz;\n"
    "    atomicAdd(out_verts + (size_t)v * 3 + 0, w * tx);\n"
    "    atomicAdd(out_verts + (size_t)v * 3 + 1, w * ty);\n"
    "    atomicAdd(out_verts + (size_t)v * 3 + 2, w * tz);\n"
    "}\n"

    /* mhr_matvec_f32 — row-parallel Y = W @ X for production pose-correctives.
     *
     * The generic gemm_f32_bias path assigns one thread to each output row,
     * which serializes all 3000 FMAs for the 55317-row MHR projection. This
     * kernel uses one block per row and reduces across D_in in shared memory.
     */
    "__global__ void mhr_matvec_f32(float *Y,\n"
    "                               const float *W,\n"
    "                               const float *X,\n"
    "                               int D_in, int D_out) {\n"
    "    int row = blockIdx.x;\n"
    "    if (row >= D_out) return;\n"
    "    int tid = threadIdx.x;\n"
    "    float acc = 0.0f;\n"
    "    const float *wr = W + (size_t)row * D_in;\n"
    "    for (int k = tid; k < D_in; k += blockDim.x) acc += wr[k] * X[k];\n"
    "    __shared__ float red[256];\n"
    "    red[tid] = acc;\n"
    "    __syncthreads();\n"
    "    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) red[tid] += red[tid + stride];\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (tid == 0) Y[row] = red[0];\n"
    "}\n"

    /* mhr_keypoints_from_mesh_f32 — Stage 12 keypoint regression.
     *
     * Computes the body path of sam3d_body_keypoints_from_mesh for the first
     * 70 rows of keypoint_mapping. Inputs are still in MHR centimeters; the
     * kernel scales to meters and applies the camera-frame y/z flip on output.
     */
    "__global__ void mhr_keypoints_from_mesh_f32(float *out_kp3d,\n"
    "                                           const float *verts_cm,\n"
    "                                           const float *global_skel_cm,\n"
    "                                           const float *keypoint_mapping,\n"
    "                                           int V, int J, int K) {\n"
    "    int idx = blockIdx.x;\n"
    "    if (idx >= K * 3) return;\n"
    "    int k = idx / 3;\n"
    "    int c = idx - k * 3;\n"
    "    int VJ = V + J;\n"
    "    const float *Wk = keypoint_mapping + (size_t)k * VJ;\n"
    "    double acc = 0.0;\n"
    "    for (int i = threadIdx.x; i < V; i += blockDim.x)\n"
    "        acc += (double)Wk[i] * (double)(verts_cm[(size_t)i * 3 + c] * 0.01f);\n"
    "    for (int j = threadIdx.x; j < J; j += blockDim.x)\n"
    "        acc += (double)Wk[V + j] * (double)(global_skel_cm[(size_t)j * 8 + c] * 0.01f);\n"
    "    __shared__ double red[256];\n"
    "    int tid = threadIdx.x;\n"
    "    red[tid] = acc;\n"
    "    __syncthreads();\n"
    "    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) red[tid] += red[tid + stride];\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (tid != 0) return;\n"
    "    float v = (float)red[0];\n"
    "    out_kp3d[idx] = (c == 0) ? v : -v;\n"
    "}\n"

    /* mhr_camera_project_f32 — camera_project + full_to_crop for K keypoints. */
    "__global__ void mhr_camera_project_f32(const float *kp3d,\n"
    "                                      float *kp2d_full,\n"
    "                                      float *kp2d_crop,\n"
    "                                      float *kp2d_depth,\n"
    "                                      float *pred_cam_t,\n"
    "                                      float pred0, float pred1, float pred2,\n"
    "                                      float bbox_scale, float bbox_cx, float bbox_cy,\n"
    "                                      float ori_w, float ori_h,\n"
    "                                      float img_w, float img_h,\n"
    "                                      float k00, float k01, float k02,\n"
    "                                      float k10, float k11, float k12,\n"
    "                                      float k02_center, float k12_center,\n"
    "                                      float a00, float a01, float a02,\n"
    "                                      float a10, float a11, float a12,\n"
    "                                      int use_intrin_center, int K) {\n"
    "    int k = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    float s = -pred0;\n"
    "    float tx = pred1;\n"
    "    float ty = -pred2;\n"
    "    float bs = bbox_scale * s + 1.0e-8f;\n"
    "    float tz = 2.0f * k00 / bs;\n"
    "    float cx, cy;\n"
    "    if (!use_intrin_center) {\n"
    "        cx = 2.0f * (bbox_cx - ori_w * 0.5f) / bs;\n"
    "        cy = 2.0f * (bbox_cy - ori_h * 0.5f) / bs;\n"
    "    } else {\n"
    "        cx = 2.0f * (bbox_cx - k02_center) / bs;\n"
    "        cy = 2.0f * (bbox_cy - k12_center) / bs;\n"
    "    }\n"
    "    float ct0 = tx + cx, ct1 = ty + cy, ct2 = tz;\n"
    "    if (k == 0 && pred_cam_t) { pred_cam_t[0] = ct0; pred_cam_t[1] = ct1; pred_cam_t[2] = ct2; }\n"
    "    if (k >= K) return;\n"
    "    float p0 = kp3d[(size_t)k * 3 + 0] + ct0;\n"
    "    float p1 = kp3d[(size_t)k * 3 + 1] + ct1;\n"
    "    float p2 = kp3d[(size_t)k * 3 + 2] + ct2;\n"
    "    if (kp2d_depth) kp2d_depth[k] = p2;\n"
    "    float invz = 1.0f / p2;\n"
    "    float xn = p0 * invz, yn = p1 * invz;\n"
    "    float u = k00 * xn + k01 * yn + k02;\n"
    "    float v = k10 * xn + k11 * yn + k12;\n"
    "    if (kp2d_full) { kp2d_full[(size_t)k * 2 + 0] = u; kp2d_full[(size_t)k * 2 + 1] = v; }\n"
    "    if (kp2d_crop) {\n"
    "        float ax = a00 * u + a01 * v + a02;\n"
    "        float ay = a10 * u + a11 * v + a12;\n"
    "        kp2d_crop[(size_t)k * 2 + 0] = ax / img_w - 0.5f;\n"
    "        kp2d_crop[(size_t)k * 2 + 1] = ay / img_h - 0.5f;\n"
    "    }\n"
    "}\n"

    /* dense_pe_tok_f32 — SAM random Fourier dense PE in token order (HW, C).
     *
     * Mirrors sam3d_body_get_dense_pe, but writes the layout consumed by the
     * decoder cross-attention directly so the GPU decoder path avoids a host
     * CHW build + transpose + upload.
     */
    "__global__ void dense_pe_tok_f32(float *out,\n"
    "                                 const float *G,\n"
    "                                 int H, int W, int npf,\n"
    "                                 int use_square_x) {\n"
    "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    int total = H * W * npf;\n"
    "    if (idx >= total) return;\n"
    "    int k = idx % npf;\n"
    "    int p = idx / npf;\n"
    "    int yi = p / W;\n"
    "    int xi = p - yi * W;\n"
    "    float yn = ((float)yi + 0.5f) / (float)H;\n"
    "    float ys = 2.0f * yn - 1.0f;\n"
    "    float x_offset = (use_square_x && W != H) ? (0.5f * (float)(H - W)) : 0.0f;\n"
    "    float xn = (use_square_x && W != H)\n"
    "        ? (((float)xi + x_offset + 0.5f) / (float)H)\n"
    "        : (((float)xi + 0.5f) / (float)W);\n"
    "    float xs = 2.0f * xn - 1.0f;\n"
    "    float v = (xs * G[k] + ys * G[npf + k]) * 6.2831853071795864769f;\n"
    "    size_t base = (size_t)p * (size_t)(npf * 2);\n"
    "    out[base + (size_t)k] = sinf(v);\n"
    "    out[base + (size_t)(npf + k)] = cosf(v);\n"
    "}\n"

    /* gemm_tiled_f32_f32
     *
     * Same launch contract as gemm_tiled_f16_f32, but W is F32. Used by the
     * optional DINOv3 fp32 precision mode while the default fast path keeps
     * F16 block weights.
     */
    "__global__ void gemm_tiled_f32_f32(float *Y, const float *W, const float *X,\n"
    "                                    const float *bias,\n"
    "                                    int n_out, int n_in, int n_tok) {\n"
    "    __shared__ float smA[16][16];\n"
    "    __shared__ float smB[16][16];\n"
    "    int tx = threadIdx.x, ty = threadIdx.y;\n"
    "    int tok_base = blockIdx.y * 16;\n"
    "    int out_base = blockIdx.x * 64;\n"
    "    int row = tok_base + ty;\n"
    "    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;\n"
    "    for (int k = 0; k < n_in; k += 16) {\n"
    "        smA[ty][tx] = (tok_base + ty < n_tok && k + tx < n_in)\n"
    "                      ? X[(tok_base + ty) * n_in + k + tx] : 0.f;\n"
    "        __syncthreads();\n"
    "        { int w_out = out_base + tx;\n"
    "          smB[ty][tx] = (w_out < n_out && k + ty < n_in)\n"
    "                        ? W[(size_t)w_out * n_in + k + ty] : 0.f; }\n"
    "        __syncthreads();\n"
    "        for (int i = 0; i < 16; i++) acc0 += smA[ty][i] * smB[i][tx];\n"
    "        __syncthreads();\n"
    "        { int w_out = out_base + 16 + tx;\n"
    "          smB[ty][tx] = (w_out < n_out && k + ty < n_in)\n"
    "                        ? W[(size_t)w_out * n_in + k + ty] : 0.f; }\n"
    "        __syncthreads();\n"
    "        for (int i = 0; i < 16; i++) acc1 += smA[ty][i] * smB[i][tx];\n"
    "        __syncthreads();\n"
    "        { int w_out = out_base + 32 + tx;\n"
    "          smB[ty][tx] = (w_out < n_out && k + ty < n_in)\n"
    "                        ? W[(size_t)w_out * n_in + k + ty] : 0.f; }\n"
    "        __syncthreads();\n"
    "        for (int i = 0; i < 16; i++) acc2 += smA[ty][i] * smB[i][tx];\n"
    "        __syncthreads();\n"
    "        { int w_out = out_base + 48 + tx;\n"
    "          smB[ty][tx] = (w_out < n_out && k + ty < n_in)\n"
    "                        ? W[(size_t)w_out * n_in + k + ty] : 0.f; }\n"
    "        __syncthreads();\n"
    "        for (int i = 0; i < 16; i++) acc3 += smA[ty][i] * smB[i][tx];\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (row < n_tok) {\n"
    "        if (out_base +      tx < n_out) Y[row * n_out + out_base +      tx] = acc0 + (bias ? bias[out_base +      tx] : 0.f);\n"
    "        if (out_base + 16 + tx < n_out) Y[row * n_out + out_base + 16 + tx] = acc1 + (bias ? bias[out_base + 16 + tx] : 0.f);\n"
    "        if (out_base + 32 + tx < n_out) Y[row * n_out + out_base + 32 + tx] = acc2 + (bias ? bias[out_base + 32 + tx] : 0.f);\n"
    "        if (out_base + 48 + tx < n_out) Y[row * n_out + out_base + 48 + tx] = acc3 + (bias ? bias[out_base + 48 + tx] : 0.f);\n"
    "    }\n"
    "}\n"

    "}  /* close extern \"C\" from cuda_kernels_common_src */\n";

#endif /* CUDA_SAM3D_BODY_KERNELS_H_ */
