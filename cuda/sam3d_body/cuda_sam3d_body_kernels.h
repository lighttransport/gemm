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
    "                                  int img_w, int base_tok) {\n"
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
    "                         * img[ci * img_w * img_w + (py*ps+kh) * img_w + (px*ps+kw)];\n"
    "        out[tok * dim + co] = sum;\n"
    "    }\n"
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
     * This is used for the default bf16 reference path.
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

    /* mhr_blend_combine_f32 — speculative MHR-on-GPU (Step 7).
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

    /* mhr_lbs_skin_f32 — speculative MHR-on-GPU (LBS / Stage 11).
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

    "}  /* close extern \"C\" from cuda_kernels_common_src */\n";

#endif /* CUDA_SAM3D_BODY_KERNELS_H_ */
