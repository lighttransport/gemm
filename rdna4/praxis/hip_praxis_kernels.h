/*
 * hip_praxis_kernels.h - HIPRTC kernel source of the praxis-1 RDNA4 (gfx1201, wave32, WMMA) runner.
 * Port of cuda/praxis/praxis_kernels.h (+ praxis_fp4_kernels.h): same kernel names / argument orders for the fp32 exact path; the fast path uses gfx12
 * v_wmma_f32_16x16x16 bf16/f16 GEMMs (one templated tiled kernel for dense bf16 weights, bf16 expert banks, NVFP4 W4A16 and FP8 W8A16 banks), a transposed-
 * formulation WMMA flash attention with the chunk-causal code mask / action-expert interval mask, and the device-side MoE dispatch.
 * (No '#' directives and no apostrophes inside: the source lives in a macro.)
 */
#ifndef HIP_PRAXIS_KERNELS_H
#define HIP_PRAXIS_KERNELS_H
#define PRK_STR(...) #__VA_ARGS__
static const char *praxis_kernel_src = PRK_STR(
extern "C" {

__device__ __forceinline__ float pr_ldw(const void *W, long long i, int dt) {
    if (dt == 0) return ((const float *)W)[i];
    unsigned short h = ((const unsigned short *)W)[i];
    if (dt == 1) return __uint_as_float(((unsigned)h) << 16);
    return (float)__builtin_bit_cast(_Float16, h);
}

__device__ __forceinline__ float pr_block_sum(float v, float *sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
    if (w == 0) {
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0) sh[0] = t;
    }
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ float pr_block_max(float v, float *sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : -3.0e38f;
    if (w == 0) {
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_down_sync(0xffffffffu, t, o));
        if (lane == 0) sh[0] = t;
    }
    __syncthreads();
    return sh[0];
}

/* Y[m][n] (+)= sum_k X[row(m)][k] * W(n,k) + bias[n].
 * wl = 0: W is [N][K] (nn.Linear weight); wl = 1: W is [K][N] (x @ W).
 * seg != 0: grouped (MoE) mode, expert z = blockIdx.z owns rows seg[z]..seg[z+1]
 * of the sorted row space and the weights W + z * wstride (elements).
 * rowmap != 0 maps a sorted row to the input row of X. */
__global__ void pr_gemm(const float *X, long long ldx, const int *rowmap, const void *W, int wdt,
                        int wl, long long wstride, const float *bias, float *Y, long long ldy,
                        int M, int N, int K, int accum, const int *seg)
{
    __shared__ float Xs[16][33];
    __shared__ float Ws[16][33];
    int m0, mend;
    long long woff = 0;
    if (seg) {
        int e = blockIdx.z;
        m0 = seg[e] + blockIdx.y * 32;
        mend = seg[e + 1];
        woff = (long long)e * wstride;
    } else {
        m0 = blockIdx.y * 32;
        mend = M;
    }
    if (m0 >= mend) return;
    int n0 = blockIdx.x * 32;
    int tid = threadIdx.x;
    int tx = tid & 15, ty = tid >> 4;
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 256;
            int r = idx >> 4, kk = idx & 15;
            int gm = m0 + r;
            float v = 0.f;
            if (gm < mend && k0 + kk < K) {
                long long xr = rowmap ? rowmap[gm] : gm;
                v = X[xr * ldx + k0 + kk];
            }
            Xs[kk][r] = v;
            int n, kw;
            if (wl == 0) { n = idx >> 4; kw = idx & 15; } else { n = idx & 31; kw = idx >> 5; }
            float wv = 0.f;
            if (n0 + n < N && k0 + kw < K) {
                long long wi = (wl == 0) ? ((long long)(n0 + n) * K + k0 + kw)
                                         : ((long long)(k0 + kw) * N + n0 + n);
                wv = pr_ldw(W, woff + wi, wdt);
            }
            Ws[kw][n] = wv;
        }
        __syncthreads();
        for (int kk = 0; kk < 16; kk++) {
            float x0 = Xs[kk][ty * 2], x1 = Xs[kk][ty * 2 + 1];
            float w0 = Ws[kk][tx * 2], w1 = Ws[kk][tx * 2 + 1];
            a00 = fmaf(x0, w0, a00); a01 = fmaf(x0, w1, a01);
            a10 = fmaf(x1, w0, a10); a11 = fmaf(x1, w1, a11);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            int gm = m0 + ty * 2 + i, gn = n0 + tx * 2 + j;
            if (gm < mend && gn < N) {
                float v = (i == 0 ? (j == 0 ? a00 : a01) : (j == 0 ? a10 : a11));
                if (bias) v += bias[gn];
                if (accum) Y[(long long)gm * ldy + gn] += v; else Y[(long long)gm * ldy + gn] = v;
            }
        }
}

/* Row r = token * heads + head of x [tokens][ldx] with C channels per head. */
__global__ void pr_rmsnorm(const float *x, long long ldx, float *y, long long ldy, const float *w,
                           int plus1, float eps, float oscale, int C, int heads)
{
    __shared__ float sh[32];
    long long r = blockIdx.x;
    long long t = r / heads, h = r % heads;
    const float *xr = x + t * ldx + h * C;
    float *yr = y + t * ldy + h * C;
    float ss = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { float v = xr[c]; ss += v * v; }
    ss = pr_block_sum(ss, sh);
    float inv = 1.0f / sqrtf(ss / (float)C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float wv = w ? (plus1 ? 1.0f + w[c] : w[c]) : 1.0f;
        yr[c] = xr[c] * inv * wv * oscale;
    }
}

/* out = (1 + scale) * rmsnorm(x) (expert AdaLN, no weight, eps 1e-4). */
__global__ void pr_adaln(const float *x, const float *scale, float *y, float eps, int C)
{
    __shared__ float sh[32];
    long long r = blockIdx.x;
    const float *xr = x + r * C, *sr = scale + r * C;
    float ss = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { float v = xr[c]; ss += v * v; }
    ss = pr_block_sum(ss, sh);
    float inv = 1.0f / sqrtf(ss / (float)C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) y[r * C + c] = (1.0f + sr[c]) * (xr[c] * inv);
}

/* LayerNorm without affine. */
__global__ void pr_layernorm(const float *x, long long ldx, float *y, float eps, int C)
{
    __shared__ float sh[32];
    long long r = blockIdx.x;
    const float *xr = x + r * ldx;
    float s = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += xr[c];
    float mean = pr_block_sum(s, sh) / (float)C;
    float v = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { float d = xr[c] - mean; v += d * d; }
    float var = pr_block_sum(v, sh) / (float)C;
    float inv = 1.0f / sqrtf(var + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) y[r * C + c] = (xr[c] - mean) * inv;
}

/* y = x * (1 + mod[mrow[t]][C + c]) + mod[mrow[t]][c]  (in place allowed). */
__global__ void pr_modulate(const float *x, float *y, const float *mod, long long ldmod,
                            const int *mrow, int C, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * C) return;
    long long t = i / C;
    int c = (int)(i - t * C);
    const float *m = mod + (long long)mrow[t] * ldmod;
    y[i] = x[i] * (1.0f + m[C + c]) + m[c];
}

/* out[i][d] = cos((t[i] * f[d] + p[d]) * 2 * pi) * 0.5 */
__global__ void pr_fourier(const float *t, int n, const float *f, const float *p, int D, float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * D) return;
    int r = i / D, d = i - r * D;
    float x = t[r] * f[d] + p[d];
    x = x * 2.0f;
    x = x * 3.14159274f;
    out[i] = cosf(x) * 0.5f;
}

/* [cos(t * f[d]), sin(t * f[d])], f[d] = freq table (expert timestep embedder). */
__global__ void pr_sincos(const float *t, int n, const float *f, int half, float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * half) return;
    int r = i / half, d = i - r * half;
    float a = t[r] * f[d];
    out[(long long)r * 2 * half + d] = cosf(a);
    out[(long long)r * 2 * half + half + d] = sinf(a);
}

__global__ void pr_swiglu(const float *a, long long lda, const float *b, long long ldb, float *o,
                          long long ldo, int H, long long rows, int mode)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * H) return;
    long long r = i / H;
    int c = (int)(i - r * H);
    float x = a[r * lda + c], g = b[r * ldb + c];
    float act;
    if (mode == 0) act = x / (1.0f + expf(-x));
    else act = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    o[r * ldo + c] = act * g;
}

/* mode 0: gelu (erf), 1: silu */
__global__ void pr_act(const float *x, float *y, long long n, int mode)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (mode == 0) y[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
    else y[i] = v / (1.0f + expf(-v));
}

__global__ void pr_add(float *y, const float *x, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

/* y[r][c] += v[c] */
__global__ void pr_add_row(float *y, const float *v, long long rows, int C)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * C) return;
    y[i] += v[i % C];
}

/* x[(k * nf + f)][c] = ft[f][c] + ce[k][c] */
__global__ void pr_future(float *x, const float *ft, const float *ce, int K, int nf, int C)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)K * nf * C) return;
    int c = (int)(i % C);
    long long r = i / C;
    int f = (int)(r % nf), k = (int)(r / nf);
    x[i] = ft[f * C + c] + ce[k * C + c];
}

__global__ void pr_copy_rows(const float *src, long long lds, float *dst, long long ldd, long long rows, int C)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * C) return;
    long long r = i / C;
    int c = (int)(i - r * C);
    dst[r * ldd + c] = src[r * lds + c];
}

/* src [C][N] (channel-first) -> dst [N][C] */
__global__ void pr_chw_to_tok(const float *src, float *dst, long long ldd, int C, long long N)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * C) return;
    long long n = i / C;
    int c = (int)(i - n * C);
    dst[n * ldd + c] = src[(long long)c * N + n];
}

/* src [c' * ph * pw][T][h][w] -> dst tokens [(t, y, x)][c'], H = h * ph, W = w * pw */
__global__ void pr_unpatch_tok(const float *src, float *dst, int Cp, int ph, int pw, int T, int h, int w)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int H = h * ph, W = w * pw;
    long long N = (long long)T * H * W;
    if (i >= N * Cp) return;
    long long n = i / Cp;
    int c = (int)(i - n * Cp);
    int t = (int)(n / ((long long)H * W));
    int rem = (int)(n - (long long)t * H * W);
    int y = rem / W, x = rem - (rem / W) * W;
    long long chan = ((long long)c * ph + (y % ph)) * pw + (x % pw);
    dst[i] = src[((chan * T + t) * h + y / ph) * w + x / pw];
}

/* tokens [(t, y, x)][c'] -> out [c' * ph * pw][T][h][w] */
__global__ void pr_patch_out(const float *tok, float *out, int Cp, int ph, int pw, int T, int h, int w)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)Cp * ph * pw * T * h * w;
    if (i >= total) return;
    int ww = (int)(i % w);
    long long r = i / w;
    int hh = (int)(r % h);
    r /= h;
    int t = (int)(r % T);
    long long chan = r / T;
    int j = (int)(chan % pw);
    long long r2 = chan / pw;
    int ii = (int)(r2 % ph);
    int c = (int)(r2 / ph);
    int H = h * ph, W = w * pw;
    long long n = ((long long)t * H + hh * ph + ii) * W + ww * pw + j;
    out[i] = tok[n * Cp + c];
}

/* pairs (j, j + half) rotate; q [tokens][ld] holds heads * hd channels. */
__global__ void pr_rope(float *q, long long ld, int heads, int hd, const float *cs, const float *sn, long long n_tok)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int half = hd / 2;
    if (i >= n_tok * heads * half) return;
    int j = (int)(i % half);
    long long r = i / half;
    int h = (int)(r % heads);
    long long t = r / heads;
    float c = cs[t * half + j], s = sn[t * half + j];
    float *p = q + t * ld + (long long)h * hd;
    float x0 = p[j], x1 = p[j + half];
    p[j] = x0 * c - x1 * s;
    p[j + half] = x1 * c + x0 * s;
}

/* Dense-masked GQA attention, one block per (query, head). mask_mode 0: bytes
 * mask[q][kv]; 1: teacher-forcing token codes. gate (optional, [Q][H] logits)
 * multiplies the head output by sigmoid(gate). Output [Q][H * hd]. */
__global__ void pr_attn(const float *Q, long long ldq, const float *K, long long ldk, const float *V,
                        long long ldv, float *O, long long ldo, const float *gate, int H, int KVH, int hd,
                        int nq, int nk, float scale, int mask_mode, const unsigned char *mask,
                        const int *qcode, const int *kcode)
{
    extern __shared__ float sm[];
    float *qs = sm;
    float *sc = sm + hd;
    float *red = sm + hd + nk;
    int qi = blockIdx.x, h = blockIdx.y;
    int kvh = h / (H / KVH);
    const float *qp = Q + (long long)qi * ldq + (long long)h * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = qp[d];
    __syncthreads();
    float lmax = -3.0e38f;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        int ok;
        if (mask_mode == 0) ok = mask[(long long)qi * nk + j] != 0;
        else {
            int q = qcode[qi], k = kcode[j];
            int hi = (q & 1) ? q : q - 3;
            ok = (k == q) || (k == 0) || (((k & 1) == 1) && (k <= hi));
        }
        float s = -3.0e38f;
        if (ok) {
            const float *kp = K + (long long)j * ldk + (long long)kvh * hd;
            float a = 0.f;
            for (int d = 0; d < hd; d++) a = fmaf(qs[d], kp[d], a);
            s = a * scale;
        }
        sc[j] = s;
        lmax = fmaxf(lmax, s);
    }
    lmax = pr_block_max(lmax, red);
    float lsum = 0.f;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        float p = (sc[j] > -1.0e38f) ? expf(sc[j] - lmax) : 0.f;
        sc[j] = p;
        lsum += p;
    }
    lsum = pr_block_sum(lsum, red);
    float g = 1.0f;
    if (gate) { float gl = gate[(long long)qi * H + h]; g = 1.0f / (1.0f + expf(-gl)); }
    float norm = g / lsum;
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float a = 0.f;
        for (int j = 0; j < nk; j++) {
            float p = sc[j];
            if (p != 0.f) a = fmaf(p, V[(long long)j * ldv + (long long)kvh * hd + d], a);
        }
        O[(long long)qi * ldo + (long long)h * hd + d] = a * norm;
    }
}

/* One thread per token: sigmoid scores, selection-only bias, top-k, normalised gates. */
__global__ void pr_topk(const float *logits, const float *bias, int *idx, float *gate, int T, int E, int k)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    unsigned char used[512];
    float sc[512];
    for (int e = 0; e < E; e++) { sc[e] = 1.0f / (1.0f + expf(-logits[(long long)t * E + e])); used[e] = 0; }
    float sum = 0.f;
    float pick[32];
    for (int j = 0; j < k; j++) {
        int best = -1;
        float bv = -3.0e38f;
        for (int e = 0; e < E; e++) {
            float b = sc[e] + bias[e];
            if (!used[e] && b > bv) { bv = b; best = e; }
        }
        if (best < 0) best = 0;   /* NaN scores: never index out of range */
        used[best] = 1;
        idx[t * k + j] = best;
        pick[j] = sc[best];
        sum += sc[best];
    }
    for (int j = 0; j < k; j++) gate[t * k + j] = pick[j] / sum;
}

/* out[t] += scale * sum_j gate[t][j] * y[tokrow[t][j]] */
__global__ void pr_combine(float *out, const float *y, const int *tokrow, const float *gate, float scale,
                           int C, long long T, int k)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    long long t = i / C;
    int c = (int)(i - t * C);
    float a = 0.f;
    for (int j = 0; j < k; j++) a += gate[t * k + j] * y[(long long)tokrow[t * k + j] * C + c];
    out[i] += scale * a;
}
}
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned short u16v8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16v8 __attribute__((ext_vector_type(8)));
typedef float f32v8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ u16 pr_f2bf(float f) {
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u16)0x7fc0;
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
__device__ __forceinline__ float pr_bf2f(u16 h) { return __uint_as_float(((u32)h) << 16); }
__device__ __forceinline__ u32 pr_pack2(float lo, float hi) { return ((u32)pr_f2bf(hi) << 16) | (u32)pr_f2bf(lo); }
__device__ __forceinline__ u16 pr_f2h(float f) {
    f = fminf(fmaxf(f, -65504.f), 65504.f);
    _Float16 h = (_Float16)f;
    return __builtin_bit_cast(u16, h);
}
__device__ __forceinline__ float pr_h2f(u16 h) { return (float)__builtin_bit_cast(_Float16, h); }
__device__ __forceinline__ u32 pr_pack2h(float lo, float hi) { return ((u32)pr_f2h(hi) << 16) | (u32)pr_f2h(lo); }
__device__ __forceinline__ float pr_ex2(float x) { return exp2f(x); }

extern "C" {
/* y[m][k] = bf16(x[m * ldx + k]), dense [M][K] output */
__global__ void pr_cvt_bf16(const float *x, long long ldx, u16 *y, int K, long long M)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * K) return;
    long long m = i / K;
    int k = (int)(i - m * K);
    y[i] = pr_f2bf(x[m * ldx + k]);
}

/* zero-centred / plain RMSNorm of one row with fp32 and/or bf16 outputs */
__global__ void pr_rmsnorm_bf(const float *x, long long ldx, float *y, u16 *yb, const float *w,
                              int plus1, float eps, float oscale, int C)
{
    __shared__ float sh[32];
    long long r = blockIdx.x;
    const float *xr = x + r * ldx;
    float ss = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { float v = xr[c]; ss += v * v; }
    ss = pr_block_sum(ss, sh);
    float inv = 1.0f / sqrtf(ss / (float)C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float wv = w ? (plus1 ? 1.0f + w[c] : w[c]) : 1.0f;
        float o = xr[c] * inv * wv * oscale;
        if (y) y[r * C + c] = o;
        if (yb) yb[r * C + c] = pr_f2bf(o);
    }
}

/* q/k preparation: partial-RoPE (pairs j, j + hd/2; table width hd/2) then per-head RMSNorm (+1 weight),
 * fp32 [tokens][ldx] -> bf16 [tokens][ldy]. One block per (token, head). */
__global__ void pr_qkprep(const float *x, long long ldx, u16 *y, long long ldy, const float *w,
                          const float *cs, const float *sn, float eps, int hd, int heads, long long tseq)
{
    __shared__ float sh[32];
    extern __shared__ float hv[];
    long long r = blockIdx.x;
    long long t = r / heads, h = r % heads;
    const float *p = x + t * ldx + h * hd;
    int half = hd >> 1;
    float ss = 0.f;
    long long tr = t % tseq;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float c = cs[tr * half + j], s = sn[tr * half + j];
        float x0 = p[j], x1 = p[j + half];
        float a = x0 * c - x1 * s, b = x1 * c + x0 * s;
        hv[j] = a; hv[j + half] = b;
        ss += a * a + b * b;
    }
    ss = pr_block_sum(ss, sh);
    float inv = 1.0f / sqrtf(ss / (float)hd + eps);
    u16 *o = y + t * ldy + h * hd;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        float wv = w ? 1.0f + w[j] : 1.0f;
        o[j] = pr_f2bf(hv[j] * inv * wv);
    }
}

/* Y[m][n] += bias[n] */
__global__ void pr_bias(float *Y, long long ldy, const float *bias, long long M, int N)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    long long m = i / N;
    int n = (int)(i - m * N);
    Y[m * ldy + n] += bias[n];
}

/* bf16 SwiGLU: o = silu(a) * b, fp32 math */
__global__ void pr_swiglu_bf(const u16 *a, long long lda, const u16 *b, long long ldb, u16 *o, long long ldo, int H, long long rows)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * H) return;
    long long r = i / H;
    int c = (int)(i - r * H);
    float x = pr_bf2f(a[r * lda + c]), g = pr_bf2f(b[r * ldb + c]);
    o[r * ldo + c] = pr_f2bf((x / (1.0f + expf(-x))) * g);
}
/* f16 SwiGLU (the W4A16 / W8A16 gate_up GEMM writes f16): o = silu(a) * b, f16 in and out */
__global__ void pr4_swiglu_f16(const u16 *a, long long lda, const u16 *b, long long ldb, u16 *o, long long ldo, int H, long long rows)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * H) return;
    long long r = i / H;
    int c = (int)(i - r * H);
    float x = pr_h2f(a[r * lda + c]), g = pr_h2f(b[r * ldb + c]);
    o[r * ldo + c] = pr_f2h((x / (1.0f + expf(-x))) * g);
}
/* fp32 rows -> f16 rows (saturating) */
__global__ void pr4_f32_to_f16(const float *x, long long ldx, u16 *y, long long ldy, long long rows, int cols)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    long long r = i / cols;
    int c = (int)(i - r * cols);
    y[r * ldy + c] = pr_f2h(x[r * ldx + c]);
}
/* tile prefix over experts: tpre[e] = sum_{i<e} ceil(n_i / bm) */
__global__ void pr4_tiles(const int *seg, int E, int bm, int *tpre)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int s = 0;
        for (int e = 0; e < E; e++) { tpre[e] = s; s += (seg[e + 1] - seg[e] + bm - 1) / bm; }
        tpre[E] = s;
    }
}

/* per K-tile code summary: [min, max, min odd code (1<<30 if none)] */
__global__ void pr_code_tiles(const int *code, int n, int bc, int *out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int nt = (n + bc - 1) / bc;
    if (t >= nt) return;
    int mn = 1 << 30, mx = -1, od = 1 << 30;
    for (int j = t * bc; j < n && j < (t + 1) * bc; j++) {
        int c = code[j];
        mn = min(mn, c); mx = max(mx, c);
        if ((c & 1) && c < od) od = c;
    }
    out[3 * t] = mn; out[3 * t + 1] = mx; out[3 * t + 2] = od;
}

}
__device__ __forceinline__ f32v8 pr_mma_bf(u16v8 a, u16v8 b, f32v8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c); }
__device__ __forceinline__ f32v8 pr_mma_h(f16v8 a, f16v8 b, f32v8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c); }

/* ---- WMMA tiled GEMM: Y[m][n] (+)= scale * sum_k A[row(m)][k] * W(n,k) + bias[n] ----
 * A: 16-bit activations [rows][lda] (bf16 when AF == 0, f16 when AF == 1), rows gathered through rowmap when given.
 * W formats (wt): 0 bf16 [N][K] (nn.Linear weight), 1 bf16 [K][N] (x @ W), 2 NVFP4 (Wq [N][K/2] nibbles low = lowest k, Wsf [N][K/16] e4m3, per-matrix fp32
 * global in wglob[e], dequantised exactly in f16), 3 FP8 e4m3 (Wq [N][K] bytes, per-output-channel fp32 scale Wsc[N]). wt 2/3 need AF == 1.
 * Grouped (seg != 0): block row-tile ty belongs to expert e = largest e with tpre[e] <= ty, rows seg[e] .. seg[e + 1], weights of expert e (bank layout [E][..]).
 * Output type ot: 0 fp32, 1 bf16, 2 f16.  Tile 64 x 128 x 32, 8 warps (2 x 4), warp tile 32 x 32; global -> register prefetch -> LDS, 16-bit k-contiguous tiles. */
__device__ __forceinline__ float pr_e2m1_val(unsigned c)
{
    float v = (c & 7u) == 0 ? 0.f : (c & 7u) == 1 ? 0.5f : (c & 7u) == 2 ? 1.f : (c & 7u) == 3 ? 1.5f : (c & 7u) == 4 ? 2.f : (c & 7u) == 5 ? 3.f : (c & 7u) == 6 ? 4.f : 6.f;
    return (c & 8u) ? -v : v;
}
__device__ __forceinline__ float pr_e4m3_val(unsigned b)
{
    unsigned e = (b >> 3) & 15u, m = b & 7u;
    float v = e == 0 ? (float)m * 0.001953125f : (1.0f + (float)m * 0.125f) * exp2f((float)((int)e - 7));
    return (b & 0x80u) ? -v : v;
}

template <int AF>
__device__ __forceinline__ void pr_mm_impl(const u16 *A, long long lda, const int *rowmap, const void *W, const unsigned char *Wsf, const float *Wsc,
        const float *wglob, long long wstride, const float *bias, void *Y, long long ldy, int M, int N, int K, int accum,
        const int *seg, const int *tpre, int E, int wt, int ot)
{
    const int AP = 40;
    __shared__ u16 As[64 * 40];
    __shared__ u16 Bs[128 * 40];
    int m0, mend, e = 0;
    if (seg) {
        int ty = blockIdx.y;
        int total = tpre[E];
        if (ty >= total) return;
        int lo = 0, hi = E;
        while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (tpre[mid] <= ty) lo = mid; else hi = mid; }
        e = lo;
        m0 = seg[e] + (ty - tpre[e]) * 64; mend = seg[e + 1];
    } else { m0 = blockIdx.y * 64; mend = M; }
    if (m0 >= mend) return;
    int bn0 = blockIdx.x * 128;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, half = lane >> 4, id = lane & 15;
    int wm = warp >> 2, wn = warp & 3;
    f32v8 acc[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) for (int q = 0; q < 8; q++) acc[i][j][q] = 0.f;
    /* per-thread staging coordinates */
    int ar = tid >> 2, ac = (tid & 3) * 8;                  /* A: row, k chunk */
    int agm = m0 + ar; bool aok = agm < mend;
    long long asrc = aok ? (rowmap ? (long long)rowmap[agm] : (long long)agm) : 0;
    int br = tid >> 1, bc = (tid & 1) * 16;                  /* B (wt 0/2/3): row n, k offset 0/16 */
    int bn = bn0 + br; bool bok = bn < N;
    const u16 *W16 = (const u16 *)W;
    const unsigned char *W8 = (const unsigned char *)W;
    long long ebase = (long long)e * N;
    int nkt = (K + 31) >> 5;
    u32x4 ra, rb0, rb1; unsigned sbyte = 0;
    auto gload = [&](int kt) {
        int k0 = kt * 32;
        ra = (u32x4){ 0u, 0u, 0u, 0u };
        if (aok && k0 + ac < K) ra = *(const u32x4 *)(A + asrc * lda + k0 + ac);
        rb0 = (u32x4){ 0u, 0u, 0u, 0u }; rb1 = rb0; sbyte = 0;
        if (wt == 0) {
            if (bok && k0 + bc < K) rb0 = *(const u32x4 *)(W16 + (ebase + bn) * (long long)K + k0 + bc);
            if (bok && k0 + bc + 8 < K) rb1 = *(const u32x4 *)(W16 + (ebase + bn) * (long long)K + k0 + bc + 8);
        } else if (wt == 1) {
            int kk = tid >> 3, nc = (tid & 7) * 16;
            int gk = k0 + kk, gn = bn0 + nc;
            const u16 *wb = W16 + (long long)e * wstride;
            if (gk < K && gn + 8 <= N) rb0 = *(const u32x4 *)(wb + (long long)gk * N + gn);
            if (gk < K && gn + 16 <= N) rb1 = *(const u32x4 *)(wb + (long long)gk * N + gn + 8);
        } else if (wt == 2) {
            if (bok && k0 + bc < K) {
                u32x2 t = *(const u32x2 *)(W8 + (ebase + bn) * (long long)(K >> 1) + ((k0 + bc) >> 1));
                rb0[0] = t[0]; rb0[1] = t[1];
                sbyte = Wsf[(ebase + bn) * (long long)(K >> 4) + ((k0 + bc) >> 4)];
            }
        } else {
            if (bok && k0 + bc < K) rb0 = *(const u32x4 *)(W8 + (ebase + bn) * (long long)K + k0 + bc);
        }
    };
    auto sstore = [&]() {
        *(u32x4 *)(As + ar * AP + ac) = ra;
        if (wt == 0) {
            *(u32x4 *)(Bs + br * AP + bc) = rb0; *(u32x4 *)(Bs + br * AP + bc + 8) = rb1;
        } else if (wt == 1) {
            int kk = tid >> 3, nc = (tid & 7) * 16;
            const u16 *e0 = (const u16 *)&rb0; const u16 *e1 = (const u16 *)&rb1;
            for (int j = 0; j < 8; j++) { Bs[(nc + j) * AP + kk] = e0[j]; Bs[(nc + 8 + j) * AP + kk] = e1[j]; }
        } else if (wt == 2) {
            /* exact: e2m1 value (f16 bit table) x e4m3 block scale (f16) in packed f16 math, product has <= 4 mantissa bits */
            unsigned e_ = (sbyte >> 3) & 15u, m_ = sbyte & 7u;
            u16 sbits = e_ ? (u16)(((e_ + 8u) << 10) | (m_ << 7)) : pr_f2h((float)m_ * 0.001953125f);
            _Float16 sch = __builtin_bit_cast(_Float16, sbits);
            typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
            f16x2 s2 = (f16x2){ sch, sch };
            const unsigned long long T0 = 0x3E003C0038000000ULL, T1 = 0x4600440042004000ULL;
            u16 o[16];
            for (int w = 0; w < 2; w++) {
                u32 word = w == 0 ? rb0[0] : rb0[1];
                for (int j = 0; j < 8; j += 2) {
                    u32 n0 = (word >> (4 * j)) & 15u, n1 = (word >> (4 * j + 4)) & 15u;
                    u32 b0 = (u32)(((n0 & 4u) ? T1 : T0) >> ((n0 & 3u) * 16)) & 0xffffu; b0 |= (n0 & 8u) << 12;
                    u32 b1 = (u32)(((n1 & 4u) ? T1 : T0) >> ((n1 & 3u) * 16)) & 0xffffu; b1 |= (n1 & 8u) << 12;
                    u32 pk = b0 | (b1 << 16);
                    f16x2 v = __builtin_bit_cast(f16x2, pk) * s2;
                    u32 r2 = __builtin_bit_cast(u32, v);
                    o[w * 8 + j] = (u16)(r2 & 0xffffu); o[w * 8 + j + 1] = (u16)(r2 >> 16);
                }
            }
            u16 *dst = Bs + br * AP + bc;
            *(u32x4 *)dst = *(const u32x4 *)&o[0]; *(u32x4 *)(dst + 8) = *(const u32x4 *)&o[8];
        } else {
            u16 o[16];
            const unsigned char *by = (const unsigned char *)&rb0;
            for (int j = 0; j < 16; j++) o[j] = pr_f2h(pr_e4m3_val(by[j]));
            u16 *dst = Bs + br * AP + bc;
            *(u32x4 *)dst = *(const u32x4 *)&o[0]; *(u32x4 *)(dst + 8) = *(const u32x4 *)&o[8];
        }
    };
    gload(0);
    for (int kt = 0; kt < nkt; kt++) {
        sstore();
        __syncthreads();
        if (kt + 1 < nkt) gload(kt + 1);
        for (int ks = 0; ks < 2; ks++) {
            u16v8 af[2], bf[2];
            for (int i = 0; i < 2; i++) af[i] = *(const u16v8 *)(As + (wm * 32 + i * 16 + id) * AP + ks * 16 + half * 8);
            for (int j = 0; j < 2; j++) bf[j] = *(const u16v8 *)(Bs + (wn * 32 + j * 16 + id) * AP + ks * 16 + half * 8);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) {
                    if (AF == 0) acc[i][j] = pr_mma_bf(af[i], bf[j], acc[i][j]);
                    else acc[i][j] = pr_mma_h(__builtin_bit_cast(f16v8, af[i]), __builtin_bit_cast(f16v8, bf[j]), acc[i][j]);
                }
        }
        __syncthreads();
    }
    float galpha = (wt == 2 && wglob) ? wglob[e] : 1.0f;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            int col = bn0 + wn * 32 + j * 16 + id;
            if (col >= N) continue;
            float cs = galpha;
            if (wt == 3) cs = Wsc[(long long)e * N + col];
            float bb = bias ? bias[col] : 0.f;
            for (int q = 0; q < 8; q++) {
                int row = m0 + wm * 32 + i * 16 + half * 8 + q;
                if (row >= mend) continue;
                float v = acc[i][j][q] * cs + bb;
                long long o = (long long)row * ldy + col;
                if (ot == 0) { float *y = (float *)Y; if (accum) y[o] += v; else y[o] = v; }
                else if (ot == 1) ((u16 *)Y)[o] = pr_f2bf(v);
                else ((u16 *)Y)[o] = pr_f2h(v);
            }
        }
}

/* Gated GQA flash attention (bf16 WMMA, fp32 online softmax) with the teacher-forcing code mask / interval mask and tile skipping.
 * Q [nq][ldq] (head h at h * HD), K/V [nk][ldk] (kv head at kvh * HD), bf16; O [nq][ldo] bf16.
 * mask (IV == false): k allowed for q iff k == q || k == 0 || (k odd && k <= hi(q)), hi(q) = q odd ? q : q - 3; qcode / kcode / ktile as in the CUDA runner.
 * mask (IV == true): qcode = [nq][6] key intervals (lo0, hi0, lo1, hi1, lo2, hi2).
 * Computed TRANSPOSED: S^T = K Q^T (A = K tile, B = Q), so the C layout of S^T (rows = keys, lane id = query) IS the B layout of the P^T V^T product and no
 * data moves between the two WMMAs; O^T[d][q] accumulates, the head output of 8 consecutive d is one 16-byte store.
 * gate (fp32 logits [nq][H], optional) scales the head output by sigmoid. LDS: K tile [32][HD+8], V tile transposed [HD][40], tile list. */
template <int HD, bool IV>
__device__ __forceinline__ void pr_flash_impl(unsigned char *smraw, const u16 *Q, long long ldq, const u16 *K, long long ldk, const u16 *V, long long ldv,
        u16 *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    const int BR = 64, BC = 32, KP = HD + 8, VP = BC + 8, NKK = HD / 16, NDT = HD / 16, NCH = HD / 8;
    const float lscale = scale * 1.4426950408889634f;
    int h = blockIdx.y, kvh = h / (H / KVH);
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, half = lane >> 4, id = lane & 15;
    int q0 = blockIdx.x * BR;
    int qi = q0 + wid * 16 + id;
    bool q_ok = qi < nq;
    u16 *Ks = (u16 *)smraw;
    u16 *Vt = Ks + BC * KP;
    int *tl = (int *)(Vt + HD * VP);
    int ntile = (nk + BC - 1) / BC;
    int *qs = tl + ntile;
    if (IV) {
        if (threadIdx.x < 3) { qs[2 * threadIdx.x] = 1 << 30; qs[2 * threadIdx.x + 1] = -1; }
        if (threadIdx.x == 0) qs[6] = 0;
        __syncthreads();
        if (threadIdx.x < BR && q0 + (int)threadIdx.x < nq) {
            for (int j = 0; j < 3; j++) {
                int lo = qcode[(q0 + threadIdx.x) * 6 + 2 * j], hi = qcode[(q0 + threadIdx.x) * 6 + 2 * j + 1];
                if (lo < hi) { atomicMin(&qs[2 * j], lo); atomicMax(&qs[2 * j + 1], hi); }
            }
        }
        __syncthreads();
        for (int t = threadIdx.x; t < ntile; t += blockDim.x) {
            int a = t * BC, b = a + BC, ok = 0;
            for (int j = 0; j < 3; j++) ok |= (qs[2 * j] < b && qs[2 * j + 1] > a);
            tl[t] = ok;
        }
    } else {
        if (threadIdx.x == 0) { qs[0] = 1 << 30; qs[1] = -1; qs[2] = -4; qs[3] = 0; }
        __syncthreads();
        if (threadIdx.x < BR && q0 + (int)threadIdx.x < nq) {
            int c = qcode[q0 + threadIdx.x];
            atomicMin(&qs[0], c); atomicMax(&qs[1], c);
            atomicMax(&qs[2], (c & 1) ? c : c - 3);
        }
        __syncthreads();
        for (int t = threadIdx.x; t < ntile; t += blockDim.x) {
            int kmin = ktile[3 * t], kmax = ktile[3 * t + 1], kodd = ktile[3 * t + 2];
            tl[t] = (kmin == 0) || (kmax >= qs[0] && kmin <= qs[1]) || (kodd <= qs[2]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int t = 0; t < ntile; t++) if (tl[t]) tl[n++] = t;
        qs[IV ? 6 : 3] = n;
    }
    __syncthreads();
    int n_tiles = qs[IV ? 6 : 3];

    const u16 *Qh = Q + (long long)h * HD;
    const u16 *Kh = K + (long long)kvh * HD;
    const u16 *Vh = V + (long long)kvh * HD;
    u16v8 qfr[NKK];
    for (int kk = 0; kk < NKK; kk++) {
        u32x4 z = (u32x4){ 0u, 0u, 0u, 0u };
        if (q_ok) z = *(const u32x4 *)(Qh + (long long)qi * ldq + kk * 16 + half * 8);
        qfr[kk] = __builtin_bit_cast(u16v8, z);
    }
    int iv[6] = { 0, 0, 0, 0, 0, 0 };
    int qc = -100, hiq = -103;
    if (IV) { if (q_ok) for (int j = 0; j < 6; j++) iv[j] = qcode[(long long)qi * 6 + j]; }
    else if (q_ok) { qc = qcode[qi]; hiq = (qc & 1) ? qc : qc - 3; }
    f32v8 oacc[NDT];
    for (int d = 0; d < NDT; d++) for (int i = 0; i < 8; i++) oacc[d][i] = 0.f;
    float m_q = -1e30f, l_q = 0.f;

    for (int t = 0; t < n_tiles; t++) {
        int kv0 = tl[t] * BC;
        __syncthreads();
        for (int idx = threadIdx.x; idx < BC * NCH; idx += blockDim.x) {
            int j = idx / NCH, d = (idx - j * NCH) * 8;
            int key = kv0 + j;
            u32x4 kvv = (u32x4){ 0u, 0u, 0u, 0u }, vvv = kvv;
            if (key < nk) {
                kvv = *(const u32x4 *)(Kh + (long long)key * ldk + d);
                vvv = *(const u32x4 *)(Vh + (long long)key * ldv + d);
            }
            *(u32x4 *)(Ks + j * KP + d) = kvv;
            const u16 *ve = (const u16 *)&vvv;
            for (int u = 0; u < 8; u++) Vt[(d + u) * VP + j] = ve[u];
        }
        __syncthreads();
        float s[2][8];
        for (int mt = 0; mt < 2; mt++) {
            f32v8 a8;
            for (int i = 0; i < 8; i++) a8[i] = 0.f;
            for (int kk = 0; kk < NKK; kk++) {
                u16v8 af = *(const u16v8 *)(Ks + (mt * 16 + id) * KP + kk * 16 + half * 8);
                a8 = pr_mma_bf(af, qfr[kk], a8);
            }
            for (int i = 0; i < 8; i++) {
                int key = kv0 + mt * 16 + half * 8 + i;
                bool ok = false;
                if (key < nk) {
                    if (IV) {
                        for (int j = 0; j < 3; j++) ok |= key >= iv[2 * j] && key < iv[2 * j + 1];
                    } else {
                        int kc = kcode[key];
                        ok = q_ok && ((kc == qc) || (kc == 0) || ((kc & 1) && kc <= hiq));
                    }
                }
                s[mt][i] = ok ? a8[i] * lscale : -1e30f;
            }
        }
        float mx = -1e30f;
        for (int mt = 0; mt < 2; mt++) for (int i = 0; i < 8; i++) mx = fmaxf(mx, s[mt][i]);
        mx = fmaxf(mx, __shfl_xor(mx, 16));
        float mn = fmaxf(m_q, mx);
        float al = pr_ex2(m_q - mn);
        m_q = mn; l_q *= al;
        for (int d = 0; d < NDT; d++) for (int i = 0; i < 8; i++) oacc[d][i] *= al;
        u16v8 pf[2];
        float ls = 0.f;
        for (int mt = 0; mt < 2; mt++)
            for (int i = 0; i < 8; i++) {
                float p = s[mt][i] > -1e29f ? pr_ex2(s[mt][i] - m_q) : 0.f;
                ls += p;
                pf[mt][i] = pr_f2bf(p);
            }
        l_q += ls;
        for (int mt = 0; mt < 2; mt++)
            for (int d = 0; d < NDT; d++) {
                u16v8 af = *(const u16v8 *)(Vt + (d * 16 + id) * VP + mt * 16 + half * 8);
                oacc[d] = pr_mma_bf(af, pf[mt], oacc[d]);
            }
    }
    float ltot = l_q + __shfl_xor(l_q, 16);
    float g = 1.f;
    if (gate && q_ok) g = 1.0f / (1.0f + expf(-gate[(long long)qi * H + h]));
    float inv = ltot > 0.f ? g / ltot : 0.f;
    if (q_ok) {
        for (int d = 0; d < NDT; d++) {
            u32x4 o;
            o[0] = pr_pack2(oacc[d][0] * inv, oacc[d][1] * inv); o[1] = pr_pack2(oacc[d][2] * inv, oacc[d][3] * inv);
            o[2] = pr_pack2(oacc[d][4] * inv, oacc[d][5] * inv); o[3] = pr_pack2(oacc[d][6] * inv, oacc[d][7] * inv);
            *(u32x4 *)(O + (long long)qi * ldo + (long long)h * HD + d * 16 + half * 8) = o;
        }
    }
}

extern "C" {
__global__ void __launch_bounds__(256) pr_mm_bf(const u16 *A, long long lda, const int *rowmap, const void *W, const unsigned char *Wsf, const float *Wsc,
        const float *wglob, long long wstride, const float *bias, void *Y, long long ldy, int M, int N, int K, int accum,
        const int *seg, const int *tpre, int E, int wt, int ot)
{ pr_mm_impl<0>(A, lda, rowmap, W, Wsf, Wsc, wglob, wstride, bias, Y, ldy, M, N, K, accum, seg, tpre, E, wt, ot); }
__global__ void __launch_bounds__(256) pr_mm_h(const u16 *A, long long lda, const int *rowmap, const void *W, const unsigned char *Wsf, const float *Wsc,
        const float *wglob, long long wstride, const float *bias, void *Y, long long ldy, int M, int N, int K, int accum,
        const int *seg, const int *tpre, int E, int wt, int ot)
{ pr_mm_impl<1>(A, lda, rowmap, W, Wsf, Wsc, wglob, wstride, bias, Y, ldy, M, N, K, accum, seg, tpre, E, wt, ot); }
__global__ void __launch_bounds__(128) pr_flash16(const u16 *Q, long long ldq, const u16 *K, long long ldk, const u16 *V, long long ldv, u16 *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm_pr_flash16[];
    pr_flash_impl<16, false>(pr_sm_pr_flash16, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flash32(const u16 *Q, long long ldq, const u16 *K, long long ldk, const u16 *V, long long ldv, u16 *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm_pr_flash32[];
    pr_flash_impl<32, false>(pr_sm_pr_flash32, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flash64(const u16 *Q, long long ldq, const u16 *K, long long ldk, const u16 *V, long long ldv, u16 *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm_pr_flash64[];
    pr_flash_impl<64, false>(pr_sm_pr_flash64, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flash128(const u16 *Q, long long ldq, const u16 *K, long long ldk, const u16 *V, long long ldv, u16 *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm_pr_flash128[];
    pr_flash_impl<128, false>(pr_sm_pr_flash128, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flashiv16(const u16 *Q, long long ldq, const u16 *K, long long ldk, const u16 *V, long long ldv, u16 *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm_pr_flashiv16[];
    pr_flash_impl<16, true>(pr_sm_pr_flashiv16, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flashiv32(const u16 *Q, long long ldq, const u16 *K, long long ldk, const u16 *V, long long ldv, u16 *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm_pr_flashiv32[];
    pr_flash_impl<32, true>(pr_sm_pr_flashiv32, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flashiv64(const u16 *Q, long long ldq, const u16 *K, long long ldk, const u16 *V, long long ldv, u16 *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm_pr_flashiv64[];
    pr_flash_impl<64, true>(pr_sm_pr_flashiv64, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flashiv128(const u16 *Q, long long ldq, const u16 *K, long long ldk, const u16 *V, long long ldv, u16 *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm_pr_flashiv128[];
    pr_flash_impl<128, true>(pr_sm_pr_flashiv128, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}

/* ---- device-side MoE dispatch ---- */
__global__ void pr_moe_count(const int *topi, int n, int *cnt)
{
    __shared__ int wsum[32];
    int e = blockIdx.x, c = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) c += (topi[i] == e);
    for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = c;
    __syncthreads();
    if (threadIdx.x == 0) { int t = 0; for (int w = 0; w < (blockDim.x >> 5); w++) t += wsum[w]; cnt[e] = t; }
}
__global__ void pr_moe_scan(const int *cnt, int E, int *seg)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) { int s = 0; seg[0] = 0; for (int e = 0; e < E; e++) { s += cnt[e]; seg[e + 1] = s; } }
}
__global__ void pr_moe_fill(const int *topi, int n, int k, const int *seg, int *tokid, int *tokrow)
{
    __shared__ int wsum[32];
    __shared__ int base;
    int e = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (threadIdx.x == 0) base = 0;
    __syncthreads();
    for (int i0 = 0; i0 < n; i0 += blockDim.x) {
        int i = i0 + threadIdx.x;
        int hit = (i < n && topi[i] == e);
        unsigned b = (unsigned)__ballot(hit);
        int wp = __popc(b & ((1u << lane) - 1u));
        if (lane == 0) wsum[warp] = __popc(b);
        __syncthreads();
        int off = 0;
        for (int w = 0; w < warp; w++) off += wsum[w];
        if (hit) { int row = seg[e] + base + off + wp; tokid[row] = i / k; tokrow[i] = row; }
        __syncthreads();
        if (threadIdx.x == 0) { int t = 0; for (int w = 0; w < nw; w++) t += wsum[w]; base += t; }
        __syncthreads();
    }
}
/* xg[row] = xb[tokid[row]]  (C multiple of 8) */
__global__ void pr_gather_bf(const u16 *xb, const int *tokid, u16 *xg, int C, long long R)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int c8 = C >> 3;
    if (i >= R * c8) return;
    long long r = i / c8;
    int c = (int)(i - r * c8);
    ((u32x4 *)xg)[r * c8 + c] = ((const u32x4 *)xb)[(long long)tokid[r] * c8 + c];
}
__global__ void pr_combine_bf(float *out, const u16 *y, const int *tokrow, const float *gate, float scale, int C, long long T, int k)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    long long t = i / C;
    int c = (int)(i - t * C);
    float a = 0.f;
    for (int j = 0; j < k; j++) a += gate[t * k + j] * pr_bf2f(y[(long long)tokrow[t * k + j] * C + c]);
    out[i] += scale * a;
}

/* ---- synthetic weights: deterministic hash uniform in [-amp, amp] ---- */
__device__ __forceinline__ unsigned pr_hash(unsigned long long x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return (unsigned)x;
}
__global__ void pr_fill_rand_bf16(u16 *p, long long n, unsigned long long seed, float amp)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u = ((float)(pr_hash(seed * 0x9e3779b97f4a7c15ULL + (unsigned long long)i) >> 8) * (1.0f / 8388608.0f)) - 1.0f;
    p[i] = pr_f2bf(u * amp);
}
}
);
#endif
