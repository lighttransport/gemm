/*
 * praxis_kernels.h - NVRTC kernel source of the praxis-1 CUDA runner.
 *
 * All activations are fp32 row-major [rows][C]. Weights are read in their
 * checkpoint dtype (0 = f32, 1 = bf16, 2 = f16) and widened to f32, so a bf16
 * checkpoint runs without a conversion copy and with exactly the fp32 values.
 * (No '#' directives and no apostrophes inside: the source lives in a macro.)
 */
#ifndef PRAXIS_KERNELS_H
#define PRAXIS_KERNELS_H

#define PRK_STR(...) #__VA_ARGS__
static const char *praxis_kernel_src = PRK_STR(
extern "C" {

__device__ __forceinline__ float pr_ldw(const void *W, long long i, int dt) {
    if (dt == 0) return ((const float *)W)[i];
    unsigned short h = ((const unsigned short *)W)[i];
    if (dt == 1) return __uint_as_float(((unsigned)h) << 16);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
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


/* ================= fast path (bf16 operands, fp32 accumulate) ================= */

__device__ __forceinline__ unsigned short pr_f2bf(float f) {
    unsigned short h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
__device__ __forceinline__ float pr_bf2f(unsigned short h) { return __uint_as_float(((unsigned)h) << 16); }
__device__ __forceinline__ unsigned pr_pack2(float lo, float hi) {
    return ((unsigned)pr_f2bf(hi) << 16) | (unsigned)pr_f2bf(lo);
}

/* y[m][k] = bf16(x[m * ldx + k]), dense [M][K] output */
__global__ void pr_cvt_bf16(const float *x, long long ldx, unsigned short *y, int K, long long M)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * K) return;
    long long m = i / K;
    int k = (int)(i - m * K);
    y[i] = pr_f2bf(x[m * ldx + k]);
}

/* zero-centred / plain RMSNorm of one row with fp32 and/or bf16 outputs */
__global__ void pr_rmsnorm_bf(const float *x, long long ldx, float *y, unsigned short *yb, const float *w,
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
__global__ void pr_qkprep(const float *x, long long ldx, unsigned short *y, long long ldy, const float *w,
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
    unsigned short *o = y + t * ldy + h * hd;
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
__global__ void pr_swiglu_bf(const unsigned short *a, long long lda, const unsigned short *b, long long ldb,
                             unsigned short *o, long long ldo, int H, long long rows)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * H) return;
    long long r = i / H;
    int c = (int)(i - r * H);
    float x = pr_bf2f(a[r * lda + c]), g = pr_bf2f(b[r * ldb + c]);
    o[r * ldo + c] = pr_f2bf((x / (1.0f + expf(-x))) * g);
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

enum { PR_FA_BR = 64, PR_FA_BC = 32 };

__device__ __forceinline__ void pr_mma16816(float &c0, float &c1, float &c2, float &c3,
                                            unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                            unsigned b0, unsigned b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float pr_ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

/* Gated GQA flash attention with the teacher-forcing code mask and tile skipping.
 * Q [nq][ldq] (head h at h * HD), K/V [nk][ldk] (kv head at kvh * HD), bf16; O [nq][ldo] bf16.
 * mask: k allowed for q iff k == q || k == 0 || (k odd && k <= hi(q)), hi(q) = q odd ? q : q - 3.
 * gate (fp32 logits [nq][H], optional) scales the head output by sigmoid. */
}
template <int HD, bool IV>
__device__ __forceinline__ void pr_flash_impl(unsigned char *smraw, const unsigned short *Q, long long ldq,
        const unsigned short *K, long long ldk, const unsigned short *V, long long ldv,
        unsigned short *O, long long ldo, const float *gate, int H, int KVH, int nq, int nk, float scale,
        const int *qcode, const int *kcode, const int *ktile)
{
    const int BR = PR_FA_BR, BC = PR_FA_BC, DK = HD / 16, BCN = BC / 8, BCK = BC / 16, DN = HD / 8, DP = HD + 8;
    const float lscale = scale * 1.4426950408889634f;
    int h = blockIdx.y, kvh = h / (H / KVH);
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int q0 = blockIdx.x * BR;
    int warp_q0 = q0 + wid * 16;
    int row_a = lane >> 2, row_b = row_a + 8, col_g = (lane & 3) * 2;
    const unsigned short *Qh = Q + (long long)h * HD;
    const unsigned short *Kh = K + (long long)kvh * HD;
    const unsigned short *Vh = V + (long long)kvh * HD;
    unsigned short *sK[2], *sV[2];
    unsigned short *smem = (unsigned short *)smraw;
    sK[0] = smem; sK[1] = sK[0] + BC * DP; sV[0] = sK[1] + BC * DP; sV[1] = sV[0] + BC * DP;
    int *tl = (int *)(smem + 4 * BC * DP);
    int ntile = (nk + BC - 1) / BC;
    int *qs = tl + ntile;          /* [0] qmin [1] qmax [2] qhimax [3] count */
    if (IV) {
        /* qcode = [nq][6] intervals (lo0, hi0, lo1, hi1, lo2, hi2): query q may read keys in the union of [lo_j, hi_j) (block-sparse by intervals) */
        if (threadIdx.x < 3) { qs[2 * threadIdx.x] = 1 << 30; qs[2 * threadIdx.x + 1] = -1; }
        if (threadIdx.x == 0) qs[6] = 0;
        __syncthreads();
        if (threadIdx.x < BR && q0 + threadIdx.x < nq) {
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
    if (threadIdx.x < BR && q0 + threadIdx.x < nq) {
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

    unsigned qfrag[DK][4];
    int gqa = warp_q0 + row_a, gqb = warp_q0 + row_b;
    bool va_ok = gqa < nq, vb_ok = gqb < nq;
    for (int kk = 0; kk < DK; kk++) {
        const unsigned short *pa = Qh + (long long)gqa * ldq + kk * 16 + col_g;
        const unsigned short *pb = Qh + (long long)gqb * ldq + kk * 16 + col_g;
        qfrag[kk][0] = va_ok ? *(const unsigned *)pa : 0u;
        qfrag[kk][1] = vb_ok ? *(const unsigned *)pb : 0u;
        qfrag[kk][2] = va_ok ? *(const unsigned *)(pa + 8) : 0u;
        qfrag[kk][3] = vb_ok ? *(const unsigned *)(pb + 8) : 0u;
    }
    int ivA[6] = { 0, 0, 0, 0, 0, 0 }, ivB[6] = { 0, 0, 0, 0, 0, 0 };
    if (IV) {
        for (int j = 0; j < 6; j++) { if (va_ok) ivA[j] = qcode[gqa * 6 + j]; if (vb_ok) ivB[j] = qcode[gqb * 6 + j]; }
    }
    int qca = (!IV && va_ok) ? qcode[gqa] : -100, qcb = (!IV && vb_ok) ? qcode[gqb] : -100;
    int hia = (qca & 1) ? qca : qca - 3, hib = (qcb & 1) ? qcb : qcb - 3;
    float ofrag[DN][4];
    for (int n = 0; n < DN; n++) for (int i = 0; i < 4; i++) ofrag[n][i] = 0.f;
    float m_a = -1e30f, m_b = -1e30f, l_a = 0.f, l_b = 0.f;

    const int total_u4 = BC * HD / 8, row_u4 = HD / 8;
    if (n_tiles > 0) {
        int kv0 = tl[0] * BC;
        for (int idx = threadIdx.x; idx < total_u4; idx += blockDim.x) {
            int j = idx / row_u4, dd = (idx - j * row_u4) * 8;
            unsigned dk = __cvta_generic_to_shared(sK[0] + j * DP + dd), dv = __cvta_generic_to_shared(sV[0] + j * DP + dd);
            int nb = (kv0 + j < nk) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dk), "l"(Kh + (long long)(kv0 + j) * ldk + dd), "r"(nb));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dv), "l"(Vh + (long long)(kv0 + j) * ldv + dd), "r"(nb));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    int ldm_row = lane & 7, ldm_col = (lane & 8) ? 8 : 0;

    for (int t = 0; t < n_tiles; t++) {
        int cur = t & 1, nxt = 1 - cur;
        int kv0 = tl[t] * BC;
        if (t + 1 < n_tiles) {
            int kv0n = tl[t + 1] * BC;
            for (int idx = threadIdx.x; idx < total_u4; idx += blockDim.x) {
                int j = idx / row_u4, dd = (idx - j * row_u4) * 8;
                unsigned dk = __cvta_generic_to_shared(sK[nxt] + j * DP + dd), dv = __cvta_generic_to_shared(sV[nxt] + j * DP + dd);
                int nb = (kv0n + j < nk) ? 16 : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dk), "l"(Kh + (long long)(kv0n + j) * ldk + dd), "r"(nb));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dv), "l"(Vh + (long long)(kv0n + j) * ldv + dd), "r"(nb));
            }
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const unsigned short *tK = sK[cur], *tV = sV[cur];
        float sf[BCN][4];
        for (int n = 0; n < BCN; n++) for (int i = 0; i < 4; i++) sf[n][i] = 0.f;
        for (int kk = 0; kk < DK; kk++) {
            unsigned bq[BCN][2];
            for (int np = 0; np < BCN; np += 2) {
                unsigned sa = __cvta_generic_to_shared(tK + (np * 8 + ldm_row + ((lane & 16) ? 8 : 0)) * DP + kk * 16 + ldm_col);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(bq[np][0]), "=r"(bq[np][1]), "=r"(bq[np + 1][0]), "=r"(bq[np + 1][1]) : "r"(sa));
            }
            for (int nn = 0; nn < BCN; nn++)
                pr_mma16816(sf[nn][0], sf[nn][1], sf[nn][2], sf[nn][3], qfrag[kk][0], qfrag[kk][1], qfrag[kk][2], qfrag[kk][3], bq[nn][0], bq[nn][1]);
        }
        for (int nn = 0; nn < BCN; nn++) {
            int c0 = kv0 + nn * 8 + col_g, c1 = c0 + 1;
            bool a0, a1, b0, b1;
            if (IV) {
                a0 = a1 = b0 = b1 = false;
                for (int j = 0; j < 3; j++) {
                    a0 |= c0 < nk && c0 >= ivA[2 * j] && c0 < ivA[2 * j + 1]; a1 |= c1 < nk && c1 >= ivA[2 * j] && c1 < ivA[2 * j + 1];
                    b0 |= c0 < nk && c0 >= ivB[2 * j] && c0 < ivB[2 * j + 1]; b1 |= c1 < nk && c1 >= ivB[2 * j] && c1 < ivB[2 * j + 1];
                }
            } else {
            int k0 = c0 < nk ? kcode[c0] : -100, k1 = c1 < nk ? kcode[c1] : -100;
            a0 = c0 < nk && qca >= 0 && ((k0 == qca) || (k0 == 0) || ((k0 & 1) && k0 <= hia));
            a1 = c1 < nk && qca >= 0 && ((k1 == qca) || (k1 == 0) || ((k1 & 1) && k1 <= hia));
            b0 = c0 < nk && qcb >= 0 && ((k0 == qcb) || (k0 == 0) || ((k0 & 1) && k0 <= hib));
            b1 = c1 < nk && qcb >= 0 && ((k1 == qcb) || (k1 == 0) || ((k1 & 1) && k1 <= hib));
            }
            sf[nn][0] = a0 ? sf[nn][0] * lscale : -1e30f;
            sf[nn][1] = a1 ? sf[nn][1] * lscale : -1e30f;
            sf[nn][2] = b0 ? sf[nn][2] * lscale : -1e30f;
            sf[nn][3] = b1 ? sf[nn][3] * lscale : -1e30f;
        }
        float mx_a = -1e30f, mx_b = -1e30f;
        for (int nn = 0; nn < BCN; nn++) {
            mx_a = fmaxf(mx_a, fmaxf(sf[nn][0], sf[nn][1]));
            mx_b = fmaxf(mx_b, fmaxf(sf[nn][2], sf[nn][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
        float mna = fmaxf(m_a, mx_a), mnb = fmaxf(m_b, mx_b);
        float al_a = pr_ex2(m_a - mna), al_b = pr_ex2(m_b - mnb);
        for (int n = 0; n < DN; n++) { ofrag[n][0] *= al_a; ofrag[n][1] *= al_a; ofrag[n][2] *= al_b; ofrag[n][3] *= al_b; }
        l_a *= al_a; l_b *= al_b;
        m_a = mna; m_b = mnb;
        unsigned pf[BCK][4];
        float rs_a = 0.f, rs_b = 0.f;
        for (int kk = 0; kk < BCK; kk++) {
            float p[8];
            for (int u = 0; u < 2; u++) {
                p[u * 4 + 0] = sf[2 * kk + u][0] > -1e29f ? pr_ex2(sf[2 * kk + u][0] - m_a) : 0.f;
                p[u * 4 + 1] = sf[2 * kk + u][1] > -1e29f ? pr_ex2(sf[2 * kk + u][1] - m_a) : 0.f;
                p[u * 4 + 2] = sf[2 * kk + u][2] > -1e29f ? pr_ex2(sf[2 * kk + u][2] - m_b) : 0.f;
                p[u * 4 + 3] = sf[2 * kk + u][3] > -1e29f ? pr_ex2(sf[2 * kk + u][3] - m_b) : 0.f;
            }
            rs_a += p[0] + p[1] + p[4] + p[5];
            rs_b += p[2] + p[3] + p[6] + p[7];
            pf[kk][0] = pr_pack2(p[0], p[1]); pf[kk][1] = pr_pack2(p[2], p[3]);
            pf[kk][2] = pr_pack2(p[4], p[5]); pf[kk][3] = pr_pack2(p[6], p[7]);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1); rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1); rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        l_a += rs_a; l_b += rs_b;
        for (int kk = 0; kk < BCK; kk++) {
            int bc_row = kk * 16 + ldm_row + ((lane & 8) ? 8 : 0);
            unsigned vb[4];
            for (int nn = 0; nn < DN; nn++) {
                unsigned b0, b1;
                if ((nn & 1) == 0) {
                    unsigned sa = __cvta_generic_to_shared(tV + bc_row * DP + nn * 8 + ((lane & 16) ? 8 : 0));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                                 : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3]) : "r"(sa));
                    b0 = vb[0]; b1 = vb[1];
                } else { b0 = vb[2]; b1 = vb[3]; }
                pr_mma16816(ofrag[nn][0], ofrag[nn][1], ofrag[nn][2], ofrag[nn][3], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], b0, b1);
            }
        }
        __syncthreads();
    }
    float ga = 1.f, gb = 1.f;
    if (gate) {
        if (va_ok) ga = 1.0f / (1.0f + expf(-gate[(long long)gqa * H + h]));
        if (vb_ok) gb = 1.0f / (1.0f + expf(-gate[(long long)gqb * H + h]));
    }
    float inv_a = (l_a > 0.f) ? ga / l_a : 0.f, inv_b = (l_b > 0.f) ? gb / l_b : 0.f;
    for (int nn = 0; nn < DN; nn++) {
        int d = nn * 8 + col_g;
        if (va_ok) *(unsigned *)(O + (long long)gqa * ldo + (long long)h * HD + d) = pr_pack2(ofrag[nn][0] * inv_a, ofrag[nn][1] * inv_a);
        if (vb_ok) *(unsigned *)(O + (long long)gqb * ldo + (long long)h * HD + d) = pr_pack2(ofrag[nn][2] * inv_b, ofrag[nn][3] * inv_b);
    }
}

extern "C" {
__global__ void __launch_bounds__(128) pr_flash64(const unsigned short *Q, long long ldq, const unsigned short *K, long long ldk,
        const unsigned short *V, long long ldv, unsigned short *O, long long ldo, const float *gate, int H, int KVH,
        int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm64[];
    pr_flash_impl<64, false>(pr_sm64, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flash128(const unsigned short *Q, long long ldq, const unsigned short *K, long long ldk,
        const unsigned short *V, long long ldv, unsigned short *O, long long ldo, const float *gate, int H, int KVH,
        int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm128[];
    pr_flash_impl<128, false>(pr_sm128, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flash16(const unsigned short *Q, long long ldq, const unsigned short *K, long long ldk,
        const unsigned short *V, long long ldv, unsigned short *O, long long ldo, const float *gate, int H, int KVH,
        int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm16[];
    pr_flash_impl<16, false>(pr_sm16, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flash32(const unsigned short *Q, long long ldq, const unsigned short *K, long long ldk,
        const unsigned short *V, long long ldv, unsigned short *O, long long ldo, const float *gate, int H, int KVH,
        int nq, int nk, float scale, const int *qcode, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_sm32[];
    pr_flash_impl<32, false>(pr_sm32, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qcode, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flashiv16(const unsigned short *Q, long long ldq, const unsigned short *K, long long ldk,
        const unsigned short *V, long long ldv, unsigned short *O, long long ldo, const float *gate, int H, int KVH,
        int nq, int nk, float scale, const int *qiv, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_smiv16[];
    pr_flash_impl<16, true>(pr_smiv16, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qiv, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flashiv32(const unsigned short *Q, long long ldq, const unsigned short *K, long long ldk,
        const unsigned short *V, long long ldv, unsigned short *O, long long ldo, const float *gate, int H, int KVH,
        int nq, int nk, float scale, const int *qiv, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_smiv32[];
    pr_flash_impl<32, true>(pr_smiv32, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qiv, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flashiv64(const unsigned short *Q, long long ldq, const unsigned short *K, long long ldk,
        const unsigned short *V, long long ldv, unsigned short *O, long long ldo, const float *gate, int H, int KVH,
        int nq, int nk, float scale, const int *qiv, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_smiv64[];
    pr_flash_impl<64, true>(pr_smiv64, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qiv, kcode, ktile);
}
__global__ void __launch_bounds__(128) pr_flashiv128(const unsigned short *Q, long long ldq, const unsigned short *K, long long ldk,
        const unsigned short *V, long long ldv, unsigned short *O, long long ldo, const float *gate, int H, int KVH,
        int nq, int nk, float scale, const int *qiv, const int *kcode, const int *ktile)
{
    extern __shared__ unsigned char pr_smiv128[];
    pr_flash_impl<128, true>(pr_smiv128, Q, ldq, K, ldk, V, ldv, O, ldo, gate, H, KVH, nq, nk, scale, qiv, kcode, ktile);
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
        unsigned b = __ballot_sync(0xffffffffu, hit);
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
__global__ void pr_gather_bf(const unsigned short *xb, const int *tokid, unsigned short *xg, int C, long long R)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int c8 = C >> 3;
    if (i >= R * c8) return;
    long long r = i / c8;
    int c = (int)(i - r * c8);
    ((uint4 *)xg)[r * c8 + c] = ((const uint4 *)xb)[(long long)tokid[r] * c8 + c];
}
__global__ void pr_combine_bf(float *out, const unsigned short *y, const int *tokrow, const float *gate, float scale,
                              int C, long long T, int k)
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
__global__ void pr_fill_rand_bf16(unsigned short *p, long long n, unsigned long long seed, float amp)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u = ((float)(pr_hash(seed * 0x9e3779b97f4a7c15ULL + (unsigned long long)i) >> 8) * (1.0f / 8388608.0f)) - 1.0f;
    p[i] = pr_f2bf(u * amp);
}

}
);

#endif
