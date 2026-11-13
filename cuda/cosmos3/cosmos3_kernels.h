/*
 * cosmos3_kernels.h - NVRTC kernel source of the Cosmos3-Edge-Policy CUDA runner.
 *
 * Activations are bf16 (unsigned short) row-major; every kernel does its arithmetic in fp32 and rounds to bf16 at the same
 * points as the PyTorch bf16 reference (each nn.Linear / RMSNorm / residual add / rope product), so layer outputs track the oracle.
 * (No '#' directives and no apostrophes inside: the source lives in a macro.)
 */
#ifndef COSMOS3_KERNELS_H
#define COSMOS3_KERNELS_H

#define C3K_STR(...) #__VA_ARGS__
static const char *cosmos3_kernel_src = C3K_STR(
extern "C" {

__device__ __forceinline__ float c3_bf2f(unsigned short h) { return __uint_as_float(((unsigned)h) << 16); }
__device__ __forceinline__ unsigned short c3_f2bf(float f) {
    unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float c3_rb(float f) { return c3_bf2f(c3_f2bf(f)); }

__device__ __forceinline__ float c3_block_sum(float v, float *sh) {
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

/* y = bf16( w * (x * rsqrt(mean(x^2) + eps)) ); Cosmos3NemotronRMSNorm. one block (256 thr) per row. in-place allowed. */
__global__ void c3_rmsnorm(const unsigned short *x, const unsigned short *w, unsigned short *y, int dim, float eps, long long ldx, long long ldy) {
    __shared__ float sh[32];
    const unsigned short *xr = x + (long long)blockIdx.x * ldx;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float v = c3_bf2f(xr[i]); s += v * v; }
    s = c3_block_sum(s, sh);
    float r = rsqrtf(s / (float)dim + eps);
    unsigned short *yr = y + (long long)blockIdx.x * ldy;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yr[i] = c3_f2bf(c3_bf2f(w[i]) * (c3_bf2f(xr[i]) * r));
}

/* per-head RMSNorm over head_dim (=128) on [rows][nh*128] (row stride ld), in place. one warp per (row, head). */
__global__ void c3_rmsnorm_heads(unsigned short *x, const unsigned short *w, int rows, int nh, float eps, long long ld) {
    long long wid = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= (long long)rows * nh) return;
    int row = (int)(wid / nh), h = (int)(wid % nh);
    unsigned short *p = x + (long long)row * ld + h * 128;
    float v[4], s = 0.f;
    for (int j = 0; j < 4; j++) { v[j] = c3_bf2f(p[lane + 32 * j]); s += v[j] * v[j]; }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    float r = rsqrtf(s * (1.0f / 128.0f) + eps);
    for (int j = 0; j < 4; j++) p[lane + 32 * j] = c3_f2bf(c3_bf2f(w[lane + 32 * j]) * (v[j] * r));
}

/* rotary: x*cos + rotate_half(x)*sin, three bf16 roundings as in torch. cs tables [rows][128] bf16. x [rows][nh*128] stride ld. */
__global__ void c3_rope(unsigned short *x, const unsigned short *cs, const unsigned short *sn, int rows, int nh, long long ld) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tot = (long long)rows * nh * 64;
    if (i >= tot) return;
    int j = (int)(i & 63);
    long long rh = i >> 6;
    int h = (int)(rh % nh), row = (int)(rh / nh);
    unsigned short *p = x + (long long)row * ld + h * 128;
    float a = c3_bf2f(p[j]), b = c3_bf2f(p[j + 64]);
    const unsigned short *c = cs + (long long)row * 128, *s = sn + (long long)row * 128;
    float lo = c3_rb(c3_rb(a * c3_bf2f(c[j])) + c3_rb(-b * c3_bf2f(s[j])));
    float hi = c3_rb(c3_rb(b * c3_bf2f(c[j + 64])) + c3_rb(a * c3_bf2f(s[j + 64])));
    p[j] = c3_f2bf(lo); p[j + 64] = c3_f2bf(hi);
}

/* copy [rows][cols] with strides (used to build rope-ed k copy etc) */
__global__ void c3_copy2d(const unsigned short *s, long long lds, unsigned short *d, long long ldd, int rows, int cols) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cols) return;
    int r = (int)(i / cols), c = (int)(i % cols);
    d[(long long)r * ldd + c] = s[(long long)r * lds + c];
}

/* y = bf16(a + b), may alias a */
__global__ void c3_add(unsigned short *a, const unsigned short *b, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = c3_f2bf(c3_bf2f(a[i]) + c3_bf2f(b[i]));
}

/* relu(x)^2 in bf16 */
__global__ void c3_relu2(unsigned short *x, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = c3_bf2f(x[i]); v = v > 0.f ? v : 0.f; x[i] = c3_f2bf(v * v); }
}

/* fp32 gemm accumulators -> bf16 with bias (fused nn.Linear(bias=True): one rounding) */
__global__ void c3_bias_f32(const float *acc, const unsigned short *bias, unsigned short *y, int rows, int cols, long long ldy) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cols) return;
    int r = (int)(i / cols), c = (int)(i % cols);
    y[(long long)r * ldy + c] = c3_f2bf(acc[i] + c3_bf2f(bias[c]));
}

/* x[rows][cols] bf16 += bias[cols] (separate bf16 add, as DomainAwareLinear: bmm -> bf16, then + bias) */
__global__ void c3_add_bias(unsigned short *x, const unsigned short *bias, int rows, int cols) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cols) return;
    int c = (int)(i % cols);
    x[i] = c3_f2bf(c3_bf2f(x[i]) + c3_bf2f(bias[c]));
}

/* x[rows][cols] += v[cols] (bf16, broadcast row) */
__global__ void c3_add_row(unsigned short *x, const unsigned short *v, int rows, int cols) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cols) return;
    x[i] = c3_f2bf(c3_bf2f(x[i]) + c3_bf2f(v[i % cols]));
}

/* fp32 latent [C][T][H][W] -> bf16 patch tokens [T*HP*WP][p*p*C], feature order (p, q, c); rows beyond H/W are zero padded */
__global__ void c3_patchify(const float *lat, unsigned short *out, int C, int T, int H, int W, int HP, int WP) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int F = 4 * C;
    long long tot = (long long)T * HP * WP * F;
    if (i >= tot) return;
    int f = (int)(i % F); long long tok = i / F;
    int c = f % C, pq = f / C, p = pq >> 1, q = pq & 1;
    int wp = (int)(tok % WP); long long r = tok / WP; int hp = (int)(r % HP); int t = (int)(r / HP);
    int h = hp * 2 + p, w = wp * 2 + q;
    float v = 0.f;
    if (h < H && w < W) v = lat[(((long long)c * T + t) * H + h) * W + w];
    out[i] = c3_f2bf(v);
}

/* patches [(T-1)*HP*WP][4C] bf16 (noisy frames 1..T-1) -> fp32 latent [C][T][H][W], frame 0 = 0 (conditioning frame masked) */
__global__ void c3_unpatchify(const unsigned short *pp, float *lat, int C, int T, int H, int W, int HP, int WP) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tot = (long long)C * T * H * W;
    if (i >= tot) return;
    int w = (int)(i % W); long long r = i / W; int h = (int)(r % H); r /= H; int t = (int)(r % T); int c = (int)(r / T);
    if (t == 0) { lat[i] = 0.f; return; }
    int tn = t - 1, hp = h >> 1, p = h & 1, wp = w >> 1, q = w & 1;
    lat[i] = c3_bf2f(pp[(((long long)tn * HP + hp) * WP + wp) * (4 * C) + (p * 2 + q) * C + c]);
}

/* fp32 -> bf16 */
__global__ void c3_f2bf_k(const float *x, unsigned short *y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = c3_f2bf(x[i]);
}
__global__ void c3_bf2f_k(const unsigned short *x, float *y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = c3_bf2f(x[i]);
}

/* gen attention / und attention. Q [nq][16*128] (ldq), K/V [nk][8*128] (ldk), GQA: head h uses kv head h/2.
 * causal: key j allowed for query i iff j <= i (+ koff). One block (128 thr) per (query, head); dynamic smem = nk floats. */
__global__ void __launch_bounds__(128) c3_attn(const unsigned short *Q, long long ldq, const unsigned short *K, const unsigned short *V, long long ldk,
                                               unsigned short *O, long long ldo, int nq, int nk, int causal, int groups, float scale) {
    extern __shared__ float sc[];
    __shared__ float red[8];
    int qi = blockIdx.x, h = blockIdx.y, kvh = h / groups;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const unsigned short *q = Q + (long long)qi * ldq + h * 128;
    float qv[4];
    for (int j = 0; j < 4; j++) qv[j] = c3_bf2f(q[lane * 4 + j]);
    int kmax = causal ? (qi + 1) : nk;
    float mx = -3.0e38f;
    for (int k = w; k < kmax; k += 4) {
        const unsigned short *kp = K + (long long)k * ldk + kvh * 128 + lane * 4;
        float s = 0.f;
        for (int j = 0; j < 4; j++) s += qv[j] * c3_bf2f(kp[j]);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        s *= scale;
        if (lane == 0) sc[k] = s;
        mx = fmaxf(mx, s);
    }
    if (lane == 0) red[w] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    float sum = 0.f;
    for (int k = threadIdx.x; k < kmax; k += 128) { float e = expf(sc[k] - mx); sc[k] = e; sum += e; }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) red[4 + w] = sum;
    __syncthreads();
    float tot = red[4] + red[5] + red[6] + red[7];
    float inv = 1.0f / tot;
    float acc = 0.f;
    const unsigned short *vp = V + kvh * 128 + threadIdx.x;
    for (int k = 0; k < kmax; k++) acc += sc[k] * c3_bf2f(vp[(long long)k * ldk]);
    O[(long long)qi * ldo + h * 128 + threadIdx.x] = c3_f2bf(acc * inv);
}

}
);
#endif
