// Native FLUX 3 kernels. BF16 storage/rounding and row-scaled E4M3FN.
// SPDX-License-Identifier: MIT
typedef unsigned short u16;
typedef u16 b8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
__device__ float bf(float x) {
    unsigned u = __float_as_uint(x);
    return __uint_as_float((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u);
}
__device__ float frombf(u16 x) {
    return __uint_as_float((unsigned)x << 16);
}
__device__ float fp8val(unsigned x) {
    unsigned e = (x >> 3) & 15, m = x & 7;
    float v = e ? ldexpf(1.f + m / 8.f, (int)e - 7) : m / 512.f;
    return (x & 128) ? -v : v;
}
__device__ float fp8round(float x) {
    // Exact nearest-even E4M3FN, including subnormals and exponent 15.
    float a = fminf(fabsf(x), 448.f);
    int e;
    frexpf(a, &e);
    float step = ldexpf(1.f, max(-9, e - 4));
    float v = nearbyintf(a / step) * step;
    return copysignf(fminf(v, 448.f), x);
}
extern "C" {
__global__ void prepare(float *y, float *scale, const float *x, int k, int quant) {
    __shared__ float s[256];
    int row = blockIdx.x, t = threadIdx.x;
    float m = 0;
    for (int j = t; j < k; j += 256)
        m = fmaxf(m, fabsf(bf(x[(long)row * k + j])));
    s[t] = m;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (t < d)
            s[t] = fmaxf(s[t], s[t + d]);
        __syncthreads();
    }
    float a = quant ? fmaxf(s[0] / 448.f, 1e-12f) : 1.f;
    if (t == 0)
        scale[row] = a;
    for (int j = t; j < k; j += 256) {
        float v = bf(x[(long)row * k + j]);
        y[(long)row * k + j] = quant ? fp8round(v / a) : v;
    }
}
// Wave32: A and B row-major, 16x16 tiles, four independent waves.
__global__ void linear(float *y, const void *w, const float *ws, const float *x, const float *xs,
                       int n, int k, int m, int quant) {
    int lane = threadIdx.x & 31, wave = threadIdx.x / 32;
    int r0 = blockIdx.y * 64 + wave * 16, c0 = blockIdx.x * 16;
    f8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int p = 0; p < k; p += 16) {
        b8 a, b;
        int r = r0 + (lane & 15), c = c0 + (lane & 15);
        for (int i = 0; i < 8; i++) {
            int j = p + (lane >> 4) * 8 + i;
            float av = (r < m && j < k) ? x[(long)r * k + j] : 0.f;
            float bv = 0;
            if (c < n && j < k)
                bv = quant ? fp8val(((const unsigned char *)w)[(long)c * k + j])
                           : frombf(((const u16 *)w)[(long)c * k + j]);
            a[i] = (u16)(__float_as_uint(av) >> 16);
            b[i] = (u16)(__float_as_uint(bv) >> 16);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
    }
    for (int i = 0; i < 8; i++) {
        int r = r0 + (lane >> 4) * 8 + i, c = c0 + (lane & 15);
        if (r < m && c < n)
            y[(long)r * n + c] = bf(acc[i] * (quant ? xs[r] * ws[c] : 1.f));
    }
}
__global__ void round_bf(float *x, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        x[i] = bf(x[i]);
}
__global__ void silu(float *y, const float *x, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        y[i] = bf(x[i] / (1.f + expf(-x[i])));
}
__global__ void add_ref(float *x, const float *ref, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        x[i] = bf(x[i] + ref[i % 3072]);
}
__global__ void lnmod(float *y, const float *x, const float *mod, const int *mode, int n) {
    __shared__ float s[256];
    int r = blockIdx.x, t = threadIdx.x;
    const int h = 3072;
    float sum = 0;
    for (int j = t; j < h; j += 256)
        sum += x[(long)r * h + j];
    s[t] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (t < d)
            s[t] += s[t + d];
        __syncthreads();
    }
    float mean = s[0] / h;
    __syncthreads();
    sum = 0;
    for (int j = t; j < h; j += 256) {
        float v = x[(long)r * h + j] - mean;
        sum += v * v;
    }
    s[t] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (t < d)
            s[t] += s[t + d];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / h + 1e-6f);
    int base = mode[r] * 3 * h;
    for (int j = t; j < h; j += 256) {
        float z = bf((x[(long)r * h + j] - mean) * inv);
        y[(long)r * h + j] = bf(bf(z * bf(1.f + mod[base + h + j])) + mod[base + j]);
    }
    (void)n;
}
__global__ void normrope(float *x, const float *w, const float *ids, int n) {
    __shared__ float s[128];
    int h = blockIdx.x % 24, r = blockIdx.x / 24, t = threadIdx.x;
    long off = (long)r * 3072 + h * 128;
    float v = x[off + t];
    s[t] = v * v;
    __syncthreads();
    for (int d = 64; d; d /= 2) {
        if (t < d)
            s[t] += s[t + d];
        __syncthreads();
    }
    float inv = rsqrtf(s[0] / 128 + 1e-6f);
    __syncthreads();
    s[t] = v * inv * w[t];
    __syncthreads();
    int p = t & ~1, axis = t / 32;
    double a = (double)ids[r * 4 + axis] * pow(10000.0, -(double)(p % 32) / 32.0);
    float c = (float)cos(a), sn = (float)sin(a);
    x[off + t] = bf((t & 1) ? s[p] * sn + s[p + 1] * c : s[p] * c - s[p + 1] * sn);
    (void)n;
}
__global__ void swiglu(float *y, const float *x, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n * 9216) {
        int r = i / 9216, j = i % 9216;
        float a = x[(long)r * 18432 + j];
        y[i] = bf(bf(a / (1.f + expf(-a))) * x[(long)r * 18432 + 9216 + j]);
    }
}
__global__ void residual(float *x, const float *a, const float *b, const float *mod,
                         const int *mode, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n * 3072) {
        int r = i / 3072, j = i % 3072;
        float gate = mod[mode[r] * 9216 + 6144 + j];
        x[i] = bf(x[i] + bf(gate * bf(a[i] + b[i])));
    }
}
// Online softmax, one wave per query/head. No quadratic attention allocation.
__global__ void attention(float *out, const float *Q, const float *K, const float *V, int n) {
    __shared__ float sk[32 * 128], sv[32 * 128];
    int h = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32,
        qi = blockIdx.y * 4 + warp;
    float q[4], o[4] = {0, 0, 0, 0};
    for (int e = 0; e < 4; e++)
        q[e] = qi < n ? Q[(long)qi * 3072 + h * 128 + lane * 4 + e] : 0;
    float mx = -1e30f, den = 0;
    for (int start = 0; start < n; start += 32) {
        for (int j = threadIdx.x; j < 4096; j += 128) {
            int r = start + j / 128, c = j % 128;
            sk[j] = r < n ? K[(long)r * 3072 + h * 128 + c] : 0;
            sv[j] = r < n ? V[(long)r * 3072 + h * 128 + c] : 0;
        }
        __syncthreads();
        for (int k = 0; k < 32 && start + k < n; k++) {
            float dot = 0;
            for (int e = 0; e < 4; e++)
                dot += q[e] * sk[k * 128 + lane * 4 + e];
            for (int d = 16; d; d /= 2)
                dot += __shfl_xor(dot, d);
            float score = dot * 0.08838834764831845f, newmx = fmaxf(mx, score),
                  a = expf(mx - newmx), p = expf(score - newmx);
            den = den * a + p;
            for (int e = 0; e < 4; e++)
                o[e] = o[e] * a + p * sv[k * 128 + lane * 4 + e];
            mx = newmx;
        }
        __syncthreads();
    }
    if (qi < n)
        for (int e = 0; e < 4; e++)
            out[(long)qi * 3072 + h * 128 + lane * 4 + e] = bf(o[e] / den);
}
}
