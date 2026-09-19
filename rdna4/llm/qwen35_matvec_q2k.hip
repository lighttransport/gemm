/* Q2_K x Q8_1 MMVQ arithmetic from llama.cpp 1859b5209, gfx1201.
 * Copyright (c) 2023-2026 The ggml authors, MIT license (see
 * reference_sampler.cpp for the complete retained license).
 * Compile without fast-math. One 256-thread block computes one output row,
 * matching the reference's eight-warp reduction order. */
extern "C" __global__ void qwen35_quantize_q81(
        signed char *q, float *scales, const float *x, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i >= n) return;
    float v = x[i], a = fabsf(v);
    for (int offset = 16; offset; offset >>= 1)
        a = fmaxf(a, __shfl_xor(a, offset, 32));
    float d = a / 127.0f;
    q[i] = a == 0.0f ? 0 : (signed char)roundf(v / d);
    if (threadIdx.x == 0) scales[blockIdx.x] = (float)(_Float16)d;
}

extern "C" __global__ void qwen35_matvec_q2k(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (row >= rows) return;
    int blocks = cols / 256, iqs = tid & 15;
    const unsigned char *w = weights + (size_t)row * blocks * 84;
    float sum = 0.0f;
    for (int b = tid / 16; b < blocks; b += 16) {
        const unsigned char *p = w + b * 84;
        const unsigned char *s = p + (iqs & ~7) + (iqs & 7) / 4;
        int v = ((const int *)(p + 16))[iqs];
        float sd = 0.0f, sm = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qb = b * 8 + 4 * (iqs / 8) + j;
            int u = ((const int *)(q + qb * 32))[iqs & 7];
            int z = __builtin_amdgcn_sudot4(true, (v >> (2*j)) & 0x03030303,
                                          true, u, 0, false);
            sd += scales[qb] * (z * (s[2*j] & 15));
            int m = s[2*j] >> 4;
            m |= m << 8; m |= m << 16;
            int zm = __builtin_amdgcn_sudot4(true, m, true, u, 0, false);
            sm += scales[qb] * zm;
        }
        float d = (float)*(const _Float16 *)(p + 80);
        float dm = (float)*(const _Float16 *)(p + 82);
        sum += d * sd - dm * sm;
    }
    __shared__ float partial[7][32];
    if (warp > 0) partial[warp - 1][lane] = sum;
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (int widx = 0; widx < 7; ++widx) sum += partial[widx][lane];
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}
