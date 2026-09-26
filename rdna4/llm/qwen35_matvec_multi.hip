/* One-token native IQ/Q2_K projections that share one Q8_1 activation,
 * issued as a single launch.  Blocks are partitioned into up to three
 * segments; each segment runs a verbatim copy of its format's standalone
 * kernel body (same codebook staging, unroll, per-lane order and warp
 * reduction), so every row is bitwise identical to the separate launches.
 * Generated from qwen35_matvec_iq.hip / qwen35_matvec_q2k.hip: keep in sync
 * (test_matvec_multi checks it).  Compiled after both sources. */

__device__ __forceinline__ void qwen35_mvseg_iq2xxs(int bx,
        
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    __shared__ unsigned long long grid_cache[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        grid_cache[i] = qwen35_iq2xxs_grid[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int token = 0;
    int row = bx * (blockDim.x / 32) + threadIdx.x / 32;
    q += (size_t)token * cols;
    scales += (size_t)token * (cols / 32);
    out += (size_t)token * rows;
    if (row >= rows) return;
    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)row * blocks * 66;
    float sum = 0.0f;
    /* Unrolled so four sub-blocks of loads are in flight (same per-lane
     * arithmetic order; ~10% on the 17408x5120 decode shape). */
#pragma unroll 4
    for (int qb = lane; qb < blocks * 8; qb += 32) {
        int b = qb / 8, ib = qb & 7;
        const unsigned char *p = w + b * 66;
        const int *u = (const int *)(q + qb * 32);
        unsigned int aux = *(const unsigned int *)(p + 6 + ib * 8);
        int z = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned long long grid = grid_cache[p[2 + ib * 8 + j]];
            unsigned int signs = qwen35_parity_signs((aux >> (7*j)) & 127);
            int lo = qwen35_signed_grid((unsigned int)grid, signs, 0);
            int hi = qwen35_signed_grid((unsigned int)(grid >> 32), signs, 4);
            z = qwen35_dot4(lo, u[2*j], z);
            z = qwen35_dot4(hi, u[2*j+1], z);
        }
        z = z * (int)((aux >> 27) | 1) / 8;
        float d = (float)*(const _Float16 *)p * scales[qb];
        sum += d * z;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}


__device__ __forceinline__ void qwen35_mvseg_iq2xs(int bx,
        
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    __shared__ unsigned long long grid_cache[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x)
        grid_cache[i] = qwen35_iq2xs_grid[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int token = 0;
    int row = bx * (blockDim.x / 32) + threadIdx.x / 32;
    q += (size_t)token * cols;
    scales += (size_t)token * (cols / 32);
    out += (size_t)token * rows;
    if (row >= rows) return;
    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)row * blocks * 74;
    float sum = 0.0f;
    for (int qb = lane; qb < blocks * 8; qb += 32) {
        int b = qb / 8, ib = qb & 7;
        const unsigned char *p = w + b * 74;
        const int *u = (const int *)(q + qb * 32);
        int z0 = 0, z1 = 0;
        unsigned int sc = p[66 + ib];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int code = ((const unsigned short *)(p + 2))[ib * 4 + j];
            unsigned long long grid = grid_cache[code & 511];
            unsigned int signs = qwen35_parity_signs(code >> 9);
            int lo = qwen35_signed_grid((unsigned int)grid, signs, 0);
            int hi = qwen35_signed_grid((unsigned int)(grid >> 32), signs, 4);
            if (j < 2) { z0 = qwen35_dot4(lo, u[2*j], z0); z0 = qwen35_dot4(hi, u[2*j+1], z0); }
            else { z1 = qwen35_dot4(lo, u[2*j], z1); z1 = qwen35_dot4(hi, u[2*j+1], z1); }
        }
        int z = (z0 * (int)(sc & 15) + z1 * (int)(sc >> 4) + (z0 + z1)/2)/4;
        float d = (float)*(const _Float16 *)p * scales[qb];
        sum += d * z;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}


__device__ __forceinline__ void qwen35_mvseg_iq2s(int bx,
        
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    int lane = threadIdx.x & 31;
    int token = 0;
    int row = bx * (blockDim.x / 32) + threadIdx.x / 32;
    q += (size_t)token * cols;
    scales += (size_t)token * (cols / 32);
    out += (size_t)token * rows;
    if (row >= rows) return;
    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)row * blocks * 82;
    float sum = 0.0f;
    for (int qb = lane; qb < blocks * 8; qb += 32) {
        int b = qb / 8, ib = qb & 7;
        const unsigned char *p = w + b * 82;
        const int *u = (const int *)(q + qb * 32);
        int z0 = 0, z1 = 0;
        unsigned int qh = p[66 + ib], sc = p[74 + ib];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int index = p[2 + ib * 4 + j] | ((qh << (8 - 2*j)) & 0x300);
            unsigned long long grid = qwen35_iq2s_grid[index];
            unsigned int signs = p[34 + ib * 4 + j];
            int lo = qwen35_signed_grid((unsigned int)grid, signs, 0);
            int hi = qwen35_signed_grid((unsigned int)(grid >> 32), signs, 4);
            if (j < 2) { z0 = qwen35_dot4(lo, u[2*j], z0); z0 = qwen35_dot4(hi, u[2*j+1], z0); }
            else { z1 = qwen35_dot4(lo, u[2*j], z1); z1 = qwen35_dot4(hi, u[2*j+1], z1); }
        }
        int z = (z0 * (int)(sc & 15) + z1 * (int)(sc >> 4) + (z0 + z1)/2)/4;
        float d = (float)*(const _Float16 *)p * scales[qb];
        sum += d * z;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}


__device__ __forceinline__ void qwen35_mvseg_iq3xxs(int bx,
        
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    __shared__ unsigned int grid_cache[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        grid_cache[i] = qwen35_iq3xxs_grid[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int token = 0;
    int row = bx * (blockDim.x / 32) + threadIdx.x / 32;
    q += (size_t)token * cols;
    scales += (size_t)token * (cols / 32);
    out += (size_t)token * rows;
    if (row >= rows) return;
    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)row * blocks * 98;
    float sum = 0.0f;
    for (int qb = lane; qb < blocks * 8; qb += 32) {
        int b = qb / 8, ib = qb & 7;
        const unsigned char *p = w + b * 98;
        const int *u = (const int *)(q + qb * 32);
        unsigned int aux = *(const unsigned int *)(p + 66 + ib * 4);
        int z = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int signs = qwen35_parity_signs((aux >> (7*j)) & 127);
            int lo = qwen35_signed_grid(grid_cache[p[2 + ib*8 + 2*j]], signs, 0);
            int hi = qwen35_signed_grid(grid_cache[p[3 + ib*8 + 2*j]], signs, 4);
            z = qwen35_dot4(lo, u[2*j], z);
            z = qwen35_dot4(hi, u[2*j+1], z);
        }
        z = ((int)(aux >> 28) * z + z/2)/2;
        float d = (float)*(const _Float16 *)p * scales[qb];
        sum += d * z;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}


__device__ __forceinline__ void qwen35_mvseg_iq3s(int bx,
        
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    int lane = threadIdx.x & 31;
    int token = 0;
    int row = bx * (blockDim.x / 32) + threadIdx.x / 32;
    q += (size_t)token * cols;
    scales += (size_t)token * (cols / 32);
    out += (size_t)token * rows;
    if (row >= rows) return;
    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)row * blocks * 110;
    float sum = 0.0f;
    for (int qb = lane; qb < blocks * 8; qb += 32) {
        int b = qb / 8, ib = qb & 7;
        const unsigned char *p = w + b * 110;
        const int *u = (const int *)(q + qb * 32);
        unsigned int qh = p[66 + ib];
        unsigned long long codes = *(const unsigned long long *)(p + 2 + ib * 8);
        unsigned int packed_signs = *(const unsigned int *)(p + 74 + ib * 4);
        int z = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int signs = (packed_signs >> (8*j)) & 255;
            int il = ((codes >> (16*j)) & 255) | ((qh << (8 - 2*j)) & 0x100);
            int ih = ((codes >> (16*j+8)) & 255) | ((qh << (7 - 2*j)) & 0x100);
            int lo = qwen35_signed_grid(qwen35_iq3s_grid[il], signs, 0);
            int hi = qwen35_signed_grid(qwen35_iq3s_grid[ih], signs, 4);
            z = qwen35_dot4(lo, u[2*j], z);
            z = qwen35_dot4(hi, u[2*j+1], z);
        }
        z *= 1 + 2 * ((p[106 + ib/2] >> ((ib & 1) * 4)) & 15);
        float d = (float)*(const _Float16 *)p * scales[qb];
        sum += d * z;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}


__device__ __forceinline__ void qwen35_mvseg_iq4xs(int bx,
        
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    int lane = threadIdx.x & 31;
    int token = 0;
    int row = bx * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    q += (size_t)token * cols;
    scales += (size_t)token * (cols / 32);
    out += (size_t)token * rows;
    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)row * blocks * 136;
    float sum = 0.0f;
    // Keep the reference's eight independent warp sums and addition order.
#pragma unroll
    for (int warp = 0; warp < 8; ++warp) {
        float partial = 0.0f;
        for (int qb = lane + 32*warp; qb < blocks*8; qb += 256) {
            int ib = qb & 7;
            const unsigned char *p = w + (qb/8)*136;
            const int *u = (const int *)(q+qb*32);
            const unsigned int *v = (const unsigned int *)(p+8+ib*16);
            int z = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                z = qwen35_dot4(qwen35_iq4_lookup(v[j]), u[j], z);
                z = qwen35_dot4(qwen35_iq4_lookup(v[j] >> 4), u[j+4], z);
            }
            int ls = ((p[4+ib/2] >> (4*(ib&1))) & 15) |
                (((*(const unsigned short *)(p+2) >> (2*ib)) & 3) << 4);
            z *= ls-32;
            float d = (float)*(const _Float16 *)p * scales[qb];
            partial += d*z;
        }
        sum += partial;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}


__device__ __forceinline__ void qwen35_mvseg_q2k_rows(int bx,
        
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = bx * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    int blocks = cols / 256, iqs = lane & 15;
    const unsigned char *w = weights + (size_t)row * blocks * 84;
    float parts[8] = {};
#pragma unroll
    for (int vw = 0; vw < 8; ++vw) {
        for (int b = lane / 16 + 2 * vw; b < blocks; b += 16) {
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
            parts[vw] += d * sd - dm * sm;
        }
    }
    float sum = parts[0];
#pragma unroll
    for (int vw = 1; vw < 8; ++vw) sum += parts[vw];
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[row] = sum;
}


__device__ __forceinline__ void qwen35_mvseg_q2k(int bx,
        
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int row = bx, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
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


/* kind: 0 IQ2_XXS, 1 IQ2_XS, 2 IQ2_S, 3 IQ3_XXS, 4 IQ3_S, 5 IQ4_XS,
 * 6 Q2_K warp-per-row, 7 Q2_K block-per-row.  Segment block counts are
 * ceil(rows / 8) (kinds 0-6) or rows (kind 7) for 256-thread blocks. */
__device__ __forceinline__ void qwen35_mvseg(int kind, int bx, float *out,
        const unsigned char *w, const signed char *q, const float *s,
        int rows, int cols) {
    switch (kind) {
    case 0: qwen35_mvseg_iq2xxs(bx, out, w, q, s, rows, cols); break;
    case 1: qwen35_mvseg_iq2xs(bx, out, w, q, s, rows, cols); break;
    case 2: qwen35_mvseg_iq2s(bx, out, w, q, s, rows, cols); break;
    case 3: qwen35_mvseg_iq3xxs(bx, out, w, q, s, rows, cols); break;
    case 4: qwen35_mvseg_iq3s(bx, out, w, q, s, rows, cols); break;
    case 5: qwen35_mvseg_iq4xs(bx, out, w, q, s, rows, cols); break;
    case 6: qwen35_mvseg_q2k_rows(bx, out, w, q, s, rows, cols); break;
    default: qwen35_mvseg_q2k(bx, out, w, q, s, rows, cols); break;
    }
}

extern "C" __global__ __launch_bounds__(256) void qwen35_matvec_native_multi(
        float *o0, const unsigned char *w0, int r0, int k0, int nb0,
        float *o1, const unsigned char *w1, int r1, int k1, int nb1,
        float *o2, const unsigned char *w2, int r2, int k2, int nb2,
        const signed char *__restrict__ q, const float *__restrict__ s,
        int cols) {
    int b = blockIdx.x;
    if (b < nb0) { qwen35_mvseg(k0, b, o0, w0, q, s, r0, cols); return; }
    b -= nb0;
    if (b < nb1) { qwen35_mvseg(k1, b, o1, w1, q, s, r1, cols); return; }
    b -= nb1;
    if (b < nb2) qwen35_mvseg(k2, b, o2, w2, q, s, r2, cols);
}

/* Two-segment form specialized on both formats, so a pair only carries its
 * own bodies' register and LDS footprint. */
template <int K0, int K1>
__device__ __forceinline__ void qwen35_mvpair(
        float *o0, const unsigned char *w0, int r0, int nb0,
        float *o1, const unsigned char *w1, int r1,
        const signed char *q, const float *s, int cols) {
    int b = blockIdx.x;
    if (b < nb0) qwen35_mvseg(K0, b, o0, w0, q, s, r0, cols);
    else qwen35_mvseg(K1, b - nb0, o1, w1, q, s, r1, cols);
}
#define QWEN35_MVPAIR(A, B) \
extern "C" __global__ __launch_bounds__(256) void qwen35_matvec_pair_##A##_##B( \
        float *o0, const unsigned char *w0, int r0, int nb0, \
        float *o1, const unsigned char *w1, int r1, \
        const signed char *__restrict__ q, const float *__restrict__ s, int cols) { \
    qwen35_mvpair<A, B>(o0, w0, r0, nb0, o1, w1, r1, q, s, cols); \
}
#define QWEN35_MVPAIR_ROW(A) QWEN35_MVPAIR(A, 0) QWEN35_MVPAIR(A, 1) \
    QWEN35_MVPAIR(A, 2) QWEN35_MVPAIR(A, 3) QWEN35_MVPAIR(A, 4) \
    QWEN35_MVPAIR(A, 5) QWEN35_MVPAIR(A, 6) QWEN35_MVPAIR(A, 7)
QWEN35_MVPAIR_ROW(0) QWEN35_MVPAIR_ROW(1) QWEN35_MVPAIR_ROW(2)
QWEN35_MVPAIR_ROW(3) QWEN35_MVPAIR_ROW(4) QWEN35_MVPAIR_ROW(5)
QWEN35_MVPAIR_ROW(6) QWEN35_MVPAIR_ROW(7)
