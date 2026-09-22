/* IQ MMVQ arithmetic from llama.cpp 1859b5209, gfx1201.
 * Copyright (c) 2023-2026 The ggml authors, MIT license (see
 * reference_sampler.cpp for the complete retained license).
 * Compile without fast-math. Each warp owns one output row. */
__device__ static const unsigned long long qwen35_iq2xxs_grid[256] = {
#include "qwen35_iq2xxs_grid.inc"
};
__device__ static const unsigned long long qwen35_iq2xs_grid[512] = {
#include "qwen35_iq2xs_grid.inc"
};
__device__ static const unsigned long long qwen35_iq2s_grid[1024] = {
#include "../../common/iq2s_grid.inc"
};
__device__ static const unsigned int qwen35_iq3xxs_grid[256] = {
#include "../../common/iq3xxs_grid.inc"
};
__device__ static const unsigned int qwen35_iq3s_grid[512] = {
#include "../../common/iq3s_grid.inc"
};
__device__ __forceinline__ int qwen35_signed_grid(unsigned int grid, unsigned int signs, int shift) {
    // Spread four sign bits into byte LSBs, then expand each to 0xff.
    // This removes a dependent table load from every grid lookup.
    unsigned int mask = ((((signs >> shift) & 15) * 0x00204081u) & 0x01010101u) * 255u;
    // These IQ grids contain strictly positive magnitudes, so the addition
    // cannot carry across byte boundaries.
    return (int)((grid ^ mask) + (mask & 0x01010101u));
}
__device__ __forceinline__ int qwen35_dot4(int a, int b, int sum) {
    return __builtin_amdgcn_sudot4(true, a, true, b, sum, false);
}

__device__ __forceinline__ unsigned int qwen35_parity_signs(unsigned int code) {
    return code | ((__builtin_popcount(code) & 1u) << 7);
}

__device__ __forceinline__ unsigned int qwen35_iq4_lookup(unsigned int indices) {
    unsigned int lo = __builtin_amdgcn_perm(0xf6eaddcfu, 0xbfad9881u, indices & 0x07070707u);
    unsigned int hi = __builtin_amdgcn_perm(0x71594535u, 0x26190d01u, indices & 0x07070707u);
    return __builtin_amdgcn_perm(hi, lo, 0x03020100u | ((indices & 0x08080808u) >> 1));
}

extern "C" __global__ void qwen35_matvec_iq4xs(
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    int lane = threadIdx.x & 31;
    int linear = blockIdx.y * gridDim.x + blockIdx.x;
    int token = linear % gridDim.y;
    int row = (linear / gridDim.y) * (blockDim.x / 32) + threadIdx.x / 32;
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

extern "C" __global__ void qwen35_matvec_iq2xxs(
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    __shared__ unsigned long long grid_cache[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        grid_cache[i] = qwen35_iq2xxs_grid[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int linear = blockIdx.y * gridDim.x + blockIdx.x;
    int token = linear % gridDim.y;
    int row = (linear / gridDim.y) * (blockDim.x / 32) + threadIdx.x / 32;
    q += (size_t)token * cols;
    scales += (size_t)token * (cols / 32);
    out += (size_t)token * rows;
    if (row >= rows) return;
    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)row * blocks * 66;
    float sum = 0.0f;
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

extern "C" __global__ __launch_bounds__(512, 1) void qwen35_matvec_iq2xs(
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    __shared__ unsigned long long grid_cache[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x)
        grid_cache[i] = qwen35_iq2xs_grid[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int linear = blockIdx.y * gridDim.x + blockIdx.x;
    int token = linear % gridDim.y;
    int row = (linear / gridDim.y) * (blockDim.x / 32) + threadIdx.x / 32;
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

/* Diagnostic ordinary-decode path: add the IQ2_XS down projection directly
 * to the residual.  The dot, per-block scale, codebook, and warp reduction
 * order match qwen35_matvec_iq2xs; only the final store is combined with the
 * already-resident residual vector. */
extern "C" __global__ __launch_bounds__(512, 1)
void qwen35_matvec_iq2xs_down_residual(
        float *__restrict__ residual, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    __shared__ unsigned long long grid_cache[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x)
        grid_cache[i] = qwen35_iq2xs_grid[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (blockDim.x / 32)) + threadIdx.x / 32;
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
            if (j < 2) {
                z0 = qwen35_dot4(lo, u[2*j], z0);
                z0 = qwen35_dot4(hi, u[2*j+1], z0);
            } else {
                z1 = qwen35_dot4(lo, u[2*j], z1);
                z1 = qwen35_dot4(hi, u[2*j+1], z1);
            }
        }
        int z = (z0 * (int)(sc & 15) + z1 * (int)(sc >> 4) +
                 (z0 + z1) / 2) / 4;
        float d = (float)*(const _Float16 *)p * scales[qb];
        sum += d * z;
    }
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) residual[row] = residual[row] + sum;
}

extern "C" __global__ void qwen35_matvec_iq2s(
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    int lane = threadIdx.x & 31;
    int linear = blockIdx.y * gridDim.x + blockIdx.x;
    int token = linear % gridDim.y;
    int row = (linear / gridDim.y) * (blockDim.x / 32) + threadIdx.x / 32;
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

extern "C" __global__ void qwen35_matvec_iq3xxs(
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    __shared__ unsigned int grid_cache[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        grid_cache[i] = qwen35_iq3xxs_grid[i];
    __syncthreads();
    int lane = threadIdx.x & 31;
    int linear = blockIdx.y * gridDim.x + blockIdx.x;
    int token = linear % gridDim.y;
    int row = (linear / gridDim.y) * (blockDim.x / 32) + threadIdx.x / 32;
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

/* Fused one-token Q/K/V projection for the native IQ3_XXS path.  The three
 * matrices still have independent output row ranges and therefore retain the
 * exact scalar row arithmetic above; a single block grid shares codebook
 * staging and the one activation-quantization launch across all three.
 * Keep this candidate opt-in until resident-device hashes and throughput are
 * measured against the three-launch reference. */
extern "C" __global__ void qwen35_matvec_iq3xxs_qkv(
        float *__restrict__ qout, float *__restrict__ kout,
        float *__restrict__ vout,
        const unsigned char *__restrict__ qw,
        const unsigned char *__restrict__ kw,
        const unsigned char *__restrict__ vw,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int qrows, int krows, int vrows, int cols) {
    __shared__ unsigned int grid_cache[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        grid_cache[i] = qwen35_iq3xxs_grid[i];
    __syncthreads();

    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int total_rows = qrows + krows + vrows;
    if (row >= total_rows) return;

    float *out;
    const unsigned char *weights;
    int local_row;
    if (row < qrows) {
        out = qout;
        weights = qw;
        local_row = row;
    } else if (row < qrows + krows) {
        out = kout;
        weights = kw;
        local_row = row - qrows;
    } else {
        out = vout;
        weights = vw;
        local_row = row - qrows - krows;
    }

    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)local_row * blocks * 98;
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
        z = ((int)(aux >> 28) * z + z/2) / 2;
        float d = (float)*(const _Float16 *)p * scales[qb];
        sum += d * z;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) out[local_row] = sum;
}

extern "C" __global__ void qwen35_matvec_iq3s(
        float *__restrict__ out, const unsigned char *__restrict__ weights,
        const signed char *__restrict__ q, const float *__restrict__ scales,
        int rows, int cols) {
    int lane = threadIdx.x & 31;
    int linear = blockIdx.y * gridDim.x + blockIdx.x;
    int token = linear % gridDim.y;
    int row = (linear / gridDim.y) * (blockDim.x / 32) + threadIdx.x / 32;
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

/* Decode each IQ grid once for several speculative activation rows.  DFlash2
 * verifies five or eight rows at a time, so keep a wider specialization in
 * addition to the lower-register four-row kernel used by dense NextN. */
template<int MAX_COUNT>
__device__ __forceinline__ void qwen35_matvec_iq_multi(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int kind, int count) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (row >= rows) return;
    int bs = kind==0 ? 66 : kind==1 ? 74 : kind==2 ? 82 : kind==3 ? 98 : 110;
    const unsigned char *w = weights+(size_t)row*(cols/256)*bs;
    float sum[MAX_COUNT] = {};
    for (int qb = lane; qb < cols/32; qb += 32) {
        int ib = qb&7;
        const unsigned char *p = w+(qb/8)*bs;
        unsigned int aux = kind==0 ? *(const unsigned int *)(p+6+ib*8) :
            kind==3 ? *(const unsigned int *)(p+66+ib*4) : 0;
        int z0[MAX_COUNT] = {}, z1[MAX_COUNT] = {};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int lo, hi, signs;
            if (kind <= 2) {
                unsigned long long grid;
                if (kind==0) {
                    grid=qwen35_iq2xxs_grid[p[2+ib*8+j]];
                    signs=qwen35_parity_signs((aux>>(7*j))&127);
                } else if (kind==1) {
                    unsigned int code=((const unsigned short *)(p+2))[ib*4+j];
                    grid=qwen35_iq2xs_grid[code&511];
                    signs=qwen35_parity_signs(code>>9);
                } else {
                    int index=p[2+ib*4+j] | ((p[66+ib]<<(8-2*j))&0x300);
                    grid=qwen35_iq2s_grid[index];
                    signs=p[34+ib*4+j];
                }
                lo=(unsigned int)grid; hi=(unsigned int)(grid>>32);
            } else if (kind==3) {
                signs=qwen35_parity_signs((aux>>(7*j))&127);
                lo=qwen35_iq3xxs_grid[p[2+ib*8+2*j]];
                hi=qwen35_iq3xxs_grid[p[3+ib*8+2*j]];
            } else {
                unsigned int qh=p[66+ib];
                signs=p[74+ib*4+j];
                lo=qwen35_iq3s_grid[p[2+ib*8+2*j] | ((qh<<(8-2*j))&0x100)];
                hi=qwen35_iq3s_grid[p[3+ib*8+2*j] | ((qh<<(7-2*j))&0x100)];
            }
            int a=qwen35_signed_grid(lo, signs, 0), b=qwen35_signed_grid(hi, signs, 4);
#pragma unroll
            for (int m = 0; m < MAX_COUNT; ++m) if (m < count) {
                const int *u=(const int *)(q+(size_t)m*cols+qb*32);
                if ((kind==1 || kind==2) && j>=2) {
                    z1[m]=qwen35_dot4(a,u[2*j],z1[m]);
                    z1[m]=qwen35_dot4(b,u[2*j+1],z1[m]);
                } else {
                    z0[m]=qwen35_dot4(a,u[2*j],z0[m]);
                    z0[m]=qwen35_dot4(b,u[2*j+1],z0[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MAX_COUNT; ++m) if (m < count) {
            int z=z0[m];
            if (kind==0) z=z*(int)((aux>>27)|1)/8;
            else if (kind==1 || kind==2) {
                int sc=p[(kind==1?66:74)+ib];
                z=(z*(sc&15)+z1[m]*(sc>>4)+(z+z1[m])/2)/4;
            } else if (kind==3) z=((int)(aux>>28)*z+z/2)/2;
            else z*=1+2*((p[106+ib/2]>>((ib&1)*4))&15);
            float d=(float)*(const _Float16 *)p * scales[(size_t)m*(cols/32)+qb];
            sum[m]+=d*z;
        }
    }
#pragma unroll
    for (int m = 0; m < MAX_COUNT; ++m) if (m < count) {
        for (int offset=16; offset; offset>>=1) sum[m]+=__shfl_xor(sum[m],offset,32);
        if (lane==0) out[(size_t)m*rows+row]=sum[m];
    }
}

extern "C" __global__ void qwen35_matvec_iq_multi4(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int kind, int count) {
    qwen35_matvec_iq_multi<4>(out,weights,q,scales,rows,cols,kind,count);
}

/* Give the compiler a constant format for the eight-row verifier.  The
 * production variants discard unused codebook branches and their live
 * values. */
extern "C" __global__ void qwen35_matvec_iq2xxs_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    qwen35_matvec_iq_multi<8>(out,weights,q,scales,rows,cols,0,count);
}
extern "C" __global__ void qwen35_matvec_iq2xs_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    qwen35_matvec_iq_multi<8>(out,weights,q,scales,rows,cols,1,count);
}
extern "C" __global__ void qwen35_matvec_iq2s_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    qwen35_matvec_iq_multi<8>(out,weights,q,scales,rows,cols,2,count);
}
extern "C" __global__ void qwen35_matvec_iq3xxs_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    qwen35_matvec_iq_multi<8>(out,weights,q,scales,rows,cols,3,count);
}
extern "C" __global__ void qwen35_matvec_iq3s_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    qwen35_matvec_iq_multi<8>(out,weights,q,scales,rows,cols,4,count);
}

/* Decode the four grids first, then finish one activation row at a time.
 * This keeps only two dot accumulators live instead of sixteen while still
 * sharing every codebook lookup across all eight verifier rows. */
template<int KIND>
__device__ __forceinline__ void qwen35_matvec_iq_fixed8_compact(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int lane=threadIdx.x&31;
    int row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (row>=rows) return;
    constexpr int bs=KIND==0 ? 66 : KIND==1 ? 74 : KIND==2 ? 82 :
                     KIND==3 ? 98 : 110;
    const unsigned char *w=weights+(size_t)row*(cols/256)*bs;
    float sum[8]={};
    for (int qb=lane; qb<cols/32; qb+=32) {
        int ib=qb&7;
        const unsigned char *p=w+(qb/8)*bs;
        unsigned int aux=KIND==0 ? *(const unsigned int *)(p+6+ib*8) :
            KIND==3 ? *(const unsigned int *)(p+66+ib*4) : 0;
        int av[4], bv[4];
#pragma unroll
        for (int j=0; j<4; ++j) {
            unsigned int lo,hi,signs;
            if (KIND<=2) {
                unsigned long long grid;
                if (KIND==0) {
                    grid=qwen35_iq2xxs_grid[p[2+ib*8+j]];
                    signs=qwen35_parity_signs((aux>>(7*j))&127);
                } else if (KIND==1) {
                    unsigned int code=((const unsigned short *)(p+2))[ib*4+j];
                    grid=qwen35_iq2xs_grid[code&511];
                    signs=qwen35_parity_signs(code>>9);
                } else {
                    int index=p[2+ib*4+j] |
                        ((p[66+ib]<<(8-2*j))&0x300);
                    grid=qwen35_iq2s_grid[index];
                    signs=p[34+ib*4+j];
                }
                lo=(unsigned int)grid;
                hi=(unsigned int)(grid>>32);
            } else if (KIND==3) {
                signs=qwen35_parity_signs((aux>>(7*j))&127);
                lo=qwen35_iq3xxs_grid[p[2+ib*8+2*j]];
                hi=qwen35_iq3xxs_grid[p[3+ib*8+2*j]];
            } else {
                unsigned int qh=p[66+ib];
                signs=p[74+ib*4+j];
                lo=qwen35_iq3s_grid[p[2+ib*8+2*j] |
                    ((qh<<(8-2*j))&0x100)];
                hi=qwen35_iq3s_grid[p[3+ib*8+2*j] |
                    ((qh<<(7-2*j))&0x100)];
            }
            av[j]=qwen35_signed_grid(lo,signs,0);
            bv[j]=qwen35_signed_grid(hi,signs,4);
        }
#pragma unroll
        for (int m=0; m<8; ++m) {
            const int *u=(const int *)(q+(size_t)m*cols+qb*32);
            int z0=0,z1=0;
#pragma unroll
            for (int j=0; j<4; ++j) {
                if ((KIND==1 || KIND==2) && j>=2) {
                    z1=qwen35_dot4(av[j],u[2*j],z1);
                    z1=qwen35_dot4(bv[j],u[2*j+1],z1);
                } else {
                    z0=qwen35_dot4(av[j],u[2*j],z0);
                    z0=qwen35_dot4(bv[j],u[2*j+1],z0);
                }
            }
            int z=z0;
            if (KIND==0) z=z*(int)((aux>>27)|1)/8;
            else if (KIND==1 || KIND==2) {
                int sc=p[(KIND==1?66:74)+ib];
                z=(z*(sc&15)+z1*(sc>>4)+(z+z1)/2)/4;
            } else if (KIND==3) z=((int)(aux>>28)*z+z/2)/2;
            else z*=1+2*((p[106+ib/2]>>((ib&1)*4))&15);
            float d=(float)*(const _Float16 *)p *
                scales[(size_t)m*(cols/32)+qb];
            sum[m]+=d*z;
        }
    }
#pragma unroll
    for (int m=0; m<8; ++m) {
        for (int offset=16; offset; offset>>=1)
            sum[m]+=__shfl_xor(sum[m],offset,32);
        if (lane==0) out[(size_t)m*rows+row]=sum[m];
    }
}

/* K=7 target verification always evaluates eight rows.  Literal formats and
 * counts let HIPRTC discard unused branches and per-activation bounds. */
#define QWEN35_IQ_FIXED8(name, kind_value) \
extern "C" __global__ void name( \
        float *out, const unsigned char *weights, const signed char *q, \
        const float *scales, int rows, int cols, int count) { \
    (void)count; \
    qwen35_matvec_iq_fixed8_compact<kind_value>( \
        out,weights,q,scales,rows,cols); \
}
QWEN35_IQ_FIXED8(qwen35_matvec_iq2xxs_fixed8, 0)
QWEN35_IQ_FIXED8(qwen35_matvec_iq2xs_fixed8, 1)
QWEN35_IQ_FIXED8(qwen35_matvec_iq2s_fixed8, 2)
QWEN35_IQ_FIXED8(qwen35_matvec_iq3s_fixed8, 4)
#undef QWEN35_IQ_FIXED8

/* IQ3_XXS is faster with the original j-major schedule despite its higher
 * register count, so retain that exact specialization for this format. */
extern "C" __global__ void qwen35_matvec_iq3xxs_fixed8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    (void)count;
    qwen35_matvec_iq_multi<8>(out,weights,q,scales,rows,cols,3,8);
}

/* IQ4_XS keeps eight virtual reference sums per output row.  Reuse decoded
 * nibbles across the verifier rows without changing that per-row summation
 * order. */
extern "C" __global__ void qwen35_matvec_iq4xs_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    int lane=threadIdx.x&31;
    int row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (row>=rows) return;
    int blocks=cols/256;
    const unsigned char *w=weights+(size_t)row*blocks*136;
    float sum[8]={};
#pragma unroll
    for (int vw=0; vw<8; ++vw) {
        float partial[8]={};
        for (int qb=lane+32*vw; qb<blocks*8; qb+=256) {
            int ib=qb&7;
            const unsigned char *p=w+(qb/8)*136;
            const unsigned int *v=(const unsigned int *)(p+8+ib*16);
            int z[8]={};
#pragma unroll
            for (int j=0; j<4; ++j) {
                int lo=qwen35_iq4_lookup(v[j]);
                int hi=qwen35_iq4_lookup(v[j]>>4);
#pragma unroll
                for (int m=0; m<8; ++m) if (m<count) {
                    const int *u=(const int *)(q+(size_t)m*cols+qb*32);
                    z[m]=qwen35_dot4(lo,u[j],z[m]);
                    z[m]=qwen35_dot4(hi,u[j+4],z[m]);
                }
            }
            int ls=((p[4+ib/2]>>(4*(ib&1)))&15) |
                   (((*(const unsigned short *)(p+2)>>(2*ib))&3)<<4);
#pragma unroll
            for (int m=0; m<8; ++m) if (m<count) {
                int zz=z[m]*(ls-32);
                float d=(float)*(const _Float16 *)p *
                        scales[(size_t)m*(cols/32)+qb];
                partial[m]+=d*zz;
            }
        }
#pragma unroll
        for (int m=0; m<8; ++m) if (m<count) sum[m]+=partial[m];
    }
#pragma unroll
    for (int m=0; m<8; ++m) if (m<count) {
        for (int offset=16; offset; offset>>=1)
            sum[m]+=__shfl_xor(sum[m],offset,32);
        if (lane==0) out[(size_t)m*rows+row]=sum[m];
    }
}

/* Qwen3.8 hidden/output projections have exactly 20 IQ4_XS blocks.  Only
 * virtual reference warps 0..4 can contribute at this width; specializing
 * that bound removes three empty phases while retaining every contributing
 * multiply and addition in the generic kernel's order. */
extern "C" __global__ void qwen35_matvec_iq4xs_5120_multi8(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols, int count) {
    int lane=threadIdx.x&31;
    int row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if(row>=rows)return;
    const int blocks=20;
    const unsigned char *w=weights+(size_t)row*blocks*136;
    float sum[8]={};
#pragma unroll
    for(int vw=0;vw<5;++vw){
        float partial[8]={};
        int qb=lane+32*vw;
        if(qb<blocks*8){
            int ib=qb&7;
            const unsigned char *p=w+(qb/8)*136;
            const unsigned int *v=(const unsigned int *)(p+8+ib*16);
            int z[8]={};
#pragma unroll
            for(int j=0;j<4;++j){
                int lo=qwen35_iq4_lookup(v[j]);
                int hi=qwen35_iq4_lookup(v[j]>>4);
#pragma unroll
                for(int m=0;m<8;++m)if(m<count){
                    const int *u=(const int *)(q+(size_t)m*cols+qb*32);
                    z[m]=qwen35_dot4(lo,u[j],z[m]);
                    z[m]=qwen35_dot4(hi,u[j+4],z[m]);
                }
            }
            int ls=((p[4+ib/2]>>(4*(ib&1)))&15) |
                   (((*(const unsigned short *)(p+2)>>(2*ib))&3)<<4);
#pragma unroll
            for(int m=0;m<8;++m)if(m<count){
                int zz=z[m]*(ls-32);
                float d=(float)*(const _Float16 *)p *
                        scales[(size_t)m*(cols/32)+qb];
                partial[m]+=d*zz;
            }
        }
#pragma unroll
        for(int m=0;m<8;++m)if(m<count)sum[m]+=partial[m];
    }
#pragma unroll
    for(int m=0;m<8;++m)if(m<count){
        for(int offset=16;offset;offset>>=1)
            sum[m]+=__shfl_xor(sum[m],offset,32);
        if(lane==0)out[(size_t)m*rows+row]=sum[m];
    }
}
