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
__device__ static const unsigned char qwen35_iq_signs[128] = {
#include "../../common/ksigns_iq2xs.inc"
};
__device__ static const unsigned int qwen35_sign_mask[16] = {
    0x00000000u, 0x000000ffu, 0x0000ff00u, 0x0000ffffu,
    0x00ff0000u, 0x00ff00ffu, 0x00ffff00u, 0x00ffffffu,
    0xff000000u, 0xff0000ffu, 0xff00ff00u, 0xff00ffffu,
    0xffff0000u, 0xffff00ffu, 0xffffff00u, 0xffffffffu
};
__device__ __forceinline__ int qwen35_signed_grid(unsigned int grid, unsigned int signs, int shift) {
    unsigned int mask = qwen35_sign_mask[(signs >> shift) & 15];
    // These IQ grids contain strictly positive magnitudes, so the addition
    // cannot carry across byte boundaries.
    return (int)((grid ^ mask) + (mask & 0x01010101u));
}
__device__ __forceinline__ int qwen35_dot4(int a, int b, int sum) {
    return __builtin_amdgcn_sudot4(true, a, true, b, sum, false);
}

extern "C" __global__ void qwen35_matvec_iq2xxs(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
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
            unsigned long long grid = qwen35_iq2xxs_grid[p[2 + ib * 8 + j]];
            unsigned int signs = qwen35_iq_signs[(aux >> (7*j)) & 127];
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

extern "C" __global__ void qwen35_matvec_iq2xs(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
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
            unsigned long long grid = qwen35_iq2xs_grid[code & 511];
            unsigned int signs = qwen35_iq_signs[code >> 9];
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

extern "C" __global__ void qwen35_matvec_iq2s(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
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
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
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
            unsigned int signs = qwen35_iq_signs[(aux >> (7*j)) & 127];
            int lo = qwen35_signed_grid(qwen35_iq3xxs_grid[p[2 + ib*8 + 2*j]], signs, 0);
            int hi = qwen35_signed_grid(qwen35_iq3xxs_grid[p[3 + ib*8 + 2*j]], signs, 4);
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

extern "C" __global__ void qwen35_matvec_iq3s(
        float *out, const unsigned char *weights, const signed char *q,
        const float *scales, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    int blocks = cols / 256;
    const unsigned char *w = weights + (size_t)row * blocks * 110;
    float sum = 0.0f;
    for (int qb = lane; qb < blocks * 8; qb += 32) {
        int b = qb / 8, ib = qb & 7;
        const unsigned char *p = w + b * 110;
        const int *u = (const int *)(q + qb * 32);
        unsigned int qh = p[66 + ib];
        int z = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int signs = p[74 + ib * 4 + j];
            int il = p[2 + ib*8 + 2*j] | ((qh << (8 - 2*j)) & 0x100);
            int ih = p[3 + ib*8 + 2*j] | ((qh << (7 - 2*j)) & 0x100);
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
