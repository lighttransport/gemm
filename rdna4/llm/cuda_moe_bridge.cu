#include "cuda_moe_bridge.h"

#include "glm5next_iq1s_grid.h"
#include "glm5next_iq2_iq3_grid.h"
#include "glm5next_iq2xs_grid.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* IQ1 routed decoding revisits expert matrices across layers/tokens.  The
 * 5060 Ti has enough free VRAM for a larger resident set, which matters on
 * this host because the CUDA device is behind PCIe Gen3 x8. */
#define GLM5NEXT_CUDA_CACHE_SLOTS 2048

struct glm5next_cuda_moe_bridge {
    int device;
    int verbose;
    float *d_hidden;
    float *d_weights;
    float *d_parts;
    signed char *d_xq;
    float *d_xscale;
    signed char *d_parts_q;
    float *d_parts_scale;
    float *d_down_parts;
    float *d_out;
    unsigned char *d_gate[8];
    unsigned char *d_up[8];
    unsigned char *d_down[8];
    unsigned char *d_shared_gate;
    unsigned char *d_shared_up;
    unsigned char *d_shared_down;
    size_t gate_bytes, up_bytes, down_bytes;
    size_t shared_gate_bytes, shared_up_bytes, shared_down_bytes;
    float *h_parts;
    const unsigned char **d_gate_ptrs;
    const unsigned char **d_up_ptrs;
    const unsigned char **d_down_ptrs;
    int alloc_experts;
    int alloc_expert_ff;
    int alloc_hidden;
    int alloc_shared_ff;
    const void *cache_gate_host[GLM5NEXT_CUDA_CACHE_SLOTS];
    const void *cache_up_host[GLM5NEXT_CUDA_CACHE_SLOTS];
    const void *cache_down_host[GLM5NEXT_CUDA_CACHE_SLOTS];
    unsigned char *cache_gate[GLM5NEXT_CUDA_CACHE_SLOTS];
    unsigned char *cache_up[GLM5NEXT_CUDA_CACHE_SLOTS];
    unsigned char *cache_down[GLM5NEXT_CUDA_CACHE_SLOTS];
    size_t cache_gu_bytes[GLM5NEXT_CUDA_CACHE_SLOTS];
    size_t cache_down_bytes[GLM5NEXT_CUDA_CACHE_SLOTS];
    size_t cache_budget;
    size_t cache_used;
    uint64_t cache_age[GLM5NEXT_CUDA_CACHE_SLOTS];
    uint64_t cache_clock;
};

static int cache_prepare(glm5next_cuda_moe_bridge *b, const void *hg,
        const void *hu, const void *hd, size_t gu_bytes, size_t down_bytes,
        unsigned char **dg, unsigned char **du, unsigned char **dd) {
    int free_slot = -1, lru = 0;
    size_t need = gu_bytes * 2 + down_bytes;
    if (need == 0 || (b->cache_budget && need > b->cache_budget)) return -1;
    for (int i = 0; i < GLM5NEXT_CUDA_CACHE_SLOTS; ++i) {
        if (!b->cache_gate_host[i] && free_slot < 0) free_slot = i;
        if (b->cache_gate_host[i] && b->cache_gate_host[i] == hg &&
            b->cache_up_host[i] == hu && b->cache_down_host[i] == hd &&
            b->cache_gu_bytes[i] == gu_bytes && b->cache_down_bytes[i] == down_bytes) {
            b->cache_age[i] = ++b->cache_clock;
            *dg = b->cache_gate[i]; *du = b->cache_up[i]; *dd = b->cache_down[i];
            return 0;
        }
        if (b->cache_age[i] < b->cache_age[lru]) lru = i;
    }
    /* Keep CUDA-side expert residency within the explicitly reserved VRAM.
     * Evict complete triples, oldest first, before allocating a new one. */
    while (b->cache_budget && b->cache_used + need > b->cache_budget) {
        uint64_t oldest = UINT64_MAX;
        int victim = -1;
        for (int i = 0; i < GLM5NEXT_CUDA_CACHE_SLOTS; ++i) {
            if (b->cache_gate_host[i] && b->cache_age[i] < oldest) {
                oldest = b->cache_age[i]; victim = i;
            }
        }
        if (victim < 0) break;
        cudaFree(b->cache_gate[victim]);
        cudaFree(b->cache_up[victim]);
        cudaFree(b->cache_down[victim]);
        b->cache_gate[victim] = b->cache_up[victim] = b->cache_down[victim] = NULL;
        size_t old = b->cache_gu_bytes[victim] * 2 + b->cache_down_bytes[victim];
        if (b->cache_used >= old) b->cache_used -= old; else b->cache_used = 0;
        b->cache_gate_host[victim] = b->cache_up_host[victim] = b->cache_down_host[victim] = NULL;
        b->cache_gu_bytes[victim] = b->cache_down_bytes[victim] = 0;
        b->cache_age[victim] = 0;
        if (free_slot < 0) free_slot = victim;
    }
    if (b->cache_budget && b->cache_used + need > b->cache_budget) return -1;
    int slot = free_slot >= 0 ? free_slot : lru;
    if (b->cache_gate_host[slot]) {
        size_t old = b->cache_gu_bytes[slot] * 2 + b->cache_down_bytes[slot];
        if (b->cache_used >= old) b->cache_used -= old; else b->cache_used = 0;
    }
    if (b->cache_gate[slot]) cudaFree(b->cache_gate[slot]);
    if (b->cache_up[slot]) cudaFree(b->cache_up[slot]);
    if (b->cache_down[slot]) cudaFree(b->cache_down[slot]);
    b->cache_gate[slot] = b->cache_up[slot] = b->cache_down[slot] = NULL;
    cudaError_t ce = cudaMalloc(&b->cache_gate[slot], gu_bytes);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda_moe: cache gate allocation (%zu) failed: %s\n",
                gu_bytes, cudaGetErrorString(ce));
        return -1;
    }
    ce = cudaMalloc(&b->cache_up[slot], gu_bytes);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda_moe: cache up allocation (%zu) failed: %s\n",
                gu_bytes, cudaGetErrorString(ce));
        cudaFree(b->cache_gate[slot]); b->cache_gate[slot] = NULL;
        return -1;
    }
    ce = cudaMalloc(&b->cache_down[slot], down_bytes);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda_moe: cache down allocation (%zu) failed: %s\n",
                down_bytes, cudaGetErrorString(ce));
        cudaFree(b->cache_gate[slot]); cudaFree(b->cache_up[slot]); cudaFree(b->cache_down[slot]);
        b->cache_gate[slot] = b->cache_up[slot] = b->cache_down[slot] = NULL;
        return -1;
    }
    b->cache_gate_host[slot] = hg; b->cache_up_host[slot] = hu;
    b->cache_down_host[slot] = hd; b->cache_gu_bytes[slot] = gu_bytes;
    b->cache_down_bytes[slot] = down_bytes; b->cache_age[slot] = ++b->cache_clock;
    b->cache_used += need;
    *dg = b->cache_gate[slot]; *du = b->cache_up[slot]; *dd = b->cache_down[slot];
    return 1;
}

__device__ __constant__ uint64_t iq1s_grid_cuda[2048];
__device__ __constant__ uint64_t iq2xxs_grid_cuda[256];
__device__ __constant__ uint64_t iq2xs_grid_cuda[512];
__device__ __constant__ uint8_t ksigns_iq2xs_cuda[128];
__device__ __constant__ uint32_t iq3xxs_grid_cuda[256];

__device__ __forceinline__ float iq1_dot_row_part(const unsigned char *mat,
        const float *x, int cols, int row, int tid, int nthreads) {
    const int nb = cols / 256;
    const unsigned char *rp = mat + (size_t)row * nb * 50;
    float sum = 0.0f;
    (void)nthreads;
    for (int b = tid >> 1; b < nb; b += 16) {
        const unsigned char *bp = rp + b * 50;
        const float d = __half2float(*(const __half *)bp);
        const unsigned char *qs = bp + 2;
        const unsigned short *qh = (const unsigned short *)(bp + 34);
        const float delta_scale = 0.125f;
        int ib0 = (tid & 1) * 4;
        for (int ib = ib0; ib < ib0 + 4; ++ib) {
            const float scale = d * (float)(2 * ((qh[ib] >> 12) & 7) + 1);
            const float delta = (qh[ib] & 0x8000) ? -delta_scale : delta_scale;
            for (int l = 0; l < 4; ++l) {
                int qi = qs[ib * 4 + l] | (((qh[ib] >> (3 * l)) & 7) << 8);
                const signed char *grid = (const signed char *)&iq1s_grid_cuda[qi];
                int base = b * 256 + ib * 32 + l * 8;
                for (int j = 0; j < 8; ++j)
                    sum += scale * ((float)grid[j] + delta) * x[base + j];
            }
        }
    }
    return sum;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__global__ void glm5next_cuda_quantize_q8_32(signed char *q, float *scale,
        const float *x, int n) {
    int g = blockIdx.x, lane = threadIdx.x, i = g * 32 + lane;
    float v = i < n ? x[i] : 0.0f, a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_down_sync(0xffffffff, a, o));
    a = __shfl_sync(0xffffffff, a, 0);
    float s = a > 0.0f ? a / 127.0f : 0.0f;
    if (lane == 0) scale[g] = s;
    if (i < n) q[i] = (signed char)(s > 0.0f ? max(-127, min(127, (int)rintf(v / s))) : 0);
}

__device__ __forceinline__ int pack4_i8(const signed char *p) {
    return (int)(unsigned char)p[0] | ((int)(unsigned char)p[1] << 8) |
           ((int)(unsigned char)p[2] << 16) | ((int)(unsigned char)p[3] << 24);
}

__global__ void glm5next_cuda_iq1_gateup_grouped_dp4a(float *dst,
        const unsigned char *const *gates, const unsigned char *const *ups,
        const signed char *xq, const float *xscale,
        int rows, int cols, int experts, float clamp) {
    int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, e = blockIdx.y;
    if (e >= experts || row >= rows) return;
    int qblocks = cols / 32, nb = cols / 256;
    float sg = 0.0f, su = 0.0f;
    const unsigned char *gw = gates[e] + (size_t)row * nb * 50;
    const unsigned char *uw = ups[e] + (size_t)row * nb * 50;
    for (int qb = lane; qb < qblocks; qb += 32) {
        int block = qb >> 3, ib = qb & 7;
        const unsigned char *gb = gw + block * 50, *ub = uw + block * 50;
        const unsigned short *gqh = (const unsigned short *)(gb + 34);
        const unsigned short *uqh = (const unsigned short *)(ub + 34);
        const unsigned char *gqs = gb + 2, *uqs = ub + 2;
        float gd = __half2float(*(const __half *)gb) * (float)(2 * ((gqh[ib] >> 12) & 7) + 1);
        float ud = __half2float(*(const __half *)ub) * (float)(2 * ((uqh[ib] >> 12) & 7) + 1);
        float gs = (gqh[ib] & 0x8000) ? -0.125f : 0.125f;
        float us = (uqh[ib] & 0x8000) ? -0.125f : 0.125f;
        const signed char *xp = xq + qb * 32;
        int dg = 0, du = 0, sumq = 0;
        for (int l = 0; l < 4; ++l) {
            int gi = gqs[ib * 4 + l] | (((gqh[ib] >> (3 * l)) & 7) << 8);
            int ui = uqs[ib * 4 + l] | (((uqh[ib] >> (3 * l)) & 7) << 8);
            const signed char *gc = (const signed char *)&iq1s_grid_cuda[gi];
            const signed char *uc = (const signed char *)&iq1s_grid_cuda[ui];
            dg += __dp4a(pack4_i8(gc), pack4_i8(xp + l * 8), 0);
            dg += __dp4a(pack4_i8(gc + 4), pack4_i8(xp + l * 8 + 4), 0);
            du += __dp4a(pack4_i8(uc), pack4_i8(xp + l * 8), 0);
            du += __dp4a(pack4_i8(uc + 4), pack4_i8(xp + l * 8 + 4), 0);
            for (int j = 0; j < 8; ++j) sumq += (int)xp[l * 8 + j];
        }
        sg += gd * xscale[qb] * ((float)dg + gs * (float)sumq);
        su += ud * xscale[qb] * ((float)du + us * (float)sumq);
    }
    sg = warp_reduce_sum(sg); su = warp_reduce_sum(su);
    if (lane == 0) {
        if (clamp > 1e-6f) { sg = fminf(sg, clamp); su = fminf(fmaxf(su, -clamp), clamp); }
        dst[(size_t)e * rows + row] = (sg / (1.0f + expf(-sg))) * su;
    }
}

__global__ void glm5next_cuda_iq1_down_grouped_dp4a(float *dst,
        const unsigned char *const *downs, const signed char *xq,
        const float *xscale, const float *weights, int rows, int cols, int experts) {
    int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, e = blockIdx.y;
    if (e >= experts || row >= rows) return;
    int qblocks = cols / 32, nb = cols / 256;
    float sum = 0.0f;
    const unsigned char *dw = downs[e] + (size_t)row * nb * 50;
    const signed char *xp = xq + (size_t)e * cols;
    const float *sc = xscale + (size_t)e * qblocks;
    for (int qb = lane; qb < qblocks; qb += 32) {
        int block = qb >> 3, ib = qb & 7;
        const unsigned char *bp = dw + block * 50;
        const unsigned short *qh = (const unsigned short *)(bp + 34);
        const unsigned char *qs = bp + 2;
        float d = __half2float(*(const __half *)bp) * (float)(2 * ((qh[ib] >> 12) & 7) + 1);
        float delta = (qh[ib] & 0x8000) ? -0.125f : 0.125f;
        const signed char *v = xp + qb * 32;
        int dot = 0, sumq = 0;
        for (int l = 0; l < 4; ++l) {
            int qi = qs[ib * 4 + l] | (((qh[ib] >> (3 * l)) & 7) << 8);
            const signed char *g = (const signed char *)&iq1s_grid_cuda[qi];
            dot += __dp4a(pack4_i8(g), pack4_i8(v + l * 8), 0);
            dot += __dp4a(pack4_i8(g + 4), pack4_i8(v + l * 8 + 4), 0);
            for (int j = 0; j < 8; ++j) sumq += (int)v[l * 8 + j];
        }
        sum += d * sc[qb] * ((float)dot + delta * (float)sumq);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) dst[(size_t)e * rows + row] = weights[e] * sum;
}

__global__ void glm5next_cuda_iq1_gateup(float *dst,
        const unsigned char *gate, const unsigned char *up,
        const float *x, int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    float g = iq1_dot_row_part(gate, x, cols, row, lane, 32);
    float u = iq1_dot_row_part(up, x, cols, row, lane, 32);
    g = warp_reduce_sum(g); u = warp_reduce_sum(u);
    if (lane == 0) dst[row] = (g / (1.0f + expf(-g))) * u;
}

__global__ void glm5next_cuda_iq1_down(float *dst,
        const unsigned char *down, const float *x, float weight,
        int rows, int cols) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    float v = iq1_dot_row_part(down, x, cols, row, lane, 32);
    v = warp_reduce_sum(v);
    if (lane == 0) dst[row] = weight * v;
}

/* Pointer-table form used by the resident routed cache.  Keeping the expert
 * index in blockIdx.y avoids one launch and one global reduction per expert. */
__global__ void glm5next_cuda_iq1_gateup_grouped(float *dst,
        const unsigned char *const *gates, const unsigned char *const *ups,
        const float *x, int rows, int cols, int experts, float clamp) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int e = blockIdx.y;
    if (e >= experts || row >= rows) return;
    float g = iq1_dot_row_part(gates[e], x, cols, row, lane, 32);
    float u = iq1_dot_row_part(ups[e], x, cols, row, lane, 32);
    g = warp_reduce_sum(g); u = warp_reduce_sum(u);
    if (lane == 0) {
        if (clamp > 1e-6f) { g = fminf(g, clamp); u = fminf(fmaxf(u, -clamp), clamp); }
        dst[(size_t)e * rows + row] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ void glm5next_cuda_iq1_down_grouped(float *dst,
        const unsigned char *const *downs, const float *x,
        const float *weights, int rows, int cols, int experts) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int e = blockIdx.y;
    if (e >= experts || row >= rows) return;
    float v = iq1_dot_row_part(downs[e], x, cols, row, lane, 32);
    v = warp_reduce_sum(v);
    if (lane == 0) dst[(size_t)e * rows + row] = weights[e] * v;
}

__device__ __forceinline__ float iq2_dot_row(const unsigned char *mat,
        const float *x, int cols, int row) {
    int lane = threadIdx.x & 31;
    int nb = cols / 256;
    const unsigned char *rp = mat + (size_t)row * nb * 66;
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char *bp = rp + b * 66;
        float d = __half2float(*(const __half *)bp);
        const unsigned short *qs = (const unsigned short *)(bp + 2);
        int yi = 0;
        for (int ib = 0; ib < 8; ++ib) {
            unsigned int a0 = qs[4 * ib] | ((unsigned int)qs[4 * ib + 1] << 16);
            unsigned int a1 = qs[4 * ib + 2] | ((unsigned int)qs[4 * ib + 3] << 16);
            float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
            const unsigned char *codes = (const unsigned char *)&a0;
            for (int l = 0; l < 4; ++l) {
                const unsigned char *grid = (const unsigned char *)&iq2xxs_grid_cuda[codes[l]];
                unsigned char signs = ksigns_iq2xs_cuda[(a1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j)
                    sum += db * (float)grid[j] * ((signs & (1 << j)) ? -1.0f : 1.0f) *
                           x[b * 256 + yi++];
            }
        }
    }
    return sum;
}

__device__ __forceinline__ float iq2xs_dot_row(const unsigned char *mat,
        const float *x, int cols, int row) {
    int lane = threadIdx.x & 31;
    int nb = cols / 256;
    const unsigned char *rp = mat + (size_t)row * nb * 74;
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char *bp = rp + b * 74;
        float d = __half2float(*(const __half *)bp);
        const unsigned short *qs = (const unsigned short *)(bp + 2);
        const unsigned char *scales = bp + 66;
        for (int ib = 0; ib < 8; ++ib) {
            for (int l = 0; l < 4; ++l) {
                float db = d * (0.5f + (float)((l & 2) ?
                    (scales[ib] >> 4) : (scales[ib] & 15))) * 0.25f;
                unsigned short q = qs[4 * ib + l];
                const unsigned char *grid = (const unsigned char *)&iq2xs_grid_cuda[q & 511];
                unsigned char signs = ksigns_iq2xs_cuda[q >> 9];
                const float *xb = x + b * 256 + ib * 32 + l * 8;
                for (int j = 0; j < 8; ++j)
                    sum += db * (float)grid[j] * ((signs & (1 << j)) ? -1.0f : 1.0f) * xb[j];
            }
        }
    }
    return sum;
}

__device__ __forceinline__ float iq3_dot_row(const unsigned char *mat,
        const float *x, int cols, int row) {
    int lane = threadIdx.x & 31;
    int nb = cols / 256;
    const unsigned char *rp = mat + (size_t)row * nb * 98;
    float sum = 0.0f;
    for (int g = lane; g < nb * 32; g += 32) {
        int b = g >> 5, rem = g & 31, sb = rem >> 2, l = rem & 3;
        const unsigned char *bp = rp + b * 98;
        float d = __half2float(*(const __half *)bp);
        const unsigned char *qs = bp + 2;
        const unsigned char *ap = bp + 66 + 4 * sb;
        unsigned int aux = (unsigned int)ap[0] | ((unsigned int)ap[1] << 8) |
                           ((unsigned int)ap[2] << 16) | ((unsigned int)ap[3] << 24);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        unsigned char signs = ksigns_iq2xs_cuda[(aux >> (7 * l)) & 127];
        const unsigned char *g1 = (const unsigned char *)&iq3xxs_grid_cuda[qs[8 * sb + 2 * l]];
        const unsigned char *g2 = (const unsigned char *)&iq3xxs_grid_cuda[qs[8 * sb + 2 * l + 1]];
        const float *xb = x + b * 256 + sb * 32 + l * 8;
        for (int j = 0; j < 4; ++j) {
            sum += db * (float)g1[j] * ((signs & (1 << j)) ? -1.0f : 1.0f) * xb[j];
            sum += db * (float)g2[j] * ((signs & (1 << (j + 4))) ? -1.0f : 1.0f) * xb[j + 4];
        }
    }
    return sum;
}

__device__ __forceinline__ float iq4nl_dot_row(const unsigned char *mat,
        const float *x, int cols, int row) {
    int lane = threadIdx.x & 31;
    int nb = cols / 32;
    const unsigned char *rp = mat + (size_t)row * nb * 18;
    const int values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                             1, 13, 25, 38, 53, 69, 89, 113};
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char *bp = rp + b * 18;
        float d = __half2float(*(const __half *)bp);
        for (int j = 0; j < 16; ++j) {
            sum += d * (float)values[bp[2 + j] & 15] * x[b * 32 + j];
            sum += d * (float)values[bp[2 + j] >> 4] * x[b * 32 + j + 16];
        }
    }
    return sum;
}

__device__ __forceinline__ float iq4xs_dot_row(const unsigned char *mat,
        const float *x, int cols, int row) {
    int lane = threadIdx.x & 31;
    int nb = cols / 256;
    const unsigned char *rp = mat + (size_t)row * nb * 136;
    const int values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                             1, 13, 25, 38, 53, 69, 89, 113};
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char *bp = rp + b * 136;
        float d = __half2float(*(const __half *)bp);
        unsigned short scales_h = *(const unsigned short *)(bp + 2);
        const unsigned char *scales_l = bp + 4;
        const unsigned char *qs = bp + 8;
        for (int ib = 0; ib < 8; ++ib) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib & 1))) & 15) |
                     (((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            const float *xb = x + b * 256 + ib * 32;
            const unsigned char *q = qs + ib * 16;
            for (int j = 0; j < 16; ++j) {
                sum += dl * (float)values[q[j] & 15] * xb[j];
                sum += dl * (float)values[q[j] >> 4] * xb[j + 16];
            }
        }
    }
    return sum;
}

__device__ __forceinline__ void q5k_scale_min(int j, const unsigned char *s,
        int *scale, int *minv) {
    if (j < 4) {
        *scale = s[j] & 63; *minv = s[j + 4] & 63;
    } else {
        *scale = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
        *minv = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float q5k_dot_row(const unsigned char *mat,
        const float *x, int cols, int row) {
    int lane = threadIdx.x & 31, nb = cols / 256;
    const unsigned char *rp = mat + (size_t)row * nb * 176;
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char *bp = rp + b * 176;
        float d = __half2float(*(const __half *)bp);
        float dm = __half2float(*(const __half *)(bp + 2));
        const unsigned char *sc = bp + 4, *qh = bp + 16, *ql = bp + 48;
        for (int c = 0; c < 4; ++c) {
            int s0, m0, s1, m1;
            q5k_scale_min(2 * c, sc, &s0, &m0);
            q5k_scale_min(2 * c + 1, sc, &s1, &m1);
            const unsigned char *q = ql + c * 32;
            for (int j = 0; j < 32; ++j) {
                int q0 = (q[j] & 15) | (((qh[j] >> (2 * c)) & 1) << 4);
                int q1 = (q[j] >> 4) | (((qh[j] >> (2 * c + 1)) & 1) << 4);
                sum += (d * s0 * q0 - dm * m0) * x[b * 256 + c * 64 + j];
                sum += (d * s1 * q1 - dm * m1) * x[b * 256 + c * 64 + 32 + j];
            }
        }
    }
    return sum;
}

__device__ __forceinline__ float q6k_dot_row(const unsigned char *mat,
        const float *x, int cols, int row) {
    int lane = threadIdx.x & 31, nb = cols / 256;
    const unsigned char *rp = mat + (size_t)row * nb * 210;
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char *bp = rp + b * 210;
        float d = __half2float(*(const __half *)(bp + 208));
        const unsigned char *ql = bp, *qh = bp + 128;
        const signed char *sc = (const signed char *)(bp + 192);
        for (int half = 0; half < 2; ++half)
            for (int part = 0; part < 4; ++part)
                for (int j = 0; j < 32; ++j) {
                    int lo = (ql[half * 64 + ((part & 1) ? 32 : 0) + j] >>
                              (part >= 2 ? 4 : 0)) & 15;
                    int hi = (qh[half * 32 + j] >> (part * 2)) & 3;
                    int q = lo | (hi << 4);
                    int si = half * 8 + part * 2 + j / 16;
                    int col = half * 128 + part * 32 + j;
                    sum += d * sc[si] * (q - 32) * x[b * 256 + col];
                }
    }
    return sum;
}

__global__ void glm5next_cuda_q5q6_gateup(float *dst, const unsigned char *gate,
        const unsigned char *up, const float *x, int rows, int cols) {
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    float g = warp_reduce_sum(q5k_dot_row(gate, x, cols, row));
    float u = warp_reduce_sum(q5k_dot_row(up, x, cols, row));
    if ((threadIdx.x & 31) == 0) dst[row] = (g / (1.0f + expf(-g))) * u;
}

__global__ void glm5next_cuda_q6_down(float *dst, const unsigned char *down,
        const float *x, int rows, int cols) {
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    float v = warp_reduce_sum(q6k_dot_row(down, x, cols, row));
    if ((threadIdx.x & 31) == 0) dst[row] = v;
}

__global__ void glm5next_cuda_iq2_gateup(float *dst, const unsigned char *gate,
        const unsigned char *up, const float *x, int rows, int cols) {
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    float g = warp_reduce_sum(iq2_dot_row(gate, x, cols, row));
    float u = warp_reduce_sum(iq2_dot_row(up, x, cols, row));
    if ((threadIdx.x & 31) == 0) dst[row] = (g / (1.0f + expf(-g))) * u;
}

__global__ void glm5next_cuda_iq3_down(float *dst, const unsigned char *down,
        const float *x, float weight, int rows, int cols) {
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    float v = warp_reduce_sum(iq3_dot_row(down, x, cols, row));
    if ((threadIdx.x & 31) == 0) dst[row] += weight * v;
}

__global__ void glm5next_cuda_iq4nl_down(float *dst, const unsigned char *down,
        const float *x, float weight, int rows, int cols) {
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows) return;
    float v = warp_reduce_sum(iq4nl_dot_row(down, x, cols, row));
    if ((threadIdx.x & 31) == 0) dst[row] += weight * v;
}

/* Grouped routed kernels: one launch covers all selected experts. */
__global__ void glm5next_cuda_iq2_gateup_grouped(float *dst,
        const unsigned char *const *gates, const unsigned char *const *ups,
        const float *x, int rows, int cols, int experts) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int e = blockIdx.y;
    if (e >= experts || row >= rows) return;
    float g = warp_reduce_sum(iq2_dot_row(gates[e], x, cols, row));
    float u = warp_reduce_sum(iq2_dot_row(ups[e], x, cols, row));
    if (lane == 0)
        dst[(size_t)e * rows + row] = (g / (1.0f + expf(-g))) * u;
}

__global__ void glm5next_cuda_iq2xs_gateup_grouped(float *dst,
        const unsigned char *const *gates, const unsigned char *const *ups,
        const float *x, int rows, int cols, int experts) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int e = blockIdx.y;
    if (e >= experts || row >= rows) return;
    float g = warp_reduce_sum(iq2xs_dot_row(gates[e], x, cols, row));
    float u = warp_reduce_sum(iq2xs_dot_row(ups[e], x, cols, row));
    if (lane == 0) dst[(size_t)e * rows + row] = (g / (1.0f + expf(-g))) * u;
}

__global__ void glm5next_cuda_iq3_down_grouped(float *dst,
        const unsigned char *const *downs, const float *parts,
        const float *weights, int rows, int cols, int experts) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int e = blockIdx.y;
    if (e >= experts || row >= rows) return;
    float v = warp_reduce_sum(iq3_dot_row(downs[e],
        parts + (size_t)e * cols, cols, row));
    if (lane == 0) dst[(size_t)e * rows + row] = weights[e] * v;
}

__global__ void glm5next_cuda_iq4nl_down_grouped(float *dst,
        const unsigned char *const *downs, const float *parts,
        const float *weights, int rows, int cols, int experts) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int e = blockIdx.y;
    if (e >= experts || row >= rows) return;
    float v = warp_reduce_sum(iq4nl_dot_row(downs[e],
        parts + (size_t)e * cols, cols, row));
    if (lane == 0) dst[(size_t)e * rows + row] = weights[e] * v;
}

__global__ void glm5next_cuda_iq4xs_down_grouped(float *dst,
        const unsigned char *const *downs, const float *parts,
        const float *weights, int rows, int cols, int experts) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + threadIdx.x / 32;
    int e = blockIdx.y;
    if (e >= experts || row >= rows) return;
    float v = warp_reduce_sum(iq4xs_dot_row(downs[e],
        parts + (size_t)e * cols, cols, row));
    if (lane == 0) dst[(size_t)e * rows + row] = weights[e] * v;
}

__global__ void glm5next_cuda_moe_reduce(float *dst, const float *parts,
        int rows, int experts) {
    int row = blockIdx.x * 256 + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int e = 0; e < experts; ++e)
        sum += parts[(size_t)e * rows + row];
    dst[row] = sum;
}

static int cuda_ok(cudaError_t e) { return e == cudaSuccess ? 0 : -1; }

static int ensure_alloc(glm5next_cuda_moe_bridge *b, size_t gate_bytes,
        size_t up_bytes, size_t down_bytes, size_t shared_gate_bytes,
        size_t shared_up_bytes, size_t shared_down_bytes, int experts,
        int expert_ff, int hidden_size, int shared_ff) {
    if (b->alloc_experts != experts || b->gate_bytes != gate_bytes ||
        b->up_bytes != up_bytes || b->down_bytes != down_bytes) {
        for (int i = 0; i < 8; ++i) {
            cudaFree(b->d_gate[i]); cudaFree(b->d_up[i]); cudaFree(b->d_down[i]);
            b->d_gate[i] = b->d_up[i] = b->d_down[i] = NULL;
        }
        for (int i = 0; i < experts; ++i) {
            cudaError_t ce = cudaMalloc(&b->d_gate[i], gate_bytes);
            if (ce != cudaSuccess) {
                fprintf(stderr, "cuda_moe: work gate allocation (%zu) failed: %s\n",
                        gate_bytes, cudaGetErrorString(ce)); return -1;
            }
            ce = cudaMalloc(&b->d_up[i], up_bytes);
            if (ce != cudaSuccess) {
                fprintf(stderr, "cuda_moe: work up allocation (%zu) failed: %s\n",
                        up_bytes, cudaGetErrorString(ce)); return -1;
            }
            ce = cudaMalloc(&b->d_down[i], down_bytes);
            if (ce != cudaSuccess) {
                fprintf(stderr, "cuda_moe: work down allocation (%zu) failed: %s\n",
                        down_bytes, cudaGetErrorString(ce)); return -1;
            }
        }
        b->gate_bytes = gate_bytes; b->up_bytes = up_bytes; b->down_bytes = down_bytes;
        b->alloc_experts = experts;
    }
    if (shared_gate_bytes > 0 &&
        (b->alloc_shared_ff != shared_ff || b->shared_gate_bytes != shared_gate_bytes ||
        b->shared_up_bytes != shared_up_bytes || b->shared_down_bytes != shared_down_bytes)) {
        cudaFree(b->d_shared_gate); cudaFree(b->d_shared_up); cudaFree(b->d_shared_down);
        if (cudaMalloc(&b->d_shared_gate, shared_gate_bytes) != cudaSuccess ||
            cudaMalloc(&b->d_shared_up, shared_up_bytes) != cudaSuccess ||
            cudaMalloc(&b->d_shared_down, shared_down_bytes) != cudaSuccess) return -1;
        b->shared_gate_bytes = shared_gate_bytes; b->shared_up_bytes = shared_up_bytes;
        b->shared_down_bytes = shared_down_bytes;
        b->alloc_shared_ff = shared_ff;
    }
    if (b->alloc_hidden != hidden_size || b->alloc_expert_ff != expert_ff) {
        cudaFree(b->d_hidden); cudaFree(b->d_weights); cudaFree(b->d_parts);
        cudaFree(b->d_xq); cudaFree(b->d_xscale); cudaFree(b->d_parts_q); cudaFree(b->d_parts_scale);
        cudaFree(b->d_down_parts); cudaFree(b->d_out);
        cudaFree(b->d_gate_ptrs); cudaFree(b->d_up_ptrs); cudaFree(b->d_down_ptrs);
        b->d_hidden = b->d_weights = b->d_parts = b->d_down_parts = b->d_out = NULL;
        b->d_xq = b->d_parts_q = NULL; b->d_xscale = b->d_parts_scale = NULL;
        b->d_gate_ptrs = b->d_up_ptrs = b->d_down_ptrs = NULL;
        cudaError_t ce = cudaMalloc(&b->d_hidden, (size_t)hidden_size * sizeof(float));
        if (ce != cudaSuccess) {
            fprintf(stderr, "cuda_moe: hidden allocation failed: %s\n", cudaGetErrorString(ce)); return -1;
        }
        ce = cudaMalloc(&b->d_weights, 8 * sizeof(float));
        if (ce != cudaSuccess) {
            fprintf(stderr, "cuda_moe: weights allocation failed: %s\n", cudaGetErrorString(ce)); return -1;
        }
        ce = cudaMalloc(&b->d_parts, (size_t)experts * expert_ff * sizeof(float));
        if (ce != cudaSuccess) {
            fprintf(stderr, "cuda_moe: parts allocation failed: %s\n", cudaGetErrorString(ce)); return -1;
        }
        ce = cudaMalloc(&b->d_down_parts,
                        (size_t)experts * hidden_size * sizeof(float));
        if (ce != cudaSuccess) {
            fprintf(stderr, "cuda_moe: down-parts allocation failed: %s\n",
                    cudaGetErrorString(ce)); return -1;
        }
        ce = cudaMalloc(&b->d_gate_ptrs, (size_t)experts * sizeof(*b->d_gate_ptrs));
        if (ce != cudaSuccess) return -1;
        ce = cudaMalloc(&b->d_up_ptrs, (size_t)experts * sizeof(*b->d_up_ptrs));
        if (ce != cudaSuccess) return -1;
        ce = cudaMalloc(&b->d_down_ptrs, (size_t)experts * sizeof(*b->d_down_ptrs));
        if (ce != cudaSuccess) return -1;
        ce = cudaMalloc(&b->d_out, (size_t)hidden_size * sizeof(float));
        if (ce != cudaSuccess) {
            fprintf(stderr, "cuda_moe: output allocation failed: %s\n", cudaGetErrorString(ce)); return -1;
        }
        ce = cudaMalloc(&b->d_xq, (size_t)hidden_size * sizeof(signed char));
        if (ce != cudaSuccess) return -1;
        ce = cudaMalloc(&b->d_xscale, (size_t)(hidden_size / 32) * sizeof(float));
        if (ce != cudaSuccess) return -1;
        ce = cudaMalloc(&b->d_parts_q, (size_t)experts * expert_ff * sizeof(signed char));
        if (ce != cudaSuccess) return -1;
        ce = cudaMalloc(&b->d_parts_scale,
                        (size_t)experts * (size_t)(expert_ff / 32) * sizeof(float));
        if (ce != cudaSuccess) return -1;
        free(b->h_parts);
        b->h_parts = (float *)malloc((size_t)hidden_size * sizeof(float));
        if (!b->h_parts) return -1;
        b->alloc_hidden = hidden_size; b->alloc_expert_ff = expert_ff;
    }
    return 0;
}

glm5next_cuda_moe_bridge *glm5next_cuda_moe_init(int device, int verbose) {
    glm5next_cuda_moe_bridge *b = (glm5next_cuda_moe_bridge *)calloc(1, sizeof(*b));
    if (!b || cudaSetDevice(device) != cudaSuccess) { free(b); return NULL; }
    b->device = device; b->verbose = verbose;
    const char *cache_mb = getenv("GLM5NEXT_CUDA_CACHE_MB");
    b->cache_budget = (size_t)(cache_mb ? strtoull(cache_mb, NULL, 10) : 12000) << 20;
    /* Leave room for the bridge work buffers and desktop CUDA clients.  The
     * 5060 Ti is display-attached, so the nominal 16 GiB is not available to
     * expert residency; cap an explicit budget against the live free amount. */
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        /* Work buffers are allocated before expert residency.  Keep 512 MiB
         * for them and the display, allowing the bridge to use the remaining
         * live VRAM instead of conservatively discarding a full GiB. */
        const size_t reserve = (size_t)512 << 20;
        size_t safe_budget = free_bytes > reserve ? free_bytes - reserve : 0;
        if (b->cache_budget > safe_budget) b->cache_budget = safe_budget;
    }
    if (verbose >= 1)
        fprintf(stderr, "cuda_moe: expert cache budget %.0f MiB\n",
                (double)b->cache_budget / (1024.0 * 1024.0));
    if (cudaMemcpyToSymbol(iq1s_grid_cuda, glm5next_iq1s_grid,
                           sizeof(glm5next_iq1s_grid)) != cudaSuccess) {
        free(b); return NULL;
    }
    if (cudaMemcpyToSymbol(iq2xxs_grid_cuda, glm5next_iq2xxs_grid,
                           sizeof(glm5next_iq2xxs_grid)) != cudaSuccess ||
        cudaMemcpyToSymbol(iq2xs_grid_cuda, glm5next_iq2xs_grid,
                           sizeof(glm5next_iq2xs_grid)) != cudaSuccess ||
        cudaMemcpyToSymbol(ksigns_iq2xs_cuda, glm5next_ksigns_iq2xs,
                           sizeof(glm5next_ksigns_iq2xs)) != cudaSuccess ||
        cudaMemcpyToSymbol(iq3xxs_grid_cuda, glm5next_iq3xxs_grid,
                           sizeof(glm5next_iq3xxs_grid)) != cudaSuccess) {
        free(b); return NULL;
    }
    return b;
}

void glm5next_cuda_moe_free(glm5next_cuda_moe_bridge *b) {
    if (!b) return;
    cudaSetDevice(b->device);
    cudaFree(b->d_hidden); cudaFree(b->d_weights); cudaFree(b->d_parts);
    cudaFree(b->d_xq); cudaFree(b->d_xscale); cudaFree(b->d_parts_q); cudaFree(b->d_parts_scale);
    cudaFree(b->d_down_parts); cudaFree(b->d_out);
    cudaFree(b->d_gate_ptrs); cudaFree(b->d_up_ptrs); cudaFree(b->d_down_ptrs);
    for (int i = 0; i < 8; ++i) { cudaFree(b->d_gate[i]); cudaFree(b->d_up[i]); cudaFree(b->d_down[i]); }
    cudaFree(b->d_shared_gate); cudaFree(b->d_shared_up); cudaFree(b->d_shared_down);
    for (int i = 0; i < GLM5NEXT_CUDA_CACHE_SLOTS; ++i) {
        cudaFree(b->cache_gate[i]); cudaFree(b->cache_up[i]); cudaFree(b->cache_down[i]);
    }
    free(b->h_parts); free(b);
}

int glm5next_cuda_moe_compute(glm5next_cuda_moe_bridge *b,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, const void *shared_gate, const void *shared_up,
        const void *shared_down, int shared_ff, float *out) {
    if (!b || !gate || !up || !down || experts <= 0 || experts > 8 ||
        expert_ff <= 0 || hidden_size <= 0 || !hidden || !weights || !out ||
        !shared_gate || !shared_up || !shared_down || shared_ff <= 0) return -1;
    cudaSetDevice(b->device);
    const size_t gate_bytes = (size_t)expert_ff * (size_t)(hidden_size / 256) * 50;
    const size_t up_bytes = gate_bytes;
    const size_t down_bytes = (size_t)hidden_size * (size_t)(expert_ff / 256) * 50;
    const size_t sg_bytes = (size_t)shared_ff * (size_t)(hidden_size / 256) * 50;
    const size_t sd_bytes = (size_t)hidden_size * (size_t)(shared_ff / 256) * 50;
    if (ensure_alloc(b, gate_bytes, up_bytes, down_bytes, sg_bytes, sg_bytes,
                     sd_bytes, experts, expert_ff, hidden_size, shared_ff) != 0) return -1;
    if (cudaMemset(b->d_out, 0, (size_t)hidden_size * sizeof(float)) != cudaSuccess) return -1;
    memset(out, 0, (size_t)hidden_size * sizeof(float));
    if (cudaMemcpy(b->d_hidden, hidden, (size_t)hidden_size * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(b->d_weights, weights, (size_t)experts * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    for (int e = 0; e < experts; ++e) {
        if (cudaMemcpy(b->d_gate[e], gate[e], gate_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(b->d_up[e], up[e], up_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(b->d_down[e], down[e], down_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
        glm5next_cuda_iq1_gateup<<<(expert_ff + 7) / 8, 256>>>(b->d_parts + (size_t)e * expert_ff,
            b->d_gate[e], b->d_up[e], b->d_hidden, expert_ff, hidden_size);
        glm5next_cuda_iq1_down<<<(hidden_size + 7) / 8, 256>>>(b->d_out, b->d_down[e],
            b->d_parts + (size_t)e * expert_ff, weights[e], hidden_size, expert_ff);
        if (cudaMemcpy(b->h_parts, b->d_out, (size_t)hidden_size * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
        for (int i = 0; i < hidden_size; ++i) out[i] += b->h_parts[i];
    }
    if (cudaMemcpy(b->d_shared_gate, shared_gate, sg_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(b->d_shared_up, shared_up, sg_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(b->d_shared_down, shared_down, sd_bytes, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    glm5next_cuda_iq1_gateup<<<(shared_ff + 7) / 8, 256>>>(b->d_parts, b->d_shared_gate,
        b->d_shared_up, b->d_hidden, shared_ff, hidden_size);
    glm5next_cuda_iq1_down<<<(hidden_size + 7) / 8, 256>>>(b->d_out, b->d_shared_down,
        b->d_parts, 1.0f, hidden_size, shared_ff);
    if (cudaMemcpy(b->h_parts, b->d_out, (size_t)hidden_size * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    for (int i = 0; i < hidden_size; ++i) out[i] += b->h_parts[i];
    return cuda_ok(cudaDeviceSynchronize());
}

int glm5next_cuda_moe_compute_iq2_iq3(glm5next_cuda_moe_bridge *b,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float *out) {
    if (!b || !gate || !up || !down || experts <= 0 || experts > 8 ||
        expert_ff <= 0 || hidden_size <= 0 || expert_ff % 256 != 0 ||
        hidden_size % 256 != 0 || !hidden || !weights || !out) {
        fprintf(stderr, "cuda_moe: IQ2/IQ3 invalid shape b=%p experts=%d ff=%d hidden=%d\n",
                (void *)b, experts, expert_ff, hidden_size);
        return -1;
    }
#define CCHK(call) do { cudaError_t ce = (call); if (ce != cudaSuccess) { \
        fprintf(stderr, "cuda_moe: %s failed: %s\n", #call, cudaGetErrorString(ce)); return -1; } } while (0)
    size_t gu_bytes = (size_t)expert_ff * (size_t)(hidden_size / 256) * 66;
    size_t dn_bytes = (size_t)hidden_size * (size_t)(expert_ff / 256) * 98;
    if (ensure_alloc(b, gu_bytes, gu_bytes, dn_bytes, 0, 0, 0,
                     experts, expert_ff, hidden_size, expert_ff) != 0) {
        fprintf(stderr, "cuda_moe: allocation failed (gate=%zu down=%zu)\n",
                gu_bytes, dn_bytes);
        return -1;
    }
    CCHK(cudaMemcpy(b->d_hidden, hidden, (size_t)hidden_size * sizeof(float),
                    cudaMemcpyHostToDevice));
    CCHK(cudaMemcpy(b->d_weights, weights, (size_t)experts * sizeof(float),
                    cudaMemcpyHostToDevice));
    memset(out, 0, (size_t)hidden_size * sizeof(float));
    CCHK(cudaMemset(b->d_out, 0, (size_t)hidden_size * sizeof(float)));
    unsigned char *cg[8], *cu[8], *cd[8];
    for (int e = 0; e < experts; ++e) {
        int cached = cache_prepare(b, gate[e], up[e], down[e], gu_bytes, dn_bytes,
                                    &cg[e], &cu[e], &cd[e]);
        if (cached < 0) return -1;
        if (cached > 0) {
            CCHK(cudaMemcpy(cg[e], gate[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHK(cudaMemcpy(cu[e], up[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHK(cudaMemcpy(cd[e], down[e], dn_bytes, cudaMemcpyHostToDevice));
        }
    }
    CCHK(cudaMemcpy(b->d_gate_ptrs, cg, (size_t)experts * sizeof(*cg),
                    cudaMemcpyHostToDevice));
    CCHK(cudaMemcpy(b->d_up_ptrs, cu, (size_t)experts * sizeof(*cu),
                    cudaMemcpyHostToDevice));
    CCHK(cudaMemcpy(b->d_down_ptrs, cd, (size_t)experts * sizeof(*cd),
                    cudaMemcpyHostToDevice));
    glm5next_cuda_iq2_gateup_grouped<<<(expert_ff + 7) / 8, 256, 0, 0>>>(
        b->d_parts, b->d_gate_ptrs, b->d_up_ptrs, b->d_hidden,
        expert_ff, hidden_size, experts);
    glm5next_cuda_iq3_down_grouped<<<(hidden_size + 7) / 8, 256, 0, 0>>>(
        b->d_down_parts, b->d_down_ptrs, b->d_parts, b->d_weights,
        hidden_size, expert_ff, experts);
    glm5next_cuda_moe_reduce<<<(hidden_size + 255) / 256, 256>>>(
        b->d_out, b->d_down_parts, hidden_size, experts);
    CCHK(cudaGetLastError());
    CCHK(cudaMemcpy(b->h_parts, b->d_out, (size_t)hidden_size * sizeof(float),
                    cudaMemcpyDeviceToHost));
    for (int i = 0; i < hidden_size; ++i) out[i] += b->h_parts[i];
#undef CCHK
    return 0;
}

int glm5next_cuda_moe_compute_iq2xs_iq3(glm5next_cuda_moe_bridge *b,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float *out) {
    if (!b || !gate || !up || !down || experts <= 0 || experts > 8 ||
        expert_ff <= 0 || expert_ff % 256 != 0 || hidden_size <= 0 ||
        hidden_size % 256 != 0 || !hidden || !weights || !out) return -1;
#define CCHK_XS(call) do { cudaError_t ce = (call); if (ce != cudaSuccess) { \
        fprintf(stderr, "cuda_moe: %s failed: %s\\n", #call, cudaGetErrorString(ce)); return -1; } } while (0)
    size_t gu_bytes = (size_t)expert_ff * (size_t)(hidden_size / 256) * 74;
    size_t dn_bytes = (size_t)hidden_size * (size_t)(expert_ff / 256) * 98;
    if (ensure_alloc(b, gu_bytes, gu_bytes, dn_bytes, 0, 0, 0,
                     experts, expert_ff, hidden_size, expert_ff) != 0) return -1;
    CCHK_XS(cudaMemcpy(b->d_hidden, hidden, (size_t)hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CCHK_XS(cudaMemcpy(b->d_weights, weights, (size_t)experts * sizeof(float), cudaMemcpyHostToDevice));
    CCHK_XS(cudaMemset(b->d_out, 0, (size_t)hidden_size * sizeof(float)));
    memset(out, 0, (size_t)hidden_size * sizeof(float));
    unsigned char *cg[8], *cu[8], *cd[8];
    for (int e = 0; e < experts; ++e) {
        int cached = cache_prepare(b, gate[e], up[e], down[e], gu_bytes, dn_bytes,
                                   &cg[e], &cu[e], &cd[e]);
        if (cached < 0) return -1;
        if (cached > 0) {
            CCHK_XS(cudaMemcpy(cg[e], gate[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHK_XS(cudaMemcpy(cu[e], up[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHK_XS(cudaMemcpy(cd[e], down[e], dn_bytes, cudaMemcpyHostToDevice));
        }
    }
    CCHK_XS(cudaMemcpy(b->d_gate_ptrs, cg, (size_t)experts * sizeof(*cg), cudaMemcpyHostToDevice));
    CCHK_XS(cudaMemcpy(b->d_up_ptrs, cu, (size_t)experts * sizeof(*cu), cudaMemcpyHostToDevice));
    CCHK_XS(cudaMemcpy(b->d_down_ptrs, cd, (size_t)experts * sizeof(*cd), cudaMemcpyHostToDevice));
    glm5next_cuda_iq2xs_gateup_grouped<<<(expert_ff + 7) / 8, 256>>>(
        b->d_parts, b->d_gate_ptrs, b->d_up_ptrs, b->d_hidden,
        expert_ff, hidden_size, experts);
    glm5next_cuda_iq3_down_grouped<<<(hidden_size + 7) / 8, 256>>>(
        b->d_down_parts, b->d_down_ptrs, b->d_parts, b->d_weights,
        hidden_size, expert_ff, experts);
    glm5next_cuda_moe_reduce<<<(hidden_size + 255) / 256, 256>>>(
        b->d_out, b->d_down_parts, hidden_size, experts);
    CCHK_XS(cudaGetLastError());
    CCHK_XS(cudaMemcpy(b->h_parts, b->d_out, (size_t)hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < hidden_size; ++i) out[i] += b->h_parts[i];
#undef CCHK_XS
    return 0;
}

int glm5next_cuda_moe_compute_iq1_begin(glm5next_cuda_moe_bridge *b,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float clamp) {
    if (!b || !gate || !up || !down || experts <= 0 || experts > 8 ||
        expert_ff <= 0 || hidden_size <= 0 || expert_ff % 256 != 0 ||
        hidden_size % 256 != 0 || !hidden || !weights) return -1;
#define CCHKIQ1(call) do { cudaError_t ce = (call); if (ce != cudaSuccess) { \
        fprintf(stderr, "cuda_moe: %s failed: %s\\n", #call, cudaGetErrorString(ce)); return -1; } } while (0)
    size_t gu_bytes = (size_t)expert_ff * (size_t)(hidden_size / 256) * 50;
    size_t dn_bytes = (size_t)hidden_size * (size_t)(expert_ff / 256) * 50;
    if (ensure_alloc(b, gu_bytes, gu_bytes, dn_bytes, 0, 0, 0,
                     experts, expert_ff, hidden_size, expert_ff) != 0) return -1;
    CCHKIQ1(cudaMemcpy(b->d_hidden, hidden, (size_t)hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CCHKIQ1(cudaMemcpy(b->d_weights, weights, (size_t)experts * sizeof(float), cudaMemcpyHostToDevice));
    CCHKIQ1(cudaMemset(b->d_out, 0, (size_t)hidden_size * sizeof(float)));
    unsigned char *cg[8], *cu[8], *cd[8];
    for (int e = 0; e < experts; ++e) {
        int cached = cache_prepare(b, gate[e], up[e], down[e], gu_bytes, dn_bytes,
                                   &cg[e], &cu[e], &cd[e]);
        if (cached < 0) return -1;
        if (cached > 0) {
            CCHKIQ1(cudaMemcpy(cg[e], gate[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHKIQ1(cudaMemcpy(cu[e], up[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHKIQ1(cudaMemcpy(cd[e], down[e], dn_bytes, cudaMemcpyHostToDevice));
        }
    }
    CCHKIQ1(cudaMemcpy(b->d_gate_ptrs, cg, (size_t)experts * sizeof(*cg), cudaMemcpyHostToDevice));
    CCHKIQ1(cudaMemcpy(b->d_up_ptrs, cu, (size_t)experts * sizeof(*cu), cudaMemcpyHostToDevice));
    CCHKIQ1(cudaMemcpy(b->d_down_ptrs, cd, (size_t)experts * sizeof(*cd), cudaMemcpyHostToDevice));
    int use_dp4a = getenv("GLM5NEXT_CUDA_IQ1_DP4A") &&
                   atoi(getenv("GLM5NEXT_CUDA_IQ1_DP4A")) != 0;
    int cuda_threads = 256;
    const char *threads_env = getenv("GLM5NEXT_CUDA_IQ1_THREADS");
    if (threads_env) {
        int t = atoi(threads_env);
        if (t == 128 || t == 256 || t == 512 || t == 1024) cuda_threads = t;
    }
    int cuda_rpb = cuda_threads / 32;
    dim3 cuda_grid_gate((expert_ff + cuda_rpb - 1) / cuda_rpb, experts);
    dim3 cuda_grid_down((hidden_size + cuda_rpb - 1) / cuda_rpb, experts);
    if (use_dp4a) {
        glm5next_cuda_quantize_q8_32<<<hidden_size / 32, 32>>>(
            b->d_xq, b->d_xscale, b->d_hidden, hidden_size);
        glm5next_cuda_iq1_gateup_grouped_dp4a<<<cuda_grid_gate, cuda_threads>>>(
            b->d_parts, b->d_gate_ptrs, b->d_up_ptrs, b->d_xq, b->d_xscale,
            expert_ff, hidden_size, experts, clamp);
        glm5next_cuda_quantize_q8_32<<<experts * (expert_ff / 32), 32>>>(
            b->d_parts_q, b->d_parts_scale, b->d_parts, experts * expert_ff);
        glm5next_cuda_iq1_down_grouped_dp4a<<<cuda_grid_down, cuda_threads>>>(
            b->d_down_parts, b->d_down_ptrs, b->d_parts_q, b->d_parts_scale,
            b->d_weights, hidden_size, expert_ff, experts);
    } else {
        glm5next_cuda_iq1_gateup_grouped<<<cuda_grid_gate, cuda_threads>>>(
            b->d_parts, b->d_gate_ptrs, b->d_up_ptrs, b->d_hidden,
            expert_ff, hidden_size, experts, clamp);
        glm5next_cuda_iq1_down_grouped<<<cuda_grid_down, cuda_threads>>>(
            b->d_down_parts, b->d_down_ptrs, b->d_parts, b->d_weights,
            hidden_size, expert_ff, experts);
    }
    glm5next_cuda_moe_reduce<<<(hidden_size + 255) / 256, 256>>>(
        b->d_out, b->d_down_parts, hidden_size, experts);
    CCHKIQ1(cudaGetLastError());
#undef CCHKIQ1
    return 0;
}

int glm5next_cuda_moe_compute_iq1_finish(glm5next_cuda_moe_bridge *b,
        int hidden_size, float *out) {
    if (!b || hidden_size <= 0 || !out) return -1;
    cudaError_t ce = cudaMemcpy(b->h_parts, b->d_out,
        (size_t)hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) return -1;
    memcpy(out, b->h_parts, (size_t)hidden_size * sizeof(float));
    return 0;
}

int glm5next_cuda_moe_compute_iq1(glm5next_cuda_moe_bridge *b,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float clamp, float *out) {
    if (glm5next_cuda_moe_compute_iq1_begin(b, gate, up, down, experts,
            expert_ff, hidden_size, hidden, weights, clamp) != 0) return -1;
    return glm5next_cuda_moe_compute_iq1_finish(b, hidden_size, out);
}

int glm5next_cuda_moe_compute_iq2_iq4nl(glm5next_cuda_moe_bridge *b,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float *out) {
    if (!b || !gate || !up || !down || experts <= 0 || experts > 8 ||
        expert_ff <= 0 || hidden_size <= 0 || expert_ff % 256 != 0 ||
        hidden_size % 256 != 0 || !hidden || !weights || !out) return -1;
#define CCHK4(call) do { cudaError_t ce = (call); if (ce != cudaSuccess) { \
        fprintf(stderr, "cuda_moe: %s failed: %s\n", #call, cudaGetErrorString(ce)); return -1; } } while (0)
    size_t gu_bytes = (size_t)expert_ff * (size_t)(hidden_size / 256) * 66;
    size_t dn_bytes = (size_t)hidden_size * (size_t)(expert_ff / 32) * 18;
    if (ensure_alloc(b, gu_bytes, gu_bytes, dn_bytes, 0, 0, 0,
                     experts, expert_ff, hidden_size, expert_ff) != 0) return -1;
    CCHK4(cudaMemcpy(b->d_hidden, hidden, (size_t)hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CCHK4(cudaMemcpy(b->d_weights, weights, (size_t)experts * sizeof(float), cudaMemcpyHostToDevice));
    memset(out, 0, (size_t)hidden_size * sizeof(float));
    CCHK4(cudaMemset(b->d_out, 0, (size_t)hidden_size * sizeof(float)));
    unsigned char *cg[8], *cu[8], *cd[8];
    for (int e = 0; e < experts; ++e) {
        int cached = cache_prepare(b, gate[e], up[e], down[e], gu_bytes, dn_bytes,
                                    &cg[e], &cu[e], &cd[e]);
        if (cached < 0) return -1;
        if (cached > 0) {
            CCHK4(cudaMemcpy(cg[e], gate[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHK4(cudaMemcpy(cu[e], up[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHK4(cudaMemcpy(cd[e], down[e], dn_bytes, cudaMemcpyHostToDevice));
        }
    }
    CCHK4(cudaMemcpy(b->d_gate_ptrs, cg, (size_t)experts * sizeof(*cg),
                     cudaMemcpyHostToDevice));
    CCHK4(cudaMemcpy(b->d_up_ptrs, cu, (size_t)experts * sizeof(*cu),
                     cudaMemcpyHostToDevice));
    CCHK4(cudaMemcpy(b->d_down_ptrs, cd, (size_t)experts * sizeof(*cd),
                     cudaMemcpyHostToDevice));
    glm5next_cuda_iq2_gateup_grouped<<<(expert_ff + 7) / 8, 256>>>(
        b->d_parts, b->d_gate_ptrs, b->d_up_ptrs, b->d_hidden,
        expert_ff, hidden_size, experts);
    glm5next_cuda_iq4nl_down_grouped<<<(hidden_size + 7) / 8, 256>>>(
        b->d_down_parts, b->d_down_ptrs, b->d_parts, b->d_weights,
        hidden_size, expert_ff, experts);
    glm5next_cuda_moe_reduce<<<(hidden_size + 255) / 256, 256>>>(
        b->d_out, b->d_down_parts, hidden_size, experts);
    CCHK4(cudaGetLastError());
    CCHK4(cudaMemcpy(b->h_parts, b->d_out, (size_t)hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < hidden_size; ++i) out[i] += b->h_parts[i];
    CCHK4(cudaDeviceSynchronize());
#undef CCHK4
    return 0;
}

int glm5next_cuda_moe_compute_iq2_iq4xs(glm5next_cuda_moe_bridge *b,
        const void *const *gate, const void *const *up, const void *const *down,
        int experts, int expert_ff, int hidden_size, const float *hidden,
        const float *weights, float *out) {
    if (!b || !gate || !up || !down || experts <= 0 || experts > 8 ||
        expert_ff <= 0 || hidden_size <= 0 || expert_ff % 256 != 0 ||
        hidden_size % 256 != 0 || !hidden || !weights || !out) return -1;
#define CCHKXS(call) do { cudaError_t ce = (call); if (ce != cudaSuccess) { \
        fprintf(stderr, "cuda_moe: %s failed: %s\n", #call, cudaGetErrorString(ce)); return -1; } } while (0)
    size_t gu_bytes = (size_t)expert_ff * (size_t)(hidden_size / 256) * 66;
    size_t dn_bytes = (size_t)hidden_size * (size_t)(expert_ff / 256) * 136;
    if (ensure_alloc(b, gu_bytes, gu_bytes, dn_bytes, 0, 0, 0,
                     experts, expert_ff, hidden_size, expert_ff) != 0) return -1;
    CCHKXS(cudaMemcpy(b->d_hidden, hidden, (size_t)hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CCHKXS(cudaMemcpy(b->d_weights, weights, (size_t)experts * sizeof(float), cudaMemcpyHostToDevice));
    memset(out, 0, (size_t)hidden_size * sizeof(float));
    CCHKXS(cudaMemset(b->d_out, 0, (size_t)hidden_size * sizeof(float)));
    unsigned char *cg[8], *cu[8], *cd[8];
    for (int e = 0; e < experts; ++e) {
        int cached = cache_prepare(b, gate[e], up[e], down[e], gu_bytes, dn_bytes,
                                   &cg[e], &cu[e], &cd[e]);
        if (cached < 0) return -1;
        if (cached > 0) {
            CCHKXS(cudaMemcpy(cg[e], gate[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHKXS(cudaMemcpy(cu[e], up[e], gu_bytes, cudaMemcpyHostToDevice));
            CCHKXS(cudaMemcpy(cd[e], down[e], dn_bytes, cudaMemcpyHostToDevice));
        }
    }
    CCHKXS(cudaMemcpy(b->d_gate_ptrs, cg, (size_t)experts * sizeof(*cg), cudaMemcpyHostToDevice));
    CCHKXS(cudaMemcpy(b->d_up_ptrs, cu, (size_t)experts * sizeof(*cu), cudaMemcpyHostToDevice));
    CCHKXS(cudaMemcpy(b->d_down_ptrs, cd, (size_t)experts * sizeof(*cd), cudaMemcpyHostToDevice));
    glm5next_cuda_iq2_gateup_grouped<<<(expert_ff + 7) / 8, 256>>>(
        b->d_parts, b->d_gate_ptrs, b->d_up_ptrs, b->d_hidden,
        expert_ff, hidden_size, experts);
    glm5next_cuda_iq4xs_down_grouped<<<(hidden_size + 7) / 8, 256>>>(
        b->d_down_parts, b->d_down_ptrs, b->d_parts, b->d_weights,
        hidden_size, expert_ff, experts);
    glm5next_cuda_moe_reduce<<<(hidden_size + 255) / 256, 256>>>(
        b->d_out, b->d_down_parts, hidden_size, experts);
    CCHKXS(cudaGetLastError());
    CCHKXS(cudaMemcpy(b->h_parts, b->d_out, (size_t)hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < hidden_size; ++i) out[i] += b->h_parts[i];
    CCHKXS(cudaDeviceSynchronize());
#undef CCHKXS
    return 0;
}

int glm5next_cuda_moe_compute_shared_q5q6(glm5next_cuda_moe_bridge *b,
        const void *gate, const void *up, const void *down,
        int expert_ff, int hidden_size, const float *hidden, float *out) {
    if (!b || !gate || !up || !down || expert_ff <= 0 || hidden_size <= 0 ||
        expert_ff % 256 != 0 || hidden_size % 256 != 0 || !hidden || !out) return -1;
#define CCHKS(call) do { cudaError_t ce = (call); if (ce != cudaSuccess) { \
        fprintf(stderr, "cuda_moe: %s failed: %s\n", #call, cudaGetErrorString(ce)); return -1; } } while (0)
    size_t gu_bytes = (size_t)expert_ff * (size_t)(hidden_size / 256) * 176;
    size_t dn_bytes = (size_t)hidden_size * (size_t)(expert_ff / 256) * 210;
    if (ensure_alloc(b, gu_bytes, gu_bytes, dn_bytes, 0, 0, 0,
                     1, expert_ff, hidden_size, expert_ff) != 0) return -1;
    unsigned char *cg, *cu, *cd;
    int cached = cache_prepare(b, gate, up, down, gu_bytes, dn_bytes, &cg, &cu, &cd);
    if (cached < 0) return -1;
    if (cached > 0) {
        CCHKS(cudaMemcpy(cg, gate, gu_bytes, cudaMemcpyHostToDevice));
        CCHKS(cudaMemcpy(cu, up, gu_bytes, cudaMemcpyHostToDevice));
        CCHKS(cudaMemcpy(cd, down, dn_bytes, cudaMemcpyHostToDevice));
    }
    CCHKS(cudaMemcpy(b->d_hidden, hidden, (size_t)hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    glm5next_cuda_q5q6_gateup<<<(expert_ff + 7) / 8, 256>>>(
        b->d_parts, cg, cu, b->d_hidden, expert_ff, hidden_size);
    glm5next_cuda_q6_down<<<(hidden_size + 7) / 8, 256>>>(
        b->d_out, cd, b->d_parts, hidden_size, expert_ff);
    CCHKS(cudaGetLastError());
    CCHKS(cudaMemcpy(b->h_parts, b->d_out, (size_t)hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    memcpy(out, b->h_parts, (size_t)hidden_size * sizeof(float));
    CCHKS(cudaDeviceSynchronize());
#undef CCHKS
    return 0;
}
