/* Triton spconv post-process: derive sorted_idx, valid_kernel, valid_kernel_seg
 * from the neighbor_map that the existing HIP runner already builds.
 *
 * Mirrors flex_gemm/kernels/cuda/spconv/migemm_neighmap_pp.cu:
 *   gray[n]   = bitmask of v in [0,V): nmap[n*V+v] != UINT32_MAX
 *   binary[n] = gray-to-binary decode of gray[n]
 *   sorted_idx = argsort(binary)
 *   For each output tile of B1 voxels (in sorted order):
 *     reduced[b]  = OR of gray[sorted_idx[n]] for n in tile
 *     vk_seg[b+1] = popcount(reduced[b])
 *   vk_seg = cumsum
 *   valid_kernel[vk_seg[b] .. vk_seg[b+1]] = bit positions set in reduced[b]
 *
 * Single-threaded reference; fast enough — only invoked once per shape per run
 * after nmap is built (microseconds for N up to 1M, V=27).
 */
#ifndef TRITON_SPCONV_PP_H
#define TRITON_SPCONV_PP_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Pack a (N, V) int32 neighbor_map (with -1 markers for empty taps) into
 * gray + binary codes. Caller pre-allocates gray[N], binary[N]. */
static inline void
t2_neigh_to_gray_binary(int N, int V, const int32_t *nmap,
                        uint32_t *gray, uint32_t *binary)
{
    for (int n = 0; n < N; n++) {
        uint32_t g = 0;
        const int32_t *row = nmap + (size_t)n * V;
        for (int v = 0; v < V; v++) {
            if (row[v] != -1) g |= (1u << v);
        }
        gray[n] = g;
        uint32_t b = g;
        for (int v = 1; v < V; v++) b ^= (g >> v);
        binary[n] = b;
    }
}

/* qsort comparator over a (binary, idx) pair packed into uint64_t. */
static int t2_cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Compute sorted_idx (argsort of binary). Output is int64 to match Triton ABI. */
static inline void
t2_argsort_binary(int N, const uint32_t *binary, int64_t *sorted_idx)
{
    /* Pack (binary[n], n) into a single u64 = (binary << 32) | n, then sort. */
    uint64_t *pkg = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
    for (int n = 0; n < N; n++) {
        pkg[n] = ((uint64_t)binary[n] << 32) | (uint32_t)n;
    }
    qsort(pkg, N, sizeof(uint64_t), t2_cmp_u64);
    for (int n = 0; n < N; n++) {
        sorted_idx[n] = (int64_t)(uint32_t)(pkg[n] & 0xFFFFFFFFu);
    }
    free(pkg);
}

/* Build valid_kernel + valid_kernel_seg given gray[] and sorted_idx[] for a
 * tile-block-size B1. Allocates *out_vk and *out_seg via malloc; caller frees.
 * out_vk_len receives the total number of valid (block, kernel-tap) pairs.
 *
 * vk_seg has length (num_blocks + 1); vk_seg[0] = 0; vk_seg[num_blocks] == total.
 * vk[vk_seg[b] .. vk_seg[b+1]] are kernel-tap indices (0..V-1) used by tile b. */
static inline void
t2_build_valid_kernel(int N, int B1, const uint32_t *gray, const int64_t *sorted_idx,
                      int32_t **out_vk, int32_t **out_seg, int *out_vk_len)
{
    int num_blocks = (N + B1 - 1) / B1;
    int32_t *seg = (int32_t *)malloc((size_t)(num_blocks + 1) * sizeof(int32_t));
    uint32_t *reduced = (uint32_t *)malloc((size_t)num_blocks * sizeof(uint32_t));
    seg[0] = 0;
    for (int b = 0; b < num_blocks; b++) {
        uint32_t r = 0;
        int n_end = (b + 1) * B1;
        if (n_end > N) n_end = N;
        for (int n = b * B1; n < n_end; n++) {
            r |= gray[sorted_idx[n]];
        }
        reduced[b] = r;
        seg[b + 1] = __builtin_popcount(r);
    }
    /* cumsum */
    for (int b = 0; b < num_blocks; b++) seg[b + 1] += seg[b];
    int total = seg[num_blocks];
    int32_t *vk = (int32_t *)malloc((size_t)total * sizeof(int32_t));
    for (int b = 0; b < num_blocks; b++) {
        uint32_t r = reduced[b];
        int p = seg[b];
        while (r) {
            int pos = __builtin_ctz(r);
            vk[p++] = pos;
            r &= r - 1;
        }
    }
    free(reduced);
    *out_vk = vk;
    *out_seg = seg;
    *out_vk_len = total;
}

#endif /* TRITON_SPCONV_PP_H */
