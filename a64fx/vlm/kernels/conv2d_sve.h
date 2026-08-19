/*
 * conv2d_sve.h - fused dual-conv2d patch-embedding for A64FX SVE.
 *
 * Targets the Qwen3-VL vision "conv2d" (patch embedding): two 16x16x3
 * convolutions with stride == kernel size, applied to the same image and
 * added together. Because the stride equals the kernel size, patches never
 * overlap, so the conv is algebraically a plain GEMM:
 *
 *     C[p, d] = sum_{j < ks} (K0[d,j] + K1[d,j]) * patch[p, j] + b[d]
 *     C : [n_patches, dim]     patch : [n_patches, ks],  ks = 3*ps*ps
 *
 * Implementation = C11 driver on top of the PROVEN production microkernel
 * micro_kernel_fp32_8x3_unroll4 (8 patches x 48 channels, 4x K-unroll).
 * The earlier hand-written 16x8 / 8x8 conv2d microkernels were abandoned:
 * a hand-rolled 8-wide ld1rw + 8-FMA inner loop mis-executes on this A64FX
 * (data-dependent accumulation loss + a phantom OOB store). The 8x48 kernel
 * is bit-exact in production, so we reuse it and keep the value in the two
 * things the generic GEMM path does NOT do for the VLM:
 *
 *   1. FUSED gather: the tile's pixels are gathered straight from the RGB
 *      image into the 8x48 A layout  A_packed[k*8 + m]  (24 KB for ks=768,
 *      stack). No intermediate [n_patches, ks] buffer, no separate
 *      pack_A pass.
 *   2. MERGED dual-conv weights: K0 + K1 are summed ONCE at cache build and
 *      packed in the 8x48 BTP layout, so the runtime is a single GEMM
 *      (not two convs).
 *
 * Tile = CONV2D_TILE = 8 consecutive patches (one MR-block of the 8x48
 * kernel). A64FX store-queue erratum keeps us at <= 8 rows per call (the
 * 8x48 kernel stores exactly 8 row-bases; a 16-row kernel writes a phantom
 * 17th row).
 *
 * Weight packing (done ONCE at cache build, mirrors pack_B_fp32 with NR=48):
 *
 *     W_packed[(d0/48) * K_r * 48 + k * 48 + i] = K0[d0+i, k] + K1[d0+i, k]
 *     where d0 = 0, 48, 96, ... and K_r = ks rounded up to a multiple of 4
 *     (pad rows k >= ks are zero).
 *
 * dim%48 remainders (16 channels for Qwen3-VL dim=1024) are handled by the
 * kernel writing into a local 8x48 buffer and copying only the valid tail.
 */
#ifndef CONV2D_SVE_H
#define CONV2D_SVE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Patches per tile (one 8x48-kernel MR-block). */
#define CONV2D_TILE 8

/* Size in floats of the packed merged weight matrix for a given shape. */
size_t conv2d_packed_w_size(int ks, int dim);

/* Pack the merged dual-conv weights once (cache build time), in the 8x48
 * BTP layout expected by micro_kernel_fp32_8x3_unroll4.
 * k0/k1 : [dim, ks] fp32 row-major each (the two conv kernels).
 * W     : output buffer of conv2d_packed_w_size(ks, dim) floats. */
void conv2d_pack_w_sve(const float *k0, const float *k1, int ks, int dim,
                       float *W);

/* Compute one tile of <= CONV2D_TILE consecutive patches (starting at
 * patch p0) of the fused dual conv2d:
 *
 *   rgb   : normalized image [height*width*3] fp32 (pixel-major RGB as
 *           produced by vision_normalize_image)
 *   W     : packed merged kernel from conv2d_pack_w_sve
 *   bias  : [dim] fp32, or NULL for no bias
 *   p0    : first patch index of the tile (row-major patch grid, W_img/ps)
 *   mcount: number of patches in this tile (1..CONV2D_TILE)
 *   out   : [n_patches, dim] fp32; this routine writes rows p0..p0+mcount-1
 *
 * Single-threaded; parallelize over non-overlapping tiles. */
void conv2d_patch_tile_sve(const float *rgb, int width, int height, int ps,
                           int dim, const float *W, const float *bias,
                           int p0, int mcount, float *out);

/* Full-image fused dual conv2d: computes all n_patches = (W/ps)*(H/ps) rows
 * of out, parallel over (M-block x N-block) with 2D blocking (OpenMP). This
 * is the function the VLM calls; conv2d_patch_tile_sve is exposed for
 * single-tile use / testing. */
void conv2d_sve_full(const float *rgb, int width, int height, int ps,
                     int dim, const float *W, const float *bias, float *out);

#ifdef __cplusplus
}
#endif

#endif /* CONV2D_SVE_H */
