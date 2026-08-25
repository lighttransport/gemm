/*
 * conv2d_sve.c - C11 driver for the fused dual-conv2d patch-embedding.
 *
 * Sits on top of the PROVEN production microkernel
 * micro_kernel_fp32_8x3_unroll4 (8 patches x 48 channels, 4x K-unroll).
 *
 * Per 8-patch tile:
 *   1. gather the tile's pixels straight from the RGB image into the 8x48
 *      A layout  A_packed[k*8 + m]  (24 KB stack for ks=768);
 *   2. loop the dim axis in 48-wide channel blocks and call the SVE asm
 *      microkernel (one call per channel block; W streamed once per tile).
 *
 * The 8x48 kernel owns the hot GEMM (24 accs, ld1rw A broadcasts, ld1w B
 * vectors, 4x-unrolled k-loop, PRFM); everything here is cold.
 */
#include "conv2d_sve.h"
#include "fused_gemm.h"   /* MR (=8), NR (=48) */

#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* from micro_kernel_fp32_8x3.S */
extern void micro_kernel_fp32_8x3_unroll4(const float *A_packed,
                                          const float *B_packed,
                                          float       *C,
                                          long         K,
                                          long         unused,
                                          long         ldc_bytes);

#define CONV2D_KS_MAX (3 * 16 * 16)  /* ps <= 16 */
#define CONV2D_NR     NR             /* 48 channels per 8x48-kernel call */

size_t conv2d_packed_w_size(int ks, int dim)
{
    int K_r = (ks + 3) & ~3;
    int nblocks = (dim + CONV2D_NR - 1) / CONV2D_NR;
    return (size_t)nblocks * K_r * CONV2D_NR;
}

/* Pack the merged dual-conv weights once, in the 8x48 BTP layout that
 * micro_kernel_fp32_8x3_unroll4 expects (identical to pack_B_fp32 with
 * NR=48):  W_packed[nb * K_r * NR + k * NR + n] = B[k, n_start + n], where
 * B = (K0 + K1)^T so  B[k, d] = K0[d, k] + K1[d, k]. */
void conv2d_pack_w_sve(const float *k0, const float *k1, int ks, int dim,
                       float *W)
{
    int K_r = (ks + 3) & ~3;
    int nblocks = (dim + CONV2D_NR - 1) / CONV2D_NR;
    for (int nb = 0; nb < nblocks; nb++) {
        int n_start = nb * CONV2D_NR;
        float *dst = W + (size_t)nb * K_r * CONV2D_NR;
        for (int k = 0; k < ks; k++) {
            for (int n = 0; n < CONV2D_NR; n++) {
                int d = n_start + n;
                dst[k * CONV2D_NR + n] = (d < dim)
                    ? k0[(size_t)d * ks + k] + k1[(size_t)d * ks + k]
                    : 0.0f;
            }
        }
        for (int k = ks; k < K_r; k++)
            for (int n = 0; n < CONV2D_NR; n++)
                dst[k * CONV2D_NR + n] = 0.0f;
    }
}

/* gather: RGB image -> 8x48-kernel A layout AT[ks, 8], zero-padded.
 * dst[k*8 + m] = pixel of patch (p0+m) at k-index k.
 * k-index order: c*ps*ps + dy*ps + dx (matches the merged-weight layout). */
static void gather_tile(const float *rgb, int width, int ps, int gw,
                        int p0, int mcount, int ks, int K_r, float *AT)
{
    int psp2 = ps * ps;
    for (int c = 0; c < 3; c++) {
        for (int dy = 0; dy < ps; dy++) {
            for (int dx = 0; dx < ps; dx++) {
                int k = c * psp2 + dy * ps + dx;
                float *dst = AT + (size_t)k * MR;
                for (int m = 0; m < mcount; m++) {
                    int p = p0 + m, py = p / gw, px = p % gw;
                    dst[m] = rgb[((size_t)(py * ps + dy) * width
                                  + px * ps + dx) * 3 + c];
                }
                /* lanes >= mcount must be exactly zero: the kernel reads all
                 * 8 lanes of every row (garbage * 0.0f => -nan). */
                for (int m = mcount; m < MR; m++)
                    dst[m] = 0.0f;
            }
        }
    }
    for (int k = ks; k < K_r; k++)
        for (int m = 0; m < MR; m++)
            AT[(size_t)k * MR + m] = 0.0f;
}

/* One 8x48-kernel block: compute C[mcount, n_count] = A[mcount, K] x W[n, K]
 * + bias for a single (mb, nb) tile. Full-size blocks write straight to C;
 * partial blocks stage into a local buffer and copy back the valid region. */
static void apply_block(const float *AT, const float *Wb, const float *b,
                        int K_r, int mcount, int n_count, int n_start, int dim,
                        float *C)
{
    if (mcount == CONV2D_TILE && n_count == CONV2D_NR) {
        micro_kernel_fp32_8x3_unroll4(AT, Wb, C, (long)K_r, 0, (long)(dim * 4));
        if (b)
            for (int m = 0; m < MR; m++)
                for (int n = 0; n < CONV2D_NR; n++)
                    C[(size_t)m * dim + n] += b[n];
    } else {
        float tmp[MR * CONV2D_NR];
        micro_kernel_fp32_8x3_unroll4(AT, Wb, tmp, (long)K_r, 0,
                                      (long)(CONV2D_NR * 4));
        (void)n_start;
        for (int m = 0; m < mcount; m++) {
            float *o = C + (size_t)m * dim;
            for (int n = 0; n < n_count; n++)
                o[n] = tmp[m * CONV2D_NR + n] + (b ? b[n] : 0.0f);
        }
    }
}

void conv2d_patch_tile_sve(const float *rgb, int width, int height, int ps,
                           int dim, const float *W, const float *bias,
                           int p0, int mcount, float *out)
{
    (void)height;
    if (mcount <= 0) return;
    if (mcount > CONV2D_TILE) mcount = CONV2D_TILE;
    int ks = 3 * ps * ps;
    if (ks > CONV2D_KS_MAX) return;  /* ps > 16 unsupported */
    int K_r = (ks + 3) & ~3;
    int gw = width / ps;

    float AT[CONV2D_KS_MAX * MR];
    gather_tile(rgb, width, ps, gw, p0, mcount, ks, K_r, AT);

    int nblocks = (dim + CONV2D_NR - 1) / CONV2D_NR;
    for (int nb = 0; nb < nblocks; nb++) {
        int n_start = nb * CONV2D_NR;
        int n_count = (n_start + CONV2D_NR <= dim) ? CONV2D_NR : dim - n_start;
        const float *Wb = W + (size_t)nb * K_r * CONV2D_NR;
        const float *b  = bias ? bias + n_start : NULL;
        float *C = out + (size_t)p0 * dim + n_start;
        apply_block(AT, Wb, b, K_r, mcount, n_count, n_start, dim, C);
    }
}

/* Full-image fused dual conv2d, parallel over (mb, nb) with 2D blocking so
 * each thread's A-slice and W-slice stay L2-resident. The gather is a
 * separate (parallel) phase writing the persistent A_packed scratch, so the
 * (mb, nb) compute loop is a pure GEMM (mirrors gemm_fp32_BTP). */
void conv2d_sve_full(const float *rgb, int width, int height, int ps,
                     int dim, const float *W, const float *bias, float *out)
{
    int ks = 3 * ps * ps;
    if (ks > CONV2D_KS_MAX) return;
    int K_r = (ks + 3) & ~3;
    int gw = width / ps, gh = height / ps, M = gw * gh;
    int M_blocks = (M + CONV2D_TILE - 1) / CONV2D_TILE;
    int N_blocks = (dim + CONV2D_NR - 1) / CONV2D_NR;

    size_t abytes = (size_t)M_blocks * K_r * MR * sizeof(float);
    float *A_packed = (float *)aligned_alloc(64, abytes);
    if (!A_packed) return;

    #pragma omp parallel
    {
        /* NO `nowait`: the (mb,nb) GEMM loop below reads A_packed, so all
         * gathers must complete first. `nowait` let apply_block start while
         * other threads were still writing A_packed -> race -> non-deterministic
         * patch_embed (only surfaced at some geometries, e.g. Kimi-K3 ps=14). */
        #pragma omp for schedule(static)
        for (int mb = 0; mb < M_blocks; mb++) {
            int m_start = mb * CONV2D_TILE;
            int m_count = (m_start + CONV2D_TILE <= M) ? CONV2D_TILE : M - m_start;
            gather_tile(rgb, width, ps, gw, m_start, m_count, ks, K_r,
                        A_packed + (size_t)mb * K_r * MR);
        }
        #pragma omp for collapse(2) schedule(static)
        for (int mb = 0; mb < M_blocks; mb++) {
            for (int nb = 0; nb < N_blocks; nb++) {
                int m_start = mb * CONV2D_TILE;
                int m_count = (m_start + CONV2D_TILE <= M) ? CONV2D_TILE : M - m_start;
                int n_start = nb * CONV2D_NR;
                int n_count = (n_start + CONV2D_NR <= dim) ? CONV2D_NR : dim - n_start;
                const float *AT = A_packed + (size_t)mb * K_r * MR;
                const float *Wb = W + (size_t)nb * K_r * CONV2D_NR;
                const float *b  = bias ? bias + n_start : NULL;
                float *C = out + (size_t)m_start * dim + n_start;
                apply_block(AT, Wb, b, K_r, m_count, n_count, n_start, dim, C);
            }
        }
    }
    free(A_packed);
}
