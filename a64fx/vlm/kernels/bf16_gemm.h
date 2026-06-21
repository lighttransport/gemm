/* bf16_gemm.h — A64FX BF16-storage, FP32-compute SVE GEMM.
 *
 * A64FX (ARMv8.2 + SVE) has NO BFMMLA/BFDOT. The win here is purely
 * memory bandwidth: B is stored as BF16 (half the bytes), loaded with
 * LD1H + shift-left-16 → FP32 in registers, accumulated in FP32.
 *
 * Layout:
 *   A    : [M, K]   FP32 row-major
 *   BT   : [K, N]   BF16 row-major   (W^T already; same shape gemm_fp32 takes)
 *   C    : [M, N]   FP32 row-major   (written, not accumulated)
 */
#ifndef BF16_GEMM_H
#define BF16_GEMM_H

#include <stddef.h>
#include <stdint.h>

/* Convert FP32 → BF16 (round-to-nearest-even). */
void f32_to_bf16(const float *src, uint16_t *dst, size_t n);

/* Convert FP32 row-major [K,N] buffer to BF16 row-major [K,N]. */
void f32_to_bf16_buf(const float *src, uint16_t *dst, size_t n);

/* GEMM: C[M,N] = A[M,K] @ B[K,N], where B is provided as BF16 in BT_bf16.
 * lda = K stride for A, ldb = N stride for BT_bf16, ldc = N stride for C. */
void gemm_bf16_BT(int M, int K, int N,
                  const float    *A,        int lda,
                  const uint16_t *BT_bf16,  int ldb,
                  float          *C,        int ldc);

/* ── Packed-B BF16 path (uses asm microkernel) ───────────────────────────
 * BTP layout matches pack_B_fp32 but with bf16 elements:
 *   per N-block: [K_rounded][NR] uint16_t, where dst[k*NR+n] = BT[k][n_start+n]
 * Use packed_B_bf16_size() to allocate and pack_B_bf16() to fill.
 */
size_t packed_B_bf16_size(int K, int N);
void   pack_B_bf16(int K, int N,
                   const uint16_t *BT_bf16, int ldb,
                   uint16_t *BTP);

void gemm_bf16_BTP(int M, int K, int N,
                   const float    *A,        int lda,
                   const uint16_t *BTP_bf16,           /* prepacked */
                   float          *C,        int ldc);

/* CMG-aware variant. See gemm_fp16_BTP_cmg for semantics. */
void gemm_bf16_BTP_cmg(int M, int K, int N,
                       const float    *A, int lda,
                       const uint16_t * const BTP_repl[/*n_cmgs*/],
                       int n_cmgs,
                       float          *C, int ldc);

/* ── Pair-interleaved BF16 (PV) path ──────────────────────────────────────
 * Two consecutive K-rows interleaved at halfword granularity so a
 * predicated ld1h{.h} places BF16 directly into the upper 16 bits of FP32
 * lanes — no LSL on the FLA pipe. Empirically +80–270% over the LSL asm
 * kernel on representative VIT shapes (1T..48T), bit-identical output.
 *
 * Allocation MUST be packed_B_bf16_pv_size(K,N) bytes; this includes a
 * 64-byte prefix that the asm kernel's "base - 2 bytes" k_even load
 * dereferences for chunk 0 of every N-block. pack_B_bf16_pv writes the
 * full allocation (prefix is zero); gemm_bf16_BTP_pv accepts the same
 * allocation base.
 */
size_t packed_B_bf16_pv_size(int K, int N);
void   pack_B_bf16_pv(int K, int N,
                      const uint16_t *BT_bf16, int ldb,
                      uint16_t *BTP_alloc);

void gemm_bf16_BTP_pv(int M, int K, int N,
                      const float    *A,         int lda,
                      const uint16_t *BTP_alloc,            /* prepacked */
                      float          *C,         int ldc);

void gemm_bf16_BTP_cmg_pv(int M, int K, int N,
                          const float    *A, int lda,
                          const uint16_t * const BTP_alloc_repl[/*n_cmgs*/],
                          int n_cmgs,
                          float          *C, int ldc);

#endif /* BF16_GEMM_H */
