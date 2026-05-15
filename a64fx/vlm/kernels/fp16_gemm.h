/* fp16_gemm.h — A64FX FP16-storage, FP32-compute SVE GEMM.
 *
 * A64FX has native FP16 vector ops, but for accumulator integrity over
 * long K dimensions we load fp16, convert to fp32 in registers (FCVT Z.S),
 * and accumulate in fp32 — same compute path as bf16_gemm.
 *
 * f32 → fp16 conversion saturates on overflow (>65504 ≈ ±Inf, may show up
 * for outlier weights). Set FPCR.FZ16 once at startup to avoid denormal traps.
 *
 * Layouts identical to bf16_gemm:
 *   A    : [M, K]   FP32 row-major
 *   BT   : [K, N]   FP16 row-major (uint16_t bag of bits)
 *   C    : [M, N]   FP32 row-major
 */
#ifndef FP16_GEMM_H
#define FP16_GEMM_H

#include <stddef.h>
#include <stdint.h>

void f32_to_fp16_buf(const float *src, uint16_t *dst, size_t n);

void gemm_fp16_BT(int M, int K, int N,
                  const float    *A,        int lda,
                  const uint16_t *BT_fp16,  int ldb,
                  float          *C,        int ldc);

/* ── Packed-B FP16 path (uses asm microkernel) ───────────────────────────
 * BTP layout matches pack_B_fp32 but with fp16 elements:
 *   per N-block: [K_rounded][NR] uint16_t, where dst[k*NR+n] = BT[k][n_start+n]
 * Use packed_B_fp16_size() to allocate and pack_B_fp16() to fill.
 */
size_t packed_B_fp16_size(int K, int N);
void   pack_B_fp16(int K, int N,
                   const uint16_t *BT_fp16, int ldb,
                   uint16_t *BTP);

void gemm_fp16_BTP(int M, int K, int N,
                   const float    *A,        int lda,
                   const uint16_t *BTP_fp16,           /* prepacked */
                   float          *C,        int ldc);

/* CMG-aware variant. BTP_repl[c] points to a CMG-local copy of BTP for c<n_cmgs.
 * Each OMP thread reads BTP_repl[my_cmg] where my_cmg = tid * n_cmgs / nthreads.
 * Caller must pin OMP threads to CMG-aligned cores (use cmg_pin_omp_threads). */
void gemm_fp16_BTP_cmg(int M, int K, int N,
                       const float    *A, int lda,
                       const uint16_t * const BTP_repl[/*n_cmgs*/],
                       int n_cmgs,
                       float          *C, int ldc);

#endif /* FP16_GEMM_H */
