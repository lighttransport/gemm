/* mv_bench_btp_bf16.c
 *
 * BTP (packed-B) BF16→FP32 GEMM microbench for A64FX.
 *
 * Compares the current BTP layout (k-major within an N-block, BF16 loaded
 * with LD1H{.s}+LSL #16) against a pair-interleaved layout that pairs two
 * consecutive K-rows at HW granularity so a predicated ld1h{.h} pulls
 * BF16 straight into the upper 16 bits of FP32 lanes — no LSL on the FLA
 * pipe.
 *
 * Microtile: MR=8 rows × NR=48 cols (3 SVE vec lanes), K unrolled by 4.
 * Matches the production BTP kernel in a64fx/vlm/kernels/bf16_gemm.c.
 *
 * Pair-interleaved BTP layout (per K-pair, per N-block of NR=48 cols):
 *   3 chunks of 64 bytes = 32 HW = 16 cols × 2 K-values interleaved
 *   chunk c covers cols [16*c .. 16*c+15]:
 *     HW 0  = B[k_even][col0]   HW 1  = B[k_odd][col0]
 *     HW 2  = B[k_even][col1]   HW 3  = B[k_odd][col1]
 *     ...
 *     HW 30 = B[k_even][col15]  HW 31 = B[k_odd][col15]
 * Then per chunk:
 *   ld1h.h p_odd/z, [chunk_ptr - 2 bytes] → B[k_even][...] as FP32
 *   ld1h.h p_odd/z, [chunk_ptr]           → B[k_odd][...]  as FP32
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *        -o mv_bench_btp_bf16 mv_bench_btp_bf16.c -lm
 * Run:   OMP_NUM_THREADS=1 numactl -C 12 -m 4 ./mv_bench_btp_bf16 [M] [N] [K]
 *
 * Default M=576 N=768 K=768 — Qwen3-VL vision block shape.
 */
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef uint16_t bf16_t;

#define MR 8
#define NR 48

static double mono_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *walloc(size_t bytes) {
    void *p = NULL;
    bytes = (bytes + 63) & ~(size_t)63;
    if (posix_memalign(&p, 64, bytes + 64) != 0) return NULL;
    memset(p, 0, bytes + 64);
    return p;
}

static bf16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    bits += ((bits >> 16) & 1) + 0x7FFF;
    return (bf16_t)(bits >> 16);
}

static float bf16_to_f32(bf16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* ─────────────────────────────────────────────────────────────────────
 * Layout helpers
 * ───────────────────────────────────────────────────────────────────── */

/* Standard BTP: per N-block of NR cols, dst[k * NR + n].
 * Total size per N-block: K_rounded × NR HW. */
static void pack_B_bf16_btp(int K, int N,
                            const bf16_t *BT, int ldb,
                            bf16_t *BTP)
{
    const int N_blocks  = (N + NR - 1) / NR;
    const int K_rounded = ((K + 3) / 4) * 4;
    for (int nb = 0; nb < N_blocks; nb++) {
        int n_start = nb * NR;
        int n_count = (n_start + NR <= N) ? NR : N - n_start;
        bf16_t *dst = BTP + (size_t)nb * K_rounded * NR;
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < n_count; n++) {
                dst[(size_t)k * NR + n] = BT[(size_t)k * ldb + n_start + n];
            }
            for (int n = n_count; n < NR; n++) dst[(size_t)k * NR + n] = 0;
        }
        for (int k = K; k < K_rounded; k++) {
            for (int n = 0; n < NR; n++) dst[(size_t)k * NR + n] = 0;
        }
    }
}

/* Pair-interleaved BTP: per N-block, K processed in pairs of 2.
 * Per K-pair: 3 chunks × 32 HW. Within chunk c (cols 16c..16c+15):
 *   HW[2*i + 0] = BT[k_pair][16c + i]
 *   HW[2*i + 1] = BT[k_pair+1][16c + i]
 * Total size per N-block: same as standard BTP. */
static void pack_B_bf16_btp_pv(int K, int N,
                               const bf16_t *BT, int ldb,
                               bf16_t *BTP)
{
    const int N_blocks  = (N + NR - 1) / NR;
    const int K_rounded = ((K + 3) / 4) * 4;
    /* +1 prepend HW per N-block of zeros so [-2] read for k_even
     * never crosses the buffer boundary at k_pair=0, chunk=0.
     * Simpler: caller allocates ((N_blocks * K_rounded * NR) + 1) HW
     * and BTP points 1 HW past the start. */
    for (int nb = 0; nb < N_blocks; nb++) {
        int n_start = nb * NR;
        int n_count = (n_start + NR <= N) ? NR : N - n_start;
        bf16_t *dst = BTP + (size_t)nb * K_rounded * NR;
        for (int kp = 0; kp < K_rounded; kp += 2) {
            int k0 = kp;
            int k1 = kp + 1;
            for (int c = 0; c < 3; c++) {
                bf16_t *chunk = dst + (size_t)(kp / 2) * (NR * 2) + c * 32;
                for (int i = 0; i < 16; i++) {
                    int col = c * 16 + i;
                    bf16_t v0 = 0, v1 = 0;
                    if (k0 < K && col < n_count)
                        v0 = BT[(size_t)k0 * ldb + n_start + col];
                    if (k1 < K && col < n_count)
                        v1 = BT[(size_t)k1 * ldb + n_start + col];
                    chunk[2 * i + 0] = v0;
                    chunk[2 * i + 1] = v1;
                }
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Microtile kernels (8 rows × 3 SVE vec cols)
 * ───────────────────────────────────────────────────────────────────── */

static inline svfloat32_t svld1_bf16_f32(svbool_t pg, const bf16_t *p) {
    svuint32_t raw = svld1uh_u32(pg, (const uint16_t *)p);
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, raw, 16));
}

#define FMA_ROW(M_IDX) \
    do { \
        float av0 = A[(size_t)(M_IDX) * lda + (k + 0)]; \
        float av1 = A[(size_t)(M_IDX) * lda + (k + 1)]; \
        float av2 = A[(size_t)(M_IDX) * lda + (k + 2)]; \
        float av3 = A[(size_t)(M_IDX) * lda + (k + 3)]; \
        a##M_IDX##0 = svmla_n_f32_x(pg, a##M_IDX##0, b0_0, av0); \
        a##M_IDX##1 = svmla_n_f32_x(pg, a##M_IDX##1, b1_0, av0); \
        a##M_IDX##2 = svmla_n_f32_x(pg, a##M_IDX##2, b2_0, av0); \
        a##M_IDX##0 = svmla_n_f32_x(pg, a##M_IDX##0, b0_1, av1); \
        a##M_IDX##1 = svmla_n_f32_x(pg, a##M_IDX##1, b1_1, av1); \
        a##M_IDX##2 = svmla_n_f32_x(pg, a##M_IDX##2, b2_1, av1); \
        a##M_IDX##0 = svmla_n_f32_x(pg, a##M_IDX##0, b0_2, av2); \
        a##M_IDX##1 = svmla_n_f32_x(pg, a##M_IDX##1, b1_2, av2); \
        a##M_IDX##2 = svmla_n_f32_x(pg, a##M_IDX##2, b2_2, av2); \
        a##M_IDX##0 = svmla_n_f32_x(pg, a##M_IDX##0, b0_3, av3); \
        a##M_IDX##1 = svmla_n_f32_x(pg, a##M_IDX##1, b1_3, av3); \
        a##M_IDX##2 = svmla_n_f32_x(pg, a##M_IDX##2, b2_3, av3); \
    } while (0)

/* LSL baseline — mirrors microtile_8x3vl in bf16_gemm.c */
static void microtile_8x3vl_lsl(
    int K,
    const float  *A,  int lda,
    const bf16_t *BT, int ldb,   /* ldb = NR for BTP */
    float        *C,  int ldc)
{
    const svbool_t pg = svptrue_b32();
    const int VL = (int)svcntw();

    svfloat32_t a00 = svdup_f32(0), a01 = svdup_f32(0), a02 = svdup_f32(0);
    svfloat32_t a10 = svdup_f32(0), a11 = svdup_f32(0), a12 = svdup_f32(0);
    svfloat32_t a20 = svdup_f32(0), a21 = svdup_f32(0), a22 = svdup_f32(0);
    svfloat32_t a30 = svdup_f32(0), a31 = svdup_f32(0), a32 = svdup_f32(0);
    svfloat32_t a40 = svdup_f32(0), a41 = svdup_f32(0), a42 = svdup_f32(0);
    svfloat32_t a50 = svdup_f32(0), a51 = svdup_f32(0), a52 = svdup_f32(0);
    svfloat32_t a60 = svdup_f32(0), a61 = svdup_f32(0), a62 = svdup_f32(0);
    svfloat32_t a70 = svdup_f32(0), a71 = svdup_f32(0), a72 = svdup_f32(0);

    int k = 0;
    for (; k + 4 <= K; k += 4) {
        svfloat32_t b0_0 = svld1_bf16_f32(pg, BT + (size_t)(k + 0) * ldb);
        svfloat32_t b1_0 = svld1_bf16_f32(pg, BT + (size_t)(k + 0) * ldb + VL);
        svfloat32_t b2_0 = svld1_bf16_f32(pg, BT + (size_t)(k + 0) * ldb + 2 * VL);
        svfloat32_t b0_1 = svld1_bf16_f32(pg, BT + (size_t)(k + 1) * ldb);
        svfloat32_t b1_1 = svld1_bf16_f32(pg, BT + (size_t)(k + 1) * ldb + VL);
        svfloat32_t b2_1 = svld1_bf16_f32(pg, BT + (size_t)(k + 1) * ldb + 2 * VL);
        svfloat32_t b0_2 = svld1_bf16_f32(pg, BT + (size_t)(k + 2) * ldb);
        svfloat32_t b1_2 = svld1_bf16_f32(pg, BT + (size_t)(k + 2) * ldb + VL);
        svfloat32_t b2_2 = svld1_bf16_f32(pg, BT + (size_t)(k + 2) * ldb + 2 * VL);
        svfloat32_t b0_3 = svld1_bf16_f32(pg, BT + (size_t)(k + 3) * ldb);
        svfloat32_t b1_3 = svld1_bf16_f32(pg, BT + (size_t)(k + 3) * ldb + VL);
        svfloat32_t b2_3 = svld1_bf16_f32(pg, BT + (size_t)(k + 3) * ldb + 2 * VL);

        FMA_ROW(0); FMA_ROW(1); FMA_ROW(2); FMA_ROW(3);
        FMA_ROW(4); FMA_ROW(5); FMA_ROW(6); FMA_ROW(7);
    }

    #define STORE_ROW(M_IDX) \
        do { \
            svst1_f32(pg, C + (size_t)(M_IDX) * ldc,          a##M_IDX##0); \
            svst1_f32(pg, C + (size_t)(M_IDX) * ldc + VL,     a##M_IDX##1); \
            svst1_f32(pg, C + (size_t)(M_IDX) * ldc + 2 * VL, a##M_IDX##2); \
        } while (0)
    STORE_ROW(0); STORE_ROW(1); STORE_ROW(2); STORE_ROW(3);
    STORE_ROW(4); STORE_ROW(5); STORE_ROW(6); STORE_ROW(7);
    #undef STORE_ROW
}

#undef FMA_ROW

/* Predicated p_odd loader: bf16 lives at odd HW positions of [base],
 * lands in upper 16 bits of .s lanes = valid FP32. */
static inline svfloat32_t svld1h_pv_odd(svbool_t p_odd_h, const bf16_t *base) {
    svuint16_t raw = svld1_u16(p_odd_h, (const uint16_t *)base);
    return svreinterpret_f32_u16(raw);
}

#define FMA_ROW_PV(M_IDX) \
    do { \
        float av0 = A[(size_t)(M_IDX) * lda + (k + 0)]; \
        float av1 = A[(size_t)(M_IDX) * lda + (k + 1)]; \
        float av2 = A[(size_t)(M_IDX) * lda + (k + 2)]; \
        float av3 = A[(size_t)(M_IDX) * lda + (k + 3)]; \
        a##M_IDX##0 = svmla_n_f32_x(pg, a##M_IDX##0, b0_0, av0); \
        a##M_IDX##1 = svmla_n_f32_x(pg, a##M_IDX##1, b1_0, av0); \
        a##M_IDX##2 = svmla_n_f32_x(pg, a##M_IDX##2, b2_0, av0); \
        a##M_IDX##0 = svmla_n_f32_x(pg, a##M_IDX##0, b0_1, av1); \
        a##M_IDX##1 = svmla_n_f32_x(pg, a##M_IDX##1, b1_1, av1); \
        a##M_IDX##2 = svmla_n_f32_x(pg, a##M_IDX##2, b2_1, av1); \
        a##M_IDX##0 = svmla_n_f32_x(pg, a##M_IDX##0, b0_2, av2); \
        a##M_IDX##1 = svmla_n_f32_x(pg, a##M_IDX##1, b1_2, av2); \
        a##M_IDX##2 = svmla_n_f32_x(pg, a##M_IDX##2, b2_2, av2); \
        a##M_IDX##0 = svmla_n_f32_x(pg, a##M_IDX##0, b0_3, av3); \
        a##M_IDX##1 = svmla_n_f32_x(pg, a##M_IDX##1, b1_3, av3); \
        a##M_IDX##2 = svmla_n_f32_x(pg, a##M_IDX##2, b2_3, av3); \
    } while (0)

/* Pair-interleaved BTP path. BT points to start of N-block pv buffer.
 * Per K-pair stride = NR * 2 HW. Per chunk (16 cols) = 32 HW.
 * k_even loaded with offset -1 HW (= -2 bytes) from chunk start. */
static void microtile_8x3vl_pv(
    int K,
    const float  *A,  int lda,
    const bf16_t *BT,            /* pv layout, no ldb arg needed */
    float        *C,  int ldc)
{
    const svbool_t pg = svptrue_b32();

    /* p_odd over halfwords: HW lanes 1,3,5,...,31 active. */
    svbool_t p_all_h = svptrue_b16();
    svuint16_t idx_h = svindex_u16(0, 1);
    svbool_t p_odd = svcmpne_n_u16(p_all_h,
                                    svand_n_u16_x(p_all_h, idx_h, 1), 0);

    svfloat32_t a00 = svdup_f32(0), a01 = svdup_f32(0), a02 = svdup_f32(0);
    svfloat32_t a10 = svdup_f32(0), a11 = svdup_f32(0), a12 = svdup_f32(0);
    svfloat32_t a20 = svdup_f32(0), a21 = svdup_f32(0), a22 = svdup_f32(0);
    svfloat32_t a30 = svdup_f32(0), a31 = svdup_f32(0), a32 = svdup_f32(0);
    svfloat32_t a40 = svdup_f32(0), a41 = svdup_f32(0), a42 = svdup_f32(0);
    svfloat32_t a50 = svdup_f32(0), a51 = svdup_f32(0), a52 = svdup_f32(0);
    svfloat32_t a60 = svdup_f32(0), a61 = svdup_f32(0), a62 = svdup_f32(0);
    svfloat32_t a70 = svdup_f32(0), a71 = svdup_f32(0), a72 = svdup_f32(0);

    int k = 0;
    for (; k + 4 <= K; k += 4) {
        /* k_pair0 = (k+0, k+1), k_pair1 = (k+2, k+3) */
        const bf16_t *kp0 = BT + (size_t)(k / 2) * (NR * 2);
        const bf16_t *kp1 = kp0 + (NR * 2);

        /* Each chunk is 32 HW. Chunk 0 of kp0 starts at kp0+0;
         * k_even is read at offset -1 HW (= -2 bytes). */
        svfloat32_t b0_0 = svld1h_pv_odd(p_odd, kp0 + 0  - 1);   /* k+0, cols 0..15 */
        svfloat32_t b1_0 = svld1h_pv_odd(p_odd, kp0 + 32 - 1);   /* k+0, cols 16..31 */
        svfloat32_t b2_0 = svld1h_pv_odd(p_odd, kp0 + 64 - 1);   /* k+0, cols 32..47 */
        svfloat32_t b0_1 = svld1h_pv_odd(p_odd, kp0 + 0);        /* k+1, cols 0..15 */
        svfloat32_t b1_1 = svld1h_pv_odd(p_odd, kp0 + 32);       /* k+1, cols 16..31 */
        svfloat32_t b2_1 = svld1h_pv_odd(p_odd, kp0 + 64);       /* k+1, cols 32..47 */
        svfloat32_t b0_2 = svld1h_pv_odd(p_odd, kp1 + 0  - 1);   /* k+2, cols 0..15 */
        svfloat32_t b1_2 = svld1h_pv_odd(p_odd, kp1 + 32 - 1);   /* k+2, cols 16..31 */
        svfloat32_t b2_2 = svld1h_pv_odd(p_odd, kp1 + 64 - 1);   /* k+2, cols 32..47 */
        svfloat32_t b0_3 = svld1h_pv_odd(p_odd, kp1 + 0);        /* k+3, cols 0..15 */
        svfloat32_t b1_3 = svld1h_pv_odd(p_odd, kp1 + 32);       /* k+3, cols 16..31 */
        svfloat32_t b2_3 = svld1h_pv_odd(p_odd, kp1 + 64);       /* k+3, cols 32..47 */

        FMA_ROW_PV(0); FMA_ROW_PV(1); FMA_ROW_PV(2); FMA_ROW_PV(3);
        FMA_ROW_PV(4); FMA_ROW_PV(5); FMA_ROW_PV(6); FMA_ROW_PV(7);
    }

    const int VL = (int)svcntw();
    #define STORE_ROW(M_IDX) \
        do { \
            svst1_f32(pg, C + (size_t)(M_IDX) * ldc,          a##M_IDX##0); \
            svst1_f32(pg, C + (size_t)(M_IDX) * ldc + VL,     a##M_IDX##1); \
            svst1_f32(pg, C + (size_t)(M_IDX) * ldc + 2 * VL, a##M_IDX##2); \
        } while (0)
    STORE_ROW(0); STORE_ROW(1); STORE_ROW(2); STORE_ROW(3);
    STORE_ROW(4); STORE_ROW(5); STORE_ROW(6); STORE_ROW(7);
    #undef STORE_ROW
}

#undef FMA_ROW_PV

/* ─────────────────────────────────────────────────────────────────────
 * ASM kernels — link against bf16_gemm prebuilt or the .S files directly
 * ───────────────────────────────────────────────────────────────────── */

extern void micro_kernel_bf16B_8x3_unroll4(
    const float    *A_packed,
    const uint16_t *B_packed_bf16,
    float          *C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes);

extern void micro_kernel_bf16B_8x3_unroll4_pv(
    const float    *A_packed,
    const uint16_t *B_packed_pv,
    float          *C,
    int64_t K,
    int64_t unused,
    int64_t ldc_bytes);

/* A_packed [K][MR] per M-block (k-major within M-block, MR=8 contiguous m). */
static void pack_A_fp32(int M, int K, const float *A, float *A_packed) {
    int M_blocks = (M + MR - 1) / MR;
    int K_r = ((K + 3) / 4) * 4;
    for (int mb = 0; mb < M_blocks; mb++) {
        int m_start = mb * MR;
        int m_count = (m_start + MR <= M) ? MR : M - m_start;
        float *dst = A_packed + (size_t)mb * K_r * MR;
        for (int k = 0; k < K; k++) {
            for (int m = 0; m < m_count; m++) dst[k * MR + m] = A[(m_start + m) * K + k];
            for (int m = m_count; m < MR; m++) dst[k * MR + m] = 0.f;
        }
        for (int k = K; k < K_r; k++)
            for (int m = 0; m < MR; m++) dst[k * MR + m] = 0.f;
    }
}

static void gemm_asm_lsl(int M, int K, int N,
                         const float *A_packed,
                         const bf16_t *BTP,
                         float *C)
{
    const int K_r = ((K + 3) / 4) * 4;
    const int M_blocks = M / MR;
    const int N_blocks = (N + NR - 1) / NR;
#pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks; nb++) {
            const float  *A_tile = A_packed + (size_t)mb * K_r * MR;
            const bf16_t *B_tile = BTP      + (size_t)nb * K_r * NR;
            float        *C_tile = C        + (size_t)mb * MR * N + nb * NR;
            micro_kernel_bf16B_8x3_unroll4(
                A_tile, B_tile, C_tile,
                (int64_t)K_r, 0, (int64_t)N * sizeof(float));
        }
    }
}

static void gemm_asm_pv(int M, int K, int N,
                        const float *A_packed,
                        const bf16_t *BTP_pv,
                        float *C)
{
    const int K_r = ((K + 3) / 4) * 4;
    const int M_blocks = M / MR;
    const int N_blocks = (N + NR - 1) / NR;
#pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks; nb++) {
            const float  *A_tile = A_packed + (size_t)mb * K_r * MR;
            const bf16_t *B_tile = BTP_pv   + (size_t)nb * K_r * NR;
            float        *C_tile = C        + (size_t)mb * MR * N + nb * NR;
            micro_kernel_bf16B_8x3_unroll4_pv(
                A_tile, B_tile, C_tile,
                (int64_t)K_r, 0, (int64_t)N * sizeof(float));
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Driver
 * ───────────────────────────────────────────────────────────────────── */

static void gemm_lsl(int M, int K, int N,
                     const float  *A,
                     const bf16_t *BTP,
                     float        *C)
{
    const int K_rounded = ((K + 3) / 4) * 4;
    const int N_blocks  = (N + NR - 1) / NR;
    const int M_blocks  = M / MR;
#pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks; nb++) {
            const float  *A_tile = A   + (size_t)mb * MR * K;
            const bf16_t *B_tile = BTP + (size_t)nb * K_rounded * NR;
            float        *C_tile = C   + (size_t)mb * MR * N + nb * NR;
            microtile_8x3vl_lsl(K_rounded,
                                A_tile, K,
                                B_tile, NR,
                                C_tile, N);
        }
    }
}

static void gemm_pv(int M, int K, int N,
                    const float  *A,
                    const bf16_t *BTP_pv,
                    float        *C)
{
    const int K_rounded = ((K + 3) / 4) * 4;
    const int N_blocks  = (N + NR - 1) / NR;
    const int M_blocks  = M / MR;
#pragma omp parallel for collapse(2) schedule(static)
    for (int mb = 0; mb < M_blocks; mb++) {
        for (int nb = 0; nb < N_blocks; nb++) {
            const float  *A_tile = A      + (size_t)mb * MR * K;
            const bf16_t *B_tile = BTP_pv + (size_t)nb * K_rounded * NR;
            float        *C_tile = C      + (size_t)mb * MR * N + nb * NR;
            microtile_8x3vl_pv(K_rounded, A_tile, K, B_tile, C_tile, N);
        }
    }
}

static int verify_close(const float *ref, const float *got, size_t n,
                        float rtol, float atol) {
    double max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < n; i++) {
        float r = ref[i], g = got[i];
        float d = fabsf(r - g);
        if (d > max_abs) max_abs = d;
        float denom = fabsf(r);
        if (denom > 1e-9f) {
            float rel = d / denom;
            if (rel > max_rel) max_rel = rel;
        }
        if (d > atol && fabsf(r) > 1e-9f && (d / fabsf(r)) > rtol) {
            fprintf(stderr, "mismatch at %zu: ref=%g got=%g (abs=%g rel=%g)\n",
                    i, r, g, d, d / fabsf(r));
            return 0;
        }
    }
    fprintf(stderr, "verify ok: max_abs=%g max_rel=%g\n", max_abs, max_rel);
    return 1;
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 576;   /* must be %MR == 0 */
    int N = (argc > 2) ? atoi(argv[2]) : 768;   /* must be %NR == 0 */
    int K = (argc > 3) ? atoi(argv[3]) : 768;   /* will be padded to %4 */
    int reps = (argc > 4) ? atoi(argv[4]) : 50;

    if (M % MR) { fprintf(stderr, "M must be multiple of %d\n", MR); return 1; }
    if (N % NR) { fprintf(stderr, "N must be multiple of %d\n", NR); return 1; }
    int K_rounded = ((K + 3) / 4) * 4;
    if (K != K_rounded) {
        fprintf(stderr, "rounding K %d → %d\n", K, K_rounded);
        K = K_rounded;
    }

    int nthr = omp_get_max_threads();
    int VL = (int)svcntw();
    fprintf(stderr, "M=%d N=%d K=%d  MR=%d NR=%d VL=%d  threads=%d  reps=%d\n",
            M, N, K, MR, NR, VL, nthr, reps);

    float  *A   = walloc((size_t)M * K * sizeof(float));
    bf16_t *BT  = walloc((size_t)K * N * sizeof(bf16_t));
    float  *Clsl = walloc((size_t)M * N * sizeof(float));
    float  *Cpv  = walloc((size_t)M * N * sizeof(float));

    /* init */
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
    for (size_t i = 0; i < (size_t)K * N; i++) {
        float f = (rand() / (float)RAND_MAX) * 2.f - 1.f;
        BT[i] = f32_to_bf16(f);
    }

    const int N_blocks = (N + NR - 1) / NR;
    size_t btp_elems = (size_t)N_blocks * K * NR;
    bf16_t *BTP    = walloc(btp_elems * sizeof(bf16_t));
    /* Pad with one extra HW before the buffer for the [-2] read. */
    bf16_t *BTP_pv_raw = walloc((btp_elems + 1) * sizeof(bf16_t));
    bf16_t *BTP_pv     = BTP_pv_raw + 1;

    pack_B_bf16_btp(K, N, BT, N, BTP);
    pack_B_bf16_btp_pv(K, N, BT, N, BTP_pv);

    /* C-path warmup + verify */
    gemm_lsl(M, K, N, A, BTP, Clsl);
    gemm_pv (M, K, N, A, BTP_pv, Cpv);
    if (!verify_close(Clsl, Cpv, (size_t)M * N, 1e-4f, 1e-3f)) {
        fprintf(stderr, "C-path VERIFY FAILED\n");
        return 1;
    }

    /* asm path: pack A, run both LSL and PV asm kernels */
    float *A_packed = walloc((size_t)((M + MR - 1) / MR) * K * MR * sizeof(float));
    float *Casm_lsl = walloc((size_t)M * N * sizeof(float));
    float *Casm_pv  = walloc((size_t)M * N * sizeof(float));
    pack_A_fp32(M, K, A, A_packed);

    gemm_asm_lsl(M, K, N, A_packed, BTP, Casm_lsl);
    gemm_asm_pv (M, K, N, A_packed, BTP_pv, Casm_pv);
    fprintf(stderr, "asm-LSL vs C-LSL: ");
    if (!verify_close(Clsl, Casm_lsl, (size_t)M * N, 1e-4f, 1e-3f)) {
        fprintf(stderr, "asm LSL VERIFY FAILED\n"); return 1;
    }
    fprintf(stderr, "asm-PV  vs C-LSL: ");
    if (!verify_close(Clsl, Casm_pv,  (size_t)M * N, 1e-4f, 1e-3f)) {
        fprintf(stderr, "asm PV  VERIFY FAILED\n"); return 1;
    }

    double flops = 2.0 * M * N * K;

    /* C-LSL bench */
    double t0 = mono_sec();
    for (int r = 0; r < reps; r++) gemm_lsl(M, K, N, A, BTP, Clsl);
    double t1 = mono_sec();
    double c_lsl_gflops = (flops * reps) / (t1 - t0) / 1e9;

    /* C-PV bench */
    t0 = mono_sec();
    for (int r = 0; r < reps; r++) gemm_pv(M, K, N, A, BTP_pv, Cpv);
    t1 = mono_sec();
    double c_pv_gflops = (flops * reps) / (t1 - t0) / 1e9;

    /* ASM-LSL bench */
    t0 = mono_sec();
    for (int r = 0; r < reps; r++) gemm_asm_lsl(M, K, N, A_packed, BTP, Casm_lsl);
    t1 = mono_sec();
    double a_lsl_gflops = (flops * reps) / (t1 - t0) / 1e9;

    /* ASM-PV bench */
    t0 = mono_sec();
    for (int r = 0; r < reps; r++) gemm_asm_pv(M, K, N, A_packed, BTP_pv, Casm_pv);
    t1 = mono_sec();
    double a_pv_gflops = (flops * reps) / (t1 - t0) / 1e9;

    printf("C-LSL  : %8.2f GFLOP/s\n", c_lsl_gflops);
    printf("C-PV   : %8.2f GFLOP/s    gain over C-LSL = %+.1f%%\n",
           c_pv_gflops, 100.0 * (c_pv_gflops / c_lsl_gflops - 1.0));
    printf("ASM-LSL: %8.2f GFLOP/s\n", a_lsl_gflops);
    printf("ASM-PV : %8.2f GFLOP/s    gain over ASM-LSL = %+.1f%%\n",
           a_pv_gflops, 100.0 * (a_pv_gflops / a_lsl_gflops - 1.0));

    free(A); free(BT); free(BTP); free(BTP_pv_raw); free(Clsl); free(Cpv);
    free(A_packed); free(Casm_lsl); free(Casm_pv);
    return 0;
}
