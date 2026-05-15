/* mv_bench_bf16_8row.c
 *
 * Microbench matching common/ggml_dequant.h matvec_bf16_8row layout:
 *   8 separate weight rows w0..w7 of length K (BF16, row-major),
 *   shared activation x of length K (FP32),
 *   output dst[0..7] = w_r · x.
 *
 * Two kernels:
 *   row8_lsl : current SVE_BF16_TO_F32 macro path
 *              (8 × LD1H{.s} + 8 × LSL + 1 × LD1W + 8 × FMLA per k-step)
 *   row8_pv  : adjacent rows paired in interleaved buffers, p_odd
 *              predicated loads extract both rows as FP32 with no LSL
 *              (4 pairs × 2 × LD1H{.h} + 1 × LD1W + 8 × FMLA per k-step)
 *
 * Pair-buffer layout (per row pair, 16 k-steps = 16 lanes per SVE vec):
 *   HW 0  = rA[0]   HW 1  = rB[0]
 *   HW 2  = rA[1]   HW 3  = rB[1]
 *   ...
 *   HW 30 = rA[15]  HW 31 = rB[15]
 * Then:
 *   ld1h.h p_odd, [pair_buf + 16*k_chunk - 2]  → rA[k..k+15] as FP32
 *   ld1h.h p_odd, [pair_buf + 16*k_chunk     ] → rB[k..k+15] as FP32
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *        -o mv_bench_bf16_8row mv_bench_bf16_8row.c -lm
 * Run:   OMP_NUM_THREADS=1 numactl -C 12 -m 4 ./mv_bench_bf16_8row [N] [K]
 *
 * N here = how many 8-row groups to run; total flops = N * 8 * 2K.
 * Default: N=31040 K=1024 → matches lm_head (8 * 31040 = 248320 rows).
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

#define SVE_BF16_TO_F32(pg, ptr) \
    svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, ptr), 16))

/* row8_lsl: copy of ggml_dequant matvec_bf16_8row */
static void row8_lsl(float *dst,
                     const bf16_t *w0, const bf16_t *w1,
                     const bf16_t *w2, const bf16_t *w3,
                     const bf16_t *w4, const bf16_t *w5,
                     const bf16_t *w6, const bf16_t *w7,
                     const float *x, int K) {
    svfloat32_t a0 = svdup_f32(0), a1 = svdup_f32(0);
    svfloat32_t a2 = svdup_f32(0), a3 = svdup_f32(0);
    svfloat32_t a4 = svdup_f32(0), a5 = svdup_f32(0);
    svfloat32_t a6 = svdup_f32(0), a7 = svdup_f32(0);
    int vl = (int)svcntw();
    svbool_t pg = svptrue_b32();
    int i = 0;
    for (; i + vl - 1 < K; i += vl) {
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, SVE_BF16_TO_F32(pg, &w0[i]), vx);
        a1 = svmla_x(pg, a1, SVE_BF16_TO_F32(pg, &w1[i]), vx);
        a2 = svmla_x(pg, a2, SVE_BF16_TO_F32(pg, &w2[i]), vx);
        a3 = svmla_x(pg, a3, SVE_BF16_TO_F32(pg, &w3[i]), vx);
        a4 = svmla_x(pg, a4, SVE_BF16_TO_F32(pg, &w4[i]), vx);
        a5 = svmla_x(pg, a5, SVE_BF16_TO_F32(pg, &w5[i]), vx);
        a6 = svmla_x(pg, a6, SVE_BF16_TO_F32(pg, &w6[i]), vx);
        a7 = svmla_x(pg, a7, SVE_BF16_TO_F32(pg, &w7[i]), vx);
    }
    if (i < K) {
        svbool_t pt = svwhilelt_b32(i, K);
        svfloat32_t vx = svld1(pt, &x[i]);
        a0 = svmla_m(pt, a0, SVE_BF16_TO_F32(pt, &w0[i]), vx);
        a1 = svmla_m(pt, a1, SVE_BF16_TO_F32(pt, &w1[i]), vx);
        a2 = svmla_m(pt, a2, SVE_BF16_TO_F32(pt, &w2[i]), vx);
        a3 = svmla_m(pt, a3, SVE_BF16_TO_F32(pt, &w3[i]), vx);
        a4 = svmla_m(pt, a4, SVE_BF16_TO_F32(pt, &w4[i]), vx);
        a5 = svmla_m(pt, a5, SVE_BF16_TO_F32(pt, &w5[i]), vx);
        a6 = svmla_m(pt, a6, SVE_BF16_TO_F32(pt, &w6[i]), vx);
        a7 = svmla_m(pt, a7, SVE_BF16_TO_F32(pt, &w7[i]), vx);
    }
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* row8_pv: 4 pair buffers (rA,rB), p_odd predicated loads.
 * pAB,pCD,pEF,pGH each hold 2*K bf16 interleaved per-element.
 *
 * Pair buf indexing (16-lane chunk):
 *   chunk c covers k=[c*16 .. c*16+15]
 *   bytes [c*64 .. c*64+63]: 32 halfwords (rA[0],rB[0],rA[1],rB[1],...)
 */
static void row8_pv(float *dst,
                    const bf16_t *pAB, const bf16_t *pCD,
                    const bf16_t *pEF, const bf16_t *pGH,
                    const float *x, int K) {
    svbool_t pg = svptrue_b32();
    svbool_t p_all_h = svptrue_b16();
    svuint16_t idx_h = svindex_u16(0, 1);
    svbool_t p_odd = svcmpne_n_u16(p_all_h,
                                    svand_n_u16_x(p_all_h, idx_h, 1), 0);

    int vl = (int)svcntw();   /* 16 fp32 lanes */
    svfloat32_t a0 = svdup_f32(0), a1 = svdup_f32(0);
    svfloat32_t a2 = svdup_f32(0), a3 = svdup_f32(0);
    svfloat32_t a4 = svdup_f32(0), a5 = svdup_f32(0);
    svfloat32_t a6 = svdup_f32(0), a7 = svdup_f32(0);

    int i = 0;
    for (; i + vl - 1 < K; i += vl) {
        /* For chunk c = i / vl, pair buf at byte offset c * 64 */
        const uint16_t *ab = (const uint16_t *)pAB + 2 * i; /* HW = 2*i */
        const uint16_t *cd = (const uint16_t *)pCD + 2 * i;
        const uint16_t *ef = (const uint16_t *)pEF + 2 * i;
        const uint16_t *gh = (const uint16_t *)pGH + 2 * i;
        /* even-position stream (HW 0,2,...): rA at -1 halfword offset
         * odd-position stream (HW 1,3,...): rB at 0 offset */
        svuint16_t vA = svld1_u16(p_odd, ab - 1);
        svuint16_t vB = svld1_u16(p_odd, ab);
        svuint16_t vC = svld1_u16(p_odd, cd - 1);
        svuint16_t vD = svld1_u16(p_odd, cd);
        svuint16_t vE = svld1_u16(p_odd, ef - 1);
        svuint16_t vF = svld1_u16(p_odd, ef);
        svuint16_t vG = svld1_u16(p_odd, gh - 1);
        svuint16_t vH = svld1_u16(p_odd, gh);
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, svreinterpret_f32(vA), vx);
        a1 = svmla_x(pg, a1, svreinterpret_f32(vB), vx);
        a2 = svmla_x(pg, a2, svreinterpret_f32(vC), vx);
        a3 = svmla_x(pg, a3, svreinterpret_f32(vD), vx);
        a4 = svmla_x(pg, a4, svreinterpret_f32(vE), vx);
        a5 = svmla_x(pg, a5, svreinterpret_f32(vF), vx);
        a6 = svmla_x(pg, a6, svreinterpret_f32(vG), vx);
        a7 = svmla_x(pg, a7, svreinterpret_f32(vH), vx);
    }
    /* tail (K not multiple of vl): unlikely for our K values; skip */
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* Driver: N groups of 8 rows. Each group computes one matvec_bf16_8row. */
static void drive_lsl(float *Y, const bf16_t *W, const float *x,
                      int N_groups, int K) {
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < N_groups; g++) {
        const bf16_t *base = W + (size_t)g * 8 * K;
        row8_lsl(&Y[g * 8],
                 base + 0 * K, base + 1 * K, base + 2 * K, base + 3 * K,
                 base + 4 * K, base + 5 * K, base + 6 * K, base + 7 * K,
                 x, K);
    }
}

static void drive_pv(float *Y, const bf16_t *Wpv, const float *x,
                     int N_groups, int K) {
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < N_groups; g++) {
        const bf16_t *base = Wpv + (size_t)g * 8 * K;
        row8_pv(&Y[g * 8],
                base + 0 * 2 * K, base + 1 * 2 * K,
                base + 2 * 2 * K, base + 3 * 2 * K,
                x, K);
    }
}

int main(int argc, char **argv) {
    int N_groups = 31040;   /* 8*31040 = 248320 rows (Qwen3.5 lm_head) */
    int K = 1024;
    if (argc > 1) N_groups = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    int vl = (int)svcntw();
    /* round K up so kernels don't need tail */
    if (K % vl) { fprintf(stderr, "K must be multiple of %d\n", vl); return 1; }

    int N_rows = N_groups * 8;
    size_t welem = (size_t)N_rows * K;
    size_t wbytes = welem * sizeof(bf16_t);
    printf("N_groups=%d (8x rows=%d) K=%d  VL(fp32)=%d  W=%.1f MB\n",
           N_groups, N_rows, K, vl, wbytes / 1048576.0);

    bf16_t *W_lsl = (bf16_t *)walloc(wbytes);   /* group-major, 8 rows packed */
    bf16_t *W_pv  = (bf16_t *)walloc(wbytes);   /* group-major, 4 pairs interleaved */
    float  *x     = (float *)walloc((size_t)K * sizeof(float));
    float  *Y_lsl = (float *)walloc((size_t)N_rows * sizeof(float));
    float  *Y_pv  = (float *)walloc((size_t)N_rows * sizeof(float));
    if (!W_lsl || !W_pv || !x || !Y_lsl || !Y_pv) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    for (int k = 0; k < K; k++) x[k] = 0.5f + 0.001f * (float)k;

    /* Fill W_lsl: 8 separate rows per group */
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < N_groups; g++) {
        for (int r = 0; r < 8; r++) {
            for (int k = 0; k < K; k++) {
                int row = g * 8 + r;
                float v = ((row * 7919 + k * 7) % 1000) * 0.001f - 0.5f;
                W_lsl[(size_t)g * 8 * K + (size_t)r * K + k] = f32_to_bf16(v);
            }
        }
    }

    /* Fill W_pv: 4 pairs per group, each pair = 2*K bf16 with
     * even HWs = pair.A, odd HWs = pair.B (per 16-lane chunk).
     * Per pair p, the two rows are 2p and 2p+1.
     * pair_buf[chunk * 64bytes + 2*lane]   = rA[chunk*16 + lane]   (even HW)
     * pair_buf[chunk * 64bytes + 2*lane+1] = rB[chunk*16 + lane]   (odd HW)
     * Indexing as bf16_t array (each elem 2 bytes):
     *   pair_buf[chunk*32 + 2*lane + 0] = rA
     *   pair_buf[chunk*32 + 2*lane + 1] = rB
     */
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < N_groups; g++) {
        for (int p = 0; p < 4; p++) {
            int rowA = g * 8 + 2 * p;
            int rowB = g * 8 + 2 * p + 1;
            bf16_t *pair = W_pv + (size_t)g * 8 * K + (size_t)p * 2 * K;
            for (int k = 0; k < K; k++) {
                int chunk = k / vl;
                int lane = k % vl;
                float vA = ((rowA * 7919 + k * 7) % 1000) * 0.001f - 0.5f;
                float vB = ((rowB * 7919 + k * 7) % 1000) * 0.001f - 0.5f;
                /* HW position: chunk*32 + 2*lane + {0,1} */
                pair[(size_t)chunk * 32 + 2 * lane + 0] = f32_to_bf16(vA);
                pair[(size_t)chunk * 32 + 2 * lane + 1] = f32_to_bf16(vB);
            }
        }
    }

    /* Reference: run row8_lsl, then verify row8_pv matches */
    drive_lsl(Y_lsl, W_lsl, x, N_groups, K);
    drive_pv(Y_pv, W_pv, x, N_groups, K);

    double max_err = 0, ref_max = 0;
    for (int i = 0; i < N_rows; i++) {
        double e = fabs((double)Y_pv[i] - (double)Y_lsl[i]);
        if (e > max_err) max_err = e;
        double r = fabs((double)Y_lsl[i]);
        if (r > ref_max) ref_max = r;
    }
    printf("row8_pv vs row8_lsl: max_abs_err=%.4e  rel=%.2e\n",
           max_err, max_err / (ref_max + 1e-30));

    /* Bench */
    const int reps = 7;
    double bw_bytes = (double)wbytes + (double)K * sizeof(float) * N_groups;
    double flops = 2.0 * (double)welem;

    /* warmup */
    drive_lsl(Y_lsl, W_lsl, x, N_groups, K);
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = mono_sec();
        drive_lsl(Y_lsl, W_lsl, x, N_groups, K);
        double dt = mono_sec() - t0;
        if (dt < best) best = dt;
    }
    printf("  row8_lsl  %.3f ms  %6.2f GB/s  %6.2f GFLOP/s\n",
           best * 1e3, bw_bytes / best / 1e9, flops / best / 1e9);

    drive_pv(Y_pv, W_pv, x, N_groups, K);
    best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = mono_sec();
        drive_pv(Y_pv, W_pv, x, N_groups, K);
        double dt = mono_sec() - t0;
        if (dt < best) best = dt;
    }
    printf("  row8_pv   %.3f ms  %6.2f GB/s  %6.2f GFLOP/s\n",
           best * 1e3, bw_bytes / best / 1e9, flops / best / 1e9);

    return 0;
}
