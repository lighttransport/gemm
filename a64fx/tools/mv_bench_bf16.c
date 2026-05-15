/* mv_bench_bf16.c - single-core BF16-weight matvec microbench for A64FX.
 *
 * Goal: optimize Y = W*x where W is [N][K] bf16 and x is [K] fp32,
 *       y is [N] fp32. FP32 accumulators throughout (A64FX has no native
 *       BF16 FMA, only widening to fp32).
 *
 * Two panel-layout kernels:
 *   panel_lsl  : Wp[blk][k][lane] uint16, 1 LD1H{Z.S} + LSL #16 per k-step
 *                (== current ggml_dequant SVE_BF16_TO_F32 macro)
 *   panel_pv   : Wp[blk][k:k+1][lane] uint16 INTERLEAVED, 2 LD1H{Z.H}
 *                with p_odd predicate at offsets -1/0 halfwords cover 2 k-steps,
 *                NO LSL — bf16 lands directly as fp32 in upper half of .s lanes
 *
 * Per k-step instruction budget on A64FX (matvec, scalar x):
 *   lsl variant : LD1H (mem) + LSL (FLA) + FMLA (FLA/FLB) + LD1RW (mem) = 4 ops
 *                 FLA usage = 2 (LSL+FMLA contend on same pipe)
 *   pv  variant : 1 LD1H per 2 k-steps (amortized) + FMLA + LD1RW = ~3 ops
 *                 FLA usage = 1 (only FMLA on FLA; FLB free)
 *
 * If matvec is FLA-bound, removing LSL halves FLA pressure.
 * If matvec is HBM-bound, the trick won't help. Either way, this isolates.
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *       -o mv_bench_bf16 mv_bench_bf16.c -lm
 * Run: OMP_NUM_THREADS=1 numactl -C 12 -m 4 ./mv_bench_bf16
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
    if (posix_memalign(&p, 64, (bytes + 63) & ~(size_t)63) != 0) return NULL;
    memset(p, 0, bytes);
    return p;
}

static bf16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    bits += ((bits >> 16) & 1) + 0x7FFF;
    return (bf16_t)(bits >> 16);
}

/* ──────────────────────────────────────────────────────────────────
 * Layout: Wp[blk=N/16][k][lane=0..15] bf16 row-major
 * Each FP32 SVE lane (16 on A64FX) accumulates a different output row.
 * One block = 16 rows = 16 BF16 per k-step.
 * ────────────────────────────────────────────────────────────────── */

/* panel_lsl: load 16 BF16 as zero-extended .s, shift left 16 → fp32 */
static void panel_lsl(float *dst, const bf16_t *wp, const float *x,
                       int N, int K) {
    svbool_t pg = svptrue_b32();
    int vls = (int)svcntw();          /* 16 fp32 lanes */
    for (int blk = 0; blk + vls - 1 < N; blk += vls) {
        const bf16_t *w = wp + (size_t)blk * K;
        svfloat32_t a0 = svdup_f32(0), a1 = svdup_f32(0);
        svfloat32_t a2 = svdup_f32(0), a3 = svdup_f32(0);
        int k = 0;
        for (; k + 3 < K; k += 4) {
            svfloat32_t vw0 = svreinterpret_f32(svlsl_x(pg,
                svld1uh_u32(pg, w + (size_t)(k + 0) * vls), 16));
            svfloat32_t vw1 = svreinterpret_f32(svlsl_x(pg,
                svld1uh_u32(pg, w + (size_t)(k + 1) * vls), 16));
            svfloat32_t vw2 = svreinterpret_f32(svlsl_x(pg,
                svld1uh_u32(pg, w + (size_t)(k + 2) * vls), 16));
            svfloat32_t vw3 = svreinterpret_f32(svlsl_x(pg,
                svld1uh_u32(pg, w + (size_t)(k + 3) * vls), 16));
            a0 = svmla_x(pg, a0, vw0, svdup_f32(x[k + 0]));
            a1 = svmla_x(pg, a1, vw1, svdup_f32(x[k + 1]));
            a2 = svmla_x(pg, a2, vw2, svdup_f32(x[k + 2]));
            a3 = svmla_x(pg, a3, vw3, svdup_f32(x[k + 3]));
        }
        for (; k < K; k++) {
            svfloat32_t vw = svreinterpret_f32(svlsl_x(pg,
                svld1uh_u32(pg, w + (size_t)k * vls), 16));
            a0 = svmla_x(pg, a0, vw, svdup_f32(x[k]));
        }
        svst1_f32(pg, &dst[blk],
                  svadd_x(pg, svadd_x(pg, a0, a1), svadd_x(pg, a2, a3)));
    }
}

/* panel_pv: 2-k-step interleaved layout.
 *   Wp2[blk][k_pair][lane*2 + parity]  where parity=0 is even k, 1 is odd k
 *   For each 16-row block, 2 k-steps occupy 32 BF16 = 64 bytes.
 *   ld1h.h with p_odd at offset -2 bytes (= -1 halfword) extracts the
 *   even-parity stream as 16 FP32; at offset 0 extracts odd-parity stream.
 *   No LSL.
 * Build the p_odd predicate once outside the loop.
 */
static void panel_pv(float *dst, const bf16_t *wp2, const float *x,
                     int N, int K) {
    svbool_t pg = svptrue_b32();
    /* p_odd: active at odd .h lane positions 1,3,5,...,31 */
    svbool_t p_all_h = svptrue_b16();
    svuint16_t idx_h = svindex_u16(0, 1);
    svbool_t p_odd_h = svcmpne_n_u16(p_all_h,
                                      svand_n_u16_x(p_all_h, idx_h, 1), 0);

    int vls = (int)svcntw();  /* 16 fp32 lanes */
    for (int blk = 0; blk + vls - 1 < N; blk += vls) {
        const bf16_t *w = wp2 + (size_t)blk * K;
        svfloat32_t a0 = svdup_f32(0), a1 = svdup_f32(0);
        svfloat32_t a2 = svdup_f32(0), a3 = svdup_f32(0);
        int k = 0;
        /* Process k-pairs: 4 pairs per outer iter → 8 k-steps per outer iter */
        for (; k + 7 < K; k += 8) {
            const bf16_t *wk = w + (size_t)k * vls;
            /* Pair 0: k, k+1 */
            svuint16_t v0e = svld1_u16(p_odd_h, (const uint16_t *)(wk + 0) - 1);
            svuint16_t v0o = svld1_u16(p_odd_h, (const uint16_t *)(wk + 0));
            /* Pair 1: k+2, k+3 */
            svuint16_t v1e = svld1_u16(p_odd_h, (const uint16_t *)(wk + 2 * vls) - 1);
            svuint16_t v1o = svld1_u16(p_odd_h, (const uint16_t *)(wk + 2 * vls));
            /* Pair 2: k+4, k+5 */
            svuint16_t v2e = svld1_u16(p_odd_h, (const uint16_t *)(wk + 4 * vls) - 1);
            svuint16_t v2o = svld1_u16(p_odd_h, (const uint16_t *)(wk + 4 * vls));
            /* Pair 3: k+6, k+7 */
            svuint16_t v3e = svld1_u16(p_odd_h, (const uint16_t *)(wk + 6 * vls) - 1);
            svuint16_t v3o = svld1_u16(p_odd_h, (const uint16_t *)(wk + 6 * vls));

            a0 = svmla_x(pg, a0, svreinterpret_f32(v0e), svdup_f32(x[k + 0]));
            a1 = svmla_x(pg, a1, svreinterpret_f32(v0o), svdup_f32(x[k + 1]));
            a2 = svmla_x(pg, a2, svreinterpret_f32(v1e), svdup_f32(x[k + 2]));
            a3 = svmla_x(pg, a3, svreinterpret_f32(v1o), svdup_f32(x[k + 3]));
            a0 = svmla_x(pg, a0, svreinterpret_f32(v2e), svdup_f32(x[k + 4]));
            a1 = svmla_x(pg, a1, svreinterpret_f32(v2o), svdup_f32(x[k + 5]));
            a2 = svmla_x(pg, a2, svreinterpret_f32(v3e), svdup_f32(x[k + 6]));
            a3 = svmla_x(pg, a3, svreinterpret_f32(v3o), svdup_f32(x[k + 7]));
        }
        for (; k + 1 < K; k += 2) {
            const bf16_t *wk = w + (size_t)k * vls;
            svuint16_t ve = svld1_u16(p_odd_h, (const uint16_t *)wk - 1);
            svuint16_t vo = svld1_u16(p_odd_h, (const uint16_t *)wk);
            a0 = svmla_x(pg, a0, svreinterpret_f32(ve), svdup_f32(x[k + 0]));
            a1 = svmla_x(pg, a1, svreinterpret_f32(vo), svdup_f32(x[k + 1]));
        }
        /* Tail single k step: fall back to lsl variant for the lone column */
        if (k < K) {
            const bf16_t *wk = w + (size_t)k * vls;
            svfloat32_t vw = svreinterpret_f32(svlsl_x(pg,
                svld1uh_u32(pg, wk), 16));
            a0 = svmla_x(pg, a0, vw, svdup_f32(x[k]));
        }
        svst1_f32(pg, &dst[blk],
                  svadd_x(pg, svadd_x(pg, a0, a1), svadd_x(pg, a2, a3)));
    }
}

/* Pure HBM bandwidth probe: sum the whole weight stream */
static float bw_probe(const bf16_t *wp, size_t nelem) {
    svbool_t pg = svptrue_b32();
    int vls = (int)svcntw();
    svfloat32_t a0=svdup_f32(0), a1=svdup_f32(0), a2=svdup_f32(0), a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0), a5=svdup_f32(0), a6=svdup_f32(0), a7=svdup_f32(0);
    size_t i = 0;
    for (; i + 8 * (size_t)vls <= nelem; i += 8 * (size_t)vls) {
        a0 = svadd_x(pg, a0, svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, wp + i + 0 * vls), 16)));
        a1 = svadd_x(pg, a1, svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, wp + i + 1 * vls), 16)));
        a2 = svadd_x(pg, a2, svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, wp + i + 2 * vls), 16)));
        a3 = svadd_x(pg, a3, svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, wp + i + 3 * vls), 16)));
        a4 = svadd_x(pg, a4, svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, wp + i + 4 * vls), 16)));
        a5 = svadd_x(pg, a5, svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, wp + i + 5 * vls), 16)));
        a6 = svadd_x(pg, a6, svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, wp + i + 6 * vls), 16)));
        a7 = svadd_x(pg, a7, svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, wp + i + 7 * vls), 16)));
    }
    svfloat32_t s = svadd_x(pg, svadd_x(pg, svadd_x(pg,a0,a1), svadd_x(pg,a2,a3)),
                                svadd_x(pg, svadd_x(pg,a4,a5), svadd_x(pg,a6,a7)));
    return svaddv(pg, s);
}

typedef void (*kfn)(float *, const bf16_t *, const float *, int, int);

int main(int argc, char **argv) {
    /* Default shape: Qwen3.5-0.8B lm_head (the 95ms/call hotspot in profile) */
    int N = 248320, K = 1024;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    int vls = (int)svcntw();
    N = (N / vls) * vls;

    size_t welem = (size_t)N * K;
    size_t wbytes = welem * sizeof(bf16_t);
    printf("N=%d K=%d  VL(fp32)=%d  W=%.1f MB\n", N, K, vls, wbytes / 1048576.0);

    bf16_t *wp_lsl = (bf16_t *)walloc(wbytes);
    bf16_t *wp_pv  = (bf16_t *)walloc(wbytes);
    float  *x      = (float  *)walloc((size_t)K * sizeof(float));
    float  *dst    = (float  *)walloc((size_t)N * sizeof(float));
    float  *dst_ref = (float *)walloc((size_t)N * sizeof(float));
    if (!wp_lsl || !wp_pv || !x || !dst || !dst_ref) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    for (int k = 0; k < K; k++) x[k] = 0.5f + 0.001f * (float)k;

    /* Fill wp_lsl in panel layout: Wp[blk][k][lane] = w[blk*vls+lane][k] */
    #pragma omp parallel for schedule(static)
    for (int blk = 0; blk < N / vls; blk++) {
        for (int k = 0; k < K; k++) {
            for (int lane = 0; lane < vls; lane++) {
                int row = blk * vls + lane;
                float v = ((row * 7919 + k * 7) % 1000) * 0.001f - 0.5f;
                wp_lsl[((size_t)blk * K + k) * vls + lane] = f32_to_bf16(v);
            }
        }
    }
    /* Fill wp_pv in 2-k-interleaved layout:
     *   for each block, pair k-steps (k, k+1).
     *   The byte layout per k-pair:
     *     pos 0 (byte 0..1): w[blk_row 0][k+1]
     *     pos 1 (byte 2..3): w[blk_row 0][k+0]
     *     pos 2 (byte 4..5): w[blk_row 1][k+1]
     *     pos 3 (byte 6..7): w[blk_row 1][k+0]
     *     ...
     *   So even .h positions hold k+1, odd .h positions hold k+0.
     *   ld1h.h with p_odd, offset 0 → reads odd positions → k+0 stream
     *   ld1h.h with p_odd, offset -2 → shifts source by -2 bytes; the
     *     active odd positions then sample memory positions 1+2j shifted by -1
     *     → byte = -2 + 2(2j+1) = 4j → even positions = k+1 stream
     * Wait — verify: ld1h.h with p_odd at lane j (j odd) reads memory[base + j*2].
     *   base=base, j=1: byte 2 (pos 1) = k+0
     *   base=base-2, j=1: byte 0 (pos 0) = k+1
     */
    #pragma omp parallel for schedule(static)
    for (int blk = 0; blk < N / vls; blk++) {
        for (int kp = 0; kp + 1 < K; kp += 2) {
            for (int lane = 0; lane < vls; lane++) {
                int row = blk * vls + lane;
                float v0 = ((row * 7919 + (kp + 0) * 7) % 1000) * 0.001f - 0.5f;
                float v1 = ((row * 7919 + (kp + 1) * 7) % 1000) * 0.001f - 0.5f;
                /* Interleaved: [k+1, k+0] per lane, lane-major */
                size_t base_h = (size_t)blk * K * vls + (size_t)kp * vls
                               + (size_t)lane * 2;
                wp_pv[base_h + 0] = f32_to_bf16(v1); /* even pos = k+1 */
                wp_pv[base_h + 1] = f32_to_bf16(v0); /* odd  pos = k+0 */
            }
        }
        /* If K odd, leave last column zero in interleaved buffer; kernel
         * tail handles via lsl path against the regular wp_lsl panel. */
    }

    /* Correctness: reference = panel_lsl */
    panel_lsl(dst_ref, wp_lsl, x, N, K);

    /* Verify panel_pv matches reference */
    memset(dst, 0, (size_t)N * sizeof(float));
    panel_pv(dst, wp_pv, x, N, K);
    double max_err = 0, ref_max = 0;
    for (int i = 0; i < N; i++) {
        double e = fabs((double)dst[i] - (double)dst_ref[i]);
        if (e > max_err) max_err = e;
        double r = fabs((double)dst_ref[i]);
        if (r > ref_max) ref_max = r;
    }
    printf("panel_pv vs panel_lsl: max_abs_err=%.4e  rel=%.2e\n",
           max_err, max_err / (ref_max + 1e-30));

    /* Benchmark each */
    struct { const char *name; kfn fn; const bf16_t *w; } ks[] = {
        {"panel_lsl (LD1H{.s}+LSL)", panel_lsl, wp_lsl},
        {"panel_pv  (p_odd predicated load)", panel_pv, wp_pv},
    };
    int nk = sizeof(ks) / sizeof(ks[0]);

    const int reps = 7;
    for (int t = 0; t < nk; t++) {
        ks[t].fn(dst, ks[t].w, x, N, K); /* warmup */
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = mono_sec();
            ks[t].fn(dst, ks[t].w, x, N, K);
            double dt = mono_sec() - t0;
            if (dt < best) best = dt;
        }
        double gbs = wbytes / best / 1e9;
        double gflops = 2.0 * welem / best / 1e9;
        printf("  %-38s  %.3f ms  %6.2f GB/s  %6.2f GFLOP/s\n",
               ks[t].name, best * 1e3, gbs, gflops);
    }

    /* BW ceiling */
    {
        volatile float sink = bw_probe(wp_lsl, welem);
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = mono_sec();
            sink = bw_probe(wp_lsl, welem);
            double dt = mono_sec() - t0;
            if (dt < best) best = dt;
        }
        (void)sink;
        printf("  %-38s  %.3f ms  %6.2f GB/s  (load+add ceiling)\n",
               "bw_probe", best * 1e3, wbytes / best / 1e9);
    }

    free(wp_lsl); free(wp_pv); free(x); free(dst); free(dst_ref);
    return 0;
}
