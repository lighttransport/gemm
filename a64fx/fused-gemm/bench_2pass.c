/*
 * 2-Pass Fused Softmax + P@V Benchmark for A64FX  (v5)
 *
 * Pass 1: exp2(S - max) -> P, compute row_max, row_sum
 * Pass 2: O = P @ V (pre-packed V, pure GEMM)
 *
 * Fixes vs v4:
 *  - V pre-packed once before timed loop (eliminates pack overhead from pass2)
 *  - All allocations 256-byte aligned (A64FX cache line)
 *  - P written col-major directly in pass1 (eliminates transpose)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <arm_sve.h>

/* A64FX: 2 GHz, 2 FP32 FMA pipes × 16 lanes = 64 FMA/cy = 128 GFLOPS */
#define CPU_GHZ    2.0
#define PEAK_GFLOPS (64.0 * CPU_GHZ)  /* 128 fp32 */
#define PEAK_GFLOPS_FP16 (128.0 * CPU_GHZ)  /* 256: 2 pipes × 32 fp16 × 2 */
#define KC_DEFAULT 256
#define L1_SETS    64
#define L1_LINE    256
static int l1_set_overlap(int width, int shift);  /* forward decl */

static volatile double tick2sec;  /* 1.0 / tick_freq() - volatile to prevent D-reg caching */

static inline uint64_t rdtick(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t tick_freq(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}

/* ─── Microkernel interfaces ─── */
typedef void (*gemm_fn)(const float*, const float*, float*,
                        int64_t, int64_t, int64_t);

extern void micro_kernel_fp32_8x3(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_8x3_accum(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_6x4_bcast(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_6x4_bcast_accum(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_10x2(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_10x2_accum(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_11x2(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_11x2_accum(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_12x2(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_12x2_accum(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_12x2_swp(
    const float*, const float*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp32_12x2_swp_accum(
    const float*, const float*, float*, int64_t, int64_t, int64_t);

/* ─── FP16 mixed-precision kernel ─── */
typedef void (*gemm_fp16_fn)(const _Float16*, const _Float16*, float*,
                              int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_swp(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_swp_accum(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_csplit(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_csplit_accum(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_dswp(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_dswp_accum(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_4k(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_4k_accum(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_8x3(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_8x3_accum(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_6x4(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_6x4_accum(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_bpre(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_bpre_accum(
    const _Float16*, const _Float16*, float*, int64_t, int64_t, int64_t);

/* ─── FP16 NOEPI kernels (fp16 C accumulation, no fp32 conversion) ─── */
typedef void (*gemm_fp16_noepi_fn)(const _Float16*, const _Float16*, _Float16*,
                                    int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi_accum(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi4(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi4_accum(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi4_prfm(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);
extern void micro_kernel_fp16_12x2_noepi4_prfm_accum(
    const _Float16*, const _Float16*, _Float16*, int64_t, int64_t, int64_t);

/* ─── FEXPA exp2 ─── */
#define FEXPA_SHIFT_U32  0x48481fc0u
static inline svfloat32_t sve_fexpa(svfloat32_t z) {
    svfloat32_t r;
    __asm__("fexpa %0.s, %1.s" : "=w"(r) : "w"(z));
    return r;
}

/* Scalar exp2 via FEXPA for testing */
static inline float fexpa_exp2_scalar(float x) {
    union { uint32_t u; float f; } su = { .u = FEXPA_SHIFT_U32 };
    float shifted = x + su.f;
    svfloat32_t v = svdup_f32(shifted);
    v = sve_fexpa(v);
    return svlastb(svptrue_b32(), v);
}

/* ─── FEXPA accuracy test ─── */
static int test_fexpa(void) {
    printf("FEXPA exp2 accuracy test:\n");
    float test_vals[] = {0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -0.5f, -0.25f,
                         1.0f, 2.0f, -8.0f, -16.0f, -0.01f};
    int n = sizeof(test_vals)/sizeof(test_vals[0]);
    float max_relerr = 0;
    int ok = 1;
    for (int i = 0; i < n; i++) {
        float x = test_vals[i];
        float got = fexpa_exp2_scalar(x);
        float ref = exp2f(x);
        float relerr = (ref != 0) ? fabsf(got - ref) / ref : fabsf(got);
        if (relerr > max_relerr) max_relerr = relerr;
        if (relerr > 0.02f) { /* 2% threshold for ~7-bit */
            printf("  FAIL: exp2(%8.4f) = %.6e  ref = %.6e  relerr = %.4f\n",
                   x, got, ref, relerr);
            ok = 0;
        }
    }
    printf("  max relative error = %.4e  %s\n\n", max_relerr, ok ? "OK" : "FAIL");
    return ok;
}

/* ─── Microkernel correctness test ─── */
static void test_kernel(const char *name, gemm_fn kern, gemm_fn kern_acc,
                        int MR, int NR) {
    int K = 64;
    float *A  = aligned_alloc(256, K * MR * 4);
    float *B  = aligned_alloc(256, K * NR * 4);
    float *C  = aligned_alloc(256, MR * NR * 4);
    size_t Cr_sz = (size_t)MR * NR * 8;
    double *Cr = aligned_alloc(256, (Cr_sz + 255) & ~(size_t)255);
    memset(Cr, 0, Cr_sz);

    srand(12345);
    for (int i = 0; i < K * MR; i++) A[i] = (float)rand()/(float)RAND_MAX - 0.5f;
    for (int i = 0; i < K * NR; i++) B[i] = (float)rand()/(float)RAND_MAX - 0.5f;

    /* Reference: C[m][n] = sum_k A[k*MR+m] * B[k*NR+n] */
    for (int m = 0; m < MR; m++)
        for (int n = 0; n < NR; n++) {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += (double)A[k*MR+m] * (double)B[k*NR+n];
            Cr[m*NR+n] = s;
        }

    /* Test zero-init kernel */
    memset(C, 0, MR * NR * 4);
    int64_t ldc_bytes = (int64_t)NR * 4;
    kern(A, B, C, (int64_t)K, 0, ldc_bytes);

    double me = 0;
    for (int m = 0; m < MR; m++)
        for (int n = 0; n < NR; n++) {
            double e = fabs((double)C[m*NR+n] - Cr[m*NR+n]);
            if (e > me) me = e;
        }
    printf("Kernel %-8s init:  maxerr = %.2e  %s\n", name, me,
           me < 1e-3 ? "OK" : "FAIL");

    /* Test accumulate kernel: do K=32 init + K=32 accum */
    if (kern_acc) {
        memset(C, 0, MR * NR * 4);
        kern(A, B, C, 32, 0, ldc_bytes);
        kern_acc(A + 32*MR, B + 32*NR, C, 32, 0, ldc_bytes);

        me = 0;
        for (int m = 0; m < MR; m++)
            for (int n = 0; n < NR; n++) {
                double e = fabs((double)C[m*NR+n] - Cr[m*NR+n]);
                if (e > me) me = e;
            }
        printf("Kernel %-8s accum: maxerr = %.2e  %s\n", name, me,
               me < 1e-3 ? "OK" : "FAIL");
    }

    free(A); free(B); free(C); free(Cr);
}

/* ─── Pass 1: softmax + write P in col-major Pp[L][MR] ─── */
/* S is column-major: S[j * ld_s + m], ld_s = M_max.
 * Process all MR rows per column → contiguous loads and stores, no scatter. */
static void pass1_softmax(
    const float *S, int ld_s,
    float *Pp,      /* output: col-major [L][MR], stride MR */
    int MR, int L,
    float *row_max, float *row_sum)
{
    union { uint32_t u; float f; } su = { .u = FEXPA_SHIFT_U32 };
    float shift = su.f;
    svbool_t pg = svwhilelt_b32(0, MR);

    /* Max-find: all MR rows simultaneously */
    svfloat32_t vmax = svdup_f32(-1e30f);
    for (int j = 0; j < L; j++)
        vmax = svmax_m(pg, vmax, svld1(pg, S + (int64_t)j * ld_s));
    float mbuf[16] __attribute__((aligned(256)));
    svst1(svptrue_b32(), mbuf, vmax);
    for (int m = 0; m < MR; m++) row_max[m] = mbuf[m];

    /* Build per-row bias vector: -max[m] + shift */
    float bbuf[16] __attribute__((aligned(256)));
    for (int m = 0; m < MR; m++) bbuf[m] = -row_max[m] + shift;
    svfloat32_t vbias = svld1(pg, bbuf);

    /* Exp2 + contiguous store to Pp + accumulate sums */
    svfloat32_t vsum = svdup_f32(0.0f);
    for (int j = 0; j < L; j++) {
        svfloat32_t s = svld1(pg, S + (int64_t)j * ld_s);
        svfloat32_t p = sve_fexpa(svadd_m(pg, s, vbias));
        vsum = svadd_m(pg, vsum, p);
        svst1(pg, Pp + j * MR, p);
    }
    float sbuf[16] __attribute__((aligned(256)));
    svst1(svptrue_b32(), sbuf, vsum);
    for (int m = 0; m < MR; m++) row_sum[m] = sbuf[m];
}

/* ─── Pass1 block: exp2 → contiguous store to col-major Pp ─── */
/*
 * Column-major S: S[j * ld_s + m].  All MR rows for column j are contiguous.
 * Process all rows simultaneously per column: 1 svld1 + 1 fadd + 1 fexpa +
 * 1 svadd + 1 svst1.  No scatter!  ~5 SVE ops per column vs ~20 with scatter.
 */
static void pass1_block_exp(
    const float *S, int ld_s, int kc, int klen,
    int MR, const float *rmax,
    float *Pp,    /* output: Pp[klen][MR] col-major */
    float *rsum)  /* accumulate partial sums */
{
    union { uint32_t u; float f; } su = { .u = FEXPA_SHIFT_U32 };
    float shift = su.f;
    svbool_t pg = svwhilelt_b32(0, MR);

    /* Build per-row bias vector */
    float bbuf[16] __attribute__((aligned(256)));
    for (int m = 0; m < MR; m++) bbuf[m] = -rmax[m] + shift;
    svfloat32_t vbias = svld1(pg, bbuf);

    /* 4× unrolled with independent sum accumulators to break dep chains */
    svfloat32_t vs0 = svdup_f32(0.0f), vs1 = svdup_f32(0.0f);
    svfloat32_t vs2 = svdup_f32(0.0f), vs3 = svdup_f32(0.0f);
    const float *S_blk = S + (int64_t)kc * ld_s;
    int j;

    for (j = 0; j + 4 <= klen; j += 4) {
        svfloat32_t s0 = svld1(pg, S_blk + (j+0) * ld_s);
        svfloat32_t s1 = svld1(pg, S_blk + (j+1) * ld_s);
        svfloat32_t s2 = svld1(pg, S_blk + (j+2) * ld_s);
        svfloat32_t s3 = svld1(pg, S_blk + (j+3) * ld_s);
        svfloat32_t p0 = sve_fexpa(svadd_m(pg, s0, vbias));
        svfloat32_t p1 = sve_fexpa(svadd_m(pg, s1, vbias));
        svfloat32_t p2 = sve_fexpa(svadd_m(pg, s2, vbias));
        svfloat32_t p3 = sve_fexpa(svadd_m(pg, s3, vbias));
        vs0 = svadd_m(pg, vs0, p0);
        vs1 = svadd_m(pg, vs1, p1);
        vs2 = svadd_m(pg, vs2, p2);
        vs3 = svadd_m(pg, vs3, p3);
        svst1(pg, Pp + (j+0) * MR, p0);
        svst1(pg, Pp + (j+1) * MR, p1);
        svst1(pg, Pp + (j+2) * MR, p2);
        svst1(pg, Pp + (j+3) * MR, p3);
    }
    for (; j < klen; j++) {
        svfloat32_t s = svld1(pg, S_blk + j * ld_s);
        svfloat32_t p = sve_fexpa(svadd_m(pg, s, vbias));
        vs0 = svadd_m(pg, vs0, p);
        svst1(pg, Pp + j * MR, p);
    }
    svfloat32_t vsum = svadd_m(pg, svadd_m(pg, vs0, vs1),
                                    svadd_m(pg, vs2, vs3));

    /* Extract per-row sums */
    float sbuf[16] __attribute__((aligned(256)));
    svst1(svptrue_b32(), sbuf, vsum);
    for (int m = 0; m < MR; m++) rsum[m] += sbuf[m];
}

/* ─── Pack V block: V[rows][d_full] → Vp[rows][NR], col tile at col0 ─── */
/* SVE-vectorized: NR=48 uses 3 vectors, NR=64 uses 4 vectors */
static inline void pack_v_block(
    const float *V, int d,
    float *Vp, int rows, int col0, int NR)
{
    svbool_t pg = svptrue_b32();

    if (col0 + NR <= d) {
        /* Fast path: full tile, no zero-padding */
        if (NR == 32) {
            for (int k = 0; k < rows; k++) {
                const float *s = V + k * d + col0;
                float *p = Vp + k * 32;
                svst1(pg, p,    svld1(pg, s));
                svst1(pg, p+16, svld1(pg, s+16));
            }
        } else if (NR == 48) {
            for (int k = 0; k < rows; k++) {
                const float *s = V + k * d + col0;
                float *p = Vp + k * 48;
                svst1(pg, p,    svld1(pg, s));
                svst1(pg, p+16, svld1(pg, s+16));
                svst1(pg, p+32, svld1(pg, s+32));
            }
        } else { /* NR == 64 */
            for (int k = 0; k < rows; k++) {
                const float *s = V + k * d + col0;
                float *p = Vp + k * 64;
                svst1(pg, p,    svld1(pg, s));
                svst1(pg, p+16, svld1(pg, s+16));
                svst1(pg, p+32, svld1(pg, s+32));
                svst1(pg, p+48, svld1(pg, s+48));
            }
        }
    } else {
        /* Edge tile: partial copy + zero-pad */
        int ncopy = d - col0;
        for (int k = 0; k < rows; k++) {
            const float *vr = V + k * d + col0;
            float *vp = Vp + k * NR;
            int j;
            for (j = 0; j < ncopy; j++) vp[j] = vr[j];
            for (; j < NR; j++) vp[j] = 0.0f;
        }
    }
}

/* ─── Normalize O /= sum ─── */
static void normalize(float *O, int ld_o, const float *rs, int M, int d)
{
    const int vl = svcntw();
    for (int m = 0; m < M; m++) {
        svfloat32_t vi = svdup_f32(1.0f / rs[m]);
        float *row = O + m * ld_o;
        int j;
        for (j = 0; j + vl <= d; j += vl)
            svst1(svptrue_b32(), row + j,
                  svmul_m(svptrue_b32(), svld1(svptrue_b32(), row + j), vi));
        if (j < d) {
            svbool_t pg = svwhilelt_b32(j, d);
            svst1(pg, row + j, svmul_m(pg, svld1(pg, row + j), vi));
        }
    }
}

/* Forward declaration (defined after prepack_v_fp16) */
static inline void store_f32_as_f16(svbool_t pg, _Float16 *dst, svfloat32_t src);

/* ─── Normalize fp16 O /= sum (SVE vectorized via fp32 widening) ─── */
/* Uses ld1h{z.s} widening load + fcvt+st1h narrowing store (16 elems/iter).
 * Avoids fp16 intrinsic type ambiguity with _Float16* on Fujitsu/Clang. */
static void normalize_fp16(_Float16 *O, int ld_o, const float *rs, int M, int d)
{
    const int vl = svcntw();  /* 16 for fp32 on 512-bit SVE */
    svbool_t pg_all = svptrue_b32();
    for (int m = 0; m < M; m++) {
        svfloat32_t vinv = svdup_f32(1.0f / rs[m]);
        _Float16 *row = O + m * ld_o;
        int j;
        for (j = 0; j + vl <= d; j += vl) {
            svfloat32_t v;
            __asm__("ld1h {%0.s}, %1/z, [%2]"
                    : "=w"(v) : "Upl"(pg_all), "r"(row + j));
            v = svmul_m(pg_all, v, vinv);
            store_f32_as_f16(pg_all, row + j, v);
        }
        if (j < d) {
            svbool_t pgt = svwhilelt_b32(j, d);
            svfloat32_t v;
            __asm__("ld1h {%0.s}, %1/z, [%2]"
                    : "=w"(v) : "Upl"(pgt), "r"(row + j));
            v = svmul_m(pgt, v, vinv);
            store_f32_as_f16(pgt, row + j, v);
        }
    }
}

/* ─── Max error: fp16 O vs double reference ─── */
static double max_error_fp16(const _Float16 *O, int ld_o,
                              const double *Or, int M, int d)
{
    double me = 0;
    for (int m = 0; m < M; m++)
        for (int j = 0; j < d; j++) {
            double e = fabs((double)(float)O[m * ld_o + j] - Or[m * d + j]);
            if (e > me) me = e;
        }
    return me;
}

/* ─── GEMM-only benchmark (pre-packed V) to measure kernel ceiling ─── */
static double bench_gemm_only(
    const float *Pp, const float *Vp_all,
    float *O, int ld_o,
    int MR, int L, int d_out, int NR, int Kc,
    int n_tiles,
    gemm_fn kern_init, gemm_fn kern_accum)
{
    int64_t ldo_bytes = (int64_t)ld_o * sizeof(float);

    /* Warmup + measure GEMM only */
    int nrep = (L <= 2048) ? 21 : 7;
    int warmup = (L <= 2048) ? 5 : 2;
    double best_sec = 1e30;

    for (int r = 0; r < nrep; r++) {
        memset(O, 0, (size_t)MR * ld_o * 4);

        uint64_t t0 = rdtick();
        for (int kc = 0; kc < L; kc += Kc) {
            int klen = (kc + Kc <= L) ? Kc : (L - kc);
            klen &= ~1;
            if (klen < 2) break;
            int kb = kc / Kc;
            const float *Pp_blk = Pp + kc * MR;

            for (int t = 0; t < n_tiles; t++) {
                /* Re-touch A(Pp) into L1 when B tile > 40KB (evicts A) */
                if (t > 0 && klen * NR * (int)sizeof(float) > 40960) {
                    const char *a = (const char *)Pp_blk;
                    for (int ln = 0; ln < klen * MR * 4; ln += 256)
                        __builtin_prefetch(a + ln, 0, 3);
                }
                const float *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
                if (kc == 0)
                    kern_init(Pp_blk, Vp, O + t * NR,
                              (int64_t)klen, 0, ldo_bytes);
                else
                    kern_accum(Pp_blk, Vp, O + t * NR,
                               (int64_t)klen, 0, ldo_bytes);
            }
        }
        uint64_t t1 = rdtick();
        double sec = (double)(t1 - t0) * tick2sec;
        if (r >= warmup && sec < best_sec) best_sec = sec;
    }

    return best_sec;
}

/* ─── Timing structure (in seconds) ─── */
typedef struct {
    double pass1;   /* softmax + P write (col-major) */
    double pass2;   /* V pack + GEMM (combined) */
    double norm;    /* normalize O /= sum */
} timing_t;

/* ─── Pre-pack all V tiles for K-blocked GEMM ─── */
static float *prepack_v(const float *V, int d, int L, int d_out,
                        int NR, int Kc, int *out_n_kblocks, int *out_n_tiles)
{
    int n_tiles = d_out / NR;
    int n_kblocks = (L + Kc - 1) / Kc;
    size_t alloc_sz = (size_t)n_kblocks * n_tiles * Kc * NR * sizeof(float);
    alloc_sz = (alloc_sz + 255) & ~(size_t)255;
    float *Vp_all = (float *)aligned_alloc(256, alloc_sz);

    for (int kc = 0; kc < L; kc += Kc) {
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        klen &= ~1;
        if (klen < 2) break;
        int kb = kc / Kc;
        for (int t = 0; t < n_tiles; t++) {
            float *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
            pack_v_block(V + kc * d, d, Vp, klen, t * NR, NR);
        }
    }
    *out_n_kblocks = n_kblocks;
    *out_n_tiles = n_tiles;
    return Vp_all;
}

/* ─── FP16 kernel self-test ─── */
static void test_kernel_fp16(const char *name,
                              gemm_fp16_fn kern, gemm_fp16_fn kern_acc,
                              int MR, int NR) {
    int K = 64;
    _Float16 *A  = (_Float16 *)aligned_alloc(256, ((size_t)K*MR*2+255)&~(size_t)255);
    _Float16 *B  = (_Float16 *)aligned_alloc(256, ((size_t)K*NR*2+255)&~(size_t)255);
    float *C     = (float *)aligned_alloc(256, ((size_t)MR*NR*4+255)&~(size_t)255);
    double *Cr   = (double *)aligned_alloc(256, ((size_t)MR*NR*8+255)&~(size_t)255);
    memset(Cr, 0, (size_t)MR*NR*8);

    srand(12345);
    for (int i = 0; i < K * MR; i++)
        A[i] = (_Float16)((float)rand()/(float)RAND_MAX - 0.5f);
    for (int i = 0; i < K * NR; i++)
        B[i] = (_Float16)((float)rand()/(float)RAND_MAX - 0.5f);

    for (int m = 0; m < MR; m++)
        for (int n = 0; n < NR; n++) {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += (double)(float)A[k*MR+m] * (double)(float)B[k*NR+n];
            Cr[m*NR+n] = s;
        }

    memset(C, 0, (size_t)MR * NR * 4);
    int64_t ldc_bytes = (int64_t)NR * 4;
    kern(A, B, C, (int64_t)K, 0, ldc_bytes);

    double me = 0;
    for (int m = 0; m < MR; m++)
        for (int n = 0; n < NR; n++) {
            double e = fabs((double)C[m*NR+n] - Cr[m*NR+n]);
            if (e > me) me = e;
        }
    printf("Kernel %-8s init:  maxerr = %.2e  %s\n", name, me,
           me < 0.1 ? "OK" : "FAIL");

    if (kern_acc) {
        memset(C, 0, (size_t)MR * NR * 4);
        kern(A, B, C, 32, 0, ldc_bytes);
        kern_acc(A + 32*MR, B + 32*NR, C, 32, 0, ldc_bytes);

        me = 0;
        for (int m = 0; m < MR; m++)
            for (int n = 0; n < NR; n++) {
                double e = fabs((double)C[m*NR+n] - Cr[m*NR+n]);
                if (e > me) me = e;
            }
        printf("Kernel %-8s accum: maxerr = %.2e  %s\n", name, me,
               me < 0.1 ? "OK" : "FAIL");
    }
    free(A); free(B); free(C); free(Cr);
}

/* ─── FP16 NOEPI kernel self-test (fp16 C output) ─── */
static void test_kernel_fp16_noepi(const char *name,
                                     gemm_fp16_noepi_fn kern,
                                     gemm_fp16_noepi_fn kern_acc,
                                     int MR, int NR) {
    int K = 64;
    _Float16 *A  = (_Float16 *)aligned_alloc(256, ((size_t)K*MR*2+255)&~(size_t)255);
    _Float16 *B  = (_Float16 *)aligned_alloc(256, ((size_t)K*NR*2+255)&~(size_t)255);
    _Float16 *C  = (_Float16 *)aligned_alloc(256, ((size_t)MR*NR*2+255)&~(size_t)255);
    double *Cr   = (double *)aligned_alloc(256, ((size_t)MR*NR*8+255)&~(size_t)255);
    memset(Cr, 0, (size_t)MR*NR*8);

    srand(12345);
    for (int i = 0; i < K * MR; i++)
        A[i] = (_Float16)((float)rand()/(float)RAND_MAX - 0.5f);
    for (int i = 0; i < K * NR; i++)
        B[i] = (_Float16)((float)rand()/(float)RAND_MAX - 0.5f);

    for (int m = 0; m < MR; m++)
        for (int n = 0; n < NR; n++) {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += (double)(float)A[k*MR+m] * (double)(float)B[k*NR+n];
            Cr[m*NR+n] = s;
        }

    memset(C, 0, (size_t)MR * NR * 2);
    int64_t ldc_bytes = (int64_t)NR * sizeof(_Float16);
    kern(A, B, C, (int64_t)K, 0, ldc_bytes);

    double me = 0;
    for (int m = 0; m < MR; m++)
        for (int n = 0; n < NR; n++) {
            double e = fabs((double)(float)C[m*NR+n] - Cr[m*NR+n]);
            if (e > me) me = e;
        }
    printf("Kernel %-8s init:  maxerr = %.2e  %s\n", name, me,
           me < 0.5 ? "OK" : "FAIL");

    if (kern_acc) {
        memset(C, 0, (size_t)MR * NR * 2);
        kern(A, B, C, 32, 0, ldc_bytes);
        kern_acc(A + 32*MR, B + 32*NR, C, 32, 0, ldc_bytes);

        me = 0;
        for (int m = 0; m < MR; m++)
            for (int n = 0; n < NR; n++) {
                double e = fabs((double)(float)C[m*NR+n] - Cr[m*NR+n]);
                if (e > me) me = e;
            }
        printf("Kernel %-8s accum: maxerr = %.2e  %s\n", name, me,
               me < 0.5 ? "OK" : "FAIL");
    }
    free(A); free(B); free(C); free(Cr);
}

/* ─── Pre-pack V as FP16 ─── */
static _Float16 *prepack_v_fp16(const float *V, int d, int L, int d_out,
                                 int NR, int Kc, int *out_n_kblocks, int *out_n_tiles)
{
    int n_tiles = d_out / NR;
    int n_kblocks = (L + Kc - 1) / Kc;
    size_t alloc_sz = (size_t)n_kblocks * n_tiles * Kc * NR * sizeof(_Float16);
    alloc_sz = (alloc_sz + 255) & ~(size_t)255;
    _Float16 *Vp_all = (_Float16 *)aligned_alloc(256, alloc_sz);
    memset(Vp_all, 0, alloc_sz);

    for (int kc = 0; kc < L; kc += Kc) {
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        klen &= ~1;
        if (klen < 2) break;
        int kb = kc / Kc;
        for (int t = 0; t < n_tiles; t++) {
            _Float16 *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
            int col0 = t * NR;
            int ncopy = (col0 + NR <= d) ? NR : (d > col0 ? d - col0 : 0);
            for (int k = 0; k < klen; k++) {
                const float *vs = V + (kc + k) * d + col0;
                _Float16 *vp = Vp + k * NR;
                for (int j = 0; j < ncopy; j++) vp[j] = (_Float16)vs[j];
            }
        }
    }
    *out_n_kblocks = n_kblocks;
    *out_n_tiles = n_tiles;
    return Vp_all;
}

/* ─── Pass1 for FP16: exp2 in fp32 → convert to fp16 → store ─── */
/* Helper: convert fp32 SVE vec to fp16 and store MR elements via st1h {z.s} */
static inline void store_f32_as_f16(svbool_t pg, _Float16 *dst, svfloat32_t src) {
    /* fcvt z.h, p/m, z.s puts fp16 in low 16 bits of each 32-bit container.
     * st1h {z.s}, p, [addr] stores those low 16 bits contiguously. */
    svfloat32_t tmp;
    __asm__("fcvt %0.h, %1/m, %2.s" : "=w"(tmp) : "Upl"(pg), "w"(src));
    svst1h(pg, (int16_t *)dst, svreinterpret_s32(tmp));
}

static void pass1_block_exp_fp16(
    const float *S, int ld_s, int kc, int klen,
    int MR, const float *rmax,
    _Float16 *Pp,   /* output: col-major [klen][MR] as fp16 */
    float *rsum)     /* accumulate partial sums in fp32 */
{
    union { uint32_t u; float f; } su = { .u = FEXPA_SHIFT_U32 };
    float shift = su.f;
    svbool_t pg = svwhilelt_b32(0, MR);

    float bbuf[16] __attribute__((aligned(256)));
    for (int m = 0; m < MR; m++) bbuf[m] = -rmax[m] + shift;
    svfloat32_t vbias = svld1(pg, bbuf);

    svfloat32_t vs0 = svdup_f32(0.0f), vs1 = svdup_f32(0.0f);
    svfloat32_t vs2 = svdup_f32(0.0f), vs3 = svdup_f32(0.0f);
    const float *S_blk = S + (int64_t)kc * ld_s;
    int j;

    for (j = 0; j + 4 <= klen; j += 4) {
        svfloat32_t s0 = svld1(pg, S_blk + (j+0) * ld_s);
        svfloat32_t s1 = svld1(pg, S_blk + (j+1) * ld_s);
        svfloat32_t s2 = svld1(pg, S_blk + (j+2) * ld_s);
        svfloat32_t s3 = svld1(pg, S_blk + (j+3) * ld_s);
        svfloat32_t p0 = sve_fexpa(svadd_m(pg, s0, vbias));
        svfloat32_t p1 = sve_fexpa(svadd_m(pg, s1, vbias));
        svfloat32_t p2 = sve_fexpa(svadd_m(pg, s2, vbias));
        svfloat32_t p3 = sve_fexpa(svadd_m(pg, s3, vbias));
        vs0 = svadd_m(pg, vs0, p0);
        vs1 = svadd_m(pg, vs1, p1);
        vs2 = svadd_m(pg, vs2, p2);
        vs3 = svadd_m(pg, vs3, p3);
        /* SVE fcvt+st1h: 2 insns per col instead of ~37 scalar ops */
        store_f32_as_f16(pg, Pp + (j+0)*MR, p0);
        store_f32_as_f16(pg, Pp + (j+1)*MR, p1);
        store_f32_as_f16(pg, Pp + (j+2)*MR, p2);
        store_f32_as_f16(pg, Pp + (j+3)*MR, p3);
    }
    for (; j < klen; j++) {
        svfloat32_t s = svld1(pg, S_blk + j * ld_s);
        svfloat32_t p = sve_fexpa(svadd_m(pg, s, vbias));
        vs0 = svadd_m(pg, vs0, p);
        store_f32_as_f16(pg, Pp + j*MR, p);
    }
    svfloat32_t vsum = svadd_m(pg, svadd_m(pg, vs0, vs1),
                                    svadd_m(pg, vs2, vs3));
    float sbuf[16] __attribute__((aligned(256)));
    svst1(svptrue_b32(), sbuf, vsum);
    for (int m = 0; m < MR; m++) rsum[m] += sbuf[m];
}

/* ─── FP16 pipelined 2-pass ─── */
static void run_2pass_fp16(
    const float *S, int ld_s,
    const _Float16 *Vp_all,
    float *O, int ld_o,
    int MR, int L, int d_out, int NR, int Kc,
    int n_tiles, int pipe_ahead,
    gemm_fp16_fn kern_init, gemm_fp16_fn kern_accum,
    timing_t *tm)
{
    int64_t ldo_bytes = (int64_t)ld_o * sizeof(float);
    int n_kblocks = (L + Kc - 1) / Kc;
    int pipe_slots = pipe_ahead + 1;

    /* FP16 ring buffer with L1-conflict padding */
    size_t slot_data = (size_t)Kc * MR * sizeof(_Float16);
    int slot_lines = (int)((slot_data + L1_LINE - 1) / L1_LINE);
    int best_pad = 0, best_ov = L1_SETS;
    for (int pad = 0; pad <= 20; pad++) {
        int stride = slot_lines + pad;
        int shift = (pipe_ahead * stride) % L1_SETS;
        int ov = l1_set_overlap(slot_lines, shift);
        if (ov < best_ov) { best_ov = ov; best_pad = pad; if (ov == 0) break; }
    }
    size_t slot_bytes = (size_t)(slot_lines + best_pad) * L1_LINE;
    size_t slot_stride_h = slot_bytes / sizeof(_Float16);
    _Float16 *Pp_ring = (_Float16 *)aligned_alloc(256, pipe_slots * slot_bytes);

    float rmax[16], rsum[16];
    memset(rsum, 0, sizeof(rsum));

    uint64_t t0 = rdtick();
    {
        svbool_t pg_mr = svwhilelt_b32(0, MR);
        svfloat32_t vmax = svdup_f32(-1e30f);
        for (int j = 0; j < L; j++)
            vmax = svmax_m(pg_mr, vmax, svld1(pg_mr, S + (int64_t)j * ld_s));
        float mbuf[16] __attribute__((aligned(256)));
        svst1(svptrue_b32(), mbuf, vmax);
        for (int m = 0; m < MR; m++) rmax[m] = mbuf[m];
    }

    int fill = (pipe_ahead < n_kblocks) ? pipe_ahead : n_kblocks;
    for (int kb = 0; kb < fill; kb++) {
        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        pass1_block_exp_fp16(S, ld_s, kc, klen, MR, rmax,
                              Pp_ring + (kb % pipe_slots) * slot_stride_h, rsum);
    }

    uint64_t t1 = rdtick();
    tm->pass1 = (double)(t1 - t0) * tick2sec;

    for (int kb = 0; kb < n_kblocks; kb++) {
        const _Float16 *Pp_blk = Pp_ring + (kb % pipe_slots) * slot_stride_h;

        if (kb + pipe_ahead < n_kblocks) {
            int ahead = kb + pipe_ahead;
            int kc = ahead * Kc;
            int klen = (kc + Kc <= L) ? Kc : (L - kc);
            pass1_block_exp_fp16(S, ld_s, kc, klen, MR, rmax,
                                  Pp_ring + (ahead % pipe_slots) * slot_stride_h, rsum);
        }

        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        klen &= ~1;
        if (klen < 2) continue;

        /* Prefetch S for next pass1 block into L1 during GEMM.
         * S lives in L2 after fill pass; bringing it to L1 before pass1
         * reduces load latency from ~30cy (L2) to ~11cy (L1). */
        int sprf_blk = kb + pipe_ahead + 1;
        int sprf_cols = 0;
        const char *sprf_base = NULL;
        if (sprf_blk < n_kblocks) {
            int sprf_kc = sprf_blk * Kc;
            sprf_cols = (sprf_kc + Kc <= L) ? Kc : (L - sprf_kc);
            sprf_base = (const char *)(S + (int64_t)sprf_kc * ld_s);
        }
        int sprf_per_tile = (sprf_cols + n_tiles - 1) / n_tiles;

        for (int t = 0; t < n_tiles; t++) {
            /* A re-touch only if B tile very large (fp16: NR*Kc*2) */
            if (t > 0 && klen * NR * (int)sizeof(_Float16) > 40960) {
                const char *a = (const char *)Pp_blk;
                for (int ln = 0; ln < klen * MR * 2; ln += 256)
                    __builtin_prefetch(a + ln, 0, 3);
            }
            const _Float16 *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
            if (kb == 0)
                kern_init(Pp_blk, Vp, O + t * NR,
                          (int64_t)klen, 0, ldo_bytes);
            else
                kern_accum(Pp_blk, Vp, O + t * NR,
                           (int64_t)klen, 0, ldo_bytes);
            /* S prefetch: spread across tiles to avoid decode burst */
            if (sprf_base) {
                int j0 = t * sprf_per_tile;
                int j1 = j0 + sprf_per_tile;
                if (j1 > sprf_cols) j1 = sprf_cols;
                for (int j = j0; j < j1; j++)
                    __builtin_prefetch(sprf_base + (int64_t)j * ld_s * sizeof(float), 0, 3);
            }
        }
    }
    uint64_t t2 = rdtick();
    tm->pass2 = (double)(t2 - t1) * tick2sec;

    normalize(O, ld_o, rsum, MR, d_out);
    uint64_t t3 = rdtick();
    tm->norm = (double)(t3 - t2) * tick2sec;

    free(Pp_ring);
}

/* ─── FP16 GEMM-only benchmark ─── */
static double bench_gemm_only_fp16(
    const _Float16 *Pp, const _Float16 *Vp_all,
    float *O, int ld_o,
    int MR, int L, int d_out, int NR, int Kc,
    int n_tiles,
    gemm_fp16_fn kern_init, gemm_fp16_fn kern_accum)
{
    int64_t ldo_bytes = (int64_t)ld_o * sizeof(float);
    int nrep = (L <= 2048) ? 21 : 7;
    int warmup = (L <= 2048) ? 5 : 2;
    double best_sec = 1e30;

    for (int r = 0; r < nrep; r++) {
        memset(O, 0, (size_t)MR * ld_o * 4);

        uint64_t t0 = rdtick();
        for (int kc = 0; kc < L; kc += Kc) {
            int klen = (kc + Kc <= L) ? Kc : (L - kc);
            klen &= ~1;
            if (klen < 2) break;
            int kb = kc / Kc;
            const _Float16 *Pp_blk = Pp + kc * MR;

            for (int t = 0; t < n_tiles; t++) {
                const _Float16 *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
                if (kc == 0)
                    kern_init(Pp_blk, Vp, O + t * NR,
                              (int64_t)klen, 0, ldo_bytes);
                else
                    kern_accum(Pp_blk, Vp, O + t * NR,
                               (int64_t)klen, 0, ldo_bytes);
            }
        }
        uint64_t t1 = rdtick();
        double sec = (double)(t1 - t0) * tick2sec;
        if (r >= warmup && sec < best_sec) best_sec = sec;
    }
    return best_sec;
}

/* ─── FP16 NOEPI GEMM-only benchmark (fp16 C) ─── */
static double bench_gemm_only_fp16_noepi(
    const _Float16 *Pp, const _Float16 *Vp_all,
    _Float16 *O, int ld_o,
    int MR, int L, int d_out, int NR, int Kc,
    int n_tiles,
    gemm_fp16_noepi_fn kern_init, gemm_fp16_noepi_fn kern_accum)
{
    int64_t ldo_bytes = (int64_t)ld_o * sizeof(_Float16);
    int nrep = (L <= 2048) ? 21 : 7;
    int warmup = (L <= 2048) ? 5 : 2;
    double best_sec = 1e30;

    for (int r = 0; r < nrep; r++) {
        memset(O, 0, (size_t)MR * ld_o * sizeof(_Float16));

        uint64_t t0 = rdtick();
        for (int kc = 0; kc < L; kc += Kc) {
            int klen = (kc + Kc <= L) ? Kc : (L - kc);
            klen &= ~1;
            if (klen < 2) break;
            int kb = kc / Kc;
            const _Float16 *Pp_blk = Pp + kc * MR;

            for (int t = 0; t < n_tiles; t++) {
                const _Float16 *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
                if (kc == 0)
                    kern_init(Pp_blk, Vp, O + t * NR,
                              (int64_t)klen, 0, ldo_bytes);
                else
                    kern_accum(Pp_blk, Vp, O + t * NR,
                               (int64_t)klen, 0, ldo_bytes);
            }
        }
        uint64_t t1 = rdtick();
        double sec = (double)(t1 - t0) * tick2sec;
        if (r >= warmup && sec < best_sec) best_sec = sec;
    }
    return best_sec;
}

/* ─── FP16 NOEPI pipelined 2-pass (fp16 C accumulation) ─── */
static void run_2pass_fp16_noepi(
    const float *S, int ld_s,
    const _Float16 *Vp_all,
    _Float16 *O, int ld_o,
    int MR, int L, int d_out, int NR, int Kc,
    int n_tiles, int pipe_ahead,
    gemm_fp16_noepi_fn kern_init, gemm_fp16_noepi_fn kern_accum,
    timing_t *tm)
{
    int64_t ldo_bytes = (int64_t)ld_o * sizeof(_Float16);
    int n_kblocks = (L + Kc - 1) / Kc;
    int pipe_slots = pipe_ahead + 1;

    /* FP16 ring buffer with L1-conflict padding */
    size_t slot_data = (size_t)Kc * MR * sizeof(_Float16);
    int slot_lines = (int)((slot_data + L1_LINE - 1) / L1_LINE);
    int best_pad = 0, best_ov = L1_SETS;
    for (int pad = 0; pad <= 20; pad++) {
        int stride = slot_lines + pad;
        int shift = (pipe_ahead * stride) % L1_SETS;
        int ov = l1_set_overlap(slot_lines, shift);
        if (ov < best_ov) { best_ov = ov; best_pad = pad; if (ov == 0) break; }
    }
    size_t slot_bytes = (size_t)(slot_lines + best_pad) * L1_LINE;
    size_t slot_stride_h = slot_bytes / sizeof(_Float16);
    _Float16 *Pp_ring = (_Float16 *)aligned_alloc(256, pipe_slots * slot_bytes);

    float rmax[16], rsum[16];
    memset(rsum, 0, sizeof(rsum));

    uint64_t t0 = rdtick();
    {
        svbool_t pg_mr = svwhilelt_b32(0, MR);
        svfloat32_t vmax = svdup_f32(-1e30f);
        for (int j = 0; j < L; j++)
            vmax = svmax_m(pg_mr, vmax, svld1(pg_mr, S + (int64_t)j * ld_s));
        float mbuf[16] __attribute__((aligned(256)));
        svst1(svptrue_b32(), mbuf, vmax);
        for (int m = 0; m < MR; m++) rmax[m] = mbuf[m];
    }

    int fill = (pipe_ahead < n_kblocks) ? pipe_ahead : n_kblocks;
    for (int kb = 0; kb < fill; kb++) {
        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        pass1_block_exp_fp16(S, ld_s, kc, klen, MR, rmax,
                              Pp_ring + (kb % pipe_slots) * slot_stride_h, rsum);
    }

    uint64_t t1 = rdtick();
    tm->pass1 = (double)(t1 - t0) * tick2sec;

    for (int kb = 0; kb < n_kblocks; kb++) {
        const _Float16 *Pp_blk = Pp_ring + (kb % pipe_slots) * slot_stride_h;

        if (kb + pipe_ahead < n_kblocks) {
            int ahead = kb + pipe_ahead;
            int kc = ahead * Kc;
            int klen = (kc + Kc <= L) ? Kc : (L - kc);
            pass1_block_exp_fp16(S, ld_s, kc, klen, MR, rmax,
                                  Pp_ring + (ahead % pipe_slots) * slot_stride_h, rsum);
        }

        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        klen &= ~1;
        if (klen < 2) continue;

        /* S prefetch for next pass1 block */
        int sprf_blk = kb + pipe_ahead + 1;
        int sprf_cols = 0;
        const char *sprf_base = NULL;
        if (sprf_blk < n_kblocks) {
            int sprf_kc = sprf_blk * Kc;
            sprf_cols = (sprf_kc + Kc <= L) ? Kc : (L - sprf_kc);
            sprf_base = (const char *)(S + (int64_t)sprf_kc * ld_s);
        }
        int sprf_per_tile = (sprf_cols + n_tiles - 1) / n_tiles;

        for (int t = 0; t < n_tiles; t++) {
            /* A re-touch only if B tile very large (fp16: NR*Kc*2) */
            if (t > 0 && klen * NR * (int)sizeof(_Float16) > 40960) {
                const char *a = (const char *)Pp_blk;
                for (int ln = 0; ln < klen * MR * 2; ln += 256)
                    __builtin_prefetch(a + ln, 0, 3);
            }
            const _Float16 *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
            if (kb == 0)
                kern_init(Pp_blk, Vp, O + t * NR,
                          (int64_t)klen, 0, ldo_bytes);
            else
                kern_accum(Pp_blk, Vp, O + t * NR,
                           (int64_t)klen, 0, ldo_bytes);
            /* S prefetch: spread across tiles */
            if (sprf_base) {
                int j0 = t * sprf_per_tile;
                int j1 = j0 + sprf_per_tile;
                if (j1 > sprf_cols) j1 = sprf_cols;
                for (int j = j0; j < j1; j++)
                    __builtin_prefetch(sprf_base + (int64_t)j * ld_s * sizeof(float), 0, 3);
            }
        }
    }
    uint64_t t2 = rdtick();
    tm->pass2 = (double)(t2 - t1) * tick2sec;

    normalize_fp16(O, ld_o, rsum, MR, d_out);
    uint64_t t3 = rdtick();
    tm->norm = (double)(t3 - t2) * tick2sec;

    free(Pp_ring);
}

/* ─── L1 cache-conflict padding for ring buffer slots ─── */
/*
 * A64FX L1D: 64KB, 4-way, 256B line, 64 sets.
 * Set index = (addr >> 8) & 63.  Same-set stride = 16KB.
 *
 * The pipeline reads slot[kb % PIPE_SLOTS] while writing slot[(kb+PIPE_AHEAD) % PIPE_SLOTS].
 * If both slots map to the same L1 sets, the write evicts the read data.
 * We pad the slot stride so that the write-slot and read-slot have minimal set overlap.
 */

/* Compute L1 set overlap between two regions of 'width' cache lines
 * shifted by 'shift' sets in a circular 64-set space. */
static int l1_set_overlap(int width, int shift) {
    int s = shift % L1_SETS;
    if (s < 0) s += L1_SETS;
    /* Count elements in [0,width) ∩ [s,s+width) mod 64 */
    int overlap = 0;
    if (s < width)
        overlap += width - s;   /* overlap at the beginning */
    if (s + width > L1_SETS)
        overlap += (s + width - L1_SETS < width) ?
                    s + width - L1_SETS : width;  /* wrap-around overlap */
    return overlap;
}

/* Find optimal padding (0..16 cache lines) for ring buffer slots to minimize
 * L1 set overlap between the pass1-write slot and GEMM-read slot.
 * Returns the padded slot stride in bytes (256-aligned). */
static size_t optimal_slot_stride(int MR, int Kc, int pipe_ahead, int *out_pad, int *out_overlap) {
    size_t slot_data = (size_t)Kc * MR * sizeof(float);
    int slot_lines = (int)((slot_data + L1_LINE - 1) / L1_LINE);

    int best_pad = 0, best_overlap = L1_SETS;
    for (int pad = 0; pad <= 16; pad++) {
        int stride_lines = slot_lines + pad;
        int shift = (pipe_ahead * stride_lines) % L1_SETS;
        int ov = l1_set_overlap(slot_lines, shift);
        if (ov < best_overlap) {
            best_overlap = ov;
            best_pad = pad;
            if (ov == 0) break;
        }
    }
    if (out_pad) *out_pad = best_pad;
    if (out_overlap) *out_overlap = best_overlap;
    return (size_t)(slot_lines + best_pad) * L1_LINE;
}

/* ─── Pipelined 2-pass: max → fill pipeline → interleave pass1_exp+GEMM ─── */
#define PIPE_AHEAD_DEFAULT 2

static void run_2pass(
    const float *S, int ld_s,
    const float *Vp_all,  /* pre-packed V tiles */
    float *O, int ld_o,
    int MR, int L, int d_out, int NR, int Kc,
    int n_tiles, int pipe_ahead,
    gemm_fn kern_init, gemm_fn kern_accum,
    timing_t *tm)
{
    int64_t ldo_bytes = (int64_t)ld_o * sizeof(float);
    int n_kblocks = (L + Kc - 1) / Kc;
    int pipe_slots = pipe_ahead + 1;

    /* Ring buffer: pipe_slots slots with L1-conflict-aware padding */
    size_t slot_bytes = optimal_slot_stride(MR, Kc, pipe_ahead, NULL, NULL);
    size_t slot_stride = slot_bytes / sizeof(float);
    float *Pp_ring = aligned_alloc(256, pipe_slots * slot_bytes);

    float rmax[16], rsum[16];
    memset(rsum, 0, sizeof(rsum));

    /* Phase 0: Find global row maxima (column-major S: all rows per column) */
    uint64_t t0 = rdtick();
    {
        svbool_t pg_mr = svwhilelt_b32(0, MR);
        svfloat32_t vmax = svdup_f32(-1e30f);
        for (int j = 0; j < L; j++)
            vmax = svmax_m(pg_mr, vmax, svld1(pg_mr, S + (int64_t)j * ld_s));
        float mbuf[16] __attribute__((aligned(256)));
        svst1(svptrue_b32(), mbuf, vmax);
        for (int m = 0; m < MR; m++) rmax[m] = mbuf[m];
    }

    /* Phase 1: Fill pipeline with first pipe_ahead pass1 blocks */
    int fill = (pipe_ahead < n_kblocks) ? pipe_ahead : n_kblocks;
    for (int kb = 0; kb < fill; kb++) {
        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        pass1_block_exp(S, ld_s, kc, klen, MR, rmax,
                        Pp_ring + (kb % pipe_slots) * slot_stride, rsum);
    }

    uint64_t t1 = rdtick();
    tm->pass1 = (double)(t1 - t0) * tick2sec;  /* max-find + fill */

    /* Phase 2: Pipeline — pass1(kb+ahead) then pass2(kb) */
    for (int kb = 0; kb < n_kblocks; kb++) {
        const float *Pp_blk = Pp_ring + (kb % pipe_slots) * slot_stride;

        /* Produce: pass1 for block kb+pipe_ahead */
        if (kb + pipe_ahead < n_kblocks) {
            int ahead = kb + pipe_ahead;
            int kc = ahead * Kc;
            int klen = (kc + Kc <= L) ? Kc : (L - kc);
            pass1_block_exp(S, ld_s, kc, klen, MR, rmax,
                            Pp_ring + (ahead % pipe_slots) * slot_stride,
                            rsum);
        }

        /* Consume: GEMM for block kb */
        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        klen &= ~1;
        if (klen < 2) continue;

        /* Prefetch S data for the NEXT pass1 block into L1 */
        {
            int next_ahead = kb + 1 + pipe_ahead;
            if (next_ahead < n_kblocks) {
                const float *base = S + (int64_t)(next_ahead * Kc) * ld_s;
                int nlines = (Kc * ld_s * (int)sizeof(float) + 255) / 256;
                for (int line = 0; line < nlines; line++)
                    __builtin_prefetch(base + line * 64, 0, 3);
            }
        }

        for (int t = 0; t < n_tiles; t++) {
            /* Re-touch A(Pp) into L1 when B tile > 40KB (evicts A) */
            if (t > 0 && klen * NR * (int)sizeof(float) > 40960) {
                const char *a = (const char *)Pp_blk;
                for (int ln = 0; ln < klen * MR * 4; ln += 256)
                    __builtin_prefetch(a + ln, 0, 3);
            }
            const float *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
            if (kb == 0)
                kern_init(Pp_blk, Vp, O + t * NR,
                          (int64_t)klen, 0, ldo_bytes);
            else
                kern_accum(Pp_blk, Vp, O + t * NR,
                           (int64_t)klen, 0, ldo_bytes);
        }
    }
    uint64_t t2 = rdtick();
    tm->pass2 = (double)(t2 - t1) * tick2sec;  /* fused pipeline */

    /* Phase 3: Normalize */
    normalize(O, ld_o, rsum, MR, d_out);
    uint64_t t3 = rdtick();
    tm->norm = (double)(t3 - t2) * tick2sec;

    free(Pp_ring);
}

/* ─── Online softmax: no separate max-find pass ─── */
/*
 * FlashAttention-style online rescaling:
 *   - Process k-blocks sequentially
 *   - Find local max per block, update running global max
 *   - When max changes: rescale O and rsum by exp2(old_max - new_max)
 *   - No pipeline needed: Pp written then immediately consumed by GEMM (L1 warm)
 */
static void run_online(
    const float *S, int ld_s,
    const float *Vp_all,
    float *O, int ld_o,
    int MR, int L, int d_out, int NR, int Kc,
    int n_tiles,
    gemm_fn kern_init, gemm_fn kern_accum,
    timing_t *tm)
{
    int64_t ldo_bytes = (int64_t)ld_o * sizeof(float);
    int n_kblocks = (L + Kc - 1) / Kc;

    /* Single Pp buffer: Kc * MR floats, fits L1 */
    size_t pp_elems = (size_t)Kc * MR;
    size_t pp_bytes = (pp_elems * sizeof(float) + 255) & ~(size_t)255;
    float *Pp = aligned_alloc(256, pp_bytes);

    float rmax[16], rsum[16];
    float rmax_buf[16] __attribute__((aligned(256)));
    for (int m = 0; m < MR; m++) rmax[m] = -1e30f;
    memset(rsum, 0, sizeof(rsum));

    svbool_t pg_mr = svwhilelt_b32(0, MR);
    const int vl = svcntw();

    uint64_t t0 = rdtick();

    for (int kb = 0; kb < n_kblocks; kb++) {
        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);

        /* Step 1: Find local max within this block's columns */
        const float *S_blk = S + (int64_t)kc * ld_s;
        svfloat32_t vlmax = svdup_f32(-1e30f);
        for (int j = 0; j < klen; j++)
            vlmax = svmax_m(pg_mr, vlmax, svld1(pg_mr, S_blk + j * ld_s));
        svst1(svptrue_b32(), rmax_buf, vlmax);

        /* Step 2: Check if max changed, rescale if needed */
        int max_changed = 0;
        for (int m = 0; m < MR; m++)
            if (rmax_buf[m] > rmax[m]) { max_changed = 1; break; }

        if (max_changed) {
            /* Compute per-row correction: corr = exp2(old_max - new_max) */
            union { uint32_t u; float f; } su = { .u = FEXPA_SHIFT_U32 };
            float shift = su.f;
            float cbuf[16] __attribute__((aligned(256)));
            for (int m = 0; m < MR; m++) {
                float new_max = (rmax_buf[m] > rmax[m]) ? rmax_buf[m] : rmax[m];
                cbuf[m] = rmax[m] - new_max + shift;
                rmax[m] = new_max;
            }
            svfloat32_t vcorr = sve_fexpa(svld1(pg_mr, cbuf));

            /* Rescale running sums */
            float sbuf[16] __attribute__((aligned(256)));
            for (int m = 0; m < MR; m++) sbuf[m] = rsum[m];
            svfloat32_t vs = svmul_m(pg_mr, svld1(pg_mr, sbuf), vcorr);
            svst1(svptrue_b32(), sbuf, vs);
            for (int m = 0; m < MR; m++) rsum[m] = sbuf[m];

            /* Rescale O: O[m][j] *= corr[m] for all m, j */
            if (kb > 0) {  /* O is non-zero only after first GEMM */
                float corr_arr[16] __attribute__((aligned(256)));
                svst1(svptrue_b32(), corr_arr, vcorr);
                for (int m = 0; m < MR; m++) {
                    float c = corr_arr[m];
                    svfloat32_t vc = svdup_f32(c);
                    float *row = O + m * ld_o;
                    int j;
                    for (j = 0; j + vl <= d_out; j += vl)
                        svst1(svptrue_b32(), row + j,
                              svmul_m(svptrue_b32(), svld1(svptrue_b32(), row + j), vc));
                    if (j < d_out) {
                        svbool_t pg = svwhilelt_b32(j, d_out);
                        svst1(pg, row + j, svmul_m(pg, svld1(pg, row + j), vc));
                    }
                }
            }
        }

        /* Step 3: pass1_block_exp (uses current rmax) */
        pass1_block_exp(S, ld_s, kc, klen, MR, rmax, Pp, rsum);

        /* Prefetch S data for NEXT block into L1 */
        if (kb + 1 < n_kblocks) {
            const float *base = S + (int64_t)((kb + 1) * Kc) * ld_s;
            int nlines = (Kc * ld_s * (int)sizeof(float) + 255) / 256;
            for (int line = 0; line < nlines; line++)
                __builtin_prefetch(base + line * 64, 0, 3);
        }

        /* Step 4: GEMM for this block */
        klen &= ~1;
        if (klen < 2) continue;
        for (int t = 0; t < n_tiles; t++) {
            const float *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
            if (kb == 0)
                kern_init(Pp, Vp, O + t * NR,
                          (int64_t)klen, 0, ldo_bytes);
            else
                kern_accum(Pp, Vp, O + t * NR,
                           (int64_t)klen, 0, ldo_bytes);
        }
    }

    uint64_t t1 = rdtick();
    tm->pass1 = 0;  /* No separate max-find pass */
    tm->pass2 = (double)(t1 - t0) * tick2sec;  /* everything fused */

    /* Normalize */
    normalize(O, ld_o, rsum, MR, d_out);
    uint64_t t2 = rdtick();
    tm->norm = (double)(t2 - t1) * tick2sec;

    free(Pp);
}

/* ─── 3-pass separated: max → full pass1 → GEMM-only (no L1 pollution) ─── */
/*
 * At large L the 2-pass pipeline's interleaved pass1 thrashes L1 during GEMM
 * (S reads + Pp writes compete with B streaming).  Separating pass1 gives GEMM
 * a clean L1 working set, matching the GEMM-only ceiling.
 *
 * Pp_all lives in L2 (~L*MR*2 bytes, e.g. 786 KB for L=32768 MR=12).
 * Phase 2 streams Pp sequentially from L2—hardware prefetcher handles it.
 */
static void run_3pass_fp16_noepi(
    const float *S, int ld_s,
    const _Float16 *Vp_all,
    _Float16 *O, int ld_o,
    int MR, int L, int d_out, int NR, int Kc,
    int n_tiles,
    gemm_fp16_noepi_fn kern_init, gemm_fp16_noepi_fn kern_accum,
    timing_t *tm)
{
    int64_t ldo_bytes = (int64_t)ld_o * sizeof(_Float16);
    int n_kblocks = (L + Kc - 1) / Kc;

    /* Full Pp buffer in L2: L × MR fp16 values */
    size_t pp_bytes = ((size_t)L * MR * sizeof(_Float16) + 255) & ~(size_t)255;
    _Float16 *Pp_all_h = (_Float16 *)aligned_alloc(256, pp_bytes);

    float rmax[16], rsum[16];
    memset(rsum, 0, sizeof(rsum));

    /* Phase 0: Find global row maxima */
    uint64_t t0 = rdtick();
    {
        svbool_t pg_mr = svwhilelt_b32(0, MR);
        svfloat32_t vmax = svdup_f32(-1e30f);
        for (int j = 0; j < L; j++)
            vmax = svmax_m(pg_mr, vmax, svld1(pg_mr, S + (int64_t)j * ld_s));
        float mbuf[16] __attribute__((aligned(256)));
        svst1(svptrue_b32(), mbuf, vmax);
        for (int m = 0; m < MR; m++) rmax[m] = mbuf[m];
    }

    /* Phase 1: Compute ALL Pp blocks (exp2 → fp16, sequential write to L2) */
    for (int kb = 0; kb < n_kblocks; kb++) {
        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        pass1_block_exp_fp16(S, ld_s, kc, klen, MR, rmax,
                              Pp_all_h + kc * MR, rsum);
    }
    uint64_t t1 = rdtick();
    tm->pass1 = (double)(t1 - t0) * tick2sec;

    /* Phase 2: Pure GEMM — clean L1 working set, no pass1 interference */
    for (int kb = 0; kb < n_kblocks; kb++) {
        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);
        klen &= ~1;
        if (klen < 2) continue;
        const _Float16 *Pp_blk = Pp_all_h + kc * MR;

        for (int t = 0; t < n_tiles; t++) {
            /* A re-touch: after 2 B tiles (64KB), A (6KB) likely evicted from L1 */
            if (t == 2) {
                const char *a = (const char *)Pp_blk;
                for (int ln = 0; ln < klen * MR * 2; ln += 256)
                    __builtin_prefetch(a + ln, 0, 3);
            }
            const _Float16 *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
            if (kb == 0)
                kern_init(Pp_blk, Vp, O + t * NR,
                          (int64_t)klen, 0, ldo_bytes);
            else
                kern_accum(Pp_blk, Vp, O + t * NR,
                           (int64_t)klen, 0, ldo_bytes);
        }
    }
    uint64_t t2 = rdtick();
    tm->pass2 = (double)(t2 - t1) * tick2sec;

    /* Phase 3: Normalize */
    normalize_fp16(O, ld_o, rsum, MR, d_out);
    uint64_t t3 = rdtick();
    tm->norm = (double)(t3 - t2) * tick2sec;

    free(Pp_all_h);
}

/* ─── Online softmax for FP16 NOEPI (no separate max-find pass) ─── */
/*
 * FlashAttention-style online rescaling with fp16 NOEPI kernels:
 *   - Process k-blocks sequentially
 *   - Find local max per block, update running global max
 *   - When max changes: rescale O (fp16) and rsum by exp2(old_max - new_max)
 *   - No pipeline needed: Pp written then immediately consumed by GEMM (L1 warm)
 *   - O rescaling uses SVE fp16 vectorized multiply (32 elements per vector)
 */
static void run_online_fp16_noepi(
    const float *S, int ld_s,
    const _Float16 *Vp_all,
    _Float16 *O, int ld_o,
    int MR, int L, int d_out, int NR, int Kc,
    int n_tiles,
    gemm_fp16_noepi_fn kern_init, gemm_fp16_noepi_fn kern_accum,
    timing_t *tm)
{
    int64_t ldo_bytes = (int64_t)ld_o * sizeof(_Float16);
    int n_kblocks = (L + Kc - 1) / Kc;

    /* Single Pp buffer: Kc * MR fp16 elements, fits L1 */
    size_t pp_elems = (size_t)Kc * MR;
    size_t pp_bytes = (pp_elems * sizeof(_Float16) + 255) & ~(size_t)255;
    _Float16 *Pp = (_Float16 *)aligned_alloc(256, pp_bytes);

    float rmax[16], rsum[16];
    float rmax_buf[16] __attribute__((aligned(256)));
    for (int m = 0; m < MR; m++) rmax[m] = -1e30f;
    memset(rsum, 0, sizeof(rsum));

    svbool_t pg_mr = svwhilelt_b32(0, MR);
    const int vl_h = svcnth();  /* 32 for fp16 on 512-bit SVE */

    uint64_t t0 = rdtick();

    for (int kb = 0; kb < n_kblocks; kb++) {
        int kc = kb * Kc;
        int klen = (kc + Kc <= L) ? Kc : (L - kc);

        /* Step 1: Find local max within this block's columns */
        const float *S_blk = S + (int64_t)kc * ld_s;
        svfloat32_t vlmax = svdup_f32(-1e30f);
        for (int j = 0; j < klen; j++)
            vlmax = svmax_m(pg_mr, vlmax, svld1(pg_mr, S_blk + j * ld_s));
        svst1(svptrue_b32(), rmax_buf, vlmax);

        /* Step 2: Check if max changed, rescale if needed */
        int max_changed = 0;
        for (int m = 0; m < MR; m++)
            if (rmax_buf[m] > rmax[m]) { max_changed = 1; break; }

        if (max_changed) {
            /* Compute per-row correction: corr = exp2(old_max - new_max) */
            union { uint32_t u; float f; } su = { .u = FEXPA_SHIFT_U32 };
            float shift = su.f;
            float cbuf[16] __attribute__((aligned(256)));
            for (int m = 0; m < MR; m++) {
                float new_max = (rmax_buf[m] > rmax[m]) ? rmax_buf[m] : rmax[m];
                cbuf[m] = rmax[m] - new_max + shift;
                rmax[m] = new_max;
            }
            svfloat32_t vcorr = sve_fexpa(svld1(pg_mr, cbuf));

            /* Rescale running sums (fp32) */
            float sbuf[16] __attribute__((aligned(256)));
            for (int m = 0; m < MR; m++) sbuf[m] = rsum[m];
            svfloat32_t vs = svmul_m(pg_mr, svld1(pg_mr, sbuf), vcorr);
            svst1(svptrue_b32(), sbuf, vs);
            for (int m = 0; m < MR; m++) rsum[m] = sbuf[m];

            /* Rescale O: O[m][j] *= corr[m] (SVE via fp32 widening) */
            if (kb > 0) {  /* O is non-zero only after first GEMM */
                float corr_arr[16] __attribute__((aligned(256)));
                svst1(svptrue_b32(), corr_arr, vcorr);
                svbool_t pg_all = svptrue_b32();
                const int vl_w = svcntw();  /* 16 for fp32 */
                for (int m = 0; m < MR; m++) {
                    svfloat32_t vc = svdup_f32(corr_arr[m]);
                    _Float16 *row = O + m * ld_o;
                    int j;
                    for (j = 0; j + vl_w <= d_out; j += vl_w) {
                        svfloat32_t v;
                        __asm__("ld1h {%0.s}, %1/z, [%2]"
                                : "=w"(v) : "Upl"(pg_all), "r"(row + j));
                        v = svmul_m(pg_all, v, vc);
                        store_f32_as_f16(pg_all, row + j, v);
                    }
                    if (j < d_out) {
                        svbool_t pg_tail = svwhilelt_b32(j, d_out);
                        svfloat32_t v;
                        __asm__("ld1h {%0.s}, %1/z, [%2]"
                                : "=w"(v) : "Upl"(pg_tail), "r"(row + j));
                        v = svmul_m(pg_tail, v, vc);
                        store_f32_as_f16(pg_tail, row + j, v);
                    }
                }
            }
        }

        /* Step 3: pass1_block_exp_fp16 (compute P = exp2(S - max), fp16 output) */
        pass1_block_exp_fp16(S, ld_s, kc, klen, MR, rmax, Pp, rsum);

        /* Prefetch S data for NEXT block into L1 */
        if (kb + 1 < n_kblocks) {
            const float *base = S + (int64_t)((kb + 1) * Kc) * ld_s;
            int nlines = (Kc * ld_s * (int)sizeof(float) + 255) / 256;
            for (int line = 0; line < nlines; line++)
                __builtin_prefetch(base + line * 64, 0, 3);
        }

        /* Step 4: GEMM for this block (noepi4 needs K multiple of 4) */
        int klen_gemm = klen & ~3;
        if (klen_gemm < 4) continue;
        for (int t = 0; t < n_tiles; t++) {
            const _Float16 *Vp = Vp_all + ((size_t)kb * n_tiles + t) * Kc * NR;
            if (kb == 0)
                kern_init(Pp, Vp, O + t * NR,
                          (int64_t)klen_gemm, 0, ldo_bytes);
            else
                kern_accum(Pp, Vp, O + t * NR,
                           (int64_t)klen_gemm, 0, ldo_bytes);
        }
    }

    uint64_t t1 = rdtick();
    tm->pass1 = 0;  /* No separate max-find pass */
    tm->pass2 = (double)(t1 - t0) * tick2sec;  /* everything fused */

    /* Normalize */
    normalize_fp16(O, ld_o, rsum, MR, d_out);
    uint64_t t2 = rdtick();
    tm->norm = (double)(t2 - t1) * tick2sec;

    free(Pp);
}

/* ─── Reference (double precision) ─── */
/* S is column-major: S[j * ld_s + m] */
static void reference(const float *S, int ld_s, const float *V, double *Or,
                      int M, int L, int d)
{
    for (int m = 0; m < M; m++) {
        double mx = -1e30;
        for (int j = 0; j < L; j++) {
            double v = (double)S[(int64_t)j * ld_s + m];
            if (v > mx) mx = v;
        }
        double sm = 0;
        size_t pr_sz = ((size_t)L * sizeof(double) + 255) & ~(size_t)255;
        double *Pr = (double *)aligned_alloc(256, pr_sz);
        memset(Pr, 0, pr_sz);
        for (int j = 0; j < L; j++) {
            Pr[j] = exp2((double)S[(int64_t)j * ld_s + m] - mx);
            sm += Pr[j];
        }
        for (int j = 0; j < L; j++) Pr[j] /= sm;
        for (int j = 0; j < d; j++) {
            double a = 0;
            for (int k = 0; k < L; k++)
                a += Pr[k] * (double)V[k * d + j];
            Or[m * d + j] = a;
        }
        free(Pr);
    }
}

static void init_rand(float *b, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++)
        b[i] = lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

/* ─── Measure max error ─── */
static double max_error(const float *O, int ld_o,
                        const double *Or, int M, int d)
{
    double me = 0;
    for (int m = 0; m < M; m++)
        for (int j = 0; j < d; j++) {
            double e = fabs((double)O[m * ld_o + j] - Or[m * d + j]);
            if (e > me) me = e;
        }
    return me;
}

int main(void)
{
    int d = 256;
    int d8 = ((d + 47) / 48) * 48;  /* 288 for 8x3 (6 tiles × 48) */
    int Kc = KC_DEFAULT;
    tick2sec = 1.0 / (double)tick_freq();

    printf("=== 2-Pass Softmax + P@V  (v5: pre-packed V, 256B aligned) ===\n");
    printf("d=%d d_pad_8x3=%d Kc=%d  CPU=%.1fGHz  Peak=%.0f GFLOPS\n",
           d, d8, Kc, CPU_GHZ, PEAK_GFLOPS);
    printf("Timer: cntvct @ %.0f MHz\n\n", 1.0/tick2sec/1e6);

    /* ── Self-tests ── */
    test_fexpa();
    test_kernel("8x3", micro_kernel_fp32_8x3,
                micro_kernel_fp32_8x3_accum, 8, 48);
    test_kernel("6x4", micro_kernel_fp32_6x4_bcast,
                micro_kernel_fp32_6x4_bcast_accum, 6, 64);
    test_kernel("10x2", micro_kernel_fp32_10x2,
                micro_kernel_fp32_10x2_accum, 10, 32);
    test_kernel("11x2", micro_kernel_fp32_11x2,
                micro_kernel_fp32_11x2_accum, 11, 32);
    test_kernel("12x2", micro_kernel_fp32_12x2,
                micro_kernel_fp32_12x2_accum, 12, 32);
    test_kernel("12x2swp", micro_kernel_fp32_12x2_swp,
                micro_kernel_fp32_12x2_swp_accum, 12, 32);
    test_kernel_fp16("h12swp", micro_kernel_fp16_12x2_swp,
                     micro_kernel_fp16_12x2_swp_accum, 12, 64);
    test_kernel_fp16("h12csp", micro_kernel_fp16_12x2_csplit,
                     micro_kernel_fp16_12x2_csplit_accum, 12, 64);
    test_kernel_fp16("h12dsp", micro_kernel_fp16_12x2_dswp,
                     micro_kernel_fp16_12x2_dswp_accum, 12, 64);
    test_kernel_fp16("h12_4k", micro_kernel_fp16_12x2_4k,
                     micro_kernel_fp16_12x2_4k_accum, 12, 64);
    test_kernel_fp16("h8x3",  micro_kernel_fp16_8x3,
                     micro_kernel_fp16_8x3_accum, 8, 96);
    test_kernel_fp16("h6x4",  micro_kernel_fp16_6x4,
                     micro_kernel_fp16_6x4_accum, 6, 128);
    test_kernel_fp16("h12bp", micro_kernel_fp16_12x2_bpre,
                     micro_kernel_fp16_12x2_bpre_accum, 12, 64);
    test_kernel_fp16_noepi("h12ne", micro_kernel_fp16_12x2_noepi,
                           micro_kernel_fp16_12x2_noepi_accum, 12, 64);
    test_kernel_fp16_noepi("h12n4", micro_kernel_fp16_12x2_noepi4,
                           micro_kernel_fp16_12x2_noepi4_accum, 12, 64);
    test_kernel_fp16_noepi("h12np", micro_kernel_fp16_12x2_noepi4_prfm,
                           micro_kernel_fp16_12x2_noepi4_prfm_accum, 12, 64);
    printf("\n");

    /* Skip fp32 benchmarks if FP16_ONLY env is set */
    if (getenv("FP16_ONLY")) goto fp16_section;

    /* ── L1 cache padding diagnostics ── */
    printf("L1 ring-buffer slot padding (64 sets, 256B line, 4-way):\n");
    {
        int mrs[] = {8, 6, 10, 11, 12};
        const char *names[] = {"8x3", "6x4", "10x2", "11x2", "12x2"};
        for (int i = 0; i < 5; i++) {
            int pad, ov;
            int slot_lines = (Kc * mrs[i] * 4 + L1_LINE - 1) / L1_LINE;
            int shift_nopad = (PIPE_AHEAD_DEFAULT * slot_lines) % L1_SETS;
            int ov_nopad = l1_set_overlap(slot_lines, shift_nopad);
            size_t stride = optimal_slot_stride(mrs[i], Kc, PIPE_AHEAD_DEFAULT, &pad, &ov);
            printf("  %-5s: slot=%2d lines, pad=%2d → stride=%2zu lines, "
                   "overlap %d/%d → %d/%d\n",
                   names[i], slot_lines, pad, stride / L1_LINE,
                   ov_nopad, slot_lines, ov, slot_lines);
        }
    }
    printf("\n");

    /* ── Benchmark ── */
    printf("%-5s %6s | %8s %8s %8s %8s | %7s %6s | %7s %6s | %9s\n",
           "Kern", "L", "fill", "pipe", "nm(us)", "tot(us)",
           "totGF", "tot%", "gmmGF", "gmm%", "maxerr");
    printf("─────────────────────────────────────────────────────"
           "────────────────────────────────────\n");

    int Ls[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    int nL = sizeof(Ls)/sizeof(Ls[0]);
    int M_max = 12;  /* max MR across all kernels */

    /* Kernel table: name, MR, NR, d_out, init_fn, accum_fn */
    struct kern_cfg {
        const char *name;
        int MR, NR, d_out;
        gemm_fn init, accum;
    } kerns[] = {
        {"8x3",  8,  48, d8, micro_kernel_fp32_8x3,       micro_kernel_fp32_8x3_accum},
        {"6x4",  6,  64, d,  micro_kernel_fp32_6x4_bcast,  micro_kernel_fp32_6x4_bcast_accum},
        {"10x2", 10, 32, d,  micro_kernel_fp32_10x2,       micro_kernel_fp32_10x2_accum},
        {"11x2", 11, 32, d,  micro_kernel_fp32_11x2,       micro_kernel_fp32_11x2_accum},
        {"12x2", 12, 32, d,  micro_kernel_fp32_12x2,       micro_kernel_fp32_12x2_accum},
        {"12swp", 12, 32, d, micro_kernel_fp32_12x2_swp,  micro_kernel_fp32_12x2_swp_accum},
    };
    int nkerns = sizeof(kerns)/sizeof(kerns[0]);

    for (int il = 0; il < nL; il++) {
        int L = Ls[il];

        srand(42);
        size_t sz_S = ((size_t)M_max * L * 4 + 255) & ~(size_t)255;
        size_t sz_V = ((size_t)L * d * 4 + 255) & ~(size_t)255;

        float *S = aligned_alloc(256, sz_S);
        float *V = aligned_alloc(256, sz_V);
        init_rand(S, (size_t)M_max * L, -2.0f, 2.0f);
        init_rand(V, (size_t)L * d, -1.0f, 1.0f);

        for (int ik = 0; ik < nkerns; ik++) {
            struct kern_cfg *k = &kerns[ik];
            int MR = k->MR, NR = k->NR, d_out = k->d_out;

            size_t sz_O = ((size_t)MR * d_out * 4 + 255) & ~(size_t)255;
            size_t sz_R = ((size_t)MR * d * 8 + 255) & ~(size_t)255;
            float  *O  = aligned_alloc(256, sz_O);
            double *Or = aligned_alloc(256, sz_R);
            memset(Or, 0, sz_R);
            reference(S, M_max, V, Or, MR, L, d);

            int nkb, nt;
            float *Vp = prepack_v(V, d, L, d_out, NR, Kc, &nkb, &nt);

            timing_t best = {1e30, 1e30, 1e30};
            int nrep = (L <= 1024) ? 11 : 5;
            int warmup = (L <= 1024) ? 3 : 1;

            float *Pp = aligned_alloc(256, ((size_t)L * MR * 4 + 255) & ~(size_t)255);

            for (int r = 0; r < nrep; r++) {
                timing_t cur;
                memset(O, 0, sz_O);
                run_2pass(S, M_max, Vp, O, d_out, MR, L, d_out, NR, Kc, nt,
                          PIPE_AHEAD_DEFAULT, k->init, k->accum, &cur);
                if (r >= warmup) {
                    double tot = cur.pass1 + cur.pass2 + cur.norm;
                    double btot = best.pass1 + best.pass2 + best.norm;
                    if (tot < btot) best = cur;
                }
            }
            double me = max_error(O, d_out, Or, MR, d);

            {
                float rmax_tmp[16], rsum_tmp[16];
                pass1_softmax(S, M_max, Pp, MR, L, rmax_tmp, rsum_tmp);
            }
            double gemm_sec = bench_gemm_only(Pp, Vp, O, d_out,
                    MR, L, d_out, NR, Kc, nt, k->init, k->accum);

            double total_s = best.pass1 + best.pass2 + best.norm;
            double flops = 2.0 * MR * (double)L * d_out;
            double gf = flops / total_s / 1e9;
            double gf_gmm = flops / gemm_sec / 1e9;

            printf("%-5s %6d | %8.1f %8.1f %8.1f %8.1f | %7.1f %5.1f%% | %7.1f %5.1f%% | %.2e\n",
                   k->name, L, best.pass1*1e6, best.pass2*1e6, best.norm*1e6,
                   total_s*1e6, gf, gf/PEAK_GFLOPS*100,
                   gf_gmm, gf_gmm/PEAK_GFLOPS*100, me);

            free(Pp); free(Vp); free(O); free(Or);
        }
        if (il < nL - 1)
            printf("─────────────────────────────────────────────────────"
                   "────────────────────────────────────\n");

        free(S); free(V);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * Parameter sweep: Kc × PIPE_AHEAD for 12×2 and 12×2_swp kernels
     * ═══════════════════════════════════════════════════════════════════ */
    printf("\n=== Parameter Sweep: Kc × PIPE_AHEAD (12×2 kernels, d=%d) ===\n", d);
    {
        struct sweep_kern {
            const char *name;
            int MR, NR, d_out;
            gemm_fn init, accum;
        } skerns[] = {
            {"12x2",     12, 32, d, micro_kernel_fp32_12x2,     micro_kernel_fp32_12x2_accum},
            {"12x2swp",  12, 32, d, micro_kernel_fp32_12x2_swp, micro_kernel_fp32_12x2_swp_accum},
        };
        int n_skerns = sizeof(skerns)/sizeof(skerns[0]);
        int sweep_Kc[] = {128, 192, 256, 320};
        int n_Kc = sizeof(sweep_Kc)/sizeof(sweep_Kc[0]);
        int sweep_pa[] = {2, 3, 4, 6};
        int n_pa = sizeof(sweep_pa)/sizeof(sweep_pa[0]);
        int sweep_L[] = {4096, 32768};
        int n_sL = sizeof(sweep_L)/sizeof(sweep_L[0]);

        printf("%-8s %6s %4s %3s | %8s %8s %8s | %7s %6s | %7s %6s | L1fit\n",
               "Kern", "L", "Kc", "PA", "fill", "pipe", "tot(us)",
               "totGF", "tot%", "gmmGF", "gmm%");
        printf("────────────────────────────────────────────────────────"
               "────────────────────────────────────\n");

        for (int isL = 0; isL < n_sL; isL++) {
            int L = sweep_L[isL];
            srand(42);
            size_t sz_S = ((size_t)M_max * L * 4 + 255) & ~(size_t)255;
            size_t sz_V = ((size_t)L * d * 4 + 255) & ~(size_t)255;
            float *S = aligned_alloc(256, sz_S);
            float *V = aligned_alloc(256, sz_V);
            init_rand(S, (size_t)M_max * L, -2.0f, 2.0f);
            init_rand(V, (size_t)L * d, -1.0f, 1.0f);

            for (int isk = 0; isk < n_skerns; isk++) {
                struct sweep_kern *sk = &skerns[isk];
                int MR = sk->MR, NR = sk->NR, d_out = sk->d_out;

                for (int iKc = 0; iKc < n_Kc; iKc++) {
                    int Kc_s = sweep_Kc[iKc];
                    /* Check L1 fit: A + B + C must fit in 64KB */
                    int l1_kb = (MR * Kc_s * 4 + NR * Kc_s * 4 + MR * NR * 4) / 1024;
                    if (l1_kb > 64) continue; /* skip if doesn't fit */

                    int nkb, nt;
                    float *Vp = prepack_v(V, d, L, d_out, NR, Kc_s, &nkb, &nt);

                    size_t sz_O = ((size_t)MR * d_out * 4 + 255) & ~(size_t)255;
                    float *O = aligned_alloc(256, sz_O);

                    for (int ipa = 0; ipa < n_pa; ipa++) {
                        int pa = sweep_pa[ipa];

                        timing_t best = {1e30, 1e30, 1e30};
                        int nrep = (L <= 4096) ? 11 : 5;
                        int warmup = (L <= 4096) ? 3 : 1;

                        for (int r = 0; r < nrep; r++) {
                            timing_t cur;
                            memset(O, 0, sz_O);
                            run_2pass(S, M_max, Vp, O, d_out, MR, L, d_out,
                                      NR, Kc_s, nt, pa,
                                      sk->init, sk->accum, &cur);
                            if (r >= warmup) {
                                double tot = cur.pass1 + cur.pass2 + cur.norm;
                                double btot = best.pass1 + best.pass2 + best.norm;
                                if (tot < btot) best = cur;
                            }
                        }

                        /* GEMM-only ceiling */
                        float *Pp = aligned_alloc(256, ((size_t)L * MR * 4 + 255) & ~(size_t)255);
                        {
                            float rmax_tmp[16], rsum_tmp[16];
                            pass1_softmax(S, M_max, Pp, MR, L, rmax_tmp, rsum_tmp);
                        }
                        double gemm_sec = bench_gemm_only(Pp, Vp, O, d_out,
                                MR, L, d_out, NR, Kc_s, nt, sk->init, sk->accum);
                        free(Pp);

                        double total_s = best.pass1 + best.pass2 + best.norm;
                        double flops = 2.0 * MR * (double)L * d_out;
                        double gf = flops / total_s / 1e9;
                        double gf_gmm = flops / gemm_sec / 1e9;

                        printf("%-8s %6d %4d %3d | %8.1f %8.1f %8.1f | %7.1f %5.1f%% | %7.1f %5.1f%% | %2dKB\n",
                               sk->name, L, Kc_s, pa,
                               best.pass1*1e6, best.pass2*1e6, total_s*1e6,
                               gf, gf/PEAK_GFLOPS*100,
                               gf_gmm, gf_gmm/PEAK_GFLOPS*100, l1_kb);
                    }
                    free(Vp); free(O);
                }
            }
            if (isL < n_sL - 1)
                printf("────────────────────────────────────────────────────────"
                       "────────────────────────────────────\n");
            free(S); free(V);
        }
    }

fp16_section:
    /* ═══════════════════════════════════════════════════════════════════
     * FP16 Mixed-Precision Benchmark
     * exp2 in fp32 → convert P to fp16 → FMLA in fp16 → accumulate fp32
     * ═══════════════════════════════════════════════════════════════════ */
    /* Enable FZ16 (flush fp16 denormals to zero) to avoid ~100cy/denorm traps.
     * Softmax outputs at large L are often < 6.1e-5 (fp16 min normal),
     * which would otherwise cause catastrophic slowdown in fp16 FMLA. */
    {
        uint64_t fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        fpcr |= (1UL << 19);  /* FZ16 bit */
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    }

    printf("\n=== FP16 Mixed Precision Sweep (exp2:fp32, FMLA:fp16, accum:fp32, d=%d, FZ16=1) ===\n", d);
    printf("Peak fp16 = %.0f GFLOPS  (fp32 peak = %.0f GFLOPS)\n\n",
           PEAK_GFLOPS_FP16, PEAK_GFLOPS);
    {
        struct fp16_kern {
            const char *name;
            gemm_fp16_fn init, accum;
            int MR, NR;
        } fp16_kerns[] = {
            {"h12sw", micro_kernel_fp16_12x2_swp,  micro_kernel_fp16_12x2_swp_accum, 12, 64},
            {"h12bp", micro_kernel_fp16_12x2_bpre, micro_kernel_fp16_12x2_bpre_accum, 12, 64},
        };
        int n_fp16_kerns = sizeof(fp16_kerns)/sizeof(fp16_kerns[0]);

        int Kc_vals[] = {128, 192, 256, 384};
        int n_Kc = sizeof(Kc_vals)/sizeof(Kc_vals[0]);

        /* Focus on large L for Kc sweep */
        int fp16_Ls[] = {4096, 8192, 16384, 32768};
        int n_fp16_L = sizeof(fp16_Ls)/sizeof(fp16_Ls[0]);

        printf("%-5s %3s %6s | %8s %8s %8s %8s | %7s %6s | %7s %6s | %9s\n",
               "Kern", "Kc", "L", "fill", "pipe", "nm(us)", "tot(us)",
               "totGF", "tot%", "gmmGF", "gmm%", "maxerr");
        printf("──────────────────────────────────────────────────────────"
               "────────────────────────────────────\n");

        for (int ik = 0; ik < n_fp16_kerns; ik++) {
            struct fp16_kern *fk = &fp16_kerns[ik];
            int MRk = fk->MR, NRk = fk->NR;
            int d_outk = ((d + NRk - 1) / NRk) * NRk;

            for (int iKc = 0; iKc < n_Kc; iKc++) {
                int Kc16 = Kc_vals[iKc];

                for (int il = 0; il < n_fp16_L; il++) {
                    int L = fp16_Ls[il];
                    if (L < Kc16 * 2) continue;

                    srand(42);
                    size_t sz_S = ((size_t)M_max * L * 4 + 255) & ~(size_t)255;
                    size_t sz_V = ((size_t)L * d * 4 + 255) & ~(size_t)255;
                    float *S = aligned_alloc(256, sz_S);
                    float *V = aligned_alloc(256, sz_V);
                    init_rand(S, (size_t)M_max * L, -2.0f, 2.0f);
                    init_rand(V, (size_t)L * d, -1.0f, 1.0f);

                    size_t sz_O = ((size_t)MRk * d_outk * 4 + 255) & ~(size_t)255;
                    size_t sz_R = ((size_t)MRk * d * 8 + 255) & ~(size_t)255;
                    float  *O  = aligned_alloc(256, sz_O);
                    double *Or = aligned_alloc(256, sz_R);
                    memset(Or, 0, sz_R);
                    reference(S, M_max, V, Or, MRk, L, d);

                    int nkb, nt;
                    _Float16 *Vp16 = prepack_v_fp16(V, d, L, d_outk, NRk, Kc16, &nkb, &nt);

                    timing_t best = {1e30, 1e30, 1e30};
                    int nrep = (L <= 4096) ? 11 : 5;
                    int warmup = (L <= 4096) ? 3 : 1;

                    for (int r = 0; r < nrep; r++) {
                        timing_t cur;
                        memset(O, 0, sz_O);
                        run_2pass_fp16(S, M_max, Vp16, O, d_outk, MRk, L, d_outk,
                                       NRk, Kc16, nt, PIPE_AHEAD_DEFAULT,
                                       fk->init, fk->accum, &cur);
                        if (r >= warmup) {
                            double tot = cur.pass1 + cur.pass2 + cur.norm;
                            double btot = best.pass1 + best.pass2 + best.norm;
                            if (tot < btot) best = cur;
                        }
                    }
                    double me = max_error(O, d_outk, Or, MRk, d);

                    /* GEMM-only measurement */
                    float *Pp32 = aligned_alloc(256, ((size_t)L * MRk * 4 + 255) & ~(size_t)255);
                    {
                        float rmax_tmp[16], rsum_tmp[16];
                        pass1_softmax(S, M_max, Pp32, MRk, L, rmax_tmp, rsum_tmp);
                    }
                    size_t pp16_sz = ((size_t)L * MRk * 2 + 255) & ~(size_t)255;
                    _Float16 *Pp16 = (_Float16 *)aligned_alloc(256, pp16_sz);
                    for (size_t i = 0; i < (size_t)L * MRk; i++)
                        Pp16[i] = (_Float16)Pp32[i];

                    double gemm_sec = bench_gemm_only_fp16(Pp16, Vp16, O, d_outk,
                            MRk, L, d_outk, NRk, Kc16, nt,
                            fk->init, fk->accum);

                    double total_s = best.pass1 + best.pass2 + best.norm;
                    double flops = 2.0 * MRk * (double)L * d_outk;
                    double gf = flops / total_s / 1e9;
                    double gf_gmm = flops / gemm_sec / 1e9;

                    printf("%-5s %3d %6d | %8.1f %8.1f %8.1f %8.1f | %7.1f %5.1f%% | %7.1f %5.1f%% | %.2e\n",
                           fk->name, Kc16, L,
                           best.pass1*1e6, best.pass2*1e6, best.norm*1e6,
                           total_s*1e6, gf, gf/PEAK_GFLOPS_FP16*100,
                           gf_gmm, gf_gmm/PEAK_GFLOPS_FP16*100, me);

                    free(Pp32); free(Pp16); free(Vp16); free(O); free(Or);
                    free(S); free(V);
                }
            }
            printf("──────────────────────────────────────────────────────────"
                   "────────────────────────────────────\n");
        }
    }

    /* ═══════════════════════════════════════════════════════════════════
     * FP16 NOEPI: fp16 C accumulation (no fp32 conversion epilogue)
     * exp2 in fp32 → convert P to fp16 → FMLA in fp16 → accumulate fp16
     * ═══════════════════════════════════════════════════════════════════ */
    printf("\n=== FP16 NOEPI (fp16 C accum, d=%d, FZ16=1) ===\n", d);
    printf("Eliminates fp16→fp32 conversion epilogue; O accumulated in fp16.\n");
    printf("Peak fp16 = %.0f GFLOPS\n\n", PEAK_GFLOPS_FP16);
    {
        struct fp16_noepi_kern {
            const char *name;
            gemm_fp16_noepi_fn init, accum;
            /* Also keep the fp32-C variant for side-by-side comparison */
            gemm_fp16_fn init32, accum32;
            int MR, NR;
        } noepi_kerns[] = {
            {"h12ne", micro_kernel_fp16_12x2_noepi,  micro_kernel_fp16_12x2_noepi_accum,
                      micro_kernel_fp16_12x2_swp,    micro_kernel_fp16_12x2_swp_accum, 12, 64},
            {"h12n4", micro_kernel_fp16_12x2_noepi4, micro_kernel_fp16_12x2_noepi4_accum,
                      micro_kernel_fp16_12x2_swp,    micro_kernel_fp16_12x2_swp_accum, 12, 64},
            {"h12np", micro_kernel_fp16_12x2_noepi4_prfm, micro_kernel_fp16_12x2_noepi4_prfm_accum,
                      micro_kernel_fp16_12x2_swp,    micro_kernel_fp16_12x2_swp_accum, 12, 64},
        };
        int n_noepi = sizeof(noepi_kerns)/sizeof(noepi_kerns[0]);

        int Kc_vals[] = {192, 256, 384};
        int n_Kc = sizeof(Kc_vals)/sizeof(Kc_vals[0]);

        int noepi_Ls[] = {4096, 8192, 16384, 32768};
        int n_noepi_L = sizeof(noepi_Ls)/sizeof(noepi_Ls[0]);

        printf("%-6s %3s %6s | %8s %8s %8s %8s | %7s %6s | %7s %6s | %9s\n",
               "Kern", "Kc", "L", "fill", "pipe", "nm(us)", "tot(us)",
               "totGF", "tot%", "gmmGF", "gmm%", "maxerr");
        printf("───────────────────────────────────────────────────────────"
               "────────────────────────────────────\n");

        for (int ik = 0; ik < n_noepi; ik++) {
            struct fp16_noepi_kern *fk = &noepi_kerns[ik];
            int MRk = fk->MR, NRk = fk->NR;
            int d_outk = ((d + NRk - 1) / NRk) * NRk;

            for (int iKc = 0; iKc < n_Kc; iKc++) {
                int Kc16 = Kc_vals[iKc];

                for (int il = 0; il < n_noepi_L; il++) {
                    int L = noepi_Ls[il];
                    if (L < Kc16 * 2) continue;

                    srand(42);
                    size_t sz_S = ((size_t)M_max * L * 4 + 255) & ~(size_t)255;
                    size_t sz_V = ((size_t)L * d * 4 + 255) & ~(size_t)255;
                    float *S = aligned_alloc(256, sz_S);
                    float *V = aligned_alloc(256, sz_V);
                    init_rand(S, (size_t)M_max * L, -2.0f, 2.0f);
                    init_rand(V, (size_t)L * d, -1.0f, 1.0f);

                    size_t sz_O16 = ((size_t)MRk * d_outk * sizeof(_Float16) + 255) & ~(size_t)255;
                    size_t sz_O32 = ((size_t)MRk * d_outk * sizeof(float) + 255) & ~(size_t)255;
                    size_t sz_R = ((size_t)MRk * d * 8 + 255) & ~(size_t)255;
                    _Float16 *O16 = (_Float16 *)aligned_alloc(256, sz_O16);
                    float    *O32 = (float *)aligned_alloc(256, sz_O32);
                    double   *Or  = (double *)aligned_alloc(256, sz_R);
                    memset(Or, 0, sz_R);
                    reference(S, M_max, V, Or, MRk, L, d);

                    int nkb, nt;
                    _Float16 *Vp16 = prepack_v_fp16(V, d, L, d_outk, NRk, Kc16, &nkb, &nt);

                    /* ── NOEPI (fp16 C) pipeline ── */
                    timing_t best_ne = {1e30, 1e30, 1e30};
                    int nrep = (L <= 4096) ? 11 : 5;
                    int warmup = (L <= 4096) ? 3 : 1;

                    for (int r = 0; r < nrep; r++) {
                        timing_t cur;
                        memset(O16, 0, sz_O16);
                        run_2pass_fp16_noepi(S, M_max, Vp16, O16, d_outk, MRk, L, d_outk,
                                              NRk, Kc16, nt, PIPE_AHEAD_DEFAULT,
                                              fk->init, fk->accum, &cur);
                        if (r >= warmup) {
                            double tot = cur.pass1 + cur.pass2 + cur.norm;
                            double btot = best_ne.pass1 + best_ne.pass2 + best_ne.norm;
                            if (tot < btot) best_ne = cur;
                        }
                    }
                    double me_ne = max_error_fp16(O16, d_outk, Or, MRk, d);

                    /* NOEPI GEMM-only */
                    size_t pp16_sz = ((size_t)L * MRk * sizeof(_Float16) + 255) & ~(size_t)255;
                    _Float16 *Pp16 = (_Float16 *)aligned_alloc(256, pp16_sz);
                    {
                        float *Pp32_tmp = aligned_alloc(256, ((size_t)L * MRk * 4 + 255) & ~(size_t)255);
                        float rmax_tmp[16], rsum_tmp[16];
                        pass1_softmax(S, M_max, Pp32_tmp, MRk, L, rmax_tmp, rsum_tmp);
                        for (size_t i = 0; i < (size_t)L * MRk; i++)
                            Pp16[i] = (_Float16)Pp32_tmp[i];
                        free(Pp32_tmp);
                    }
                    double gemm_ne = bench_gemm_only_fp16_noepi(Pp16, Vp16, O16, d_outk,
                            MRk, L, d_outk, NRk, Kc16, nt, fk->init, fk->accum);

                    double total_ne = best_ne.pass1 + best_ne.pass2 + best_ne.norm;
                    double flops = 2.0 * MRk * (double)L * d_outk;
                    double gf_ne = flops / total_ne / 1e9;
                    double gf_gmm_ne = flops / gemm_ne / 1e9;

                    printf("%-6s %3d %6d | %8.1f %8.1f %8.1f %8.1f | %7.1f %5.1f%% | %7.1f %5.1f%% | %.2e\n",
                           fk->name, Kc16, L,
                           best_ne.pass1*1e6, best_ne.pass2*1e6, best_ne.norm*1e6,
                           total_ne*1e6, gf_ne, gf_ne/PEAK_GFLOPS_FP16*100,
                           gf_gmm_ne, gf_gmm_ne/PEAK_GFLOPS_FP16*100, me_ne);

                    /* ── 3-pass separated: max+pass1 → GEMM (no L1 pollution) ── */
                    {
                        timing_t best_3p = {1e30, 1e30, 1e30};
                        for (int r = 0; r < nrep; r++) {
                            timing_t cur;
                            memset(O16, 0, sz_O16);
                            run_3pass_fp16_noepi(S, M_max, Vp16, O16, d_outk, MRk, L, d_outk,
                                                  NRk, Kc16, nt,
                                                  fk->init, fk->accum, &cur);
                            if (r >= warmup) {
                                double tot = cur.pass1 + cur.pass2 + cur.norm;
                                double btot = best_3p.pass1 + best_3p.pass2 + best_3p.norm;
                                if (tot < btot) best_3p = cur;
                            }
                        }
                        double me_3p = max_error_fp16(O16, d_outk, Or, MRk, d);
                        double total_3p = best_3p.pass1 + best_3p.pass2 + best_3p.norm;
                        double gf_3p = flops / total_3p / 1e9;

                        printf("3p_%-3s %3d %6d | %8.1f %8.1f %8.1f %8.1f | %7.1f %5.1f%% | %7.1f %5.1f%% | %.2e  (3-pass)\n",
                               fk->name + 3, Kc16, L,
                               best_3p.pass1*1e6, best_3p.pass2*1e6, best_3p.norm*1e6,
                               total_3p*1e6, gf_3p, gf_3p/PEAK_GFLOPS_FP16*100,
                               gf_gmm_ne, gf_gmm_ne/PEAK_GFLOPS_FP16*100, me_3p);
                    }

                    /* ── Online softmax NOEPI (no max-find pass) ── */
                    {
                        timing_t best_onl = {1e30, 1e30, 1e30};
                        for (int r = 0; r < nrep; r++) {
                            timing_t cur;
                            memset(O16, 0, sz_O16);
                            run_online_fp16_noepi(S, M_max, Vp16, O16, d_outk, MRk, L, d_outk,
                                                   NRk, Kc16, nt,
                                                   fk->init, fk->accum, &cur);
                            if (r >= warmup) {
                                double tot = cur.pass1 + cur.pass2 + cur.norm;
                                double btot = best_onl.pass1 + best_onl.pass2 + best_onl.norm;
                                if (tot < btot) best_onl = cur;
                            }
                        }
                        double me_onl = max_error_fp16(O16, d_outk, Or, MRk, d);
                        double total_onl = best_onl.pass1 + best_onl.pass2 + best_onl.norm;
                        double gf_onl = flops / total_onl / 1e9;

                        printf("onl_%-2s %3d %6d | %8.1f %8.1f %8.1f %8.1f | %7.1f %5.1f%% | %7.1f %5.1f%% | %.2e  (online)\n",
                               fk->name + 3, Kc16, L,
                               best_onl.pass1*1e6, best_onl.pass2*1e6, best_onl.norm*1e6,
                               total_onl*1e6, gf_onl, gf_onl/PEAK_GFLOPS_FP16*100,
                               gf_gmm_ne, gf_gmm_ne/PEAK_GFLOPS_FP16*100, me_onl);
                    }

                    /* ── fp32-C SWP reference for comparison ── */
                    {
                        timing_t best_sw = {1e30, 1e30, 1e30};
                        for (int r = 0; r < nrep; r++) {
                            timing_t cur;
                            memset(O32, 0, sz_O32);
                            run_2pass_fp16(S, M_max, Vp16, O32, d_outk, MRk, L, d_outk,
                                           NRk, Kc16, nt, PIPE_AHEAD_DEFAULT,
                                           fk->init32, fk->accum32, &cur);
                            if (r >= warmup) {
                                double tot = cur.pass1 + cur.pass2 + cur.norm;
                                double btot = best_sw.pass1 + best_sw.pass2 + best_sw.norm;
                                if (tot < btot) best_sw = cur;
                            }
                        }
                        double me_sw = max_error(O32, d_outk, Or, MRk, d);
                        double total_sw = best_sw.pass1 + best_sw.pass2 + best_sw.norm;
                        double gf_sw = flops / total_sw / 1e9;

                        /* fp32-C GEMM-only */
                        _Float16 *Pp16_sw = (_Float16 *)aligned_alloc(256, pp16_sz);
                        memcpy(Pp16_sw, Pp16, pp16_sz);
                        double gemm_sw = bench_gemm_only_fp16(Pp16_sw, Vp16, O32, d_outk,
                                MRk, L, d_outk, NRk, Kc16, nt,
                                fk->init32, fk->accum32);
                        double gf_gmm_sw = flops / gemm_sw / 1e9;

                        printf("%-6s %3d %6d | %8.1f %8.1f %8.1f %8.1f | %7.1f %5.1f%% | %7.1f %5.1f%% | %.2e  (fp32-C ref)\n",
                               "h12sw", Kc16, L,
                               best_sw.pass1*1e6, best_sw.pass2*1e6, best_sw.norm*1e6,
                               total_sw*1e6, gf_sw, gf_sw/PEAK_GFLOPS_FP16*100,
                               gf_gmm_sw, gf_gmm_sw/PEAK_GFLOPS_FP16*100, me_sw);
                        free(Pp16_sw);
                    }

                    free(Pp16); free(Vp16); free(O16); free(O32); free(Or);
                    free(S); free(V);
                }
            }
            printf("───────────────────────────────────────────────────────────"
                   "────────────────────────────────────\n");
        }
    }

    return 0;
}
