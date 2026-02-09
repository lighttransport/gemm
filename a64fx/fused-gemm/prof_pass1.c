/*
 * prof_pass1.c — fapp profiling of pass1 softmax phases for 12×2 kernel
 *
 * Profiles separately:
 *   1. max_find: streaming max reduction over S
 *   2. exp_scatter: FEXPA exp2 + scatter-store to col-major Pp
 *   3. exp_contig: alternative — FEXPA exp2 + contiguous-store (row-major)
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *       -o prof_pass1 prof_pass1.c -lfjprof -lm
 * Run:
 *   fapp -C -d ./rep_p1_paN -Hevent=paN ./prof_pass1
 *   fapppx -A -d ./rep_p1_paN -Icpupa -tcsv -o p1_paN.csv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <arm_sve.h>
/* fapp profiler API (avoid including full header to not conflict with arm_sve.h) */
extern void fapp_start(const char *, int, int);
extern void fapp_stop(const char *, int, int);

#define FEXPA_SHIFT_U32 0x48481fc0u
#define MR      12
#define KC      256

static inline uint64_t rdtick(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t tick_freq(void) {
    uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}

static inline svfloat32_t sve_fexpa(svfloat32_t z) {
    svfloat32_t r;
    __asm__("fexpa %0.s, %1.s" : "=w"(r) : "w"(z));
    return r;
}

/* ── Phase 1: find row max ── */
static void phase_max_find(const float *S, int ld_s, int L, float *rmax)
{
    const int vl = svcntw();
    for (int m = 0; m < MR; m++) {
        const float *sr = S + (int64_t)m * ld_s;
        svfloat32_t vm = svdup_f32(-1e30f);
        int j;
        for (j = 0; j + vl <= L; j += vl)
            vm = svmax_m(svptrue_b32(), vm, svld1(svptrue_b32(), sr + j));
        if (j < L) {
            svbool_t pg = svwhilelt_b32(j, L);
            vm = svmax_m(pg, vm, svld1(pg, sr + j));
        }
        rmax[m] = svmaxv(svptrue_b32(), vm);
    }
}

/* ── Phase 2a: exp2 + scatter store to col-major Pp[klen][MR] ── */
static void phase_exp_scatter(
    const float *S, int ld_s, int kc, int klen,
    const float *rmax, float *Pp, float *rsum)
{
    const int vl = svcntw();
    union { uint32_t u; float f; } su = { .u = FEXPA_SHIFT_U32 };
    float shift = su.f;
    svint32_t idx = svindex_s32(0, MR);

    for (int m = 0; m < MR; m++) {
        const float *sr = S + (int64_t)m * ld_s + kc;
        float bias = -rmax[m] + shift;
        svfloat32_t vb = svdup_f32(bias);
        svfloat32_t vs = svdup_f32(0.0f);
        int j;
        for (j = 0; j + vl <= klen; j += vl) {
            svbool_t pg = svptrue_b32();
            svfloat32_t p = sve_fexpa(svadd_m(pg, svld1(pg, sr + j), vb));
            vs = svadd_m(pg, vs, p);
            svst1_scatter_s32index_f32(pg, Pp + j * MR + m, idx, p);
        }
        if (j < klen) {
            svbool_t pg = svwhilelt_b32(j, klen);
            svfloat32_t p = sve_fexpa(svadd_m(pg, svld1(pg, sr + j), vb));
            vs = svadd_m(pg, vs, p);
            svst1_scatter_s32index_f32(pg, Pp + j * MR + m, idx, p);
        }
        rsum[m] += svaddv(svptrue_b32(), vs);
    }
}

/* ── Phase 2b: exp2 + contiguous store to row-major Pp_row[MR][klen] ── */
static void phase_exp_contig(
    const float *S, int ld_s, int kc, int klen,
    const float *rmax, float *Pp_row, float *rsum)
{
    const int vl = svcntw();
    union { uint32_t u; float f; } su = { .u = FEXPA_SHIFT_U32 };
    float shift = su.f;

    for (int m = 0; m < MR; m++) {
        const float *sr = S + (int64_t)m * ld_s + kc;
        float *dst = Pp_row + m * klen;
        float bias = -rmax[m] + shift;
        svfloat32_t vb = svdup_f32(bias);
        svfloat32_t vs = svdup_f32(0.0f);
        int j;
        for (j = 0; j + vl <= klen; j += vl) {
            svbool_t pg = svptrue_b32();
            svfloat32_t p = sve_fexpa(svadd_m(pg, svld1(pg, sr + j), vb));
            vs = svadd_m(pg, vs, p);
            svst1(pg, dst + j, p);
        }
        if (j < klen) {
            svbool_t pg = svwhilelt_b32(j, klen);
            svfloat32_t p = sve_fexpa(svadd_m(pg, svld1(pg, sr + j), vb));
            vs = svadd_m(pg, vs, p);
            svst1(pg, dst + j, p);
        }
        rsum[m] += svaddv(svptrue_b32(), vs);
    }
}

/* ── Phase 2c: exp2 + scatter, vectorized across MR rows (gather S, contig store) ── */
static void phase_exp_gather(
    const float *S, int ld_s, int kc, int klen,
    const float *rmax, float *Pp, float *rsum)
{
    union { uint32_t u; float f; } su = { .u = FEXPA_SHIFT_U32 };
    float shift = su.f;

    /* Build bias vector and gather offsets for MR rows */
    svbool_t pg_mr = svwhilelt_b32(0, MR);
    float bias_arr[16] __attribute__((aligned(64)));
    int64_t off_arr[16] __attribute__((aligned(64)));
    for (int m = 0; m < MR; m++) {
        bias_arr[m] = -rmax[m] + shift;
        off_arr[m] = (int64_t)m * ld_s * (int64_t)sizeof(float);
    }
    svfloat32_t vbias = svld1(pg_mr, bias_arr);
    svfloat32_t vsum = svdup_f32(0.0f);

    /* Base pointer for column kc */
    const char *S_base = (const char *)(S + kc);

    /* Process one column at a time: gather MR rows, contiguous store MR elements */
    svint64_t offsets_lo = svld1_s64(svwhilelt_b64(0, (MR < 8 ? MR : 8)), off_arr);
    svint64_t offsets_hi = svld1_s64(svwhilelt_b64(0, (MR > 8 ? MR - 8 : 0)), off_arr + 8);

    for (int j = 0; j < klen; j++) {
        /* Gather S[0*ld_s+kc+j], S[1*ld_s+kc+j], ..., S[(MR-1)*ld_s+kc+j] */
        const char *col_base = S_base + j * sizeof(float);
        /* Manual gather: scalar loads */
        float tmp[16] __attribute__((aligned(64)));
        for (int m = 0; m < MR; m++)
            tmp[m] = S[(int64_t)m * ld_s + kc + j];
        svfloat32_t s_vals = svld1(pg_mr, tmp);

        svfloat32_t p = sve_fexpa(svadd_m(pg_mr, s_vals, vbias));
        vsum = svadd_m(pg_mr, vsum, p);

        /* Contiguous store: Pp[j*MR + 0..MR-1] */
        svst1(pg_mr, Pp + j * MR, p);
    }

    /* Extract partial sums */
    float sum_arr[16] __attribute__((aligned(64)));
    svst1(pg_mr, sum_arr, vsum);
    for (int m = 0; m < MR; m++)
        rsum[m] += sum_arr[m];
}

int main(void)
{
    int L = 8192;
    int n_kblocks = L / KC;
    volatile double tick2sec = 1.0 / (double)tick_freq();

    printf("prof_pass1: MR=%d L=%d Kc=%d n_kblocks=%d\n", MR, L, KC, n_kblocks);

    /* Allocate */
    size_t sz_S = ((size_t)MR * L * 4 + 255) & ~(size_t)255;
    size_t sz_Pp = ((size_t)L * MR * 4 + 255) & ~(size_t)255;
    float *S     = aligned_alloc(256, sz_S);
    float *Pp    = aligned_alloc(256, sz_Pp);
    float *Pp_row = aligned_alloc(256, sz_Pp);

    srand(42);
    for (size_t i = 0; i < (size_t)MR * L; i++)
        S[i] = -2.0f + 4.0f * ((float)rand() / (float)RAND_MAX);

    float rmax[16], rsum[16];

    /* Warmup */
    phase_max_find(S, L, L, rmax);

    int nrep = 500;

    /* ── Section 1: max_find ── */
    fapp_start("max_find", 1, 0);
    for (int r = 0; r < nrep; r++) {
        phase_max_find(S, L, L, rmax);
    }
    fapp_stop("max_find", 1, 0);

    /* ── Section 2: exp2 + scatter (current implementation) ── */
    fapp_start("exp_scatter", 2, 0);
    for (int r = 0; r < nrep; r++) {
        memset(rsum, 0, sizeof(rsum));
        for (int kb = 0; kb < n_kblocks; kb++)
            phase_exp_scatter(S, L, kb * KC, KC, rmax,
                              Pp + kb * KC * MR, rsum);
    }
    fapp_stop("exp_scatter", 2, 0);

    /* ── Section 3: exp2 + contiguous store (row-major, no scatter) ── */
    fapp_start("exp_contig", 3, 0);
    for (int r = 0; r < nrep; r++) {
        memset(rsum, 0, sizeof(rsum));
        for (int kb = 0; kb < n_kblocks; kb++)
            phase_exp_contig(S, L, kb * KC, KC, rmax,
                             Pp_row + kb * KC * MR, rsum);
    }
    fapp_stop("exp_contig", 3, 0);

    /* ── Section 4: exp2 + gather/contig (vectorize across MR) ── */
    fapp_start("exp_gather", 4, 0);
    for (int r = 0; r < nrep; r++) {
        memset(rsum, 0, sizeof(rsum));
        for (int kb = 0; kb < n_kblocks; kb++)
            phase_exp_gather(S, L, kb * KC, KC, rmax,
                             Pp + kb * KC * MR, rsum);
    }
    fapp_stop("exp_gather", 4, 0);

    /* Timing reference (no fapp overhead) */
    uint64_t t0, t1;

    t0 = rdtick();
    for (int r = 0; r < nrep; r++)
        phase_max_find(S, L, L, rmax);
    t1 = rdtick();
    printf("max_find:    %8.1f us/rep\n", (double)(t1-t0)*tick2sec*1e6/nrep);

    t0 = rdtick();
    for (int r = 0; r < nrep; r++) {
        memset(rsum, 0, sizeof(rsum));
        for (int kb = 0; kb < n_kblocks; kb++)
            phase_exp_scatter(S, L, kb*KC, KC, rmax, Pp+kb*KC*MR, rsum);
    }
    t1 = rdtick();
    printf("exp_scatter: %8.1f us/rep\n", (double)(t1-t0)*tick2sec*1e6/nrep);

    t0 = rdtick();
    for (int r = 0; r < nrep; r++) {
        memset(rsum, 0, sizeof(rsum));
        for (int kb = 0; kb < n_kblocks; kb++)
            phase_exp_contig(S, L, kb*KC, KC, rmax, Pp_row+kb*KC*MR, rsum);
    }
    t1 = rdtick();
    printf("exp_contig:  %8.1f us/rep\n", (double)(t1-t0)*tick2sec*1e6/nrep);

    t0 = rdtick();
    for (int r = 0; r < nrep; r++) {
        memset(rsum, 0, sizeof(rsum));
        for (int kb = 0; kb < n_kblocks; kb++)
            phase_exp_gather(S, L, kb*KC, KC, rmax, Pp+kb*KC*MR, rsum);
    }
    t1 = rdtick();
    printf("exp_gather:  %8.1f us/rep\n", (double)(t1-t0)*tick2sec*1e6/nrep);

    free(S); free(Pp); free(Pp_row);
    return 0;
}
