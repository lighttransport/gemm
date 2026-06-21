/* mv_bench_bf16_8row.c
 *
 * Microbench for the production bf16_pv matvec (common/ggml_dequant.h
 * matvec_bf16_8row_pv), measuring achievable HBM read bandwidth.
 *
 * Placement mirrors the model (see common/transformer.h tf_cmg_pin_thread /
 * tf_bf16_pv_fill): the allowed cores are split into per-CMG teams of 12, each
 * CMG gets its OWN weight buffer pinned to that CMG's NUMA node (mbind
 * MPOL_BIND|MPOL_MF_MOVE), threads are pinned, and every thread first-touches
 * and then reads only its own CMG's slice. A single interleaved buffer with a
 * bare `omp parallel for` caps near 1/3 of node BW, so that older harness could
 * not be trusted as a ceiling reference.
 *
 * Three kernels, all over the SAME per-CMG buffers:
 *   sve_sum   : 8-accumulator pure-streaming u64 sum (the empirical HBM roof;
 *               copy of ring_attn_bench.c sve_sum). Target ~900 GB/s/node.
 *   row8_pv   : byte-exact copy of production matvec_bf16_8row_pv (variant 0).
 *   row8_pv_v1: UF2 wide-load variant -- two 16-col chunks per iteration, 16
 *               weight loads issued before any FMA, 16 accumulators combined at
 *               the tail. Widens the in-flight load front to better cover HBM
 *               latency on A64FX's limited OoO window. Selected by env
 *               TF_BF16PV_VARIANT=1 (default 0 = row8_pv).
 *   row8_lsl  : LSL reference, bit-exact correctness oracle for the pv kernels.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *        -o mv_bench_bf16_8row mv_bench_bf16_8row.c -lm
 * Run:   ./mv_bench_bf16_8row [N_groups] [K]      (no numactl: it self-pins)
 *
 * N_groups = number of 8-row groups (split across CMGs). Default N=31040
 * K=1024 -> 508 MB of weights (>> caches), streamed from HBM each rep.
 */
#define _GNU_SOURCE
#include <arm_sve.h>
#include <errno.h>
#include <sched.h>
#include <sys/mman.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>

typedef uint16_t bf16_t;

#define CMG_CORES 12

static double mono_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* NUMA node owning logical CPU `cpu` (parsed from sysfs; no libnuma). */
static int node_of_cpu(int cpu) {
    for (int nd = 0; nd < 16; nd++) {
        char path[64], buf[256];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", nd);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char *got = fgets(buf, sizeof(buf), f);
        fclose(f);
        if (!got) continue;
        char *p = buf;
        while (*p && *p != '\n') {
            int a = (int)strtol(p, &p, 10), b = a;
            if (*p == '-') { p++; b = (int)strtol(p, &p, 10); }
            if (cpu >= a && cpu <= b) return nd;
            while (*p == ',') p++;
        }
    }
    return -1;
}

static bf16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    bits += ((bits >> 16) & 1) + 0x7FFF;
    return (bf16_t)(bits >> 16);
}

#define SVE_BF16_TO_F32(pg, ptr) \
    svreinterpret_f32(svlsl_x(pg, svld1uh_u32(pg, ptr), 16))

static volatile uint64_t g_sink;  /* defeat DCE of the streaming sum */

/* 8-accumulator unrolled u64 streaming sum -- the HBM ceiling probe.
 * Copy of ring_attn_bench.c sve_sum: eight independent vector accumulators
 * expose enough in-flight loads to saturate a CMG's HBM. */
static inline uint64_t sve_sum(const uint64_t *restrict b, size_t lo, size_t hi) {
    svbool_t pg = svptrue_b64();
    uint64_t vl = svcntd();
    size_t step = vl * 8;
    svuint64_t a0=svdup_u64(0),a1=svdup_u64(0),a2=svdup_u64(0),a3=svdup_u64(0);
    svuint64_t a4=svdup_u64(0),a5=svdup_u64(0),a6=svdup_u64(0),a7=svdup_u64(0);
    for (size_t i = lo; i + step <= hi; i += step) {
        a0 = svadd_u64_x(pg, a0, svld1_u64(pg, &b[i + 0*vl]));
        a1 = svadd_u64_x(pg, a1, svld1_u64(pg, &b[i + 1*vl]));
        a2 = svadd_u64_x(pg, a2, svld1_u64(pg, &b[i + 2*vl]));
        a3 = svadd_u64_x(pg, a3, svld1_u64(pg, &b[i + 3*vl]));
        a4 = svadd_u64_x(pg, a4, svld1_u64(pg, &b[i + 4*vl]));
        a5 = svadd_u64_x(pg, a5, svld1_u64(pg, &b[i + 5*vl]));
        a6 = svadd_u64_x(pg, a6, svld1_u64(pg, &b[i + 6*vl]));
        a7 = svadd_u64_x(pg, a7, svld1_u64(pg, &b[i + 7*vl]));
    }
    a0 = svadd_u64_x(pg,a0,a1); a2 = svadd_u64_x(pg,a2,a3);
    a4 = svadd_u64_x(pg,a4,a5); a6 = svadd_u64_x(pg,a6,a7);
    a0 = svadd_u64_x(pg,a0,a2); a4 = svadd_u64_x(pg,a4,a6);
    return svaddv_u64(pg, svadd_u64_x(pg, a0, a4));
}

/* row8_lsl: copy of ggml_dequant matvec_bf16_8row (correctness oracle) */
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
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* row8_pv (variant 0): byte-exact copy of production matvec_bf16_8row_pv. */
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
        const uint16_t *ab = (const uint16_t *)pAB + 2 * i;
        const uint16_t *cd = (const uint16_t *)pCD + 2 * i;
        const uint16_t *ef = (const uint16_t *)pEF + 2 * i;
        const uint16_t *gh = (const uint16_t *)pGH + 2 * i;
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
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* row8_pv_v1 (variant 1): UF2 wide-load. Two 16-col chunks per iteration ->
 * 16 weight loads + 2 activation loads issued before the FMAs, accumulated
 * into 16 independent chains (a*,b*) combined at the tail. Wider load front to
 * cover HBM latency; second accumulator set also breaks the per-row FMA
 * dependency chain. Falls back to a single-chunk tail when K % (2*vl) != 0. */
static void row8_pv_v1(float *dst,
                       const bf16_t *pAB, const bf16_t *pCD,
                       const bf16_t *pEF, const bf16_t *pGH,
                       const float *x, int K) {
    svbool_t pg = svptrue_b32();
    svbool_t p_all_h = svptrue_b16();
    svuint16_t idx_h = svindex_u16(0, 1);
    svbool_t p_odd = svcmpne_n_u16(p_all_h,
                                    svand_n_u16_x(p_all_h, idx_h, 1), 0);
    int vl = (int)svcntw();   /* 16 */
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    svfloat32_t b0=svdup_f32(0),b1=svdup_f32(0),b2=svdup_f32(0),b3=svdup_f32(0);
    svfloat32_t b4=svdup_f32(0),b5=svdup_f32(0),b6=svdup_f32(0),b7=svdup_f32(0);
    int i = 0;
    for (; i + 2 * vl - 1 < K; i += 2 * vl) {
        /* chunk 0 at hw 2*i; chunk 1 (cols i+vl..) at hw 2*(i+vl) = 2*i + 2*vl */
        const uint16_t *ab = (const uint16_t *)pAB + 2 * i;
        const uint16_t *cd = (const uint16_t *)pCD + 2 * i;
        const uint16_t *ef = (const uint16_t *)pEF + 2 * i;
        const uint16_t *gh = (const uint16_t *)pGH + 2 * i;
        const uint16_t *ab1 = ab + 2 * vl, *cd1 = cd + 2 * vl;
        const uint16_t *ef1 = ef + 2 * vl, *gh1 = gh + 2 * vl;
        /* 16 weight loads issued up front */
        svuint16_t vA = svld1_u16(p_odd, ab - 1),  vB = svld1_u16(p_odd, ab);
        svuint16_t vC = svld1_u16(p_odd, cd - 1),  vD = svld1_u16(p_odd, cd);
        svuint16_t vE = svld1_u16(p_odd, ef - 1),  vF = svld1_u16(p_odd, ef);
        svuint16_t vG = svld1_u16(p_odd, gh - 1),  vH = svld1_u16(p_odd, gh);
        svuint16_t wA = svld1_u16(p_odd, ab1 - 1), wB = svld1_u16(p_odd, ab1);
        svuint16_t wC = svld1_u16(p_odd, cd1 - 1), wD = svld1_u16(p_odd, cd1);
        svuint16_t wE = svld1_u16(p_odd, ef1 - 1), wF = svld1_u16(p_odd, ef1);
        svuint16_t wG = svld1_u16(p_odd, gh1 - 1), wH = svld1_u16(p_odd, gh1);
        svfloat32_t vx0 = svld1(pg, &x[i]);
        svfloat32_t vx1 = svld1(pg, &x[i + vl]);
        a0 = svmla_x(pg, a0, svreinterpret_f32(vA), vx0);
        a1 = svmla_x(pg, a1, svreinterpret_f32(vB), vx0);
        a2 = svmla_x(pg, a2, svreinterpret_f32(vC), vx0);
        a3 = svmla_x(pg, a3, svreinterpret_f32(vD), vx0);
        a4 = svmla_x(pg, a4, svreinterpret_f32(vE), vx0);
        a5 = svmla_x(pg, a5, svreinterpret_f32(vF), vx0);
        a6 = svmla_x(pg, a6, svreinterpret_f32(vG), vx0);
        a7 = svmla_x(pg, a7, svreinterpret_f32(vH), vx0);
        b0 = svmla_x(pg, b0, svreinterpret_f32(wA), vx1);
        b1 = svmla_x(pg, b1, svreinterpret_f32(wB), vx1);
        b2 = svmla_x(pg, b2, svreinterpret_f32(wC), vx1);
        b3 = svmla_x(pg, b3, svreinterpret_f32(wD), vx1);
        b4 = svmla_x(pg, b4, svreinterpret_f32(wE), vx1);
        b5 = svmla_x(pg, b5, svreinterpret_f32(wF), vx1);
        b6 = svmla_x(pg, b6, svreinterpret_f32(wG), vx1);
        b7 = svmla_x(pg, b7, svreinterpret_f32(wH), vx1);
    }
    /* single-chunk remainder */
    for (; i + vl - 1 < K; i += vl) {
        const uint16_t *ab = (const uint16_t *)pAB + 2 * i;
        const uint16_t *cd = (const uint16_t *)pCD + 2 * i;
        const uint16_t *ef = (const uint16_t *)pEF + 2 * i;
        const uint16_t *gh = (const uint16_t *)pGH + 2 * i;
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, svreinterpret_f32(svld1_u16(p_odd, ab - 1)), vx);
        a1 = svmla_x(pg, a1, svreinterpret_f32(svld1_u16(p_odd, ab)), vx);
        a2 = svmla_x(pg, a2, svreinterpret_f32(svld1_u16(p_odd, cd - 1)), vx);
        a3 = svmla_x(pg, a3, svreinterpret_f32(svld1_u16(p_odd, cd)), vx);
        a4 = svmla_x(pg, a4, svreinterpret_f32(svld1_u16(p_odd, ef - 1)), vx);
        a5 = svmla_x(pg, a5, svreinterpret_f32(svld1_u16(p_odd, ef)), vx);
        a6 = svmla_x(pg, a6, svreinterpret_f32(svld1_u16(p_odd, gh - 1)), vx);
        a7 = svmla_x(pg, a7, svreinterpret_f32(svld1_u16(p_odd, gh)), vx);
    }
    a0 = svadd_x(pg,a0,b0); a1 = svadd_x(pg,a1,b1);
    a2 = svadd_x(pg,a2,b2); a3 = svadd_x(pg,a3,b3);
    a4 = svadd_x(pg,a4,b4); a5 = svadd_x(pg,a5,b5);
    a6 = svadd_x(pg,a6,b6); a7 = svadd_x(pg,a7,b7);
    dst[0] = svaddv(pg, a0); dst[1] = svaddv(pg, a1);
    dst[2] = svaddv(pg, a2); dst[3] = svaddv(pg, a3);
    dst[4] = svaddv(pg, a4); dst[5] = svaddv(pg, a5);
    dst[6] = svaddv(pg, a6); dst[7] = svaddv(pg, a7);
}

/* row8_pv_pf (variant 3): p_odd kernel + dist-8 L2 software prefetch (the
 * production TF_BF16PV_PREFETCH path). MEASURED to HELP substantially in this
 * microbench: +28% at K=3584 (545->700 GB/s) and +7% at K=18944 (708->760),
 * reaching 700-800 GB/s = 84-91% of the sve_sum streaming ceiling. The kernel
 * is partly HBM-latency-bound at moderate K, not purely throughput-bound; the
 * SW prefetch covers latency the ~16 HW prefetcher slots/CMG can't (48 streams
 * contend). Whether this transfers end-to-end is a separate question (the
 * matvec is one phase among attn/barriers) -- see the 9B A/B. */
static void row8_pv_pf(float *dst,
                       const bf16_t *pAB, const bf16_t *pCD,
                       const bf16_t *pEF, const bf16_t *pGH,
                       const float *x, int K) {
    svbool_t pg = svptrue_b32();
    svbool_t p_all_h = svptrue_b16();
    svuint16_t idx_h = svindex_u16(0, 1);
    svbool_t p_odd = svcmpne_n_u16(p_all_h, svand_n_u16_x(p_all_h, idx_h, 1), 0);
    int vl = (int)svcntw();
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    const int PFD_HW = 8 * 32;
    int i = 0;
    for (; i + vl - 1 < K; i += vl) {
        const uint16_t *ab = (const uint16_t *)pAB + 2*i, *cd = (const uint16_t *)pCD + 2*i;
        const uint16_t *ef = (const uint16_t *)pEF + 2*i, *gh = (const uint16_t *)pGH + 2*i;
        __builtin_prefetch(ab + PFD_HW, 0, 2); __builtin_prefetch(cd + PFD_HW, 0, 2);
        __builtin_prefetch(ef + PFD_HW, 0, 2); __builtin_prefetch(gh + PFD_HW, 0, 2);
        svfloat32_t vx = svld1(pg, &x[i]);
        a0 = svmla_x(pg, a0, svreinterpret_f32(svld1_u16(p_odd, ab - 1)), vx);
        a1 = svmla_x(pg, a1, svreinterpret_f32(svld1_u16(p_odd, ab)), vx);
        a2 = svmla_x(pg, a2, svreinterpret_f32(svld1_u16(p_odd, cd - 1)), vx);
        a3 = svmla_x(pg, a3, svreinterpret_f32(svld1_u16(p_odd, cd)), vx);
        a4 = svmla_x(pg, a4, svreinterpret_f32(svld1_u16(p_odd, ef - 1)), vx);
        a5 = svmla_x(pg, a5, svreinterpret_f32(svld1_u16(p_odd, ef)), vx);
        a6 = svmla_x(pg, a6, svreinterpret_f32(svld1_u16(p_odd, gh - 1)), vx);
        a7 = svmla_x(pg, a7, svreinterpret_f32(svld1_u16(p_odd, gh)), vx);
    }
    dst[0]=svaddv(pg,a0); dst[1]=svaddv(pg,a1); dst[2]=svaddv(pg,a2); dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4); dst[5]=svaddv(pg,a5); dst[6]=svaddv(pg,a6); dst[7]=svaddv(pg,a7);
}

/* dispatch one 8-row group by variant. pvgrp = pair-interleaved group base,
 * rmgrp = row-major group base (variant 2 only). */
static inline void do_group(int variant, float *y, const bf16_t *pvgrp,
                            const bf16_t *rmgrp, const float *x, int K) {
    if (variant == 3)
        row8_pv_pf(y, pvgrp+0*2*K, pvgrp+1*2*K, pvgrp+2*2*K, pvgrp+3*2*K, x, K);
    else if (variant == 2)
        row8_lsl(y, rmgrp+0*K, rmgrp+1*K, rmgrp+2*K, rmgrp+3*K,
                 rmgrp+4*K, rmgrp+5*K, rmgrp+6*K, rmgrp+7*K, x, K);
    else if (variant == 1)
        row8_pv_v1(y, pvgrp+0*2*K, pvgrp+1*2*K, pvgrp+2*2*K, pvgrp+3*2*K, x, K);
    else
        row8_pv(y, pvgrp+0*2*K, pvgrp+1*2*K, pvgrp+2*2*K, pvgrp+3*2*K, x, K);
}

/* ---- per-CMG layout: cores split into teams of 12, each its own buffer ---- */

int main(int argc, char **argv) {
    int N_groups = 31040;   /* 8*31040 = 248320 rows (Qwen3.5 lm_head scale) */
    int K = 1024;
    if (argc > 1) N_groups = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    int vl = (int)svcntw();
    if (K % vl) { fprintf(stderr, "K must be multiple of %d\n", vl); return 1; }

    const char *ve = getenv("TF_BF16PV_VARIANT");
    int variant = (ve && *ve) ? atoi(ve) : 0;

    /* allowed cores -> CMG teams of 12 */
    cpu_set_t allowed;
    sched_getaffinity(0, sizeof(allowed), &allowed);
    int cores[256], ncore = 0;
    for (int c = 0; c < CPU_SETSIZE && ncore < 256; c++)
        if (CPU_ISSET(c, &allowed)) cores[ncore++] = c;
    int ncmg = ncore / CMG_CORES, cores_per = CMG_CORES;
    if (ncmg < 1) { ncmg = 1; cores_per = ncore < 1 ? 1 : ncore; }
    int nthr = ncmg * cores_per;

    /* split groups across CMGs (each CMG gets a contiguous block) */
    int gpc = N_groups / ncmg;            /* groups per CMG */
    N_groups = gpc * ncmg;                 /* round so all CMGs equal */
    int N_rows = N_groups * 8;
    size_t gbytes = (size_t)gpc * 8 * K * sizeof(bf16_t);  /* per-CMG weight bytes */

    printf("N_groups=%d (rows=%d) K=%d VL=%d  CMGs=%d x %d cores  "
           "per-CMG W=%.1f MB  variant=%d\n",
           N_groups, N_rows, K, vl, ncmg, cores_per, gbytes / 1048576.0, variant);

    /* per-CMG weight buffers: raw mmap + MADV_NOHUGEPAGE, EXACTLY like the
     * production tf_bf16_pv_alloc. No mbind / set_mempolicy -- placement comes
     * purely from first-touch under the default (local) policy when each CMG's
     * own pinned threads fill their slice below. (posix_memalign's heap arena
     * pre-faults pages onto the main thread's node and ruins this.) */
    bf16_t *Wpv[64] = {0};   /* pair-interleaved (variants 0,1) */
    bf16_t *Wrm[64] = {0};   /* row-major (variant 2 = LSL) */
    float  *Y[64]   = {0};
    int need_rm = (variant == 2);
    for (int c = 0; c < ncmg; c++) {
        Wpv[c] = (bf16_t *)mmap(NULL, gbytes, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (Wpv[c] == MAP_FAILED) { fprintf(stderr, "mmap Wpv failed\n"); return 1; }
#ifdef MADV_NOHUGEPAGE
        madvise(Wpv[c], gbytes, MADV_NOHUGEPAGE);
#endif
        if (need_rm) {
            Wrm[c] = (bf16_t *)mmap(NULL, gbytes, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (Wrm[c] == MAP_FAILED) { fprintf(stderr, "mmap Wrm failed\n"); return 1; }
#ifdef MADV_NOHUGEPAGE
            madvise(Wrm[c], gbytes, MADV_NOHUGEPAGE);
#endif
        }
        if (posix_memalign((void **)&Y[c], 64, (size_t)gpc * 8 * sizeof(float)) != 0) {
            fprintf(stderr, "alloc Y failed\n"); return 1;
        }
    }
    /* shared activation (tiny, stays in L1) */
    float *x = NULL;
    if (posix_memalign((void **)&x, 64, (size_t)K * sizeof(float)) != 0) return 1;
    for (int k = 0; k < K; k++) x[k] = 0.5f + 0.001f * (float)k;

    /* per-CMG fill (= first-touch on local node) by the owning threads.
     * pv layout per group: 4 pairs, each 2*K bf16, even HW = rowA, odd = rowB,
     * indexed pair[chunk*32 + 2*lane + {0,1}]. Global row id keeps the value
     * formula identical to the lsl oracle. */
    double t_ceiling = 0, t_pv = 0;
    double ref_max = 0, max_err = 0;
    /* iterations per timed region: stream ~2 GiB/lane so OMP barrier wake-up
     * latency (passive wait policy) is amortized to noise, like bench_node_bw. */
    size_t lane_bytes_est = (size_t)(gpc / cores_per + 1) * 8 * K * sizeof(bf16_t);
    int iters = (int)(2.0 * (1UL << 30) / (double)lane_bytes_est);
    if (iters < 8) iters = 8; if (iters > 2000) iters = 2000;

#pragma omp parallel num_threads(nthr)
    {
        int t = omp_get_thread_num();
        int c = t / cores_per, lane = t % cores_per;
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(cores[t], &s);
        sched_setaffinity(0, sizeof(s), &s);

        /* this lane's group range within CMG c */
        int per = gpc / cores_per, extra = gpc % cores_per;
        int g0 = per * lane + (lane < extra ? lane : extra);
        int gn = per + (lane < extra ? 1 : 0);
        int g1 = g0 + gn;

        for (int lg = g0; lg < g1; lg++) {
            int gg = c * gpc + lg;   /* global group id */
            bf16_t *grp = Wpv[c] + (size_t)lg * 8 * K;
            for (int p = 0; p < 4; p++) {
                int rowA = gg * 8 + 2 * p, rowB = gg * 8 + 2 * p + 1;
                bf16_t *pair = grp + (size_t)p * 2 * K;
                for (int k = 0; k < K; k++) {
                    int chunk = k / vl, ln = k % vl;
                    float vA = ((rowA * 7919 + k * 7) % 1000) * 0.001f - 0.5f;
                    float vB = ((rowB * 7919 + k * 7) % 1000) * 0.001f - 0.5f;
                    pair[(size_t)chunk * 32 + 2 * ln + 0] = f32_to_bf16(vA);
                    pair[(size_t)chunk * 32 + 2 * ln + 1] = f32_to_bf16(vB);
                }
            }
            if (need_rm) {   /* row-major copy for the LSL variant */
                bf16_t *rgrp = Wrm[c] + (size_t)lg * 8 * K;
                for (int r = 0; r < 8; r++) {
                    int row = gg * 8 + r;
                    bf16_t *rr = rgrp + (size_t)r * K;
                    for (int k = 0; k < K; k++)
                        rr[k] = f32_to_bf16(((row * 7919 + k * 7) % 1000) * 0.001f - 0.5f);
                }
            }
        }
#pragma omp barrier
#pragma omp master
        {
            /* where did pages actually land? probe node of a sample page in
             * each CMG buffer via get_mempolicy(MPOL_F_ADDR|MPOL_F_NODE). */
            for (int cc = 0; cc < ncmg; cc++) {
                int nmid = -1, nend = -1;
                syscall(SYS_get_mempolicy, &nmid, NULL, 0UL,
                        (void *)(Wpv[cc] + (gbytes/2/sizeof(bf16_t))), 3UL);
                syscall(SYS_get_mempolicy, &nend, NULL, 0UL,
                        (void *)(Wpv[cc] + (gbytes/sizeof(bf16_t)) - 64), 3UL);
                fprintf(stderr, "[placement] CMG %d buf mid->node %d  end->node %d (want %d)\n",
                        cc, nmid, nend, node_of_cpu(cores[cc * cores_per]));
            }
        }
#pragma omp barrier

        /* ---- ceiling: sve_sum over this lane's slice ----
         * iters passes per timed region (master-timed across barriers) to
         * amortize barrier wake-up; report best of 6 trials. */
        const uint64_t *u64 = (const uint64_t *)Wpv[c];
        size_t u64_lo = (size_t)g0 * 8 * K * sizeof(bf16_t) / 8;
        size_t u64_hi = (size_t)g1 * 8 * K * sizeof(bf16_t) / 8;
        uint64_t local = 0;
        for (int w = 0; w < 3; w++) local += sve_sum(u64, u64_lo, u64_hi);  /* warmup */
        static double mt0, mt1; double best = 1e30;
        for (int trial = 0; trial < 6; trial++) {
#pragma omp barrier
#pragma omp master
            mt0 = mono_sec();
#pragma omp barrier
            for (int it = 0; it < iters; it++) local += sve_sum(u64, u64_lo, u64_hi);
#pragma omp barrier
#pragma omp master
            { mt1 = mono_sec(); if (mt1 - mt0 < best) best = mt1 - mt0; }
        }
#pragma omp master
        t_ceiling = best / iters;
        if (lane == 0 && c == 0) g_sink = local;

        /* ---- matvec over this lane's groups (selected variant) ---- */
        for (int rep = 0; rep < 2; rep++)              /* warmup */
            for (int lg = g0; lg < g1; lg++)
                do_group(variant, Y[c] + lg * 8, Wpv[c] + (size_t)lg * 8 * K,
                         need_rm ? Wrm[c] + (size_t)lg * 8 * K : NULL, x, K);
        best = 1e30;
        for (int trial = 0; trial < 6; trial++) {
#pragma omp barrier
#pragma omp master
            mt0 = mono_sec();
#pragma omp barrier
            for (int it = 0; it < iters; it++)
                for (int lg = g0; lg < g1; lg++)
                    do_group(variant, Y[c] + lg * 8, Wpv[c] + (size_t)lg * 8 * K,
                             need_rm ? Wrm[c] + (size_t)lg * 8 * K : NULL, x, K);
#pragma omp barrier
#pragma omp master
            { mt1 = mono_sec(); if (mt1 - mt0 < best) best = mt1 - mt0; }
        }
#pragma omp master
        t_pv = best / iters;

        /* ---- correctness: recompute via lsl from the same logical weights.
         * Check only the first chunk of groups per lane (enough to validate the
         * kernel; full-range check would just redo the whole stream). ---- */
        double le = 0, lr = 0;
        int gchk = (g1 - g0 > 64) ? g0 + 64 : g1;
        if (K > 16384) gchk = g0;   /* TLS row buffer guard */
        for (int lg = g0; lg < gchk; lg++) {
            int gg = c * gpc + lg;
            /* rebuild 8 row-major rows for this group from the formula */
            static __thread bf16_t rows[8][16384];
            float ref[8];
            for (int r = 0; r < 8; r++)
                for (int k = 0; k < K; k++) {
                    int row = gg * 8 + r;
                    float v = ((row * 7919 + k * 7) % 1000) * 0.001f - 0.5f;
                    rows[r][k] = f32_to_bf16(v);
                }
            row8_lsl(ref, rows[0], rows[1], rows[2], rows[3],
                     rows[4], rows[5], rows[6], rows[7], x, K);
            for (int r = 0; r < 8; r++) {
                double e = fabs((double)Y[c][lg*8+r] - (double)ref[r]);
                if (e > le) le = e;
                double rr = fabs((double)ref[r]);
                if (rr > lr) lr = rr;
            }
        }
#pragma omp critical
        { if (le > max_err) max_err = le; if (lr > ref_max) ref_max = lr; }
    }

    double tot_bytes = (double)gbytes * ncmg;
    const char *vname = variant == 2 ? "row8_lsl  " :
                        variant == 1 ? "row8_pv_v1" : "row8_pv   ";
    printf("  sve_sum   (ceiling)  %.3f ms  %7.2f GB/s  (%.1f%% of ceiling below)\n",
           t_ceiling * 1e3, tot_bytes / t_ceiling / 1e9, 100.0);
    printf("  %s           %.3f ms  %7.2f GB/s  %7.2f GFLOP/s  (%.1f%% of ceiling)\n",
           vname, t_pv * 1e3, tot_bytes / t_pv / 1e9,
           2.0 * (double)N_rows * K / t_pv / 1e9,
           100.0 * t_ceiling / t_pv);
    printf("  correctness vs lsl: max_abs_err=%.4e  rel=%.2e\n",
           max_err, max_err / (ref_max + 1e-30));
    return 0;
}
