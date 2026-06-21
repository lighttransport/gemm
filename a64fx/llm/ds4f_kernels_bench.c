/*
 * DS4F Stage-0 micro-bench: validate + time the on-demand dequant matvecs
 * (split-layout FP8 E4M3 dense, split-layout MXFP4 experts) in isolation —
 * no MPI, no model. De-risks the two biggest unknowns before integration:
 *   (1) on-demand dequant throughput on SVE (GB/s vs HBM ceiling), and
 *   (2) the E8M0 scale convention (validated against a scalar reference).
 *
 * Each kernel is checked against a scalar dequant+dot reference (correctness)
 * and timed single-thread (effective GB/s + ns/call + B/elem).
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *       -I../../common -o ds4f_kernels_bench ds4f_kernels_bench.c -lm
 *   OMP_NUM_THREADS=1 ./ds4f_kernels_bench
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <arm_sve.h>

#include "ggml_dequant.h"   /* brings the DS4F matvecs + e8m0/fp8 helpers */

/* ---- timing ------------------------------------------------------------ */
static inline uint64_t rdcyc(void) {
    uint64_t v; __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
static inline uint64_t rdfreq(void) {
    uint64_t v; __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(v)); return v;
}

/* ---- tiny deterministic PRNG (splitmix64) ------------------------------ */
static uint64_t sm_state;
static inline uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint8_t rand_byte(void) { return (uint8_t)(sm_next() & 0xff); }
static inline float   rand_unit(void) { /* in [-1,1) */
    return (float)((int64_t)(sm_next() & 0xffffff) - 0x800000) / (float)0x800000;
}

/* ---- scalar references ------------------------------------------------- */
/* FP8 E4M3 byte -> f32 (scalar, via the same bit decode the LUT is built from) */
static inline float fp8_e4m3_ref(uint8_t b) {
    uint32_t bits = ds4f_fp8_e4m3_to_fp32_bits(b);
    float f; memcpy(&f, &bits, 4); return f;
}

/* FP8 dense dot, double accumulation (accuracy ground truth). */
static float fp8_dot_dbl(const uint8_t *w, const uint8_t *escale,
                         const float *x, int K) {
    double acc = 0.0;
    for (int c = 0; c < K; c++) {
        float s = ggml_e8m0_to_fp32(escale[c >> 7]);
        acc += (double)(fp8_e4m3_ref(w[c]) * s) * (double)x[c];
    }
    return (float)acc;
}
/* FP8 dense dot, f32 16-lane accumulation mimicking matvec_fp8e4m3_8row's
 * reduction order (lane j sums cols c with c%16==j; x pre-scaled by block s).
 * Kernel-vs-this isolates SVE indexing/unpack bugs from accumulation precision. */
static float fp8_dot_lane(const uint8_t *w, const uint8_t *escale,
                          const float *x, int K) {
    float lane[16] = {0};
    for (int c0 = 0; c0 < K; c0 += 128) {
        float s = ggml_e8m0_to_fp32(escale[c0 >> 7]);
        for (int c = c0; c < c0 + 128; c += 16)
            for (int j = 0; j < 16; j++)
                lane[j] += fp8_e4m3_ref(w[c + j]) * (x[c + j] * s);
    }
    float acc = 0.f; for (int j = 0; j < 16; j++) acc += lane[j];
    return acc;
}

/* split-MXFP4 dot, double accumulation (accuracy ground truth). */
static float mxfp4_dot_dbl(const uint8_t *w, const uint8_t *s,
                           const float *x, int K) {
    double acc = 0.0;
    int nb = K / 32;
    for (int b = 0; b < nb; b++) {
        float sc = ggml_e8m0_to_fp32(s[b]);
        const uint8_t *bp = w + (size_t)b * 16;
        for (int j = 0; j < 16; j++) {
            acc += (double)(ds4f_kvalues_mxfp4_f32[bp[j] & 0xf] * sc) * (double)x[b * 32 + j];
            acc += (double)(ds4f_kvalues_mxfp4_f32[bp[j] >> 4] * sc) * (double)x[b * 32 + j + 16];
        }
    }
    return (float)acc;
}
/* split-MXFP4 dot, f32 16-lane accumulation mimicking matvec_mxfp4_8row:
 * lane j += sc_b*(wlo[b,j]*x[b*32+j] + whi[b,j]*x[b*32+j+16]). */
static float mxfp4_dot_lane(const uint8_t *w, const uint8_t *s,
                            const float *x, int K) {
    float lane[16] = {0};
    int nb = K / 32;
    for (int b = 0; b < nb; b++) {
        float sc = ggml_e8m0_to_fp32(s[b]);
        const uint8_t *bp = w + (size_t)b * 16;
        for (int j = 0; j < 16; j++) {
            float p = ds4f_kvalues_mxfp4_f32[bp[j] & 0xf] * x[b * 32 + j]
                    + ds4f_kvalues_mxfp4_f32[bp[j] >> 4] * x[b * 32 + j + 16];
            lane[j] += p * sc;
        }
    }
    float acc = 0.f; for (int j = 0; j < 16; j++) acc += lane[j];
    return acc;
}

static double maxrelerr(const float *a, const float *b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        double den = fabs((double)b[i]) + 1e-6;
        double r = d / den;
        if (r > m) m = r;
    }
    return m;
}

/* ====================================================================== */
int main(int argc, char **argv) {
    int K = (argc > 1) ? atoi(argv[1]) : 4096;   /* dense/expert contraction dim */
    if (K % 128 != 0) { fprintf(stderr, "K must be multiple of 128\n"); return 1; }
    sm_state = 0xD5F0000000000001ULL;
    double freq = (double)rdfreq();
    printf("DS4F Stage-0 kernel bench  K=%d  cntfrq=%.3f MHz\n", K, freq / 1e6);

    /* ---- e8m0 convention check (standard 2^(x-127), x in [1,254]) ----- */
    {
        double m = 0.0; int bad = 0;
        for (int x = 1; x <= 254; x++) {
            float got = ggml_e8m0_to_fp32((uint8_t)x);
            float exp = ldexpf(1.0f, x - 127);
            double r = fabs((double)got - (double)exp) / (double)exp;
            if (r > m) m = r;
            if (r > 1e-12) bad++;
        }
        printf("[e8m0]  2^(x-127) max_rel_err=%.3e  mismatches=%d  -> %s\n",
               m, bad, bad == 0 ? "OK" : "FAIL");
    }

    const float *x;
    float *xbuf = (float *)aligned_alloc(256, (size_t)K * sizeof(float));
    for (int i = 0; i < K; i++) xbuf[i] = rand_unit();
    x = xbuf;

    /* ================= FP8 E4M3 dense (1.0 B/elem) ===================== */
    {
        uint32_t lut[256]; ds4f_init_fp8_e4m3_lut(lut);
        int nblk = K / 128;
        uint8_t *w = (uint8_t *)aligned_alloc(256, (size_t)8 * K);
        uint8_t *escale = (uint8_t *)aligned_alloc(256, (size_t)nblk);
        for (int b = 0; b < nblk; b++) escale[b] = (uint8_t)(125 + (rand_byte() % 5)); /* 2^-2..2^2 */
        for (size_t i = 0; i < (size_t)8 * K; i++) {
            uint8_t b = rand_byte();
            if ((b & 0x78) == 0x78) b &= ~0x08;   /* force exp != 15 (avoid NaN) */
            w[i] = b;
        }
        const uint8_t *wr[8];
        for (int r = 0; r < 8; r++) wr[r] = w + (size_t)r * K;

        float dst[8], refd[8], refl[8];
        for (int r = 0; r < 8; r++) { refd[r] = fp8_dot_dbl(wr[r], escale, x, K);
                                      refl[r] = fp8_dot_lane(wr[r], escale, x, K); }
        matvec_fp8e4m3_8row(dst, wr[0], wr[1], wr[2], wr[3], wr[4], wr[5], wr[6], wr[7],
                            escale, lut, x, K);
        double errl = maxrelerr(dst, refl, 8), errd = maxrelerr(dst, refd, 8);
        printf("[fp8 ]  vs-lane-f32=%.3e (kernel correctness) vs-double=%.3e (f32 acc) -> %s\n",
               errl, errd, errl < 2e-5 ? "OK" : "FAIL");

        /* perf */
        volatile float sink = 0.f;
        int warm = 200, iters = 4000;
        for (int it = 0; it < warm; it++) {
            matvec_fp8e4m3_8row(dst, wr[0], wr[1], wr[2], wr[3], wr[4], wr[5], wr[6], wr[7],
                                escale, lut, x, K); sink += dst[0];
        }
        uint64_t t0 = rdcyc();
        for (int it = 0; it < iters; it++) {
            matvec_fp8e4m3_8row(dst, wr[0], wr[1], wr[2], wr[3], wr[4], wr[5], wr[6], wr[7],
                                escale, lut, x, K); sink += dst[0];
        }
        uint64_t t1 = rdcyc();
        double sec = (double)(t1 - t0) / freq;
        double bytes = (double)iters * ((double)8 * K + nblk);  /* weights + scale */
        double macs  = (double)iters * 8.0 * K;
        printf("[fp8 ]  %.1f GB/s  %.2f ns/call  %.3f B/elem  %.0f Mmac/s  (sink=%.3g)\n",
               bytes / sec / 1e9, sec / iters * 1e9, bytes / macs,
               macs / sec / 1e6, (double)sink);
        free(w); free(escale);
    }

    /* ================= MXFP4 split experts (0.53 B/elem) ============== */
    {
        int nb = K / 32;
        uint8_t *w = (uint8_t *)aligned_alloc(256, (size_t)8 * (K / 2));
        uint8_t *s = (uint8_t *)aligned_alloc(256, (size_t)8 * nb);
        for (size_t i = 0; i < (size_t)8 * (K / 2); i++) w[i] = rand_byte();
        for (size_t i = 0; i < (size_t)8 * nb; i++) s[i] = (uint8_t)(123 + (rand_byte() % 9)); /* 2^-4..2^4 */
        const uint8_t *wr[8], *sr[8];
        for (int r = 0; r < 8; r++) { wr[r] = w + (size_t)r * (K / 2); sr[r] = s + (size_t)r * nb; }

        float dst[8], refd[8], refl[8];
        for (int r = 0; r < 8; r++) { refd[r] = mxfp4_dot_dbl(wr[r], sr[r], x, K);
                                      refl[r] = mxfp4_dot_lane(wr[r], sr[r], x, K); }
        matvec_mxfp4_8row(dst, wr[0], wr[1], wr[2], wr[3], wr[4], wr[5], wr[6], wr[7],
                          sr[0], sr[1], sr[2], sr[3], sr[4], sr[5], sr[6], sr[7], x, K);
        double errl = maxrelerr(dst, refl, 8), errd = maxrelerr(dst, refd, 8);
        printf("[mxfp4] vs-lane-f32=%.3e (kernel correctness) vs-double=%.3e (f32 acc) -> %s\n",
               errl, errd, errl < 2e-5 ? "OK" : "FAIL");

        volatile float sink = 0.f;
        int warm = 200, iters = 4000;
        for (int it = 0; it < warm; it++) {
            matvec_mxfp4_8row(dst, wr[0], wr[1], wr[2], wr[3], wr[4], wr[5], wr[6], wr[7],
                              sr[0], sr[1], sr[2], sr[3], sr[4], sr[5], sr[6], sr[7], x, K);
            sink += dst[0];
        }
        uint64_t t0 = rdcyc();
        for (int it = 0; it < iters; it++) {
            matvec_mxfp4_8row(dst, wr[0], wr[1], wr[2], wr[3], wr[4], wr[5], wr[6], wr[7],
                              sr[0], sr[1], sr[2], sr[3], sr[4], sr[5], sr[6], sr[7], x, K);
            sink += dst[0];
        }
        uint64_t t1 = rdcyc();
        double sec = (double)(t1 - t0) / freq;
        double bytes = (double)iters * ((double)8 * (K / 2) + 8 * nb); /* nibbles + scale */
        double macs  = (double)iters * 8.0 * K;
        printf("[mxfp4] %.1f GB/s  %.2f ns/call  %.3f B/elem  %.0f Mmac/s  (sink=%.3g)\n",
               bytes / sec / 1e9, sec / iters * 1e9, bytes / macs,
               macs / sec / 1e6, (double)sink);
        free(w); free(s);
    }

    free(xbuf);
    return 0;
}
