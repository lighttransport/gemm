/* Synthetic mHC micro-benchmark: glm53f_mhc_post_pre_sve at production shape (4 streams x 4096, 24 mixing rows of bf16
 * 16384), 90 distinct weight sets rotated, modes GLM53F_MHC_FAST=0/1/2; also checks that mode 2 stays close to mode 1.
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I. -I../.. bench_glm53f_mhc.c -lm
 * run:   OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active FLIB_BARRIER=HARD ./a.out */
#define _GNU_SOURCE
#include "glm53f_mhc_sve.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    enum { SITES = 90 };
    uint16_t *fn[SITES]; float *base[SITES], *scale[SITES]; glm53f_mhc_site site[SITES];
    uint16_t *norm = aligned_alloc(256, 4096 * 2);
    for (int i = 0; i < 4096; ++i) norm[i] = 0x3f80;
    for (int s = 0; s < SITES; ++s) {
        fn[s] = aligned_alloc(256, (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT * 2); base[s] = aligned_alloc(256, 64 * 4); scale[s] = aligned_alloc(256, 64);
        for (size_t i = 0; i < (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT; ++i) fn[s][i] = (uint16_t)(0x3c00 + ((i * 2654435761u >> 20) & 0x3ff) + ((i & 1) << 15));
        for (int k = 0; k < GLM53F_MHC_MIX; ++k) base[s][k] = 0.01f * k;
        for (int k = 0; k < 3; ++k) scale[s][k] = 0.1f;
        site[s] = (glm53f_mhc_site){fn[s], base[s], scale[s]};
    }
    { /* Sinkhorn: SVE version must equal the scalar one exactly */
        int bad = 0; double ts = 0, tv = 0;
        for (int trial = 0; trial < 1000; ++trial) {
            float a[16], b[16];
            for (int i = 0; i < 16; ++i) a[i] = b[i] = sinf(1.7f * i + trial) * 3.f;
            double t0 = now(); glm53f_mhc_sinkhorn(a, 4, 20, 1e-6f); double t1 = now();
            glm53f_mhc_sinkhorn4_sve(b, 20, 1e-6f); double t2 = now();
            ts += t1 - t0; tv += t2 - t1;
            for (int i = 0; i < 16; ++i) if (memcmp(&a[i], &b[i], 4)) ++bad;
        }
        printf("sinkhorn scalar %.2f us, sve %.2f us, mismatching lanes %d / 16000\n", ts / 1000 * 1e6, tv / 1000 * 1e6, bad);
    }
    float *streams = aligned_alloc(256, GLM53F_MHC_FLAT * 4), *sub = aligned_alloc(256, 4096 * 4);
    glm53f_mhc_scratch *sc = aligned_alloc(256, sizeof(*sc));
    for (int mode = 0; mode <= 2; ++mode) {
        glm53f_mhc_fast_mode = mode;
        for (int i = 0; i < GLM53F_MHC_FLAT; ++i) streams[i] = sinf(0.01f * i);
        for (int i = 0; i < 4096; ++i) sub[i] = cosf(0.02f * i);
        memset(sc, 0, sizeof(*sc));
        glm53f_mhc_pre_sve(sc, streams, &site[0], norm);
        double t0 = now();
        for (int rep = 0; rep < 20; ++rep)
            for (int s = 1; s < SITES; ++s) glm53f_mhc_post_pre_sve(streams, sub, sc, &site[s], norm);
        double dt = (now() - t0) / (20.0 * (SITES - 1));
        double cs = 0; for (int i = 0; i < 4096; ++i) cs += sc->normalized[i];
        double t1 = now();
        for (int rep = 0; rep < 20; ++rep)
            for (int s = 1; s < SITES; ++s) glm53f_mhc_pre_sve(sc, streams, &site[s], norm);
        double dpre = (now() - t1) / (20.0 * (SITES - 1));
        printf("  pre-only mode=%d: %.1f us/call\n", mode, dpre * 1e6);
        printf("mhc post_pre mode=%d: %.1f us/call  checksum(normalized)=%.6f\n", mode, dt * 1e6, cs);
    }
    glm53f_mhc_fast_mode = 2; glm53f_mhc_stamp_on = 1; memset(glm53f_mhc_stamp, 0, sizeof(glm53f_mhc_stamp));
    for (int rep = 0; rep < 20; ++rep) for (int s = 1; s < SITES; ++s) glm53f_mhc_post_pre_sve(streams, sub, sc, &site[s], norm);
    const double n = glm53f_mhc_stamp[7];
    printf("mode2 tid0 timeline us: post+ss %.1f  dots %.1f  wait-barrier %.1f  logits+sinkhorn %.1f  collapse %.1f  barrier2 %.1f  normalize %.1f\n",
           glm53f_mhc_stamp[0] / n * 1e6, glm53f_mhc_stamp[1] / n * 1e6, glm53f_mhc_stamp[2] / n * 1e6, glm53f_mhc_stamp[3] / n * 1e6,
           glm53f_mhc_stamp[4] / n * 1e6, glm53f_mhc_stamp[5] / n * 1e6, glm53f_mhc_stamp[6] / n * 1e6);
    return 0;
}
