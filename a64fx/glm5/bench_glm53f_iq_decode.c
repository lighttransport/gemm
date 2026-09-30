/* Synthetic single-node benchmark of the decode routed-expert step (glm53f_iq_expert_weighted) at production shape:
 * ~5 expert parts per call (gate/up Q4_K 512x4096, down Q5_K 4096x256), 47 threads, weights cycled through a pool
 * larger than the caches so every call streams from HBM.  IQ_FAST=0/1 (env GLM53F_IQ_FAST) selects the row kernels.
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -I. -Ikern -I../.. \
 *          bench_glm53f_iq_decode.c kern/glm53f_kern_*.c kern/glm53f_kern_gemm_asm.S -lm -lpthread
 * run:   OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores [PARTS=5] [POOL=96] ./a.out */
#define _GNU_SOURCE
#include "glm53f_iq_bridge.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <arm_sve.h>
extern double glm53f_iq_stage_us[6];
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
enum { H = 4096, INTER = 256 };
int main(void) {
    const int parts = getenv("PARTS") ? atoi(getenv("PARTS")) : 5, pool = getenv("POOL") ? atoi(getenv("POOL")) : 96;
    const int iters = getenv("ITERS") ? atoi(getenv("ITERS")) : 400;
    if (getenv("INTERLEAVE") && atoi(getenv("INTERLEAVE"))) { unsigned long m = 0xF0UL; syscall(SYS_set_mempolicy, 3L, &m, 8UL); }
    const size_t grb = glm53f_iq_row_size(12, H), drb = glm53f_iq_row_size(13, INTER);
    const size_t gbytes = (size_t)2 * INTER * grb, dbytes = (size_t)H * drb;
    uint8_t **gu = malloc(pool * sizeof(*gu)), **dn = malloc(pool * sizeof(*dn));
    srand(3);
    for (int p = 0; p < pool; ++p) {
        if (getenv("CONTIG")) { /* one big allocation for the whole pool (like the production blob) */
            static uint8_t *arena; const size_t gs = (gbytes + 4095) / 4096 * 4096, ds = (dbytes + 4095) / 4096 * 4096;
            if (!arena) arena = aligned_alloc(2 << 20, (size_t)pool * (gs + ds) + (2 << 20));
            gu[p] = arena + (size_t)p * (gs + ds); dn[p] = gu[p] + gs;
        } else { gu[p] = aligned_alloc(256, (gbytes + 255) / 256 * 256); dn[p] = aligned_alloc(256, (dbytes + 255) / 256 * 256); }
        if (glm53f_iq_affine_enabled()) glm53f_iq_place_part(gu[p], grb, 2 * INTER, dn[p], drb, H);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < gbytes; ++i) gu[p][i] = (uint8_t)(i * 2654435761u >> 13) + p;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < dbytes; ++i) dn[p][i] = (uint8_t)(i * 2246822519u >> 11) + p;
        for (int r = 0; r < 2 * INTER; ++r) for (int b = 0; b < H / 256; ++b) { _Float16 d = (_Float16)0.002f, m = (_Float16)0.001f; memcpy(gu[p] + r * grb + b * 144, &d, 2); memcpy(gu[p] + r * grb + b * 144 + 2, &m, 2); }
        for (int r = 0; r < H; ++r) { _Float16 d = (_Float16)0.002f, m = (_Float16)0.001f; memcpy(dn[p] + r * drb, &d, 2); memcpy(dn[p] + r * drb + 2, &m, 2); }
    }
    float *x = aligned_alloc(256, H * 4), *out = aligned_alloc(256, H * 4), *up = aligned_alloc(256, 9 * 1024 * 4), *act = aligned_alloc(256, 9 * 512 * 4);
    for (int i = 0; i < H; ++i) x[i] = sinf(0.37f * i) + 0.1f * cosf(0.11f * i);
    float w[9]; for (int k = 0; k < 9; ++k) w[k] = 0.3f;
    double best = 1e9, sum = 0; float chk = 0;
    for (int it = -20; it < iters; ++it) {
        glm53f_iq_part pr[9];
        for (int k = 0; k < parts; ++k) { int p = ((it + 20) * 7 + k * 13) % pool; pr[k] = (glm53f_iq_part){gu[p], dn[p], 12, 13, INTER}; }
        double t0 = now();
        glm53f_iq_expert_weighted(out, pr, w, parts, x, up, act);
        double dt = now() - t0;
        if (it >= 0) { sum += dt; if (dt < best) best = dt; chk += out[it % H]; }
    }
    const int pfd = getenv("PFD") ? atoi(getenv("PFD")) : 0, pfl = getenv("PFL") ? atoi(getenv("PFL")) : 0;
    if (getenv("BIGREAD")) { /* plain contiguous streaming-read bandwidth: MB per call over one big array */
        const size_t total = (size_t)512 << 20, per = (size_t)atoi(getenv("BIGREAD")) << 20;
        uint8_t *big = aligned_alloc(4096, total);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < total; i += 4096) big[i] = 1;
        double s2 = 0; float sink = 0; int n = 0;
        for (int it = -10; it < 200; ++it) {
            const size_t off = ((size_t)(it + 10) * per * 3) % (total - per);
            double t0 = now();
#pragma omp parallel reduction(+:sink)
            {
                const int nt = omp_get_num_threads(), tid = omp_get_thread_num();
                const svbool_t pg = svptrue_b32(), p8 = svptrue_b8();
                svfloat32_t acc = svdup_f32(0);
                size_t g0 = off + per * tid / nt / 256 * 256, g1 = off + per * (tid + 1) / nt / 256 * 256;
                for (size_t o = g0; o < g1; o += 256) {
                    acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(p8, big + o)));
                    acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(p8, big + o + 64)));
                    acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(p8, big + o + 128)));
                    acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(p8, big + o + 192)));
                }
                sink += svaddv_f32(pg, acc);
            }
            if (it >= 0) { s2 += now() - t0; ++n; }
        }
        printf("BIGREAD %zu MB/call: mean %.1f us -> %.0f GB/s (sink %g)\n", per >> 20, s2 / n * 1e6, per / (s2 / n) / 1e9, sink);
        return 0;
    }
    if (getenv("READ")) { /* memory floor: same bytes, plain streaming loads, same static partition, one region per call */
        double s2 = 0, b2 = 1e9; float sink = 0;
        for (int it = -20; it < iters; ++it) {
            double t0 = now();
#pragma omp parallel reduction(+:sink)
            {
                const int nt = omp_get_num_threads(), tid = omp_get_thread_num();
                svfloat32_t acc = svdup_f32(0);
                const svbool_t pg = svptrue_b32();
                for (int k = 0; k < parts; ++k) {
                    int p = ((it + 20) * 7 + k * 13) % pool;
                    size_t g0 = gbytes * tid / nt / 64 * 64, g1 = gbytes * (tid + 1) / nt / 64 * 64;
                    for (size_t o = g0; o < g1; o += 256) {
                        if (pfd) { __builtin_prefetch(gu[p] + o + pfd, 0, 0); __builtin_prefetch(gu[p] + o + pfd + 128, 0, 0); }
                        acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(svptrue_b8(), gu[p] + o)));
                        acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(svptrue_b8(), gu[p] + o + 64)));
                        acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(svptrue_b8(), gu[p] + o + 128)));
                        acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(svptrue_b8(), gu[p] + o + 192)));
                    }
                }
#pragma omp barrier
                for (int k = 0; k < parts; ++k) {
                    int p = ((it + 20) * 7 + k * 13) % pool;
                    size_t g0 = dbytes * tid / nt / 64 * 64, g1 = dbytes * (tid + 1) / nt / 64 * 64;
                    for (size_t o = g0; o + 256 <= g1; o += 256) {
                        if (pfd) { __builtin_prefetch(dn[p] + o + pfd, 0, 0); __builtin_prefetch(dn[p] + o + pfd + 128, 0, 0); }
                        acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(svptrue_b8(), dn[p] + o)));
                        acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(svptrue_b8(), dn[p] + o + 64)));
                        acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(svptrue_b8(), dn[p] + o + 128)));
                        acc = svadd_f32_x(pg, acc, svreinterpret_f32_u8(svld1_u8(svptrue_b8(), dn[p] + o + 192)));
                    }
                }
                sink += svaddv_f32(pg, acc);
            }
            double dt = now() - t0;
            if (it >= 0) { s2 += dt; if (dt < b2) b2 = dt; }
        }
        printf("READ floor (gate/up, barrier, down): mean %.1f us best %.1f us -> %.0f GB/s (sink %g)\n", s2 / iters * 1e6, b2 * 1e6, parts * (gbytes + dbytes) / (s2 / iters) / 1e9, sink);
    }
    printf("iq_expert_weighted parts=%d threads=%d fast=%s interleave=%s: mean %.1f us best %.1f us  (weights/call %.2f MB -> %.0f GB/s at mean) chk=%g\n",
           parts, omp_get_max_threads(), getenv("GLM53F_IQ_FAST") ? getenv("GLM53F_IQ_FAST") : "1", getenv("INTERLEAVE") ? getenv("INTERLEAVE") : "0", sum / iters * 1e6, best * 1e6,
           parts * (gbytes + dbytes) / 1e6, parts * (gbytes + dbytes) / (sum / iters) / 1e9, chk);
    if (glm53f_iq_stage_us[5] > 0) {
        const double c = glm53f_iq_stage_us[5];
        printf("  stages us/call: quant+prep %.1f  gate/up %.1f  swiglu %.1f  act-quant %.1f  down %.1f\n", glm53f_iq_stage_us[0] / c,
               glm53f_iq_stage_us[1] / c, glm53f_iq_stage_us[2] / c, glm53f_iq_stage_us[3] / c, glm53f_iq_stage_us[4] / c);
    }
    return 0;
}
