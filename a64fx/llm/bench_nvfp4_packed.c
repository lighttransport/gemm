/* Prototype K-major eight-row GGUF NVFP4 W4A8 decode on A64FX.
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve+dotprod -fopenmp \
 *        bench_nvfp4_packed.c -lm -o build/bench_nvfp4_packed
 * Run on four CMGs with the hugepage settings in run_qwen38_nvfp4_cmg4.sh:
 *        OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
 *        numactl --physcpubind=12-59 --localalloc ./build/bench_nvfp4_packed
 * Synthetic throughput probe only: no model repacking or token validation.
 */
#define _GNU_SOURCE
#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

enum { K = 5120, ROWS = 131072, NR = 8, NS = 4, NB = K / 64 };

typedef struct { uint8_t d[4], qs[32]; } source_block;
typedef struct { float d[8]; uint8_t qs[64]; } packed_subblock;
typedef struct { packed_subblock s[4]; } packed_block;
typedef struct { int8_t lo[64], hi[64]; float scale; } activation_block;

static double wall_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float scale_ref(uint8_t d) {
    if (d == 0 || d == 0x7f) return 0.0f;
    int e = (d >> 3) & 15, m = d & 7;
    return e ? ldexpf(1.0f + (float)m / 8.0f, e - 8)
             : ldexpf((float)m, -10);
}

static void pack_tile(packed_block *restrict dst,
                      const source_block *restrict src, int tile) {
    for (int b = 0; b < NB; b++) {
        for (int s = 0; s < NS; s++) {
            packed_subblock *p = &dst[b].s[s];
            for (int r = 0; r < NR; r++) {
                const source_block *q = src + ((size_t)tile * NR + r) * NB + b;
                p->d[r] = scale_ref(q->d[s]);
                memcpy(p->qs + r * 8, q->qs + s * 8, 8);
            }
        }
    }
}

static void quantize_activation(activation_block *out, const float *x) {
    for (int b = 0; b < K / 16; b++) {
        float maxabs = 0.0f;
        for (int j = 0; j < 16; j++) {
            float a = fabsf(x[b * 16 + j]);
            if (a > maxabs) maxabs = a;
        }
        float sc = maxabs / 127.0f;
        float inv = sc > 0 ? 1.0f / sc : 0.0f;
        out[b].scale = sc;
        for (int r = 0; r < NR; r++) {
            for (int j = 0; j < 8; j++) {
                out[b].lo[r * 8 + j] = (int8_t)lrintf(x[b * 16 + j] * inv);
                out[b].hi[r * 8 + j] = (int8_t)lrintf(x[b * 16 + 8 + j] * inv);
            }
        }
    }
}

static void packed_dot8(float *dst, const packed_block *w,
                        const activation_block *a) {
    static const int8_t lut_data[64] = {
        0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12
    };
    const svbool_t pb = svptrue_b8();
    const svbool_t p8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
    svint8_t lut = svld1_s8(pb, lut_data);
    svfloat32_t acc0 = svdup_f32(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (int b = 0; b < NB; b++) {
        for (int s = 0; s < NS; s++) {
            const packed_subblock *p = &w[b].s[s];
            const activation_block *xq = &a[b * NS + s];
            svuint8_t bytes = svld1_u8(pb, p->qs);
            svint8_t lo = svtbl_s8(lut, svand_n_u8_x(pb, bytes, 15));
            svint8_t hi = svtbl_s8(lut, svlsr_n_u8_x(pb, bytes, 4));
            svint8_t al = svld1_s8(pb, xq->lo);
            svint8_t ah = svld1_s8(pb, xq->hi);
            svint32_t dots = svdot_s32(svdot_s32(svdup_s32(0), lo, al), hi, ah);
            svint32_t row_dots = svadd_s32_x(p8, svuzp1_s32(dots, dots),
                                                svuzp2_s32(dots, dots));
            svfloat32_t scale = svmul_n_f32_x(p8, svld1(p8, p->d), xq->scale);
            switch (s) {
            case 0: acc0 = svmla_x(p8, acc0, svcvt_f32_s32_x(p8, row_dots), scale); break;
            case 1: acc1 = svmla_x(p8, acc1, svcvt_f32_s32_x(p8, row_dots), scale); break;
            case 2: acc2 = svmla_x(p8, acc2, svcvt_f32_s32_x(p8, row_dots), scale); break;
            default: acc3 = svmla_x(p8, acc3, svcvt_f32_s32_x(p8, row_dots), scale); break;
            }
        }
    }
    svst1(p8, dst, svadd_f32_x(p8, svadd_f32_x(p8, acc0, acc1),
                                    svadd_f32_x(p8, acc2, acc3)));
}

int main(void) {
    if (sizeof(source_block) != 36 || sizeof(packed_block) != 384) return 2;
    size_t source_size = (size_t)ROWS * NB * sizeof(source_block);
    size_t packed_size = (size_t)(ROWS / NR) * NB * sizeof(packed_block);
    source_block *src = aligned_alloc(256, source_size);
    packed_block *packed = aligned_alloc(256, packed_size);
    activation_block *act = aligned_alloc(256, (K / 16) * sizeof(*act));
    float *output = aligned_alloc(256, ROWS * sizeof(float));
    if (!src || !packed || !act || !output) return 3;
    uint32_t rng = 1;
    uint8_t *bytes = (uint8_t *)src;
    for (size_t i = 0; i < source_size; i++) {
        rng = rng * 1664525u + 1013904223u;
        bytes[i] = (uint8_t)(rng >> 24);
    }
    for (size_t i = 0; i < (size_t)ROWS * NB; i++)
        for (int s = 0; s < NS; s++) src[i].d[s] &= 7;
    float x[K];
    for (int j = 0; j < K; j++) x[j] = sinf((float)j * 0.013f);
    quantize_activation(act, x);
    double p0 = wall_sec();
#pragma omp parallel for schedule(static)
    for (int tile = 0; tile < ROWS / NR; tile++)
        pack_tile(packed + (size_t)tile * NB, src, tile);
    double p1 = wall_sec();
    packed_dot8(output, packed, act);
    static const int8_t codes[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    double err2 = 0, ref2 = 0;
    for (int r = 0; r < NR; r++) {
        double ref = 0;
        for (int b = 0; b < NB; b++) {
            const source_block *q = src + (size_t)r * NB + b;
            for (int s = 0; s < NS; s++) {
                float scale = scale_ref(q->d[s]);
                for (int j = 0; j < 8; j++) {
                    uint8_t v = q->qs[s * 8 + j];
                    ref += (double)codes[v & 15] * scale * x[b * 64 + s * 16 + j];
                    ref += (double)codes[v >> 4] * scale * x[b * 64 + s * 16 + 8 + j];
                }
            }
        }
        double e = output[r] - ref;
        err2 += e * e; ref2 += ref * ref;
    }
    double best = 1e100;
    for (int trial = 0; trial < 4; trial++) {
        double t0 = wall_sec();
#pragma omp parallel for schedule(static)
        for (int tile = 0; tile < ROWS / NR; tile++)
            packed_dot8(output + (size_t)tile * NR,
                        packed + (size_t)tile * NB, act);
        double elapsed = wall_sec() - t0;
        if (elapsed < best) best = elapsed;
        printf("trial=%d seconds=%.6f packed_GBps=%.1f source_GBps=%.1f\n",
               trial, elapsed, packed_size / elapsed / 1e9,
               source_size / elapsed / 1e9);
    }
    printf("rows=%d K=%d packed_GB=%.3f pack_seconds=%.3f rel_l2=%.6g best_source_GBps=%.1f\n",
           ROWS, K, packed_size / 1e9, p1-p0, sqrt(err2 / ref2),
           source_size / best / 1e9);
    free(src); free(packed); free(act); free(output);
    return 0;
}
