/* test_btp_pv_prod.c
 * Exercises production wrappers gemm_bf16_BTP and gemm_bf16_BTP_pv.
 * Verifies bit-identical output on a VIT-like shape, then times both. */
#include "bf16_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

typedef unsigned short bf16_t;

static double mono(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static bf16_t f32_to_bf16_one(float f) {
    unsigned int u; memcpy(&u, &f, 4);
    u += ((u >> 16) & 1) + 0x7FFF;
    return (bf16_t)(u >> 16);
}

int main(int argc, char **argv) {
    int M = (argc>1)?atoi(argv[1]):576;
    int N = (argc>2)?atoi(argv[2]):768;
    int K = (argc>3)?atoi(argv[3]):768;
    int reps = (argc>4)?atoi(argv[4]):50;

    fprintf(stderr, "shape M=%d N=%d K=%d threads=%d reps=%d\n",
            M, N, K, omp_get_max_threads(), reps);

    float *A = aligned_alloc(64, (size_t)M*K*sizeof(float));
    bf16_t *BT = aligned_alloc(64, (size_t)K*N*sizeof(bf16_t));
    float *C1 = aligned_alloc(64, (size_t)M*N*sizeof(float));
    float *C2 = aligned_alloc(64, (size_t)M*N*sizeof(float));

    srand(42);
    for (size_t i=0; i<(size_t)M*K; i++) A[i] = (rand()/(float)RAND_MAX)*2.f-1.f;
    for (size_t i=0; i<(size_t)K*N; i++) BT[i] = f32_to_bf16_one((rand()/(float)RAND_MAX)*2.f-1.f);

    /* Standard BTP */
    size_t btp_bytes = packed_B_bf16_size(K, N);
    bf16_t *BTP = aligned_alloc(64, btp_bytes);
    memset(BTP, 0, btp_bytes);
    pack_B_bf16(K, N, BT, N, BTP);

    /* PV BTP */
    size_t pv_bytes = packed_B_bf16_pv_size(K, N);
    bf16_t *BTP_pv = aligned_alloc(64, pv_bytes);
    memset(BTP_pv, 0, pv_bytes);
    pack_B_bf16_pv(K, N, BT, N, BTP_pv);

    /* Run + verify */
    gemm_bf16_BTP   (M, K, N, A, K, BTP,    C1, N);
    gemm_bf16_BTP_pv(M, K, N, A, K, BTP_pv, C2, N);
    double max_abs = 0;
    for (size_t i=0; i<(size_t)M*N; i++) {
        double d = fabs(C1[i] - C2[i]);
        if (d > max_abs) max_abs = d;
    }
    fprintf(stderr, "max_abs(C_lsl, C_pv) = %g\n", max_abs);
    if (max_abs != 0.0) { fprintf(stderr, "VERIFY FAILED\n"); return 1; }

    double flops = 2.0*M*N*K;
    double t0 = mono();
    for (int r=0; r<reps; r++) gemm_bf16_BTP(M, K, N, A, K, BTP, C1, N);
    double t1 = mono();
    double lsl = flops*reps/(t1-t0)/1e9;

    t0 = mono();
    for (int r=0; r<reps; r++) gemm_bf16_BTP_pv(M, K, N, A, K, BTP_pv, C2, N);
    t1 = mono();
    double pv  = flops*reps/(t1-t0)/1e9;

    printf("prod LSL: %8.2f GFLOP/s\n", lsl);
    printf("prod PV : %8.2f GFLOP/s   gain = %+.1f%%\n",
           pv, 100.0*(pv/lsl-1.0));
    return 0;
}
