// Numerical correctness test for the DS4F-model-generated SVE1.0 kernel
// matmul_kernel_8x1_f32_sve (ds4f_matmul_kernel_think2_fixed.S).
//
// Computes C[m][0:VL) += A[m][k]*B[k][0:VL) for m=0..7, sum over k=0..KDIM-1.
// We build B with a TRUE row stride N > VL (32 vs VL=16), so the "advance B
// by ldc floats/row" vs "advance B by VL floats/row" interpretations produce
// GENUINELY DIFFERENT reads of the same buffer -- letting us empirically
// determine which stride the compiled kernel actually uses, resolving the
// ambiguity noted in the model's own (unapplied) self-correction comments.
//
// ref_ldc: reference assuming the kernel advances B by ldc floats/row (i.e.
//          B's true stride == ldc, the standard no-padding row-major GEMM
//          convention: B[k][j] = B_flat[k*ldc + j]).
// ref_vl:  reference assuming the kernel (mistakenly) advances B by VL
//          floats/row instead: B[k][j] = B_flat[k*VL + j].
// Whichever the kernel's actual output matches tells us which stride it uses.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VL 16
#define KDIM 8
#define N  32   /* B's true row width; N > VL so N-stride vs VL-stride diverge */

extern void matmul_kernel_8x1_f32_sve(const float *A, const float *B,
                                       float *C, long ldc, long kk);

static float A[8 * KDIM];
static float B[KDIM * N];
static float C_kernel[8 * N];   /* sized for ldc=N row stride -- the kernel's st1w uses ldc*4 byte row spacing */
static float ref_ldc[8][VL];
static float ref_vl[8][VL];

static void fill(float *buf, int n, unsigned seed) {
    unsigned x = seed;
    for (int i = 0; i < n; i++) {
        x = x * 1103515245u + 12345u;
        /* small signed values, deterministic, no huge dynamic range */
        buf[i] = ((int)(x >> 16) % 2000 - 1000) / 100.0f;
    }
}

static double relerr(float a, float b) {
    double d = fabs((double)a - (double)b);
    double denom = fabs((double)b);
    return denom > 1e-9 ? d / denom : d;
}

int main(void) {
    fill(A, 8 * KDIM, 12345u);
    fill(B, KDIM * N, 999u);

    /* reference #1: B's true row stride == ldc (== N here) */
    for (int m = 0; m < 8; m++)
        for (int j = 0; j < VL; j++) {
            double s = 0.0;
            for (int k = 0; k < KDIM; k++) s += (double)A[m * KDIM + k] * (double)B[k * N + j];
            ref_ldc[m][j] = (float)s;
        }
    /* reference #2: (hypothetical) B's row stride == VL instead of ldc */
    for (int m = 0; m < 8; m++)
        for (int j = 0; j < VL; j++) {
            double s = 0.0;
            for (int k = 0; k < KDIM; k++) s += (double)A[m * KDIM + k] * (double)B[k * VL + j];
            ref_vl[m][j] = (float)s;
        }

    memset(C_kernel, 0, sizeof(C_kernel));
    matmul_kernel_8x1_f32_sve(A, B, C_kernel, /*ldc=*/N, /*KDIM=*/KDIM);

    double max_err_ldc = 0.0, max_err_vl = 0.0;
    int mismatches_ldc = 0, mismatches_vl = 0;
    for (int m = 0; m < 8; m++) {
        for (int j = 0; j < VL; j++) {
            float got = C_kernel[m * N /*kernel's ldc-based store stride*/ + j];
            /* NOTE: kernel stores with row stride = ldc = N (per its own st1w
             * address arithmetic, x22=ldc), so read C_kernel with stride N. */
            double e1 = relerr(got, ref_ldc[m][j]);
            double e2 = relerr(got, ref_vl[m][j]);
            if (e1 > max_err_ldc) max_err_ldc = e1;
            if (e2 > max_err_vl) max_err_vl = e2;
            if (e1 > 1e-4) mismatches_ldc++;
            if (e2 > 1e-4) mismatches_vl++;
        }
    }

    printf("=== matmul_kernel_8x1_f32_sve correctness test (VL=%d KDIM=%d N=%d ldc=%d) ===\n", VL, KDIM, N, N);
    printf("vs ref_ldc (B stride=ldc=N, standard convention): max relerr=%.3e  mismatches=%d/%d  -> %s\n",
           max_err_ldc, mismatches_ldc, 8*VL, mismatches_ldc == 0 ? "MATCH" : "DIFFER");
    printf("vs ref_vl  (B stride=VL, the model's flagged-but-unapplied fix): max relerr=%.3e  mismatches=%d/%d  -> %s\n",
           max_err_vl, mismatches_vl, 8*VL, mismatches_vl == 0 ? "MATCH" : "DIFFER");

    printf("\nfirst-row sample: kernel=[");
    for (int j = 0; j < 4; j++) printf("%.4f ", C_kernel[j]);
    printf("...]  ref_ldc=[");
    for (int j = 0; j < 4; j++) printf("%.4f ", ref_ldc[0][j]);
    printf("...]  ref_vl=[");
    for (int j = 0; j < 4; j++) printf("%.4f ", ref_vl[0][j]);
    printf("...]\n");

    if (mismatches_ldc == 0) { printf("\nVERDICT: kernel is NUMERICALLY CORRECT (matches standard ldc-stride GEMM semantics).\n"); return 0; }
    if (mismatches_vl == 0)  { printf("\nVERDICT: kernel matches the VL-stride interpretation instead (unexpected).\n"); return 1; }
    printf("\nVERDICT: kernel matches NEITHER reference -- genuine bug beyond the stride ambiguity.\n");
    return 2;
}
