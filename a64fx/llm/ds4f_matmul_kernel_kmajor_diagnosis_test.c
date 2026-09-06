#include <stdio.h>
#include <string.h>
#define VL 16
extern void matmul_kernel_8x1_f32_sve(const float *A, const float *B, float *C, long ldc, long kk);
int main(void) {
    int K = 4;
    float A_rowmajor[8*4];      /* A[m][k], row-major (K contiguous per row), the DOCUMENTED layout */
    float A_kmajor[4*8];        /* A_packed[k*8+m] = A[m][k], the layout the kernel ACTUALLY reads */
    float B[4*VL];
    float C[8*VL];
    float ref[8*VL];
    for (int m = 0; m < 8; m++) for (int k = 0; k < K; k++) {
        float v = (float)(m*K+k+1);
        A_rowmajor[m*K+k] = v;
        A_kmajor[k*8+m] = v;
    }
    for (int k = 0; k < K; k++) for (int j = 0; j < VL; j++) B[k*VL+j] = (float)((k*VL+j+1) % 7 - 3);
    for (int m = 0; m < 8; m++) for (int j = 0; j < VL; j++) {
        double s = 0; for (int k = 0; k < K; k++) s += (double)A_rowmajor[m*K+k]*(double)B[k*VL+j];
        ref[m*VL+j] = (float)s;
    }

    memset(C, 0, sizeof(C));
    matmul_kernel_8x1_f32_sve(A_kmajor, B, C, VL, K);   /* feed the K-MAJOR layout */
    int bad = 0;
    for (int i = 0; i < 8*VL; i++) if (C[i] != ref[i]) bad++;
    printf("K-major A layout: %d/%d mismatches -> %s\n", bad, 8*VL, bad==0 ? "MATCH (CORRECT with K-major A)" : "still differs");
    return bad;
}
