#include "gemm_pack.h"
#include <string.h>

void pack_a(const float *A, int lda, float *A_pack, int mc, int K)
{
    memset(A_pack, 0, (size_t)MR * K * sizeof(float));
    for (int m = 0; m < mc; m++) {
        const float *src_row = A + m * lda;
        for (int k = 0; k < K; k++) {
            A_pack[k * MR + m] = src_row[k];
        }
    }
}

void pack_b(const float *B, int ldb, float *B_pack, int nc, int K)
{
    memset(B_pack, 0, (size_t)NR * K * sizeof(float));
    for (int n = 0; n < nc; n++) {
        const float *src_row = B + n * ldb;
        for (int k = 0; k < K; k++) {
            B_pack[k * NR + n] = src_row[k];
        }
    }
}
