#include "gemm_pack.h"
#include <string.h>

void pack_a(const float *A, int lda, float *A_pack, int mc, int K)
{
    /* Pack A[mc×K] row-major into column-panel: A_pack[k * MR + m] = A[m][k]
     * Iterate rows in outer loop for sequential reads of source matrix.
     */

    /* Zero the entire pack buffer first (handles zero-padding for mc < MR) */
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
    /* Pack B[nc×K] row-major into row-panel: B_pack[k * NR + n] = B[n][k]
     * Iterate rows in outer loop for sequential reads of source matrix.
     */

    /* Zero the entire pack buffer first (handles zero-padding for nc < NR) */
    memset(B_pack, 0, (size_t)NR * K * sizeof(float));

    for (int n = 0; n < nc; n++) {
        const float *src_row = B + n * ldb;
        for (int k = 0; k < K; k++) {
            B_pack[k * NR + n] = src_row[k];
        }
    }
}
