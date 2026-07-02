/* micro-repro: batched bf16 GEMM under qlair (single core, no comm) */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "utofu.h"
#include "glm5.h"
#include "glm5_impl.h"
#define MK(s) do { ssize_t w_ = write(1, s, sizeof(s)-1); (void)w_; } while (0)
int main(void){
    int rows = 384, cols = 384, N = 2;
    uint16_t *W = glm5_amalloc((size_t)rows*cols*2);
    float *X = glm5_amalloc((size_t)N*cols*4), *Y = glm5_amalloc((size_t)N*rows*4);
    glm5_sm = 0; glm5_fill_bf16(W, (size_t)rows*cols, 0.03f);
    for (int i = 0; i < N*cols; i++) X[i] = 0.001f * (i % 97);
    MK("[mv]");
    float *Y1 = glm5_amalloc((size_t)rows*4);
    glm5_mv_bf16(Y1, W, X, rows, cols);          /* proven path */
    MK("[gemm1]");
    glm5_gemm_bf16(Y, W, X, 1, rows, cols);      /* batched N=1 */
    MK("[gemm2]");
    glm5_gemm_bf16(Y, W, X, 2, rows, cols);      /* batched N=2 */
    MK("[chk]");
    int bad = 0;
    for (int r = 0; r < rows; r++) if (Y[r] != Y1[r]) bad++;
    printf(" GEMM %s (%d diffs row0 vs mv)\n", bad ? "FAIL" : "PASS", bad);
    return 0;
}
