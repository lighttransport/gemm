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
    int rows = 384, cols = 384, N = 2, NMAX = 6;  /* NMAX: largest N exercised below */
    uint16_t *W = glm5_amalloc((size_t)rows*cols*2);
    /* X/Y must be sized for NMAX, not N: the N2=3..6 loop below reads X rows [0,N2) and
     * writes Y rows [0,N2). Sizing for N=2 made Y's OOB rows alias Y1 (and X's OOB rows
     * alias Y), so the N>=3 checks compared shifted copies of the reference -> guaranteed
     * FAIL on ANY correct machine (reproduced under qemu-aarch64 too, with heap corruption). */
    float *X = glm5_amalloc((size_t)NMAX*cols*4), *Y = glm5_amalloc((size_t)NMAX*rows*4);
    glm5_sm = 0; glm5_fill_bf16(W, (size_t)rows*cols, 0.03f);
    for (int i = 0; i < NMAX*cols; i++) X[i] = 0.001f * (i % 97);
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
    printf(" GEMM N=2 %s (%d diffs row0 vs mv)\n", bad ? "FAIL" : "PASS", bad);
    /* N>=3 exercises the SVE 3-stream kernel glm5_bf16_4row_3x_acc (t+2<N path) —
     * NOT covered by N<=2. Reference: per-token glm5_mv_bf16. */
    for (int N2 = 3; N2 <= 6; N2++) {
        glm5_gemm_bf16(Y, W, X, N2, rows, cols);
        int b2 = 0; float worst = 0;
        for (int t = 0; t < N2; t++) {
            glm5_mv_bf16(Y1, W, X + (size_t)t*cols, rows, cols);
            for (int r = 0; r < rows; r++) {
                float d = Y[(size_t)t*rows + r] - Y1[r];
                float ad = d < 0 ? -d : d, aa = Y1[r] < 0 ? -Y1[r] : Y1[r];
                float rel = ad / (aa > 1e-6f ? aa : 1e-6f);
                if (rel > 1e-3f) { b2++; if (rel > worst) worst = rel; }
            }
        }
        printf(" GEMM N=%d %s (%d bad, worst_rel %d ppm)\n", N2, b2 ? "FAIL" : "PASS", b2, (int)(worst*1e6f));
    }
    return 0;
}
