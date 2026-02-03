/*
 * Test if ultra-fast exp2 produces acceptable softmax results
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

extern void exp2_ultra_rows(
    const int32_t* S, float* P, int M, int Nc,
    float scale, float neg_max, int ld_s, int ld_p);

extern void exp2_rows(
    const int32_t* S, float* P, int M, int Nc,
    float scale, float max_val, int ld_s, int ld_p);

int main() {
    int M = 4, Nc = 64;

    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* P_ref = aligned_alloc(64, M * Nc * sizeof(float));
    float* P_ultra = aligned_alloc(64, M * Nc * sizeof(float));

    srand(42);
    for (int i = 0; i < M * Nc; i++) {
        S[i] = (rand() % 201) - 100;  // -100 to 100
    }

    float scale = 1.0f / 64.0f;
    float max_val = 1.5f;

    /* Compute exp2 with both methods */
    exp2_rows(S, P_ref, M, Nc, scale, max_val, Nc, Nc);
    exp2_ultra_rows(S, P_ultra, M, Nc, scale, -max_val, Nc, Nc);

    printf("=== Softmax Accuracy Test ===\n\n");

    /* For each row, compute softmax and compare */
    for (int i = 0; i < M; i++) {
        float sum_ref = 0, sum_ultra = 0;
        for (int j = 0; j < Nc; j++) {
            sum_ref += P_ref[i * Nc + j];
            sum_ultra += P_ultra[i * Nc + j];
        }

        printf("Row %d:\n", i);
        printf("  Sum ref: %.6f, Sum ultra: %.6f\n", sum_ref, sum_ultra);

        /* Compute normalized softmax */
        float max_softmax_err = 0;
        float avg_softmax_err = 0;

        for (int j = 0; j < Nc; j++) {
            float sm_ref = P_ref[i * Nc + j] / sum_ref;
            float sm_ultra = P_ultra[i * Nc + j] / sum_ultra;

            float err = fabsf(sm_ref - sm_ultra);
            if (err > max_softmax_err) max_softmax_err = err;
            avg_softmax_err += err;
        }
        avg_softmax_err /= Nc;

        printf("  Softmax max abs error: %.6f\n", max_softmax_err);
        printf("  Softmax avg abs error: %.6f\n\n", avg_softmax_err);

        /* Check if top-k elements are preserved */
        int top1_ref = 0, top1_ultra = 0;
        float max_p_ref = P_ref[i * Nc], max_p_ultra = P_ultra[i * Nc];
        for (int j = 1; j < Nc; j++) {
            if (P_ref[i * Nc + j] > max_p_ref) {
                max_p_ref = P_ref[i * Nc + j];
                top1_ref = j;
            }
            if (P_ultra[i * Nc + j] > max_p_ultra) {
                max_p_ultra = P_ultra[i * Nc + j];
                top1_ultra = j;
            }
        }
        printf("  Top-1 index: ref=%d, ultra=%d, match=%s\n",
               top1_ref, top1_ultra, top1_ref == top1_ultra ? "YES" : "NO");
    }

    /* Test with typical attention scores (smaller range) */
    printf("\n=== Typical Attention Scores Test ===\n");
    for (int i = 0; i < M * Nc; i++) {
        S[i] = (rand() % 41) - 20;  // -20 to 20 (more typical after scale)
    }

    exp2_rows(S, P_ref, M, Nc, scale, max_val, Nc, Nc);
    exp2_ultra_rows(S, P_ultra, M, Nc, scale, -max_val, Nc, Nc);

    float total_max_err = 0;
    for (int i = 0; i < M; i++) {
        float sum_ref = 0, sum_ultra = 0;
        for (int j = 0; j < Nc; j++) {
            sum_ref += P_ref[i * Nc + j];
            sum_ultra += P_ultra[i * Nc + j];
        }

        float max_err = 0;
        for (int j = 0; j < Nc; j++) {
            float sm_ref = P_ref[i * Nc + j] / sum_ref;
            float sm_ultra = P_ultra[i * Nc + j] / sum_ultra;
            float err = fabsf(sm_ref - sm_ultra);
            if (err > max_err) max_err = err;
        }
        if (max_err > total_max_err) total_max_err = max_err;
        printf("Row %d: max softmax error = %.6f\n", i, max_err);
    }
    printf("\nOverall max softmax error: %.6f (%.2f%%)\n",
           total_max_err, total_max_err * 100);

    free(S);
    free(P_ref);
    free(P_ultra);

    return 0;
}
