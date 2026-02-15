#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

extern void exp2_flash_tiled_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

int main() {
    int M = 4, Nc = 8, D = 64;
    int ld_s = Nc;
    int ld_v = D * 4;
    int ld_o = D * 4;

    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));
    float* P_ref = aligned_alloc(64, M * Nc * sizeof(float));

    // Set specific test values
    for (int i = 0; i < M * Nc; i++) S[i] = i * 10; // 0, 10, 20, ...
    for (int i = 0; i < Nc * D; i++) V[i] = 1.0f;
    for (int i = 0; i < M * D; i++) O[i] = 0;
    for (int i = 0; i < M * Nc; i++) P[i] = 0;

    float scale = 0.1f;
    float max_val = 2.0f;

    // Reference exp2
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < Nc; k++) {
            float x = (float)S[m * ld_s + k] * scale - max_val;
            P_ref[m * Nc + k] = exp2f(x);
        }
    }

    printf("Reference P (first 8):\n");
    for (int i = 0; i < 8; i++) printf("  P_ref[%d] = %f (input=%d, x=%f)\n", 
        i, P_ref[i], S[i], (float)S[i]*scale - max_val);

    printf("\nCalling kernel...\n");
    exp2_flash_tiled_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);

    printf("Kernel P (first 8):\n");
    for (int i = 0; i < 8; i++) printf("  P[%d] = %f\n", i, P[i]);

    printf("\nO[0]=%f (expected=%f)\n", O[0], P_ref[0]*Nc);

    free(S); free(V); free(O); free(P); free(P_ref);
    return 0;
}
