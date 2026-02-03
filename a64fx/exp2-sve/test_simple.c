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
    int M = 4, Nc = 4, D = 64;
    int ld_s = Nc;  // S is [4][4]
    int ld_v = D * 4;
    int ld_o = D * 4;

    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));

    // Simple test: S[m][k] = k (so all rows have same values)
    // Row 0: S[0..3] = 0,1,2,3
    // Row 1: S[4..7] = 0,1,2,3
    // etc.
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < Nc; k++) {
            S[m * Nc + k] = k;
        }
    }
    
    // V = all 1s
    for (int i = 0; i < Nc * D; i++) V[i] = 1.0f;
    for (int i = 0; i < M * D; i++) O[i] = 0;
    for (int i = 0; i < M * Nc; i++) P[i] = -999.0f;

    float scale = 1.0f;  // simple scale
    float max_val = 0.0f; // no offset

    printf("S layout (M=%d, Nc=%d, ld_s=%d):\n", M, Nc, ld_s);
    for (int m = 0; m < M; m++) {
        printf("  Row %d:", m);
        for (int k = 0; k < Nc; k++) {
            printf(" %d", S[m * Nc + k]);
        }
        printf("\n");
    }
    
    printf("\nExpected exp2(S*scale - max) = exp2(S):\n");
    for (int m = 0; m < M; m++) {
        printf("  Row %d:", m);
        for (int k = 0; k < Nc; k++) {
            printf(" %.1f", exp2f((float)S[m*Nc+k] * scale - max_val));
        }
        printf("\n");
    }

    printf("\nCalling kernel...\n");
    exp2_flash_tiled_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);

    printf("Kernel P output:\n");
    for (int m = 0; m < M; m++) {
        printf("  Row %d:", m);
        for (int k = 0; k < Nc; k++) {
            printf(" %.1f", P[m * Nc + k]);
        }
        printf("\n");
    }

    free(S); free(V); free(O); free(P);
    return 0;
}
