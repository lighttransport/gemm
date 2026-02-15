#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "exp2_fmla_fused.h"

int main() {
    int M = 4, Nc = 64, D = 64;
    
    printf("Allocating...\n");
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O_ref = aligned_alloc(64, M * D * sizeof(float));
    float* O_test = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));
    float* O_two = aligned_alloc(64, M * D * sizeof(float));
    
    printf("Initializing...\n");
    srand(42);
    for (int i = 0; i < M * Nc; i++) S[i] = (rand() % 201) - 100;
    for (int i = 0; i < Nc * D; i++) V[i] = ((float)(rand() % 1000) - 500) / 500.0f;
    
    float scale = 1.0f / 64.0f;
    float max_val = 1.5f;
    
    printf("Computing reference...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) O_ref[i * D + j] = 0.0f;
    }
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < Nc; k++) {
            float s = (float)S[i * Nc + k];
            float p = exp2f(s * scale - max_val);
            for (int j = 0; j < D; j++) {
                O_ref[i * D + j] += p * V[k * D + j];
            }
        }
    }
    printf("Reference O[0:4]: %.3f %.3f %.3f %.3f\n", 
           O_ref[0], O_ref[1], O_ref[2], O_ref[3]);
    
    printf("Testing exp2_fmla_fp32_4x4...\n");
    memset(O_test, 0, M * D * sizeof(float));
    exp2_fmla_fp32_4x4(S, V, O_test, Nc, scale, max_val, Nc, D*4, D*4);
    printf("Test O[0:4]: %.3f %.3f %.3f %.3f\n",
           O_test[0], O_test[1], O_test[2], O_test[3]);
    
    printf("Testing two-pass...\n");
    exp2_rows(S, P, M, Nc, scale, max_val, Nc, Nc);
    printf("P[0:4]: %.6f %.6f %.6f %.6f\n", P[0], P[1], P[2], P[3]);
    
    memset(O_two, 0, M * D * sizeof(float));
    gemm_fp32_4x4(P, V, O_two, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    printf("Two-pass O[0:4]: %.3f %.3f %.3f %.3f\n",
           O_two[0], O_two[1], O_two[2], O_two[3]);
    
    printf("Done!\n");
    
    free(S);
    free(V);
    free(O_ref);
    free(O_test);
    free(P);
    free(O_two);
    return 0;
}
