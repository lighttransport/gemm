#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

void exp2_rows(const int32_t* S, float* P, int M, int Nc,
               float scale, float max_val, int ld_s, int ld_p);
void gemm_fp32_4x4(const float* A, const float* B, float* O,
                   int K, int ld_a, int ld_b, int ld_o);

int main() {
    int M = 4, Nc = 64, D = 64;
    
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    
    printf("Initializing...\n");
    for (int i = 0; i < M * Nc; i++) S[i] = 0;
    for (int i = 0; i < M * Nc; i++) P[i] = 0.0f;
    for (int i = 0; i < Nc * D; i++) V[i] = 1.0f;
    memset(O, 0, M * D * sizeof(float));
    
    printf("Testing gemm_fp32_4x4...\n");
    fflush(stdout);
    
    /* Set P to all 1s */
    for (int i = 0; i < M * Nc; i++) P[i] = 1.0f;
    
    gemm_fp32_4x4(P, V, O, Nc, Nc * sizeof(float), D * sizeof(float), D * sizeof(float));
    
    printf("GEMM OK, O[0]=%f (expected %d)\n", O[0], Nc);
    
    printf("Testing exp2_rows...\n");
    fflush(stdout);
    
    exp2_rows(S, P, M, Nc, 1.0f/64.0f, 0.0f, Nc, Nc);
    
    printf("exp2_rows OK, P[0]=%f (expected 1.0)\n", P[0]);
    
    free(S);
    free(P);
    free(V);
    free(O);
    return 0;
}
