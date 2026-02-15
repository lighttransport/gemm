#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern void exp2_rows(const int32_t* S, float* P, int M, int Nc, 
                      float scale, float max_val, int ld_s, int ld_p);
extern void gemm_fp32_4x4(const float* A, const float* B, float* O,
                          int K, int ld_a, int ld_b, int ld_o);

int main() {
    printf("Test baseline (exp2_rows + gemm_fp32_4x4)\n");

    int M = 4, Nc = 64, D = 64;
    int ld_s = Nc;
    int ld_v = D * 4;
    int ld_o = D * 4;
    int ld_a = Nc * 4;

    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));

    for (int i = 0; i < M * Nc; i++) S[i] = i % 100;
    for (int i = 0; i < Nc * D; i++) V[i] = 0.01f;
    for (int i = 0; i < M * D; i++) O[i] = 0;
    for (int i = 0; i < M * Nc; i++) P[i] = 0;

    printf("Calling exp2_rows...\n");
    fflush(stdout);
    exp2_rows(S, P, M, Nc, 0.125f, 5.0f, ld_s, Nc);
    printf("P[0]=%f\n", P[0]);

    printf("Calling gemm_fp32_4x4...\n");
    fflush(stdout);
    gemm_fp32_4x4(P, V, O, Nc, ld_a, ld_v, ld_o);
    printf("O[0]=%f\n", O[0]);

    free(S); free(V); free(O); free(P);
    return 0;
}
