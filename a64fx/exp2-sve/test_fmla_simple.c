#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void exp2_fmla_fp32_vec(
    const int32_t* S, const float* V, float* O,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o);

int main() {
    int Nc = 64;
    int D = 64;
    int M = 16;
    
    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    
    for (int i = 0; i < M * Nc; i++) S[i] = 0;
    for (int i = 0; i < Nc * D; i++) V[i] = 1.0f;
    for (int i = 0; i < M * D; i++) O[i] = 0.0f;
    
    printf("Testing exp2_fmla_fp32_vec with Nc=%d, M=%d...\n", Nc, M);
    fflush(stdout);
    
    exp2_fmla_fp32_vec(S, V, O, Nc, 1.0f/64.0f, 0.0f, Nc, D*4, D*4);
    
    printf("Done! O[0]=%.3f\n", O[0]);
    
    free(S);
    free(V);
    free(O);
    return 0;
}
