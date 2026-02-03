#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern void exp2_flash_tiled_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

int main() {
    int M = 4, Nc = 4, D = 64;
    int ld_s = Nc;
    int ld_v = D * 4;
    int ld_o = D * 4;

    // Static allocation to avoid heap issues
    static int32_t S[4*4];
    static float V[4*64];
    static float O[4*64];
    static float P[4*4];

    for (int i = 0; i < M * Nc; i++) S[i] = i % 10;
    for (int i = 0; i < Nc * D; i++) V[i] = 0.01f;

    printf("Before kernel\n");
    exp2_flash_tiled_4x4(S, V, O, P, Nc, 0.1f, 5.0f, ld_s, ld_v, ld_o);
    printf("After kernel, P[0]=%f\n", P[0]);

    return 0;
}
