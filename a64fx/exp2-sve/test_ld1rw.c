#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arm_sve.h>

extern void exp2_flash_tiled_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

int main() {
    printf("Test LD1RW kernel\n");

    int M = 4, Nc = 8, D = 64;
    int ld_s = Nc;
    int ld_v = D * 4;
    int ld_o = D * 4;

    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));

    if (!S || !V || !O || !P) {
        printf("Alloc failed\n");
        return 1;
    }

    for (int i = 0; i < M * Nc; i++) S[i] = i;
    for (int i = 0; i < Nc * D; i++) V[i] = 0.01f;
    for (int i = 0; i < M * D; i++) O[i] = 0;
    for (int i = 0; i < M * Nc; i++) P[i] = 0;

    printf("Calling kernel...\n");
    fflush(stdout);

    exp2_flash_tiled_4x4(S, V, O, P, Nc, 0.125f, 5.0f, ld_s, ld_v, ld_o);

    printf("Done. O[0]=%f, P[0]=%f\n", O[0], P[0]);

    free(S); free(V); free(O); free(P);
    return 0;
}
