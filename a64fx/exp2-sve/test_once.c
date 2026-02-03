#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern void exp2_flash_tiled_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

int main() {
    int M = 4, Nc = 64, D = 64;
    int ld_s = Nc;
    int ld_v = D * 4;
    int ld_o = D * 4;

    int32_t* S = aligned_alloc(64, M * Nc * sizeof(int32_t));
    float* V = aligned_alloc(64, Nc * D * sizeof(float));
    float* O = aligned_alloc(64, M * D * sizeof(float));
    float* P = aligned_alloc(64, M * Nc * sizeof(float));

    for (int i = 0; i < M * Nc; i++) S[i] = i % 100;
    for (int i = 0; i < Nc * D; i++) V[i] = 0.01f;
    for (int i = 0; i < M * D; i++) O[i] = 0;

    printf("Calling kernel once...\n");
    exp2_flash_tiled_4x4(S, V, O, P, Nc, 0.1f, 5.0f, ld_s, ld_v, ld_o);
    printf("P[0]=%f, O[0]=%f\n", P[0], O[0]);

    printf("Calling again...\n");
    exp2_flash_tiled_4x4(S, V, O, P, Nc, 0.1f, 5.0f, ld_s, ld_v, ld_o);
    printf("P[0]=%f, O[0]=%f\n", P[0], O[0]);

    printf("Calling 100 times...\n");
    for (int i = 0; i < 100; i++) {
        exp2_flash_tiled_4x4(S, V, O, P, Nc, 0.1f, 5.0f, ld_s, ld_v, ld_o);
    }
    printf("Done.\n");

    printf("Freeing S...\n");
    free(S);
    printf("Freeing V...\n");
    free(V);
    printf("Freeing O...\n");
    free(O);
    printf("Freeing P...\n");
    free(P);
    printf("All freed.\n");

    return 0;
}
