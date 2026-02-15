#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

extern void exp2_flash_tiled_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

// Static buffers
#define NC_MAX 256
static int32_t S[4*NC_MAX] __attribute__((aligned(64)));
static float V[NC_MAX*64] __attribute__((aligned(64)));
static float O[4*64] __attribute__((aligned(64)));
static float P[4*NC_MAX] __attribute__((aligned(64)));

int main() {
    int M = 4, D = 64;

    for (int Nc = 1; Nc <= 64; Nc++) {
        int ld_s = Nc;
        int ld_v = D * 4;
        int ld_o = D * 4;

        for (int i = 0; i < M * Nc; i++) S[i] = i % 100;
        for (int i = 0; i < Nc * D; i++) V[i] = 0.01f;
        for (int i = 0; i < M * D; i++) O[i] = 0;

        printf("Testing Nc=%d... ", Nc);
        fflush(stdout);
        exp2_flash_tiled_4x4(S, V, O, P, Nc, 0.1f, 5.0f, ld_s, ld_v, ld_o);
        printf("OK, P[0]=%f\n", P[0]);
    }

    return 0;
}
