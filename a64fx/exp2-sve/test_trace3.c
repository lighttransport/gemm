#include <stdio.h>
#include <stdint.h>
#include <arm_sve.h>

// Force no inlining
__attribute__((noinline))
extern void exp2_flash_tiled_4x4(
    const int32_t* S, const float* V, float* O, float* P,
    int Nc, float scale, float max_val,
    int ld_s, int ld_v, int ld_o
);

// Static buffers
static int32_t S[4*4] __attribute__((aligned(64)));
static float V[4*64] __attribute__((aligned(64)));
static float O[4*64] __attribute__((aligned(64)));
static float P[4*4] __attribute__((aligned(64)));

int main() {
    int M = 4, Nc = 4, D = 64;
    int ld_s = Nc;
    int ld_v = D * 4;
    int ld_o = D * 4;

    // Set S[m][k] = k for all m
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < Nc; k++) {
            S[m * Nc + k] = k;  // 0, 1, 2, 3 for each row
        }
    }
    for (int i = 0; i < Nc * D; i++) V[i] = 1.0f;
    for (int i = 0; i < M * D; i++) O[i] = 0;
    for (int i = 0; i < M * Nc; i++) P[i] = -999.0f;

    float scale = 1.0f;
    float max_val = 0.0f;

    // Force args to be in correct registers
    asm volatile("" ::: "memory");
    
    exp2_flash_tiled_4x4(S, V, O, P, Nc, scale, max_val, ld_s, ld_v, ld_o);

    asm volatile("" ::: "memory");
    
    printf("P output:\n");
    for (int m = 0; m < M; m++) {
        printf("  Row %d:", m);
        for (int k = 0; k < Nc; k++) {
            printf(" %.4f", P[m * Nc + k]);
        }
        printf("\n");
    }

    return 0;
}
