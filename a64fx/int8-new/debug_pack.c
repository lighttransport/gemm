#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "gemm_pack.h"

int main() {
    const int M = 5;
    const int N = 64;
    const int K = 256;

    // Allocate and initialize A (all 1s)
    int8_t* A = (int8_t*)malloc(M * K);
    for (int i = 0; i < M * K; i++) A[i] = 1;

    // Allocate and initialize B (all 1s)
    int8_t* B = (int8_t*)malloc(N * K);
    for (int i = 0; i < N * K; i++) B[i] = 1;

    // Pack A
    int8_t* Apack = (int8_t*)aligned_alloc(256, M * K);
    pack_A_5x256(A, K, Apack, M);

    // Pack B
    int8_t* Bpack = (int8_t*)aligned_alloc(256, N * K);
    pack_B_64x256(B, K, Bpack, N);

    // Check A packing (first 16 bytes of each row)
    printf("Apack (first 16 bytes per row):\n");
    for (int m = 0; m < 5; m++) {
        printf("  Row %d: ", m);
        for (int k = 0; k < 16; k++) {
            printf("%3d ", (int)Apack[m * K + k]);
        }
        printf("\n");
    }
    printf("\n");

    // Check B packing (first k-group, all 4 vectors)
    printf("Bpack (k=0 group, showing first 4 bytes of each column):\n");
    printf("Vec 0 (cols 0-15):\n");
    for (int col = 0; col < 16; col++) {
        int offset = col * 4;  // Each column has 4 bytes
        printf("  Col %2d: [%d,%d,%d,%d]\n", col,
               (int)Bpack[offset], (int)Bpack[offset+1],
               (int)Bpack[offset+2], (int)Bpack[offset+3]);
    }

    printf("\nVec 1 (cols 16-31):\n");
    for (int col = 0; col < 16; col++) {
        int offset = 64 + col * 4;  // Vec 1 starts at byte 64
        printf("  Col %2d: [%d,%d,%d,%d]\n", 16 + col,
               (int)Bpack[offset], (int)Bpack[offset+1],
               (int)Bpack[offset+2], (int)Bpack[offset+3]);
    }

    printf("\nVec 2 (cols 32-47):\n");
    for (int col = 0; col < 16; col++) {
        int offset = 128 + col * 4;  // Vec 2 starts at byte 128
        printf("  Col %2d: [%d,%d,%d,%d]\n", 32 + col,
               (int)Bpack[offset], (int)Bpack[offset+1],
               (int)Bpack[offset+2], (int)Bpack[offset+3]);
    }

    printf("\nVec 3 (cols 48-63):\n");
    for (int col = 0; col < 16; col++) {
        int offset = 192 + col * 4;  // Vec 3 starts at byte 192
        printf("  Col %2d: [%d,%d,%d,%d]\n", 48 + col,
               (int)Bpack[offset], (int)Bpack[offset+1],
               (int)Bpack[offset+2], (int)Bpack[offset+3]);
    }

    // Also check bytes in row 4 around positions that might be problematic
    printf("\nApack row 4 detailed (bytes 0-63):\n");
    for (int k = 0; k < 64; k += 16) {
        printf("  Bytes %3d-%3d: ", k, k+15);
        for (int i = 0; i < 16; i++) {
            printf("%3d ", (int)Apack[4 * K + k + i]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(Apack);
    free(Bpack);

    return 0;
}
