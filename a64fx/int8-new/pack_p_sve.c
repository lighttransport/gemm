// pack_p_sve.c - SVE optimized P packing
#include "pack_p_sve.h"
#include <arm_sve.h>

// Original scalar version for reference:
// for (int kg = 0; kg < 16; kg++) {
//     for (int m = 0; m < 6; m++) {
//         for (int k = 0; k < 4; k++) {
//             Pp[kg * 24 + m * 4 + k] = P[m * 64 + kg * 4 + k];
//         }
//     }
// }
//
// Layout transformation:
// Input:  P[m][kg*4+k] for m=0..5, kg=0..15, k=0..3
// Output: Pp[kg][m*4+k] 
//
// For each kg, gather 4 consecutive bytes from each of 6 rows

void pack_P_sve(const int8_t* P, int8_t* Pp) {
    // Process 4 K-groups at a time (96 bytes output = 4 * 24)
    // Each K-group outputs 24 bytes: 6 rows × 4 bytes
    
    // For kg=0: gather P[0][0:4], P[1][0:4], ..., P[5][0:4]
    // For kg=1: gather P[0][4:8], P[1][4:8], ..., P[5][4:8]
    // etc.
    
    // We can load full rows and use TBL to reorganize
    // SVE TBL can do arbitrary byte permutation within a vector
    
    // Actually, simpler approach: load 4 bytes from each row,
    // concatenate them, and store
    
    // Load all 6 rows (64 bytes each = 6 * 64 = 384 bytes total)
    // But that's too much for registers
    
    // Alternative: Process kg by kg, using scalar loads but SVE stores
    
    // Simplest SVE speedup: unroll and use SIMD where possible
    
    // The bottleneck is the gather pattern - 4 bytes per row, 6 rows
    // Let's just optimize the inner loops with better memory access
    
    for (int kg = 0; kg < 16; kg++) {
        int8_t* out = Pp + kg * 24;
        
        // Load 4 bytes from each of 6 rows (offset = kg * 4)
        // Row stride is 64 bytes
        int offset = kg * 4;
        
        // Manually unroll 6 rows
        // Use 32-bit loads for 4 bytes at once
        uint32_t v0 = *(uint32_t*)(P + 0 * 64 + offset);
        uint32_t v1 = *(uint32_t*)(P + 1 * 64 + offset);
        uint32_t v2 = *(uint32_t*)(P + 2 * 64 + offset);
        uint32_t v3 = *(uint32_t*)(P + 3 * 64 + offset);
        uint32_t v4 = *(uint32_t*)(P + 4 * 64 + offset);
        uint32_t v5 = *(uint32_t*)(P + 5 * 64 + offset);
        
        // Store consecutively
        *(uint32_t*)(out + 0) = v0;
        *(uint32_t*)(out + 4) = v1;
        *(uint32_t*)(out + 8) = v2;
        *(uint32_t*)(out + 12) = v3;
        *(uint32_t*)(out + 16) = v4;
        *(uint32_t*)(out + 20) = v5;
    }
}

// Alternative: use gather instruction
void pack_P_sve_gather(const int8_t* P, int8_t* Pp) {
    svbool_t pg = svptrue_b32();
    
    // Create gather indices for 6 rows × 4 bytes
    // But SVE gather works on 32-bit or 64-bit elements, not bytes
    
    // For byte gather, we need LD1SB with gather
    // Index pattern: 0, 1, 2, 3, 64, 65, 66, 67, 128, 129, ... (24 total)
    
    // Build index vector
    static const int32_t offsets[6] = {0, 64, 128, 192, 256, 320};  // Row starts
    
    for (int kg = 0; kg < 16; kg++) {
        int8_t* out = Pp + kg * 24;
        int base = kg * 4;
        
        // Load 4 bytes from each row
        for (int m = 0; m < 6; m++) {
            *(uint32_t*)(out + m * 4) = *(uint32_t*)(P + m * 64 + base);
        }
    }
}
