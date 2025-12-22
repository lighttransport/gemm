#ifndef FLASH_ATTN_2PASS_H
#define FLASH_ATTN_2PASS_H

#include <stdint.h>
#include <stddef.h>

// ============================================
// Tile parameters
// ============================================
// From OoO test: FMA density is critical (41% efficiency at 49% FMA ratio)
// Target: maximize FMA ratio, minimize auxiliary instructions

#define BR 4        // Query block rows
#define BC 64       // Key/Value block cols
#define D  64       // Head dimension
#define VL 16       // SVE vector length (floats)

// FLOPs analysis:
// Pass 1 (S = Q @ K^T): BR * BC * D * 2 = 4 * 64 * 64 * 2 = 32,768
// Pass 2 (O = P @ V):   BR * D * BC * 2 = 4 * 64 * 64 * 2 = 32,768
// exp:                  BR * BC * ~8 ops = 4 * 64 * 8 = 2,048
// Total: ~67,584 FLOPs per tile

#define ALIGN 64

// ============================================
// ASM Kernel prototypes
// ============================================

// Pass 1: S[4,64] = Q[4,64] @ K[64,64]^T + row_max
extern void pass1_qkt_rowmax(
    const float* Q,     // [4, 64]
    const float* K,     // [64, 64]
    float* S,           // [4, 64] output
    float* m            // [4] row max output
);

// Pass 2: O[4,64] = softmax(S[4,64]) @ V[64,64]
extern void pass2_softmax_pv(
    const float* S,     // [4, 64]
    const float* V,     // [64, 64]
    const float* m,     // [4] row max
    float* O,           // [4, 64] output
    float* l            // [4] row sum output
);

// ============================================
// C wrappers
// ============================================

void flash_attention_tile(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* S_scratch,
    float* m,
    float* l
);

void flash_attention_ref(
    const float* Q,
    const float* K,
    const float* V,
    float* O
);

void normalize_output(float* O, const float* l);

#endif
