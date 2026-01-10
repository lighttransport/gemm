#ifndef FLASH_ATTENTION_FUSED_V2_H
#define FLASH_ATTENTION_FUSED_V2_H

#include <stdint.h>
#include <stddef.h>

// Tile sizes optimized for A64FX L1D cache (64 KB)
#define FA_TILE_BR 48      // Q rows per tile (8 × 6-row microkernels)
#define FA_TILE_BC 64      // K/V columns per tile (1 × 64-col microkernel)

// Fixed-point scale constants
#define LOG2E_Q8 369       // log2(e) * 256 = 1.4427 * 256
#define SCORE_SCALE_128 23 // Approx 1/sqrt(128) * 256
#define SCORE_SCALE_256 16 // Approx 1/sqrt(256) * 256

// Online softmax state per row
typedef struct {
    int32_t m;          // Running max (Q8.8 fixed-point)
    int64_t l;          // Running sum (Q16.16 fixed-point)
    int32_t rescale;    // Rescale factor for O (Q16.16)
} flash_softmax_state_t;

// Main API: Fused FlashAttention forward pass
void flash_attention_fused_forward(
    const int8_t* Q,        // [L, head_dim] INT8 query matrix
    const int8_t* K,        // [L, head_dim] INT8 key matrix
    const int8_t* V,        // [L, head_dim] INT8 value matrix
    float* O,               // [L, head_dim] FP32 output
    float* logsumexp,       // [L] FP32 logsumexp (optional, NULL to skip)
    int64_t L,              // Sequence length
    int64_t head_dim);      // Head dimension (128 or 256)

// Version with pre-packed K/V (for GQA reuse across Q heads)
void flash_attention_fused_forward_packed(
    const int8_t* Q,        // [L, head_dim] INT8
    const int8_t* Kp,       // Packed K
    const int8_t* Vp,       // Packed V (transposed)
    float* O,               // [L, head_dim] FP32
    float* logsumexp,       // [L] FP32 (optional)
    int64_t L,
    int64_t head_dim);

// Packing functions
void pack_k_for_flash_attention(
    const int8_t* K,        // [L, head_dim] row-major
    int8_t* Kp,             // Packed output [L/64][head_dim/4][4][64]
    int64_t L,
    int64_t head_dim);

void pack_v_for_flash_attention(
    const int8_t* V,        // [L, head_dim] row-major
    int8_t* Vp,             // Packed output (transposed) [head_dim/64][L/4][4][64]
    int64_t L,
    int64_t head_dim);

// Size calculation helpers
static inline size_t flash_kp_size(int64_t L, int64_t head_dim) {
    int64_t L_tiles = (L + 63) / 64;
    int64_t K_groups = (head_dim + 3) / 4;
    return (size_t)(L_tiles * K_groups * 256);
}

static inline size_t flash_vp_size(int64_t L, int64_t head_dim) {
    int64_t head_tiles = (head_dim + 63) / 64;
    int64_t L_groups = (L + 3) / 4;
    return (size_t)(head_tiles * L_groups * 256);
}

// Aligned memory allocation (64-byte alignment for A64FX cache lines)
void* flash_aligned_alloc(size_t size);
void flash_aligned_free(void* ptr);

#endif // FLASH_ATTENTION_FUSED_V2_H
