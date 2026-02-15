// flash_attention_int16.h
// Flash Attention with INT16 attention weights for A64FX SVE
//
// Pipeline:
// 1. Q@K^T: INT8 SDOT -> INT32 accumulator
// 2. Softmax: INT32 -> FP32 -> fast approximation
// 3. Quantize: FP32 -> INT16 attention weights
// 4. P@V: INT16 SDOT with V INT8->INT16 widening load
//
// Benefits:
// - INT16 attention weights preserve more precision than INT8
// - V widening load enables INT16 SDOT without repacking V
// - Fast piecewise linear softmax avoids expensive exp/div

#ifndef FLASH_ATTENTION_INT16_H
#define FLASH_ATTENTION_INT16_H

#include <stdint.h>
#include <stddef.h>

// Tile sizes optimized for A64FX L1D cache (64 KB)
#define FA16_TILE_BR 32     // Q rows per tile
#define FA16_TILE_BC 64     // K/V columns per tile (1 SVE vector width for INT16)

// Scale factors
#define FA16_SOFTMAX_SCALE 32767  // INT16 max for normalized softmax
#define FA16_LOG2E_Q8 369         // log2(e) * 256

// Online softmax state per row
typedef struct {
    float m;        // Running max (FP32)
    float l;        // Running sum of exp (FP32)
    float rescale;  // Rescale factor for O (FP32)
} flash16_softmax_state_t;

// Main API: Flash Attention with INT16 attention weights
void flash_attention_int16_forward(
    const int8_t* Q,        // [L, head_dim] INT8 query
    const int8_t* K,        // [L, head_dim] INT8 key
    const int8_t* V,        // [L, head_dim] INT8 value
    float* O,               // [L, head_dim] FP32 output
    float* logsumexp,       // [L] FP32 logsumexp (optional, NULL to skip)
    int64_t L,              // Sequence length
    int64_t head_dim,       // Head dimension (typically 128)
    float scale);           // Attention scale (typically 1/sqrt(head_dim))

// Packing functions
void pack_k_int16(
    const int8_t* K,        // [L, head_dim] row-major
    int8_t* Kp,             // Packed for Q@K^T INT8 SDOT
    int64_t L,
    int64_t head_dim);

void pack_v_int16(
    const int8_t* V,        // [L, head_dim] row-major
    int8_t* Vp,             // Packed for P@V with INT16 widening
    int64_t L,
    int64_t head_dim);

// Memory helpers
void* flash16_aligned_alloc(size_t size);
void flash16_aligned_free(void* ptr);

// Size calculation
static inline size_t flash16_workspace_size(int64_t L, int64_t head_dim) {
    // S_chunk: BR × BC × 4 bytes (INT32)
    // P_chunk: BR × BC × 2 bytes (INT16)
    // O_acc: BR × head_dim × 4 bytes (INT32)
    size_t s_size = FA16_TILE_BR * FA16_TILE_BC * sizeof(int32_t);
    size_t p_size = FA16_TILE_BR * FA16_TILE_BC * sizeof(int16_t);
    size_t o_size = FA16_TILE_BR * head_dim * sizeof(int32_t);
    return s_size + p_size + o_size + 256;  // Extra for alignment
}

#endif // FLASH_ATTENTION_INT16_H
