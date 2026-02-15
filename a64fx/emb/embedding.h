// embedding.h
// SVE-optimized Embedding Layer for A64FX
//
// Supports:
// - Forward pass (token lookup)
// - Backward pass (gradient accumulation)
// - Combined token + position embedding
// - FP32 and FP64 precision

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Assembly Implementations (embedding_asm.S)
//=============================================================================

// Forward pass: output[i, :] = emb_table[indices[i], :]
void embedding_fwd_f32_asm(const int32_t* indices, const float* emb_table,
                           float* output, size_t batch_size, size_t hidden_dim);

// Forward pass batched (processes 4 tokens at once for better bandwidth)
void embedding_fwd_f32_batched_asm(const int32_t* indices, const float* emb_table,
                                   float* output, size_t batch_size, size_t hidden_dim);

// Backward pass: grad_embedding[indices[i], :] += grad_output[i, :]
// Note: Assumes no duplicate indices in batch for correctness
void embedding_bwd_f32_asm(const int32_t* indices, const float* grad_output,
                           float* grad_embedding, size_t batch_size,
                           size_t hidden_dim, size_t vocab_size);

// Forward pass FP64
void embedding_fwd_f64_asm(const int32_t* indices, const double* emb_table,
                           double* output, size_t batch_size, size_t hidden_dim);

// Backward pass FP64
void embedding_bwd_f64_asm(const int32_t* indices, const double* grad_output,
                           double* grad_embedding, size_t batch_size,
                           size_t hidden_dim, size_t vocab_size);

// Combined token + position embedding
// output[i, :] = token_emb[token_ids[i], :] + pos_emb[position_ids[i], :]
void embedding_fwd_with_pos_f32_asm(const int32_t* token_ids,
                                    const int32_t* position_ids,
                                    const float* token_emb,
                                    const float* pos_emb,
                                    float* output,
                                    size_t batch_size, size_t hidden_dim);

// Zero-initialize gradient table
void embedding_grad_zero_f32_asm(float* grad_embedding,
                                 size_t vocab_size, size_t hidden_dim);

// Backward pass with sorted indices (handles duplicates correctly)
// Requires pre-sorted indices and original position mapping
void embedding_bwd_sorted_f32_asm(const int32_t* sorted_indices,
                                  const int32_t* original_positions,
                                  const float* grad_output,
                                  float* grad_embedding,
                                  size_t batch_size, size_t hidden_dim);

// Forward pass: SVE gather/scatter (16 tokens column-wise)
// Constraint: vocab_size * hidden_dim < 2^32
void embedding_fwd_f32_gather_asm(const int32_t* indices, const float* emb_table,
                                   float* output, size_t seq_len, size_t hidden_dim);

// Forward pass: deep-prefetch stream (8x unroll, prefetch 8 rows ahead)
void embedding_fwd_f32_stream_asm(const int32_t* indices, const float* emb_table,
                                   float* output, size_t seq_len, size_t hidden_dim);

// Forward pass: sorted core (contiguous read, indirect write)
// Called after counting sort; sorted_indices[i] = emb index, sorted_order[i] = output pos
void embedding_fwd_f32_sorted_core_asm(const int32_t* sorted_indices,
                                        const int32_t* sorted_order,
                                        const float* emb_table,
                                        float* output,
                                        size_t count, size_t hidden_dim);

// Forward pass: stream with index prefetch + deeper row prefetch (16 ahead)
void embedding_fwd_f32_stream_ipf_asm(const int32_t* indices, const float* emb_table,
                                       float* output, size_t seq_len, size_t hidden_dim);

// Scatter one embedding row to multiple output positions (for dedup)
// Loads row in register-cached chunks, stores to all positions per chunk.
void embedding_fwd_f32_scatter_row_asm(const float* src_row, float* output_base,
                                        const int32_t* positions,
                                        size_t n_positions, size_t hidden_dim);

// Forward pass FP16: deep-prefetch stream (8x unroll, prefetch 8 rows ahead)
// Same strategy as FP32 stream but for __fp16 (2 bytes per element)
void embedding_fwd_f16_stream_asm(const int32_t* indices, const void* emb_table,
                                   void* output, size_t seq_len, size_t hidden_dim);

//=============================================================================
// C Reference Implementations
//=============================================================================

static inline void embedding_fwd_f32_ref(const int32_t* indices,
                                         const float* emb_table,
                                         float* output,
                                         size_t batch_size,
                                         size_t hidden_dim)
{
    for (size_t i = 0; i < batch_size; i++) {
        int32_t idx = indices[i];
        const float* src = emb_table + (size_t)idx * hidden_dim;
        float* dst = output + i * hidden_dim;
        for (size_t j = 0; j < hidden_dim; j++) {
            dst[j] = src[j];
        }
    }
}

static inline void embedding_bwd_f32_ref(const int32_t* indices,
                                         const float* grad_output,
                                         float* grad_embedding,
                                         size_t batch_size,
                                         size_t hidden_dim,
                                         size_t vocab_size)
{
    (void)vocab_size;  // Not used in simple implementation
    for (size_t i = 0; i < batch_size; i++) {
        int32_t idx = indices[i];
        const float* src = grad_output + i * hidden_dim;
        float* dst = grad_embedding + (size_t)idx * hidden_dim;
        for (size_t j = 0; j < hidden_dim; j++) {
            dst[j] += src[j];
        }
    }
}

static inline void embedding_fwd_f64_ref(const int32_t* indices,
                                         const double* emb_table,
                                         double* output,
                                         size_t batch_size,
                                         size_t hidden_dim)
{
    for (size_t i = 0; i < batch_size; i++) {
        int32_t idx = indices[i];
        const double* src = emb_table + (size_t)idx * hidden_dim;
        double* dst = output + i * hidden_dim;
        for (size_t j = 0; j < hidden_dim; j++) {
            dst[j] = src[j];
        }
    }
}

static inline void embedding_bwd_f64_ref(const int32_t* indices,
                                         const double* grad_output,
                                         double* grad_embedding,
                                         size_t batch_size,
                                         size_t hidden_dim,
                                         size_t vocab_size)
{
    (void)vocab_size;
    for (size_t i = 0; i < batch_size; i++) {
        int32_t idx = indices[i];
        const double* src = grad_output + i * hidden_dim;
        double* dst = grad_embedding + (size_t)idx * hidden_dim;
        for (size_t j = 0; j < hidden_dim; j++) {
            dst[j] += src[j];
        }
    }
}

static inline void embedding_fwd_with_pos_f32_ref(const int32_t* token_ids,
                                                  const int32_t* position_ids,
                                                  const float* token_emb,
                                                  const float* pos_emb,
                                                  float* output,
                                                  size_t batch_size,
                                                  size_t hidden_dim)
{
    for (size_t i = 0; i < batch_size; i++) {
        int32_t tok_idx = token_ids[i];
        int32_t pos_idx = position_ids[i];
        const float* tok_src = token_emb + (size_t)tok_idx * hidden_dim;
        const float* pos_src = pos_emb + (size_t)pos_idx * hidden_dim;
        float* dst = output + i * hidden_dim;
        for (size_t j = 0; j < hidden_dim; j++) {
            dst[j] = tok_src[j] + pos_src[j];
        }
    }
}

static inline void embedding_grad_zero_f32_ref(float* grad_embedding,
                                               size_t vocab_size,
                                               size_t hidden_dim)
{
    size_t total = vocab_size * hidden_dim;
    for (size_t i = 0; i < total; i++) {
        grad_embedding[i] = 0.0f;
    }
}

//=============================================================================
// Counting sort helper for sorted embedding lookup
//=============================================================================
static inline void counting_sort_indices(const int32_t* indices, size_t n,
                                          int32_t* sorted_indices,
                                          int32_t* sorted_order,
                                          size_t vocab_size)
{
    int32_t* counts = (int32_t*)calloc(vocab_size, sizeof(int32_t));
    if (!counts) return;

    // Count occurrences
    for (size_t i = 0; i < n; i++)
        counts[indices[i]]++;

    // Prefix sum
    int32_t total = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        int32_t c = counts[i];
        counts[i] = total;
        total += c;
    }

    // Build sorted arrays
    for (size_t i = 0; i < n; i++) {
        int32_t idx = indices[i];
        int32_t pos = counts[idx]++;
        sorted_order[pos] = (int32_t)i;   // original position
        sorted_indices[pos] = idx;          // embedding index
    }

    free(counts);
}

//=============================================================================
// Sorted embedding forward: counting sort + contiguous-read ASM copy
//=============================================================================
static inline void embedding_fwd_f32_sorted(const int32_t* indices,
                                             const float* emb_table,
                                             float* output,
                                             size_t seq_len, size_t hidden_dim,
                                             size_t vocab_size)
{
    int32_t* sorted_indices = (int32_t*)malloc(seq_len * sizeof(int32_t));
    int32_t* sorted_order = (int32_t*)malloc(seq_len * sizeof(int32_t));
    if (!sorted_indices || !sorted_order) {
        free(sorted_indices);
        free(sorted_order);
        return;
    }

    counting_sort_indices(indices, seq_len, sorted_indices, sorted_order, vocab_size);

    embedding_fwd_f32_sorted_core_asm(sorted_indices, sorted_order,
                                       emb_table, output, seq_len, hidden_dim);

    free(sorted_indices);
    free(sorted_order);
}

//=============================================================================
// Deduplicated embedding forward: sort, then load each unique row once
// and scatter to all output positions via register-cached scatter kernel.
// Saves (N-1) row loads per group of N duplicate indices.
//=============================================================================
static inline void embedding_fwd_f32_dedup(const int32_t* indices,
                                            const float* emb_table,
                                            float* output,
                                            size_t seq_len, size_t hidden_dim,
                                            size_t vocab_size)
{
    int32_t* sorted_indices = (int32_t*)malloc(seq_len * sizeof(int32_t));
    int32_t* sorted_order   = (int32_t*)malloc(seq_len * sizeof(int32_t));
    if (!sorted_indices || !sorted_order) {
        free(sorted_indices);
        free(sorted_order);
        return;
    }

    counting_sort_indices(indices, seq_len, sorted_indices, sorted_order, vocab_size);

    // Process groups of identical indices
    size_t i = 0;
    while (i < seq_len) {
        int32_t idx = sorted_indices[i];
        const float* src = emb_table + (size_t)idx * hidden_dim;

        // Find group end
        size_t j = i + 1;
        while (j < seq_len && sorted_indices[j] == idx) j++;

        // Scatter this row to all positions in the group
        embedding_fwd_f32_scatter_row_asm(src, output,
                                           sorted_order + i,
                                           j - i, hidden_dim);
        i = j;
    }

    free(sorted_indices);
    free(sorted_order);
}

#ifdef __cplusplus
}
#endif

#endif // EMBEDDING_H
