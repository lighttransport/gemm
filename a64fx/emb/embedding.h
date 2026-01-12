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

#ifdef __cplusplus
}
#endif

#endif // EMBEDDING_H
