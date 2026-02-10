#ifndef CROSS_ENTROPY_SVE_H
#define CROSS_ENTROPY_SVE_H

#include <stdint.h>

/* Forward: scalar loss for one token position
 * loss = -logits[target] + max(logits) + log(sum(exp(logits - max)))
 * 2-pass with SW prefetch + 8× unroll.
 */
float cross_entropy_fwd_f32(const float *logits, int target, int V);

/* Forward with fp16 input (computes in fp32), 4× fp16-vec unrolled */
float cross_entropy_fwd_f16(const uint16_t *logits_f16, int target, int V);

/* Blocked forward: fuses max + exp+sum into 1 L2 pass via blocking.
 * 1 L2 read + 1 L1 read per element (vs 2 L2 reads in 2-pass).
 */
float cross_entropy_fwd_blocked_f32(const float *logits, int target, int V);

/* Forward batch: OpenMP over tokens */
void cross_entropy_batch_f32(const float *logits, const int *targets,
                             float *losses, int batch_tokens, int V);

/* Backward: grad[i] = softmax(logits)_i - 1{i==target}
 * Requires max_val and sum_exp from forward pass.
 * 8× unrolled + SW prefetch.
 */
void cross_entropy_bwd_f32(const float *logits, int target, int V,
                           float max_val, float sum_exp, float *grad);

/* Combined forward + backward (3-pass: max, exp+sum, grad)
 * Returns loss, writes grad[0..V-1].
 * 8× unrolled + SW prefetch on all passes.
 */
float cross_entropy_fwd_bwd_f32(const float *logits, int target, int V,
                                float *grad);

/* Blocked forward + backward:
 * Phase 1: blocked max+exp+sum (1 L2 pass via blocking)
 * Phase 2: gradient pass (1 L2 read + 1 write)
 * Total: 2 L2 reads + 1 write (vs 3+1 in 3-pass)
 */
float cross_entropy_fwd_bwd_blocked_f32(const float *logits, int target, int V,
                                         float *grad);

/* Blocked forward with FP16 input (1 L2 pass via blocking, fp16→fp32) */
float cross_entropy_fwd_blocked_f16(const uint16_t *logits_f16, int target, int V);

/* Backward with FP16 input: grad[i] = softmax_i - 1{i==target}
 * Reads fp16 logits, computes in fp32, writes fp32 grad.
 */
void cross_entropy_bwd_f16(const uint16_t *logits_f16, int target, int V,
                           float max_val, float sum_exp, float *grad);

/* Combined forward + backward with FP16 input (3-pass)
 * Returns loss, writes fp32 grad[0..V-1].
 */
float cross_entropy_fwd_bwd_f16(const uint16_t *logits_f16, int target, int V,
                                float *grad);

/* Blocked forward + backward with FP16 input:
 * Phase 1: blocked max+exp+sum (1 L2 pass, fp16→fp32)
 * Phase 2: streaming grad (fp16 read + fp32 write)
 */
float cross_entropy_fwd_bwd_blocked_f16(const uint16_t *logits_f16, int target,
                                         int V, float *grad);

/* Batch forward (blocked) with OpenMP */
void cross_entropy_batch_blocked_f32(const float *logits, const int *targets,
                                      float *losses, int batch_tokens, int V);

/* Batch forward (FP16) with OpenMP */
void cross_entropy_batch_f16(const uint16_t *logits_f16, const int *targets,
                              float *losses, int batch_tokens, int V);

/* Scalar baseline for accuracy comparison */
float cross_entropy_scalar_f32(const float *logits, int target, int V);
double cross_entropy_ref_f64(const float *logits, int target, int V);

#endif /* CROSS_ENTROPY_SVE_H */
