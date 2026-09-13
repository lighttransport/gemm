/* SPDX-License-Identifier: MIT
 * Native policy/value network. All tensor interfaces are contiguous NHWC.
 * This implementation is independent of shogi engines and ML frameworks. */
#ifndef GEMM_GN_H
#define GEMM_GN_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct gn_model gn_model;
typedef struct {
    /* version 1 uses ReLU; version 2 uses SiLU for all configurable
     * activations. The version is checkpointed and changes graph semantics. */
    uint32_t version, side, inputs, actions, channels, blocks;
    uint32_t attention_every, head_dim, value_channels, value_hidden;
    uint64_t seed;
    size_t memory_limit;
} gn_config;
typedef struct {
    float policy, value, grad_norm;
    uint64_t step;
} gn_metrics;
gn_config gn_default_config(void);
/* backend: cpu (FP32), cuda, hip; cuda-fp32/hip-fp32 are diagnostic paths.
 * cuda-legacy/hip-legacy retain the original kernels for A/B checks.
 * hip-blaslt opts into an SDK-enabled hybrid: hipBLASLt for inference and
 * large long-K training matrices, native WMMA elsewhere. Build HIPBLASLT=1.
 * Full batch-16 gradient qualification is unresolved; see RDNA4.md.
 * cuda-int8/cuda-int16 are EXPERIMENTAL quantized-operand matrix paths, with
 * FP32 master weights/moments and non-matrix math. Neither is qualified for
 * full-network training; int16 uses four INT8 products, not native INT16 MMA.
 * hip-bf16: single BF16 product with FP32 accumulation during training too.
 * hip-fp16-blaslt: single FP16 product with FP32 accumulation. This SDK-only
 * path trades FP16 exponent range for three extra mantissa bits versus BF16.
 * hip-bf16x3: two BF16 components and three products in forward/backward.
 * cuda-bf16x3 is the analogous experimental three-product SM120 path.
 * hip-bf16x3-dx/dw/forward retain three products only in the named phases;
 * these asymmetric experiments currently require the -blaslt SDK build.
 * hip-bf16x3-fp16back-blaslt uses compensated BF16 forward, one-product
 * FP16/FP32 backward, and compensated linear dX propagation.
 * Append -fast for deterministic RX 9070 XT batch-64 hipBLASLt plans.
 * Append -tuned for benchmark-only timed hipBLASLt algorithm selection.
 * hip-bf16-mixed: six products forward (including inference), three backward.
 * All three accept an optional -blaslt suffix in an SDK-enabled build.
 * Their matrix accumulators remain FP32; they are not accuracy-qualified.
 * hip-bf16-acc / hip-bf16-acc128: native BF16 accumulators throughout each dot
 * or per K=128 partial, respectively (the latter widens partials to FP32).
 * hip-int8 / hip-int8-i64: INT8 WMMA with INT32 / widened INT64 partial sums.
 * hip-int16: four signed/unsigned INT8 products, INT32 partials, INT64 combine.
 * All reduced-precision HIP paths are EXPERIMENTAL and retain FP32 master,
 * optimizer, gradient storage, dequantization, and non-matrix arithmetic.
 * Unsupported backends fail explicitly; there is no silent CPU fallback. */
gn_model *gn_create(const gn_config *config, const char *backend, int device);
void gn_destroy(gn_model *model);
const char *gn_error(void);
const gn_config *gn_configuration(const gn_model *model);
size_t gn_parameter_count(const gn_model *model);
size_t gn_memory_used(const gn_model *model);
/* Useful matrix arithmetic of the current graph: forward (inference) or
 * forward+backward (training), including attention. Excludes padding,
 * precision compensation, optimizer and non-matrix operations. FMA = 2. */
double gn_matrix_flops(const gn_model *model);
/* Useful convolution/linear operations only, excluding FP32 attention. This
 * is an operation-equivalent count for quantized integer matrix backends. */
double gn_gemm_flops(const gn_model *model);
/* Actual operand-product operations selected by a backend for conv/linear
 * matrices, excluding padding and attention. Used for peak-rate accounting. */
double gn_gemm_product_ops(const gn_model *model, const char *backend);
uint64_t gn_step(const gn_model *model);
/* policy [batch, side*side, actions], wdl [batch,3] probabilities. */
int gn_infer(gn_model *, size_t batch, const float *input, float *policy, float *wdl);
/* Policy targets: -1 means illegal; nonnegative probabilities sum to one.
 * WDL targets: 0=win, 1=draw, 2=loss, from the player-to-move perspective.
 * backward accumulates parameter gradients; update averages accumulated batches
 * by sample count, clips global norm, and applies decoupled AdamW. */
int gn_backward(gn_model *, size_t batch, const float *input, const float *policy_targets,
                const uint32_t *wdl_targets, gn_metrics *metrics);
int gn_update(gn_model *, float learning_rate, float weight_decay, float clip_norm,
              gn_metrics *metrics);
void gn_zero_grad(gn_model *);
int gn_save(const gn_model *, const char *path);
gn_model *gn_load(const char *path, const char *backend, int device);
/* Diagnostics and independent reference validation. */
size_t gn_tensor_count(const gn_model *);
/* Replay sampling shares the checkpointed RNG state. */
uint64_t gn_random(gn_model *);
const char *gn_tensor(gn_model *, size_t index, size_t *rows, size_t *cols, float **data,
                      float **gradient);
#ifdef __cplusplus
}
#endif
#endif
