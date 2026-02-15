/*
 * SVE exp2 kernel using FEXPA instruction
 *
 * Header file for C integration
 */

#ifndef EXP2_FEXPA_H
#define EXP2_FEXPA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * exp2_fexpa_softmax: Compute exp2((x - max) * scale) for softmax
 *
 * @param in      Input int32 values
 * @param out     Output float values (exp2 results)
 * @param n       Number of elements
 * @param scale   Scale factor (typically 1/sqrt(D) * log2(e))
 * @param max_val Maximum value for numerical stability
 *
 * Computes: out[i] = exp2((in[i] - max_val) * scale)
 * Used for softmax: exp(x - max) = exp2((x - max) * log2(e))
 */
void exp2_fexpa_softmax(
    const int32_t* in,
    float* out,
    int n,
    float scale,
    int32_t max_val
);

/*
 * exp2_fexpa_simple: Simple exp2 for float input
 *
 * @param in  Input float values
 * @param out Output float values (exp2 results)
 * @param n   Number of elements
 *
 * Computes: out[i] = exp2(in[i])
 */
void exp2_fexpa_simple(
    const float* in,
    float* out,
    int n
);

#ifdef __cplusplus
}
#endif

#endif /* EXP2_FEXPA_H */
