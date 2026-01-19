// activation.c - Activation functions for INT32 intermediate values
#include <stdint.h>
#include <stddef.h>
#include <math.h>

// Squared ReLU: max(0, x)^2
void squared_relu_int32_sve(const int32_t* input, int32_t* output, size_t length, int scale_shift) {
    for (size_t i = 0; i < length; i++) {
        int32_t x = input[i];
        if (x <= 0) {
            output[i] = 0;
        } else {
            // x * x with scaling to prevent overflow
            int64_t x64 = (int64_t)x;
            int64_t sq = (x64 * x64) >> scale_shift;
            output[i] = (int32_t)sq;
        }
    }
}

// Shift-GELU: Approximate GELU(x - shift)
// GELU(x) ≈ x * sigmoid(1.702 * x)
// Simplified: GELU(x) ≈ x * max(0, min(1, 0.5 + 0.25*x))
void shift_gelu_int32_sve(const int32_t* input, int32_t* output, size_t length, int shift) {
    for (size_t i = 0; i < length; i++) {
        int32_t x = input[i] - shift;

        // Approximate sigmoid: max(0, min(1, 0.5 + 0.25*x))
        // Assuming Q8.24 fixed point
        int32_t sig_approx = (1 << 23) + (x >> 2);  // 0.5 + 0.25*x
        if (sig_approx < 0) sig_approx = 0;
        if (sig_approx > (1 << 24)) sig_approx = (1 << 24);  // Clamp to [0, 1]

        // GELU(x) ≈ x * sigmoid_approx
        int64_t result = ((int64_t)x * sig_approx) >> 24;
        output[i] = (int32_t)result;
    }
}

// SiLU/Swish: x * sigmoid(x)
void silu_int32_sve(const int32_t* input, int32_t* output, size_t length) {
    for (size_t i = 0; i < length; i++) {
        int32_t x = input[i];

        // Approximate sigmoid: max(0, min(1, 0.5 + 0.25*x))
        int32_t sig_approx = (1 << 23) + (x >> 2);
        if (sig_approx < 0) sig_approx = 0;
        if (sig_approx > (1 << 24)) sig_approx = (1 << 24);

        // SiLU(x) = x * sigmoid(x)
        int64_t result = ((int64_t)x * sig_approx) >> 24;
        output[i] = (int32_t)result;
    }
}
