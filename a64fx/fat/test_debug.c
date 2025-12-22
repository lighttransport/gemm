// test_debug.c - Debug rmsnorm internals

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void rmsnorm_debug(const float* input, float* debug_out, size_t dim, float eps);

int main() {
    size_t dim = 64;
    float eps = 1e-5f;

    float* input = aligned_alloc(64, dim * sizeof(float));
    float* debug_out = aligned_alloc(64, (4 + 16) * sizeof(float));

    for (size_t i = 0; i < dim; i++) {
        input[i] = 1.0f;
    }

    rmsnorm_debug(input, debug_out, dim, eps);

    printf("Debug RMSNorm (dim=%zu, all inputs=1.0):\n", dim);
    printf("  sum_sq       = %f (expected: %f)\n", debug_out[0], (float)dim);
    printf("  mean_sq      = %f (expected: 1.0)\n", debug_out[1]);
    printf("  variance_eps = %f (expected: 1.00001)\n", debug_out[2]);
    printf("  inv_std      = %f (expected: ~1.0)\n", debug_out[3]);
    printf("  output[0]    = %f (expected: ~1.0)\n", debug_out[4]);
    printf("  output[1]    = %f\n", debug_out[5]);
    printf("  output[15]   = %f\n", debug_out[4+15]);

    // Reference
    float sum_sq_ref = 0;
    for (size_t i = 0; i < dim; i++) {
        sum_sq_ref += input[i] * input[i];
    }
    float mean_sq_ref = sum_sq_ref / dim;
    float inv_std_ref = 1.0f / sqrtf(mean_sq_ref + eps);
    printf("\nReference:\n");
    printf("  sum_sq  = %f\n", sum_sq_ref);
    printf("  mean_sq = %f\n", mean_sq_ref);
    printf("  inv_std = %f\n", inv_std_ref);

    free(input);
    free(debug_out);
    return 0;
}
