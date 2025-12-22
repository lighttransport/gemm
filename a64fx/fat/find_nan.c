#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly5(const float* input, float* output, size_t count);

int main() {
    size_t count = 16384;
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output = aligned_alloc(64, count * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < count; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }
    
    exp_f32_poly5(input, output, count);
    
    // Find bad values
    int nan_count = 0, inf_count = 0, big_err_count = 0;
    for (size_t i = 0; i < count; i++) {
        float expected = expf(input[i]);
        if (isnan(output[i])) {
            if (nan_count < 5) printf("NaN at [%zu]: input=%.6f\n", i, input[i]);
            nan_count++;
        } else if (isinf(output[i])) {
            if (inf_count < 5) printf("Inf at [%zu]: input=%.6f\n", i, input[i]);
            inf_count++;
        } else {
            float rel_err = fabsf(output[i] - expected) / fabsf(expected);
            if (rel_err > 1.0) {
                if (big_err_count < 5) {
                    printf("Big err at [%zu]: input=%.6f output=%.6e expected=%.6e rel=%.2e\n",
                           i, input[i], output[i], expected, rel_err);
                }
                big_err_count++;
            }
        }
    }
    
    printf("\nTotal: %d NaN, %d Inf, %d big errors (>100%%)\n", nan_count, inf_count, big_err_count);
    
    free(input);
    free(output);
    return 0;
}
