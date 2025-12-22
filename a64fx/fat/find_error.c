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
    
    float max_rel = 0;
    size_t max_idx = 0;
    for (size_t i = 0; i < count; i++) {
        float expected = expf(input[i]);
        float rel = fabsf(output[i] - expected) / fabsf(expected);
        if (rel > max_rel) {
            max_rel = rel;
            max_idx = i;
        }
    }
    
    printf("Worst case at index %zu:\n", max_idx);
    float x = input[max_idx];
    float got = output[max_idx];
    float expected = expf(x);
    printf("  x = %.10f\n", x);
    printf("  got = %.10e\n", got);
    printf("  expected = %.10e\n", expected);
    printf("  rel_err = %.2e\n", max_rel);
    
    // Check a few around that point
    printf("\nValues around the worst case:\n");
    for (int d = -2; d <= 2; d++) {
        size_t i = max_idx + d;
        if (i >= count) continue;
        float expected = expf(input[i]);
        float rel = fabsf(output[i] - expected) / fabsf(expected);
        printf("  [%zu] x=%.6f got=%.6e exp=%.6e rel=%.2e\n",
               i, input[i], output[i], expected, rel);
    }
    
    free(input);
    free(output);
    return 0;
}
