#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly5(const float* input, float* output, size_t count);

int main() {
    size_t count = 16384;
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output = aligned_alloc(64, count * sizeof(float));
    float* ref = aligned_alloc(64, count * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < count; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }
    
    // Compute reference
    for (size_t i = 0; i < count; i++) ref[i] = expf(input[i]);
    
    // Test poly5
    exp_f32_poly5(input, output, count);
    
    // Find largest errors
    float max_rel = 0;
    size_t max_idx = 0;
    int count_big = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > 1.0) {
            count_big++;
            if (count_big < 20) {
                printf("BIG ERROR at [%zu]: input=%.6f output=%.6e ref=%.6e rel=%.2e\n",
                       i, input[i], output[i], ref[i], rel);
            }
        }
        if (rel > max_rel) {
            max_rel = rel;
            max_idx = i;
        }
    }
    
    printf("\nTotal big errors (>100%%): %d\n", count_big);
    printf("Max rel at [%zu]: input=%.6f output=%.6e ref=%.6e rel=%.2e\n",
           max_idx, input[max_idx], output[max_idx], ref[max_idx], max_rel);
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
