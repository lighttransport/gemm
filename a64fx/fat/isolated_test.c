#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);
extern void exp_f32_poly5(const float* input, float* output, size_t count);

int main() {
    size_t count = 16384;
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output = aligned_alloc(64, count * sizeof(float));
    float* ref = aligned_alloc(64, count * sizeof(float));
    
    // Same setup as bench_poly.c
    srand(42);
    for (size_t i = 0; i < count; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }
    for (size_t i = 0; i < count; i++) ref[i] = expf(input[i]);
    
    // Test poly1
    exp_f32_poly1(input, output, count);
    float max_rel1 = 0, max_abs1 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max_rel1) max_rel1 = rel;
        if (err > max_abs1) max_abs1 = err;
    }
    printf("poly1: max_rel=%.2e, max_abs=%.2e\n", max_rel1, max_abs1);
    
    // Test poly5
    exp_f32_poly5(input, output, count);
    float max_rel5 = 0, max_abs5 = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max_rel5) max_rel5 = rel;
        if (err > max_abs5) max_abs5 = err;
    }
    printf("poly5: max_rel=%.2e, max_abs=%.2e\n", max_rel5, max_abs5);
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
