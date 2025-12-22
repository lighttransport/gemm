#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);
extern void exp_f32_poly2(const float* input, float* output, size_t count);
extern void exp_f32_poly3(const float* input, float* output, size_t count);
extern void exp_f32_poly4(const float* input, float* output, size_t count);
extern void exp_f32_poly5(const float* input, float* output, size_t count);

static void test_f32(const char* name,
                     void (*fn)(const float*, float*, size_t),
                     const float* input, const float* ref,
                     float* output, size_t count) {
    fn(input, output, count);
    float max_rel = 0, max_abs = 0;
    for (size_t i = 0; i < count; i++) {
        float err = fabsf(output[i] - ref[i]);
        float rel = (ref[i] != 0) ? err / fabsf(ref[i]) : err;
        if (rel > max_rel) max_rel = rel;
        if (err > max_abs) max_abs = err;
    }
    printf("  %s: max_rel=%.2e, max_abs=%.2e\n", name, max_rel, max_abs);
}

int main() {
    size_t count = 16384;
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output = aligned_alloc(64, count * sizeof(float));
    float* ref = aligned_alloc(64, count * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < count; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }
    for (size_t i = 0; i < count; i++) ref[i] = expf(input[i]);
    
    test_f32("poly1", exp_f32_poly1, input, ref, output, count);
    test_f32("poly2", exp_f32_poly2, input, ref, output, count);
    test_f32("poly3", exp_f32_poly3, input, ref, output, count);
    test_f32("poly4", exp_f32_poly4, input, ref, output, count);
    test_f32("poly5", exp_f32_poly5, input, ref, output, count);
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
