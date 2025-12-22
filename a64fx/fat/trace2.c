#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);
extern void exp_f32_poly2(const float* input, float* output, size_t count);

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
    
    printf("Before any function call:\n");
    printf("  input[0] = %.10f\n", input[0]);
    printf("  ref[0]   = %.10e\n", ref[0]);
    
    // Call poly1
    exp_f32_poly1(input, output, count);
    printf("\nAfter poly1:\n");
    printf("  input[0]  = %.10f\n", input[0]);
    printf("  output[0] = %.10e\n", output[0]);
    printf("  ref[0]    = %.10e\n", ref[0]);
    printf("  expf(input[0]) = %.10e\n", expf(input[0]));
    
    // Call poly2
    exp_f32_poly2(input, output, count);
    printf("\nAfter poly2:\n");
    printf("  input[0]  = %.10f\n", input[0]);
    printf("  output[0] = %.10e\n", output[0]);
    printf("  ref[0]    = %.10e\n", ref[0]);
    printf("  expf(input[0]) = %.10e\n", expf(input[0]));
    
    // Double check: is ref[0] still same as original expf result?
    srand(42);
    float orig_input0 = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    printf("\nVerification:\n");
    printf("  Original input[0] should be: %.10f\n", orig_input0);
    printf("  expf(orig_input0) = %.10e\n", expf(orig_input0));
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
