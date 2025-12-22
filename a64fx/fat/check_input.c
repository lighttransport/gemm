#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);

int main() {
    size_t count = 16;
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output = aligned_alloc(64, count * sizeof(float));
    
    for (size_t i = 0; i < count; i++) {
        input[i] = (float)i * 0.5f;
    }
    
    printf("Before exp_f32_poly1:\n");
    for (int i = 0; i < 8; i++) printf("  input[%d] = %.6f\n", i, input[i]);
    
    exp_f32_poly1(input, output, count);
    
    printf("\nAfter exp_f32_poly1:\n");
    for (int i = 0; i < 8; i++) printf("  input[%d] = %.6f\n", i, input[i]);
    
    free(input);
    free(output);
    return 0;
}
