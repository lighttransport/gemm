#include <stdio.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);

int main() {
    float input[16] = {0.0f, 0.5f, 1.0f, -0.5f, 2.0f, -2.0f, 0.1f, -0.1f};
    float output[16];
    
    exp_f32_poly1(input, output, 8);
    
    printf("Testing exp_f32_poly1:\n");
    for (int i = 0; i < 8; i++) {
        float expected = expf(input[i]);
        float rel_err = fabsf(output[i] - expected) / expected;
        printf("  exp(%8.4f) = %12.6f (expected %12.6f, rel_err=%.2e)\n", 
               input[i], output[i], expected, rel_err);
    }
    return 0;
}
