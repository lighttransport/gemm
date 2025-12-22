#include <stdio.h>
#include <math.h>

extern void exp_f32_poly1(const float* input, float* output, size_t count);
extern void exp_f32_poly5(const float* input, float* output, size_t count);
extern void exp_f64_poly5(const double* input, double* output, size_t count);

int main() {
    float input32[16];
    float output32[16];
    double input64[8];
    double output64[8];
    
    printf("FP32 poly1 and poly5 tests:\n");
    for (int i = 0; i < 16; i++) {
        float x = -10.0f + i * 1.25f;  // Range [-10, 10]
        input32[i] = x;
    }
    
    exp_f32_poly1(input32, output32, 16);
    printf("poly1:\n");
    for (int i = 0; i < 16; i++) {
        float expected = expf(input32[i]);
        float rel_err = fabsf(output32[i] - expected) / fabsf(expected);
        printf("  exp(%7.2f) = %12.4e (expected %12.4e, rel_err=%.2e)\n",
               input32[i], output32[i], expected, rel_err);
    }
    
    exp_f32_poly5(input32, output32, 16);
    printf("\npoly5:\n");
    for (int i = 0; i < 16; i++) {
        float expected = expf(input32[i]);
        float rel_err = fabsf(output32[i] - expected) / fabsf(expected);
        printf("  exp(%7.2f) = %12.4e (expected %12.4e, rel_err=%.2e)\n",
               input32[i], output32[i], expected, rel_err);
    }
    
    printf("\nFP64 poly5 test:\n");
    for (int i = 0; i < 8; i++) {
        input64[i] = -10.0 + i * 2.5;
    }
    exp_f64_poly5(input64, output64, 8);
    for (int i = 0; i < 8; i++) {
        double expected = exp(input64[i]);
        double rel_err = fabs(output64[i] - expected) / fabs(expected);
        printf("  exp(%7.2f) = %12.4e (expected %12.4e, rel_err=%.2e)\n",
               input64[i], output64[i], expected, rel_err);
    }
    
    return 0;
}
