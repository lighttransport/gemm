/*
 * Quick test for exp2 polynomial
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

extern void exp2_poly_simple_v2(const float* in, float* out, int n);
extern void exp2_poly_softmax_v2(const int32_t* in, float* out, int n,
                                  float scale, int32_t max_val);

int main() {
    // Test simple exp2 with small range
    printf("=== Simple exp2 test ===\n");
    float test_vals[] = {0.0f, 0.5f, 1.0f, -1.0f, -0.5f, 2.0f, -2.0f, 10.0f, -10.0f};
    int n = sizeof(test_vals) / sizeof(test_vals[0]);
    float out[16], ref[16];

    for (int i = 0; i < n; i++) ref[i] = exp2f(test_vals[i]);
    exp2_poly_simple_v2(test_vals, out, n);

    printf("Input         Reference       Polynomial      RelErr\n");
    for (int i = 0; i < n; i++) {
        float rel_err = fabsf(out[i] - ref[i]) / ref[i] * 100;
        printf("%10.4f   %14.6e   %14.6e   %.4f%%\n",
               test_vals[i], ref[i], out[i], rel_err);
    }

    // Test softmax exp2 with typical QK scaling
    printf("\n=== Softmax exp2 test (typical QK values) ===\n");
    // Simulate QK scores in range [0, 10000] with max=10000, scale=0.01*log2(e)
    int32_t qk_vals[] = {10000, 9000, 8000, 7000, 5000, 0, -5000, -10000};
    int32_t max_val = 10000;
    float scale = 0.01f * 1.4426950408889634f;  // 0.01 * log2(e)
    int m = sizeof(qk_vals) / sizeof(qk_vals[0]);

    float out_sm[16], ref_sm[16];
    for (int i = 0; i < m; i++) {
        float x = ((float)qk_vals[i] - (float)max_val) * scale;
        ref_sm[i] = exp2f(x);
    }
    exp2_poly_softmax_v2(qk_vals, out_sm, m, scale, max_val);

    printf("QK            x             Reference       Polynomial      RelErr\n");
    for (int i = 0; i < m; i++) {
        float x = ((float)qk_vals[i] - (float)max_val) * scale;
        float rel_err = fabsf(out_sm[i] - ref_sm[i]) / ref_sm[i] * 100;
        printf("%8d   %10.4f   %14.6e   %14.6e   %.4f%%\n",
               qk_vals[i], x, ref_sm[i], out_sm[i], rel_err);
    }

    return 0;
}
