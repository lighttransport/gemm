// Debug exp_f64 step by step
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f64_debug(const double* input, double* output);

int main() {
    double input;
    double* output = aligned_alloc(64, 8 * 64);  // 8 doubles * VL

    double test_vals[] = {-1.0, 0.0, 0.5, 1.0, 2.0};
    int n_tests = sizeof(test_vals) / sizeof(test_vals[0]);

    for (int t = 0; t < n_tests; t++) {
        input = test_vals[t];
        exp_f64_debug(&input, output);

        printf("x=%.2f:\n", input);
        printf("  n (rounded)     = %.6f\n", output[0]);
        printf("  r (reduced)     = %.6f\n", output[8]);   // offset by VL/8 = 8 doubles
        printf("  2^n             = %.6e\n", output[16]);
        printf("  poly(r)         = %.6f\n", output[24]);
        printf("  final           = %.6e\n", output[32]);
        printf("  expected exp(x) = %.6e\n", exp(input));
        printf("\n");
    }

    free(output);
    return 0;
}
