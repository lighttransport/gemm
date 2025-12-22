// Test 2^n computation for FP64
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Test function in assembly
extern void test_2n_f64(const double* input, double* output, size_t count);

int main() {
    size_t count = 16;
    double* input = aligned_alloc(64, count * sizeof(double));
    double* output = aligned_alloc(64, count * sizeof(double));

    // Input is the "n" value (integer stored as double)
    for (size_t i = 0; i < count; i++) {
        input[i] = (double)((int)i - 8);  // -8 to +7
    }

    test_2n_f64(input, output, count);

    for (size_t i = 0; i < count; i++) {
        double n = input[i];
        double expected = 1.0;
        for (int j = 0; j < (int)n; j++) expected *= 2.0;
        for (int j = 0; j > (int)n; j--) expected /= 2.0;
        printf("  n=%.0f: output=%.6e expected=%.6e\n", n, output[i], expected);
    }

    free(input);
    free(output);
    return 0;
}
