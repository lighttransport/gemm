// Simple FP64 exp test
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void exp_f64_poly1(const double* input, double* output, size_t count);
extern void exp_f64_poly5(const double* input, double* output, size_t count);

int main() {
    size_t count = 16;
    double* input = aligned_alloc(64, count * sizeof(double));
    double* output = aligned_alloc(64, count * sizeof(double));

    // Simple test values
    for (size_t i = 0; i < count; i++) {
        input[i] = (double)i - 8.0;  // -8 to +7
    }

    printf("Testing exp_f64_poly1:\n");
    exp_f64_poly1(input, output, count);
    for (size_t i = 0; i < count; i++) {
        double ref = exp(input[i]);
        double rel = fabs(output[i] - ref) / ref;
        printf("  x=%.1f: output=%.6e ref=%.6e rel_err=%.2e\n",
               input[i], output[i], ref, rel);
    }

    printf("\nTesting exp_f64_poly5:\n");
    exp_f64_poly5(input, output, count);
    for (size_t i = 0; i < count; i++) {
        double ref = exp(input[i]);
        double rel = fabs(output[i] - ref) / ref;
        printf("  x=%.1f: output=%.6e ref=%.6e rel_err=%.2e\n",
               input[i], output[i], ref, rel);
    }

    free(input);
    free(output);
    return 0;
}
