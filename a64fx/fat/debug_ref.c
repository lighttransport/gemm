#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void exp_ref_f32(const float* input, float* output, size_t count) {
    for (size_t i = 0; i < count; i++) output[i] = expf(input[i]);
}

int main() {
    size_t count = 16384;
    float* input = aligned_alloc(64, count * sizeof(float));
    float* ref1 = aligned_alloc(64, count * sizeof(float));
    float* ref2 = aligned_alloc(64, count * sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < count; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }
    
    // Method 1: exp_ref_f32
    exp_ref_f32(input, ref1, count);
    
    // Method 2: direct loop
    for (size_t i = 0; i < count; i++) ref2[i] = expf(input[i]);
    
    // Compare
    int diff_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (ref1[i] != ref2[i]) {
            if (diff_count < 5) {
                printf("Diff at [%zu]: ref1=%.10e ref2=%.10e\n", i, ref1[i], ref2[i]);
            }
            diff_count++;
        }
    }
    printf("Total differences: %d\n", diff_count);
    
    // Show first few from each
    printf("\nFirst 5 values:\n");
    for (int i = 0; i < 5; i++) {
        printf("  input[%d]=%.6f ref1=%.6e ref2=%.6e\n", i, input[i], ref1[i], ref2[i]);
    }
    
    free(input);
    free(ref1);
    free(ref2);
    return 0;
}
