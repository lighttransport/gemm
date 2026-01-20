// test_layernorm.c - Test INT8/INT16 LayerNorm and RMSNorm

#include "layernorm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Helper: Check if two arrays are close
static int check_close_int8(const int8_t* a, const int8_t* b, size_t N, int tolerance) {
    int errors = 0;
    for (size_t i = 0; i < N; i++) {
        int diff = abs(a[i] - b[i]);
        if (diff > tolerance) {
            if (errors < 10) {
                printf("  Mismatch at %zu: %d vs %d (diff=%d)\n", i, a[i], b[i], diff);
            }
            errors++;
        }
    }
    return errors;
}

static int check_close_int16(const int16_t* a, const int16_t* b, size_t N, int tolerance) {
    int errors = 0;
    for (size_t i = 0; i < N; i++) {
        int diff = abs(a[i] - b[i]);
        if (diff > tolerance) {
            if (errors < 10) {
                printf("  Mismatch at %zu: %d vs %d (diff=%d)\n", i, a[i], b[i], diff);
            }
            errors++;
        }
    }
    return errors;
}

// Reference LayerNorm in floating point
static void layernorm_ref(const float* input, float* output,
                         const float* gamma, const float* beta,
                         float epsilon, size_t N)
{
    // Compute mean
    double sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += input[i];
    }
    float mean = (float)(sum / N);

    // Compute variance
    double var_sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        float diff = input[i] - mean;
        var_sum += diff * diff;
    }
    float variance = (float)(var_sum / N);

    // Normalize
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    for (size_t i = 0; i < N; i++) {
        float norm = (input[i] - mean) * inv_std;
        output[i] = norm * gamma[i] + beta[i];
    }
}

// Reference RMSNorm in floating point
static void rmsnorm_ref(const float* input, float* output,
                       const float* gamma, float epsilon, size_t N)
{
    // Compute mean squared
    double sq_sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sq_sum += input[i] * input[i];
    }
    float mean_sq = (float)(sq_sum / N);

    // Normalize
    float inv_rms = 1.0f / sqrtf(mean_sq + epsilon);
    for (size_t i = 0; i < N; i++) {
        output[i] = input[i] * inv_rms * gamma[i];
    }
}

// Test INT8 LayerNorm
static void test_layernorm_int8(size_t N) {
    printf("\n=== Testing INT8 LayerNorm (N=%zu) ===\n", N);

    // Allocate
    int8_t* input = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t* output = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t* gamma = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t* beta = (int8_t*)malloc(N * sizeof(int8_t));

    float* input_f = (float*)malloc(N * sizeof(float));
    float* output_ref_f = (float*)malloc(N * sizeof(float));
    int8_t* output_ref = (int8_t*)malloc(N * sizeof(int8_t));
    float* gamma_f = (float*)malloc(N * sizeof(float));
    float* beta_f = (float*)malloc(N * sizeof(float));

    // Initialize with random values
    for (size_t i = 0; i < N; i++) {
        input[i] = (int8_t)((rand() % 200) - 100);
        gamma[i] = 64 + (rand() % 64); // 0.5 to 1.0
        beta[i] = (rand() % 20) - 10;

        input_f[i] = input[i] / 127.0f;
        gamma_f[i] = gamma[i] / 127.0f;
        beta_f[i] = beta[i] / 127.0f;
    }

    // Run reference
    layernorm_ref(input_f, output_ref_f, gamma_f, beta_f, 1e-5f, N);
    for (size_t i = 0; i < N; i++) {
        float val = output_ref_f[i] * 127.0f;
        if (val < -128.0f) val = -128.0f;
        if (val > 127.0f) val = 127.0f;
        output_ref[i] = (int8_t)val;
    }

    // Run INT8 implementation
    int32_t epsilon = (int32_t)(1e-5f * (1 << 24)); // Q8.24 format
    layernorm_int8(input, output, gamma, beta, epsilon, 1.0f/127.0f, 1.0f/127.0f, N);

    // Check results
    int errors = check_close_int8(output, output_ref, N, 5); // Allow tolerance of 5
    if (errors == 0) {
        printf("✓ PASSED\n");
    } else {
        printf("✗ FAILED: %d/%zu elements differ\n", errors, N);
    }

    // Print sample
    printf("Sample (first 10 elements):\n");
    printf("  Input:    ");
    for (int i = 0; i < 10 && i < (int)N; i++) printf("%4d ", input[i]);
    printf("\n");
    printf("  Output:   ");
    for (int i = 0; i < 10 && i < (int)N; i++) printf("%4d ", output[i]);
    printf("\n");
    printf("  Expected: ");
    for (int i = 0; i < 10 && i < (int)N; i++) printf("%4d ", output_ref[i]);
    printf("\n");

    free(input); free(output); free(gamma); free(beta);
    free(input_f); free(output_ref_f); free(output_ref);
    free(gamma_f); free(beta_f);
}

// Test INT8 RMSNorm
static void test_rmsnorm_int8(size_t N) {
    printf("\n=== Testing INT8 RMSNorm (N=%zu) ===\n", N);

    // Allocate
    int8_t* input = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t* output = (int8_t*)malloc(N * sizeof(int8_t));
    int8_t* gamma = (int8_t*)malloc(N * sizeof(int8_t));

    float* input_f = (float*)malloc(N * sizeof(float));
    float* output_ref_f = (float*)malloc(N * sizeof(float));
    int8_t* output_ref = (int8_t*)malloc(N * sizeof(int8_t));
    float* gamma_f = (float*)malloc(N * sizeof(float));

    // Initialize
    for (size_t i = 0; i < N; i++) {
        input[i] = (int8_t)((rand() % 200) - 100);
        gamma[i] = 64 + (rand() % 64);

        input_f[i] = input[i] / 127.0f;
        gamma_f[i] = gamma[i] / 127.0f;
    }

    // Run reference
    rmsnorm_ref(input_f, output_ref_f, gamma_f, 1e-5f, N);
    for (size_t i = 0; i < N; i++) {
        float val = output_ref_f[i] * 127.0f;
        if (val < -128.0f) val = -128.0f;
        if (val > 127.0f) val = 127.0f;
        output_ref[i] = (int8_t)val;
    }

    // Run INT8 implementation
    int32_t epsilon = (int32_t)(1e-5f * (1 << 24));
    rmsnorm_int8(input, output, gamma, epsilon, 1.0f/127.0f, 1.0f/127.0f, N);

    // Check results
    int errors = check_close_int8(output, output_ref, N, 5);
    if (errors == 0) {
        printf("✓ PASSED\n");
    } else {
        printf("✗ FAILED: %d/%zu elements differ\n", errors, N);
    }

    // Print sample
    printf("Sample (first 10 elements):\n");
    printf("  Input:    ");
    for (int i = 0; i < 10 && i < (int)N; i++) printf("%4d ", input[i]);
    printf("\n");
    printf("  Output:   ");
    for (int i = 0; i < 10 && i < (int)N; i++) printf("%4d ", output[i]);
    printf("\n");
    printf("  Expected: ");
    for (int i = 0; i < 10 && i < (int)N; i++) printf("%4d ", output_ref[i]);
    printf("\n");

    free(input); free(output); free(gamma);
    free(input_f); free(output_ref_f); free(output_ref); free(gamma_f);
}

// Test INT16 LayerNorm
static void test_layernorm_int16(size_t N) {
    printf("\n=== Testing INT16 LayerNorm (N=%zu) ===\n", N);

    int16_t* input = (int16_t*)malloc(N * sizeof(int16_t));
    int16_t* output = (int16_t*)malloc(N * sizeof(int16_t));
    int16_t* gamma = (int16_t*)malloc(N * sizeof(int16_t));
    int16_t* beta = (int16_t*)malloc(N * sizeof(int16_t));

    float* input_f = (float*)malloc(N * sizeof(float));
    float* output_ref_f = (float*)malloc(N * sizeof(float));
    int16_t* output_ref = (int16_t*)malloc(N * sizeof(int16_t));
    float* gamma_f = (float*)malloc(N * sizeof(float));
    float* beta_f = (float*)malloc(N * sizeof(float));

    // Initialize
    for (size_t i = 0; i < N; i++) {
        input[i] = (int16_t)((rand() % 20000) - 10000);
        gamma[i] = 8192 + (rand() % 8192); // 0.5 to 1.0 in Q14
        beta[i] = (rand() % 2000) - 1000;

        input_f[i] = input[i] / 16384.0f;
        gamma_f[i] = gamma[i] / 16384.0f;
        beta_f[i] = beta[i] / 16384.0f;
    }

    // Run reference
    layernorm_ref(input_f, output_ref_f, gamma_f, beta_f, 1e-5f, N);
    for (size_t i = 0; i < N; i++) {
        float val = output_ref_f[i] * 16384.0f;
        if (val < -32768.0f) val = -32768.0f;
        if (val > 32767.0f) val = 32767.0f;
        output_ref[i] = (int16_t)val;
    }

    // Run INT16 implementation
    int32_t epsilon = (int32_t)(1e-5f * (1 << 16));
    layernorm_int16(input, output, gamma, beta, epsilon, 1.0f/16384.0f, 1.0f/16384.0f, N);

    // Check results
    int errors = check_close_int16(output, output_ref, N, 50);
    if (errors == 0) {
        printf("✓ PASSED\n");
    } else {
        printf("✗ FAILED: %d/%zu elements differ\n", errors, N);
    }

    free(input); free(output); free(gamma); free(beta);
    free(input_f); free(output_ref_f); free(output_ref);
    free(gamma_f); free(beta_f);
}

// Test INT16 RMSNorm
static void test_rmsnorm_int16(size_t N) {
    printf("\n=== Testing INT16 RMSNorm (N=%zu) ===\n", N);

    int16_t* input = (int16_t*)malloc(N * sizeof(int16_t));
    int16_t* output = (int16_t*)malloc(N * sizeof(int16_t));
    int16_t* gamma = (int16_t*)malloc(N * sizeof(int16_t));

    float* input_f = (float*)malloc(N * sizeof(float));
    float* output_ref_f = (float*)malloc(N * sizeof(float));
    int16_t* output_ref = (int16_t*)malloc(N * sizeof(int16_t));
    float* gamma_f = (float*)malloc(N * sizeof(float));

    // Initialize
    for (size_t i = 0; i < N; i++) {
        input[i] = (int16_t)((rand() % 20000) - 10000);
        gamma[i] = 8192 + (rand() % 8192);

        input_f[i] = input[i] / 16384.0f;
        gamma_f[i] = gamma[i] / 16384.0f;
    }

    // Run reference
    rmsnorm_ref(input_f, output_ref_f, gamma_f, 1e-5f, N);
    for (size_t i = 0; i < N; i++) {
        float val = output_ref_f[i] * 16384.0f;
        if (val < -32768.0f) val = -32768.0f;
        if (val > 32767.0f) val = 32767.0f;
        output_ref[i] = (int16_t)val;
    }

    // Run INT16 implementation
    int32_t epsilon = (int32_t)(1e-5f * (1 << 16));
    rmsnorm_int16(input, output, gamma, epsilon, 1.0f/16384.0f, 1.0f/16384.0f, N);

    // Check results
    int errors = check_close_int16(output, output_ref, N, 50);
    if (errors == 0) {
        printf("✓ PASSED\n");
    } else {
        printf("✗ FAILED: %d/%zu elements differ\n", errors, N);
    }

    free(input); free(output); free(gamma);
    free(input_f); free(output_ref_f); free(output_ref); free(gamma_f);
}

int main(void) {
    printf("================================================================\n");
    printf("INT8/INT16 LayerNorm and RMSNorm Test Suite\n");
    printf("================================================================\n");

    srand(42);

    // Test various sizes
    test_layernorm_int8(64);
    test_layernorm_int8(256);
    test_layernorm_int8(512);
    test_layernorm_int8(1024);

    test_rmsnorm_int8(64);
    test_rmsnorm_int8(256);
    test_rmsnorm_int8(512);
    test_rmsnorm_int8(1024);

    test_layernorm_int16(64);
    test_layernorm_int16(256);
    test_layernorm_int16(512);
    test_layernorm_int16(1024);

    test_rmsnorm_int16(64);
    test_rmsnorm_int16(256);
    test_rmsnorm_int16(512);
    test_rmsnorm_int16(1024);

    printf("\n================================================================\n");
    printf("All tests completed\n");
    printf("================================================================\n");

    return 0;
}
