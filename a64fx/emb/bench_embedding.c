// bench_embedding.c
// Benchmark and test for SVE-optimized Embedding kernels
//
// Usage: ./bench_embedding [iterations] [batch_size] [hidden_dim] [vocab_size]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "embedding.h"

//=============================================================================
// Timing utilities
//=============================================================================
static inline double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

//=============================================================================
// Random number generation (PCG32)
//=============================================================================
static uint64_t pcg_state = 0x853c49e6748fea9bULL;
static uint64_t pcg_inc = 0xda3e39cb94b95bdbULL;

static uint32_t pcg32(void)
{
    uint64_t oldstate = pcg_state;
    pcg_state = oldstate * 6364136223846793005ULL + pcg_inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void pcg_seed(uint64_t seed)
{
    pcg_state = seed;
    pcg32();
}

static float rand_float(void)
{
    return (float)pcg32() / (float)UINT32_MAX;
}

static int32_t rand_int(int32_t max)
{
    return (int32_t)(pcg32() % (uint32_t)max);
}

//=============================================================================
// Comparison utilities
//=============================================================================
static float max_abs_diff_f32(const float* a, const float* b, size_t n)
{
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

static double max_abs_diff_f64(const double* a, const double* b, size_t n)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

//=============================================================================
// Test functions
//=============================================================================
static int test_embedding_fwd_f32(size_t batch_size, size_t hidden_dim,
                                  size_t vocab_size)
{
    printf("  Testing embedding_fwd_f32 (batch=%zu, dim=%zu, vocab=%zu)...\n",
           batch_size, hidden_dim, vocab_size);

    // Allocate
    int32_t* indices = aligned_alloc(64, batch_size * sizeof(int32_t));
    float* emb_table = aligned_alloc(64, vocab_size * hidden_dim * sizeof(float));
    float* output_ref = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));
    float* output_asm = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));

    if (!indices || !emb_table || !output_ref || !output_asm) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    // Initialize
    for (size_t i = 0; i < batch_size; i++) {
        indices[i] = rand_int((int32_t)vocab_size);
    }
    for (size_t i = 0; i < vocab_size * hidden_dim; i++) {
        emb_table[i] = rand_float() * 2.0f - 1.0f;
    }

    // Reference
    embedding_fwd_f32_ref(indices, emb_table, output_ref, batch_size, hidden_dim);

    // Assembly
    embedding_fwd_f32_asm(indices, emb_table, output_asm, batch_size, hidden_dim);

    // Compare
    float max_diff = max_abs_diff_f32(output_ref, output_asm, batch_size * hidden_dim);
    int passed = (max_diff < 1e-6f);

    printf("    max_diff = %e, %s\n", max_diff, passed ? "PASSED" : "FAILED");

    free(indices);
    free(emb_table);
    free(output_ref);
    free(output_asm);

    return passed ? 0 : 1;
}

static int test_embedding_fwd_f32_batched(size_t batch_size, size_t hidden_dim,
                                          size_t vocab_size)
{
    printf("  Testing embedding_fwd_f32_batched (batch=%zu, dim=%zu, vocab=%zu)...\n",
           batch_size, hidden_dim, vocab_size);

    int32_t* indices = aligned_alloc(64, batch_size * sizeof(int32_t));
    float* emb_table = aligned_alloc(64, vocab_size * hidden_dim * sizeof(float));
    float* output_ref = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));
    float* output_asm = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));

    if (!indices || !emb_table || !output_ref || !output_asm) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    for (size_t i = 0; i < batch_size; i++) {
        indices[i] = rand_int((int32_t)vocab_size);
    }
    for (size_t i = 0; i < vocab_size * hidden_dim; i++) {
        emb_table[i] = rand_float() * 2.0f - 1.0f;
    }

    embedding_fwd_f32_ref(indices, emb_table, output_ref, batch_size, hidden_dim);
    embedding_fwd_f32_batched_asm(indices, emb_table, output_asm, batch_size, hidden_dim);

    float max_diff = max_abs_diff_f32(output_ref, output_asm, batch_size * hidden_dim);
    int passed = (max_diff < 1e-6f);

    printf("    max_diff = %e, %s\n", max_diff, passed ? "PASSED" : "FAILED");

    free(indices);
    free(emb_table);
    free(output_ref);
    free(output_asm);

    return passed ? 0 : 1;
}

static int test_embedding_bwd_f32(size_t batch_size, size_t hidden_dim,
                                  size_t vocab_size)
{
    printf("  Testing embedding_bwd_f32 (batch=%zu, dim=%zu, vocab=%zu)...\n",
           batch_size, hidden_dim, vocab_size);

    int32_t* indices = aligned_alloc(64, batch_size * sizeof(int32_t));
    float* grad_output = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));
    float* grad_emb_ref = aligned_alloc(64, vocab_size * hidden_dim * sizeof(float));
    float* grad_emb_asm = aligned_alloc(64, vocab_size * hidden_dim * sizeof(float));

    if (!indices || !grad_output || !grad_emb_ref || !grad_emb_asm) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    // Use unique indices to avoid accumulation order issues
    for (size_t i = 0; i < batch_size; i++) {
        indices[i] = (int32_t)(i % vocab_size);
    }
    for (size_t i = 0; i < batch_size * hidden_dim; i++) {
        grad_output[i] = rand_float() * 2.0f - 1.0f;
    }

    // Zero initialize gradients
    memset(grad_emb_ref, 0, vocab_size * hidden_dim * sizeof(float));
    memset(grad_emb_asm, 0, vocab_size * hidden_dim * sizeof(float));

    // Reference
    embedding_bwd_f32_ref(indices, grad_output, grad_emb_ref,
                          batch_size, hidden_dim, vocab_size);

    // Assembly
    embedding_bwd_f32_asm(indices, grad_output, grad_emb_asm,
                          batch_size, hidden_dim, vocab_size);

    // Compare
    float max_diff = max_abs_diff_f32(grad_emb_ref, grad_emb_asm,
                                      vocab_size * hidden_dim);
    int passed = (max_diff < 1e-5f);

    printf("    max_diff = %e, %s\n", max_diff, passed ? "PASSED" : "FAILED");

    free(indices);
    free(grad_output);
    free(grad_emb_ref);
    free(grad_emb_asm);

    return passed ? 0 : 1;
}

static int test_embedding_fwd_f64(size_t batch_size, size_t hidden_dim,
                                  size_t vocab_size)
{
    printf("  Testing embedding_fwd_f64 (batch=%zu, dim=%zu, vocab=%zu)...\n",
           batch_size, hidden_dim, vocab_size);

    int32_t* indices = aligned_alloc(64, batch_size * sizeof(int32_t));
    double* emb_table = aligned_alloc(64, vocab_size * hidden_dim * sizeof(double));
    double* output_ref = aligned_alloc(64, batch_size * hidden_dim * sizeof(double));
    double* output_asm = aligned_alloc(64, batch_size * hidden_dim * sizeof(double));

    if (!indices || !emb_table || !output_ref || !output_asm) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    for (size_t i = 0; i < batch_size; i++) {
        indices[i] = rand_int((int32_t)vocab_size);
    }
    for (size_t i = 0; i < vocab_size * hidden_dim; i++) {
        emb_table[i] = (double)rand_float() * 2.0 - 1.0;
    }

    embedding_fwd_f64_ref(indices, emb_table, output_ref, batch_size, hidden_dim);
    embedding_fwd_f64_asm(indices, emb_table, output_asm, batch_size, hidden_dim);

    double max_diff = max_abs_diff_f64(output_ref, output_asm, batch_size * hidden_dim);
    int passed = (max_diff < 1e-14);

    printf("    max_diff = %e, %s\n", max_diff, passed ? "PASSED" : "FAILED");

    free(indices);
    free(emb_table);
    free(output_ref);
    free(output_asm);

    return passed ? 0 : 1;
}

static int test_embedding_fwd_with_pos_f32(size_t batch_size, size_t hidden_dim,
                                           size_t vocab_size, size_t max_pos)
{
    printf("  Testing embedding_fwd_with_pos_f32 (batch=%zu, dim=%zu)...\n",
           batch_size, hidden_dim);

    int32_t* token_ids = aligned_alloc(64, batch_size * sizeof(int32_t));
    int32_t* position_ids = aligned_alloc(64, batch_size * sizeof(int32_t));
    float* token_emb = aligned_alloc(64, vocab_size * hidden_dim * sizeof(float));
    float* pos_emb = aligned_alloc(64, max_pos * hidden_dim * sizeof(float));
    float* output_ref = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));
    float* output_asm = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));

    if (!token_ids || !position_ids || !token_emb || !pos_emb ||
        !output_ref || !output_asm) {
        printf("    FAILED: Memory allocation\n");
        return 1;
    }

    for (size_t i = 0; i < batch_size; i++) {
        token_ids[i] = rand_int((int32_t)vocab_size);
        position_ids[i] = (int32_t)(i % max_pos);
    }
    for (size_t i = 0; i < vocab_size * hidden_dim; i++) {
        token_emb[i] = rand_float() * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < max_pos * hidden_dim; i++) {
        pos_emb[i] = rand_float() * 0.1f - 0.05f;
    }

    embedding_fwd_with_pos_f32_ref(token_ids, position_ids, token_emb, pos_emb,
                                   output_ref, batch_size, hidden_dim);
    embedding_fwd_with_pos_f32_asm(token_ids, position_ids, token_emb, pos_emb,
                                   output_asm, batch_size, hidden_dim);

    float max_diff = max_abs_diff_f32(output_ref, output_asm, batch_size * hidden_dim);
    int passed = (max_diff < 1e-6f);

    printf("    max_diff = %e, %s\n", max_diff, passed ? "PASSED" : "FAILED");

    free(token_ids);
    free(position_ids);
    free(token_emb);
    free(pos_emb);
    free(output_ref);
    free(output_asm);

    return passed ? 0 : 1;
}

//=============================================================================
// Benchmark functions
//=============================================================================
static void bench_embedding_fwd_f32(size_t iterations, size_t batch_size,
                                    size_t hidden_dim, size_t vocab_size)
{
    printf("\nBenchmark: embedding_fwd_f32\n");
    printf("  batch_size=%zu, hidden_dim=%zu, vocab_size=%zu, iterations=%zu\n",
           batch_size, hidden_dim, vocab_size, iterations);

    int32_t* indices = aligned_alloc(64, batch_size * sizeof(int32_t));
    float* emb_table = aligned_alloc(64, vocab_size * hidden_dim * sizeof(float));
    float* output = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));

    for (size_t i = 0; i < batch_size; i++) {
        indices[i] = rand_int((int32_t)vocab_size);
    }
    for (size_t i = 0; i < vocab_size * hidden_dim; i++) {
        emb_table[i] = rand_float();
    }

    // Warmup
    for (size_t i = 0; i < 10; i++) {
        embedding_fwd_f32_asm(indices, emb_table, output, batch_size, hidden_dim);
    }

    // Benchmark reference
    double t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        embedding_fwd_f32_ref(indices, emb_table, output, batch_size, hidden_dim);
    }
    double t1 = get_time_sec();
    double ref_time = (t1 - t0) / iterations;

    // Benchmark assembly
    t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        embedding_fwd_f32_asm(indices, emb_table, output, batch_size, hidden_dim);
    }
    t1 = get_time_sec();
    double asm_time = (t1 - t0) / iterations;

    // Benchmark batched assembly
    t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        embedding_fwd_f32_batched_asm(indices, emb_table, output, batch_size, hidden_dim);
    }
    t1 = get_time_sec();
    double batched_time = (t1 - t0) / iterations;

    // Memory bandwidth calculation
    // Reads: batch_size * hidden_dim * 4 bytes (embedding rows)
    // Writes: batch_size * hidden_dim * 4 bytes (output)
    // Plus index reads: batch_size * 4 bytes
    size_t bytes_moved = batch_size * hidden_dim * sizeof(float) * 2 +
                         batch_size * sizeof(int32_t);
    double bw_ref = bytes_moved / ref_time / 1e9;
    double bw_asm = bytes_moved / asm_time / 1e9;
    double bw_batched = bytes_moved / batched_time / 1e9;

    printf("  Reference:    %.3f us, %.2f GB/s\n", ref_time * 1e6, bw_ref);
    printf("  ASM:          %.3f us, %.2f GB/s (%.2fx speedup)\n",
           asm_time * 1e6, bw_asm, ref_time / asm_time);
    printf("  ASM (batched): %.3f us, %.2f GB/s (%.2fx speedup)\n",
           batched_time * 1e6, bw_batched, ref_time / batched_time);

    free(indices);
    free(emb_table);
    free(output);
}

static void bench_embedding_bwd_f32(size_t iterations, size_t batch_size,
                                    size_t hidden_dim, size_t vocab_size)
{
    printf("\nBenchmark: embedding_bwd_f32\n");
    printf("  batch_size=%zu, hidden_dim=%zu, vocab_size=%zu, iterations=%zu\n",
           batch_size, hidden_dim, vocab_size, iterations);

    int32_t* indices = aligned_alloc(64, batch_size * sizeof(int32_t));
    float* grad_output = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));
    float* grad_emb = aligned_alloc(64, vocab_size * hidden_dim * sizeof(float));

    // Unique indices to avoid race conditions in timing
    for (size_t i = 0; i < batch_size; i++) {
        indices[i] = (int32_t)(i % vocab_size);
    }
    for (size_t i = 0; i < batch_size * hidden_dim; i++) {
        grad_output[i] = rand_float();
    }

    // Warmup
    memset(grad_emb, 0, vocab_size * hidden_dim * sizeof(float));
    for (size_t i = 0; i < 10; i++) {
        embedding_bwd_f32_asm(indices, grad_output, grad_emb,
                              batch_size, hidden_dim, vocab_size);
    }

    // Benchmark reference
    memset(grad_emb, 0, vocab_size * hidden_dim * sizeof(float));
    double t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        embedding_bwd_f32_ref(indices, grad_output, grad_emb,
                              batch_size, hidden_dim, vocab_size);
    }
    double t1 = get_time_sec();
    double ref_time = (t1 - t0) / iterations;

    // Benchmark assembly
    memset(grad_emb, 0, vocab_size * hidden_dim * sizeof(float));
    t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        embedding_bwd_f32_asm(indices, grad_output, grad_emb,
                              batch_size, hidden_dim, vocab_size);
    }
    t1 = get_time_sec();
    double asm_time = (t1 - t0) / iterations;

    // Bandwidth: read grad_output + read/write grad_embedding rows
    size_t bytes_moved = batch_size * hidden_dim * sizeof(float) * 3 +
                         batch_size * sizeof(int32_t);
    double bw_ref = bytes_moved / ref_time / 1e9;
    double bw_asm = bytes_moved / asm_time / 1e9;

    printf("  Reference: %.3f us, %.2f GB/s\n", ref_time * 1e6, bw_ref);
    printf("  ASM:       %.3f us, %.2f GB/s (%.2fx speedup)\n",
           asm_time * 1e6, bw_asm, ref_time / asm_time);

    free(indices);
    free(grad_output);
    free(grad_emb);
}

static void bench_embedding_fwd_f64(size_t iterations, size_t batch_size,
                                    size_t hidden_dim, size_t vocab_size)
{
    printf("\nBenchmark: embedding_fwd_f64\n");
    printf("  batch_size=%zu, hidden_dim=%zu, vocab_size=%zu, iterations=%zu\n",
           batch_size, hidden_dim, vocab_size, iterations);

    int32_t* indices = aligned_alloc(64, batch_size * sizeof(int32_t));
    double* emb_table = aligned_alloc(64, vocab_size * hidden_dim * sizeof(double));
    double* output = aligned_alloc(64, batch_size * hidden_dim * sizeof(double));

    for (size_t i = 0; i < batch_size; i++) {
        indices[i] = rand_int((int32_t)vocab_size);
    }
    for (size_t i = 0; i < vocab_size * hidden_dim; i++) {
        emb_table[i] = rand_float();
    }

    // Warmup
    for (size_t i = 0; i < 10; i++) {
        embedding_fwd_f64_asm(indices, emb_table, output, batch_size, hidden_dim);
    }

    // Benchmark reference
    double t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        embedding_fwd_f64_ref(indices, emb_table, output, batch_size, hidden_dim);
    }
    double t1 = get_time_sec();
    double ref_time = (t1 - t0) / iterations;

    // Benchmark assembly
    t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        embedding_fwd_f64_asm(indices, emb_table, output, batch_size, hidden_dim);
    }
    t1 = get_time_sec();
    double asm_time = (t1 - t0) / iterations;

    size_t bytes_moved = batch_size * hidden_dim * sizeof(double) * 2 +
                         batch_size * sizeof(int32_t);
    double bw_ref = bytes_moved / ref_time / 1e9;
    double bw_asm = bytes_moved / asm_time / 1e9;

    printf("  Reference: %.3f us, %.2f GB/s\n", ref_time * 1e6, bw_ref);
    printf("  ASM:       %.3f us, %.2f GB/s (%.2fx speedup)\n",
           asm_time * 1e6, bw_asm, ref_time / asm_time);

    free(indices);
    free(emb_table);
    free(output);
}

static void bench_embedding_with_pos_f32(size_t iterations, size_t batch_size,
                                         size_t hidden_dim, size_t vocab_size,
                                         size_t max_pos)
{
    printf("\nBenchmark: embedding_fwd_with_pos_f32\n");
    printf("  batch_size=%zu, hidden_dim=%zu, vocab_size=%zu, max_pos=%zu\n",
           batch_size, hidden_dim, vocab_size, max_pos);

    int32_t* token_ids = aligned_alloc(64, batch_size * sizeof(int32_t));
    int32_t* position_ids = aligned_alloc(64, batch_size * sizeof(int32_t));
    float* token_emb = aligned_alloc(64, vocab_size * hidden_dim * sizeof(float));
    float* pos_emb = aligned_alloc(64, max_pos * hidden_dim * sizeof(float));
    float* output = aligned_alloc(64, batch_size * hidden_dim * sizeof(float));

    for (size_t i = 0; i < batch_size; i++) {
        token_ids[i] = rand_int((int32_t)vocab_size);
        position_ids[i] = (int32_t)(i % max_pos);
    }
    for (size_t i = 0; i < vocab_size * hidden_dim; i++) {
        token_emb[i] = rand_float();
    }
    for (size_t i = 0; i < max_pos * hidden_dim; i++) {
        pos_emb[i] = rand_float();
    }

    // Warmup
    for (size_t i = 0; i < 10; i++) {
        embedding_fwd_with_pos_f32_asm(token_ids, position_ids, token_emb,
                                       pos_emb, output, batch_size, hidden_dim);
    }

    // Benchmark reference
    double t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        embedding_fwd_with_pos_f32_ref(token_ids, position_ids, token_emb,
                                       pos_emb, output, batch_size, hidden_dim);
    }
    double t1 = get_time_sec();
    double ref_time = (t1 - t0) / iterations;

    // Benchmark assembly
    t0 = get_time_sec();
    for (size_t i = 0; i < iterations; i++) {
        embedding_fwd_with_pos_f32_asm(token_ids, position_ids, token_emb,
                                       pos_emb, output, batch_size, hidden_dim);
    }
    t1 = get_time_sec();
    double asm_time = (t1 - t0) / iterations;

    // Bandwidth: read 2 embedding rows + write 1 output row per token
    size_t bytes_moved = batch_size * hidden_dim * sizeof(float) * 3 +
                         batch_size * sizeof(int32_t) * 2;
    double bw_ref = bytes_moved / ref_time / 1e9;
    double bw_asm = bytes_moved / asm_time / 1e9;

    printf("  Reference: %.3f us, %.2f GB/s\n", ref_time * 1e6, bw_ref);
    printf("  ASM:       %.3f us, %.2f GB/s (%.2fx speedup)\n",
           asm_time * 1e6, bw_asm, ref_time / asm_time);

    free(token_ids);
    free(position_ids);
    free(token_emb);
    free(pos_emb);
    free(output);
}

//=============================================================================
// Main
//=============================================================================
int main(int argc, char** argv)
{
    size_t iterations = 1000;
    size_t batch_size = 512;
    size_t hidden_dim = 4096;
    size_t vocab_size = 32000;
    size_t max_pos = 2048;

    if (argc > 1) iterations = (size_t)atol(argv[1]);
    if (argc > 2) batch_size = (size_t)atol(argv[2]);
    if (argc > 3) hidden_dim = (size_t)atol(argv[3]);
    if (argc > 4) vocab_size = (size_t)atol(argv[4]);

    printf("==============================================\n");
    printf("SVE Embedding Benchmark for A64FX\n");
    printf("==============================================\n");
    printf("Parameters:\n");
    printf("  iterations  = %zu\n", iterations);
    printf("  batch_size  = %zu\n", batch_size);
    printf("  hidden_dim  = %zu\n", hidden_dim);
    printf("  vocab_size  = %zu\n", vocab_size);
    printf("  max_pos     = %zu\n", max_pos);
    printf("\n");

    pcg_seed(42);

    // Run tests
    printf("=== Correctness Tests ===\n");
    int failures = 0;

    // Test with various sizes
    failures += test_embedding_fwd_f32(batch_size, hidden_dim, vocab_size);
    failures += test_embedding_fwd_f32_batched(batch_size, hidden_dim, vocab_size);
    failures += test_embedding_bwd_f32(batch_size, hidden_dim, vocab_size);
    failures += test_embedding_fwd_f64(batch_size, hidden_dim, vocab_size);
    failures += test_embedding_fwd_with_pos_f32(batch_size, hidden_dim, vocab_size, max_pos);

    // Test edge cases
    printf("\n  Edge cases:\n");
    failures += test_embedding_fwd_f32(1, hidden_dim, vocab_size);     // single token
    failures += test_embedding_fwd_f32(batch_size, 64, vocab_size);    // small dim
    failures += test_embedding_fwd_f32(batch_size, 63, vocab_size);    // non-VL aligned
    failures += test_embedding_fwd_f32(3, hidden_dim, vocab_size);     // non-batch-4 aligned

    if (failures > 0) {
        printf("\n!!! %d tests FAILED !!!\n", failures);
    } else {
        printf("\nAll tests PASSED!\n");
    }

    // Run benchmarks
    printf("\n=== Performance Benchmarks ===\n");
    bench_embedding_fwd_f32(iterations, batch_size, hidden_dim, vocab_size);
    bench_embedding_bwd_f32(iterations, batch_size, hidden_dim, vocab_size);
    bench_embedding_fwd_f64(iterations, batch_size, hidden_dim, vocab_size);
    bench_embedding_with_pos_f32(iterations, batch_size, hidden_dim, vocab_size, max_pos);

    // Test different hidden dimensions
    printf("\n=== Hidden Dimension Scaling ===\n");
    size_t dims[] = {256, 512, 1024, 2048, 4096, 8192};
    for (size_t i = 0; i < sizeof(dims)/sizeof(dims[0]); i++) {
        bench_embedding_fwd_f32(iterations/2, batch_size, dims[i], vocab_size);
    }

    printf("\n==============================================\n");
    printf("Benchmark complete.\n");

    return failures;
}
