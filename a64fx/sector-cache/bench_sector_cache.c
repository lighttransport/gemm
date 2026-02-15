/*
 * A64FX Sector Cache Hint (Tagged Pointer) Test
 *
 * A64FX L1 Data Cache Sector Cache Feature:
 * - L1D cache: 64KB, 4-way set associative
 * - 2 sectors: Sector 0 (way 0,1) for data reuse, Sector 1 (way 2,3) for streaming
 * - Tagged pointer: bits 59:56 of virtual address control sector assignment
 *
 * Tag values (bits 59:56):
 *   0x0: Normal (no hint, uses both sectors)
 *   0x1: Sector 0 preferred (reuse data)
 *   0x2: Sector 1 preferred (streaming data)
 *   0x9: Sector 0 strong hint
 *   0xA: Sector 1 strong hint
 *   0xB: Bypass L1 cache
 *
 * Use fapp profiling to measure L1 cache hit/miss rates
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#ifdef USE_FAPP
#include <fj_tool/fapp.h>
#endif

// Sector cache tag values
#define TAG_NORMAL      0x0ULL  // No hint
#define TAG_SECTOR0     0x1ULL  // Sector 0 (reuse)
#define TAG_SECTOR1     0x2ULL  // Sector 1 (streaming)
#define TAG_SECTOR0_S   0x9ULL  // Sector 0 strong hint
#define TAG_SECTOR1_S   0xAULL  // Sector 1 strong hint
#define TAG_BYPASS_L1   0xBULL  // Bypass L1

// Apply sector cache tag to pointer
#define APPLY_TAG(ptr, tag) ((void*)((uint64_t)(ptr) | ((tag) << 56)))
#define REMOVE_TAG(ptr) ((void*)((uint64_t)(ptr) & 0x00FFFFFFFFFFFFFFULL))

// Cache line size
#define CACHE_LINE 256  // A64FX L1 cache line = 256 bytes

// Test parameters
#define L1_SIZE      (64 * 1024)        // 64KB L1
#define L2_SIZE      (8 * 1024 * 1024)  // 8MB L2
#define ARRAY_SIZE   (L1_SIZE * 2)      // 128KB (larger than L1)
#define SMALL_ARRAY  (L1_SIZE / 4)      // 16KB (fits in half L1)
#define NUM_ITERS    100

// Aligned allocation
static void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

// Initialize array with random values
static void init_array(float* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 1000) / 100.0f;
    }
}

// Measure time in cycles
static inline uint64_t read_cycle(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_freq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

/*
 * Test 1: Streaming read without sector hint
 * Data will compete for all cache ways
 */
static double test_streaming_no_hint(float* data, size_t n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;

#ifdef USE_FAPP
    fapp_start("streaming_no_hint", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        float local_sum = 0.0f;
        for (size_t i = 0; i < n; i += 8) {
            local_sum += data[i] + data[i+1] + data[i+2] + data[i+3];
            local_sum += data[i+4] + data[i+5] + data[i+6] + data[i+7];
        }
        sum += local_sum;
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("streaming_no_hint", 1, 0);
#endif

    (void)sum;
    return (double)(end - start);
}

/*
 * Test 2: Streaming read with Sector 1 hint (streaming sector)
 * Data goes to streaming sector, won't evict reuse data
 */
static double test_streaming_sector1(float* data, size_t n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;
    float* tagged_data = (float*)APPLY_TAG(data, TAG_SECTOR1);

#ifdef USE_FAPP
    fapp_start("streaming_sector1", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        float local_sum = 0.0f;
        for (size_t i = 0; i < n; i += 8) {
            local_sum += tagged_data[i] + tagged_data[i+1] + tagged_data[i+2] + tagged_data[i+3];
            local_sum += tagged_data[i+4] + tagged_data[i+5] + tagged_data[i+6] + tagged_data[i+7];
        }
        sum += local_sum;
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("streaming_sector1", 1, 0);
#endif

    (void)sum;
    return (double)(end - start);
}

/*
 * Test 3: Reuse pattern - same small data accessed repeatedly
 * Without hint: competes with streaming data
 */
static double test_reuse_no_hint(float* data, size_t n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;

#ifdef USE_FAPP
    fapp_start("reuse_no_hint", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        float local_sum = 0.0f;
        for (size_t i = 0; i < n; i += 8) {
            local_sum += data[i] + data[i+1] + data[i+2] + data[i+3];
            local_sum += data[i+4] + data[i+5] + data[i+6] + data[i+7];
        }
        sum += local_sum;
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("reuse_no_hint", 1, 0);
#endif

    (void)sum;
    return (double)(end - start);
}

/*
 * Test 4: Reuse pattern with Sector 0 hint (reuse sector)
 * Data stays in reuse sector
 */
static double test_reuse_sector0(float* data, size_t n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;
    float* tagged_data = (float*)APPLY_TAG(data, TAG_SECTOR0);

#ifdef USE_FAPP
    fapp_start("reuse_sector0", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        float local_sum = 0.0f;
        for (size_t i = 0; i < n; i += 8) {
            local_sum += tagged_data[i] + tagged_data[i+1] + tagged_data[i+2] + tagged_data[i+3];
            local_sum += tagged_data[i+4] + tagged_data[i+5] + tagged_data[i+6] + tagged_data[i+7];
        }
        sum += local_sum;
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("reuse_sector0", 1, 0);
#endif

    (void)sum;
    return (double)(end - start);
}

/*
 * Test 5: Mixed access - reuse data + streaming data
 * Without sector hints, streaming evicts reuse data
 */
static double test_mixed_no_hint(float* reuse_data, float* stream_data,
                                  size_t reuse_n, size_t stream_n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;

#ifdef USE_FAPP
    fapp_start("mixed_no_hint", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        float local_sum = 0.0f;
        // Access reuse data
        for (size_t i = 0; i < reuse_n; i += 8) {
            local_sum += reuse_data[i] + reuse_data[i+1] + reuse_data[i+2] + reuse_data[i+3];
        }
        // Stream through large array (evicts reuse data without hints)
        for (size_t i = 0; i < stream_n; i += 8) {
            local_sum += stream_data[i];
        }
        // Access reuse data again (cache miss if evicted)
        for (size_t i = 0; i < reuse_n; i += 8) {
            local_sum += reuse_data[i] + reuse_data[i+1] + reuse_data[i+2] + reuse_data[i+3];
        }
        sum += local_sum;
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("mixed_no_hint", 1, 0);
#endif

    (void)sum;
    return (double)(end - start);
}

/*
 * Test 6: Mixed access with sector hints
 * Reuse data in Sector 0, streaming data in Sector 1
 * Streaming won't evict reuse data
 */
static double test_mixed_with_hint(float* reuse_data, float* stream_data,
                                    size_t reuse_n, size_t stream_n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;
    float* tagged_reuse = (float*)APPLY_TAG(reuse_data, TAG_SECTOR0);
    float* tagged_stream = (float*)APPLY_TAG(stream_data, TAG_SECTOR1);

#ifdef USE_FAPP
    fapp_start("mixed_with_hint", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        float local_sum = 0.0f;
        // Access reuse data (Sector 0)
        for (size_t i = 0; i < reuse_n; i += 8) {
            local_sum += tagged_reuse[i] + tagged_reuse[i+1] + tagged_reuse[i+2] + tagged_reuse[i+3];
        }
        // Stream through large array (Sector 1 - won't evict Sector 0)
        for (size_t i = 0; i < stream_n; i += 8) {
            local_sum += tagged_stream[i];
        }
        // Access reuse data again (should still be in cache!)
        for (size_t i = 0; i < reuse_n; i += 8) {
            local_sum += tagged_reuse[i] + tagged_reuse[i+1] + tagged_reuse[i+2] + tagged_reuse[i+3];
        }
        sum += local_sum;
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("mixed_with_hint", 1, 0);
#endif

    (void)sum;
    return (double)(end - start);
}

/*
 * Test 7: L1 bypass test
 * Direct to L2, useful when data won't be reused soon
 */
static double test_bypass_l1(float* data, size_t n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;
    float* tagged_data = (float*)APPLY_TAG(data, TAG_BYPASS_L1);

#ifdef USE_FAPP
    fapp_start("bypass_l1", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        float local_sum = 0.0f;
        for (size_t i = 0; i < n; i += 8) {
            local_sum += tagged_data[i] + tagged_data[i+1] + tagged_data[i+2] + tagged_data[i+3];
        }
        sum += local_sum;
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("bypass_l1", 1, 0);
#endif

    (void)sum;
    return (double)(end - start);
}

/*
 * Test 8: SVE streaming with sector hint
 */
static double test_sve_streaming_sector1(float* data, size_t n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;
    float* tagged_data = (float*)APPLY_TAG(data, TAG_SECTOR1);

#ifdef USE_FAPP
    fapp_start("sve_streaming_sector1", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        svfloat32_t vsum = svdup_f32(0.0f);
        svbool_t pg = svptrue_b32();

        for (size_t i = 0; i < n; i += svcntw()) {
            svfloat32_t va = svld1_f32(pg, &tagged_data[i]);
            vsum = svadd_f32_m(pg, vsum, va);
        }

        sum += svaddv_f32(pg, vsum);
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("sve_streaming_sector1", 1, 0);
#endif

    (void)sum;
    return (double)(end - start);
}

/*
 * Test 9: GEMM-like pattern with sector hints
 * A matrix: reuse (loaded multiple times)
 * B matrix: streaming (loaded once per K block)
 */
static double test_gemm_pattern_with_hint(float* A, float* B, float* C,
                                           int M, int N, int K, int iters) {
    uint64_t start, end;
    float* tagged_A = (float*)APPLY_TAG(A, TAG_SECTOR0);  // Reuse
    float* tagged_B = (float*)APPLY_TAG(B, TAG_SECTOR1);  // Streaming

#ifdef USE_FAPP
    fapp_start("gemm_with_hint", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += tagged_A[i * K + k] * tagged_B[k * N + j];
                }
                C[i * N + j] = acc;
            }
        }
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("gemm_with_hint", 1, 0);
#endif

    return (double)(end - start);
}

static double test_gemm_pattern_no_hint(float* A, float* B, float* C,
                                         int M, int N, int K, int iters) {
    uint64_t start, end;

#ifdef USE_FAPP
    fapp_start("gemm_no_hint", 1, 0);
#endif

    start = read_cycle();
    for (int iter = 0; iter < iters; iter++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = acc;
            }
        }
    }
    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop("gemm_no_hint", 1, 0);
#endif

    return (double)(end - start);
}

int main(int argc, char* argv[]) {
    printf("=== A64FX Sector Cache Hint Test ===\n\n");

    // Get frequency
    uint64_t freq = get_freq();
    printf("Timer frequency: %lu Hz\n\n", freq);

    // Allocate arrays
    float* large_array = (float*)aligned_alloc_wrapper(CACHE_LINE, ARRAY_SIZE);
    float* small_array = (float*)aligned_alloc_wrapper(CACHE_LINE, SMALL_ARRAY);
    float* stream_array = (float*)aligned_alloc_wrapper(CACHE_LINE, ARRAY_SIZE);

    if (!large_array || !small_array || !stream_array) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    size_t large_n = ARRAY_SIZE / sizeof(float);
    size_t small_n = SMALL_ARRAY / sizeof(float);

    // Initialize
    init_array(large_array, large_n);
    init_array(small_array, small_n);
    init_array(stream_array, large_n);

    printf("Array sizes:\n");
    printf("  Large array: %d KB (%zu elements)\n", (int)(ARRAY_SIZE / 1024), large_n);
    printf("  Small array: %d KB (%zu elements)\n", (int)(SMALL_ARRAY / 1024), small_n);
    printf("  L1 cache: %d KB\n\n", L1_SIZE / 1024);

    printf("Sector Cache Tag Values:\n");
    printf("  TAG_NORMAL (0x0):    No hint, uses both sectors\n");
    printf("  TAG_SECTOR0 (0x1):   Sector 0 preferred (reuse)\n");
    printf("  TAG_SECTOR1 (0x2):   Sector 1 preferred (streaming)\n");
    printf("  TAG_BYPASS_L1 (0xB): Bypass L1 cache\n\n");

    double cycles;
    double bytes;
    double bandwidth;

    // Warmup
    test_streaming_no_hint(large_array, large_n, 10);

    printf("=== Test Results ===\n\n");

    // Test 1 & 2: Streaming comparison
    printf("--- Streaming Access Pattern ---\n");

    cycles = test_streaming_no_hint(large_array, large_n, NUM_ITERS);
    bytes = (double)ARRAY_SIZE * NUM_ITERS;
    bandwidth = bytes / (cycles / freq) / 1e9;
    printf("Streaming (no hint):     %10.0f cycles, %.2f GB/s\n", cycles, bandwidth);

    cycles = test_streaming_sector1(large_array, large_n, NUM_ITERS);
    bandwidth = bytes / (cycles / freq) / 1e9;
    printf("Streaming (Sector 1):    %10.0f cycles, %.2f GB/s\n", cycles, bandwidth);

    cycles = test_sve_streaming_sector1(large_array, large_n, NUM_ITERS);
    bandwidth = bytes / (cycles / freq) / 1e9;
    printf("SVE Streaming (Sector 1):%10.0f cycles, %.2f GB/s\n", cycles, bandwidth);

    printf("\n");

    // Test 3 & 4: Reuse comparison
    printf("--- Reuse Access Pattern (small array, many iterations) ---\n");

    cycles = test_reuse_no_hint(small_array, small_n, NUM_ITERS * 10);
    bytes = (double)SMALL_ARRAY * NUM_ITERS * 10;
    bandwidth = bytes / (cycles / freq) / 1e9;
    printf("Reuse (no hint):         %10.0f cycles, %.2f GB/s\n", cycles, bandwidth);

    cycles = test_reuse_sector0(small_array, small_n, NUM_ITERS * 10);
    bandwidth = bytes / (cycles / freq) / 1e9;
    printf("Reuse (Sector 0):        %10.0f cycles, %.2f GB/s\n", cycles, bandwidth);

    printf("\n");

    // Test 5 & 6: Mixed access (most important test!)
    printf("--- Mixed Access Pattern (reuse + streaming) ---\n");
    printf("This test shows the real benefit of sector cache hints!\n");
    printf("Reuse array: %d KB, Stream array: %d KB\n\n", (int)(SMALL_ARRAY/1024), (int)(ARRAY_SIZE/1024));

    cycles = test_mixed_no_hint(small_array, stream_array, small_n, large_n, NUM_ITERS);
    printf("Mixed (no hint):         %10.0f cycles\n", cycles);

    cycles = test_mixed_with_hint(small_array, stream_array, small_n, large_n, NUM_ITERS);
    printf("Mixed (with hints):      %10.0f cycles\n", cycles);

    printf("\n");

    // Test 7: L1 bypass
    printf("--- L1 Bypass Test ---\n");

    cycles = test_streaming_no_hint(large_array, large_n, NUM_ITERS);
    bytes = (double)ARRAY_SIZE * NUM_ITERS;
    bandwidth = bytes / (cycles / freq) / 1e9;
    printf("Normal access:           %10.0f cycles, %.2f GB/s\n", cycles, bandwidth);

    cycles = test_bypass_l1(large_array, large_n, NUM_ITERS);
    bandwidth = bytes / (cycles / freq) / 1e9;
    printf("L1 Bypass (0xB):         %10.0f cycles, %.2f GB/s\n", cycles, bandwidth);

    printf("\n");

    // Test 8 & 9: GEMM pattern
    printf("--- GEMM-like Pattern ---\n");
    int M = 64, N = 64, K = 64;
    float* A = (float*)aligned_alloc_wrapper(CACHE_LINE, M * K * sizeof(float));
    float* B = (float*)aligned_alloc_wrapper(CACHE_LINE, K * N * sizeof(float));
    float* C = (float*)aligned_alloc_wrapper(CACHE_LINE, M * N * sizeof(float));

    if (A && B && C) {
        init_array(A, M * K);
        init_array(B, K * N);
        memset(C, 0, M * N * sizeof(float));

        cycles = test_gemm_pattern_no_hint(A, B, C, M, N, K, NUM_ITERS);
        printf("GEMM (no hint):          %10.0f cycles\n", cycles);

        memset(C, 0, M * N * sizeof(float));
        cycles = test_gemm_pattern_with_hint(A, B, C, M, N, K, NUM_ITERS);
        printf("GEMM (with hints):       %10.0f cycles\n", cycles);

        free(A);
        free(B);
        free(C);
    }

    printf("\n=== Summary ===\n");
    printf("Sector cache hints help when:\n");
    printf("1. You have data that will be reused (put in Sector 0)\n");
    printf("2. You have streaming data accessed once (put in Sector 1)\n");
    printf("3. Streaming data would otherwise evict reuse data\n");
    printf("\nUse fapp profiling to see L1 cache hit/miss rates:\n");
    printf("  fapp -C -d <output_dir> -Hevent=Cache ./bench_sector_cache\n");

    // Cleanup
    free(large_array);
    free(small_array);
    free(stream_array);

    return 0;
}
