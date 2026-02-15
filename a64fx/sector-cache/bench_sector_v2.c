/*
 * A64FX Sector Cache Hint Test V2 - Improved version
 *
 * Better cache pressure scenarios to demonstrate sector cache effect
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

// Sector cache tag values (bits 59:56 of virtual address)
#define TAG_NORMAL      0x0ULL
#define TAG_SECTOR0     0x1ULL  // Reuse sector (way 0,1)
#define TAG_SECTOR1     0x2ULL  // Streaming sector (way 2,3)
#define TAG_SECTOR0_S   0x9ULL  // Strong sector 0
#define TAG_SECTOR1_S   0xAULL  // Strong sector 1
#define TAG_BYPASS_L1   0xBULL  // Bypass L1

// Apply/remove tag macros
#define APPLY_TAG(ptr, tag) ((void*)((uint64_t)(ptr) | ((tag) << 56)))
#define REMOVE_TAG(ptr)     ((void*)((uint64_t)(ptr) & 0x00FFFFFFFFFFFFFFULL))

// Force pointer through memory to prevent compiler optimization
#define FORCE_PTR(ptr) do { \
    void* volatile _tmp = (ptr); \
    (ptr) = _tmp; \
} while(0)

// Cache parameters
#define L1_SIZE      (64 * 1024)   // 64KB L1D
#define L1_WAYS      4
#define L1_LINE      256           // 256B cache line
#define SECTOR_SIZE  (L1_SIZE / 2) // 32KB per sector

// Test sizes designed to stress cache
#define REUSE_SIZE   (24 * 1024)        // 24KB - fits in one sector
#define STREAM_SIZE  (256 * 1024)       // 256KB - much larger than L1
#define NUM_ITERS    50

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

// Memory barrier
static inline void memory_fence(void) {
    __asm__ volatile("dmb ish" ::: "memory");
}

// Flush data from cache
static void flush_cache(void* ptr, size_t size) {
    char* p = (char*)ptr;
    for (size_t i = 0; i < size; i += L1_LINE) {
        __asm__ volatile("dc civac, %0" :: "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
}

/*
 * Test: Mixed workload demonstrating sector cache benefit
 *
 * Scenario: We have "hot" data that should stay in cache (reuse)
 *           and "cold" data that's streaming through (one-time access)
 *
 * Without hints: Streaming data evicts hot data
 * With hints: Hot data in Sector 0, streaming in Sector 1 = hot data stays
 */
static double test_mixed_workload_no_hint(
    float* reuse_data, float* stream_data,
    size_t reuse_n, size_t stream_n, int outer_iters)
{
    volatile float sum = 0.0f;
    uint64_t start, end;

    // Flush caches first
    flush_cache(reuse_data, reuse_n * sizeof(float));
    flush_cache(stream_data, stream_n * sizeof(float));
    memory_fence();

    start = read_cycle();

    for (int outer = 0; outer < outer_iters; outer++) {
        float local_sum = 0.0f;

        // Step 1: Load hot/reuse data into cache (simulates initial load)
        for (size_t i = 0; i < reuse_n; i += 16) {
            local_sum += reuse_data[i];
        }

        // Step 2: Stream through cold data (this can evict hot data!)
        for (size_t i = 0; i < stream_n; i += 16) {
            local_sum += stream_data[i];
        }

        // Step 3: Access hot data again - if evicted, will miss
        for (size_t i = 0; i < reuse_n; i += 16) {
            local_sum += reuse_data[i];
        }

        sum += local_sum;
    }

    end = read_cycle();
    memory_fence();

    (void)sum;
    return (double)(end - start);
}

static double test_mixed_workload_with_hint(
    float* reuse_data, float* stream_data,
    size_t reuse_n, size_t stream_n, int outer_iters)
{
    volatile float sum = 0.0f;
    uint64_t start, end;

    // Apply tags - compiler can't optimize these away due to FORCE_PTR
    float* tagged_reuse = (float*)APPLY_TAG(reuse_data, TAG_SECTOR0);
    float* tagged_stream = (float*)APPLY_TAG(stream_data, TAG_SECTOR1);
    FORCE_PTR(tagged_reuse);
    FORCE_PTR(tagged_stream);

    // Flush caches first
    flush_cache(reuse_data, reuse_n * sizeof(float));
    flush_cache(stream_data, stream_n * sizeof(float));
    memory_fence();

    start = read_cycle();

    for (int outer = 0; outer < outer_iters; outer++) {
        float local_sum = 0.0f;

        // Step 1: Load hot data into Sector 0
        for (size_t i = 0; i < reuse_n; i += 16) {
            local_sum += tagged_reuse[i];
        }

        // Step 2: Stream through cold data in Sector 1 (won't evict Sector 0!)
        for (size_t i = 0; i < stream_n; i += 16) {
            local_sum += tagged_stream[i];
        }

        // Step 3: Access hot data again - should still be in Sector 0!
        for (size_t i = 0; i < reuse_n; i += 16) {
            local_sum += tagged_reuse[i];
        }

        sum += local_sum;
    }

    end = read_cycle();
    memory_fence();

    (void)sum;
    return (double)(end - start);
}

/*
 * SVE version for better memory bandwidth utilization
 */
static double test_mixed_sve_no_hint(
    float* reuse_data, float* stream_data,
    size_t reuse_n, size_t stream_n, int outer_iters)
{
    volatile float sum = 0.0f;
    uint64_t start, end;
    svbool_t pg = svptrue_b32();

    flush_cache(reuse_data, reuse_n * sizeof(float));
    flush_cache(stream_data, stream_n * sizeof(float));
    memory_fence();

    start = read_cycle();

    for (int outer = 0; outer < outer_iters; outer++) {
        svfloat32_t vsum = svdup_f32(0.0f);

        // Load reuse data
        for (size_t i = 0; i < reuse_n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &reuse_data[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }

        // Stream through cold data
        for (size_t i = 0; i < stream_n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &stream_data[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }

        // Access reuse data again
        for (size_t i = 0; i < reuse_n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &reuse_data[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }

        sum += svaddv_f32(pg, vsum);
    }

    end = read_cycle();
    memory_fence();

    (void)sum;
    return (double)(end - start);
}

static double test_mixed_sve_with_hint(
    float* reuse_data, float* stream_data,
    size_t reuse_n, size_t stream_n, int outer_iters)
{
    volatile float sum = 0.0f;
    uint64_t start, end;
    svbool_t pg = svptrue_b32();

    float* tagged_reuse = (float*)APPLY_TAG(reuse_data, TAG_SECTOR0);
    float* tagged_stream = (float*)APPLY_TAG(stream_data, TAG_SECTOR1);
    FORCE_PTR(tagged_reuse);
    FORCE_PTR(tagged_stream);

    flush_cache(reuse_data, reuse_n * sizeof(float));
    flush_cache(stream_data, stream_n * sizeof(float));
    memory_fence();

    start = read_cycle();

    for (int outer = 0; outer < outer_iters; outer++) {
        svfloat32_t vsum = svdup_f32(0.0f);

        // Load reuse data to Sector 0
        for (size_t i = 0; i < reuse_n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &tagged_reuse[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }

        // Stream cold data to Sector 1
        for (size_t i = 0; i < stream_n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &tagged_stream[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }

        // Access reuse data again (should be in Sector 0)
        for (size_t i = 0; i < reuse_n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &tagged_reuse[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }

        sum += svaddv_f32(pg, vsum);
    }

    end = read_cycle();
    memory_fence();

    (void)sum;
    return (double)(end - start);
}

/*
 * L1 Bypass test - measure direct-to-L2 performance
 */
static double test_bypass_streaming(float* data, size_t n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;
    svbool_t pg = svptrue_b32();

    float* tagged = (float*)APPLY_TAG(data, TAG_BYPASS_L1);
    FORCE_PTR(tagged);

    flush_cache(data, n * sizeof(float));
    memory_fence();

    start = read_cycle();

    for (int iter = 0; iter < iters; iter++) {
        svfloat32_t vsum = svdup_f32(0.0f);
        for (size_t i = 0; i < n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &tagged[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }
        sum += svaddv_f32(pg, vsum);
    }

    end = read_cycle();
    memory_fence();

    (void)sum;
    return (double)(end - start);
}

static double test_normal_streaming(float* data, size_t n, int iters) {
    volatile float sum = 0.0f;
    uint64_t start, end;
    svbool_t pg = svptrue_b32();

    flush_cache(data, n * sizeof(float));
    memory_fence();

    start = read_cycle();

    for (int iter = 0; iter < iters; iter++) {
        svfloat32_t vsum = svdup_f32(0.0f);
        for (size_t i = 0; i < n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &data[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }
        sum += svaddv_f32(pg, vsum);
    }

    end = read_cycle();
    memory_fence();

    (void)sum;
    return (double)(end - start);
}

int main(int argc, char* argv[]) {
    printf("=== A64FX Sector Cache Hint Test V2 ===\n\n");

    uint64_t freq = get_freq();
    printf("Timer frequency: %lu Hz\n", freq);
    printf("L1 Data Cache: %d KB (4-way, 2 sectors)\n", L1_SIZE / 1024);
    printf("Sector size: %d KB each\n", SECTOR_SIZE / 1024);
    printf("\n");

    // Allocate aligned buffers
    float* reuse_data = NULL;
    float* stream_data = NULL;
    posix_memalign((void**)&reuse_data, L1_LINE, REUSE_SIZE);
    posix_memalign((void**)&stream_data, L1_LINE, STREAM_SIZE);

    if (!reuse_data || !stream_data) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    size_t reuse_n = REUSE_SIZE / sizeof(float);
    size_t stream_n = STREAM_SIZE / sizeof(float);

    // Initialize with random data
    for (size_t i = 0; i < reuse_n; i++) {
        reuse_data[i] = (float)(i % 100) / 10.0f;
    }
    for (size_t i = 0; i < stream_n; i++) {
        stream_data[i] = (float)(i % 100) / 10.0f;
    }

    printf("Test configuration:\n");
    printf("  Reuse data: %d KB (fits in one sector)\n", (int)(REUSE_SIZE / 1024));
    printf("  Stream data: %d KB (>> L1 size, will cause evictions)\n", (int)(STREAM_SIZE / 1024));
    printf("  Iterations: %d\n", NUM_ITERS);
    printf("\n");

    double cycles_no_hint, cycles_with_hint;
    double speedup;

    // Warmup
    test_normal_streaming(stream_data, stream_n, 5);

    printf("=== Test 1: Mixed Workload (Scalar) ===\n");
    printf("Pattern: Load reuse -> Stream cold -> Access reuse again\n\n");

    cycles_no_hint = test_mixed_workload_no_hint(reuse_data, stream_data,
                                                  reuse_n, stream_n, NUM_ITERS);
    cycles_with_hint = test_mixed_workload_with_hint(reuse_data, stream_data,
                                                      reuse_n, stream_n, NUM_ITERS);
    speedup = cycles_no_hint / cycles_with_hint;

    printf("  No hint:     %.0f cycles\n", cycles_no_hint);
    printf("  With hints:  %.0f cycles\n", cycles_with_hint);
    printf("  Speedup:     %.2fx\n", speedup);
    printf("\n");

    printf("=== Test 2: Mixed Workload (SVE) ===\n");

    cycles_no_hint = test_mixed_sve_no_hint(reuse_data, stream_data,
                                             reuse_n, stream_n, NUM_ITERS);
    cycles_with_hint = test_mixed_sve_with_hint(reuse_data, stream_data,
                                                 reuse_n, stream_n, NUM_ITERS);
    speedup = cycles_no_hint / cycles_with_hint;

    printf("  No hint:     %.0f cycles\n", cycles_no_hint);
    printf("  With hints:  %.0f cycles\n", cycles_with_hint);
    printf("  Speedup:     %.2fx\n", speedup);
    printf("\n");

    printf("=== Test 3: L1 Bypass (TAG 0xB) ===\n");
    printf("Streaming large array - bypass L1 to go directly to L2\n\n");

    double bytes = (double)STREAM_SIZE * NUM_ITERS;

    cycles_no_hint = test_normal_streaming(stream_data, stream_n, NUM_ITERS);
    double bw_normal = bytes / (cycles_no_hint / freq) / 1e9;

    cycles_with_hint = test_bypass_streaming(stream_data, stream_n, NUM_ITERS);
    double bw_bypass = bytes / (cycles_with_hint / freq) / 1e9;

    printf("  Normal:      %.0f cycles (%.2f GB/s)\n", cycles_no_hint, bw_normal);
    printf("  L1 Bypass:   %.0f cycles (%.2f GB/s)\n", cycles_with_hint, bw_bypass);
    printf("  Speedup:     %.2fx\n", cycles_no_hint / cycles_with_hint);
    printf("\n");

    printf("=== Summary ===\n");
    printf("- L1 Bypass (0xB): Effective for streaming, avoids L1 pollution\n");
    printf("- Sector hints: Most effective when reuse data fits in one sector\n");
    printf("  and streaming data would otherwise evict it\n");

    free(reuse_data);
    free(stream_data);

    return 0;
}
