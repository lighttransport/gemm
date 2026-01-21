// bench_pure_load.c
// Test pure load throughput with minimal address calculation

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

static inline uint64_t rdtsc(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline svint32_t sve_ld1rw_s32(svbool_t pg, const int32_t* ptr) {
    svint32_t result;
    __asm__ volatile(
        "ld1rw {%0.s}, %1/z, [%2]"
        : "=w"(result)
        : "Upl"(pg), "r"(ptr)
        : "memory"
    );
    return result;
}

#define USE_RESULT(v) __asm__ volatile("" :: "w"(v) : "memory")

// Test 1: Just ld1b loads (64B each), sequential
void test_ld1b_seq(const int8_t* data, int n_loads) {
    svbool_t pg = svptrue_b8();
    const int8_t* p = data;

    for (int i = 0; i < n_loads; i++) {
        svint8_t v = svld1_s8(pg, p);
        USE_RESULT(v);
        p += 64;
    }
}

// Test 2: ld1b with fixed stride (256B apart, like K matrix)
void test_ld1b_stride256(const int8_t* data, int n_iters) {
    svbool_t pg = svptrue_b8();
    const int8_t* p = data;

    for (int i = 0; i < n_iters; i++) {
        svint8_t k0 = svld1_s8(pg, p);
        svint8_t k1 = svld1_s8(pg, p + 64);
        svint8_t k2 = svld1_s8(pg, p + 128);
        svint8_t k3 = svld1_s8(pg, p + 192);
        USE_RESULT(k0); USE_RESULT(k1); USE_RESULT(k2); USE_RESULT(k3);
        p += 256;
    }
}

// Test 3: Just ld1rw loads (4B broadcast)
void test_ld1rw_seq(const int32_t* data, int n_loads) {
    svbool_t pg = svptrue_b32();
    const int32_t* p = data;

    for (int i = 0; i < n_loads; i++) {
        svint32_t v = sve_ld1rw_s32(pg, p);
        USE_RESULT(v);
        p += 1;  // 4 bytes
    }
}

// Test 4: ld1rw with large stride (256B apart, like Q matrix rows)
void test_ld1rw_stride256(const int8_t* data, int n_iters) {
    svbool_t pg = svptrue_b32();
    const int8_t* p = data;

    for (int i = 0; i < n_iters; i++) {
        svint32_t q0 = sve_ld1rw_s32(pg, (const int32_t*)p);
        svint32_t q1 = sve_ld1rw_s32(pg, (const int32_t*)(p + 256));
        svint32_t q2 = sve_ld1rw_s32(pg, (const int32_t*)(p + 512));
        svint32_t q3 = sve_ld1rw_s32(pg, (const int32_t*)(p + 768));
        USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3);
        p += 4;
    }
}

// Test 5: Mixed ld1b + ld1rw (like actual kernel)
void test_mixed(const int8_t* K, const int8_t* Q, int n_iters) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svptrue_b32();
    const int8_t* k = K;
    const int8_t* q = Q;

    for (int i = 0; i < n_iters; i++) {
        svint8_t k0 = svld1_s8(pg, k);
        svint8_t k1 = svld1_s8(pg, k + 64);
        svint8_t k2 = svld1_s8(pg, k + 128);
        svint8_t k3 = svld1_s8(pg, k + 192);
        k += 256;

        svint32_t q0 = sve_ld1rw_s32(pg32, (const int32_t*)q);
        svint32_t q1 = sve_ld1rw_s32(pg32, (const int32_t*)(q + 256));
        svint32_t q2 = sve_ld1rw_s32(pg32, (const int32_t*)(q + 512));
        svint32_t q3 = sve_ld1rw_s32(pg32, (const int32_t*)(q + 768));
        q += 4;

        USE_RESULT(k0); USE_RESULT(k1); USE_RESULT(k2); USE_RESULT(k3);
        USE_RESULT(q0); USE_RESULT(q1); USE_RESULT(q2); USE_RESULT(q3);
    }
}

int main(int argc, char** argv) {
    int iters = 10000;
    int warmup = 1000;

    if (argc > 1) iters = atoi(argv[1]);

    printf("==============================================\n");
    printf("Pure Load Throughput Tests\n");
    printf("==============================================\n");
    printf("A64FX: 2 load pipes x 64B = 128 B/cycle peak\n\n");

    // Allocate large buffer to avoid cache conflicts
    int8_t* data = (int8_t*)aligned_alloc(256, 64 * 1024);  // 64KB
    memset(data, 1, 64 * 1024);

    printf("%-30s %8s %10s %10s %8s\n", "Test", "Loads", "Bytes", "Cycles", "B/cy");
    printf("------------------------------ -------- ---------- ---------- --------\n");

    // Test 1: ld1b sequential (256 loads x 64B = 16KB)
    {
        int n_loads = 256;
        size_t bytes = n_loads * 64;

        for (int i = 0; i < warmup; i++) test_ld1b_seq(data, n_loads);

        uint64_t start = rdtsc();
        for (int i = 0; i < iters; i++) test_ld1b_seq(data, n_loads);
        uint64_t end = rdtsc();

        double cycles = (double)(end - start) / iters * 20.0;
        printf("%-30s %8d %10zu %10.1f %8.1f\n", "ld1b seq (64B)", n_loads, bytes, cycles, bytes/cycles);
    }

    // Test 2: ld1b stride 256 (64 iters x 4 loads x 64B = 16KB)
    {
        int n_iters = 64;
        size_t bytes = n_iters * 4 * 64;

        for (int i = 0; i < warmup; i++) test_ld1b_stride256(data, n_iters);

        uint64_t start = rdtsc();
        for (int i = 0; i < iters; i++) test_ld1b_stride256(data, n_iters);
        uint64_t end = rdtsc();

        double cycles = (double)(end - start) / iters * 20.0;
        printf("%-30s %8d %10zu %10.1f %8.1f\n", "ld1b stride256 (4x64B)", n_iters*4, bytes, cycles, bytes/cycles);
    }

    // Test 3: ld1rw sequential (256 loads x 4B = 1KB)
    {
        int n_loads = 256;
        size_t bytes = n_loads * 4;

        for (int i = 0; i < warmup; i++) test_ld1rw_seq((const int32_t*)data, n_loads);

        uint64_t start = rdtsc();
        for (int i = 0; i < iters; i++) test_ld1rw_seq((const int32_t*)data, n_loads);
        uint64_t end = rdtsc();

        double cycles = (double)(end - start) / iters * 20.0;
        printf("%-30s %8d %10zu %10.1f %8.1f\n", "ld1rw seq (4B broadcast)", n_loads, bytes, cycles, bytes/cycles);
    }

    // Test 4: ld1rw stride 256 (64 iters x 4 loads x 4B = 1KB)
    {
        int n_iters = 64;
        size_t bytes = n_iters * 4 * 4;

        for (int i = 0; i < warmup; i++) test_ld1rw_stride256(data, n_iters);

        uint64_t start = rdtsc();
        for (int i = 0; i < iters; i++) test_ld1rw_stride256(data, n_iters);
        uint64_t end = rdtsc();

        double cycles = (double)(end - start) / iters * 20.0;
        printf("%-30s %8d %10zu %10.1f %8.1f\n", "ld1rw stride256 (4x4B)", n_iters*4, bytes, cycles, bytes/cycles);
    }

    // Test 5: Mixed ld1b + ld1rw (64 iters x (4x64B + 4x4B) = 17.4KB)
    {
        int n_iters = 64;
        size_t bytes = n_iters * (4*64 + 4*4);

        int8_t* K = data;
        int8_t* Q = data + 32*1024;  // Separate to avoid conflicts

        for (int i = 0; i < warmup; i++) test_mixed(K, Q, n_iters);

        uint64_t start = rdtsc();
        for (int i = 0; i < iters; i++) test_mixed(K, Q, n_iters);
        uint64_t end = rdtsc();

        double cycles = (double)(end - start) / iters * 20.0;
        printf("%-30s %8d %10zu %10.1f %8.1f\n", "mixed ld1b+ld1rw (Q@K pattern)", n_iters*8, bytes, cycles, bytes/cycles);
    }

    printf("\n");
    printf("Expected: 128 B/cycle for ld1b (2 pipes x 64B)\n");
    printf("ld1rw: loads 4B but broadcasts to 64B vector\n");

    free(data);
    return 0;
}
