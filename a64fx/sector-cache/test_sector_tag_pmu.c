/*
 * A64FX Sector Cache Tag Bit PMU Verification Test
 *
 * Proves that sector cache tag bits (address bits [59:56]) reach hardware
 * by measuring SC3 group PMU events via fapp:
 *
 *   0x0250/0x0252 (L1_PIPEx_VAL_IU_TAG_ADRS_SCE): fires when SCE bit = 1
 *   0x02a0/0x02a1 (L1_PIPEx_VAL_IU_NOT_SEC0):     fires when sector_id != 0
 *
 * Tag bit layout (address bits [59:56]):
 *   Bit 59: SCE (Sector Cache Enable)
 *   Bit 58: PFE (Prefetch Enable)
 *   Bit 57: sector_id[1]
 *   Bit 56: sector_id[0]
 *
 * Four test regions with identical SVE streaming work, different tags:
 *   tag_0x0: SCE=0, sector_id=0  -> SCE PMU ~0%, NOT_SEC0 ~0%
 *   tag_0x2: SCE=0, sector_id=2  -> SCE PMU ~0%, NOT_SEC0 >50%
 *   tag_0xA: SCE=1, sector_id=2  -> SCE PMU >50%, NOT_SEC0 >50%
 *   tag_0xB: SCE=1, sector_id=3  -> SCE PMU >50%, NOT_SEC0 >50%, L1 bypass
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#ifdef USE_FAPP
#include <fj_tool/fapp.h>
#endif

/* Sector cache tag values (bits [59:56] of virtual address) */
#define TAG_0x0  0x0ULL   /* Normal: no SCE, sector_id=0 */
#define TAG_0x2  0x2ULL   /* sector_id=2, no SCE */
#define TAG_0xA  0xAULL   /* SCE=1, sector_id=2 (strong sector 1) */
#define TAG_0xB  0xBULL   /* SCE=1, sector_id=3 (L1 bypass) */

/* Apply tag to pointer (set bits [59:56]) */
#define APPLY_TAG(ptr, tag) ((void*)((uint64_t)(ptr) | ((tag) << 56)))

/* Force pointer through memory to prevent compiler from stripping tag */
#define FORCE_PTR(ptr) do { \
    void* volatile _tmp = (ptr); \
    (ptr) = _tmp; \
} while(0)

/* Cache parameters */
#define L1_LINE      256           /* A64FX L1 cache line = 256 bytes */
#define DATA_SIZE    (256 * 1024)  /* 256KB test array */
#define NUM_ITERS    200           /* iterations per region */

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

static inline void memory_fence(void) {
    __asm__ volatile("dmb ish" ::: "memory");
}

/* Flush data from cache using dc civac */
static void flush_cache(void* ptr, size_t size) {
    char* p = (char*)ptr;
    for (size_t i = 0; i < size; i += L1_LINE) {
        __asm__ volatile("dc civac, %0" :: "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
}

/*
 * SVE streaming sum kernel.
 * Identical work for every tag -- only the pointer's upper bits differ.
 * MUST be noinline: if inlined, the compiler sees all tagged pointers alias
 * the same data and deduplicates them, stripping the tag bits.
 */
static __attribute__((noinline)) uint64_t run_streaming_sve(float* data, float* tagged_ptr,
                                   size_t n, int iters,
                                   const char* region_name) {
    volatile float sink = 0.0f;
    svbool_t pg = svptrue_b32();
    uint64_t start, end;

    /* Compiler barrier: keep tagged_ptr alive in a register */
    __asm__ volatile("" : "+r"(tagged_ptr));

    /* Flush using the real (untagged) address */
    flush_cache(data, n * sizeof(float));
    memory_fence();

#ifdef USE_FAPP
    fapp_start(region_name, 1, 0);
#endif

    start = read_cycle();

    for (int iter = 0; iter < iters; iter++) {
        svfloat32_t vsum = svdup_f32(0.0f);
        for (size_t i = 0; i < n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &tagged_ptr[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }
        sink += svaddv_f32(pg, vsum);
    }

    end = read_cycle();

#ifdef USE_FAPP
    fapp_stop(region_name, 1, 0);
#endif

    memory_fence();
    (void)sink;
    return end - start;
}

int main(int argc, char* argv[]) {
    printf("=== A64FX Sector Cache Tag Bit PMU Verification ===\n\n");

    uint64_t freq = get_freq();
    size_t n = DATA_SIZE / sizeof(float);

    printf("Timer frequency : %lu Hz\n", freq);
    printf("Data size       : %d KB\n", DATA_SIZE / 1024);
    printf("Iterations      : %d\n", NUM_ITERS);
    printf("SVE vector len  : %lu bits\n", svcntb() * 8);
    printf("\n");

    /* Allocate aligned buffer */
    float* data = NULL;
    posix_memalign((void**)&data, L1_LINE, DATA_SIZE);
    if (!data) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Initialize with non-zero data */
    for (size_t i = 0; i < n; i++) {
        data[i] = (float)(i % 100) / 10.0f;
    }

    /* Create tagged pointers */
    float* ptr_0x0 = (float*)data;                          /* tag 0x0: no tag */
    float* ptr_0x2 = (float*)APPLY_TAG(data, TAG_0x2);
    float* ptr_0xA = (float*)APPLY_TAG(data, TAG_0xA);
    float* ptr_0xB = (float*)APPLY_TAG(data, TAG_0xB);

    /* Force through memory so compiler can't strip tags */
    FORCE_PTR(ptr_0x0);
    FORCE_PTR(ptr_0x2);
    FORCE_PTR(ptr_0xA);
    FORCE_PTR(ptr_0xB);

    /* ---- Pointer Tag Verification ---- */
    printf("=== Pointer Tag Verification ===\n");
    printf("ptr (tag=0x0) = 0x%016lx  upper_nibble=0x%lX  %s\n",
           (uint64_t)ptr_0x0, ((uint64_t)ptr_0x0 >> 56) & 0xF,
           (((uint64_t)ptr_0x0 >> 56) & 0xF) == 0x0 ? "OK" : "FAIL");
    printf("ptr (tag=0x2) = 0x%016lx  upper_nibble=0x%lX  %s\n",
           (uint64_t)ptr_0x2, ((uint64_t)ptr_0x2 >> 56) & 0xF,
           (((uint64_t)ptr_0x2 >> 56) & 0xF) == 0x2 ? "OK" : "FAIL");
    printf("ptr (tag=0xA) = 0x%016lx  upper_nibble=0x%lX  %s\n",
           (uint64_t)ptr_0xA, ((uint64_t)ptr_0xA >> 56) & 0xF,
           (((uint64_t)ptr_0xA >> 56) & 0xF) == 0xA ? "OK" : "FAIL");
    printf("ptr (tag=0xB) = 0x%016lx  upper_nibble=0x%lX  %s\n",
           (uint64_t)ptr_0xB, ((uint64_t)ptr_0xB >> 56) & 0xF,
           (((uint64_t)ptr_0xB >> 56) & 0xF) == 0xB ? "OK" : "FAIL");
    printf("\n");

    /* ---- Tag Bit Layout Reference ---- */
    printf("=== Tag Bit Layout (bits [59:56]) ===\n");
    printf("Tag  Binary  SCE  sector_id  Expected SCE PMU  Expected NOT_SEC0 PMU\n");
    printf("0x0  0000    0    0          ~0%%               ~0%%\n");
    printf("0x2  0010    0    2          ~0%%               >50%%\n");
    printf("0xA  1010    1    2          >50%%              >50%%\n");
    printf("0xB  1011    1    3          >50%%              >50%%\n");
    printf("\n");

    /* ---- Warmup ---- */
    run_streaming_sve(data, ptr_0x0, n, 5, "warmup");

    /* ---- Run all four regions ---- */
    uint64_t cyc_0x0 = run_streaming_sve(data, ptr_0x0, n, NUM_ITERS, "tag_0x0");
    uint64_t cyc_0x2 = run_streaming_sve(data, ptr_0x2, n, NUM_ITERS, "tag_0x2");
    uint64_t cyc_0xA = run_streaming_sve(data, ptr_0xA, n, NUM_ITERS, "tag_0xA");
    uint64_t cyc_0xB = run_streaming_sve(data, ptr_0xB, n, NUM_ITERS, "tag_0xB");

    /* ---- Results ---- */
    double bytes_total = (double)DATA_SIZE * NUM_ITERS;

    printf("=== Cycle Results ===\n");
    printf("tag_0x0: %10lu cycles (%6.2f GB/s) [baseline]\n",
           cyc_0x0, bytes_total / ((double)cyc_0x0 / freq) / 1e9);
    printf("tag_0x2: %10lu cycles (%6.2f GB/s) ratio=%.2fx\n",
           cyc_0x2, bytes_total / ((double)cyc_0x2 / freq) / 1e9,
           (double)cyc_0x0 / cyc_0x2);
    printf("tag_0xA: %10lu cycles (%6.2f GB/s) ratio=%.2fx\n",
           cyc_0xA, bytes_total / ((double)cyc_0xA / freq) / 1e9,
           (double)cyc_0x0 / cyc_0xA);
    printf("tag_0xB: %10lu cycles (%6.2f GB/s) ratio=%.2fx  <-- L1 bypass\n",
           cyc_0xB, bytes_total / ((double)cyc_0xB / freq) / 1e9,
           (double)cyc_0x0 / cyc_0xB);
    printf("\n");

    /* ---- Interpretation ---- */
    double ratio_0xB = (double)cyc_0x0 / cyc_0xB;
    printf("=== Quick Interpretation ===\n");
    if (ratio_0xB > 1.3) {
        printf("PASS: tag_0xB shows %.2fx speedup -> tags reach hardware\n", ratio_0xB);
    } else {
        printf("WARN: tag_0xB shows only %.2fx ratio (expected ~1.5-2.0x)\n", ratio_0xB);
        printf("      -> tags may not be reaching hardware; check disassembly\n");
    }
    printf("\nRun with fapp to verify SC3 PMU events (SCE / NOT_SEC0 counts).\n");
    printf("See run_pmu_verify.sh for fapp commands.\n");

    /* ---- PMU Ratio Formulas ---- */
    printf("\n=== PMU Ratio Formulas (for fapp output) ===\n");
    printf("SCE_usage_ratio  = (0x0250 + 0x0252) / (0x0240 + 0x0241)\n");
    printf("NOT_SEC0_ratio   = (0x02a0 + 0x02a1) / (0x0240 + 0x0241)\n");
    printf("\n");

    free(data);
    return 0;
}
