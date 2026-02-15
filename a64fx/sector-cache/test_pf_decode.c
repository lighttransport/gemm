/*
 * A64FX IMP_PF_STREAM_DETECT_CTRL_EL0 (S3_3_C11_C4_0) — Bit Field Decoder
 *
 * Phase 1: Bit-by-bit scan to find exact valid mask
 * Phase 2: Field boundary identification
 * Phase 3: Streaming benchmark to measure prefetch impact per field
 *
 * Build:
 *   fcc -Nclang -O2 -march=armv8.2-a+sve -o test_pf_decode test_pf_decode.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

/* ===== Register access ===== */

static inline uint64_t rd_pf(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, S3_3_C11_C4_0" : "=r"(v));
    return v;
}

static inline void wr_pf(uint64_t v) {
    __asm__ volatile("msr S3_3_C11_C4_0, %0" :: "r"(v));
    __asm__ volatile("isb" ::: "memory");
}

/* ===== Timer ===== */

static inline uint64_t rdcyc(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

/* cntvct_el0 frequency (typically 25MHz on A64FX) */
static inline uint64_t rdfreq(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}

static inline void memory_fence(void) {
    __asm__ volatile("dmb ish" ::: "memory");
}

/* ===== Cache flush ===== */

static void flush_cache(void* ptr, size_t size) {
    char* p = (char*)ptr;
    for (size_t i = 0; i < size; i += 256) {
        __asm__ volatile("dc civac, %0" :: "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb ish" ::: "memory");
}

/* ===== Streaming benchmark (SVE, noinline) ===== */

static __attribute__((noinline)) uint64_t stream_sum(float* data, size_t n, int iters) {
    svbool_t pg = svptrue_b32();
    volatile float sink = 0.0f;

    __asm__ volatile("" : "+r"(data));

    uint64_t start = rdcyc();

    for (int iter = 0; iter < iters; iter++) {
        svfloat32_t vsum = svdup_f32(0.0f);
        for (size_t i = 0; i < n; i += svcntw()) {
            svfloat32_t v = svld1_f32(pg, &data[i]);
            vsum = svadd_f32_m(pg, vsum, v);
        }
        sink += svaddv_f32(pg, vsum);
    }

    uint64_t end = rdcyc();
    (void)sink;
    return end - start;
}

int main(int argc, char* argv[]) {
    uint64_t orig = rd_pf();

    printf("=== IMP_PF_STREAM_DETECT_CTRL_EL0 Bit Field Decoder ===\n\n");
    printf("Original value: 0x%016lx\n\n", orig);

    /* ===== Phase 1: Bit-by-bit scan ===== */
    printf("--- Phase 1: Valid Bit Scan ---\n");
    uint64_t valid_mask = 0;

    for (int bit = 0; bit < 64; bit++) {
        uint64_t test = 1ULL << bit;
        wr_pf(test);
        uint64_t rb = rd_pf();
        if (rb & test) {
            valid_mask |= test;
        }
        /* Check for any other bits that also got set (linked fields) */
        if (rb & ~test) {
            printf("  bit %2d: set caused side-effect bits 0x%016lx\n", bit, rb & ~test);
        }
    }
    wr_pf(orig); /* restore */

    printf("\n  Valid bit mask: 0x%016lx\n\n", valid_mask);

    /* Print bit map */
    printf("  Bit map (63→0):\n  ");
    for (int bit = 63; bit >= 0; bit--) {
        printf("%c", (valid_mask >> bit) & 1 ? '1' : '.');
        if (bit % 4 == 0) printf(" ");
    }
    printf("\n  ");
    for (int bit = 63; bit >= 0; bit--) {
        if (bit % 4 == 0) {
            printf("%-4d", bit);
            bit -= 0; /* already at the boundary */
        } else {
            printf(" ");
        }
    }
    printf("\n\n");

    /* ===== Phase 2: Identify fields ===== */
    printf("--- Phase 2: Field Identification ---\n\n");

    /* Group consecutive valid bits into fields */
    int in_field = 0;
    int field_start = 0;
    int nfields = 0;
    int field_lo[32], field_hi[32];

    for (int bit = 63; bit >= -1; bit--) {
        int is_valid = (bit >= 0) ? ((valid_mask >> bit) & 1) : 0;
        if (is_valid && !in_field) {
            field_start = bit;
            in_field = 1;
        } else if (!is_valid && in_field) {
            field_hi[nfields] = field_start;
            field_lo[nfields] = bit + 1;
            nfields++;
            in_field = 0;
        }
    }

    printf("  Found %d fields:\n\n", nfields);
    printf("  %-8s  %-8s  %-6s  %-12s\n", "Bits", "Width", "MaxVal", "Mask");
    printf("  %-8s  %-8s  %-6s  %-12s\n", "----", "-----", "------", "----");
    for (int f = 0; f < nfields; f++) {
        int width = field_hi[f] - field_lo[f] + 1;
        uint64_t fmask = ((1ULL << width) - 1) << field_lo[f];
        printf("  [%2d:%2d]  %d bits    %-6d  0x%016lx\n",
               field_hi[f], field_lo[f], width, (1 << width) - 1, fmask);
    }
    printf("\n");

    /* Test each field: try all possible values and verify readback */
    printf("  Per-field value scan:\n\n");
    for (int f = 0; f < nfields; f++) {
        int width = field_hi[f] - field_lo[f] + 1;
        int max_val = (1 << width) - 1;
        uint64_t fmask = ((1ULL << width) - 1) << field_lo[f];

        printf("  Field [%d:%d] (%d-bit):", field_hi[f], field_lo[f], width);
        int all_match = 1;
        for (int v = 0; v <= max_val; v++) {
            uint64_t test = (uint64_t)v << field_lo[f];
            wr_pf(test);
            uint64_t rb = rd_pf();
            uint64_t got = (rb & fmask) >> field_lo[f];
            if ((int)got != v) {
                printf(" wrote %d→got %lu", v, got);
                all_match = 0;
            }
        }
        if (all_match) {
            printf(" all %d values [0..%d] accepted", max_val + 1, max_val);
        }
        printf("\n");
    }
    wr_pf(orig);
    printf("\n");

    /* ===== Phase 3: Streaming benchmark per field ===== */
    printf("--- Phase 3: Prefetch Impact Benchmark ---\n\n");

    /* Allocate test buffer: 4MB (fits in L2, much larger than L1) */
    size_t buf_size = 4 * 1024 * 1024;
    size_t n = buf_size / sizeof(float);
    float* data = NULL;
    posix_memalign((void**)&data, 256, buf_size);
    for (size_t i = 0; i < n; i++) data[i] = (float)(i % 100) / 10.0f;

    int iters = 50;

    /* Baseline: original register value */
    wr_pf(orig);
    flush_cache(data, buf_size);
    memory_fence();
    uint64_t cyc_base = stream_sum(data, n, iters);

    /* Baseline with all zeros */
    wr_pf(0);
    flush_cache(data, buf_size);
    memory_fence();
    uint64_t cyc_zero = stream_sum(data, n, iters);

    /* Baseline with all valid bits set */
    wr_pf(valid_mask);
    flush_cache(data, buf_size);
    memory_fence();
    uint64_t cyc_allset = stream_sum(data, n, iters);

    wr_pf(orig);

    uint64_t freq = rdfreq();
    printf("  Timer frequency: %lu Hz\n\n", freq);

    double bw_base   = (double)buf_size * iters / ((double)cyc_base)   * (double)freq / 1e9;
    double bw_zero   = (double)buf_size * iters / ((double)cyc_zero)   * (double)freq / 1e9;
    double bw_allset = (double)buf_size * iters / ((double)cyc_allset) * (double)freq / 1e9;

    printf("  Streaming 4MB × %d iters (SVE ld1w sum):\n\n", iters);
    printf("  %-24s  %12s  %8s  %8s\n", "Config", "Cycles", "GB/s", "Ratio");
    printf("  %-24s  %12s  %8s  %8s\n", "------", "------", "----", "-----");
    printf("  %-24s  %12lu  %8.2f  %8s\n", "original (0x0)", cyc_base, bw_base, "1.00x");
    printf("  %-24s  %12lu  %8.2f  %7.2fx\n", "all zeros", cyc_zero, bw_zero,
           (double)cyc_base / cyc_zero);
    printf("  %-24s  %12lu  %8.2f  %7.2fx\n", "all valid bits set", cyc_allset, bw_allset,
           (double)cyc_base / cyc_allset);
    printf("\n");

    /* Per-field sweep: try each field at each possible value */
    printf("  Per-field performance sweep:\n\n");
    for (int f = 0; f < nfields; f++) {
        int width = field_hi[f] - field_lo[f] + 1;
        int max_val = (1 << width) - 1;

        printf("  Field [%d:%d] (%d-bit, max=%d):\n",
               field_hi[f], field_lo[f], width, max_val);
        printf("    %-6s  %12s  %8s  %8s\n", "Value", "Cycles", "GB/s", "vs base");

        for (int v = 0; v <= max_val; v++) {
            uint64_t test = (uint64_t)v << field_lo[f];
            wr_pf(test);
            flush_cache(data, buf_size);
            memory_fence();
            uint64_t cyc = stream_sum(data, n, iters);
            double bw = (double)buf_size * iters / ((double)cyc) * (double)freq / 1e9;
            printf("    %-6d  %12lu  %8.2f  %7.2fx\n",
                   v, cyc, bw, (double)cyc_base / cyc);
        }
        printf("\n");
    }

    wr_pf(orig);
    printf("  Register restored to original: 0x%016lx\n\n", rd_pf());

    free(data);
    return 0;
}
