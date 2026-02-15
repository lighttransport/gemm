#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lz.h"
#include "lz_fmt.h"

/* ------------------------------------------------------------------ */
/* Timing                                                              */
/* ------------------------------------------------------------------ */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* Simple LCG for reproducible pseudo-random data                      */
/* ------------------------------------------------------------------ */
static uint32_t rng_state = 12345u;
static uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

/* ------------------------------------------------------------------ */
/* Data generators                                                     */
/* ------------------------------------------------------------------ */
static void gen_zeros(uint8_t *buf, size_t n) {
    memset(buf, 0, n);
}

static void gen_random(uint8_t *buf, size_t n) {
    rng_state = 42;
    for (size_t i = 0; i < n; i++)
        buf[i] = (uint8_t)(rng_next() >> 16);
}

/* Simulated FP16 KV cache: many repeated exponent patterns */
static void gen_fp16(uint8_t *buf, size_t n) {
    rng_state = 77;
    for (size_t i = 0; i + 1 < n; i += 2) {
        /* FP16: sign(1) exp(5) frac(10)
         * Use small range of exponents with varying fractions */
        int exp = 14 + (int)(rng_next() % 5);  /* exponents 14-18 */
        int frac = (int)(rng_next() & 0x3FF);
        uint16_t val = (uint16_t)((exp << 10) | frac);
        buf[i]     = (uint8_t)(val);
        buf[i + 1] = (uint8_t)(val >> 8);
    }
    if (n & 1) buf[n - 1] = 0;
}

/* Simulated INT8 quantized weights: Gaussian-like distribution */
static void gen_int8(uint8_t *buf, size_t n) {
    rng_state = 99;
    for (size_t i = 0; i < n; i++) {
        /* Sum of 4 uniform -> approx Gaussian, centered at 128 */
        int sum = 0;
        for (int j = 0; j < 4; j++)
            sum += (int)(rng_next() & 0x3F);
        int val = sum - 128 + 128;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        buf[i] = (uint8_t)val;
    }
}

/* Mixed: 50% zero blocks, 50% random (simulates sparse activations) */
static void gen_sparse(uint8_t *buf, size_t n) {
    rng_state = 55;
    for (size_t i = 0; i < n; i += 64) {
        size_t chunk = (n - i < 64) ? n - i : 64;
        if (rng_next() & 1)
            memset(buf + i, 0, chunk);
        else
            for (size_t j = 0; j < chunk; j++)
                buf[i + j] = (uint8_t)(rng_next() >> 16);
    }
}

/* ------------------------------------------------------------------ */
/* Test infrastructure                                                 */
/* ------------------------------------------------------------------ */
typedef struct {
    const char *name;
    void (*gen)(uint8_t *, size_t);
} dataset_t;

static const dataset_t datasets[] = {
    { "zeros",     gen_zeros  },
    { "random",    gen_random },
    { "fp16_kv",   gen_fp16   },
    { "int8_wt",   gen_int8   },
    { "sparse",    gen_sparse },
};
#define N_DATASETS (sizeof(datasets) / sizeof(datasets[0]))

static int test_roundtrip(const char *label, const uint8_t *src, size_t src_size,
                           int level, int use_asm) {
    size_t bound = lz_compress_bound(src_size);
    uint8_t *comp = (uint8_t *)malloc(bound + LZ_SAFETY_MARGIN);
    uint8_t *decomp = (uint8_t *)malloc(src_size + LZ_SAFETY_MARGIN);
    if (!comp || !decomp) { free(comp); free(decomp); return -1; }
    memset(comp + bound, 0xDD, LZ_SAFETY_MARGIN);
    memset(decomp, 0xCC, src_size + LZ_SAFETY_MARGIN);

    size_t csize = lz_compress(src, src_size, comp, bound, level);
    if (csize == 0 && src_size > 0) {
        printf("  FAIL [%s] L%d: compress returned 0\n", label, level);
        free(comp); free(decomp);
        return 1;
    }

    size_t dsize;
    if (use_asm)
        dsize = lz_decompress_asm(comp, csize, decomp, src_size + LZ_SAFETY_MARGIN);
    else
        dsize = lz_decompress(comp, csize, decomp, src_size + LZ_SAFETY_MARGIN);

    int ok = (dsize == src_size) && (memcmp(src, decomp, src_size) == 0);
    if (!ok) {
        printf("  FAIL [%s] L%d %s: src=%zu comp=%zu decomp=%zu",
               label, level, use_asm ? "ASM" : "C  ",
               src_size, csize, dsize);
        if (dsize == src_size) {
            /* Find first mismatch */
            for (size_t i = 0; i < src_size; i++) {
                if (src[i] != decomp[i]) {
                    printf(" first_diff@%zu (0x%02x vs 0x%02x)", i, src[i], decomp[i]);
                    break;
                }
            }
        }
        printf("\n");
    }

    free(comp);
    free(decomp);
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Correctness tests                                                   */
/* ------------------------------------------------------------------ */
static int run_correctness(void) {
    int fails = 0;
    printf("=== Correctness Tests ===\n");

    /* Edge cases */
    {
        uint8_t empty[1] = {0};
        fails += test_roundtrip("empty", empty, 0, LZ_LEVEL_FAST, 0);
        fails += test_roundtrip("empty", empty, 0, LZ_LEVEL_FAST, 1);
    }
    {
        uint8_t one[1] = {0x42};
        fails += test_roundtrip("1byte", one, 1, LZ_LEVEL_FAST, 0);
        fails += test_roundtrip("1byte", one, 1, LZ_LEVEL_FAST, 1);
    }
    {
        uint8_t three[3] = {0xAA, 0xBB, 0xCC};
        fails += test_roundtrip("3byte", three, 3, LZ_LEVEL_FAST, 0);
        fails += test_roundtrip("3byte", three, 3, LZ_LEVEL_FAST, 1);
    }

    /* Standard sizes */
    size_t sizes[] = { 256, 1024, 4096, 16384, 65536, 256*1024 };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int d = 0; d < (int)N_DATASETS; d++) {
        for (int s = 0; s < nsizes; s++) {
            uint8_t *buf = (uint8_t *)malloc(sizes[s]);
            datasets[d].gen(buf, sizes[s]);

            char label[64];
            snprintf(label, sizeof(label), "%s/%zuK",
                     datasets[d].name, sizes[s] / 1024);

            fails += test_roundtrip(label, buf, sizes[s], LZ_LEVEL_FAST, 0);
            fails += test_roundtrip(label, buf, sizes[s], LZ_LEVEL_FAST, 1);
            fails += test_roundtrip(label, buf, sizes[s], LZ_LEVEL_BEST, 0);
            fails += test_roundtrip(label, buf, sizes[s], LZ_LEVEL_BEST, 1);

            free(buf);
        }
    }

    if (fails == 0)
        printf("  All correctness tests PASSED\n");
    else
        printf("  %d tests FAILED\n", fails);

    return fails;
}

/* ------------------------------------------------------------------ */
/* Token analysis: parse compressed stream and count instructions       */
/* ------------------------------------------------------------------ */
typedef struct {
    int total_tokens;
    int fast_match;       /* lit<15, mat 1..14, offset>=16 */
    int fast_lit_only;    /* lit<15, mat==0 */
    int fast_rle;         /* lit<15, mat 1..14, offset==1 */
    int fast_overlap;     /* lit<15, mat 1..14, offset 2..15 */
    int slow_lit;         /* lit==15 */
    int slow_match;       /* mat==15 */
    long total_lit_bytes;
    long total_match_bytes;
    long total_out_bytes;
    long total_rle_bytes;
    long total_overlap_bytes;
    int stored_blocks;
} token_stats_t;

/*
 * Instruction counts per ASM path (counted from lz_decompress_a64fx.S):
 *
 * FAST_MATCH (lit>0, mat 1..14, offset>=16):
 *   .Lloop top:  cmp, b.hs                                 = 2
 *   tag decode:  ldrb, lsr, and                             = 3
 *   slow check:  cmp, b.eq, cmp, b.eq                      = 4
 *   lit copy:    cbz(nt), ldp, stp                          = 3
 *   lit advance: add, add                                   = 2
 *   match check: cbz(nt)                                    = 1
 *   offset+len:  ldrh, add                                  = 2
 *   match setup: sub, cmp, b.lo(nt)                         = 3
 *   match copy:  ldp, stp, ldp, stp                         = 4
 *   finish:      add, b                                     = 2
 *   TOTAL = 26
 *
 * FAST_MATCH (lit==0, mat 1..14, offset>=16):
 *   Same but cbz w10 IS taken, skips ldp+stp                = 24
 *
 * FAST_LIT_ONLY (lit>0, mat==0):
 *   top + decode + slow_check + lit_copy + advance          = 14
 *   cbz w11 TAKEN → back to loop                            = 1
 *   TOTAL = 15
 *
 * FAST_LIT_ONLY (lit==0, mat==0):
 *   top + decode + slow_check                               = 9
 *   cbz w10 TAKEN (skip lit)                                = 1
 *   advance                                                 = 2
 *   cbz w11 TAKEN                                           = 1
 *   TOTAL = 13
 *
 * OVERLAP (offset 2..15):
 *   Same as FAST_MATCH up to b.lo TAKEN                     = 20
 *   add x6                                                  = 1
 *   cmp #1, b.ne TAKEN                                      = 2
 *   Per byte: ldrb, strb, cmp, b.lo                         = 4/byte
 *   Final: b .Lloop                                         = 1
 *   TOTAL = 24 + 4*match_len
 *
 * RLE (offset==1):
 *   Same up to b.lo TAKEN                                   = 20
 *   add x6, cmp #1, b.ne(nt)                               = 3
 *   ldrb + 3x orr (broadcast)                               = 4
 *   Per 8 bytes: sub, cmp, b.lt(nt), str, b                = 5
 *   Tail + loop back ≈ 3
 *   TOTAL ≈ 30 + 5*ceil(match_len/8)
 */

/* Instruction model constants */
#define INSN_FAST_MATCH_LIT    26   /* lit>0, offset>=16 */
#define INSN_FAST_MATCH_NOLIT  24   /* lit==0, offset>=16 */
#define INSN_FAST_LIT_ONLY     15   /* lit>0, mat==0 */
#define INSN_FAST_LIT_ONLY_Z   13   /* lit==0, mat==0 (degenerate) */
#define INSN_OVERLAP_BASE      24   /* fixed cost before byte loop */
#define INSN_OVERLAP_PER_BYTE   4
#define INSN_RLE_BASE          30   /* fixed cost before 8-byte loop */
#define INSN_RLE_PER_8BYTES     5
#define INSN_SLOW_LIT_BASE     14   /* tag decode + ext read */
#define INSN_SLOW_LIT_PER_EXT   4   /* per extension byte */
#define INSN_SLOW_LIT_PER_16    7   /* per 16-byte bulk copy */
#define INSN_SLOW_LIT_PER_TAIL  4   /* per tail byte */

static int read_ext_val(const uint8_t **ip, const uint8_t *end, int *ext_count) {
    int val = 0;
    while (*ip < end) {
        uint8_t b = *(*ip)++;
        val += b;
        (*ext_count)++;
        if (b != 255) break;
    }
    return val;
}

static void analyze_block_tokens(const uint8_t *src, size_t src_size,
                                  token_stats_t *st) {
    const uint8_t *ip = src;
    const uint8_t *ip_end = src + src_size;

    while (ip < ip_end) {
        uint8_t tag = *ip++;
        int lit_code = tag >> 4;
        int mat_code = tag & 0xF;
        int ext_count = 0;

        st->total_tokens++;

        /* Literal length */
        int lit_len = lit_code;
        int is_slow_lit = (lit_code == 15);
        if (is_slow_lit)
            lit_len += read_ext_val(&ip, ip_end, &ext_count);

        st->total_lit_bytes += lit_len;
        ip += lit_len; /* skip literal bytes */

        if (mat_code == 0) {
            /* Literal-only token */
            st->fast_lit_only++;
            st->total_out_bytes += lit_len;
            if (is_slow_lit) st->slow_lit++;
            continue;
        }

        /* Read offset */
        if (ip + 2 > ip_end) break;
        int offset = ip[0] | (ip[1] << 8);
        ip += 2;

        /* Match length */
        int match_len = mat_code + 3;
        int is_slow_match = (mat_code == 15);
        if (is_slow_match) {
            match_len += read_ext_val(&ip, ip_end, &ext_count);
            st->slow_match++;
        }
        if (is_slow_lit) st->slow_lit++;

        st->total_match_bytes += match_len;
        st->total_out_bytes += lit_len + match_len;

        if (offset == 1) {
            st->fast_rle++;
            st->total_rle_bytes += match_len;
        } else if (offset < 16) {
            st->fast_overlap++;
            st->total_overlap_bytes += match_len;
        } else {
            st->fast_match++;
        }
    }
}

static void analyze_compressed(const uint8_t *comp, size_t csize,
                                token_stats_t *st) {
    memset(st, 0, sizeof(*st));

    if (csize < sizeof(lz_frame_header_t)) return;

    lz_frame_header_t hdr;
    memcpy(&hdr, comp, sizeof(hdr));
    if (hdr.magic != LZ_MAGIC) return;

    const uint8_t *ip = comp + sizeof(lz_frame_header_t);
    const uint8_t *ip_end = comp + csize;

    while (ip + 4 <= ip_end) {
        uint32_t block_hdr = lz_read32(ip);
        ip += 4;
        int stored = (block_hdr & LZ_BLOCK_STORED) != 0;
        uint32_t block_size = block_hdr & ~LZ_BLOCK_STORED;
        if (ip + block_size > ip_end) break;

        if (stored) {
            st->stored_blocks++;
            st->total_out_bytes += block_size;
        } else {
            analyze_block_tokens(ip, block_size, st);
        }
        ip += block_size;
    }
}

/* Compute weighted average instruction count per token */
static double compute_avg_insn_per_token(const token_stats_t *st) {
    if (st->total_tokens == 0) return 0;

    long total_insns = 0;

    /* Fast match tokens (offset >= 16) */
    total_insns += (long)st->fast_match * INSN_FAST_MATCH_LIT;

    /* Literal-only tokens */
    total_insns += (long)st->fast_lit_only * INSN_FAST_LIT_ONLY;

    /* RLE tokens (offset == 1) */
    total_insns += (long)st->fast_rle * INSN_RLE_BASE;
    /* Add per-8-byte cost for RLE */
    if (st->fast_rle > 0) {
        long avg_rle_len = st->total_rle_bytes / st->fast_rle;
        total_insns += (long)st->fast_rle * (((avg_rle_len + 7) / 8) * INSN_RLE_PER_8BYTES);
    }

    /* Overlap tokens (offset 2..15) */
    total_insns += (long)st->fast_overlap * INSN_OVERLAP_BASE;
    if (st->fast_overlap > 0) {
        long avg_ovl_len = st->total_overlap_bytes / st->fast_overlap;
        total_insns += (long)st->fast_overlap * (avg_ovl_len * INSN_OVERLAP_PER_BYTE);
    }

    /* Slow path penalty approximation: +8 insns per slow-path token */
    total_insns += (long)st->slow_lit * 8;
    total_insns += (long)st->slow_match * 8;

    return (double)total_insns / st->total_tokens;
}

static void run_token_analysis(void) {
    printf("\n=== Token Analysis (256 KB, per compressed block) ===\n");

    size_t size = 256 * 1024;
    uint8_t *buf = (uint8_t *)malloc(size);
    size_t bound = lz_compress_bound(size);
    uint8_t *comp = (uint8_t *)malloc(bound);

    int levels[] = { LZ_LEVEL_FAST, LZ_LEVEL_BEST };

    for (int li = 0; li < 2; li++) {
        int level = levels[li];
        printf("\n  Level %d:\n", level);
        printf("  %-10s %7s %7s %7s %7s %7s %5s | %8s %6s %8s | %8s %7s\n",
               "Dataset", "tokens", "fmatch", "litonly", "rle", "overlap",
               "fast%", "out/tk", "lit/tk", "mat/tk",
               "insn/tk", "insn/B");

        for (int d = 0; d < (int)N_DATASETS; d++) {
            datasets[d].gen(buf, size);
            size_t csize = lz_compress(buf, size, comp, bound, level);

            token_stats_t st;
            analyze_compressed(comp, csize, &st);

            double avg_insn = compute_avg_insn_per_token(&st);
            double avg_out = st.total_tokens > 0
                ? (double)st.total_out_bytes / st.total_tokens : 0;
            double avg_lit = st.total_tokens > 0
                ? (double)st.total_lit_bytes / st.total_tokens : 0;
            double avg_mat = st.total_tokens > 0
                ? (double)st.total_match_bytes / st.total_tokens : 0;
            double insn_per_byte = avg_out > 0 ? avg_insn / avg_out : 0;

            int pct_fast = st.total_tokens > 0
                ? (int)(100.0 * (st.fast_match + st.fast_lit_only)
                        / st.total_tokens) : 0;

            printf("  %-10s %7d %7d %7d %7d %7d %4d%% | %8.1f %6.1f %8.1f | %8.1f %7.3f\n",
                   datasets[d].name,
                   st.total_tokens, st.fast_match, st.fast_lit_only,
                   st.fast_rle, st.fast_overlap,
                   pct_fast,
                   avg_out, avg_lit, avg_mat,
                   avg_insn, insn_per_byte);
        }
    }

    free(buf);
    free(comp);
}

/* ------------------------------------------------------------------ */
/* Compression ratio report                                            */
/* ------------------------------------------------------------------ */
static void run_ratio(void) {
    printf("\n=== Compression Ratio ===\n");
    printf("  %-12s %8s  %6s %6s  %6s %6s\n",
           "Dataset", "Size", "L3_cmp", "L3_%", "L9_cmp", "L9_%");

    size_t size = 256 * 1024;
    uint8_t *buf = (uint8_t *)malloc(size);
    size_t bound = lz_compress_bound(size);
    uint8_t *comp = (uint8_t *)malloc(bound);

    for (int d = 0; d < (int)N_DATASETS; d++) {
        datasets[d].gen(buf, size);

        size_t c3 = lz_compress(buf, size, comp, bound, LZ_LEVEL_FAST);
        size_t c9 = lz_compress(buf, size, comp, bound, LZ_LEVEL_BEST);

        printf("  %-12s %6zuK  %6zu %5.1f%%  %6zu %5.1f%%\n",
               datasets[d].name, size / 1024,
               c3, 100.0 * c3 / size,
               c9, 100.0 * c9 / size);
    }

    free(buf);
    free(comp);
}

/* ------------------------------------------------------------------ */
/* Throughput benchmark                                                */
/* ------------------------------------------------------------------ */
static void bench_throughput(const char *name, const uint8_t *src, size_t src_size,
                              int level, int iters) {
    size_t bound = lz_compress_bound(src_size);
    uint8_t *comp = (uint8_t *)malloc(bound + LZ_SAFETY_MARGIN);
    uint8_t *decomp = (uint8_t *)malloc(src_size + LZ_SAFETY_MARGIN);

    /* Compress once to get compressed data */
    size_t csize = lz_compress(src, src_size, comp, bound, level);

    /* Benchmark compress */
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        lz_compress(src, src_size, comp, bound, level);
    double t1 = now_sec();
    double comp_mbs = (double)src_size * iters / (t1 - t0) / 1e6;

    /* Benchmark decompress (C) */
    t0 = now_sec();
    for (int i = 0; i < iters; i++)
        lz_decompress(comp, csize, decomp, src_size + LZ_SAFETY_MARGIN);
    t1 = now_sec();
    double dec_c_mbs = (double)src_size * iters / (t1 - t0) / 1e6;

    /* Benchmark decompress (ASM) */
    t0 = now_sec();
    for (int i = 0; i < iters; i++)
        lz_decompress_asm(comp, csize, decomp, src_size + LZ_SAFETY_MARGIN);
    t1 = now_sec();
    double dec_asm_mbs = (double)src_size * iters / (t1 - t0) / 1e6;

    printf("  %-12s L%d  comp %7.1f MB/s  dec_C %7.1f MB/s  dec_ASM %7.1f MB/s  "
           "ratio %.1f%%\n",
           name, level, comp_mbs, dec_c_mbs, dec_asm_mbs,
           100.0 * csize / src_size);

    free(comp);
    free(decomp);
}

static void run_throughput(void) {
    printf("\n=== Throughput (256 KB) ===\n");

    size_t size = 256 * 1024;
    uint8_t *buf = (uint8_t *)malloc(size);
    int iters = 200;

    for (int d = 0; d < (int)N_DATASETS; d++) {
        datasets[d].gen(buf, size);
        bench_throughput(datasets[d].name, buf, size, LZ_LEVEL_FAST, iters);
        bench_throughput(datasets[d].name, buf, size, LZ_LEVEL_BEST, iters);
    }

    free(buf);
}

/* ------------------------------------------------------------------ */
/* Large data throughput                                                */
/* ------------------------------------------------------------------ */
static void run_throughput_large(void) {
    printf("\n=== Throughput (1 MB) ===\n");

    size_t size = 1024 * 1024;
    uint8_t *buf = (uint8_t *)malloc(size);
    int iters = 50;

    for (int d = 0; d < (int)N_DATASETS; d++) {
        datasets[d].gen(buf, size);
        bench_throughput(datasets[d].name, buf, size, LZ_LEVEL_FAST, iters);
    }

    free(buf);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {
    printf("LZ77 Compression Benchmark for A64FX Integer Pipeline\n");
    printf("======================================================\n\n");

    int fails = run_correctness();
    run_ratio();
    run_token_analysis();
    run_throughput();
    run_throughput_large();

    printf("\n%s\n", fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return fails ? 1 : 0;
}
