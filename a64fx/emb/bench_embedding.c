// bench_embedding.c
// Long-context embedding forward benchmark for A64FX
//
// Tests kernel variants across seqlen/dim/thread sweeps.
// Includes Zipfian distribution benchmarks to reveal L2 cache amplification.
//
// Build: make          (single-core)
//        make omp      (multi-core with OpenMP)
// Run:   OMP_NUM_THREADS=48 OMP_PROC_BIND=close ./bench_embedding_omp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "embedding.h"

//=============================================================================
// Constants
//=============================================================================
#define HBM2_PEAK_GBS  1024.0   // Total HBM2 peak bandwidth (GB/s)
#define VOCAB_SIZE     151936    // Qwen2.5 vocabulary

//=============================================================================
// Timing
//=============================================================================
static inline double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

//=============================================================================
// RNG (PCG32)
//=============================================================================
static uint64_t pcg_state = 0x853c49e6748fea9bULL;
static uint64_t pcg_inc   = 0xda3e39cb94b95bdbULL;

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

static int32_t rand_int(int32_t max)
{
    return (int32_t)(pcg32() % (uint32_t)max);
}

//=============================================================================
// Zipfian Distribution
//=============================================================================
static double* zipf_cdf = NULL;
static size_t  zipf_n = 0;

static void zipf_init(double alpha, size_t n)
{
    zipf_cdf = (double*)malloc((n + 1) * sizeof(double));
    zipf_n = n;
    double sum = 0.0;
    for (size_t k = 1; k <= n; k++) {
        sum += 1.0 / pow((double)k, alpha);
        zipf_cdf[k] = sum;
    }
    // Normalize
    for (size_t k = 1; k <= n; k++)
        zipf_cdf[k] /= sum;
    zipf_cdf[0] = 0.0;
}

static int32_t zipf_sample(void)
{
    double u = (double)(pcg32() & 0xFFFFFFFF) / 4294967296.0;
    // Binary search for rank
    size_t lo = 1, hi = zipf_n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (zipf_cdf[mid] < u)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (int32_t)(lo - 1);  // 0-indexed
}

static void zipf_free(void)
{
    free(zipf_cdf);
    zipf_cdf = NULL;
    zipf_n = 0;
}

// Fill indices with Zipfian distribution; return number of unique indices
static size_t fill_zipfian(int32_t* indices, size_t n, double alpha, size_t vocab)
{
    if (alpha <= 0.0) {
        // Uniform
        for (size_t i = 0; i < n; i++)
            indices[i] = rand_int((int32_t)vocab);
    } else {
        zipf_init(alpha, vocab);
        for (size_t i = 0; i < n; i++)
            indices[i] = zipf_sample();
        zipf_free();
    }

    // Count unique
    char* seen = (char*)calloc(vocab, 1);
    size_t unique = 0;
    if (seen) {
        for (size_t i = 0; i < n; i++) {
            if (!seen[indices[i]]) {
                seen[indices[i]] = 1;
                unique++;
            }
        }
        free(seen);
    }
    return unique;
}

//=============================================================================
// Comparison
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

//=============================================================================
// OMP wrappers
//=============================================================================
typedef void (*emb_kernel_fn)(const int32_t*, const float*, float*, size_t, size_t);

static void embedding_fwd_f32_omp(const int32_t* indices, const float* emb_table,
                                   float* output, size_t seq_len, size_t hidden_dim,
                                   emb_kernel_fn kernel)
{
#ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk = (seq_len + (size_t)nthreads - 1) / (size_t)nthreads;
        size_t start = (size_t)tid * chunk;
        size_t end = start + chunk;
        if (end > seq_len) end = seq_len;
        if (start < seq_len)
            kernel(indices + start, emb_table, output + start * hidden_dim,
                   end - start, hidden_dim);
    }
#else
    kernel(indices, emb_table, output, seq_len, hidden_dim);
#endif
}

static void embedding_fwd_f32_sorted_omp(const int32_t* indices, const float* emb_table,
                                           float* output, size_t seq_len, size_t hidden_dim,
                                           size_t vocab_size)
{
    int32_t* sorted_indices = (int32_t*)malloc(seq_len * sizeof(int32_t));
    int32_t* sorted_order   = (int32_t*)malloc(seq_len * sizeof(int32_t));
    if (!sorted_indices || !sorted_order) {
        free(sorted_indices);
        free(sorted_order);
        return;
    }

    counting_sort_indices(indices, seq_len, sorted_indices, sorted_order, vocab_size);

#ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk = (seq_len + (size_t)nthreads - 1) / (size_t)nthreads;
        size_t start = (size_t)tid * chunk;
        size_t end = start + chunk;
        if (end > seq_len) end = seq_len;
        if (start < seq_len)
            embedding_fwd_f32_sorted_core_asm(sorted_indices + start,
                                               sorted_order + start,
                                               emb_table, output,
                                               end - start, hidden_dim);
    }
#else
    embedding_fwd_f32_sorted_core_asm(sorted_indices, sorted_order,
                                       emb_table, output, seq_len, hidden_dim);
#endif

    free(sorted_indices);
    free(sorted_order);
}

// Dedup OMP: sort globally, then each thread does dedup scatter on its chunk
static void embedding_fwd_f32_dedup_omp(const int32_t* indices, const float* emb_table,
                                          float* output, size_t seq_len, size_t hidden_dim,
                                          size_t vocab_size)
{
    int32_t* si = (int32_t*)malloc(seq_len * sizeof(int32_t));
    int32_t* so = (int32_t*)malloc(seq_len * sizeof(int32_t));
    if (!si || !so) { free(si); free(so); return; }

    counting_sort_indices(indices, seq_len, si, so, vocab_size);

#ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk = (seq_len + (size_t)nthreads - 1) / (size_t)nthreads;
        size_t start = (size_t)tid * chunk;
        size_t end = start + chunk;
        if (end > seq_len) end = seq_len;

        // Process groups within [start, end) using dedup scatter
        size_t i = start;
        while (i < end) {
            int32_t idx = si[i];
            const float* src = emb_table + (size_t)idx * hidden_dim;
            size_t j = i + 1;
            while (j < end && si[j] == idx) j++;
            embedding_fwd_f32_scatter_row_asm(src, output, so + i, j - i, hidden_dim);
            i = j;
        }
    }
#else
    embedding_fwd_f32_dedup(indices, emb_table, output, seq_len, hidden_dim, vocab_size);
#endif

    free(si);
    free(so);
}

//=============================================================================
// Correctness tests
//=============================================================================
static int test_kernel(const char* name, emb_kernel_fn kernel,
                       size_t seq_len, size_t hidden_dim, size_t vocab_size)
{
    int32_t* indices   = (int32_t*)aligned_alloc(256, seq_len * sizeof(int32_t));
    float* emb_table   = (float*)aligned_alloc(256, vocab_size * hidden_dim * sizeof(float));
    float* output_ref  = (float*)aligned_alloc(256, seq_len * hidden_dim * sizeof(float));
    float* output_test = (float*)aligned_alloc(256, seq_len * hidden_dim * sizeof(float));

    if (!indices || !emb_table || !output_ref || !output_test) {
        printf("  %-20s seq=%6zu dim=%4zu  SKIP (alloc fail)\n", name, seq_len, hidden_dim);
        free(indices); free(emb_table); free(output_ref); free(output_test);
        return 0;
    }

    for (size_t i = 0; i < seq_len; i++)
        indices[i] = rand_int((int32_t)vocab_size);
    for (size_t i = 0; i < vocab_size * hidden_dim; i++)
        emb_table[i] = (float)(i % 1000) * 0.001f - 0.5f;

    embedding_fwd_f32_ref(indices, emb_table, output_ref, seq_len, hidden_dim);

    memset(output_test, 0, seq_len * hidden_dim * sizeof(float));
    kernel(indices, emb_table, output_test, seq_len, hidden_dim);

    float diff = max_abs_diff_f32(output_ref, output_test, seq_len * hidden_dim);
    int passed = (diff < 1e-6f);
    printf("  %-20s seq=%6zu dim=%4zu  diff=%e  %s\n",
           name, seq_len, hidden_dim, diff, passed ? "PASS" : "FAIL");

    free(indices); free(emb_table); free(output_ref); free(output_test);
    return passed ? 0 : 1;
}

static int test_sorted_kernel(const char* name,
                               size_t seq_len, size_t hidden_dim, size_t vocab_size,
                               int use_dedup)
{
    int32_t* indices   = (int32_t*)aligned_alloc(256, seq_len * sizeof(int32_t));
    float* emb_table   = (float*)aligned_alloc(256, vocab_size * hidden_dim * sizeof(float));
    float* output_ref  = (float*)aligned_alloc(256, seq_len * hidden_dim * sizeof(float));
    float* output_test = (float*)aligned_alloc(256, seq_len * hidden_dim * sizeof(float));

    if (!indices || !emb_table || !output_ref || !output_test) {
        printf("  %-20s seq=%6zu dim=%4zu  SKIP (alloc fail)\n", name, seq_len, hidden_dim);
        free(indices); free(emb_table); free(output_ref); free(output_test);
        return 0;
    }

    for (size_t i = 0; i < seq_len; i++)
        indices[i] = rand_int((int32_t)vocab_size);
    for (size_t i = 0; i < vocab_size * hidden_dim; i++)
        emb_table[i] = (float)(i % 1000) * 0.001f - 0.5f;

    embedding_fwd_f32_ref(indices, emb_table, output_ref, seq_len, hidden_dim);

    memset(output_test, 0, seq_len * hidden_dim * sizeof(float));
    if (use_dedup)
        embedding_fwd_f32_dedup(indices, emb_table, output_test, seq_len, hidden_dim, vocab_size);
    else
        embedding_fwd_f32_sorted(indices, emb_table, output_test, seq_len, hidden_dim, vocab_size);

    float diff = max_abs_diff_f32(output_ref, output_test, seq_len * hidden_dim);
    int passed = (diff < 1e-6f);
    printf("  %-20s seq=%6zu dim=%4zu  diff=%e  %s\n",
           name, seq_len, hidden_dim, diff, passed ? "PASS" : "FAIL");

    free(indices); free(emb_table); free(output_ref); free(output_test);
    return passed ? 0 : 1;
}

static int run_correctness_tests(void)
{
    printf("=== Correctness Tests ===\n");
    int failures = 0;

    size_t dims[]    = {1024, 2048, 4096};
    size_t seqlens[] = {1, 15, 16, 17, 64, 256, 1024};
    size_t vocab     = 32000;

    for (size_t di = 0; di < sizeof(dims)/sizeof(dims[0]); di++) {
        for (size_t si = 0; si < sizeof(seqlens)/sizeof(seqlens[0]); si++) {
            pcg_seed(42 + di * 100 + si);
            failures += test_kernel("baseline",    embedding_fwd_f32_asm,            seqlens[si], dims[di], vocab);
            failures += test_kernel("batched",     embedding_fwd_f32_batched_asm,    seqlens[si], dims[di], vocab);
            failures += test_kernel("stream",      embedding_fwd_f32_stream_asm,     seqlens[si], dims[di], vocab);
            failures += test_kernel("stream_ipf",  embedding_fwd_f32_stream_ipf_asm, seqlens[si], dims[di], vocab);
            failures += test_kernel("gather",      embedding_fwd_f32_gather_asm,     seqlens[si], dims[di], vocab);
            failures += test_sorted_kernel("sorted", seqlens[si], dims[di], vocab, 0);
            failures += test_sorted_kernel("dedup",  seqlens[si], dims[di], vocab, 1);
        }
    }

    // Non-VL-aligned dim
    pcg_seed(99);
    failures += test_kernel("stream_ipf(1000)", embedding_fwd_f32_stream_ipf_asm, 64, 1000, vocab);
    failures += test_sorted_kernel("dedup(dim=1000)", 64, 1000, vocab, 1);

    if (failures)
        printf("\n!!! %d correctness tests FAILED !!!\n\n", failures);
    else
        printf("\nAll correctness tests PASSED.\n\n");
    return failures;
}

//=============================================================================
// Benchmark infrastructure
//=============================================================================
typedef struct {
    const char* name;
    double time_ms;
    double gb_s;
    double pct_peak;
} bench_result_t;

static double compute_bytes_moved(size_t seq_len, size_t hidden_dim)
{
    return (double)seq_len * (double)hidden_dim * 4.0 * 2.0
           + (double)seq_len * 4.0;
}

static bench_result_t bench_single(const char* name, emb_kernel_fn kernel,
                                    const int32_t* indices, const float* emb_table,
                                    float* output, size_t seq_len, size_t hidden_dim,
                                    int use_omp, int iterations)
{
    bench_result_t r;
    r.name = name;

    for (int i = 0; i < 3; i++) {
        if (use_omp)
            embedding_fwd_f32_omp(indices, emb_table, output, seq_len, hidden_dim, kernel);
        else
            kernel(indices, emb_table, output, seq_len, hidden_dim);
    }

    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        if (use_omp)
            embedding_fwd_f32_omp(indices, emb_table, output, seq_len, hidden_dim, kernel);
        else
            kernel(indices, emb_table, output, seq_len, hidden_dim);
    }
    double t1 = get_time_sec();

    double avg_sec = (t1 - t0) / iterations;
    double bytes = compute_bytes_moved(seq_len, hidden_dim);
    r.time_ms  = avg_sec * 1e3;
    r.gb_s     = bytes / avg_sec / 1e9;
    r.pct_peak = r.gb_s / HBM2_PEAK_GBS * 100.0;
    return r;
}

// Bench function for sorted/dedup variants (takes vocab_size, mode)
typedef enum { MODE_SORTED, MODE_DEDUP } sort_mode_t;

static bench_result_t bench_sort_variant(const char* name,
                                          const int32_t* indices, const float* emb_table,
                                          float* output, size_t seq_len, size_t hidden_dim,
                                          size_t vocab_size, sort_mode_t mode,
                                          int use_omp, int iterations)
{
    bench_result_t r;
    r.name = name;

    for (int i = 0; i < 3; i++) {
        if (mode == MODE_DEDUP) {
            if (use_omp)
                embedding_fwd_f32_dedup_omp(indices, emb_table, output, seq_len, hidden_dim, vocab_size);
            else
                embedding_fwd_f32_dedup(indices, emb_table, output, seq_len, hidden_dim, vocab_size);
        } else {
            if (use_omp)
                embedding_fwd_f32_sorted_omp(indices, emb_table, output, seq_len, hidden_dim, vocab_size);
            else
                embedding_fwd_f32_sorted(indices, emb_table, output, seq_len, hidden_dim, vocab_size);
        }
    }

    double t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) {
        if (mode == MODE_DEDUP) {
            if (use_omp)
                embedding_fwd_f32_dedup_omp(indices, emb_table, output, seq_len, hidden_dim, vocab_size);
            else
                embedding_fwd_f32_dedup(indices, emb_table, output, seq_len, hidden_dim, vocab_size);
        } else {
            if (use_omp)
                embedding_fwd_f32_sorted_omp(indices, emb_table, output, seq_len, hidden_dim, vocab_size);
            else
                embedding_fwd_f32_sorted(indices, emb_table, output, seq_len, hidden_dim, vocab_size);
        }
    }
    double t1 = get_time_sec();

    double avg_sec = (t1 - t0) / iterations;
    double bytes = compute_bytes_moved(seq_len, hidden_dim);
    r.time_ms  = avg_sec * 1e3;
    r.gb_s     = bytes / avg_sec / 1e9;
    r.pct_peak = r.gb_s / HBM2_PEAK_GBS * 100.0;
    return r;
}

static void print_result_row(bench_result_t r, size_t seq_len, size_t hidden_dim, int nthreads)
{
    char seq_buf[16];
    if (seq_len >= 1024)
        snprintf(seq_buf, sizeof(seq_buf), "%zuK", seq_len / 1024);
    else
        snprintf(seq_buf, sizeof(seq_buf), "%zu", seq_len);
    printf("  %-14s | %6s | %4zu | %3d thr | %8.2f ms | %7.1f GB/s | %5.1f%%\n",
           r.name, seq_buf, hidden_dim, nthreads, r.time_ms, r.gb_s, r.pct_peak);
}

//=============================================================================
// Main benchmark sweep (uniform random indices)
//=============================================================================
static void run_benchmarks(void)
{
    printf("====================================================================\n");
    printf("Embedding Forward Benchmark — A64FX (HBM2 peak: %.0f GB/s)\n", HBM2_PEAK_GBS);
    printf("Vocab: %d, FP32, uniform random indices\n", VOCAB_SIZE);
    printf("====================================================================\n\n");

    size_t seqlens[] = {1024, 4096, 16384, 65536, 262144};
    size_t dims[]    = {1024, 2048, 4096};
    int n_seqlens = sizeof(seqlens) / sizeof(seqlens[0]);
    int n_dims    = sizeof(dims) / sizeof(dims[0]);

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    int use_omp = (nthreads > 1);

    printf("Threads: %d, OMP: %s\n\n", nthreads, use_omp ? "yes" : "no");

    size_t max_dim = 4096;
    size_t emb_bytes = (size_t)VOCAB_SIZE * max_dim * sizeof(float);
    float* emb_table = (float*)aligned_alloc(256, emb_bytes);
    if (!emb_table) {
        printf("ERROR: cannot allocate embedding table (%.2f GB)\n", emb_bytes / 1e9);
        return;
    }
    memset(emb_table, 0, emb_bytes);
    pcg_seed(123);
    for (size_t i = 0; i < (size_t)VOCAB_SIZE * max_dim; i += 64)
        emb_table[i] = (float)(pcg32() & 0xFFFF) * 1e-4f;

    printf("  %-14s | %6s | %4s | %7s | %11s | %11s | %6s\n",
           "Kernel", "SeqLen", "Dim", "Threads", "Time", "BW", "Peak%");
    printf("  %-14s-+-%6s-+-%4s-+-%7s-+-%11s-+-%11s-+-%6s\n",
           "--------------", "------", "----", "-------", "-----------", "-----------", "------");

    for (int di = 0; di < n_dims; di++) {
        size_t dim = dims[di];

        for (int si = 0; si < n_seqlens; si++) {
            size_t seq = seqlens[si];

            size_t out_bytes = seq * dim * sizeof(float);
            if (out_bytes > 8UL * 1024 * 1024 * 1024) {
                printf("  (skipping seq=%zu dim=%zu — output %.1f GB)\n",
                       seq, dim, out_bytes / 1e9);
                continue;
            }

            int32_t* indices = (int32_t*)aligned_alloc(256, seq * sizeof(int32_t));
            float* output    = (float*)aligned_alloc(256, out_bytes);
            if (!indices || !output) {
                printf("  (skipping seq=%zu dim=%zu — alloc fail)\n", seq, dim);
                free(indices); free(output);
                continue;
            }

            pcg_seed(42 + seq + dim);
            for (size_t i = 0; i < seq; i++)
                indices[i] = rand_int(VOCAB_SIZE);
            memset(output, 0, out_bytes);

            int iters = 1;
            if (seq <= 4096)       iters = 20;
            else if (seq <= 16384) iters = 5;
            else if (seq <= 65536) iters = 3;

            bench_result_t r;

            r = bench_single("stream", embedding_fwd_f32_stream_asm,
                              indices, emb_table, output, seq, dim, use_omp, iters);
            print_result_row(r, seq, dim, nthreads);

            r = bench_single("stream_ipf", embedding_fwd_f32_stream_ipf_asm,
                              indices, emb_table, output, seq, dim, use_omp, iters);
            print_result_row(r, seq, dim, nthreads);

            r = bench_sort_variant("sorted", indices, emb_table, output,
                                    seq, dim, VOCAB_SIZE, MODE_SORTED, use_omp, iters);
            print_result_row(r, seq, dim, nthreads);

            r = bench_sort_variant("dedup", indices, emb_table, output,
                                    seq, dim, VOCAB_SIZE, MODE_DEDUP, use_omp, iters);
            print_result_row(r, seq, dim, nthreads);

            printf("  ---\n");

            free(indices);
            free(output);
        }
    }

    free(emb_table);
}

//=============================================================================
// Zipfian Distribution Benchmark
// Shows L2 cache amplification effect for realistic token distributions
//=============================================================================
static void run_zipfian_benchmark(void)
{
    printf("\n====================================================================\n");
    printf("Zipfian Distribution Benchmark (seq=64K, vocab=%d)\n", VOCAB_SIZE);
    printf("Reveals L2 cache amplification for realistic distributions\n");
    printf("====================================================================\n\n");

    size_t seq = 65536;
    size_t dims[] = {1024, 4096};
    double alphas[] = {0.0, 0.8, 1.0, 1.2};
    int n_dims = sizeof(dims) / sizeof(dims[0]);
    int n_alphas = sizeof(alphas) / sizeof(alphas[0]);

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    int use_omp = (nthreads > 1);
    int iters = 3;

    size_t max_dim = 4096;
    size_t emb_bytes = (size_t)VOCAB_SIZE * max_dim * sizeof(float);
    float* emb_table = (float*)aligned_alloc(256, emb_bytes);
    if (!emb_table) { printf("ERROR: alloc fail\n"); return; }
    memset(emb_table, 0, emb_bytes);
    pcg_seed(200);
    for (size_t i = 0; i < (size_t)VOCAB_SIZE * max_dim; i += 64)
        emb_table[i] = (float)(pcg32() & 0xFFFF) * 1e-4f;

    printf("  %-14s | %4s | %5s | %6s | %3s | %8s | %11s | %6s\n",
           "Kernel", "Dim", "alpha", "Unique", "Thr", "Time", "BW", "Peak%");
    printf("  %-14s-+-%4s-+-%5s-+-%6s-+-%3s-+-%8s-+-%11s-+-%6s\n",
           "--------------", "----", "-----", "------", "---", "--------", "-----------", "------");

    for (int di = 0; di < n_dims; di++) {
        size_t dim = dims[di];
        size_t out_bytes = seq * dim * sizeof(float);

        int32_t* indices = (int32_t*)aligned_alloc(256, seq * sizeof(int32_t));
        float* output    = (float*)aligned_alloc(256, out_bytes);
        if (!indices || !output) {
            printf("  (alloc fail for dim=%zu)\n", dim);
            free(indices); free(output);
            continue;
        }

        for (int ai = 0; ai < n_alphas; ai++) {
            double alpha = alphas[ai];
            pcg_seed(300 + ai * 17 + di * 7);
            size_t unique = fill_zipfian(indices, seq, alpha, VOCAB_SIZE);
            memset(output, 0, out_bytes);

            bench_result_t r;

            r = bench_single("stream", embedding_fwd_f32_stream_asm,
                              indices, emb_table, output, seq, dim, use_omp, iters);
            printf("  %-14s | %4zu | %5.1f | %5zuK | %3d | %5.2f ms | %7.1f GB/s | %5.1f%%\n",
                   r.name, dim, alpha, unique/1024, nthreads, r.time_ms, r.gb_s, r.pct_peak);

            r = bench_single("stream_ipf", embedding_fwd_f32_stream_ipf_asm,
                              indices, emb_table, output, seq, dim, use_omp, iters);
            printf("  %-14s | %4zu | %5.1f | %5zuK | %3d | %5.2f ms | %7.1f GB/s | %5.1f%%\n",
                   r.name, dim, alpha, unique/1024, nthreads, r.time_ms, r.gb_s, r.pct_peak);

            r = bench_sort_variant("sorted", indices, emb_table, output,
                                    seq, dim, VOCAB_SIZE, MODE_SORTED, use_omp, iters);
            printf("  %-14s | %4zu | %5.1f | %5zuK | %3d | %5.2f ms | %7.1f GB/s | %5.1f%%\n",
                   r.name, dim, alpha, unique/1024, nthreads, r.time_ms, r.gb_s, r.pct_peak);

            r = bench_sort_variant("dedup", indices, emb_table, output,
                                    seq, dim, VOCAB_SIZE, MODE_DEDUP, use_omp, iters);
            printf("  %-14s | %4zu | %5.1f | %5zuK | %3d | %5.2f ms | %7.1f GB/s | %5.1f%%\n",
                   r.name, dim, alpha, unique/1024, nthreads, r.time_ms, r.gb_s, r.pct_peak);

            printf("  ---\n");
        }

        free(indices);
        free(output);
    }

    free(emb_table);
}

//=============================================================================
// Multi-core scaling test
//=============================================================================
static void run_scaling_test(void)
{
#ifdef _OPENMP
    printf("\n====================================================================\n");
    printf("Multi-Core Scaling Test (seq=64K, dim=4096, vocab=%d)\n", VOCAB_SIZE);
    printf("====================================================================\n\n");

    size_t seq = 65536;
    size_t dim = 4096;
    int iters = 3;

    float* emb_table = (float*)aligned_alloc(256, (size_t)VOCAB_SIZE * dim * sizeof(float));
    int32_t* indices = (int32_t*)aligned_alloc(256, seq * sizeof(int32_t));
    float* output    = (float*)aligned_alloc(256, seq * dim * sizeof(float));
    if (!emb_table || !indices || !output) {
        printf("  Allocation failed\n");
        free(emb_table); free(indices); free(output);
        return;
    }

    memset(emb_table, 0, (size_t)VOCAB_SIZE * dim * sizeof(float));
    pcg_seed(77);
    for (size_t i = 0; i < seq; i++)
        indices[i] = rand_int(VOCAB_SIZE);
    memset(output, 0, seq * dim * sizeof(float));

    int thread_counts[] = {1, 12, 24, 48};
    int n_tc = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("  %-14s | %3s | %11s | %11s | %6s\n",
           "Kernel", "Thr", "Time", "BW", "Peak%");
    printf("  %-14s-+-%3s-+-%11s-+-%11s-+-%6s\n",
           "--------------", "---", "-----------", "-----------", "------");

    for (int ti = 0; ti < n_tc; ti++) {
        int nt = thread_counts[ti];
        omp_set_num_threads(nt);

        bench_result_t r;
        int use = (nt > 1);

        r = bench_single("stream", embedding_fwd_f32_stream_asm,
                          indices, emb_table, output, seq, dim, use, iters);
        printf("  %-14s | %3d | %8.2f ms | %7.1f GB/s | %5.1f%%\n",
               r.name, nt, r.time_ms, r.gb_s, r.pct_peak);

        r = bench_single("stream_ipf", embedding_fwd_f32_stream_ipf_asm,
                          indices, emb_table, output, seq, dim, use, iters);
        printf("  %-14s | %3d | %8.2f ms | %7.1f GB/s | %5.1f%%\n",
               r.name, nt, r.time_ms, r.gb_s, r.pct_peak);

        r = bench_sort_variant("sorted", indices, emb_table, output,
                                seq, dim, VOCAB_SIZE, MODE_SORTED, use, iters);
        printf("  %-14s | %3d | %8.2f ms | %7.1f GB/s | %5.1f%%\n",
               r.name, nt, r.time_ms, r.gb_s, r.pct_peak);

        r = bench_sort_variant("dedup", indices, emb_table, output,
                                seq, dim, VOCAB_SIZE, MODE_DEDUP, use, iters);
        printf("  %-14s | %3d | %8.2f ms | %7.1f GB/s | %5.1f%%\n",
               r.name, nt, r.time_ms, r.gb_s, r.pct_peak);

        printf("  ---\n");
    }

    free(emb_table);
    free(indices);
    free(output);
#else
    printf("\n(Scaling test requires OpenMP — skipped)\n");
#endif
}

//=============================================================================
// Main
//=============================================================================
int main(int argc, char** argv)
{
    int skip_correctness = 0;
    int bench_only_zipf = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench-only") == 0)
            skip_correctness = 1;
        if (strcmp(argv[i], "--zipf-only") == 0) {
            skip_correctness = 1;
            bench_only_zipf = 1;
        }
    }

    printf("======================================================\n");
    printf("SVE Embedding Forward Benchmark — A64FX\n");
    printf("======================================================\n");
    printf("Vocab=%d, FP32, HBM2 peak=%.0f GB/s\n", VOCAB_SIZE, HBM2_PEAK_GBS);
#ifdef _OPENMP
    printf("OpenMP: max_threads=%d\n", omp_get_max_threads());
#else
    printf("OpenMP: disabled (single-core)\n");
#endif
    printf("\n");

    int failures = 0;
    if (!skip_correctness)
        failures = run_correctness_tests();

    if (failures) {
        printf("Aborting benchmarks due to correctness failures.\n");
        return 1;
    }

    if (bench_only_zipf) {
        run_zipfian_benchmark();
    } else {
        run_benchmarks();
        run_zipfian_benchmark();
        run_scaling_test();
    }

    printf("\nDone.\n");
    return 0;
}
