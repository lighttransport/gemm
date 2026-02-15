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
// FP16 helpers
//=============================================================================
// Use uint16_t as storage type; fcc __fp16 for conversion
#ifdef __ARM_FP16_FORMAT_IEEE
typedef __fp16 fp16_t;
#else
typedef uint16_t fp16_t;
#endif

static float fp16_to_f32(fp16_t v) {
#ifdef __ARM_FP16_FORMAT_IEEE
    return (float)v;
#else
    // Minimal software decode (for correctness test only)
    uint16_t bits = v;
    uint32_t sign = (bits >> 15) & 1;
    uint32_t exp  = (bits >> 10) & 0x1f;
    uint32_t frac = bits & 0x3ff;
    uint32_t f32;
    if (exp == 0) {
        if (frac == 0) f32 = sign << 31;
        else { exp = 1; while (!(frac & 0x400)) { frac <<= 1; exp--; }
               frac &= 0x3ff; f32 = (sign<<31) | ((exp+127-15)<<23) | (frac<<13); }
    } else if (exp == 31) { f32 = (sign<<31) | 0x7f800000 | (frac<<13); }
    else { f32 = (sign<<31) | ((exp+127-15)<<23) | (frac<<13); }
    float result; memcpy(&result, &f32, 4); return result;
#endif
}

static fp16_t f32_to_fp16(float v) {
#ifdef __ARM_FP16_FORMAT_IEEE
    return (fp16_t)v;
#else
    uint32_t f32; memcpy(&f32, &v, 4);
    uint32_t sign = (f32 >> 31) & 1;
    int32_t exp = ((f32 >> 23) & 0xff) - 127 + 15;
    uint32_t frac = (f32 >> 13) & 0x3ff;
    if (exp <= 0) return (fp16_t)(sign << 15);
    if (exp >= 31) return (fp16_t)((sign << 15) | 0x7c00);
    return (fp16_t)((sign << 15) | (exp << 10) | frac);
#endif
}

//=============================================================================
// NUMA-Replicated FP16 CMG Scaling Test
//
// Architecture:
//   A64FX has 4 CMGs (Core Memory Groups), each with 12 cores + local HBM2.
//   Per-CMG HBM2 bandwidth: ~256 GB/s read, ~128 GB/s write.
//   Inter-CMG ring bandwidth: ~100 GB/s total (shared).
//
// Problem: With a single embedding table, remote CMGs bottleneck on the ring.
//   4 CMGs reading one table → ~78 GB/s total (ring-limited).
//
// Solution: Replicate FP16 table to each CMG's local HBM2.
//   - FP16 halves memory: 151936 × 4096 × 2B = 1.17 GB per replica
//   - 4 replicas = 4.69 GB (fits easily in 32 GB HBM2)
//   - Each CMG reads LOCAL table, writes LOCAL output → zero inter-CMG traffic
//   - Sequence split into N equal chunks, one per CMG
//
// Expected: linear scaling — 2 CMG = 2×, 3 CMG = 3×, 4 CMG = 4× of 1 CMG.
//=============================================================================
#ifdef _OPENMP
#define MAX_CMGS 4
#define CORES_PER_CMG 12

static void run_cmg_scaling_test(void)
{
    printf("\n====================================================================\n");
    printf("NUMA-Replicated FP16 CMG Scaling (vocab=%d)\n", VOCAB_SIZE);
    printf("Each CMG gets local FP16 table + local output + sequence/N partition\n");
    printf("====================================================================\n\n");

    printf("  How it works:\n");
    printf("  1. Allocate N_CMG copies of FP16 embedding table\n");
    printf("  2. Allocate N_CMG separate output buffers\n");
    printf("  3. First-touch init each table+output from that CMG's 12 threads\n");
    printf("     (demand paging → pages placed on touching CMG's local HBM2)\n");
    printf("  4. Split sequence into N_CMG equal chunks\n");
    printf("  5. Each CMG's 12 threads process their chunk:\n");
    printf("     LOCAL table read + LOCAL output write → zero inter-CMG traffic\n");
    printf("  6. FP16 halves memory: 151936 × 4096 × 2B = 1.17 GB per replica\n\n");

    size_t dim = 4096;
    size_t seq = 262144;  // 256K tokens
    size_t table_elems = (size_t)VOCAB_SIZE * dim;
    size_t table_bytes = table_elems * sizeof(fp16_t);
    int iters = 5;

    printf("  Config: seq=%zuK, dim=%zu, vocab=%d\n", seq/1024, dim, VOCAB_SIZE);
    printf("  FP16 table: %.2f GB per replica\n", table_bytes / 1e9);
    printf("  FP16 output: %.2f GB total\n", seq * dim * sizeof(fp16_t) / 1e9);
    printf("\n");

    // Allocate index array (shared, small — 1MB, fits in L2 after warmup)
    int32_t* indices = (int32_t*)aligned_alloc(256, seq * sizeof(int32_t));
    if (!indices) { printf("  ERROR: index alloc\n"); return; }
    pcg_seed(555);
    for (size_t i = 0; i < seq; i++)
        indices[i] = rand_int(VOCAB_SIZE);

    // Master FP16 table (initialized sequentially, then copied per-CMG)
    fp16_t* master_table = (fp16_t*)aligned_alloc(256, table_bytes);
    if (!master_table) { printf("  ERROR: master table alloc\n"); free(indices); return; }
    pcg_seed(777);
    for (size_t i = 0; i < table_elems; i++)
        master_table[i] = f32_to_fp16((float)(pcg32() & 0xFFFF) * 1e-4f - 3.0f);

    // Per-CMG table replicas, output buffers, and index replicas
    fp16_t* cmg_tables[MAX_CMGS] = {NULL};
    fp16_t* cmg_outputs[MAX_CMGS] = {NULL};
    int32_t* cmg_indices[MAX_CMGS] = {NULL};

    printf("  %-18s | %3s | %11s | %11s | %6s | %7s\n",
           "Config", "Thr", "Time", "BW", "Peak%", "Scaling");
    printf("  %-18s-+-%3s-+-%11s-+-%11s-+-%6s-+-%7s\n",
           "------------------", "---", "-----------", "-----------", "------", "-------");

    double bw_1cmg = 0.0;

    int cmg_counts[] = {1, 2, 3, 4};
    int n_configs = sizeof(cmg_counts) / sizeof(cmg_counts[0]);

    for (int ci = 0; ci < n_configs; ci++) {
        int n_cmg = cmg_counts[ci];
        int total_threads = n_cmg * CORES_PER_CMG;
        size_t base_cmg_seq = seq / (size_t)n_cmg;

        // Allocate per-CMG: table replicas + output buffers + index replicas
        int alloc_ok = 1;
        for (int c = 0; c < n_cmg; c++) {
            cmg_tables[c] = (fp16_t*)aligned_alloc(256, table_bytes);
            size_t this_seq = (c == n_cmg - 1) ? (seq - (size_t)c * base_cmg_seq) : base_cmg_seq;
            cmg_outputs[c] = (fp16_t*)aligned_alloc(256, this_seq * dim * sizeof(fp16_t));
            cmg_indices[c] = (int32_t*)aligned_alloc(256, this_seq * sizeof(int32_t));
            if (!cmg_tables[c] || !cmg_outputs[c] || !cmg_indices[c]) {
                printf("  ERROR: CMG%d alloc failed\n", c);
                alloc_ok = 0;
                break;
            }
        }
        if (!alloc_ok) goto cleanup;

        // First-touch each table/output/indices from its CMG's threads
        // OMP_PROC_BIND=close ensures threads 0-11 → CMG0, 12-23 → CMG1, etc.
        omp_set_num_threads(total_threads);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int cmg = tid / CORES_PER_CMG;
            int local_tid = tid % CORES_PER_CMG;

            if (cmg < n_cmg) {
                // First-touch table replica (12 threads per table)
                size_t tbl_chunk = (table_elems + CORES_PER_CMG - 1) / CORES_PER_CMG;
                size_t tbl_start = (size_t)local_tid * tbl_chunk;
                size_t tbl_end = tbl_start + tbl_chunk;
                if (tbl_end > table_elems) tbl_end = table_elems;
                if (tbl_start < tbl_end)
                    memcpy(cmg_tables[cmg] + tbl_start, master_table + tbl_start,
                           (tbl_end - tbl_start) * sizeof(fp16_t));

                // This CMG's sequence parameters
                size_t cmg_start_idx = (size_t)cmg * base_cmg_seq;
                size_t this_seq = (cmg == n_cmg - 1)
                    ? (seq - cmg_start_idx) : base_cmg_seq;

                // First-touch index replica (copy this CMG's index chunk locally)
                size_t idx_chunk = (this_seq + CORES_PER_CMG - 1) / CORES_PER_CMG;
                size_t idx_start = (size_t)local_tid * idx_chunk;
                size_t idx_end = idx_start + idx_chunk;
                if (idx_end > this_seq) idx_end = this_seq;
                if (idx_start < idx_end)
                    memcpy(cmg_indices[cmg] + idx_start,
                           indices + cmg_start_idx + idx_start,
                           (idx_end - idx_start) * sizeof(int32_t));

                // First-touch output buffer
                size_t out_elems = this_seq * dim;
                size_t out_chunk = (out_elems + CORES_PER_CMG - 1) / CORES_PER_CMG;
                size_t out_start = (size_t)local_tid * out_chunk;
                size_t out_end = out_start + out_chunk;
                if (out_end > out_elems) out_end = out_elems;
                if (out_start < out_end)
                    memset(cmg_outputs[cmg] + out_start, 0,
                           (out_end - out_start) * sizeof(fp16_t));
            }
        }

        // Warmup (3 iterations)
        for (int w = 0; w < 3; w++) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int cmg = tid / CORES_PER_CMG;
                int local_tid = tid % CORES_PER_CMG;

                if (cmg < n_cmg) {
                    size_t cmg_start_idx = (size_t)cmg * base_cmg_seq;
                    size_t this_seq = (cmg == n_cmg - 1)
                        ? (seq - cmg_start_idx) : base_cmg_seq;
                    size_t thr_chunk = (this_seq + CORES_PER_CMG - 1) / CORES_PER_CMG;
                    size_t thr_start = (size_t)local_tid * thr_chunk;
                    size_t thr_end = thr_start + thr_chunk;
                    if (thr_end > this_seq) thr_end = this_seq;

                    if (thr_start < thr_end) {
                        embedding_fwd_f16_stream_asm(
                            cmg_indices[cmg] + thr_start,  // LOCAL indices
                            cmg_tables[cmg],
                            cmg_outputs[cmg] + thr_start * dim,
                            thr_end - thr_start,
                            dim);
                    }
                }
            }
        }

        // Timed runs — take best of N to filter OS noise
        double best_sec = 1e30;
        for (int it = 0; it < iters; it++) {
            double t0 = get_time_sec();
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int cmg = tid / CORES_PER_CMG;
                int local_tid = tid % CORES_PER_CMG;

                if (cmg < n_cmg) {
                    size_t cmg_start_idx = (size_t)cmg * base_cmg_seq;
                    size_t this_seq = (cmg == n_cmg - 1)
                        ? (seq - cmg_start_idx) : base_cmg_seq;
                    size_t thr_chunk = (this_seq + CORES_PER_CMG - 1) / CORES_PER_CMG;
                    size_t thr_start = (size_t)local_tid * thr_chunk;
                    size_t thr_end = thr_start + thr_chunk;
                    if (thr_end > this_seq) thr_end = this_seq;

                    if (thr_start < thr_end) {
                        // ALL data LOCAL: indices, table, output
                        embedding_fwd_f16_stream_asm(
                            cmg_indices[cmg] + thr_start,  // LOCAL indices
                            cmg_tables[cmg],               // LOCAL table
                            cmg_outputs[cmg] + thr_start * dim,  // LOCAL output
                            thr_end - thr_start,
                            dim);
                    }
                }
            }
            double t1 = get_time_sec();
            double elapsed = t1 - t0;
            if (elapsed < best_sec) best_sec = elapsed;
        }

        // Bytes moved: read table rows + write output (FP16 = 2 bytes/elem)
        // + read indices (4 bytes/token)
        double bytes = (double)seq * (double)dim * 2.0 * 2.0 + (double)seq * 4.0;
        double bw = bytes / best_sec / 1e9;
        double pct = bw / HBM2_PEAK_GBS * 100.0;

        if (ci == 0) bw_1cmg = bw;
        double scaling = (bw_1cmg > 0.0) ? bw / bw_1cmg : 0.0;

        char label[64];
        snprintf(label, sizeof(label), "FP16 %dCMG replicated", n_cmg);
        printf("  %-18s | %3d | %8.2f ms | %7.1f GB/s | %5.1f%% | %5.2fx\n",
               label, total_threads, best_sec * 1e3, bw, pct, scaling);

        // Free per-CMG allocations
        for (int c = 0; c < n_cmg; c++) {
            free(cmg_tables[c]);  cmg_tables[c] = NULL;
            free(cmg_outputs[c]); cmg_outputs[c] = NULL;
            free(cmg_indices[c]); cmg_indices[c] = NULL;
        }
    }

    // Also run single-CMG FP32 for comparison
    {
        float* f32_table = (float*)aligned_alloc(256, (size_t)VOCAB_SIZE * dim * sizeof(float));
        float* f32_output = (float*)aligned_alloc(256, seq * dim * sizeof(float));
        if (f32_table && f32_output) {
            for (size_t i = 0; i < table_elems; i++)
                f32_table[i] = fp16_to_f32(master_table[i]);
            memset(f32_output, 0, seq * dim * sizeof(float));

            omp_set_num_threads(CORES_PER_CMG);
            embedding_fwd_f32_omp(indices, f32_table, f32_output, seq, dim,
                                   embedding_fwd_f32_stream_asm);
            double t0 = get_time_sec();
            for (int it = 0; it < iters; it++)
                embedding_fwd_f32_omp(indices, f32_table, f32_output, seq, dim,
                                       embedding_fwd_f32_stream_asm);
            double t1 = get_time_sec();
            double avg = (t1 - t0) / iters;
            double bw_bytes = (double)seq * (double)dim * 4.0 * 2.0 + (double)seq * 4.0;
            double bw = bw_bytes / avg / 1e9;
            printf("  ---\n");
            printf("  %-18s | %3d | %8.2f ms | %7.1f GB/s | %5.1f%% | (ref)\n",
                   "FP32 1CMG baseline", CORES_PER_CMG, avg * 1e3, bw,
                   bw / HBM2_PEAK_GBS * 100.0);
        }
        free(f32_table);
        free(f32_output);
    }

    printf("\n");
    printf("  Note: Linear scaling expected when each CMG accesses only LOCAL HBM2.\n");
    printf("  Inter-CMG ring (~100 GB/s) is the bottleneck with shared tables.\n");

    goto done;

cleanup:
    for (int c = 0; c < MAX_CMGS; c++) {
        free(cmg_tables[c]);  cmg_tables[c] = NULL;
        free(cmg_outputs[c]); cmg_outputs[c] = NULL;
        free(cmg_indices[c]); cmg_indices[c] = NULL;
    }

done:
    free(indices);
    free(master_table);
}
#endif // _OPENMP

//=============================================================================
// Main
//=============================================================================
int main(int argc, char** argv)
{
    int skip_correctness = 0;
    int bench_only_zipf = 0;
    int cmg_only = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench-only") == 0)
            skip_correctness = 1;
        if (strcmp(argv[i], "--zipf-only") == 0) {
            skip_correctness = 1;
            bench_only_zipf = 1;
        }
        if (strcmp(argv[i], "--cmg-only") == 0) {
            skip_correctness = 1;
            cmg_only = 1;
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

    if (cmg_only) {
#ifdef _OPENMP
        run_cmg_scaling_test();
#else
        printf("CMG scaling requires OpenMP.\n");
#endif
    } else if (bench_only_zipf) {
        run_zipfian_benchmark();
    } else {
        run_benchmarks();
        run_zipfian_benchmark();
        run_scaling_test();
#ifdef _OPENMP
        run_cmg_scaling_test();
#endif
    }

    printf("\nDone.\n");
    return 0;
}
