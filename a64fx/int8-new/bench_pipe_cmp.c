// bench_pipe_cmp.c
// Benchmark comparing baseline vs pipelined kernel

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t rdtsc(void) {
    uint64_t t;
    asm volatile("mrs %0, CNTVCT_EL0" : "=r"(t));
    return t;
}

// External kernel declarations
extern void kernel_6x4_kloop(const int8_t* A, const int8_t* B, int32_t* C,
                             int K, int ldc);
extern void kernel_6x4_pipeV2(const int8_t* A, const int8_t* B, int32_t* C,
                              int K, int ldc);
extern void kernel_6x4_deep(const int8_t* A, const int8_t* B, int32_t* C,
                            int K, int ldc);
extern void kernel_6x4_prfm(const int8_t* A, const int8_t* B, int32_t* C,
                            int K, int ldc);

#define MR 6
#define NR 64

static inline size_t align256(size_t size) {
    return (size + 255) & ~255UL;
}

// Pack A: [K/4][6][4]
static void pack_A(const int8_t* A, int8_t* Ap, int M, int K, int lda) {
    int m_tiles = (M + MR - 1) / MR;
    for (int mt = 0; mt < m_tiles; mt++) {
        int m0 = mt * MR;
        int mr = (m0 + MR <= M) ? MR : (M - m0);
        int8_t* dst = Ap + mt * (K / 4) * 24;
        for (int k = 0; k < K; k += 4) {
            for (int m = 0; m < MR; m++) {
                for (int kk = 0; kk < 4; kk++) {
                    *dst++ = (m < mr && k + kk < K) ? A[(m0 + m) * lda + k + kk] : 0;
                }
            }
        }
    }
}

// Pack B: [N/64][K/4][64][4]
static void pack_B(const int8_t* B, int8_t* Bp, int N, int K, int ldb) {
    int n_tiles = (N + NR - 1) / NR;
    for (int nt = 0; nt < n_tiles; nt++) {
        int n0 = nt * NR;
        int nr = (n0 + NR <= N) ? NR : (N - n0);
        int8_t* dst = Bp + nt * (K / 4) * 256;
        for (int k = 0; k < K; k += 4) {
            for (int n = 0; n < NR; n++) {
                for (int kk = 0; kk < 4; kk++) {
                    *dst++ = (n < nr && k + kk < K) ? B[(n0 + n) * ldb + k + kk] : 0;
                }
            }
        }
    }
}

// Reference implementation
static void gemm_ref(const int8_t* A, const int8_t* B, int32_t* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * lda + k] * (int32_t)B[n * ldb + k];
            }
            C[m * ldc + n] = sum;
        }
    }
}

// GEMM using baseline kernel
static void gemm_baseline(const int8_t* Ap, const int8_t* Bp,
                          int32_t* C, int ldc, int M, int N, int K) {
    int m_tiles = (M + MR - 1) / MR;
    int n_tiles = N / NR;
    int k4 = K / 4;

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int nt = 0; nt < n_tiles; nt++) {
            kernel_6x4_kloop(Ap + mt * k4 * 24,
                            Bp + nt * k4 * 256,
                            C + mt * MR * ldc + nt * NR,
                            K, ldc * 4);
        }
    }
}

// GEMM using pipelined kernel
static void gemm_pipelined(const int8_t* Ap, const int8_t* Bp,
                           int32_t* C, int ldc, int M, int N, int K) {
    int m_tiles = (M + MR - 1) / MR;
    int n_tiles = N / NR;
    int k4 = K / 4;

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int nt = 0; nt < n_tiles; nt++) {
            kernel_6x4_pipeV2(Ap + mt * k4 * 24,
                              Bp + nt * k4 * 256,
                              C + mt * MR * ldc + nt * NR,
                              K, ldc * 4);
        }
    }
}

// GEMM using deep pipelined kernel (4× unroll) - DISABLED due to bug
static void gemm_deep(const int8_t* Ap, const int8_t* Bp,
                      int32_t* C, int ldc, int M, int N, int K) {
    int m_tiles = (M + MR - 1) / MR;
    int n_tiles = N / NR;
    int k4 = K / 4;

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int nt = 0; nt < n_tiles; nt++) {
            kernel_6x4_deep(Ap + mt * k4 * 24,
                            Bp + nt * k4 * 256,
                            C + mt * MR * ldc + nt * NR,
                            K, ldc * 4);
        }
    }
}

// GEMM using prefetch-optimized kernel
static void gemm_prfm(const int8_t* Ap, const int8_t* Bp,
                      int32_t* C, int ldc, int M, int N, int K) {
    int m_tiles = (M + MR - 1) / MR;
    int n_tiles = N / NR;
    int k4 = K / 4;

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int nt = 0; nt < n_tiles; nt++) {
            kernel_6x4_prfm(Ap + mt * k4 * 24,
                            Bp + nt * k4 * 256,
                            C + mt * MR * ldc + nt * NR,
                            K, ldc * 4);
        }
    }
}

int main(void) {
    printf("=== Pipelined Kernel Benchmark ===\n\n");

    int M = 192, N = 12288, K = 4096;
    int m_tiles = (M + MR - 1) / MR;
    int n_tiles = N / NR;

    printf("Config: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Tiles: %d M-tiles × %d N-tiles = %d total\n\n", m_tiles, n_tiles, m_tiles * n_tiles);

    int64_t ops = 2LL * M * N * K;
    printf("Ops per GEMM: %.2f GOP\n", ops / 1e9);
    printf("Peak: 512 GIOPS @ 2GHz\n\n");

    // Allocate
    int8_t* A = aligned_alloc(256, align256(M * K));
    int8_t* B = aligned_alloc(256, align256(N * K));
    int8_t* Ap = aligned_alloc(256, align256(m_tiles * (K/4) * 24));
    int8_t* Bp = aligned_alloc(256, align256(n_tiles * (K/4) * 256));
    int32_t* C1 = aligned_alloc(256, align256(M * N * 4));
    int32_t* C2 = aligned_alloc(256, align256(M * N * 4));
    int32_t* C3 = aligned_alloc(256, align256(M * N * 4));
    int32_t* C4 = aligned_alloc(256, align256(M * N * 4));
    int32_t* C_ref = aligned_alloc(256, align256(30 * 128 * 4));

    // Initialize
    for (int i = 0; i < M * K; i++) A[i] = (i % 7) - 3;
    for (int i = 0; i < N * K; i++) B[i] = ((i * 3) % 7) - 3;

    // Pack
    pack_A(A, Ap, M, K, K);
    pack_B(B, Bp, N, K, K);

    // Warmup
    memset(C1, 0, M * N * 4);
    memset(C2, 0, M * N * 4);
    memset(C3, 0, M * N * 4);
    memset(C4, 0, M * N * 4);
    gemm_baseline(Ap, Bp, C1, N, M, N, K);
    gemm_pipelined(Ap, Bp, C2, N, M, N, K);
    gemm_deep(Ap, Bp, C3, N, M, N, K);
    gemm_prfm(Ap, Bp, C4, N, M, N, K);

    // Correctness check
    printf("Checking correctness on 30×128 subset...\n");
    gemm_ref(A, B, C_ref, 30, 128, K, K, K, 128);

    int errors_base = 0, errors_pipe = 0, errors_deep = 0, errors_prfm = 0;
    for (int m = 0; m < 30; m++) {
        for (int n = 0; n < 128; n++) {
            if (C1[m * N + n] != C_ref[m * 128 + n] && errors_base < 5) {
                printf("baseline[%d,%d]: got %d, expected %d\n",
                       m, n, C1[m * N + n], C_ref[m * 128 + n]);
                errors_base++;
            }
            if (C2[m * N + n] != C_ref[m * 128 + n] && errors_pipe < 5) {
                printf("pipeV2[%d,%d]: got %d, expected %d\n",
                       m, n, C2[m * N + n], C_ref[m * 128 + n]);
                errors_pipe++;
            }
            if (C3[m * N + n] != C_ref[m * 128 + n] && errors_deep < 5) {
                printf("deep[%d,%d]: got %d, expected %d\n",
                       m, n, C3[m * N + n], C_ref[m * 128 + n]);
                errors_deep++;
            }
            if (C4[m * N + n] != C_ref[m * 128 + n] && errors_prfm < 5) {
                printf("prfm[%d,%d]: got %d, expected %d\n",
                       m, n, C4[m * N + n], C_ref[m * 128 + n]);
                errors_prfm++;
            }
        }
    }
    printf("Baseline: %s\n", errors_base == 0 ? "PASS" : "FAIL");
    printf("PipeV2: %s\n", errors_pipe == 0 ? "PASS" : "FAIL");
    printf("Deep (4x unroll): %s\n", errors_deep == 0 ? "PASS" : "FAIL");
    printf("Prefetch: %s\n\n", errors_prfm == 0 ? "PASS" : "FAIL");

    // Performance benchmarks
    const int iters = 10;
    double total_gops = 193.27;  // 2*M*N*K*iters / 1e9

    // Baseline kernel
    memset(C1, 0, M * N * 4);
    asm volatile("isb" ::: "memory");
    uint64_t t0 = rdtsc();
    for (int i = 0; i < iters; i++) {
        gemm_baseline(Ap, Bp, C1, N, M, N, K);
        asm volatile("" ::: "memory");
    }
    uint64_t t1 = rdtsc();
    asm volatile("isb" ::: "memory");

    // Pipelined kernel
    memset(C2, 0, M * N * 4);
    asm volatile("isb" ::: "memory");
    uint64_t t2 = rdtsc();
    for (int i = 0; i < iters; i++) {
        gemm_pipelined(Ap, Bp, C2, N, M, N, K);
        asm volatile("" ::: "memory");
    }
    uint64_t t3 = rdtsc();
    asm volatile("isb" ::: "memory");

    // Deep pipelined kernel (4x unroll) - skip timing since it's buggy
    // Just verify correctness above

    // Prefetch-optimized kernel
    memset(C4, 0, M * N * 4);
    asm volatile("isb" ::: "memory");
    uint64_t t4 = rdtsc();
    for (int i = 0; i < iters; i++) {
        gemm_prfm(Ap, Bp, C4, N, M, N, K);
        asm volatile("" ::: "memory");
    }
    uint64_t t5 = rdtsc();
    asm volatile("isb" ::: "memory");

    uint64_t ticks_base = t1 - t0;
    uint64_t ticks_pipe = t3 - t2;
    uint64_t ticks_prfm = t5 - t4;

    double ticks_base_f = (double)ticks_base;
    double ticks_pipe_f = (double)ticks_pipe;
    double ticks_prfm_f = (double)ticks_prfm;

    double time_base_ms = ticks_base_f / 1e8 * 1000.0;
    double time_pipe_ms = ticks_pipe_f / 1e8 * 1000.0;
    double time_prfm_ms = ticks_prfm_f / 1e8 * 1000.0;
    double gops_base = total_gops / (ticks_base_f / 1e8);
    double gops_pipe = total_gops / (ticks_pipe_f / 1e8);
    double gops_prfm = total_gops / (ticks_prfm_f / 1e8);

    printf("=== Performance Results (%d iterations) ===\n\n", iters);

    printf("Baseline kernel (kernel_6x4_kloop):\n");
    printf("  Ticks: %lu\n", (unsigned long)ticks_base);
    printf("  Time: %.1f ms\n", time_base_ms);
    printf("  GIOPS: %.1f (%.1f%% peak)\n\n", gops_base, gops_base / 512.0 * 100.0);

    printf("Pipelined kernel (kernel_6x4_pipeV2):\n");
    printf("  Ticks: %lu\n", (unsigned long)ticks_pipe);
    printf("  Time: %.1f ms\n", time_pipe_ms);
    printf("  GIOPS: %.1f (%.1f%% peak)\n", gops_pipe, gops_pipe / 512.0 * 100.0);
    printf("  Speedup vs baseline: %.2fx\n\n", gops_pipe / gops_base);

    printf("Prefetch-optimized kernel (kernel_6x4_prfm):\n");
    printf("  Ticks: %lu\n", (unsigned long)ticks_prfm);
    printf("  Time: %.1f ms\n", time_prfm_ms);
    printf("  GIOPS: %.1f (%.1f%% peak)\n", gops_prfm, gops_prfm / 512.0 * 100.0);
    printf("  Speedup vs baseline: %.2fx\n\n", gops_prfm / gops_base);

    // Cycle analysis
    double cycles_per_gemm_base = ticks_base_f / (double)iters * 20.0;
    double cycles_per_gemm_pipe = ticks_pipe_f / (double)iters * 20.0;
    double cycles_per_gemm_prfm = ticks_prfm_f / (double)iters * 20.0;
    double cycles_per_tile_base = cycles_per_gemm_base / (double)(m_tiles * n_tiles);
    double cycles_per_tile_pipe = cycles_per_gemm_pipe / (double)(m_tiles * n_tiles);
    double cycles_per_tile_prfm = cycles_per_gemm_prfm / (double)(m_tiles * n_tiles);
    double cycles_per_kiter_base = cycles_per_tile_base / (double)(K / 4);
    double cycles_per_kiter_pipe = cycles_per_tile_pipe / (double)(K / 4);
    double cycles_per_kiter_prfm = cycles_per_tile_prfm / (double)(K / 4);

    printf("=== Cycle Analysis ===\n");
    printf("Cycles per GEMM:\n");
    printf("  Baseline:  %.1fM cycles\n", cycles_per_gemm_base / 1e6);
    printf("  PipeV2:    %.1fM cycles\n", cycles_per_gemm_pipe / 1e6);
    printf("  Prefetch:  %.1fM cycles\n", cycles_per_gemm_prfm / 1e6);
    printf("Cycles per K-iteration (ideal: 12):\n");
    printf("  Baseline:  %.2f cycles\n", cycles_per_kiter_base);
    printf("  PipeV2:    %.2f cycles\n", cycles_per_kiter_pipe);
    printf("  Prefetch:  %.2f cycles\n", cycles_per_kiter_prfm);

    // Cleanup
    free(A); free(B); free(Ap); free(Bp);
    free(C1); free(C2); free(C3); free(C4); free(C_ref);

    return 0;
}
