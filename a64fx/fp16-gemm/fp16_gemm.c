#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

// A64FX SVE is fixed to 512-bit (64 bytes), so each vector holds 32 FP16 lanes.
// B panels are laid out as 4 vectors per k (256 bytes stride) to match the
// assembly kernels' ld1h pattern and fixed pointer increment.
#define VL_BYTES 64
#define VL_ELEM  (VL_BYTES / sizeof(uint16_t))
#define NR 4
#define MR5 5
#define MR6 6
#define DEFAULT_M 8192
#define DEFAULT_K 512

extern void micro_kernel_5x4_f16_sve(const __fp16* Ap, const __fp16* Bp,
                                     __fp16* C, int64_t ldc, int64_t K);
extern void micro_kernel_6x4_f16_sve(const __fp16* Ap, const __fp16* Bp,
                                     __fp16* C, int64_t ldc, int64_t K);

static inline float to_float(__fp16 h) { return (float)h; }

static void fill_random(__fp16* buf, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        buf[i] = (__fp16)(r * 2.0f - 1.0f);
    }
}

static void* aligned_alloc_64(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return NULL;
    }
    return ptr;
}

static inline int64_t round_up(int64_t x, int64_t m) {
    return (x + m - 1) / m * m;
}

// Pack A into contiguous panels: for each k, store MR rows
static void pack_A_full(const __fp16* A, int64_t lda, __fp16* Ap,
                        int64_t M, int64_t K, int mr) {
    int64_t M_pad = round_up(M, mr);
    for (int64_t ir = 0; ir < M_pad; ir += mr) {
        for (int64_t k = 0; k < K; k++) {
            for (int i = 0; i < mr; i++) {
                int64_t row = ir + i;
                __fp16 v = (row < M) ? A[row * lda + k] : (__fp16)0.0f;
                *Ap++ = v;
            }
        }
    }
}

// Pack B into 4 vectors per k (stride 256 bytes per k)
static void pack_B(const __fp16* B, int64_t ldb, __fp16* Bp, int64_t K) {
    const int64_t vec_elems = VL_ELEM;
    for (int64_t k = 0; k < K; k++) {
        for (int j = 0; j < NR; j++) {
            memcpy(Bp, &B[k * ldb + j * vec_elems],
                   (size_t)vec_elems * sizeof(__fp16));
            Bp += vec_elems;
        }
    }
}

// Reference GEMM for a small MR x (NR*VL_ELEM) tile
static void gemm_ref(const __fp16* A, const __fp16* B, __fp16* C,
                     int mr, int64_t K) {
    const int64_t N = NR * VL_ELEM;
    for (int i = 0; i < mr; i++) {
        for (int64_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                acc += to_float(A[i * K + k]) * to_float(B[k * N + j]);
            }
            C[i * N + j] = (__fp16)acc;
        }
    }
}

static float max_abs_diff(const __fp16* a, const __fp16* b, size_t n) {
    float maxd = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(to_float(a[i]) - to_float(b[i]));
        if (d > maxd) maxd = d;
    }
    return maxd;
}

static inline uint64_t read_cntvct(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t read_cntfrq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static uint64_t read_cntfrq_el0(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));

    int64_t M = DEFAULT_M;
    int64_t K = DEFAULT_K;  // default depth; can be overridden via argv/env
    const char* k_env = getenv("K");
    if (argc > 1) {
        long val = strtol(argv[1], NULL, 10);
        if (val > 0) M = val;
    }
    if (argc > 2) {
        long val = strtol(argv[2], NULL, 10);
        if (val > 0) K = val;
    } else if (k_env) {
        long val = strtol(k_env, NULL, 10);
        if (val > 0) K = val;
    }

    const int64_t N = NR * VL_ELEM;   // logical N dimension (vector-width times 4)
    const int64_t ldc = N;

    int64_t M_pad5 = round_up(M, MR5);
    int64_t M_pad6 = round_up(M, MR6);

    size_t size_A5 = (size_t)M * (size_t)K * sizeof(__fp16);
    size_t size_A6 = (size_t)M * (size_t)K * sizeof(__fp16);
    size_t size_Ap5 = (size_t)M_pad5 * (size_t)K * sizeof(__fp16);
    size_t size_Ap6 = (size_t)M_pad6 * (size_t)K * sizeof(__fp16);
    size_t size_B  = (size_t)K * (size_t)N * sizeof(__fp16);
    size_t size_C5 = (size_t)M_pad5 * (size_t)N * sizeof(__fp16);
    size_t size_C6 = (size_t)M_pad6 * (size_t)N * sizeof(__fp16);

    __fp16* A5   = aligned_alloc_64(size_A5);
    __fp16* Ap5  = aligned_alloc_64(size_Ap5);
    __fp16* C5   = aligned_alloc_64(size_C5);

    __fp16* A6   = aligned_alloc_64(size_A6);
    __fp16* Ap6  = aligned_alloc_64(size_Ap6);
    __fp16* C6   = aligned_alloc_64(size_C6);

    __fp16* B    = aligned_alloc_64(size_B);
    __fp16* Bp   = aligned_alloc_64(size_B);

    if (!A5 || !Ap5 || !C5 ||
        !A6 || !Ap6 || !C6 ||
        !B  || !Bp) {
        fprintf(stderr, "Allocation failed\n");
        free(A5); free(Ap5); free(C5);
        free(A6); free(Ap6); free(C6);
        free(B);  free(Bp);
        return 1;
    }

    fill_random(A5, M * K);
    memcpy(A6, A5, size_A6); // reuse same data
    fill_random(B, K * N);

    pack_A_full(A5, K, Ap5, M, K, MR5);
    pack_A_full(A6, K, Ap6, M, K, MR6);
    pack_B(B, N, Bp, K);

    printf("M = %ld, K = %ld, N = %ld (NR=%d, VL_ELEM=%ld)\n",
           (long)M, (long)K, (long)N, NR, (long)VL_ELEM);
    const char* debug_timing = getenv("DEBUG_TIMING");
    if (debug_timing) {
        printf("cntfrq_el0 = %lu\n", (unsigned long)read_cntfrq_el0());
    }

    // Benchmark 5x4
    memset(C5, 0, size_C5);
    (void)now_sec();
    double t0 = now_sec();
    for (int64_t i = 0; i < M_pad5; i += MR5) {
        const __fp16* Ap_tile = Ap5 + (i / MR5) * (MR5 * K);
        __fp16* C_tile = C5 + i * N;
        micro_kernel_5x4_f16_sve(Ap_tile, Bp, C_tile, ldc, K);
    }
    double t1 = now_sec();
    double flops5 = (double)M_pad5 * (double)N * (double)K * 2.0;
    double gflops5 = flops5 / (t1 - t0) * 1e-9;
    double dt5 = t1 - t0;
    printf("5x4 kernel: dt=%.6f s (%.3f ms), %.2f GFLOPS (padded M=%ld)\n",
           dt5, dt5 * 1e3, gflops5, (long)M_pad5);
    if (debug_timing) {
        printf("  5x4 timing raw: t0=%.6f t1=%.6f dt=%.6f\n", t0, t1, dt5);
    }

    // Benchmark 6x4
    memset(C6, 0, size_C6);
    (void)now_sec();
    double t2 = now_sec();
    for (int64_t i = 0; i < M_pad6; i += MR6) {
        const __fp16* Ap_tile = Ap6 + (i / MR6) * (MR6 * K);
        __fp16* C_tile = C6 + i * N;
        micro_kernel_6x4_f16_sve(Ap_tile, Bp, C_tile, ldc, K);
    }
    double t3 = now_sec();
    double flops6 = (double)M_pad6 * (double)N * (double)K * 2.0;
    double dt6 = t3 - t2;
    double gflops6 = flops6 / dt6 * 1e-9;
    printf("6x4 kernel: dt=%.6f s (%.3f ms), %.2f GFLOPS (padded M=%ld)\n",
           dt6, dt6 * 1e3, gflops6, (long)M_pad6);
    if (debug_timing) {
        printf("  6x4 timing raw: t2=%.6f t3=%.6f dt=%.6f\n", t2, t3, dt6);
    }

    // Quick correctness on first block to ensure kernel matches ref
    __fp16 C_ref_block[MR6 * NR * VL_ELEM] __attribute__((aligned(64)));
    memset(C_ref_block, 0, sizeof(C_ref_block));
    gemm_ref(A6, B, C_ref_block, MR6, K);
    float diff6 = max_abs_diff(C_ref_block, C6, (size_t)MR6 * (size_t)N);
    const float tol = 1e-1f;
    if (diff6 > tol) {
        fprintf(stderr, "Validation warning (first 6 rows diff=%.3e)\n", diff6);
    }

    free(A5); free(Ap5); free(C5);
    free(A6); free(Ap6); free(C6);
    free(B);  free(Bp);
    return 0;
}
