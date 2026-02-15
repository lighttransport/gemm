#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define NUM_CORES 12
#define MR 6
#define NR 64

extern void micro_kernel_6x4_sector_unroll3(const int8_t*, const int8_t*, int32_t*, int64_t, int64_t, int64_t);

static inline void* fast_alloc(size_t size) {
    void* ptr = NULL;
    posix_memalign(&ptr, 256, size);
    if (ptr) memset(ptr, 0, size);
    return ptr;
}

static void pack_q(const int8_t* Q, int8_t* Qp, int64_t M, int64_t K, int64_t M_tiles) {
    int64_t K4 = K / 4;
    #pragma omp parallel for schedule(static)
    for (int64_t mt = 0; mt < M_tiles; mt++) {
        for (int64_t k4 = 0; k4 < K4; k4++) {
            for (int row = 0; row < MR; row++) {
                int8_t* dst = Qp + mt * (K4 * MR * 4) + k4 * (MR * 4) + row * 4;
                const int8_t* src = Q + (mt * MR + row) * K + k4 * 4;
                memcpy(dst, src, 4);
            }
        }
    }
}

static void pack_k(const int8_t* K_raw, int8_t* Kp, int64_t N, int64_t K, int64_t N_tiles) {
    int64_t K4 = K / 4;
    #pragma omp parallel for schedule(static)
    for (int64_t nt = 0; nt < N_tiles; nt++) {
        for (int64_t k4 = 0; k4 < K4; k4++) {
            for (int vec = 0; vec < 4; vec++) {
                for (int lane = 0; lane < 16; lane++) {
                    int64_t n_idx = nt * NR + vec * 16 + lane;
                    int8_t* dst = Kp + nt * (K4 * 256) + k4 * 256 + vec * 64 + lane * 4;
                    const int8_t* src = K_raw + n_idx * K + k4 * 4;
                    memcpy(dst, src, 4);
                }
            }
        }
    }
}

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double run_benchmark(int8_t* Qp, int8_t* Kp, int32_t* S,
                   int64_t M_tiles, int64_t N_tiles, int64_t K,
                   int64_t Qp_tile_size, int64_t Kp_tile_size,
                   int64_t C_stride, int niters) {
    // Warmup
    for (int w = 0; w < 10; w++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int64_t nt_per_thread = (N_tiles + NUM_CORES - 1) / NUM_CORES;
            int64_t nt_start = tid * nt_per_thread;
            int64_t nt_end = (nt_start + nt_per_thread < N_tiles) ? nt_start + nt_per_thread : N_tiles;

            for (int64_t nt = nt_start; nt < nt_end; nt++) {
                const int8_t* Kpp = Kp + nt * Kp_tile_size;
                for (int64_t mt = 0; mt < M_tiles; mt++) {
                    const int8_t* Qpp = Qp + mt * Qp_tile_size;
                    int32_t* Sp = S + (mt * N_tiles + nt) * MR * 4 * 16;
                    micro_kernel_6x4_sector_unroll3(Qpp, Kpp, Sp, K, 0, C_stride);
                }
            }
        }
    }

    double best_sdot = 0;

    for (int run = 0; run < 10; run++) {
        double t0 = get_time();

        for (int iter = 0; iter < niters; iter++) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int64_t nt_per_thread = (N_tiles + NUM_CORES - 1) / NUM_CORES;
                int64_t nt_start = tid * nt_per_thread;
                int64_t nt_end = (nt_start + nt_per_thread < N_tiles) ? nt_start + nt_per_thread : N_tiles;

                for (int64_t nt = nt_start; nt < nt_end; nt++) {
                    const int8_t* Kpp = Kp + nt * Kp_tile_size;
                    for (int64_t mt = 0; mt < M_tiles; mt++) {
                        const int8_t* Qpp = Qp + mt * Qp_tile_size;
                        int32_t* Sp = S + (mt * N_tiles + nt) * MR * 4 * 16;
                        micro_kernel_6x4_sector_unroll3(Qpp, Kpp, Sp, K, 0, C_stride);
                    }
                }
            }
        }

        double elapsed = get_time() - t0;
        double sdots_per_tile = (double)K / 4.0 * 24.0;
        double total_tiles = M_tiles * N_tiles;
        double total_sdots = sdots_per_tile * total_tiles * niters;
        double sdot_per_cycle = total_sdots / (elapsed * 2e9) / NUM_CORES;

        if (sdot_per_cycle > best_sdot) best_sdot = sdot_per_cycle;
    }

    return best_sdot;
}

int main(int argc, char** argv) {
    int64_t M = 78;
    int64_t N = 8192;
    int64_t K = 512;
    int niters = 200;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0) niters = atoi(argv[++i]);
    }

    omp_set_num_threads(NUM_CORES);

    int64_t M_tiles = M / MR;
    int64_t N_tiles = N / NR;
    int64_t K4 = K / 4;
    int64_t C_stride = 4 * 16 * sizeof(int32_t);
    int64_t Kp_tile_size = K4 * 256;
    int64_t Qp_tile_size = K4 * MR * 4;

    size_t Q_size = M * K;
    size_t Qp_size = M_tiles * Qp_tile_size;
    size_t K_size = N * K;
    size_t Kp_size = N_tiles * Kp_tile_size;
    size_t S_size = M_tiles * N_tiles * MR * 4 * 16 * sizeof(int32_t);

    int8_t* Q = (int8_t*)fast_alloc(Q_size);
    int8_t* Qp = (int8_t*)fast_alloc(Qp_size);
    int8_t* K_mat = (int8_t*)fast_alloc(K_size);
    int8_t* Kp = (int8_t*)fast_alloc(Kp_size);
    int32_t* S = (int32_t*)fast_alloc(S_size);

    srand(42);
    for (size_t i = 0; i < Q_size; i++) Q[i] = (int8_t)((rand() % 64) - 32);
    for (size_t i = 0; i < K_size; i++) K_mat[i] = (int8_t)((rand() % 64) - 32);

    pack_q(Q, Qp, M, K, M_tiles);
    pack_k(K_mat, Kp, N, K, N_tiles);

    double sdot = run_benchmark(Qp, Kp, S, M_tiles, N_tiles, K, Qp_tile_size, Kp_tile_size, C_stride, niters);
    double efficiency = sdot / 2.0 * 100.0;
    double gops = sdot * 2 * NUM_CORES * 16 * 4 * 2;

    printf("M=%3ld K=%3ld N=%5ld: %.3f SDOT/cycle (%5.1f%%) %7.0f GOPS\n",
           M, K, N, sdot, efficiency, gops);

    free(Q); free(Qp); free(K_mat); free(Kp); free(S);
    return 0;
}
