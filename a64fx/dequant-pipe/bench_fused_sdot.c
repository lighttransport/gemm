#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

enum { BLOCK_BYTES = 4096, K_BLOCK = 128, N_TILE = 64, MAX_CORES = 12 };

extern void fused_i4_i8_m1_k4_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_i4_i8_m1_k4_lut_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_i4_i8_m1_k4_pipe_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_i4_i8_m1_k4_super_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_i4_i16_m1_k2_sve(const uint8_t *, const int16_t *, int64_t *);

typedef enum { PATH_I8, PATH_I16 } path_kind;
typedef enum { KERNEL_SHIFT, KERNEL_LUT, KERNEL_PIPE, KERNEL_SUPER } kernel_kind;
typedef void (*fused_i8_fn)(const uint8_t *, const int8_t *, int32_t *);

static fused_i8_fn select_i8_kernel(kernel_kind kernel)
{
    if (kernel == KERNEL_LUT) return fused_i4_i8_m1_k4_lut_sve;
    if (kernel == KERNEL_PIPE) return fused_i4_i8_m1_k4_pipe_sve;
    if (kernel == KERNEL_SUPER) return fused_i4_i8_m1_k4_super_sve;
    return fused_i4_i8_m1_k4_sve;
}

static const char *kernel_name(kernel_kind kernel)
{
    if (kernel == KERNEL_LUT) return "lut";
    if (kernel == KERNEL_PIPE) return "pipe";
    if (kernel == KERNEL_SUPER) return "super";
    return "shift";
}

typedef struct {
    const uint8_t *packed;
    const void *activation;
    size_t bytes;
    int iterations;
    int cpu;
    path_kind path;
    kernel_kind kernel;
    atomic_int *ready;
    atomic_int *start;
    uint64_t begin;
    uint64_t end;
    uint64_t checksum;
    int error;
} worker;

static inline uint64_t cntvct(void)
{
    uint64_t value;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(value));
    return value;
}

static inline uint64_t cntfrq(void)
{
    uint64_t value;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(value));
    return value;
}

static int pin_cpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void *run_worker(void *opaque)
{
    worker *w = opaque;
    int32_t out8[4 * N_TILE] __attribute__((aligned(256)));
    int64_t out16[2 * N_TILE] __attribute__((aligned(256)));
    const size_t group = w->path == PATH_I8 ? 4 * BLOCK_BYTES : 2 * BLOCK_BYTES;
    if (pin_cpu(w->cpu) != 0) w->error = errno ? errno : EINVAL;
    atomic_fetch_add_explicit(w->ready, 1, memory_order_release);
    while (!atomic_load_explicit(w->start, memory_order_acquire))
        __asm__ volatile("yield");
    w->begin = cntvct();
    if (!w->error) {
        for (int iteration = 0; iteration < w->iterations; ++iteration) {
            for (size_t offset = 0; offset < w->bytes; offset += group) {
                if (w->path == PATH_I8) {
                    fused_i8_fn kernel = select_i8_kernel(w->kernel);
                    kernel(w->packed + offset, w->activation, out8);
                    w->checksum += (uint32_t)out8[(offset / group) & 255u];
                } else {
                    fused_i4_i16_m1_k2_sve(w->packed + offset, w->activation, out16);
                    w->checksum += (uint64_t)out16[(offset / group) & 127u];
                }
            }
        }
    }
    w->end = cntvct();
    return NULL;
}

static int nibble_i4(uint8_t x)
{
    return (int)(int8_t)(x << 4) >> 4;
}

static int verify(kernel_kind kernel_kind)
{
    uint8_t packed[4 * BLOCK_BYTES] __attribute__((aligned(256)));
    uint8_t packed_super[4 * BLOCK_BYTES] __attribute__((aligned(256)));
    int8_t act8[4 * K_BLOCK] __attribute__((aligned(256)));
    int16_t act16[2 * K_BLOCK] __attribute__((aligned(256)));
    int32_t out8[4 * N_TILE] __attribute__((aligned(256)));
    int64_t out16[2 * N_TILE] __attribute__((aligned(256)));
    for (size_t i = 0; i < sizeof(packed); ++i)
        packed[i] = (uint8_t)(i * 29u + (i >> 3) * 7u);
    for (size_t kg = 0; kg < K_BLOCK / 4; ++kg)
        for (size_t block = 0; block < 4; ++block)
            memcpy(packed_super + (kg * 4 + block) * 128,
                   packed + block * BLOCK_BYTES + kg * 128, 128);
    for (size_t i = 0; i < sizeof(act8); ++i) act8[i] = (int8_t)((i * 11u) % 23u - 11);
    for (size_t i = 0; i < sizeof(act16) / sizeof(act16[0]); ++i)
        act16[i] = (int16_t)((i * 17u) % 257u - 128);
    fused_i8_fn kernel = select_i8_kernel(kernel_kind);
    kernel(kernel_kind == KERNEL_SUPER ? packed_super : packed, act8, out8);
    fused_i4_i16_m1_k2_sve(packed, act16, out16);
    for (size_t b = 0; b < 4; ++b) {
        for (size_t n = 0; n < N_TILE; ++n) {
            int64_t ref = 0;
            for (size_t k = 0; k < K_BLOCK; ++k) {
                size_t nib = k / 4 * N_TILE * 4 + n / 16 * 64 + n % 16 * 4 + k % 4;
                uint8_t byte = packed[b * BLOCK_BYTES + nib / 2];
                int weight = nibble_i4((nib & 1u) ? byte >> 4 : byte);
                ref += weight * act8[b * K_BLOCK + k];
            }
            if (out8[b * N_TILE + n] != ref) {
                fprintf(stderr, "INT8 mismatch block=%zu n=%zu got=%d ref=%ld\n",
                        b, n, out8[b * N_TILE + n], (long)ref);
                return -1;
            }
        }
    }
    for (size_t b = 0; b < 2; ++b) {
        for (size_t n = 0; n < N_TILE; ++n) {
            int64_t ref = 0;
            for (size_t k = 0; k < K_BLOCK; ++k) {
                size_t byte_index = k / 4 * 128 + n / 32 * 64 + n % 16 * 4 + k % 4;
                uint8_t byte = packed[b * BLOCK_BYTES + byte_index];
                int weight = nibble_i4((n & 16u) ? byte >> 4 : byte);
                ref += (int64_t)weight * act16[b * K_BLOCK + k];
            }
            if (out16[b * N_TILE + n] != ref) {
                fprintf(stderr, "INT16 mismatch block=%zu n=%zu got=%ld ref=%ld\n",
                        b, n, (long)out16[b * N_TILE + n], (long)ref);
                return -1;
            }
        }
    }
    puts("fused correctness: PASS (INT4->INT8 SDOT and INT4->INT16 SDOT)");
    return 0;
}

static void usage(const char *name)
{
    fprintf(stderr, "usage: %s [--path int8|int16] [--kernel shift|lut|pipe|super] [--cores N] [--mib N] "
                    "[--iterations N] [--trials N] [--core-base N] [--skew-kib N] [--verify]\n", name);
}

int main(int argc, char **argv)
{
    path_kind path = PATH_I8;
    kernel_kind kernel = KERNEL_SHIFT;
    int cores = 12, iterations = 10, trials = 3, core_base = 12;
    size_t mib = 240;
    size_t skew_kib = 0;
    int do_verify = 0;
    static const struct option options[] = {
        {"path", required_argument, NULL, 'p'}, {"cores", required_argument, NULL, 'c'},
        {"kernel", required_argument, NULL, 'k'},
        {"mib", required_argument, NULL, 'm'}, {"iterations", required_argument, NULL, 'i'},
        {"trials", required_argument, NULL, 't'}, {"core-base", required_argument, NULL, 'b'},
        {"skew-kib", required_argument, NULL, 's'},
        {"verify", no_argument, NULL, 'v'}, {NULL, 0, NULL, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "p:k:c:m:i:t:b:s:v", options, NULL)) != -1) {
        switch (opt) {
        case 'p':
            if (!strcmp(optarg, "int8")) path = PATH_I8;
            else if (!strcmp(optarg, "int16")) path = PATH_I16;
            else { usage(argv[0]); return 2; }
            break;
        case 'k':
            if (!strcmp(optarg, "shift")) kernel = KERNEL_SHIFT;
            else if (!strcmp(optarg, "lut")) kernel = KERNEL_LUT;
            else if (!strcmp(optarg, "pipe")) kernel = KERNEL_PIPE;
            else if (!strcmp(optarg, "super")) kernel = KERNEL_SUPER;
            else { usage(argv[0]); return 2; }
            break;
        case 'c': cores = atoi(optarg); break;
        case 'm': mib = strtoull(optarg, NULL, 0); break;
        case 'i': iterations = atoi(optarg); break;
        case 't': trials = atoi(optarg); break;
        case 'b': core_base = atoi(optarg); break;
        case 's': skew_kib = strtoull(optarg, NULL, 0); break;
        case 'v': do_verify = 1; break;
        default: usage(argv[0]); return 2;
        }
    }
    if (do_verify && verify(kernel) != 0) return 1;
    if (cores < 1 || cores > MAX_CORES || iterations < 1 || trials < 1 || trials > 32) return 2;
    size_t group = path == PATH_I8 ? 4 * BLOCK_BYTES : 2 * BLOCK_BYTES;
    size_t skew = skew_kib * 1024u;
    if (skew % group != 0) {
        fprintf(stderr, "--skew-kib must preserve the %zu-byte kernel group alignment\n", group);
        return 2;
    }
    size_t bytes = mib * 1024u * 1024u;
    bytes -= bytes % ((size_t)cores * group);
    if (!bytes) return 2;
    uint8_t *packed = NULL;
    size_t allocation_bytes = bytes + (size_t)(cores - 1) * skew;
    if (posix_memalign((void **)&packed, 256, allocation_bytes) != 0) return 1;
    (void)madvise(packed, allocation_bytes, MADV_HUGEPAGE);
    if (pin_cpu(core_base) != 0) { perror("affinity"); return 1; }
    for (size_t i = 0; i < allocation_bytes; i += 256)
        for (size_t j = 0; j < 256; ++j) packed[i + j] = (uint8_t)(i + j * 13u);
    int8_t act8[4 * K_BLOCK] __attribute__((aligned(256)));
    int16_t act16[2 * K_BLOCK] __attribute__((aligned(256)));
    for (size_t i = 0; i < sizeof(act8); ++i) act8[i] = (int8_t)(i % 15u - 7);
    for (size_t i = 0; i < sizeof(act16) / sizeof(act16[0]); ++i) act16[i] = (int16_t)(i % 127u - 63);
    double values[32];
    for (int run = 0; run <= trials; ++run) {
        pthread_t threads[MAX_CORES];
        worker workers[MAX_CORES];
        atomic_int ready, start;
        atomic_init(&ready, 0); atomic_init(&start, 0);
        size_t per_core = bytes / (size_t)cores;
        for (int c = 0; c < cores; ++c) {
            workers[c] = (worker){ .packed = packed + (size_t)c * (per_core + skew),
                .activation = path == PATH_I8 ? (const void *)act8 : (const void *)act16,
                .bytes = per_core, .iterations = iterations, .cpu = core_base + c,
                .path = path, .ready = &ready, .start = &start };
            workers[c].kernel = kernel;
            int rc = pthread_create(&threads[c], NULL, run_worker, &workers[c]);
            if (rc) { fprintf(stderr, "pthread_create: %s\n", strerror(rc)); return 1; }
        }
        while (atomic_load_explicit(&ready, memory_order_acquire) != cores) __asm__ volatile("yield");
        atomic_store_explicit(&start, 1, memory_order_release);
        uint64_t first = UINT64_MAX, last = 0, checksum = 0;
        for (int c = 0; c < cores; ++c) {
            pthread_join(threads[c], NULL);
            if (workers[c].error) { fprintf(stderr, "worker: %s\n", strerror(workers[c].error)); return 1; }
            if (workers[c].begin < first) first = workers[c].begin;
            if (workers[c].end > last) last = workers[c].end;
            checksum += workers[c].checksum;
        }
        double seconds = (double)(last - first) / (double)cntfrq();
        double bandwidth = (double)bytes * iterations / seconds / 1e9;
        if (run > 0) values[run - 1] = bandwidth;
        printf("path=%s kernel=%s cores=%d skew_kib=%zu %s=%d packed_GB/s=%.2f logical_Gweight/s=%.2f checksum=%lu\n",
               path == PATH_I8 ? "int8" : "int16",
               path == PATH_I16 ? "split" :
                   kernel_name(kernel),
               cores, skew_kib, run == 0 ? "warmup" : "trial", run == 0 ? 1 : run,
               bandwidth, 2.0 * bandwidth, (unsigned long)checksum);
    }
    for (int i = 0; i < trials; ++i)
        for (int j = i + 1; j < trials; ++j)
            if (values[j] < values[i]) { double x = values[i]; values[i] = values[j]; values[j] = x; }
    printf("summary path=%s kernel=%s cores=%d skew_kib=%zu packed_GB/s_median=%.2f best=%.2f\n",
           path == PATH_I8 ? "int8" : "int16",
           path == PATH_I16 ? "split" :
               kernel_name(kernel),
           cores, skew_kib, values[trials / 2], values[trials - 1]);
    free(packed);
    return 0;
}
