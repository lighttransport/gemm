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
#include "fused_opt.h"

enum { BLOCK_BYTES = 4096, K_BLOCK = 128, N_TILE = 64, MAX_CORES = 12 };

extern void fused_i4_i8_m1_k4_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_i4_i8_m1_k4_lut_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_i4_i8_m1_k4_pipe_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_i4_i8_m1_k4_super_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_i4_i16_m1_k2_sve(const uint8_t *, const int16_t *, int64_t *);
extern void fused_i4_i16_m1_k2_super_sve(const uint8_t *, const int16_t *, int64_t *);
extern void fused_fp4_i16_m1_k2_super_sve(const uint8_t *, const int16_t *, int64_t *);
extern void fused_i4_i16x8_m1_k2_super_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_fp4_i16x8_m1_k2_super_sve(const uint8_t *, const int8_t *, int32_t *);
extern void fused_i4_f16_m1_k4_super_sve(const uint8_t *, const _Float16 *, _Float16 *);
extern void fused_fp4_f16_m1_k4_super_sve(const uint8_t *, const _Float16 *, _Float16 *);

typedef enum { PATH_I8, PATH_I16, PATH_I16X8, PATH_F16, PATH_FULL } path_kind;
typedef enum { FORMAT_I4, FORMAT_FP4 } format_kind;
typedef enum { KERNEL_SHIFT, KERNEL_LUT, KERNEL_PIPE, KERNEL_SUPER, KERNEL_OPT, KERNEL_OPT2 } kernel_kind;
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
    if (kernel == KERNEL_OPT) return "opt";
    if (kernel == KERNEL_OPT2) return "opt2";
    return "shift";
}

static const char *reported_kernel_name(path_kind path, kernel_kind kernel)
{
    if (kernel >= KERNEL_OPT) return kernel_name(kernel);
    if (kernel == KERNEL_SUPER) return "super";
    if (path == PATH_I16) return "split";
    return kernel_name(kernel);
}

typedef struct {
    const uint8_t *packed;
    const void *activation;
    size_t bytes;
    int iterations;
    int cpu;
    path_kind path;
    format_kind format;
    kernel_kind kernel;
    atomic_int *ready;
    atomic_int *start;
    uint64_t begin;
    uint64_t end;
    uint64_t checksum;
    int error;
    int read_only;
} worker;

extern void hbm_read_256_sve(const uint8_t *, size_t);

static size_t group_bytes(path_kind path)
{
    if (path == PATH_FULL) return FUSED_FULL_BYTES;
    return path == PATH_I16 || path == PATH_I16X8 ? 8192 : 16384;
}

static const char *path_name(path_kind path)
{
    static const char *names[] = {"int8", "int16", "int16x8", "fp16", "int16x8-full"};
    return names[path];
}

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
    int32_t out16x8[2 * N_TILE] __attribute__((aligned(256)));
    _Float16 outf16[4 * N_TILE] __attribute__((aligned(256)));
    const size_t group = group_bytes(w->path);
    w->error = pin_cpu(w->cpu);
    atomic_fetch_add_explicit(w->ready, 1, memory_order_release);
    while (!atomic_load_explicit(w->start, memory_order_acquire))
        __asm__ volatile("yield");
    w->begin = cntvct();
    if (!w->error) {
        for (int iteration = 0; iteration < w->iterations; ++iteration) {
            if (w->read_only) {
                hbm_read_256_sve(w->packed, w->bytes);
                continue;
            }
            for (size_t offset = 0; offset < w->bytes; offset += group) {
                if (w->path == PATH_I8) {
                    fused_i8_fn kernel = select_i8_kernel(w->kernel);
                    kernel(w->packed + offset, w->activation, out8);
                    w->checksum += (uint32_t)out8[(offset / group) & 255u];
                } else if (w->path == PATH_I16) {
                    if (w->kernel == KERNEL_OPT) {
                        if (w->format == FORMAT_FP4)
                            fused_fp4_i16_opt_sve(w->packed + offset, w->activation, out16);
                        else
                            fused_i4_i16_opt_sve(w->packed + offset, w->activation, out16);
                    } else if (w->kernel != KERNEL_SUPER)
                        fused_i4_i16_m1_k2_sve(w->packed + offset,
                                               w->activation, out16);
                    else if (w->format == FORMAT_FP4)
                        fused_fp4_i16_m1_k2_super_sve(w->packed + offset,
                                                      w->activation, out16);
                    else
                        fused_i4_i16_m1_k2_super_sve(w->packed + offset,
                                                    w->activation, out16);
                    w->checksum += (uint64_t)out16[(offset / group) & 127u];
                } else if (w->path == PATH_FULL) {
                    if (w->format == FORMAT_FP4)
                        fused_fp4_i16x8_full_sve(w->packed + offset, w->activation, out16x8);
                    else
                        fused_i4_i16x8_full_sve(w->packed + offset, w->activation, out16x8);
                    w->checksum += (uint32_t)out16x8[(offset / group) & 127u];
                } else if (w->path == PATH_I16X8) {
                    if (w->format == FORMAT_FP4)
                        fused_fp4_i16x8_m1_k2_super_sve(w->packed + offset,
                                                        w->activation, out16x8);
                    else
                        fused_i4_i16x8_m1_k2_super_sve(w->packed + offset,
                                                       w->activation, out16x8);
                    w->checksum += (uint32_t)out16x8[(offset / group) & 127u];
                } else {
                    if (w->kernel == KERNEL_OPT) {
                        if (w->format == FORMAT_FP4)
                            fused_fp4_f16_opt1_sve(w->packed + offset, w->activation, outf16);
                        else
                            fused_i4_f16_opt1_sve(w->packed + offset, w->activation, outf16);
                    } else if (w->kernel == KERNEL_OPT2) {
                        if (w->format == FORMAT_FP4)
                            fused_fp4_f16_opt2_sve(w->packed + offset, w->activation, outf16);
                        else
                            fused_i4_f16_opt2_sve(w->packed + offset, w->activation, outf16);
                    } else if (w->format == FORMAT_FP4)
                        fused_fp4_f16_m1_k4_super_sve(w->packed + offset,
                                                      w->activation, outf16);
                    else
                        fused_i4_f16_m1_k4_super_sve(w->packed + offset,
                                                    w->activation, outf16);
                    union { _Float16 h; uint16_t u; } bits = {
                        .h = outf16[(offset / group) & 255u]
                    };
                    w->checksum += bits.u;
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

static int nibble_fp4(uint8_t x)
{
    static const int8_t table[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };
    return table[x & 15u];
}

/* Exact on [-32768, 32639]: a = lo + 256*hi with two signed bytes. */
static int pack_i16x8(int8_t *dst, const int16_t *src, size_t blocks)
{
    for (size_t b = 0; b < blocks; ++b) {
        for (size_t k = 0; k < K_BLOCK; ++k) {
            int value = src[b * K_BLOCK + k];
            int lo = (int8_t)value;
            int hi = (value - lo) / 256;
            if (hi < -128 || hi > 127) return -1;
            dst[b * 2 * K_BLOCK + k] = (int8_t)lo;
            dst[b * 2 * K_BLOCK + K_BLOCK + k] = (int8_t)hi;
        }
    }
    return 0;
}

static int verify(kernel_kind kernel_kind)
{
    uint8_t packed[4 * BLOCK_BYTES] __attribute__((aligned(256)));
    uint8_t packed_super[4 * BLOCK_BYTES] __attribute__((aligned(256)));
    int8_t act8[4 * K_BLOCK] __attribute__((aligned(256)));
    int16_t act16[2 * K_BLOCK] __attribute__((aligned(256)));
    int8_t act16x8[4 * K_BLOCK] __attribute__((aligned(256)));
    int32_t out8[4 * N_TILE] __attribute__((aligned(256)));
    int64_t out16[2 * N_TILE] __attribute__((aligned(256)));
    _Float16 actf16[4 * K_BLOCK] __attribute__((aligned(256)));
    _Float16 outf16[4 * N_TILE] __attribute__((aligned(256)));
    for (size_t i = 0; i < sizeof(packed); ++i)
        packed[i] = (uint8_t)(i * 29u + (i >> 3) * 7u);
    for (size_t kg = 0; kg < K_BLOCK / 4; ++kg)
        for (size_t block = 0; block < 4; ++block)
            memcpy(packed_super + (kg * 4 + block) * 128,
                   packed + block * BLOCK_BYTES + kg * 128, 128);
    for (size_t i = 0; i < sizeof(act8); ++i) act8[i] = (int8_t)((i * 11u) % 23u - 11);
    for (size_t i = 0; i < sizeof(act16) / sizeof(act16[0]); ++i)
        act16[i] = (int16_t)((i * 17u) % 257u - 128);
    if (pack_i16x8(act16x8, act16, 2) != 0) return -1;
    for (size_t i = 0; i < sizeof(actf16) / sizeof(actf16[0]); ++i)
        actf16[i] = (i & 1u) ? (_Float16)-1.0f : (_Float16)1.0f;
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

    /* Repack the split-nibble INT16 layout into one sequential two-block stream. */
    for (size_t kg = 0; kg < K_BLOCK / 4; ++kg)
        for (size_t b = 0; b < 2; ++b)
            memcpy(packed_super + (kg * 2 + b) * 128,
                   packed + b * BLOCK_BYTES + kg * 128, 128);
    fused_i4_i16_m1_k2_super_sve(packed_super, act16, out16);
    for (size_t b = 0; b < 2; ++b) for (size_t n = 0; n < N_TILE; ++n) {
        int64_t ref = 0;
        for (size_t k = 0; k < K_BLOCK; ++k) {
            size_t byte_index = (k / 4 * 2 + b) * 128 +
                                n / 32 * 64 + n % 16 * 4 + k % 4;
            uint8_t byte = packed_super[byte_index];
            ref += (int64_t)nibble_i4((n & 16u) ? byte >> 4 : byte) *
                   act16[b * K_BLOCK + k];
        }
        if (out16[b * N_TILE + n] != ref) {
            fprintf(stderr, "INT16 super mismatch block=%zu n=%zu got=%ld ref=%ld\n",
                    b, n, (long)out16[b * N_TILE + n], (long)ref);
            return -1;
        }
    }
    fused_fp4_i16_m1_k2_super_sve(packed_super, act16, out16);
    for (size_t b = 0; b < 2; ++b) for (size_t n = 0; n < N_TILE; ++n) {
        int64_t ref = 0;
        for (size_t k = 0; k < K_BLOCK; ++k) {
            size_t byte_index = (k / 4 * 2 + b) * 128 +
                                n / 32 * 64 + n % 16 * 4 + k % 4;
            uint8_t byte = packed_super[byte_index];
            ref += (int64_t)nibble_fp4((n & 16u) ? byte >> 4 : byte) *
                   act16[b * K_BLOCK + k];
        }
        if (out16[b * N_TILE + n] != ref) {
            fprintf(stderr, "FP4 INT16 mismatch block=%zu n=%zu got=%ld ref=%ld\n",
                    b, n, (long)out16[b * N_TILE + n], (long)ref);
            return -1;
        }
    }

    int32_t out16x8[2 * N_TILE] __attribute__((aligned(256)));
    for (int fp4 = 0; fp4 < 2; ++fp4) {
        if (fp4)
            fused_fp4_i16x8_m1_k2_super_sve(packed_super, act16x8, out16x8);
        else
            fused_i4_i16x8_m1_k2_super_sve(packed_super, act16x8, out16x8);
        for (size_t b = 0; b < 2; ++b) for (size_t n = 0; n < N_TILE; ++n) {
            int32_t ref = 0;
            for (size_t k = 0; k < K_BLOCK; ++k) {
                size_t byte_index = (k / 4 * 2 + b) * 128 +
                                    n / 32 * 64 + n % 16 * 4 + k % 4;
                uint8_t byte = packed_super[byte_index];
                int weight = fp4 ? nibble_fp4((n & 16u) ? byte >> 4 : byte) :
                                   nibble_i4((n & 16u) ? byte >> 4 : byte);
                ref += weight * act16[b * K_BLOCK + k];
            }
            if (out16x8[b * N_TILE + n] != ref) {
                fprintf(stderr, "%s INT16x8 mismatch block=%zu n=%zu got=%d ref=%d\n",
                        fp4 ? "FP4" : "INT4", b, n, out16x8[b * N_TILE + n], ref);
                return -1;
            }
        }
    }

    /* N-lane FP16 layout: four K=128 blocks are adjacent for every K scalar. */
    for (size_t k = 0; k < K_BLOCK; ++k)
        for (size_t b = 0; b < 4; ++b)
            for (size_t n = 0; n < N_TILE / 2; ++n)
                packed_super[(k * 4 + b) * 32 + n] =
                    (uint8_t)((k * 19 + b * 37 + n * 11) & 255u);
    fused_i4_f16_m1_k4_super_sve(packed_super, actf16, outf16);
    for (size_t b = 0; b < 4; ++b) for (size_t n = 0; n < N_TILE; ++n) {
        int ref = 0;
        for (size_t k = 0; k < K_BLOCK; ++k) {
            uint8_t byte = packed_super[(k * 4 + b) * 32 + n % 32];
            int weight = nibble_i4(n < 32 ? byte : byte >> 4);
            ref += weight * (actf16[b * K_BLOCK + k] < 0 ? -1 : 1);
        }
        if ((float)outf16[b * N_TILE + n] != (float)ref) {
            fprintf(stderr, "INT4 FP16 mismatch block=%zu n=%zu got=%g ref=%d\n",
                    b, n, (double)outf16[b * N_TILE + n], ref);
            return -1;
        }
    }
    fused_fp4_f16_m1_k4_super_sve(packed_super, actf16, outf16);
    for (size_t b = 0; b < 4; ++b) for (size_t n = 0; n < N_TILE; ++n) {
        int ref = 0;
        for (size_t k = 0; k < K_BLOCK; ++k) {
            uint8_t byte = packed_super[(k * 4 + b) * 32 + n % 32];
            int weight = nibble_fp4(n < 32 ? byte : byte >> 4);
            ref += weight * (actf16[b * K_BLOCK + k] < 0 ? -1 : 1);
        }
        if ((float)outf16[b * N_TILE + n] != (float)ref) {
            fprintf(stderr, "FP4 FP16 mismatch block=%zu n=%zu got=%g ref=%d\n",
                    b, n, (double)outf16[b * N_TILE + n], ref);
            return -1;
        }
    }
    puts("fused correctness: PASS (INT4 W4A8; INT4/FP4 W4A16 SDOT, radix-256 SDOT, and FP16 FMA)");
    return 0;
}

static void usage(const char *name)
{
    fprintf(stderr, "usage: %s [--format int4|fp4] [--path int8|int16|int16x8|int16x8-full|fp16] [--kernel shift|lut|pipe|super|opt|opt2] [--cores N] [--mib N] "
                    "[--iterations N] [--trials N] [--core-base N] [--skew-kib N] [--paired-baseline] [--compare-kernels] [--verify]\n", name);
}

int main(int argc, char **argv)
{
    path_kind path = PATH_I8;
    format_kind format = FORMAT_I4;
    kernel_kind kernel = KERNEL_SHIFT;
    int cores = 12, iterations = 10, trials = 3, core_base = 12;
    size_t mib = 240;
    size_t skew_kib = 0;
    int do_verify = 0, paired = 0, compare = 0;
    static const struct option options[] = {
        {"path", required_argument, NULL, 'p'}, {"format", required_argument, NULL, 'f'},
        {"cores", required_argument, NULL, 'c'},
        {"kernel", required_argument, NULL, 'k'},
        {"mib", required_argument, NULL, 'm'}, {"iterations", required_argument, NULL, 'i'},
        {"trials", required_argument, NULL, 't'}, {"core-base", required_argument, NULL, 'b'},
        {"skew-kib", required_argument, NULL, 's'},
        {"paired-baseline", no_argument, NULL, 'r'},
        {"compare-kernels", no_argument, NULL, 'a'},
        {"verify", no_argument, NULL, 'v'}, {NULL, 0, NULL, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "p:f:k:c:m:i:t:b:s:vra", options, NULL)) != -1) {
        switch (opt) {
        case 'p':
            if (!strcmp(optarg, "int8")) path = PATH_I8;
            else if (!strcmp(optarg, "int16")) path = PATH_I16;
            else if (!strcmp(optarg, "int16x8")) path = PATH_I16X8;
            else if (!strcmp(optarg, "int16x8-full")) path = PATH_FULL;
            else if (!strcmp(optarg, "fp16")) path = PATH_F16;
            else { usage(argv[0]); return 2; }
            break;
        case 'f':
            if (!strcmp(optarg, "int4")) format = FORMAT_I4;
            else if (!strcmp(optarg, "fp4")) format = FORMAT_FP4;
            else { usage(argv[0]); return 2; }
            break;
        case 'k':
            if (!strcmp(optarg, "shift")) kernel = KERNEL_SHIFT;
            else if (!strcmp(optarg, "lut")) kernel = KERNEL_LUT;
            else if (!strcmp(optarg, "pipe")) kernel = KERNEL_PIPE;
            else if (!strcmp(optarg, "super")) kernel = KERNEL_SUPER;
            else if (!strcmp(optarg, "opt")) kernel = KERNEL_OPT;
            else if (!strcmp(optarg, "opt2")) kernel = KERNEL_OPT2;
            else { usage(argv[0]); return 2; }
            break;
        case 'c': cores = atoi(optarg); break;
        case 'm': mib = strtoull(optarg, NULL, 0); break;
        case 'i': iterations = atoi(optarg); break;
        case 't': trials = atoi(optarg); break;
        case 'b': core_base = atoi(optarg); break;
        case 's': skew_kib = strtoull(optarg, NULL, 0); break;
        case 'v': do_verify = 1; break;
        case 'r': paired = 1; break;
        case 'a': compare = 1; break;
        default: usage(argv[0]); return 2;
        }
    }
    if (do_verify && (verify(kernel >= KERNEL_OPT ? KERNEL_SUPER : kernel) != 0 ||
                      verify_fused_opt() != 0)) return 1;
    if (cores < 1 || cores > MAX_CORES || iterations < 1 || trials < 1 || trials > 32) return 2;
    if (format == FORMAT_FP4 && path == PATH_I8) {
        fprintf(stderr, "FP4 requires --path int16 or fp16\n");
        return 2;
    }
    if (kernel >= KERNEL_OPT && path != PATH_F16 && path != PATH_FULL && path != PATH_I16) {
        fprintf(stderr, "opt kernels require int16, fp16 or int16x8-full\n");
        return 2;
    }
    if (kernel == KERNEL_OPT2 && path != PATH_F16) {
        fprintf(stderr, "opt2 requires fp16\n");
        return 2;
    }
    if (path == PATH_FULL && kernel != KERNEL_OPT) {
        fprintf(stderr, "int16x8-full requires --kernel opt\n");
        return 2;
    }
    if (compare && ((path != PATH_I16 && path != PATH_F16) || kernel < KERNEL_OPT)) {
        fprintf(stderr, "--compare-kernels requires optimized int16 or fp16\n");
        return 2;
    }
    if ((format == FORMAT_FP4 || path == PATH_F16 || path == PATH_I16X8) &&
        kernel < KERNEL_SUPER) {
        fprintf(stderr, "FP4, FP16, and INT16x8 paths require --kernel super\n");
        return 2;
    }
    size_t group = group_bytes(path);
    size_t skew = skew_kib * 1024u;
    if (skew % group != 0) {
        fprintf(stderr, "--skew-kib must preserve the %zu-byte kernel group alignment\n", group);
        return 2;
    }
    size_t weight_group = path == PATH_FULL ? 8192 : group;
    size_t packed_bytes = mib * 1024u * 1024u;
    packed_bytes -= packed_bytes % ((size_t)cores * weight_group);
    size_t bytes = packed_bytes / weight_group * group;
    if (!bytes) return 2;
    uint8_t *packed = NULL;
    size_t allocation_bytes = bytes + (size_t)(cores - 1) * skew;
    int pin_error = pin_cpu(core_base);
    if (pin_error) { fprintf(stderr, "affinity: %s\n", strerror(pin_error)); return 1; }
    if (posix_memalign((void **)&packed, 2u * 1024u * 1024u, allocation_bytes) != 0) return 1;
    (void)madvise(packed, allocation_bytes, MADV_HUGEPAGE);
    for (size_t i = 0; i < allocation_bytes; i += 256)
        for (size_t j = 0; j < 256; ++j) packed[i + j] = (uint8_t)(i + j * 13u);
    if (path == PATH_FULL)
        for (int c = 0; c < cores; ++c)
            for (size_t offset = 0; offset < bytes / (size_t)cores; offset += group)
                pack_weight_sums(packed + (size_t)c * (bytes / (size_t)cores + skew) + offset,
                                 format == FORMAT_FP4);
    report_fused_mapping(packed);
    printf("# packed_bytes=%zu metadata_bytes=%zu group_bytes=%zu\n",
           packed_bytes, bytes - packed_bytes, group);
    int8_t act8[4 * K_BLOCK] __attribute__((aligned(256)));
    int16_t act16[2 * K_BLOCK] __attribute__((aligned(256)));
    int8_t act16x8[4 * K_BLOCK] __attribute__((aligned(256)));
    _Float16 actf16[4 * K_BLOCK] __attribute__((aligned(256)));
    for (size_t i = 0; i < sizeof(act8); ++i) act8[i] = (int8_t)(i % 15u - 7);
    for (size_t i = 0; i < sizeof(act16) / sizeof(act16[0]); ++i) act16[i] = (int16_t)(i % 127u - 63);
    if (pack_i16x8(act16x8, act16, 2) != 0) return 1;
    if (path == PATH_FULL) {
        for (int i = 0; i < 256; ++i) act16[i] = (int16_t)(i * 257 - 32768);
        uint64_t pack_begin = cntvct();
        for (int i = 0; i < 10000; ++i) pack_i16x8_full(act16x8, act16, 2);
        printf("# activation_pack_ns=%.2f per_256_values\n",
               (double)(cntvct() - pack_begin) * 1e9 / (double)cntfrq() / 10000);
    }
    for (size_t i = 0; i < sizeof(actf16) / sizeof(actf16[0]); ++i)
        actf16[i] = (_Float16)((int)(i % 7u) - 3);
    double values[32], paired_read[2] = {0,0};
    for (int phase = paired ? 0 : 1; phase <= (paired ? 2 : 1); ++phase) {
        for (int reference = phase == 1 && compare; reference >= 0; --reference) {
            kernel_kind measured_kernel = reference ? KERNEL_SUPER : kernel;
            for (int run = 0; run <= trials; ++run) {
                pthread_t threads[MAX_CORES];
                worker workers[MAX_CORES];
                atomic_int ready, start;
                atomic_init(&ready, 0); atomic_init(&start, 0);
                size_t per_core = bytes / (size_t)cores;
                for (int c = 0; c < cores; ++c) {
                    workers[c] = (worker){ .packed = packed + (size_t)c * (per_core + skew),
                        .activation = path == PATH_I8 ? (const void *)act8 :
                                      path == PATH_I16 ? (const void *)act16 :
                                      path == PATH_I16X8 || path == PATH_FULL ? (const void *)act16x8 :
                                      (const void *)actf16,
                        .bytes = per_core, .iterations = iterations, .cpu = core_base + c,
                        .path = path, .format = format, .read_only = phase != 1,
                        .ready = &ready, .start = &start };
                    workers[c].kernel = measured_kernel;
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
                double bandwidth = (double)(phase == 1 ? packed_bytes : bytes) * iterations / seconds / 1e9;
                if (run > 0) values[run - 1] = bandwidth;
                printf("format=%s path=%s kernel=%s cores=%d skew_kib=%zu %s=%d %s=%.2f checksum=%lu\n",
                       format == FORMAT_I4 ? "int4" : "fp4",
                       phase == 1 ? path_name(path) : "paired-read",
                       reported_kernel_name(path, measured_kernel),
                       cores, skew_kib, run == 0 ? "warmup" : "trial", run == 0 ? 1 : run,
                       phase == 1 ? "packed_GB/s" : "read_GB/s",
                       bandwidth, (unsigned long)checksum);
            }
            for (int i = 0; i < trials; ++i)
                for (int j = i + 1; j < trials; ++j)
                    if (values[j] < values[i]) { double x = values[i]; values[i] = values[j]; values[j] = x; }
            double median = (values[(trials - 1) / 2] + values[trials / 2]) / 2;
            printf("summary format=%s path=%s kernel=%s cores=%d skew_kib=%zu %s=%.2f best=%.2f\n",
                   format == FORMAT_I4 ? "int4" : "fp4",
                   phase == 1 ? path_name(path) : "paired-read",
                   reported_kernel_name(path, measured_kernel),
                   cores, skew_kib, phase == 1 ? "packed_GB/s_median" : "read_GB/s_median",
                   median, values[trials - 1]);
            if (phase != 1) paired_read[phase / 2] = median;
        }
    }
    if (paired) printf("# paired_read_before=%.2f after=%.2f qualified=%s\n",
                       paired_read[0], paired_read[1],
                       paired_read[0] >= 220 && paired_read[1] >= 220 ? "yes" : "no");
    free(packed);
    return 0;
}
