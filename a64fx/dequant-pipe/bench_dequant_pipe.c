#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "hwb_compat.h"

enum {
    N_TILE = 64,
    K_BLOCK = 128,
    PACKED_BLOCK_BYTES = N_TILE * K_BLOCK / 2,
    MAX_PAIRS = 6,
    MAX_M = 6,
    M1_K_FUSE = 4,
};

typedef _Float16 fp16_t;

typedef enum { FORMAT_INT4, FORMAT_FP4 } quant_format;
typedef enum { PATH_W8A8, PATH_W8A16_I16, PATH_W8A16_AUTO, PATH_W8A16_F16 } arithmetic_path;
typedef enum { SYNC_HWBAR, SYNC_ATOMIC } sync_mode;

typedef struct {
    quant_format format;
    arithmetic_path requested_path;
    arithmetic_path path;
    sync_mode sync;
    int m;
    size_t n;
    size_t k;
    size_t chunk_bytes;
    int iterations;
    int trials;
    int core_base;
    int pairs;
    bool sweep_m;
    bool sweep_chunks;
    bool sweep_pairs;
    bool verify;
    bool self_test;
    bool peak;
    bool unsafe_scale;
} config;

extern void dequant_i4_i8_sve(const uint8_t *, int8_t *, size_t);
extern void dequant_fp4_i8_sve(const uint8_t *, int8_t *, size_t);
extern void dequant_i4_i16_sve(const uint8_t *, int16_t *, size_t);
extern void dequant_fp4_i16_sve(const uint8_t *, int16_t *, size_t);
extern void dequant_i4_f16_block_sve(const uint8_t *, fp16_t *, const fp16_t *);
extern void dequant_fp4_f16_block_sve(const uint8_t *, fp16_t *, const fp16_t *);
extern void gemm_i8_m6_stream_sve(const int8_t *, const int8_t *, int32_t *, size_t);
extern void gemm_i8_m1_k4_sve(const int8_t *, const int8_t *, int32_t *);

#define DECL_GEMM(kind, m, out_t) \
    extern void gemm_##kind##_m##m##_sve(const void *, const void *, out_t *)
DECL_GEMM(i8, 1, int32_t);
DECL_GEMM(i8, 2, int32_t);
DECL_GEMM(i8, 4, int32_t);
DECL_GEMM(i8, 6, int32_t);
DECL_GEMM(i16, 1, int64_t);
DECL_GEMM(i16, 2, int64_t);
DECL_GEMM(i16, 4, int64_t);
DECL_GEMM(i16, 6, int64_t);
DECL_GEMM(f16, 1, fp16_t);
DECL_GEMM(f16, 2, fp16_t);
DECL_GEMM(f16, 4, fp16_t);
DECL_GEMM(f16, 6, fp16_t);
#undef DECL_GEMM

typedef void (*gemm_i8_fn)(const void *, const void *, int32_t *);
typedef void (*gemm_i16_fn)(const void *, const void *, int64_t *);
typedef void (*gemm_f16_fn)(const void *, const void *, fp16_t *);

static inline uint64_t read_cntvct(void)
{
    uint64_t value;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(value));
    return value;
}

static inline uint64_t read_cntfrq(void)
{
    uint64_t value;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(value));
    return value;
}

static inline void shared_fence(void)
{
    __asm__ volatile("dmb ish" ::: "memory");
}

static const char *format_name(quant_format format)
{
    return format == FORMAT_INT4 ? "int4" : "e2m1-fp4";
}

static const char *path_name(arithmetic_path path)
{
    switch (path) {
    case PATH_W8A8: return "w8a8-sdot";
    case PATH_W8A16_I16: return "w8a16-int16-sdot";
    case PATH_W8A16_AUTO: return "w8a16-auto";
    case PATH_W8A16_F16: return "w8a16-fp16-fmla";
    }
    return "unknown";
}

static const char *sync_name(sync_mode mode)
{
    return mode == SYNC_HWBAR ? "hardware" : "atomic-control";
}

static void *aligned_zero_alloc(size_t bytes)
{
    void *ptr = NULL;
    size_t rounded = (bytes + 255u) & ~(size_t)255u;
    if (rounded == 0) rounded = 256;
    if (posix_memalign(&ptr, 256, rounded) != 0) return NULL;
    memset(ptr, 0, rounded);
    (void)madvise(ptr, rounded, MADV_HUGEPAGE);
    return ptr;
}

static int pin_to_cpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static bool cpu_is_available(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return false;
    return cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &set);
}

static unsigned read_cpu_khz(int cpu)
{
    char path[128];
    unsigned value = 0;
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE *fp = fopen(path, "r");
    if (fp != NULL) {
        if (fscanf(fp, "%u", &value) != 1) value = 0;
        fclose(fp);
    }
    return value;
}

static uint8_t weight_nibble(size_t n, size_t k, quant_format format)
{
    uint32_t x = (uint32_t)(n * 13u + k * 7u + (n >> 3) * 5u + (k >> 4));
    x ^= x >> 7;
    uint8_t code = (uint8_t)(x & 15u);
    if (format == FORMAT_FP4 && (code & 7u) == 0u && ((n + k) & 3u) != 0u)
        code ^= 1u;
    return code;
}

static int fp4_lattice(uint8_t code)
{
    static const int8_t table[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };
    return table[code & 15u];
}

static int integer_weight(uint8_t code, quant_format format)
{
    if (format == FORMAT_FP4) return fp4_lattice(code);
    return (int)(int8_t)(code << 4) >> 4;
}

static float weight_scale_value(size_t n, size_t kb, bool unsafe)
{
    if (unsafe && n == 0 && kb == 0) return 131072.0f;
    return 0.03125f * (1.0f + (float)((n + kb * 3u) % 7u) / 16.0f);
}

static float activation_scale_value(int row, size_t kb, arithmetic_path path)
{
    float base = path == PATH_W8A8 ? (1.0f / 32.0f) : (1.0f / 512.0f);
    return base * (1.0f + (float)((size_t)row + kb) / 64.0f);
}

static int activation_code(int row, size_t k, arithmetic_path path)
{
    uint32_t x = (uint32_t)(k * 11u + (size_t)row * 17u + (k >> 5));
    x ^= x >> 6;
    if (path == PATH_W8A8) return (int)(x % 15u) - 7;
    return (int)(x % 127u) - 63;
}

/*
 * The M=6 INT8 kernel streams one K quartet for all six rows at a time.  This
 * lets its assembly schedule overlap the next weight-vector loads with the
 * current 24 SDOT instructions without spending six pointer increments.
 */
static void pack_i8_m6_activation(int8_t *dst, size_t kb)
{
    for (size_t kg = 0; kg < K_BLOCK / 4; ++kg) {
        for (int row = 0; row < 6; ++row)
            for (size_t q = 0; q < 4; ++q)
                dst[kg * 24 + (size_t)row * 4 + q] =
                    (int8_t)activation_code(row, kb * K_BLOCK + kg * 4 + q,
                                            PATH_W8A8);
    }
}

static inline void put_nibble(uint8_t *dst, size_t index, uint8_t code)
{
    if ((index & 1u) == 0) dst[index >> 1] = code & 15u;
    else dst[index >> 1] |= (uint8_t)((code & 15u) << 4);
}

static void pack_dot_block(uint8_t *dst, size_t tile, size_t kb, quant_format format)
{
    size_t index = 0;
    memset(dst, 0, PACKED_BLOCK_BYTES);
    for (size_t kg = 0; kg < K_BLOCK / 4; ++kg) {
        for (size_t nv = 0; nv < 4; ++nv) {
            for (size_t lane = 0; lane < 16; ++lane) {
                for (size_t q = 0; q < 4; ++q) {
                    size_t n = tile * N_TILE + nv * 16 + lane;
                    size_t k = kb * K_BLOCK + kg * 4 + q;
                    put_nibble(dst, index++, weight_nibble(n, k, format));
                }
            }
        }
    }
}

static void pack_f16_block(uint8_t *dst, size_t tile, size_t kb, quant_format format)
{
    size_t index = 0;
    memset(dst, 0, PACKED_BLOCK_BYTES);
    for (size_t kk = 0; kk < K_BLOCK; ++kk) {
        for (size_t nn = 0; nn < N_TILE; ++nn) {
            size_t n = tile * N_TILE + nn;
            size_t k = kb * K_BLOCK + kk;
            put_nibble(dst, index++, weight_nibble(n, k, format));
        }
    }
}

typedef struct {
    uint8_t *packed;
    float *weight_scales;
    float *activation_scales;
    void *activations;
    float *output;
    size_t n_tiles;
    size_t k_blocks;
    size_t total_blocks;
    size_t packed_bytes;
    size_t activation_block_bytes;
} bench_data;

static void free_bench_data(bench_data *data)
{
    free(data->packed);
    free(data->weight_scales);
    free(data->activation_scales);
    free(data->activations);
    free(data->output);
    memset(data, 0, sizeof(*data));
}

static bool fp16_contract_safe(const config *cfg, const bench_data *data, const char **reason)
{
    const float min_subnormal = ldexpf(1.0f, -24);
    for (size_t block = 0; block < data->total_blocks; ++block) {
        size_t kb = block % data->k_blocks;
        const float *ws = data->weight_scales + block * N_TILE;
        for (size_t nn = 0; nn < N_TILE; ++nn) {
            float code_min = cfg->format == FORMAT_FP4 ? 0.5f : 1.0f;
            float code_max = cfg->format == FORMAT_FP4 ? 6.0f : 8.0f;
            float wmin = code_min * ws[nn];
            float wmax = code_max * ws[nn];
            fp16_t wh_min = (fp16_t)wmin;
            fp16_t wh_max = (fp16_t)wmax;
            if (!isfinite((float)wh_max) || (wmin != 0.0f && (float)wh_min == 0.0f)) {
                *reason = "weight conversion underflow/overflow";
                return false;
            }
            for (int row = 0; row < cfg->m; ++row) {
                float as = data->activation_scales[(size_t)row * data->k_blocks + kb];
                float amin = as;
                float amax = 63.0f * as;
                fp16_t ah_min = (fp16_t)amin;
                fp16_t ah_max = (fp16_t)amax;
                if (!isfinite((float)ah_max) || (float)ah_min == 0.0f) {
                    *reason = "activation conversion underflow/overflow";
                    return false;
                }
                if (wmin * amin < min_subnormal) {
                    *reason = "FP16 product can underflow";
                    return false;
                }
                if (wmax * amax * (float)K_BLOCK > 65504.0f) {
                    *reason = "FP16 block accumulator can overflow";
                    return false;
                }
            }
        }
    }
    return true;
}

static int prepare_data(config *cfg, bench_data *data)
{
    memset(data, 0, sizeof(*data));
    data->n_tiles = cfg->n / N_TILE;
    data->k_blocks = cfg->k / K_BLOCK;
    data->total_blocks = data->n_tiles * data->k_blocks;
    data->packed_bytes = data->total_blocks * PACKED_BLOCK_BYTES;
    data->packed = aligned_zero_alloc(data->packed_bytes);
    data->weight_scales = aligned_zero_alloc(data->total_blocks * N_TILE * sizeof(float));
    data->activation_scales = aligned_zero_alloc((size_t)cfg->m * data->k_blocks * sizeof(float));
    data->output = aligned_zero_alloc((size_t)cfg->m * cfg->n * sizeof(float));
    if (!data->packed || !data->weight_scales || !data->activation_scales || !data->output)
        goto oom;

    for (size_t tile = 0; tile < data->n_tiles; ++tile) {
        for (size_t kb = 0; kb < data->k_blocks; ++kb) {
            size_t block = tile * data->k_blocks + kb;
            for (size_t nn = 0; nn < N_TILE; ++nn) {
                size_t n = tile * N_TILE + nn;
                data->weight_scales[block * N_TILE + nn] =
                    weight_scale_value(n, kb, cfg->unsafe_scale);
            }
        }
    }
    for (int row = 0; row < cfg->m; ++row)
        for (size_t kb = 0; kb < data->k_blocks; ++kb)
            data->activation_scales[(size_t)row * data->k_blocks + kb] =
                activation_scale_value(row, kb, cfg->requested_path);

    if (cfg->requested_path == PATH_W8A16_AUTO || cfg->requested_path == PATH_W8A16_F16) {
        const char *reason = NULL;
        bool safe = fp16_contract_safe(cfg, data, &reason);
        if (!safe && cfg->requested_path == PATH_W8A16_F16) {
            fprintf(stderr, "FP16 path rejected: %s\n", reason);
            return -1;
        }
        cfg->path = safe ? PATH_W8A16_F16 : PATH_W8A16_I16;
        if (!safe)
            printf("FP16 safety gate: %s; using INT16 SDOT for this run\n", reason);
    } else {
        cfg->path = cfg->requested_path;
    }

    bool f16_layout = cfg->path == PATH_W8A16_F16;
    for (size_t tile = 0; tile < data->n_tiles; ++tile) {
        for (size_t kb = 0; kb < data->k_blocks; ++kb) {
            uint8_t *block = data->packed + (tile * data->k_blocks + kb) * PACKED_BLOCK_BYTES;
            if (f16_layout) pack_f16_block(block, tile, kb, cfg->format);
            else pack_dot_block(block, tile, kb, cfg->format);
        }
    }

    size_t elem_size = cfg->path == PATH_W8A8 ? sizeof(int8_t) : sizeof(int16_t);
    data->activation_block_bytes = (size_t)cfg->m * K_BLOCK * elem_size;
    data->activations = aligned_zero_alloc(data->k_blocks * data->activation_block_bytes);
    if (!data->activations) goto oom;
    for (size_t kb = 0; kb < data->k_blocks; ++kb) {
        if (cfg->path == PATH_W8A8 && cfg->m == 6) {
            pack_i8_m6_activation((int8_t *)data->activations +
                                   kb * data->activation_block_bytes, kb);
            continue;
        }
        for (int row = 0; row < cfg->m; ++row) {
            float as = data->activation_scales[(size_t)row * data->k_blocks + kb];
            for (size_t kk = 0; kk < K_BLOCK; ++kk) {
                size_t index = (kb * (size_t)cfg->m + (size_t)row) * K_BLOCK + kk;
                int code = activation_code(row, kb * K_BLOCK + kk, cfg->path);
                if (cfg->path == PATH_W8A8)
                    ((int8_t *)data->activations)[index] = (int8_t)code;
                else if (cfg->path == PATH_W8A16_I16)
                    ((int16_t *)data->activations)[index] = (int16_t)code;
                else
                    ((fp16_t *)data->activations)[index] = (fp16_t)((float)code * as);
            }
        }
    }
    return 0;

oom:
    fprintf(stderr, "allocation failed while preparing %.1f MiB of packed weights\n",
            (double)data->packed_bytes / 1048576.0);
    free_bench_data(data);
    return -1;
}

typedef struct {
    alignas(256) atomic_uint arrivals;
    atomic_uint generation;
    unsigned char padding[256 - 2 * sizeof(atomic_uint)];
} soft_pair_barrier;

typedef struct {
    uint8_t *weights;
    float *scales;
    size_t first_block;
    size_t blocks;
} ring_slot;

struct run_context;

typedef struct {
    struct run_context *run;
    int pair;
    bool producer;
    int cpu;
    int barrier_descriptor;
    long barrier_window;
    soft_pair_barrier soft;
    ring_slot slots[2];
    uint64_t start_tick;
    uint64_t end_tick;
    uint64_t barrier_ticks;
    double checksum;
    int error;
} worker_context;

typedef struct run_context {
    config *cfg;
    bench_data *data;
    worker_context workers[MAX_PAIRS * 2];
    atomic_int ready;
    atomic_int start;
    atomic_int setup_error;
    size_t blocks_per_chunk;
    size_t expanded_block_bytes;
    size_t pair_blocks;
    size_t chunks_per_pair;
} run_context;

static void soft_barrier_wait(soft_pair_barrier *barrier)
{
    unsigned generation = atomic_load_explicit(&barrier->generation, memory_order_acquire);
    if (atomic_fetch_add_explicit(&barrier->arrivals, 1, memory_order_acq_rel) == 1) {
        atomic_store_explicit(&barrier->arrivals, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&barrier->generation, 1, memory_order_release);
    } else {
        while (atomic_load_explicit(&barrier->generation, memory_order_acquire) == generation)
            __asm__ volatile("yield");
    }
}

static void pair_barrier(worker_context *worker)
{
    uint64_t start = read_cntvct();
    shared_fence();
    if (worker->run->cfg->sync == SYNC_HWBAR)
        vhbm_bar(worker->barrier_window);
    else
        soft_barrier_wait(&worker->run->workers[worker->pair * 2].soft);
    shared_fence();
    worker->barrier_ticks += read_cntvct() - start;
}

static gemm_i8_fn select_i8_kernel(int m)
{
    switch (m) {
    case 1: return gemm_i8_m1_sve;
    case 2: return gemm_i8_m2_sve;
    case 4: return gemm_i8_m4_sve;
    case 6: return gemm_i8_m6_sve;
    default: return NULL;
    }
}

static gemm_i16_fn select_i16_kernel(int m)
{
    switch (m) {
    case 1: return gemm_i16_m1_sve;
    case 2: return gemm_i16_m2_sve;
    case 4: return gemm_i16_m4_sve;
    case 6: return gemm_i16_m6_sve;
    default: return NULL;
    }
}

static gemm_f16_fn select_f16_kernel(int m)
{
    switch (m) {
    case 1: return gemm_f16_m1_sve;
    case 2: return gemm_f16_m2_sve;
    case 4: return gemm_f16_m4_sve;
    case 6: return gemm_f16_m6_sve;
    default: return NULL;
    }
}

static void producer_fill(worker_context *worker, ring_slot *slot,
                          size_t first_block, size_t blocks)
{
    run_context *run = worker->run;
    config *cfg = run->cfg;
    bench_data *data = run->data;
    const uint8_t *src = data->packed + first_block * PACKED_BLOCK_BYTES;
    slot->first_block = first_block;
    slot->blocks = blocks;

    if (cfg->path == PATH_W8A8) {
        size_t bytes = blocks * PACKED_BLOCK_BYTES;
        if (cfg->format == FORMAT_INT4)
            dequant_i4_i8_sve(src, (int8_t *)slot->weights, bytes);
        else
            dequant_fp4_i8_sve(src, (int8_t *)slot->weights, bytes);
        for (size_t b = 0; b < blocks; ++b) {
            float factor = cfg->format == FORMAT_FP4 ? 0.5f : 1.0f;
            const float *input = data->weight_scales + (first_block + b) * N_TILE;
            float *output = slot->scales + b * N_TILE;
            for (size_t n = 0; n < N_TILE; ++n) output[n] = input[n] * factor;
        }
    } else if (cfg->path == PATH_W8A16_I16) {
        size_t bytes = blocks * PACKED_BLOCK_BYTES;
        if (cfg->format == FORMAT_INT4)
            dequant_i4_i16_sve(src, (int16_t *)slot->weights, bytes);
        else
            dequant_fp4_i16_sve(src, (int16_t *)slot->weights, bytes);
        for (size_t b = 0; b < blocks; ++b) {
            float factor = cfg->format == FORMAT_FP4 ? 0.5f : 1.0f;
            const float *input = data->weight_scales + (first_block + b) * N_TILE;
            float *output = slot->scales + b * N_TILE;
            for (size_t n = 0; n < N_TILE; ++n) output[n] = input[n] * factor;
        }
    } else {
        for (size_t b = 0; b < blocks; ++b) {
            fp16_t scales[N_TILE] __attribute__((aligned(256)));
            float factor = cfg->format == FORMAT_FP4 ? 0.5f : 1.0f;
            const float *input = data->weight_scales + (first_block + b) * N_TILE;
            for (size_t n = 0; n < N_TILE; ++n) scales[n] = (fp16_t)(input[n] * factor);
            fp16_t *output = (fp16_t *)(slot->weights + b * run->expanded_block_bytes);
            const uint8_t *input_codes = src + b * PACKED_BLOCK_BYTES;
            if (cfg->format == FORMAT_INT4)
                dequant_i4_f16_block_sve(input_codes, output, scales);
            else
                dequant_fp4_f16_block_sve(input_codes, output, scales);
        }
    }
}

static void consume_i8_block(worker_context *worker, const ring_slot *slot,
                             size_t local_block, int32_t *partial)
{
    run_context *run = worker->run;
    config *cfg = run->cfg;
    bench_data *data = run->data;
    size_t block = slot->first_block + local_block;
    size_t tile = block / data->k_blocks;
    size_t kb = block % data->k_blocks;
    const int8_t *weights = (const int8_t *)(slot->weights + local_block * run->expanded_block_bytes);
    const int8_t *activation = (const int8_t *)data->activations +
                               kb * data->activation_block_bytes;
    select_i8_kernel(cfg->m)(weights, activation, partial);
    const float *ws = slot->scales + local_block * N_TILE;
    for (int row = 0; row < cfg->m; ++row) {
        float scale = data->activation_scales[(size_t)row * data->k_blocks + kb];
        float *out = data->output + (size_t)row * cfg->n + tile * N_TILE;
        const int32_t *part = partial + (size_t)row * N_TILE;
        for (size_t n = 0; n < N_TILE; ++n) out[n] += (float)part[n] * ws[n] * scale;
    }
}

static void consume_i8_m1_k4(worker_context *worker, const ring_slot *slot,
                             size_t local_block, int32_t *partial)
{
    run_context *run = worker->run;
    bench_data *data = run->data;
    size_t block = slot->first_block + local_block;
    size_t tile = block / data->k_blocks;
    size_t first_kb = block % data->k_blocks;
    const int8_t *weights = (const int8_t *)(slot->weights +
                             local_block * run->expanded_block_bytes);
    const int8_t *activation = (const int8_t *)data->activations +
                               first_kb * data->activation_block_bytes;
    gemm_i8_m1_k4_sve(weights, activation, partial);
    float *out = data->output + tile * N_TILE;
    for (size_t fused = 0; fused < M1_K_FUSE; ++fused) {
        size_t kb = first_kb + fused;
        const float *ws = slot->scales + (local_block + fused) * N_TILE;
        const int32_t *part = partial + fused * N_TILE;
        float scale = data->activation_scales[kb];
        for (size_t n = 0; n < N_TILE; ++n)
            out[n] += (float)part[n] * ws[n] * scale;
    }
}

static void consume_i16_block(worker_context *worker, const ring_slot *slot,
                              size_t local_block, int64_t *partial)
{
    run_context *run = worker->run;
    config *cfg = run->cfg;
    bench_data *data = run->data;
    size_t block = slot->first_block + local_block;
    size_t tile = block / data->k_blocks;
    size_t kb = block % data->k_blocks;
    const uint8_t *weights = slot->weights + local_block * run->expanded_block_bytes;
    const int16_t *activation = (const int16_t *)data->activations + kb * (size_t)cfg->m * K_BLOCK;
    gemm_i16_fn kernel = select_i16_kernel(cfg->m);
    kernel(weights, activation, partial);
    kernel(weights + 256, activation, partial + (size_t)cfg->m * 32);
    const float *ws = slot->scales + local_block * N_TILE;
    for (int row = 0; row < cfg->m; ++row) {
        float scale = data->activation_scales[(size_t)row * data->k_blocks + kb];
        float *out = data->output + (size_t)row * cfg->n + tile * N_TILE;
        const int64_t *lo = partial + (size_t)row * 32;
        const int64_t *hi = partial + (size_t)cfg->m * 32 + (size_t)row * 32;
        for (size_t n = 0; n < 32; ++n) {
            out[n] += (float)lo[n] * ws[n] * scale;
            out[n + 32] += (float)hi[n] * ws[n + 32] * scale;
        }
    }
}

static void consume_f16_block(worker_context *worker, const ring_slot *slot,
                              size_t local_block, fp16_t *partial)
{
    run_context *run = worker->run;
    config *cfg = run->cfg;
    bench_data *data = run->data;
    size_t block = slot->first_block + local_block;
    size_t tile = block / data->k_blocks;
    size_t kb = block % data->k_blocks;
    const fp16_t *weights = (const fp16_t *)(slot->weights + local_block * run->expanded_block_bytes);
    const fp16_t *activation = (const fp16_t *)data->activations + kb * (size_t)cfg->m * K_BLOCK;
    select_f16_kernel(cfg->m)(weights, activation, partial);
    for (int row = 0; row < cfg->m; ++row) {
        float *out = data->output + (size_t)row * cfg->n + tile * N_TILE;
        const fp16_t *part = partial + (size_t)row * N_TILE;
        for (size_t n = 0; n < N_TILE; ++n) out[n] += (float)part[n];
    }
}

static void *worker_main(void *opaque)
{
    worker_context *worker = opaque;
    run_context *run = worker->run;
    config *cfg = run->cfg;
    if (pin_to_cpu(worker->cpu) != 0) {
        worker->error = errno ? errno : EINVAL;
        atomic_store(&run->setup_error, 1);
    }
    if (cfg->sync == SYNC_HWBAR && worker->error == 0) {
        long requested = -1;
        worker->barrier_window = vhbm_bar_assign(worker->barrier_descriptor, &requested);
        if (worker->barrier_window < 0) {
            worker->error = (int)-worker->barrier_window;
            atomic_store(&run->setup_error, 1);
        }
    }

    int32_t *partial_i8 = NULL;
    int64_t *partial_i16 = NULL;
    fp16_t *partial_f16 = NULL;
    if (!worker->producer) {
        if (cfg->path == PATH_W8A8)
            partial_i8 = aligned_zero_alloc((size_t)(cfg->m == 1 ? M1_K_FUSE : cfg->m) *
                                            N_TILE * sizeof(*partial_i8));
        else if (cfg->path == PATH_W8A16_I16)
            partial_i16 = aligned_zero_alloc((size_t)cfg->m * N_TILE * sizeof(*partial_i16));
        else
            partial_f16 = aligned_zero_alloc((size_t)cfg->m * N_TILE * sizeof(*partial_f16));
        if (!partial_i8 && !partial_i16 && !partial_f16) {
            worker->error = ENOMEM;
            atomic_store(&run->setup_error, 1);
        }
    }

    atomic_fetch_add(&run->ready, 1);
    while (!atomic_load_explicit(&run->start, memory_order_acquire)) __asm__ volatile("yield");
    worker->start_tick = read_cntvct();

    if (worker->error == 0 && !atomic_load(&run->setup_error)) {
        size_t total_chunks = run->chunks_per_pair * (size_t)cfg->iterations;
        size_t pair_first = (size_t)worker->pair * run->pair_blocks;
        for (size_t sequence = 0; sequence < total_chunks; ++sequence) {
            size_t chunk = sequence % run->chunks_per_pair;
            size_t offset = chunk * run->blocks_per_chunk;
            size_t blocks = run->pair_blocks - offset;
            if (blocks > run->blocks_per_chunk) blocks = run->blocks_per_chunk;
            ring_slot *slot = &run->workers[worker->pair * 2].slots[sequence & 1u];
            if (worker->producer)
                producer_fill(worker, slot, pair_first + offset, blocks);
            pair_barrier(worker);
            if (!worker->producer) {
                for (size_t b = 0; b < slot->blocks;) {
                    size_t global_block = slot->first_block + b;
                    size_t kb = global_block % run->data->k_blocks;
                    if (cfg->path == PATH_W8A8 && cfg->m == 1 &&
                        b + M1_K_FUSE <= slot->blocks &&
                        kb + M1_K_FUSE <= run->data->k_blocks) {
                        consume_i8_m1_k4(worker, slot, b, partial_i8);
                        b += M1_K_FUSE;
                    } else if (cfg->path == PATH_W8A8) {
                        consume_i8_block(worker, slot, b++, partial_i8);
                    } else if (cfg->path == PATH_W8A16_I16) {
                        consume_i16_block(worker, slot, b, partial_i16);
                        ++b;
                    } else {
                        consume_f16_block(worker, slot, b, partial_f16);
                        ++b;
                    }
                }
            }
        }
    }
    worker->end_tick = read_cntvct();

    if (!worker->producer) {
        size_t tile_first = ((size_t)worker->pair * run->pair_blocks) / run->data->k_blocks;
        size_t tile_count = run->pair_blocks / run->data->k_blocks;
        for (int row = 0; row < cfg->m; ++row)
            for (size_t n = tile_first * N_TILE; n < (tile_first + tile_count) * N_TILE; ++n)
                worker->checksum += run->data->output[(size_t)row * cfg->n + n];
    }
    free(partial_i8);
    free(partial_i16);
    free(partial_f16);
    if (cfg->sync == SYNC_HWBAR && worker->barrier_window >= 0) {
        int rc = vhbm_bar_unassign(worker->barrier_descriptor);
        if (rc != 0 && worker->error == 0) worker->error = rc < 0 ? -rc : rc;
    }
    return NULL;
}

typedef struct {
    double seconds;
    double compressed_gbs;
    double expanded_gbs;
    double logical_gweights;
    double gops;
    double barrier_ns;
    double barrier_percent;
    double checksum;
    unsigned cpu_khz_before;
    unsigned cpu_khz_after;
} run_result;

static void release_ring(run_context *run)
{
    for (int pair = 0; pair < run->cfg->pairs; ++pair) {
        worker_context *producer = &run->workers[pair * 2];
        for (int slot = 0; slot < 2; ++slot) {
            free(producer->slots[slot].weights);
            free(producer->slots[slot].scales);
            producer->slots[slot].weights = NULL;
            producer->slots[slot].scales = NULL;
        }
    }
}

static int allocate_ring(run_context *run)
{
    size_t weight_bytes = run->blocks_per_chunk * run->expanded_block_bytes;
    size_t scale_bytes = run->blocks_per_chunk * N_TILE * sizeof(float);
    for (int pair = 0; pair < run->cfg->pairs; ++pair) {
        worker_context *producer = &run->workers[pair * 2];
        for (int slot = 0; slot < 2; ++slot) {
            producer->slots[slot].weights = aligned_zero_alloc(weight_bytes);
            producer->slots[slot].scales = aligned_zero_alloc(scale_bytes);
            if (!producer->slots[slot].weights || !producer->slots[slot].scales) {
                release_ring(run);
                return -1;
            }
        }
    }
    return 0;
}

static int init_hardware_barriers(run_context *run)
{
    if (access("/dev/xos_hwb", R_OK | W_OK) != 0) {
        fprintf(stderr,
                "hardware barrier device /dev/xos_hwb is unavailable: %s\n"
                "headline mode requires six real barrier blades; use --sync atomic only as a control\n",
                strerror(errno));
        return -1;
    }
    for (int pair = 0; pair < run->cfg->pairs; ++pair) {
        int producer_cpu = run->cfg->core_base + pair;
        int consumer_cpu = run->cfg->core_base + MAX_PAIRS + pair;
        unsigned long mask = (1UL << producer_cpu) | (1UL << consumer_cpu);
        int descriptor = vhbm_bar_init(mask);
        if (descriptor <= 0) {
            int error = descriptor < 0 ? -descriptor : EIO;
            fprintf(stderr,
                    "hardware barrier allocation failed for CPUs %d,%d: rc=%d (%s)\n"
                    "headline mode requires six real barrier blades; use --sync atomic only as a control\n",
                    producer_cpu, consumer_cpu, descriptor, strerror(error));
            for (int previous = 0; previous < pair; ++previous)
                (void)vhbm_bar_fini(run->workers[previous * 2].barrier_descriptor);
            return -1;
        }
        run->workers[pair * 2].barrier_descriptor = descriptor;
        run->workers[pair * 2 + 1].barrier_descriptor = descriptor;
    }
    return 0;
}

static void fini_hardware_barriers(run_context *run)
{
    for (int pair = 0; pair < run->cfg->pairs; ++pair) {
        int rc = vhbm_bar_fini(run->workers[pair * 2].barrier_descriptor);
        if (rc != 0)
            fprintf(stderr, "warning: hardware barrier %d cleanup returned %d\n", pair, rc);
    }
}

static int run_trial(config *cfg, bench_data *data, run_result *result)
{
    run_context run;
    pthread_t threads[MAX_PAIRS * 2];
    memset(&run, 0, sizeof(run));
    run.cfg = cfg;
    run.data = data;
    run.blocks_per_chunk = cfg->chunk_bytes / PACKED_BLOCK_BYTES;
    if (run.blocks_per_chunk == 0) run.blocks_per_chunk = 1;
    run.expanded_block_bytes = cfg->path == PATH_W8A8 ?
        N_TILE * K_BLOCK : N_TILE * K_BLOCK * sizeof(int16_t);
    run.pair_blocks = data->total_blocks / (size_t)cfg->pairs;
    run.chunks_per_pair = (run.pair_blocks + run.blocks_per_chunk - 1) / run.blocks_per_chunk;
    atomic_init(&run.ready, 0);
    atomic_init(&run.start, 0);
    atomic_init(&run.setup_error, 0);
    memset(data->output, 0, (size_t)cfg->m * cfg->n * sizeof(float));

    for (int pair = 0; pair < cfg->pairs; ++pair) {
        worker_context *producer = &run.workers[pair * 2];
        worker_context *consumer = &run.workers[pair * 2 + 1];
        producer->run = &run;
        producer->pair = pair;
        producer->producer = true;
        producer->cpu = cfg->core_base + pair;
        producer->barrier_window = -1;
        atomic_init(&producer->soft.arrivals, 0);
        atomic_init(&producer->soft.generation, 0);
        consumer->run = &run;
        consumer->pair = pair;
        consumer->producer = false;
        consumer->cpu = cfg->core_base + MAX_PAIRS + pair;
        consumer->barrier_window = -1;
    }
    if (allocate_ring(&run) != 0) {
        fprintf(stderr, "ring allocation failed\n");
        return -1;
    }
    if (cfg->sync == SYNC_HWBAR && init_hardware_barriers(&run) != 0) {
        release_ring(&run);
        return -1;
    }

    result->cpu_khz_before = read_cpu_khz(cfg->core_base);
    int created = 0;
    for (int index = 0; index < cfg->pairs * 2; ++index) {
        int rc = pthread_create(&threads[index], NULL, worker_main, &run.workers[index]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create failed: %s\n", strerror(rc));
            atomic_store(&run.setup_error, 1);
            break;
        }
        ++created;
    }
    while (atomic_load(&run.ready) < created) __asm__ volatile("yield");
    atomic_store_explicit(&run.start, 1, memory_order_release);
    for (int index = 0; index < created; ++index) pthread_join(threads[index], NULL);
    result->cpu_khz_after = read_cpu_khz(cfg->core_base);

    int failed = created != cfg->pairs * 2 || atomic_load(&run.setup_error);
    uint64_t first = UINT64_MAX, last = 0, barrier_sum = 0;
    double checksum = 0.0;
    for (int index = 0; index < cfg->pairs * 2; ++index) {
        worker_context *worker = &run.workers[index];
        if (worker->start_tick < first) first = worker->start_tick;
        if (worker->end_tick > last) last = worker->end_tick;
        barrier_sum += worker->barrier_ticks;
        checksum += worker->checksum;
        if (worker->error != 0) {
            fprintf(stderr, "worker cpu %d failed: %s\n", worker->cpu, strerror(worker->error));
            failed = 1;
        }
    }

    if (cfg->sync == SYNC_HWBAR) fini_hardware_barriers(&run);
    if (!failed && last > first) {
        double timer_hz = (double)read_cntfrq();
        double seconds = (double)(last - first) / timer_hz;
        double compressed = (double)data->packed_bytes * cfg->iterations;
        double expanded = (double)data->total_blocks * run.expanded_block_bytes * cfg->iterations;
        double weights = (double)cfg->n * (double)cfg->k * cfg->iterations;
        double operations = 2.0 * weights * cfg->m;
        size_t syncs_per_worker = run.chunks_per_pair * (size_t)cfg->iterations;
        result->seconds = seconds;
        result->compressed_gbs = compressed / seconds / 1.0e9;
        result->expanded_gbs = expanded / seconds / 1.0e9;
        result->logical_gweights = weights / seconds / 1.0e9;
        result->gops = operations / seconds / 1.0e9;
        result->barrier_ns = syncs_per_worker ?
            ((double)barrier_sum / (cfg->pairs * 2) / (double)syncs_per_worker) / timer_hz * 1.0e9 : 0.0;
        result->barrier_percent = seconds > 0.0 ?
            ((double)barrier_sum / (cfg->pairs * 2) / timer_hz) / seconds * 100.0 : 0.0;
        result->checksum = checksum;
    }
    release_ring(&run);
    return failed ? -1 : 0;
}

static int verify_output(const config *cfg, const bench_data *data)
{
    double max_abs = 0.0, max_rel = 0.0;
    size_t worst_row = 0, worst_n = 0;
    for (int row = 0; row < cfg->m; ++row) {
        for (size_t n = 0; n < cfg->n; ++n) {
            float expected = 0.0f;
            for (size_t kb = 0; kb < data->k_blocks; ++kb) {
                float ws = weight_scale_value(n, kb, cfg->unsafe_scale);
                float as = data->activation_scales[(size_t)row * data->k_blocks + kb];
                if (cfg->path == PATH_W8A16_F16) {
                    float factor = cfg->format == FORMAT_FP4 ? 0.5f : 1.0f;
                    fp16_t hs = (fp16_t)(ws * factor);
                    fp16_t acc = (fp16_t)0.0f;
                    for (size_t kk = 0; kk < K_BLOCK; ++kk) {
                        size_t k = kb * K_BLOCK + kk;
                        fp16_t w = (fp16_t)((float)integer_weight(weight_nibble(n, k, cfg->format),
                                                                  cfg->format) * (float)hs);
                        fp16_t a = (fp16_t)((float)activation_code(row, k, cfg->path) * as);
                        acc = (fp16_t)((float)acc + (float)w * (float)a);
                    }
                    expected += (float)acc;
                } else {
                    int64_t dot = 0;
                    for (size_t kk = 0; kk < K_BLOCK; ++kk) {
                        size_t k = kb * K_BLOCK + kk;
                        dot += (int64_t)integer_weight(weight_nibble(n, k, cfg->format), cfg->format) *
                               activation_code(row, k, cfg->path);
                    }
                    float factor = cfg->format == FORMAT_FP4 ? 0.5f : 1.0f;
                    expected += (float)dot * ws * factor * as;
                }
            }
            expected *= (float)cfg->iterations;
            float actual = data->output[(size_t)row * cfg->n + n];
            double abs_error = fabs((double)actual - expected);
            double rel_error = abs_error / fmax(1.0e-6, fabs((double)expected));
            if (abs_error > max_abs) {
                max_abs = abs_error;
                worst_row = (size_t)row;
                worst_n = n;
            }
            if (rel_error > max_rel) max_rel = rel_error;
        }
    }
    double abs_limit = cfg->path == PATH_W8A16_F16 ? 0.5 : 1.0e-4;
    double rel_limit = cfg->path == PATH_W8A16_F16 ? 0.03 : 1.0e-5;
    printf("verification: max_abs=%.6g max_rel=%.6g worst=(row=%zu,n=%zu) %s\n",
           max_abs, max_rel, worst_row, worst_n,
           (max_abs <= abs_limit || max_rel <= rel_limit) ? "PASS" : "FAIL");
    return (max_abs <= abs_limit || max_rel <= rel_limit) ? 0 : -1;
}

static int test_dequant(void)
{
    uint8_t packed[64] __attribute__((aligned(256)));
    int8_t out8[128] __attribute__((aligned(256)));
    int16_t out16[128] __attribute__((aligned(256)));
    for (size_t i = 0; i < 128; ++i) put_nibble(packed, i, (uint8_t)(i & 15u));
    dequant_i4_i8_sve(packed, out8, sizeof(packed));
    for (size_t i = 0; i < 128; ++i)
        if (out8[i] != integer_weight((uint8_t)(i & 15u), FORMAT_INT4)) return -1;
    dequant_fp4_i8_sve(packed, out8, sizeof(packed));
    for (size_t i = 0; i < 128; ++i)
        if (out8[i] != integer_weight((uint8_t)(i & 15u), FORMAT_FP4)) return -1;
    dequant_i4_i16_sve(packed, out16, sizeof(packed));
    for (size_t i = 0; i < 128; ++i)
        if (out16[i] != integer_weight((uint8_t)(i & 15u), FORMAT_INT4)) return -1;
    dequant_fp4_i16_sve(packed, out16, sizeof(packed));
    for (size_t i = 0; i < 128; ++i)
        if (out16[i] != integer_weight((uint8_t)(i & 15u), FORMAT_FP4)) return -1;
    return 0;
}

static int test_integer_kernels(quant_format format, int m)
{
    uint8_t packed[PACKED_BLOCK_BYTES] __attribute__((aligned(256)));
    int8_t weights8[N_TILE * K_BLOCK] __attribute__((aligned(256)));
    int16_t weights16[N_TILE * K_BLOCK] __attribute__((aligned(256)));
    int8_t act8[MAX_M * K_BLOCK] __attribute__((aligned(256)));
    int8_t act8_m6[MAX_M * K_BLOCK] __attribute__((aligned(256)));
    int16_t act16[MAX_M * K_BLOCK] __attribute__((aligned(256)));
    int32_t out8[MAX_M * N_TILE] __attribute__((aligned(256)));
    int64_t out16[MAX_M * N_TILE] __attribute__((aligned(256)));
    pack_dot_block(packed, 0, 0, format);
    if (format == FORMAT_INT4) {
        dequant_i4_i8_sve(packed, weights8, sizeof(packed));
        dequant_i4_i16_sve(packed, weights16, sizeof(packed));
    } else {
        dequant_fp4_i8_sve(packed, weights8, sizeof(packed));
        dequant_fp4_i16_sve(packed, weights16, sizeof(packed));
    }
    for (int row = 0; row < m; ++row) {
        for (size_t k = 0; k < K_BLOCK; ++k) {
            act8[(size_t)row * K_BLOCK + k] = (int8_t)activation_code(row, k, PATH_W8A8);
            act16[(size_t)row * K_BLOCK + k] = (int16_t)activation_code(row, k, PATH_W8A16_I16);
        }
    }
    if (m == 6) pack_i8_m6_activation(act8_m6, 0);
    select_i8_kernel(m)(weights8, m == 6 ? act8_m6 : act8, out8);
    gemm_i16_fn i16_kernel = select_i16_kernel(m);
    i16_kernel(weights16, act16, out16);
    i16_kernel((const uint8_t *)weights16 + 256, act16, out16 + (size_t)m * 32);
    for (int row = 0; row < m; ++row) {
        for (size_t n = 0; n < N_TILE; ++n) {
            int64_t ref8 = 0, ref16 = 0;
            for (size_t k = 0; k < K_BLOCK; ++k) {
                int weight = integer_weight(weight_nibble(n, k, format), format);
                ref8 += (int64_t)weight * act8[(size_t)row * K_BLOCK + k];
                ref16 += (int64_t)weight * act16[(size_t)row * K_BLOCK + k];
            }
            size_t i16_index = n < 32 ? (size_t)row * 32 + n :
                (size_t)m * 32 + (size_t)row * 32 + n - 32;
            if (out8[(size_t)row * N_TILE + n] != ref8 || out16[i16_index] != ref16) {
                fprintf(stderr,
                        "kernel mismatch format=%s M=%d row=%d n=%zu: i8=%d/%ld i16=%ld/%ld\n",
                        format_name(format), m, row, n,
                        out8[(size_t)row * N_TILE + n], (long)ref8,
                        (long)out16[i16_index], (long)ref16);
                return -1;
            }
        }
    }
    return 0;
}

static int test_f16_kernel(quant_format format, int m)
{
    uint8_t packed[PACKED_BLOCK_BYTES] __attribute__((aligned(256)));
    fp16_t weights[N_TILE * K_BLOCK] __attribute__((aligned(256)));
    fp16_t scales[N_TILE] __attribute__((aligned(256)));
    fp16_t activation[MAX_M * K_BLOCK] __attribute__((aligned(256)));
    fp16_t output[MAX_M * N_TILE] __attribute__((aligned(256)));
    pack_f16_block(packed, 0, 0, format);
    float factor = format == FORMAT_FP4 ? 0.5f : 1.0f;
    for (size_t n = 0; n < N_TILE; ++n)
        scales[n] = (fp16_t)(weight_scale_value(n, 0, false) * factor);
    if (format == FORMAT_INT4) dequant_i4_f16_block_sve(packed, weights, scales);
    else dequant_fp4_f16_block_sve(packed, weights, scales);
    for (int row = 0; row < m; ++row) {
        float as = activation_scale_value(row, 0, PATH_W8A16_F16);
        for (size_t k = 0; k < K_BLOCK; ++k)
            activation[(size_t)row * K_BLOCK + k] =
                (fp16_t)((float)activation_code(row, k, PATH_W8A16_F16) * as);
    }
    select_f16_kernel(m)(weights, activation, output);
    for (int row = 0; row < m; ++row) {
        for (size_t n = 0; n < N_TILE; ++n) {
            fp16_t reference = (fp16_t)0.0f;
            for (size_t k = 0; k < K_BLOCK; ++k)
                reference = (fp16_t)((float)reference +
                    (float)weights[k * N_TILE + n] * (float)activation[(size_t)row * K_BLOCK + k]);
            float error = fabsf((float)output[(size_t)row * N_TILE + n] - (float)reference);
            if (error > 0.25f) {
                fprintf(stderr, "FP16 kernel mismatch format=%s M=%d row=%d n=%zu error=%g\n",
                        format_name(format), m, row, n, error);
                return -1;
            }
        }
    }
    return 0;
}

static int run_self_test(void)
{
    if (test_dequant() != 0) {
        fprintf(stderr, "dequant exhaustive test FAILED\n");
        return -1;
    }
    const int rows[] = {1, 2, 4, 6};
    for (int format = FORMAT_INT4; format <= FORMAT_FP4; ++format) {
        for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
            if (test_integer_kernels((quant_format)format, rows[i]) != 0) return -1;
            if (test_f16_kernel((quant_format)format, rows[i]) != 0) return -1;
        }
    }
    printf("self-test: exhaustive dequant + INT8/INT16/FP16 kernels PASS\n");
    return 0;
}

static int run_peak_test(int core)
{
    uint8_t packed[PACKED_BLOCK_BYTES] __attribute__((aligned(256)));
    int8_t weights[N_TILE * K_BLOCK] __attribute__((aligned(256)));
    int8_t activation[MAX_M * K_BLOCK] __attribute__((aligned(256)));
    int32_t output[MAX_M * N_TILE] __attribute__((aligned(256)));
    pack_dot_block(packed, 0, 0, FORMAT_INT4);
    dequant_i4_i8_sve(packed, weights, sizeof(packed));
    pack_i8_m6_activation(activation, 0);
    if (pin_to_cpu(core) != 0) {
        fprintf(stderr, "cannot pin peak test to CPU %d\n", core);
        return -1;
    }
    const size_t groups = 8192;
    const int iterations = 400;
    int8_t *stream_weights = aligned_zero_alloc(groups * 256);
    int8_t *stream_activation = aligned_zero_alloc(groups * 24);
    if (!stream_weights || !stream_activation) {
        free(stream_weights);
        free(stream_activation);
        fprintf(stderr, "cannot allocate standalone SDOT stream\n");
        return -1;
    }
    for (size_t group = 0; group < groups; ++group) {
        memcpy(stream_weights + group * 256, weights + (group % 32) * 256, 256);
        memcpy(stream_activation + group * 24, activation + (group % 32) * 24, 24);
    }
    for (int i = 0; i < 10; ++i)
        gemm_i8_m6_stream_sve(stream_weights, stream_activation, output, groups);
    unsigned khz0 = read_cpu_khz(core);
    uint64_t start = read_cntvct();
    for (int i = 0; i < iterations; ++i)
        gemm_i8_m6_stream_sve(stream_weights, stream_activation, output, groups);
    uint64_t end = read_cntvct();
    unsigned khz1 = read_cpu_khz(core);
    double seconds = (double)(end - start) / (double)read_cntfrq();
    double hz = 1000.0 * (double)(khz0 && khz1 ? (khz0 + khz1) / 2u : 2000000u);
    double sdot = (double)iterations * (double)groups * 4.0 * MAX_M;
    double per_cycle = sdot / (seconds * hz);
    double checksum = 0.0;
    for (size_t i = 0; i < MAX_M * N_TILE; ++i) checksum += output[i];
    printf("standalone M=6 L2-load+SDOT: %.4f SDOT/cycle, %.2f%% of 2/cycle peak "
           "(%zu K-groups, cpu %.0f MHz, checksum %.0f), target 96%%: %s\n",
           per_cycle, per_cycle * 50.0, groups, hz / 1.0e6, checksum,
           per_cycle >= 1.92 ? "PASS" : "MISS");
    free(stream_weights);
    free(stream_activation);
    return per_cycle >= 1.92 ? 0 : 1;
}

static void usage(const char *program)
{
    printf("Usage: %s [options]\n"
           "  --format int4|fp4             packed weight format (default int4)\n"
           "  --path w8a8|i16|auto|fp16     arithmetic path (default w8a8)\n"
           "  --sync hwbar|atomic           pair synchronization (default hwbar)\n"
           "  --m 1|2|4|6                   decode rows (default 1)\n"
           "  --n N                         output columns, multiple of 384 (default 49152)\n"
           "  --k K                         reduction size, multiple of 128 (default 8192)\n"
           "  --chunk-kib KiB               packed bytes per handoff (default 32)\n"
           "  --iterations I                complete weight streams per trial (default 10)\n"
           "  --trials T                    measured trials (default 3)\n"
           "  --core-base CPU               first CPU of the CMG (default 12)\n"
           "  --pairs 4|5|6                 active producer/consumer pairs (default 6)\n"
           "  --sweep-m                     run M=1,2,4,6\n"
           "  --sweep-chunks                run 4,16,32,64,256 KiB handoffs\n"
           "  --sweep-pairs                 run 4,5,6 pairs\n"
           "  --verify                      scalar verification (use a small N/K)\n"
           "  --peak                        run standalone M=6 SDOT issue test first\n"
           "  --unsafe-scale                inject FP16 overflow and test auto fallback\n"
           "  --self-test                   exhaustive decode/kernel tests and exit\n",
           program);
}

static int parse_positive(const char *text, size_t *value)
{
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0' || parsed == 0) return -1;
    *value = (size_t)parsed;
    return 0;
}

static int parse_options(int argc, char **argv, config *cfg)
{
    static const struct option options[] = {
        {"format", required_argument, NULL, 'f'},
        {"path", required_argument, NULL, 'p'},
        {"sync", required_argument, NULL, 's'},
        {"m", required_argument, NULL, 'm'},
        {"n", required_argument, NULL, 'n'},
        {"k", required_argument, NULL, 'k'},
        {"chunk-kib", required_argument, NULL, 'c'},
        {"iterations", required_argument, NULL, 'i'},
        {"trials", required_argument, NULL, 't'},
        {"core-base", required_argument, NULL, 'b'},
        {"pairs", required_argument, NULL, 'P'},
        {"sweep-m", no_argument, NULL, 1000},
        {"sweep-chunks", no_argument, NULL, 1001},
        {"verify", no_argument, NULL, 1002},
        {"peak", no_argument, NULL, 1003},
        {"unsafe-scale", no_argument, NULL, 1004},
        {"self-test", no_argument, NULL, 1005},
        {"sweep-pairs", no_argument, NULL, 1006},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option;
    while ((option = getopt_long(argc, argv, "f:p:s:m:n:k:c:i:t:b:P:h", options, NULL)) != -1) {
        size_t parsed;
        switch (option) {
        case 'f':
            if (strcmp(optarg, "int4") == 0) cfg->format = FORMAT_INT4;
            else if (strcmp(optarg, "fp4") == 0) cfg->format = FORMAT_FP4;
            else return -1;
            break;
        case 'p':
            if (strcmp(optarg, "w8a8") == 0) cfg->requested_path = PATH_W8A8;
            else if (strcmp(optarg, "i16") == 0) cfg->requested_path = PATH_W8A16_I16;
            else if (strcmp(optarg, "auto") == 0) cfg->requested_path = PATH_W8A16_AUTO;
            else if (strcmp(optarg, "fp16") == 0) cfg->requested_path = PATH_W8A16_F16;
            else return -1;
            break;
        case 's':
            if (strcmp(optarg, "hwbar") == 0) cfg->sync = SYNC_HWBAR;
            else if (strcmp(optarg, "atomic") == 0) cfg->sync = SYNC_ATOMIC;
            else return -1;
            break;
        case 'm':
            if (parse_positive(optarg, &parsed) != 0) return -1;
            cfg->m = (int)parsed;
            break;
        case 'n': if (parse_positive(optarg, &cfg->n) != 0) return -1; break;
        case 'k': if (parse_positive(optarg, &cfg->k) != 0) return -1; break;
        case 'c':
            if (parse_positive(optarg, &parsed) != 0 || parsed > SIZE_MAX / 1024) return -1;
            cfg->chunk_bytes = parsed * 1024;
            break;
        case 'i':
            if (parse_positive(optarg, &parsed) != 0 || parsed > INT32_MAX) return -1;
            cfg->iterations = (int)parsed;
            break;
        case 't':
            if (parse_positive(optarg, &parsed) != 0 || parsed > 99) return -1;
            cfg->trials = (int)parsed;
            break;
        case 'b':
            if (parse_positive(optarg, &parsed) != 0 || parsed >= CPU_SETSIZE) return -1;
            cfg->core_base = (int)parsed;
            break;
        case 'P':
            if (parse_positive(optarg, &parsed) != 0 || parsed > MAX_PAIRS) return -1;
            cfg->pairs = (int)parsed;
            break;
        case 1000: cfg->sweep_m = true; break;
        case 1001: cfg->sweep_chunks = true; break;
        case 1002: cfg->verify = true; break;
        case 1003: cfg->peak = true; break;
        case 1004: cfg->unsafe_scale = true; break;
        case 1005: cfg->self_test = true; break;
        case 1006: cfg->sweep_pairs = true; break;
        case 'h': usage(argv[0]); exit(0);
        default: return -1;
        }
    }
    return optind == argc ? 0 : -1;
}

static int compare_double(const void *lhs, const void *rhs)
{
    double a = *(const double *)lhs, b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static int benchmark_one(config cfg)
{
    if ((cfg.n / N_TILE) % (size_t)cfg.pairs != 0) {
        fprintf(stderr, "N=%zu does not divide evenly across %d pairs\n", cfg.n, cfg.pairs);
        return -1;
    }
    bench_data data;
    if (prepare_data(&cfg, &data) != 0) return -1;
    printf("\nformat=%s path=%s requested=%s sync=%s pairs=%d M=%d N=%zu K=%zu "
           "chunk=%zu KiB iterations=%d\n",
           format_name(cfg.format), path_name(cfg.path), path_name(cfg.requested_path),
           sync_name(cfg.sync), cfg.pairs, cfg.m, cfg.n, cfg.k,
           cfg.chunk_bytes / 1024, cfg.iterations);
    size_t expanded_block = cfg.path == PATH_W8A8 ?
        (size_t)N_TILE * K_BLOCK : (size_t)N_TILE * K_BLOCK * sizeof(int16_t);
    size_t ring_slot = (cfg.chunk_bytes / PACKED_BLOCK_BYTES) * expanded_block;
    double ring_ways = (double)(2 * cfg.pairs * ring_slot) / (512.0 * 1024.0);
    printf("packed=%.2f MiB expanded/block=%zu B block128_scales=FP32 "
           "cores=P%d-%d/C%d-%d ring-slot=%zu KiB double-ring=%.2f MiB (%.1f way-equiv)\n",
           (double)data.packed_bytes / 1048576.0,
           expanded_block, cfg.core_base, cfg.core_base + cfg.pairs - 1,
           cfg.core_base + MAX_PAIRS, cfg.core_base + MAX_PAIRS + cfg.pairs - 1,
           ring_slot / 1024, (double)(2 * cfg.pairs * ring_slot) / 1048576.0,
           ring_ways);

    double *bandwidths = calloc((size_t)cfg.trials, sizeof(*bandwidths));
    if (!bandwidths) {
        free_bench_data(&data);
        return -1;
    }
    run_result last = {0};
    for (int trial = 0; trial < cfg.trials; ++trial) {
        run_result result = {0};
        if (run_trial(&cfg, &data, &result) != 0) {
            free(bandwidths);
            free_bench_data(&data);
            return -1;
        }
        bandwidths[trial] = result.compressed_gbs;
        last = result;
        double metadata_bytes = cfg.path == PATH_W8A16_F16 ?
            (double)data.total_blocks * N_TILE * sizeof(float) * cfg.iterations :
            2.0 * (double)data.total_blocks * N_TILE * sizeof(float) * cfg.iterations;
        printf("trial %d: %.3f ms packed %.2f GB/s expanded %.2f GB/s "
               "weights %.2f G/s compute %.2f GOP/s metadata %.2f GB/s "
               "barrier %.1f ns (%.2f%%) checksum %.9e cpu %u/%u MHz\n",
               trial + 1, result.seconds * 1.0e3, result.compressed_gbs,
               result.expanded_gbs, result.logical_gweights, result.gops,
               metadata_bytes / result.seconds / 1.0e9,
               result.barrier_ns, result.barrier_percent, result.checksum,
               result.cpu_khz_before / 1000, result.cpu_khz_after / 1000);
    }
    qsort(bandwidths, (size_t)cfg.trials, sizeof(*bandwidths), compare_double);
    double median = bandwidths[cfg.trials / 2];
    double best = bandwidths[cfg.trials - 1];
    printf("summary: packed median %.2f GB/s best %.2f GB/s target 240 GB/s: %s\n",
           median, best, best >= 240.0 ? "PASS" : "MISS");
    int verify_rc = 0;
    if (cfg.verify) verify_rc = verify_output(&cfg, &data);
    (void)last;
    free(bandwidths);
    free_bench_data(&data);
    return verify_rc;
}

int main(int argc, char **argv)
{
    config cfg = {
        .format = FORMAT_INT4,
        .requested_path = PATH_W8A8,
        .path = PATH_W8A8,
        .sync = SYNC_HWBAR,
        .m = 1,
        .n = 49152,
        .k = 8192,
        .chunk_bytes = 32 * 1024,
        .iterations = 10,
        .trials = 3,
        .core_base = 12,
        .pairs = 6,
    };
    if (parse_options(argc, argv, &cfg) != 0) {
        usage(argv[0]);
        return 2;
    }
    if (cfg.self_test) return run_self_test() == 0 ? 0 : 1;
    if (cfg.m != 1 && cfg.m != 2 && cfg.m != 4 && cfg.m != 6) {
        fprintf(stderr, "M must be one of 1, 2, 4, or 6\n");
        return 2;
    }
    if (cfg.pairs < 4 || cfg.pairs > MAX_PAIRS) {
        fprintf(stderr, "pair count must be 4, 5, or 6\n");
        return 2;
    }
    if (cfg.n % N_TILE != 0 || cfg.k % K_BLOCK != 0) {
        fprintf(stderr, "N must be a multiple of %d and K a multiple of %d\n",
                N_TILE, K_BLOCK);
        return 2;
    }
    if (cfg.chunk_bytes % PACKED_BLOCK_BYTES != 0) {
        fprintf(stderr, "chunk size must be a multiple of %d packed bytes (4 KiB)\n",
                PACKED_BLOCK_BYTES);
        return 2;
    }
    int active_pairs = cfg.sweep_pairs ? MAX_PAIRS : cfg.pairs;
    for (int pair = 0; pair < active_pairs; ++pair) {
        int producer_cpu = cfg.core_base + pair;
        int consumer_cpu = cfg.core_base + MAX_PAIRS + pair;
        if (!cpu_is_available(producer_cpu) || !cpu_is_available(consumer_cpu)) {
            fprintf(stderr, "pair CPUs %d,%d are not available in this process affinity mask\n",
                    producer_cpu, consumer_cpu);
            return 2;
        }
    }
    if (pin_to_cpu(cfg.core_base) != 0) {
        fprintf(stderr, "cannot pin initialization to CPU %d\n", cfg.core_base);
        return 2;
    }
    printf("A64FX dequant->L2->GEMM experiment: %s producer/consumer pairs, timer=%lu Hz\n",
           cfg.sweep_pairs ? "4/5/6" : (cfg.pairs == 4 ? "four" :
           (cfg.pairs == 5 ? "five" : "six")), (unsigned long)read_cntfrq());
    if (cfg.peak) {
        int peak_rc = run_peak_test(cfg.core_base + MAX_PAIRS);
        if (peak_rc < 0) return 1;
    }

    const int m_values[] = {1, 2, 4, 6};
    const size_t chunk_values[] = {4, 16, 32, 64, 256};
    const int pair_values[] = {4, 5, 6};
    size_t m_count = cfg.sweep_m ? sizeof(m_values) / sizeof(m_values[0]) : 1;
    size_t chunk_count = cfg.sweep_chunks ? sizeof(chunk_values) / sizeof(chunk_values[0]) : 1;
    size_t pair_count = cfg.sweep_pairs ? sizeof(pair_values) / sizeof(pair_values[0]) : 1;
    int rc = 0;
    for (size_t pi = 0; pi < pair_count; ++pi) {
        config current = cfg;
        if (cfg.sweep_pairs) current.pairs = pair_values[pi];
        for (size_t mi = 0; mi < m_count; ++mi) {
            if (cfg.sweep_m) current.m = m_values[mi];
            for (size_t ci = 0; ci < chunk_count; ++ci) {
                if (cfg.sweep_chunks) current.chunk_bytes = chunk_values[ci] * 1024;
                if (benchmark_one(current) != 0) rc = 1;
            }
        }
    }
    return rc;
}
