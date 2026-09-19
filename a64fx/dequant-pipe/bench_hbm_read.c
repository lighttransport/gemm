#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

enum { MIN_CORES = 4, MAX_CORES = 8, CACHE_LINE_BYTES = 256 };

typedef enum { MODE_READ, MODE_PREFETCH, MODE_NULL_COPY, MODE_NULL_DEQUANT } bench_mode;

typedef struct {
    int cores;
    int core_base;
    int iterations;
    int trials;
    size_t bytes;
    size_t chunk_bytes;
    bool sweep;
    bench_mode mode;
} config;

struct run_state;

typedef struct {
    struct run_state *run;
    int cpu;
    const uint8_t *source;
    size_t bytes;
    uint8_t *ring[2];
    uint64_t begin;
    uint64_t end;
    int error;
} worker;

typedef struct run_state {
    const config *cfg;
    atomic_int ready;
    atomic_int start;
    worker workers[MAX_CORES];
} run_state;

extern void hbm_read_256_sve(const uint8_t *, size_t);
extern void hbm_read_256_pf_sve(const uint8_t *, size_t);
extern void null_expand1_sve(const uint8_t *, uint8_t *, size_t);
extern void null_expand2_sve(const uint8_t *, uint8_t *, size_t);

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

static int pin_to_cpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static bool cpu_available(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    return cpu >= 0 && cpu < CPU_SETSIZE &&
           sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_ISSET(cpu, &set);
}

static unsigned read_cpu_khz(int cpu)
{
    char path[128];
    unsigned value = 0;
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE *fp = fopen(path, "r");
    if (fp != NULL) {
        if (fscanf(fp, "%u", &value) != 1) value = 0;
        fclose(fp);
    }
    return value;
}

static void *worker_main(void *opaque)
{
    worker *item = opaque;
    run_state *run = item->run;
    if (pin_to_cpu(item->cpu) != 0) item->error = errno ? errno : EINVAL;
    atomic_fetch_add_explicit(&run->ready, 1, memory_order_release);
    while (!atomic_load_explicit(&run->start, memory_order_acquire))
        __asm__ volatile("yield");
    item->begin = read_cntvct();
    if (item->error == 0) {
        for (int iteration = 0; iteration < run->cfg->iterations; ++iteration) {
            if (run->cfg->mode == MODE_PREFETCH) {
                hbm_read_256_pf_sve(item->source, item->bytes);
            } else if (run->cfg->mode == MODE_READ) {
                hbm_read_256_sve(item->source, item->bytes);
            } else {
                size_t chunks = item->bytes / run->cfg->chunk_bytes;
                for (size_t chunk = 0; chunk < chunks; ++chunk) {
                    if (run->cfg->mode == MODE_NULL_COPY)
                        null_expand1_sve(item->source + chunk * run->cfg->chunk_bytes,
                                         item->ring[chunk & 1u], run->cfg->chunk_bytes);
                    else
                        null_expand2_sve(item->source + chunk * run->cfg->chunk_bytes,
                                         item->ring[chunk & 1u], run->cfg->chunk_bytes);
                }
            }
        }
    }
    item->end = read_cntvct();
    return NULL;
}

static int run_trial(const config *cfg, const uint8_t *source, size_t active_bytes,
                     double *seconds)
{
    run_state run;
    pthread_t threads[MAX_CORES];
    memset(&run, 0, sizeof(run));
    run.cfg = cfg;
    atomic_init(&run.ready, 0);
    atomic_init(&run.start, 0);
    size_t bytes_per_core = active_bytes / (size_t)cfg->cores;
    for (int index = 0; index < cfg->cores; ++index) {
        run.workers[index].run = &run;
        run.workers[index].cpu = cfg->core_base + index;
        run.workers[index].source = source + (size_t)index * bytes_per_core;
        run.workers[index].bytes = bytes_per_core;
        if (cfg->mode == MODE_NULL_COPY || cfg->mode == MODE_NULL_DEQUANT) {
            size_t expansion = cfg->mode == MODE_NULL_DEQUANT ? 2u : 1u;
            size_t ring_bytes = expansion * cfg->chunk_bytes;
            for (int slot = 0; slot < 2; ++slot) {
                if (posix_memalign((void **)&run.workers[index].ring[slot],
                                   CACHE_LINE_BYTES, ring_bytes) != 0) {
                    fprintf(stderr, "cannot allocate null-dequant ring\n");
                    return -1;
                }
                memset(run.workers[index].ring[slot], 0, ring_bytes);
            }
        }
        int rc = pthread_create(&threads[index], NULL, worker_main, &run.workers[index]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create: %s\n", strerror(rc));
            return -1;
        }
    }
    while (atomic_load_explicit(&run.ready, memory_order_acquire) < cfg->cores)
        __asm__ volatile("yield");
    atomic_store_explicit(&run.start, 1, memory_order_release);
    for (int index = 0; index < cfg->cores; ++index)
        pthread_join(threads[index], NULL);

    uint64_t first = UINT64_MAX;
    uint64_t last = 0;
    for (int index = 0; index < cfg->cores; ++index) {
        if (run.workers[index].error != 0) {
            fprintf(stderr, "cannot pin worker to CPU %d: %s\n",
                    run.workers[index].cpu, strerror(run.workers[index].error));
            return -1;
        }
        if (run.workers[index].begin < first) first = run.workers[index].begin;
        if (run.workers[index].end > last) last = run.workers[index].end;
    }
    if (cfg->mode == MODE_NULL_COPY || cfg->mode == MODE_NULL_DEQUANT) {
        for (int index = 0; index < cfg->cores; ++index) {
            for (int slot = 0; slot < 2; ++slot)
                free(run.workers[index].ring[slot]);
        }
    }
    *seconds = (double)(last - first) / (double)read_cntfrq();
    return *seconds > 0.0 ? 0 : -1;
}

static int compare_double(const void *lhs, const void *rhs)
{
    double a = *(const double *)lhs;
    double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static int benchmark_one(config cfg, const uint8_t *source)
{
    bool stores = cfg.mode == MODE_NULL_COPY || cfg.mode == MODE_NULL_DEQUANT;
    size_t unit = stores ? cfg.chunk_bytes : CACHE_LINE_BYTES;
    size_t quantum = (size_t)cfg.cores * unit;
    size_t active_bytes = cfg.bytes / quantum * quantum;
    size_t bytes_per_core = active_bytes / (size_t)cfg.cores;
    double *rates = calloc((size_t)cfg.trials, sizeof(*rates));
    if (rates == NULL || active_bytes == 0) {
        free(rates);
        return -1;
    }
    printf("\nHBM read: cores=%d CPUs=%d-%d mode=%s total=%.3f MiB "
           "per-core=%.3f MiB iterations=%d\n",
           cfg.cores, cfg.core_base, cfg.core_base + cfg.cores - 1,
           cfg.mode == MODE_PREFETCH ? "prefetch-8-lines" :
           cfg.mode == MODE_NULL_COPY ? "null-copy-1x" :
           cfg.mode == MODE_NULL_DEQUANT ? "null-dequant-2x" : "baseline",
           (double)active_bytes / 1048576.0,
           (double)bytes_per_core / 1048576.0, cfg.iterations);
    if (cfg.mode == MODE_NULL_DEQUANT)
        printf("inner loop: 2 x 64-byte SVE loads per 128 packed bytes; "
               "two iterations consume one 256-byte cache line; %zu lines/pass\n",
               active_bytes / CACHE_LINE_BYTES);
    else
        printf("inner loop: 4 x 64-byte SVE loads = one 256-byte cache line; "
               "%zu lines/pass\n", active_bytes / CACHE_LINE_BYTES);
    if (stores) {
        size_t expansion = cfg.mode == MODE_NULL_DEQUANT ? 2u : 1u;
        printf("handoff: %zux expansion, two %zu KiB slots/core (%.1f KiB/core)\n",
               expansion, expansion * cfg.chunk_bytes / 1024u,
               2.0 * expansion * cfg.chunk_bytes / 1024.0);
    }
    for (int trial = 0; trial < cfg.trials; ++trial) {
        double seconds;
        unsigned khz0 = read_cpu_khz(cfg.core_base);
        if (run_trial(&cfg, source, active_bytes, &seconds) != 0) {
            free(rates);
            return -1;
        }
        unsigned khz1 = read_cpu_khz(cfg.core_base);
        rates[trial] = (double)active_bytes * cfg.iterations / seconds / 1.0e9;
        printf("trial %d: %.3f ms packed-HBM %.2f GB/s (%.2f GB/s/core)",
               trial + 1, seconds * 1.0e3, rates[trial], rates[trial] / cfg.cores);
        if (stores) {
            double expansion = cfg.mode == MODE_NULL_DEQUANT ? 2.0 : 1.0;
            printf(" L2-ring-write %.2f GB/s", expansion * rates[trial]);
        }
        printf(" cpu %u/%u MHz\n", khz0 / 1000, khz1 / 1000);
    }
    qsort(rates, (size_t)cfg.trials, sizeof(*rates), compare_double);
    double median = rates[cfg.trials / 2];
    double best = rates[cfg.trials - 1];
    if (stores)
        printf("summary: packed-HBM median %.2f GB/s best %.2f GB/s\n", median, best);
    else
        printf("summary: median %.2f GB/s best %.2f GB/s target 240 GB/s: %s\n",
               median, best, median >= 228.0 ? "WITHIN 5%" : "MISS");
    free(rates);
    return 0;
}

static int parse_positive(const char *text, size_t *value)
{
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) return -1;
    *value = (size_t)parsed;
    return 0;
}

static void usage(const char *program)
{
    printf("Usage: %s [--cores 4..8 | --sweep-cores] "
           "[--mode baseline|prefetch|null-copy|null-dequant]\n"
           "          [--mib MiB] [--chunk-kib KiB] [--iterations I] "
           "[--trials T] [--core-base CPU]\n",
           program);
}

static int parse_options(int argc, char **argv, config *cfg)
{
    static const struct option options[] = {
        {"cores", required_argument, NULL, 'c'},
        {"sweep-cores", no_argument, NULL, 's'},
        {"mode", required_argument, NULL, 'M'},
        {"mib", required_argument, NULL, 'm'},
        {"chunk-kib", required_argument, NULL, 'k'},
        {"iterations", required_argument, NULL, 'i'},
        {"trials", required_argument, NULL, 't'},
        {"core-base", required_argument, NULL, 'b'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option;
    while ((option = getopt_long(argc, argv, "c:sM:m:k:i:t:b:h", options, NULL)) != -1) {
        size_t parsed;
        switch (option) {
        case 'c':
            if (parse_positive(optarg, &parsed) != 0) return -1;
            cfg->cores = (int)parsed;
            break;
        case 's': cfg->sweep = true; break;
        case 'M':
            if (strcmp(optarg, "baseline") == 0) cfg->mode = MODE_READ;
            else if (strcmp(optarg, "prefetch") == 0) cfg->mode = MODE_PREFETCH;
            else if (strcmp(optarg, "null-copy") == 0) cfg->mode = MODE_NULL_COPY;
            else if (strcmp(optarg, "null-dequant") == 0) cfg->mode = MODE_NULL_DEQUANT;
            else return -1;
            break;
        case 'm':
            if (parse_positive(optarg, &parsed) != 0 || parsed > SIZE_MAX / 1048576u)
                return -1;
            cfg->bytes = parsed * 1048576u;
            break;
        case 'k':
            if (parse_positive(optarg, &parsed) != 0 || parsed > SIZE_MAX / 1024u)
                return -1;
            cfg->chunk_bytes = parsed * 1024u;
            break;
        case 'i':
            if (parse_positive(optarg, &parsed) != 0) return -1;
            cfg->iterations = (int)parsed;
            break;
        case 't':
            if (parse_positive(optarg, &parsed) != 0) return -1;
            cfg->trials = (int)parsed;
            break;
        case 'b':
            if (parse_positive(optarg, &parsed) != 0) return -1;
            cfg->core_base = (int)parsed;
            break;
        case 'h': usage(argv[0]); exit(0);
        default: return -1;
        }
    }
    return optind == argc ? 0 : -1;
}

int main(int argc, char **argv)
{
    config cfg = {
        .cores = MAX_CORES,
        .core_base = 12,
        .iterations = 20,
        .trials = 3,
        .bytes = 240u * 1048576u,
        .chunk_bytes = 128u * 1024u,
        .sweep = false,
        .mode = MODE_READ,
    };
    if (parse_options(argc, argv, &cfg) != 0 || cfg.cores < MIN_CORES ||
        cfg.cores > MAX_CORES || cfg.iterations <= 0 || cfg.trials <= 0 ||
        cfg.chunk_bytes == 0 || cfg.chunk_bytes % CACHE_LINE_BYTES != 0) {
        usage(argv[0]);
        return 2;
    }
    int largest = cfg.sweep ? MAX_CORES : cfg.cores;
    for (int index = 0; index < largest; ++index) {
        if (!cpu_available(cfg.core_base + index)) {
            fprintf(stderr, "CPU %d is unavailable\n", cfg.core_base + index);
            return 2;
        }
    }
    if (pin_to_cpu(cfg.core_base) != 0) {
        fprintf(stderr, "cannot pin allocator to CPU %d\n", cfg.core_base);
        return 2;
    }
    void *allocation = NULL;
    if (posix_memalign(&allocation, 2u * 1024u * 1024u, cfg.bytes) != 0) {
        fprintf(stderr, "cannot allocate %.1f MiB source\n", (double)cfg.bytes / 1048576.0);
        return 1;
    }
    (void)madvise(allocation, cfg.bytes, MADV_HUGEPAGE);
    uint8_t *source = allocation;
    for (size_t offset = 0; offset < cfg.bytes; offset += 4096)
        source[offset] = (uint8_t)(offset >> 12);

    int rc = 0;
    if (cfg.sweep) {
        for (int cores = MIN_CORES; cores <= MAX_CORES; ++cores) {
            config current = cfg;
            current.cores = cores;
            if (benchmark_one(current, source) != 0) rc = 1;
        }
    } else if (benchmark_one(cfg, source) != 0) {
        rc = 1;
    }
    free(source);
    return rc;
}
