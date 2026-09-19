#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <sched.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "hwb_compat.h"

enum {
    MAX_PAIRS = 6,
    CONSUMER_OFFSET = 6,
    L2_WAY_BYTES = 512 * 1024,
};

typedef enum { SYNC_HWBAR, SYNC_ATOMIC } sync_mode;

typedef struct {
    int pairs;
    int expansion;
    int core_base;
    size_t bytes;
    size_t chunk_bytes;
    int iterations;
    int trials;
    bool sweep_pairs;
    sync_mode sync;
} config;

typedef struct {
    alignas(256) atomic_uint arrivals;
    atomic_uint generation;
    unsigned char padding[256 - 2 * sizeof(atomic_uint)];
} soft_barrier;

typedef struct {
    soft_barrier barrier;
    uint8_t *ring[2];
    uint8_t *sink;
    int descriptor;
} pair_state;

struct run_state;

typedef struct {
    struct run_state *run;
    int pair;
    int cpu;
    bool producer;
    long barrier_window;
    uint64_t begin;
    uint64_t end;
    uint64_t barrier_ticks;
    int error;
} worker_state;

typedef struct run_state {
    const config *cfg;
    uint8_t *source;
    pair_state pairs[MAX_PAIRS];
    worker_state workers[MAX_PAIRS * 2];
    atomic_int ready;
    atomic_int start;
    atomic_int failed;
    size_t bytes_per_pair;
    size_t chunks_per_pair;
} run_state;

extern void null_expand1_sve(const uint8_t *, uint8_t *, size_t);
extern void null_expand2_sve(const uint8_t *, uint8_t *, size_t);
extern void null_expand4_sve(const uint8_t *, uint8_t *, size_t);
extern void null_copy_sve(const uint8_t *, uint8_t *, size_t);

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
    return sched_getaffinity(0, sizeof(set), &set) == 0 &&
           cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &set);
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

static void *aligned_alloc_zero(size_t alignment, size_t bytes)
{
    void *ptr = NULL;
    size_t rounded = (bytes + alignment - 1) & ~(alignment - 1);
    if (posix_memalign(&ptr, alignment, rounded) != 0) return NULL;
    memset(ptr, 0, rounded);
    (void)madvise(ptr, rounded, MADV_HUGEPAGE);
    return ptr;
}

static void soft_barrier_wait(soft_barrier *barrier)
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

static void pair_barrier(worker_state *worker)
{
    uint64_t begin = read_cntvct();
    shared_fence();
    if (worker->run->cfg->sync == SYNC_HWBAR)
        vhbm_bar(worker->barrier_window);
    else
        soft_barrier_wait(&worker->run->pairs[worker->pair].barrier);
    shared_fence();
    worker->barrier_ticks += read_cntvct() - begin;
}

static void null_expand(const config *cfg, const uint8_t *src, uint8_t *dst)
{
    if (cfg->expansion == 1) null_expand1_sve(src, dst, cfg->chunk_bytes);
    else if (cfg->expansion == 2) null_expand2_sve(src, dst, cfg->chunk_bytes);
    else null_expand4_sve(src, dst, cfg->chunk_bytes);
}

static void *worker_main(void *opaque)
{
    worker_state *worker = opaque;
    run_state *run = worker->run;
    const config *cfg = run->cfg;
    pair_state *pair = &run->pairs[worker->pair];
    if (pin_to_cpu(worker->cpu) != 0) {
        worker->error = errno ? errno : EINVAL;
        atomic_store(&run->failed, 1);
    }
    if (cfg->sync == SYNC_HWBAR && worker->error == 0) {
        long requested = -1;
        worker->barrier_window = vhbm_bar_assign(pair->descriptor, &requested);
        if (worker->barrier_window < 0) {
            worker->error = (int)-worker->barrier_window;
            atomic_store(&run->failed, 1);
        }
    }
    atomic_fetch_add(&run->ready, 1);
    while (!atomic_load_explicit(&run->start, memory_order_acquire))
        __asm__ volatile("yield");
    worker->begin = read_cntvct();
    if (!atomic_load(&run->failed)) {
        size_t sequences = run->chunks_per_pair * (size_t)cfg->iterations;
        size_t pair_offset = (size_t)worker->pair * run->bytes_per_pair;
        for (size_t sequence = 0; sequence < sequences; ++sequence) {
            size_t chunk = sequence % run->chunks_per_pair;
            uint8_t *ring = pair->ring[sequence & 1u];
            if (worker->producer) {
                const uint8_t *src = run->source + pair_offset + chunk * cfg->chunk_bytes;
                null_expand(cfg, src, ring);
            }
            pair_barrier(worker);
            if (!worker->producer)
                null_copy_sve(ring, pair->sink, cfg->chunk_bytes * (size_t)cfg->expansion);
        }
    }
    shared_fence();
    worker->end = read_cntvct();
    if (cfg->sync == SYNC_HWBAR && worker->barrier_window >= 0) {
        int rc = vhbm_bar_unassign(pair->descriptor);
        if (rc != 0 && worker->error == 0) worker->error = rc < 0 ? -rc : rc;
    }
    return NULL;
}

static void release_buffers(run_state *run)
{
    for (int pair = 0; pair < run->cfg->pairs; ++pair) {
        free(run->pairs[pair].ring[0]);
        free(run->pairs[pair].ring[1]);
        free(run->pairs[pair].sink);
    }
}

static int allocate_buffers(run_state *run)
{
    size_t expanded = run->cfg->chunk_bytes * (size_t)run->cfg->expansion;
    for (int pair = 0; pair < run->cfg->pairs; ++pair) {
        pair_state *state = &run->pairs[pair];
        atomic_init(&state->barrier.arrivals, 0);
        atomic_init(&state->barrier.generation, 0);
        /* Capacity is reported in 512 KiB way-equivalents; do not force all
         * buffers to the same cache-index color with 512 KiB alignment. */
        state->ring[0] = aligned_alloc_zero(256, expanded);
        state->ring[1] = aligned_alloc_zero(256, expanded);
        state->sink = aligned_alloc_zero(256, expanded);
        if (!state->ring[0] || !state->ring[1] || !state->sink) {
            release_buffers(run);
            return -1;
        }
    }
    return 0;
}

static int init_hardware_barriers(run_state *run)
{
    if (access("/dev/xos_hwb", R_OK | W_OK) != 0) {
        fprintf(stderr, "hardware barrier device /dev/xos_hwb is unavailable: %s\n",
                strerror(errno));
        return -1;
    }
    for (int pair = 0; pair < run->cfg->pairs; ++pair) {
        int pcpu = run->cfg->core_base + pair;
        int ccpu = run->cfg->core_base + CONSUMER_OFFSET + pair;
        int descriptor = vhbm_bar_init((1UL << pcpu) | (1UL << ccpu));
        if (descriptor <= 0) {
            fprintf(stderr, "hardware barrier allocation failed for CPUs %d,%d: %d\n",
                    pcpu, ccpu, descriptor);
            for (int previous = 0; previous < pair; ++previous)
                (void)vhbm_bar_fini(run->pairs[previous].descriptor);
            return -1;
        }
        run->pairs[pair].descriptor = descriptor;
    }
    return 0;
}

static void fini_hardware_barriers(run_state *run)
{
    for (int pair = 0; pair < run->cfg->pairs; ++pair)
        (void)vhbm_bar_fini(run->pairs[pair].descriptor);
}

typedef struct {
    double seconds;
    double packed_gbs;
    double barrier_ns;
    double barrier_percent;
} result;

static int run_trial(const config *cfg, uint8_t *source, result *out)
{
    run_state run;
    pthread_t threads[MAX_PAIRS * 2];
    memset(&run, 0, sizeof(run));
    run.cfg = cfg;
    run.source = source;
    run.bytes_per_pair = cfg->bytes / (size_t)cfg->pairs;
    run.chunks_per_pair = run.bytes_per_pair / cfg->chunk_bytes;
    atomic_init(&run.ready, 0);
    atomic_init(&run.start, 0);
    atomic_init(&run.failed, 0);
    if (allocate_buffers(&run) != 0) {
        fprintf(stderr, "null-pipeline buffer allocation failed\n");
        return -1;
    }
    if (cfg->sync == SYNC_HWBAR && init_hardware_barriers(&run) != 0) {
        release_buffers(&run);
        return -1;
    }
    for (int pair = 0; pair < cfg->pairs; ++pair) {
        worker_state *producer = &run.workers[pair * 2];
        worker_state *consumer = &run.workers[pair * 2 + 1];
        producer->run = &run;
        producer->pair = pair;
        producer->cpu = cfg->core_base + pair;
        producer->producer = true;
        producer->barrier_window = -1;
        consumer->run = &run;
        consumer->pair = pair;
        consumer->cpu = cfg->core_base + CONSUMER_OFFSET + pair;
        consumer->barrier_window = -1;
    }
    int created = 0;
    for (int index = 0; index < cfg->pairs * 2; ++index) {
        int rc = pthread_create(&threads[index], NULL, worker_main, &run.workers[index]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create: %s\n", strerror(rc));
            atomic_store(&run.failed, 1);
            break;
        }
        ++created;
    }
    while (atomic_load(&run.ready) < created) __asm__ volatile("yield");
    atomic_store_explicit(&run.start, 1, memory_order_release);
    for (int index = 0; index < created; ++index) pthread_join(threads[index], NULL);

    int failed = created != cfg->pairs * 2 || atomic_load(&run.failed);
    uint64_t first = UINT64_MAX, last = 0, barrier_ticks = 0;
    for (int index = 0; index < cfg->pairs * 2; ++index) {
        worker_state *worker = &run.workers[index];
        if (worker->begin < first) first = worker->begin;
        if (worker->end > last) last = worker->end;
        barrier_ticks += worker->barrier_ticks;
        if (worker->error != 0) failed = 1;
    }
    if (cfg->sync == SYNC_HWBAR) fini_hardware_barriers(&run);
    if (!failed && last > first) {
        double timer = (double)read_cntfrq();
        out->seconds = (double)(last - first) / timer;
        out->packed_gbs = (double)cfg->bytes * cfg->iterations / out->seconds / 1.0e9;
        size_t syncs = run.chunks_per_pair * (size_t)cfg->iterations;
        out->barrier_ns = ((double)barrier_ticks / (cfg->pairs * 2) / syncs) /
                          timer * 1.0e9;
        out->barrier_percent = ((double)barrier_ticks / (cfg->pairs * 2) / timer) /
                               out->seconds * 100.0;
    }

    if (!failed) {
        for (int pair = 0; pair < cfg->pairs && !failed; ++pair) {
            size_t input_base = (size_t)pair * run.bytes_per_pair +
                                (run.chunks_per_pair - 1) * cfg->chunk_bytes;
            size_t expanded = cfg->chunk_bytes * (size_t)cfg->expansion;
            for (size_t offset = 0; offset < expanded; offset += 4093) {
                size_t segment = offset / (64u * (size_t)cfg->expansion);
                size_t within = offset % 64u;
                uint8_t expected = source[input_base + segment * 64u + within];
                if (run.pairs[pair].sink[offset] != expected) failed = 1;
            }
        }
    }
    release_buffers(&run);
    return failed ? -1 : 0;
}

static int compare_double(const void *lhs, const void *rhs)
{
    double a = *(const double *)lhs, b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static int benchmark_one(config cfg, uint8_t *source)
{
    if (cfg.bytes % ((size_t)cfg.pairs * cfg.chunk_bytes) != 0) {
        fprintf(stderr, "input bytes must divide evenly into pair chunks\n");
        return -1;
    }
    size_t expanded = cfg.chunk_bytes * (size_t)cfg.expansion;
    double ways = (double)(3 * cfg.pairs * expanded) / L2_WAY_BYTES;
    printf("\nnull transport: pairs=%d expansion=%dx sync=%s packed=%.1f MiB "
           "chunk=%zu KiB expanded-slot=%zu KiB L2-ring+sink=%.2f MiB (%.1f way-equiv)\n",
           cfg.pairs, cfg.expansion, cfg.sync == SYNC_HWBAR ? "hardware" : "atomic-control",
           (double)cfg.bytes / 1048576.0, cfg.chunk_bytes / 1024, expanded / 1024,
           (double)(3 * cfg.pairs * expanded) / 1048576.0, ways);
    double *rates = calloc((size_t)cfg.trials, sizeof(*rates));
    if (!rates) return -1;
    for (int trial = 0; trial < cfg.trials; ++trial) {
        result measured = {0};
        unsigned khz0 = read_cpu_khz(cfg.core_base);
        if (run_trial(&cfg, source, &measured) != 0) {
            free(rates);
            return -1;
        }
        unsigned khz1 = read_cpu_khz(cfg.core_base);
        rates[trial] = measured.packed_gbs;
        double expanded_rate = measured.packed_gbs * cfg.expansion;
        double l2_path_rate = 2.0 * measured.packed_gbs + 3.0 * expanded_rate;
        printf("trial %d: %.3f ms packed-HBM %.2f GB/s ring-write/read %.2f GB/s "
               "sink-write %.2f GB/s nominal-L2-path %.2f GB/s barrier %.1f ns "
               "(%.2f%%) cpu %u/%u MHz verify PASS\n",
               trial + 1, measured.seconds * 1.0e3, measured.packed_gbs,
               expanded_rate, expanded_rate, l2_path_rate, measured.barrier_ns,
               measured.barrier_percent, khz0 / 1000, khz1 / 1000);
    }
    qsort(rates, (size_t)cfg.trials, sizeof(*rates), compare_double);
    double median = rates[cfg.trials / 2], best = rates[cfg.trials - 1];
    printf("summary: packed-HBM median %.2f GB/s best %.2f GB/s target 240 GB/s: %s\n",
           median, best, median >= 240.0 ? "PASS" : "MISS");
    free(rates);
    return 0;
}

static void usage(const char *program)
{
    printf("Usage: %s [--sync hwbar|atomic] [--pairs 4|5|6] [--sweep-pairs]\n"
           "          [--expansion 1|2|4] [--mib MiB] [--chunk-kib KiB]\n"
           "          [--iterations I] [--trials T] [--core-base CPU]\n", program);
}

static int parse_size(const char *text, size_t *value)
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
        {"sync", required_argument, NULL, 's'},
        {"pairs", required_argument, NULL, 'p'},
        {"sweep-pairs", no_argument, NULL, 1000},
        {"expansion", required_argument, NULL, 'e'},
        {"mib", required_argument, NULL, 'm'},
        {"chunk-kib", required_argument, NULL, 'c'},
        {"iterations", required_argument, NULL, 'i'},
        {"trials", required_argument, NULL, 't'},
        {"core-base", required_argument, NULL, 'b'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option;
    while ((option = getopt_long(argc, argv, "s:p:e:m:c:i:t:b:h", options, NULL)) != -1) {
        size_t parsed;
        switch (option) {
        case 's':
            if (strcmp(optarg, "hwbar") == 0) cfg->sync = SYNC_HWBAR;
            else if (strcmp(optarg, "atomic") == 0) cfg->sync = SYNC_ATOMIC;
            else return -1;
            break;
        case 'p':
            if (parse_size(optarg, &parsed) != 0) return -1;
            cfg->pairs = (int)parsed;
            break;
        case 1000: cfg->sweep_pairs = true; break;
        case 'e':
            if (parse_size(optarg, &parsed) != 0) return -1;
            cfg->expansion = (int)parsed;
            break;
        case 'm':
            if (parse_size(optarg, &parsed) != 0 || parsed > SIZE_MAX / 1048576u) return -1;
            cfg->bytes = parsed * 1048576u;
            break;
        case 'c':
            if (parse_size(optarg, &parsed) != 0 || parsed > SIZE_MAX / 1024u) return -1;
            cfg->chunk_bytes = parsed * 1024u;
            break;
        case 'i':
            if (parse_size(optarg, &parsed) != 0) return -1;
            cfg->iterations = (int)parsed;
            break;
        case 't':
            if (parse_size(optarg, &parsed) != 0) return -1;
            cfg->trials = (int)parsed;
            break;
        case 'b':
            if (parse_size(optarg, &parsed) != 0) return -1;
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
        .pairs = 6,
        .expansion = 2,
        .core_base = 12,
        .bytes = 240u * 1048576u,
        .chunk_bytes = 128u * 1024u,
        .iterations = 5,
        .trials = 3,
        .sync = SYNC_HWBAR,
    };
    if (parse_options(argc, argv, &cfg) != 0) {
        usage(argv[0]);
        return 2;
    }
    if (cfg.pairs < 4 || cfg.pairs > MAX_PAIRS ||
        (cfg.expansion != 1 && cfg.expansion != 2 && cfg.expansion != 4) ||
        cfg.chunk_bytes % 4096u != 0 || cfg.iterations <= 0 || cfg.trials <= 0) {
        usage(argv[0]);
        return 2;
    }
    int active_pairs = cfg.sweep_pairs ? MAX_PAIRS : cfg.pairs;
    for (int pair = 0; pair < active_pairs; ++pair) {
        int pcpu = cfg.core_base + pair;
        int ccpu = cfg.core_base + CONSUMER_OFFSET + pair;
        if (!cpu_available(pcpu) || !cpu_available(ccpu)) {
            fprintf(stderr, "CPUs %d,%d are unavailable\n", pcpu, ccpu);
            return 2;
        }
    }
    if (pin_to_cpu(cfg.core_base) != 0) return 2;
    uint8_t *source = aligned_alloc_zero(2u * 1024u * 1024u, cfg.bytes);
    if (!source) {
        fprintf(stderr, "cannot allocate %.1f MiB source\n", (double)cfg.bytes / 1048576.0);
        return 1;
    }
    for (size_t i = 0; i < cfg.bytes; ++i)
        source[i] = (uint8_t)(i * 13u + (i >> 8));

    const int pair_values[] = {4, 5, 6};
    size_t count = cfg.sweep_pairs ? 3 : 1;
    int rc = 0;
    for (size_t index = 0; index < count; ++index) {
        config current = cfg;
        if (cfg.sweep_pairs) current.pairs = pair_values[index];
        if (benchmark_one(current, source) != 0) rc = 1;
    }
    free(source);
    return rc;
}
