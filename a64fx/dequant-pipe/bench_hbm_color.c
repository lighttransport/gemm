#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
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
#include <unistd.h>

enum { CACHE_LINE = 256, MAX_CORES = 12, MAX_TRIALS = 31 };

extern void hbm_read_256_sve(const uint8_t *, size_t);

typedef struct {
    const uint8_t *source;
    size_t bytes;
    int cpu;
    int iterations;
    atomic_int *ready;
    atomic_int *start;
    uint64_t begin;
    uint64_t end;
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

static void *worker_main(void *opaque)
{
    worker *w = opaque;
    if (pin_cpu(w->cpu) != 0) w->error = errno ? errno : EINVAL;
    atomic_fetch_add_explicit(w->ready, 1, memory_order_release);
    while (!atomic_load_explicit(w->start, memory_order_acquire))
        __asm__ volatile("yield");
    w->begin = cntvct();
    if (!w->error)
        for (int i = 0; i < w->iterations; ++i)
            hbm_read_256_sve(w->source, w->bytes);
    w->end = cntvct();
    return NULL;
}

static int compare_double(const void *lhs, const void *rhs)
{
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static int measure(const uint8_t *arena, size_t bytes, size_t skew, int cores,
                   int core_base, int iterations, int trials,
                   double *median, double *best)
{
    const size_t per_core = bytes / (size_t)cores;
    double rates[MAX_TRIALS];
    for (int run = 0; run <= trials; ++run) {
        pthread_t threads[MAX_CORES];
        worker workers[MAX_CORES];
        atomic_int ready, start;
        atomic_init(&ready, 0);
        atomic_init(&start, 0);
        for (int c = 0; c < cores; ++c) {
            workers[c] = (worker){
                .source = arena + (size_t)c * (per_core + skew),
                .bytes = per_core,
                .cpu = core_base + c,
                .iterations = iterations,
                .ready = &ready,
                .start = &start,
            };
            int rc = pthread_create(&threads[c], NULL, worker_main, &workers[c]);
            if (rc != 0) {
                fprintf(stderr, "pthread_create: %s\n", strerror(rc));
                return -1;
            }
        }
        while (atomic_load_explicit(&ready, memory_order_acquire) != cores)
            __asm__ volatile("yield");
        atomic_store_explicit(&start, 1, memory_order_release);
        uint64_t first = UINT64_MAX, last = 0;
        for (int c = 0; c < cores; ++c) {
            pthread_join(threads[c], NULL);
            if (workers[c].error) {
                fprintf(stderr, "CPU %d: %s\n", workers[c].cpu,
                        strerror(workers[c].error));
                return -1;
            }
            if (workers[c].begin < first) first = workers[c].begin;
            if (workers[c].end > last) last = workers[c].end;
        }
        if (run != 0) {
            double seconds = (double)(last - first) / (double)cntfrq();
            rates[run - 1] = (double)bytes * iterations / seconds / 1.0e9;
        }
    }
    qsort(rates, (size_t)trials, sizeof(rates[0]), compare_double);
    *median = rates[trials / 2];
    *best = rates[trials - 1];
    return 0;
}

static size_t parse_size(const char *text, const char *name)
{
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 0);
    if (errno || end == text || *end != '\0') {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(2);
    }
    return (size_t)value;
}

static void usage(const char *name)
{
    fprintf(stderr,
            "usage: %s [--cores N] [--core-base CPU] [--mib N] "
            "[--iterations N] [--trials N]\n"
            "          [--min-skew-kib N] [--max-skew-kib N] "
            "[--step-bytes N] [--bit-sweep]\n"
            "          [--sweep-base] [--fixed-skew-kib N] "
            "[--max-base-kib N] [--page-mode thp|base]\n",
            name);
}

static void report_mapping(const void *address)
{
    FILE *file = fopen("/proc/self/smaps", "r");
    char line[512];
    uintptr_t target = (uintptr_t)address;
    bool found = false;
    if (!file) return;
    while (fgets(line, sizeof(line), file)) {
        unsigned long long first, last;
        if (sscanf(line, "%llx-%llx", &first, &last) == 2)
            found = target >= first && target < last;
        else if (found && (!strncmp(line, "KernelPageSize:", 15) ||
                           !strncmp(line, "MMUPageSize:", 12) ||
                           !strncmp(line, "AnonHugePages:", 14)))
            printf("# %s", line);
    }
    fclose(file);
}

static void report_pfns(const uint8_t *arena, size_t bytes, int cores)
{
    const uint64_t pfn_mask = (UINT64_C(1) << 55) - 1;
    const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    const size_t per_core = bytes / (size_t)cores;
    int fd = open("/proc/self/pagemap", O_RDONLY);
    bool any_pfn = false;
    if (fd < 0) {
        printf("# pagemap=unavailable errno=%d\n", errno);
        return;
    }
    printf("# core_start_pfns=");
    for (int core = 0; core < cores; ++core) {
        uintptr_t address = (uintptr_t)(arena + (size_t)core * per_core);
        uint64_t entry = 0;
        off_t offset = (off_t)((address / page_size) * sizeof(entry));
        ssize_t got = pread(fd, &entry, sizeof(entry), offset);
        uint64_t pfn = entry & pfn_mask;
        if (core) putchar(',');
        if (got != (ssize_t)sizeof(entry) || !(entry & (UINT64_C(1) << 63)))
            printf("unreadable");
        else if (!pfn)
            printf("masked");
        else {
            printf("0x%llx", (unsigned long long)pfn);
            any_pfn = true;
        }
    }
    putchar('\n');
    printf("# physical_address_bits=%s\n", any_pfn ? "available" : "masked");
    close(fd);
}

int main(int argc, char **argv)
{
    int cores = 12, core_base = 12, iterations = 5, trials = 3;
    size_t mib = 240, min_skew_kib = 0, max_skew_kib = 64, step = 256;
    size_t fixed_skew_kib = 0, max_base_kib = 2048;
    bool bit_sweep = false, sweep_base = false, use_thp = true;
    static const struct option options[] = {
        {"cores", required_argument, NULL, 'c'},
        {"core-base", required_argument, NULL, 'b'},
        {"mib", required_argument, NULL, 'm'},
        {"iterations", required_argument, NULL, 'i'},
        {"trials", required_argument, NULL, 't'},
        {"min-skew-kib", required_argument, NULL, 1000},
        {"max-skew-kib", required_argument, NULL, 1001},
        {"step-bytes", required_argument, NULL, 1002},
        {"bit-sweep", no_argument, NULL, 1003},
        {"sweep-base", no_argument, NULL, 1004},
        {"fixed-skew-kib", required_argument, NULL, 1005},
        {"max-base-kib", required_argument, NULL, 1006},
        {"page-mode", required_argument, NULL, 1007},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    int option;
    while ((option = getopt_long(argc, argv, "c:b:m:i:t:h", options, NULL)) != -1) {
        switch (option) {
        case 'c': cores = (int)parse_size(optarg, "cores"); break;
        case 'b': core_base = (int)parse_size(optarg, "core-base"); break;
        case 'm': mib = parse_size(optarg, "mib"); break;
        case 'i': iterations = (int)parse_size(optarg, "iterations"); break;
        case 't': trials = (int)parse_size(optarg, "trials"); break;
        case 1000: min_skew_kib = parse_size(optarg, "min-skew-kib"); break;
        case 1001: max_skew_kib = parse_size(optarg, "max-skew-kib"); break;
        case 1002: step = parse_size(optarg, "step-bytes"); break;
        case 1003: bit_sweep = true; break;
        case 1004: sweep_base = true; break;
        case 1005: fixed_skew_kib = parse_size(optarg, "fixed-skew-kib"); break;
        case 1006: max_base_kib = parse_size(optarg, "max-base-kib"); break;
        case 1007:
            if (!strcmp(optarg, "thp")) use_thp = true;
            else if (!strcmp(optarg, "base")) use_thp = false;
            else { fprintf(stderr, "page mode must be thp or base\n"); return 2; }
            break;
        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 2;
        }
    }
    if (cores < 1 || cores > MAX_CORES || iterations < 1 || trials < 1 ||
        trials > MAX_TRIALS || step == 0 || step % CACHE_LINE != 0 ||
        min_skew_kib > max_skew_kib || mib > SIZE_MAX / 1048576u) {
        usage(argv[0]);
        return 2;
    }
    size_t bytes = mib * 1048576u;
    bytes -= bytes % ((size_t)cores * CACHE_LINE);
    size_t max_skew = max_skew_kib * 1024u;
    size_t fixed_skew = fixed_skew_kib * 1024u;
    size_t max_base = max_base_kib * 1024u;
    if (fixed_skew % CACHE_LINE != 0 || max_base % CACHE_LINE != 0) {
        fprintf(stderr, "base and fixed skew must be 256-byte aligned\n");
        return 2;
    }
    size_t allocation_skew = sweep_base ? fixed_skew : max_skew;
    size_t allocation_bytes = bytes + (size_t)(cores - 1) * allocation_skew +
                              (sweep_base ? max_base : 0);
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 1;
    allocation_bytes = (allocation_bytes + (size_t)page_size - 1) /
                       (size_t)page_size * (size_t)page_size;
    uint8_t *arena = mmap(NULL, allocation_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED)
        return 1;
    if (madvise(arena, allocation_bytes,
                use_thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0) {
        perror("madvise");
        munmap(arena, allocation_bytes);
        return 1;
    }
    if (pin_cpu(core_base) != 0) {
        fprintf(stderr, "cannot pin allocator to CPU %d: %s\n", core_base,
                strerror(errno));
        munmap(arena, allocation_bytes);
        return 1;
    }
    for (size_t offset = 0; offset < allocation_bytes; offset += 4096)
        arena[offset] = (uint8_t)(offset >> 12);

    printf("# A64FX HBM/L2 color sweep; one warm-up per point\n");
    printf("# cores=%d core_base=%d payload_mib=%zu per_core_mib=%.6f iterations=%d trials=%d\n",
           cores, core_base, mib, (double)(bytes / (size_t)cores) / 1048576.0,
           iterations, trials);
    printf("# arena=%p allocation_bytes=%zu page_mode=%s\n", (void *)arena,
           allocation_bytes, use_thp ? "thp" : "base");
    report_mapping(arena);
    report_pfns(arena, bytes, cores);
    printf("base_offset_bytes,skew_bytes,changed_bit,median_GBps,best_GBps,percent_of_256GBps\n");
    int rc = 0;
    if (sweep_base) {
        for (size_t base_offset = 0; base_offset <= max_base; base_offset += step) {
            double median, best;
            if (measure(arena + base_offset, bytes, fixed_skew, cores, core_base,
                        iterations, trials, &median, &best) != 0) {
                rc = 1;
                break;
            }
            printf("%zu,%zu,-1,%.3f,%.3f,%.2f\n", base_offset, fixed_skew,
                   median, best, median / 2.56);
            fflush(stdout);
            if (max_base - base_offset < step) break;
        }
    } else if (bit_sweep) {
        for (unsigned bit = 8; bit <= 20; ++bit) {
            size_t skew = (size_t)1u << bit;
            if (skew > max_skew) break;
            double median, best;
            if (measure(arena, bytes, skew, cores, core_base, iterations, trials,
                        &median, &best) != 0) { rc = 1; break; }
            printf("0,%zu,%u,%.3f,%.3f,%.2f\n", skew, bit, median, best,
                   median / 2.56);
            fflush(stdout);
        }
    } else {
        size_t first = min_skew_kib * 1024u;
        size_t last = max_skew_kib * 1024u;
        first = (first + step - 1) / step * step;
        for (size_t skew = first; skew <= last; skew += step) {
            double median, best;
            if (measure(arena, bytes, skew, cores, core_base, iterations, trials,
                        &median, &best) != 0) { rc = 1; break; }
            printf("0,%zu,-1,%.3f,%.3f,%.2f\n", skew, median, best,
                   median / 2.56);
            fflush(stdout);
            if (last - skew < step) break;
        }
    }
    munmap(arena, allocation_bytes);
    return rc;
}
