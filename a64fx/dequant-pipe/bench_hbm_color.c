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
typedef enum { PAGE_THP, PAGE_BASE, PAGE_XOS } page_mode;

extern void hbm_read_256_sve(const uint8_t *, size_t);
extern void q8_stream_256_sve(const uint8_t *, size_t);
#define DECLARE_COMPUTE(kind, n) extern void hbm_read_256_##kind##_##n##_sve(const uint8_t *, size_t)
DECLARE_COMPUTE(sdot, 4); DECLARE_COMPUTE(sdot, 8); DECLARE_COMPUTE(sdot, 12);
DECLARE_COMPUTE(sdot, 16); DECLARE_COMPUTE(sdot, 24); DECLARE_COMPUTE(sdot, 32);
DECLARE_COMPUTE(sdot, 40); DECLARE_COMPUTE(sdot, 48); DECLARE_COMPUTE(sdot, 64);
DECLARE_COMPUTE(sdot, 52); DECLARE_COMPUTE(sdot, 56); DECLARE_COMPUTE(sdot, 60);
DECLARE_COMPUTE(fmla, 4); DECLARE_COMPUTE(fmla, 8); DECLARE_COMPUTE(fmla, 12);
DECLARE_COMPUTE(fmla, 16); DECLARE_COMPUTE(fmla, 24); DECLARE_COMPUTE(fmla, 32);
DECLARE_COMPUTE(fmla, 40); DECLARE_COMPUTE(fmla, 48); DECLARE_COMPUTE(fmla, 64);
DECLARE_COMPUTE(fmla, 52); DECLARE_COMPUTE(fmla, 56); DECLARE_COMPUTE(fmla, 60);
typedef void (*stream_fn)(const uint8_t *, size_t);

typedef struct {
    const uint8_t *source;
    size_t bytes;
    int cpu;
    int iterations;
    stream_fn stream;
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
            w->stream(w->source, w->bytes);
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
                   stream_fn stream,
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
                .stream = stream,
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
            "[--max-base-kib N] [--page-mode thp|base|xos]\n"
            "          [--op read|sdot|fmla] [--ops-per-line N] [--paired-baseline]\n",
            name);
}

static stream_fn select_stream(const char *op, int count)
{
    if (!strcmp(op, "read") && count == 0) return hbm_read_256_sve;
    if (!strcmp(op, "q8read") && count == 0) return q8_stream_256_sve;
#define SELECT(kind, n) if (!strcmp(op, #kind) && count == n) return hbm_read_256_##kind##_##n##_sve
    SELECT(sdot, 4); SELECT(sdot, 8); SELECT(sdot, 12); SELECT(sdot, 16);
    SELECT(sdot, 24); SELECT(sdot, 32); SELECT(fmla, 4); SELECT(fmla, 8);
    SELECT(fmla, 12); SELECT(fmla, 16); SELECT(fmla, 24); SELECT(fmla, 32);
    SELECT(sdot, 40); SELECT(sdot, 48); SELECT(sdot, 64);
    SELECT(fmla, 40); SELECT(fmla, 48); SELECT(fmla, 64);
    SELECT(sdot, 52); SELECT(sdot, 56); SELECT(sdot, 60);
    SELECT(fmla, 52); SELECT(fmla, 56); SELECT(fmla, 60);
    return NULL;
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
    bool bit_sweep = false, sweep_base = false;
    page_mode pages = PAGE_THP;
    const char *op_name = "read";
    int ops_per_line = 0;
    bool paired_baseline = false;
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
        {"op", required_argument, NULL, 1008},
        {"ops-per-line", required_argument, NULL, 1009},
        {"paired-baseline", no_argument, NULL, 1010},
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
            if (!strcmp(optarg, "thp")) pages = PAGE_THP;
            else if (!strcmp(optarg, "base")) pages = PAGE_BASE;
            else if (!strcmp(optarg, "xos")) pages = PAGE_XOS;
            else { fprintf(stderr, "page mode must be thp, base, or xos\n"); return 2; }
            break;
        case 1008: op_name = optarg; break;
        case 1009: ops_per_line = (int)parse_size(optarg, "ops-per-line"); break;
        case 1010: paired_baseline = true; break;
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
    stream_fn stream = select_stream(op_name, ops_per_line);
    if (!stream) {
        fprintf(stderr, "unsupported --op/--ops-per-line combination\n");
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
    uint8_t *arena;
    if (pages == PAGE_XOS) {
        if (posix_memalign((void **)&arena, 2u * 1024u * 1024u,
                           allocation_bytes) != 0)
            return 1;
    } else {
        arena = mmap(NULL, allocation_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (arena == MAP_FAILED)
            return 1;
        if (madvise(arena, allocation_bytes,
                    pages == PAGE_THP ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0) {
            perror("madvise");
            munmap(arena, allocation_bytes);
            return 1;
        }
    }
    if (pin_cpu(core_base) != 0) {
        fprintf(stderr, "cannot pin allocator to CPU %d: %s\n", core_base,
                strerror(errno));
        if (pages == PAGE_XOS) free(arena); else munmap(arena, allocation_bytes);
        return 1;
    }
    for (size_t offset = 0; offset < allocation_bytes; offset += 4096)
        arena[offset] = (uint8_t)(offset >> 12);

    printf("# A64FX HBM/L2 color sweep; one warm-up per point\n");
    printf("# cores=%d core_base=%d payload_mib=%zu per_core_mib=%.6f iterations=%d trials=%d\n",
           cores, core_base, mib, (double)(bytes / (size_t)cores) / 1048576.0,
           iterations, trials);
    printf("# op=%s ops_per_256B_line=%d\n", op_name, ops_per_line);
    const char *page_name = pages == PAGE_XOS ? "xos" :
                            pages == PAGE_THP ? "thp" : "base";
    printf("# arena=%p allocation_bytes=%zu page_mode=%s\n", (void *)arena,
           allocation_bytes, page_name);
    report_mapping(arena);
    report_pfns(arena, bytes, cores);
    printf("base_offset_bytes,skew_bytes,changed_bit,median_GBps,best_GBps,percent_of_256GBps,paired_read_GBps,percent_of_paired\n");
    int rc = 0;
    if (sweep_base) {
        for (size_t base_offset = 0; base_offset <= max_base; base_offset += step) {
            double median, best;
            double read_median = 0.0, read_best;
            if (paired_baseline && measure(arena + base_offset, bytes, fixed_skew,
                    cores, core_base, iterations, trials, hbm_read_256_sve,
                    &read_median, &read_best) != 0) { rc = 1; break; }
            if (measure(arena + base_offset, bytes, fixed_skew, cores, core_base,
                        iterations, trials, stream, &median, &best) != 0) {
                rc = 1;
                break;
            }
            printf("%zu,%zu,-1,%.3f,%.3f,%.2f,%.3f,%.2f\n", base_offset, fixed_skew,
                   median, best, median / 2.56, read_median,
                   read_median ? 100.0 * median / read_median : 0.0);
            fflush(stdout);
            if (max_base - base_offset < step) break;
        }
    } else if (bit_sweep) {
        for (unsigned bit = 8; bit <= 20; ++bit) {
            size_t skew = (size_t)1u << bit;
            if (skew > max_skew) break;
            double median, best;
            double read_median = 0.0, read_best;
            if (paired_baseline && measure(arena, bytes, skew, cores, core_base,
                    iterations, trials, hbm_read_256_sve, &read_median,
                    &read_best) != 0) { rc = 1; break; }
            if (measure(arena, bytes, skew, cores, core_base, iterations, trials, stream,
                        &median, &best) != 0) { rc = 1; break; }
            printf("0,%zu,%u,%.3f,%.3f,%.2f,%.3f,%.2f\n", skew, bit, median, best,
                   median / 2.56, read_median,
                   read_median ? 100.0 * median / read_median : 0.0);
            fflush(stdout);
        }
    } else {
        size_t first = min_skew_kib * 1024u;
        size_t last = max_skew_kib * 1024u;
        first = (first + step - 1) / step * step;
        for (size_t skew = first; skew <= last; skew += step) {
            double median, best;
            double read_median = 0.0, read_best;
            if (paired_baseline && measure(arena, bytes, skew, cores, core_base,
                    iterations, trials, hbm_read_256_sve, &read_median,
                    &read_best) != 0) { rc = 1; break; }
            if (measure(arena, bytes, skew, cores, core_base, iterations, trials, stream,
                        &median, &best) != 0) { rc = 1; break; }
            printf("0,%zu,-1,%.3f,%.3f,%.2f,%.3f,%.2f\n", skew, median, best,
                   median / 2.56, read_median,
                   read_median ? 100.0 * median / read_median : 0.0);
            fflush(stdout);
            if (last - skew < step) break;
        }
    }
    if (pages == PAGE_XOS) free(arena); else munmap(arena, allocation_bytes);
    return rc;
}
