#define _GNU_SOURCE
#include "q8_k128.h"

#include <getopt.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_CORES = 48 };

typedef struct {
    const uint8_t *weights;
    size_t records;
    int cpu, iterations;
    atomic_int *ready, *start;
    uint64_t begin, end;
    float checksum;
} worker;

static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v)); return v; }
static uint64_t frequency(void) { uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }

static int pin_cpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void *run_worker(void *arg)
{
    worker *w = (worker *)arg;
    float x[Q8_K128_K], y[Q8_K128_N];
    for (int k = 0; k < 128; k++) x[k] = (float)(k % 13 - 6) / 17.0f;
    if (pin_cpu(w->cpu)) return NULL;
    atomic_fetch_add_explicit(w->ready, 1, memory_order_release);
    while (!atomic_load_explicit(w->start, memory_order_acquire)) __asm__ volatile("yield");
    w->begin = ticks();
    for (int it = 0; it < w->iterations; it++) {
        for (size_t r = 0; r < w->records; r++) {
            memset(y, 0, sizeof(y));
            q8_k128_f32(w->weights + r * Q8_K128_RECORD_BYTES, x, y);
            w->checksum += y[(r + (size_t)it) % Q8_K128_N];
        }
    }
    w->end = ticks();
    return NULL;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    int cores = 12, core_base = 12, iterations = 8, trials = 5;
    size_t mib = 240;
    double target = 200.0;
    static const struct option opts[] = {
        {"cores", 1, 0, 'c'}, {"core-base", 1, 0, 'p'}, {"iterations", 1, 0, 'i'},
        {"trials", 1, 0, 't'}, {"mib", 1, 0, 'm'}, {"target-gbps", 1, 0, 'g'},
        {0, 0, 0, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "c:p:i:t:m:g:", opts, NULL)) != -1) {
        if (opt == 'c') cores = atoi(optarg);
        else if (opt == 'p') core_base = atoi(optarg);
        else if (opt == 'i') iterations = atoi(optarg);
        else if (opt == 't') trials = atoi(optarg);
        else if (opt == 'm') mib = strtoull(optarg, NULL, 0);
        else if (opt == 'g') target = strtod(optarg, NULL);
        else return 2;
    }
    if (cores < 1 || cores > MAX_CORES || core_base < 0 ||
        core_base + cores > CPU_SETSIZE || iterations < 1 || trials < 1 || trials > 31 || !mib)
        return 2;
    size_t records = mib * 1024u * 1024u / Q8_K128_RECORD_BYTES;
    records -= records % (size_t)cores;
    size_t bytes = records * Q8_K128_RECORD_BYTES;
    uint8_t *weights = NULL;
    if (!records || posix_memalign((void **)&weights, 2u * 1024u * 1024u, bytes)) return 1;
    uint32_t rng = 1;
    for (size_t r = 0; r < records; r++) {
        uint8_t *record = weights + r * Q8_K128_RECORD_BYTES;
        uint16_t one = 0x3c00;
        for (int i = 0; i < Q8_K128_SCALE_BYTES / 2; i++) ((uint16_t *)record)[i] = one;
        for (int i = 0; i < Q8_K128_WEIGHT_BYTES; i++) {
            rng = rng * 1664525u + 1013904223u;
            record[Q8_K128_SCALE_BYTES + i] = (uint8_t)(rng >> 24);
        }
    }
    double values[32];
    for (int tr = 0; tr <= trials; tr++) {
        pthread_t tids[MAX_CORES];
        worker ws[MAX_CORES];
        atomic_int ready, start;
        atomic_init(&ready, 0); atomic_init(&start, 0);
        size_t per = records / (size_t)cores;
        for (int c = 0; c < cores; c++) {
            ws[c] = (worker){weights + (size_t)c * per * Q8_K128_RECORD_BYTES,
                per, core_base + c, iterations, &ready, &start, 0, 0, 0};
            if (pthread_create(&tids[c], NULL, run_worker, &ws[c])) return 1;
        }
        while (atomic_load_explicit(&ready, memory_order_acquire) != cores) __asm__ volatile("yield");
        atomic_store_explicit(&start, 1, memory_order_release);
        uint64_t first = UINT64_MAX, last = 0;
        float checksum = 0;
        for (int c = 0; c < cores; c++) {
            pthread_join(tids[c], NULL);
            if (ws[c].begin < first) first = ws[c].begin;
            if (ws[c].end > last) last = ws[c].end;
            checksum += ws[c].checksum;
        }
        double seconds = (double)(last - first) / frequency();
        double gbps = (double)bytes * iterations / seconds / 1e9;
        if (tr) values[tr - 1] = gbps;
        printf("trial=%d stored_bytes=%zu seconds=%.6f GB/s=%.3f checksum=%a\n",
               tr, bytes, seconds, gbps, checksum);
    }
    qsort(values, (size_t)trials, sizeof(values[0]), cmp_double);
    double median = values[trials / 2];
    printf("q8_k128 median=%.3f GB/s target=%.1f GB/s %s\n",
           median, target, median >= target ? "PASS" : "FAIL");
    free(weights);
    return median >= target ? 0 : 1;
}
