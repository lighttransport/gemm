// test_timer.c - Timer diagnostic
#include <stdio.h>
#include <stdint.h>
#include <time.h>

static inline uint64_t read_timer_cntpct(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_timer_freq_cntfrq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

int main() {
    printf("=== Timer Diagnostic ===\n\n");

    // Test cntvct and cntfrq
    uint64_t freq = get_timer_freq_cntfrq();
    uint64_t t0 = read_timer_cntpct();

    // Busy wait
    volatile int sum = 0;
    for (int i = 0; i < 100000000; i++) {
        sum += i;
    }

    uint64_t t1 = read_timer_cntpct();

    printf("cntvct_el0 method:\n");
    printf("  freq = %lu\n", freq);
    printf("  t0 = %lu\n", t0);
    printf("  t1 = %lu\n", t1);
    printf("  t1-t0 = %lu\n", t1 - t0);
    printf("  time = %.6f seconds\n", (double)(t1 - t0) / (double)freq);

    // Test clock_gettime
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    sum = 0;
    for (int i = 0; i < 100000000; i++) {
        sum += i;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    double time_clock = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) / 1e9;

    printf("\nclock_gettime method:\n");
    printf("  time = %.6f seconds\n", time_clock);

    printf("\nsum = %d (to prevent optimization)\n", sum);

    return 0;
}
