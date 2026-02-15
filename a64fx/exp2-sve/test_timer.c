/*
 * Simple timer test
 */
#include <stdio.h>
#include <stdint.h>

static inline uint64_t get_cycles(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t get_freq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

volatile int sink = 0;

int main(void) {
    uint64_t freq = get_freq();
    printf("Timer frequency: %lu Hz (%.1f MHz)\n", freq, freq / 1e6);

    /* Simple timing test */
    uint64_t start = get_cycles();
    for (int i = 0; i < 1000000; i++) {
        sink = i;
    }
    uint64_t end = get_cycles();

    printf("\nSimple loop test:\n");
    printf("  Start: %lu\n", start);
    printf("  End:   %lu\n", end);
    printf("  Diff:  %lu\n", end - start);
    printf("  Time:  %.3f ms\n", (double)(end - start) / freq * 1000.0);

    /* Multiple samples */
    printf("\nMultiple samples:\n");
    for (int i = 0; i < 5; i++) {
        start = get_cycles();
        for (int j = 0; j < 100000; j++) {
            sink = j;
        }
        end = get_cycles();
        printf("  Sample %d: start=%lu end=%lu diff=%lu\n",
               i, start, end, end - start);
    }

    return 0;
}
