/*
 * pmu.h - Lightweight PMU Counter Library for A64FX
 *
 * Zero-dependency library for reading ARM PMU hardware counters
 * via Linux perf_event_open() syscall. Supports up to 6 hardware
 * counters per measurement (A64FX PMUv3 limit).
 *
 * Usage:
 *   pmu_ctx_t ctx;
 *   pmu_init(&ctx, PMU_GROUP_BASIC, PMU_GROUP_BASIC_N);
 *   pmu_start(&ctx);
 *   // ... kernel under test ...
 *   pmu_stop(&ctx);
 *   pmu_read(&ctx);
 *   pmu_print(&ctx, "my_kernel");
 *   pmu_fini(&ctx);
 */
#ifndef PMU_H
#define PMU_H

#include <stdint.h>

#define PMU_MAX_EVENTS 6

typedef struct pmu_ctx {
    int       fds[PMU_MAX_EVENTS];   /* perf_event file descriptors       */
    uint16_t  events[PMU_MAX_EVENTS];/* event codes                       */
    uint64_t  values[PMU_MAX_EVENTS];/* counter values after pmu_read()   */
    uint64_t  cycles;                /* cycle count (cntvct_el0 delta)     */
    uint64_t  t_start;               /* cntvct_el0 at pmu_start()         */
    int       n;                     /* number of events                  */
} pmu_ctx_t;

/* Lifecycle */
int  pmu_init(pmu_ctx_t *ctx, const uint16_t *events, int n);
void pmu_fini(pmu_ctx_t *ctx);

/* Measurement */
void pmu_start(pmu_ctx_t *ctx);
void pmu_stop(pmu_ctx_t *ctx);
void pmu_read(pmu_ctx_t *ctx);
void pmu_reset(pmu_ctx_t *ctx);

/* Output */
void pmu_print(const pmu_ctx_t *ctx, const char *label);
void pmu_print_csv(const pmu_ctx_t *ctx, const char *label);
void pmu_print_csv_header(const pmu_ctx_t *ctx);

/* Access results */
uint64_t pmu_value(const pmu_ctx_t *ctx, int idx);
uint64_t pmu_cycles(const pmu_ctx_t *ctx);

/* Timer (no perf dependency, always available) */
static inline uint64_t pmu_rdtsc(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

static inline uint64_t pmu_freq(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}

/* Event name lookup */
const char *pmu_event_name(uint16_t code);

/* Predefined event groups (max 6 events each) */
extern const uint16_t PMU_GROUP_BASIC[];
extern const uint16_t PMU_GROUP_L2[];
extern const uint16_t PMU_GROUP_SECTOR[];
extern const uint16_t PMU_GROUP_PREFETCH[];
extern const uint16_t PMU_GROUP_ENERGY[];
extern const uint16_t PMU_GROUP_FP[];
extern const int PMU_GROUP_BASIC_N;
extern const int PMU_GROUP_L2_N;
extern const int PMU_GROUP_SECTOR_N;
extern const int PMU_GROUP_PREFETCH_N;
extern const int PMU_GROUP_ENERGY_N;
extern const int PMU_GROUP_FP_N;

#endif /* PMU_H */
