/*
 * pmu.c - Lightweight PMU Counter Library for A64FX
 *
 * Uses Linux perf_event_open() syscall directly - no library dependencies.
 * Supports grouped counters with synchronized start/stop.
 */

#include "pmu.h"
#include "pmu_events.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>

/* ========================================================================
 * perf_event_open syscall wrapper
 * ======================================================================== */
static long
perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu,
                int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

/* ========================================================================
 * Event name lookup (binary search on sorted table)
 * ======================================================================== */
const char *pmu_event_name(uint16_t code)
{
    int lo = 0, hi = (int)PMU_EVENT_TABLE_SIZE - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (pmu_event_table[mid].code == code)
            return pmu_event_table[mid].name;
        else if (pmu_event_table[mid].code < code)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return "UNKNOWN";
}

/* ========================================================================
 * Predefined event groups
 * ======================================================================== */

/* BASIC: cycles, instructions, L1 cache, stalls */
const uint16_t PMU_GROUP_BASIC[] = {
    PMU_CPU_CYCLES, PMU_INST_RETIRED,
    PMU_L1D_CACHE, PMU_L1D_CACHE_REFILL,
    PMU_LD_COMP_WAIT, PMU_LD_COMP_WAIT_L2_MISS
};
const int PMU_GROUP_BASIC_N = 6;

/* L2: L2 cache events */
const uint16_t PMU_GROUP_L2[] = {
    PMU_CPU_CYCLES, PMU_L2D_CACHE,
    PMU_L2D_CACHE_REFILL, PMU_L2D_CACHE_WB,
    PMU_L2_MISS_WAIT, PMU_LD_COMP_WAIT_L1_MISS
};
const int PMU_GROUP_L2_N = 6;

/* SECTOR: sector cache tag events */
const uint16_t PMU_GROUP_SECTOR[] = {
    PMU_CPU_CYCLES,
    PMU_L1_PIPE0_VAL, PMU_L1_PIPE1_VAL,
    PMU_L1_PIPE0_VAL_IU_TAG_ADRS_SCE, PMU_L1_PIPE1_VAL_IU_TAG_ADRS_SCE,
    PMU_L1_PIPE0_VAL_IU_NOT_SEC0
};
const int PMU_GROUP_SECTOR_N = 6;

/* PREFETCH: HW prefetch breakdown */
const uint16_t PMU_GROUP_PREFETCH[] = {
    PMU_CPU_CYCLES,
    PMU_L1D_CACHE_REFILL_DM, PMU_L1D_CACHE_REFILL_HWPRF,
    PMU_L2D_CACHE_REFILL_DM, PMU_L2D_CACHE_REFILL_HWPRF,
    PMU_L1_MISS_WAIT
};
const int PMU_GROUP_PREFETCH_N = 6;

/* ENERGY: energy counters */
const uint16_t PMU_GROUP_ENERGY[] = {
    PMU_CPU_CYCLES, PMU_INST_RETIRED,
    PMU_EA_CORE, PMU_EA_L2,
    PMU_EA_MEMORY, PMU_STALL_BACKEND
};
const int PMU_GROUP_ENERGY_N = 6;

/* FP: floating-point / SVE */
const uint16_t PMU_GROUP_FP[] = {
    PMU_CPU_CYCLES, PMU_INST_RETIRED,
    PMU_SVE_INST_RETIRED, PMU_FP_SPEC,
    PMU_FP_FMA_SPEC, PMU_SIMD_INST_RETIRED
};
const int PMU_GROUP_FP_N = 6;

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

int pmu_init(pmu_ctx_t *ctx, const uint16_t *events, int n)
{
    if (n < 1 || n > PMU_MAX_EVENTS) {
        fprintf(stderr, "pmu_init: n=%d out of range [1,%d]\n",
                n, PMU_MAX_EVENTS);
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->n = n;
    for (int i = 0; i < n; i++) {
        ctx->events[i] = events[i];
        ctx->fds[i] = -1;
    }

    int leader_fd = -1;

    for (int i = 0; i < n; i++) {
        struct perf_event_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.size = sizeof(attr);
        attr.type = PERF_TYPE_RAW;
        attr.config = events[i];
        attr.disabled = (i == 0) ? 1 : 0;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;

        int fd = (int)perf_event_open(&attr, 0, -1, leader_fd, 0);
        if (fd < 0) {
            fprintf(stderr, "pmu_init: perf_event_open failed for "
                    "event 0x%04x (%s): ",
                    events[i], pmu_event_name(events[i]));
            perror("");
            /* Clean up already-opened fds */
            for (int j = 0; j < i; j++) {
                close(ctx->fds[j]);
                ctx->fds[j] = -1;
            }
            return -1;
        }

        ctx->fds[i] = fd;
        if (i == 0) leader_fd = fd;
    }

    return 0;
}

void pmu_fini(pmu_ctx_t *ctx)
{
    for (int i = 0; i < ctx->n; i++) {
        if (ctx->fds[i] >= 0) {
            close(ctx->fds[i]);
            ctx->fds[i] = -1;
        }
    }
}

/* ========================================================================
 * Measurement
 * ======================================================================== */

void pmu_start(pmu_ctx_t *ctx)
{
    /* Reset all counters in the group */
    ioctl(ctx->fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    /* Record start timestamp */
    ctx->t_start = pmu_rdtsc();
    /* Enable all counters in the group */
    ioctl(ctx->fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

void pmu_stop(pmu_ctx_t *ctx)
{
    /* Disable all counters in the group */
    ioctl(ctx->fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    /* Record elapsed cycles */
    ctx->cycles = pmu_rdtsc() - ctx->t_start;
}

void pmu_read(pmu_ctx_t *ctx)
{
    for (int i = 0; i < ctx->n; i++) {
        uint64_t val = 0;
        if (read(ctx->fds[i], &val, sizeof(val)) != sizeof(val)) {
            fprintf(stderr, "pmu_read: read failed for event %d\n", i);
            ctx->values[i] = 0;
        } else {
            ctx->values[i] = val;
        }
    }
}

void pmu_reset(pmu_ctx_t *ctx)
{
    ioctl(ctx->fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    memset(ctx->values, 0, sizeof(ctx->values));
    ctx->cycles = 0;
    ctx->t_start = 0;
}

/* ========================================================================
 * Access results
 * ======================================================================== */

uint64_t pmu_value(const pmu_ctx_t *ctx, int idx)
{
    if (idx < 0 || idx >= ctx->n) return 0;
    return ctx->values[idx];
}

uint64_t pmu_cycles(const pmu_ctx_t *ctx)
{
    return ctx->cycles;
}

/* ========================================================================
 * Helper: find event index by code, returns -1 if not present
 * ======================================================================== */
static int find_event(const pmu_ctx_t *ctx, uint16_t code)
{
    for (int i = 0; i < ctx->n; i++)
        if (ctx->events[i] == code) return i;
    return -1;
}

/* ========================================================================
 * Output
 * ======================================================================== */

void pmu_print(const pmu_ctx_t *ctx, const char *label)
{
    printf("=== PMU: %s ===\n", label ? label : "(no label)");
    printf("  %-32s %'20lu  (cntvct_el0 delta)\n",
           "wall_cycles", (unsigned long)ctx->cycles);

    for (int i = 0; i < ctx->n; i++) {
        printf("  %-32s %'20lu  (0x%04x)\n",
               pmu_event_name(ctx->events[i]),
               (unsigned long)ctx->values[i],
               ctx->events[i]);
    }

    /* Derived metrics */
    int idx_cycles = find_event(ctx, PMU_CPU_CYCLES);
    int idx_inst   = find_event(ctx, PMU_INST_RETIRED);
    int idx_l1d    = find_event(ctx, PMU_L1D_CACHE);
    int idx_l1dr   = find_event(ctx, PMU_L1D_CACHE_REFILL);
    int idx_l2d    = find_event(ctx, PMU_L2D_CACHE);
    int idx_l2dr   = find_event(ctx, PMU_L2D_CACHE_REFILL);
    int idx_l2miss = find_event(ctx, PMU_LD_COMP_WAIT_L2_MISS);
    int idx_pipe0  = find_event(ctx, PMU_L1_PIPE0_VAL);
    int idx_pipe1  = find_event(ctx, PMU_L1_PIPE1_VAL);
    int idx_sce0   = find_event(ctx, PMU_L1_PIPE0_VAL_IU_TAG_ADRS_SCE);
    int idx_sce1   = find_event(ctx, PMU_L1_PIPE1_VAL_IU_TAG_ADRS_SCE);
    int idx_ea_c   = find_event(ctx, PMU_EA_CORE);
    int idx_ea_l2  = find_event(ctx, PMU_EA_L2);
    int idx_ea_mem = find_event(ctx, PMU_EA_MEMORY);

    printf("  --- derived ---\n");

    if (idx_cycles >= 0 && idx_inst >= 0 && ctx->values[idx_cycles] > 0) {
        double ipc = (double)ctx->values[idx_inst] /
                     (double)ctx->values[idx_cycles];
        printf("  %-32s %20.3f\n", "IPC", ipc);
    }

    if (idx_l1d >= 0 && idx_l1dr >= 0 && ctx->values[idx_l1d] > 0) {
        double rate = (double)ctx->values[idx_l1dr] /
                      (double)ctx->values[idx_l1d];
        printf("  %-32s %19.2f%%\n", "L1D miss rate", rate * 100.0);
    }

    if (idx_l2d >= 0 && idx_l2dr >= 0 && ctx->values[idx_l2d] > 0) {
        double rate = (double)ctx->values[idx_l2dr] /
                      (double)ctx->values[idx_l2d];
        printf("  %-32s %19.2f%%\n", "L2D miss rate", rate * 100.0);
    }

    if (idx_cycles >= 0 && idx_l2miss >= 0 && ctx->values[idx_cycles] > 0) {
        double ratio = (double)ctx->values[idx_l2miss] /
                       (double)ctx->values[idx_cycles];
        printf("  %-32s %19.2f%%\n", "Mem stall ratio", ratio * 100.0);
    }

    if (idx_pipe0 >= 0 && idx_pipe1 >= 0 &&
        idx_sce0 >= 0 && idx_sce1 >= 0) {
        uint64_t pipe_total = ctx->values[idx_pipe0] + ctx->values[idx_pipe1];
        uint64_t sce_total  = ctx->values[idx_sce0]  + ctx->values[idx_sce1];
        if (pipe_total > 0) {
            double usage = (double)sce_total / (double)pipe_total;
            printf("  %-32s %19.2f%%\n", "SCE usage", usage * 100.0);
        }
    }

    if (idx_ea_c >= 0 && idx_ea_l2 >= 0 && idx_ea_mem >= 0) {
        /* A64FX (2.0/2.2 GHz, 48 cores): core=8nJ, L2=32nJ, mem=256nJ */
        double nj = (double)ctx->values[idx_ea_c]   * 8.0 +
                    (double)ctx->values[idx_ea_l2]  * 32.0 +
                    (double)ctx->values[idx_ea_mem] * 256.0;
        printf("  %-32s %17.1f uJ\n", "Energy (est.)", nj / 1000.0);
    }

    printf("\n");
}

void pmu_print_csv_header(const pmu_ctx_t *ctx)
{
    printf("label,wall_cycles");
    for (int i = 0; i < ctx->n; i++)
        printf(",%s", pmu_event_name(ctx->events[i]));
    printf("\n");
}

void pmu_print_csv(const pmu_ctx_t *ctx, const char *label)
{
    printf("%s,%lu", label ? label : "", (unsigned long)ctx->cycles);
    for (int i = 0; i < ctx->n; i++)
        printf(",%lu", (unsigned long)ctx->values[i]);
    printf("\n");
}
