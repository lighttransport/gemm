/*
 * profiler.h - Per-op timing and FLOP/IOP counting profiler
 *
 * Usage:
 *   #define PROFILER_IMPLEMENTATION
 *   #include "profiler.h"  // in exactly one translation unit
 *
 *   #include "profiler.h"  // in other files (header-only declarations)
 *
 * API:
 *   prof_begin(name, component, layer, op_type, precision)
 *   prof_end(name, flops, iops)
 *   prof_summary()     // print table to stderr
 *   prof_reset()
 *   prof_enable(0/1)   // runtime toggle
 *
 * Zero-overhead disable: #define PROF_DISABLED before include.
 */
#ifndef PROFILER_H
#define PROFILER_H

#ifdef PROF_DISABLED

#define prof_begin(name, comp, layer, op, prec) ((void)0)
#define prof_end(name, flops, iops) ((void)0)
#define prof_summary() ((void)0)
#define prof_reset() ((void)0)
#define prof_enable(x) ((void)0)

#else

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PROF_MAX_ENTRIES 2048
#define PROF_NAME_LEN 48
#define PROF_COMP_LEN 16
#define PROF_OP_LEN 16
#define PROF_PREC_LEN 8

typedef struct {
    char name[PROF_NAME_LEN];
    char component[PROF_COMP_LEN]; /* "vision", "llm", "gpu_vision", "gpu_llm" */
    int layer;                     /* -1 for non-layer ops */
    char op_type[PROF_OP_LEN];    /* "matmul", "layernorm", etc. */
    char precision[PROF_PREC_LEN]; /* "FP32", "FP16", etc. */
    double total_time_ns;
    double total_flops;
    double total_iops;
    int calls;
    /* Active timer */
    uint64_t start_ns;
} prof_entry;

typedef struct {
    prof_entry entries[PROF_MAX_ENTRIES];
    int n_entries;
    int enabled;
} prof_state;

extern prof_state g_prof;

void prof_begin_f(const char *name, const char *component, int layer,
                  const char *op_type, const char *precision);
void prof_end_f(const char *name, double flops, double iops);
void prof_summary_f(void);
void prof_reset_f(void);
void prof_enable_f(int enable);

#define prof_begin(name, comp, layer, op, prec) prof_begin_f(name, comp, layer, op, prec)
#define prof_end(name, flops, iops) prof_end_f(name, flops, iops)
#define prof_summary() prof_summary_f()
#define prof_reset() prof_reset_f()
#define prof_enable(x) prof_enable_f(x)

#ifdef __cplusplus
}
#endif

#endif /* PROF_DISABLED */

/* ======================================================================== */
#ifdef PROFILER_IMPLEMENTATION

#include <stdio.h>
#include <string.h>
#include <time.h>

prof_state g_prof = {{}, 0, 1};

static uint64_t prof_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int prof_find(const char *name, const char *component, int layer) {
    for (int i = 0; i < g_prof.n_entries; i++) {
        if (g_prof.entries[i].layer == layer &&
            strcmp(g_prof.entries[i].name, name) == 0 &&
            strcmp(g_prof.entries[i].component, component) == 0)
            return i;
    }
    return -1;
}

void prof_begin_f(const char *name, const char *component, int layer,
                  const char *op_type, const char *precision) {
    if (!g_prof.enabled) return;
    int idx = prof_find(name, component, layer);
    if (idx < 0) {
        if (g_prof.n_entries >= PROF_MAX_ENTRIES) return;
        idx = g_prof.n_entries++;
        memset(&g_prof.entries[idx], 0, sizeof(prof_entry));
        strncpy(g_prof.entries[idx].name, name, PROF_NAME_LEN - 1);
        strncpy(g_prof.entries[idx].component, component, PROF_COMP_LEN - 1);
        g_prof.entries[idx].layer = layer;
        strncpy(g_prof.entries[idx].op_type, op_type, PROF_OP_LEN - 1);
        strncpy(g_prof.entries[idx].precision, precision, PROF_PREC_LEN - 1);
    }
    g_prof.entries[idx].start_ns = prof_now_ns();
}

void prof_end_f(const char *name, double flops, double iops) {
    if (!g_prof.enabled) return;
    uint64_t end = prof_now_ns();
    /* Find most recently started entry with this name (search backwards for layer aggregation) */
    int idx = -1;
    for (int i = g_prof.n_entries - 1; i >= 0; i--) {
        if (strcmp(g_prof.entries[i].name, name) == 0 && g_prof.entries[i].start_ns != 0) {
            idx = i;
            break;
        }
    }
    if (idx < 0) return;
    uint64_t elapsed = end - g_prof.entries[idx].start_ns;
    g_prof.entries[idx].total_time_ns += (double)elapsed;
    g_prof.entries[idx].total_flops += flops;
    g_prof.entries[idx].total_iops += iops;
    g_prof.entries[idx].calls++;
    g_prof.entries[idx].start_ns = 0;
}

void prof_summary_f(void) {
    if (g_prof.n_entries == 0) return;

    fprintf(stderr, "\n=== Profiling Summary ===\n");
    fprintf(stderr, "%-24s %-12s %5s %10s %10s %10s %6s %s\n",
            "Op", "Comp", "Layer", "Time(ms)", "GFLOPs", "GFLOP/s", "Calls", "Prec");
    fprintf(stderr, "%-24s %-12s %5s %10s %10s %10s %6s %s\n",
            "------------------------", "------------", "-----",
            "----------", "----------", "----------", "------", "----");

    double grand_time = 0, grand_flops = 0, grand_iops = 0;

    for (int i = 0; i < g_prof.n_entries; i++) {
        prof_entry *e = &g_prof.entries[i];
        double ms = e->total_time_ns / 1e6;
        double gflops = e->total_flops / 1e9;
        double gflops_s = ms > 0 ? (e->total_flops / 1e9) / (ms / 1e3) : 0;
        char layer_str[8];
        if (e->layer < 0) snprintf(layer_str, sizeof(layer_str), "-");
        else snprintf(layer_str, sizeof(layer_str), "%d", e->layer);

        fprintf(stderr, "%-24s %-12s %5s %10.2f %10.4f %10.1f %6d %s\n",
                e->name, e->component, layer_str, ms, gflops, gflops_s, e->calls, e->precision);

        grand_time += e->total_time_ns;
        grand_flops += e->total_flops;
        grand_iops += e->total_iops;
    }

    /* Aggregate by op_type */
    fprintf(stderr, "\n--- By Op Type ---\n");
    fprintf(stderr, "  %-20s %10s %10s %10s\n", "Op Type", "Time(ms)", "GFLOPs", "GFLOP/s");

    /* Simple aggregation: collect unique op_types */
    char seen_ops[64][PROF_OP_LEN];
    int n_seen = 0;
    for (int i = 0; i < g_prof.n_entries; i++) {
        int found = 0;
        for (int j = 0; j < n_seen; j++) {
            if (strcmp(seen_ops[j], g_prof.entries[i].op_type) == 0) { found = 1; break; }
        }
        if (!found && n_seen < 64) {
            strncpy(seen_ops[n_seen], g_prof.entries[i].op_type, PROF_OP_LEN - 1);
            seen_ops[n_seen][PROF_OP_LEN - 1] = '\0';
            n_seen++;
        }
    }
    for (int j = 0; j < n_seen; j++) {
        double t = 0, f = 0;
        for (int i = 0; i < g_prof.n_entries; i++) {
            if (strcmp(g_prof.entries[i].op_type, seen_ops[j]) == 0) {
                t += g_prof.entries[i].total_time_ns;
                f += g_prof.entries[i].total_flops;
            }
        }
        double ms = t / 1e6;
        double gf = f / 1e9;
        double gs = ms > 0 ? gf / (ms / 1e3) : 0;
        fprintf(stderr, "  %-20s %10.2f %10.4f %10.1f\n", seen_ops[j], ms, gf, gs);
    }

    /* Aggregate by component */
    fprintf(stderr, "\n--- By Component ---\n");
    fprintf(stderr, "  %-20s %10s %10s %10s\n", "Component", "Time(ms)", "GFLOPs", "GFLOP/s");

    char seen_comps[16][PROF_COMP_LEN];
    int n_comps = 0;
    for (int i = 0; i < g_prof.n_entries; i++) {
        int found = 0;
        for (int j = 0; j < n_comps; j++) {
            if (strcmp(seen_comps[j], g_prof.entries[i].component) == 0) { found = 1; break; }
        }
        if (!found && n_comps < 16) {
            strncpy(seen_comps[n_comps], g_prof.entries[i].component, PROF_COMP_LEN - 1);
            seen_comps[n_comps][PROF_COMP_LEN - 1] = '\0';
            n_comps++;
        }
    }
    for (int j = 0; j < n_comps; j++) {
        double t = 0, f = 0;
        for (int i = 0; i < g_prof.n_entries; i++) {
            if (strcmp(g_prof.entries[i].component, seen_comps[j]) == 0) {
                t += g_prof.entries[i].total_time_ns;
                f += g_prof.entries[i].total_flops;
            }
        }
        double ms = t / 1e6;
        double gf = f / 1e9;
        double gs = ms > 0 ? gf / (ms / 1e3) : 0;
        fprintf(stderr, "  %-20s %10.2f %10.4f %10.1f\n", seen_comps[j], ms, gf, gs);
    }

    /* Grand total */
    fprintf(stderr, "\n--- Grand Total ---\n");
    double grand_ms = grand_time / 1e6;
    double grand_gf = grand_flops / 1e9;
    double grand_gi = grand_iops / 1e9;
    fprintf(stderr, "  Time:     %.2f ms\n", grand_ms);
    fprintf(stderr, "  GFLOPs:   %.4f (%.1f GFLOP/s)\n", grand_gf, grand_ms > 0 ? grand_gf / (grand_ms / 1e3) : 0);
    if (grand_gi > 0)
        fprintf(stderr, "  GIOPs:    %.4f\n", grand_gi);
    fprintf(stderr, "\n");
}

void prof_reset_f(void) {
    g_prof.n_entries = 0;
}

void prof_enable_f(int enable) {
    g_prof.enabled = enable;
}

#endif /* PROFILER_IMPLEMENTATION */
#endif /* PROFILER_H */
