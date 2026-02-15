// bench_mem_pattern.c
// Benchmark driver for memory access pattern analysis
//
// Compares NOP kernel (memory-only) vs original compute kernel
// to isolate memory vs compute bottlenecks in fused attention.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// External kernel declarations
extern void mem_access_kernel_nop(
    const int8_t* Q,
    const int8_t* K_int,
    const int8_t* V_t_int,
    int32_t* O
);

extern void mem_access_fused_d256_c(
    const int8_t* Q,
    const int8_t* K_int,
    const int8_t* V_t,
    int32_t* O
);

extern void mem_access_fused_d256_debug(
    const int8_t* Q,
    const int8_t* K_int,
    const int8_t* V_t,
    int32_t* O,
    int verbose
);

// High-precision timer counter
static inline uint64_t rdtimer(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

// Get timer counter frequency
static inline uint64_t get_timer_freq(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

// CPU frequency in Hz (set by job submission -L "freq=2000")
#define CPU_FREQ_HZ 2000000000UL

// Memory traffic analysis constants
// Phase 1 (Q@K^T): 64 iters x (4x64B K + 4x4B Q) = 64 x 272 = 17,408 bytes loaded
// Phase 2 (Quant): 4x64B = 256 bytes stored to stack
// Phase 3 (P@V):   4 D-tiles x 16 N_groups x (4x64B V + 4x4B P) = 4 x 16 x 272 = 17,408 bytes loaded
//                  4 D-tiles x (16x64B O) = 4 x 1024 = 4,096 bytes stored
// Total: 34,816 bytes loaded, 4,352 bytes stored

#define BYTES_LOADED_PHASE1  (64 * (4*64 + 4*4))      // 17,408
#define BYTES_STORED_PHASE2  (4 * 64)                  // 256
#define BYTES_LOADED_PHASE3  (4 * 16 * (4*64 + 4*4))  // 17,408
#define BYTES_STORED_PHASE3  (4 * 16 * 64)            // 4,096

#define TOTAL_BYTES_LOADED   (BYTES_LOADED_PHASE1 + BYTES_LOADED_PHASE3)  // 34,816
#define TOTAL_BYTES_STORED   (BYTES_STORED_PHASE2 + BYTES_STORED_PHASE3)  // 4,352

// Theoretical minimum cycles (L1 fully resident)
// Load: 34,816 / 128 byte/cycle = 272 cycles
// Store: 4,352 / 64 byte/cycle = 68 cycles
// => Memory-bound at ~272 cycles
#define THEORETICAL_MIN_CYCLES_LOAD  (TOTAL_BYTES_LOADED / 128.0)
#define THEORETICAL_MIN_CYCLES_STORE (TOTAL_BYTES_STORED / 64.0)

// Buffer sizes
#define Q_SIZE      (4 * 256)           // [4, 256] int8
#define K_INT_SIZE  (64 * 64 * 4)       // [64, 64, 4] int8 interleaved
#define V_T_SIZE    (16 * 256 * 4)      // [16, 256, 4] int8 interleaved
#define O_SIZE      (4 * 256)           // [4, 256] int32

void print_memory_analysis(void) {
    printf("=== Memory Traffic Analysis ===\n");
    printf("\nPhase 1 (Q@K^T):\n");
    printf("  K loads:  64 iters x 4x64B = %d bytes\n", 64 * 4 * 64);
    printf("  Q loads:  64 iters x 4x4B  = %d bytes\n", 64 * 4 * 4);
    printf("  Total:    %d bytes\n", BYTES_LOADED_PHASE1);

    printf("\nPhase 2 (Quantize):\n");
    printf("  P stores: 4x64B = %d bytes (stack)\n", BYTES_STORED_PHASE2);

    printf("\nPhase 3 (P@V):\n");
    printf("  V loads:  4 D-tiles x 16 N-groups x 4x64B = %d bytes\n", 4 * 16 * 4 * 64);
    printf("  P loads:  4 D-tiles x 16 N-groups x 4x4B  = %d bytes\n", 4 * 16 * 4 * 4);
    printf("  O stores: 4 D-tiles x 16x64B = %d bytes\n", BYTES_STORED_PHASE3);
    printf("  Total loaded: %d bytes\n", BYTES_LOADED_PHASE3);

    printf("\nTotal Memory Traffic:\n");
    printf("  Bytes loaded:  %d (%.1f KB)\n", TOTAL_BYTES_LOADED, TOTAL_BYTES_LOADED/1024.0);
    printf("  Bytes stored:  %d (%.1f KB)\n", TOTAL_BYTES_STORED, TOTAL_BYTES_STORED/1024.0);

    printf("\nTheoretical Minimum (L1 resident, 128 B/cycle load, 64 B/cycle store):\n");
    printf("  Load-bound:  %.1f cycles\n", THEORETICAL_MIN_CYCLES_LOAD);
    printf("  Store-bound: %.1f cycles\n", THEORETICAL_MIN_CYCLES_STORE);
    printf("  Limiting:    %.1f cycles (load-bound)\n\n",
           THEORETICAL_MIN_CYCLES_LOAD > THEORETICAL_MIN_CYCLES_STORE ?
           THEORETICAL_MIN_CYCLES_LOAD : THEORETICAL_MIN_CYCLES_STORE);
}

void print_working_set_analysis(void) {
    printf("=== Working Set Analysis ===\n");
    printf("  Q:     %d bytes (%.1f KB)\n", Q_SIZE, Q_SIZE/1024.0);
    printf("  K_int: %d bytes (%.1f KB)\n", K_INT_SIZE, K_INT_SIZE/1024.0);
    printf("  V_t:   %d bytes (%.1f KB)\n", V_T_SIZE, V_T_SIZE/1024.0);
    printf("  O:     %d bytes (%.1f KB)\n", O_SIZE * 4, O_SIZE * 4/1024.0);
    printf("  P:     256 bytes (stack)\n");
    int total = Q_SIZE + K_INT_SIZE + V_T_SIZE + O_SIZE * 4 + 256;
    printf("  Total: %d bytes (%.1f KB)\n\n", total, total/1024.0);
    printf("  L1D cache: 64 KB -> %s in L1\n",
           total <= 65536 ? "Fits" : "Does NOT fit");
}

int main(int argc, char* argv[]) {
    int warmup_iters = 100;
    int bench_iters = 1000;
    int verbose = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            bench_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warmup_iters = atoi(argv[++i]);
        }
    }

    printf("Memory Access Pattern Analysis for Fused (Q@K^T)@V\n");
    printf("==================================================\n\n");

    // Print analysis
    print_memory_analysis();
    print_working_set_analysis();

    // Allocate aligned buffers
    int8_t* Q = aligned_alloc(256, Q_SIZE);
    int8_t* K_int = aligned_alloc(256, K_INT_SIZE);
    int8_t* V_t = aligned_alloc(256, V_T_SIZE);
    int32_t* O = aligned_alloc(256, O_SIZE * sizeof(int32_t));

    if (!Q || !K_int || !V_t || !O) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize with non-zero data
    for (int i = 0; i < Q_SIZE; i++) Q[i] = (int8_t)(i & 0x7F);
    for (int i = 0; i < K_INT_SIZE; i++) K_int[i] = (int8_t)(i & 0x7F);
    for (int i = 0; i < V_T_SIZE; i++) V_t[i] = (int8_t)(i & 0x7F);
    memset(O, 0, O_SIZE * sizeof(int32_t));

    // Debug: print addresses
    if (verbose) {
        printf("=== Address Debug ===\n");
        mem_access_fused_d256_debug(Q, K_int, V_t, O, 1);
        printf("\n");
    }

    uint64_t timer_freq = get_timer_freq();
    double cycle_scale = (double)CPU_FREQ_HZ / timer_freq;  // Convert timer ticks to CPU cycles

    printf("Timer frequency:   %lu Hz (%.0f MHz)\n", timer_freq, timer_freq / 1e6);
    printf("CPU frequency:     %lu Hz (%.0f MHz)\n", CPU_FREQ_HZ, CPU_FREQ_HZ / 1e6);
    printf("Tick-to-cycle:     %.1fx\n\n", cycle_scale);

    // =========================================================================
    // Benchmark ASM NOP kernel
    // =========================================================================
    printf("=== ASM NOP Kernel (mem_access_kernel_nop) ===\n");

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        mem_access_kernel_nop(Q, K_int, V_t, O);
    }

    // Benchmark
    uint64_t start = rdtimer();
    for (int i = 0; i < bench_iters; i++) {
        mem_access_kernel_nop(Q, K_int, V_t, O);
    }
    uint64_t end = rdtimer();

    double ticks_asm = (double)(end - start) / bench_iters;
    double cycles_asm = ticks_asm * cycle_scale;
    double time_ns_asm = cycles_asm / (CPU_FREQ_HZ / 1e9);

    printf("  Iterations:       %d\n", bench_iters);
    printf("  Timer ticks:      %.1f\n", ticks_asm);
    printf("  CPU cycles:       %.1f\n", cycles_asm);
    printf("  Time/kernel:      %.1f ns\n", time_ns_asm);
    printf("\n  Bandwidth Analysis (vs CPU cycles):\n");
    printf("    Load BW:        %.1f byte/cycle (peak: 128)\n", TOTAL_BYTES_LOADED / cycles_asm);
    printf("    Store BW:       %.1f byte/cycle (peak: 64)\n", TOTAL_BYTES_STORED / cycles_asm);
    printf("    Load efficiency:  %.1f%%\n", (TOTAL_BYTES_LOADED / cycles_asm) / 128.0 * 100);
    printf("    Store efficiency: %.1f%%\n", (TOTAL_BYTES_STORED / cycles_asm) / 64.0 * 100);
    printf("    vs theoretical min: %.2fx\n", cycles_asm / THEORETICAL_MIN_CYCLES_LOAD);

    // =========================================================================
    // Benchmark C intrinsics kernel
    // =========================================================================
    printf("\n=== C Intrinsics Kernel (mem_access_fused_d256_c) ===\n");

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        mem_access_fused_d256_c(Q, K_int, V_t, O);
    }

    // Benchmark
    start = rdtimer();
    for (int i = 0; i < bench_iters; i++) {
        mem_access_fused_d256_c(Q, K_int, V_t, O);
    }
    end = rdtimer();

    double ticks_c = (double)(end - start) / bench_iters;
    double cycles_c = ticks_c * cycle_scale;
    double time_ns_c = cycles_c / (CPU_FREQ_HZ / 1e9);

    printf("  Iterations:       %d\n", bench_iters);
    printf("  Timer ticks:      %.1f\n", ticks_c);
    printf("  CPU cycles:       %.1f\n", cycles_c);
    printf("  Time/kernel:      %.1f ns\n", time_ns_c);
    printf("\n  Bandwidth Analysis (vs CPU cycles):\n");
    printf("    Load BW:        %.1f byte/cycle (peak: 128)\n", TOTAL_BYTES_LOADED / cycles_c);
    printf("    Store BW:       %.1f byte/cycle (peak: 64)\n", TOTAL_BYTES_STORED / cycles_c);
    printf("    Load efficiency:  %.1f%%\n", (TOTAL_BYTES_LOADED / cycles_c) / 128.0 * 100);
    printf("    Store efficiency: %.1f%%\n", (TOTAL_BYTES_STORED / cycles_c) / 64.0 * 100);
    printf("    vs theoretical min: %.2fx\n", cycles_c / THEORETICAL_MIN_CYCLES_LOAD);

    // =========================================================================
    // Summary
    // =========================================================================
    printf("\n=== Summary ===\n");
    printf("  ASM NOP kernel:   %.1f cycles (%.1f ns)\n", cycles_asm, time_ns_asm);
    printf("  C NOP kernel:     %.1f cycles (%.1f ns)\n", cycles_c, time_ns_c);
    printf("  Theoretical min:  %.1f cycles (L1 resident, load-bound)\n", THEORETICAL_MIN_CYCLES_LOAD);
    printf("\n  Analysis:\n");

    double ratio = cycles_asm / THEORETICAL_MIN_CYCLES_LOAD;
    if (ratio < 1.5) {
        printf("    Memory subsystem: Excellent (%.1fx theoretical)\n", ratio);
        printf("    -> L1 resident with near-peak bandwidth\n");
    } else if (ratio < 3.0) {
        printf("    Memory subsystem: Good (%.1fx theoretical)\n", ratio);
        printf("    -> Mostly L1 hits, some pipeline stalls or conflicts\n");
    } else if (ratio < 6.0) {
        printf("    Memory subsystem: Moderate (%.1fx theoretical)\n", ratio);
        printf("    -> Significant L2 traffic or cache conflicts\n");
    } else {
        printf("    Memory subsystem: Poor (%.1fx theoretical)\n", ratio);
        printf("    -> Heavy L2 spills or memory bottleneck\n");
    }

    double overhead_cycles = cycles_asm - THEORETICAL_MIN_CYCLES_LOAD;
    printf("\n  Memory overhead:  %.1f cycles (%.1f%% of total)\n",
           overhead_cycles, overhead_cycles / cycles_asm * 100);
    printf("  Effective load throughput: %.1f bytes/cycle\n",
           TOTAL_BYTES_LOADED / cycles_asm);

    // Cleanup
    free(Q);
    free(K_int);
    free(V_t);
    free(O);

    return 0;
}
