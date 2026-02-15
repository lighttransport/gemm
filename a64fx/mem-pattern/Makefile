# Makefile for memory access pattern analysis
# Isolates load/store pipeline behavior by replacing SDOT with NOPs
#
# Usage:
#   make              - Cross-compile with fccpx (default, for login node)
#   make native       - Native compile with fcc (for compute node job)
#   make debug        - Build with debug symbols for profiling
#   make COMPILER=gcc - Use gcc (if available)
#
# Profiling (run on compute node):
#   make run-perf     - Profile with Linux perf (cache, cycles, instructions)
#   make run-fapp     - Profile with Fujitsu fapp (detailed HW counters)

# Default to cross-compiler for login node builds
COMPILER ?= fccpx
CC = $(COMPILER)

# FCC/FCCPX flags (Fujitsu compilers)
ifeq ($(COMPILER),fcc)
    CFLAGS = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
    CFLAGS_DEBUG = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -g
    ASFLAGS = -Nclang -c -march=armv8.2-a+sve
    ASFLAGS_DEBUG = -Nclang -c -march=armv8.2-a+sve -g
    LDFLAGS = -Nclang
    LDFLAGS_DEBUG = -Nclang -g
else ifeq ($(COMPILER),fccpx)
    CFLAGS = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
    CFLAGS_DEBUG = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -g
    ASFLAGS = -Nclang -c -march=armv8.2-a+sve
    ASFLAGS_DEBUG = -Nclang -c -march=armv8.2-a+sve -g
    LDFLAGS = -Nclang
    LDFLAGS_DEBUG = -Nclang -g
else
    CFLAGS = -O3 -march=armv8.2-a+sve -ffp-contract=fast
    CFLAGS_DEBUG = -O3 -march=armv8.2-a+sve -ffp-contract=fast -g
    ASFLAGS = -c -march=armv8.2-a+sve
    ASFLAGS_DEBUG = -c -march=armv8.2-a+sve -g
    LDFLAGS =
    LDFLAGS_DEBUG = -g
endif

# Source files
SRCS_C = bench_mem_pattern.c mem_access_fused_d256.c
SRCS_S = mem_access_kernel_nop.S
OBJS = bench_mem_pattern.o mem_access_fused_d256.o mem_access_kernel_nop.o
OBJS_DEBUG = bench_mem_pattern.debug.o mem_access_fused_d256.debug.o mem_access_kernel_nop.debug.o

# Profiling directories
PERF_DIR = perf_output
FAPP_DIR = fapp_output

# Targets
all: bench_mem_pattern bench_l2_pattern

# Convenience target for native compilation (compute node)
native:
	$(MAKE) COMPILER=fcc

# Convenience target for cross compilation (login node)
cross:
	$(MAKE) COMPILER=fccpx

# Debug build with symbols for profiling
debug: bench_mem_pattern_debug

# Main benchmark executable
bench_mem_pattern: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Debug executable with symbols
bench_mem_pattern_debug: $(OBJS_DEBUG)
	$(CC) $(LDFLAGS_DEBUG) -o $@ $^

# L2 cache pattern benchmark
bench_l2_pattern: bench_l2_pattern.o
	$(CC) $(LDFLAGS) -o $@ $^

bench_l2_pattern.o: bench_l2_pattern.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Object files (release)
bench_mem_pattern.o: bench_mem_pattern.c
	$(CC) $(CFLAGS) -c -o $@ $<

mem_access_fused_d256.o: mem_access_fused_d256.c
	$(CC) $(CFLAGS) -c -o $@ $<

mem_access_kernel_nop.o: mem_access_kernel_nop.S
	$(CC) $(ASFLAGS) -o $@ $<

# Object files (debug)
bench_mem_pattern.debug.o: bench_mem_pattern.c
	$(CC) $(CFLAGS_DEBUG) -c -o $@ $<

mem_access_fused_d256.debug.o: mem_access_fused_d256.c
	$(CC) $(CFLAGS_DEBUG) -c -o $@ $<

mem_access_kernel_nop.debug.o: mem_access_kernel_nop.S
	$(CC) $(ASFLAGS_DEBUG) -o $@ $<

# ============================================================================
# Profiling targets (run these on compute node via job submission)
# ============================================================================

# Linux perf profiling - basic stats
run-perf-stat:
	@mkdir -p $(PERF_DIR)
	perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores ./bench_mem_pattern -i 10000 2>&1 | tee $(PERF_DIR)/perf_stat.txt

# Linux perf profiling - detailed with record
run-perf-record:
	@mkdir -p $(PERF_DIR)
	perf record -e cycles,cache-misses,L1-dcache-load-misses -o $(PERF_DIR)/perf.data ./bench_mem_pattern -i 10000
	perf report -i $(PERF_DIR)/perf.data --stdio > $(PERF_DIR)/perf_report.txt
	@echo "Perf report saved to $(PERF_DIR)/perf_report.txt"

# Linux perf - annotate hotspots (requires debug build)
run-perf-annotate: bench_mem_pattern_debug
	@mkdir -p $(PERF_DIR)
	perf record -e cycles -o $(PERF_DIR)/perf_annotate.data ./bench_mem_pattern_debug -i 10000
	perf annotate -i $(PERF_DIR)/perf_annotate.data --stdio > $(PERF_DIR)/perf_annotate.txt
	@echo "Perf annotation saved to $(PERF_DIR)/perf_annotate.txt"

# Fujitsu fapp profiling - CPU performance
run-fapp-cpu:
	@mkdir -p $(FAPP_DIR)
	fapp -C -d $(FAPP_DIR)/cpu -Icpupa ./bench_mem_pattern -i 10000
	fapppx -A -d $(FAPP_DIR)/cpu -Icpupa -tcsv -o $(FAPP_DIR)/fapp_cpu.csv
	@echo "FAPP CPU report saved to $(FAPP_DIR)/fapp_cpu.csv"

# Fujitsu fapp profiling - Cache analysis
run-fapp-cache:
	@mkdir -p $(FAPP_DIR)
	fapp -C -d $(FAPP_DIR)/cache -Icache ./bench_mem_pattern -i 10000
	fapppx -A -d $(FAPP_DIR)/cache -Icache -tcsv -o $(FAPP_DIR)/fapp_cache.csv
	@echo "FAPP cache report saved to $(FAPP_DIR)/fapp_cache.csv"

# Fujitsu fapp profiling - Memory bandwidth
run-fapp-mem:
	@mkdir -p $(FAPP_DIR)
	fapp -C -d $(FAPP_DIR)/mem -Imem ./bench_mem_pattern -i 10000
	fapppx -A -d $(FAPP_DIR)/mem -Imem -tcsv -o $(FAPP_DIR)/fapp_mem.csv
	@echo "FAPP memory report saved to $(FAPP_DIR)/fapp_mem.csv"

# Fujitsu fapp - All profiling combined
run-fapp-all: run-fapp-cpu run-fapp-cache run-fapp-mem
	@echo "All FAPP reports generated in $(FAPP_DIR)/"

# Combined profiling
run-profile: run-perf-stat run-fapp-cpu run-fapp-cache
	@echo "Profiling complete. Check $(PERF_DIR)/ and $(FAPP_DIR)/"

# L2 profiling targets
run-l2-bench:
	./bench_l2_pattern 1000 100

run-l2-perf:
	@mkdir -p $(PERF_DIR)
	perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./bench_l2_pattern 1000 100 2>&1 | tee $(PERF_DIR)/perf_l2.txt

run-l2-fapp:
	@mkdir -p $(FAPP_DIR)
	fapp -C -d $(FAPP_DIR)/l2 -Icpupa ./bench_l2_pattern 1000 100
	fapppx -A -d $(FAPP_DIR)/l2 -Icpupa -ttext -o $(FAPP_DIR)/fapp_l2.txt
	@cat $(FAPP_DIR)/fapp_l2.txt

clean:
	rm -f *.o bench_mem_pattern bench_mem_pattern_debug bench_l2_pattern
	rm -rf $(PERF_DIR) $(FAPP_DIR)

clean-profile:
	rm -rf $(PERF_DIR) $(FAPP_DIR)

.PHONY: all native cross debug run-perf-stat run-perf-record run-perf-annotate \
        run-fapp-cpu run-fapp-cache run-fapp-mem run-fapp-all run-profile \
        run-l2-bench run-l2-perf run-l2-fapp clean clean-profile
