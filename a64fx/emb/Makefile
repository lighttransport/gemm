# Makefile for A64FX SVE Embedding Kernels
#
# Usage:
#   make                    - Build single-core benchmark
#   make omp                - Build multi-core (OpenMP) benchmark
#   make COMPILER=fcc       - Build with fcc (native)
#   make COMPILER=fccpx     - Build with fccpx (cross-compile from login node)

#=============================================================================
# Compiler Selection (default: fcc)
#=============================================================================
COMPILER ?= fcc

ifeq ($(COMPILER),fcc)
    CC = fcc
    AS = fcc
    CFLAGS = -O3 -Nclang -march=armv8.2-a+sve -mcpu=a64fx -ffast-math -Wall
    ASFLAGS = -Nclang -march=armv8.2-a+sve -mcpu=a64fx
    LDFLAGS = -lm
    OMP_FLAG = -fopenmp
else ifeq ($(COMPILER),fccpx)
    CC = fccpx
    AS = fccpx
    CFLAGS = -O3 -Nclang -march=armv8.2-a+sve -mcpu=a64fx -ffast-math -Wall
    ASFLAGS = -Nclang -march=armv8.2-a+sve -mcpu=a64fx
    LDFLAGS = -lm
    OMP_FLAG = -fopenmp
else ifeq ($(COMPILER),clang)
    CC = clang
    AS = clang
    CFLAGS = -O3 --target=aarch64-linux-gnu -march=armv8.2-a+sve -mcpu=a64fx \
             -ffast-math -Wall
    ASFLAGS = --target=aarch64-linux-gnu -march=armv8.2-a+sve -mcpu=a64fx
    LDFLAGS = -lm --target=aarch64-linux-gnu
    OMP_FLAG = -fopenmp
else
    $(error Unknown COMPILER=$(COMPILER). Use fcc, fccpx, or clang)
endif

#=============================================================================
# Targets
#=============================================================================
BENCH_TARGET     = bench_embedding
BENCH_OMP_TARGET = bench_embedding_omp
ASM_OBJ          = embedding_asm.o

#=============================================================================
# Phony targets
#=============================================================================
.PHONY: all omp clean run run-omp test help

#=============================================================================
# Default: single-core
#=============================================================================
all: $(BENCH_TARGET)

#=============================================================================
# OpenMP: multi-core
#=============================================================================
omp: $(BENCH_OMP_TARGET)

#=============================================================================
# Single-core benchmark
#=============================================================================
$(BENCH_TARGET): bench_embedding.c embedding.h $(ASM_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_embedding.c $(ASM_OBJ) $(LDFLAGS)

#=============================================================================
# OpenMP benchmark
#=============================================================================
bench_embedding_omp.o: bench_embedding.c embedding.h
	$(CC) $(CFLAGS) $(OMP_FLAG) -c -o $@ bench_embedding.c

$(BENCH_OMP_TARGET): bench_embedding_omp.o $(ASM_OBJ)
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $@ $^ $(LDFLAGS)

#=============================================================================
# Assembly object
#=============================================================================
$(ASM_OBJ): embedding_asm.S
	$(CC) $(ASFLAGS) -c -o $@ $<

#=============================================================================
# Run targets
#=============================================================================
run: $(BENCH_TARGET)
	./$(BENCH_TARGET)

run-omp: $(BENCH_OMP_TARGET)
	OMP_NUM_THREADS=48 ./$(BENCH_OMP_TARGET)

# Quick correctness test (single-core)
test: $(BENCH_TARGET)
	./$(BENCH_TARGET)

# Bench-only (skip correctness)
bench: $(BENCH_TARGET)
	./$(BENCH_TARGET) --bench-only

bench-omp: $(BENCH_OMP_TARGET)
	OMP_NUM_THREADS=48 ./$(BENCH_OMP_TARGET) --bench-only

#=============================================================================
# Disassembly
#=============================================================================
disasm: $(ASM_OBJ)
	objdump -d $(ASM_OBJ) > embedding_asm.dis
	@echo "Disassembly: embedding_asm.dis"

#=============================================================================
# Clean
#=============================================================================
clean:
	rm -f *.o *.dis $(BENCH_TARGET) $(BENCH_OMP_TARGET)

#=============================================================================
# Help
#=============================================================================
help:
	@echo "A64FX SVE Embedding Kernels"
	@echo "============================"
	@echo ""
	@echo "Build targets:"
	@echo "  make           - Build single-core benchmark"
	@echo "  make omp       - Build OpenMP multi-core benchmark"
	@echo ""
	@echo "Run targets:"
	@echo "  make run       - Run single-core benchmark"
	@echo "  make run-omp   - Run 48-thread OpenMP benchmark"
	@echo "  make test      - Quick correctness test"
	@echo "  make bench     - Performance only (skip correctness)"
	@echo ""
	@echo "Kernels:"
	@echo "  baseline  - Sequential row copy with next-row prefetch"
	@echo "  batched   - 4-token batched row copy"
	@echo "  stream    - 8x unroll + deep prefetch (8 rows ahead)"
	@echo "  gather    - SVE gather/scatter (16 tokens column-wise)"
	@echo "  sorted    - Counting sort + contiguous read + indirect write"
	@echo ""
	@echo "Compiler: $(COMPILER) (override with COMPILER=fcc|fccpx|clang)"
