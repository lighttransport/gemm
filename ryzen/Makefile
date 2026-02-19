#
# Makefile for AMD Zen2 GEMM, Flash Attention & NN Ops Benchmark
#
# Usage:
#   make              - Build all benchmarks
#   make gemm         - Build and run GEMM benchmark
#   make flash        - Build and run Flash Attention benchmark
#   make nn           - Build and run NN Ops benchmark (RMSNorm, LayerNorm, FFN)
#   make clean        - Clean build artifacts
#

# Compiler selection
COMPILER ?= gcc

ifeq ($(COMPILER),gcc)
    CC = gcc
    CFLAGS = -O3 -march=znver2 -mtune=znver2 -mavx2 -mfma -ffast-math
    CFLAGS += -funroll-loops -fomit-frame-pointer
    CFLAGS += -Wall -Wextra -Wpedantic
    LDFLAGS = -lm
else ifeq ($(COMPILER),clang)
    CC = clang
    CFLAGS = -O3 -march=znver2 -mtune=znver2 -mavx2 -mfma -ffast-math
    CFLAGS += -funroll-loops -fomit-frame-pointer
    CFLAGS += -Wall -Wextra -Wpedantic
    LDFLAGS = -lm
else
    $(error Unknown COMPILER=$(COMPILER). Use gcc or clang)
endif

# Debug build
ifdef DEBUG
    CFLAGS = -O0 -g -march=znver2 -mavx2 -mfma
    CFLAGS += -Wall -Wextra -Wpedantic
    CFLAGS += -fsanitize=address -fsanitize=undefined
    LDFLAGS += -fsanitize=address -fsanitize=undefined
endif

# Profile build
ifdef PROFILE
    CFLAGS += -pg -g
    LDFLAGS += -pg
endif

# Headers
GEMM_HEADERS = gemm_avx2.h
FLASH_HEADERS = flash_attn_avx2.h
NN_HEADERS = nn_ops_avx2.h

# Targets
BENCH_GEMM = bench_gemm
BENCH_FLASH = bench_flash_attn
BENCH_NN = bench_nn_ops

.PHONY: all clean gemm flash nn run help disasm

all: $(BENCH_GEMM) $(BENCH_FLASH) $(BENCH_NN)

# GEMM benchmark
$(BENCH_GEMM): bench_gemm.o gemm_avx2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_gemm.o: bench_gemm.c $(GEMM_HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

gemm_avx2.o: gemm_avx2.c $(GEMM_HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Flash Attention benchmark
$(BENCH_FLASH): bench_flash_attn.o flash_attn_avx2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_flash_attn.o: bench_flash_attn.c $(FLASH_HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

flash_attn_avx2.o: flash_attn_avx2.c $(FLASH_HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# NN Ops benchmark (RMSNorm, LayerNorm, FFN, Activations)
$(BENCH_NN): bench_nn_ops.o nn_ops_avx2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_nn_ops.o: bench_nn_ops.c $(NN_HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

nn_ops_avx2.o: nn_ops_avx2.c $(NN_HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Run targets
gemm: $(BENCH_GEMM)
	./$(BENCH_GEMM)

flash: $(BENCH_FLASH)
	./$(BENCH_FLASH)

nn: $(BENCH_NN)
	./$(BENCH_NN)

run: gemm

# Run GEMM with specific matrix size
run-%: $(BENCH_GEMM)
	./$(BENCH_GEMM) $*

# Generate disassembly for analysis
disasm: $(BENCH_GEMM) $(BENCH_FLASH) $(BENCH_NN)
	objdump -d -M intel $(BENCH_GEMM) > $(BENCH_GEMM).asm
	objdump -d -M intel $(BENCH_FLASH) > $(BENCH_FLASH).asm
	objdump -d -M intel $(BENCH_NN) > $(BENCH_NN).asm
	@echo "Disassembly saved to *.asm files"

# Show GEMM kernel disassembly
gemm-asm: $(BENCH_GEMM)
	objdump -d -M intel $(BENCH_GEMM) | \
		sed -n '/^[0-9a-f]* <sgemm_kernel_6x16>:/,/^[0-9a-f]* </p' | head -300

# Show Flash Attention kernel disassembly
flash-asm: $(BENCH_FLASH)
	objdump -d -M intel $(BENCH_FLASH) | \
		sed -n '/^[0-9a-f]* <exp256_ps>:/,/^[0-9a-f]* </p' | head -100

# Show RMSNorm disassembly
nn-asm: $(BENCH_NN)
	objdump -d -M intel $(BENCH_NN) | \
		sed -n '/^[0-9a-f]* <rmsnorm_f32_avx2>:/,/^[0-9a-f]* </p' | head -200

clean:
	rm -f *.o $(BENCH_GEMM) $(BENCH_FLASH) $(BENCH_NN) *.asm

help:
	@echo "AMD Zen2 Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)  - Build all benchmarks"
	@echo "  gemm           - Build and run GEMM benchmark"
	@echo "  flash          - Build and run Flash Attention benchmark"
	@echo "  nn             - Build and run NN Ops (RMSNorm, LayerNorm, FFN)"
	@echo "  run            - Same as 'gemm'"
	@echo "  run-<N>        - GEMM with NxNxN matrix (e.g., make run-1024)"
	@echo "  disasm         - Generate disassembly for all"
	@echo "  gemm-asm       - Show GEMM micro-kernel assembly"
	@echo "  flash-asm      - Show exp256 kernel assembly"
	@echo "  nn-asm         - Show RMSNorm assembly"
	@echo "  clean          - Remove build artifacts"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  COMPILER=gcc   - Use GCC (default)"
	@echo "  COMPILER=clang - Use Clang"
	@echo "  DEBUG=1        - Debug build with sanitizers"
	@echo "  PROFILE=1      - Build with profiling support"
	@echo ""
	@echo "Examples:"
	@echo "  make                      # Build all"
	@echo "  make gemm                 # Run GEMM benchmark"
	@echo "  make flash                # Run Flash Attention benchmark"
	@echo "  make nn                   # Run NN Ops benchmark"
	@echo "  make COMPILER=clang nn    # NN Ops with Clang"
