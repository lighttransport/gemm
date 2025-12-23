#
# Makefile for AMD Zen2 GEMM Benchmark
#
# Usage:
#   make              - Build with GCC (default)
#   make COMPILER=gcc - Build with GCC
#   make COMPILER=clang - Build with Clang
#   make run          - Build and run benchmark
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

# Source files
GEMM_SRCS = gemm_avx2.c
BENCH_SRCS = bench_gemm.c
HEADERS = gemm_avx2.h

# Object files
GEMM_OBJS = $(GEMM_SRCS:.c=.o)
BENCH_OBJS = $(BENCH_SRCS:.c=.o)

# Targets
BENCH_TARGET = bench_gemm

.PHONY: all clean run disasm help

all: $(BENCH_TARGET)

$(BENCH_TARGET): $(BENCH_OBJS) $(GEMM_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(BENCH_TARGET)
	./$(BENCH_TARGET)

# Run with specific matrix size
run-%: $(BENCH_TARGET)
	./$(BENCH_TARGET) $*

# Generate disassembly for analysis
disasm: $(BENCH_TARGET)
	objdump -d -M intel $(BENCH_TARGET) > $(BENCH_TARGET).asm
	@echo "Disassembly saved to $(BENCH_TARGET).asm"
	@echo "Look for sgemm_kernel_6x16 function"

# Show just the kernel disassembly
kernel-asm: $(BENCH_TARGET)
	objdump -d -M intel $(BENCH_TARGET) | \
		sed -n '/^[0-9a-f]* <sgemm_kernel_6x16>:/,/^[0-9a-f]* </p' | head -300

clean:
	rm -f $(GEMM_OBJS) $(BENCH_OBJS) $(BENCH_TARGET) $(BENCH_TARGET).asm
	rm -f *.o *.asm

help:
	@echo "AMD Zen2 GEMM Benchmark Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)  - Build benchmark executable"
	@echo "  run            - Build and run with default sizes"
	@echo "  run-<N>        - Run with NxNxN matrix (e.g., make run-1024)"
	@echo "  disasm         - Generate full disassembly"
	@echo "  kernel-asm     - Show micro-kernel disassembly"
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
	@echo "  make                      # Build with GCC"
	@echo "  make COMPILER=clang       # Build with Clang"
	@echo "  make run                  # Run benchmark"
	@echo "  make run-2048             # Benchmark 2048x2048x2048"
	@echo "  make DEBUG=1              # Debug build"
