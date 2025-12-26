# Makefile for A64FX SVE Embedding Kernels
#
# Usage:
#   make                    - Build with fcc (native on compute node)
#   make COMPILER=fcc       - Build with fcc (native)
#   make COMPILER=fccpx     - Build with fccpx (cross-compile from login node)
#   make COMPILER=clang     - Build with clang (cross-compile)

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
else ifeq ($(COMPILER),fccpx)
    CC = fccpx
    AS = fccpx
    CFLAGS = -O3 -Nclang -march=armv8.2-a+sve -mcpu=a64fx -ffast-math -Wall
    ASFLAGS = -Nclang -march=armv8.2-a+sve -mcpu=a64fx
    LDFLAGS = -lm
else ifeq ($(COMPILER),clang)
    CC = clang
    AS = clang
    CFLAGS = -O3 --target=aarch64-linux-gnu -march=armv8.2-a+sve -mcpu=a64fx \
             -ffast-math -Wall
    ASFLAGS = --target=aarch64-linux-gnu -march=armv8.2-a+sve -mcpu=a64fx
    LDFLAGS = -lm --target=aarch64-linux-gnu
else
    $(error Unknown COMPILER=$(COMPILER). Use fcc, fccpx, or clang)
endif

#=============================================================================
# Targets
#=============================================================================
BENCH_TARGET = bench_embedding
BENCH_OBJS = bench_embedding.o embedding_asm.o

#=============================================================================
# Phony targets
#=============================================================================
.PHONY: all clean run bench help test

#=============================================================================
# Default target
#=============================================================================
all: $(BENCH_TARGET)

#=============================================================================
# Benchmark executable
#=============================================================================
$(BENCH_TARGET): $(BENCH_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

#=============================================================================
# Run benchmark
#=============================================================================
run: $(BENCH_TARGET)
	./$(BENCH_TARGET) 1000 512 4096 32000

bench: $(BENCH_TARGET)
	./$(BENCH_TARGET) 10000 512 4096 32000

# Quick test with smaller parameters
test: $(BENCH_TARGET)
	./$(BENCH_TARGET) 100 64 512 1000

# Large batch test
large: $(BENCH_TARGET)
	./$(BENCH_TARGET) 1000 2048 4096 32000

#=============================================================================
# Generic compilation rules
#=============================================================================
%.o: %.c embedding.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.S
	$(CC) $(ASFLAGS) -c -o $@ $<

#=============================================================================
# Disassembly
#=============================================================================
disasm: embedding_asm.o
	objdump -d embedding_asm.o > embedding_asm.dis
	@echo "Disassembly: embedding_asm.dis"

#=============================================================================
# Clean
#=============================================================================
clean:
	rm -f *.o *.dis $(BENCH_TARGET)

#=============================================================================
# Help
#=============================================================================
help:
	@echo "A64FX SVE Embedding Kernels"
	@echo "============================"
	@echo ""
	@echo "Compiler selection (COMPILER=fcc|fccpx|clang):"
	@echo "  make                     - Build with fcc (native on compute node)"
	@echo "  make COMPILER=fcc        - Build with fcc (native)"
	@echo "  make COMPILER=fccpx      - Build with fccpx (cross-compile from login)"
	@echo "  make COMPILER=clang      - Build with clang (cross-compile)"
	@echo ""
	@echo "Current compiler: $(COMPILER)"
	@echo "  CC=$(CC)"
	@echo "  CFLAGS=$(CFLAGS)"
	@echo ""
	@echo "Targets:"
	@echo "  make all      - Build benchmark executable"
	@echo "  make run      - Build and run with default parameters"
	@echo "  make bench    - Build and run extended benchmark"
	@echo "  make test     - Build and run quick test"
	@echo "  make large    - Build and run large batch benchmark"
	@echo "  make disasm   - Generate disassembly of embedding_asm.o"
	@echo "  make clean    - Remove all build artifacts"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "Kernels:"
	@echo "  - embedding_fwd_f32_asm:          Forward pass FP32"
	@echo "  - embedding_fwd_f32_batched_asm:  Forward pass FP32 (4-token batched)"
	@echo "  - embedding_bwd_f32_asm:          Backward pass FP32"
	@echo "  - embedding_fwd_f64_asm:          Forward pass FP64"
	@echo "  - embedding_bwd_f64_asm:          Backward pass FP64"
	@echo "  - embedding_fwd_with_pos_f32_asm: Token + position embedding FP32"
	@echo "  - embedding_grad_zero_f32_asm:    Zero gradient table"
	@echo "  - embedding_bwd_sorted_f32_asm:   Backward with sorted indices"
	@echo ""
	@echo "Architecture: A64FX (Fugaku)"
	@echo "  SVE vector length: 512 bits"
	@echo "  FP64: 8 elements/vector"
	@echo "  FP32: 16 elements/vector"
