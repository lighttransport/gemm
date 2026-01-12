# Makefile for A64FX SVE Philox RNG Kernels
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
BENCH_TARGET = bench_philox
BENCH_OBJS = bench_philox.o philox_asm.o

#=============================================================================
# Phony targets
#=============================================================================
.PHONY: all clean run bench help test small large

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
	./$(BENCH_TARGET) 1000 1048576

bench: $(BENCH_TARGET)
	./$(BENCH_TARGET) 10000 1048576

# Quick test with smaller parameters
test: $(BENCH_TARGET)
	./$(BENCH_TARGET) 100 10000

# Small count test
small: $(BENCH_TARGET)
	./$(BENCH_TARGET) 1000 1024

# Large count test
large: $(BENCH_TARGET)
	./$(BENCH_TARGET) 1000 16777216

#=============================================================================
# Generic compilation rules
#=============================================================================
%.o: %.c philox.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.S
	$(CC) $(ASFLAGS) -c -o $@ $<

#=============================================================================
# Disassembly
#=============================================================================
disasm: philox_asm.o
	objdump -d philox_asm.o > philox_asm.dis
	@echo "Disassembly: philox_asm.dis"

#=============================================================================
# Clean
#=============================================================================
clean:
	rm -f *.o *.dis $(BENCH_TARGET)

#=============================================================================
# Help
#=============================================================================
help:
	@echo "A64FX SVE Philox RNG Kernels"
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
	@echo "  make run      - Build and run with default parameters (1M numbers)"
	@echo "  make bench    - Build and run extended benchmark"
	@echo "  make test     - Build and run quick test"
	@echo "  make small    - Build and run small count test (1K numbers)"
	@echo "  make large    - Build and run large count test (16M numbers)"
	@echo "  make disasm   - Generate disassembly of philox_asm.o"
	@echo "  make clean    - Remove all build artifacts"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "Kernels:"
	@echo "  - philox4x32_scalar:          Scalar reference implementation"
	@echo "  - philox4x32_sve_u32:         Generate uint32 random numbers"
	@echo "  - philox4x32_sve_f32:         Generate float [0,1) random numbers"
	@echo "  - philox4x32_sve_normal_f32:  Generate uniforms for Box-Muller"
	@echo "  - philox4x32_sve_bytes:       Generate random bytes"
	@echo "  - philox_dropout_mask_sve:    Generate dropout mask"
	@echo ""
	@echo "Philox-4x32-10 Properties:"
	@echo "  - Counter-based (reproducible, parallelizable)"
	@echo "  - Passes BigCrush statistical tests"
	@echo "  - 4 outputs per state (128 bits)"
	@echo "  - 10 rounds for security"
	@echo ""
	@echo "Architecture: A64FX (Fugaku)"
	@echo "  SVE vector length: 512 bits"
	@echo "  uint32: 16 elements/vector"
	@echo "  Generates 64 random uint32 per iteration (4 vectors)"
