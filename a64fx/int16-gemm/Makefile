# Makefile for INT16 SDOT GEMM benchmark
# A64FX SVE implementation

# Compilers
CC_FCC = fccpx
CC_GCC = gcc

# Default compiler
COMPILER ?= fccpx

ifeq ($(COMPILER),fccpx)
    CC = $(CC_FCC)
    CFLAGS = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
    ASFLAGS = -Nclang -c
    LDFLAGS = -Nclang
else ifeq ($(COMPILER),gcc)
    CC = $(CC_GCC)
    CFLAGS = -O3 -march=armv8.2-a+sve -ffp-contract=fast
    ASFLAGS = -c
    LDFLAGS =
endif

# Targets
TARGETS = bench_int16_sdot bench_int16_sdot_asm bench_int16_l2stream

# Source files
BENCH_SRC = bench_int16_sdot.c
KERNEL_ASM = int16_sdot_kernel.S
L2STREAM_SRC = bench_int16_l2stream.c
L2STREAM_ASM = int16_sdot_l2stream.S

# Object files
KERNEL_OBJ = int16_sdot_kernel.o
L2STREAM_OBJ = int16_sdot_l2stream.o

.PHONY: all clean run test help

all: $(TARGETS)

# C intrinsic only version
bench_int16_sdot: $(BENCH_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# C + ASM kernel version
bench_int16_sdot_asm: $(BENCH_SRC) $(KERNEL_OBJ)
	$(CC) $(CFLAGS) -DHAS_ASM_KERNEL -o $@ $^ $(LDFLAGS)

# Assembly kernel
$(KERNEL_OBJ): $(KERNEL_ASM)
	$(CC) $(ASFLAGS) -o $@ $<

# L2 streaming benchmark
bench_int16_l2stream: $(L2STREAM_SRC) $(L2STREAM_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(L2STREAM_OBJ): $(L2STREAM_ASM)
	$(CC) $(ASFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) $(KERNEL_OBJ) $(L2STREAM_OBJ)

# Run benchmarks
run: all
	@echo "=== Running C Intrinsic Version ==="
	./bench_int16_sdot 1000000
	@echo ""
	@echo "=== Running ASM Kernel Version ==="
	./bench_int16_sdot_asm 1000000

# Quick test
test: all
	./bench_int16_sdot 10000
	./bench_int16_sdot_asm 10000

# Help
help:
	@echo "INT16 SDOT GEMM Benchmark"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build all targets (default compiler: fccpx)"
	@echo "  make COMPILER=gcc - Build with GCC"
	@echo "  make run          - Run benchmarks"
	@echo "  make test         - Quick test"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Targets:"
	@echo "  bench_int16_sdot      - C intrinsic version"
	@echo "  bench_int16_sdot_asm  - C + ASM kernel version"
