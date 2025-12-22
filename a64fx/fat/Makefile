# Makefile for A64FX SVE Kernels
# Includes: Flash Attention, GEMM, LayerNorm/RMSNorm, Activation Functions
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
    # Fujitsu compiler (native on A64FX compute node)
    CC = fcc
    AS = fcc
    CFLAGS = -O3 -Nclang -march=armv8.2-a+sve -mcpu=a64fx -ffast-math -Wall
    ASFLAGS = -Nclang -march=armv8.2-a+sve -mcpu=a64fx
    LDFLAGS = -lm
else ifeq ($(COMPILER),fccpx)
    # Fujitsu cross compiler (from login node to A64FX)
    CC = fccpx
    AS = fccpx
    CFLAGS = -O3 -Nclang -march=armv8.2-a+sve -mcpu=a64fx -ffast-math -Wall
    ASFLAGS = -Nclang -march=armv8.2-a+sve -mcpu=a64fx
    LDFLAGS = -lm
else ifeq ($(COMPILER),clang)
    # Clang cross compiler targeting A64FX (uses integrated assembler)
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
# Flash Attention (2-pass)
#=============================================================================
FLASH_SRCS = main.c flash_attn.c
FLASH_ASM = pass1_qkt_rowmax.S pass2_softmax_pv.S
FLASH_TARGET = flash_2pass

#=============================================================================
# GEMM Benchmark
#=============================================================================
GEMM_TARGET = bench_gemm_raw
GEMM_OBJS = bench_gemm_raw.o gemm_raw.o flash_fused_nofaddv.o test_flash_s.o

#=============================================================================
# Exp Benchmark
#=============================================================================
EXP_TARGET = bench_exp
EXP_OBJS = bench_exp.o exp_bench.o test_exp_simple.o test_fp16.o

#=============================================================================
# Poly Benchmark
#=============================================================================
POLY_TARGET = bench_poly
POLY_OBJS = bench_poly.o exp_poly_bench.o

#=============================================================================
# Simple Benchmark
#=============================================================================
SIMPLE_TARGET = bench_simple

#=============================================================================
# Compare Benchmark
#=============================================================================
COMPARE_TARGET = bench_compare
COMPARE_OBJS = bench_compare.o exp_intrinsics.o

COMPARE_FAST_TARGET = bench_compare_fast
COMPARE_FAST_OBJS = bench_compare.o exp_intrinsics_fast.o

#=============================================================================
# Opt Benchmark
#=============================================================================
OPT_TARGET = bench_opt
OPT_OBJS = bench_opt.o exp_intrinsics_opt.o

#=============================================================================
# LayerNorm/RMSNorm Benchmark
#=============================================================================
LAYERNORM_TARGET = bench_layernorm
LAYERNORM_OBJS = bench_layernorm.o layernorm_intrinsics.o layernorm_asm.o

#=============================================================================
# Activation Functions Benchmark (GELU, SiLU, QuickGELU, SwiGLU)
#=============================================================================
ACTIVATION_TARGET = bench_activation
ACTIVATION_OBJS = bench_activation.o activation_intrinsics.o activation_asm.o

#=============================================================================
# Phony targets
#=============================================================================
.PHONY: all clean run flash gemm exp poly simple compare opt layernorm activation help test

#=============================================================================
# Default target
#=============================================================================
all: $(FLASH_TARGET) $(GEMM_TARGET) $(LAYERNORM_TARGET) $(ACTIVATION_TARGET)

#=============================================================================
# Flash Attention
#=============================================================================
$(FLASH_TARGET): $(FLASH_SRCS:.c=.o) $(FLASH_ASM:.S=.o)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

flash: $(FLASH_TARGET)
	./$(FLASH_TARGET) 10000

#=============================================================================
# GEMM Benchmark
#=============================================================================
$(GEMM_TARGET): $(GEMM_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gemm: $(GEMM_TARGET)
	./$(GEMM_TARGET) 10000

#=============================================================================
# Exp Benchmark
#=============================================================================
$(EXP_TARGET): $(EXP_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

exp: $(EXP_TARGET)
	./$(EXP_TARGET) 10000

#=============================================================================
# Poly Benchmark
#=============================================================================
$(POLY_TARGET): $(POLY_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

poly: $(POLY_TARGET)
	./$(POLY_TARGET) 10000

#=============================================================================
# Simple Benchmark
#=============================================================================
$(SIMPLE_TARGET): bench_simple.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

simple: $(SIMPLE_TARGET)
	./$(SIMPLE_TARGET) 10000

#=============================================================================
# Compare Benchmark
#=============================================================================
$(COMPARE_TARGET): $(COMPARE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(COMPARE_FAST_TARGET): $(COMPARE_FAST_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

compare: $(COMPARE_TARGET)
	./$(COMPARE_TARGET) 10000

#=============================================================================
# Opt Benchmark
#=============================================================================
$(OPT_TARGET): $(OPT_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

opt: $(OPT_TARGET)
	./$(OPT_TARGET) 10000

#=============================================================================
# LayerNorm/RMSNorm Benchmark
#=============================================================================
$(LAYERNORM_TARGET): $(LAYERNORM_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

layernorm: $(LAYERNORM_TARGET)
	./$(LAYERNORM_TARGET) 1000 4096

#=============================================================================
# Activation Functions Benchmark
#=============================================================================
$(ACTIVATION_TARGET): $(ACTIVATION_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

activation: $(ACTIVATION_TARGET)
	./$(ACTIVATION_TARGET) 1000 4096

#=============================================================================
# Generic compilation rules
#=============================================================================
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.S
	$(CC) $(ASFLAGS) -c -o $@ $<

#=============================================================================
# Header dependencies
#=============================================================================
main.o flash_attn.o: flash_attn_2pass.h

#=============================================================================
# Disassembly targets
#=============================================================================
disasm: $(FLASH_TARGET)
	objdump -d $(FLASH_TARGET) > $(FLASH_TARGET).dis
	@echo "Disassembly: $(FLASH_TARGET).dis"

disasm-kernels: pass1_qkt_rowmax.o pass2_softmax_pv.o
	objdump -d pass1_qkt_rowmax.o > pass1.dis
	objdump -d pass2_softmax_pv.o > pass2.dis
	@echo "Pass1: pass1.dis"
	@echo "Pass2: pass2.dis"

#=============================================================================
# Test targets (check_*, test_*, find_*)
#=============================================================================
test:
	@for t in check_* test_* find_*; do \
		if [ -x "$$t" ] && [ -f "$$t" ]; then \
			echo "Running $$t..."; \
			./$$t || true; \
		fi \
	done

#=============================================================================
# Clean
#=============================================================================
clean:
	rm -f *.o *.dis
	rm -f $(FLASH_TARGET) $(GEMM_TARGET) $(EXP_TARGET) $(POLY_TARGET)
	rm -f $(SIMPLE_TARGET) $(COMPARE_TARGET) $(COMPARE_FAST_TARGET)
	rm -f $(OPT_TARGET) $(LAYERNORM_TARGET) $(ACTIVATION_TARGET)
	rm -f careful_test minimal_test debug_bench debug_ref trace_bench
	@for f in check_* test_* find_*; do \
		case "$$f" in *.c|*.h|*.S|*.s) ;; *) rm -f "$$f" ;; esac \
	done

#=============================================================================
# Help
#=============================================================================
help:
	@echo "A64FX SVE Kernel Benchmarks"
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
	@echo "Main targets:"
	@echo "  make all          - Build flash, gemm, layernorm, activation"
	@echo "  make flash        - Build and run Flash Attention benchmark"
	@echo "  make gemm         - Build and run GEMM benchmark"
	@echo "  make layernorm    - Build and run LayerNorm/RMSNorm benchmark"
	@echo "  make activation   - Build and run activation functions benchmark"
	@echo ""
	@echo "Additional targets:"
	@echo "  make exp          - Build and run exp() benchmark"
	@echo "  make poly         - Build and run polynomial exp benchmark"
	@echo "  make simple       - Build and run simple benchmark"
	@echo "  make compare      - Build and run compare benchmark"
	@echo "  make opt          - Build and run optimized exp benchmark"
	@echo ""
	@echo "Utility targets:"
	@echo "  make clean        - Remove all build artifacts"
	@echo "  make disasm       - Generate disassembly"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Architecture: A64FX (Fugaku)"
	@echo "  SVE vector length: 512 bits"
	@echo "  FP64: 8 elements/vector"
	@echo "  FP32: 16 elements/vector"
