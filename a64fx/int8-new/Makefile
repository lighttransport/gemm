# Makefile for INT8 GEMM with SVE SDOT on A64FX
# Supports both 5x4 and 6x4 microkernel variants

COMPILER ?= fccpx
CC = $(COMPILER)

# FCC/FCCPX flags
ifeq ($(COMPILER),fccpx)
    CFLAGS = -Nclang -O2 -march=armv8.2-a+sve -fopenmp
    ASFLAGS = -Nclang -c
    LDFLAGS = -Nclang -fopenmp
else ifeq ($(COMPILER),gcc)
    CFLAGS = -O2 -march=armv8.2-a+sve -fopenmp
    ASFLAGS = -c
    LDFLAGS = -fopenmp
else
    CFLAGS = -O2 -march=armv8.2-a+sve -fopenmp
    ASFLAGS = -c
    LDFLAGS = -fopenmp
endif

# Object files
PACK_OBJ = gemm_pack.o
DRIVER_OBJ = gemm_driver.o
KERNEL_5X4_OBJ = kernel_5x4.o
KERNEL_6X4_OBJ = kernel_6x4.o
KERNEL_6X4_OPT_OBJ = kernel_6x4_opt.o
KERNEL_6X4_ULTRA_OBJ = kernel_6x4_ultra.o
KERNEL_6X4_PIPE_OBJ = kernel_6x4_pipe.o
KERNEL_6X4_UNROLL_OBJ = kernel_6x4_unroll.o
KERNEL_6X4_PIPE2_OBJ = kernel_6x4_pipe2.o
KERNEL_STAGE2_K64_OBJ = kernel_stage2_k64.o
PACK_OPT_OBJ = gemm_pack_opt.o
PREPACKED_OBJ = gemm_prepacked.o
FUSED_OBJ = fused_gemm.o
SOFTMAX_OBJ = softmax_exp2.o
ONLINE_SOFTMAX_OBJ = online_softmax.o
ATTENTION_OBJ = fused_attention.o
FFN_OBJ = ffn_int8.o
KERNEL_6X4_KLOOP_OBJ = kernel_6x4_kloop.o
KERNEL_6X4_KLOOP_OPT_OBJ = kernel_6x4_kloop_opt.o

# Benchmark executables
BENCH_5X4 = bench_5x4
BENCH_6X4 = bench_6x4
BENCH_COMPARE = bench_compare
TEST_SIMPLE = test_simple
DEBUG_PACK = debug_pack
BENCH_ULTRA = bench_ultra
BENCH_UNROLL = bench_unroll
BENCH_TIGHT = bench_tight
BENCH_PREPACKED2 = bench_prepacked2
BENCH_PIPE2 = bench_pipe2
BENCH_FULL = bench_full
BENCH_FUSED = bench_fused
BENCH_ATTENTION = bench_attention
BENCH_FFN = bench_ffn
BENCH_FFN_OPT = bench_ffn_opt
BENCH_KLOOP = bench_kloop
BENCH_FFN_V2 = bench_ffn_v2
BENCH_FFN_V3 = bench_ffn_v3
FFN_V3_OBJ = ffn_int8_v3.o

.PHONY: all clean help 5x4 6x4 compare test debug ultra unroll tight prepacked2 pipe2 full fused attention ffn ffn_opt kloop ffn_v2 ffn_v3

all: help

help:
	@echo "INT8 GEMM Benchmark for A64FX SVE SDOT"
	@echo ""
	@echo "Targets:"
	@echo "  5x4        - Build 5x4 microkernel benchmark"
	@echo "  6x4        - Build 6x4 microkernel benchmark"
	@echo "  compare    - Build comparison benchmark (both kernels)"
	@echo "  test       - Build simple correctness test"
	@echo "  debug      - Build packing debug tool"
	@echo "  ultra      - Build ultra-optimized kernel (target: 95%+ peak)"
	@echo "  all        - Show this help"
	@echo "  clean      - Remove all build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make 5x4 COMPILER=fccpx    # Default compiler"
	@echo "  make 6x4 COMPILER=gcc      # Use GCC instead"
	@echo ""

5x4: $(BENCH_5X4)

6x4: $(BENCH_6X4)

compare: $(BENCH_COMPARE)

test: $(TEST_SIMPLE)

debug: $(DEBUG_PACK)

ultra: $(BENCH_ULTRA)

unroll: $(BENCH_UNROLL)

tight: $(BENCH_TIGHT)

prepacked2: $(BENCH_PREPACKED2)

pipe2: $(BENCH_PIPE2)

full: $(BENCH_FULL)

fused: $(BENCH_FUSED)

attention: $(BENCH_ATTENTION)

ffn: $(BENCH_FFN)

ffn_opt: $(BENCH_FFN_OPT)

kloop: $(BENCH_KLOOP)

ffn_v2: $(BENCH_FFN_V2)

ffn_v3: $(BENCH_FFN_V3)

silu: bench_silu

# Build 5x4 benchmark (link both kernels to support potential comparison)
$(BENCH_5X4): bench_gemm.c $(PACK_OBJ) $(DRIVER_OBJ) $(KERNEL_5X4_OBJ) $(KERNEL_6X4_OBJ)
	$(CC) $(CFLAGS) -DUSE_5X4 -o $@ bench_gemm.c $(PACK_OBJ) $(DRIVER_OBJ) $(KERNEL_5X4_OBJ) $(KERNEL_6X4_OBJ) $(LDFLAGS) -lm

# Build 6x4 benchmark (link both kernels to support potential comparison)
$(BENCH_6X4): bench_gemm.c $(PACK_OBJ) $(DRIVER_OBJ) $(KERNEL_5X4_OBJ) $(KERNEL_6X4_OBJ)
	$(CC) $(CFLAGS) -DUSE_6X4 -o $@ bench_gemm.c $(PACK_OBJ) $(DRIVER_OBJ) $(KERNEL_5X4_OBJ) $(KERNEL_6X4_OBJ) $(LDFLAGS) -lm

# Build comparison benchmark (both kernels)
$(BENCH_COMPARE): bench_gemm.c $(PACK_OBJ) $(DRIVER_OBJ) $(KERNEL_5X4_OBJ) $(KERNEL_6X4_OBJ)
	$(CC) $(CFLAGS) -DUSE_BOTH -o $@ bench_gemm.c $(PACK_OBJ) $(DRIVER_OBJ) $(KERNEL_5X4_OBJ) $(KERNEL_6X4_OBJ) $(LDFLAGS) -lm

# Build simple test
$(TEST_SIMPLE): test_simple.c $(PACK_OBJ) $(DRIVER_OBJ) $(KERNEL_5X4_OBJ) $(KERNEL_6X4_OBJ)
	$(CC) $(CFLAGS) -o $@ test_simple.c $(PACK_OBJ) $(DRIVER_OBJ) $(KERNEL_5X4_OBJ) $(KERNEL_6X4_OBJ) $(LDFLAGS) -lm

# Build packing debug tool
$(DEBUG_PACK): debug_pack.c $(PACK_OBJ)
	$(CC) $(CFLAGS) -o $@ debug_pack.c $(PACK_OBJ) $(LDFLAGS)

# Build ultra-optimized benchmark (ld1rw + interleaved packing)
$(BENCH_ULTRA): bench_ultra.c $(PACK_OPT_OBJ) $(KERNEL_6X4_ULTRA_OBJ) $(KERNEL_6X4_PIPE_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_ultra.c $(PACK_OPT_OBJ) $(KERNEL_6X4_ULTRA_OBJ) $(KERNEL_6X4_PIPE_OBJ) $(LDFLAGS) -lm

# Build fully unrolled kernel benchmark (target: 95%+ efficiency)
$(BENCH_UNROLL): bench_unroll.c $(PREPACKED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_ULTRA_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_unroll.c $(PREPACKED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_ULTRA_OBJ) $(LDFLAGS) -lm

# Build tight loop benchmark
$(BENCH_TIGHT): bench_tight.c $(PREPACKED_OBJ) $(KERNEL_6X4_ULTRA_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_tight.c $(PREPACKED_OBJ) $(KERNEL_6X4_ULTRA_OBJ) $(LDFLAGS) -lm

# Build prepacked2 benchmark
$(BENCH_PREPACKED2): bench_prepacked2.c $(PREPACKED_OBJ) $(KERNEL_6X4_ULTRA_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_prepacked2.c $(PREPACKED_OBJ) $(KERNEL_6X4_ULTRA_OBJ) $(LDFLAGS) -lm

# Build pipe2 benchmark (pipelined unrolled kernel comparison)
$(BENCH_PIPE2): bench_pipe2.c $(PREPACKED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_PIPE2_OBJ) $(KERNEL_6X4_ULTRA_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_pipe2.c $(PREPACKED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_PIPE2_OBJ) $(KERNEL_6X4_ULTRA_OBJ) $(LDFLAGS) -lm

# Build full GEMM benchmark with unrolled kernel
$(BENCH_FULL): bench_full.c $(PREPACKED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_ULTRA_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_full.c $(PREPACKED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_ULTRA_OBJ) $(LDFLAGS) -lm

# Build fused GEMM benchmark: O = (A @ B^T) @ C
$(BENCH_FUSED): bench_fused.c $(FUSED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_STAGE2_K64_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_fused.c $(FUSED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_STAGE2_K64_OBJ) $(LDFLAGS) -lm

# Build fused attention benchmark: O = softmax(Q @ K^T / sqrt(d)) @ V
$(BENCH_ATTENTION): bench_attention.c $(ATTENTION_OBJ) $(SOFTMAX_OBJ) $(ONLINE_SOFTMAX_OBJ) $(FUSED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_STAGE2_K64_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_attention.c $(ATTENTION_OBJ) $(SOFTMAX_OBJ) $(ONLINE_SOFTMAX_OBJ) $(FUSED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_STAGE2_K64_OBJ) $(LDFLAGS) -lm

# Build debug attention benchmark (L=512 only)
bench_attention_debug: bench_attention_debug.c $(ATTENTION_OBJ) $(SOFTMAX_OBJ) $(ONLINE_SOFTMAX_OBJ) $(FUSED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_STAGE2_K64_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_attention_debug.c $(ATTENTION_OBJ) $(SOFTMAX_OBJ) $(ONLINE_SOFTMAX_OBJ) $(FUSED_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_STAGE2_K64_OBJ) $(LDFLAGS) -lm

# Build INT8 FFN benchmark (Qwen3-style SwiGLU)
$(BENCH_FFN): bench_ffn.c $(FFN_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_ULTRA_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_ffn.c $(FFN_OBJ) $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_ULTRA_OBJ) $(LDFLAGS) -lm

# Build optimized INT8 FFN benchmark (kernel_6x4_kloop - target: 90%+)
$(BENCH_FFN_OPT): bench_ffn_opt.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_KLOOP_OPT_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_ffn_opt.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_KLOOP_OPT_OBJ) $(LDFLAGS) -lm

# Build K-loop kernel benchmark for various K values
$(BENCH_KLOOP): bench_kloop.c $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_KLOOP_OPT_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_kloop.c $(KERNEL_6X4_UNROLL_OBJ) $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_KLOOP_OPT_OBJ) $(LDFLAGS) -lm

# Build FFN V2 benchmark (kernel_6x4_kloop, no K-tiling overhead)
$(BENCH_FFN_V2): bench_ffn_v2.c ffn_int8_v2.c $(KERNEL_6X4_KLOOP_OBJ) $(FFN_OBJ) $(KERNEL_6X4_ULTRA_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_ffn_v2.c $(KERNEL_6X4_KLOOP_OBJ) $(FFN_OBJ) $(KERNEL_6X4_ULTRA_OBJ) $(LDFLAGS) -lm

# Build FFN V3 benchmark (fast SiLU with SVE polynomial approximation)
$(BENCH_FFN_V3): bench_ffn_v3.c $(FFN_V3_OBJ) $(KERNEL_6X4_KLOOP_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_ffn_v3.c $(FFN_V3_OBJ) $(KERNEL_6X4_KLOOP_OBJ) $(LDFLAGS) -lm

# Build standalone SiLU benchmark
bench_silu: bench_silu.c silu_fast.h
	$(CC) $(CFLAGS) -o $@ bench_silu.c $(LDFLAGS) -lm

# Build FFN GEMM-only benchmark
bench_ffn_gemm: bench_ffn_gemm.c $(KERNEL_6X4_KLOOP_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_ffn_gemm.c $(KERNEL_6X4_KLOOP_OBJ) $(LDFLAGS)

# Build N-tile batched kernel benchmark
KERNEL_6X4_NTILE_OBJ = kernel_6x4_ntile.o
bench_ntile: bench_ntile.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_NTILE_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_ntile.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_NTILE_OBJ) $(LDFLAGS)

# Compile C source files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile assembly files
%.o: %.S
	$(CC) $(ASFLAGS) -o $@ $<

# Clean build artifacts
clean:
	rm -f $(PACK_OBJ) $(DRIVER_OBJ) $(KERNEL_5X4_OBJ) $(KERNEL_6X4_OBJ)
	rm -f $(BENCH_5X4) $(BENCH_6X4) $(BENCH_COMPARE) $(TEST_SIMPLE) $(DEBUG_PACK)
	rm -f *.o

# Dependencies
$(PACK_OBJ): gemm_pack.c gemm_pack.h
$(DRIVER_OBJ): gemm_driver.c gemm_driver.h gemm_pack.h
$(KERNEL_5X4_OBJ): kernel_5x4.S
$(KERNEL_6X4_OBJ): kernel_6x4.S
$(KERNEL_6X4_OPT_OBJ): kernel_6x4_opt.S
$(KERNEL_6X4_ULTRA_OBJ): kernel_6x4_ultra.S
$(KERNEL_6X4_PIPE_OBJ): kernel_6x4_pipe.S
$(KERNEL_6X4_UNROLL_OBJ): kernel_6x4_unroll.S
$(KERNEL_6X4_PIPE2_OBJ): kernel_6x4_pipe2.S
$(KERNEL_STAGE2_K64_OBJ): kernel_stage2_k64.S
$(PACK_OPT_OBJ): gemm_pack_opt.c gemm_pack_opt.h
$(PREPACKED_OBJ): gemm_prepacked.c gemm_prepacked.h
$(FUSED_OBJ): fused_gemm.c fused_gemm.h
$(SOFTMAX_OBJ): softmax_exp2.c softmax_exp2.h
$(ONLINE_SOFTMAX_OBJ): online_softmax.c online_softmax.h softmax_exp2.h
$(ATTENTION_OBJ): fused_attention.c fused_attention.h fused_gemm.h softmax_exp2.h online_softmax.h
$(FFN_OBJ): ffn_int8.c ffn_int8.h
$(FFN_V3_OBJ): ffn_int8_v3.c ffn_int8_v3.h silu_fast.h
$(KERNEL_6X4_KLOOP_OBJ): kernel_6x4_kloop.S
$(KERNEL_6X4_KLOOP_OPT_OBJ): kernel_6x4_kloop_opt.S

# Build 5x4 pipelined kernel benchmark
KERNEL_5X4_PIPE_OBJ = kernel_5x4_pipe.o
KERNEL_5X4_PIPE2_OBJ = kernel_5x4_pipe2.o
bench_5x4_pipe: bench_5x4_pipe.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_5X4_PIPE_OBJ) $(KERNEL_5X4_PIPE2_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_5x4_pipe.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_5X4_PIPE_OBJ) $(KERNEL_5X4_PIPE2_OBJ) $(LDFLAGS)

$(KERNEL_5X4_PIPE_OBJ): kernel_5x4_pipe.S
$(KERNEL_5X4_PIPE2_OBJ): kernel_5x4_pipe2.S

# Build all-N optimized kernel benchmark
KERNEL_6X4_ALLN_OBJ = kernel_6x4_allN.o
bench_allN: bench_allN.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_ALLN_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_allN.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_ALLN_OBJ) $(LDFLAGS)

$(KERNEL_6X4_ALLN_OBJ): kernel_6x4_allN.S

# Build pipelined kernel comparison benchmark
KERNEL_6X4_PIPEV2_OBJ = kernel_6x4_pipeV2.o
KERNEL_6X4_DEEP_OBJ = kernel_6x4_deep.o
KERNEL_6X4_PRFM_OBJ = kernel_6x4_prfm.o
bench_pipe_cmp: bench_pipe_cmp.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_PIPEV2_OBJ) $(KERNEL_6X4_DEEP_OBJ) $(KERNEL_6X4_PRFM_OBJ)
	$(CC) $(CFLAGS) -o $@ bench_pipe_cmp.c $(KERNEL_6X4_KLOOP_OBJ) $(KERNEL_6X4_PIPEV2_OBJ) $(KERNEL_6X4_DEEP_OBJ) $(KERNEL_6X4_PRFM_OBJ) $(LDFLAGS)

$(KERNEL_6X4_PIPEV2_OBJ): kernel_6x4_pipeV2.S
$(KERNEL_6X4_DEEP_OBJ): kernel_6x4_deep.S
$(KERNEL_6X4_PRFM_OBJ): kernel_6x4_prfm.S
