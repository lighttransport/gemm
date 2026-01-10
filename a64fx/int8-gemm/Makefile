# Makefile for INT8 SDOT GEMM benchmark
# A64FX SVE implementation

# Compilers
CC_FCC = fcc
CC_GCC = gcc
CC_CLANG = clang

# Default compiler
COMPILER ?= fccpx

ifeq ($(COMPILER),fccpx)
    CC = $(CC_FCC)
    # Use -O2 to avoid memory ordering issues with -O3
    CFLAGS = -Nclang -O2 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp
    ASFLAGS = -Nclang -c
    LDFLAGS = -Nclang -fopenmp
else ifeq ($(COMPILER),gcc)
    CC = $(CC_GCC)
    CFLAGS = -O2 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp
    ASFLAGS = -c
    LDFLAGS = -fopenmp
else ifeq ($(COMPILER),clang)
    CC = $(CC_CLANG)
    CFLAGS = -O2 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp
    ASFLAGS = -c
    LDFLAGS = -fopenmp
endif

# Targets
TARGETS = bench_int8_sdot bench_int8_sdot_asm bench_int8_l2stream bench_unroll gemm_tiled bench_igemm_simple bench_peak bench_llm_gemm bench_fp_gemm gemm_parallel bench_batch_gemm bench_profile bench_overlap bench_vec bench_tile bench_fused bench_pipeline gemm_sector bench_compare bench_sector_simple bench_flash_int8 bench_softmax_int8 bench_qkt_gemm bench_tiled_l2 bench_exp2 bench_fused_attn bench_exp2_sve bench_attention bench_flash_fused bench_flash_gemm bench_flash_gemm_opt bench_flash_fused_v2

# Source files
BENCH_SRC = bench_int8_sdot.c
BENCH_L2_SRC = bench_int8_l2stream.c
KERNEL_ASM = int8_sdot_kernel.S
KERNEL_L2_ASM = int8_sdot_l2stream.S

# Object files
KERNEL_OBJ = int8_sdot_kernel.o
KERNEL_L2_OBJ = int8_sdot_l2stream.o

.PHONY: all clean run

all: $(TARGETS)

# C intrinsic only version
bench_int8_sdot: $(BENCH_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# C + ASM kernel version
bench_int8_sdot_asm: $(BENCH_SRC) $(KERNEL_OBJ)
	$(CC) $(CFLAGS) -DHAS_ASM_KERNEL -o $@ $^ $(LDFLAGS)

# Assembly kernel
$(KERNEL_OBJ): $(KERNEL_ASM)
	$(CC) $(ASFLAGS) -o $@ $<

# L2 streaming benchmark
bench_int8_l2stream: $(BENCH_L2_SRC) $(KERNEL_L2_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(KERNEL_L2_OBJ): $(KERNEL_L2_ASM)
	$(CC) $(ASFLAGS) -o $@ $<

# B-unrolling comparison benchmark
bench_unroll: bench_unroll.c $(KERNEL_L2_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Tiled GEMM benchmark
gemm_tiled: gemm_tiled.c $(KERNEL_L2_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Simple iGEMM benchmark (SDOT vs MLA comparison)
bench_igemm_simple: bench_igemm_simple.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Peak throughput benchmark (5x4 x 2 unroll like dgemm.S)
bench_peak: bench_peak.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# LLM GEMM benchmark (INT8 SDOT)
bench_llm_gemm: bench_llm_gemm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# FP16/FP32 GEMM benchmark
bench_fp_gemm: bench_fp_gemm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Parallel INT8 GEMM (pthread + OpenMP for reference)
gemm_parallel: gemm_parallel.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDFLAGS)

# Batch GEMM benchmark (no sync between GEMMs)
bench_batch_gemm: bench_batch_gemm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Profiling benchmark for efficiency analysis
bench_profile: bench_profile.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Overlap test: SDOT (SIMD) vs SDOT+IntOps
bench_overlap: bench_overlap.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Vector output comparison
bench_vec: bench_vec.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Tile size comparison (6×4, 4×4)
bench_tile: bench_tile.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Fused SADDV + activation
bench_fused: bench_fused.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Pipelined load-compute (5×4 kernel)
bench_pipeline: bench_pipeline.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Sector cache parallel GEMM
gemm_sector: gemm_sector.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Comparison benchmark (pre-packed vs traditional)
bench_compare: bench_compare.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Simple sector cache comparison
bench_sector_simple: bench_sector_simple.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# INT8 Flash Attention benchmark
bench_flash_int8: bench_flash_int8.c flash_attention_int8.c flash_attention_int8.h
	$(CC) $(CFLAGS) -o $@ bench_flash_int8.c flash_attention_int8.c $(LDFLAGS) -lm

# INT8 Softmax with Stochastic Rounding benchmark
bench_softmax_int8: bench_softmax_int8.c softmax_int8.c softmax_int8.h flash_attention_int8.c flash_attention_int8.h
	$(CC) $(CFLAGS) -o $@ bench_softmax_int8.c softmax_int8.c flash_attention_int8.c $(LDFLAGS) -lm

# Q@K^T GEMM benchmark
bench_qkt_gemm: bench_qkt_gemm.c flash_attention_int8.c flash_attention_int8.h
	$(CC) $(CFLAGS) -o $@ bench_qkt_gemm.c flash_attention_int8.c $(LDFLAGS) -lm

# L2-Aware Tiled GEMM benchmark (native fcc for best performance)
# Objects for tiled GEMM
GQA_KERNEL_OBJ = gqa_micro_kernel_simple.o
GQA_PACK_OBJ = gqa_pack.o
GQA_TILED_OBJ = gqa_gemm_tiled.o

$(GQA_KERNEL_OBJ): gqa_micro_kernel_simple.S
	$(CC_FCC) -c -o $@ $<

$(GQA_PACK_OBJ): gqa_pack.c gqa_pack.h
	$(CC_FCC) -O3 -mcpu=a64fx -c -o $@ $<

$(GQA_TILED_OBJ): gqa_gemm_tiled.c gqa_gemm_tiled.h gqa_pack.h
	$(CC_FCC) -O3 -mcpu=a64fx -c -o $@ $<

bench_tiled_l2: bench_tiled_l2.c $(GQA_TILED_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) gqa_gemm_tiled.h gqa_pack.h
	$(CC_FCC) -O3 -mcpu=a64fx -o $@ bench_tiled_l2.c $(GQA_TILED_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) -lpthread -lm

# INT32 exp2 approximation benchmark
EXP2_OBJ = exp2_int.o

$(EXP2_OBJ): exp2_int.c exp2_int.h
	$(CC_FCC) -O3 -mcpu=a64fx -c -o $@ $<

bench_exp2: bench_exp2.c $(EXP2_OBJ) softmax_int8.o exp2_int.h softmax_int8.h
	$(CC_FCC) -O3 -mcpu=a64fx -o $@ bench_exp2.c $(EXP2_OBJ) softmax_int8.o -lpthread -lm

softmax_int8.o: softmax_int8.c softmax_int8.h
	$(CC_FCC) -O3 -mcpu=a64fx -c -o $@ $<

# Fused GEMM + Softmax benchmark
FUSED_ATTN_OBJ = fused_attention.o

$(FUSED_ATTN_OBJ): fused_attention.c fused_attention.h gqa_gemm_tiled.h gqa_pack.h exp2_int.h
	$(CC_FCC) -O3 -mcpu=a64fx -c -o $@ $<

# Optimized fused attention
FUSED_ATTN_OPT_OBJ = fused_attention_opt.o

$(FUSED_ATTN_OPT_OBJ): fused_attention_opt.c fused_attention_opt.h gqa_gemm_tiled.h gqa_pack.h exp2_int.h
	$(CC_FCC) -O3 -mcpu=a64fx -c -o $@ $<

bench_fused_attn: bench_fused_attn.c $(FUSED_ATTN_OBJ) $(FUSED_ATTN_OPT_OBJ) $(GQA_TILED_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) fused_attention.h fused_attention_opt.h
	$(CC_FCC) -O3 -mcpu=a64fx -o $@ bench_fused_attn.c $(FUSED_ATTN_OBJ) $(FUSED_ATTN_OPT_OBJ) $(GQA_TILED_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) -lpthread -lm

# SVE Assembly exp2 kernel benchmark
EXP2_SVE_KERNEL_OBJ = exp2_sve_kernel.o

$(EXP2_SVE_KERNEL_OBJ): exp2_sve_kernel.S
	$(CC_FCC) -c -o $@ $<

bench_exp2_sve: bench_exp2_sve.c $(EXP2_SVE_KERNEL_OBJ) $(EXP2_OBJ) $(GQA_PACK_OBJ) exp2_sve_kernel.h exp2_int.h
	$(CC_FCC) -O3 -mcpu=a64fx -o $@ bench_exp2_sve.c $(EXP2_SVE_KERNEL_OBJ) $(EXP2_OBJ) $(GQA_PACK_OBJ) -lpthread -lm

# Complete Attention Forward Pass (Single-Core)
ATTENTION_FULL_OBJ = attention_full.o

$(ATTENTION_FULL_OBJ): attention_full.c attention_full.h gqa_gemm_tiled.h gqa_pack.h exp2_int.h fused_attention.h
	$(CC_FCC) -O3 -mcpu=a64fx -c -o $@ $<

bench_attention: bench_attention.c $(ATTENTION_FULL_OBJ) $(GQA_TILED_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) $(FUSED_ATTN_OBJ) attention_full.h
	$(CC_FCC) -O3 -mcpu=a64fx -o $@ bench_attention.c $(ATTENTION_FULL_OBJ) $(GQA_TILED_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) $(FUSED_ATTN_OBJ) -lpthread -lm

# FlashAttention-style Fused Attention
FLASH_ATTN_FUSED_OBJ = flash_attn_fused.o

$(FLASH_ATTN_FUSED_OBJ): flash_attn_fused.c flash_attn_fused.h gqa_pack.h gqa_gemm_tiled.h exp2_int.h attention_full.h fused_attention.h
	$(CC_FCC) -O3 -mcpu=a64fx -c -o $@ $<

bench_flash_fused: bench_flash_fused.c $(FLASH_ATTN_FUSED_OBJ) $(ATTENTION_FULL_OBJ) $(GQA_TILED_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) $(FUSED_ATTN_OBJ) flash_attn_fused.h
	$(CC_FCC) -O3 -mcpu=a64fx -o $@ bench_flash_fused.c $(FLASH_ATTN_FUSED_OBJ) $(ATTENTION_FULL_OBJ) $(GQA_TILED_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) $(FUSED_ATTN_OBJ) -lpthread -lm

# GEMM + Online Softmax benchmark (no integer division)
bench_flash_gemm: bench_flash_gemm.c $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) gqa_pack.h exp2_int.h
	$(CC_FCC) -O3 -mcpu=a64fx -o $@ bench_flash_gemm.c $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) -lpthread -lm

# Optimized GEMM kernel
GEMM_OPT_OBJ = gemm_kernel_opt.o

$(GEMM_OPT_OBJ): gemm_kernel_opt.S
	$(CC_FCC) -c -o $@ $<

# Optimized GEMM + Softmax benchmark
bench_flash_gemm_opt: bench_flash_gemm_opt.c $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) $(GEMM_OPT_OBJ) gqa_pack.h exp2_int.h
	$(CC_FCC) -O3 -mcpu=a64fx -o $@ bench_flash_gemm_opt.c $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) $(GEMM_OPT_OBJ) -lpthread -lm

# Fused FlashAttention V2 (L1D-optimized, full fusion)
FLASH_FUSED_V2_OBJ = flash_attention_fused_v2.o

$(FLASH_FUSED_V2_OBJ): flash_attention_fused_v2.c flash_attention_fused_v2.h gqa_pack.h exp2_int.h
	$(CC_FCC) -O3 -mcpu=a64fx -c -o $@ $<

bench_flash_fused_v2: bench_flash_fused_v2.c $(FLASH_FUSED_V2_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) flash_attention_fused_v2.h
	$(CC_FCC) -O3 -mcpu=a64fx -fopenmp -o $@ bench_flash_fused_v2.c $(FLASH_FUSED_V2_OBJ) $(GQA_PACK_OBJ) $(GQA_KERNEL_OBJ) $(EXP2_OBJ) -lpthread -lm

clean:
	rm -f $(TARGETS) $(KERNEL_OBJ) $(KERNEL_L2_OBJ) $(GQA_KERNEL_OBJ) $(GQA_PACK_OBJ) $(GQA_TILED_OBJ) $(EXP2_OBJ) $(FUSED_ATTN_OBJ) $(FUSED_ATTN_OPT_OBJ) $(EXP2_SVE_KERNEL_OBJ) $(ATTENTION_FULL_OBJ) $(FLASH_ATTN_FUSED_OBJ) $(GEMM_OPT_OBJ) $(FLASH_FUSED_V2_OBJ) softmax_int8.o

# Run benchmarks
run: all
	@echo "=== Running C Intrinsic Version ==="
	./bench_int8_sdot 1000000
	@echo ""
	@echo "=== Running ASM Kernel Version ==="
	./bench_int8_sdot_asm 1000000

# Run L2 streaming benchmark
run-l2: bench_int8_l2stream
	@echo "=== Running L2 Streaming Benchmark ==="
	./bench_int8_l2stream 1024 10000

# Quick test
test: all
	./bench_int8_sdot 10000
	./bench_int8_sdot_asm 10000
	./bench_int8_l2stream 256 1000

# Run INT8 Flash Attention benchmark
run-flash: bench_flash_int8
	@echo "=== INT8 Flash Attention Benchmark ==="
	./bench_flash_int8

# Help
help:
	@echo "INT8 SDOT GEMM Benchmark"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build all targets (default compiler: fccpx)"
	@echo "  make COMPILER=gcc - Build with GCC"
	@echo "  make run          - Run L1 benchmarks"
	@echo "  make run-l2       - Run L2 streaming benchmark"
	@echo "  make test         - Quick test"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Targets:"
	@echo "  bench_int8_sdot      - C intrinsic version (L1)"
	@echo "  bench_int8_sdot_asm  - C + ASM kernel version (L1)"
	@echo "  bench_int8_l2stream  - L2 streaming with sector cache"
