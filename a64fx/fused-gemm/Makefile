# Makefile for Fused GEMM (A@B)@C on A64FX
# Uses Fujitsu compiler for A64FX native compilation

# Compiler selection
# fcc: Fujitsu C native compiler for A64FX
# fccpx: Fujitsu C cross compiler (for Intel host)
CC = fcc
# For cross-compilation from Intel host:
# CC = fccpx

# Compiler flags
CFLAGS = -O3 -Kfast,openmp
# -O3: High optimization level
# -Kfast: Fujitsu fast math
# -Kopenmp: Enable OpenMP
# SVE is enabled by default on A64FX

# Assembly flags
ASFLAGS = -c

# Linker flags
LDFLAGS = -Kopenmp -lm

# Source files
SRCS = fused_gemm.c pack_matrices.c bench_fused.c
ASM_SRCS = micro_kernel_8x3.S
OBJS = $(SRCS:.c=.o) $(ASM_SRCS:.S=.o)

# Target
TARGET = bench_fused

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c fused_gemm.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.S
	$(CC) $(ASFLAGS) -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

# Run benchmark
run: $(TARGET)
	./$(TARGET)

# Submit job to Fugaku
submit: $(TARGET)
	pjsub -g hp250467 -L "freq=2200,eco_state=0,rscgrp=small,node=1,elapse=00:10:00" --no-check-directory ./run.sh

# Generate assembly listing for debugging
fused_gemm.s: fused_gemm.c fused_gemm.h
	$(CC) $(CFLAGS) -S -o $@ $<

pack_matrices.s: pack_matrices.c fused_gemm.h
	$(CC) $(CFLAGS) -S -o $@ $<

# Debug build
debug: CFLAGS = -g -O0 -Kopenmp,sve -Nclang
debug: clean $(TARGET)

# Profile build (with FAPP support)
profile: CFLAGS += -Ntl_notrt
profile: clean $(TARGET)

# ─── 2-Pass Softmax + P@V benchmark ─────────────────────
# Uses -Nclang for arm_sve.h intrinsics + FEXPA inline asm
2PASS_CC = fcc
2PASS_CFLAGS = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
2PASS_ASFLAGS = -c
2PASS_LDFLAGS = -lm

# Copy 8x3 kernel from int8-cmg if not present locally
../int8-cmg/micro_kernel_fp32_8x3.S:
	@echo "ERROR: micro_kernel_fp32_8x3.S not found in ../int8-cmg/"
	@exit 1

2PASS_KERN_OBJS = micro_kernel_fp32_8x3.o micro_kernel_fp32_8x3_accum.o \
                  micro_kernel_fp32_6x4_bcast.o micro_kernel_fp32_6x4_bcast_accum.o \
                  micro_kernel_fp32_10x2.o micro_kernel_fp32_10x2_accum.o \
                  micro_kernel_fp32_11x2.o micro_kernel_fp32_11x2_accum.o \
                  micro_kernel_fp32_12x2.o micro_kernel_fp32_12x2_accum.o \
                  micro_kernel_fp32_12x2_swp.o micro_kernel_fp32_12x2_swp_accum.o \
                  micro_kernel_fp16_12x2_swp.o micro_kernel_fp16_12x2_swp_accum.o \
                  micro_kernel_fp16_12x2_csplit.o micro_kernel_fp16_12x2_csplit_accum.o \
                  micro_kernel_fp16_12x2_dswp.o micro_kernel_fp16_12x2_dswp_accum.o \
                  micro_kernel_fp16_12x2_4k.o micro_kernel_fp16_12x2_4k_accum.o \
                  micro_kernel_fp16_8x3.o micro_kernel_fp16_8x3_accum.o \
                  micro_kernel_fp16_6x4.o micro_kernel_fp16_6x4_accum.o \
                  micro_kernel_fp16_12x2_bpre.o micro_kernel_fp16_12x2_bpre_accum.o \
                  micro_kernel_fp16_12x2_noepi.o micro_kernel_fp16_12x2_noepi_accum.o \
                  micro_kernel_fp16_12x2_noepi4.o micro_kernel_fp16_12x2_noepi4_accum.o

bench_2pass: bench_2pass.o $(2PASS_KERN_OBJS)
	$(2PASS_CC) $(2PASS_CFLAGS) -o $@ $^ $(2PASS_LDFLAGS)

bench_2pass.o: bench_2pass.c
	$(2PASS_CC) $(2PASS_CFLAGS) -c -o $@ $<

micro_kernel_fp32_%.o: micro_kernel_fp32_%.S
	$(2PASS_CC) $(2PASS_ASFLAGS) -o $@ $<

micro_kernel_fp16_%.o: micro_kernel_fp16_%.S
	$(2PASS_CC) $(2PASS_ASFLAGS) -o $@ $<

clean_2pass:
	rm -f bench_2pass bench_2pass.o $(2PASS_KERN_OBJS)

submit_2pass: bench_2pass
	pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00" --no-check-directory ./run_2pass.sh

# ─── L1-Resident Ceiling Benchmark ─────────────────────
L1_CEIL_KERN_OBJS = micro_kernel_fp32_12x2_swp.o micro_kernel_fp32_12x2_swp_accum.o \
                    micro_kernel_fp32_8x3.o micro_kernel_fp32_8x3_accum.o \
                    micro_kernel_fp32_6x4_bcast.o micro_kernel_fp32_6x4_bcast_accum.o \
                    micro_kernel_fp32_6x4_swp.o micro_kernel_fp32_6x4_swp_accum.o

bench_l1_ceiling: bench_l1_ceiling.o $(L1_CEIL_KERN_OBJS)
	$(2PASS_CC) $(2PASS_CFLAGS) -o $@ $^ $(2PASS_LDFLAGS)

bench_l1_ceiling.o: bench_l1_ceiling.c
	$(2PASS_CC) $(2PASS_CFLAGS) -c -o $@ $<

# ─── FP32 90% Target Fused Pipeline Benchmark ──────────
FP32_90_KERN_OBJS = micro_kernel_fp32_12x2_swp.o micro_kernel_fp32_12x2_swp_accum.o \
                    micro_kernel_fp32_8x3.o micro_kernel_fp32_8x3_accum.o \
                    micro_kernel_fp32_6x4_bcast.o micro_kernel_fp32_6x4_bcast_accum.o \
                    micro_kernel_fp32_6x4_swp.o micro_kernel_fp32_6x4_swp_accum.o

bench_fp32_90: bench_fp32_90.o $(FP32_90_KERN_OBJS)
	$(2PASS_CC) $(2PASS_CFLAGS) -o $@ $^ $(2PASS_LDFLAGS)

bench_fp32_90.o: bench_fp32_90.c
	$(2PASS_CC) $(2PASS_CFLAGS) -c -o $@ $<

clean_fp32_90:
	rm -f bench_fp32_90 bench_fp32_90.o bench_l1_ceiling bench_l1_ceiling.o \
	      $(L1_CEIL_KERN_OBJS) $(FP32_90_KERN_OBJS)

# ─── Multi-Core FP32 Fused Pipeline Benchmark ────────────
MC_KERN_OBJS = micro_kernel_fp32_12x2_swp.o micro_kernel_fp32_12x2_swp_accum.o

bench_fp32_90_mc: bench_fp32_90_mc.o $(MC_KERN_OBJS)
	$(2PASS_CC) $(2PASS_CFLAGS) -fopenmp -o $@ $^ $(2PASS_LDFLAGS)

bench_fp32_90_mc.o: bench_fp32_90_mc.c
	$(2PASS_CC) $(2PASS_CFLAGS) -fopenmp -c -o $@ $<

clean_mc:
	rm -f bench_fp32_90_mc bench_fp32_90_mc.o $(MC_KERN_OBJS)

# ─── Optimized FP32 PA=0 + Kc Sweep Benchmark ────────────
OPT_KERN_OBJS = micro_kernel_fp32_12x2_swp.o micro_kernel_fp32_12x2_swp_accum.o \
                micro_kernel_fp32_8x3.o micro_kernel_fp32_8x3_accum.o \
                micro_kernel_fp32_6x4_bcast.o micro_kernel_fp32_6x4_bcast_accum.o \
                micro_kernel_fp32_6x4_swp.o micro_kernel_fp32_6x4_swp_accum.o

bench_fp32_opt: bench_fp32_opt.o $(OPT_KERN_OBJS)
	$(2PASS_CC) $(2PASS_CFLAGS) -fopenmp -o $@ $^ $(2PASS_LDFLAGS)

bench_fp32_opt.o: bench_fp32_opt.c
	$(2PASS_CC) $(2PASS_CFLAGS) -fopenmp -c -o $@ $<

clean_opt:
	rm -f bench_fp32_opt bench_fp32_opt.o $(OPT_KERN_OBJS)
