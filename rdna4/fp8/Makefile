# Makefile for RDNA4 FP8 WMMA GEMM benchmark.
#
# Mirrors rdna4/vlm pattern: HIPRTC-compiled kernels, no hipcc at build time.
# Targets the 8-bit (e4m3) WMMA path on gfx1201 (RX 9070 XT).

CC = gcc
CXX = g++
HIPCC = hipcc
ROCM_PATH ?= /opt/rocm
ROCM_LLVM ?= $(ROCM_PATH)/llvm/bin

CFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

ROCEW_SRC = ../rocew.c
BENCH = bench_fp8_gemm
BENCH_FA = bench_fp8_fa

.PHONY: all clean run bench-mm0 bench-fa help

BENCH_EXTRACTED = bench_fp8_extracted

all: $(BENCH) $(BENCH_FA) $(BENCH_EXTRACTED)

$(BENCH_EXTRACTED): bench_fp8_extracted.cpp fp8_kernel_gfx1201.co
	$(HIPCC) -O3 --offload-arch=gfx1201 -o $@ bench_fp8_extracted.cpp

# Extract the FP8 kernel ELF from the local ROCm hipBLASLt Tensile library at
# build time. The artifact (~38 MB) is gitignored — re-derived per checkout.
fp8_kernel_gfx1201.co: extract_fp8_kernel.sh
	./extract_fp8_kernel.sh $@

$(BENCH): bench_fp8_gemm.c $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h
	$(CC) $(CFLAGS) -o $@ bench_fp8_gemm.c $(ROCEW_SRC) $(LDFLAGS) \
	    -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lstdc++

$(BENCH_FA): bench_fp8_fa.c $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h
	$(CC) $(CFLAGS) -o $@ bench_fp8_fa.c $(ROCEW_SRC) $(LDFLAGS) \
	    -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lstdc++

bench-mm0: $(BENCH)
	./$(BENCH) --shape mm0 --iters 200 --check

bench-fa: $(BENCH_FA)
	./$(BENCH_FA) --n-tok 1024 --heads 16 --head-dim 64 --iters 100

run: $(BENCH)
	./$(BENCH) --shape all --iters 200

clean:
	rm -f $(BENCH) $(BENCH_FA) $(BENCH_EXTRACTED) fp8_kernel_gfx1201.co

help:
	@echo "RDNA4 FP8 WMMA GEMM benchmark"
	@echo ""
	@echo "Build:"
	@echo "  make              - build $(BENCH)"
	@echo ""
	@echo "Run:"
	@echo "  ./$(BENCH) [--shape mm0|qkv|attn_out|ffn_up|ffn_down|all]"
	@echo "             [--mode baseline|pipe|all] [--iters N] [--check] [--no-bias]"
	@echo "             [--abs-max V]"
	@echo ""
	@echo "Reference: rdna4/vlm/optimized-gemm-guide.md, peak_efficiency_playbook.md"
