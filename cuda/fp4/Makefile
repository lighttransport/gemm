# Makefile for FP4 GEMM / mma probe using cuew
#
# No CUDA SDK required at compile time - cuew dynamically loads the driver API,
# cuBLASLt is dlopen'd at runtime, and the mma probe compiles its kernel with
# NVRTC at runtime (targeting sm_120a).

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm

CUEW = ../cuew.c ../cuew.h

.PHONY: all clean cublas probe w4a4 help

BINS = cublas_fp4_gemm mma_fp4_probe fp4_gemm_test fp4_w4a4_gemm fp4_w4a4_qkv_test fp4_w4a4_opt

all: $(BINS)

# cuBLASLt FP4 GEMM (NVFP4 vs MXFP4 vs BF16/FP8)
cublas_fp4_gemm: cublas_fp4_gemm.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ cublas_fp4_gemm.c ../cuew.c $(LDFLAGS)

# Raw mma.sync issue-rate probe (e2m1 vs e4m3 vs bf16), NVRTC sm_120a
mma_fp4_probe: mma_fp4_probe.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ mma_fp4_probe.c ../cuew.c $(LDFLAGS)

# NVFP4 block-scale mma fragment+scale layout validation vs CPU ref (sm_120a)
fp4_gemm_test: fp4_gemm_test.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ fp4_gemm_test.c ../cuew.c $(LDFLAGS)

# Tiled W4A4 [M,N,K] GEMM (1 warp/16x8 tile) validated vs CPU ref (sm_120a)
fp4_w4a4_gemm: fp4_w4a4_gemm.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ fp4_w4a4_gemm.c ../cuew.c $(LDFLAGS)

# Full W4A4 op (quant+gemm+lora) on real block-0 to_qkv data in /tmp/qkv_*.bin
fp4_w4a4_qkv_test: fp4_w4a4_qkv_test.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ fp4_w4a4_qkv_test.c ../cuew.c $(LDFLAGS)

# Optimized W4A4 GEMM dev/verify bench: naive oracle vs shared-mem tiled (sync,
# shipped in cuda/qimg) vs cp.async double-buffer; all bit-exact, ~4-6x over naive
fp4_w4a4_opt: fp4_w4a4_opt.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ fp4_w4a4_opt.c ../cuew.c $(LDFLAGS)

cublas: cublas_fp4_gemm
	./cublas_fp4_gemm

probe: mma_fp4_probe
	./mma_fp4_probe

# Run the self-contained W4A4 validators (qkv_test needs /tmp/qkv_*.bin, run it manually)
w4a4: fp4_gemm_test fp4_w4a4_gemm
	./fp4_gemm_test
	./fp4_w4a4_gemm

clean:
	rm -f $(BINS)

help:
	@echo "make            - build all binaries"
	@echo "make cublas     - build+run cuBLASLt FP4 GEMM (NVFP4/MXFP4 vs BF16/FP8)"
	@echo "make probe      - build+run raw mma.sync issue-rate probe (sm_120a)"
	@echo "make w4a4       - build+run the W4A4 block-scale + tiled-GEMM validators"
	@echo "make clean      - remove binaries"
