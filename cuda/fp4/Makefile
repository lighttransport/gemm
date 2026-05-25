# Makefile for FP4 GEMM / mma probe using cuew
#
# No CUDA SDK required at compile time - cuew dynamically loads the driver API,
# cuBLASLt is dlopen'd at runtime, and the mma probe compiles its kernel with
# NVRTC at runtime (targeting sm_120a).

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm

CUEW = ../cuew.c ../cuew.h

.PHONY: all clean cublas probe help

all: cublas_fp4_gemm mma_fp4_probe

# cuBLASLt FP4 GEMM (NVFP4 vs MXFP4 vs BF16/FP8)
cublas_fp4_gemm: cublas_fp4_gemm.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ cublas_fp4_gemm.c ../cuew.c $(LDFLAGS)

# Raw mma.sync issue-rate probe (e2m1 vs e4m3 vs bf16), NVRTC sm_120a
mma_fp4_probe: mma_fp4_probe.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ mma_fp4_probe.c ../cuew.c $(LDFLAGS)

cublas: cublas_fp4_gemm
	./cublas_fp4_gemm

probe: mma_fp4_probe
	./mma_fp4_probe

clean:
	rm -f cublas_fp4_gemm mma_fp4_probe

help:
	@echo "make            - build both binaries"
	@echo "make cublas     - build+run cuBLASLt FP4 GEMM (NVFP4/MXFP4 vs BF16/FP8)"
	@echo "make probe      - build+run raw mma.sync issue-rate probe (sm_120a)"
	@echo "make clean      - remove binaries"
