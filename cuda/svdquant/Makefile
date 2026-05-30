# cuda/svdquant — CUDA SVDQuant forward validation.
#
# No CUDA SDK at compile time: cuew dlopen's the driver API, cublasew dlopen's
# cuBLAS, and cuda/fp4_w4a4.h's kernels are NVRTC-compiled at runtime (sm_120a).
# Reuses the host decode helpers from cpu/svdquant/svdquant_cpu.h and the .npy
# loader from common/npy_io.h.

CC      ?= gcc
CFLAGS   = -O2 -Wall -Wextra -I.. -I../../common -I../../cpu/svdquant
LDFLAGS  = -ldl -lm
DUMPS    = ../../ref/svdquant/dumps

CUEW = ../cuew.c ../cuew.h
DEPS = ../fp4_w4a4.h ../cublasew.h ../cuew.h \
       ../../common/npy_io.h ../../cpu/svdquant/svdquant_cpu.h

.PHONY: all test clean

all: test_cuda_svdquant

test_cuda_svdquant: test_cuda_svdquant.c $(DEPS) $(CUEW) ../cublasew.c
	$(CC) $(CFLAGS) -o $@ test_cuda_svdquant.c ../cuew.c ../cublasew.c $(LDFLAGS)

$(DUMPS)/manifest.json:
	cd ../../ref/svdquant && python3 gen_svdquant_ref.py --out dumps

test: test_cuda_svdquant $(DUMPS)/manifest.json
	./test_cuda_svdquant $(DUMPS)

clean:
	rm -f test_cuda_svdquant
