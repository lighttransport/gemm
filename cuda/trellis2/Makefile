# Makefile for CUDA TRELLIS.2 Stage 1 runner
#
# No nvcc required — kernels compiled at runtime via NVRTC.
# Only needs gcc + CUDA runtime libraries (libcuda.so, libnvrtc.so).

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC = ../cuew.c
RUNNER_SRC = cuda_trellis2_runner.c

HDRS = cuda_trellis2_runner.h cuda_trellis2_kernels.h cuda_trellis2_ops.h \
       ../cuew.h ../cuda_kernels_common.h ../cuda_runner_common.h \
       ../../common/safetensors.h ../../common/marching_cubes.h

.PHONY: all clean

all: test_cuda_trellis2

test_cuda_trellis2: test_cuda_trellis2.c $(RUNNER_SRC) $(CUEW_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ test_cuda_trellis2.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

clean:
	rm -f test_cuda_trellis2
