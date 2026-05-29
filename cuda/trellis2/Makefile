# Makefile for CUDA TRELLIS.2 Stage 1 runner
#
# No nvcc required — kernels compiled at runtime via NVRTC.
# Only needs gcc + CUDA runtime libraries (libcuda.so, libnvrtc.so).

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC = ../cuew.c
CUBLAS_SRC = ../cublasew.c
RUNNER_SRC = cuda_trellis2_runner.c

HDRS = cuda_trellis2_runner.h cuda_trellis2_kernels.h cuda_trellis2_ops.h \
       ../cuew.h ../cuda_kernels_common.h ../cuda_runner_common.h \
       ../cublasew.h \
       ../../common/safetensors.h ../../common/marching_cubes.h

.PHONY: all clean

all: test_cuda_trellis2 verify_decoder verify_stage1 verify_stage2 verify_stage2_full verify_stage3 verify_shape_decoder

test_cuda_trellis2: test_cuda_trellis2.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ test_cuda_trellis2.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(LDFLAGS)

# Shared library for the Python ctypes binding (ref/trellis2/trellis2_cmod.py).
# Same sources as the exe minus the test main; CUDA/NVRTC are dlopen'd via cuew.
libcuda_trellis2.so: $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(HDRS)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(LDFLAGS)

verify_decoder: verify_decoder.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_decoder.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(LDFLAGS)

verify_stage1: verify_stage1.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_stage1.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(LDFLAGS)

verify_stage2: verify_stage2.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_stage2.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(LDFLAGS)

verify_stage2_full: verify_stage2_full.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_stage2_full.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(LDFLAGS)

verify_stage3: verify_stage3.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_stage3.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(LDFLAGS)

verify_shape_decoder: verify_shape_decoder.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_shape_decoder.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLAS_SRC) $(LDFLAGS)

clean:
	rm -f test_cuda_trellis2 verify_decoder verify_stage1 verify_stage2 verify_stage2_full verify_stage3 verify_shape_decoder libcuda_trellis2.so
