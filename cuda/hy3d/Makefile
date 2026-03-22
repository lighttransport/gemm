# Makefile for CUDA Hunyuan3D-2.1 runner
#
# No nvcc required - kernels compiled at runtime via NVRTC.
# Only needs gcc + CUDA runtime libraries (libcuda.so, libnvrtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC = ../cuew.c
RUNNER_SRC = cuda_hy3d_runner.c

HDRS = cuda_hy3d_runner.h cuda_hy3d_kernels.h cuda_hy3d_ops.h \
       ../cuew.h ../cuda_kernels_common.h ../cuda_runner_common.h \
       ../../common/safetensors.h ../../common/marching_cubes.h

.PHONY: all verify clean help

all: test_cuda_hy3d

verify: verify_dinov2 verify_vae

test_cuda_hy3d: test_cuda_hy3d.c $(RUNNER_SRC) $(CUEW_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ test_cuda_hy3d.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_dinov2: verify_dinov2.c $(RUNNER_SRC) $(CUEW_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_dinov2.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_vae: verify_vae.c $(RUNNER_SRC) $(CUEW_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_vae.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

clean:
	rm -f test_cuda_hy3d verify_dinov2 verify_vae

help:
	@echo "CUDA Hunyuan3D-2.1 Runner (NVRTC-based, no nvcc needed)"
	@echo "========================================================"
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_cuda_hy3d"
	@echo "  make verify       - Build verification tools (verify_dinov2, verify_vae)"
	@echo ""
	@echo "Verify:"
	@echo "  ./verify_dinov2 <conditioner.safetensors> [--ref-dir ref/output/]"
	@echo "  ./verify_vae <vae.safetensors> [--ref-dir ref/output/] [--grid-res 8]"
	@echo ""
	@echo "Run:"
	@echo "  ./test_cuda_hy3d <cond.st> <model.st> <vae.st> [-i image.ppm] [-o output.obj]"
