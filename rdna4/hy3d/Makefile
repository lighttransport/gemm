# Makefile for HIP/ROCm Hunyuan3D-2.1 runner (RDNA4)
#
# No hipcc required - kernels compiled at runtime via HIPRTC.
# Only needs gcc + ROCm runtime libraries (libamdhip64.so, libhiprtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm -lpthread

ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_hy3d_runner.c

HDRS = hip_hy3d_runner.h hip_hy3d_kernels.h hip_hy3d_ops.h \
       ../rocew.h ../hip_kernels_common.h ../hip_runner_common.h \
       ../../common/safetensors.h ../../common/marching_cubes.h

.PHONY: all verify clean help

all: test_hip_hy3d wmma_selftest

verify: verify_dinov2 verify_vae verify_dit

test_hip_hy3d: test_hip_hy3d.c $(RUNNER_SRC) $(ROCEW_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ test_hip_hy3d.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_dinov2: verify_dinov2.c $(RUNNER_SRC) $(ROCEW_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_dinov2.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_vae: verify_vae.c $(RUNNER_SRC) $(ROCEW_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_vae.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_dit: verify_dit.c $(RUNNER_SRC) $(ROCEW_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ verify_dit.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

wmma_selftest: wmma_selftest.c $(ROCEW_SRC) $(HDRS)
	$(CC) $(CFLAGS) -o $@ wmma_selftest.c $(ROCEW_SRC) $(LDFLAGS)

clean:
	rm -f test_hip_hy3d verify_dinov2 verify_vae verify_dit wmma_selftest

help:
	@echo "HIP/ROCm Hunyuan3D-2.1 Runner (HIPRTC-based, no hipcc needed)"
	@echo "================================================================"
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_hip_hy3d"
	@echo "  make verify       - Build verification tools (verify_dinov2, verify_vae)"
	@echo ""
	@echo "Verify:"
	@echo "  ./verify_dinov2 <conditioner.safetensors> [--ref-dir ref/output/]"
	@echo "  ./verify_vae <vae.safetensors> [--ref-dir ref/output/] [--grid-res 8]"
	@echo ""
	@echo "Run:"
	@echo "  ./test_hip_hy3d <cond.st> <model.st> <vae.st> [-i image.ppm] [-o output.obj]"
