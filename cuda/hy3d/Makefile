# Makefile for CUDA Hunyuan3D-2.1 runner
#
# No nvcc required - kernels compiled at runtime via NVRTC.
# Only needs gcc + CUDA runtime libraries (libcuda.so, libnvrtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC = ../cuew.c
RUNNER_SRC = cuda_hy3d_runner.c
TEST_SRC = test_cuda_hy3d.c

TARGET = test_cuda_hy3d

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) cuda_hy3d_runner.h ../cuew.h \
           ../cuda_kernels_common.h ../cuda_runner_common.h \
           ../../common/safetensors.h ../../common/marching_cubes.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

help:
	@echo "CUDA Hunyuan3D-2.1 Runner (NVRTC-based, no nvcc needed)"
	@echo "========================================================"
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_cuda_hy3d"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_cuda_hy3d <conditioner.st> <model.st> <vae.st> [-i image.ppm] [-o output.obj]"
	@echo ""
	@echo "Options:"
	@echo "  -i FILE       Input PPM image"
	@echo "  -o FILE       Output OBJ mesh (default: output.obj)"
	@echo "  --ply FILE    Also write PLY mesh"
	@echo "  -s N          Diffusion steps (default: 30)"
	@echo "  -g FLOAT      CFG guidance scale (default: 7.5)"
	@echo "  --grid N      Marching cubes grid resolution (default: 256)"
	@echo "  --seed N      Random seed (default: 42)"
	@echo "  -d N          CUDA device ID (default: 0)"
	@echo ""
	@echo "Pipeline: Image -> DINOv2 -> DiT diffusion -> ShapeVAE -> MC -> Mesh"
	@echo ""
	@echo "Requirements:"
	@echo "  - libcuda.so (CUDA driver)"
	@echo "  - libnvrtc.so (NVRTC runtime compiler)"
	@echo "  - GPU with sm_70+ (V100 through Blackwell)"
