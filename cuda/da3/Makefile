# Makefile for CUDA DA3 depth estimation runner
#
# No nvcc required - kernels compiled at runtime via NVRTC.
# Only needs gcc + CUDA runtime libraries (libcuda.so, libnvrtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I.. -mavx2 -mfma -march=native
LDFLAGS = -ldl -lm -lpthread

# Sources
CUEW_SRC = ../cuew.c
RUNNER_SRC = cuda_da3_runner.c
TEST_SRC = test_cuda_da3.c

# Target
TARGET = test_cuda_da3

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) cuda_da3_runner.h ../cuew.h \
           ../../common/gguf_loader.h ../../common/ggml_dequant.h ../../common/depth_anything3.h \
           ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <da3.gguf> [-i input.ppm] [-o output.pgm] [-d device_id]"

clean:
	rm -f $(TARGET)

help:
	@echo "CUDA DA3 Depth Estimation Runner (NVRTC-based, no nvcc needed)"
	@echo "==============================================================="
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_cuda_da3"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_cuda_da3 <da3.gguf> [-i input.ppm] [-o output.pgm] [-d device_id]"
	@echo ""
	@echo "Example:"
	@echo "  ./test_cuda_da3 da3-small-f16.gguf -i photo.ppm -o depth.pgm"
	@echo ""
	@echo "Architecture: GPU backbone (12 ViT blocks) + CPU DPT head"
	@echo ""
	@echo "Requirements:"
	@echo "  - libcuda.so (CUDA driver)"
	@echo "  - libnvrtc.so (NVRTC runtime compiler)"
	@echo "  - GPU with sm_70+ (V100 through Blackwell)"
