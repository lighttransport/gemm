# Makefile for CUDA VLM vision encoder
#
# No nvcc required - kernels compiled at runtime via NVRTC.
# Only needs gcc + CUDA runtime libraries (libcuda.so, libnvrtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I.. -mavx2 -mfma -march=native
LDFLAGS = -ldl -lm -lpthread

# Sources
CUEW_SRC = ../cuew.c
RUNNER_SRC = cuda_vision_encoder.c
TEST_SRC = test_cuda_vision.c

# Target
TARGET = test_cuda_vision

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) cuda_vision_encoder.h ../cuew.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <mmproj.gguf> [--f16] [--image-size N]"

clean:
	rm -f $(TARGET)

help:
	@echo "CUDA VLM Vision Encoder (NVRTC-based, no nvcc needed)"
	@echo "====================================================="
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_cuda_vision"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_cuda_vision <mmproj.gguf> [--f16] [--image-size N]"
	@echo ""
	@echo "Example:"
	@echo "  ./test_cuda_vision /mnt/disk01/models/qwen3-5/mmproj-F32.gguf"
	@echo "  ./test_cuda_vision /mnt/disk01/models/qwen3-5/mmproj-F32.gguf --f16"
	@echo ""
	@echo "Requirements:"
	@echo "  - libcuda.so (CUDA driver)"
	@echo "  - libnvrtc.so (NVRTC runtime compiler)"
	@echo "  - GPU with sm_70+ (V100 through Blackwell)"
