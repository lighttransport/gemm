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
LLM_RUNNER_SRC = ../llm/cuda_llm_runner.c
TEST_SRC = test_cuda_vision.c

# Targets
TARGET = test_cuda_vision
COMPARE = compare_vs_llamacpp
VLM = test_cuda_vlm

.PHONY: all clean run help compare

all: $(TARGET) $(COMPARE)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) cuda_vision_encoder.h ../cuew.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(COMPARE): compare_vs_llamacpp.c $(RUNNER_SRC) $(CUEW_SRC) cuda_vision_encoder.h ../cuew.h
	$(CC) $(CFLAGS) -o $@ compare_vs_llamacpp.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VLM): test_cuda_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(CUEW_SRC) cuda_vision_encoder.h ../llm/cuda_llm_runner.h ../cuew.h
	$(CC) $(CFLAGS) -o $@ test_cuda_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

# dump_clip_embd requires llama.cpp build (link against libmtmd.so)
# Usage: make dump_clip_embd LLAMA_DIR=/path/to/llama.cpp
LLAMA_DIR ?= /home/syoyo/work/llama.cpp
dump_clip_embd: dump_clip_embd.cpp
	g++ -O2 -std=c++17 \
		-I$(LLAMA_DIR)/vendor/stb \
		-I$(LLAMA_DIR)/tools/mtmd \
		-I$(LLAMA_DIR)/ggml/include \
		-I$(LLAMA_DIR)/include \
		-L$(LLAMA_DIR)/build/bin \
		-Wl,-rpath,$(LLAMA_DIR)/build/bin \
		-o $@ $< -lmtmd -lllama -lm

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <mmproj.gguf> [--f16] [--image-size N]"

clean:
	rm -f $(TARGET) $(COMPARE) $(VLM) dump_clip_embd

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
