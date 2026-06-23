# Makefile for CUDA VLM vision encoder
#
# No nvcc required - kernels compiled at runtime via NVRTC.
# Only needs gcc + CUDA runtime libraries (libcuda.so, libnvrtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I.. -mavx2 -mfma -march=native
LDFLAGS = -ldl -lm -lpthread -lstdc++

# Sources
CUEW_SRC = ../cuew.c
CBLASW_SRC = ../cublasew.c
RUNNER_SRC = cuda_vision_encoder.c
LLM_RUNNER_SRC = ../llm/cuda_llm_runner.c
TEST_SRC = test_cuda_vision.c

# Targets
TARGET = test_cuda_vision
COMPARE = compare_vs_llamacpp
VLM = test_cuda_vlm
GEMMA4_VLM = test_cuda_gemma4_vlm
LLAMACPP_VISION = llamacpp_vision_standalone

.PHONY: all clean run help compare

all: $(TARGET) $(COMPARE)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) cuda_vision_encoder.h ../cuew.h ../cublasew.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) $(LDFLAGS)

$(COMPARE): compare_vs_llamacpp.c $(RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) cuda_vision_encoder.h ../cuew.h ../cublasew.h
	$(CC) $(CFLAGS) -o $@ compare_vs_llamacpp.c $(RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) $(LDFLAGS)

# llama.cpp vision bridge (for reference path)
LLAMA_DIR ?= /home/syoyo/work/llama.cpp
LLAMA_CXXFLAGS = -O2 -std=c++17 \
	-I$(LLAMA_DIR)/tools/mtmd \
	-I$(LLAMA_DIR)/ggml/include \
	-I$(LLAMA_DIR)/include
LLAMA_LDFLAGS = \
	-L$(LLAMA_DIR)/build/bin \
	-Wl,-rpath,$(LLAMA_DIR)/build/bin \
	-lmtmd -lllama -lggml-base -lggml-cpu -lggml -lstdc++ -lm -lpthread

LLAMACPP_VISION_BIN = llamacpp_vision_standalone

$(LLAMACPP_VISION_BIN): llamacpp_vision_standalone.cpp
	g++ -O2 -std=c++17 \
	  $(LLAMA_CXXFLAGS) \
	  -I$(LLAMA_DIR)/tools/mtmd/include/stb \
	  -o $@ $< \
	  -L$(LLAMA_DIR)/build/bin \
	  -Wl,-rpath,$(LLAMA_DIR)/build/bin \
	  -lmtmd -lllama -lggml-base -lggml-cpu -lggml -lm -lpthread

$(VLM): test_cuda_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) $(LLAMACPP_BRIDGE_SO) \
        cuda_vision_encoder.h ../llm/cuda_llm_runner.h ../cuew.h ../cublasew.h llamacpp_vision_bridge.h
	$(CC) $(CFLAGS) -I$(LLAMA_DIR)/tools/mtmd -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/include \
	  -o $@ test_cuda_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) \
	  -L. -Wl,-rpath,'$$ORIGIN' $(LLAMACPP_BRIDGE_SO) \
	  $(LDFLAGS) $(LLAMA_LDFLAGS)

$(GEMMA4_VLM): test_cuda_gemma4_vlm.c $(LLM_RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) ../llm/cuda_llm_runner.h ../cuew.h ../cublasew.h
	$(CC) $(CFLAGS) -o $@ test_cuda_gemma4_vlm.c $(LLM_RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) $(LDFLAGS)

# Profile build: enables per-step timing inside g4v_encode_gemma4uv.
$(GEMMA4_VLM)_profile: test_cuda_gemma4_vlm.c $(LLM_RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) ../llm/cuda_llm_runner.h ../cuew.h ../cublasew.h
	$(CC) $(CFLAGS) -DGEMMA4_PROFILE_MS -o $@ test_cuda_gemma4_vlm.c $(LLM_RUNNER_SRC) $(CUEW_SRC) $(CBLASW_SRC) $(LDFLAGS)

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
	rm -f $(TARGET) $(COMPARE) $(VLM) $(GEMMA4_VLM) dump_clip_embd

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
