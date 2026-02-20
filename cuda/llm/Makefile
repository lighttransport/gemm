# Makefile for CUDA LLM runner
#
# No nvcc required - kernels compiled at runtime via NVRTC.
# Only needs gcc + CUDA runtime libraries (libcuda.so, libnvrtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I.. -mavx2 -mfma -march=native
LDFLAGS = -ldl -lm -lpthread

# Sources
CUEW_SRC = ../cuew.c
RUNNER_SRC = cuda_llm_runner.c
TEST_SRC = test_cuda_llm.c

# Target
TARGET = test_cuda_llm

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) cuda_llm_runner.h ../cuew.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]"

clean:
	rm -f $(TARGET)

help:
	@echo "CUDA LLM Runner (NVRTC-based, no nvcc needed)"
	@echo "=============================================="
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_cuda_llm"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_cuda_llm <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]"
	@echo ""
	@echo "Example:"
	@echo "  ./test_cuda_llm Qwen3-Embedding-0.6B-f16.gguf -t \"Hello world\" -n 4"
	@echo ""
	@echo "Requirements:"
	@echo "  - libcuda.so (CUDA driver)"
	@echo "  - libnvrtc.so (NVRTC runtime compiler)"
	@echo "  - GPU with sm_70+ (V100 through Blackwell)"
