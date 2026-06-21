# Makefile for CUDA LLM runner

CC = gcc
NVCC = nvcc
CFLAGS = -O3 -Wall -Wextra -I.. -mavx2 -mfma -march=native
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC = ../cuew.c
CUBLASEW_SRC = ../cublasew.c
RUNNER_SRC = cuda_llm_runner.c
TEST_SRC = test_cuda_llm.c
MOE_CUBIN = moe_gpu_kernels.cubin
FA2_CUBIN = fa2_kernels.cubin
MMQ_CUBIN = mmq_kernels.cubin

# Vendored llama.cpp MMQ headers (mmq_vendor/) + the upstream tree for the rest.
# Override LLAMA_CUDA if llama.cpp lives elsewhere.
LLAMA_DIR ?= /home/syoyo/work/llama.cpp
LLAMA_CUDA = $(LLAMA_DIR)/ggml/src/ggml-cuda
MMQ_INCLUDES = -Immq_vendor -I$(LLAMA_CUDA) -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/ggml/src

TARGET = test_cuda_llm

.PHONY: all clean run help cubin

all: $(MOE_CUBIN) $(FA2_CUBIN) $(MMQ_CUBIN) $(TARGET)

cubin: $(MOE_CUBIN) $(FA2_CUBIN) $(MMQ_CUBIN)

$(MOE_CUBIN): moe_gpu_kernels.cu
	$(NVCC) -cubin -arch=sm_120 -o $@ $<

$(FA2_CUBIN): fa2_kernels.cu
	$(NVCC) -cubin -arch=sm_120 -o $@ $<

# Vendored llama.cpp mul_mat_q (stream-K int8 MMQ). sm_120a for Turing-MMA m16n8k32.
$(MMQ_CUBIN): mmq_kernels.cu mmq_vendor/mmq.cuh
	$(NVCC) -cubin -arch=sm_120a -std=c++17 -O3 $(MMQ_INCLUDES) -o $@ $<

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(CUBLASEW_SRC) $(MOE_CUBIN) $(FA2_CUBIN) \
          cuda_llm_runner.h ../cuew.h ../cublasew.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(CUBLASEW_SRC) $(LDFLAGS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]"

clean:
	rm -f $(TARGET) $(MOE_CUBIN)

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
