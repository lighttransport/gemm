# Makefile for CUDA LLM runner

CC = gcc
NVCC = nvcc
CFLAGS = -O3 -Wall -Wextra -D_GNU_SOURCE -I.. -mavx2 -mfma -march=native
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC = ../cuew.c
CUBLASEW_SRC = ../cublasew.c
RUNNER_SRC = cuda_llm_runner.c
TEST_SRC = test_cuda_llm.c
MOE_CUBIN = moe_gpu_kernels.cubin
FA2_CUBIN = fa2_kernels.cubin
MMQ_CUBIN = mmq_kernels.cubin
MMVQ_CUBIN = mmvq_kernels.cubin
DS4F_DENSE_CUBIN = ds4f_dense_kernels.cubin

# Vendored llama.cpp MMQ headers (mmq_vendor/) + the upstream tree for the rest.
# Override LLAMA_CUDA if llama.cpp lives elsewhere.
LLAMA_DIR ?= /home/syoyo/work/llama.cpp
LLAMA_CUDA = $(LLAMA_DIR)/ggml/src/ggml-cuda
MMQ_INCLUDES = -Immq_vendor -I$(LLAMA_CUDA) -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/ggml/src

TARGET = test_cuda_llm

.PHONY: all clean run help cubin

all: $(MOE_CUBIN) $(FA2_CUBIN) $(MMQ_CUBIN) $(MMVQ_CUBIN) $(TARGET)

cubin: $(MOE_CUBIN) $(FA2_CUBIN) $(MMQ_CUBIN) $(MMVQ_CUBIN) $(DS4F_DENSE_CUBIN)

$(DS4F_DENSE_CUBIN): ds4f_dense_kernels.cu
	$(NVCC) -cubin -arch=sm_120 -std=c++17 -O3 -o $@ $<

$(MOE_CUBIN): moe_gpu_kernels.cu
	$(NVCC) -cubin -arch=sm_120 -o $@ $<

$(FA2_CUBIN): fa2_kernels.cu
	$(NVCC) -cubin -arch=sm_120 -o $@ $<

# Vendored llama.cpp mul_mat_q (stream-K int8 MMQ). sm_120a for Turing-MMA m16n8k32.
$(MMQ_CUBIN): mmq_kernels.cu mmq_vendor/mmq.cuh
	$(NVCC) -cubin -arch=sm_120a -std=c++17 -O3 $(MMQ_INCLUDES) -o $@ $<

# Vendored llama.cpp mul_mat_vec_q (dp4a decode matvec, ncols_dst=1).
$(MMVQ_CUBIN): mmvq_kernels.cu mmq_vendor/mmvq.cuh
	$(NVCC) -cubin -arch=sm_120a -std=c++17 -O3 $(MMQ_INCLUDES) -o $@ $<

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(CUBLASEW_SRC) $(MOE_CUBIN) $(FA2_CUBIN) \
          cuda_llm_runner.h half_convert.h ../cuew.h ../cublasew.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(CUBLASEW_SRC) $(LDFLAGS)

test_half_convert: test_half_convert.c half_convert.h
	$(CC) -O2 -Wall -Wextra -Wpedantic -mf16c -o $@ $< -lm

test_gemma4_st: test_gemma4_st.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLASEW_SRC) $(MOE_CUBIN) $(FA2_CUBIN) \
          cuda_llm_runner.h half_convert.h ../cuew.h ../cublasew.h \
          ../../common/gemma4_audio_encoder.h ../../common/gemma4_audio_mel.h
	$(CC) $(CFLAGS) -o $@ test_gemma4_st.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLASEW_SRC) $(LDFLAGS)

test_qwen35_st: test_qwen35_st.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLASEW_SRC) $(MOE_CUBIN) $(FA2_CUBIN) \
          cuda_llm_runner.h half_convert.h ../cuew.h ../cublasew.h
	$(CC) $(CFLAGS) -o $@ test_qwen35_st.c $(RUNNER_SRC) $(CUEW_SRC) $(CUBLASEW_SRC) $(LDFLAGS)

# Parity check of the Qwen3.5-2B safetensors text tower against the HF fp32 dump
# (tmp/praxis_ref/text, from ref/praxis/gen_text_ref.py): all 25 hidden states + cond must pass
# compare.py (rel-L2 <= 0.005, cos >= 0.9999). Override QWEN35_DIR / QWEN35_REF as needed.
PRAXIS_ROOT ?= $(HOME)/praxis
QWEN35_DIR ?= $(PRAXIS_ROOT)/qwen3.5-2b
QWEN35_REF ?= tmp/praxis_ref/text
QWEN35_OUT ?= tmp/praxis_out/text_check
.PHONY: test_qwen35_st_check
test_qwen35_st_check: test_qwen35_st
	@test -f ../../$(QWEN35_REF)/cond.npy || { echo "missing reference $(QWEN35_REF): run python3 ref/praxis/gen_text_ref.py --out $(QWEN35_REF)"; exit 1; }
	cd ../.. && ./cuda/llm/test_qwen35_st $(QWEN35_DIR) -i $(QWEN35_REF)/ids.npy -o $(QWEN35_OUT) -p 32 --prof
	cd ../.. && python3 ref/praxis/compare.py --ref-dir $(QWEN35_REF) --ours-dir $(QWEN35_OUT) --skip-missing --tol 0.005 | tee $(QWEN35_OUT)/compare.log
	@test "`grep -c '^PASS' ../../$(QWEN35_OUT)/compare.log`" = 26 || { echo "test_qwen35_st_check: FAILED (expected 26 PASS: layer_00..24 + cond)"; exit 1; }
	@echo "test_qwen35_st_check: OK"

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]"

clean:
	rm -f $(TARGET) $(MOE_CUBIN) test_half_convert

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
