# Makefile for HIP LLM runner (RDNA4)
#
# No hipcc required - kernels compiled at runtime via HIPRTC.
# Only needs gcc + ROCm runtime libraries (libamdhip64.so, libhiprtc.so) on the system.

CC = gcc
HIPCC ?= hipcc
ROCM_PATH ?= /opt/rocm
ROCM_LIB ?= $(dir $(firstword $(wildcard $(ROCM_PATH)/lib/libhipblaslt.so $(ROCM_PATH)/core-*/lib/libhipblaslt.so)))
CFLAGS = -O3 -Wall -Wextra -D_GNU_SOURCE -I.. -march=native -fopenmp
LDFLAGS = -ldl -lm -lpthread -fopenmp

# Sources
ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_llm_runner.c
TEST_SRC = test_hip_llm.c

# hipBLASLt bridge (optional). Build with `make HIPBLASLT=1` to enable WMMA-class
# prefill GEMMs via hipBLASLt. Without it, the runner falls back to per-token GEMV.
HIPBLASLT ?= 1

# Target
TARGET = test_hip_llm

.PHONY: all clean run api cpu-expert-bench glm5next-contract glm5next-ref glm5next-cpu-kda glm5next-cpu-full glm5next-cpu-runtime glm5next-hip-kda help

all: $(TARGET)

ifeq ($(HIPBLASLT),1)
CFLAGS += -DLLM_HIPBLASLT_ENABLED
LDFLAGS += -L$(ROCM_LIB) -Wl,-rpath,$(ROCM_LIB) -lhipblaslt -lstdc++
BLASLT_OBJ = mm_blaslt_bridge.o
else
BLASLT_OBJ =
endif

mm_blaslt_bridge.o: mm_blaslt_bridge.cpp mm_blaslt_bridge.h
	$(HIPCC) -O3 -fPIE --offload-arch=gfx1201 -c -o $@ $<

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) hip_llm_runner.h ../rocew.h $(BLASLT_OBJ)
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(BLASLT_OBJ) $(LDFLAGS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]"

api: $(TARGET)
	@echo "Usage: python3 codex_server.py <model.gguf> [--port 8080] [--context 4096]"

cpu-expert-bench: bench_cpu_qwen_expert

glm5next-contract: test_glm5next_contract

glm5next-ref: test_glm5next_ref

glm5next-cpu-kda: test_glm5next_cpu_kda

glm5next-cpu-full: test_glm5next_cpu_full

glm5next-cpu-runtime: test_glm5next_cpu_runtime

test_glm5next_cpu_runtime: test_glm5next_cpu_runtime.c ../../common/glm5next_cpu_runtime.h
	$(CC) -O2 -Wall -Wextra -Wpedantic -I.. -o $@ $< -lm

test_glm5next_cpu_full: test_glm5next_cpu_full.c ../../common/glm5next_cpu_kda.h
	$(CC) -O2 -Wall -Wextra -Wpedantic -I.. -o $@ $< -lm

glm5next-hip-kda: $(TARGET)
	./$(TARGET) --verify-glm5next-kda 128

test_glm5next_cpu_kda: test_glm5next_cpu_kda.c ../../common/glm5next_cpu_kda.h ../../common/glm5next_ref.h
	$(CC) -O2 -Wall -Wextra -Wpedantic -I.. -o $@ $< -lm

test_glm5next_ref: test_glm5next_ref.c ../../common/glm5next_ref.h ../../common/glm5next.h ../../common/glm53f_ref.h
	$(CC) -O2 -Wall -Wextra -Wpedantic -I.. -o $@ $< -lm

test_glm5next_contract: test_glm5next_contract.c ../../common/glm5next.h ../../common/gguf_loader.h
	$(CC) -O2 -Wall -Wextra -Wpedantic -I.. -o $@ $< -lm

bench_cpu_qwen_expert: bench_cpu_qwen_expert.c
	$(CC) -O3 -Wall -Wextra -march=native -fopenmp -o $@ $< -ldl -lm

clean:
	rm -f $(TARGET) bench_cpu_qwen_expert test_glm5next_contract test_glm5next_ref test_glm5next_cpu_kda test_glm5next_cpu_full test_glm5next_cpu_runtime mm_blaslt_bridge.o

help:
	@echo "HIP LLM Runner (HIPRTC-based, no hipcc needed)"
	@echo "================================================"
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_hip_llm"
	@echo "  make clean        - Remove built files"
	@echo "  make glm5next-contract MODEL=/path/to/GLM5Next.gguf"
	@echo ""
	@echo "Run:"
	@echo "  ./test_hip_llm <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]"
	@echo ""
	@echo "Example:"
	@echo "  ./test_hip_llm Qwen3-Embedding-0.6B-f16.gguf -t \"Hello world\" -n 4"
	@echo "  python3 codex_server.py model.gguf --context 4096 --port 8080"
	@echo "  GLM5NEXT_HIP_DSA=1 GLM5NEXT_HIP_KDA=1 GLM5NEXT_HIP_MOE=1 ./test_hip_llm GLM5Next.gguf --gpu-only-bench --prefill-len 1 --decode 1"
	@echo "  GLM5NEXT_HIP_MOE_LAYER=N limits streamed MoE bring-up to one layer"
	@echo "  GLM5NEXT_HIP_MOE_ROUTER=0 keeps the MoE router matvec on CPU (HIP is default)"
	@echo "  GLM5NEXT_HIP_INDEXER=1 moves DSA indexer projections to HIP"
	@echo "  GLM5NEXT_HIP_INDEXER_CACHE=1 retains indexer weights on HIP (uses more VRAM)"
	@echo "  GLM5NEXT_HIP_MHC=1 moves mHC projection matvecs to HIP"
	@echo "  GLM5NEXT_HIP_OUTPUT=1 moves the final vocabulary projection to HIP"
	@echo "  GLM5NEXT_HIP_DSA_PROJ=1 moves DSA q_a/q_b/kv_a projections to HIP"
	@echo "  GLM5NEXT_HIP_DSA_PROJ_LAYER=N limits DSA projection offload to one layer"
	@echo "  GLM5NEXT_CPU_OMP=1 enables OpenMP rows for large CPU projections"
	@echo ""
	@echo "Requirements:"
	@echo "  - libamdhip64.so (HIP runtime)"
	@echo "  - libhiprtc.so (HIPRTC runtime compiler)"
	@echo "  - AMD GPU with RDNA4 (gfx1200/gfx1201)"
