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
LDFLAGS += -lstdc++

# Sources
ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_llm_runner.c
TEST_SRC = test_hip_llm.c

# hipBLASLt bridge (optional). Build with `make HIPBLASLT=1` to enable WMMA-class
# prefill GEMMs via hipBLASLt. Without it, the runner falls back to per-token GEMV.
HIPBLASLT ?= 1
CUDA_MOE ?= 0
HIPBLASLT_HEADERS := $(firstword $(wildcard \
  $(ROCM_PATH)/include/hipblaslt/hipblaslt.h \
  $(ROCM_PATH)/core-*/include/hipblaslt/hipblaslt.h))

# Target
TARGET = test_hip_llm
QWEN38_MODEL ?= /mnt/nvme01/models/q38nf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
QSA_MOE_CACHE_MB ?= 128

.PHONY: all clean run api gsq gsq-target iq2-bw profile-test bench-256k target-gate target-gate-fast target-gate-batch target-gate-4k hipblaslt-status qsa-256k qsa-256k-device qsa-256k-device-warp approx-coherence fp8-test cpu-expert-bench glm5next-contract glm5next-ref glm5next-cpu-kda glm5next-cpu-full glm5next-cpu-runtime glm5next-nextn glm5next-hip-kda half2-test help

all: $(TARGET)

test_fattn_vec_half2: test_fattn_vec_half2.cu
	$(HIPCC) -O2 --offload-arch=gfx1201 $< -o $@

half2-test: test_fattn_vec_half2
	env LD_LIBRARY_PATH=$(ROCM_PATH)/core-10.0/lib ./test_fattn_vec_half2

.PHONY: kv-q8q8-test
kv-q8q8-test:
	python3 test_kv_q8q8.py --hipcc $(HIPCC) --out tmp/kv-q8q8
	env LD_LIBRARY_PATH=$(ROCM_LIB) ./tmp/kv-q8q8/test

ifeq ($(HIPBLASLT),1)
CFLAGS += -DLLM_HIPBLASLT_ENABLED
LDFLAGS += -L$(ROCM_LIB) -Wl,-rpath,$(ROCM_LIB) -lhipblaslt -lhipblas -lstdc++
BLASLT_OBJ = mm_blaslt_bridge.o
else
BLASLT_OBJ =
endif

ifeq ($(HIPBLASLT),1)
ifeq ($(HIPBLASLT_HEADERS),)
$(warning hipBLASLt headers not found; building the bridge's scalar fallback)
CFLAGS += -DMM_BLASLT_DISABLE
BLASLT_BRIDGE_CFLAGS += -DMM_BLASLT_DISABLE
endif
endif

ifeq ($(CUDA_MOE),1)
CFLAGS += -DCUDA_MOE_BRIDGE
LDFLAGS += -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lstdc++
CUDA_MOE_OBJ = cuda_moe_bridge.o
else
CUDA_MOE_OBJ =
endif

cuda_moe_bridge.o: cuda_moe_bridge.cu cuda_moe_bridge.h glm5next_iq1s_grid.h glm5next_iq2_iq3_grid.h
	nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -c -o $@ $<

mm_blaslt_bridge.o: mm_blaslt_bridge.cpp mm_blaslt_bridge.h
	$(HIPCC) -O3 -fPIE $(BLASLT_BRIDGE_CFLAGS) --offload-arch=gfx1201 -c -o $@ $<

reference_sampler.o: reference_sampler.cpp reference_sampler.h
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -c -o $@ $<

qwen35_attention_q8.inc: qwen35_attention_q8.hip
	python3 -c 'import json,pathlib; print("static const char qwen35_attention_q8_source[] = " + json.dumps(pathlib.Path("$<").read_text()) + ";")' > $@

qwen35_matvec_q2k.inc: qwen35_matvec_q2k.hip
	python3 -c 'import json,pathlib; print("static const char qwen35_matvec_q2k_source[] = " + json.dumps(pathlib.Path("$<").read_text()) + ";")' > $@

qwen35_matvec_iq.inc: qwen35_matvec_iq.hip embed_hip_source.py ../../common/iq2s_grid.inc ../../common/iq3xxs_grid.inc ../../common/iq3s_grid.inc ../../common/ksigns_iq2xs.inc
	python3 embed_hip_source.py $< qwen35_matvec_iq_source > $@

# Other test executables include the runner translation unit directly.
HIPRTC_INCLUDES = qwen35_attention_q8.inc qwen35_matvec_q2k.inc qwen35_matvec_iq.inc qwen35_reference_math.h
$(TARGET) tmp/test_hip_qwen4_moe_stage tmp/test_hip_qwen4_moe_stage_large tmp/test_hip_qwen4_attention test_glm5next_nextn: $(HIPRTC_INCLUDES)

LLAMA_REFERENCE ?= ../../tmp/qwen38/reference-build
.PHONY: reference-attention-test
reference-attention-test:
	mkdir -p tmp
	env TMPDIR=$(abspath tmp) $(HIPCC) -O3 -std=c++17 --offload-arch=gfx1201 -DGGML_USE_HIP \
	    -I$(LLAMA_REFERENCE)/source/ggml/include -I$(LLAMA_REFERENCE)/source/ggml/src \
	    -I$(LLAMA_REFERENCE)/source/ggml/src/ggml-cuda test_reference_attention.cpp \
	    -L$(LLAMA_REFERENCE)/build/bin -Wl,-rpath,$(abspath $(LLAMA_REFERENCE)/build/bin) \
	    -lggml-base -o tmp/test_reference_attention
	env LD_LIBRARY_PATH=$(ROCM_LIB) ./tmp/test_reference_attention

.PHONY: reference-sampler-test
reference-sampler-test: reference_sampler.o
	mkdir -p tmp
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off \
	    -I$(LLAMA_REFERENCE)/source/include -I$(LLAMA_REFERENCE)/source/ggml/include \
	    test_reference_sampler.cpp reference_sampler.o \
	    -L$(LLAMA_REFERENCE)/build/bin -Wl,-rpath,$(abspath $(LLAMA_REFERENCE)/build/bin) \
	    -Wl,-rpath-link,$(ROCM_LIB) -Wl,-rpath,$(ROCM_LIB) \
	    -lllama -o tmp/test_reference_sampler
	env LD_LIBRARY_PATH=$(ROCM_LIB) ./tmp/test_reference_sampler

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) hip_llm_runner.h generation_trace.h qwen35_reference_math.h qwen4_moe_stage.h qwen4_ple_split_test.h qwen4_phase_scratch.h qwen4_hc_native_kernels.h qwen4_hc_native_test.h qwen4_ssm_native_kernels.h qwen4_ssm_native_test.h qwen4_moe_native_kernels.h qwen4_moe_native_test.h qwen4_nextn_forward.h qwen4_nextn_ref.h qwen4_qsa_hip.h qwen4_qsa_test.h qwen4_verify_window.h ../../common/qwen4_qsa.h ../hip_runner_common.h ../rocew.h reference_sampler.o $(BLASLT_OBJ) $(CUDA_MOE_OBJ)
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) reference_sampler.o $(BLASLT_OBJ) $(CUDA_MOE_OBJ) $(LDFLAGS)

.PHONY: moe-stage-test moe-stage-gpu-test moe-stage-quality
moe-stage-test:
	mkdir -p tmp
	$(CC) -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror test_qwen4_moe_stage.c -o tmp/test_qwen4_moe_stage
	./tmp/test_qwen4_moe_stage
	python3 test_copy_lifecycle.py
	python3 test_qwen38_target_gate.py
	bash test_qwen38_profiles.sh

tmp/test_hip_qwen4_moe_stage: test_hip_qwen4_moe_stage.c $(RUNNER_SRC) qwen4_moe_stage.h qwen4_ple_split_test.h qwen4_phase_scratch.h qwen4_hc_native_kernels.h qwen4_hc_native_test.h qwen4_ssm_native_kernels.h qwen4_ssm_native_test.h qwen4_moe_native_kernels.h qwen4_moe_native_test.h hip_llm_runner.h $(ROCEW_SRC) $(BLASLT_OBJ) $(CUDA_MOE_OBJ)
	mkdir -p tmp
	$(CC) $(CFLAGS) -o $@ test_hip_qwen4_moe_stage.c $(ROCEW_SRC) $(BLASLT_OBJ) $(CUDA_MOE_OBJ) $(LDFLAGS)

tmp/test_hip_qwen4_moe_stage_large: test_hip_qwen4_moe_stage.c $(RUNNER_SRC) qwen4_moe_stage.h qwen4_ple_split_test.h qwen4_phase_scratch.h qwen4_hc_native_kernels.h qwen4_hc_native_test.h qwen4_ssm_native_kernels.h qwen4_ssm_native_test.h qwen4_moe_native_kernels.h qwen4_moe_native_test.h hip_llm_runner.h $(ROCEW_SRC) $(BLASLT_OBJ) $(CUDA_MOE_OBJ)
	mkdir -p tmp
	$(CC) $(CFLAGS) -DQWEN4_STAGE_TEST_DIM=2560 -o $@ test_hip_qwen4_moe_stage.c $(ROCEW_SRC) $(BLASLT_OBJ) $(CUDA_MOE_OBJ) $(LDFLAGS)

tmp/test_hip_qwen4_attention: test_hip_qwen4_attention.c $(RUNNER_SRC) qwen4_moe_stage.h qwen4_ple_split_test.h qwen4_phase_scratch.h qwen4_hc_native_kernels.h qwen4_hc_native_test.h qwen4_ssm_native_kernels.h qwen4_ssm_native_test.h qwen4_moe_native_kernels.h qwen4_moe_native_test.h hip_llm_runner.h $(ROCEW_SRC) $(BLASLT_OBJ) $(CUDA_MOE_OBJ)
	mkdir -p tmp
	$(CC) $(CFLAGS) -o $@ test_hip_qwen4_attention.c $(ROCEW_SRC) $(BLASLT_OBJ) $(CUDA_MOE_OBJ) $(LDFLAGS)

.PHONY: qwen4-attention-gpu-test
qwen4-attention-gpu-test: tmp/test_hip_qwen4_attention
	timeout --foreground 180s ./tmp/test_hip_qwen4_attention
	@if timeout --foreground 180s ./tmp/test_hip_qwen4_attention --unsafe; then \
	    echo "ERROR: attention barrier mutation was not detected"; exit 1; \
	else code=$$?; test $$code -eq 1; fi

moe-stage-gpu-test: tmp/test_hip_qwen4_moe_stage
	timeout --foreground 180s ./tmp/test_hip_qwen4_moe_stage

moe-stage-quality: $(TARGET)
	./test_qwen38_staging_quality.sh

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]"

api: $(TARGET)
	@echo "Usage: python3 codex_server.py <model.gguf> [--port 8080] [--context 4096]"

gsq: $(TARGET)
	./run_qwen38_gsq_rocm.sh

gsq-target: $(TARGET)
	./bench_qwen38_gsq_decode.sh

iq2-bw: $(TARGET)
	./bench_iq2_matvec_bw.sh

profile-test:
	./test_qwen38_profiles.sh

approx-coherence: $(TARGET)
	./test_qwen38_approx_coherence.sh

fp8-test:
	$(CC) -O2 -Wall -Wextra -Wpedantic -I.. -o tmp/test_fp8_e4m3 ../test_fp8_e4m3.c ../rocew.c -ldl -lm
	./tmp/test_fp8_e4m3

bench-256k: $(TARGET)
	./bench_qwen38_256k.sh

# In-process N-repeat repeatability + throughput gate.  A profile must produce
# an identical first token and sequence hash on every repeat before promotion.
target-gate: $(TARGET)
	./bench_qwen38_target.sh

target-gate-fast: $(TARGET)
	QWEN38_TARGET_PROFILE=fast ./bench_qwen38_target.sh

target-gate-batch: $(TARGET)
	QWEN38_TARGET_PROFILE=batch ./bench_qwen38_target.sh

# Deterministic single-dispatch 4K batched profile (BMAX=4096).
target-gate-4k: $(TARGET)
	QWEN38_TARGET_PROFILE=batch4k ./bench_qwen38_target.sh

hipblaslt-status:
	@printf 'hipBLASLt headers: %s\n' '$(if $(HIPBLASLT_HEADERS),available,missing)'
	@printf 'hipBLASLt runtime: %s\n' '$(if $(ROCM_LIB),$(ROCM_LIB)libhipblaslt.so,not-found)'
	@if [ -z "$(HIPBLASLT_HEADERS)" ]; then echo 'mode: scalar bridge fallback (no acceleration)'; else echo 'mode: accelerated bridge available'; fi

# Full-window QSA allocation/numerics gate. This does not generate a 256K
# response; it exercises the index cache and selected-attention verifier.
qsa-256k: $(TARGET)
	LLM_QWEN4_QSA_VERIFY_FULL_WINDOW=1 ./$(TARGET) --verify-qwen4-qsa -s 262144 --moe-cache-mb $(QSA_MOE_CACHE_MB) "$(QWEN38_MODEL)"

qsa-256k-device: $(TARGET)
	LLM_QWEN4_QSA_DEVICE_SELECT=1 LLM_QWEN4_QSA_VERIFY_FULL_WINDOW=1 ./$(TARGET) --verify-qwen4-qsa -s 262144 --moe-cache-mb $(QSA_MOE_CACHE_MB) "$(QWEN38_MODEL)"

qsa-256k-device-warp: $(TARGET)
	LLM_QWEN4_QSA_DEVICE_SELECT=1 LLM_QWEN4_QSA_WARP_ATTN=1 LLM_QWEN4_QSA_VERIFY_FULL_WINDOW=1 ./$(TARGET) --verify-qwen4-qsa -s 262144 --moe-cache-mb $(QSA_MOE_CACHE_MB) "$(QWEN38_MODEL)"

cpu-expert-bench: bench_cpu_qwen_expert

glm5next-contract: test_glm5next_contract

glm5next-ref: test_glm5next_ref

glm5next-cpu-kda: test_glm5next_cpu_kda

glm5next-cpu-full: test_glm5next_cpu_full

glm5next-cpu-runtime: test_glm5next_cpu_runtime

glm5next-nextn: test_glm5next_nextn

test_glm5next_nextn: test_glm5next_nextn.c $(RUNNER_SRC) $(ROCEW_SRC) hip_llm_runner.h ../rocew.h $(BLASLT_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(RUNNER_SRC) $(ROCEW_SRC) $(BLASLT_OBJ) $(LDFLAGS)

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
	rm -f $(TARGET) test_fattn_vec_half2 bench_cpu_qwen_expert test_glm5next_contract test_glm5next_ref test_glm5next_cpu_kda test_glm5next_cpu_full test_glm5next_cpu_runtime test_glm5next_nextn mm_blaslt_bridge.o reference_sampler.o cuda_moe_bridge.o qwen35_attention_q8.inc qwen35_matvec_q2k.inc qwen35_matvec_iq.inc

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
	@echo "  GLM5NEXT_HIP_NEXTN=1 enables the GLM5Next NextN fusion projection on HIP"
	@echo "  GLM5NEXT_HIP_DSA_CACHE=1 opts into persistent DSA weights (streaming is default)"
	@echo "  GLM5NEXT_HIP_MOE_LAYER=N limits streamed MoE bring-up to one layer"
	@echo "  GLM5NEXT_HIP_KDA_LAYER=N limits KDA bring-up to one layer"
	@echo "  GLM5NEXT_HIP_KDA_CACHE=0 disables selected-layer persistent KDA weights"
	@echo "  GLM5NEXT_HIP_MOE_ROUTER=0 keeps the MoE router matvec on CPU (HIP is default)"
	@echo "  GLM5NEXT_HIP_INDEXER=1 moves DSA indexer projections to HIP"
	@echo "  GLM5NEXT_HIP_INDEXER_CACHE=1 retains indexer weights on HIP (uses more VRAM)"
	@echo "  GLM5NEXT_HIP_MHC=1 moves mHC projection matvecs to HIP"
	@echo "  GLM5NEXT_HIP_MHC_FINISH=1 experimental HIP mHC finish (CPU finish is default)"
	@echo "  GLM5NEXT_HIP_OUTPUT=1 moves the final vocabulary projection to HIP"
	@echo "  GLM5NEXT_HIP_DSA_PROJ=1 moves DSA q_a/q_b/kv_a projections to HIP"
	@echo "  GLM5NEXT_HIP_DSA_PROJ_LAYER=N limits DSA projection offload to one layer"
	@echo "  GLM5NEXT_CPU_OMP=1 enables OpenMP rows for large CPU projections"
	@echo ""
	@echo "Requirements:"
	@echo "  - libamdhip64.so (HIP runtime)"
	@echo "  - libhiprtc.so (HIPRTC runtime compiler)"
	@echo "  - AMD GPU with RDNA4 (gfx1200/gfx1201)"
