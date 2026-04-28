# Makefile for HIP/ROCm VLM vision encoder (RDNA4)
#
# No hipcc required - kernels compiled at runtime via HIPRTC.
# Only needs gcc + ROCm runtime libraries (libamdhip64.so, libhiprtc.so) on the system.

CC = gcc
CXX = g++
HIPCC = hipcc
ROCM_LLVM ?= /opt/rocm/llvm/bin
CFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

# Sources
ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_vision_encoder.c
LLM_RUNNER_SRC = ../llm/hip_llm_runner.c

# Targets
VISION = test_hip_vision
VLM = test_hip_vlm
BENCH_GEMM = bench_vlm_gemm
BENCH_ROCBLAS = bench_vlm_rocblas
BENCH_HIPBLASLT = bench_vlm_hipblaslt
GEN_DIR = generated
MM0_ASM_LDS_STRIDE ?= 144
MM0_ASM_HIP = $(GEN_DIR)/mm0_bf16_asm.hip
MM0_ASM_S = $(GEN_DIR)/mm0_bf16_asm.s
MM0_ASM_CO = $(GEN_DIR)/mm0_bf16_asm.co
MM0_ASM_GLOAD_HIP = $(GEN_DIR)/mm0_bf16_asm_gload.hip
MM0_ASM_GLOAD_S = $(GEN_DIR)/mm0_bf16_asm_gload.s
MM0_ASM_GLOAD_CO = $(GEN_DIR)/mm0_bf16_asm_gload.co
MM0_ASM_FROM_S_O = $(GEN_DIR)/mm0_bf16_asm_from_s.o
MM0_ASM_FROM_S_CO = $(GEN_DIR)/mm0_bf16_asm_from_s.co
MM0_ASM_HALFSPLIT_S = $(GEN_DIR)/mm0_bf16_asm_halfsplit.s
MM0_ASM_HALFSPLIT_O = $(GEN_DIR)/mm0_bf16_asm_halfsplit.o
MM0_ASM_HALFSPLIT_CO = $(GEN_DIR)/mm0_bf16_asm_halfsplit.co
MM0_ASM_STOREEARLY20_S = $(GEN_DIR)/mm0_bf16_asm_storeearly20.s
MM0_ASM_STOREEARLY20_O = $(GEN_DIR)/mm0_bf16_asm_storeearly20.o
MM0_ASM_STOREEARLY20_CO = $(GEN_DIR)/mm0_bf16_asm_storeearly20.co
MM0_ASM_BFIRST2_S = $(GEN_DIR)/mm0_bf16_asm_bfirst2.s
MM0_ASM_BFIRST2_O = $(GEN_DIR)/mm0_bf16_asm_bfirst2.o
MM0_ASM_BFIRST2_CO = $(GEN_DIR)/mm0_bf16_asm_bfirst2.co
MM0_DIRECTA_HIP = $(GEN_DIR)/mm0_bf16_directa.hip
MM0_DIRECTA_CO = $(GEN_DIR)/mm0_bf16_directa.co

.PHONY: all clean run help gen-mm0-asm bench-mm0-asm bench-mm0-asm-gload bench-mm0-asm-s bench-mm0-asm-halfsplit bench-mm0-asm-storeearly20 bench-mm0-asm-bfirst2 bench-mm0-directa

all: $(VISION) $(VLM) $(BENCH_GEMM)

$(VISION): test_hip_vision.c $(RUNNER_SRC) $(ROCEW_SRC) hip_vision_encoder.h ../rocew.h
	$(CC) $(CFLAGS) -o $@ test_hip_vision.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VLM): test_hip_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(ROCEW_SRC) hip_vision_encoder.h ../llm/hip_llm_runner.h ../rocew.h
	$(CC) $(CFLAGS) -o $@ test_hip_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(BENCH_GEMM): bench_vlm_gemm.c $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h
	$(CC) $(CFLAGS) -o $@ bench_vlm_gemm.c $(ROCEW_SRC) $(LDFLAGS)

$(BENCH_ROCBLAS): bench_vlm_rocblas.cpp
	$(HIPCC) -O3 --offload-arch=gfx1201 -o $@ $< -lrocblas

$(BENCH_HIPBLASLT): bench_vlm_hipblaslt.cpp
	$(HIPCC) -O3 --offload-arch=gfx1201 -o $@ $< -lhipblaslt

gen-mm0-asm $(MM0_ASM_HIP): gen_mm0_bf16_asm.py
	python3 gen_mm0_bf16_asm.py --out-dir $(GEN_DIR) --lds-stride $(MM0_ASM_LDS_STRIDE)

$(MM0_ASM_GLOAD_HIP): gen_mm0_bf16_asm.py
	python3 gen_mm0_bf16_asm.py --out-dir $(GEN_DIR) --basename mm0_bf16_asm_gload --global-layout coalesced

$(MM0_ASM_S): $(MM0_ASM_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --cuda-device-only -S -o $@ $<

$(MM0_ASM_CO): $(MM0_ASM_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --genco -o $@ $<

$(MM0_ASM_GLOAD_S): $(MM0_ASM_GLOAD_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --cuda-device-only -S -o $@ $<

$(MM0_ASM_GLOAD_CO): $(MM0_ASM_GLOAD_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --genco -o $@ $<

$(MM0_ASM_FROM_S_O): $(MM0_ASM_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_FROM_S_CO): $(MM0_ASM_FROM_S_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_HALFSPLIT_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant halfsplit $< $@

$(MM0_ASM_HALFSPLIT_O): $(MM0_ASM_HALFSPLIT_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_HALFSPLIT_CO): $(MM0_ASM_HALFSPLIT_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_STOREEARLY20_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant storeearly20 $< $@

$(MM0_ASM_STOREEARLY20_O): $(MM0_ASM_STOREEARLY20_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_STOREEARLY20_CO): $(MM0_ASM_STOREEARLY20_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_ASM_BFIRST2_S): $(MM0_ASM_S) patch_mm0_asm_schedule.py
	python3 patch_mm0_asm_schedule.py --variant bfirst2 $< $@

$(MM0_ASM_BFIRST2_O): $(MM0_ASM_BFIRST2_S)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c -o $@ $<

$(MM0_ASM_BFIRST2_CO): $(MM0_ASM_BFIRST2_O)
	$(ROCM_LLVM)/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nostdlib -Wl,-shared -o $@ $<

$(MM0_DIRECTA_HIP): gen_mm0_bf16_directa.py
	python3 gen_mm0_bf16_directa.py --out-dir $(GEN_DIR)

$(MM0_DIRECTA_CO): $(MM0_DIRECTA_HIP)
	$(HIPCC) -O3 -ffast-math --offload-arch=gfx1201 --genco -o $@ $<

bench-mm0-asm: $(BENCH_GEMM) $(MM0_ASM_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-gload: $(BENCH_GEMM) $(MM0_ASM_GLOAD_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_GLOAD_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-s: $(BENCH_GEMM) $(MM0_ASM_FROM_S_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_FROM_S_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-halfsplit: $(BENCH_GEMM) $(MM0_ASM_HALFSPLIT_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_HALFSPLIT_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-storeearly20: $(BENCH_GEMM) $(MM0_ASM_STOREEARLY20_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_STOREEARLY20_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-asm-bfirst2: $(BENCH_GEMM) $(MM0_ASM_BFIRST2_CO)
	VLM_GEMM_ASM_CO=$(MM0_ASM_BFIRST2_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

bench-mm0-directa: $(BENCH_GEMM) $(MM0_DIRECTA_CO)
	VLM_GEMM_ASM_CO=$(MM0_DIRECTA_CO) ./$(BENCH_GEMM) --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check

# dump_clip_embd requires a llama.cpp build with libmtmd/libllama.
# Usage: make dump_clip_embd LLAMA_DIR=/home/syoyo/work/llama.cpp
LLAMA_DIR ?= /home/syoyo/work/llama.cpp
dump_clip_embd: dump_clip_embd.cpp
	$(CXX) -O2 -std=c++17 \
		-I../../common \
		-I$(LLAMA_DIR)/vendor/stb \
		-I$(LLAMA_DIR)/tools/mtmd \
		-I$(LLAMA_DIR)/ggml/include \
		-I$(LLAMA_DIR)/include \
		-L$(LLAMA_DIR)/build/bin \
		-Wl,-rpath,$(LLAMA_DIR)/build/bin \
		-o $@ $< -lmtmd -lllama -lggml -lggml-base -lm

run: $(VISION)
	@echo "Usage: ./$(VISION) <mmproj.gguf> [--image path] [--image-size N] [--ref embd.bin] [--warmup N] [--iters N] [--f32|--f16|--bf16]"

clean:
	rm -f $(VISION) $(VLM) $(BENCH_GEMM) $(BENCH_ROCBLAS) $(BENCH_HIPBLASLT) dump_clip_embd $(MM0_ASM_S) $(MM0_ASM_CO) $(MM0_ASM_GLOAD_S) $(MM0_ASM_GLOAD_CO) $(MM0_ASM_FROM_S_O) $(MM0_ASM_FROM_S_CO) $(MM0_ASM_HALFSPLIT_S) $(MM0_ASM_HALFSPLIT_O) $(MM0_ASM_HALFSPLIT_CO) $(MM0_ASM_STOREEARLY20_S) $(MM0_ASM_STOREEARLY20_O) $(MM0_ASM_STOREEARLY20_CO) $(MM0_ASM_BFIRST2_S) $(MM0_ASM_BFIRST2_O) $(MM0_ASM_BFIRST2_CO) $(MM0_DIRECTA_CO)

help:
	@echo "HIP VLM Vision Encoder (HIPRTC-based, no hipcc needed, RDNA4)"
	@echo "============================================================="
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_hip_vision and test_hip_vlm"
	@echo "  make bench-mm0-asm - Generate/build external mm0 BF16 kernel and benchmark it"
	@echo "    override LDS padding with MM0_ASM_LDS_STRIDE=N, e.g. N=128 for the old baseline"
	@echo "  make bench-mm0-asm-gload - Generate/build coalesced-global-load mm0 BF16 kernel and benchmark it"
	@echo "  make bench-mm0-asm-s - Assemble/link generated AMDGPU .s and benchmark it"
	@echo "  make bench-mm0-asm-halfsplit - Assemble/link patched half-split schedule and benchmark it"
	@echo "  make bench-mm0-asm-storeearly20 - Assemble/link patched early-store schedule and benchmark it"
	@echo "  make bench-mm0-asm-bfirst2 - Assemble/link second-half B-first LDS load schedule and benchmark it"
	@echo "  make bench-mm0-directa - Generate direct-A/LDS-B external kernel and benchmark it"
	@echo "  make dump_clip_embd LLAMA_DIR=/home/syoyo/work/llama.cpp"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_hip_vision <mmproj.gguf> [--image path] [--image-size N] [--ref embd.bin] [--f32|--f16|--bf16]"
	@echo "  ./test_hip_vlm <model.gguf> <mmproj.gguf> <image.jpg> [-n max_tokens] [--vision-f32|--vision-f16|--vision-bf16]"
	@echo "  ./bench_vlm_gemm [--dtype f16|bf16] [--shape all|qkv|attn_out|ffn_up|ffn_down|mm0|mm2] [--mode cta|cta64|direct|mm0spec|mm0vec|mm0pipe|mm0asm|mm0pipegl|mm0pipemid|mm0pipe8|mm0pipek64|mm0pipe4n64] [--iters N] [--check] [--no-bias]"
	@echo "    mm0asm loads VLM_GEMM_ASM_CO and expects kernel gemm_mm0_bf16_asm"
	@echo "  make bench_vlm_rocblas && ./bench_vlm_rocblas [--dtype f16|bf16] [--shape all] [--iters N]"
	@echo "  make bench_vlm_hipblaslt && ./bench_vlm_hipblaslt [--dtype f16|bf16] [--shape all] [--epilogue none|bias|gelu_bias] [--iters N] [--algos N] [--workspace-mb N] [--print-algos|--list-only] [--algo-index N]"
	@echo "  ./dump_hipblaslt_kernel_asm.sh <code-object-or-kernel-name> [/tmp/out-prefix]"
	@echo ""
	@echo "Requirements:"
	@echo "  - libamdhip64.so (HIP runtime)"
	@echo "  - libhiprtc.so (HIPRTC runtime compiler)"
	@echo "  - AMD GPU with RDNA4 (gfx1200/gfx1201)"
