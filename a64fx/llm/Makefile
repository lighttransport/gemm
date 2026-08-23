# a64fx/llm Makefile
#
# Builds an end-to-end VLM→LLM runner that pairs the existing a64fx vision
# encoder (a64fx/vlm) with common/transformer.h.
#
# Usage:
#   make CC=fcc                     # Fujitsu fcc, sequential
#   make CC=fcc OPENMP=1            # Fujitsu fcc + OpenMP (recommended)
#   make CC=mpifcc OPENMP=1         # Fujitsu MPI wrapper + OpenMP
#   make CC=gcc OPENMP=1            # gcc + OpenMP
#
# Outputs:
#   build/llm_runner   - end-to-end runner (text or text+image)

CC      ?= fcc
HOSTCC  ?= cc
OPENMP  ?=
DEBUG   ?=
EXTRA_CFLAGS ?=

LLM_DIR     := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT        := $(LLM_DIR)..
COMMON      := $(ROOT)/../common
VLM_DIR     := $(ROOT)/vlm
VLM_INC     := $(VLM_DIR)/include
VLM_KERNELS := $(VLM_DIR)/kernels
VLM_SRC     := $(VLM_DIR)/src
VLM_BUILD   := $(VLM_DIR)/build
BUILD       := $(LLM_DIR)build
Q8V2_KERNEL := $(ROOT)/gemma4-kernels/kernel_q8v2_3x4.S
Q8B6_KERNEL := $(ROOT)/int8-cmg/micro_kernel_6x4_no_sector.S
BF16_PODD_KERNEL := $(ROOT)/gemma4-kernels/sgemm_bf16_2x12.S

# ── Compiler-specific flags ────────────────────────────────────────────
ifeq ($(CC),fcc)
    BASE_CFLAGS  := -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
    BASE_CFLAGS  += -Wno-unused-function -Wno-unused-variable
    # -g -rdynamic: symbol names for the SIGSEGV backtrace handler (on by default). These do NOT
    # change codegen at -O3 -- they only keep the symbol table, so a crash reports
    # "ds4f_pf_qnr_worker -> ds4f_rope_apply" instead of a bare hex address. On a multi-node run with
    # no core dump and no debugger, that is the difference between a 10-minute fix and a lost day.
    BASE_CFLAGS  += -g -rdynamic
    OMP_FLAG     := -fopenmp
    ASMC         := fcc
    HWBLIB       := -lhwb
else ifeq ($(CC),mpifcc)
    BASE_CFLAGS  := -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
    BASE_CFLAGS  += -Wno-unused-function -Wno-unused-variable
    OMP_FLAG     := -fopenmp
    ASMC         := fcc
    HWBLIB       := -lhwb
else ifeq ($(CC),fccpx)
    BASE_CFLAGS  := -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
    BASE_CFLAGS  += -Wno-unused-function -Wno-unused-variable
    OMP_FLAG     := -fopenmp
    ASMC         := fccpx
    HWBLIB       := -lhwb
else ifeq ($(CC),gcc)
    BASE_CFLAGS  := -O3 -march=armv8.2-a+sve -ffp-contract=fast
    BASE_CFLAGS  += -Wall -Wno-unused-function -Wno-unused-variable
    OMP_FLAG     := -fopenmp
    ASMC         := gcc
    HWBLIB       := -lhwb
else ifeq ($(CC),clang)
    BASE_CFLAGS  := -O3 -march=armv8.2-a+sve -ffp-contract=fast
    BASE_CFLAGS  += -Wall -Wno-unused-function -Wno-unused-variable
    OMP_FLAG     := -fopenmp
    ASMC         := clang
    HWBLIB       := -lhwb
else
    BASE_CFLAGS  := -O2
    OMP_FLAG     :=
    ASMC         := $(CC)
    HWBLIB       :=
endif

CFLAGS  := $(BASE_CFLAGS) -std=c11 -D_GNU_SOURCE \
           -I$(VLM_INC) -I$(VLM_KERNELS) -I$(VLM_SRC) -I$(COMMON)
LDFLAGS := -lm -lpthread $(HWBLIB)

# Qwen's exact Q8 decode path benefits from Fujitsu's fast scalar/SVE
# scheduling while retaining the same generated greedy output in the 128-token
# comparison. Keep this target-specific so unrelated runners retain their
# existing floating-point flags.
ifeq ($(CC),fcc)
QWEN_CFLAGS := -Kfast
else
QWEN_CFLAGS :=
endif

ifdef OPENMP
    CFLAGS  += $(OMP_FLAG) -DUSE_OPENMP=1
    LDFLAGS += $(OMP_FLAG)
endif

ifdef FAPP
    CFLAGS  += -DGLM5_USE_FAPP=1
endif

ifdef DEBUG
    CFLAGS := $(subst -O3,-O0,$(CFLAGS)) -g
endif

CFLAGS += $(EXTRA_CFLAGS)

# ── VLM object files (built by a64fx/vlm's Makefile) ──────────────────
# We reuse the already-built artifacts so we don't double-compile the
# vision encoder. Run `make CC=fcc OPENMP=1` in a64fx/vlm/ first.
VLM_OBJS := \
    $(VLM_BUILD)/src_vit_a64fx.o \
    $(VLM_BUILD)/src_vlm_parallel.o \
    $(VLM_BUILD)/src_tensor_dump.o \
    $(VLM_BUILD)/src_cmg_pool.o \
    $(VLM_BUILD)/kernels_fused_gemm.o \
    $(VLM_BUILD)/kernels_pack_matrices.o \
    $(VLM_BUILD)/kernels_norm_sve.o \
    $(VLM_BUILD)/kernels_bf16_gemm.o \
    $(VLM_BUILD)/kernels_fp16_gemm.o \
    $(VLM_BUILD)/kernels_fp32_gemm.o \
    $(VLM_BUILD)/kernels_micro_kernel_fp32_8x3.o \
    $(VLM_BUILD)/kernels_micro_kernel_fp16B_8x3.o \
    $(VLM_BUILD)/kernels_micro_kernel_bf16B_8x3.o \
    $(VLM_BUILD)/kernels_micro_kernel_bf16B_8x3_pv.o

MAIN_SRC := $(LLM_DIR)llm_runner.c
PP_SRC   := $(LLM_DIR)pp_runner.c
TP_SRC   := $(LLM_DIR)tp_runner.c
EP_SRC   := $(LLM_DIR)ep_runner.c
DS4F_SRC := $(LLM_DIR)ds4f_runner.c
DS4F_EP_SRC := $(LLM_DIR)ds4f_ep_runner.c
DS4F_DECBW_SRC := $(LLM_DIR)ds4f_decode_bw_bench.c
# header deps: ds4f.h / ggml_dequant.h are not in the .c prereqs, so editing them
# would otherwise leave the binaries stale (a no-op `make`). Track them explicitly.
DS4F_HDRS := $(LLM_DIR)../../common/ds4f.h $(LLM_DIR)../../common/ds4f_impl.h $(LLM_DIR)../../common/ggml_dequant.h
# MiniMax-M3 (text) synthetic EP runner (a64fx/m3). Self-contained: common/m3.h +
# m3_impl.h + uTofu. Links libtofucom; no vlm objs.
M3_EP_SRC := $(LLM_DIR)../m3/m3_ep_runner.c
M3_HDRS   := $(COMMON)/m3.h $(COMMON)/m3_impl.h $(COMMON)/ggml_dequant.h
GLM5_EP_SRC := $(LLM_DIR)../glm5/glm5_ep_runner.c
GLM5_HDRS   := $(COMMON)/glm5.h $(COMMON)/glm5_impl.h $(COMMON)/glm5_mem.h $(COMMON)/glm5_mxfp8.h $(COMMON)/glm5_int8.h $(COMMON)/ggml_dequant.h \
               $(LLM_DIR)../utofu-tests/tp_allreduce.h $(LLM_DIR)../utofu-tests/tofu_demo.h

# ── Targets ────────────────────────────────────────────────────────────
# ds4f_bake MUST be here: the alias target name matches ds4f_bake.c, so without .PHONY make's
# built-in "%: %.c" rule fires and re-links the finished binary as if it were an object.
.PHONY: all clean vlm_objs qwen38_runner qwen38_pp_runner qwen38_prefill_runner qwen38_tp_stage pp_runner tp_runner tp_runner_prof ep_runner ds4f_runner ds4f_ep_runner ds4f_decode_bw ds4f_bake m3_ep_runner glm5_ep_runner glm5_ep_runner_fapp glm5_stage glm52_gguf_convert glm5_real_test glm5_fp8_kernel_test

all: $(BUILD)/llm_runner

qwen38_runner: $(BUILD)/qwen38_runner
$(BUILD)/qwen38_runner: $(LLM_DIR)qwen38_runner.c $(COMMON)/transformer.h $(COMMON)/gguf_loader.h | $(BUILD)
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -o $@ $(LLM_DIR)qwen38_runner.c $(LDFLAGS)

qwen38_pp_runner: $(BUILD)/qwen38_pp_runner
$(BUILD)/qwen38_pp_runner: $(LLM_DIR)qwen38_pp_runner.c $(COMMON)/transformer.h $(COMMON)/gguf_loader.h | $(BUILD)
	OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43 /opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifcc \
		$(CFLAGS) -o $@ $(LLM_DIR)qwen38_pp_runner.c $(LDFLAGS)

qwen38_prefill_runner: $(BUILD)/qwen38_prefill_runner
$(BUILD)/qwen38_prefill_runner: $(LLM_DIR)qwen38_prefill_runner.c $(COMMON)/transformer.h $(COMMON)/gguf_loader.h $(COMMON)/ggml_dequant.h $(ROOT)/utofu-tests/tp_allreduce.h $(ROOT)/utofu-tests/tp_rsag4.h $(VLM_KERNELS)/micro_kernel_bf16B_8x3_pv.S $(BF16_PODD_KERNEL) $(Q8V2_KERNEL) $(Q8B6_KERNEL) | $(BUILD)
	OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43 /opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifcc \
		$(filter-out -g -rdynamic,$(CFLAGS)) -DTF_LINK_BF16PV48 -DTF_LINK_PODD -DTF_HAVE_Q8V2 $(QWEN_CFLAGS) -o $@ \
		$(LLM_DIR)qwen38_prefill_runner.c $(VLM_KERNELS)/micro_kernel_bf16B_8x3_pv.S $(BF16_PODD_KERNEL) $(Q8V2_KERNEL) $(Q8B6_KERNEL) $(LDFLAGS) -ltofucom

qwen38_tp_stage: $(BUILD)/qwen38_tp_stage
$(BUILD)/qwen38_tp_stage: $(LLM_DIR)qwen38_tp_stage.c $(LLM_DIR)qwen38_tp_stage.h $(COMMON)/gguf_loader.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(LLM_DIR)qwen38_tp_stage.c -lm

$(BUILD):
	@mkdir -p $(BUILD)

# Force the vlm objects to be present (delegate to a64fx/vlm).
vlm_objs:
	$(MAKE) -C $(VLM_DIR) CC=$(CC) OPENMP=$(OPENMP)

$(BUILD)/llm_runner: $(MAIN_SRC) vlm_objs | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(MAIN_SRC) $(VLM_OBJS) $(LDFLAGS)

# Multi-node pipeline-parallel decode runner (pure uTofu, no MPI in the binary;
# mpiexec only places ranks). Links libtofucom for uTofu one-sided Put/barrier.
pp_runner: $(BUILD)/pp_runner
$(BUILD)/pp_runner: $(PP_SRC) vlm_objs | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(PP_SRC) $(VLM_OBJS) $(LDFLAGS) -ltofucom

# Multi-node tensor-parallel decode runner (pure uTofu, no MPI in the binary).
tp_runner: $(BUILD)/tp_runner
$(BUILD)/tp_runner: $(TP_SRC) $(COMMON)/transformer.h $(COMMON)/gguf_loader.h $(Q8V2_KERNEL) $(Q8B6_KERNEL) | $(BUILD)
	$(CC) $(filter-out -g -rdynamic,$(CFLAGS)) -DTF_HAVE_Q8V2 $(QWEN_CFLAGS) -o $@ \
		$(TP_SRC) $(Q8V2_KERNEL) $(Q8B6_KERNEL) $(LDFLAGS) -ltofucom

# Same as tp_runner but with -DTF_POOL_PROFILE: dumps the per-dispatch work-vs-wait
# and per-tid matvec/barrier/attn decode breakdown at pool shutdown. Separate binary
# so production tp_runner stays profile-free.
tp_runner_prof: $(BUILD)/tp_runner_prof
$(BUILD)/tp_runner_prof: $(TP_SRC) vlm_objs | $(BUILD)
	$(CC) $(CFLAGS) -DTF_POOL_PROFILE -o $@ $(TP_SRC) $(VLM_OBJS) $(LDFLAGS) -ltofucom

# Multi-node expert-parallel decode runner for MoE models. Shards experts across
# ranks; non-expert weights replicated. Pure uTofu, no MPI in the binary.
ep_runner: $(BUILD)/ep_runner
$(BUILD)/ep_runner: $(EP_SRC) | vlm_objs $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(EP_SRC) $(VLM_OBJS) $(LDFLAGS) -ltofucom

# DeepSeek-V4-Flash synthetic perf/plumbing harness (Stage 1, single node).
# Self-contained: only common/ds4f.h (+ ggml_dequant.h). No vlm objs, no uTofu.
ds4f_runner: $(BUILD)/ds4f_runner
$(BUILD)/ds4f_runner: $(DS4F_SRC) $(DS4F_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(DS4F_SRC) $(LDFLAGS)

# DeepSeek-V4-Flash multinode EP harness (Stage 2, all-reduce combine).
# Self-contained: common/ds4f.h + uTofu (a64fx/utofu-tests headers). Links
# libtofucom for one-sided Put/barrier; no vlm objs.
ds4f_ep_runner: $(BUILD)/ds4f_ep_runner
$(BUILD)/ds4f_ep_runner: $(DS4F_EP_SRC) $(DS4F_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(DS4F_EP_SRC) $(LDFLAGS) -ltofucom

# MiniMax-M3 multinode EP harness (synthetic; GQA + sigmoid-MoE + SwiGLU-OAI).
m3_ep_runner: $(BUILD)/m3_ep_runner
$(BUILD)/m3_ep_runner: $(M3_EP_SRC) $(M3_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(M3_EP_SRC) $(LDFLAGS) -ltofucom

# M3 weight stager (safetensors -> per-rank /local blob) and single-node real-weight test.
m3_stage: $(BUILD)/m3_stage
$(BUILD)/m3_stage: $(COMMON)/m3_stage.c $(COMMON)/safetensors.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(COMMON)/m3_stage.c $(LDFLAGS)
m3_real_test: $(BUILD)/m3_real_test
$(BUILD)/m3_real_test: $(LLM_DIR)../m3/m3_real_test.c $(M3_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(LLM_DIR)../m3/m3_real_test.c $(LDFLAGS)

# GLM-5.2 192-node BF16 bring-up path.
glm5_ep_runner: $(BUILD)/glm5_ep_runner
$(BUILD)/glm5_ep_runner: $(GLM5_EP_SRC) $(GLM5_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(GLM5_EP_SRC) $(LDFLAGS) -ltofucom
glm5_ep_runner_fapp: $(BUILD)/glm5_ep_runner_fapp
$(BUILD)/glm5_ep_runner_fapp: $(GLM5_EP_SRC) $(GLM5_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -DGLM5_USE_FAPP=1 -o $@ $(GLM5_EP_SRC) $(LDFLAGS) -ltofucom
glm5_stage: $(BUILD)/glm5_stage
$(BUILD)/glm5_stage: $(COMMON)/glm5_stage.c $(COMMON)/safetensors.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(COMMON)/glm5_stage.c $(LDFLAGS)
glm52_gguf_convert: $(BUILD)/glm52_gguf_convert
$(BUILD)/glm52_gguf_convert: $(LLM_DIR)../glm5/glm52_gguf_convert.c $(COMMON)/gguf_loader.h $(COMMON)/ggml_dequant.h | $(BUILD)
	$(HOSTCC) -O2 -std=c11 -D_GNU_SOURCE -I$(COMMON) -o $@ $(LLM_DIR)../glm5/glm52_gguf_convert.c -lm
glm5_real_test: $(BUILD)/glm5_real_test
$(BUILD)/glm5_real_test: $(LLM_DIR)../glm5/glm5_real_test.c $(GLM5_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(LLM_DIR)../glm5/glm5_real_test.c $(LDFLAGS)
glm5_fp8_kernel_test: $(LLM_DIR)../glm5/glm5_fp8_kernel_test
$(LLM_DIR)../glm5/glm5_fp8_kernel_test: $(LLM_DIR)../glm5/glm5_fp8_kernel_test.c $(GLM5_HDRS)
	$(CC) $(CFLAGS) -o $@ $(LLM_DIR)../glm5/glm5_fp8_kernel_test.c $(LDFLAGS)

# DeepSeek-V4-Flash single-node decode-bandwidth bench (no MPI, no vlm objs, no
# model alloc). Reproduces the M=1 cold-HBM decode regime through the production
# pinned pool to roofline FP8/bf16/MXFP4 dequant kernels. Runs directly on-node.
ds4f_decode_bw: $(BUILD)/ds4f_decode_bw
$(BUILD)/ds4f_decode_bw: $(DS4F_DECBW_SRC) $(DS4F_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(DS4F_DECBW_SRC) $(LDFLAGS)

# OFFLINE dense bake: safetensors FP8 -> the kernel-ready BF16_PV / Q8_PV layouts, written
# permanently under ~/models/<model>-fast/. Removes the runtime bf16 promotion peak (which is
# what makes the ds4f 8-node decode floor "load-peak-tight" and blocks Q8 dense on ds4fbase
# entirely). Single node, no allocation, no MPI. Calls the SAME promote/repack workers the
# runtime uses, so the bytes are identical by construction. See the header of ds4f_bake.c.
ds4f_bake: $(BUILD)/ds4f_bake
$(BUILD)/ds4f_bake: $(LLM_DIR)ds4f_bake.c $(DS4F_HDRS) $(COMMON)/safetensors.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(LLM_DIR)ds4f_bake.c $(LDFLAGS)

clean:
	rm -rf $(BUILD)
