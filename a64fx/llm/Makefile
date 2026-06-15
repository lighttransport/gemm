# a64fx/llm Makefile
#
# Builds an end-to-end VLM→LLM runner that pairs the existing a64fx vision
# encoder (a64fx/vlm) with common/transformer.h.
#
# Usage:
#   make CC=fcc                     # Fujitsu fcc, sequential
#   make CC=fcc OPENMP=1            # Fujitsu fcc + OpenMP (recommended)
#   make CC=gcc OPENMP=1            # gcc + OpenMP
#
# Outputs:
#   build/llm_runner   - end-to-end runner (text or text+image)

CC      ?= fcc
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

# ── Compiler-specific flags ────────────────────────────────────────────
ifeq ($(CC),fcc)
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

ifdef OPENMP
    CFLAGS  += $(OMP_FLAG) -DUSE_OPENMP=1
    LDFLAGS += $(OMP_FLAG)
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

# ── Targets ────────────────────────────────────────────────────────────
.PHONY: all clean vlm_objs pp_runner tp_runner tp_runner_prof ep_runner ds4f_runner ds4f_ep_runner ds4f_decode_bw m3_ep_runner

all: $(BUILD)/llm_runner

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
$(BUILD)/tp_runner: $(TP_SRC) vlm_objs | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(TP_SRC) $(VLM_OBJS) $(LDFLAGS) -ltofucom

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

# DeepSeek-V4-Flash single-node decode-bandwidth bench (no MPI, no vlm objs, no
# model alloc). Reproduces the M=1 cold-HBM decode regime through the production
# pinned pool to roofline FP8/bf16/MXFP4 dequant kernels. Runs directly on-node.
ds4f_decode_bw: $(BUILD)/ds4f_decode_bw
$(BUILD)/ds4f_decode_bw: $(DS4F_DECBW_SRC) $(DS4F_HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(DS4F_DECBW_SRC) $(LDFLAGS)

clean:
	rm -rf $(BUILD)
