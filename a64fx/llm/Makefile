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

# ── Targets ────────────────────────────────────────────────────────────
.PHONY: all clean vlm_objs

all: $(BUILD)/llm_runner

$(BUILD):
	@mkdir -p $(BUILD)

# Force the vlm objects to be present (delegate to a64fx/vlm).
vlm_objs:
	$(MAKE) -C $(VLM_DIR) CC=$(CC) OPENMP=$(OPENMP)

$(BUILD)/llm_runner: $(MAIN_SRC) vlm_objs | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(MAIN_SRC) $(VLM_OBJS) $(LDFLAGS)

clean:
	rm -rf $(BUILD)
