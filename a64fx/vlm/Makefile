# a64fx/vlm Makefile
#
# Usage:
#   make CC=fcc                     # Fujitsu fcc, sequential
#   make CC=fcc OPENMP=1            # Fujitsu fcc, OpenMP
#   make CC=gcc                     # gcc, C11-thread backend
#   make CC=clang OPENMP=1          # clang + libomp
#   make CC=fcc REF=1               # also build cpu/vlm test_vision with -DVLM_DUMP_REFERENCE=1
#
# Outputs:
#   build/vlm_runner   - a64fx vision-only runner
#   build/tensor_diff  - VLMD dump validator

CC      ?= fcc
OPENMP  ?=
DEBUG   ?=
REF     ?=

VLM_DIR     := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMMON      := $(VLM_DIR)../../common
KERNELS     := $(VLM_DIR)kernels
INC         := $(VLM_DIR)include
SRC         := $(VLM_DIR)src
TOOLS       := $(VLM_DIR)tools
BUILD       := $(VLM_DIR)build

# ── Compiler-specific flags ────────────────────────────────────────────
# fcc native compiler (Fujitsu) wants -Nclang for C11/clang frontend.
ifeq ($(CC),fcc)
    BASE_CFLAGS  := -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
    BASE_CFLAGS  += -Wno-unused-function -Wno-unused-variable
    OMP_FLAG     := -fopenmp
    ASMC         := fcc
else ifeq ($(CC),fccpx)
    BASE_CFLAGS  := -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
    BASE_CFLAGS  += -Wno-unused-function -Wno-unused-variable
    OMP_FLAG     := -fopenmp
    ASMC         := fccpx
else ifeq ($(CC),gcc)
    BASE_CFLAGS  := -O3 -march=armv8.2-a+sve -ffp-contract=fast
    BASE_CFLAGS  += -Wall -Wno-unused-function -Wno-unused-variable
    OMP_FLAG     := -fopenmp
    ASMC         := gcc
else ifeq ($(CC),clang)
    BASE_CFLAGS  := -O3 -march=armv8.2-a+sve -ffp-contract=fast
    BASE_CFLAGS  += -Wall -Wno-unused-function -Wno-unused-variable
    OMP_FLAG     := -fopenmp
    ASMC         := clang
else
    BASE_CFLAGS  := -O2
    OMP_FLAG     :=
    ASMC         := $(CC)
endif

CFLAGS  := $(BASE_CFLAGS) -std=c11 -D_GNU_SOURCE -I$(INC) -I$(KERNELS) -I$(COMMON)
LDFLAGS := -lm -lpthread

# C++ test (test_conv2d_sve.cpp) is compiled with the SAME compiler as the C
# kernels (fcc in C++ mode) so the OpenMP runtime matches — g++'s libgomp does
# not resolve fcc's __kmpc_* symbols. Headers have extern "C" guards, so the C
# kernel objects link in cleanly.
ifeq ($(CC),fcc)
    CXXFLAGS := -Nclang -O2 -march=armv8.2-a+sve -std=c++17 -D_GNU_SOURCE -I$(INC) -I$(KERNELS) -I$(COMMON)
else ifeq ($(CC),fccpx)
    CXXFLAGS := -Nclang -O2 -march=armv8.2-a+sve -std=c++17 -D_GNU_SOURCE -I$(INC) -I$(KERNELS) -I$(COMMON)
else
    CXXFLAGS := -O2 -march=armv8.2-a+sve -std=c++17 -D_GNU_SOURCE -I$(INC) -I$(KERNELS) -I$(COMMON)
endif
ifdef OPENMP
    CXXFLAGS += $(OMP_FLAG)
endif

# fcc/fccpx support OpenMP, and the OpenMP backend of vlm_parallel scales far
# better than the C11-thrd fallback (which clusters its worker threads on a few
# cores on A64FX: ~1.4x over 48 cores vs ~25x for OpenMP, and the attention is
# the dominant stage). Default to OpenMP for the Fujitsu compilers; pass
# OPENMP=0 to force the C11-thrd backend.
ifeq ($(OPENMP),)
  ifeq ($(CC),fcc)
    OPENMP := 1
  else ifeq ($(CC),fccpx)
    OPENMP := 1
  endif
endif

ifdef OPENMP
    CFLAGS  += $(OMP_FLAG) -DUSE_OPENMP=1
    LDFLAGS += $(OMP_FLAG)
endif

ifdef DEBUG
    CFLAGS := $(subst -O3,-O0,$(CFLAGS)) -g
endif

# ── Sources ────────────────────────────────────────────────────────────
# M2: norm_sve.c is wired in; vit_a64fx calls layernorm_batch_fwd_f32.
KERNEL_C := $(KERNELS)/fused_gemm.c $(KERNELS)/pack_matrices.c $(KERNELS)/norm_sve.c $(KERNELS)/bf16_gemm.c $(KERNELS)/fp16_gemm.c $(KERNELS)/fp32_gemm.c $(KERNELS)/conv2d_sve.c $(KERNELS)/int8_gemm.c
KERNEL_S := $(KERNELS)/micro_kernel_fp32_8x3.S $(KERNELS)/micro_kernel_fp16B_8x3.S $(KERNELS)/micro_kernel_bf16B_8x3.S $(KERNELS)/micro_kernel_bf16B_8x3_pv.S $(KERNELS)/kernel_6x4_int8.S
RUNTIME  := $(SRC)/vlm_parallel.c $(SRC)/tensor_dump.c $(SRC)/vit_a64fx.c $(SRC)/cmg_pool.c
MAIN_SRC := $(SRC)/vlm_runner.c

KERNEL_OBJ := $(patsubst $(KERNELS)/%.c,$(BUILD)/kernels_%.o,$(KERNEL_C)) \
              $(patsubst $(KERNELS)/%.S,$(BUILD)/kernels_%.o,$(KERNEL_S))
RUNTIME_OBJ := $(patsubst $(SRC)/%.c,$(BUILD)/src_%.o,$(RUNTIME))

# ── Targets ────────────────────────────────────────────────────────────
.PHONY: all clean ref test

all: $(BUILD)/vlm_runner $(BUILD)/tensor_diff $(BUILD)/test_cmg_pool

$(BUILD):
	@mkdir -p $(BUILD)

# kernels (kernels/*.c)
$(BUILD)/kernels_%.o: $(KERNELS)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

# kernels (kernels/*.S) — fcc accepts asm via -c, same as gcc
$(BUILD)/kernels_%.o: $(KERNELS)/%.S | $(BUILD)
	$(ASMC) $(CFLAGS) -c -o $@ $<

# runtime sources
$(BUILD)/src_%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/vlm_runner: $(MAIN_SRC) $(RUNTIME_OBJ) $(KERNEL_OBJ) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(MAIN_SRC) $(RUNTIME_OBJ) $(KERNEL_OBJ) $(LDFLAGS)

$(BUILD)/tensor_diff: $(TOOLS)/tensor_diff.c $(SRC)/tensor_dump.c | $(BUILD)
	$(CC) $(CFLAGS) -I$(INC) -o $@ $< $(SRC)/tensor_dump.c $(LDFLAGS)

$(BUILD)/test_cmg_pool: $(TOOLS)/test_cmg_pool.c $(BUILD)/src_cmg_pool.o | $(BUILD)
	$(CC) $(CFLAGS) -I$(SRC) -o $@ $< $(BUILD)/src_cmg_pool.o $(LDFLAGS)

$(BUILD)/test_int8_gemm: $(TOOLS)/test_int8_gemm.c $(BUILD)/kernels_int8_gemm.o $(BUILD)/kernels_kernel_6x4_int8.o | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< $(BUILD)/kernels_int8_gemm.o $(BUILD)/kernels_kernel_6x4_int8.o $(LDFLAGS)

$(BUILD)/test_conv2d_sve: $(TOOLS)/test_conv2d_sve.cpp $(BUILD)/kernels_conv2d_sve.o $(BUILD)/kernels_micro_kernel_fp32_8x3.o | $(BUILD)
	$(CC) $(CXXFLAGS) -o $@ $< $(BUILD)/kernels_conv2d_sve.o $(BUILD)/kernels_micro_kernel_fp32_8x3.o -lstdc++ $(LDFLAGS)

# `make test` — build + run the kernel unit tests (regression gate); each exits
# non-zero on failure. int8 GEMM: rel(L2) < 5% vs the fp32 ref. fused: must be
# bit-identical to the non-fused path. conv2d SVE: within 1e-3 of the scalar ref.
test: $(BUILD)/test_int8_gemm $(BUILD)/test_conv2d_sve $(BUILD)/test_cmg_pool
	@echo "=== int8 GEMM vs fp32 ref (expect rel L2 < 5%, OK) ==="
	$(BUILD)/test_int8_gemm 8
	@echo "=== fused int8 GEMM (expect BIT-IDENTICAL, OK) ==="
	$(BUILD)/test_int8_gemm fused
	@echo "=== conv2d SVE vs scalar ref (expect PASS) ==="
	$(BUILD)/test_conv2d_sve
	@echo "=== cmg pool NUMA sanity (expect OK) ==="
	$(BUILD)/test_cmg_pool
	@echo "ALL VLM UNIT TESTS PASSED"

# Optional: rebuild cpu/vlm test_vision with dump hooks for reference dumps.
# Invoked as `make REF=1`. The cpu Makefile uses gcc — this just kicks it.
# -D_GNU_SOURCE is required: transformer.h uses the GNU CPU_ZERO/CPU_SET
# affinity macros (plain POSIX <sched.h> on this glibc only defines the sized
# __CPU_*_S variants), so without it they are parsed as function calls and the
# reference binary fails to link.
ref:
	$(MAKE) -C $(COMMON)/../cpu/vlm CFLAGS="-O2 -D_GNU_SOURCE -DVLM_DUMP_REFERENCE=1 -Wall -Wextra"

clean:
	rm -rf $(BUILD)
