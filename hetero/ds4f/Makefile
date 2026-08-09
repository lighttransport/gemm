# hetero/ds4f — DeepSeek-V4-Flash on Threadripper 1950X (AVX2) + RX 9070 XT (gfx1201)
#
# Plain gcc only: HIP is dlopen'd through ../../rdna4/rocew.c and device kernels
# are compiled at startup with HIPRTC, so no ROCm SDK is needed to build.

CC      ?= gcc
ARCH    ?= znver1
CFLAGS  ?= -O3 -Wall -Wextra -std=c11 -D_GNU_SOURCE -march=$(ARCH) -mavx2 -mfma -mf16c -ffp-contract=fast
INCS    := -I. -I../../common -I../../rdna4
LDFLAGS := -lm -lpthread -ldl

BUILD := build

.PHONY: all clean verify test hip-test real-hip-test real-dual-test forward-hip-test real-token-test prefill-test real-prefill-test mtp-test cuda-test dual-check dual-test x86-runner sve-check
all: $(BUILD)/bench_expert_bw $(BUILD)/test_ds4f_kernels $(BUILD)/test_hip_ds4f_dense \
    $(BUILD)/test_hip_ds4f_real $(BUILD)/test_hip_ds4f_forward $(BUILD)/test_ds4f_real_tokens \
    $(BUILD)/test_ds4f_prefill $(BUILD)/test_ds4f_prefill_real $(BUILD)/test_ds4f_mtp $(BUILD)/ds4f_runner

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/bench_expert_bw: bench_expert_bw.c mxfp4_avx2.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -fopenmp -o $@ bench_expert_bw.c $(LDFLAGS)

$(BUILD)/test_ds4f_kernels: test_ds4f_kernels.c ../../common/ds4f_matvec_avx2.h \
                            ../../common/ds4f_impl.h ../../common/ds4f.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -o $@ test_ds4f_kernels.c $(LDFLAGS)

$(BUILD)/test_hip_ds4f_dense: test_hip_ds4f_dense.c hip_ds4f_dense.c \
                              hip_ds4f_dense.h hip_ds4f_kernels.h \
                              ../../rdna4/rocew.c ../../rdna4/rocew.h \
                              ../../rdna4/hip_runner_common.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -o $@ test_hip_ds4f_dense.c hip_ds4f_dense.c \
		../../rdna4/rocew.c $(LDFLAGS)

$(BUILD)/test_cuda_ds4f_dense: test_cuda_ds4f_dense.c ../../cuda/cuew.c \
                                ../../cuda/cuew.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -I../../cuda -o $@ test_cuda_ds4f_dense.c \
		../../cuda/cuew.c $(LDFLAGS)

$(BUILD)/test_cuda_ds4f_real_dense: test_cuda_ds4f_real_dense.c cuda_ds4f_dense.c \
                                     cuda_ds4f_dense.h ../../common/ds4f.h \
                                     ../../common/ds4f_impl.h ../../cuda/cuew.c | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -I../../cuda -o $@ test_cuda_ds4f_real_dense.c \
		cuda_ds4f_dense.c ../../cuda/cuew.c ../../cuda/cublasew.c $(LDFLAGS)

$(BUILD)/test_hip_ds4f_real: test_hip_ds4f_real.c hip_ds4f_dense.c \
                             dual_ds4f_prefill.c dual_ds4f_prefill.h \
                             cuda_ds4f_mxfp4.c cuda_ds4f_mxfp4.h \
                             hip_ds4f_dense.h hip_ds4f_kernels.h \
                             ../../common/ds4f.h ../../common/ds4f_impl.h \
                             ../../rdna4/rocew.c ../../rdna4/rocew.h \
                             ../../rdna4/hip_runner_common.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -I../../cuda -o $@ test_hip_ds4f_real.c \
		hip_ds4f_dense.c dual_ds4f_prefill.c cuda_ds4f_mxfp4.c \
		../../rdna4/rocew.c ../../cuda/cuew.c $(LDFLAGS)

$(BUILD)/test_hip_ds4f_forward: test_hip_ds4f_forward.c hip_ds4f_dense.c \
                                hip_ds4f_dense.h hip_ds4f_kernels.h \
                                ../../common/ds4f.h ../../common/ds4f_impl.h \
                                ../../rdna4/rocew.c ../../rdna4/rocew.h \
                                ../../rdna4/hip_runner_common.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -o $@ test_hip_ds4f_forward.c hip_ds4f_dense.c \
		../../rdna4/rocew.c $(LDFLAGS)

$(BUILD)/test_ds4f_real_tokens: test_ds4f_real_tokens.c \
                                ../../common/ds4f.h ../../common/ds4f_impl.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -o $@ test_ds4f_real_tokens.c $(LDFLAGS)

$(BUILD)/test_ds4f_prefill: test_ds4f_prefill.c \
                            ../../common/ds4f.h ../../common/ds4f_impl.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -o $@ test_ds4f_prefill.c $(LDFLAGS)

$(BUILD)/test_ds4f_prefill_real: test_ds4f_prefill_real.c \
                                 ../../common/ds4f.h ../../common/ds4f_impl.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -o $@ test_ds4f_prefill_real.c $(LDFLAGS)

$(BUILD)/test_ds4f_mtp: test_ds4f_mtp.c ../../common/ds4f.h ../../common/ds4f_impl.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -o $@ test_ds4f_mtp.c $(LDFLAGS)

$(BUILD)/ds4f_runner: ../../a64fx/llm/ds4f_runner.c \
                      ../../common/ds4f.h ../../common/ds4f_impl.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -o $@ ../../a64fx/llm/ds4f_runner.c $(LDFLAGS)

$(BUILD)/test_cuda_ds4f_mxfp4: test_cuda_ds4f_mxfp4.c cuda_ds4f_mxfp4.c \
                               cuda_ds4f_mxfp4.h ../../cuda/cuew.c ../../cuda/cuew.h | $(BUILD)
	$(CC) $(CFLAGS) -I../../cuda -o $@ test_cuda_ds4f_mxfp4.c cuda_ds4f_mxfp4.c \
		../../cuda/cuew.c $(LDFLAGS)

$(BUILD)/dual_ds4f_prefill.o: dual_ds4f_prefill.c dual_ds4f_prefill.h \
                              hip_ds4f_dense.h cuda_ds4f_mxfp4.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -I../../cuda -c -o $@ dual_ds4f_prefill.c

$(BUILD)/test_dual_ds4f_prefill: test_dual_ds4f_prefill.c dual_ds4f_prefill.c \
                                 cuda_ds4f_mxfp4.c \
                                 dual_ds4f_prefill.h hip_ds4f_dense.c hip_ds4f_dense.h \
                                 hip_ds4f_kernels.h ../../rdna4/rocew.c \
                                 ../../cuda/cuew.c | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -I../../cuda -o $@ test_dual_ds4f_prefill.c \
		dual_ds4f_prefill.c cuda_ds4f_mxfp4.c hip_ds4f_dense.c ../../rdna4/rocew.c \
		../../cuda/cuew.c $(LDFLAGS)

verify: $(BUILD)/bench_expert_bw
	$(BUILD)/bench_expert_bw --mode verify

test: $(BUILD)/test_ds4f_kernels
	$(BUILD)/test_ds4f_kernels

hip-test: $(BUILD)/test_hip_ds4f_dense
	$(BUILD)/test_hip_ds4f_dense

real-hip-test: $(BUILD)/test_hip_ds4f_real
	$(BUILD)/test_hip_ds4f_real

# Reproducible staged NVIDIA+AMD prefill gate. Override STAGE_DIR, EP_SIZE,
# EP_RANK, LAYERS, and PREFILL_BATCH for another shard or context size.
# HIP_ORDERED_FP8_LAYERS defaults to LAYERS so the gate is argmax-exact
# (0 mismatches); set it to 0 to run the faster non-ordered dense reduction.
#
# Thread default is one thread per PHYSICAL core (no SMT siblings).  The
# spin-wait prefill pool contends on SMT siblings and adds run-to-run variance
# (threads=32 on this 16-core/32-thread part measured 33-43 tok/s; physical 16
# is a steady ~42).  The old hard default 48 oversubscribed and dropped the
# prefill to ~17 tok/s.  Set THREADS=N to override.
PHYS_CORES := $(shell lscpu -p=core 2>/dev/null | grep -v '^#' | cut -d, -f2 | sort -u | wc -l)
ifeq ($(strip $(PHYS_CORES)),0)
  PHYS_CORES := $(shell nproc)
endif
real-dual-test: $(BUILD)/test_hip_ds4f_real
	@test -n "$(STAGE_DIR)" || { echo "usage: make -C hetero/ds4f real-dual-test STAGE_DIR=/tmp/ds4f_nocopy_ep8" >&2; exit 2; }
	@test -f ../../cuda/llm/mmq_kernels.cubin || { echo "build cuda/llm/mmq_kernels.cubin first" >&2; exit 2; }
	$(BUILD)/test_hip_ds4f_real --stage-dir $(STAGE_DIR) \
		--ep-size $(or $(EP_SIZE),8) --ep-rank $(or $(EP_RANK),0) \
		--layers $(or $(LAYERS),43) --bank-layers $(or $(BANK_LAYERS),43) \
		--threads $(or $(THREADS),$(PHYS_CORES)) --cmgs $(or $(CMGS),4) \
		--prefill-batch $(or $(PREFILL_BATCH),64) \
		--prefill-context $(or $(PREFILL_CONTEXT),0) \
		--dual-gpu 1 --cuda-device $(or $(CUDA_DEVICE),0) \
		--dual-cuda-terms $(or $(DUAL_CUDA_TERMS),1) \
		--hip-device $(or $(HIP_DEVICE),0) --hip-verbose $(or $(HIP_VERBOSE),0) \
		--hip-exact-prefill $(or $(HIP_EXACT_PREFILL),0) \
		--hip-ordered-fp8-layers $(or $(HIP_ORDERED_FP8_LAYERS),$(or $(LAYERS),43)) \
		--hip-fused-shared-ffn $(or $(HIP_FUSED_SHARED_FFN),1) \
		--dual-cuda-small-buckets $(or $(DUAL_CUDA_SMALL_BUCKETS),0) \
		--dual-cuda-resident-from $(or $(DUAL_CUDA_RESIDENT_FROM),0) \
		--prefill-repeat $(or $(PREFILL_REPEAT),1)

forward-hip-test: $(BUILD)/test_hip_ds4f_forward
	$(BUILD)/test_hip_ds4f_forward

real-token-test: $(BUILD)/test_ds4f_real_tokens
	$(BUILD)/test_ds4f_real_tokens

prefill-test: $(BUILD)/test_ds4f_prefill
	$(BUILD)/test_ds4f_prefill

real-prefill-test: $(BUILD)/test_ds4f_prefill_real
	$(BUILD)/test_ds4f_prefill_real

mtp-test: $(BUILD)/test_ds4f_mtp
	$(BUILD)/test_ds4f_mtp

cuda-test: $(BUILD)/test_cuda_ds4f_mxfp4
	@test -f ../../cuda/llm/mmq_kernels.cubin || { echo "build cuda/llm/mmq_kernels.cubin first" >&2; exit 2; }
	$(BUILD)/test_cuda_ds4f_mxfp4 64
	$(BUILD)/test_cuda_ds4f_mxfp4 128

dual-check: $(BUILD)/dual_ds4f_prefill.o

dual-test: $(BUILD)/test_dual_ds4f_prefill
	@test -f ../../cuda/llm/mmq_kernels.cubin || { echo "build cuda/llm/mmq_kernels.cubin first" >&2; exit 2; }
	$(BUILD)/test_dual_ds4f_prefill

x86-runner: $(BUILD)/ds4f_runner

# The DS4F model headers are shared with the A64FX runners in a64fx/llm. Every
# change to their SVE guards must keep that build compiling; this checks the
# __ARM_FEATURE_SVE branch without needing the Fujitsu toolchain.
sve-check:
	@printf '#include "ds4f.h"\nint main(void){return 0;}\n' > $(BUILD)/sve_tu.c
	clang --target=aarch64-linux-gnu -march=armv8.2-a+sve -O1 -std=c11 \
	      -D_GNU_SOURCE -I../../common -fsyntax-only $(BUILD)/sve_tu.c
	@echo "SVE branch OK"

clean:
	rm -rf $(BUILD)
