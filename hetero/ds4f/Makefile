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

.PHONY: all clean verify test sve-check
all: $(BUILD)/bench_expert_bw $(BUILD)/test_ds4f_kernels

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/bench_expert_bw: bench_expert_bw.c mxfp4_avx2.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -fopenmp -o $@ bench_expert_bw.c $(LDFLAGS)

$(BUILD)/test_ds4f_kernels: test_ds4f_kernels.c ../../common/ds4f_matvec_avx2.h \
                            ../../common/ds4f_impl.h ../../common/ds4f.h | $(BUILD)
	$(CC) $(CFLAGS) $(INCS) -o $@ test_ds4f_kernels.c $(LDFLAGS)

verify: $(BUILD)/bench_expert_bw
	$(BUILD)/bench_expert_bw --mode verify

test: $(BUILD)/test_ds4f_kernels
	$(BUILD)/test_ds4f_kernels

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
