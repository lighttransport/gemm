# NOT `CC ?= fcc`: make predefines CC=cc, whose origin is "default" rather than
# "undefined", so ?= never fires and the build silently used gcc -- which drops
# -march=armv8.2-a+sve, compiles the (incomplete) non-SVE fallbacks, and fails at
# link with undefined references to laguna_exp_shift_sum.  An explicit CC=... on
# the command line or in the environment still wins.
ifeq ($(origin CC),default)
  CC := fcc
endif
OPENMP ?= 1

ROOT := $(abspath ../..)
COMMON := $(ROOT)/common
UTOFU := $(ROOT)/a64fx/utofu-tests
BUILD := build

ifeq ($(CC),fcc)
  CFLAGS := -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE
  OMPFLAG := -fopenmp
else
  CFLAGS := -O3 -std=c11 -D_GNU_SOURCE -Wall -Wextra -Wpedantic
  OMPFLAG := -fopenmp
endif
ifeq ($(OPENMP),1)
  CFLAGS += $(OMPFLAG)
  LDFLAGS += $(OMPFLAG)
endif
CFLAGS += -I$(COMMON) -I$(UTOFU)
LDFLAGS += -lm -lpthread
TOFU_LIBS ?= -ltofucom

.PHONY: all clean test bench bf16 fp8 fp8-kvfp16
all: $(BUILD)/laguna_s21_stage $(BUILD)/laguna_s21_ep_runner

# Pure-bf16 reference build (~/models/laguna-s21, unquantized): same runner with
# -DLAGUNA_BF16. Shares the stager (stage the bf16 checkpoint with LAGUNA_NSHARDS=46).
bf16: $(BUILD)/laguna_s21_bf16_ep_runner
$(BUILD)/laguna_s21_bf16_ep_runner: laguna_s21_ep_runner.c laguna_s21.h laguna_tofu.inc | $(BUILD)
	$(CC) $(CFLAGS) -DLAGUNA_BF16 -o $@ laguna_s21_ep_runner.c $(LDFLAGS) $(TOFU_LIBS)

# fp8 build (~/models/laguna-s21-fp8): bf16 linears + fp8-e4m3 block-128 experts.
# Stage the 24-shard checkpoint with LAGUNA_NSHARDS=24.
fp8: $(BUILD)/laguna_s21_fp8_ep_runner
$(BUILD)/laguna_s21_fp8_ep_runner: laguna_s21_ep_runner.c laguna_s21.h laguna_tofu.inc | $(BUILD)
	$(CC) $(CFLAGS) -DLAGUNA_FP8 -o $@ laguna_s21_ep_runner.c $(LDFLAGS) $(TOFU_LIBS)

# Quality A/B: same FP8-weight runner with IEEE FP16 rather than BF16 KV.
fp8-kvfp16: $(BUILD)/laguna_s21_fp8_kvfp16_ep_runner
$(BUILD)/laguna_s21_fp8_kvfp16_ep_runner: laguna_s21_ep_runner.c laguna_s21.h laguna_tofu.inc | $(BUILD)
	$(CC) $(CFLAGS) -DLAGUNA_FP8 -DLAGUNA_KV_FP16 -o $@ laguna_s21_ep_runner.c $(LDFLAGS) $(TOFU_LIBS)

$(BUILD):
	mkdir -p $@

$(BUILD)/laguna_s21_stage: laguna_s21_stage.c $(COMMON)/safetensors.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ laguna_s21_stage.c $(LDFLAGS)

$(BUILD)/laguna_s21_ep_runner: laguna_s21_ep_runner.c laguna_s21.h laguna_tofu.inc | $(BUILD)
	$(CC) $(CFLAGS) -o $@ laguna_s21_ep_runner.c $(LDFLAGS) $(TOFU_LIBS)

test: $(BUILD)/laguna_s21_ep_runner $(BUILD)/laguna_s21_bf16_ep_runner \
      $(BUILD)/laguna_s21_fp8_ep_runner $(BUILD)/laguna_s21_fp8_kvfp16_ep_runner
	$(BUILD)/laguna_s21_ep_runner --self-test
	$(BUILD)/laguna_s21_bf16_ep_runner --self-test
	$(BUILD)/laguna_s21_fp8_ep_runner --self-test
	$(BUILD)/laguna_s21_fp8_kvfp16_ep_runner --self-test

# Single-node synthetic decode benchmark (no uTofu). Run: OMP_NUM_THREADS=47 \
#   XOS_MMM_L_PAGING_POLICY=demand:demand:demand build/bench_decode 48 1 30
bench: $(BUILD)/bench_decode
$(BUILD)/bench_decode: bench_decode.c laguna_s21_ep_runner.c laguna_s21.h | $(BUILD)
	$(CC) $(CFLAGS) -DLAGUNA_BENCH -o $@ bench_decode.c $(LDFLAGS)

clean:
	rm -rf $(BUILD)
