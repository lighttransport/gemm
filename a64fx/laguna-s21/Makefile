CC ?= fcc
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

.PHONY: all clean test bench
all: $(BUILD)/laguna_s21_stage $(BUILD)/laguna_s21_ep_runner

$(BUILD):
	mkdir -p $@

$(BUILD)/laguna_s21_stage: laguna_s21_stage.c $(COMMON)/safetensors.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ laguna_s21_stage.c $(LDFLAGS)

$(BUILD)/laguna_s21_ep_runner: laguna_s21_ep_runner.c laguna_s21.h laguna_tofu.inc | $(BUILD)
	$(CC) $(CFLAGS) -o $@ laguna_s21_ep_runner.c $(LDFLAGS) $(TOFU_LIBS)

test: $(BUILD)/laguna_s21_ep_runner
	$(BUILD)/laguna_s21_ep_runner --self-test

# Single-node synthetic decode benchmark (no uTofu). Run: OMP_NUM_THREADS=47 \
#   XOS_MMM_L_PAGING_POLICY=demand:demand:demand build/bench_decode 48 1 30
bench: $(BUILD)/bench_decode
$(BUILD)/bench_decode: bench_decode.c laguna_s21_ep_runner.c laguna_s21.h | $(BUILD)
	$(CC) $(CFLAGS) -DLAGUNA_BENCH -o $@ bench_decode.c $(LDFLAGS)

clean:
	rm -rf $(BUILD)
