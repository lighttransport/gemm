# CPU runner for FAIR's SAM 3D Objects.
# Scaffolded to mirror cpu/sam3/Makefile. Stage kernels land module-by-
# module; verify_*.c targets grow as each stage is ported.

COMPILER ?= gcc
ARCH ?= zen2
COMMON = ../../common

ifeq ($(ARCH),zen2)
    ARCH_FLAGS = -march=znver2 -mtune=znver2
else ifeq ($(ARCH),zen3)
    ARCH_FLAGS = -march=znver3 -mtune=znver3
else ifeq ($(ARCH),zen4)
    ARCH_FLAGS = -march=znver4 -mtune=znver4
else ifeq ($(ARCH),native)
    ARCH_FLAGS = -march=native
else
    ARCH_FLAGS = -march=x86-64-v3
endif

ifeq ($(COMPILER),gcc)
    CC = gcc
else ifeq ($(COMPILER),clang)
    CC = clang
else
    $(error Unknown COMPILER=$(COMPILER). Use gcc or clang)
endif

CFLAGS = -O3 $(ARCH_FLAGS) -mavx2 -mfma -mf16c -ffast-math
CFLAGS += -funroll-loops -fomit-frame-pointer -Wall -Wextra -fopenmp
LDFLAGS = -lm -lpthread -fopenmp

ifdef DEBUG
    CFLAGS = -O0 -g $(ARCH_FLAGS) -mavx2 -mfma -mf16c -Wall -Wextra -fopenmp
    CFLAGS += -fsanitize=address -fsanitize=undefined
    LDFLAGS += -fsanitize=address -fsanitize=undefined
endif

HEADERS = $(wildcard $(COMMON)/*.h) sam3d_runner.h
RUNNER_SRC = sam3d_runner.c

VERIFIES = verify_dinov2 verify_cond_fuser verify_ss_dit verify_flow_solver \
           verify_ss_decoder verify_slat_dit verify_slat_gs

.PHONY: all clean
all: test_sam3d $(VERIFIES)

test_sam3d: test_sam3d.c $(RUNNER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ test_sam3d.c $(RUNNER_SRC) $(LDFLAGS)

verify_dinov2: verify_dinov2.c $(RUNNER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_dinov2.c $(RUNNER_SRC) $(LDFLAGS)

verify_cond_fuser: verify_cond_fuser.c $(RUNNER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_cond_fuser.c $(RUNNER_SRC) $(LDFLAGS)

verify_ss_dit: verify_ss_dit.c $(RUNNER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_ss_dit.c $(RUNNER_SRC) $(LDFLAGS)

verify_flow_solver: verify_flow_solver.c $(COMMON)/sam3d_shortcut_solver.h
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_flow_solver.c -lm

verify_ss_decoder: verify_ss_decoder.c $(RUNNER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_ss_decoder.c $(RUNNER_SRC) $(LDFLAGS)

verify_slat_dit: verify_slat_dit.c $(RUNNER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_slat_dit.c $(RUNNER_SRC) $(LDFLAGS)

verify_slat_gs: verify_slat_gs.c $(RUNNER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_slat_gs.c $(RUNNER_SRC) $(LDFLAGS)

# Convenience targets for running verifies against a ref dump.
CKPT   ?= $(MODELS)/sam3d/pipeline.yaml
REFDIR ?= /tmp/sam3d_ref

verify-dinov2: verify_dinov2
	./verify_dinov2 --ckpt $(CKPT) --refdir $(REFDIR)
verify-cond-fuser: verify_cond_fuser
	./verify_cond_fuser --ckpt $(CKPT) --refdir $(REFDIR)
verify-ss-dit: verify_ss_dit
	./verify_ss_dit --ckpt $(CKPT) --refdir $(REFDIR)
verify-flow-solver: verify_flow_solver
	./verify_flow_solver --refdir $(REFDIR)
verify-ss-decoder: verify_ss_decoder
	./verify_ss_decoder --ckpt $(CKPT) --refdir $(REFDIR)
verify-slat-dit: verify_slat_dit
	./verify_slat_dit --ckpt $(CKPT) --refdir $(REFDIR)
verify-slat-gs: verify_slat_gs
	./verify_slat_gs --ckpt $(CKPT) --refdir $(REFDIR)

clean:
	rm -f test_sam3d $(VERIFIES)
