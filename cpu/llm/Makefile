# Makefile for CPU LLM inference runners
#
# Targets:
#   test_transformer          - Single-process inference
#   test_transformer_parallel - Distributed parallel inference (custom comm)
#   test_transformer_mpi      - Distributed parallel inference (MPI backend)
#
# Usage:
#   make                    - Build all (except MPI)
#   make ARCH=native        - Build for current CPU
#   make COMPILER=clang     - Build with Clang
#   make DEBUG=1            - Debug build with sanitizers
#   make mpi                - Build MPI variant
#   make clean              - Remove built files

COMPILER ?= gcc
ARCH ?= zen2

COMMON = ../../common

# Architecture flags
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

# Compiler selection
ifeq ($(COMPILER),gcc)
    CC = gcc
    CFLAGS = -O2 $(ARCH_FLAGS) -mavx2 -mfma -mf16c
    CFLAGS += -Wall -Wextra
else ifeq ($(COMPILER),clang)
    CC = clang
    CFLAGS = -O2 $(ARCH_FLAGS) -mavx2 -mfma -mf16c
    CFLAGS += -Wall -Wextra
else
    $(error Unknown COMPILER=$(COMPILER). Use gcc or clang)
endif

LDFLAGS = -lm -lpthread

# Debug build
ifdef DEBUG
    CFLAGS = -O0 -g $(ARCH_FLAGS) -mavx2 -mfma -mf16c
    CFLAGS += -Wall -Wextra
    CFLAGS += -fsanitize=address -fsanitize=undefined
    LDFLAGS += -fsanitize=address -fsanitize=undefined
endif

HEADERS = $(wildcard $(COMMON)/*.h)

.PHONY: all mpi clean

all: test_transformer test_transformer_parallel

test_transformer: test_transformer.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_transformer_parallel: test_transformer_parallel.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

mpi: test_transformer_parallel.c $(HEADERS)
	mpicc $(CFLAGS) -DUSE_MPI -o test_transformer_mpi $< $(LDFLAGS)

clean:
	rm -f test_transformer test_transformer_parallel test_transformer_mpi
