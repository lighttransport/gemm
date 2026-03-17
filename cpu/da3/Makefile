# Makefile for CPU Depth-Anything-3 depth estimation
#
# Usage:
#   make                    - Build for Zen2
#   make ARCH=native        - Build for current CPU
#   make COMPILER=clang     - Build with Clang
#   make DEBUG=1            - Debug build with sanitizers
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
    CFLAGS = -O3 $(ARCH_FLAGS) -mavx2 -mfma -mf16c -ffast-math
    CFLAGS += -funroll-loops -fomit-frame-pointer
    CFLAGS += -Wall -Wextra
else ifeq ($(COMPILER),clang)
    CC = clang
    CFLAGS = -O3 $(ARCH_FLAGS) -mavx2 -mfma -mf16c -ffast-math
    CFLAGS += -funroll-loops -fomit-frame-pointer
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

# Profile build
ifdef PROFILE
    CFLAGS += -pg -g
    LDFLAGS += -pg
endif

HEADERS = $(wildcard $(COMMON)/*.h)

.PHONY: all clean

all: test_da3

test_da3: test_da3.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f test_da3
