# Makefile for unit tests
#
# Usage:
#   make                    - Build all tests
#   make ARCH=native        - Build for current CPU
#   make run                - Build and run all tests
#   make clean              - Remove built files

COMPILER ?= gcc
ARCH ?= native

COMMON = ../common

# Architecture flags
ifeq ($(ARCH),native)
    ARCH_FLAGS = -march=native
else ifeq ($(ARCH),zen2)
    ARCH_FLAGS = -march=znver2 -mtune=znver2
else
    ARCH_FLAGS = -march=x86-64-v3
endif

# Compiler selection
ifeq ($(COMPILER),gcc)
    CC = gcc
else ifeq ($(COMPILER),clang)
    CC = clang
else
    $(error Unknown COMPILER=$(COMPILER). Use gcc or clang)
endif

CFLAGS = -O2 $(ARCH_FLAGS) -Wall -Wextra
LDFLAGS = -lm

HEADERS = $(wildcard $(COMMON)/*.h)
TARGETS = test_bpe_tokenizer test_gguf_loader test_safetensors

.PHONY: all run clean

all: $(TARGETS)

test_bpe_tokenizer: test_bpe_tokenizer.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_gguf_loader: test_gguf_loader.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_safetensors: test_safetensors.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGETS)
	@echo "=== Running unit tests ==="
	@for t in $(TARGETS); do echo "--- $$t ---"; ./$$t || exit 1; done

clean:
	rm -f $(TARGETS)
