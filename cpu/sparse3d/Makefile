# Makefile for sparse3d unit tests

COMPILER ?= gcc
ARCH ?= zen2

COMMON = ../../common

ifeq ($(ARCH),zen2)
    ARCH_FLAGS = -march=znver2 -mtune=znver2
else ifeq ($(ARCH),native)
    ARCH_FLAGS = -march=native
else
    ARCH_FLAGS = -march=x86-64-v3
endif

ifeq ($(COMPILER),clang)
    CC = clang
else
    CC = gcc
endif

CFLAGS = -O3 $(ARCH_FLAGS) -mavx2 -mfma -mf16c
CFLAGS += -funroll-loops -fomit-frame-pointer
CFLAGS += -Wall -Wextra
LDFLAGS = -lm -lpthread

ifdef DEBUG
    CFLAGS = -O0 -g $(ARCH_FLAGS) -mavx2 -mfma -mf16c
    CFLAGS += -Wall -Wextra
    CFLAGS += -fsanitize=address -fsanitize=undefined
    LDFLAGS += -fsanitize=address -fsanitize=undefined
endif

HEADERS = $(wildcard $(COMMON)/*.h)

.PHONY: all clean test

all: test_sparse3d

test_sparse3d: test_sparse3d.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -o $@ $< $(LDFLAGS)

test: test_sparse3d
	./test_sparse3d

clean:
	rm -f test_sparse3d
