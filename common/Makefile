# Makefile for CPU-only DA3 depth estimation
#
# Targets AMD Zen2 (znver2) with AVX2/FMA/F16C by default.
# Also supports generic x86-64 and native builds.
#
# Usage:
#   make                    - Build for Zen2
#   make ARCH=native        - Build for current CPU
#   make ARCH=generic       - Build for generic x86-64 (AVX2)
#   make COMPILER=clang     - Build with Clang
#   make DEBUG=1            - Debug build with sanitizers
#   make run                - Show usage
#   make clean              - Remove built files

COMPILER ?= gcc
ARCH ?= zen2

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

# Target
TARGET = test_da3

HEADERS = gguf_loader.h ggml_dequant.h safetensors.h depth_anything3.h

.PHONY: all clean run help

all: $(TARGET)

$(TARGET): test_da3.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ test_da3.c $(LDFLAGS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <da3.gguf|model.safetensors> [-t threads] [-i input.ppm] [-o output.pgm]"
	@echo "       ./$(TARGET) model.safetensors -i photo.ppm --full -t 4"

clean:
	rm -f $(TARGET)

help:
	@echo "CPU-only DA3 Depth Estimation (GGUF + Safetensors)"
	@echo "==================================================="
	@echo ""
	@echo "Build:"
	@echo "  make                    - Build for Zen2 (default)"
	@echo "  make ARCH=native        - Build for current CPU"
	@echo "  make ARCH=zen3          - Build for Zen3"
	@echo "  make ARCH=zen4          - Build for Zen4"
	@echo "  make ARCH=generic       - Build for generic x86-64 (AVX2)"
	@echo "  make COMPILER=clang     - Build with Clang"
	@echo "  make DEBUG=1            - Debug build with sanitizers"
	@echo "  make PROFILE=1          - Build with profiling"
	@echo "  make clean              - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_da3 <da3.gguf|model.safetensors> [-t threads] [-i input.ppm] [-o output.pgm]"
	@echo "  ./test_da3 model.safetensors -i photo.ppm --full -t 4"
	@echo ""
	@echo "Options:"
	@echo "  -t <threads>    Number of threads (default: 4)"
	@echo "  -i <input.ppm>  Input image (default: synthetic gradient)"
	@echo "  -o <output>     Output file (.pgm)"
	@echo "  -c <config>     config.json path (auto-detected if next to .safetensors)"
	@echo "  --full          Run all output modalities"
	@echo "  --pose          Enable pose output"
	@echo "  --rays          Enable rays output"
	@echo "  --gaussians     Enable gaussians output"
	@echo ""
	@echo "Architecture: Pure CPU with AVX2/FMA/F16C GEMM + multi-threaded backbone"
	@echo ""
	@echo "Requirements:"
	@echo "  - x86-64 CPU with AVX2 + FMA + F16C (Zen2+, Haswell+)"
