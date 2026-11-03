#===============
# options
#===============

CUDA_PATH ?= /usr/local/cuda
GPU_ARCH ?= auto
GEMMUL8_PATH ?= ..
CFLAGS ?=
EXE ?=
RUNNER ?=

#===============
# CUDA setup
#===============

ifeq ($(GPU_ARCH),auto)
override GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
endif

export PATH := $(CUDA_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

COMPILER := nvcc
FLAGS := -std=c++20 -O3 -diag-suppress 177 -I$(GEMMUL8_PATH)/include
LIBS := -lcublas -lcudart -lcublasLt -lcuda -lnvidia-ml -ldl $(GEMMUL8_PATH)/lib/libgemmul8.a
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

#===============
# Files
#===============

CU_SRC := $(wildcard *.cu)
CU_OBJ := $(CU_SRC:.cu=.o)
TARGETS := $(CU_SRC:.cu=)
HEADERS := $(wildcard *.hpp)

.PHONY: all clean run run_% print_targets FORCE

all: $(TARGETS)

%: %.o
	$(COMPILER) $(FLAGS) $(CFLAGS) $(ARCH) $< -o $@ $(LIBS)

%.o: %.cu $(HEADERS)
	$(COMPILER) $(FLAGS) $(CFLAGS) $(ARCH) -c $< -o $@

run:
	@if [ -z "$(EXE)" ]; then \
	    echo "Usage: make run EXE=<executable>"; \
	    exit 1; \
	fi
	$(RUNNER) ./$(EXE)

run_%: FORCE
	@set -e; \
	pattern="$*"; \
	targets="$(filter test_$*_%,$(TARGETS))"; \
	if [ -z "$$targets" ]; then \
	    echo "No executable matching 'test_$$pattern_*'."; \
	    echo "Available targets: $(TARGETS)"; \
	    exit 1; \
	fi; \
	echo "Running: $$targets"; \
	for exe in $$targets; do \
	    echo ""; \
	    echo "============================================================"; \
	    echo "Running $$exe"; \
	    echo "============================================================"; \
	    $(MAKE) --no-print-directory $$exe; \
	    $(RUNNER) ./$$exe; \
	done

print_targets:
	@echo "CU_SRC  = $(CU_SRC)"
	@echo "TARGETS = $(TARGETS)"

FORCE:

clean:
	rm -f $(CU_OBJ) $(TARGETS)
