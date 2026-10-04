# MiniMax H3 on CUDA: the shared RDNA4 H3 graph built on the native Hunyuan CUDA runtime.
ROOT := ../..
H3 := $(ROOT)/rdna4/minimax_h3
NATIVE := $(ROOT)/cuda/hunyuan_video15_native
BUILD ?= $(ROOT)/tmp/video-cuda/h3-build
export TMPDIR := $(abspath $(BUILD))
export CUDA_CACHE_PATH := $(abspath $(BUILD))/cuda-cache
CXX ?= g++
CC ?= gcc
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
# Hardware FP16 conversion for host-side VAE tile blending.
ifeq ($(shell uname -m),x86_64)
CXXFLAGS += -mf16c
endif
CFLAGS ?= -O2 -Wall -Wextra
LDLIBS := -ldl -lm -lpcre2-8 -pthread
COMMON := $(BUILD)/gpu.o $(BUILD)/loader.o $(BUILD)/cuew.o $(BUILD)/cublasew.o
HEADERS := $(wildcard $(NATIVE)/*.hpp) $(H3)/runtime.hpp $(H3)/kernels.hpp $(H3)/h3.h ../cuew.h ../cublasew.h ../fa2/cuda_fa2_kernels.h $(ROOT)/common/safetensors.h
.PHONY: all test clean
all: $(BUILD)/h3_cuda $(BUILD)/libh3_cuda.so $(BUILD)/component_probe
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: $(NATIVE)/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@
$(BUILD)/cuew.o: ../cuew.c ../cuew.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/cublasew.o: ../cublasew.c ../cublasew.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/h3.o: $(H3)/h3.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@
$(BUILD)/h3_cuda: $(H3)/runner.cpp $(BUILD)/h3.o $(COMMON)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/libh3_cuda.so: $(BUILD)/h3.o $(COMMON)
	$(CXX) -shared $^ $(LDLIBS) -o $@
$(BUILD)/component_probe: $(H3)/component_probe.cpp $(BUILD)/h3.o $(COMMON)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
test: all
	$(BUILD)/h3_cuda --validate --prompt 'A red ball rolling on a wooden table.'
	cd $(H3) && H3_TEST_LIBRARY=$(abspath $(BUILD)/libh3_cuda.so) PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_host test_captures test_streaming
clean:
	rm -f $(BUILD)/*.o $(BUILD)/h3_cuda $(BUILD)/libh3_cuda.so $(BUILD)/component_probe
