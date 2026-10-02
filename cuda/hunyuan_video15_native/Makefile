ROOT := ../..
BUILD ?= $(ROOT)/tmp/hv15-native/build
export TMPDIR := $(abspath $(BUILD))
export CUDA_CACHE_PATH := $(abspath $(BUILD))/cuda-cache
CXX ?= g++
CC ?= gcc
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CFLAGS ?= -O2 -Wall -Wextra
LDLIBS := -ldl -lm -lpcre2-8 -pthread
SOURCES := gpu.cpp loader.cpp encoders.cpp dit.cpp vae.cpp hv15_native.cpp
OBJECTS := $(addprefix $(BUILD)/,$(SOURCES:.cpp=.o)) $(BUILD)/cuew.o $(BUILD)/cublasew.o
HEADERS := $(wildcard *.hpp) hv15_native.h ../cuew.h ../cublasew.h ../gemm/cuda_gemm_ptx_kernels.h ../fa2/cuda_fa2_kernels.h $(ROOT)/common/safetensors.h $(ROOT)/common/stb_image.h
.PHONY: all test compile-kernels clean
all: $(BUILD)/hv15n $(BUILD)/libhv15n.so $(BUILD)/tokenizer_probe $(BUILD)/component_probe
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: %.cpp $(HEADERS) | $(BUILD)
	TMPDIR=$(abspath $(BUILD)) $(CXX) $(CXXFLAGS) -fPIC -c $< -o $@
$(BUILD)/cuew.o: ../cuew.c ../cuew.h | $(BUILD)
	TMPDIR=$(abspath $(BUILD)) $(CC) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/cublasew.o: ../cublasew.c ../cublasew.h | $(BUILD)
	TMPDIR=$(abspath $(BUILD)) $(CC) $(CFLAGS) -fPIC -c $< -o $@
$(BUILD)/hv15n: runner.cpp $(OBJECTS)
	TMPDIR=$(abspath $(BUILD)) $(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/libhv15n.so: $(OBJECTS)
	$(CXX) -shared $^ $(LDLIBS) -o $@
$(BUILD)/test_host: test_host.cpp $(OBJECTS)
	TMPDIR=$(abspath $(BUILD)) $(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/test_gpu: test_gpu.cpp $(OBJECTS)
	TMPDIR=$(abspath $(BUILD)) $(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/component_probe: component_probe.cpp $(OBJECTS)
	TMPDIR=$(abspath $(BUILD)) $(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/tokenizer_probe: tokenizer_probe.cpp $(BUILD)/loader.o tokenizer.hpp host.hpp
	TMPDIR=$(abspath $(BUILD)) $(CXX) $(CXXFLAGS) tokenizer_probe.cpp $(BUILD)/loader.o -lpcre2-8 -o $@
test: all $(BUILD)/test_host
	$(BUILD)/test_host
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p 'test_*.py'
compile-kernels: $(BUILD)/hv15n
	$(BUILD)/hv15n --compile-kernels
clean:
	rm -f $(OBJECTS) $(BUILD)/hv15n $(BUILD)/libhv15n.so $(BUILD)/test_host $(BUILD)/test_gpu $(BUILD)/component_probe $(BUILD)/tokenizer_probe $(BUILD)/replay_probe

$(BUILD)/replay_probe: replay_probe.cpp $(OBJECTS)
	TMPDIR=$(abspath $(BUILD)) $(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@
