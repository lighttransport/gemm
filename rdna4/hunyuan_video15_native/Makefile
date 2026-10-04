ROOT := ../..
NATIVE := $(ROOT)/cuda/hunyuan_video15_native
BUILD ?= $(ROOT)/tmp/video-rocm/hv15-build
export TMPDIR := $(abspath $(BUILD))
CXX ?= g++
CC ?= gcc
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS += -DHV15N_ROCM
LDLIBS := -ldl -lm -lpcre2-8 -pthread
SOURCES := loader encoders dit vae hv15_native
OBJECTS := $(addprefix $(BUILD)/,$(addsuffix .o,$(SOURCES))) $(BUILD)/gpu_hip.o $(BUILD)/rocew.o
HEADERS := ../video_common/aotriton_bridge.h ../video_common/kernels.hpp $(wildcard $(NATIVE)/*.hpp) $(NATIVE)/hv15_native.h $(wildcard ../video_common/*.hpp) ../rocew.h $(ROOT)/common/safetensors.h $(ROOT)/common/stb_image.h
.PHONY: all test test-gpu compile-kernels clean
all: $(BUILD)/hv15n_rocm $(BUILD)/libhv15n_rocm.so $(BUILD)/component_probe $(BUILD)/tokenizer_probe
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: $(NATIVE)/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -c $< -o $@
$(BUILD)/gpu_hip.o: gpu_hip.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -c $< -o $@
$(BUILD)/rocew.o: ../rocew.c ../rocew.h | $(BUILD)
	$(CC) -O2 -Wall -Wextra -fPIC -c $< -o $@
$(BUILD)/hv15n_rocm: $(NATIVE)/runner.cpp $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/libhv15n_rocm.so: $(OBJECTS)
	$(CXX) -shared $^ $(LDLIBS) -o $@
$(BUILD)/component_probe: $(NATIVE)/component_probe.cpp $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/tokenizer_probe: $(NATIVE)/tokenizer_probe.cpp $(BUILD)/loader.o $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/loader.o -lpcre2-8 -o $@
$(BUILD)/test_host: $(NATIVE)/test_host.cpp $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDLIBS) -o $@
test: all $(BUILD)/test_host
	$(BUILD)/test_host
compile-kernels: all
	$(BUILD)/hv15n_rocm --compile-kernels
clean:
	rm -f $(OBJECTS) $(BUILD)/hv15n_rocm $(BUILD)/libhv15n_rocm.so $(BUILD)/test_host $(BUILD)/test_gpu $(BUILD)/test_int8 $(BUILD)/bench_f16 $(BUILD)/test_aotriton $(BUILD)/libvideo_aotriton.so $(BUILD)/component_probe $(BUILD)/tokenizer_probe

../video_common/kernels.hpp: ../video_common/make_kernels.py ../video_common/gemm.hip ../video_common/attention.hip ../video_common/flex_attention.hip ../video_common/precision.hip ../qimg/hip_qimg_kernels.h
	python3 ../video_common/make_kernels.py $@
$(BUILD)/test_gpu: $(NATIVE)/test_gpu.cpp $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/test_int8: ../video_common/test_gpu.cpp ../video_common/test_norm_boundary.hpp ../minimax_h3/kernels.hpp $(BUILD)/gpu_hip.o $(BUILD)/loader.o $(BUILD)/rocew.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/gpu_hip.o $(BUILD)/loader.o $(BUILD)/rocew.o $(LDLIBS) -o $@
test-gpu: all $(BUILD)/test_gpu $(BUILD)/test_int8
	$(BUILD)/test_gpu repo-only
	$(BUILD)/test_int8

$(BUILD)/bench_f16: ../video_common/bench_f16.cpp $(BUILD)/gpu_hip.o $(BUILD)/loader.o $(BUILD)/rocew.o $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/gpu_hip.o $(BUILD)/loader.o $(BUILD)/rocew.o $(LDLIBS) -o $@
.PHONY: bench-f16
bench-f16: $(BUILD)/bench_f16
	$(BUILD)/bench_f16

AOTRITON_ROOT ?= /opt/aotriton
.PHONY: aotriton
aotriton: $(BUILD)/libvideo_aotriton.so
$(BUILD)/libvideo_aotriton.so: ../video_common/aotriton_bridge.cpp ../video_common/aotriton_bridge.h | $(BUILD)
	test -f "$(AOTRITON_ROOT)/include/aotriton/flash.h"
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -I"$(AOTRITON_ROOT)/include" -shared -fPIC $< -L"$(AOTRITON_ROOT)/lib" -laotriton_v2 -L/opt/rocm/lib -Wl,--no-as-needed -lamdhip64 -Wl,--as-needed -Wl,-rpath,"$(AOTRITON_ROOT)/lib:/opt/rocm/lib" -o $@

$(BUILD)/test_aotriton: ../video_common/test_aotriton.cpp $(BUILD)/gpu_hip.o $(BUILD)/loader.o $(BUILD)/rocew.o $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/gpu_hip.o $(BUILD)/loader.o $(BUILD)/rocew.o $(LDLIBS) -o $@
.PHONY: test-aotriton
test-aotriton: $(BUILD)/test_aotriton
	$(BUILD)/test_aotriton $(AOTRITON_BRIDGE)
