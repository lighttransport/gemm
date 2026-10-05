ROOT := ../..
BUILD ?= $(ROOT)/tmp/video-rocm/h3-build
export TMPDIR := $(abspath $(BUILD))
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS += -DHV15N_ROCM
LDLIBS := -ldl -lm -lpcre2-8 -pthread
COMMON := $(BUILD)/gpu_hip.o $(BUILD)/loader.o $(BUILD)/rocew.o
HEADERS := $(wildcard $(ROOT)/cuda/hunyuan_video15_native/*.hpp) ../video_common/kernels.hpp ../video_common/hip_platform.hpp ../rocew.h $(ROOT)/common/safetensors.h $(ROOT)/common/stb_image.h
HEADERS += ../video_common/aotriton_bridge.h
AOTRITON_ROOT ?= /opt/aotriton
.PHONY: all test clean aotriton
all: $(BUILD)/h3_rocm $(BUILD)/libh3_rocm.so $(BUILD)/component_probe
aotriton: $(BUILD)/libvideo_aotriton.so
$(BUILD)/libvideo_aotriton.so: ../video_common/aotriton_bridge.cpp ../video_common/aotriton_bridge.h blaslt.cpp | $(BUILD)
	test -f "$(AOTRITON_ROOT)/include/aotriton/flash.h"
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -I"$(AOTRITON_ROOT)/include" -shared -fPIC $< blaslt.cpp -L"$(AOTRITON_ROOT)/lib" -laotriton_v2 -L/opt/rocm/lib -lhipblaslt -Wl,--no-as-needed -lamdhip64 -Wl,--as-needed -Wl,-rpath,"$(AOTRITON_ROOT)/lib:/opt/rocm/lib" -o $@
$(BUILD):
	mkdir -p $@
../video_common/kernels.hpp: ../video_common/make_kernels.py ../video_common/gemm.hip ../video_common/attention.hip ../video_common/flex_attention.hip ../video_common/precision.hip ../qimg/hip_qimg_kernels.h
	python3 ../video_common/make_kernels.py $@
$(BUILD)/gpu_hip.o: ../hunyuan_video15_native/gpu_hip.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -c $< -o $@
$(BUILD)/loader.o: $(ROOT)/cuda/hunyuan_video15_native/loader.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -c $< -o $@
$(BUILD)/rocew.o: ../rocew.c ../rocew.h | $(BUILD)
	$(CC) -O2 -Wall -Wextra -fPIC -c $< -o $@
$(BUILD)/h3.o: h3.cpp runtime.hpp kernels.hpp h3.h ../video_common/kernels.hpp ../video_common/hip_platform.hpp $(wildcard $(ROOT)/cuda/hunyuan_video15_native/*.hpp) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -c $< -o $@
$(BUILD)/h3_rocm: runner.cpp $(BUILD)/h3.o $(COMMON)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDLIBS) -o $@
$(BUILD)/libh3_rocm.so: $(BUILD)/h3.o $(COMMON)
	$(CXX) -shared $^ $(LDLIBS) -o $@
test: all
	$(BUILD)/h3_rocm --validate --prompt 'A red ball rolling on a wooden table.'
	H3_TEST_LIBRARY=$(abspath $(BUILD)/libh3_rocm.so) PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -p 'test_*.py'
clean:
	rm -f $(BUILD)/h3.o $(COMMON) $(BUILD)/h3_rocm $(BUILD)/libh3_rocm.so $(BUILD)/component_probe $(BUILD)/libvideo_aotriton.so

$(BUILD)/component_probe: component_probe.cpp runtime.hpp kernels.hpp $(BUILD)/h3.o $(COMMON) | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/h3.o $(COMMON) $(LDLIBS) -o $@
