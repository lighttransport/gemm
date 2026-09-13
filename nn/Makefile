CC ?= cc
.DEFAULT_GOAL := all
CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Wpedantic
BUILD ?= build
HIPBLASLT ?= 0
ROCM_PATH ?= /opt/rocm/core
HIPBLASLT_INCLUDE ?= $(ROCM_PATH)/include
HIPBLASLT_LIB ?= $(ROCM_PATH)/lib
OBJECTS := $(BUILD)/gn.o $(BUILD)/gn_cpu.o $(BUILD)/gn_gpu.o $(BUILD)/cuew.o $(BUILD)/rocew.o
ifeq ($(HIPBLASLT),1)
OBJECTS := $(filter-out $(BUILD)/gn_gpu.o,$(OBJECTS)) $(BUILD)/gn_gpu_lt.o $(BUILD)/gn_hipblaslt.o
LDLIBS += -L$(HIPBLASLT_LIB) -Wl,-rpath,$(HIPBLASLT_LIB) -lhipblaslt -lamdhip64 -lstdc++
endif
.PHONY: all check clean FORCE
FORCE:
$(BUILD)/backend.config: FORCE | $(BUILD)
	@python3 -B build_config.py $@ '$(HIPBLASLT)' '$(HIPBLASLT_INCLUDE)' '$(HIPBLASLT_LIB)'
all: $(BUILD)/test_gn $(BUILD)/libgn.so $(BUILD)/gn_tool $(BUILD)/test_gpu
$(BUILD):
	mkdir -p $@
$(BUILD)/gn.o: gn.c gn.h gn_internal.h gn_cpu.h ../common/safetensors.h ../common/safetensors_writer.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -c gn.c -o $@
$(BUILD)/gn_kernels.inc: gn_kernels.cu gn_sm120.cuh gn_rdna4.cuh embed_kernels.py | $(BUILD)
	python3 -B embed_kernels.py $< $@
$(BUILD)/gn_gpu.o: gn_gpu.c gn.h gn_internal.h $(BUILD)/gn_kernels.inc $(BUILD)/backend.config
	$(CC) $(CFLAGS) -Wno-overlength-strings -I$(BUILD) -fPIC -c $< -o $@
$(BUILD)/gn_gpu_lt.o: gn_gpu.c gn.h gn_internal.h gn_hipblaslt.h $(BUILD)/gn_kernels.inc $(BUILD)/backend.config
	$(CC) $(CFLAGS) -DGN_HIPBLASLT=1 -Wno-overlength-strings -I$(BUILD) -fPIC -c $< -o $@
$(BUILD)/gn_cpu.o: gn_cpu.c gn_cpu.h gn.h gn_internal.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -pthread -c $< -o $@
$(BUILD)/cuew.o: ../cuda/cuew.c | $(BUILD)
	$(CC) -O2 -fPIC -c $< -o $@
$(BUILD)/rocew.o: ../rdna4/rocew.c | $(BUILD)
	$(CC) -O2 -fPIC -c $< -o $@
$(BUILD)/gn_hipblaslt.o: gn_hipblaslt.cpp gn_hipblaslt.h $(BUILD)/backend.config | $(BUILD)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -fPIC -D__HIP_PLATFORM_AMD__ -I$(ROCM_PATH)/include -I$(HIPBLASLT_INCLUDE) -c $< -o $@
$(BUILD)/libgn.so: $(OBJECTS)
	$(CC) -shared $^ -lm -ldl -pthread $(LDLIBS) -o $@
$(BUILD)/test_gn: test_gn.c $(OBJECTS)
	$(CC) $(CFLAGS) $^ -lm -ldl -pthread $(LDLIBS) -o $@
$(BUILD)/gn_tool: gn_tool.c gn_replay.h $(OBJECTS)
	$(CC) $(CFLAGS) gn_tool.c $(OBJECTS) -lm -ldl -pthread $(LDLIBS) -o $@
$(BUILD)/test_gpu: test_gpu.c $(OBJECTS)
	$(CC) $(CFLAGS) $^ -lm -ldl -pthread $(LDLIBS) -o $@
check: all
	$(BUILD)/test_gn $(BUILD)/roundtrip.safetensors
.PHONY: sm120-bench
sm120-bench: $(BUILD)/bench_sm120
$(BUILD)/bench_sm120: bench_sm120.cu gn_kernels.cu gn_sm120.cuh gn_rdna4.cuh | $(BUILD)
	nvcc -O3 -std=c++17 -arch=sm_120 -Xptxas=-v $< -o $@
.PHONY: rdna4-bench
rdna4-bench: $(BUILD)/bench_rdna4
$(BUILD)/bench_rdna4: bench_rdna4.cpp gn_kernels.cu gn_sm120.cuh gn_rdna4.cuh $(BUILD)/backend.config $(if $(filter 1,$(HIPBLASLT)),$(BUILD)/gn_hipblaslt.o) | $(BUILD)
	TMPDIR=$(abspath $(BUILD)) hipcc -O3 -std=c++17 --offload-arch=gfx1201 $(if $(filter 1,$(HIPBLASLT)),-DGN_HIPBLASLT=1) $< $(if $(filter 1,$(HIPBLASLT)),-x none $(BUILD)/gn_hipblaslt.o) -Wl,-rpath,$(ROCM_PATH)/lib $(LDLIBS) -o $@
clean:
	$(RM) $(OBJECTS) $(BUILD)/gn_kernels.inc $(BUILD)/libgn.so $(BUILD)/test_gn $(BUILD)/test_gpu $(BUILD)/gn_tool $(BUILD)/roundtrip.safetensors
