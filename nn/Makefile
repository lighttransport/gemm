CC ?= cc
CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Wpedantic
BUILD ?= build
OBJECTS := $(BUILD)/gn.o $(BUILD)/gn_cpu.o $(BUILD)/gn_gpu.o $(BUILD)/cuew.o $(BUILD)/rocew.o
.PHONY: all check clean
all: $(BUILD)/test_gn $(BUILD)/libgn.so $(BUILD)/gn_tool $(BUILD)/test_gpu
$(BUILD):
	mkdir -p $@
$(BUILD)/gn.o: gn.c gn.h gn_internal.h gn_cpu.h ../common/safetensors.h ../common/safetensors_writer.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -c gn.c -o $@
$(BUILD)/gn_kernels.inc: gn_kernels.cu gn_sm120.cuh embed_kernels.py | $(BUILD)
	python3 -B embed_kernels.py $< $@
$(BUILD)/gn_gpu.o: gn_gpu.c gn.h gn_internal.h $(BUILD)/gn_kernels.inc
	$(CC) $(CFLAGS) -Wno-overlength-strings -I$(BUILD) -fPIC -c $< -o $@
$(BUILD)/gn_cpu.o: gn_cpu.c gn_cpu.h gn.h gn_internal.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -pthread -c $< -o $@
$(BUILD)/cuew.o: ../cuda/cuew.c | $(BUILD)
	$(CC) -O2 -fPIC -c $< -o $@
$(BUILD)/rocew.o: ../rdna4/rocew.c | $(BUILD)
	$(CC) -O2 -fPIC -c $< -o $@
$(BUILD)/libgn.so: $(OBJECTS)
	$(CC) -shared $^ -lm -ldl -pthread -o $@
$(BUILD)/test_gn: test_gn.c $(OBJECTS)
	$(CC) $(CFLAGS) $^ -lm -ldl -pthread -o $@
$(BUILD)/gn_tool: gn_tool.c gn_replay.h $(OBJECTS)
	$(CC) $(CFLAGS) gn_tool.c $(OBJECTS) -lm -ldl -pthread -o $@
$(BUILD)/test_gpu: test_gpu.c $(OBJECTS)
	$(CC) $(CFLAGS) $^ -lm -ldl -pthread -o $@
check: all
	$(BUILD)/test_gn $(BUILD)/roundtrip.safetensors
.PHONY: sm120-bench
sm120-bench: $(BUILD)/bench_sm120
$(BUILD)/bench_sm120: bench_sm120.cu gn_kernels.cu gn_sm120.cuh | $(BUILD)
	nvcc -O3 -std=c++17 -arch=sm_120 -Xptxas=-v $< -o $@
clean:
	$(RM) $(OBJECTS) $(BUILD)/gn_kernels.inc $(BUILD)/libgn.so $(BUILD)/test_gn $(BUILD)/test_gpu $(BUILD)/gn_tool $(BUILD)/roundtrip.safetensors
