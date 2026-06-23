CC ?= gcc
NVCC ?= /usr/local/cuda/bin/nvcc

CFLAGS ?= -O3 -march=native -Wall -Wextra -Wpedantic
CPPFLAGS += -I../../common
LDFLAGS += -lm

TARGETS = quant_analyze

all: $(TARGETS)

quant_analyze: src/quant_analyze.c ../../common/safetensors.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ src/quant_analyze.c $(LDFLAGS)

cuda: int8_dp4a_bench int16_cuda_bench

int8_dp4a_bench: src/int8_dp4a_bench.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "error: $(NVCC) not found; load CUDA or set NVCC=/path/to/nvcc"; exit 127; }
	$(NVCC) -O3 -arch=sm_70 -lineinfo -o $@ $<

int16_cuda_bench: src/int16_cuda_bench.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "error: $(NVCC) not found; load CUDA or set NVCC=/path/to/nvcc"; exit 127; }
	$(NVCC) -O3 -arch=sm_70 -lineinfo -o $@ $<

clean:
	rm -f quant_analyze int8_dp4a_bench int16_cuda_bench

.PHONY: all cuda clean
