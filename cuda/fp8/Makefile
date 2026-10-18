# Makefile for FP8 GEMM using cuew
#
# No CUDA SDK required - uses cuew to dynamically load CUDA driver API

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm
FP8_TMP_DIR ?= $(abspath ../../tmp/fp8)
export TMPDIR := $(FP8_TMP_DIR)
FP8_BENCH_ARGS ?= --shape all

# Source files
SRCS = fp8_gemm.c ../cuew.c
TARGET = fp8_gemm
TARGET_CUBLAS = cublas_fp8_gemm
TARGET_TEST_TYPES = test_fp8_types

.PHONY: all clean run help cublas test-unit bench test sanitize

all: $(TARGET) build/bench_fp8

$(TARGET): $(SRCS) fp8_types.h fp8_gemm_kernels.h ../cuew.h | $(FP8_TMP_DIR)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

# cuBLAS FP8 GEMM (requires libcublasLt.so)
$(TARGET_CUBLAS): cublas_fp8_gemm.c fp8_types.h ../cuew.c ../cuew.h | $(FP8_TMP_DIR)
	$(CC) $(CFLAGS) -o $@ cublas_fp8_gemm.c ../cuew.c $(LDFLAGS)

cublas: $(TARGET_CUBLAS)
	./$(TARGET_CUBLAS)

# cuBLAS BF16 GEMM
cublas_bf16_gemm: cublas_bf16_gemm.c ../cuew.c ../cuew.h | $(FP8_TMP_DIR)
	$(CC) $(CFLAGS) -o $@ cublas_bf16_gemm.c ../cuew.c $(LDFLAGS)

bf16: cublas_bf16_gemm
	./cublas_bf16_gemm

# cuBLAS FP16 GEMM
cublas_fp16_gemm: cublas_fp16_gemm.c ../cuew.c ../cuew.h | $(FP8_TMP_DIR)
	$(CC) $(CFLAGS) -o $@ cublas_fp16_gemm.c ../cuew.c $(LDFLAGS)

fp16: cublas_fp16_gemm
	./cublas_fp16_gemm

# Unit test (CPU-only, no GPU required)
$(TARGET_TEST_TYPES): test_fp8_types.c fp8_types.h | $(FP8_TMP_DIR)
	$(CC) $(CFLAGS) -o $@ test_fp8_types.c -lm

test-unit: $(TARGET_TEST_TYPES)
	./$(TARGET_TEST_TYPES)

run: $(TARGET)
	./$(TARGET)

build $(FP8_TMP_DIR):
	mkdir -p $@

build/bench_fp8: bench_fp8.c fp8_types.h fp8_gemm_kernels.h ../gemm/cuda_gemm_ptx_kernels.h ../int4/bench_common.h ../int4/int4.h ../int4/int4_numeric.h ../cuew.c ../cuew.h ../cublasew.c ../cublasew.h | build $(FP8_TMP_DIR)
	$(CC) $(CFLAGS) -Wno-unused-const-variable -o $@ bench_fp8.c ../cuew.c ../cublasew.c $(LDFLAGS)

bench: build/bench_fp8
	./build/bench_fp8 $(FP8_BENCH_ARGS) --csv build/fp8_compare.csv

test: test-unit $(TARGET) $(TARGET_CUBLAS) build/bench_fp8
	./build/bench_fp8 --m 128 --n 256 --k 128 --full-check --warmup 1 --iters 2 --rounds 1
	./build/bench_fp8 --m 3 --n 136 --k 64 --kernel ptx --full-check --warmup 1 --iters 2 --rounds 1
	./$(TARGET_CUBLAS) -m 128 -n 256 -k 128
	./$(TARGET) --tile 128 -m 128 -n 256 -k 128 -i 2
	./$(TARGET) --naive -m 3 -n 7 -k 17 -i 2

sanitize: build/bench_fp8 $(TARGET)
	compute-sanitizer --tool memcheck --error-exitcode 1 ./build/bench_fp8 --m 128 --n 256 --k 128 --full-check --warmup 1 --iters 1 --rounds 1
	compute-sanitizer --tool memcheck --error-exitcode 1 ./build/bench_fp8 --m 3 --n 136 --k 64 --kernel ptx --full-check --warmup 1 --iters 1 --rounds 1
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(TARGET) --naive -m 3 -n 7 -k 17 -i 1
	compute-sanitizer --tool racecheck --error-exitcode 1 ./build/bench_fp8 --m 128 --n 128 --k 64 --full-check --warmup 1 --iters 1 --rounds 1
	compute-sanitizer --tool racecheck --error-exitcode 1 ./build/bench_fp8 --m 3 --n 136 --k 64 --kernel ptx --full-check --warmup 1 --iters 1 --rounds 1

clean:
	rm -f $(TARGET) $(TARGET_TEST_TYPES) $(TARGET_CUBLAS) cublas_bf16_gemm cublas_fp16_gemm
	rm -rf build

help:
	@echo "FP8 GEMM using cuew (no CUDA SDK required)"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build fp8_gemm"
	@echo "  make run      - Build and run"
	@echo "  make test     - CPU conversions and exhaustive small GPU GEMMs"
	@echo "  make bench    - Verified repeated FP8/BF16 shape sweep, CSV in build/"
	@echo "  make sanitize - Compute Sanitizer memory and race checks"
	@echo "  make clean    - Remove built files"
	@echo ""
	@echo "Features:"
	@echo "  - FP8 E4M3 and E5M2 formats"
	@echo "  - 128x128x128 matrix multiplication"
	@echo "  - FP32 accumulator"
	@echo "  - PTX kernel loaded via cuModuleLoadData"
