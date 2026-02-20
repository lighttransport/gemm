# Makefile for FP8 GEMM using cuew
#
# No CUDA SDK required - uses cuew to dynamically load CUDA driver API

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm

# Source files
SRCS = fp8_gemm.c ../cuew.c
TARGET = fp8_gemm
TARGET_CUBLAS = cublas_fp8_gemm
TARGET_TEST_TYPES = test_fp8_types

.PHONY: all clean run help cublas test-unit

all: $(TARGET)

$(TARGET): $(SRCS) fp8_types.h ../cuew.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

# cuBLAS FP8 GEMM (requires libcublasLt.so)
$(TARGET_CUBLAS): cublas_fp8_gemm.c ../cuew.c ../cuew.h
	$(CC) $(CFLAGS) -o $@ cublas_fp8_gemm.c ../cuew.c $(LDFLAGS)

cublas: $(TARGET_CUBLAS)
	./$(TARGET_CUBLAS)

# cuBLAS BF16 GEMM
cublas_bf16_gemm: cublas_bf16_gemm.c ../cuew.c ../cuew.h
	$(CC) $(CFLAGS) -o $@ cublas_bf16_gemm.c ../cuew.c $(LDFLAGS)

bf16: cublas_bf16_gemm
	./cublas_bf16_gemm

# cuBLAS FP16 GEMM
cublas_fp16_gemm: cublas_fp16_gemm.c ../cuew.c ../cuew.h
	$(CC) $(CFLAGS) -o $@ cublas_fp16_gemm.c ../cuew.c $(LDFLAGS)

fp16: cublas_fp16_gemm
	./cublas_fp16_gemm

# Unit test (CPU-only, no GPU required)
$(TARGET_TEST_TYPES): test_fp8_types.c fp8_types.h
	$(CC) $(CFLAGS) -o $@ test_fp8_types.c -lm

test-unit: $(TARGET_TEST_TYPES)
	./$(TARGET_TEST_TYPES)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET_TEST_TYPES)

help:
	@echo "FP8 GEMM using cuew (no CUDA SDK required)"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build fp8_gemm"
	@echo "  make run      - Build and run"
	@echo "  make clean    - Remove built files"
	@echo ""
	@echo "Features:"
	@echo "  - FP8 E4M3 and E5M2 formats"
	@echo "  - 128x128x128 matrix multiplication"
	@echo "  - FP32 accumulator"
	@echo "  - PTX kernel loaded via cuModuleLoadData"
