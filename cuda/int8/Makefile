# Makefile for INT8 GEMM, Attention, and FFN
#
# No CUDA SDK required for attention/ffn - uses cuew for gemm only

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm

# Headers
HEADERS = int8_types.h int8_approx.h

# Targets
TARGET_GEMM = int8_gemm
TARGET_ATTN = int8_attention
TARGET_FFN = int8_ffn
TARGET_CUBLAS = cublas_int8_gemm
TARGET_TEST_TYPES = test_int8_types
TARGET_TEST_APPROX = test_int8_approx

.PHONY: all clean run help bench sr attention ffn test test-unit cublas

all: $(TARGET_GEMM) $(TARGET_ATTN) $(TARGET_FFN)

# cuBLAS INT8 GEMM (requires libcublasLt.so)
$(TARGET_CUBLAS): cublas_int8_gemm.c ../cuew.c ../cuew.h
	$(CC) $(CFLAGS) -o $@ cublas_int8_gemm.c ../cuew.c $(LDFLAGS)

cublas: $(TARGET_CUBLAS)
	./$(TARGET_CUBLAS)

# INT8 GEMM (requires CUDA)
$(TARGET_GEMM): int8_gemm.c ../cuew.c $(HEADERS) ../cuew.h
	$(CC) $(CFLAGS) -o $@ int8_gemm.c ../cuew.c $(LDFLAGS)

# INT8 Flash Attention (CPU reference)
$(TARGET_ATTN): int8_attention.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ int8_attention.c -lm

# INT8 FFN (CPU reference)
$(TARGET_FFN): int8_ffn.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ int8_ffn.c -lm

# Unit tests (CPU-only, no GPU required)
$(TARGET_TEST_TYPES): test_int8_types.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ test_int8_types.c -lm

$(TARGET_TEST_APPROX): test_int8_approx.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ test_int8_approx.c -lm

# Run targets
run: $(TARGET_GEMM)
	./$(TARGET_GEMM)

bench: $(TARGET_GEMM)
	./$(TARGET_GEMM) --bench

sr: $(TARGET_GEMM)
	./$(TARGET_GEMM) --sr

attention: $(TARGET_ATTN)
	./$(TARGET_ATTN)

ffn: $(TARGET_FFN)
	./$(TARGET_FFN)

ffn-bench: $(TARGET_FFN)
	./$(TARGET_FFN) --bench

# Unit tests (CPU-only, no GPU required)
test-unit: $(TARGET_TEST_TYPES) $(TARGET_TEST_APPROX)
	@echo "=== Testing INT8 Types ==="
	./$(TARGET_TEST_TYPES)
	@echo ""
	@echo "=== Testing INT8 Approximations ==="
	./$(TARGET_TEST_APPROX)

# Test all components
test: $(TARGET_ATTN) $(TARGET_FFN) $(TARGET_TEST_TYPES) $(TARGET_TEST_APPROX)
	@echo "=== Testing INT8 Types ==="
	./$(TARGET_TEST_TYPES)
	@echo ""
	@echo "=== Testing INT8 Approximations ==="
	./$(TARGET_TEST_APPROX)
	@echo ""
	@echo "=== Testing INT8 Flash Attention ==="
	./$(TARGET_ATTN) --seq 32 --dim 32
	@echo ""
	@echo "=== Testing INT8 FFN ==="
	./$(TARGET_FFN) --seq 32 --hidden 64 --ffn 256
	@echo ""
	@echo "=== Activation Benchmark ==="
	./$(TARGET_FFN) --bench

clean:
	rm -f $(TARGET_GEMM) $(TARGET_ATTN) $(TARGET_FFN) $(TARGET_TEST_TYPES) $(TARGET_TEST_APPROX)

help:
	@echo "INT8 Neural Network Components"
	@echo "=============================="
	@echo ""
	@echo "Build targets:"
	@echo "  make              - Build all (gemm, attention, ffn)"
	@echo "  make int8_gemm    - Build INT8 GEMM only"
	@echo "  make int8_attention - Build INT8 Flash Attention"
	@echo "  make int8_ffn     - Build INT8 FFN"
	@echo ""
	@echo "Run targets:"
	@echo "  make run          - Run INT8 GEMM"
	@echo "  make bench        - Run GEMM benchmark"
	@echo "  make sr           - Run with stochastic rounding"
	@echo "  make attention    - Run Flash Attention test"
	@echo "  make ffn          - Run FFN test"
	@echo "  make ffn-bench    - Run activation benchmark"
	@echo "  make test         - Run all tests"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "INT8 GEMM Features:"
	@echo "  - Signed (s8) and Unsigned (u8) INT8"
	@echo "  - INT32 accumulator, Tensor Core MMA (SM 8.9+)"
	@echo "  - Stochastic rounding with xoroshiro128+"
	@echo ""
	@echo "INT8 Flash Attention Features:"
	@echo "  - Tiled attention with online softmax"
	@echo "  - Integer exp() approximation"
	@echo "  - Q8.8 fixed-point arithmetic"
	@echo ""
	@echo "INT8 FFN Features:"
	@echo "  - GELU, SiLU, ReLU activations (integer approx)"
	@echo "  - Standard and Gated (GLU) FFN variants"
	@echo "  - Polynomial and LUT-based approximations"
