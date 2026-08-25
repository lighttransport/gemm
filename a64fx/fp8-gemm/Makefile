CC = fcc
CFLAGS ?= -Nclang -O3 -std=c11 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp
WARN = -Wall -Wextra -Wpedantic
.PHONY: all test clean
all: test_fp8_gemm bench_fp8_stream
test_fp8_gemm: test_fp8_gemm.c fp8_gemm.c fp8_gemm.h fp8_m1.S
	$(CC) $(CFLAGS) $(WARN) test_fp8_gemm.c fp8_gemm.c fp8_m1.S -lm -o $@
bench_fp8_stream: bench_fp8_stream.c fp8_gemm.c fp8_gemm.h fp8_m1.S
	$(CC) $(CFLAGS) $(WARN) bench_fp8_stream.c fp8_gemm.c fp8_m1.S -lm -o $@
test: test_fp8_gemm
	./test_fp8_gemm
clean:
	rm -f test_fp8_gemm bench_fp8_stream
