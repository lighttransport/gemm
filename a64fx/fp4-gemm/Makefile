CC = fcc
CFLAGS ?= -Nclang -O3 -std=c11 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp
WARN = -Wall -Wextra -Wpedantic
LDLIBS = -lm

.PHONY: all test clean
all: test_fp4_gemm bench_fp4_gemm

test_fp4_gemm: test_fp4_gemm.c fp4_gemm.c fp4_gemm.h
	$(CC) $(CFLAGS) $(WARN) test_fp4_gemm.c fp4_gemm.c $(LDLIBS) -o $@

bench_fp4_gemm: bench_fp4_gemm.c fp4_gemm.c fp4_gemm.h
	$(CC) $(CFLAGS) $(WARN) bench_fp4_gemm.c fp4_gemm.c $(LDLIBS) -o $@

test: test_fp4_gemm
	./test_fp4_gemm

clean:
	rm -f test_fp4_gemm bench_fp4_gemm
