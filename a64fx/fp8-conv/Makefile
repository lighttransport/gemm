# Makefile for FP8 conversion and GEMM benchmark on A64FX

CC = fcc
CFLAGS = -Ofast -Kfast,openmp -Nclang

# Targets
all: bench_fp8_convert bench_fp8_gemm bench_opt_convert

bench_fp8_convert: bench_fp8_convert.c fp8_convert.c fp8_convert.h
	$(CC) $(CFLAGS) -o $@ bench_fp8_convert.c fp8_convert.c

bench_fp8_gemm: bench_fp8_gemm.c fp8_gemm.c fp8_gemm.h fp8_convert.c fp8_convert.h fp8_kernel_asm.S
	$(CC) $(CFLAGS) -o $@ bench_fp8_gemm.c fp8_gemm.c fp8_convert.c fp8_kernel_asm.S

bench_opt_convert: bench_opt_convert.c fp8_kernel_asm.S
	$(CC) $(CFLAGS) -o $@ bench_opt_convert.c fp8_kernel_asm.S

test_kernel: test_kernel.c fp8_kernel_asm.S
	$(CC) $(CFLAGS) -o $@ test_kernel.c fp8_kernel_asm.S

bench_full_cost: bench_full_cost.c fp8_kernel_asm.S
	$(CC) $(CFLAGS) -o $@ bench_full_cost.c fp8_kernel_asm.S

clean:
	rm -f bench_fp8_convert bench_fp8_gemm bench_opt_convert test_kernel bench_full_cost *.o

run_convert: bench_fp8_convert
	./bench_fp8_convert

run_gemm: bench_fp8_gemm
	./bench_fp8_gemm

.PHONY: all clean run_convert run_gemm
