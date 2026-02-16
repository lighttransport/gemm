# FP8 Quantization Kernels Makefile for A64FX
# Using fcc native compiler (running on A64FX)

CC = fcc
CFLAGS = -O3 -march=armv8.2-a+sve -Kopenmp -Kfast

# For better optimization
CFLAGS += -Kparallel -Koptmsg=2

# Alternative with clang (if available)
# CC = clang
# CFLAGS = -O3 -march=armv8.2-a+sve -fopenmp

LDFLAGS = -lm

.PHONY: all clean run compare bench

all: bench_fp8 bench_opt bench_compare bench_all

bench_fp8: fp8_quant.c bench_fp8.c fp8_quant.h
	$(CC) $(CFLAGS) -o $@ fp8_quant.c bench_fp8.c $(LDFLAGS)

bench_opt: fp8_quant.c fp8_quant_opt.c bench_opt.c fp8_quant.h fp8_quant_opt.h
	$(CC) $(CFLAGS) -o $@ fp8_quant.c fp8_quant_opt.c bench_opt.c $(LDFLAGS)

bench_compare: fp8_quant.c fp8_quant_opt.c fp8_quant_opt8.c bench_compare.c fp8_quant.h fp8_quant_opt.h fp8_quant_opt8.h
	$(CC) $(CFLAGS) -o $@ fp8_quant.c fp8_quant_opt.c fp8_quant_opt8.c bench_compare.c $(LDFLAGS)

bench_all: fp8_quant.c fp8_quant_opt.c fp8_quant_opt8.c fp8_quant_swp.c bench_all.c fp8_quant.h fp8_quant_opt.h fp8_quant_opt8.h fp8_quant_swp.h
	$(CC) $(CFLAGS) -o $@ fp8_quant.c fp8_quant_opt.c fp8_quant_opt8.c fp8_quant_swp.c bench_all.c $(LDFLAGS)

clean:
	rm -f bench_fp8 bench_opt bench_compare *.o *.s

run: bench_fp8
	./bench_fp8

compare: bench_compare
	./bench_compare

# Debug build
debug: CFLAGS = -O0 -g -march=armv8.2-a+sve
debug: bench_fp8

# Show generated assembly for the SVE kernel
asm: fp8_quant.c fp8_quant.h
	$(CC) $(CFLAGS) -S -o fp8_quant.s fp8_quant.c
	@echo "Assembly saved to fp8_quant.s"
