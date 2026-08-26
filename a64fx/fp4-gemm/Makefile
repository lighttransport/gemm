CC = fcc
CFLAGS ?= -Nclang -O3 -std=c11 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp
WARN = -Wall -Wextra -Wpedantic
LDLIBS = -lm
ASM = fp4_n32_l1.S fp4_dequant_sve.S fp4_m1_t4.S fp4_m1_t8.S fp4_m1_t8_half.S fp4_m1_t8_affine.S fp4_m1_t12.S fp4_sdot_m1.S fp4_sdot4_m1.S fp4_pair_lut_m1.S fp4_pair_tbl_m1.S

.PHONY: all test fapp clean
all: test_fp4_gemm bench_fp4_gemm bench_fp4_single bench_fp4_ex bench_fp4_stream bench_fp4_sdot_error

test_fp4_gemm: test_fp4_gemm.c fp4_gemm.c fp4_gemm.h $(ASM)
	$(CC) $(CFLAGS) $(WARN) test_fp4_gemm.c fp4_gemm.c $(ASM) $(LDLIBS) -o $@

bench_fp4_gemm: bench_fp4_gemm.c fp4_gemm.c fp4_gemm.h $(ASM)
	$(CC) $(CFLAGS) $(WARN) bench_fp4_gemm.c fp4_gemm.c $(ASM) $(LDLIBS) -o $@

bench_fp4_single: bench_fp4_single.c fp4_gemm.c fp4_gemm.h $(ASM)
	$(CC) $(CFLAGS) $(WARN) bench_fp4_single.c fp4_gemm.c $(ASM) $(LDLIBS) -o $@

bench_fp4_ex: bench_fp4_ex.c fp4_ex_only.S
	$(CC) $(CFLAGS) $(WARN) bench_fp4_ex.c fp4_ex_only.S -o $@

bench_fp4_stream: bench_fp4_stream.c fp4_gemm.c fp4_gemm.h $(ASM)
	$(CC) $(CFLAGS) $(WARN) bench_fp4_stream.c fp4_gemm.c $(ASM) $(LDLIBS) -o $@

bench_fp4_sdot_error: bench_fp4_sdot_error.c fp4_gemm.c fp4_gemm.h $(ASM)
	$(CC) $(CFLAGS) $(WARN) bench_fp4_sdot_error.c fp4_gemm.c $(ASM) $(LDLIBS) -o $@

bench_fp4_fapp: bench_fp4_fapp.c fp4_gemm.c fp4_gemm.h $(ASM)
	$(CC) $(CFLAGS) $(WARN) -DUSE_FAPP bench_fp4_fapp.c fp4_gemm.c $(ASM) $(LDLIBS) -o $@

fapp: bench_fp4_fapp

test: test_fp4_gemm
	./test_fp4_gemm

clean:
	rm -f test_fp4_gemm bench_fp4_gemm bench_fp4_single bench_fp4_ex bench_fp4_stream bench_fp4_sdot_error bench_fp4_fapp
