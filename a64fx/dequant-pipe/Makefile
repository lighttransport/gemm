CC = fcc
CFLAGS ?= -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall -Wextra -Wpedantic -pthread
LDFLAGS ?= -pthread -lm -lhwb

TARGETS := test_e4_pack test_q8_k128 bench_q8_k128 bench_w8 bench_dequant_pipe bench_null_pipe bench_hbm_read bench_hbm_color bench_fused_sdot

.PHONY: all clean test

all: $(TARGETS)

bench_dequant_pipe: bench_dequant_pipe.o kernels.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_null_pipe: bench_null_pipe.o kernels.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_hbm_read: bench_hbm_read.o kernels.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_hbm_color: bench_hbm_color.o kernels.o q8_k128.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_fused_sdot: bench_fused_sdot.o kernels.o kernels_opt.o fused_opt.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_dequant_pipe.o: bench_dequant_pipe.c hwb_compat.h
	$(CC) $(CFLAGS) -c $< -o $@

bench_null_pipe.o: bench_null_pipe.c hwb_compat.h
	$(CC) $(CFLAGS) -c $< -o $@

bench_hbm_read.o: bench_hbm_read.c
	$(CC) $(CFLAGS) -c $< -o $@

bench_hbm_color.o: bench_hbm_color.c
	$(CC) $(CFLAGS) -c $< -o $@

bench_fused_sdot.o: bench_fused_sdot.c fused_opt.h
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.S
	$(CC) -Nclang -march=armv8.2-a+sve -c $< -o $@

kernels_opt.o: kernels_opt.S
	$(CC) -Nclang -march=armv8.2-a+sve -c $< -o $@

fused_opt.o: fused_opt.c fused_opt.h
	$(CC) $(CFLAGS) -c $< -o $@

test: bench_dequant_pipe bench_fused_sdot bench_w8 test_e4_pack test_q8_k128
	./test_e4_pack
	./test_q8_k128
	./bench_w8 --verify --cores 1 --mib 1 --iterations 1 --trials 1
	./bench_dequant_pipe --self-test
	./bench_fused_sdot --verify --format int4 --path int16 --kernel super \
		--cores 1 --mib 1 --iterations 1 --trials 1

clean:
	rm -f $(TARGETS) bench_w8.o w8.o q8_k128.o e4_pack.o e4_pack_c.o bench_dequant_pipe.o bench_null_pipe.o bench_hbm_read.o bench_hbm_color.o bench_fused_sdot.o kernels.o kernels_opt.o fused_opt.o

test_q8_k128: test_q8_k128.c q8_k128.h q8_k128.o
	$(CC) $(CFLAGS) test_q8_k128.c q8_k128.o $(LDFLAGS) -o $@

bench_q8_k128: bench_q8_k128.c q8_k128.h q8_k128.o
	$(CC) $(CFLAGS) bench_q8_k128.c q8_k128.o $(LDFLAGS) -o $@

q8_k128.o: q8_k128.S q8_k128.h
	$(CC) -Nclang -march=armv8.2-a+sve -c $< -o $@

bench_w8: bench_w8.o e4_pack.o e4_pack_c.o w8.o fused_opt.o kernels_opt.o kernels.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_w8.o: bench_w8.c w8.h fused_opt.h e4_pack.h
	$(CC) $(CFLAGS) -c $< -o $@

w8.o: w8.S
	$(CC) -Nclang -march=armv8.2-a+sve -c $< -o $@

e4_pack.o: e4_pack.S
	$(CC) -Nclang -march=armv8.2-a+sve -c $< -o $@

e4_pack_c.o: e4_pack.c e4_pack.h w8.h
	$(CC) $(CFLAGS) -c $< -o $@

test_e4_pack: test_e4_pack.c e4_pack.h w8.h e4_pack.o e4_pack_c.o w8.o
	$(CC) $(CFLAGS) test_e4_pack.c e4_pack.o e4_pack_c.o w8.o $(LDFLAGS) -o $@

# Opt-in A15 activation quantization correctness gate.
test_q8_k128_a15: test_q8_k128_a15.c q8_k128.h q8_k128.o
	$(CC) $(CFLAGS) test_q8_k128_a15.c q8_k128.o $(LDFLAGS) -o $@
