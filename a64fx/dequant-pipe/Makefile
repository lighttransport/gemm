CC = fcc
CFLAGS ?= -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -Wall -Wextra -Wpedantic -pthread
LDFLAGS ?= -pthread -lm -lhwb

TARGETS := bench_dequant_pipe bench_null_pipe bench_hbm_read bench_hbm_color bench_fused_sdot

.PHONY: all clean test

all: $(TARGETS)

bench_dequant_pipe: bench_dequant_pipe.o kernels.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_null_pipe: bench_null_pipe.o kernels.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_hbm_read: bench_hbm_read.o kernels.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_hbm_color: bench_hbm_color.o kernels.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_fused_sdot: bench_fused_sdot.o kernels.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

bench_dequant_pipe.o: bench_dequant_pipe.c hwb_compat.h
	$(CC) $(CFLAGS) -c $< -o $@

bench_null_pipe.o: bench_null_pipe.c hwb_compat.h
	$(CC) $(CFLAGS) -c $< -o $@

bench_hbm_read.o: bench_hbm_read.c
	$(CC) $(CFLAGS) -c $< -o $@

bench_hbm_color.o: bench_hbm_color.c
	$(CC) $(CFLAGS) -c $< -o $@

bench_fused_sdot.o: bench_fused_sdot.c
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.S
	$(CC) -Nclang -march=armv8.2-a+sve -c $< -o $@

test: bench_dequant_pipe
	./bench_dequant_pipe --self-test

clean:
	rm -f $(TARGETS) bench_dequant_pipe.o bench_null_pipe.o bench_hbm_read.o bench_hbm_color.o bench_fused_sdot.o kernels.o
