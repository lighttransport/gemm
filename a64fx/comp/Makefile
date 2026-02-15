CC      = fcc
CFLAGS  = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
LDFLAGS = -lm

SRCS_C  = lz_compress.c lz_decompress.c
SRCS_S  = lz_decompress_a64fx.S
HDRS    = lz.h lz_fmt.h lz_hash.h

all: bench_lz

bench_lz: bench_lz.c $(SRCS_C) $(SRCS_S) $(HDRS)
	$(CC) $(CFLAGS) -o $@ bench_lz.c $(SRCS_C) $(SRCS_S) $(LDFLAGS)

clean:
	rm -f bench_lz

.PHONY: all clean
