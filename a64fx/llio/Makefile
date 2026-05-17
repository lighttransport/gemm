# LLIO localtmp file I/O benchmarks.
#
#   make             - native build with fcc (run directly on A64FX node)
#   make cross       - cross-compile from login node with fccpx
#   make COMPILER=gcc

COMPILER ?= fcc

ifeq ($(COMPILER),fcc)
    CC = fcc
    CFLAGS = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
else ifeq ($(COMPILER),fccpx)
    CC = fccpx
    CFLAGS = -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
else
    CC = $(COMPILER)
    CFLAGS = -O3 -march=armv8.2-a+sve -ffp-contract=fast
endif

LDFLAGS =

BINS = bench_fopen bench_mmap

all: $(BINS)

cross:
	$(MAKE) COMPILER=fccpx

bench_fopen: bench_fopen.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

bench_mmap: bench_mmap.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINS)

.PHONY: all cross clean
