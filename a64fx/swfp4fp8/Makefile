CC ?= fcc
CFLAGS ?= -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast
WARN = -Wall -Wextra -Wpedantic
OMP = -fopenmp
LDLIBS = -lm -lpthread
BUILD_TMP ?= $(CURDIR)/tmp
export TMPDIR = $(BUILD_TMP)

.PHONY: all test bench clean

all: test_swfp4fp8 bench_swfp4fp8

swfp4fp8.o: swfp4fp8.c swfp4fp8.h
	mkdir -p $(BUILD_TMP)
	$(CC) $(CFLAGS) $(WARN) $(OMP) -c $< -o $@

swfp4_sdot.o: swfp4_sdot.S
	$(CC) -Nclang -march=armv8.2-a+sve -c $< -o $@

test_swfp4fp8: test_swfp4fp8.c swfp4fp8.o swfp4_sdot.o
	mkdir -p $(BUILD_TMP)
	$(CC) $(CFLAGS) $(WARN) $(OMP) $^ $(LDLIBS) -o $@

bench_swfp4fp8: bench_swfp4fp8.c swfp4fp8.o swfp4_sdot.o
	mkdir -p $(BUILD_TMP)
	$(CC) $(CFLAGS) $(WARN) $(OMP) $^ $(LDLIBS) -o $@

test: test_swfp4fp8
	./test_swfp4fp8

bench: bench_swfp4fp8
	./bench_swfp4fp8 --quick

clean:
	rm -f swfp4fp8.o swfp4_sdot.o test_swfp4fp8 bench_swfp4fp8
