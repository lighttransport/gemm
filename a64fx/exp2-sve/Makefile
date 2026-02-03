# Makefile for exp2 FEXPA kernel

# Use native fcc compiler
CC = fcc
CFLAGS = -Ofast -Kfast,sve -Kopenmp
LDFLAGS = -Kopenmp -lm

# For clang (alternative)
# CC = clang-21
# CFLAGS = -O3 -march=armv8.2-a+sve -Wall -fopenmp
# LDFLAGS = -fopenmp -lm

TARGETS = bench_exp2

all: $(TARGETS)

bench_exp2: bench_exp2.c exp2_fexpa.S exp2_fexpa.h
	$(CC) $(CFLAGS) -o $@ bench_exp2.c exp2_fexpa.S $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
