# Makefile for L1D Cache Set Partitioning Benchmark

CC = fcc
CFLAGS = -Nclang -Ofast -march=armv8.2-a+sve -ffj-ocl -ffj-no-largepage
LDFLAGS = -lm

TARGET = bench_cache_partition
SRC = bench_cache_partition.c

.PHONY: all fapp clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

fapp: $(SRC)
	$(CC) $(CFLAGS) -DUSE_FAPP -o $(TARGET) $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -rf fapp_sc1 fapp_sc2 fapp_*_result.*
