# Makefile for A64FX Sector Cache Test

# Use fcc for native A64FX execution
CC = fcc
CFLAGS = -Nclang -Ofast -march=armv8.2-a+sve -ffj-ocl -ffj-no-largepage
LDFLAGS = -lm

TARGET = bench_sector_cache

SRCS = bench_sector_cache.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -rf fapp_output fapp_pa1 fapp_pa2
