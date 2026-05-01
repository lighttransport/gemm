# Makefile for RDNA4 GGML dequantization benchmark.
#
# Kernels are compiled at runtime via HIPRTC, matching the other rdna4 runners.

CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -I.. -I../.. -march=native
LDFLAGS ?= -ldl -lm -lpthread

ROCEW_SRC = ../rocew.c
TARGET = bench_dequant

.PHONY: all clean run

all: $(TARGET)

$(TARGET): bench_dequant.c $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h ../../common/ggml_dequant.h ../../common/gguf_loader.h
	$(CC) $(CFLAGS) -o $@ bench_dequant.c $(ROCEW_SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) --verify --bench --type all --mode all --rows 1024 --cols 1024 --tokens 16 --iters 20

clean:
	rm -f $(TARGET)
