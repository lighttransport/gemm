# Makefile for FP16 GEMM micro-kernels (A64FX SVE)

# Compilers
CC_FCC = fccpx
CC_GCC = gcc

# Default compiler
COMPILER ?= fccpx

ifeq ($(COMPILER),fccpx)
    CC = $(CC_FCC)
    CFLAGS = -Nclang -O3 -std=c11 -march=armv8.2-a+sve -ffp-contract=fast
    ASFLAGS = -Nclang -c
    LDFLAGS = -Nclang
else ifeq ($(COMPILER),gcc)
    CC = $(CC_GCC)
    CFLAGS = -O3 -std=c11 -march=armv8.2-a+sve -ffp-contract=fast
    ASFLAGS = -c
    LDFLAGS =
endif

TARGET = bench_fp16_gemm
OBJS = fp16_gemm.o fp16_kernel_5x4.o fp16_kernel_6x4.o

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

fp16_gemm.o: fp16_gemm.c
	$(CC) $(CFLAGS) -c -o $@ $<

fp16_kernel_5x4.o: fp16_kernel_5x4.S
	$(CC) $(ASFLAGS) -o $@ $<

fp16_kernel_6x4.o: fp16_kernel_6x4.S
	$(CC) $(ASFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) $(OBJS)

run: $(TARGET)
	./$(TARGET)
