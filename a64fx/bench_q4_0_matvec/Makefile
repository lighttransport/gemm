# Makefile for Q4_0 matvec benchmark on A64FX
#
# Build (cross-compile):
#   make                    # Build with default flags
#   make ONTHEFLY=1         # Also enable on-the-fly int8 path
#
# Run on A64FX:
#   ./bench_q4_0_matvec
#
# Native build on A64FX:
#   gcc-14 -O3 -march=armv8.2-a+sve -ffp-contract=fast -D_GNU_SOURCE \
#       -I../../common bench_q4_0_matvec.c -lm -o bench_q4_0_matvec

# Cross-compiler for A64FX
CC      = aarch64-linux-gnu-gcc-14
CFLAGS  = -O3 -static -march=armv8.2-a+sve -ffp-contract=fast -D_GNU_SOURCE -Wno-unused-variable
LDFLAGS = -lm -lpthread
TARGET  = bench_q4_0_matvec

# Set ONTHEFLY=1 to enable the on-the-fly int8 path (slower due to dequant in hot loop)
ifeq ($(ONTHEFLY),1)
CFLAGS += -DTF_USE_INT8_SDOT_Q4_0
endif

# Set NATIVE=1 to build with the host compiler (for testing on x86 with qemu)
ifeq ($(NATIVE),1)
CC      = gcc
CFLAGS  = -O3 -D_GNU_SOURCE -Wall
LDFLAGS = -lm
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): bench_q4_0_matvec.c
	$(CC) $(CFLAGS) -I../../common $< $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)
