COMPILER ?= gcc
CC := $(COMPILER)

CFLAGS ?= -O3 -Wall -Wextra -Wpedantic -std=c11
LDFLAGS ?= -lm

OBJS = turboquant_cpu.o turboquant_cpu_x86.o
TARGET = test_turboquant_cpu

.PHONY: all clean run

all: $(TARGET)

turboquant_cpu.o: turboquant_cpu.c turboquant_cpu.h turboquant_cpu_internal.h
	$(CC) $(CFLAGS) -c -o $@ $<

turboquant_cpu_x86.o: turboquant_cpu_x86.c turboquant_cpu.h turboquant_cpu_internal.h
	$(CC) $(CFLAGS) -msse2 -c -o $@ $<

$(TARGET): test_turboquant_cpu.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ test_turboquant_cpu.c $(OBJS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) --verify --n 4096 --iters 2000

clean:
	rm -f $(TARGET) $(OBJS)
