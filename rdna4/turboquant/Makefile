CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -Wno-unused-function -Wno-format-truncation -std=c11 -I.. -I../.. -I../../cpu/turboquant -march=native
LDFLAGS ?= -ldl -lm -lpthread

ROCEW_SRC = ../rocew.c
CPU_TQ_SRC = ../../cpu/turboquant/turboquant_cpu.c ../../cpu/turboquant/turboquant_cpu_x86.c
TARGET = bench_turboquant

.PHONY: all clean run

all: $(TARGET)

$(TARGET): bench_turboquant.c $(ROCEW_SRC) $(CPU_TQ_SRC) ../rocew.h ../hip_runner_common.h ../../cpu/turboquant/turboquant_cpu.h
	$(CC) $(CFLAGS) -o $@ bench_turboquant.c $(ROCEW_SRC) $(CPU_TQ_SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) --verify --rows 1024 --cols 128 --iters 200

clean:
	rm -f $(TARGET)
