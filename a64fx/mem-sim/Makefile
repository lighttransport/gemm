# Makefile for A64FX Memory Access Simulator

CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

TARGET = memsim
OBJS = memsim_main.o memsim.o

# Example files
EXAMPLES = example_sequential.txt \
           example_gemm_d256.txt \
           example_gemm_d512.txt \
           example_gemm_d512_prefetch.txt \
           example_zfill.txt

.PHONY: all clean test run_all

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

memsim_main.o: memsim_main.c memsim.h
	$(CC) $(CFLAGS) -c $<

memsim.o: memsim.c memsim.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJS)

# Run all examples
test: $(TARGET)
	@echo "Running all examples..."
	@echo ""
	@echo "=========================================="
	@echo "Example 1: Sequential Access"
	@echo "=========================================="
	./$(TARGET) example_sequential.txt
	@echo ""
	@echo "=========================================="
	@echo "Example 2: GEMM D=256 (Good Cache)"
	@echo "=========================================="
	./$(TARGET) example_gemm_d256.txt
	@echo ""
	@echo "=========================================="
	@echo "Example 3: GEMM D=512 (Cache Pressure)"
	@echo "=========================================="
	./$(TARGET) example_gemm_d512.txt
	@echo ""
	@echo "=========================================="
	@echo "Example 4: GEMM D=512 with Prefetching"
	@echo "=========================================="
	./$(TARGET) example_gemm_d512_prefetch.txt
	@echo ""
	@echo "=========================================="
	@echo "Example 5: Zero-fill Optimization"
	@echo "=========================================="
	./$(TARGET) example_zfill.txt

# Run individual examples
run_sequential: $(TARGET)
	./$(TARGET) example_sequential.txt

run_d256: $(TARGET)
	./$(TARGET) example_gemm_d256.txt

run_d512: $(TARGET)
	./$(TARGET) example_gemm_d512.txt

run_d512_prefetch: $(TARGET)
	./$(TARGET) example_gemm_d512_prefetch.txt

run_zfill: $(TARGET)
	./$(TARGET) example_zfill.txt

# Help
help:
	@echo "A64FX Memory Access Simulator"
	@echo ""
	@echo "Targets:"
	@echo "  all                - Build simulator"
	@echo "  clean              - Remove build artifacts"
	@echo "  test               - Run all examples"
	@echo "  run_sequential     - Run sequential access example"
	@echo "  run_d256           - Run GEMM D=256 example"
	@echo "  run_d512           - Run GEMM D=512 example"
	@echo "  run_d512_prefetch  - Run GEMM D=512 with prefetching"
	@echo ""
	@echo "Usage:"
	@echo "  ./memsim <input_file>"
	@echo ""
	@echo "Example:"
	@echo "  ./memsim example_sequential.txt"
