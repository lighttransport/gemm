# Makefile for CUDA utilities and tests
#
# CPU-only tests (no GPU required)

CC = gcc
CFLAGS = -O3 -Wall -Wextra
LDFLAGS = -lm

TARGET_PTX_VALIDATE = test_ptx_validate

.PHONY: all clean test-unit test-all

all: $(TARGET_PTX_VALIDATE)

$(TARGET_PTX_VALIDATE): test_ptx_validate.c ptx_validate.h
	$(CC) $(CFLAGS) -o $@ test_ptx_validate.c $(LDFLAGS)

test-unit: $(TARGET_PTX_VALIDATE)
	./$(TARGET_PTX_VALIDATE)

# Run all CPU-only unit tests across subdirectories
test-all: test-unit
	$(MAKE) -C int8 test-unit
	$(MAKE) -C fp8 test-unit

clean:
	rm -f $(TARGET_PTX_VALIDATE)
