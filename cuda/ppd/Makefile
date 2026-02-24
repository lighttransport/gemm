# Makefile for CUDA PPD (Pixel-Perfect-Depth) runner
#
# No nvcc required - kernels compiled at runtime via NVRTC.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC = ../cuew.c
RUNNER_SRC = cuda_ppd_runner.c
TEST_SRC = test_cuda_ppd.c

TARGET = test_cuda_ppd

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) cuda_ppd_runner.h ../cuew.h \
           ../../common/pth_loader.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
