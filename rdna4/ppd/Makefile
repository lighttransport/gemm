# Makefile for HIP/ROCm PPD (Pixel-Perfect-Depth) runner
#
# No hipcc required - kernels compiled at runtime via HIPRTC.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm -lpthread

ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_ppd_runner.c
TEST_SRC = test_hip_ppd.c

TARGET = test_hip_ppd

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) hip_ppd_runner.h ../rocew.h \
           ../hip_kernels_common.h ../hip_runner_common.h \
           ../../common/pth_loader.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
