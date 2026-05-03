# FlashAttention forward benchmark for sm_120 (Blackwell GeForce, RTX 5060 Ti)
#
# Mirrors cuda/gemm/. Host driver compiles with gcc + cuew (no nvcc required);
# device kernels are NVRTC-compiled at runtime.

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I.. -I../../common -mavx2 -mfma -march=native
LDFLAGS = -ldl -lm -lpthread

SRCS = bench_cuda_fa.c ../cuew.c ../cublasew.c
DEPS = cuda_fa_kernels.h ../cuew.h ../cublasew.h ../cuda_runner_common.h

TARGET = bench_cuda_fa

.PHONY: all clean smoke run-f16

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

smoke: $(TARGET)
	./$(TARGET) --dtype f16 --mode all --shape qwen3_512

run-f16: $(TARGET)
	./$(TARGET) --dtype f16 --mode all --shape all

clean:
	rm -f $(TARGET)
