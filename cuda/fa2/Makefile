# FlashAttention-2 (bf16/fp16/fp8) for sm_120 (Blackwell GeForce, RTX 5060 Ti).
#
# Follows the gau-nernst "fa-5090" recipe (Ampere-class: mma.sync.m16n8k16,
# cp.async, ldmatrix, XOR swizzle). Host driver compiles with gcc + cuew (no
# nvcc); device kernels are NVRTC-compiled at runtime.

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I.. -I../../common -mavx2 -mfma -march=native
LDFLAGS = -ldl -lm -lpthread

SRCS = bench_cuda_fa2.c ../cuew.c ../cublasew.c
DEPS = cuda_fa2_kernels.h ../cuew.h ../cublasew.h ../cuda_runner_common.h

TARGET = bench_cuda_fa2

.PHONY: all clean smoke run-bf16 run-fp8

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

smoke: $(TARGET)
	./$(TARGET) --dtype bf16 --mode all --shape qwen3_512 --causal 0
	./$(TARGET) --dtype bf16 --mode mma --shape qwen3_512 --causal 1

run-bf16: $(TARGET)
	./$(TARGET) --dtype bf16 --mode all --shape all --causal 0

run-fp8: $(TARGET)
	./$(TARGET) --dtype fp8 --mode all --shape all --causal 0

clean:
	rm -f $(TARGET)
