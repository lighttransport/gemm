# Virtual-human face deformer on CUDA (driver API via cuew, NVRTC kernels; no nvcc).
#   make && ./bench_vhuman_deformer_cuda <head>/rig/rig_deformer.safetensors [frames]
CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I.. -I../../common -mavx2 -mfma
LDFLAGS = -ldl -lm -lpthread

SRCS = vhuman_deformer_cuda.c ../cuew.c ../../ryzen/vhuman_deformer.c ../../ryzen/lightrig_mlp2.c ../../ryzen/gemm_avx2.c
DEPS = vhuman_deformer_cuda.h ../../ryzen/vhuman_deformer.h ../cuew.h ../cuda_runner_common.h

.PHONY: all clean

all: bench_vhuman_deformer_cuda libvhuman_deformer_cuda.so

bench_vhuman_deformer_cuda: bench_vhuman_deformer_cuda.c $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) -o $@ bench_vhuman_deformer_cuda.c $(SRCS) $(LDFLAGS)

libvhuman_deformer_cuda.so: $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f bench_vhuman_deformer_cuda libvhuman_deformer_cuda.so
