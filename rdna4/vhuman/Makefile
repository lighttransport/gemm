CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall -Wextra -fopenmp -mavx2 -mfma
export TMPDIR := $(abspath ../../tmp/vhuman-hip)
all: bench_vhuman_deformer_hip libvhuman_deformer_hip.so
bench_vhuman_deformer_hip: vhuman_deformer_hip.c ../../cuda/vhuman/vhuman_deformer_cuda.c ../../rdna4/cuda_driver_compat.h ../../ryzen/vhuman_deformer.c ../../ryzen/vhuman_deformer.h ../../cuda/vhuman/bench_vhuman_deformer_cuda.c ../rocew.c | $(TMPDIR)
	$(CC) $(CFLAGS) -o $@ ../../cuda/vhuman/bench_vhuman_deformer_cuda.c vhuman_deformer_hip.c ../../ryzen/vhuman_deformer.c ../../ryzen/lightrig_mlp2.c ../../ryzen/gemm_avx2.c ../rocew.c -ldl -lm
$(TMPDIR):
	mkdir -p $@
clean:
	rm -f bench_vhuman_deformer_hip libvhuman_deformer_hip.so
.PHONY: all clean

libvhuman_deformer_hip.so: vhuman_deformer_hip.c ../../cuda/vhuman/vhuman_deformer_cuda.c ../../rdna4/cuda_driver_compat.h ../../ryzen/vhuman_deformer.c | $(TMPDIR)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ vhuman_deformer_hip.c ../../ryzen/vhuman_deformer.c ../../ryzen/lightrig_mlp2.c ../../ryzen/gemm_avx2.c ../rocew.c -ldl -lm
