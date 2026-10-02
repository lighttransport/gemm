# Virtual-human face deformer on CUDA (driver API via cuew, NVRTC kernels; no nvcc).
#   make && ./bench_vhuman_deformer_cuda <head>/rig/rig_deformer.safetensors [frames]
CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I.. -I../../common -mavx2 -mfma
LDFLAGS = -ldl -lm -lpthread

SRCS = vhuman_deformer_cuda.c ../cuew.c ../../ryzen/vhuman_deformer.c ../../ryzen/lightrig_mlp2.c ../../ryzen/gemm_avx2.c
DEPS = vhuman_deformer_cuda.h ../../ryzen/vhuman_deformer.h ../cuew.h ../cuda_runner_common.h

.PHONY: all clean test-training

all: bench_vhuman_deformer_cuda libvhuman_deformer_cuda.so vhuman_models libvhuman_runtime.so libvhuman_splat.so libvhuman_training_cuda.so

bench_vhuman_deformer_cuda: bench_vhuman_deformer_cuda.c $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) -o $@ bench_vhuman_deformer_cuda.c $(SRCS) $(LDFLAGS)

libvhuman_deformer_cuda.so: $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f bench_vhuman_deformer_cuda libvhuman_deformer_cuda.so vhuman_models gemm_avx2.o libvhuman_runtime.so libvhuman_splat.so splat_cuew.o
	rm -f libvhuman_training_cuda.so test_vhuman_training_cuda

# Native RMBG2, MoGe2 camera and reconstruction cue models.
gemm_avx2.o: ../../ryzen/gemm_avx2.c ../../ryzen/gemm_avx2.h
	$(CC) -O3 -mavx2 -mfma -fPIC -c $< -o $@

vhuman_models: ../../common/vhuman_models.c $(wildcard ../../common/*.h) ../../common/swin_runner.c gemm_avx2.o ../cuew.c ../gemm/cuda_linear_f32.h ../gemm/cuda_gemm_f32_kernels.h
	$(CC) $(CFLAGS) -fopenmp -mf16c -DSWIN_HAVE_AVX2 -DSWIN_CUDA -Wno-unused-function -o $@ $< gemm_avx2.o ../cuew.c $(LDFLAGS)

libvhuman_runtime.so: vhuman_runtime.c vhuman_runtime.h ../cuew.c ../cuew.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ vhuman_runtime.c ../cuew.c $(LDFLAGS)

splat_cuew.o: ../cuew.c ../cuew.h
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

libvhuman_splat.so: vhuman_splat.cpp vhuman_splat_kernels.h splat_cuew.o
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wpedantic -fPIC -shared -o $@ vhuman_splat.cpp splat_cuew.o $(LDFLAGS)

libvhuman_training_cuda.so: vhuman_training.cpp vhuman_training.h vhuman_training_kernels.h ../gemm/cuda_gemm_f32_kernels.h splat_cuew.o
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wpedantic -fPIC -shared -o $@ vhuman_training.cpp splat_cuew.o $(LDFLAGS)

test_vhuman_training_cuda: test_vhuman_training.c vhuman_training.h libvhuman_training_cuda.so
	$(CC) -O3 -std=c11 -Wall -Wextra -Wpedantic -o $@ $< -L. -lvhuman_training_cuda -Wl,-rpath,'$$ORIGIN' -lm

test-training: test_vhuman_training_cuda
	./test_vhuman_training_cuda
