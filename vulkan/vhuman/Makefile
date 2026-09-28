# Virtual-human face deformer on Vulkan compute (vkew runtime loading).
#   make && ./bench_vhuman_deformer_vk <head>/rig/rig_deformer.safetensors [frames]
CC       = gcc
CXX      = g++
GLSLC   ?= glslc
CFLAGS   = -O3 -Wall -Wextra -mavx2 -mfma
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -I../deps -mavx2 -mfma
LDFLAGS  = -ldl -lm -lpthread

CSRCS  = ../../ryzen/vhuman_deformer.c ../../ryzen/lightrig_mlp2.c ../../ryzen/gemm_avx2.c
CXXSRC = vhuman_deformer_vk.cc ../deps/vulkan-runner.cc ../deps/vkew.cc

.PHONY: all clean

all: bench_vhuman_deformer_vk libvhuman_deformer_vk.so

vh_deform.spv.inc: shaders/vh_deform.comp
	$(GLSLC) -O --target-env=vulkan1.2 -mfmt=c -o $@ $<

bench_vhuman_deformer_vk: bench_vhuman_deformer_vk.cc $(CXXSRC) $(CSRCS) vh_deform.spv.inc vhuman_deformer_vk.h
	$(CXX) $(CXXFLAGS) -I. -c vhuman_deformer_vk.cc -o vhuman_deformer_vk.o
	$(CXX) $(CXXFLAGS) -c ../deps/vulkan-runner.cc -o vulkan-runner.o
	$(CXX) $(CXXFLAGS) -c ../deps/vkew.cc -o vkew.o
	$(CC) $(CFLAGS) -c ../../ryzen/vhuman_deformer.c -o vhuman_deformer.o
	$(CC) $(CFLAGS) -c ../../ryzen/lightrig_mlp2.c -o lightrig_mlp2.o
	$(CC) $(CFLAGS) -c ../../ryzen/gemm_avx2.c -o gemm_avx2.o
	$(CXX) $(CXXFLAGS) -o $@ bench_vhuman_deformer_vk.cc vhuman_deformer_vk.o vulkan-runner.o vkew.o \
		vhuman_deformer.o lightrig_mlp2.o gemm_avx2.o $(LDFLAGS)

libvhuman_deformer_vk.so: $(CXXSRC) $(CSRCS) vh_deform.spv.inc vhuman_deformer_vk.h
	$(CXX) $(CXXFLAGS) -fPIC -I. -c vhuman_deformer_vk.cc -o vhuman_deformer_vk.pic.o
	$(CXX) $(CXXFLAGS) -fPIC -c ../deps/vulkan-runner.cc -o vulkan-runner.pic.o
	$(CXX) $(CXXFLAGS) -fPIC -c ../deps/vkew.cc -o vkew.pic.o
	$(CC) $(CFLAGS) -fPIC -c ../../ryzen/vhuman_deformer.c -o vhuman_deformer.pic.o
	$(CC) $(CFLAGS) -fPIC -c ../../ryzen/lightrig_mlp2.c -o lightrig_mlp2.pic.o
	$(CC) $(CFLAGS) -fPIC -c ../../ryzen/gemm_avx2.c -o gemm_avx2.pic.o
	$(CXX) -shared -o $@ vhuman_deformer_vk.pic.o vulkan-runner.pic.o vkew.pic.o vhuman_deformer.pic.o \
		lightrig_mlp2.pic.o gemm_avx2.pic.o $(LDFLAGS)

clean:
	rm -f *.o bench_vhuman_deformer_vk libvhuman_deformer_vk.so vh_deform.spv.inc
