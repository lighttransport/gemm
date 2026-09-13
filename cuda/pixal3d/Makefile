CUDA_PATH ?= /usr/local/cuda-13.2
NVCC ?= $(CUDA_PATH)/bin/nvcc
export TMPDIR := $(abspath ../../tmp/pixal3d)
all: libpixal3d_cuda.so
libpixal3d_cuda.so: gpu.cu ../../common/pixal3d_gpu.inc | $(TMPDIR)
	$(NVCC) -std=c++17 -O3 -arch=sm_120 -Xcompiler=-fPIC -shared $< -lcublas -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64 -o $@.new
	mv $@.new $@
$(TMPDIR):
	mkdir -p $@
clean:
	rm -f libpixal3d_cuda.so
.PHONY: all clean
