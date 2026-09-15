CUDA_PATH ?= /usr/local/cuda-13.2
NVCC ?= $(CUDA_PATH)/bin/nvcc
export TMPDIR := $(abspath ../../tmp/pixal3d)
all: libpixal3d_cuda.so
libpixal3d_cuda.so: gpu.cu $(TMPDIR)/fa2.inc ../../common/pixal3d_gpu.inc ../../common/pixal3d_resident.inc ../../common/pixal3d_mma.inc ../../common/pixal3d_device.h | $(TMPDIR)
	$(NVCC) -std=c++17 -O3 --fmad=false -arch=sm_120 -Xcompiler=-fPIC -shared $< -lcublas -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64 -o $@.new
	mv $@.new $@
$(TMPDIR)/fa2.inc: extract_fa2.py ../fa2/cuda_fa2_kernels.h | $(TMPDIR)
	python3 extract_fa2.py $@
$(TMPDIR):
	mkdir -p $@
clean:
	rm -f libpixal3d_cuda.so
.PHONY: all clean
