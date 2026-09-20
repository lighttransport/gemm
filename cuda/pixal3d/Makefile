CUDA_PATH ?= /usr/local/cuda-13.2
NVCC ?= $(CUDA_PATH)/bin/nvcc
GPU_ARCH ?= sm_120
SUPPORTED_GPU_ARCHS := sm_80 sm_89 sm_120
ifeq ($(filter $(GPU_ARCH),$(SUPPORTED_GPU_ARCHS)),)
$(error GPU_ARCH must be one of $(SUPPORTED_GPU_ARCHS))
endif
export TMPDIR := $(abspath ../../tmp/pixal3d)
ARCH_CHECK_OBJECTS := $(addprefix $(TMPDIR)/pixal3d-gpu-,$(addsuffix .o,$(SUPPORTED_GPU_ARCHS)))
all: libpixal3d_cuda.so
libpixal3d_cuda.so: gpu.cu $(TMPDIR)/fa2.inc $(TMPDIR)/pixal3d-gpu-arch ../../common/pixal3d_gpu.inc ../../common/pixal3d_resident.inc ../../common/pixal3d_mma.inc ../../common/pixal3d_device.h | $(TMPDIR)
	$(NVCC) -std=c++17 -O3 --fmad=false -arch=$(GPU_ARCH) -Xcompiler=-fPIC -shared $< -lcublas -Xlinker -rpath -Xlinker $(CUDA_PATH)/lib64 -o $@.new
	mv $@.new $@
$(TMPDIR)/pixal3d-gpu-%.o: gpu.cu $(TMPDIR)/fa2.inc ../../common/pixal3d_gpu.inc ../../common/pixal3d_resident.inc ../../common/pixal3d_mma.inc ../../common/pixal3d_device.h | $(TMPDIR)
	$(NVCC) -std=c++17 -O3 --fmad=false -arch=$* -Xcompiler=-fPIC -c $< -o $@
check-architectures: $(ARCH_CHECK_OBJECTS)
	rm -f $(ARCH_CHECK_OBJECTS)
$(TMPDIR)/fa2.inc: extract_fa2.py ../fa2/cuda_fa2_kernels.h | $(TMPDIR)
	python3 extract_fa2.py $@
$(TMPDIR)/pixal3d-gpu-arch: FORCE | $(TMPDIR)
	@if ! test -f $@ || ! test "`cat $@`" = "$(GPU_ARCH)"; then echo $(GPU_ARCH) > $@; fi
$(TMPDIR):
	mkdir -p $@
clean:
	rm -f libpixal3d_cuda.so $(TMPDIR)/pixal3d-gpu-arch
FORCE:
.PHONY: all clean check-architectures FORCE
