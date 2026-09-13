HIPCC ?= hipcc
GPU_ARCH ?= gfx1201
ROCM_LIB ?= /opt/rocm/core/lib
export TMPDIR := $(abspath ../../tmp/pixal3d)
all: libpixal3d_rocm.so
libpixal3d_rocm.so: gpu.hip ../../common/pixal3d_gpu.inc ../../common/pixal3d_hipblas_abi.h | $(TMPDIR)
	$(HIPCC) -std=c++17 -O3 --offload-arch=$(GPU_ARCH) -fPIC -shared $< -L$(ROCM_LIB) -Wl,-rpath,$(ROCM_LIB) -lhipblas -o $@.new
	mv $@.new $@
$(TMPDIR):
	mkdir -p $@
clean:
	rm -f libpixal3d_rocm.so
.PHONY: all clean
