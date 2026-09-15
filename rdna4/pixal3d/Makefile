HIPCC ?= hipcc
GPU_ARCH ?= gfx1201
ROCM_LIB ?= /opt/rocm/core/lib
export TMPDIR := $(abspath ../../tmp/pixal3d)
all: libpixal3d_rocm.so
libpixal3d_rocm.so: gpu.hip $(TMPDIR)/wmma_attention.inc ../../common/pixal3d_gpu.inc ../../common/pixal3d_hipblas_abi.h ../../common/pixal3d_resident.inc ../../common/pixal3d_mma.inc ../../common/pixal3d_device.h | $(TMPDIR)
	$(HIPCC) -std=c++17 -O3 -ffp-contract=off --offload-arch=$(GPU_ARCH) -fPIC -shared $< -L$(ROCM_LIB) -Wl,-rpath,$(ROCM_LIB) -lhipblas -o $@.new
	mv $@.new $@
$(TMPDIR)/wmma_attention.inc: extract_attention.py ../trellis2/hip_trellis2_kernels.h | $(TMPDIR)
	python3 extract_attention.py $@
$(TMPDIR):
	mkdir -p $@
clean:
	rm -f libpixal3d_rocm.so
.PHONY: all clean
