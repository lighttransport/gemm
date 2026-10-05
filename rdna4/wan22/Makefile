ROOT := ../..
BUILD ?= $(ROOT)/tmp/video-rocm/wan22-build
HIPCC ?= /opt/rocm/bin/hipcc
PYTHON ?= $(ROOT)/tmp/vhuman-rocm-venv/bin/python
export TMPDIR := $(abspath $(BUILD))
.PHONY: all test clean
all: $(BUILD)/libwan22_hip.so
$(BUILD):
	mkdir -p $@
$(BUILD)/libwan22_hip.so: projection.hip blaslt.hip ../video_common/gemm.hip | $(BUILD)
	$(HIPCC) -O3 -std=c++17 -Wall -Wextra -fPIC -shared --offload-arch=gfx1201 projection.hip blaslt.hip -lhipblaslt -o $@
test: all
	mkdir -p $(ROOT)/tmp/pixal3d/device-locks
	LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:$${LD_LIBRARY_PATH} PYTHONDONTWRITEBYTECODE=1 flock $(ROOT)/tmp/pixal3d/device-locks/rocm-0.lock $(PYTHON) test_hip.py
clean:
	rm -f $(BUILD)/libwan22_hip.so
