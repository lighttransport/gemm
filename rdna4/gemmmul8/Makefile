.DEFAULT_GOAL := all
HIPCC ?= /opt/rocm/bin/hipcc
HIP_PATH := $(shell /opt/rocm/bin/hipconfig --path)
export TMPDIR := $(abspath ../../tmp)
CXXFLAGS = -O3 -std=c++20 -Wall -Wextra --offload-arch=gfx1201
$(TMPDIR):
	mkdir -p $@
.PHONY: all test bench clean vendor campaign tune profile
all: bench_wmma bench_modular test_modular test_crt test_encoding bench_ozaki libozaki2.so test_api tune_ozaki
bench_wmma: wmma.hip | $(TMPDIR)
	$(HIPCC) $(CXXFLAGS) $< -L$(HIP_PATH)/lib -Wl,-rpath,$(HIP_PATH)/lib -lrocblas -o $@
bench_modular: bench_modular.hip vendor wmma_modular.hpp
	$(HIPCC) $(CXXFLAGS) bench_modular.hip -Lvendor/GEMMul8/lib -Wl,-rpath,'$$ORIGIN/vendor/GEMMul8/lib' -lgemmul8 -o $@
test_modular: test_modular.hip wmma_modular.hip wmma_modular.hpp | $(TMPDIR)
	$(HIPCC) $(CXXFLAGS) test_modular.hip wmma_modular.hip -o $@
test_crt: test_crt.hip vendor/GEMMul8/src/oz2/undo_scaling/crt_tree.hpp vendor/GEMMul8/src/oz2/common/fp8_plan.hpp | $(TMPDIR)
	$(HIPCC) $(CXXFLAGS) -fconstexpr-steps=4194304 $< -lmpfr -lgmp -o $@
test_encoding: test_encoding.hip vendor/GEMMul8/src/oz2/mod/mod_core.hpp vendor/GEMMul8/src/oz2/common/fp8_awe_lut.hpp | $(TMPDIR)
	$(HIPCC) $(CXXFLAGS) -fconstexpr-steps=4194304 -DOCML_BASIC_ROUNDED_OPERATIONS -DGPU_ARCH=gfx1201 $< -L$(HIP_PATH)/lib -Wl,-rpath,$(HIP_PATH)/lib -lhipblas -lhipblaslt -o $@
vendor: | $(TMPDIR)
	@if [ wmma_modular.hip -nt vendor/GEMMul8/src/oz2/rdna4/cu/wmma_modular.cu ] || [ wmma_modular.hpp -nt vendor/GEMMul8/src/oz2/rdna4/cu/wmma_modular.cu ]; then touch vendor/GEMMul8/src/oz2/rdna4/cu/wmma_modular.cu; fi
	$(MAKE) -C vendor/GEMMul8 BACKEND=hip HIP_PATH=$(HIP_PATH) GPU_ARCH=gfx1201 OPS=gemm OZ2_BACKENDS="INT8 FP8" TEMPDIR=$(TMPDIR)/gemmmul8/buildtmp
bench_ozaki: bench_ozaki.hip libozaki2.so
	$(HIPCC) $(CXXFLAGS) bench_ozaki.hip -L. -Wl,-rpath,'$$ORIGIN' -lozaki2 -L$(HIP_PATH)/lib -Wl,-rpath,$(HIP_PATH)/lib -lrocblas -lamdhip64 -lmpfr -o $@
libozaki2.so: ozaki2.hip ozaki2.hpp vendor
	$(HIPCC) $(CXXFLAGS) -fPIC -shared ozaki2.hip -Lvendor/GEMMul8/lib -Wl,-rpath,'$$ORIGIN/vendor/GEMMul8/lib' -lgemmul8 -L$(HIP_PATH)/lib -Wl,-rpath,$(HIP_PATH)/lib -lhipblaslt -lamdhip64 -o $@
tune_ozaki: tune_ozaki.hip libozaki2.so
	$(HIPCC) $(CXXFLAGS) tune_ozaki.hip -L. -Wl,-rpath,'$$ORIGIN' -lozaki2 -o $@
test_api: test_api.hip libozaki2.so
	$(HIPCC) $(CXXFLAGS) test_api.hip -L. -Wl,-rpath,'$$ORIGIN' -lozaki2 -lmpfr -o $@
test: bench_wmma test_modular test_crt test_encoding bench_ozaki test_api
	./test_api
	./test_modular
	./test_crt
	./test_encoding
	./bench_ozaki --m 17 --n 19 --k 21 --backend int8 --iters 2 --min-sample-ms 0.1
	./bench_ozaki --m 17 --n 19 --k 21 --backend fp8 --iters 2 --min-sample-ms 0.1
	./bench_ozaki --m 2 --n 3 --k 15873 --backend fp8 --iters 1 --warmups 1 --min-sample-ms 0.1
	./bench_ozaki --m 2 --n 3 --k 65537 --backend int8 --iters 1 --warmups 1 --min-sample-ms 0.1
	./bench_wmma 17 33 19
	./bench_wmma 257 193 255
	./bench_wmma 17 19 65536
bench: bench_ozaki
	./bench_ozaki --m 1024 --n 1024 --k 1024 --backend int8
	./bench_ozaki --m 1024 --n 1024 --k 1024 --backend fp8
campaign: bench_ozaki
	python3 benchmark_campaign.py
tune: tune_ozaki
	python3 tune_campaign.py
profile:
	./profile_build.sh
clean:
	rm -f bench_wmma bench_modular test_modular test_crt test_encoding bench_ozaki libozaki2.so test_api tune_ozaki
	$(MAKE) -C vendor/GEMMul8 clean BACKEND=hip HIP_PATH=$(HIP_PATH) GPU_ARCH=gfx1201 OPS=gemm
