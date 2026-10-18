# Makefile for FP4 GEMM / mma probe using cuew
#
# Standalone tools need no CUDA SDK at compile time: cuew loads the driver API
# and NVRTC compiles kernels at runtime. The common `bench` target also builds
# cuda/int4's device CUBINs with nvcc.

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I..
LDFLAGS = -ldl -lm

CUEW = ../cuew.c ../cuew.h
FP4_TMP_DIR ?= $(abspath ../../tmp/fp4)
export TMPDIR := $(FP4_TMP_DIR)
FP4_BENCH_ARGS ?= --shape all --mode all --variant 8 --legacy-fp4 1
CUTLASS_DIR ?= $(abspath ../../tmp/flash-attention-src/csrc/cutlass)
NVCC ?= nvcc
CUTLASS_FLAGS = -O3 -DNDEBUG -std=c++17 -arch=sm_120a -lineinfo \
    -Xptxas=-warn-spills --expt-relaxed-constexpr -diag-suppress=20012 -Xcompiler=-fPIC -shared
CUTLASS_INCLUDES = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
CUTLASS_HEADER = $(CUTLASS_DIR)/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp

.PHONY: all clean cublas cublas-bench cublas-test cublas-sanitize probe w4a4 bench help
.PHONY: cublas-tune cutlass-bench cutlass-sweep cutlass-test cutlass-sanitize

BINS = cublas_fp4_gemm mma_fp4_probe fp4_gemm_test fp4_w4a4_gemm fp4_w4a4_qkv_test fp4_w4a4_opt flux2_fp4_w4a4_test

all: $(BINS)

$(BINS): | $(FP4_TMP_DIR)

$(FP4_TMP_DIR) build:
	mkdir -p $@

# cuBLASLt FP4 GEMM (NVFP4 vs MXFP4 vs BF16/FP8)
cublas_fp4_gemm: cublas_fp4_gemm.c ../int4/bench_common.h ../int4/int4.h ../int4/int4_numeric.h $(CUEW)
	$(CC) $(CFLAGS) -o $@ cublas_fp4_gemm.c ../cuew.c $(LDFLAGS)

# Raw mma.sync issue-rate probe (e2m1 vs e4m3 vs bf16), NVRTC sm_120a
mma_fp4_probe: mma_fp4_probe.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ mma_fp4_probe.c ../cuew.c $(LDFLAGS)

# NVFP4 block-scale mma fragment+scale layout validation vs CPU ref (sm_120a)
fp4_gemm_test: fp4_gemm_test.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ fp4_gemm_test.c ../cuew.c $(LDFLAGS)

# Tiled W4A4 [M,N,K] GEMM (1 warp/16x8 tile) validated vs CPU ref (sm_120a)
fp4_w4a4_gemm: fp4_w4a4_gemm.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ fp4_w4a4_gemm.c ../cuew.c $(LDFLAGS)

# Full W4A4 op (quant+gemm+lora) on real block-0 to_qkv data in /tmp/qkv_*.bin
fp4_w4a4_qkv_test: fp4_w4a4_qkv_test.c $(CUEW)
	$(CC) $(CFLAGS) -o $@ fp4_w4a4_qkv_test.c ../cuew.c $(LDFLAGS)

# Optimized W4A4 GEMM dev/verify bench: naive oracle vs shared-mem tiled (sync,
# shipped in cuda/qimg) vs cp.async double-buffer; all bit-exact, ~4-6x over naive
fp4_w4a4_opt: fp4_w4a4_opt.c fp4_gemm_kernels.h $(CUEW)
	$(CC) $(CFLAGS) -o $@ fp4_w4a4_opt.c ../cuew.c $(LDFLAGS)

# Exercise the shared cuda/fp4_w4a4.h API at FLUX.2-Klein shapes, verify the
# dispatch (quant+gemm+combine) matches a CPU reference bit-for-bit.
flux2_fp4_w4a4_test: flux2_fp4_w4a4_test.c ../fp4_w4a4.h $(CUEW)
	$(CC) $(CFLAGS) -o $@ flux2_fp4_w4a4_test.c ../cuew.c $(LDFLAGS)

cublas: cublas_fp4_gemm
	./cublas_fp4_gemm

cublas-bench: cublas_fp4_gemm | build
	./cublas_fp4_gemm -m 8192 -n 8192 -k 8192 --format nvfp4 --output all --csv build/cublas_nvfp4_8k.csv

cublas-tune: cublas_fp4_gemm | build
	./cublas_fp4_gemm --format nvfp4 --output bf16 --exhaustive --csv build/cublas_tuning_8k.csv

# Optional external CUTLASS dependency. The selected kernel uses unmodified
# upstream headers; generated overrides below are separate experiments.
build/libcutlass_fp4_release.so: cutlass_fp4.cu $(CUTLASS_HEADER) | build $(FP4_TMP_DIR)
	$(NVCC) $(CUTLASS_FLAGS) -DFP4_WINNER_ONLY $(CUTLASS_INCLUDES) $< -o $@

build/libcutlass_fp4_release_sweep.so: cutlass_fp4.cu $(CUTLASS_HEADER) | build $(FP4_TMP_DIR)
	$(NVCC) $(CUTLASS_FLAGS) $(CUTLASS_INCLUDES) $< -o $@

build/libcutlass_fp4_static.so: cutlass_fp4.cu $(CUTLASS_HEADER) | build $(FP4_TMP_DIR)
	$(NVCC) $(CUTLASS_FLAGS) -DFP4_EXTRA_TILES -DFP4_STATIC_SCHEDULER $(CUTLASS_INCLUDES) $< -o $@

build/warp_release_include/.stamp: prepare_cutlass_fp4.py $(CUTLASS_HEADER) | build
	python3 $< $(CUTLASS_DIR) build/warp_release_include
	touch $@

build/unroll4_include/.stamp: prepare_cutlass_fp4.py $(CUTLASS_HEADER) | build
	python3 $< $(CUTLASS_DIR) build/unroll4_include --handoff cta --unroll 4
	touch $@

build/libcutlass_fp4_warp_release.so: cutlass_fp4.cu build/warp_release_include/.stamp | $(FP4_TMP_DIR)
	$(NVCC) $(CUTLASS_FLAGS) -DFP4_WINNER_ONLY -Ibuild/warp_release_include $(CUTLASS_INCLUDES) $< -o $@

build/libcutlass_fp4_unroll4.so: cutlass_fp4.cu build/unroll4_include/.stamp | $(FP4_TMP_DIR)
	$(NVCC) $(CUTLASS_FLAGS) -DFP4_WINNER_ONLY -DFP4_NO_SOURCE -Ibuild/unroll4_include $(CUTLASS_INCLUDES) $< -o $@

cutlass-bench: cublas_fp4_gemm build/libcutlass_fp4_release.so
	./cublas_fp4_gemm --format nvfp4 --output bf16 --cutlass ./build/libcutlass_fp4_release.so \
	    --raster 2 --swizzle 4 --compare-cublas --csv build/cutlass_nvfp4_8k.csv

cutlass-sweep: cublas_fp4_gemm build/libcutlass_fp4_release_sweep.so
	./cublas_fp4_gemm --format nvfp4 --output bf16 --cutlass ./build/libcutlass_fp4_release_sweep.so \
	    --raster 2 --swizzle 4 --compare-cublas --csv build/cutlass_sweep_8k.csv

cutlass-test: cublas_fp4_gemm build/libcutlass_fp4_release.so
	./cublas_fp4_gemm --format nvfp4 --output bf16 --cutlass ./build/libcutlass_fp4_release.so \
	    -m 128 -n 256 -k 128 --full-check --compare-cublas --iters 2 --rounds 1
	./cublas_fp4_gemm --format nvfp4 --output bf16 --cutlass ./build/libcutlass_fp4_release.so \
	    -m 384 -n 128 -k 320 --full-check --compare-cublas --iters 2 --rounds 1

cutlass-sanitize: cublas_fp4_gemm build/libcutlass_fp4_release.so
	compute-sanitizer --tool memcheck --error-exitcode 1 ./cublas_fp4_gemm \
	    --format nvfp4 --output bf16 --cutlass ./build/libcutlass_fp4_release.so \
	    -m 384 -n 128 -k 320 --full-check --warmup 1 --iters 1 --rounds 1
	compute-sanitizer --tool racecheck --error-exitcode 1 ./cublas_fp4_gemm \
	    --format nvfp4 --output bf16 --cutlass ./build/libcutlass_fp4_release.so \
	    -m 384 -n 128 -k 320 --full-check --warmup 1 --iters 1 --rounds 1

cublas-test: cublas_fp4_gemm | build
	./cublas_fp4_gemm -m 128 -n 256 -k 128 --format nvfp4 --output all --full-check --iters 2 --rounds 1 --algos 8
	./cublas_fp4_gemm -m 256 -n 128 -k 192 --format nvfp4 --output all --full-check --iters 2 --rounds 1 --algos 8

cublas-sanitize: cublas_fp4_gemm
	compute-sanitizer --tool memcheck --error-exitcode 1 ./cublas_fp4_gemm -m 128 -n 256 -k 128 --format nvfp4 --output all --full-check --iters 1 --rounds 1 --algos 4

probe: mma_fp4_probe
	./mma_fp4_probe

# Run the self-contained W4A4 validators (qkv_test needs /tmp/qkv_*.bin, run it manually)
w4a4: fp4_gemm_test fp4_w4a4_gemm
	./fp4_gemm_test
	./fp4_w4a4_gemm

# A common runner uses identical packed FP4 inputs, validation and event timing
# for these original kernels and the new FP4/E0M3 kernels. Variant 8 fixes tiling.
bench: | build $(FP4_TMP_DIR)
	$(MAKE) -C ../int4 all
	../int4/build/bench_int4 $(FP4_BENCH_ARGS) --csv build/fp4_compare.csv

clean:
	rm -f $(BINS)
	rm -rf build

help:
	@echo "make            - build all binaries"
	@echo "make cublas     - build+run cuBLASLt FP4 GEMM (NVFP4/MXFP4 vs BF16/FP8)"
	@echo "make cublas-bench - tune and benchmark 8192-cubed NVFP4 with F32/BF16/F16 outputs"
	@echo "make cublas-test  - verify all outputs with varying data and swizzled K16 scales"
	@echo "make cublas-sanitize - check cuBLAS NVFP4 memory accesses"
	@echo "make cublas-tune - search supported cuBLAS FP4 IDs, tiles, stages and swizzles"
	@echo "make cutlass-bench - benchmark selected dense FP4 tile; CUTLASS_DIR must point to an SM120 checkout"
	@echo "make cutlass-sweep - compare 14 CUTLASS tile/pipeline/schedule variants"
	@echo "make cutlass-test|cutlass-sanitize - validate the selected CUTLASS kernel"
	@echo "make probe      - build+run raw mma.sync issue-rate probe (sm_120a)"
	@echo "make w4a4       - build+run the W4A4 block-scale + tiled-GEMM validators"
	@echo "make bench      - compare original FP4 kernels, new FP4/E0M3, and cuBLAS"
	@echo "make clean      - remove binaries"
