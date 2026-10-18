CC = gcc
NVCC ?= nvcc
PYTHON ?= python3
CUDA_HOME ?= /usr/local/cuda
# CUDA 13.1 names E0M3 in SASS; CUDA 13.2's disassembler omits that spelling.
CUOBJDUMP ?= $(firstword $(wildcard /usr/local/cuda-13.1/bin/cuobjdump) $(CUDA_HOME)/bin/cuobjdump)
BUILD := build
INT4_TMP_DIR ?= $(abspath ../../tmp/int4)
export TMPDIR := $(INT4_TMP_DIR)
CFLAGS := -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic -I..
LDLIBS := -ldl -lm
NVCCFLAGS := -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-warn-spills
HOST := $(BUILD)/int4.o $(BUILD)/cuew.o
DEPS := int4.h int4_numeric.h ../cuew.h
VARIANTS := $(BUILD)/e0m3_e0m3.cubin $(BUILD)/e0m3_e2m1.cubin $(BUILD)/e2m1_e0m3.cubin
CUTLASS_DIR ?= $(abspath ../../tmp/flash-attention-src/csrc/cutlass)
CUTLASS_HEADER := $(CUTLASS_DIR)/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
CUTLASS_RUNNER := ../fp4/cublas_fp4_gemm
CUTLASS_FLAGS := -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_120a,code=sm_120a \
    --compress-mode none -lineinfo -Xptxas=-warn-spills --expt-relaxed-constexpr \
    -diag-suppress=20012 -Xcompiler=-fPIC -shared -DFP4_WINNER_ONLY

.PHONY: all clean test test-cpu test-gpu probe bench sanitize
.PHONY: cutlass-bench cutlass-control cutlass-test cutlass-sanitize
.PHONY: decode-bench decode-test decode-sanitize
all: $(BUILD)/bench_int4 $(BUILD)/test_int4 $(VARIANTS) $(BUILD)/toolchain.txt

$(BUILD) $(INT4_TMP_DIR):
	mkdir -p $@

$(BUILD)/e2m1_e2m1.cubin: kernels.cu int4_numeric.h | $(BUILD) $(INT4_TMP_DIR)
	$(NVCC) $(NVCCFLAGS) -cubin $< -o $@

$(BUILD)/patch.stamp: $(BUILD)/e2m1_e2m1.cubin patch_formats.py
	$(PYTHON) patch_formats.py $< $(BUILD) --cuobjdump $(CUOBJDUMP) --scratch-dir $(INT4_TMP_DIR)
	touch $@

$(BUILD)/toolchain.txt: Makefile | $(BUILD) $(INT4_TMP_DIR)
	$(NVCC) --version > $@
	$(CUOBJDUMP) --version >> $@
	@echo 'NVCCFLAGS=$(NVCCFLAGS)' >> $@

$(VARIANTS): $(BUILD)/patch.stamp
	@test -f $@ || { rm -f $(BUILD)/patch.stamp; $(MAKE) $(BUILD)/patch.stamp; }

$(BUILD)/int4.o: int4.c $(DEPS) | $(BUILD) $(INT4_TMP_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# The existing dynamic loaders use POSIX dlsym function-pointer conversions.
$(BUILD)/cuew.o: ../cuew.c ../cuew.h | $(BUILD) $(INT4_TMP_DIR)
	$(CC) $(CFLAGS) -Wno-pedantic -c $< -o $@

$(BUILD)/cublasew.o: ../cublasew.c ../cublasew.h ../cuew.h | $(BUILD) $(INT4_TMP_DIR)
	$(CC) $(CFLAGS) -Wno-pedantic -c $< -o $@

$(BUILD)/fp4_legacy.o: fp4_legacy.c fp4_legacy.h ../fp4/fp4_gemm_kernels.h $(DEPS) | $(BUILD) $(INT4_TMP_DIR)
	$(CC) $(CFLAGS) -Wno-overlength-strings -c $< -o $@

$(BUILD)/bench_int4: bench_int4.c bench_common.h fp4_legacy.h $(HOST) $(DEPS) $(BUILD)/cublasew.o $(BUILD)/fp4_legacy.o | $(BUILD) $(INT4_TMP_DIR)
	$(CC) $(CFLAGS) -o $@ bench_int4.c $(HOST) $(BUILD)/cublasew.o $(BUILD)/fp4_legacy.o $(LDLIBS)

$(BUILD)/test_int4: test_int4.c bench_common.h $(HOST) $(DEPS) | $(BUILD) $(INT4_TMP_DIR)
	$(CC) $(CFLAGS) -o $@ test_int4.c $(HOST) $(LDLIBS)

test-cpu: $(BUILD)/test_int4 $(BUILD)/patch.stamp
	./$(BUILD)/test_int4 --cpu
	$(PYTHON) -m unittest -v test_patch_formats.py

test-gpu: all
	./$(BUILD)/test_int4

test: test-cpu test-gpu

probe: all
	./$(BUILD)/bench_int4 --mode probe

bench: all
	./$(BUILD)/bench_int4 --shape all --mode all --csv $(BUILD)/results.csv
	./$(BUILD)/bench_int4 --shape all --mode packed --scales unit --csv $(BUILD)/unit.csv

sanitize: all
	$(CUDA_HOME)/bin/compute-sanitizer --tool memcheck --error-exitcode 1 ./$(BUILD)/test_int4 --sanitizer
	$(CUDA_HOME)/bin/compute-sanitizer --tool racecheck --error-exitcode 1 ./$(BUILD)/test_int4 --sanitizer

# Reuse the tuned FP4 tile/pipeline/host adapter. Only verified OMMA format
# bytes differ in the final E0M3 library; never include a PTX JIT fallback.
$(BUILD)/libcutlass_e0m3_base.so: ../fp4/cutlass_fp4.cu Makefile $(CUTLASS_HEADER) | $(BUILD) $(INT4_TMP_DIR)
	$(NVCC) $(CUTLASS_FLAGS) -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include $< -o $@

$(BUILD)/libcutlass_e0m3.so $(BUILD)/libcutlass_e0m3.sass $(BUILD)/libcutlass_e0m3.patch.json &: \
        $(BUILD)/libcutlass_e0m3_base.so patch_cutlass.py patch_formats.py | $(INT4_TMP_DIR)
	$(PYTHON) patch_cutlass.py $< $(BUILD)/libcutlass_e0m3.so --cuobjdump $(CUOBJDUMP) --scratch-dir $(INT4_TMP_DIR)

$(CUTLASS_RUNNER): ../fp4/cublas_fp4_gemm.c bench_common.h int4.h int4_numeric.h ../cuew.c ../cuew.h
	$(MAKE) -C ../fp4 cublas_fp4_gemm

cutlass-bench: $(CUTLASS_RUNNER) $(BUILD)/libcutlass_e0m3.so
	$(CUTLASS_RUNNER) --format e0m3 --output bf16 --cutlass ./$(BUILD)/libcutlass_e0m3.so \
	    --raster 2 --swizzle 4 --compare-cublas --csv $(BUILD)/cutlass_e0m3_8k.csv

cutlass-control: $(CUTLASS_RUNNER) $(BUILD)/libcutlass_e0m3_base.so
	$(CUTLASS_RUNNER) --format nvfp4 --output bf16 --cutlass ./$(BUILD)/libcutlass_e0m3_base.so \
	    --raster 2 --swizzle 4 --compare-cublas --csv $(BUILD)/cutlass_fp4_control_8k.csv

cutlass-test: $(CUTLASS_RUNNER) $(BUILD)/libcutlass_e0m3.so $(BUILD)/libcutlass_e0m3.patch.json $(BUILD)/libcutlass_e0m3.sass
	$(PYTHON) -m unittest -v test_patch_cutlass.py test_cutlass_e0m3.py

cutlass-sanitize: $(CUTLASS_RUNNER) $(BUILD)/libcutlass_e0m3.so
	$(CUDA_HOME)/bin/compute-sanitizer --tool memcheck --error-exitcode 1 $(CUTLASS_RUNNER) \
	    --format e0m3 --output bf16 --cutlass ./$(BUILD)/libcutlass_e0m3.so \
	    -m 384 -n 128 -k 320 --full-check --warmup 1 --iters 1 --rounds 1

	$(CUDA_HOME)/bin/compute-sanitizer --tool racecheck --error-exitcode 1 $(CUTLASS_RUNNER) \
	    --format e0m3 --output bf16 --cutlass ./$(BUILD)/libcutlass_e0m3.so \
	    -m 384 -n 128 -k 320 --full-check --warmup 1 --iters 1 --rounds 1

$(BUILD)/libdecode_base.so: decode.cu Makefile | $(BUILD) $(INT4_TMP_DIR)
	$(NVCC) -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_120a,code=sm_120a \
	    --compress-mode none -lineinfo -Xptxas=-warn-spills -Xcompiler=-fPIC -shared $< -o $@

$(BUILD)/libdecode_e0m3.so $(BUILD)/libdecode_e0m3.sass $(BUILD)/libdecode_e0m3.patch.json &: \
        $(BUILD)/libdecode_base.so patch_cutlass.py patch_formats.py | $(INT4_TMP_DIR)
	$(PYTHON) patch_cutlass.py $< $(BUILD)/libdecode_e0m3.so --cuobjdump $(CUOBJDUMP) --scratch-dir $(INT4_TMP_DIR)

decode-bench: $(CUTLASS_RUNNER) $(BUILD)/libdecode_e0m3.so $(BUILD)/libcutlass_e0m3.so
	$(PYTHON) bench_decode.py

decode-test: $(CUTLASS_RUNNER) $(BUILD)/libdecode_e0m3.so
	$(PYTHON) -m unittest -v test_decode.py

decode-sanitize: $(CUTLASS_RUNNER) $(BUILD)/libdecode_e0m3.so
	$(CUDA_HOME)/bin/compute-sanitizer --tool memcheck --error-exitcode 1 $(CUTLASS_RUNNER) \
	    --format e0m3 --output bf16 --cutlass ./$(BUILD)/libdecode_e0m3.so \
	    -m 17 -n 40 -k 576 --full-check --warmup 1 --iters 1 --rounds 1
	$(CUDA_HOME)/bin/compute-sanitizer --tool racecheck --error-exitcode 1 $(CUTLASS_RUNNER) \
	    --format e0m3 --output bf16 --cutlass ./$(BUILD)/libdecode_e0m3.so \
	    -m 17 -n 40 -k 576 --full-check --warmup 1 --iters 1 --rounds 1

clean:
	rm -rf $(BUILD)
