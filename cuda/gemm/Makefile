# Tensor-core GEMM benchmark for sm_120 (Blackwell GeForce, RTX 5060 Ti)
#
# Two build artifacts:
#  1. bench_cuda_gemm        — host driver, gcc + cuew + cublasew (no nvcc).
#                              Builds always; PTX kernels via NVRTC at runtime.
#  2. libcutile_gemm.so      — CUTLASS-backed GEMM, built with nvcc.
#                              Built only by `make cutile`; loaded via dlopen
#                              at runtime when --mode cutile is requested.

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I.. -I../../common -mavx2 -mfma -march=native
LDFLAGS = -ldl -lm -lpthread

SRCS = bench_cuda_gemm.c ../cuew.c ../cublasew.c
DEPS = cuda_gemm_ptx_kernels.h ../cuew.h ../cublasew.h \
       ../cuda_runner_common.h

TARGET = bench_cuda_gemm

# CUTLASS path (override with CUTLASS_DIR=...) — must contain include/cutlass/.
CUTLASS_DIR ?= ../../third_party/cutlass

NVCC      = nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=sm_120 \
            -Xcompiler -fPIC -shared \
            -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include \
            --expt-relaxed-constexpr --expt-extended-lambda \
            -Wno-deprecated-gpu-targets

CUTILE_SO = libcutile_gemm.so

.PHONY: all clean run-f16 run-bf16 run-fp8 run-all smoke cutile run-cutile

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

cutile: $(CUTILE_SO)

$(CUTILE_SO): cutile_gemm.cu
	@if [ ! -d "$(CUTLASS_DIR)/include/cutlass" ]; then \
	    echo "ERROR: CUTLASS not found at $(CUTLASS_DIR)."; \
	    echo "       git clone --depth 1 https://github.com/NVIDIA/cutlass $(CUTLASS_DIR)"; \
	    exit 1; \
	fi
	$(NVCC) $(NVCCFLAGS) -o $@ cutile_gemm.cu

smoke: $(TARGET)
	./$(TARGET) --dtype f16 --mode all --shape square_1k

run-f16: $(TARGET)
	./$(TARGET) --dtype f16  --mode all --shape all

run-bf16: $(TARGET)
	./$(TARGET) --dtype bf16 --mode all --shape all

run-fp8: $(TARGET)
	./$(TARGET) --dtype fp8  --mode all --shape all

run-cutile: $(TARGET) $(CUTILE_SO)
	./$(TARGET) --dtype f16  --mode cutile --shape all
	./$(TARGET) --dtype bf16 --mode cutile --shape all

run-all: run-f16 run-bf16 run-fp8

clean:
	rm -f $(TARGET) $(CUTILE_SO)
