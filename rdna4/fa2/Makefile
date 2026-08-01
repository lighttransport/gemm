CC = gcc
CXX = g++
ROCM_PATH ?= /opt/rocm

CXXFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread
ROCEW_SRC = ../rocew.c

BENCH = bench_fa2

.PHONY: all clean bench check tests help

all: $(BENCH)

$(BENCH): bench_fa2.cpp $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h
	$(CXX) $(CXXFLAGS) -o $@ bench_fa2.cpp $(ROCEW_SRC) $(LDFLAGS) \
	    -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64

# bf16 WMMA isolation tests (BF16_DEBUG.md): each takes f16|bf16, prints L1err
tests: wmma_min_rtc wmma_min_aot_f16 wmma_min_aot_bf16 wmma_mid_rtc
wmma_min_rtc wmma_mid_rtc: %: %.cpp $(ROCEW_SRC)
	$(CXX) -O3 -I.. -o $@ $< $(ROCEW_SRC) $(LDFLAGS) -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64
wmma_min_aot_%: wmma_min_aot.cpp
	$(ROCM_PATH)/bin/hipcc -O3 --offload-arch=gfx1201 $(if $(findstring bf16,$@),-DT_BF16,) -o $@ $<

bench: $(BENCH)
	./$(BENCH) --n-tok 4096 --heads 16 --iters 100 --mode all

check: $(BENCH)
	./$(BENCH) --n-tok 256 --heads 2 --iters 20 --mode all --check

clean:
	rm -f $(BENCH) wmma_min_rtc wmma_min_aot_f16 wmma_min_aot_bf16 wmma_mid_rtc

help:
	@echo "RDNA4 fa2 BF16/FP16/FP8 flash attention (blog swizzle + double-buffer)"
	@echo "  make -C rdna4/fa2 && rdna4/fa2/bench_fa2 --mode all --check --n-tok 256"
	@echo "  modes: b16 b16s b16d  f16 f16s  fp8 fp8s  all   (s=swizzle d=double-buf)"
