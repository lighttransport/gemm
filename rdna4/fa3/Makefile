CC = gcc
CXX = g++
ROCM_PATH ?= /opt/rocm

CXXFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread
ROCEW_SRC = ../rocew.c

BENCH    = bench_fa3_fp8
BENCH_V2 = bench_fa3_fp8_v2
BENCH_F16 = bench_fa3_fp16

.PHONY: all clean run bench bench-v2 bench-f16 help

all: $(BENCH) $(BENCH_V2) $(BENCH_F16)

$(BENCH): bench_fa3_fp8.cpp $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h
	$(CXX) $(CXXFLAGS) -o $@ bench_fa3_fp8.cpp $(ROCEW_SRC) $(LDFLAGS) \
	    -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64

$(BENCH_V2): bench_fa3_fp8_v2.cpp $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h
	$(CXX) $(CXXFLAGS) -o $@ bench_fa3_fp8_v2.cpp $(ROCEW_SRC) $(LDFLAGS) \
	    -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64

$(BENCH_F16): bench_fa3_fp16.cpp $(ROCEW_SRC) ../rocew.h ../hip_runner_common.h
	$(CXX) $(CXXFLAGS) -o $@ bench_fa3_fp16.cpp $(ROCEW_SRC) $(LDFLAGS) \
	    -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64

bench: $(BENCH)
	./$(BENCH) --n-tok 1024 --heads 16 --iters 100 --mode all

bench-v2: $(BENCH_V2)
	./$(BENCH_V2) --mode all --n-tok 4096 --heads 16 --iters 100

bench-f16: $(BENCH_F16)
	./$(BENCH_F16) --mode all --n-tok 4096 --heads 16 --iters 100

run: bench

clean:
	rm -f $(BENCH) $(BENCH_V2) $(BENCH_F16)

help:
	@echo "RDNA4 FP8 FA3-style attention benchmarks"
	@echo "  make -C rdna4/fa3"
	@echo "  rdna4/fa3/bench_fa3_fp8 --mode ..."
	@echo "  rdna4/fa3/bench_fa3_fp8_v2 --mode v1|v2|v3|v4|all"
