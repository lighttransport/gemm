CC ?= gcc
CFLAGS = -O3 -Wall -Wextra -Wno-unused-function -fopenmp
DA2_EXTRA_SRC ?=
DA2_EXTRA_FLAGS ?=
DA2_EXTRA_DEPS ?=
DA2_EXTRA_LIBS ?=
AVX_OBJ =
ifneq (,$(filter x86_64 amd64,$(shell uname -m)))
CFLAGS += -DDA2_HAVE_AVX2 -mavx2 -mfma -mf16c
AVX_OBJ = gemm_avx2.o
endif
.DEFAULT_GOAL := da2_depth
gemm_avx2.o: ../../ryzen/gemm_avx2.c ../../ryzen/gemm_avx2.h
	$(CC) -O3 -mavx2 -mfma -c $< -o $@
da2_depth: ../../common/da2_runner.c $(wildcard ../../common/*.h) $(AVX_OBJ) $(DA2_EXTRA_DEPS) $(DA2_EXTRA_SRC)
	$(CC) $(CFLAGS) $(DA2_EXTRA_FLAGS) -o $@ $< $(AVX_OBJ) $(DA2_EXTRA_SRC) -lm -lpthread $(DA2_EXTRA_LIBS)
test_da2_ops: ../../cpu/da2/test_da2_ops.c ../../common/da2_runner.c $(wildcard ../../common/*.h) $(AVX_OBJ) $(DA2_EXTRA_DEPS) $(DA2_EXTRA_SRC)
	$(CC) $(CFLAGS) $(DA2_EXTRA_FLAGS) -o $@ $< $(AVX_OBJ) $(DA2_EXTRA_SRC) -lm -lpthread $(DA2_EXTRA_LIBS)
.PHONY: clean
clean:
	rm -f da2_depth test_da2_ops gemm_avx2.o
