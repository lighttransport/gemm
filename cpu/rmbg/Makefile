CC ?= gcc
CFLAGS = -O3 -Wall -Wextra -Wno-unused-function -fopenmp
AVX_OBJ =
SWIN_EXTRA_FLAGS ?=
SWIN_EXTRA_SRC ?=
SWIN_EXTRA_DEPS ?=
SWIN_EXTRA_LIBS ?=
ifneq (,$(filter x86_64 amd64,$(shell uname -m)))
CFLAGS += -DSWIN_HAVE_AVX2 -mavx2 -mfma
AVX_OBJ = gemm_avx2.o
endif
.DEFAULT_GOAL := swin_backbone
gemm_avx2.o: ../../ryzen/gemm_avx2.c ../../ryzen/gemm_avx2.h
	$(CC) -O3 -mavx2 -mfma -c $< -o $@
swin_backbone: ../../common/swin_runner.c ../../common/swin_v1.h ../../common/safetensors.h $(AVX_OBJ) $(SWIN_EXTRA_SRC) $(SWIN_EXTRA_DEPS)
	$(CC) $(CFLAGS) $(SWIN_EXTRA_FLAGS) -o $@ $< $(AVX_OBJ) $(SWIN_EXTRA_SRC) -lm -lpthread $(SWIN_EXTRA_LIBS)
test_swin: ../../cpu/rmbg/test_swin.c ../../common/swin_runner.c ../../common/swin_v1.h ../../common/safetensors.h $(AVX_OBJ) $(SWIN_EXTRA_SRC) $(SWIN_EXTRA_DEPS)
	$(CC) $(CFLAGS) $(SWIN_EXTRA_FLAGS) -o $@ $< $(AVX_OBJ) $(SWIN_EXTRA_SRC) -lm -lpthread $(SWIN_EXTRA_LIBS)
.PHONY: test clean
test: test_swin
	./test_swin
clean:
	rm -f swin_backbone test_swin gemm_avx2.o
