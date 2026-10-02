CC ?= gcc
CFLAGS = -O3 -Wall -Wextra -Wno-unused-function -fopenmp
VH_EXTRA_FLAGS ?=
VH_EXTRA_SRC ?=
VH_EXTRA_LIBS ?=
VH_EXTRA_DEPS ?=
AVX_OBJ =
ifneq (,$(filter x86_64 amd64,$(shell uname -m)))
CFLAGS += -DSWIN_HAVE_AVX2 -mavx2 -mfma -mf16c
AVX_OBJ = gemm_avx2.o
endif
.DEFAULT_GOAL := vhuman_models
libvhuman_motion.so: ../../common/vhuman_motion.c ../../common/swin_runner.c $(wildcard ../../common/*.h) $(AVX_OBJ)
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $< $(AVX_OBJ) -lm -lpthread
test_vhuman_ops: test_vhuman_ops.c ../../common/vhuman_models.c $(wildcard ../../common/*.h) $(AVX_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(AVX_OBJ) -lm -lpthread
.PHONY: test
test: test_vhuman_ops test_vhuman_training
	./test_vhuman_ops
	./test_vhuman_training
gemm_avx2.o: ../../ryzen/gemm_avx2.c ../../ryzen/gemm_avx2.h
	$(CC) -O3 -mavx2 -mfma -fPIC -c $< -o $@
vhuman_models: ../../common/vhuman_models.c $(wildcard ../../common/*.h) ../../common/swin_runner.c $(AVX_OBJ) $(VH_EXTRA_SRC) $(VH_EXTRA_DEPS)
	$(CC) $(CFLAGS) $(VH_EXTRA_FLAGS) -o $@ $< $(AVX_OBJ) $(VH_EXTRA_SRC) -lm -lpthread $(VH_EXTRA_LIBS)
.PHONY: clean
clean:
	rm -f vhuman_models libvhuman_motion.so libvhuman_landmarks.so libvhuman_training.so test_vhuman_ops test_vhuman_training gemm_avx2.o tokenizer_probe

tokenizer_probe: tokenizer_probe.c ../../common/gguf_loader.h ../../common/bpe_tokenizer.h
	$(CC) $(CFLAGS) -o $@ $< -lm

libvhuman_landmarks.so: landmarks.cpp $(AVX_OBJ)
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp $(if $(AVX_OBJ),-mavx2 -mfma) -fPIC -shared -o $@ $< $(AVX_OBJ)

libvhuman_training.so: training.cpp corrective_training.cpp $(AVX_OBJ)
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp $(if $(AVX_OBJ),-DVH_TRAIN_AVX2 -mavx2 -mfma) -fPIC -shared -o $@ $< $(AVX_OBJ)

test_vhuman_training: test_vhuman_training.cpp training.cpp corrective_training.cpp $(AVX_OBJ)
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wpedantic -fopenmp $(if $(AVX_OBJ),-DVH_TRAIN_AVX2 -mavx2 -mfma) -o $@ $< $(AVX_OBJ)
