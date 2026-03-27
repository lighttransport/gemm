CC ?= gcc
CFLAGS = -O2 -mavx2 -mfma -Wall -Wno-unused-function -I../../common -I..
LDFLAGS = -lm -ldl -lpthread

.PHONY: all clean test-init test-load test-dit test-gen

all: test_cuda_qimg

test_cuda_qimg: test_cuda_qimg.c cuda_qimg_runner.h \
		../cuda_runner_common.h ../cuda_kernels_common.h ../cuew.h \
		../../common/safetensors.h ../../common/gguf_loader.h \
		../../common/ggml_dequant.h ../../common/transformer.h \
		../../common/bpe_tokenizer.h ../../common/qwen_image_scheduler.h \
		../../common/qwen_image_dit.h ../../common/qwen_image_vae.h \
		../../common/qwen_image_text_encoder.h
	$(CC) $(CFLAGS) -o $@ $< ../cuew.c $(LDFLAGS)

test-init: test_cuda_qimg
	./test_cuda_qimg --test-init

test-load: test_cuda_qimg
	./test_cuda_qimg --test-load

test-dit: test_cuda_qimg
	./test_cuda_qimg --test-dit --height 64 --width 64

test-gen: test_cuda_qimg
	./test_cuda_qimg --generate \
		--height 256 --width 256 --steps 20 --seed 42 \
		--prompt "a red apple on a white table"

clean:
	rm -f test_cuda_qimg *.ppm *.npy
