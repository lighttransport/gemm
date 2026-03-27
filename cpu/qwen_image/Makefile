CC ?= gcc
CFLAGS = -O2 -Wall -Wno-unused-function -I../../common
LDFLAGS = -lm -lpthread

.PHONY: all clean test-sched test-vae test-dit test-enc test-gen

all: test_qwen_image

test_qwen_image: test_qwen_image.c \
		../../common/qwen_image_scheduler.h \
		../../common/qwen_image_dit.h \
		../../common/qwen_image_vae.h \
		../../common/qwen_image_text_encoder.h \
		../../common/transformer.h \
		../../common/bpe_tokenizer.h \
		../../common/safetensors.h \
		../../common/gguf_loader.h \
		../../common/ggml_dequant.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Quick tests
test-sched: test_qwen_image
	./test_qwen_image --test-sched

test-vae: test_qwen_image
	./test_qwen_image --test-vae /mnt/disk01/models/qwen-image/vae/qwen_image_vae.safetensors \
		--height 64 --width 64

test-dit: test_qwen_image
	./test_qwen_image --test-dit /mnt/disk01/models/qwen-image/diffusion-models/qwen-image-Q4_0.gguf \
		--height 64 --width 64

test-enc: test_qwen_image
	./test_qwen_image --test-enc /mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
		--prompt "a red apple on a white table"

test-gen: test_qwen_image
	./test_qwen_image --generate \
		/mnt/disk01/models/qwen-image/diffusion-models/qwen-image-Q4_0.gguf \
		/mnt/disk01/models/qwen-image/vae/qwen_image_vae.safetensors \
		/mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
		--height 256 --width 256 --steps 20 --seed 42 \
		--prompt "a red apple on a white table"

clean:
	rm -f test_qwen_image *.npy *.ppm
