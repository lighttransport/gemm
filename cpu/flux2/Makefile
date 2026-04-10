CC ?= gcc
CFLAGS = -O2 -mavx2 -mfma -fopenmp -Wall -Wno-unused-function -I../../common
LDFLAGS = -lm -lpthread -fopenmp

MODEL_DIR = /mnt/disk01/models/klein2-4b
DIT_BASE  = $(MODEL_DIR)/diffusion_models/flux-2-klein-base-4b-fp8.safetensors
DIT_DIST  = $(MODEL_DIR)/diffusion_models/flux-2-klein-4b-fp8.safetensors
VAE       = $(MODEL_DIR)/vae/flux2-vae.safetensors
ENC       = $(MODEL_DIR)/text_encoder
TOK       = /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf

.PHONY: all clean test-sched test-vae test-dit test-enc test-gen-dist test-gen-base

all: test_flux2

test_flux2: test_flux2.c \
		../../common/safetensors.h \
		../../common/gguf_loader.h \
		../../common/ggml_dequant.h \
		../../common/bpe_tokenizer.h \
		../../common/transformer.h \
		../../common/qwen_image_scheduler.h \
		../../common/flux2_klein_dit.h \
		../../common/flux2_klein_vae.h \
		../../common/flux2_klein_text_encoder.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test-sched: test_flux2
	./test_flux2 --test-sched

test-vae: test_flux2
	./test_flux2 --test-vae --vae $(VAE) --height 64 --width 64

test-dit: test_flux2
	./test_flux2 --test-dit --dit $(DIT_DIST) --height 64 --width 64

test-enc: test_flux2
	./test_flux2 --test-enc --enc $(ENC) --tok $(TOK) \
		--prompt "a red apple on a white table"

test-gen-dist: test_flux2
	./test_flux2 --generate \
		--dit $(DIT_DIST) --vae $(VAE) --enc $(ENC) --tok $(TOK) \
		--height 256 --width 256 --steps 4 --seed 42 \
		--prompt "a red apple on a white table"

test-gen-base: test_flux2
	./test_flux2 --generate --base \
		--dit $(DIT_BASE) --vae $(VAE) --enc $(ENC) --tok $(TOK) \
		--height 256 --width 256 --steps 20 --seed 42 \
		--prompt "a red apple on a white table"

clean:
	rm -f test_flux2 *.npy *.ppm
