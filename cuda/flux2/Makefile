CC ?= gcc
CFLAGS = -O2 -mavx2 -mfma -mf16c -fopenmp -Wall -Wno-unused-function -I../../common -I..
LDFLAGS = -lm -ldl -lpthread -fopenmp

MODEL_DIR = /mnt/disk01/models/klein2-4b
DIT_BASE  = $(MODEL_DIR)/diffusion_models/flux-2-klein-base-4b-fp8.safetensors
DIT_DIST  = $(MODEL_DIR)/diffusion_models/flux-2-klein-4b-fp8.safetensors
VAE       = $(MODEL_DIR)/vae/flux2-vae.safetensors
ENC       = $(MODEL_DIR)/text_encoder
TOK       = /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf

.PHONY: all clean test-init test-load test-dit test-vae test-gen-dist test-gen-base

all: test_cuda_flux2

test_cuda_flux2: test_cuda_flux2.c cuda_flux2_runner.h \
		../cuda_runner_common.h ../cuda_kernels_common.h ../cuew.h \
		../llm/cuda_llm_runner.h ../llm/cuda_llm_runner.c \
		../../common/safetensors.h ../../common/gguf_loader.h \
		../../common/ggml_dequant.h ../../common/transformer.h \
		../../common/bpe_tokenizer.h ../../common/qwen_image_scheduler.h \
		../../common/flux2_klein_dit.h ../../common/flux2_klein_vae.h \
		../../common/flux2_klein_text_encoder.h
	$(CC) $(CFLAGS) -o $@ $< ../llm/cuda_llm_runner.c ../cuew.c $(LDFLAGS)

test-init: test_cuda_flux2
	./test_cuda_flux2 --test-init

test-load: test_cuda_flux2
	./test_cuda_flux2 --test-load

test-dit: test_cuda_flux2
	./test_cuda_flux2 --test-dit --height 64 --width 64

test-vae: test_cuda_flux2
	./test_cuda_flux2 --test-vae --height 64 --width 64

test-gen-dist: test_cuda_flux2
	./test_cuda_flux2 --generate \
		--enc $(ENC) --tok $(TOK) \
		--height 256 --width 256 --steps 4 --seed 42 \
		--prompt "a red apple on a white table"

test-gen-base: test_cuda_flux2
	./test_cuda_flux2 --generate --base \
		--enc $(ENC) --tok $(TOK) \
		--height 256 --width 256 --steps 20 --seed 42 \
		--prompt "a red apple on a white table"

clean:
	rm -f test_cuda_flux2 test_fp8_gemm *.ppm *.npy *.bin
