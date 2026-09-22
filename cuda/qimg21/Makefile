PYTHON ?= ../../tmp/qimg21-ref-venv/bin/python
CUDA_HOME ?= /usr/local/cuda-13.1
PYTORCH_SOURCE ?= ../../tmp/pytorch-src
FLASH_ATTN_SOURCE ?= ../../tmp/flash-attention-src
TORCH_INCLUDE ?= ../../tmp/qimg21-ref-venv/lib/python3.12/site-packages/torch/include

.PHONY: all setup setup-exact-sources verify-exact-sources verify-generated-artifacts generate reference compare regress native native-exact native-text-exact native-vae clean

all: test_cuda_qimg21 native

test_cuda_qimg21: test_cuda_qimg21.py
	@chmod +x test_cuda_qimg21.py
	@ln -sf test_cuda_qimg21.py test_cuda_qimg21

native: test_cuda_qimg21_native test_cuda_qimg21_vae test_cuda_qimg21_vae_encode

native-exact: test_cuda_qimg21_native libq21_cutlass_attention.so libq21_flash_attention.so qwen21_rope_freqs.npy qwen21_text_rope.npy

native-text-exact: test_cuda_qimg21_text libq21_flash_attention.so qwen21_text_rope.npy

libq21_cutlass_attention.so: cutlass_attention_plugin.cu | verify-exact-sources
	@test -f $(PYTORCH_SOURCE)/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h
	@test -f $(FLASH_ATTN_SOURCE)/csrc/cutlass/include/cutlass/cutlass.h
	$(CUDA_HOME)/bin/nvcc -O3 -std=c++17 --expt-relaxed-constexpr --shared -Xcompiler=-fPIC \
		-gencode arch=compute_120,code=sm_120 -I$(PYTORCH_SOURCE) -I$(TORCH_INCLUDE) \
		-I$(FLASH_ATTN_SOURCE)/csrc/cutlass/include -o $@ $<

libq21_flash_attention.so: flash_attention_plugin.cu | verify-exact-sources
	@test -f $(FLASH_ATTN_SOURCE)/csrc/flash_attn/src/flash_fwd_kernel.h
	$(CUDA_HOME)/bin/nvcc -O3 -std=c++17 --expt-relaxed-constexpr --shared -Xcompiler=-fPIC \
		-gencode arch=compute_120,code=sm_120 -I$(FLASH_ATTN_SOURCE) \
		-I$(FLASH_ATTN_SOURCE)/csrc/cutlass/include -o $@ $<

qwen21_rope_freqs.npy: export_rope_frequencies.py
	$(PYTHON) $< --out $@

qwen21_text_rope.npy: export_text_rope.py
	$(PYTHON) $< --out $@

test_cuda_qimg21_native: test_cuda_qimg21_native.c quant_weights.h edit_runtime.h edit_kernels.h mma64_kernels.h norm_vector_kernels.h rope_table_kernels.h joint_layout.h ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c \
		-ldl -lm -lpthread

test_cuda_qimg21_vae: test_cuda_qimg21_vae.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_vae.c ../../cuda/cuew.c ../../cuda/cublasew.c \
		-ldl -lm -lpthread

native-vae: test_cuda_qimg21_vae test_cuda_qimg21_vae_encode

test_cuda_qimg21_vae_encode: test_cuda_qimg21_vae_encode.c test_cuda_qimg21_vae.c ../../common/stb_image.h ../../common/stb_image_resize2.h ../../common/stb_image_write.h ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_vae_encode.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_quant_weights: test_quant_weights.c quant_weights.h ../../common/safetensors.h
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -o $@ test_quant_weights.c -lm

test_joint_layout: test_joint_layout.c joint_layout.h
	$(CC) -O2 -std=c11 -Wall -Wextra -Wpedantic -o $@ test_joint_layout.c

test_edit_kernels: test_edit_kernels.c edit_kernels.h joint_layout.h test_cuda_qimg21_native.c quant_weights.h ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_edit_kernels.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_cuda_qimg21_text: test_cuda_qimg21_text.c qwen_tokenizer_json.h text_kernels.h quant_weights.h test_cuda_qimg21_native.c ../../common/bpe_tokenizer.h ../../common/glm5_bpe.h ../../common/gguf_loader.h ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_text.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_cuda_qimg21_vision: test_cuda_qimg21_vision.c test_cuda_qimg21_native.c quant_weights.h libq21_cutlass_attention.so ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_vision.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_bf16_attention: test_bf16_attention.c mma64_kernels.h test_cuda_qimg21_native.c edit_runtime.h edit_kernels.h joint_layout.h ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_bf16_attention.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_norm_vector: test_norm_vector.c norm_vector_kernels.h test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_norm_vector.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_rotary_arithmetic: test_rotary_arithmetic.c test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_rotary_arithmetic.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_text_projection: test_text_projection.c test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_text_projection.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_tanh_bf16: test_tanh_bf16.c test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_tanh_bf16.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_scheduler: test_scheduler.c test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_scheduler.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_weight_upload: test_weight_upload.c test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_weight_upload.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

setup:
	./setup_reference.sh

setup-exact-sources:
	./setup_exact_sources.sh

verify-exact-sources:
	PYTORCH_SOURCE=$(abspath $(PYTORCH_SOURCE)) FLASH_ATTN_SOURCE=$(abspath $(FLASH_ATTN_SOURCE)) \
		./setup_exact_sources.sh --verify

verify-generated-artifacts: qwen21_rope_freqs.npy qwen21_text_rope.npy
	sha256sum -c generated-artifacts.sha256

generate: test_cuda_qimg21
	$(PYTHON) test_cuda_qimg21.py --generate --model /mnt/nvme01/models/qimg-21 \
		--prompt "a red apple on a white table" --height 1024 --width 1024 \
		--steps 40 --seed 42 --out qwen_image21.png

reference:
	./setup_reference.sh
	$(PYTHON) reference.py --model /mnt/nvme01/models/qimg-21 --prompt "a red apple on a white table" \
		--height 256 --width 256 --steps 1 --seed 42 --dump-dir ../../tmp/qimg21-reference

compare:
	$(PYTHON) compare.py --reference-dir ../../tmp/qimg21-reference --runner-dir ../../tmp/qimg21-runner

regress:
	$(PYTHON) regression.py --model /mnt/nvme01/models/qimg-21

clean:
	rm -f test_cuda_qimg21 test_cuda_qimg21_native test_cuda_qimg21_vae test_cuda_qimg21_vae_encode test_cuda_qimg21_text test_cuda_qimg21_vision test_weight_upload test_quant_weights test_joint_layout test_edit_kernels test_scheduler libq21_cutlass_attention.so libq21_flash_attention.so qwen21_rope_freqs.npy qwen21_text_rope.npy qwen_image21.png
