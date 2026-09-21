PYTHON ?= ../../tmp/qimg21-ref-venv/bin/python

.PHONY: all setup generate reference compare regress native native-vae clean

all: test_cuda_qimg21 native

test_cuda_qimg21: test_cuda_qimg21.py
	@chmod +x test_cuda_qimg21.py
	@ln -sf test_cuda_qimg21.py test_cuda_qimg21

native: test_cuda_qimg21_native test_cuda_qimg21_vae

test_cuda_qimg21_native: test_cuda_qimg21_native.c quant_weights.h edit_runtime.h edit_kernels.h mma64_kernels.h joint_layout.h ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c \
		-ldl -lm -lpthread

test_cuda_qimg21_vae: test_cuda_qimg21_vae.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_vae.c ../../cuda/cuew.c ../../cuda/cublasew.c \
		-ldl -lm -lpthread

native-vae: test_cuda_qimg21_vae

test_quant_weights: test_quant_weights.c quant_weights.h ../../common/safetensors.h
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -o $@ test_quant_weights.c -lm

test_joint_layout: test_joint_layout.c joint_layout.h
	$(CC) -O2 -std=c11 -Wall -Wextra -Wpedantic -o $@ test_joint_layout.c

test_edit_kernels: test_edit_kernels.c edit_kernels.h joint_layout.h test_cuda_qimg21_native.c quant_weights.h ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_edit_kernels.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_cuda_qimg21_text: test_cuda_qimg21_text.c text_kernels.h quant_weights.h test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_text.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_bf16_attention: test_bf16_attention.c mma64_kernels.h test_cuda_qimg21_native.c edit_runtime.h edit_kernels.h joint_layout.h ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_bf16_attention.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_scheduler: test_scheduler.c test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_scheduler.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

test_weight_upload: test_weight_upload.c test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_weight_upload.c ../../cuda/cuew.c ../../cuda/cublasew.c -ldl -lm -lpthread

setup:
	./setup_reference.sh

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
	rm -f test_cuda_qimg21 test_cuda_qimg21_native test_cuda_qimg21_vae test_cuda_qimg21_text test_weight_upload test_quant_weights test_joint_layout test_edit_kernels test_scheduler qwen_image21.png
