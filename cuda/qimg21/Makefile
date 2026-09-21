PYTHON ?= ../../tmp/qimg21-ref-venv/bin/python

.PHONY: all setup generate reference compare regress native native-vae clean

all: test_cuda_qimg21 native

test_cuda_qimg21: test_cuda_qimg21.py
	@chmod +x test_cuda_qimg21.py
	@ln -sf test_cuda_qimg21.py test_cuda_qimg21

native: test_cuda_qimg21_native test_cuda_qimg21_vae

test_cuda_qimg21_native: test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_native.c ../../cuda/cuew.c ../../cuda/cublasew.c \
		-ldl -lm -lpthread

test_cuda_qimg21_vae: test_cuda_qimg21_vae.c ../../cuda/cuew.c ../../cuda/cublasew.c
	$(CC) -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra -I../../common -I../../cuda -I../qimg \
		-o $@ test_cuda_qimg21_vae.c ../../cuda/cuew.c ../../cuda/cublasew.c \
		-ldl -lm -lpthread

native-vae: test_cuda_qimg21_vae

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
	rm -f test_cuda_qimg21 test_cuda_qimg21_native test_cuda_qimg21_vae qwen_image21.png
