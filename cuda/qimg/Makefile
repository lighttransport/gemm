CC ?= gcc
CFLAGS = -O2 -Wall -Wno-unused-function -I../../common -I..
LDFLAGS = -lm -ldl -lpthread

.PHONY: all clean test-init test-load

all: test_cuda_qimg

test_cuda_qimg: test_cuda_qimg.c cuda_qimg_runner.h \
		../cuda_runner_common.h ../cuda_kernels_common.h ../cuew.h
	$(CC) $(CFLAGS) -o $@ $< ../cuew.c $(LDFLAGS)

test-init: test_cuda_qimg
	./test_cuda_qimg --test-init

test-load: test_cuda_qimg
	./test_cuda_qimg --test-load

clean:
	rm -f test_cuda_qimg
