CC ?= gcc
CXX ?= g++
PYTHON ?= python3
CFLAGS = -O2 -Wall -Wextra -fPIC
CXXFLAGS = $(CFLAGS) -std=c++17
LDFLAGS = -ldl -lm

.PHONY: all test test-host clean
all: libgenia_cuda.so test_genia

genia_kernels.inc: embed_kernels.py genia_kernels.cuh ../../common/genia_math.h ../../common/genia_gaussian.h
	$(PYTHON) embed_kernels.py $@

cuew.o: ../cuew.c ../cuew.h
	$(CC) $(CFLAGS) -c $< -o $@

genia_cuda.o: genia_cuda.cpp genia_cuda.h genia_kernels.inc ../cuda_runner_common.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

libgenia_cuda.so: genia_cuda.o cuew.o
	$(CXX) -shared $^ $(LDFLAGS) -o $@

test_genia: test_genia.cpp genia_cuda.o cuew.o ../../common/genia_math.h ../../common/sam3d_pose_tokens.h
	$(CXX) $(CXXFLAGS) $< genia_cuda.o cuew.o $(LDFLAGS) -o $@

test: test_genia
	./test_genia --gpu

test-host: test_genia
	./test_genia

clean:
	rm -f *.o libgenia_cuda.so test_genia genia_kernels.inc
