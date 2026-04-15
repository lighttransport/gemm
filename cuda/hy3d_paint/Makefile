# Makefile for Hunyuan3D-2.1 texture-gen native C/CUDA runners.
#
# Initial iteration: just the xatlas UV unwrap standalone. Neural stages
# remain in ref/hy3d/run_texturegen.py (Python) until individually ported.

CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -I../.. -I../../common
LDFLAGS  = -lpthread

.PHONY: all clean help

all: test_uv_unwrap

test_uv_unwrap: test_uv_unwrap.cc ../../common/xatlas.cc ../../common/xatlas.h
	$(CXX) $(CXXFLAGS) -o $@ test_uv_unwrap.cc ../../common/xatlas.cc $(LDFLAGS)

clean:
	rm -f test_uv_unwrap

help:
	@echo "Hunyuan3D-2.1 texture-gen native runners"
	@echo "========================================"
	@echo ""
	@echo "Build:"
	@echo "  make test_uv_unwrap   - xatlas UV unwrap CLI (C++)"
	@echo ""
	@echo "Run:"
	@echo "  ./test_uv_unwrap /tmp/hy3d_ref.obj /tmp/hy3d_ref_uv.obj"
