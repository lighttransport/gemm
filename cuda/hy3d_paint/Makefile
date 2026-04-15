# Makefile for Hunyuan3D-2.1 texture-gen native C/CUDA runners.
#
# Initial iteration: just the xatlas UV unwrap standalone. Neural stages
# remain in ref/hy3d/run_texturegen.py (Python) until individually ported.

CC       = gcc
CXX      = g++
CFLAGS   = -O3 -Wall -Wextra -I../.. -I../../common -I..
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -I../.. -I../../common -I..
LDFLAGS  = -lpthread
CULDFLAGS= -ldl -lm -lpthread

CUEW_SRC = ../cuew.c

.PHONY: all clean help

all: test_uv_unwrap test_raster test_view_maps test_vertex_inpaint test_back_project test_swiglu

test_uv_unwrap: test_uv_unwrap.cc ../../common/xatlas.cc ../../common/xatlas.h
	$(CXX) $(CXXFLAGS) -o $@ test_uv_unwrap.cc ../../common/xatlas.cc $(LDFLAGS)

test_raster: test_raster.c cuda_paint_raster_kernels.h $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h
	$(CC) $(CFLAGS) -o $@ test_raster.c $(CUEW_SRC) $(CULDFLAGS)

test_view_maps: test_view_maps.c cuda_paint_raster_kernels.h $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h
	$(CC) $(CFLAGS) -o $@ test_view_maps.c $(CUEW_SRC) $(CULDFLAGS)

test_vertex_inpaint: test_vertex_inpaint.cc mesh_vertex_inpaint.h
	$(CXX) $(CXXFLAGS) -o $@ test_vertex_inpaint.cc

test_back_project: test_back_project.c cuda_paint_raster_kernels.h $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h
	$(CC) $(CFLAGS) -o $@ test_back_project.c $(CUEW_SRC) $(CULDFLAGS)

test_swiglu: test_swiglu.c cuda_paint_nn_kernels.h $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h
	$(CC) $(CFLAGS) -o $@ test_swiglu.c $(CUEW_SRC) $(CULDFLAGS)

clean:
	rm -f test_uv_unwrap test_raster test_view_maps test_vertex_inpaint test_back_project test_swiglu

help:
	@echo "Hunyuan3D-2.1 texture-gen native runners"
	@echo "========================================"
	@echo ""
	@echo "Build:"
	@echo "  make test_uv_unwrap   - xatlas UV unwrap CLI (C++)"
	@echo "  make test_raster      - native NVRTC triangle rasterizer (C)"
	@echo ""
	@echo "Run:"
	@echo "  ./test_uv_unwrap /tmp/hy3d_ref.obj /tmp/hy3d_ref_uv.obj"
	@echo "  ./test_raster /tmp/hy3d_ref.obj /tmp/hy3d_raster 512"
