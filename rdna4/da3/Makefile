# Makefile for HIP/ROCm DA3 depth estimation runner (RDNA4)
#
# No hipcc required - kernels compiled at runtime via HIPRTC.
# Only needs gcc/g++ + ROCm runtime libraries (libamdhip64.so, libhiprtc.so) on the system.

CC = gcc
CXX = g++
CFLAGS = -O3 -Wall -Wextra -I.. -march=native
CXXFLAGS = -O3 -I.. -march=native
LDFLAGS = -ldl -lm -lpthread -lstdc++

# Sources
ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_da3_runner.c
TEST_SRC = test_hip_da3.c
TINYEXR_SRC = ../../common/tinyexr_impl.cc

# Target
TARGET = test_hip_da3

.PHONY: all clean run help

all: $(TARGET)

tinyexr_impl.o: $(TINYEXR_SRC) ../../common/tinyexr.h
	$(CXX) $(CXXFLAGS) -c -o $@ $(TINYEXR_SRC)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) tinyexr_impl.o hip_da3_runner.h ../rocew.h \
           ../hip_kernels_common.h ../hip_runner_common.h \
           ../../common/gguf_loader.h ../../common/ggml_dequant.h ../../common/depth_anything3.h \
           ../../common/cpu_compute.h ../../common/safetensors.h \
           ../../common/tinyexr.h ../../common/exr_writer.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) tinyexr_impl.o $(LDFLAGS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <da3.gguf> [-i input.ppm] [-o output.pgm|output.exr] [-d device_id]"

clean:
	rm -f $(TARGET) tinyexr_impl.o

help:
	@echo "HIP/ROCm DA3 Depth Estimation Runner (HIPRTC-based, no hipcc needed)"
	@echo "====================================================================="
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_hip_da3"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_hip_da3 <da3.gguf> [-i input.ppm] [-o output.pgm|output.exr] [-d device_id]"
	@echo ""
	@echo "Example:"
	@echo "  ./test_hip_da3 da3-small-f16.gguf -i photo.ppm -o depth.exr"
	@echo ""
	@echo "Architecture: GPU backbone (12 ViT blocks) + GPU DPT head (RDNA4)"
	@echo ""
	@echo "Requirements:"
	@echo "  - libamdhip64.so (HIP runtime)"
	@echo "  - libhiprtc.so (HIPRTC runtime compiler)"
	@echo "  - GPU with RDNA4 (gfx1200/gfx1201)"
