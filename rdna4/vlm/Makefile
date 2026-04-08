# Makefile for HIP/ROCm VLM vision encoder (RDNA4)
#
# No hipcc required - kernels compiled at runtime via HIPRTC.
# Only needs gcc + ROCm runtime libraries (libamdhip64.so, libhiprtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

# Sources
ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_vision_encoder.c
LLM_RUNNER_SRC = ../llm/hip_llm_runner.c

# Targets
VLM = test_hip_vlm

.PHONY: all clean run help

all: $(VLM)

$(VLM): test_hip_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(ROCEW_SRC) hip_vision_encoder.h ../llm/hip_llm_runner.h ../rocew.h
	$(CC) $(CFLAGS) -o $@ test_hip_vlm.c $(RUNNER_SRC) $(LLM_RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

run: $(VLM)
	@echo "Usage: ./$(VLM) <model.gguf> <mmproj.gguf> <image.jpg> [-n max_tokens] [--resize dynamic|fit]"

clean:
	rm -f $(VLM)

help:
	@echo "HIP VLM Vision Encoder (HIPRTC-based, no hipcc needed, RDNA4)"
	@echo "============================================================="
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_hip_vlm"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_hip_vlm <model.gguf> <mmproj.gguf> <image.jpg> [-n max_tokens]"
	@echo ""
	@echo "Requirements:"
	@echo "  - libamdhip64.so (HIP runtime)"
	@echo "  - libhiprtc.so (HIPRTC runtime compiler)"
	@echo "  - AMD GPU with RDNA4 (gfx1200/gfx1201)"
