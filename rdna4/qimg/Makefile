# Makefile for HIP/ROCm Qwen-Image text-to-image runner (RDNA4)
#
# No hipcc required - kernels compiled at runtime via HIPRTC.
# Only needs gcc/g++ + ROCm runtime libraries (libamdhip64.so, libhiprtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

# Sources
ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_qimg_runner.c
LLM_SRC = ../llm/hip_llm_runner.c
TEST_SRC = test_hip_qimg.c

# Target
TARGET = test_hip_qimg
VERIFY_DIT = verify_qimg_dit

COMMON_DEPS = hip_qimg_runner.h hip_qimg_kernels.h \
              ../rocew.h ../hip_kernels_common.h ../hip_runner_common.h \
              ../../common/safetensors.h ../../common/ggml_dequant.h \
              ../../common/gguf_loader.h ../../common/bpe_tokenizer.h \
              ../../common/transformer.h ../../common/qwen_image_text_encoder.h \
              ../../common/qwen_image_dit.h ../../common/qwen_image_vae.h \
              ../../common/qwen_image_scheduler.h \
              ../llm/hip_llm_runner.h ../llm/hip_llm_runner.c

.PHONY: all clean run help

all: $(TARGET) $(VERIFY_DIT)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(LLM_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_DIT): verify_qimg_dit.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_qimg_dit.c $(RUNNER_SRC) $(LLM_SRC) $(ROCEW_SRC) $(LDFLAGS)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) --dit <dit.safetensors> [--vae <vae.safetensors>] [--steps N] [-o out.ppm] [-d dev]"

clean:
	rm -f $(TARGET) $(VERIFY_DIT)

help:
	@echo "HIP/ROCm Qwen-Image Text-to-Image Runner (HIPRTC-based, no hipcc needed)"
	@echo "========================================================================"
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_hip_qimg"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_hip_qimg --dit <dit.safetensors> [--vae <vae.safetensors>]"
	@echo "                  [--height H] [--width W] [--steps N] [--seed S]"
	@echo "                  [-o out.ppm] [-d dev] [-v]"
	@echo ""
	@echo "Modes:"
	@echo "  --test-init       Initialize HIP, compile kernels"
	@echo "  --test-dit        Run single DiT step and print diagnostics"
	@echo "  --test-vae        Run VAE decode test"
	@echo "  --generate        Full N-step denoising + optional VAE decode"
	@echo ""
	@echo "Architecture: GPU DiT (60 dual-stream blocks) on RDNA4"
	@echo ""
	@echo "Requirements:"
	@echo "  - libamdhip64.so (HIP runtime)"
	@echo "  - libhiprtc.so (HIPRTC runtime compiler)"
	@echo "  - GPU with RDNA4 (gfx1200/gfx1201)"
