# Makefile for HIP/ROCm Flux.2 Klein text-to-image runner (RDNA4)
#
# No hipcc required - kernels compiled at runtime via HIPRTC.
# Only needs gcc/g++ + ROCm runtime libraries (libamdhip64.so, libhiprtc.so) on the system.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

# Sources
ROCEW_SRC = ../rocew.c
RUNNER_SRC = hip_flux2_runner.c
TEST_SRC = test_hip_flux2.c

# Target
TARGET = test_hip_flux2
VERIFY_VAE = verify_vae
VERIFY_DIT = verify_dit

COMMON_DEPS = hip_flux2_runner.h hip_flux2_kernels.h \
              ../rocew.h ../hip_kernels_common.h ../hip_runner_common.h \
              ../../common/safetensors.h ../../common/ggml_dequant.h \
              ../../common/flux2_klein_dit.h ../../common/flux2_klein_vae.h \
              ../../common/qwen_image_scheduler.h

.PHONY: all clean run help verify

all: $(TARGET) $(VERIFY_VAE) $(VERIFY_DIT)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_VAE): verify_vae.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_vae.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_DIT): verify_dit.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_dit.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

wmma_selftest: wmma_selftest.c $(ROCEW_SRC) ../rocew.h ../hip_kernels_common.h ../hip_runner_common.h
	$(CC) $(CFLAGS) -o $@ wmma_selftest.c $(ROCEW_SRC) $(LDFLAGS)

verify: $(VERIFY_DIT)
	./$(VERIFY_DIT) --dit $(DIT) --lat 8

# Default weight paths — override on the command line if your files live elsewhere.
DIT ?= /mnt/disk1/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors
VAE ?= /mnt/disk1/models/klein2-4b/vae/diffusion_pytorch_model.safetensors
ENC ?=
TOK ?=
TXT_BIN ?=
PROMPT ?= a red apple on a white table

test-gen: $(TARGET)
	./$(TARGET) --generate --dit $(DIT) --vae $(VAE) \
	  $(if $(ENC),--enc $(ENC) --tok $(TOK)) \
	  $(if $(TXT_BIN),--txt-bin $(TXT_BIN)) \
	  --prompt "$(PROMPT)" --steps 4 --size 256 --seed 42 \
	  -o hip_flux2_gen.ppm

test-verify-dit: $(VERIFY_DIT)
	./$(VERIFY_DIT) --dit $(DIT) --lat 8 --ntxt 16 --threads 16

run: $(TARGET)
	@echo "Usage: ./$(TARGET) --dit <dit.safetensors> [--vae <vae.safetensors>] [--steps N] [-o out.ppm] [-d dev]"

clean:
	rm -f $(TARGET) $(VERIFY_VAE) $(VERIFY_DIT)

help:
	@echo "HIP/ROCm Flux.2 Klein Text-to-Image Runner (HIPRTC-based, no hipcc needed)"
	@echo "==========================================================================="
	@echo ""
	@echo "Build:"
	@echo "  make              - Build test_hip_flux2"
	@echo "  make clean        - Remove built files"
	@echo ""
	@echo "Run:"
	@echo "  ./test_hip_flux2 --dit <dit.safetensors> [--vae <vae.safetensors>]"
	@echo "                   [--steps N] [--size S] [-o out.ppm] [-d dev] [-v]"
	@echo "                   [--test-dit]"
	@echo ""
	@echo "Modes:"
	@echo "  --test-dit        Run single DiT step and print diagnostics"
	@echo "  (default)         Full N-step denoising + optional VAE decode"
	@echo ""
	@echo "Architecture: GPU DiT (5 double + 20 single blocks) on RDNA4"
	@echo ""
	@echo "Requirements:"
	@echo "  - libamdhip64.so (HIP runtime)"
	@echo "  - libhiprtc.so (HIPRTC runtime compiler)"
	@echo "  - GPU with RDNA4 (gfx1200/gfx1201)"
