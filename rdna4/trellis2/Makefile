# Makefile for HIP/ROCm TRELLIS.2 Stage 1 runner (RDNA4)
#
# No hipcc required — kernels compiled at runtime via HIPRTC.
# Only needs gcc + ROCm runtime (libamdhip64.so, libhiprtc.so) on PATH.

CC     = gcc
CFLAGS = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

ROCEW_SRC   = ../rocew.c
RUNNER_SRC  = hip_trellis2_runner.c
TEST_SRC    = test_hip_trellis2.c

TARGET       = test_hip_trellis2
VERIFY_DIT   = verify_dit
VERIFY_DEC   = verify_decoder

COMMON_DEPS = hip_trellis2_runner.h hip_trellis2_kernels.h \
              ../rocew.h ../hip_kernels_common.h ../hip_runner_common.h \
              ../../common/safetensors.h ../../common/ggml_dequant.h \
              ../../common/trellis2_dit.h ../../common/trellis2_ss_decoder.h

.PHONY: all clean run help verify-dit verify-dec

all: $(TARGET) $(VERIFY_DIT) $(VERIFY_DEC)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_DIT): verify_dit.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_dit.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_DEC): verify_decoder.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_decoder.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

# Default weight paths — override on the command line
DIT     ?= /mnt/disk1/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors
DECODER ?= /mnt/disk1/models/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors
FEATURES ?= /tmp/t2ref/ref_features.npy
NOISE    ?= /tmp/t2ref/ref_noise_init.npy
OUTDIR   ?= /tmp/t2hip

verify-dit: $(VERIFY_DIT)
	./$(VERIFY_DIT) --dit $(DIT) --threads 16

verify-dec: $(VERIFY_DEC)
	./$(VERIFY_DEC) --decoder $(DECODER) --threads 16

dump-blocks: $(TARGET)
	mkdir -p $(OUTDIR)
	./$(TARGET) --dit $(DIT) --decoder $(DECODER) \
	  --features $(FEATURES) --noise $(NOISE) \
	  --dump-blocks --output-dir $(OUTDIR)

full: $(TARGET)
	mkdir -p $(OUTDIR)
	./$(TARGET) --dit $(DIT) --decoder $(DECODER) \
	  --features $(FEATURES) --noise $(NOISE) \
	  --full --output-dir $(OUTDIR) --steps 12

compare-blocks:
	@echo "Comparing block hidden states with PyTorch reference..."
	@for N in $$(seq -f '%02g' 0 29); do \
	  python3 ../../ref/trellis2/make_comparison.py \
	    /tmp/t2ref/ref_block$${N}_hidden.npy \
	    $(OUTDIR)/hip_block$${N}_hidden.npy \
	    --label "block $$N" || true; \
	done

clean:
	rm -f $(TARGET) $(VERIFY_DIT) $(VERIFY_DEC)

help:
	@echo "HIP/ROCm TRELLIS.2 Stage 1 Runner (HIPRTC-based, no hipcc needed)"
	@echo "==================================================================="
	@echo ""
	@echo "Build:"
	@echo "  make                  - Build all (test + verify_dit + verify_decoder)"
	@echo "  make clean            - Remove built files"
	@echo ""
	@echo "Verify GPU vs CPU:"
	@echo "  make verify-dit       - Compare HIP DiT vs CPU reference"
	@echo "  make verify-dec       - Compare HIP decoder vs CPU reference"
	@echo ""
	@echo "End-to-end test:"
	@echo "  make full             - Full 12-step sampling + decode -> occupancy.npy"
	@echo ""
	@echo "Layer-by-layer comparison workflow:"
	@echo "  # Step 1: Generate PyTorch reference"
	@echo "  cd ../../ref/trellis2"
	@echo "  python gen_stage1_ref.py --features ... --stage1 $(DIT) --output-dir /tmp/t2ref"
	@echo "  python dump_dit_intermediates.py --features /tmp/t2ref/ref_features.npy \\"
	@echo "    --noise /tmp/t2ref/ref_noise_init.npy --stage1 $(DIT) --output-dir /tmp/t2ref"
	@echo ""
	@echo "  # Step 2: Dump HIP hidden states"
	@echo "  make dump-blocks FEATURES=/tmp/t2ref/ref_features.npy NOISE=/tmp/t2ref/ref_noise_init.npy"
	@echo ""
	@echo "  # Step 3: Compare"
	@echo "  make compare-blocks"
	@echo ""
	@echo "Override defaults:"
	@echo "  make verify-dit DIT=/path/to/dit.safetensors"
	@echo "  make full FEATURES=/path/features.npy NOISE=/path/noise.npy"
