# CUDA SAM 3 runner (NVRTC, no nvcc).
CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC   = ../cuew.c
RUNNER_SRC = cuda_sam3_runner.c
TEST_SRC   = test_cuda_sam3.c
TARGET     = test_cuda_sam3
VERIFY_PE  = verify_patch_embed
VERIFY_VIT = verify_vit
VERIFY_FPN = verify_fpn
VERIFY_TXT = verify_text
VERIFY_DETR= verify_detr
VERIFY_DDEC= verify_detr_dec
VERIFY_DOT = verify_dot_score
VERIFY_MASK= verify_mask
VERIFY_FINAL= verify_final
VERIFY_MMA = verify_mma_gemm

COMMON_DEPS = cuda_sam3_runner.h cuda_sam3_kernels.h \
              ../cuew.h ../cuda_kernels_common.h ../cuda_runner_common.h ../cuda_hip_compat.h \
              ../../common/safetensors.h

.PHONY: all clean
all: $(TARGET) $(VERIFY_PE) $(VERIFY_VIT) $(VERIFY_FPN) $(VERIFY_TXT) $(VERIFY_DETR) $(VERIFY_DDEC) $(VERIFY_DOT) $(VERIFY_MASK) $(VERIFY_FINAL) $(VERIFY_MMA)

BPE_SRC = ../../cpu/sam3/sam3_clip_bpe.c
$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(BPE_SRC) $(COMMON_DEPS) ../../cpu/sam3/sam3_clip_bpe.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(BPE_SRC) $(LDFLAGS)

$(VERIFY_PE): verify_patch_embed.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_patch_embed.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_VIT): verify_vit.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_vit.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_FPN): verify_fpn.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_fpn.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_TXT): verify_text.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_text.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_DETR): verify_detr.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_detr.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_DDEC): verify_detr_dec.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_detr_dec.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_DOT): verify_dot_score.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_dot_score.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_MASK): verify_mask.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_mask.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_FINAL): verify_final.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_final.c $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

# Self-contained microtest: validates that gemm_f16_f32 (MMA tensor-core) matches
# gemm_tiled_f16_f32 (non-MMA) on sm_120. Does not require the model or ref dumps.
$(VERIFY_MMA): verify_mma_gemm.c $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h ../cuda_kernels_common.h
	$(CC) $(CFLAGS) -o $@ verify_mma_gemm.c $(CUEW_SRC) $(LDFLAGS)

CKPT   ?= /path/to/sam3.model.safetensors
REFDIR ?= /tmp/sam3_ref_cat

verify-patch-embed: $(VERIFY_PE)
	./$(VERIFY_PE) --ckpt $(CKPT) --refdir $(REFDIR)

verify-vit-block0: $(VERIFY_VIT)
	./$(VERIFY_VIT) --ckpt $(CKPT) --refdir $(REFDIR) --target block0

verify-vit-block31: $(VERIFY_VIT)
	./$(VERIFY_VIT) --ckpt $(CKPT) --refdir $(REFDIR) --target block31

verify-vit-final: $(VERIFY_VIT)
	./$(VERIFY_VIT) --ckpt $(CKPT) --refdir $(REFDIR) --target final

verify-fpn: $(VERIFY_FPN)
	./$(VERIFY_FPN) --ckpt $(CKPT) --refdir $(REFDIR)

verify-text: $(VERIFY_TXT)
	./$(VERIFY_TXT) --ckpt $(CKPT) --refdir $(REFDIR)

verify-detr: $(VERIFY_DETR)
	./$(VERIFY_DETR) --ckpt $(CKPT) --refdir $(REFDIR)

verify-detr-dec: $(VERIFY_DDEC)
	./$(VERIFY_DDEC) --ckpt $(CKPT) --refdir $(REFDIR)

clean:
	rm -f $(TARGET) $(VERIFY_PE) $(VERIFY_VIT) $(VERIFY_FPN) $(VERIFY_TXT) \
	      $(VERIFY_DETR) $(VERIFY_DDEC) $(VERIFY_DOT) $(VERIFY_MASK) $(VERIFY_FINAL) \
	      $(VERIFY_MMA)
