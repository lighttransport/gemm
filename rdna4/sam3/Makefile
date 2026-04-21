# HIP/ROCm SAM 3 runner for RDNA4 (HIPRTC, no hipcc).
CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

ROCEW_SRC  = ../rocew.c
RUNNER_SRC = hip_sam3_runner.c
TEST_SRC   = test_hip_sam3.c
TARGET     = test_hip_sam3
VERIFY_PE  = verify_patch_embed
VERIFY_VIT = verify_vit
VERIFY_FPN = verify_fpn
VERIFY_TXT = verify_text
VERIFY_DETR= verify_detr
VERIFY_DDEC= verify_detr_dec
VERIFY_DOT = verify_dot_score
VERIFY_MASK= verify_mask
VERIFY_FINAL= verify_final

COMMON_DEPS = hip_sam3_runner.h hip_sam3_kernels.h \
              ../rocew.h ../hip_kernels_common.h ../hip_runner_common.h \
              ../../common/safetensors.h

.PHONY: all clean
all: $(TARGET) $(VERIFY_PE) $(VERIFY_VIT) $(VERIFY_FPN) $(VERIFY_TXT) $(VERIFY_DETR) $(VERIFY_DDEC) $(VERIFY_DOT) $(VERIFY_MASK) $(VERIFY_FINAL)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_PE): verify_patch_embed.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_patch_embed.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_VIT): verify_vit.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_vit.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_FPN): verify_fpn.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_fpn.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_TXT): verify_text.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_text.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_DETR): verify_detr.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_detr.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_DDEC): verify_detr_dec.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_detr_dec.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_DOT): verify_dot_score.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_dot_score.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_MASK): verify_mask.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_mask.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_FINAL): verify_final.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_final.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

CKPT   ?= /mnt/disk1/models/sam3/sam3.model.safetensors
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
	rm -f $(TARGET) $(VERIFY_PE) $(VERIFY_VIT) $(VERIFY_FPN) $(VERIFY_TXT) $(VERIFY_DETR)
