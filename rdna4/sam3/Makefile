# HIP/ROCm SAM 3 runner for RDNA4 (HIPRTC, no hipcc).
CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

ROCEW_SRC  = ../rocew.c
RUNNER_SRC = hip_sam3_runner.c
TEST_SRC   = test_hip_sam3.c
TARGET     = test_hip_sam3
VERIFY_PE  = verify_patch_embed

COMMON_DEPS = hip_sam3_runner.h hip_sam3_kernels.h \
              ../rocew.h ../hip_kernels_common.h ../hip_runner_common.h \
              ../../common/safetensors.h

.PHONY: all clean
all: $(TARGET) $(VERIFY_PE)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

$(VERIFY_PE): verify_patch_embed.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_patch_embed.c $(RUNNER_SRC) $(ROCEW_SRC) $(LDFLAGS)

CKPT   ?= /mnt/disk1/models/sam3/sam3.model.safetensors
REFDIR ?= /tmp/sam3_ref_cat

verify-patch-embed: $(VERIFY_PE)
	./$(VERIFY_PE) --ckpt $(CKPT) --refdir $(REFDIR)

clean:
	rm -f $(TARGET) $(VERIFY_PE)
