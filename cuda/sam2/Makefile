CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

RUNNER_SRC = cuda_sam2_runner.c
TEST_SRC   = test_cuda_sam2.c
CUEW_SRC   = ../cuew.c
TARGET     = test_cuda_sam2
VERIFY_PATCH = verify_patch_embed
VERIFY_B0    = verify_block0
VERIFY_B012  = verify_blocks012
VERIFY_BB    = verify_backbone
VERIFY_FPN   = verify_fpn
VERIFY_PR    = verify_prompt
VERIFY_MD    = verify_mask_decoder

.PHONY: all clean
all: $(TARGET) $(VERIFY_PATCH) $(VERIFY_B0) $(VERIFY_B012) $(VERIFY_BB) $(VERIFY_FPN) $(VERIFY_PR) $(VERIFY_MD)

$(VERIFY_PR): verify_prompt.c ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ verify_prompt.c -lm

$(VERIFY_MD): verify_mask_decoder.c ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ verify_mask_decoder.c -lm

VERIFY_MD_CUDA = verify_mask_decoder_cuda
$(VERIFY_MD_CUDA): verify_mask_decoder_cuda.c $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ verify_mask_decoder_cuda.c $(CUEW_SRC) $(LDFLAGS)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) cuda_sam2_runner.h ../cuew.h ../cuda_runner_common.h ../cuda_kernels_common.h
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_PATCH): verify_patch_embed.c $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ verify_patch_embed.c $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_B0): verify_block0.c $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ verify_block0.c $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_B012): verify_blocks012.c $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ verify_blocks012.c $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_BB): verify_backbone.c $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ verify_backbone.c $(CUEW_SRC) $(LDFLAGS)

$(VERIFY_FPN): verify_fpn.c $(CUEW_SRC) ../cuew.h ../cuda_runner_common.h ../../common/safetensors.h
	$(CC) $(CFLAGS) -o $@ verify_fpn.c $(CUEW_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET) $(VERIFY_PATCH) $(VERIFY_B0) $(VERIFY_B012)
