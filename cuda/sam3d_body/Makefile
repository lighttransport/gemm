# CUDA SAM 3D Body runner (NVRTC, no nvcc).
#
# Step 3a: cuew + NVRTC + safetensors upload wired. Forward kernels
# land in Step 3b. sam3d_body_cpu.c is the dedicated TU that supplies
# the CPU decoder + MHR implementations consumed by the runner's
# run_decoder MHR-in-the-loop path (Step 9).

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I. -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC   = ../cuew.c
RUNNER_SRC = cuda_sam3d_body_runner.c
CPU_SRC    = sam3d_body_cpu.c
TEST_SRC   = test_cuda_sam3d_body.c
TARGET     = test_cuda_sam3d_body

VERIFIES = verify_dinov3 verify_vith verify_ray_cond verify_build_tokens verify_decoder_layer verify_kp_update verify_mhr_head verify_decoder verify_mhr verify_end_to_end

COMMON_DEPS = cuda_sam3d_body_runner.h cuda_sam3d_body_kernels.h \
              ../cuew.h ../cuda_kernels_common.h ../cuda_runner_common.h \
              ../cuda_hip_compat.h ../../common/safetensors.h \
              ../../common/sam3d_body_decoder.h ../../common/sam3d_body_mhr.h

CPU_DEPS    = $(CPU_SRC) ../../common/sam3d_body_decoder.h \
              ../../common/sam3d_body_mhr.h

.PHONY: all clean
all: $(TARGET) $(VERIFIES)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_dinov3: verify_dinov3.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_dinov3.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_vith: verify_vith.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_vith.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_ray_cond: verify_ray_cond.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ray_cond.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_build_tokens: verify_build_tokens.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_build_tokens.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_decoder_layer: verify_decoder_layer.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_decoder_layer.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_kp_update: verify_kp_update.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_kp_update.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_mhr_head: verify_mhr_head.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_mhr_head.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_decoder: verify_decoder.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_decoder.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_mhr: verify_mhr.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_mhr.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_end_to_end: verify_end_to_end.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_end_to_end.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

MODELS ?= /mnt/disk01/models
SFT    ?= $(MODELS)/sam3d-body/safetensors
MHR    ?= $(MODELS)/sam3d-body/safetensors
REFDIR ?= /tmp/sam3d_body_ref
VITH_REFDIR ?= /tmp/sam3d_body_vith_ref

verify-dinov3: verify_dinov3
	./verify_dinov3 --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-vith: verify_vith
	./verify_vith --safetensors-dir $(SFT) --refdir $(VITH_REFDIR)

verify-ray-cond: verify_ray_cond
	./verify_ray_cond --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-build-tokens: verify_build_tokens
	./verify_build_tokens --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-decoder-layer: verify_decoder_layer
	./verify_decoder_layer --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-kp-update: verify_kp_update
	./verify_kp_update --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-mhr-head: verify_mhr_head
	./verify_mhr_head --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-decoder: verify_decoder
	./verify_decoder --safetensors-dir $(SFT) --mhr-assets $(MHR) --refdir $(REFDIR)

clean:
	rm -f $(TARGET) $(VERIFIES)
