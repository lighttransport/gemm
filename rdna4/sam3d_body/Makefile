# HIP/ROCm SAM 3D Body runner (RDNA4, HIPRTC, no hipcc).
#
# Port of cuda/sam3d_body/Makefile. CPU helpers (sam3d_body_cpu.c)
# supply the decoder + MHR implementations consumed by the runner's
# MHR-in-the-loop path.

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I. -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

ROCEW_SRC  = ../rocew.c
RUNNER_SRC = hip_sam3d_body_runner.c
CPU_SRC    = sam3d_body_cpu.c
TEST_SRC   = test_hip_sam3d_body.c
TARGET     = test_hip_sam3d_body

VERIFIES = verify_dinov3 verify_vith verify_ray_cond verify_build_tokens \
           verify_decoder_layer verify_kp_update verify_mhr_head \
           verify_decoder verify_mhr verify_end_to_end

COMMON_DEPS = hip_sam3d_body_runner.h hip_sam3d_body_kernels.h \
              ../rocew.h ../hip_kernels_common.h ../hip_runner_common.h \
              ../../common/safetensors.h \
              ../../common/sam3d_body_decoder.h \
              ../../common/sam3d_body_mhr.h

CPU_DEPS    = $(CPU_SRC) ../../common/sam3d_body_decoder.h \
              ../../common/sam3d_body_mhr.h

.PHONY: all clean
all: $(TARGET) $(VERIFIES)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_dinov3: verify_dinov3.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_dinov3.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_vith: verify_vith.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_vith.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_ray_cond: verify_ray_cond.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ray_cond.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_build_tokens: verify_build_tokens.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_build_tokens.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_decoder_layer: verify_decoder_layer.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_decoder_layer.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_kp_update: verify_kp_update.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_kp_update.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_mhr_head: verify_mhr_head.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_mhr_head.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_decoder: verify_decoder.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_decoder.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_mhr: verify_mhr.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_mhr.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_end_to_end: verify_end_to_end.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_end_to_end.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

MODELS ?= /mnt/disk1/models
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
