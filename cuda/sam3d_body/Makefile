# CUDA SAM 3D Body runner (NVRTC, no nvcc).
#
# sam3d_body_cpu.c is the dedicated TU that supplies the CPU decoder +
# MHR implementations consumed by the runner's MHR-in-the-loop path.

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
              ../../common/sam3d_body_decoder.h ../../common/sam3d_body_mhr.h \
              ../../common/tiny_gltf.h ../../common/stb_image_write.h

CPU_DEPS    = $(CPU_SRC) ../../common/sam3d_body_decoder.h \
              ../../common/sam3d_body_mhr.h

.PHONY: all clean check-backbone check-e2e-image
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
BACKBONE ?= dinov3
REFDIR_DINOV3 ?= /tmp/sam3d_body_ref
VITH_REFDIR ?= /tmp/sam3d_body_vith_ref
REFDIR ?= $(if $(filter vith,$(BACKBONE)),$(VITH_REFDIR),$(REFDIR_DINOV3))
E2E_IMAGE ?=
E2E_BBOX ?=
E2E_THRESHOLD ?= $(if $(filter vith,$(BACKBONE)),1.5e-1,3e-2)
E2E_THRESHOLD_2D ?= $(if $(filter vith,$(BACKBONE)),200,10)

check-backbone:
	@if [ "$(BACKBONE)" != "dinov3" ] && [ "$(BACKBONE)" != "vith" ]; then \
		echo "BACKBONE must be dinov3 or vith"; exit 2; \
	fi

check-e2e-image:
	@if [ -z "$(E2E_IMAGE)" ] || [ -z "$(E2E_BBOX)" ]; then \
		echo "Set E2E_IMAGE=path and E2E_BBOX='x0 y0 x1 y1' for verify-end-to-end"; exit 2; \
	fi

verify-dinov3: check-backbone verify_dinov3
	./verify_dinov3 --safetensors-dir $(SFT) --refdir $(REFDIR_DINOV3)

verify-vith: check-backbone verify_vith
	./verify_vith --safetensors-dir $(SFT) --refdir $(VITH_REFDIR)

verify-ray-cond: check-backbone verify_ray_cond
	./verify_ray_cond --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE)

verify-build-tokens: check-backbone verify_build_tokens
	./verify_build_tokens --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE)

verify-decoder-layer: check-backbone verify_decoder_layer
	./verify_decoder_layer --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE)

verify-kp-update: check-backbone verify_kp_update
	./verify_kp_update --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE)

verify-mhr-head: check-backbone verify_mhr_head
	./verify_mhr_head --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE)

verify-decoder: check-backbone verify_decoder
	./verify_decoder --safetensors-dir $(SFT) --mhr-assets $(MHR) --refdir $(REFDIR) --backbone $(BACKBONE)

verify-mhr: check-backbone verify_mhr
	./verify_mhr --safetensors-dir $(SFT) --mhr-assets-dir $(MHR)

verify-end-to-end: check-backbone check-e2e-image verify_end_to_end
	./verify_end_to_end --safetensors-dir $(SFT) --mhr-assets $(MHR) --refdir $(REFDIR) --image "$(E2E_IMAGE)" --bbox $(E2E_BBOX) --backbone $(BACKBONE) --threshold $(E2E_THRESHOLD) --threshold-2d $(E2E_THRESHOLD_2D)

clean:
	rm -f $(TARGET) $(VERIFIES)
