# CPU runner for FAIR's SAM 3D Body.
# Builds DINOv3-H+ and ViT-H verifier coverage plus the raw-image runner.

COMPILER ?= gcc
ARCH ?= zen2
COMMON = ../../common

ifeq ($(ARCH),zen2)
    ARCH_FLAGS = -march=znver2 -mtune=znver2
else ifeq ($(ARCH),zen3)
    ARCH_FLAGS = -march=znver3 -mtune=znver3
else ifeq ($(ARCH),zen4)
    ARCH_FLAGS = -march=znver4 -mtune=znver4
else ifeq ($(ARCH),native)
    ARCH_FLAGS = -march=native
else
    ARCH_FLAGS = -march=x86-64-v3
endif

ifeq ($(COMPILER),gcc)
    CC = gcc
else ifeq ($(COMPILER),clang)
    CC = clang
else
    $(error Unknown COMPILER=$(COMPILER). Use gcc or clang)
endif

CFLAGS  = -O3 $(ARCH_FLAGS) -mavx2 -mfma -mf16c -ffast-math
CFLAGS += -funroll-loops -fomit-frame-pointer -Wall -Wextra -fopenmp
LDFLAGS = -lm -lpthread -fopenmp

ifdef DEBUG
    CFLAGS  = -O0 -g $(ARCH_FLAGS) -mavx2 -mfma -mf16c -Wall -Wextra -fopenmp
    CFLAGS += -fsanitize=address -fsanitize=undefined
    LDFLAGS += -fsanitize=address -fsanitize=undefined
endif

HEADERS    = $(wildcard $(COMMON)/*.h) sam3d_body_runner.h
RUNNER_SRC = sam3d_body_runner.c

VERIFIES = verify_dinov3 verify_vith verify_ray_cond verify_ray_cond_xyz verify_dense_pe verify_warp_matrix verify_preprocess_image verify_tokens verify_decoder_layer verify_kp_update verify_decoder_e2e verify_camera_project verify_mhr_head verify_mhr_head_decode verify_mhr_stages verify_decoder_full verify_decoder_forward verify_decoder verify_mhr verify_end_to_end verify_rt_detr_preprocess verify_rt_detr_backbone verify_rt_detr_encoder verify_rt_detr_decoder verify_rt_detr_postprocess verify_rt_detr_e2e

.PHONY: all clean check-backbone check-e2e-image
all: test_sam3d_body $(VERIFIES)

test_sam3d_body: test_sam3d_body.c $(RUNNER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ test_sam3d_body.c $(RUNNER_SRC) $(LDFLAGS)

verify_dinov3: verify_dinov3.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_dinov3.c $(LDFLAGS)

verify_vith: verify_vith.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_vith.c $(LDFLAGS)

verify_ray_cond: verify_ray_cond.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_ray_cond.c $(LDFLAGS)

verify_dense_pe: verify_dense_pe.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_dense_pe.c $(LDFLAGS)

verify_ray_cond_xyz: verify_ray_cond_xyz.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_ray_cond_xyz.c $(LDFLAGS)

verify_warp_matrix: verify_warp_matrix.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_warp_matrix.c $(LDFLAGS)

verify_preprocess_image: verify_preprocess_image.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_preprocess_image.c $(LDFLAGS)

verify_tokens: verify_tokens.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_tokens.c $(LDFLAGS)

verify_decoder_layer: verify_decoder_layer.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_decoder_layer.c $(LDFLAGS)

verify_kp_update: verify_kp_update.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_kp_update.c $(LDFLAGS)

verify_decoder_e2e: verify_decoder_e2e.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_decoder_e2e.c $(LDFLAGS)

verify_camera_project: verify_camera_project.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_camera_project.c $(LDFLAGS)

verify_mhr_head: verify_mhr_head.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_mhr_head.c $(LDFLAGS)

verify_mhr_stages: verify_mhr_stages.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_mhr_stages.c $(LDFLAGS)

verify_mhr_head_decode: verify_mhr_head_decode.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_mhr_head_decode.c $(LDFLAGS)

verify_decoder_full: verify_decoder_full.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_decoder_full.c $(LDFLAGS)

verify_decoder_forward: verify_decoder_forward.c $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_decoder_forward.c $(LDFLAGS)

verify_decoder: verify_decoder.c
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_decoder.c $(LDFLAGS)

verify_mhr: verify_mhr.c
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_mhr.c $(LDFLAGS)

verify_end_to_end: verify_end_to_end.c $(RUNNER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_end_to_end.c $(RUNNER_SRC) $(LDFLAGS)

verify_rt_detr_preprocess: verify_rt_detr_preprocess.c $(COMMON)/rt_detr.h $(COMMON)/safetensors.h $(COMMON)/npy_io.h $(COMMON)/stb_image.h
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_rt_detr_preprocess.c $(LDFLAGS)

verify_rt_detr_backbone: verify_rt_detr_backbone.c $(COMMON)/rt_detr.h $(COMMON)/safetensors.h $(COMMON)/npy_io.h
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_rt_detr_backbone.c $(LDFLAGS)

verify_rt_detr_encoder: verify_rt_detr_encoder.c $(COMMON)/rt_detr.h $(COMMON)/safetensors.h $(COMMON)/npy_io.h
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_rt_detr_encoder.c $(LDFLAGS)

verify_rt_detr_decoder: verify_rt_detr_decoder.c $(COMMON)/rt_detr.h $(COMMON)/safetensors.h $(COMMON)/npy_io.h
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_rt_detr_decoder.c $(LDFLAGS)

verify_rt_detr_postprocess: verify_rt_detr_postprocess.c $(COMMON)/rt_detr.h $(COMMON)/safetensors.h $(COMMON)/npy_io.h
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_rt_detr_postprocess.c $(LDFLAGS)

verify_rt_detr_e2e: verify_rt_detr_e2e.c $(COMMON)/rt_detr.h $(COMMON)/safetensors.h $(COMMON)/npy_io.h $(COMMON)/stb_image.h
	$(CC) $(CFLAGS) -I$(COMMON) -I. -o $@ verify_rt_detr_e2e.c $(LDFLAGS)

# Convenience targets for running verifies against a ref dump.
MODELS ?= /mnt/disk01/models
SFT    ?= $(MODELS)/sam3d-body/safetensors
MHR    ?= $(SFT)
BACKBONE ?= dinov3
REFDIR_DINOV3 ?= /tmp/sam3d_body_ref
REFDIR_VITH ?= /tmp/sam3d_body_vith_ref
REFDIR ?= $(if $(filter vith,$(BACKBONE)),$(REFDIR_VITH),$(REFDIR_DINOV3))
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
		echo "Set E2E_IMAGE=path and E2E_BBOX='x0 y0 x1 y1' for verify-end-to-end-raw"; exit 2; \
	fi

verify-dinov3: check-backbone verify_dinov3
	./verify_dinov3 --safetensors-dir $(SFT) --refdir $(REFDIR_DINOV3)
verify-vith: check-backbone verify_vith
	./verify_vith --safetensors-dir $(SFT) --refdir $(REFDIR_VITH) -t 8
verify-ray-cond: check-backbone verify_ray_cond
	./verify_ray_cond --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE)
verify-dense-pe: check-backbone verify_dense_pe
	./verify_dense_pe --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE)
verify-ray-cond-xyz: check-backbone verify_ray_cond_xyz
	./verify_ray_cond_xyz --refdir $(REFDIR)
verify-warp-matrix: check-backbone verify_warp_matrix
	./verify_warp_matrix --refdir $(REFDIR)
verify-preprocess-image: check-backbone verify_preprocess_image
	./verify_preprocess_image --refdir $(REFDIR)
verify-tokens: check-backbone verify_tokens
	./verify_tokens --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE)
verify-decoder-layer: check-backbone verify_decoder_layer
	./verify_decoder_layer --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE) -t 8
verify-kp-update: check-backbone verify_kp_update
	./verify_kp_update --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE) -t 8
verify-decoder-e2e: check-backbone verify_decoder_e2e
	./verify_decoder_e2e --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE) -t 8
verify-camera-project: check-backbone verify_camera_project
	./verify_camera_project --refdir $(REFDIR)
verify-mhr-head: check-backbone verify_mhr_head
	./verify_mhr_head --safetensors-dir $(SFT) --refdir $(REFDIR) --backbone $(BACKBONE) -t 8
verify-mhr-stages: check-backbone verify_mhr_stages
	./verify_mhr_stages --mhr-assets $(MHR) --refdir $(REFDIR) -t 8
verify-decoder-full: check-backbone verify_decoder_full
	./verify_decoder_full --safetensors-dir $(SFT) --mhr-assets $(MHR) --refdir $(REFDIR) --backbone $(BACKBONE) -t 8
verify-decoder-forward: check-backbone verify_decoder_forward
	./verify_decoder_forward --safetensors-dir $(SFT) --mhr-assets $(MHR) --refdir $(REFDIR) --backbone $(BACKBONE) -t 8
verify-decoder: check-backbone verify_decoder_full
	./verify_decoder_full --safetensors-dir $(SFT) --mhr-assets $(MHR) --refdir $(REFDIR) --backbone $(BACKBONE) -t 8
verify-mhr: check-backbone verify_mhr_stages
	./verify_mhr_stages --mhr-assets $(MHR) --refdir $(REFDIR) -t 8
verify-end-to-end: check-backbone verify_end_to_end
	./verify_end_to_end --safetensors-dir $(SFT) --mhr-assets $(MHR) --refdir $(REFDIR) --backbone $(BACKBONE)
verify-end-to-end-raw: check-backbone check-e2e-image verify_end_to_end
	./verify_end_to_end --safetensors-dir $(SFT) --mhr-assets $(MHR) --refdir $(REFDIR) --image "$(E2E_IMAGE)" --bbox $(E2E_BBOX) --backbone $(BACKBONE) --threshold $(E2E_THRESHOLD) --threshold-2d $(E2E_THRESHOLD_2D)

clean:
	rm -f test_sam3d_body $(VERIFIES)
