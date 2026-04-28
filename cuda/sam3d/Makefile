# CUDA SAM 3D Objects runner (NVRTC, no nvcc).
#
# Phase 0: scaffold.
# Phase 1a: DINOv2 wired via CPU fallback (sam3d_cpu.c) so verify_dinov2
# can pin numerics before per-stage NVRTC kernels land. Subsequent
# phases swap CPU calls for GPU kernels stage by stage.

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I. -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

CUEW_SRC   = ../cuew.c
RUNNER_SRC = cuda_sam3d_runner.c
CPU_SRC    = sam3d_cpu.c
TEST_SRC   = test_cuda_sam3d.c
TARGET     = test_cuda_sam3d

VERIFIES = verify_dinov2 verify_cond_fuser verify_ss_dit verify_ss_decoder \
           verify_slat_dit verify_slat_gs verify_ssdit_block_realw \
           verify_ssdit_full_realw verify_ss_dit_ode \
           verify_slat_transformer_block_realw verify_slat_input_block0_realw \
           verify_slat_input_block1_realw

KERNEL_TESTS = test_layernorm_token test_dinov2_gpu_upload test_dinov2_patch_embed \
               test_dinov2_prepend_pos test_gemm_f32_bias test_qkv_split_sdpa \
               test_layerscale_add test_mlp test_dinov2_block test_dinov2_forward \
               test_silu_mul test_fuser_proj test_ppe_linear test_ppe_window_pack \
               test_sdpa_batched test_ppe_block test_ppe_cls_pos_extract \
               test_ppe_full test_ppe_gpu_load test_timestep_embed \
               test_t_embedder test_adaln_modulation test_modulated_ln \
               test_multi_head_rmsnorm test_mot_self_attn test_mot_cross_attn \
               test_ssdit_ffn test_ssdit_latent_proj test_ssdit_block_forward \
               test_ssdit_gpu_load test_pixel_shuffle_3d test_slat_ape \
               test_slat_downsample test_slat_upsample test_slat_submconv3d \
               test_slat_self_attn test_slat_cross_attn test_slat_resblock \
               test_slat_mlp test_slat_self_attn_residual \
               test_slat_cross_attn_residual test_slat_transformer_block

COMMON_DEPS = cuda_sam3d_runner.h cuda_sam3d_kernels.h sam3d_cpu.h \
              cuda_sam3d_ssdit_gpu.h cuda_sam3d_ssdit_forward.h cuda_sam3d_ssdit_outer.h \
              cuda_sam3d_slat_dit_gpu.h cuda_sam3d_slat_dit_forward.h \
              cuda_sam3d_ss_decoder_gpu.h cuda_sam3d_ss_decoder_forward.h \
              ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h \
              ../../common/safetensors.h ../../common/dinov2.h \
              ../../common/cpu_compute.h ../../common/ggml_dequant.h \
              ../../common/sam3d_cond_fuser.h \
              ../../common/sam3d_ss_flow_dit.h \
              ../../common/sam3d_shortcut_solver.h \
              ../../common/trellis2_ss_decoder.h \
              ../../common/sam3d_slat_dit.h \
              ../../common/sparse3d.h \
              ../../common/sam3d_gs_decoder.h \
              ../../common/gs_ply_writer.h \
              ../../common/stb_image.h \
              ../../common/npy_io.h

CPU_DEPS    = $(CPU_SRC) sam3d_cpu.h ../../common/dinov2.h \
              ../../common/cpu_compute.h ../../common/ggml_dequant.h \
              ../../common/sam3d_cond_fuser.h \
              ../../common/sam3d_ss_flow_dit.h \
              ../../common/sam3d_shortcut_solver.h \
              ../../common/trellis2_ss_decoder.h \
              ../../common/sam3d_slat_dit.h \
              ../../common/sparse3d.h \
              ../../common/sam3d_gs_decoder.h

.PHONY: all clean
all: $(TARGET) $(VERIFIES) $(KERNEL_TESTS)

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_dinov2: verify_dinov2.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_dinov2.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_cond_fuser: verify_cond_fuser.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_cond_fuser.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_ss_dit: verify_ss_dit.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ss_dit.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_ss_decoder: verify_ss_decoder.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ss_decoder.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_slat_dit: verify_slat_dit.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_dit.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_slat_gs: verify_slat_gs.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_gs.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

test_layernorm_token: test_layernorm_token.c cuda_sam3d_kernels.h \
                      ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_layernorm_token.c $(CUEW_SRC) $(LDFLAGS)

test_dinov2_gpu_upload: test_dinov2_gpu_upload.c cuda_sam3d_dinov2_gpu.h \
                       ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h \
                       ../../common/safetensors.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_dinov2_gpu_upload.c $(CUEW_SRC) $(LDFLAGS)

test_dinov2_patch_embed: test_dinov2_patch_embed.c cuda_sam3d_kernels.h \
                        ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_dinov2_patch_embed.c $(CUEW_SRC) $(LDFLAGS)

test_dinov2_prepend_pos: test_dinov2_prepend_pos.c cuda_sam3d_kernels.h \
                        ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_dinov2_prepend_pos.c $(CUEW_SRC) $(LDFLAGS)

test_gemm_f32_bias: test_gemm_f32_bias.c cuda_sam3d_kernels.h \
                    ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_gemm_f32_bias.c $(CUEW_SRC) $(LDFLAGS)

test_qkv_split_sdpa: test_qkv_split_sdpa.c cuda_sam3d_kernels.h \
                     ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_qkv_split_sdpa.c $(CUEW_SRC) $(LDFLAGS)

test_layerscale_add: test_layerscale_add.c cuda_sam3d_kernels.h \
                     ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_layerscale_add.c $(CUEW_SRC) $(LDFLAGS)

test_mlp: test_mlp.c cuda_sam3d_kernels.h \
          ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_mlp.c $(CUEW_SRC) $(LDFLAGS)

test_dinov2_block: test_dinov2_block.c cuda_sam3d_kernels.h cuda_sam3d_dinov2_forward.h \
                   ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ test_dinov2_block.c $(CUEW_SRC) $(LDFLAGS)

test_silu_mul: test_silu_mul.c cuda_sam3d_kernels.h \
               ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_silu_mul.c $(CUEW_SRC) $(LDFLAGS)

test_timestep_embed: test_timestep_embed.c cuda_sam3d_kernels.h \
                     ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_timestep_embed.c $(CUEW_SRC) $(LDFLAGS)

test_t_embedder: test_t_embedder.c cuda_sam3d_kernels.h \
                 ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_t_embedder.c $(CUEW_SRC) $(LDFLAGS)

test_adaln_modulation: test_adaln_modulation.c cuda_sam3d_kernels.h \
                       ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_adaln_modulation.c $(CUEW_SRC) $(LDFLAGS)

test_modulated_ln: test_modulated_ln.c cuda_sam3d_kernels.h \
                   ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_modulated_ln.c $(CUEW_SRC) $(LDFLAGS)

test_multi_head_rmsnorm: test_multi_head_rmsnorm.c cuda_sam3d_kernels.h \
                         ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_multi_head_rmsnorm.c $(CUEW_SRC) $(LDFLAGS)

test_mot_self_attn: test_mot_self_attn.c cuda_sam3d_kernels.h \
                    ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_mot_self_attn.c $(CUEW_SRC) $(LDFLAGS)

test_mot_cross_attn: test_mot_cross_attn.c cuda_sam3d_kernels.h \
                     ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_mot_cross_attn.c $(CUEW_SRC) $(LDFLAGS)

test_ssdit_ffn: test_ssdit_ffn.c cuda_sam3d_kernels.h \
                ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ssdit_ffn.c $(CUEW_SRC) $(LDFLAGS)

test_ssdit_latent_proj: test_ssdit_latent_proj.c cuda_sam3d_kernels.h \
                        ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ssdit_latent_proj.c $(CUEW_SRC) $(LDFLAGS)

test_ssdit_block_forward: test_ssdit_block_forward.c cuda_sam3d_kernels.h \
                          ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ssdit_block_forward.c $(CUEW_SRC) $(LDFLAGS)

test_ssdit_gpu_load: test_ssdit_gpu_load.c cuda_sam3d_ssdit_gpu.h \
                     ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ssdit_gpu_load.c $(CUEW_SRC) $(LDFLAGS)

test_pixel_shuffle_3d: test_pixel_shuffle_3d.c cuda_sam3d_kernels.h \
                      ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_pixel_shuffle_3d.c $(CUEW_SRC) $(LDFLAGS)

test_slat_ape: test_slat_ape.c cuda_sam3d_kernels.h \
               ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_ape.c $(CUEW_SRC) $(LDFLAGS)

test_slat_downsample: test_slat_downsample.c cuda_sam3d_kernels.h \
                      ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_downsample.c $(CUEW_SRC) $(LDFLAGS)

test_slat_upsample: test_slat_upsample.c cuda_sam3d_kernels.h \
                    ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_upsample.c $(CUEW_SRC) $(LDFLAGS)

test_slat_submconv3d: test_slat_submconv3d.c cuda_sam3d_kernels.h \
                      ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_submconv3d.c $(CUEW_SRC) $(LDFLAGS)

test_slat_self_attn: test_slat_self_attn.c cuda_sam3d_kernels.h \
                     ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_self_attn.c $(CUEW_SRC) $(LDFLAGS)

test_slat_cross_attn: test_slat_cross_attn.c cuda_sam3d_kernels.h \
                      ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_cross_attn.c $(CUEW_SRC) $(LDFLAGS)

test_slat_resblock: test_slat_resblock.c cuda_sam3d_kernels.h \
                    ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_resblock.c $(CUEW_SRC) $(LDFLAGS)

test_slat_mlp: test_slat_mlp.c cuda_sam3d_kernels.h \
               ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_mlp.c $(CUEW_SRC) $(LDFLAGS)

test_slat_self_attn_residual: test_slat_self_attn_residual.c cuda_sam3d_kernels.h \
                              ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_self_attn_residual.c $(CUEW_SRC) $(LDFLAGS)

test_slat_cross_attn_residual: test_slat_cross_attn_residual.c cuda_sam3d_kernels.h \
                               ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_cross_attn_residual.c $(CUEW_SRC) $(LDFLAGS)

test_slat_transformer_block: test_slat_transformer_block.c cuda_sam3d_kernels.h \
                             ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_transformer_block.c $(CUEW_SRC) $(LDFLAGS)

verify_slat_transformer_block_realw: verify_slat_transformer_block_realw.c cuda_sam3d_kernels.h \
                                     cuda_sam3d_slat_dit_gpu.h cuda_sam3d_slat_dit_forward.h \
                                     ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h \
                                     ../../common/sam3d_slat_dit.h ../../common/sam3d_ss_flow_dit.h \
                                     ../../common/safetensors.h ../../common/npy_io.h \
                                     $(RUNNER_SRC) $(CPU_DEPS) $(CUEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_transformer_block_realw.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_slat_input_block0_realw: verify_slat_input_block0_realw.c cuda_sam3d_kernels.h \
                                ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h \
                                ../../common/sam3d_slat_dit.h ../../common/npy_io.h \
                                $(RUNNER_SRC) $(CPU_DEPS) $(CUEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_input_block0_realw.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_slat_input_block1_realw: verify_slat_input_block1_realw.c cuda_sam3d_kernels.h \
                                ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h \
                                ../../common/sam3d_slat_dit.h ../../common/npy_io.h \
                                $(RUNNER_SRC) $(CPU_DEPS) $(CUEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_input_block1_realw.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

verify_ssdit_block_realw: verify_ssdit_block_realw.c cuda_sam3d_ssdit_gpu.h \
                          cuda_sam3d_ssdit_forward.h cuda_sam3d_kernels.h \
                          ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ verify_ssdit_block_realw.c $(CUEW_SRC) $(LDFLAGS)

verify_ssdit_full_realw: verify_ssdit_full_realw.c cuda_sam3d_ssdit_gpu.h \
                         cuda_sam3d_ssdit_forward.h cuda_sam3d_ssdit_outer.h \
                         cuda_sam3d_kernels.h \
                         ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ssdit_full_realw.c $(CUEW_SRC) $(LDFLAGS)

verify_ss_dit_ode: verify_ss_dit_ode.c $(RUNNER_SRC) $(CUEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ss_dit_ode.c $(RUNNER_SRC) $(CPU_SRC) $(CUEW_SRC) $(LDFLAGS)

test_fuser_proj: test_fuser_proj.c cuda_sam3d_kernels.h \
                 ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_fuser_proj.c $(CUEW_SRC) $(LDFLAGS)

test_ppe_linear: test_ppe_linear.c cuda_sam3d_kernels.h \
                 ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_linear.c $(CUEW_SRC) $(LDFLAGS)

test_ppe_window_pack: test_ppe_window_pack.c cuda_sam3d_kernels.h \
                      ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_window_pack.c $(CUEW_SRC) $(LDFLAGS)

test_sdpa_batched: test_sdpa_batched.c cuda_sam3d_kernels.h \
                   ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_sdpa_batched.c $(CUEW_SRC) $(LDFLAGS)

test_ppe_block: test_ppe_block.c cuda_sam3d_kernels.h \
                ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_block.c $(CUEW_SRC) $(LDFLAGS)

test_ppe_cls_pos_extract: test_ppe_cls_pos_extract.c cuda_sam3d_kernels.h \
                          ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_cls_pos_extract.c $(CUEW_SRC) $(LDFLAGS)

test_ppe_full: test_ppe_full.c cuda_sam3d_kernels.h cuda_sam3d_ppe_forward.h \
               ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_full.c $(CUEW_SRC) $(LDFLAGS)

test_ppe_gpu_load: test_ppe_gpu_load.c cuda_sam3d_kernels.h cuda_sam3d_ppe_forward.h \
                   cuda_sam3d_ppe_gpu.h ../cuew.h ../cuda_runner_common.h \
                   ../cuda_hip_compat.h ../../common/safetensors.h \
                   ../../common/sam3d_cond_fuser.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ test_ppe_gpu_load.c $(CUEW_SRC) $(LDFLAGS)

test_dinov2_forward: test_dinov2_forward.c cuda_sam3d_kernels.h cuda_sam3d_dinov2_forward.h \
                     cuda_sam3d_dinov2_gpu.h \
                     ../cuew.h ../cuda_runner_common.h ../cuda_hip_compat.h \
                     ../../common/safetensors.h ../../common/dinov2.h \
                     ../../common/cpu_compute.h ../../common/ggml_dequant.h $(CUEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ test_dinov2_forward.c $(CUEW_SRC) $(LDFLAGS)

MODELS ?= /mnt/disk01/models
SFT    ?= $(MODELS)/sam3d/safetensors
YAML   ?= $(MODELS)/sam3d/checkpoints/pipeline.yaml
REFDIR ?= /tmp/sam3d_ref

smoke: $(TARGET)
	./$(TARGET) --safetensors-dir $(SFT) -v

verify-dinov2: verify_dinov2
	./verify_dinov2 --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-cond-fuser: verify_cond_fuser
	./verify_cond_fuser --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-ss-dit: verify_ss_dit
	./verify_ss_dit --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-ss-decoder: verify_ss_decoder
	./verify_ss_decoder --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-slat-dit: verify_slat_dit
	./verify_slat_dit --safetensors-dir $(SFT) --refdir $(REFDIR)

verify-slat-gs: verify_slat_gs
	./verify_slat_gs --safetensors-dir $(SFT) --refdir $(REFDIR)

clean:
	rm -f $(TARGET) $(VERIFIES) $(KERNEL_TESTS)
