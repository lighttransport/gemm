# HIP/ROCm SAM 3D Objects (RDNA4) runner (HIPRTC, no hipcc).
#
# Phase 0: scaffold.
# Phase 1a: DINOv2 wired via CPU fallback (sam3d_cpu.c) so verify_dinov2
# can pin numerics before per-stage HIPRTC kernels land. Subsequent
# phases swap CPU calls for GPU kernels stage by stage.

CC      = gcc
CFLAGS  = -O3 -Wall -Wextra -I. -I.. -march=native
LDFLAGS = -ldl -lm -lpthread

ROCEW_SRC   = ../rocew.c
RUNNER_SRC = hip_sam3d_runner.c
CPU_SRC    = sam3d_cpu.c
TEST_SRC   = test_hip_sam3d.c
TARGET     = test_hip_sam3d

VERIFIES = verify_dinov2 verify_cond_fuser verify_ss_dit verify_ss_decoder \
           verify_slat_dit verify_slat_gs verify_ssdit_block_realw \
           verify_ssdit_full_realw verify_ss_dit_ode \
           verify_slat_transformer_block_realw verify_slat_input_block0_realw \
           verify_slat_input_block1_realw verify_slat_out_block0_realw \
           verify_slat_out_block1_realw

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

COMMON_DEPS = hip_sam3d_runner.h hip_sam3d_kernels.h sam3d_cpu.h \
              hip_sam3d_ssdit_gpu.h hip_sam3d_ssdit_forward.h hip_sam3d_ssdit_outer.h \
              hip_sam3d_slat_dit_gpu.h hip_sam3d_slat_dit_forward.h \
              hip_sam3d_ss_decoder_gpu.h hip_sam3d_ss_decoder_forward.h \
              ../rocew.h ../hip_runner_common.h  \
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

$(TARGET): $(TEST_SRC) $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ $(TEST_SRC) $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_dinov2: verify_dinov2.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_dinov2.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_cond_fuser: verify_cond_fuser.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_cond_fuser.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_ss_dit: verify_ss_dit.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ss_dit.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_ss_decoder: verify_ss_decoder.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ss_decoder.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_slat_dit: verify_slat_dit.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_dit.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_slat_gs: verify_slat_gs.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_gs.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

test_layernorm_token: test_layernorm_token.c hip_sam3d_kernels.h \
                      ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_layernorm_token.c $(ROCEW_SRC) $(LDFLAGS)

test_dinov2_gpu_upload: test_dinov2_gpu_upload.c hip_sam3d_dinov2_gpu.h \
                       ../rocew.h ../hip_runner_common.h  \
                       ../../common/safetensors.h $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_dinov2_gpu_upload.c $(ROCEW_SRC) $(LDFLAGS)

test_dinov2_patch_embed: test_dinov2_patch_embed.c hip_sam3d_kernels.h \
                        ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_dinov2_patch_embed.c $(ROCEW_SRC) $(LDFLAGS)

test_dinov2_prepend_pos: test_dinov2_prepend_pos.c hip_sam3d_kernels.h \
                        ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_dinov2_prepend_pos.c $(ROCEW_SRC) $(LDFLAGS)

test_gemm_f32_bias: test_gemm_f32_bias.c hip_sam3d_kernels.h \
                    ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_gemm_f32_bias.c $(ROCEW_SRC) $(LDFLAGS)

test_qkv_split_sdpa: test_qkv_split_sdpa.c hip_sam3d_kernels.h \
                     ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_qkv_split_sdpa.c $(ROCEW_SRC) $(LDFLAGS)

test_layerscale_add: test_layerscale_add.c hip_sam3d_kernels.h \
                     ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_layerscale_add.c $(ROCEW_SRC) $(LDFLAGS)

test_mlp: test_mlp.c hip_sam3d_kernels.h \
          ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_mlp.c $(ROCEW_SRC) $(LDFLAGS)

test_dinov2_block: test_dinov2_block.c hip_sam3d_kernels.h hip_sam3d_dinov2_forward.h \
                   ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ test_dinov2_block.c $(ROCEW_SRC) $(LDFLAGS)

test_silu_mul: test_silu_mul.c hip_sam3d_kernels.h \
               ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_silu_mul.c $(ROCEW_SRC) $(LDFLAGS)

test_timestep_embed: test_timestep_embed.c hip_sam3d_kernels.h \
                     ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_timestep_embed.c $(ROCEW_SRC) $(LDFLAGS)

test_t_embedder: test_t_embedder.c hip_sam3d_kernels.h \
                 ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_t_embedder.c $(ROCEW_SRC) $(LDFLAGS)

test_adaln_modulation: test_adaln_modulation.c hip_sam3d_kernels.h \
                       ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_adaln_modulation.c $(ROCEW_SRC) $(LDFLAGS)

test_modulated_ln: test_modulated_ln.c hip_sam3d_kernels.h \
                   ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_modulated_ln.c $(ROCEW_SRC) $(LDFLAGS)

test_multi_head_rmsnorm: test_multi_head_rmsnorm.c hip_sam3d_kernels.h \
                         ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_multi_head_rmsnorm.c $(ROCEW_SRC) $(LDFLAGS)

test_mot_self_attn: test_mot_self_attn.c hip_sam3d_kernels.h \
                    ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_mot_self_attn.c $(ROCEW_SRC) $(LDFLAGS)

test_mot_cross_attn: test_mot_cross_attn.c hip_sam3d_kernels.h \
                     ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_mot_cross_attn.c $(ROCEW_SRC) $(LDFLAGS)

test_ssdit_ffn: test_ssdit_ffn.c hip_sam3d_kernels.h \
                ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ssdit_ffn.c $(ROCEW_SRC) $(LDFLAGS)

test_ssdit_latent_proj: test_ssdit_latent_proj.c hip_sam3d_kernels.h \
                        ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ssdit_latent_proj.c $(ROCEW_SRC) $(LDFLAGS)

test_ssdit_block_forward: test_ssdit_block_forward.c hip_sam3d_kernels.h \
                          ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ssdit_block_forward.c $(ROCEW_SRC) $(LDFLAGS)

test_ssdit_gpu_load: test_ssdit_gpu_load.c hip_sam3d_ssdit_gpu.h \
                     ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ssdit_gpu_load.c $(ROCEW_SRC) $(LDFLAGS)

test_pixel_shuffle_3d: test_pixel_shuffle_3d.c hip_sam3d_kernels.h \
                      ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_pixel_shuffle_3d.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_ape: test_slat_ape.c hip_sam3d_kernels.h \
               ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_ape.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_downsample: test_slat_downsample.c hip_sam3d_kernels.h \
                      ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_downsample.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_upsample: test_slat_upsample.c hip_sam3d_kernels.h \
                    ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_upsample.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_submconv3d: test_slat_submconv3d.c hip_sam3d_kernels.h \
                      ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_submconv3d.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_self_attn: test_slat_self_attn.c hip_sam3d_kernels.h \
                     ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_self_attn.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_cross_attn: test_slat_cross_attn.c hip_sam3d_kernels.h \
                      ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_cross_attn.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_resblock: test_slat_resblock.c hip_sam3d_kernels.h \
                    ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_resblock.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_mlp: test_slat_mlp.c hip_sam3d_kernels.h \
               ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_mlp.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_self_attn_residual: test_slat_self_attn_residual.c hip_sam3d_kernels.h \
                              ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_self_attn_residual.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_cross_attn_residual: test_slat_cross_attn_residual.c hip_sam3d_kernels.h \
                               ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_cross_attn_residual.c $(ROCEW_SRC) $(LDFLAGS)

test_slat_transformer_block: test_slat_transformer_block.c hip_sam3d_kernels.h \
                             ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_slat_transformer_block.c $(ROCEW_SRC) $(LDFLAGS)

verify_slat_transformer_block_realw: verify_slat_transformer_block_realw.c hip_sam3d_kernels.h \
                                     hip_sam3d_slat_dit_gpu.h hip_sam3d_slat_dit_forward.h \
                                     ../rocew.h ../hip_runner_common.h  \
                                     ../../common/sam3d_slat_dit.h ../../common/sam3d_ss_flow_dit.h \
                                     ../../common/safetensors.h ../../common/npy_io.h \
                                     $(RUNNER_SRC) $(CPU_DEPS) $(ROCEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_transformer_block_realw.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_slat_input_block0_realw: verify_slat_input_block0_realw.c hip_sam3d_kernels.h \
                                ../rocew.h ../hip_runner_common.h  \
                                ../../common/sam3d_slat_dit.h ../../common/npy_io.h \
                                $(RUNNER_SRC) $(CPU_DEPS) $(ROCEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_input_block0_realw.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_slat_input_block1_realw: verify_slat_input_block1_realw.c hip_sam3d_kernels.h \
                                ../rocew.h ../hip_runner_common.h  \
                                ../../common/sam3d_slat_dit.h ../../common/npy_io.h \
                                $(RUNNER_SRC) $(CPU_DEPS) $(ROCEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_input_block1_realw.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_slat_out_block0_realw: verify_slat_out_block0_realw.c hip_sam3d_kernels.h \
                              ../rocew.h ../hip_runner_common.h  \
                              ../../common/sam3d_slat_dit.h ../../common/npy_io.h \
                              $(RUNNER_SRC) $(CPU_DEPS) $(ROCEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_out_block0_realw.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_slat_out_block1_realw: verify_slat_out_block1_realw.c hip_sam3d_kernels.h \
                              ../rocew.h ../hip_runner_common.h  \
                              ../../common/sam3d_slat_dit.h ../../common/npy_io.h \
                              $(RUNNER_SRC) $(CPU_DEPS) $(ROCEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_slat_out_block1_realw.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

verify_ssdit_block_realw: verify_ssdit_block_realw.c hip_sam3d_ssdit_gpu.h \
                          hip_sam3d_ssdit_forward.h hip_sam3d_kernels.h \
                          ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ verify_ssdit_block_realw.c $(ROCEW_SRC) $(LDFLAGS)

verify_ssdit_full_realw: verify_ssdit_full_realw.c hip_sam3d_ssdit_gpu.h \
                         hip_sam3d_ssdit_forward.h hip_sam3d_ssdit_outer.h \
                         hip_sam3d_kernels.h \
                         ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ssdit_full_realw.c $(ROCEW_SRC) $(LDFLAGS)

verify_ss_dit_ode: verify_ss_dit_ode.c $(RUNNER_SRC) $(ROCEW_SRC) $(COMMON_DEPS) $(CPU_DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ verify_ss_dit_ode.c $(RUNNER_SRC) $(CPU_SRC) $(ROCEW_SRC) $(LDFLAGS)

test_fuser_proj: test_fuser_proj.c hip_sam3d_kernels.h \
                 ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_fuser_proj.c $(ROCEW_SRC) $(LDFLAGS)

test_ppe_linear: test_ppe_linear.c hip_sam3d_kernels.h \
                 ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_linear.c $(ROCEW_SRC) $(LDFLAGS)

test_ppe_window_pack: test_ppe_window_pack.c hip_sam3d_kernels.h \
                      ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_window_pack.c $(ROCEW_SRC) $(LDFLAGS)

test_sdpa_batched: test_sdpa_batched.c hip_sam3d_kernels.h \
                   ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_sdpa_batched.c $(ROCEW_SRC) $(LDFLAGS)

test_ppe_block: test_ppe_block.c hip_sam3d_kernels.h \
                ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_block.c $(ROCEW_SRC) $(LDFLAGS)

test_ppe_cls_pos_extract: test_ppe_cls_pos_extract.c hip_sam3d_kernels.h \
                          ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_cls_pos_extract.c $(ROCEW_SRC) $(LDFLAGS)

test_ppe_full: test_ppe_full.c hip_sam3d_kernels.h hip_sam3d_ppe_forward.h \
               ../rocew.h ../hip_runner_common.h  $(ROCEW_SRC)
	$(CC) $(CFLAGS) -o $@ test_ppe_full.c $(ROCEW_SRC) $(LDFLAGS)

test_ppe_gpu_load: test_ppe_gpu_load.c hip_sam3d_kernels.h hip_sam3d_ppe_forward.h \
                   hip_sam3d_ppe_gpu.h ../rocew.h ../hip_runner_common.h \
                    ../../common/safetensors.h \
                   ../../common/sam3d_cond_fuser.h $(ROCEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ test_ppe_gpu_load.c $(ROCEW_SRC) $(LDFLAGS)

test_dinov2_forward: test_dinov2_forward.c hip_sam3d_kernels.h hip_sam3d_dinov2_forward.h \
                     hip_sam3d_dinov2_gpu.h \
                     ../rocew.h ../hip_runner_common.h  \
                     ../../common/safetensors.h ../../common/dinov2.h \
                     ../../common/cpu_compute.h ../../common/ggml_dequant.h $(ROCEW_SRC)
	$(CC) $(CFLAGS) -fopenmp -o $@ test_dinov2_forward.c $(ROCEW_SRC) $(LDFLAGS)

MODELS ?= /mnt/disk1/models
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
