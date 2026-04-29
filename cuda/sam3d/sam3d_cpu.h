/* sam3d_cpu.h — wrapper API surface for sam3d_cpu.c.
 *
 * Used internally by cuda_sam3d_runner.c during Phase 1a (CPU-fallback
 * bring-up). Replaces with NVRTC kernels stage-by-stage in later
 * phases — the API stays stable so verify_*.c is unaffected. */
#ifndef SAM3D_CPU_H_
#define SAM3D_CPU_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sam3d_cpu_dinov2 sam3d_cpu_dinov2;

sam3d_cpu_dinov2 *sam3d_cpu_dinov2_load(const char *path);
void              sam3d_cpu_dinov2_free(sam3d_cpu_dinov2 *w);
int  sam3d_cpu_dinov2_image_size(const sam3d_cpu_dinov2 *w);
int  sam3d_cpu_dinov2_dim       (const sam3d_cpu_dinov2 *w);
int  sam3d_cpu_dinov2_n_register(const sam3d_cpu_dinov2 *w);

/* Forward [image | mask] branches. mask may be NULL → single branch.
 * Returns malloc'd [n_tokens_out × dim_out] f32; caller free()s.
 * Output drops the n_register tokens, matching cpu/sam3d numerics. */
float *sam3d_cpu_dinov2_encode_rgba(sam3d_cpu_dinov2 *w,
                                    const uint8_t *rgba, int iw, int ih,
                                    const uint8_t *mask, int mw, int mh,
                                    int n_threads,
                                    int *n_tokens_out, int *dim_out);

/* Preprocess only — bilinear-resize + ImageNet-normalize to [3, S, S].
 * Returns malloc'd CHW f32 buffer (caller free()s). S = image_size(w). */
float *sam3d_cpu_dinov2_preprocess_rgb (sam3d_cpu_dinov2 *w,
                                        const uint8_t *rgba, int iw, int ih);
float *sam3d_cpu_dinov2_preprocess_mask(sam3d_cpu_dinov2 *w,
                                        const uint8_t *mask, int mw, int mh);

/* CondEmbedderFuser (PointPatchEmbed + Llama SwiGLU projections + pos
 * embeds). Holds both the PPE model and the fuser model — they share
 * the same safetensors_dir lookup pattern and are always used together. */
typedef struct sam3d_cpu_fuser sam3d_cpu_fuser;

sam3d_cpu_fuser *sam3d_cpu_fuser_load(const char *safetensors_dir);
void             sam3d_cpu_fuser_free(sam3d_cpu_fuser *w);
int              sam3d_cpu_fuser_dim_out(const sam3d_cpu_fuser *w);

/* Run the fuser. dino_tokens is [dino_n × dino_dim] f32 with
 * dino_n = n_branches × per_branch (n_branches ∈ {1, 2}; v1 uses 2 →
 * image+mask). Pointmap may be NULL — output then carries only the
 * dino branches. Returns malloc'd [n_tokens_out × dim_out] f32; caller
 * free()s. */
float *sam3d_cpu_fuser_run(sam3d_cpu_fuser *w,
                           const float *dino_tokens,
                           int dino_n, int dino_dim,
                           const float *pointmap_xyz, int ph, int pw,
                           int n_threads,
                           int *n_tokens_out, int *dim_out);

/* Same as sam3d_cpu_fuser_run, but accepts pre-computed PPE tokens
 * directly (already passed through sam3d_ppe_encode or its GPU
 * equivalent). Skips the PPE encode step. Used by the CUDA runner once
 * PPE is on-device. ppe_tokens may be NULL → output carries only the
 * dino branches. */
float *sam3d_cpu_fuser_run_with_ppe_tokens(sam3d_cpu_fuser *w,
                           const float *dino_tokens,
                           int dino_n, int dino_dim,
                           const float *ppe_tokens,
                           int n_ppe, int ppe_dim,
                           int n_threads,
                           int *n_tokens_out, int *dim_out);

/* PPE geometry accessors — needed by the CUDA runner to size the
 * device buffer it hands back to the fuser. */
int sam3d_cpu_fuser_ppe_num_patches(const sam3d_cpu_fuser *w);  /* Np */
int sam3d_cpu_fuser_ppe_input_size (const sam3d_cpu_fuser *w);  /* S  */
int sam3d_cpu_fuser_ppe_embed_dim  (const sam3d_cpu_fuser *w);  /* D  */
struct sam3d_ppe_model   *sam3d_cpu_fuser_ppe_model  (sam3d_cpu_fuser *w);
struct sam3d_fuser_model *sam3d_cpu_fuser_fuser_model(sam3d_cpu_fuser *w);

/* SS Flow DiT (sparse-structure stage-1, shortcut model). */
typedef struct sam3d_cpu_ss_dit sam3d_cpu_ss_dit;

sam3d_cpu_ss_dit *sam3d_cpu_ss_dit_load(const char *safetensors_dir);
void              sam3d_cpu_ss_dit_free(sam3d_cpu_ss_dit *w);

int sam3d_cpu_ss_dit_n_blocks      (const sam3d_cpu_ss_dit *w);
int sam3d_cpu_ss_dit_dim           (const sam3d_cpu_ss_dit *w);
int sam3d_cpu_ss_dit_cond_channels (const sam3d_cpu_ss_dit *w);
int sam3d_cpu_ss_dit_is_shortcut   (const sam3d_cpu_ss_dit *w);
int sam3d_cpu_ss_dit_n_latents     (void);
int sam3d_cpu_ss_dit_lat_elts      (int modality_id);
struct sam3d_ss_flow_dit_model *sam3d_cpu_ss_dit_model(sam3d_cpu_ss_dit *w);

/* Per-call forward — pass-through to sam3d_ss_flow_dit_forward.
 * latents_in/out: 5 buffers, sized by sam3d_cpu_ss_dit_lat_elts(i).
 * cond: [n_cond × cond_channels] f32. */
int sam3d_cpu_ss_dit_forward(sam3d_cpu_ss_dit *w,
                             const float *const *latents_in,
                             float *const *latents_out,
                             const float *cond, int n_cond,
                             float t, float d, int n_threads);

/* Full shortcut ODE integration. Seeds Gaussian noise at t=1, denoises
 * over `steps` shortcut steps, persists SHAPE modality as NCDHW
 * [8,16,16,16] in caller-allocated ss_latent (8*16^3 floats). */
int sam3d_cpu_ss_dit_run_ode(sam3d_cpu_ss_dit *w,
                             const float *cond, int n_cond,
                             int steps, uint64_t seed, float cfg_scale,
                             int n_threads,
                             float *ss_latent_ncdhw /*[8*16*16*16]*/);

/* SS-VAE 3D-conv decoder (TRELLIS.2-compatible). */
typedef struct sam3d_cpu_ss_dec sam3d_cpu_ss_dec;

sam3d_cpu_ss_dec *sam3d_cpu_ss_dec_load(const char *safetensors_dir);
void              sam3d_cpu_ss_dec_free(sam3d_cpu_ss_dec *w);
void             *sam3d_cpu_ss_dec_model(sam3d_cpu_ss_dec *w);

/* Forward pass. latent_ncdhw: [8,16,16,16] f32. Returns malloc'd 64³
 * occupancy logits (caller free()s); NULL on failure. */
float *sam3d_cpu_ss_dec_forward(sam3d_cpu_ss_dec *w,
                                const float *latent_ncdhw,
                                int n_threads);

/* SLAT Flow DiT (stage-2 sparse + shift-window). */
typedef struct sam3d_cpu_slat_dit sam3d_cpu_slat_dit;

sam3d_cpu_slat_dit *sam3d_cpu_slat_dit_load(const char *safetensors_dir);
void                sam3d_cpu_slat_dit_free(sam3d_cpu_slat_dit *w);

int sam3d_cpu_slat_dit_in_channels  (const sam3d_cpu_slat_dit *w);
int sam3d_cpu_slat_dit_out_channels (const sam3d_cpu_slat_dit *w);
int sam3d_cpu_slat_dit_cond_channels(const sam3d_cpu_slat_dit *w);
void *sam3d_cpu_slat_dit_model(sam3d_cpu_slat_dit *w);

typedef int (*sam3d_cpu_slat_transformer_hook_fn)(void *user,
                                                  float *feats, int N,
                                                  const int32_t *coords,
                                                  const float *t_emb,
                                                  const float *cond,
                                                  int n_cond,
                                                  int dim, int n_blocks);
void sam3d_cpu_slat_dit_set_transformer_hook(sam3d_cpu_slat_transformer_hook_fn fn,
                                             void *user);

typedef int (*sam3d_cpu_slat_ape_transformer_hook_fn)(void *user,
                                                      float *feats, int N,
                                                      const int32_t *coords,
                                                      const float *t_emb,
                                                      const float *cond,
                                                      int n_cond,
                                                      int dim, int n_blocks);
void sam3d_cpu_slat_dit_set_ape_transformer_hook(sam3d_cpu_slat_ape_transformer_hook_fn fn,
                                                 void *user);

typedef int (*sam3d_cpu_slat_input_layer_hook_fn)(void *user,
                                                  void *xp,
                                                  const void *input_w,
                                                  const void *input_b,
                                                  int out_channels);
void sam3d_cpu_slat_dit_set_input_layer_hook(sam3d_cpu_slat_input_layer_hook_fn fn,
                                             void *user);

typedef int (*sam3d_cpu_slat_io_block_hook_fn)(void *user,
                                               int is_output,
                                               int block_idx,
                                               const void *bk,
                                               void *xp,
                                               const float *t_emb,
                                               const int32_t *up_target_coords,
                                               int up_target_N,
                                               int dim,
                                               float ln_eps);
void sam3d_cpu_slat_dit_set_io_block_hook(sam3d_cpu_slat_io_block_hook_fn fn,
                                          void *user);

typedef int (*sam3d_cpu_slat_final_layer_hook_fn)(void *user,
                                                  void *xp,
                                                  const void *out_w,
                                                  const void *out_b,
                                                  int out_channels,
                                                  float eps);
void sam3d_cpu_slat_dit_set_final_layer_hook(sam3d_cpu_slat_final_layer_hook_fn fn,
                                             void *user);

/* Single forward: coords[N,4] (b,z,y,x) i32, feats[N,in_ch] f32, returns
 * malloc'd out_feats[N,out_ch] f32 on success; NULL on failure. */
float *sam3d_cpu_slat_dit_forward(sam3d_cpu_slat_dit *w,
                                  const int32_t *coords,
                                  const float *feats, int N,
                                  float t,
                                  const float *cond, int n_cond,
                                  int n_threads);

/* Full ODE: voxel prune from occupancy logits (occ > 0), seed noise feats,
 * flow_matching loop for `steps` iterations, apply SLAT un-normalization
 * (mean/std hardcoded for facebook/sam-3d-objects). On success, returns
 * malloc'd `*out_coords` [N×4 i32] and `*out_feats` [N×out_channels f32]
 * (caller frees both); writes N to `*out_n`. */
int sam3d_cpu_slat_dit_run_ode(sam3d_cpu_slat_dit *w,
                               const float *occupancy /*64³*/,
                               int D, int H, int W,
                               const float *cond, int n_cond,
                               int steps, uint64_t seed, int n_threads,
                               int32_t **out_coords, float **out_feats,
                               int *out_n);

/* Same ODE as above, but starts from precomputed sparse coords[N,4].
 * Used by the CUDA runner once occupancy pruning has happened on GPU. */
int sam3d_cpu_slat_dit_run_ode_from_coords(sam3d_cpu_slat_dit *w,
                                           const int32_t *coords, int n_coords,
                                           const float *cond, int n_cond,
                                           int steps, uint64_t seed, int n_threads,
                                           int32_t **out_coords, float **out_feats,
                                           int *out_n);

/* SLAT GS decoder (sparse transformer + per-voxel Gaussian decode head). */
typedef struct sam3d_cpu_gs_decoder sam3d_cpu_gs_decoder;

sam3d_cpu_gs_decoder *sam3d_cpu_gs_decoder_load(const char *safetensors_dir);
void                  sam3d_cpu_gs_decoder_free(sam3d_cpu_gs_decoder *w);

int sam3d_cpu_gs_decoder_in_channels  (const sam3d_cpu_gs_decoder *w);
int sam3d_cpu_gs_decoder_out_channels (const sam3d_cpu_gs_decoder *w);
int sam3d_cpu_gs_decoder_num_gaussians(const sam3d_cpu_gs_decoder *w);
void *sam3d_cpu_gs_decoder_model(sam3d_cpu_gs_decoder *w);

typedef int (*sam3d_cpu_gs_input_ape_hook_fn)(void *user,
                                              const int32_t *coords,
                                              const float *feats,
                                              int N, int in_channels,
                                              const void *input_w,
                                              const void *input_b,
                                              int dim,
                                              float **out_h);
typedef int (*sam3d_cpu_gs_final_layer_hook_fn)(void *user,
                                                const float *h,
                                                int N, int dim,
                                                const void *out_w,
                                                const void *out_b,
                                                int out_channels,
                                                float eps,
                                                float **out_feats);
typedef int (*sam3d_cpu_gs_window_attn_hook_fn)(void *user,
                                                float *out,
                                                const float *qkv,
                                                const void *x,
                                                int window_size,
                                                const int shift[3],
                                                int n_heads,
                                                int head_dim);
typedef int (*sam3d_cpu_gs_attn_block_hook_fn)(void *user,
                                               float *h,
                                               const void *x,
                                               int N, int dim,
                                               const void *blk,
                                               int window_size,
                                               const int shift[3],
                                               int n_heads,
                                               int head_dim,
                                               float eps);
typedef int (*sam3d_cpu_gs_mlp_hook_fn)(void *user,
                                        float *h,
                                        int N, int dim,
                                        const void *blk,
                                        int hidden,
                                        float eps);
typedef int (*sam3d_cpu_gs_block_hook_fn)(void *user,
                                          float *h,
                                          const void *x,
                                          int N, int dim,
                                          const void *blk,
                                          int window_size,
                                          const int shift[3],
                                          int n_heads,
                                          int head_dim,
                                          int hidden,
                                          float eps);
typedef int (*sam3d_cpu_gs_stack_hook_fn)(void *user,
                                          float *h,
                                          const void *x,
                                          int N, int dim,
                                          const void *blocks,
                                          int n_blocks,
                                          int window_size,
                                          int n_heads,
                                          int head_dim,
                                          int hidden,
                                          float eps);
typedef int (*sam3d_cpu_gs_transformer_hook_fn)(void *user,
                                                const void *x,
                                                const void *m,
                                                float **out_feats);
void sam3d_cpu_gs_decoder_set_input_ape_hook(sam3d_cpu_gs_input_ape_hook_fn fn,
                                             void *user);
void sam3d_cpu_gs_decoder_set_final_layer_hook(sam3d_cpu_gs_final_layer_hook_fn fn,
                                               void *user);
void sam3d_cpu_gs_decoder_set_window_attn_hook(sam3d_cpu_gs_window_attn_hook_fn fn,
                                               void *user);
void sam3d_cpu_gs_decoder_set_attn_block_hook(sam3d_cpu_gs_attn_block_hook_fn fn,
                                              void *user);
void sam3d_cpu_gs_decoder_set_mlp_hook(sam3d_cpu_gs_mlp_hook_fn fn,
                                       void *user);
void sam3d_cpu_gs_decoder_set_block_hook(sam3d_cpu_gs_block_hook_fn fn,
                                         void *user);
void sam3d_cpu_gs_decoder_set_stack_hook(sam3d_cpu_gs_stack_hook_fn fn,
                                         void *user);
void sam3d_cpu_gs_decoder_set_transformer_hook(sam3d_cpu_gs_transformer_hook_fn fn,
                                               void *user);

/* Run the SLAT GS decoder transformer (input_layer + APE + N blocks +
 * out_layer). Returns malloc'd [N × out_channels] f32; caller free()s. */
float *sam3d_cpu_gs_decoder_transformer(sam3d_cpu_gs_decoder *w,
                                        const int32_t *coords,
                                        const float *feats, int N,
                                        int n_threads);

/* Decode raw out_feats[N, out_channels] into per-gaussian buffers. Each
 * out pointer may be NULL to skip. Buffer sizes:
 *   xyz_out        [N*G, 3]
 *   dc_out         [N*G, 3]
 *   scaling_out    [N*G, 3]
 *   rotation_out   [N*G, 4]
 *   opacity_out    [N*G]
 * G = sam3d_cpu_gs_decoder_num_gaussians(). */
int sam3d_cpu_gs_decoder_to_representation(sam3d_cpu_gs_decoder *w,
                                           const int32_t *coords,
                                           const float *feats_out, int N,
                                           float *xyz_out, float *dc_out,
                                           float *scaling_out, float *rotation_out,
                                           float *opacity_out);

/* Convert per-gaussian raw rep into the 17-channel INRIA-PLY storage
 * convention used by both runners. xyz/dc/rot pass through; opacity gets
 * +opacity_bias (still pre-sigmoid logit); scale becomes
 * log(softplus(raw + inv_softplus(scaling_bias))). Writes [N*G, 17] f32
 * into out_ply (caller-allocated, contiguous). */
int sam3d_cpu_gs_decoder_pack_ply(const sam3d_cpu_gs_decoder *w,
                                  const float *xyz, const float *dc,
                                  const float *scl, const float *rot,
                                  const float *op,
                                  int total, int stride, float *out_ply);

#ifdef __cplusplus
}
#endif

#endif /* SAM3D_CPU_H_ */
