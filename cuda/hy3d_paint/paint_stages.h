/*
 * paint_stages.h - extern wrappers for the per-stage runner TUs.
 *
 * Each stage runner header (cuda_paint_vae_runner.h, cuda_paint_unet_runner.h,
 * ...) defines its own file-local helpers under overlapping names (k_conv,
 * load_resblock, upload_st, ...). They cannot be co-included into one TU.
 * Each runner therefore lives in its own .c file (paint_stage_vae.c,
 * paint_stage_unet.c, ...) which exposes only the opaque pointer + entry
 * points declared here. The pipeline orchestrator includes only this header.
 */

#ifndef PAINT_STAGES_H_
#define PAINT_STAGES_H_

#include "../cuew.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct paint_stage_vae paint_stage_vae;

paint_stage_vae *paint_stage_vae_create(CUdevice dev, const char *vae_path);
void paint_stage_vae_decode(paint_stage_vae *s,
                             CUdeviceptr d_lat, int lat_h, int lat_w,
                             CUdeviceptr d_rgb,
                             CUdeviceptr d_a, CUdeviceptr d_b,
                             CUdeviceptr d_t1, CUdeviceptr d_t2,
                             CUdeviceptr d_qnc, CUdeviceptr d_knc,
                             CUdeviceptr d_vnc, CUdeviceptr d_ync);
void paint_stage_vae_destroy(paint_stage_vae *s);

/* ===== UNet stage =========================================================
 * Wraps the dual-stream UNet2p5DConditionModel + per-step scheduler-driven
 * forward extracted from test_paint_unet's out_loop block. The orchestrator
 * supplies conditioning tensors (embeds_normal / embeds_position /
 * encoder_hidden_states / ref_latents / dino_hidden_states) on the host;
 * the stage owns weights + caches + workspace device buffers.
 *
 * Ordering of calls per pipeline run:
 *   1. paint_stage_unet_create
 *   2. paint_stage_unet_set_conditioning
 *   3. paint_stage_unet_run_dual    (once, populates RA cache)
 *   4. paint_stage_unet_run_step    (N_steps times, with current x_host /
 *                                    timestep — caller does UniPC update)
 *   5. paint_stage_unet_destroy
 */
typedef struct paint_stage_unet paint_stage_unet;

typedef struct {
    int B_outer;            /* outer batch (typically 1) */
    int N_pbr;              /* materials, e.g. 2 (albedo, mr) */
    int N_gen;              /* views per material, e.g. 2 */
    int N_ref;              /* reference views (dual stream), e.g. 1 */
    int H0, W0;             /* latent H/W (e.g. 64) */
    int M_text;             /* text token count (e.g. 77) */
    int cross_dim;          /* text cross dim (e.g. 1024) */
    int T_dino;             /* DINO tokens (e.g. 257) */
    int C_dino_in;          /* DINO embed dim (e.g. 1536) */
} paint_unet_config;

paint_stage_unet *paint_stage_unet_create(CUdevice dev,
                                           const char *unet_safetensors_path,
                                           const paint_unet_config *cfg);

void paint_stage_unet_set_conditioning(paint_stage_unet *s,
    const float *embeds_normal,           /* [N_gen, 4, H0, W0] */
    const float *embeds_position,         /* [N_gen, 4, H0, W0] */
    const float *encoder_hidden_states,   /* [N_pbr, M_text, cross_dim] */
    const float *ref_latents,             /* [N_ref, 4, H0, W0] */
    const float *dino_hidden_states);     /* [T_dino, C_dino_in] */

void paint_stage_unet_run_dual(paint_stage_unet *s);

/* Run one UNet forward at `timestep` with current latent `x_host`
 * [Beff_main * 4 * H0 * W0 = N_pbr*N_gen*4*H0*W0]; writes the predicted
 * noise into `noise_pred_host` (same shape). */
void paint_stage_unet_run_step(paint_stage_unet *s, long long timestep,
                                const float *x_host, float *noise_pred_host);

void paint_stage_unet_destroy(paint_stage_unet *s);

/* ===== View-maps stage ====================================================
 * Renders per-view (normal_rgb, position_rgb) maps from a mesh, mirroring
 * hy3dpaint MeshRender.render_normal_multiview / render_position_multiview
 * with shader_type=face, camera_type=orth, camera_distance=1.45,
 * ortho_scale=1.2, scale_factor=1.15, auto_center=True. 6 views fixed:
 * azim {0,90,180,270,0,180} × elev {0,0,0,0,+90,-90}.
 *
 * Output buffers are device-resident, contiguous [N_views, H, W, 3] f32 in
 * [0,1] range with white background, ready to feed VAE-encode.
 */
typedef struct paint_stage_view_maps paint_stage_view_maps;

paint_stage_view_maps *paint_stage_view_maps_create(CUdevice dev, int res);

/* Set the source mesh (overwrites any prior mesh; coords are mutated by the
 * stage's set_mesh transform: negate XY, swap YZ, auto-center, scale_factor). */
void paint_stage_view_maps_set_mesh(paint_stage_view_maps *s,
                                     const float *vtx_pos, int n_verts,
                                     const int *tri_idx, int n_tris);

/* Render all 6 views. Output device buffers must each hold N_views*H*W*C
 * floats: normal/position are [N,H,W,3] (C=3); depth/visible/cos are
 * [N,H,W] (C=1). Any device buffer may be 0 to skip. out_w2c (if non-NULL)
 * receives [N,16] per-view world->camera matrices (column-major, applied as
 * row-vector by back_project_sample_f32). out_proj (if non-NULL) receives
 * [proj00, proj11] for the orthographic projection (same value for both).
 *
 * depth is camera-space z (un-normalized), matching back_project_sample_f32's
 * `cz = w2c * world_pos`. cos is dot(face_normal_cam, +z_camera_forward) i.e.
 * -face_normal_cam.z (lookat = (0,0,-1) in camera space), with the standard
 * paint pipeline 75° angle threshold (cos < cos(75°) → 0). visible is 1.f
 * for any pixel covered by a triangle, 0.f otherwise. */
void paint_stage_view_maps_render(paint_stage_view_maps *s,
                                   CUdeviceptr d_normal_out,
                                   CUdeviceptr d_position_out,
                                   CUdeviceptr d_depth_out,
                                   CUdeviceptr d_visible_out,
                                   CUdeviceptr d_cos_out,
                                   float *out_w2c,
                                   float *out_proj);

void paint_stage_view_maps_destroy(paint_stage_view_maps *s);

/* Apply the same set_mesh pre-render transform (axis swap + auto-center +
 * scale_factor) to an arbitrary [n,3] world-space xyz buffer. Used to lift a
 * pyref-generated tex_pos into the rendered coord frame so back_project's
 * w2c (produced by view_maps) lines up with the texel positions. */
void paint_stage_view_maps_apply_mesh_transform(paint_stage_view_maps *s,
                                                  const float *in_xyz,
                                                  float *out_xyz, int n);

/* ===== DINOv2-giant encoder stage ========================================
 * Wraps the 40-layer DINOv2-G-40 encoder used by the paint pipeline as image
 * conditioning. Input: [1,3,224,224] f32 host buffer (BitImageProcessor
 * preprocessed). Output: [1, 257, 1536] f32 host buffer (final LN hidden
 * state) suitable as `dino_hidden_states` for paint_stage_unet. */
typedef struct paint_stage_dinov2g paint_stage_dinov2g;

paint_stage_dinov2g *paint_stage_dinov2g_create(CUdevice dev,
                                                 const char *weights_path);
void paint_stage_dinov2g_run(paint_stage_dinov2g *s,
                              const float *image_f32, float *out_f32);
void paint_stage_dinov2g_destroy(paint_stage_dinov2g *s);

/* ===== Back-project + bake-blend stage ===================================
 * Wraps the per-view back_project + GPU bake-blend chain (back_project_sample_f32
 * -> bake_blend_count_f32 -> bake_blend_accum_f32 -> bake_blend_finalize_f32),
 * matching MeshRender.fast_bake_texture incl. the "skip view if 99% painted"
 * behavior. Atlas tensors are uploaded once via set_atlas; per-view inputs
 * are passed host-side to add_view (the stage uploads them onto its scratch
 * device buffers). finalize downloads the merged bake_tex + bake_mask. */
typedef struct paint_stage_back_project paint_stage_back_project;

paint_stage_back_project *paint_stage_back_project_create(CUdevice dev,
                                                           int Htex, int Wtex,
                                                           int C);

void paint_stage_back_project_set_atlas(paint_stage_back_project *s,
                                         const float *tex_pos,
                                         const int *tex_cov);

void paint_stage_back_project_begin(paint_stage_back_project *s);

/* Run one view: returns 0 if accumulated, 1 if the view was skipped because
 * it was already ≥99% painted by previous views. */
int paint_stage_back_project_add_view(paint_stage_back_project *s,
                                       const float *image,
                                       const float *depth,
                                       const float *visible,
                                       const float *cos_img,
                                       const float *w2c_4x4,
                                       int Himg, int Wimg,
                                       float proj00, float proj11);

void paint_stage_back_project_finalize(paint_stage_back_project *s,
                                        float *out_bake,
                                        float *out_mask);

void paint_stage_back_project_destroy(paint_stage_back_project *s);

#ifdef __cplusplus
}
#endif

#endif /* PAINT_STAGES_H_ */
