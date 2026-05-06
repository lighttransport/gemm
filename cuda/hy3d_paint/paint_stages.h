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

/* Render all 6 views. Output device buffers must each hold N_views*H*W*3
 * floats (typically 6 * res * res * 3). Either may be NULL to skip. */
void paint_stage_view_maps_render(paint_stage_view_maps *s,
                                   CUdeviceptr d_normal_out,
                                   CUdeviceptr d_position_out);

void paint_stage_view_maps_destroy(paint_stage_view_maps *s);

#ifdef __cplusplus
}
#endif

#endif /* PAINT_STAGES_H_ */
