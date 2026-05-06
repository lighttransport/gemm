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

#ifdef __cplusplus
}
#endif

#endif /* PAINT_STAGES_H_ */
