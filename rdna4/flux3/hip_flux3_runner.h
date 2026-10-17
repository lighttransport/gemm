/* SPDX-License-Identifier: MIT */
#ifndef HIP_FLUX3_RUNNER_H
#define HIP_FLUX3_RUNNER_H
#include <stddef.h>
typedef struct hip_flux3_runner hip_flux3_runner;
/* Keep reserve_mib free for other processes. Only gfx12 is supported. */
hip_flux3_runner *hip_flux3_init(int device, size_t reserve_mib, int verbose);
/* Optional diagnostic activation dumps; directory must already exist. */
void hip_flux3_set_trace(hip_flux3_runner *r, const char *directory);
int hip_flux3_load(hip_flux3_runner *r, const char *checkpoint);
/* Single sample, image/condition/text order. All arrays CPU F32.
 * Features: image/condition [N,128], text [N,20480]; ids [N,4].
 * Timesteps normalized [0,1]; target is required for distilled flow maps.
 * Empty condition stream is allowed. Image and text must be nonempty. */
int hip_flux3_step(hip_flux3_runner *r, const float *image, const float *image_ids, int ni,
                   const float *condition, const float *condition_ids, int nc, const float *text,
                   const float *text_ids, int nt, float timestep, float target, float *velocity);
int hip_flux3_selftest(hip_flux3_runner *r);
void hip_flux3_free(hip_flux3_runner *r);
#endif
