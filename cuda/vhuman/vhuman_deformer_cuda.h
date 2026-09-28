/*
 * CUDA backend of the virtual-human face deformer (ryzen/vhuman_deformer.h).
 *
 * The host evaluates the rig per frame (controls -> morph weights incl. the
 * ML correctives, skinning matrices: vh_deformer_prepare); the device holds
 * the rest shape, all morphs and the skin weights, and runs one fused kernel
 * per batch: blend morphs (register-tiled over FRAMES_PER_THREAD frames, so
 * the morph matrix is read once per tile) then 4-influence LBS.
 * Driver API through cuew, kernels compiled at run time with NVRTC.
 */
#ifndef VHUMAN_DEFORMER_CUDA_H
#define VHUMAN_DEFORMER_CUDA_H

#include <stddef.h>

#include "../../ryzen/vhuman_deformer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vh_gpu vh_gpu;

/* device index; returns NULL without a CUDA driver/device. */
vh_gpu *vh_gpu_create(vh_deformer *d, int device, int verbose);
void vh_gpu_free(vh_gpu *g);
const char *vh_gpu_name(const vh_gpu *g);

/* controls: frames x C; out: frames x V x 3 on the host (NULL: keep on the
 * device). Timings (ms) are optional: prepare (host rig), upload, kernel,
 * download. Returns 0 on success. */
int vh_gpu_eval_batch(vh_gpu *g, const float *controls, size_t frames, int use_ml, float *out, double *ms4);

#ifdef __cplusplus
}
#endif

#endif
