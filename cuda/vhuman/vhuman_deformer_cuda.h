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
#include <stdint.h>

#include "../../ryzen/vhuman_deformer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vh_gpu vh_gpu;

/* device index; returns NULL without a CUDA driver/device. */
vh_gpu *vh_gpu_create(vh_deformer *d, int device, int verbose);
/* Retain the device primary context; borrow the caller's CUDA stream.
 * The caller must keep that stream alive until vh_gpu_free. */
vh_gpu *vh_gpu_create_shared(vh_deformer *d, int device, uintptr_t stream, int verbose);
/* Enqueue one frame without a device/stream completion wait. Host inputs are
 * copied to persistent pinned staging. Call/consume on the borrowed stream.
 * The returned device view is borrowed, overwritten on the next submit, and
 * invalid after free. Do not retain it across submits without a device copy. */
int vh_gpu_submit(vh_gpu *g, const float *controls, int use_ml);
uintptr_t vh_gpu_vertices_device(const vh_gpu *g);
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
