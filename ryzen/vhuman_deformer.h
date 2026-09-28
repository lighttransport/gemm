/*
 * Virtual-human face deformer (CPU runtime of server/vhuman/rig).
 *
 * Evaluates a rig package (rig_deformer.safetensors, written by
 * server/vhuman/rig/native.py) for control vectors:
 *   x = clamp(controls); corrective inputs c_p = min(1, w_p prod clamp(x_i,0,1))
 *   joint deltas = M [x|c] -> local/world/skinning matrices
 *   morph weights: blendshapes from [x|c], ML correctives from an MLP2
 *   (lightrig_mlp2.c) -> PCA coefficients
 *   p = rest + sum_m w_m morph_m   (AXPY over active morphs, or one
 *                                   sgemm_avx2 for a batch of frames)
 *   out = LBS(p) with 4 influences
 * Units: metres, head frame (+Y up, face +Z).
 */
#ifndef VHUMAN_DEFORMER_H
#define VHUMAN_DEFORMER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vh_deformer vh_deformer;

vh_deformer *vh_deformer_load(const char *path);
void vh_deformer_free(vh_deformer *d);

size_t vh_deformer_controls(const vh_deformer *d);   /* control count (order: rig.json) */
size_t vh_deformer_vertices(const vh_deformer *d);   /* welded template vertices */
size_t vh_deformer_morphs(const vh_deformer *d);     /* blendshapes + ML targets */
int vh_deformer_has_ml(const vh_deformer *d);

/* Morph weights for one frame (controls: vh_deformer_controls floats). */
void vh_deformer_weights(vh_deformer *d, const float *controls, float *weights);

/* One frame: out = V x 3 positions. use_ml = 0 disables the ML correctives. */
void vh_deformer_eval(vh_deformer *d, const float *controls, int use_ml, float *out);

/* A batch: controls (frames x C), out (frames x V x 3). Morphs are applied
 * with one GEMM (frames x M) x (M x 3V). scratch: see vh_deformer_batch_scratch. */
size_t vh_deformer_batch_scratch(const vh_deformer *d, size_t frames);   /* floats */
void vh_deformer_eval_batch(vh_deformer *d, const float *controls, size_t frames, int use_ml, float *scratch,
                            float *out);

#ifdef __cplusplus
}
#endif

#endif
