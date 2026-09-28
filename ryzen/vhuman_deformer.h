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

/* Post-skinning contact projection (exact lid/eyeball, lip/teeth, lip/tongue
 * and upper/lower lip contacts; see server/vhuman/rig/contacts.py). On by
 * default when the package has contact tensors; 0 iterations disables it.
 * Applied by vh_deformer_eval and vh_deformer_eval_batch. */
int vh_deformer_has_contacts(const vh_deformer *d);
void vh_deformer_set_contact_iterations(vh_deformer *d, int iterations);
int vh_deformer_contact_iterations(const vh_deformer *d);
/* Project one frame's positions (V x 3, in place) with that frame's skin12. */
void vh_deformer_project(const vh_deformer *d, const float *skin12, float *pos);

/* Contact tensors for GPU backends (NULL/0 without contacts). */
typedef struct {
    size_t ne, nl, ns, np, nc;
    const int *eye_ids, *eye_joint, *lip_ids, *sph_joint, *pair_u, *pair_l, *up_joints;
    const float *eye_center, *eye_thr, *sph_center, *sph_weight, *sph_thr, *pair_floor;
    const int *verts, *nbr_ptr, *nbr_idx;     /* contact vertices and their graph (CSR); may be NULL */
} vh_contacts;
#define VH_CONTACT_SMOOTH_STEPS 2
const vh_contacts *vh_deformer_contacts(const vh_deformer *d);

/* Host side of GPU backends: the per-frame rig state (morph weights, M
 * floats; skinning matrices, J x 12 floats: the top 3 rows, row-major). */
size_t vh_deformer_joints(const vh_deformer *d);
void vh_deformer_prepare(vh_deformer *d, const float *controls, int use_ml, float *weights, float *skin12);
/* Batched form (the MLP as two GEMMs); scratch: vh_deformer_prepare_scratch floats. */
size_t vh_deformer_prepare_scratch(const vh_deformer *d, size_t frames);
void vh_deformer_prepare_batch(vh_deformer *d, const float *controls, size_t frames, int use_ml, float *weights,
                               float *skin12, float *scratch);
const float *vh_deformer_rest(const vh_deformer *d);           /* V x 3 */
const float *vh_deformer_morph(const vh_deformer *d);          /* M x V x 3 */
const int *vh_deformer_skin_joints(const vh_deformer *d);      /* V x 4 */
const float *vh_deformer_skin_weights(const vh_deformer *d);   /* V x 4 */

#ifdef __cplusplus
}
#endif

#endif
