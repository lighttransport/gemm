#ifndef VHUMAN_MOBILE_NATIVE_H
#define VHUMAN_MOBILE_NATIVE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct vh_mobile vh_mobile;
/* F32 little-endian package. No ML framework, allocation or IO during eval.
 * Rotations are four native axis-angle joints, translation is native metres.
 * The returned vertices are in the candidate's H frame. */
vh_mobile *vh_mobile_load(const char *path);
void vh_mobile_free(vh_mobile *model);
size_t vh_mobile_vertices(const vh_mobile *model);
size_t vh_mobile_expressions(const vh_mobile *model);
int vh_mobile_eval(vh_mobile *model, const float *expression,
                   const float *rotations, const float *translation, float *out);
/* H-frame affine joint transform after eval: R[9], translation[3]. */
int vh_mobile_joint_transform(const vh_mobile *model, unsigned joint, float *out);
#ifdef __cplusplus
}
#endif
#endif
