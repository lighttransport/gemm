/* Retained Stage-1 pose modalities, independent of backend/weight loaders. */
#ifndef SAM3D_POSE_TOKENS_H
#define SAM3D_POSE_TOKENS_H
#include <stddef.h>
#include <string.h>
typedef struct {
    float values[13]; /* normalized 6D rotation, translation, log-scale, translation-scale */
    int valid;
} sam3d_pose_tokens;
static inline void sam3d_pose_tokens_store(sam3d_pose_tokens *pose, float *const *modalities)
{
    memcpy(pose->values, modalities[1], 6*sizeof(float));
    memcpy(pose->values+6, modalities[2], 3*sizeof(float));
    memcpy(pose->values+9, modalities[3], 3*sizeof(float));
    pose->values[12] = modalities[4][0];
    pose->valid = 1;
}
static inline int sam3d_pose_tokens_get(const sam3d_pose_tokens *pose, int modality,
    float *out, size_t capacity, int *out_n, int *out_c)
{
    static const int channels[4] = {6, 3, 3, 1}, offsets[4] = {0, 6, 9, 12};
    if (!pose || modality < 1 || modality > 4) return -1;
    if (!pose->valid) return -4;
    int c = channels[modality-1];
    if (out && capacity < (size_t)c) return -1;
    if (out_n) *out_n = 1;
    if (out_c) *out_c = c;
    if (out) memcpy(out, pose->values+offsets[modality-1], c*sizeof(float));
    return 0;
}
#endif
