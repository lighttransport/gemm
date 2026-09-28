/*
 * Vulkan compute backend of the virtual-human face deformer
 * (ryzen/vhuman_deformer.h): the host prepares the per-frame rig state
 * (vh_deformer_prepare), one compute shader (shaders/vh_deform.comp, SPIR-V
 * embedded at build time) blends the morphs, tiled over 8 frames, and skins.
 * Vulkan is loaded at run time (vkew); works on NVIDIA, AMD and Intel.
 */
#ifndef VHUMAN_DEFORMER_VK_H
#define VHUMAN_DEFORMER_VK_H

#include <stddef.h>

#include "../../ryzen/vhuman_deformer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vh_vk vh_vk;

vh_vk *vh_vk_create(vh_deformer *d, int device, int verbose);   /* NULL without Vulkan */
void vh_vk_free(vh_vk *g);
const char *vh_vk_name(const vh_vk *g);
/* after the first evaluation: bit 0 inputs in device-local host-visible memory,
 * bit 1 host-cached readback memory */
int vh_vk_memory_flags(const vh_vk *g);
/* controls: frames x C; out: frames x V x 3 (host). ms4: prepare, upload,
 * dispatch (submit to completion), download. Returns 0 on success. */
int vh_vk_eval_batch(vh_vk *g, const float *controls, size_t frames, int use_ml, float *out, double *ms4);

#ifdef __cplusplus
}
#endif

#endif
