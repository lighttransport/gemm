/*
 * sam3d_body_cpu.c — translation unit that pulls in the CPU decoder
 * (sam3d_body_decoder.h) and MHR (sam3d_body_mhr.h) implementations.
 * Used by the CUDA runner to drive in-the-loop CPU helpers
 * (decode_pose_raw, mhr_forward, keypoints_from_mesh, camera_project)
 * without copying the implementations into runner.c.
 *
 * SAFETENSORS_IMPLEMENTATION is provided exclusively by
 * hip_sam3d_body_runner.c; we pre-include the safetensors header here
 * (declarations only) so its include guard fires before
 * sam3d_body_decoder.h's own SAFETENSORS_IMPLEMENTATION-prefixed include
 * runs, leaving the runner.c TU as the single provider of safetensors
 * symbols.
 *
 * SAM3D_BODY_DECODER_FULL_IMPLEMENTATION is intentionally NOT set —
 * the CUDA runner orchestrates the per-layer decoder + MHR-in-the-loop
 * cycle manually so the heavy GEMMs run on the GPU.
 */

#include "../../common/safetensors.h"

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#define SAM3D_BODY_MHR_IMPLEMENTATION
#include "../../common/sam3d_body_decoder.h"
#include "../../common/sam3d_body_mhr.h"
