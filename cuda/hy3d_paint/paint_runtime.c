/*
 * paint_runtime.c - Single TU that owns the heavy IMPLEMENTATION macros for
 * shared runtime headers, so any number of paint-pipeline TUs can include the
 * stage runner headers (cuda_paint_unet_runner.h, cuda_paint_vae_runner.h, ...)
 * without duplicate-symbol errors at link time.
 *
 * Mirror the IMPLEMENTATION-guard pattern landed in 5ddffe9 / 1a29b84:
 *   - The runner headers gate their dep impls on <RUNNER>_IMPLEMENTATION,
 *     defined by exactly one TU per binary.
 *   - For the multi-stage paint pipeline binary, that "one TU" is this file —
 *     no stage runner needs to claim the impls anymore.
 */

#include "../cuew.h"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"
