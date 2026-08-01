/*
 * cpu_trellis2_runner.h - thin CPU TRELLIS.2 Stage 1 runner (image -> 3D structure)
 *
 * Wraps the header-only CPU modules (dinov3.h + trellis2_stage1.h +
 * trellis2_ss_decoder.h) behind a small opaque-handle C ABI that mirrors
 * cuda_trellis2_runner.h, so the Python ctypes binding
 * (ref/trellis2/trellis2_cmod.py) can drive CPU and CUDA the same way.
 *
 * Unlike the test harness (cpu/trellis2/test_trellis2.c), the runner loads
 * weights once and keeps them resident across predict() calls.
 *
 * Usage:
 *   cpu_trellis2_runner *r = cpu_trellis2_init(8, 1);
 *   cpu_trellis2_load_weights(r, "dinov3.st", "stage1.st", "decoder.st");
 *   float *occ = cpu_trellis2_predict(r, rgb, w, h, 42);  // [64*64*64] logits
 *   cpu_trellis2_free_buffer(occ);
 *   cpu_trellis2_free(r);
 */
#ifndef CPU_TRELLIS2_RUNNER_H
#define CPU_TRELLIS2_RUNNER_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cpu_trellis2_runner cpu_trellis2_runner;

/* Allocate a runner. n_threads<=0 picks a small default. */
cpu_trellis2_runner *cpu_trellis2_init(int n_threads, int verbose);

/* Load DINOv3 + Stage-1 DiT + structure decoder weights (kept resident).
 * Returns 0 on success, nonzero on any load failure. */
int cpu_trellis2_load_weights(cpu_trellis2_runner *r,
                              const char *dinov3_path,
                              const char *stage1_path,
                              const char *decoder_path);

/* Run the full pipeline on a raw RGB image (HWC uint8, 3 channels).
 * Returns a malloc'd CPU float[64*64*64] occupancy grid (logits), or NULL.
 * The CPU sampler's step count / guidance are fixed by the checkpoint, so
 * there are no n_steps / cfg arguments here. Caller frees via
 * cpu_trellis2_free_buffer(). */
float *cpu_trellis2_predict(cpu_trellis2_runner *r,
                            const uint8_t *rgb, int w, int h,
                            uint32_t seed);

/* Free a buffer returned by cpu_trellis2_predict (plain CPU malloc).
 * Provided so ctypes/FFI callers free with this .so's allocator. */
void cpu_trellis2_free_buffer(void *p);

void cpu_trellis2_free(cpu_trellis2_runner *r);

#ifdef __cplusplus
}
#endif

#endif /* CPU_TRELLIS2_RUNNER_H */
