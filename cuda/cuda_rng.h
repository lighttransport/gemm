/*
 * cuda_rng.h - Shared RNG helpers for CUDA runners
 *
 * Thin wrapper around common/philox_rng.h so CUDA model runners can share
 * the same deterministic CPU-side noise generation before H2D upload.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */
#ifndef CUDA_RNG_H
#define CUDA_RNG_H

#include <stddef.h>
#include <stdint.h>
#include "../common/philox_rng.h"

static inline void cuda_rng_fill_uniform_f32(float *out, size_t n, uint64_t seed, uint64_t offset_u32) {
    philox_fill_uniform_f32(out, n, seed, offset_u32);
}

static inline void cuda_rng_fill_normal_f32(float *out, size_t n, uint64_t seed, uint64_t offset_u32) {
    philox_fill_normal_f32(out, n, seed, offset_u32);
}

#endif /* CUDA_RNG_H */
