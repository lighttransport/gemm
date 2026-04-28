/*
 * verify_npy.h — shared helpers for sam3d verify_*.c binaries.
 *
 * Thin shim over common/npy_io.h, kept for backward compatibility with
 * the existing sam3d verify_*.c sources. Prefer including
 * "../../common/npy_io.h" (or "npy_io.h" with the right -I) directly
 * in new code.
 */
#ifndef SAM3D_VERIFY_NPY_H
#define SAM3D_VERIFY_NPY_H

#include "../../common/npy_io.h"

static inline void *verify_read_npy(const char *path, int *ndim, int *dims,
                                    int *is_f32)
{
    return npy_load(path, ndim, dims, is_f32);
}

static inline float verify_max_abs(const float *a, const float *b, int n,
                                   double *mean_out)
{
    return npy_max_abs_f32(a, b, n, mean_out);
}

#endif  /* SAM3D_VERIFY_NPY_H */
