/*
 * gs_ply_writer.h — INRIA-gsplat-compatible 3D Gaussian PLY writer.
 *
 * Binary little-endian PLY with fields:
 *   x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity
 *   scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
 *
 * Storage convention (matches the original INRIA 3D-GS paper and the
 * gsplat viewer ecosystem):
 *   - opacity is stored as the pre-sigmoid logit; viewers apply sigmoid.
 *   - scale_{0,1,2} are stored as log(scale); viewers apply exp.
 *   - rot_{0,1,2,3} are stored un-normalized; viewers normalize to a
 *     unit quaternion before use.
 *
 * This writer is purely format-concerned — callers are responsible for
 * applying model-specific biases / activations so the stored values
 * match the storage convention above. For SAM-3D-Objects that means:
 *     opacity_stored = raw_opacity + opacity_bias
 *     scale_stored   = log(softplus(raw_scaling + inv_softplus(scaling_bias)))
 *
 * Single-header; define GS_PLY_WRITER_IMPLEMENTATION in exactly one TU.
 */
#ifndef GS_PLY_WRITER_H
#define GS_PLY_WRITER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Write n gaussians to path as an INRIA-compatible binary PLY.
 * All inputs are fp32, tightly packed. Pass NULL for `normals` to
 * emit zeros.
 * Returns 0 on success, non-zero on error. */
int gs_ply_write(const char *path, int n,
                 const float *xyz,      /* [n, 3] world-space positions      */
                 const float *normals,  /* [n, 3] or NULL (written as 0)     */
                 const float *f_dc,     /* [n, 3] SH DC coefficients         */
                 const float *opacity,  /* [n]    pre-sigmoid logit          */
                 const float *scaling,  /* [n, 3] pre-exp log-scale          */
                 const float *rotation);/* [n, 4] quaternion (un-normalized) */

#ifdef __cplusplus
}
#endif

#endif /* GS_PLY_WRITER_H */

/* ================================================================== */
#ifdef GS_PLY_WRITER_IMPLEMENTATION
#ifndef GS_PLY_WRITER_IMPL_ONCE
#define GS_PLY_WRITER_IMPL_ONCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int gs_ply_is_little_endian(void) {
    const unsigned int u = 1;
    return *(const unsigned char *)&u == 1;
}

int gs_ply_write(const char *path, int n,
                 const float *xyz, const float *normals, const float *f_dc,
                 const float *opacity, const float *scaling, const float *rotation)
{
    if (!path || n < 0 || !xyz || !f_dc || !opacity || !scaling || !rotation)
        return -1;
    if (!gs_ply_is_little_endian()) {
        fprintf(stderr, "gs_ply_write: big-endian host not supported\n");
        return -2;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "gs_ply_write: cannot open %s for writing\n", path);
        return -3;
    }

    fprintf(f,
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex %d\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n", n);

    const float zero3[3] = {0.0f, 0.0f, 0.0f};
    float row[17];
    for (int i = 0; i < n; i++) {
        memcpy(&row[0],  xyz + i * 3,          3 * sizeof(float));
        memcpy(&row[3],  normals ? normals + i * 3 : zero3, 3 * sizeof(float));
        memcpy(&row[6],  f_dc + i * 3,         3 * sizeof(float));
        row[9] = opacity[i];
        memcpy(&row[10], scaling + i * 3,      3 * sizeof(float));
        memcpy(&row[13], rotation + i * 4,     4 * sizeof(float));
        if (fwrite(row, sizeof(float), 17, f) != 17) {
            fprintf(stderr, "gs_ply_write: short write on vertex %d\n", i);
            fclose(f);
            return -4;
        }
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "gs_ply_write: close failed\n");
        return -5;
    }
    return 0;
}

#endif /* GS_PLY_WRITER_IMPL_ONCE */
#endif /* GS_PLY_WRITER_IMPLEMENTATION */
