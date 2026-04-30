/*
 * sam3d_xatlas.h - small C ABI wrapper around vendored xatlas.
 *
 * Produces an indexed mesh with xatlas-split vertices, normalized UVs,
 * and xrefs back to the caller's original vertex array.
 */
#ifndef SAM3D_XATLAS_H
#define SAM3D_XATLAS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sam3d_xatlas_mesh {
    float *positions;      /* [n_vertices, 3] copied from input via xrefs */
    float *uvs;            /* [n_vertices, 2] normalized [0,1] */
    uint32_t *xrefs;       /* [n_vertices] original input vertex index */
    uint32_t *indices;     /* [n_indices], triangles */
    int n_vertices;
    int n_indices;
    int atlas_width;
    int atlas_height;
    int chart_count;
} sam3d_xatlas_mesh;

int sam3d_xatlas_generate(const float *positions, int n_positions,
                          const int *triangles, int n_triangles,
                          int atlas_resolution,
                          sam3d_xatlas_mesh *out);

void sam3d_xatlas_free(sam3d_xatlas_mesh *m);

#ifdef __cplusplus
}
#endif

#endif /* SAM3D_XATLAS_H */
