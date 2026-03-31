/*
 * trellis2_fdg_mesh.h - Flexible Dual Grid mesh extraction
 *
 * Converts sparse voxel predictions [N, 7] to triangle mesh.
 * Port of o_voxel/convert/flexible_dual_grid.py::flexible_dual_grid_to_mesh
 *
 * Usage:
 *   #define T2_FDG_MESH_IMPLEMENTATION
 *   #include "trellis2_fdg_mesh.h"
 *
 * API:
 *   t2_fdg_mesh t2_fdg_to_mesh(const int32_t *coords, const float *feats,
 *       int N, float voxel_size, const float aabb[6]);
 *   void t2_fdg_mesh_free(t2_fdg_mesh *m);
 *   int t2_fdg_write_obj(const char *path, const t2_fdg_mesh *m);
 */
#ifndef T2_FDG_MESH_H
#define T2_FDG_MESH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *vertices;    /* [n_verts, 3] */
    int   *triangles;   /* [n_tris, 3] */
    int    n_verts;
    int    n_tris;
} t2_fdg_mesh;

/* Extract mesh from shape decoder output.
 * coords: [N, 3] int32 (z, y, x) — NO batch dimension
 * feats: [N, 7] = (vertex_xyz(3), intersected(3), split_weight(1))
 * voxel_size: size of each voxel in world units
 * aabb: [6] = (min_x, min_y, min_z, max_x, max_y, max_z) */
t2_fdg_mesh t2_fdg_to_mesh(const int32_t *coords, const float *feats,
                              int N, float voxel_size, const float aabb[6]);

void t2_fdg_mesh_free(t2_fdg_mesh *m);
int t2_fdg_write_obj(const char *path, const t2_fdg_mesh *m);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef T2_FDG_MESH_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Simple spatial hash for voxel lookup */
typedef struct {
    int64_t *keys;
    int     *vals;
    int      capacity;
} fdg_hash;

static int64_t fdg_hash_key(int z, int y, int x) {
    return ((int64_t)z << 40) | ((int64_t)(y & 0xFFFFF) << 20) | (int64_t)(x & 0xFFFFF);
}

static fdg_hash fdg_hash_build(const int32_t *coords, int N) {
    int cap = N * 4;  /* load factor 0.25 */
    if (cap < 64) cap = 64;
    fdg_hash h;
    h.capacity = cap;
    h.keys = (int64_t *)malloc((size_t)cap * sizeof(int64_t));
    h.vals = (int *)malloc((size_t)cap * sizeof(int));
    memset(h.keys, 0xFF, (size_t)cap * sizeof(int64_t));  /* -1 = empty */

    for (int i = 0; i < N; i++) {
        int64_t key = fdg_hash_key(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
        unsigned slot = (unsigned)(key * 0x9E3779B97F4A7C15ULL >> 32) % (unsigned)cap;
        while (h.keys[slot] != -1) { slot = (slot + 1) % (unsigned)cap; }
        h.keys[slot] = key;
        h.vals[slot] = i;
    }
    return h;
}

static int fdg_hash_lookup(const fdg_hash *h, int z, int y, int x) {
    int64_t key = fdg_hash_key(z, y, x);
    unsigned slot = (unsigned)(key * 0x9E3779B97F4A7C15ULL >> 32) % (unsigned)h->capacity;
    while (1) {
        if (h->keys[slot] == key) return h->vals[slot];
        if (h->keys[slot] == -1) return -1;
        slot = (slot + 1) % (unsigned)h->capacity;
    }
}

static void fdg_hash_free(fdg_hash *h) {
    free(h->keys); free(h->vals);
}

/* Edge neighbor offsets: for each axis (x,y,z), the 4 voxels sharing that edge */
static const int edge_offsets[3][4][3] = {
    /* x-axis edges: 4 voxels around an x-edge */
    {{0,0,0}, {0,0,1}, {0,1,1}, {0,1,0}},
    /* y-axis edges */
    {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}},
    /* z-axis edges */
    {{0,0,0}, {0,1,0}, {1,1,0}, {1,0,0}},
};

static float vec3_cross_dot(const float a[3], const float b[3], const float c[3], const float d[3]) {
    /* cross(b-a, c-a) · cross(c-b, d-b) */
    float n1[3], n2[3];
    float ba[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    float ca[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
    n1[0] = ba[1]*ca[2] - ba[2]*ca[1];
    n1[1] = ba[2]*ca[0] - ba[0]*ca[2];
    n1[2] = ba[0]*ca[1] - ba[1]*ca[0];
    float cb[3] = {c[0]-b[0], c[1]-b[1], c[2]-b[2]};
    float db[3] = {d[0]-b[0], d[1]-b[1], d[2]-b[2]};
    n2[0] = cb[1]*db[2] - cb[2]*db[1];
    n2[1] = cb[2]*db[0] - cb[0]*db[2];
    n2[2] = cb[0]*db[1] - cb[1]*db[0];
    return fabsf(n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]);
}

t2_fdg_mesh t2_fdg_to_mesh(const int32_t *coords, const float *feats,
                              int N, float voxel_size, const float aabb[6]) {
    t2_fdg_mesh mesh = {0};

    /* 1. Compute mesh vertices: (coord + dual_vertex) * voxel_size + aabb_min */
    float *verts = (float *)malloc((size_t)N * 3 * sizeof(float));
    for (int i = 0; i < N; i++) {
        /* feats[i, 0:3] = vertex offsets (sigmoid applied by caller) */
        float vx = feats[i * 7 + 0];
        float vy = feats[i * 7 + 1];
        float vz = feats[i * 7 + 2];
        verts[i * 3 + 0] = ((float)coords[i * 3 + 2] + vx) * voxel_size + aabb[0];  /* x */
        verts[i * 3 + 1] = ((float)coords[i * 3 + 1] + vy) * voxel_size + aabb[1];  /* y */
        verts[i * 3 + 2] = ((float)coords[i * 3 + 0] + vz) * voxel_size + aabb[2];  /* z */
    }

    /* 2. Build spatial hash */
    fdg_hash hash = fdg_hash_build(coords, N);

    /* 3. Find quads from intersected edges */
    int max_quads = N * 3;
    int *quads = (int *)malloc((size_t)max_quads * 4 * sizeof(int));
    int n_quads = 0;

    for (int i = 0; i < N; i++) {
        int z = coords[i * 3], y = coords[i * 3 + 1], x = coords[i * 3 + 2];
        for (int axis = 0; axis < 3; axis++) {
            /* Check if this voxel's edge in this axis is intersected */
            if (feats[i * 7 + 3 + axis] <= 0.0f) continue;

            /* Find 4 neighboring voxels sharing this edge */
            int qi[4];
            int valid = 1;
            for (int k = 0; k < 4; k++) {
                int nz = z + edge_offsets[axis][k][0];
                int ny = y + edge_offsets[axis][k][1];
                int nx = x + edge_offsets[axis][k][2];
                qi[k] = fdg_hash_lookup(&hash, nz, ny, nx);
                if (qi[k] < 0) { valid = 0; break; }
            }
            if (!valid) continue;

            if (n_quads >= max_quads) {
                max_quads *= 2;
                quads = (int *)realloc(quads, (size_t)max_quads * 4 * sizeof(int));
            }
            memcpy(quads + n_quads * 4, qi, 4 * sizeof(int));
            n_quads++;
        }
    }

    fdg_hash_free(&hash);

    /* 4. Split quads into triangles (choose split minimizing normal deviation) */
    int *tris = (int *)malloc((size_t)n_quads * 6 * sizeof(int));
    int n_tris = 0;

    for (int q = 0; q < n_quads; q++) {
        int *qi = quads + q * 4;
        float *v0 = verts + qi[0] * 3, *v1 = verts + qi[1] * 3;
        float *v2 = verts + qi[2] * 3, *v3 = verts + qi[3] * 3;

        /* Try split 1: (0,1,2) + (0,2,3) */
        float align0 = vec3_cross_dot(v0, v1, v2, v3);
        /* Try split 2: (0,1,3) + (3,1,2) */
        float align1 = vec3_cross_dot(v0, v1, v3, v1);

        /* Use split_weight if available */
        float sw0 = feats[qi[0]*7+6] * feats[qi[2]*7+6];
        float sw1 = feats[qi[1]*7+6] * feats[qi[3]*7+6];

        if (sw0 > sw1) {
            /* Split 1: (0,1,2), (0,2,3) */
            tris[n_tris*3+0]=qi[0]; tris[n_tris*3+1]=qi[1]; tris[n_tris*3+2]=qi[2]; n_tris++;
            tris[n_tris*3+0]=qi[0]; tris[n_tris*3+1]=qi[2]; tris[n_tris*3+2]=qi[3]; n_tris++;
        } else {
            /* Split 2: (0,1,3), (3,1,2) */
            tris[n_tris*3+0]=qi[0]; tris[n_tris*3+1]=qi[1]; tris[n_tris*3+2]=qi[3]; n_tris++;
            tris[n_tris*3+0]=qi[3]; tris[n_tris*3+1]=qi[1]; tris[n_tris*3+2]=qi[2]; n_tris++;
        }
    }
    free(quads);

    mesh.vertices = verts;
    mesh.triangles = tris;
    mesh.n_verts = N;
    mesh.n_tris = n_tris;

    fprintf(stderr, "fdg_mesh: %d verts, %d quads -> %d triangles\n", N, n_quads, n_tris);
    return mesh;
}

void t2_fdg_mesh_free(t2_fdg_mesh *m) {
    free(m->vertices); free(m->triangles);
    m->vertices = NULL; m->triangles = NULL;
    m->n_verts = m->n_tris = 0;
}

int t2_fdg_write_obj(const char *path, const t2_fdg_mesh *m) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "# FDG mesh: %d verts, %d tris\n", m->n_verts, m->n_tris);
    for (int i = 0; i < m->n_verts; i++)
        fprintf(f, "v %f %f %f\n", m->vertices[i*3], m->vertices[i*3+1], m->vertices[i*3+2]);
    for (int i = 0; i < m->n_tris; i++)
        fprintf(f, "f %d %d %d\n", m->triangles[i*3]+1, m->triangles[i*3+1]+1, m->triangles[i*3+2]+1);
    fclose(f);
    fprintf(stderr, "Wrote %s\n", path);
    return 0;
}

#endif /* T2_FDG_MESH_IMPLEMENTATION */
#endif /* T2_FDG_MESH_H */
