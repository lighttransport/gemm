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
 *   t2_fdg_mesh t2_fdg_to_mesh_bzyx(const int32_t *coords, const float *feats,
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

/* Variant for coords stored as [N,4] int32 (batch, z, y, x). */
t2_fdg_mesh t2_fdg_to_mesh_bzyx(const int32_t *coords, const float *feats,
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

static unsigned fdg_hash_slot(int64_t key, int capacity) {
    uint32_t h = (uint32_t)(((uint64_t)key * 0x9E3779B97F4A7C15ULL) >> 32);
    return (unsigned)(((uint64_t)h * (uint32_t)capacity) >> 32);
}

static fdg_hash fdg_hash_build(const int32_t *coords, int N, int stride, int base) {
    int cap = N * 4;  /* load factor 0.25 */
    if (cap < 64) cap = 64;
    fdg_hash h;
    h.capacity = cap;
    h.keys = (int64_t *)malloc((size_t)cap * sizeof(int64_t));
    h.vals = (int *)malloc((size_t)cap * sizeof(int));
    memset(h.keys, 0xFF, (size_t)cap * sizeof(int64_t));  /* -1 = empty */

    for (int i = 0; i < N; i++) {
        const int32_t *c = coords + (size_t)i * stride + base;
        int64_t key = fdg_hash_key(c[0], c[1], c[2]);
        unsigned slot = fdg_hash_slot(key, cap);
        while (h.keys[slot] != -1) {
            slot++;
            if (slot == (unsigned)cap) slot = 0;
        }
        h.keys[slot] = key;
        h.vals[slot] = i;
    }
    return h;
}

static int fdg_hash_lookup(const fdg_hash *h, int z, int y, int x) {
    int64_t key = fdg_hash_key(z, y, x);
    unsigned slot = fdg_hash_slot(key, h->capacity);
    while (1) {
        if (h->keys[slot] == key) return h->vals[slot];
        if (h->keys[slot] == -1) return -1;
        slot++;
        if (slot == (unsigned)h->capacity) slot = 0;
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

static t2_fdg_mesh t2_fdg_to_mesh_strided(const int32_t *coords, const float *feats,
                                             int N, int stride, int base,
                                             float voxel_size, const float aabb[6]) {
    t2_fdg_mesh mesh = {0};

    /* 1. Compute mesh vertices: (coord + dual_vertex) * voxel_size + aabb_min */
    float *verts = (float *)malloc((size_t)N * 3 * sizeof(float));
    for (int i = 0; i < N; i++) {
        const int32_t *c = coords + (size_t)i * stride + base;
        /* feats[i, 0:3] = vertex offsets (sigmoid applied by caller) */
        float vx = feats[i * 7 + 0];
        float vy = feats[i * 7 + 1];
        float vz = feats[i * 7 + 2];
        verts[i * 3 + 0] = ((float)c[2] + vx) * voxel_size + aabb[0];  /* x */
        verts[i * 3 + 1] = ((float)c[1] + vy) * voxel_size + aabb[1];  /* y */
        verts[i * 3 + 2] = ((float)c[0] + vz) * voxel_size + aabb[2];  /* z */
    }

    /* 2. Build spatial hash */
    fdg_hash hash = fdg_hash_build(coords, N, stride, base);

    /* 3. Find quads from intersected edges and split them immediately.
     * This preserves the old quad scan order while avoiding a large temporary
     * quad list before the final triangle buffer. */
    int max_tris = N * 3;
    if (max_tris < 2) max_tris = 2;
    int *tris = (int *)malloc((size_t)max_tris * 3 * sizeof(int));
    int n_quads = 0;
    int n_tris = 0;

    for (int i = 0; i < N; i++) {
        const int32_t *c = coords + (size_t)i * stride + base;
        int z = c[0], y = c[1], x = c[2];
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

            if (n_tris + 2 > max_tris) {
                max_tris *= 2;
                tris = (int *)realloc(tris, (size_t)max_tris * 3 * sizeof(int));
            }
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
            n_quads++;
        }
    }

    fdg_hash_free(&hash);
    if (n_tris > 0) {
        int *shrunk = (int *)realloc(tris, (size_t)n_tris * 3 * sizeof(int));
        if (shrunk) tris = shrunk;
    }

    mesh.vertices = verts;
    mesh.triangles = tris;
    mesh.n_verts = N;
    mesh.n_tris = n_tris;

    fprintf(stderr, "fdg_mesh: %d verts, %d quads -> %d triangles\n", N, n_quads, n_tris);
    return mesh;
}

t2_fdg_mesh t2_fdg_to_mesh(const int32_t *coords, const float *feats,
                              int N, float voxel_size, const float aabb[6]) {
    return t2_fdg_to_mesh_strided(coords, feats, N, 3, 0, voxel_size, aabb);
}

t2_fdg_mesh t2_fdg_to_mesh_bzyx(const int32_t *coords, const float *feats,
                                   int N, float voxel_size, const float aabb[6]) {
    return t2_fdg_to_mesh_strided(coords, feats, N, 4, 1, voxel_size, aabb);
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
