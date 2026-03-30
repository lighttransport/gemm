/*
 * trellis2_pbr.h - PBR material extraction from texture decoder output
 *
 * Converts 6-channel sparse voxel field to per-vertex PBR attributes
 * by trilinear sampling at mesh vertex positions.
 *
 * Usage:
 *   #define T2_PBR_IMPLEMENTATION
 *   #include "trellis2_pbr.h"
 *
 * API:
 *   t2_pbr_field t2_pbr_from_decoder(feats, coords, N, resolution);
 *   void t2_pbr_sample_vertices(field, vertices, n_verts, out_colors);
 *   void t2_pbr_free(field);
 *   int  t2_pbr_write_colored_obj(path, mesh_verts, mesh_tris, n_verts, n_tris, colors);
 */
#ifndef T2_PBR_H
#define T2_PBR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Per-vertex PBR attributes */
typedef struct {
    float r, g, b;       /* base color [0,1] */
    float metallic;      /* [0,1] */
    float roughness;     /* [0,1] */
    float alpha;         /* [0,1] */
} t2_pbr_attr;

/* Sparse PBR voxel field */
typedef struct {
    float *feats;        /* [N, 6]: RGB, metallic, roughness, alpha */
    int32_t *coords;     /* [N, 3]: z, y, x (no batch dim) */
    int N;
    int resolution;      /* grid resolution (e.g., 512) */
    /* Hash table for fast lookup */
    uint64_t *hash_keys;
    int32_t  *hash_vals;
    int hash_cap;
} t2_pbr_field;

/* Create PBR field from texture decoder output.
 * feats: [N, 6] raw decoder output (will be scaled * 0.5 + 0.5)
 * coords: [N, 4] (batch, z, y, x) — batch dim stripped
 * resolution: voxel grid resolution */
t2_pbr_field t2_pbr_from_decoder(const float *feats, const int32_t *coords,
                                   int N, int resolution);

/* Sample PBR attributes at mesh vertex positions via trilinear interpolation.
 * vertices: [n_verts, 3] in [-0.5, 0.5] normalized space
 * out: [n_verts] PBR attributes */
void t2_pbr_sample_vertices(const t2_pbr_field *field,
                              const float *vertices, int n_verts,
                              t2_pbr_attr *out);

void t2_pbr_free(t2_pbr_field *f);

/* Write OBJ with per-vertex colors (Kd in MTL) */
int t2_pbr_write_colored_obj(const char *path,
                               const float *vertices, const int *triangles,
                               int n_verts, int n_tris,
                               const t2_pbr_attr *colors);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef T2_PBR_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t t2pbr_hash_key(int z, int y, int x) {
    return ((uint64_t)(z & 0xFFFFF) << 40) | ((uint64_t)(y & 0xFFFFF) << 20) | (uint64_t)(x & 0xFFFFF);
}

t2_pbr_field t2_pbr_from_decoder(const float *feats, const int32_t *coords,
                                   int N, int resolution) {
    t2_pbr_field f = {0};
    f.N = N;
    f.resolution = resolution;

    /* Copy and scale features: raw -> * 0.5 + 0.5 -> clamp [0,1] */
    f.feats = (float *)malloc((size_t)N * 6 * sizeof(float));
    for (int i = 0; i < N * 6; i++) {
        float v = feats[i] * 0.5f + 0.5f;
        f.feats[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }

    /* Strip batch dimension from coords: [N, 4] -> [N, 3] */
    f.coords = (int32_t *)malloc((size_t)N * 3 * sizeof(int32_t));
    for (int i = 0; i < N; i++) {
        f.coords[i * 3 + 0] = coords[i * 4 + 1];  /* z */
        f.coords[i * 3 + 1] = coords[i * 4 + 2];  /* y */
        f.coords[i * 3 + 2] = coords[i * 4 + 3];  /* x */
    }

    /* Build hash table */
    int cap = N * 4; if (cap < 64) cap = 64;
    /* Round up to power of 2 */
    int p = 1; while (p < cap) p <<= 1; cap = p;
    f.hash_cap = cap;
    f.hash_keys = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
    f.hash_vals = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    memset(f.hash_keys, 0xFF, (size_t)cap * sizeof(uint64_t));

    for (int i = 0; i < N; i++) {
        uint64_t key = t2pbr_hash_key(f.coords[i*3], f.coords[i*3+1], f.coords[i*3+2]);
        unsigned slot = (unsigned)((key * 0x9E3779B97F4A7C15ULL) >> 32) & (unsigned)(cap - 1);
        while (f.hash_keys[slot] != (uint64_t)-1) slot = (slot + 1) & (unsigned)(cap - 1);
        f.hash_keys[slot] = key;
        f.hash_vals[slot] = i;
    }

    return f;
}

static int t2pbr_lookup(const t2_pbr_field *f, int z, int y, int x) {
    uint64_t key = t2pbr_hash_key(z, y, x);
    unsigned slot = (unsigned)((key * 0x9E3779B97F4A7C15ULL) >> 32) & (unsigned)(f->hash_cap - 1);
    for (;;) {
        if (f->hash_keys[slot] == key) return f->hash_vals[slot];
        if (f->hash_keys[slot] == (uint64_t)-1) return -1;
        slot = (slot + 1) & (unsigned)(f->hash_cap - 1);
    }
}

/* Trilinear interpolation from sparse voxel field */
static void t2pbr_trilinear(const t2_pbr_field *f, float fz, float fy, float fx,
                              float out[6]) {
    int iz = (int)floorf(fz), iy = (int)floorf(fy), ix = (int)floorf(fx);
    float dz = fz - iz, dy = fy - iy, dx = fx - ix;

    memset(out, 0, 6 * sizeof(float));
    float total_w = 0;

    /* Sample 8 corners, skip absent voxels */
    for (int dzi = 0; dzi <= 1; dzi++) {
        for (int dyi = 0; dyi <= 1; dyi++) {
            for (int dxi = 0; dxi <= 1; dxi++) {
                int ni = t2pbr_lookup(f, iz + dzi, iy + dyi, ix + dxi);
                if (ni < 0) continue;
                float w = (dzi ? dz : 1-dz) * (dyi ? dy : 1-dy) * (dxi ? dx : 1-dx);
                const float *feat = f->feats + ni * 6;
                for (int c = 0; c < 6; c++) out[c] += w * feat[c];
                total_w += w;
            }
        }
    }
    /* Renormalize if some corners were missing */
    if (total_w > 0 && total_w < 0.999f) {
        float inv = 1.0f / total_w;
        for (int c = 0; c < 6; c++) out[c] *= inv;
    }
}

void t2_pbr_sample_vertices(const t2_pbr_field *field,
                              const float *vertices, int n_verts,
                              t2_pbr_attr *out) {
    float res = (float)field->resolution;
    for (int i = 0; i < n_verts; i++) {
        /* Vertex position [-0.5, 0.5] -> voxel coords [0, resolution) */
        float vx = (vertices[i * 3 + 0] + 0.5f) * res;
        float vy = (vertices[i * 3 + 1] + 0.5f) * res;
        float vz = (vertices[i * 3 + 2] + 0.5f) * res;

        float attr[6];
        t2pbr_trilinear(field, vz, vy, vx, attr);

        out[i].r = attr[0];
        out[i].g = attr[1];
        out[i].b = attr[2];
        out[i].metallic = attr[3];
        out[i].roughness = attr[4];
        out[i].alpha = attr[5];
    }
}

void t2_pbr_free(t2_pbr_field *f) {
    free(f->feats); free(f->coords);
    free(f->hash_keys); free(f->hash_vals);
    memset(f, 0, sizeof(*f));
}

int t2_pbr_write_colored_obj(const char *path,
                               const float *vertices, const int *triangles,
                               int n_verts, int n_tris,
                               const t2_pbr_attr *colors) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "# TRELLIS.2 textured mesh: %d verts, %d tris\n", n_verts, n_tris);

    /* Write vertices with color (OBJ extension: v x y z r g b) */
    for (int i = 0; i < n_verts; i++) {
        fprintf(f, "v %f %f %f %f %f %f\n",
                vertices[i*3], vertices[i*3+1], vertices[i*3+2],
                colors[i].r, colors[i].g, colors[i].b);
    }

    /* Write faces (1-indexed) */
    for (int i = 0; i < n_tris; i++) {
        fprintf(f, "f %d %d %d\n",
                triangles[i*3]+1, triangles[i*3+1]+1, triangles[i*3+2]+1);
    }

    fclose(f);
    fprintf(stderr, "Wrote %s (%d verts, %d tris, with vertex colors)\n",
            path, n_verts, n_tris);
    return 0;
}

#endif /* T2_PBR_IMPLEMENTATION */
#endif /* T2_PBR_H */
