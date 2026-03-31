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

/* Write OBJ with per-vertex colors (v x y z r g b extension) */
int t2_pbr_write_colored_obj(const char *path,
                               const float *vertices, const int *triangles,
                               int n_verts, int n_tris,
                               const t2_pbr_attr *colors);

/* Write OBJ + MTL with per-triangle UV mapping and PBR texture maps.
 * Generates: <base>.obj, <base>.mtl, <base>_basecolor.ppm,
 *            <base>_roughness.ppm, <base>_metallic.ppm
 * tex_size: texture resolution (e.g., 1024 or 2048) */
int t2_pbr_write_textured_obj(const char *base_path,
                                const float *vertices, const int *triangles,
                                int n_verts, int n_tris,
                                const t2_pbr_attr *vert_colors,
                                int tex_size);

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

/* ---- Simple PPM image writer ---- */
static void t2pbr_write_ppm(const char *path, const uint8_t *rgb, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, (size_t)w * h * 3, f);
    fclose(f);
}

/* ---- Per-triangle UV packing ---- */
/* Each triangle gets a small rectangular patch in the texture atlas.
 * Triangles are packed row-by-row with a configurable patch size. */

static void t2pbr_barycentric(float px, float py,
                                float x0, float y0, float x1, float y1, float x2, float y2,
                                float *u, float *v, float *w) {
    float d = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2);
    if (fabsf(d) < 1e-10f) { *u = *v = *w = 1.0f/3.0f; return; }
    *u = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / d;
    *v = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / d;
    *w = 1.0f - *u - *v;
}

int t2_pbr_write_textured_obj(const char *base_path,
                                const float *vertices, const int *triangles,
                                int n_verts, int n_tris,
                                const t2_pbr_attr *vert_colors,
                                int tex_size) {
    /* Patch size for each triangle in the atlas (in pixels) */
    int patch = 8;  /* 8x8 pixel patch per triangle */
    int margin = 1; /* 1 pixel margin */
    int psz = patch + margin * 2;
    int cols = tex_size / psz;
    if (cols < 1) cols = 1;
    int rows_needed = (n_tris + cols - 1) / cols;
    if (rows_needed * psz > tex_size) {
        /* Need larger texture or smaller patches */
        patch = 4; margin = 1; psz = patch + margin * 2;
        cols = tex_size / psz;
        rows_needed = (n_tris + cols - 1) / cols;
    }

    /* Allocate UV coordinates [n_tris, 3, 2] */
    float *uvs = (float *)calloc((size_t)n_tris * 6, sizeof(float));

    /* Allocate texture images */
    uint8_t *tex_bc = (uint8_t *)calloc((size_t)tex_size * tex_size * 3, 1); /* base color */
    uint8_t *tex_mr = (uint8_t *)calloc((size_t)tex_size * tex_size * 3, 1); /* metallic-roughness */

    /* Pack triangles into atlas and bake vertex colors */
    for (int ti = 0; ti < n_tris; ti++) {
        int row = ti / cols;
        int col = ti % cols;
        if (row * psz + psz > tex_size) continue;  /* overflow, skip */

        /* UV origin for this triangle's patch */
        float u_base = (float)(col * psz + margin) / (float)tex_size;
        float v_base = (float)(row * psz + margin) / (float)tex_size;
        float u_span = (float)patch / (float)tex_size;
        float v_span = (float)patch / (float)tex_size;

        /* Map triangle to UV rectangle: v0=(0,0), v1=(1,0), v2=(0.5,1) */
        uvs[ti * 6 + 0] = u_base;                  uvs[ti * 6 + 1] = v_base;
        uvs[ti * 6 + 2] = u_base + u_span;         uvs[ti * 6 + 3] = v_base;
        uvs[ti * 6 + 4] = u_base + u_span * 0.5f;  uvs[ti * 6 + 5] = v_base + v_span;

        /* Get vertex colors for this triangle */
        int i0 = triangles[ti * 3 + 0];
        int i1 = triangles[ti * 3 + 1];
        int i2 = triangles[ti * 3 + 2];
        const t2_pbr_attr *c0 = &vert_colors[i0];
        const t2_pbr_attr *c1 = &vert_colors[i1];
        const t2_pbr_attr *c2 = &vert_colors[i2];

        /* Rasterize into texture patch */
        int px0 = col * psz + margin;
        int py0 = row * psz + margin;
        for (int py = 0; py < patch; py++) {
            for (int px = 0; px < patch; px++) {
                /* Map pixel to barycentric coords in the triangle UV */
                float fx = ((float)px + 0.5f) / (float)patch;
                float fy = ((float)py + 0.5f) / (float)patch;
                float bu, bv, bw;
                t2pbr_barycentric(fx, fy, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f, &bu, &bv, &bw);

                /* Skip if outside triangle */
                if (bu < -0.01f || bv < -0.01f || bw < -0.01f) continue;
                /* Clamp */
                if (bu < 0) bu = 0; if (bv < 0) bv = 0; if (bw < 0) bw = 0;
                float s = bu + bv + bw; bu /= s; bv /= s; bw /= s;

                /* Interpolate PBR attributes */
                float r = bu * c0->r + bv * c1->r + bw * c2->r;
                float g = bu * c0->g + bv * c1->g + bw * c2->g;
                float b = bu * c0->b + bv * c1->b + bw * c2->b;
                float met = bu * c0->metallic + bv * c1->metallic + bw * c2->metallic;
                float rgh = bu * c0->roughness + bv * c1->roughness + bw * c2->roughness;

                int tx = px0 + px, ty = py0 + py;
                if (tx >= tex_size || ty >= tex_size) continue;
                /* Flip Y for image (top-left origin) */
                int iy = tex_size - 1 - ty;
                int idx = (iy * tex_size + tx) * 3;
                tex_bc[idx+0] = (uint8_t)(r < 0 ? 0 : r > 1 ? 255 : (int)(r * 255));
                tex_bc[idx+1] = (uint8_t)(g < 0 ? 0 : g > 1 ? 255 : (int)(g * 255));
                tex_bc[idx+2] = (uint8_t)(b < 0 ? 0 : b > 1 ? 255 : (int)(b * 255));
                /* Metallic-roughness: [unused, roughness, metallic] per glTF convention */
                tex_mr[idx+0] = 0;
                tex_mr[idx+1] = (uint8_t)(rgh < 0 ? 0 : rgh > 1 ? 255 : (int)(rgh * 255));
                tex_mr[idx+2] = (uint8_t)(met < 0 ? 0 : met > 1 ? 255 : (int)(met * 255));
            }
        }
    }

    /* Write texture images */
    char path_buf[512];
    snprintf(path_buf, sizeof(path_buf), "%s_basecolor.ppm", base_path);
    t2pbr_write_ppm(path_buf, tex_bc, tex_size, tex_size);
    fprintf(stderr, "Wrote %s\n", path_buf);

    snprintf(path_buf, sizeof(path_buf), "%s_metallic_roughness.ppm", base_path);
    t2pbr_write_ppm(path_buf, tex_mr, tex_size, tex_size);
    fprintf(stderr, "Wrote %s\n", path_buf);

    free(tex_bc); free(tex_mr);

    /* Write MTL file */
    snprintf(path_buf, sizeof(path_buf), "%s.mtl", base_path);
    {
        FILE *f = fopen(path_buf, "w");
        if (!f) { free(uvs); return -1; }
        const char *bn = strrchr(base_path, '/');
        bn = bn ? bn + 1 : base_path;
        fprintf(f, "# TRELLIS.2 PBR material\n");
        fprintf(f, "newmtl trellis2_pbr\n");
        fprintf(f, "Kd 1.0 1.0 1.0\n");
        fprintf(f, "map_Kd %s_basecolor.ppm\n", bn);
        fprintf(f, "# PBR extensions\n");
        fprintf(f, "map_Pr %s_metallic_roughness.ppm\n", bn);  /* roughness map */
        fprintf(f, "map_Pm %s_metallic_roughness.ppm\n", bn);  /* metallic map */
        fclose(f);
        fprintf(stderr, "Wrote %s\n", path_buf);
    }

    /* Write OBJ file with UVs */
    snprintf(path_buf, sizeof(path_buf), "%s.obj", base_path);
    {
        FILE *f = fopen(path_buf, "w");
        if (!f) { free(uvs); return -1; }
        const char *bn = strrchr(base_path, '/');
        bn = bn ? bn + 1 : base_path;
        fprintf(f, "# TRELLIS.2 textured mesh: %d verts, %d tris\n", n_verts, n_tris);
        fprintf(f, "mtllib %s.mtl\n", bn);
        fprintf(f, "usemtl trellis2_pbr\n\n");

        /* Vertices */
        for (int i = 0; i < n_verts; i++)
            fprintf(f, "v %f %f %f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);

        /* UV coordinates (3 per triangle) */
        for (int ti = 0; ti < n_tris; ti++) {
            fprintf(f, "vt %f %f\n", uvs[ti*6+0], uvs[ti*6+1]);
            fprintf(f, "vt %f %f\n", uvs[ti*6+2], uvs[ti*6+3]);
            fprintf(f, "vt %f %f\n", uvs[ti*6+4], uvs[ti*6+5]);
        }

        /* Faces with UV indices (f v/vt v/vt v/vt) */
        for (int ti = 0; ti < n_tris; ti++) {
            int vt_base = ti * 3 + 1;  /* 1-indexed */
            fprintf(f, "f %d/%d %d/%d %d/%d\n",
                    triangles[ti*3+0]+1, vt_base,
                    triangles[ti*3+1]+1, vt_base+1,
                    triangles[ti*3+2]+1, vt_base+2);
        }
        fclose(f);
        fprintf(stderr, "Wrote %s (%d verts, %d tris, UV mapped)\n", path_buf, n_verts, n_tris);
    }

    free(uvs);
    return 0;
}

#endif /* T2_PBR_IMPLEMENTATION */
#endif /* T2_PBR_H */
