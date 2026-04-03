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

/* ---- Image writer (PNG via stb_image_write, fallback to PPM) ---- */
#ifdef STB_IMAGE_WRITE_IMPLEMENTATION
/* stb_image_write already included with implementation */
#define T2PBR_HAS_PNG 1
#elif defined(STBI_WRITE_NO_STDIO) || defined(STBIW_MALLOC)
/* stb_image_write header already included */
#define T2PBR_HAS_PNG 1
#else
/* Try to include stb_image_write.h (header-only, no implementation here) */
#if __has_include("stb_image_write.h")
#include "stb_image_write.h"
#define T2PBR_HAS_PNG 1
#else
#define T2PBR_HAS_PNG 0
#endif
#endif

static void t2pbr_write_image(const char *path, const uint8_t *rgb, int w, int h) {
#if T2PBR_HAS_PNG
    /* Use PNG if path ends with .png */
    const char *ext = strrchr(path, '.');
    if (ext && (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0)) {
        stbi_write_png(path, w, h, 3, rgb, w * 3);
        return;
    }
#endif
    /* Fallback: PPM */
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, (size_t)w * h * 3, f);
    fclose(f);
}

/* ---- Chart-based UV unwrapping ---- */
/* Groups adjacent triangles into charts, projects each chart onto its
 * best-fit 2D plane via PCA, then packs charts into a texture atlas. */

static void t2pbr_barycentric(float px, float py,
                                float x0, float y0, float x1, float y1, float x2, float y2,
                                float *u, float *v, float *w) {
    float d = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2);
    if (fabsf(d) < 1e-10f) { *u = *v = *w = 1.0f/3.0f; return; }
    *u = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / d;
    *v = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / d;
    *w = 1.0f - *u - *v;
}

/* Build adjacency and find connected components (charts) */
static int t2pbr_find_charts(const int *triangles, int n_tris, int n_verts, int *tri_chart) {
    /* Build edge-to-triangle adjacency via vertex sharing */
    /* Simple union-find over triangles sharing an edge */
    int *parent = (int *)malloc((size_t)n_tris * sizeof(int));
    for (int i = 0; i < n_tris; i++) parent[i] = i;

    /* Find root with path compression */
    #define FIND(x) { int r = x; while (parent[r] != r) r = parent[r]; \
                      int c = x; while (parent[c] != r) { int t = parent[c]; parent[c] = r; c = t; } x = r; }
    /* Build vertex-to-triangle map */
    int *vert_tri = (int *)malloc((size_t)n_verts * sizeof(int));
    memset(vert_tri, -1, (size_t)n_verts * sizeof(int));

    for (int ti = 0; ti < n_tris; ti++) {
        for (int k = 0; k < 3; k++) {
            int vi = triangles[ti * 3 + k];
            if (vert_tri[vi] >= 0) {
                /* Merge this triangle with the one that shares vertex vi */
                int a = ti, b = vert_tri[vi];
                FIND(a); FIND(b);
                if (a != b) parent[a] = b;
            }
            vert_tri[vi] = ti;
        }
    }
    free(vert_tri);

    /* Assign chart IDs */
    int n_charts = 0;
    int *chart_id = (int *)malloc((size_t)n_tris * sizeof(int));
    memset(chart_id, -1, (size_t)n_tris * sizeof(int));
    for (int i = 0; i < n_tris; i++) {
        int r = i; FIND(r);
        if (chart_id[r] < 0) chart_id[r] = n_charts++;
        tri_chart[i] = chart_id[r];
    }
    #undef FIND
    free(parent); free(chart_id);
    return n_charts;
}

int t2_pbr_write_textured_obj(const char *base_path,
                                const float *vertices, const int *triangles,
                                int n_verts, int n_tris,
                                const t2_pbr_attr *vert_colors,
                                int tex_size) {
    /* 1. Find charts (connected components of triangles) */
    int *tri_chart = (int *)malloc((size_t)n_tris * sizeof(int));
    int n_charts = t2pbr_find_charts(triangles, n_tris, n_verts, tri_chart);
    fprintf(stderr, "UV: %d charts from %d triangles\n", n_charts, n_tris);

    /* 2. For each chart, compute 2D projection via triangle-normal-based axes.
     * Per-vertex UV = project 3D position onto chart's 2D plane. */
    float *vert_uv = (float *)calloc((size_t)n_verts * 2, sizeof(float));
    int *vert_mapped = (int *)calloc((size_t)n_verts, sizeof(int));

    /* Chart bounding boxes in UV space [n_charts][4] = (u_min, v_min, u_max, v_max) */
    float *chart_bbox = (float *)malloc((size_t)n_charts * 4 * sizeof(float));
    for (int c = 0; c < n_charts; c++) {
        chart_bbox[c*4+0] = 1e30f; chart_bbox[c*4+1] = 1e30f;
        chart_bbox[c*4+2] = -1e30f; chart_bbox[c*4+3] = -1e30f;
    }

    /* Project each chart's vertices to 2D using average normal as projection plane */
    for (int ci = 0; ci < n_charts; ci++) {
        /* Compute average normal and centroid for this chart */
        float nx = 0, ny = 0, nz = 0, cx = 0, cy = 0, cz = 0;
        int count = 0;
        for (int ti = 0; ti < n_tris; ti++) {
            if (tri_chart[ti] != ci) continue;
            for (int k = 0; k < 3; k++) {
                int vi = triangles[ti * 3 + k];
                cx += vertices[vi*3]; cy += vertices[vi*3+1]; cz += vertices[vi*3+2];
                count++;
            }
            /* Triangle normal */
            int i0 = triangles[ti*3], i1 = triangles[ti*3+1], i2 = triangles[ti*3+2];
            float e1x = vertices[i1*3]-vertices[i0*3], e1y = vertices[i1*3+1]-vertices[i0*3+1], e1z = vertices[i1*3+2]-vertices[i0*3+2];
            float e2x = vertices[i2*3]-vertices[i0*3], e2y = vertices[i2*3+1]-vertices[i0*3+1], e2z = vertices[i2*3+2]-vertices[i0*3+2];
            nx += e1y*e2z - e1z*e2y; ny += e1z*e2x - e1x*e2z; nz += e1x*e2y - e1y*e2x;
        }
        if (count == 0) continue;
        cx /= count; cy /= count; cz /= count;
        float nl = sqrtf(nx*nx + ny*ny + nz*nz);
        if (nl < 1e-10f) { nx = 0; ny = 0; nz = 1; nl = 1; }
        nx /= nl; ny /= nl; nz /= nl;

        /* Build 2D axes: tangent and bitangent orthogonal to normal */
        float tx, ty, tz;
        if (fabsf(nx) < 0.9f) { tx = 0; ty = -nz; tz = ny; }
        else                   { tx = -nz; ty = 0; tz = nx; }
        float tl = sqrtf(tx*tx + ty*ty + tz*tz);
        tx /= tl; ty /= tl; tz /= tl;
        float bx = ny*tz - nz*ty, by = nz*tx - nx*tz, bz = nx*ty - ny*tx;

        /* Project vertices to 2D */
        for (int ti = 0; ti < n_tris; ti++) {
            if (tri_chart[ti] != ci) continue;
            for (int k = 0; k < 3; k++) {
                int vi = triangles[ti * 3 + k];
                if (vert_mapped[vi]) continue;
                float dx = vertices[vi*3] - cx, dy = vertices[vi*3+1] - cy, dz = vertices[vi*3+2] - cz;
                vert_uv[vi * 2 + 0] = dx*tx + dy*ty + dz*tz;
                vert_uv[vi * 2 + 1] = dx*bx + dy*by + dz*bz;
                vert_mapped[vi] = 1;
                /* Update chart bbox */
                if (vert_uv[vi*2] < chart_bbox[ci*4]) chart_bbox[ci*4] = vert_uv[vi*2];
                if (vert_uv[vi*2+1] < chart_bbox[ci*4+1]) chart_bbox[ci*4+1] = vert_uv[vi*2+1];
                if (vert_uv[vi*2] > chart_bbox[ci*4+2]) chart_bbox[ci*4+2] = vert_uv[vi*2];
                if (vert_uv[vi*2+1] > chart_bbox[ci*4+3]) chart_bbox[ci*4+3] = vert_uv[vi*2+1];
            }
        }
    }
    free(vert_mapped);

    /* 3. Pack charts into atlas using greedy shelf packing */
    float *chart_offset = (float *)calloc((size_t)n_charts * 2, sizeof(float));
    float *chart_scale = (float *)malloc((size_t)n_charts * sizeof(float));
    {
        float shelf_y = 0, shelf_h = 0, shelf_x = 0;
        float margin_uv = 2.0f / (float)tex_size;

        for (int ci = 0; ci < n_charts; ci++) {
            float w = chart_bbox[ci*4+2] - chart_bbox[ci*4+0];
            float h = chart_bbox[ci*4+3] - chart_bbox[ci*4+1];
            if (w < 1e-10f) w = 1e-5f;
            if (h < 1e-10f) h = 1e-5f;

            /* Scale to fit reasonable atlas fraction */
            float max_chart_frac = 0.3f;  /* max 30% of atlas per chart */
            float sc = max_chart_frac / fmaxf(w, h);
            /* Clamp to avoid tiny charts */
            float min_px = 4.0f / (float)tex_size;
            if (sc * w < min_px) sc = min_px / w;
            if (sc * h < min_px) sc = min_px / h;

            float cw = sc * w + margin_uv;
            float ch_h = sc * h + margin_uv;

            if (shelf_x + cw > 1.0f) {
                /* New shelf */
                shelf_y += shelf_h + margin_uv;
                shelf_h = 0;
                shelf_x = 0;
            }
            chart_offset[ci * 2 + 0] = shelf_x + margin_uv - sc * chart_bbox[ci*4+0];
            chart_offset[ci * 2 + 1] = shelf_y + margin_uv - sc * chart_bbox[ci*4+1];
            chart_scale[ci] = sc;
            shelf_x += cw;
            if (ch_h > shelf_h) shelf_h = ch_h;
        }
    }
    free(chart_bbox);

    /* 4. Transform vertex UVs to final atlas coordinates */
    for (int ti = 0; ti < n_tris; ti++) {
        int ci = tri_chart[ti];
        float sc = chart_scale[ci];
        float ox = chart_offset[ci * 2], oy = chart_offset[ci * 2 + 1];
        for (int k = 0; k < 3; k++) {
            int vi = triangles[ti * 3 + k];
            vert_uv[vi * 2 + 0] = vert_uv[vi * 2 + 0] * sc + ox;
            vert_uv[vi * 2 + 1] = vert_uv[vi * 2 + 1] * sc + oy;
        }
    }
    free(chart_offset); free(chart_scale); free(tri_chart);

    /* 5. Rasterize triangles into texture maps */
    uint8_t *tex_bc = (uint8_t *)calloc((size_t)tex_size * tex_size * 3, 1);
    uint8_t *tex_mr = (uint8_t *)calloc((size_t)tex_size * tex_size * 3, 1);

    for (int ti = 0; ti < n_tris; ti++) {
        int i0 = triangles[ti*3], i1 = triangles[ti*3+1], i2 = triangles[ti*3+2];
        float u0 = vert_uv[i0*2], v0 = vert_uv[i0*2+1];
        float u1 = vert_uv[i1*2], v1 = vert_uv[i1*2+1];
        float u2 = vert_uv[i2*2], v2 = vert_uv[i2*2+1];
        const t2_pbr_attr *c0 = &vert_colors[i0], *c1 = &vert_colors[i1], *c2 = &vert_colors[i2];

        /* Bounding box in pixel space */
        float umin = fminf(u0, fminf(u1, u2)), umax = fmaxf(u0, fmaxf(u1, u2));
        float vmin = fminf(v0, fminf(v1, v2)), vmax = fmaxf(v0, fmaxf(v1, v2));
        int px0 = (int)(umin * tex_size) - 1, px1 = (int)(umax * tex_size) + 2;
        int py0 = (int)(vmin * tex_size) - 1, py1 = (int)(vmax * tex_size) + 2;
        if (px0 < 0) px0 = 0; if (py0 < 0) py0 = 0;
        if (px1 >= tex_size) px1 = tex_size - 1;
        if (py1 >= tex_size) py1 = tex_size - 1;

        for (int py = py0; py <= py1; py++) {
            for (int px = px0; px <= px1; px++) {
                float fu = ((float)px + 0.5f) / (float)tex_size;
                float fv = ((float)py + 0.5f) / (float)tex_size;
                float bu, bv, bw;
                t2pbr_barycentric(fu, fv, u0, v0, u1, v1, u2, v2, &bu, &bv, &bw);
                if (bu < -0.01f || bv < -0.01f || bw < -0.01f) continue;
                if (bu < 0) bu = 0; if (bv < 0) bv = 0; if (bw < 0) bw = 0;
                float s = bu + bv + bw; bu /= s; bv /= s; bw /= s;

                float r = bu*c0->r + bv*c1->r + bw*c2->r;
                float g = bu*c0->g + bv*c1->g + bw*c2->g;
                float b = bu*c0->b + bv*c1->b + bw*c2->b;
                float met = bu*c0->metallic + bv*c1->metallic + bw*c2->metallic;
                float rgh = bu*c0->roughness + bv*c1->roughness + bw*c2->roughness;

                int iy = tex_size - 1 - py;
                int idx = (iy * tex_size + px) * 3;
                tex_bc[idx+0] = (uint8_t)(r < 0 ? 0 : r > 1 ? 255 : (int)(r*255));
                tex_bc[idx+1] = (uint8_t)(g < 0 ? 0 : g > 1 ? 255 : (int)(g*255));
                tex_bc[idx+2] = (uint8_t)(b < 0 ? 0 : b > 1 ? 255 : (int)(b*255));
                tex_mr[idx+0] = 0;
                tex_mr[idx+1] = (uint8_t)(rgh < 0 ? 0 : rgh > 1 ? 255 : (int)(rgh*255));
                tex_mr[idx+2] = (uint8_t)(met < 0 ? 0 : met > 1 ? 255 : (int)(met*255));
            }
        }
    }

    /* Write texture images */
    char path_buf[512];
    snprintf(path_buf, sizeof(path_buf), "%s_basecolor.png", base_path);
    t2pbr_write_image(path_buf, tex_bc, tex_size, tex_size);
    fprintf(stderr, "Wrote %s\n", path_buf);

    snprintf(path_buf, sizeof(path_buf), "%s_metallic_roughness.png", base_path);
    t2pbr_write_image(path_buf, tex_mr, tex_size, tex_size);
    fprintf(stderr, "Wrote %s\n", path_buf);

    free(tex_bc); free(tex_mr);

    /* Write MTL file */
    snprintf(path_buf, sizeof(path_buf), "%s.mtl", base_path);
    {
        FILE *f = fopen(path_buf, "w");
        if (!f) { free(vert_uv); return -1; }
        const char *bn = strrchr(base_path, '/');
        bn = bn ? bn + 1 : base_path;
        fprintf(f, "# TRELLIS.2 PBR material\n");
        fprintf(f, "newmtl trellis2_pbr\n");
        fprintf(f, "Kd 1.0 1.0 1.0\n");
        fprintf(f, "map_Kd %s_basecolor.png\n", bn);
        fprintf(f, "# PBR extensions\n");
        fprintf(f, "map_Pr %s_metallic_roughness.png\n", bn);  /* roughness map */
        fprintf(f, "map_Pm %s_metallic_roughness.png\n", bn);  /* metallic map */
        fclose(f);
        fprintf(stderr, "Wrote %s\n", path_buf);
    }

    /* Write OBJ file with per-vertex UVs */
    snprintf(path_buf, sizeof(path_buf), "%s.obj", base_path);
    {
        FILE *f = fopen(path_buf, "w");
        if (!f) { free(vert_uv); return -1; }
        const char *bn = strrchr(base_path, '/');
        bn = bn ? bn + 1 : base_path;
        fprintf(f, "# TRELLIS.2 textured mesh: %d verts, %d tris, %d charts\n",
                n_verts, n_tris, n_charts);
        fprintf(f, "mtllib %s.mtl\n", bn);
        fprintf(f, "usemtl trellis2_pbr\n\n");

        for (int i = 0; i < n_verts; i++)
            fprintf(f, "v %f %f %f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);

        for (int i = 0; i < n_verts; i++)
            fprintf(f, "vt %f %f\n", vert_uv[i*2], vert_uv[i*2+1]);

        for (int ti = 0; ti < n_tris; ti++)
            fprintf(f, "f %d/%d %d/%d %d/%d\n",
                    triangles[ti*3+0]+1, triangles[ti*3+0]+1,
                    triangles[ti*3+1]+1, triangles[ti*3+1]+1,
                    triangles[ti*3+2]+1, triangles[ti*3+2]+1);
        fclose(f);
        fprintf(stderr, "Wrote %s (%d verts, %d tris, %d charts)\n",
                path_buf, n_verts, n_tris, n_charts);
    }

    free(vert_uv);
    return 0;
}

#endif /* T2_PBR_IMPLEMENTATION */
#endif /* T2_PBR_H */
