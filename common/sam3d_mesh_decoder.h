/*
 * sam3d_mesh_decoder.h - SAM3D learned SLAT mesh decoder, CPU path.
 *
 * Runs the mesh decoder checkpoint trunk (shared sparse transformer) plus
 * the two SparseSubdivideBlock3d upsample heads, then extracts a mesh
 * from the predicted per-cube SDF, deformation, FlexiCubes weights,
 * vertex RGB channels, and upstream DMC ambiguity check table. Marching
 * cubes remains as a fallback if the FlexiCubes-style extractor cannot
 * produce faces.
 */
#ifndef SAM3D_MESH_DECODER_H
#define SAM3D_MESH_DECODER_H

#include <stdint.h>
#include "sam3d_gs_decoder.h"
#include "marching_cubes.h"
#include "flexicubes_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    qtensor gn_w, gn_b;
    qtensor conv1_w, conv1_b;
    qtensor gn2_w, gn2_b;
    qtensor conv2_w, conv2_b;
    qtensor skip_w, skip_b;
    int in_c;
    int out_c;
} sam3d_mesh_up_block;

typedef struct {
    sam3d_gs_decoder_model trunk;
    sam3d_mesh_up_block up[2];
    qtensor out_w, out_b;
    int out_channels;
    int mesh_res;
    st_context *st_ctx;
} sam3d_mesh_decoder_model;

sam3d_mesh_decoder_model *sam3d_mesh_decoder_load_safetensors(const char *path);
void sam3d_mesh_decoder_free(sam3d_mesh_decoder_model *m);

int sam3d_mesh_decoder_decode(const sam3d_mesh_decoder_model *m,
                              const int32_t *coords,
                              const float *feats,
                              int N, int C,
                              mc_mesh *out_mesh,
                              int n_threads);
int sam3d_mesh_decoder_decode_rgb(const sam3d_mesh_decoder_model *m,
                                  const int32_t *coords,
                                  const float *feats,
                                  int N, int C,
                                  mc_mesh *out_mesh,
                                  float **out_rgb,
                                  int n_threads);

#ifdef __cplusplus
}
#endif

#endif /* SAM3D_MESH_DECODER_H */

#ifdef SAM3D_MESH_DECODER_IMPLEMENTATION
#ifndef SAM3D_MESH_DECODER_IMPL_ONCE
#define SAM3D_MESH_DECODER_IMPL_ONCE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int s3dm_load_block(st_context *ctx, int i, sam3d_gs_block *b)
{
    char p[128];
#define FIND_(field, name) do { \
    snprintf(p, sizeof(p), "blocks.%d.%s", i, name); \
    if (qt_find(ctx, p, &b->field) != 0) return -1; \
} while (0)
    FIND_(attn_qkv_w, "attn.to_qkv.weight");
    FIND_(attn_qkv_b, "attn.to_qkv.bias");
    FIND_(attn_out_w, "attn.to_out.weight");
    FIND_(attn_out_b, "attn.to_out.bias");
    FIND_(mlp_fc1_w,  "mlp.mlp.0.weight");
    FIND_(mlp_fc1_b,  "mlp.mlp.0.bias");
    FIND_(mlp_fc2_w,  "mlp.mlp.2.weight");
    FIND_(mlp_fc2_b,  "mlp.mlp.2.bias");
#undef FIND_
    return 0;
}

static int s3dm_load_up(st_context *ctx, int i, sam3d_mesh_up_block *b)
{
    char p[160];
#define FIND_(field, name) do { \
    snprintf(p, sizeof(p), "upsample.%d.%s", i, name); \
    if (qt_find(ctx, p, &b->field) != 0) return -1; \
} while (0)
    FIND_(gn_w,    "act_layers.0.weight");
    FIND_(gn_b,    "act_layers.0.bias");
    FIND_(conv1_w, "out_layers.0.conv.weight");
    FIND_(conv1_b, "out_layers.0.conv.bias");
    FIND_(gn2_w,   "out_layers.1.weight");
    FIND_(gn2_b,   "out_layers.1.bias");
    FIND_(conv2_w, "out_layers.3.conv.weight");
    FIND_(conv2_b, "out_layers.3.conv.bias");
    FIND_(skip_w,  "skip_connection.conv.weight");
    FIND_(skip_b,  "skip_connection.conv.bias");
#undef FIND_
    b->in_c = b->gn_w.n_cols;
    b->out_c = b->conv2_b.n_cols;
    return b->in_c > 0 && b->out_c > 0 ? 0 : -1;
}

sam3d_mesh_decoder_model *sam3d_mesh_decoder_load_safetensors(const char *path)
{
    st_context *ctx = safetensors_open(path);
    if (!ctx) {
        fprintf(stderr, "sam3d_mesh_decoder: cannot open %s\n", path);
        return NULL;
    }
    sam3d_mesh_decoder_model *m = (sam3d_mesh_decoder_model *)calloc(1, sizeof(*m));
    if (!m) { safetensors_close(ctx); return NULL; }
    m->st_ctx = ctx;

    sam3d_gs_decoder_model *t = &m->trunk;
    t->st_ctx = ctx;
    t->dim = 768;
    t->n_heads = 12;
    t->head_dim = 64;
    t->in_channels = 8;
    t->resolution = 64;
    t->window_size = 8;
    t->mlp_ratio = 4.0f;
    t->ln_eps = 1e-5f;

    int rc = 0;
    rc |= qt_find(ctx, "input_layer.weight", &t->input_w);
    rc |= qt_find(ctx, "input_layer.bias",   &t->input_b);
    if (rc) { sam3d_mesh_decoder_free(m); return NULL; }

    int n_blocks = 0;
    for (int i = 0; i < 64; i++) {
        char p[128];
        snprintf(p, sizeof(p), "blocks.%d.attn.to_qkv.weight", i);
        if (safetensors_find(ctx, p) < 0) break;
        n_blocks = i + 1;
    }
    if (n_blocks <= 0) { sam3d_mesh_decoder_free(m); return NULL; }
    t->n_blocks = n_blocks;
    t->blocks = (sam3d_gs_block *)calloc((size_t)n_blocks, sizeof(sam3d_gs_block));
    if (!t->blocks) { sam3d_mesh_decoder_free(m); return NULL; }
    for (int i = 0; i < n_blocks; i++) {
        if (s3dm_load_block(ctx, i, &t->blocks[i]) != 0) {
            sam3d_mesh_decoder_free(m); return NULL;
        }
    }
    if (s3dm_load_up(ctx, 0, &m->up[0]) != 0 ||
        s3dm_load_up(ctx, 1, &m->up[1]) != 0 ||
        qt_find(ctx, "out_layer.weight", &m->out_w) != 0 ||
        qt_find(ctx, "out_layer.bias", &m->out_b) != 0) {
        sam3d_mesh_decoder_free(m); return NULL;
    }
    m->out_channels = m->out_b.n_cols;
    m->mesh_res = 256;
    fprintf(stderr,
            "sam3d_mesh_decoder: loaded %d blocks, up %d->%d->%d, out=%d res=%d\n",
            t->n_blocks, m->up[0].in_c, m->up[0].out_c, m->up[1].out_c,
            m->out_channels, m->mesh_res);
    return m;
}

void sam3d_mesh_decoder_free(sam3d_mesh_decoder_model *m)
{
    if (!m) return;
    free(m->trunk.blocks);
    if (m->st_ctx) safetensors_close(m->st_ctx);
    free(m);
}

static void s3dm_groupnorm32(float *dst, const float *src,
                             const qtensor *wq, const qtensor *bq,
                             int N, int C)
{
    const int G = 32;
    int group_c = C / G;
    const float *w = (const float *)wq->data;
    const float *b = (const float *)bq->data;
    for (int i = 0; i < N; i++) {
        const float *x = src + (size_t)i * C;
        float *y = dst + (size_t)i * C;
        for (int g = 0; g < G; g++) {
            int c0 = g * group_c;
            float mean = 0.0f;
            for (int c = 0; c < group_c; c++) mean += x[c0 + c];
            mean /= (float)group_c;
            float var = 0.0f;
            for (int c = 0; c < group_c; c++) {
                float d = x[c0 + c] - mean;
                var += d * d;
            }
            float inv = 1.0f / sqrtf(var / (float)group_c + 1e-5f);
            for (int c = 0; c < group_c; c++) {
                int k = c0 + c;
                y[k] = (x[k] - mean) * inv * w[k] + b[k];
            }
        }
    }
}

static sp3d_tensor *s3dm_subdivide(const sp3d_tensor *t)
{
    int N = t->N, C = t->C;
    int out_N = N * 8;
    int32_t *coords = (int32_t *)malloc((size_t)out_N * 4 * sizeof(int32_t));
    float *feats = (float *)malloc((size_t)out_N * C * sizeof(float));
    if (!coords || !feats) { free(coords); free(feats); return NULL; }
    int k = 0;
    for (int i = 0; i < N; i++) {
        int32_t b = t->coords[i * 4 + 0];
        int32_t z = t->coords[i * 4 + 1] * 2;
        int32_t y = t->coords[i * 4 + 2] * 2;
        int32_t x = t->coords[i * 4 + 3] * 2;
        for (int s = 0; s < 8; s++, k++) {
            coords[k * 4 + 0] = b;
            coords[k * 4 + 1] = z + ((s >> 2) & 1);
            coords[k * 4 + 2] = y + ((s >> 1) & 1);
            coords[k * 4 + 3] = x + (s & 1);
            memcpy(feats + (size_t)k * C, t->feats + (size_t)i * C,
                   (size_t)C * sizeof(float));
        }
    }
    sp3d_tensor *out = sp3d_create(coords, feats, out_N, C, t->batch_size);
    free(coords); free(feats);
    return out;
}

static sp3d_tensor *s3dm_up_block(const sam3d_mesh_up_block *b,
                                  const sp3d_tensor *x,
                                  int n_threads)
{
    int N = x->N;
    float *norm = (float *)malloc((size_t)N * b->in_c * sizeof(float));
    if (!norm) return NULL;
    s3dm_groupnorm32(norm, x->feats, &b->gn_w, &b->gn_b, N, b->in_c);
    sp3d_silu(norm, N * b->in_c);
    sp3d_tensor *xn = sp3d_replace_feats(x, norm, b->in_c);
    free(norm);
    if (!xn) return NULL;

    sp3d_tensor *hsub = s3dm_subdivide(xn);
    sp3d_free(xn);
    sp3d_tensor *xsub = s3dm_subdivide(x);
    if (!hsub || !xsub) { sp3d_free(hsub); sp3d_free(xsub); return NULL; }

    int M = hsub->N;
    float *h1 = (float *)malloc((size_t)M * b->out_c * sizeof(float));
    float *h2 = (float *)malloc((size_t)M * b->out_c * sizeof(float));
    float *skip = (float *)malloc((size_t)M * b->out_c * sizeof(float));
    if (!h1 || !h2 || !skip) {
        free(h1); free(h2); free(skip); sp3d_free(hsub); sp3d_free(xsub); return NULL;
    }
    sp3d_conv3d_forward(h1, hsub, (const float *)b->conv1_w.data,
                        (const float *)b->conv1_b.data,
                        b->in_c, b->out_c, 3, n_threads);
    s3dm_groupnorm32(h1, h1, &b->gn2_w, &b->gn2_b, M, b->out_c);
    sp3d_silu(h1, M * b->out_c);
    sp3d_tensor *ht = sp3d_replace_feats(hsub, h1, b->out_c);
    sp3d_conv3d_forward(h2, ht, (const float *)b->conv2_w.data,
                        (const float *)b->conv2_b.data,
                        b->out_c, b->out_c, 3, n_threads);
    sp3d_linear(skip, xsub->feats, M, &b->skip_w, &b->skip_b,
                b->out_c, b->in_c, n_threads);
    for (int i = 0; i < M * b->out_c; i++) h2[i] += skip[i];

    sp3d_tensor *out = sp3d_replace_feats(hsub, h2, b->out_c);
    free(h1); free(h2); free(skip);
    sp3d_free(ht); sp3d_free(hsub); sp3d_free(xsub);
    return out;
}

typedef struct {
    float *data;
    int count, cap;
} s3dm_fbuf;

typedef struct {
    int *data;
    int count, cap;
} s3dm_ibuf;

typedef struct {
    uint64_t key;
    int order;
    int vd;
    float gamma;
    float s0;
} s3dm_fc_edge_rec;

typedef struct {
    s3dm_fc_edge_rec *data;
    int count, cap;
} s3dm_ebuf;

static int s3dm_fbuf_push3(s3dm_fbuf *b, float x, float y, float z)
{
    if (b->count + 3 > b->cap) {
        int nc = b->cap ? b->cap * 2 : 4096;
        while (nc < b->count + 3) nc *= 2;
        float *p = (float *)realloc(b->data, (size_t)nc * sizeof(float));
        if (!p) return -1;
        b->data = p; b->cap = nc;
    }
    b->data[b->count++] = x;
    b->data[b->count++] = y;
    b->data[b->count++] = z;
    return 0;
}

static int s3dm_ibuf_push3(s3dm_ibuf *b, int a, int c, int d)
{
    if (a == c || a == d || c == d) return 0;
    if (b->count + 3 > b->cap) {
        int nc = b->cap ? b->cap * 2 : 4096;
        while (nc < b->count + 3) nc *= 2;
        int *p = (int *)realloc(b->data, (size_t)nc * sizeof(int));
        if (!p) return -1;
        b->data = p; b->cap = nc;
    }
    b->data[b->count++] = a;
    b->data[b->count++] = c;
    b->data[b->count++] = d;
    return 0;
}

static int s3dm_ebuf_push(s3dm_ebuf *b, uint64_t key, int order,
                          int vd, float gamma, float s0)
{
    if (b->count + 1 > b->cap) {
        int nc = b->cap ? b->cap * 2 : 4096;
        s3dm_fc_edge_rec *p =
            (s3dm_fc_edge_rec *)realloc(b->data, (size_t)nc * sizeof(*p));
        if (!p) return -1;
        b->data = p; b->cap = nc;
    }
    b->data[b->count].key = key;
    b->data[b->count].order = order;
    b->data[b->count].vd = vd;
    b->data[b->count].gamma = gamma;
    b->data[b->count].s0 = s0;
    b->count++;
    return 0;
}

static int s3dm_fc_edge_cmp(const void *pa, const void *pb)
{
    const s3dm_fc_edge_rec *a = (const s3dm_fc_edge_rec *)pa;
    const s3dm_fc_edge_rec *b = (const s3dm_fc_edge_rec *)pb;
    if (a->key < b->key) return -1;
    if (a->key > b->key) return 1;
    return (a->order > b->order) - (a->order < b->order);
}

static inline float s3dm_sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline size_t s3dm_grid_idx3(int z, int y, int x, int rv)
{
    return ((size_t)z * (size_t)rv + (size_t)y) * (size_t)rv + (size_t)x;
}

static uint64_t s3dm_fc_edge_key(int z, int y, int x, int edge)
{
    static const int cc[8][3] = {
        {0,0,0},{1,0,0},{0,1,0},{1,1,0},
        {0,0,1},{1,0,1},{0,1,1},{1,1,1}
    };
    static const int ec[12][2] = {
        {0,1},{1,5},{4,5},{0,4},{2,3},{3,7},
        {6,7},{2,6},{2,0},{3,1},{7,5},{6,4}
    };
    int a = ec[edge][0], b = ec[edge][1];
    int z0 = z + cc[a][0], y0 = y + cc[a][1], x0 = x + cc[a][2];
    int z1 = z + cc[b][0], y1 = y + cc[b][1], x1 = x + cc[b][2];
    int axis = (z0 != z1) ? 0 : ((y0 != y1) ? 1 : 2);
    if (z1 < z0) z0 = z1;
    if (y1 < y0) y0 = y1;
    if (x1 < x0) x0 = x1;
    return ((uint64_t)axis << 60) |
           ((uint64_t)(uint32_t)z0 << 40) |
           ((uint64_t)(uint32_t)y0 << 20) |
           (uint64_t)(uint32_t)x0;
}

static int s3dm_fc_case_id(const float *sdf, int z, int y, int x, int rv)
{
    static const int cc[8][3] = {
        {0,0,0},{1,0,0},{0,1,0},{1,1,0},
        {0,0,1},{1,0,1},{0,1,1},{1,1,1}
    };
    int case_id = 0;
    for (int c = 0; c < 8; c++) {
        int vz = z + cc[c][0], vy = y + cc[c][1], vx = x + cc[c][2];
        if (sdf[s3dm_grid_idx3(vz, vy, vx, rv)] < 0.0f) case_id |= (1 << c);
    }
    return case_id;
}

static int s3dm_fc_resolve_case_id(const float *sdf, int z, int y, int x,
                                   int res, int rv, int case_id)
{
    const signed short *cfg = fc_check_table[case_id];
    if (cfg[0] != 1) return case_id;
    int az = z + cfg[1], ay = y + cfg[2], ax = x + cfg[3];
    if (az < 0 || ay < 0 || ax < 0 || az >= res || ay >= res || ax >= res) {
        return case_id;
    }
    int adj_case = s3dm_fc_case_id(sdf, az, ay, ax, rv);
    return fc_check_table[adj_case][0] == 1 ? cfg[4] : case_id;
}

static int s3dm_flexicubes_from_dense(const sp3d_tensor *cubes,
                                      const float *sdf,
                                      const float *def,
                                      const float *rgb_grid,
                                      int res,
                                      mc_mesh *out_mesh,
                                      float **out_rgb)
{
    static const int cc[8][3] = {
        {0,0,0},{1,0,0},{0,1,0},{1,1,0},
        {0,0,1},{1,0,1},{0,1,1},{1,1,1}
    };
    static const int ec[12][2] = {
        {0,1},{1,5},{4,5},{0,4},{2,3},{3,7},
        {6,7},{2,6},{2,0},{3,1},{7,5},{6,4}
    };
    int rv = res + 1;
    size_t ncubes = (size_t)res * (size_t)res * (size_t)res;
    int *cube_row = (int *)malloc(ncubes * sizeof(int));
    if (!cube_row) return -1;
    for (size_t i = 0; i < ncubes; i++) cube_row[i] = -1;
    for (int i = 0; i < cubes->N; i++) {
        int z = cubes->coords[i * 4 + 1];
        int y = cubes->coords[i * 4 + 2];
        int x = cubes->coords[i * 4 + 3];
        if (z < 0 || y < 0 || x < 0 || z >= res || y >= res || x >= res) continue;
        cube_row[((size_t)z * (size_t)res + (size_t)y) * (size_t)res + (size_t)x] = i;
    }

    s3dm_fbuf verts = {0}, cols = {0};
    s3dm_ibuf faces = {0};
    s3dm_ebuf edges = {0};
    int order = 0;
    float def_scale = (1.0f - 1e-8f) / ((float)res * 2.0f);
    int rc = -1;

    for (int z = 0; z < res; z++) {
        for (int y = 0; y < res; y++) {
            for (int x = 0; x < res; x++) {
                float val[8], pos[8][3], col[8][3];
                int case_id = 0;
                for (int c = 0; c < 8; c++) {
                    int vz = z + cc[c][0], vy = y + cc[c][1], vx = x + cc[c][2];
                    size_t gi = s3dm_grid_idx3(vz, vy, vx, rv);
                    val[c] = sdf[gi];
                    if (val[c] < 0.0f) case_id |= (1 << c);
                    pos[c][0] = (float)vz / (float)res - 0.5f +
                                def_scale * tanhf(def[gi * 3 + 0]);
                    pos[c][1] = (float)vy / (float)res - 0.5f +
                                def_scale * tanhf(def[gi * 3 + 1]);
                    pos[c][2] = (float)vx / (float)res - 0.5f +
                                def_scale * tanhf(def[gi * 3 + 2]);
                    if (rgb_grid) {
                        col[c][0] = rgb_grid[gi * 3 + 0];
                        col[c][1] = rgb_grid[gi * 3 + 1];
                        col[c][2] = rgb_grid[gi * 3 + 2];
                    } else {
                        col[c][0] = col[c][1] = col[c][2] = 0.8f;
                    }
                }
                case_id = s3dm_fc_resolve_case_id(sdf, z, y, x, res, rv, case_id);
                if (case_id == 0 || case_id == 255 || fc_num_vd_table[case_id] <= 0) continue;
                int row = cube_row[((size_t)z * (size_t)res + (size_t)y) * (size_t)res + (size_t)x];
                const float *f = (row >= 0) ? cubes->feats + (size_t)row * cubes->C : NULL;
                float gamma = (f && cubes->C >= 53) ?
                    (s3dm_sigmoidf(f[52]) * 0.99f + 0.005f) : 0.5f;

                for (int g = 0; g < 4; g++) {
                    if (fc_dmc_table[case_id][g][0] < 0) continue;
                    float pacc[3] = {0,0,0}, cacc[3] = {0,0,0}, wsum = 0.0f;
                    int group_edges[7], nge = 0;
                    for (int j = 0; j < 7; j++) {
                        int edge = fc_dmc_table[case_id][g][j];
                        if (edge < 0) break;
                        group_edges[nge++] = edge;
                        int c0 = ec[edge][0], c1 = ec[edge][1];
                        float a0 = (f && cubes->C >= 53) ?
                            (tanhf(f[44 + c0]) * 0.99f + 1.0f) : 1.0f;
                        float a1 = (f && cubes->C >= 53) ?
                            (tanhf(f[44 + c1]) * 0.99f + 1.0f) : 1.0f;
                        float w0 = val[c1] * a1;
                        float w1 = -val[c0] * a0;
                        float den = w0 + w1;
                        float ep[3], ecv[3];
                        if (fabsf(den) < 1e-12f) {
                            for (int k = 0; k < 3; k++) {
                                ep[k] = 0.5f * (pos[c0][k] + pos[c1][k]);
                                ecv[k] = 0.5f * (col[c0][k] + col[c1][k]);
                            }
                        } else {
                            float inv = 1.0f / den;
                            for (int k = 0; k < 3; k++) {
                                ep[k] = (w0 * pos[c0][k] + w1 * pos[c1][k]) * inv;
                                ecv[k] = (w0 * col[c0][k] + w1 * col[c1][k]) * inv;
                            }
                        }
                        float beta = (f && cubes->C >= 53) ?
                            (tanhf(f[32 + edge]) * 0.99f + 1.0f) : 1.0f;
                        for (int k = 0; k < 3; k++) {
                            pacc[k] += beta * ep[k];
                            cacc[k] += beta * ecv[k];
                        }
                        wsum += beta;
                    }
                    if (nge <= 0 || wsum <= 1e-12f) continue;
                    int vd = verts.count / 3;
                    if (s3dm_fbuf_push3(&verts, pacc[0] / wsum, pacc[1] / wsum,
                                        pacc[2] / wsum) != 0) goto done;
                    if (out_rgb && s3dm_fbuf_push3(&cols, cacc[0] / wsum,
                                                   cacc[1] / wsum,
                                                   cacc[2] / wsum) != 0) goto done;
                    for (int j = 0; j < nge; j++) {
                        int edge = group_edges[j];
                        int c0 = ec[edge][0];
                        if (s3dm_ebuf_push(&edges, s3dm_fc_edge_key(z, y, x, edge),
                                           order++, vd, gamma, val[c0]) != 0) goto done;
                    }
                }
            }
        }
    }
    if (verts.count <= 0 || edges.count <= 0) goto done;
    qsort(edges.data, (size_t)edges.count, sizeof(edges.data[0]), s3dm_fc_edge_cmp);
    for (int i = 0; i < edges.count;) {
        int j = i + 1;
        while (j < edges.count && edges.data[j].key == edges.data[i].key) j++;
        if (j - i == 4) {
            int qraw[4] = {edges.data[i].vd, edges.data[i+1].vd,
                           edges.data[i+2].vd, edges.data[i+3].vd};
            int q[4];
            if (edges.data[i].s0 > 0.0f) {
                q[0] = qraw[0]; q[1] = qraw[1]; q[2] = qraw[3]; q[3] = qraw[2];
            } else {
                q[0] = qraw[2]; q[1] = qraw[3]; q[2] = qraw[1]; q[3] = qraw[0];
            }
            float g0 = edges.data[i].gamma, g1 = edges.data[i+1].gamma;
            float g2 = edges.data[i+2].gamma, g3 = edges.data[i+3].gamma;
            if (g0 * g2 > g1 * g3) {
                if (s3dm_ibuf_push3(&faces, q[0], q[1], q[2]) != 0 ||
                    s3dm_ibuf_push3(&faces, q[0], q[2], q[3]) != 0) goto done;
            } else {
                if (s3dm_ibuf_push3(&faces, q[0], q[1], q[3]) != 0 ||
                    s3dm_ibuf_push3(&faces, q[3], q[1], q[2]) != 0) goto done;
            }
        }
        i = j;
    }
    if (verts.count > 0 && faces.count > 0) {
        out_mesh->vertices = verts.data;
        out_mesh->n_verts = verts.count / 3;
        out_mesh->triangles = faces.data;
        out_mesh->n_tris = faces.count / 3;
        verts.data = NULL; faces.data = NULL;
        if (out_rgb) {
            *out_rgb = cols.data;
            cols.data = NULL;
        }
        rc = 0;
    }

done:
    free(cube_row);
    free(verts.data);
    free(faces.data);
    free(edges.data);
    free(cols.data);
    return rc;
}

static int s3dm_sdf_mesh_from_features(const sam3d_mesh_decoder_model *m,
                                       const sp3d_tensor *cubes,
                                       mc_mesh *out_mesh,
                                       float **out_rgb)
{
    int res = m->mesh_res;
    int rv = res + 1;
    size_t ngrid = (size_t)rv * rv * rv;
    float *sdf_sum = (float *)malloc(ngrid * sizeof(float));
    float *rgb_sum = out_rgb ? (float *)calloc(ngrid * 3, sizeof(float)) : NULL;
    float *def_sum = (float *)calloc(ngrid * 3, sizeof(float));
    int *sdf_cnt = (int *)calloc(ngrid, sizeof(int));
    if (!sdf_sum || !sdf_cnt || !def_sum || (out_rgb && !rgb_sum)) {
        free(sdf_sum); free(sdf_cnt); free(rgb_sum); free(def_sum); return -1;
    }
    for (size_t i = 0; i < ngrid; i++) sdf_sum[i] = 1.0f;
    static const int cc[8][3] = {
        {0,0,0},{1,0,0},{0,1,0},{1,1,0},
        {0,0,1},{1,0,1},{0,1,1},{1,1,1}
    };
    float bias = -1.0f / (float)res;
    for (int i = 0; i < cubes->N; i++) {
        int z = cubes->coords[i * 4 + 1];
        int y = cubes->coords[i * 4 + 2];
        int x = cubes->coords[i * 4 + 3];
        if (z < 0 || y < 0 || x < 0 || z >= res || y >= res || x >= res) continue;
        const float *f = cubes->feats + (size_t)i * cubes->C;
        for (int c = 0; c < 8; c++) {
            int vz = z + cc[c][0], vy = y + cc[c][1], vx = x + cc[c][2];
            size_t idx = ((size_t)vz * rv + vy) * rv + vx;
            if (sdf_cnt[idx] == 0) sdf_sum[idx] = 0.0f;
            sdf_sum[idx] += f[c] + bias;
            if (cubes->C >= 32) {
                const float *def = f + 8 + c * 3;
                def_sum[idx * 3 + 0] += def[0];
                def_sum[idx * 3 + 1] += def[1];
                def_sum[idx * 3 + 2] += def[2];
            }
            if (rgb_sum && cubes->C >= 101) {
                const float *clr = f + 53 + c * 6;
                for (int ch = 0; ch < 3; ch++) {
                    float v = clr[ch];
                    rgb_sum[idx * 3 + ch] += 1.0f / (1.0f + expf(-v));
                }
            }
            sdf_cnt[idx]++;
        }
    }
    for (size_t i = 0; i < ngrid; i++) {
        if (sdf_cnt[i] > 0) sdf_sum[i] /= (float)sdf_cnt[i];
        if (rgb_sum && sdf_cnt[i] > 0) {
            rgb_sum[i * 3 + 0] /= (float)sdf_cnt[i];
            rgb_sum[i * 3 + 1] /= (float)sdf_cnt[i];
            rgb_sum[i * 3 + 2] /= (float)sdf_cnt[i];
        }
        if (sdf_cnt[i] > 0) {
            def_sum[i * 3 + 0] /= (float)sdf_cnt[i];
            def_sum[i * 3 + 1] /= (float)sdf_cnt[i];
            def_sum[i * 3 + 2] /= (float)sdf_cnt[i];
        }
    }
    if (cubes->C >= 53 &&
        s3dm_flexicubes_from_dense(cubes, sdf_sum, def_sum, rgb_sum, res,
                                   out_mesh, out_rgb) == 0) {
        free(sdf_sum); free(sdf_cnt);
        free(rgb_sum); free(def_sum);
        return 0;
    }

    float bounds[6] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
    *out_mesh = mc_marching_cubes(sdf_sum, rv, rv, rv, 0.0f, bounds);
    if (out_mesh->n_verts > 0) {
        float def_scale = (1.0f - 1e-8f) / ((float)res * 2.0f);
        for (int i = 0; i < out_mesh->n_verts; i++) {
            float *p = out_mesh->vertices + (size_t)i * 3;
            float gx = (p[0] + 0.5f) * (float)res;
            float gy = (p[1] + 0.5f) * (float)res;
            float gz = (p[2] + 0.5f) * (float)res;
            int x0 = (int)floorf(gx), y0 = (int)floorf(gy), z0 = (int)floorf(gz);
            float tx = gx - (float)x0, ty = gy - (float)y0, tz = gz - (float)z0;
            if (x0 < 0) { x0 = 0; tx = 0.0f; } else if (x0 >= res) { x0 = res - 1; tx = 1.0f; }
            if (y0 < 0) { y0 = 0; ty = 0.0f; } else if (y0 >= res) { y0 = res - 1; ty = 1.0f; }
            if (z0 < 0) { z0 = 0; tz = 0.0f; } else if (z0 >= res) { z0 = res - 1; tz = 1.0f; }
            float d[3] = {0.0f, 0.0f, 0.0f};
            for (int dx = 0; dx <= 1; dx++) {
                float wx = dx ? tx : (1.0f - tx);
                for (int dy = 0; dy <= 1; dy++) {
                    float wy = dy ? ty : (1.0f - ty);
                    for (int dz = 0; dz <= 1; dz++) {
                        float wz = dz ? tz : (1.0f - tz);
                        float w = wx * wy * wz;
                        size_t idx = ((size_t)(x0 + dx) * rv + (y0 + dy)) * rv + (z0 + dz);
                        d[0] += w * def_sum[idx * 3 + 0];
                        d[1] += w * def_sum[idx * 3 + 1];
                        d[2] += w * def_sum[idx * 3 + 2];
                    }
                }
            }
            p[0] += def_scale * tanhf(d[0]);
            p[1] += def_scale * tanhf(d[1]);
            p[2] += def_scale * tanhf(d[2]);
        }
    }
    if (out_rgb && out_mesh->n_verts > 0 && rgb_sum) {
        float *rgb = (float *)malloc((size_t)out_mesh->n_verts * 3 * sizeof(float));
        if (!rgb) {
            free(sdf_sum); free(sdf_cnt); free(rgb_sum); free(def_sum);
            return -1;
        }
        for (int i = 0; i < out_mesh->n_verts; i++) {
            const float *p = out_mesh->vertices + (size_t)i * 3;
            int gx = (int)lrintf((p[0] + 0.5f) * (float)res);
            int gy = (int)lrintf((p[1] + 0.5f) * (float)res);
            int gz = (int)lrintf((p[2] + 0.5f) * (float)res);
            if (gx < 0) gx = 0; else if (gx > res) gx = res;
            if (gy < 0) gy = 0; else if (gy > res) gy = res;
            if (gz < 0) gz = 0; else if (gz > res) gz = res;
            size_t idx = ((size_t)gx * rv + gy) * rv + gz;
            rgb[i * 3 + 0] = rgb_sum[idx * 3 + 0];
            rgb[i * 3 + 1] = rgb_sum[idx * 3 + 1];
            rgb[i * 3 + 2] = rgb_sum[idx * 3 + 2];
        }
        *out_rgb = rgb;
    }
    free(sdf_sum); free(sdf_cnt);
    free(rgb_sum); free(def_sum);
    return (out_mesh->n_verts > 0 && out_mesh->n_tris > 0) ? 0 : -1;
}

int sam3d_mesh_decoder_decode(const sam3d_mesh_decoder_model *m,
                              const int32_t *coords,
                              const float *feats,
                              int N, int C,
                              mc_mesh *out_mesh,
                              int n_threads)
{
    return sam3d_mesh_decoder_decode_rgb(m, coords, feats, N, C, out_mesh,
                                         NULL, n_threads);
}

int sam3d_mesh_decoder_decode_rgb(const sam3d_mesh_decoder_model *m,
                                  const int32_t *coords,
                                  const float *feats,
                                  int N, int C,
                                  mc_mesh *out_mesh,
                                  float **out_rgb,
                                  int n_threads)
{
    if (!m || !coords || !feats || N <= 0 || C != 8 || !out_mesh) return -1;
    memset(out_mesh, 0, sizeof(*out_mesh));
    if (out_rgb) *out_rgb = NULL;
    sp3d_tensor *x = sp3d_create(coords, feats, N, C, 1);
    if (!x) return -1;
    float *h = NULL;
    int rc = sam3d_gs_decoder_hidden_transformer(&m->trunk, x, &h, n_threads);
    sp3d_free(x);
    if (rc != 0 || !h) return -2;
    sp3d_tensor *t = sp3d_create(coords, h, N, m->trunk.dim, 1);
    free(h);
    if (!t) return -2;
    sp3d_tensor *u0 = s3dm_up_block(&m->up[0], t, n_threads);
    sp3d_free(t);
    if (!u0) return -3;
    sp3d_tensor *u1 = s3dm_up_block(&m->up[1], u0, n_threads);
    sp3d_free(u0);
    if (!u1) return -4;
    float *out = (float *)malloc((size_t)u1->N * m->out_channels * sizeof(float));
    if (!out) { sp3d_free(u1); return -5; }
    sp3d_linear(out, u1->feats, u1->N, &m->out_w, &m->out_b,
                m->out_channels, u1->C, n_threads);
    sp3d_tensor *cube_feats = sp3d_replace_feats(u1, out, m->out_channels);
    free(out); sp3d_free(u1);
    if (!cube_feats) return -5;
    rc = s3dm_sdf_mesh_from_features(m, cube_feats, out_mesh, out_rgb);
    sp3d_free(cube_feats);
    return rc;
}

#endif /* SAM3D_MESH_DECODER_IMPL_ONCE */
#endif /* SAM3D_MESH_DECODER_IMPLEMENTATION */
