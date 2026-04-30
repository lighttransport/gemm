/*
 * tiny_gltf.h - tiny single-header binary glTF 2.0 mesh writer.
 *
 * This is intentionally small: it writes one triangle mesh as .glb with
 * POSITION float32 vertices and uint32 triangle indices. Optional paths
 * add COLOR_0 vertex colors or bake vertex colors into an embedded PNG
 * texture with generated TEXCOORD_0.
 */
#ifndef TINY_GLTF_H
#define TINY_GLTF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tinygltf_write_glb_mesh(const char *path,
                            const float *positions, int n_positions,
                            const int *triangles, int n_triangles);
int tinygltf_write_glb_mesh_rgb(const char *path,
                                const float *positions,
                                const float *colors_rgb,
                                int n_positions,
                                const int *triangles,
                                int n_triangles);
int tinygltf_write_glb_mesh_rgb_texture(const char *path,
                                        const float *positions,
                                        const float *colors_rgb,
                                        int n_positions,
                                        const int *triangles,
                                        int n_triangles,
                                        int texture_size);
int tinygltf_write_glb_mesh_rgb_texture_indexed(const char *path,
                                                const float *positions,
                                                const float *colors_rgb,
                                                const float *uvs,
                                                int n_positions,
                                                const uint32_t *indices,
                                                int n_indices,
                                                int texture_size);

#ifdef __cplusplus
}
#endif

#endif /* TINY_GLTF_H */

#ifdef TINY_GLTF_IMPLEMENTATION
#ifndef TINY_GLTF_IMPL_ONCE
#define TINY_GLTF_IMPL_ONCE

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stb_image_write.h"

static uint32_t tinygltf_pad4_u32(uint32_t n)
{
    return (n + 3u) & ~3u;
}

static int tinygltf_is_little_endian(void)
{
    const uint32_t x = 1;
    return *((const unsigned char *)&x) == 1;
}

static int tinygltf_write_u32(FILE *f, uint32_t v)
{
    return fwrite(&v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static float tinygltf_clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static void tinygltf_bary2(float px, float py,
                           float x0, float y0, float x1, float y1,
                           float x2, float y2,
                           float *b0, float *b1, float *b2)
{
    float d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (d > -1e-20f && d < 1e-20f) {
        *b0 = *b1 = *b2 = 1.0f / 3.0f;
        return;
    }
    *b0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d;
    *b1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d;
    *b2 = 1.0f - *b0 - *b1;
}

static int tinygltf_ceil_sqrt_int(int n)
{
    int r = 1;
    while (r > 0 && r * r < n) r++;
    return r;
}

int tinygltf_write_glb_mesh(const char *path,
                            const float *positions, int n_positions,
                            const int *triangles, int n_triangles)
{
    return tinygltf_write_glb_mesh_rgb(path, positions, NULL, n_positions,
                                       triangles, n_triangles);
}

int tinygltf_write_glb_mesh_rgb(const char *path,
                                const float *positions,
                                const float *colors_rgb,
                                int n_positions,
                                const int *triangles,
                                int n_triangles)
{
    if (!path || !positions || !triangles || n_positions <= 0 ||
        n_triangles <= 0 || !tinygltf_is_little_endian()) {
        return -1;
    }

    float mn[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float mx[3] = {-FLT_MAX,-FLT_MAX,-FLT_MAX };
    for (int i = 0; i < n_positions; i++) {
        const float *p = positions + (size_t)i * 3;
        for (int c = 0; c < 3; c++) {
            if (p[c] < mn[c]) mn[c] = p[c];
            if (p[c] > mx[c]) mx[c] = p[c];
        }
    }

    uint32_t max_index = 0;
    for (int i = 0; i < n_triangles * 3; i++) {
        if (triangles[i] < 0 || triangles[i] >= n_positions) return -1;
        if ((uint32_t)triangles[i] > max_index) max_index = (uint32_t)triangles[i];
    }

    uint32_t pos_bytes = (uint32_t)((size_t)n_positions * 3 * sizeof(float));
    uint32_t pos_padded = tinygltf_pad4_u32(pos_bytes);
    uint32_t col_offset = pos_padded;
    uint32_t col_bytes = colors_rgb ?
        (uint32_t)((size_t)n_positions * 3 * sizeof(float)) : 0u;
    uint32_t col_padded = colors_rgb ? tinygltf_pad4_u32(col_bytes) : 0u;
    uint32_t idx_offset = pos_padded + col_padded;
    uint32_t idx_bytes = (uint32_t)((size_t)n_triangles * 3 * sizeof(uint32_t));
    uint32_t bin_len = tinygltf_pad4_u32(idx_offset + idx_bytes);

    char json[4096];
    int json_len;
    if (colors_rgb) {
        json_len = snprintf(json, sizeof(json),
            "{\"asset\":{\"version\":\"2.0\",\"generator\":\"tiny_gltf.h\"},"
            "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
            "\"nodes\":[{\"mesh\":0}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"COLOR_0\":1},"
            "\"indices\":2,\"material\":0,\"mode\":4}]}],"
            "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[1.0,1.0,1.0,1.0],"
            "\"metallicFactor\":0.0,\"roughnessFactor\":0.8},\"doubleSided\":true}],"
            "\"buffers\":[{\"byteLength\":%u}],"
            "\"bufferViews\":["
            "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
            "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
            "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963}],"
            "\"accessors\":["
            "{\"bufferView\":0,\"byteOffset\":0,\"componentType\":5126,\"count\":%d,"
            "\"type\":\"VEC3\",\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
            "{\"bufferView\":1,\"byteOffset\":0,\"componentType\":5126,\"count\":%d,"
            "\"type\":\"VEC3\"},"
            "{\"bufferView\":2,\"byteOffset\":0,\"componentType\":5125,\"count\":%d,"
            "\"type\":\"SCALAR\",\"min\":[0],\"max\":[%u]}]}",
            bin_len, pos_bytes, col_offset, col_bytes, idx_offset, idx_bytes,
            n_positions, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
            n_positions, n_triangles * 3, max_index);
    } else {
        json_len = snprintf(json, sizeof(json),
            "{\"asset\":{\"version\":\"2.0\",\"generator\":\"tiny_gltf.h\"},"
            "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
            "\"nodes\":[{\"mesh\":0}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},"
            "\"indices\":1,\"material\":0,\"mode\":4}]}],"
            "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorFactor\":[0.8,0.8,0.8,1.0],"
            "\"metallicFactor\":0.0,\"roughnessFactor\":0.8},\"doubleSided\":true}],"
            "\"buffers\":[{\"byteLength\":%u}],"
            "\"bufferViews\":["
            "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
            "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963}],"
            "\"accessors\":["
            "{\"bufferView\":0,\"byteOffset\":0,\"componentType\":5126,\"count\":%d,"
            "\"type\":\"VEC3\",\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
            "{\"bufferView\":1,\"byteOffset\":0,\"componentType\":5125,\"count\":%d,"
            "\"type\":\"SCALAR\",\"min\":[0],\"max\":[%u]}]}",
            bin_len, pos_bytes, idx_offset, idx_bytes, n_positions,
            mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
            n_triangles * 3, max_index);
    }
    if (json_len <= 0 || json_len >= (int)sizeof(json)) return -1;

    uint32_t json_padded = tinygltf_pad4_u32((uint32_t)json_len);
    uint32_t total_len = 12u + 8u + json_padded + 8u + bin_len;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int rc = 0;
    rc |= tinygltf_write_u32(f, 0x46546C67u); /* glTF */
    rc |= tinygltf_write_u32(f, 2u);
    rc |= tinygltf_write_u32(f, total_len);
    rc |= tinygltf_write_u32(f, json_padded);
    rc |= tinygltf_write_u32(f, 0x4E4F534Au); /* JSON */
    if (fwrite(json, 1, (size_t)json_len, f) != (size_t)json_len) rc = -1;
    for (uint32_t i = (uint32_t)json_len; i < json_padded; i++)
        if (fputc(' ', f) == EOF) rc = -1;
    rc |= tinygltf_write_u32(f, bin_len);
    rc |= tinygltf_write_u32(f, 0x004E4942u); /* BIN */
    if (fwrite(positions, 1, pos_bytes, f) != pos_bytes) rc = -1;
    for (uint32_t i = pos_bytes; i < pos_padded; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (colors_rgb) {
        if (fwrite(colors_rgb, 1, col_bytes, f) != col_bytes) rc = -1;
        for (uint32_t i = col_bytes; i < col_padded; i++)
            if (fputc(0, f) == EOF) rc = -1;
    }
    for (int i = 0; i < n_triangles * 3; i++) {
        uint32_t idx = (uint32_t)triangles[i];
        rc |= tinygltf_write_u32(f, idx);
    }
    for (uint32_t i = idx_offset + idx_bytes; i < bin_len; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (fclose(f) != 0) rc = -1;
    return rc == 0 ? 0 : -1;
}

int tinygltf_write_glb_mesh_rgb_texture(const char *path,
                                        const float *positions,
                                        const float *colors_rgb,
                                        int n_positions,
                                        const int *triangles,
                                        int n_triangles,
                                        int texture_size)
{
    if (!path || !positions || !colors_rgb || !triangles ||
        n_positions <= 0 || n_triangles <= 0 || texture_size < 16 ||
        !tinygltf_is_little_endian()) {
        return -1;
    }
    for (int i = 0; i < n_triangles * 3; i++) {
        if (triangles[i] < 0 || triangles[i] >= n_positions) return -1;
    }

    int grid = tinygltf_ceil_sqrt_int(n_triangles);
    if (grid <= 0 || grid > 16384) return -1;
    int min_texture_size = grid * 4;
    if (texture_size < min_texture_size) texture_size = min_texture_size;
    int tile = texture_size / grid;
    if (tile < 4) return -1;

    int out_n = n_triangles * 3;
    float *out_pos = (float *)malloc((size_t)out_n * 3 * sizeof(float));
    float *out_uv = (float *)malloc((size_t)out_n * 2 * sizeof(float));
    uint32_t *out_idx = (uint32_t *)malloc((size_t)out_n * sizeof(uint32_t));
    unsigned char *tex = (unsigned char *)malloc((size_t)texture_size *
                                                 texture_size * 3);
    if (!out_pos || !out_uv || !out_idx || !tex) {
        free(out_pos); free(out_uv); free(out_idx); free(tex);
        return -1;
    }
    memset(tex, 255, (size_t)texture_size * texture_size * 3);

    float mn[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float mx[3] = {-FLT_MAX,-FLT_MAX,-FLT_MAX };
    for (int ti = 0; ti < n_triangles; ti++) {
        int cell_x = ti % grid;
        int cell_y = ti / grid;
        int x0 = cell_x * tile + 1;
        int y0 = cell_y * tile + 1;
        int x1 = cell_x * tile + tile - 2;
        int y1 = cell_y * tile + 1;
        int x2 = cell_x * tile + 1;
        int y2 = cell_y * tile + tile - 2;
        if (x1 >= texture_size) x1 = texture_size - 1;
        if (y2 >= texture_size) y2 = texture_size - 1;
        float uvp[3][2] = {
            {((float)x0 + 0.5f) / (float)texture_size,
             ((float)y0 + 0.5f) / (float)texture_size},
            {((float)x1 + 0.5f) / (float)texture_size,
             ((float)y1 + 0.5f) / (float)texture_size},
            {((float)x2 + 0.5f) / (float)texture_size,
             ((float)y2 + 0.5f) / (float)texture_size}
        };
        float avg[3] = {0.0f, 0.0f, 0.0f};
        for (int k = 0; k < 3; k++) {
            int vi = triangles[ti * 3 + k];
            const float *p = positions + (size_t)vi * 3;
            const float *c = colors_rgb + (size_t)vi * 3;
            int oi = ti * 3 + k;
            out_pos[oi * 3 + 0] = p[0];
            out_pos[oi * 3 + 1] = p[1];
            out_pos[oi * 3 + 2] = p[2];
            out_uv[oi * 2 + 0] = uvp[k][0];
            out_uv[oi * 2 + 1] = uvp[k][1];
            out_idx[oi] = (uint32_t)oi;
            for (int ch = 0; ch < 3; ch++) {
                if (p[ch] < mn[ch]) mn[ch] = p[ch];
                if (p[ch] > mx[ch]) mx[ch] = p[ch];
                avg[ch] += tinygltf_clampf(c[ch], 0.0f, 1.0f);
            }
        }
        for (int ch = 0; ch < 3; ch++) avg[ch] /= 3.0f;
        int px0 = cell_x * tile, py0 = cell_y * tile;
        int px1 = px0 + tile - 1, py1 = py0 + tile - 1;
        if (px1 >= texture_size) px1 = texture_size - 1;
        if (py1 >= texture_size) py1 = texture_size - 1;
        for (int py = py0; py <= py1; py++) {
            for (int px = px0; px <= px1; px++) {
                int off = (py * texture_size + px) * 3;
                tex[off + 0] = (unsigned char)(avg[0] * 255.0f + 0.5f);
                tex[off + 1] = (unsigned char)(avg[1] * 255.0f + 0.5f);
                tex[off + 2] = (unsigned char)(avg[2] * 255.0f + 0.5f);
            }
        }
        for (int py = y0 - 1; py <= y2 + 1 && py < texture_size; py++) {
            if (py < 0) continue;
            for (int px = x0 - 1; px <= x1 + 1 && px < texture_size; px++) {
                if (px < 0) continue;
                float b0, b1, b2;
                tinygltf_bary2((float)px + 0.5f, (float)py + 0.5f,
                               (float)x0, (float)y0, (float)x1, (float)y1,
                               (float)x2, (float)y2, &b0, &b1, &b2);
                if (b0 < -0.02f || b1 < -0.02f || b2 < -0.02f) continue;
                b0 = tinygltf_clampf(b0, 0.0f, 1.0f);
                b1 = tinygltf_clampf(b1, 0.0f, 1.0f);
                b2 = tinygltf_clampf(b2, 0.0f, 1.0f);
                float s = b0 + b1 + b2;
                if (s <= 1e-20f) continue;
                b0 /= s; b1 /= s; b2 /= s;
                int v0 = triangles[ti * 3 + 0];
                int v1 = triangles[ti * 3 + 1];
                int v2 = triangles[ti * 3 + 2];
                const float *c0 = colors_rgb + (size_t)v0 * 3;
                const float *c1 = colors_rgb + (size_t)v1 * 3;
                const float *c2 = colors_rgb + (size_t)v2 * 3;
                int off = (py * texture_size + px) * 3;
                for (int ch = 0; ch < 3; ch++) {
                    float c = b0 * c0[ch] + b1 * c1[ch] + b2 * c2[ch];
                    c = tinygltf_clampf(c, 0.0f, 1.0f);
                    tex[off + ch] = (unsigned char)(c * 255.0f + 0.5f);
                }
            }
        }
    }

    int png_len = 0;
    unsigned char *png = stbi_write_png_to_mem(tex, texture_size * 3,
                                               texture_size, texture_size,
                                               3, &png_len);
    free(tex);
    if (!png || png_len <= 0) {
        free(out_pos); free(out_uv); free(out_idx); free(png);
        return -1;
    }

    uint32_t pos_bytes = (uint32_t)((size_t)out_n * 3 * sizeof(float));
    uint32_t pos_padded = tinygltf_pad4_u32(pos_bytes);
    uint32_t uv_offset = pos_padded;
    uint32_t uv_bytes = (uint32_t)((size_t)out_n * 2 * sizeof(float));
    uint32_t uv_padded = tinygltf_pad4_u32(uv_bytes);
    uint32_t idx_offset = uv_offset + uv_padded;
    uint32_t idx_bytes = (uint32_t)((size_t)out_n * sizeof(uint32_t));
    uint32_t idx_padded = tinygltf_pad4_u32(idx_bytes);
    uint32_t img_offset = idx_offset + idx_padded;
    uint32_t img_bytes = (uint32_t)png_len;
    uint32_t bin_len = tinygltf_pad4_u32(img_offset + img_bytes);

    char json[4096];
    int json_len = snprintf(json, sizeof(json),
        "{\"asset\":{\"version\":\"2.0\",\"generator\":\"tiny_gltf.h\"},"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
        "\"nodes\":[{\"mesh\":0}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"TEXCOORD_0\":1},"
        "\"indices\":2,\"material\":0,\"mode\":4}]}],"
        "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},"
        "\"metallicFactor\":0.0,\"roughnessFactor\":0.8},\"doubleSided\":true}],"
        "\"textures\":[{\"sampler\":0,\"source\":0}],"
        "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9729,\"wrapS\":33071,\"wrapT\":33071}],"
        "\"images\":[{\"bufferView\":3,\"mimeType\":\"image/png\"}],"
        "\"buffers\":[{\"byteLength\":%u}],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}],"
        "\"accessors\":["
        "{\"bufferView\":0,\"byteOffset\":0,\"componentType\":5126,\"count\":%d,"
        "\"type\":\"VEC3\",\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
        "{\"bufferView\":1,\"byteOffset\":0,\"componentType\":5126,\"count\":%d,"
        "\"type\":\"VEC2\",\"min\":[0,0],\"max\":[1,1]},"
        "{\"bufferView\":2,\"byteOffset\":0,\"componentType\":5125,\"count\":%d,"
        "\"type\":\"SCALAR\",\"min\":[0],\"max\":[%u]}]}",
        bin_len, pos_bytes, uv_offset, uv_bytes, idx_offset, idx_bytes,
        img_offset, img_bytes, out_n, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
        out_n, out_n, (uint32_t)(out_n - 1));
    if (json_len <= 0 || json_len >= (int)sizeof(json)) {
        free(out_pos); free(out_uv); free(out_idx); free(png);
        return -1;
    }

    uint32_t json_padded = tinygltf_pad4_u32((uint32_t)json_len);
    uint32_t total_len = 12u + 8u + json_padded + 8u + bin_len;
    FILE *f = fopen(path, "wb");
    if (!f) {
        free(out_pos); free(out_uv); free(out_idx); free(png);
        return -1;
    }
    int rc = 0;
    rc |= tinygltf_write_u32(f, 0x46546C67u);
    rc |= tinygltf_write_u32(f, 2u);
    rc |= tinygltf_write_u32(f, total_len);
    rc |= tinygltf_write_u32(f, json_padded);
    rc |= tinygltf_write_u32(f, 0x4E4F534Au);
    if (fwrite(json, 1, (size_t)json_len, f) != (size_t)json_len) rc = -1;
    for (uint32_t i = (uint32_t)json_len; i < json_padded; i++)
        if (fputc(' ', f) == EOF) rc = -1;
    rc |= tinygltf_write_u32(f, bin_len);
    rc |= tinygltf_write_u32(f, 0x004E4942u);
    if (fwrite(out_pos, 1, pos_bytes, f) != pos_bytes) rc = -1;
    for (uint32_t i = pos_bytes; i < pos_padded; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (fwrite(out_uv, 1, uv_bytes, f) != uv_bytes) rc = -1;
    for (uint32_t i = uv_bytes; i < uv_padded; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (fwrite(out_idx, 1, idx_bytes, f) != idx_bytes) rc = -1;
    for (uint32_t i = idx_bytes; i < idx_padded; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (fwrite(png, 1, img_bytes, f) != img_bytes) rc = -1;
    for (uint32_t i = img_offset + img_bytes; i < bin_len; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (fclose(f) != 0) rc = -1;
    free(out_pos); free(out_uv); free(out_idx); free(png);
    return rc == 0 ? 0 : -1;
}

int tinygltf_write_glb_mesh_rgb_texture_indexed(const char *path,
                                                const float *positions,
                                                const float *colors_rgb,
                                                const float *uvs,
                                                int n_positions,
                                                const uint32_t *indices,
                                                int n_indices,
                                                int texture_size)
{
    if (!path || !positions || !colors_rgb || !uvs || !indices ||
        n_positions <= 0 || n_indices <= 0 || (n_indices % 3) != 0 ||
        texture_size < 16 || !tinygltf_is_little_endian()) {
        return -1;
    }

    uint32_t max_index = 0;
    for (int i = 0; i < n_indices; i++) {
        if (indices[i] >= (uint32_t)n_positions) return -1;
        if (indices[i] > max_index) max_index = indices[i];
    }

    unsigned char *tex = (unsigned char *)malloc((size_t)texture_size *
                                                 texture_size * 3);
    if (!tex) return -1;
    memset(tex, 255, (size_t)texture_size * texture_size * 3);

    float mn[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float mx[3] = {-FLT_MAX,-FLT_MAX,-FLT_MAX };
    for (int i = 0; i < n_positions; i++) {
        const float *p = positions + (size_t)i * 3;
        for (int ch = 0; ch < 3; ch++) {
            if (p[ch] < mn[ch]) mn[ch] = p[ch];
            if (p[ch] > mx[ch]) mx[ch] = p[ch];
        }
    }

    int n_tris = n_indices / 3;
    for (int ti = 0; ti < n_tris; ti++) {
        uint32_t i0 = indices[ti * 3 + 0];
        uint32_t i1 = indices[ti * 3 + 1];
        uint32_t i2 = indices[ti * 3 + 2];
        float u0 = tinygltf_clampf(uvs[(size_t)i0 * 2 + 0], 0.0f, 1.0f);
        float v0 = tinygltf_clampf(uvs[(size_t)i0 * 2 + 1], 0.0f, 1.0f);
        float u1 = tinygltf_clampf(uvs[(size_t)i1 * 2 + 0], 0.0f, 1.0f);
        float v1 = tinygltf_clampf(uvs[(size_t)i1 * 2 + 1], 0.0f, 1.0f);
        float u2 = tinygltf_clampf(uvs[(size_t)i2 * 2 + 0], 0.0f, 1.0f);
        float v2 = tinygltf_clampf(uvs[(size_t)i2 * 2 + 1], 0.0f, 1.0f);
        float x0 = u0 * (float)(texture_size - 1);
        float y0 = v0 * (float)(texture_size - 1);
        float x1 = u1 * (float)(texture_size - 1);
        float y1 = v1 * (float)(texture_size - 1);
        float x2 = u2 * (float)(texture_size - 1);
        float y2 = v2 * (float)(texture_size - 1);
        float minx = x0 < x1 ? (x0 < x2 ? x0 : x2) : (x1 < x2 ? x1 : x2);
        float maxx = x0 > x1 ? (x0 > x2 ? x0 : x2) : (x1 > x2 ? x1 : x2);
        float miny = y0 < y1 ? (y0 < y2 ? y0 : y2) : (y1 < y2 ? y1 : y2);
        float maxy = y0 > y1 ? (y0 > y2 ? y0 : y2) : (y1 > y2 ? y1 : y2);
        int px0 = (int)minx - 1, px1 = (int)maxx + 2;
        int py0 = (int)miny - 1, py1 = (int)maxy + 2;
        if (px0 < 0) px0 = 0;
        if (py0 < 0) py0 = 0;
        if (px1 >= texture_size) px1 = texture_size - 1;
        if (py1 >= texture_size) py1 = texture_size - 1;

        const float *c0 = colors_rgb + (size_t)i0 * 3;
        const float *c1 = colors_rgb + (size_t)i1 * 3;
        const float *c2 = colors_rgb + (size_t)i2 * 3;
        for (int py = py0; py <= py1; py++) {
            for (int px = px0; px <= px1; px++) {
                float b0, b1, b2;
                tinygltf_bary2((float)px + 0.5f, (float)py + 0.5f,
                               x0, y0, x1, y1, x2, y2, &b0, &b1, &b2);
                if (b0 < -0.02f || b1 < -0.02f || b2 < -0.02f) continue;
                b0 = tinygltf_clampf(b0, 0.0f, 1.0f);
                b1 = tinygltf_clampf(b1, 0.0f, 1.0f);
                b2 = tinygltf_clampf(b2, 0.0f, 1.0f);
                float s = b0 + b1 + b2;
                if (s <= 1e-20f) continue;
                b0 /= s; b1 /= s; b2 /= s;
                int off = (py * texture_size + px) * 3;
                for (int ch = 0; ch < 3; ch++) {
                    float c = b0 * c0[ch] + b1 * c1[ch] + b2 * c2[ch];
                    c = tinygltf_clampf(c, 0.0f, 1.0f);
                    tex[off + ch] = (unsigned char)(c * 255.0f + 0.5f);
                }
            }
        }
    }

    int png_len = 0;
    unsigned char *png = stbi_write_png_to_mem(tex, texture_size * 3,
                                               texture_size, texture_size,
                                               3, &png_len);
    free(tex);
    if (!png || png_len <= 0) {
        free(png);
        return -1;
    }

    uint32_t pos_bytes = (uint32_t)((size_t)n_positions * 3 * sizeof(float));
    uint32_t pos_padded = tinygltf_pad4_u32(pos_bytes);
    uint32_t uv_offset = pos_padded;
    uint32_t uv_bytes = (uint32_t)((size_t)n_positions * 2 * sizeof(float));
    uint32_t uv_padded = tinygltf_pad4_u32(uv_bytes);
    uint32_t idx_offset = uv_offset + uv_padded;
    uint32_t idx_bytes = (uint32_t)((size_t)n_indices * sizeof(uint32_t));
    uint32_t idx_padded = tinygltf_pad4_u32(idx_bytes);
    uint32_t img_offset = idx_offset + idx_padded;
    uint32_t img_bytes = (uint32_t)png_len;
    uint32_t bin_len = tinygltf_pad4_u32(img_offset + img_bytes);

    char json[4096];
    int json_len = snprintf(json, sizeof(json),
        "{\"asset\":{\"version\":\"2.0\",\"generator\":\"tiny_gltf.h\"},"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
        "\"nodes\":[{\"mesh\":0}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"TEXCOORD_0\":1},"
        "\"indices\":2,\"material\":0,\"mode\":4}]}],"
        "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":0},"
        "\"metallicFactor\":0.0,\"roughnessFactor\":0.8},\"doubleSided\":true}],"
        "\"textures\":[{\"sampler\":0,\"source\":0}],"
        "\"samplers\":[{\"magFilter\":9729,\"minFilter\":9729,\"wrapS\":33071,\"wrapT\":33071}],"
        "\"images\":[{\"bufferView\":3,\"mimeType\":\"image/png\"}],"
        "\"buffers\":[{\"byteLength\":%u}],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34962},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u,\"target\":34963},"
        "{\"buffer\":0,\"byteOffset\":%u,\"byteLength\":%u}],"
        "\"accessors\":["
        "{\"bufferView\":0,\"byteOffset\":0,\"componentType\":5126,\"count\":%d,"
        "\"type\":\"VEC3\",\"min\":[%.9g,%.9g,%.9g],\"max\":[%.9g,%.9g,%.9g]},"
        "{\"bufferView\":1,\"byteOffset\":0,\"componentType\":5126,\"count\":%d,"
        "\"type\":\"VEC2\",\"min\":[0,0],\"max\":[1,1]},"
        "{\"bufferView\":2,\"byteOffset\":0,\"componentType\":5125,\"count\":%d,"
        "\"type\":\"SCALAR\",\"min\":[0],\"max\":[%u]}]}",
        bin_len, pos_bytes, uv_offset, uv_bytes, idx_offset, idx_bytes,
        img_offset, img_bytes, n_positions, mn[0], mn[1], mn[2],
        mx[0], mx[1], mx[2], n_positions, n_indices, max_index);
    if (json_len <= 0 || json_len >= (int)sizeof(json)) {
        free(png);
        return -1;
    }

    uint32_t json_padded = tinygltf_pad4_u32((uint32_t)json_len);
    uint32_t total_len = 12u + 8u + json_padded + 8u + bin_len;
    FILE *f = fopen(path, "wb");
    if (!f) {
        free(png);
        return -1;
    }
    int rc = 0;
    rc |= tinygltf_write_u32(f, 0x46546C67u);
    rc |= tinygltf_write_u32(f, 2u);
    rc |= tinygltf_write_u32(f, total_len);
    rc |= tinygltf_write_u32(f, json_padded);
    rc |= tinygltf_write_u32(f, 0x4E4F534Au);
    if (fwrite(json, 1, (size_t)json_len, f) != (size_t)json_len) rc = -1;
    for (uint32_t i = (uint32_t)json_len; i < json_padded; i++)
        if (fputc(' ', f) == EOF) rc = -1;
    rc |= tinygltf_write_u32(f, bin_len);
    rc |= tinygltf_write_u32(f, 0x004E4942u);
    if (fwrite(positions, 1, pos_bytes, f) != pos_bytes) rc = -1;
    for (uint32_t i = pos_bytes; i < pos_padded; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (fwrite(uvs, 1, uv_bytes, f) != uv_bytes) rc = -1;
    for (uint32_t i = uv_bytes; i < uv_padded; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (fwrite(indices, 1, idx_bytes, f) != idx_bytes) rc = -1;
    for (uint32_t i = idx_bytes; i < idx_padded; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (fwrite(png, 1, img_bytes, f) != img_bytes) rc = -1;
    for (uint32_t i = img_offset + img_bytes; i < bin_len; i++)
        if (fputc(0, f) == EOF) rc = -1;
    if (fclose(f) != 0) rc = -1;
    free(png);
    return rc == 0 ? 0 : -1;
}

#endif /* TINY_GLTF_IMPL_ONCE */
#endif /* TINY_GLTF_IMPLEMENTATION */
