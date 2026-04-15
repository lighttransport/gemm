/*
 * mesh_vertex_inpaint.h - vertex-space texture inpainting for
 * Hunyuan3D-2.1 paint. Port of the upstream pybind11 extension at
 * hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp. Same
 * algorithm, plain-C++ API (no pybind dependencies).
 *
 * Header-only. `#define MESH_VERTEX_INPAINT_IMPLEMENTATION` in exactly
 * one translation unit before including.
 *
 * The upstream pipeline uses this to fill UV-texture pixels that no
 * multiview sample projected onto. It's per-vertex rather than
 * per-texel: each mesh vertex gets a colour (from whatever UV pixel it
 * maps to, or from a graph-diffusion average of its colored neighbours),
 * then every face-vertex writes its colour back to the texture.
 *
 * Inputs (all pointers, column-major-free C layout):
 *   texture   [H*W*C] float    input albedo / MR texture (modified in-place
 *                                into `out_texture`)
 *   mask      [H*W]   uint8    which UV pixels already have a valid sample
 *                                (non-zero = known)
 *   vtx_pos   [V*3]   float    vertex positions (world or normalised,
 *                                used only for the 1/d^2 weighting)
 *   vtx_uv    [U*2]   float    UV coords in [0, 1]
 *   pos_idx   [F*3]   int32    triangle face indices into vtx_pos
 *   uv_idx    [F*3]   int32    triangle face indices into vtx_uv
 *   V, U, F, H, W, C
 *
 * Outputs:
 *   out_texture [H*W*C] float  inpainted texture (caller allocates)
 *   out_mask    [H*W]   uint8  updated mask (caller allocates)
 *
 * Methods:
 *   MVI_SMOOTH  - iterative neighbour averaging, matches upstream
 *                 meshVerticeInpaint(method="smooth"). This is what
 *                 hy3dpaint actually uses.
 *   MVI_FORWARD - single front-propagation sweep, matches upstream
 *                 meshVerticeInpaint(method="forward"). Provided for
 *                 completeness.
 */
#ifndef MESH_VERTEX_INPAINT_H
#define MESH_VERTEX_INPAINT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MVI_SMOOTH  = 0,
    MVI_FORWARD = 1,
};

void mesh_vertex_inpaint(
    const float *texture, const uint8_t *mask,
    const float *vtx_pos, const float *vtx_uv,
    const int32_t *pos_idx, const int32_t *uv_idx,
    int V, int U, int F, int H, int W, int C,
    int method,
    float *out_texture, uint8_t *out_mask);

#ifdef __cplusplus
}
#endif

#ifdef MESH_VERTEX_INPAINT_IMPLEMENTATION

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <vector>

namespace mvi_impl {

static inline void uv_to_pixel(float u, float v, int W, int H, int &row, int &col) {
    /* Matches upstream calculateUVCoordinates():
     *   uv_v = round(vtx_uv[*,0] * (W-1))
     *   uv_u = round((1 - vtx_uv[*,1]) * (H-1))   // note row = 1-v */
    float fc = u * (float)(W - 1);
    float fr = (1.0f - v) * (float)(H - 1);
    col = (int)(fc + (fc >= 0.0f ? 0.5f : -0.5f));
    row = (int)(fr + (fr >= 0.0f ? 0.5f : -0.5f));
    if (col < 0) col = 0; else if (col >= W) col = W - 1;
    if (row < 0) row = 0; else if (row >= H) row = H - 1;
}

static inline float dist_weight(const float *p0, const float *p1) {
    float dx = p0[0] - p1[0];
    float dy = p0[1] - p1[1];
    float dz = p0[2] - p1[2];
    float d  = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (d < 1e-4f) d = 1e-4f;
    float w = 1.0f / d;
    return w * w;
}

struct MeshCtx {
    const float *texture;
    const uint8_t *mask;
    const float *vtx_pos;
    const float *vtx_uv;
    const int32_t *pos_idx;
    const int32_t *uv_idx;
    int V, U, F, H, W, C;
};

static void build_graph(const MeshCtx &ctx, std::vector<std::vector<int>> &G) {
    G.assign((size_t)ctx.V, std::vector<int>());
    /* Same directed-edge accumulation as upstream (each face's three
     * consecutive vertex pairs). Mirrors buildGraph() exactly. */
    for (int f = 0; f < ctx.F; f++) {
        for (int k = 0; k < 3; k++) {
            int a = ctx.pos_idx[f * 3 + k];
            int b = ctx.pos_idx[f * 3 + (k + 1) % 3];
            G[a].push_back(b);
        }
    }
}

/* Initialise vtx_mask[V] and vtx_color[V*C] from the input texture/mask.
 * Also collects vertices that land on an unfilled UV pixel into `uncolored`. */
static void init_vertex_data(const MeshCtx &ctx,
                              std::vector<float> &vtx_mask,
                              std::vector<float> &vtx_color,
                              std::vector<int> *uncolored) {
    vtx_mask.assign((size_t)ctx.V, 0.0f);
    vtx_color.assign((size_t)ctx.V * ctx.C, 0.0f);
    if (uncolored) uncolored->clear();
    for (int f = 0; f < ctx.F; f++) {
        for (int k = 0; k < 3; k++) {
            int uv_idx = ctx.uv_idx[f * 3 + k];
            int v_idx  = ctx.pos_idx[f * 3 + k];
            int row, col;
            uv_to_pixel(ctx.vtx_uv[uv_idx * 2 + 0],
                         ctx.vtx_uv[uv_idx * 2 + 1],
                         ctx.W, ctx.H, row, col);
            size_t pix = (size_t)row * ctx.W + col;
            if (ctx.mask[pix] > 0) {
                vtx_mask[v_idx] = 1.0f;
                for (int c = 0; c < ctx.C; c++) {
                    vtx_color[(size_t)v_idx * ctx.C + c] =
                        ctx.texture[pix * ctx.C + c];
                }
            } else if (uncolored) {
                uncolored->push_back(v_idx);
            }
        }
    }
}

/* Iterative smoothing: each pass visits every uncolored vertex and
 * averages the colors of its already-colored neighbours, weighted by
 * 1/d^2 in world space. Stops after two passes that don't reduce the
 * uncolored count (matching upstream's smooth_count==2 semantics). */
static void smoothing(const MeshCtx &ctx,
                       const std::vector<std::vector<int>> &G,
                       std::vector<float> &vtx_mask,
                       std::vector<float> &vtx_color,
                       const std::vector<int> &uncolored) {
    int smooth_count = 2;
    int last = 0;
    while (smooth_count > 0) {
        int still_uncolored = 0;
        for (int v : uncolored) {
            std::vector<float> sum((size_t)ctx.C, 0.0f);
            float total_w = 0.0f;
            const float *p0 = ctx.vtx_pos + (size_t)v * 3;
            for (int n : G[v]) {
                if (vtx_mask[n] > 0.0f) {
                    const float *pn = ctx.vtx_pos + (size_t)n * 3;
                    float w = dist_weight(p0, pn);
                    const float *cn = &vtx_color[(size_t)n * ctx.C];
                    for (int c = 0; c < ctx.C; c++) sum[c] += cn[c] * w;
                    total_w += w;
                }
            }
            if (total_w > 0.0f) {
                float *cv = &vtx_color[(size_t)v * ctx.C];
                for (int c = 0; c < ctx.C; c++) cv[c] = sum[c] / total_w;
                vtx_mask[v] = 1.0f;
            } else {
                still_uncolored++;
            }
        }
        if (last == still_uncolored) smooth_count--;
        else                         smooth_count++;
        last = still_uncolored;
    }
}

/* Front-propagation variant. Queue-based BFS from the known-colored
 * frontier, applying distance-weighted contributions to neighbours. */
static void forward_propagate(const MeshCtx &ctx,
                               const std::vector<std::vector<int>> &G,
                               std::vector<float> &vtx_mask,
                               std::vector<float> &vtx_color) {
    std::queue<int> active;
    for (int v = 0; v < ctx.V; v++)
        if (vtx_mask[v] == 1.0f) active.push(v);
    while (!active.empty()) {
        std::queue<int> pending;
        while (!active.empty()) {
            int v = active.front(); active.pop();
            const float *p0 = ctx.vtx_pos + (size_t)v * 3;
            const float *cv = &vtx_color[(size_t)v * ctx.C];
            for (int n : G[v]) {
                if (vtx_mask[n] > 0.0f) continue;
                const float *pn = ctx.vtx_pos + (size_t)n * 3;
                float w = dist_weight(p0, pn);
                float *cn = &vtx_color[(size_t)n * ctx.C];
                for (int c = 0; c < ctx.C; c++) cn[c] += cv[c] * w;
                if (vtx_mask[n] == 0.0f) pending.push(n);
                vtx_mask[n] -= w;
            }
        }
        while (!pending.empty()) {
            int v = pending.front(); pending.pop();
            float *cv = &vtx_color[(size_t)v * ctx.C];
            float denom = -vtx_mask[v];
            if (denom > 0.0f)
                for (int c = 0; c < ctx.C; c++) cv[c] /= denom;
            vtx_mask[v] = 1.0f;
            active.push(v);
        }
    }
}

/* Splat the now-colored vertices back into the output texture + mask.
 * Empty pixels stay at their original values (no change). */
static void splat_to_texture(const MeshCtx &ctx,
                              const std::vector<float> &vtx_mask,
                              const std::vector<float> &vtx_color,
                              float *out_texture, uint8_t *out_mask) {
    std::memcpy(out_texture, ctx.texture,
                (size_t)ctx.H * ctx.W * ctx.C * sizeof(float));
    std::memcpy(out_mask, ctx.mask,
                (size_t)ctx.H * ctx.W * sizeof(uint8_t));
    for (int f = 0; f < ctx.F; f++) {
        for (int k = 0; k < 3; k++) {
            int uv_idx = ctx.uv_idx[f * 3 + k];
            int v_idx  = ctx.pos_idx[f * 3 + k];
            if (vtx_mask[v_idx] != 1.0f) continue;
            int row, col;
            uv_to_pixel(ctx.vtx_uv[uv_idx * 2 + 0],
                         ctx.vtx_uv[uv_idx * 2 + 1],
                         ctx.W, ctx.H, row, col);
            size_t pix = (size_t)row * ctx.W + col;
            for (int c = 0; c < ctx.C; c++)
                out_texture[pix * ctx.C + c] =
                    vtx_color[(size_t)v_idx * ctx.C + c];
            out_mask[pix] = 255;
        }
    }
}

} /* namespace mvi_impl */

extern "C" void mesh_vertex_inpaint(
    const float *texture, const uint8_t *mask,
    const float *vtx_pos, const float *vtx_uv,
    const int32_t *pos_idx, const int32_t *uv_idx,
    int V, int U, int F, int H, int W, int C,
    int method,
    float *out_texture, uint8_t *out_mask)
{
    mvi_impl::MeshCtx ctx{ texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx,
                            V, U, F, H, W, C };
    std::vector<float> vtx_mask, vtx_color;
    std::vector<std::vector<int>> G;
    mvi_impl::build_graph(ctx, G);

    if (method == MVI_SMOOTH) {
        std::vector<int> uncolored;
        mvi_impl::init_vertex_data(ctx, vtx_mask, vtx_color, &uncolored);
        mvi_impl::smoothing(ctx, G, vtx_mask, vtx_color, uncolored);
    } else {
        mvi_impl::init_vertex_data(ctx, vtx_mask, vtx_color, nullptr);
        mvi_impl::forward_propagate(ctx, G, vtx_mask, vtx_color);
    }

    mvi_impl::splat_to_texture(ctx, vtx_mask, vtx_color, out_texture, out_mask);
}

#endif /* MESH_VERTEX_INPAINT_IMPLEMENTATION */
#endif /* MESH_VERTEX_INPAINT_H */
