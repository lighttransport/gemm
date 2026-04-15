/*
 * test_vertex_inpaint.cc - Standalone test of the native vertex-space
 * texture inpainter. Runs both methods (smooth + forward) on a
 * synthetic mesh where some UV pixels are known and the rest are to
 * be filled by graph diffusion. Writes .npy outputs for comparison
 * against the upstream pybind11 module (mesh_inpaint_processor).
 *
 * Usage:
 *   ./test_vertex_inpaint [tex_size]     # default 64
 *     -> /tmp/mvi_smooth_tex.npy, /tmp/mvi_smooth_mask.npy,
 *        /tmp/mvi_forward_tex.npy, /tmp/mvi_forward_mask.npy,
 *        /tmp/mvi_input_tex.npy, /tmp/mvi_input_mask.npy,
 *        /tmp/mvi_mesh.npz (positions, uvs, pos_idx, uv_idx)
 *
 * Verify against Python:
 *   cd ../../ref/hy3d && .venv/bin/python -c "
 *   import numpy as np, mesh_inpaint_processor as mip
 *   tex = np.load('/tmp/mvi_input_tex.npy')
 *   mask = np.load('/tmp/mvi_input_mask.npy')
 *   npz = np.load('/tmp/mvi_mesh.npz')
 *   for m in ('smooth', 'forward'):
 *       out_tex, out_mask = mip.meshVerticeInpaint(
 *           tex, mask, npz['pos'], npz['uv'], npz['pos_idx'], npz['uv_idx'], m)
 *       cu_tex  = np.load(f'/tmp/mvi_{m}_tex.npy')
 *       cu_mask = np.load(f'/tmp/mvi_{m}_mask.npy')
 *       dt = np.abs(out_tex - cu_tex); dm = np.abs(out_mask.astype(int) - cu_mask.astype(int))
 *       print(f'{m}: tex max={dt.max():.2e} mean={dt.mean():.2e}, mask diff={int(dm.sum())}')
 *   "
 */

#define MESH_VERTEX_INPAINT_IMPLEMENTATION
#include "mesh_vertex_inpaint.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>

/* ---- minimal .npy / .npz writers ---- */

static void write_npy(const char *path, const char *dtype,
                      const int *shape, int ndims,
                      const void *data, size_t elem_bytes) {
    FILE *f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "ERROR: cannot open %s\n", path); return; }
    std::fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0}; std::fwrite(ver, 1, 2, f);
    char hdr[256], shape_s[128] = "";
    size_t total = 1;
    for (int i = 0; i < ndims; i++) {
        char tmp[32]; std::snprintf(tmp, sizeof(tmp), "%d, ", shape[i]);
        std::strcat(shape_s, tmp);
        total *= (size_t)shape[i];
    }
    int hlen = std::snprintf(hdr, sizeof(hdr),
        "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", dtype, shape_s);
    int tot = 10 + hlen + 1;
    int pad = ((tot + 63) / 64) * 64 - tot;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    std::fwrite(&header_len, 2, 1, f);
    std::fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) std::fputc(' ', f);
    std::fputc('\n', f);
    std::fwrite(data, elem_bytes, total, f);
    std::fclose(f);
}

int main(int argc, char **argv) {
    int H = (argc >= 2) ? std::atoi(argv[1]) : 64;
    int W = H;
    const int C = 3;

    /* Build a toy mesh: a grid of (Nx * Ny) vertices over a unit square,
     * triangulated into quads, with UVs matching positions. */
    const int Nx = 9, Ny = 9;
    const int Vcount = Nx * Ny;
    const int Fcount = (Nx - 1) * (Ny - 1) * 2;

    std::vector<float> pos  ((size_t)Vcount * 3);
    std::vector<float> uv   ((size_t)Vcount * 2);
    std::vector<int32_t> pos_idx((size_t)Fcount * 3);
    std::vector<int32_t> uv_idx ((size_t)Fcount * 3);

    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
            int i = y * Nx + x;
            float fx = (float)x / (float)(Nx - 1);
            float fy = (float)y / (float)(Ny - 1);
            pos[i*3+0] = fx;
            pos[i*3+1] = fy;
            pos[i*3+2] = 0.0f;
            uv[i*2+0] = fx;
            uv[i*2+1] = fy;
        }
    }
    int fi = 0;
    for (int y = 0; y < Ny - 1; y++) {
        for (int x = 0; x < Nx - 1; x++) {
            int a = y * Nx + x;
            int b = a + 1;
            int c = a + Nx;
            int d = c + 1;
            pos_idx[fi*3+0] = a; pos_idx[fi*3+1] = b; pos_idx[fi*3+2] = c;
            uv_idx [fi*3+0] = a; uv_idx [fi*3+1] = b; uv_idx [fi*3+2] = c;
            fi++;
            pos_idx[fi*3+0] = b; pos_idx[fi*3+1] = d; pos_idx[fi*3+2] = c;
            uv_idx [fi*3+0] = b; uv_idx [fi*3+1] = d; uv_idx [fi*3+2] = c;
            fi++;
        }
    }

    /* Input texture: a smooth colour gradient in an L-shape region,
     * rest of the atlas left as zero. Mask marks the L-shape pixels. */
    std::vector<float>  tex ((size_t)H * W * C, 0.0f);
    std::vector<uint8_t> mask((size_t)H * W, 0);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            bool in = (x < W / 2 && y < H * 3 / 4) || (y < H / 3);
            if (in) {
                float u = (float)x / (float)(W - 1);
                float v = (float)y / (float)(H - 1);
                tex[(y * W + x) * C + 0] = u;
                tex[(y * W + x) * C + 1] = v;
                tex[(y * W + x) * C + 2] = 0.5f * (1.0f - u - v) + 0.5f;
                mask[y * W + x] = 255;
            }
        }
    }

    /* Run both methods */
    std::vector<float>  out_tex_smooth ((size_t)H * W * C);
    std::vector<uint8_t> out_mask_smooth((size_t)H * W);
    mesh_vertex_inpaint(tex.data(), mask.data(), pos.data(), uv.data(),
                        pos_idx.data(), uv_idx.data(),
                        Vcount, Vcount, Fcount, H, W, C,
                        MVI_SMOOTH,
                        out_tex_smooth.data(), out_mask_smooth.data());

    std::vector<float>  out_tex_fwd ((size_t)H * W * C);
    std::vector<uint8_t> out_mask_fwd((size_t)H * W);
    mesh_vertex_inpaint(tex.data(), mask.data(), pos.data(), uv.data(),
                        pos_idx.data(), uv_idx.data(),
                        Vcount, Vcount, Fcount, H, W, C,
                        MVI_FORWARD,
                        out_tex_fwd.data(), out_mask_fwd.data());

    int known = 0, smooth_new = 0, fwd_new = 0;
    for (int i = 0; i < H * W; i++) {
        if (mask[i] > 0) known++;
        if (mask[i] == 0 && out_mask_smooth[i] > 0) smooth_new++;
        if (mask[i] == 0 && out_mask_fwd[i]    > 0) fwd_new++;
    }
    std::fprintf(stderr,
        "Input mesh: %d verts, %d faces, tex %dx%d\n"
        "Input mask: %d / %d known pixels (%.1f%%)\n"
        "Smooth  filled %d new pixels\n"
        "Forward filled %d new pixels\n",
        Vcount, Fcount, H, W, known, H*W,
        100.0 * known / (double)(H * W),
        smooth_new, fwd_new);

    /* Dump inputs + outputs + mesh for cross-check */
    int sh3[3] = {H, W, C};
    int sh2[2] = {H, W};
    write_npy("/tmp/mvi_input_tex.npy",  "<f4", sh3, 3, tex.data(),  sizeof(float));
    write_npy("/tmp/mvi_input_mask.npy", "|u1", sh2, 2, mask.data(), sizeof(uint8_t));
    write_npy("/tmp/mvi_smooth_tex.npy", "<f4", sh3, 3, out_tex_smooth.data(),  sizeof(float));
    write_npy("/tmp/mvi_smooth_mask.npy","|u1", sh2, 2, out_mask_smooth.data(), sizeof(uint8_t));
    write_npy("/tmp/mvi_forward_tex.npy","<f4", sh3, 3, out_tex_fwd.data(),  sizeof(float));
    write_npy("/tmp/mvi_forward_mask.npy","|u1", sh2, 2, out_mask_fwd.data(), sizeof(uint8_t));

    int shV3[2] = {Vcount, 3};
    int shV2[2] = {Vcount, 2};
    int shF3[2] = {Fcount, 3};
    write_npy("/tmp/mvi_pos.npy",    "<f4", shV3, 2, pos.data(),    sizeof(float));
    write_npy("/tmp/mvi_uv.npy",     "<f4", shV2, 2, uv.data(),     sizeof(float));
    write_npy("/tmp/mvi_pos_idx.npy","<i4", shF3, 2, pos_idx.data(),sizeof(int32_t));
    write_npy("/tmp/mvi_uv_idx.npy", "<i4", shF3, 2, uv_idx.data(), sizeof(int32_t));
    std::fprintf(stderr, "Wrote /tmp/mvi_*.npy for cross-check\n");
    return 0;
}
