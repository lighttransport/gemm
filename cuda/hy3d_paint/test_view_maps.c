/*
 * test_view_maps.c - Render per-view normal + position maps for a mesh
 * using the native NVRTC triangle rasterizer + barycentric interpolator.
 *
 * Bit-for-bit target: hy3dpaint MeshRender.render_normal_multiview /
 * render_position_multiview with shader_type="face", camera_type="orth",
 * camera_distance=1.45, ortho_scale=1.2, scale_factor=1.15, auto_center=True.
 *
 * Validate against the reference dumps produced by ref/hy3d/dump_view_maps.py.
 *
 * Usage:
 *   ./test_view_maps <mesh.obj> [out_prefix] [resolution]
 *     -> writes <prefix>_view{V}_{normal,position}.{ppm,npy}
 *
 * Build:
 *   make test_view_maps
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_raster_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ==== OBJ reader ========================================================== */

typedef struct {
    float *pos;
    int   *tri;
    int    n_verts;
    int    n_tris;
} obj_mesh;

static int read_obj(const char *path, obj_mesh *m) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }
    int cap_v = 1 << 14, cap_t = 1 << 14;
    m->pos = (float *)malloc((size_t)cap_v * 3 * sizeof(float));
    m->tri = (int *)  malloc((size_t)cap_t * 3 * sizeof(int));
    m->n_verts = 0; m->n_tris = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3) {
                if (m->n_verts >= cap_v) {
                    cap_v *= 2;
                    m->pos = (float *)realloc(m->pos, (size_t)cap_v * 3 * sizeof(float));
                }
                m->pos[m->n_verts * 3 + 0] = x;
                m->pos[m->n_verts * 3 + 1] = y;
                m->pos[m->n_verts * 3 + 2] = z;
                m->n_verts++;
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            int idx[3] = {0, 0, 0};
            const char *p = line + 2;
            int k = 0;
            while (*p && k < 3) {
                while (*p == ' ' || *p == '\t') p++;
                if (!*p || *p == '\n') break;
                idx[k++] = atoi(p);
                while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
            }
            if (k == 3) {
                for (int i = 0; i < 3; i++) {
                    if (idx[i] < 0) idx[i] = m->n_verts + idx[i];
                    else idx[i] -= 1;
                }
                if (m->n_tris >= cap_t) {
                    cap_t *= 2;
                    m->tri = (int *)realloc(m->tri, (size_t)cap_t * 3 * sizeof(int));
                }
                m->tri[m->n_tris * 3 + 0] = idx[0];
                m->tri[m->n_tris * 3 + 1] = idx[1];
                m->tri[m->n_tris * 3 + 2] = idx[2];
                m->n_tris++;
            }
        }
    }
    fclose(f);
    return 0;
}

/* ==== .npy / .ppm writers ================================================= */

static void write_npy(const char *path, const char *dtype,
                      const int *shape, int ndims,
                      const void *data, size_t elem_bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256], shape_s[128] = "";
    size_t total = 1;
    for (int i = 0; i < ndims; i++) {
        char tmp[32]; snprintf(tmp, sizeof(tmp), "%d, ", shape[i]);
        strcat(shape_s, tmp);
        total *= (size_t)shape[i];
    }
    int hlen = snprintf(hdr, sizeof(hdr),
        "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", dtype, shape_s);
    int tot = 10 + hlen + 1;
    int pad = ((tot + 63) / 64) * 64 - tot;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, elem_bytes, total, f);
    fclose(f);
}

static void write_ppm_rgb_from_f32(const char *path, const float *rgb,
                                    int W, int H, float lo, float hi) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    float rng = hi - lo;
    for (int i = 0; i < W * H; i++) {
        for (int c = 0; c < 3; c++) {
            float v = (rgb[i*3+c] - lo) / rng;
            if (v < 0.f) v = 0.f;
            if (v > 1.f) v = 1.f;
            uint8_t b = (uint8_t)(v * 255.f + 0.5f);
            fputc(b, f);
        }
    }
    fclose(f);
}

/* ==== Reference camera + projection (mirrors hy3dpaint camera_utils.py) === */

/* All matrices stored in column-major float[16] (so M[col*4+row] indexing). */

static void mat4_zero(float *m) { memset(m, 0, 16 * sizeof(float)); }

/* hy3dpaint get_orthographic_projection_matrix:
 *   ortho[0,0] = 2/(r-l)
 *   ortho[1,1] = 2/(t-b)
 *   ortho[2,2] = -2/(f-n)
 *   ortho[0,3] = -(r+l)/(r-l)
 *   ortho[1,3] = -(t+b)/(t-b)
 *   ortho[2,3] = -(f+n)/(f-n)
 *   ortho[3,3] = 1   (np.eye starts at identity)
 *
 * NOTE: unlike standard OpenGL ortho, [3,3] stays = 1 (eye init). For the
 * symmetric ortho_scale=1.2 case used here, terms [0,3]/[1,3] vanish anyway. */
static void mat4_hy3d_ortho(float *m, float l, float r, float b, float t,
                              float n, float fr) {
    mat4_zero(m);
    /* col-major: m[col*4 + row] */
    m[0*4 + 0] =  2.f / (r - l);
    m[1*4 + 1] =  2.f / (t - b);
    m[2*4 + 2] = -2.f / (fr - n);
    m[3*4 + 0] = -(r + l) / (r - l);
    m[3*4 + 1] = -(t + b) / (t - b);
    m[3*4 + 2] = -(fr + n) / (fr - n);
    m[3*4 + 3] =  1.f;
}

/* hy3dpaint get_mv_matrix: returns w2c. Up is always Z.
 * Done in double precision so the elev=±90 degenerate case matches numpy
 * exactly — the (cos(±pi/2)=6.12e-17) tie-breaking for `right` is sensitive
 * to float rounding and would otherwise rotate the top/bottom views by 180°. */
static void mat4_hy3d_view(float *out, float elev_deg, float azim_deg,
                             float dist) {
    double e_rad = -((double)elev_deg) * 3.141592653589793 / 180.0;
    double a_rad = ((double)azim_deg + 90.0) * 3.141592653589793 / 180.0;
    double ce = cos(e_rad), se = sin(e_rad);
    double ca = cos(a_rad), sa = sin(a_rad);
    double cam[3] = { (double)dist * ce * ca,
                      (double)dist * ce * sa,
                      (double)dist * se };

    double lookat[3] = { -cam[0], -cam[1], -cam[2] };
    double ll = sqrt(lookat[0]*lookat[0] + lookat[1]*lookat[1] + lookat[2]*lookat[2]);
    lookat[0] /= ll; lookat[1] /= ll; lookat[2] /= ll;

    double up[3] = { 0.0, 0.0, 1.0 };
    double right[3] = {
        lookat[1]*up[2] - lookat[2]*up[1],
        lookat[2]*up[0] - lookat[0]*up[2],
        lookat[0]*up[1] - lookat[1]*up[0],
    };
    double rl = sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    right[0] /= rl; right[1] /= rl; right[2] /= rl;

    up[0] = right[1]*lookat[2] - right[2]*lookat[1];
    up[1] = right[2]*lookat[0] - right[0]*lookat[2];
    up[2] = right[0]*lookat[1] - right[1]*lookat[0];
    double ul = sqrt(up[0]*up[0] + up[1]*up[1] + up[2]*up[2]);
    up[0] /= ul; up[1] /= ul; up[2] /= ul;

    /* R = [right | up | -lookat]  (3x3, column vectors).
     * w2c = [R^T | -R^T cam]. In row-major:
     *   w2c[0,:3] = right;  w2c[1,:3] = up;  w2c[2,:3] = -lookat
     *   w2c[0,3]  = -dot(right, cam); etc.
     * Convert to column-major: out[col*4 + row] = w2c[row, col]. */
    mat4_zero(out);
    out[0*4 + 0] = (float)right[0];  out[1*4 + 0] = (float)right[1];  out[2*4 + 0] = (float)right[2];
    out[0*4 + 1] = (float)up[0];     out[1*4 + 1] = (float)up[1];     out[2*4 + 1] = (float)up[2];
    out[0*4 + 2] = (float)(-lookat[0]); out[1*4 + 2] = (float)(-lookat[1]); out[2*4 + 2] = (float)(-lookat[2]);
    out[3*4 + 0] = (float)(-(right[0]*cam[0] + right[1]*cam[1] + right[2]*cam[2]));
    out[3*4 + 1] = (float)(-(up[0]*cam[0] + up[1]*cam[1] + up[2]*cam[2]));
    out[3*4 + 2] = (float)(-(-lookat[0]*cam[0] - lookat[1]*cam[1] - lookat[2]*cam[2]));
    out[3*4 + 3] = 1.f;
}

static void mat4_mul(float *out, const float *a, const float *b) {
    float r[16];
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            float s = 0.f;
            for (int k = 0; k < 4; k++) s += a[k * 4 + rr] * b[c * 4 + k];
            r[c * 4 + rr] = s;
        }
    memcpy(out, r, 16 * sizeof(float));
}

static void apply_mvp(const float *pos, int n, const float *M, float *out) {
    for (int i = 0; i < n; i++) {
        float x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        out[i*4+0] = M[0]*x + M[4]*y + M[8] *z + M[12];
        out[i*4+1] = M[1]*x + M[5]*y + M[9] *z + M[13];
        out[i*4+2] = M[2]*x + M[6]*y + M[10]*z + M[14];
        out[i*4+3] = M[3]*x + M[7]*y + M[11]*z + M[15];
    }
}

/* ==== main ================================================================ */

static const float default_azims_deg[6] = {  0.f, 90.f, 180.f, 270.f,   0.f, 180.f };
static const float default_elevs_deg[6] = {  0.f,  0.f,   0.f,   0.f,  90.f, -90.f };

#define HY3D_CAMERA_DISTANCE 1.45f
#define HY3D_ORTHO_SCALE     1.2f
#define HY3D_SCALE_FACTOR    1.15f

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <mesh.obj> [out_prefix] [resolution]\n", argv[0]);
        return 1;
    }
    const char *obj_path = argv[1];
    const char *prefix   = argc >= 3 ? argv[2] : "views";
    int res              = argc >= 4 ? atoi(argv[3]) : 512;

    obj_mesh m = {0};
    if (read_obj(obj_path, &m) != 0) return 1;
    fprintf(stderr, "Loaded %s: %d v, %d t\n", obj_path, m.n_verts, m.n_tris);

    /* set_mesh: negate X+Y, then swap Y/Z. */
    for (int i = 0; i < m.n_verts; i++) {
        float x = m.pos[i*3+0], y = m.pos[i*3+1], z = m.pos[i*3+2];
        x = -x; y = -y;
        /* swap Y and Z */
        m.pos[i*3+0] = x;
        m.pos[i*3+1] = z;
        m.pos[i*3+2] = y;
    }

    /* auto_center: scale = max_dist_from_center * 2; vtx = (vtx-center) * sf/scale */
    float mn[3] = {m.pos[0], m.pos[1], m.pos[2]};
    float mx[3] = {mn[0], mn[1], mn[2]};
    for (int i = 1; i < m.n_verts; i++)
        for (int j = 0; j < 3; j++) {
            float v = m.pos[i*3+j];
            if (v < mn[j]) mn[j] = v;
            if (v > mx[j]) mx[j] = v;
        }
    float ctr[3] = { 0.5f*(mn[0]+mx[0]), 0.5f*(mn[1]+mx[1]), 0.5f*(mn[2]+mx[2]) };
    float maxd2 = 0.f;
    for (int i = 0; i < m.n_verts; i++) {
        float dx = m.pos[i*3+0] - ctr[0];
        float dy = m.pos[i*3+1] - ctr[1];
        float dz = m.pos[i*3+2] - ctr[2];
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > maxd2) maxd2 = d2;
    }
    float scale = sqrtf(maxd2) * 2.0f;
    float k = HY3D_SCALE_FACTOR / scale;
    for (int i = 0; i < m.n_verts; i++) {
        m.pos[i*3+0] = (m.pos[i*3+0] - ctr[0]) * k;
        m.pos[i*3+1] = (m.pos[i*3+1] - ctr[1]) * k;
        m.pos[i*3+2] = (m.pos[i*3+2] - ctr[2]) * k;
    }

    /* Per-vertex position attribute: tex_position = 0.5 - vtx_pos / scale_factor.
     * Linear in vtx coords so barycentric interpolation reproduces the formula
     * exactly with no need for a face-shader path. */
    float *pos_attr = (float *)malloc((size_t)m.n_verts * 3 * sizeof(float));
    for (int i = 0; i < m.n_verts; i++) {
        pos_attr[i*3+0] = 0.5f - m.pos[i*3+0] / HY3D_SCALE_FACTOR;
        pos_attr[i*3+1] = 0.5f - m.pos[i*3+1] / HY3D_SCALE_FACTOR;
        pos_attr[i*3+2] = 0.5f - m.pos[i*3+2] / HY3D_SCALE_FACTOR;
    }

    /* CUDA + kernel compile */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    CUmodule mod;
    int sm = cu_compile_kernels(&mod, dev,
                                cuda_paint_raster_kernels_src,
                                "hy3d_paint_raster", 1, "HY3D-PAINT");
    if (sm < 0) return 1;
    CUfunction f_raster, f_bary, f_interp, f_facenrm, f_lookup;
    cuModuleGetFunction(&f_raster,  mod, "rasterize_faces_f32");
    cuModuleGetFunction(&f_bary,    mod, "resolve_bary_f32");
    cuModuleGetFunction(&f_interp,  mod, "interpolate_attr_f32");
    cuModuleGetFunction(&f_facenrm, mod, "compute_face_normals_f32");
    cuModuleGetFunction(&f_lookup,  mod, "lookup_face_attr_f32");

    /* Static device buffers */
    size_t f_bytes  = (size_t)m.n_tris  * 3 * sizeof(int);
    size_t v3_bytes = (size_t)m.n_verts * 3 * sizeof(float);
    size_t fn_bytes = (size_t)m.n_tris  * 3 * sizeof(float);
    CUdeviceptr d_F, d_Vw, d_FN, d_P, d_bg;
    cuMemAlloc(&d_F,  f_bytes);
    cuMemAlloc(&d_Vw, v3_bytes);   /* world-space verts (for face-normal calc) */
    cuMemAlloc(&d_FN, fn_bytes);
    cuMemAlloc(&d_P,  v3_bytes);
    cuMemAlloc(&d_bg, 3 * sizeof(float));
    cuMemcpyHtoD(d_F,  m.tri,    f_bytes);
    cuMemcpyHtoD(d_Vw, m.pos,    v3_bytes);
    cuMemcpyHtoD(d_P,  pos_attr, v3_bytes);
    float bg_white[3] = { 1.f, 1.f, 1.f };
    cuMemcpyHtoD(d_bg, bg_white, 3 * sizeof(float));

    /* World-space face normals — one cross-product per triangle. */
    {
        int nf = m.n_tris;
        void *args[] = { &d_Vw, &d_F, &d_FN, &nf };
        unsigned grid = (unsigned)((nf + 255) / 256);
        cuLaunchKernel(f_facenrm, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
    }

    /* Per-view transient buffers */
    size_t v4_bytes = (size_t)m.n_verts * 4 * sizeof(float);
    size_t pix = (size_t)res * res;
    CUdeviceptr d_V, d_zbuf, d_fidx, d_bary, d_nrm_img, d_pos_img;
    cuMemAlloc(&d_V,       v4_bytes);
    cuMemAlloc(&d_zbuf,    pix * sizeof(uint64_t));
    cuMemAlloc(&d_fidx,    pix * sizeof(int32_t));
    cuMemAlloc(&d_bary,    pix * 3 * sizeof(float));
    cuMemAlloc(&d_nrm_img, pix * 3 * sizeof(float));
    cuMemAlloc(&d_pos_img, pix * 3 * sizeof(float));

    /* Persistent zbuffer reset host buffer */
    uint64_t *zbuf_init = (uint64_t *)malloc(pix * sizeof(uint64_t));
    {
        uint64_t sentinel = (uint64_t)2147483647ULL * 2147483647ULL + 2147483646ULL;
        for (size_t i = 0; i < pix; i++) zbuf_init[i] = sentinel;
    }

    float *clip = (float *)malloc(v4_bytes);
    float *h_nrm = (float *)malloc(pix * 3 * sizeof(float));
    float *h_pos = (float *)malloc(pix * 3 * sizeof(float));

    for (int v = 0; v < 6; v++) {
        float elev = default_elevs_deg[v];
        float azim = default_azims_deg[v];

        float proj[16], view[16], mvp[16];
        mat4_hy3d_ortho(proj,
                          -HY3D_ORTHO_SCALE * 0.5f,  HY3D_ORTHO_SCALE * 0.5f,
                          -HY3D_ORTHO_SCALE * 0.5f,  HY3D_ORTHO_SCALE * 0.5f,
                          0.1f, 100.f);
        mat4_hy3d_view(view, elev, azim, HY3D_CAMERA_DISTANCE);
        mat4_mul(mvp, proj, view);
        apply_mvp(m.pos, m.n_verts, mvp, clip);
        cuMemcpyHtoD(d_V, clip, v4_bytes);

        cuMemcpyHtoD(d_zbuf, zbuf_init, pix * sizeof(uint64_t));

        /* rasterize */
        {
            int nf = m.n_tris, W = res, H = res;
            void *args[] = { &d_V, &d_F, &d_zbuf, &nf, &W, &H };
            unsigned grid = (unsigned)((nf + 255) / 256);
            cuLaunchKernel(f_raster, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        /* resolve_bary */
        {
            int W = res, H = res;
            void *args[] = { &d_V, &d_F, &d_zbuf, &d_fidx, &d_bary, &W, &H };
            unsigned grid = (unsigned)((pix + 255) / 256);
            cuLaunchKernel(f_bary, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        /* face-normal lookup (face-shader path) */
        {
            int W = res, H = res, C = 3;
            void *args[] = { &d_FN, &d_fidx, &d_bg, &d_nrm_img, &W, &H, &C };
            unsigned grid = (unsigned)((pix + 255) / 256);
            /* Background must equal post-(n+1)*0.5 white = 1.0, so we feed
             * the kernel bg=1.0 — but the kernel writes bg directly, then we
             * apply (n+1)*0.5 below for non-empty pixels only. Simpler: write
             * raw face normals here, then post-process per-pixel using fidx. */
            cuLaunchKernel(f_lookup, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        /* interpolate position (per-vertex; linear so bary is exact) */
        {
            int W = res, H = res, C = 3;
            void *args[] = { &d_P, &d_F, &d_fidx, &d_bary, &d_bg, &d_pos_img,
                             &W, &H, &C };
            unsigned grid = (unsigned)((pix + 255) / 256);
            cuLaunchKernel(f_interp, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        cuCtxSynchronize();

        cuMemcpyDtoH(h_nrm, d_nrm_img, pix * 3 * sizeof(float));
        cuMemcpyDtoH(h_pos, d_pos_img, pix * 3 * sizeof(float));

        /* Apply (n+1)*0.5 to covered pixels; empty pixels currently hold the
         * bg [1,1,1], which already matches post-formula bg. We need to know
         * coverage — we re-fetch fidx only to count below; here apply blindly
         * to all and then overwrite empties with 1.0. */
        int *h_fidx = (int *)malloc(pix * sizeof(int));
        cuMemcpyDtoH(h_fidx, d_fidx, pix * sizeof(int));
        for (size_t i = 0; i < pix; i++) {
            if (h_fidx[i] > 0) {
                h_nrm[i*3+0] = (h_nrm[i*3+0] + 1.f) * 0.5f;
                h_nrm[i*3+1] = (h_nrm[i*3+1] + 1.f) * 0.5f;
                h_nrm[i*3+2] = (h_nrm[i*3+2] + 1.f) * 0.5f;
            } else {
                /* lookup_face_attr_f32 already wrote bg=[1,1,1] for empty */
            }
        }

        char path[1024];
        int sh3[3] = {res, res, 3};
        snprintf(path, sizeof(path), "%s_view%d_normal.ppm", prefix, v);
        write_ppm_rgb_from_f32(path, h_nrm, res, res, 0.f, 1.f);
        snprintf(path, sizeof(path), "%s_view%d_normal.npy", prefix, v);
        write_npy(path, "<f4", sh3, 3, h_nrm, sizeof(float));
        snprintf(path, sizeof(path), "%s_view%d_position.ppm", prefix, v);
        write_ppm_rgb_from_f32(path, h_pos, res, res, 0.f, 1.f);
        snprintf(path, sizeof(path), "%s_view%d_position.npy", prefix, v);
        write_npy(path, "<f4", sh3, 3, h_pos, sizeof(float));

        int covered = 0;
        for (size_t i = 0; i < pix; i++) if (h_fidx[i] > 0) covered++;
        fprintf(stderr,
                "view %d  elev=%+.0f azim=%+.0f  coverage=%d (%.1f%%)\n",
                v, elev, azim, covered, 100.0 * covered / (double)pix);
        free(h_fidx);
    }

    free(h_nrm); free(h_pos); free(clip); free(zbuf_init);
    cuMemFree(d_V); cuMemFree(d_F); cuMemFree(d_Vw); cuMemFree(d_FN);
    cuMemFree(d_P); cuMemFree(d_bg);
    cuMemFree(d_zbuf); cuMemFree(d_fidx); cuMemFree(d_bary);
    cuMemFree(d_nrm_img); cuMemFree(d_pos_img);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(pos_attr);
    free(m.pos); free(m.tri);
    return 0;
}
