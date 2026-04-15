/*
 * test_view_maps.c - Render per-view normal + position maps for a mesh
 * using the native NVRTC triangle rasterizer + barycentric interpolator.
 *
 * Produces exactly the input control maps that the Hunyuan3D-2.1 paint
 * multiview diffusion UNet consumes. For the default 6-view configuration
 * (front / right / back / left / top / bottom, orthographic) the output
 * closely mirrors what hy3dpaint's MeshRender.render_normal_multiview and
 * render_position_multiview produce.
 *
 * Usage:
 *   ./test_view_maps <mesh.obj> [out_prefix] [resolution]
 *     -> writes <prefix>_view{V}_normal.ppm, <prefix>_view{V}_position.ppm
 *        plus <prefix>_view{V}_normal.npy / _position.npy for diffing.
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

/* ==== OBJ reader + vertex normals (same as test_raster.c) ================= */

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

/* Area-weighted vertex normals. Out is [n_verts, 3] float. */
static void compute_vertex_normals(const obj_mesh *m, float *out) {
    memset(out, 0, (size_t)m->n_verts * 3 * sizeof(float));
    for (int t = 0; t < m->n_tris; t++) {
        int i0 = m->tri[t*3+0], i1 = m->tri[t*3+1], i2 = m->tri[t*3+2];
        float *p0 = (float *)m->pos + i0 * 3;
        float *p1 = (float *)m->pos + i1 * 3;
        float *p2 = (float *)m->pos + i2 * 3;
        float e1[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
        float e2[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
        /* face normal = cross(e1, e2); |n| is 2*area so stays area-weighted */
        float nx = e1[1]*e2[2] - e1[2]*e2[1];
        float ny = e1[2]*e2[0] - e1[0]*e2[2];
        float nz = e1[0]*e2[1] - e1[1]*e2[0];
        out[i0*3+0] += nx; out[i0*3+1] += ny; out[i0*3+2] += nz;
        out[i1*3+0] += nx; out[i1*3+1] += ny; out[i1*3+2] += nz;
        out[i2*3+0] += nx; out[i2*3+1] += ny; out[i2*3+2] += nz;
    }
    for (int v = 0; v < m->n_verts; v++) {
        float x = out[v*3+0], y = out[v*3+1], z = out[v*3+2];
        float len = sqrtf(x*x + y*y + z*z);
        if (len > 1e-12f) {
            out[v*3+0] = x / len;
            out[v*3+1] = y / len;
            out[v*3+2] = z / len;
        }
    }
}

/* ==== .npy writer + .ppm writer =========================================== */

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

/* Float [H,W,3] -> uint8 PPM. For normal maps already in [0,1] range. */
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

/* ==== 4x4 matrix helpers (column-major) =================================== */

static void mat4_identity(float *m) {
    for (int i = 0; i < 16; i++) m[i] = 0.f;
    m[0] = m[5] = m[10] = m[15] = 1.f;
}

static void mat4_ortho(float *m, float l, float r, float b, float t,
                        float znear, float zfar) {
    memset(m, 0, 16 * sizeof(float));
    m[0]  =  2.f / (r - l);
    m[5]  =  2.f / (t - b);
    m[10] = -2.f / (zfar - znear);
    m[12] = -(r + l) / (r - l);
    m[13] = -(t + b) / (t - b);
    m[14] = -(zfar + znear) / (zfar - znear);
    m[15] =  1.f;
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

static void mat4_lookat(float *m, const float *eye, const float *target,
                         const float *up) {
    float f[3] = { target[0]-eye[0], target[1]-eye[1], target[2]-eye[2] };
    float fl = sqrtf(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
    f[0] /= fl; f[1] /= fl; f[2] /= fl;
    float s[3] = { f[1]*up[2]-f[2]*up[1], f[2]*up[0]-f[0]*up[2], f[0]*up[1]-f[1]*up[0] };
    float sl = sqrtf(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
    s[0] /= sl; s[1] /= sl; s[2] /= sl;
    float u[3] = { s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0] };
    mat4_identity(m);
    m[0] = s[0]; m[4] = s[1]; m[8]  = s[2];
    m[1] = u[0]; m[5] = u[1]; m[9]  = u[2];
    m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
    m[12] = -(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]);
    m[13] = -(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]);
    m[14] =  (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]);
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

/* ==== main ================================================================= */

/* Default Hunyuan3D candidate views (first 6 entries of
 * Hunyuan3DPaintConfig.candidate_camera_{azims,elevs}).
 * Angles in degrees. */
static const float default_azims_deg[6] = {  0.f, 90.f, 180.f, 270.f,   0.f, 180.f };
static const float default_elevs_deg[6] = {  0.f,  0.f,   0.f,   0.f,  90.f, -90.f };

static void view_eye_from_angles(float elev_deg, float azim_deg, float dist,
                                   float *eye_out) {
    float e = elev_deg * 3.14159265358979f / 180.f;
    float a = azim_deg * 3.14159265358979f / 180.f;
    /* y-up: elevation rotates around X, azimuth around Y. */
    float cz =  cosf(e) * cosf(a);
    float cx =  cosf(e) * sinf(a);
    float cy =  sinf(e);
    eye_out[0] = dist * cx;
    eye_out[1] = dist * cy;
    eye_out[2] = dist * cz;
}

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

    /* Normalise into unit box */
    float mn[3] = {m.pos[0], m.pos[1], m.pos[2]};
    float mx[3] = {mn[0], mn[1], mn[2]};
    for (int i = 1; i < m.n_verts; i++)
        for (int j = 0; j < 3; j++) {
            float v = m.pos[i*3+j];
            if (v < mn[j]) mn[j] = v;
            if (v > mx[j]) mx[j] = v;
        }
    float ctr[3] = { 0.5f*(mn[0]+mx[0]), 0.5f*(mn[1]+mx[1]), 0.5f*(mn[2]+mx[2]) };
    float ext   = fmaxf(mx[0]-mn[0], fmaxf(mx[1]-mn[1], mx[2]-mn[2]));
    float scale = 1.9f / ext;   /* leaves small margin inside ortho [-1, 1] */
    for (int i = 0; i < m.n_verts; i++) {
        m.pos[i*3+0] = (m.pos[i*3+0] - ctr[0]) * scale;
        m.pos[i*3+1] = (m.pos[i*3+1] - ctr[1]) * scale;
        m.pos[i*3+2] = (m.pos[i*3+2] - ctr[2]) * scale;
    }

    /* Vertex normals (world-space, area-weighted) */
    float *normals = (float *)malloc((size_t)m.n_verts * 3 * sizeof(float));
    compute_vertex_normals(&m, normals);

    /* Position "color" is vtx_pos re-mapped into [0,1] for easy visualisation;
     * matches hy3dpaint render_position's 0.5 + pos/scale recentring. */
    float *pos_attr = (float *)malloc((size_t)m.n_verts * 3 * sizeof(float));
    for (int i = 0; i < m.n_verts; i++) {
        pos_attr[i*3+0] = 0.5f + 0.5f * m.pos[i*3+0];
        pos_attr[i*3+1] = 0.5f + 0.5f * m.pos[i*3+1];
        pos_attr[i*3+2] = 0.5f + 0.5f * m.pos[i*3+2];
    }

    /* Init CUDA, compile kernels */
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
    CUfunction f_raster, f_bary, f_interp;
    cuModuleGetFunction(&f_raster, mod, "rasterize_faces_f32");
    cuModuleGetFunction(&f_bary,   mod, "resolve_bary_f32");
    cuModuleGetFunction(&f_interp, mod, "interpolate_attr_f32");

    /* Static uploads: F, normals, pos_attr, bg */
    size_t f_bytes = (size_t)m.n_tris  * 3 * sizeof(int);
    size_t v3_bytes = (size_t)m.n_verts * 3 * sizeof(float);
    CUdeviceptr d_F, d_N, d_P, d_bg;
    cuMemAlloc(&d_F, f_bytes);
    cuMemAlloc(&d_N, v3_bytes);
    cuMemAlloc(&d_P, v3_bytes);
    cuMemAlloc(&d_bg, 3 * sizeof(float));
    cuMemcpyHtoD(d_F, m.tri, f_bytes);
    cuMemcpyHtoD(d_N, normals, v3_bytes);
    cuMemcpyHtoD(d_P, pos_attr, v3_bytes);
    float bg_white[3] = { 1.f, 1.f, 1.f };
    cuMemcpyHtoD(d_bg, bg_white, 3 * sizeof(float));

    /* Per-view transient uploads */
    size_t v4_bytes = (size_t)m.n_verts * 4 * sizeof(float);
    size_t pix = (size_t)res * res;
    CUdeviceptr d_V, d_zbuf, d_fidx, d_bary, d_nrm_img, d_pos_img;
    cuMemAlloc(&d_V, v4_bytes);
    cuMemAlloc(&d_zbuf, pix * sizeof(uint64_t));
    cuMemAlloc(&d_fidx, pix * sizeof(int32_t));
    cuMemAlloc(&d_bary, pix * 3 * sizeof(float));
    cuMemAlloc(&d_nrm_img, pix * 3 * sizeof(float));
    cuMemAlloc(&d_pos_img, pix * 3 * sizeof(float));

    float *clip = (float *)malloc(v4_bytes);

    for (int v = 0; v < 6; v++) {
        float elev = default_elevs_deg[v];
        float azim = default_azims_deg[v];
        float eye[3], tgt[3] = {0.f,0.f,0.f};
        /* For ±90° elevation (top / bottom) use +Z as "up" so the look-at
         * doesn't collapse against the Y axis. */
        float up[3] = {0.f, 1.f, 0.f};
        if (fabsf(elev) > 89.f) { up[0]=0.f; up[1]=0.f; up[2] = elev > 0 ? -1.f : 1.f; }
        view_eye_from_angles(elev, azim, 3.f, eye);

        float proj[16], view[16], mvp[16];
        mat4_ortho(proj, -1.1f, 1.1f, -1.1f, 1.1f, 0.1f, 10.f);
        mat4_lookat(view, eye, tgt, up);
        mat4_mul(mvp, proj, view);
        apply_mvp(m.pos, m.n_verts, mvp, clip);
        cuMemcpyHtoD(d_V, clip, v4_bytes);

        /* Reset zbuffer */
        {
            uint64_t sentinel =
                (uint64_t)2147483647ULL * 2147483647ULL + 2147483646ULL;
            uint64_t *init = (uint64_t *)malloc(pix * sizeof(uint64_t));
            for (size_t i = 0; i < pix; i++) init[i] = sentinel;
            cuMemcpyHtoD(d_zbuf, init, pix * sizeof(uint64_t));
            free(init);
        }

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
        /* interpolate normals + positions */
        for (int k = 0; k < 2; k++) {
            CUdeviceptr attr = (k == 0) ? d_N : d_P;
            CUdeviceptr out  = (k == 0) ? d_nrm_img : d_pos_img;
            int W = res, H = res, C = 3;
            void *args[] = { &attr, &d_F, &d_fidx, &d_bary, &d_bg, &out,
                             &W, &H, &C };
            unsigned grid = (unsigned)((pix + 255) / 256);
            cuLaunchKernel(f_interp, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        cuCtxSynchronize();

        /* Download + write */
        float *h_nrm = (float *)malloc(pix * 3 * sizeof(float));
        float *h_pos = (float *)malloc(pix * 3 * sizeof(float));
        cuMemcpyDtoH(h_nrm, d_nrm_img, pix * 3 * sizeof(float));
        cuMemcpyDtoH(h_pos, d_pos_img, pix * 3 * sizeof(float));

        char path[1024];
        int sh3[3] = {res, res, 3};
        /* Remap normals from [-1,1] to [0,1] for the PPM and the .npy,
         * matching hy3dpaint render_normal(normalize_rgb=True). */
        float *h_nrm_rgb = (float *)malloc(pix * 3 * sizeof(float));
        for (size_t i = 0; i < pix; i++) {
            for (int c = 0; c < 3; c++)
                h_nrm_rgb[i*3+c] = 0.5f * h_nrm[i*3+c] + 0.5f;
        }
        snprintf(path, sizeof(path), "%s_view%d_normal.ppm", prefix, v);
        write_ppm_rgb_from_f32(path, h_nrm_rgb, res, res, 0.f, 1.f);
        snprintf(path, sizeof(path), "%s_view%d_normal.npy", prefix, v);
        write_npy(path, "<f4", sh3, 3, h_nrm_rgb, sizeof(float));

        snprintf(path, sizeof(path), "%s_view%d_position.ppm", prefix, v);
        write_ppm_rgb_from_f32(path, h_pos, res, res, 0.f, 1.f);
        snprintf(path, sizeof(path), "%s_view%d_position.npy", prefix, v);
        write_npy(path, "<f4", sh3, 3, h_pos, sizeof(float));

        int covered = 0;
        {
            int *h_fidx = (int *)malloc(pix * sizeof(int));
            cuMemcpyDtoH(h_fidx, d_fidx, pix * sizeof(int));
            for (size_t i = 0; i < pix; i++) if (h_fidx[i] > 0) covered++;
            free(h_fidx);
        }
        fprintf(stderr, "view %d  elev=%+.0f azim=%+.0f  coverage=%d (%.1f%%)\n",
                v, elev, azim, covered, 100.0 * covered / (double)pix);

        free(h_nrm); free(h_pos); free(h_nrm_rgb);
    }

    cuMemFree(d_V); cuMemFree(d_F); cuMemFree(d_N); cuMemFree(d_P);
    cuMemFree(d_bg); cuMemFree(d_zbuf); cuMemFree(d_fidx); cuMemFree(d_bary);
    cuMemFree(d_nrm_img); cuMemFree(d_pos_img);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(clip); free(normals); free(pos_attr);
    free(m.pos); free(m.tri);
    return 0;
}
