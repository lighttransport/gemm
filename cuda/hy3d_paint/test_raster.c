/*
 * test_raster.c - Standalone test of the native Hunyuan3D-2.1 paint
 * triangle rasterizer (cuda_paint_raster_kernels.h). Loads an OBJ,
 * transforms it through a fixed MVP, runs rasterize_faces_f32 +
 * resolve_bary_f32 on the GPU, and writes the findex map + barycentric
 * buffer as .npy files for comparison against the upstream PyTorch
 * custom_rasterizer.
 *
 * Usage:
 *   ./test_raster <input.obj> [out_prefix] [resolution]
 *     -> writes <prefix>_findices.npy, <prefix>_bary.npy, <prefix>_clip.npy
 *
 * Build:
 *   make test_raster
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

/* ---- Minimal OBJ reader (positions + triangle indices only) ---- */

typedef struct {
    float *pos;    /* [n_verts * 3] */
    int   *tri;    /* [n_tris * 3] */
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

/* ---- Minimal .npy writer ---- */

static void write_npy(const char *path, const char *dtype,
                      const int *shape, int ndims,
                      const void *data, size_t elem_bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return; }
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

/* ---- Simple 4x4 column-major matrix helpers ---- */

static void mat4_identity(float *m) {
    for (int i = 0; i < 16; i++) m[i] = 0.f;
    m[0] = m[5] = m[10] = m[15] = 1.f;
}

static void mat4_perspective(float *m, float fov_rad, float aspect,
                              float znear, float zfar) {
    float f = 1.f / tanf(fov_rad * 0.5f);
    memset(m, 0, 16 * sizeof(float));
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = (zfar + znear) / (znear - zfar);
    m[11] = -1.f;
    m[14] = (2.f * zfar * znear) / (znear - zfar);
}

/* out = a * b, column-major */
static void mat4_mul(float *out, const float *a, const float *b) {
    float r[16];
    for (int c = 0; c < 4; c++) {
        for (int rr = 0; rr < 4; rr++) {
            float s = 0.f;
            for (int k = 0; k < 4; k++) s += a[k * 4 + rr] * b[c * 4 + k];
            r[c * 4 + rr] = s;
        }
    }
    memcpy(out, r, 16 * sizeof(float));
}

/* Look-at from eye toward target with up. Column-major. */
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

/* Transform [N,3] positions by column-major 4x4 M into [N,4] clip-space. */
static void apply_mvp(const float *pos, int n, const float *M, float *out) {
    for (int i = 0; i < n; i++) {
        float x = pos[i*3+0], y = pos[i*3+1], z = pos[i*3+2];
        out[i*4+0] = M[0]*x + M[4]*y + M[8] *z + M[12];
        out[i*4+1] = M[1]*x + M[5]*y + M[9] *z + M[13];
        out[i*4+2] = M[2]*x + M[6]*y + M[10]*z + M[14];
        out[i*4+3] = M[3]*x + M[7]*y + M[11]*z + M[15];
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <mesh.obj> [out_prefix] [resolution]\n\n"
            "Writes <prefix>_findices.npy [H,W int32],\n"
            "       <prefix>_bary.npy     [H,W,3 float32],\n"
            "       <prefix>_clip.npy     [V,4 float32] (clip-space verts)\n",
            argv[0]);
        return 1;
    }
    const char *obj_path = argv[1];
    const char *prefix   = argc >= 3 ? argv[2] : "raster";
    int res              = argc >= 4 ? atoi(argv[3]) : 512;

    obj_mesh m = {0};
    if (read_obj(obj_path, &m) != 0) return 1;
    fprintf(stderr, "Loaded %s: %d verts, %d tris\n", obj_path, m.n_verts, m.n_tris);

    /* Fit mesh into a unit box centered at origin. */
    float mn[3] = {m.pos[0], m.pos[1], m.pos[2]};
    float mx[3] = {mn[0], mn[1], mn[2]};
    for (int i = 1; i < m.n_verts; i++) {
        for (int j = 0; j < 3; j++) {
            float v = m.pos[i*3+j];
            if (v < mn[j]) mn[j] = v;
            if (v > mx[j]) mx[j] = v;
        }
    }
    float ctr[3] = { 0.5f*(mn[0]+mx[0]), 0.5f*(mn[1]+mx[1]), 0.5f*(mn[2]+mx[2]) };
    float ext[3] = { mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2] };
    float scale  = 2.f / fmaxf(ext[0], fmaxf(ext[1], ext[2]));
    for (int i = 0; i < m.n_verts; i++) {
        m.pos[i*3+0] = (m.pos[i*3+0] - ctr[0]) * scale;
        m.pos[i*3+1] = (m.pos[i*3+1] - ctr[1]) * scale;
        m.pos[i*3+2] = (m.pos[i*3+2] - ctr[2]) * scale;
    }
    fprintf(stderr, "Mesh normalised: bbox %.3f %.3f %.3f\n",
            ext[0]*scale, ext[1]*scale, ext[2]*scale);

    /* Camera: look at origin from +Z = 3, perspective 45°. */
    float proj[16], view[16], mvp[16];
    mat4_perspective(proj, 45.f * 3.14159265358979f / 180.f, 1.f, 0.1f, 10.f);
    float eye[3] = {0.f, 0.f, 3.f}, tgt[3] = {0.f, 0.f, 0.f}, up[3] = {0.f, 1.f, 0.f};
    mat4_lookat(view, eye, tgt, up);
    mat4_mul(mvp, proj, view);

    float *clip = (float *)malloc((size_t)m.n_verts * 4 * sizeof(float));
    apply_mvp(m.pos, m.n_verts, mvp, clip);

    /* Init CUDA */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    char name[256]; cuDeviceGetName(name, sizeof(name), dev);
    fprintf(stderr, "GPU: %s\n", name);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    /* Compile kernels */
    CUmodule mod;
    int sm = cu_compile_kernels(&mod, dev,
                                cuda_paint_raster_kernels_src,
                                "hy3d_paint_raster", 1, "HY3D-PAINT");
    if (sm < 0) return 1;

    CUfunction f_raster, f_bary;
    cuModuleGetFunction(&f_raster, mod, "rasterize_faces_f32");
    cuModuleGetFunction(&f_bary,   mod, "resolve_bary_f32");

    /* Upload V, F; allocate zbuffer init'd to a "big" int64; allocate outputs */
    size_t v_bytes = (size_t)m.n_verts * 4 * sizeof(float);
    size_t f_bytes = (size_t)m.n_tris  * 3 * sizeof(int);
    size_t pix     = (size_t)res * res;
    CUdeviceptr d_V, d_F, d_zbuf, d_fidx, d_bary;
    cuMemAlloc(&d_V,    v_bytes);
    cuMemAlloc(&d_F,    f_bytes);
    cuMemAlloc(&d_zbuf, pix * sizeof(uint64_t));
    cuMemAlloc(&d_fidx, pix * sizeof(int32_t));
    cuMemAlloc(&d_bary, pix * 3 * sizeof(float));
    cuMemcpyHtoD(d_V, clip, v_bytes);
    cuMemcpyHtoD(d_F, m.tri, f_bytes);

    /* Initialise zbuffer with the same sentinel upstream uses:
     *   maxint = (INT64)INT32_MAX * INT32_MAX + (INT32_MAX - 1)
     * We upload once from a host-side buffer so every pixel's initial
     * token loses every atomicMin comparison until a face writes. */
    {
        uint64_t sentinel = (uint64_t)2147483647ULL * 2147483647ULL + 2147483646ULL;
        uint64_t *init = (uint64_t *)malloc(pix * sizeof(uint64_t));
        for (size_t i = 0; i < pix; i++) init[i] = sentinel;
        cuMemcpyHtoD(d_zbuf, init, pix * sizeof(uint64_t));
        free(init);
    }

    /* Launch rasterize_faces: one thread per face */
    {
        int num_faces = m.n_tris;
        int width = res, height = res;
        void *args[] = { &d_V, &d_F, &d_zbuf, &num_faces, &width, &height };
        unsigned grid = (unsigned)((num_faces + 255) / 256);
        cuLaunchKernel(f_raster, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
    }

    /* Launch resolve_bary: one thread per pixel */
    {
        int width = res, height = res;
        void *args[] = { &d_V, &d_F, &d_zbuf, &d_fidx, &d_bary, &width, &height };
        unsigned grid = (unsigned)((pix + 255) / 256);
        cuLaunchKernel(f_bary, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
    }
    cuCtxSynchronize();

    /* Download + dump */
    int   *h_fidx = (int *)  malloc(pix * sizeof(int));
    float *h_bary = (float *)malloc(pix * 3 * sizeof(float));
    cuMemcpyDtoH(h_fidx, d_fidx, pix * sizeof(int));
    cuMemcpyDtoH(h_bary, d_bary, pix * 3 * sizeof(float));

    /* Stats */
    int covered = 0, fmax = 0;
    for (size_t i = 0; i < pix; i++) {
        if (h_fidx[i] > 0) { covered++; if (h_fidx[i] > fmax) fmax = h_fidx[i]; }
    }
    fprintf(stderr, "Coverage: %d / %zu (%.1f%%), max face = %d\n",
            covered, pix, 100.0 * covered / (double)pix, fmax);

    char path[1024];
    int sh2[2] = {res, res};
    int sh3[3] = {res, res, 3};
    int sh4[2] = {m.n_verts, 4};
    snprintf(path, sizeof(path), "%s_findices.npy", prefix);
    write_npy(path, "<i4", sh2, 2, h_fidx, sizeof(int));
    fprintf(stderr, "Wrote %s\n", path);
    snprintf(path, sizeof(path), "%s_bary.npy", prefix);
    write_npy(path, "<f4", sh3, 3, h_bary, sizeof(float));
    fprintf(stderr, "Wrote %s\n", path);
    snprintf(path, sizeof(path), "%s_clip.npy", prefix);
    write_npy(path, "<f4", sh4, 2, clip, sizeof(float));
    fprintf(stderr, "Wrote %s\n", path);

    cuMemFree(d_V); cuMemFree(d_F); cuMemFree(d_zbuf);
    cuMemFree(d_fidx); cuMemFree(d_bary);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(h_fidx); free(h_bary); free(clip);
    free(m.pos); free(m.tri);
    return 0;
}
