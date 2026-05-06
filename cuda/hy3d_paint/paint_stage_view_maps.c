/*
 * paint_stage_view_maps.c - per-view normal+position render TU.
 *
 * Lifts test_view_maps.c's render loop into the per-stage opaque API
 * declared in paint_stages.h. Owns the raster module + 5 kernels and the
 * mesh-conditioned device buffers. Shape and numerics are unchanged: 6
 * views (azim {0,90,180,270,0,180}, elev {0,0,0,0,+90,-90}), face-shader
 * normals, per-vertex linearly-interpolated position, white background.
 *
 * Camera math lives here because it depends on the per-view azim/elev pair
 * picked by the stage; the orchestrator never sees mat4_*.
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_raster_kernels.h"
#include "paint_stages.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HY3D_CAMERA_DISTANCE 1.45f
#define HY3D_ORTHO_SCALE     1.2f
#define HY3D_SCALE_FACTOR    1.15f
#define HY3D_N_VIEWS         6

static const float k_azims_deg[HY3D_N_VIEWS] = {  0.f, 90.f, 180.f, 270.f,   0.f, 180.f };
static const float k_elevs_deg[HY3D_N_VIEWS] = {  0.f,  0.f,   0.f,   0.f,  90.f, -90.f };

struct paint_stage_view_maps {
    int res;
    /* Module + kernels */
    CUmodule mod;
    CUfunction f_raster, f_bary, f_interp, f_facenrm, f_lookup;

    /* Per-mesh resident buffers */
    int n_verts, n_tris;
    float *h_pos;          /* transformed mesh */
    CUdeviceptr d_F;       /* triangles */
    CUdeviceptr d_Vw;      /* world-space verts (for face-normal calc) */
    CUdeviceptr d_FN;      /* face normals */
    CUdeviceptr d_P;       /* per-vertex position attribute */
    CUdeviceptr d_bg;      /* white bg */

    /* Per-view transient device buffers */
    CUdeviceptr d_V, d_zbuf, d_fidx, d_bary, d_nrm_img, d_pos_img;

    /* Per-view scratch */
    uint64_t *zbuf_init;
    float *clip;
    int   *h_fidx;
    float *h_nrm;
    float *h_pos_img;
};

static void mat4_zero(float *m) { memset(m, 0, 16 * sizeof(float)); }

static void mat4_hy3d_ortho(float *m, float l, float r, float b, float t,
                              float n, float fr) {
    mat4_zero(m);
    m[0*4 + 0] =  2.f / (r - l);
    m[1*4 + 1] =  2.f / (t - b);
    m[2*4 + 2] = -2.f / (fr - n);
    m[3*4 + 0] = -(r + l) / (r - l);
    m[3*4 + 1] = -(t + b) / (t - b);
    m[3*4 + 2] = -(fr + n) / (fr - n);
    m[3*4 + 3] =  1.f;
}

/* see test_view_maps.c::mat4_hy3d_view — done in double for elev=±90 stability */
static void mat4_hy3d_view(float *out, float elev_deg, float azim_deg, float dist) {
    double e_rad = -((double)elev_deg) * 3.141592653589793 / 180.0;
    double a_rad = ((double)azim_deg + 90.0) * 3.141592653589793 / 180.0;
    double ce = cos(e_rad), se = sin(e_rad);
    double ca = cos(a_rad), sa = sin(a_rad);
    double cam[3] = { dist * ce * ca, dist * ce * sa, dist * se };
    double lookat[3] = { -cam[0], -cam[1], -cam[2] };
    double ll = sqrt(lookat[0]*lookat[0] + lookat[1]*lookat[1] + lookat[2]*lookat[2]);
    lookat[0]/=ll; lookat[1]/=ll; lookat[2]/=ll;
    double up[3] = { 0.0, 0.0, 1.0 };
    double right[3] = {
        lookat[1]*up[2] - lookat[2]*up[1],
        lookat[2]*up[0] - lookat[0]*up[2],
        lookat[0]*up[1] - lookat[1]*up[0],
    };
    double rl = sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    right[0]/=rl; right[1]/=rl; right[2]/=rl;
    up[0] = right[1]*lookat[2] - right[2]*lookat[1];
    up[1] = right[2]*lookat[0] - right[0]*lookat[2];
    up[2] = right[0]*lookat[1] - right[1]*lookat[0];
    double ul = sqrt(up[0]*up[0] + up[1]*up[1] + up[2]*up[2]);
    up[0]/=ul; up[1]/=ul; up[2]/=ul;
    mat4_zero(out);
    out[0*4+0]=(float)right[0];  out[1*4+0]=(float)right[1];  out[2*4+0]=(float)right[2];
    out[0*4+1]=(float)up[0];     out[1*4+1]=(float)up[1];     out[2*4+1]=(float)up[2];
    out[0*4+2]=(float)(-lookat[0]); out[1*4+2]=(float)(-lookat[1]); out[2*4+2]=(float)(-lookat[2]);
    out[3*4+0]=(float)(-(right[0]*cam[0]+right[1]*cam[1]+right[2]*cam[2]));
    out[3*4+1]=(float)(-(up[0]*cam[0]+up[1]*cam[1]+up[2]*cam[2]));
    out[3*4+2]=(float)(-(-lookat[0]*cam[0]-lookat[1]*cam[1]-lookat[2]*cam[2]));
    out[3*4+3]=1.f;
}

static void mat4_mul(float *out, const float *a, const float *b) {
    float r[16];
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            float s = 0.f;
            for (int k = 0; k < 4; k++) s += a[k*4+rr] * b[c*4+k];
            r[c*4+rr] = s;
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

paint_stage_view_maps *paint_stage_view_maps_create(CUdevice dev, int res) {
    paint_stage_view_maps *s = (paint_stage_view_maps *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->res = res;
    int sm = cu_compile_kernels(&s->mod, dev, cuda_paint_raster_kernels_src,
                                 "hy3d_paint_raster", 1, "HY3D-PAINT");
    if (sm < 0) { free(s); return NULL; }
    cuModuleGetFunction(&s->f_raster,  s->mod, "rasterize_faces_f32");
    cuModuleGetFunction(&s->f_bary,    s->mod, "resolve_bary_f32");
    cuModuleGetFunction(&s->f_interp,  s->mod, "interpolate_attr_f32");
    cuModuleGetFunction(&s->f_facenrm, s->mod, "compute_face_normals_f32");
    cuModuleGetFunction(&s->f_lookup,  s->mod, "lookup_face_attr_f32");

    size_t pix = (size_t)res * res;
    cuMemAlloc(&s->d_zbuf,    pix * sizeof(uint64_t));
    cuMemAlloc(&s->d_fidx,    pix * sizeof(int32_t));
    cuMemAlloc(&s->d_bary,    pix * 3 * sizeof(float));
    cuMemAlloc(&s->d_nrm_img, pix * 3 * sizeof(float));
    cuMemAlloc(&s->d_pos_img, pix * 3 * sizeof(float));
    cuMemAlloc(&s->d_bg,      3 * sizeof(float));
    float bg_white[3] = { 1.f, 1.f, 1.f };
    cuMemcpyHtoD(s->d_bg, bg_white, 3 * sizeof(float));

    s->zbuf_init = (uint64_t *)malloc(pix * sizeof(uint64_t));
    uint64_t sentinel = (uint64_t)2147483647ULL * 2147483647ULL + 2147483646ULL;
    for (size_t i = 0; i < pix; i++) s->zbuf_init[i] = sentinel;

    s->h_fidx   = (int *)  malloc(pix * sizeof(int));
    s->h_nrm    = (float *)malloc(pix * 3 * sizeof(float));
    s->h_pos_img= (float *)malloc(pix * 3 * sizeof(float));
    return s;
}

void paint_stage_view_maps_set_mesh(paint_stage_view_maps *s,
                                     const float *vtx_pos, int n_verts,
                                     const int *tri_idx, int n_tris) {
    /* Re-allocate per-mesh buffers */
    if (s->h_pos) { free(s->h_pos); s->h_pos = NULL; }
    if (s->d_F)  { cuMemFree(s->d_F);  s->d_F  = 0; }
    if (s->d_Vw) { cuMemFree(s->d_Vw); s->d_Vw = 0; }
    if (s->d_FN) { cuMemFree(s->d_FN); s->d_FN = 0; }
    if (s->d_P)  { cuMemFree(s->d_P);  s->d_P  = 0; }
    if (s->d_V)  { cuMemFree(s->d_V);  s->d_V  = 0; }

    s->n_verts = n_verts;
    s->n_tris  = n_tris;
    s->h_pos = (float *)malloc((size_t)n_verts * 3 * sizeof(float));
    memcpy(s->h_pos, vtx_pos, (size_t)n_verts * 3 * sizeof(float));

    /* set_mesh transform: negate X+Y, swap Y/Z. */
    for (int i = 0; i < n_verts; i++) {
        float x = s->h_pos[i*3+0], y = s->h_pos[i*3+1], z = s->h_pos[i*3+2];
        s->h_pos[i*3+0] = -x;
        s->h_pos[i*3+1] =  z;
        s->h_pos[i*3+2] = -y;
    }
    /* auto_center + scale_factor */
    float mn[3] = {s->h_pos[0], s->h_pos[1], s->h_pos[2]};
    float mx[3] = {mn[0], mn[1], mn[2]};
    for (int i = 1; i < n_verts; i++)
        for (int j = 0; j < 3; j++) {
            float v = s->h_pos[i*3+j];
            if (v < mn[j]) mn[j] = v;
            if (v > mx[j]) mx[j] = v;
        }
    float ctr[3] = { 0.5f*(mn[0]+mx[0]), 0.5f*(mn[1]+mx[1]), 0.5f*(mn[2]+mx[2]) };
    float maxd2 = 0.f;
    for (int i = 0; i < n_verts; i++) {
        float dx = s->h_pos[i*3+0] - ctr[0];
        float dy = s->h_pos[i*3+1] - ctr[1];
        float dz = s->h_pos[i*3+2] - ctr[2];
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > maxd2) maxd2 = d2;
    }
    float scale = sqrtf(maxd2) * 2.0f;
    float k = HY3D_SCALE_FACTOR / scale;
    for (int i = 0; i < n_verts; i++) {
        s->h_pos[i*3+0] = (s->h_pos[i*3+0] - ctr[0]) * k;
        s->h_pos[i*3+1] = (s->h_pos[i*3+1] - ctr[1]) * k;
        s->h_pos[i*3+2] = (s->h_pos[i*3+2] - ctr[2]) * k;
    }

    /* Per-vertex position attribute (linear in vtx coords) */
    float *pos_attr = (float *)malloc((size_t)n_verts * 3 * sizeof(float));
    for (int i = 0; i < n_verts; i++) {
        pos_attr[i*3+0] = 0.5f - s->h_pos[i*3+0] / HY3D_SCALE_FACTOR;
        pos_attr[i*3+1] = 0.5f - s->h_pos[i*3+1] / HY3D_SCALE_FACTOR;
        pos_attr[i*3+2] = 0.5f - s->h_pos[i*3+2] / HY3D_SCALE_FACTOR;
    }

    size_t f_bytes  = (size_t)n_tris  * 3 * sizeof(int);
    size_t v3_bytes = (size_t)n_verts * 3 * sizeof(float);
    size_t v4_bytes = (size_t)n_verts * 4 * sizeof(float);
    size_t fn_bytes = (size_t)n_tris  * 3 * sizeof(float);
    cuMemAlloc(&s->d_F,  f_bytes);
    cuMemAlloc(&s->d_Vw, v3_bytes);
    cuMemAlloc(&s->d_FN, fn_bytes);
    cuMemAlloc(&s->d_P,  v3_bytes);
    cuMemAlloc(&s->d_V,  v4_bytes);
    cuMemcpyHtoD(s->d_F,  tri_idx,  f_bytes);
    cuMemcpyHtoD(s->d_Vw, s->h_pos, v3_bytes);
    cuMemcpyHtoD(s->d_P,  pos_attr, v3_bytes);
    free(pos_attr);

    /* Compute world-space face normals once. */
    int nf = n_tris;
    void *args[] = { &s->d_Vw, &s->d_F, &s->d_FN, &nf };
    unsigned grid = (unsigned)((nf + 255) / 256);
    cuLaunchKernel(s->f_facenrm, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);

    s->clip = (float *)realloc(s->clip, v4_bytes);
}

void paint_stage_view_maps_render(paint_stage_view_maps *s,
                                   CUdeviceptr d_normal_out,
                                   CUdeviceptr d_position_out) {
    const int res = s->res;
    const size_t pix = (size_t)res * res;
    const size_t per_view_bytes = pix * 3 * sizeof(float);

    for (int v = 0; v < HY3D_N_VIEWS; v++) {
        float proj[16], view[16], mvp[16];
        mat4_hy3d_ortho(proj,
                          -HY3D_ORTHO_SCALE * 0.5f, HY3D_ORTHO_SCALE * 0.5f,
                          -HY3D_ORTHO_SCALE * 0.5f, HY3D_ORTHO_SCALE * 0.5f,
                          0.1f, 100.f);
        mat4_hy3d_view(view, k_elevs_deg[v], k_azims_deg[v], HY3D_CAMERA_DISTANCE);
        mat4_mul(mvp, proj, view);
        apply_mvp(s->h_pos, s->n_verts, mvp, s->clip);
        cuMemcpyHtoD(s->d_V, s->clip, (size_t)s->n_verts * 4 * sizeof(float));
        cuMemcpyHtoD(s->d_zbuf, s->zbuf_init, pix * sizeof(uint64_t));

        int nf = s->n_tris, W = res, H = res, C = 3;
        {
            void *args[] = { &s->d_V, &s->d_F, &s->d_zbuf, &nf, &W, &H };
            unsigned grid = (unsigned)((nf + 255) / 256);
            cuLaunchKernel(s->f_raster, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        {
            void *args[] = { &s->d_V, &s->d_F, &s->d_zbuf, &s->d_fidx, &s->d_bary, &W, &H };
            unsigned grid = (unsigned)((pix + 255) / 256);
            cuLaunchKernel(s->f_bary, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        {
            void *args[] = { &s->d_FN, &s->d_fidx, &s->d_bg, &s->d_nrm_img, &W, &H, &C };
            unsigned grid = (unsigned)((pix + 255) / 256);
            cuLaunchKernel(s->f_lookup, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        {
            void *args[] = { &s->d_P, &s->d_F, &s->d_fidx, &s->d_bary, &s->d_bg,
                             &s->d_pos_img, &W, &H, &C };
            unsigned grid = (unsigned)((pix + 255) / 256);
            cuLaunchKernel(s->f_interp, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        cuCtxSynchronize();

        /* Apply (n+1)*0.5 to covered pixels on host (cheap; matches PyTorch
         * pipeline ordering exactly: lookup wrote raw face normals or bg=1.0
         * for empty pixels — for those we leave them at 1.0). */
        cuMemcpyDtoH(s->h_nrm,    s->d_nrm_img, per_view_bytes);
        cuMemcpyDtoH(s->h_fidx,   s->d_fidx,    pix * sizeof(int));
        for (size_t i = 0; i < pix; i++) {
            if (s->h_fidx[i] > 0) {
                s->h_nrm[i*3+0] = (s->h_nrm[i*3+0] + 1.f) * 0.5f;
                s->h_nrm[i*3+1] = (s->h_nrm[i*3+1] + 1.f) * 0.5f;
                s->h_nrm[i*3+2] = (s->h_nrm[i*3+2] + 1.f) * 0.5f;
            }
        }
        if (d_normal_out)
            cuMemcpyHtoD(d_normal_out + (CUdeviceptr)v * per_view_bytes,
                         s->h_nrm, per_view_bytes);
        if (d_position_out)
            cuMemcpyDtoD(d_position_out + (CUdeviceptr)v * per_view_bytes,
                         s->d_pos_img, per_view_bytes);
    }
}

void paint_stage_view_maps_destroy(paint_stage_view_maps *s) {
    if (!s) return;
    if (s->d_F)  cuMemFree(s->d_F);
    if (s->d_Vw) cuMemFree(s->d_Vw);
    if (s->d_FN) cuMemFree(s->d_FN);
    if (s->d_P)  cuMemFree(s->d_P);
    if (s->d_V)  cuMemFree(s->d_V);
    if (s->d_bg) cuMemFree(s->d_bg);
    if (s->d_zbuf)    cuMemFree(s->d_zbuf);
    if (s->d_fidx)    cuMemFree(s->d_fidx);
    if (s->d_bary)    cuMemFree(s->d_bary);
    if (s->d_nrm_img) cuMemFree(s->d_nrm_img);
    if (s->d_pos_img) cuMemFree(s->d_pos_img);
    free(s->h_pos); free(s->clip); free(s->zbuf_init);
    free(s->h_fidx); free(s->h_nrm); free(s->h_pos_img);
    if (s->mod) cuModuleUnload(s->mod);
    free(s);
}
