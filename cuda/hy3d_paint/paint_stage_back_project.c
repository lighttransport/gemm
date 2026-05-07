/*
 * paint_stage_back_project.c - per-view back-project + GPU bake-blend stage.
 *
 * Lifts the multi-view path of test_paint_back_project_e2e.c (--gpu-bake)
 * into the opaque-API per-stage TU pattern. Owns its own CUmodule built
 * from cuda_paint_raster_kernels.h; only allocates per-view scratch once.
 *
 * Constants (match MeshRender.fast_bake_texture):
 *   depth_thres = 3e-3
 *   exp_w       = 6.0
 *   bake_eps    = 1e-8
 *   skip_threshold = 0.99 (painted/visible)
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_raster_kernels.h"
#include "paint_stages.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct paint_stage_back_project {
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUmodule mod;
    int owns_ctx;

    int Htex, Wtex, C;
    int Himg_max, Wimg_max;  /* current scratch capacity */

    CUfunction f_bp;
    CUfunction f_bk_count;
    CUfunction f_bk_accum;
    CUfunction f_bk_finalize;

    /* Atlas (uploaded once via set_atlas) */
    CUdeviceptr d_tex_pos;   /* [Htex*Wtex*3] f32 */
    CUdeviceptr d_tex_cov;   /* [Htex*Wtex]   i32 */

    /* Per-view scratch */
    CUdeviceptr d_image;
    CUdeviceptr d_depth;
    CUdeviceptr d_vis;
    CUdeviceptr d_cos;
    CUdeviceptr d_w2c;
    CUdeviceptr d_out_tex;   /* [Htex*Wtex*C] f32 */
    CUdeviceptr d_out_cos;   /* [Htex*Wtex]   f32 */

    /* Bake accumulator */
    CUdeviceptr d_tex_merge; /* [Htex*Wtex*C] f32 */
    CUdeviceptr d_trust;     /* [Htex*Wtex]   f32 */
    CUdeviceptr d_bake;      /* [Htex*Wtex*C] f32 (finalize output) */
    CUdeviceptr d_bake_mask; /* [Htex*Wtex]   f32 */
    CUdeviceptr d_view_sum;
    CUdeviceptr d_painted_sum;
};

typedef struct paint_stage_back_project bp_t;

static void ensure_view_scratch(bp_t *s, int Himg, int Wimg) {
    if (Himg <= s->Himg_max && Wimg <= s->Wimg_max) return;
    if (s->d_image) { cuMemFree(s->d_image); cuMemFree(s->d_depth);
                      cuMemFree(s->d_vis); cuMemFree(s->d_cos); }
    size_t img_n = (size_t)Himg * Wimg;
    cuMemAlloc(&s->d_image, img_n * s->C * sizeof(float));
    cuMemAlloc(&s->d_depth, img_n * sizeof(float));
    cuMemAlloc(&s->d_vis,   img_n * sizeof(float));
    cuMemAlloc(&s->d_cos,   img_n * sizeof(float));
    s->Himg_max = Himg; s->Wimg_max = Wimg;
}

paint_stage_back_project *paint_stage_back_project_create(CUdevice dev,
                                                           int Htex, int Wtex,
                                                           int C) {
    bp_t *s = (bp_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->dev = dev; s->Htex = Htex; s->Wtex = Wtex; s->C = C;
    if (cuCtxGetCurrent(&s->ctx) != CUDA_SUCCESS || s->ctx == NULL) {
        cuCtxCreate(&s->ctx, 0, dev);
        s->owns_ctx = 1;
    }
    cuStreamCreate(&s->stream, 0);
    if (cu_compile_kernels(&s->mod, dev, cuda_paint_raster_kernels_src,
                            "hy3d_paint_raster", 1, "HY3D-PAINT") < 0) {
        free(s); return NULL;
    }
    cuModuleGetFunction(&s->f_bp,          s->mod, "back_project_sample_f32");
    cuModuleGetFunction(&s->f_bk_count,    s->mod, "bake_blend_count_f32");
    cuModuleGetFunction(&s->f_bk_accum,    s->mod, "bake_blend_accum_f32");
    cuModuleGetFunction(&s->f_bk_finalize, s->mod, "bake_blend_finalize_f32");

    size_t tex_n = (size_t)Htex * Wtex;
    cuMemAlloc(&s->d_tex_pos,    tex_n * 3 * sizeof(float));
    cuMemAlloc(&s->d_tex_cov,    tex_n * sizeof(int));
    cuMemAlloc(&s->d_w2c,        16 * sizeof(float));
    cuMemAlloc(&s->d_out_tex,    tex_n * C * sizeof(float));
    cuMemAlloc(&s->d_out_cos,    tex_n * sizeof(float));
    cuMemAlloc(&s->d_tex_merge,  tex_n * C * sizeof(float));
    cuMemAlloc(&s->d_trust,      tex_n * sizeof(float));
    cuMemAlloc(&s->d_bake,       tex_n * C * sizeof(float));
    cuMemAlloc(&s->d_bake_mask,  tex_n * sizeof(float));
    cuMemAlloc(&s->d_view_sum,    sizeof(int));
    cuMemAlloc(&s->d_painted_sum, sizeof(int));
    return s;
}

void paint_stage_back_project_set_atlas(paint_stage_back_project *s,
                                         const float *tex_pos,
                                         const int *tex_cov) {
    size_t tex_n = (size_t)s->Htex * s->Wtex;
    cuMemcpyHtoD(s->d_tex_pos, tex_pos, tex_n * 3 * sizeof(float));
    cuMemcpyHtoD(s->d_tex_cov, tex_cov, tex_n * sizeof(int));
}

void paint_stage_back_project_begin(paint_stage_back_project *s) {
    size_t tex_n = (size_t)s->Htex * s->Wtex;
    cuMemsetD8Async(s->d_tex_merge, 0, tex_n * s->C * sizeof(float), s->stream);
    cuMemsetD8Async(s->d_trust,     0, tex_n * sizeof(float),        s->stream);
}

int paint_stage_back_project_add_view(paint_stage_back_project *s,
                                       const float *image,
                                       const float *depth,
                                       const float *visible,
                                       const float *cos_img,
                                       const float *w2c_4x4,
                                       int Himg, int Wimg,
                                       float proj00, float proj11) {
    ensure_view_scratch(s, Himg, Wimg);
    size_t tex_n = (size_t)s->Htex * s->Wtex;
    size_t img_n = (size_t)Himg * Wimg;
    int C = s->C;

    cuMemcpyHtoDAsync(s->d_image, image,    img_n * C * sizeof(float), s->stream);
    cuMemcpyHtoDAsync(s->d_depth, depth,    img_n * sizeof(float),     s->stream);
    cuMemcpyHtoDAsync(s->d_vis,   visible,  img_n * sizeof(float),     s->stream);
    cuMemcpyHtoDAsync(s->d_cos,   cos_img,  img_n * sizeof(float),     s->stream);
    cuMemcpyHtoDAsync(s->d_w2c,   w2c_4x4,  16 * sizeof(float),        s->stream);
    cuMemsetD8Async(s->d_out_tex, 0, tex_n * C * sizeof(float), s->stream);
    cuMemsetD8Async(s->d_out_cos, 0, tex_n * sizeof(float),     s->stream);

    float depth_thres = 3e-3f;
    int Htex = s->Htex, Wtex = s->Wtex, Himg_i = Himg, Wimg_i = Wimg, C_i = C;
    void *bp_args[] = {
        &s->d_tex_pos, &s->d_tex_cov, &s->d_image, &s->d_depth,
        &s->d_vis, &s->d_cos, &s->d_w2c,
        &proj00, &proj11, &depth_thres,
        &Htex, &Wtex, &Himg_i, &Wimg_i, &C_i,
        &s->d_out_tex, &s->d_out_cos
    };
    unsigned grid = (unsigned)((tex_n + 255) / 256);
    cuLaunchKernel(s->f_bp, grid, 1, 1, 256, 1, 1, 0, s->stream, bp_args, NULL);

    int N_tex = (int)tex_n, zero = 0;
    cuMemcpyHtoDAsync(s->d_view_sum,    &zero, sizeof(int), s->stream);
    cuMemcpyHtoDAsync(s->d_painted_sum, &zero, sizeof(int), s->stream);
    void *cargs[] = { &s->d_out_cos, &s->d_trust, &s->d_view_sum,
                      &s->d_painted_sum, &N_tex };
    cuLaunchKernel(s->f_bk_count, grid, 1, 1, 256, 1, 1, 0, s->stream, cargs, NULL);
    cuStreamSynchronize(s->stream);

    int view_sum = 0, painted_sum = 0;
    cuMemcpyDtoH(&view_sum,    s->d_view_sum,    sizeof(int));
    cuMemcpyDtoH(&painted_sum, s->d_painted_sum, sizeof(int));
    if (view_sum > 0 && (double)painted_sum / view_sum > 0.99) return 1;

    float exp_w = 6.0f;
    void *aargs[] = { &s->d_out_tex, &s->d_out_cos, &exp_w,
                      &s->d_tex_merge, &s->d_trust, &N_tex, &C_i };
    cuLaunchKernel(s->f_bk_accum, grid, 1, 1, 256, 1, 1, 0, s->stream, aargs, NULL);
    return 0;
}

void paint_stage_back_project_finalize(paint_stage_back_project *s,
                                        float *out_bake, float *out_mask) {
    size_t tex_n = (size_t)s->Htex * s->Wtex;
    int N_tex = (int)tex_n, C_i = s->C;
    float bake_eps = 1e-8f;
    void *fargs[] = { &s->d_tex_merge, &s->d_trust, &bake_eps,
                      &s->d_bake, &s->d_bake_mask, &N_tex, &C_i };
    unsigned grid = (unsigned)((tex_n + 255) / 256);
    cuLaunchKernel(s->f_bk_finalize, grid, 1, 1, 256, 1, 1, 0, s->stream, fargs, NULL);
    cuStreamSynchronize(s->stream);
    if (out_bake) cuMemcpyDtoH(out_bake, s->d_bake, tex_n * s->C * sizeof(float));
    if (out_mask) cuMemcpyDtoH(out_mask, s->d_bake_mask, tex_n * sizeof(float));
}

void paint_stage_back_project_destroy(paint_stage_back_project *s) {
    if (!s) return;
    if (s->d_tex_pos)    cuMemFree(s->d_tex_pos);
    if (s->d_tex_cov)    cuMemFree(s->d_tex_cov);
    if (s->d_image)      cuMemFree(s->d_image);
    if (s->d_depth)      cuMemFree(s->d_depth);
    if (s->d_vis)        cuMemFree(s->d_vis);
    if (s->d_cos)        cuMemFree(s->d_cos);
    if (s->d_w2c)        cuMemFree(s->d_w2c);
    if (s->d_out_tex)    cuMemFree(s->d_out_tex);
    if (s->d_out_cos)    cuMemFree(s->d_out_cos);
    if (s->d_tex_merge)  cuMemFree(s->d_tex_merge);
    if (s->d_trust)      cuMemFree(s->d_trust);
    if (s->d_bake)       cuMemFree(s->d_bake);
    if (s->d_bake_mask)  cuMemFree(s->d_bake_mask);
    if (s->d_view_sum)   cuMemFree(s->d_view_sum);
    if (s->d_painted_sum)cuMemFree(s->d_painted_sum);
    if (s->mod)    cuModuleUnload(s->mod);
    if (s->stream) cuStreamDestroy(s->stream);
    if (s->owns_ctx && s->ctx) cuCtxDestroy(s->ctx);
    free(s);
}
