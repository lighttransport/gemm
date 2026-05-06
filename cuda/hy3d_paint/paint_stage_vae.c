/*
 * paint_stage_vae.c - VAE decode stage TU for the top-level paint pipeline.
 *
 * Owns the only TU that includes cuda_paint_vae_runner.h, so its file-local
 * helpers (k_conv, load_resblock, upload_st, ...) don't collide with sibling
 * stage runners. Exposes the opaque API declared in paint_stages.h.
 */

#include "cuda_paint_vae_runner.h"
#include "paint_stages.h"

struct paint_stage_vae {
    pvae_kernels kk;
    pvae_decoder dec;
};

paint_stage_vae *paint_stage_vae_create(CUdevice dev, const char *vae_path) {
    paint_stage_vae *s = (paint_stage_vae *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    int sm = cu_compile_kernels(&s->kk.mod, dev,
                                 cuda_paint_vae_kernels_src,
                                 "hy3d_paint_vae", 1, "HY3D-PAINT-VAE");
    if (sm < 0) { free(s); return NULL; }
    cuModuleGetFunction(&s->kk.f_gn,        s->kk.mod, "vae_groupnorm_f32");
    cuModuleGetFunction(&s->kk.f_conv,      s->kk.mod, "vae_conv2d_f32");
    cuModuleGetFunction(&s->kk.f_conv_down, s->kk.mod, "vae_conv2d_down_f32");
    cuModuleGetFunction(&s->kk.f_up2x,      s->kk.mod, "vae_upsample2x_f32");
    cuModuleGetFunction(&s->kk.f_add,       s->kk.mod, "vae_add_f32");
    cuModuleGetFunction(&s->kk.f_attn,      s->kk.mod, "vae_attn_f32");
    cuModuleGetFunction(&s->kk.f_chw_nc,    s->kk.mod, "vae_chw_to_nc_f32");
    cuModuleGetFunction(&s->kk.f_nc_chw,    s->kk.mod, "vae_nc_to_chw_f32");

    st_context *st = safetensors_open(vae_path);
    if (!st) {
        fprintf(stderr, "ERROR: cannot open %s\n", vae_path);
        cuModuleUnload(s->kk.mod); free(s); return NULL;
    }
    load_decoder(st, &s->dec);
    safetensors_close(st);
    return s;
}

void paint_stage_vae_decode(paint_stage_vae *s,
                             CUdeviceptr d_lat, int lat_h, int lat_w,
                             CUdeviceptr d_rgb,
                             CUdeviceptr d_a, CUdeviceptr d_b,
                             CUdeviceptr d_t1, CUdeviceptr d_t2,
                             CUdeviceptr d_qnc, CUdeviceptr d_knc,
                             CUdeviceptr d_vnc, CUdeviceptr d_ync) {
    decode(&s->kk, &s->dec, d_lat, lat_h, lat_w, d_rgb,
           d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync);
}

void paint_stage_vae_destroy(paint_stage_vae *s) {
    if (!s) return;
    if (s->kk.mod) cuModuleUnload(s->kk.mod);
    free(s);
}
