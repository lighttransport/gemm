/*
 * paint_stage_vae.c - VAE decode stage TU for the top-level paint pipeline.
 *
 * Owns the only TU that includes cuda_paint_vae_runner.h, so its file-local
 * helpers (k_conv, load_resblock, upload_st, ...) don't collide with sibling
 * stage runners. Exposes the opaque API declared in paint_stages.h.
 */

#include "cuda_paint_vae_runner.h"
#include "paint_stages.h"

#include "../cuda_kernels_common.h"
#include "../cuda_fp8_mma_kernels.h"
#include "../gemm/cuda_gemm_ptx_kernels.h"  /* k_gemm_bf16_v7_src */
#include "paint_fp8_v7_kernels.h"           /* k_paint_fp8_v7_src */

struct paint_stage_vae {
    pvae_kernels kk;
    pvae_decoder dec;
    pvae_encoder enc;
    int enc_loaded;
};

paint_stage_vae *paint_stage_vae_create(CUdevice dev, const char *vae_path) {
    paint_stage_vae *s = (paint_stage_vae *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    /* Assemble TC prelude (common + mma + bf16 + fp8v7) ahead of vae kernels —
     * mirrors paint_stage_unet.c so VAE convs can dispatch through im2col +
     * gemm_fp8_v7 / gemm_bf16_v7. */
    const char *common_close = "\n} /* close cuda_kernels_common_src extern C */\n";
    const char *mma_open  =
        "\nextern \"C\" {\n"
        "__device__ __constant__ unsigned short d_fp8_to_bf16_lut[256];\n"
        "__device__ __forceinline__ float to_bf16(float f) {\n"
        "    unsigned int b; memcpy(&b, &f, 4);\n"
        "    if (((b >> 23) & 0xFF) == 0xFF && (b & 0x7FFFFF)) {\n"
        "        unsigned int qn = 0x7FC00000u; float r; memcpy(&r, &qn, 4); return r;\n"
        "    }\n"
        "    unsigned int rnd = 0x7FFFu + ((b >> 16) & 1u);\n"
        "    b = (b + rnd) & 0xFFFF0000u;\n"
        "    float out; memcpy(&out, &b, 4); return out;\n"
        "}\n";
    const char *mma_close = "\n} /* extern C (fp8_mma_kernels) */\n";
    const char *bf16_open  = "\nextern \"C\" {\n";
    const char *bf16_quant =
        "__global__ void quant_bf16(unsigned short *out, const float *in, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i >= n) return;\n"
        "    unsigned int u = __float_as_uint(in[i]);\n"
        "    unsigned int round = ((u >> 16) & 1u) + 0x7fffu;\n"
        "    out[i] = (unsigned short)((u + round) >> 16);\n"
        "}\n"
        "__global__ void add_bias_inplace_f32(\n"
        "    float *Y, const float *bias, int rows, int cols, int has_bias) {\n"
        "    if (!has_bias) return;\n"
        "    int j = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    int i = blockIdx.y;\n"
        "    if (j >= cols || i >= rows) return;\n"
        "    Y[(size_t)i * cols + j] += bias[j];\n"
        "}\n";
    const char *bf16_close = "\n} /* extern C (bf16 v7 + helpers) */\n";
    const char *fp8v7_open  = "\nextern \"C\" {\n";
    const char *fp8v7_close = "\n} /* extern C (fp8 v7 fused) */\n";
    size_t l_common = strlen(cuda_kernels_common_src);
    size_t l_cc     = strlen(common_close);
    size_t l_open   = strlen(mma_open);
    size_t l_mma    = strlen(fp8_mma_kernels_src);
    size_t l_close  = strlen(mma_close);
    size_t l_bopen  = strlen(bf16_open);
    size_t l_bq     = strlen(bf16_quant);
    size_t l_bv7    = strlen(k_gemm_bf16_v7_src);
    size_t l_bclose = strlen(bf16_close);
    size_t l_f8open = strlen(fp8v7_open);
    size_t l_f8src  = strlen(k_paint_fp8_v7_src);
    size_t l_f8close= strlen(fp8v7_close);
    size_t l_vae    = strlen(cuda_paint_vae_kernels_src);
    char *src = (char *)malloc(l_common + l_cc + l_open + l_mma + l_close +
                                l_bopen + l_bq + l_bv7 + l_bclose +
                                l_f8open + l_f8src + l_f8close +
                                l_vae + 1);
    char *p = src;
    memcpy(p, cuda_kernels_common_src, l_common); p += l_common;
    memcpy(p, common_close, l_cc);                p += l_cc;
    memcpy(p, mma_open, l_open);                  p += l_open;
    memcpy(p, fp8_mma_kernels_src, l_mma);        p += l_mma;
    memcpy(p, mma_close, l_close);                p += l_close;
    memcpy(p, bf16_open, l_bopen);                p += l_bopen;
    memcpy(p, bf16_quant, l_bq);                  p += l_bq;
    memcpy(p, k_gemm_bf16_v7_src, l_bv7);         p += l_bv7;
    memcpy(p, bf16_close, l_bclose);              p += l_bclose;
    memcpy(p, fp8v7_open, l_f8open);              p += l_f8open;
    memcpy(p, k_paint_fp8_v7_src, l_f8src);       p += l_f8src;
    memcpy(p, fp8v7_close, l_f8close);            p += l_f8close;
    memcpy(p, cuda_paint_vae_kernels_src, l_vae); p += l_vae;
    *p = '\0';
    int sm = cu_compile_kernels(&s->kk.mod, dev, src,
                                 "hy3d_paint_vae", 1, "HY3D-PAINT-VAE");
    free(src);
    if (sm < 0) { free(s); return NULL; }
    cuModuleGetFunction(&s->kk.f_gn,        s->kk.mod, "vae_groupnorm_f32");
    cuModuleGetFunction(&s->kk.f_conv,      s->kk.mod, "vae_conv2d_f32");
    cuModuleGetFunction(&s->kk.f_conv_down, s->kk.mod, "vae_conv2d_down_f32");
    cuModuleGetFunction(&s->kk.f_up2x,      s->kk.mod, "vae_upsample2x_f32");
    cuModuleGetFunction(&s->kk.f_add,       s->kk.mod, "vae_add_f32");
    cuModuleGetFunction(&s->kk.f_attn,      s->kk.mod, "vae_attn_f32");
    cuModuleGetFunction(&s->kk.f_chw_nc,    s->kk.mod, "vae_chw_to_nc_f32");
    cuModuleGetFunction(&s->kk.f_nc_chw,    s->kk.mod, "vae_nc_to_chw_f32");
    /* TC dispatch (optional; nullified if not present). */
    if (cuModuleGetFunction(&s->kk.f_im2col_3x3_p1,    s->kk.mod, "pvae_im2col_3x3_p1_f32")    != CUDA_SUCCESS) s->kk.f_im2col_3x3_p1    = NULL;
    if (cuModuleGetFunction(&s->kk.f_im2col_3x3_p1_s2, s->kk.mod, "pvae_im2col_3x3_p1_s2_f32") != CUDA_SUCCESS) s->kk.f_im2col_3x3_p1_s2 = NULL;
    if (cuModuleGetFunction(&s->kk.f_t_hwc_chw,        s->kk.mod, "pvae_t_hwc_to_chw_f32")     != CUDA_SUCCESS) s->kk.f_t_hwc_chw        = NULL;
    if (cuModuleGetFunction(&s->kk.f_gemm_fp8_mt4,     s->kk.mod, "gemm_bf16_pipe_mt4_scaled_f32") != CUDA_SUCCESS) s->kk.f_gemm_fp8_mt4   = NULL;
    if (cuModuleGetFunction(&s->kk.f_gemm_fp8,         s->kk.mod, "gemm_bf16_pipe_scaled_f32")     != CUDA_SUCCESS) s->kk.f_gemm_fp8       = NULL;
    if (cuModuleGetFunction(&s->kk.f_reduce_max_abs,   s->kk.mod, "reduce_max_abs_f32")            != CUDA_SUCCESS) s->kk.f_reduce_max_abs = NULL;
    if (cuModuleGetFunction(&s->kk.f_quantize_fp8,     s->kk.mod, "quantize_to_fp8_e4m3")          != CUDA_SUCCESS) s->kk.f_quantize_fp8   = NULL;
    if (cuModuleGetFunction(&s->kk.f_quant_bf16,       s->kk.mod, "quant_bf16")                    != CUDA_SUCCESS) s->kk.f_quant_bf16     = NULL;
    if (cuModuleGetFunction(&s->kk.f_add_bias_f32,     s->kk.mod, "add_bias_inplace_f32")          != CUDA_SUCCESS) s->kk.f_add_bias_f32   = NULL;
    if (cuModuleGetFunction(&s->kk.f_gemm_bf16_v7,     s->kk.mod, "gemm_bf16_v7")                  != CUDA_SUCCESS) s->kk.f_gemm_bf16_v7   = NULL;
    if (s->kk.f_gemm_bf16_v7) {
        cuFuncSetAttribute(s->kk.f_gemm_bf16_v7,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           24 * 1024);
    }
    if (cuModuleGetFunction(&s->kk.f_gemm_fp8_v7_fused, s->kk.mod, "gemm_fp8_v7_fused") != CUDA_SUCCESS)
        s->kk.f_gemm_fp8_v7_fused = NULL;

    st_context *st = safetensors_open(vae_path);
    if (!st) {
        fprintf(stderr, "ERROR: cannot open %s\n", vae_path);
        cuModuleUnload(s->kk.mod); free(s); return NULL;
    }
    load_decoder(st, &s->dec);
    load_encoder(st, &s->enc);
    s->enc_loaded = 1;
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

void paint_stage_vae_encode(paint_stage_vae *s,
                             CUdeviceptr d_img, int img_h, int img_w,
                             CUdeviceptr d_lat,
                             CUdeviceptr d_a, CUdeviceptr d_b,
                             CUdeviceptr d_t1, CUdeviceptr d_t2,
                             CUdeviceptr d_qnc, CUdeviceptr d_knc,
                             CUdeviceptr d_vnc, CUdeviceptr d_ync) {
    encode(&s->kk, &s->enc, d_img, img_h, img_w, d_lat,
           d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync);
}

void paint_stage_vae_destroy(paint_stage_vae *s) {
    if (!s) return;
    if (s->kk.mod) cuModuleUnload(s->kk.mod);
    free(s);
}
