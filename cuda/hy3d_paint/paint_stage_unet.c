/*
 * paint_stage_unet.c - dual-stream UNet UniPC step driver TU.
 *
 * Lifts the out_loop block from test_paint_unet.c into a reusable opaque API
 * (declared in paint_stages.h). Sole TU including cuda_paint_unet_runner.h so
 * its file-local helpers don't collide with other stage runners.
 *
 * The orchestrator calls:
 *   create -> set_conditioning -> run_dual (once) -> run_step (per UniPC step)
 *
 * No correctness changes vs test_paint_unet --stage out_loop: same kernel
 * sequence, same device-buffer layout, same dual-stream cache semantics.
 * The only differences are (a) conditioning comes from caller-supplied host
 * tensors instead of pyref .npy files, and (b) per-step pyref validation is
 * dropped (the orchestrator drives validation through its own checks).
 */

#include "cuda_paint_unet_runner.h"
#include "paint_stages.h"

#include "../cuda_kernels_common.h"
#include "../cuda_fp8_mma_kernels.h"
#include "../gemm/cuda_gemm_ptx_kernels.h"  /* k_gemm_bf16_v7_src */
#include "paint_fp8_v7_kernels.h"           /* k_paint_fp8_v7_src (fused FP8 v7) */

#include <stdint.h>

#define MAX_N_BLOCKS 16

struct paint_stage_unet {
    paint_unet_config cfg;

    /* Derived dims */
    int Beff_main;          /* B_outer * N_pbr * N_gen */
    int Beff_dual;          /* B_outer * N_ref */
    size_t per_view;        /* 4 * H0 * W0 */
    size_t per_in_main;     /* IC_main * H0 * W0 = 12 * H0 * W0 */
    size_t txt_per;         /* M_text * cross_dim */
    size_t x_n;             /* Beff_main * per_view (latent element count) */

    /* Kernels + module */
    pu_kernels kk;

    /* Conditioning host scratch (rebuilt by set_conditioning, reused by
     * run_step's input pack). */
    float *packed_main;     /* [Beff_main, 12, H0, W0] - sample slot rewritten per step */
    float *packed_dual;     /* [Beff_dual, 4, H0, W0]  - ref latents */
    float *text_tiled_main; /* [Beff_main, M_text, cross_dim] */

    /* Time-embedding linear weights */
    CUdeviceptr l1_w, l1_b, l2_w, l2_b;          /* main */
    CUdeviceptr l1_wd, l1_bd, l2_wd, l2_bd;      /* dual */

    /* Pre-tiled text inputs */
    CUdeviceptr d_text_dual, d_text_m;

    /* Pre-tiled DINO conditioning (set_conditioning fills d_dino) */
    CUdeviceptr d_dino;
    int M_dino;                                  /* T_dino * EXTRA = T_dino * 4 */

    /* DINO projection weights (resident; reused per set_conditioning) */
    CUdeviceptr dino_pw, dino_pb, dino_png, dino_pnb;

    /* Reference text-clip for dual stream (resident) */
    CUdeviceptr d_text_clip_ref;

    /* Timestep device buffers */
    CUdeviceptr d_ts_main, d_ts_dual;

    /* conv_in / conv_out / norm_out weights */
    CUdeviceptr cw_d, cb_d;                      /* dual conv_in */
    CUdeviceptr cw, cb;                          /* main conv_in */
    CUdeviceptr ng, nb_w, ow, ob_w;              /* main conv_norm_out + conv_out */

    /* Block weights */
    pu_down_block dbd[4];   pu_mid_block midd;   pu_up_block ubd[4]; /* dual */
    pu_down_block db[4];    pu_mid_block mid;    pu_up_block ub[4];  /* main */

    /* Per-step scratch on device */
    CUdeviceptr d_temb_in_d, d_temb_h1_d, d_temb_d;
    CUdeviceptr d_temb_in_m, d_temb_h1_m, d_temb_m;
    CUdeviceptr d_in_raw_d, d_in_raw_m;
    CUdeviceptr d_concat;
    pu_workspace ws;

    /* RA cache backing storage (g_ra_cache slots) — per CFG chunk so we can
     * cache run_dual results across all timesteps for each chunk. */
    pu_ra_slot ra_slots[PAINT_UNET_MAX_CHUNKS][MAX_N_BLOCKS];
    int chunk_dual_done[PAINT_UNET_MAX_CHUNKS];
    int active_chunk;

    /* Whether conditioning has been set */
    int cond_set;
};

paint_stage_unet *paint_stage_unet_create(CUdevice dev,
                                           const char *unet_safetensors_path,
                                           const paint_unet_config *cfg) {
    (void)dev; /* current ctx is assumed to be active */
    paint_stage_unet *s = (paint_stage_unet *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->cfg = *cfg;
    s->Beff_main   = cfg->B_outer * cfg->N_pbr * cfg->N_gen;
    s->Beff_dual   = cfg->B_outer * cfg->N_ref;
    s->per_view    = (size_t)4 * cfg->H0 * cfg->W0;
    s->per_in_main = (size_t)12 * cfg->H0 * cfg->W0;
    s->txt_per     = (size_t)cfg->M_text * cfg->cross_dim;
    s->x_n         = (size_t)s->Beff_main * s->per_view;

    /* Module compile + kernel handles. Concatenate the BF16/FP8 MMA kernel
     * source so flash_attn_bf16_hd64_xq + cast_f32_to_bf16 are reachable from
     * the same module — same trick as paint_stage_dinov2g.c. */
    /* cuda_kernels_common_src opens an extern "C" block that the runner is
     * supposed to close. The two paint kernel strings (fp8_mma and unet) each
     * open their own extern "C" wrappers, so close common's first. */
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
    /* Pure BF16 path siblings: F32->BF16 quant, in-place bias add, v7 GEMM.
     * Wrapped in their own extern "C" block so v7's `typedef bf16_raw` and
     * extern signatures don't collide with the other kernel sources. */
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
    /* Native FP8 v7 fused: own extern "C" block (declares its own typedef
     * fp8_raw and `extern "C" __global__ gemm_fp8_v7_fused`). */
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
    size_t l_unet   = strlen(cuda_paint_unet_kernels_src);
    char *src = (char *)malloc(l_common + l_cc + l_open + l_mma + l_close +
                                l_bopen + l_bq + l_bv7 + l_bclose +
                                l_f8open + l_f8src + l_f8close +
                                l_unet + 1);
    char *p = src;
    memcpy(p, cuda_kernels_common_src, l_common);  p += l_common;
    memcpy(p, common_close, l_cc);                 p += l_cc;
    memcpy(p, mma_open, l_open);                   p += l_open;
    memcpy(p, fp8_mma_kernels_src, l_mma);         p += l_mma;
    memcpy(p, mma_close, l_close);                 p += l_close;
    memcpy(p, bf16_open, l_bopen);                 p += l_bopen;
    memcpy(p, bf16_quant, l_bq);                   p += l_bq;
    memcpy(p, k_gemm_bf16_v7_src, l_bv7);          p += l_bv7;
    memcpy(p, bf16_close, l_bclose);               p += l_bclose;
    memcpy(p, fp8v7_open, l_f8open);               p += l_f8open;
    memcpy(p, k_paint_fp8_v7_src, l_f8src);        p += l_f8src;
    memcpy(p, fp8v7_close, l_f8close);             p += l_f8close;
    memcpy(p, cuda_paint_unet_kernels_src, l_unet);p += l_unet;
    *p = '\0';
    int sm = cu_compile_kernels(&s->kk.mod, dev, src,
                                 "hy3d_paint_unet", 1, "HY3D-PAINT-UNET");
    free(src);
    if (sm < 0) { free(s); return NULL; }
    /* Resolve kernel handles (mirrors test_paint_unet main()). */
    cuModuleGetFunction(&s->kk.f_tse,     s->kk.mod, "unet_timestep_embed_f32");
    cuModuleGetFunction(&s->kk.f_lin,     s->kk.mod, "unet_linear_f32");
    cuModuleGetFunction(&s->kk.f_silu,    s->kk.mod, "unet_silu_f32");
    cuModuleGetFunction(&s->kk.f_conv,    s->kk.mod, "unet_conv2d_f32");
    cuModuleGetFunction(&s->kk.f_gn,      s->kk.mod, "unet_groupnorm_f32");
    cuModuleGetFunction(&s->kk.f_addc,    s->kk.mod, "unet_add_chan_f32");
    cuModuleGetFunction(&s->kk.f_add,     s->kk.mod, "unet_add_f32");
    cuModuleGetFunction(&s->kk.f_ln,      s->kk.mod, "unet_layernorm_f32");
    cuModuleGetFunction(&s->kk.f_chw_nc,  s->kk.mod, "unet_chw_to_nc_f32");
    cuModuleGetFunction(&s->kk.f_nc_chw,  s->kk.mod, "unet_nc_to_chw_f32");
    cuModuleGetFunction(&s->kk.f_mha,     s->kk.mod, "unet_mha_f32");
    cuModuleGetFunction(&s->kk.f_geglu,   s->kk.mod, "unet_geglu_f32");
    cuModuleGetFunction(&s->kk.f_conv_s2, s->kk.mod, "unet_conv2d_stride2_f32");
    cuModuleGetFunction(&s->kk.f_up2x,    s->kk.mod, "unet_upsample2x_f32");
    cuModuleGetFunction(&s->kk.f_concat,  s->kk.mod, "unet_concat_chan_f32");
    cuModuleGetFunction(&s->kk.f_rope,    s->kk.mod, "unet_rope_apply_f32");
    cuModuleGetFunction(&s->kk.f_ra_split_v, s->kk.mod, "unet_ra_split_v_f32");

    /* FP8 GEMM dispatch (Phase 4.9.2) — optional. */
    if (cuModuleGetFunction(&s->kk.f_gemm_fp8_mt4,   s->kk.mod, "gemm_bf16_pipe_mt4_scaled_f32") != CUDA_SUCCESS) s->kk.f_gemm_fp8_mt4   = NULL;
    if (cuModuleGetFunction(&s->kk.f_gemm_fp8,       s->kk.mod, "gemm_bf16_pipe_scaled_f32")     != CUDA_SUCCESS) s->kk.f_gemm_fp8       = NULL;
    if (cuModuleGetFunction(&s->kk.f_reduce_max_abs, s->kk.mod, "reduce_max_abs_f32")            != CUDA_SUCCESS) s->kk.f_reduce_max_abs = NULL;
    if (cuModuleGetFunction(&s->kk.f_quantize_fp8,   s->kk.mod, "quantize_to_fp8_e4m3")          != CUDA_SUCCESS) s->kk.f_quantize_fp8   = NULL;
    if (cuModuleGetFunction(&s->kk.f_im2col_3x3_p1,  s->kk.mod, "unet_im2col_3x3_p1_f32")        != CUDA_SUCCESS) s->kk.f_im2col_3x3_p1  = NULL;
    if (cuModuleGetFunction(&s->kk.f_im2col_3x3_p1_s2,s->kk.mod, "unet_im2col_3x3_p1_s2_f32")     != CUDA_SUCCESS) s->kk.f_im2col_3x3_p1_s2 = NULL;
    if (cuModuleGetFunction(&s->kk.f_t_hwc_chw,      s->kk.mod, "unet_t_hwc_to_chw_f32")         != CUDA_SUCCESS) s->kk.f_t_hwc_chw      = NULL;
    /* Pure BF16 path (Phase 4.9.4). */
    if (cuModuleGetFunction(&s->kk.f_quant_bf16,     s->kk.mod, "quant_bf16")                    != CUDA_SUCCESS) s->kk.f_quant_bf16     = NULL;
    if (cuModuleGetFunction(&s->kk.f_add_bias_f32,   s->kk.mod, "add_bias_inplace_f32")          != CUDA_SUCCESS) s->kk.f_add_bias_f32   = NULL;
    if (cuModuleGetFunction(&s->kk.f_gemm_bf16_v7,   s->kk.mod, "gemm_bf16_v7")                  != CUDA_SUCCESS) s->kk.f_gemm_bf16_v7   = NULL;
    if (s->kk.f_gemm_bf16_v7) {
        /* v7 needs ~24 KiB of dynamic shared memory; opt-in via cuFuncSetAttribute. */
        cuFuncSetAttribute(s->kk.f_gemm_bf16_v7,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           24 * 1024);
    }
    if (cuModuleGetFunction(&s->kk.f_gemm_fp8_v7_fused, s->kk.mod, "gemm_fp8_v7_fused") != CUDA_SUCCESS)
        s->kk.f_gemm_fp8_v7_fused = NULL;
    if (s->kk.f_gemm_fp8_v7_fused) {
        cuFuncSetAttribute(s->kk.f_gemm_fp8_v7_fused,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           24 * 1024);
    }
    /* BF16 TC kernels (optional; missing → fall back to f_mha automatically). */
    if (cuModuleGetFunction(&s->kk.f_cast_bf16,         s->kk.mod, "cast_f32_to_bf16")        != CUDA_SUCCESS) s->kk.f_cast_bf16 = NULL;
    if (cuModuleGetFunction(&s->kk.f_attn_bf16_hd64,    s->kk.mod, "flash_attn_bf16_hd64")    != CUDA_SUCCESS) s->kk.f_attn_bf16_hd64 = NULL;
    if (cuModuleGetFunction(&s->kk.f_attn_bf16_hd64_xq, s->kk.mod, "flash_attn_bf16_hd64_xq") != CUDA_SUCCESS) s->kk.f_attn_bf16_hd64_xq = NULL;
    /* Initialize the FP8→BF16 LUT used by other MMA kernels (the attn path
     * doesn't read it, but the module global must exist when present). */
    {
        CUdeviceptr d_lut; size_t lut_sz;
        if (cuModuleGetGlobal(&d_lut, &lut_sz, s->kk.mod, "d_fp8_to_bf16_lut") == CUDA_SUCCESS &&
            lut_sz == 256 * sizeof(uint16_t)) {
            uint16_t lut[256];
            for (int i = 0; i < 256; i++) {
                int sign = (i >> 7) & 1;
                int exp  = (i >> 3) & 0xF;
                int mant = i & 0x7;
                float v;
                if (exp == 0 && mant == 0) v = 0.f;
                else if (exp == 15 && mant == 7) v = 0.f;
                else if (exp == 0) v = ((float)mant / 8.f) * (1.f / 64.f);
                else v = (1.f + (float)mant / 8.f) * exp2f((float)(exp - 7));
                if (sign) v = -v;
                uint32_t b; memcpy(&b, &v, 4);
                uint32_t rb = 0x7FFFu + ((b >> 16) & 1u);
                lut[i] = (uint16_t)((b + rb) >> 16);
            }
            cuMemcpyHtoD(d_lut, lut, sizeof(lut));
        }
    }

    /* Open weights */
    st_context *st = safetensors_open(unet_safetensors_path);
    if (!st) {
        fprintf(stderr, "ERROR: cannot open %s\n", unet_safetensors_path);
        cuModuleUnload(s->kk.mod); free(s); return NULL;
    }

    /* Time embedding linears (main + dual) */
    s->l1_w  = upload_st(st, "time_embedding.linear_1.weight");
    s->l1_b  = upload_st(st, "time_embedding.linear_1.bias");
    s->l2_w  = upload_st(st, "time_embedding.linear_2.weight");
    s->l2_b  = upload_st(st, "time_embedding.linear_2.bias");
    s->l1_wd = upload_st(st, "unet_dual.time_embedding.linear_1.weight");
    s->l1_bd = upload_st(st, "unet_dual.time_embedding.linear_1.bias");
    s->l2_wd = upload_st(st, "unet_dual.time_embedding.linear_2.weight");
    s->l2_bd = upload_st(st, "unet_dual.time_embedding.linear_2.bias");

    /* Dual conv_in + blocks */
    g_load_wp = "unet_dual.";
    s->cw_d = upload_st(st, "unet_dual.conv_in.weight");
    s->cb_d = upload_st(st, "unet_dual.conv_in.bias");
    load_down_block(st, &s->dbd[0], 0,  320,  320,  5, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_down_block(st, &s->dbd[1], 1,  320,  640, 10, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_down_block(st, &s->dbd[2], 2,  640, 1280, 20, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_down_block(st, &s->dbd[3], 3, 1280, 1280, 20, 0, 0, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_mid_block(st, &s->midd, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_up_block(st, &s->ubd[0], 0, 1280, 1280, 1280,  0, 0, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_up_block(st, &s->ubd[1], 1,  640, 1280, 1280, 20, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_up_block(st, &s->ubd[2], 2,  320,  640, 1280, 10, 1, 1, cfg->cross_dim, 0, 0, 0, 0, 0, 0);
    load_up_block(st, &s->ubd[3], 3,  320,  320,  640,  5, 1, 0, cfg->cross_dim, 0, 0, 0, 0, 0, 0);

    /* Main conv_in + blocks (full 4-attn-path config) */
    g_load_wp = "";
    s->cw = upload_st(st, "conv_in.weight");
    s->cb = upload_st(st, "conv_in.bias");
    load_down_block(st, &s->db[0], 0,  320,  320,  5, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_down_block(st, &s->db[1], 1,  320,  640, 10, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_down_block(st, &s->db[2], 2,  640, 1280, 20, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_down_block(st, &s->db[3], 3, 1280, 1280, 20, 0, 0, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_mid_block(st, &s->mid, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_up_block(st, &s->ub[0], 0, 1280, 1280, 1280,  0, 0, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_up_block(st, &s->ub[1], 1,  640, 1280, 1280, 20, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_up_block(st, &s->ub[2], 2,  320,  640, 1280, 10, 1, 1, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);
    load_up_block(st, &s->ub[3], 3,  320,  320,  640,  5, 1, 0, cfg->cross_dim, 1, 1, 1, 1, cfg->N_pbr, cfg->N_gen);

    s->ng   = upload_st(st, "conv_norm_out.weight");
    s->nb_w = upload_st(st, "conv_norm_out.bias");
    s->ow   = upload_st(st, "conv_out.weight");
    s->ob_w = upload_st(st, "conv_out.bias");

    /* DINO image proj weights — kept resident; set_conditioning runs the
     * Linear+LayerNorm with the caller-supplied dino_hidden_states. */
    s->M_dino = cfg->T_dino * 4;            /* EXTRA=4 */
    s->dino_pw  = upload_st(st, "image_proj_model_dino.proj.weight");
    s->dino_pb  = upload_st(st, "image_proj_model_dino.proj.bias");
    s->dino_png = upload_st(st, "image_proj_model_dino.norm.weight");
    s->dino_pnb = upload_st(st, "image_proj_model_dino.norm.bias");
    s->d_dino = 0; /* lazy alloc in set_conditioning */

    /* Dual-stream reference text-clip (resident, broadcast at set_conditioning). */
    s->d_text_clip_ref = upload_st(st, "learned_text_clip_ref");
    s->d_text_dual = 0;
    s->d_text_m    = 0;

    safetensors_close(st);

    /* Workspace + per-step buffers */
    const int H0 = cfg->H0, W0 = cfg->W0, M_text = cfg->M_text;
    size_t MAX_ACT  = (size_t)s->Beff_main * 1280 * H0 * W0;
    size_t MAX_CCAT = (size_t)s->Beff_main * 960  * H0 * W0;
    size_t MAX_FF_GH= (size_t)s->Beff_main * 320  * H0 * W0 * 2 * 4;
    size_t MAX_FF_H = (size_t)s->Beff_main * 320  * H0 * W0 * 4;
    size_t MAX_BNC  = (size_t)s->Beff_main * 320  * H0 * W0;
    if ((size_t)s->Beff_main * 1280 * 16 * 16 > MAX_BNC) MAX_BNC = (size_t)s->Beff_main * 1280 * 16 * 16;
    if ((size_t)s->Beff_main *  640 * 32 * 32 > MAX_BNC) MAX_BNC = (size_t)s->Beff_main *  640 * 32 * 32;
    size_t MAX_BMC = MAX_BNC;
    if ((size_t)s->Beff_main * 1280 * M_text > MAX_BMC) MAX_BMC = (size_t)s->Beff_main * 1280 * M_text;
    if ((size_t)s->Beff_main * 1280 * s->M_dino > MAX_BMC) MAX_BMC = (size_t)s->Beff_main * 1280 * s->M_dino;

    cuMemAlloc(&s->ws.d_a, MAX_ACT * sizeof(float));
    cuMemAlloc(&s->ws.d_b, MAX_ACT * sizeof(float));
    cuMemAlloc(&s->ws.d_t1, MAX_CCAT * sizeof(float));
    cuMemAlloc(&s->ws.d_t2, MAX_CCAT * sizeof(float));
    cuMemAlloc(&s->ws.d_temb_act,  s->Beff_main * 1280 * sizeof(float));
    cuMemAlloc(&s->ws.d_temb_proj, s->Beff_main * 1280 * sizeof(float));
    cuMemAlloc(&s->ws.X.d_resid, MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_nc,    MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_nc_b,  MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_norm,  MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_q,     MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_k,     MAX_BMC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_v,     MAX_BMC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_attn,  MAX_BNC * sizeof(float));
    cuMemAlloc(&s->ws.X.d_ff_gh, MAX_FF_GH * sizeof(float));
    cuMemAlloc(&s->ws.X.d_ff_h,  MAX_FF_H * sizeof(float));
    cuMemAlloc(&s->d_concat,     MAX_CCAT * sizeof(float));

    cuMemAlloc(&s->d_temb_in_d, s->Beff_dual * 320 * sizeof(float));
    cuMemAlloc(&s->d_temb_h1_d, s->Beff_dual * 1280 * sizeof(float));
    cuMemAlloc(&s->d_temb_d,    s->Beff_dual * 1280 * sizeof(float));
    cuMemAlloc(&s->d_temb_in_m, s->Beff_main * 320 * sizeof(float));
    cuMemAlloc(&s->d_temb_h1_m, s->Beff_main * 1280 * sizeof(float));
    cuMemAlloc(&s->d_temb_m,    s->Beff_main * 1280 * sizeof(float));
    cuMemAlloc(&s->d_in_raw_d,  (size_t)s->Beff_dual * 4 * H0 * W0 * sizeof(float));
    cuMemAlloc(&s->d_in_raw_m,  (size_t)s->Beff_main * 12 * H0 * W0 * sizeof(float));
    cuMemAlloc(&s->d_ts_main,   s->Beff_main * sizeof(int64_t));
    cuMemAlloc(&s->d_ts_dual,   s->Beff_dual * sizeof(int64_t));

    /* RA cache: backed by a fixed 16-slot array; runner header reads from
     * the global g_ra_cache. */
    s->active_chunk = 0;
    for (int c = 0; c < PAINT_UNET_MAX_CHUNKS; c++) s->chunk_dual_done[c] = 0;
    g_ra_cache.slots = s->ra_slots[0];
    g_ra_cache.n_slots = MAX_N_BLOCKS;
    g_ra_cache.idx = 0;
    g_ra_n_ref = cfg->N_ref;

    /* Host scratch */
    s->packed_main     = (float *)malloc(s->Beff_main * s->per_in_main * sizeof(float));
    s->packed_dual     = (float *)malloc(s->Beff_dual * s->per_view    * sizeof(float));
    s->text_tiled_main = (float *)malloc(s->Beff_main * s->txt_per     * sizeof(float));

    /* BF16 attention scratch + dispatch flag. Sized for the largest Q/K/V
     * tensor across all attention call sites: self-attn at the shallowest
     * level (B*N*dim with N=H0*W0=4096, dim=320) and cross-attn with DINO
     * tokens (B*M*dim with M=T_dino*4=1028, dim up to 1280). The .max here
     * conservatively covers both. */
    {
        const char *e = getenv("PAINT_BF16_ATTN");
        int want = (e == NULL) || (e[0] != '0');
        int have = (s->kk.f_cast_bf16 && s->kk.f_attn_bf16_hd64_xq);
        g_paint_use_bf16_attn = (want && have) ? 1 : 0;
        if (g_paint_use_bf16_attn) {
            size_t self_q = (size_t)s->Beff_main * (size_t)H0 * W0 * 320; /* deepest self N=4096*ch=320 */
            size_t cross_q = (size_t)s->Beff_main * (size_t)H0 * W0 * 320;
            size_t cross_kv_dino = (size_t)s->Beff_main * (size_t)s->M_dino * 1280;
            size_t cross_kv_text = (size_t)s->Beff_main * (size_t)M_text * 1280;
            size_t self_q_deep = (size_t)s->Beff_main * (size_t)16 * 16 * 1280;
            size_t mx = self_q;
            if (cross_q > mx) mx = cross_q;
            if (cross_kv_dino > mx) mx = cross_kv_dino;
            if (cross_kv_text > mx) mx = cross_kv_text;
            if (self_q_deep > mx) mx = self_q_deep;
            g_paint_qkv_bf16_max_elem = mx;
            cuMemAlloc(&g_paint_d_qbf, mx * sizeof(unsigned short));
            cuMemAlloc(&g_paint_d_kbf, mx * sizeof(unsigned short));
            cuMemAlloc(&g_paint_d_vbf, mx * sizeof(unsigned short));
            fprintf(stderr, "[paint_stage_unet] BF16_ATTN=1 (scratch %.1f MB / Q,K,V)\n",
                    (double)mx * 2.0 / (1024.0*1024.0));
        } else {
            fprintf(stderr, "[paint_stage_unet] BF16_ATTN=0 (env=%s have=%d)\n",
                    e ? e : "(unset)", have);
        }
    }

    /* FP8 GEMM dispatch (Phase 4.9.2): MT4 BF16-pipe with per-tensor FP8 weights.
     * Requires the three NVRTC kernels and the BF16 cast kernel; lazily prequant
     * weights on first k_linear call. Activation scratch sized to the largest
     * X tensor that flows into a linear (FF input = 4C at top resolution). */
    {
        /* Phase 4.9.2: per-tensor FP8 BF16-pipe GEMM (MT1) for all linears
         * with n_in%32==0, n_out%256==0. Default ON; PAINT_FP8_GEMM=0 falls
         * back to scalar f_lin (matches today's BF16-attn baseline bit-for-bit). */
        const char *e = getenv("PAINT_FP8_GEMM");
        int want = (e == NULL) || (e[0] != '0');
        int have = (s->kk.f_gemm_fp8 && s->kk.f_reduce_max_abs &&
                    s->kk.f_quantize_fp8);
        g_paint_use_fp8_gemm = (want && have) ? 1 : 0;
        if (g_paint_use_fp8_gemm) {
            const char *m4 = getenv("PAINT_FP8_GEMM_MT4");
            int want_mt4 = (m4 == NULL) || (m4[0] != '0');
            g_paint_use_fp8_gemm_mt4 = (want_mt4 && s->kk.f_gemm_fp8_mt4) ? 1 : 0;
            fprintf(stderr, "[paint_stage_unet] FP8_GEMM=1 (BF16-pipe %s)\n",
                    g_paint_use_fp8_gemm_mt4 ? "MT4+MT1" : "MT1");
        } else {
            fprintf(stderr, "[paint_stage_unet] FP8_GEMM=0 (env=%s have=%d)\n",
                    e ? e : "(unset)", have);
        }
    }
    {
        /* Phase 4.9.3: conv2d → im2col + FP8 GEMM. Default ON when GEMM ON
         * and the two helper kernels resolved. PAINT_FP8_CONV=0 disables. */
        const char *e = getenv("PAINT_FP8_CONV");
        int want = (e == NULL) || (e[0] != '0');
        int have = (g_paint_use_fp8_gemm && s->kk.f_im2col_3x3_p1 &&
                    s->kk.f_t_hwc_chw);
        g_paint_use_fp8_conv = (want && have) ? 1 : 0;
        if (g_paint_use_fp8_conv)
            fprintf(stderr, "[paint_stage_unet] FP8_CONV=1 (3x3-pad1 im2col+FP8-GEMM)\n");
        else
            fprintf(stderr, "[paint_stage_unet] FP8_CONV=0 (env=%s have=%d)\n",
                    e ? e : "(unset)", have);
    }
    {
        /* Phase 4.9.4: pure BF16 path for k_linear (linears only). When ON,
         * weights get an extra BF16 mirror at FP8-prequant time and k_linear
         * dispatches via quant_bf16 + gemm_bf16_v7 + add_bias_inplace_f32.
         * Default OFF — keeps the FP8 path hot and bit-identical. Set
         * PAINT_BF16_GEMM=1 to A/B against FP8. */
        const char *e = getenv("PAINT_BF16_GEMM");
        int want = (e != NULL) && (e[0] != '0');
        int have = (s->kk.f_gemm_bf16_v7 && s->kk.f_quant_bf16 &&
                    s->kk.f_add_bias_f32 && s->kk.f_cast_bf16);
        g_paint_use_bf16_gemm = (want && have) ? 1 : 0;
        if (g_paint_use_bf16_gemm)
            fprintf(stderr, "[paint_stage_unet] BF16_GEMM=1 (v7 BF16xBF16 m16n8k16, F32 accum)\n");
        else if (want)
            fprintf(stderr, "[paint_stage_unet] BF16_GEMM=0 (env=%s have=%d)\n",
                    e, have);
    }
    {
        /* Native FP8 v7 fused GEMM (PAINT_FP8_V7=1). Uses gemm_fp8_v7_fused +
         * reduce_max_abs + quantize_to_fp8_e4m3 + the existing FP8 weight
         * registry (e->w_fp8 + e->w_scale). 5060 Ti has ~2x BF16 FP8 ceiling.
         * Constraint: K%64==0 and M>=256 (v7's 4x4 panel swizzle). Default
         * OFF; non-conforming calls fall through to BF16 v7 / BF16-pipe FP8. */
        const char *e = getenv("PAINT_FP8_V7");
        int want = (e != NULL) && (e[0] != '0');
        int have = (s->kk.f_gemm_fp8_v7_fused && s->kk.f_quantize_fp8 &&
                    s->kk.f_reduce_max_abs);
        g_paint_use_fp8_v7 = (want && have) ? 1 : 0;
        if (g_paint_use_fp8_v7)
            fprintf(stderr, "[paint_stage_unet] FP8_V7=1 (native e4m3 m16n8k32, fused descale+bias)\n");
        else if (want)
            fprintf(stderr, "[paint_stage_unet] FP8_V7=0 (env=%s have=%d)\n", e, have);
    }
    {
        const char *de = getenv("PAINT_FP8_DEBUG");
        g_paint_fp8_debug = (de && de[0] != '0') ? 1 : 0;
        if (g_paint_fp8_debug)
            fprintf(stderr, "[paint_stage_unet] FP8_DEBUG=1 (per-launch sync)\n");
        const char *pe = getenv("PAINT_PROFILE");
        g_paint_profile = (pe && pe[0] != '0') ? 1 : 0;
        if (g_paint_profile) {
            fprintf(stderr, "[paint_stage_unet] PROFILE=1 (per-kernel cuEvent timing)\n");
            atexit(paint_prof_dump);
        }
    }

    return s;
}

void paint_stage_unet_set_conditioning(paint_stage_unet *s,
    const float *embeds_normal,
    const float *embeds_position,
    const float *encoder_hidden_states,
    const float *ref_latents,
    const float *dino_hidden_states) {

    const paint_unet_config *cfg = &s->cfg;
    const int N_PBR = cfg->N_pbr, N_GEN = cfg->N_gen;
    const size_t per_view = s->per_view, per_in_main = s->per_in_main;
    const size_t txt_per = s->txt_per;

    /* Pack en/ep into the conditioning slots of packed_main.
     * Layout: per-batch dst[12,H,W] with channel slots [sample | normal | position]. */
    for (int p = 0; p < N_PBR; p++)
        for (int g = 0; g < N_GEN; g++) {
            int b = p * N_GEN + g;
            float *dst = s->packed_main + (size_t)b * per_in_main;
            memcpy(dst + per_view,     embeds_normal   + (size_t)g * per_view, per_view * sizeof(float));
            memcpy(dst + 2 * per_view, embeds_position + (size_t)g * per_view, per_view * sizeof(float));
        }

    /* Dual ref latents */
    memcpy(s->packed_dual, ref_latents, (size_t)s->Beff_dual * per_view * sizeof(float));

    /* Tile text per material */
    for (int p = 0; p < N_PBR; p++)
        for (int g = 0; g < N_GEN; g++) {
            int b = p * N_GEN + g;
            memcpy(s->text_tiled_main + (size_t)b * txt_per,
                   encoder_hidden_states + (size_t)p * txt_per,
                   txt_per * sizeof(float));
        }
    if (!s->d_text_m)
        cuMemAlloc(&s->d_text_m, (size_t)s->Beff_main * txt_per * sizeof(float));
    cuMemcpyHtoD(s->d_text_m, s->text_tiled_main, (size_t)s->Beff_main * txt_per * sizeof(float));

    /* Dual text: broadcast learned_text_clip_ref [1, M_text, cross_dim] to
     * Beff_dual rows. */
    if (!s->d_text_dual)
        cuMemAlloc(&s->d_text_dual, (size_t)s->Beff_dual * txt_per * sizeof(float));
    for (int b = 0; b < s->Beff_dual; b++)
        cuMemcpyDtoD(s->d_text_dual + (CUdeviceptr)b * txt_per * sizeof(float),
                     s->d_text_clip_ref, txt_per * sizeof(float));

    /* DINO projection: Linear(C_in -> EXTRA*CTX) + LayerNorm(CTX), then
     * broadcast to Beff_main rows. */
    {
        const int CTX = 1024, EXTRA = 4;
        const int T = cfg->T_dino, Cin = cfg->C_dino_in;
        const int rows_out = T * EXTRA;
        size_t dino_per = (size_t)rows_out * CTX * sizeof(float);
        CUdeviceptr d_din, d_dlin, d_done;
        cuMemAlloc(&d_din,  (size_t)T * Cin * sizeof(float));
        cuMemAlloc(&d_dlin, (size_t)T * EXTRA * CTX * sizeof(float));
        cuMemAlloc(&d_done, dino_per);
        cuMemcpyHtoD(d_din, dino_hidden_states, (size_t)T * Cin * sizeof(float));
        k_linear(&s->kk, d_dlin, d_din, s->dino_pw, s->dino_pb, T, Cin, EXTRA * CTX);
        k_layernorm(&s->kk, d_done, d_dlin, s->dino_png, s->dino_pnb, rows_out, CTX);
        if (!s->d_dino)
            cuMemAlloc(&s->d_dino, (size_t)s->Beff_main * dino_per);
        for (int b = 0; b < s->Beff_main; b++)
            cuMemcpyDtoD(s->d_dino + (CUdeviceptr)b * dino_per, d_done, dino_per);
        cuMemFree(d_din); cuMemFree(d_dlin); cuMemFree(d_done);
    }

    s->cond_set = 1;
    /* Note: do NOT reset chunk_dual_done here — set_conditioning may be called
     * once per step (cheap rebuild of conditioning device buffers) while the
     * RA cache from a prior run_dual stays valid for the active chunk. The
     * chunk's dual_done is invalidated only by set_chunk on a fresh chunk. */
}

void paint_stage_unet_set_chunk(paint_stage_unet *s, int chunk_id) {
    if (chunk_id < 0 || chunk_id >= PAINT_UNET_MAX_CHUNKS) {
        fprintf(stderr, "[paint_stage_unet] ERROR: chunk_id %d out of range\n", chunk_id);
        return;
    }
    s->active_chunk = chunk_id;
    g_ra_cache.slots = s->ra_slots[chunk_id];
    g_ra_cache.idx = 0;
}

void paint_stage_unet_run_dual(paint_stage_unet *s) {
    if (!s->cond_set) {
        fprintf(stderr, "[paint_stage_unet] ERROR: set_conditioning first\n");
        return;
    }
    const paint_unet_config *cfg = &s->cfg;
    const int H0 = cfg->H0, W0 = cfg->W0, M_text = cfg->M_text;
    const int IC_dual = 4;

    g_ra_mode = 1;
    g_ra_cache.slots = s->ra_slots[s->active_chunk];
    g_ra_cache.idx = 0;

    int64_t ts_dual_arr[16];
    for (int b = 0; b < s->Beff_dual; b++) ts_dual_arr[b] = 0;
    cuMemcpyHtoD(s->d_ts_dual, ts_dual_arr, s->Beff_dual * sizeof(int64_t));
    k_timestep_embed(&s->kk, s->d_temb_in_d, s->d_ts_dual, s->Beff_dual, 320);
    k_linear(&s->kk, s->d_temb_h1_d, s->d_temb_in_d, s->l1_wd, s->l1_bd, s->Beff_dual, 320, 1280);
    k_silu(&s->kk, s->d_temb_h1_d, s->Beff_dual * 1280);
    k_linear(&s->kk, s->d_temb_d, s->d_temb_h1_d, s->l2_wd, s->l2_bd, s->Beff_dual, 1280, 1280);

    size_t in_n_d = (size_t)s->Beff_dual * IC_dual * H0 * W0;
    cuMemcpyHtoD(s->d_in_raw_d, s->packed_dual, in_n_d * sizeof(float));
    for (int b = 0; b < s->Beff_dual; b++) {
        CUdeviceptr ib = s->d_in_raw_d + (CUdeviceptr)b * IC_dual * H0 * W0 * sizeof(float);
        CUdeviceptr ob = s->ws.d_a     + (CUdeviceptr)b * 320     * H0 * W0 * sizeof(float);
        k_conv(&s->kk, ob, ib, s->cw_d, s->cb_d, IC_dual, H0, W0, 320, 3, 3, 1);
    }
    pu_skip_stack ssd = {.top = 0, .B = s->Beff_dual};
    skip_push_copy(&ssd, s->ws.d_a, 320, H0, W0);
    int H = H0, W = W0;
    run_down_block(&s->kk, &s->dbd[0], s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_down_block(&s->kk, &s->dbd[1], s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_down_block(&s->kk, &s->dbd[2], s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_down_block(&s->kk, &s->dbd[3], s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_mid_block(&s->kk, &s->midd, s->ws.d_a, s->ws.d_b, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, H, W, M_text, &s->ws);
    run_up_block(&s->kk, &s->ubd[0], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_up_block(&s->kk, &s->ubd[1], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_up_block(&s->kk, &s->ubd[2], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    run_up_block(&s->kk, &s->ubd[3], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_d, s->d_text_dual, 0, 0, s->Beff_dual, &H, &W, M_text, &s->ws, &ssd);
    s->chunk_dual_done[s->active_chunk] = 1;
    fprintf(stderr, "[paint_stage_unet] dual-stream RA cache populated for chunk %d (%d slots)\n",
            s->active_chunk, g_ra_cache.idx);
}

void paint_stage_unet_run_step(paint_stage_unet *s, long long timestep,
                                const float *x_host, float *noise_pred_host) {
    if (!s->chunk_dual_done[s->active_chunk]) {
        fprintf(stderr, "[paint_stage_unet] ERROR: run_dual first for chunk %d\n",
                s->active_chunk);
        return;
    }
    /* Point the read-side cache at the active chunk's slots. */
    g_ra_cache.slots = s->ra_slots[s->active_chunk];
    const paint_unet_config *cfg = &s->cfg;
    const int H0 = cfg->H0, W0 = cfg->W0, M_text = cfg->M_text;
    const int N_PBR = cfg->N_pbr, N_GEN = cfg->N_gen;
    const int IC_main = 12;
    const size_t per_view = s->per_view, per_in_main = s->per_in_main;
    int H = H0, W = W0;

    /* Pack current x into the sample slot of packed_main */
    for (int p = 0; p < N_PBR; p++)
        for (int g = 0; g < N_GEN; g++) {
            int b = p * N_GEN + g;
            float *dst = s->packed_main + (size_t)b * per_in_main;
            memcpy(dst, x_host + (size_t)b * per_view, per_view * sizeof(float));
        }
    size_t in_n_m = (size_t)s->Beff_main * IC_main * H0 * W0;
    cuMemcpyHtoD(s->d_in_raw_m, s->packed_main, in_n_m * sizeof(float));

    /* Timestep embed for current t */
    int64_t ts_main_arr[16];
    for (int b = 0; b < s->Beff_main; b++) ts_main_arr[b] = (int64_t)timestep;
    cuMemcpyHtoD(s->d_ts_main, ts_main_arr, s->Beff_main * sizeof(int64_t));
    k_timestep_embed(&s->kk, s->d_temb_in_m, s->d_ts_main, s->Beff_main, 320);
    k_linear(&s->kk, s->d_temb_h1_m, s->d_temb_in_m, s->l1_w, s->l1_b, s->Beff_main, 320, 1280);
    k_silu(&s->kk, s->d_temb_h1_m, s->Beff_main * 1280);
    k_linear(&s->kk, s->d_temb_m, s->d_temb_h1_m, s->l2_w, s->l2_b, s->Beff_main, 1280, 1280);

    /* Main forward (RA mode='r', read in-order from cache) */
    g_ra_mode = 2; g_ra_cache.idx = 0;
    for (int b = 0; b < s->Beff_main; b++) {
        CUdeviceptr ib = s->d_in_raw_m + (CUdeviceptr)b * IC_main * H0 * W0 * sizeof(float);
        CUdeviceptr ob = s->ws.d_a     + (CUdeviceptr)b * 320     * H0 * W0 * sizeof(float);
        k_conv(&s->kk, ob, ib, s->cw, s->cb, IC_main, H0, W0, 320, 3, 3, 1);
    }
    pu_skip_stack ss = {.top = 0, .B = s->Beff_main};
    skip_push_copy(&ss, s->ws.d_a, 320, H0, W0);
    run_down_block(&s->kk, &s->db[0], s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_down_block(&s->kk, &s->db[1], s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_down_block(&s->kk, &s->db[2], s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_down_block(&s->kk, &s->db[3], s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_mid_block(&s->kk, &s->mid, s->ws.d_a, s->ws.d_b, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, H, W, M_text, &s->ws);
    run_up_block(&s->kk, &s->ub[0], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_up_block(&s->kk, &s->ub[1], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_up_block(&s->kk, &s->ub[2], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);
    run_up_block(&s->kk, &s->ub[3], s->ws.d_a, s->ws.d_b, s->d_concat, s->d_temb_m, s->d_text_m, s->d_dino, s->M_dino, s->Beff_main, &H, &W, M_text, &s->ws, &ss);

    for (int b = 0; b < s->Beff_main; b++) {
        CUdeviceptr xb = s->ws.d_a + (CUdeviceptr)b * 320 * H * W * sizeof(float);
        CUdeviceptr yb = s->ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
        k_groupnorm(&s->kk, yb, xb, s->ng, s->nb_w, 320, H * W, 32, 1);
    }
    for (int b = 0; b < s->Beff_main; b++) {
        CUdeviceptr yb = s->ws.d_b + (CUdeviceptr)b * 320 * H * W * sizeof(float);
        CUdeviceptr ob = s->ws.d_a + (CUdeviceptr)b * 4   * H * W * sizeof(float);
        k_conv(&s->kk, ob, yb, s->ow, s->ob_w, 320, H, W, 4, 3, 3, 1);
    }
    /* DtoH on default stream is implicitly synchronous — no explicit sync needed. */
    cuMemcpyDtoH(noise_pred_host, s->ws.d_a, s->x_n * sizeof(float));
}

void paint_stage_unet_destroy(paint_stage_unet *s) {
    if (!s) return;
    free(s->packed_main); free(s->packed_dual); free(s->text_tiled_main);
    if (s->kk.mod) cuModuleUnload(s->kk.mod);
    g_ra_mode = 0;
    free(s);
}
