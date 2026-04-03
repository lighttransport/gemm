/*
 * hip_da3_runner.c - HIP/ROCm DA3 depth estimation via HIPRTC-compiled kernels
 *
 * GPU-accelerated approach (RDNA4, gfx1200/gfx1201):
 *   - Preprocessing + patch embedding: GPU
 *   - 12 transformer blocks (backbone): GPU (F16 tiled GEMM, tiled FlashAttention)
 *   - DPT head (token processing, RefineNet fusion, output convs): GPU
 *
 * Compiles with plain gcc (no hipcc). Uses rocew for dynamic HIP/HIPRTC loading.
 * F16 weights on GPU, F32 compute. Single-stream sequential kernel launches.
 *
 * Key differences from CUDA version:
 *   - No MMA/tensor core kernels (RDNA4 has no MMA)
 *   - gemm_tiled_f16_f32 is the PRIMARY GEMM kernel
 *   - No FP8 GEMM (no hardware FP8 MMA on RDNA4)
 *   - PTX inline ASM replaced with HIP builtins
 *
 * Port of cuda_da3_runner.c for AMD ROCm/HIP.
 */

/* CPU inference library (for preprocessing) */
#define GGML_DEQUANT_IMPLEMENTATION
#define DEPTH_ANYTHING3_IMPLEMENTATION
#include "../../common/depth_anything3.h"

#include "hip_da3_runner.h"
#include "../rocew.h"
#include "../hip_kernels_common.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ======================================================================== */
/* DA3-specific HIP kernels (compiled at runtime via HIPRTC)                */
/* Shared kernels (layernorm, GEMM, attention, etc.) are in                 */
/* hip_kernels_common.h. This string is concatenated after them.            */
/* ======================================================================== */

static const char *hip_da3_specific_kernels =
"\n"
"/* ---- DA3: qk_layernorm_f32: per-head LN on Q/K with stride ---- */\n"
"/* stride = distance in floats between same element in consecutive tokens */\n"
"__global__ void qk_layernorm_f32(float *vec, const float *w, const float *b,\n"
"                                   int n_tok, int n_heads, int head_dim,\n"
"                                   int stride, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int idx = blockIdx.x;\n"
"    int t = idx / n_heads, h = idx % n_heads;\n"
"    if (t >= n_tok) return;\n"
"    int tid = threadIdx.x;\n"
"    float *v = vec + t * stride + h * head_dim;\n"
"    float val = (tid < head_dim) ? v[tid] : 0.0f;\n"
"    /* Mean */\n"
"    float s = val;\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid+r];\n"
"        __syncthreads();\n"
"    }\n"
"    float mean = sdata[0] / (float)head_dim;\n"
"    /* Var */\n"
"    float d = val - mean;\n"
"    s = d * d;\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid+r];\n"
"        __syncthreads();\n"
"    }\n"
"    float inv = rsqrtf(sdata[0] / (float)head_dim + eps);\n"
"    if (tid < head_dim)\n"
"        v[tid] = (val - mean) * inv * w[tid] + b[tid];\n"
"}\n"
"\n"
"/* ---- 5. rope_2d_f32: 2D RoPE with stride ---- */\n"
"__global__ void rope_2d_f32(float *vec, const int *pos_y, const int *pos_x,\n"
"                             int n_tok, int n_heads, int head_dim,\n"
"                             int stride, float freq_base) {\n"
"    int t = blockIdx.x;\n"
"    if (t >= n_tok) return;\n"
"    int tid = threadIdx.x;\n"
"    int half = head_dim / 2;\n"
"    int quarter = half / 2;\n"
"    int total = n_heads * quarter;\n"
"    if (tid >= total) return;\n"
"    int h = tid / quarter;\n"
"    int j = tid % quarter;\n"
"    float py = (float)pos_y[t];\n"
"    float px = (float)pos_x[t];\n"
"    float *v = vec + t * stride + h * head_dim;\n"
"    float freq = 1.0f / powf(freq_base, (float)(2*j) / (float)half);\n"
"    /* Y rotation: first half */\n"
"    float ty = py * freq;\n"
"    float cy = cosf(ty), sy = sinf(ty);\n"
"    float v0y = v[j], v1y = v[j + quarter];\n"
"    v[j]           = v0y * cy - v1y * sy;\n"
"    v[j + quarter] = v0y * sy + v1y * cy;\n"
"    /* X rotation: second half */\n"
"    float tx = px * freq;\n"
"    float cx = cosf(tx), sx = sinf(tx);\n"
"    float v0x = v[half + j], v1x = v[half + j + quarter];\n"
"    v[half + j]           = v0x * cx - v1x * sx;\n"
"    v[half + j + quarter] = v0x * sx + v1x * cx;\n"
"}\n"
"\n"
"/* ---- 7. swiglu_f32: dst = silu(gate) * up ---- */\n"
"__global__ void swiglu_f32(float *dst, const float *gate_up, int hidden, int n_tok) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = hidden * n_tok;\n"
"    if (i >= total) return;\n"
"    int t = i / hidden, j = i % hidden;\n"
"    const float *gu = gate_up + t * 2 * hidden;\n"
"    float g = gu[j];\n"
"    g = g / (1.0f + expf(-g));\n"
"    dst[t * hidden + j] = g * gu[j + hidden];\n"
"}\n"
"\n"
"/* ---- 9. layerscale_add_f32: hidden[i] += proj[i] * gamma[i%dim] ---- */\n"
"__global__ void layerscale_add_f32(float *hidden, const float *proj, const float *gamma,\n"
"                                    int dim, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) hidden[i] += proj[i] * gamma[i % dim];\n"
"}\n"
"\n"
"/* ---- 12. depth_activation ---- */\n"
"__global__ void depth_activation(float *out, int hw) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < hw) {\n"
"        out[i]      = expf(out[i]);\n"
"        out[i + hw] = expf(out[i + hw]) + 1.0f;\n"  /* expp1: exp(x) + 1, per DA3 source */
"    }\n"
"}\n"
"\n"
"/* ---- 14. conv2d_f32 ---- */\n"
"__global__ void conv2d_f32(float *dst, const float *src, const float *weight,\n"
"                            const float *bias, int H, int W, int Ci, int Co,\n"
"                            int kH, int kW, int stride, int pad) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int Ho = (H + 2*pad - kH) / stride + 1;\n"
"    int Wo = (W + 2*pad - kW) / stride + 1;\n"
"    int total = Co * Ho * Wo;\n"
"    if (idx >= total) return;\n"
"    int co = idx / (Ho * Wo);\n"
"    int rem = idx % (Ho * Wo);\n"
"    int oh = rem / Wo, ow = rem % Wo;\n"
"    float sum = bias ? bias[co] : 0.0f;\n"
"    for (int ci = 0; ci < Ci; ci++) {\n"
"        for (int kh = 0; kh < kH; kh++) {\n"
"            int ih = oh * stride - pad + kh;\n"
"            if (ih < 0 || ih >= H) continue;\n"
"            for (int kw = 0; kw < kW; kw++) {\n"
"                int iw = ow * stride - pad + kw;\n"
"                if (iw < 0 || iw >= W) continue;\n"
"                sum += weight[((co*Ci+ci)*kH+kh)*kW+kw] * src[ci*H*W + ih*W + iw];\n"
"            }\n"
"        }\n"
"    }\n"
"    dst[idx] = sum;\n"
"}\n"
"\n"
"/* ---- 15. dpt_cls_concat: extract patches + concat CLS ---- */\n"
"__global__ void dpt_cls_concat(float *dst, const float *src, int np, int dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = np * 2 * dim;\n"
"    if (idx >= total) return;\n"
"    int p = idx / (2 * dim);\n"
"    int j = idx % (2 * dim);\n"
"    dst[idx] = (j < dim) ? src[(1 + p) * dim + j] : src[j - dim];\n"
"}\n"
"\n"
"/* ---- 17. dpt_tok_to_chw: token-major [np,C] -> spatial CHW [C,H,W] ---- */\n"
"__global__ void dpt_tok_to_chw(float *dst, const float *src, int C, int gH, int gW) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = C * gH * gW;\n"
"    if (idx >= total) return;\n"
"    int c = idx / (gH * gW);\n"
"    int rem = idx % (gH * gW);\n"
"    int p = (rem / gW) * gW + (rem % gW);\n"
"    dst[idx] = src[p * C + c];\n"
"}\n"
"\n"
"/* ---- 22. deconv_scatter_f32: scatter GEMM output to spatial CHW for ConvTranspose2d ---- */\n"
"/* Input: Y[Hi*Wi, kH*kW*Co] (GEMM output). Output: dst[Co, Ho, Wo]. */\n"
"/* For stride-aligned ConvT (kH==stride_h, kW==stride_w): each output pixel has */\n"
"/* exactly 1 source, selected by (oh%stride, ow%stride). */\n"
"__global__ void deconv_scatter_f32(float *dst, const float *Y, const float *bias,\n"
"                                     int Co, int Hi, int Wi, int Ho, int Wo,\n"
"                                     int kH, int kW, int stride_h, int stride_w) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = Co * Ho * Wo;\n"
"    if (idx >= total) return;\n"
"    int co = idx / (Ho * Wo);\n"
"    int rem = idx % (Ho * Wo);\n"
"    int oh = rem / Wo, ow = rem % Wo;\n"
"    int kh = oh % stride_h, kw = ow % stride_w;\n"
"    int ih = oh / stride_h, iw = ow / stride_w;\n"
"    int group = kh * kW + kw;\n"
"    int pos = ih * Wi + iw;\n"
"    int n_groups = kH * kW;\n"
"    float v = Y[pos * (n_groups * Co) + group * Co + co];\n"
"    dst[idx] = v + (bias ? bias[co] : 0.0f);\n"
"}\n"
"\n"
"/* ---- 24. groupnorm_f32: per-channel spatial normalization ---- */\n"
"/* For GroupNorm(G) where G==C: normalize each channel across H*W. */\n"
"/* Grid: (C, 1), threads: 256. */\n"
"__global__ void groupnorm_f32(float *dst, const float *src, const float *w,\n"
"                                const float *b, int C, int HW, float eps) {\n"
"    int c = blockIdx.x;\n"
"    if (c >= C) return;\n"
"    int tid = threadIdx.x;\n"
"    extern __shared__ float sdata[];\n"
"    float s = 0.0f;\n"
"    for (int i = tid; i < HW; i += blockDim.x)\n"
"        s += src[c * HW + i];\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)HW;\n"
"    s = 0.0f;\n"
"    for (int i = tid; i < HW; i += blockDim.x) {\n"
"        float d = src[c * HW + i] - mean; s += d * d; }\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) {\n"
"        if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)HW + eps);\n"
"    float sc = w ? w[c] : 1.0f;\n"
"    float bi = b ? b[c] : 0.0f;\n"
"    for (int i = tid; i < HW; i += blockDim.x)\n"
"        dst[c * HW + i] = (src[c * HW + i] - mean) * inv * sc + bi;\n"
"}\n"
"\n"
"/* Per-position channel LayerNorm: normalize across C channels at each (h,w). */\n"
"/* Data layout: [C, H, W] = src[c * HW + hw]. Grid: ceil(HW/256), threads: 256. */\n"
"__global__ void channel_layernorm_f32(float *dst, const float *src, const float *w,\n"
"                                       const float *b, int C, int HW, float eps) {\n"
"    int hw = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (hw >= HW) return;\n"
"    float mean = 0.0f;\n"
"    for (int c = 0; c < C; c++) mean += src[c * HW + hw];\n"
"    mean /= (float)C;\n"
"    float var = 0.0f;\n"
"    for (int c = 0; c < C; c++) { float d = src[c * HW + hw] - mean; var += d * d; }\n"
"    float inv = rsqrtf(var / (float)C + eps);\n"
"    for (int c = 0; c < C; c++)\n"
"        dst[c * HW + hw] = (src[c * HW + hw] - mean) * inv * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);\n"
"}\n"
"\n"
"} /* extern C */\n"
;

/* Error macros from shared header: HIP_CHECK, HIP_CHECK_NULL */
#define CHECK_HIP HIP_CHECK
#define CHECK_HIP_NULL HIP_CHECK_NULL

/* ======================================================================== */
/* Runner state                                                             */
/* ======================================================================== */

typedef struct {
    void *ln1_w, *ln1_b;
    void *attn_qkv_w, *attn_qkv_b;
    void *attn_q_norm_w, *attn_q_norm_b;
    void *attn_k_norm_w, *attn_k_norm_b;
    void *attn_out_w, *attn_out_b;
    void *ls1, *ls2;
    void *ln2_w, *ln2_b;
    void *ffn_gate_up_w, *ffn_gate_up_b;
    void *ffn_up_w, *ffn_up_b;
    void *ffn_down_w, *ffn_down_b;
    int has_qk_norm, has_swiglu;
    int qkv_rows, qkv_cols;
    int out_rows, out_cols;
    int ffn_gu_rows, ffn_up_rows, ffn_down_rows;
    int ffn_gu_cols, ffn_up_cols, ffn_down_cols;
} hip_da3_layer;

typedef struct {
    void *norm_w, *norm_b;              /* Head LayerNorm F32 */
    void *proj_w[4], *proj_b[4];        /* 1x1 proj: w=F16, b=F32 */
    int proj_rows[4];                        /* oc_val per level */
    void *upsample_0_w, *upsample_0_b;  /* ConvT 4x4 s4: w=FP16 transposed [kH*kW*Co,Ci] */
    void *upsample_1_w, *upsample_1_b;  /* ConvT 2x2 s2: w=FP16 transposed [kH*kW*Co,Ci] */
    void *downsample_w, *downsample_b;   /* Conv 3x3 s2: F16 */
    void *adapter_w[4];                 /* Conv 3x3 no-bias: F32 */
    void *fuse_out_w[4], *fuse_out_b[4]; /* 1x1 out_conv: F32 */
    void *fuse_rcu1_c1_w[4], *fuse_rcu1_c1_b[4];
    void *fuse_rcu1_c2_w[4], *fuse_rcu1_c2_b[4];
    void *fuse_rcu2_c1_w[4], *fuse_rcu2_c1_b[4];
    void *fuse_rcu2_c2_w[4], *fuse_rcu2_c2_b[4];
    void *neck_w, *neck_b;
    void *out_0_w, *out_0_b;
    void *out_2_w, *out_2_b;
    int has_rcu1[4], has_rcu2[4];
    int out_mid;  /* intermediate channels in output_conv2 (typically 32) */
} dpt_gpu_weights;

struct hip_da3_runner {
    hipDevice_t device;
    hipCtx_t context;
    hipStream_t stream;
    int verbose;

    hipModule_t module;
    hipFunction_t fn_layernorm_f32;
    hipFunction_t fn_gemm_tiled_f16_f32;
    hipFunction_t fn_flash_attn_tiled_f32;
    hipFunction_t fn_add_bias_f32;
    hipFunction_t fn_qk_layernorm_f32;
    hipFunction_t fn_rope_2d_f32;
    hipFunction_t fn_swiglu_f32;
    hipFunction_t fn_gelu_f32;
    hipFunction_t fn_layerscale_add_f32;
    hipFunction_t fn_add_f32;
    hipFunction_t fn_relu_f32;
    hipFunction_t fn_depth_activation;
    hipFunction_t fn_bilinear_upsample_f32;
    hipFunction_t fn_conv2d_f32;
    hipFunction_t fn_dpt_cls_concat;
    hipFunction_t fn_dpt_tok_to_chw;
    hipFunction_t fn_kv_transpose;
    hipFunction_t fn_resize_normalize;
    hipFunction_t fn_patch_embed_conv2d;
    hipFunction_t fn_cls_pos_embed;
    hipFunction_t fn_deconv_scatter_f32;
    hipFunction_t fn_groupnorm_f32;
    hipFunction_t fn_channel_layernorm_f32;
    hipFunction_t fn_silu_f32;

    /* Model params */
    int n_blocks, dim, n_heads, head_dim, ffn_hidden;
    int patch_size, image_size, grid_h, grid_w, n_patches, n_tokens;
    float ln_eps;
    int rope_start, qk_norm_start;
    int feature_layers[4];
    int use_swiglu;
    int head_features, head_out_channels[4];
    float image_mean[3], image_std[3];

    /* GPU weights */
    void *d_patch_embed_w, *d_patch_embed_b;
    void *d_cls_token, *d_pos_embed;
    hip_da3_layer *layers;

    /* Preprocessing buffers */
    void *d_img_norm;   /* [3, target_h, target_w] float */
    void *d_img_raw;    /* reusable raw RGB upload buffer */
    size_t d_img_raw_cap;     /* current capacity in bytes */
    void *d_result;     /* reusable final result buffer */
    size_t d_result_cap;

    /* Scratch buffers */
    void *d_hidden, *d_hidden2, *d_ln_buf, *d_qkv, *d_attn_out;
    void *d_ffn_buf, *d_ffn_mid, *d_proj_out;
    void *d_pos_y, *d_pos_x;
    void *d_features[4]; /* saved backbone features */

    /* DPT head GPU weights */
    dpt_gpu_weights dpt_w;

    /* DPT scratch buffers */
    void *d_dpt_cat;       /* [np, 2*dim] or fusion scratch */
    void *d_dpt_ln;        /* [np, 2*dim] or fusion scratch */
    void *d_dpt_proj;      /* [np, max_oc] */
    void *d_dpt_chw;       /* [max_oc, gh, gw] */
    void *d_dpt_spatial[4]; /* per-level after resize */
    void *d_dpt_adapted[4]; /* per-level after adapter */
    void *d_dpt_fused;     /* [feat, max_h, max_w] for fusion */
    void *d_dpt_tmp;       /* scratch for RCU/bilinear */
    void *d_dpt_tmp2;      /* scratch for RCU conv mid */
    void *d_dpt_out;       /* [2, 148, 148] final output */

    /* CameraDec weights (Phase 1: pose estimation) */
    struct {
        void *backbone_norm_w, *backbone_norm_b;   /* F32 */
        void *mlp_w[2], *mlp_b[2];                /* F16, [dim*2, dim*2] */
        void *fc_t_w, *fc_t_b;                    /* F32 [3, dim*2] (tiny, CPU matmul) */
        void *fc_qvec_w, *fc_qvec_b;              /* F32 [4, dim*2] */
        void *fc_fov_w, *fc_fov_b;                /* F32 [2, dim*2] */
        int mlp_dim;                                    /* dim*2 */
        int loaded;
    } cam_dec;

    /* CameraEnc weights (Phase 3: pose conditioning) */
    struct {
        void *fc1_w, *fc1_b, *fc2_w, *fc2_b;        /* F16, pose MLP 9->dim->dim*2 */
        hip_da3_layer *trunk;                          /* 4 transformer blocks */
        int n_trunk_blocks;
        void *trunk_norm_w, *trunk_norm_b;         /* F32 */
        void *token_norm_w, *token_norm_b;         /* F32 */
        int trunk_dim;                                  /* typically dim */
        int loaded;
    } cam_enc;

    /* DPT Aux Branch weights (Phase 2: rays + sky seg) */
    struct {
        /* Aux RefineNet (same structure as main fuse weights) */
        void *fuse_out_w[4], *fuse_out_b[4];
        void *fuse_rcu1_c1_w[4], *fuse_rcu1_c1_b[4];
        void *fuse_rcu1_c2_w[4], *fuse_rcu1_c2_b[4];
        void *fuse_rcu2_c1_w[4], *fuse_rcu2_c1_b[4];
        void *fuse_rcu2_c2_w[4], *fuse_rcu2_c2_b[4];
        int has_rcu1[4], has_rcu2[4];
        /* Per-level output conv chains: output_conv1_aux (5 Conv2d each, F32 weights) */
        void *oc1_w[4][5], *oc1_b[4][5];
        int oc1_ci[4][5], oc1_co[4][5];
        int oc1_count[4];
        /* output_conv2_aux: Conv2d(128,32,3) + GroupNorm(32) + Conv2d(32,7,1) */
        void *oc2_conv_w[4], *oc2_conv_b[4];
        void *oc2_gn_w[4], *oc2_gn_b[4];
        void *oc2_out_w[4], *oc2_out_b[4];
        int loaded;
    } dpt_aux;

    /* DPT Aux scratch buffers */
    void *d_aux_out[4];        /* per-level 7-channel output [7, sp_h, sp_w] */
    void *d_aux_scratch;       /* scratch for aux output conv chains */

    /* GSDPT weights (Phase 4: 3D Gaussian estimation) */
    struct {
        dpt_gpu_weights dpt;                           /* standard DPT weights */
        void *merger_w[3], *merger_b[3];          /* images_merger Conv2d, F32 weights */
        int merger_ci[3], merger_co[3];                /* 3->32->64->128 */
        int gs_out_channels;                           /* 38 */
        int loaded;
    } gsdpt;

    /* GSDPT scratch buffers */
    void *d_gs_merged;         /* images_merger output [128, mg_h, mg_w] */
    void *d_gs_out;            /* [38, fh, fw] gaussian output */
    int gs_merger_h, gs_merger_w;    /* merger output spatial dims */

    /* Nested metric model (Phase 6) */
    struct {
        int n_blocks, dim, n_heads, head_dim, ffn_hidden;
        int feature_layers[4], rope_start, qk_norm_start;
        int use_swiglu;
        void *d_patch_embed_w, *d_patch_embed_b;
        void *d_cls_token, *d_pos_embed;
        hip_da3_layer *layers;
        dpt_gpu_weights dpt_w;
        void *d_features[4];
        int loaded;
    } metric;

    /* CPU model for preprocessing */
    da3_model *cpu_model;

    /* Host output */
    float *h_output;
    int loaded;
};

/* ======================================================================== */
/* HIPRTC compilation                                                       */
/* ======================================================================== */

static int da3_compile_kernels(hip_da3_runner *r) {
    /* Concatenate shared + DA3-specific kernel source */
    size_t len1 = strlen(hip_kernels_common_src);
    size_t len2 = strlen(hip_da3_specific_kernels);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, hip_kernels_common_src, len1);
    memcpy(full_src + len1, hip_da3_specific_kernels, len2 + 1);

    int rc = hip_compile_kernels(&r->module, r->device, full_src,
                                  "da3_kernels", r->verbose, "hip_da3");
    free(full_src);
    if (rc < 0) return -1;

    hipError_t err;
#define GET_FN(name) do { \
    err = hipModuleGetFunction(&r->fn_##name, r->module, #name); \
    if (err != hipSuccess) { fprintf(stderr, "hip_da3: kernel '%s' not found\n", #name); return -1; } \
} while(0)

    /* Shared kernels (from hip_kernels_common.h) */
    GET_FN(layernorm_f32);
    GET_FN(gemm_tiled_f16_f32);
    GET_FN(flash_attn_tiled_f32);
    GET_FN(add_bias_f32);
    GET_FN(gelu_f32);
    GET_FN(add_f32);
    GET_FN(relu_f32);
    GET_FN(bilinear_upsample_f32);
    GET_FN(kv_transpose);
    GET_FN(resize_normalize);
    GET_FN(patch_embed_conv2d);
    GET_FN(cls_pos_embed);
    GET_FN(silu_f32);

    /* DA3-specific kernels */
    GET_FN(qk_layernorm_f32);
    GET_FN(rope_2d_f32);
    GET_FN(swiglu_f32);
    GET_FN(layerscale_add_f32);
    GET_FN(depth_activation);
    GET_FN(conv2d_f32);
    GET_FN(dpt_cls_concat);
    GET_FN(dpt_tok_to_chw);
    GET_FN(deconv_scatter_f32);
    GET_FN(groupnorm_f32);
    GET_FN(channel_layernorm_f32);

#undef GET_FN

    if (r->verbose >= 1)
        fprintf(stderr, "hip_da3: %d kernels compiled\n", 25);
    return 0;
}

/* ======================================================================== */
/* Upload tensor to GPU                                                     */
/* ======================================================================== */

#define upload_tensor_raw hip_upload_raw

static void *upload_tensor_f32(const qtensor *t) {
    if (!t->data) return NULL;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else {
        memset(buf, 0, (size_t)n * sizeof(float));
    }
    void *d = NULL;
    if (hipMalloc(&d, (size_t)n * sizeof(float)) != hipSuccess) { free(buf); return NULL; }
    hipMemcpy(d, buf, (size_t)n * sizeof(float), hipMemcpyHostToDevice);
    free(buf);
    return d;
}

static void *upload_tensor_f16(const qtensor *t) {
    if (!t->data || t->type != GGML_TYPE_F16) return upload_tensor_f32(t);
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    return upload_tensor_raw(t->data, (size_t)n * 2);
}

/* FP32 -> FP16 conversion (from shared header) */
#define fp32_to_fp16_raw hip_f32_to_f16

/* Upload tensor to GPU as FP16 (converts F32 -> FP16 on CPU if needed) */
static void *upload_tensor_f32_as_f16(const qtensor *t) {
    if (!t->data) return NULL;
    int n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= (int)t->dims[d];
    if (t->type == GGML_TYPE_F16)
        return upload_tensor_raw(t->data, (size_t)n * 2);
    uint16_t *buf = (uint16_t *)malloc((size_t)n * 2);
    if (t->type == GGML_TYPE_F32) {
        const float *src = (const float *)t->data;
        for (int i = 0; i < n; i++) buf[i] = fp32_to_fp16_raw(src[i]);
    } else {
        memset(buf, 0, (size_t)n * 2);
    }
    void *d = NULL;
    if (hipMalloc(&d, (size_t)n * 2) != hipSuccess) { free(buf); return NULL; }
    hipMemcpy(d, buf, (size_t)n * 2, hipMemcpyHostToDevice);
    free(buf);
    return d;
}

/* Upload ConvTranspose2d weight transposed for GEMM-based deconv.
 * Input layout: [Ci, Co, kH, kW] (PyTorch ConvTranspose2d).
 * Output layout: [kH*kW*Co, Ci] as FP16 (GEMM W matrix).
 * For stride-aligned deconvs where kH==stride and kW==stride. */
static void *upload_deconv_weight_f16(const qtensor *t, int Ci, int Co, int kH, int kW) {
    if (!t->data) return NULL;
    int N = kH * kW * Co;  /* GEMM n_out */
    int K = Ci;             /* GEMM n_in */
    size_t total = (size_t)Ci * Co * kH * kW;
    uint16_t *buf = (uint16_t *)malloc((size_t)N * K * 2);

    /* Get FP32 source */
    float *f32 = NULL;
    int need_free = 0;
    if (t->type == GGML_TYPE_F32) {
        f32 = (float *)t->data;
    } else if (t->type == GGML_TYPE_F16) {
        f32 = (float *)malloc(total * sizeof(float));
        const uint16_t *src = (const uint16_t *)t->data;
        for (size_t i = 0; i < total; i++) f32[i] = ggml_fp16_to_fp32(src[i]);
        need_free = 1;
    } else {
        memset(buf, 0, (size_t)N * K * 2);
        void *d = NULL;
        if (hipMalloc(&d, (size_t)N * K * 2) != hipSuccess) { free(buf); return NULL; }
        hipMemcpy(d, buf, (size_t)N * K * 2, hipMemcpyHostToDevice);
        free(buf);
        return d;
    }

    /* Transpose [Ci, Co, kH, kW] -> [kH*kW*Co, Ci] as FP16 */
    for (int ci = 0; ci < Ci; ci++) {
        for (int co = 0; co < Co; co++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw_i = 0; kw_i < kW; kw_i++) {
                    int src_idx = ci * (Co * kH * kW) + co * (kH * kW) + kh * kW + kw_i;
                    int dst_row = (kh * kW + kw_i) * Co + co;
                    int dst_idx = dst_row * Ci + ci;
                    buf[dst_idx] = fp32_to_fp16_raw(f32[src_idx]);
                }
            }
        }
    }

    if (need_free) free(f32);

    void *d = NULL;
    if (hipMalloc(&d, (size_t)N * K * 2) != hipSuccess) { free(buf); return NULL; }
    hipMemcpy(d, buf, (size_t)N * K * 2, hipMemcpyHostToDevice);
    free(buf);
    return d;
}

/* ======================================================================== */
/* Public API: init                                                         */
/* ======================================================================== */

hip_da3_runner *hip_da3_init(int device_id, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_da3: rocew init failed (no HIP/HIPRTC?)\n");
        return NULL;
    }
    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        fprintf(stderr, "hip_da3: hipInit failed\n");
        return NULL;
    }

    hip_da3_runner *r = (hip_da3_runner *)calloc(1, sizeof(hip_da3_runner));
    r->verbose = verbose;

    r->device = device_id;
    HIP_CHECK_NULL(hipSetDevice(device_id));
    HIP_CHECK_NULL(hipCtxCreate(&r->context, 0, r->device));
    HIP_CHECK_NULL(hipStreamCreate(&r->stream));

    if (da3_compile_kernels(r) != 0) {
        fprintf(stderr, "hip_da3: kernel compilation failed\n");
        free(r);
        return NULL;
    }

    return r;
}

/* ======================================================================== */
/* Public API: load_weights                                                 */
/* ======================================================================== */

/* GGUF KV helpers */
static int da3g_int(const gguf_context *g, const char *k, int d) {
    int i = gguf_find_key(g, k);
    if (i < 0) return d;
    if (g->kv[i].type == GGUF_TYPE_UINT32) return (int)g->kv[i].value.u32;
    if (g->kv[i].type == GGUF_TYPE_INT32) return g->kv[i].value.i32;
    return d;
}
static float da3g_float(const gguf_context *g, const char *k, float d) {
    int i = gguf_find_key(g, k);
    if (i < 0) return d;
    if (g->kv[i].type == GGUF_TYPE_FLOAT32) return g->kv[i].value.f32;
    return d;
}

static qtensor da3g_tensor(const gguf_context *g, const char *name) {
    qtensor t = {0};
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name.str, name) == 0) {
            t.data = gguf_tensor_data(g, (int)i);
            t.type = g->tensors[i].type;
            t.n_dims = (int)g->tensors[i].n_dims;
            for (int d = 0; d < t.n_dims; d++) t.dims[d] = g->tensors[i].dims[d];
            t.n_cols = (int)g->tensors[i].dims[0];
            t.n_rows = (t.n_dims >= 2) ? (int)g->tensors[i].dims[1] : 1;
            break;
        }
    }
    return t;
}

int hip_da3_load_weights(hip_da3_runner *r, gguf_context *gguf) {
    /* Load hyperparameters */
    r->dim        = da3g_int(gguf, "da3.embed_dim", 384);
    r->n_heads    = da3g_int(gguf, "da3.n_heads", 6);
    r->head_dim   = da3g_int(gguf, "da3.head_dim", 64);
    r->n_blocks   = da3g_int(gguf, "da3.n_blocks", 12);
    r->ffn_hidden = da3g_int(gguf, "da3.ffn_hidden", 1024);
    r->patch_size = da3g_int(gguf, "da3.patch_size", 14);
    r->image_size = da3g_int(gguf, "da3.image_size", 518);
    r->ln_eps     = da3g_float(gguf, "da3.ln_eps", 1e-6f);
    r->rope_start = da3g_int(gguf, "da3.rope_start_layer", 4);
    r->qk_norm_start = da3g_int(gguf, "da3.qk_norm_start_layer", 4);
    r->head_features = da3g_int(gguf, "da3.head.features", 64);

    r->grid_h = r->image_size / r->patch_size;
    r->grid_w = r->grid_h;
    r->n_patches = r->grid_h * r->grid_w;
    r->n_tokens = r->n_patches + 1;

    /* Feature layers */
    int fl = gguf_find_key(gguf, "da3.feature_layers");
    if (fl >= 0 && gguf->kv[fl].type == GGUF_TYPE_ARRAY) {
        int32_t *a = (int32_t *)gguf->kv[fl].value.arr.data;
        for (int i = 0; i < 4; i++) r->feature_layers[i] = a[i];
    } else {
        r->feature_layers[0] = 5; r->feature_layers[1] = 7;
        r->feature_layers[2] = 9; r->feature_layers[3] = 11;
    }

    /* Head out_channels */
    int oc = gguf_find_key(gguf, "da3.head.out_channels");
    if (oc >= 0 && gguf->kv[oc].type == GGUF_TYPE_ARRAY) {
        int32_t *a = (int32_t *)gguf->kv[oc].value.arr.data;
        for (int i = 0; i < 4; i++) r->head_out_channels[i] = a[i];
    } else {
        r->head_out_channels[0] = 48; r->head_out_channels[1] = 96;
        r->head_out_channels[2] = 192; r->head_out_channels[3] = 384;
    }

    /* Image normalization */
    r->image_mean[0] = 0.485f; r->image_mean[1] = 0.456f; r->image_mean[2] = 0.406f;
    r->image_std[0] = 0.229f; r->image_std[1] = 0.224f; r->image_std[2] = 0.225f;

    /* Detect FFN type */
    r->use_swiglu = 0;
    { qtensor t = da3g_tensor(gguf, "da3.blk.0.ffn_gate_up.weight"); if (t.data) r->use_swiglu = 1; }

    /* Upload backbone embeddings (F32 on GPU) */
    {
        qtensor t;
        t = da3g_tensor(gguf, "da3.cls_token");     r->d_cls_token = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.pos_embed");      r->d_pos_embed = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.patch_embed.weight"); r->d_patch_embed_w = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.patch_embed.bias");   r->d_patch_embed_b = upload_tensor_f32(&t);
    }

    /* Upload transformer blocks */
    int nb = r->n_blocks;
    r->layers = (hip_da3_layer *)calloc((size_t)nb, sizeof(hip_da3_layer));
    for (int L = 0; L < nb; L++) {
        hip_da3_layer *ly = &r->layers[L];
        char name[128];
        qtensor t;

#define LOAD_F32(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    t = da3g_tensor(gguf, name); ly->field = upload_tensor_f32(&t);
#define LOAD_F16(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    t = da3g_tensor(gguf, name); ly->field = upload_tensor_f16(&t);

        LOAD_F32(ln1_w, "da3.blk.%d.ln1.weight", L)
        LOAD_F32(ln1_b, "da3.blk.%d.ln1.bias", L)

        LOAD_F16(attn_qkv_w, "da3.blk.%d.attn_qkv.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.attn_qkv.weight", L);
        t = da3g_tensor(gguf, name);
        ly->qkv_rows = t.n_rows; ly->qkv_cols = t.n_cols;
        LOAD_F32(attn_qkv_b, "da3.blk.%d.attn_qkv.bias", L)

        LOAD_F32(attn_q_norm_w, "da3.blk.%d.attn_q_norm.weight", L)
        LOAD_F32(attn_q_norm_b, "da3.blk.%d.attn_q_norm.bias", L)
        LOAD_F32(attn_k_norm_w, "da3.blk.%d.attn_k_norm.weight", L)
        LOAD_F32(attn_k_norm_b, "da3.blk.%d.attn_k_norm.bias", L)
        ly->has_qk_norm = (ly->attn_q_norm_w != NULL);

        LOAD_F16(attn_out_w, "da3.blk.%d.attn_out.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.attn_out.weight", L);
        t = da3g_tensor(gguf, name);
        ly->out_rows = t.n_rows; ly->out_cols = t.n_cols;
        LOAD_F32(attn_out_b, "da3.blk.%d.attn_out.bias", L)

        LOAD_F32(ls1, "da3.blk.%d.ls1", L)
        LOAD_F32(ls2, "da3.blk.%d.ls2", L)
        LOAD_F32(ln2_w, "da3.blk.%d.ln2.weight", L)
        LOAD_F32(ln2_b, "da3.blk.%d.ln2.bias", L)

        LOAD_F16(ffn_gate_up_w, "da3.blk.%d.ffn_gate_up.weight", L)
        if (ly->ffn_gate_up_w) {
            snprintf(name, sizeof(name), "da3.blk.%d.ffn_gate_up.weight", L);
            t = da3g_tensor(gguf, name);
            ly->ffn_gu_rows = t.n_rows; ly->ffn_gu_cols = t.n_cols;
            ly->has_swiglu = 1;
        }
        LOAD_F32(ffn_gate_up_b, "da3.blk.%d.ffn_gate_up.bias", L)

        LOAD_F16(ffn_up_w, "da3.blk.%d.ffn_up.weight", L)
        if (ly->ffn_up_w) {
            snprintf(name, sizeof(name), "da3.blk.%d.ffn_up.weight", L);
            t = da3g_tensor(gguf, name);
            ly->ffn_up_rows = t.n_rows; ly->ffn_up_cols = t.n_cols;
        }
        LOAD_F32(ffn_up_b, "da3.blk.%d.ffn_up.bias", L)

        LOAD_F16(ffn_down_w, "da3.blk.%d.ffn_down.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.ffn_down.weight", L);
        t = da3g_tensor(gguf, name);
        ly->ffn_down_rows = t.n_rows; ly->ffn_down_cols = t.n_cols;
        LOAD_F32(ffn_down_b, "da3.blk.%d.ffn_down.bias", L)

#undef LOAD_F32
#undef LOAD_F16
    }

    /* Upload DPT head weights to GPU */
    {
        qtensor t;
        char name[128];
        dpt_gpu_weights *dw = &r->dpt_w;

        /* Head LayerNorm */
        t = da3g_tensor(gguf, "da3.head.norm.weight"); dw->norm_w = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.head.norm.bias");   dw->norm_b = upload_tensor_f32(&t);

        /* Projection layers: w as F16 for gemm, bias as F32 */
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.proj.%d.weight", i);
            t = da3g_tensor(gguf, name);
            dw->proj_w[i] = upload_tensor_f16(&t);
            dw->proj_rows[i] = t.n_rows;
            snprintf(name, sizeof(name), "da3.head.proj.%d.bias", i);
            t = da3g_tensor(gguf, name);
            dw->proj_b[i] = upload_tensor_f32(&t);
        }

        /* Spatial alignment */
        {
            int oc0 = r->head_out_channels[0];
            t = da3g_tensor(gguf, "da3.head.upsample_0.weight");
            dw->upsample_0_w = upload_deconv_weight_f16(&t, oc0, oc0, 4, 4);
            t = da3g_tensor(gguf, "da3.head.upsample_0.bias");
            dw->upsample_0_b = upload_tensor_f32(&t);

            int oc1 = r->head_out_channels[1];
            t = da3g_tensor(gguf, "da3.head.upsample_1.weight");
            dw->upsample_1_w = upload_deconv_weight_f16(&t, oc1, oc1, 2, 2);
            t = da3g_tensor(gguf, "da3.head.upsample_1.bias");
            dw->upsample_1_b = upload_tensor_f32(&t);
        }
        t = da3g_tensor(gguf, "da3.head.downsample.weight");  dw->downsample_w = upload_tensor_f32_as_f16(&t);
        t = da3g_tensor(gguf, "da3.head.downsample.bias");    dw->downsample_b = upload_tensor_f32(&t);

        /* Adapter convolutions */
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.adapter.%d.weight", i);
            t = da3g_tensor(gguf, name);
            dw->adapter_w[i] = upload_tensor_f32_as_f16(&t);
        }

        /* RefineNet fusion blocks */
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_out_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_out_b[i] = upload_tensor_f32(&t);

            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu1_c1_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu1_c1_b[i] = upload_tensor_f32(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu1_c2_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu1_c2_b[i] = upload_tensor_f32(&t);
            dw->has_rcu1[i] = (dw->fuse_rcu1_c1_w[i] != NULL);

            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu2_c1_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu2_c1_b[i] = upload_tensor_f32(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.weight", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu2_c2_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.bias", i);
            t = da3g_tensor(gguf, name); dw->fuse_rcu2_c2_b[i] = upload_tensor_f32(&t);
            dw->has_rcu2[i] = (dw->fuse_rcu2_c1_w[i] != NULL);
        }

        /* Output convolutions */
        t = da3g_tensor(gguf, "da3.head.neck.weight");  dw->neck_w  = upload_tensor_f32_as_f16(&t);
        t = da3g_tensor(gguf, "da3.head.neck.bias");    dw->neck_b  = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.head.out_0.weight"); dw->out_0_w = upload_tensor_f32_as_f16(&t);
        t = da3g_tensor(gguf, "da3.head.out_0.bias");   dw->out_0_b = upload_tensor_f32(&t);
        dw->out_mid = t.n_cols;
        t = da3g_tensor(gguf, "da3.head.out_2.weight"); dw->out_2_w = upload_tensor_f32(&t);
        t = da3g_tensor(gguf, "da3.head.out_2.bias");   dw->out_2_b = upload_tensor_f32(&t);

        if (r->verbose >= 1)
            fprintf(stderr, "hip_da3: DPT head weights uploaded to GPU\n");
    }

    /* Allocate backbone scratch buffers */
    int nt = r->n_tokens;
    int dim = r->dim;
    int np = r->n_patches;
    int gh = r->grid_h;
    int max_ffn = r->use_swiglu ? 2 * r->ffn_hidden : 4 * dim;

    CHECK_HIP(hipMalloc(&r->d_hidden,   (size_t)nt * dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_hidden2,  (size_t)nt * dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_ln_buf,   (size_t)nt * dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_qkv,      (size_t)nt * 3 * dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_attn_out, (size_t)nt * dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_ffn_buf,  (size_t)nt * max_ffn * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_ffn_mid,  (size_t)nt * r->ffn_hidden * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_proj_out, (size_t)nt * dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_pos_y,    (size_t)nt * sizeof(int)));
    CHECK_HIP(hipMalloc(&r->d_pos_x,    (size_t)nt * sizeof(int)));
    CHECK_HIP(hipMalloc(&r->d_img_norm, (size_t)3 * r->image_size * r->image_size * sizeof(float)));

    for (int i = 0; i < 4; i++)
        CHECK_HIP(hipMalloc(&r->d_features[i], (size_t)nt * dim * sizeof(float)));

    /* Allocate DPT head scratch buffers */
    {
        int feat = r->head_features;
        int oc_max = r->head_out_channels[3];
        int sp_h[4], sp_w[4];
        sp_h[0] = sp_w[0] = (gh - 1) * 4 + 4;
        sp_h[1] = sp_w[1] = (gh - 1) * 2 + 2;
        sp_h[2] = sp_w[2] = gh;
        sp_h[3] = sp_w[3] = (gh + 2 - 3) / 2 + 1;
        int max_hw = sp_h[0] * sp_w[0];

        size_t large_sz = (size_t)feat * max_hw;
        if (large_sz < (size_t)np * 2 * dim)
            large_sz = (size_t)np * 2 * dim;

        CHECK_HIP(hipMalloc(&r->d_dpt_cat,  large_sz * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_ln,   large_sz * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_proj, (size_t)np * oc_max * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_chw,  (size_t)oc_max * gh * gh * sizeof(float)));

        for (int i = 0; i < 4; i++) {
            int oc_i = r->head_out_channels[i];
            CHECK_HIP(hipMalloc(&r->d_dpt_spatial[i],
                                 (size_t)oc_i * sp_h[i] * sp_w[i] * sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_dpt_adapted[i],
                                 (size_t)feat * sp_h[i] * sp_w[i] * sizeof(float)));
        }

        CHECK_HIP(hipMalloc(&r->d_dpt_fused, large_sz * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_tmp,   large_sz * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_tmp2,  large_sz * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_out,   (size_t)2 * max_hw * sizeof(float)));

        if (r->verbose >= 1)
            fprintf(stderr, "hip_da3: DPT scratch buffers allocated (~%.1f MB)\n",
                    (float)(4 * large_sz + np * oc_max + oc_max * gh * gh + 2 * max_hw) * 4 / 1e6f);
    }

    /* Upload position arrays for RoPE */
    int *py = (int *)calloc((size_t)nt, sizeof(int));
    int *px = (int *)calloc((size_t)nt, sizeof(int));
    for (int p = 0; p < np; p++) {
        py[1 + p] = p / r->grid_w;
        px[1 + p] = p % r->grid_w;
    }
    hipMemcpy(r->d_pos_y, py, (size_t)nt * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(r->d_pos_x, px, (size_t)nt * sizeof(int), hipMemcpyHostToDevice);
    free(py); free(px);

    r->h_output = (float *)malloc((size_t)nt * dim * sizeof(float));

    /* Load CPU model for preprocessing */
    r->cpu_model = da3_load(gguf);
    if (!r->cpu_model) {
        fprintf(stderr, "hip_da3: warning: failed to load CPU model\n");
    }

    r->loaded = 1;

    if (r->verbose >= 1)
        fprintf(stderr, "hip_da3: loaded %d blocks, dim=%d, tokens=%d, swiglu=%d\n",
                nb, dim, nt, r->use_swiglu);
    return 0;
}

/* ======================================================================== */
/* Public API: load_safetensors                                             */
/* ======================================================================== */

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

typedef struct { const char *st_suffix; const char *gguf_name; } st_name_map;

/* NOTE: build_st_name_map and da3s_tensor are identical to the CUDA version.
 * They are model-format helpers, not GPU-specific. */
static st_name_map *build_st_name_map(const st_context *st, int *out_count);
static qtensor da3s_tensor(const st_context *st, const st_name_map *map, int map_count,
                           const char *gguf_name);

/* Include the large safetensors name-mapping logic.
 * This is identical to the CUDA version (pure CPU string matching). */
/* For brevity, we define the full implementation inline: */

static st_name_map *build_st_name_map(const st_context *st, int *out_count) {
    const char *bb_prefix = NULL;
    const char *hd_prefix = NULL;
    static const char *bb_candidates[] = {
        "model.da3.backbone.pretrained.", "model.backbone.pretrained.",
        "backbone.pretrained.", "backbone.", "pretrained.", "encoder.", NULL
    };
    static const char *hd_candidates[] = {
        "model.da3.head.", "model.head.", "head.", "depth_head.", "dpt_head.", NULL
    };
    for (int c = 0; bb_candidates[c]; c++) {
        for (int i = 0; i < st->n_tensors; i++) {
            if (strstr(safetensors_name(st, i), "blocks.") &&
                strncmp(safetensors_name(st, i), bb_candidates[c], strlen(bb_candidates[c])) == 0) {
                bb_prefix = bb_candidates[c];
                goto bb_found;
            }
        }
    }
bb_found:
    for (int c = 0; hd_candidates[c]; c++) {
        for (int i = 0; i < st->n_tensors; i++) {
            if (strstr(safetensors_name(st, i), "projects.") &&
                strncmp(safetensors_name(st, i), hd_candidates[c], strlen(hd_candidates[c])) == 0) {
                hd_prefix = hd_candidates[c];
                goto hd_found;
            }
        }
    }
hd_found:

    static const struct { const char *st; const char *gg; } blk_map[] = {
        {"norm1.weight","ln1.weight"},{"norm1.bias","ln1.bias"},
        {"attn.qkv.weight","attn_qkv.weight"},{"attn.qkv.bias","attn_qkv.bias"},
        {"attn.q_norm.weight","attn_q_norm.weight"},{"attn.q_norm.bias","attn_q_norm.bias"},
        {"attn.k_norm.weight","attn_k_norm.weight"},{"attn.k_norm.bias","attn_k_norm.bias"},
        {"attn.proj.weight","attn_out.weight"},{"attn.proj.bias","attn_out.bias"},
        {"ls1.gamma","ls1"},{"ls2.gamma","ls2"},
        {"norm2.weight","ln2.weight"},{"norm2.bias","ln2.bias"},
        {"mlp.w12.weight","ffn_gate_up.weight"},{"mlp.w12.bias","ffn_gate_up.bias"},
        {"mlp.w3.weight","ffn_down.weight"},{"mlp.w3.bias","ffn_down.bias"},
        {"mlp.fc1.weight","ffn_up.weight"},{"mlp.fc1.bias","ffn_up.bias"},
        {"mlp.fc2.weight","ffn_down.weight"},{"mlp.fc2.bias","ffn_down.bias"},
        {NULL, NULL}
    };

    static const struct { const char *st; const char *gg; } rn_map[] = {
        {"out_conv.weight","out.weight"},{"out_conv.bias","out.bias"},
        {"resConfUnit1.conv1.weight","rcu1.conv1.weight"},{"resConfUnit1.conv1.bias","rcu1.conv1.bias"},
        {"resConfUnit1.conv2.weight","rcu1.conv2.weight"},{"resConfUnit1.conv2.bias","rcu1.conv2.bias"},
        {"resConfUnit2.conv1.weight","rcu2.conv1.weight"},{"resConfUnit2.conv1.bias","rcu2.conv1.bias"},
        {"resConfUnit2.conv2.weight","rcu2.conv2.weight"},{"resConfUnit2.conv2.bias","rcu2.conv2.bias"},
        {NULL, NULL}
    };

    st_name_map *map = (st_name_map *)calloc((size_t)st->n_tensors, sizeof(st_name_map));
    int n = 0;

    static const struct { const char *st; const char *gg; } cam_dec_map[] = {
        {"backbone.0.weight","cam_dec.mlp.0.weight"},{"backbone.0.bias","cam_dec.mlp.0.bias"},
        {"backbone.2.weight","cam_dec.mlp.2.weight"},{"backbone.2.bias","cam_dec.mlp.2.bias"},
        {"fc_t.weight","cam_dec.fc_t.weight"},{"fc_t.bias","cam_dec.fc_t.bias"},
        {"fc_qvec.weight","cam_dec.fc_qvec.weight"},{"fc_qvec.bias","cam_dec.fc_qvec.bias"},
        {"fc_fov.0.weight","cam_dec.fc_fov.weight"},{"fc_fov.0.bias","cam_dec.fc_fov.bias"},
        {NULL, NULL}
    };

    static const struct { const char *st; const char *gg; } cam_enc_pose_map[] = {
        {"pose_branch.fc1.weight","cam_enc.fc1.weight"},{"pose_branch.fc1.bias","cam_enc.fc1.bias"},
        {"pose_branch.fc2.weight","cam_enc.fc2.weight"},{"pose_branch.fc2.bias","cam_enc.fc2.bias"},
        {"trunk_norm.weight","cam_enc.trunk_norm.weight"},{"trunk_norm.bias","cam_enc.trunk_norm.bias"},
        {"token_norm.weight","cam_enc.token_norm.weight"},{"token_norm.bias","cam_enc.token_norm.bias"},
        {NULL, NULL}
    };

    for (int i = 0; i < st->n_tensors; i++) {
        const char *key = safetensors_name(st, i);
        char gguf_name[256] = {0};

        if (bb_prefix && strncmp(key, bb_prefix, strlen(bb_prefix)) == 0) {
            const char *s = key + strlen(bb_prefix);
            if (strcmp(s, "cls_token") == 0) strcpy(gguf_name, "da3.cls_token");
            else if (strcmp(s, "pos_embed") == 0) strcpy(gguf_name, "da3.pos_embed");
            else if (strcmp(s, "patch_embed.proj.weight") == 0) strcpy(gguf_name, "da3.patch_embed.weight");
            else if (strcmp(s, "patch_embed.proj.bias") == 0) strcpy(gguf_name, "da3.patch_embed.bias");
            else if (strcmp(s, "norm.weight") == 0) strcpy(gguf_name, "da3.backbone_norm.weight");
            else if (strcmp(s, "norm.bias") == 0) strcpy(gguf_name, "da3.backbone_norm.bias");
            else if (strncmp(s, "blocks.", 7) == 0) {
                int L_idx = 0;
                const char *rest = s + 7;
                while (*rest >= '0' && *rest <= '9') { L_idx = L_idx * 10 + (*rest - '0'); rest++; }
                if (*rest == '.') rest++;
                for (int j = 0; blk_map[j].st; j++) {
                    if (strcmp(rest, blk_map[j].st) == 0) {
                        snprintf(gguf_name, sizeof(gguf_name), "da3.blk.%d.%s", L_idx, blk_map[j].gg);
                        break;
                    }
                }
            }
        }
        else if (strncmp(key, "model.da3.cam_dec.", 18) == 0 || strncmp(key, "model.cam_dec.", 14) == 0) {
            const char *s = strncmp(key, "model.da3.", 10) == 0 ? key + 18 : key + 14;
            for (int j = 0; cam_dec_map[j].st; j++) {
                if (strcmp(s, cam_dec_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.%s", cam_dec_map[j].gg); break; }
            }
        }
        else if (strncmp(key, "model.da3.cam_enc.", 18) == 0 || strncmp(key, "model.cam_enc.", 14) == 0) {
            const char *s = strncmp(key, "model.da3.", 10) == 0 ? key + 18 : key + 14;
            for (int j = 0; cam_enc_pose_map[j].st; j++) {
                if (strcmp(s, cam_enc_pose_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.%s", cam_enc_pose_map[j].gg); break; }
            }
            if (!gguf_name[0] && strncmp(s, "trunk.", 6) == 0) {
                int L_idx = 0;
                const char *rest = s + 6;
                while (*rest >= '0' && *rest <= '9') { L_idx = L_idx * 10 + (*rest - '0'); rest++; }
                if (*rest == '.') rest++;
                for (int j = 0; blk_map[j].st; j++) {
                    if (strcmp(rest, blk_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.cam_enc.trunk.%d.%s", L_idx, blk_map[j].gg); break; }
                }
            }
        }
        else if (strncmp(key, "model.da3.gs_head.", 18) == 0 || strncmp(key, "model.gs_head.", 14) == 0) {
            const char *s = strncmp(key, "model.da3.", 10) == 0 ? key + 18 : key + 14;
            if (strncmp(s, "images_merger.", 14) == 0) {
                const char *ms = s + 14;
                int idx = -1;
                if (ms[0] >= '0' && ms[0] <= '9') { idx = ms[0] - '0'; if (idx == 2) idx = 1; else if (idx == 4) idx = 2; else if (idx != 0) idx = -1; }
                if (idx >= 0 && ms[1] == '.') snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.merger.%d.%s", idx, ms + 2);
            } else if (strcmp(s, "norm.weight") == 0) strcpy(gguf_name, "da3.gsdpt.head.norm.weight");
            else if (strcmp(s, "norm.bias") == 0) strcpy(gguf_name, "da3.gsdpt.head.norm.bias");
            else if (strncmp(s, "projects.", 9) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.proj.%d.%s", s[9]-'0', s+11); }
            else if (strncmp(s, "resize_layers.", 14) == 0) {
                int idx = s[14] - '0'; const char *wb = s + 16;
                if (idx == 0) snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.upsample_0.%s", wb);
                else if (idx == 1) snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.upsample_1.%s", wb);
                else if (idx == 3) snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.downsample.%s", wb);
            } else if (strncmp(s, "scratch.", 8) == 0) {
                const char *ss = s + 8;
                for (int li = 1; li <= 4; li++) { char pfx[32]; snprintf(pfx, sizeof(pfx), "layer%d_rn.weight", li);
                    if (strcmp(ss, pfx) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.adapter.%d.weight", li-1); break; } }
                if (!gguf_name[0]) { for (int ri = 1; ri <= 4; ri++) { char pfx[32]; snprintf(pfx, sizeof(pfx), "refinenet%d.", ri); size_t plen = strlen(pfx);
                    if (strncmp(ss, pfx, plen) == 0) { const char *rn = ss + plen;
                        for (int j = 0; rn_map[j].st; j++) { if (strcmp(rn, rn_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.fuse.%d.%s", ri-1, rn_map[j].gg); break; } } break; } } }
                if (!gguf_name[0]) { static const struct { const char *st; const char *gg; } gsout_map[] = {
                    {"output_conv1.weight","da3.gsdpt.head.neck.weight"},{"output_conv1.bias","da3.gsdpt.head.neck.bias"},
                    {"output_conv2.0.weight","da3.gsdpt.head.out_0.weight"},{"output_conv2.0.bias","da3.gsdpt.head.out_0.bias"},
                    {"output_conv2.2.weight","da3.gsdpt.head.out_2.weight"},{"output_conv2.2.bias","da3.gsdpt.head.out_2.bias"},
                    {NULL, NULL} };
                    for (int j = 0; gsout_map[j].st; j++) { if (strcmp(ss, gsout_map[j].st) == 0) { strcpy(gguf_name, gsout_map[j].gg); break; } } }
            }
        }
        else if (hd_prefix && strncmp(key, hd_prefix, strlen(hd_prefix)) == 0) {
            const char *s = key + strlen(hd_prefix);
            if (strcmp(s, "norm.weight") == 0) strcpy(gguf_name, "da3.head.norm.weight");
            else if (strcmp(s, "norm.bias") == 0) strcpy(gguf_name, "da3.head.norm.bias");
            else if (strncmp(s, "projects.", 9) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.head.proj.%d.%s", s[9]-'0', s+11); }
            else if (strncmp(s, "resize_layers.", 14) == 0) {
                int idx = s[14] - '0'; const char *wb = s + 16;
                if (idx == 0) snprintf(gguf_name, sizeof(gguf_name), "da3.head.upsample_0.%s", wb);
                else if (idx == 1) snprintf(gguf_name, sizeof(gguf_name), "da3.head.upsample_1.%s", wb);
                else if (idx == 3) snprintf(gguf_name, sizeof(gguf_name), "da3.head.downsample.%s", wb);
            } else if (strncmp(s, "scratch.", 8) == 0) {
                const char *ss = s + 8;
                for (int li = 1; li <= 4; li++) { char pfx[32]; snprintf(pfx, sizeof(pfx), "layer%d_rn.weight", li);
                    if (strcmp(ss, pfx) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.head.adapter.%d.weight", li-1); break; } }
                if (!gguf_name[0]) { for (int ri = 1; ri <= 4; ri++) { char pfx[32]; snprintf(pfx, sizeof(pfx), "refinenet%d.", ri); size_t plen = strlen(pfx);
                    if (strncmp(ss, pfx, plen) == 0) { const char *rn = ss + plen;
                        for (int j = 0; rn_map[j].st; j++) { if (strcmp(rn, rn_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.head.fuse.%d.%s", ri-1, rn_map[j].gg); break; } } break; } } }
                if (!gguf_name[0]) { for (int ri = 1; ri <= 4; ri++) { char pfx[32]; snprintf(pfx, sizeof(pfx), "refinenet%d_aux.", ri); size_t plen = strlen(pfx);
                    if (strncmp(ss, pfx, plen) == 0) { const char *rn = ss + plen;
                        for (int j = 0; rn_map[j].st; j++) { if (strcmp(rn, rn_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_fuse.%d.%s", ri-1, rn_map[j].gg); break; } } break; } } }
                if (!gguf_name[0] && strncmp(ss, "output_conv1_aux.", 17) == 0) {
                    int level = ss[17]-'0';
                    if (level >= 0 && level < 4 && ss[18] == '.') { int ci = ss[19]-'0';
                        if (ci >= 0 && ci <= 4 && ss[20] == '.') snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_oc1.%d.%d.%s", level, ci, ss+21); } }
                if (!gguf_name[0] && strncmp(ss, "output_conv2_aux.", 17) == 0) {
                    int level = ss[17]-'0';
                    if (level >= 0 && level < 4 && ss[18] == '.') { int si = ss[19]-'0';
                        if (ss[20] == '.') { const char *wb = ss + 21;
                            if (si == 0) snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_oc2.%d.conv.%s", level, wb);
                            else if (si == 2) snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_oc2.%d.gn.%s", level, wb);
                            else if (si == 5) snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_oc2.%d.out.%s", level, wb); } } }
                if (!gguf_name[0] && !strstr(ss, "_aux")) {
                    static const struct { const char *st; const char *gg; } out_map[] = {
                        {"output_conv1.weight","da3.head.neck.weight"},{"output_conv1.bias","da3.head.neck.bias"},
                        {"output_conv2.0.weight","da3.head.out_0.weight"},{"output_conv2.0.bias","da3.head.out_0.bias"},
                        {"output_conv2.2.weight","da3.head.out_2.weight"},{"output_conv2.2.bias","da3.head.out_2.bias"},
                        {NULL, NULL} };
                    for (int j = 0; out_map[j].st; j++) { if (strcmp(ss, out_map[j].st) == 0) { strcpy(gguf_name, out_map[j].gg); break; } } }
            }
        }

        if (gguf_name[0]) {
            map[n].st_suffix = key;
            map[n].gguf_name = strdup(gguf_name);
            n++;
        }
    }

    *out_count = n;
    return map;
}

static qtensor da3s_tensor(const st_context *st, const st_name_map *map, int map_count,
                           const char *gguf_name) {
    qtensor t = {0};
    for (int m = 0; m < map_count; m++) {
        if (strcmp(map[m].gguf_name, gguf_name) != 0) continue;
        int si = safetensors_find(st, map[m].st_suffix);
        if (si < 0) break;
        t.data = safetensors_data(st, si);
        const char *dt = safetensors_dtype(st, si);
        if (strcmp(dt, "F32") == 0) t.type = 0;
        else if (strcmp(dt, "F16") == 0) t.type = 1;
        else if (strcmp(dt, "BF16") == 0) t.type = 30;
        else t.type = 0;
        t.n_dims = safetensors_ndims(st, si);
        const uint64_t *shape = safetensors_shape(st, si);
        for (int d = 0; d < t.n_dims; d++)
            t.dims[d] = shape[t.n_dims - 1 - d];
        t.n_cols = (int)t.dims[0];
        t.n_rows = (t.n_dims >= 2) ? (int)t.dims[1] : 1;
        break;
    }
    return t;
}

int hip_da3_load_safetensors(hip_da3_runner *r, const char *st_path, const char *config_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) return -1;

    if (r->verbose >= 1)
        fprintf(stderr, "hip_da3: safetensors: %d tensors\n", st->n_tensors);

    int map_count = 0;
    st_name_map *map = build_st_name_map(st, &map_count);
    if (r->verbose >= 1)
        fprintf(stderr, "hip_da3: mapped %d tensors\n", map_count);

    /* Detect model params (identical logic to CUDA version) */
    int embed_dim = 384, n_heads = 6, head_dim = 64, n_blocks = 12, ffn_hidden = 0;
    int head_features = 64;
    int head_oc[4] = {48, 96, 192, 384};
    int feature_layers[4] = {5, 7, 9, 11};
    int rope_start = 4, qknorm_start = 4;
    int has_swiglu = 0;

    if (config_path) {
        FILE *f = fopen(config_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            char *buf = (char *)malloc(sz); size_t nr = fread(buf, 1, sz, f); (void)nr; fclose(f);
            json_val *root = json_parse(buf, (int)sz);
            if (root) {
                json_val *cfg = json_obj_get(root, "config");
                if (cfg) {
                    json_val *anyview = json_obj_get(cfg, "anyview");
                    json_val *cfg_src = anyview ? anyview : cfg;
                    json_val *head = json_obj_get(cfg_src, "head");
                    if (head) {
                        json_val *v = json_obj_get(head, "features");
                        if (v && v->type == JSON_NUMBER) head_features = (int)v->num;
                        json_val *oc_arr = json_obj_get(head, "out_channels");
                        if (oc_arr && oc_arr->type == JSON_ARRAY)
                            for (int i = 0; i < 4 && i < oc_arr->arr.count; i++)
                                head_oc[i] = (int)oc_arr->arr.items[i].num;
                    }
                    json_val *net = json_obj_get(cfg_src, "net");
                    if (net) {
                        json_val *ol = json_obj_get(net, "out_layers");
                        if (ol && ol->type == JSON_ARRAY)
                            for (int i = 0; i < 4 && i < ol->arr.count; i++)
                                feature_layers[i] = (int)ol->arr.items[i].num;
                        json_val *rs = json_obj_get(net, "rope_start");
                        if (rs && rs->type == JSON_NUMBER) rope_start = (int)rs->num;
                        json_val *qs = json_obj_get(net, "qknorm_start");
                        if (qs && qs->type == JSON_NUMBER) qknorm_start = (int)qs->num;
                    }
                }
                json_free(root);
            }
            free(buf);
        }
    }

    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "patch_embed.proj.weight")) { embed_dim = (int)safetensors_shape(st, i)[0]; break; }
    }
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "attn.q_norm.weight") && !strstr(nm, "_aux")) {
            int hd = (int)safetensors_shape(st, i)[0]; n_heads = embed_dim / hd; head_dim = hd; break; }
    }
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "blocks.0.mlp.w12.weight")) { ffn_hidden = (int)safetensors_shape(st, i)[0] / 2; has_swiglu = 1; break; }
        if (strstr(nm, "blocks.0.mlp.fc1.weight")) { ffn_hidden = (int)safetensors_shape(st, i)[0]; break; }
    }
    n_blocks = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *p = strstr(nm, "blocks.");
        if (p && !strstr(nm, "_aux")) { p += 7; int blk = 0;
            while (*p >= '0' && *p <= '9') { blk = blk * 10 + (*p - '0'); p++; }
            if (blk + 1 > n_blocks) n_blocks = blk + 1; }
    }

    r->dim = embed_dim; r->n_heads = n_heads; r->head_dim = head_dim;
    r->n_blocks = n_blocks; r->ffn_hidden = ffn_hidden;
    r->patch_size = 14; r->image_size = 518; r->ln_eps = 1e-6f;
    r->rope_start = rope_start; r->qk_norm_start = qknorm_start;
    r->head_features = head_features; r->use_swiglu = has_swiglu;
    r->grid_h = r->image_size / r->patch_size; r->grid_w = r->grid_h;
    r->n_patches = r->grid_h * r->grid_w; r->n_tokens = r->n_patches + 1;
    for (int i = 0; i < 4; i++) r->feature_layers[i] = feature_layers[i];
    for (int i = 0; i < 4; i++) r->head_out_channels[i] = head_oc[i];
    r->image_mean[0] = 0.485f; r->image_mean[1] = 0.456f; r->image_mean[2] = 0.406f;
    r->image_std[0] = 0.229f; r->image_std[1] = 0.224f; r->image_std[2] = 0.225f;

    if (r->verbose >= 1) {
        fprintf(stderr, "hip_da3: model: dim=%d, n_heads=%d, head_dim=%d, n_blocks=%d, ffn=%d, swiglu=%d\n",
                embed_dim, n_heads, head_dim, n_blocks, ffn_hidden, has_swiglu);
    }

    /* Upload backbone embeddings */
    { qtensor t;
        t = da3s_tensor(st, map, map_count, "da3.cls_token");     r->d_cls_token = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.pos_embed");     r->d_pos_embed = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.patch_embed.weight"); r->d_patch_embed_w = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.patch_embed.bias");   r->d_patch_embed_b = upload_tensor_f32(&t);
    }

    /* Upload transformer blocks (F16 weights, no FP8 on RDNA4) */
    int nb = r->n_blocks;
    r->layers = (hip_da3_layer *)calloc((size_t)nb, sizeof(hip_da3_layer));
    for (int L = 0; L < nb; L++) {
        hip_da3_layer *ly = &r->layers[L];
        char name[128]; qtensor t;
#define ST_F32(field, fmt, ...) snprintf(name, sizeof(name), fmt, __VA_ARGS__); t = da3s_tensor(st, map, map_count, name); ly->field = upload_tensor_f32(&t);
#define ST_F16(field, fmt, ...) snprintf(name, sizeof(name), fmt, __VA_ARGS__); t = da3s_tensor(st, map, map_count, name); ly->field = upload_tensor_f32_as_f16(&t);
        ST_F32(ln1_w, "da3.blk.%d.ln1.weight", L)  ST_F32(ln1_b, "da3.blk.%d.ln1.bias", L)
        ST_F16(attn_qkv_w, "da3.blk.%d.attn_qkv.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.attn_qkv.weight", L); t = da3s_tensor(st, map, map_count, name); ly->qkv_rows = t.n_rows; ly->qkv_cols = t.n_cols;
        ST_F32(attn_qkv_b, "da3.blk.%d.attn_qkv.bias", L)
        ST_F32(attn_q_norm_w, "da3.blk.%d.attn_q_norm.weight", L)  ST_F32(attn_q_norm_b, "da3.blk.%d.attn_q_norm.bias", L)
        ST_F32(attn_k_norm_w, "da3.blk.%d.attn_k_norm.weight", L)  ST_F32(attn_k_norm_b, "da3.blk.%d.attn_k_norm.bias", L)
        ly->has_qk_norm = (ly->attn_q_norm_w != NULL);
        ST_F16(attn_out_w, "da3.blk.%d.attn_out.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.attn_out.weight", L); t = da3s_tensor(st, map, map_count, name); ly->out_rows = t.n_rows; ly->out_cols = t.n_cols;
        ST_F32(attn_out_b, "da3.blk.%d.attn_out.bias", L)
        ST_F32(ls1, "da3.blk.%d.ls1", L)  ST_F32(ls2, "da3.blk.%d.ls2", L)
        ST_F32(ln2_w, "da3.blk.%d.ln2.weight", L)  ST_F32(ln2_b, "da3.blk.%d.ln2.bias", L)
        ST_F16(ffn_gate_up_w, "da3.blk.%d.ffn_gate_up.weight", L)
        if (ly->ffn_gate_up_w) { snprintf(name, sizeof(name), "da3.blk.%d.ffn_gate_up.weight", L); t = da3s_tensor(st, map, map_count, name); ly->ffn_gu_rows = t.n_rows; ly->ffn_gu_cols = t.n_cols; ly->has_swiglu = 1; }
        ST_F32(ffn_gate_up_b, "da3.blk.%d.ffn_gate_up.bias", L)
        ST_F16(ffn_up_w, "da3.blk.%d.ffn_up.weight", L)
        if (ly->ffn_up_w) { snprintf(name, sizeof(name), "da3.blk.%d.ffn_up.weight", L); t = da3s_tensor(st, map, map_count, name); ly->ffn_up_rows = t.n_rows; ly->ffn_up_cols = t.n_cols; }
        ST_F32(ffn_up_b, "da3.blk.%d.ffn_up.bias", L)
        ST_F16(ffn_down_w, "da3.blk.%d.ffn_down.weight", L)
        snprintf(name, sizeof(name), "da3.blk.%d.ffn_down.weight", L); t = da3s_tensor(st, map, map_count, name); ly->ffn_down_rows = t.n_rows; ly->ffn_down_cols = t.n_cols;
        ST_F32(ffn_down_b, "da3.blk.%d.ffn_down.bias", L)
#undef ST_F32
#undef ST_F16
    }

    /* Upload DPT head weights (same pattern as GGUF path, using safetensors lookups) */
    /* NOTE: For brevity, the remaining safetensors loading follows the exact same
     * pattern as the GGUF path (above) and the CUDA version - uploading DPT head,
     * CameraDec, CameraEnc, DPT Aux, GSDPT weights. The code is structurally
     * identical but uses da3s_tensor() instead of da3g_tensor(). */
    {
        qtensor t; char name[128];
        dpt_gpu_weights *dw = &r->dpt_w;
        t = da3s_tensor(st, map, map_count, "da3.head.norm.weight"); dw->norm_w = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.norm.bias");   dw->norm_b = upload_tensor_f32(&t);
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.proj.%d.weight", i); t = da3s_tensor(st, map, map_count, name); dw->proj_w[i] = upload_tensor_f32_as_f16(&t); dw->proj_rows[i] = t.n_rows;
            snprintf(name, sizeof(name), "da3.head.proj.%d.bias", i); t = da3s_tensor(st, map, map_count, name); dw->proj_b[i] = upload_tensor_f32(&t);
        }
        { int oc0 = r->head_out_channels[0];
            t = da3s_tensor(st, map, map_count, "da3.head.upsample_0.weight"); dw->upsample_0_w = upload_deconv_weight_f16(&t, oc0, oc0, 4, 4);
            t = da3s_tensor(st, map, map_count, "da3.head.upsample_0.bias"); dw->upsample_0_b = upload_tensor_f32(&t);
            int oc1 = r->head_out_channels[1];
            t = da3s_tensor(st, map, map_count, "da3.head.upsample_1.weight"); dw->upsample_1_w = upload_deconv_weight_f16(&t, oc1, oc1, 2, 2);
            t = da3s_tensor(st, map, map_count, "da3.head.upsample_1.bias"); dw->upsample_1_b = upload_tensor_f32(&t);
        }
        t = da3s_tensor(st, map, map_count, "da3.head.downsample.weight"); dw->downsample_w = upload_tensor_f32_as_f16(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.downsample.bias"); dw->downsample_b = upload_tensor_f32(&t);
        for (int i = 0; i < 4; i++) { snprintf(name, sizeof(name), "da3.head.adapter.%d.weight", i); t = da3s_tensor(st, map, map_count, name); dw->adapter_w[i] = upload_tensor_f32_as_f16(&t); }
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.weight", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_out_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.bias", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_out_b[i] = upload_tensor_f32(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.weight", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu1_c1_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.bias", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu1_c1_b[i] = upload_tensor_f32(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.weight", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu1_c2_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.bias", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu1_c2_b[i] = upload_tensor_f32(&t);
            dw->has_rcu1[i] = (dw->fuse_rcu1_c1_w[i] != NULL);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.weight", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu2_c1_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.bias", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu2_c1_b[i] = upload_tensor_f32(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.weight", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu2_c2_w[i] = upload_tensor_f32_as_f16(&t);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.bias", i); t = da3s_tensor(st, map, map_count, name); dw->fuse_rcu2_c2_b[i] = upload_tensor_f32(&t);
            dw->has_rcu2[i] = (dw->fuse_rcu2_c1_w[i] != NULL);
        }
        t = da3s_tensor(st, map, map_count, "da3.head.neck.weight"); dw->neck_w = upload_tensor_f32_as_f16(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.neck.bias"); dw->neck_b = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.out_0.weight"); dw->out_0_w = upload_tensor_f32_as_f16(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.out_0.bias"); dw->out_0_b = upload_tensor_f32(&t);
        dw->out_mid = t.n_cols;
        t = da3s_tensor(st, map, map_count, "da3.head.out_2.weight"); dw->out_2_w = upload_tensor_f32(&t);
        t = da3s_tensor(st, map, map_count, "da3.head.out_2.bias"); dw->out_2_b = upload_tensor_f32(&t);
    }

    /* Upload CameraDec weights */
    {
        qtensor t;
        t = da3s_tensor(st, map, map_count, "da3.backbone_norm.weight");
        if (t.data) {
            r->cam_dec.backbone_norm_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.backbone_norm.bias");
            r->cam_dec.backbone_norm_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.mlp.0.weight");
            r->cam_dec.mlp_w[0] = upload_tensor_f32_as_f16(&t);
            r->cam_dec.mlp_dim = t.n_rows;
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.mlp.0.bias");
            r->cam_dec.mlp_b[0] = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.mlp.2.weight");
            r->cam_dec.mlp_w[1] = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.mlp.2.bias");
            r->cam_dec.mlp_b[1] = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_t.weight");
            r->cam_dec.fc_t_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_t.bias");
            r->cam_dec.fc_t_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_qvec.weight");
            r->cam_dec.fc_qvec_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_qvec.bias");
            r->cam_dec.fc_qvec_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_fov.weight");
            r->cam_dec.fc_fov_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_dec.fc_fov.bias");
            r->cam_dec.fc_fov_b = upload_tensor_f32(&t);
            r->cam_dec.loaded = 1;
            if (r->verbose >= 1)
                fprintf(stderr, "hip_da3: CameraDec weights loaded (mlp_dim=%d)\n", r->cam_dec.mlp_dim);
        }
    }

    /* Upload CameraEnc weights */
    {
        qtensor t; char name[128];
        t = da3s_tensor(st, map, map_count, "da3.cam_enc.fc1.weight");
        if (t.data) {
            r->cam_enc.fc1_w = upload_tensor_f32_as_f16(&t);
            r->cam_enc.trunk_dim = t.n_rows;
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.fc1.bias");
            r->cam_enc.fc1_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.fc2.weight");
            r->cam_enc.fc2_w = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.fc2.bias");
            r->cam_enc.fc2_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.trunk_norm.weight");
            r->cam_enc.trunk_norm_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.trunk_norm.bias");
            r->cam_enc.trunk_norm_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.token_norm.weight");
            r->cam_enc.token_norm_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.cam_enc.token_norm.bias");
            r->cam_enc.token_norm_b = upload_tensor_f32(&t);
            r->cam_enc.n_trunk_blocks = 0;
            for (int L = 0; L < 8; L++) {
                snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.ln1.weight", L);
                t = da3s_tensor(st, map, map_count, name);
                if (!t.data) break;
                r->cam_enc.n_trunk_blocks = L + 1;
            }
            int ntb = r->cam_enc.n_trunk_blocks;
            if (ntb > 0) {
                r->cam_enc.trunk = (hip_da3_layer *)calloc((size_t)ntb, sizeof(hip_da3_layer));
                for (int L = 0; L < ntb; L++) {
                    hip_da3_layer *ly = &r->cam_enc.trunk[L];
#define CE_F32(field, fmt, ...) snprintf(name, sizeof(name), fmt, __VA_ARGS__); t = da3s_tensor(st, map, map_count, name); ly->field = upload_tensor_f32(&t);
#define CE_F16(field, fmt, ...) snprintf(name, sizeof(name), fmt, __VA_ARGS__); t = da3s_tensor(st, map, map_count, name); ly->field = upload_tensor_f32_as_f16(&t);
                    CE_F32(ln1_w, "da3.cam_enc.trunk.%d.ln1.weight", L)
                    CE_F32(ln1_b, "da3.cam_enc.trunk.%d.ln1.bias", L)
                    CE_F16(attn_qkv_w, "da3.cam_enc.trunk.%d.attn_qkv.weight", L)
                    snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.attn_qkv.weight", L);
                    t = da3s_tensor(st, map, map_count, name);
                    ly->qkv_rows = t.n_rows; ly->qkv_cols = t.n_cols;
                    CE_F32(attn_qkv_b, "da3.cam_enc.trunk.%d.attn_qkv.bias", L)
                    CE_F16(attn_out_w, "da3.cam_enc.trunk.%d.attn_out.weight", L)
                    snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.attn_out.weight", L);
                    t = da3s_tensor(st, map, map_count, name);
                    ly->out_rows = t.n_rows; ly->out_cols = t.n_cols;
                    CE_F32(attn_out_b, "da3.cam_enc.trunk.%d.attn_out.bias", L)
                    CE_F32(ls1, "da3.cam_enc.trunk.%d.ls1", L)
                    CE_F32(ls2, "da3.cam_enc.trunk.%d.ls2", L)
                    CE_F32(ln2_w, "da3.cam_enc.trunk.%d.ln2.weight", L)
                    CE_F32(ln2_b, "da3.cam_enc.trunk.%d.ln2.bias", L)
                    CE_F16(ffn_up_w, "da3.cam_enc.trunk.%d.ffn_up.weight", L)
                    if (ly->ffn_up_w) { snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.ffn_up.weight", L); t = da3s_tensor(st, map, map_count, name); ly->ffn_up_rows = t.n_rows; ly->ffn_up_cols = t.n_cols; }
                    CE_F32(ffn_up_b, "da3.cam_enc.trunk.%d.ffn_up.bias", L)
                    CE_F16(ffn_down_w, "da3.cam_enc.trunk.%d.ffn_down.weight", L)
                    snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.ffn_down.weight", L); t = da3s_tensor(st, map, map_count, name); ly->ffn_down_rows = t.n_rows; ly->ffn_down_cols = t.n_cols;
                    CE_F32(ffn_down_b, "da3.cam_enc.trunk.%d.ffn_down.bias", L)
                    ly->has_qk_norm = 0; ly->has_swiglu = 0;
#undef CE_F32
#undef CE_F16
                }
            }
            r->cam_enc.loaded = 1;
            if (r->verbose >= 1)
                fprintf(stderr, "hip_da3: CameraEnc weights loaded (%d trunk blocks, dim=%d)\n", ntb, r->cam_enc.trunk_dim);
        }
    }

    /* Upload DPT Aux Branch weights */
    {
        qtensor t; char name[128];
        t = da3s_tensor(st, map, map_count, "da3.head.aux_fuse.0.out.weight");
        if (t.data) {
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.out.weight", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_out_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.out.bias", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_out_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv1.weight", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_rcu1_c1_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv1.bias", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_rcu1_c1_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv2.weight", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_rcu1_c2_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv2.bias", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_rcu1_c2_b[i] = upload_tensor_f32(&t);
                r->dpt_aux.has_rcu1[i] = (r->dpt_aux.fuse_rcu1_c1_w[i] != NULL);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv1.weight", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_rcu2_c1_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv1.bias", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_rcu2_c1_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv2.weight", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_rcu2_c2_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv2.bias", i); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.fuse_rcu2_c2_b[i] = upload_tensor_f32(&t);
                r->dpt_aux.has_rcu2[i] = (r->dpt_aux.fuse_rcu2_c1_w[i] != NULL);
            }
            for (int lv = 0; lv < 4; lv++) {
                r->dpt_aux.oc1_count[lv] = 0;
                for (int ci = 0; ci < 5; ci++) {
                    snprintf(name, sizeof(name), "da3.head.aux_oc1.%d.%d.weight", lv, ci);
                    t = da3s_tensor(st, map, map_count, name);
                    if (!t.data) break;
                    r->dpt_aux.oc1_w[lv][ci] = upload_tensor_f32(&t);
                    r->dpt_aux.oc1_co[lv][ci] = (t.n_dims == 4) ? (int)t.dims[3] : t.n_rows;
                    r->dpt_aux.oc1_ci[lv][ci] = (t.n_dims == 4) ? (int)t.dims[2] : (t.n_cols / 9);
                    snprintf(name, sizeof(name), "da3.head.aux_oc1.%d.%d.bias", lv, ci);
                    t = da3s_tensor(st, map, map_count, name);
                    r->dpt_aux.oc1_b[lv][ci] = upload_tensor_f32(&t);
                    r->dpt_aux.oc1_count[lv] = ci + 1;
                }
            }
            for (int lv = 0; lv < 4; lv++) {
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.conv.weight", lv); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.oc2_conv_w[lv] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.conv.bias", lv); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.oc2_conv_b[lv] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.gn.weight", lv); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.oc2_gn_w[lv] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.gn.bias", lv); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.oc2_gn_b[lv] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.out.weight", lv); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.oc2_out_w[lv] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.out.bias", lv); t = da3s_tensor(st, map, map_count, name); r->dpt_aux.oc2_out_b[lv] = upload_tensor_f32(&t);
            }
            r->dpt_aux.loaded = 1;
            if (r->verbose >= 1)
                fprintf(stderr, "hip_da3: DPT Aux Branch weights loaded\n");
        }
    }

    /* Upload GSDPT weights */
    {
        qtensor t; char name[128];
        t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.proj.0.weight");
        if (t.data) {
            dpt_gpu_weights *gdw = &r->gsdpt.dpt;
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.norm.weight"); gdw->norm_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.norm.bias"); gdw->norm_b = upload_tensor_f32(&t);
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.gsdpt.head.proj.%d.weight", i); t = da3s_tensor(st, map, map_count, name); gdw->proj_w[i] = upload_tensor_f32_as_f16(&t); gdw->proj_rows[i] = t.n_rows;
                snprintf(name, sizeof(name), "da3.gsdpt.head.proj.%d.bias", i); t = da3s_tensor(st, map, map_count, name); gdw->proj_b[i] = upload_tensor_f32(&t);
            }
            { int oc0 = r->head_out_channels[0];
                t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.upsample_0.weight"); if (t.data) gdw->upsample_0_w = upload_deconv_weight_f16(&t, oc0, oc0, 4, 4);
                t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.upsample_0.bias"); gdw->upsample_0_b = upload_tensor_f32(&t);
                int oc1 = r->head_out_channels[1];
                t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.upsample_1.weight"); if (t.data) gdw->upsample_1_w = upload_deconv_weight_f16(&t, oc1, oc1, 2, 2);
                t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.upsample_1.bias"); gdw->upsample_1_b = upload_tensor_f32(&t);
            }
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.downsample.weight"); gdw->downsample_w = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.downsample.bias"); gdw->downsample_b = upload_tensor_f32(&t);
            for (int i = 0; i < 4; i++) { snprintf(name, sizeof(name), "da3.gsdpt.head.adapter.%d.weight", i); t = da3s_tensor(st, map, map_count, name); gdw->adapter_w[i] = upload_tensor_f32_as_f16(&t); }
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.out.weight", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_out_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.out.bias", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_out_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv1.weight", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_rcu1_c1_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv1.bias", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_rcu1_c1_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv2.weight", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_rcu1_c2_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv2.bias", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_rcu1_c2_b[i] = upload_tensor_f32(&t);
                gdw->has_rcu1[i] = (gdw->fuse_rcu1_c1_w[i] != NULL);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv1.weight", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_rcu2_c1_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv1.bias", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_rcu2_c1_b[i] = upload_tensor_f32(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv2.weight", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_rcu2_c2_w[i] = upload_tensor_f32_as_f16(&t);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv2.bias", i); t = da3s_tensor(st, map, map_count, name); gdw->fuse_rcu2_c2_b[i] = upload_tensor_f32(&t);
                gdw->has_rcu2[i] = (gdw->fuse_rcu2_c1_w[i] != NULL);
            }
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.neck.weight"); gdw->neck_w = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.neck.bias"); gdw->neck_b = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.out_0.weight"); gdw->out_0_w = upload_tensor_f32_as_f16(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.out_0.bias"); gdw->out_0_b = upload_tensor_f32(&t);
            gdw->out_mid = t.n_cols;
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.out_2.weight"); gdw->out_2_w = upload_tensor_f32(&t);
            t = da3s_tensor(st, map, map_count, "da3.gsdpt.head.out_2.bias"); gdw->out_2_b = upload_tensor_f32(&t);
            r->gsdpt.gs_out_channels = t.n_cols;
            r->gsdpt.merger_ci[0] = 3;  r->gsdpt.merger_co[0] = 32;
            r->gsdpt.merger_ci[1] = 32; r->gsdpt.merger_co[1] = 64;
            r->gsdpt.merger_ci[2] = 64; r->gsdpt.merger_co[2] = 128;
            for (int mi = 0; mi < 3; mi++) {
                snprintf(name, sizeof(name), "da3.gsdpt.merger.%d.weight", mi);
                t = da3s_tensor(st, map, map_count, name);
                if (t.data) {
                    r->gsdpt.merger_w[mi] = upload_tensor_f32(&t);
                    r->gsdpt.merger_co[mi] = (t.n_dims == 4) ? (int)t.dims[3] : t.n_rows;
                    snprintf(name, sizeof(name), "da3.gsdpt.merger.%d.bias", mi);
                    t = da3s_tensor(st, map, map_count, name);
                    r->gsdpt.merger_b[mi] = upload_tensor_f32(&t);
                }
            }
            r->gsdpt.loaded = 1;
            if (r->verbose >= 1)
                fprintf(stderr, "hip_da3: GSDPT weights loaded (out_ch=%d, out_mid=%d, merger=%d→%d→%d)\n",
                        r->gsdpt.gs_out_channels, gdw->out_mid,
                        r->gsdpt.merger_co[0], r->gsdpt.merger_co[1], r->gsdpt.merger_co[2]);
        }
    }

    /* Allocate scratch buffers */
    int nt = r->n_tokens, dim = r->dim, np = r->n_patches, gh = r->grid_h;
    int max_ffn = r->use_swiglu ? 2 * r->ffn_hidden : 4 * dim;
    CHECK_HIP(hipMalloc(&r->d_hidden, (size_t)nt*dim*sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_hidden2, (size_t)nt*dim*sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_ln_buf, (size_t)nt*dim*sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_qkv, (size_t)nt*3*dim*sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_attn_out, (size_t)nt*dim*sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_ffn_buf, (size_t)nt*max_ffn*sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_ffn_mid, (size_t)nt*r->ffn_hidden*sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_proj_out, (size_t)nt*dim*sizeof(float)));
    CHECK_HIP(hipMalloc(&r->d_pos_y, (size_t)nt*sizeof(int)));
    CHECK_HIP(hipMalloc(&r->d_pos_x, (size_t)nt*sizeof(int)));
    CHECK_HIP(hipMalloc(&r->d_img_norm, (size_t)3*r->image_size*r->image_size*sizeof(float)));
    for (int i = 0; i < 4; i++) CHECK_HIP(hipMalloc(&r->d_features[i], (size_t)nt*dim*sizeof(float)));
    { int feat = r->head_features, oc_max = r->head_out_channels[3];
        int sp_h[4], sp_w[4];
        sp_h[0] = sp_w[0] = (gh-1)*4+4; sp_h[1] = sp_w[1] = (gh-1)*2+2;
        sp_h[2] = sp_w[2] = gh; sp_h[3] = sp_w[3] = (gh+2-3)/2+1;
        int max_hw = sp_h[0]*sp_w[0];
        size_t large_sz = (size_t)feat*max_hw;
        if (large_sz < (size_t)np*2*dim) large_sz = (size_t)np*2*dim;
        CHECK_HIP(hipMalloc(&r->d_dpt_cat, large_sz*sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_ln, large_sz*sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_proj, (size_t)np*oc_max*sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_chw, (size_t)oc_max*gh*gh*sizeof(float)));
        for (int i = 0; i < 4; i++) { int oc_i = r->head_out_channels[i];
            CHECK_HIP(hipMalloc(&r->d_dpt_spatial[i], (size_t)oc_i*sp_h[i]*sp_w[i]*sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_dpt_adapted[i], (size_t)feat*sp_h[i]*sp_w[i]*sizeof(float))); }
        CHECK_HIP(hipMalloc(&r->d_dpt_fused, large_sz*sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_tmp, large_sz*sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_tmp2, large_sz*sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_dpt_out, (size_t)2*max_hw*sizeof(float)));
        if (r->dpt_aux.loaded) {
            for (int i = 0; i < 4; i++)
                CHECK_HIP(hipMalloc(&r->d_aux_out[i], (size_t)7*sp_h[0]*sp_w[0]*sizeof(float)));
            CHECK_HIP(hipMalloc(&r->d_aux_scratch, (size_t)256*max_hw*sizeof(float)));
            if (r->verbose >= 1) fprintf(stderr, "hip_da3: DPT Aux scratch allocated\n");
        }
        if (r->gsdpt.loaded) {
            int mg_h = r->image_size, mg_w = r->image_size;
            for (int s = 0; s < 3; s++) { mg_h = (mg_h + 2*1 - 3)/2 + 1; mg_w = (mg_w + 2*1 - 3)/2 + 1; }
            r->gs_merger_h = mg_h; r->gs_merger_w = mg_w;
            CHECK_HIP(hipMalloc(&r->d_gs_merged, (size_t)128*mg_h*mg_w*sizeof(float)));
            int gs_oc = r->gsdpt.gs_out_channels; if (gs_oc < 2) gs_oc = 38;
            CHECK_HIP(hipMalloc(&r->d_gs_out, (size_t)gs_oc*max_hw*sizeof(float)));
            if (r->verbose >= 1)
                fprintf(stderr, "hip_da3: GSDPT scratch allocated (%d channels, merger=%dx%d)\n", gs_oc, mg_h, mg_w);
        }
    }
    int *py = (int *)calloc((size_t)nt, sizeof(int));
    int *px = (int *)calloc((size_t)nt, sizeof(int));
    for (int p = 0; p < np; p++) { py[1+p] = p/r->grid_w; px[1+p] = p%r->grid_w; }
    hipMemcpy(r->d_pos_y, py, (size_t)nt*sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(r->d_pos_x, px, (size_t)nt*sizeof(int), hipMemcpyHostToDevice);
    free(py); free(px);
    r->h_output = (float *)malloc((size_t)nt*dim*sizeof(float));
    r->cpu_model = NULL;
    r->loaded = 1;

    if (r->verbose >= 1) {
        fprintf(stderr, "hip_da3: loaded %d blocks, dim=%d, tokens=%d, swiglu=%d (safetensors)\n",
                nb, dim, nt, r->use_swiglu);
        fprintf(stderr, "hip_da3: modules: cam_dec=%d cam_enc=%d dpt_aux=%d gsdpt=%d\n",
                r->cam_dec.loaded, r->cam_enc.loaded, r->dpt_aux.loaded, r->gsdpt.loaded);
    }

    for (int i = 0; i < map_count; i++) free((void *)map[i].gguf_name);
    free(map);
    safetensors_close(st);
    return 0;
}

/* ======================================================================== */
/* Kernel launch helpers                                                    */
/* ======================================================================== */

#define KL(fn, gx, gy, gz, bx, by, bz, smem, stream, args) \
    hipModuleLaunchKernel(fn, gx, gy, gz, bx, by, bz, smem, stream, args, NULL)

static void kl_layernorm(hip_da3_runner *r, void *dst, void *src,
                          void *w, void *b, int n_tok, int dim) {
    float eps = r->ln_eps;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    KL(r->fn_layernorm_f32, (unsigned)n_tok, 1, 1, 256, 1, 1,
       256 * sizeof(float), r->stream, args);
}

static void kl_gemm(hip_da3_runner *r, void *Y, void *W_f16,
                     void *X, void *bias, int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W_f16, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    KL(r->fn_gemm_tiled_f16_f32, gx, gy, 1, 16, 16, 1, 0, r->stream, args);
}

static void kl_qk_layernorm(hip_da3_runner *r, void *vec, void *w, void *b,
                               int n_tok, int n_heads, int head_dim, int stride) {
    if (!w) return;
    float eps = r->ln_eps;
    int grid = n_tok * n_heads;
    int threads = 64;
    void *args[] = {&vec, &w, &b, &n_tok, &n_heads, &head_dim, &stride, &eps};
    KL(r->fn_qk_layernorm_f32, (unsigned)grid, 1, 1, (unsigned)threads, 1, 1,
       (unsigned)(threads * sizeof(float)), r->stream, args);
}

static void kl_rope_2d(hip_da3_runner *r, void *vec, void *pos_y, void *pos_x,
                         int n_tok, int n_heads, int head_dim, int stride) {
    int quarter = head_dim / 4;
    int threads = n_heads * quarter;
    float freq_base = 10000.0f;
    void *args[] = {&vec, &pos_y, &pos_x, &n_tok, &n_heads, &head_dim, &stride, &freq_base};
    KL(r->fn_rope_2d_f32, (unsigned)n_tok, 1, 1, (unsigned)threads, 1, 1, 0, r->stream, args);
}

static void kl_kv_transpose(hip_da3_runner *r, void *K_t, void *V_t,
                              void *qkv, int n_tok, int dim, int n_heads, int head_dim) {
    int total = n_tok * dim;
    int grid = (total + 255) / 256;
    void *args[] = {&K_t, &V_t, &qkv, &n_tok, &dim, &n_heads, &head_dim};
    KL(r->fn_kv_transpose, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_attn_prefill(hip_da3_runner *r, void *out, void *qkv,
                              void *K_t, void *V_t,
                              int n_tok, int dim, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    void *args[] = {&out, &qkv, &K_t, &V_t, &n_tok, &dim, &n_heads, &head_dim, &scale};
    int bq = 64, bkv = 16;
    unsigned smem_size = (unsigned)(2 * bkv * head_dim * sizeof(float));
    unsigned gy = (unsigned)((n_tok + bq - 1) / bq);
    KL(r->fn_flash_attn_tiled_f32, (unsigned)n_heads, gy, 1, (unsigned)bq, 1, 1,
       smem_size, r->stream, args);
}

static void kl_swiglu(hip_da3_runner *r, void *dst, void *gate_up, int hidden, int n_tok) {
    int total = hidden * n_tok;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &gate_up, &hidden, &n_tok};
    KL(r->fn_swiglu_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_gelu(hip_da3_runner *r, void *x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    KL(r->fn_gelu_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_layerscale_add(hip_da3_runner *r, void *hidden, void *proj, void *gamma, int dim, int n) {
    if (!gamma) {
        int grid = (n + 255) / 256;
        void *args[] = {&hidden, &proj, &n};
        KL(r->fn_add_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
        return;
    }
    int grid = (n + 255) / 256;
    void *args[] = {&hidden, &proj, &gamma, &dim, &n};
    KL(r->fn_layerscale_add_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_cls_concat(hip_da3_runner *r, void *dst, void *src, int np, int dim) {
    int total = np * 2 * dim;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &src, &np, &dim};
    KL(r->fn_dpt_cls_concat, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_tok_to_chw(hip_da3_runner *r, void *dst, void *src, int C, int gH, int gW) {
    int total = C * gH * gW;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &src, &C, &gH, &gW};
    KL(r->fn_dpt_tok_to_chw, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_conv2d(hip_da3_runner *r, void *dst, void *src, void *w, void *b,
                        int H, int W, int Ci, int Co, int kH, int kW, int stride, int pad) {
    int Ho = (H + 2*pad - kH) / stride + 1;
    int Wo = (W + 2*pad - kW) / stride + 1;
    int total = Co * Ho * Wo;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &src, &w, &b, &H, &W, &Ci, &Co, &kH, &kW, &stride, &pad};
    KL(r->fn_conv2d_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

/* Conv2d via scalar kernel. Weight w_f16 is FP16 [Co, Ci*kH*kW].
 * Dequant to F32 on host, upload, use scalar conv2d_f32 kernel. */
static void kl_conv_gemm(hip_da3_runner *r, void *dst, void *src,
                           void *w_f16, void *bias, int H, int W,
                           int Ci, int Co, int kH, int kW, int stride, int pad) {
    int K = Ci * kH * kW;
    int n = Co * K;
    uint16_t *h_w16 = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
    float *h_w32 = (float *)malloc((size_t)n * sizeof(float));
    hipMemcpy(h_w16, w_f16, (size_t)n * sizeof(uint16_t), hipMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) h_w32[i] = ggml_fp16_to_fp32(h_w16[i]);
    void *d_w32 = NULL;
    hipMalloc(&d_w32, (size_t)n * sizeof(float));
    hipMemcpy(d_w32, h_w32, (size_t)n * sizeof(float), hipMemcpyHostToDevice);
    free(h_w16); free(h_w32);
    kl_conv2d(r, dst, src, d_w32, bias, H, W, Ci, Co, kH, kW, stride, pad);
    hipFree(d_w32);
}

static void kl_deconv_gemm_scatter(hip_da3_runner *r, void *dst, void *X, void *W_f16,
                                     void *bias, void *scratch, int Ci, int Co,
                                     int Hi, int Wi, int kH, int kW, int stride) {
    int N = kH * kW * Co;
    void *null_bias = NULL;
    kl_gemm(r, scratch, W_f16, X, null_bias, N, Ci, Hi * Wi);
    int Ho = (Hi - 1) * stride + kH;
    int Wo = (Wi - 1) * stride + kW;
    int total = Co * Ho * Wo;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &scratch, &bias, &Co, &Hi, &Wi, &Ho, &Wo, &kH, &kW, &stride, &stride};
    KL(r->fn_deconv_scatter_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_bilinear(hip_da3_runner *r, void *dst, void *src, int C, int Hi, int Wi, int Ho, int Wo) {
    int total = C * Ho * Wo;
    int grid = (total + 255) / 256;
    void *args[] = {&dst, &src, &C, &Hi, &Wi, &Ho, &Wo};
    KL(r->fn_bilinear_upsample_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_relu_inplace(hip_da3_runner *r, void *x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    KL(r->fn_relu_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_add_inplace(hip_da3_runner *r, void *dst, void *src, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&dst, &src, &n};
    KL(r->fn_add_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_channel_layernorm(hip_da3_runner *r, void *dst, void *src,
                                   void *w, void *b, int C, int HW) {
    float eps = 1e-5f;
    int grid = (HW + 255) / 256;
    void *args[] = {&dst, &src, &w, &b, &C, &HW, &eps};
    KL(r->fn_channel_layernorm_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

static void kl_silu_inplace(hip_da3_runner *r, void *x, int n) {
    int grid = (n + 255) / 256;
    void *args[] = {&x, &n};
    KL(r->fn_silu_f32, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args);
}

/* RCU: relu -> conv3x3 -> relu -> conv3x3 + residual */
static void kl_rcu(hip_da3_runner *r, void *out, void *x,
                     void *c1w, void *c1b, void *c2w, void *c2b,
                     int C, int H, int W) {
    int sz = C * H * W;
    hipMemcpyAsync(r->d_dpt_tmp2, x, (size_t)sz * sizeof(float), hipMemcpyDeviceToDevice, r->stream);
    kl_relu_inplace(r, r->d_dpt_tmp2, sz);
    kl_conv_gemm(r, r->d_dpt_tmp, r->d_dpt_tmp2, c1w, c1b, H, W, C, C, 3, 3, 1, 1);
    kl_relu_inplace(r, r->d_dpt_tmp, sz);
    kl_conv_gemm(r, out, r->d_dpt_tmp, c2w, c2b, H, W, C, C, 3, 3, 1, 1);
    kl_add_inplace(r, out, x, sz);
}

/* GPU RefineNet fusion block (generalized with explicit weight pointers) */
static void gpu_refinenet_w(hip_da3_runner *r,
                              void *feat, int fH, int fW,
                              void *deeper, int dH, int dW, int features,
                              void *result_buf,
                              void *fuse_out_w, void *fuse_out_b,
                              void *rcu1_c1_w, void *rcu1_c1_b,
                              void *rcu1_c2_w, void *rcu1_c2_b,
                              void *rcu2_c1_w, void *rcu2_c1_b,
                              void *rcu2_c2_w, void *rcu2_c2_b,
                              int has_rcu1, int has_rcu2) {
    int sz = features * fH * fW;
    if (deeper) kl_bilinear(r, r->d_dpt_cat, deeper, features, dH, dW, fH, fW);
    hipMemcpyAsync(result_buf, feat, (size_t)sz * sizeof(float), hipMemcpyDeviceToDevice, r->stream);
    if (deeper) {
        if (has_rcu1) {
            kl_rcu(r, r->d_dpt_ln, r->d_dpt_cat, rcu1_c1_w, rcu1_c1_b, rcu1_c2_w, rcu1_c2_b, features, fH, fW);
            kl_add_inplace(r, result_buf, r->d_dpt_ln, sz);
        } else {
            kl_add_inplace(r, result_buf, r->d_dpt_cat, sz);
        }
    }
    if (has_rcu2) {
        kl_rcu(r, r->d_dpt_cat, result_buf, rcu2_c1_w, rcu2_c1_b, rcu2_c2_w, rcu2_c2_b, features, fH, fW);
        hipMemcpyAsync(result_buf, r->d_dpt_cat, (size_t)sz * sizeof(float), hipMemcpyDeviceToDevice, r->stream);
    }
    kl_conv_gemm(r, r->d_dpt_cat, result_buf, fuse_out_w, fuse_out_b, fH, fW, features, features, 1, 1, 1, 0);
    hipMemcpyAsync(result_buf, r->d_dpt_cat, (size_t)sz * sizeof(float), hipMemcpyDeviceToDevice, r->stream);
}

static void gpu_refinenet(hip_da3_runner *r, int stage, void *feat, int fH, int fW,
                            void *deeper, int dH, int dW, int features, void *result_buf) {
    dpt_gpu_weights *dw = &r->dpt_w;
    gpu_refinenet_w(r, feat, fH, fW, deeper, dH, dW, features, result_buf,
                     dw->fuse_out_w[stage], dw->fuse_out_b[stage],
                     dw->fuse_rcu1_c1_w[stage], dw->fuse_rcu1_c1_b[stage],
                     dw->fuse_rcu1_c2_w[stage], dw->fuse_rcu1_c2_b[stage],
                     dw->fuse_rcu2_c1_w[stage], dw->fuse_rcu2_c1_b[stage],
                     dw->fuse_rcu2_c2_w[stage], dw->fuse_rcu2_c2_b[stage],
                     dw->has_rcu1[stage], dw->has_rcu2[stage]);
}

/* CameraDec: backbone_norm(CLS) -> MLP -> 3 linear heads -> pose[9] (CPU) */
static void run_camera_dec(hip_da3_runner *r, float *pose_out) {
    int dim = r->dim;
    int mlp_dim = r->cam_dec.mlp_dim;
    kl_layernorm(r, r->d_ln_buf, r->d_hidden, r->cam_dec.backbone_norm_w, r->cam_dec.backbone_norm_b, 1, dim);
    kl_gemm(r, r->d_attn_out, r->cam_dec.mlp_w[0], r->d_ln_buf, r->cam_dec.mlp_b[0], mlp_dim, dim, 1);
    kl_gelu(r, r->d_attn_out, mlp_dim);
    kl_gemm(r, r->d_proj_out, r->cam_dec.mlp_w[1], r->d_attn_out, r->cam_dec.mlp_b[1], mlp_dim, mlp_dim, 1);
    kl_gelu(r, r->d_proj_out, mlp_dim);

    float *h_mlp = (float *)malloc((size_t)mlp_dim * sizeof(float));
    hipDeviceSynchronize();
    hipMemcpy(h_mlp, r->d_proj_out, (size_t)mlp_dim * sizeof(float), hipMemcpyDeviceToHost);

    float *h_fc_t_w = (float *)malloc(3*mlp_dim*sizeof(float)); float *h_fc_t_b = (float *)malloc(3*sizeof(float));
    float *h_fc_q_w = (float *)malloc(4*mlp_dim*sizeof(float)); float *h_fc_q_b = (float *)malloc(4*sizeof(float));
    float *h_fc_f_w = (float *)malloc(2*mlp_dim*sizeof(float)); float *h_fc_f_b = (float *)malloc(2*sizeof(float));
    hipMemcpy(h_fc_t_w, r->cam_dec.fc_t_w, 3*mlp_dim*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_fc_t_b, r->cam_dec.fc_t_b, 3*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_fc_q_w, r->cam_dec.fc_qvec_w, 4*mlp_dim*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_fc_q_b, r->cam_dec.fc_qvec_b, 4*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_fc_f_w, r->cam_dec.fc_fov_w, 2*mlp_dim*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_fc_f_b, r->cam_dec.fc_fov_b, 2*sizeof(float), hipMemcpyDeviceToHost);

    for (int o = 0; o < 3; o++) { float s = h_fc_t_b[o]; for (int k = 0; k < mlp_dim; k++) s += h_fc_t_w[o*mlp_dim+k]*h_mlp[k]; pose_out[o] = s; }
    for (int o = 0; o < 4; o++) { float s = h_fc_q_b[o]; for (int k = 0; k < mlp_dim; k++) s += h_fc_q_w[o*mlp_dim+k]*h_mlp[k]; pose_out[3+o] = s; }
    for (int o = 0; o < 2; o++) { float s = h_fc_f_b[o]; for (int k = 0; k < mlp_dim; k++) s += h_fc_f_w[o*mlp_dim+k]*h_mlp[k]; pose_out[7+o] = s; }

    free(h_mlp); free(h_fc_t_w); free(h_fc_t_b); free(h_fc_q_w); free(h_fc_q_b); free(h_fc_f_w); free(h_fc_f_b);
}

/* Aux DPT: rays + sky segmentation (reuses d_dpt_adapted from main DPT) */
static void run_aux_dpt(hip_da3_runner *r, int *sp_h, int *sp_w, int features,
                          void **d_aux_out) {
    int feat = features;

    /* Bottom-up aux RefineNet fusion (same structure as main but with aux weights) */
    void *aux_fused = r->d_dpt_fused; /* reuse after main DPT completes */

    /* Level 3 (deepest) */
    gpu_refinenet_w(r, r->d_dpt_adapted[3], sp_h[3], sp_w[3],
                     NULL, 0, 0, feat, aux_fused,
                     r->dpt_aux.fuse_out_w[3], r->dpt_aux.fuse_out_b[3],
                     r->dpt_aux.fuse_rcu1_c1_w[3], r->dpt_aux.fuse_rcu1_c1_b[3],
                     r->dpt_aux.fuse_rcu1_c2_w[3], r->dpt_aux.fuse_rcu1_c2_b[3],
                     r->dpt_aux.fuse_rcu2_c1_w[3], r->dpt_aux.fuse_rcu2_c1_b[3],
                     r->dpt_aux.fuse_rcu2_c2_w[3], r->dpt_aux.fuse_rcu2_c2_b[3],
                     r->dpt_aux.has_rcu1[3], r->dpt_aux.has_rcu2[3]);
    int fh = sp_h[3], fw = sp_w[3];

    /* Level 2 */
    gpu_refinenet_w(r, r->d_dpt_adapted[2], sp_h[2], sp_w[2],
                     aux_fused, fh, fw, feat, aux_fused,
                     r->dpt_aux.fuse_out_w[2], r->dpt_aux.fuse_out_b[2],
                     r->dpt_aux.fuse_rcu1_c1_w[2], r->dpt_aux.fuse_rcu1_c1_b[2],
                     r->dpt_aux.fuse_rcu1_c2_w[2], r->dpt_aux.fuse_rcu1_c2_b[2],
                     r->dpt_aux.fuse_rcu2_c1_w[2], r->dpt_aux.fuse_rcu2_c1_b[2],
                     r->dpt_aux.fuse_rcu2_c2_w[2], r->dpt_aux.fuse_rcu2_c2_b[2],
                     r->dpt_aux.has_rcu1[2], r->dpt_aux.has_rcu2[2]);
    fh = sp_h[2]; fw = sp_w[2];

    /* Level 1 */
    gpu_refinenet_w(r, r->d_dpt_adapted[1], sp_h[1], sp_w[1],
                     aux_fused, fh, fw, feat, aux_fused,
                     r->dpt_aux.fuse_out_w[1], r->dpt_aux.fuse_out_b[1],
                     r->dpt_aux.fuse_rcu1_c1_w[1], r->dpt_aux.fuse_rcu1_c1_b[1],
                     r->dpt_aux.fuse_rcu1_c2_w[1], r->dpt_aux.fuse_rcu1_c2_b[1],
                     r->dpt_aux.fuse_rcu2_c1_w[1], r->dpt_aux.fuse_rcu2_c1_b[1],
                     r->dpt_aux.fuse_rcu2_c2_w[1], r->dpt_aux.fuse_rcu2_c2_b[1],
                     r->dpt_aux.has_rcu1[1], r->dpt_aux.has_rcu2[1]);
    fh = sp_h[1]; fw = sp_w[1];

    /* Level 0 */
    gpu_refinenet_w(r, r->d_dpt_adapted[0], sp_h[0], sp_w[0],
                     aux_fused, fh, fw, feat, aux_fused,
                     r->dpt_aux.fuse_out_w[0], r->dpt_aux.fuse_out_b[0],
                     r->dpt_aux.fuse_rcu1_c1_w[0], r->dpt_aux.fuse_rcu1_c1_b[0],
                     r->dpt_aux.fuse_rcu1_c2_w[0], r->dpt_aux.fuse_rcu1_c2_b[0],
                     r->dpt_aux.fuse_rcu2_c1_w[0], r->dpt_aux.fuse_rcu2_c1_b[0],
                     r->dpt_aux.fuse_rcu2_c2_w[0], r->dpt_aux.fuse_rcu2_c2_b[0],
                     r->dpt_aux.has_rcu1[0], r->dpt_aux.has_rcu2[0]);

    /* Debug: check aux_fused values after refinenet */
    if (r->verbose >= 2) {
        hipDeviceSynchronize();
        float dbg[8];
        hipMemcpy(dbg, aux_fused, sizeof(dbg), hipMemcpyDeviceToHost);
        fprintf(stderr, "  aux_fused[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
    }

    /* Only the LAST level (index 3, finest resolution) is used for output.
     * Python: fused_aux_pyr[-1] -> output_conv2_aux[-1] -> final output.
     * output_conv1_aux[3]: 5 Conv2d, no activations (dense indices .0-.4)
     * output_conv2_aux[3]: Conv2d(128,32,3) -> LayerNorm(32) -> ReLU -> Conv2d(32,7,1) */
    {
        int lv = 3;  /* last level = finest */
        int oh = sp_h[0], ow = sp_w[0];

        /* output_conv1_aux[3]: 5 Conv2d chain (NO activations) */
        void *cur = r->d_aux_scratch;
        hipMemcpyAsync(cur, aux_fused, (size_t)feat * oh * ow * sizeof(float), hipMemcpyDeviceToDevice, r->stream);
        int ci = feat;
        for (int ci_idx = 0; ci_idx < r->dpt_aux.oc1_count[lv]; ci_idx++) {
            int co = r->dpt_aux.oc1_co[lv][ci_idx];
            void *dst_buf = (ci_idx % 2 == 0) ? r->d_dpt_tmp : r->d_aux_scratch;
            void *src_buf = (ci_idx % 2 == 0) ? r->d_aux_scratch : r->d_dpt_tmp;
            if (ci_idx == 0) src_buf = cur;
            kl_conv2d(r, dst_buf, src_buf,
                       r->dpt_aux.oc1_w[lv][ci_idx], r->dpt_aux.oc1_b[lv][ci_idx],
                       oh, ow, ci, co, 3, 3, 1, 1);
            if (r->verbose >= 2) {
                hipDeviceSynchronize();
                float dbg[4];
                hipMemcpy(dbg, dst_buf, sizeof(dbg), hipMemcpyDeviceToHost);
                fprintf(stderr, "    oc1[3].%d (%d->%d): %.4e %.4e %.4e %.4e\n",
                        ci_idx, ci, co, dbg[0], dbg[1], dbg[2], dbg[3]);
            }
            ci = co;
            cur = dst_buf;
        }

        /* output_conv2_aux[3]: Conv2d(128,32,3) + LayerNorm(32) + ReLU + Conv2d(32,7,1) */
        void *ln_in = r->d_dpt_tmp2;
        kl_conv2d(r, ln_in, cur,
                   r->dpt_aux.oc2_conv_w[lv], r->dpt_aux.oc2_conv_b[lv],
                   oh, ow, ci, 32, 3, 3, 1, 1);
        if (r->verbose >= 2) {
            hipDeviceSynchronize();
            float dbg[8];
            hipMemcpy(dbg, ln_in, sizeof(dbg), hipMemcpyDeviceToHost);
            fprintf(stderr, "  aux oc2 conv[0..7]: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
            int mid = 32 * (oh * ow / 2);
            hipMemcpy(dbg, (char *)ln_in + (size_t)mid * sizeof(float), sizeof(dbg), hipMemcpyDeviceToHost);
            fprintf(stderr, "  aux oc2 conv[mid]: %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }
        void *relu_src = ln_in;
        /* Use LayerNorm weights for this level (NULL = default init w=1, b=0).
         * Only level 0 has trained LayerNorm weights in safetensors;
         * levels 1-3 use default init (normalize without learned scale/bias). */
        void *ln_w = r->dpt_aux.oc2_gn_w[lv];  /* may be NULL */
        void *ln_b = r->dpt_aux.oc2_gn_b[lv];  /* may be NULL */
        {
            void *ln_out = r->d_aux_scratch;
            kl_channel_layernorm(r, ln_out, ln_in, ln_w, ln_b, 32, oh * ow);
            relu_src = ln_out;
            if (r->verbose >= 2) {
                hipDeviceSynchronize();
                float dbg[8];
                hipMemcpy(dbg, ln_out, sizeof(dbg), hipMemcpyDeviceToHost);
                fprintf(stderr, "  aux oc2 ln[0..7]: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",
                        dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
                int mid = 32 * (oh * ow / 2);
                hipMemcpy(dbg, (char *)ln_out + (size_t)mid * sizeof(float), sizeof(dbg), hipMemcpyDeviceToHost);
                fprintf(stderr, "  aux oc2 ln[mid]: %.4e %.4e %.4e %.4e\n",
                        dbg[0], dbg[1], dbg[2], dbg[3]);
            }
        }
        kl_relu_inplace(r, relu_src, 32 * oh * ow);
        if (r->verbose >= 2) {
            hipDeviceSynchronize();
            float dbg[8];
            hipMemcpy(dbg, relu_src, sizeof(dbg), hipMemcpyDeviceToHost);
            fprintf(stderr, "  aux oc2 relu[0..7]: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
        }
        kl_conv2d(r, d_aux_out[0], relu_src,
                   r->dpt_aux.oc2_out_w[lv], r->dpt_aux.oc2_out_b[lv],
                   oh, ow, 32, 7, 1, 1, 1, 0);
        if (r->verbose >= 2) {
            hipDeviceSynchronize();
            float dbg[8];
            hipMemcpy(dbg, d_aux_out[0], sizeof(dbg), hipMemcpyDeviceToHost);
            fprintf(stderr, "  aux final[0..7]: %.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
            int mid = 7 * (oh * ow / 2);
            hipMemcpy(dbg, (char *)d_aux_out[0] + (size_t)mid * sizeof(float), sizeof(dbg), hipMemcpyDeviceToHost);
            fprintf(stderr, "  aux final[mid]: %.4e %.4e %.4e %.4e\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }
    }
}

/* ======================================================================== */
/* Public API: predict                                                      */
/* ======================================================================== */

da3_hip_result hip_da3_predict(hip_da3_runner *r, const uint8_t *rgb, int img_w, int img_h) {
    da3_full_result full = hip_da3_predict_full(r, rgb, img_w, img_h, DA3_OUTPUT_DEPTH, NULL);
    da3_hip_result result = {0};
    result.depth = full.depth; result.confidence = full.confidence;
    result.width = full.width; result.height = full.height;
    free(full.rays); free(full.ray_confidence); free(full.sky_seg);
    free(full.gaussians); free(full.metric_depth);
    return result;
}

/* ======================================================================== */
/* Public API: predict_full                                                 */
/* ======================================================================== */

da3_full_result hip_da3_predict_full(hip_da3_runner *r, const uint8_t *rgb,
                                        int img_w, int img_h, int output_flags,
                                        const float *pose_in) {
    da3_full_result result = {0};
    if (!r->loaded) return result;

    int dim = r->dim, nt = r->n_tokens, np = r->n_patches;
    int gh = r->grid_h, gw = r->grid_w, ps = r->patch_size;
    int target_h = gh * ps, target_w = gw * ps;
    double t0, t1;
    struct timespec ts;

    /* GPU: Preprocess + Patch Embed + CLS + PosEmbed */
    clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    size_t img_bytes = (size_t)img_w * img_h * 3;
    if (img_bytes > r->d_img_raw_cap) {
        if (r->d_img_raw) hipFree(r->d_img_raw);
        hipMalloc(&r->d_img_raw, img_bytes);
        r->d_img_raw_cap = img_bytes;
    }
    hipMemcpy(r->d_img_raw, rgb, img_bytes, hipMemcpyHostToDevice);

    { int total = target_h * target_w; int grid = (total + 255) / 256;
        float istd0 = 1.0f/r->image_std[0], istd1 = 1.0f/r->image_std[1], istd2 = 1.0f/r->image_std[2];
        float m0 = r->image_mean[0], m1 = r->image_mean[1], m2 = r->image_mean[2];
        void *args[] = {&r->d_img_norm, &r->d_img_raw, &img_w, &img_h, &target_w, &target_h, &m0, &m1, &m2, &istd0, &istd1, &istd2};
        KL(r->fn_resize_normalize, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args); }

    { int img_dim = target_w;
        void *args[] = {&r->d_hidden, &r->d_img_norm, &r->d_patch_embed_w, &r->d_patch_embed_b, &gw, &dim, &ps, &img_dim};
        KL(r->fn_patch_embed_conv2d, (unsigned)np, 1, 1, 256, 1, 1, 0, r->stream, args); }

    { int total = nt * dim; int grid = (total + 255) / 256;
        void *args[] = {&r->d_hidden, &r->d_cls_token, &r->d_pos_embed, &nt, &dim};
        KL(r->fn_cls_pos_embed, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args); }

    clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1) fprintf(stderr, "hip_da3: GPU preprocess+embed: %.1f ms\n", (t1-t0)*1000);

    /* GPU: Transformer Blocks */
    clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec * 1e-9;
    int stride_3dim = 3 * dim;

    for (int L = 0; L < r->n_blocks; L++) {
        hip_da3_layer *ly = &r->layers[L];
        kl_layernorm(r, r->d_ln_buf, r->d_hidden, ly->ln1_w, ly->ln1_b, nt, dim);
        kl_gemm(r, r->d_qkv, ly->attn_qkv_w, r->d_ln_buf, ly->attn_qkv_b, ly->qkv_rows, ly->qkv_cols, nt);

        if (L >= r->qk_norm_start && ly->has_qk_norm) {
            kl_qk_layernorm(r, r->d_qkv, ly->attn_q_norm_w, ly->attn_q_norm_b, nt, r->n_heads, r->head_dim, stride_3dim);
            void *k_base = (char *)r->d_qkv + (size_t)dim * sizeof(float);
            kl_qk_layernorm(r, k_base, ly->attn_k_norm_w, ly->attn_k_norm_b, nt, r->n_heads, r->head_dim, stride_3dim);
        }

        if (L >= r->rope_start) {
            kl_rope_2d(r, r->d_qkv, r->d_pos_y, r->d_pos_x, nt, r->n_heads, r->head_dim, stride_3dim);
            void *k_base = (char *)r->d_qkv + (size_t)dim * sizeof(float);
            kl_rope_2d(r, k_base, r->d_pos_y, r->d_pos_x, nt, r->n_heads, r->head_dim, stride_3dim);
        }

        { void *K_t = r->d_ffn_buf;
            void *V_t = (char *)r->d_ffn_buf + (size_t)nt * dim * sizeof(float);
            kl_kv_transpose(r, K_t, V_t, r->d_qkv, nt, dim, r->n_heads, r->head_dim);
            kl_attn_prefill(r, r->d_attn_out, r->d_qkv, K_t, V_t, nt, dim, r->n_heads, r->head_dim);
        }

        kl_gemm(r, r->d_proj_out, ly->attn_out_w, r->d_attn_out, ly->attn_out_b, ly->out_rows, ly->out_cols, nt);
        kl_layerscale_add(r, r->d_hidden, r->d_proj_out, ly->ls1, dim, nt * dim);
        kl_layernorm(r, r->d_ln_buf, r->d_hidden, ly->ln2_w, ly->ln2_b, nt, dim);

        if (ly->has_swiglu && ly->ffn_gate_up_w) {
            kl_gemm(r, r->d_ffn_buf, ly->ffn_gate_up_w, r->d_ln_buf, ly->ffn_gate_up_b, ly->ffn_gu_rows, ly->ffn_gu_cols, nt);
            int hid = ly->ffn_gu_rows / 2;
            kl_swiglu(r, r->d_ffn_mid, r->d_ffn_buf, hid, nt);
            kl_gemm(r, r->d_proj_out, ly->ffn_down_w, r->d_ffn_mid, ly->ffn_down_b, ly->ffn_down_rows, hid, nt);
        } else if (ly->ffn_up_w) {
            kl_gemm(r, r->d_ffn_buf, ly->ffn_up_w, r->d_ln_buf, ly->ffn_up_b, ly->ffn_up_rows, ly->ffn_up_cols, nt);
            kl_gelu(r, r->d_ffn_buf, nt * ly->ffn_up_rows);
            kl_gemm(r, r->d_proj_out, ly->ffn_down_w, r->d_ffn_buf, ly->ffn_down_b, ly->ffn_down_rows, ly->ffn_up_rows, nt);
        }

        kl_layerscale_add(r, r->d_hidden, r->d_proj_out, ly->ls2, dim, nt * dim);

        for (int fi = 0; fi < 4; fi++) {
            if (L == r->feature_layers[fi])
                hipMemcpyAsync(r->d_features[fi], r->d_hidden, (size_t)nt*dim*sizeof(float), hipMemcpyDeviceToDevice, r->stream);
        }
    }

    hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1) fprintf(stderr, "hip_da3: GPU backbone (%d blocks): %.1f ms\n", r->n_blocks, (t1-t0)*1000);

    /* CameraDec (Phase 1) */
    if (r->cam_dec.loaded && (output_flags & DA3_OUTPUT_POSE)) {
        if (pose_in) memcpy(result.pose, pose_in, 9*sizeof(float));
        else run_camera_dec(r, result.pose);
        result.has_pose = 1;
    }

    /* GPU: DPT Head */
    clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    if (!r->dpt_w.proj_w[0]) { fprintf(stderr, "hip_da3: no DPT head weights\n"); return result; }

    dpt_gpu_weights *dw = &r->dpt_w;
    int feat = r->head_features, head_dim_in = dim * 2;
    int sp_h[4], sp_w[4];
    sp_h[0] = sp_w[0] = (gh-1)*4+4; sp_h[1] = sp_w[1] = (gh-1)*2+2;
    sp_h[2] = sp_w[2] = gh; sp_h[3] = sp_w[3] = (gh+2-3)/2+1;

    /* Token processing + projection for each feature level */
    for (int fi = 0; fi < 4; fi++) {
        int oc_val = r->head_out_channels[fi];
        kl_cls_concat(r, r->d_dpt_cat, r->d_features[fi], np, dim);
        if (dw->norm_w) kl_layernorm(r, r->d_dpt_ln, r->d_dpt_cat, dw->norm_w, dw->norm_b, np, head_dim_in);
        else hipMemcpyAsync(r->d_dpt_ln, r->d_dpt_cat, (size_t)np*head_dim_in*sizeof(float), hipMemcpyDeviceToDevice, r->stream);
        kl_gemm(r, r->d_dpt_proj, dw->proj_w[fi], r->d_dpt_ln, dw->proj_b[fi], oc_val, head_dim_in, np);

        if (fi == 0) kl_deconv_gemm_scatter(r, r->d_dpt_spatial[0], r->d_dpt_proj, dw->upsample_0_w, dw->upsample_0_b, r->d_dpt_ln, oc_val, oc_val, gh, gw, 4, 4, 4);
        else if (fi == 1) kl_deconv_gemm_scatter(r, r->d_dpt_spatial[1], r->d_dpt_proj, dw->upsample_1_w, dw->upsample_1_b, r->d_dpt_ln, oc_val, oc_val, gh, gw, 2, 2, 2);
        else if (fi == 2) { kl_tok_to_chw(r, r->d_dpt_chw, r->d_dpt_proj, oc_val, gh, gw); hipMemcpyAsync(r->d_dpt_spatial[2], r->d_dpt_chw, (size_t)oc_val*gh*gw*sizeof(float), hipMemcpyDeviceToDevice, r->stream); }
        else { kl_tok_to_chw(r, r->d_dpt_chw, r->d_dpt_proj, oc_val, gh, gw); kl_conv_gemm(r, r->d_dpt_spatial[3], r->d_dpt_chw, dw->downsample_w, dw->downsample_b, gh, gw, oc_val, oc_val, 3, 3, 2, 1); }

        void *null_bias = NULL;
        kl_conv_gemm(r, r->d_dpt_adapted[fi], r->d_dpt_spatial[fi], dw->adapter_w[fi], null_bias, sp_h[fi], sp_w[fi], oc_val, feat, 3, 3, 1, 1);
    }

    /* Bottom-up RefineNet fusion */
    gpu_refinenet(r, 3, r->d_dpt_adapted[3], sp_h[3], sp_w[3], NULL, 0, 0, feat, r->d_dpt_fused);
    int fh = sp_h[3], fw = sp_w[3];
    gpu_refinenet(r, 2, r->d_dpt_adapted[2], sp_h[2], sp_w[2], r->d_dpt_fused, fh, fw, feat, r->d_dpt_fused); fh = sp_h[2]; fw = sp_w[2];
    gpu_refinenet(r, 1, r->d_dpt_adapted[1], sp_h[1], sp_w[1], r->d_dpt_fused, fh, fw, feat, r->d_dpt_fused); fh = sp_h[1]; fw = sp_w[1];
    gpu_refinenet(r, 0, r->d_dpt_adapted[0], sp_h[0], sp_w[0], r->d_dpt_fused, fh, fw, feat, r->d_dpt_fused); fh = sp_h[0]; fw = sp_w[0];

    /* Output convolutions */
    int feat_half = feat / 2; if (feat_half < 1) feat_half = 1;
    int out_mid = dw->out_mid > 0 ? dw->out_mid : feat_half;
    kl_conv_gemm(r, r->d_dpt_tmp, r->d_dpt_fused, dw->neck_w, dw->neck_b, fh, fw, feat, feat_half, 3, 3, 1, 1);
    kl_relu_inplace(r, r->d_dpt_tmp, feat_half*fh*fw);
    kl_conv_gemm(r, r->d_dpt_tmp2, r->d_dpt_tmp, dw->out_0_w, dw->out_0_b, fh, fw, feat_half, out_mid, 3, 3, 1, 1);
    kl_relu_inplace(r, r->d_dpt_tmp2, out_mid*fh*fw);
    kl_conv2d(r, r->d_dpt_out, r->d_dpt_tmp2, dw->out_2_w, dw->out_2_b, fh, fw, out_mid, 2, 1, 1, 1, 0);

    { int hw = fh*fw; int grid = (hw+255)/256;
        void *args[] = {&r->d_dpt_out, &hw};
        KL(r->fn_depth_activation, (unsigned)grid, 1, 1, 256, 1, 1, 0, r->stream, args); }

    /* Bilinear upsample to original resolution */
    { size_t result_sz = (size_t)2*img_h*img_w*sizeof(float);
        if (result_sz > r->d_result_cap) { if (r->d_result) hipFree(r->d_result); hipMalloc(&r->d_result, result_sz); r->d_result_cap = result_sz; }
        kl_bilinear(r, r->d_result, r->d_dpt_out, 2, fh, fw, img_h, img_w);
        hipDeviceSynchronize();
        float *h_result = (float *)malloc(result_sz);
        hipMemcpy(h_result, r->d_result, result_sz, hipMemcpyDeviceToHost);
        result.width = img_w; result.height = img_h;
        result.depth = (float *)malloc((size_t)img_w*img_h*sizeof(float));
        result.confidence = (float *)malloc((size_t)img_w*img_h*sizeof(float));
        memcpy(result.depth, h_result, (size_t)img_w*img_h*sizeof(float));
        memcpy(result.confidence, h_result + img_h*img_w, (size_t)img_w*img_h*sizeof(float));
        free(h_result); }

    clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    if (r->verbose >= 1) fprintf(stderr, "hip_da3: GPU DPT head: %.1f ms\n", (t1-t0)*1000);

    /* ─── Aux DPT (Phase 2): rays + sky segmentation ─── */
    if (r->dpt_aux.loaded && (output_flags & DA3_OUTPUT_RAYS)) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double ax0 = ts.tv_sec + ts.tv_nsec * 1e-9;

        /* Run aux branch (reuses d_dpt_adapted from main DPT token processing) */
        run_aux_dpt(r, sp_h, sp_w, feat, r->d_aux_out);

        /* Only last level (index 3) is used — single output in d_aux_out[0] */
        int oh = sp_h[0], ow = sp_w[0];

        /* Bilinear upsample 7 channels to original resolution */
        int npix = img_w * img_h;
        result.rays = (float *)malloc((size_t)6 * npix * sizeof(float));
        result.ray_confidence = (float *)malloc((size_t)npix * sizeof(float));

        void *d_aux_full = r->d_dpt_tmp;
        kl_bilinear(r, d_aux_full, r->d_aux_out[0], 7, oh, ow, img_h, img_w);
        hipDeviceSynchronize();

        float *h_aux_full = (float *)malloc((size_t)7 * npix * sizeof(float));
        hipMemcpy(h_aux_full, d_aux_full, (size_t)7 * npix * sizeof(float), hipMemcpyDeviceToHost);

        /* Split: rays = channels 0-5 (linear activation = identity),
         *        ray_confidence = channel 6 (expp1 = exp(x) + 1) */
        memcpy(result.rays, h_aux_full, (size_t)6 * npix * sizeof(float));
        /* Apply expp1 activation to ray confidence: exp(x) + 1 */
        {
            const float *src = h_aux_full + 6 * npix;
            for (int i = 0; i < npix; i++)
                result.ray_confidence[i] = expf(src[i]) + 1.0f;
        }
        /* Sky segmentation: not available from DualDPT (only from metric DPT branch) */
        result.sky_seg = NULL;
        free(h_aux_full);

        result.has_rays = 1;

        clock_gettime(CLOCK_MONOTONIC, &ts);
        double ax1 = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (r->verbose >= 1)
            fprintf(stderr, "hip_da3: Aux DPT (rays+sky): %.1f ms\n", (ax1-ax0)*1000);
    }

    /* ─── GSDPT (Phase 4): 3D Gaussian estimation ─── */
    if (r->gsdpt.loaded && (output_flags & DA3_OUTPUT_GAUSSIANS)) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double gs0 = ts.tv_sec + ts.tv_nsec * 1e-9;

        dpt_gpu_weights *gdw = &r->gsdpt.dpt;
        int gs_oc = r->gsdpt.gs_out_channels;
        if (gs_oc < 2) gs_oc = 38;

        /* Run images_merger: 3 stride-2 Conv2d on normalized image -> d_gs_merged
         * Input: d_img_norm [3, target_h, target_w] */
        {
            void *cur = r->d_img_norm;
            void *bufs[2] = {r->d_dpt_tmp, r->d_dpt_tmp2};
            int mh = target_h, mw = target_w;
            for (int mi = 0; mi < 3; mi++) {
                if (!r->gsdpt.merger_w[mi]) break;
                int mci = r->gsdpt.merger_ci[mi];
                int mco = r->gsdpt.merger_co[mi];
                int moh = (mh + 2 * 1 - 3) / 2 + 1;
                int mow = (mw + 2 * 1 - 3) / 2 + 1;
                void *dst = (mi == 2) ? r->d_gs_merged : bufs[mi % 2];
                kl_conv2d(r, dst, cur, r->gsdpt.merger_w[mi], r->gsdpt.merger_b[mi],
                           mh, mw, mci, mco, 3, 3, 2, 1);
                /* Activation at indices 1,3 only (not after last conv at index 4) */
                if (mi < 2)
                    kl_silu_inplace(r, dst, mco * moh * mow);
                cur = dst;
                mh = moh; mw = mow;
            }
        }

        /* GSDPT token processing + refinenet (same pipeline as main DPT) */
        /* Reuses d_features from backbone as input */

        for (int fi = 0; fi < 4; fi++) {
            int oc_val = r->head_out_channels[fi];

            kl_cls_concat(r, r->d_dpt_cat, r->d_features[fi], np, dim);
            if (gdw->norm_w)
                kl_layernorm(r, r->d_dpt_ln, r->d_dpt_cat, gdw->norm_w, gdw->norm_b,
                              np, head_dim_in);
            else
                hipMemcpyAsync(r->d_dpt_ln, r->d_dpt_cat,
                                   (size_t)np * head_dim_in * sizeof(float), hipMemcpyDeviceToDevice, r->stream);

            kl_gemm(r, r->d_dpt_proj, gdw->proj_w[fi], r->d_dpt_ln, gdw->proj_b[fi],
                     oc_val, head_dim_in, np);

            if (fi == 0)
                kl_deconv_gemm_scatter(r, r->d_dpt_spatial[0], r->d_dpt_proj,
                                        gdw->upsample_0_w, gdw->upsample_0_b, r->d_dpt_ln,
                                        oc_val, oc_val, gh, gw, 4, 4, 4);
            else if (fi == 1)
                kl_deconv_gemm_scatter(r, r->d_dpt_spatial[1], r->d_dpt_proj,
                                        gdw->upsample_1_w, gdw->upsample_1_b, r->d_dpt_ln,
                                        oc_val, oc_val, gh, gw, 2, 2, 2);
            else if (fi == 2) {
                kl_tok_to_chw(r, r->d_dpt_chw, r->d_dpt_proj, oc_val, gh, gw);
                hipMemcpyAsync(r->d_dpt_spatial[2], r->d_dpt_chw,
                                   (size_t)oc_val * gh * gw * sizeof(float), hipMemcpyDeviceToDevice, r->stream);
            } else {
                kl_tok_to_chw(r, r->d_dpt_chw, r->d_dpt_proj, oc_val, gh, gw);
                kl_conv_gemm(r, r->d_dpt_spatial[3], r->d_dpt_chw,
                              gdw->downsample_w, gdw->downsample_b,
                              gh, gw, oc_val, oc_val, 3, 3, 2, 1);
            }

            void *null_bias = NULL;
            kl_conv_gemm(r, r->d_dpt_adapted[fi], r->d_dpt_spatial[fi],
                          gdw->adapter_w[fi], null_bias,
                          sp_h[fi], sp_w[fi], oc_val, feat, 3, 3, 1, 1);
        }

        /* TODO: Inject merger features at level 0 (add to d_dpt_adapted[0]) */
        /* For now, skip merger injection — just run standard DPT pipeline */

        /* Bottom-up RefineNet fusion with GSDPT weights */
        gpu_refinenet_w(r, r->d_dpt_adapted[3], sp_h[3], sp_w[3],
                        NULL, 0, 0, feat, r->d_dpt_fused,
                        gdw->fuse_out_w[3], gdw->fuse_out_b[3],
                        gdw->fuse_rcu1_c1_w[3], gdw->fuse_rcu1_c1_b[3],
                        gdw->fuse_rcu1_c2_w[3], gdw->fuse_rcu1_c2_b[3],
                        gdw->fuse_rcu2_c1_w[3], gdw->fuse_rcu2_c1_b[3],
                        gdw->fuse_rcu2_c2_w[3], gdw->fuse_rcu2_c2_b[3],
                        gdw->has_rcu1[3], gdw->has_rcu2[3]);
        int gs_fh = sp_h[3], gs_fw = sp_w[3];
        gpu_refinenet_w(r, r->d_dpt_adapted[2], sp_h[2], sp_w[2],
                        r->d_dpt_fused, gs_fh, gs_fw, feat, r->d_dpt_fused,
                        gdw->fuse_out_w[2], gdw->fuse_out_b[2],
                        gdw->fuse_rcu1_c1_w[2], gdw->fuse_rcu1_c1_b[2],
                        gdw->fuse_rcu1_c2_w[2], gdw->fuse_rcu1_c2_b[2],
                        gdw->fuse_rcu2_c1_w[2], gdw->fuse_rcu2_c1_b[2],
                        gdw->fuse_rcu2_c2_w[2], gdw->fuse_rcu2_c2_b[2],
                        gdw->has_rcu1[2], gdw->has_rcu2[2]);
        gs_fh = sp_h[2]; gs_fw = sp_w[2];
        gpu_refinenet_w(r, r->d_dpt_adapted[1], sp_h[1], sp_w[1],
                        r->d_dpt_fused, gs_fh, gs_fw, feat, r->d_dpt_fused,
                        gdw->fuse_out_w[1], gdw->fuse_out_b[1],
                        gdw->fuse_rcu1_c1_w[1], gdw->fuse_rcu1_c1_b[1],
                        gdw->fuse_rcu1_c2_w[1], gdw->fuse_rcu1_c2_b[1],
                        gdw->fuse_rcu2_c1_w[1], gdw->fuse_rcu2_c1_b[1],
                        gdw->fuse_rcu2_c2_w[1], gdw->fuse_rcu2_c2_b[1],
                        gdw->has_rcu1[1], gdw->has_rcu2[1]);
        gs_fh = sp_h[1]; gs_fw = sp_w[1];
        gpu_refinenet_w(r, r->d_dpt_adapted[0], sp_h[0], sp_w[0],
                        r->d_dpt_fused, gs_fh, gs_fw, feat, r->d_dpt_fused,
                        gdw->fuse_out_w[0], gdw->fuse_out_b[0],
                        gdw->fuse_rcu1_c1_w[0], gdw->fuse_rcu1_c1_b[0],
                        gdw->fuse_rcu1_c2_w[0], gdw->fuse_rcu1_c2_b[0],
                        gdw->fuse_rcu2_c1_w[0], gdw->fuse_rcu2_c1_b[0],
                        gdw->fuse_rcu2_c2_w[0], gdw->fuse_rcu2_c2_b[0],
                        gdw->has_rcu1[0], gdw->has_rcu2[0]);
        gs_fh = sp_h[0]; gs_fw = sp_w[0];

        /* Output convolutions -> gs_oc channels
         * output_conv1 (neck): Conv2d(256, 128, 3) — NO ReLU (single .0 index)
         * + inject upsampled merger features (element-wise add)
         * output_conv2: Conv2d(128, 32, 3) + ReLU + Conv2d(32, gs_oc, 1) */
        int gs_feat_half = feat / 2;
        if (gs_feat_half < 1) gs_feat_half = 1;
        int gs_out_mid = gdw->out_mid > 0 ? gdw->out_mid : gs_feat_half;

        /* output_conv1 (neck): Conv2d(256, 128, 3, pad=1), no activation */
        kl_conv_gemm(r, r->d_dpt_tmp, r->d_dpt_fused,
                      gdw->neck_w, gdw->neck_b,
                      gs_fh, gs_fw, feat, gs_feat_half, 3, 3, 1, 1);

        /* Inject merger features: upsample d_gs_merged [128, mg_h, mg_w] -> [128, gs_fh, gs_fw]
         * then add to neck output */
        if (r->d_gs_merged && r->gs_merger_h > 0) {
            kl_bilinear(r, r->d_dpt_tmp2, r->d_gs_merged,
                         gs_feat_half, r->gs_merger_h, r->gs_merger_w, gs_fh, gs_fw);
            kl_add_inplace(r, r->d_dpt_tmp, r->d_dpt_tmp2, gs_feat_half * gs_fh * gs_fw);
        }

        /* output_conv2: Conv2d(128, 32, 3, pad=1) + ReLU + Conv2d(32, gs_oc, 1) */
        kl_conv_gemm(r, r->d_dpt_tmp2, r->d_dpt_tmp,
                      gdw->out_0_w, gdw->out_0_b,
                      gs_fh, gs_fw, gs_feat_half, gs_out_mid, 3, 3, 1, 1);
        kl_relu_inplace(r, r->d_dpt_tmp2, gs_out_mid * gs_fh * gs_fw);
        kl_conv2d(r, r->d_gs_out, r->d_dpt_tmp2,
                   gdw->out_2_w, gdw->out_2_b,
                   gs_fh, gs_fw, gs_out_mid, gs_oc, 1, 1, 1, 0);

        /* Download gaussians */
        hipDeviceSynchronize();

        /* Bilinear upsample each channel to output resolution */
        int npix = img_w * img_h;
        size_t gs_full_sz = (size_t)gs_oc * npix * sizeof(float);
        void *d_gs_full = NULL;
        hipMalloc(&d_gs_full, gs_full_sz);
        kl_bilinear(r, d_gs_full, r->d_gs_out, gs_oc, gs_fh, gs_fw, img_h, img_w);
        hipDeviceSynchronize();
        result.gaussians = (float *)malloc(gs_full_sz);
        hipMemcpy(result.gaussians, d_gs_full, gs_full_sz, hipMemcpyDeviceToHost);
        hipFree(d_gs_full);

        result.has_gaussians = 1;

        clock_gettime(CLOCK_MONOTONIC, &ts);
        double gs1 = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (r->verbose >= 1)
            fprintf(stderr, "hip_da3: GSDPT (%d channels): %.1f ms\n",
                    gs_oc, (gs1-gs0)*1000);
    }

    return result;
}

/* ======================================================================== */
/* Public API: free                                                         */
/* ======================================================================== */

void hip_da3_free(hip_da3_runner *r) {
    if (!r) return;

    if (r->layers) {
        for (int L = 0; L < r->n_blocks; L++) {
            hip_da3_layer *ly = &r->layers[L];
            if (ly->ln1_w) hipFree(ly->ln1_w); if (ly->ln1_b) hipFree(ly->ln1_b);
            if (ly->attn_qkv_w) hipFree(ly->attn_qkv_w); if (ly->attn_qkv_b) hipFree(ly->attn_qkv_b);
            if (ly->attn_q_norm_w) hipFree(ly->attn_q_norm_w); if (ly->attn_q_norm_b) hipFree(ly->attn_q_norm_b);
            if (ly->attn_k_norm_w) hipFree(ly->attn_k_norm_w); if (ly->attn_k_norm_b) hipFree(ly->attn_k_norm_b);
            if (ly->attn_out_w) hipFree(ly->attn_out_w); if (ly->attn_out_b) hipFree(ly->attn_out_b);
            if (ly->ls1) hipFree(ly->ls1); if (ly->ls2) hipFree(ly->ls2);
            if (ly->ln2_w) hipFree(ly->ln2_w); if (ly->ln2_b) hipFree(ly->ln2_b);
            if (ly->ffn_gate_up_w) hipFree(ly->ffn_gate_up_w); if (ly->ffn_gate_up_b) hipFree(ly->ffn_gate_up_b);
            if (ly->ffn_up_w) hipFree(ly->ffn_up_w); if (ly->ffn_up_b) hipFree(ly->ffn_up_b);
            if (ly->ffn_down_w) hipFree(ly->ffn_down_w); if (ly->ffn_down_b) hipFree(ly->ffn_down_b);
        }
        free(r->layers);
    }

    if (r->d_cls_token) hipFree(r->d_cls_token); if (r->d_pos_embed) hipFree(r->d_pos_embed);
    if (r->d_patch_embed_w) hipFree(r->d_patch_embed_w); if (r->d_patch_embed_b) hipFree(r->d_patch_embed_b);
    if (r->d_img_norm) hipFree(r->d_img_norm); if (r->d_img_raw) hipFree(r->d_img_raw);
    if (r->d_result) hipFree(r->d_result);
    if (r->d_hidden) hipFree(r->d_hidden); if (r->d_hidden2) hipFree(r->d_hidden2);
    if (r->d_ln_buf) hipFree(r->d_ln_buf); if (r->d_qkv) hipFree(r->d_qkv);
    if (r->d_attn_out) hipFree(r->d_attn_out); if (r->d_ffn_buf) hipFree(r->d_ffn_buf);
    if (r->d_ffn_mid) hipFree(r->d_ffn_mid); if (r->d_proj_out) hipFree(r->d_proj_out);
    if (r->d_pos_y) hipFree(r->d_pos_y); if (r->d_pos_x) hipFree(r->d_pos_x);
    for (int i = 0; i < 4; i++) if (r->d_features[i]) hipFree(r->d_features[i]);

    { dpt_gpu_weights *dw_p = &r->dpt_w;
        if (dw_p->norm_w) hipFree(dw_p->norm_w); if (dw_p->norm_b) hipFree(dw_p->norm_b);
        for (int i = 0; i < 4; i++) {
            if (dw_p->proj_w[i]) hipFree(dw_p->proj_w[i]); if (dw_p->proj_b[i]) hipFree(dw_p->proj_b[i]);
            if (dw_p->adapter_w[i]) hipFree(dw_p->adapter_w[i]);
            if (dw_p->fuse_out_w[i]) hipFree(dw_p->fuse_out_w[i]); if (dw_p->fuse_out_b[i]) hipFree(dw_p->fuse_out_b[i]);
            if (dw_p->fuse_rcu1_c1_w[i]) hipFree(dw_p->fuse_rcu1_c1_w[i]); if (dw_p->fuse_rcu1_c1_b[i]) hipFree(dw_p->fuse_rcu1_c1_b[i]);
            if (dw_p->fuse_rcu1_c2_w[i]) hipFree(dw_p->fuse_rcu1_c2_w[i]); if (dw_p->fuse_rcu1_c2_b[i]) hipFree(dw_p->fuse_rcu1_c2_b[i]);
            if (dw_p->fuse_rcu2_c1_w[i]) hipFree(dw_p->fuse_rcu2_c1_w[i]); if (dw_p->fuse_rcu2_c1_b[i]) hipFree(dw_p->fuse_rcu2_c1_b[i]);
            if (dw_p->fuse_rcu2_c2_w[i]) hipFree(dw_p->fuse_rcu2_c2_w[i]); if (dw_p->fuse_rcu2_c2_b[i]) hipFree(dw_p->fuse_rcu2_c2_b[i]);
        }
        if (dw_p->upsample_0_w) hipFree(dw_p->upsample_0_w); if (dw_p->upsample_0_b) hipFree(dw_p->upsample_0_b);
        if (dw_p->upsample_1_w) hipFree(dw_p->upsample_1_w); if (dw_p->upsample_1_b) hipFree(dw_p->upsample_1_b);
        if (dw_p->downsample_w) hipFree(dw_p->downsample_w); if (dw_p->downsample_b) hipFree(dw_p->downsample_b);
        if (dw_p->neck_w) hipFree(dw_p->neck_w); if (dw_p->neck_b) hipFree(dw_p->neck_b);
        if (dw_p->out_0_w) hipFree(dw_p->out_0_w); if (dw_p->out_0_b) hipFree(dw_p->out_0_b);
        if (dw_p->out_2_w) hipFree(dw_p->out_2_w); if (dw_p->out_2_b) hipFree(dw_p->out_2_b);
    }

    if (r->d_dpt_cat) hipFree(r->d_dpt_cat); if (r->d_dpt_ln) hipFree(r->d_dpt_ln);
    if (r->d_dpt_proj) hipFree(r->d_dpt_proj); if (r->d_dpt_chw) hipFree(r->d_dpt_chw);
    for (int i = 0; i < 4; i++) { if (r->d_dpt_spatial[i]) hipFree(r->d_dpt_spatial[i]); if (r->d_dpt_adapted[i]) hipFree(r->d_dpt_adapted[i]); }
    if (r->d_dpt_fused) hipFree(r->d_dpt_fused); if (r->d_dpt_tmp) hipFree(r->d_dpt_tmp);
    if (r->d_dpt_tmp2) hipFree(r->d_dpt_tmp2); if (r->d_dpt_out) hipFree(r->d_dpt_out);

    /* Free CameraDec weights */
    if (r->cam_dec.loaded) {
        if (r->cam_dec.backbone_norm_w) hipFree(r->cam_dec.backbone_norm_w); if (r->cam_dec.backbone_norm_b) hipFree(r->cam_dec.backbone_norm_b);
        for (int i = 0; i < 2; i++) { if (r->cam_dec.mlp_w[i]) hipFree(r->cam_dec.mlp_w[i]); if (r->cam_dec.mlp_b[i]) hipFree(r->cam_dec.mlp_b[i]); }
        if (r->cam_dec.fc_t_w) hipFree(r->cam_dec.fc_t_w); if (r->cam_dec.fc_t_b) hipFree(r->cam_dec.fc_t_b);
        if (r->cam_dec.fc_qvec_w) hipFree(r->cam_dec.fc_qvec_w); if (r->cam_dec.fc_qvec_b) hipFree(r->cam_dec.fc_qvec_b);
        if (r->cam_dec.fc_fov_w) hipFree(r->cam_dec.fc_fov_w); if (r->cam_dec.fc_fov_b) hipFree(r->cam_dec.fc_fov_b);
    }

    for (int i = 0; i < 4; i++) if (r->d_aux_out[i]) hipFree(r->d_aux_out[i]);
    if (r->d_aux_scratch) hipFree(r->d_aux_scratch);
    if (r->d_gs_merged) hipFree(r->d_gs_merged); if (r->d_gs_out) hipFree(r->d_gs_out);

    free(r->h_output);
    if (r->cpu_model) da3_free(r->cpu_model);
    if (r->module) hipModuleUnload(r->module);
    if (r->stream) hipStreamDestroy(r->stream);
    if (r->context) hipCtxDestroy(r->context);
    free(r);
}
