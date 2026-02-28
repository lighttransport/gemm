/*
 * cuda_vision_encoder.c - CUDA vision encoder for Qwen3-VL mmproj
 *
 * Compiles with plain gcc (no nvcc). Uses cuew for dynamic CUDA/NVRTC loading.
 * Supports F32 (verification) and F16 (performance) weight modes.
 * Single-stream sequential kernel launches.
 */

#include "../../common/ggml_dequant.h"

#include "cuda_vision_encoder.h"
#include "../cuew.h"
#include "../cuda_kernels_common.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Aliases for shared macros */
#define CHECK_CU CU_CHECK
#define CHECK_CU_NULL CU_CHECK_NULL

/* ======================================================================== */
/* Vision-specific CUDA kernels (compiled at runtime via NVRTC)             */
/* Shared kernels (layernorm, GEMM, gelu, add, etc.) are in                 */
/* cuda_kernels_common.h. This string is concatenated after them.           */
/* ======================================================================== */

static const char *cuda_vlm_specific_kernels =
"\n"
"/* ---- gemm_f32_f32: Naive tiled F32 GEMM for verification ---- */\n"
"/* Y[tok][i] = sum_j(W[i][j] * X[tok][j]) + bias[i] */\n"
"/* Grid: (ceil(n_out/TILE), ceil(n_tok/TILE)), Block: (TILE, TILE) */\n"
"#define TILE_F32 16\n"
"__global__ void gemm_f32_f32(float *Y, const float *W, const float *X,\n"
"                              const float *bias,\n"
"                              int n_out, int n_in, int n_tok) {\n"
"    __shared__ float sW[TILE_F32][TILE_F32];\n"
"    __shared__ float sX[TILE_F32][TILE_F32];\n"
"    int bx = blockIdx.x * TILE_F32;\n"
"    int by = blockIdx.y * TILE_F32;\n"
"    int tx = threadIdx.x;\n"
"    int ty = threadIdx.y;\n"
"    int row = by + ty;  /* token index */\n"
"    int col = bx + tx;  /* output dim index */\n"
"    float sum = 0.0f;\n"
"    for (int t = 0; t < n_in; t += TILE_F32) {\n"
"        /* Load W tile: W[col][t+ty] — but col is the output row of W */\n"
"        if (col < n_out && (t + ty) < n_in)\n"
"            sW[tx][ty] = W[(size_t)col * n_in + t + ty];\n"
"        else\n"
"            sW[tx][ty] = 0.0f;\n"
"        /* Load X tile: X[row][t+tx] */\n"
"        if (row < n_tok && (t + tx) < n_in)\n"
"            sX[ty][tx] = X[(size_t)row * n_in + t + tx];\n"
"        else\n"
"            sX[ty][tx] = 0.0f;\n"
"        __syncthreads();\n"
"        for (int k = 0; k < TILE_F32; k++)\n"
"            sum += sW[tx][k] * sX[ty][k];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < n_tok && col < n_out) {\n"
"        float b = (bias) ? bias[col] : 0.0f;\n"
"        Y[(size_t)row * n_out + col] = sum + b;\n"
"    }\n"
"}\n"
"\n"
"/* ---- patch_embed_dual_f32: Dual Conv2D patch extraction ---- */\n"
"/* Grid: (n_patches), Block: (256) */\n"
"__global__ void patch_embed_dual_f32(float *out, const float *rgb,\n"
"                                      const float *w0, const float *w1,\n"
"                                      const float *bias,\n"
"                                      int gw, int dim, int ps, int img_w) {\n"
"    int patch = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    int ks = ps * ps * 3;\n"
"    for (int d = tid; d < dim; d += blockDim.x) {\n"
"        float sum = bias ? bias[d] : 0.0f;\n"
"        for (int c = 0; c < 3; c++) {\n"
"            for (int ky = 0; ky < ps; ky++) {\n"
"                for (int kx = 0; kx < ps; kx++) {\n"
"                    int iy = py * ps + ky;\n"
"                    int ix = px * ps + kx;\n"
"                    float pix = rgb[(iy * img_w + ix) * 3 + c];\n"
"                    int ki = c * ps * ps + ky * ps + kx;\n"
"                    sum += w0[d * ks + ki] * pix;\n"
"                    if (w1) sum += w1[d * ks + ki] * pix;\n"
"                }\n"
"            }\n"
"        }\n"
"        out[patch * dim + d] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ---- add_pos_embd: add position embeddings via indirection map ---- */\n"
"/* Grid: (n_patches), Block: (256) */\n"
"__global__ void add_pos_embd(float *hidden, const float *pos_emb,\n"
"                               const int *pos_map, int dim) {\n"
"    int p = blockIdx.x;\n"
"    int orig_p = pos_map[p];\n"
"    int tid = threadIdx.x;\n"
"    for (int d = tid; d < dim; d += blockDim.x)\n"
"        hidden[p * dim + d] += pos_emb[orig_p * dim + d];\n"
"}\n"
"\n"
"/* ---- add_pos_embd_direct: add pre-interpolated position embeddings ---- */\n"
"/* Grid: (n_patches), Block: (256) */\n"
"__global__ void add_pos_embd_direct(float *hidden, const float *pos_emb,\n"
"                                      int dim, int n) {\n"
"    int p = blockIdx.x;\n"
"    if (p >= n) return;\n"
"    for (int d = threadIdx.x; d < dim; d += blockDim.x)\n"
"        hidden[p * dim + d] += pos_emb[p * dim + d];\n"
"}\n"
"\n"
"/* ---- rope_vision_f32: M-RoPE on Q and K ---- */\n"
"/* Grid: (n_patches * n_heads), Block: (half_dim) */\n"
"__global__ void rope_vision_f32(float *qkv, const float *rope_cos,\n"
"                                  const float *rope_sin,\n"
"                                  int n_patches, int n_heads,\n"
"                                  int dim, int head_dim, int half) {\n"
"    int idx = blockIdx.x;\n"
"    int p = idx / n_heads;\n"
"    int h = idx % n_heads;\n"
"    int i = threadIdx.x;\n"
"    if (i >= half) return;\n"
"    float cos_t = rope_cos[p * head_dim + 2 * i];\n"
"    float sin_t = rope_sin[p * head_dim + 2 * i];\n"
"    /* Q */\n"
"    float *q = qkv + p * 3 * dim + h * head_dim;\n"
"    float q0 = q[i], q1 = q[i + half];\n"
"    q[i]        = q0 * cos_t - q1 * sin_t;\n"
"    q[i + half] = q0 * sin_t + q1 * cos_t;\n"
"    /* K */\n"
"    float *k = qkv + p * 3 * dim + dim + h * head_dim;\n"
"    float k0 = k[i], k1 = k[i + half];\n"
"    k[i]        = k0 * cos_t - k1 * sin_t;\n"
"    k[i + half] = k0 * sin_t + k1 * cos_t;\n"
"}\n"
"\n"
"/* ---- attn_full_f32: Full NxN self-attention per head ---- */\n"
"/* Grid: (n_heads), Block: (256) */\n"
"/* Uses shared memory for softmax reduction. */\n"
"__global__ void attn_full_f32(float *out, const float *qkv,\n"
"                                int n_patches, int dim, int n_heads,\n"
"                                int head_dim, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int dim3 = 3 * dim;\n"
"    /* Process queries sequentially */\n"
"    for (int qi = 0; qi < n_patches; qi++) {\n"
"        const float *q_h = qkv + qi * dim3 + h * head_dim;\n"
"        /* Compute scores QK^T for this query */\n"
"        for (int ki = tid; ki < n_patches; ki += nt) {\n"
"            const float *k_h = qkv + ki * dim3 + dim + h * head_dim;\n"
"            float score = 0.0f;\n"
"            for (int d = 0; d < head_dim; d++)\n"
"                score += q_h[d] * k_h[d];\n"
"            smem[ki] = score * scale;\n"
"        }\n"
"        __syncthreads();\n"
"        /* Softmax: find max */\n"
"        float local_max = -1e30f;\n"
"        for (int ki = tid; ki < n_patches; ki += nt)\n"
"            if (smem[ki] > local_max) local_max = smem[ki];\n"
"        /* Reduce max in shared memory */\n"
"        smem[n_patches + tid] = local_max;\n"
"        __syncthreads();\n"
"        for (int r = nt/2; r > 0; r >>= 1) {\n"
"            if (tid < r && smem[n_patches + tid + r] > smem[n_patches + tid])\n"
"                smem[n_patches + tid] = smem[n_patches + tid + r];\n"
"            __syncthreads();\n"
"        }\n"
"        float max_val = smem[n_patches];\n"
"        /* Exp and sum */\n"
"        float local_sum = 0.0f;\n"
"        for (int ki = tid; ki < n_patches; ki += nt) {\n"
"            smem[ki] = expf(smem[ki] - max_val);\n"
"            local_sum += smem[ki];\n"
"        }\n"
"        smem[n_patches + tid] = local_sum;\n"
"        __syncthreads();\n"
"        for (int r = nt/2; r > 0; r >>= 1) {\n"
"            if (tid < r) smem[n_patches + tid] += smem[n_patches + tid + r];\n"
"            __syncthreads();\n"
"        }\n"
"        float inv_sum = 1.0f / smem[n_patches];\n"
"        /* Normalize */\n"
"        for (int ki = tid; ki < n_patches; ki += nt)\n"
"            smem[ki] *= inv_sum;\n"
"        __syncthreads();\n"
"        /* Weighted sum of V */\n"
"        for (int d = tid; d < head_dim; d += nt) {\n"
"            float sum = 0.0f;\n"
"            for (int vi = 0; vi < n_patches; vi++) {\n"
"                sum += smem[vi] * qkv[vi * dim3 + 2 * dim + h * head_dim + d];\n"
"            }\n"
"            out[qi * dim + h * head_dim + d] = sum;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"}\n"
"\n"
"/* ---- spatial_merge_f32: Gather 2x2 patches into merged tokens ---- */\n"
"/* Grid: (n_merged), Block: (256) */\n"
"__global__ void spatial_merge_f32(float *dst, const float *src,\n"
"                                    int gw, int sm, int dim) {\n"
"    int m = blockIdx.x;\n"
"    int mgw = gw / sm;\n"
"    int my = m / mgw, mx = m % mgw;\n"
"    int merged_dim = dim * sm * sm;\n"
"    int tid = threadIdx.x;\n"
"    for (int di = tid; di < merged_dim; di += blockDim.x) {\n"
"        int sub = di / dim;\n"
"        int d = di % dim;\n"
"        int sy = sub / sm, sx = sub % sm;\n"
"        int py = my * sm + sy;\n"
"        int px = mx * sm + sx;\n"
"        dst[m * merged_dim + di] = src[(py * gw + px) * dim + d];\n"
"    }\n"
"}\n"
"\n"
"} /* extern C */\n"
;

/* ======================================================================== */
/* Runner struct                                                            */
/* ======================================================================== */

typedef struct {
    CUdeviceptr w_f32;   /* F32 weight [n_out, n_in] */
    CUdeviceptr w_f16;   /* F16 weight [n_out, n_in] (if use_f16) */
    CUdeviceptr bias;    /* F32 bias [n_out] (always F32) */
} gpu_weight;

typedef struct {
    gpu_weight attn_qkv;    /* [3*dim, dim] */
    gpu_weight attn_out;    /* [dim, dim] */
    gpu_weight ffn_up;      /* [ffn_dim, dim] */
    gpu_weight ffn_down;    /* [dim, ffn_dim] */
    CUdeviceptr ln1_w, ln1_b;  /* F32 [dim] */
    CUdeviceptr ln2_w, ln2_b;  /* F32 [dim] */
} gpu_vit_block;

typedef struct {
    gpu_weight fc1;   /* [merged_dim, merged_dim] */
    gpu_weight fc2;   /* [proj_dim, merged_dim] */
    CUdeviceptr norm_w, norm_b;  /* F32 [merged_dim] */
} gpu_deepstack;

struct cuda_vision_runner {
    CUdevice device;
    CUcontext context;
    CUstream stream;
    int verbose;
    int use_f16;

    CUmodule module;
    /* Shared kernels */
    CUfunction fn_layernorm_f32;
    CUfunction fn_gemm_f16_f32;
    CUfunction fn_gelu_f32;
    CUfunction fn_add_f32;
    CUfunction fn_add_bias_f32;
    /* Vision-specific kernels */
    CUfunction fn_gemm_f32_f32;
    CUfunction fn_patch_embed_dual_f32;
    CUfunction fn_add_pos_embd;
    CUfunction fn_add_pos_embd_direct;
    CUfunction fn_rope_vision_f32;
    CUfunction fn_attn_full_f32;
    CUfunction fn_spatial_merge_f32;

    /* Model hyperparams */
    int n_blocks;
    int dim;
    int n_heads;
    int head_dim;
    int ffn_dim;
    int patch_size;
    int image_size;
    int n_patches;
    int proj_dim;
    int spatial_merge;
    int n_merged;
    float ln_eps;
    float image_mean[3];
    float image_std[3];

    /* Dynamic resolution support */
    int max_patches;           /* max patches for buffer allocation (0 = use n_patches) */
    int max_merged;            /* max merged tokens */
    int max_pixels;            /* max pixel count for RGB buffer */
    float *h_pos_embd;         /* CPU copy of original pos embedding [n_patches * dim] */
    CUdeviceptr d_pos_interp;  /* GPU buffer for interpolated pos embedding [max_patches * dim] */

    /* GPU weights: patch embeddings */
    CUdeviceptr d_patch_w0;     /* F32 [dim, ps*ps*3] */
    CUdeviceptr d_patch_w1;     /* F32 [dim, ps*ps*3] (second conv, may be 0) */
    CUdeviceptr d_patch_bias;   /* F32 [dim] */

    /* Position embedding */
    CUdeviceptr d_pos_embd;     /* F32 [n_patches, dim] */

    /* Blocks */
    gpu_vit_block *blocks;

    /* DeepStack */
    int n_deepstack;
    int *deepstack_indices;
    gpu_deepstack *deepstack;

    /* Post LN */
    CUdeviceptr d_post_ln_w, d_post_ln_b;  /* F32 [dim] */

    /* MM projection */
    gpu_weight mm0;   /* [merged_dim, merged_dim] */
    gpu_weight mm2;   /* [proj_dim, merged_dim] */

    /* Scratch buffers (allocated on load) */
    CUdeviceptr d_hidden;     /* [max_patches * dim] */
    CUdeviceptr d_hidden2;    /* [max_patches * dim] */
    CUdeviceptr d_qkv;        /* [max_patches * 3 * dim] */
    CUdeviceptr d_attn_out;   /* [max_patches * dim] */
    CUdeviceptr d_ffn_buf;    /* [max_patches * ffn_dim] */
    CUdeviceptr d_ln_buf;     /* [max_patches * dim] */
    CUdeviceptr d_merge_buf;  /* [n_merged * merged_dim] */
    CUdeviceptr d_mm_buf;     /* [n_merged * merged_dim] */
    CUdeviceptr d_mm_out;     /* [n_merged * proj_dim] */
    CUdeviceptr d_rgb;        /* [max_pixels * 3] */
    CUdeviceptr d_rope_cos;   /* [max_patches * head_dim] */
    CUdeviceptr d_rope_sin;   /* [max_patches * head_dim] */
    CUdeviceptr d_pos_map;    /* [max_patches] int */
    CUdeviceptr d_ds_feats;   /* deepstack feature accumulation */

    /* Host output */
    float *h_output;
    int loaded;
};

/* ======================================================================== */
/* NVRTC compilation                                                        */
/* ======================================================================== */

static int vlm_compile_kernels(cuda_vision_runner *r) {
    size_t len1 = strlen(cuda_kernels_common_src);
    size_t len2 = strlen(cuda_vlm_specific_kernels);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, cuda_kernels_common_src, len1);
    memcpy(full_src + len1, cuda_vlm_specific_kernels, len2 + 1);

    /* Use custom NVRTC compilation without --use_fast_math to ensure
     * correct MMA tensor core behavior on Blackwell (sm_120) */
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, r->device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, r->device);
    int sm = major * 10 + minor;

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_vlm: compiling kernels for sm_%d ...\n", sm);

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, full_src, "vlm_kernels.cu", 0, NULL, NULL) != NVRTC_SUCCESS) {
        free(full_src);
        return -1;
    }

    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d", sm);
    /* No --use_fast_math: ensures correct F16 MMA behavior */
    const char *opts[] = { arch };
    nvrtcResult nres = nvrtcCompileProgram(prog, 1, opts);

    if (nres != NVRTC_SUCCESS) {
        size_t log_sz;
        nvrtcGetProgramLogSize(prog, &log_sz);
        if (log_sz > 1) {
            char *log = (char *)malloc(log_sz);
            nvrtcGetProgramLog(prog, log);
            fprintf(stderr, "cuda_vlm: NVRTC log:\n%s\n", log);
            free(log);
        }
        nvrtcDestroyProgram(&prog);
        free(full_src);
        return -1;
    }

    size_t ptx_sz;
    nvrtcGetPTXSize(prog, &ptx_sz);
    char *ptx = (char *)malloc(ptx_sz);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);
    free(full_src);

    if (r->verbose >= 3) {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/vlm_kernels.ptx");
        FILE *fp = fopen(path, "w");
        if (fp) { fwrite(ptx, 1, ptx_sz, fp); fclose(fp);
            fprintf(stderr, "cuda_vlm: PTX saved to %s\n", path); }
    }

    {
        CUresult lerr = cuModuleLoadDataEx(&r->module, ptx, 0, NULL, NULL);
        free(ptx);
        if (lerr != CUDA_SUCCESS) return -1;
    }

    CUresult err;
#define GET_FN(name) do { \
    err = cuModuleGetFunction(&r->fn_##name, r->module, #name); \
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_vlm: kernel '%s' not found\n", #name); return -1; } \
} while(0)

    /* Shared kernels */
    GET_FN(layernorm_f32);
    GET_FN(gemm_f16_f32);
    GET_FN(gelu_f32);
    GET_FN(add_f32);
    GET_FN(add_bias_f32);

    /* Vision-specific kernels */
    GET_FN(gemm_f32_f32);
    GET_FN(patch_embed_dual_f32);
    GET_FN(add_pos_embd);
    GET_FN(add_pos_embd_direct);
    GET_FN(rope_vision_f32);
    GET_FN(attn_full_f32);
    GET_FN(spatial_merge_f32);

#undef GET_FN

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_vlm: %d kernels compiled (sm_%d)\n", 12, sm);
    return 0;
}

/* ======================================================================== */
/* Weight upload helpers                                                    */
/* ======================================================================== */

/* Helper to find a tensor in GGUF by name */
static int vlm_find_tensor(const gguf_context *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

/* Helper struct for tensor info */
typedef struct {
    const void *data;
    int type;
    int n_cols;
    int n_rows;
    int n_elem;
} vlm_tensor_info;

static vlm_tensor_info vlm_get_tensor(const gguf_context *g, const char *name, int req) {
    vlm_tensor_info t = {0};
    int idx = vlm_find_tensor(g, name);
    if (idx < 0) {
        if (req) fprintf(stderr, "cuda_vlm: missing tensor '%s'\n", name);
        return t;
    }
    t.data = gguf_tensor_data(g, idx);
    t.type = (int)g->tensors[idx].type;
    t.n_cols = (int)g->tensors[idx].dims[0];
    t.n_rows = (g->tensors[idx].n_dims >= 2) ? (int)g->tensors[idx].dims[1] : 1;
    /* Compute total elements as product of all dimensions */
    t.n_elem = 1;
    for (int d = 0; d < (int)g->tensors[idx].n_dims; d++)
        t.n_elem *= (int)g->tensors[idx].dims[d];
    return t;
}

static int vlm_get_int(const gguf_context *g, const char *key, int def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_UINT32) return (int)g->kv[idx].value.u32;
    if (g->kv[idx].type == GGUF_TYPE_INT32) return g->kv[idx].value.i32;
    return def;
}

static float vlm_get_float(const gguf_context *g, const char *key, float def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_FLOAT32) return g->kv[idx].value.f32;
    return def;
}

/* Dequantize full tensor to F32 and upload */
static CUdeviceptr vlm_upload_f32(const vlm_tensor_info *t) {
    if (!t->data) return 0;
    int n = t->n_elem;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        /* Dequantize row by row */
        size_t rb = dequant_row_size(t->type, t->n_cols);
        for (int row = 0; row < t->n_rows; row++) {
            const void *row_data = (const uint8_t *)t->data + row * rb;
            dequant_row(t->type, row_data, buf + row * t->n_cols, t->n_cols);
        }
    }
    CUdeviceptr d;
    if (cuMemAlloc(&d, (size_t)n * sizeof(float)) != CUDA_SUCCESS) { free(buf); return 0; }
    cuMemcpyHtoD(d, buf, (size_t)n * sizeof(float));
    free(buf);
    return d;
}

/* Upload tensor as F16 (converting from F32 if needed) */
static CUdeviceptr vlm_upload_f16(const vlm_tensor_info *t) {
    if (!t->data) return 0;
    int n = t->n_elem;
    if (t->type == GGML_TYPE_F16) {
        /* Direct copy */
        return cu_upload_raw(t->data, (size_t)n * 2);
    }
    /* Convert F32 -> F16 */
    float *f32_buf = NULL;
    if (t->type == GGML_TYPE_F32) {
        f32_buf = (float *)t->data;
    } else {
        /* Dequant to F32 first */
        f32_buf = (float *)malloc((size_t)n * sizeof(float));
        size_t rb = dequant_row_size(t->type, t->n_cols);
        for (int row = 0; row < t->n_rows; row++) {
            const void *row_data = (const uint8_t *)t->data + row * rb;
            dequant_row(t->type, row_data, f32_buf + row * t->n_cols, t->n_cols);
        }
    }
    uint16_t *h16 = (uint16_t *)malloc((size_t)n * 2);
    for (int i = 0; i < n; i++) h16[i] = cu_f32_to_f16(f32_buf[i]);
    if (f32_buf != (float *)t->data) free(f32_buf);
    CUdeviceptr d;
    if (cuMemAlloc(&d, (size_t)n * 2) != CUDA_SUCCESS) { free(h16); return 0; }
    cuMemcpyHtoD(d, h16, (size_t)n * 2);
    free(h16);
    return d;
}

/* Upload a weight matrix (with optional F16 for performance mode) */
static gpu_weight vlm_upload_weight(const vlm_tensor_info *w, const vlm_tensor_info *b, int use_f16) {
    gpu_weight gw = {0};
    if (use_f16) {
        gw.w_f16 = vlm_upload_f16(w);
    } else {
        gw.w_f32 = vlm_upload_f32(w);
    }
    if (b && b->data) {
        gw.bias = vlm_upload_f32(b);
    }
    return gw;
}

/* ======================================================================== */
/* Public API: init                                                         */
/* ======================================================================== */

cuda_vision_runner *cuda_vision_init(int device_id, int verbose, int use_f16) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuda_vlm: cuew init failed (no CUDA/NVRTC?)\n");
        return NULL;
    }
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_vlm: cuInit failed\n");
        return NULL;
    }

    cuda_vision_runner *r = (cuda_vision_runner *)calloc(1, sizeof(cuda_vision_runner));
    r->verbose = verbose;
    r->use_f16 = use_f16;

    CU_CHECK_NULL(cuDeviceGet(&r->device, device_id));
    CU_CHECK_NULL(cuCtxCreate(&r->context, 0, r->device));
    CU_CHECK_NULL(cuStreamCreate(&r->stream, CU_STREAM_DEFAULT));

    if (vlm_compile_kernels(r) != 0) {
        fprintf(stderr, "cuda_vlm: kernel compilation failed\n");
        free(r);
        return NULL;
    }

    return r;
}

void cuda_vision_set_max_pixels(cuda_vision_runner *r, int max_pixels) {
    if (r) r->max_pixels = max_pixels;
}

/* ======================================================================== */
/* Public API: load_weights                                                 */
/* ======================================================================== */

int cuda_vision_load_weights(cuda_vision_runner *r, gguf_context *g) {
    if (!r || !g) return -1;

    /* Read hyperparameters */
    r->n_blocks    = vlm_get_int(g, "clip.vision.block_count", 24);
    r->dim         = vlm_get_int(g, "clip.vision.embedding_length", 1024);
    r->n_heads     = vlm_get_int(g, "clip.vision.attention.head_count", 16);
    r->ffn_dim     = vlm_get_int(g, "clip.vision.feed_forward_length", 4096);
    r->patch_size  = vlm_get_int(g, "clip.vision.patch_size", 16);
    r->image_size  = vlm_get_int(g, "clip.vision.image_size", 768);
    r->proj_dim    = vlm_get_int(g, "clip.vision.projection_dim", 2048);
    r->spatial_merge = vlm_get_int(g, "clip.vision.spatial_merge_size", 2);
    r->ln_eps      = vlm_get_float(g, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    r->head_dim    = r->dim / r->n_heads;

    int ps = r->patch_size;
    int gs = r->image_size / ps;
    r->n_patches = gs * gs;
    r->n_merged  = r->n_patches / (r->spatial_merge * r->spatial_merge);

    /* Compute max buffer sizes for dynamic resolution */
    {
        int mp = r->max_pixels > 0 ? r->max_pixels / (ps * ps) : r->n_patches;
        if (mp < r->n_patches) mp = r->n_patches;
        r->max_patches = mp;
        r->max_merged = mp / (r->spatial_merge * r->spatial_merge);
    }

    /* Image mean/std */
    int idx = gguf_find_key(g, "clip.vision.image_mean");
    if (idx >= 0) {
        float *d = (float *)g->kv[idx].value.arr.data;
        r->image_mean[0] = d[0]; r->image_mean[1] = d[1]; r->image_mean[2] = d[2];
    }
    idx = gguf_find_key(g, "clip.vision.image_std");
    if (idx >= 0) {
        float *d = (float *)g->kv[idx].value.arr.data;
        r->image_std[0] = d[0]; r->image_std[1] = d[1]; r->image_std[2] = d[2];
    }

    fprintf(stderr, "cuda_vlm: dim=%d heads=%d blocks=%d ffn=%d patch=%d image=%d patches=%d merged=%d proj=%d f16=%d max_patches=%d\n",
            r->dim, r->n_heads, r->n_blocks, r->ffn_dim,
            r->patch_size, r->image_size, r->n_patches, r->n_merged, r->proj_dim, r->use_f16,
            r->max_patches);

    int dim = r->dim;
    int mp = r->max_patches;
    int max_merged = r->max_merged;
    int sm = r->spatial_merge;
    int merged_dim = dim * sm * sm;

    /* Patch embeddings (always F32 — small, applied once) */
    vlm_tensor_info t_pw0 = vlm_get_tensor(g, "v.patch_embd.weight", 1);
    vlm_tensor_info t_pw1 = vlm_get_tensor(g, "v.patch_embd.weight.1", 0);
    vlm_tensor_info t_pb  = vlm_get_tensor(g, "v.patch_embd.bias", 0);
    r->d_patch_w0 = vlm_upload_f32(&t_pw0);
    r->d_patch_w1 = vlm_upload_f32(&t_pw1);
    r->d_patch_bias = vlm_upload_f32(&t_pb);
    if (t_pw1.data)
        fprintf(stderr, "cuda_vlm: loaded dual conv2d patch embeddings\n");

    /* Position embedding (always F32) — keep CPU copy for interpolation */
    vlm_tensor_info t_pos = vlm_get_tensor(g, "v.position_embd.weight", 1);
    r->d_pos_embd = vlm_upload_f32(&t_pos);
    r->h_pos_embd = (float *)malloc(t_pos.n_elem * sizeof(float));
    if (t_pos.type == GGML_TYPE_F32) {
        memcpy(r->h_pos_embd, t_pos.data, t_pos.n_elem * sizeof(float));
    } else {
        dequant_row(t_pos.type, t_pos.data, r->h_pos_embd, t_pos.n_elem);
    }

    /* Blocks */
    r->blocks = (gpu_vit_block *)calloc(r->n_blocks, sizeof(gpu_vit_block));
    for (int l = 0; l < r->n_blocks; l++) {
        char name[128];
        gpu_vit_block *blk = &r->blocks[l];

        /* QKV */
        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.weight", l);
        vlm_tensor_info tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.bias", l);
        vlm_tensor_info tb = vlm_get_tensor(g, name, 1);
        blk->attn_qkv = vlm_upload_weight(&tw, &tb, r->use_f16);

        /* Attn out */
        snprintf(name, sizeof(name), "v.blk.%d.attn_out.weight", l);
        tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.attn_out.bias", l);
        tb = vlm_get_tensor(g, name, 1);
        blk->attn_out = vlm_upload_weight(&tw, &tb, r->use_f16);

        /* FFN up */
        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.weight", l);
        tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.bias", l);
        tb = vlm_get_tensor(g, name, 1);
        blk->ffn_up = vlm_upload_weight(&tw, &tb, r->use_f16);

        /* FFN down */
        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.weight", l);
        tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.bias", l);
        tb = vlm_get_tensor(g, name, 1);
        blk->ffn_down = vlm_upload_weight(&tw, &tb, r->use_f16);

        /* LayerNorms (always F32) */
        snprintf(name, sizeof(name), "v.blk.%d.ln1.weight", l);
        vlm_tensor_info tln = vlm_get_tensor(g, name, 1);
        blk->ln1_w = vlm_upload_f32(&tln);
        snprintf(name, sizeof(name), "v.blk.%d.ln1.bias", l);
        tln = vlm_get_tensor(g, name, 1);
        blk->ln1_b = vlm_upload_f32(&tln);

        snprintf(name, sizeof(name), "v.blk.%d.ln2.weight", l);
        tln = vlm_get_tensor(g, name, 1);
        blk->ln2_w = vlm_upload_f32(&tln);
        snprintf(name, sizeof(name), "v.blk.%d.ln2.bias", l);
        tln = vlm_get_tensor(g, name, 1);
        blk->ln2_b = vlm_upload_f32(&tln);
    }

    /* DeepStack */
    idx = gguf_find_key(g, "clip.vision.is_deepstack_layers");
    if (idx >= 0) {
        uint8_t *flags = (uint8_t *)g->kv[idx].value.arr.data;
        int n = (int)g->kv[idx].value.arr.n;
        int ns = 0;
        for (int i = 0; i < n; i++) if (flags[i]) ns++;
        r->n_deepstack = ns;
        r->deepstack_indices = (int *)malloc(ns * sizeof(int));
        r->deepstack = (gpu_deepstack *)calloc(ns, sizeof(gpu_deepstack));
        int si = 0;
        for (int i = 0; i < n; i++) {
            if (!flags[i]) continue;
            r->deepstack_indices[si] = i;
            char name[128];

            snprintf(name, sizeof(name), "v.deepstack.%d.fc1.weight", i);
            vlm_tensor_info tw = vlm_get_tensor(g, name, 1);
            snprintf(name, sizeof(name), "v.deepstack.%d.fc1.bias", i);
            vlm_tensor_info tb = vlm_get_tensor(g, name, 1);
            r->deepstack[si].fc1 = vlm_upload_weight(&tw, &tb, r->use_f16);

            snprintf(name, sizeof(name), "v.deepstack.%d.fc2.weight", i);
            tw = vlm_get_tensor(g, name, 1);
            snprintf(name, sizeof(name), "v.deepstack.%d.fc2.bias", i);
            tb = vlm_get_tensor(g, name, 1);
            r->deepstack[si].fc2 = vlm_upload_weight(&tw, &tb, r->use_f16);

            snprintf(name, sizeof(name), "v.deepstack.%d.norm.weight", i);
            vlm_tensor_info tln = vlm_get_tensor(g, name, 1);
            r->deepstack[si].norm_w = vlm_upload_f32(&tln);
            snprintf(name, sizeof(name), "v.deepstack.%d.norm.bias", i);
            tln = vlm_get_tensor(g, name, 1);
            r->deepstack[si].norm_b = vlm_upload_f32(&tln);

            si++;
        }
        fprintf(stderr, "cuda_vlm: %d deepstack layers at:", ns);
        for (int i = 0; i < ns; i++) fprintf(stderr, " %d", r->deepstack_indices[i]);
        fprintf(stderr, "\n");
    }

    /* Post LN */
    vlm_tensor_info tln = vlm_get_tensor(g, "v.post_ln.weight", 1);
    r->d_post_ln_w = vlm_upload_f32(&tln);
    tln = vlm_get_tensor(g, "v.post_ln.bias", 1);
    r->d_post_ln_b = vlm_upload_f32(&tln);

    /* MM projection */
    vlm_tensor_info tw, tb;
    tw = vlm_get_tensor(g, "mm.0.weight", 1);
    tb = vlm_get_tensor(g, "mm.0.bias", 1);
    r->mm0 = vlm_upload_weight(&tw, &tb, r->use_f16);

    tw = vlm_get_tensor(g, "mm.2.weight", 1);
    tb = vlm_get_tensor(g, "mm.2.bias", 1);
    r->mm2 = vlm_upload_weight(&tw, &tb, r->use_f16);

    /* Allocate scratch buffers (sized for max_patches, not n_patches) */
    {
        size_t rgb_pixels = r->max_pixels > 0 ? (size_t)r->max_pixels : (size_t)r->image_size * r->image_size;
        CHECK_CU(cuMemAlloc(&r->d_hidden,    (size_t)mp * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_hidden2,   (size_t)mp * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_qkv,       (size_t)mp * 3 * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_attn_out,  (size_t)mp * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ffn_buf,   (size_t)mp * r->ffn_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ln_buf,    (size_t)mp * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_merge_buf, (size_t)max_merged * merged_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_mm_buf,    (size_t)max_merged * merged_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_mm_out,    (size_t)max_merged * r->proj_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_rgb,       rgb_pixels * 3 * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_rope_cos,  (size_t)mp * r->head_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_rope_sin,  (size_t)mp * r->head_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_pos_map,   (size_t)mp * sizeof(int)));
        CHECK_CU(cuMemAlloc(&r->d_pos_interp,(size_t)mp * dim * sizeof(float)));
    }

    /* DeepStack feature buffer */
    if (r->n_deepstack > 0) {
        CHECK_CU(cuMemAlloc(&r->d_ds_feats,
            (size_t)max_merged * r->n_deepstack * r->proj_dim * sizeof(float)));
    }

    int total_embd = r->proj_dim * (1 + r->n_deepstack);
    r->h_output = (float *)malloc((size_t)max_merged * total_embd * sizeof(float));

    r->loaded = 1;
    fprintf(stderr, "cuda_vlm: weights loaded, VRAM for weights ~%.1f MB\n",
            (r->use_f16 ? 0.5f : 1.0f) * (float)(
                (size_t)r->n_blocks * (3*dim*dim + dim*dim + r->ffn_dim*dim + dim*r->ffn_dim) +
                merged_dim*merged_dim + r->proj_dim*merged_dim +
                r->n_deepstack * (merged_dim*merged_dim + r->proj_dim*merged_dim)
            ) * sizeof(float) / (1024.0f * 1024.0f));

    return 0;
}

/* ======================================================================== */
/* GEMM dispatch: F32 or F16                                                */
/* ======================================================================== */

/* Launch a GEMM: Y[n_tok, n_out] = X[n_tok, n_in] * W^T[n_out, n_in] + bias */
static void vlm_gemm(cuda_vision_runner *r, CUdeviceptr d_Y, const gpu_weight *w,
                       CUdeviceptr d_X, int n_tok, int n_out, int n_in) {
    /* Copy const fields to locals for void* args array */
    CUdeviceptr d_W, d_bias;
    d_bias = w->bias;
    if (r->use_f16 && w->w_f16) {
        /* F16 MMA tensor core path */
        d_W = w->w_f16;
        int grid_x = (n_out + 255) / 256;
        int grid_y = (n_tok + 15) / 16;
        size_t smem = 16 * 16 * sizeof(float);
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok };
        cuLaunchKernel(r->fn_gemm_f16_f32,
                       grid_x, grid_y, 1,
                       128, 1, 1,
                       smem, r->stream,
                       args, NULL);
    } else {
        /* F32 tiled path */
        d_W = w->w_f32;
        int grid_x = (n_out + 15) / 16;
        int grid_y = (n_tok + 15) / 16;
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok };
        cuLaunchKernel(r->fn_gemm_f32_f32,
                       grid_x, grid_y, 1,
                       16, 16, 1,
                       0, r->stream,
                       args, NULL);
    }
}

/* ======================================================================== */
/* Public API: encode                                                       */
/* ======================================================================== */

float *cuda_vision_encode(cuda_vision_runner *r, const float *rgb_norm, int width, int height) {
    if (!r || !r->loaded) return NULL;

    int ps = r->patch_size;
    int dim = r->dim;
    int n_heads = r->n_heads;
    int head_dim = r->head_dim;
    int ffn_dim = r->ffn_dim;
    int gw = width / ps;
    int gh = height / ps;
    int n_patches = gw * gh;
    int sm = r->spatial_merge;
    int merged_dim = dim * sm * sm;
    int n_merged = n_patches / (sm * sm);

    if (n_patches > r->max_patches) {
        fprintf(stderr, "cuda_vlm: too many patches %d (max %d)\n", n_patches, r->max_patches);
        return NULL;
    }

    fprintf(stderr, "cuda_vlm: encoding %dx%d image (%d patches, %d merged tokens)\n",
            width, height, n_patches, n_merged);

    /* 1. Upload RGB to GPU */
    cuMemcpyHtoD(r->d_rgb, rgb_norm, (size_t)width * height * 3 * sizeof(float));

    /* 2. Patch embedding (dual conv2d) */
    fprintf(stderr, "  patch embedding (dual conv2d)...\n");
    {
        int img_w = width;
        void *args[] = {
            &r->d_hidden, &r->d_rgb,
            &r->d_patch_w0, &r->d_patch_w1, &r->d_patch_bias,
            &gw, &dim, &ps, &img_w
        };
        cuLaunchKernel(r->fn_patch_embed_dual_f32,
                       n_patches, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    }

    /* Debug: check patch embedding output */
    if (r->verbose >= 2) {
        cuStreamSynchronize(r->stream);
        float dbg[8];
        cuMemcpyDtoH(dbg, r->d_hidden, 8 * sizeof(float));
        fprintf(stderr, "  [DBG] hidden after patch_embed: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
    }

    /* 3. Position embeddings (bilinear interpolation for dynamic resolution) */
    fprintf(stderr, "  position embeddings...\n");
    {
        int orig_gw = r->image_size / ps;
        int orig_gh = orig_gw;  /* original grid is square */

        if (gw == orig_gw && gh == orig_gh) {
            /* Exact match: use direct indirection (no interpolation needed) */
            int *pos_map = (int *)malloc(n_patches * sizeof(int));
            for (int py = 0; py < gh; py++)
                for (int px = 0; px < gw; px++)
                    pos_map[py * gw + px] = py * orig_gw + px;
            cuMemcpyHtoD(r->d_pos_map, pos_map, n_patches * sizeof(int));
            free(pos_map);

            void *args[] = { &r->d_hidden, &r->d_pos_embd, &r->d_pos_map, &dim };
            cuLaunchKernel(r->fn_add_pos_embd,
                           n_patches, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        } else {
            /* Bilinear interpolation on CPU, upload to d_pos_interp */
            fprintf(stderr, "  interpolating pos embedding: %dx%d -> %dx%d\n",
                    orig_gw, orig_gh, gw, gh);
            float *interp = (float *)malloc((size_t)n_patches * dim * sizeof(float));
            for (int py = 0; py < gh; py++) {
                float sy = (float)py * (orig_gh - 1) / (gh > 1 ? gh - 1 : 1);
                int y0 = (int)sy, y1 = (y0 + 1 < orig_gh) ? y0 + 1 : y0;
                float wy = sy - y0;
                for (int px = 0; px < gw; px++) {
                    float sx = (float)px * (orig_gw - 1) / (gw > 1 ? gw - 1 : 1);
                    int x0 = (int)sx, x1 = (x0 + 1 < orig_gw) ? x0 + 1 : x0;
                    float wx = sx - x0;
                    int dst_idx = (py * gw + px) * dim;
                    int s00 = (y0 * orig_gw + x0) * dim;
                    int s01 = (y0 * orig_gw + x1) * dim;
                    int s10 = (y1 * orig_gw + x0) * dim;
                    int s11 = (y1 * orig_gw + x1) * dim;
                    for (int d = 0; d < dim; d++) {
                        interp[dst_idx + d] =
                            r->h_pos_embd[s00+d] * (1-wy)*(1-wx) +
                            r->h_pos_embd[s01+d] * (1-wy)*wx +
                            r->h_pos_embd[s10+d] * wy*(1-wx) +
                            r->h_pos_embd[s11+d] * wy*wx;
                    }
                }
            }
            cuMemcpyHtoD(r->d_pos_interp, interp, (size_t)n_patches * dim * sizeof(float));
            free(interp);

            /* Add interpolated pos embedding directly (no pos_map needed) */
            void *args[] = { &r->d_hidden, &r->d_pos_interp, &dim, &n_patches };
            cuLaunchKernel(r->fn_add_pos_embd_direct,
                           n_patches, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }
    }

    /* 4. Precompute M-RoPE cos/sin on host, upload to GPU */
    {
        int half = head_dim / 2;
        int sect_size = head_dim / 4;
        float freq_base = 10000.0f;
        float theta_scale = powf(freq_base, -2.0f / (float)half);
        float *rope_cos = (float *)malloc(n_patches * head_dim * sizeof(float));
        float *rope_sin = (float *)malloc(n_patches * head_dim * sizeof(float));

        for (int p = 0; p < n_patches; p++) {
            int py = p / gw;
            int px = p % gw;
            float p_t = (float)py, p_h = (float)px, p_w = (float)py, p_e = (float)px;
            float cur_t = p_t, cur_h = p_h, cur_w = p_w, cur_e = p_e;

            for (int i0 = 0; i0 < head_dim; i0 += 2) {
                int sector = i0 / 2;
                if (sector == 0) cur_t = p_t;
                if (sector == sect_size) cur_h = p_h;
                if (sector == 2 * sect_size) cur_w = p_w;
                if (sector == 3 * sect_size) cur_e = p_e;

                float theta;
                if (sector < sect_size) theta = cur_t;
                else if (sector < 2 * sect_size) theta = cur_h;
                else if (sector < 3 * sect_size) theta = cur_w;
                else theta = cur_e;

                rope_cos[p * head_dim + i0] = cosf(theta);
                rope_sin[p * head_dim + i0] = sinf(theta);
                rope_cos[p * head_dim + i0 + 1] = cosf(theta);
                rope_sin[p * head_dim + i0 + 1] = sinf(theta);

                cur_t *= theta_scale;
                cur_h *= theta_scale;
                cur_w *= theta_scale;
                cur_e *= theta_scale;
            }
        }

        cuMemcpyHtoD(r->d_rope_cos, rope_cos, n_patches * head_dim * sizeof(float));
        cuMemcpyHtoD(r->d_rope_sin, rope_sin, n_patches * head_dim * sizeof(float));
        free(rope_cos);
        free(rope_sin);
    }

    /* 5. ViT blocks */
    int half = head_dim / 2;
    int ds_count = 0;

    for (int l = 0; l < r->n_blocks; l++) {
        if (l == 0 || l == r->n_blocks - 1 || (l + 1) % 6 == 0)
            fprintf(stderr, "  vit block %d/%d\n", l, r->n_blocks);

        gpu_vit_block *blk = &r->blocks[l];

        /* LayerNorm1 */
        {
            float eps = r->ln_eps;
            size_t smem = 256 * sizeof(float);
            void *args[] = { &r->d_ln_buf, &r->d_hidden, &blk->ln1_w, &blk->ln1_b, &dim, &eps };
            cuLaunchKernel(r->fn_layernorm_f32,
                           n_patches, 1, 1,
                           256, 1, 1,
                           smem, r->stream,
                           args, NULL);
        }

        /* QKV projection */
        {
            int n_out = 3 * dim;
            vlm_gemm(r, r->d_qkv, &blk->attn_qkv, r->d_ln_buf, n_patches, n_out, dim);
        }

        /* Debug: check QKV right after GEMM (before RoPE) */
        if (l == 0 && r->verbose >= 2) {
            cuStreamSynchronize(r->stream);
            float dbg[8];
            cuMemcpyDtoH(dbg, r->d_qkv, 8 * sizeof(float));
            fprintf(stderr, "  [DBG] qkv after GEMM (pre-RoPE), block 0: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
            cuMemcpyDtoH(dbg, r->d_ln_buf, 8 * sizeof(float));
            fprintf(stderr, "  [DBG] ln_buf (QKV input), block 0: %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            /* Check weight values */
            if (r->use_f16 && blk->attn_qkv.w_f16) {
                uint16_t wdbg[8];
                cuMemcpyDtoH(wdbg, blk->attn_qkv.w_f16, 8 * sizeof(uint16_t));
                fprintf(stderr, "  [DBG] QKV weight F16[0..3]: %04x %04x %04x %04x (%.6f %.6f %.6f %.6f)\n",
                        wdbg[0], wdbg[1], wdbg[2], wdbg[3],
                        ggml_fp16_to_fp32(wdbg[0]), ggml_fp16_to_fp32(wdbg[1]),
                        ggml_fp16_to_fp32(wdbg[2]), ggml_fp16_to_fp32(wdbg[3]));
            } else if (blk->attn_qkv.w_f32) {
                float wdbg[4];
                cuMemcpyDtoH(wdbg, blk->attn_qkv.w_f32, 4 * sizeof(float));
                fprintf(stderr, "  [DBG] QKV weight F32[0..3]: %.6f %.6f %.6f %.6f\n",
                        wdbg[0], wdbg[1], wdbg[2], wdbg[3]);
            }
        }

        /* M-RoPE on Q and K */
        {
            void *args[] = {
                &r->d_qkv, &r->d_rope_cos, &r->d_rope_sin,
                &n_patches, &n_heads, &dim, &head_dim, &half
            };
            cuLaunchKernel(r->fn_rope_vision_f32,
                           n_patches * n_heads, 1, 1,
                           half, 1, 1,
                           0, r->stream,
                           args, NULL);
        }

        /* Multi-head self-attention */
        {
            float scale = 1.0f / sqrtf((float)head_dim);
            /* Shared memory: n_patches (scores) + 256 (reduction scratch) */
            size_t smem = (n_patches + 256) * sizeof(float);
            void *args[] = {
                &r->d_attn_out, &r->d_qkv,
                &n_patches, &dim, &n_heads, &head_dim, &scale
            };
            cuLaunchKernel(r->fn_attn_full_f32,
                           n_heads, 1, 1,
                           256, 1, 1,
                           smem, r->stream,
                           args, NULL);
        }

        /* Attn output projection */
        vlm_gemm(r, r->d_hidden2, &blk->attn_out, r->d_attn_out, n_patches, dim, dim);

        /* Residual: hidden += hidden2 */
        {
            int n = n_patches * dim;
            int grid = (n + 255) / 256;
            void *args[] = { &r->d_hidden, &r->d_hidden2, &n };
            cuLaunchKernel(r->fn_add_f32,
                           grid, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }

        /* LayerNorm2 */
        {
            float eps = r->ln_eps;
            size_t smem = 256 * sizeof(float);
            void *args[] = { &r->d_ln_buf, &r->d_hidden, &blk->ln2_w, &blk->ln2_b, &dim, &eps };
            cuLaunchKernel(r->fn_layernorm_f32,
                           n_patches, 1, 1,
                           256, 1, 1,
                           smem, r->stream,
                           args, NULL);
        }

        /* FFN: up -> GELU -> down */
        vlm_gemm(r, r->d_ffn_buf, &blk->ffn_up, r->d_ln_buf, n_patches, ffn_dim, dim);

        {
            int n = n_patches * ffn_dim;
            int grid = (n + 255) / 256;
            void *args[] = { &r->d_ffn_buf, &n };
            cuLaunchKernel(r->fn_gelu_f32,
                           grid, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }

        vlm_gemm(r, r->d_hidden2, &blk->ffn_down, r->d_ffn_buf, n_patches, dim, ffn_dim);

        /* Residual: hidden += hidden2 */
        {
            int n = n_patches * dim;
            int grid = (n + 255) / 256;
            void *args[] = { &r->d_hidden, &r->d_hidden2, &n };
            cuLaunchKernel(r->fn_add_f32,
                           grid, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }

        /* Debug: check hidden and qkv after first block */
        if (l == 0 && r->verbose >= 2) {
            cuStreamSynchronize(r->stream);
            float dbg[8];
            cuMemcpyDtoH(dbg, r->d_hidden, 8 * sizeof(float));
            fprintf(stderr, "  [DBG] hidden after block 0: %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            cuMemcpyDtoH(dbg, r->d_qkv, 8 * sizeof(float));
            fprintf(stderr, "  [DBG] qkv[0..3] after block 0: %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }

        /* DeepStack extraction */
        for (int ds = 0; ds < r->n_deepstack; ds++) {
            if (r->deepstack_indices[ds] != l) continue;

            fprintf(stderr, "  deepstack at layer %d\n", l);
            gpu_deepstack *dsl = &r->deepstack[ds];

            /* Spatial merge current hidden -> merge_buf */
            {
                void *args[] = { &r->d_merge_buf, &r->d_hidden, &gw, &sm, &dim };
                cuLaunchKernel(r->fn_spatial_merge_f32,
                               n_merged, 1, 1,
                               256, 1, 1,
                               0, r->stream,
                               args, NULL);
            }

            /* LayerNorm on merge_buf */
            {
                float eps = r->ln_eps;
                size_t smem = 256 * sizeof(float);
                void *args[] = { &r->d_merge_buf, &r->d_merge_buf,
                                 &dsl->norm_w, &dsl->norm_b, &merged_dim, &eps };
                cuLaunchKernel(r->fn_layernorm_f32,
                               n_merged, 1, 1,
                               256, 1, 1,
                               smem, r->stream,
                               args, NULL);
            }

            /* fc1: [merged_dim -> merged_dim] */
            vlm_gemm(r, r->d_mm_buf, &dsl->fc1, r->d_merge_buf, n_merged, merged_dim, merged_dim);

            /* GELU */
            {
                int n = n_merged * merged_dim;
                int grid = (n + 255) / 256;
                void *args[] = { &r->d_mm_buf, &n };
                cuLaunchKernel(r->fn_gelu_f32,
                               grid, 1, 1,
                               256, 1, 1,
                               0, r->stream,
                               args, NULL);
            }

            /* fc2: [merged_dim -> proj_dim] */
            vlm_gemm(r, r->d_mm_out, &dsl->fc2, r->d_mm_buf, n_merged, r->proj_dim, merged_dim);

            /* Copy to deepstack feature buffer at offset ds_count */
            {
                size_t offset = (size_t)ds_count * n_merged * r->proj_dim * sizeof(float);
                CUdeviceptr dst = r->d_ds_feats + offset;
                cuMemcpyDtoDAsync(dst, r->d_mm_out,
                                   (size_t)n_merged * r->proj_dim * sizeof(float),
                                   r->stream);
            }
            ds_count++;
        }
    }

    /* 6. Post LayerNorm */
    fprintf(stderr, "  post layernorm...\n");
    {
        float eps = r->ln_eps;
        size_t smem = 256 * sizeof(float);
        void *args[] = { &r->d_hidden, &r->d_hidden, &r->d_post_ln_w, &r->d_post_ln_b, &dim, &eps };
        cuLaunchKernel(r->fn_layernorm_f32,
                       n_patches, 1, 1,
                       256, 1, 1,
                       smem, r->stream,
                       args, NULL);
    }

    /* 7. Final spatial merge */
    fprintf(stderr, "  spatial merge...\n");
    {
        void *args[] = { &r->d_merge_buf, &r->d_hidden, &gw, &sm, &dim };
        cuLaunchKernel(r->fn_spatial_merge_f32,
                       n_merged, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    }

    /* 8. MM projection: mm.0 -> GELU -> mm.2 */
    fprintf(stderr, "  mm projection...\n");
    vlm_gemm(r, r->d_mm_buf, &r->mm0, r->d_merge_buf, n_merged, merged_dim, merged_dim);

    {
        int n = n_merged * merged_dim;
        int grid = (n + 255) / 256;
        void *args[] = { &r->d_mm_buf, &n };
        cuLaunchKernel(r->fn_gelu_f32,
                       grid, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    }

    vlm_gemm(r, r->d_mm_out, &r->mm2, r->d_mm_buf, n_merged, r->proj_dim, merged_dim);

    /* Debug: check mm_out */
    if (r->verbose >= 2) {
        cuStreamSynchronize(r->stream);
        float dbg[8];
        cuMemcpyDtoH(dbg, r->d_mm_out, 8 * sizeof(float));
        fprintf(stderr, "  [DBG] mm_out: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
        cuMemcpyDtoH(dbg, r->d_merge_buf, 8 * sizeof(float));
        fprintf(stderr, "  [DBG] merge_buf: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
        cuMemcpyDtoH(dbg, r->d_mm_buf, 8 * sizeof(float));
        fprintf(stderr, "  [DBG] mm_buf (after gelu): %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
    }

    /* 9. Synchronize and copy results to host */
    cuStreamSynchronize(r->stream);

    int total_embd = r->proj_dim * (1 + r->n_deepstack);
    float *result = (float *)calloc(n_merged * total_embd, sizeof(float));

    /* Copy main embeddings */
    float *mm_host = (float *)malloc((size_t)n_merged * r->proj_dim * sizeof(float));
    cuMemcpyDtoH(mm_host, r->d_mm_out, (size_t)n_merged * r->proj_dim * sizeof(float));

    /* Copy deepstack features */
    float *ds_host = NULL;
    if (ds_count > 0) {
        ds_host = (float *)malloc((size_t)ds_count * n_merged * r->proj_dim * sizeof(float));
        cuMemcpyDtoH(ds_host, r->d_ds_feats,
                      (size_t)ds_count * n_merged * r->proj_dim * sizeof(float));
    }

    /* Interleave: [main, ds0, ds1, ...] per token */
    for (int t = 0; t < n_merged; t++) {
        float *dst = result + t * total_embd;
        memcpy(dst, mm_host + t * r->proj_dim, r->proj_dim * sizeof(float));
        for (int d = 0; d < ds_count; d++) {
            memcpy(dst + (1 + d) * r->proj_dim,
                   ds_host + d * n_merged * r->proj_dim + t * r->proj_dim,
                   r->proj_dim * sizeof(float));
        }
    }

    free(mm_host);
    free(ds_host);

    fprintf(stderr, "  vision encoding done: %d tokens of dim %d (main %d + %d deepstack)\n",
            n_merged, total_embd, r->proj_dim, ds_count);

    return result;
}

/* ======================================================================== */
/* Public API: accessors                                                    */
/* ======================================================================== */

int cuda_vision_n_merged(const cuda_vision_runner *r) {
    return r ? r->n_merged : 0;
}

int cuda_vision_proj_dim(const cuda_vision_runner *r) {
    return r ? r->proj_dim : 0;
}

int cuda_vision_total_embd(const cuda_vision_runner *r) {
    return r ? r->proj_dim * (1 + r->n_deepstack) : 0;
}

/* ======================================================================== */
/* Public API: free                                                         */
/* ======================================================================== */

static void vlm_free_weight(gpu_weight *w) {
    if (w->w_f32) cuMemFree(w->w_f32);
    if (w->w_f16) cuMemFree(w->w_f16);
    if (w->bias) cuMemFree(w->bias);
    memset(w, 0, sizeof(*w));
}

void cuda_vision_free(cuda_vision_runner *r) {
    if (!r) return;

    /* Free weights */
    if (r->d_patch_w0) cuMemFree(r->d_patch_w0);
    if (r->d_patch_w1) cuMemFree(r->d_patch_w1);
    if (r->d_patch_bias) cuMemFree(r->d_patch_bias);
    if (r->d_pos_embd) cuMemFree(r->d_pos_embd);

    if (r->blocks) {
        for (int l = 0; l < r->n_blocks; l++) {
            gpu_vit_block *blk = &r->blocks[l];
            vlm_free_weight(&blk->attn_qkv);
            vlm_free_weight(&blk->attn_out);
            vlm_free_weight(&blk->ffn_up);
            vlm_free_weight(&blk->ffn_down);
            if (blk->ln1_w) cuMemFree(blk->ln1_w);
            if (blk->ln1_b) cuMemFree(blk->ln1_b);
            if (blk->ln2_w) cuMemFree(blk->ln2_w);
            if (blk->ln2_b) cuMemFree(blk->ln2_b);
        }
        free(r->blocks);
    }

    if (r->deepstack) {
        for (int i = 0; i < r->n_deepstack; i++) {
            vlm_free_weight(&r->deepstack[i].fc1);
            vlm_free_weight(&r->deepstack[i].fc2);
            if (r->deepstack[i].norm_w) cuMemFree(r->deepstack[i].norm_w);
            if (r->deepstack[i].norm_b) cuMemFree(r->deepstack[i].norm_b);
        }
        free(r->deepstack);
    }
    free(r->deepstack_indices);

    if (r->d_post_ln_w) cuMemFree(r->d_post_ln_w);
    if (r->d_post_ln_b) cuMemFree(r->d_post_ln_b);
    vlm_free_weight(&r->mm0);
    vlm_free_weight(&r->mm2);

    /* Free scratch buffers */
    if (r->d_hidden) cuMemFree(r->d_hidden);
    if (r->d_hidden2) cuMemFree(r->d_hidden2);
    if (r->d_qkv) cuMemFree(r->d_qkv);
    if (r->d_attn_out) cuMemFree(r->d_attn_out);
    if (r->d_ffn_buf) cuMemFree(r->d_ffn_buf);
    if (r->d_ln_buf) cuMemFree(r->d_ln_buf);
    if (r->d_merge_buf) cuMemFree(r->d_merge_buf);
    if (r->d_mm_buf) cuMemFree(r->d_mm_buf);
    if (r->d_mm_out) cuMemFree(r->d_mm_out);
    if (r->d_rgb) cuMemFree(r->d_rgb);
    if (r->d_rope_cos) cuMemFree(r->d_rope_cos);
    if (r->d_rope_sin) cuMemFree(r->d_rope_sin);
    if (r->d_pos_map) cuMemFree(r->d_pos_map);
    if (r->d_pos_interp) cuMemFree(r->d_pos_interp);
    if (r->d_ds_feats) cuMemFree(r->d_ds_feats);

    free(r->h_pos_embd);
    free(r->h_output);

    /* Destroy CUDA objects */
    if (r->module) cuModuleUnload(r->module);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->context) cuCtxDestroy(r->context);

    free(r);
}
