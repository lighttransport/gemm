/* S3 dense FP8/E8M0 HIPRTC bring-up runner. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../../common/ds4f.h"
#include "hip_ds4f_dense.h"
#include "hip_ds4f_kernels.h"
#include "../../rdna4/hip_kernels_common.h"
#include "../../rdna4/rocew.h"

#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../../rdna4/hip_runner_common.h"

#include <math.h>
#include <limits.h>
#include <time.h>

/* Decode issues independent projection groups (qkv is three tensors); one
 * slot per member lets a whole group be in flight instead of paying a
 * launch + event-sync + blocking download per tensor. */
enum { HIP_DS4F_ASYNC_MAX = 8, HIP_DS4F_GEMM_MAX = 256 };

typedef struct {
    void *dw, *ds;
    const void *hw, *hs;
    int rows, cols, scale_cols, kind, owner;
} hip_ds4f_matrix;

typedef struct {
    const ds4f_layer *layer;
    void *dw, *ds;
    int raw;
} hip_ds4f_resident_layer;

typedef struct {
    void *w, *s, *x, *y;
    int n_out, n_in, n_tok, x_stride, y_stride, x_off, y_off;
} hip_ds4f_mxfp4_task;

enum {
    HIP_DS4F_MATRIX_FP8 = 0,
    HIP_DS4F_MATRIX_BF16 = 1,
    HIP_DS4F_MATRIX_FP8_BF16 = 2,
    HIP_DS4F_MATRIX_FP8_FP16 = 3,
    HIP_DS4F_MATRIX_FP8_ORDERED = 4,
    HIP_DS4F_MATRIX_MXFP4 = 5,
    HIP_DS4F_MATRIX_FP8_ROWSCALE = 6
};

static int matrix_is_bf16(int kind) {
    return kind == HIP_DS4F_MATRIX_BF16 || kind == HIP_DS4F_MATRIX_FP8_BF16;
}

static int matrix_is_fp16(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8_FP16;
}

static int matrix_is_fp8(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8 || kind == HIP_DS4F_MATRIX_FP8_ORDERED ||
           kind == HIP_DS4F_MATRIX_FP8_ROWSCALE;
}

static int matrix_is_mxfp4(int kind) {
    return kind == HIP_DS4F_MATRIX_MXFP4;
}

static int matrix_is_fp8_ordered(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8_ORDERED;
}

static int matrix_is_fp8_rowscale(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8_ROWSCALE;
}

static int matrix_is_fp8_promoted(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8_BF16 ||
           kind == HIP_DS4F_MATRIX_FP8_FP16;
}

struct hip_ds4f_dense {
    ds4f_mem_pool *mem;
    int device_id;
    int verbose;
    hipModule_t module;
    hipModule_t swiglu_module;
    hipFunction_t matvec;
    hipFunction_t bf16_matvec;
    hipFunction_t f16_matvec;
    hipFunction_t blockdiag_matvec;
    hipFunction_t gemm_fp8;
    hipFunction_t gemm_fp8_wmma;
    hipFunction_t gemm_fp8_wmma64;
    hipFunction_t gemm_fp8_ordered;
    hipFunction_t gemm_mxfp4;
    hipFunction_t gemm_mxfp4_grouped;
    hipFunction_t gemm_mxfp4_grouped_wmma;
    hipFunction_t gemm_mxfp4_grouped_wmma32;
    hipFunction_t gemm_mxfp4_grouped_wmma64;
    hipFunction_t mxfp4_matvec;
    hipFunction_t mxfp4_grouped_matvec;
    hipFunction_t gemm_fp8_rowscale;
    hipFunction_t gemm_bf16;
    hipFunction_t gemm_bf16_wmma;
    hipFunction_t gemm_f16;
    hipFunction_t swiglu;
    hipFunction_t gather_group, scatter_group;
    void *ffn_dx, *ffn_dg, *ffn_du, *ffn_dy;
    size_t ffn_dx_b, ffn_dg_b, ffn_du_b, ffn_dy_b;
    void *op_dx,*op_xt,*op_yt,*op_di,*op_dy;
    size_t op_dx_b,op_xt_b,op_yt_b,op_di_b,op_dy_b;
    hipFunction_t prefill_attn;
    hipFunction_t prefill_attn_wmma;
    hipFunction_t apply_rope, qnorm_rope_heads;
    hipFunction_t rmsnorm_bf16;
    hipFunction_t gemm_fp8_grouped_wmma;
    hipFunction_t gemm_fp8_grouped_wmma64;

    hip_ds4f_matrix *matrices;
    int n_matrices, cap_matrices;
    int current;
    void *dx, *dy;
    size_t x_bytes, y_bytes;

    hipStream_t stream;
    int qkv_fuse, qkv_device_chain, attn_device_chain, attn_no_d2h;
    int fp8_wmma_mode, bf16_wmma, attn_wmma, oproj_group_wmma, mxfp4_wmma;
    int decode_kv_resident;
    const uint16_t *decode_kv_host;
    int decode_kv_slots, decode_kv_dim;
    hipEvent_t done;
    int block_threads;
    int pending;
    int pending_id;
    int pending_rows;

    hipStream_t multi_stream[HIP_DS4F_ASYNC_MAX];
    hipEvent_t multi_done[HIP_DS4F_ASYNC_MAX];
    void *multi_dx[HIP_DS4F_ASYNC_MAX], *multi_dy[HIP_DS4F_ASYNC_MAX];
    size_t multi_x_bytes[HIP_DS4F_ASYNC_MAX], multi_y_bytes[HIP_DS4F_ASYNC_MAX];
    size_t multi_out_bytes[HIP_DS4F_ASYNC_MAX];
    float *multi_y[HIP_DS4F_ASYNC_MAX];
    int multi_pending, multi_n;

    /* Pinned bounce buffers.  Every host transfer here was pageable, which the
     * driver stages through its own bounce buffer anyway -- doing it explicitly
     * lets the copies run as real async DMA on the stream.  NULL whenever the
     * loaded driver lacks hipHostMalloc, in which case the pageable path below
     * is used unchanged. */
    void *pin_hx, *pin_hy;
    size_t pin_hx_cap, pin_hy_cap;
    void *multi_hx[HIP_DS4F_ASYNC_MAX], *multi_hy[HIP_DS4F_ASYNC_MAX];
    size_t multi_hx_cap[HIP_DS4F_ASYNC_MAX], multi_hy_cap[HIP_DS4F_ASYNC_MAX];

    void *gemm_dx, *gemm_dy;
    void *gemm_dy_tile;
    size_t gemm_dy_tile_bytes;
    float *gemm_yh_tile;
    size_t gemm_yh_tile_bytes;
    void *gemm_mxfp4_tasks;
    size_t gemm_mxfp4_tasks_bytes;
    void *fp8_lut;
    size_t gemm_x_bytes, gemm_y_bytes;
    float *gemm_x_pack, *gemm_y_pack;
    size_t gemm_x_pack_bytes, gemm_y_pack_bytes;
    void *attn_q, *attn_kv, *attn_sink, *attn_y;
    size_t attn_q_bytes, attn_kv_bytes, attn_sink_bytes, attn_y_bytes;
    int attn_device_ready, attn_device_M;
    const float *attn_device_host;
    void *rope_c, *rope_s;
    size_t rope_c_bytes, rope_s_bytes;
    const float *rope_host_c, *rope_host_s;
    int rope_host_pos, rope_host_m, rope_host_pairs;
    void *qkv_x, *qkv_lat, *qkv_norm, *qkv_q, *qkv_kv, *qkv_norm_w;
    size_t qkv_x_b, qkv_lat_b, qkv_norm_b, qkv_q_b, qkv_kv_b, qkv_norm_w_b;
    int qkv_device_ready, qkv_device_M;
    const float *qkv_device_host;
    void *gemm_multi_dy[HIP_DS4F_GEMM_MAX];
    size_t gemm_multi_y_bytes[HIP_DS4F_GEMM_MAX];
    const ds4f_layer *stream_layer;
    void *stream_dw, *stream_ds;
    hipStream_t stream_copy;
    hipEvent_t stream_copy_done[2];
    void *stream_slot_dw[2], *stream_slot_ds[2];
    size_t stream_slot_dw_bytes[2], stream_slot_ds_bytes[2];
    void *stream_slot_hreg[2];
    size_t stream_slot_hreg_bytes[2];
    void *stream_slot_hw[2], *stream_slot_hs[2];
    size_t stream_slot_hw_bytes[2], stream_slot_hs_bytes[2];
    int expert_pinned_staging;
    const ds4f_layer *stream_slot_layer[2];
    int stream_active_slot, stream_pending_slot;
    const ds4f_layer *stream_pending_layer;
    pthread_t stream_copy_thread;
    pthread_mutex_t stream_copy_mu;
    pthread_cond_t stream_copy_cv;
    pthread_cond_t stream_copy_ready_cv;
    int stream_copy_stop, stream_copy_request, stream_copy_submitted;
    int stream_copy_result, stream_copy_request_slot;
    double prof_routed_seconds;
    unsigned long prof_routed_calls;
    unsigned long prof_host_registers;
    size_t prof_host_register_bytes;
    hip_ds4f_resident_layer *resident_layers;
    int n_resident_layers, cap_resident_layers;
};

void hip_ds4f_dense_set_prefill_features(hip_ds4f_dense *ctx, int qkv_fuse,
    int qkv_chain, int attn_chain, int no_d2h, int fp8_wmma, int bf16_wmma,
    int attn_wmma, int oproj_group_wmma, int mxfp4_wmma, int block_threads) {
    if (!ctx) return;
    ctx->qkv_fuse = qkv_fuse != 0;
    ctx->qkv_device_chain = qkv_chain != 0;
    ctx->attn_device_chain = attn_chain != 0;
    ctx->attn_no_d2h = no_d2h != 0;
    ctx->fp8_wmma_mode = fp8_wmma;
    ctx->bf16_wmma = bf16_wmma != 0;
    ctx->attn_wmma = attn_wmma != 0;
    ctx->oproj_group_wmma = oproj_group_wmma;
    ctx->mxfp4_wmma = mxfp4_wmma;
    if (block_threads == 64 || block_threads == 128 || block_threads == 256)
        ctx->block_threads = block_threads;
}

void hip_ds4f_dense_set_expert_pinned_staging(hip_ds4f_dense *ctx, int enabled) {
    if (ctx) ctx->expert_pinned_staging = enabled != 0;
}

void hip_ds4f_dense_set_decode_features(hip_ds4f_dense *ctx, int kv_resident) {
    if (!ctx) return;
    ctx->decode_kv_resident = kv_resident != 0;
    if (!ctx->decode_kv_resident) {
        ctx->decode_kv_host = NULL;
        ctx->decode_kv_slots = ctx->decode_kv_dim = 0;
    }
}

void hip_ds4f_dense_invalidate_decode_kv(hip_ds4f_dense *ctx) {
    if (!ctx) return;
    ctx->decode_kv_host = NULL;
    ctx->decode_kv_slots = 0;
    ctx->decode_kv_dim = 0;
    ctx->qkv_device_ready = 0;
}

static int valid_dims(int rows, int cols) {
    return rows > 0 && cols > 0 && rows <= INT_MAX / cols;
}

static void *stream_copy_worker(void *opaque) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    for (;;) {
        pthread_mutex_lock(&ctx->stream_copy_mu);
        while (!ctx->stream_copy_request && !ctx->stream_copy_stop)
            pthread_cond_wait(&ctx->stream_copy_cv, &ctx->stream_copy_mu);
        if (ctx->stream_copy_stop) {
            pthread_mutex_unlock(&ctx->stream_copy_mu);
            break;
        }
        int slot = ctx->stream_copy_request_slot;
        const ds4f_layer *layer = ctx->stream_slot_layer[slot];
        ctx->stream_copy_request = 0;
        pthread_mutex_unlock(&ctx->stream_copy_mu);

        int ok = hipSetDevice(ctx->device_id) == hipSuccess;
        const ds4f_tensor *ex[] = { layer ? layer->ex_w1 : NULL,
                                    layer ? layer->ex_w2 : NULL,
                                    layer ? layer->ex_w3 : NULL };
        size_t wo = 0, so = 0;
        int pinned_stage = ctx->stream_slot_hw[slot] && ctx->stream_slot_hs[slot];
        for (size_t wi = 0; ok && wi < sizeof(ex) / sizeof(ex[0]); ++wi)
            if (ex[wi]) for (int e = 0; e < layer->n_owned; ++e) {
                const ds4f_tensor *t = &ex[wi][e];
                size_t wb = (size_t)t->rows * (size_t)(t->cols / 2);
                size_t sb = (size_t)t->rows * (size_t)(t->cols / 32);
                void *dw = (uint8_t *)ctx->stream_slot_dw[slot] + wo;
                void *ds = (uint8_t *)ctx->stream_slot_ds[slot] + so;
                if (pinned_stage) {
                    memcpy((uint8_t *)ctx->stream_slot_hw[slot] + wo, t->w, wb);
                    memcpy((uint8_t *)ctx->stream_slot_hs[slot] + so, t->scale, sb);
                } else if (hipMemcpyAsync(dw, t->w, wb, hipMemcpyHostToDevice,
                                          ctx->stream_copy) != hipSuccess ||
                           hipMemcpyAsync(ds, t->scale, sb, hipMemcpyHostToDevice,
                                          ctx->stream_copy) != hipSuccess) ok = 0;
                wo += wb; so += sb;
            }
        if (ok && pinned_stage &&
            (hipMemcpyAsync(ctx->stream_slot_dw[slot], ctx->stream_slot_hw[slot],
                            wo, hipMemcpyHostToDevice, ctx->stream_copy) != hipSuccess ||
             hipMemcpyAsync(ctx->stream_slot_ds[slot], ctx->stream_slot_hs[slot],
                            so, hipMemcpyHostToDevice, ctx->stream_copy) != hipSuccess))
            ok = 0;
        if (ok && hipEventRecord(ctx->stream_copy_done[slot], ctx->stream_copy) != hipSuccess)
            ok = 0;
        pthread_mutex_lock(&ctx->stream_copy_mu);
        ctx->stream_copy_result = ok ? 0 : -1;
        ctx->stream_copy_submitted = 1;
        pthread_cond_broadcast(&ctx->stream_copy_ready_cv);
        pthread_mutex_unlock(&ctx->stream_copy_mu);
    }
    return NULL;
}

static void clear_matrices(hip_ds4f_dense *ctx) {
    for (int i = 0; i < ctx->n_matrices; ++i) {
        if (ctx->matrices[i].dw && !ctx->matrices[i].owner) hipFree(ctx->matrices[i].dw);
        if (ctx->matrices[i].ds && !ctx->matrices[i].owner) hipFree(ctx->matrices[i].ds);
    }
    ctx->n_matrices = 0;
    ctx->current = -1;
    ctx->stream_layer = NULL;
    if (ctx->stream_dw) hipFree(ctx->stream_dw);
    if (ctx->stream_ds) hipFree(ctx->stream_ds);
    ctx->stream_dw = ctx->stream_ds = NULL;
    for (int i = 0; i < ctx->n_resident_layers; ++i) {
        if (ctx->resident_layers[i].dw) hipFree(ctx->resident_layers[i].dw);
        if (ctx->resident_layers[i].ds) hipFree(ctx->resident_layers[i].ds);
    }
    ctx->resident_layers = NULL;
    ctx->n_resident_layers = ctx->cap_resident_layers = 0;
}

static void release_matrix(hip_ds4f_dense *ctx, int id) {
    if (!ctx || id < 0 || id >= ctx->n_matrices) return;
    if (ctx->matrices[id].dw && !ctx->matrices[id].owner) hipFree(ctx->matrices[id].dw);
    if (ctx->matrices[id].ds && !ctx->matrices[id].owner) hipFree(ctx->matrices[id].ds);
    memset(&ctx->matrices[id], 0, sizeof(ctx->matrices[id]));
    ctx->matrices[id].kind = -1;
}

static int append_device_matrix(hip_ds4f_dense *ctx, void *dw, void *ds,
                                const void *hw, const void *hs,
                                int rows, int cols, int scale_cols,
                                int kind, int owner) {
    int id = -1;
    for (int i = 0; i < ctx->n_matrices; ++i)
        if (!ctx->matrices[i].dw && !ctx->matrices[i].ds) { id = i; break; }
    if (id < 0 && ctx->n_matrices == ctx->cap_matrices) {
        int cap = ctx->cap_matrices ? ctx->cap_matrices * 2 : 8;
        if (cap < ctx->n_matrices || cap > INT_MAX / (int)sizeof(*ctx->matrices)) return -1;
        hip_ds4f_matrix *p = (hip_ds4f_matrix *)ds4f_mem_realloc(
            ctx->mem, ctx->matrices,
            (size_t)ctx->cap_matrices * sizeof(*ctx->matrices),
            (size_t)cap * sizeof(*ctx->matrices), 64);
        if (!p) return -1;
        ctx->matrices = p; ctx->cap_matrices = cap;
    }
    if (id < 0) id = ctx->n_matrices++;
    ctx->matrices[id] = (hip_ds4f_matrix){ dw, ds, hw, hs, rows, cols, scale_cols, kind, owner };
    ctx->current = id;
    return id;
}

static void release_transient_experts(hip_ds4f_dense *ctx,
                                      const ds4f_tensor *const *w1,
                                      const ds4f_tensor *const *w3,
                                      const ds4f_tensor *const *w2,
                                      int n, int transient[3][HIP_DS4F_GEMM_MAX]) {
    const ds4f_tensor *const *w[3] = { w1, w3, w2 };
    for (int j = 0; j < 3; ++j) for (int s = 0; s < n; ++s)
        if (transient[j][s] >= 0) {
            release_matrix(ctx, transient[j][s]);
            ((ds4f_tensor *)w[j][s])->gpu_id = -1;
        }
}

hip_ds4f_dense *hip_ds4f_dense_create_ex(int device_id, int verbose, int precise_math) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_ds4f_dense: failed to initialize HIP/hipRTC\n");
        return NULL;
    }
    if (hipSetDevice(device_id) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: cannot select device %d\n", device_id);
        return NULL;
    }

    ds4f_mem_pool *mem = ds4f_mem_pool_create();
    if (!mem) return NULL;
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)ds4f_mem_calloc(mem, 1, sizeof(*ctx), 64);
    if (!ctx) {
        ds4f_mem_pool_destroy(mem);
        return NULL;
    }
    ctx->mem = mem;
    ctx->device_id = device_id;
    ctx->verbose = verbose;
    ctx->current = -1;
    ctx->stream_active_slot = -1;
    ctx->stream_pending_slot = -1;
    /* One row is one block. 128 threads is the best measured gfx1201 point:
     * it halves the wave reduction/launch footprint versus the original 256,
     * while retaining enough lanes for the 1K--8K column projections. */
    ctx->block_threads = 128;
    size_t common_len = strlen(hip_kernels_common_src);
    size_t dense_len = strlen(hip_ds4f_dense_kernels_src);
    char *full_src = (char *)ds4f_mem_alloc(ctx->mem, common_len + dense_len + 2, 64, 0);
    if (!full_src) {
        ds4f_mem_pool_destroy(ctx->mem);
        return NULL;
    }
    memcpy(full_src, hip_kernels_common_src, common_len);
    memcpy(full_src + common_len, hip_ds4f_dense_kernels_src, dense_len);
    full_src[common_len + dense_len] = '}';
    full_src[common_len + dense_len + 1] = '\0';
    if (hip_compile_kernels_ex(&ctx->module, device_id, full_src,
                            "hip_ds4f_dense.hip", verbose,
                            "hip_ds4f_dense", precise_math) < 0 ||
        hipModuleGetFunction(&ctx->matvec, ctx->module,
                             "ds4f_dense_fp8_matvec") != hipSuccess ||
        hipModuleGetFunction(&ctx->bf16_matvec, ctx->module,
                             "ds4f_dense_bf16_matvec") != hipSuccess ||
        hipModuleGetFunction(&ctx->f16_matvec, ctx->module,
                             "ds4f_dense_f16_matvec") != hipSuccess ||
        hipModuleGetFunction(&ctx->blockdiag_matvec, ctx->module,
                             "ds4f_dense_fp8_blockdiag") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_fp8, ctx->module,
                             "ds4f_dense_fp8_gemm") != hipSuccess ||
    hipModuleGetFunction(&ctx->gemm_fp8_wmma, ctx->module,
                         "ds4f_dense_fp8_wmma_f16_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_fp8_wmma64, ctx->module,
                             "ds4f_dense_fp8_wmma_f16_gemm64") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_fp8_ordered, ctx->module,
                             "ds4f_dense_fp8_ordered_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_mxfp4, ctx->module,
                             "ds4f_dense_mxfp4_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_mxfp4_grouped, ctx->module,
                             "ds4f_dense_mxfp4_grouped_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->mxfp4_matvec, ctx->module,
                             "ds4f_dense_mxfp4_matvec") != hipSuccess ||
        hipModuleGetFunction(&ctx->mxfp4_grouped_matvec, ctx->module,
                             "ds4f_dense_mxfp4_grouped_matvec") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_bf16, ctx->module,
                             "ds4f_dense_bf16_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_bf16_wmma, ctx->module,
                             "ds4f_dense_bf16_wmma_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_f16, ctx->module,
                             "gemm_tiled_f16_f32") != hipSuccess ||
        hipModuleGetFunction(&ctx->prefill_attn, ctx->module,
                             "ds4f_dense_prefill_attn") != hipSuccess ||
        hipModuleGetFunction(&ctx->gather_group,ctx->module,
                             "ds4f_gather_group") != hipSuccess ||
        hipModuleGetFunction(&ctx->scatter_group,ctx->module,
                             "ds4f_scatter_group") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_fp8_rowscale, ctx->module,
                             "ds4f_dense_fp8_rowscale_gemm") != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: failed to build matvec module\n");
        if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
        ds4f_mem_pool_destroy(ctx->mem);
        return NULL;
    }
    hipModuleGetFunction(&ctx->gemm_mxfp4_grouped_wmma, ctx->module,
                         "ds4f_dense_mxfp4_grouped_wmma");
    hipModuleGetFunction(&ctx->gemm_mxfp4_grouped_wmma32, ctx->module,
                         "ds4f_dense_mxfp4_grouped_wmma32");
    hipModuleGetFunction(&ctx->gemm_mxfp4_grouped_wmma64, ctx->module,
                         "ds4f_dense_mxfp4_grouped_wmma64");
    hipModuleGetFunction(&ctx->prefill_attn_wmma, ctx->module,
                         "ds4f_dense_prefill_attn_wmma");
    hipModuleGetFunction(&ctx->apply_rope, ctx->module, "ds4f_apply_rope");
    hipModuleGetFunction(&ctx->qnorm_rope_heads, ctx->module, "ds4f_qnorm_rope_heads");
    hipModuleGetFunction(&ctx->rmsnorm_bf16, ctx->module, "ds4f_rmsnorm_bf16");
    hipModuleGetFunction(&ctx->gemm_fp8_grouped_wmma, ctx->module, "ds4f_dense_fp8_grouped_wmma");
    hipModuleGetFunction(&ctx->gemm_fp8_grouped_wmma64, ctx->module, "ds4f_dense_fp8_grouped_wmma64");
    /* The fused shared-FFN SwiGLU lives in its own module, ALWAYS compiled
     * precise: clang's -ffast-math approximates the SiLU's division (even
     * through __fdiv_rn / __frcp_rn / double div), so it cannot ride in the
     * fast-compiled dense module without changing its bit-exactness.  A
     * separate compile keeps the dense module's math mode -- and therefore
     * its GEMM results -- exactly as configured. */
    if (hip_compile_kernels_ex(&ctx->swiglu_module, device_id,
                               hip_ds4f_swiglu_kernels_src,
                               "hip_ds4f_swiglu.hip", verbose,
                               "hip_ds4f_swiglu", 1) < 0 ||
        hipModuleGetFunction(&ctx->swiglu, ctx->swiglu_module,
                             "ds4f_swiglu_inplace") != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: failed to build precise swiglu module\n");
        if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
        ds4f_mem_pool_destroy(ctx->mem);
        return NULL;
    }
    {
        uint32_t lut[256];
        ds4f_init_fp8_e4m3fn_lut(lut);
        if (hipMalloc(&ctx->fp8_lut, sizeof(lut)) != hipSuccess ||
            hipMemcpy(ctx->fp8_lut, lut, sizeof(lut), hipMemcpyHostToDevice) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: FP8 LUT upload failed\n");
            if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
            if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
            ds4f_mem_pool_destroy(ctx->mem);
            return NULL;
        }
    }
    if (hipStreamCreate(&ctx->stream) != hipSuccess ||
        hipEventCreate(&ctx->done) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: failed to create stream/event\n");
        if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
        if (ctx->stream && hipStreamDestroy) hipStreamDestroy(ctx->stream);
        if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
        ds4f_mem_pool_destroy(ctx->mem);
        return NULL;
    }
    for (int i = 0; i < HIP_DS4F_ASYNC_MAX; ++i) {
        if (hipStreamCreate(&ctx->multi_stream[i]) != hipSuccess ||
            hipEventCreate(&ctx->multi_done[i]) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: failed to create async streams/events\n");
            for (int j = 0; j <= i; ++j) {
                if (ctx->multi_done[j] && hipEventDestroy) hipEventDestroy(ctx->multi_done[j]);
                if (ctx->multi_stream[j] && hipStreamDestroy) hipStreamDestroy(ctx->multi_stream[j]);
            }
            if (ctx->done && hipEventDestroy) hipEventDestroy(ctx->done);
            if (ctx->stream && hipStreamDestroy) hipStreamDestroy(ctx->stream);
            if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
            if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
            ds4f_mem_pool_destroy(ctx->mem);
            return NULL;
        }
    }
    if (hipStreamCreateWithFlags(&ctx->stream_copy, hipStreamNonBlocking) != hipSuccess ||
        hipEventCreate(&ctx->stream_copy_done[0]) != hipSuccess ||
        hipEventCreate(&ctx->stream_copy_done[1]) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: failed to create expert prefetch stream/events\n");
        if (ctx->stream_copy_done[0] && hipEventDestroy) hipEventDestroy(ctx->stream_copy_done[0]);
        if (ctx->stream_copy_done[1] && hipEventDestroy) hipEventDestroy(ctx->stream_copy_done[1]);
        if (ctx->stream_copy && hipStreamDestroy) hipStreamDestroy(ctx->stream_copy);
        for (int j = 0; j < HIP_DS4F_ASYNC_MAX; ++j) {
            if (ctx->multi_done[j] && hipEventDestroy) hipEventDestroy(ctx->multi_done[j]);
            if (ctx->multi_stream[j] && hipStreamDestroy) hipStreamDestroy(ctx->multi_stream[j]);
        }
        if (ctx->done && hipEventDestroy) hipEventDestroy(ctx->done);
        if (ctx->stream && hipStreamDestroy) hipStreamDestroy(ctx->stream);
        if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
        if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
        ds4f_mem_pool_destroy(ctx->mem);
        return NULL;
    }
    pthread_mutex_init(&ctx->stream_copy_mu, NULL);
    pthread_cond_init(&ctx->stream_copy_cv, NULL);
    pthread_cond_init(&ctx->stream_copy_ready_cv, NULL);
    if (pthread_create(&ctx->stream_copy_thread, NULL, stream_copy_worker, ctx) != 0) {
        pthread_cond_destroy(&ctx->stream_copy_ready_cv);
        pthread_cond_destroy(&ctx->stream_copy_cv);
        pthread_mutex_destroy(&ctx->stream_copy_mu);
        for (int i = 0; i < 2; ++i) {
            if (ctx->stream_copy_done[i] && hipEventDestroy) hipEventDestroy(ctx->stream_copy_done[i]);
        }
        if (ctx->stream_copy && hipStreamDestroy) hipStreamDestroy(ctx->stream_copy);
        for (int j = 0; j < HIP_DS4F_ASYNC_MAX; ++j) {
            if (ctx->multi_done[j] && hipEventDestroy) hipEventDestroy(ctx->multi_done[j]);
            if (ctx->multi_stream[j] && hipStreamDestroy) hipStreamDestroy(ctx->multi_stream[j]);
        }
        if (ctx->done && hipEventDestroy) hipEventDestroy(ctx->done);
        if (ctx->stream && hipStreamDestroy) hipStreamDestroy(ctx->stream);
        if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
        if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
        ds4f_mem_pool_destroy(ctx->mem);
        return NULL;
    }
    return ctx;
}

hip_ds4f_dense *hip_ds4f_dense_create(int device_id, int verbose) {
    return hip_ds4f_dense_create_ex(device_id, verbose, -1);
}

void hip_ds4f_dense_destroy(hip_ds4f_dense *ctx) {
    if (!ctx) return;
    { const char *p = getenv("DS4F_PROF");
      if (p && atoi(p) && ctx->prof_routed_calls)
          fprintf(stderr, "  hip_route  %9.3f s  calls=%lu avg=%.3f ms\n",
                  ctx->prof_routed_seconds, ctx->prof_routed_calls,
                  1000.0 * ctx->prof_routed_seconds / ctx->prof_routed_calls); }
    { const char *p = getenv("DS4F_PROF");
      if (p && atoi(p))
          fprintf(stderr, "  hip_hreg             calls=%lu total=%.1f GiB\n",
                  ctx->prof_host_registers,
                  (double)ctx->prof_host_register_bytes / (1024.0*1024.0*1024.0)); }
    pthread_mutex_lock(&ctx->stream_copy_mu);
    ctx->stream_copy_stop = 1;
    pthread_cond_signal(&ctx->stream_copy_cv);
    pthread_mutex_unlock(&ctx->stream_copy_mu);
    pthread_join(ctx->stream_copy_thread, NULL);
    pthread_cond_destroy(&ctx->stream_copy_ready_cv);
    pthread_cond_destroy(&ctx->stream_copy_cv);
    pthread_mutex_destroy(&ctx->stream_copy_mu);
    if (ctx->pending && ctx->done) hipEventSynchronize(ctx->done);
    if (ctx->multi_pending) {
        for (int i = 0; i < ctx->multi_n; ++i)
            if (ctx->multi_done[i]) hipEventSynchronize(ctx->multi_done[i]);
    }
    clear_matrices(ctx);
    if (ctx->dx) hipFree(ctx->dx);
    if (ctx->dy) hipFree(ctx->dy);
    if (ctx->gemm_dx) hipFree(ctx->gemm_dx);
    if (ctx->gemm_dy) hipFree(ctx->gemm_dy);
    if (ctx->gemm_dy_tile) hipFree(ctx->gemm_dy_tile);
    if (ctx->gemm_yh_tile) hipHostFree(ctx->gemm_yh_tile);
    if (ctx->attn_q) hipFree(ctx->attn_q);
    if (ctx->attn_kv) hipFree(ctx->attn_kv);
    if (ctx->attn_sink) hipFree(ctx->attn_sink);
    if (ctx->attn_y) hipFree(ctx->attn_y);
    if (ctx->rope_c) hipFree(ctx->rope_c);
    if (ctx->rope_s) hipFree(ctx->rope_s);
    if (ctx->qkv_x) hipFree(ctx->qkv_x);
    if (ctx->qkv_lat) hipFree(ctx->qkv_lat);
    if (ctx->qkv_norm) hipFree(ctx->qkv_norm);
    if (ctx->qkv_q) hipFree(ctx->qkv_q);
    if (ctx->qkv_kv) hipFree(ctx->qkv_kv);
    if (ctx->qkv_norm_w) hipFree(ctx->qkv_norm_w);
    if (ctx->op_dy) hipFree(ctx->op_dy);
    if (ctx->op_di) hipFree(ctx->op_di);
    if (ctx->op_yt) hipFree(ctx->op_yt);
    if (ctx->op_xt) hipFree(ctx->op_xt);
    if (ctx->op_dx) hipFree(ctx->op_dx);
    if (ctx->gemm_mxfp4_tasks) hipFree(ctx->gemm_mxfp4_tasks);
    if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
    for (int i = 0; i < HIP_DS4F_GEMM_MAX; ++i)
        if (ctx->gemm_multi_dy[i]) hipFree(ctx->gemm_multi_dy[i]);
    for (int i = 0; i < 2; ++i) {
        if (ctx->stream_slot_hreg[i] && hipHostUnregister)
            hipHostUnregister(ctx->stream_slot_hreg[i]);
        if (ctx->stream_slot_hw[i] && hipHostFree)
            hipHostFree(ctx->stream_slot_hw[i]);
        if (ctx->stream_slot_hs[i] && hipHostFree)
            hipHostFree(ctx->stream_slot_hs[i]);
        if (ctx->stream_slot_dw[i]) hipFree(ctx->stream_slot_dw[i]);
        if (ctx->stream_slot_ds[i]) hipFree(ctx->stream_slot_ds[i]);
        if (ctx->stream_copy_done[i] && hipEventDestroy) hipEventDestroy(ctx->stream_copy_done[i]);
    }
    if (ctx->stream_copy && hipStreamDestroy) hipStreamDestroy(ctx->stream_copy);
    for (int i = 0; i < HIP_DS4F_ASYNC_MAX; ++i) {
        if (ctx->multi_dx[i]) hipFree(ctx->multi_dx[i]);
        if (ctx->multi_dy[i]) hipFree(ctx->multi_dy[i]);
        if (ctx->multi_done[i] && hipEventDestroy) hipEventDestroy(ctx->multi_done[i]);
        if (ctx->multi_stream[i] && hipStreamDestroy) hipStreamDestroy(ctx->multi_stream[i]);
    }
    if (ctx->done && hipEventDestroy) hipEventDestroy(ctx->done);
    if (ctx->stream && hipStreamDestroy) hipStreamDestroy(ctx->stream);
    if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
    if (ctx->swiglu_module && hipModuleUnload) hipModuleUnload(ctx->swiglu_module);
    ds4f_mem_pool_destroy(ctx->mem);
}

/* Grow a pinned host staging buffer.  Returns -1 (and leaves *buf NULL) when
 * pinned memory is unavailable, so callers fall back to the pageable copy. */
static int ensure_pinned(void **buf, size_t *cap, size_t bytes) {
    if (*buf && *cap >= bytes) return 0;
    if (!hipHostMalloc) return -1;
    void *p = NULL;
    if (hipHostMalloc(&p, bytes, hipHostMallocDefault) != hipSuccess || !p)
        return -1;
    if (*buf) { if (hipHostFree) hipHostFree(*buf); else if (hipFreeHost) hipFreeHost(*buf); }
    *buf = p; *cap = bytes;
    return 0;
}

static int ensure_vectors(hip_ds4f_dense *ctx, int rows, int cols) {
    size_t x_bytes = (size_t)cols * sizeof(float);
    size_t y_bytes = (size_t)rows * sizeof(float);
    if (ctx->dx && ctx->dy && ctx->x_bytes >= x_bytes && ctx->y_bytes >= y_bytes)
        return 0;

    void *dx = NULL, *dy = NULL;
    if (hipMalloc(&dx, x_bytes) != hipSuccess ||
        hipMalloc(&dy, y_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: vector allocation failed\n");
        if (dx) hipFree(dx);
        if (dy) hipFree(dy);
        return -1;
    }
    if (ctx->dx) hipFree(ctx->dx);
    if (ctx->dy) hipFree(ctx->dy);
    ctx->dx = dx; ctx->dy = dy;
    ctx->x_bytes = x_bytes; ctx->y_bytes = y_bytes;
    return 0;
}

static int ensure_multi_vectors(hip_ds4f_dense *ctx, int slot, int rows, int cols) {
    size_t x_bytes = (size_t)cols * sizeof(float);
    size_t y_bytes = (size_t)rows * sizeof(float);
    if (ctx->multi_dx[slot] && ctx->multi_dy[slot] &&
        ctx->multi_x_bytes[slot] >= x_bytes && ctx->multi_y_bytes[slot] >= y_bytes)
        return 0;
    void *dx = NULL, *dy = NULL;
    if (hipMalloc(&dx, x_bytes) != hipSuccess ||
        hipMalloc(&dy, y_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: async vector allocation failed\n");
        if (dx) hipFree(dx);
        if (dy) hipFree(dy);
        return -1;
    }
    if (ctx->multi_dx[slot]) hipFree(ctx->multi_dx[slot]);
    if (ctx->multi_dy[slot]) hipFree(ctx->multi_dy[slot]);
    ctx->multi_dx[slot] = dx;
    ctx->multi_dy[slot] = dy;
    ctx->multi_x_bytes[slot] = x_bytes;
    ctx->multi_y_bytes[slot] = y_bytes;
    return 0;
}

static int hip_ds4f_dense_add_storage(hip_ds4f_dense *ctx,
                                      const uint8_t *w, const uint8_t *s,
                                      int rows, int cols, int scale_cols,
                                      size_t w_bytes, size_t s_bytes, int kind) {
    if (!ctx || !w || !s || !valid_dims(rows, cols)) {
        fprintf(stderr, "hip_ds4f_dense: invalid matrix arguments\n");
        return -1;
    }
    if (ctx->pending || ctx->multi_pending) {
        fprintf(stderr, "hip_ds4f_dense: cannot add a matrix with work in flight\n");
        return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;

    void *dw = NULL, *ds = NULL;
    if (hipMalloc(&dw, w_bytes) != hipSuccess ||
        hipMalloc(&ds, s_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: matrix allocation failed\n");
        if (dw) hipFree(dw);
        if (ds) hipFree(ds);
        return -1;
    }
    if (hipMemcpy(dw, w, w_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(ds, s, s_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: matrix upload failed\n");
        hipFree(dw); hipFree(ds);
        return -1;
    }
    int id = append_device_matrix(ctx, dw, ds, w, s, rows, cols,
                                  scale_cols, kind, 0);
    if (id < 0) { hipFree(dw); hipFree(ds); }
    return id;
}

/* Upload a transient matrix on the compute stream.  The synchronous helper
 * above is appropriate for permanent dense bindings, but routed serving binds
 * up to eighteen expert matrices per layer; blocking on every copy turns PCIe
 * latency into the dominant term. */
static int hip_ds4f_dense_add_storage_async(hip_ds4f_dense *ctx,
                                            const uint8_t *w, const uint8_t *s,
                                            int rows, int cols, int scale_cols,
                                            size_t w_bytes, size_t s_bytes, int kind) {
    if (!ctx || !w || !s || !valid_dims(rows, cols) || ctx->pending || ctx->multi_pending ||
        hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    void *dw = NULL, *ds = NULL;
    if (hipMalloc(&dw, w_bytes) != hipSuccess || hipMalloc(&ds, s_bytes) != hipSuccess) {
        if (dw) hipFree(dw); if (ds) hipFree(ds); return -1;
    }
    if (hipMemcpyAsync(dw, w, w_bytes, hipMemcpyHostToDevice, ctx->stream) != hipSuccess ||
        hipMemcpyAsync(ds, s, s_bytes, hipMemcpyHostToDevice, ctx->stream) != hipSuccess) {
        hipFree(dw); hipFree(ds); return -1;
    }
    int id = append_device_matrix(ctx, dw, ds, w, s, rows, cols, scale_cols, kind, 0);
    if (id < 0) { hipFree(dw); hipFree(ds); }
    return id;
}

int hip_ds4f_dense_add(hip_ds4f_dense *ctx,
                       const uint8_t *w, const uint8_t *s,
                       int rows, int cols) {
    int sc = (cols + 127) / 128;
    int sr = (rows + 127) / 128;
    return hip_ds4f_dense_add_storage(ctx, w, s, rows, cols, sc,
        (size_t)rows * (size_t)cols, (size_t)sr * (size_t)sc,
        HIP_DS4F_MATRIX_FP8);
}

int hip_ds4f_dense_bind_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!t) return -1;
    int id = hip_ds4f_dense_add(ctx, (const uint8_t *)t->w, t->scale,
                                t->rows, t->cols);
    if (id >= 0) t->gpu_id = id;
    return id;
}

int hip_ds4f_dense_bind_mxfp4_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_MXFP4 || !t->w || !t->scale ||
        !valid_dims(t->rows, t->cols) || (t->cols & 31)) return -1;
    int id = hip_ds4f_dense_add_storage(ctx, (const uint8_t *)t->w,
        t->scale, t->rows, t->cols, t->cols / 32,
        (size_t)t->rows * (size_t)(t->cols / 2),
        (size_t)t->rows * (size_t)(t->cols / 32), HIP_DS4F_MATRIX_MXFP4);
    if (id >= 0) t->gpu_id = id;
    return id;
}

static int hip_ds4f_dense_bind_mxfp4_tensor_async(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_MXFP4 || !t->w || !t->scale ||
        !valid_dims(t->rows, t->cols) || (t->cols & 31)) return -1;
    int id = hip_ds4f_dense_add_storage_async(ctx, (const uint8_t *)t->w,
        t->scale, t->rows, t->cols, t->cols / 32,
        (size_t)t->rows * (size_t)(t->cols / 2),
        (size_t)t->rows * (size_t)(t->cols / 32), HIP_DS4F_MATRIX_MXFP4);
    if (id >= 0) t->gpu_id = id;
    return id;
}

int hip_ds4f_dense_bind_mxfp4_widened_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    static const float lut[16] = { 0.f,1.f,2.f,3.f,4.f,6.f,8.f,12.f,
                                   0.f,-1.f,-2.f,-3.f,-4.f,-6.f,-8.f,-12.f };
    if (!ctx || !t || t->type != DS4F_MXFP4 || !t->w || !t->scale ||
        !valid_dims(t->rows, t->cols) || (t->cols & 127)) return -1;
    int sc = (t->cols + 127) / 128;
    size_t wb = (size_t)t->rows * (size_t)t->cols;
    size_t sb = (size_t)t->rows * (size_t)sc;
    uint8_t *fw = (uint8_t *)ds4f_mem_alloc(ctx->mem, wb, 64, 0);
    uint8_t *fs = (uint8_t *)ds4f_mem_alloc(ctx->mem, sb, 64, 0);
    if (!fw || !fs) return -1;
    const uint8_t *w = (const uint8_t *)t->w, *s = t->scale;
    int rb = t->cols / 2, rs = t->cols / 32;
    for (int r = 0; r < t->rows; ++r) {
        for (int bc = 0; bc < sc; ++bc) {
            /* Raw MXFP4 stores the E8M0 byte one below the exponent byte
             * consumed by the regular FP8 kernels. Keep the exponent in
             * that byte domain while normalizing the FP8 payload. */
            int emax = 0;
            for (int b = 0; b < 4; ++b) {
                int e = s[(size_t)r * rs + bc * 4 + b];
                int se = e ? e - 1 : 0;
                if (se > emax) emax = se;
            }
            fs[(size_t)r * sc + bc] = (uint8_t)emax;
            for (int j = 0; j < 128; ++j) {
                int col = bc * 128 + j, block = col >> 5, p = col & 31;
                int off = p >> 1;
                uint8_t q = w[(size_t)r * rb + (size_t)block * 16 + off];
                int nib = (p & 1) ? q >> 4 : q & 15;
                int se = s[(size_t)r * rs + block] ?
                         (int)s[(size_t)r * rs + block] - 1 : 0;
                float v = s[(size_t)r * rs + block] == 0
                    ? 0.f : lut[nib] * ldexpf(1.0f, se - emax);
                fw[(size_t)r * t->cols + col] = hip_f32_to_fp8_e4m3(v);
            }
        }
    }
    int id = hip_ds4f_dense_add_storage(ctx, fw, fs, t->rows, t->cols,
        sc, wb, sb, HIP_DS4F_MATRIX_FP8_ROWSCALE);
    if (id >= 0) t->gpu_id = id;
    return id;
}

static int widen_mxfp4_host(const ds4f_tensor *t, uint8_t *fw, uint8_t *fs) {
    static const float lut[16] = { 0.f,1.f,2.f,3.f,4.f,6.f,8.f,12.f,
                                   0.f,-1.f,-2.f,-3.f,-4.f,-6.f,-8.f,-12.f };
    int sc = (t->cols + 127) / 128, rb = t->cols / 2, rs = t->cols / 32;
    const uint8_t *w = (const uint8_t *)t->w, *s = t->scale;
    for (int r = 0; r < t->rows; ++r) for (int bc = 0; bc < sc; ++bc) {
        int emax = 0;
        for (int b = 0; b < 4; ++b) {
            int e = s[(size_t)r * rs + bc * 4 + b];
            if (e && e - 1 > emax) emax = e - 1;
        }
        fs[(size_t)r * sc + bc] = (uint8_t)emax;
        for (int j = 0; j < 128; ++j) {
            int col = bc * 128 + j, block = col >> 5, p = col & 31;
            uint8_t q = w[(size_t)r * rb + (size_t)block * 16 + (p >> 1)];
            int nib = (p & 1) ? q >> 4 : q & 15;
            int e = s[(size_t)r * rs + block];
            float v = e ? lut[nib] * ldexpf(1.0f, (e - 1) - emax) : 0.f;
            fw[(size_t)r * t->cols + col] = hip_f32_to_fp8_e4m3(v);
        }
    }
    return 0;
}

static int stream_layer_impl(void *opaque, const ds4f_layer *layer, int raw, int keep) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (!ctx || !layer || ctx->pending || ctx->multi_pending) return -1;
    for (int i = 0; i < ctx->n_resident_layers; ++i)
        if (ctx->resident_layers[i].layer == layer && ctx->resident_layers[i].raw == raw)
            return 0;
    if (ctx->stream_layer == layer) return 0;
    if (ctx->stream_layer) {
        const ds4f_layer *old = ctx->stream_layer;
        const ds4f_tensor *old_ex[] = { old->ex_w1, old->ex_w2, old->ex_w3 };
        for (size_t w = 0; w < sizeof(old_ex) / sizeof(old_ex[0]); ++w)
            if (old_ex[w]) for (int e = 0; e < old->n_owned; ++e) {
                ds4f_tensor *t = (ds4f_tensor *)&old_ex[w][e];
                if (t->gpu_id >= 0) { release_matrix(ctx, t->gpu_id); t->gpu_id = -1; }
            }
    }
    ctx->stream_layer = NULL;
    if (ctx->stream_dw) hipFree(ctx->stream_dw);
    if (ctx->stream_ds) hipFree(ctx->stream_ds);
    ctx->stream_dw = ctx->stream_ds = NULL;
    const ds4f_tensor *ex[] = { layer->ex_w1, layer->ex_w2, layer->ex_w3 };
    size_t wtotal = 0, stotal = 0;
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi) {
        if (!ex[wi]) return -1;
        for (int e = 0; e < layer->n_owned; ++e) {
            const ds4f_tensor *t = &ex[wi][e];
            if (t->type != DS4F_MXFP4 || !valid_dims(t->rows, t->cols) || (t->cols & 127)) return -1;
            wtotal += raw ? (size_t)t->rows * (size_t)(t->cols / 2)
                          : (size_t)t->rows * (size_t)t->cols;
            stotal += raw ? (size_t)t->rows * (size_t)(t->cols / 32)
                          : (size_t)t->rows * (size_t)((t->cols + 127) / 128);
        }
    }
    uint8_t *hw = (uint8_t *)ds4f_mem_alloc(ctx->mem, wtotal, 256, 0);
    uint8_t *hs = (uint8_t *)ds4f_mem_alloc(ctx->mem, stotal, 256, 0);
    if (!hw || !hs || hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    size_t wo = 0, so = 0;
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi)
        for (int e = 0; e < layer->n_owned; ++e) {
            ds4f_tensor *t = (ds4f_tensor *)&ex[wi][e];
            size_t wb = raw ? (size_t)t->rows * (size_t)(t->cols / 2)
                            : (size_t)t->rows * (size_t)t->cols;
            size_t sb = raw ? (size_t)t->rows * (size_t)(t->cols / 32)
                            : (size_t)t->rows * (size_t)((t->cols + 127) / 128);
            if (raw) {
                memcpy(hw + wo, t->w, wb);
                memcpy(hs + so, t->scale, sb);
            } else if (widen_mxfp4_host(t, hw + wo, hs + so) != 0) return -1;
            t->gpu_id = -1;
            wo += wb; so += sb;
        }
    if (hipMalloc(&ctx->stream_dw, wtotal) != hipSuccess ||
        hipMalloc(&ctx->stream_ds, stotal) != hipSuccess ||
        hipMemcpy(ctx->stream_dw, hw, wtotal, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(ctx->stream_ds, hs, stotal, hipMemcpyHostToDevice) != hipSuccess) {
        if (ctx->stream_dw) hipFree(ctx->stream_dw);
        if (ctx->stream_ds) hipFree(ctx->stream_ds);
        ctx->stream_dw = ctx->stream_ds = NULL;
        return -1;
    }
    wo = so = 0;
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi)
        for (int e = 0; e < layer->n_owned; ++e) {
            ds4f_tensor *t = (ds4f_tensor *)&ex[wi][e];
            size_t wb = raw ? (size_t)t->rows * (size_t)(t->cols / 2)
                            : (size_t)t->rows * (size_t)t->cols;
            size_t sb = raw ? (size_t)t->rows * (size_t)(t->cols / 32)
                            : (size_t)t->rows * (size_t)((t->cols + 127) / 128);
            int id = append_device_matrix(ctx, (uint8_t *)ctx->stream_dw + wo,
                (uint8_t *)ctx->stream_ds + so, hw + wo, hs + so,
                t->rows, t->cols, raw ? t->cols / 32 : (t->cols + 127) / 128,
                raw ? HIP_DS4F_MATRIX_MXFP4 : HIP_DS4F_MATRIX_FP8_ROWSCALE, 1);
            if (id < 0) return -1;
            t->gpu_id = id; wo += wb; so += sb;
        }
    if (keep) {
        if (ctx->n_resident_layers == ctx->cap_resident_layers) {
            int cap = ctx->cap_resident_layers ? ctx->cap_resident_layers * 2 : 8;
            hip_ds4f_resident_layer *p = (hip_ds4f_resident_layer *)ds4f_mem_realloc(
                ctx->mem, ctx->resident_layers,
                (size_t)ctx->cap_resident_layers * sizeof(*ctx->resident_layers),
                (size_t)cap * sizeof(*ctx->resident_layers), 64);
            if (!p) return -1;
            ctx->resident_layers = p;
            ctx->cap_resident_layers = cap;
        }
        ctx->resident_layers[ctx->n_resident_layers++] =
            (hip_ds4f_resident_layer){ layer, ctx->stream_dw, ctx->stream_ds, raw };
        ctx->stream_dw = ctx->stream_ds = NULL;
        ctx->stream_layer = NULL;
    } else {
        ctx->stream_layer = layer;
    }
    return 0;
}

static size_t mxfp4_layer_bytes(const ds4f_layer *layer, int raw) {
    if (!layer) return 0;
    const ds4f_tensor *ex[] = { layer->ex_w1, layer->ex_w2, layer->ex_w3 };
    size_t total = 0;
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi) {
        if (!ex[wi]) return 0;
        for (int e = 0; e < layer->n_owned; ++e) {
            const ds4f_tensor *t = &ex[wi][e];
            if (t->type != DS4F_MXFP4 || !valid_dims(t->rows, t->cols)) return 0;
            size_t wb = raw ? (size_t)t->rows * (size_t)(t->cols / 2)
                            : (size_t)t->rows * (size_t)t->cols;
            size_t sb = raw ? (size_t)t->rows * (size_t)(t->cols / 32)
                            : (size_t)t->rows * (size_t)((t->cols + 127) / 128);
            if (SIZE_MAX - total < wb + sb) return 0;
            total += wb + sb;
        }
    }
    return total;
}

int hip_ds4f_dense_recommend_mxfp4_resident_layers(
    void *opaque, const ds4f_layer *layers, int n_layers, int raw, int reserve_mb) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (!ctx || !layers || n_layers <= 0 || hipSetDevice(ctx->device_id) != hipSuccess)
        return 0;
    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 0;
    (void)total_bytes;
    size_t reserve = (size_t)(reserve_mb > 0 ? reserve_mb : 512) * 1024u * 1024u;
    size_t max_slot = 0;
    for (int i = 0; i < n_layers; ++i) {
        size_t bytes = mxfp4_layer_bytes(&layers[i], raw);
        if (!bytes) return 0;
        if (bytes > max_slot) max_slot = bytes;
    }
    /* Async prefill keeps two raw expert slots alive. Account for both before
     * admitting the resident prefix; this avoids discovering the limit only
     * when prefetch reaches the first non-resident layer. */
    size_t fixed = reserve;
    if (max_slot > (SIZE_MAX - fixed) / 2) return 0;
    fixed += 2 * max_slot;
    if (free_bytes <= fixed) return 0;
    size_t budget = free_bytes - fixed;
    size_t used = 0;
    int fit = 0;
    for (; fit < n_layers; ++fit) {
        size_t bytes = mxfp4_layer_bytes(&layers[fit], raw);
        if (bytes > budget - used) break;
        used += bytes;
    }
    return fit;
}

int hip_ds4f_dense_stream_layer(void *opaque, const ds4f_layer *layer) {
    return stream_layer_impl(opaque, layer, 0, 0);
}

int hip_ds4f_dense_stream_layer_raw(void *opaque, const ds4f_layer *layer) {
    return stream_layer_impl(opaque, layer, 1, 0);
}

static void release_stream_slot(hip_ds4f_dense *ctx, int slot) {
    const ds4f_layer *layer = ctx->stream_slot_layer[slot];
    if (!layer) return;
    const ds4f_tensor *ex[] = { layer->ex_w1, layer->ex_w2, layer->ex_w3 };
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi)
        if (ex[wi]) for (int e = 0; e < layer->n_owned; ++e) {
            ds4f_tensor *t = (ds4f_tensor *)&ex[wi][e];
            if (t->gpu_id >= 0) { release_matrix(ctx, t->gpu_id); t->gpu_id = -1; }
        }
    if (ctx->stream_slot_hreg[slot] && hipHostUnregister) {
        hipHostUnregister(ctx->stream_slot_hreg[slot]);
        ctx->stream_slot_hreg[slot] = NULL;
        ctx->stream_slot_hreg_bytes[slot] = 0;
    }
    ctx->stream_slot_layer[slot] = NULL;
}

static int ensure_stream_slot(hip_ds4f_dense *ctx, int slot,
                              size_t wbytes, size_t sbytes) {
    if (ctx->stream_slot_dw[slot] && ctx->stream_slot_dw_bytes[slot] >= wbytes &&
        ctx->stream_slot_ds[slot] && ctx->stream_slot_ds_bytes[slot] >= sbytes)
        return 0;
    void *dw = NULL, *ds = NULL;
    if (hipMalloc(&dw, wbytes) != hipSuccess || hipMalloc(&ds, sbytes) != hipSuccess) {
        if (dw) hipFree(dw);
        if (ds) hipFree(ds);
        return -1;
    }
    if (ctx->stream_slot_dw[slot]) hipFree(ctx->stream_slot_dw[slot]);
    if (ctx->stream_slot_ds[slot]) hipFree(ctx->stream_slot_ds[slot]);
    ctx->stream_slot_dw[slot] = dw; ctx->stream_slot_ds[slot] = ds;
    ctx->stream_slot_dw_bytes[slot] = wbytes;
    ctx->stream_slot_ds_bytes[slot] = sbytes;
    return 0;
}

static int ensure_stream_host_slot(hip_ds4f_dense *ctx, int slot,
                                   size_t wbytes, size_t sbytes) {
    if (!ctx->expert_pinned_staging || !hipHostMalloc || !hipHostFree) return -1;
    if (ctx->stream_slot_hw[slot] && ctx->stream_slot_hw_bytes[slot] >= wbytes &&
        ctx->stream_slot_hs[slot] && ctx->stream_slot_hs_bytes[slot] >= sbytes)
        return 0;
    void *hw = NULL, *hs = NULL;
    if (hipHostMalloc(&hw, wbytes, hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc(&hs, sbytes, hipHostMallocDefault) != hipSuccess) {
        if (hw) hipHostFree(hw);
        if (hs) hipHostFree(hs);
        return -1;
    }
    if (ctx->stream_slot_hw[slot]) hipHostFree(ctx->stream_slot_hw[slot]);
    if (ctx->stream_slot_hs[slot]) hipHostFree(ctx->stream_slot_hs[slot]);
    ctx->stream_slot_hw[slot] = hw; ctx->stream_slot_hs[slot] = hs;
    ctx->stream_slot_hw_bytes[slot] = wbytes;
    ctx->stream_slot_hs_bytes[slot] = sbytes;
    return 0;
}

static int prefetch_layer_raw_async(hip_ds4f_dense *ctx, const ds4f_layer *layer) {
    if (!ctx || !layer || ctx->pending || ctx->multi_pending || !ctx->stream_copy)
        return -1;
    /* The A/B/decode gate may have used the legacy synchronous streamer before
     * batched prefill attaches the asynchronous lifecycle. Reclaim that one
     * stale stream before reserving a double-buffer slot. */
    if (ctx->stream_layer) {
        const ds4f_layer *old = ctx->stream_layer;
        const ds4f_tensor *old_ex[] = { old->ex_w1, old->ex_w2, old->ex_w3 };
        for (size_t wi = 0; wi < sizeof(old_ex) / sizeof(old_ex[0]); ++wi)
            if (old_ex[wi]) for (int e = 0; e < old->n_owned; ++e) {
                ds4f_tensor *t = (ds4f_tensor *)&old_ex[wi][e];
                if (t->gpu_id >= 0) { release_matrix(ctx, t->gpu_id); t->gpu_id = -1; }
            }
        if (ctx->stream_dw) hipFree(ctx->stream_dw);
        if (ctx->stream_ds) hipFree(ctx->stream_ds);
        ctx->stream_dw = ctx->stream_ds = NULL;
        ctx->stream_layer = NULL;
    }
    for (int i = 0; i < ctx->n_resident_layers; ++i)
        if (ctx->resident_layers[i].layer == layer && ctx->resident_layers[i].raw)
            return 0;
    if (ctx->stream_pending_layer == layer ||
        (ctx->stream_active_slot >= 0 && ctx->stream_slot_layer[ctx->stream_active_slot] == layer))
        return 0;
    if (ctx->stream_pending_layer) return -1;

    const ds4f_tensor *ex[] = { layer->ex_w1, layer->ex_w2, layer->ex_w3 };
    size_t wtotal = 0, stotal = 0;
    uintptr_t host_lo = UINTPTR_MAX, host_hi = 0;
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi)
        if (ex[wi]) for (int e = 0; e < layer->n_owned; ++e) {
            const ds4f_tensor *t = &ex[wi][e];
            if (t->type != DS4F_MXFP4 || !valid_dims(t->rows, t->cols) ||
                (t->cols & 127)) return -1;
            wtotal += (size_t)t->rows * (size_t)(t->cols / 2);
            stotal += (size_t)t->rows * (size_t)(t->cols / 32);
            size_t wb = (size_t)t->rows * (size_t)(t->cols / 2);
            size_t sb = (size_t)t->rows * (size_t)(t->cols / 32);
            uintptr_t wp = (uintptr_t)t->w, sp = (uintptr_t)t->scale;
            if (wp < host_lo) host_lo = wp;
            if (sp < host_lo) host_lo = sp;
            if (wp + wb > host_hi) host_hi = wp + wb;
            if (sp + sb > host_hi) host_hi = sp + sb;
        }
    int slot = ctx->stream_active_slot < 0 ? 0 : 1 - ctx->stream_active_slot;
    if (ctx->stream_slot_layer[slot]) release_stream_slot(ctx, slot);
    if (ensure_stream_slot(ctx, slot, wtotal, stotal) != 0) return -1;
    int pinned_stage = ensure_stream_host_slot(ctx, slot, wtotal, stotal) == 0;
    /* The nocopy stage is one contiguous virtual mapping. Registering the
     * current layer lets ROCm DMA directly from its resident file-cache pages
     * instead of internally staging 1536 pageable tensor copies. Keep two
     * layer ranges registered so N+1 can transfer while N computes. */
    if (!pinned_stage && hipHostRegister && host_lo < host_hi) {
        uintptr_t lo = host_lo & ~(uintptr_t)4095;
        uintptr_t hi = (host_hi + 4095) & ~(uintptr_t)4095;
        size_t span = hi - lo;
        if (span <= (size_t)6 * 1024 * 1024 * 1024ULL &&
            hipHostRegister((void *)lo, span, hipHostRegisterDefault) == hipSuccess) {
            ctx->stream_slot_hreg[slot] = (void *)lo;
            ctx->stream_slot_hreg_bytes[slot] = span;
            ctx->prof_host_registers++;
            ctx->prof_host_register_bytes += span;
        }
    }
    ctx->stream_slot_layer[slot] = layer;

    size_t wo = 0, so = 0;
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi)
        for (int e = 0; e < layer->n_owned; ++e) {
            ds4f_tensor *t = (ds4f_tensor *)&ex[wi][e];
            size_t wb = (size_t)t->rows * (size_t)(t->cols / 2);
            size_t sb = (size_t)t->rows * (size_t)(t->cols / 32);
            void *dw = (uint8_t *)ctx->stream_slot_dw[slot] + wo;
            void *ds = (uint8_t *)ctx->stream_slot_ds[slot] + so;
            int id = append_device_matrix(ctx, dw, ds, t->w, t->scale,
                                          t->rows, t->cols, t->cols / 32,
                                          HIP_DS4F_MATRIX_MXFP4, 1);
            if (id < 0) {
                release_stream_slot(ctx, slot);
                return -1;
            }
            t->gpu_id = id; wo += wb; so += sb;
        }
    ctx->stream_pending_layer = layer;
    ctx->stream_pending_slot = slot;
    pthread_mutex_lock(&ctx->stream_copy_mu);
    ctx->stream_copy_submitted = 0;
    ctx->stream_copy_result = -1;
    ctx->stream_copy_request_slot = slot;
    ctx->stream_copy_request = 1;
    pthread_cond_signal(&ctx->stream_copy_cv);
    pthread_mutex_unlock(&ctx->stream_copy_mu);
    return 0;
}

int hip_ds4f_dense_prefetch_layer_raw(void *opaque, const ds4f_layer *layer) {
    return prefetch_layer_raw_async((hip_ds4f_dense *)opaque, layer);
}

int hip_ds4f_dense_begin_layer(void *opaque, const ds4f_layer *layer) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (!ctx || !layer) return -1;
    for (int i = 0; i < ctx->n_resident_layers; ++i)
        if (ctx->resident_layers[i].layer == layer && ctx->resident_layers[i].raw)
            return 0;
    if (ctx->stream_active_slot >= 0 && ctx->stream_slot_layer[ctx->stream_active_slot] == layer) return 0;
    if (ctx->stream_pending_layer != layer || ctx->stream_pending_slot < 0) return -1;
    int slot = ctx->stream_pending_slot;
    pthread_mutex_lock(&ctx->stream_copy_mu);
    while (!ctx->stream_copy_submitted && !ctx->stream_copy_stop)
        pthread_cond_wait(&ctx->stream_copy_ready_cv, &ctx->stream_copy_mu);
    int copy_result = ctx->stream_copy_result;
    pthread_mutex_unlock(&ctx->stream_copy_mu);
    if (copy_result != 0) return -1;
    if (hipStreamWaitEvent(ctx->stream, ctx->stream_copy_done[slot], 0) != hipSuccess)
        return -1;
    if (ctx->stream_active_slot >= 0 && ctx->stream_active_slot != slot)
        release_stream_slot(ctx, ctx->stream_active_slot);
    ctx->stream_active_slot = slot;
    ctx->stream_pending_slot = -1;
    ctx->stream_pending_layer = NULL;
    return 0;
}

int hip_ds4f_dense_resident_mxfp4_layer(void *opaque, const ds4f_layer *layer, int raw) {
    return stream_layer_impl(opaque, layer, raw != 0, 1);
}

typedef struct {
    uint64_t hits;
    int layer, expert;
} hip_ds4f_hot_expert;

typedef struct {
    uint64_t hits;
    int layer;
} hip_ds4f_hot_layer;

static int hot_expert_cmp(const void *ap, const void *bp) {
    const hip_ds4f_hot_expert *a = (const hip_ds4f_hot_expert *)ap;
    const hip_ds4f_hot_expert *b = (const hip_ds4f_hot_expert *)bp;
    if (a->hits != b->hits) return a->hits < b->hits ? 1 : -1;
    if (a->layer != b->layer) return a->layer - b->layer;
    return a->expert - b->expert;
}

static int hot_layer_cmp(const void *ap, const void *bp) {
    const hip_ds4f_hot_layer *a = (const hip_ds4f_hot_layer *)ap;
    const hip_ds4f_hot_layer *b = (const hip_ds4f_hot_layer *)bp;
    if (a->hits != b->hits) return a->hits < b->hits ? 1 : -1;
    return a->layer - b->layer;
}

static size_t expert_bundle_bytes(const ds4f_layer *layer, int slot) {
    if (!layer || slot < 0 || slot >= layer->n_owned) return 0;
    const ds4f_tensor *t[3] = { &layer->ex_w1[slot], &layer->ex_w3[slot],
                                &layer->ex_w2[slot] };
    size_t total = 0;
    for (int i = 0; i < 3; ++i) {
        if (t[i]->type != DS4F_MXFP4 || !valid_dims(t[i]->rows, t[i]->cols) ||
            (t[i]->cols & 31)) return 0;
        size_t wb = (size_t)t[i]->rows * (size_t)(t[i]->cols / 2);
        size_t sb = (size_t)t[i]->rows * (size_t)(t[i]->cols / 32);
        if (SIZE_MAX - total < wb + sb) return 0;
        total += wb + sb;
    }
    return total;
}

int hip_ds4f_dense_cache_hot_experts(void *opaque, void *model_opaque,
                                     int cache_mb, int reserve_mb, int stats) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    ds4f_model *model = (ds4f_model *)model_opaque;
    if (!ctx || !model || cache_mb == 0 || !model->route_hits ||
        hipSetDevice(ctx->device_id) != hipSuccess) return 0;
    const int L = model->cfg.n_layers, E = model->cfg.n_experts;
    if (L <= 0 || E <= 0 || (size_t)L > SIZE_MAX / (size_t)E) return -1;
    const size_t ncand = (size_t)L * (size_t)E;
    hip_ds4f_hot_expert *cand = (hip_ds4f_hot_expert *)malloc(ncand * sizeof(*cand));
    if (!cand) return -1;
    size_t n = 0;
    for (int l = 0; l < L; ++l) for (int e = 0; e < E; ++e) {
        uint64_t hits = model->route_hits[(size_t)l * E + e];
        if (hits && e % model->ep_size == model->ep_rank)
            cand[n++] = (hip_ds4f_hot_expert){ hits, l, e };
    }
    qsort(cand, n, sizeof(*cand), hot_expert_cmp);

    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        free(cand); return -1;
    }
    (void)total_bytes;
    const size_t reserve = (size_t)(reserve_mb > 0 ? reserve_mb : 1536) * 1048576u;
    size_t budget = free_bytes > reserve ? free_bytes - reserve : 0;
    if (cache_mb > 0 && (size_t)cache_mb * 1048576u < budget)
        budget = (size_t)cache_mb * 1048576u;
    size_t used = 0;
    uint64_t covered = 0, total_hits = 0;
    int admitted = 0;
    for (size_t i = 0; i < n; ++i) total_hits += cand[i].hits;
    for (size_t i = 0; i < n; ++i) {
        ds4f_layer *layer = &model->layers[cand[i].layer];
        int slot = cand[i].expert / model->ep_size;
        size_t bytes = expert_bundle_bytes(layer, slot);
        if (!bytes || bytes > budget - used) continue;
        ds4f_tensor *t[3] = { &layer->ex_w1[slot], &layer->ex_w3[slot],
                              &layer->ex_w2[slot] };
        if (t[0]->gpu_id >= 0 && t[1]->gpu_id >= 0 && t[2]->gpu_id >= 0) {
            covered += cand[i].hits;
            continue;
        }
        if (hip_ds4f_dense_bind_mxfp4_tensor(ctx, t[0]) < 0 ||
            hip_ds4f_dense_bind_mxfp4_tensor(ctx, t[1]) < 0 ||
            hip_ds4f_dense_bind_mxfp4_tensor(ctx, t[2]) < 0) {
            free(cand); return -1;
        }
        used += bytes;
        covered += cand[i].hits;
        admitted++;
    }

    /* A routed decode can use the GPU callback only when all six selected
     * experts in a layer are resident.  Prompt-hot admission normally leaves
     * most of the VRAM budget unused because only experts observed in the
     * prompt are candidates.  Fill that slack with complete hot layers so a
     * future decode token gets an all-GPU routed step instead of paying a
     * host fallback for a partially cached layer. */
    hip_ds4f_hot_layer *layers = (hip_ds4f_hot_layer *)calloc((size_t)L, sizeof(*layers));
    if (layers) {
        for (int l = 0; l < L; ++l) layers[l].layer = l;
        for (int l = 0; l < L; ++l)
            for (int e = 0; e < E; ++e)
                if (e % model->ep_size == model->ep_rank)
                    layers[l].hits += model->route_hits[(size_t)l * E + e];
        qsort(layers, (size_t)L, sizeof(*layers), hot_layer_cmp);
        for (int li = 0; li < L; ++li) {
            int l = layers[li].layer;
            size_t layer_bytes = 0;
            for (int e = 0; e < E; ++e) if (e % model->ep_size == model->ep_rank)
                layer_bytes += expert_bundle_bytes(&model->layers[l], e / model->ep_size);
            if (!layer_bytes || layer_bytes > budget - used) continue;
            for (int e = 0; e < E; ++e) {
                if (e % model->ep_size != model->ep_rank) continue;
                ds4f_tensor *t[3] = {
                    &model->layers[l].ex_w1[e / model->ep_size],
                    &model->layers[l].ex_w3[e / model->ep_size],
                    &model->layers[l].ex_w2[e / model->ep_size] };
                if (t[0]->gpu_id < 0 || t[1]->gpu_id < 0 || t[2]->gpu_id < 0) {
                    if (hip_ds4f_dense_bind_mxfp4_tensor(ctx, t[0]) < 0 ||
                        hip_ds4f_dense_bind_mxfp4_tensor(ctx, t[1]) < 0 ||
                        hip_ds4f_dense_bind_mxfp4_tensor(ctx, t[2]) < 0) {
                        free(layers); free(cand); return -1;
                    }
                }
            }
            used += layer_bytes;
            admitted += E / model->ep_size;
        }
        free(layers);
    }
    if (stats || ctx->verbose) {
        fprintf(stderr,
                "hip_ds4f_dense: prompt-hot expert cache bundles=%d bytes=%.3f GB "
                "training_coverage=%.2f%% reserve=%d MB\n",
                admitted, (double)used / 1e9,
                total_hits ? 100.0 * (double)covered / (double)total_hits : 0.0,
                reserve_mb > 0 ? reserve_mb : 1536);
    }
    free(cand);
    return admitted;
}

int hip_ds4f_dense_bind_fp8_ordered_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_FP8) return -1;
    int id = hip_ds4f_dense_bind_tensor(ctx, t);
    if (id >= 0) ctx->matrices[id].kind = HIP_DS4F_MATRIX_FP8_ORDERED;
    return id;
}

int hip_ds4f_dense_bind_bf16_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_BF16 || !t->w ||
        !valid_dims(t->rows, t->cols) || ctx->pending || ctx->multi_pending)
        return -1;
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    size_t w_bytes = (size_t)t->rows * (size_t)t->cols * sizeof(uint16_t);
    void *dw = NULL;
    if (hipMalloc(&dw, w_bytes) != hipSuccess ||
        hipMemcpy(dw, t->w, w_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: BF16 matrix upload failed\n");
        if (dw) hipFree(dw);
        return -1;
    }
    if (ctx->n_matrices == ctx->cap_matrices) {
        int cap = ctx->cap_matrices ? ctx->cap_matrices * 2 : 8;
        if (cap < ctx->n_matrices || cap > INT_MAX / (int)sizeof(*ctx->matrices)) {
            hipFree(dw);
            return -1;
        }
        hip_ds4f_matrix *p = (hip_ds4f_matrix *)ds4f_mem_realloc(
            ctx->mem, ctx->matrices,
            (size_t)ctx->cap_matrices * sizeof(*ctx->matrices),
            (size_t)cap * sizeof(*ctx->matrices), 64);
        if (!p) {
            hipFree(dw);
            return -1;
        }
        ctx->matrices = p;
        ctx->cap_matrices = cap;
    }
    int id = ctx->n_matrices++;
    ctx->matrices[id] = (hip_ds4f_matrix){ dw, NULL, t->w, NULL,
                                           t->rows, t->cols, 0,
                                           HIP_DS4F_MATRIX_BF16 };
    ctx->current = id;
    t->gpu_id = id;
    return id;
}

static int fp8_scaled_to_f16(uint8_t w, uint8_t scale, uint16_t *out) {
    uint32_t bits = ds4f_fp8_e4m3fn_to_fp32_bits(w);
    float v;
    memcpy(&v, &bits, sizeof(v));
    v *= ggml_e8m0_to_fp32(scale);
    if (!isfinite(v) || fabsf(v) > 65504.0f) return -1;
    union { float f; uint32_t u; } fu = { v };
    uint32_t sign_bit = (fu.u >> 16) & 0x8000u;
    int fexp = (int)((fu.u >> 23) & 0xffu) - 127;
    uint32_t mantissa = fu.u & 0x7fffffu;
    uint16_t h;
    if (fexp < -24) h = (uint16_t)sign_bit;
    else if (fexp < -14) {
        mantissa |= 0x800000u;
        h = (uint16_t)(sign_bit | (mantissa >> (13 + (-14 - fexp))));
    } else {
        h = (uint16_t)(sign_bit | ((uint32_t)(fexp + 15) << 10) |
                       (mantissa >> 13));
    }
    /* The conversion is exact for the normal FP8 range. Check the edge cases
     * (FP8 subnormals combined with very small E8M0 scales) so this mode never
     * silently changes a weight and is unavailable rather than approximate. */
    uint32_t hb = h;
    uint32_t sign = (hb >> 15) & 1u, exp = (hb >> 10) & 31u, mant = hb & 1023u;
    float back;
    if (exp == 0) {
        if (mant == 0) back = 0.0f;
        else {
            int sh = 0; uint32_t m = mant;
            while ((m & 1024u) == 0) { m <<= 1; ++sh; }
            m &= 1023u;
            uint32_t fb = (sign << 31) | ((uint32_t)(113 - sh) << 23) | (m << 13);
            memcpy(&back, &fb, sizeof(back));
        }
    } else {
        uint32_t fb = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        memcpy(&back, &fb, sizeof(back));
    }
    if (back != v) return -1;
    *out = h;
    return 0;
}

int hip_ds4f_dense_bind_fp8_fp16_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_FP8 || !t->w || !t->scale ||
        !valid_dims(t->rows, t->cols) || ctx->pending || ctx->multi_pending)
        return -1;
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;

    size_t elems = (size_t)t->rows * (size_t)t->cols;
    size_t w_bytes = elems * sizeof(uint16_t);
    uint16_t *promoted = (uint16_t *)ds4f_mem_alloc(ctx->mem, w_bytes, 64, 0);
    if (!promoted) return -1;
    int scale_cols = (t->cols + 127) / 128;
    for (int r = 0; r < t->rows; ++r) {
        const uint8_t *wr = (const uint8_t *)t->w + (size_t)r * t->cols;
        const uint8_t *sr = t->scale + (size_t)(r >> 7) * scale_cols;
        uint16_t *dr = promoted + (size_t)r * t->cols;
        for (int c = 0; c < t->cols; ++c)
            if (fp8_scaled_to_f16(wr[c], sr[c >> 7], &dr[c]) != 0) {
                fprintf(stderr, "hip_ds4f_dense: FP8->FP16 exact conversion unavailable at row=%d col=%d w=0x%02x scale=0x%02x\n",
                        r, c, wr[c], sr[c >> 7]);
                return -1;
            }
    }

    void *dw = NULL;
    if (hipMalloc(&dw, w_bytes) != hipSuccess ||
        hipMemcpy(dw, promoted, w_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        if (dw) hipFree(dw);
        return -1;
    }
    if (ctx->n_matrices == ctx->cap_matrices) {
        int cap = ctx->cap_matrices ? ctx->cap_matrices * 2 : 8;
        if (cap < ctx->n_matrices || cap > INT_MAX / (int)sizeof(*ctx->matrices)) {
            hipFree(dw);
            return -1;
        }
        hip_ds4f_matrix *p = (hip_ds4f_matrix *)ds4f_mem_realloc(
            ctx->mem, ctx->matrices,
            (size_t)ctx->cap_matrices * sizeof(*ctx->matrices),
            (size_t)cap * sizeof(*ctx->matrices), 64);
        if (!p) {
            hipFree(dw);
            return -1;
        }
        ctx->matrices = p;
        ctx->cap_matrices = cap;
    }
    int id = ctx->n_matrices++;
    ctx->matrices[id] = (hip_ds4f_matrix){ dw, NULL, t->w, t->scale,
                                           t->rows, t->cols, scale_cols,
                                           HIP_DS4F_MATRIX_FP8_FP16 };
    ctx->current = id;
    t->gpu_id = id;
    return id;
}

int hip_ds4f_dense_bind_fp8_bf16_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_FP8 || !t->w || !t->scale ||
        !valid_dims(t->rows, t->cols) || ctx->pending || ctx->multi_pending)
        return -1;
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;

    size_t elems = (size_t)t->rows * (size_t)t->cols;
    size_t w_bytes = elems * sizeof(uint16_t);
    uint16_t *promoted = (uint16_t *)ds4f_mem_alloc(ctx->mem, w_bytes, 64, 0);
    if (!promoted) return -1;
    int scale_cols = (t->cols + 127) / 128;
    for (int r = 0; r < t->rows; ++r) {
        const uint8_t *wr = (const uint8_t *)t->w + (size_t)r * t->cols;
        const uint8_t *sr = t->scale + (size_t)(r >> 7) * scale_cols;
        uint16_t *dr = promoted + (size_t)r * t->cols;
        for (int c = 0; c < t->cols; ++c) {
            uint32_t bits = ds4f_fp8_e4m3fn_to_fp32_bits(wr[c]);
            float v;
            memcpy(&v, &bits, sizeof(v));
            v *= ggml_e8m0_to_fp32(sr[c >> 7]);
            memcpy(&bits, &v, sizeof(bits));
            dr[c] = (uint16_t)(bits >> 16);
        }
    }

    void *dw = NULL;
    if (hipMalloc(&dw, w_bytes) != hipSuccess ||
        hipMemcpy(dw, promoted, w_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        if (dw) hipFree(dw);
        return -1;
    }
    if (ctx->n_matrices == ctx->cap_matrices) {
        int cap = ctx->cap_matrices ? ctx->cap_matrices * 2 : 8;
        if (cap < ctx->n_matrices || cap > INT_MAX / (int)sizeof(*ctx->matrices)) {
            hipFree(dw);
            return -1;
        }
        hip_ds4f_matrix *p = (hip_ds4f_matrix *)ds4f_mem_realloc(
            ctx->mem, ctx->matrices,
            (size_t)ctx->cap_matrices * sizeof(*ctx->matrices),
            (size_t)cap * sizeof(*ctx->matrices), 64);
        if (!p) {
            hipFree(dw);
            return -1;
        }
        ctx->matrices = p;
        ctx->cap_matrices = cap;
    }
    int id = ctx->n_matrices++;
    ctx->matrices[id] = (hip_ds4f_matrix){ dw, NULL, t->w, t->scale,
                                           t->rows, t->cols, scale_cols,
                                           HIP_DS4F_MATRIX_FP8_BF16 };
    ctx->current = id;
    t->gpu_id = id;
    return id;
}

int hip_ds4f_dense_load(hip_ds4f_dense *ctx,
                        const uint8_t *w, const uint8_t *s,
                        int rows, int cols) {
    if (!ctx || ctx->pending || ctx->multi_pending) {
        fprintf(stderr, "hip_ds4f_dense: invalid load or work in flight\n");
        return -1;
    }
    clear_matrices(ctx);
    return hip_ds4f_dense_add(ctx, w, s, rows, cols);
}

static int matrix_get(const hip_ds4f_dense *ctx, int id,
                      const hip_ds4f_matrix **out) {
    if (!ctx || id < 0 || id >= ctx->n_matrices || !out ||
        !ctx->matrices[id].dw) {
        fprintf(stderr, "hip_ds4f_dense: invalid matrix id %d\n", id);
        return -1;
    }
    *out = &ctx->matrices[id];
    return 0;
}

static int matrix_view(const hip_ds4f_matrix *mat, const ds4f_tensor *t,
                       void **dw_out, void **ds_out) {
    if (!mat || !t || !dw_out || !ds_out || mat->cols != t->cols ||
        mat->rows < t->rows) return -1;
    size_t row_bytes = (t->type == DS4F_FP8 || t->type == DS4F_MXFP4)
        ? (size_t)t->cols : (size_t)t->cols * sizeof(uint16_t);
    int row0 = 0;
    if (mat->rows != t->rows) {
        if (!mat->hw || !t->w || (const uint8_t *)t->w < (const uint8_t *)mat->hw ||
            !row_bytes) return -1;
        size_t delta = (size_t)((const uint8_t *)t->w - (const uint8_t *)mat->hw);
        if (delta % row_bytes != 0) return -1;
        row0 = (int)(delta / row_bytes);
        if (row0 < 0 || row0 > mat->rows - t->rows) return -1;
        if (t->type == DS4F_FP8) {
            if (!mat->hs || row0 % 128 != 0 ||
                t->scale != (const uint8_t *)mat->hs +
                    (size_t)(row0 / 128) * (size_t)mat->scale_cols)
                return -1;
        }
    }
    *dw_out = (uint8_t *)(void *)mat->dw + (size_t)row0 * row_bytes;
    *ds_out = mat->ds;
    if (t->type == DS4F_FP8 && row0)
        *ds_out = (uint8_t *)(void *)mat->ds +
                  (size_t)(row0 / 128) * (size_t)mat->scale_cols;
    return 0;
}

static int launch_matvec_async(hip_ds4f_dense *ctx, hipFunction_t fn, int id,
                               void *dw, void *ds, int rows, int cols,
                               int scale_cols, const float *x, int x_cols) {
    if (!ctx || !fn || !x || rows <= 0 || cols <= 0 || x_cols < cols) {
        fprintf(stderr, "hip_ds4f_dense: invalid matvec launch\n");
        return -1;
    }
    if (ctx->pending || ctx->multi_pending) {
        fprintf(stderr, "hip_ds4f_dense: previous matvec is still in flight\n");
        return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess ||
        ensure_vectors(ctx, rows, x_cols) != 0)
        return -1;
    size_t xb = (size_t)x_cols * sizeof(float);
    const void *xsrc = x;
    if (ensure_pinned(&ctx->pin_hx, &ctx->pin_hx_cap, xb) == 0) {
        memcpy(ctx->pin_hx, x, xb);
        xsrc = ctx->pin_hx;
    }
    if (hipMemcpyAsync(ctx->dx, xsrc, xb,
                       hipMemcpyHostToDevice, ctx->stream) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: activation upload failed\n");
        hipStreamSynchronize(ctx->stream);
        return -1;
    }

    void *args[] = { &dw, &ds, &ctx->dx, &ctx->dy, &rows, &cols, &scale_cols };
    hipError_t err = hipModuleLaunchKernel(fn,
        (unsigned int)((rows + (ctx->block_threads >> 5) - 1) / (ctx->block_threads >> 5)),
        1, 1, (unsigned int)ctx->block_threads, 1, 1, 0,
        ctx->stream, args, NULL);
    if (err != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: kernel launch failed (%d)\n", (int)err);
        hipStreamSynchronize(ctx->stream);
        return -1;
    }
    if (hipEventRecord(ctx->done, ctx->stream) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: event record failed\n");
        hipStreamSynchronize(ctx->stream);
        return -1;
    }
    ctx->pending = 1;
    ctx->pending_id = id;
    ctx->pending_rows = rows;
    return 0;
}

int hip_ds4f_dense_matvec_id_async(hip_ds4f_dense *ctx, int id, const float *x) {
    const hip_ds4f_matrix *mat = NULL;
    if (!x || matrix_get(ctx, id, &mat) != 0) {
        fprintf(stderr, "hip_ds4f_dense: no matrix or invalid input\n");
        return -1;
    }
    if (matrix_is_mxfp4(mat->kind)) {
        /* M=1 decode expert matvec: the 16-row gemm tile wastes the FMA on 15
         * zero rows, but the weights are read once at VRAM bandwidth -- far
         * faster than the CPU MXFP4 single-token decode for a routed expert. */
        if (ctx->pending || ctx->multi_pending) {
            fprintf(stderr, "hip_ds4f_dense: previous matvec is still in flight\n");
            return -1;
        }
        if (hipSetDevice(ctx->device_id) != hipSuccess ||
            ensure_vectors(ctx, mat->rows, mat->cols) != 0)
            return -1;
        size_t xb = (size_t)mat->cols * sizeof(float);
        const void *xsrc = x;
        if (ensure_pinned(&ctx->pin_hx, &ctx->pin_hx_cap, xb) == 0) {
            memcpy(ctx->pin_hx, x, xb);
            xsrc = ctx->pin_hx;
        }
        if (hipMemcpyAsync(ctx->dx, xsrc, xb, hipMemcpyHostToDevice, ctx->stream) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: activation upload failed\n");
            hipStreamSynchronize(ctx->stream);
            return -1;
        }
        int n_out = mat->rows, n_in = mat->cols;
        void *dw = mat->dw, *ds = mat->ds, *dx = ctx->dx, *dy = ctx->dy;
        void *args[] = { &dy, &dw, &ds, &dx, &n_out, &n_in };
        hipError_t err = hipModuleLaunchKernel(ctx->mxfp4_matvec,
            (unsigned int)((n_out + 7) / 8), 1, 1, 256, 1, 1, 0,
            ctx->stream, args, NULL);
        if (err != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: mxfp4 matvec launch failed (%d)\n", (int)err);
            hipStreamSynchronize(ctx->stream);
            return -1;
        }
        if (hipEventRecord(ctx->done, ctx->stream) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: event record failed\n");
            hipStreamSynchronize(ctx->stream);
            return -1;
        }
        ctx->pending = 1;
        ctx->pending_id = id;
        ctx->pending_rows = mat->rows;
        return 0;
    }
    hipFunction_t fn = matrix_is_fp16(mat->kind) ? ctx->f16_matvec :
                       matrix_is_bf16(mat->kind) ? ctx->bf16_matvec : ctx->matvec;
    return launch_matvec_async(ctx, fn, id, mat->dw, mat->ds,
                               mat->rows, mat->cols, mat->scale_cols,
                               x, mat->cols);
}

int hip_ds4f_dense_matvec_id(hip_ds4f_dense *ctx, int id,
                             const float *x, float *y) {
    if (hip_ds4f_dense_matvec_id_async(ctx, id, x) != 0) return -1;
    return hip_ds4f_dense_wait(ctx, y);
}

int hip_ds4f_dense_matvec_tensor(void *opaque, float *dst,
                                 const ds4f_tensor *t, const float *x) {
    if (!opaque || !t || (t->type != DS4F_FP8 && t->type != DS4F_BF16 &&
                           t->type != DS4F_MXFP4) || t->gpu_id < 0)
        return -1;
    return hip_ds4f_dense_matvec_id((hip_ds4f_dense *)opaque, t->gpu_id, x, dst);
}

int hip_ds4f_dense_matvec_blockdiag(
    void *opaque, float *dst, const ds4f_tensor *t, const float *xbase,
    int gin, int glora, int goff) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    const hip_ds4f_matrix *mat = NULL;
    if (!ctx || !dst || !t || !xbase || t->type != DS4F_FP8 ||
        t->gpu_id < 0 || gin <= 0 || glora <= 0 || goff < 0 ||
        matrix_get(ctx, t->gpu_id, &mat) != 0 ||
        t->rows != mat->rows || t->cols != mat->cols) {
        fprintf(stderr, "hip_ds4f_dense: invalid block-diagonal matvec\n");
        return -1;
    }
    int groups_end = (goff + t->rows + glora - 1) / glora;
    if (groups_end <= 0 || groups_end > INT_MAX / gin) return -1;
    if (ctx->pending || ctx->multi_pending ||
        hipSetDevice(ctx->device_id) != hipSuccess ||
        ensure_vectors(ctx, t->rows, groups_end * gin) != 0)
        return -1;
    if (hipMemcpyAsync(ctx->dx, xbase,
                       (size_t)groups_end * (size_t)gin * sizeof(float),
                       hipMemcpyHostToDevice, ctx->stream) != hipSuccess)
        return -1;
    int rows = t->rows, cols = t->cols, scale_cols = mat->scale_cols;
    void *dw = mat->dw, *ds = mat->ds;
    void *args[] = { &dw, &ds, &ctx->dx, &ctx->dy,
                     &rows, &cols, &scale_cols, &gin, &glora, &goff };
    if (hipModuleLaunchKernel(ctx->blockdiag_matvec,
            (unsigned int)((rows + (ctx->block_threads >> 5) - 1) / (ctx->block_threads >> 5)),
            1, 1, (unsigned int)ctx->block_threads, 1, 1, 0,
            ctx->stream, args, NULL) != hipSuccess ||
        hipEventRecord(ctx->done, ctx->stream) != hipSuccess)
        return -1;
    ctx->pending = 1;
    ctx->pending_id = t->gpu_id;
    ctx->pending_rows = rows;
    return hip_ds4f_dense_wait(ctx, dst);
}

static int ensure_gemm_vectors(hip_ds4f_dense *ctx, size_t x_bytes, size_t y_bytes) {
    if (ctx->gemm_dx && ctx->gemm_dy && ctx->gemm_x_bytes >= x_bytes &&
        ctx->gemm_y_bytes >= y_bytes)
        return 0;
    void *dx = NULL, *dy = NULL;
    if (hipMalloc(&dx, x_bytes) != hipSuccess ||
        hipMalloc(&dy, y_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: batched GEMM allocation failed\n");
        if (dx) hipFree(dx);
        if (dy) hipFree(dy);
        return -1;
    }
    if (ctx->gemm_dx) hipFree(ctx->gemm_dx);
    if (ctx->gemm_dy) hipFree(ctx->gemm_dy);
    ctx->gemm_dx = dx; ctx->gemm_dy = dy;
    ctx->gemm_x_bytes = x_bytes; ctx->gemm_y_bytes = y_bytes;
    return 0;
}

static int ensure_gemm_y_tile(hip_ds4f_dense *ctx, size_t y_bytes) {
    if (ctx->gemm_dy_tile && ctx->gemm_dy_tile_bytes >= y_bytes) return 0;
    void *dy = NULL;
    if (hipMalloc(&dy, y_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: batched GEMM tile allocation failed\n");
        return -1;
    }
    if (ctx->gemm_dy_tile) hipFree(ctx->gemm_dy_tile);
    ctx->gemm_dy_tile = dy; ctx->gemm_dy_tile_bytes = y_bytes;
    return 0;
}

static int ensure_gemm_yh_tile(hip_ds4f_dense *ctx, size_t y_bytes) {
    if (ctx->gemm_yh_tile && ctx->gemm_yh_tile_bytes >= y_bytes) return 0;
    void *yh_mem = NULL;
    if (hipHostMalloc(&yh_mem, y_bytes, 0) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: batched GEMM host tile allocation failed\n");
        return -1;
    }
    if (ctx->gemm_yh_tile) hipHostFree(ctx->gemm_yh_tile);
    ctx->gemm_yh_tile = (float *)yh_mem; ctx->gemm_yh_tile_bytes = y_bytes;
    return 0;
}

static int ensure_gemm_x(hip_ds4f_dense *ctx, size_t x_bytes) {
    if (ctx->gemm_dx && ctx->gemm_x_bytes >= x_bytes) return 0;
    void *dx = NULL;
    if (hipMalloc(&dx, x_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: batched GEMM input allocation failed\n");
        return -1;
    }
    if (ctx->gemm_dx) hipFree(ctx->gemm_dx);
    ctx->gemm_dx = dx;
    ctx->gemm_x_bytes = x_bytes;
    return 0;
}

static int ensure_gemm_multi_outputs(hip_ds4f_dense *ctx, int n,
                                     const size_t *bytes) {
    for (int i = 0; i < n; ++i) {
        if (ctx->gemm_multi_dy[i] && ctx->gemm_multi_y_bytes[i] >= bytes[i])
            continue;
        void *dy = NULL;
        if (hipMalloc(&dy, bytes[i]) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: fused GEMM output allocation failed\n");
            if (dy) hipFree(dy);
            return -1;
        }
        if (ctx->gemm_multi_dy[i]) hipFree(ctx->gemm_multi_dy[i]);
        ctx->gemm_multi_dy[i] = dy;
        ctx->gemm_multi_y_bytes[i] = bytes[i];
    }
    return 0;
}

static int ensure_mxfp4_task_buffer(hip_ds4f_dense *ctx, int n) {
    size_t bytes = (size_t)n * sizeof(hip_ds4f_mxfp4_task);
    if (ctx->gemm_mxfp4_tasks && ctx->gemm_mxfp4_tasks_bytes >= bytes)
        return 0;
    void *p = NULL;
    if (hipMalloc(&p, bytes) != hipSuccess) return -1;
    if (ctx->gemm_mxfp4_tasks) hipFree(ctx->gemm_mxfp4_tasks);
    ctx->gemm_mxfp4_tasks = p;
    ctx->gemm_mxfp4_tasks_bytes = bytes;
    return 0;
}

static int ensure_attn_buffer(void **p, size_t *have, size_t need) {
    if (*p && *have >= need) return 0;
    void *q = NULL;
    if (hipMalloc(&q, need ? need : 1) != hipSuccess) return -1;
    if (*p) hipFree(*p);
    *p = q; *have = need;
    return 0;
}

int hip_ds4f_dense_prefill_attention(
    void *opaque, float *dst, const float *q, const uint16_t *kv,
    const float *sink, const float *rcos, const float *rsin,
    int rope_offset, int rope_pairs, int M, int pos0, int n_heads,
    int head_dim, int kv_dim, int kv_slots, int window, float scale) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (!ctx || !dst || !q || !kv || !sink || M < 1 || n_heads < 1 ||
        head_dim < 1 || kv_dim < 1 || kv_dim > head_dim || kv_slots < 1 ||
        window < 1 || window > 128 || pos0 < 0 ||
        !valid_dims(M * n_heads, head_dim) ||
        !valid_dims(kv_slots, kv_dim)) return -1;
    int base = pos0 - window + 1;
    if (base < 0) base = 0;
    int end = pos0 + M;
    if (end > kv_slots) return -1;
    int copy_slots = end - base;
    if (copy_slots < 1) copy_slots = 1;
    int resident_kv = ctx->decode_kv_resident && M == 1;
    if (resident_kv) { base = 0; copy_slots = kv_slots; }
    size_t qb = (size_t)M * (size_t)n_heads * (size_t)head_dim * sizeof(float);
    size_t kb = (size_t)copy_slots * (size_t)kv_dim * sizeof(uint16_t);
    size_t sb = (size_t)n_heads * sizeof(float);
    if (ensure_attn_buffer(&ctx->attn_q, &ctx->attn_q_bytes, qb) != 0 ||
        ensure_attn_buffer(&ctx->attn_kv, &ctx->attn_kv_bytes, kb) != 0 ||
        ensure_attn_buffer(&ctx->attn_sink, &ctx->attn_sink_bytes, sb) != 0 ||
        ensure_attn_buffer(&ctx->attn_y, &ctx->attn_y_bytes, qb) != 0 ||
        hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    ctx->attn_device_ready = 0;
    int qchained = ctx->qkv_device_chain && ctx->attn_device_chain &&
                   ctx->qkv_device_ready && ctx->qkv_device_M == M &&
                   ctx->qkv_device_host == q && ctx->qnorm_rope_heads &&
                   rcos && rsin;
    if ((!qchained && hipMemcpy(ctx->attn_q, q, qb, hipMemcpyHostToDevice) != hipSuccess) ||
        (!resident_kv && hipMemcpy(ctx->attn_kv, kv + (size_t)base * kv_dim, kb,
                  hipMemcpyHostToDevice) != hipSuccess) ||
        hipMemcpy(ctx->attn_sink, sink, sb, hipMemcpyHostToDevice) != hipSuccess)
        return -1;
    if (resident_kv) {
        size_t full_bytes=(size_t)kv_slots*kv_dim*sizeof(uint16_t);
        size_t one_bytes=(size_t)kv_dim*sizeof(uint16_t);
        if (ctx->decode_kv_host != kv || ctx->decode_kv_slots != kv_slots ||
            ctx->decode_kv_dim != kv_dim) {
            if (hipMemcpy(ctx->attn_kv, kv, full_bytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
            ctx->decode_kv_host=kv; ctx->decode_kv_slots=kv_slots; ctx->decode_kv_dim=kv_dim;
        } else if (hipMemcpy((uint8_t *)ctx->attn_kv + (size_t)(pos0 % kv_slots)*one_bytes,
                             kv + (size_t)(pos0 % kv_slots)*kv_dim, one_bytes,
                             hipMemcpyHostToDevice) != hipSuccess) return -1;
    }
    void *dy = ctx->attn_y, *dkv = ctx->attn_kv,
         *dq = qchained ? ctx->qkv_q : ctx->attn_q, *dsink = ctx->attn_sink;
    if (qchained) {
        size_t cb = (size_t)M * (size_t)rope_pairs * sizeof(float);
        if (ensure_attn_buffer(&ctx->rope_c, &ctx->rope_c_bytes, cb) != 0 ||
            ensure_attn_buffer(&ctx->rope_s, &ctx->rope_s_bytes, cb) != 0 ||
            hipMemcpy(ctx->rope_c, rcos + (size_t)pos0*rope_pairs, cb,
                      hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(ctx->rope_s, rsin + (size_t)pos0*rope_pairs, cb,
                      hipMemcpyHostToDevice) != hipSuccess) return -1;
        float eps = 1.0e-6f;
        void *ra[] = { &ctx->qkv_q, &ctx->rope_c, &ctx->rope_s, &M,
                       &n_heads, &head_dim, &rope_offset, &rope_pairs, &eps };
        if (hipModuleLaunchKernel(ctx->qnorm_rope_heads,
                (unsigned)(M*n_heads),1,1,256,1,1,0,ctx->stream,ra,NULL) != hipSuccess) return -1;
    }
    int groups = (n_heads + 7) >> 3;
    int use_wmma = ctx->attn_wmma && ctx->prefill_attn_wmma &&
                   head_dim == 512 && kv_dim == 512;
    unsigned int gx = (unsigned int)(use_wmma ? n_heads * ((M + 15) / 16) : M * groups);
    int local_pos0 = resident_kv ? pos0 : pos0 - base;
    void *args[] = { &dy, &dkv, &dq, &dsink, &M, &local_pos0, &n_heads,
                     &head_dim, &kv_dim, &copy_slots, &window, &scale };
    hipError_t err = hipModuleLaunchKernel(use_wmma ? ctx->prefill_attn_wmma : ctx->prefill_attn,
                                            gx, 1, 1, 256, 1, 1, 0, ctx->stream,
                                            args, NULL);
    int no_d2h = ctx->attn_no_d2h && ctx->attn_device_chain;
    if (err != hipSuccess) return -1;
    /* Attention and O-projection are queued on the same stream.  In the
     * device-chain case O-projection consumes attn_y directly, so waiting here
     * only inserts a host round-trip between two dependent GPU kernels.  Keep
     * the synchronization for legacy host-visible/fallback callers. */
    if (no_d2h) {
        /* dependency is carried by ctx->stream; synchronize at final D2H */
    } else if (hipStreamSynchronize(ctx->stream) != hipSuccess ||
               hipMemcpy(dst, ctx->attn_y, qb, hipMemcpyDeviceToHost) != hipSuccess)
        return -1;
    if (ctx->attn_device_chain) {
        ctx->attn_device_ready = 1;
        ctx->attn_device_M = M;
        ctx->attn_device_host = dst;
    }
    return 0;
}

static int launch_gemm_dev(hip_ds4f_dense *ctx, const hip_ds4f_matrix *mat, int row0, void *dY, void *dX, int N, int K, int M);
static int ensure_dev_buf(void **p, size_t *have, size_t need);
int hip_ds4f_dense_prefill_attn_oproj(void *opaque,float *dst,const float *q,const uint16_t *kv,
 const float *sink,const float *rcos,const float *rsin,const ds4f_tensor *wa,const ds4f_tensor *wb,
 int M,int pos0,int nh,int hd,int kd,int slots,int window,float scale,int ro,int rp,int groups,int gin,int lora,int H,int C,int ointer){
 hip_ds4f_dense *c=(hip_ds4f_dense*)opaque; const hip_ds4f_matrix *a=NULL,*b=NULL;
 if(!c||!dst||!q||!kv||!sink||!rcos||!rsin||!wa||!wb||!c->apply_rope||M<1||groups*gin!=H||groups*lora!=ointer||wa->gpu_id<0||wb->gpu_id<0||matrix_get(c,wa->gpu_id,&a)!=0||matrix_get(c,wb->gpu_id,&b)!=0||a->rows!=ointer||a->cols!=gin||b->rows!=C||b->cols!=ointer||!matrix_is_fp8(a->kind)||!matrix_is_fp8(b->kind))return -1;
 int base=pos0-window+1;if(base<0)base=0;int end=pos0+M;if(end>slots)return -1;int ns=end-base;if(ns<1)ns=1;
 size_t qb=(size_t)M*nh*hd*4,kb=(size_t)ns*kd*2,sb=(size_t)nh*4,cb=(size_t)M*rp*4,xtb=(size_t)M*gin*4,ytb=(size_t)M*lora*4,dib=(size_t)M*ointer*4,dyb=(size_t)M*C*4;
 if(hipSetDevice(c->device_id)!=hipSuccess||ensure_attn_buffer(&c->attn_q,&c->attn_q_bytes,qb)||ensure_attn_buffer(&c->attn_kv,&c->attn_kv_bytes,kb)||ensure_attn_buffer(&c->attn_sink,&c->attn_sink_bytes,sb)||ensure_attn_buffer(&c->attn_y,&c->attn_y_bytes,qb)||ensure_attn_buffer(&c->rope_c,&c->rope_c_bytes,cb)||ensure_attn_buffer(&c->rope_s,&c->rope_s_bytes,cb)||ensure_dev_buf(&c->op_xt,&c->op_xt_b,xtb)||ensure_dev_buf(&c->op_yt,&c->op_yt_b,ytb)||ensure_dev_buf(&c->op_di,&c->op_di_b,dib)||ensure_dev_buf(&c->op_dy,&c->op_dy_b,dyb))return -1;
 if(hipMemcpyAsync(c->attn_q,q,qb,hipMemcpyHostToDevice,c->stream)!=hipSuccess||hipMemcpyAsync(c->attn_kv,kv+(size_t)base*kd,kb,hipMemcpyHostToDevice,c->stream)!=hipSuccess||hipMemcpyAsync(c->attn_sink,sink,sb,hipMemcpyHostToDevice,c->stream)!=hipSuccess)return -1;
 int rope_cached = c->rope_host_c == rcos && c->rope_host_s == rsin && c->rope_host_pos == pos0 &&
                   c->rope_host_m == M && c->rope_host_pairs == rp;
 if (!rope_cached) {
     if(hipMemcpyAsync(c->rope_c,rcos+(size_t)pos0*rp,cb,hipMemcpyHostToDevice,c->stream)!=hipSuccess||
        hipMemcpyAsync(c->rope_s,rsin+(size_t)pos0*rp,cb,hipMemcpyHostToDevice,c->stream)!=hipSuccess)return -1;
     c->rope_host_c=rcos; c->rope_host_s=rsin; c->rope_host_pos=pos0; c->rope_host_m=M; c->rope_host_pairs=rp;
 }
 void *dy=c->attn_y,*dkv=c->attn_kv,*dq=c->attn_q,*ds=c->attn_sink;int lp=pos0-base;int use=c->attn_wmma&&c->prefill_attn_wmma&&hd==512&&kd==512;void *aa[]={&dy,&dkv,&dq,&ds,&M,&lp,&nh,&hd,&kd,&ns,&window,&scale};unsigned gx=(unsigned)(use?nh*((M+15)/16):M*((nh+7)>>3));
 if(hipModuleLaunchKernel(use?c->prefill_attn_wmma:c->prefill_attn,gx,1,1,256,1,1,0,c->stream,aa,NULL)!=hipSuccess)return -1;void *ra[]={&c->attn_y,&c->rope_c,&c->rope_s,&M,&hd,&H,&ro,&rp};if(hipModuleLaunchKernel(c->apply_rope,(unsigned)(((size_t)M*rp+255)/256),1,1,256,1,1,0,c->stream,ra,NULL)!=hipSuccess)return -1;
 int omode = c->oproj_group_wmma;
 int grouped_op = omode!=0 && c->gemm_fp8_grouped_wmma && matrix_is_fp8(a->kind);
 if (grouped_op) {
     hip_ds4f_mxfp4_task task[32]; memset(task,0,sizeof(task));
     for (int g=0; g<groups; ++g) {
         task[g].w=(uint8_t *)a->dw+(size_t)g*lora*gin;
         task[g].s=(uint8_t *)a->ds+(size_t)(g*lora/128)*a->scale_cols;
         task[g].x=c->attn_y; task[g].y=c->op_di; task[g].n_out=lora; task[g].n_in=gin; task[g].n_tok=M;
         task[g].x_stride=H; task[g].y_stride=ointer; task[g].x_off=g*gin; task[g].y_off=g*lora;
     }
     if (ensure_mxfp4_task_buffer(c,groups)!=0 || hipMemcpyAsync(c->gemm_mxfp4_tasks,task,(size_t)groups*sizeof(*task),hipMemcpyHostToDevice,c->stream)!=hipSuccess)return -1;
     int sc=a->scale_cols; void *ga[]={&c->gemm_mxfp4_tasks,&groups,&sc};
     if (hipModuleLaunchKernel(omode>=2 && c->gemm_fp8_grouped_wmma64 ? c->gemm_fp8_grouped_wmma64 : c->gemm_fp8_grouped_wmma,
             (unsigned)((lora+(omode>=2?63:15))/(omode>=2?64:16)), (unsigned)((M+(omode>=2?63:15))/(omode>=2?64:16)), (unsigned)groups,
             omode>=2 ? 256 : 32,1,1,0,c->stream,ga,NULL)!=hipSuccess)return -1;
 } else for(int g=0;g<groups;g++){size_t n=(size_t)M*gin;int off=g*gin;void *ga[]={&c->op_xt,&c->attn_y,&M,&gin,&H,&off};if(hipModuleLaunchKernel(c->gather_group,(unsigned)((n+255)/256),1,1,256,1,1,0,c->stream,ga,NULL)!=hipSuccess||launch_gemm_dev(c,a,g*lora,c->op_yt,c->op_xt,lora,gin,M)!=0)return -1;n=(size_t)M*lora;off=g*lora;void *sa[]={&c->op_di,&c->op_yt,&M,&lora,&ointer,&off};if(hipModuleLaunchKernel(c->scatter_group,(unsigned)((n+255)/256),1,1,256,1,1,0,c->stream,sa,NULL)!=hipSuccess)return -1;}
 if(launch_gemm_dev(c,b,0,c->op_dy,c->op_di,C,ointer,M)!=0||hipMemcpyAsync(dst,c->op_dy,dyb,hipMemcpyDeviceToHost,c->stream)!=hipSuccess||hipStreamSynchronize(c->stream)!=hipSuccess)return -1;return 0;
}

/* Prefer pinned memory for the GEMM staging buffers.  A batch-64 wq_b GEMM
 * downloads 8.4 MB per call, so on this host's PCIe gen3 x8 link the transfer
 * costs more than the arithmetic; a pageable copy makes the driver bounce it
 * through its own staging buffer on top of that.  Falls back to arena memory
 * when the driver has no hipHostMalloc. */
static int ensure_gemm_host_pack(hip_ds4f_dense *ctx, size_t x_bytes, size_t y_bytes) {
    if (x_bytes > ctx->gemm_x_pack_bytes) {
        void *p = NULL;
        if (!hipHostMalloc || hipHostMalloc(&p, x_bytes, hipHostMallocDefault) != hipSuccess)
            p = ds4f_mem_alloc(ctx->mem, x_bytes, 64, 0);
        if (!p) return -1;
        ctx->gemm_x_pack = (float *)p; ctx->gemm_x_pack_bytes = x_bytes;
    }
    if (y_bytes > ctx->gemm_y_pack_bytes) {
        void *p = NULL;
        if (!hipHostMalloc || hipHostMalloc(&p, y_bytes, hipHostMallocDefault) != hipSuccess)
            p = ds4f_mem_alloc(ctx->mem, y_bytes, 64, 0);
        if (!p) return -1;
        ctx->gemm_y_pack = (float *)p; ctx->gemm_y_pack_bytes = y_bytes;
    }
    return 0;
}

/* Launch one batched GEMM with caller-supplied device pointers.  Same kernels
 * and same launch geometry as hip_ds4f_dense_gemm_tensor(); it just does not
 * own the staging or the transfers, so chained projections can stay resident. */
static int launch_gemm_dev(hip_ds4f_dense *ctx, const hip_ds4f_matrix *mat,
                           int row0, void *dY, void *dX, int N, int K, int M) {
    unsigned int gx = (unsigned int)((N + 63) / 64);
    unsigned int gy = (unsigned int)((M + 15) / 16);
    int n_out = N, n_in = K, n_tok = M, scale_cols = mat->scale_cols;
    void *lut = ctx->fp8_lut;
    hipFunction_t fn;
    size_t soff;
    if (matrix_is_mxfp4(mat->kind)) {
        int n_out = N, n_in = K, n_tok = M;
        void *dy = dY;
        void *dw = (uint8_t *)mat->dw + (size_t)row0 * (size_t)(K / 2);
        void *ds = (uint8_t *)mat->ds + (size_t)row0 * (size_t)(K / 32);
        void *args[] = { &dy, &dw, &ds, &dX, &n_out, &n_in, &n_tok };
        return hipModuleLaunchKernel(ctx->gemm_mxfp4,
            (unsigned)((N + 63) / 64), (unsigned)((M + 15) / 16), 1,
            16, 16, 1, 0, ctx->stream, args, NULL) == hipSuccess ? 0 : -1;
    }
    if (matrix_is_fp8_rowscale(mat->kind)) { fn = ctx->gemm_fp8_rowscale; soff = (size_t)row0 * (size_t)mat->scale_cols; }
    else if (matrix_is_fp8_ordered(mat->kind)) { fn = ctx->gemm_fp8_ordered; soff = (size_t)(row0 / 128) * (size_t)mat->scale_cols; }
    else if (matrix_is_fp8(mat->kind)) { fn = ctx->gemm_fp8; soff = (size_t)(row0 / 128) * (size_t)mat->scale_cols; }
    else return -1;   /* promoted BF16/FP16 and MXFP4 keep the unfused path */
    void *dw = (uint8_t *)(void *)mat->dw + (size_t)row0 * (size_t)K;
    void *ds = (uint8_t *)(void *)mat->ds + soff;
    if (matrix_is_fp8(mat->kind) && ctx->fp8_wmma_mode != 0 && M >= 128) {
        int wm = ctx->fp8_wmma_mode;
        void *args[] = { &dY, &dw, &ds, &dX, &N, &K, &M, &scale_cols };
        return hipModuleLaunchKernel(wm >= 2 ? ctx->gemm_fp8_wmma64 : ctx->gemm_fp8_wmma,
            (unsigned int)((N + (wm >= 2 ? 63 : 127)) / (wm >= 2 ? 64 : 128)),
            (unsigned int)((M + (wm >= 2 ? 63 : 127)) / (wm >= 2 ? 64 : 128)), 1,
            256, 1, 1, 0, ctx->stream, args, NULL) == hipSuccess ? 0 : -1;
    }
    void *args[] = { &dY, &dw, &ds, &dX, &lut, &n_out, &n_in, &n_tok, &scale_cols };
    return hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                 ctx->stream, args, NULL) == hipSuccess ? 0 : -1;
}

static int ensure_dev_buf(void **buf, size_t *cap, size_t bytes) {
    if (*buf && *cap >= bytes) return 0;
    void *p = NULL;
    if (hipMalloc(&p, bytes) != hipSuccess) return -1;
    if (*buf) hipFree(*buf);
    *buf = p; *cap = bytes;
    return 0;
}

int hip_ds4f_dense_prefill_qkv(void *opaque, float *q, float *kv, const float *x,
    const ds4f_tensor *wqa, const ds4f_tensor *wkv, const ds4f_tensor *wqb,
    const uint16_t *qnorm, int M, int C, int q_lora, int H, int kv_lora) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (ctx) ctx->qkv_device_ready = 0;
    const hip_ds4f_matrix *ma = NULL, *mk = NULL, *mb = NULL;
    if (!ctx || !q || !kv || !x || !wqa || !wkv || !wqb || !qnorm || M < 1 ||
        wqa->gpu_id < 0 || wkv->gpu_id < 0 || wqb->gpu_id < 0 ||
        matrix_get(ctx, wqa->gpu_id, &ma) != 0 || matrix_get(ctx, wkv->gpu_id, &mk) != 0 ||
        matrix_get(ctx, wqb->gpu_id, &mb) != 0 || !matrix_is_fp8(ma->kind) ||
        !matrix_is_fp8(mk->kind) || !matrix_is_fp8(mb->kind))
        return -1;
    size_t xb=(size_t)M*C*4, lb=(size_t)M*q_lora*4, kb=(size_t)M*kv_lora*4, qb=(size_t)M*H*4;
    if (ensure_dev_buf(&ctx->qkv_x,&ctx->qkv_x_b,xb) || ensure_dev_buf(&ctx->qkv_lat,&ctx->qkv_lat_b,lb) ||
        ensure_dev_buf(&ctx->qkv_norm,&ctx->qkv_norm_b,lb) || ensure_dev_buf(&ctx->qkv_q,&ctx->qkv_q_b,qb) ||
        ensure_dev_buf(&ctx->qkv_kv,&ctx->qkv_kv_b,kb) || ensure_dev_buf(&ctx->qkv_norm_w,&ctx->qkv_norm_w_b,(size_t)q_lora*2)) return -1;
    if (hipMemcpyAsync(ctx->qkv_x,x,xb,hipMemcpyHostToDevice,ctx->stream)!=hipSuccess ||
        hipMemcpyAsync(ctx->qkv_norm_w,qnorm,(size_t)q_lora*2,hipMemcpyHostToDevice,ctx->stream)!=hipSuccess) return -1;
    if (launch_gemm_dev(ctx,ma,0,ctx->qkv_lat,ctx->qkv_x,q_lora,C,M)!=0 ||
        launch_gemm_dev(ctx,mk,0,ctx->qkv_kv,ctx->qkv_x,kv_lora,C,M)!=0) return -1;
    int n=q_lora; float eps=1.0e-6f; void *na[]={&ctx->qkv_norm,&ctx->qkv_lat,&ctx->qkv_norm_w,&M,&n,&eps};
    if (!ctx->rmsnorm_bf16 || hipModuleLaunchKernel(ctx->rmsnorm_bf16,(unsigned)M,1,1,256,1,1,0,ctx->stream,na,NULL)!=hipSuccess) return -1;
    if (launch_gemm_dev(ctx,mb,0,ctx->qkv_q,ctx->qkv_norm,H,q_lora,M)!=0) return -1;
    int q_device_chain = ctx->qkv_device_chain && ctx->attn_device_chain;
    if ((!q_device_chain && hipMemcpyAsync(q,ctx->qkv_q,qb,hipMemcpyDeviceToHost,ctx->stream)!=hipSuccess) ||
        hipMemcpyAsync(kv,ctx->qkv_kv,kb,hipMemcpyDeviceToHost,ctx->stream)!=hipSuccess ||
        hipStreamSynchronize(ctx->stream)!=hipSuccess) return -1;
    ctx->qkv_device_ready = 1;
    ctx->qkv_device_M = M;
    ctx->qkv_device_host = q;
    return 0;
}

/* Fused shared expert: w1/w3 -> SwiGLU -> w2, entirely on the device.
 *
 * The unfused form uploads x twice, downloads two [M, inter] intermediates,
 * uploads their SwiGLU product back, then downloads [M, C].  Here x goes up
 * once and only [M, C] comes back.  Arithmetic is unchanged: the same GEMM
 * kernels run on the same shapes and ds4f_swiglu_inplace mirrors
 * ds4f_pf_swiglu_worker term for term. */
int hip_ds4f_dense_shared_ffn_begin(void *opaque, float *dst,
                              const ds4f_tensor *w1, const ds4f_tensor *w3,
                              const ds4f_tensor *w2, const float *x,
                              int M, int inter, int C, float lim) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    const hip_ds4f_matrix *m1 = NULL, *m3 = NULL, *m2 = NULL;
    if (!ctx || !dst || !x || !w1 || !w3 || !w2 || M < 1 ||
        ctx->pending || ctx->multi_pending ||
        w1->gpu_id < 0 || w3->gpu_id < 0 || w2->gpu_id < 0 ||
        matrix_get(ctx, w1->gpu_id, &m1) != 0 ||
        matrix_get(ctx, w3->gpu_id, &m3) != 0 ||
        matrix_get(ctx, w2->gpu_id, &m2) != 0 ||
        w1->rows != inter || w3->rows != inter || w2->cols != inter ||
        w1->cols != C || w3->cols != C || w2->rows != C)
        return -1;
    size_t xb = (size_t)M * C * sizeof(float);
    size_t ib = (size_t)M * inter * sizeof(float);
    size_t yb = (size_t)M * C * sizeof(float);
    if (hipSetDevice(ctx->device_id) != hipSuccess ||
        ensure_dev_buf(&ctx->ffn_dx, &ctx->ffn_dx_b, xb) != 0 ||
        ensure_dev_buf(&ctx->ffn_dg, &ctx->ffn_dg_b, ib) != 0 ||
        ensure_dev_buf(&ctx->ffn_du, &ctx->ffn_du_b, ib) != 0 ||
        ensure_dev_buf(&ctx->ffn_dy, &ctx->ffn_dy_b, yb) != 0)
        return -1;
    if (ensure_gemm_host_pack(ctx, xb, yb) == 0) {
        memcpy(ctx->gemm_x_pack, x, xb);
        x = ctx->gemm_x_pack;
    }
    if (hipMemcpyAsync(ctx->ffn_dx, x, xb, hipMemcpyHostToDevice,
                       ctx->stream) != hipSuccess)
        return -1;
    if (launch_gemm_dev(ctx, m1, 0, ctx->ffn_dg, ctx->ffn_dx, inter, C, M) != 0 ||
        launch_gemm_dev(ctx, m3, 0, ctx->ffn_du, ctx->ffn_dx, inter, C, M) != 0)
        return -1;
    { int n = M * inter;
      unsigned int grid = (unsigned int)((n + 255) / 256);
      void *dg = ctx->ffn_dg, *du = ctx->ffn_du;
      void *args[] = { &dg, &du, &n, &lim };
      if (hipModuleLaunchKernel(ctx->swiglu, grid, 1, 1, 256, 1, 1, 0,
                                ctx->stream, args, NULL) != hipSuccess)
          return -1; }
    if (launch_gemm_dev(ctx, m2, 0, ctx->ffn_dy, ctx->ffn_dg, C, inter, M) != 0)
        return -1;
    (void)dst;
    return 0;
}

int hip_ds4f_dense_shared_ffn_wait(void *opaque,float *dst,int M,int C) {
    hip_ds4f_dense *ctx=(hip_ds4f_dense *)opaque;
    size_t yb=(size_t)M*C*sizeof(float);
    if(!ctx||!dst||!ctx->ffn_dy)return -1;
    float *out = dst;
    if (ctx->gemm_y_pack && ctx->gemm_y_pack_bytes >= yb) out = ctx->gemm_y_pack;
    if (hipMemcpyAsync(out, ctx->ffn_dy, yb, hipMemcpyDeviceToHost,
                       ctx->stream) != hipSuccess ||
        hipStreamSynchronize(ctx->stream) != hipSuccess)
        return -1;
    if (out != dst) memcpy(dst, out, yb);
    return 0;
}

int hip_ds4f_dense_shared_ffn(void *opaque,float *dst,
                              const ds4f_tensor *w1,const ds4f_tensor *w3,
                              const ds4f_tensor *w2,const float *x,
                              int M,int inter,int C,float lim) {
    if(hip_ds4f_dense_shared_ffn_begin(opaque,dst,w1,w3,w2,x,M,inter,C,lim)!=0)return -1;
    return hip_ds4f_dense_shared_ffn_wait(opaque,dst,M,C);
}

/* Submit a heterogeneous batch of MXFP4 GEMMs as one 3-D launch.  Routed
 * prefill has up to 256 expert buckets with different token counts; launching
 * one kernel per bucket made launch latency dominate the full 43-layer model.
 * The grouped kernel uses blockIdx.z as the task index and bounds-checks each
 * task's token count, so the grid only needs the largest bucket dimensions. */
static int launch_mxfp4_tasks(hip_ds4f_dense *ctx,
                              const hip_ds4f_mxfp4_task *task, int n) {
    unsigned int gx = 0, gy = 0;
    int all_m1 = 1;
    if (!ctx || !task || n < 1 || n > HIP_DS4F_GEMM_MAX ||
        ensure_mxfp4_task_buffer(ctx, n) != 0)
        return -1;
    for (int i = 0; i < n; ++i) {
        unsigned int tx = (unsigned int)((task[i].n_out + 63) / 64);
        unsigned int ty = (unsigned int)((task[i].n_tok + 15) / 16);
        if (tx > gx) gx = tx;
        if (ty > gy) gy = ty;
        if (task[i].n_tok != 1) all_m1 = 0;
    }
    if (hipMemcpyAsync(ctx->gemm_mxfp4_tasks, task,
                       (size_t)n * sizeof(*task), hipMemcpyHostToDevice,
                       ctx->stream) != hipSuccess)
        return -1;
    int ntasks = n;
    void *args[] = { &ctx->gemm_mxfp4_tasks, &ntasks };
    int use_wmma64 = ctx->mxfp4_wmma >= 2 && !all_m1 &&
                       ctx->gemm_mxfp4_grouped_wmma64;
    int use_wmma32 = !use_wmma64 && ctx->mxfp4_wmma >= 2 && !all_m1 && gy >= 2 &&
                       ctx->gemm_mxfp4_grouped_wmma32;
    int use_wmma = !use_wmma32 && ctx->mxfp4_wmma != 0 && !all_m1 &&
                   ctx->gemm_mxfp4_grouped_wmma;
    hipFunction_t fn = use_wmma64 ? ctx->gemm_mxfp4_grouped_wmma64 :
                       (use_wmma32 ? ctx->gemm_mxfp4_grouped_wmma32 :
                       (use_wmma ? ctx->gemm_mxfp4_grouped_wmma :
                       (all_m1 ? ctx->mxfp4_grouped_matvec :
                        ctx->gemm_mxfp4_grouped)));
    unsigned int launch_gx = all_m1
        ? (unsigned int)((task[0].n_out + 7) / 8)
        : (use_wmma64 ? (unsigned int)((task[0].n_out + 63) / 64) :
           (use_wmma32 ? (unsigned int)((task[0].n_out + 31) / 32) :
           (use_wmma ? (unsigned int)((task[0].n_out + 15) / 16) : gx)));
    return hipModuleLaunchKernel(fn, launch_gx,
        use_wmma64 ? (unsigned int)((16 * gy + 63) / 64) :
          ((use_wmma || use_wmma32) ? gy : (all_m1 ? (unsigned int)n : gy)),
        (use_wmma || use_wmma32 || use_wmma64) ? (unsigned int)n : (all_m1 ? 1u : (unsigned int)n),
        use_wmma64 ? 256u : (use_wmma32 ? 64u : (use_wmma ? 32u : (all_m1 ? 256u : 16u))),
        (use_wmma || use_wmma32 || use_wmma64) ? 1u : (all_m1 ? 1u : 16u), 1, 0,
        ctx->stream, args, NULL) == hipSuccess ? 0 : -1;
}

int hip_ds4f_dense_routed_ffn(void *opaque, float *dst, const float *x,
    const ds4f_tensor *const *w1, const ds4f_tensor *const *w3,
    const ds4f_tensor *const *w2, const int *counts, const int *offsets,
    int n_experts, int total, int C, int inter, float lim) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    struct timespec prof_t0;
    clock_gettime(CLOCK_MONOTONIC, &prof_t0);
    if (!ctx || !dst || !x || !w1 || !w3 || !w2 || !counts || !offsets ||
        n_experts < 1 || n_experts > 256 || total < 1 || C < 1 || inter < 1 ||
        offsets[0] != 0 || offsets[n_experts] != total)
        return -1;
    /* Exact mHC/Tier-B2 callers leave routed weights host-resident. Upload
     * only non-empty expert buckets for this invocation; the temporary matrix
     * ids are released after synchronization below. */
    int transient[3][HIP_DS4F_GEMM_MAX];
    memset(transient, -1, sizeof(transient));
    for (int s = 0; s < n_experts; ++s) {
        if (counts[s] <= 0) continue;
        ds4f_tensor *tw[3] = { (ds4f_tensor *)w1[s], (ds4f_tensor *)w3[s],
                               (ds4f_tensor *)w2[s] };
        for (int j = 0; j < 3; ++j) {
            if (!tw[j] || tw[j]->gpu_id >= 0) continue;
            transient[j][s] = hip_ds4f_dense_bind_mxfp4_tensor_async(ctx, tw[j]);
            if (transient[j][s] < 0) goto routed_transient_fail;
        }
    }
    size_t xb = (size_t)total * C * 4, ib = (size_t)total * inter * 4;
    if (hipSetDevice(ctx->device_id) != hipSuccess ||
        ensure_dev_buf(&ctx->ffn_dx, &ctx->ffn_dx_b, xb) != 0 ||
        ensure_dev_buf(&ctx->ffn_dg, &ctx->ffn_dg_b, ib) != 0 ||
        ensure_dev_buf(&ctx->ffn_du, &ctx->ffn_du_b, ib) != 0 ||
        ensure_dev_buf(&ctx->ffn_dy, &ctx->ffn_dy_b, xb) != 0)
        goto routed_transient_fail;
    const float *xh = x;
    if (ensure_gemm_host_pack(ctx, xb, xb) == 0) {
        memcpy(ctx->gemm_x_pack, x, xb);
        xh = ctx->gemm_x_pack;
    }
    if (hipMemcpyAsync(ctx->ffn_dx, xh, xb, hipMemcpyHostToDevice,
                       ctx->stream) != hipSuccess) goto routed_transient_fail;
    hip_ds4f_mxfp4_task gate[HIP_DS4F_GEMM_MAX];
    hip_ds4f_mxfp4_task up[HIP_DS4F_GEMM_MAX];
    hip_ds4f_mxfp4_task down[HIP_DS4F_GEMM_MAX];
    int active = 0;
    const hip_ds4f_matrix *a = NULL, *u = NULL, *d = NULL;
    for (int s = 0; s < n_experts; ++s) {
        int cnt = counts[s];
        if (cnt < 0 || offsets[s] < 0 || offsets[s] + cnt != offsets[s+1]) goto routed_transient_fail;
        if (cnt == 0) continue;
        if (!w1[s] || !w3[s] || !w2[s] || w1[s]->gpu_id < 0 ||
            w3[s]->gpu_id < 0 || w2[s]->gpu_id < 0 ||
            matrix_get(ctx, w1[s]->gpu_id, &a) != 0 ||
            matrix_get(ctx, w3[s]->gpu_id, &u) != 0 ||
            matrix_get(ctx, w2[s]->gpu_id, &d) != 0 ||
            !matrix_is_mxfp4(a->kind) || !matrix_is_mxfp4(u->kind) ||
            !matrix_is_mxfp4(d->kind) || w1[s]->rows != inter ||
            w1[s]->cols != C || w3[s]->rows != inter || w3[s]->cols != C ||
            w2[s]->rows != C || w2[s]->cols != inter)
            goto routed_transient_fail;
        size_t offx = (size_t)offsets[s] * C * 4;
        size_t offi = (size_t)offsets[s] * inter * 4;
        void *dx = (uint8_t *)ctx->ffn_dx + offx;
        void *dg = (uint8_t *)ctx->ffn_dg + offi;
        void *du = (uint8_t *)ctx->ffn_du + offi;
        void *dy = (uint8_t *)ctx->ffn_dy + offx;
        gate[active] = (hip_ds4f_mxfp4_task){ a->dw, a->ds, dx, dg,
                                             inter, C, cnt };
        up[active] = (hip_ds4f_mxfp4_task){ u->dw, u->ds, dx, du,
                                           inter, C, cnt };
        down[active] = (hip_ds4f_mxfp4_task){ d->dw, d->ds, dg, dy,
                                             C, inter, cnt };
        active++;
    }
    if (active < 1 || launch_mxfp4_tasks(ctx, gate, active) != 0 ||
        launch_mxfp4_tasks(ctx, up, active) != 0)
        goto routed_transient_fail;
    int n = total * inter;
    void *dg = ctx->ffn_dg, *du = ctx->ffn_du;
    void *sa[] = { &dg, &du, &n, &lim };
    if (hipModuleLaunchKernel(ctx->swiglu, (unsigned)((n+255)/256), 1, 1,
                              256, 1, 1, 0, ctx->stream, sa, NULL) != hipSuccess ||
        launch_mxfp4_tasks(ctx, down, active) != 0)
        goto routed_transient_fail;
    float *out = dst;
    if (ctx->gemm_y_pack && ctx->gemm_y_pack_bytes >= xb) out = ctx->gemm_y_pack;
    if (hipMemcpyAsync(out, ctx->ffn_dy, xb, hipMemcpyDeviceToHost,
                       ctx->stream) != hipSuccess ||
        hipStreamSynchronize(ctx->stream) != hipSuccess) goto routed_transient_fail;
    if (out != dst) memcpy(dst, out, xb);
    { struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
      ctx->prof_routed_seconds += (double)(t1.tv_sec - prof_t0.tv_sec) +
          1e-9 * (double)(t1.tv_nsec - prof_t0.tv_nsec);
      ctx->prof_routed_calls++; }
    release_transient_experts(ctx, w1, w3, w2, n_experts, transient);
    return 0;

routed_transient_fail:
    release_transient_experts(ctx, w1, w3, w2, n_experts, transient);
    return -1;
}

int hip_ds4f_dense_oproj(void *opaque,float *dst,const ds4f_tensor *wa,
    const ds4f_tensor *wb,const float *x,int M,int groups,int gin,int lora,
    int H,int C,int ointer) {
    hip_ds4f_dense *ctx=(hip_ds4f_dense *)opaque;
    const hip_ds4f_matrix *a=NULL,*b=NULL;
    if(!ctx||!dst||!wa||!wb||!x||M<1||groups*gin!=H||groups*lora!=ointer||
       wa->gpu_id<0||wb->gpu_id<0||matrix_get(ctx,wa->gpu_id,&a)!=0||
       matrix_get(ctx,wb->gpu_id,&b)!=0||a->rows!=ointer||a->cols!=gin||
       b->rows!=C||b->cols!=ointer||!matrix_is_fp8(a->kind)||!matrix_is_fp8(b->kind))return -1;
    size_t dxb=(size_t)M*H*4,xtb=(size_t)M*gin*4,ytb=(size_t)M*lora*4;
    size_t dib=(size_t)M*ointer*4,dyb=(size_t)M*C*4;
    if(hipSetDevice(ctx->device_id)!=hipSuccess||
       ensure_dev_buf(&ctx->op_dx,&ctx->op_dx_b,dxb)||
       ensure_dev_buf(&ctx->op_xt,&ctx->op_xt_b,xtb)||
       ensure_dev_buf(&ctx->op_yt,&ctx->op_yt_b,ytb)||
       ensure_dev_buf(&ctx->op_di,&ctx->op_di_b,dib)||
       ensure_dev_buf(&ctx->op_dy,&ctx->op_dy_b,dyb))return -1;
    int chained = ctx->attn_device_ready && ctx->attn_device_M == M &&
                  ctx->attn_device_host == x;
    if (!chained) {
        const float *xh=x;
        if(ensure_gemm_host_pack(ctx,dxb,dyb)==0){memcpy(ctx->gemm_x_pack,x,dxb);xh=ctx->gemm_x_pack;}
        if(hipMemcpyAsync(ctx->op_dx,xh,dxb,hipMemcpyHostToDevice,ctx->stream)!=hipSuccess)return -1;
    }
    void *op_input = chained ? ctx->attn_y : ctx->op_dx;
    /* Decode stays on the well-tested gather -> per-group GEMM path.  The
     * grouped WMMA variant is reserved for batched prefill; on M=1 its extra
     * task upload and launch do not improve end-to-end latency. */
    for(int g=0;g<groups;g++){
        size_t n=(size_t)M*gin;int off=g*gin;
        void *ga[]={&ctx->op_xt,&op_input,&M,&gin,&H,&off};
        if(hipModuleLaunchKernel(ctx->gather_group,(unsigned)((n+255)/256),1,1,256,1,1,0,ctx->stream,ga,NULL)!=hipSuccess||
           launch_gemm_dev(ctx,a,g*lora,ctx->op_yt,ctx->op_xt,lora,gin,M)!=0)return -1;
        n=(size_t)M*lora;off=g*lora;
        void *sa[]={&ctx->op_di,&ctx->op_yt,&M,&lora,&ointer,&off};
        if(hipModuleLaunchKernel(ctx->scatter_group,(unsigned)((n+255)/256),1,1,256,1,1,0,ctx->stream,sa,NULL)!=hipSuccess)return -1;
    }
    if(launch_gemm_dev(ctx,b,0,ctx->op_dy,ctx->op_di,C,ointer,M)!=0)return -1;
    float *out=dst;if(ctx->gemm_y_pack&&ctx->gemm_y_pack_bytes>=dyb)out=ctx->gemm_y_pack;
    if(hipMemcpyAsync(out,ctx->op_dy,dyb,hipMemcpyDeviceToHost,ctx->stream)!=hipSuccess||
       hipStreamSynchronize(ctx->stream)!=hipSuccess)return -1;
    if(out!=dst)memcpy(dst,out,dyb);
    ctx->attn_device_ready = 0;
    return 0;
}

int hip_ds4f_dense_gemm_tensor(
    void *opaque, float *dst, const ds4f_tensor *t, const float *x,
    int M, int Ystride, int Xstride) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    const hip_ds4f_matrix *mat = NULL;
    int row0 = 0;
    if (!ctx || !dst || !x || !t || M < 1 || Ystride < t->rows ||
        Xstride < t->cols || t->gpu_id < 0 ||
        (t->type != DS4F_FP8 && t->type != DS4F_BF16 && t->type != DS4F_MXFP4) ||
        matrix_get(ctx, t->gpu_id, &mat) != 0 || mat->cols != t->cols ||
        (t->type == DS4F_FP8 && !matrix_is_fp8(mat->kind) &&
         !matrix_is_fp8_promoted(mat->kind)) ||
        (t->type == DS4F_BF16 && mat->kind != HIP_DS4F_MATRIX_BF16) ||
        (t->type == DS4F_MXFP4 && !matrix_is_mxfp4(mat->kind) &&
         !matrix_is_fp8(mat->kind))) {
        fprintf(stderr, "hip_ds4f_dense: invalid batched GEMM arguments\n");
        return -1;
    }
    if (mat->rows != t->rows) {
        if (matrix_is_fp8_promoted(mat->kind)) return -1;
        size_t wbpr = t->type == DS4F_FP8
            ? (size_t)t->cols : (size_t)t->cols * sizeof(uint16_t);
        if (!mat->hw || !t->w || (const uint8_t *)t->w < (const uint8_t *)mat->hw)
            return -1;
        size_t delta = (size_t)((const uint8_t *)t->w - (const uint8_t *)mat->hw);
        if (!wbpr || delta % wbpr != 0) return -1;
        row0 = (int)(delta / wbpr);
        if (row0 < 0 || row0 > mat->rows - t->rows) return -1;
        if (t->type == DS4F_FP8) {
            int sbc = mat->scale_cols;
            if (!mat->hs || row0 % 128 != 0 ||
                t->scale != (const uint8_t *)mat->hs +
                    (size_t)(row0 / 128) * (size_t)sbc)
                return -1;
        }
    }
    if (ctx->pending || ctx->multi_pending ||
        hipSetDevice(ctx->device_id) != hipSuccess)
        return -1;

    int N = t->rows, K = t->cols;
    size_t x_elems = (size_t)M * K, y_elems = (size_t)M * N;
    size_t x_bytes = x_elems * sizeof(float), y_bytes = y_elems * sizeof(float);
    const float *xh = x;
    float *yh = dst;
    if (Xstride != K || Ystride != N) {
        if (ensure_gemm_host_pack(ctx,
                                  Xstride != K ? x_bytes : 0,
                                  Ystride != N ? y_bytes : 0) != 0)
            return -1;
        if (Xstride != K) {
            for (int mm = 0; mm < M; mm++)
                memcpy(ctx->gemm_x_pack + (size_t)mm*K,
                       x + (size_t)mm*Xstride, (size_t)K*sizeof(float));
            xh = ctx->gemm_x_pack;
        }
        if (Ystride != N) yh = ctx->gemm_y_pack;
    }
    /* Large outputs (the M~4096 head is [M, 129280] = 2.1 GB) must not need a
     * full-size device scratch: on the 16 GB RX 9070 XT the resident expert
     * set + prefill buffers leave no room, the gemm fails, and ds4f_gemm()
     * silently falls back to a ~20x slower CPU GEMM (the batch-4096 taper).
     * Tile the N dimension so the scratch stays <= 256 MB when Ystride == N
     * (the head / wq_b layouts). */
    size_t tiled_max = 256u << 20;
    const char *nt = getenv("DS4F_NO_GEMM_TILE");
    int tile = (!nt && Ystride == N && y_bytes > tiled_max) ? 1 : 0;
    int nchunk = 1, chunkN = N;
    if (tile) {
        nchunk = (int)((y_bytes + tiled_max - 1) / tiled_max);
        chunkN = ((N + nchunk - 1) / nchunk + 63) & ~63;
        if (chunkN > N) chunkN = N;
        if (chunkN < 64) chunkN = 64;
        if (ensure_gemm_x(ctx, x_bytes) != 0 ||
            ensure_gemm_y_tile(ctx, (size_t)M * (size_t)chunkN * sizeof(float)) != 0 ||
            ensure_gemm_yh_tile(ctx, (size_t)M * (size_t)chunkN * sizeof(float)) != 0)
            return -1;
    } else if (ensure_gemm_vectors(ctx, x_bytes, y_bytes) != 0) return -1;
    /* Route both directions through the pinned pack buffers even when the
     * strides already match, so the transfers are real async DMA. */
    if (ensure_gemm_host_pack(ctx, x_bytes, y_bytes) == 0) {
        if (xh != ctx->gemm_x_pack) {
            memcpy(ctx->gemm_x_pack, xh, x_bytes);
            xh = ctx->gemm_x_pack;
        }
    }
    if (hipMemcpy(ctx->gemm_dx, xh, x_bytes, hipMemcpyHostToDevice) != hipSuccess)
        return -1;

    int n_in = K, n_tok = M;
    for (int c = 0; c < nchunk; ++c) {
        int c0 = c * chunkN, cn = (c0 + chunkN <= N) ? chunkN : N - c0;
        unsigned int gx = (unsigned int)((cn + 63) / 64);
        unsigned int gy = (unsigned int)((M + 15) / 16);
        int n_out = cn;
        hipError_t err;
        void *dy = tile ? ctx->gemm_dy_tile : ctx->gemm_dy;
        void *dx = ctx->gemm_dx;
        if (matrix_is_mxfp4(mat->kind)) {
            void *dw = (uint8_t *)(void *)mat->dw + (size_t)c0 * (size_t)(K / 2);
            void *ds = (uint8_t *)(void *)mat->ds + (size_t)c0 * (size_t)(K / 32);
            void *args[] = { &dy, &dw, &ds, &dx, &n_out, &n_in, &n_tok };
            err = hipModuleLaunchKernel(ctx->gemm_mxfp4, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_fp8_rowscale(mat->kind)) {
            void *dw = (uint8_t *)(void *)mat->dw + (size_t)row0 * (size_t)K + (size_t)c0 * (size_t)K;
            void *ds = (uint8_t *)(void *)mat->ds + (size_t)row0 * (size_t)mat->scale_cols + (size_t)c0 * (size_t)mat->scale_cols;
            void *lut = ctx->fp8_lut;
            int scale_cols = mat->scale_cols;
            void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
            err = hipModuleLaunchKernel(ctx->gemm_fp8_rowscale, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_fp8_ordered(mat->kind)) {
            void *dw = (uint8_t *)(void *)mat->dw + (size_t)row0 * (size_t)K + (size_t)c0 * (size_t)K;
            void *ds = (uint8_t *)(void *)mat->ds + (size_t)(row0 / 128) * (size_t)mat->scale_cols + (size_t)(c0 / 128) * (size_t)mat->scale_cols;
            void *lut = ctx->fp8_lut;
            int scale_cols = mat->scale_cols;
            void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
            err = hipModuleLaunchKernel(ctx->gemm_fp8_ordered, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_fp8(mat->kind)) {
            void *dw = (uint8_t *)(void *)mat->dw + (size_t)row0 * (size_t)K + (size_t)c0 * (size_t)K;
            void *ds = (uint8_t *)(void *)mat->ds + (size_t)(row0 / 128) * (size_t)mat->scale_cols + (size_t)(c0 / 128) * (size_t)mat->scale_cols;
            int scale_cols = mat->scale_cols;
            if (ctx->fp8_wmma_mode != 0 && M >= 128) {
                int wm = ctx->fp8_wmma_mode;
                void *args[] = { &dy, &dw, &ds, &dx, &n_out, &n_in, &n_tok, &scale_cols };
                err = hipModuleLaunchKernel(wm >= 2 ? ctx->gemm_fp8_wmma64 : ctx->gemm_fp8_wmma,
                    (unsigned int)((n_out + (wm >= 2 ? 63 : 127)) / (wm >= 2 ? 64 : 128)),
                    (unsigned int)((n_tok + (wm >= 2 ? 63 : 127)) / (wm >= 2 ? 64 : 128)), 1,
                    256, 1, 1, 0, ctx->stream, args, NULL);
            } else {
                void *lut = ctx->fp8_lut;
                void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
                err = hipModuleLaunchKernel(ctx->gemm_fp8, gx, gy, 1, 16, 16, 1, 0,
                                            ctx->stream, args, NULL);
            }
        } else if (matrix_is_bf16(mat->kind)) {
            void *dw = (uint8_t *)(void *)mat->dw + (size_t)row0 * (size_t)K * sizeof(uint16_t) + (size_t)c0 * (size_t)K * sizeof(uint16_t);
            if (ctx->bf16_wmma && M >= 128) {
                void *args[] = { &dy, &dw, &dx, &n_out, &n_in, &n_tok };
                err = hipModuleLaunchKernel(ctx->gemm_bf16_wmma,
                    (unsigned int)((n_out + 127) / 128),
                    (unsigned int)((n_tok + 127) / 128), 1,
                    256, 1, 1, 0, ctx->stream, args, NULL);
            } else {
                void *bias = NULL;
                void *args[] = { &dy, &dw, &dx, &bias, &n_out, &n_in, &n_tok };
                err = hipModuleLaunchKernel(ctx->gemm_bf16, gx, gy, 1, 16, 16, 1, 0,
                                            ctx->stream, args, NULL);
            }
        } else {
            void *dw = (uint8_t *)(void *)mat->dw + (size_t)row0 * (size_t)K * sizeof(uint16_t) + (size_t)c0 * (size_t)K * sizeof(uint16_t);
            void *bias = NULL;
            void *args[] = { &dy, &dw, &dx, &bias, &n_out, &n_in, &n_tok };
            err = hipModuleLaunchKernel(ctx->gemm_f16, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        }
        if (err != hipSuccess || hipStreamSynchronize(ctx->stream) != hipSuccess)
            return -1;
        size_t cbytes = (size_t)M * (size_t)cn * sizeof(float);
        float *dstr = (float *)yh + (size_t)c0;
        if (tile) {
            if (hipMemcpy(ctx->gemm_yh_tile, dy, cbytes, hipMemcpyDeviceToHost) != hipSuccess)
                return -1;
            for (int mm = 0; mm < M; ++mm)
                memcpy(dstr + (size_t)mm * N, (float *)ctx->gemm_yh_tile + (size_t)mm * cn,
                       (size_t)cn * sizeof(float));
        } else if (ctx->gemm_y_pack && ctx->gemm_y_pack_bytes >= y_bytes) {
            if (hipMemcpyAsync(ctx->gemm_y_pack, dy, cbytes,
                               hipMemcpyDeviceToHost, ctx->stream) != hipSuccess ||
                hipStreamSynchronize(ctx->stream) != hipSuccess)
                return -1;
            if (yh != ctx->gemm_y_pack) memcpy(yh, ctx->gemm_y_pack, cbytes);
        } else if (hipMemcpy(yh, dy, cbytes, hipMemcpyDeviceToHost) != hipSuccess) return -1;
    }

    if (Ystride != N) for (int mm = 0; mm < M; mm++)
        memcpy(dst + (size_t)mm*Ystride, yh + (size_t)mm*N, (size_t)N*sizeof(float));
    return 0;
}

int hip_ds4f_dense_gemm_tensors(
    void *opaque, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, const int *M, const int *Ystride,
    const int *Xstride, int n) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    const hip_ds4f_matrix *mat[HIP_DS4F_GEMM_MAX] = { NULL };
    size_t ybytes[HIP_DS4F_GEMM_MAX] = { 0 };
    if (!ctx || !dst || !t || !x || !M || !Ystride || !Xstride ||
        n < 1 || n > HIP_DS4F_GEMM_MAX || ctx->pending || ctx->multi_pending)
        return -1;
    int m0 = M[0], k0 = 0;
    if (m0 < 1 || !dst[0] || !x[0] || !t[0] || Xstride[0] < t[0]->cols ||
        t[0]->gpu_id < 0 || matrix_get(ctx, t[0]->gpu_id, &mat[0]) != 0 ||
        mat[0]->rows < t[0]->rows || mat[0]->cols != t[0]->cols ||
        (t[0]->type != DS4F_FP8 && t[0]->type != DS4F_BF16 && t[0]->type != DS4F_MXFP4) ||
        (t[0]->type == DS4F_FP8 && !matrix_is_fp8(mat[0]->kind) &&
         !matrix_is_fp8_promoted(mat[0]->kind)) ||
        (t[0]->type == DS4F_BF16 && mat[0]->kind != HIP_DS4F_MATRIX_BF16) ||
        (t[0]->type == DS4F_MXFP4 && !matrix_is_mxfp4(mat[0]->kind) &&
         !matrix_is_fp8_rowscale(mat[0]->kind)))
        return -1;
    k0 = t[0]->cols;
    for (int i = 0; i < n; ++i) {
        void *view_dw = NULL, *view_ds = NULL;
        if (!dst[i] || !x[i] || !t[i] || M[i] < 1 ||
            t[i]->cols != k0 || Ystride[i] < t[i]->rows ||
            Xstride[i] < t[i]->cols || t[i]->gpu_id < 0 ||
            matrix_get(ctx, t[i]->gpu_id, &mat[i]) != 0 ||
            mat[i]->rows < t[i]->rows || mat[i]->cols != t[i]->cols ||
            (t[i]->type != DS4F_FP8 && t[i]->type != DS4F_BF16 && t[i]->type != DS4F_MXFP4) ||
            (t[i]->type == DS4F_FP8 && !matrix_is_fp8(mat[i]->kind) &&
             !matrix_is_fp8_promoted(mat[i]->kind)) ||
            (t[i]->type == DS4F_BF16 && mat[i]->kind != HIP_DS4F_MATRIX_BF16) ||
            (t[i]->type == DS4F_MXFP4 && !matrix_is_mxfp4(mat[i]->kind) &&
             !matrix_is_fp8_rowscale(mat[i]->kind)) ||
            matrix_view(mat[i], t[i], &view_dw, &view_ds) != 0)
            return -1;
        ybytes[i] = (size_t)M[i] * (size_t)t[i]->rows * sizeof(float);
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    size_t xoff[HIP_DS4F_GEMM_MAX], xbytes = 0;
    for (int i = 0; i < n; ++i) {
        /* ds4f_gemm_pair() submits projections that consume the exact same
         * token matrix (qkv and shared w1/w3).  Pack that input once and let
         * both device tasks read it; the old code duplicated the host copy,
         * device upload, and cache footprint for every pair member. */
        if (i > 0 && M[i] == M[0] && Xstride[i] == Xstride[0] &&
            x[i] == x[0]) {
            xoff[i] = xoff[0];
            continue;
        }
        xoff[i] = xbytes;
        xbytes += (size_t)M[i] * (size_t)k0 * sizeof(float);
    }
    if (ensure_gemm_host_pack(ctx, xbytes, 0) != 0) return -1;
    for (int i = 0; i < n; ++i) {
        if (i > 0 && xoff[i] == xoff[0] && M[i] == M[0] &&
            Xstride[i] == Xstride[0] && x[i] == x[0])
            continue;
        for (int mm = 0; mm < M[i]; ++mm)
            memcpy((uint8_t *)ctx->gemm_x_pack + xoff[i] + (size_t)mm*k0*sizeof(float),
                   x[i] + (size_t)mm*Xstride[i], (size_t)k0*sizeof(float));
    }
    if (ensure_gemm_x(ctx, xbytes) != 0 ||
        ensure_gemm_multi_outputs(ctx, n, ybytes) != 0)
        return -1;
    if (hipMemcpy(ctx->gemm_dx, ctx->gemm_x_pack, xbytes, hipMemcpyHostToDevice) != hipSuccess)
        return -1;
    int grouped_raw = n > 1;
    int grouped_m1 = grouped_raw;
    unsigned int group_gx = 0, group_gy = 0;
    if (grouped_raw) {
        hip_ds4f_mxfp4_task task[HIP_DS4F_GEMM_MAX];
        memset(task, 0, (size_t)n * sizeof(*task));
        for (int i = 0; i < n; ++i) {
            if (!matrix_is_mxfp4(mat[i]->kind)) { grouped_raw = grouped_m1 = 0; break; }
            if (M[i] != 1) grouped_m1 = 0;
            task[i].w = mat[i]->dw;
            task[i].s = mat[i]->ds;
            task[i].x = (uint8_t *)ctx->gemm_dx + xoff[i];
            task[i].y = ctx->gemm_multi_dy[i];
            task[i].n_out = t[i]->rows;
            task[i].n_in = k0;
            task[i].n_tok = M[i];
            unsigned int gx = (unsigned int)((t[i]->rows + 63) / 64);
            unsigned int gy = (unsigned int)((M[i] + 15) / 16);
            if (gx > group_gx) group_gx = gx;
            if (gy > group_gy) group_gy = gy;
        }
        if (grouped_raw) {
            if (ensure_mxfp4_task_buffer(ctx, n) != 0 ||
                hipMemcpy(ctx->gemm_mxfp4_tasks, task,
                          (size_t)n * sizeof(*task), hipMemcpyHostToDevice) != hipSuccess)
                return -1;
            int ntasks = n;
            void *args[] = { &ctx->gemm_mxfp4_tasks, &ntasks };
            int wmode = ctx->mxfp4_wmma;
            int use_wmma32 = wmode >= 2 && !grouped_m1 &&
                           ctx->gemm_mxfp4_grouped_wmma32 && M[0] >= 32;
            int use_wmma = !use_wmma32 && wmode != 0 && !grouped_m1 &&
                           ctx->gemm_mxfp4_grouped_wmma && M[0] >= 32;
            hipFunction_t group_fn = use_wmma32 ? ctx->gemm_mxfp4_grouped_wmma32 : (use_wmma ? ctx->gemm_mxfp4_grouped_wmma :
                (grouped_m1 ? ctx->mxfp4_grouped_matvec : ctx->gemm_mxfp4_grouped));
            unsigned int launch_gx = grouped_m1
                ? (unsigned int)((t[0]->rows + 7) / 8) : group_gx;
            if (use_wmma || use_wmma32) launch_gx = (unsigned int)((t[0]->rows + (use_wmma32 ? 31 : 15)) / (use_wmma32 ? 32 : 16));
            if (hipModuleLaunchKernel(group_fn,
                    launch_gx, (use_wmma || use_wmma32) ? group_gy : (grouped_m1 ? (unsigned int)n : group_gy),
                    (use_wmma || use_wmma32) ? (unsigned int)n : (grouped_m1 ? 1u : (unsigned int)n),
                    use_wmma32 ? 64u : (use_wmma ? 32u : (grouped_m1 ? 256u : 16u)), use_wmma32 ? 1u : (use_wmma ? 1u : (grouped_m1 ? 1u : 16u)), 1, 0,
                    ctx->stream, args, NULL) != hipSuccess)
                return -1;
        }
    }
    if (!grouped_raw) for (int i = 0; i < n; ++i) {
        void *dw = NULL, *ds = NULL;
        if (matrix_view(mat[i], t[i], &dw, &ds) != 0) return -1;
        void *dx = (uint8_t *)ctx->gemm_dx + xoff[i];
        void *dy = ctx->gemm_multi_dy[i];
        hipFunction_t fn;
        unsigned int gx = (unsigned int)((t[i]->rows + 63) / 64);
        unsigned int gy = (unsigned int)((M[i] + 15) / 16);
        hipError_t err;
        if (matrix_is_mxfp4(mat[i]->kind)) {
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            void *args[] = { &dy, &dw, &ds, &dx, &n_out, &n_in, &n_tok };
            fn = ctx->gemm_mxfp4;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_fp8_rowscale(mat[i]->kind)) {
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            int scale_cols = mat[i]->scale_cols;
            void *lut = ctx->fp8_lut;
            void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
            fn = ctx->gemm_fp8_rowscale;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_fp8_ordered(mat[i]->kind)) {
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            int scale_cols = mat[i]->scale_cols;
            void *lut = ctx->fp8_lut;
            void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
            fn = ctx->gemm_fp8_ordered;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_fp8(mat[i]->kind)) {
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            int scale_cols = mat[i]->scale_cols;
            if (ctx->fp8_wmma_mode != 0 && M[i] >= 128) {
                int wm = ctx->fp8_wmma_mode;
                void *args[] = { &dy, &dw, &ds, &dx, &n_out, &n_in, &n_tok, &scale_cols };
                fn = wm >= 2 ? ctx->gemm_fp8_wmma64 : ctx->gemm_fp8_wmma;
                err = hipModuleLaunchKernel(fn,
                    (unsigned int)((n_out + (wm >= 2 ? 63 : 127)) / (wm >= 2 ? 64 : 128)),
                    (unsigned int)((n_tok + (wm >= 2 ? 63 : 127)) / (wm >= 2 ? 64 : 128)), 1,
                    256, 1, 1, 0, ctx->stream, args, NULL);
            } else {
                void *lut = ctx->fp8_lut;
                void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
                fn = ctx->gemm_fp8;
                err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                            ctx->stream, args, NULL);
            }
        } else if (matrix_is_bf16(mat[i]->kind)) {
            void *bias = NULL;
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            void *args[] = { &dy, &dw, &dx, &bias, &n_out, &n_in, &n_tok };
            fn = ctx->gemm_bf16;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else {
            void *bias = NULL;
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            void *args[] = { &dy, &dw, &dx, &bias, &n_out, &n_in, &n_tok };
            fn = ctx->gemm_f16;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        }
        if (err != hipSuccess) return -1;
    }
    if (hipStreamSynchronize(ctx->stream) != hipSuccess) return -1;
    size_t max_ybytes = 0;
    for (int i = 0; i < n; ++i) if (ybytes[i] > max_ybytes) max_ybytes = ybytes[i];
    if (max_ybytes > 0 && ensure_gemm_host_pack(ctx, 0, max_ybytes) != 0) return -1;
    for (int i = 0; i < n; ++i) {
        float *yh = dst[i];
        if (Ystride[i] == t[i]->rows) {
            if (hipMemcpy(yh, ctx->gemm_multi_dy[i], ybytes[i], hipMemcpyDeviceToHost) != hipSuccess)
                return -1;
        } else {
            if (hipMemcpy(ctx->gemm_y_pack, ctx->gemm_multi_dy[i], ybytes[i],
                          hipMemcpyDeviceToHost) != hipSuccess)
                return -1;
            for (int mm = 0; mm < M[i]; ++mm)
                memcpy(yh + (size_t)mm*Ystride[i],
                       ctx->gemm_y_pack + (size_t)mm*t[i]->rows,
                       (size_t)t[i]->rows*sizeof(float));
        }
    }
    return 0;
}

int hip_ds4f_dense_matvec_tensors_async(
    void *opaque, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, int n) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (!ctx || !dst || !t || !x || n < 1 || n > HIP_DS4F_ASYNC_MAX ||
        ctx->pending || ctx->multi_pending) {
        fprintf(stderr, "hip_ds4f_dense: invalid async tensor batch\n");
        return -1;
    }
    const hip_ds4f_matrix *mat[HIP_DS4F_ASYNC_MAX] = { NULL, NULL };
    for (int i = 0; i < n; ++i) {
        if (!dst[i] || !x[i] || !t[i] ||
            (t[i]->type != DS4F_FP8 && t[i]->type != DS4F_BF16) ||
            t[i]->gpu_id < 0 || matrix_get(ctx, t[i]->gpu_id, &mat[i]) != 0 ||
            (t[i]->type == DS4F_FP8 && !matrix_is_fp8(mat[i]->kind) &&
             !matrix_is_fp8_promoted(mat[i]->kind)) ||
            (t[i]->type == DS4F_BF16 && mat[i]->kind != HIP_DS4F_MATRIX_BF16))
            return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    for (int i = 0; i < n; ++i) {
        if (ensure_multi_vectors(ctx, i, mat[i]->rows, mat[i]->cols) != 0)
            goto fail;
        size_t xbi = (size_t)mat[i]->cols * sizeof(float);
        const void *xsrc = x[i];
        if (ensure_pinned(&ctx->multi_hx[i], &ctx->multi_hx_cap[i], xbi) == 0) {
            memcpy(ctx->multi_hx[i], x[i], xbi);
            xsrc = ctx->multi_hx[i];
        }
        if (hipMemcpyAsync(ctx->multi_dx[i], xsrc, xbi,
                           hipMemcpyHostToDevice, ctx->multi_stream[i]) != hipSuccess)
            goto fail;
        void *args[] = { (void *)&mat[i]->dw, (void *)&mat[i]->ds,
                         &ctx->multi_dx[i], &ctx->multi_dy[i],
                         (void *)&mat[i]->rows, (void *)&mat[i]->cols,
                         (void *)&mat[i]->scale_cols };
        hipFunction_t fn = matrix_is_fp16(mat[i]->kind) ? ctx->f16_matvec :
                           matrix_is_bf16(mat[i]->kind) ? ctx->bf16_matvec : ctx->matvec;
        if (hipModuleLaunchKernel(fn,
                (unsigned int)((mat[i]->rows + (ctx->block_threads >> 5) - 1) / (ctx->block_threads >> 5)),
                1, 1, (unsigned int)ctx->block_threads, 1, 1, 0,
                ctx->multi_stream[i], args, NULL) != hipSuccess)
            goto fail;
        if (hipEventRecord(ctx->multi_done[i], ctx->multi_stream[i]) != hipSuccess)
            goto fail;
    }
    for (int i = 0; i < n; ++i) {
        ctx->multi_y[i] = dst[i];
        ctx->multi_out_bytes[i] = (size_t)mat[i]->rows * sizeof(float);
        ctx->multi_n = n;
    }
    ctx->multi_pending = 1;
    return 0;

fail:
    fprintf(stderr, "hip_ds4f_dense: async tensor launch failed\n");
    for (int i = 0; i < n; ++i)
        hipStreamSynchronize(ctx->multi_stream[i]);
    return -1;
}

int hip_ds4f_dense_wait_tensors(void *opaque) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (!ctx || !ctx->multi_pending || ctx->multi_n < 1 ||
        ctx->multi_n > HIP_DS4F_ASYNC_MAX)
        return -1;
    int n = ctx->multi_n;
    /* Queue every download first, then synchronize once per stream, so the
     * copies overlap instead of serializing behind per-tensor event waits. */
    int pinned = 1;
    for (int i = 0; i < n; ++i)
        if (ensure_pinned(&ctx->multi_hy[i], &ctx->multi_hy_cap[i],
                          ctx->multi_out_bytes[i]) != 0) { pinned = 0; break; }
    if (pinned) {
        for (int i = 0; i < n; ++i)
            if (hipMemcpyAsync(ctx->multi_hy[i], ctx->multi_dy[i],
                               ctx->multi_out_bytes[i], hipMemcpyDeviceToHost,
                               ctx->multi_stream[i]) != hipSuccess) {
                fprintf(stderr, "hip_ds4f_dense: async tensor wait/copy failed\n");
                return -1;
            }
        for (int i = 0; i < n; ++i) {
            if (hipStreamSynchronize(ctx->multi_stream[i]) != hipSuccess) {
                fprintf(stderr, "hip_ds4f_dense: async tensor wait/copy failed\n");
                return -1;
            }
            memcpy(ctx->multi_y[i], ctx->multi_hy[i], ctx->multi_out_bytes[i]);
        }
    } else for (int i = 0; i < n; ++i) {
        if (hipEventSynchronize(ctx->multi_done[i]) != hipSuccess ||
            hipMemcpy(ctx->multi_y[i], ctx->multi_dy[i], ctx->multi_out_bytes[i],
                      hipMemcpyDeviceToHost) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: async tensor wait/copy failed\n");
            return -1;
        }
    }
    ctx->multi_pending = 0;
    ctx->multi_n = 0;
    return 0;
}

int hip_ds4f_dense_matvec_loaded_async(hip_ds4f_dense *ctx, const float *x) {
    if (!ctx) return -1;
    return hip_ds4f_dense_matvec_id_async(ctx, ctx->current, x);
}

int hip_ds4f_dense_wait(hip_ds4f_dense *ctx, float *y) {
    if (!ctx || !ctx->pending || !y) {
        fprintf(stderr, "hip_ds4f_dense: no pending matvec or invalid output\n");
        return -1;
    }
    if (ctx->pending_rows <= 0) return -1;
    size_t yb = (size_t)ctx->pending_rows * sizeof(float);
    if (ensure_pinned(&ctx->pin_hy, &ctx->pin_hy_cap, yb) == 0) {
        if (hipMemcpyAsync(ctx->pin_hy, ctx->dy, yb,
                           hipMemcpyDeviceToHost, ctx->stream) != hipSuccess ||
            hipStreamSynchronize(ctx->stream) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: kernel synchronization/copy failed\n");
            return -1;
        }
        memcpy(y, ctx->pin_hy, yb);
    } else if (hipEventSynchronize(ctx->done) != hipSuccess ||
               hipMemcpy(y, ctx->dy, yb, hipMemcpyDeviceToHost) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: kernel synchronization/copy failed\n");
        return -1;
    }
    ctx->pending = 0;
    ctx->pending_rows = 0;
    return 0;
}

int hip_ds4f_dense_matvec_loaded(hip_ds4f_dense *ctx,
                                 const float *x, float *y) {
    if (hip_ds4f_dense_matvec_loaded_async(ctx, x) != 0) return -1;
    return hip_ds4f_dense_wait(ctx, y);
}

int hip_ds4f_dense_matvec(hip_ds4f_dense *ctx,
                          const uint8_t *w, const uint8_t *s,
                          const float *x, float *y,
                          int rows, int cols) {
    if (hip_ds4f_dense_load(ctx, w, s, rows, cols) < 0) return -1;
    return hip_ds4f_dense_matvec_loaded(ctx, x, y);
}
