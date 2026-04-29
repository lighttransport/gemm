/* Composed DINOv2 forward on the device.
 *
 * Glue that strings the Phase 1b kernels together — patch_embed,
 * prepend_cls_reg, add_pos_embed, layernorm_token, gemm_f32_bias,
 * qkv_split, sdpa, layerscale_add, gelu_inplace — into a single
 * forward over the 24 ViT blocks plus the final LayerNorm.
 *
 * Phase 1b.7a (CURRENT): exposes `cs3d_dinov2_block_forward`, a single
 * pre-attn-LN-through-post-MLP-residual block. Standalone-tested by
 * test_dinov2_block.
 *
 * Phase 1b.7b (NEXT): exposes `cs3d_dinov2_gpu_forward`, the
 * image→tokens entrypoint that loops the block forward over all blocks
 * and applies patch_embed/prepend/pos_embed before + final LN after.
 * Wires into the runner in 1b.8.
 */

#ifndef CUDA_SAM3D_DINOV2_FORWARD_H_
#define CUDA_SAM3D_DINOV2_FORWARD_H_

#include "../cuew.h"
#include "../cuda_hip_compat.h"
#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pre-fetched CUfunction handles for one DINOv2 forward pass. Looked up
 * once from a CUmodule so the per-step launches don't pay the cost. */
typedef struct {
    CUfunction layernorm_token_f32;
    CUfunction gemm_f32_bias;
    CUfunction qkv_split_f32;
    CUfunction sdpa_f32;
    CUfunction layerscale_add_f32;
    CUfunction gelu_inplace_f32;
    /* Step kernels — only used by the full forward, not the per-block. */
    CUfunction dinov2_patch_embed_f32;
    CUfunction dinov2_prepend_cls_reg_f32;
    CUfunction dinov2_add_pos_embed_f32;
} cs3d_dinov2_fns;

/* Resolves all kernel handles from `mod`. Returns 0 on success, <0 if
 * any required kernel is missing (any partially-set handles stay set
 * but the struct should not be used). */
static inline int cs3d_dinov2_fns_lookup(cs3d_dinov2_fns *f, CUmodule mod)
{
#define LOOKUP_(name) \
    if (cuModuleGetFunction(&f->name, mod, #name) != CUDA_SUCCESS) { \
        fprintf(stderr, "cs3d_dinov2_fns: lookup %s failed\n", #name); \
        return -1; \
    }
    LOOKUP_(layernorm_token_f32);
    LOOKUP_(gemm_f32_bias);
    LOOKUP_(qkv_split_f32);
    LOOKUP_(sdpa_f32);
    LOOKUP_(layerscale_add_f32);
    LOOKUP_(gelu_inplace_f32);
    LOOKUP_(dinov2_patch_embed_f32);
    LOOKUP_(dinov2_prepend_cls_reg_f32);
    LOOKUP_(dinov2_add_pos_embed_f32);
#undef LOOKUP_
    return 0;
}

/* Workspace required by one block forward. Allocate once per stream;
 * sized for n_tokens × dim × {1 (ln_buf) + 3 (qkv) + 1 (Q) + 1 (K) +
 * 1 (V) + 1 (attn_out) + 1 (proj_out) + ffn/dim (ffn_buf)} F32 elements
 * (the per-row count is dim except ffn_buf which is ffn). */
typedef struct {
    int n_tokens, dim, ffn;
    CUdeviceptr ln_buf;       /* [n_tokens, dim]      */
    CUdeviceptr qkv;          /* [n_tokens, 3*dim]    */
    CUdeviceptr Q, K, V;      /* [n_tokens, dim] each */
    CUdeviceptr attn_out;     /* [n_tokens, dim]      */
    CUdeviceptr proj_out;     /* [n_tokens, dim]      */
    CUdeviceptr ffn_buf;      /* [n_tokens, ffn]      */
} cs3d_dinov2_block_ws;

static inline int cs3d_dinov2_block_ws_alloc(cs3d_dinov2_block_ws *w,
                                             int n_tokens, int dim, int ffn)
{
    w->n_tokens = n_tokens;
    w->dim      = dim;
    w->ffn      = ffn;
    size_t nd = (size_t)n_tokens * dim * sizeof(float);
    size_t nf = (size_t)n_tokens * ffn * sizeof(float);
    if (cuMemAlloc(&w->ln_buf,   nd)   != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->qkv,      3*nd) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->Q,        nd)   != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->K,        nd)   != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->V,        nd)   != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->attn_out, nd)   != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->proj_out, nd)   != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&w->ffn_buf,  nf)   != CUDA_SUCCESS) return -1;
    return 0;
}

static inline void cs3d_dinov2_block_ws_free(cs3d_dinov2_block_ws *w)
{
    if (w->ln_buf)   cuMemFree(w->ln_buf);
    if (w->qkv)      cuMemFree(w->qkv);
    if (w->Q)        cuMemFree(w->Q);
    if (w->K)        cuMemFree(w->K);
    if (w->V)        cuMemFree(w->V);
    if (w->attn_out) cuMemFree(w->attn_out);
    if (w->proj_out) cuMemFree(w->proj_out);
    if (w->ffn_buf)  cuMemFree(w->ffn_buf);
}

/* Per-block weight pointers. Same layout as cs3d_dinov2_gpu_block but
 * decoupled so the block forward can be unit-tested without the loader. */
typedef struct {
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr qkv_w, qkv_b;
    CUdeviceptr proj_w, proj_b;
    CUdeviceptr ls1;
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr fc1_w, fc1_b;
    CUdeviceptr fc2_w, fc2_b;
    CUdeviceptr ls2;
} cs3d_dinov2_block_w;

/* Run one DINOv2 ViT block on `hidden` ([n_tokens, dim]) in-place.
 *
 *   ln_buf   = LN(hidden, ln1)
 *   qkv      = ln_buf @ qkv_w^T + qkv_b           (gemm)
 *   Q,K,V    = split(qkv)
 *   attn_out = SDPA(Q, K, V; n_heads, head_dim, scale=1/sqrt(head_dim))
 *   proj_out = attn_out @ proj_w^T + proj_b       (gemm)
 *   hidden  += proj_out * ls1                     (layerscale_add)
 *   ln_buf   = LN(hidden, ln2)
 *   ffn_buf  = ln_buf @ fc1_w^T + fc1_b           (gemm)
 *   ffn_buf  = gelu_inplace(ffn_buf)
 *   proj_out = ffn_buf @ fc2_w^T + fc2_b          (gemm)
 *   hidden  += proj_out * ls2                     (layerscale_add)
 *
 * Returns 0 on success, <0 on launch failure. Does NOT synchronize —
 * caller drives cuCtxSynchronize / streams as needed.
 */
static inline int cs3d_dinov2_block_forward(
    const cs3d_dinov2_fns *f, cs3d_dinov2_block_ws *ws,
    cs3d_dinov2_block_w *bw, CUdeviceptr hidden,
    int n_tokens, int dim, int ffn, int n_heads, int head_dim, float ln_eps)
{
    int LN_THREADS = 256;
    /* shared mem for LN reduce: 2 * threads */
    unsigned ln_shmem = (unsigned)(2 * LN_THREADS * sizeof(float));
    int affine = 1;

    /* 1. pre-attn LN. */
    {
        void *args[] = { &ws->ln_buf, &hidden, &bw->ln1_w, &bw->ln1_b,
                         &n_tokens, &dim, &ln_eps, &affine };
        if (cuLaunchKernel(f->layernorm_token_f32,
                           n_tokens, 1, 1, LN_THREADS, 1, 1,
                           ln_shmem, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 2. QKV gemm: [n_tokens, dim] @ [3*dim, dim]^T + [3*dim] → [n_tokens, 3*dim]. */
    {
        int Dout = 3 * dim;
        void *args[] = { &ws->qkv, &ws->ln_buf, &bw->qkv_w, &bw->qkv_b,
                         &n_tokens, &dim, &Dout };
        int gx = (n_tokens + 15) / 16, gy = (Dout + 15) / 16;
        if (cuLaunchKernel(f->gemm_f32_bias,
                           gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 3. split fused QKV → Q, K, V. */
    {
        int total = n_tokens * dim;
        int threads = 256, blocks = (total + threads - 1) / threads;
        void *args[] = { &ws->Q, &ws->K, &ws->V, &ws->qkv, &n_tokens, &dim };
        if (cuLaunchKernel(f->qkv_split_f32,
                           blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 4. SDPA. */
    {
        int N_q = n_tokens, N_k = n_tokens;
        int threads = 256;
        unsigned shmem = (unsigned)((threads + N_k) * sizeof(float));
        float scale = 1.0f / sqrtf((float)head_dim);
        void *args[] = { &ws->attn_out, &ws->Q, &ws->K, &ws->V,
                         &N_q, &N_k, &n_heads, &head_dim, &scale };
        if (cuLaunchKernel(f->sdpa_f32,
                           N_q, n_heads, 1, threads, 1, 1,
                           shmem, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 5. attn out_proj gemm: [n_tokens, dim] @ [dim, dim]^T + [dim] → [n_tokens, dim]. */
    {
        int Dout = dim;
        void *args[] = { &ws->proj_out, &ws->attn_out, &bw->proj_w, &bw->proj_b,
                         &n_tokens, &dim, &Dout };
        int gx = (n_tokens + 15) / 16, gy = (Dout + 15) / 16;
        if (cuLaunchKernel(f->gemm_f32_bias,
                           gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 6. ls1 + residual: hidden += proj_out * ls1. */
    {
        int total = n_tokens * dim;
        int threads = 256, blocks = (total + threads - 1) / threads;
        void *args[] = { &hidden, &ws->proj_out, &bw->ls1, &n_tokens, &dim };
        if (cuLaunchKernel(f->layerscale_add_f32,
                           blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 7. pre-MLP LN. */
    {
        void *args[] = { &ws->ln_buf, &hidden, &bw->ln2_w, &bw->ln2_b,
                         &n_tokens, &dim, &ln_eps, &affine };
        if (cuLaunchKernel(f->layernorm_token_f32,
                           n_tokens, 1, 1, LN_THREADS, 1, 1,
                           ln_shmem, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 8. fc1 gemm: [n_tokens, dim] @ [ffn, dim]^T + [ffn] → [n_tokens, ffn]. */
    {
        void *args[] = { &ws->ffn_buf, &ws->ln_buf, &bw->fc1_w, &bw->fc1_b,
                         &n_tokens, &dim, &ffn };
        int gx = (n_tokens + 15) / 16, gy = (ffn + 15) / 16;
        if (cuLaunchKernel(f->gemm_f32_bias,
                           gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 9. exact GELU in place. */
    {
        int total = n_tokens * ffn;
        int threads = 256, blocks = (total + threads - 1) / threads;
        void *args[] = { &ws->ffn_buf, &total };
        if (cuLaunchKernel(f->gelu_inplace_f32,
                           blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 10. fc2 gemm: [n_tokens, ffn] @ [dim, ffn]^T + [dim] → [n_tokens, dim]. */
    {
        int Dout = dim;
        void *args[] = { &ws->proj_out, &ws->ffn_buf, &bw->fc2_w, &bw->fc2_b,
                         &n_tokens, &ffn, &Dout };
        int gx = (n_tokens + 15) / 16, gy = (Dout + 15) / 16;
        if (cuLaunchKernel(f->gemm_f32_bias,
                           gx, gy, 1, 16, 16, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 11. ls2 + residual: hidden += proj_out * ls2. */
    {
        int total = n_tokens * dim;
        int threads = 256, blocks = (total + threads - 1) / threads;
        void *args[] = { &hidden, &ws->proj_out, &bw->ls2, &n_tokens, &dim };
        if (cuLaunchKernel(f->layerscale_add_f32,
                           blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    return 0;
}

/* Full DINOv2 forward on the device.
 *
 *   d_img   [3, image_size, image_size] f32, ImageNet-normalized.
 *   d_out   [n_tokens, dim] f32  — caller-allocated.
 *
 * Sequence: patch_embed → prepend_cls_reg → add_pos_embed → 24 blocks
 * × cs3d_dinov2_block_forward → final LN(norm_w, norm_b).
 *
 * Phase 1b.7b assumes runtime grid == orig_grid (no bicubic resample
 * of pos_embed), matching the CPU port's fast path on 518/14 = 37.
 *
 * Returns 0 on success, <0 on launch failure. Synchronizes once at the
 * end so the result is host-readable when the call returns.
 */
/* Full definition required (struct member access below). Caller must
 * include cuda_sam3d_dinov2_gpu.h before this header. */

static inline int cs3d_dinov2_gpu_forward(
    const cs3d_dinov2_fns *f, cs3d_dinov2_block_ws *ws,
    const cs3d_dinov2_gpu *g, CUdeviceptr d_img, CUdeviceptr d_out)
{
    int n_tok    = g->n_tokens;
    int dim      = g->dim;
    int ffn      = g->ffn_hidden;
    int n_heads  = g->n_heads;
    int head_dim = g->head_dim;
    int ps       = g->patch_size;
    int gw       = g->grid_w;
    int img_w    = g->image_size;
    int n_pat    = g->n_patches;
    int n_reg    = g->n_register;
    float ln_eps = g->ln_eps;
    int LN_THREADS = 256;
    unsigned ln_shmem = (unsigned)(2 * LN_THREADS * sizeof(float));
    int affine = 1;

    /* 1. patch_embed: writes patch tokens into rows [1+n_reg .. 1+n_reg+np). */
    {
        int base_tok = 1 + n_reg;
        int blocks   = n_pat;
        int threads  = 256;
        void *args[] = { &d_out, &d_img, (void *)&g->patch_w, (void *)&g->patch_b,
                         &gw, &dim, &ps, &img_w, &base_tok };
        if (cuLaunchKernel(f->dinov2_patch_embed_f32,
                           blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 2. prepend CLS + register tokens. */
    {
        int blocks = 1 + n_reg;
        int threads = 256;
        void *args[] = { &d_out, (void *)&g->cls_token, (void *)&g->register_tokens,
                         &n_reg, &dim };
        if (cuLaunchKernel(f->dinov2_prepend_cls_reg_f32,
                           blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 3. add pos_embed (CLS + patches; register tokens are skipped). */
    {
        int blocks = 1 + n_pat;
        int threads = 256;
        void *args[] = { &d_out, (void *)&g->pos_embed, &n_reg, &n_pat, &dim };
        if (cuLaunchKernel(f->dinov2_add_pos_embed_f32,
                           blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) return -1;
    }
    /* 4. 24 ViT blocks. */
    for (int L = 0; L < g->n_blocks; L++) {
        cs3d_dinov2_block_w bw;
        bw.ln1_w  = g->blocks[L].norm1_w;
        bw.ln1_b  = g->blocks[L].norm1_b;
        bw.qkv_w  = g->blocks[L].qkv_w;
        bw.qkv_b  = g->blocks[L].qkv_b;
        bw.proj_w = g->blocks[L].proj_w;
        bw.proj_b = g->blocks[L].proj_b;
        bw.ls1    = g->blocks[L].ls1;
        bw.ln2_w  = g->blocks[L].norm2_w;
        bw.ln2_b  = g->blocks[L].norm2_b;
        bw.fc1_w  = g->blocks[L].fc1_w;
        bw.fc1_b  = g->blocks[L].fc1_b;
        bw.fc2_w  = g->blocks[L].fc2_w;
        bw.fc2_b  = g->blocks[L].fc2_b;
        bw.ls2    = g->blocks[L].ls2;
        if (cs3d_dinov2_block_forward(f, ws, &bw, d_out,
                                      n_tok, dim, ffn, n_heads, head_dim, ln_eps) < 0)
            return -1;
    }
    /* 5. final LN — write into ws->ln_buf, then copy back to d_out so the
     *    contract "result is in d_out" holds. */
    if (g->norm_w) {
        void *args[] = { &ws->ln_buf, &d_out, (void *)&g->norm_w, (void *)&g->norm_b,
                         &n_tok, &dim, &ln_eps, &affine };
        if (cuLaunchKernel(f->layernorm_token_f32,
                           n_tok, 1, 1, LN_THREADS, 1, 1,
                           ln_shmem, 0, args, NULL) != CUDA_SUCCESS) return -1;
        if (cuMemcpyDtoD(d_out, ws->ln_buf,
                         (size_t)n_tok * dim * sizeof(float)) != CUDA_SUCCESS) return -1;
    }
    if (cuCtxSynchronize() != CUDA_SUCCESS) return -1;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_SAM3D_DINOV2_FORWARD_H_ */
