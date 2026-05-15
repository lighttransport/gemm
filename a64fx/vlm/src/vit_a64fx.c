/*
 * vit_a64fx.c - M1 baseline a64fx Qwen3-VL vision encoder.
 *
 * Drop-in replacement for vision_encode() that follows the same numeric
 * pipeline (see common/vision_encoder.h:554-1012) but routes hot GEMMs
 * through a64fx/fused-gemm and threads everything through vlm_parallel.
 *
 * M1 scope:
 *   - FP32 only (--dtype bf16/fp16 are wired up but fall back to FP32).
 *   - gemm_fp32 for all GEMMs (8x48 microkernel).
 *   - Scalar exp/tanh for softmax/GELU (M2 swaps to FEXPA / asm).
 *   - Per-stage tensor dump via VLMD writer when enabled.
 *
 * Output layout matches the scalar reference exactly so tensor_diff can
 * compare bit-for-bit (up to fp32 tolerance).
 */
#include "vit_a64fx.h"

#include "vlm_parallel.h"
#include "tensor_dump.h"
#include "cmg_pool.h"

/* These give us vision_model / vision_block / vision_deepstack / qtensor
 * and the dequant_row symbol. We pull declarations only — the writer TU
 * (vlm_runner.c) provides the implementation via *_IMPLEMENTATION macros. */
#include "../../../common/gguf_loader.h"
#include "../../../common/ggml_dequant.h"
#include "../../../common/qtensor_utils.h"   /* qtensor struct (TRANSFORMER_H not set) */
#include "../../../common/vision_encoder.h"

#include "../kernels/fused_gemm.h"
#include "../kernels/norm_sve.h"
#include "../kernels/sve_math.h"
#include "../kernels/bf16_gemm.h"
#include "../kernels/fp16_gemm.h"

#include <arm_sve.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ───────────────────────── small utilities ───────────────────────── */

static float *xcalloc_f(size_t n) {
    float *p = (float *)calloc(n ? n : 1, sizeof(float));
    if (!p) { fprintf(stderr, "vit_a64fx: OOM allocating %zu floats\n", n); exit(1); }
    return p;
}
static float *xmalloc_f(size_t n) {
    float *p = (float *)malloc((n ? n : 1) * sizeof(float));
    if (!p) { fprintf(stderr, "vit_a64fx: OOM allocating %zu floats\n", n); exit(1); }
    return p;
}

/* Dequant a full row tensor (used for [n]-sized bias / norm tensors). */
static void deq_vec(const qtensor *t, float *dst) {
    if (!t || !t->data) return;
    dequant_row(t->type, t->data, dst, t->n_cols);
}

/* Dequant an entire [n_out × n_in] weight matrix into a fresh FP32 buffer
 * laid out row-major n_out × n_in. Returned buffer must be freed. */
static float *deq_W(const qtensor *t) {
    int rows = t->n_rows;
    int cols = t->n_cols;
    float *W = xmalloc_f((size_t)rows * cols);
    /* Mirror vit_dequant_row in vision_encoder.h: each row has fixed
     * byte stride determined by ggml_type. dequant_row works per-row. */
    size_t row_bytes = dequant_row_size(t->type, cols);
    const uint8_t *base = (const uint8_t *)t->data;
    for (int i = 0; i < rows; i++) {
        dequant_row(t->type, base + (size_t)i * row_bytes, W + (size_t)i * cols, cols);
    }
    return W;
}

/* Transpose W[n_out, n_in] → BT[n_in, n_out] (row-major) for gemm_fp32. */
static float *deq_W_T(const qtensor *t) {
    if (!t || !t->data) return NULL;
    int n_out = t->n_rows;
    int n_in  = t->n_cols;
    float *W  = deq_W(t);
    float *BT = xmalloc_f((size_t)n_in * n_out);
    for (int i = 0; i < n_out; i++) {
        const float *src = W + (size_t)i * n_in;
        float *dst_col = BT + i;
        for (int j = 0; j < n_in; j++) dst_col[(size_t)j * n_out] = src[j];
    }
    free(W);
    return BT;
}

static float *deq_vec_xalloc(const qtensor *t, int n) {
    if (!t || !t->data) return NULL;
    float *v = xcalloc_f(n);
    dequant_row(t->type, t->data, v, t->n_cols);
    return v;
}

/* ───────────────────────── weight cache ─────────────────────────
 *
 * One-time-built pre-dequantized + pre-transposed weights so subsequent
 * encode calls skip dequant + transpose + pack. BT layouts match what
 * vit_gemm_bias_mt needs: B is [n_in, n_out] row-major.
 */
/* CMG replicas of a packed BTP buffer. p[c] is the replica pinned to CMG c
 * via cmg_alloc; bytes is the per-replica size for cmg_free. When replication
 * is inactive (single-CMG mode) only p[0] is populated. */
typedef struct {
    uint16_t *p[CMG_MAX];
    size_t    bytes;
} btp_repl;

typedef struct {
    /* attn (FP32 path) */
    float *BT_qkv;   /* [dim, 3*dim]    */
    float *b_qkv;    /* [3*dim]         */
    float *BT_o;     /* [dim, dim]      */
    float *b_o;      /* [dim]           */
    /* ffn */
    float *BT_u;     /* [dim, ffn]      */
    float *b_u;      /* [ffn]           */
    float *BT_d;     /* [ffn, dim]      */
    float *b_d;      /* [dim]           */
    /* norms */
    float *g1, *b1;  /* LN1 */
    float *g2, *b2;  /* LN2 */
    /* BF16 mirrors (non-NULL when cache dtype == BF16) */
    uint16_t *BT_qkv_bf;
    uint16_t *BT_o_bf;
    uint16_t *BT_u_bf;
    uint16_t *BT_d_bf;
    /* FP16 mirrors (non-NULL when cache dtype == FP16) */
    uint16_t *BT_qkv_fp;
    uint16_t *BT_o_fp;
    uint16_t *BT_u_fp;
    uint16_t *BT_d_fp;
    /* CMG replicas (populated by vit_a64fx_cache_replicate). When active,
     * BT_*_fp and BT_*_bf above point at p[0] of the corresponding repl. */
    btp_repl qkv_r, o_r, u_r, d_r;
} block_cache;

typedef struct {
    float *g_norm, *b_norm;       /* [merged_dim] */
    float *BT_fc1, *b_fc1;         /* [merged_dim, merged_dim], [merged_dim] */
    float *BT_fc2, *b_fc2;         /* [merged_dim, proj_dim],   [proj_dim]   */
    uint16_t *BT_fc1_bf;
    uint16_t *BT_fc2_bf;
    uint16_t *BT_fc1_fp;
    uint16_t *BT_fc2_fp;
    btp_repl fc1_r, fc2_r;
} deepstack_cache;

struct vit_a64fx_cache {
    int dtype;                    /* enum vit_dtype: VIT_DTYPE_FP32 or BF16 */
    int n_blocks;
    int n_deepstack;
    int dim, ffn_dim, merged_dim, proj_dim;

    /* patch embed (kept row-major; not used by gemm) */
    float *patch_k0, *patch_k1, *patch_b;
    int    patch_ks;

    /* position embedding (FP32, [orig_n_patches × dim]) */
    float *pos;
    int    pos_n;

    block_cache     *blocks;
    deepstack_cache *deepstack;

    /* post LN */
    float *post_gamma, *post_beta;

    /* MM proj */
    float *BT_mm0,    *b_mm0;
    float *BT_mm2,    *b_mm2;
    uint16_t *BT_mm0_bf;
    uint16_t *BT_mm2_bf;
    uint16_t *BT_mm0_fp;
    uint16_t *BT_mm2_fp;
    btp_repl mm0_r, mm2_r;

    /* CMG replication state. n_cmgs > 0 after vit_a64fx_cache_replicate succeeds;
     * GEMM dispatch then routes per-thread reads to the CMG-local replica. */
    int n_cmgs;
};

/* Pre-pack FP32 BT [K,N] row-major into BTP bf16 form (packed for asm kernel).
 * Returns a buffer of packed_B_bf16_size(K,N) bytes; frees the FP32 source. */
static uint16_t *take_bf16_packed(float **pbt, int K, int N) {
    if (!pbt || !*pbt) return NULL;
    const int NR_ = NR;
    const int N_blocks  = (N + NR_ - 1) / NR_;
    const int K_rounded = ((K + 3) / 4) * 4;
    size_t bytes = (size_t)N_blocks * K_rounded * NR_ * sizeof(uint16_t);
    uint16_t *btp = (uint16_t *)aligned_alloc(64, bytes);
    if (!btp) { fprintf(stderr, "vit_a64fx: OOM packed bf16 alloc (%zu)\n", bytes); exit(1); }
    memset(btp, 0, bytes); /* zero-pad NR-tail and K-tail */
    const float *bt = *pbt;
    for (int nb = 0; nb < N_blocks; nb++) {
        int n_start = nb * NR_;
        int n_count = (n_start + NR_ <= N) ? NR_ : N - n_start;
        uint16_t *dst = btp + (size_t)nb * K_rounded * NR_;
        for (int k = 0; k < K; k++) {
            f32_to_bf16_buf(bt + (size_t)k * N + n_start,
                            dst + (size_t)k * NR_,
                            (size_t)n_count);
        }
    }
    free(*pbt);
    *pbt = NULL;
    return btp;
}

/* Pre-pack FP32 BT [K,N] row-major into BTP fp16 form (packed for asm kernel).
 * Returns a buffer of packed_B_fp16_size(K,N) bytes; frees the FP32 source. */
static uint16_t *take_fp16_packed(float **pbt, int K, int N) {
    if (!pbt || !*pbt) return NULL;
    const int NR_ = NR;
    const int N_blocks  = (N + NR_ - 1) / NR_;
    const int K_rounded = ((K + 3) / 4) * 4;
    size_t bytes = (size_t)N_blocks * K_rounded * NR_ * sizeof(uint16_t);
    uint16_t *btp = (uint16_t *)aligned_alloc(64, bytes);
    if (!btp) { fprintf(stderr, "vit_a64fx: OOM packed fp16 alloc (%zu)\n", bytes); exit(1); }
    memset(btp, 0, bytes); /* zero-pad NR-tail and K-tail */
    const float *bt = *pbt;
    for (int nb = 0; nb < N_blocks; nb++) {
        int n_start = nb * NR_;
        int n_count = (n_start + NR_ <= N) ? NR_ : N - n_start;
        uint16_t *dst = btp + (size_t)nb * K_rounded * NR_;
        for (int k = 0; k < K; k++) {
            f32_to_fp16_buf(bt + (size_t)k * N + n_start,
                            dst + (size_t)k * NR_,
                            (size_t)n_count);
        }
    }
    free(*pbt);
    *pbt = NULL;
    return btp;
}

/* ───────────────────────── CMG replication ─────────────────────────
 *
 * After single-pointer BTP buffers are built (BT_*_fp / BT_*_bf), this step
 * allocates a CMG-pinned copy on every CMG via mmap+mbind and parallel-memcpys
 * the data so each replica is first-touched by its CMG's leader thread.
 *
 * The replica pointers live in `btp_repl.p[CMG_MAX]`. The original single
 * pointer (BT_*_fp / BT_*_bf) is retargeted at p[0] (CMG0 replica) so the
 * existing non-CMG-aware GEMM dispatch path continues to work; the old
 * aligned_alloc buffer is freed.
 *
 * Replication runs across n_cmgs parallel OMP threads, one per CMG, each
 * pinned to its leader core. memcpy on a CMG-pinned thread reading from CMG0
 * (where the source lives) writes into CMG-local pages — cross-CMG read,
 * local write. Source data on CMG0 is shared so it can serve all 4 readers
 * concurrently; total replication cost is ~ (cache_bytes × n_cmgs) / cross-CMG BW.
 */
static void btp_repl_take(btp_repl *r, uint16_t **single, int n_cmgs, size_t bytes) {
    if (!r || !single || !*single || bytes == 0) return;
    r->bytes = bytes;
    const uint16_t *src = *single;
    /* Allocate all replicas first (mbind is per-region, no contention). */
    for (int c = 0; c < n_cmgs; c++) {
        r->p[c] = (uint16_t *)cmg_alloc(c, bytes);
        if (!r->p[c]) {
            fprintf(stderr, "btp_repl_take: cmg_alloc(%d, %zu) failed\n", c, bytes);
            exit(1);
        }
    }
#ifdef _OPENMP
    #pragma omp parallel num_threads(n_cmgs)
#endif
    {
#ifdef _OPENMP
        int my = omp_get_thread_num();
#else
        int my = 0;
#endif
        if (my < n_cmgs) {
            cmg_pin_thread(my, 0);              /* pin to leader core of CMG `my` */
            memcpy(r->p[my], src, bytes);       /* first-touch: pages land on CMG `my` */
        }
    }
    free(*single);
    *single = r->p[0];   /* preserve legacy pointer semantics; cache_free aware */
}

/* Pick the replica for the calling OMP thread's CMG; falls back to p[0]. */
static inline uint16_t *btp_pick(const btp_repl *r) {
    if (!r || r->bytes == 0) return NULL;
    int c = cmg_self();
    if (c < 0) c = 0;
    uint16_t *p = r->p[c];
    return p ? p : r->p[0];
}

/* Pin OMP threads to CMG-aligned cores: tid → CMG (tid/CORES_PER_CMG).
 * Caller should set OMP_NUM_THREADS = n_cmgs * cores_per_cmg before invoking. */
static void cmg_pin_omp_threads(int n_cmgs) {
    int cpc = cmg_cores_per_cmg();
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        int cmg = tid / cpc;
        int loc = tid % cpc;
        if (cmg < n_cmgs && loc < cpc) {
            cmg_pin_thread(cmg, loc);
        }
    }
}

struct vit_a64fx_cache *vit_a64fx_cache_build(struct vision_model *vm, int dtype) {
    if (!vm) return NULL;
    struct vit_a64fx_cache *c =
        (struct vit_a64fx_cache *)calloc(1, sizeof(*c));
    if (!c) return NULL;

    int dim     = vm->dim;
    int ffn_dim = vm->ffn_dim;
    int sm      = vm->spatial_merge;
    int merged  = dim * sm * sm;
    int ps      = vm->patch_size;
    int ks      = ps * ps * 3;

    c->dtype       = dtype;
    c->n_blocks    = vm->n_blocks;
    c->n_deepstack = vm->n_deepstack;
    c->dim         = dim;
    c->ffn_dim     = ffn_dim;
    c->merged_dim  = merged;
    c->proj_dim    = vm->proj_dim;
    c->patch_ks    = ks;

    /* patch embeddings (untransposed, used in patch_body) */
    c->patch_k0 = xmalloc_f((size_t)dim * ks);
    dequant_row(vm->patch_embd_w.type, vm->patch_embd_w.data, c->patch_k0, dim * ks);
    if (vm->patch_embd_w1.data) {
        c->patch_k1 = xmalloc_f((size_t)dim * ks);
        dequant_row(vm->patch_embd_w1.type, vm->patch_embd_w1.data,
                    c->patch_k1, dim * ks);
    }
    c->patch_b = deq_vec_xalloc(&vm->patch_embd_b, dim);

    /* position embed (full original grid) */
    c->pos_n = vm->n_patches;
    c->pos   = xmalloc_f((size_t)dim * c->pos_n);
    dequant_row(vm->position_embd.type, vm->position_embd.data, c->pos,
                dim * c->pos_n);

    c->blocks = (block_cache *)calloc((size_t)vm->n_blocks, sizeof(block_cache));
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int l = 0; l < vm->n_blocks; l++) {
        vision_block *blk = &vm->blocks[l];
        block_cache *bc = &c->blocks[l];
        bc->BT_qkv = deq_W_T(&blk->attn_qkv_w);
        bc->b_qkv  = deq_vec_xalloc(&blk->attn_qkv_b, 3 * dim);
        bc->BT_o   = deq_W_T(&blk->attn_out_w);
        bc->b_o    = deq_vec_xalloc(&blk->attn_out_b, dim);
        bc->BT_u   = deq_W_T(&blk->ffn_up_w);
        bc->b_u    = deq_vec_xalloc(&blk->ffn_up_b, ffn_dim);
        bc->BT_d   = deq_W_T(&blk->ffn_down_w);
        bc->b_d    = deq_vec_xalloc(&blk->ffn_down_b, dim);
        bc->g1     = deq_vec_xalloc(&blk->ln1_w, dim);
        bc->b1     = deq_vec_xalloc(&blk->ln1_b, dim);
        bc->g2     = deq_vec_xalloc(&blk->ln2_w, dim);
        bc->b2     = deq_vec_xalloc(&blk->ln2_b, dim);
    }

    if (vm->n_deepstack > 0) {
        c->deepstack = (deepstack_cache *)calloc((size_t)vm->n_deepstack,
                                                  sizeof(deepstack_cache));
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int d = 0; d < vm->n_deepstack; d++) {
            vision_deepstack *dsl = &vm->deepstack[d];
            deepstack_cache  *dc  = &c->deepstack[d];
            dc->g_norm = deq_vec_xalloc(&dsl->norm_w, merged);
            dc->b_norm = deq_vec_xalloc(&dsl->norm_b, merged);
            dc->BT_fc1 = deq_W_T(&dsl->fc1_w);
            dc->b_fc1  = deq_vec_xalloc(&dsl->fc1_b, merged);
            dc->BT_fc2 = deq_W_T(&dsl->fc2_w);
            dc->b_fc2  = deq_vec_xalloc(&dsl->fc2_b, vm->proj_dim);
        }
    }

    c->post_gamma = deq_vec_xalloc(&vm->post_ln_w, dim);
    c->post_beta  = deq_vec_xalloc(&vm->post_ln_b, dim);

    c->BT_mm0 = deq_W_T(&vm->mm0_w);
    c->b_mm0  = deq_vec_xalloc(&vm->mm0_b, merged);
    c->BT_mm2 = deq_W_T(&vm->mm2_w);
    c->b_mm2  = deq_vec_xalloc(&vm->mm2_b, vm->proj_dim);

    if (dtype == VIT_DTYPE_BF16) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int l = 0; l < vm->n_blocks; l++) {
            block_cache *bc = &c->blocks[l];
            bc->BT_qkv_bf = take_bf16_packed(&bc->BT_qkv, dim,     3 * dim);
            bc->BT_o_bf   = take_bf16_packed(&bc->BT_o,   dim,     dim);
            bc->BT_u_bf   = take_bf16_packed(&bc->BT_u,   dim,     ffn_dim);
            bc->BT_d_bf   = take_bf16_packed(&bc->BT_d,   ffn_dim, dim);
        }
        if (c->deepstack) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int d = 0; d < vm->n_deepstack; d++) {
                deepstack_cache *dc = &c->deepstack[d];
                dc->BT_fc1_bf = take_bf16_packed(&dc->BT_fc1, merged, merged);
                dc->BT_fc2_bf = take_bf16_packed(&dc->BT_fc2, merged, vm->proj_dim);
            }
        }
        c->BT_mm0_bf = take_bf16_packed(&c->BT_mm0, merged, merged);
        c->BT_mm2_bf = take_bf16_packed(&c->BT_mm2, merged, vm->proj_dim);
    } else if (dtype == VIT_DTYPE_FP16) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int l = 0; l < vm->n_blocks; l++) {
            block_cache *bc = &c->blocks[l];
            bc->BT_qkv_fp = take_fp16_packed(&bc->BT_qkv, dim,     3 * dim);
            bc->BT_o_fp   = take_fp16_packed(&bc->BT_o,   dim,     dim);
            bc->BT_u_fp   = take_fp16_packed(&bc->BT_u,   dim,     ffn_dim);
            bc->BT_d_fp   = take_fp16_packed(&bc->BT_d,   ffn_dim, dim);
        }
        if (c->deepstack) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int d = 0; d < vm->n_deepstack; d++) {
                deepstack_cache *dc = &c->deepstack[d];
                dc->BT_fc1_fp = take_fp16_packed(&dc->BT_fc1, merged, merged);
                dc->BT_fc2_fp = take_fp16_packed(&dc->BT_fc2, merged, vm->proj_dim);
            }
        }
        c->BT_mm0_fp = take_fp16_packed(&c->BT_mm0, merged, merged);
        c->BT_mm2_fp = take_fp16_packed(&c->BT_mm2, merged, vm->proj_dim);
    }

    return c;
}

/* Replicate every packed BTP buffer to all n_cmgs CMGs. Each replica is
 * allocated via mmap+mbind on its CMG; data is parallel-memcpy'd by a
 * CMG-pinned leader thread (first-touch on the bound node). The legacy
 * BT_*_fp / BT_*_bf single pointers are retargeted at the CMG0 replica.
 * Returns 0 on success, -1 on no-op (already replicated or dtype FP32). */
int vit_a64fx_cache_replicate(struct vit_a64fx_cache *c, int n_cmgs) {
    if (!c) return -1;
    if (c->n_cmgs > 0) return 0;
    if (n_cmgs <= 0 || n_cmgs > CMG_MAX) n_cmgs = CMG_MAX;
    if (c->dtype != /* VIT_DTYPE_BF16 */ 1 && c->dtype != /* VIT_DTYPE_FP16 */ 2) {
        return -1;
    }

    int dim     = c->dim;
    int ffn_dim = c->ffn_dim;
    int merged  = c->merged_dim;
    int proj    = c->proj_dim;

    size_t qkv_bytes, o_bytes, u_bytes, d_bytes;
    size_t fc1_bytes, fc2_bytes, mm0_bytes, mm2_bytes;
    int is_fp16 = (c->dtype == /* VIT_DTYPE_FP16 */ 2);
    if (is_fp16) {
        qkv_bytes = packed_B_fp16_size(dim,     3 * dim);
        o_bytes   = packed_B_fp16_size(dim,     dim);
        u_bytes   = packed_B_fp16_size(dim,     ffn_dim);
        d_bytes   = packed_B_fp16_size(ffn_dim, dim);
        fc1_bytes = packed_B_fp16_size(merged,  merged);
        fc2_bytes = packed_B_fp16_size(merged,  proj);
        mm0_bytes = fc1_bytes;
        mm2_bytes = fc2_bytes;
    } else {
        qkv_bytes = packed_B_bf16_size(dim,     3 * dim);
        o_bytes   = packed_B_bf16_size(dim,     dim);
        u_bytes   = packed_B_bf16_size(dim,     ffn_dim);
        d_bytes   = packed_B_bf16_size(ffn_dim, dim);
        fc1_bytes = packed_B_bf16_size(merged,  merged);
        fc2_bytes = packed_B_bf16_size(merged,  proj);
        mm0_bytes = fc1_bytes;
        mm2_bytes = fc2_bytes;
    }

    /* Per-block weights. */
    for (int l = 0; l < c->n_blocks; l++) {
        block_cache *bc = &c->blocks[l];
        if (is_fp16) {
            btp_repl_take(&bc->qkv_r, &bc->BT_qkv_fp, n_cmgs, qkv_bytes);
            btp_repl_take(&bc->o_r,   &bc->BT_o_fp,   n_cmgs, o_bytes);
            btp_repl_take(&bc->u_r,   &bc->BT_u_fp,   n_cmgs, u_bytes);
            btp_repl_take(&bc->d_r,   &bc->BT_d_fp,   n_cmgs, d_bytes);
        } else {
            btp_repl_take(&bc->qkv_r, &bc->BT_qkv_bf, n_cmgs, qkv_bytes);
            btp_repl_take(&bc->o_r,   &bc->BT_o_bf,   n_cmgs, o_bytes);
            btp_repl_take(&bc->u_r,   &bc->BT_u_bf,   n_cmgs, u_bytes);
            btp_repl_take(&bc->d_r,   &bc->BT_d_bf,   n_cmgs, d_bytes);
        }
    }

    /* Deepstack. */
    if (c->deepstack) {
        for (int d = 0; d < c->n_deepstack; d++) {
            deepstack_cache *dc = &c->deepstack[d];
            if (is_fp16) {
                btp_repl_take(&dc->fc1_r, &dc->BT_fc1_fp, n_cmgs, fc1_bytes);
                btp_repl_take(&dc->fc2_r, &dc->BT_fc2_fp, n_cmgs, fc2_bytes);
            } else {
                btp_repl_take(&dc->fc1_r, &dc->BT_fc1_bf, n_cmgs, fc1_bytes);
                btp_repl_take(&dc->fc2_r, &dc->BT_fc2_bf, n_cmgs, fc2_bytes);
            }
        }
    }

    /* MM proj. */
    if (is_fp16) {
        btp_repl_take(&c->mm0_r, &c->BT_mm0_fp, n_cmgs, mm0_bytes);
        btp_repl_take(&c->mm2_r, &c->BT_mm2_fp, n_cmgs, mm2_bytes);
    } else {
        btp_repl_take(&c->mm0_r, &c->BT_mm0_bf, n_cmgs, mm0_bytes);
        btp_repl_take(&c->mm2_r, &c->BT_mm2_bf, n_cmgs, mm2_bytes);
    }

    c->n_cmgs = n_cmgs;
    return 0;
}

/* Release a CMG replica set. If r->bytes is 0, this is a no-op (replica
 * inactive); otherwise frees each non-NULL p[c] via cmg_free. */
static void btp_repl_free(btp_repl *r) {
    if (!r || r->bytes == 0) return;
    for (int c = 0; c < CMG_MAX; c++) {
        if (r->p[c]) {
            cmg_free(r->p[c], r->bytes);
            r->p[c] = NULL;
        }
    }
    r->bytes = 0;
}

void vit_a64fx_cache_free(struct vit_a64fx_cache *c) {
    if (!c) return;
    int replicated = (c->n_cmgs > 0);
    free(c->patch_k0); free(c->patch_k1); free(c->patch_b); free(c->pos);
    if (c->blocks) {
        for (int l = 0; l < c->n_blocks; l++) {
            block_cache *b = &c->blocks[l];
            free(b->BT_qkv);    free(b->b_qkv);
            free(b->BT_o);      free(b->b_o);
            free(b->BT_u);      free(b->b_u);
            free(b->BT_d);      free(b->b_d);
            free(b->g1);        free(b->b1);
            free(b->g2);        free(b->b2);
            if (replicated) {
                btp_repl_free(&b->qkv_r);
                btp_repl_free(&b->o_r);
                btp_repl_free(&b->u_r);
                btp_repl_free(&b->d_r);
                /* legacy single pointers alias p[0] (now freed) — don't double-free */
            } else {
                free(b->BT_qkv_bf); free(b->BT_o_bf);
                free(b->BT_u_bf);   free(b->BT_d_bf);
                free(b->BT_qkv_fp); free(b->BT_o_fp);
                free(b->BT_u_fp);   free(b->BT_d_fp);
            }
        }
        free(c->blocks);
    }
    if (c->deepstack) {
        for (int d = 0; d < c->n_deepstack; d++) {
            deepstack_cache *dc = &c->deepstack[d];
            free(dc->g_norm); free(dc->b_norm);
            free(dc->BT_fc1); free(dc->b_fc1);
            free(dc->BT_fc2); free(dc->b_fc2);
            if (replicated) {
                btp_repl_free(&dc->fc1_r);
                btp_repl_free(&dc->fc2_r);
            } else {
                free(dc->BT_fc1_bf); free(dc->BT_fc2_bf);
                free(dc->BT_fc1_fp); free(dc->BT_fc2_fp);
            }
        }
        free(c->deepstack);
    }
    free(c->post_gamma); free(c->post_beta);
    free(c->BT_mm0); free(c->b_mm0);
    free(c->BT_mm2); free(c->b_mm2);
    if (replicated) {
        btp_repl_free(&c->mm0_r);
        btp_repl_free(&c->mm2_r);
    } else {
        free(c->BT_mm0_bf); free(c->BT_mm2_bf);
        free(c->BT_mm0_fp); free(c->BT_mm2_fp);
    }
    free(c);
}

/* ───────────────────────── threaded primitives ───────────────────────── */

/* Bias add: y[t, i] += bias[i]  for t in [t0, t1) */
typedef struct {
    float       *Y;        /* [n_tokens, n_out] */
    const float *bias;     /* [n_out] */
    int          n_out;
} bias_body_args;

static void bias_body(int tid, int t0, int t1, void *arg) {
    (void)tid;
    bias_body_args *a = (bias_body_args *)arg;
    int n = a->n_out;
    for (int t = t0; t < t1; t++) {
        float *y = a->Y + (size_t)t * n;
        for (int i = 0; i < n; i++) y[i] += a->bias[i];
    }
}

static void add_bias_mt(vlm_pool *pool, float *Y, const float *bias,
                        int n_tokens, int n_out) {
    if (!bias) return;
    bias_body_args a = { Y, bias, n_out };
    vlm_parallel_for(pool, n_tokens, 1, bias_body, &a);
}

/* GEMM driver:  Y[n_tokens, n_out] = X[n_tokens, n_in] @ W^T[n_in, n_out] + bias[n_out]
 *
 * Inputs:
 *   X  : [n_tokens, n_in]      row-major
 *   W  : [n_out,    n_in]      row-major  ← weight layout in GGUF
 *   bias: [n_out] or NULL
 *
 * Implementation: compute B = W^T into a transpose buffer once per call (since
 * we need [n_in, n_out] for gemm_fp32). For M1 this transpose is OK because
 * we dequant W from qtensor anyway. M2 keeps a pre-packed B and skips this. */
typedef struct {
    const float *W;        /* [n_out, n_in] row-major */
    float       *BT;       /* [n_in, n_out] row-major (output of transpose) */
    int          n_out;
    int          n_in;
} transpose_args;

static void transpose_body(int tid, int t0, int t1, void *arg) {
    (void)tid;
    transpose_args *a = (transpose_args *)arg;
    int n_out = a->n_out;
    int n_in  = a->n_in;
    for (int i = t0; i < t1; i++) {
        const float *src = a->W + (size_t)i * n_in;
        float *dst_col = a->BT + i; /* column i of BT */
        for (int j = 0; j < n_in; j++) {
            dst_col[(size_t)j * n_out] = src[j];
        }
    }
}

static void vit_gemm_bias_mt(vlm_pool *pool,
                             float *Y,            /* [n_tokens, n_out] */
                             const float *Wfp32,  /* [n_out, n_in] dequant'd */
                             const float *bias,   /* [n_out] or NULL */
                             const float *X,      /* [n_tokens, n_in] */
                             int n_tokens, int n_out, int n_in)
{
    /* Transpose W → BT[n_in, n_out] so gemm_fp32 sees row-major B */
    float *BT = xmalloc_f((size_t)n_in * n_out);
    transpose_args ta = { Wfp32, BT, n_out, n_in };
    vlm_parallel_for(pool, n_out, 8, transpose_body, &ta);

    gemm_fp32(n_tokens, n_in, n_out,
              X,  n_in,
              BT, n_out,
              Y,  n_out);

    free(BT);

    add_bias_mt(pool, Y, bias, n_tokens, n_out);
}

/* Pre-transposed B variant: caller supplies BT[n_in, n_out] directly,
 * so we skip dequant+transpose entirely (M2 weight-cache fast path). */
static void vit_gemm_bias_BT_mt(vlm_pool *pool,
                                float *Y,
                                const float *BT,   /* [n_in, n_out] row-major */
                                const float *bias,
                                const float *X,
                                int n_tokens, int n_out, int n_in)
{
    gemm_fp32(n_tokens, n_in, n_out,
              X,  n_in,
              BT, n_out,
              Y,  n_out);
    add_bias_mt(pool, Y, bias, n_tokens, n_out);
}

/* M3 BF16 path: B is bf16 prepacked into BTP [N_blocks][K_rounded][NR] form.
 * Asm microkernel loads LD1H + LSL #16 → FP32 in registers,
 * accumulates FP32, stores FP32 C. */
static void vit_gemm_bias_BT_bf16_mt(vlm_pool *pool,
                                     float *Y,
                                     const uint16_t *BTP_bf16,
                                     const float    *bias,
                                     const float    *X,
                                     int n_tokens, int n_out, int n_in)
{
    gemm_bf16_BTP(n_tokens, n_in, n_out,
                  X, n_in,
                  BTP_bf16,
                  Y, n_out);
    add_bias_mt(pool, Y, bias, n_tokens, n_out);
}

/* CMG-aware BF16 GEMM. Each OMP thread reads its CMG-local replica via
 * gemm_bf16_BTP_cmg. r->p[c] is the BTP buffer mbind'd to CMG c. */
static void vit_gemm_bias_BT_bf16_cmg_mt(vlm_pool *pool,
                                         float *Y,
                                         const btp_repl *r,
                                         int n_cmgs,
                                         const float    *bias,
                                         const float    *X,
                                         int n_tokens, int n_out, int n_in)
{
    /* btp_repl::p has type uint16_t *p[CMG_MAX]; pass as const-pointer array. */
    gemm_bf16_BTP_cmg(n_tokens, n_in, n_out,
                      X, n_in,
                      (const uint16_t * const *)r->p, n_cmgs,
                      Y, n_out);
    add_bias_mt(pool, Y, bias, n_tokens, n_out);
}

/* Old unpacked-BT C-intrinsic BF16 path; kept available but no longer the default. */
static void vit_gemm_bias_BT_bf16_unpacked_mt(vlm_pool *pool,
                                              float *Y,
                                              const uint16_t *BT_bf16,
                                              const float    *bias,
                                              const float    *X,
                                              int n_tokens, int n_out, int n_in)
{
    gemm_bf16_BT(n_tokens, n_in, n_out,
                 X,       n_in,
                 BT_bf16, n_out,
                 Y,       n_out);
    add_bias_mt(pool, Y, bias, n_tokens, n_out);
}

/* M4 FP16 path: B is fp16 prepacked into BTP [N_blocks][K_rounded][NR] form.
 * Asm microkernel loads LD1H + FCVT Z.S, Pg/M, Z.H → FP32 in registers,
 * accumulates FP32, stores FP32 C. */
static void vit_gemm_bias_BT_fp16_mt(vlm_pool *pool,
                                     float *Y,
                                     const uint16_t *BTP_fp16,
                                     const float    *bias,
                                     const float    *X,
                                     int n_tokens, int n_out, int n_in)
{
    gemm_fp16_BTP(n_tokens, n_in, n_out,
                  X, n_in,
                  BTP_fp16,
                  Y, n_out);
    add_bias_mt(pool, Y, bias, n_tokens, n_out);
}

/* CMG-aware FP16 GEMM. */
static void vit_gemm_bias_BT_fp16_cmg_mt(vlm_pool *pool,
                                         float *Y,
                                         const btp_repl *r,
                                         int n_cmgs,
                                         const float    *bias,
                                         const float    *X,
                                         int n_tokens, int n_out, int n_in)
{
    gemm_fp16_BTP_cmg(n_tokens, n_in, n_out,
                      X, n_in,
                      (const uint16_t * const *)r->p, n_cmgs,
                      Y, n_out);
    add_bias_mt(pool, Y, bias, n_tokens, n_out);
}

/* Old unpacked-BT C-intrinsic path; kept available but no longer the default. */
static void vit_gemm_bias_BT_fp16_unpacked_mt(vlm_pool *pool,
                                              float *Y,
                                              const uint16_t *BT_fp16,
                                              const float    *bias,
                                              const float    *X,
                                              int n_tokens, int n_out, int n_in)
{
    gemm_fp16_BT(n_tokens, n_in, n_out,
                 X,       n_in,
                 BT_fp16, n_out,
                 Y,       n_out);
    add_bias_mt(pool, Y, bias, n_tokens, n_out);
}

/* Dispatch helper: picks FP16 → BF16 → FP32 based on which mirror is set.
 * When `r` is non-NULL and r->p[1] is populated (i.e. cache_replicate has run),
 * route through the CMG-aware GEMM so each OMP thread reads its CMG-local copy. */
static inline void gemm_BT_dispatch(vlm_pool *pool, float *Y,
                                    const float    *BT_f32,
                                    const uint16_t *BT_bf16,
                                    const uint16_t *BT_fp16,
                                    const btp_repl *r,
                                    int n_cmgs,
                                    const float    *bias,
                                    const float    *X,
                                    int n_tokens, int n_out, int n_in)
{
    int replicated = (r && n_cmgs > 1 && r->p[1] != NULL);
    if (BT_fp16) {
        if (replicated) vit_gemm_bias_BT_fp16_cmg_mt(pool, Y, r, n_cmgs, bias, X, n_tokens, n_out, n_in);
        else            vit_gemm_bias_BT_fp16_mt(pool, Y, BT_fp16, bias, X, n_tokens, n_out, n_in);
    } else if (BT_bf16) {
        if (replicated) vit_gemm_bias_BT_bf16_cmg_mt(pool, Y, r, n_cmgs, bias, X, n_tokens, n_out, n_in);
        else            vit_gemm_bias_BT_bf16_mt(pool, Y, BT_bf16, bias, X, n_tokens, n_out, n_in);
    } else {
        vit_gemm_bias_BT_mt(pool, Y, BT_f32, bias, X, n_tokens, n_out, n_in);
    }
}

/* ───────────────────────── per-stage helpers ───────────────────────── */

/* LayerNorm (mean/var/affine).  y = (x - mu)/sqrt(var+eps) * gamma + beta.
 *
 * M2: routes through layernorm_batch_fwd_f32 (norm_sve.c) — 8x-unrolled SVE
 * with FRSQRTE+2NR rsqrt. norm_sve's OpenMP backend uses its own omp parallel
 * for; the vlm_pool argument is unused for LN. */
typedef struct {
    float       *Y;
    const float *X;
    const float *gamma;
    const float *beta;
    int          dim;
    float        eps;
} ln_args;

static void ln_body(int tid, int t0, int t1, void *arg) {
    (void)tid;
    ln_args *a = (ln_args *)arg;
    int dim = a->dim;
    float eps = a->eps;
    for (int t = t0; t < t1; t++) {
        layernorm_fwd_f32(a->X + (size_t)t * dim,
                          a->Y + (size_t)t * dim,
                          a->gamma, a->beta, eps, dim);
    }
}

static void layernorm_mt(vlm_pool *pool,
                         float *Y, const float *X,
                         const float *gamma, const float *beta,
                         int n_tokens, int dim, float eps) {
#ifdef USE_OPENMP
    (void)pool;
    layernorm_batch_fwd_f32(X, Y, gamma, beta, eps, n_tokens, dim);
#else
    /* Route SVE single-row LN through the pool when norm_sve's own omp is off. */
    ln_args a = { Y, X, gamma, beta, dim, eps };
    vlm_parallel_for(pool, n_tokens, 1, ln_body, &a);
#endif
}

/* ── SVE tanh via FEXPA ──
 * tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
 * Clamp |x| ≤ 8 first — tanh saturates and we avoid FEXPA overflow.
 */
static inline svfloat32_t sve_tanh_f32(svbool_t pg, svfloat32_t x,
                                       svfloat32_t vlog2e, svfloat32_t vshift) {
    svfloat32_t hi  = svdup_f32(8.0f);
    svfloat32_t lo  = svdup_f32(-8.0f);
    svfloat32_t xc  = svmax_f32_x(pg, x, lo);
    xc              = svmin_f32_x(pg, xc, hi);
    svfloat32_t two_x = svadd_f32_x(pg, xc, xc);
    svfloat32_t e     = sve_exp_fexpa(pg, two_x, vlog2e, vshift);
    svfloat32_t num   = svsub_n_f32_x(pg, e, 1.0f);
    svfloat32_t den   = svadd_n_f32_x(pg, e, 1.0f);
    return svdiv_f32_x(pg, num, den);
}

/* GELU exact-tanh (matches scalar reference at vision_encoder.h:231):
 *   y = 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3))) */
typedef struct { float *X; size_t total; } gelu_args;

static void gelu_sve_inplace(float *x, int64_t n) {
    const svbool_t pg = svptrue_b32();
    const int VL = (int)svcntw();
    svfloat32_t vlog2e = svdup_f32(LOG2E);
    svfloat32_t vshift = svdup_f32(fexpa_shift_f32());
    svfloat32_t c0     = svdup_f32(0.7978845608f);   /* sqrt(2/pi) */
    svfloat32_t c1     = svdup_f32(0.044715f);
    svfloat32_t half   = svdup_f32(0.5f);

    int64_t i = 0;
    int64_t nv = n - (n % VL);
    for (; i < nv; i += VL) {
        svfloat32_t v  = svld1_f32(pg, x + i);
        svfloat32_t v2 = svmul_f32_x(pg, v, v);
        svfloat32_t v3 = svmul_f32_x(pg, v2, v);
        /* inner = c0 * (v + c1 * v^3)  */
        svfloat32_t inner = svmla_f32_x(pg, v, c1, v3);
        inner = svmul_f32_x(pg, inner, c0);
        svfloat32_t t = sve_tanh_f32(pg, inner, vlog2e, vshift);
        svfloat32_t y = svadd_n_f32_x(pg, t, 1.0f);
        y = svmul_f32_x(pg, y, v);
        y = svmul_f32_x(pg, y, half);
        svst1_f32(pg, x + i, y);
    }
    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32_s64(i, n);
        svfloat32_t v  = svld1_f32(pg_tail, x + i);
        svfloat32_t v2 = svmul_f32_x(pg_tail, v, v);
        svfloat32_t v3 = svmul_f32_x(pg_tail, v2, v);
        svfloat32_t inner = svmla_f32_x(pg_tail, v, c1, v3);
        inner = svmul_f32_x(pg_tail, inner, c0);
        svfloat32_t t = sve_tanh_f32(pg_tail, inner, vlog2e, vshift);
        svfloat32_t y = svadd_n_f32_x(pg_tail, t, 1.0f);
        y = svmul_f32_x(pg_tail, y, v);
        y = svmul_f32_x(pg_tail, y, half);
        svst1_f32(pg_tail, x + i, y);
    }
}

static void gelu_body(int tid, int t0, int t1, void *arg) {
    (void)tid;
    gelu_args *a = (gelu_args *)arg;
    gelu_sve_inplace(a->X + t0, (int64_t)(t1 - t0));
}
static void gelu_mt(vlm_pool *pool, float *X, size_t n) {
    if (n > 0x7fffffff) {
        gelu_sve_inplace(X, (int64_t)n);
        return;
    }
    gelu_args g = { X, n };
    vlm_parallel_for(pool, (int)n, 1024, gelu_body, &g);
}

/* Softmax over n elements in-place. SVE FEXPA path; pass 1 reduces max,
 * pass 2 computes exp(x-max) and accumulates sum, pass 3 normalizes. */
static void softmax_row(float *x, int n) {
    const svbool_t pg = svptrue_b32();
    const int VL = (int)svcntw();
    svfloat32_t vlog2e = svdup_f32(LOG2E);
    svfloat32_t vshift = svdup_f32(fexpa_shift_f32());

    /* Pass 1: max */
    svfloat32_t vmax = svdup_f32(-FLT_MAX);
    int i = 0;
    int nv = n - (n % VL);
    for (; i < nv; i += VL) {
        svfloat32_t v = svld1_f32(pg, x + i);
        vmax = svmax_f32_x(pg, vmax, v);
    }
    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32_s32(i, n);
        svfloat32_t v = svld1_f32(pg_tail, x + i);
        vmax = svmax_f32_m(pg_tail, vmax, v);
    }
    float m = svmaxv_f32(pg, vmax);
    svfloat32_t vmaxb = svdup_f32(m);

    /* Pass 2: exp(x - max), accumulate sum, store back */
    svfloat32_t vsum = svdup_f32(0.0f);
    i = 0;
    for (; i < nv; i += VL) {
        svfloat32_t v = svld1_f32(pg, x + i);
        v = svsub_f32_x(pg, v, vmaxb);
        svfloat32_t e = sve_exp_fexpa(pg, v, vlog2e, vshift);
        vsum = svadd_f32_x(pg, vsum, e);
        svst1_f32(pg, x + i, e);
    }
    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32_s32(i, n);
        svfloat32_t v = svld1_f32(pg_tail, x + i);
        v = svsub_f32_x(pg_tail, v, vmaxb);
        svfloat32_t e = sve_exp_fexpa(pg_tail, v, vlog2e, vshift);
        vsum = svadd_f32_m(pg_tail, vsum, e);
        svst1_f32(pg_tail, x + i, e);
    }
    float s = svaddv_f32(pg, vsum);

    /* Pass 3: scale by 1/sum */
    float inv = 1.0f / s;
    svfloat32_t vinv = svdup_f32(inv);
    i = 0;
    for (; i < nv; i += VL) {
        svfloat32_t v = svld1_f32(pg, x + i);
        v = svmul_f32_x(pg, v, vinv);
        svst1_f32(pg, x + i, v);
    }
    if (i < n) {
        svbool_t pg_tail = svwhilelt_b32_s32(i, n);
        svfloat32_t v = svld1_f32(pg_tail, x + i);
        v = svmul_f32_x(pg_tail, v, vinv);
        svst1_f32(pg_tail, x + i, v);
    }
}

/* ───────────────────────── attention (head-parallel) ───────────────────────── */

typedef struct {
    const float *qkv;       /* [n_patches, 3*dim] */
    float       *attn_out;  /* [n_patches, dim] */
    int          n_patches;
    int          dim;
    int          head_dim;
    int          n_heads;
    float        scale;
    int          q_tile;    /* queries per work unit */
    int          n_qtiles;  /* ceil(n_patches / q_tile) */
    vlm_pool    *pool;
} attn_args;

/* SVE inner product over head_dim elements. With SVE VL=16 (A64FX) and
 * head_dim=64 this is exactly 4 vector loads each. */
static inline float sve_dot_hd(const float *a, const float *b, int hd) {
    const svbool_t pg = svptrue_b32();
    const int VL = (int)svcntw();
    svfloat32_t acc = svdup_f32(0.0f);
    int d = 0;
    int hv = hd - (hd % VL);
    for (; d < hv; d += VL) {
        svfloat32_t va = svld1_f32(pg, a + d);
        svfloat32_t vb = svld1_f32(pg, b + d);
        acc = svmla_f32_x(pg, acc, va, vb);
    }
    if (d < hd) {
        svbool_t pgt = svwhilelt_b32_s32(d, hd);
        svfloat32_t va = svld1_f32(pgt, a + d);
        svfloat32_t vb = svld1_f32(pgt, b + d);
        acc = svmla_f32_m(pgt, acc, va, vb);
    }
    return svaddv_f32(pg, acc);
}

/* SVE AXPY over head_dim:  out[d] += w * vh[d]. */
static inline void sve_axpy_hd(float *out, const float *vh, float w, int hd) {
    const svbool_t pg = svptrue_b32();
    const int VL = (int)svcntw();
    svfloat32_t vw = svdup_f32(w);
    int d = 0;
    int hv = hd - (hd % VL);
    for (; d < hv; d += VL) {
        svfloat32_t vo = svld1_f32(pg, out + d);
        svfloat32_t vv = svld1_f32(pg, vh + d);
        vo = svmla_f32_x(pg, vo, vw, vv);
        svst1_f32(pg, out + d, vo);
    }
    if (d < hd) {
        svbool_t pgt = svwhilelt_b32_s32(d, hd);
        svfloat32_t vo = svld1_f32(pgt, out + d);
        svfloat32_t vv = svld1_f32(pgt, vh + d);
        vo = svmla_f32_m(pgt, vo, vw, vv);
        svst1_f32(pgt, out + d, vo);
    }
}

/* Work unit = one (head, query-tile) pair, flattened as
 * w = h * n_qtiles + qt. Tiling the query dimension lets attention scale
 * past n_heads threads (16) up to n_heads * n_qtiles — needed for 48-core
 * multi-CMG runs where head-only partition leaves 2/3 of cores idle. */
static void attn_body(int tid, int w0, int w1, void *arg) {
    attn_args *a = (attn_args *)arg;
    int np = a->n_patches;
    int dim = a->dim;
    int hd = a->head_dim;
    float scale = a->scale;
    int qt = a->q_tile;
    int nqt = a->n_qtiles;

    /* per-thread scratch: only q_tile rows of the score matrix are live */
    float *att = (float *)vlm_pool_scratch(a->pool, tid,
                                            (size_t)qt * np * sizeof(float));

    for (int w = w0; w < w1; w++) {
        int h  = w / nqt;
        int q0 = (w % nqt) * qt;
        int q1 = q0 + qt; if (q1 > np) q1 = np;

        /* Q · Kᵀ (SVE inner product over head_dim, scaled). */
        for (int qi = q0; qi < q1; qi++) {
            const float *qh = a->qkv + (size_t)qi * 3 * dim + h * hd;
            float *att_row = att + (size_t)(qi - q0) * np;
            for (int ki = 0; ki < np; ki++) {
                const float *kh = a->qkv + (size_t)ki * 3 * dim + dim + h * hd;
                att_row[ki] = sve_dot_hd(qh, kh, hd) * scale;
            }
        }
        /* softmax per row (SVE FEXPA) */
        for (int qi = q0; qi < q1; qi++)
            softmax_row(att + (size_t)(qi - q0) * np, np);
        /* attn · V (SVE AXPY over head_dim per V row) */
        for (int qi = q0; qi < q1; qi++) {
            float *out = a->attn_out + (size_t)qi * dim + h * hd;
            memset(out, 0, hd * sizeof(float));
            const float *att_row = att + (size_t)(qi - q0) * np;
            for (int vi = 0; vi < np; vi++) {
                const float *vh = a->qkv + (size_t)vi * 3 * dim + 2 * dim + h * hd;
                sve_axpy_hd(out, vh, att_row[vi], hd);
            }
        }
    }
}

static void attention_mt(vlm_pool *pool,
                         const float *qkv, float *attn_out,
                         int n_patches, int dim, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    memset(attn_out, 0, (size_t)n_patches * dim * sizeof(float));
    /* Pick a query tile so n_heads * n_qtiles comfortably exceeds the pool
     * size; cap the tile at 64 rows to bound per-thread scratch. */
    int nthr = vlm_pool_size(pool);
    int q_tile = n_patches;
    if (n_heads < nthr) {
        int want = (nthr + n_heads - 1) / n_heads;          /* tiles needed */
        q_tile = (n_patches + want - 1) / want;
        if (q_tile > 64) q_tile = 64;
        if (q_tile < 1)  q_tile = 1;
    }
    int n_qtiles = (n_patches + q_tile - 1) / q_tile;
    attn_args a = { qkv, attn_out, n_patches, dim, head_dim, n_heads, scale,
                    q_tile, n_qtiles, pool };
    vlm_parallel_for(pool, n_heads * n_qtiles, 1, attn_body, &a);
}

/* ───────────────────────── patch embedding ───────────────────────── */

typedef struct {
    const float *rgb_norm;
    int          width;
    int          ps;
    int          gw;
    int          dim;
    int          kernel_size;
    const float *kernel0;    /* [dim, ks] */
    const float *kernel1;    /* may be NULL */
    const float *bias;       /* may be NULL */
    float       *out;        /* [n_patches, dim] */
} patch_args;

static void patch_body(int tid, int t0, int t1, void *arg) {
    (void)tid;
    patch_args *a = (patch_args *)arg;
    int ps = a->ps, gw = a->gw, dim = a->dim, ks = a->kernel_size, W = a->width;

    float *patch = (float *)alloca((size_t)ks * sizeof(float));
    for (int p = t0; p < t1; p++) {
        int py = p / gw;
        int px = p % gw;
        /* gather patch pixels in CHW order, matching scalar ref */
        for (int c = 0; c < 3; c++) {
            for (int dy = 0; dy < ps; dy++) {
                for (int dx = 0; dx < ps; dx++) {
                    int iy = py * ps + dy;
                    int ix = px * ps + dx;
                    patch[c * ps * ps + dy * ps + dx] =
                        a->rgb_norm[(iy * W + ix) * 3 + c];
                }
            }
        }
        float *out = a->out + (size_t)p * dim;
        for (int d = 0; d < dim; d++) {
            float sum = 0.0f;
            const float *k0 = a->kernel0 + (size_t)d * ks;
            for (int j = 0; j < ks; j++) sum += k0[j] * patch[j];
            if (a->kernel1) {
                const float *k1 = a->kernel1 + (size_t)d * ks;
                for (int j = 0; j < ks; j++) sum += k1[j] * patch[j];
            }
            out[d] = sum + (a->bias ? a->bias[d] : 0.0f);
        }
    }
}

static void patch_embed_mt(vlm_pool *pool,
                           const float *rgb_norm, int width,
                           int gw, int gh, int ps, int dim,
                           const float *kernel0, const float *kernel1,
                           const float *bias,
                           float *out) {
    int n_patches = gw * gh;
    int ks = ps * ps * 3;
    patch_args a = { rgb_norm, width, ps, gw, dim, ks, kernel0, kernel1, bias, out };
    vlm_parallel_for(pool, n_patches, 1, patch_body, &a);
}

/* ───────────────────────── M-RoPE ───────────────────────── */

static void mrope_apply(float *qkv, int n_patches, int n_heads, int head_dim,
                        int dim, const float *rope_cos, const float *rope_sin)
{
    int half = head_dim / 2;
    for (int p = 0; p < n_patches; p++) {
        float *q = qkv + (size_t)p * 3 * dim;
        float *k = qkv + (size_t)p * 3 * dim + dim;
        for (int h = 0; h < n_heads; h++) {
            float *qh = q + h * head_dim;
            float *kh = k + h * head_dim;
            for (int i = 0; i < half; i++) {
                float ct = rope_cos[p * head_dim + 2 * i];
                float st = rope_sin[p * head_dim + 2 * i];
                float q0 = qh[i], q1 = qh[i + half];
                qh[i]        = q0 * ct - q1 * st;
                qh[i + half] = q0 * st + q1 * ct;
                float k0 = kh[i], k1 = kh[i + half];
                kh[i]        = k0 * ct - k1 * st;
                kh[i + half] = k0 * st + k1 * ct;
            }
        }
    }
}

static void mrope_build_cache(int n_patches, int gw, int gh,
                              int head_dim, float *rope_cos, float *rope_sin)
{
    int half = head_dim / 2;
    int sect = head_dim / 4;
    float base = 10000.0f;
    float scale = powf(base, -2.0f / (float)half);
    for (int p = 0; p < n_patches; p++) {
        int py = p / gw;
        int px = p % gw;
        float p_t = (float)py;
        float p_h = (float)px;
        float p_w = (float)py;
        float p_e = (float)px;
        float cur_t = p_t, cur_h = p_h, cur_w = p_w, cur_e = p_e;
        for (int i0 = 0; i0 < head_dim; i0 += 2) {
            int sector = i0 / 2;
            if (sector == 0)         cur_t = p_t;
            if (sector == sect)      cur_h = p_h;
            if (sector == 2 * sect)  cur_w = p_w;
            if (sector == 3 * sect)  cur_e = p_e;
            float theta;
            if      (sector < sect)       theta = cur_t;
            else if (sector < 2 * sect)   theta = cur_h;
            else if (sector < 3 * sect)   theta = cur_w;
            else                          theta = cur_e;
            float c = cosf(theta), s = sinf(theta);
            rope_cos[p * head_dim + i0]     = c;
            rope_sin[p * head_dim + i0]     = s;
            rope_cos[p * head_dim + i0 + 1] = c;
            rope_sin[p * head_dim + i0 + 1] = s;
            cur_t *= scale; cur_h *= scale; cur_w *= scale; cur_e *= scale;
            (void)gh;
        }
    }
}

/* ───────────────────────── tensor dump shorthand ───────────────────────── */
static inline void dump2(struct vlmd_writer *w, const char *name, int layer,
                         int rows, int cols, const float *data) {
    if (w) vlmd_dump_f32_2d(w, name, layer, rows, cols, data);
}

/* ───────────────────────── stage timing ─────────────────────────
 * Enable with VLM_STAGE_TIMING=1 in the env. Times accumulate across
 * blocks and print as a breakdown at end of encode.
 */
#include <time.h>
typedef enum {
    ST_PATCH = 0, ST_POS, ST_LN, ST_QKV, ST_ROPE, ST_ATTN, ST_AOUT,
    ST_FFN_UP, ST_GELU, ST_FFN_DN, ST_DS, ST_POST, ST_MM, ST_OTHER,
    ST__COUNT
} stage_id;
static const char *st_name[ST__COUNT] = {
    "patch_embed","pos_emb","layernorm","qkv","mrope","attn","attn_out",
    "ffn_up","gelu","ffn_down","deepstack","post_ln","mm_proj","other"
};

typedef struct {
    int    enabled;
    double t[ST__COUNT];
    double last;
} stage_timer;

static double mono_sec_local(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void st_init(stage_timer *t) {
    const char *e = getenv("VLM_STAGE_TIMING");
    t->enabled = (e && *e && *e != '0');
    for (int i = 0; i < ST__COUNT; i++) t->t[i] = 0.0;
    t->last = t->enabled ? mono_sec_local() : 0.0;
}
static inline void st_tick(stage_timer *t, stage_id id) {
    if (!t->enabled) return;
    double now = mono_sec_local();
    t->t[id] += now - t->last;
    t->last = now;
}
static void st_print(const stage_timer *t) {
    if (!t->enabled) return;
    double total = 0.0;
    for (int i = 0; i < ST__COUNT; i++) total += t->t[i];
    fprintf(stderr, "stage timings (total %.3f s):\n", total);
    for (int i = 0; i < ST__COUNT; i++) {
        if (t->t[i] < 1e-6) continue;
        fprintf(stderr, "  %-10s %.3f s  (%.1f%%)\n",
                st_name[i], t->t[i], 100.0 * t->t[i] / total);
    }
}

/* ───────────────────────── main encode ───────────────────────── */

float *vit_a64fx_encode(struct vision_model *vm,
                        const float *rgb_norm,
                        int width, int height,
                        const vit_a64fx_opts *opts,
                        int *out_n_merged, int *out_embd)
{
    if (!vm || !rgb_norm || !opts || !opts->pool) return NULL;
    vlm_pool *pool = opts->pool;
    struct vlmd_writer *dump = opts->dump;
    struct vit_a64fx_cache *cache = opts->cache;

    int ps       = vm->patch_size;
    int dim      = vm->dim;
    int n_heads  = vm->n_heads;
    int head_dim = vm->head_dim;
    int ffn_dim  = vm->ffn_dim;
    int sm       = vm->spatial_merge;
    int gw       = width / ps;
    int gh       = height / ps;
    int n_patches = gw * gh;
    int merged_dim = dim * sm * sm;
    int n_merged  = n_patches / (sm * sm);
    int n_ds      = vm->n_deepstack;
    int total_embd = vm->proj_dim * (1 + n_ds);

    if (n_patches > vm->n_patches) {
        fprintf(stderr, "vit_a64fx: too many patches %d (max %d)\n",
                n_patches, vm->n_patches);
        return NULL;
    }

    const char *storage = "fp32";
    if (cache) {
        if (cache->dtype == VIT_DTYPE_BF16)      storage = "bf16";
        else if (cache->dtype == VIT_DTYPE_FP16) { storage = "fp16"; set_fpcr_fz16(); }
    }
    /* If the cache has been NUMA-replicated, pin OMP workers to CMG-aligned
     * cores so the linear tid → CMG mapping used inside the CMG GEMM kernels
     * lines up with actual hardware affinity. */
    if (cache && cache->n_cmgs > 1) {
        cmg_pin_omp_threads(cache->n_cmgs);
    }
    fprintf(stderr, "vit_a64fx: storage=%s image=%dx%d patches=%d merged=%d threads=%d cmgs=%d\n",
            storage, width, height, n_patches, n_merged, vlm_pool_size(pool),
            cache ? cache->n_cmgs : 0);

    stage_timer st; st_init(&st);

    /* Allocate stage buffers */
    float *hidden    = xcalloc_f((size_t)n_patches * dim);
    float *hidden2   = xcalloc_f((size_t)n_patches * dim);
    float *qkv       = xcalloc_f((size_t)n_patches * 3 * dim);
    float *attn_out  = xcalloc_f((size_t)n_patches * dim);
    float *ffn_buf   = xcalloc_f((size_t)n_patches * ffn_dim);
    float *ln_buf    = xcalloc_f((size_t)n_patches * dim);
    float *merge_buf = xcalloc_f((size_t)n_merged * merged_dim);
    float *deepstack_feats = xcalloc_f((size_t)n_merged * (n_ds ? n_ds : 1) * vm->proj_dim);

    /* ── 1. patch_embed ── */
    int ks = ps * ps * 3;
    float *kernel0_tmp = NULL, *kernel1_tmp = NULL, *patch_bias_tmp = NULL;
    const float *kernel0 = NULL, *kernel1 = NULL, *patch_bias = NULL;
    if (cache) {
        kernel0    = cache->patch_k0;
        kernel1    = cache->patch_k1;
        patch_bias = cache->patch_b;
    } else {
        kernel0_tmp = xmalloc_f((size_t)dim * ks);
        dequant_row(vm->patch_embd_w.type, vm->patch_embd_w.data, kernel0_tmp, dim * ks);
        kernel0 = kernel0_tmp;
        if (vm->patch_embd_w1.data) {
            kernel1_tmp = xmalloc_f((size_t)dim * ks);
            dequant_row(vm->patch_embd_w1.type, vm->patch_embd_w1.data,
                        kernel1_tmp, dim * ks);
            kernel1 = kernel1_tmp;
        }
        if (vm->patch_embd_b.data) {
            patch_bias_tmp = xcalloc_f(dim);
            deq_vec(&vm->patch_embd_b, patch_bias_tmp);
            patch_bias = patch_bias_tmp;
        }
    }
    patch_embed_mt(pool, rgb_norm, width, gw, gh, ps, dim, kernel0, kernel1, patch_bias, hidden);
    free(kernel0_tmp); free(kernel1_tmp); free(patch_bias_tmp);
    st_tick(&st, ST_PATCH);
    dump2(dump, "patch_embed", -1, n_patches, dim, hidden);

    /* ── 2. position embeddings ── */
    {
        int orig_n = vm->n_patches;
        int orig_gw = vm->image_size / ps;
        const float *pos = NULL;
        float *pos_tmp = NULL;
        if (cache) {
            pos = cache->pos;
        } else {
            pos_tmp = xmalloc_f((size_t)dim * orig_n);
            dequant_row(vm->position_embd.type, vm->position_embd.data, pos_tmp,
                        dim * orig_n);
            pos = pos_tmp;
        }
        for (int py = 0; py < gh; py++) {
            for (int px = 0; px < gw; px++) {
                int p = py * gw + px;
                int op = py * orig_gw + px;
                for (int d = 0; d < dim; d++) hidden[p * dim + d] += pos[op * dim + d];
            }
        }
        free(pos_tmp);
    }
    st_tick(&st, ST_POS);
    dump2(dump, "pos_emb_added", -1, n_patches, dim, hidden);

    /* ── 2b. M-RoPE cache ── */
    float *rope_cos = xmalloc_f((size_t)n_patches * head_dim);
    float *rope_sin = xmalloc_f((size_t)n_patches * head_dim);
    mrope_build_cache(n_patches, gw, gh, head_dim, rope_cos, rope_sin);
    st_tick(&st, ST_ROPE);

    /* ── 3. ViT blocks ── */
    int ds_count = 0;
    for (int l = 0; l < vm->n_blocks; l++) {
        vision_block *blk = &vm->blocks[l];
        block_cache *bc = cache ? &cache->blocks[l] : NULL;

        /* LN1 */
        if (bc) {
            layernorm_mt(pool, ln_buf, hidden, bc->g1, bc->b1,
                         n_patches, dim, vm->ln_eps);
        } else {
            float *g1 = xcalloc_f(dim);  deq_vec(&blk->ln1_w, g1);
            float *b1 = xcalloc_f(dim);  deq_vec(&blk->ln1_b, b1);
            layernorm_mt(pool, ln_buf, hidden, g1, b1, n_patches, dim, vm->ln_eps);
            free(g1); free(b1);
        }
        st_tick(&st, ST_LN);
        dump2(dump, "ln1", l, n_patches, dim, ln_buf);

        /* QKV proj */
        if (bc) {
            gemm_BT_dispatch(pool, qkv, bc->BT_qkv, bc->BT_qkv_bf, bc->BT_qkv_fp,
                             &bc->qkv_r, cache->n_cmgs, bc->b_qkv,
                             ln_buf, n_patches, 3 * dim, dim);
        } else {
            float *Wqkv = deq_W(&blk->attn_qkv_w);
            float *bqkv = xcalloc_f(3 * dim);  deq_vec(&blk->attn_qkv_b, bqkv);
            vit_gemm_bias_mt(pool, qkv, Wqkv, bqkv, ln_buf, n_patches, 3 * dim, dim);
            free(Wqkv); free(bqkv);
        }
        st_tick(&st, ST_QKV);
        dump2(dump, "qkv", l, n_patches, 3 * dim, qkv);

        /* M-RoPE on Q, K (not V) */
        mrope_apply(qkv, n_patches, n_heads, head_dim, dim, rope_cos, rope_sin);
        st_tick(&st, ST_ROPE);
        dump2(dump, "mrope", l, n_patches, 3 * dim, qkv);

        /* Attention */
        attention_mt(pool, qkv, attn_out, n_patches, dim, n_heads, head_dim);
        st_tick(&st, ST_ATTN);
        dump2(dump, "attn", l, n_patches, dim, attn_out);

        /* Attn out proj */
        if (bc) {
            gemm_BT_dispatch(pool, hidden2, bc->BT_o, bc->BT_o_bf, bc->BT_o_fp,
                             &bc->o_r, cache->n_cmgs, bc->b_o,
                             attn_out, n_patches, dim, dim);
        } else {
            float *Wo = deq_W(&blk->attn_out_w);
            float *bo = xcalloc_f(dim);  deq_vec(&blk->attn_out_b, bo);
            vit_gemm_bias_mt(pool, hidden2, Wo, bo, attn_out, n_patches, dim, dim);
            free(Wo); free(bo);
        }
        st_tick(&st, ST_AOUT);
        dump2(dump, "attn_out", l, n_patches, dim, hidden2);

        /* Residual */
        for (size_t i = 0; i < (size_t)n_patches * dim; i++) hidden[i] += hidden2[i];

        /* LN2 */
        if (bc) {
            layernorm_mt(pool, ln_buf, hidden, bc->g2, bc->b2,
                         n_patches, dim, vm->ln_eps);
        } else {
            float *g2 = xcalloc_f(dim);  deq_vec(&blk->ln2_w, g2);
            float *b2 = xcalloc_f(dim);  deq_vec(&blk->ln2_b, b2);
            layernorm_mt(pool, ln_buf, hidden, g2, b2, n_patches, dim, vm->ln_eps);
            free(g2); free(b2);
        }
        st_tick(&st, ST_LN);
        dump2(dump, "ln2", l, n_patches, dim, ln_buf);

        /* FFN up */
        if (bc) {
            gemm_BT_dispatch(pool, ffn_buf, bc->BT_u, bc->BT_u_bf, bc->BT_u_fp,
                             &bc->u_r, cache->n_cmgs, bc->b_u,
                             ln_buf, n_patches, ffn_dim, dim);
        } else {
            float *Wu = deq_W(&blk->ffn_up_w);
            float *bu = xcalloc_f(ffn_dim);  deq_vec(&blk->ffn_up_b, bu);
            vit_gemm_bias_mt(pool, ffn_buf, Wu, bu, ln_buf, n_patches, ffn_dim, dim);
            free(Wu); free(bu);
        }
        st_tick(&st, ST_FFN_UP);
        dump2(dump, "ffn_up", l, n_patches, ffn_dim, ffn_buf);

        /* GELU */
        gelu_mt(pool, ffn_buf, (size_t)n_patches * ffn_dim);
        st_tick(&st, ST_GELU);
        dump2(dump, "gelu", l, n_patches, ffn_dim, ffn_buf);

        /* FFN down */
        if (bc) {
            gemm_BT_dispatch(pool, hidden2, bc->BT_d, bc->BT_d_bf, bc->BT_d_fp,
                             &bc->d_r, cache->n_cmgs, bc->b_d,
                             ffn_buf, n_patches, dim, ffn_dim);
        } else {
            float *Wd = deq_W(&blk->ffn_down_w);
            float *bd = xcalloc_f(dim);  deq_vec(&blk->ffn_down_b, bd);
            vit_gemm_bias_mt(pool, hidden2, Wd, bd, ffn_buf, n_patches, dim, ffn_dim);
            free(Wd); free(bd);
        }
        st_tick(&st, ST_FFN_DN);
        dump2(dump, "ffn_down", l, n_patches, dim, hidden2);

        /* Residual */
        for (size_t i = 0; i < (size_t)n_patches * dim; i++) hidden[i] += hidden2[i];
        dump2(dump, "block_out", l, n_patches, dim, hidden);

        /* DeepStack (if this layer is a deepstack source) */
        for (int ds = 0; ds < vm->n_deepstack; ds++) {
            if (vm->deepstack_indices[ds] != l) continue;
            vision_deepstack *dsl = &vm->deepstack[ds];
            deepstack_cache *dc = cache ? &cache->deepstack[ds] : NULL;

            int mgw = gw / sm, mgh = gh / sm;
            /* spatial merge into merge_buf */
            for (int my = 0; my < mgh; my++) {
                for (int mx = 0; mx < mgw; mx++) {
                    float *dst = merge_buf + (size_t)(my * mgw + mx) * merged_dim;
                    int di = 0;
                    for (int sy = 0; sy < sm; sy++) {
                        for (int sx = 0; sx < sm; sx++) {
                            int py2 = my * sm + sy;
                            int px2 = mx * sm + sx;
                            const float *src = hidden + (size_t)(py2 * gw + px2) * dim;
                            memcpy(dst + di, src, dim * sizeof(float));
                            di += dim;
                        }
                    }
                }
            }

            /* DS norm (in-place) */
            if (dc) {
                layernorm_mt(pool, merge_buf, merge_buf, dc->g_norm, dc->b_norm,
                             n_merged, merged_dim, vm->ln_eps);
            } else {
                float *gn = xcalloc_f(merged_dim);  deq_vec(&dsl->norm_w, gn);
                float *bn = xcalloc_f(merged_dim);  deq_vec(&dsl->norm_b, bn);
                layernorm_mt(pool, merge_buf, merge_buf, gn, bn,
                             n_merged, merged_dim, vm->ln_eps);
                free(gn); free(bn);
            }

            /* fc1 → merged_dim */
            float *ds_buf = xmalloc_f((size_t)n_merged * merged_dim);
            if (dc) {
                gemm_BT_dispatch(pool, ds_buf, dc->BT_fc1, dc->BT_fc1_bf, dc->BT_fc1_fp,
                                 &dc->fc1_r, cache->n_cmgs, dc->b_fc1,
                                 merge_buf, n_merged, merged_dim, merged_dim);
            } else {
                float *Wf1 = deq_W(&dsl->fc1_w);
                float *bf1 = xcalloc_f(merged_dim);  deq_vec(&dsl->fc1_b, bf1);
                vit_gemm_bias_mt(pool, ds_buf, Wf1, bf1, merge_buf,
                                 n_merged, merged_dim, merged_dim);
                free(Wf1); free(bf1);
            }
            dump2(dump, "ds_fc1", l, n_merged, merged_dim, ds_buf);

            gelu_mt(pool, ds_buf, (size_t)n_merged * merged_dim);

            /* fc2 → proj_dim */
            float *ds_out = xmalloc_f((size_t)n_merged * vm->proj_dim);
            if (dc) {
                gemm_BT_dispatch(pool, ds_out, dc->BT_fc2, dc->BT_fc2_bf, dc->BT_fc2_fp,
                                 &dc->fc2_r, cache->n_cmgs, dc->b_fc2,
                                 ds_buf, n_merged, vm->proj_dim, merged_dim);
            } else {
                float *Wf2 = deq_W(&dsl->fc2_w);
                float *bf2 = xcalloc_f(vm->proj_dim);  deq_vec(&dsl->fc2_b, bf2);
                vit_gemm_bias_mt(pool, ds_out, Wf2, bf2, ds_buf,
                                 n_merged, vm->proj_dim, merged_dim);
                free(Wf2); free(bf2);
            }
            dump2(dump, "ds_fc2", l, n_merged, vm->proj_dim, ds_out);

            memcpy(deepstack_feats + (size_t)ds_count * n_merged * vm->proj_dim,
                   ds_out, (size_t)n_merged * vm->proj_dim * sizeof(float));
            ds_count++;
            free(ds_buf); free(ds_out);
            st_tick(&st, ST_DS);
        }
    }

    free(rope_cos); free(rope_sin);

    /* ── 4. post LN (in-place) ── */
    if (cache) {
        layernorm_mt(pool, hidden, hidden, cache->post_gamma, cache->post_beta,
                     n_patches, dim, vm->ln_eps);
    } else {
        float *gp = xcalloc_f(dim);  deq_vec(&vm->post_ln_w, gp);
        float *bp = xcalloc_f(dim);  deq_vec(&vm->post_ln_b, bp);
        layernorm_mt(pool, hidden, hidden, gp, bp, n_patches, dim, vm->ln_eps);
        free(gp); free(bp);
    }
    st_tick(&st, ST_POST);
    dump2(dump, "post_ln", -1, n_patches, dim, hidden);

    /* ── 5. spatial merge ── */
    {
        int mgw = gw / sm, mgh = gh / sm;
        for (int my = 0; my < mgh; my++) {
            for (int mx = 0; mx < mgw; mx++) {
                float *dst = merge_buf + (size_t)(my * mgw + mx) * merged_dim;
                int di = 0;
                for (int sy = 0; sy < sm; sy++) {
                    for (int sx = 0; sx < sm; sx++) {
                        int py2 = my * sm + sy;
                        int px2 = mx * sm + sx;
                        const float *src = hidden + (size_t)(py2 * gw + px2) * dim;
                        memcpy(dst + di, src, dim * sizeof(float));
                        di += dim;
                    }
                }
            }
        }
    }
    dump2(dump, "merge", -1, n_merged, merged_dim, merge_buf);

    /* ── 6. mm projection: mm0 → GELU → mm2 ── */
    float *mm_buf = xmalloc_f((size_t)n_merged * merged_dim);
    float *mm_out = xmalloc_f((size_t)n_merged * vm->proj_dim);

    if (cache) {
        gemm_BT_dispatch(pool, mm_buf, cache->BT_mm0, cache->BT_mm0_bf, cache->BT_mm0_fp,
                         &cache->mm0_r, cache->n_cmgs, cache->b_mm0,
                         merge_buf, n_merged, merged_dim, merged_dim);
    } else {
        float *Wm0 = deq_W(&vm->mm0_w);
        float *bm0 = xcalloc_f(merged_dim);  deq_vec(&vm->mm0_b, bm0);
        vit_gemm_bias_mt(pool, mm_buf, Wm0, bm0, merge_buf,
                         n_merged, merged_dim, merged_dim);
        free(Wm0); free(bm0);
    }
    dump2(dump, "mm0", -1, n_merged, merged_dim, mm_buf);

    gelu_mt(pool, mm_buf, (size_t)n_merged * merged_dim);
    dump2(dump, "mm_gelu", -1, n_merged, merged_dim, mm_buf);

    if (cache) {
        gemm_BT_dispatch(pool, mm_out, cache->BT_mm2, cache->BT_mm2_bf, cache->BT_mm2_fp,
                         &cache->mm2_r, cache->n_cmgs, cache->b_mm2,
                         mm_buf, n_merged, vm->proj_dim, merged_dim);
    } else {
        float *Wm2 = deq_W(&vm->mm2_w);
        float *bm2 = xcalloc_f(vm->proj_dim);  deq_vec(&vm->mm2_b, bm2);
        vit_gemm_bias_mt(pool, mm_out, Wm2, bm2, mm_buf,
                         n_merged, vm->proj_dim, merged_dim);
        free(Wm2); free(bm2);
    }
    dump2(dump, "mm2", -1, n_merged, vm->proj_dim, mm_out);
    st_tick(&st, ST_MM);

    /* ── 7. concat main + deepstack features per token ── */
    float *result = xcalloc_f((size_t)n_merged * total_embd);
    for (int t = 0; t < n_merged; t++) {
        float *dst = result + (size_t)t * total_embd;
        memcpy(dst, mm_out + (size_t)t * vm->proj_dim, vm->proj_dim * sizeof(float));
        for (int d = 0; d < ds_count; d++) {
            memcpy(dst + (size_t)(1 + d) * vm->proj_dim,
                   deepstack_feats + (size_t)d * n_merged * vm->proj_dim
                                   + (size_t)t * vm->proj_dim,
                   vm->proj_dim * sizeof(float));
        }
    }
    dump2(dump, "final", -1, n_merged, total_embd, result);

    /* Cleanup */
    free(hidden); free(hidden2); free(qkv); free(attn_out);
    free(ffn_buf); free(ln_buf); free(merge_buf); free(deepstack_feats);
    free(mm_buf); free(mm_out);

    st_print(&st);

    if (out_n_merged) *out_n_merged = n_merged;
    if (out_embd)     *out_embd     = total_embd;
    return result;
}
