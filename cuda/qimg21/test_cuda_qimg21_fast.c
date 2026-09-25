/* Fast Qwen-Image 2.1 denoiser for CUDA.
 *
 * The parity harness (test_cuda_qimg21_native.c) stays the numerical oracle.
 * This runner keeps the same exact-mode operation order (vector4 Welford
 * normalization, host-table-exact RoPE, PyTorch memory-efficient attention,
 * BF16 rounding boundaries) but stores activations as BF16, keeps weights
 * resident within a VRAM budget and streams the remaining blocks from pinned
 * host memory on a copy stream, and never synchronizes inside a step.
 *
 * Text and condition-image prefix tokens use the zero-timestep modulation and
 * attend only to earlier prefix tokens (causal_condition), so their per-block
 * post-RoPE K/V are step independent. They are computed once per CFG branch
 * (diffusers' prefix KV cache); every denoising step then runs only the target
 * image tokens against [cached prefix; target] keys.
 *
 * The command line is a superset of the harness, so regression.py and
 * editing_regression.py can drive it through --native-bin/--native-binary. */
#define main qimg21_oracle_main
#include "test_cuda_qimg21_native.c"
#undef main
#include "qimg21_fast_kernels.h"
#include <time.h>

#define Q21F_D 4096
#define Q21F_F 12288
#define Q21F_HEADS 32
#define Q21F_BLOCKS 32
#define Q21F_MIB ((size_t)1 << 20)

#define CK(call) do { CUresult ck_ = (call); if (ck_ != CUDA_SUCCESS) { \
    fprintf(stderr, "fast: %s failed (%d) at %s:%d\n", #call, (int)ck_, __FILE__, __LINE__); goto fail; } } while (0)
#define REQ(cond, ...) do { if (!(cond)) { fprintf(stderr, "fast: " __VA_ARGS__); fputc('\n', stderr); goto fail; } } while (0)

typedef int (*q21f_attention_fn)(void *, const void *, const void *, const void *, int, int, int, int, int,
                                 int, int, void *);

/* Per-block BF16 weight blob: fused QKV [3D,D], out [D,D], fused gate|proj
 * [2F,D], mlp.out [D,F], then F32 norm_q/norm_k [128]. */
enum { Q21F_QKV, Q21F_OUT, Q21F_GP, Q21F_MO, Q21F_NQ, Q21F_NK, Q21F_PARTS };
typedef struct {
    size_t offset[Q21F_PARTS];
    size_t bytes;
} q21f_layout;

typedef struct {
    int resident;
    CUdeviceptr dev;
    void *host;
} q21f_block;

typedef struct {
    CUdevice device;
    CUcontext context;
    CUstream compute, copy;
    cublasew_context *blas;
    CUmodule module;
    CUfunction cast_bf16, txt_norm, gelu, silu, mod_prepare, scale_prepare, norm_mod,
        qk_norm_rope, swiglu, euler, cfg_combine, checksum;
    void *plugin;
    q21f_attention_fn attention;
    q21f_layout layout;
    q21f_block block[Q21F_BLOCKS];
    int streamed[Q21F_BLOCKS], n_streamed;
    CUdeviceptr slot[2];
    CUevent loaded[2], freed[2];
    unsigned long long use;
    /* Non-block weights, BF16 unless noted. */
    CUdeviceptr txt_norm_w /* F32 */, txt_in, txt_out, img_in, t1, t2, modulation, norm_out, proj_out;
    size_t allocated, peak, budget;
    int verbose, fused_gemm;
    /* --verify-slots: device checksum of each streamed slot use */
    CUdeviceptr verify;
    int verify_max;
    /* --trace: asynchronous activation checksums per operation */
    CUdeviceptr trace_buf;
    int trace_n;
    char trace_tag[8192][24];
} q21f_runtime;

static size_t q21f_now_bytes(q21f_runtime *rt, size_t add) {
    rt->allocated += add;
    if (rt->allocated > rt->peak) rt->peak = rt->allocated;
    return add;
}

static CUdeviceptr q21f_alloc(q21f_runtime *rt, size_t bytes) {
    CUdeviceptr p = 0;
    if (!bytes) return 0;
    if (rt->budget && rt->allocated + bytes > rt->budget) {
        fprintf(stderr, "fast: allocation of %.1f MiB exceeds the %.1f MiB budget (%.1f MiB in use)\n",
                bytes / (double)Q21F_MIB, rt->budget / (double)Q21F_MIB, rt->allocated / (double)Q21F_MIB);
        return 0;
    }
    if (cuMemAlloc(&p, bytes) != CUDA_SUCCESS) {
        fprintf(stderr, "fast: cuMemAlloc(%.1f MiB) failed\n", bytes / (double)Q21F_MIB);
        return 0;
    }
    q21f_now_bytes(rt, bytes);
    return p;
}

static double q21f_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- Weights ---------------------------------------------------------- */

static void q21f_layout_init(q21f_layout *l) {
    size_t D = Q21F_D, F = Q21F_F, o = 0;
    size_t sizes[Q21F_PARTS] = {3 * D * D * 2, D * D * 2, 2 * F * D * 2, D * F * 2, 128 * 4, 128 * 4};
    for (int i = 0; i < Q21F_PARTS; i++) {
        l->offset[i] = o;
        o += (sizes[i] + 255) & ~(size_t)255;
    }
    l->bytes = o;
}

/* Copy one BF16 matrix (or an F32/BF16 vector as F32) into a host blob. */
static int q21f_copy_tensor(const qimg21_shards *s, const char *name, void *dst, size_t count, int to_f32) {
    int idx;
    st_context *st = find_tensor(s, name, &idx);
    if (!st) { fprintf(stderr, "fast: missing tensor %s\n", name); return -1; }
    const char *dt = safetensors_dtype(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const void *src = safetensors_data(st, idx);
    if (!to_f32) {
        if (strcmp(dt, "BF16") || nbytes != count * 2) {
            fprintf(stderr, "fast: %s must be BF16 with %zu elements\n", name, count);
            return -1;
        }
        memcpy(dst, src, nbytes);
        return 0;
    }
    float *out = (float *)dst;
    if (!strcmp(dt, "F32") && nbytes == count * 4) memcpy(out, src, nbytes);
    else if (!strcmp(dt, "BF16") && nbytes == count * 2) {
        const uint16_t *b = (const uint16_t *)src;
        for (size_t i = 0; i < count; i++) { uint32_t u = (uint32_t)b[i] << 16; memcpy(&out[i], &u, 4); }
    } else {
        fprintf(stderr, "fast: %s has unexpected dtype %s\n", name, dt);
        return -1;
    }
    return 0;
}

static int q21f_pack_block(const qimg21_shards *s, const q21f_layout *l, int b, uint8_t *blob) {
    size_t D = Q21F_D, F = Q21F_F;
    char name[160];
    static const char *qkv[3] = {"attn.to_q", "attn.to_k", "attn.to_v"};
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "transformer_blocks.%d.%s.weight", b, qkv[i]);
        if (q21f_copy_tensor(s, name, blob + l->offset[Q21F_QKV] + (size_t)i * D * D * 2, D * D, 0)) return -1;
    }
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_out.0.weight", b);
    if (q21f_copy_tensor(s, name, blob + l->offset[Q21F_OUT], D * D, 0)) return -1;
    snprintf(name, sizeof(name), "transformer_blocks.%d.img_mlp.gate_layer.weight", b);
    if (q21f_copy_tensor(s, name, blob + l->offset[Q21F_GP], F * D, 0)) return -1;
    snprintf(name, sizeof(name), "transformer_blocks.%d.img_mlp.proj.weight", b);
    if (q21f_copy_tensor(s, name, blob + l->offset[Q21F_GP] + F * D * 2, F * D, 0)) return -1;
    snprintf(name, sizeof(name), "transformer_blocks.%d.img_mlp.out.weight", b);
    if (q21f_copy_tensor(s, name, blob + l->offset[Q21F_MO], D * F, 0)) return -1;
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_q.weight", b);
    if (q21f_copy_tensor(s, name, blob + l->offset[Q21F_NQ], 128, 1)) return -1;
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_k.weight", b);
    if (q21f_copy_tensor(s, name, blob + l->offset[Q21F_NK], 128, 1)) return -1;
    return 0;
}

/* All uploads are ordered on the compute stream. A synchronous cuMemcpyHtoD
 * runs on the legacy stream, which does not order with the non-blocking
 * compute stream, and from pageable memory it may return before its DMA
 * lands; under PCIe contention kernels then read stale data. wait=1 is for
 * pinned staging buffers that the caller reuses immediately. */
static int q21f_upload(q21f_runtime *rt, CUdeviceptr d, const void *h, size_t bytes, int wait) {
    if (cuMemcpyHtoDAsync(d, h, bytes, rt->compute) != CUDA_SUCCESS) return -1;
    return wait && cuStreamSynchronize(rt->compute) != CUDA_SUCCESS ? -1 : 0;
}

static CUdeviceptr q21f_upload_tensor(q21f_runtime *rt, const qimg21_shards *s, const char *name,
                                      size_t count, int to_f32, void *staging) {
    size_t bytes = count * (to_f32 ? 4 : 2);
    CUdeviceptr d = q21f_alloc(rt, bytes);
    if (!d) return 0;
    if (q21f_copy_tensor(s, name, staging, count, to_f32) ||
        q21f_upload(rt, d, staging, bytes, 1)) {
        cuMemFree(d);
        rt->allocated -= bytes;
        return 0;
    }
    return d;
}

/* ---- Streaming -------------------------------------------------------- */

/* Streamed blocks are used in a fixed cyclic order; use u occupies slot u%2
 * and its copy is queued once use u-2 has released that slot. */
static int q21f_queue_copy(q21f_runtime *rt, unsigned long long use) {
    int slot = (int)(use & 1);
    int b = rt->streamed[use % (unsigned)rt->n_streamed];
    if (use >= 2 && cuStreamWaitEvent(rt->copy, rt->freed[slot], 0) != CUDA_SUCCESS) return -1;
    if (cuMemcpyHtoDAsync(rt->slot[slot], rt->block[b].host, rt->layout.bytes, rt->copy) != CUDA_SUCCESS ||
        cuEventRecord(rt->loaded[slot], rt->copy) != CUDA_SUCCESS)
        return -1;
    return 0;
}

static CUdeviceptr q21f_block_begin(q21f_runtime *rt, int b) {
    if (rt->block[b].resident) return rt->block[b].dev;
    int slot = (int)(rt->use & 1);
    if (rt->streamed[rt->use % (unsigned)rt->n_streamed] != b) {
        fprintf(stderr, "fast: streamed block order violated at block %d\n", b);
        return 0;
    }
    if (cuStreamWaitEvent(rt->compute, rt->loaded[slot], 0) != CUDA_SUCCESS) return 0;
    if (rt->verify && rt->use < (unsigned long long)rt->verify_max) {
        CUdeviceptr out = rt->verify + rt->use * 8, p = rt->slot[slot];
        unsigned long long words = rt->layout.bytes / 4;
        unsigned stride = 4099;
        void *a[] = {&out, &p, &words, &stride};
        if (cuLaunchKernel(rt->checksum, 1, 1, 1, 256, 1, 1, 0, rt->compute, a, NULL) != CUDA_SUCCESS) return 0;
    }
    return rt->slot[slot];
}

static int q21f_block_end(q21f_runtime *rt, int b) {
    if (rt->block[b].resident) return 0;
    int slot = (int)(rt->use & 1);
    if (cuEventRecord(rt->freed[slot], rt->compute) != CUDA_SUCCESS) return -1;
    rt->use++;
    return q21f_queue_copy(rt, rt->use + 1);
}

/* ---- Kernels, GEMM, attention ----------------------------------------- */

static int q21f_launch(q21f_runtime *rt, CUfunction f, unsigned gx, unsigned gy, unsigned bx, void **args) {
    return cuLaunchKernel(f, gx, gy, 1, bx, 1, 1, 0, rt->compute, args, NULL) == CUDA_SUCCESS ? 0 : -1;
}

static int q21f_launch_n(q21f_runtime *rt, CUfunction f, size_t n, void **args) {
    return q21f_launch(rt, f, (unsigned)((n + 255) / 256), 1, 256, args);
}

/* Row-major Y[M,N] (stride ldy) = X[M,K] (stride ldx) * W[N,K]^T. */
static int q21f_gemm(q21f_runtime *rt, CUdeviceptr y, int ldy, CUdeviceptr w, CUdeviceptr x, int ldx,
                     int m, int n, int k) {
    return cublasew_gemm_bf16_bf16_bf16_rowmajor_nt_ld(rt->blas, y, ldy, w, x, ldx, m, n, k);
}

/* QKV and gate|proj projections. Fused, one GEMM covers the concatenated
 * weight rows; otherwise each projection runs as the separate [M,n] GEMM the
 * reference framework issues, writing into its column slice. */
static int q21f_gemm_parts(q21f_runtime *rt, CUdeviceptr y, CUdeviceptr w, CUdeviceptr x, int m, int parts,
                           int n, int k) {
    if (rt->fused_gemm) return q21f_gemm(rt, y, parts * n, w, x, k, m, parts * n, k);
    for (int i = 0; i < parts; i++)
        if (q21f_gemm(rt, y + (size_t)i * n * 2, parts * n, w + (size_t)i * n * k * 2, x, k, m, n, k)) return -1;
    return 0;
}

/* Diagnostic: queue a device checksum of a range without synchronizing. */
static void q21f_mark(q21f_runtime *rt, const char *tag, int b, CUdeviceptr p, size_t bytes) {
    if (!rt->trace_buf || rt->trace_n >= 8192) return;
    CUdeviceptr out = rt->trace_buf + (size_t)rt->trace_n * 8;
    unsigned long long words = bytes / 4;
    unsigned stride = 1;
    void *a[] = {&out, &p, &words, &stride};
    snprintf(rt->trace_tag[rt->trace_n], 24, "b%02d %s", b, tag);
    if (cuLaunchKernel(rt->checksum, 1, 1, 1, 256, 1, 1, 0, rt->compute, a, NULL) == CUDA_SUCCESS) rt->trace_n++;
}

static int q21f_attend(q21f_runtime *rt, CUdeviceptr out, CUdeviceptr q, CUdeviceptr k, CUdeviceptr v,
                       int nq, int nk, int mask) {
    int D = Q21F_D;
    return rt->attention((void *)(uintptr_t)out, (const void *)(uintptr_t)q, (const void *)(uintptr_t)k,
                         (const void *)(uintptr_t)v, nq, nk, Q21F_HEADS, D, D, D, mask, (void *)rt->compute);
}

static int q21f_norm_mod(q21f_runtime *rt, CUdeviceptr out, CUdeviceptr h, CUdeviceptr y, CUdeviceptr g,
                         CUdeviceptr f, int rows) {
    int D = Q21F_D;
    float eps = 1e-6f;
    void *a[] = {&out, &h, &y, &g, &f, &D, &eps};
    return q21f_launch(rt, rt->norm_mod, (unsigned)rows, 1, 128, a);
}

/* ---- Model state ------------------------------------------------------ */

/* One CFG branch: its prefix (text, plus condition images when editing), the
 * embedded prefix hidden state, the cached per-block prefix K/V, a RoPE table
 * for [prefix; target] and the working K/V rows [prefix; target]. */
typedef struct {
    int nt, prefix;
    q21_joint_layout layout; /* editing only */
    int edit;
    const float *prompt;     /* [nt, D] F32 */
    CUdeviceptr prefix_hidden;      /* [prefix][D] after txt_in/img_in */
    CUdeviceptr cache_k, cache_v;   /* [32][prefix][D] */
    CUdeviceptr work_k, work_v;     /* [prefix + target][D] */
    CUdeviceptr rope;               /* F32 [prefix + target][128] */
} q21f_branch;

typedef struct {
    int target, rows_max;
    CUdeviceptr hidden, x, qkv, q, attn, y, gp, prompt_f32, cond_bf16, temb, temb2, tsilu, te, mod, modc,
        scale, fscale, latent, sample, pred;
    int nc;
    const float *condition; /* [nc, 64] F32 */
} q21f_state;

static void q21f_rope_rows(float *table, const float *base, const int *pos3, int rows) {
    for (int t = 0; t < rows; t++)
        for (int j = 0; j < 128; j += 2) {
            int axis = j < 16 ? 0 : (j < 72 ? 1 : 2), pos = pos3[t * 3 + axis];
            int row = pos >= 0 ? pos : 8192 + pos + 1024;
            table[t * 128 + j] = base[(size_t)row * 128 + j];
            table[t * 128 + j + 1] = base[(size_t)row * 128 + j + 1];
        }
}

/* Positions of every joint token [prefix + target] as (axis0, axis1, axis2). */
static int *q21f_positions(const q21f_branch *br, int ih, int iw) {
    int n = br->prefix + ih * iw;
    int *p = (int *)malloc((size_t)n * 3 * sizeof(int));
    if (!p) return NULL;
    if (br->edit) {
        memcpy(p, br->layout.position, (size_t)n * 3 * sizeof(int));
        return p;
    }
    for (int t = 0; t < n; t++) {
        if (t < br->nt) { p[t * 3] = p[t * 3 + 1] = p[t * 3 + 2] = t; continue; }
        int r = (t - br->nt) / iw, c = (t - br->nt) % iw;
        p[t * 3] = br->nt;
        p[t * 3 + 1] = -(ih - ih / 2) + r;
        p[t * 3 + 2] = -(iw - iw / 2) + c;
    }
    return p;
}

static int q21f_valid_rope(const npy_f32 *base, const int *pos, int n) {
    for (int i = 0; i < n * 3; i++) {
        int row = pos[i] >= 0 ? pos[i] : 8192 + pos[i] + 1024;
        if (row < 0 || (size_t)row >= base->shape[0]) return 0;
    }
    return 1;
}

/* txt_in (zero-centred RMSNorm, in_layer, GELU(tanh), out_layer) and, when
 * editing, img_in of the condition latents, scattered into joint prefix order. */
static int q21f_embed_prefix(q21f_runtime *rt, q21f_state *st, q21f_branch *br) {
    int D = Q21F_D, P = br->prefix, nt = br->nt;
    CUdeviceptr x = st->x;
    CK(cuMemcpyHtoDAsync(st->prompt_f32, br->prompt, (size_t)nt * D * 4, rt->compute));
    {
        float eps = 1e-6f;
        void *a[] = {&x, &st->prompt_f32, &rt->txt_norm_w, &D, &eps};
        REQ(!q21f_launch(rt, rt->txt_norm, (unsigned)nt, 1, 256, a), "txt_norm launch");
    }
    REQ(!q21f_gemm(rt, st->y, D, rt->txt_in, x, D, nt, D, D), "txt_in GEMM");
    { int n = nt * D; void *a[] = {&st->y, &n}; REQ(!q21f_launch_n(rt, rt->gelu, (size_t)n, a), "gelu"); }
    CUdeviceptr text = br->edit ? st->attn : br->prefix_hidden;
    REQ(!q21f_gemm(rt, text, D, rt->txt_out, st->y, D, nt, D, D), "txt_out GEMM");
    if (br->edit) {
        int n = st->nc * 64;
        CK(cuMemcpyHtoDAsync(st->y, st->condition, (size_t)n * 4, rt->compute));
        void *c[] = {&st->cond_bf16, &st->y, &n};
        REQ(!q21f_launch_n(rt, rt->cast_bf16, (size_t)n, c), "condition cast");
        CUdeviceptr img = st->attn + (size_t)nt * D * 2;
        REQ(!q21f_gemm(rt, img, D, rt->img_in, st->cond_bf16, 64, st->nc, D, 64), "condition img_in GEMM");
        for (int t = 0; t < P; t++) {
            int ti = br->layout.text_index[t], ii = br->layout.image_index[t];
            REQ(ti >= 0 || (ii >= 0 && ii < st->nc), "invalid prefix layout row %d", t);
            size_t src = ti >= 0 ? (size_t)ti : (size_t)nt + (size_t)ii;
            CK(cuMemcpyDtoDAsync(br->prefix_hidden + (size_t)t * D * 2, st->attn + src * D * 2, (size_t)D * 2,
                                 rt->compute));
        }
    }
    return 0;
fail:
    return -1;
}

/* Joint-sequence rows handled by one pass of the 32 blocks. PREFIX rows use
 * the zero-timestep modulation and fill the K/V cache; TARGET rows attend to
 * [cached prefix; target]; JOINT recomputes [prefix; target] together like
 * the uncached reference, optionally storing the prefix K/V (diffusers'
 * extract step). */
enum { Q21F_PREFIX, Q21F_TARGET, Q21F_JOINT };

static int q21f_rows(const q21f_branch *br, int kind, int N) {
    return kind == Q21F_PREFIX ? br->prefix : kind == Q21F_TARGET ? N : br->prefix + N;
}

/* Modulated norm over one branch's rows: prefix rows take modulation row 1,
 * target rows row 0. which selects (scale, gate) pair 0 (attention) or 1 (MLP)
 * of the norm and, for the gated residual, the preceding pair. */
static int q21f_branch_norm(q21f_runtime *rt, q21f_state *st, const q21f_branch *br, int kind, size_t row0,
                            int N, int pair, CUdeviceptr y, CUdeviceptr final_scale) {
    int D = Q21F_D, P = br->prefix;
    int prefix_rows = kind == Q21F_TARGET ? 0 : P, target_rows = kind == Q21F_PREFIX ? 0 : N;
    int gate_pair = pair ? 0 : 1;
    for (int part = 0; part < 2; part++) {
        int rows = part ? target_rows : prefix_rows, mrow = part ? 0 : 1;
        if (!rows) continue;
        size_t r = row0 + (part ? (size_t)prefix_rows : 0);
        CUdeviceptr f = final_scale ? final_scale + (size_t)mrow * D * 4
                                    : st->modc + (size_t)(mrow * 4 + pair * 2) * D * 4;
        CUdeviceptr g = st->modc + (size_t)(mrow * 4 + gate_pair * 2 + 1) * D * 4;
        CUdeviceptr yy = y ? y + r * D * 2 : 0;
        if (q21f_norm_mod(rt, st->x + r * D * 2, st->hidden + r * D * 2, yy, g, f, rows)) return -1;
    }
    return 0;
}

static int q21f_pass(q21f_runtime *rt, q21f_state *st, q21f_branch *br, int nb, int kind, int store_cache,
                     CUdeviceptr pred) {
    int D = Q21F_D, F = Q21F_F, N = st->target;
    size_t base[2];
    int M = 0;
    for (int i = 0; i < nb; i++) { base[i] = (size_t)M; M += q21f_rows(&br[i], kind, N); }
    CUdeviceptr h = st->hidden;
    /* Initial hidden: embedded prefix and/or img_in(latents) for each branch. */
    if (kind != Q21F_PREFIX)
        REQ(!q21f_gemm(rt, st->y, D, rt->img_in, st->latent, 64, N, D, 64), "img_in GEMM");
    for (int i = 0; i < nb; i++) {
        size_t r = base[i];
        if (kind != Q21F_TARGET) {
            CK(cuMemcpyDtoDAsync(h + r * D * 2, br[i].prefix_hidden, (size_t)br[i].prefix * D * 2, rt->compute));
            r += (size_t)br[i].prefix;
        }
        if (kind != Q21F_PREFIX) CK(cuMemcpyDtoDAsync(h + r * D * 2, st->y, (size_t)N * D * 2, rt->compute));
    }
    for (int b = 0; b < Q21F_BLOCKS; b++) {
        CUdeviceptr w = q21f_block_begin(rt, b);
        REQ(w, "block %d weights unavailable", b);
        const q21f_layout *l = &rt->layout;
        for (int i = 0; i < nb; i++)
            REQ(!q21f_branch_norm(rt, st, &br[i], kind, base[i], N, 0, b ? st->y : 0, 0), "norm1");
        q21f_mark(rt, "x1", b, st->x, (size_t)M * D * 2);
        q21f_mark(rt, "h", b, h, (size_t)M * D * 2);
        REQ(!q21f_gemm_parts(rt, st->qkv, w + l->offset[Q21F_QKV], st->x, M, 3, D, D), "QKV");
        q21f_mark(rt, "qkv", b, st->qkv, (size_t)M * 3 * D * 2);
        CUdeviceptr nq = w + l->offset[Q21F_NQ], nk = w + l->offset[Q21F_NK];
        int last_prefix = kind == Q21F_PREFIX && b == Q21F_BLOCKS - 1;
        for (int i = 0; i < nb; i++) {
            q21f_branch *r = &br[i];
            int P = r->prefix, rows = q21f_rows(r, kind, N), heads = Q21F_HEADS;
            /* K/V rows: PREFIX writes the block's cache directly; the other
             * kinds write [prefix; target] working rows. */
            CUdeviceptr kbuf = kind == Q21F_PREFIX ? r->cache_k + (size_t)b * P * D * 2 : r->work_k;
            CUdeviceptr vbuf = kind == Q21F_PREFIX ? r->cache_v + (size_t)b * P * D * 2 : r->work_v;
            int kv_row = kind == Q21F_TARGET ? P : 0;
            CUdeviceptr table = r->rope + (size_t)(kind == Q21F_TARGET ? P : 0) * 128 * 4;
            if (kind == Q21F_TARGET) {
                CK(cuMemcpyDtoDAsync(r->work_k, r->cache_k + (size_t)b * P * D * 2, (size_t)P * D * 2, rt->compute));
                CK(cuMemcpyDtoDAsync(r->work_v, r->cache_v + (size_t)b * P * D * 2, (size_t)P * D * 2, rt->compute));
            }
            CUdeviceptr qkv = st->qkv + base[i] * 3 * D * 2, q = st->q + base[i] * D * 2;
            void *a[] = {&q, &kbuf, &vbuf, &qkv, &nq, &nk, &table, &heads, &kv_row};
            REQ(!q21f_launch(rt, rt->qk_norm_rope, (unsigned)rows, Q21F_HEADS, 32, a), "rope");
            if (kind == Q21F_JOINT && store_cache) {
                CK(cuMemcpyDtoDAsync(r->cache_k + (size_t)b * P * D * 2, r->work_k, (size_t)P * D * 2, rt->compute));
                CK(cuMemcpyDtoDAsync(r->cache_v + (size_t)b * P * D * 2, r->work_v, (size_t)P * D * 2, rt->compute));
            }
            if (last_prefix) continue;
            CUdeviceptr out = st->attn + base[i] * D * 2;
            if (kind != Q21F_TARGET)
                /* Text runs are causal; each condition image is bidirectional
                 * and sees all earlier tokens. */
                for (int s = 0; s < P;) {
                    int e = s + 1, id = r->edit ? r->layout.image_id[s] : -1;
                    while (e < P && (r->edit ? r->layout.image_id[e] : -1) == id) e++;
                    REQ(!q21f_attend(rt, out + (size_t)s * D * 2, q + (size_t)s * D * 2, kbuf, vbuf, e - s, e,
                                     id < 0 ? 2 : 0), "prefix attention");
                    s = e;
                }
            if (kind != Q21F_PREFIX) {
                size_t t0 = kind == Q21F_JOINT ? (size_t)P : 0;
                REQ(!q21f_attend(rt, out + t0 * D * 2, q + t0 * D * 2, kbuf, vbuf, N, P + N, 0), "attention");
            }
        }
        if (last_prefix) { REQ(!q21f_block_end(rt, b), "block end"); break; }
        q21f_mark(rt, "q", b, st->q, (size_t)M * D * 2);
        q21f_mark(rt, "k", b, br[0].work_k, (size_t)(br[0].prefix + N) * D * 2);
        q21f_mark(rt, "attn", b, st->attn, (size_t)M * D * 2);
        REQ(!q21f_gemm(rt, st->y, D, w + l->offset[Q21F_OUT], st->attn, D, M, D, D), "out");
        q21f_mark(rt, "out", b, st->y, (size_t)M * D * 2);
        for (int i = 0; i < nb; i++)
            REQ(!q21f_branch_norm(rt, st, &br[i], kind, base[i], N, 1, st->y, 0), "norm2");
        q21f_mark(rt, "x2", b, st->x, (size_t)M * D * 2);
        REQ(!q21f_gemm_parts(rt, st->gp, w + l->offset[Q21F_GP], st->x, M, 2, F, D), "gate|proj");
        q21f_mark(rt, "gp", b, st->gp, (size_t)M * 2 * F * 2);
        { int rows = M, ff = F; void *a[] = {&st->gp, &rows, &ff};
          REQ(!q21f_launch_n(rt, rt->swiglu, (size_t)M * F, a), "swiglu"); }
        REQ(!q21f_gemm(rt, st->y, D, w + l->offset[Q21F_MO], st->gp, 2 * F, M, D, F), "mlp.out");
        q21f_mark(rt, "mlp", b, st->y, (size_t)M * D * 2);
        REQ(!q21f_block_end(rt, b), "block end");
    }
    if (kind == Q21F_PREFIX) return 0;
    /* Final modulated norm and proj_out on each branch's target rows. */
    for (int i = 0; i < nb; i++) {
        size_t t0 = base[i] + (kind == Q21F_JOINT ? (size_t)br[i].prefix : 0);
        q21f_branch target_only = br[i];
        REQ(!q21f_branch_norm(rt, st, &target_only, Q21F_TARGET, t0, N, 0, st->y, st->fscale), "final norm");
        REQ(!q21f_gemm(rt, pred + (size_t)i * N * 64 * 2, 64, rt->proj_out, st->x + t0 * D * 2, D, N, 64, D),
            "proj_out");
    }
    return 0;
fail:
    return -1;
}

/* Time embedding for one step: modulation constants for both rows and the
 * final-norm scale. Row 1 (zero timestep) is step independent. pinned_te is a
 * slice owned by this call: the upload runs asynchronously. */
static int q21f_time(q21f_runtime *rt, q21f_state *st, float t, uint16_t *pinned_te) {
    int D = Q21F_D;
    float te[512];
    for (int i = 0; i < 128; i++) {
        float f = expf(-logf(10000.f) * (float)i / 128.f), a = t * 1000.f * f;
        te[i] = cosf(a); te[128 + i] = sinf(a); te[256 + i] = 1.0f; te[384 + i] = 0.0f;
    }
    for (int i = 0; i < 512; i++) pinned_te[i] = qimg_f32_to_bf16_rne(te[i]);
    CK(cuMemcpyHtoDAsync(st->te, pinned_te, 512 * 2, rt->compute));
    REQ(!q21f_gemm(rt, st->temb, D, rt->t1, st->te, 256, 2, D, 256), "time linear_1");
    { int n = 2 * D; void *a[] = {&st->temb, &n}; REQ(!q21f_launch_n(rt, rt->silu, (size_t)n, a), "silu"); }
    REQ(!q21f_gemm(rt, st->temb2, D, rt->t2, st->temb, D, 2, D, D), "time linear_2");
    CK(cuMemcpyDtoDAsync(st->tsilu, st->temb2, (size_t)2 * D * 2, rt->compute));
    { int n = 2 * D; void *a[] = {&st->tsilu, &n}; REQ(!q21f_launch_n(rt, rt->silu, (size_t)n, a), "silu"); }
    REQ(!q21f_gemm(rt, st->mod, 4 * D, rt->modulation, st->tsilu, D, 2, 4 * D, D), "modulation");
    { void *a[] = {&st->modc, &st->mod, &D}; REQ(!q21f_launch_n(rt, rt->mod_prepare, (size_t)8 * D, a), "mod"); }
    REQ(!q21f_gemm(rt, st->scale, D, rt->norm_out, st->tsilu, D, 2, D, D), "norm_out");
    { int n = 2 * D; void *a[] = {&st->fscale, &st->scale, &n};
      REQ(!q21f_launch_n(rt, rt->scale_prepare, (size_t)n, a), "scale"); }
    return 0;
fail:
    return -1;
}

/* ---- Planning --------------------------------------------------------- */

typedef struct {
    size_t fixed, activations, kv, block_bytes;
    int resident, streamed;
} q21f_plan;

static size_t q21f_state_bytes(int rows, int target, int branches, int nt_max, int nc) {
    size_t D = Q21F_D, F = Q21F_F, r = (size_t)rows;
    size_t b = r * D * 2 * 5          /* hidden, x, q, attn, y */
             + r * 3 * D * 2          /* qkv */
             + r * 2 * F * 2          /* gate|proj */
             + (size_t)nt_max * D * 4 /* F32 prompt */
             + (size_t)(nc > 0 ? nc : 1) * 64 * 2
             + (size_t)target * 64 * (4 + 2) /* sample, latent */
             + (size_t)target * branches * 64 * 2 /* pred */
             + 64 * D * 4;            /* time/modulation vectors */
    return b;
}

/* ---- CLI -------------------------------------------------------------- */

static void q21f_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model DIR --prompt-embeds E.npy --latents L.npy [--height-tokens H --width-tokens W]\n"
            "  [--steps N | --timestep T] [--negative-prompt-embeds NEG.npy --guidance-scale S]\n"
            "  [--editing-layout L.txt --condition-latents C.npy [--negative-editing-layout NL.txt]]\n"
            "  [--vram-budget-mib MIB] [--cfg-batch 0|1] [--fused-gemm 0|1] [--kv-cache on|off]\n"
            "  [--prefix-pass extract|separate] [--plan-only] [--profile]\n"
            "  diagnostics: [--trace] [--verify-slots]\n"
            "  [--attention cutlass-efficient] [--normalization vector4] [--rope host-table-exact]\n"
            "  [--attention-plugin PATH] [--rope-table-base PATH]\n"
            "  [--out O.npy] [--dump-dir DIR] [--pred-dir DIR] [--quiet|--verbose]\n", argv0);
}

int main(int argc, char **argv) {
    const char *model = NULL, *prompt_path = NULL, *latent_path = NULL, *negative_path = NULL;
    const char *layout_path = NULL, *negative_layout_path = NULL, *condition_path = NULL;
    const char *out_path = "native_latents.npy", *dump_dir = NULL, *pred_dir = NULL;
    const char *plugin_path = "cuda/qimg21/libq21_fast_attention.so";
    const char *rope_path = "cuda/qimg21/qwen21_rope_freqs.npy";
    int ih = 16, iw = 16, steps = 1, verbose = 1, cfg_batch = 1, plan_only = 0, profile = 0, fused_gemm = 1;
    int kv_cache = 1, extract = 1, trace = 0, verify_slots = 0;
    double budget_mib = 0;
    float guidance = 1.0f, manual_t = -1.0f;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
        if (!strcmp(a, "--model") && more) model = argv[++i];
        else if (!strcmp(a, "--prompt-embeds") && more) prompt_path = argv[++i];
        else if (!strcmp(a, "--latents") && more) latent_path = argv[++i];
        else if (!strcmp(a, "--negative-prompt-embeds") && more) negative_path = argv[++i];
        else if (!strcmp(a, "--guidance-scale") && more) guidance = (float)atof(argv[++i]);
        else if (!strcmp(a, "--editing-layout") && more) layout_path = argv[++i];
        else if (!strcmp(a, "--negative-editing-layout") && more) negative_layout_path = argv[++i];
        else if (!strcmp(a, "--condition-latents") && more) condition_path = argv[++i];
        else if (!strcmp(a, "--height-tokens") && more) ih = atoi(argv[++i]);
        else if (!strcmp(a, "--width-tokens") && more) iw = atoi(argv[++i]);
        else if (!strcmp(a, "--steps") && more) steps = atoi(argv[++i]);
        else if (!strcmp(a, "--timestep") && more) manual_t = (float)atof(argv[++i]);
        else if (!strcmp(a, "--out") && more) out_path = argv[++i];
        else if (!strcmp(a, "--dump-dir") && more) dump_dir = argv[++i];
        else if (!strcmp(a, "--pred-dir") && more) pred_dir = argv[++i];
        else if (!strcmp(a, "--vram-budget-mib") && more) budget_mib = atof(argv[++i]);
        else if (!strcmp(a, "--cfg-batch") && more) cfg_batch = atoi(argv[++i]);
        else if (!strcmp(a, "--fused-gemm") && more) fused_gemm = atoi(argv[++i]);
        else if (!strcmp(a, "--kv-cache") && more) {
            const char *m = argv[++i];
            if (strcmp(m, "on") && strcmp(m, "off")) { q21f_usage(argv[0]); return 2; }
            kv_cache = !strcmp(m, "on");
        } else if (!strcmp(a, "--prefix-pass") && more) {
            const char *m = argv[++i];
            if (strcmp(m, "extract") && strcmp(m, "separate")) { q21f_usage(argv[0]); return 2; }
            extract = !strcmp(m, "extract");
        }
        else if (!strcmp(a, "--attention-plugin") && more) plugin_path = argv[++i];
        else if (!strcmp(a, "--rope-table-base") && more) rope_path = argv[++i];
        else if (!strcmp(a, "--plan-only")) plan_only = 1;
        else if (!strcmp(a, "--trace")) trace = 1;
        else if (!strcmp(a, "--verify-slots")) verify_slots = 1;
        else if (!strcmp(a, "--profile")) profile = 1;
        else if (!strcmp(a, "--verbose")) verbose = 2;
        else if (!strcmp(a, "--quiet")) verbose = 0;
        else if (!strcmp(a, "--attention") && more) {
            const char *m = argv[++i];
            if (strcmp(m, "cutlass-efficient")) {
                fprintf(stderr, "fast: only --attention cutlass-efficient is implemented (got %s)\n", m);
                return 2;
            }
        } else if ((!strcmp(a, "--normalization") || !strcmp(a, "--rope")) && more) {
            const char *m = argv[++i];
            if (strcmp(m, "vector4") && strcmp(m, "host-table-exact")) {
                fprintf(stderr, "fast: implements vector4 normalization and host-table-exact RoPE only (got %s %s)\n",
                        a, m);
                return 2;
            }
        } else { q21f_usage(argv[0]); return 2; }
    }
    if (!model || !prompt_path || !latent_path || ih < 1 || iw < 1 || ih > 1024 || iw > 1024 ||
        steps < 1 || steps > 100 || (manual_t >= 0.0f && steps != 1) || cfg_batch < 0 || cfg_batch > 1) {
        q21f_usage(argv[0]);
        return 2;
    }
    if (!!layout_path != !!condition_path || (negative_layout_path && !layout_path) ||
        (layout_path && negative_path && !negative_layout_path)) {
        fprintf(stderr, "fast: editing requires a layout plus condition latents; editing CFG also needs a negative layout\n");
        return 2;
    }
    if (negative_path && guidance <= 1.0f) {
        fprintf(stderr, "fast: --guidance-scale must be > 1 with negative embeds\n");
        return 2;
    }

    int rc = 1, N = ih * iw, nb = negative_path ? 2 : 1;
    npy_f32 pe = {0}, ne = {0}, la = {0}, cond = {0}, rope = {0};
    q21f_runtime rt;
    q21f_state st;
    q21f_branch br[2];
    qimg21_shards shards = {{0}, 0};
    void *staging = NULL;
    uint16_t *pinned_te = NULL;
    float *host_out = NULL, *sigmas = NULL;
    uint16_t *host_pred = NULL;
    memset(&rt, 0, sizeof(rt));
    memset(&st, 0, sizeof(st));
    memset(br, 0, sizeof(br));
    rt.verbose = verbose;
    rt.fused_gemm = fused_gemm;

    REQ(!npy_read_f32(prompt_path, &pe) && !npy_read_f32(latent_path, &la), "cannot read inputs");
    REQ(!negative_path || !npy_read_f32(negative_path, &ne), "cannot read negative embeds");
    REQ(!npy_read_f32(rope_path, &rope) && rope.ndim == 2 && rope.shape[1] == 128,
        "invalid RoPE table %s", rope_path);
    br[0].nt = pe.ndim == 3 && pe.shape[0] == 1 ? (int)pe.shape[1] : (pe.ndim == 2 ? (int)pe.shape[0] : 0);
    br[0].prompt = pe.data;
    if (negative_path) {
        br[1].nt = ne.ndim == 3 && ne.shape[0] == 1 ? (int)ne.shape[1] : (ne.ndim == 2 ? (int)ne.shape[0] : 0);
        br[1].prompt = ne.data;
    }
    int ni = la.ndim == 3 && la.shape[0] == 1 ? (int)la.shape[1] : (la.ndim == 2 ? (int)la.shape[0] : 0);
    REQ(br[0].nt > 0 && pe.shape[pe.ndim - 1] == Q21F_D && ni == N && la.shape[la.ndim - 1] == 64 &&
        (!negative_path || (br[1].nt > 0 && ne.shape[ne.ndim - 1] == Q21F_D)),
        "expected embeds [T,4096], optional negative embeds [U,4096], and latents [H*W,64]");
    if (condition_path) {
        REQ(!npy_read_f32(condition_path, &cond), "cannot read condition latents");
        st.nc = cond.ndim == 3 && cond.shape[0] == 1 ? (int)cond.shape[1] : (cond.ndim == 2 ? (int)cond.shape[0] : 0);
        REQ(st.nc > 0 && cond.shape[cond.ndim - 1] == 64, "condition latents must be [N,64]");
        st.condition = cond.data;
    }
    for (int i = 0; i < nb; i++) {
        const char *path = i ? negative_layout_path : layout_path;
        if (path) {
            int slots, h, w;
            REQ(!q21_layout_read(path, &br[i].layout, &slots, &h, &w) && slots == br[i].nt && h == ih &&
                w == iw && br[i].layout.image_tokens == st.nc + N, "invalid editing layout %s", path);
            br[i].edit = 1;
            br[i].prefix = br[i].layout.prefix;
        } else br[i].prefix = br[i].nt;
    }

    /* ---- Device ---- */
    REQ(cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) == CUEW_SUCCESS, "cuewInit failed");
    CK(cuInit(0));
    CK(cuDeviceGet(&rt.device, 0));
    CK(cuDevicePrimaryCtxRetain(&rt.context, rt.device));
    CK(cuCtxSetCurrent(rt.context));
    CK(cuStreamCreate(&rt.compute, CU_STREAM_NON_BLOCKING));
    CK(cuStreamCreate(&rt.copy, CU_STREAM_NON_BLOCKING));
    REQ(cublasewCreate(&rt.blas, rt.compute) == 0, "cuBLAS unavailable");
    REQ(cu_compile_kernels_ex(&rt.module, rt.device, q21f_kernel_src, "qimg21_fast.cu", verbose,
                              "fast", 0) >= 0, "kernel compile failed");
#define GETF(field, name) CK(cuModuleGetFunction(&rt.field, rt.module, name))
    GETF(cast_bf16, "cast_bf16"); GETF(txt_norm, "txt_norm"); GETF(gelu, "gelu_bf16"); GETF(silu, "silu_bf16");
    GETF(mod_prepare, "mod_prepare"); GETF(scale_prepare, "scale_prepare"); GETF(norm_mod, "norm_mod");
    GETF(qk_norm_rope, "qk_norm_rope"); GETF(swiglu, "swiglu");
    GETF(euler, "euler"); GETF(cfg_combine, "cfg_combine"); GETF(checksum, "checksum");
#undef GETF
    rt.plugin = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
    REQ(rt.plugin && (rt.attention = (q21f_attention_fn)dlsym(rt.plugin, "q21f_attention")),
        "cannot load attention plugin %s: %s", plugin_path, dlerror());

    /* ---- Plan ---- */
    q21f_layout_init(&rt.layout);
    int nt_max = br[0].nt > br[1].nt ? br[0].nt : br[1].nt;
    int prefix_max = br[0].prefix > br[1].prefix ? br[0].prefix : br[1].prefix;
    int batch = cfg_batch ? nb : 1;
    st.target = N;
    st.rows_max = N * batch;
    if (st.rows_max < prefix_max + N) st.rows_max = prefix_max + N;
    if (st.rows_max < nt_max + st.nc) st.rows_max = nt_max + st.nc;
    q21f_plan plan = {0};
    plan.block_bytes = rt.layout.bytes;
    plan.fixed = (size_t)(2 * Q21F_D * Q21F_D + Q21F_D * 64 + Q21F_D * 256 + Q21F_D * Q21F_D +
                          4 * Q21F_D * Q21F_D + Q21F_D * Q21F_D + 64 * Q21F_D) * 2 + Q21F_D * 4;
    plan.activations = q21f_state_bytes(st.rows_max, N, nb, nt_max, st.nc);
    for (int i = 0; i < nb; i++)
        plan.kv += ((size_t)Q21F_BLOCKS * br[i].prefix + br[i].prefix + N) * Q21F_D * 2 * 2 +
                   (size_t)br[i].prefix * Q21F_D * 2 + (size_t)(br[i].prefix + N) * 128 * 4;
    size_t free_bytes = 0, total_bytes = 0;
    CK(cuMemGetInfo(&free_bytes, &total_bytes));
    /* The budget covers this process: explicit allocations plus a fixed
     * allowance for the context, cuBLAS and module images. */
    const size_t allowance = 512 * Q21F_MIB;
    size_t budget = budget_mib > 0 ? (size_t)(budget_mib * Q21F_MIB) : free_bytes;
    size_t usable = budget > allowance ? budget - allowance : 0;
    if (usable > free_bytes - (free_bytes > 256 * Q21F_MIB ? 256 * Q21F_MIB : 0))
        usable = free_bytes > 256 * Q21F_MIB ? free_bytes - 256 * Q21F_MIB : 0;
    size_t base = plan.fixed + plan.activations + plan.kv;
    REQ(usable > base + 2 * plan.block_bytes, "budget too small: need at least %.0f MiB for activations, "
        "K/V and two streaming slots, have %.0f MiB usable", (base + 2 * plan.block_bytes) / (double)Q21F_MIB,
        usable / (double)Q21F_MIB);
    if (usable >= base + Q21F_BLOCKS * plan.block_bytes) plan.resident = Q21F_BLOCKS;
    else {
        plan.resident = (int)((usable - base - 2 * plan.block_bytes) / plan.block_bytes);
        if (plan.resident > Q21F_BLOCKS - 3) plan.resident = Q21F_BLOCKS - 3;
    }
    plan.streamed = Q21F_BLOCKS - plan.resident;
    rt.budget = usable;
    fprintf(stderr, "fast: plan %d resident + %d streamed blocks (%.1f MiB each); fixed %.0f, activations %.0f, "
            "K/V %.0f MiB; usable %.0f of %.0f MiB free\n", plan.resident, plan.streamed,
            plan.block_bytes / (double)Q21F_MIB, plan.fixed / (double)Q21F_MIB,
            plan.activations / (double)Q21F_MIB, plan.kv / (double)Q21F_MIB, usable / (double)Q21F_MIB,
            free_bytes / (double)Q21F_MIB);
    if (plan_only) { rc = 0; goto fail; }

    /* Interleave streamed blocks so each copy overlaps resident compute. */
    for (int b = 0; b < Q21F_BLOCKS; b++) {
        int streamed = plan.streamed && ((b + 1) * plan.streamed / Q21F_BLOCKS != b * plan.streamed / Q21F_BLOCKS);
        rt.block[b].resident = !streamed;
        if (streamed) rt.streamed[rt.n_streamed++] = b;
    }

    /* ---- Weights ---- */
    double load_start = q21f_seconds();
    char path[2048];
    for (int i = 1; i <= 2; i++) {
        snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model-%05d-of-00002.safetensors", model, i);
        shards.st[shards.n] = safetensors_open(path);
        REQ(shards.st[shards.n], "cannot open %s", path);
        shards.n++;
    }
    CK(cuMemHostAlloc(&staging, rt.layout.bytes, 0));
    rt.txt_norm_w = q21f_upload_tensor(&rt, &shards, "txt_in.text_norm.weight", Q21F_D, 1, staging);
    rt.txt_in = q21f_upload_tensor(&rt, &shards, "txt_in.in_layer.weight", (size_t)Q21F_D * Q21F_D, 0, staging);
    rt.txt_out = q21f_upload_tensor(&rt, &shards, "txt_in.out_layer.weight", (size_t)Q21F_D * Q21F_D, 0, staging);
    rt.img_in = q21f_upload_tensor(&rt, &shards, "img_in.weight", (size_t)Q21F_D * 64, 0, staging);
    rt.t1 = q21f_upload_tensor(&rt, &shards, "time_text_embed.timestep_embedder.linear_1.weight",
                               (size_t)Q21F_D * 256, 0, staging);
    rt.t2 = q21f_upload_tensor(&rt, &shards, "time_text_embed.timestep_embedder.linear_2.weight",
                               (size_t)Q21F_D * Q21F_D, 0, staging);
    rt.modulation = q21f_upload_tensor(&rt, &shards, "modulation.1.weight", (size_t)4 * Q21F_D * Q21F_D, 0,
                                       staging);
    rt.norm_out = q21f_upload_tensor(&rt, &shards, "norm_out.linear.weight", (size_t)Q21F_D * Q21F_D, 0, staging);
    rt.proj_out = q21f_upload_tensor(&rt, &shards, "proj_out.weight", (size_t)64 * Q21F_D, 0, staging);
    REQ(rt.txt_norm_w && rt.txt_in && rt.txt_out && rt.img_in && rt.t1 && rt.t2 && rt.modulation &&
        rt.norm_out && rt.proj_out, "non-block weights failed to load");
    for (int b = 0; b < Q21F_BLOCKS; b++) {
        if (rt.block[b].resident) {
            REQ(!q21f_pack_block(&shards, &rt.layout, b, (uint8_t *)staging), "block %d", b);
            rt.block[b].dev = q21f_alloc(&rt, rt.layout.bytes);
            REQ(rt.block[b].dev, "block %d allocation", b);
            REQ(!q21f_upload(&rt, rt.block[b].dev, staging, rt.layout.bytes, 1), "block %d upload", b);
        } else {
            CK(cuMemHostAlloc(&rt.block[b].host, rt.layout.bytes, 0));
            REQ(!q21f_pack_block(&shards, &rt.layout, b, (uint8_t *)rt.block[b].host), "block %d", b);
        }
    }
    if (trace) {
        rt.trace_buf = q21f_alloc(&rt, 8192 * 8);
        REQ(rt.trace_buf, "trace buffer");
        CK(cuMemsetD8Async(rt.trace_buf, 0, 8192 * 8, rt.compute));
    }
    if (rt.n_streamed && verify_slots) {
        rt.verify_max = 4096;
        rt.verify = q21f_alloc(&rt, (size_t)rt.verify_max * 8);
        REQ(rt.verify, "verify buffer");
        CK(cuMemsetD8Async(rt.verify, 0, (size_t)rt.verify_max * 8, rt.compute));
    }
    if (rt.n_streamed) {
        for (int s = 0; s < 2; s++) {
            rt.slot[s] = q21f_alloc(&rt, rt.layout.bytes);
            REQ(rt.slot[s], "streaming slot allocation");
            CK(cuEventCreate(&rt.loaded[s], CU_EVENT_DISABLE_TIMING));
            CK(cuEventCreate(&rt.freed[s], CU_EVENT_DISABLE_TIMING));
        }
        REQ(!q21f_queue_copy(&rt, 0) && !q21f_queue_copy(&rt, 1), "initial block copies");
    }
    for (int i = 0; i < shards.n; i++) { safetensors_close(shards.st[i]); shards.st[i] = NULL; }
    shards.n = 0;
    fprintf(stderr, "fast: weights ready in %.2f s (%.0f MiB device)\n", q21f_seconds() - load_start,
            rt.allocated / (double)Q21F_MIB);

    /* ---- State ---- */
    {
        size_t D = Q21F_D, F = Q21F_F, R = (size_t)st.rows_max;
#define SA(field, bytes) do { st.field = q21f_alloc(&rt, (bytes)); REQ(st.field, "state %s", #field); } while (0)
        SA(hidden, R * D * 2); SA(x, R * D * 2); SA(qkv, R * 3 * D * 2); SA(q, R * D * 2); SA(attn, R * D * 2);
        SA(y, R * D * 2); SA(gp, R * 2 * F * 2); SA(prompt_f32, (size_t)nt_max * D * 4);
        SA(cond_bf16, (size_t)(st.nc > 0 ? st.nc : 1) * 64 * 2);
        SA(te, 512 * 2); SA(temb, 2 * D * 2); SA(temb2, 2 * D * 2); SA(tsilu, 2 * D * 2); SA(mod, 8 * D * 2);
        SA(modc, 8 * D * 4); SA(scale, 2 * D * 2); SA(fscale, 2 * D * 4);
        SA(latent, (size_t)N * 64 * 2); SA(sample, (size_t)N * 64 * 4); SA(pred, (size_t)N * nb * 64 * 2);
#undef SA
        for (int i = 0; i < nb; i++) {
            size_t P = (size_t)br[i].prefix;
            br[i].cache_k = q21f_alloc(&rt, Q21F_BLOCKS * P * D * 2);
            br[i].cache_v = q21f_alloc(&rt, Q21F_BLOCKS * P * D * 2);
            br[i].work_k = q21f_alloc(&rt, (P + N) * D * 2);
            br[i].work_v = q21f_alloc(&rt, (P + N) * D * 2);
            br[i].prefix_hidden = q21f_alloc(&rt, P * D * 2);
            br[i].rope = q21f_alloc(&rt, (P + N) * 128 * 4);
            REQ(br[i].cache_k && br[i].cache_v && br[i].work_k && br[i].work_v && br[i].prefix_hidden &&
                br[i].rope, "branch %d K/V allocation", i);
            int *pos = q21f_positions(&br[i], ih, iw);
            REQ(pos && q21f_valid_rope(&rope, pos, (int)P + N), "RoPE positions outside the table");
            float *table = (float *)malloc((P + N) * 128 * sizeof(float));
            REQ(table, "RoPE table allocation");
            q21f_rope_rows(table, rope.data, pos, (int)P + N);
            /* Pageable async copies are staged before returning, so the
             * table can be freed right away. */
            int e = q21f_upload(&rt, br[i].rope, table, (P + N) * 128 * 4, 0);
            free(table);
            free(pos);
            REQ(!e, "RoPE upload");
        }
    }
    CK(cuMemHostAlloc((void **)&pinned_te, (size_t)(steps + 1) * 512 * 2, 0));
    host_out = (float *)malloc((size_t)N * 64 * sizeof(float));
    host_pred = (uint16_t *)malloc((size_t)N * nb * 64 * sizeof(uint16_t));
    sigmas = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    REQ(host_out && host_pred && sigmas, "host allocation");
    fprintf(stderr, "fast: device allocations %.0f MiB (peak %.0f MiB)\n", rt.allocated / (double)Q21F_MIB,
            rt.peak / (double)Q21F_MIB);

    /* ---- Sampling ---- */
    if (manual_t >= 0.0f) { sigmas[0] = manual_t; sigmas[1] = 0.0f; }
    else qimg21_flow_sigmas(steps, N, sigmas);
    REQ(!q21f_upload(&rt, st.sample, la.data, (size_t)N * 64 * 4, 0), "latent upload");
    { int n = N * 64; void *a[] = {&st.latent, &st.sample, &n};
      REQ(!q21f_launch_n(&rt, rt.cast_bf16, (size_t)n, a), "latent cast"); }
    if (dump_dir) mkdir(dump_dir, 0755);
    if (pred_dir) mkdir(pred_dir, 0755);

    double prefill_start = q21f_seconds();
    for (int i = 0; i < nb; i++) REQ(!q21f_embed_prefix(&rt, &st, &br[i]), "prefix embedding %d", i);
    if (kv_cache && !extract) {
        /* The zero-timestep modulation row is step independent. */
        REQ(!q21f_time(&rt, &st, 0.0f, pinned_te + (size_t)steps * 512), "time embedding");
        for (int i = 0; i < nb; i++)
            REQ(!q21f_pass(&rt, &st, &br[i], 1, Q21F_PREFIX, 1, 0), "prefix pass %d", i);
    }
    CK(cuStreamSynchronize(rt.compute));
    double prefill_s = q21f_seconds() - prefill_start;
    CUevent ev0 = NULL, ev1 = NULL;
    if (profile) { CK(cuEventCreate(&ev0, 0)); CK(cuEventCreate(&ev1, 0)); }
    double loop_start = q21f_seconds();
    for (int s = 0; s < steps; s++) {
        float model_t = manual_t >= 0.0f ? manual_t :
            qimg21_round_bf16_host(qimg21_round_bf16_host(sigmas[s] * 1000.0f) / 1000.0f);
        if (profile) CK(cuEventRecord(ev0, rt.compute));
        REQ(!q21f_time(&rt, &st, model_t, pinned_te + (size_t)s * 512), "time embedding");
        if (!kv_cache || (extract && s == 0)) {
            /* Joint [prefix; target] pass per branch, as the uncached
             * reference computes every step; with the cache on, this is the
             * extract step that stores the prefix K/V. */
            for (int i = 0; i < nb; i++)
                REQ(!q21f_pass(&rt, &st, &br[i], 1, Q21F_JOINT, kv_cache, st.pred + (size_t)i * N * 64 * 2),
                    "joint pass");
        } else if (batch == nb) REQ(!q21f_pass(&rt, &st, br, nb, Q21F_TARGET, 0, st.pred), "decode");
        else
            for (int i = 0; i < nb; i++)
                REQ(!q21f_pass(&rt, &st, &br[i], 1, Q21F_TARGET, 0, st.pred + (size_t)i * N * 64 * 2), "decode");
        int n = N * 64, cfg = nb > 1;
        CUdeviceptr neg = st.pred + (size_t)N * 64 * 2;
        if (pred_dir) {
            if (cfg) { void *a[] = {&st.pred, &neg, &n, &guidance};
                       REQ(!q21f_launch_n(&rt, rt.cfg_combine, (size_t)n, a), "cfg"); }
            CK(cuMemcpyDtoHAsync(host_pred, st.pred, (size_t)n * 2, rt.compute));
            CK(cuStreamSynchronize(rt.compute));
            for (int j = 0; j < n; j++) { uint32_t u = (uint32_t)host_pred[j] << 16; memcpy(&host_out[j], &u, 4); }
            snprintf(path, sizeof(path), "%s/pred_%03d.npy", pred_dir, s);
            REQ(!npy_write_f32(path, host_out, (size_t)n, N, 64), "write %s", path);
            cfg = 0; /* already combined in place */
        }
        float dt = sigmas[s + 1] - sigmas[s];
        { void *a[] = {&st.sample, &st.latent, &st.pred, &neg, &n, &guidance, &dt, &cfg};
          REQ(!q21f_launch_n(&rt, rt.euler, (size_t)n, a), "euler"); }
        if (profile) {
            float ms = 0;
            CK(cuEventRecord(ev1, rt.compute));
            CK(cuEventSynchronize(ev1));
            CK(cuEventElapsedTime(&ms, ev0, ev1));
            fprintf(stderr, "fast: step %d/%d sigma=%.7f %.1f ms\n", s + 1, steps, sigmas[s], ms);
        } else if (verbose >= 1)
            fprintf(stderr, "fast: step %d/%d sigma=%.7f\n", s + 1, steps, sigmas[s]);
        if (dump_dir) {
            CK(cuMemcpyDtoHAsync(host_out, st.sample, (size_t)n * 4, rt.compute));
            CK(cuStreamSynchronize(rt.compute));
            snprintf(path, sizeof(path), "%s/step_%03d.npy", dump_dir, s);
            REQ(!npy_write_f32(path, host_out, (size_t)n, N, 64), "write %s", path);
        }
    }
    CK(cuMemcpyDtoHAsync(host_out, st.sample, (size_t)N * 64 * 4, rt.compute));
    CK(cuStreamSynchronize(rt.compute));
    double loop_s = q21f_seconds() - loop_start;
    REQ(!npy_write_f32(out_path, host_out, (size_t)N * 64, N, 64), "write %s", out_path);
    if (rt.trace_buf) {
        unsigned long long *sums = (unsigned long long *)calloc(8192, 8);
        REQ(sums, "trace host buffer");
        CK(cuMemcpyDtoH(sums, rt.trace_buf, 8192 * 8));
        for (int i = 0; i < rt.trace_n; i++) fprintf(stderr, "trace %s %016llx\n", rt.trace_tag[i], sums[i]);
        free(sums);
    }
    if (rt.verify) {
        unsigned long long uses = rt.use < (unsigned long long)rt.verify_max ? rt.use : (unsigned long long)rt.verify_max;
        unsigned long long *got = (unsigned long long *)calloc(uses ? uses : 1, 8), expect[Q21F_BLOCKS];
        REQ(got, "verify host buffer");
        CK(cuMemcpyDtoH(got, rt.verify, uses * 8));
        for (int k = 0; k < rt.n_streamed; k++) {
            const unsigned *p = (const unsigned *)rt.block[rt.streamed[k]].host;
            unsigned long long words = rt.layout.bytes / 4, sum = 0;
            for (unsigned t = 0; t < 256; t++)
                for (unsigned long long i = (unsigned long long)t * 4099; i < words; i += 256ull * 4099) sum += p[i];
            expect[k] = sum;
        }
        int bad = 0;
        for (unsigned long long u = 0; u < uses; u++)
            if (got[u] != expect[u % (unsigned)rt.n_streamed]) {
                if (bad++ < 20)
                    fprintf(stderr, "fast: VERIFY slot use %llu (block %d) checksum mismatch\n", u,
                            rt.streamed[u % (unsigned)rt.n_streamed]);
            }
        fprintf(stderr, "fast: verified %llu streamed slot uses, %d mismatches\n", uses, bad);
        free(got);
    }
    size_t free_after = 0;
    cuMemGetInfo(&free_after, &total_bytes);
    fprintf(stderr, "fast: prefill %.3f s, %d steps %.3f s (%.3f s/step), device allocations peak %.0f MiB, "
            "device free %.0f MiB\n", prefill_s, steps, loop_s, loop_s / steps, rt.peak / (double)Q21F_MIB,
            free_after / (double)Q21F_MIB);
    fprintf(stderr, "fast: wrote %s (%d tokens x 64, %d steps)\n", out_path, N, steps);
    rc = 0;
fail:
    if (rt.compute) cuStreamSynchronize(rt.compute);
    if (rt.copy) cuStreamSynchronize(rt.copy);
    for (int i = 0; i < shards.n; i++) if (shards.st[i]) safetensors_close(shards.st[i]);
    for (int b = 0; b < Q21F_BLOCKS; b++) {
        if (rt.block[b].dev) cuMemFree(rt.block[b].dev);
        if (rt.block[b].host) cuMemFreeHost(rt.block[b].host);
    }
    for (int i = 0; i < 2; i++) q21_layout_free(&br[i].layout);
    if (staging) cuMemFreeHost(staging);
    if (pinned_te) cuMemFreeHost(pinned_te);
    if (rt.plugin) dlclose(rt.plugin);
    free(host_out); free(host_pred); free(sigmas);
    npy_free(&pe); npy_free(&ne); npy_free(&la); npy_free(&cond); npy_free(&rope);
    return rc;
}
