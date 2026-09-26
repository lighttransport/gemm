/* Host dispatch for the self-owned RDNA4 WMMA GEMM in gemm_wmma.hip.
 * Shared by hip_llm_runner.c and bench_llm_gemm.c so both run identical
 * kernels and tile selection.  Requires rocew.h (HIP driver API) and
 * gemm_wmma.inc (gemm_wmma_source) to be included first.
 *
 *   Y[M,N] f32 = X[M,K] x W[N,K]^T, X/W row-major BF16 or F16, or W a
 *   GGML-quantized matrix decoded in the tile loader (gemm_wmma_run_q).
 *
 * Env overrides (tuning / A/B only):
 *   LLM_GEMM_WMMA_VARIANT=<index into gemm_wmma_variants>
 *   LLM_GEMM_WMMA_SPLITK=<n>   force split-K factor (1 disables) */
#ifndef GEMM_WMMA_DISPATCH_H
#define GEMM_WMMA_DISPATCH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

enum { GEMM_WMMA_BF16 = 0, GEMM_WMMA_F16 = 1 };

typedef struct {
    const char *name;
    int dtype, bm, bn, threads;
} gemm_wmma_variant;

static const gemm_wmma_variant gemm_wmma_variants[] = {
    { "gemm_wmma_bf16_128x128_w2x2", GEMM_WMMA_BF16, 128, 128, 128 },
    { "gemm_wmma_bf16_128x64_w2x2",  GEMM_WMMA_BF16, 128,  64, 128 },
    { "gemm_wmma_bf16_64x128_w2x2",  GEMM_WMMA_BF16,  64, 128, 128 },
    { "gemm_wmma_bf16_64x64_w2x2",   GEMM_WMMA_BF16,  64,  64, 128 },
    { "gemm_wmma_bf16_128x128_w2x4", GEMM_WMMA_BF16, 128, 128, 256 },
    { "gemm_wmma_bf16_128x256_w2x4", GEMM_WMMA_BF16, 128, 256, 256 },
    { "gemm_wmma_bf16_256x128_w4x2", GEMM_WMMA_BF16, 256, 128, 256 },
    { "gemm_wmma_f16_128x128_w2x2",  GEMM_WMMA_F16,  128, 128, 128 },
    { "gemm_wmma_f16_64x64_w2x2",    GEMM_WMMA_F16,   64,  64, 128 },
    { "gemm_wmma_bf16_32x128_w1x4",  GEMM_WMMA_BF16,  32, 128, 128 },
    { "gemm_wmma_bf16_32x64_w1x4",   GEMM_WMMA_BF16,  32,  64, 128 },
    { "gemm_wmma_bf16_64x128_w1x4",  GEMM_WMMA_BF16,  64, 128, 128 },
};
#define GEMM_WMMA_NVARIANTS ((int)(sizeof(gemm_wmma_variants) / sizeof(gemm_wmma_variants[0])))

/* Fused-dequant weight formats (gemm_wmma_q_<name>_{128x128,64x64}). */
enum {
    GEMM_WMMA_Q_IQ2_XXS, GEMM_WMMA_Q_IQ2_XS, GEMM_WMMA_Q_IQ2_S,
    GEMM_WMMA_Q_IQ3_XXS, GEMM_WMMA_Q_IQ3_S, GEMM_WMMA_Q_IQ1_S,
    GEMM_WMMA_Q_IQ1_M, GEMM_WMMA_Q_IQ4_XS, GEMM_WMMA_Q_Q2_K,
    GEMM_WMMA_Q_NFMT
};
static const char *const gemm_wmma_q_names[GEMM_WMMA_Q_NFMT] = {
    "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m",
    "iq4_xs", "q2_k",
};

/* Fused tile variants: {suffix, bm, bn, threads}. */
typedef struct { const char *suffix; int bm, bn, threads; } gemm_wmma_q_tile;
static const gemm_wmma_q_tile gemm_wmma_q_tiles[] = {
    { "128x128",   128, 128, 128 },
    { "64x64",      64,  64, 128 },
    { "256x128",   256, 128, 256 },
    { "128x128w8", 128, 128, 256 },
    { "256x64",    256,  64, 256 },
};
#define GEMM_WMMA_Q_NTILE ((int)(sizeof(gemm_wmma_q_tiles) / sizeof(gemm_wmma_q_tiles[0])))

typedef struct {
    hipModule_t module;
    hipFunction_t fn[GEMM_WMMA_NVARIANTS];
    hipFunction_t fn_q[GEMM_WMMA_Q_NFMT][GEMM_WMMA_Q_NTILE];
    hipFunction_t fn_reduce;
    int n_cu;
    int force_variant;       /* -1 = auto */
    int force_splitk;        /* 0 = auto */
    int force_qtile;         /* -1 = auto (LLM_GEMM_WMMA_QTILE) */
    void *ws;                /* split-K partials */
    size_t ws_bytes;
    int ws_fixed;            /* 1: never reallocate (runner, graph-safe) */
    void *ws_stream;         /* if set, only this stream may split K */
} gemm_wmma_ctx;

static inline int gemm_wmma_init(gemm_wmma_ctx *c, int device, int verbose) {
    memset(c, 0, sizeof(*c));
    c->force_variant = -1;
    if (hip_compile_kernels_ex(&c->module, device, gemm_wmma_source,
                               "gemm_wmma.hip", verbose, "gemm_wmma", 0) <= 0)
        return -1;
    for (int i = 0; i < GEMM_WMMA_NVARIANTS; i++)
        if (hipModuleGetFunction(&c->fn[i], c->module,
                                 gemm_wmma_variants[i].name) != hipSuccess)
            return -1;
    if (hipModuleGetFunction(&c->fn_reduce, c->module,
                             "gemm_wmma_splitk_reduce") != hipSuccess)
        return -1;
    for (int f = 0; f < GEMM_WMMA_Q_NFMT; f++)
        for (int t = 0; t < GEMM_WMMA_Q_NTILE; t++) {
            char name[64];
            snprintf(name, sizeof(name), "gemm_wmma_q_%s_%s", gemm_wmma_q_names[f],
                     gemm_wmma_q_tiles[t].suffix);
            if (hipModuleGetFunction(&c->fn_q[f][t], c->module, name) != hipSuccess)
                return -1;
        }
    hipDeviceProp_t props;
    c->n_cu = 64;
    if (hipGetDeviceProperties(&props, device) == hipSuccess &&
        props.multiProcessorCount > 0)
        c->n_cu = props.multiProcessorCount;
    /* gfx12 reports WGPs (2 CUs each); split-K sizing wants CUs. */
    if (c->n_cu <= 48) c->n_cu *= 2;
    const char *e = getenv("LLM_GEMM_WMMA_VARIANT");
    if (e && *e) c->force_variant = atoi(e);
    e = getenv("LLM_GEMM_WMMA_SPLITK");
    if (e && *e) c->force_splitk = atoi(e);
    c->force_qtile = -1;
    e = getenv("LLM_GEMM_WMMA_QTILE");
    if (e && *e) c->force_qtile = atoi(e);
    return 0;
}

static inline void gemm_wmma_destroy(gemm_wmma_ctx *c) {
    if (c->ws) hipFree(c->ws);
    if (c->module) hipModuleUnload(c->module);
    memset(c, 0, sizeof(*c));
}

/* Kernel requirements: K % 32 == 0 and 16-byte aligned X/W base pointers
 * (row alignment then follows). */
static inline int gemm_wmma_supported(const void *W, const void *X,
                                      int M, int N, int K) {
    return M > 0 && N > 0 && K >= 32 && (K & 31) == 0 &&
           (((uintptr_t)W | (uintptr_t)X) & 15) == 0;
}

/* Preallocate a fixed split-K workspace owned by one stream (the runner's
 * main stream); other streams run unsplit, so no allocation or sharing
 * happens on the hot path. */
static inline int gemm_wmma_reserve(gemm_wmma_ctx *c, size_t bytes, void *stream) {
    if (c->ws) hipFree(c->ws);
    c->ws = NULL;
    c->ws_bytes = 0;
    if (bytes && hipMalloc(&c->ws, bytes) != hipSuccess) return -1;
    c->ws_bytes = bytes;
    c->ws_fixed = 1;
    c->ws_stream = stream;
    return 0;
}

/* Tile / split-K selection, tuned on gfx1201 (RX 9070 XT, 64 CU) with
 * bench_llm_gemm at the Qwen3.8-27B prefill shapes:
 *   - M >= 128: 128x128 / 4 waves wins every N >= 1024 shape at M=512
 *     (grid walks M fastest so each weight tile is streamed ~once);
 *   - M < 128 or skinny N: 32x64 (M <= 32) or 64x64 tiles; these shapes are
 *     weight-bandwidth bound (tail chunks / SSM alpha,beta N=48);
 *   - split K while the grid has fewer than ~n_cu/2 tiles per split pass,
 *     keeping >= 16 K steps per split (attn K/V N=1024 -> 2, N=48 -> 4). */
static inline void gemm_wmma_select(const gemm_wmma_ctx *c, int dtype, int M, int N,
                             int K, int *variant, int *splitk) {
    int v;
    if (dtype == GEMM_WMMA_F16) v = (M >= 128 && N >= 128) ? 7 : 8;
    else if (M >= 128 && N >= 128) v = 0;
    else if (M <= 32) v = 10;
    else v = 3;
    int s = 1;
    const gemm_wmma_variant *vv = &gemm_wmma_variants[v];
    long tiles = (long)((M + vv->bm - 1) / vv->bm) * ((N + vv->bn - 1) / vv->bn);
    while (tiles * s * 2 <= c->n_cu && K / (32 * s * 2) >= 16 && s < 8) s *= 2;
    if (c->force_variant >= 0 && c->force_variant < GEMM_WMMA_NVARIANTS &&
        gemm_wmma_variants[c->force_variant].dtype == dtype)
        v = c->force_variant;
    if (c->force_splitk > 0) s = c->force_splitk;
    *variant = v;
    *splitk = s;
}

/* Launch fn over a (bm x bn)-tiled grid with split-K `splitk` (>= 1). */
static inline int gemm_wmma_launch(gemm_wmma_ctx *c, hipFunction_t fn, int bm,
                                   int bn, int threads, void *Y, const void *W,
                                   const void *X, int M, int N, int K,
                                   void *stream, int splitk) {
    if (splitk < 1) splitk = 1;
    if (splitk > K / 32) splitk = K / 32;
    void *out = Y;
    size_t part_bytes = (size_t)M * N * sizeof(float);
    if (splitk > 1 && (((uintptr_t)Y & 15) ||
                       (c->ws_stream && stream != c->ws_stream)))
        splitk = 1;
    if (splitk > 1 && c->ws_fixed)
        while (splitk > 1 && part_bytes * splitk > c->ws_bytes) splitk >>= 1;
    if (splitk > 1) {
        size_t need = part_bytes * splitk;
        if (need > c->ws_bytes) {
            if (c->ws) hipFree(c->ws);
            c->ws = NULL;
            c->ws_bytes = 0;
            if (hipMalloc(&c->ws, need) != hipSuccess) return -1;
            c->ws_bytes = need;
        }
        out = c->ws;
    }
    void *args[] = { &out, &W, &X, &N, &K, &M };
    hipError_t err = hipModuleLaunchKernel(fn,
        (unsigned)((M + bm - 1) / bm), (unsigned)((N + bn - 1) / bn),
        (unsigned)splitk, (unsigned)threads, 1, 1, 0, (hipStream_t)stream,
        args, NULL);
    if (err != hipSuccess) {
        fprintf(stderr, "gemm_wmma: launch failed (M=%d N=%d K=%d tile=%dx%d err=%d)\n",
                M, N, K, bm, bn, (int)err);
        return -1;
    }
    if (splitk > 1) {
        long n = (long)M * N;
        void *rargs[] = { &Y, &c->ws, &splitk, &n };
        unsigned blocks = (unsigned)((n / 4 + 255) / 256 + 1);
        err = hipModuleLaunchKernel(c->fn_reduce, blocks, 1, 1, 256, 1, 1, 0,
                                    (hipStream_t)stream, rargs, NULL);
        if (err != hipSuccess) return -1;
    }
    return 0;
}

/* Launch with an explicit variant/split (bench) or auto (variant < 0). */
static inline int gemm_wmma_run_ex(gemm_wmma_ctx *c, int dtype, void *Y, const void *W,
                                   const void *X, int M, int N, int K, void *stream,
                                   int variant, int splitk) {
    if (!gemm_wmma_supported(W, X, M, N, K)) return -1;
    if (variant < 0) gemm_wmma_select(c, dtype, M, N, K, &variant, &splitk);
    const gemm_wmma_variant *vv = &gemm_wmma_variants[variant];
    if (vv->dtype != dtype) return -1;
    return gemm_wmma_launch(c, c->fn[variant], vv->bm, vv->bn, vv->threads,
                            Y, W, X, M, N, K, stream, splitk);
}

/* Fused dequant GEMM: W is a GGML-quantized [N, K] matrix of format `fmt`
 * (row-major, K % 256 == 0), X is BF16.  Split-K follows the dense rule. */
static inline int gemm_wmma_q_supported(int fmt, const void *W, const void *X,
                                        int M, int N, int K) {
    return fmt >= 0 && fmt < GEMM_WMMA_Q_NFMT && M > 0 && N > 0 &&
           K >= 256 && (K & 255) == 0 && (((uintptr_t)W | (uintptr_t)X) & 15) == 0;
}
static inline int gemm_wmma_run_q_ex(gemm_wmma_ctx *c, int fmt, void *Y, const void *W,
                                     const void *X, int M, int N, int K, void *stream,
                                     int tile, int splitk) {
    if (!gemm_wmma_q_supported(fmt, W, X, M, N, K)) return -1;
    /* Tuned with bench_llm_gemm --quant <fmt> over all nine formats at M=512:
     * decode ALU dominates, so fewer M tiles per weight tile (256 rows) and
     * 2 decoded vectors per thread (256 threads) win for wide N; 8-wave
     * 128x128 wins the N=5120 outputs and partial chunks; 64x64 skinny N. */
    if (tile < 0) {
        if (N < 128 || M < 64) tile = 1;                 /* 64x64 */
        else if (M >= 256 && N >= 6144) tile = 2;        /* 256x128 */
        else tile = 3;                                   /* 128x128, 8 waves */
    }
    if (c->force_qtile >= 0 && c->force_qtile < GEMM_WMMA_Q_NTILE) tile = c->force_qtile;
    const gemm_wmma_q_tile *t = &gemm_wmma_q_tiles[tile];
    if (splitk <= 0) {
        long tiles = (long)((M + t->bm - 1) / t->bm) * ((N + t->bn - 1) / t->bn);
        splitk = 1;
        while (tiles * splitk * 2 <= c->n_cu && K / (32 * splitk * 2) >= 16 && splitk < 8)
            splitk *= 2;
        if (c->force_splitk > 0) splitk = c->force_splitk;
    }
    return gemm_wmma_launch(c, c->fn_q[fmt][tile], t->bm, t->bn, t->threads, Y, W, X,
                            M, N, K, stream, splitk);
}
static inline int gemm_wmma_run_q(gemm_wmma_ctx *c, int fmt, void *Y, const void *W,
                                  const void *X, int M, int N, int K, void *stream) {
    return gemm_wmma_run_q_ex(c, fmt, Y, W, X, M, N, K, stream, -1, 0);
}

static inline int gemm_wmma_run(gemm_wmma_ctx *c, int dtype, void *Y, const void *W,
                         const void *X, int M, int N, int K, void *stream) {
    return gemm_wmma_run_ex(c, dtype, Y, W, X, M, N, K, stream, -1, 0);
}

#endif /* GEMM_WMMA_DISPATCH_H */
