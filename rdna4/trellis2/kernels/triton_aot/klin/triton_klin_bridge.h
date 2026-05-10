/* Triton klin (Linear) bridge: load AOT-extracted Triton hsacos for the
 * 8 tex_dec klin shapes and dispatch matmul+bias calls to them.
 *
 *   Y[M,N] f32 = (X[M,K] bf16 @ W[N,K] bf16.T) + bias[N] f32
 *
 * Shapes are keyed by (K, N) — unique across the 8 tex_dec klin ops.
 * M is a runtime arg; the kernel masks rm < M, so chunked calls (M ≤ M_max)
 * hit the same compiled binary.
 *
 * Single-header. Define TRITON_KLIN_BRIDGE_IMPL in one .c file.
 *
 * API:
 *   int  t2_klin_init(const char *kernels_dir);   // scans dir for tagged shapes
 *   int  t2_klin_has_shape(int K, int N);
 *   int  t2_klin_has_shape_silu(int K, int N);    // 1 iff fused bias+silu variant present
 *   int  t2_klin_run(int M, int K, int N, ... );  // bias-only
 *   int  t2_klin_run_silu(int M, int K, int N, ...); // bias+silu fused
 *        Returns 0 on launch success; -1 if (K,N) not registered.
 *   void t2_klin_release(void);
 */
#ifndef TRITON_KLIN_BRIDGE_H
#define TRITON_KLIN_BRIDGE_H

#ifdef TRITON_BRIDGE_USE_ROCEW
#include "../../../rocew.h"
#else
#include <hip/hip_runtime.h>
#endif

int  t2_klin_init(const char *kernels_dir);
int  t2_klin_has_shape(int K, int N);
int  t2_klin_has_shape_silu(int K, int N);
int  t2_klin_run(int M, int K, int N,
                 const void *d_x_bf16, const void *d_w_bf16,
                 const void *d_bias_f32, void *d_y_f32,
                 hipStream_t stream);
int  t2_klin_run_silu(int M, int K, int N,
                      const void *d_x_bf16, const void *d_w_bf16,
                      const void *d_bias_f32, void *d_y_f32,
                      hipStream_t stream);
void t2_klin_release(void);

#ifdef TRITON_KLIN_BRIDGE_IMPL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    int K, N;
    int M_max;
    int BM, BN, BK;
    int num_warps;
    int shared;
    int silu;           /* 0 = bias-only, 1 = bias+silu fused epilogue */
    hipModule_t mod;
    hipFunction_t kfn;
    void *d_zero_bias;  /* allocated lazily if caller passes NULL bias */
} t2_klin_shape;

#define T2_KLIN_MAX_SHAPES 16
static t2_klin_shape g_kshapes[T2_KLIN_MAX_SHAPES];
static int g_n_kshapes = 0;

static const char *_kl_err_str(hipError_t e) {
#ifdef TRITON_BRIDGE_USE_ROCEW
    static char buf[32]; snprintf(buf, sizeof buf, "hipError=%d", (int)e);
    return buf;
#else
    return hipGetErrorString(e);
#endif
}

/* Scan a single shape dir: kernels/<tag>/{kernel.hsaco,kernel.json}.
 * Tile params come from shapes.json (parsed by caller; here we accept them
 * directly via _kl_load_one). */
static int _kl_load_one(const char *kernels_dir, const char *tag,
                        int M_max, int K, int N,
                        int BM, int BN, int BK, int num_warps, int shared,
                        int silu, t2_klin_shape *out)
{
    char p[1024];
    snprintf(p, sizeof p, "%s/%s/kernel.hsaco", kernels_dir, tag);
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "t2_klin: open %s failed\n", p); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz); fread(buf, 1, sz, f); fclose(f);
    hipError_t e = hipModuleLoadData(&out->mod, buf); free(buf);
    if (e != hipSuccess) {
        fprintf(stderr, "t2_klin: hipModuleLoadData(%s) %s\n", tag, _kl_err_str(e));
        return -1;
    }
    const char *fname = silu ? "klin_bf16_silu_kernel" : "klin_bf16_kernel";
    e = hipModuleGetFunction(&out->kfn, out->mod, fname);
    if (e != hipSuccess) {
        fprintf(stderr, "t2_klin: hipModuleGetFunction(%s, %s) %s\n",
                tag, fname, _kl_err_str(e));
        return -1;
    }
    out->K = K; out->N = N; out->M_max = M_max;
    out->BM = BM; out->BN = BN; out->BK = BK;
    out->num_warps = num_warps; out->shared = shared;
    out->silu = silu;
    out->d_zero_bias = NULL;
    return 0;
}

/* shapes.json parser: extract { tag, M_max, K, N, BM, BN, BK, num_warps, shared } per object. */
static int _kl_json_int(const char *blk, const char *key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(blk, pat); if (!p) return -1;
    p += strlen(pat); while (*p == ' ') p++;
    return atoi(p);
}
static int _kl_json_str(const char *blk, const char *key, char *out, int outsz) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(blk, pat); if (!p) return -1;
    p = strchr(p + strlen(pat), '"'); if (!p) return -1;
    p++; const char *q = strchr(p, '"'); if (!q) return -1;
    int n = q - p; if (n >= outsz) n = outsz - 1;
    memcpy(out, p, n); out[n] = 0;
    return 0;
}

int t2_klin_init(const char *kernels_dir)
{
    char jpath[1024];
    snprintf(jpath, sizeof jpath, "%s/../shapes.json", kernels_dir);
    FILE *f = fopen(jpath, "rb");
    if (!f) {
        /* allow shapes.json in same dir as kernels root */
        snprintf(jpath, sizeof jpath, "%s/shapes.json", kernels_dir);
        f = fopen(jpath, "rb");
    }
    if (!f) { fprintf(stderr, "t2_klin: shapes.json not found near %s\n", kernels_dir); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *js = (char *)malloc(sz + 1); fread(js, 1, sz, f); js[sz] = 0; fclose(f);

    /* Walk top-level objects: scan for '{' ... '}' pairs. */
    const char *p = js;
    while ((p = strchr(p, '{')) && g_n_kshapes < T2_KLIN_MAX_SHAPES) {
        const char *q = strchr(p, '}'); if (!q) break;
        int n = q - p + 1;
        char *blk = (char *)malloc(n + 1);
        memcpy(blk, p, n); blk[n] = 0;
        char tag[128];
        if (_kl_json_str(blk, "tag", tag, sizeof tag) == 0) {
            int Mm = _kl_json_int(blk, "M_max");
            int K  = _kl_json_int(blk, "K");
            int N  = _kl_json_int(blk, "N");
            int BM = _kl_json_int(blk, "BM");
            int BN = _kl_json_int(blk, "BN");
            int BK = _kl_json_int(blk, "BK");
            int nw = _kl_json_int(blk, "num_warps");
            int sh = _kl_json_int(blk, "shared");
            int silu = _kl_json_int(blk, "silu");
            if (silu < 0) silu = 0;
            if (K > 0 && N > 0 && BM > 0 && BN > 0 && BK > 0 && nw > 0) {
                if (_kl_load_one(kernels_dir, tag, Mm, K, N, BM, BN, BK, nw, sh,
                                 silu, &g_kshapes[g_n_kshapes]) == 0) {
                    fprintf(stderr, "t2_klin: %-22s  K=%d N=%d  tile=%dx%dx%d nw=%d shared=%d silu=%d\n",
                            tag, K, N, BM, BN, BK, nw, sh, silu);
                    g_n_kshapes++;
                }
            }
        }
        free(blk);
        p = q + 1;
    }
    free(js);
    fprintf(stderr, "t2_klin: %d shapes registered\n", g_n_kshapes);
    return g_n_kshapes > 0 ? 0 : -1;
}

static t2_klin_shape *_kl_find2(int K, int N, int silu) {
    for (int i = 0; i < g_n_kshapes; i++) {
        if (g_kshapes[i].K == K && g_kshapes[i].N == N
            && g_kshapes[i].silu == silu) return &g_kshapes[i];
    }
    return NULL;
}

int t2_klin_has_shape(int K, int N) { return _kl_find2(K, N, 0) != NULL; }
int t2_klin_has_shape_silu(int K, int N) { return _kl_find2(K, N, 1) != NULL; }

static int _t2_klin_run_impl(t2_klin_shape *s, int M, int K, int N,
                             const void *d_x_bf16, const void *d_w_bf16,
                             const void *d_bias_f32, void *d_y_f32,
                             hipStream_t stream);

int t2_klin_run(int M, int K, int N,
                const void *d_x_bf16, const void *d_w_bf16,
                const void *d_bias_f32, void *d_y_f32,
                hipStream_t stream)
{
    t2_klin_shape *s = _kl_find2(K, N, 0);
    if (!s) return -1;
    return _t2_klin_run_impl(s, M, K, N, d_x_bf16, d_w_bf16, d_bias_f32, d_y_f32, stream);
}

int t2_klin_run_silu(int M, int K, int N,
                     const void *d_x_bf16, const void *d_w_bf16,
                     const void *d_bias_f32, void *d_y_f32,
                     hipStream_t stream)
{
    t2_klin_shape *s = _kl_find2(K, N, 1);
    if (!s) return -1;
    return _t2_klin_run_impl(s, M, K, N, d_x_bf16, d_w_bf16, d_bias_f32, d_y_f32, stream);
}

static int _t2_klin_run_impl(t2_klin_shape *s, int M, int K, int N,
                             const void *d_x_bf16, const void *d_w_bf16,
                             const void *d_bias_f32, void *d_y_f32,
                             hipStream_t stream)
{

    /* Caller may chunk M arbitrarily; kernel masks handle M < BM at the tail. */
    if (!d_bias_f32) {
        if (!s->d_zero_bias) {
            if (hipMalloc(&s->d_zero_bias, (size_t)N * sizeof(float)) != hipSuccess) return -1;
            hipMemset(s->d_zero_bias, 0, (size_t)N * sizeof(float));
        }
        d_bias_f32 = s->d_zero_bias;
    }

    int grid_x = (M + s->BM - 1) / s->BM;
    int grid_y = (N + s->BN - 1) / s->BN;
    int block  = s->num_warps * 32;
    /* Kernel signature (constexpr strides baked in):
     *   X_ptr, W_ptr, B_ptr, Y_ptr, M, N, K        (constexpr: sxm,sxk,swn,swk,sym,syn,BM,BN,BK)
     */
    /* AMDGPU kernarg layout per `llvm-readelf --notes` on the hsaco:
     *   ptrs (X,W,B,Y) | M,N,K (i32 each) | 2 trailing null ptrs
     * The two trailing buffers are LLIR readnone artefacts (same as the
     * spconv kernels) — pass two null ptrs to satisfy kernarg_segment_size. */
    void *null_ptr = NULL;
    void *args[] = {
        (void *)&d_x_bf16, (void *)&d_w_bf16,
        (void *)&d_bias_f32, (void *)&d_y_f32,
        &M, &N, &K,
        &null_ptr, &null_ptr,
    };
    hipError_t e = hipModuleLaunchKernel(s->kfn, grid_x, grid_y, 1,
                                         block, 1, 1,
                                         s->shared, stream, args, NULL);
    if (e != hipSuccess) {
        fprintf(stderr, "t2_klin_run launch fail (M=%d K=%d N=%d): %s\n",
                M, K, N, _kl_err_str(e));
        return -1;
    }
    return 0;
}

void t2_klin_release(void)
{
    for (int i = 0; i < g_n_kshapes; i++) {
        if (g_kshapes[i].d_zero_bias) hipFree(g_kshapes[i].d_zero_bias);
        if (g_kshapes[i].mod) hipModuleUnload(g_kshapes[i].mod);
    }
    g_n_kshapes = 0;
}

#endif /* TRITON_KLIN_BRIDGE_IMPL */
#endif /* TRITON_KLIN_BRIDGE_H */
