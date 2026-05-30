/*
 * test_cuda_svdquant.c — CUDA SVDQuant forward validation against the PyTorch
 * reference dumped by ref/svdquant/gen_svdquant_ref.py.
 *
 * Paths:
 *   INT4 (w4a16 + w4a4) and NVFP4 w4a16 : host-decode the 4-bit residual weight
 *       to f32, then run the forward as pedantic FP32 cuBLAS GEMMs
 *       (residual + the two rank-128 low-rank GEMMs). This is a correctness
 *       test, not a perf path — there is no INT4 tensor-core kernel needed.
 *   NVFP4 w4a4 : the native HW path — cuda/fp4_w4a4.h's fp4_w4a4_gemm runs the
 *       sm_120a block-scale mma.sync on the residual (quantizing the activation
 *       on-device); the low-rank branch + bias are added separately. Skipped
 *       gracefully if the kernel will not compile (non-sm_120a GPU).
 *
 * Gates rel_L2(impl, y_svdq): SGEMM cases <= 2e-4 (f32 reduction order);
 * NVFP4-w4a4 <= 1e-2 (the kernel re-quantizes activations vs numpy).
 *
 * Build: gcc -O2 -I.. -o test_cuda_svdquant test_cuda_svdquant.c \
 *            ../cuew.c ../cublasew.c -ldl -lm
 * Usage: ./test_cuda_svdquant [dump_dir]   (default ../../ref/svdquant/dumps)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuew.h"
#include "../cublasew.h"
#include "../fp4_w4a4.h"

#include "npy_io.h"
#include "svdquant_cpu.h"   /* host decode/quant helpers, shared with the CPU test */

#define CHECK_CUDA(call) do { CUresult e = (call); if (e != CUDA_SUCCESS) { \
    const char *s; cuGetErrorString(e, &s); \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, s); exit(1); } } while (0)

static const char *g_dir;

static void *load(const char *name, int *ndim, int *dims, int *is_f32) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", g_dir, name);
    void *p = npy_load(path, ndim, dims, is_f32);
    if (!p) { fprintf(stderr, "FATAL: cannot load %s\n", path); exit(2); }
    return p;
}
static float *load_f32(const char *n)  { int a, b[8], f; void *p = load(n, &a, b, &f); if (!f) { fprintf(stderr, "%s not f32\n", n); exit(2);} return (float*)p; }
static uint8_t *load_u8(const char *n) { int a, b[8], f; return (uint8_t*)load(n, &a, b, &f); }
static int32_t *load_i32(const char *n){ int a, b[8], f; return (int32_t*)load(n, &a, b, &f); }
static const char *key(const char *p, const char *s) { static char b[256]; snprintf(b, sizeof(b), "%s%s", p, s); return b; }

static double rel_l2(const float *a, const float *b, int n) {
    double dn = 0, bn = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; dn += d*d; bn += (double)b[i]*b[i]; }
    return sqrt(dn) / (sqrt(bn) + 1e-30);
}
static double cosine(const float *a, const float *b, int n) {
    double ab = 0, aa = 0, bb = 0;
    for (int i = 0; i < n; i++) { ab += (double)a[i]*b[i]; aa += (double)a[i]*a[i]; bb += (double)b[i]*b[i]; }
    return ab / (sqrt(aa) * sqrt(bb) + 1e-30);
}

static CUdeviceptr up_f32(const float *h, size_t n) {
    CUdeviceptr d; CHECK_CUDA(cuMemAlloc(&d, n * 4)); CHECK_CUDA(cuMemcpyHtoD(d, h, n * 4)); return d;
}

/* globals shared across cases */
static int OUT, IN, TOK, RANK;
static cublasew_context *blas;
static fp4_w4a4_ctx fp4ctx;
static int have_fp4;
static float *g_x, *g_bias, *g_yfp;

/* pedantic Y[m,n_out] = X[m,n_in] @ W[n_out,n_in]^T */
static void gemm_nt(CUdeviceptr Y, CUdeviceptr W, CUdeviceptr X, int m, int n_out, int n_in) {
    if (cublasew_gemm_f32_pedantic_rowmajor_nt(blas, Y, W, X, m, n_out, n_in) != 0) {
        fprintf(stderr, "cublasew gemm failed\n"); exit(1);
    }
}

/* Compute the low-rank term lo[TOK,OUT] = (x @ ld^T) @ lu^T on device, copy to host. */
static void lowrank_host(const float *lu, const float *ld, float *lo_host) {
    CUdeviceptr d_x = up_f32(g_x, (size_t)TOK * IN);
    CUdeviceptr d_ld = up_f32(ld, (size_t)RANK * IN);
    CUdeviceptr d_lu = up_f32(lu, (size_t)OUT * RANK);
    CUdeviceptr d_la, d_lo;
    CHECK_CUDA(cuMemAlloc(&d_la, (size_t)TOK * RANK * 4));
    CHECK_CUDA(cuMemAlloc(&d_lo, (size_t)TOK * OUT * 4));
    gemm_nt(d_la, d_ld, d_x, TOK, RANK, IN);
    gemm_nt(d_lo, d_lu, d_la, TOK, OUT, RANK);
    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(lo_host, d_lo, (size_t)TOK * OUT * 4));
    cuMemFree(d_x); cuMemFree(d_ld); cuMemFree(d_lu); cuMemFree(d_la); cuMemFree(d_lo);
}

static void assemble(float *y, const float *resid, const float *lo) {
    for (int t = 0; t < TOK; t++)
        for (int o = 0; o < OUT; o++) {
            size_t i = (size_t)t * OUT + o;
            y[i] = resid[i] + lo[i] + g_bias[o];
        }
}

/* SGEMM path: decode residual to f32 (R), residual GEMM with x_act, add low-rank+bias. */
static void run_sgemm_case(const char *pf, const float *R, const float *x_act,
                           const float *lu, const float *ld, float *y) {
    CUdeviceptr d_R = up_f32(R, (size_t)OUT * IN);
    CUdeviceptr d_xa = up_f32(x_act, (size_t)TOK * IN);
    CUdeviceptr d_resid;
    CHECK_CUDA(cuMemAlloc(&d_resid, (size_t)TOK * OUT * 4));
    gemm_nt(d_resid, d_R, d_xa, TOK, OUT, IN);
    CHECK_CUDA(cuCtxSynchronize());
    float *resid = (float *)malloc((size_t)TOK * OUT * 4);
    float *lo = (float *)malloc((size_t)TOK * OUT * 4);
    CHECK_CUDA(cuMemcpyDtoH(resid, d_resid, (size_t)TOK * OUT * 4));
    lowrank_host(lu, ld, lo);
    assemble(y, resid, lo);
    free(resid); free(lo);
    cuMemFree(d_R); cuMemFree(d_xa); cuMemFree(d_resid);
    (void)pf;
}

/* Native HW path for NVFP4-w4a4: fp4_w4a4_gemm residual + host low-rank + bias. */
static void run_fp4_case(const float *xr, CUdeviceptr d_qw, CUdeviceptr d_ws, CUdeviceptr d_wcwt,
                         const float *lu, const float *ld, float *y) {
    CUdeviceptr d_xr = up_f32(xr, (size_t)TOK * IN);
    CUdeviceptr d_resid;
    CHECK_CUDA(cuMemAlloc(&d_resid, (size_t)TOK * OUT * 4));
    CHECK_CUDA(cuMemsetD8(d_resid, 0, (size_t)TOK * OUT * 4));
    /* residual = quant(xr) @ R_nvfp4^T * wcwt  (kernel quantizes xr internally) */
    fp4_w4a4_gemm(&fp4ctx, d_resid, d_qw, d_ws, d_wcwt, d_xr, /*bias*/0, OUT, IN, TOK);
    CHECK_CUDA(cuCtxSynchronize());
    float *resid = (float *)malloc((size_t)TOK * OUT * 4);
    float *lo = (float *)malloc((size_t)TOK * OUT * 4);
    CHECK_CUDA(cuMemcpyDtoH(resid, d_resid, (size_t)TOK * OUT * 4));
    lowrank_host(lu, ld, lo);
    assemble(y, resid, lo);
    free(resid); free(lo);
    cuMemFree(d_xr); cuMemFree(d_resid);
}

int main(int argc, char **argv) {
    g_dir = (argc > 1) ? argv[1] : "../../ref/svdquant/dumps";

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    CHECK_CUDA(cuInit(0));
    CUdevice dev; CUcontext cctx;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CHECK_CUDA(cuCtxCreate(&cctx, 0, dev));
    if (cublasewInit() != 0 || cublasewCreate(&blas, 0) != 0) {
        fprintf(stderr, "cublasew init failed\n"); return 1;
    }
    have_fp4 = (fp4_w4a4_compile(&fp4ctx, /*stream*/0, /*verbose*/0) == 0);

    int nd, dms[8], f32;
    int32_t *dimv = (int32_t *)load("dims", &nd, dms, &f32);
    OUT = dimv[0]; IN = dimv[1]; TOK = dimv[2]; RANK = dimv[3]; free(dimv);
    printf("svdquant CUDA test  dir=%s  OUT=%d IN=%d TOK=%d RANK=%d  fp4_hw=%s\n",
           g_dir, OUT, IN, TOK, RANK, have_fp4 ? "yes" : "no(skip nvfp4_w4a4)");

    g_x = load_f32("x"); g_bias = load_f32("bias"); g_yfp = load_f32("y_fp");

    float *R = (float *)malloc((size_t)OUT * IN * 4);
    float *xr = (float *)malloc((size_t)TOK * IN * 4);
    float *xact = (float *)malloc((size_t)TOK * IN * 4);
    float *y = (float *)malloc((size_t)TOK * OUT * 4);

    struct { const char *pf, *fmt, *scope; double gate; } cs[] = {
        {"int4_w4a16",  "int4",  "w4a16", 2e-4},
        {"int4_w4a4",   "int4",  "w4a4",  2e-4},
        {"nvfp4_w4a16", "nvfp4", "w4a16", 2e-4},
        {"nvfp4_w4a4",  "nvfp4", "w4a4",  1e-2},
    };
    int ncs = (int)(sizeof(cs) / sizeof(cs[0])), fail = 0;

    for (int ci = 0; ci < ncs; ci++) {
        const char *pf = cs[ci].pf;
        int is_int4 = !strcmp(cs[ci].fmt, "int4");
        int is_a16  = !strcmp(cs[ci].scope, "w4a16");
        int is_fp4_hw = (!is_int4 && !is_a16);

        if (is_fp4_hw && !have_fp4) {
            printf("  %-12s SKIP (sm_120a fp4 kernel unavailable)\n", pf);
            continue;
        }

        float *smooth = load_f32(key(pf, "_smooth"));
        float *lu = load_f32(key(pf, "_lora_up"));
        float *ld = load_f32(key(pf, "_lora_down"));
        float *y_svdq = load_f32(key(pf, "_y_svdq"));

        sq_smooth_div(g_x, smooth, xr, TOK, IN);

        CUdeviceptr d_qw = 0, d_ws = 0, d_wcwt = 0;
        if (is_int4) {
            uint8_t *qint4 = load_u8(key(pf, "_qint4"));
            float *wscale = load_f32(key(pf, "_wscale"));
            sq_unpack_int4_residual(qint4, wscale, R, OUT, IN, 64);
            free(qint4); free(wscale);
        } else {
            int32_t *qw = load_i32(key(pf, "_qw"));
            uint8_t *ws = load_u8(key(pf, "_ws"));
            float *wcwt = load_f32(key(pf, "_wcwt"));
            sq_unpack_nvfp4_residual(qw, ws, wcwt, R, OUT, IN, 16);
            if (is_fp4_hw) {   /* keep packed weights on device for the HW kernel */
                d_qw = up_f32((const float *)qw, (size_t)OUT * (IN / 8));  /* i32 bytes, reuse helper */
                CHECK_CUDA(cuMemAlloc(&d_ws, (size_t)OUT * (IN / 16)));
                CHECK_CUDA(cuMemcpyHtoD(d_ws, ws, (size_t)OUT * (IN / 16)));
                d_wcwt = up_f32(wcwt, (size_t)OUT);
            }
            free(qw); free(ws); free(wcwt);
        }

        if (is_fp4_hw) {
            run_fp4_case(xr, d_qw, d_ws, d_wcwt, lu, ld, y);
            cuMemFree(d_qw); cuMemFree(d_ws); cuMemFree(d_wcwt);
        } else {
            if (is_a16) memcpy(xact, xr, (size_t)TOK * IN * 4);
            else        sq_quant_act_int4_g64(xr, xact, TOK, IN, 64);
            run_sgemm_case(pf, R, xact, lu, ld, y);
        }

        double rl = rel_l2(y, y_svdq, TOK * OUT);
        double csim = cosine(y, y_svdq, TOK * OUT);
        double mx = (double)npy_max_abs_f32(y, y_svdq, TOK * OUT, NULL);
        double rl_fp = rel_l2(y, g_yfp, TOK * OUT);
        int ok = (rl <= cs[ci].gate);
        if (!ok) fail = 1;
        printf("  %-12s rel_L2(svdq)=%.3e cos=%.7f max|d|=%.3e | rel_L2(fp)=%.4f  [gate %.0e]  %s\n",
               pf, rl, csim, mx, rl_fp, cs[ci].gate, ok ? "PASS" : "FAIL");

        free(smooth); free(lu); free(ld); free(y_svdq);
    }

    free(R); free(xr); free(xact); free(y); free(g_x); free(g_bias); free(g_yfp);
    if (have_fp4) fp4_w4a4_free(&fp4ctx);
    printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
    return fail;
}
