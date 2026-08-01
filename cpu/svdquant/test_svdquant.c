/*
 * test_svdquant.c — CPU SVDQuant forward validation against the PyTorch
 * reference dumped by ref/svdquant/gen_svdquant_ref.py.
 *
 * For each of the 4 cases (int4/nvfp4 x w4a16/w4a4) it rebuilds the residual
 * weight, runs sq_forward, and compares to:
 *   - y_svdq  (the PyTorch SVDQuant forward, SAME scheme)  <- GATE: rel_L2
 *   - y_fp    (full-precision x@W^T+b)                     <- reported quant floor
 *
 * Usage: ./test_svdquant [dump_dir]   (default ../../ref/svdquant/dumps)
 * Exit 0 iff every case passes its gate.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "npy_io.h"
#include "svdquant_cpu.h"

static const char *g_dir;

static void *load(const char *name, int *ndim, int *dims, int *is_f32) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", g_dir, name);
    void *p = npy_load(path, ndim, dims, is_f32);
    if (!p) { fprintf(stderr, "FATAL: cannot load %s\n", path); exit(2); }
    return p;
}
static float *load_f32(const char *name) {
    int nd, dims[8], f32;
    void *p = load(name, &nd, dims, &f32);
    if (!f32) { fprintf(stderr, "FATAL: %s not f32\n", name); exit(2); }
    return (float *)p;
}
static uint8_t *load_u8(const char *name) {
    int nd, dims[8], f32;
    return (uint8_t *)load(name, &nd, dims, &f32);
}
static int32_t *load_i32(const char *name) {
    int nd, dims[8], f32;
    return (int32_t *)load(name, &nd, dims, &f32);
}
/* concat case prefix + suffix */
static const char *key(const char *prefix, const char *suffix) {
    static char buf[256];
    snprintf(buf, sizeof(buf), "%s%s", prefix, suffix);
    return buf;
}

static double rel_l2(const float *a, const float *b, int n) {
    double dn = 0.0, bn = 0.0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; dn += d * d; bn += (double)b[i] * b[i]; }
    return sqrt(dn) / (sqrt(bn) + 1e-30);
}
static double cosine(const float *a, const float *b, int n) {
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (int i = 0; i < n; i++) { ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
    return ab / (sqrt(aa) * sqrt(bb) + 1e-30);
}

int main(int argc, char **argv) {
    g_dir = (argc > 1) ? argv[1] : "../../ref/svdquant/dumps";

    int nd, dims[8], f32;
    int32_t *dimv = (int32_t *)load("dims", &nd, dims, &f32);
    int OUT = dimv[0], IN = dimv[1], TOK = dimv[2], RANK = dimv[3];
    free(dimv);
    printf("svdquant CPU test  dir=%s  OUT=%d IN=%d TOK=%d RANK=%d\n", g_dir, OUT, IN, TOK, RANK);

    float *x = load_f32("x");
    float *bias = load_f32("bias");
    float *y_fp = load_f32("y_fp");

    /* gate on rel_L2(impl, y_svdq); same scheme, double-vs-f32 only (~6e-8 floor). */
    const double GATE = 1e-5;

    struct { const char *prefix, *fmt, *scope; } cases[] = {
        {"int4_w4a16", "int4", "w4a16"},
        {"int4_w4a4", "int4", "w4a4"},
        {"nvfp4_w4a16", "nvfp4", "w4a16"},
        {"nvfp4_w4a4", "nvfp4", "w4a4"},
    };
    int ncase = (int)(sizeof(cases) / sizeof(cases[0]));
    int fail = 0;

    float *R = (float *)malloc((size_t)OUT * IN * sizeof(float));
    float *xr = (float *)malloc((size_t)TOK * IN * sizeof(float));
    float *xact = (float *)malloc((size_t)TOK * IN * sizeof(float));
    float *y = (float *)malloc((size_t)TOK * OUT * sizeof(float));

    for (int ci = 0; ci < ncase; ci++) {
        const char *pf = cases[ci].prefix;
        int is_int4 = (strcmp(cases[ci].fmt, "int4") == 0);
        int is_a16 = (strcmp(cases[ci].scope, "w4a16") == 0);

        float *smooth = load_f32(key(pf, "_smooth"));
        float *lu = load_f32(key(pf, "_lora_up"));
        float *ld = load_f32(key(pf, "_lora_down"));
        float *y_svdq = load_f32(key(pf, "_y_svdq"));

        /* residual weight */
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
            free(qw); free(ws); free(wcwt);
        }

        /* activation path */
        sq_smooth_div(x, smooth, xr, TOK, IN);
        double act_drift = -1.0;
        if (is_a16) {
            memcpy(xact, xr, (size_t)TOK * IN * sizeof(float));
        } else if (is_int4) {
            /* exercise the CPU's own activation quantizer (rintf == torch.round) */
            sq_quant_act_int4_g64(xr, xact, TOK, IN, 64);
            float *xr_dq = load_f32(key(pf, "_xr_dq"));     /* diagnostic vs the driver's */
            act_drift = rel_l2(xact, xr_dq, TOK * IN);
            free(xr_dq);
        } else {
            /* NVFP4 activation: consume the driver's dequantized codes (no CPU e4m3 RN) */
            float *xr_dq = load_f32(key(pf, "_xr_dq"));
            memcpy(xact, xr_dq, (size_t)TOK * IN * sizeof(float));
            free(xr_dq);
        }

        sq_forward(xact, x, R, lu, ld, bias, y, TOK, OUT, IN, RANK);

        double rl = rel_l2(y, y_svdq, TOK * OUT);
        double cs = cosine(y, y_svdq, TOK * OUT);
        double mx = (double)npy_max_abs_f32(y, y_svdq, TOK * OUT, NULL);
        double rl_fp = rel_l2(y, y_fp, TOK * OUT);
        int ok = (rl <= GATE);
        if (!ok) fail = 1;
        printf("  %-12s rel_L2(svdq)=%.3e cos=%.7f max|d|=%.3e | rel_L2(fp)=%.4f  %s\n",
               pf, rl, cs, mx, rl_fp, ok ? "PASS" : "FAIL");
        if (act_drift >= 0)
            printf("               (act-quant drift vs ref = %.3e)\n", act_drift);

        free(smooth); free(lu); free(ld); free(y_svdq);
    }

    free(R); free(xr); free(xact); free(y); free(x); free(bias); free(y_fp);
    printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
    return fail;
}
