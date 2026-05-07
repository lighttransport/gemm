/* test_paint_unipc.c — replay UniPC port against the diffusers oracle.
 *
 * Loads:
 *   /tmp/hy3d_paint_unipc_ref/{timesteps,sigmas,x0,model_out_*,x_after_*}.npy
 * Runs the port for N=15 steps and compares both the precomputed schedule
 * tables (timesteps + sigmas) and per-step x_after.
 *
 * Build: see Makefile (test_paint_unipc target).
 * Run:   ./test_paint_unipc [refdir]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda_paint_unipc.h"

/* Minimal .npy reader (f32 / i64, contiguous, little-endian). */
static int npy_read(const char *path, void **out_data, size_t *out_bytes,
                    char *out_dtype) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return -1; }
    unsigned char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6)) {
        fclose(f); fprintf(stderr, "%s: bad magic\n", path); return -1;
    }
    unsigned char ver[2]; fread(ver, 1, 2, f);
    unsigned int hlen;
    if (ver[0] == 1) {
        unsigned short h16; fread(&h16, 2, 1, f); hlen = h16;
    } else {
        fread(&hlen, 4, 1, f);
    }
    char *hdr = (char*)malloc(hlen + 1);
    fread(hdr, 1, hlen, f); hdr[hlen] = 0;
    char dtype = 0;
    if (strstr(hdr, "'<f4'") || strstr(hdr, "'f4'")) dtype = 'f';
    else if (strstr(hdr, "'<i8'") || strstr(hdr, "'i8'")) dtype = 'i';
    else { fprintf(stderr, "%s: unsupported dtype: %s\n", path, hdr); free(hdr); fclose(f); return -1; }
    /* parse shape -> total numel */
    char *sh = strstr(hdr, "'shape':");
    if (!sh) sh = strstr(hdr, "\"shape\":");
    sh = strchr(sh, '(') + 1;
    size_t numel = 1;
    char *p = sh;
    while (*p && *p != ')') {
        if (*p >= '0' && *p <= '9') {
            size_t v = strtoull(p, &p, 10);
            numel *= v ? v : 1;
        } else p++;
    }
    free(hdr);
    size_t esz = (dtype == 'f') ? 4 : 8;
    void *data = malloc(numel * esz);
    if (fread(data, esz, numel, f) != numel) {
        fprintf(stderr, "%s: short read\n", path); free(data); fclose(f); return -1;
    }
    fclose(f);
    *out_data  = data;
    *out_bytes = numel * esz;
    if (out_dtype) *out_dtype = dtype;
    return (int)numel;
}

static void cmp_f32(const char *tag, const float *a, const float *b, size_t n) {
    double sum_ae = 0.0, max_ae = 0.0;
    double sum_a2 = 0.0, sum_b2 = 0.0, sum_ab = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        double ad = d < 0 ? -d : d;
        sum_ae += ad;
        if (ad > max_ae) max_ae = ad;
        sum_a2 += (double)a[i] * (double)a[i];
        sum_b2 += (double)b[i] * (double)b[i];
        sum_ab += (double)a[i] * (double)b[i];
    }
    double mae = sum_ae / (double)n;
    double corr = sum_ab / (sqrt(sum_a2) * sqrt(sum_b2) + 1e-30);
    int ok = (mae < 1e-5) && (max_ae < 1e-3);
    printf("  %-22s mae=%.4e max=%.4e corr=%.6f  %s\n",
           tag, mae, max_ae, corr, ok ? "OK" : "**MISMATCH**");
}

int main(int argc, char **argv) {
    const char *refdir = (argc > 1) ? argv[1] : "/tmp/hy3d_paint_unipc_ref";
    char path[1024];

    /* shape: [4,4,8,8] f32; numel = 1024 */
    const size_t numel = 4 * 4 * 8 * 8;
    const int    N     = 15;

    void *p_ts = NULL, *p_sg = NULL, *p_x0 = NULL;
    size_t b;
    snprintf(path, sizeof(path), "%s/timesteps.npy", refdir);
    int nts = npy_read(path, &p_ts, &b, NULL);
    snprintf(path, sizeof(path), "%s/sigmas.npy", refdir);
    int nsg = npy_read(path, &p_sg, &b, NULL);
    snprintf(path, sizeof(path), "%s/x0.npy", refdir);
    int nx0 = npy_read(path, &p_x0, &b, NULL);
    if (nts != N || nsg != N + 1 || nx0 != (int)numel) {
        fprintf(stderr, "shape mismatch: nts=%d nsg=%d nx0=%d\n", nts, nsg, nx0);
        return 1;
    }
    long long *ref_ts = (long long*)p_ts;
    float     *ref_sg = (float*)    p_sg;
    float     *x      = (float*)    p_x0;   /* in-place */

    pu_unipc s;
    pu_unipc_init(&s, N, numel);

    /* validate schedule */
    int sched_ok = 1;
    for (int i = 0; i < N; i++) {
        if (s.timesteps[i] != ref_ts[i]) {
            fprintf(stderr, "  timesteps[%d]: got %lld want %lld\n",
                    i, (long long)s.timesteps[i], (long long)ref_ts[i]);
            sched_ok = 0;
        }
    }
    double max_sig_ae = 0.0;
    for (int i = 0; i <= N; i++) {
        double d = fabs((double)s.sigmas[i] - (double)ref_sg[i]);
        if (d > max_sig_ae) max_sig_ae = d;
    }
    printf("schedule: timesteps %s, max sigma diff = %.4e\n",
           sched_ok ? "OK" : "**MISMATCH**", max_sig_ae);

    /* run and compare per-step */
    int all_ok = 1;
    for (int i = 0; i < N; i++) {
        void *p_m = NULL, *p_x = NULL;
        snprintf(path, sizeof(path), "%s/model_out_%d.npy", refdir, i);
        int nm = npy_read(path, &p_m, &b, NULL);
        snprintf(path, sizeof(path), "%s/x_after_%d.npy", refdir, i);
        int nxa = npy_read(path, &p_x, &b, NULL);
        if (nm != (int)numel || nxa != (int)numel) {
            fprintf(stderr, "step %d: size mismatch\n", i); return 1;
        }
        pu_unipc_step(&s, (const float*)p_m, x);
        char tag[64];
        snprintf(tag, sizeof(tag), "step %2d (t=%4lld)", i, (long long)ref_ts[i]);
        double sum_ae = 0.0, max_ae = 0.0;
        const float *xb = (const float*)p_x;
        for (size_t j = 0; j < numel; j++) {
            double d = fabs((double)x[j] - (double)xb[j]);
            sum_ae += d;
            if (d > max_ae) max_ae = d;
        }
        double mae = sum_ae / (double)numel;
        int ok = mae < 1e-5 && max_ae < 1e-3;
        if (!ok) all_ok = 0;
        printf("  %s mae=%.4e max=%.4e %s\n", tag, mae, max_ae, ok ? "OK" : "**MISMATCH**");
        free(p_m); free(p_x);
    }

    pu_unipc_free(&s);
    free(p_ts); free(p_sg); free(p_x0);
    printf("\nresult: %s\n", (sched_ok && all_ok) ? "PASS" : "FAIL");
    return (sched_ok && all_ok) ? 0 : 1;
}
