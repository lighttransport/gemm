/*
 * test_hip_slat_traj — 12-step SLAT trajectory with reference inputs.
 *
 * Loads ref SLAT noise + coords + cond, runs full CFG sampler, prints
 * denorm min/max/std. Compare to reference 08_shape_slat_denorm_feats.
 *
 * Usage:
 *   test_hip_slat_traj <slat_dit.safetensors> <manifest.json> <dump_dir>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "hip_trellis2_runner.h"
#include "../../common/npy_io.h"

static void *load_or_die(const char *dir, const char *name,
                         int *ndim, int *dims, int *is_f32) {
    char p[1024]; snprintf(p, sizeof(p), "%s/%s", dir, name);
    void *r = npy_load(p, ndim, dims, is_f32);
    if (!r) { fprintf(stderr, "missing %s\n", p); exit(2); }
    return r;
}

static int parse_shape_norm(const char *path, float *mean, float *std) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz+1); fread(buf,1,sz,f); buf[sz]=0; fclose(f);
    char *root = strstr(buf, "\"shape_slat_normalization\"");
    if (!root) { free(buf); return -1; }
    char *m = strstr(root, "\"mean\"");
    char *s = strstr(root, "\"std\"");
    if (!m || !s) { free(buf); return -1; }
    char *lb = strchr(m, '['); int i=0;
    while (lb && i<32 && *lb != ']') {
        lb++;
        char *end; float v = strtof(lb, &end);
        if (end != lb) { mean[i++] = v; lb = end; }
        while (*lb && *lb!=',' && *lb!=']') lb++;
    }
    lb = strchr(s, '['); i=0;
    while (lb && i<32 && *lb != ']') {
        lb++;
        char *end; float v = strtof(lb, &end);
        if (end != lb) { std[i++] = v; lb = end; }
        while (*lb && *lb!=',' && *lb!=']') lb++;
    }
    free(buf); return 0;
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <slat_dit.safetensors> <manifest.json> <dump_dir>\n", argv[0]);
        return 2;
    }
    const char *st_path = argv[1];
    const char *manifest_path = argv[2];
    const char *dir = argv[3];

    hip_trellis2_runner *r = hip_trellis2_init(0, 1);
    if (!r) return 3;
    if (hip_trellis2_load_slat_dit(r, st_path) != 0) return 3;

    int nd, dims[8], is_f32;

    /* Initial noise [N, 32] — use the reference shape_slat_noise_feats. */
    float *x = (float *)load_or_die(dir, "06_shape_slat_noise_feats.npy", &nd, dims, &is_f32);
    if (!is_f32 || nd != 2 || dims[1] != 32) { fprintf(stderr, "bad noise shape\n"); return 4; }
    int N = dims[0];

    int32_t *coords = (int32_t *)load_or_die(dir, "05_ss_coords.npy", &nd, dims, &is_f32);
    if (is_f32 || dims[0] != N || dims[1] != 4) { fprintf(stderr, "bad coords shape\n"); return 4; }

    float *cond_raw = (float *)load_or_die(dir, "01_dinov3_cond_512.npy", &nd, dims, &is_f32);
    int n_cond, cond_C;
    if (nd == 3) { n_cond = dims[1]; cond_C = dims[2]; }
    else         { n_cond = dims[0]; cond_C = dims[1]; }

    float slat_mean[32], slat_std[32];
    if (parse_shape_norm(manifest_path, slat_mean, slat_std) != 0) { fprintf(stderr, "parse manifest fail\n"); return 4; }

    fprintf(stderr, "N=%d n_cond=%d cond_C=%d\n", N, n_cond, cond_C);

    /* zero uncond. */
    float *neg_cond = (float *)calloc((size_t)n_cond * cond_C, sizeof(float));
    float *v_cond   = (float *)malloc((size_t)N * 32 * sizeof(float));
    float *v_uncond = (float *)malloc((size_t)N * 32 * sizeof(float));

    const int STEPS = 12;
    const float rescale_t = 3.0f, cfg = 7.5f, rescale = 0.7f, sm = 1e-5f;

    double t_total = 0;
    for (int step = 0; step < STEPS; step++) {
        float ta = 1.0f - (float)step/STEPS, tb = 1.0f - (float)(step+1)/STEPS;
        float tc = rescale_t*ta/(1.0f+(rescale_t-1.0f)*ta);
        float tn = rescale_t*tb/(1.0f+(rescale_t-1.0f)*tb);
        float dt = tc - tn;
        int use_cfg = (tc >= 0.6f && tc <= 1.0f);

        double t0 = now_ms();
        hip_trellis2_invalidate_slat_kv(r);
        if (hip_trellis2_slat_dit_step(r, x, coords, N, tc, cond_raw, n_cond, v_cond) != 0) {
            fprintf(stderr, "cond step %d fail\n", step); return 5;
        }
        if (use_cfg) {
            hip_trellis2_invalidate_slat_kv(r);
            if (hip_trellis2_slat_dit_step(r, x, coords, N, tc, neg_cond, n_cond, v_uncond) != 0) {
                fprintf(stderr, "uncond step %d fail\n", step); return 5;
            }
        }

        int n = N * 32;
        double v_min=1e30, v_max=-1e30, v_sum=0, v_sum2=0;
        for (int i = 0; i < n; i++) { if(v_cond[i]<v_min)v_min=v_cond[i]; if(v_cond[i]>v_max)v_max=v_cond[i]; v_sum+=v_cond[i]; v_sum2+=(double)v_cond[i]*v_cond[i]; }
        double v_std = sqrt((v_sum2 - v_sum*v_sum/n)/(n-1));

        if (use_cfg) {
            float coeff = sm + (1.0f - sm) * tc;
            float one_m_sm = 1.0f - sm;
            double sum_p=0,s2_p=0,sum_c=0,s2_c=0;
            for (int i = 0; i < n; i++) {
                float vcfg = cfg * v_cond[i] + (1.0f - cfg) * v_uncond[i];
                v_uncond[i] = vcfg;
                float x0p = one_m_sm * x[i] - coeff * v_cond[i];
                float x0c = one_m_sm * x[i] - coeff * vcfg;
                sum_p+=x0p; s2_p+=(double)x0p*x0p;
                sum_c+=x0c; s2_c+=(double)x0c*x0c;
            }
            double n_d=n;
            double std_p = sqrt((s2_p - sum_p*sum_p/n_d)/(n_d-1.0));
            double std_c = sqrt((s2_c - sum_c*sum_c/n_d)/(n_d-1.0));
            float ratio = std_c > 1e-8 ? (float)(std_p/std_c) : 1.0f;
            float sc = rescale * ratio + (1.0f - rescale);
            for (int i = 0; i < n; i++) {
                float x0c = one_m_sm * x[i] - coeff * v_uncond[i];
                float pred = (one_m_sm * x[i] - sc * x0c) / coeff;
                x[i] -= dt * pred;
            }
        } else {
            for (int i = 0; i < n; i++) x[i] -= dt * v_cond[i];
        }
        double step_ms = now_ms() - t0; t_total += step_ms;

        /* x stats */
        double xmn=1e30,xmx=-1e30,xsum=0,xs2=0;
        for (int i = 0; i < n; i++) { if(x[i]<xmn)xmn=x[i]; if(x[i]>xmx)xmx=x[i]; xsum+=x[i]; xs2+=(double)x[i]*x[i]; }
        double xstd = sqrt((xs2 - xsum*xsum/n)/(n-1));
        fprintf(stderr, "step %02d/%d t=%.4f->%.4f %s  v_std=%.4f  x:[%.3f,%.3f] mean=%.4f std=%.4f  %.1f ms\n",
                step+1, STEPS, tc, tn, use_cfg?"CFG":"noG",
                v_std, xmn, xmx, xsum/n, xstd, step_ms);
    }
    fprintf(stderr, "SLAT total %.1f ms\n", t_total);

    /* Denormalize */
    for (int i = 0; i < N; i++)
        for (int c = 0; c < 32; c++)
            x[i*32+c] = x[i*32+c] * slat_std[c] + slat_mean[c];

    double mn=1e30,mx=-1e30,sum=0,s2=0;
    for (int i = 0; i < N*32; i++) { if(x[i]<mn)mn=x[i]; if(x[i]>mx)mx=x[i]; sum+=x[i]; s2+=(double)x[i]*x[i]; }
    double m = sum/(N*32), std_v = sqrt((s2 - sum*sum/(N*32))/(N*32-1.0));
    printf("denorm slat: N=%d min=%.4f max=%.4f mean=%.4f std=%.4f\n", N, mn, mx, m, std_v);
    printf("REF expected: min~-25 max~27 std~6.0\n");

    /* Compare to reference. */
    float *ref = (float *)load_or_die(dir, "08_shape_slat_denorm_feats.npy", &nd, dims, &is_f32);
    if (nd == 2 && dims[0] == N) {
        int n = N * 32;
        double dot=0, na=0, nb=0; float mxa=0;
        for (int i = 0; i < n; i++) {
            double a=x[i], b=ref[i];
            dot+=a*b; na+=a*a; nb+=b*b;
            float d = fabsf((float)(a-b)); if (d>mxa) mxa=d;
        }
        double cos = dot/(sqrt(na)*sqrt(nb)+1e-30);
        printf("vs ref: cosine=%.6f max_abs=%.4f\n", cos, mxa);
    }

    free(neg_cond); free(v_cond); free(v_uncond);
    free(x); free(coords); free(cond_raw); free(ref);
    hip_trellis2_free(r);
    return 0;
}
