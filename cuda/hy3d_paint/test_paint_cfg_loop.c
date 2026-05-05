/* test_paint_cfg_loop.c — replay the Hunyuan3D-2.1 paint denoising
 * loop's UniPC + 3-way CFG combine math against the diffusers oracle.
 *
 * Loads /tmp/hy3d_paint_cfg_loop_ref/* (see ref/hy3d/dump_paint_cfg_loop.py)
 * and runs:
 *   for i in 0..steps-1:
 *     load model_out_<i>.npy  (3-way batch)
 *     CFG combine using view_scales (per-view azim mapping)
 *     pu_unipc_step
 *   compare x[i] against x_after_<i>.npy
 *
 * Validates the integration of cuda_paint_unipc.h with the paint
 * pipeline's unusual 3-way CFG (uncond → ref → full, per-view azim
 * scale). Self-contained: no NVRTC, no cuew, no UNet.
 *
 * Build: see Makefile (test_paint_cfg_loop target).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda_paint_unipc.h"

/* ---- minimal .npy reader (f32 / i64, contiguous, little-endian) ---- */
static int npy_read(const char *path, void **out_data, size_t *out_bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return -1; }
    unsigned char magic[6];
    size_t r = fread(magic, 1, 6, f);
    if (r != 6 || memcmp(magic, "\x93NUMPY", 6)) {
        fclose(f); fprintf(stderr, "%s: bad magic\n", path); return -1;
    }
    unsigned char ver[2]; r = fread(ver, 1, 2, f); (void)r;
    unsigned int hlen;
    if (ver[0] == 1) {
        unsigned short h16; r = fread(&h16, 2, 1, f); hlen = h16;
    } else {
        r = fread(&hlen, 4, 1, f);
    }
    char *hdr = (char*)malloc(hlen + 1);
    r = fread(hdr, 1, hlen, f); hdr[hlen] = 0;
    char dtype = 0;
    if (strstr(hdr, "'<f4'") || strstr(hdr, "'f4'")) dtype = 'f';
    else if (strstr(hdr, "'<i8'") || strstr(hdr, "'i8'")) dtype = 'i';
    else { fprintf(stderr, "%s: unsupported dtype: %s\n", path, hdr); free(hdr); fclose(f); return -1; }
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
    return (int)numel;
}

/* CFG combine: noise_pred = u + g*vs*(r - u) + g*vs*(f - r)
 * model layout per row: 3 chunks of size Beff*C*H*W stacked along batch.
 * vs is per-row [Beff] f32; broadcasts across (C,H,W). */
static void cfg_combine(const float *m, float *out, size_t Beff, size_t C, size_t H, size_t W,
                        float guidance, const float *vs_per_row) {
    size_t spc = C * H * W;
    const float *u = m;
    const float *r = m + Beff * spc;
    const float *fl = m + 2 * Beff * spc;
    for (size_t b = 0; b < Beff; b++) {
        float gv = guidance * vs_per_row[b];
        const float *ub = u  + b * spc;
        const float *rb = r  + b * spc;
        const float *fb = fl + b * spc;
        float *ob = out + b * spc;
        for (size_t k = 0; k < spc; k++) {
            float v = ub[k] + gv * (rb[k] - ub[k]);
            v += gv * (fb[k] - rb[k]);
            ob[k] = v;
        }
    }
}

int main(int argc, char **argv) {
    const char *refdir = (argc > 1) ? argv[1] : "/tmp/hy3d_paint_cfg_loop_ref";
    char path[1024];

    /* meta from dump (defaults match the script): batch=1 n_pbr=2 n_gen=6
     * c=4 h=8 w=8 steps=5 guidance=3.0. We re-derive shapes from the npy
     * files directly so a manual --steps/--shape change in the dumper
     * doesn't silently break us. */
    void *p_az = NULL, *p_vs = NULL, *p_x0 = NULL;
    size_t b;
    snprintf(path, sizeof(path), "%s/azims.npy", refdir);
    int N_gen = npy_read(path, &p_az, &b);
    snprintf(path, sizeof(path), "%s/view_scales.npy", refdir);
    int nvs = npy_read(path, &p_vs, &b);
    snprintf(path, sizeof(path), "%s/x0.npy", refdir);
    int nx0 = npy_read(path, &p_x0, &b);
    if (nvs != N_gen) {
        fprintf(stderr, "view_scales/azims size mismatch: %d vs %d\n", nvs, N_gen);
        return 1;
    }
    /* infer Beff*C*H*W from x0; assume B=1, N_pbr=2 (paint pipeline fixed) */
    int B = 1, N_pbr = 2;
    int Beff = B * N_pbr * N_gen;
    int chw  = nx0 / Beff;
    if (Beff * chw != nx0) {
        fprintf(stderr, "x0 size %d not divisible by Beff=%d\n", nx0, Beff); return 1;
    }
    /* assume c=4, h=w=sqrt(chw/4) */
    int c = 4, hw2 = chw / c, h = (int)sqrt((double)hw2), w = h;
    if (c * h * w != chw) {
        fprintf(stderr, "non-square chw layout: chw=%d, can't infer h,w\n", chw); return 1;
    }
    printf("dims: B=%d N_pbr=%d N_gen=%d C=%d H=%d W=%d  Beff=%d  chw=%d\n",
           B, N_pbr, N_gen, c, h, w, Beff, chw);

    /* steps inferred by counting model_out files */
    int N = 0;
    for (;;) {
        snprintf(path, sizeof(path), "%s/model_out_%d.npy", refdir, N);
        FILE *fp = fopen(path, "rb");
        if (!fp) break;
        fclose(fp); N++;
    }
    printf("steps: %d\n", N);
    const float guidance = 3.0f;

    /* per-row view_scale: [N_pbr, N_gen] tiled to [Beff].
     * pipeline does view_scales.repeat(n_pbr).view(-1) -> length = n_pbr*n_gen.
     * Then with B>1 it would tile B times. */
    float *vs_per_row = (float*)malloc(sizeof(float) * (size_t)Beff);
    const float *vs = (const float*)p_vs;
    for (int p = 0; p < N_pbr; p++)
        for (int v = 0; v < N_gen; v++)
            vs_per_row[p * N_gen + v] = vs[v];
    /* (B=1, so no further tile) */

    /* scheduler */
    pu_unipc s;
    pu_unipc_init(&s, N, (size_t)Beff * chw);

    float *x = (float*)p_x0;  /* in-place [Beff, c, h, w] */
    float *noise_pred = (float*)malloc(sizeof(float) * (size_t)Beff * chw);

    int all_ok = 1;
    for (int i = 0; i < N; i++) {
        void *p_m = NULL, *p_x = NULL;
        snprintf(path, sizeof(path), "%s/model_out_%d.npy", refdir, i);
        int nm = npy_read(path, &p_m, &b);
        snprintf(path, sizeof(path), "%s/x_after_%d.npy", refdir, i);
        int nxa = npy_read(path, &p_x, &b);
        if (nm != 3 * Beff * chw || nxa != Beff * chw) {
            fprintf(stderr, "step %d: size mismatch nm=%d nxa=%d\n", i, nm, nxa); return 1;
        }

        cfg_combine((const float*)p_m, noise_pred, Beff, c, h, w,
                    guidance, vs_per_row);

        /* optionally compare against dumped noise_pred */
        void *p_np = NULL;
        snprintf(path, sizeof(path), "%s/noise_pred_%d.npy", refdir, i);
        int nnp = npy_read(path, &p_np, &b);
        if (nnp == Beff * chw) {
            const float *npb = (const float*)p_np;
            double max_ae = 0.0;
            for (int k = 0; k < Beff * chw; k++) {
                double d = fabs((double)noise_pred[k] - (double)npb[k]);
                if (d > max_ae) max_ae = d;
            }
            if (max_ae > 1e-4) {
                fprintf(stderr, "step %d: CFG combine drift max=%.4e\n", i, max_ae);
                all_ok = 0;
            }
        }
        free(p_np);

        pu_unipc_step(&s, noise_pred, x);

        const float *xb = (const float*)p_x;
        double sum_ae = 0.0, max_ae = 0.0;
        for (int k = 0; k < Beff * chw; k++) {
            double d = fabs((double)x[k] - (double)xb[k]);
            sum_ae += d;
            if (d > max_ae) max_ae = d;
        }
        double mae = sum_ae / (double)(Beff * chw);
        int ok = mae < 1e-4 && max_ae < 1e-3;
        if (!ok) all_ok = 0;
        printf("  step %2d  mae=%.4e max=%.4e  %s\n",
               i, mae, max_ae, ok ? "OK" : "**MISMATCH**");
        free(p_m); free(p_x);
    }

    pu_unipc_free(&s);
    free(p_az); free(p_vs); free(p_x0);
    free(vs_per_row); free(noise_pred);
    printf("\nresult: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
