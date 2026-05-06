/*
 * test_paint_back_project_stage.c - smoke harness for paint_stage_back_project.
 *
 * Replays a multi-view dump from ref/hy3d/dump_paint_back_project.py through
 * the opaque API: create -> set_atlas -> begin -> add_view×N -> finalize ->
 * diff bake_tex / bake_mask vs the pyref oracle (bake_tex.npy / bake_trust.npy).
 *
 * Usage:
 *   ./test_paint_back_project_stage [<refdir>]
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cuew.h"
#include "paint_stages.h"

static void *npy_read(const char *path, size_t *out_numel, size_t *out_esz) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "open %s failed\n", path); return NULL; }
    unsigned char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6)) { fclose(f); return NULL; }
    unsigned char ver[2]; if (fread(ver, 1, 2, f) != 2) { fclose(f); return NULL; }
    unsigned int hlen;
    if (ver[0] == 1) {
        unsigned short h16; if (fread(&h16, 2, 1, f) != 1) { fclose(f); return NULL; }
        hlen = h16;
    } else {
        if (fread(&hlen, 4, 1, f) != 1) { fclose(f); return NULL; }
    }
    char *hdr = (char*)malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = 0;
    size_t esz = 0;
    if (strstr(hdr, "'<f4'") || strstr(hdr, "'f4'")) esz = 4;
    else if (strstr(hdr, "'<i4'") || strstr(hdr, "'i4'")) esz = 4;
    else if (strstr(hdr, "'|u1'") || strstr(hdr, "'u1'")) esz = 1;
    else { fprintf(stderr, "%s: unsupported dtype: %s\n", path, hdr); free(hdr); fclose(f); return NULL; }
    char *sh = strstr(hdr, "'shape':"); sh = strchr(sh, '(') + 1;
    size_t numel = 1; char *p = sh;
    while (*p && *p != ')') {
        if (*p >= '0' && *p <= '9') { size_t v = strtoull(p, &p, 10); numel *= v ? v : 1; }
        else p++;
    }
    free(hdr);
    void *data = malloc(numel * esz);
    if (fread(data, esz, numel, f) != numel) { free(data); fclose(f); return NULL; }
    fclose(f);
    if (out_numel) *out_numel = numel;
    if (out_esz)   *out_esz   = esz;
    return data;
}

int main(int argc, char **argv) {
    const char *refdir = argc >= 2 ? argv[1] : "/tmp/hy3d_paint_bp_ref";
    char path[1024]; size_t esz;

    size_t n_tp, n_tc, n_pr, n_bt, n_btr;
    snprintf(path, sizeof(path), "%s/tex_pos.npy", refdir);
    float *p_tex_pos = (float*)npy_read(path, &n_tp, &esz); if (!p_tex_pos) return 1;
    snprintf(path, sizeof(path), "%s/tex_cov.npy", refdir);
    int *p_tex_cov = (int*)npy_read(path, &n_tc, &esz); if (!p_tex_cov) return 1;
    snprintf(path, sizeof(path), "%s/proj.npy", refdir);
    float *p_proj = (float*)npy_read(path, &n_pr, &esz); if (!p_proj) return 1;
    snprintf(path, sizeof(path), "%s/bake_tex.npy", refdir);
    float *p_bake_tex = (float*)npy_read(path, &n_bt, &esz); if (!p_bake_tex) return 1;
    snprintf(path, sizeof(path), "%s/bake_trust.npy", refdir);
    float *p_bake_trust = (float*)npy_read(path, &n_btr, &esz); if (!p_bake_trust) return 1;

    int Htex = (int)sqrt((double)n_tc), Wtex = Htex, C = 3;
    float proj00 = p_proj[0], proj11 = p_proj[1];

    int N = 0; while (N < 32) {
        snprintf(path, sizeof(path), "%s/view_%d_image.npy", refdir, N);
        FILE *fp = fopen(path, "rb"); if (!fp) break; fclose(fp); N++;
    }
    if (N == 0) { fprintf(stderr, "no view_*_image.npy in %s\n", refdir); return 1; }

    float **h_image = malloc(N * sizeof(*h_image));
    float **h_depth = malloc(N * sizeof(*h_depth));
    float **h_vis   = malloc(N * sizeof(*h_vis));
    float **h_cos   = malloc(N * sizeof(*h_cos));
    float **h_w2c   = malloc(N * sizeof(*h_w2c));
    int Himg = 0, Wimg = 0;
    for (int v = 0; v < N; v++) {
        size_t n;
        snprintf(path, sizeof(path), "%s/view_%d_image.npy", refdir, v);
        h_image[v] = (float*)npy_read(path, &n, &esz);
        if (Himg == 0) { Himg = (int)sqrt((double)n / C); Wimg = Himg; }
        snprintf(path, sizeof(path), "%s/view_%d_depth.npy", refdir, v);
        h_depth[v] = (float*)npy_read(path, &n, &esz);
        snprintf(path, sizeof(path), "%s/view_%d_visible.npy", refdir, v);
        h_vis[v]   = (float*)npy_read(path, &n, &esz);
        snprintf(path, sizeof(path), "%s/view_%d_cos.npy", refdir, v);
        h_cos[v]   = (float*)npy_read(path, &n, &esz);
        snprintf(path, sizeof(path), "%s/view_%d_w2c.npy", refdir, v);
        h_w2c[v]   = (float*)npy_read(path, &n, &esz);
    }
    fprintf(stderr, "tex %dx%d  img %dx%d  C=%d  N=%d  proj=(%.4f,%.4f)\n",
            Htex, Wtex, Himg, Wimg, C, N, proj00, proj11);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    paint_stage_back_project *s = paint_stage_back_project_create(dev, Htex, Wtex, C);
    if (!s) { fprintf(stderr, "stage create failed\n"); return 1; }
    paint_stage_back_project_set_atlas(s, p_tex_pos, p_tex_cov);
    paint_stage_back_project_begin(s);
    int n_skipped = 0;
    for (int v = 0; v < N; v++) {
        int sk = paint_stage_back_project_add_view(s,
            h_image[v], h_depth[v], h_vis[v], h_cos[v], h_w2c[v],
            Himg, Wimg, proj00, proj11);
        if (sk) { fprintf(stderr, "  view %d: skipped\n", v); n_skipped++; }
    }

    size_t tex_n = (size_t)Htex * Wtex;
    float *bake = (float*)malloc(tex_n * C * sizeof(float));
    float *bake_mask = (float*)malloc(tex_n * sizeof(float));
    paint_stage_back_project_finalize(s, bake, bake_mask);

    int mask_mismatch = 0;
    double tex_max = 0, tex_sum = 0; size_t tex_n_diff = 0;
    for (size_t i = 0; i < tex_n; i++) {
        if ((bake_mask[i] > 0) != (p_bake_trust[i] > 0)) mask_mismatch++;
        if (bake_mask[i] > 0 || p_bake_trust[i] > 0) {
            for (int k = 0; k < C; k++) {
                double d = fabs((double)bake[i*C+k] - (double)p_bake_tex[i*C+k]);
                if (d > tex_max) tex_max = d;
                tex_sum += d; tex_n_diff++;
            }
        }
    }
    fprintf(stderr,
        "skipped=%d  bake_mask mismatch=%d   bake_tex mae=%.3e max=%.3e (%zu)\n",
        n_skipped, mask_mismatch,
        tex_n_diff ? tex_sum / (double)tex_n_diff : 0.0, tex_max, tex_n_diff);
    int ok = (mask_mismatch == 0) && (tex_max < 1e-4);
    fprintf(stderr, "result: %s\n", ok ? "PASS" : "FAIL");

    for (int v = 0; v < N; v++) {
        free(h_image[v]); free(h_depth[v]); free(h_vis[v]);
        free(h_cos[v]); free(h_w2c[v]);
    }
    free(h_image); free(h_depth); free(h_vis); free(h_cos); free(h_w2c);
    free(bake); free(bake_mask);
    free(p_tex_pos); free(p_tex_cov); free(p_proj);
    free(p_bake_tex); free(p_bake_trust);
    paint_stage_back_project_destroy(s);
    cuCtxDestroy(ctx);
    return ok ? 0 : 1;
}
