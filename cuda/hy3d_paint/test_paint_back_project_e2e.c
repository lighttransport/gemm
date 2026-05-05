/*
 * test_paint_back_project_e2e.c — replay the back_project oracle dumped by
 * ref/hy3d/dump_paint_back_project.py against the CUDA kernel
 * `back_project_sample_f32`. Loads .npy inputs (tex_pos / tex_cov / image /
 * depth / visible / cos / w2c / proj), runs the kernel once, diffs the
 * device output against ref_tex / ref_cos.
 *
 * This is the unit-validation harness for Phase 4.11c.2 (per-view
 * back-projection). Multi-view bake-blend is the next step.
 *
 * Build: see Makefile (test_paint_back_project_e2e target).
 * Run:   ./test_paint_back_project_e2e [/tmp/hy3d_paint_bp_ref]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_raster_kernels.h"

/* Minimal .npy reader (f32 / i32, contiguous, little-endian). Returns
 * malloc'd buffer; *out_numel = element count, *out_esz = element size. */
static void *npy_read(const char *path, size_t *out_numel, size_t *out_esz) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return NULL; }
    unsigned char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6)) {
        fclose(f); fprintf(stderr, "%s: bad magic\n", path); return NULL;
    }
    unsigned char ver[2];
    if (fread(ver, 1, 2, f) != 2) { fclose(f); return NULL; }
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
    else { fprintf(stderr, "%s: unsupported dtype: %s\n", path, hdr); free(hdr); fclose(f); return NULL; }
    char *sh = strstr(hdr, "'shape':"); if (!sh) sh = strstr(hdr, "\"shape\":");
    sh = strchr(sh, '(') + 1;
    size_t numel = 1; char *p = sh;
    while (*p && *p != ')') {
        if (*p >= '0' && *p <= '9') {
            size_t v = strtoull(p, &p, 10);
            numel *= v ? v : 1;
        } else p++;
    }
    free(hdr);
    void *data = malloc(numel * esz);
    if (fread(data, esz, numel, f) != numel) {
        fprintf(stderr, "%s: short read\n", path);
        free(data); fclose(f); return NULL;
    }
    fclose(f);
    if (out_numel) *out_numel = numel;
    if (out_esz)   *out_esz   = esz;
    return data;
}

/* Per-view back_project: launches `back_project_sample_f32` once for view
 * `vi`. Caller has uploaded per-view inputs to d_image[vi]/d_depth[vi]/etc
 * and a per-view d_w2c. Output is written into d_out_tex[vi] / d_out_cos[vi].
 */
static void launch_back_project(CUfunction f_bp, CUstream s,
        CUdeviceptr d_tex_pos, CUdeviceptr d_tex_cov,
        CUdeviceptr d_image, CUdeviceptr d_depth,
        CUdeviceptr d_vis, CUdeviceptr d_cos, CUdeviceptr d_w2c,
        float proj00, float proj11, float depth_thres,
        int Htex, int Wtex, int Himg, int Wimg, int C,
        CUdeviceptr d_out_tex, CUdeviceptr d_out_cos) {
    void *args[] = {
        &d_tex_pos, &d_tex_cov, &d_image, &d_depth, &d_vis, &d_cos, &d_w2c,
        &proj00, &proj11, &depth_thres,
        &Htex, &Wtex, &Himg, &Wimg, &C,
        &d_out_tex, &d_out_cos
    };
    unsigned grid = (unsigned)((Htex * Wtex + 255) / 256);
    cuLaunchKernel(f_bp, grid, 1, 1, 256, 1, 1, 0, s, args, NULL);
}

static int run_single_view(const char *refdir);
static int run_multi_view (const char *refdir);

int main(int argc, char **argv) {
    int multiview = 0;
    const char *refdir = "/tmp/hy3d_paint_bp_ref";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--multiview")) multiview = 1;
        else refdir = argv[i];
    }
    if (multiview) return run_multi_view(refdir);
    return run_single_view(refdir);
}

static int run_single_view(const char *refdir) {
    char path[1024];
    /* Load the dump. We re-derive shapes from each file's element count and
     * the meta we know from the dumper (--htex / --himg / C=3). */
    void *p_tex_pos = NULL, *p_tex_cov = NULL, *p_image = NULL,
         *p_depth = NULL, *p_visible = NULL, *p_cos = NULL,
         *p_w2c = NULL, *p_proj = NULL,
         *p_ref_tex = NULL, *p_ref_cos = NULL;
    size_t n_tp, n_tc, n_im, n_d, n_v, n_c, n_w, n_pr, n_rt, n_rc, esz;

#define LOAD(var, nvar, name) do { \
    snprintf(path, sizeof(path), "%s/%s", refdir, name); \
    var = npy_read(path, &nvar, &esz); \
    if (!var) return 1; \
} while (0)
    LOAD(p_tex_pos, n_tp, "tex_pos.npy");
    LOAD(p_tex_cov, n_tc, "tex_cov.npy");
    LOAD(p_image,   n_im, "image.npy");
    LOAD(p_depth,   n_d,  "depth.npy");
    LOAD(p_visible, n_v,  "visible.npy");
    LOAD(p_cos,     n_c,  "cos.npy");
    LOAD(p_w2c,     n_w,  "w2c.npy");
    LOAD(p_proj,    n_pr, "proj.npy");
    LOAD(p_ref_tex, n_rt, "ref_tex.npy");
    LOAD(p_ref_cos, n_rc, "ref_cos.npy");
#undef LOAD

    int Htex = (int)sqrt((double)n_tc);
    int Wtex = Htex;
    int C = 3;
    int Himg = (int)sqrt((double)n_d);
    int Wimg = Himg;
    if ((size_t)Htex*Wtex != n_tc || (size_t)Himg*Wimg != n_d
        || n_tp != (size_t)Htex*Wtex*3 || n_im != (size_t)Himg*Wimg*C
        || n_w != 16 || n_pr != 2 || n_rt != n_im / Himg/Wimg * Htex*Wtex) {
        fprintf(stderr, "shape sanity check failed: tex=%d^2 img=%d^2 "
                "n_tp=%zu n_im=%zu n_rt=%zu\n", Htex, Himg, n_tp, n_im, n_rt);
        return 1;
    }
    float proj00 = ((float*)p_proj)[0];
    float proj11 = ((float*)p_proj)[1];
    fprintf(stderr, "tex %dx%d  img %dx%d  C=%d  proj=(%.4f,%.4f)\n",
            Htex, Wtex, Himg, Wimg, C, proj00, proj11);

    /* CUDA init */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    CUmodule mod;
    if (cu_compile_kernels(&mod, dev, cuda_paint_raster_kernels_src,
                           "hy3d_paint_raster", 1, "HY3D-PAINT") < 0) return 1;
    CUfunction f_bp;
    cuModuleGetFunction(&f_bp, mod, "back_project_sample_f32");

    /* Upload */
    size_t tex_n = (size_t)Htex * Wtex;
    size_t img_n = (size_t)Himg * Wimg;
    CUdeviceptr d_tex_pos, d_tex_cov, d_image, d_depth, d_vis, d_cos, d_w2c,
                d_out_tex, d_out_cos;
    cuMemAlloc(&d_tex_pos, tex_n * 3 * sizeof(float));
    cuMemAlloc(&d_tex_cov, tex_n * sizeof(int));
    cuMemAlloc(&d_image,   img_n * C * sizeof(float));
    cuMemAlloc(&d_depth,   img_n * sizeof(float));
    cuMemAlloc(&d_vis,     img_n * sizeof(float));
    cuMemAlloc(&d_cos,     img_n * sizeof(float));
    cuMemAlloc(&d_w2c,     16 * sizeof(float));
    cuMemAlloc(&d_out_tex, tex_n * C * sizeof(float));
    cuMemAlloc(&d_out_cos, tex_n * sizeof(float));
    cuMemcpyHtoD(d_tex_pos, p_tex_pos, tex_n * 3 * sizeof(float));
    cuMemcpyHtoD(d_tex_cov, p_tex_cov, tex_n * sizeof(int));
    cuMemcpyHtoD(d_image,   p_image,   img_n * C * sizeof(float));
    cuMemcpyHtoD(d_depth,   p_depth,   img_n * sizeof(float));
    cuMemcpyHtoD(d_vis,     p_visible, img_n * sizeof(float));
    cuMemcpyHtoD(d_cos,     p_cos,     img_n * sizeof(float));
    cuMemcpyHtoD(d_w2c,     p_w2c,     16 * sizeof(float));
    /* Outputs zeroed */
    void *zt = calloc(tex_n * C, sizeof(float));
    void *zc = calloc(tex_n,     sizeof(float));
    cuMemcpyHtoD(d_out_tex, zt, tex_n * C * sizeof(float)); free(zt);
    cuMemcpyHtoD(d_out_cos, zc, tex_n     * sizeof(float)); free(zc);

    float depth_thres = 3e-3f;
    launch_back_project(f_bp, 0,
        d_tex_pos, d_tex_cov, d_image, d_depth, d_vis, d_cos, d_w2c,
        proj00, proj11, depth_thres,
        Htex, Wtex, Himg, Wimg, C, d_out_tex, d_out_cos);
    cuCtxSynchronize();

    /* Download */
    float *out_tex = (float*)malloc(tex_n * C * sizeof(float));
    float *out_cos = (float*)malloc(tex_n     * sizeof(float));
    cuMemcpyDtoH(out_tex, d_out_tex, tex_n * C * sizeof(float));
    cuMemcpyDtoH(out_cos, d_out_cos, tex_n     * sizeof(float));

    /* Diff against pyref. Compare on the union mask (out_cos>0 OR ref_cos>0)
     * to catch both false negatives and false positives. */
    const float *ref_tex = (const float*)p_ref_tex;
    const float *ref_cos = (const float*)p_ref_cos;
    int filled_dev = 0, filled_ref = 0, mismatch_mask = 0;
    double tex_max = 0, tex_sum = 0; size_t tex_n_diff = 0;
    double cos_max = 0, cos_sum = 0;
    for (size_t i = 0; i < tex_n; i++) {
        int dvr = out_cos[i] > 0.f, rvr = ref_cos[i] > 0.f;
        if (dvr) filled_dev++;
        if (rvr) filled_ref++;
        if (dvr != rvr) mismatch_mask++;
        double dc = fabs((double)out_cos[i] - (double)ref_cos[i]);
        if (dc > cos_max) cos_max = dc;
        cos_sum += dc;
        if (dvr || rvr) {
            for (int k = 0; k < C; k++) {
                double d = fabs((double)out_tex[i*C+k] - (double)ref_tex[i*C+k]);
                if (d > tex_max) tex_max = d;
                tex_sum += d; tex_n_diff++;
            }
        }
    }
    fprintf(stderr,
        "filled  dev=%d  ref=%d  mask_mismatch=%d (%.3f%%)\n",
        filled_dev, filled_ref, mismatch_mask,
        100.0 * mismatch_mask / (double)tex_n);
    fprintf(stderr,
        "cos_map mae=%.3e max=%.3e\n",
        cos_sum / (double)tex_n, cos_max);
    fprintf(stderr,
        "texture mae=%.3e max=%.3e   (over %zu diffed values)\n",
        tex_sum / (double)tex_n_diff, tex_max, tex_n_diff);

    int ok = (mismatch_mask == 0) && (tex_max < 1e-4) && (cos_max < 1e-5);
    fprintf(stderr, "result: %s\n", ok ? "PASS" : "FAIL");

    cuMemFree(d_tex_pos); cuMemFree(d_tex_cov); cuMemFree(d_image);
    cuMemFree(d_depth); cuMemFree(d_vis); cuMemFree(d_cos); cuMemFree(d_w2c);
    cuMemFree(d_out_tex); cuMemFree(d_out_cos);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    free(p_tex_pos); free(p_tex_cov); free(p_image); free(p_depth);
    free(p_visible); free(p_cos); free(p_w2c); free(p_proj);
    free(p_ref_tex); free(p_ref_cos); free(out_tex); free(out_cos);
    return ok ? 0 : 1;
}

static int run_multi_view(const char *refdir) {
    char path[1024];
    void *p_tex_pos = NULL, *p_tex_cov = NULL, *p_proj = NULL,
         *p_bake_tex = NULL, *p_bake_trust = NULL;
    size_t n_tp, n_tc, n_pr, n_bt, n_btr, esz;
#define LOAD(var, nvar, name) do { \
    snprintf(path, sizeof(path), "%s/%s", refdir, name); \
    var = npy_read(path, &nvar, &esz); \
    if (!var) return 1; \
} while (0)
    LOAD(p_tex_pos, n_tp, "tex_pos.npy");
    LOAD(p_tex_cov, n_tc, "tex_cov.npy");
    LOAD(p_proj,    n_pr, "proj.npy");
    LOAD(p_bake_tex,   n_bt,  "bake_tex.npy");
    LOAD(p_bake_trust, n_btr, "bake_trust.npy");
    int Htex = (int)sqrt((double)n_tc), Wtex = Htex, C = 3;
    float proj00 = ((float*)p_proj)[0], proj11 = ((float*)p_proj)[1];
    /* Detect view count from existing files (1..32). */
    int N = 0;
    while (N < 32) {
        snprintf(path, sizeof(path), "%s/view_%d_image.npy", refdir, N);
        FILE *fp = fopen(path, "rb"); if (!fp) break; fclose(fp); N++;
    }
    if (N == 0) { fprintf(stderr, "no view_*_image.npy in %s\n", refdir); return 1; }

    /* Inputs per view: (image, depth, visible, cos, w2c). Loaded once into
     * device memory to mirror what the integrated runner will do. */
    float **h_image = malloc(N * sizeof(*h_image));
    float **h_depth = malloc(N * sizeof(*h_depth));
    float **h_vis   = malloc(N * sizeof(*h_vis));
    float **h_cos   = malloc(N * sizeof(*h_cos));
    float **h_w2c   = malloc(N * sizeof(*h_w2c));
    int Himg = 0, Wimg = 0;
    for (int v = 0; v < N; v++) {
        size_t n;
        snprintf(path, sizeof(path), "%s/view_%d_image.npy", refdir, v);
        h_image[v] = npy_read(path, &n, &esz); if (!h_image[v]) return 1;
        if (Himg == 0) { Himg = (int)sqrt((double)n / C); Wimg = Himg; }
        snprintf(path, sizeof(path), "%s/view_%d_depth.npy", refdir, v);
        h_depth[v] = npy_read(path, &n, &esz);
        snprintf(path, sizeof(path), "%s/view_%d_visible.npy", refdir, v);
        h_vis[v]   = npy_read(path, &n, &esz);
        snprintf(path, sizeof(path), "%s/view_%d_cos.npy", refdir, v);
        h_cos[v]   = npy_read(path, &n, &esz);
        snprintf(path, sizeof(path), "%s/view_%d_w2c.npy", refdir, v);
        h_w2c[v]   = npy_read(path, &n, &esz);
    }
    fprintf(stderr, "tex %dx%d  img %dx%d  C=%d  N_views=%d  proj=(%.4f,%.4f)\n",
            Htex, Wtex, Himg, Wimg, C, N, proj00, proj11);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    CUmodule mod;
    if (cu_compile_kernels(&mod, dev, cuda_paint_raster_kernels_src,
                           "hy3d_paint_raster", 1, "HY3D-PAINT") < 0) return 1;
    CUfunction f_bp; cuModuleGetFunction(&f_bp, mod, "back_project_sample_f32");

    size_t tex_n = (size_t)Htex * Wtex;
    size_t img_n = (size_t)Himg * Wimg;
    /* Per-view device buffers */
    CUdeviceptr d_tex_pos, d_tex_cov;
    cuMemAlloc(&d_tex_pos, tex_n * 3 * sizeof(float));
    cuMemAlloc(&d_tex_cov, tex_n     * sizeof(int));
    cuMemcpyHtoD(d_tex_pos, p_tex_pos, tex_n * 3 * sizeof(float));
    cuMemcpyHtoD(d_tex_cov, p_tex_cov, tex_n     * sizeof(int));
    CUdeviceptr d_image, d_depth, d_vis, d_cos, d_w2c, d_out_tex, d_out_cos;
    cuMemAlloc(&d_image,   img_n * C * sizeof(float));
    cuMemAlloc(&d_depth,   img_n     * sizeof(float));
    cuMemAlloc(&d_vis,     img_n     * sizeof(float));
    cuMemAlloc(&d_cos,     img_n     * sizeof(float));
    cuMemAlloc(&d_w2c,     16        * sizeof(float));
    cuMemAlloc(&d_out_tex, tex_n * C * sizeof(float));
    cuMemAlloc(&d_out_cos, tex_n     * sizeof(float));

    /* Per-view CUDA back_project, results downloaded for host-side bake. */
    float **out_tex = malloc(N * sizeof(*out_tex));
    float **out_cos = malloc(N * sizeof(*out_cos));
    void *zt = calloc(tex_n * C, sizeof(float));
    void *zc = calloc(tex_n,     sizeof(float));
    for (int v = 0; v < N; v++) {
        cuMemcpyHtoD(d_image, h_image[v], img_n * C * sizeof(float));
        cuMemcpyHtoD(d_depth, h_depth[v], img_n     * sizeof(float));
        cuMemcpyHtoD(d_vis,   h_vis[v],   img_n     * sizeof(float));
        cuMemcpyHtoD(d_cos,   h_cos[v],   img_n     * sizeof(float));
        cuMemcpyHtoD(d_w2c,   h_w2c[v],   16        * sizeof(float));
        cuMemcpyHtoD(d_out_tex, zt, tex_n * C * sizeof(float));
        cuMemcpyHtoD(d_out_cos, zc, tex_n     * sizeof(float));
        launch_back_project(f_bp, 0,
            d_tex_pos, d_tex_cov, d_image, d_depth, d_vis, d_cos, d_w2c,
            proj00, proj11, 3e-3f, Htex, Wtex, Himg, Wimg, C,
            d_out_tex, d_out_cos);
        cuCtxSynchronize();
        out_tex[v] = malloc(tex_n * C * sizeof(float));
        out_cos[v] = malloc(tex_n     * sizeof(float));
        cuMemcpyDtoH(out_tex[v], d_out_tex, tex_n * C * sizeof(float));
        cuMemcpyDtoH(out_cos[v], d_out_cos, tex_n     * sizeof(float));
    }
    free(zt); free(zc);

    /* Bake-blend on host: weight = cos^exp; tex_merge = sum(tex*w)/sum(w);
     * trust = (sum(w) > 1e-8). Mirror MeshRender.fast_bake_texture's "skip
     * view if 99% already painted" optimization (matters for order-dependent
     * test reproducibility). */
    const float exp_w = 6.0f;
    float *tex_merge = calloc(tex_n * C, sizeof(float));
    float *trust     = calloc(tex_n,     sizeof(float));
    for (int v = 0; v < N; v++) {
        size_t view_sum = 0, painted_sum = 0;
        for (size_t i = 0; i < tex_n; i++) {
            int rvr = out_cos[v][i] > 0.f;
            if (rvr) view_sum++;
            if (rvr && trust[i] > 0.f) painted_sum++;
        }
        if (view_sum > 0 && (double)painted_sum / view_sum > 0.99) {
            fprintf(stderr, "  view %d: skipped (99%% painted)\n", v);
            continue;
        }
        for (size_t i = 0; i < tex_n; i++) {
            float w = powf(out_cos[v][i], exp_w);
            for (int k = 0; k < C; k++) tex_merge[i*C+k] += out_tex[v][i*C+k] * w;
            trust[i] += w;
        }
    }
    float *bake = malloc(tex_n * C * sizeof(float));
    float *bake_mask = malloc(tex_n * sizeof(float));
    for (size_t i = 0; i < tex_n; i++) {
        float t = trust[i] > 1e-8f ? trust[i] : 1e-8f;
        for (int k = 0; k < C; k++) bake[i*C+k] = tex_merge[i*C+k] / t;
        bake_mask[i] = trust[i] > 1e-8f ? 1.f : 0.f;
    }

    /* Diff vs pyref bake. */
    const float *ref_bake  = (const float*)p_bake_tex;
    const float *ref_trust = (const float*)p_bake_trust;
    int mask_mismatch = 0;
    double tex_max = 0, tex_sum = 0; size_t tex_n_diff = 0;
    for (size_t i = 0; i < tex_n; i++) {
        if ((bake_mask[i] > 0) != (ref_trust[i] > 0)) mask_mismatch++;
        if (bake_mask[i] > 0 || ref_trust[i] > 0) {
            for (int k = 0; k < C; k++) {
                double d = fabs((double)bake[i*C+k] - (double)ref_bake[i*C+k]);
                if (d > tex_max) tex_max = d;
                tex_sum += d; tex_n_diff++;
            }
        }
    }
    fprintf(stderr,
        "bake_mask mismatch=%d   bake_tex mae=%.3e max=%.3e (over %zu)\n",
        mask_mismatch, tex_sum / (double)tex_n_diff, tex_max, tex_n_diff);
    int ok = (mask_mismatch == 0) && (tex_max < 1e-4);
    fprintf(stderr, "result: %s\n", ok ? "PASS" : "FAIL");

    for (int v = 0; v < N; v++) {
        free(h_image[v]); free(h_depth[v]); free(h_vis[v]);
        free(h_cos[v]); free(h_w2c[v]);
        free(out_tex[v]); free(out_cos[v]);
    }
    free(h_image); free(h_depth); free(h_vis); free(h_cos); free(h_w2c);
    free(out_tex); free(out_cos);
    free(tex_merge); free(trust); free(bake); free(bake_mask);
    free(p_tex_pos); free(p_tex_cov); free(p_proj);
    free(p_bake_tex); free(p_bake_trust);
    cuMemFree(d_tex_pos); cuMemFree(d_tex_cov); cuMemFree(d_image);
    cuMemFree(d_depth); cuMemFree(d_vis); cuMemFree(d_cos); cuMemFree(d_w2c);
    cuMemFree(d_out_tex); cuMemFree(d_out_cos);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return ok ? 0 : 1;
}
#undef LOAD
