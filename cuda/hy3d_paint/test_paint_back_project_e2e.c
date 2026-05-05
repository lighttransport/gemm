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

int main(int argc, char **argv) {
    const char *refdir = (argc > 1) ? argv[1] : "/tmp/hy3d_paint_bp_ref";
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
    int Htex_i = Htex, Wtex_i = Wtex, Himg_i = Himg, Wimg_i = Wimg, C_i = C;
    void *args[] = {
        &d_tex_pos, &d_tex_cov, &d_image, &d_depth, &d_vis, &d_cos, &d_w2c,
        &proj00, &proj11, &depth_thres,
        &Htex_i, &Wtex_i, &Himg_i, &Wimg_i, &C_i,
        &d_out_tex, &d_out_cos
    };
    unsigned grid = (unsigned)((tex_n + 255) / 256);
    cuLaunchKernel(f_bp, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
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
