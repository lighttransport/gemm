/*
 * test_back_project.c - Round-trip test of the native back-projection
 * bake kernel (back_project_sample_f32).
 *
 * Setup (all host-side, no mesh file needed):
 *   - A 1x1 quad centered at the origin, with its UV atlas matching
 *     world-space x/y (so tex_pos[u,v] = (u-0.5, v-0.5, 0)).
 *   - Orthographic camera looking down -Z from (0,0,3), +Y up,
 *     view matrix = lookAt(eye=(0,0,3), target=(0,0,0), up=(0,1,0)),
 *     proj = ortho(-0.6..0.6, -0.6..0.6, 0.1..10).
 *   - Synthetic "view image" at 128x128: a smooth RGB gradient.
 *   - visible_mask = all ones; cos_img = 1 everywhere; depth_img =
 *     camera-space z of the quad's plane (= -3).
 *
 * Expected outcome: the sampled UV texture should be the gradient image
 * resampled into UV space. For an ortho camera and identity-ish
 * alignment the back-projected texture should match the input image to
 * within F32 bilinear-sampling noise. We dump both to .npy and diff
 * them plus compare to a NumPy ref in the commit log.
 *
 * Build:
 *   make test_back_project
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_raster_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_npy(const char *path, const char *dtype,
                      const int *shape, int ndims,
                      const void *data, size_t elem_bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256], shape_s[128] = "";
    size_t total = 1;
    for (int i = 0; i < ndims; i++) {
        char tmp[32]; snprintf(tmp, sizeof(tmp), "%d, ", shape[i]);
        strcat(shape_s, tmp);
        total *= (size_t)shape[i];
    }
    int hlen = snprintf(hdr, sizeof(hdr),
        "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", dtype, shape_s);
    int tot = 10 + hlen + 1;
    int pad = ((tot + 63) / 64) * 64 - tot;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, elem_bytes, total, f);
    fclose(f);
}

int main(int argc, char **argv) {
    int Htex = (argc >= 2) ? atoi(argv[1]) : 128;
    int Wtex = Htex;
    int Himg = (argc >= 3) ? atoi(argv[2]) : 128;
    int Wimg = Himg;
    const int C = 3;
    const char *prefix = (argc >= 4) ? argv[3] : "/tmp/hy3d_bp";

    /* Synthetic tex_pos: a 1x1 quad in the XY plane at z=0.
     * tex_pos[row, col] = ((col+0.5)/W - 0.5, -((row+0.5)/H - 0.5), 0).
     * Note: row is Y-down in image convention but the world y is up, so
     * negate to keep the plane aligned with the view. */
    float *tex_pos = (float *)malloc((size_t)Htex * Wtex * 3 * sizeof(float));
    int *tex_cov = (int *)malloc((size_t)Htex * Wtex * sizeof(int));
    for (int r = 0; r < Htex; r++) {
        for (int c = 0; c < Wtex; c++) {
            int i = r * Wtex + c;
            float u = ((float)c + 0.5f) / (float)Wtex;
            float v = ((float)r + 0.5f) / (float)Htex;
            tex_pos[i*3+0] = u - 0.5f;
            tex_pos[i*3+1] = -(v - 0.5f);
            tex_pos[i*3+2] = 0.0f;
            tex_cov[i] = 1;
        }
    }

    /* Synthetic view image: gradient. R = col/W, G = row/H, B = 0.5. */
    float *image = (float *)malloc((size_t)Himg * Wimg * C * sizeof(float));
    for (int r = 0; r < Himg; r++) {
        for (int c = 0; c < Wimg; c++) {
            int i = r * Wimg + c;
            image[i*C+0] = (float)c / (float)(Wimg - 1);
            image[i*C+1] = (float)r / (float)(Himg - 1);
            image[i*C+2] = 0.5f;
        }
    }

    /* View: eye = (0, 0, 3), target = origin, up = (0, 1, 0).
     * World->camera row-vector form stored column-major: the standard
     * lookAt produces:
     *   s = right     = (1, 0, 0)
     *   u = up        = (0, 1, 0)
     *   f = -forward  = (0, 0, 1)     (camera looks down -Z)
     * w2c = [[s.x, s.y, s.z, -s.eye],
     *        [u.x, u.y, u.z, -u.eye],
     *        [f.x, f.y, f.z, -f.eye],
     *        [0,   0,   0,   1      ]]
     * Applied column-major, that is: m[0]=s.x, m[4]=s.y, m[8]=s.z, m[12]=-s.eye, ... */
    float w2c[16] = {0};
    w2c[0] =  1.f; w2c[4] = 0.f; w2c[8]  = 0.f; w2c[12] = 0.f;     /* row 0: s */
    w2c[1] =  0.f; w2c[5] = 1.f; w2c[9]  = 0.f; w2c[13] = 0.f;     /* row 1: u */
    w2c[2] =  0.f; w2c[6] = 0.f; w2c[10] = 1.f; w2c[14] = -3.f;   /* row 2: f, -f.eye = -(0,0,1)·(0,0,3) = -3 */
    w2c[3] =  0.f; w2c[7] = 0.f; w2c[11] = 0.f; w2c[15] = 1.f;

    /* For orthographic [-0.6, 0.6]^2 near/far [0.1, 10] the diagonal
     * factors are 2/(0.6 - -0.6) = 1.666... on x/y. */
    float proj00 = 2.f / 1.2f;
    float proj11 = 2.f / 1.2f;

    /* Depth map: in camera coords, the plane z_cam = w2c * world_z = w2c[10]*0 + w2c[14] = -3 */
    float *depth_img = (float *)malloc((size_t)Himg * Wimg * sizeof(float));
    for (int i = 0; i < Himg * Wimg; i++) depth_img[i] = -3.0f;

    float *visible_img = (float *)malloc((size_t)Himg * Wimg * sizeof(float));
    float *cos_img     = (float *)malloc((size_t)Himg * Wimg * sizeof(float));
    for (int i = 0; i < Himg * Wimg; i++) { visible_img[i] = 1.0f; cos_img[i] = 1.0f; }

    /* Init CUDA */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    CUmodule mod;
    int sm = cu_compile_kernels(&mod, dev,
                                cuda_paint_raster_kernels_src,
                                "hy3d_paint_raster", 1, "HY3D-PAINT");
    if (sm < 0) return 1;
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
    cuMemcpyHtoD(d_tex_pos, tex_pos, tex_n * 3 * sizeof(float));
    cuMemcpyHtoD(d_tex_cov, tex_cov, tex_n * sizeof(int));
    cuMemcpyHtoD(d_image,   image,   img_n * C * sizeof(float));
    cuMemcpyHtoD(d_depth,   depth_img, img_n * sizeof(float));
    cuMemcpyHtoD(d_vis,     visible_img, img_n * sizeof(float));
    cuMemcpyHtoD(d_cos,     cos_img,   img_n * sizeof(float));
    cuMemcpyHtoD(d_w2c,     w2c,       16 * sizeof(float));
    /* Zero the outputs */
    uint8_t *zero_tex = (uint8_t *)calloc(tex_n * C * sizeof(float), 1);
    cuMemcpyHtoD(d_out_tex, zero_tex, tex_n * C * sizeof(float));
    free(zero_tex);
    uint8_t *zero_cos = (uint8_t *)calloc(tex_n * sizeof(float), 1);
    cuMemcpyHtoD(d_out_cos, zero_cos, tex_n * sizeof(float));
    free(zero_cos);

    /* Launch */
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
    float *out_tex = (float *)malloc(tex_n * C * sizeof(float));
    float *out_cos = (float *)malloc(tex_n * sizeof(float));
    cuMemcpyDtoH(out_tex, d_out_tex, tex_n * C * sizeof(float));
    cuMemcpyDtoH(out_cos, d_out_cos, tex_n * sizeof(float));

    int filled = 0;
    for (size_t i = 0; i < tex_n; i++) if (out_cos[i] > 0.f) filled++;
    fprintf(stderr,
        "tex: %dx%d  img: %dx%d  filled = %d / %zu (%.1f%%)\n",
        Htex, Wtex, Himg, Wimg, filled, tex_n, 100.0 * filled / (double)tex_n);

    /* Round-trip diff against the input image, resampled at the exact
     * points our back-projection would query.
     * For this axis-aligned plane, x = col - 0.5, y = -(row - 0.5), and
     * the orthographic map sends (x, y) -> pixel (fx, fy) = ((px*0.5+0.5)*Wimg,
     * (py*0.5+0.5)*Himg) where px, py = x*proj00, y*proj11. */
    double max_err = 0.0, sum_err = 0.0;
    int n_err = 0;
    for (int r = 0; r < Htex; r++) {
        for (int c = 0; c < Wtex; c++) {
            int i = r * Wtex + c;
            float u = ((float)c + 0.5f) / (float)Wtex;
            float v = ((float)r + 0.5f) / (float)Htex;
            float x = u - 0.5f, y = -(v - 0.5f);
            float px = x * proj00, py = y * proj11;
            if (px < -1.f || px > 1.f || py < -1.f || py > 1.f) continue;
            float fx = (px * 0.5f + 0.5f) * (float)Wimg;
            float fy = (py * 0.5f + 0.5f) * (float)Himg;
            int ix = (int)fx, iy = (int)fy;
            float wx = fx - (float)ix, wy = fy - (float)iy;
            if (ix < 0) ix = 0; if (ix >= Wimg) ix = Wimg - 1;
            if (iy < 0) iy = 0; if (iy >= Himg) iy = Himg - 1;
            int ix1 = ix + 1; if (ix1 >= Wimg) ix1 = Wimg - 1;
            int iy1 = iy + 1; if (iy1 >= Himg) iy1 = Himg - 1;
            for (int k = 0; k < C; k++) {
                float a00 = image[(iy  * Wimg + ix ) * C + k];
                float a01 = image[(iy  * Wimg + ix1) * C + k];
                float a10 = image[(iy1 * Wimg + ix ) * C + k];
                float a11 = image[(iy1 * Wimg + ix1) * C + k];
                float ref_v = (a00 * (1.f - wx) + a01 * wx) * (1.f - wy)
                             + (a10 * (1.f - wx) + a11 * wx) * wy;
                float d = fabsf(out_tex[i * C + k] - ref_v);
                if (d > max_err) max_err = d;
                sum_err += d;
                n_err++;
            }
        }
    }
    fprintf(stderr, "Round-trip vs host bilinear ref:  max=%.3e mean=%.3e\n",
            max_err, sum_err / (double)n_err);

    char path[512];
    int sh3[3] = {Htex, Wtex, C};
    int sh2[2] = {Htex, Wtex};
    int sh_img[3] = {Himg, Wimg, C};
    snprintf(path, sizeof(path), "%s_out_tex.npy", prefix);
    write_npy(path, "<f4", sh3, 3, out_tex, sizeof(float));
    snprintf(path, sizeof(path), "%s_out_cos.npy", prefix);
    write_npy(path, "<f4", sh2, 2, out_cos, sizeof(float));
    snprintf(path, sizeof(path), "%s_image.npy", prefix);
    write_npy(path, "<f4", sh_img, 3, image, sizeof(float));
    fprintf(stderr, "Wrote %s_out_tex.npy, %s_out_cos.npy, %s_image.npy\n",
            prefix, prefix, prefix);

    cuMemFree(d_tex_pos); cuMemFree(d_tex_cov); cuMemFree(d_image);
    cuMemFree(d_depth); cuMemFree(d_vis); cuMemFree(d_cos); cuMemFree(d_w2c);
    cuMemFree(d_out_tex); cuMemFree(d_out_cos);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(tex_pos); free(tex_cov); free(image); free(depth_img);
    free(visible_img); free(cos_img); free(out_tex); free(out_cos);
    return 0;
}
