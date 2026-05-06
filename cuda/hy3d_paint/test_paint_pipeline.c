/*
 * test_paint_pipeline.c - Top-level paint pipeline orchestrator (Phase 4.12).
 *
 * Drives the per-stage runners through their paint_stages.h opaque API. Each
 * stage runner lives in its own TU (paint_stage_*.c) so their file-local
 * helpers don't collide. paint_runtime.c owns the heavy SAFETENSORS /
 * CUDA_RUNNER_COMMON impls that all stages share.
 *
 * Current stages wired:
 *   - VAE decode  (paint_stage_vae)
 * Pending stages: UNet UniPC loop, view_maps render, DINOv2 conditioning,
 *                 back_project + bake + inpaint, OBJ/PNG writeout.
 *
 * Usage (smoke):
 *   ./test_paint_pipeline <vae.safetensors> <latent.npy> <out_recon.npy>
 */

#include "../cuew.h"
#include "paint_stages.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static float *read_npy_f32(const char *path, int *out_ndim,
                            uint64_t *out_shape, size_t *out_n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6)) { fclose(f); return NULL; }
    uint8_t ver[2]; if (fread(ver, 1, 2, f) != 2) { fclose(f); return NULL; }
    uint16_t hlen; if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return NULL; }
    char hdr[1024]; if (hlen >= sizeof(hdr) || fread(hdr, 1, hlen, f) != hlen) { fclose(f); return NULL; }
    hdr[hlen] = 0;
    const char *p = strstr(hdr, "'shape': ("); if (!p) { fclose(f); return NULL; }
    p += strlen("'shape': (");
    int nd = 0; uint64_t shape[8]; size_t total = 1;
    while (*p && *p != ')') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ')') break;
        char *end; uint64_t v = strtoull(p, &end, 10);
        shape[nd++] = v; total *= v; p = end;
    }
    *out_ndim = nd;
    for (int i = 0; i < nd; i++) out_shape[i] = shape[i];
    *out_n = total;
    float *buf = (float *)malloc(total * sizeof(float));
    if (fread(buf, sizeof(float), total, f) != total) { free(buf); fclose(f); return NULL; }
    fclose(f); return buf;
}

static void write_npy_f32(const char *path, const float *data, const int *shape, int ndim) {
    FILE *f = fopen(path, "wb"); if (!f) return;
    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256], shape_s[128] = ""; size_t total = 1;
    for (int i = 0; i < ndim; i++) { char tmp[32]; snprintf(tmp, sizeof(tmp), "%d, ", shape[i]); strcat(shape_s, tmp); total *= (size_t)shape[i]; }
    int hlen = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shape_s);
    int tot = 10 + hlen + 1; int pad = ((tot + 63) / 64) * 64 - tot;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), total, f);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <vae.safetensors> <latent.npy> <out_recon.npy>\n",
            argv[0]);
        return 1;
    }
    const char *vae_path = argv[1];
    const char *lat_path = argv[2];
    const char *out_path = argv[3];

    int nd; uint64_t shape[8]; size_t total;
    float *lat = read_npy_f32(lat_path, &nd, shape, &total);
    if (!lat) { fprintf(stderr, "ERROR: cannot read %s\n", lat_path); return 1; }
    int B = 1, IC, IH, IW;
    if (nd == 3 && shape[0] == 4) { IC = 4; IH = (int)shape[1]; IW = (int)shape[2]; }
    else if (nd == 4 && shape[1] == 4) { B = (int)shape[0]; IC = 4; IH = (int)shape[2]; IW = (int)shape[3]; }
    else { fprintf(stderr, "ERROR: expected [4,H,W] or [B,4,H,W]\n"); return 1; }
    int OC = 3, OH = IH * 8, OW = IW * 8;
    fprintf(stderr, "[pipeline] latent [%d,%d,%d,%d] -> recon [%d,%d,%d,%d]\n",
            B, IC, IH, IW, B, OC, OH, OW);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    paint_stage_vae *vae = paint_stage_vae_create(dev, vae_path);
    if (!vae) return 1;
    fprintf(stderr, "[pipeline] VAE stage ready\n");

    /* Worst-case workspace as in test_paint_vae main(). */
    int LH = IH, LW = IW;
    size_t cands[] = {
        (size_t)512 * LH * LW,
        (size_t)512 * (LH*2) * (LW*2),
        (size_t)512 * (LH*4) * (LW*4),
        (size_t)256 * (LH*4) * (LW*4),
        (size_t)256 * (LH*8) * (LW*8),
        (size_t)128 * (LH*8) * (LW*8),
    };
    size_t max_n = 0;
    for (size_t i = 0; i < sizeof(cands)/sizeof(cands[0]); i++)
        if (cands[i] > max_n) max_n = cands[i];
    size_t attn_n = (size_t)512 * IH * IW;

    CUdeviceptr d_in, d_out, d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync;
    cuMemAlloc(&d_in,  IC * (size_t)IH * IW * sizeof(float));
    cuMemAlloc(&d_out, OC * (size_t)OH * OW * sizeof(float));
    cuMemAlloc(&d_a, max_n * sizeof(float));
    cuMemAlloc(&d_b, max_n * sizeof(float));
    cuMemAlloc(&d_t1, max_n * sizeof(float));
    cuMemAlloc(&d_t2, max_n * sizeof(float));
    cuMemAlloc(&d_qnc, attn_n * sizeof(float));
    cuMemAlloc(&d_knc, attn_n * sizeof(float));
    cuMemAlloc(&d_vnc, attn_n * sizeof(float));
    cuMemAlloc(&d_ync, attn_n * sizeof(float));

    size_t in_per = (size_t)IC * IH * IW;
    size_t out_per = (size_t)OC * OH * OW;
    float *out_buf = (float *)malloc((size_t)B * out_per * sizeof(float));
    for (int bi = 0; bi < B; bi++) {
        cuMemcpyHtoD(d_in, lat + (size_t)bi * in_per, in_per * sizeof(float));
        paint_stage_vae_decode(vae, d_in, IH, IW, d_out,
                                d_a, d_b, d_t1, d_t2,
                                d_qnc, d_knc, d_vnc, d_ync);
        cuCtxSynchronize();
        cuMemcpyDtoH(out_buf + (size_t)bi * out_per, d_out, out_per * sizeof(float));
    }

    if (B == 1) { int sh3[3] = {OC, OH, OW}; write_npy_f32(out_path, out_buf, sh3, 3); }
    else        { int sh4[4] = {B, OC, OH, OW}; write_npy_f32(out_path, out_buf, sh4, 4); }
    fprintf(stderr, "[pipeline] wrote %s\n", out_path);

    free(out_buf); free(lat);
    paint_stage_vae_destroy(vae);
    cuCtxDestroy(ctx);
    return 0;
}
