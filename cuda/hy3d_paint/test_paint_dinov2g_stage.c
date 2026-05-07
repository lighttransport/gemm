/*
 * test_paint_dinov2g_stage.c - smoke harness for paint_stage_dinov2g.
 *
 * Drives only the opaque API: create -> run -> destroy. Loads a [1,3,224,224]
 * f32 .npy (from ref/hy3d/dump_dinov2_giant.py), writes [1,257,1536] output.
 *
 * Usage:
 *   ./test_paint_dinov2g_stage <model.safetensors> <input.npy> [<out_prefix>]
 */

#include "../cuew.h"
#include "paint_stages.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEQ_LEN 257
#define HIDDEN  1536
#define IMG_SIZE 224

static float *read_npy_f32(const char *path, int *shape_out, int *ndims_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[6]; fread(magic, 1, 6, f);
    uint8_t ver[2]; fread(ver, 1, 2, f);
    uint16_t hlen; fread(&hlen, 2, 1, f);
    char *hdr = (char *)malloc(hlen + 1);
    fread(hdr, 1, hlen, f); hdr[hlen] = '\0';
    const char *sp = strstr(hdr, "'shape': (");
    size_t total = 1; int ndims = 0;
    if (sp) {
        sp += 10;
        while (*sp && *sp != ')') {
            if (*sp >= '0' && *sp <= '9') {
                int d = (int)strtol(sp, (char **)&sp, 10);
                shape_out[ndims++] = d; total *= (size_t)d;
            } else sp++;
        }
    }
    *ndims_out = ndims;
    free(hdr);
    float *data = (float *)malloc(total * sizeof(float));
    if (fread(data, sizeof(float), total, f) != total) { free(data); fclose(f); return NULL; }
    fclose(f); return data;
}

static void write_npy_f32(const char *path, const int *shape, int ndims, const float *data) {
    FILE *f = fopen(path, "wb");
    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256], shape_s[128] = ""; size_t total = 1;
    for (int i = 0; i < ndims; i++) {
        char tmp[32]; snprintf(tmp,sizeof(tmp),"%d, ",shape[i]);
        strcat(shape_s, tmp); total *= (size_t)shape[i];
    }
    int hlen = snprintf(hdr, sizeof(hdr),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shape_s);
    int tot = 10 + hlen + 1; int pad = ((tot+63)/64)*64 - tot;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), total, f);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <model.safetensors> <input.npy> [<out_prefix>]\n", argv[0]);
        return 1;
    }
    const char *weights = argv[1];
    const char *input_path = argv[2];
    const char *out_prefix = argc >= 4 ? argv[3] : "/tmp/hy3d_dinov2g_stage";

    int shape[4]; int ndims;
    float *input = read_npy_f32(input_path, shape, &ndims);
    if (!input) { fprintf(stderr, "ERROR: cannot read %s\n", input_path); return 1; }
    if (ndims != 4 || shape[0] != 1 || shape[1] != 3 ||
        shape[2] != IMG_SIZE || shape[3] != IMG_SIZE) {
        fprintf(stderr, "ERROR: expected [1,3,%d,%d]\n", IMG_SIZE, IMG_SIZE);
        return 1;
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    paint_stage_dinov2g *s = paint_stage_dinov2g_create(dev, weights);
    if (!s) { fprintf(stderr, "ERROR: stage create failed\n"); return 1; }

    float *output = (float *)malloc((size_t)SEQ_LEN * HIDDEN * sizeof(float));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    paint_stage_dinov2g_run(s, input, output);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
    fprintf(stderr, "DINOv2-giant stage: forward = %.1f ms\n", ms);

    double mn = output[0], mx = output[0], sum = 0.0;
    for (int i = 0; i < SEQ_LEN * HIDDEN; i++) {
        if (output[i] < mn) mn = output[i];
        if (output[i] > mx) mx = output[i];
        sum += output[i];
    }
    double mean = sum / (double)(SEQ_LEN * HIDDEN);
    double var = 0.0;
    for (int i = 0; i < SEQ_LEN * HIDDEN; i++) { double d = output[i] - mean; var += d*d; }
    double std = sqrt(var / (double)(SEQ_LEN * HIDDEN));
    fprintf(stderr, "output: min=%.4f max=%.4f mean=%.4f std=%.4f\n", mn, mx, mean, std);

    char path[512];
    int out_shape[3] = {1, SEQ_LEN, HIDDEN};
    snprintf(path, sizeof(path), "%s_output.npy", out_prefix);
    write_npy_f32(path, out_shape, 3, output);
    fprintf(stderr, "wrote %s\n", path);

    free(input); free(output);
    paint_stage_dinov2g_destroy(s);
    cuCtxDestroy(ctx);
    return 0;
}
