/*
 * verify_prompt.c — CPU verifier for SAM2 prompt encoder + image-wide PE.
 *
 * Validates three tensors from /tmp/sam2_trace/:
 *   prompt_sparse.npy  (1,1,2,256)   point embedding (pad=True → 1 point + 1 pad)
 *   prompt_dense.npy   (1,256,64,64) no_mask_embed broadcast
 *   md_image_pe.npy    (1,256,64,64) image-wide positional embedding
 *
 * All computed via tiny ops; no CUDA needed.
 *
 * Usage: ./verify_prompt <model.safetensors> <refdir> [x y]
 *   (x,y) defaults to (256,256) matching gen_prompt_mask_trace_ref.py.
 */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float *read_npy_f32(const char *path, int dims[5], int *ndims) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    uint8_t h10[10]; if (fread(h10, 1, 10, f) != 10) { fclose(f); return NULL; }
    if (memcmp(h10, "\x93NUMPY", 6)) { fclose(f); return NULL; }
    uint16_t hlen = (uint16_t)(h10[8] | (h10[9] << 8));
    char *hdr = (char *)malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = '\0';
    if (!strstr(hdr, "'descr': '<f4'")) { free(hdr); fclose(f); return NULL; }
    char *p = strchr(hdr, '('), *q = strchr(hdr, ')');
    p++; int n = 0;
    while (p < q && n < 5) {
        while (p < q && (*p < '0' || *p > '9')) p++;
        if (p >= q) break;
        dims[n++] = (int)strtol(p, &p, 10);
    }
    free(hdr);
    size_t cnt = 1; for (int i = 0; i < n; i++) cnt *= (size_t)dims[i];
    float *x = (float *)malloc(cnt * sizeof(float));
    if (fread(x, sizeof(float), cnt, f) != cnt) { free(x); fclose(f); return NULL; }
    fclose(f); *ndims = n; return x;
}

static int load_tensor(st_context *st, const char *name, float **out) {
    int i = safetensors_find(st, name);
    if (i < 0) { fprintf(stderr, "missing %s\n", name); return -1; }
    if (strcmp(safetensors_dtype(st, i), "F32")) { fprintf(stderr, "not f32: %s\n", name); return -1; }
    *out = (float *)safetensors_data(st, i);
    return 0;
}

static void diff(const char *name, const float *a, const float *b, size_t n) {
    double mad = 0.0; float mxd = 0.f;
    for (size_t i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d > mxd) mxd = d; mad += d; }
    mad /= (double)n;
    fprintf(stderr, "  %-18s: max_abs=%.6g mean_abs=%.6g\n", name, mxd, mad);
}

/* Positional embedding: coords (N, 2) in [0,1] -> (N, 256).
 *   c = 2*coords - 1                         (N,2)
 *   c = c @ W                                (N,128)  where W is (2,128)
 *   c = 2*pi*c
 *   out = concat(sin(c), cos(c))             (N,256)
 */
static void pos_embed(const float *coords01, int N, const float *W, float *out) {
    for (int n = 0; n < N; n++) {
        float x = 2.f * coords01[n*2+0] - 1.f;
        float y = 2.f * coords01[n*2+1] - 1.f;
        for (int k = 0; k < 128; k++) {
            float v = x * W[0*128 + k] + y * W[1*128 + k];
            v *= 2.f * (float)M_PI;
            out[n*256 + k]       = sinf(v);
            out[n*256 + k + 128] = cosf(v);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <model.safetensors> <refdir> [x y]\n", argv[0]); return 1; }
    const char *ckpt = argv[1]; const char *refdir = argv[2]; char path[1024];
    /* Points are in 1024² model-input space (post-processor scaling).
     * Default matches trace: image 640x426, click (256,256) → (409.6, 615.3615).
     */
    float px = argc > 3 ? (float)atof(argv[3]) : 409.6f;
    float py = argc > 4 ? (float)atof(argv[4]) : 615.3615f;

    st_context *st = safetensors_open(ckpt);
    if (!st) { fprintf(stderr, "safetensors_open failed\n"); return 3; }

    float *W_prompt, *W_image, *not_a_point, *no_mask, *point_embed;
    if (load_tensor(st, "prompt_encoder.shared_embedding.positional_embedding", &W_prompt)) return 4;
    if (load_tensor(st, "shared_image_embedding.positional_embedding", &W_image)) return 4;
    if (load_tensor(st, "prompt_encoder.not_a_point_embed.weight", &not_a_point)) return 4;
    if (load_tensor(st, "prompt_encoder.no_mask_embed.weight", &no_mask)) return 4;
    if (load_tensor(st, "prompt_encoder.point_embed.weight", &point_embed)) return 4;

    /* ---- sparse prompt embedding ---- */
    /* Points (after +0.5, pad=True appends (0,0) with label=-1):
     *   p0 = (x+0.5, y+0.5), label=1
     *   p1 = (0,0),           label=-1
     * Normalize by 1024 → shape (2, 2). Run pos_embed → (2, 256).
     * Replace pad row with not_a_point_embed. Add point_embed[label] to real point.
     */
    float coords01[4] = {
        (px + 0.5f) / 1024.f, (py + 0.5f) / 1024.f,
        0.5f / 1024.f,        0.5f / 1024.f,
    };
    float sparse[2*256];
    pos_embed(coords01, 2, W_prompt, sparse);
    /* Row 0: label=1 → add point_embed.weight[1] */
    for (int k = 0; k < 256; k++) sparse[k] += point_embed[1*256 + k];
    /* Row 1: label=-1 → replace with not_a_point_embed */
    for (int k = 0; k < 256; k++) sparse[256 + k] = not_a_point[k];

    /* ---- dense (no_mask) ---- */
    /* (B=1, 256, 64, 64) broadcast of no_mask_embed (1,256). BCHW layout. */
    float *dense = (float *)malloc((size_t)256*64*64*sizeof(float));
    for (int c = 0; c < 256; c++)
        for (int i = 0; i < 64*64; i++)
            dense[c*64*64 + i] = no_mask[c];

    /* ---- image-wide PE ---- */
    /* grid = ones(64,64); y = cumsum0(-0.5)/64; x = cumsum1(-0.5)/64
     *   → y[i,j]=(i+0.5)/64, x[i,j]=(j+0.5)/64
     * coords = stack([x,y], -1). Run pos_embed (using shared_image_embedding W).
     * Permute HWC → CHW, unsqueeze batch. Output (1,256,64,64).
     */
    float *coords = (float *)malloc((size_t)64*64*2*sizeof(float));
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++) {
            coords[(i*64+j)*2+0] = ((float)j + 0.5f) / 64.f;
            coords[(i*64+j)*2+1] = ((float)i + 0.5f) / 64.f;
        }
    float *pe_hwc = (float *)malloc((size_t)64*64*256*sizeof(float));
    pos_embed(coords, 64*64, W_image, pe_hwc);
    /* Permute HWC→CHW */
    float *pe_chw = (float *)malloc((size_t)256*64*64*sizeof(float));
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++)
            for (int c = 0; c < 256; c++)
                pe_chw[c*64*64 + i*64 + j] = pe_hwc[(i*64+j)*256 + c];

    /* ---- Compare ---- */
    int d[5], nd; float *ref;

    snprintf(path, sizeof(path), "%s/prompt_sparse.npy", refdir);
    ref = read_npy_f32(path, d, &nd);
    if (ref) { diff("prompt_sparse", sparse, ref, 2*256); free(ref); }

    snprintf(path, sizeof(path), "%s/prompt_dense.npy", refdir);
    ref = read_npy_f32(path, d, &nd);
    if (ref) { diff("prompt_dense", dense, ref, (size_t)256*64*64); free(ref); }

    snprintf(path, sizeof(path), "%s/md_image_pe.npy", refdir);
    ref = read_npy_f32(path, d, &nd);
    if (ref) { diff("md_image_pe", pe_chw, ref, (size_t)256*64*64); free(ref); }

    free(dense); free(coords); free(pe_hwc); free(pe_chw);
    safetensors_close(st);
    return 0;
}
