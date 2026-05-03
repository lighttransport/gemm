/*
 * verify_dense_pe — port-level sanity for sam3d_body_get_dense_pe
 * (PromptEncoder.pe_layer.forward(image_embedding_size)).
 *
 * Reference dump decoder_layer0_in__context_pe.npy is the per-layer
 * "context_pe" injected into the decoder; for layer 0 this is the raw
 * SAM dense PE before any decoder mixing.  It is dumped in token form
 * (1, H*W, 1280); we permute to CHW (1280, H, W) and diff against
 * sam3d_body_get_dense_pe(model, H, W, ...). DINOv3 uses 32x32;
 * ViT-H uses 32x24.
 *
 * Usage:
 *   verify_dense_pe --safetensors-dir <dir> --refdir /tmp/sam3d_body_ref
 *                   [--threshold F] [--backbone dinov3|vith]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "npy_io.h"

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void resolve_variant_path(const char *dir, const char *bucket,
                                 const char *tag, char *out, size_t out_sz)
{
    snprintf(out, out_sz, "%s/sam3d_body_%s_%s.safetensors",
             dir, tag, bucket);
    if (file_exists(out)) return;
    snprintf(out, out_sz, "%s/sam3d_body_%s.safetensors", dir, bucket);
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    const char *backbone = "dinov3";
    float threshold = 1e-5f;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--backbone") && i+1 < argc) {
            backbone = argv[++i];
            if (strcmp(backbone, "dinov3") && strcmp(backbone, "vith")) {
                fprintf(stderr, "unknown --backbone %s (use dinov3|vith)\n",
                        backbone);
                return 2;
            }
        }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--threshold F] [--backbone dinov3|vith]\n",
                argv[0]);
        return 2;
    }

    char dec_path[1024], mhr_path[1024];
    resolve_variant_path(sft_dir, "decoder", backbone,
                         dec_path, sizeof(dec_path));
    resolve_variant_path(sft_dir, "mhr_head", backbone,
                         mhr_path, sizeof(mhr_path));
    sam3d_body_decoder_model *m = sam3d_body_decoder_load(dec_path, mhr_path);
    if (!m) {
        fprintf(stderr, "[verify_dense_pe] decoder_load failed: %s\n", dec_path);
        return 3;
    }

    /* --- ref: token (1, 1024, 1280) → CHW (1280, 32, 32) --- */
    char ref_path[1024];
    snprintf(ref_path, sizeof(ref_path),
             "%s/decoder_layer0_in__context_pe.npy", refdir);
    int nd, dims[8];
    float *ref_tok = (float *)npy_load(ref_path, &nd, dims, NULL);
    if (!ref_tok) {
        fprintf(stderr, "[verify_dense_pe] missing %s\n", ref_path);
        sam3d_body_decoder_free(m); return 4;
    }
    int N, C;
    if (nd == 3)      { N = dims[1]; C = dims[2]; }
    else if (nd == 2) { N = dims[0]; C = dims[1]; }
    else {
        fprintf(stderr, "[verify_dense_pe] bad ndim=%d\n", nd);
        free(ref_tok); sam3d_body_decoder_free(m); return 5;
    }
    int H = 0, W = 0;
    {
        char img_path[1024];
        int img_nd = 0, img_dims[8] = {0};
        snprintf(img_path, sizeof(img_path),
                 "%s/image_embeddings_after_ray.npy", refdir);
        float *img = (float *)npy_load(img_path, &img_nd, img_dims, NULL);
        if (img && img_nd == 4 && img_dims[0] == 1 && img_dims[1] == C &&
            img_dims[2] > 0 && img_dims[3] > 0 &&
            img_dims[2] * img_dims[3] == N) {
            H = img_dims[2];
            W = img_dims[3];
        }
        free(img);
    }
    if (H == 0 || W == 0) {
        if (N == 1024) { H = 32; W = 32; }
        else if (N == 768) { H = 32; W = 24; }
    }
    if (H <= 0 || W <= 0 || N != H * W || C != 1280) {
        fprintf(stderr,
                "[verify_dense_pe] shape mismatch N=%d C=%d H=%d W=%d\n",
                N, C, H, W);
        free(ref_tok); sam3d_body_decoder_free(m); return 6;
    }

    float *ref_chw = (float *)malloc((size_t)C * H * W * sizeof(float));
    for (int n = 0; n < H * W; n++)
        for (int c = 0; c < C; c++)
            ref_chw[(size_t)c * H * W + n] = ref_tok[(size_t)n * C + c];
    free(ref_tok);

    /* --- ours --- */
    float *ours = (float *)malloc((size_t)C * H * W * sizeof(float));
    int rc = sam3d_body_get_dense_pe(m, H, W, /*n_threads=*/1, ours);
    if (rc != SAM3D_BODY_DECODER_E_OK) {
        fprintf(stderr, "[verify_dense_pe] get_dense_pe rc=%d\n", rc);
        free(ours); free(ref_chw); sam3d_body_decoder_free(m); return 7;
    }

    /* --- diff --- */
    double sum = 0.0; float mx = 0.0f; size_t mxi = 0;
    const size_t total = (size_t)C * H * W;
    for (size_t i = 0; i < total; i++) {
        float d = fabsf(ours[i] - ref_chw[i]);
        if (d > mx) { mx = d; mxi = i; }
        sum += d;
    }
    double mean = sum / (double)total;
    int fail = (mx >= threshold);
    fprintf(stderr, "[verify_dense_pe] %-32s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            "dense_pe", mx, mxi, mean,
            threshold, fail ? "FAIL" : "OK");

    free(ours); free(ref_chw);
    sam3d_body_decoder_free(m);
    return fail ? 1 : 0;
}
