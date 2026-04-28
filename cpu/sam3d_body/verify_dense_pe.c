/*
 * verify_dense_pe — port-level sanity for sam3d_body_get_dense_pe
 * (PromptEncoder.pe_layer.forward(image_embedding_size)).
 *
 * Reference dump decoder_layer0_in__context_pe.npy is the per-layer
 * "context_pe" injected into the decoder; for layer 0 this is the raw
 * SAM dense PE before any decoder mixing.  It is dumped in token form
 * (1, 1024, 1280); we permute to CHW (1280, 32, 32) and diff against
 * sam3d_body_get_dense_pe(model, 32, 32, ...).
 *
 * Usage:
 *   verify_dense_pe --safetensors-dir <dir> --refdir /tmp/sam3d_body_ref
 *                   [--threshold F]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "npy_io.h"

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    float threshold = 1e-5f;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--threshold F]\n",
                argv[0]);
        return 2;
    }

    char dec_path[1024], mhr_path[1024];
    snprintf(dec_path, sizeof(dec_path),
             "%s/sam3d_body_decoder.safetensors", sft_dir);
    snprintf(mhr_path, sizeof(mhr_path),
             "%s/sam3d_body_mhr_head.safetensors", sft_dir);
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
    /* Expected shape (1, 1024, 1280) or (1024, 1280). */
    int N, C;
    if (nd == 3)      { N = dims[1]; C = dims[2]; }
    else if (nd == 2) { N = dims[0]; C = dims[1]; }
    else {
        fprintf(stderr, "[verify_dense_pe] bad ndim=%d\n", nd);
        free(ref_tok); sam3d_body_decoder_free(m); return 5;
    }
    const int H = 32, W = 32;
    if (N != H * W || C != 1280) {
        fprintf(stderr, "[verify_dense_pe] shape mismatch N=%d C=%d (want %d,1280)\n",
                N, C, H * W);
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
            "dense_pe (1280,32,32)", mx, mxi, mean,
            threshold, fail ? "FAIL" : "OK");

    free(ours); free(ref_chw);
    sam3d_body_decoder_free(m);
    return fail ? 1 : 0;
}
