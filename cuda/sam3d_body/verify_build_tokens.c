/*
 * verify_build_tokens (CUDA) — diff CUDA build_tokens output against
 * /tmp/sam3d_body_ref/decoder_layer0_in__{x,x_pe}.npy.
 *
 * Inputs:
 *   <refdir>/init_to_token_in.npy     (1,1,525)   f32
 *   <refdir>/prev_to_token_in.npy     (1,1,522)   f32
 *   <refdir>/prompt_to_token_in.npy   (1,1,1280)  f32
 *   <refdir>/decoder_layer0_in__x.npy    (1,145,1024) f32
 *   <refdir>/decoder_layer0_in__x_pe.npy (1,145,1024) f32
 */

#include "cuda_sam3d_body_runner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/npy_io.h"

static int diff_report(const char *label, const float *a, const float *b,
                       size_t n, float thresh, float mean_thresh, int tok_dim)
{
    if (n == 0) {
        fprintf(stderr, "[cuda verify_build_tokens] %s  empty diff FAIL\n",
                label);
        return 1;
    }
    double sum = 0.0; float mx = 0.0f; size_t mx_i = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) { mx = d; mx_i = i; }
        sum += d;
    }
    double mean = sum / (double)n;
    int tok = (int)(mx_i / (size_t)tok_dim);
    int dim = (int)(mx_i % (size_t)tok_dim);
    fprintf(stderr, "[cuda verify_build_tokens] %s  max_abs=%.6e (tok=%d dim=%d)  "
                    "mean_abs=%.6e  (max_gate=%.1e mean_gate=%.1e)\n",
            label, mx, tok, dim, mean, thresh, mean_thresh);
    return (mx < thresh && mean < mean_thresh) ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    /* Linear(1280→1024) f32 dot products → ~1e-4 max_abs vs PyTorch f32. */
    float threshold = 5e-4f;
    float mean_threshold = 1e-4f;
    int device = 0, verbose = 0;
    const char *precision = "bf16";
    cuda_sam3d_body_backbone_t backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--mean-threshold") && i+1 < argc) mean_threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--backbone") && i+1 < argc) {
            const char *v = argv[++i];
            if      (!strcmp(v, "dinov3")) backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;
            else if (!strcmp(v, "vith"))   backbone = CUDA_SAM3D_BODY_BACKBONE_VITH;
            else {
                fprintf(stderr, "unknown --backbone %s (use dinov3|vith)\n", v);
                return 2;
            }
        }
        else if (!strcmp(argv[i], "--device") && i+1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc) precision = argv[++i];
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir DIR --refdir DIR "
                        "[--threshold F] [--mean-threshold F] "
                        "[--backbone dinov3|vith] [--device N] "
                        "[--precision bf16|fp16] [-v]\n",
                argv[0]);
        return 2;
    }

    char path[1024];
    int nd = 0, dims[8] = {0};

    snprintf(path, sizeof(path), "%s/init_to_token_in.npy", refdir);
    float *init_in = (float *)npy_load(path, &nd, dims, NULL);
    if (!init_in || dims[nd-1] != 525) {
        fprintf(stderr, "[cuda verify_build_tokens] missing/invalid %s\n", path);
        free(init_in); return 3;
    }
    snprintf(path, sizeof(path), "%s/prev_to_token_in.npy", refdir);
    float *prev_in = (float *)npy_load(path, &nd, dims, NULL);
    if (!prev_in || dims[nd-1] != 522) {
        fprintf(stderr, "[cuda verify_build_tokens] missing/invalid %s\n", path);
        free(init_in); free(prev_in); return 3;
    }
    snprintf(path, sizeof(path), "%s/prompt_to_token_in.npy", refdir);
    float *prompt_in = (float *)npy_load(path, &nd, dims, NULL);
    if (!prompt_in || dims[nd-1] != 1280) {
        fprintf(stderr, "[cuda verify_build_tokens] missing/invalid %s\n", path);
        free(init_in); free(prev_in); free(prompt_in); return 3;
    }
    snprintf(path, sizeof(path), "%s/decoder_layer0_in__x.npy", refdir);
    float *ref_x = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_x || nd != 3 || dims[1] != 145 || dims[2] != 1024) {
        fprintf(stderr, "[cuda verify_build_tokens] missing/invalid %s\n", path);
        free(init_in); free(prev_in); free(prompt_in); free(ref_x); return 3;
    }
    snprintf(path, sizeof(path), "%s/decoder_layer0_in__x_pe.npy", refdir);
    float *ref_xpe = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_xpe || nd != 3 || dims[1] != 145 || dims[2] != 1024) {
        fprintf(stderr, "[cuda verify_build_tokens] missing/invalid %s\n", path);
        free(init_in); free(prev_in); free(prompt_in);
        free(ref_x); free(ref_xpe); return 3;
    }

    cuda_sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .image_size      = 512,
        .device_ordinal  = device,
        .verbose         = verbose,
        .precision       = precision,
        .backbone        = backbone,
    };
    cuda_sam3d_body_ctx *ctx = cuda_sam3d_body_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "create failed\n");
        free(init_in); free(prev_in); free(prompt_in);
        free(ref_x); free(ref_xpe); return 5;
    }

    const int N_TOK = 145, DIM = 1024;
    float *x    = (float *)calloc((size_t)N_TOK * DIM, sizeof(float));
    float *x_pe = (float *)calloc((size_t)N_TOK * DIM, sizeof(float));
    if (!x || !x_pe) {
        cuda_sam3d_body_destroy(ctx);
        free(init_in); free(prev_in); free(prompt_in);
        free(ref_x); free(ref_xpe); free(x); free(x_pe); return 6;
    }

    int rc = cuda_sam3d_body_debug_run_build_tokens(ctx, init_in, prev_in,
                                                    prompt_in, x, x_pe);
    if (rc != 0) {
        fprintf(stderr, "debug_run_build_tokens rc=%d\n", rc);
        cuda_sam3d_body_destroy(ctx);
        free(init_in); free(prev_in); free(prompt_in);
        free(ref_x); free(ref_xpe); free(x); free(x_pe); return 7;
    }

    int rc1 = diff_report("x   ", x,    ref_x,   (size_t)N_TOK * DIM,
                          threshold, mean_threshold, DIM);
    int rc2 = diff_report("x_pe", x_pe, ref_xpe, (size_t)N_TOK * DIM,
                          threshold, mean_threshold, DIM);

    cuda_sam3d_body_destroy(ctx);
    free(init_in); free(prev_in); free(prompt_in);
    free(ref_x); free(ref_xpe); free(x); free(x_pe);
    return (rc1 || rc2) ? 1 : 0;
}
