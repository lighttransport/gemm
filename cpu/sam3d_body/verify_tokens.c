/*
 * verify_tokens — diff sam3d_body token-construction (step 4d) against
 * the pytorch reference dump.
 *
 * Construction (single image, body branch, empty prompts):
 *   x    [145, 1024] = [init_token, prev_token, prompt_token,
 *                       hand_box_embedding(2), keypoint_embedding(70),
 *                       keypoint3d_embedding(70)]
 *   x_pe [145, 1024] = [0, prev_token, prompt_token, zeros...]
 *
 * Inputs pulled from the ref dump:
 *   init_to_token_in.npy     (1,1,525)   f32
 *   prev_to_token_in.npy     (1,1,522)   f32
 *   prompt_to_token_in.npy   (1,1,1280)  f32
 *
 * Reference outputs:
 *   decoder_layer0_in__x.npy    (1,145,1024) f32
 *   decoder_layer0_in__x_pe.npy (1,145,1024) f32
 *
 * Usage:
 *   verify_tokens --safetensors-dir <dir> --refdir <dir>
 *                 [--threshold F] [-v]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "npy_io.h"

static void diff_report(const char *label, const float *a, const float *b,
                        size_t n, float threshold, int *worst_tok, int tok_dim,
                        int *rc_out)
{
    double sum = 0.0; float mx = 0.0f; size_t mx_i = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) { mx = d; mx_i = i; }
        sum += d;
    }
    double mean = sum / (double)n;
    int tok = (int)(mx_i / (size_t)tok_dim);
    int dim = (int)(mx_i % (size_t)tok_dim);
    float mean_gate = threshold * 0.15f;
    fprintf(stderr, "[verify_tokens] %s  max_abs=%.6e (tok=%d dim=%d)  "
                    "mean_abs=%.6e  (max_gate=%.1e mean_gate=%.1e)\n",
            label, mx, tok, dim, mean, threshold, mean_gate);
    if (worst_tok) *worst_tok = tok;
    if (mx >= threshold || mean >= mean_gate) *rc_out = 1;
}

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
    /* Linear(1280→1024) + f32 dot-products accumulate ~1e-4 abs error vs
     * the torch reference. Gates match observed f32 floor with small margin. */
    float threshold = 5e-4f;
    int verbose = 0;
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
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--threshold F] [--backbone dinov3|vith] [-v]\n", argv[0]);
        return 2;
    }
    (void)verbose;

    char path[1024];
    int nd = 0, dims[8] = {0};

    snprintf(path, sizeof(path), "%s/init_to_token_in.npy", refdir);
    float *init_in = (float *)npy_load(path, &nd, dims, NULL);
    if (!init_in || dims[nd-1] != 525) {
        fprintf(stderr, "[verify_tokens] missing/invalid %s\n", path);
        free(init_in); return 3;
    }
    snprintf(path, sizeof(path), "%s/prev_to_token_in.npy", refdir);
    float *prev_in = (float *)npy_load(path, &nd, dims, NULL);
    if (!prev_in || dims[nd-1] != 522) {
        fprintf(stderr, "[verify_tokens] missing/invalid %s\n", path);
        free(init_in); free(prev_in); return 3;
    }
    snprintf(path, sizeof(path), "%s/prompt_to_token_in.npy", refdir);
    float *prompt_in = (float *)npy_load(path, &nd, dims, NULL);
    if (!prompt_in || dims[nd-1] != 1280) {
        fprintf(stderr, "[verify_tokens] missing/invalid %s\n", path);
        free(init_in); free(prev_in); free(prompt_in); return 3;
    }

    snprintf(path, sizeof(path), "%s/decoder_layer0_in__x.npy", refdir);
    float *ref_x = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_x || nd != 3 || dims[0] != 1 || dims[1] != 145 || dims[2] != 1024) {
        fprintf(stderr, "[verify_tokens] missing/invalid %s\n", path);
        free(init_in); free(prev_in); free(prompt_in); free(ref_x); return 3;
    }
    snprintf(path, sizeof(path), "%s/decoder_layer0_in__x_pe.npy", refdir);
    float *ref_xpe = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_xpe || nd != 3 || dims[0] != 1 || dims[1] != 145 || dims[2] != 1024) {
        fprintf(stderr, "[verify_tokens] missing/invalid %s\n", path);
        free(init_in); free(prev_in); free(prompt_in);
        free(ref_x); free(ref_xpe); return 3;
    }

    char mhr_path[1024];
    resolve_variant_path(sft_dir, "decoder", backbone, path, sizeof(path));
    resolve_variant_path(sft_dir, "mhr_head", backbone,
                         mhr_path, sizeof(mhr_path));
    sam3d_body_decoder_model *m = sam3d_body_decoder_load(path, mhr_path);
    if (!m) { free(init_in); free(prev_in); free(prompt_in);
              free(ref_x); free(ref_xpe); return 5; }

    const int N_TOK = 145, DIM = 1024;
    float *x    = (float *)calloc((size_t)N_TOK * DIM, sizeof(float));
    float *x_pe = (float *)calloc((size_t)N_TOK * DIM, sizeof(float));
    if (!x || !x_pe) {
        sam3d_body_decoder_free(m);
        free(init_in); free(prev_in); free(prompt_in);
        free(ref_x); free(ref_xpe); free(x); free(x_pe); return 6;
    }

    int rc = sam3d_body_build_tokens(m, init_in, prev_in, prompt_in, 1, x, x_pe);
    if (rc != SAM3D_BODY_DECODER_E_OK) {
        fprintf(stderr, "[verify_tokens] build_tokens rc=%d\n", rc);
        sam3d_body_decoder_free(m);
        free(init_in); free(prev_in); free(prompt_in);
        free(ref_x); free(ref_xpe); free(x); free(x_pe); return 7;
    }

    int rc_out = 0;
    int worst_x = 0, worst_pe = 0;
    size_t n = (size_t)N_TOK * DIM;
    diff_report("x   ", x,    ref_x,   n, threshold, &worst_x, DIM, &rc_out);
    diff_report("x_pe", x_pe, ref_xpe, n, threshold, &worst_pe, DIM, &rc_out);

    sam3d_body_decoder_free(m);
    free(init_in); free(prev_in); free(prompt_in);
    free(ref_x); free(ref_xpe); free(x); free(x_pe);
    return rc_out;
}
