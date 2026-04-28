/*
 * verify_ss_decoder — diff the SS-VAE 3D-conv decoder output (64³ occupancy
 * logits) against /tmp/sam3d_ref/ss_dec_out.npy.
 *
 * Architecture is identical to TRELLIS.2's SS decoder (3 stages, pixel_shuffle
 * upsample, GroupNorm+SiLU+Conv ResBlocks) and the safetensors key layout
 * matches exactly, so we reuse `common/trellis2_ss_decoder.h` without
 * modification.
 *
 * Expects in $REFDIR:
 *   ss_dec_in.npy   [8, 16, 16, 16]  f32   — SS-DiT output for the shape stream,
 *                                           reshaped from (4096, 8) to NCDHW
 *   ss_dec_out.npy  [1, 64, 64, 64]  f32   — reference occupancy logits
 */
#include "safetensors.h"            /* declarations only; impl lives in sam3d_runner.c */
#include "trellis2_ss_decoder.h"    /* impl linked in from sam3d_runner.c */

#include "sam3d_runner.h"
#include "verify_npy.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *ckpt = NULL, *refdir = NULL, *sft_dir = NULL;
    int verbose = 0, n_threads = 1;
    float threshold = 5e-2f;  /* bf16 floor */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt    = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc)       n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v"))                     verbose = 1;
        else if (!strcmp(argv[i], "--use-ref-inputs"))       { /* implied */ }
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!ckpt) {
        fprintf(stderr, "Usage: %s --ckpt <pipeline.yaml> [--safetensors-dir <dir>] "
                        "--refdir <dir> [--threshold F] [-t N] [-v]\n", argv[0]);
        return 2;
    }

    sam3d_config cfg = {
        .pipeline_yaml = ckpt,
        .safetensors_dir = sft_dir,
        .seed = 42,
        .verbose = verbose,
    };
    sam3d_ctx *ctx = sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_create failed\n"); return 5; }

    char path[1280];
    snprintf(path, sizeof(path), "%s/sam3d_ss_decoder.safetensors",
             sam3d_safetensors_dir(ctx));
    fprintf(stderr, "[verify_ss_decoder] loading %s\n", path);

    t2_ss_dec *dec = t2_ss_dec_load(path);
    if (!dec) {
        fprintf(stderr, "[verify_ss_decoder] loader failed\n");
        sam3d_destroy(ctx);
        return 3;
    }
    fprintf(stderr, "[verify_ss_decoder] loaded OK (GN groups=%d)\n", dec->gn_groups);

    if (!refdir) {
        fprintf(stderr, "[verify_ss_decoder] no --refdir; loader smoke-test only\n");
        t2_ss_dec_free(dec);
        sam3d_destroy(ctx);
        return 0;
    }

    /* Load input latent (accept [8,16,16,16], [16,16,16,8] or [4096,8]). */
    snprintf(path, sizeof(path), "%s/ss_dec_in.npy", refdir);
    int nd = 0, dims[8] = {0}, is_f32 = 0;
    float *in = (float *)verify_read_npy(path, &nd, dims, &is_f32);
    if (!in || !is_f32) {
        fprintf(stderr, "missing/bad %s\n", path);
        free(in); t2_ss_dec_free(dec); sam3d_destroy(ctx); return 4;
    }
    int in_n = 1;
    for (int i = 0; i < nd; i++) in_n *= dims[i];
    if (in_n != 8 * 16 * 16 * 16) {
        fprintf(stderr, "bad ss_dec_in numel=%d (want 32768)\n", in_n);
        free(in); t2_ss_dec_free(dec); sam3d_destroy(ctx); return 4;
    }
    /* If input is in (N=4096, C=8) layout, transpose to NCDHW=(8,16,16,16).
     * The shape latent from the DiT comes out [token, channel]. */
    float *nchw = (float *)malloc((size_t)in_n * sizeof(float));
    if (nd == 2 && dims[0] == 4096 && dims[1] == 8) {
        for (int n = 0; n < 4096; n++)
            for (int c = 0; c < 8; c++)
                nchw[c * 4096 + n] = in[n * 8 + c];
    } else if (nd == 4 && dims[0] == 16 && dims[1] == 16 && dims[2] == 16 && dims[3] == 8) {
        for (int n = 0; n < 4096; n++)
            for (int c = 0; c < 8; c++)
                nchw[c * 4096 + n] = in[n * 8 + c];
    } else {
        memcpy(nchw, in, (size_t)in_n * sizeof(float));
    }
    free(in);

    /* Load reference output. */
    snprintf(path, sizeof(path), "%s/ss_dec_out.npy", refdir);
    int ref_nd = 0, ref_dims[8] = {0}, ref_f32 = 0;
    float *ref = (float *)verify_read_npy(path, &ref_nd, ref_dims, &ref_f32);
    if (!ref || !ref_f32) {
        fprintf(stderr, "missing/bad %s\n", path);
        free(nchw); free(ref); t2_ss_dec_free(dec); sam3d_destroy(ctx); return 4;
    }
    int ref_n = 1;
    for (int i = 0; i < ref_nd; i++) ref_n *= ref_dims[i];
    if (ref_n != 64 * 64 * 64) {
        fprintf(stderr, "bad ss_dec_out numel=%d (want 262144)\n", ref_n);
        free(nchw); free(ref); t2_ss_dec_free(dec); sam3d_destroy(ctx); return 4;
    }

    /* Forward. */
    fprintf(stderr, "[verify_ss_decoder] running forward (n_threads=%d)\n", n_threads);
    float *out = t2_ss_dec_forward(dec, nchw, n_threads);
    if (!out) {
        fprintf(stderr, "forward failed\n");
        free(nchw); free(ref); t2_ss_dec_free(dec); sam3d_destroy(ctx); return 5;
    }

    double mean = 0.0;
    float mx = verify_max_abs(out, ref, ref_n, &mean);
    int ok = (mx < threshold);
    fprintf(stderr, "[verify_ss_decoder] 64³  max_abs=%.6e  mean_abs=%.6e  %s\n",
            (double)mx, mean, ok ? "OK" : "FAIL");

    free(out); free(nchw); free(ref);
    t2_ss_dec_free(dec);
    sam3d_destroy(ctx);
    return ok ? 0 : 1;
}
