/*
 * verify_vith (CUDA) — diff CUDA ViT-H/16 encoder output against
 * /tmp/sam3d_body_vith_ref/vith_tokens.npy (shared with the CPU
 * cpu/sam3d_body/verify_vith).
 *
 * Inputs:
 *   <refdir>/vith_input.npy   (1, 3, 512, 384) f32 — pre-normalized image
 *   <refdir>/vith_tokens.npy  (1, 1280, 32, 24) f32 — patch grid output
 *
 * Loads sam3d_body_vith.safetensors via the runner with
 * cfg.backbone = CUDA_SAM3D_BODY_BACKBONE_VITH, feeds the pre-normalized
 * tensor via debug_set_normalized_input, runs encoder (no CLS prefix),
 * compares (768, 1280) flat output against the (1, 1280, 32, 24) ref.
 *
 * Gates track the bf16 forward floor of the upstream reference (the same
 * floor used by cpu/sam3d_body/verify_vith.c): max=5e-1, mean=2e-2.
 */

#include "cuda_sam3d_body_runner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/npy_io.h"

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    /* See cpu/sam3d_body/verify_vith.c for the gate justification: the
     * upstream ViT-H runs in bf16, drift compounds over 32 blocks and
     * floors max≈3.5e-1, mean≈1.4e-2. We track the same floor with a
     * tight mean and looser max. */
    float threshold = 5e-1f;
    float mean_threshold = 2e-2f;
    int device = 0, verbose = 0;
    const char *precision = "bf16";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--mean-threshold") && i+1 < argc) mean_threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--device") && i+1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc) precision = argv[++i];
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir DIR --refdir DIR "
                        "[--threshold F] [--mean-threshold F] "
                        "[--device N] [--precision bf16|fp16] [-v]\n",
                argv[0]);
        return 2;
    }

    char path[1024];

    /* Pre-normalized image (1,3,H,W) — H=512, W=384 for ViT-H. */
    int in_nd = 0, in_dims[8] = {0}, in_f32 = 0;
    snprintf(path, sizeof(path), "%s/vith_input.npy", refdir);
    float *in = (float *)npy_load(path, &in_nd, in_dims, &in_f32);
    if (!in || !in_f32 || in_nd != 4 || in_dims[0] != 1 || in_dims[1] != 3) {
        fprintf(stderr, "[cuda verify_vith] missing/invalid %s\n", path);
        free(in); return 3;
    }
    int H = in_dims[2], W = in_dims[3];

    /* Patch grid reference (1,D,Ph,Pw) — typically (1,1280,32,24). */
    int ref_nd = 0, ref_dims[8] = {0}, ref_f32 = 0;
    snprintf(path, sizeof(path), "%s/vith_tokens.npy", refdir);
    float *ref = (float *)npy_load(path, &ref_nd, ref_dims, &ref_f32);
    if (!ref || !ref_f32 || ref_nd != 4 || ref_dims[0] != 1) {
        fprintf(stderr, "[cuda verify_vith] missing/invalid %s\n", path);
        free(in); free(ref); return 3;
    }
    int D = ref_dims[1], Ph = ref_dims[2], Pw = ref_dims[3];
    fprintf(stderr, "[cuda verify_vith] ref: input=(1,3,%d,%d) tokens=(1,%d,%d,%d)\n",
            H, W, D, Ph, Pw);

    cuda_sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .image_size      = H,                             /* H=512 */
        .device_ordinal  = device,
        .verbose         = verbose,
        .precision       = precision,
        .backbone        = CUDA_SAM3D_BODY_BACKBONE_VITH,
    };
    cuda_sam3d_body_ctx *ctx = cuda_sam3d_body_create(&cfg);
    if (!ctx) { fprintf(stderr, "create failed\n"); free(in); free(ref); return 5; }

    int rc = cuda_sam3d_body_debug_set_normalized_input(ctx, in, H, W);
    if (rc != 0) { fprintf(stderr, "set_normalized_input rc=%d\n", rc);
        cuda_sam3d_body_destroy(ctx); free(in); free(ref); return 6; }
    free(in);

    rc = cuda_sam3d_body_run_encoder(ctx);
    if (rc != 0) { fprintf(stderr, "run_encoder rc=%d\n", rc);
        cuda_sam3d_body_destroy(ctx); free(ref); return 6; }

    int out_n = 0, out_dim = 0;
    cuda_sam3d_body_get_encoder_tokens(ctx, NULL, &out_n, &out_dim);
    if (out_dim != D) {
        fprintf(stderr, "dim mismatch: ours=%d ref=%d\n", out_dim, D);
        cuda_sam3d_body_destroy(ctx); free(ref); return 7;
    }
    int n_patches = Ph * Pw;
    if (out_n != n_patches) {
        fprintf(stderr, "n_tok mismatch: ours=%d ref patches=%d (ViT-H has no CLS)\n",
                out_n, n_patches);
        cuda_sam3d_body_destroy(ctx); free(ref); return 7;
    }
    float *ours = (float *)malloc((size_t)out_n * out_dim * sizeof(float));
    cuda_sam3d_body_get_encoder_tokens(ctx, ours, &out_n, &out_dim);

    /* ours[py*Pw + px, d]  ≙  ref[0, d, py, px] */
    double sum = 0.0;
    float mx = 0.0f;
    int mx_d = 0, mx_py = 0, mx_px = 0;
    size_t n = (size_t)n_patches * D;
    for (int py = 0; py < Ph; py++) {
        for (int px = 0; px < Pw; px++) {
            const float *op = ours + (size_t)(py * Pw + px) * D;
            for (int d = 0; d < D; d++) {
                float rv = ref[((0 * D + d) * Ph + py) * Pw + px];
                float dv = fabsf(op[d] - rv);
                if (dv > mx) { mx = dv; mx_d = d; mx_py = py; mx_px = px; }
                sum += dv;
            }
        }
    }
    double mean_abs = sum / (double)n;
    fprintf(stderr, "[cuda verify_vith] patches=(%d,%d) D=%d  "
                    "max_abs=%.6e (d=%d py=%d px=%d) "
                    "mean_abs=%.6e  (max_gate=%.1e mean_gate=%.1e)\n",
            Ph, Pw, D, mx, mx_d, mx_py, mx_px, mean_abs, threshold, mean_threshold);
    int rc_out = (mx < threshold && mean_abs < mean_threshold) ? 0 : 1;

    free(ours);
    cuda_sam3d_body_destroy(ctx);
    free(ref);
    return rc_out;
}
