/*
 * verify_vith — diff our ViT-H/16 backbone output (sam-3d-body
 * vit_hmr_512_384 variant) against the pytorch reference dump.
 *
 * Usage:
 *   verify_vith --safetensors-dir <dir> --refdir /tmp/sam3d_body_vith_ref \
 *               [--threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR (produced by ref/sam3d-body/gen_image_ref.py
 * pointed at the vith ckpt):
 *   vith_input.npy   (1, 3, 512, 384) f32 — image tensor fed to the
 *                                            ViT-H backbone, already
 *                                            ImageNet-norm + W-axis
 *                                            cropped from 512×512.
 *   vith_tokens.npy  (1, 1280, 32, 24) f32 — backbone output (after
 *                                             last_norm, channels-first).
 *
 * Bypasses the runner: loads sam3d_body_vith.safetensors directly and
 * feeds the pre-normalized tensor via
 * `sam3d_body_vit_encode_from_normalized`.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"
#define SAM3D_BODY_VIT_IMPLEMENTATION
#include "sam3d_body_vit.h"
#include "npy_io.h"

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    /* 32-layer ViT-H with vanilla GELU MLP. Upstream forward runs in
     * bf16 (FP16_TYPE=bfloat16) so the reference is bf16-rounded
     * compounded over 32 blocks; our fp32 forward drifts max≈3.5e-1
     * mean≈1.4e-2 from that even with FP32 weights (verified
     * 2026-04-26). Gate set to track the bf16 floor; tighten if
     * we ever switch to a bf16 forward path. */
    float threshold = 5e-1f;
    int n_threads = 1, verbose = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--threshold F] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }

    char path[1024];

    /* Load preprocessed image (1, 3, 512, 384) f32 */
    int in_nd = 0, in_dims[8] = {0}, in_f32 = 0;
    snprintf(path, sizeof(path), "%s/vith_input.npy", refdir);
    float *in = (float *)npy_load(path, &in_nd, in_dims, &in_f32);
    if (!in || !in_f32 || in_nd != 4 ||
        in_dims[0] != 1 || in_dims[1] != 3) {
        fprintf(stderr, "[verify_vith] missing/invalid %s (need (1,3,H,W) f32)\n", path);
        free(in); return 3;
    }
    int H = in_dims[2], W = in_dims[3];

    /* Load reference tokens (1, dim, Hp, Wp) f32 */
    int ref_nd = 0, ref_dims[8] = {0}, ref_f32 = 0;
    snprintf(path, sizeof(path), "%s/vith_tokens.npy", refdir);
    float *ref = (float *)npy_load(path, &ref_nd, ref_dims, &ref_f32);
    if (!ref || !ref_f32 || ref_nd != 4 || ref_dims[0] != 1) {
        fprintf(stderr, "[verify_vith] missing/invalid %s (need (1,D,Ph,Pw) f32)\n", path);
        free(in); free(ref); return 3;
    }
    int D = ref_dims[1], Ph = ref_dims[2], Pw = ref_dims[3];
    fprintf(stderr, "[verify_vith] ref: input=(1,3,%d,%d) tokens=(1,%d,%d,%d)\n",
            H, W, D, Ph, Pw);
    (void)verbose;

    snprintf(path, sizeof(path), "%s/sam3d_body_vith.safetensors", sft_dir);
    fprintf(stderr, "[verify_vith] loading encoder: %s\n", path);
    sam3d_body_vit_model *m = sam3d_body_vit_load_safetensors(path);
    if (!m) { fprintf(stderr, "load failed\n"); free(in); free(ref); return 5; }

    sam3d_body_vit_result r = sam3d_body_vit_encode_from_normalized(
            m, in, W, H, n_threads);
    free(in);
    if (!r.tokens) {
        sam3d_body_vit_free(m); free(ref);
        return 6;
    }

    if (r.dim != D || r.grid_h != Ph || r.grid_w != Pw) {
        fprintf(stderr,
                "[verify_vith] shape mismatch: ours=(%d,%d,%d) ref=(1,%d,%d,%d)\n",
                r.n_patches, r.grid_h, r.dim, D, Ph, Pw);
        sam3d_body_vit_result_free(&r);
        sam3d_body_vit_free(m); free(ref);
        return 7;
    }

    /* ours: tokens[(py*Pw + px) * D + d]
     * ref:  ref[((0*D + d)*Ph + py)*Pw + px] (channels-first) */
    double sum = 0.0;
    float mx = 0.0f;
    int mx_d = 0, mx_py = 0, mx_px = 0;
    size_t n = (size_t)Ph * Pw * D;
    for (int py = 0; py < Ph; py++) {
        for (int px = 0; px < Pw; px++) {
            const float *op = r.tokens + (size_t)(py * Pw + px) * D;
            for (int d = 0; d < D; d++) {
                float rv = ref[((0 * D + d) * Ph + py) * Pw + px];
                float dv = fabsf(op[d] - rv);
                if (dv > mx) { mx = dv; mx_d = d; mx_py = py; mx_px = px; }
                sum += dv;
            }
        }
    }
    double mean_abs = sum / (double)n;
    /* mean budget tracks the observed bf16 forward floor (≈1.4e-2)
     * with a 1.5× headroom; a real bug typically blows up the mean
     * before the max so this is a tight catch even at the loose max. */
    float mean_gate = 2e-2f;
    fprintf(stderr,
            "[verify_vith] patches=(%d,%d) D=%d  "
            "max_abs=%.6e (d=%d py=%d px=%d) "
            "mean_abs=%.6e  (max_gate=%.1e mean_gate=%.1e)\n",
            Ph, Pw, D, mx, mx_d, mx_py, mx_px, mean_abs,
            threshold, mean_gate);
    int rc_out = (mx < threshold && mean_abs < mean_gate) ? 0 : 1;

    sam3d_body_vit_result_free(&r);
    sam3d_body_vit_free(m);
    free(ref);
    return rc_out;
}
