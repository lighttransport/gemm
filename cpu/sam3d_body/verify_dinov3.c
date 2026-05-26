/*
 * verify_dinov3 — diff our DINOv3 encoder output against the pytorch
 * reference dump for sam-3d-body-dinov3.
 *
 * Usage:
 *   verify_dinov3 --safetensors-dir <dir> --refdir /tmp/sam3d_body_ref \
 *                 [--threshold F] [--mean-threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR (produced by ref/sam3d-body/gen_image_ref.py):
 *   dinov3_input.npy   (1, 3, 512, 512) f32  — image tensor fed to the
 *                                               DINOv3 backbone, already
 *                                               mean/std-normalized by
 *                                               SAM3DBody meta-arch.
 *   dinov3_tokens.npy  (1, 1280, 32, 32) f32 — backbone output (patch
 *                                               tokens only, with final
 *                                               learned LN applied).
 *
 * Bypasses the runner: loads sam3d_body_dinov3.safetensors directly and
 * feeds the pre-normalized tensor via `dinov3_encode_from_normalized`.
 * This eliminates the u8 round-trip on the input so we can tell apart
 * quantization drift from f16-matmul drift during diagnosis.
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
#define DINOV3_IMPLEMENTATION
#include "dinov3.h"
#include "npy_io.h"

static int ref_backbone_is_float32(const char *refdir)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/backbone_dtype.txt", refdir);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[64] = {0};
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, "float32") != NULL ||
           strstr(buf, "torch.float32") != NULL;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    /* 32-layer ViT with f16 large weights accumulates visible max_abs
     * outliers vs the upstream BF16 reference. Square 512x512 refs sit
     * around 1.4e-1; rectangular 512x384 refs currently sit around 3.7e-1.
     * Pick the default gate after reading the ref geometry. */
    float threshold = -1.0f;
    float mean_threshold = -1.0f;
    int n_threads = 1, verbose = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--mean-threshold")  && i+1 < argc) mean_threshold = strtof(argv[++i], NULL);
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
                "[--threshold F] [--mean-threshold F] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }

    /* Load the pre-normalized input and the reference token grid. */
    char path[1024];
    int in_nd = 0, in_dims[8] = {0}, in_f32 = 0;
    snprintf(path, sizeof(path), "%s/dinov3_input.npy", refdir);
    float *in = (float *)npy_load(path, &in_nd, in_dims, &in_f32);
    if (!in || !in_f32 || in_nd != 4 ||
        in_dims[0] != 1 || in_dims[1] != 3) {
        fprintf(stderr, "[verify_dinov3] missing/invalid %s (need (1,3,H,W) f32)\n", path);
        free(in); return 3;
    }
    int H = in_dims[2], W = in_dims[3];

    int ref_nd = 0, ref_dims[8] = {0}, ref_f32 = 0;
    snprintf(path, sizeof(path), "%s/dinov3_tokens.npy", refdir);
    float *ref = (float *)npy_load(path, &ref_nd, ref_dims, &ref_f32);
    if (!ref || !ref_f32 || ref_nd != 4 || ref_dims[0] != 1) {
        fprintf(stderr, "[verify_dinov3] missing/invalid %s (need (1,D,Ph,Pw) f32)\n", path);
        free(in); free(ref); return 3;
    }
    int D = ref_dims[1], Ph = ref_dims[2], Pw = ref_dims[3];
    fprintf(stderr, "[verify_dinov3] ref: input=(1,3,%d,%d) tokens=(1,%d,%d,%d)\n",
            H, W, D, Ph, Pw);
    (void)verbose;
    int ref_is_f32 = ref_backbone_is_float32(refdir);
    if (threshold < 0.0f)
        threshold = ref_is_f32 ? 1e-3f : ((H == W) ? 2e-1f : 5e-1f);
    if (mean_threshold < 0.0f)
        mean_threshold = ref_is_f32 ? 1e-5f : 1e-2f;

    snprintf(path, sizeof(path), "%s/sam3d_body_dinov3.safetensors", sft_dir);
    fprintf(stderr, "[verify_dinov3] loading encoder: %s\n", path);
    dinov3_model *m = dinov3_load_safetensors(path);
    if (!m) { fprintf(stderr, "load failed\n"); free(in); free(ref); return 5; }
    m->use_learned_final_norm = 1;

    dinov3_result r = dinov3_encode_from_normalized(m, in, W, H, n_threads);
    free(in);
    if (!r.features) { dinov3_free(m); free(ref); return 6; }

    int out_n = r.n_tokens, out_c = r.dim;
    int n_patches = Ph * Pw;
    int patch_start = out_n - n_patches;
    if (out_c != D || patch_start < 1) {
        fprintf(stderr, "[verify_dinov3] shape mismatch: ours=(%d,%d) ref=(1,%d,%d,%d) → patch_start=%d\n",
                out_n, out_c, D, Ph, Pw, patch_start);
        dinov3_result_free(&r); dinov3_free(m); free(ref);
        return 7;
    }

    /* ours[patch_start + py*Pw + px, d]  ≙  ref[0, d, py, px] */
    double sum = 0.0;
    float mx = 0.0f;
    int mx_d = 0, mx_py = 0, mx_px = 0;
    size_t n = (size_t)n_patches * D;
    for (int py = 0; py < Ph; py++) {
        for (int px = 0; px < Pw; px++) {
            const float *op = r.features + (size_t)(patch_start + py * Pw + px) * D;
            for (int d = 0; d < D; d++) {
                float rv = ref[((0 * D + d) * Ph + py) * Pw + px];
                float dv = fabsf(op[d] - rv);
                if (dv > mx) { mx = dv; mx_d = d; mx_py = py; mx_px = px; }
                sum += dv;
            }
        }
    }
    double mean_abs = sum / (double)n;
    fprintf(stderr, "[verify_dinov3] patches=(%d,%d) D=%d  "
                    "max_abs=%.6e (d=%d py=%d px=%d) "
                    "mean_abs=%.6e  (max_gate=%.1e mean_gate=%.1e)\n",
            Ph, Pw, D, mx, mx_d, mx_py, mx_px, mean_abs,
            threshold, mean_threshold);
    int rc_out = (mx < threshold && mean_abs < mean_threshold) ? 0 : 1;

    dinov3_result_free(&r);
    dinov3_free(m);
    free(ref);
    return rc_out;
}
