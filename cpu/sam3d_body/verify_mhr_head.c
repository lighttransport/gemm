/*
 * verify_mhr_head — diff (a) decoder.norm_final applied to the token
 * stack and (b) head_pose.proj + head_camera.proj applied to tokens[0]
 * against the reference dumps.
 *
 * Inputs pulled from the ref dump:
 *   decoder_layer5_out__tokens.npy   (1,145,1024) f32 — input to norm_final
 *   decoder_out_norm_final.npy       (1,145,1024) f32 — reference post-LN
 *   head_pose_proj_raw.npy           (1,519)      f32 — head_pose.proj(tok0)
 *   head_camera_proj_raw.npy         (1,3)        f32 — head_camera.proj(tok0)
 *
 * Note: the "_raw" dumps are the *pre-init-add* projections — i.e. they
 * record head_{pose,camera}.proj(tok0) before init_pose / init_camera are
 * summed in. We verify that form here; the init add lands with the full
 * decoder_forward wiring (step 4g).
 *
 * Usage:
 *   verify_mhr_head --safetensors-dir <dir> --refdir <dir>
 *                   [--threshold F] [-t N] [-v]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "npy_io.h"

static int diff_report(const char *label, const float *a, const float *b,
                       size_t n, float threshold)
{
    double sum = 0.0; float mx = 0.0f; size_t mx_i = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) { mx = d; mx_i = i; }
        sum += d;
    }
    double mean = sum / (double)n;
    float mean_gate = threshold * 0.15f;
    int fail = (mx >= threshold || mean >= mean_gate);
    fprintf(stderr, "[verify_mhr_head] %-22s max_abs=%.6e (i=%zu)  "
                    "mean_abs=%.6e  (max_gate=%.1e mean_gate=%.1e) %s\n",
            label, mx, mx_i, mean, threshold, mean_gate,
            fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
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
    /* LN + 2 linears accumulate f32 round-off; empirical floor ≲ 1e-5. */
    float threshold = 1e-4f;
    int n_threads = 1, verbose = 0;
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
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--threshold F] [--backbone dinov3|vith] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }
    (void)verbose;

    char path[1024];
    int nd = 0, dims[8] = {0};

    snprintf(path, sizeof(path), "%s/decoder_layer5_out__tokens.npy", refdir);
    float *tokens_in = (float *)npy_load(path, &nd, dims, NULL);
    if (!tokens_in || nd != 3 || dims[0] != 1 || dims[1] != 145 || dims[2] != 1024) {
        fprintf(stderr, "[verify_mhr_head] missing/invalid %s\n", path);
        free(tokens_in); return 3;
    }
    snprintf(path, sizeof(path), "%s/decoder_out_norm_final.npy", refdir);
    float *ref_norm = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_norm || nd != 3 || dims[0] != 1 || dims[1] != 145 || dims[2] != 1024) {
        fprintf(stderr, "[verify_mhr_head] missing/invalid %s\n", path);
        free(tokens_in); free(ref_norm); return 3;
    }
    snprintf(path, sizeof(path), "%s/head_pose_proj_raw.npy", refdir);
    float *ref_pose = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_pose || dims[nd-1] != 519) {
        fprintf(stderr, "[verify_mhr_head] missing/invalid %s\n", path);
        free(tokens_in); free(ref_norm); free(ref_pose); return 3;
    }
    snprintf(path, sizeof(path), "%s/head_camera_proj_raw.npy", refdir);
    float *ref_cam = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_cam || dims[nd-1] != 3) {
        fprintf(stderr, "[verify_mhr_head] missing/invalid %s\n", path);
        free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam); return 3;
    }

    char mhr_path[1024];
    resolve_variant_path(sft_dir, "decoder", backbone, path, sizeof(path));
    resolve_variant_path(sft_dir, "mhr_head", backbone,
                         mhr_path, sizeof(mhr_path));
    sam3d_body_decoder_model *m = sam3d_body_decoder_load(path, mhr_path);
    if (!m) {
        free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam);
        return 5;
    }

    const int N_Q = 145, D = 1024;
    float *x_norm = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *pose_raw = (float *)malloc((size_t)519 * sizeof(float));
    float *cam_raw  = (float *)malloc((size_t)3   * sizeof(float));
    if (!x_norm || !pose_raw || !cam_raw) {
        sam3d_body_decoder_free(m);
        free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam);
        free(x_norm); free(pose_raw); free(cam_raw);
        return 6;
    }

    int rc = sam3d_body_norm_final(m, tokens_in, N_Q, n_threads, x_norm);
    if (rc != SAM3D_BODY_DECODER_E_OK) {
        fprintf(stderr, "[verify_mhr_head] norm_final rc=%d\n", rc);
        sam3d_body_decoder_free(m);
        free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam);
        free(x_norm); free(pose_raw); free(cam_raw);
        return 7;
    }

    rc = sam3d_body_apply_heads_raw(m, x_norm /* tokens_norm[0] */,
                                    n_threads, pose_raw, cam_raw);
    if (rc != SAM3D_BODY_DECODER_E_OK) {
        fprintf(stderr, "[verify_mhr_head] apply_heads_raw rc=%d\n", rc);
        sam3d_body_decoder_free(m);
        free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam);
        free(x_norm); free(pose_raw); free(cam_raw);
        return 7;
    }

    int rc_out = 0;
    rc_out |= diff_report("norm_final (tokens)", x_norm, ref_norm,
                          (size_t)N_Q * D, threshold);
    rc_out |= diff_report("head_pose.proj(tok0)", pose_raw, ref_pose, 519, threshold);
    rc_out |= diff_report("head_camera.proj(tok0)", cam_raw, ref_cam, 3, threshold);

    sam3d_body_decoder_free(m);
    free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam);
    free(x_norm); free(pose_raw); free(cam_raw);
    return rc_out;
}
