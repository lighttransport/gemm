/*
 * verify_mhr_head (CUDA) — diff hip_sam3d_body_debug_run_norm_and_heads
 * against /tmp/sam3d_body_ref/{decoder_out_norm_final, head_pose_proj_raw,
 * head_camera_proj_raw}.npy. Mirrors cpu/sam3d_body/verify_mhr_head.c.
 *
 * Inputs:
 *   decoder_layer5_out__tokens.npy   (1, 145, 1024) f32 — pre-norm tokens
 *   decoder_out_norm_final.npy       (1, 145, 1024) f32 — post-norm tokens
 *   head_pose_proj_raw.npy           (519,)         f32 — pre-init-add pose
 *   head_camera_proj_raw.npy         (3,)           f32 — pre-init-add cam
 */

#include "hip_sam3d_body_runner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/npy_io.h"

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
    fprintf(stderr, "[cuda verify_mhr_head] %-22s max_abs=%.6e (i=%zu)  "
                    "mean_abs=%.6e  (max=%.1e mean=%.1e) %s\n",
            label, mx, mx_i, mean, threshold, mean_gate,
            fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    /* CPU floor ~1e-5; CUDA matches f32 LN+GEMM precision, set 1e-4 budget. */
    float threshold = 1e-4f;
    int device = 0, verbose = 0;
    const char *precision = "bf16";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--device") && i+1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc) precision = argv[++i];
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir DIR --refdir DIR "
                        "[--threshold F] [--device N] "
                        "[--precision bf16|fp16] [-v]\n", argv[0]);
        return 2;
    }

    char path[1024];
    int nd = 0, dims[8] = {0};

    snprintf(path, sizeof(path), "%s/decoder_layer5_out__tokens.npy", refdir);
    float *tokens_in = (float *)npy_load(path, &nd, dims, NULL);
    if (!tokens_in || nd != 3 || dims[0] != 1 || dims[1] != 145 || dims[2] != 1024) {
        fprintf(stderr, "[cuda verify_mhr_head] missing/invalid %s\n", path);
        free(tokens_in); return 3;
    }
    snprintf(path, sizeof(path), "%s/decoder_out_norm_final.npy", refdir);
    float *ref_norm = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_norm || nd != 3 || dims[0] != 1 || dims[1] != 145 || dims[2] != 1024) {
        fprintf(stderr, "[cuda verify_mhr_head] missing/invalid %s\n", path);
        free(tokens_in); free(ref_norm); return 3;
    }
    snprintf(path, sizeof(path), "%s/head_pose_proj_raw.npy", refdir);
    float *ref_pose = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_pose || dims[nd-1] != 519) {
        fprintf(stderr, "[cuda verify_mhr_head] missing/invalid %s\n", path);
        free(tokens_in); free(ref_norm); free(ref_pose); return 3;
    }
    snprintf(path, sizeof(path), "%s/head_camera_proj_raw.npy", refdir);
    float *ref_cam = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref_cam || dims[nd-1] != 3) {
        fprintf(stderr, "[cuda verify_mhr_head] missing/invalid %s\n", path);
        free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam); return 3;
    }

    hip_sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .image_size      = 512,
        .device_ordinal  = device,
        .verbose         = verbose,
        .precision       = precision,
    };
    hip_sam3d_body_ctx *ctx = hip_sam3d_body_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "[cuda verify_mhr_head] create failed\n");
        free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam);
        return 5;
    }

    const int N_Q = 145, D = 1024;
    float *x_norm   = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *pose_raw = (float *)malloc((size_t)519 * sizeof(float));
    float *cam_raw  = (float *)malloc((size_t)3   * sizeof(float));
    if (!x_norm || !pose_raw || !cam_raw) {
        hip_sam3d_body_destroy(ctx);
        free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam);
        free(x_norm); free(pose_raw); free(cam_raw);
        return 6;
    }

    int rc = hip_sam3d_body_debug_run_norm_and_heads(ctx, tokens_in, N_Q,
                                                      x_norm, pose_raw, cam_raw);
    if (rc != 0) {
        fprintf(stderr, "[cuda verify_mhr_head] run rc=%d\n", rc);
        hip_sam3d_body_destroy(ctx);
        free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam);
        free(x_norm); free(pose_raw); free(cam_raw);
        return 7;
    }

    int rc_out = 0;
    rc_out |= diff_report("norm_final (tokens)",    x_norm,   ref_norm,
                          (size_t)N_Q * D, threshold);
    rc_out |= diff_report("head_pose.proj(tok0)",   pose_raw, ref_pose, 519, threshold);
    rc_out |= diff_report("head_camera.proj(tok0)", cam_raw,  ref_cam,  3,   threshold);

    hip_sam3d_body_destroy(ctx);
    free(tokens_in); free(ref_norm); free(ref_pose); free(ref_cam);
    free(x_norm); free(pose_raw); free(cam_raw);
    return rc_out;
}
