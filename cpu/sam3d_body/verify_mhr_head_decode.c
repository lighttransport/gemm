/*
 * verify_mhr_head_decode — diff stages 1-5 (head_pose raw → mhr_model_params)
 * and stage 12 (keypoints from mesh) against the reference dumps.
 *
 * Inputs:
 *   mhr_params__pred_pose_raw.npy   (1, 266)        — global_rot_6d + body_cont
 *   head_pose_proj_raw.npy          (1, 519)        — full raw head output
 *   mhr_params__pred_vertices.npy   (1, V, 3)       — final flipped verts (m)
 *   mhr_params__pred_joint_coords.npy (1, J, 3)     — final flipped jcoords (m)
 *
 * Refs:
 *   mhr_params__global_rot.npy      (1, 3)
 *   mhr_params__body_pose.npy       (1, 133)
 *   mhr_params__mhr_model_params.npy (1, 204)
 *   mhr_params__shape.npy           (1, 45)
 *   mhr_params__face.npy            (1, 72)
 *   mhr_params__pred_keypoints_3d.npy (1, 70, 3)
 *
 * Usage:
 *   verify_mhr_head_decode --safetensors-dir <dir> --refdir <dir>
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "npy_io.h"

static int diff(const char *label, const float *a, const float *b,
                size_t n, float thresh)
{
    double sum = 0.0; float mx = 0.0f; size_t mxi = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) { mx = d; mxi = i; }
        sum += d;
    }
    double mean = sum / (double)n;
    int fail = mx >= thresh;
    fprintf(stderr, "[verify_mhr_head_decode] %-30s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            label, mx, mxi, mean, thresh, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir <dir> --refdir <dir>\n", argv[0]);
        return 2;
    }

    char p[1024];
    snprintf(p, sizeof(p), "%s/sam3d_body_decoder.safetensors", sft_dir);
    char p2[1024];
    snprintf(p2, sizeof(p2), "%s/sam3d_body_mhr_head.safetensors", sft_dir);
    sam3d_body_decoder_model *m = sam3d_body_decoder_load(p, p2);
    if (!m) { fprintf(stderr, "load failed\n"); return 3; }

    int nd, dims[8];

    /* Reconstruct the full 519-vec head output from the per-segment ref
     * dumps (which capture pred = proj(x) + init_estimate post-add).
     *
     *   [0..266)   = mhr_params__pred_pose_raw  (= global_rot_6d + body_cont)
     *   [266..311) = mhr_params__shape          (45)
     *   [311..339) = mhr_params__scale          (28)
     *   [339..447) = mhr_params__hand           (108)
     *   [447..519) = mhr_params__face           (72) — zeroed in body branch
     */
    float pose_full[519];
    snprintf(p, sizeof(p), "%s/mhr_params__pred_pose_raw.npy", refdir);
    float *t = (float *)npy_load(p, &nd, dims, NULL);
    if (!t) { fprintf(stderr, "missing %s\n", p); goto fail; }
    memcpy(pose_full +   0, t, 266 * sizeof(float)); free(t);
    snprintf(p, sizeof(p), "%s/mhr_params__shape.npy", refdir);
    t = (float *)npy_load(p, &nd, dims, NULL);
    memcpy(pose_full + 266, t,  45 * sizeof(float)); free(t);
    snprintf(p, sizeof(p), "%s/mhr_params__scale.npy", refdir);
    t = (float *)npy_load(p, &nd, dims, NULL);
    memcpy(pose_full + 311, t,  28 * sizeof(float)); free(t);
    snprintf(p, sizeof(p), "%s/mhr_params__hand.npy", refdir);
    t = (float *)npy_load(p, &nd, dims, NULL);
    memcpy(pose_full + 339, t, 108 * sizeof(float)); free(t);
    snprintf(p, sizeof(p), "%s/mhr_params__face.npy", refdir);
    t = (float *)npy_load(p, &nd, dims, NULL);
    memcpy(pose_full + 447, t,  72 * sizeof(float)); free(t);

    float mp[204], shape[45], face[72];
    /* Reference dump `mhr_params__*` is captured via forward_hook on every
     * MHRHead module — the LAST hand-branch call wins (head_pose_hand). So
     * we test the enable_hand_model=True path here. */
    int rc = sam3d_body_decode_pose_raw(m, pose_full, /*enable_hand_model*/1,
                                        mp, shape, face);
    if (rc) { fprintf(stderr, "decode_pose_raw rc=%d\n", rc); goto fail; }

    /* Refs. */
    int rc_out = 0;

    snprintf(p, sizeof(p), "%s/mhr_params__shape.npy", refdir);
    float *ref_shape = (float *)npy_load(p, &nd, dims, NULL);
    rc_out |= diff("stage5  pred_shape", shape, ref_shape, 45, 1e-5f);
    free(ref_shape);

    snprintf(p, sizeof(p), "%s/mhr_params__face.npy", refdir);
    float *ref_face = (float *)npy_load(p, &nd, dims, NULL);
    rc_out |= diff("stage5  pred_face (zeroed)", face, ref_face, 72, 1e-5f);
    free(ref_face);

    /* Note: mp[3..6] now holds the wrist-transformed XYZ Euler (per
     * mhr_forward's enable_hand_model branch), not the raw ZYX. The
     * mhr_params__global_rot.npy dump is the pre-transform ZYX. The
     * implicit verification of the wrist transform is via the full
     * mhr_model_params diff below. */
    snprintf(p, sizeof(p), "%s/mhr_params__mhr_model_params.npy", refdir);
    float *ref_mp = (float *)npy_load(p, &nd, dims, NULL);
    rc_out |= diff("stages1-5 mhr_model_params", mp, ref_mp, 204, 5e-4f);
    free(ref_mp);

    /* Stage 12 keypoints. Inputs: pred_vertices (already FLIPPED in ref),
     * pred_joint_coords (already FLIPPED). We need pre-flip values; just
     * un-flip here. */
    snprintf(p, sizeof(p), "%s/mhr_params__pred_vertices.npy", refdir);
    float *verts_flipped = (float *)npy_load(p, &nd, dims, NULL);
    int V = dims[1];
    snprintf(p, sizeof(p), "%s/mhr_params__pred_joint_coords.npy", refdir);
    float *jc_flipped = (float *)npy_load(p, &nd, dims, NULL);
    int J = dims[1];

    float *verts_unflipped = (float *)malloc(V * 3 * sizeof(float));
    float *jc_unflipped    = (float *)malloc(J * 3 * sizeof(float));
    for (int i = 0; i < V; i++) {
        verts_unflipped[i*3 + 0] = verts_flipped[i*3 + 0];
        verts_unflipped[i*3 + 1] = -verts_flipped[i*3 + 1];
        verts_unflipped[i*3 + 2] = -verts_flipped[i*3 + 2];
    }
    for (int i = 0; i < J; i++) {
        jc_unflipped[i*3 + 0] = jc_flipped[i*3 + 0];
        jc_unflipped[i*3 + 1] = -jc_flipped[i*3 + 1];
        jc_unflipped[i*3 + 2] = -jc_flipped[i*3 + 2];
    }

    float kpts[70 * 3];
    rc = sam3d_body_keypoints_from_mesh(m, verts_unflipped, jc_unflipped,
                                        /*enable_hand_model*/1, 1, kpts);
    if (rc) { fprintf(stderr, "keypoints rc=%d\n", rc); goto fail; }

    snprintf(p, sizeof(p), "%s/mhr_params__pred_keypoints_3d.npy", refdir);
    float *ref_kpts = (float *)npy_load(p, &nd, dims, NULL);
    rc_out |= diff("stage12 keypoints (70,3)", kpts, ref_kpts, 70 * 3, 5e-5f);
    free(ref_kpts);
    free(verts_flipped); free(jc_flipped);
    free(verts_unflipped); free(jc_unflipped);

    sam3d_body_decoder_free(m);
    return rc_out;

fail:
    sam3d_body_decoder_free(m);
    return 4;
}
