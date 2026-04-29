/*
 * verify_mhr_stages — diff each MHR skinning stage against the
 * reference dumps. Covers stages 6, 7/8, 10A/B/C, 11. Stages 1-5 live
 * in the decoder head (verify_mhr_head.c) and stage 12 (keypoint
 * regression) lives in the head path.
 *
 * Usage:
 *   verify_mhr_stages --mhr-assets <dir> --refdir /tmp/sam3d_body_ref \
 *                     [--threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR (produced by ref/sam3d-body/gen_image_ref.py):
 *   mhr_params__shape.npy             (1, 45)      — blend_shape input
 *   mhr_params__face.npy              (1, 72)      — face_expressions input
 *   mhr_params__mhr_model_params.npy  (1, 204)     — parameter_transform input
 *   mhr_joint_parameters.npy          (1, 889)     — stage 6 ref
 *   mhr_blend_shape_out.npy           (1, 18439,3) — stage 10A ref
 *   mhr_face_expressions_out.npy      (1, 18439,3) — stage 10B ref
 *   mhr_pose_correctives_out.npy      (1, 18439,3) — stage 10C ref
 *   mhr_output__skel_state.npy        (1, 127, 8)  — stage 8 ref (global)
 *   mhr_output__verts.npy             (1, 18439,3) — stage 11 ref
 *
 * The per-stage budget widens downstream as f32 round-off accumulates:
 *   stage 6:  ≲ 1e-4   (single 249-wide dot, fp64 accumulator in C)
 *   stage 8:  ≲ 1e-4   (8-dim skel state after 4-stage prefix product, fp64)
 *   stage 10: ≲ 5e-4   (45×18439×3 accumulation vs torch einsum)
 *   stage 10C:≲ 5e-3   (750→3000 sparse + 3000→55317 dense)
 *   stage 11: ≲ 5e-3   (cm-scale vertices, LBS scatter-add)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"
#define SAM3D_BODY_MHR_IMPLEMENTATION
#include "sam3d_body_mhr.h"
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
    fprintf(stderr, "[verify_mhr_stages] %-30s max_abs=%.6e (i=%zu)  "
                    "mean_abs=%.6e  (max=%.1e mean=%.1e) %s\n",
            label, mx, mx_i, mean, threshold, mean_gate,
            fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *mhr_dir = NULL, *refdir = NULL;
    float threshold = 5e-3f;
    int n_threads = 1, verbose = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--mhr-assets") && i+1 < argc) mhr_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")     && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")  && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!mhr_dir || !refdir) {
        fprintf(stderr, "Usage: %s --mhr-assets <dir> --refdir <dir> "
                        "[--threshold F] [-t N] [-v]\n", argv[0]);
        return 2;
    }
    (void)verbose;

    char path[1024];
    snprintf(path, sizeof(path), "%s/sam3d_body_mhr_jit.safetensors", mhr_dir);
    char jsonp[1024];
    snprintf(jsonp, sizeof(jsonp), "%s/sam3d_body_mhr_jit.json", mhr_dir);
    sam3d_body_mhr_assets *a = sam3d_body_mhr_load(path, jsonp);
    if (!a) {
        fprintf(stderr, "[verify_mhr_stages] failed to load %s\n", path);
        return 3;
    }

    int nd = 0, dims[8] = {0}, f32 = 0;

    /* ---- inputs ---- */
    snprintf(path, sizeof(path), "%s/mhr_params__shape.npy", refdir);
    float *shape = (float *)npy_load(path, &nd, dims, &f32);
    if (!shape || dims[nd-1] != S3DM_N_SHAPE) {
        fprintf(stderr, "[verify_mhr_stages] missing/bad %s\n", path);
        sam3d_body_mhr_free(a); free(shape); return 4;
    }
    snprintf(path, sizeof(path), "%s/mhr_params__face.npy", refdir);
    float *face = (float *)npy_load(path, &nd, dims, &f32);
    if (!face || dims[nd-1] != S3DM_N_FACE) {
        fprintf(stderr, "[verify_mhr_stages] missing/bad %s\n", path);
        sam3d_body_mhr_free(a); free(shape); free(face); return 4;
    }
    snprintf(path, sizeof(path), "%s/mhr_params__mhr_model_params.npy", refdir);
    float *modelp = (float *)npy_load(path, &nd, dims, &f32);
    if (!modelp || dims[nd-1] != S3DM_N_MODEL_PARAMS) {
        fprintf(stderr, "[verify_mhr_stages] missing/bad %s\n", path);
        sam3d_body_mhr_free(a); free(shape); free(face); free(modelp); return 4;
    }

    /* ---- refs ---- */
    #define LOAD_REF(name, expect_n) \
        snprintf(path, sizeof(path), "%s/%s.npy", refdir, #name);  \
        float *ref_##name = (float *)npy_load(path, &nd, dims, &f32); \
        if (!ref_##name) { fprintf(stderr, "[verify_mhr_stages] missing %s\n", path); \
                           goto cleanup_fail; } \
        { size_t _n = 1; for (int _i=0; _i<nd; _i++) _n *= (size_t)dims[_i]; \
          if (_n != (size_t)(expect_n)) { \
              fprintf(stderr, "[verify_mhr_stages] %s size mismatch %zu vs %d\n", \
                      path, _n, (int)(expect_n)); goto cleanup_fail; } }

    LOAD_REF(mhr_joint_parameters,     S3DM_N_PTRANS_OUT);
    LOAD_REF(mhr_blend_shape_out,      S3DM_N_VERTS * 3);
    LOAD_REF(mhr_face_expressions_out, S3DM_N_VERTS * 3);
    LOAD_REF(mhr_pose_correctives_out, S3DM_N_VERTS * 3);
    LOAD_REF(mhr_output__skel_state,   S3DM_N_JOINTS * 8);
    LOAD_REF(mhr_output__verts,        S3DM_N_VERTS * 3);

    const int V = S3DM_N_VERTS;
    const int J = S3DM_N_JOINTS;
    int rc_out = 0;

    /* Stage 6: parameter_transform. */
    float *our_jp = (float *)malloc((size_t)S3DM_N_PTRANS_OUT * sizeof(float));
    int r = sam3d_body_mhr_parameter_transform(a, modelp, 1, n_threads, our_jp);
    if (r) { fprintf(stderr, "stage6 rc=%d\n", r); rc_out |= 1; }
    else    rc_out |= diff_report("stage6  parameter_transform",
                                  our_jp, ref_mhr_joint_parameters,
                                  S3DM_N_PTRANS_OUT, 1e-3f);

    /* Stage 7/8: joint_params -> local_skel -> global_skel. */
    float *our_local  = (float *)malloc((size_t)J * 8 * sizeof(float));
    float *our_global = (float *)malloc((size_t)J * 8 * sizeof(float));
    r = sam3d_body_mhr_joint_params_to_local_skel(a, our_jp, 1, our_local);
    if (r) { fprintf(stderr, "stage7 rc=%d\n", r); rc_out |= 1; }
    r = sam3d_body_mhr_local_to_global_skel(a, our_local, 1, our_global);
    if (r) { fprintf(stderr, "stage8 rc=%d\n", r); rc_out |= 1; }
    else    rc_out |= diff_report("stage7+8 global_skel_state",
                                  our_global, ref_mhr_output__skel_state,
                                  (size_t)J * 8, 1e-3f);

    /* Stage 10A: blend_shape. */
    float *our_bs = (float *)malloc((size_t)V * 3 * sizeof(float));
    r = sam3d_body_mhr_blend_shape(a, shape, 1, n_threads, our_bs);
    if (r) { fprintf(stderr, "stage10A rc=%d\n", r); rc_out |= 1; }
    else    rc_out |= diff_report("stage10A blend_shape",
                                  our_bs, ref_mhr_blend_shape_out,
                                  (size_t)V * 3, 5e-4f);

    /* Stage 10B: face_expressions. */
    float *our_fe = (float *)malloc((size_t)V * 3 * sizeof(float));
    r = sam3d_body_mhr_face_expressions(a, face, 1, n_threads, our_fe);
    if (r) { fprintf(stderr, "stage10B rc=%d\n", r); rc_out |= 1; }
    else    rc_out |= diff_report("stage10B face_expressions",
                                  our_fe, ref_mhr_face_expressions_out,
                                  (size_t)V * 3, 5e-4f);

    /* Stage 10C: pose_correctives. This is the heavy GEMV (3000→55317). */
    float *our_pc = (float *)malloc((size_t)V * 3 * sizeof(float));
    r = sam3d_body_mhr_pose_correctives(a, our_jp, 1, n_threads, our_pc);
    if (r) { fprintf(stderr, "stage10C rc=%d\n", r); rc_out |= 1; }
    else    rc_out |= diff_report("stage10C pose_correctives",
                                  our_pc, ref_mhr_pose_correctives_out,
                                  (size_t)V * 3, threshold);

    /* Stage 11: LBS with the reference linear_model_unposed = bs + fe + pc
     * (feeding ref directly so LBS drift is isolated from stage-10 drift). */
    float *our_rest = (float *)malloc((size_t)V * 3 * sizeof(float));
    for (int i = 0; i < V * 3; i++)
        our_rest[i] = ref_mhr_blend_shape_out[i]
                    + ref_mhr_face_expressions_out[i]
                    + ref_mhr_pose_correctives_out[i];
    float *our_verts = (float *)malloc((size_t)V * 3 * sizeof(float));
    r = sam3d_body_mhr_skin_points(a, ref_mhr_output__skel_state, our_rest,
                                   1, our_verts);
    if (r) { fprintf(stderr, "stage11 rc=%d\n", r); rc_out |= 1; }
    else    rc_out |= diff_report("stage11 LBS (ref inputs)",
                                  our_verts, ref_mhr_output__verts,
                                  (size_t)V * 3, threshold);

    /* End-to-end: run sam3d_body_mhr_forward and compare to ref verts. */
    float *e2e_verts = (float *)malloc((size_t)V * 3 * sizeof(float));
    float *e2e_skel  = (float *)malloc((size_t)J * 8 * sizeof(float));
    r = sam3d_body_mhr_forward(a, modelp, shape, face, 1, /*apply_correctives=*/1,
                               n_threads, NULL, e2e_verts, e2e_skel);
    if (r) { fprintf(stderr, "e2e rc=%d\n", r); rc_out |= 1; }
    else    rc_out |= diff_report("e2e     mhr_forward",
                                  e2e_verts, ref_mhr_output__verts,
                                  (size_t)V * 3, threshold);

    free(our_jp); free(our_local); free(our_global);
    free(our_bs); free(our_fe); free(our_pc);
    free(our_rest); free(our_verts); free(e2e_verts); free(e2e_skel);
    free(ref_mhr_joint_parameters);
    free(ref_mhr_blend_shape_out);
    free(ref_mhr_face_expressions_out);
    free(ref_mhr_pose_correctives_out);
    free(ref_mhr_output__skel_state);
    free(ref_mhr_output__verts);
    sam3d_body_mhr_free(a);
    free(shape); free(face); free(modelp);
    return rc_out;

cleanup_fail:
    sam3d_body_mhr_free(a);
    free(shape); free(face); free(modelp);
    return 5;
}
