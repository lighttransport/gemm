/*
 * verify_mhr (CUDA) — sanity check for the speculative MHR-on-GPU
 * helpers (exploratory; Step 7 is officially CLOSED via CPU OpenMP per
 * PORT.md).
 *
 * Validates that each GPU helper matches the CPU reference bit-exact (or
 * within 1 ULP) on deterministic-random inputs. No /tmp/sam3d_body_ref/
 * NPY needed — the CPU sam3d_body_mhr_* impls are the reference.
 *
 *   - hip_sam3d_body_debug_run_blend_shape       (45 → V*3)
 *   - hip_sam3d_body_debug_run_face_expressions  (72 → V*3)
 *   - hip_sam3d_body_debug_run_pose_correctives  (127×7 → V*3)
 */

#include "hip_sam3d_body_runner.h"
#include "../../common/sam3d_body_mhr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    fprintf(stderr, "[cuda verify_mhr] %-22s max_abs=%.6e (i=%zu)  "
                    "mean_abs=%.6e  (max=%.1e mean=%.1e) %s\n",
            label, mx, mx_i, mean, threshold, mean_gate,
            fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

/* xorshift32 for reproducible coeffs across runs. */
static uint32_t xs_state = 0xC0FFEE42u;
static float    rand_f32_unit(void)
{
    xs_state ^= xs_state << 13;
    xs_state ^= xs_state >> 17;
    xs_state ^= xs_state << 5;
    return ((float)(xs_state & 0xFFFFFFu) / (float)0x1000000u) * 2.0f - 1.0f;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL;
    const char *mhr_dir = NULL;
    float threshold = 5e-5f;
    int device = 0, verbose = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--mhr-assets-dir")  && i+1 < argc) mhr_dir = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--device")          && i+1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !mhr_dir) {
        fprintf(stderr, "Usage: %s --safetensors-dir DIR --mhr-assets-dir DIR "
                        "[--threshold F] [--device N] [-v]\n", argv[0]);
        return 2;
    }

    hip_sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .mhr_assets_dir  = mhr_dir,
        .image_size      = 512,
        .device_ordinal  = device,
        .verbose         = verbose,
        .precision       = "bf16",
    };
    hip_sam3d_body_ctx *ctx = hip_sam3d_body_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "[cuda verify_mhr] create failed\n");
        return 5;
    }

    /* CPU reference uses the same MHR assets via a side-channel load —
     * load them again on the host so we can call sam3d_body_mhr_blend_shape
     * directly (the runner's cpu_mhr is private). */
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/sam3d_body_mhr_jit.safetensors", mhr_dir);
    snprintf(p2, sizeof(p2), "%s/sam3d_body_mhr_jit.json",        mhr_dir);
    sam3d_body_mhr_assets *cpu_mhr = sam3d_body_mhr_load(p1, p2);
    if (!cpu_mhr) {
        fprintf(stderr, "[cuda verify_mhr] CPU MHR load failed\n");
        hip_sam3d_body_destroy(ctx);
        return 6;
    }

    const int V_d = 18439 * 3;
    float *cpu_out = (float *)malloc((size_t)V_d * sizeof(float));
    float *gpu_out = (float *)malloc((size_t)V_d * sizeof(float));
    if (!cpu_out || !gpu_out) {
        free(cpu_out); free(gpu_out);
        sam3d_body_mhr_free(cpu_mhr);
        hip_sam3d_body_destroy(ctx);
        return 7;
    }

    int rc_out = 0;

    /* ---- blend_shape (45 basis) ---- */
    {
        float coeffs[45];
        for (int i = 0; i < 45; i++) coeffs[i] = 0.5f * rand_f32_unit();

        int r = sam3d_body_mhr_blend_shape(cpu_mhr, coeffs, 1, 1, cpu_out);
        if (r != 0) {
            fprintf(stderr, "[cuda verify_mhr] CPU blend_shape failed rc=%d\n", r);
            rc_out = 8;
        } else {
            r = hip_sam3d_body_debug_run_blend_shape(ctx, coeffs, gpu_out);
            if (r != 0) {
                fprintf(stderr, "[cuda verify_mhr] GPU blend_shape failed rc=%d\n", r);
                rc_out = 8;
            } else {
                rc_out |= diff_report("blend_shape", gpu_out, cpu_out,
                                      (size_t)V_d, threshold);
            }
        }
    }

    /* ---- face_expressions (72 basis) ---- */
    {
        float coeffs[72];
        for (int i = 0; i < 72; i++) coeffs[i] = 0.3f * rand_f32_unit();

        int r = sam3d_body_mhr_face_expressions(cpu_mhr, coeffs, 1, 1, cpu_out);
        if (r != 0) {
            fprintf(stderr, "[cuda verify_mhr] CPU face_expressions failed rc=%d\n", r);
            rc_out = 9;
        } else {
            r = hip_sam3d_body_debug_run_face_expressions(ctx, coeffs, gpu_out);
            if (r != 0) {
                fprintf(stderr, "[cuda verify_mhr] GPU face_expressions failed rc=%d\n", r);
                rc_out = 9;
            } else {
                rc_out |= diff_report("face_expressions", gpu_out, cpu_out,
                                      (size_t)V_d, threshold);
            }
        }
    }

    /* ---- pose_correctives (joint_params 127×7 → V*3) ---- */
    {
        const int J = 127;
        float jp[127 * 7];
        for (int j = 0; j < J; j++) {
            jp[j*7 + 0] = 0.05f * rand_f32_unit(); /* tx */
            jp[j*7 + 1] = 0.05f * rand_f32_unit(); /* ty */
            jp[j*7 + 2] = 0.05f * rand_f32_unit(); /* tz */
            jp[j*7 + 3] = 0.30f * rand_f32_unit(); /* rx (radians) */
            jp[j*7 + 4] = 0.30f * rand_f32_unit(); /* ry */
            jp[j*7 + 5] = 0.30f * rand_f32_unit(); /* rz */
            jp[j*7 + 6] = 0.05f * rand_f32_unit(); /* log-scale */
        }
        int r = sam3d_body_mhr_pose_correctives(cpu_mhr, jp, 1, 1, cpu_out);
        if (r != 0) {
            fprintf(stderr, "[cuda verify_mhr] CPU pose_correctives failed rc=%d\n", r);
            rc_out = 10;
        } else {
            r = hip_sam3d_body_debug_run_pose_correctives(ctx, jp, gpu_out);
            if (r != 0) {
                fprintf(stderr, "[cuda verify_mhr] GPU pose_correctives failed rc=%d\n", r);
                rc_out = 10;
            } else {
                /* Floating-point reduction order is identical (single thread per
                 * output); expect bit-exact match like blend_shape. Use a small
                 * f32-noise budget anyway. */
                rc_out |= diff_report("pose_correctives", gpu_out, cpu_out,
                                      (size_t)V_d, threshold);
            }
        }
    }

    /* ---- LBS skin_points (global_skel + rest_verts → V*3) ---- */
    {
        const int J = 127;
        const int V = 18439;
        float *global_skel = (float *)malloc((size_t)J * 8 * sizeof(float));
        float *rverts      = (float *)malloc((size_t)V * 3 * sizeof(float));
        if (!global_skel || !rverts) {
            free(global_skel); free(rverts);
            free(cpu_out); free(gpu_out);
            sam3d_body_mhr_free(cpu_mhr);
            hip_sam3d_body_destroy(ctx);
            return 11;
        }
        /* Deterministic-random global_skel near identity. Both CPU and GPU
         * pipe this through skel_multiply(global, inverse_bind_pose) before
         * scatter, so results are directly comparable. */
        for (int j = 0; j < J; j++) {
            float *s = global_skel + (size_t)j * 8;
            s[0] = 0.02f * rand_f32_unit();
            s[1] = 0.02f * rand_f32_unit();
            s[2] = 0.02f * rand_f32_unit();
            s[3] = 0.05f * rand_f32_unit();
            s[4] = 0.05f * rand_f32_unit();
            s[5] = 0.05f * rand_f32_unit();
            s[6] = 1.0f;
            s[7] = 1.0f + 0.01f * rand_f32_unit();
        }
        for (int i = 0; i < V * 3; i++) rverts[i] = 10.0f * rand_f32_unit();

        int r = sam3d_body_mhr_skin_points(cpu_mhr, global_skel, rverts, 1, cpu_out);
        if (r != 0) {
            fprintf(stderr, "[cuda verify_mhr] CPU skin_points failed rc=%d\n", r);
            rc_out = 11;
        } else {
            r = hip_sam3d_body_debug_run_lbs_skin(ctx, global_skel, rverts, gpu_out);
            if (r != 0) {
                fprintf(stderr, "[cuda verify_mhr] GPU lbs_skin failed rc=%d\n", r);
                rc_out = 11;
            } else {
                /* atomicAdd reduction order differs from CPU's serial scatter
                 * order — per-vert sum has up to ~16 contributions, expect
                 * ~1e-5 absolute drift on cm-scale verts. */
                rc_out |= diff_report("lbs_skin", gpu_out, cpu_out,
                                      (size_t)V_d, 5e-4f);
            }
        }
        free(global_skel); free(rverts);
    }

    free(cpu_out); free(gpu_out);
    sam3d_body_mhr_free(cpu_mhr);
    hip_sam3d_body_destroy(ctx);
    return rc_out;
}
