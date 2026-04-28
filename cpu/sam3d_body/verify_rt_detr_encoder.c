/* verify_rt_detr_encoder.c — verify HybridEncoder forward (FPN+PAN+AIFI)
 *
 * Loads /tmp/rt_detr_ref/bb_s{3,4,5}.npy as the encoder inputs, runs the C
 * HybridEncoder, and compares the 3 fused 256-channel feature maps against
 * enc_out_s{3,4,5}.npy.
 *
 * Tolerance: max_abs ≤ 1e-2, mean_abs ≤ 1e-3. Encoder fp32 inference is
 * deterministic but accumulates conv + matmul + softmax error across the
 * AIFI layer + 2 CSPRepLayer FPN blocks + 2 CSPRepLayer PAN blocks.
 */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define RT_DETR_IMPLEMENTATION
#include "../../common/rt_detr.h"
#include "../../common/npy_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *MODEL_PATH = "/mnt/disk01/models/rt_detr_s/model.safetensors";

static int diff_check(const char *name, const float *a, const float *b,
                      int n, float max_budget, double mean_budget)
{
    double mean = 0.0;
    float mx = npy_max_abs_f32(a, b, n, &mean);
    int ok = (mx <= max_budget) && (mean <= mean_budget);
    fprintf(stderr, "  %s: max_abs=%.6f mean_abs=%.6f  %s (budget %.4f / %.4f)\n",
            name, mx, mean, ok ? "PASS" : "FAIL", max_budget, mean_budget);
    return ok;
}

int main(int argc, char **argv) {
    const char *model = (argc > 1) ? argv[1] : MODEL_PATH;
    const char *ref_dir = (argc > 2) ? argv[2] : "/tmp/rt_detr_ref";

    char path[256];
    rt_detr_t *m = rt_detr_load(model);
    if (!m) { fprintf(stderr, "FAIL: load model\n"); return 1; }
    fprintf(stderr, "loaded %s\n", model);

    int ndim, dims[8];
    snprintf(path, sizeof(path), "%s/bb_s3.npy", ref_dir);
    float *bb_s3 = (float *)npy_load(path, &ndim, dims, NULL);
    snprintf(path, sizeof(path), "%s/bb_s4.npy", ref_dir);
    float *bb_s4 = (float *)npy_load(path, &ndim, dims, NULL);
    snprintf(path, sizeof(path), "%s/bb_s5.npy", ref_dir);
    float *bb_s5 = (float *)npy_load(path, &ndim, dims, NULL);
    if (!bb_s3 || !bb_s4 || !bb_s5) {
        fprintf(stderr, "FAIL: load backbone refs\n");
        rt_detr_free(m); return 1;
    }

    float *out_s3 = (float *)malloc(256 * 80 * 80 * sizeof(float));
    float *out_s4 = (float *)malloc(256 * 40 * 40 * sizeof(float));
    float *out_s5 = (float *)malloc(256 * 20 * 20 * sizeof(float));
    if (!out_s3 || !out_s4 || !out_s5) { fprintf(stderr, "FAIL: alloc\n"); return 1; }

    fprintf(stderr, "running HybridEncoder...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (rt_detr_forward_encoder(m, bb_s3, bb_s4, bb_s5,
                                out_s3, out_s4, out_s5) != 0) {
        fprintf(stderr, "FAIL: forward_encoder\n");
        free(bb_s3); free(bb_s4); free(bb_s5);
        free(out_s3); free(out_s4); free(out_s5);
        rt_detr_free(m); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "  forward_encoder: %.3f s\n", dt);

    snprintf(path, sizeof(path), "%s/enc_out_s3.npy", ref_dir);
    float *r3 = (float *)npy_load(path, &ndim, dims, NULL);
    snprintf(path, sizeof(path), "%s/enc_out_s4.npy", ref_dir);
    float *r4 = (float *)npy_load(path, &ndim, dims, NULL);
    snprintf(path, sizeof(path), "%s/enc_out_s5.npy", ref_dir);
    float *r5 = (float *)npy_load(path, &ndim, dims, NULL);
    if (!r3 || !r4 || !r5) { fprintf(stderr, "FAIL: load encoder refs\n"); return 1; }

    fprintf(stderr, "diffs:\n");
    int ok = 1;
    ok &= diff_check("enc_s3", out_s3, r3, 256 * 80 * 80, 1e-2f, 1e-3);
    ok &= diff_check("enc_s4", out_s4, r4, 256 * 40 * 40, 1e-2f, 1e-3);
    ok &= diff_check("enc_s5", out_s5, r5, 256 * 20 * 20, 1e-2f, 1e-3);

    free(bb_s3); free(bb_s4); free(bb_s5);
    free(out_s3); free(out_s4); free(out_s5);
    free(r3); free(r4); free(r5);
    rt_detr_free(m);
    fprintf(stderr, "%s: rt_detr_encoder\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
