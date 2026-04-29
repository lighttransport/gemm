/* verify_rt_detr_backbone.c — verify R18-VD backbone forward
 *
 * Loads /tmp/rt_detr_ref/input.npy as the model input (canonical, matches
 * HF's PIL preprocessing), runs the C backbone, and compares the 3
 * feature maps against bb_s{3,4,5}.npy.
 *
 * Tolerance budget: max_abs ≤ 5e-3. R18-VD inference is fp32 deterministic
 * so this should be near-zero — main source of drift is conv accumulation
 * order. Mean budget: ≤ 1e-4.
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
static const char *REF_INPUT  = "/tmp/rt_detr_ref/input.npy";
static const char *REF_S3     = "/tmp/rt_detr_ref/bb_s3.npy";
static const char *REF_S4     = "/tmp/rt_detr_ref/bb_s4.npy";
static const char *REF_S5     = "/tmp/rt_detr_ref/bb_s5.npy";

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

    char ref_input[256], ref_s3[256], ref_s4[256], ref_s5[256];
    snprintf(ref_input, sizeof(ref_input), "%s/input.npy", ref_dir);
    snprintf(ref_s3, sizeof(ref_s3), "%s/bb_s3.npy", ref_dir);
    snprintf(ref_s4, sizeof(ref_s4), "%s/bb_s4.npy", ref_dir);
    snprintf(ref_s5, sizeof(ref_s5), "%s/bb_s5.npy", ref_dir);
    (void)REF_INPUT; (void)REF_S3; (void)REF_S4; (void)REF_S5;

    rt_detr_t *m = rt_detr_load(model);
    if (!m) { fprintf(stderr, "FAIL: load model\n"); return 1; }
    fprintf(stderr, "loaded %s\n", model);

    int ndim, dims[8];
    float *input = (float *)npy_load(ref_input, &ndim, dims, NULL);
    if (!input) { fprintf(stderr, "FAIL: load %s\n", ref_input); rt_detr_free(m); return 1; }
    if (ndim != 4 || dims[0] != 1 || dims[1] != 3 ||
        dims[2] != 640 || dims[3] != 640) {
        fprintf(stderr, "FAIL: input shape\n");
        free(input); rt_detr_free(m); return 1;
    }

    float *s3 = (float *)malloc(128 * 80 * 80 * sizeof(float));
    float *s4 = (float *)malloc(256 * 40 * 40 * sizeof(float));
    float *s5 = (float *)malloc(512 * 20 * 20 * sizeof(float));
    if (!s3 || !s4 || !s5) { fprintf(stderr, "FAIL: alloc\n"); return 1; }

    fprintf(stderr, "running backbone...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (rt_detr_forward_backbone(m, input, s3, s4, s5) != 0) {
        fprintf(stderr, "FAIL: forward_backbone\n");
        free(input); free(s3); free(s4); free(s5); rt_detr_free(m); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "  forward_backbone: %.3f s\n", dt);

    float *r3 = (float *)npy_load(ref_s3, &ndim, dims, NULL);
    float *r4 = (float *)npy_load(ref_s4, &ndim, dims, NULL);
    float *r5 = (float *)npy_load(ref_s5, &ndim, dims, NULL);
    if (!r3 || !r4 || !r5) { fprintf(stderr, "FAIL: load refs\n"); return 1; }

    fprintf(stderr, "diffs:\n");
    int ok = 1;
    ok &= diff_check("bb_s3", s3, r3, 128 * 80 * 80, 5e-3f, 1e-4);
    ok &= diff_check("bb_s4", s4, r4, 256 * 40 * 40, 5e-3f, 1e-4);
    ok &= diff_check("bb_s5", s5, r5, 512 * 20 * 20, 5e-3f, 1e-4);

    free(input); free(s3); free(s4); free(s5);
    free(r3); free(r4); free(r5);
    rt_detr_free(m);
    fprintf(stderr, "%s: rt_detr_backbone\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
