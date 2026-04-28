/* verify_rt_detr_decoder.c — verify decoder (deformable msattn + 3 layers)
 *
 * Loads /tmp/rt_detr_ref/enc_out_s{3,4,5}.npy, runs the C decoder pipeline,
 * and compares the (300, 80) class logits and (300, 4) cxcywh boxes against
 * dec_logits.npy / dec_boxes.npy.
 *
 * Tolerances:
 *   - logits: max_abs ≤ 5e-2, mean_abs ≤ 5e-3 (encoder fp32 + 3 decoder layers
 *     of softmax + bilinear sampling + LN + sigmoid composition)
 *   - boxes:  max_abs ≤ 5e-3, mean_abs ≤ 5e-4 (sigmoid compresses errors,
 *     so boxes are strictly tighter than logits)
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
    snprintf(path, sizeof(path), "%s/enc_out_s3.npy", ref_dir);
    float *enc_s3 = (float *)npy_load(path, &ndim, dims, NULL);
    snprintf(path, sizeof(path), "%s/enc_out_s4.npy", ref_dir);
    float *enc_s4 = (float *)npy_load(path, &ndim, dims, NULL);
    snprintf(path, sizeof(path), "%s/enc_out_s5.npy", ref_dir);
    float *enc_s5 = (float *)npy_load(path, &ndim, dims, NULL);
    if (!enc_s3 || !enc_s4 || !enc_s5) {
        fprintf(stderr, "FAIL: load encoder refs\n");
        rt_detr_free(m); return 1;
    }

    float *out_logits = (float *)malloc(300 * 80 * sizeof(float));
    float *out_boxes  = (float *)malloc(300 * 4  * sizeof(float));
    if (!out_logits || !out_boxes) { fprintf(stderr, "FAIL: alloc\n"); return 1; }

    fprintf(stderr, "running decoder...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (rt_detr_forward_decoder(m, enc_s3, enc_s4, enc_s5,
                                out_logits, out_boxes) != 0) {
        fprintf(stderr, "FAIL: forward_decoder\n");
        free(enc_s3); free(enc_s4); free(enc_s5);
        free(out_logits); free(out_boxes);
        rt_detr_free(m); return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "  forward_decoder: %.3f s\n", dt);

    snprintf(path, sizeof(path), "%s/dec_logits.npy", ref_dir);
    float *r_logits = (float *)npy_load(path, &ndim, dims, NULL);
    snprintf(path, sizeof(path), "%s/dec_boxes.npy", ref_dir);
    float *r_boxes  = (float *)npy_load(path, &ndim, dims, NULL);
    if (!r_logits || !r_boxes) { fprintf(stderr, "FAIL: load decoder refs\n"); return 1; }

    fprintf(stderr, "diffs:\n");
    int ok = 1;
    ok &= diff_check("dec_logits", out_logits, r_logits, 300 * 80, 5e-2f, 5e-3);
    ok &= diff_check("dec_boxes",  out_boxes,  r_boxes,  300 * 4,  5e-3f, 5e-4);

    /* Sanity: report top-5 by max class score and the predicted class. */
    fprintf(stderr, "C top-5 detections:\n");
    int top5[5];
    float top5s[5];
    for (int k = 0; k < 5; k++) { top5[k] = -1; top5s[k] = -1e30f; }
    for (int q = 0; q < 300; q++) {
        const float *r = out_logits + q * 80;
        float mx = r[0]; int mc = 0;
        for (int c = 1; c < 80; c++) if (r[c] > mx) { mx = r[c]; mc = c; }
        float score = 1.0f / (1.0f + expf(-mx));
        for (int k = 0; k < 5; k++) {
            if (score > top5s[k]) {
                for (int j = 4; j > k; j--) { top5[j] = top5[j-1]; top5s[j] = top5s[j-1]; }
                top5[k] = q; top5s[k] = score; (void)mc;
                break;
            }
        }
    }
    for (int k = 0; k < 5; k++) {
        if (top5[k] < 0) continue;
        const float *r = out_logits + top5[k] * 80;
        float mx = r[0]; int mc = 0;
        for (int c = 1; c < 80; c++) if (r[c] > mx) { mx = r[c]; mc = c; }
        const float *bx = out_boxes + top5[k] * 4;
        fprintf(stderr, "    q=%-3d cls=%-2d score=%.3f bbox=(%.3f,%.3f,%.3f,%.3f)\n",
                top5[k], mc, top5s[k], bx[0], bx[1], bx[2], bx[3]);
    }

    free(enc_s3); free(enc_s4); free(enc_s5);
    free(out_logits); free(out_boxes);
    free(r_logits); free(r_boxes);
    rt_detr_free(m);
    fprintf(stderr, "%s: rt_detr_decoder\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
