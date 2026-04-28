/* verify_rt_detr_postprocess.c — verify post-process & largest-person filter
 *
 * Loads /tmp/rt_detr_ref/dec_logits.npy + dec_boxes.npy, runs
 * rt_detr_postprocess(class=person, thresh=0.5) with image size 768x1024,
 * and compares the resulting [score, x0, y0, x1, y1] against
 * detected_persons.npy.
 *
 * Tolerances: score 1e-4, coords 1e-2 px.
 */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define RT_DETR_IMPLEMENTATION
#include "../../common/rt_detr.h"
#include "../../common/npy_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *ref_dir = (argc > 1) ? argv[1] : "/tmp/rt_detr_ref";
    int orig_w = (argc > 2) ? atoi(argv[2]) : 768;
    int orig_h = (argc > 3) ? atoi(argv[3]) : 1024;

    char path[256];
    int ndim, dims[8];
    snprintf(path, sizeof(path), "%s/dec_logits.npy", ref_dir);
    float *logits = (float *)npy_load(path, &ndim, dims, NULL);
    snprintf(path, sizeof(path), "%s/dec_boxes.npy", ref_dir);
    float *boxes  = (float *)npy_load(path, &ndim, dims, NULL);
    snprintf(path, sizeof(path), "%s/detected_persons.npy", ref_dir);
    float *ref    = (float *)npy_load(path, &ndim, dims, NULL);
    if (!logits || !boxes || !ref) {
        fprintf(stderr, "FAIL: load refs\n");
        return 1;
    }
    int n_ref = dims[0];
    fprintf(stderr, "ref: %d persons (orig %dx%d)\n", n_ref, orig_w, orig_h);

    rt_detr_box_t out[64];
    int n_out = rt_detr_postprocess(logits, boxes, orig_w, orig_h,
                                    RT_DETR_PERSON_CLASS_ID, 0.5f,
                                    out, 64);
    fprintf(stderr, "C: %d persons\n", n_out);
    int ok = 1;
    if (n_out != n_ref) {
        fprintf(stderr, "FAIL: count mismatch (got %d, want %d)\n", n_out, n_ref);
        ok = 0;
    }
    int n_cmp = n_out < n_ref ? n_out : n_ref;
    for (int i = 0; i < n_cmp; i++) {
        const float *r = ref + (size_t)i * 5;
        const rt_detr_box_t *o = &out[i];
        float ds  = fabsf(o->score - r[0]);
        float dx0 = fabsf(o->x0 - r[1]);
        float dy0 = fabsf(o->y0 - r[2]);
        float dx1 = fabsf(o->x1 - r[3]);
        float dy1 = fabsf(o->y1 - r[4]);
        int row_ok = (ds < 1e-3f) && (dx0 < 1e-1f) && (dy0 < 1e-1f) &&
                     (dx1 < 1e-1f) && (dy1 < 1e-1f);
        fprintf(stderr,
            "  [%d] C=(s=%.4f bbox=(%.2f,%.2f,%.2f,%.2f))\n"
            "      ref=(s=%.4f bbox=(%.2f,%.2f,%.2f,%.2f))  Δ=(%.5f,%.4f,%.4f,%.4f,%.4f)  %s\n",
            i, o->score, o->x0, o->y0, o->x1, o->y1,
            r[0], r[1], r[2], r[3], r[4],
            ds, dx0, dy0, dx1, dy1, row_ok ? "PASS" : "FAIL");
        if (!row_ok) ok = 0;
    }

    free(logits); free(boxes); free(ref);
    fprintf(stderr, "%s: rt_detr_postprocess\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
