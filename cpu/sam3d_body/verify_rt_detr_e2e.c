/* verify_rt_detr_e2e.c — full image → largest-person bbox pipeline
 *
 * Loads person.jpg, runs rt_detr_detect_largest_person, compares
 * against /tmp/rt_detr_ref/detected_persons.npy.
 *
 * Note: end-to-end CPU forward includes our manual bilinear preprocess
 * (PIL antialias differs from ours by ~0.09 max_abs at edges, ~0.0037
 * mean — which propagates through the network). So tolerances are
 * looser than the per-stage verifies.
 */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define RT_DETR_IMPLEMENTATION
#include "../../common/rt_detr.h"
#include "../../common/npy_io.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *MODEL_PATH = "/mnt/disk01/models/rt_detr_s/model.safetensors";
static const char *IMG_PATH   =
    "/home/syoyo/work/gemm/main/web/public/sam3_compare/person.jpg";
static const char *REF_PERSONS = "/tmp/rt_detr_ref/detected_persons.npy";

int main(int argc, char **argv) {
    const char *model = (argc > 1) ? argv[1] : MODEL_PATH;
    const char *img   = (argc > 2) ? argv[2] : IMG_PATH;
    const char *ref   = (argc > 3) ? argv[3] : REF_PERSONS;

    rt_detr_t *m = rt_detr_load(model);
    if (!m) { fprintf(stderr, "FAIL: rt_detr_load(%s)\n", model); return 1; }
    fprintf(stderr, "loaded %s\n", model);

    int w, h, ch;
    uint8_t *rgb = stbi_load(img, &w, &h, &ch, 3);
    if (!rgb) { fprintf(stderr, "FAIL: load image %s\n", img); return 1; }
    fprintf(stderr, "image: %s (%dx%d)\n", img, w, h);

    rt_detr_box_t out;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = rt_detr_detect_largest_person(m, rgb, w, h, 0.5f, &out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    free(rgb);

    if (rc != 0) {
        fprintf(stderr, "FAIL: no person detected\n");
        rt_detr_free(m);
        return 1;
    }
    fprintf(stderr, "  rt_detr_detect_largest_person: %.3f s\n", dt);
    fprintf(stderr, "  C:   score=%.4f bbox=(%.2f,%.2f,%.2f,%.2f)\n",
            out.score, out.x0, out.y0, out.x1, out.y1);

    int ndim, dims[8];
    float *r = (float *)npy_load(ref, &ndim, dims, NULL);
    if (!r || dims[0] < 1) {
        fprintf(stderr, "FAIL: load %s\n", ref);
        rt_detr_free(m); return 1;
    }
    fprintf(stderr, "  ref: score=%.4f bbox=(%.2f,%.2f,%.2f,%.2f)\n",
            r[0], r[1], r[2], r[3], r[4]);
    /* Loose tolerances: the preprocess differs from PIL antialiased
     * resize, so we allow ~5px / ~0.05 score drift. */
    int ok = (fabsf(out.score - r[0]) < 0.05f) &&
             (fabsf(out.x0    - r[1]) < 5.0f) &&
             (fabsf(out.y0    - r[2]) < 5.0f) &&
             (fabsf(out.x1    - r[3]) < 5.0f) &&
             (fabsf(out.y1    - r[4]) < 5.0f);
    fprintf(stderr, "  Δ: dscore=%.4f bbox_l1=(%.2f,%.2f,%.2f,%.2f)\n",
            fabsf(out.score - r[0]),
            fabsf(out.x0 - r[1]), fabsf(out.y0 - r[2]),
            fabsf(out.x1 - r[3]), fabsf(out.y1 - r[4]));

    free(r);
    rt_detr_free(m);
    fprintf(stderr, "%s: rt_detr_e2e\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
