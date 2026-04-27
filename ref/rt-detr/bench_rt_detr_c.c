#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#define SAFETENSORS_IMPLEMENTATION
#define RT_DETR_IMPLEMENTATION
#include "../../common/rt_detr.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static double mean(const double *v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    return n ? s / (double)n : 0.0;
}

int main(int argc, char **argv)
{
    const char *model = "/mnt/disk01/models/rt_detr_s/model.safetensors";
    const char *image = "/home/syoyo/work/gemm/main/cpu/sam3d_body/samples/dancing.jpg";
    int warmup = 1, runs = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--image") && i + 1 < argc) image = argv[++i];
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc) runs = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--model PATH] [--image IMG] [--warmup N] [--runs N]\n", argv[0]);
            return 2;
        }
    }

    double t0 = now_ms();
    rt_detr_t *det = rt_detr_load(model);
    if (!det) {
        fprintf(stderr, "rt_detr_load failed: %s\n", model);
        return 3;
    }
    double t_load = now_ms() - t0;

    int w = 0, h = 0, ch = 0;
    t0 = now_ms();
    uint8_t *rgb = stbi_load(image, &w, &h, &ch, 3);
    if (!rgb) {
        fprintf(stderr, "cannot decode %s\n", image);
        rt_detr_free(det);
        return 4;
    }
    double t_decode = now_ms() - t0;

    double *ts = (double *)calloc((size_t)runs, sizeof(double));
    if (!ts) {
        stbi_image_free(rgb);
        rt_detr_free(det);
        return 5;
    }
    rt_detr_box_t box = {0};
    int kept = 0;
    for (int i = 0; i < warmup + runs; i++) {
        t0 = now_ms();
        int rc = rt_detr_detect_largest_person(det, rgb, w, h, 0.5f, &box);
        double dt = now_ms() - t0;
        if (rc != 0) {
            fprintf(stderr, "detect failed at iter %d\n", i);
            free(ts);
            stbi_image_free(rgb);
            rt_detr_free(det);
            return 6;
        }
        if (i >= warmup) {
            ts[kept++] = dt;
            printf("[bench_c] run%d detect=%.3f ms score=%.4f bbox=(%.1f,%.1f,%.1f,%.1f)\n",
                   kept, dt, box.score, box.x0, box.y0, box.x1, box.y1);
        } else {
            printf("[bench_c] warmup detect=%.3f ms score=%.4f bbox=(%.1f,%.1f,%.1f,%.1f)\n",
                   dt, box.score, box.x0, box.y0, box.x1, box.y1);
        }
    }

    double mn = ts[0], mx = ts[0];
    for (int i = 1; i < kept; i++) {
        if (ts[i] < mn) mn = ts[i];
        if (ts[i] > mx) mx = ts[i];
    }
    printf("[bench_c] image=%s size=%dx%d\n", image, w, h);
    printf("[bench_c] load=%.3f ms image_decode=%.3f ms\n", t_load, t_decode);
    printf("[bench_c] summary detect mean=%.3f ms min=%.3f max=%.3f runs=%d\n",
           mean(ts, kept), mn, mx, kept);

    free(ts);
    stbi_image_free(rgb);
    rt_detr_free(det);
    return 0;
}
