/*
 * test_ppd.c - Test program for CPU Pixel-Perfect-Depth
 *
 * Usage:
 *   gcc -O3 -mf16c -mavx2 -mfma -o test_ppd test_ppd.c -lm -lpthread
 *   ./test_ppd <ppd.pth> <da2.pth> -i input.ppm -o depth.pgm [-t threads] [-v]
 */

#define PTH_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define PIXEL_PERFECT_DEPTH_IMPLEMENTATION
#include "pth_loader.h"
#include "ggml_dequant.h"
#include "pixel_perfect_depth.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Read binary PPM (P6) */
static uint8_t *read_ppm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    char magic[4];
    if (!fgets(magic, sizeof(magic), f) || (magic[0] != 'P' || magic[1] != '6')) {
        fprintf(stderr, "%s: not a P6 PPM\n", path);
        fclose(f); return NULL;
    }
    /* Skip comments */
    int c;
    while ((c = fgetc(f)) == '#') while ((c = fgetc(f)) != '\n' && c != EOF);
    ungetc(c, f);
    int maxval;
    if (fscanf(f, "%d %d %d", w, h, &maxval) != 3) {
        fprintf(stderr, "%s: bad PPM header\n", path);
        fclose(f); return NULL;
    }
    fgetc(f); /* consume newline */
    size_t sz = (size_t)(*w) * (*h) * 3;
    uint8_t *data = (uint8_t *)malloc(sz);
    if (fread(data, 1, sz, f) != sz) {
        fprintf(stderr, "%s: short read\n", path);
        free(data); fclose(f); return NULL;
    }
    fclose(f);
    return data;
}

/* Write 16-bit PGM (P5) */
static void write_pgm16(const char *path, const float *depth, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fprintf(f, "P5\n%d %d\n65535\n", w, h);
    /* Find min/max for normalization */
    float mn = depth[0], mx = depth[0];
    for (int i = 1; i < w * h; i++) {
        if (depth[i] < mn) mn = depth[i];
        if (depth[i] > mx) mx = depth[i];
    }
    float range = mx - mn;
    if (range < 1e-8f) range = 1.0f;
    for (int i = 0; i < w * h; i++) {
        uint16_t v = (uint16_t)((depth[i] - mn) / range * 65535.0f + 0.5f);
        uint8_t hi = (uint8_t)(v >> 8), lo = (uint8_t)(v & 0xFF);
        fputc(hi, f); fputc(lo, f); /* big-endian */
    }
    fclose(f);
    fprintf(stderr, "wrote %s (%dx%d, depth range [%.4f, %.4f])\n", path, w, h, mn, mx);
}

int main(int argc, char **argv) {
    const char *ppd_path = NULL, *sem_path = NULL;
    const char *input = NULL, *output = "depth_cpu.pgm";
    int n_threads = 4, verbose = 0;

    /* Parse positional args: ppd.pth sem.pth */
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) { input = argv[++i]; }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) { output = argv[++i]; }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) { n_threads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-v") == 0) { verbose++; }
        else if (argv[i][0] != '-') {
            if (pos == 0) ppd_path = argv[i];
            else if (pos == 1) sem_path = argv[i];
            pos++;
        }
    }

    if (!ppd_path || !sem_path || !input) {
        fprintf(stderr, "Usage: %s <ppd.pth> <da2.pth> -i input.ppm [-o output.pgm] [-t threads] [-v]\n",
                argv[0]);
        return 1;
    }

    /* Load image */
    int w, h;
    uint8_t *rgb = read_ppm(input, &w, &h);
    if (!rgb) return 1;
    fprintf(stderr, "input: %s (%dx%d)\n", input, w, h);

    /* Load model */
    ppd_model *m = ppd_load(ppd_path, sem_path, verbose + 1);
    if (!m) { free(rgb); return 1; }

    /* Predict */
    ppd_result res = ppd_predict(m, rgb, w, h, n_threads);
    if (!res.depth) {
        fprintf(stderr, "prediction failed\n");
        ppd_free(m); free(rgb);
        return 1;
    }

    /* Write output */
    write_pgm16(output, res.depth, res.width, res.height);

    /* Compute stats */
    float sum = 0, mn = res.depth[0], mx = res.depth[0];
    int n = res.width * res.height;
    for (int i = 0; i < n; i++) {
        sum += res.depth[i];
        if (res.depth[i] < mn) mn = res.depth[i];
        if (res.depth[i] > mx) mx = res.depth[i];
    }
    fprintf(stderr, "depth stats: min=%.4f max=%.4f mean=%.4f\n", mn, mx, sum / n);

    /* Spatial coherence check */
    double diff_sum = 0;
    int diff_count = 0;
    for (int y = 0; y < res.height; y++) {
        for (int x = 0; x < res.width - 1; x++) {
            float d = res.depth[y * res.width + x] - res.depth[y * res.width + x + 1];
            diff_sum += (d < 0 ? -d : d);
            diff_count++;
        }
    }
    fprintf(stderr, "avg adjacent diff: %.6f (spatial coherence check)\n",
            diff_sum / diff_count);

    ppd_result_free(&res);
    ppd_free(m);
    free(rgb);
    return 0;
}
