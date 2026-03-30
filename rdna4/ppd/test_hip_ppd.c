/*
 * test_hip_ppd.c - Test program for HIP/ROCm PPD runner
 *
 * Usage:
 *   ./test_hip_ppd <ppd.pth> <depth_anything_v2_vitl.pth> [-i image.ppm] [-o depth.pgm]
 */
#include "hip_ppd_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple PPM reader (P6 binary) */
static uint8_t *read_ppm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char hdr[4];
    if (!fgets(hdr, sizeof(hdr), f) || hdr[0] != 'P' || hdr[1] != '6') {
        fclose(f); return NULL;
    }
    /* Skip comments */
    int c;
    while ((c = fgetc(f)) == '#') while (fgetc(f) != '\n');
    ungetc(c, f);
    int maxval;
    if (fscanf(f, "%d %d %d", w, h, &maxval) != 3) { fclose(f); return NULL; }
    fgetc(f); /* skip newline */
    size_t sz = (size_t)(*w) * (*h) * 3;
    uint8_t *data = (uint8_t *)malloc(sz);
    if (fread(data, 1, sz, f) != sz) { free(data); fclose(f); return NULL; }
    fclose(f);
    return data;
}

/* Simple PGM writer (P5 binary, 16-bit) */
static void write_pgm(const char *path, const float *depth, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n65535\n", w, h);
    /* Find min/max for normalization */
    float dmin = 1e30f, dmax = -1e30f;
    for (int i = 0; i < w * h; i++) {
        if (depth[i] < dmin) dmin = depth[i];
        if (depth[i] > dmax) dmax = depth[i];
    }
    float range = dmax - dmin;
    if (range < 1e-6f) range = 1.0f;
    for (int i = 0; i < w * h; i++) {
        uint16_t v = (uint16_t)((depth[i] - dmin) / range * 65535.0f);
        uint8_t hi = v >> 8, lo = v & 0xff;
        fputc(hi, f); fputc(lo, f);
    }
    fclose(f);
    fprintf(stderr, "Wrote depth to %s (range: %.4f - %.4f)\n", path, dmin, dmax);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <ppd.pth> <da2_vitl.pth> [-i image.ppm] [-o depth.pgm]\n",
                argv[0]);
        return 1;
    }

    const char *ppd_path = argv[1];
    const char *sem_path = argv[2];
    const char *img_path = NULL;
    const char *out_path = "depth_ppd.pgm";
    int verbose = 1;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) img_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) verbose = atoi(argv[++i]);
    }

    fprintf(stderr, "Initializing HIP PPD runner...\n");
    hip_ppd_runner *r = hip_ppd_init(0, verbose);
    if (!r) { fprintf(stderr, "Failed to init HIP\n"); return 1; }

    fprintf(stderr, "Loading weights...\n");
    if (hip_ppd_load_weights(r, ppd_path, sem_path) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        hip_ppd_free(r);
        return 1;
    }

    if (img_path) {
        int w, h;
        uint8_t *rgb = read_ppm(img_path, &w, &h);
        if (!rgb) { fprintf(stderr, "Cannot read %s\n", img_path); hip_ppd_free(r); return 1; }

        fprintf(stderr, "Running inference on %s (%dx%d)...\n", img_path, w, h);
        ppd_result res = hip_ppd_predict(r, rgb, w, h);
        free(rgb);

        if (res.depth) {
            write_pgm(out_path, res.depth, res.width, res.height);
            ppd_result_free(&res);
        }
    } else {
        fprintf(stderr, "No input image specified. Weight loading test passed.\n");
    }

    hip_ppd_free(r);
    fprintf(stderr, "Done.\n");
    return 0;
}
