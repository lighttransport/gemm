#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_all(const char *path, size_t *count)
{
    FILE *file = fopen(path, "rb");
    long bytes;
    float *values;
    if (!file || fseek(file, 0, SEEK_END) || (bytes = ftell(file)) < 0 ||
        bytes % (long)sizeof(float) || fseek(file, 0, SEEK_SET)) return NULL;
    values = malloc((size_t)bytes);
    if (!values || fread(values, 1, (size_t)bytes, file) != (size_t)bytes ||
        fclose(file)) { free(values); return NULL; }
    *count = (size_t)bytes / sizeof(*values);
    return values;
}

int main(int argc, char **argv)
{
    double atol = 1e-4, rtol = 1e-3, min_cos = 0.999999;
    size_t left_offset = 0, right_offset = 0, limit = 0;
    const char *left = NULL, *right = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--atol") && i + 1 < argc) atol = strtod(argv[++i], NULL);
        else if (!strcmp(argv[i], "--rtol") && i + 1 < argc) rtol = strtod(argv[++i], NULL);
        else if (!strcmp(argv[i], "--cos") && i + 1 < argc) min_cos = strtod(argv[++i], NULL);
        else if (!strcmp(argv[i], "--left-offset") && i + 1 < argc) left_offset = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--right-offset") && i + 1 < argc) right_offset = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) limit = strtoull(argv[++i], NULL, 0);
        else if (!left) left = argv[i]; else if (!right) right = argv[i]; else return 2;
    }
    if (!left || !right) {
        fprintf(stderr, "usage: %s [--atol N] [--rtol N] [--cos N] LEFT RIGHT\n", argv[0]);
        return 2;
    }
    size_t nl = 0, nr = 0, bad = 0, failed = 0, worst = 0;
    float *a = read_all(left, &nl), *b = read_all(right, &nr);
    if (!a || !b || left_offset > nl || right_offset > nr ||
        (!limit && (nl - left_offset != nr - right_offset)) ||
        (limit && (limit > nl - left_offset || limit > nr - right_offset))) {
        fprintf(stderr, "compare_f32: invalid inputs (%s)\n", strerror(errno));
        free(a); free(b); return 2;
    }
    a += left_offset; b += right_offset;
    nl = limit ? limit : nl - left_offset;
    double max_abs = 0, max_rel = 0, sum2 = 0, aa = 0, bb = 0, ab = 0;
    for (size_t i = 0; i < nl; ++i) {
        if (!isfinite(a[i]) || !isfinite(b[i])) { bad++; failed++; continue; }
        double error = fabs((double)a[i] - b[i]);
        double scale = fmax(fabs((double)a[i]), fabs((double)b[i]));
        double rel = scale ? error / scale : 0;
        if (error > max_abs) { max_abs = error; worst = i; }
        if (rel > max_rel) max_rel = rel;
        if (error > atol + rtol * scale) failed++;
        sum2 += error * error; aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i]; ab += (double)a[i] * b[i];
    }
    double cosine = aa && bb ? ab / sqrt(aa * bb) : 0;
    printf("Q38FN_COMPARE count=%zu nonfinite=%zu failed=%zu max_abs=%.9g "
           "max_rel=%.9g rmse=%.9g cosine=%.12g worst=%zu left=%.9g right=%.9g\n",
           nl, bad, failed, max_abs, max_rel, sqrt(sum2 / nl), cosine, worst,
           a[worst], b[worst]);
    free(a - left_offset); free(b - right_offset);
    return failed || cosine < min_cos ? 1 : 0;
}
