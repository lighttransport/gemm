#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3_dense.h"
#include "k3_runtime.h"

typedef struct {
    uint64_t offset, nbytes;
    int rows, cols;
    char name[512];
} entry;
typedef k3_bf16_matrix matrix;
static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int manifest(const char *path, entry *out) {
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char line[1024];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#')
            continue;
        unsigned long long o, b;
        char dt[16];
        int nd, r, c;
        if (sscanf(line, "%llu %llu %15s %d %d %d %511s", &o, &b, dt, &nd, &r,
                   &c, out[n].name) != 7 ||
            strcmp(dt, "BF16") || nd != 2 || n >= 2) {
            fclose(f);
            return -1;
        }
        out[n].offset = o;
        out[n].nbytes = b;
        out[n].rows = r;
        out[n].cols = c;
        ++n;
    }
    fclose(f);
    return n;
}
static entry *find(entry *e, const char *s) {
    for (int i = 0; i < 2; ++i)
        if (strstr(e[i].name, s))
            return &e[i];
    return NULL;
}
static uint64_t rs = 0x4b3344454e534501ULL;
static float rnd(void) {
    rs += 0x9e3779b97f4a7c15ULL;
    uint64_t z = rs;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return ((z ^ (z >> 31)) >> 40) / 8388608.f - 1.f;
}
static void mv(float *y, const matrix *m, const float *x, int threads) {
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < m->rows; r += 8) {
        const uint16_t *w = m->weight + (size_t)r * m->cols;
        matvec_bf16_8row(y + r, w, w + m->cols, w + 2 * m->cols,
                         w + 3 * m->cols, w + 4 * m->cols, w + 5 * m->cols,
                         w + 6 * m->cols, w + 7 * m->cols, x, m->cols);
    }
}
static void evict(float *b, size_t n, int th) {
    omp_set_num_threads(th);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 16)
        b[i] += 1;
}
static void perf(const matrix *r, const matrix *d, const float *x, float *yr,
                 float *yd, int th, int use_fused) {
    size_t en = (size_t)192 * 1024 * 1024 / 4;
    float *eb = calloc(en, 4);
    double sec = 0;
    int iters = 10;
    for (int i = 0; i < iters; ++i) {
        evict(eb, en, th);
        double t = now_sec();
        if (use_fused)
            k3_dense_pair_bf16(yr, yd, r, d, x, th);
        else {
            mv(yr, r, x, th);
            mv(yd, d, x, th);
        }
        sec += now_sec() - t;
    }
    double bytes =
               2.0 * (r->rows * (double)r->cols + d->rows * (double)d->cols),
           sum = 0;
    for (int i = 0; i < r->rows; ++i)
        sum += yr[i];
    for (int i = 0; i < d->rows; ++i)
        sum += yd[i];
    printf("PROBE dense mode=%s threads=%d us=%.3f GB/s=%.2f checksum=%+.6e\n",
           use_fused ? "fused" : "separate", th, sec / iters * 1e6,
           bytes / (sec / iters) / 1e9, sum);
    free(eb);
}
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s BLOB MANIFEST\n", argv[0]);
        return 2;
    }
    entry e[2];
    if (manifest(argv[2], e) != 2)
        return 2;
    k3_apply_numa_interleave();
    size_t sz;
    uint8_t *b = k3_load_blob_anon(argv[1], &sz);
    if (!b)
        return 2;
    entry *re = find(e, "gate.weight"),
          *de = find(e, "routed_expert_down_proj");
    if (!re || !de)
        return 2;
    matrix r = {(uint16_t *)(b + re->offset), re->rows, re->cols},
           d = {(uint16_t *)(b + de->offset), de->rows, de->cols};
    float *x = malloc((size_t)7168 * 4), *yr = malloc((size_t)r.rows * 4),
          *yd = malloc((size_t)d.rows * 4);
    for (int i = 0; i < 7168; ++i)
        x[i] = rnd() * .125f;
    mv(yr, &r, x, 48);
    double ref = 0;
    for (int i = 0; i < 7168; ++i)
        ref += (double)bf16_to_f32_scalar(r.weight[i]) * x[i];
    double err = fabs(ref - yr[0]);
    printf("[dense-row0] abs_err=%.3e %s\n", err, err < 2e-5 ? "OK" : "FAIL");
    int ts[] = {24, 28, 32, 36, 40, 44, 47, 48};
    for (int i = 0; i < 8; ++i) {
        perf(&r, &d, x, yr, yd, ts[i], 0);
        perf(&r, &d, x, yr, yd, ts[i], 1);
    }
    free(x);
    free(yr);
    free(yd);
    free(b);
    return err < 2e-5 ? 0 : 1;
}
