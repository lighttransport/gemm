#define _GNU_SOURCE
#define GGML_DEQUANT_IMPLEMENTATION
#include "k3_quant.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    size_t off, nbytes;
    int type, ndims;
    size_t shape[3];
    char name[512];
} entry;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int dtype(const char *s) {
    return k3_quant_type_from_name(s);
}

static int load_manifest(const char *path, entry **out, int *count, size_t *blob_bytes) {
    FILE *f = fopen(path, "r");
    if (!f) return errno ? errno : EIO;
    int cap = 128, n = 0;
    entry *a = calloc((size_t)cap, sizeof(*a));
    char line[2048];
    *blob_bytes = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            unsigned long long b = 0;
            if (sscanf(line, "# K3GGUFV1 %*[^b]blob_bytes=%llu", &b) == 1)
                *blob_bytes = (size_t)b;
            continue;
        }
        if (!line[0] || line[0] == '\n') continue;
        if (n == cap) {
            cap *= 2;
            entry *next = realloc(a, (size_t)cap * sizeof(*a));
            if (!next) { free(a); fclose(f); return ENOMEM; }
            a = next;
        }
        char typ[32];
        char *p = line;
        unsigned long long off, bytes;
        int nd;
        int used = 0;
        if (sscanf(p, "%llu %llu %31s %d %n", &off, &bytes, typ, &nd, &used) != 4 ||
            nd < 1 || nd > 3) { free(a); fclose(f); return EINVAL; }
        p += used;
        a[n].off = (size_t)off; a[n].nbytes = (size_t)bytes;
        a[n].type = dtype(typ); a[n].ndims = nd;
        if (!a[n].type) { free(a); fclose(f); return EINVAL; }
        for (int i = 0; i < nd; ++i) {
            char *end;
            unsigned long long d = strtoull(p, &end, 10);
            if (end == p) { free(a); fclose(f); return EINVAL; }
            a[n].shape[i] = (size_t)d; p = end;
        }
        while (*p == ' ' || *p == '\t') ++p;
        p[strcspn(p, "\r\n")] = 0;
        if (!*p || strlen(p) >= sizeof a[n].name) { free(a); fclose(f); return EINVAL; }
        strcpy(a[n].name, p);
        ++n;
    }
    fclose(f);
    *out = a; *count = n;
    return 0;
}

static int read_blob(const char *path, uint8_t **out, size_t bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return errno;
    struct stat st;
    if (fstat(fd, &st) || (size_t)st.st_size < bytes) { close(fd); return EINVAL; }
    uint8_t *p = malloc(bytes);
    if (!p) { close(fd); return ENOMEM; }
    size_t done = 0;
    while (done < bytes) {
        size_t want = bytes - done;
        if (want > 8 * 1024 * 1024) want = 8 * 1024 * 1024;
        ssize_t got = pread(fd, p + done, want, (off_t)done);
        if (got <= 0) { free(p); close(fd); return EIO; }
        done += (size_t)got;
    }
    (void)posix_fadvise(fd, 0, (off_t)bytes, POSIX_FADV_DONTNEED);
    close(fd); *out = p;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s MANIFEST BLOB [reps]\n", argv[0]);
        return 2;
    }
    int reps = argc == 4 ? atoi(argv[3]) : 3;
    if (reps < 1) reps = 1;
    entry *a = NULL; int n = 0; size_t blob_bytes = 0;
    int rc = load_manifest(argv[1], &a, &n, &blob_bytes);
    if (rc) { fprintf(stderr, "manifest: %s\n", strerror(rc)); return 2; }
    if (!blob_bytes) { fprintf(stderr, "manifest has no blob size\n"); free(a); return 2; }
    uint8_t *blob = NULL;
    rc = read_blob(argv[2], &blob, blob_bytes);
    if (rc) { fprintf(stderr, "blob: %s\n", strerror(rc)); free(a); return 2; }
    double layer_ms = 0.0;
    int measured = 0;
    for (int i = 0; i < n; ++i) {
        if (a[i].ndims < 2) continue;
        int cols = (int)a[i].shape[0];
        size_t rows64 = 1;
        for (int d = 1; d < a[i].ndims; ++d) rows64 *= a[i].shape[d];
        size_t bench_rows64 = rows64;
        if (a[i].ndims == 3 && strstr(a[i].name, "exps") != NULL) {
            /* One decode token routes a small expert set, not all 896 experts.
             * The staged expert axis is contiguous, so benchmark the first
             * eight local experts as the rank-local active-set proxy. */
            size_t active = a[i].shape[2] < 8 ? a[i].shape[2] : 8;
            bench_rows64 = a[i].shape[1] * active;
        }
        if (rows64 > 1000000 || bench_rows64 > 1000000 ||
            !k3_quant_valid_shape(a[i].type, 1, cols)) continue;
        int rows = (int)bench_rows64;
        float *x = malloc((size_t)cols * sizeof(*x));
        float *y = malloc((size_t)rows * sizeof(*y));
        if (!x || !y) { free(x); free(y); free(blob); free(a); return 3; }
        for (int c = 0; c < cols; ++c) x[c] = 0.001f * (float)((c % 127) - 63);
        k3_quant_matrix m = {blob + a[i].off, a[i].type, rows, cols,
                             a[i].nbytes / rows64};
        int threads = omp_get_max_threads();
        int mode = k3_quant_kernel_mode_env();
        (void)k3_quant_matvec_mode(y, &m, x, threads, mode);
        double t0 = now_s();
        for (int r = 0; r < reps; ++r)
            (void)k3_quant_matvec_mode(y, &m, x, threads, mode);
        double ms = (now_s() - t0) * 1000.0 / reps;
        layer_ms += ms; ++measured;
        printf("K3_QBENCH mode=%s tensor=%s type=%s rows=%d cols=%d ms=%.3f tok/s=%.3f\n",
               mode == K3_QUANT_REFERENCE ? "reference" :
               mode == K3_QUANT_SVE_Q8 ? "sve-q8" : "sve-a16",
               a[i].name, k3_quant_type_name(a[i].type), rows, cols, ms,
               ms > 0.0 ? 1000.0 / ms : 0.0);
        free(x); free(y);
    }
    printf("K3_QBENCH_LAYER measured=%d ms=%.3f quant_projection_tok_s=%.3f blob_mib=%.1f\n",
           measured, layer_ms, layer_ms > 0.0 ? 1000.0 / layer_ms : 0.0,
           (double)blob_bytes / (1024.0 * 1024.0));
    free(blob); free(a);
    return measured ? 0 : 3;
}
