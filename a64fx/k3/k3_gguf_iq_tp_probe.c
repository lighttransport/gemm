#define _GNU_SOURCE
#include "k3_gguf_expert_tp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int read_blob(const char *path, unsigned char **out, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unsigned char *p = malloc(bytes ? bytes : 1);
    if (!p) { fclose(f); return -1; }
    size_t got = fread(p, 1, bytes, f);
    fclose(f);
    if (got != bytes) { free(p); return -1; }
    *out = p;
    return 0;
}

static const k3_gguf_tp_segment *find_segment(
        const k3_gguf_tp_manifest *m, int role, int expert) {
    for (size_t i = 0; i < m->count; ++i)
        if (m->segments[i].role == role && m->segments[i].expert == expert)
            return &m->segments[i];
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 8) {
        fprintf(stderr, "usage: %s MANIFEST BLOB [reps] [mode=ref|a16|q8] [rank nodes layer]\n", argv[0]);
        return 2;
    }
    int reps = argc > 3 ? atoi(argv[3]) : 3;
    int mode = K3_QUANT_SVE_A16;
    if (argc > 4) {
        if (!strcmp(argv[4], "ref")) mode = K3_QUANT_REFERENCE;
        else if (!strcmp(argv[4], "q8")) mode = K3_QUANT_SVE_Q8;
        else if (strcmp(argv[4], "a16")) return 2;
    }
    if (reps <= 0) return 2;
    int rank = argc > 5 ? atoi(argv[5]) : 0;
    int nodes = argc > 6 ? atoi(argv[6]) : 12;
    int layer = argc > 7 ? atoi(argv[7]) : 1;
    k3_gguf_tp_manifest m;
    if (k3_gguf_tp_manifest_load(&m, argv[1], rank, nodes, layer, 0)) return 3;
    unsigned char *blob = NULL;
    if (read_blob(argv[2], &blob, m.blob_bytes)) {
        k3_gguf_tp_manifest_free(&m); return 4;
    }
    float *x = malloc(3584 * sizeof(*x));
    float *gate = malloc(256 * sizeof(*gate));
    float *up = malloc(256 * sizeof(*up));
    float *down = malloc(3584 * sizeof(*down));
    if (!x || !gate || !up || !down) return 5;
    for (int i = 0; i < 3584; ++i) x[i] = (float)((i % 29) - 14) * 0.007f;
    int experts = 0, bad = 0;
    double t0 = now_sec();
    for (int rep = 0; rep < reps; ++rep) {
        int last = -1;
        for (size_t i = 0; i < m.count; ++i) {
            int expert = m.segments[i].expert;
            if (expert == last) continue;
            last = expert;
            const k3_gguf_tp_segment *w1 = find_segment(&m, 1, expert);
            const k3_gguf_tp_segment *w2 = find_segment(&m, 2, expert);
            const k3_gguf_tp_segment *w3 = find_segment(&m, 3, expert);
            if (!w1 || !w2 || !w3 || w1->kind != 1 || w2->kind != 2 ||
                w3->kind != 1 || w1->col_count != 3584 ||
                w2->col_count != 256 || w3->col_count != 3584) { bad = 1; continue; }
            const unsigned char *p1 = blob + w1->blob_offset;
            const unsigned char *p2 = blob + w2->blob_offset;
            const unsigned char *p3 = blob + w3->blob_offset;
            size_t rb1 = w1->nbytes / (size_t)w1->row_count;
            size_t rb2 = w2->blob_row_bytes;
            int rc = k3_quant_matvec_mode(gate,
                &(k3_quant_matrix){p1, w1->type, w1->row_count,
                                   w1->col_count, rb1}, x, 40, mode);
            rc |= k3_quant_matvec_mode(up,
                &(k3_quant_matrix){p3, w3->type, w3->row_count,
                                   w3->col_count, w3->nbytes / (size_t)w3->row_count},
                x, 40, mode);
            rc |= k3_quant_matvec_mode(down,
                &(k3_quant_matrix){p2, w2->type, w2->row_count,
                                   w2->col_count, rb2}, gate, 40, mode);
            if (rc || !isfinite(down[0])) bad = 1;
            ++experts;
        }
    }
    double elapsed = now_sec() - t0;
    printf("K3_IQTP_PROBE layer=%d rank=%d nodes=%d experts=%d reps=%d mode=%s "
           "seconds=%.6f expert_triplets_per_sec=%.3f status=%s\n",
           m.layer, m.rank, m.nodes, experts / reps, reps,
           mode == K3_QUANT_REFERENCE ? "ref" : mode == K3_QUANT_SVE_Q8 ? "q8" : "a16",
           elapsed, (double)(experts / reps) / (elapsed + 1e-12), bad ? "FAIL" : "PASS");
    free(x); free(gate); free(up); free(down); free(blob);
    k3_gguf_tp_manifest_free(&m);
    return bad;
}
