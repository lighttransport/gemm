#ifndef K3_GGUF_EXPERT_TP_H
#define K3_GGUF_EXPERT_TP_H

/* Compact loader for K3GGUFIQTP1 rank-local expert manifests.  The manifest
 * is deliberately line-oriented so the runtime can validate it without a
 * JSON dependency. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_quant.h"

typedef struct {
    int role;                 /* 1=w1, 2=w2, 3=w3, 4=router, 5=down, 6=up, 7=norm */
    int expert;
    int kind;                 /* 1=rows, 2=cols */
    int row_start, row_count;
    int col_start, col_count;
    size_t blob_offset, nbytes, blob_row_bytes;
    int type;
} k3_gguf_tp_segment;

typedef struct {
    int layer, rank, nodes;
    size_t blob_bytes;
    size_t count;
    k3_gguf_tp_segment *segments;
} k3_gguf_tp_manifest;

static inline int k3_gguf_tp_role(const char *s) {
    if (!strcmp(s, "w1")) return 1;
    if (!strcmp(s, "w2")) return 2;
    if (!strcmp(s, "w3")) return 3;
    if (!strcmp(s, "router")) return 4;
    if (!strcmp(s, "routed_down")) return 5;
    if (!strcmp(s, "routed_up")) return 6;
    if (!strcmp(s, "routed_norm")) return 7;
    return 0;
}

static inline int k3_gguf_tp_kind(const char *s) {
    if (!strcmp(s, "rows")) return 1;
    if (!strcmp(s, "cols")) return 2;
    return 0;
}

static inline void k3_gguf_tp_manifest_free(k3_gguf_tp_manifest *m) {
    if (!m) return;
    free(m->segments);
    memset(m, 0, sizeof(*m));
}

static inline int k3_gguf_tp_manifest_load(k3_gguf_tp_manifest *m,
                                           const char *path, int rank,
                                           int nodes, int layer,
                                           size_t blob_bytes) {
    if (!m || !path || rank < 0 || nodes <= 0 || rank >= nodes) return -1;
    memset(m, 0, sizeof(*m));
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[512];
    if (!fgets(line, sizeof(line), f) ||
        strncmp(line, "# K3GGUFIQTP1 ", 14)) {
        fclose(f); return -1;
    }
    int declared_segments = 0;
    unsigned long long declared_bytes = 0;
    if (sscanf(line, "# K3GGUFIQTP1 layer=%d rank=%d nodes=%d segments=%d blob_bytes=%llu",
               &m->layer, &m->rank, &m->nodes, &declared_segments,
               &declared_bytes) != 5 || m->layer != layer || m->rank != rank ||
        m->nodes != nodes || declared_segments < 0 ||
        (blob_bytes && declared_bytes != blob_bytes)) {
        fclose(f); return -1;
    }
    if (!fgets(line, sizeof(line), f) || line[0] != '#') {
        fclose(f); return -1;
    }
    if (declared_segments > 0) {
        m->segments = calloc((size_t)declared_segments, sizeof(*m->segments));
        if (!m->segments) { fclose(f); return -1; }
    }
    for (int i = 0; i < declared_segments; ++i) {
        char role[16], kind[16], type[32];
        unsigned long long off, nbytes, row_bytes;
        k3_gguf_tp_segment *s = &m->segments[i];
        if (!fgets(line, sizeof(line), f) ||
            sscanf(line, "%15s %d %15s %d %d %d %d %llu %llu %llu %31s",
                   role, &s->expert, kind, &s->row_start, &s->row_count,
                   &s->col_start, &s->col_count, &off, &nbytes, &row_bytes,
                   type) != 11) {
            k3_gguf_tp_manifest_free(m); fclose(f); return -1;
        }
        s->role = k3_gguf_tp_role(role);
        s->kind = k3_gguf_tp_kind(kind);
        s->blob_offset = (size_t)off;
        s->nbytes = (size_t)nbytes;
        s->blob_row_bytes = (size_t)row_bytes;
        s->type = k3_quant_type_from_name(type);
        if (!s->role || !s->kind ||
            (s->role != 4 && s->role != 5 && s->role != 6 && s->role != 7 &&
             (s->expert < 0 || s->expert >= 896)) ||
            ((s->role >= 4 && s->role <= 7) && s->expert != -1) ||
            s->row_start < 0 || s->row_count <= 0 || s->col_start < 0 ||
            s->col_count <= 0 || s->type <= 0 ||
            s->blob_offset > declared_bytes || s->nbytes > declared_bytes - s->blob_offset ||
            (s->kind == 2 && (!s->blob_row_bytes ||
                              s->nbytes != (size_t)s->row_count * s->blob_row_bytes))) {
            k3_gguf_tp_manifest_free(m); fclose(f); return -1;
        }
    }
    fclose(f);
    m->blob_bytes = (size_t)declared_bytes;
    m->count = (size_t)declared_segments;
    return 0;
}

#endif
