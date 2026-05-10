/*
 * safetensors_writer.h - Single-file safetensors *writer*, paired with safetensors.h.
 *
 * Companion to safetensors.h (which is read-only). Writer assembles a
 * key->(dtype, shape, raw bytes) collection in memory, then emits a single
 * safetensors file with the canonical 8-byte header_size + JSON header +
 * concatenated data sections.
 *
 * Usage:
 *   #define SAFETENSORS_WRITER_IMPLEMENTATION
 *   #include "safetensors_writer.h"
 *
 *   stw_writer *w = stw_create();
 *   uint64_t shape[2] = { (uint64_t)V, 3 };
 *   stw_add(w, "vertices", "F32", shape, 2, vertices_ptr, V * 3 * sizeof(float));
 *   stw_save(w, "mesh.safetensors");
 *   stw_destroy(w);
 *
 * No JSON escaping: tensor names must be printable ASCII without `"`/`\\`.
 */
#ifndef SAFETENSORS_WRITER_H
#define SAFETENSORS_WRITER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STW_MAX_DIMS 8

typedef struct {
    char    *name;
    char     dtype_str[8];
    uint64_t shape[STW_MAX_DIMS];
    int      n_dims;
    void    *data;     /* owned copy */
    size_t   nbytes;
} stw_tensor;

typedef struct {
    stw_tensor *tensors;
    int         n_tensors;
    int         _cap;
} stw_writer;

stw_writer *stw_create(void);
void        stw_destroy(stw_writer *w);

/* Copies `data` into the writer; caller may free its buffer immediately. */
int  stw_add(stw_writer *w,
             const char *name,
             const char *dtype_str,
             const uint64_t *shape, int n_dims,
             const void *data, size_t nbytes);

/* Returns 0 on success, nonzero on failure. */
int  stw_save(const stw_writer *w, const char *path);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef SAFETENSORS_WRITER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

stw_writer *stw_create(void) {
    stw_writer *w = (stw_writer *)calloc(1, sizeof(stw_writer));
    return w;
}

void stw_destroy(stw_writer *w) {
    if (!w) return;
    for (int i = 0; i < w->n_tensors; i++) {
        free(w->tensors[i].name);
        free(w->tensors[i].data);
    }
    free(w->tensors);
    free(w);
}

int stw_add(stw_writer *w,
            const char *name,
            const char *dtype_str,
            const uint64_t *shape, int n_dims,
            const void *data, size_t nbytes) {
    if (!w || !name || !dtype_str || n_dims < 0 || n_dims > STW_MAX_DIMS) return -1;
    if (nbytes && !data) return -1;

    if (w->n_tensors >= w->_cap) {
        int nc = w->_cap ? w->_cap * 2 : 8;
        stw_tensor *nt = (stw_tensor *)realloc(w->tensors, nc * sizeof(stw_tensor));
        if (!nt) return -1;
        w->tensors = nt;
        w->_cap = nc;
    }
    stw_tensor *t = &w->tensors[w->n_tensors];
    memset(t, 0, sizeof(*t));
    t->name = strdup(name);
    strncpy(t->dtype_str, dtype_str, 7);
    t->dtype_str[7] = '\0';
    t->n_dims = n_dims;
    for (int d = 0; d < n_dims; d++) t->shape[d] = shape[d];
    t->nbytes = nbytes;
    if (nbytes) {
        t->data = malloc(nbytes);
        if (!t->data) { free(t->name); return -1; }
        memcpy(t->data, data, nbytes);
    }
    w->n_tensors++;
    return 0;
}

/* Append printf-style to a growable string. */
typedef struct { char *p; size_t len; size_t cap; } stw_sb;
static void stw_sb_putc(stw_sb *s, char c) {
    if (s->len + 1 >= s->cap) {
        size_t nc = s->cap ? s->cap * 2 : 256;
        s->p = (char *)realloc(s->p, nc);
        s->cap = nc;
    }
    s->p[s->len++] = c;
    s->p[s->len] = '\0';
}
static void stw_sb_puts(stw_sb *s, const char *str) {
    while (*str) stw_sb_putc(s, *str++);
}
static void stw_sb_putu(stw_sb *s, uint64_t v) {
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%llu", (unsigned long long)v);
    for (int i = 0; i < n; i++) stw_sb_putc(s, buf[i]);
}

int stw_save(const stw_writer *w, const char *path) {
    if (!w || !path) return -1;

    /* Build JSON header. */
    stw_sb sb = {0};
    stw_sb_putc(&sb, '{');
    uint64_t cur_off = 0;
    for (int i = 0; i < w->n_tensors; i++) {
        const stw_tensor *t = &w->tensors[i];
        if (i > 0) stw_sb_putc(&sb, ',');
        stw_sb_putc(&sb, '"');
        stw_sb_puts(&sb, t->name);
        stw_sb_puts(&sb, "\":{\"dtype\":\"");
        stw_sb_puts(&sb, t->dtype_str);
        stw_sb_puts(&sb, "\",\"shape\":[");
        for (int d = 0; d < t->n_dims; d++) {
            if (d) stw_sb_putc(&sb, ',');
            stw_sb_putu(&sb, t->shape[d]);
        }
        stw_sb_puts(&sb, "],\"data_offsets\":[");
        stw_sb_putu(&sb, cur_off);
        stw_sb_putc(&sb, ',');
        stw_sb_putu(&sb, cur_off + t->nbytes);
        stw_sb_puts(&sb, "]}");
        cur_off += t->nbytes;
    }
    stw_sb_putc(&sb, '}');

    /* Pad header to 8-byte alignment for the data section that follows. */
    size_t header_len = sb.len;
    size_t pad = (8 - (header_len % 8)) % 8;
    for (size_t k = 0; k < pad; k++) stw_sb_putc(&sb, ' ');
    header_len = sb.len;

    FILE *fp = fopen(path, "wb");
    if (!fp) { free(sb.p); return -1; }
    uint64_t hsz = (uint64_t)header_len;
    if (fwrite(&hsz, 8, 1, fp) != 1) goto fail;
    if (fwrite(sb.p, 1, header_len, fp) != header_len) goto fail;
    for (int i = 0; i < w->n_tensors; i++) {
        const stw_tensor *t = &w->tensors[i];
        if (t->nbytes && fwrite(t->data, 1, t->nbytes, fp) != t->nbytes) goto fail;
    }
    fclose(fp);
    free(sb.p);
    return 0;
fail:
    fclose(fp);
    free(sb.p);
    return -1;
}

#endif /* SAFETENSORS_WRITER_IMPLEMENTATION */
#endif /* SAFETENSORS_WRITER_H */
