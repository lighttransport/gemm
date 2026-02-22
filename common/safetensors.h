/*
 * safetensors.h - Single-file safetensors loader with built-in JSON parser
 *
 * Usage:
 *   #define SAFETENSORS_IMPLEMENTATION
 *   #include "safetensors.h"
 *
 * API:
 *   st_context     *safetensors_open(const char *path);
 *   void            safetensors_close(st_context *ctx);
 *   int             safetensors_find(const st_context *ctx, const char *name);
 *   const char     *safetensors_name(const st_context *ctx, int i);
 *   const char     *safetensors_dtype(const st_context *ctx, int i);
 *   int             safetensors_ndims(const st_context *ctx, int i);
 *   const uint64_t *safetensors_shape(const st_context *ctx, int i);
 *   void           *safetensors_data(const st_context *ctx, int i);
 *   size_t          safetensors_nbytes(const st_context *ctx, int i);
 *   size_t          safetensors_dtype_size(const char *dtype_str);
 *
 * JSON API (general-purpose):
 *   json_val       *json_parse(const char *src, int len);
 *   void            json_free(json_val *v);
 *   json_val       *json_obj_get(const json_val *obj, const char *key);
 */
#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- JSON types ---- */

typedef enum {
    JSON_NULL, JSON_TRUE, JSON_FALSE, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT
} json_type_t;

typedef struct json_val {
    json_type_t type;
    union {
        double num;
        struct { char *ptr; int len; } str;
        struct { struct json_val *items; int count; int _cap; } arr;
        struct { char **keys; struct json_val *vals; int count; int _cap; } obj;
    };
} json_val;

json_val   *json_parse(const char *src, int len);
void        json_free(json_val *v);
json_val   *json_obj_get(const json_val *obj, const char *key);

/* ---- Safetensors types ---- */

#define ST_MAX_DIMS 8

typedef struct {
    char       *name;
    char        dtype_str[8];      /* "F32", "BF16", "F16", etc. */
    uint64_t    shape[ST_MAX_DIMS];
    int         n_dims;
    size_t      offset;            /* byte offset within data section */
    size_t      nbytes;            /* data_offsets[1] - data_offsets[0] */
} st_tensor_info;

typedef struct {
    st_tensor_info *tensors;
    int             n_tensors;
    void           *map_base;      /* mmap'd whole file */
    size_t          map_size;
    uint8_t        *data;          /* = map_base + 8 + header_size */
} st_context;

st_context     *safetensors_open(const char *path);
void            safetensors_close(st_context *ctx);
int             safetensors_find(const st_context *ctx, const char *name);
const char     *safetensors_name(const st_context *ctx, int i);
const char     *safetensors_dtype(const st_context *ctx, int i);
int             safetensors_ndims(const st_context *ctx, int i);
const uint64_t *safetensors_shape(const st_context *ctx, int i);
void           *safetensors_data(const st_context *ctx, int i);
size_t          safetensors_nbytes(const st_context *ctx, int i);
size_t          safetensors_dtype_size(const char *dtype_str);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef SAFETENSORS_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ---- JSON parser internals ---- */

typedef struct { const char *src; int pos, len; } jp_ctx;

static void jp__ws(jp_ctx *c) {
    while (c->pos < c->len) {
        char ch = c->src[c->pos];
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
            c->pos++;
        else
            break;
    }
}

static int jp__eat(jp_ctx *c, char ch) {
    jp__ws(c);
    if (c->pos < c->len && c->src[c->pos] == ch) {
        c->pos++;
        return 1;
    }
    return 0;
}

static char jp__peek(jp_ctx *c) {
    jp__ws(c);
    return (c->pos < c->len) ? c->src[c->pos] : '\0';
}

static char *jp__str(jp_ctx *c, int *out_len) {
    jp__ws(c);
    if (c->pos >= c->len || c->src[c->pos] != '"') return NULL;
    c->pos++; /* skip opening " */
    int start = c->pos;
    /* first pass: count output length */
    int olen = 0;
    int p = start;
    while (p < c->len && c->src[p] != '"') {
        if (c->src[p] == '\\') {
            p++;
            if (p < c->len) {
                if (c->src[p] == 'u') { p += 4; olen++; } /* \uXXXX -> '?' */
                else { p++; olen++; }
            }
        } else {
            p++; olen++;
        }
    }
    char *out = (char *)malloc(olen + 1);
    int o = 0;
    p = start;
    while (p < c->len && c->src[p] != '"') {
        if (c->src[p] == '\\') {
            p++;
            if (p < c->len) {
                switch (c->src[p]) {
                    case '"': case '\\': case '/': out[o++] = c->src[p]; break;
                    case 'b': out[o++] = '\b'; break;
                    case 'f': out[o++] = '\f'; break;
                    case 'n': out[o++] = '\n'; break;
                    case 'r': out[o++] = '\r'; break;
                    case 't': out[o++] = '\t'; break;
                    case 'u': out[o++] = '?'; p += 4; break;
                    default:  out[o++] = c->src[p]; break;
                }
                p++;
            }
        } else {
            out[o++] = c->src[p++];
        }
    }
    out[o] = '\0';
    if (p < c->len) p++; /* skip closing " */
    c->pos = p;
    if (out_len) *out_len = o;
    return out;
}

static json_val jp__value(jp_ctx *c); /* forward decl */

static json_val jp__number(jp_ctx *c) {
    jp__ws(c);
    int start = c->pos;
    /* scan number chars */
    while (c->pos < c->len) {
        char ch = c->src[c->pos];
        if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E')
            c->pos++;
        else
            break;
    }
    char buf[64];
    int n = c->pos - start;
    if (n >= (int)sizeof(buf)) n = (int)sizeof(buf) - 1;
    memcpy(buf, c->src + start, n);
    buf[n] = '\0';
    json_val v;
    v.type = JSON_NUMBER;
    v.num = strtod(buf, NULL);
    return v;
}

static json_val jp__array(jp_ctx *c) {
    json_val v;
    v.type = JSON_ARRAY;
    v.arr.items = NULL;
    v.arr.count = 0;
    v.arr._cap = 0;
    c->pos++; /* skip '[' */
    if (jp__peek(c) == ']') { c->pos++; return v; }
    for (;;) {
        json_val item = jp__value(c);
        if (v.arr.count >= v.arr._cap) {
            v.arr._cap = v.arr._cap ? v.arr._cap * 2 : 8;
            v.arr.items = (json_val *)realloc(v.arr.items, v.arr._cap * sizeof(json_val));
        }
        v.arr.items[v.arr.count++] = item;
        if (!jp__eat(c, ',')) break;
    }
    jp__eat(c, ']');
    return v;
}

static json_val jp__object(jp_ctx *c) {
    json_val v;
    v.type = JSON_OBJECT;
    v.obj.keys = NULL;
    v.obj.vals = NULL;
    v.obj.count = 0;
    v.obj._cap = 0;
    c->pos++; /* skip '{' */
    if (jp__peek(c) == '}') { c->pos++; return v; }
    for (;;) {
        int klen;
        char *key = jp__str(c, &klen);
        if (!key) break;
        jp__eat(c, ':');
        json_val val = jp__value(c);
        if (v.obj.count >= v.obj._cap) {
            v.obj._cap = v.obj._cap ? v.obj._cap * 2 : 16;
            v.obj.keys = (char **)realloc(v.obj.keys, v.obj._cap * sizeof(char *));
            v.obj.vals = (json_val *)realloc(v.obj.vals, v.obj._cap * sizeof(json_val));
        }
        v.obj.keys[v.obj.count] = key;
        v.obj.vals[v.obj.count] = val;
        v.obj.count++;
        if (!jp__eat(c, ',')) break;
    }
    jp__eat(c, '}');
    return v;
}

static json_val jp__value(jp_ctx *c) {
    json_val v;
    char ch = jp__peek(c);
    switch (ch) {
    case '"':
        v.type = JSON_STRING;
        v.str.ptr = jp__str(c, &v.str.len);
        return v;
    case '{': return jp__object(c);
    case '[': return jp__array(c);
    case 't': c->pos += 4; v.type = JSON_TRUE; return v;
    case 'f': c->pos += 5; v.type = JSON_FALSE; return v;
    case 'n': c->pos += 4; v.type = JSON_NULL; return v;
    default:  return jp__number(c);
    }
}

static void json_free_contents(json_val *v) {
    switch (v->type) {
    case JSON_STRING:
        free(v->str.ptr);
        break;
    case JSON_ARRAY:
        for (int i = 0; i < v->arr.count; i++)
            json_free_contents(&v->arr.items[i]);
        free(v->arr.items);
        break;
    case JSON_OBJECT:
        for (int i = 0; i < v->obj.count; i++) {
            free(v->obj.keys[i]);
            json_free_contents(&v->obj.vals[i]);
        }
        free(v->obj.keys);
        free(v->obj.vals);
        break;
    default:
        break;
    }
}

json_val *json_parse(const char *src, int len) {
    jp_ctx c = { src, 0, len };
    json_val *root = (json_val *)malloc(sizeof(json_val));
    *root = jp__value(&c);
    return root;
}

void json_free(json_val *v) {
    if (!v) return;
    json_free_contents(v);
    free(v);
}

json_val *json_obj_get(const json_val *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (int i = 0; i < obj->obj.count; i++) {
        if (strcmp(obj->obj.keys[i], key) == 0)
            return &obj->obj.vals[i];
    }
    return NULL;
}

/* ---- Safetensors loader ---- */

size_t safetensors_dtype_size(const char *dtype_str) {
    static const struct { const char *name; size_t size; } tbl[] = {
        {"F32", 4}, {"F16", 2}, {"BF16", 2}, {"F64", 8},
        {"I8", 1}, {"U8", 1}, {"I16", 2}, {"I32", 4}, {"I64", 8},
        {"BOOL", 1}, {"F8_E4M3", 1}, {"F8_E5M2", 1},
    };
    for (int i = 0; i < (int)(sizeof(tbl) / sizeof(tbl[0])); i++) {
        if (strcmp(tbl[i].name, dtype_str) == 0) return tbl[i].size;
    }
    return 0;
}

st_context *safetensors_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "safetensors: cannot open %s\n", path); return NULL; }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 8) {
        fprintf(stderr, "safetensors: file too small\n");
        close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    void *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fprintf(stderr, "safetensors: mmap failed\n");
        return NULL;
    }

    uint64_t header_size;
    memcpy(&header_size, map, 8); /* LE on LE host */
    if (8 + header_size > file_size) {
        fprintf(stderr, "safetensors: invalid header_size %lu\n", (unsigned long)header_size);
        munmap(map, file_size);
        return NULL;
    }

    json_val *root = json_parse((const char *)map + 8, (int)header_size);
    if (!root || root->type != JSON_OBJECT) {
        fprintf(stderr, "safetensors: JSON parse failed\n");
        json_free(root);
        munmap(map, file_size);
        return NULL;
    }

    /* Count tensors (skip __metadata__) */
    int n_tensors = 0;
    for (int i = 0; i < root->obj.count; i++) {
        if (strcmp(root->obj.keys[i], "__metadata__") != 0)
            n_tensors++;
    }

    st_tensor_info *tensors = (st_tensor_info *)calloc(n_tensors, sizeof(st_tensor_info));
    int ti = 0;
    for (int i = 0; i < root->obj.count; i++) {
        if (strcmp(root->obj.keys[i], "__metadata__") == 0) continue;
        json_val *entry = &root->obj.vals[i];
        if (entry->type != JSON_OBJECT) continue;

        st_tensor_info *t = &tensors[ti++];
        t->name = strdup(root->obj.keys[i]);

        /* dtype */
        json_val *jdtype = json_obj_get(entry, "dtype");
        if (jdtype && jdtype->type == JSON_STRING) {
            int dl = jdtype->str.len < 7 ? jdtype->str.len : 7;
            memcpy(t->dtype_str, jdtype->str.ptr, dl);
            t->dtype_str[dl] = '\0';
        }

        /* shape */
        json_val *jshape = json_obj_get(entry, "shape");
        if (jshape && jshape->type == JSON_ARRAY) {
            t->n_dims = jshape->arr.count < ST_MAX_DIMS ? jshape->arr.count : ST_MAX_DIMS;
            for (int d = 0; d < t->n_dims; d++)
                t->shape[d] = (uint64_t)jshape->arr.items[d].num;
        }

        /* data_offsets */
        json_val *joff = json_obj_get(entry, "data_offsets");
        if (joff && joff->type == JSON_ARRAY && joff->arr.count >= 2) {
            uint64_t off0 = (uint64_t)joff->arr.items[0].num;
            uint64_t off1 = (uint64_t)joff->arr.items[1].num;
            t->offset = (size_t)off0;
            t->nbytes = (size_t)(off1 - off0);
        }
    }

    json_free(root);

    st_context *ctx = (st_context *)malloc(sizeof(st_context));
    ctx->tensors = tensors;
    ctx->n_tensors = n_tensors;
    ctx->map_base = map;
    ctx->map_size = file_size;
    ctx->data = (uint8_t *)map + 8 + header_size;
    return ctx;
}

void safetensors_close(st_context *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_tensors; i++)
        free(ctx->tensors[i].name);
    free(ctx->tensors);
    munmap(ctx->map_base, ctx->map_size);
    free(ctx);
}

int safetensors_find(const st_context *ctx, const char *name) {
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) return i;
    }
    return -1;
}

const char *safetensors_name(const st_context *ctx, int i) {
    return ctx->tensors[i].name;
}

const char *safetensors_dtype(const st_context *ctx, int i) {
    return ctx->tensors[i].dtype_str;
}

int safetensors_ndims(const st_context *ctx, int i) {
    return ctx->tensors[i].n_dims;
}

const uint64_t *safetensors_shape(const st_context *ctx, int i) {
    return ctx->tensors[i].shape;
}

void *safetensors_data(const st_context *ctx, int i) {
    return ctx->data + ctx->tensors[i].offset;
}

size_t safetensors_nbytes(const st_context *ctx, int i) {
    return ctx->tensors[i].nbytes;
}

#endif /* SAFETENSORS_IMPLEMENTATION */
#endif /* SAFETENSORS_H */
