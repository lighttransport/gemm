/*
 * gguf_loader.h - Single-file GGUF v2/v3 loader with optional mmap support
 *
 * Usage:
 *   #define GGUF_LOADER_IMPLEMENTATION
 *   #include "gguf_loader.h"
 *
 * API:
 *   gguf_context *gguf_open(const char *path, int use_mmap);
 *   void gguf_close(gguf_context *ctx);
 *   int gguf_find_key(const gguf_context *ctx, const char *key);
 *   const char *gguf_tensor_name(const gguf_context *ctx, int i);
 *   void *gguf_tensor_data(const gguf_context *ctx, int i);
 *   size_t gguf_tensor_size(const gguf_context *ctx, int i);
 */
#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF value types */
enum gguf_value_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT
};

/* ggml tensor data types */
enum ggml_dtype {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_Q4_0_4_4 = 31,
    GGML_TYPE_Q4_0_4_8 = 32,
    GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_COUNT
};

typedef struct {
    uint64_t len;
    char *str;  /* NOT null-terminated in file, but we null-terminate in memory */
} gguf_str;

typedef struct {
    gguf_str key;
    uint32_t type; /* gguf_value_type */
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        uint8_t  b;
        gguf_str str;
        struct {
            uint32_t type; /* element type */
            uint64_t n;
            void *data;    /* raw array data (strings stored as gguf_str*) */
        } arr;
    } value;
} gguf_kv;

typedef struct {
    gguf_str name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type; /* ggml_dtype */
    uint64_t offset; /* offset from start of data section */
} gguf_tensor_info;

typedef struct {
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    gguf_kv *kv;
    gguf_tensor_info *tensors;
    uint32_t alignment;
    size_t data_offset; /* byte offset in file where tensor data starts */
    uint8_t *data;      /* pointer to tensor data (mmap'd or malloc'd) */
    size_t data_size;
    int use_mmap;
#ifdef _WIN32
    void *map_handle;
    void *file_handle;
    void *map_base;
    size_t map_size;
#else
    void *map_base;
    size_t map_size;
    int fd;
#endif
} gguf_context;

gguf_context *gguf_open(const char *path, int use_mmap);
void gguf_close(gguf_context *ctx);
int gguf_find_key(const gguf_context *ctx, const char *key);
const char *gguf_tensor_name(const gguf_context *ctx, int i);
void *gguf_tensor_data(const gguf_context *ctx, int i);
size_t gguf_tensor_size(const gguf_context *ctx, int i);
const char *gguf_type_name(uint32_t type);
const char *ggml_type_name(uint32_t type);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef GGUF_LOADER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* block_size, type_size pairs for ggml types */
static const struct { int block_size; int type_size; } ggml_type_info[] = {
    [GGML_TYPE_F32]     = {1, 4},
    [GGML_TYPE_F16]     = {1, 2},
    [GGML_TYPE_Q4_0]    = {32, 18},
    [GGML_TYPE_Q4_1]    = {32, 20},
    [GGML_TYPE_Q5_0]    = {32, 22},
    [GGML_TYPE_Q5_1]    = {32, 24},
    [GGML_TYPE_Q8_0]    = {32, 34},
    [GGML_TYPE_Q8_1]    = {32, 36},
    [GGML_TYPE_Q2_K]    = {256, 84},
    [GGML_TYPE_Q3_K]    = {256, 110},
    [GGML_TYPE_Q4_K]    = {256, 144},
    [GGML_TYPE_Q5_K]    = {256, 176},
    [GGML_TYPE_Q6_K]    = {256, 210},
    [GGML_TYPE_Q8_K]    = {256, 292},
    [GGML_TYPE_IQ2_XXS] = {256, 66},
    [GGML_TYPE_IQ2_XS]  = {256, 74},
    [GGML_TYPE_IQ3_XXS] = {256, 98},
    [GGML_TYPE_IQ1_S]   = {256, 50},
    [GGML_TYPE_IQ4_NL]  = {32, 18},
    [GGML_TYPE_IQ3_S]   = {256, 110},
    [GGML_TYPE_IQ2_S]   = {256, 82},
    [GGML_TYPE_IQ4_XS]  = {256, 136},
    [GGML_TYPE_I8]      = {1, 1},
    [GGML_TYPE_I16]     = {1, 2},
    [GGML_TYPE_I32]     = {1, 4},
    [GGML_TYPE_I64]     = {1, 8},
    [GGML_TYPE_F64]     = {1, 8},
    [GGML_TYPE_IQ1_M]   = {256, 56},
    [GGML_TYPE_BF16]    = {1, 2},
    [GGML_TYPE_Q4_0_4_4] = {32, 18},
    [GGML_TYPE_Q4_0_4_8] = {32, 18},
    [GGML_TYPE_Q4_0_8_8] = {32, 18},
    [GGML_TYPE_TQ1_0]   = {256, 54},
    [GGML_TYPE_TQ2_0]   = {256, 66},
};

const char *gguf_type_name(uint32_t type) {
    static const char *names[] = {
        "uint8","int8","uint16","int16","uint32","int32","float32",
        "bool","string","array","uint64","int64","float64"
    };
    if (type < GGUF_TYPE_COUNT) return names[type];
    return "unknown";
}

const char *ggml_type_name(uint32_t type) {
    static const char *names[] = {
        "F32","F16","Q4_0","Q4_1","???","???","Q5_0","Q5_1",
        "Q8_0","Q8_1","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","Q8_K",
        "IQ2_XXS","IQ2_XS","IQ3_XXS","IQ1_S","IQ4_NL","IQ3_S","IQ2_S","IQ4_XS",
        "I8","I16","I32","I64","F64","IQ1_M","BF16",
        "Q4_0_4_4","Q4_0_4_8","Q4_0_8_8","TQ1_0","TQ2_0"
    };
    if (type < GGML_TYPE_COUNT) return names[type];
    return "unknown";
}

static int gguf_read_str(FILE *f, gguf_str *s) {
    if (fread(&s->len, 8, 1, f) != 1) return -1;
    s->str = (char *)malloc(s->len + 1);
    if (!s->str) return -1;
    if (s->len > 0 && fread(s->str, 1, s->len, f) != s->len) {
        free(s->str); s->str = NULL; return -1;
    }
    s->str[s->len] = '\0';
    return 0;
}

static size_t gguf_value_type_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static int gguf_read_value(FILE *f, gguf_kv *kv, uint32_t type) {
    kv->type = type;
    switch (type) {
        case GGUF_TYPE_UINT8:   return fread(&kv->value.u8,  1, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_INT8:    return fread(&kv->value.i8,  1, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_UINT16:  return fread(&kv->value.u16, 2, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_INT16:   return fread(&kv->value.i16, 2, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_UINT32:  return fread(&kv->value.u32, 4, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_INT32:   return fread(&kv->value.i32, 4, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_FLOAT32: return fread(&kv->value.f32, 4, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_BOOL:    return fread(&kv->value.b,   1, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_UINT64:  return fread(&kv->value.u64, 8, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_INT64:   return fread(&kv->value.i64, 8, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_FLOAT64: return fread(&kv->value.f64, 8, 1, f) == 1 ? 0 : -1;
        case GGUF_TYPE_STRING:  return gguf_read_str(f, &kv->value.str);
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type;
            uint64_t n;
            if (fread(&elem_type, 4, 1, f) != 1) return -1;
            if (fread(&n, 8, 1, f) != 1) return -1;
            kv->value.arr.type = elem_type;
            kv->value.arr.n = n;
            if (elem_type == GGUF_TYPE_STRING) {
                gguf_str *strs = (gguf_str *)calloc(n, sizeof(gguf_str));
                if (!strs) return -1;
                for (uint64_t i = 0; i < n; i++) {
                    if (gguf_read_str(f, &strs[i]) != 0) {
                        for (uint64_t j = 0; j < i; j++) free(strs[j].str);
                        free(strs);
                        return -1;
                    }
                }
                kv->value.arr.data = strs;
            } else {
                size_t elem_sz = gguf_value_type_size(elem_type);
                if (elem_sz == 0) return -1;
                void *buf = malloc(elem_sz * n);
                if (!buf) return -1;
                if (fread(buf, elem_sz, n, f) != n) { free(buf); return -1; }
                kv->value.arr.data = buf;
            }
            return 0;
        }
        default: return -1;
    }
}

static void gguf_free_kv(gguf_kv *kv) {
    free(kv->key.str);
    if (kv->type == GGUF_TYPE_STRING) {
        free(kv->value.str.str);
    } else if (kv->type == GGUF_TYPE_ARRAY) {
        if (kv->value.arr.type == GGUF_TYPE_STRING) {
            gguf_str *strs = (gguf_str *)kv->value.arr.data;
            for (uint64_t i = 0; i < kv->value.arr.n; i++) free(strs[i].str);
        }
        free(kv->value.arr.data);
    }
}

gguf_context *gguf_open(const char *path, int use_mmap) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gguf: cannot open %s\n", path); return NULL; }

    /* magic */
    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747u) { /* "GGUF" LE */
        fprintf(stderr, "gguf: bad magic\n"); fclose(f); return NULL;
    }

    uint32_t version;
    if (fread(&version, 4, 1, f) != 1 || (version != 2 && version != 3)) {
        fprintf(stderr, "gguf: unsupported version %u\n", version); fclose(f); return NULL;
    }

    uint64_t n_tensors, n_kv;
    if (fread(&n_tensors, 8, 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&n_kv, 8, 1, f) != 1) { fclose(f); return NULL; }

    gguf_context *ctx = (gguf_context *)calloc(1, sizeof(gguf_context));
    if (!ctx) { fclose(f); return NULL; }
    ctx->version = version;
    ctx->n_kv = n_kv;
    ctx->n_tensors = n_tensors;
    ctx->alignment = 32;
    ctx->use_mmap = use_mmap;

    /* parse KV pairs */
    ctx->kv = (gguf_kv *)calloc(n_kv, sizeof(gguf_kv));
    if (!ctx->kv && n_kv > 0) goto fail;
    for (uint64_t i = 0; i < n_kv; i++) {
        if (gguf_read_str(f, &ctx->kv[i].key) != 0) goto fail;
        uint32_t vtype;
        if (fread(&vtype, 4, 1, f) != 1) goto fail;
        if (gguf_read_value(f, &ctx->kv[i], vtype) != 0) goto fail;
    }

    /* check for custom alignment */
    int align_idx = gguf_find_key(ctx, "general.alignment");
    if (align_idx >= 0 && ctx->kv[align_idx].type == GGUF_TYPE_UINT32) {
        ctx->alignment = ctx->kv[align_idx].value.u32;
    }

    /* parse tensor infos */
    ctx->tensors = (gguf_tensor_info *)calloc(n_tensors, sizeof(gguf_tensor_info));
    if (!ctx->tensors && n_tensors > 0) goto fail;
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info *ti = &ctx->tensors[i];
        if (gguf_read_str(f, &ti->name) != 0) goto fail;
        uint32_t n_dims;
        if (fread(&n_dims, 4, 1, f) != 1) goto fail;
        ti->n_dims = n_dims;
        memset(ti->dims, 0, sizeof(ti->dims));
        for (uint32_t d = 0; d < n_dims; d++) {
            if (fread(&ti->dims[d], 8, 1, f) != 1) goto fail;
        }
        if (fread(&ti->type, 4, 1, f) != 1) goto fail;
        if (fread(&ti->offset, 8, 1, f) != 1) goto fail;
    }

    /* compute data offset: align current file position */
    {
        long pos = ftell(f);
        uint32_t a = ctx->alignment;
        size_t aligned = ((size_t)pos + a - 1) / a * a;
        ctx->data_offset = aligned;
    }

    /* compute total data size */
    {
        size_t max_end = 0;
        for (uint64_t i = 0; i < n_tensors; i++) {
            size_t sz = gguf_tensor_size(ctx, (int)i);
            size_t end = ctx->tensors[i].offset + sz;
            if (end > max_end) max_end = end;
        }
        ctx->data_size = max_end;
    }

    /* load data */
    if (use_mmap) {
#ifdef _WIN32
        fclose(f); f = NULL;
        ctx->file_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (ctx->file_handle == INVALID_HANDLE_VALUE) goto fail;
        LARGE_INTEGER file_size;
        GetFileSizeEx(ctx->file_handle, &file_size);
        ctx->map_size = (size_t)file_size.QuadPart;
        ctx->map_handle = CreateFileMappingA(ctx->file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!ctx->map_handle) goto fail;
        ctx->map_base = MapViewOfFile(ctx->map_handle, FILE_MAP_READ, 0, 0, 0);
        if (!ctx->map_base) goto fail;
        ctx->data = (uint8_t *)ctx->map_base + ctx->data_offset;
#else
        fclose(f); f = NULL;
        ctx->fd = open(path, O_RDONLY);
        if (ctx->fd < 0) goto fail;
        struct stat st;
        if (fstat(ctx->fd, &st) != 0) goto fail;
        ctx->map_size = (size_t)st.st_size;
        ctx->map_base = mmap(NULL, ctx->map_size, PROT_READ, MAP_PRIVATE, ctx->fd, 0);
        if (ctx->map_base == MAP_FAILED) { ctx->map_base = NULL; goto fail; }
        ctx->data = (uint8_t *)ctx->map_base + ctx->data_offset;
#endif
    } else {
        ctx->data = (uint8_t *)malloc(ctx->data_size);
        if (!ctx->data) goto fail;
        fseek(f, (long)ctx->data_offset, SEEK_SET);
        if (fread(ctx->data, 1, ctx->data_size, f) != ctx->data_size) goto fail;
        fclose(f); f = NULL;
    }

    if (f) fclose(f);
    return ctx;

fail:
    if (f) fclose(f);
    gguf_close(ctx);
    return NULL;
}

void gguf_close(gguf_context *ctx) {
    if (!ctx) return;
    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->n_kv; i++) gguf_free_kv(&ctx->kv[i]);
        free(ctx->kv);
    }
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) free(ctx->tensors[i].name.str);
        free(ctx->tensors);
    }
    if (ctx->use_mmap) {
#ifdef _WIN32
        if (ctx->map_base) UnmapViewOfFile(ctx->map_base);
        if (ctx->map_handle) CloseHandle(ctx->map_handle);
        if (ctx->file_handle && ctx->file_handle != INVALID_HANDLE_VALUE) CloseHandle(ctx->file_handle);
#else
        if (ctx->map_base) munmap(ctx->map_base, ctx->map_size);
        if (ctx->fd > 0) close(ctx->fd);
#endif
    } else {
        free(ctx->data);
    }
    free(ctx);
}

int gguf_find_key(const gguf_context *ctx, const char *key) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (ctx->kv[i].key.str && strcmp(ctx->kv[i].key.str, key) == 0)
            return (int)i;
    }
    return -1;
}

const char *gguf_tensor_name(const gguf_context *ctx, int i) {
    if (i < 0 || (uint64_t)i >= ctx->n_tensors) return NULL;
    return ctx->tensors[i].name.str;
}

void *gguf_tensor_data(const gguf_context *ctx, int i) {
    if (i < 0 || (uint64_t)i >= ctx->n_tensors || !ctx->data) return NULL;
    return ctx->data + ctx->tensors[i].offset;
}

size_t gguf_tensor_size(const gguf_context *ctx, int i) {
    if (i < 0 || (uint64_t)i >= ctx->n_tensors) return 0;
    const gguf_tensor_info *ti = &ctx->tensors[i];
    uint64_t n_elements = 1;
    for (uint32_t d = 0; d < ti->n_dims; d++) n_elements *= ti->dims[d];
    if (ti->type < GGML_TYPE_COUNT) {
        int bs = ggml_type_info[ti->type].block_size;
        int ts = ggml_type_info[ti->type].type_size;
        if (bs == 0) return 0;
        return (size_t)((n_elements + bs - 1) / bs) * ts;
    }
    return 0;
}

#endif /* GGUF_LOADER_IMPLEMENTATION */
#endif /* GGUF_LOADER_H */
