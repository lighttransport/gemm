/*
 * pth_loader.h - Single-file PyTorch .pth/.pt checkpoint loader
 *
 * Loads PyTorch state_dict checkpoints saved with torch.save().
 * Supports ZIP-based format (PyTorch 1.6+, uncompressed stores only).
 * Handles nested dicts (e.g., {'model': state_dict}).
 *
 * Usage:
 *   #define PTH_LOADER_IMPLEMENTATION
 *   #include "pth_loader.h"
 *
 * API:
 *   pth_context    *pth_open(const char *path);
 *   void            pth_close(pth_context *ctx);
 *   int             pth_find(const pth_context *ctx, const char *name);
 *   int             pth_count(const pth_context *ctx);
 *   const char     *pth_name(const pth_context *ctx, int i);
 *   const char     *pth_dtype(const pth_context *ctx, int i);
 *   int             pth_ndims(const pth_context *ctx, int i);
 *   const uint64_t *pth_shape(const pth_context *ctx, int i);
 *   void           *pth_data(const pth_context *ctx, int i);
 *   size_t          pth_nbytes(const pth_context *ctx, int i);
 *   size_t          pth_dtype_size(const char *dtype_str);
 */
#ifndef PTH_LOADER_H
#define PTH_LOADER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PTH_MAX_DIMS 8

typedef struct {
    char       *name;
    char        dtype_str[8];      /* "F32", "F16", "BF16" */
    uint64_t    shape[PTH_MAX_DIMS];
    int         n_dims;
    void       *data;              /* pointer into mmap'd ZIP */
    size_t      nbytes;
} pth_tensor_info;

typedef struct {
    pth_tensor_info *tensors;
    int              n_tensors;
    void            *map_base;     /* mmap'd whole file */
    size_t           map_size;
} pth_context;

pth_context    *pth_open(const char *path);
void            pth_close(pth_context *ctx);
int             pth_find(const pth_context *ctx, const char *name);
int             pth_count(const pth_context *ctx);
const char     *pth_name(const pth_context *ctx, int i);
const char     *pth_dtype(const pth_context *ctx, int i);
int             pth_ndims(const pth_context *ctx, int i);
const uint64_t *pth_shape(const pth_context *ctx, int i);
void           *pth_data(const pth_context *ctx, int i);
size_t          pth_nbytes(const pth_context *ctx, int i);
size_t          pth_dtype_size(const char *dtype_str);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef PTH_LOADER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ---- Minimal ZIP reader ---- */

/* ZIP End-of-Central-Directory record (fixed part = 22 bytes) */
#define ZIP_EOCD_SIG   0x06054b50
#define ZIP_CD_SIG     0x02014b50
#define ZIP_LF_SIG     0x04034b50

static uint16_t zip_u16(const uint8_t *p) { return p[0] | (p[1] << 8); }
static uint32_t zip_u32(const uint8_t *p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
}

typedef struct {
    char    *name;
    size_t   data_offset;   /* offset of actual file data in the archive */
    size_t   comp_size;     /* compressed size (must equal uncomp for store) */
    size_t   uncomp_size;
    uint16_t method;        /* 0 = store */
} zip_entry;

typedef struct {
    zip_entry *entries;
    int        n_entries;
} zip_dir;

/* Find EOCD record by scanning backwards from end of file */
static const uint8_t *zip__find_eocd(const uint8_t *data, size_t size) {
    if (size < 22) return NULL;
    /* EOCD can have a variable-length comment, scan up to 65557 bytes back */
    size_t scan = size < 65557 ? size : 65557;
    for (size_t i = 22; i <= scan; i++) {
        const uint8_t *p = data + size - i;
        if (zip_u32(p) == ZIP_EOCD_SIG) return p;
    }
    return NULL;
}

/* Parse central directory into zip_dir. Returns 0 on success. */
static int zip__parse(const uint8_t *data, size_t size, zip_dir *dir) {
    const uint8_t *eocd = zip__find_eocd(data, size);
    if (!eocd) {
        fprintf(stderr, "pth: cannot find ZIP EOCD record\n");
        return -1;
    }

    uint16_t n_entries   = zip_u16(eocd + 10);
    uint32_t cd_offset   = zip_u32(eocd + 16);

    if (cd_offset >= size) {
        fprintf(stderr, "pth: invalid CD offset %u\n", cd_offset);
        return -1;
    }

    dir->entries = (zip_entry *)calloc(n_entries, sizeof(zip_entry));
    dir->n_entries = n_entries;

    const uint8_t *p = data + cd_offset;
    for (int i = 0; i < n_entries; i++) {
        if (p + 46 > data + size) goto fail;
        if (zip_u32(p) != ZIP_CD_SIG) {
            fprintf(stderr, "pth: bad CD signature at entry %d\n", i);
            goto fail;
        }

        uint16_t name_len  = zip_u16(p + 28);
        uint16_t extra_len = zip_u16(p + 30);
        uint16_t comm_len  = zip_u16(p + 32);
        uint16_t method    = zip_u16(p + 10);
        uint32_t comp_sz   = zip_u32(p + 20);
        uint32_t uncomp_sz = zip_u32(p + 24);
        uint32_t lf_offset = zip_u32(p + 42);

        if (p + 46 + name_len > data + size) goto fail;

        dir->entries[i].name = (char *)malloc(name_len + 1);
        memcpy(dir->entries[i].name, p + 46, name_len);
        dir->entries[i].name[name_len] = '\0';
        dir->entries[i].method = method;
        dir->entries[i].comp_size = comp_sz;
        dir->entries[i].uncomp_size = uncomp_sz;

        /* Compute actual data offset from local file header */
        if (lf_offset + 30 > size) goto fail;
        const uint8_t *lf = data + lf_offset;
        if (zip_u32(lf) != ZIP_LF_SIG) goto fail;
        uint16_t lf_name_len  = zip_u16(lf + 26);
        uint16_t lf_extra_len = zip_u16(lf + 28);
        dir->entries[i].data_offset = lf_offset + 30 + lf_name_len + lf_extra_len;

        p += 46 + name_len + extra_len + comm_len;
    }
    return 0;

fail:
    for (int j = 0; j < n_entries; j++) free(dir->entries[j].name);
    free(dir->entries);
    dir->entries = NULL;
    dir->n_entries = 0;
    return -1;
}

static void zip__free(zip_dir *dir) {
    for (int i = 0; i < dir->n_entries; i++) free(dir->entries[i].name);
    free(dir->entries);
    dir->entries = NULL;
    dir->n_entries = 0;
}

/* Find a ZIP entry by suffix (e.g., "data.pkl", "data/0") */
static int zip__find_suffix(const zip_dir *dir, const char *suffix) {
    int slen = (int)strlen(suffix);
    for (int i = 0; i < dir->n_entries; i++) {
        int nlen = (int)strlen(dir->entries[i].name);
        if (nlen >= slen && strcmp(dir->entries[i].name + nlen - slen, suffix) == 0)
            return i;
    }
    return -1;
}

/* ---- Minimal Pickle parser ---- */

/* Pickle opcodes we need to handle */
#define PK_PROTO          0x80
#define PK_FRAME          0x95
#define PK_STOP           0x2e  /* '.' */
#define PK_EMPTY_DICT     0x7d  /* '}' */
#define PK_MARK           0x28  /* '(' */
#define PK_SETITEMS       0x75  /* 'u' */
#define PK_SETITEM        0x73  /* 's' */
#define PK_SHORT_BINUNICODE 0x8c
#define PK_BINUNICODE     0x58  /* 'X' */
#define PK_GLOBAL         0x63  /* 'c' */
#define PK_STACK_GLOBAL   0x93
#define PK_REDUCE         0x52  /* 'R' */
#define PK_BUILD          0x62  /* 'b' */
/* TUPLE (0x85) pops mark..TOS into a tuple; same opcode as "TUPLE1" in protocol<2.
 * In protocol>=2, 0x85=TUPLE1, 0x86=TUPLE2, 0x87=TUPLE3.
 * The mark-based TUPLE uses opcode 0x85 only in old protocols; in practice
 * protocol>=2 uses MARK + TUPLE (opcode 't' = 0x74) for variable-length tuples. */
#define PK_TUPLE_VAR      0x74  /* 't' - pop mark..TOS into tuple (protocol 0-1) */
#define PK_TUPLE1         0x85
#define PK_TUPLE2         0x86
#define PK_TUPLE3         0x87
#define PK_EMPTY_TUPLE    0x29  /* ')' */
#define PK_BINPUT         0x71  /* 'q' */
#define PK_LONG_BINPUT    0x72  /* 'r' */
#define PK_BINGET         0x68  /* 'h' */
#define PK_LONG_BINGET    0x6a  /* 'j' */
#define PK_BININT         0x4a  /* 'J' */
#define PK_BININT1        0x4b  /* 'K' */
#define PK_BININT2        0x4d  /* 'M' */
#define PK_LONG1          0x8a
#define PK_BINFLOAT       0x47  /* 'G' */
#define PK_NEWTRUE        0x88
#define PK_NEWFALSE       0x89
#define PK_NONE           0x4e  /* 'N' */
#define PK_BINPERSID      0x51  /* 'Q' */
#define PK_EMPTY_LIST     0x5d  /* ']' */
#define PK_APPENDS        0x65  /* 'e' */
#define PK_APPEND         0x61  /* 'a' */
#define PK_SHORT_BINSTRING 0x55 /* 'U' */
#define PK_BINSTRING      0x54  /* 'T' */

/* Pickle value types */
typedef enum {
    PV_NONE, PV_INT, PV_FLOAT, PV_STRING, PV_BOOL,
    PV_TUPLE, PV_LIST, PV_DICT, PV_STORAGE_REF,
    PV_GLOBAL, PV_MARK, PV_REBUILD_TENSOR, PV_ORDERED_DICT
} pv_type;

/* Storage reference from BINPERSID */
typedef struct {
    char dtype_str[8];   /* "F32", "F16", "BF16" */
    char storage_key[64]; /* e.g., "0", "1", ... */
    int64_t numel;
    int dtype_size;
} pk_storage_ref;

/* Tensor info extracted from _rebuild_tensor_v2 */
typedef struct {
    pk_storage_ref storage;
    int64_t storage_offset;
    int64_t shape[PTH_MAX_DIMS];
    int64_t stride[PTH_MAX_DIMS];
    int n_dims;
} pk_tensor_rebuild;

/* Pickle value (tagged union) */
#define PV_TUPLE_MAX 8

typedef struct pk_val {
    pv_type type;
    union {
        int64_t i;
        double f;
        int b;
        struct { char *ptr; int len; } str;
        struct { char module[128]; char name[128]; } global;
        pk_storage_ref storage;
        pk_tensor_rebuild tensor;
        struct { struct pk_val *items; int count; int cap; } tuple; /* also list */
        struct {
            char **keys;
            struct pk_val *vals;
            int count;
            int cap;
        } dict;
    };
} pk_val;

/* Pickle parser state */
#define PK_STACK_MAX  2048
#define PK_MEMO_MAX   4096

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t pos;
    pk_val stack[PK_STACK_MAX];
    int sp;
    pk_val memo[PK_MEMO_MAX];
    int memo_used[PK_MEMO_MAX]; /* 0 = unused */
} pk_state;

static void pk_val_free_contents(pk_val *v) {
    switch (v->type) {
    case PV_STRING:
        free(v->str.ptr);
        v->str.ptr = NULL;
        break;
    case PV_TUPLE:
    case PV_LIST:
        for (int i = 0; i < v->tuple.count; i++)
            pk_val_free_contents(&v->tuple.items[i]);
        free(v->tuple.items);
        v->tuple.items = NULL;
        break;
    case PV_DICT:
    case PV_ORDERED_DICT:
        for (int i = 0; i < v->dict.count; i++) {
            free(v->dict.keys[i]);
            pk_val_free_contents(&v->dict.vals[i]);
        }
        free(v->dict.keys);
        free(v->dict.vals);
        v->dict.keys = NULL;
        v->dict.vals = NULL;
        break;
    default:
        break;
    }
}

static pk_val pk_val_none(void) { pk_val v; memset(&v, 0, sizeof(v)); v.type = PV_NONE; return v; }
static pk_val pk_val_int(int64_t i) { pk_val v; memset(&v, 0, sizeof(v)); v.type = PV_INT; v.i = i; return v; }
static pk_val pk_val_float(double f) { pk_val v; memset(&v, 0, sizeof(v)); v.type = PV_FLOAT; v.f = f; return v; }
static pk_val pk_val_bool(int b) { pk_val v; memset(&v, 0, sizeof(v)); v.type = PV_BOOL; v.b = b; return v; }
static pk_val pk_val_mark(void) { pk_val v; memset(&v, 0, sizeof(v)); v.type = PV_MARK; return v; }

static pk_val pk_val_string(const char *s, int len) {
    pk_val v;
    memset(&v, 0, sizeof(v));
    v.type = PV_STRING;
    v.str.ptr = (char *)malloc(len + 1);
    memcpy(v.str.ptr, s, len);
    v.str.ptr[len] = '\0';
    v.str.len = len;
    return v;
}

static pk_val pk_val_copy(const pk_val *src) {
    pk_val v = *src;
    switch (src->type) {
    case PV_STRING:
        v.str.ptr = (char *)malloc(src->str.len + 1);
        memcpy(v.str.ptr, src->str.ptr, src->str.len + 1);
        break;
    case PV_TUPLE:
    case PV_LIST:
        if (src->tuple.count > 0) {
            v.tuple.items = (pk_val *)malloc(src->tuple.count * sizeof(pk_val));
            v.tuple.cap = src->tuple.count;
            for (int i = 0; i < src->tuple.count; i++)
                v.tuple.items[i] = pk_val_copy(&src->tuple.items[i]);
        }
        break;
    case PV_DICT:
    case PV_ORDERED_DICT:
        if (src->dict.count > 0) {
            v.dict.keys = (char **)malloc(src->dict.count * sizeof(char *));
            v.dict.vals = (pk_val *)malloc(src->dict.count * sizeof(pk_val));
            v.dict.cap = src->dict.count;
            for (int i = 0; i < src->dict.count; i++) {
                v.dict.keys[i] = strdup(src->dict.keys[i]);
                v.dict.vals[i] = pk_val_copy(&src->dict.vals[i]);
            }
        }
        break;
    default:
        break;
    }
    return v;
}

static void pk__push(pk_state *s, pk_val v) {
    if (s->sp >= PK_STACK_MAX) {
        fprintf(stderr, "pth: pickle stack overflow\n");
        return;
    }
    s->stack[s->sp++] = v;
}

static pk_val pk__pop(pk_state *s) {
    if (s->sp <= 0) {
        fprintf(stderr, "pth: pickle stack underflow\n");
        return pk_val_none();
    }
    return s->stack[--s->sp];
}

static pk_val *pk__top(pk_state *s) {
    if (s->sp <= 0) return NULL;
    return &s->stack[s->sp - 1];
}

static uint8_t pk__u8(pk_state *s) {
    if (s->pos >= s->size) return 0;
    return s->data[s->pos++];
}

static uint16_t pk__u16(pk_state *s) {
    if (s->pos + 2 > s->size) return 0;
    uint16_t v = s->data[s->pos] | (s->data[s->pos + 1] << 8);
    s->pos += 2;
    return v;
}

static uint32_t pk__u32(pk_state *s) {
    if (s->pos + 4 > s->size) return 0;
    uint32_t v = s->data[s->pos] | (s->data[s->pos + 1] << 8) |
                 (s->data[s->pos + 2] << 16) | ((uint32_t)s->data[s->pos + 3] << 24);
    s->pos += 4;
    return v;
}

static int64_t pk__i32(pk_state *s) {
    return (int32_t)pk__u32(s);
}

static double pk__f64_be(pk_state *s) {
    if (s->pos + 8 > s->size) return 0.0;
    /* BINFLOAT is big-endian IEEE 754 */
    uint8_t buf[8];
    for (int i = 0; i < 8; i++) buf[7 - i] = s->data[s->pos + i];
    s->pos += 8;
    double v;
    memcpy(&v, buf, 8);
    return v;
}

static uint64_t pk__u64(pk_state *s) {
    if (s->pos + 8 > s->size) return 0;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)s->data[s->pos + i]) << (i * 8);
    s->pos += 8;
    return v;
}

/* Read a newline-terminated string from pickle stream */
static int pk__readline(pk_state *s, char *buf, int bufsize) {
    int n = 0;
    while (s->pos < s->size && n < bufsize - 1) {
        char c = (char)s->data[s->pos++];
        if (c == '\n') break;
        buf[n++] = c;
    }
    buf[n] = '\0';
    return n;
}

/* Map PyTorch storage type string to dtype */
static void pk__storage_dtype(const char *type_str, char *out_dtype, int *out_size) {
    static const struct { const char *name; const char *dtype; int size; } tbl[] = {
        {"FloatStorage",    "F32",  4},
        {"DoubleStorage",   "F64",  8},
        {"HalfStorage",     "F16",  2},
        {"BFloat16Storage", "BF16", 2},
        {"ByteStorage",     "U8",   1},
        {"CharStorage",     "I8",   1},
        {"ShortStorage",    "I16",  2},
        {"IntStorage",      "I32",  4},
        {"LongStorage",     "I64",  8},
        {"BoolStorage",     "BOOL", 1},
    };
    for (int i = 0; i < (int)(sizeof(tbl) / sizeof(tbl[0])); i++) {
        if (strstr(type_str, tbl[i].name)) {
            strcpy(out_dtype, tbl[i].dtype);
            *out_size = tbl[i].size;
            return;
        }
    }
    /* New-style dtype enum (PyTorch 2.0+) */
    if (strstr(type_str, "float32") || strstr(type_str, "torch.float32")) {
        strcpy(out_dtype, "F32"); *out_size = 4;
    } else if (strstr(type_str, "float16") || strstr(type_str, "torch.float16")) {
        strcpy(out_dtype, "F16"); *out_size = 2;
    } else if (strstr(type_str, "bfloat16") || strstr(type_str, "torch.bfloat16")) {
        strcpy(out_dtype, "BF16"); *out_size = 2;
    } else if (strstr(type_str, "float64") || strstr(type_str, "torch.float64")) {
        strcpy(out_dtype, "F64"); *out_size = 8;
    } else {
        strcpy(out_dtype, "F32"); *out_size = 4;
        fprintf(stderr, "pth: unknown storage type '%s', defaulting to F32\n", type_str);
    }
}

/* Find mark position on stack */
static int pk__find_mark(pk_state *s) {
    for (int i = s->sp - 1; i >= 0; i--) {
        if (s->stack[i].type == PV_MARK) return i;
    }
    return -1;
}

/* Parse pickle stream. Returns top-of-stack value. */
static pk_val pk__parse(pk_state *s) {
    while (s->pos < s->size) {
        uint8_t op = pk__u8(s);

        switch (op) {
        case PK_PROTO: {
            pk__u8(s); /* protocol version, ignore */
            break;
        }
        case PK_FRAME: {
            pk__u64(s); /* frame size, ignore */
            break;
        }
        case PK_STOP: {
            if (s->sp > 0) return s->stack[s->sp - 1];
            return pk_val_none();
        }
        case PK_EMPTY_DICT: {
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_DICT;
            pk__push(s, v);
            break;
        }
        case PK_MARK: {
            pk__push(s, pk_val_mark());
            break;
        }
        case PK_SETITEMS: {
            /* Pop key-value pairs until mark, set on dict below mark */
            int mark = pk__find_mark(s);
            if (mark < 0 || mark < 1) break;
            int n_pairs = (s->sp - mark - 1) / 2;
            pk_val *dict = &s->stack[mark - 1];
            if (dict->type != PV_DICT && dict->type != PV_ORDERED_DICT) {
                /* Pop everything and continue */
                s->sp = mark - 1;
                pk__push(s, *dict);
                break;
            }
            for (int i = 0; i < n_pairs; i++) {
                pk_val *key = &s->stack[mark + 1 + i * 2];
                pk_val *val = &s->stack[mark + 1 + i * 2 + 1];
                if (key->type == PV_STRING) {
                    if (dict->dict.count >= dict->dict.cap) {
                        dict->dict.cap = dict->dict.cap ? dict->dict.cap * 2 : 32;
                        dict->dict.keys = (char **)realloc(dict->dict.keys,
                            dict->dict.cap * sizeof(char *));
                        dict->dict.vals = (pk_val *)realloc(dict->dict.vals,
                            dict->dict.cap * sizeof(pk_val));
                    }
                    dict->dict.keys[dict->dict.count] = key->str.ptr;
                    key->str.ptr = NULL; /* transfer ownership */
                    dict->dict.vals[dict->dict.count] = *val;
                    /* Zero out val to prevent double-free */
                    memset(val, 0, sizeof(pk_val));
                    dict->dict.count++;
                } else {
                    pk_val_free_contents(key);
                    pk_val_free_contents(val);
                }
            }
            s->sp = mark; /* remove mark */
            break;
        }
        case PK_SETITEM: {
            if (s->sp < 3) break;
            pk_val val = pk__pop(s);
            pk_val key = pk__pop(s);
            pk_val *dict = pk__top(s);
            if (dict && (dict->type == PV_DICT || dict->type == PV_ORDERED_DICT)
                && key.type == PV_STRING) {
                if (dict->dict.count >= dict->dict.cap) {
                    dict->dict.cap = dict->dict.cap ? dict->dict.cap * 2 : 32;
                    dict->dict.keys = (char **)realloc(dict->dict.keys,
                        dict->dict.cap * sizeof(char *));
                    dict->dict.vals = (pk_val *)realloc(dict->dict.vals,
                        dict->dict.cap * sizeof(pk_val));
                }
                dict->dict.keys[dict->dict.count] = key.str.ptr;
                key.str.ptr = NULL;
                dict->dict.vals[dict->dict.count] = val;
                dict->dict.count++;
            } else {
                pk_val_free_contents(&key);
                pk_val_free_contents(&val);
            }
            break;
        }
        case PK_SHORT_BINUNICODE: {
            uint8_t len = pk__u8(s);
            if (s->pos + len > s->size) break;
            pk__push(s, pk_val_string((const char *)s->data + s->pos, len));
            s->pos += len;
            break;
        }
        case PK_BINUNICODE: {
            uint32_t len = pk__u32(s);
            if (s->pos + len > s->size) break;
            pk__push(s, pk_val_string((const char *)s->data + s->pos, len));
            s->pos += len;
            break;
        }
        case PK_SHORT_BINSTRING: {
            uint8_t len = pk__u8(s);
            if (s->pos + len > s->size) break;
            pk__push(s, pk_val_string((const char *)s->data + s->pos, len));
            s->pos += len;
            break;
        }
        case PK_BINSTRING: {
            uint32_t len = pk__u32(s);
            if (s->pos + len > s->size) break;
            pk__push(s, pk_val_string((const char *)s->data + s->pos, len));
            s->pos += len;
            break;
        }
        case PK_GLOBAL: {
            /* Read module\nname\n */
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_GLOBAL;
            pk__readline(s, v.global.module, sizeof(v.global.module));
            pk__readline(s, v.global.name, sizeof(v.global.name));
            pk__push(s, v);
            break;
        }
        case PK_STACK_GLOBAL: {
            /* Pop name, pop module */
            pk_val name = pk__pop(s);
            pk_val module = pk__pop(s);
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_GLOBAL;
            if (module.type == PV_STRING) {
                int ml = module.str.len < 127 ? module.str.len : 127;
                memcpy(v.global.module, module.str.ptr, ml);
                v.global.module[ml] = '\0';
            }
            if (name.type == PV_STRING) {
                int nl = name.str.len < 127 ? name.str.len : 127;
                memcpy(v.global.name, name.str.ptr, nl);
                v.global.name[nl] = '\0';
            }
            pk_val_free_contents(&name);
            pk_val_free_contents(&module);
            pk__push(s, v);
            break;
        }
        case PK_REDUCE: {
            /* Pop args tuple, pop callable, call callable(args) */
            pk_val args = pk__pop(s);
            pk_val callable = pk__pop(s);

            if (callable.type == PV_GLOBAL) {
                if (strcmp(callable.global.name, "OrderedDict") == 0) {
                    /* collections.OrderedDict() */
                    pk_val v;
                    memset(&v, 0, sizeof(v));
                    v.type = PV_ORDERED_DICT;
                    pk_val_free_contents(&args);
                    pk__push(s, v);
                } else if (strcmp(callable.global.name, "_rebuild_tensor_v2") == 0 ||
                           strcmp(callable.global.name, "_rebuild_tensor_v3") == 0) {
                    /* _rebuild_tensor_v2(storage, storage_offset, size, stride, ...) */
                    pk_val v;
                    memset(&v, 0, sizeof(v));
                    v.type = PV_REBUILD_TENSOR;

                    if (args.type == PV_TUPLE && args.tuple.count >= 4) {
                        /* args[0] = storage ref */
                        if (args.tuple.items[0].type == PV_STORAGE_REF) {
                            v.tensor.storage = args.tuple.items[0].storage;
                        }
                        /* args[1] = storage offset */
                        if (args.tuple.items[1].type == PV_INT) {
                            v.tensor.storage_offset = args.tuple.items[1].i;
                        }
                        /* args[2] = size tuple */
                        if (args.tuple.items[2].type == PV_TUPLE) {
                            pk_val *sz = &args.tuple.items[2];
                            v.tensor.n_dims = sz->tuple.count < PTH_MAX_DIMS ?
                                              sz->tuple.count : PTH_MAX_DIMS;
                            for (int d = 0; d < v.tensor.n_dims; d++) {
                                if (sz->tuple.items[d].type == PV_INT)
                                    v.tensor.shape[d] = sz->tuple.items[d].i;
                            }
                        }
                        /* args[3] = stride tuple */
                        if (args.tuple.items[3].type == PV_TUPLE) {
                            pk_val *st = &args.tuple.items[3];
                            for (int d = 0; d < v.tensor.n_dims && d < st->tuple.count; d++) {
                                if (st->tuple.items[d].type == PV_INT)
                                    v.tensor.stride[d] = st->tuple.items[d].i;
                            }
                        }
                    }
                    pk_val_free_contents(&args);
                    pk__push(s, v);
                } else {
                    /* Unknown callable, push None */
                    pk_val_free_contents(&args);
                    pk__push(s, pk_val_none());
                }
            } else {
                pk_val_free_contents(&callable);
                pk_val_free_contents(&args);
                pk__push(s, pk_val_none());
            }
            break;
        }
        case PK_BUILD: {
            /* Pop state, pop obj, obj.__setstate__(state) — ignore for our purposes */
            pk_val state = pk__pop(s);
            /* If top is an OrderedDict and state is a dict, merge items */
            pk_val *obj = pk__top(s);
            if (obj && (obj->type == PV_ORDERED_DICT || obj->type == PV_DICT)
                && (state.type == PV_DICT || state.type == PV_ORDERED_DICT)) {
                /* Transfer items from state to obj */
                for (int i = 0; i < state.dict.count; i++) {
                    if (obj->dict.count >= obj->dict.cap) {
                        obj->dict.cap = obj->dict.cap ? obj->dict.cap * 2 : 32;
                        obj->dict.keys = (char **)realloc(obj->dict.keys,
                            obj->dict.cap * sizeof(char *));
                        obj->dict.vals = (pk_val *)realloc(obj->dict.vals,
                            obj->dict.cap * sizeof(pk_val));
                    }
                    obj->dict.keys[obj->dict.count] = state.dict.keys[i];
                    state.dict.keys[i] = NULL;
                    obj->dict.vals[obj->dict.count] = state.dict.vals[i];
                    memset(&state.dict.vals[i], 0, sizeof(pk_val));
                    obj->dict.count++;
                }
                free(state.dict.keys);
                free(state.dict.vals);
                state.dict.keys = NULL;
                state.dict.vals = NULL;
                state.dict.count = 0;
            } else {
                pk_val_free_contents(&state);
            }
            break;
        }
        case PK_TUPLE_VAR: {
            /* Pop items from mark to stack top, push as tuple */
            int mark = pk__find_mark(s);
            if (mark < 0) break;
            int count = s->sp - mark - 1;
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_TUPLE;
            v.tuple.count = count;
            v.tuple.cap = count > 0 ? count : 1;
            v.tuple.items = (pk_val *)malloc(v.tuple.cap * sizeof(pk_val));
            for (int i = 0; i < count; i++)
                v.tuple.items[i] = s->stack[mark + 1 + i];
            s->sp = mark;
            pk__push(s, v);
            break;
        }
        case PK_TUPLE1: {
            if (s->sp < 1) break;
            pk_val a = pk__pop(s);
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_TUPLE;
            v.tuple.count = 1;
            v.tuple.cap = 1;
            v.tuple.items = (pk_val *)malloc(sizeof(pk_val));
            v.tuple.items[0] = a;
            pk__push(s, v);
            break;
        }
        case PK_TUPLE2: {
            if (s->sp < 2) break;
            pk_val b = pk__pop(s);
            pk_val a = pk__pop(s);
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_TUPLE;
            v.tuple.count = 2;
            v.tuple.cap = 2;
            v.tuple.items = (pk_val *)malloc(2 * sizeof(pk_val));
            v.tuple.items[0] = a;
            v.tuple.items[1] = b;
            pk__push(s, v);
            break;
        }
        case PK_TUPLE3: {
            if (s->sp < 3) break;
            pk_val c = pk__pop(s);
            pk_val b = pk__pop(s);
            pk_val a = pk__pop(s);
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_TUPLE;
            v.tuple.count = 3;
            v.tuple.cap = 3;
            v.tuple.items = (pk_val *)malloc(3 * sizeof(pk_val));
            v.tuple.items[0] = a;
            v.tuple.items[1] = b;
            v.tuple.items[2] = c;
            pk__push(s, v);
            break;
        }
        case PK_EMPTY_TUPLE: {
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_TUPLE;
            pk__push(s, v);
            break;
        }
        case PK_EMPTY_LIST: {
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_LIST;
            pk__push(s, v);
            break;
        }
        case PK_APPENDS: {
            int mark = pk__find_mark(s);
            if (mark < 0 || mark < 1) break;
            pk_val *list = &s->stack[mark - 1];
            if (list->type == PV_LIST || list->type == PV_TUPLE) {
                int count = s->sp - mark - 1;
                for (int i = 0; i < count; i++) {
                    if (list->tuple.count >= list->tuple.cap) {
                        list->tuple.cap = list->tuple.cap ? list->tuple.cap * 2 : 16;
                        list->tuple.items = (pk_val *)realloc(list->tuple.items,
                            list->tuple.cap * sizeof(pk_val));
                    }
                    list->tuple.items[list->tuple.count++] = s->stack[mark + 1 + i];
                }
            } else {
                for (int i = mark + 1; i < s->sp; i++)
                    pk_val_free_contents(&s->stack[i]);
            }
            s->sp = mark;
            break;
        }
        case PK_APPEND: {
            if (s->sp < 2) break;
            pk_val item = pk__pop(s);
            pk_val *list = pk__top(s);
            if (list && (list->type == PV_LIST || list->type == PV_TUPLE)) {
                if (list->tuple.count >= list->tuple.cap) {
                    list->tuple.cap = list->tuple.cap ? list->tuple.cap * 2 : 16;
                    list->tuple.items = (pk_val *)realloc(list->tuple.items,
                        list->tuple.cap * sizeof(pk_val));
                }
                list->tuple.items[list->tuple.count++] = item;
            } else {
                pk_val_free_contents(&item);
            }
            break;
        }
        case PK_BINPUT: {
            uint32_t idx = pk__u8(s);
            if (idx < PK_MEMO_MAX && s->sp > 0) {
                if (s->memo_used[idx]) pk_val_free_contents(&s->memo[idx]);
                s->memo[idx] = pk_val_copy(&s->stack[s->sp - 1]);
                s->memo_used[idx] = 1;
            }
            break;
        }
        case PK_LONG_BINPUT: {
            uint32_t idx = pk__u32(s);
            if (idx < PK_MEMO_MAX && s->sp > 0) {
                if (s->memo_used[idx]) pk_val_free_contents(&s->memo[idx]);
                s->memo[idx] = pk_val_copy(&s->stack[s->sp - 1]);
                s->memo_used[idx] = 1;
            }
            break;
        }
        case PK_BINGET: {
            uint32_t idx = pk__u8(s);
            if (idx < PK_MEMO_MAX && s->memo_used[idx]) {
                pk__push(s, pk_val_copy(&s->memo[idx]));
            } else {
                pk__push(s, pk_val_none());
            }
            break;
        }
        case PK_LONG_BINGET: {
            uint32_t idx = pk__u32(s);
            if (idx < PK_MEMO_MAX && s->memo_used[idx]) {
                pk__push(s, pk_val_copy(&s->memo[idx]));
            } else {
                pk__push(s, pk_val_none());
            }
            break;
        }
        case PK_BININT: {
            pk__push(s, pk_val_int(pk__i32(s)));
            break;
        }
        case PK_BININT1: {
            pk__push(s, pk_val_int(pk__u8(s)));
            break;
        }
        case PK_BININT2: {
            pk__push(s, pk_val_int(pk__u16(s)));
            break;
        }
        case PK_LONG1: {
            uint8_t n = pk__u8(s);
            if (n == 0) {
                pk__push(s, pk_val_int(0));
            } else {
                /* Read n bytes as little-endian signed integer */
                int64_t val = 0;
                for (int i = 0; i < n && i < 8; i++) {
                    val |= ((int64_t)pk__u8(s)) << (i * 8);
                }
                /* Sign extend */
                if (n < 8 && n > 0 && (val & ((int64_t)1 << (n * 8 - 1)))) {
                    val |= ~(((int64_t)1 << (n * 8)) - 1);
                }
                /* Skip remaining bytes if n > 8 */
                for (int i = 8; i < n; i++) pk__u8(s);
                pk__push(s, pk_val_int(val));
            }
            break;
        }
        case PK_BINFLOAT: {
            pk__push(s, pk_val_float(pk__f64_be(s)));
            break;
        }
        case PK_NEWTRUE: {
            pk__push(s, pk_val_bool(1));
            break;
        }
        case PK_NEWFALSE: {
            pk__push(s, pk_val_bool(0));
            break;
        }
        case PK_NONE: {
            pk__push(s, pk_val_none());
            break;
        }
        case PK_BINPERSID: {
            /* Pop persistent ID tuple, resolve to storage ref */
            pk_val pid = pk__pop(s);
            pk_val v;
            memset(&v, 0, sizeof(v));
            v.type = PV_STORAGE_REF;

            if (pid.type == PV_TUPLE && pid.tuple.count >= 5) {
                /* Old-style: ('storage', StorageType, key, device, numel) */
                /* pid.items[0] = "storage" string */
                /* pid.items[1] = global (e.g., torch.FloatStorage) */
                /* pid.items[2] = key string (e.g., "0") */
                /* pid.items[3] = device string (e.g., "cpu") */
                /* pid.items[4] = numel int */

                /* Get storage type for dtype */
                if (pid.tuple.items[1].type == PV_GLOBAL) {
                    char type_str[256];
                    snprintf(type_str, sizeof(type_str), "%s.%s",
                             pid.tuple.items[1].global.module,
                             pid.tuple.items[1].global.name);
                    pk__storage_dtype(type_str, v.storage.dtype_str, &v.storage.dtype_size);
                }

                /* Get storage key */
                if (pid.tuple.items[2].type == PV_STRING) {
                    int kl = pid.tuple.items[2].str.len < 63 ? pid.tuple.items[2].str.len : 63;
                    memcpy(v.storage.storage_key, pid.tuple.items[2].str.ptr, kl);
                    v.storage.storage_key[kl] = '\0';
                }

                /* Get numel */
                if (pid.tuple.items[4].type == PV_INT) {
                    v.storage.numel = pid.tuple.items[4].i;
                }
            } else if (pid.type == PV_TUPLE && pid.tuple.count >= 3) {
                /* New-style (PyTorch 2.0+): may have different structure */
                /* Try to extract what we can */
                if (pid.tuple.items[0].type == PV_STRING) {
                    /* ('storage', ...) */
                    if (pid.tuple.items[1].type == PV_GLOBAL) {
                        char type_str[256];
                        snprintf(type_str, sizeof(type_str), "%s.%s",
                                 pid.tuple.items[1].global.module,
                                 pid.tuple.items[1].global.name);
                        pk__storage_dtype(type_str, v.storage.dtype_str, &v.storage.dtype_size);
                    }
                    if (pid.tuple.items[2].type == PV_STRING) {
                        int kl = pid.tuple.items[2].str.len < 63 ? pid.tuple.items[2].str.len : 63;
                        memcpy(v.storage.storage_key, pid.tuple.items[2].str.ptr, kl);
                        v.storage.storage_key[kl] = '\0';
                    }
                }
            }

            pk_val_free_contents(&pid);
            pk__push(s, v);
            break;
        }
        default: {
            fprintf(stderr, "pth: unknown pickle opcode 0x%02x at pos %zu\n",
                    op, s->pos - 1);
            return pk_val_none();
        }
        }
    }

    if (s->sp > 0) return s->stack[s->sp - 1];
    return pk_val_none();
}

/* ---- Tensor extraction from parsed pickle ---- */

/* Recursively find the state_dict in a nested structure.
 * For torch.save(state_dict): top-level is the dict.
 * For torch.save({'model': state_dict}): find 'model' key.
 * Returns pointer to the dict pk_val (not owned). */
static pk_val *pk__find_state_dict(pk_val *root) {
    if (!root) return NULL;

    /* If it's a dict/ordered_dict with tensor values, it's the state_dict */
    if (root->type == PV_DICT || root->type == PV_ORDERED_DICT) {
        /* Check if any value is a tensor */
        for (int i = 0; i < root->dict.count; i++) {
            if (root->dict.vals[i].type == PV_REBUILD_TENSOR)
                return root;
        }
        /* Otherwise, check nested dicts for common keys */
        static const char *sd_keys[] = {
            "model", "state_dict", "model_state_dict", "net", "params", NULL
        };
        for (int k = 0; sd_keys[k]; k++) {
            for (int i = 0; i < root->dict.count; i++) {
                if (strcmp(root->dict.keys[i], sd_keys[k]) == 0) {
                    pk_val *inner = pk__find_state_dict(&root->dict.vals[i]);
                    if (inner) return inner;
                }
            }
        }
        /* Last resort: check all nested dicts */
        for (int i = 0; i < root->dict.count; i++) {
            if (root->dict.vals[i].type == PV_DICT ||
                root->dict.vals[i].type == PV_ORDERED_DICT) {
                pk_val *inner = pk__find_state_dict(&root->dict.vals[i]);
                if (inner) return inner;
            }
        }
    }
    return NULL;
}

/* Count tensors in a state_dict */
static int pk__count_tensors(const pk_val *dict) {
    int count = 0;
    for (int i = 0; i < dict->dict.count; i++) {
        if (dict->dict.vals[i].type == PV_REBUILD_TENSOR)
            count++;
    }
    return count;
}

/* ---- Public API ---- */

size_t pth_dtype_size(const char *dtype_str) {
    static const struct { const char *name; size_t size; } tbl[] = {
        {"F32", 4}, {"F16", 2}, {"BF16", 2}, {"F64", 8},
        {"I8", 1}, {"U8", 1}, {"I16", 2}, {"I32", 4}, {"I64", 8},
        {"BOOL", 1},
    };
    for (int i = 0; i < (int)(sizeof(tbl) / sizeof(tbl[0])); i++) {
        if (strcmp(tbl[i].name, dtype_str) == 0) return tbl[i].size;
    }
    return 0;
}

pth_context *pth_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "pth: cannot open %s\n", path); return NULL; }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 22) {
        fprintf(stderr, "pth: file too small\n");
        close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    void *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fprintf(stderr, "pth: mmap failed\n");
        return NULL;
    }

    const uint8_t *data = (const uint8_t *)map;

    /* Parse ZIP directory */
    zip_dir dir;
    memset(&dir, 0, sizeof(dir));
    if (zip__parse(data, file_size, &dir) != 0) {
        fprintf(stderr, "pth: not a valid ZIP file (PyTorch .pth must be ZIP format)\n");
        munmap(map, file_size);
        return NULL;
    }

    /* Find data.pkl entry */
    int pkl_idx = zip__find_suffix(&dir, "data.pkl");
    if (pkl_idx < 0) {
        fprintf(stderr, "pth: cannot find data.pkl in ZIP\n");
        zip__free(&dir);
        munmap(map, file_size);
        return NULL;
    }

    if (dir.entries[pkl_idx].method != 0) {
        fprintf(stderr, "pth: data.pkl is compressed (only store method supported)\n");
        zip__free(&dir);
        munmap(map, file_size);
        return NULL;
    }

    /* Parse pickle */
    const uint8_t *pkl_data = data + dir.entries[pkl_idx].data_offset;
    size_t pkl_size = dir.entries[pkl_idx].uncomp_size;

    pk_state *ps = (pk_state *)calloc(1, sizeof(pk_state));
    ps->data = pkl_data;
    ps->size = pkl_size;
    ps->pos = 0;
    ps->sp = 0;

    pk_val root = pk__parse(ps);

    /* Find the state_dict (handles nested dicts) */
    pk_val *sd = pk__find_state_dict(&root);
    if (!sd) {
        fprintf(stderr, "pth: cannot find state_dict with tensors\n");
        /* Clean up */
        for (int i = 0; i < ps->sp; i++) pk_val_free_contents(&ps->stack[i]);
        for (int i = 0; i < PK_MEMO_MAX; i++) {
            if (ps->memo_used[i]) pk_val_free_contents(&ps->memo[i]);
        }
        if (root.type != PV_NONE) pk_val_free_contents(&root);
        free(ps);
        zip__free(&dir);
        munmap(map, file_size);
        return NULL;
    }

    int n_tensors = pk__count_tensors(sd);
    pth_tensor_info *tensors = (pth_tensor_info *)calloc(n_tensors, sizeof(pth_tensor_info));

    /* Determine the archive prefix (e.g., "archive/" or model-specific) */
    /* Find any data/ entry to detect prefix */
    char prefix[256] = "";
    for (int i = 0; i < dir.n_entries; i++) {
        char *dp = strstr(dir.entries[i].name, "data/");
        if (dp && dp != dir.entries[i].name) {
            int plen = (int)(dp - dir.entries[i].name);
            if (plen < (int)sizeof(prefix)) {
                memcpy(prefix, dir.entries[i].name, plen);
                prefix[plen] = '\0';
            }
            break;
        } else if (dp == dir.entries[i].name) {
            prefix[0] = '\0';
            break;
        }
    }

    int ti = 0;
    for (int i = 0; i < sd->dict.count; i++) {
        if (sd->dict.vals[i].type != PV_REBUILD_TENSOR) continue;

        pk_tensor_rebuild *tr = &sd->dict.vals[i].tensor;
        pth_tensor_info *t = &tensors[ti];

        t->name = strdup(sd->dict.keys[i]);
        memcpy(t->dtype_str, tr->storage.dtype_str, sizeof(t->dtype_str));
        t->n_dims = tr->n_dims;
        for (int d = 0; d < tr->n_dims; d++)
            t->shape[d] = (uint64_t)tr->shape[d];

        /* Compute nbytes from shape and dtype */
        size_t elem_size = pth_dtype_size(t->dtype_str);
        size_t n_elements = 1;
        for (int d = 0; d < t->n_dims; d++) n_elements *= t->shape[d];
        t->nbytes = n_elements * elem_size;

        /* Find the storage data in ZIP */
        char storage_path[512];
        snprintf(storage_path, sizeof(storage_path), "%sdata/%s",
                 prefix, tr->storage.storage_key);

        int zip_idx = -1;
        for (int z = 0; z < dir.n_entries; z++) {
            if (strcmp(dir.entries[z].name, storage_path) == 0) {
                zip_idx = z;
                break;
            }
        }

        if (zip_idx >= 0) {
            size_t byte_offset = tr->storage_offset * elem_size;
            t->data = (void *)(data + dir.entries[zip_idx].data_offset + byte_offset);

            /* Validate data range */
            size_t data_end = dir.entries[zip_idx].data_offset + byte_offset + t->nbytes;
            if (data_end > file_size) {
                fprintf(stderr, "pth: tensor '%s' data extends past file end\n", t->name);
                for (int j = 0; j <= ti; j++) free(tensors[j].name);
                free(tensors);
                for (int j = 0; j < ps->sp; j++) pk_val_free_contents(&ps->stack[j]);
                for (int j = 0; j < PK_MEMO_MAX; j++) {
                    if (ps->memo_used[j]) pk_val_free_contents(&ps->memo[j]);
                }
                free(ps);
                zip__free(&dir);
                munmap(map, file_size);
                return NULL;
            }
        } else {
            fprintf(stderr, "pth: warning: cannot find storage '%s' for tensor '%s'\n",
                    storage_path, t->name);
            t->data = NULL;
        }

        ti++;
    }

    /* Clean up pickle state */
    /* Don't free root itself since it might be on the stack - just free memo */
    for (int i = 0; i < PK_MEMO_MAX; i++) {
        if (ps->memo_used[i]) pk_val_free_contents(&ps->memo[i]);
    }
    /* Free stack items that aren't the root */
    for (int i = 0; i < ps->sp; i++) {
        /* The root value is on the stack; sd points into it.
         * We've extracted what we need, so free everything. */
        pk_val_free_contents(&ps->stack[i]);
    }
    free(ps);
    zip__free(&dir);

    pth_context *ctx = (pth_context *)malloc(sizeof(pth_context));
    ctx->tensors = tensors;
    ctx->n_tensors = n_tensors;
    ctx->map_base = map;
    ctx->map_size = file_size;

    fprintf(stderr, "pth: loaded %d tensors from %s\n", n_tensors, path);
    return ctx;
}

void pth_close(pth_context *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_tensors; i++)
        free(ctx->tensors[i].name);
    free(ctx->tensors);
    munmap(ctx->map_base, ctx->map_size);
    free(ctx);
}

int pth_find(const pth_context *ctx, const char *name) {
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) return i;
    }
    return -1;
}

int pth_count(const pth_context *ctx) {
    return ctx->n_tensors;
}

const char *pth_name(const pth_context *ctx, int i) {
    return ctx->tensors[i].name;
}

const char *pth_dtype(const pth_context *ctx, int i) {
    return ctx->tensors[i].dtype_str;
}

int pth_ndims(const pth_context *ctx, int i) {
    return ctx->tensors[i].n_dims;
}

const uint64_t *pth_shape(const pth_context *ctx, int i) {
    return ctx->tensors[i].shape;
}

void *pth_data(const pth_context *ctx, int i) {
    return ctx->tensors[i].data;
}

size_t pth_nbytes(const pth_context *ctx, int i) {
    return ctx->tensors[i].nbytes;
}

#endif /* PTH_LOADER_IMPLEMENTATION */
#endif /* PTH_LOADER_H */
