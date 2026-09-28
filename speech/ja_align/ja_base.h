/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * ja_base.h - self-contained foundation for ja_align (no repo dependencies):
 *   - minimal safetensors reader (mmap + header parse; F32/BF16/F16 -> F32 on demand)
 *   - packed SGEMM (AVX2/FMA micro-kernel with portable scalar fallback, OpenMP optional)
 *   - small math helpers (LayerNorm, GELU, log-softmax), .npy writer
 * Written from the safetensors format description (header length + JSON + raw data).
 * Define JA_BASE_IMPLEMENTATION in exactly one translation unit.
 */
#ifndef JA_BASE_H
#define JA_BASE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- safetensors ---- */
typedef struct {
    char name[160];
    char dtype[8];
    int ndim;
    int64_t shape[6];
    size_t off, nbytes;
} ja_st_tensor;

typedef struct {
    void *map;
    size_t map_size;
    const uint8_t *data;
    ja_st_tensor *t;
    int n;
    char meta_inter[16];  /* __metadata__.inter_ctc_layer, if present */
} ja_st;

int                 ja_st_open(ja_st *st, const char *path);
void                ja_st_close(ja_st *st);
const ja_st_tensor *ja_st_find(const ja_st *st, const char *name);
/* Returns a malloc'd F32 copy; *count receives the element count. NULL if missing. */
float              *ja_st_f32(const ja_st *st, const char *name, size_t *count);

/* ---- GEMM: C[M,N] (+)= A[M,K] (row stride lda) * W[N,K]^T, W pre-packed ---- */
#define JA_NR 16
#define JA_MR 6
typedef struct { float *p; int n, k; } ja_packed;

void ja_pack_nk(ja_packed *dst, const float *w, int n, int k);  /* W[n][k] row-major */
void ja_packed_free(ja_packed *p);
void ja_sgemm(int m, const float *a, int lda, const ja_packed *b, float *c, int ldc, int accum);

/* ---- helpers ---- */
void  ja_layernorm(float *out, const float *x, const float *w, const float *b, int n, float eps);
float ja_gelu(float x);
void  ja_log_softmax(float *x, int n);
int   ja_npy_save_f32(const char *path, const float *d, int ndim, const int *dims);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef JA_BASE_IMPLEMENTATION

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define JA_HAVE_AVX2 1
#endif

/* -- safetensors header: {"name":{"dtype":"BF16","shape":[a,b],"data_offsets":[s,e]},...} -- */

static const char *ja__skip_ws(const char *p) { while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++; return p; }

/* reads a JSON string without escapes beyond \" and \\ (tensor names are plain ASCII) */
static const char *ja__str(const char *p, char *out, size_t cap) {
    size_t n = 0;
    if (*p != '"') return NULL;
    p++;
    while (*p && *p != '"') {
        char c = *p++;
        if (c == '\\' && *p) c = *p++;
        if (n + 1 < cap) out[n++] = c;
    }
    out[n] = 0;
    return *p == '"' ? p + 1 : NULL;
}

/* skip any JSON value */
static const char *ja__skip_value(const char *p) {
    p = ja__skip_ws(p);
    if (*p == '"') { char tmp[4]; (void)tmp; p++; while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; p++; } return *p ? p + 1 : p; }
    if (*p == '{' || *p == '[') {
        int depth = 0;
        do {
            if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; p++; } }
            else if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') depth--;
            if (*p) p++;
        } while (*p && depth > 0);
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

int ja_st_open(ja_st *st, const char *path) {
    memset(st, 0, sizeof(*st));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat sb;
    if (fstat(fd, &sb) || sb.st_size < 8) { close(fd); return -1; }
    st->map_size = (size_t)sb.st_size;
    st->map = mmap(NULL, st->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (st->map == MAP_FAILED) { st->map = NULL; return -1; }
    const uint8_t *b = (const uint8_t *)st->map;
    uint64_t hl = 0;
    for (int i = 7; i >= 0; i--) hl = hl << 8 | b[i];
    if (8 + hl > st->map_size) { ja_st_close(st); return -1; }
    char *hdr = (char *)malloc(hl + 1);
    memcpy(hdr, b + 8, hl);
    hdr[hl] = 0;
    st->data = b + 8 + hl;
    int cap = 64;
    st->t = (ja_st_tensor *)calloc((size_t)cap, sizeof(ja_st_tensor));
    const char *p = ja__skip_ws(hdr);
    if (*p == '{') p++;
    for (;;) {
        p = ja__skip_ws(p);
        if (*p == '}' || !*p) break;
        char key[160];
        p = ja__str(p, key, sizeof(key));
        if (!p) break;
        p = ja__skip_ws(p);
        if (*p == ':') p++;
        p = ja__skip_ws(p);
        if (!strcmp(key, "__metadata__")) {
            const char *m = strstr(p, "\"inter_ctc_layer\"");
            const char *end = ja__skip_value(p);
            if (m && m < end) {
                m = strchr(m + 17, '"');
                if (m) ja__str(m, st->meta_inter, sizeof(st->meta_inter));
            }
            p = end;
        } else {
            if (st->n == cap) { cap *= 2; st->t = (ja_st_tensor *)realloc(st->t, sizeof(ja_st_tensor) * (size_t)cap); }
            ja_st_tensor *t = &st->t[st->n];
            memset(t, 0, sizeof(*t));
            snprintf(t->name, sizeof(t->name), "%s", key);
            const char *end = ja__skip_value(p);
            const char *q;
            if ((q = strstr(p, "\"dtype\"")) && q < end) {
                q = ja__skip_ws(strchr(q + 7, ':') + 1);
                ja__str(q, t->dtype, sizeof(t->dtype));
            }
            if ((q = strstr(p, "\"shape\"")) && q < end) {
                q = strchr(q, '[') + 1;
                while (*q && *q != ']') {
                    q = ja__skip_ws(q);
                    if (*q == ']') break;
                    if (t->ndim < 6) t->shape[t->ndim++] = strtoll(q, (char **)&q, 10);
                    q = ja__skip_ws(q);
                    if (*q == ',') q++;
                }
            }
            if ((q = strstr(p, "\"data_offsets\"")) && q < end) {
                q = strchr(q, '[') + 1;
                size_t s0 = (size_t)strtoull(q, (char **)&q, 10);
                q = strchr(q, ',') + 1;
                size_t s1 = (size_t)strtoull(q, (char **)&q, 10);
                t->off = s0; t->nbytes = s1 - s0;
            }
            st->n++;
            p = end;
        }
        p = ja__skip_ws(p);
        if (*p == ',') p++;
    }
    free(hdr);
    return 0;
}

void ja_st_close(ja_st *st) {
    if (st->map) munmap(st->map, st->map_size);
    free(st->t);
    memset(st, 0, sizeof(*st));
}

const ja_st_tensor *ja_st_find(const ja_st *st, const char *name) {
    for (int i = 0; i < st->n; i++) if (!strcmp(st->t[i].name, name)) return &st->t[i];
    return NULL;
}

static float ja__half_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (e == 0) {
        if (m == 0) u = s;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; u = s | (e << 23) | (m << 13); }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 127 - 15) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

float *ja_st_f32(const ja_st *st, const char *name, size_t *count) {
    const ja_st_tensor *t = ja_st_find(st, name);
    if (!t) return NULL;
    size_t esz = !strcmp(t->dtype, "F32") ? 4 : 2;
    size_t n = t->nbytes / esz;
    float *d = (float *)malloc(sizeof(float) * (n ? n : 1));
    const uint8_t *src = st->data + t->off;
    if (esz == 4) memcpy(d, src, n * 4);
    else if (!strcmp(t->dtype, "BF16"))
        for (size_t i = 0; i < n; i++) { uint32_t u = (uint32_t)(src[2 * i] | src[2 * i + 1] << 8) << 16; memcpy(&d[i], &u, 4); }
    else if (!strcmp(t->dtype, "F16"))
        for (size_t i = 0; i < n; i++) d[i] = ja__half_to_f32((uint16_t)(src[2 * i] | src[2 * i + 1] << 8));
    else { free(d); return NULL; }
    if (count) *count = n;
    return d;
}

/* -- GEMM -- */

static void *ja__aligned(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes ? bytes : 64)) return NULL;
    return p;
}

void ja_pack_nk(ja_packed *dst, const float *w, int n, int k) {
    int np = (n + JA_NR - 1) / JA_NR;
    dst->n = n; dst->k = k;
    dst->p = (float *)ja__aligned((size_t)np * k * JA_NR * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int pi = 0; pi < np; pi++) {
        float *pp = dst->p + (size_t)pi * k * JA_NR;
        for (int kk = 0; kk < k; kk++)
            for (int j = 0; j < JA_NR; j++) {
                int nn = pi * JA_NR + j;
                pp[(size_t)kk * JA_NR + j] = nn < n ? w[(size_t)nn * k + kk] : 0.0f;
            }
    }
}

void ja_packed_free(ja_packed *p) { free(p->p); p->p = NULL; }

static void ja__kernel(int k, const float *const *ar, const float *bp, float *tile) {
#ifdef JA_HAVE_AVX2
    __m256 c[JA_MR][2];
    for (int i = 0; i < JA_MR; i++) c[i][0] = c[i][1] = _mm256_setzero_ps();
    for (int kk = 0; kk < k; kk++) {
        __m256 b0 = _mm256_load_ps(bp), b1 = _mm256_load_ps(bp + 8);
        bp += JA_NR;
        for (int i = 0; i < JA_MR; i++) {
            __m256 a = _mm256_broadcast_ss(ar[i] + kk);
            c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
        }
    }
    for (int i = 0; i < JA_MR; i++) {
        _mm256_storeu_ps(tile + i * JA_NR, c[i][0]);
        _mm256_storeu_ps(tile + i * JA_NR + 8, c[i][1]);
    }
#else
    for (int i = 0; i < JA_MR * JA_NR; i++) tile[i] = 0.0f;
    for (int kk = 0; kk < k; kk++, bp += JA_NR)
        for (int i = 0; i < JA_MR; i++) {
            float a = ar[i][kk];
            for (int j = 0; j < JA_NR; j++) tile[i * JA_NR + j] += a * bp[j];
        }
#endif
}

#define JA_KC 512
void ja_sgemm(int m, const float *a, int lda, const ja_packed *b, float *c, int ldc, int accum) {
    int n = b->n, k = b->k;
    int nb_m = (m + JA_MR - 1) / JA_MR, nb_n = (n + JA_NR - 1) / JA_NR;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int bn = 0; bn < nb_n; bn++)
        for (int bm = 0; bm < nb_m; bm++) {
            float tile[JA_MR * JA_NR], sum[JA_MR * JA_NR];
            int m0 = bm * JA_MR, mr = m - m0 < JA_MR ? m - m0 : JA_MR;
            int n0 = bn * JA_NR, nr = n - n0 < JA_NR ? n - n0 : JA_NR;
            const float *ar[JA_MR];
            memset(sum, 0, sizeof(sum));
            for (int k0 = 0; k0 < k; k0 += JA_KC) {
                int kc = k - k0 < JA_KC ? k - k0 : JA_KC;
                for (int i = 0; i < JA_MR; i++) ar[i] = a + (size_t)(m0 + (i < mr ? i : 0)) * lda + k0;
                ja__kernel(kc, ar, b->p + ((size_t)bn * k + k0) * JA_NR, tile);
                for (int i = 0; i < JA_MR * JA_NR; i++) sum[i] += tile[i];
            }
            for (int i = 0; i < mr; i++) {
                float *cr = c + (size_t)(m0 + i) * ldc + n0;
                if (accum) for (int j = 0; j < nr; j++) cr[j] += sum[i * JA_NR + j];
                else       for (int j = 0; j < nr; j++) cr[j] = sum[i * JA_NR + j];
            }
        }
}

void ja_layernorm(float *out, const float *x, const float *w, const float *b, int n, float eps) {
    double mu = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= n;
    for (int i = 0; i < n; i++) { double d = x[i] - mu; var += d * d; }
    var /= n;
    float r = (float)(1.0 / sqrt(var + eps));
    for (int i = 0; i < n; i++) out[i] = (float)(x[i] - mu) * r * w[i] + b[i];
}

float ja_gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

void ja_log_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) s += exp((double)x[i] - mx);
    float l = (float)(mx + log(s));
    for (int i = 0; i < n; i++) x[i] -= l;
}

int ja_npy_save_f32(const char *path, const float *d, int ndim, const int *dims) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    char hdr[256], shape[128] = "";
    size_t n = 1, off = 0;
    for (int i = 0; i < ndim; i++) {
        off += (size_t)snprintf(shape + off, sizeof(shape) - off, "%d%s", dims[i], ndim == 1 ? "," : (i + 1 < ndim ? ", " : ""));
        n *= (size_t)dims[i];
    }
    int hl = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shape);
    int pad = (64 - (10 + hl + 1) % 64) % 64;
    uint16_t hlen = (uint16_t)(hl + pad + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    size_t w = fwrite(d, 4, n, f);
    fclose(f);
    return w == n ? 0 : -1;
}

#endif /* JA_BASE_IMPLEMENTATION */
#endif /* JA_BASE_H */
