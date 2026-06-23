#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *model_dir;
    const char *tensor;
    const char *scheme;
    const char *csv;
    const char *act_stat;
    const char *dump_prefix;
    const char *fp16_chunks;
    int rows_cap;
    int cols_cap;
    int block;
    int svd_rank;
    int svd_iters;
    int x_rows;
    float smooth_alpha;
} opts_t;

typedef struct {
    char name[64];
    double mse, mae, max_abs, rel_l2, cosine, sqnr_db, sat_pct;
    size_t n;
} metrics_t;

static void print_metric(metrics_t m);

static float fp8_e4m3fn_to_f32(uint8_t x)
{
    uint8_t sign = (x >> 7) & 1, exp = (x >> 3) & 0xf, mant = x & 7;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = (uint32_t)sign << 31;
        } else {
            int sh = 0;
            while ((mant & 4) == 0) {
                mant <<= 1;
                sh++;
            }
            mant &= 3;
            bits = ((uint32_t)sign << 31) | ((uint32_t)(127 - 7 - sh) << 23) | ((uint32_t)mant << 20);
        }
    } else if (exp == 15 && mant == 7) {
        bits = ((uint32_t)sign << 31) | (0xffu << 23) | (1u << 22);
    } else {
        bits = ((uint32_t)sign << 31) | ((uint32_t)(exp + 120) << 23) | ((uint32_t)mant << 20);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float bf16_to_f32(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t f32_to_fp16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        if (((x >> 23) & 0xff) == 0xff && mant)
            return (uint16_t)(sign | 0x7e00u);
        return (uint16_t)(sign | 0x7c00u);
    }
    uint32_t half_mant = mant >> 13;
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
        half_mant++;
        if (half_mant == 0x400u) {
            half_mant = 0;
            exp++;
            if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_mant);
}

static float fp16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3ffu;
            bits = sign | (uint32_t)(e + 127) << 23 | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s --model DIR --tensor NAME [--scheme all|tensor|row|block|block_mse|block_p99|smooth|awq|svd|int4|fp4|i16|i16_row|i16_block|i16_smooth|i16_awq|i16_svd|fp16]\n"
            "          [--rows N] [--cols N] [--block N] [--smooth-alpha A]\n"
            "          [--act-stat file] [--svd-rank N] [--svd-iters N]\n"
            "          [--x-rows N] [--fp16-chunks 16,32,...] [--csv out.csv] [--dump-int8 prefix]\n",
            argv0);
}

static int parse_int(const char *s)
{
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v <= 0 || v > 2147483647L) {
        fprintf(stderr, "bad integer: %s\n", s);
        exit(2);
    }
    return (int)v;
}

static opts_t parse_opts(int argc, char **argv)
{
    opts_t o = {0};
    o.scheme = "all";
    o.rows_cap = 512;
    o.cols_cap = 4096;
    o.block = 128;
    o.svd_rank = 4;
    o.svd_iters = 8;
    o.x_rows = 0;
    o.smooth_alpha = 0.5f;
    o.fp16_chunks = "16,32,64,128,256,512";
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char **dst = NULL;
        if (!strcmp(a, "--model")) dst = &o.model_dir;
        else if (!strcmp(a, "--tensor")) dst = &o.tensor;
        else if (!strcmp(a, "--scheme")) dst = &o.scheme;
        else if (!strcmp(a, "--csv")) dst = &o.csv;
        else if (!strcmp(a, "--act-stat")) dst = &o.act_stat;
        else if (!strcmp(a, "--dump-int8")) dst = &o.dump_prefix;
        else if (!strcmp(a, "--fp16-chunks")) dst = &o.fp16_chunks;
        if (dst) {
            if (++i == argc) { usage(argv[0]); exit(2); }
            *dst = argv[i];
            continue;
        }
        if (!strcmp(a, "--rows")) o.rows_cap = parse_int(argv[++i]);
        else if (!strcmp(a, "--cols")) o.cols_cap = parse_int(argv[++i]);
        else if (!strcmp(a, "--block")) o.block = parse_int(argv[++i]);
        else if (!strcmp(a, "--svd-rank")) o.svd_rank = parse_int(argv[++i]);
        else if (!strcmp(a, "--svd-iters")) o.svd_iters = parse_int(argv[++i]);
        else if (!strcmp(a, "--x-rows")) o.x_rows = parse_int(argv[++i]);
        else if (!strcmp(a, "--smooth-alpha")) o.smooth_alpha = strtof(argv[++i], NULL);
        else { usage(argv[0]); exit(2); }
    }
    if (!o.model_dir || !o.tensor) {
        usage(argv[0]);
        exit(2);
    }
    return o;
}

static char *slurp(const char *path, size_t *len_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    buf[n] = 0;
    fclose(f);
    if (len_out) *len_out = (size_t)n;
    return buf;
}

static char *find_shard(const char *model_dir, const char *tensor)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", model_dir);
    size_t len = 0;
    char *json = slurp(path, &len);
    if (!json) {
        fprintf(stderr, "cannot read %s: %s\n", path, strerror(errno));
        return NULL;
    }
    size_t pat_len = strlen(tensor) + 4;
    char *pat = (char *)malloc(pat_len + 1);
    snprintf(pat, pat_len + 1, "\"%s\"", tensor);
    char *p = strstr(json, pat);
    free(pat);
    if (!p) {
        fprintf(stderr, "tensor not in index: %s\n", tensor);
        free(json);
        return NULL;
    }
    p = strchr(p + 1, ':');
    if (!p) { free(json); return NULL; }
    p = strchr(p, '"');
    if (!p) { free(json); return NULL; }
    char *q = strchr(++p, '"');
    if (!q) { free(json); return NULL; }
    size_t n = (size_t)(q - p);
    char *out = (char *)malloc(n + 1);
    memcpy(out, p, n);
    out[n] = 0;
    free(json);
    return out;
}

static float *load_act_stat(const char *path, int cols)
{
    if (!path) return NULL;
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open act stat %s: %s\n", path, strerror(errno));
        exit(1);
    }
    float *a = (float *)malloc((size_t)cols * sizeof(float));
    for (int i = 0; i < cols; i++) {
        if (fscanf(f, "%f", &a[i]) != 1) {
            fprintf(stderr, "act stat needs at least %d floats\n", cols);
            exit(1);
        }
        if (a[i] <= 0 || !isfinite(a[i])) a[i] = 1.0f;
    }
    fclose(f);
    return a;
}

static void dequant_sample(float *w, st_context *ctx, int wi, int si, int rows, int cols)
{
    const char *dtype = safetensors_dtype(ctx, wi);
    const uint64_t *shape = safetensors_shape(ctx, wi);
    int full_cols = (int)shape[1];

    if (strcmp(dtype, "F8_E4M3") == 0) {
        const uint8_t *fp8 = (const uint8_t *)safetensors_data(ctx, wi);
        const float *sc = (const float *)safetensors_data(ctx, si);
        int sb_cols = (full_cols + 127) / 128;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                float v = fp8_e4m3fn_to_f32(fp8[(size_t)r * full_cols + c]);
                float s = sc[(size_t)(r / 128) * sb_cols + (c / 128)];
                w[(size_t)r * cols + c] = v * s;
            }
        }
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *bf = (const uint16_t *)safetensors_data(ctx, wi);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++)
                w[(size_t)r * cols + c] = bf16_to_f32(bf[(size_t)r * full_cols + c]);
        }
    }
}

static metrics_t score(const char *name, const float *ref, const float *approx, const int8_t *q, size_t n)
{
    metrics_t m = {0};
    snprintf(m.name, sizeof(m.name), "%s", name);
    m.n = n;
    double se = 0, ae = 0, maxe = 0, ref2 = 0, app2 = 0, dot = 0;
    size_t sat = 0;
    for (size_t i = 0; i < n; i++) {
        double e = (double)approx[i] - ref[i];
        double a = fabs(e);
        se += e * e;
        ae += a;
        if (a > maxe) maxe = a;
        ref2 += (double)ref[i] * ref[i];
        app2 += (double)approx[i] * approx[i];
        dot += (double)ref[i] * approx[i];
        if (q && (q[i] == 127 || q[i] == -127 || q[i] == -128)) sat++;
    }
    m.mse = se / (double)n;
    m.mae = ae / (double)n;
    m.max_abs = maxe;
    m.rel_l2 = sqrt(se / (ref2 + 1e-30));
    m.cosine = dot / (sqrt(ref2 * app2) + 1e-30);
    m.sqnr_db = 10.0 * log10((ref2 + 1e-30) / (se + 1e-30));
    m.sat_pct = 100.0 * (double)sat / (double)n;
    return m;
}

static int8_t qround(float x)
{
    int v = (int)lrintf(x);
    if (v > 127) v = 127;
    if (v < -127) v = -127;
    return (int8_t)v;
}

static metrics_t quant_fp16_storage(const float *w, float *out, uint16_t *h, int rows, int cols)
{
    size_t n = (size_t)rows * cols;
    for (size_t i = 0; i < n; i++) {
        h[i] = f32_to_fp16(w[i]);
        out[i] = fp16_to_f32(h[i]);
    }
    return score("fp16", w, out, NULL, n);
}

static int16_t qround16(float x)
{
    int v = (int)lrintf(x);
    if (v > 32767) v = 32767;
    if (v < -32767) v = -32767;
    return (int16_t)v;
}

static metrics_t score_i16(const char *name, const float *ref, const float *approx, const int16_t *q, size_t n)
{
    metrics_t m = score(name, ref, approx, NULL, n);
    size_t sat = 0;
    for (size_t i = 0; i < n; i++)
        if (q && (q[i] == 32767 || q[i] == -32767 || q[i] == -32768)) sat++;
    m.sat_pct = 100.0 * (double)sat / (double)n;
    return m;
}

static metrics_t quant_i16_tensor(const float *w, float *out, int16_t *q, int rows, int cols)
{
    size_t n = (size_t)rows * cols;
    float maxabs = 0.0f;
    for (size_t i = 0; i < n; i++) if (fabsf(w[i]) > maxabs) maxabs = fabsf(w[i]);
    float s = maxabs > 0.0f ? maxabs / 32767.0f : 1.0f;
    for (size_t i = 0; i < n; i++) {
        q[i] = qround16(w[i] / s);
        out[i] = (float)q[i] * s;
    }
    return score_i16("i16", w, out, q, n);
}

static metrics_t quant_i16_row(const float *w, float *out, int16_t *q, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float maxabs = 0.0f;
        for (int c = 0; c < cols; c++) {
            float a = fabsf(w[(size_t)r * cols + c]);
            if (a > maxabs) maxabs = a;
        }
        float s = maxabs > 0.0f ? maxabs / 32767.0f : 1.0f;
        for (int c = 0; c < cols; c++) {
            size_t i = (size_t)r * cols + c;
            q[i] = qround16(w[i] / s);
            out[i] = (float)q[i] * s;
        }
    }
    return score_i16("i16_row", w, out, q, (size_t)rows * cols);
}

static metrics_t quant_i16_block_scaled(const char *name, const float *w, const float *col_scale,
                                        float *out, int16_t *q, int rows, int cols, int block)
{
    for (int rb = 0; rb < rows; rb += block) {
        int re = rb + block < rows ? rb + block : rows;
        for (int cb = 0; cb < cols; cb += block) {
            int ce = cb + block < cols ? cb + block : cols;
            float maxabs = 0.0f;
            for (int r = rb; r < re; r++) {
                for (int c = cb; c < ce; c++) {
                    float cs = col_scale ? col_scale[c] : 1.0f;
                    float a = fabsf(w[(size_t)r * cols + c] * cs);
                    if (a > maxabs) maxabs = a;
                }
            }
            float s = maxabs > 0.0f ? maxabs / 32767.0f : 1.0f;
            for (int r = rb; r < re; r++) {
                for (int c = cb; c < ce; c++) {
                    size_t i = (size_t)r * cols + c;
                    float cs = col_scale ? col_scale[c] : 1.0f;
                    q[i] = qround16((w[i] * cs) / s);
                    out[i] = ((float)q[i] * s) / cs;
                }
            }
        }
    }
    return score_i16(name, w, out, q, (size_t)rows * cols);
}

static metrics_t quant_tensor(const float *w, float *out, int8_t *q, int rows, int cols)
{
    size_t n = (size_t)rows * cols;
    float maxabs = 0.0f;
    for (size_t i = 0; i < n; i++) if (fabsf(w[i]) > maxabs) maxabs = fabsf(w[i]);
    float s = maxabs > 0 ? maxabs / 127.0f : 1.0f;
    for (size_t i = 0; i < n; i++) {
        q[i] = qround(w[i] / s);
        out[i] = (float)q[i] * s;
    }
    return score("tensor", w, out, q, n);
}

static metrics_t quant_row(const float *w, float *out, int8_t *q, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float maxabs = 0.0f;
        for (int c = 0; c < cols; c++) if (fabsf(w[(size_t)r * cols + c]) > maxabs) maxabs = fabsf(w[(size_t)r * cols + c]);
        float s = maxabs > 0 ? maxabs / 127.0f : 1.0f;
        for (int c = 0; c < cols; c++) {
            size_t i = (size_t)r * cols + c;
            q[i] = qround(w[i] / s);
            out[i] = (float)q[i] * s;
        }
    }
    return score("row", w, out, q, (size_t)rows * cols);
}

static metrics_t quant_block_scaled(const char *name, const float *w, const float *col_scale,
                                    float *out, int8_t *q, int rows, int cols, int block)
{
    for (int rb = 0; rb < rows; rb += block) {
        int re = rb + block < rows ? rb + block : rows;
        for (int cb = 0; cb < cols; cb += block) {
            int ce = cb + block < cols ? cb + block : cols;
            float maxabs = 0.0f;
            for (int r = rb; r < re; r++) {
                for (int c = cb; c < ce; c++) {
                    float v = w[(size_t)r * cols + c] * (col_scale ? col_scale[c] : 1.0f);
                    if (fabsf(v) > maxabs) maxabs = fabsf(v);
                }
            }
            float s = maxabs > 0 ? maxabs / 127.0f : 1.0f;
            for (int r = rb; r < re; r++) {
                for (int c = cb; c < ce; c++) {
                    size_t i = (size_t)r * cols + c;
                    float cs = col_scale ? col_scale[c] : 1.0f;
                    q[i] = qround((w[i] * cs) / s);
                    out[i] = ((float)q[i] * s) / cs;
                }
            }
        }
    }
    return score(name, w, out, q, (size_t)rows * cols);
}

static metrics_t quant_int4_block(const float *w, float *out, int8_t *q, int rows, int cols, int block)
{
    for (int rb = 0; rb < rows; rb += block) {
        int re = rb + block < rows ? rb + block : rows;
        for (int cb = 0; cb < cols; cb += block) {
            int ce = cb + block < cols ? cb + block : cols;
            float maxabs = 0.0f;
            for (int r = rb; r < re; r++) {
                for (int c = cb; c < ce; c++) {
                    float a = fabsf(w[(size_t)r * cols + c]);
                    if (a > maxabs) maxabs = a;
                }
            }
            float s = maxabs > 0.0f ? maxabs / 7.0f : 1.0f;
            for (int r = rb; r < re; r++) {
                for (int c = cb; c < ce; c++) {
                    size_t i = (size_t)r * cols + c;
                    int v = (int)lrintf(w[i] / s);
                    if (v > 7) v = 7;
                    if (v < -7) v = -7;
                    q[i] = (int8_t)v;
                    out[i] = (float)v * s;
                }
            }
        }
    }
    return score("int4", w, out, q, (size_t)rows * cols);
}

static float fp4_e2m1_dequant(float x)
{
    static const float cb[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabsf(x);
    int best = 0;
    float be = fabsf(ax - cb[0]);
    for (int i = 1; i < 8; i++) {
        float e = fabsf(ax - cb[i]);
        if (e < be) {
            be = e;
            best = i;
        }
    }
    return sign * cb[best];
}

static metrics_t quant_fp4_block(const float *w, float *out, int8_t *q, int rows, int cols, int block)
{
    for (int rb = 0; rb < rows; rb += block) {
        int re = rb + block < rows ? rb + block : rows;
        for (int cb = 0; cb < cols; cb += block) {
            int ce = cb + block < cols ? cb + block : cols;
            float maxabs = 0.0f;
            for (int r = rb; r < re; r++) {
                for (int c = cb; c < ce; c++) {
                    float a = fabsf(w[(size_t)r * cols + c]);
                    if (a > maxabs) maxabs = a;
                }
            }
            float s = maxabs > 0.0f ? maxabs / 6.0f : 1.0f;
            for (int r = rb; r < re; r++) {
                for (int c = cb; c < ce; c++) {
                    size_t i = (size_t)r * cols + c;
                    float code = fp4_e2m1_dequant(w[i] / s);
                    q[i] = (int8_t)lrintf(code);
                    out[i] = code * s;
                }
            }
        }
    }
    return score("fp4", w, out, NULL, (size_t)rows * cols);
}

static void quant_one_block(const float *w, const float *col_scale, float *out, int8_t *q,
                            int rows, int cols, int rb, int re, int cb, int ce, float clip)
{
    (void)rows;
    float s = clip > 0.0f ? clip / 127.0f : 1.0f;
    for (int r = rb; r < re; r++) {
        for (int c = cb; c < ce; c++) {
            size_t i = (size_t)r * cols + c;
            float cs = col_scale ? col_scale[c] : 1.0f;
            float x = w[i] * cs;
            if (x > clip) x = clip;
            if (x < -clip) x = -clip;
            q[i] = qround(x / s);
            out[i] = ((float)q[i] * s) / cs;
        }
    }
}

static double block_sse_for_clip(const float *w, const float *col_scale, int rows, int cols,
                                 int rb, int re, int cb, int ce, float clip)
{
    (void)rows;
    float s = clip > 0.0f ? clip / 127.0f : 1.0f;
    double se = 0.0;
    for (int r = rb; r < re; r++) {
        for (int c = cb; c < ce; c++) {
            size_t i = (size_t)r * cols + c;
            float cs = col_scale ? col_scale[c] : 1.0f;
            float x = w[i] * cs;
            if (x > clip) x = clip;
            if (x < -clip) x = -clip;
            int8_t qi = qround(x / s);
            double y = ((double)qi * s) / cs;
            double e = y - w[i];
            se += e * e;
        }
    }
    return se;
}

static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static metrics_t quant_block_clipped(const char *name, const float *w, const float *col_scale,
                                     float *out, int8_t *q, int rows, int cols, int block,
                                     int mse_search, float percentile)
{
    int max_block_n = block * block;
    float *absbuf = NULL;
    if (percentile > 0.0f) {
        absbuf = (float *)malloc((size_t)max_block_n * sizeof(float));
        if (!absbuf) {
            fprintf(stderr, "oom in percentile block quant\n");
            exit(1);
        }
    }
    for (int rb = 0; rb < rows; rb += block) {
        int re = rb + block < rows ? rb + block : rows;
        for (int cb = 0; cb < cols; cb += block) {
            int ce = cb + block < cols ? cb + block : cols;
            float maxabs = 0.0f;
            int n = 0;
            for (int r = rb; r < re; r++) {
                for (int c = cb; c < ce; c++) {
                    float cs = col_scale ? col_scale[c] : 1.0f;
                    float a = fabsf(w[(size_t)r * cols + c] * cs);
                    if (a > maxabs) maxabs = a;
                    if (absbuf) absbuf[n++] = a;
                }
            }
            float clip = maxabs;
            if (percentile > 0.0f && n > 0) {
                qsort(absbuf, (size_t)n, sizeof(float), cmp_float);
                int idx = (int)floorf(percentile * (float)(n - 1));
                if (idx < 0) idx = 0;
                if (idx >= n) idx = n - 1;
                clip = absbuf[idx];
                if (clip <= 0.0f) clip = maxabs;
            }
            if (mse_search && maxabs > 0.0f) {
                float best_clip = clip;
                double best = block_sse_for_clip(w, col_scale, rows, cols, rb, re, cb, ce, best_clip);
                for (int k = 0; k <= 20; k++) {
                    float f = 1.0f - 0.025f * (float)k;
                    float cand = maxabs * f;
                    double se = block_sse_for_clip(w, col_scale, rows, cols, rb, re, cb, ce, cand);
                    if (se < best) {
                        best = se;
                        best_clip = cand;
                    }
                }
                clip = best_clip;
            }
            quant_one_block(w, col_scale, out, q, rows, cols, rb, re, cb, ce, clip);
        }
    }
    free(absbuf);
    return score(name, w, out, q, (size_t)rows * cols);
}

static metrics_t quant_smooth(const float *w, float *out, int8_t *q, int rows, int cols,
                              int block, float alpha, const float *act)
{
    float *cs = (float *)malloc((size_t)cols * sizeof(float));
    for (int c = 0; c < cols; c++) {
        float wmax = 0.0f;
        for (int r = 0; r < rows; r++) {
            float a = fabsf(w[(size_t)r * cols + c]);
            if (a > wmax) wmax = a;
        }
        float amax = act ? act[c] : 1.0f;
        cs[c] = powf(amax + 1e-12f, alpha) / powf(wmax + 1e-12f, 1.0f - alpha);
        if (!isfinite(cs[c]) || cs[c] <= 0.0f) cs[c] = 1.0f;
    }
    metrics_t m = quant_block_scaled("smooth", w, cs, out, q, rows, cols, block);
    free(cs);
    return m;
}

static metrics_t quant_i16_smooth(const float *w, float *out, int16_t *q, int rows, int cols,
                                  int block, float alpha, const float *act)
{
    float *cs = (float *)malloc((size_t)cols * sizeof(float));
    for (int c = 0; c < cols; c++) {
        float wmax = 0.0f;
        for (int r = 0; r < rows; r++) {
            float a = fabsf(w[(size_t)r * cols + c]);
            if (a > wmax) wmax = a;
        }
        float amax = act ? act[c] : 1.0f;
        cs[c] = powf(amax + 1e-12f, alpha) / powf(wmax + 1e-12f, 1.0f - alpha);
        if (!isfinite(cs[c]) || cs[c] <= 0.0f) cs[c] = 1.0f;
    }
    metrics_t m = quant_i16_block_scaled("i16_smooth", w, cs, out, q, rows, cols, block);
    free(cs);
    return m;
}

static metrics_t quant_awq_grid(const float *w, float *out, int8_t *q, int rows, int cols,
                                int block, const float *act)
{
    static const float alphas[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float *tmp = (float *)malloc((size_t)rows * cols * sizeof(float));
    int8_t *tq = (int8_t *)malloc((size_t)rows * cols);
    float *cs = (float *)malloc((size_t)cols * sizeof(float));
    if (!tmp || !tq || !cs) {
        fprintf(stderr, "oom in awq grid\n");
        exit(1);
    }
    metrics_t best = {0};
    best.mse = INFINITY;
    for (size_t ai = 0; ai < sizeof(alphas) / sizeof(alphas[0]); ai++) {
        float alpha = alphas[ai];
        for (int c = 0; c < cols; c++) {
            double rms = 0.0;
            float wmax = 0.0f;
            for (int r = 0; r < rows; r++) {
                float v = fabsf(w[(size_t)r * cols + c]);
                rms += (double)v * v;
                if (v > wmax) wmax = v;
            }
            rms = sqrt(rms / (double)rows);
            float imp = act ? act[c] : (float)rms;
            cs[c] = powf(imp + 1e-12f, alpha) / powf(wmax + 1e-12f, 1.0f - alpha);
            if (!isfinite(cs[c]) || cs[c] <= 0.0f) cs[c] = 1.0f;
        }
        metrics_t m = quant_block_clipped("awq", w, cs, tmp, tq, rows, cols, block, 1, 0.0f);
        if (m.mse < best.mse) {
            best = m;
            memcpy(out, tmp, (size_t)rows * cols * sizeof(float));
            memcpy(q, tq, (size_t)rows * cols);
        }
    }
    snprintf(best.name, sizeof(best.name), "awq");
    free(tmp);
    free(tq);
    free(cs);
    return best;
}

static metrics_t quant_i16_awq_grid(const float *w, float *out, int16_t *q, int rows, int cols,
                                    int block, const float *act)
{
    static const float alphas[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float *tmp = (float *)malloc((size_t)rows * cols * sizeof(float));
    int16_t *tq = (int16_t *)malloc((size_t)rows * cols * sizeof(int16_t));
    float *cs = (float *)malloc((size_t)cols * sizeof(float));
    if (!tmp || !tq || !cs) {
        fprintf(stderr, "oom in i16 awq grid\n");
        exit(1);
    }
    metrics_t best = {0};
    best.mse = INFINITY;
    for (size_t ai = 0; ai < sizeof(alphas) / sizeof(alphas[0]); ai++) {
        float alpha = alphas[ai];
        for (int c = 0; c < cols; c++) {
            double rms = 0.0;
            float wmax = 0.0f;
            for (int r = 0; r < rows; r++) {
                float v = fabsf(w[(size_t)r * cols + c]);
                rms += (double)v * v;
                if (v > wmax) wmax = v;
            }
            rms = sqrt(rms / (double)rows);
            float imp = act ? act[c] : (float)rms;
            cs[c] = powf(imp + 1e-12f, alpha) / powf(wmax + 1e-12f, 1.0f - alpha);
            if (!isfinite(cs[c]) || cs[c] <= 0.0f) cs[c] = 1.0f;
        }
        metrics_t m = quant_i16_block_scaled("i16_awq", w, cs, tmp, tq, rows, cols, block);
        if (m.mse < best.mse) {
            best = m;
            memcpy(out, tmp, (size_t)rows * cols * sizeof(float));
            memcpy(q, tq, (size_t)rows * cols * sizeof(int16_t));
        }
    }
    snprintf(best.name, sizeof(best.name), "i16_awq");
    free(tmp);
    free(tq);
    free(cs);
    return best;
}

static uint32_t rng_u32(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static float synth_act(int row, int col)
{
    uint32_t x = (uint32_t)(row + 1) * 747796405u ^ (uint32_t)(col + 17) * 2891336453u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    float u = ((x >> 8) & 0xffffff) * (1.0f / 16777215.0f);
    float v = 2.0f * u - 1.0f;
    if ((x & 255u) == 0) v *= 6.0f;
    return v;
}

static metrics_t matmul_error_bits(const char *name, const float *w, const float *wq,
                                   int x_rows, int rows, int cols, int x_bits)
{
    float *y = (float *)malloc((size_t)x_rows * rows * sizeof(float));
    float *yq = (float *)malloc((size_t)x_rows * rows * sizeof(float));
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    float *xdq = (float *)malloc((size_t)cols * sizeof(float));
    if (!y || !yq || !x || !xdq) {
        fprintf(stderr, "oom in matmul error\n");
        exit(1);
    }
    for (int m = 0; m < x_rows; m++) {
        float maxabs = 0.0f;
        for (int c = 0; c < cols; c++) {
            x[c] = synth_act(m, c);
            if (fabsf(x[c]) > maxabs) maxabs = fabsf(x[c]);
        }
        float denom = x_bits == 16 ? 32767.0f : 127.0f;
        float xs = maxabs > 0.0f ? maxabs / denom : 1.0f;
        for (int c = 0; c < cols; c++) {
            if (x_bits == 16)
                xdq[c] = (float)qround16(x[c] / xs) * xs;
            else
                xdq[c] = (float)qround(x[c] / xs) * xs;
        }
        for (int r = 0; r < rows; r++) {
            double acc = 0.0, accq = 0.0;
            const float *wr = w + (size_t)r * cols;
            const float *wqr = wq + (size_t)r * cols;
            for (int c = 0; c < cols; c++) {
                acc += (double)x[c] * wr[c];
                accq += (double)xdq[c] * wqr[c];
            }
            y[(size_t)m * rows + r] = (float)acc;
            yq[(size_t)m * rows + r] = (float)accq;
        }
    }
    char label[64];
    snprintf(label, sizeof(label), "%s+x", name);
    metrics_t metric = score(label, y, yq, NULL, (size_t)x_rows * rows);
    free(y);
    free(yq);
    free(x);
    free(xdq);
    return metric;
}

static metrics_t matmul_error(const char *name, const float *w, const float *wq,
                              int x_rows, int rows, int cols)
{
    return matmul_error_bits(name, w, wq, x_rows, rows, cols, 8);
}

static metrics_t matmul_error_fp16_chunk(const float *w, const uint16_t *wh,
                                         int x_rows, int rows, int cols, int chunk)
{
    float *y = (float *)malloc((size_t)x_rows * rows * sizeof(float));
    float *yq = (float *)malloc((size_t)x_rows * rows * sizeof(float));
    float *x = (float *)malloc((size_t)cols * sizeof(float));
    uint16_t *xh = (uint16_t *)malloc((size_t)cols * sizeof(uint16_t));
    if (!y || !yq || !x || !xh) {
        fprintf(stderr, "oom in fp16 matmul error\n");
        exit(1);
    }
    if (chunk <= 0) chunk = cols;
    for (int m = 0; m < x_rows; m++) {
        for (int c = 0; c < cols; c++) {
            x[c] = synth_act(m, c);
            xh[c] = f32_to_fp16(x[c]);
        }
        for (int r = 0; r < rows; r++) {
            double ref = 0.0;
            const float *wr = w + (size_t)r * cols;
            const uint16_t *whr = wh + (size_t)r * cols;
            for (int c = 0; c < cols; c++)
                ref += (double)x[c] * wr[c];

            float total = 0.0f;
            for (int k0 = 0; k0 < cols; k0 += chunk) {
                int kend = k0 + chunk < cols ? k0 + chunk : cols;
                float partial = 0.0f;
                for (int c = k0; c < kend; c++)
                    partial = fmaf(fp16_to_f32(xh[c]), fp16_to_f32(whr[c]), partial);
                total += partial;
            }
            y[(size_t)m * rows + r] = (float)ref;
            yq[(size_t)m * rows + r] = total;
        }
    }
    char label[64];
    snprintf(label, sizeof(label), "fp16+x%d", chunk);
    metrics_t metric = score(label, y, yq, NULL, (size_t)x_rows * rows);
    free(y);
    free(yq);
    free(x);
    free(xh);
    return metric;
}

static void print_fp16_chunk_metrics(const char *chunks, const float *w, const uint16_t *wh,
                                     int x_rows, int rows, int cols)
{
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", chunks ? chunks : "16,32,64,128,256,512");
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        int chunk = atoi(tok);
        if (chunk > 0)
            print_metric(matmul_error_fp16_chunk(w, wh, x_rows, rows, cols, chunk));
    }
}

static metrics_t quant_svd(const float *w, float *out, int8_t *q, int rows, int cols,
                           int block, int rank, int iters)
{
    metrics_t base = quant_block_scaled("svd", w, NULL, out, q, rows, cols, block);
    (void)base;
    float *res = (float *)malloc((size_t)rows * cols * sizeof(float));
    float *u = (float *)malloc((size_t)rows * sizeof(float));
    float *v = (float *)malloc((size_t)cols * sizeof(float));
    float *tmp = (float *)malloc((size_t)(rows > cols ? rows : cols) * sizeof(float));
    if (!res || !u || !v || !tmp) {
        fprintf(stderr, "oom in svd residual\n");
        exit(1);
    }
    for (size_t i = 0, n = (size_t)rows * cols; i < n; i++) res[i] = w[i] - out[i];
    uint32_t seed = 1;
    for (int k = 0; k < rank; k++) {
        double nv = 0.0;
        for (int c = 0; c < cols; c++) {
            v[c] = ((int)(rng_u32(&seed) & 0xffff) - 32768) / 32768.0f;
            nv += (double)v[c] * v[c];
        }
        nv = 1.0 / sqrt(nv + 1e-30);
        for (int c = 0; c < cols; c++) v[c] = (float)(v[c] * nv);
        for (int it = 0; it < iters; it++) {
            double nu = 0.0;
            for (int r = 0; r < rows; r++) {
                double s = 0.0;
                for (int c = 0; c < cols; c++) s += (double)res[(size_t)r * cols + c] * v[c];
                u[r] = (float)s;
                nu += s * s;
            }
            nu = 1.0 / sqrt(nu + 1e-30);
            for (int r = 0; r < rows; r++) u[r] = (float)(u[r] * nu);
            nv = 0.0;
            for (int c = 0; c < cols; c++) {
                double s = 0.0;
                for (int r = 0; r < rows; r++) s += (double)res[(size_t)r * cols + c] * u[r];
                tmp[c] = (float)s;
                nv += s * s;
            }
            nv = 1.0 / sqrt(nv + 1e-30);
            for (int c = 0; c < cols; c++) v[c] = (float)(tmp[c] * nv);
        }
        double sigma = 0.0;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                sigma += (double)u[r] * res[(size_t)r * cols + c] * v[c];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                size_t i = (size_t)r * cols + c;
                float corr = (float)(sigma * u[r] * v[c]);
                out[i] += corr;
                res[i] -= corr;
            }
        }
    }
    free(res);
    free(u);
    free(v);
    free(tmp);
    return score("svd", w, out, q, (size_t)rows * cols);
}

static metrics_t quant_i16_svd(const float *w, float *out, int16_t *q, int rows, int cols,
                               int block, int rank, int iters)
{
    metrics_t base = quant_i16_block_scaled("i16_svd", w, NULL, out, q, rows, cols, block);
    (void)base;
    float *res = (float *)malloc((size_t)rows * cols * sizeof(float));
    float *u = (float *)malloc((size_t)rows * sizeof(float));
    float *v = (float *)malloc((size_t)cols * sizeof(float));
    float *tmp = (float *)malloc((size_t)(rows > cols ? rows : cols) * sizeof(float));
    if (!res || !u || !v || !tmp) {
        fprintf(stderr, "oom in i16 svd residual\n");
        exit(1);
    }
    for (size_t i = 0, n = (size_t)rows * cols; i < n; i++) res[i] = w[i] - out[i];
    uint32_t seed = 1;
    for (int k = 0; k < rank; k++) {
        double nv = 0.0;
        for (int c = 0; c < cols; c++) {
            v[c] = ((int)(rng_u32(&seed) & 0xffff) - 32768) / 32768.0f;
            nv += (double)v[c] * v[c];
        }
        nv = 1.0 / sqrt(nv + 1e-30);
        for (int c = 0; c < cols; c++) v[c] = (float)(v[c] * nv);
        for (int it = 0; it < iters; it++) {
            double nu = 0.0;
            for (int r = 0; r < rows; r++) {
                double s = 0.0;
                for (int c = 0; c < cols; c++) s += (double)res[(size_t)r * cols + c] * v[c];
                u[r] = (float)s;
                nu += s * s;
            }
            nu = 1.0 / sqrt(nu + 1e-30);
            for (int r = 0; r < rows; r++) u[r] = (float)(u[r] * nu);
            nv = 0.0;
            for (int c = 0; c < cols; c++) {
                double s = 0.0;
                for (int r = 0; r < rows; r++) s += (double)res[(size_t)r * cols + c] * u[r];
                tmp[c] = (float)s;
                nv += s * s;
            }
            nv = 1.0 / sqrt(nv + 1e-30);
            for (int c = 0; c < cols; c++) v[c] = (float)(tmp[c] * nv);
        }
        double sigma = 0.0;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                sigma += (double)u[r] * res[(size_t)r * cols + c] * v[c];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                size_t i = (size_t)r * cols + c;
                float corr = (float)(sigma * u[r] * v[c]);
                out[i] += corr;
                res[i] -= corr;
            }
        }
    }
    free(res);
    free(u);
    free(v);
    free(tmp);
    return score_i16("i16_svd", w, out, q, (size_t)rows * cols);
}

static void print_metric(metrics_t m)
{
    printf("%-8s n=%zu rmse=%.8g mae=%.8g max=%.8g rel_l2=%.8g cosine=%.10f sqnr_db=%.3f sat=%.3f%%\n",
           m.name, m.n, sqrt(m.mse), m.mae, m.max_abs, m.rel_l2, m.cosine, m.sqnr_db, m.sat_pct);
}

static void print_metric_as(const char *name, metrics_t m)
{
    printf("%-8s n=%zu rmse=%.8g mae=%.8g max=%.8g rel_l2=%.8g cosine=%.10f sqnr_db=%.3f sat=%.3f%%\n",
           name, m.n, sqrt(m.mse), m.mae, m.max_abs, m.rel_l2, m.cosine, m.sqnr_db, m.sat_pct);
}

static void append_csv(const char *path, const char *tensor, int rows, int cols, metrics_t m)
{
    if (!path) return;
    int need_header = 0;
    FILE *probe = fopen(path, "r");
    if (!probe) need_header = 1;
    else fclose(probe);
    FILE *f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "cannot append csv %s: %s\n", path, strerror(errno));
        return;
    }
    if (need_header)
        fprintf(f, "tensor,scheme,rows,cols,n,rmse,mae,max_abs,rel_l2,cosine,sqnr_db,sat_pct\n");
    fprintf(f, "%s,%s,%d,%d,%zu,%.9g,%.9g,%.9g,%.9g,%.12g,%.9g,%.9g\n",
            tensor, m.name, rows, cols, m.n, sqrt(m.mse), m.mae, m.max_abs,
            m.rel_l2, m.cosine, m.sqnr_db, m.sat_pct);
    fclose(f);
}

static void dump_int8(const char *prefix, const int8_t *q, int rows, int cols)
{
    if (!prefix) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s.i8", prefix);
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s: %s\n", path, strerror(errno));
        return;
    }
    fwrite(q, 1, (size_t)rows * cols, f);
    fclose(f);
}

static int wants(const char *scheme, const char *name)
{
    return !strcmp(scheme, "all") || !strcmp(scheme, name);
}

int main(int argc, char **argv)
{
    opts_t o = parse_opts(argc, argv);
    char *shard = find_shard(o.model_dir, o.tensor);
    if (!shard) return 1;
    char shard_path[1024];
    snprintf(shard_path, sizeof(shard_path), "%s/%s", o.model_dir, shard);
    free(shard);

    st_context *ctx = safetensors_open(shard_path);
    if (!ctx) {
        fprintf(stderr, "failed to open shard %s\n", shard_path);
        return 1;
    }
    int wi = safetensors_find(ctx, o.tensor);
    char scale_name[512];
    snprintf(scale_name, sizeof(scale_name), "%s_scale_inv", o.tensor);
    int si = safetensors_find(ctx, scale_name);
    if (wi < 0) {
        fprintf(stderr, "missing tensor: %s\n", o.tensor);
        safetensors_close(ctx);
        return 1;
    }
    const uint64_t *shape = safetensors_shape(ctx, wi);
    int nd = safetensors_ndims(ctx, wi);
    const char *dtype = safetensors_dtype(ctx, wi);
    if (nd != 2 || (strcmp(dtype, "F8_E4M3") != 0 && strcmp(dtype, "BF16") != 0)) {
        fprintf(stderr, "expected 2D F8_E4M3 or BF16 weight, got dtype=%s ndims=%d\n", dtype, nd);
        safetensors_close(ctx);
        return 1;
    }
    if (strcmp(dtype, "F8_E4M3") == 0 && si < 0) {
        fprintf(stderr, "missing FP8 scale tensor: %s\n", scale_name);
        safetensors_close(ctx);
        return 1;
    }
    if (strcmp(dtype, "F8_E4M3") == 0 && strcmp(safetensors_dtype(ctx, si), "F32") != 0) {
        fprintf(stderr, "expected F32 scale_inv, got %s\n", safetensors_dtype(ctx, si));
        safetensors_close(ctx);
        return 1;
    }
    int rows = (int)shape[0] < o.rows_cap ? (int)shape[0] : o.rows_cap;
    int cols = (int)shape[1] < o.cols_cap ? (int)shape[1] : o.cols_cap;
    size_t n = (size_t)rows * cols;
    float *w = (float *)malloc(n * sizeof(float));
    float *out = (float *)malloc(n * sizeof(float));
    int8_t *q = (int8_t *)malloc(n);
    int16_t *q16 = (int16_t *)malloc(n * sizeof(int16_t));
    uint16_t *h16 = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!w || !out || !q || !q16 || !h16) {
        fprintf(stderr, "oom for sample %dx%d\n", rows, cols);
        return 1;
    }
    dequant_sample(w, ctx, wi, si, rows, cols);
    printf("tensor=%s shard=%s dtype=%s full=%llux%llu sample=%dx%d block=%d scale=%s\n",
           o.tensor, shard_path, dtype, (unsigned long long)shape[0], (unsigned long long)shape[1],
           rows, cols, o.block, si >= 0 ? scale_name : "none");

    if (wants(o.scheme, "tensor")) { metrics_t m = quant_tensor(w, out, q, rows, cols); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("tensor+x", matmul_error("tensor", w, out, o.x_rows, rows, cols)); }
    if (wants(o.scheme, "row")) { metrics_t m = quant_row(w, out, q, rows, cols); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("row+x", matmul_error("row", w, out, o.x_rows, rows, cols)); }
    if (wants(o.scheme, "block")) { metrics_t m = quant_block_scaled("block", w, NULL, out, q, rows, cols, o.block); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("block+x", matmul_error("block", w, out, o.x_rows, rows, cols)); dump_int8(o.dump_prefix, q, rows, cols); }
    if (wants(o.scheme, "block_mse")) { metrics_t m = quant_block_clipped("block_mse", w, NULL, out, q, rows, cols, o.block, 1, 0.0f); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("bmse+x", matmul_error("block_mse", w, out, o.x_rows, rows, cols)); }
    if (wants(o.scheme, "block_p99")) { metrics_t m = quant_block_clipped("block_p99", w, NULL, out, q, rows, cols, o.block, 0, 0.999f); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("bp99+x", matmul_error("block_p99", w, out, o.x_rows, rows, cols)); }
    if (wants(o.scheme, "int4")) { metrics_t m = quant_int4_block(w, out, q, rows, cols, o.block); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("int4+x", matmul_error("int4", w, out, o.x_rows, rows, cols)); }
    if (wants(o.scheme, "fp4")) { metrics_t m = quant_fp4_block(w, out, q, rows, cols, o.block); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("fp4+x", matmul_error("fp4", w, out, o.x_rows, rows, cols)); }
    if (wants(o.scheme, "smooth")) {
        float *act = load_act_stat(o.act_stat, cols);
        metrics_t m = quant_smooth(w, out, q, rows, cols, o.block, o.smooth_alpha, act);
        print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m);
        if (o.x_rows) print_metric_as("smooth+x", matmul_error("smooth", w, out, o.x_rows, rows, cols));
        free(act);
    }
    if (wants(o.scheme, "awq")) {
        float *act = load_act_stat(o.act_stat, cols);
        metrics_t m = quant_awq_grid(w, out, q, rows, cols, o.block, act);
        print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m);
        if (o.x_rows) print_metric_as("awq+x", matmul_error("awq", w, out, o.x_rows, rows, cols));
        free(act);
    }
    if (wants(o.scheme, "svd")) { metrics_t m = quant_svd(w, out, q, rows, cols, o.block, o.svd_rank, o.svd_iters); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("svd+x", matmul_error("svd", w, out, o.x_rows, rows, cols)); }
    if (wants(o.scheme, "i16")) { metrics_t m = quant_i16_tensor(w, out, q16, rows, cols); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("i16+x", matmul_error_bits("i16", w, out, o.x_rows, rows, cols, 16)); }
    if (wants(o.scheme, "i16_row")) { metrics_t m = quant_i16_row(w, out, q16, rows, cols); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("i16row+x", matmul_error_bits("i16_row", w, out, o.x_rows, rows, cols, 16)); }
    if (wants(o.scheme, "i16_block")) { metrics_t m = quant_i16_block_scaled("i16_block", w, NULL, out, q16, rows, cols, o.block); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("i16blk+x", matmul_error_bits("i16_block", w, out, o.x_rows, rows, cols, 16)); }
    if (wants(o.scheme, "i16_smooth")) {
        float *act = load_act_stat(o.act_stat, cols);
        metrics_t m = quant_i16_smooth(w, out, q16, rows, cols, o.block, o.smooth_alpha, act);
        print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m);
        if (o.x_rows) print_metric_as("i16sm+x", matmul_error_bits("i16_smooth", w, out, o.x_rows, rows, cols, 16));
        free(act);
    }
    if (wants(o.scheme, "i16_awq")) {
        float *act = load_act_stat(o.act_stat, cols);
        metrics_t m = quant_i16_awq_grid(w, out, q16, rows, cols, o.block, act);
        print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m);
        if (o.x_rows) print_metric_as("i16awq+x", matmul_error_bits("i16_awq", w, out, o.x_rows, rows, cols, 16));
        free(act);
    }
    if (wants(o.scheme, "i16_svd")) { metrics_t m = quant_i16_svd(w, out, q16, rows, cols, o.block, o.svd_rank, o.svd_iters); print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m); if (o.x_rows) print_metric_as("i16svd+x", matmul_error_bits("i16_svd", w, out, o.x_rows, rows, cols, 16)); }
    if (wants(o.scheme, "fp16")) {
        metrics_t m = quant_fp16_storage(w, out, h16, rows, cols);
        print_metric(m); append_csv(o.csv, o.tensor, rows, cols, m);
        if (o.x_rows) print_fp16_chunk_metrics(o.fp16_chunks, w, h16, o.x_rows, rows, cols);
    }

    free(w);
    free(out);
    free(q);
    free(q16);
    free(h16);
    safetensors_close(ctx);
    return 0;
}
