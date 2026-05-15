/*
 * tensor_diff.c - validate one VLMD dump dir against another.
 *
 * Usage:
 *   tensor_diff <ref_dir> <opt_dir> [--dtype fp32|bf16|fp16]
 *                                    [--abs A] [--rel R]
 *                                    [--only NAME]
 *                                    [--quiet]
 *
 * Iterates ref_dir/manifest.txt; for each tensor finds the matching file in
 * opt_dir (same filename) and computes max-abs, max-rel, RMSE. Pass/fail per
 * tolerance — defaults per-dtype:
 *
 *   fp32:  abs=1e-5  rel=1e-4
 *   bf16:  abs=5e-3  rel=5e-3
 *   fp16:  abs=5e-4  rel=1e-3
 *
 * Tolerances scale absolute by 100x for tensors with dim products > 1e6
 * (cumulative error from long reductions).
 *
 * Exit code: 0 = all PASS, 1 = any FAIL.
 */

#include "tensor_dump.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* fp16 / bf16 → fp32 (table-free, slow path is fine here). */
static float bf16_to_f32(uint16_t v) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)v) << 16;
    return c.f;
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 0x1u;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) { bits = s << 31; }
        else {
            /* subnormal */
            while (!(m & 0x400u)) { m <<= 1; e -= 1; }
            e += 1;
            m &= 0x3ffu;
            bits = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 31) {
        bits = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        bits = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
    union { uint32_t u; float f; } c; c.u = bits;
    return c.f;
}

static int load_as_f32(const char *path, float **out, size_t *out_n, int *dt_out, char *name_out) {
    vlmd_header h;
    void *data = NULL;
    size_t bytes = 0;
    if (vlmd_read(path, &h, &data, &bytes) != 0) return -1;
    size_t n = vlmd_numel(&h);
    float *f = (float *)malloc(n * sizeof(float));
    if (!f) { free(data); return -1; }
    switch (h.dtype) {
        case VLMD_F32: memcpy(f, data, n * sizeof(float)); break;
        case VLMD_BF16: {
            const uint16_t *u = (const uint16_t *)data;
            for (size_t i = 0; i < n; i++) f[i] = bf16_to_f32(u[i]);
            break;
        }
        case VLMD_F16: {
            const uint16_t *u = (const uint16_t *)data;
            for (size_t i = 0; i < n; i++) f[i] = f16_to_f32(u[i]);
            break;
        }
        case VLMD_I32: {
            const int32_t *u = (const int32_t *)data;
            for (size_t i = 0; i < n; i++) f[i] = (float)u[i];
            break;
        }
        default:
            free(data); free(f);
            return -1;
    }
    free(data);
    *out = f;
    *out_n = n;
    if (dt_out) *dt_out = (int)h.dtype;
    if (name_out) {
        strncpy(name_out, h.name, VLMD_NAME_MAX - 1);
        name_out[VLMD_NAME_MAX - 1] = '\0';
    }
    return 0;
}

typedef struct {
    double max_abs;
    double max_rel;
    double rmse;
    double ref_norm;
    double opt_norm;
    size_t numel;
    int    nan_count;
    int    inf_count;
} diff_stats;

static void compute_diff(const float *a, const float *b, size_t n, diff_stats *s) {
    memset(s, 0, sizeof(*s));
    s->numel = n;
    double sse = 0.0;
    for (size_t i = 0; i < n; i++) {
        float r = a[i];
        float o = b[i];
        if (isnan(r) || isnan(o)) { s->nan_count++; continue; }
        if (isinf(r) || isinf(o)) { s->inf_count++; continue; }
        double d = (double)r - (double)o;
        double ad = fabs(d);
        if (ad > s->max_abs) s->max_abs = ad;
        double denom = fabs((double)r);
        if (denom > 1e-7) {
            double rd = ad / denom;
            if (rd > s->max_rel) s->max_rel = rd;
        }
        sse += d * d;
        s->ref_norm += (double)r * r;
        s->opt_norm += (double)o * o;
    }
    s->rmse = (n > 0) ? sqrt(sse / (double)n) : 0.0;
    s->ref_norm = sqrt(s->ref_norm);
    s->opt_norm = sqrt(s->opt_norm);
}

static void usage(const char *p) {
    fprintf(stderr,
        "Usage: %s <ref_dir> <opt_dir> [--dtype fp32|bf16|fp16]\n"
        "                                [--abs A] [--rel R]\n"
        "                                [--only NAME] [--quiet]\n", p);
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 2; }
    const char *ref_dir = argv[1];
    const char *opt_dir = argv[2];
    const char *only = NULL;
    int quiet = 0;
    double abs_tol = 1e-5;
    double rel_tol = 1e-4;
    int abs_set = 0, rel_set = 0;
    const char *dtype_s = "fp32";

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--dtype") && i + 1 < argc) {
            dtype_s = argv[++i];
            if      (!strcasecmp(dtype_s, "fp32")) { abs_tol = 1e-5; rel_tol = 1e-4; }
            else if (!strcasecmp(dtype_s, "bf16")) { abs_tol = 5e-3; rel_tol = 5e-3; }
            else if (!strcasecmp(dtype_s, "fp16")) { abs_tol = 5e-4; rel_tol = 1e-3; }
            else { fprintf(stderr, "unknown dtype %s\n", dtype_s); return 2; }
        } else if (!strcmp(argv[i], "--abs") && i + 1 < argc) {
            abs_tol = atof(argv[++i]); abs_set = 1;
        } else if (!strcmp(argv[i], "--rel") && i + 1 < argc) {
            rel_tol = atof(argv[++i]); rel_set = 1;
        } else if (!strcmp(argv[i], "--only") && i + 1 < argc) {
            only = argv[++i];
        } else if (!strcmp(argv[i], "--quiet")) {
            quiet = 1;
        } else {
            fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    (void)abs_set; (void)rel_set;

    char manpath[1024];
    snprintf(manpath, sizeof(manpath), "%s/manifest.txt", ref_dir);
    FILE *mf = fopen(manpath, "r");
    if (!mf) { fprintf(stderr, "cannot open '%s'\n", manpath); return 2; }

    fprintf(stderr, "tensor_diff: ref='%s'  opt='%s'  dtype=%s  abs<=%g  rel<=%g\n",
            ref_dir, opt_dir, dtype_s, abs_tol, rel_tol);
    if (!quiet) {
        fprintf(stderr,
            "%-28s %5s %-8s %12s %12s %12s %10s %s\n",
            "name", "layer", "shape", "max_abs", "max_rel", "rmse", "numel", "result");
    }

    int total = 0, passed = 0, failed = 0, missing = 0;
    char line[1024];
    while (fgets(line, sizeof(line), mf)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        /* fields: filename name layer dtype ndim d0 d1 ... */
        char fname[256], tname[256], dts[16];
        int layer = -1, ndim = 0;
        int read_ok = sscanf(line, "%255s %255s %d %15s %d", fname, tname, &layer, dts, &ndim);
        if (read_ok < 5) continue;
        if (only && strcmp(tname, only) != 0) continue;

        char rpath[1024], opath[1024];
        snprintf(rpath, sizeof(rpath), "%s/%s", ref_dir, fname);
        snprintf(opath, sizeof(opath), "%s/%s", opt_dir, fname);

        float *ref_f = NULL, *opt_f = NULL;
        size_t rn = 0, on = 0;
        int rdt = 0, odt = 0;
        if (load_as_f32(rpath, &ref_f, &rn, &rdt, NULL) != 0) {
            fprintf(stderr, "missing ref: %s\n", fname);
            missing++; total++; continue;
        }
        if (load_as_f32(opath, &opt_f, &on, &odt, NULL) != 0) {
            fprintf(stderr, "missing opt: %s\n", fname);
            free(ref_f); missing++; total++; continue;
        }
        if (rn != on) {
            fprintf(stderr, "size mismatch %s: ref=%zu opt=%zu\n", fname, rn, on);
            free(ref_f); free(opt_f); failed++; total++; continue;
        }

        diff_stats s;
        compute_diff(ref_f, opt_f, rn, &s);

        /* shape string */
        char shape[64] = "";
        char *p = shape;
        char *ep = shape + sizeof(shape);
        const char *l = line;
        /* skip 5 fields to dims */
        for (int j = 0; j < 5; j++) { while (*l && *l != ' ' && *l != '\n') l++; while (*l == ' ') l++; }
        for (int d = 0; d < ndim && p < ep - 8; d++) {
            unsigned dim = 0;
            int got = sscanf(l, "%u", &dim);
            if (got != 1) break;
            int w = snprintf(p, ep - p, "%s%u", d ? "x" : "", dim);
            if (w < 0 || w >= ep - p) break;
            p += w;
            while (*l && *l != ' ' && *l != '\n') l++;
            while (*l == ' ') l++;
        }

        /* large-tensor tolerance scaling for cumulative reductions */
        double abs_eff = abs_tol;
        if (s.numel > 1000000ull) abs_eff = abs_tol * 100.0;

        int pass = (s.max_abs <= abs_eff) && (s.max_rel <= rel_tol)
                && (s.nan_count == 0) && (s.inf_count == 0);
        if (pass) passed++; else failed++;
        total++;

        if (!quiet || !pass) {
            fprintf(stderr,
                "%-28s %5d %-8s %12.4e %12.4e %12.4e %10zu  %s%s\n",
                tname, layer, shape,
                s.max_abs, s.max_rel, s.rmse, s.numel,
                pass ? "PASS" : "FAIL",
                (s.nan_count || s.inf_count) ? " (NaN/Inf)" : "");
        }
        free(ref_f); free(opt_f);
    }
    fclose(mf);

    fprintf(stderr, "\n=== Summary: %d total  %d pass  %d fail  %d missing ===\n",
            total, passed, failed, missing);
    return (failed == 0 && missing == 0) ? 0 : 1;
}
