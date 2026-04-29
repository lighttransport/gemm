/*
 * npy_io.h — minimal little-endian .npy reader shared by verify_*.c
 * binaries across the cpu/sam3d, cpu/sam3d_body, and cuda/sam3d_body
 * ports.
 *
 * Supports rank ≤ 8 arrays of dtype 'f4' (float32), 'i4' (int32), or
 * 'u1' (uint8). No pickle, no fortran_order, no big-endian, no other
 * dtypes. The caller owns the returned malloc'd buffer.
 *
 * Functions are `static`, so this header may be included from multiple
 * translation units without an extra .c file.
 */
#ifndef GEMM_COMMON_NPY_IO_H
#define GEMM_COMMON_NPY_IO_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reads a .npy file. On success:
 *   - returns a malloc'd buffer the caller must free();
 *   - sets *ndim and dims[0..*ndim) to the array shape (rank-0 → ndim=0);
 *   - sets *is_f32 = 1 iff dtype is 'f4' (float32), 0 otherwise.
 * Returns NULL on any error (open failure, header parse, short read).
 *
 * Element size is inferred from the dtype tag: 'f4' / 'i4' = 4 bytes,
 * 'u1' = 1 byte. Other tags read 1 byte/element (caller's
 * responsibility to verify *is_f32 if they expect float).
 */
static void *npy_load(const char *path, int *ndim, int *dims, int *is_f32) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 8, SEEK_SET);
    uint16_t hl = 0;
    if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = 0;
    *ndim = 0;
    char *sp = strstr(hdr, "shape");
    if (sp) {
        sp = strchr(sp, '(');
        if (sp) {
            sp++;
            while (*sp && *sp != ')') {
                while (*sp == ' ' || *sp == ',') sp++;
                if (*sp == ')') break;
                dims[(*ndim)++] = (int)strtol(sp, &sp, 10);
                if (*ndim >= 8) break;
            }
        }
    }
    int is_i32 = (strstr(hdr, "i4") != NULL);
    int f32 = (strstr(hdr, "f4") != NULL);
    if (is_f32) *is_f32 = f32;
    int elt_bytes = (f32 || is_i32) ? 4 : 1;
    size_t n = 1;
    for (int i = 0; i < *ndim; i++) n *= (size_t)dims[i];
    if (*ndim == 0) n = 1;
    void *d = malloc(n * elt_bytes);
    size_t got = fread(d, elt_bytes, n, f);
    fclose(f); free(hdr);
    if (got != n) { free(d); return NULL; }
    return d;
}

/* Element-wise max/mean abs-diff between two f32 arrays of length n.
 * Returns max_abs; writes mean_abs to *mean_out if non-NULL. */
static float npy_max_abs_f32(const float *a, const float *b, int n,
                             double *mean_out)
{
    double sum = 0.0;
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    if (mean_out) *mean_out = sum / (n > 0 ? n : 1);
    return mx;
}

#endif  /* GEMM_COMMON_NPY_IO_H */
