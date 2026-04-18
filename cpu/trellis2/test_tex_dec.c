/* Stage 2: texture decoder test.
 *
 * Loads the TRELLIS.2 texture decoder (SparseUnetVaeDecoder, 6 out channels:
 * base_color[3] + metallic + roughness + alpha) and runs it on PyTorch-dumped
 * tex_slat intermediates, comparing against pbr_voxel_feats.npy.
 *
 * Inputs:
 *   tex_dec_next_dc_f16c32_fp16.safetensors  (from TRELLIS.2-4B/ckpts)
 *   tex_slat_feats.npy  [N, 32]  (from ref/trellis2/gen_stage2_ref.py dump)
 *   tex_slat_coords.npy [N, 4]
 *   pbr_voxel_feats.npy [N', 6]  (expected output, for comparison)
 */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal npy readers (copied from test_shape_dec.c). */
static float *read_npy_f32(const char *p, int *nd, int *dd) {
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 8, SEEK_SET);
    uint16_t hl;
    fread(&hl, 2, 1, f);
    char *h = malloc(hl + 1);
    fread(h, 1, hl, f);
    h[hl] = 0;
    *nd = 0;
    char *sp = strstr(h, "shape");
    if (sp) {
        sp = strchr(sp, '(');
        if (sp) {
            sp++;
            while (*sp && *sp != ')') {
                while (*sp == ' ' || *sp == ',') sp++;
                if (*sp == ')') break;
                dd[*nd] = (int)strtol(sp, &sp, 10);
                (*nd)++;
            }
        }
    }
    size_t n = 1;
    for (int i = 0; i < *nd; i++) n *= dd[i];
    float *d = malloc(n * sizeof(float));
    fread(d, sizeof(float), n, f);
    fclose(f);
    free(h);
    return d;
}

static int32_t *read_npy_i32(const char *p, int *nd, int *dd) {
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 8, SEEK_SET);
    uint16_t hl;
    fread(&hl, 2, 1, f);
    char *h = malloc(hl + 1);
    fread(h, 1, hl, f);
    h[hl] = 0;
    *nd = 0;
    char *sp = strstr(h, "shape");
    if (sp) {
        sp = strchr(sp, '(');
        if (sp) {
            sp++;
            while (*sp && *sp != ')') {
                while (*sp == ' ' || *sp == ',') sp++;
                if (*sp == ')') break;
                dd[*nd] = (int)strtol(sp, &sp, 10);
                (*nd)++;
            }
        }
    }
    size_t n = 1;
    for (int i = 0; i < *nd; i++) n *= dd[i];
    int32_t *d = malloc(n * sizeof(int32_t));
    fread(d, sizeof(int32_t), n, f);
    fclose(f);
    free(h);
    return d;
}

/* Hash actual pbr_voxel coords so we can align rows to our result. The decoder
 * expands each input voxel 8x per C2S stage, so row order will differ between
 * our output and the reference dump even if feature content matches. */
static uint64_t pack_coord(int32_t z, int32_t y, int32_t x) {
    return ((uint64_t)(uint32_t)z << 40) | ((uint64_t)(uint32_t)y << 20) |
           (uint64_t)(uint32_t)x;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "Usage: %s <tex_dec.st> <tex_slat_feats.npy> "
                "<tex_slat_coords.npy> <pbr_voxel_feats.npy> "
                "[pbr_voxel_coords.npy] [-t threads]\n",
                argv[0]);
        return 1;
    }
    int n_threads = 4;
    const char *ref_coords_path = NULL;
    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            ref_coords_path = argv[i];
        }
    }

    t2_shape_dec *dec = t2_shape_dec_load(argv[1]);
    if (!dec) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    int nd, dd[8];
    float *slat_feats = read_npy_f32(argv[2], &nd, dd);
    int N = dd[0];
    int C = nd >= 2 ? dd[1] : 1;
    fprintf(stderr, "tex_slat feats: [%d, %d]\n", N, C);

    int32_t *slat_coords = read_npy_i32(argv[3], &nd, dd);
    fprintf(stderr, "tex_slat coords: [%d, %d]\n", dd[0], dd[1]);

    sp3d_tensor *slat = sp3d_create(slat_coords, slat_feats, N, C, 1);
    free(slat_feats);
    free(slat_coords);

    fprintf(stderr, "\nRunning tex decoder (%d threads)...\n", n_threads);
    t2_shape_dec_result r = t2_shape_dec_forward(dec, slat, n_threads);
    fprintf(stderr, "decoded: N=%d, C=%d\n", r.N, r.C);
    if (r.C != 6) {
        fprintf(stderr,
                "warning: expected C=6 (PBR), got C=%d — is this really the "
                "tex decoder?\n",
                r.C);
    }

    /* Pipeline applies `* 0.5 + 0.5` after decoder forward. */
    for (int i = 0; i < r.N * r.C; i++) r.feats[i] = r.feats[i] * 0.5f + 0.5f;

    int ref_nd, ref_dd[8];
    float *ref_feats = read_npy_f32(argv[4], &ref_nd, ref_dd);
    int ref_N = ref_dd[0], ref_C = ref_nd >= 2 ? ref_dd[1] : 1;
    fprintf(stderr, "ref pbr_voxel feats: [%d, %d]\n", ref_N, ref_C);

    if (ref_N != r.N || ref_C != r.C) {
        fprintf(stderr,
                "shape mismatch: got [%d,%d] vs ref [%d,%d] — cannot compare "
                "directly\n",
                r.N, r.C, ref_N, ref_C);
        goto stats_only;
    }

    /* Align rows by coordinate if reference coords provided. */
    if (ref_coords_path) {
        int rc_nd, rc_dd[8];
        int32_t *ref_coords = read_npy_i32(ref_coords_path, &rc_nd, rc_dd);
        fprintf(stderr, "ref pbr_voxel coords: [%d, %d]\n", rc_dd[0], rc_dd[1]);
        int stride_ref = rc_dd[1];
        int stride_out = 4; /* our result has [batch, z, y, x] */

        /* Build map from packed coord -> row index in our output. */
        typedef struct {
            uint64_t k;
            int v;
        } kv;
        kv *map = (kv *)calloc((size_t)r.N * 2, sizeof(kv));
        int mcap = r.N * 2;
        for (int i = 0; i < r.N; i++) {
            int32_t z = r.coords[i * stride_out + 1];
            int32_t y = r.coords[i * stride_out + 2];
            int32_t x = r.coords[i * stride_out + 3];
            uint64_t k = pack_coord(z, y, x) + 1; /* avoid 0 */
            int h = (int)((k * 2654435761u) & (uint32_t)(mcap - 1));
            while (map[h].k && map[h].k != k) h = (h + 1) & (mcap - 1);
            map[h].k = k;
            map[h].v = i;
        }

        double sum_sq_err = 0, sum_sq_ref = 0, max_abs = 0;
        int matched = 0, missing = 0;
        for (int i = 0; i < ref_N; i++) {
            int off = rc_nd >= 2 && stride_ref == 4 ? 1 : 0;
            int32_t z = ref_coords[i * stride_ref + off];
            int32_t y = ref_coords[i * stride_ref + off + 1];
            int32_t x = ref_coords[i * stride_ref + off + 2];
            uint64_t k = pack_coord(z, y, x) + 1;
            int h = (int)((k * 2654435761u) & (uint32_t)(mcap - 1));
            while (map[h].k && map[h].k != k) h = (h + 1) & (mcap - 1);
            if (!map[h].k) {
                missing++;
                continue;
            }
            int j = map[h].v;
            matched++;
            for (int c = 0; c < r.C; c++) {
                float a = r.feats[j * r.C + c];
                float b = ref_feats[i * ref_C + c];
                double d = (double)a - b;
                sum_sq_err += d * d;
                sum_sq_ref += (double)b * b;
                if (fabs(d) > max_abs) max_abs = fabs(d);
            }
        }
        double rel = sqrt(sum_sq_err / (sum_sq_ref + 1e-30));
        fprintf(stderr,
                "\ncompared: matched=%d/%d missing=%d rel_err=%.6f "
                "max_abs_err=%.6f\n",
                matched, ref_N, missing, rel, max_abs);
        free(map);
        free(ref_coords);
    }

stats_only: {
    double s[6] = {0}, s2[6] = {0};
    for (int i = 0; i < r.N; i++) {
        for (int c = 0; c < r.C && c < 6; c++) {
            float v = r.feats[i * r.C + c];
            s[c] += v;
            s2[c] += (double)v * v;
        }
    }
    const char *names[6] = {"R", "G", "B", "metal", "rough", "alpha"};
    fprintf(stderr, "\nper-channel mean/std (post-affine):\n");
    for (int c = 0; c < r.C && c < 6; c++) {
        double m = s[c] / r.N;
        double v = s2[c] / r.N - m * m;
        fprintf(stderr, "  %-5s mean=%.4f std=%.4f\n", names[c], m,
                sqrt(v > 0 ? v : 0));
    }
}

    free(ref_feats);
    t2_shape_dec_result_free(&r);
    sp3d_free(slat);
    t2_shape_dec_free(dec);
    return 0;
}
