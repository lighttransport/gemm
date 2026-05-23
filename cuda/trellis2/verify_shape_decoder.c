/* verify_shape_decoder.c - Compare CUDA SC-VAE decoder against a reference.
 * Usage: ./verify_shape_decoder <decoder.st> <slat.npy> <coords.npy>
 *        [--ref-feats <ref.npy> --ref-coords <ref_coords.npy>] [--skip-cpu]
 *        [--start-stage <0..4> --start-block <block>]
 *
 * slat.npy must be denormalized [N,32] features. coords.npy is [N,4]
 * int32 (batch,z,y,x).
 */
#include "cuda_trellis2_runner.h"
#include "../../common/sparse3d.h"
#include "../../common/trellis2_shape_decoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static const float *g_project_out_w = NULL;
static int g_project_out_ch = 0;

typedef struct {
    int ch;
    double value;
    double abs_value;
} t2_project_contrib;

static float *read_npy_f32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot read %s\n", path); return NULL; }
    fseek(f, 8, SEEK_SET);
    uint16_t hl;
    if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != (size_t)hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = 0;
    *ndim = 0;
    char *sp = strstr(hdr, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') {
            while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break;
            dims[*ndim] = (int)strtol(sp, &sp, 10);
            (*ndim)++;
            if (*ndim >= 8) break;
        }}}
    size_t n = 1; for (int i = 0; i < *ndim; i++) n *= (size_t)dims[i];
    float *data = (float *)malloc(n * sizeof(float));
    if (fread(data, sizeof(float), n, f) != n) { free(data); free(hdr); fclose(f); return NULL; }
    fclose(f); free(hdr);
    return data;
}

static int32_t *read_npy_i32(const char *path, int *ndim, int *dims) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot read %s\n", path); return NULL; }
    fseek(f, 8, SEEK_SET);
    uint16_t hl;
    if (fread(&hl, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = (char *)malloc(hl + 1);
    if (fread(hdr, 1, hl, f) != (size_t)hl) { free(hdr); fclose(f); return NULL; }
    hdr[hl] = 0;
    *ndim = 0;
    char *sp = strstr(hdr, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')') {
            while (*sp == ' ' || *sp == ',') sp++;
            if (*sp == ')') break;
            dims[*ndim] = (int)strtol(sp, &sp, 10);
            (*ndim)++;
            if (*ndim >= 8) break;
        }}}
    size_t n = 1; for (int i = 0; i < *ndim; i++) n *= (size_t)dims[i];
    int32_t *data = (int32_t *)malloc(n * sizeof(int32_t));
    if (fread(data, sizeof(int32_t), n, f) != n) { free(data); free(hdr); fclose(f); return NULL; }
    fclose(f); free(hdr);
    return data;
}

static void write_npy_f32(const char *path, const float *data, int N, int C) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    char hdr[256];
    int hl = snprintf(hdr, sizeof hdr,
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", N, C);
    while ((hl + 10) % 16 != 0) hdr[hl++] = ' ';
    hdr[hl++] = '\n';
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    uint16_t hl16 = (uint16_t)hl; fwrite(&hl16, 2, 1, f);
    fwrite(hdr, 1, hl, f);
    fwrite(data, sizeof(float), (size_t)N * C, f);
    fclose(f);
}

static void write_npy_i32(const char *path, const int32_t *data, int N, int C) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    char hdr[256];
    int hl = snprintf(hdr, sizeof hdr,
        "{'descr': '<i4', 'fortran_order': False, 'shape': (%d, %d), }", N, C);
    while ((hl + 10) % 16 != 0) hdr[hl++] = ' ';
    hdr[hl++] = '\n';
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    uint16_t hl16 = (uint16_t)hl; fwrite(&hl16, 2, 1, f);
    fwrite(hdr, 1, hl, f);
    fwrite(data, sizeof(int32_t), (size_t)N * C, f);
    fclose(f);
}

static void compare_and_print(const char *ref_label,
                              const float *ref_feats, const int32_t *ref_coords,
                              int ref_N, int ref_C,
                              const float *cuda_feats, const int32_t *cuda_coords,
                              int cuda_N, int cuda_C,
                              int *coord_mismatch_out,
                              double *corr_out, double *rel_out, double *max_abs_out) {
    int coord_mismatch = 0;
    if (cuda_N != ref_N || cuda_C != ref_C) {
        coord_mismatch = -1;
    } else {
        for (int i = 0; i < ref_N * 4; i++)
            if (cuda_coords[i] != ref_coords[i]) coord_mismatch++;
    }

    int total = (cuda_N == ref_N && cuda_C == ref_C) ? ref_N * ref_C : 0;
    double sr = 0, sc = 0, sr2 = 0, sc2 = 0, src = 0, sd2 = 0;
    double max_abs = 0;
    int max_idx = -1;
    for (int i = 0; i < total; i++) {
        double a = ref_feats[i], b = cuda_feats[i], d = a - b;
        sr += a; sc += b; sr2 += a * a; sc2 += b * b; src += a * b; sd2 += d * d;
        if (fabs(d) > max_abs) { max_abs = fabs(d); max_idx = i; }
    }
    double corr = 0, rel = 0;
    if (total > 0) {
        double mr = sr / total, mc = sc / total;
        double vr = sr2 / total - mr * mr;
        double vc = sc2 / total - mc * mc;
        corr = (vr > 0 && vc > 0) ? (src / total - mr * mc) / sqrt(vr * vc) : 0;
        rel = sqrt(sd2) / sqrt(sr2);
    }
    fprintf(stderr, "%s coord mismatches: %d\n", ref_label, coord_mismatch);
    fprintf(stderr, "%s correlation: %.8f\n", ref_label, corr);
    fprintf(stderr, "%s rel L2: %.8g\n", ref_label, rel);
    fprintf(stderr, "%s max abs: %.8g\n", ref_label, max_abs);
    if (max_idx >= 0 && ref_C > 0) {
        fprintf(stderr, "%s max idx: row=%d col=%d ref=%.8g cuda=%.8g diff=%.8g\n",
                ref_label, max_idx / ref_C, max_idx % ref_C,
                ref_feats[max_idx], cuda_feats[max_idx],
                cuda_feats[max_idx] - ref_feats[max_idx]);
    }
    if (total > 0) {
        fprintf(stderr, "%s[:4]: %.5f %.5f %.5f %.5f\n",
                ref_label, ref_feats[0], ref_feats[1], ref_feats[2], ref_feats[3]);
        fprintf(stderr, "CUDA[:4]: %.5f %.5f %.5f %.5f\n",
                cuda_feats[0], cuda_feats[1], cuda_feats[2], cuda_feats[3]);
    }
    if (g_project_out_w && g_project_out_ch > 0 &&
        coord_mismatch == 0 && ref_C == cuda_C && ref_C == 64) {
        double max_proj = 0.0;
        int max_row = -1, max_col = -1;
        double max_signed = 0.0;
        for (int r = 0; r < ref_N; r++) {
            for (int oc = 0; oc < g_project_out_ch; oc++) {
                double acc = 0.0;
                const float *w = g_project_out_w + (size_t)oc * ref_C;
                for (int c = 0; c < ref_C; c++) {
                    double d = (double)cuda_feats[(size_t)r * ref_C + c] -
                               (double)ref_feats[(size_t)r * ref_C + c];
                    acc += d * (double)w[c];
                }
                double a = fabs(acc);
                if (a > max_proj) {
                    max_proj = a;
                    max_row = r;
                    max_col = oc;
                    max_signed = acc;
                }
            }
        }
        fprintf(stderr, "%s projected output-delta max abs: %.8g\n",
                ref_label, max_proj);
        if (max_row >= 0) {
            fprintf(stderr, "%s projected output-delta max idx: row=%d col=%d diff=%.8g\n",
                    ref_label, max_row, max_col, max_signed);
            t2_project_contrib top[8] = {{0}};
            const float *w = g_project_out_w + (size_t)max_col * ref_C;
            for (int c = 0; c < ref_C; c++) {
                double d = (double)cuda_feats[(size_t)max_row * ref_C + c] -
                           (double)ref_feats[(size_t)max_row * ref_C + c];
                double v = d * (double)w[c];
                double av = fabs(v);
                for (int k = 0; k < 8; k++) {
                    if (av > top[k].abs_value) {
                        for (int m = 7; m > k; m--) top[m] = top[m - 1];
                        top[k].ch = c;
                        top[k].value = v;
                        top[k].abs_value = av;
                        break;
                    }
                }
            }
            fprintf(stderr, "%s projected top channels:", ref_label);
            for (int k = 0; k < 8 && top[k].abs_value > 0.0; k++) {
                fprintf(stderr, " c%d=%+.3g", top[k].ch, top[k].value);
            }
            fprintf(stderr, "\n");
        }
    }

    if (coord_mismatch_out) *coord_mismatch_out = coord_mismatch;
    if (corr_out) *corr_out = corr;
    if (rel_out) *rel_out = rel;
    if (max_abs_out) *max_abs_out = max_abs;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <decoder.st> <slat.npy> <coords.npy> "
                "[--ref-feats <ref.npy> --ref-coords <ref_coords.npy>] [--skip-cpu] "
                "[--start-stage <0..4> --start-block <block>]\n", argv[0]);
        return 1;
    }

    const char *ref_feats_path = NULL;
    const char *ref_coords_path = NULL;
    int skip_cpu = 0;
    int start_stage = -1;
    int start_block = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--ref-feats") && i + 1 < argc) {
            ref_feats_path = argv[++i];
        } else if (!strcmp(argv[i], "--ref-coords") && i + 1 < argc) {
            ref_coords_path = argv[++i];
        } else if (!strcmp(argv[i], "--skip-cpu")) {
            skip_cpu = 1;
        } else if (!strcmp(argv[i], "--start-stage") && i + 1 < argc) {
            start_stage = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--start-block") && i + 1 < argc) {
            start_block = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }
    if ((ref_feats_path && !ref_coords_path) || (!ref_feats_path && ref_coords_path)) {
        fprintf(stderr, "--ref-feats and --ref-coords must be provided together\n");
        return 1;
    }

    int nd, dd[8];
    float *slat = read_npy_f32(argv[2], &nd, dd);
    if (!slat || nd < 2) return 1;
    int N = dd[0], C = dd[1];
    int32_t *coords = read_npy_i32(argv[3], &nd, dd);
    if (!coords || nd < 2) return 1;
    fprintf(stderr, "Input: slat=[%d,%d], coords=[%d,%d]\n", N, C, dd[0], dd[1]);
    if (start_stage >= 0) {
        skip_cpu = 1;
        fprintf(stderr, "Start-from-intermediate: stage=%d block=%d input_C=%d\n",
                start_stage, start_block, C);
    }

    t2_shape_dec_result cpu_ref = {0};
    if (!skip_cpu) {
        t2_shape_dec *cpu_dec = t2_shape_dec_load(argv[1]);
        if (!cpu_dec) return 1;
        sp3d_tensor *tin = sp3d_create(coords, slat, N, C, 1);
        cpu_ref = t2_shape_dec_forward(cpu_dec, tin, 4);
        sp3d_free(tin);
        t2_shape_dec_free(cpu_dec);
        fprintf(stderr, "CPU:  N=%d C=%d\n", cpu_ref.N, cpu_ref.C);
    }

    float *ext_ref_feats = NULL;
    int32_t *ext_ref_coords = NULL;
    int ext_ref_N = 0, ext_ref_C = 0;
    t2_shape_dec *project_dec = NULL;
    if (ref_feats_path) {
        int rfd[8], rcd[8], rnd = 0, cnd = 0;
        ext_ref_feats = read_npy_f32(ref_feats_path, &rnd, rfd);
        ext_ref_coords = read_npy_i32(ref_coords_path, &cnd, rcd);
        if (!ext_ref_feats || !ext_ref_coords || rnd < 2 || cnd < 2) return 1;
        ext_ref_N = rfd[0];
        ext_ref_C = rfd[1];
        fprintf(stderr, "External ref: N=%d C=%d coords=[%d,%d]\n",
                ext_ref_N, ext_ref_C, rcd[0], rcd[1]);
    }
    if (getenv("T2_VERIFY_PROJECT_OUT")) {
        project_dec = t2_shape_dec_load(argv[1]);
        if (!project_dec || !project_dec->output_w) {
            fprintf(stderr, "Cannot load output_layer.weight for projection diagnostic\n");
            return 1;
        }
        g_project_out_w = project_dec->output_w;
        g_project_out_ch = project_dec->out_channels;
    }

    cuda_trellis2_runner *r = cuda_trellis2_init(0, 1);
    if (!r) return 1;
    if (cuda_trellis2_load_shape_decoder(r, argv[1]) != 0) return 1;
    float *cuda_feats = NULL;
    int32_t *cuda_coords = NULL;
    int cuda_N = 0, cuda_C = 0;
    if (start_stage >= 0) {
        if (cuda_trellis2_run_shape_decoder_from_alloc(r, slat, C, coords, N,
                                                       start_stage, start_block,
                                                       &cuda_feats, &cuda_coords,
                                                       &cuda_N, &cuda_C) != 0) return 1;
    } else {
        if (cuda_trellis2_run_shape_decoder_alloc(r, slat, coords, N,
                                                  &cuda_feats, &cuda_coords,
                                                  &cuda_N, &cuda_C) != 0) return 1;
    }
    fprintf(stderr, "CUDA: N=%d C=%d\n", cuda_N, cuda_C);
    {
        const char *dump_feats = getenv("T2_VERIFY_DUMP_FEATS");
        const char *dump_coords = getenv("T2_VERIFY_DUMP_COORDS");
        if (dump_feats && dump_feats[0]) write_npy_f32(dump_feats, cuda_feats, cuda_N, cuda_C);
        if (dump_coords && dump_coords[0]) write_npy_i32(dump_coords, cuda_coords, cuda_N, 4);
    }

    int best_mismatch = 0;
    double best_corr = 1.0;
    if (!skip_cpu) {
        int cm = 0;
        double corr = 0, rel = 0, max_abs = 0;
        compare_and_print("CPU", cpu_ref.feats, cpu_ref.coords, cpu_ref.N, cpu_ref.C,
                          cuda_feats, cuda_coords, cuda_N, cuda_C,
                          &cm, &corr, &rel, &max_abs);
        best_mismatch = cm;
        best_corr = corr;
    }
    if (ext_ref_feats) {
        if (!skip_cpu) {
            compare_and_print("CPU-vs-PyTorch", ext_ref_feats, ext_ref_coords,
                              ext_ref_N, ext_ref_C,
                              cpu_ref.feats, cpu_ref.coords,
                              cpu_ref.N, cpu_ref.C,
                              NULL, NULL, NULL, NULL);
        }
        int cm = 0;
        double corr = 0, rel = 0, max_abs = 0;
        compare_and_print("PyTorch", ext_ref_feats, ext_ref_coords, ext_ref_N, ext_ref_C,
                          cuda_feats, cuda_coords, cuda_N, cuda_C,
                          &cm, &corr, &rel, &max_abs);
        best_mismatch = cm;
        best_corr = corr;
    }

    free(slat); free(coords); free(cuda_feats); free(cuda_coords);
    free(ext_ref_feats); free(ext_ref_coords);
    t2_shape_dec_free(project_dec);
    t2_shape_dec_result_free(&cpu_ref);
    cuda_trellis2_free(r);
    return (best_mismatch == 0 && best_corr > 0.999) ? 0 : 1;
}
