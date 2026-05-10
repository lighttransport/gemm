/*
 * test_hip_tex_dec.c - HIP shape_dec test harness (Milestone C refactor).
 *
 * Pipeline core lives in hip_shape_dec_pipeline.{h,c}; this driver handles
 * CLI parsing, npy I/O, optional --cache injection, mesh save, ref compare.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "../../common/trellis2_shape_decoder.h"
#include "../../common/trellis2_fdg_mesh.h"

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "hip_tex_dec_kernels.h"
#include "hip_shape_dec_pipeline.h"

static float *read_npy_f32(const char *p, int *nd, int *dd) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET); uint16_t hl; fread(&hl, 2, 1, f);
    char *h = malloc(hl + 1); fread(h, 1, hl, f); h[hl] = 0; *nd = 0;
    char *sp = strstr(h, "shape"); sp = sp ? strchr(sp, '(') : NULL;
    if (sp) { sp++; while (*sp && *sp != ')') {
        while (*sp == ' ' || *sp == ',') sp++;
        if (*sp == ')') break;
        dd[(*nd)++] = (int)strtol(sp, &sp, 10);
    } }
    size_t n = 1; for (int i = 0; i < *nd; i++) n *= dd[i];
    float *d = malloc(n * sizeof(float)); fread(d, sizeof(float), n, f);
    fclose(f); free(h); return d;
}
static int32_t *read_npy_i32(const char *p, int *nd, int *dd) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET); uint16_t hl; fread(&hl, 2, 1, f);
    char *h = malloc(hl + 1); fread(h, 1, hl, f); h[hl] = 0; *nd = 0;
    char *sp = strstr(h, "shape"); sp = sp ? strchr(sp, '(') : NULL;
    if (sp) { sp++; while (*sp && *sp != ')') {
        while (*sp == ' ' || *sp == ',') sp++;
        if (*sp == ')') break;
        dd[(*nd)++] = (int)strtol(sp, &sp, 10);
    } }
    size_t n = 1; for (int i = 0; i < *nd; i++) n *= dd[i];
    int32_t *d = malloc(n * sizeof(int32_t)); fread(d, sizeof(int32_t), n, f);
    fclose(f); free(h); return d;
}
static int64_t *read_npy_i64(const char *p, int *nd, int *dd) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 8, SEEK_SET); uint16_t hl; fread(&hl, 2, 1, f);
    char *h = malloc(hl + 1); fread(h, 1, hl, f); h[hl] = 0; *nd = 0;
    char *sp = strstr(h, "shape"); sp = sp ? strchr(sp, '(') : NULL;
    if (sp) { sp++; while (*sp && *sp != ')') {
        while (*sp == ' ' || *sp == ',') sp++;
        if (*sp == ')') break;
        dd[(*nd)++] = (int)strtol(sp, &sp, 10);
    } }
    size_t n = 1; for (int i = 0; i < *nd; i++) n *= dd[i];
    int64_t *d = malloc(n * sizeof(int64_t)); fread(d, sizeof(int64_t), n, f);
    fclose(f); free(h); return d;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <shape_dec.st> <feats.npy> <coords.npy>\n"
                "  [--cache <dir>] [--ref <npy>] [--full] [--save-mesh <obj>]\n", argv[0]);
        return 1;
    }
    const char *cache_dir = "/tmp/tex_knight_r512";
    const char *ref_path = NULL;
    const char *save_out_path = NULL;
    const char *save_mesh_path = NULL;
    /* Note: --full / --stop-* flags are accepted but ignored; pipeline always
     * runs full forward now. The test still parses them for back-compat. */
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--cache") && i+1<argc) cache_dir = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i+1<argc) ref_path = argv[++i];
        else if (!strcmp(argv[i], "--full")) { /* always full now */ }
        else if (!strcmp(argv[i], "--stop-stage") && i+1<argc) i++;
        else if (!strcmp(argv[i], "--stop-block") && i+1<argc) i++;
        else if (!strcmp(argv[i], "--stop-op") && i+1<argc) i++;
        else if (!strcmp(argv[i], "--after-c2s")) { /* ignored */ }
        else if (!strcmp(argv[i], "--save-out") && i+1<argc) save_out_path = argv[++i];
        else if (!strcmp(argv[i], "--save-mesh") && i+1<argc) save_mesh_path = argv[++i];
    }

    t2_shape_dec *dec = t2_shape_dec_load(argv[1]);
    if (!dec) return 1;

    int fnd, fdd[8], cnd, cdd[8];
    float *slat = read_npy_f32(argv[2], &fnd, fdd);
    int N = fdd[0], slat_C = fnd>=2 ? fdd[1] : 1;
    int32_t *coords = read_npy_i32(argv[3], &cnd, cdd);

    /* Load optional cache. If any file is missing, we silently fall through to
     * unguided synth for that stage. */
    hip_shape_dec_cache cache = {0};
    cache.n_stages = dec->n_stages;
    int scales[T2SD_MAX_STAGES] = {16, 8, 4, 2, 1};
    int have_any_cache = 0;
    for (int s = 0; s < dec->n_stages && s < T2SD_MAX_STAGES; s++) {
        char p[512]; int dn, d2[8];
        snprintf(p, sizeof p, "%s/cache_scale%d_c2s_idx.npy", cache_dir, scales[s]);
        FILE *probe = fopen(p, "rb");
        if (!probe) {
            fprintf(stderr, "missing cache: %s (stage %d unguided)\n", p, s);
            continue;
        }
        fclose(probe);
        cache.gi[s] = read_npy_i64(p, &dn, d2); cache.gN[s] = d2[0]; have_any_cache = 1;
        snprintf(p, sizeof p, "%s/cache_scale%d_c2s_subidx.npy", cache_dir, scales[s]);
        {
            int32_t *s32 = read_npy_i32(p, &dn, d2);
            int n_el = d2[0];
            int64_t *s64 = (int64_t *)malloc((size_t)n_el * sizeof(int64_t));
            for (int t = 0; t < n_el; t++) s64[t] = (int64_t)s32[t];
            free(s32);
            cache.gs[s] = s64;
        }
        snprintf(p, sizeof p, "%s/cache_scale%d_c2s_x_coords.npy", cache_dir, scales[s]);
        cache.gxc[s] = read_npy_i32(p, &dn, d2);
    }
    for (int s = 0; s < dec->n_stages && s < T2SD_MAX_STAGES; s++) {
        char p[512]; int nd, dd[8];
        snprintf(p, sizeof p, "%s/stage%d_convnext_nmap.npy", cache_dir, s);
        FILE *fp = fopen(p, "rb");
        if (fp) {
            fclose(fp);
            uint32_t *buf = (uint32_t *)read_npy_i32(p, &nd, dd);
            cache.nmap_cn[s] = buf; cache.nmap_cn_N[s] = dd[0]; have_any_cache = 1;
            fprintf(stderr, "loaded %s: (%d, %d)\n", p, dd[0], dd[1]);
        }
        snprintf(p, sizeof p, "%s/stage%d_post_c2s_nmap.npy", cache_dir, s);
        fp = fopen(p, "rb");
        if (fp) {
            fclose(fp);
            uint32_t *buf = (uint32_t *)read_npy_i32(p, &nd, dd);
            cache.nmap_pc[s] = buf; cache.nmap_pc_N[s] = dd[0];
            fprintf(stderr, "loaded %s: (%d, %d)\n", p, dd[0], dd[1]);
        }
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != 0) return 1;
    hipSetDevice(0);
    hipStream_t stream = NULL;
    if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
        fprintf(stderr, "T2-TEX: hipStreamCreateWithFlags failed; using default stream\n");
        stream = NULL;
    }
    hipModule_t mod;
    if (hip_compile_kernels(&mod, 0, hip_tex_dec_kernels_src, "tex_dec", 1, "HIP") <= 0) return 1;

    hip_shape_dec_ctx *ctx = hip_shape_dec_ctx_create(mod, stream, dec, 1);
    if (!ctx) return 1;

    float *d_out = NULL; int32_t *d_coords_out = NULL; int Nf = 0;
    int rc = hip_shape_dec_forward_ex(ctx, slat, coords, N, slat_C,
                                       have_any_cache ? &cache : NULL,
                                       &d_out, &d_coords_out, &Nf);
    if (rc != 0) { fprintf(stderr, "shape_dec forward failed\n"); return 1; }

    int out_ch = dec->out_channels;
    int cur_N = Nf;
    float *h_out = (float *)malloc((size_t)cur_N*out_ch*sizeof(float));
    if (stream) hipStreamSynchronize(stream);
    hipMemcpy(h_out, d_out, (size_t)cur_N*out_ch*sizeof(float), hipMemcpyDeviceToHost);

    /* Stats */
    double s_abs = 0, s_sq = 0; float mx = 0, mn = 0;
    for (size_t i = 0; i < (size_t)cur_N*out_ch; i++) {
        float v = h_out[i];
        s_abs += fabs(v); s_sq += (double)v*v;
        if (v > mx) mx = v; if (v < mn) mn = v;
    }
    size_t total = (size_t)cur_N*out_ch;
    fprintf(stderr, "HIP out: N=%d C=%d mean_abs=%.4f rms=%.4f min=%.3f max=%.3f\n",
            cur_N, out_ch, s_abs/total, sqrt(s_sq/total), mn, mx);

    if (save_mesh_path && out_ch == 7) {
        int32_t *h_coords4 = (int32_t *)malloc((size_t)cur_N * 4 * sizeof(int32_t));
        hipMemcpy(h_coords4, d_coords_out, (size_t)cur_N * 4 * sizeof(int32_t),
                  hipMemcpyDeviceToHost);
        float *raw = (float *)malloc((size_t)cur_N * 7 * sizeof(float));
        memcpy(raw, h_out, (size_t)cur_N * 7 * sizeof(float));
        float voxel_margin = 0.5f;
        for (int i = 0; i < cur_N; i++) {
            float *f = raw + (size_t)i * 7;
            for (int j = 0; j < 3; j++)
                f[j] = (1.0f + 2.0f*voxel_margin) / (1.0f + expf(-f[j])) - voxel_margin;
            f[6] = logf(1.0f + expf(f[6]));
        }
        int32_t *coords3 = (int32_t *)malloc((size_t)cur_N * 3 * sizeof(int32_t));
        for (int i = 0; i < cur_N; i++) {
            coords3[i*3+0] = h_coords4[i*4+1];
            coords3[i*3+1] = h_coords4[i*4+2];
            coords3[i*3+2] = h_coords4[i*4+3];
        }
        int max_coord = 0;
        for (size_t i = 0; i < (size_t)cur_N*3; i++)
            if (coords3[i] > max_coord) max_coord = coords3[i];
        float aabb[6] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
        float vs = (aabb[3] - aabb[0]) / (float)(max_coord + 1);
        fprintf(stderr, "Extracting mesh (voxel_size=%.4f, max_coord=%d)...\n",
                vs, max_coord);
        t2_fdg_mesh mesh = t2_fdg_to_mesh(coords3, raw, cur_N, vs, aabb);
        for (int i = 0; i < mesh.n_verts; i++) {
            float t = mesh.vertices[i*3+0];
            mesh.vertices[i*3+0] = mesh.vertices[i*3+2];
            mesh.vertices[i*3+2] = t;
        }
        for (int i = 0; i < mesh.n_tris; i++) {
            int t = mesh.triangles[i*3+1];
            mesh.triangles[i*3+1] = mesh.triangles[i*3+2];
            mesh.triangles[i*3+2] = t;
        }
        if (mesh.n_tris > 0) t2_fdg_write_obj(save_mesh_path, &mesh);
        fprintf(stderr, "HIP mesh: %d verts, %d tris -> %s\n",
                mesh.n_verts, mesh.n_tris, save_mesh_path);
        t2_fdg_mesh_free(&mesh);
        free(coords3); free(raw); free(h_coords4);
    }

    if (out_ch != 7) {
        for (size_t i = 0; i < total; i++) h_out[i] = h_out[i] * 0.5f + 0.5f;
    }

    if (ref_path) {
        int rn, rd[8];
        float *ref = read_npy_f32(ref_path, &rn, rd);
        if (ref && rd[0] == cur_N && rd[1] == out_ch) {
            double sse=0, sref=0; float mx2=0;
            for (size_t i = 0; i < total; i++) {
                double dv = (double)h_out[i] - ref[i];
                sse+=dv*dv; sref+=(double)ref[i]*ref[i];
                float a=(float)fabs(dv); if (a>mx2) mx2=a;
            }
            fprintf(stderr, "vs ref: rel=%.3e max=%.3e\n", sqrt(sse/(sref+1e-30)), mx2);
        } else {
            fprintf(stderr, "ref shape [%d,%d] vs ours [%d,%d]\n", rd[0], rd[1], cur_N, out_ch);
        }
        free(ref);
    }
    if (save_out_path) {
        FILE *f = fopen(save_out_path, "wb");
        if (f) {
            char hdr[256]; int hl = snprintf(hdr, sizeof hdr,
                "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",
                cur_N, out_ch);
            while ((hl + 10) % 16 != 0) { hdr[hl++] = ' '; } hdr[hl++] = '\n'; hdr[hl] = 0;
            fwrite("\x93NUMPY\x01\x00", 1, 8, f);
            uint16_t hl16 = (uint16_t)hl; fwrite(&hl16, 2, 1, f);
            fwrite(hdr, 1, hl, f);
            fwrite(h_out, sizeof(float), (size_t)cur_N*out_ch, f);
            fclose(f);
            fprintf(stderr, "saved HIP output to %s [%d,%d]\n", save_out_path, cur_N, out_ch);
        }
    }
    free(h_out);
    if (d_out) hipFree(d_out);
    if (d_coords_out) hipFree(d_coords_out);
    return 0;
}
