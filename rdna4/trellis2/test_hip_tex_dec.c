/*
 * test_hip_tex_dec.c - Full HIP TRELLIS.2 texture decoder (F32).
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
#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "hip_tex_dec_kernels.h"

#define SYNC() hipDeviceSynchronize()

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

typedef struct { hipFunction_t ins, conv, conv_tiled, conv_nmap, conv_nmap_tiled, ln, silu, gelu, add, lin, gather, resrep; } K;

static void hash_build(K *k, void *keys, void *vals, int cap_mask, void *coords, int N) {
    void *a[] = {&keys, &vals, &cap_mask, &coords, &N};
    hipModuleLaunchKernel(k->ins, (N+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); SYNC();
}
static void kspconv(K *k, void *out, void *in, void *co, void *w, void *b,
                    void *keys, void *vals, int cap_mask, int N, int inC, int outC) {
    void *a[] = {&out, &in, &co, &w, &b, &keys, &vals, &cap_mask, &inC, &outC};
    if ((outC % 64 == 0) && (inC % 32 == 0)) {
        hipModuleLaunchKernel(k->conv_tiled, N, outC / 64, 1, 64, 1, 1, 0, 0, a, NULL);
    } else {
        hipModuleLaunchKernel(k->conv, N, 1, 1, 256, 1, 1, 0, 0, a, NULL);
    }
    SYNC();
}
/* Nmap-driven variant: uses flex_gemm's precomputed neighbor_map (not coord hash).
 * Required for full checkpoint compatibility — weights trained against flex_gemm's
 * opaque neighbor enumeration. Fall back to tiled-hash path if nmap is NULL. */
static void kspconv_nmap(K *k, void *out, void *in, void *nmap, void *w, void *b,
                         void *co, void *keys, void *vals, int cap_mask,
                         int N, int inC, int outC) {
    if (nmap) {
        void *a[] = {&out, &in, &nmap, &w, &b, &inC, &outC};
        if ((outC % 64 == 0) && (inC % 32 == 0)) {
            hipModuleLaunchKernel(k->conv_nmap_tiled, N, outC / 64, 1, 64, 1, 1, 0, 0, a, NULL);
        } else {
            hipModuleLaunchKernel(k->conv_nmap, N, 1, 1, 256, 1, 1, 0, 0, a, NULL);
        }
        SYNC();
    } else {
        kspconv(k, out, in, co, w, b, keys, vals, cap_mask, N, inC, outC);
    }
}
static void kln(K *k, void *o, void *i, void *w, void *b, int N, int C, int hw, int hb) {
    float eps = 1e-6f;
    void *a[] = {&o, &i, &w, &b, &C, &eps, &hw, &hb};
    hipModuleLaunchKernel(k->ln, N, 1, 1, 256, 1, 1, 0, 0, a, NULL); SYNC();
}
static void ksilu(K *k, void *o, void *i, int n) {
    void *a[] = {&o, &i, &n};
    hipModuleLaunchKernel(k->silu, (n+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); SYNC();
}
static void kgelu(K *k, void *x, int n) {
    void *a[] = {&x, &n};
    hipModuleLaunchKernel(k->gelu, (n+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); SYNC();
}
static void kadd(K *k, void *x, void *y, int n) {
    void *a[] = {&x, &y, &n};
    hipModuleLaunchKernel(k->add, (n+255)/256, 1, 1, 256, 1, 1, 0, 0, a, NULL); SYNC();
}
static void klin(K *k, void *o, void *i, void *w, void *b, int N, int inC, int outC) {
    int gx = (outC+15)/16, gy = (N+15)/16;
    void *a[] = {&o, &i, &w, &b, &N, &inC, &outC};
    hipModuleLaunchKernel(k->lin, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL); SYNC();
}
static void kgather(K *k, void *hf, void *xf, void *hc, void *xc,
                    void *idx, void *si, int Nf, int Co, int Ci8) {
    int mx = Co > Ci8 ? Co : Ci8; int gy = (mx+255)/256;
    void *a[] = {&hf, &xf, &hc, &xc, &idx, &si, &Co, &Ci8};
    hipModuleLaunchKernel(k->gather, Nf, gy, 1, 256, 1, 1, 0, 0, a, NULL); SYNC();
}
static void kresrep(K *k, void *h, void *x, int N, int Co, int Ci8) {
    void *a[] = {&h, &x, &N, &Co, &Ci8};
    hipModuleLaunchKernel(k->resrep, N, 1, 1, 256, 1, 1, 0, 0, a, NULL); SYNC();
}

static void run_convnext(K *k, const t2sd_convnext *blk, int C, int N,
    void *d_feats, void *d_coords, void *d_keys, void *d_vals, int cap_mask,
    void *d_tmp, void *d_mlp, void *d_nmap) {
    void *dwc = hip_upload_raw(blk->conv_w, (size_t)C * 27 * C * sizeof(float));
    void *dwb = hip_upload_raw(blk->conv_b, (size_t)C * sizeof(float));
    void *dnw = hip_upload_raw(blk->norm_w, (size_t)C * sizeof(float));
    void *dnb = hip_upload_raw(blk->norm_b, (size_t)C * sizeof(float));
    void *dm0 = hip_upload_raw(blk->mlp0_w, (size_t)4*C*C*sizeof(float));
    void *dm0b= hip_upload_raw(blk->mlp0_b, (size_t)4*C*sizeof(float));
    void *dm2 = hip_upload_raw(blk->mlp2_w, (size_t)C*4*C*sizeof(float));
    void *dm2b= hip_upload_raw(blk->mlp2_b, (size_t)C*sizeof(float));
    kspconv_nmap(k, d_tmp, d_feats, d_nmap, dwc, dwb, d_coords, d_keys, d_vals, cap_mask, N, C, C);
    kln(k, d_tmp, d_tmp, dnw, dnb, N, C, 1, 1);
    klin(k, d_mlp, d_tmp, dm0, dm0b, N, C, 4*C);
    /* SparseConvNeXtBlock3d uses nn.SiLU (sparse_unet_vae.py:280), not GELU. */
    ksilu(k, d_mlp, d_mlp, N * 4 * C);
    klin(k, d_tmp, d_mlp, dm2, dm2b, N, 4*C, C);
    kadd(k, d_feats, d_tmp, N * C);
    hipFree(dwc); hipFree(dwb); hipFree(dnw); hipFree(dnb);
    hipFree(dm0); hipFree(dm0b); hipFree(dm2); hipFree(dm2b);
}

typedef struct { void *feats, *coords, *keys, *vals; int N, C, cap_mask; } DevSparse;

static void dump_dev_f32(const char *path, void *d, int N, int C) {
    float *h = (float *)malloc((size_t)N*C*sizeof(float));
    hipMemcpy(h, d, (size_t)N*C*sizeof(float), hipMemcpyDeviceToHost);
    FILE *f = fopen(path, "wb"); if (!f) { free(h); return; }
    char hdr[256]; int hl = snprintf(hdr, sizeof hdr,
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", N, C);
    while ((hl + 10) % 16 != 0) { hdr[hl++] = ' '; } hdr[hl++] = '\n'; hdr[hl] = 0;
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    uint16_t hl16 = (uint16_t)hl; fwrite(&hl16, 2, 1, f);
    fwrite(hdr, 1, hl, f); fwrite(h, sizeof(float), (size_t)N*C, f);
    fclose(f); free(h);
}

static DevSparse run_c2s(K *k, const t2sd_c2s *blk, int Nc,
    void *d_feats, void *d_coords, void *d_keys, void *d_vals, int cap_mask,
    const int64_t *idx, const int64_t *si, const int32_t *xcoords, int Nf,
    void *d_nmap_coarse, void *d_nmap_fine) {
    int Ci = blk->C_in, Co = blk->C_out, Ci8 = Ci/8, Cexp = Co*8;
    void *dn1w = hip_upload_raw(blk->norm1_w, (size_t)Ci*sizeof(float));
    void *dn1b = hip_upload_raw(blk->norm1_b, (size_t)Ci*sizeof(float));
    void *dc1w = hip_upload_raw(blk->conv1_w, (size_t)Cexp*27*Ci*sizeof(float));
    void *dc1b = hip_upload_raw(blk->conv1_b, (size_t)Cexp*sizeof(float));
    void *dc2w = hip_upload_raw(blk->conv2_w, (size_t)Co*27*Co*sizeof(float));
    void *dc2b = hip_upload_raw(blk->conv2_b, (size_t)Co*sizeof(float));
    void *dn=NULL, *de=NULL, *dhf=NULL, *dxf=NULL, *dhn=NULL, *dout=NULL;
    hipMalloc(&dn, (size_t)Nc*Ci*sizeof(float));
    hipMalloc(&de, (size_t)Nc*Cexp*sizeof(float));
    hipMalloc(&dhf,(size_t)Nf*Co*sizeof(float));
    hipMalloc(&dxf,(size_t)Nf*Ci8*sizeof(float));
    hipMalloc(&dhn,(size_t)Nf*Co*sizeof(float));
    hipMalloc(&dout,(size_t)Nf*Co*sizeof(float));
    kln(k, dn, d_feats, dn1w, dn1b, Nc, Ci, 1, 1);
    ksilu(k, dn, dn, Nc*Ci);
    if (getenv("HIP_C2S_DUMP")) dump_dev_f32("/tmp/hip_c2s_pre_conv1.npy", dn, Nc, Ci);
    kspconv_nmap(k, de, dn, d_nmap_coarse, dc1w, dc1b, d_coords, d_keys, d_vals, cap_mask, Nc, Ci, Cexp);
    if (getenv("HIP_C2S_DUMP")) dump_dev_f32("/tmp/hip_c2s_post_conv1.npy", de, Nc, Cexp);
    void *didx = hip_upload_raw(idx, (size_t)Nf*sizeof(int64_t));
    void *dsi  = hip_upload_raw(si,  (size_t)Nf*sizeof(int64_t));
    void *dxc  = hip_upload_raw(xcoords, (size_t)Nf*4*sizeof(int32_t));
    kgather(k, dhf, dxf, de, d_feats, didx, dsi, Nf, Co, Ci8);
    if (getenv("HIP_C2S_DUMP")) {
        dump_dev_f32("/tmp/hip_c2s_post_updown_h.npy", dhf, Nf, Co);
        dump_dev_f32("/tmp/hip_c2s_post_updown_x.npy", dxf, Nf, Ci8);
    }
    hipFree(dn); hipFree(de);
    hipFree(dn1w); hipFree(dn1b); hipFree(dc1w); hipFree(dc1b);
    hipFree(didx); hipFree(dsi);
    kln(k, dhn, dhf, NULL, NULL, Nf, Co, 0, 0);
    ksilu(k, dhn, dhn, Nf*Co);
    if (getenv("HIP_C2S_DUMP")) dump_dev_f32("/tmp/hip_c2s_pre_conv2.npy", dhn, Nf, Co);
    hipFree(dhf);
    int cap = 1; while (cap < Nf*2) cap <<= 1; int cm = cap - 1;
    void *fk=NULL, *fv=NULL;
    hipMalloc(&fk, (size_t)cap*sizeof(uint64_t));
    hipMalloc(&fv, (size_t)cap*sizeof(int32_t));
    hipMemset(fk, 0, (size_t)cap*sizeof(uint64_t));
    hipMemset(fv, 0xff, (size_t)cap*sizeof(int32_t));
    hash_build(k, fk, fv, cm, dxc, Nf);
    kspconv_nmap(k, dout, dhn, d_nmap_fine, dc2w, dc2b, dxc, fk, fv, cm, Nf, Co, Co);
    if (getenv("HIP_C2S_DUMP")) dump_dev_f32("/tmp/hip_c2s_post_conv2.npy", dout, Nf, Co);
    hipFree(dc2w); hipFree(dc2b); hipFree(dhn);
    kresrep(k, dout, dxf, Nf, Co, Ci8);
    hipFree(dxf);
    DevSparse ds = { dout, dxc, fk, fv, Nf, Co, cm };
    return ds;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <tex_dec.st> <feats.npy> <coords.npy>\n"
                "  [--cache <dir>] [--ref <npy>] [--full]\n"
                "  [--stop-stage <N>] [--stop-block <N>] [--stop-op <N>] [--after-c2s]\n"
                "  --full: run entire pipeline including output_layer.\n"
                "  --stop-stage N: stop AFTER ConvNeXt blocks of stage N (before its C2S);\n"
                "                  combine with --after-c2s to include stage N's C2S.\n", argv[0]);
        return 1;
    }
    const char *cache_dir = "/tmp/tex_knight_r512";
    const char *ref_path = NULL;
    int stop_stage = -1;
    int stop_block = -1;  /* stop after N blocks of stage 0 (requires stop_stage==0) */
    int stop_op = -1;     /* within block 0: 0=post-conv, 1=post-ln, 2=post-mlp */
    int after_c2s = 0;    /* stop AFTER stop_stage's C2S (instead of pre-c2s) */
    int run_full = 0;     /* --full: run whole pipeline + output_layer */
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--cache") && i+1<argc) cache_dir = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i+1<argc) ref_path = argv[++i];
        else if (!strcmp(argv[i], "--stop-stage") && i+1<argc) stop_stage = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stop-block") && i+1<argc) stop_block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stop-op") && i+1<argc) stop_op = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--after-c2s")) after_c2s = 1;
        else if (!strcmp(argv[i], "--full")) run_full = 1;
    }
    if (run_full) stop_stage = 99;  /* past n_stages so loop finishes all stages+C2S */

    t2_shape_dec *dec = t2_shape_dec_load(argv[1]);
    if (!dec) return 1;

    int fnd, fdd[8], cnd, cdd[8];
    float *slat = read_npy_f32(argv[2], &fnd, fdd);
    int N = fdd[0], slat_C = fnd>=2 ? fdd[1] : 1;
    int32_t *coords = read_npy_i32(argv[3], &cnd, cdd);

    int scales[4] = {16, 8, 4, 2};
    int64_t *gi[4]={0}, *gs[4]={0}; int32_t *gxc[4]={0}; int gN[4]={0};
    for (int s = 0; s < dec->n_stages && s < 4; s++) {
        char p[512]; int dn, d2[8];
        snprintf(p, sizeof p, "%s/cache_scale%d_c2s_idx.npy", cache_dir, scales[s]);
        gi[s] = read_npy_i64(p, &dn, d2); gN[s] = d2[0];
        snprintf(p, sizeof p, "%s/cache_scale%d_c2s_subidx.npy", cache_dir, scales[s]);
        /* subidx is saved as int32 in flex_gemm cache (sub.nonzero()[:, -1]).
         * Widen to int64 so the gather kernel's long long * arg reads right. */
        {
            int32_t *s32 = read_npy_i32(p, &dn, d2);
            int n_el = d2[0];
            int64_t *s64 = (int64_t *)malloc((size_t)n_el * sizeof(int64_t));
            for (int t = 0; t < n_el; t++) s64[t] = (int64_t)s32[t];
            free(s32);
            gs[s] = s64;
        }
        snprintf(p, sizeof p, "%s/cache_scale%d_c2s_x_coords.npy", cache_dir, scales[s]);
        gxc[s] = read_npy_i32(p, &dn, d2);
    }
    /* Per-stage flex_gemm neighbor_map dumps. stageS_convnext_nmap covers all
     * ConvNeXt convs at stage S + C2S conv1 (pre-C2S coords). stageS_post_c2s_nmap
     * covers C2S conv2 (post-C2S coords) which == stage(S+1)_convnext_nmap. */
    uint32_t *nmap_cn[4] = {0}; int nmap_cn_N[4] = {0};
    uint32_t *nmap_pc[4] = {0}; int nmap_pc_N[4] = {0};
    for (int s = 0; s < dec->n_stages && s < 4; s++) {
        char p[512]; int nd, dd[8];
        snprintf(p, sizeof p, "%s/stage%d_convnext_nmap.npy", cache_dir, s);
        FILE *fp = fopen(p, "rb");
        if (fp) {
            fclose(fp);
            uint32_t *buf = (uint32_t *)read_npy_i32(p, &nd, dd);
            nmap_cn[s] = buf; nmap_cn_N[s] = dd[0];
            fprintf(stderr, "loaded %s: (%d, %d)\n", p, dd[0], dd[1]);
        } else {
            fprintf(stderr, "missing nmap: %s (falling back to coord-hash)\n", p);
        }
        snprintf(p, sizeof p, "%s/stage%d_post_c2s_nmap.npy", cache_dir, s);
        fp = fopen(p, "rb");
        if (fp) {
            fclose(fp);
            uint32_t *buf = (uint32_t *)read_npy_i32(p, &nd, dd);
            nmap_pc[s] = buf; nmap_pc_N[s] = dd[0];
            fprintf(stderr, "loaded %s: (%d, %d)\n", p, dd[0], dd[1]);
        }
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != 0) return 1;
    hipSetDevice(0);
    hipModule_t mod;
    if (hip_compile_kernels(&mod, 0, hip_tex_dec_kernels_src, "tex_dec", 1, "HIP") <= 0) return 1;
    K k = {0};
    hipModuleGetFunction(&k.ins, mod, "hash_insert_kernel");
    hipModuleGetFunction(&k.conv, mod, "sparse_conv3d_f32");
    hipModuleGetFunction(&k.conv_tiled, mod, "sparse_conv3d_tiled_f32");
    hipModuleGetFunction(&k.conv_nmap, mod, "sparse_conv3d_nmap_f32");
    hipModuleGetFunction(&k.conv_nmap_tiled, mod, "sparse_conv3d_nmap_tiled_f32");
    hipModuleGetFunction(&k.ln, mod, "t2_layernorm_f32");
    hipModuleGetFunction(&k.silu, mod, "t2_silu_f32");
    hipModuleGetFunction(&k.gelu, mod, "t2_gelu_f32");
    hipModuleGetFunction(&k.add, mod, "t2_add_f32");
    hipModuleGetFunction(&k.lin, mod, "t2_linear_f32");
    hipModuleGetFunction(&k.gather, mod, "t2_c2s_gather_f32");
    hipModuleGetFunction(&k.resrep, mod, "t2_residual_repeat_f32");

    int C0 = dec->channels[0];
    void *d_slat = hip_upload_raw(slat, (size_t)N*slat_C*sizeof(float));
    void *d_flw = hip_upload_raw(dec->from_latent_w, (size_t)C0*slat_C*sizeof(float));
    void *d_flb = hip_upload_raw(dec->from_latent_b, (size_t)C0*sizeof(float));
    void *d_feats = NULL; hipMalloc(&d_feats, (size_t)N*C0*sizeof(float));
    klin(&k, d_feats, d_slat, d_flw, d_flb, N, slat_C, C0);
    hipFree(d_slat); hipFree(d_flw); hipFree(d_flb);

    void *d_coords = hip_upload_raw(coords, (size_t)N*4*sizeof(int32_t));
    int cap = 1; while (cap < N*2) cap <<= 1; int cap_mask = cap - 1;
    void *d_keys=NULL, *d_vals=NULL;
    hipMalloc(&d_keys, (size_t)cap*sizeof(uint64_t));
    hipMalloc(&d_vals, (size_t)cap*sizeof(int32_t));
    hipMemset(d_keys, 0, (size_t)cap*sizeof(uint64_t));
    hipMemset(d_vals, 0xff, (size_t)cap*sizeof(int32_t));
    hash_build(&k, d_keys, d_vals, cap_mask, d_coords, N);

    void *d_tmp=NULL, *d_mlp=NULL;
    hipMalloc(&d_tmp, (size_t)N*C0*sizeof(float));
    hipMalloc(&d_mlp, (size_t)N*4*C0*sizeof(float));

    void *d_nmap_cn[4] = {0};
    void *d_nmap_pc[4] = {0};
    for (int s = 0; s < dec->n_stages && s < 4; s++) {
        if (nmap_cn[s]) d_nmap_cn[s] = hip_upload_raw(nmap_cn[s], (size_t)nmap_cn_N[s]*27*sizeof(uint32_t));
        if (nmap_pc[s]) d_nmap_pc[s] = hip_upload_raw(nmap_pc[s], (size_t)nmap_pc_N[s]*27*sizeof(uint32_t));
    }

    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0, 0);

    int cur_N = N;
    /* stop_stage == -1 short-circuits right after from_latent for bisection. */
    for (int s = 0; s < dec->n_stages && stop_stage >= 0; s++) {
        int nc = dec->n_convnext[s]; int ch = dec->channels[s];
        fprintf(stderr, "stage %d: %d ConvNeXt(C=%d), N=%d\n", s, nc, ch, cur_N);
        for (int b = 0; b < nc; b++) {
            if (s == 0 && b == 0 && stop_op >= 0) {
                /* Inline block 0 with early exit after conv/ln/mlp. */
                const t2sd_convnext *blk = &dec->convnext[0][0];
                int C = ch;
                void *dwc = hip_upload_raw(blk->conv_w, (size_t)C*27*C*sizeof(float));
                void *dwb = hip_upload_raw(blk->conv_b, (size_t)C*sizeof(float));
                kspconv_nmap(&k, d_tmp, d_feats, d_nmap_cn[0], dwc, dwb, d_coords, d_keys, d_vals, cap_mask, cur_N, C, C);
                hipFree(dwc); hipFree(dwb);
                /* Copy d_tmp into d_feats so report path reads the right buffer. */
                hipMemcpy(d_feats, d_tmp, (size_t)cur_N*C*sizeof(float), hipMemcpyDeviceToDevice);
                if (stop_op == 0) goto done_stages;
                void *dnw = hip_upload_raw(blk->norm_w, (size_t)C*sizeof(float));
                void *dnb = hip_upload_raw(blk->norm_b, (size_t)C*sizeof(float));
                kln(&k, d_feats, d_feats, dnw, dnb, cur_N, C, 1, 1);
                hipFree(dnw); hipFree(dnb);
                if (stop_op == 1) goto done_stages;
                void *dm0 = hip_upload_raw(blk->mlp0_w, (size_t)4*C*C*sizeof(float));
                void *dm0b= hip_upload_raw(blk->mlp0_b, (size_t)4*C*sizeof(float));
                void *dm2 = hip_upload_raw(blk->mlp2_w, (size_t)C*4*C*sizeof(float));
                void *dm2b= hip_upload_raw(blk->mlp2_b, (size_t)C*sizeof(float));
                klin(&k, d_mlp, d_feats, dm0, dm0b, cur_N, C, 4*C);
                ksilu(&k, d_mlp, d_mlp, cur_N*4*C);
                klin(&k, d_feats, d_mlp, dm2, dm2b, cur_N, 4*C, C);
                hipFree(dm0); hipFree(dm0b); hipFree(dm2); hipFree(dm2b);
                if (stop_op == 2) goto done_stages;
                continue;  /* residual add skipped; op >= 3 == full block so fall through */
            }
            run_convnext(&k, &dec->convnext[s][b], ch, cur_N,
                d_feats, d_coords, d_keys, d_vals, cap_mask, d_tmp, d_mlp, d_nmap_cn[s]);
            if (s == stop_stage && stop_block >= 0 && b == stop_block) goto done_stages;
        }
        if (s == stop_stage && !after_c2s) break;
        if (dec->c2s[s].conv1_w) {
            fprintf(stderr, "  c2s %d->%d, N_fine=%d\n",
                dec->c2s[s].C_in, dec->c2s[s].C_out, gN[s]);
            /* coarse conv1 uses pre-c2s nmap (== stageS_convnext_nmap);
             * fine conv2 uses post-c2s nmap (== stage(S+1)_convnext_nmap). */
            void *d_fine = d_nmap_pc[s] ? d_nmap_pc[s]
                                        : (s+1 < 4 ? d_nmap_cn[s+1] : NULL);
            DevSparse ds = run_c2s(&k, &dec->c2s[s], cur_N,
                d_feats, d_coords, d_keys, d_vals, cap_mask,
                gi[s], gs[s], gxc[s], gN[s],
                d_nmap_cn[s], d_fine);
            hipFree(d_feats); hipFree(d_coords); hipFree(d_keys); hipFree(d_vals);
            d_feats = ds.feats; d_coords = ds.coords;
            d_keys = ds.keys; d_vals = ds.vals; cap_mask = ds.cap_mask; cur_N = ds.N;
            int nxt = (s+1<dec->n_stages) ? dec->channels[s+1] : ds.C;
            hipFree(d_tmp); hipMalloc(&d_tmp, (size_t)cur_N*nxt*sizeof(float));
            hipFree(d_mlp); hipMalloc(&d_mlp, (size_t)cur_N*4*nxt*sizeof(float));
            if (s == stop_stage && after_c2s) goto done_stages;
        }
    }
done_stages:;

    int Cf = dec->c2s[dec->n_stages-1].C_out;
    int out_ch = dec->out_channels;
    void *d_out = NULL;
    int have_out = 0;
    if (stop_stage >= dec->n_stages - 1 && !after_c2s) {
        kln(&k, d_feats, d_feats, NULL, NULL, cur_N, Cf, 0, 0);
        void *d_ow = hip_upload_raw(dec->output_w, (size_t)out_ch*Cf*sizeof(float));
        void *d_ob = hip_upload_raw(dec->output_b, (size_t)out_ch*sizeof(float));
        hipMalloc(&d_out, (size_t)cur_N*out_ch*sizeof(float));
        klin(&k, d_out, d_feats, d_ow, d_ob, cur_N, Cf, out_ch);
        hipFree(d_ow); hipFree(d_ob);
        have_out = 1;
    }

    hipEventRecord(e1, 0);
    hipEventSynchronize(e1);
    float t = 0; hipEventElapsedTime(&t, e0, e1);
    fprintf(stderr, "HIP tex_dec: %.1f s, N=%d\n", t/1000.0f, cur_N);

    int post_c2s_ch = (after_c2s && stop_stage >= 0 && stop_stage < dec->n_stages)
                      ? dec->c2s[stop_stage].C_out : 0;
    int out_C_report = have_out ? out_ch
        : (stop_stage < 0 ? C0
           : (post_c2s_ch ? post_c2s_ch
              : (stop_stage >= dec->n_stages-1 ? Cf : dec->channels[stop_stage])));
    void *src = have_out ? d_out : d_feats;
    float *h_out = (float *)malloc((size_t)cur_N*out_C_report*sizeof(float));
    hipMemcpy(h_out, src, (size_t)cur_N*out_C_report*sizeof(float), hipMemcpyDeviceToHost);

    /* Stats */
    double s_abs = 0, s_sq = 0; float mx = 0, mn = 0;
    for (size_t i = 0; i < (size_t)cur_N*out_C_report; i++) {
        float v = h_out[i];
        s_abs += fabs(v); s_sq += (double)v*v;
        if (v > mx) mx = v; if (v < mn) mn = v;
    }
    size_t total = (size_t)cur_N*out_C_report;
    fprintf(stderr, "HIP out: N=%d C=%d mean_abs=%.4f rms=%.4f min=%.3f max=%.3f\n",
            cur_N, out_C_report, s_abs/total, sqrt(s_sq/total), mn, mx);

    /* decode_tex_slat applies `raw * 0.5 + 0.5` before saving (see
     * trellis2_texturing.py:282). Mirror that here for ref comparison. */
    if (have_out) {
        for (size_t i = 0; i < total; i++) h_out[i] = h_out[i] * 0.5f + 0.5f;
    }

    if (ref_path) {
        int rn, rd[8];
        float *ref = read_npy_f32(ref_path, &rn, rd);
        if (ref && rd[0] == cur_N && rd[1] == out_C_report) {
            double sse=0, sref=0; float mx2=0;
            /* Per-channel stats to diagnose channel-specific drift. */
            double cse[16] = {0}, cref[16] = {0}; float cmx[16] = {0};
            int Cn = out_C_report < 16 ? out_C_report : 16;
            for (size_t i = 0; i < total; i++) {
                double dv = (double)h_out[i] - ref[i];
                sse+=dv*dv; sref+=(double)ref[i]*ref[i];
                float a=(float)fabs(dv); if (a>mx2) mx2=a;
                int c = (int)(i % out_C_report);
                if (c < Cn) {
                    cse[c] += dv*dv; cref[c] += (double)ref[i]*ref[i];
                    if (a > cmx[c]) cmx[c] = a;
                }
            }
            fprintf(stderr, "vs ref: rel=%.3e max=%.3e\n", sqrt(sse/(sref+1e-30)), mx2);
            for (int c = 0; c < Cn; c++) {
                fprintf(stderr, "  ch%d rel=%.3e max=%.3e\n", c,
                        sqrt(cse[c]/(cref[c]+1e-30)), cmx[c]);
            }
            fprintf(stderr, "HIP[0..5]: "); for (int i=0;i<6;i++) fprintf(stderr,"%+.3f ", h_out[i]);
            fprintf(stderr, "\nREF[0..5]: "); for (int i=0;i<6;i++) fprintf(stderr,"%+.3f ", ref[i]);
            fprintf(stderr, "\n");
        } else {
            fprintf(stderr, "ref shape [%d,%d] vs ours [%d,%d]\n", rd[0], rd[1], cur_N, out_C_report);
        }
        free(ref);
    }
    free(h_out);
    return 0;
}
