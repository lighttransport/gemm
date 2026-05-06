/*
 * test_paint_pipeline.c - Top-level paint pipeline orchestrator (Phase 4.12).
 *
 * Drives all per-stage runners through the paint_stages.h opaque API. Each
 * runner is its own TU (paint_stage_*.c) so file-local helpers don't collide;
 * paint_runtime.c owns the SAFETENSORS_IMPLEMENTATION the heavy stages need.
 *
 * Stages wired:
 *   - view_maps     (paint_stage_view_maps)
 *   - dinov2g       (paint_stage_dinov2g)
 *   - unet          (paint_stage_unet, with cuda_paint_unipc.h)
 *   - vae           (paint_stage_vae)
 *   - back_project  (paint_stage_back_project)
 *
 * Subcommand-based CLI; each subcommand exercises one stage. The 'chain'
 * mode runs view_maps → dinov2g → unet → vae → back_project end-to-end,
 * borrowing per-view depth/visibility and text conditioning from a pyref
 * dump where the orchestrator can't yet synthesize them itself.
 *
 * Usage:
 *   ./test_paint_pipeline view-maps  <mesh.obj> <out_prefix> [res]
 *   ./test_paint_pipeline dinov2g    <weights> <input.npy> <out_prefix>
 *   ./test_paint_pipeline vae        <vae.safetensors> <latent.npy> <out.npy>
 *   ./test_paint_pipeline unet       <unet.safetensors> <ref_dir> [out.npy]
 *   ./test_paint_pipeline back-project <bp_refdir>
 *   ./test_paint_pipeline chain      <mesh.obj> <dinov2g.safetensors> \
 *                                     <dinov2g_input.npy> <unet.safetensors> \
 *                                     <unet_refdir> <vae.safetensors> \
 *                                     <bp_refdir> <outdir>
 */

#include "../cuew.h"
#include "paint_stages.h"
#include "cuda_paint_unipc.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../common/stb_image_write.h"

static void write_bake_png(const char *path, const float *bake,
                           const float *mask, int H, int W) {
    uint8_t *u8 = (uint8_t *)malloc((size_t)H * W * 3);
    for (int i = 0; i < H * W; i++) {
        int covered = mask && mask[i] <= 0.f;
        for (int c = 0; c < 3; c++) {
            float v = covered ? 1.f : bake[i*3+c];  /* magenta? keep white bg */
            if (v < 0) v = 0; if (v > 1) v = 1;
            u8[i*3+c] = (uint8_t)(v * 255.f + 0.5f);
        }
        if (covered) { u8[i*3+0]=255; u8[i*3+1]=0; u8[i*3+2]=255; }
    }
    int ok = stbi_write_png(path, W, H, 3, u8, W * 3);
    fprintf(stderr, "%s %s (%dx%d)\n", ok?"wrote":"FAILED", path, W, H);
    free(u8);
}

/* ===== shared .npy helpers ============================================== */

static void *npy_read(const char *path, int *out_nd, uint64_t *shape,
                       size_t *out_n, char *out_dt) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    char magic[6]; if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6)) { fclose(f); return NULL; }
    uint8_t ver[2]; if (fread(ver, 1, 2, f) != 2) { fclose(f); return NULL; }
    uint16_t hlen; if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return NULL; }
    char hdr[1024]; if (hlen >= sizeof(hdr) || fread(hdr, 1, hlen, f) != hlen) { fclose(f); return NULL; }
    hdr[hlen] = 0;
    const char *dp = strstr(hdr, "'descr':");
    if (dp) { out_dt[0] = dp[11]; out_dt[1] = dp[12]; out_dt[2] = dp[13]; out_dt[3] = 0; }
    else    { out_dt[0] = '<'; out_dt[1] = 'f'; out_dt[2] = '4'; out_dt[3] = 0; }
    const char *p = strstr(hdr, "'shape': ("); if (!p) { fclose(f); return NULL; }
    p += strlen("'shape': (");
    int nd = 0; uint64_t sh[8]; size_t total = 1;
    while (*p && *p != ')') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ')') break;
        char *end; uint64_t v = strtoull(p, &end, 10);
        sh[nd++] = v; total *= v; p = end;
    }
    *out_nd = nd; for (int i = 0; i < nd; i++) shape[i] = sh[i]; *out_n = total;
    int elsz = (out_dt[0] == 'i' && out_dt[1] == '8') ? 8 : 4;
    void *buf = malloc(total * elsz);
    if (fread(buf, elsz, total, f) != total) { free(buf); fclose(f); return NULL; }
    fclose(f); return buf;
}

static void npy_write_f32(const char *path, const float *data,
                           const int *sh, int nd) {
    FILE *f = fopen(path, "wb"); if (!f) return;
    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256], shape_s[128] = ""; size_t total = 1;
    for (int i = 0; i < nd; i++) { char tmp[32]; snprintf(tmp,sizeof(tmp),"%d, ",sh[i]); strcat(shape_s,tmp); total*=(size_t)sh[i]; }
    int hl = snprintf(hdr, sizeof(hdr), "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shape_s);
    int tot = 10 + hl + 1; int pad = ((tot+63)/64)*64 - tot;
    uint16_t header_len = (uint16_t)(hl + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), total, f);
    fclose(f);
}

/* ===== minimal .obj loader (positions + triangle indices only) ========== */

typedef struct { float *pos; int *tri; int n_verts, n_tris; } obj_mesh;

static int read_obj(const char *path, obj_mesh *m) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    int cap_v = 1<<14, cap_t = 1<<14;
    m->pos = (float *)malloc((size_t)cap_v * 3 * sizeof(float));
    m->tri = (int *)  malloc((size_t)cap_t * 3 * sizeof(int));
    m->n_verts = 0; m->n_tris = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        if (line[0]=='v' && line[1]==' ') {
            float x,y,z;
            if (sscanf(line+2, "%f %f %f", &x,&y,&z)==3) {
                if (m->n_verts >= cap_v) { cap_v *= 2; m->pos = (float *)realloc(m->pos, (size_t)cap_v*3*sizeof(float)); }
                m->pos[m->n_verts*3+0]=x; m->pos[m->n_verts*3+1]=y; m->pos[m->n_verts*3+2]=z; m->n_verts++;
            }
        } else if (line[0]=='f' && line[1]==' ') {
            int idx[3]={0,0,0}; const char *p = line+2; int k=0;
            while (*p && k<3) {
                while (*p==' '||*p=='\t') p++;
                if (!*p||*p=='\n') break;
                idx[k++]=atoi(p);
                while (*p && *p!=' ' && *p!='\t' && *p!='\n') p++;
            }
            if (k==3) {
                for (int i=0;i<3;i++) idx[i] = idx[i] < 0 ? m->n_verts + idx[i] : idx[i] - 1;
                if (m->n_tris >= cap_t) { cap_t *= 2; m->tri = (int *)realloc(m->tri, (size_t)cap_t*3*sizeof(int)); }
                m->tri[m->n_tris*3+0]=idx[0]; m->tri[m->n_tris*3+1]=idx[1]; m->tri[m->n_tris*3+2]=idx[2]; m->n_tris++;
            }
        }
    }
    fclose(f); return 0;
}

/* ===== subcommand: view-maps ============================================ */

static int cmd_view_maps(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "view-maps <mesh.obj> <out_prefix> [res]\n"); return 1; }
    const char *obj_path = argv[2];
    const char *prefix   = argv[3];
    int res = argc >= 5 ? atoi(argv[4]) : 512;
    obj_mesh m = {0};
    if (read_obj(obj_path, &m) != 0) return 1;
    fprintf(stderr, "[view-maps] %s: %d v, %d t\n", obj_path, m.n_verts, m.n_tris);
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    paint_stage_view_maps *vm = paint_stage_view_maps_create(dev, res);
    paint_stage_view_maps_set_mesh(vm, m.pos, m.n_verts, m.tri, m.n_tris);
    int N = 6; size_t per = (size_t)res * res * 3;
    CUdeviceptr d_n, d_p; cuMemAlloc(&d_n, N * per * sizeof(float)); cuMemAlloc(&d_p, N * per * sizeof(float));
    paint_stage_view_maps_render(vm, d_n, d_p, 0, 0, 0, NULL, NULL);
    float *h_n = malloc(N * per * sizeof(float)), *h_p = malloc(N * per * sizeof(float));
    cuMemcpyDtoH(h_n, d_n, N * per * sizeof(float));
    cuMemcpyDtoH(h_p, d_p, N * per * sizeof(float));
    int sh[4] = {N, res, res, 3}; char path[1024];
    snprintf(path, sizeof(path), "%s_normal.npy", prefix);   npy_write_f32(path, h_n, sh, 4);
    snprintf(path, sizeof(path), "%s_position.npy", prefix); npy_write_f32(path, h_p, sh, 4);
    fprintf(stderr, "[view-maps] wrote %s_{normal,position}.npy (6×%d²×3)\n", prefix, res);
    free(h_n); free(h_p); cuMemFree(d_n); cuMemFree(d_p);
    paint_stage_view_maps_destroy(vm); cuCtxDestroy(ctx);
    free(m.pos); free(m.tri); return 0;
}

/* ===== subcommand: dinov2g ============================================== */

static int cmd_dinov2g(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "dinov2g <weights> <input.npy> <out_prefix>\n"); return 1; }
    int nd; uint64_t sh[8]; size_t n; char dt[8];
    float *input = (float *)npy_read(argv[3], &nd, sh, &n, dt);
    if (!input) return 1;
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    paint_stage_dinov2g *s = paint_stage_dinov2g_create(dev, argv[2]);
    if (!s) return 1;
    float *out = (float *)malloc((size_t)257 * 1536 * sizeof(float));
    paint_stage_dinov2g_run(s, input, out);
    int osh[3] = {1, 257, 1536}; char p[1024];
    snprintf(p, sizeof(p), "%s_output.npy", argv[4]);
    npy_write_f32(p, out, osh, 3);
    fprintf(stderr, "[dinov2g] wrote %s\n", p);
    free(input); free(out);
    paint_stage_dinov2g_destroy(s); cuCtxDestroy(ctx); return 0;
}

/* ===== subcommand: vae ================================================== */

static int cmd_vae(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "vae <vae.safetensors> <latent.npy> <out.npy>\n"); return 1; }
    int nd; uint64_t sh[8]; size_t total; char dt[8];
    float *lat = (float *)npy_read(argv[3], &nd, sh, &total, dt);
    if (!lat) return 1;
    int B = 1, IH, IW;
    if (nd == 3 && sh[0] == 4) { IH = (int)sh[1]; IW = (int)sh[2]; }
    else if (nd == 4 && sh[1] == 4) { B = (int)sh[0]; IH = (int)sh[2]; IW = (int)sh[3]; }
    else { fprintf(stderr, "ERROR: expected [4,H,W] or [B,4,H,W]\n"); return 1; }
    int OH = IH * 8, OW = IW * 8;
    fprintf(stderr, "[vae] latent [%d,4,%d,%d] -> recon [%d,3,%d,%d]\n", B, IH, IW, B, OH, OW);
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    paint_stage_vae *vae = paint_stage_vae_create(dev, argv[2]);
    if (!vae) return 1;
    size_t cands[] = { (size_t)512*IH*IW, (size_t)512*(IH*2)*(IW*2), (size_t)512*(IH*4)*(IW*4),
                       (size_t)256*(IH*4)*(IW*4), (size_t)256*(IH*8)*(IW*8), (size_t)128*(IH*8)*(IW*8) };
    size_t max_n = 0; for (size_t i = 0; i < sizeof(cands)/sizeof(*cands); i++) if (cands[i]>max_n) max_n=cands[i];
    size_t attn_n = (size_t)512 * IH * IW;
    CUdeviceptr d_in, d_out, d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync;
    cuMemAlloc(&d_in, 4 * (size_t)IH * IW * sizeof(float));
    cuMemAlloc(&d_out, 3 * (size_t)OH * OW * sizeof(float));
    cuMemAlloc(&d_a, max_n*sizeof(float)); cuMemAlloc(&d_b, max_n*sizeof(float));
    cuMemAlloc(&d_t1, max_n*sizeof(float)); cuMemAlloc(&d_t2, max_n*sizeof(float));
    cuMemAlloc(&d_qnc, attn_n*sizeof(float)); cuMemAlloc(&d_knc, attn_n*sizeof(float));
    cuMemAlloc(&d_vnc, attn_n*sizeof(float)); cuMemAlloc(&d_ync, attn_n*sizeof(float));
    size_t in_per = (size_t)4*IH*IW, out_per = (size_t)3*OH*OW;
    float *out_buf = (float *)malloc((size_t)B * out_per * sizeof(float));
    for (int bi = 0; bi < B; bi++) {
        cuMemcpyHtoD(d_in, lat + bi*in_per, in_per*sizeof(float));
        paint_stage_vae_decode(vae, d_in, IH, IW, d_out, d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync);
        cuCtxSynchronize();
        cuMemcpyDtoH(out_buf + bi*out_per, d_out, out_per*sizeof(float));
    }
    if (B == 1) { int s3[3] = {3, OH, OW}; npy_write_f32(argv[4], out_buf, s3, 3); }
    else        { int s4[4] = {B, 3, OH, OW}; npy_write_f32(argv[4], out_buf, s4, 4); }
    fprintf(stderr, "[vae] wrote %s\n", argv[4]);
    free(out_buf); free(lat);
    paint_stage_vae_destroy(vae); cuCtxDestroy(ctx); return 0;
}

/* ===== subcommand: unet (UniPC loop, mirrors test_paint_unet_stage) ===== */

static int cmd_unet(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "unet <unet.safetensors> <ref_dir> [out.npy]\n"); return 1; }
    const char *unet_path = argv[2], *ref_dir = argv[3];
    const char *save_final = argc >= 5 ? argv[4] : NULL;
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    char path[512]; int nd; uint64_t sh[8]; size_t n; char dt[8];
#define LD(var, name) do { snprintf(path,sizeof(path),"%s/%s",ref_dir,name); var = npy_read(path,&nd,sh,&n,dt); if (!var){fprintf(stderr,"missing %s\n",path); return 1;} } while(0)
    float *en, *ep, *text_in, *ref_lat, *dino_in, *x0; int64_t *ts;
    LD(en, "in_embeds_normal.npy");
    LD(ep, "in_embeds_position.npy");
    LD(text_in, "in_encoder_hidden_states.npy");
    int M_text = (int)sh[2], cross_dim = (int)sh[3];
    LD(ref_lat, "in_ref_latents.npy");
    LD(dino_in, "in_dino_hidden_states.npy");
    int T_dino = (int)sh[1], C_dino_in = (int)sh[2];
    LD(x0, "loop_x0.npy");
    LD(ts, "loop_timesteps.npy");
    int N_steps = (int)n;
#undef LD

    paint_unet_config cfg = {
        .B_outer=1, .N_pbr=2, .N_gen=2, .N_ref=1, .H0=64, .W0=64,
        .M_text=M_text, .cross_dim=cross_dim, .T_dino=T_dino, .C_dino_in=C_dino_in
    };
    int Beff = cfg.B_outer * cfg.N_pbr * cfg.N_gen;
    size_t x_n = (size_t)Beff * 4 * cfg.H0 * cfg.W0;
    fprintf(stderr, "[unet] Beff=%d N_steps=%d M_text=%d cross=%d T_dino=%d\n",
            Beff, N_steps, M_text, cross_dim, T_dino);

    paint_stage_unet *u = paint_stage_unet_create(dev, unet_path, &cfg);
    if (!u) return 1;
    paint_stage_unet_set_conditioning(u, (float*)en, (float*)ep, (float*)text_in,
                                       (float*)ref_lat, (float*)dino_in);
    paint_stage_unet_run_dual(u);

    float *x = malloc(x_n*sizeof(float)), *np_buf = malloc(x_n*sizeof(float));
    memcpy(x, x0, x_n*sizeof(float));
    pu_unipc sch; pu_unipc_init(&sch, N_steps, x_n);
    for (int i = 0; i < N_steps; i++) {
        paint_stage_unet_run_step(u, (long long)ts[i], x, np_buf);
        pu_unipc_step(&sch, np_buf, x);
    }
    if (save_final) {
        const float inv = 1.0f / 0.18215f;
        for (size_t k = 0; k < x_n; k++) x[k] *= inv;
        int s4[4] = {Beff, 4, cfg.H0, cfg.W0};
        npy_write_f32(save_final, x, s4, 4);
        fprintf(stderr, "[unet] wrote final latent -> %s\n", save_final);
    }
    pu_unipc_free(&sch);
    free(x); free(np_buf);
    free(en); free(ep); free(text_in); free(ref_lat); free(dino_in); free(x0); free(ts);
    paint_stage_unet_destroy(u); cuCtxDestroy(ctx); return 0;
}

/* ===== subcommand: back-project ========================================= */

static int cmd_back_project(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "back-project <bp_refdir>\n"); return 1; }
    const char *refdir = argv[2];
    char path[1024]; int nd; uint64_t sh[8]; size_t nn; char dt[8];
#define LD(var, name) do { snprintf(path,sizeof(path),"%s/%s",refdir,name); var = npy_read(path,&nd,sh,&nn,dt); if (!var){fprintf(stderr,"missing %s\n",path); return 1;} } while(0)
    float *tex_pos, *proj, *bake_ref, *trust_ref; int *tex_cov;
    LD(tex_pos, "tex_pos.npy");      size_t n_tp = nn;
    LD(tex_cov, "tex_cov.npy");      size_t n_tc = nn;
    LD(proj,    "proj.npy");
    LD(bake_ref,  "bake_tex.npy");
    LD(trust_ref, "bake_trust.npy");
#undef LD
    (void)n_tp;
    int Htex = (int)sqrt((double)n_tc), Wtex = Htex, C = 3;
    float p00 = proj[0], p11 = proj[1];
    int N = 0; while (N < 32) {
        snprintf(path, sizeof(path), "%s/view_%d_image.npy", refdir, N);
        FILE *fp = fopen(path, "rb"); if (!fp) break; fclose(fp); N++;
    }
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);
    paint_stage_back_project *bp = paint_stage_back_project_create(dev, Htex, Wtex, C);
    paint_stage_back_project_set_atlas(bp, tex_pos, tex_cov);
    paint_stage_back_project_begin(bp);
    int Himg = 0, Wimg = 0;
    for (int v = 0; v < N; v++) {
        float *im, *de, *vi, *co, *w2c;
#define LDV(var, name) do { snprintf(path,sizeof(path),"%s/view_%d_%s.npy",refdir,v,name); var=(float*)npy_read(path,&nd,sh,&nn,dt); } while(0)
        LDV(im,"image"); if (Himg==0){ Himg=(int)sqrt((double)nn/C); Wimg=Himg; }
        LDV(de,"depth"); LDV(vi,"visible"); LDV(co,"cos"); LDV(w2c,"w2c");
#undef LDV
        paint_stage_back_project_add_view(bp, im, de, vi, co, w2c, Himg, Wimg, p00, p11);
        free(im); free(de); free(vi); free(co); free(w2c);
    }
    size_t tex_n = (size_t)Htex * Wtex;
    float *bake = malloc(tex_n*C*sizeof(float)), *mask = malloc(tex_n*sizeof(float));
    paint_stage_back_project_finalize(bp, bake, mask);
    int mm = 0; double mx = 0;
    for (size_t i = 0; i < tex_n; i++) {
        if ((mask[i]>0)!=(trust_ref[i]>0)) mm++;
        if (mask[i]>0||trust_ref[i]>0) for (int k=0;k<C;k++){double d=fabs(bake[i*C+k]-bake_ref[i*C+k]); if (d>mx) mx=d;}
    }
    fprintf(stderr, "[back-project] N=%d mask_mismatch=%d max_diff=%.3e\n", N, mm, mx);
    free(bake); free(mask); free(tex_pos); free(tex_cov); free(proj); free(bake_ref); free(trust_ref);
    paint_stage_back_project_destroy(bp); cuCtxDestroy(ctx);
    return (mm==0 && mx<1e-4) ? 0 : 1;
}

/* ===== subcommand: chain (full pipeline demo) =========================== */

static int cmd_chain(int argc, char **argv) {
    if (argc < 10) {
        fprintf(stderr,
            "chain <mesh.obj> <dinov2g.safetensors> <dinov2g_input.npy> "
            "<unet.safetensors> <unet_refdir> <vae.safetensors> "
            "<bp_refdir> <outdir>\n");
        return 1;
    }
    const char *mesh_path = argv[2];
    const char *dg_path   = argv[3];
    const char *dg_input  = argv[4];
    const char *unet_path = argv[5];
    const char *unet_ref  = argv[6];
    const char *vae_path  = argv[7];
    const char *bp_ref    = argv[8];
    const char *outdir    = argv[9];

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 1;
    cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    /* 1. view_maps */
    obj_mesh m = {0};
    if (read_obj(mesh_path, &m) != 0) return 1;
    fprintf(stderr, "[chain] mesh %s: %d v, %d t\n", mesh_path, m.n_verts, m.n_tris);
    paint_stage_view_maps *vm = paint_stage_view_maps_create(dev, 512);
    paint_stage_view_maps_set_mesh(vm, m.pos, m.n_verts, m.tri, m.n_tris);
    int N=6; size_t per = (size_t)512*512*3, per1 = (size_t)512*512;
    CUdeviceptr d_n, d_p, d_de, d_vi, d_co;
    cuMemAlloc(&d_n, N*per*sizeof(float)); cuMemAlloc(&d_p, N*per*sizeof(float));
    cuMemAlloc(&d_de, N*per1*sizeof(float)); cuMemAlloc(&d_vi, N*per1*sizeof(float));
    cuMemAlloc(&d_co, N*per1*sizeof(float));
    float w2c_all[6*16], proj_diag[2];
    paint_stage_view_maps_render(vm, d_n, d_p, d_de, d_vi, d_co, w2c_all, proj_diag);
    cuCtxSynchronize();
    fprintf(stderr, "[chain] view_maps: 6 views @ 512² rendered (incl. depth/vis/cos)\n");
    /* Snapshot rendered per-view depth/vis/cos to host for back_project. */
    float *all_depth = malloc(N*per1*sizeof(float));
    float *all_vis   = malloc(N*per1*sizeof(float));
    float *all_cos   = malloc(N*per1*sizeof(float));
    cuMemcpyDtoH(all_depth, d_de, N*per1*sizeof(float));
    cuMemcpyDtoH(all_vis,   d_vi, N*per1*sizeof(float));
    cuMemcpyDtoH(all_cos,   d_co, N*per1*sizeof(float));
    {   /* dump for inspection */
        int sd[3]={N,512,512}; char tp[512];
        snprintf(tp,sizeof(tp),"%s/chain_depth.npy",outdir);  npy_write_f32(tp,all_depth,sd,3);
        snprintf(tp,sizeof(tp),"%s/chain_visible.npy",outdir);npy_write_f32(tp,all_vis,sd,3);
        snprintf(tp,sizeof(tp),"%s/chain_cos.npy",outdir);    npy_write_f32(tp,all_cos,sd,3);
        snprintf(tp,sizeof(tp),"%s/chain_w2c.npy",outdir); int sw[2]={N,16};
        npy_write_f32(tp,w2c_all,sw,2);
        fprintf(stderr,"[chain] dumped depth/visible/cos/w2c (proj=%.4f,%.4f)\n",
                proj_diag[0],proj_diag[1]);
    }
    cuMemFree(d_n); cuMemFree(d_p); cuMemFree(d_de); cuMemFree(d_vi); cuMemFree(d_co);
    free(m.pos); free(m.tri);
    /* vm stays alive — needed below to transform tex_pos. */

    /* 2. dinov2g */
    int nd; uint64_t sh[8]; size_t nn; char dt[8];
    float *dg_in = (float*)npy_read(dg_input, &nd, sh, &nn, dt);
    if (!dg_in) return 1;
    paint_stage_dinov2g *dg = paint_stage_dinov2g_create(dev, dg_path);
    if (!dg) return 1;
    float *dino_h = malloc((size_t)257*1536*sizeof(float));
    paint_stage_dinov2g_run(dg, dg_in, dino_h);
    fprintf(stderr, "[chain] dinov2g: encoded ref image -> [1,257,1536]\n");
    free(dg_in);
    paint_stage_dinov2g_destroy(dg);

    /* 3. unet (uses unet_refdir for text/normal/position conditioning, our
     * dino_h for the image conditioning). */
    char path[512];
    float *en, *ep, *text_in, *ref_lat, *x0; int64_t *ts;
#define LD(var, name) do { snprintf(path,sizeof(path),"%s/%s",unet_ref,name); var = npy_read(path,&nd,sh,&nn,dt); if (!var){fprintf(stderr,"missing %s\n",path); return 1;} } while(0)
    LD(en,"in_embeds_normal.npy");
    LD(ep,"in_embeds_position.npy");
    LD(text_in,"in_encoder_hidden_states.npy");
    int M_text=(int)sh[2], cross_dim=(int)sh[3];
    LD(ref_lat,"in_ref_latents.npy");
    LD(x0,"loop_x0.npy");
    LD(ts,"loop_timesteps.npy"); int N_steps=(int)nn;
#undef LD
    paint_unet_config cfg = { .B_outer=1, .N_pbr=2, .N_gen=2, .N_ref=1, .H0=64, .W0=64,
                              .M_text=M_text, .cross_dim=cross_dim, .T_dino=257, .C_dino_in=1536 };
    int Beff = cfg.B_outer*cfg.N_pbr*cfg.N_gen;
    size_t x_n = (size_t)Beff*4*cfg.H0*cfg.W0;
    paint_stage_unet *u = paint_stage_unet_create(dev, unet_path, &cfg);
    if (!u) return 1;
    paint_stage_unet_set_conditioning(u, (float*)en, (float*)ep, (float*)text_in,
                                       (float*)ref_lat, dino_h);
    paint_stage_unet_run_dual(u);
    float *x = malloc(x_n*sizeof(float)), *np_buf = malloc(x_n*sizeof(float));
    memcpy(x, x0, x_n*sizeof(float));
    pu_unipc sch; pu_unipc_init(&sch, N_steps, x_n);
    for (int i = 0; i < N_steps; i++) {
        paint_stage_unet_run_step(u, (long long)ts[i], x, np_buf);
        pu_unipc_step(&sch, np_buf, x);
    }
    const float inv = 1.0f/0.18215f; for (size_t k=0;k<x_n;k++) x[k]*=inv;
    fprintf(stderr, "[chain] unet: %d UniPC steps -> %d latents\n", N_steps, Beff);
    snprintf(path, sizeof(path), "%s/chain_latents.npy", outdir);
    int s4[4]={Beff,4,cfg.H0,cfg.W0}; npy_write_f32(path, x, s4, 4);
    pu_unipc_free(&sch);
    free(en); free(ep); free(text_in); free(ref_lat); free(x0); free(ts);
    paint_stage_unet_destroy(u);
    free(np_buf); free(dino_h);

    /* 4. vae decode each latent */
    paint_stage_vae *vae = paint_stage_vae_create(dev, vae_path);
    if (!vae) return 1;
    int IH=cfg.H0, IW=cfg.W0, OH=IH*8, OW=IW*8;
    size_t cands[]={(size_t)512*IH*IW,(size_t)512*(IH*2)*(IW*2),(size_t)512*(IH*4)*(IW*4),
                    (size_t)256*(IH*4)*(IW*4),(size_t)256*(IH*8)*(IW*8),(size_t)128*(IH*8)*(IW*8)};
    size_t max_n=0; for(size_t i=0;i<sizeof(cands)/sizeof(*cands);i++) if(cands[i]>max_n) max_n=cands[i];
    size_t attn_n=(size_t)512*IH*IW;
    CUdeviceptr d_in,d_out,d_a,d_b,d_t1,d_t2,d_qnc,d_knc,d_vnc,d_ync;
    cuMemAlloc(&d_in,4*(size_t)IH*IW*sizeof(float));
    cuMemAlloc(&d_out,3*(size_t)OH*OW*sizeof(float));
    cuMemAlloc(&d_a,max_n*sizeof(float)); cuMemAlloc(&d_b,max_n*sizeof(float));
    cuMemAlloc(&d_t1,max_n*sizeof(float)); cuMemAlloc(&d_t2,max_n*sizeof(float));
    cuMemAlloc(&d_qnc,attn_n*sizeof(float)); cuMemAlloc(&d_knc,attn_n*sizeof(float));
    cuMemAlloc(&d_vnc,attn_n*sizeof(float)); cuMemAlloc(&d_ync,attn_n*sizeof(float));
    size_t in_per=(size_t)4*IH*IW, out_per=(size_t)3*OH*OW;
    float *views = malloc((size_t)Beff * out_per * sizeof(float));
    for (int bi = 0; bi < Beff; bi++) {
        cuMemcpyHtoD(d_in, x + bi*in_per, in_per*sizeof(float));
        paint_stage_vae_decode(vae, d_in, IH, IW, d_out, d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync);
        cuCtxSynchronize();
        cuMemcpyDtoH(views + bi*out_per, d_out, out_per*sizeof(float));
    }
    /* VAE outputs in [-1,1]; shift to [0,1] for back_project (matches
     * StableDiffusionPipeline.decode_latents: image = (image / 2 + 0.5).clamp(0,1)). */
    {
        size_t nv = (size_t)Beff * out_per;
        for (size_t k = 0; k < nv; k++) {
            float v = views[k] * 0.5f + 0.5f;
            if (v < 0.f) v = 0.f; if (v > 1.f) v = 1.f;
            views[k] = v;
        }
    }
    fprintf(stderr, "[chain] vae: %d latents -> %d RGB views @ %d²\n", Beff, Beff, OH);
    snprintf(path, sizeof(path), "%s/chain_views.npy", outdir);
    int sv[4]={Beff,3,OH,OW}; npy_write_f32(path, views, sv, 4);
    free(x); cuMemFree(d_in); cuMemFree(d_out);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_t1); cuMemFree(d_t2);
    cuMemFree(d_qnc); cuMemFree(d_knc); cuMemFree(d_vnc); cuMemFree(d_ync);
    paint_stage_vae_destroy(vae);

    /* 5. back_project — atlas (tex_pos/tex_cov) + bake oracles still come from
     * bp_refdir (UV unwrap is upstream of this stage). Per-view depth/visible/
     * cos/w2c are now produced by view_maps (above), and per-view RGB images
     * are the vae-decoded views. proj diagonal is reported by view_maps. */
    float *tex_pos, *bake_ref, *trust_ref; int *tex_cov;
#define LD2(var, name) do { snprintf(path,sizeof(path),"%s/%s",bp_ref,name); var = npy_read(path,&nd,sh,&nn,dt); if (!var){fprintf(stderr,"missing %s\n",path); return 1;} } while(0)
    LD2(tex_pos,"tex_pos.npy"); LD2(tex_cov,"tex_cov.npy");
    LD2(bake_ref,"bake_tex.npy"); LD2(trust_ref,"bake_trust.npy");
#undef LD2
    int Htex=(int)sqrt((double)nn), Wtex=Htex;
    int Nv = Beff < N ? Beff : N;  /* min(rendered views, decoded views) */
    /* Lift pyref tex_pos (raw mesh world frame) into our rendered coord frame
     * so view_maps' w2c is consistent with the texel positions. */
    int n_texels = (int)nn;
    float *tex_pos_xform = (float *)malloc((size_t)n_texels * 3 * sizeof(float));
    paint_stage_view_maps_apply_mesh_transform(vm, tex_pos, tex_pos_xform, n_texels);
    paint_stage_view_maps_destroy(vm);
    paint_stage_back_project *bp = paint_stage_back_project_create(dev, Htex, Wtex, 3);
    paint_stage_back_project_set_atlas(bp, tex_pos_xform, tex_cov);
    free(tex_pos_xform);
    paint_stage_back_project_begin(bp);
    int Himg = OH, Wimg = OW;  /* vae output */
    if (Himg != 512 || Wimg != 512) {
        fprintf(stderr, "[chain] WARN: vae output %dx%d != view_maps 512x512; "
                "depth/vis/cos resolution mismatch will skip all texels\n",
                Himg, Wimg);
    }
    for (int v = 0; v < Nv; v++) {
        paint_stage_back_project_add_view(bp,
            views + (size_t)v * out_per,
            all_depth + (size_t)v * per1,
            all_vis   + (size_t)v * per1,
            all_cos   + (size_t)v * per1,
            w2c_all + v * 16,
            Himg, Wimg, proj_diag[0], proj_diag[1]);
    }
    free(all_depth); free(all_vis); free(all_cos);
    size_t tex_n = (size_t)Htex*Wtex;
    float *bake = malloc(tex_n*3*sizeof(float)), *mask = malloc(tex_n*sizeof(float));
    paint_stage_back_project_finalize(bp, bake, mask);
    int sb[3]={Htex,Wtex,3}; snprintf(path,sizeof(path),"%s/chain_bake.npy",outdir);
    npy_write_f32(path, bake, sb, 3);
    snprintf(path, sizeof(path), "%s/chain_bake.png", outdir);
    write_bake_png(path, bake, mask, Htex, Wtex);
    int sm[2]={Htex,Wtex}; snprintf(path,sizeof(path),"%s/chain_mask.npy",outdir);
    npy_write_f32(path, mask, sm, 2);
    int mm=0; double mx=0;
    for (size_t i=0;i<tex_n;i++) {
        if ((mask[i]>0)!=(trust_ref[i]>0)) mm++;
        if (mask[i]>0||trust_ref[i]>0) for (int k=0;k<3;k++){double d=fabs(bake[i*3+k]-bake_ref[i*3+k]); if(d>mx)mx=d;}
    }
    fprintf(stderr, "[chain] back-project: N=%d Htex=%d mask_mismatch=%d max_diff=%.3e\n",
            Nv, Htex, mm, mx);
    free(bake); free(mask); free(tex_pos); free(tex_cov); free(bake_ref); free(trust_ref);
    free(views);
    paint_stage_back_project_destroy(bp);
    cuCtxDestroy(ctx);
    fprintf(stderr, "[chain] DONE — outputs in %s\n", outdir);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <subcommand> [args...]\n"
            "  view-maps    <mesh.obj> <out_prefix> [res]\n"
            "  dinov2g      <weights> <input.npy> <out_prefix>\n"
            "  vae          <vae.safetensors> <latent.npy> <out.npy>\n"
            "  unet         <unet.safetensors> <ref_dir> [out.npy]\n"
            "  back-project <bp_refdir>\n"
            "  chain        <mesh.obj> <dinov2g.safetensors> <dinov2g_input.npy> "
                          "<unet.safetensors> <unet_refdir> <vae.safetensors> "
                          "<bp_refdir> <outdir>\n", argv[0]);
        return 1;
    }
    const char *cmd = argv[1];
    if      (!strcmp(cmd, "view-maps"))    return cmd_view_maps(argc, argv);
    else if (!strcmp(cmd, "dinov2g"))      return cmd_dinov2g(argc, argv);
    else if (!strcmp(cmd, "vae"))          return cmd_vae(argc, argv);
    else if (!strcmp(cmd, "unet"))         return cmd_unet(argc, argv);
    else if (!strcmp(cmd, "back-project")) return cmd_back_project(argc, argv);
    else if (!strcmp(cmd, "chain"))        return cmd_chain(argc, argv);
    fprintf(stderr, "unknown subcommand: %s\n", cmd);
    return 1;
}
