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
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../common/stb_image_write.h"

#include "mesh_vertex_inpaint.h"
#include "../../common/safetensors.h"

extern int paint_xatlas_unwrap(
    const float *vtx_pos, int n_verts,
    const int   *tri_idx, int n_tris,
    int   **out_vmap, float **out_uvs, int **out_uv_idx,
    int    *out_n_uv_verts, int *out_atlas_w, int *out_atlas_h);

/* CPU UV-space triangle rasterizer.
 *
 * Mirrors dump_paint_back_project.extract_tex_position_dense: uv ([0,1] with
 * v already flipped) → ndc * 2 - 1, then for each triangle write per-texel
 * barycentric-interpolated vtx_pos_uv into tex_pos and a 0/1 cov mask.
 * uv_idx already indexes vtx_pos_uv (per-UV-vert) so no vmap gather here. */
static void uv_rasterize_atlas(const float *vtx_pos_uv, /* [U,3] */
                                const float *uvs_flipped, /* [U,2] */
                                const int   *uv_idx,   /* [F,3] */
                                int U, int F, int Htex, int Wtex,
                                float *tex_pos, /* [Htex,Wtex,3] */
                                int   *tex_cov  /* [Htex,Wtex]   */)
{
    (void)U;
    memset(tex_pos, 0, (size_t)Htex*Wtex*3*sizeof(float));
    memset(tex_cov, 0, (size_t)Htex*Wtex*sizeof(int));
    for (int t = 0; t < F; t++) {
        int ia = uv_idx[t*3+0], ib = uv_idx[t*3+1], ic = uv_idx[t*3+2];
        float ax = uvs_flipped[ia*2+0]*2.f-1.f, ay = uvs_flipped[ia*2+1]*2.f-1.f;
        float bx = uvs_flipped[ib*2+0]*2.f-1.f, by = uvs_flipped[ib*2+1]*2.f-1.f;
        float cx = uvs_flipped[ic*2+0]*2.f-1.f, cy = uvs_flipped[ic*2+1]*2.f-1.f;
        /* ndc → pixel: x = (ndc+1)/2 * W - 0.5, same for y. */
        float pax = (ax+1.f)*0.5f*Wtex - 0.5f, pay = (ay+1.f)*0.5f*Htex - 0.5f;
        float pbx = (bx+1.f)*0.5f*Wtex - 0.5f, pby = (by+1.f)*0.5f*Htex - 0.5f;
        float pcx = (cx+1.f)*0.5f*Wtex - 0.5f, pcy = (cy+1.f)*0.5f*Htex - 0.5f;
        float min_x = pax<pbx?pax:pbx; if (pcx<min_x) min_x=pcx;
        float max_x = pax>pbx?pax:pbx; if (pcx>max_x) max_x=pcx;
        float min_y = pay<pby?pay:pby; if (pcy<min_y) min_y=pcy;
        float max_y = pay>pby?pay:pby; if (pcy>max_y) max_y=pcy;
        int x0 = (int)floorf(min_x), x1 = (int)ceilf(max_x);
        int y0 = (int)floorf(min_y), y1 = (int)ceilf(max_y);
        if (x0<0) x0=0; if (y0<0) y0=0;
        if (x1>=Wtex) x1=Wtex-1; if (y1>=Htex) y1=Htex-1;
        float denom = (pby-pcy)*(pax-pcx) + (pcx-pbx)*(pay-pcy);
        if (fabsf(denom) < 1e-12f) continue;
        float inv = 1.f/denom;
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                float fx = (float)x, fy = (float)y;
                float w0 = ((pby-pcy)*(fx-pcx) + (pcx-pbx)*(fy-pcy)) * inv;
                float w1 = ((pcy-pay)*(fx-pcx) + (pax-pcx)*(fy-pcy)) * inv;
                float w2 = 1.f - w0 - w1;
                if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
                int idx = y*Wtex + x;
                if (tex_cov[idx]) continue; /* first writer wins */
                tex_cov[idx] = 1;
                for (int k = 0; k < 3; k++)
                    tex_pos[idx*3+k] = w0*vtx_pos_uv[ia*3+k]
                                     + w1*vtx_pos_uv[ib*3+k]
                                     + w2*vtx_pos_uv[ic*3+k];
            }
        }
    }
}

static int write_textured_obj(const char *obj_path, const char *mtl_path,
                              const char *png_basename,
                              const float *verts, int nv,
                              const float *uvs, int nu,
                              const int32_t *faces, int nf) {
    FILE *fp = fopen(obj_path, "w"); if (!fp) return 1;
    const char *mtl_base = strrchr(mtl_path, '/');
    mtl_base = mtl_base ? mtl_base + 1 : mtl_path;
    fprintf(fp, "mtllib %s\nusemtl paint\n", mtl_base);
    for (int i = 0; i < nv; i++)
        fprintf(fp, "v %.6f %.6f %.6f\n",
                verts[i*3+0], verts[i*3+1], verts[i*3+2]);
    for (int i = 0; i < nu; i++)
        fprintf(fp, "vt %.6f %.6f\n", uvs[i*2+0], uvs[i*2+1]);
    for (int i = 0; i < nf; i++) {
        int a=faces[i*3+0]+1, b=faces[i*3+1]+1, c=faces[i*3+2]+1;
        fprintf(fp, "f %d/%d %d/%d %d/%d\n", a,a, b,b, c,c);
    }
    fclose(fp);
    fp = fopen(mtl_path, "w"); if (!fp) return 1;
    fprintf(fp, "newmtl paint\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd %s\n",
            png_basename);
    fclose(fp);
    return 0;
}

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
    /* Snapshot normal/position RGB views to host so we can VAE-encode them
     * for unet conditioning (replaces the random in_embeds_{normal,position}
     * the wrapper dump previously supplied). [N,H,W,3] f32 in [0,1]. */
    float *all_nrm = malloc(N*per*sizeof(float));
    float *all_pos = malloc(N*per*sizeof(float));
    cuMemcpyDtoH(all_nrm, d_n, N*per*sizeof(float));
    cuMemcpyDtoH(all_pos, d_p, N*per*sizeof(float));
    {   int sd[4]={N,512,512,3}; char tp[512];
        snprintf(tp,sizeof(tp),"%s/chain_normal.npy",outdir);
        npy_write_f32(tp,all_nrm,sd,4);
        snprintf(tp,sizeof(tp),"%s/chain_position.npy",outdir);
        npy_write_f32(tp,all_pos,sd,4);
    }
    cuMemFree(d_n); cuMemFree(d_p); cuMemFree(d_de); cuMemFree(d_vi); cuMemFree(d_co);
    /* vm + m stay alive — needed below for xatlas unwrap, atlas raster,
     * and OBJ writeout. */

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

    /* 3. VAE-encode 2 selected views of normal/position into unet conditioning
     * latents [B=1,N_gen=2,4,64,64] (replaces synthetic random embeds the
     * wrapper dump previously supplied). text + ref_lat still come from
     * unet_refdir (CLIP text encoder + ref image VAE-encode not yet ported).
     * View selection: front (azim 0, idx 0) and back (azim 180, idx 2) of
     * the 6-view candidate set. */
    char path[512];
    float *text_in = NULL, *ref_lat = NULL;
    int M_text = 77, cross_dim = 1024;
    /* Load learned_text_clip_{albedo,mr} from the unet safetensors and stack
     * as [B=1, N_pbr=2, M_text=77, cross_dim=1024]. Production path:
     * pipeline.py:268-275 (use_learned_text_clip=True bypasses CLIP entirely;
     * encoder_hidden_states is just two learned [77,1024] params concatenated
     * along the N_pbr axis, repeated for batch_size=1). */
    {
        st_context *st = safetensors_open(unet_path);
        if (!st) { fprintf(stderr, "[chain] cannot open %s for text\n", unet_path); return 1; }
        const char *names[2] = {"learned_text_clip_albedo", "learned_text_clip_mr"};
        text_in = (float *)malloc((size_t)2 * 77 * 1024 * sizeof(float));
        for (int i = 0; i < 2; i++) {
            int idx = safetensors_find(st, names[i]);
            if (idx < 0) {
                fprintf(stderr, "[chain] %s missing in unet weights\n", names[i]);
                safetensors_close(st); return 1;
            }
            const char *dts = safetensors_dtype(st, idx);
            if (strcmp(dts, "F32") != 0) {
                fprintf(stderr, "[chain] %s dtype=%s (need F32)\n", names[i], dts);
                safetensors_close(st); return 1;
            }
            memcpy(text_in + (size_t)i * 77 * 1024,
                   safetensors_data(st, idx),
                   (size_t)77 * 1024 * sizeof(float));
        }
        safetensors_close(st);
        fprintf(stderr, "[chain] text: loaded learned_text_clip_{albedo,mr} -> [1,2,77,1024]\n");
    }
#define LD(var, name) do { snprintf(path,sizeof(path),"%s/%s",unet_ref,name); var = npy_read(path,&nd,sh,&nn,dt); if (!var){fprintf(stderr,"missing %s\n",path); return 1;} } while(0)
    LD(ref_lat,"in_ref_latents.npy");
#undef LD
    int N_gen_cfg = 6;
    {
        const char *e = getenv("CHAIN_N_GEN");
        if (e && atoi(e) > 0) N_gen_cfg = atoi(e);
        if (N_gen_cfg != 2 && N_gen_cfg != 6) {
            fprintf(stderr, "CHAIN_N_GEN must be 2 or 6 (got %d)\n", N_gen_cfg); return 1;
        }
    }
    paint_unet_config cfg = { .B_outer=1, .N_pbr=2, .N_gen=N_gen_cfg, .N_ref=1, .H0=64, .W0=64,
                              .M_text=M_text, .cross_dim=cross_dim, .T_dino=257, .C_dino_in=1536 };
    int Beff = cfg.B_outer*cfg.N_pbr*cfg.N_gen;
    size_t x_n = (size_t)Beff*4*cfg.H0*cfg.W0;

    /* Spin up VAE early so we can encode normal/position views right now.
     * Workspace buffers are sized for the larger of encode (512²×128 ch) and
     * decode (same): both peak at 128×512×512 floats. */
    paint_stage_vae *vae = paint_stage_vae_create(dev, vae_path);
    if (!vae) return 1;
    int IH=cfg.H0, IW=cfg.W0, OH=IH*8, OW=IW*8;
    size_t vae_cands[] = {
        (size_t)512*IH*IW, (size_t)512*(IH*2)*(IW*2), (size_t)512*(IH*4)*(IW*4),
        (size_t)256*(IH*4)*(IW*4), (size_t)256*(IH*8)*(IW*8),
        (size_t)128*(IH*8)*(IW*8)
    };
    size_t vae_max_n = 0;
    for (size_t i = 0; i < sizeof(vae_cands)/sizeof(*vae_cands); i++)
        if (vae_cands[i] > vae_max_n) vae_max_n = vae_cands[i];
    size_t vae_attn_n = (size_t)512*IH*IW;
    CUdeviceptr d_img, d_lat, d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync;
    cuMemAlloc(&d_img, 3*(size_t)OH*OW*sizeof(float));
    cuMemAlloc(&d_lat, 4*(size_t)IH*IW*sizeof(float));
    cuMemAlloc(&d_a, vae_max_n*sizeof(float));
    cuMemAlloc(&d_b, vae_max_n*sizeof(float));
    cuMemAlloc(&d_t1, vae_max_n*sizeof(float));
    cuMemAlloc(&d_t2, vae_max_n*sizeof(float));
    cuMemAlloc(&d_qnc, vae_attn_n*sizeof(float));
    cuMemAlloc(&d_knc, vae_attn_n*sizeof(float));
    cuMemAlloc(&d_vnc, vae_attn_n*sizeof(float));
    cuMemAlloc(&d_ync, vae_attn_n*sizeof(float));

    /* HWC[H,W,3]∈[0,1] → CHW[3,H,W]∈[-1,1] then VAE-encode → [4,64,64]
     * scaled by 0.18215 (diffusers scaling_factor). */
    int N_gen = cfg.N_gen;
    /* N_gen=2 picks front+back; N_gen=6 picks all 6 candidate views (matching
     * view_maps order: azim {0,90,180,270,0,180} elev {0,0,0,0,+90,-90}). */
    int sel_views_all[6] = {0, 1, 2, 3, 4, 5};
    int sel_views_2 [2] = {0, 2};
    const int *sel_views = (N_gen == 6) ? sel_views_all : sel_views_2;
    size_t img_per = 3*(size_t)OH*OW, lat_per = 4*(size_t)IH*IW;
    float *embeds_normal   = malloc((size_t)N_gen * lat_per * sizeof(float));
    float *embeds_position = malloc((size_t)N_gen * lat_per * sizeof(float));
    float *img_chw = malloc(img_per * sizeof(float));
    for (int kind = 0; kind < 2; kind++) {  /* 0=normal, 1=position */
        const float *src = kind ? all_pos : all_nrm;
        float *dst = kind ? embeds_position : embeds_normal;
        for (int gi = 0; gi < N_gen; gi++) {
            const float *vh = src + (size_t)sel_views[gi] * per;
            for (int y = 0; y < OH; y++)
                for (int xx = 0; xx < OW; xx++)
                    for (int c = 0; c < 3; c++)
                        img_chw[c*OH*OW + y*OW + xx] =
                            vh[(y*OW + xx)*3 + c] * 2.f - 1.f;
            cuMemcpyHtoD(d_img, img_chw, img_per * sizeof(float));
            paint_stage_vae_encode(vae, d_img, OH, OW, d_lat,
                                    d_a, d_b, d_t1, d_t2,
                                    d_qnc, d_knc, d_vnc, d_ync);
            cuCtxSynchronize();
            cuMemcpyDtoH(dst + (size_t)gi*lat_per, d_lat, lat_per*sizeof(float));
            for (size_t k = 0; k < lat_per; k++)
                dst[gi*lat_per + k] *= 0.18215f;
        }
    }
    free(all_nrm); free(all_pos);
    fprintf(stderr, "[chain] vae-encode: %d normal + %d position views -> embeds\n", N_gen, N_gen);

    /* ref_lat: VAE-encode the input ref image at 512² (matches production
     * encode_images: (x-0.5)*2 then VAE encode then *0.18215). Source comes
     * from CHAIN_REF_IMAGE env var (NPY [3,512,512] or [512,512,3], in [0,1]).
     * If absent, fall back to oracle in_ref_latents.npy. */
    {
        const char *ref_img_path = getenv("CHAIN_REF_IMAGE");
        if (ref_img_path) {
            int rnd = 0; size_t rsh[8] = {0}, rnn = 0; int rdt = 0;
            float *rimg = (float *)npy_read(ref_img_path, &rnd, rsh, &rnn, &rdt);
            if (!rimg || rnn != (size_t)3*OH*OW) {
                fprintf(stderr, "[chain] CHAIN_REF_IMAGE: bad shape (need 3*%d*%d, got %zu)\n",
                        OH, OW, rnn);
                return 1;
            }
            int is_hwc = (rnd == 3 && (int)rsh[2] == 3);
            for (int y = 0; y < OH; y++)
                for (int xx = 0; xx < OW; xx++)
                    for (int c = 0; c < 3; c++) {
                        float v = is_hwc ? rimg[(y*OW + xx)*3 + c]
                                          : rimg[c*OH*OW + y*OW + xx];
                        img_chw[c*OH*OW + y*OW + xx] = v * 2.f - 1.f;
                    }
            free(rimg);
            cuMemcpyHtoD(d_img, img_chw, img_per * sizeof(float));
            paint_stage_vae_encode(vae, d_img, OH, OW, d_lat,
                                    d_a, d_b, d_t1, d_t2,
                                    d_qnc, d_knc, d_vnc, d_ync);
            cuCtxSynchronize();
            ref_lat = (float *)malloc(lat_per * sizeof(float));
            cuMemcpyDtoH(ref_lat, d_lat, lat_per * sizeof(float));
            for (size_t k = 0; k < lat_per; k++) ref_lat[k] *= 0.18215f;
            fprintf(stderr, "[chain] vae-encode: ref image %s -> ref_lat\n", ref_img_path);
        }
    }
    free(img_chw);

    paint_stage_unet *u = paint_stage_unet_create(dev, unet_path, &cfg);
    if (!u) return 1;

    int N_steps = 30;
    {
        const char *e = getenv("CHAIN_STEPS");
        if (e && atoi(e) > 0) N_steps = atoi(e);
    }
    unsigned long long seed = 42;
    {
        const char *e = getenv("CHAIN_SEED");
        if (e) seed = strtoull(e, NULL, 10);
    }
    /* Classifier-free guidance. CHAIN_CFG: 0=off, 2=2-chunk uncond/full,
     * 3=3-chunk uncond/ref/full (default; matches production pipeline.py:660-688).
     * 3-chunk: noise = uncond + g*vs*(ref-uncond) + g*vs*(full-ref). 2-chunk:
     * noise = uncond + g*vs*(full-uncond). CHAIN_CFG_SCALE=<f> sets g (default 3.0). */
    int cfg_mode = 3; /* 0=off, 2=2-chunk, 3=3-chunk */
    float cfg_scale = 3.0f;
    {
        const char *e = getenv("CHAIN_CFG");
        if (e) {
            int v = atoi(e);
            cfg_mode = (v == 0) ? 0 : (v == 2 ? 2 : 3);
        }
        const char *g = getenv("CHAIN_CFG_SCALE");
        if (g) cfg_scale = (float)atof(g);
    }
    int do_cfg = (cfg_mode != 0);
    /* view_scale per row: cam_mapping(azim) where azim per view comes from
     * the 6-view candidate set {0,90,180,270,0,180}. cam_mapping: azim<90 →
     * azim/90+1; 90<=azim<330 → 2.0; else → -azim/90+5.0. For N_gen=2
     * sel_views={0,2}: vs={1.0, 2.0}. For N_gen=6: vs={1.0,2.0,2.0,2.0,1.0,2.0}.
     * Tiled across N_pbr=2 → length Beff. */
    float vs_per_row[12];
    {
        float per_view_vs[6] = {1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 2.0f};
        for (int p = 0; p < cfg.N_pbr; p++)
            for (int g = 0; g < N_gen; g++)
                vs_per_row[p * N_gen + g] = per_view_vs[sel_views[g]];
    }

    /* Zero conditioning buffers for the uncond pass. Shapes match real ones. */
    float *zero_en = NULL, *zero_ep = NULL, *zero_text = NULL,
          *zero_ref = NULL, *zero_dino = NULL;
    if (do_cfg) {
        zero_en   = calloc((size_t)cfg.N_gen * 4 * cfg.H0 * cfg.W0, sizeof(float));
        zero_ep   = calloc((size_t)cfg.N_gen * 4 * cfg.H0 * cfg.W0, sizeof(float));
        zero_text = calloc((size_t)cfg.N_pbr * cfg.M_text * cfg.cross_dim, sizeof(float));
        zero_ref  = calloc((size_t)cfg.N_ref * 4 * cfg.H0 * cfg.W0, sizeof(float));
        zero_dino = calloc((size_t)cfg.T_dino * cfg.C_dino_in, sizeof(float));
    }

    /* Pre-populate the RA cache for each CFG chunk. run_dual is the expensive
     * reference-branch UNet forward; conditioning per chunk is invariant
     * across timesteps, so we cache it once per chunk and reuse across steps.
     * set_conditioning inside the step loop is cheap (memcpy + 1 small linear
     * + 1 layernorm). */
    if (!do_cfg) {
        paint_stage_unet_set_chunk(u, 0);
        paint_stage_unet_set_conditioning(u, embeds_normal, embeds_position,
                                           (float*)text_in, (float*)ref_lat, dino_h);
        paint_stage_unet_run_dual(u);
    } else if (cfg_mode == 2) {
        paint_stage_unet_set_chunk(u, 0);
        paint_stage_unet_set_conditioning(u, zero_en, zero_ep, zero_text, zero_ref, zero_dino);
        paint_stage_unet_run_dual(u);
        paint_stage_unet_set_chunk(u, 1);
        paint_stage_unet_set_conditioning(u, embeds_normal, embeds_position,
                                           (float*)text_in, (float*)ref_lat, dino_h);
        paint_stage_unet_run_dual(u);
    } else {
        paint_stage_unet_set_chunk(u, 0);
        paint_stage_unet_set_conditioning(u, zero_en, zero_ep, zero_text, zero_ref, zero_dino);
        paint_stage_unet_run_dual(u);
        paint_stage_unet_set_chunk(u, 1);
        paint_stage_unet_set_conditioning(u, zero_en, zero_ep, zero_text,
                                           (float*)ref_lat, zero_dino);
        paint_stage_unet_run_dual(u);
        paint_stage_unet_set_chunk(u, 2);
        paint_stage_unet_set_conditioning(u, embeds_normal, embeds_position,
                                           (float*)text_in, (float*)ref_lat, dino_h);
        paint_stage_unet_run_dual(u);
    }
    float *x = malloc(x_n*sizeof(float)), *np_buf = malloc(x_n*sizeof(float));
    float *np_uncond = do_cfg ? malloc(x_n*sizeof(float)) : NULL;
    float *np_ref    = (cfg_mode == 3) ? malloc(x_n*sizeof(float)) : NULL;
    /* Box-Muller N(0,1) seeded with splitmix64 → xoshiro for determinism. */
    {
        unsigned long long s = seed ? seed : 1ULL;
        size_t i = 0;
        while (i < x_n) {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27; s *= 2685821657736338717ULL;
            unsigned long long u1 = s;
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27; s *= 2685821657736338717ULL;
            unsigned long long u2 = s;
            double r1 = ((u1 >> 11) + 1) * (1.0 / 9007199254740993.0);
            double r2 = ((u2 >> 11) + 1) * (1.0 / 9007199254740993.0);
            double mag = sqrt(-2.0 * log(r1));
            double a = 2.0 * 3.14159265358979323846 * r2;
            x[i++] = (float)(mag * cos(a));
            if (i < x_n) x[i++] = (float)(mag * sin(a));
        }
    }
    pu_unipc sch; pu_unipc_init(&sch, N_steps, x_n);
    fprintf(stderr, "[chain] unet: %d steps, seed=%llu, cfg=%dchunk scale=%.2f, ts=[",
            N_steps, seed, cfg_mode, cfg_scale);
    for (int i = 0; i < N_steps; i++)
        fprintf(stderr, "%lld%s", sch.timesteps[i], i+1<N_steps?",":"]\n");
    size_t spc = 4 * (size_t)cfg.H0 * cfg.W0;
    struct timespec _ts0, _ts1; clock_gettime(CLOCK_MONOTONIC, &_ts0);
    for (int i = 0; i < N_steps; i++) {
        struct timespec _tsa, _tsb, _tsc, _tsd;
        clock_gettime(CLOCK_MONOTONIC, &_tsa);
        if (cfg_mode == 3) {
            /* uncond: zero everything */
            paint_stage_unet_set_chunk(u, 0);
            paint_stage_unet_set_conditioning(u, zero_en, zero_ep, zero_text,
                                               zero_ref, zero_dino);
            paint_stage_unet_run_step(u, sch.timesteps[i], x, np_uncond);
            clock_gettime(CLOCK_MONOTONIC, &_tsb);
            /* ref: real ref_lat only; embeds/text/dino zero */
            paint_stage_unet_set_chunk(u, 1);
            paint_stage_unet_set_conditioning(u, zero_en, zero_ep, zero_text,
                                               (float*)ref_lat, zero_dino);
            paint_stage_unet_run_step(u, sch.timesteps[i], x, np_ref);
            clock_gettime(CLOCK_MONOTONIC, &_tsc);
            /* full: all real */
            paint_stage_unet_set_chunk(u, 2);
            paint_stage_unet_set_conditioning(u, embeds_normal, embeds_position,
                                               (float*)text_in, (float*)ref_lat, dino_h);
            paint_stage_unet_run_step(u, sch.timesteps[i], x, np_buf);
            clock_gettime(CLOCK_MONOTONIC, &_tsd);
            fprintf(stderr, "[chain] step %2d/%d: c0=%.2fs c1=%.2fs c2=%.2fs\n", i+1, N_steps,
                    (_tsb.tv_sec-_tsa.tv_sec)+(_tsb.tv_nsec-_tsa.tv_nsec)*1e-9,
                    (_tsc.tv_sec-_tsb.tv_sec)+(_tsc.tv_nsec-_tsb.tv_nsec)*1e-9,
                    (_tsd.tv_sec-_tsc.tv_sec)+(_tsd.tv_nsec-_tsc.tv_nsec)*1e-9);
            /* combine: np = u + g*vs*(r - u) + g*vs*(f - r) */
            for (int b = 0; b < Beff; b++) {
                float gv = cfg_scale * vs_per_row[b];
                float *uc = np_uncond + (size_t)b * spc;
                float *re = np_ref    + (size_t)b * spc;
                float *fl = np_buf    + (size_t)b * spc;
                for (size_t k = 0; k < spc; k++)
                    fl[k] = uc[k] + gv * (re[k] - uc[k]) + gv * (fl[k] - re[k]);
            }
        } else if (cfg_mode == 2) {
            paint_stage_unet_set_chunk(u, 0);
            paint_stage_unet_set_conditioning(u, zero_en, zero_ep, zero_text,
                                               zero_ref, zero_dino);
            paint_stage_unet_run_step(u, sch.timesteps[i], x, np_uncond);
            paint_stage_unet_set_chunk(u, 1);
            paint_stage_unet_set_conditioning(u, embeds_normal, embeds_position,
                                               (float*)text_in, (float*)ref_lat, dino_h);
            paint_stage_unet_run_step(u, sch.timesteps[i], x, np_buf);
            for (int b = 0; b < Beff; b++) {
                float gv = cfg_scale * vs_per_row[b];
                float *uc = np_uncond + (size_t)b * spc;
                float *fl = np_buf    + (size_t)b * spc;
                for (size_t k = 0; k < spc; k++)
                    fl[k] = uc[k] + gv * (fl[k] - uc[k]);
            }
        } else {
            paint_stage_unet_run_step(u, sch.timesteps[i], x, np_buf);
        }
        pu_unipc_step(&sch, np_buf, x);
    }
    clock_gettime(CLOCK_MONOTONIC, &_ts1);
    fprintf(stderr, "[chain] unet total: %.2fs (%d steps cfg=%d)\n",
            (_ts1.tv_sec-_ts0.tv_sec)+(_ts1.tv_nsec-_ts0.tv_nsec)*1e-9, N_steps, cfg_mode);
    const float inv = 1.0f/0.18215f; for (size_t k=0;k<x_n;k++) x[k]*=inv;
    fprintf(stderr, "[chain] unet: %d UniPC steps -> %d latents\n", N_steps, Beff);
    snprintf(path, sizeof(path), "%s/chain_latents.npy", outdir);
    int s4[4]={Beff,4,cfg.H0,cfg.W0}; npy_write_f32(path, x, s4, 4);
    pu_unipc_free(&sch);
    free(text_in); free(ref_lat);
    free(embeds_normal); free(embeds_position);
    paint_stage_unet_destroy(u);
    free(np_buf); free(np_uncond); free(np_ref); free(dino_h);
    free(zero_en); free(zero_ep); free(zero_text); free(zero_ref); free(zero_dino);

    /* 4. vae decode each latent — reuse the VAE workspace allocated above. */
    size_t in_per=(size_t)4*IH*IW, out_per=(size_t)3*OH*OW;
    float *views = malloc((size_t)Beff * out_per * sizeof(float));
    for (int bi = 0; bi < Beff; bi++) {
        cuMemcpyHtoD(d_lat, x + bi*in_per, in_per*sizeof(float));
        paint_stage_vae_decode(vae, d_lat, IH, IW, d_img, d_a, d_b, d_t1, d_t2, d_qnc, d_knc, d_vnc, d_ync);
        cuCtxSynchronize();
        cuMemcpyDtoH(views + bi*out_per, d_img, out_per*sizeof(float));
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
    free(x); cuMemFree(d_img); cuMemFree(d_lat);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_t1); cuMemFree(d_t2);
    cuMemFree(d_qnc); cuMemFree(d_knc); cuMemFree(d_vnc); cuMemFree(d_ync);
    paint_stage_vae_destroy(vae);

    /* 5. xatlas UV unwrap → tex_pos/tex_cov. Bake oracles (bake_tex/bake_trust)
     * are loaded from bp_refdir for diff stats only. Per-view depth/visible/
     * cos/w2c come from view_maps; per-view RGB are the vae-decoded views. */
    int Htex = 128, Wtex = 128;
    int *vmap = NULL, *uv_idx = NULL;
    float *uvs = NULL;
    int U = 0, atlas_w = 0, atlas_h = 0;
    if (paint_xatlas_unwrap(m.pos, m.n_verts, m.tri, m.n_tris,
                             &vmap, &uvs, &uv_idx, &U,
                             &atlas_w, &atlas_h) != 0) {
        fprintf(stderr, "[chain] xatlas unwrap failed\n"); return 1;
    }
    fprintf(stderr, "[chain] xatlas: %d uv_verts, atlas %dx%d (rasterizing to %dx%d)\n",
            U, atlas_w, atlas_h, Htex, Wtex);
    /* Per-UV-vert raw position via vmap, then apply set_mesh transform so the
     * atlas lives in the same coord frame as view_maps' rendered geometry. */
    float *vtx_pos_uv = malloc((size_t)U * 3 * sizeof(float));
    for (int i = 0; i < U; i++) {
        int src = vmap[i];
        vtx_pos_uv[i*3+0] = m.pos[src*3+0];
        vtx_pos_uv[i*3+1] = m.pos[src*3+1];
        vtx_pos_uv[i*3+2] = m.pos[src*3+2];
    }
    float *vtx_pos_uv_n = malloc((size_t)U * 3 * sizeof(float));
    paint_stage_view_maps_apply_mesh_transform(vm, vtx_pos_uv, vtx_pos_uv_n, U);
    /* Flip UV v for raster (matches MeshRender.set_mesh's vtx_uv[:,1]=1-v). */
    float *uvs_flip = malloc((size_t)U * 2 * sizeof(float));
    for (int i = 0; i < U; i++) {
        uvs_flip[i*2+0] = uvs[i*2+0];
        uvs_flip[i*2+1] = 1.f - uvs[i*2+1];
    }
    size_t tex_nn = (size_t)Htex*Wtex;
    float *tex_pos = malloc(tex_nn * 3 * sizeof(float));
    int   *tex_cov = malloc(tex_nn * sizeof(int));
    uv_rasterize_atlas(vtx_pos_uv_n, uvs_flip, uv_idx, U, m.n_tris,
                        Htex, Wtex, tex_pos, tex_cov);
    int n_cov = 0; for (size_t i = 0; i < tex_nn; i++) n_cov += tex_cov[i];
    fprintf(stderr, "[chain] uv-raster: %d / %zu texels covered\n", n_cov, tex_nn);
    { int sp[3]={Htex,Wtex,3}, sc[2]={Htex,Wtex};
      char p2[1024];
      snprintf(p2,sizeof(p2),"%s/chain_tex_pos.npy",outdir); npy_write_f32(p2,tex_pos,sp,3);
      float *tcf = malloc(tex_nn*sizeof(float)); for (size_t i=0;i<tex_nn;i++) tcf[i]=(float)tex_cov[i];
      snprintf(p2,sizeof(p2),"%s/chain_tex_cov.npy",outdir); npy_write_f32(p2,tcf,sc,2);
      free(tcf);
    }
    free(uvs_flip); free(vtx_pos_uv);
    /* Oracle bake for diff (optional). */
    float *bake_ref = NULL, *trust_ref = NULL;
    snprintf(path,sizeof(path),"%s/bake_tex.npy",bp_ref);
    bake_ref = npy_read(path,&nd,sh,&nn,dt);
    snprintf(path,sizeof(path),"%s/bake_trust.npy",bp_ref);
    trust_ref = npy_read(path,&nd,sh,&nn,dt);
    int Nv = Beff < N ? Beff : N;
    paint_stage_view_maps_destroy(vm);
    paint_stage_back_project *bp = paint_stage_back_project_create(dev, Htex, Wtex, 3);
    paint_stage_back_project_set_atlas(bp, tex_pos, tex_cov);
    paint_stage_back_project_begin(bp);
    int Himg = OH, Wimg = OW;  /* vae output */
    if (Himg != 512 || Wimg != 512) {
        fprintf(stderr, "[chain] WARN: vae output %dx%d != view_maps 512x512; "
                "depth/vis/cos resolution mismatch will skip all texels\n",
                Himg, Wimg);
    }
    /* views are [B,3,H,W] CHW; back_project_sample_f32 indexes as HWC. */
    float *view_hwc = malloc(out_per * sizeof(float));
    for (int v = 0; v < Nv; v++) {
        const float *src = views + (size_t)v * out_per;
        for (int y = 0; y < Himg; y++)
            for (int xx = 0; xx < Wimg; xx++)
                for (int c = 0; c < 3; c++)
                    view_hwc[(y*Wimg + xx)*3 + c] = src[c*Himg*Wimg + y*Wimg + xx];
        paint_stage_back_project_add_view(bp,
            view_hwc,
            all_depth + (size_t)v * per1,
            all_vis   + (size_t)v * per1,
            all_cos   + (size_t)v * per1,
            w2c_all + v * 16,
            Himg, Wimg, proj_diag[0], proj_diag[1]);
    }
    free(view_hwc);
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

    /* 6. inpaint + write textured.{obj,mtl,png} using our xatlas atlas. */
    {
        /* mesh_vertex_inpaint expects vtx_pos[V,3] indexed by faces[F,3] for
         * the position graph, and vtx_uv[U,2] indexed by uv_idx[F,3] for the
         * UV→pixel mapping. Use original (raw) per-vertex positions and
         * unflipped UVs (the inpaint helper does its own row=1-v flip). */
        uint8_t *m_u8 = malloc(tex_n);
        for (size_t i=0;i<tex_n;i++) m_u8[i] = mask[i]>0.f ? 255 : 0;
        float *out_tex = malloc(tex_n*3*sizeof(float));
        uint8_t *out_msk = malloc(tex_n);
        mesh_vertex_inpaint(bake, m_u8, m.pos, uvs, m.tri, uv_idx,
                            m.n_verts, U, m.n_tris, Htex, Wtex, 3, MVI_SMOOTH,
                            out_tex, out_msk);
        uint8_t *tex8 = malloc(tex_n*3);
        for (size_t i=0;i<tex_n*3;i++) {
            float v = out_tex[i]*255.f;
            if (v<0) v=0; if (v>255) v=255;
            tex8[i] = (uint8_t)(v+0.5f);
        }
        char p2[1024];
        snprintf(p2,sizeof(p2),"%s/textured.png",outdir);
        stbi_write_png(p2, Wtex, Htex, 3, tex8, Wtex*3);
        fprintf(stderr, "[chain] wrote %s\n", p2);

        /* OBJ: emit per-UV-vertex raw positions via vmap, unflipped UVs. */
        float *uv_verts = malloc((size_t)U*3*sizeof(float));
        for (int i=0;i<U;i++) {
            int src = vmap[i];
            uv_verts[i*3+0] = m.pos[src*3+0];
            uv_verts[i*3+1] = m.pos[src*3+1];
            uv_verts[i*3+2] = m.pos[src*3+2];
        }
        char po[1024], pm[1024];
        snprintf(po,sizeof(po),"%s/textured.obj",outdir);
        snprintf(pm,sizeof(pm),"%s/textured.mtl",outdir);
        if (write_textured_obj(po, pm, "textured.png",
                               uv_verts, U, uvs, U, uv_idx, m.n_tris) == 0)
            fprintf(stderr, "[chain] wrote %s + %s\n", po, pm);
        free(uv_verts);
        free(tex8); free(out_tex); free(out_msk); free(m_u8);
    }
    free(vmap); free(uvs); free(uv_idx); free(vtx_pos_uv_n);
    free(m.pos); free(m.tri);

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
