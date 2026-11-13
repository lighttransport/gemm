/*
 * cuda_cosmos3_runner.h - CUDA (NVRTC + cuBLAS) runner for nvidia/Cosmos3-Edge-Policy-DROID (Cosmos3OmniTransformer,
 * "cosmos3_edge_nemotron_dense": 28 Mixture-of-Transformers layers, understanding + generation tower).
 *
 *   #define CUDA_COSMOS3_IMPLEMENTATION   (once, after SAFETENSORS_IMPLEMENTATION)
 *   #include "cuda_cosmos3_runner.h"
 *
 * Design (see README.md):
 *  - bf16 activations/weights, cuBLAS bf16 GEMMs (fp32 accumulate), fp32 math inside kernels, bf16 rounding points as in torch.
 *  - The understanding (text) tower only sees text tokens with causal attention, so its hidden states / K / V are INDEPENDENT of the
 *    denoising step. It is run ONCE per prompt (cond + uncond), its weights are streamed layer by layer from the mmap'd shards (never
 *    resident), and only the per-layer K (after k_norm_und_for_gen + rope) / V are kept (14 MB). The generation tower (2.8 GB bf16) is
 *    the only resident transformer weight set; embed_tokens (0.5 GB) and lm_head (0.5 GB, unused by the policy) stay on the host.
 *  - Generation attention: gen tokens attend to [und K/V cache | own K/V] (non-causal, GQA 16/8, head_dim 128).
 *  - Host UniPC (order 2, bh2, flow prediction, lower_order_final) sampler, vision + action latents with separate scheduler state.
 */
#ifndef CUDA_COSMOS3_RUNNER_H
#define CUDA_COSMOS3_RUNNER_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cosmos3_runner cosmos3_runner;

/* dir: checkpoint root (containing transformer/) or the transformer directory itself. */
cosmos3_runner *cosmos3_load(const char *dir, int device, int verbose);
void cosmos3_free(cosmos3_runner *r);
void cosmos3_set_dump_dir(cosmos3_runner *r, const char *dir);   /* per-layer/per-step npy dumps (oracle names) */
void cosmos3_set_dump_steps(cosmos3_runner *r, int nsteps);      /* layer dumps for the first nsteps steps (default 1) */

/* Geometry of one policy call: latent [48][T][H][W] (H,W are the cropped latent dims, patch grid ceil(H/2) x ceil(W/2)), action chunk length, fps, domain id. */
int cosmos3_set_geometry(cosmos3_runner *r, int T, int H, int W, int chunk, float fps, int domain_id, int raw_action_dim);
/* Prompt slot 0 = cond, 1 = uncond: token ids (already template-tokenized, incl. trailing eos/vision_start). Builds embeddings + rope tables. */
int cosmos3_set_prompt(cosmos3_runner *r, int slot, const long long *ids, int n);
/* Run the understanding tower for the given slots (streams its weights once) and build the K/V caches. */
int cosmos3_prepare_und(cosmos3_runner *r, int nslots);
/* One generation-tower pass for prompt slot: vision latent fp32 [48][T][H][W], action fp32 [chunk][64], integer timestep t (1000-scale).
 * Outputs (fp32, may be NULL): velocity vision [48][T][H][W] (frame 0 zero), action prediction [chunk][64] (raw, unmasked). */
int cosmos3_gen_pass(cosmos3_runner *r, int slot, int step_tag, const float *lat, const float *act, int t, float *vel_v, float *vel_a);

typedef struct {
    int steps; float guidance; double flow_shift; int have_uncond;
} cosmos3_sample_cfg;
/* Full denoising loop (UniPC). lat_init: [48][T][H][W] with the clean conditioning frame 0 in place; act_init [chunk][64] noise (cols >= raw dim zero).
 * Outputs: act_out [chunk][64] (final action latents), lat_out [48][T][H][W] (final video latent). */
int cosmos3_sample(cosmos3_runner *r, const cosmos3_sample_cfg *c, const float *lat_init, const float *act_init, float *act_out, float *lat_out);

/* accounting */
typedef struct { double peak_alloc_mb; double weights_mb; double min_free_mb; double base_free_mb; double und_ms, gen_ms_last; double h2d_und_mb; } cosmos3_stats;
void cosmos3_get_stats(const cosmos3_runner *r, cosmos3_stats *s);
int cosmos3_timestep_for(const cosmos3_runner *r, int steps, int i, double flow_shift);

/* npy helpers (fp32) */
float *cosmos3_npy_read(const char *path, long long *n);
void cosmos3_npy_write(const char *path, const float *d, int nd, const long long *shape);

#ifdef __cplusplus
}
#endif
#endif

#ifdef CUDA_COSMOS3_IMPLEMENTATION
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#include <omp.h>
#include "../cuew.h"
#include "../cublasew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../../common/safetensors.h"
#include "cosmos3_kernels.h"
#include "../praxis/praxis_kernels.h"   /* reuse: block-sparse tensor-core flash attention (pr_flashiv128, interval masks) */

#define C3_NL 28
#define C3_D 2048
#define C3_NH 16
#define C3_NKV 8
#define C3_HD 128
#define C3_FF 9216
#define C3_LC 48
#define C3_PD 192
#define C3_ADIM 64
#define C3_VOCAB 131072

typedef struct { CUdeviceptr ln_in, q, k, v, out, ln_post, up, down, qn, kn, kng; } c3_layer;

typedef struct {
    int n, nund, kv_rows, ng_alloc;
    long long *ids;
    unsigned short *emb_host;      /* [nund][D] bf16 embeddings (re-uploaded at every prepare_und: the und pass updates und_h in place) */
    CUdeviceptr und_h;             /* [nund][D] bf16 hidden state of the und tower */
    CUdeviceptr cos_u, sin_u, cos_g, sin_g;   /* rope tables */
    CUdeviceptr iv_u, iv_g;                   /* flash interval tables [rows][6]: und causal [0,i+1), gen full [0,nk) */
    CUdeviceptr kc[C3_NL], vc[C3_NL];         /* K/V caches [nund+ngen][1024] (first nund rows filled by the und pass) */
    int ready;
} c3_prompt;

typedef struct { int have; float *mo[2]; float *last; int lower, this_order, idx; } c3_uni;

struct cosmos3_runner {
    int verbose, dev_i;
    CUdevice dev; CUcontext ctx; CUmodule mod, mod_pr; cublasew_context *cb; CUfunction f_fiv; int naive_attn;
    CUfunction f_rms, f_rmsh, f_rope, f_add, f_relu2, f_bias, f_addbias, f_addrow, f_patch, f_unpatch, f_f2bf, f_bf2f, f_attn, f_copy;
    st_context *st[2]; int nst;
    char dir[1024];
    /* resident */
    c3_layer gen[C3_NL];
    CUdeviceptr proj_in_w, proj_in_b, proj_out_w, proj_out_b, norm_gen, act_in_fc, act_in_b, act_out_fc, act_out_b, act_embed;
    const unsigned short *embed_host;          /* mmap'd embed_tokens [vocab][D] */
    const float *te_w1, *te_b1, *te_w2, *te_b2; /* fp32 time embedder (host, mmap) */
    /* und weight staging (one layer) */
    c3_layer und_slot[2]; CUstream copy_stream; CUevent ev_loaded[2], ev_done[2]; int pinned_maps;
    /* geometry */
    int T, H, W, HP, WP, nvis, chunk, ngen, domain, raw_adim; float fps;
    c3_prompt pr[2];
    /* scratch */
    int scr_rows;
    CUdeviceptr gen_h, bufX, bufQ, bufK, bufV, bufA, bufM, bufF, d_lat, d_tok, d_temb, d_vel, d_act32, d_actx, d_actbf, d_actf, d_pred;
    CUdeviceptr act_in_slice_w;
    /* accounting */
    size_t alloc_now, alloc_peak, weights_bytes; size_t base_free, min_free; double und_ms, gen_ms; double h2d_und;
    char dump_dir[1024]; int dump_steps;
    double inv_freq[64];
    int geom_ok;
};

#define C3_CK(x) do { CUresult _e = (x); if (_e != CUDA_SUCCESS) { const char *_s = "?"; cuGetErrorString(_e, &_s); fprintf(stderr, "cosmos3: CUDA error %s:%d: %s\n", __FILE__, __LINE__, _s); return -1; } } while (0)

static void c3_track_free(cosmos3_runner *r) { size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot); if (fr < r->min_free) r->min_free = fr; }
static CUdeviceptr c3_alloc(cosmos3_runner *r, size_t n) {
    CUdeviceptr p = 0;
    if (cuMemAlloc(&p, n ? n : 16) != CUDA_SUCCESS) { fprintf(stderr, "cosmos3: out of device memory (%.1f MB request)\n", n / 1048576.0); return 0; }
    r->alloc_now += n; if (r->alloc_now > r->alloc_peak) r->alloc_peak = r->alloc_now;
    c3_track_free(r);
    return p;
}
static void c3_free(cosmos3_runner *r, CUdeviceptr *p, size_t n) { if (*p) { cuMemFree(*p); *p = 0; r->alloc_now -= n; } }
static double c3_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }

static unsigned short c3_f2bf_h(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
static float c3_bf2f_h(unsigned short h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

/* ---- npy ---- */
float *cosmos3_npy_read(const char *path, long long *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    unsigned char m[10];
    if (fread(m, 1, 10, f) != 10 || memcmp(m, "\x93NUMPY", 6) || m[6] == 2) { fclose(f); return NULL; }
    int hl = m[8] | (m[9] << 8); char *h = (char *)calloc(1, hl + 1);
    if (fread(h, 1, hl, f) != (size_t)hl) { fclose(f); free(h); return NULL; }
    if (!strstr(h, "<f4")) { fprintf(stderr, "%s: only <f4 supported\n", path); fclose(f); free(h); return NULL; }
    char *p = strchr(strstr(h, "shape"), '(') + 1; long long cnt = 1;
    while (*p && *p != ')') { while (*p == ' ' || *p == ',') p++; if (*p == ')') break; cnt *= strtoll(p, &p, 10); }
    float *d = (float *)malloc(4 * (size_t)(cnt ? cnt : 1));
    size_t got = fread(d, 4, (size_t)cnt, f); fclose(f); free(h);
    if ((long long)got != cnt) { free(d); return NULL; }
    if (n) *n = cnt;
    return d;
}
void cosmos3_npy_write(const char *path, const float *d, int nd, const long long *shape) {
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "cosmos3: cannot write %s\n", path); return; }
    char hdr[256], sh[200] = ""; size_t n = 1;
    for (int i = 0; i < nd; i++) { char t[32]; snprintf(t, sizeof t, "%lld, ", shape[i]); strcat(sh, t); n *= shape[i]; }
    int hl = snprintf(hdr, sizeof hdr, "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", sh);
    int pad = (64 - (10 + hl + 1) % 64) % 64; unsigned short hlen = (unsigned short)(hl + pad + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f); fwrite(&hlen, 2, 1, f); fwrite(hdr, 1, hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f); fwrite(d, 4, n, f); fclose(f);
}
static void c3_dump_bf(cosmos3_runner *r, const char *name, CUdeviceptr d, long long rows, long long cols) {
    if (!r->dump_dir[0]) return;
    size_t n = (size_t)(rows * cols);
    unsigned short *h = (unsigned short *)malloc(2 * n); float *f = (float *)malloc(4 * n);
    cuCtxSynchronize(); cuMemcpyDtoH(h, d, 2 * n);
    for (size_t i = 0; i < n; i++) f[i] = c3_bf2f_h(h[i]);
    char p[1300]; snprintf(p, sizeof p, "%s/%s.npy", r->dump_dir, name);
    long long sh[2] = { rows, cols }; cosmos3_npy_write(p, f, cols > 0 ? 2 : 1, sh);
    free(h); free(f);
}
static void c3_dump_f(cosmos3_runner *r, const char *name, const float *f, int nd, const long long *sh) {
    if (!r->dump_dir[0]) return;
    char p[1300]; snprintf(p, sizeof p, "%s/%s.npy", r->dump_dir, name); cosmos3_npy_write(p, f, nd, sh);
}
void cosmos3_set_dump_dir(cosmos3_runner *r, const char *dir) {
    if (dir) { snprintf(r->dump_dir, sizeof r->dump_dir, "%s", dir); char b[1100]; snprintf(b, sizeof b, "%s", dir);
        for (char *p = b + 1; *p; p++) if (*p == '/') { *p = 0; mkdir(b, 0755); *p = '/'; } mkdir(b, 0755); }
    else r->dump_dir[0] = 0;
}
void cosmos3_set_dump_steps(cosmos3_runner *r, int n) { r->dump_steps = n; }

/* ---- weights ---- */
static int c3_find(cosmos3_runner *r, const char *name, int *which) {
    for (int s = 0; s < r->nst; s++) { int i = safetensors_find(r->st[s], name); if (i >= 0) { *which = s; return i; } }
    return -1;
}
static const void *c3_host(cosmos3_runner *r, const char *name, size_t *nbytes) {
    int w, i = c3_find(r, name, &w);
    if (i < 0) { fprintf(stderr, "cosmos3: missing tensor %s\n", name); return NULL; }
    if (nbytes) *nbytes = safetensors_nbytes(r->st[w], i);
    return safetensors_data(r->st[w], i);
}
static CUdeviceptr c3_upload(cosmos3_runner *r, const char *name, size_t expect_bytes, int count_weights) {
    size_t nb = 0; const void *h = c3_host(r, name, &nb);
    if (!h) return 0;
    if (expect_bytes && nb != expect_bytes) { fprintf(stderr, "cosmos3: %s size %zu != expected %zu\n", name, nb, expect_bytes); return 0; }
    CUdeviceptr d = c3_alloc(r, nb); if (!d) return 0;
    if (cuMemcpyHtoD(d, h, nb) != CUDA_SUCCESS) return 0;
    if (count_weights) r->weights_bytes += nb;
    return d;
}
static int c3_json_num(const char *txt, const char *key, double *v) {
    char k[128]; snprintf(k, sizeof k, "\"%s\"", key);
    const char *p = strstr(txt, k); if (!p) return -1;
    p = strchr(p, ':'); if (!p) return -1;
    *v = strtod(p + 1, NULL); return 0;
}

static int c3_load_layer_gen(cosmos3_runner *r, int l) {
    char n[160]; c3_layer *L = &r->gen[l];
#define UP(field, fmt, bytes) snprintf(n, sizeof n, "layers.%d." fmt, l); if (!(L->field = c3_upload(r, n, bytes, 1))) return -1;
    UP(ln_in, "input_layernorm_moe_gen.weight", 2 * C3_D)
    UP(q, "self_attn.add_q_proj.weight", 2ull * C3_D * C3_D)
    UP(k, "self_attn.add_k_proj.weight", 2ull * 1024 * C3_D)
    UP(v, "self_attn.add_v_proj.weight", 2ull * 1024 * C3_D)
    UP(qn, "self_attn.norm_added_q.weight", 2 * C3_HD)
    UP(kn, "self_attn.norm_added_k.weight", 2 * C3_HD)
    UP(out, "self_attn.to_add_out.weight", 2ull * C3_D * C3_D)
    UP(ln_post, "post_attention_layernorm_moe_gen.weight", 2 * C3_D)
    UP(up, "mlp_moe_gen.up_proj.weight", 2ull * C3_FF * C3_D)
    UP(down, "mlp_moe_gen.down_proj.weight", 2ull * C3_D * C3_FF)
#undef UP
    return 0;
}

/* ---- load ---- */
cosmos3_runner *cosmos3_load(const char *dir, int device, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) { fprintf(stderr, "cosmos3: cuewInit failed\n"); return NULL; }
    CU_CHECK_NULL(cuInit(0));
    cosmos3_runner *r = (cosmos3_runner *)calloc(1, sizeof *r);
    r->verbose = verbose; r->dump_steps = 1;
    CU_CHECK_NULL(cuDeviceGet(&r->dev, device));
    CU_CHECK_NULL(cuDevicePrimaryCtxRetain(&r->ctx, r->dev));
    CU_CHECK_NULL(cuCtxSetCurrent(r->ctx));
    { size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot); r->base_free = r->min_free = fr; }
    if (cublasewInit() != 0 || cublasewCreate(&r->cb, 0) != 0) { fprintf(stderr, "cosmos3: cuBLAS unavailable\n"); free(r); return NULL; }
    char path[1200];
    snprintf(path, sizeof path, "%s/transformer/config.json", dir);
    struct stat sb; if (stat(path, &sb) != 0) snprintf(r->dir, sizeof r->dir, "%s", dir); else snprintf(r->dir, sizeof r->dir, "%s/transformer", dir);
    snprintf(path, sizeof path, "%s/config.json", r->dir);
    { FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cosmos3: no %s\n", path); free(r); return NULL; }
      char buf[8192]; size_t n = fread(buf, 1, sizeof buf - 1, f); buf[n] = 0; fclose(f);
      double v; int bad = 0;
      if (c3_json_num(buf, "hidden_size", &v) || (int)v != C3_D) bad = 1;
      if (c3_json_num(buf, "num_hidden_layers", &v) || (int)v != C3_NL) bad = 1;
      if (c3_json_num(buf, "num_attention_heads", &v) || (int)v != C3_NH) bad = 1;
      if (c3_json_num(buf, "num_key_value_heads", &v) || (int)v != C3_NKV) bad = 1;
      if (c3_json_num(buf, "intermediate_size", &v) || (int)v != C3_FF) bad = 1;
      if (bad) { fprintf(stderr, "cosmos3: config.json does not match the compiled-in Cosmos3-Edge dims\n"); free(r); return NULL; }
      double theta = 1e8; c3_json_num(buf, "rope_theta", &theta);
      for (int j = 0; j < 64; j++) r->inv_freq[j] = 1.0 / pow(theta, (2.0 * j) / 128.0);
      /* torch computes inv_freq in fp32: round the same way */
      for (int j = 0; j < 64; j++) r->inv_freq[j] = (double)(1.0f / powf((float)theta, (float)(2 * j) / 128.0f));
    }
    /* shards */
    { char ip[1200]; snprintf(ip, sizeof ip, "%s/diffusion_pytorch_model.safetensors.index.json", r->dir);
      FILE *f = fopen(ip, "rb"); char shard[64][256]; int ns = 0;
      if (f) { fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); char *t = (char *)malloc(sz + 1); (void)!fread(t, 1, sz, f); t[sz] = 0; fclose(f);
        const char *p = t; while ((p = strstr(p, "diffusion_pytorch_model-"))) { char nm[256]; int k = 0; while (p[k] && p[k] != '"' && k < 255) { nm[k] = p[k]; k++; } nm[k] = 0; p += k;
            int dup = 0; for (int i = 0; i < ns; i++) if (!strcmp(shard[i], nm)) dup = 1; if (!dup && ns < 64) strcpy(shard[ns++], nm); }
        free(t); }
      else { snprintf(shard[0], 256, "diffusion_pytorch_model.safetensors"); ns = 1; }
      if (ns > 2) { fprintf(stderr, "cosmos3: more than 2 shards\n"); ns = 2; }
      for (int i = 0; i < ns; i++) { snprintf(ip, sizeof ip, "%s/%s", r->dir, shard[i]); r->st[i] = safetensors_open(ip); if (!r->st[i]) { fprintf(stderr, "cosmos3: cannot open %s\n", ip); free(r); return NULL; } }
      r->nst = ns; }
    if (cu_compile_kernels_ex(&r->mod, r->dev, cosmos3_kernel_src, "cosmos3.cu", verbose, "cosmos3", 0) < 0) { free(r); return NULL; }
    struct { CUfunction *f; const char *n; } fns[] = { { &r->f_rms, "c3_rmsnorm" }, { &r->f_rmsh, "c3_rmsnorm_heads" }, { &r->f_rope, "c3_rope" }, { &r->f_add, "c3_add" },
        { &r->f_relu2, "c3_relu2" }, { &r->f_bias, "c3_bias_f32" }, { &r->f_addbias, "c3_add_bias" }, { &r->f_addrow, "c3_add_row" }, { &r->f_patch, "c3_patchify" },
        { &r->f_unpatch, "c3_unpatchify" }, { &r->f_f2bf, "c3_f2bf_k" }, { &r->f_bf2f, "c3_bf2f_k" }, { &r->f_attn, "c3_attn" }, { &r->f_copy, "c3_copy2d" } };
    for (size_t i = 0; i < sizeof fns / sizeof fns[0]; i++)
        if (cuModuleGetFunction(fns[i].f, r->mod, fns[i].n) != CUDA_SUCCESS) { fprintf(stderr, "cosmos3: missing kernel %s\n", fns[i].n); free(r); return NULL; }
    { const char *e = getenv("COSMOS3_NAIVE_ATTN"); r->naive_attn = e && *e && *e != '0'; }
    if (cu_compile_kernels_ex(&r->mod_pr, r->dev, praxis_kernel_src, "praxis.cu", verbose, "praxis", 0) < 0 ||
        cuModuleGetFunction(&r->f_fiv, r->mod_pr, "pr_flashiv128") != CUDA_SUCCESS) { fprintf(stderr, "cosmos3: cannot load praxis flash kernel\n"); free(r); return NULL; }
    /* host-side tensors */
    size_t nb;
    r->embed_host = (const unsigned short *)c3_host(r, "embed_tokens.weight", &nb);
    r->te_w1 = (const float *)c3_host(r, "time_embedder.linear_1.weight", NULL); r->te_b1 = (const float *)c3_host(r, "time_embedder.linear_1.bias", NULL);
    r->te_w2 = (const float *)c3_host(r, "time_embedder.linear_2.weight", NULL); r->te_b2 = (const float *)c3_host(r, "time_embedder.linear_2.bias", NULL);
    if (!r->embed_host || !r->te_w1 || !r->te_b1 || !r->te_w2 || !r->te_b2) { free(r); return NULL; }
    /* resident small tensors */
    if (!(r->proj_in_w = c3_upload(r, "proj_in.weight", 2ull * C3_D * C3_PD, 1)) || !(r->proj_in_b = c3_upload(r, "proj_in.bias", 2 * C3_D, 1)) ||
        !(r->proj_out_w = c3_upload(r, "proj_out.weight", 2ull * C3_PD * C3_D, 1)) || !(r->proj_out_b = c3_upload(r, "proj_out.bias", 2 * C3_PD, 1)) ||
        !(r->norm_gen = c3_upload(r, "norm_moe_gen.weight", 2 * C3_D, 1)) || !(r->act_embed = c3_upload(r, "action_modality_embed", 2 * C3_D, 1))) { free(r); return NULL; }
    /* domain tables are 32 x (131072) bf16 = 8 MB each: upload whole, slice per domain */
    if (!(r->act_in_fc = c3_upload(r, "action_proj_in.fc.weight", 0, 1)) || !(r->act_in_b = c3_upload(r, "action_proj_in.bias.weight", 0, 1)) ||
        !(r->act_out_fc = c3_upload(r, "action_proj_out.fc.weight", 0, 1)) || !(r->act_out_b = c3_upload(r, "action_proj_out.bias.weight", 0, 1))) { free(r); return NULL; }
    double t0 = c3_now();
    for (int l = 0; l < C3_NL; l++) if (c3_load_layer_gen(r, l)) { fprintf(stderr, "cosmos3: generation tower load failed at layer %d\n", l); free(r); return NULL; }
    /* und staging: two one-layer device slots (double buffer) fed from the mmap'd shards on a copy stream */
    for (int sl = 0; sl < 2; sl++) {
        c3_layer *U = &r->und_slot[sl];
        if (!(U->ln_in = c3_alloc(r, 2 * C3_D)) || !(U->q = c3_alloc(r, 2ull * C3_D * C3_D)) || !(U->k = c3_alloc(r, 2ull * 1024 * C3_D)) || !(U->v = c3_alloc(r, 2ull * 1024 * C3_D)) ||
            !(U->kng = c3_alloc(r, 2 * C3_HD)) || !(U->out = c3_alloc(r, 2ull * C3_D * C3_D)) || !(U->ln_post = c3_alloc(r, 2 * C3_D)) || !(U->up = c3_alloc(r, 2ull * C3_FF * C3_D)) ||
            !(U->down = c3_alloc(r, 2ull * C3_D * C3_FF))) { free(r); return NULL; }
        cuEventCreate(&r->ev_loaded[sl], CU_EVENT_DISABLE_TIMING); cuEventCreate(&r->ev_done[sl], CU_EVENT_DISABLE_TIMING);
    }
    cuStreamCreate(&r->copy_stream, CU_STREAM_NON_BLOCKING);
    { /* pin the (file-backed, read-only) shard mappings so H2D DMA reads the page cache directly */
      const char *e = getenv("COSMOS3_NO_PIN");
      if (!(e && *e && *e != '0'))
          for (int i = 0; i < r->nst; i++)
              if (cuMemHostRegister(r->st[i]->map_base, r->st[i]->map_size, CU_MEMHOSTREGISTER_READ_ONLY) == CUDA_SUCCESS) r->pinned_maps++;
              else if (verbose) fprintf(stderr, "cosmos3: note: cuMemHostRegister(read-only file mapping) unsupported for shard %d; und weights stream from pageable memory\n", i);
    }
    if (verbose) fprintf(stderr, "cosmos3: loaded generation tower + heads: %.2f GB resident (%.1f s); und tower streams per layer (slot %.0f MB)\n",
                         r->weights_bytes / 1073741824.0, (c3_now() - t0) / 1e3, (2.0 * (2ull * C3_D * C3_D + 2ull * 1024 * C3_D + 2ull * C3_FF * C3_D)) / 1048576.0 * 2);
    return r;
}

void cosmos3_free(cosmos3_runner *r) { if (!r) return; /* process-exit teardown is left to the driver (context is retained) */ }

/* ---- geometry / prompts ---- */
static void c3_build_rope(const cosmos3_runner *r, const float *pos /*[3][n]*/, int n, unsigned short *cs, unsigned short *sn) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 64; j++) {
            int ax = (j < 60 && j % 3 == 1) ? 1 : (j < 60 && j % 3 == 2) ? 2 : 0;
            float f = (float)r->inv_freq[j] * pos[(size_t)ax * n + i];
            unsigned short c = c3_f2bf_h((float)cos((double)f)), s = c3_f2bf_h((float)sin((double)f));
            cs[(size_t)i * 128 + j] = cs[(size_t)i * 128 + j + 64] = c; sn[(size_t)i * 128 + j] = sn[(size_t)i * 128 + j + 64] = s;
        }
}

int cosmos3_set_geometry(cosmos3_runner *r, int T, int H, int W, int chunk, float fps, int domain, int raw_adim) {
    r->T = T; r->H = H; r->W = W; r->HP = (H + 1) / 2; r->WP = (W + 1) / 2; r->nvis = T * r->HP * r->WP; r->chunk = chunk; r->ngen = r->nvis + chunk;
    r->fps = fps; r->domain = domain; r->raw_adim = raw_adim; r->geom_ok = 1;
    return 0;
}

static int c3_scratch(cosmos3_runner *r, int rows) {
    if (rows <= r->scr_rows) return 0;
    CUdeviceptr *bufs[] = { &r->bufX, &r->bufQ, &r->bufK, &r->bufV, &r->bufA, &r->bufM, &r->bufF };
    size_t sz_old[] = { 2ull * C3_D, 2ull * C3_D, 2ull * 1024, 2ull * 1024, 2ull * C3_D, 2ull * C3_FF, 4ull * C3_D };
    for (int i = 0; i < 7; i++) { c3_free(r, bufs[i], sz_old[i] * r->scr_rows); if (!(*bufs[i] = c3_alloc(r, sz_old[i] * rows))) return -1; }
    r->scr_rows = rows;
    return 0;
}

static void c3_prompt_release(cosmos3_runner *r, c3_prompt *P) {
    if (!P->und_h) return;
    c3_free(r, &P->und_h, 2ull * P->nund * C3_D);
    for (int l = 0; l < C3_NL; l++) { c3_free(r, &P->kc[l], 2ull * P->kv_rows * 1024); c3_free(r, &P->vc[l], 2ull * P->kv_rows * 1024); }
    c3_free(r, &P->cos_u, 2ull * P->nund * 128); c3_free(r, &P->sin_u, 2ull * P->nund * 128);
    c3_free(r, &P->cos_g, 2ull * P->ng_alloc * 128); c3_free(r, &P->sin_g, 2ull * P->ng_alloc * 128);
    c3_free(r, &P->iv_u, 24ull * P->nund); c3_free(r, &P->iv_g, 24ull * P->ng_alloc);
}
int cosmos3_set_prompt(cosmos3_runner *r, int slot, const long long *ids, int n) {
    if (!r->geom_ok) { fprintf(stderr, "cosmos3: set_geometry first\n"); return -1; }
    c3_prompt *P = &r->pr[slot];
    P->n = n; P->nund = n; free(P->ids); P->ids = (long long *)malloc(8 * (size_t)n); memcpy(P->ids, ids, 8 * (size_t)n);
    int nk = n + r->ngen;
    if (c3_scratch(r, (n > r->ngen ? n : r->ngen))) return -1;
    c3_prompt_release(r, P);
    P->kv_rows = nk; P->ng_alloc = r->ngen;
    P->und_h = c3_alloc(r, 2ull * n * C3_D);
    for (int l = 0; l < C3_NL; l++) { P->kc[l] = c3_alloc(r, 2ull * nk * 1024); P->vc[l] = c3_alloc(r, 2ull * nk * 1024); if (!P->kc[l] || !P->vc[l]) return -1; }
    /* embeddings (host gather from the mmap'd table) */
    unsigned short *e = (unsigned short *)malloc(2ull * n * C3_D);
    for (int i = 0; i < n; i++) { if (ids[i] < 0 || ids[i] >= C3_VOCAB) { fprintf(stderr, "cosmos3: bad token id\n"); return -1; } memcpy(e + (size_t)i * C3_D, r->embed_host + (size_t)ids[i] * C3_D, 2 * C3_D); }
    C3_CK(cuMemcpyHtoD(P->und_h, e, 2ull * n * C3_D)); free(P->emb_host); P->emb_host = e;
    /* positions: text arange; vision: fps-modulated t + (h, w) grid; action: (frame+1)/fps*6 + off. Mirrors diffusers get_3d_mrope_ids_*. */
    int ng = r->ngen; float off = (float)(n + 15000);
    float *pu = (float *)malloc(4ull * 3 * n), *pg = (float *)malloc(4ull * 3 * ng);
    for (int a = 0; a < 3; a++) for (int i = 0; i < n; i++) pu[(size_t)a * n + i] = (float)i;
    int T = r->T, HP = r->HP, WP = r->WP; float maxp = 0.f; (void)maxp;
    float tps = r->fps / 4.0f, btps = 24.0f / 4.0f;
    for (int t = 0; t < T; t++) for (int h = 0; h < HP; h++) for (int w = 0; w < WP; w++) {
        int i = (t * HP + h) * WP + w;
        pg[i] = (T > 1) ? ((float)t / tps * btps + off) : (float)((int)off + t);
        pg[(size_t)ng + i] = (float)h; pg[(size_t)2 * ng + i] = (float)w;
    }
    float atps = r->fps / 1.0f;
    for (int f = 0; f < r->chunk; f++) { int i = r->nvis + f; pg[i] = ((float)(f + 1) / atps * btps + off); pg[(size_t)ng + i] = 0.f; pg[(size_t)2 * ng + i] = 0.f; }
    unsigned short *cu_ = (unsigned short *)malloc(2ull * n * 128), *su_ = (unsigned short *)malloc(2ull * n * 128), *cg_ = (unsigned short *)malloc(2ull * ng * 128), *sg_ = (unsigned short *)malloc(2ull * ng * 128);
    c3_build_rope(r, pu, n, cu_, su_); c3_build_rope(r, pg, ng, cg_, sg_);
    P->cos_u = c3_alloc(r, 2ull * n * 128); P->sin_u = c3_alloc(r, 2ull * n * 128); P->cos_g = c3_alloc(r, 2ull * ng * 128); P->sin_g = c3_alloc(r, 2ull * ng * 128);
    cuMemcpyHtoD(P->cos_u, cu_, 2ull * n * 128); cuMemcpyHtoD(P->sin_u, su_, 2ull * n * 128); cuMemcpyHtoD(P->cos_g, cg_, 2ull * ng * 128); cuMemcpyHtoD(P->sin_g, sg_, 2ull * ng * 128);
    if (r->dump_dir[0] && slot == 0) {
        float *f = (float *)malloc(4ull * ng * 128); long long sh[2] = { ng, 128 };
        for (int i = 0; i < ng * 128; i++) f[i] = c3_bf2f_h(cg_[i]); c3_dump_f(r, "cos_gen", f, 2, sh);
        for (int i = 0; i < ng * 128; i++) f[i] = c3_bf2f_h(sg_[i]); c3_dump_f(r, "sin_gen", f, 2, sh);
        long long sh3[2] = { 3, ng }; c3_dump_f(r, "position_ids_gen", pg, 2, sh3);
        float *g = (float *)malloc(4ull * n * 128); long long shu[2] = { n, 128 };
        for (int i = 0; i < n * 128; i++) g[i] = c3_bf2f_h(cu_[i]); c3_dump_f(r, "cos_und", g, 2, shu);
        for (int i = 0; i < n * 128; i++) g[i] = c3_bf2f_h(su_[i]); c3_dump_f(r, "sin_und", g, 2, shu);
        free(f); free(g);
    }
    { int *iu = (int *)calloc(6ull * n, 4), *ig = (int *)calloc(6ull * ng, 4);
      for (int i = 0; i < n; i++) { iu[6 * i] = 0; iu[6 * i + 1] = i + 1; }
      for (int i = 0; i < ng; i++) { ig[6 * i] = 0; ig[6 * i + 1] = nk; }
      P->iv_u = c3_alloc(r, 24ull * n); P->iv_g = c3_alloc(r, 24ull * ng);
      cuMemcpyHtoD(P->iv_u, iu, 24ull * n); cuMemcpyHtoD(P->iv_g, ig, 24ull * ng); free(iu); free(ig); }
    free(pu); free(pg); free(cu_); free(su_); free(cg_); free(sg_);
    P->ready = 0;
    return 0;
}

/* ---- GEMM / kernel launch helpers ---- */
static int c3_gemm(cosmos3_runner *r, int M, int N, int K, CUdeviceptr W, CUdeviceptr X, CUdeviceptr Y) {
    if (cublasew_gemm_ex_rm(r->cb, 0, M, N, K, W, K, 1, X, K, 1, Y, N, 1, 0.f)) { fprintf(stderr, "cosmos3: gemm %dx%dx%d failed\n", M, N, K); return -1; }
    return 0;
}
#define C3_LAUNCH(f, gx, gy, bx, shm, ...) do { void *_a[] = { __VA_ARGS__ }; C3_CK(cuLaunchKernel(f, gx, gy, 1, bx, 1, 1, shm, 0, _a, NULL)); } while (0)
#define G1(n, b) (unsigned)(((long long)(n) + (b) - 1) / (b))

static int c3_rms(cosmos3_runner *r, CUdeviceptr x, CUdeviceptr w, CUdeviceptr y, int rows) {
    int dim = C3_D; float eps = 1e-5f; long long ld = C3_D;
    C3_LAUNCH(r->f_rms, (unsigned)rows, 1, 256, 0, &x, &w, &y, &dim, &eps, &ld, &ld); return 0;
}
static int c3_rmsh(cosmos3_runner *r, CUdeviceptr x, CUdeviceptr w, int rows, int nh, long long ld) {
    float eps = 1e-5f; long long warps = (long long)rows * nh;
    C3_LAUNCH(r->f_rmsh, G1(warps * 32, 128), 1, 128, 0, &x, &w, &rows, &nh, &eps, &ld); return 0;
}
static int c3_rope_k(cosmos3_runner *r, CUdeviceptr x, CUdeviceptr cs, CUdeviceptr sn, int rows, int nh, long long ld) {
    long long tot = (long long)rows * nh * 64;
    C3_LAUNCH(r->f_rope, G1(tot, 256), 1, 256, 0, &x, &cs, &sn, &rows, &nh, &ld); return 0;
}
static int c3_addk(cosmos3_runner *r, CUdeviceptr a, CUdeviceptr b, long long n) { C3_LAUNCH(r->f_add, G1(n, 256), 1, 256, 0, &a, &b, &n); return 0; }
static int c3_relu2k(cosmos3_runner *r, CUdeviceptr x, long long n) { C3_LAUNCH(r->f_relu2, G1(n, 256), 1, 256, 0, &x, &n); return 0; }
static int c3_attn(cosmos3_runner *r, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr O, int nq, int nk, int causal, CUdeviceptr iv) {
    long long ldq = C3_D, ldk = 1024, ldo = C3_D; int groups = C3_NH / C3_NKV; float scale = 1.0f / sqrtf((float)C3_HD);
    if (!r->naive_attn) {
        /* tensor-core block-sparse flash (praxis pr_flashiv128): bf16 P, fp32 accumulate; per-query key intervals from iv */
        int H = C3_NH, KVH = C3_NKV; CUdeviceptr gate = 0, kcode = 0, ktile = 0;
        int ntile = (nk + 31) / 32; unsigned shm = (unsigned)(4 * 32 * (128 + 8) * 2 + 4 * (ntile + 8));
        if (shm > 48 * 1024) { C3_CK(cuFuncSetAttribute(r->f_fiv, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)shm)); }
        C3_LAUNCH(r->f_fiv, (unsigned)((nq + 63) / 64), C3_NH, 128, shm, &Q, &ldq, &K, &ldk, &V, &ldk, &O, &ldo, &gate, &H, &KVH, &nq, &nk, &scale, &iv, &kcode, &ktile);
        return 0;
    }
    unsigned shm = (unsigned)(nk * 4);
    if (shm > 48 * 1024) { C3_CK(cuFuncSetAttribute(r->f_attn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)shm)); }
    C3_LAUNCH(r->f_attn, (unsigned)nq, C3_NH, 128, shm, &Q, &ldq, &K, &V, &ldk, &O, &ldo, &nq, &nk, &causal, &groups, &scale); return 0;
}

/* ---- understanding tower (once per prompt, weights streamed) ---- */
static int c3_load_und_layer(cosmos3_runner *r, int l, int sl) {
    char n[160]; c3_layer *U = &r->und_slot[sl]; size_t nb;
    struct { CUdeviceptr d; const char *fmt; } t[] = { { U->ln_in, "layers.%d.input_layernorm.weight" }, { U->q, "layers.%d.self_attn.to_q.weight" }, { U->k, "layers.%d.self_attn.to_k.weight" },
        { U->v, "layers.%d.self_attn.to_v.weight" }, { U->kng, "layers.%d.self_attn.k_norm_und_for_gen.weight" }, { U->out, "layers.%d.self_attn.to_out.weight" },
        { U->ln_post, "layers.%d.post_attention_layernorm.weight" }, { U->up, "layers.%d.mlp.up_proj.weight" }, { U->down, "layers.%d.mlp.down_proj.weight" } };
    /* slot reuse: wait for the compute that read this slot two layers ago */
    if (l >= 2) C3_CK(cuStreamWaitEvent(r->copy_stream, r->ev_done[sl], 0));
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        snprintf(n, sizeof n, t[i].fmt, l);
        const void *h = c3_host(r, n, &nb); if (!h) return -1;
        C3_CK(cuMemcpyHtoDAsync(t[i].d, h, nb, r->copy_stream)); r->h2d_und += nb;
    }
    C3_CK(cuEventRecord(r->ev_loaded[sl], r->copy_stream));
    return 0;
}

int cosmos3_prepare_und(cosmos3_runner *r, int nslots) {
    double t0 = c3_now(); r->h2d_und = 0;
    if (c3_load_und_layer(r, 0, 0)) return -1;
    for (int l = 0; l < C3_NL; l++) {
        c3_layer *U = &r->und_slot[l & 1];
        if (l + 1 < C3_NL && c3_load_und_layer(r, l + 1, (l + 1) & 1)) return -1;   /* prefetch next layer while this one computes */
        C3_CK(cuStreamWaitEvent(0, r->ev_loaded[l & 1], 0));
        for (int s = 0; s < nslots; s++) {
            c3_prompt *P = &r->pr[s]; int n = P->nund; long long cnt = (long long)n * C3_D;
            char nm[64];
            if (l == 0) C3_CK(cuMemcpyHtoD(P->und_h, P->emb_host, 2ull * n * C3_D));
            if (l == 0 && r->dump_dir[0]) { snprintf(nm, sizeof nm, "s0_%s_embed_text", s == 0 ? "cond" : "uncond"); c3_dump_bf(r, nm, P->und_h, n, C3_D); }
            if (c3_rms(r, P->und_h, U->ln_in, r->bufX, n)) return -1;
            if (c3_gemm(r, n, C3_D, C3_D, U->q, r->bufX, r->bufQ)) return -1;
            if (c3_gemm(r, n, 1024, C3_D, U->k, r->bufX, r->bufK)) return -1;
            if (c3_gemm(r, n, 1024, C3_D, U->v, r->bufX, P->vc[l])) return -1;
            /* k for the generation tower: copy, k_norm_und_for_gen, rope */
            { long long ld = 1024; int rows = n, cols = 1024; long long lds = 1024;
              C3_LAUNCH(r->f_copy, G1((long long)rows * cols, 256), 1, 256, 0, &r->bufK, &lds, &P->kc[l], &ld, &rows, &cols); }
            if (c3_rmsh(r, P->kc[l], U->kng, n, C3_NKV, 1024)) return -1;
            if (c3_rope_k(r, P->kc[l], P->cos_u, P->sin_u, n, C3_NKV, 1024)) return -1;
            if (c3_rope_k(r, r->bufK, P->cos_u, P->sin_u, n, C3_NKV, 1024)) return -1;
            if (c3_rope_k(r, r->bufQ, P->cos_u, P->sin_u, n, C3_NH, C3_D)) return -1;
            if (c3_attn(r, r->bufQ, r->bufK, P->vc[l], r->bufA, n, n, 1, P->iv_u)) return -1;
            if (c3_gemm(r, n, C3_D, C3_D, U->out, r->bufA, r->bufX)) return -1;
            if (c3_addk(r, P->und_h, r->bufX, cnt)) return -1;
            if (c3_rms(r, P->und_h, U->ln_post, r->bufX, n)) return -1;
            if (c3_gemm(r, n, C3_FF, C3_D, U->up, r->bufX, r->bufM)) return -1;
            if (c3_relu2k(r, r->bufM, (long long)n * C3_FF)) return -1;
            if (c3_gemm(r, n, C3_D, C3_FF, U->down, r->bufM, r->bufX)) return -1;
            if (c3_addk(r, P->und_h, r->bufX, cnt)) return -1;
            if (r->dump_dir[0]) { snprintf(nm, sizeof nm, "s0_%s_layer%02d_und", s == 0 ? "cond" : "uncond", l); c3_dump_bf(r, nm, P->und_h, n, C3_D); }
        }
        C3_CK(cuEventRecord(r->ev_done[l & 1], 0));
    }
    C3_CK(cuCtxSynchronize());
    for (int s = 0; s < nslots; s++) r->pr[s].ready = 1;
    r->und_ms = c3_now() - t0; c3_track_free(r);
    return 0;
}

/* ---- time embedding (host, fp32 like the fp32 time_embedder) ---- */
static void c3_time_embed(const cosmos3_runner *r, float t, unsigned short *out /*[2048] bf16*/) {
    float ts = t * 0.001f; float in[256], h1[2048], h2[2048];
    for (int j = 0; j < 128; j++) { float fr = expf(-logf(10000.0f) * (float)j / 128.0f); float a = ts * fr; in[j] = cosf(a); in[128 + j] = sinf(a); }
    #pragma omp parallel for
    for (int o = 0; o < 2048; o++) { const float *w = r->te_w1 + (size_t)o * 256; float s = r->te_b1[o]; for (int i = 0; i < 256; i++) s += w[i] * in[i]; h1[o] = s / (1.0f + expf(-s)); }
    #pragma omp parallel for
    for (int o = 0; o < 2048; o++) { const float *w = r->te_w2 + (size_t)o * 2048; float s = r->te_b2[o]; for (int i = 0; i < 2048; i++) s += w[i] * h1[i]; h2[o] = s; }
    for (int o = 0; o < 2048; o++) out[o] = c3_f2bf_h(h2[o]);
}

/* ---- generation pass ---- */
static int c3_gemm_f32(cosmos3_runner *r, int M, int N, int K, CUdeviceptr W, CUdeviceptr X, CUdeviceptr Yf32) {
    if (cublasew_gemm_ex_rm(r->cb, 0, M, N, K, W, K, 1, X, K, 1, Yf32, N, 0, 0.f)) { fprintf(stderr, "cosmos3: gemm_f32 %dx%dx%d failed\n", M, N, K); return -1; }
    return 0;
}
static int c3_bias_k(cosmos3_runner *r, CUdeviceptr acc, CUdeviceptr bias, CUdeviceptr y, int rows, int cols) {
    long long ldy = cols; C3_LAUNCH(r->f_bias, G1((long long)rows * cols, 256), 1, 256, 0, &acc, &bias, &y, &rows, &cols, &ldy); return 0;
}

int cosmos3_gen_pass(cosmos3_runner *r, int slot, int step_tag, const float *lat, const float *act, int t, float *vel_v, float *vel_a) {
    c3_prompt *P = &r->pr[slot];
    if (!P->ready) { fprintf(stderr, "cosmos3: und tower not prepared for slot %d\n", slot); return -1; }
    double t0 = c3_now();
    int ng = r->ngen, nv = r->nvis, n = P->nund, nk = n + ng, T = r->T, H = r->H, W = r->W, HP = r->HP, WP = r->WP, ch = r->chunk, C = C3_LC;
    long long latn = (long long)C * T * H * W;
    int dump = (step_tag < r->dump_steps) && r->dump_dir[0]; char nm[96]; const char *pn = slot == 0 ? "cond" : "uncond";
    if (!r->gen_h) {
        r->gen_h = c3_alloc(r, 2ull * ng * C3_D); r->d_lat = c3_alloc(r, 4ull * latn); r->d_tok = c3_alloc(r, 2ull * nv * C3_PD); r->d_temb = c3_alloc(r, 2 * C3_D);
        r->d_vel = c3_alloc(r, 4ull * latn); r->d_act32 = c3_alloc(r, 4ull * ch * C3_ADIM); r->d_actx = c3_alloc(r, 2ull * ch * C3_ADIM);
        r->d_actbf = c3_alloc(r, 2ull * ch * C3_ADIM); r->d_actf = c3_alloc(r, 4ull * ch * C3_ADIM); r->d_pred = c3_alloc(r, 2ull * nv * C3_PD);
        if (!r->gen_h || !r->d_lat || !r->d_tok || !r->d_vel || !r->d_act32 || !r->d_pred || !r->d_actf) return -1;
    }
    unsigned short temb[2048]; c3_time_embed(r, (float)t, temb);
    C3_CK(cuMemcpyHtoD(r->d_temb, temb, 4096));
    C3_CK(cuMemcpyHtoD(r->d_lat, lat, 4ull * latn));
    C3_CK(cuMemcpyHtoD(r->d_act32, act, 4ull * ch * C3_ADIM));
    { long long tot = (long long)nv * C3_PD;
      C3_LAUNCH(r->f_patch, G1(tot, 256), 1, 256, 0, &r->d_lat, &r->d_tok, &C, &T, &H, &W, &HP, &WP); }
    if (c3_gemm_f32(r, nv, C3_D, C3_PD, r->proj_in_w, r->d_tok, r->bufF)) return -1;
    if (c3_bias_k(r, r->bufF, r->proj_in_b, r->gen_h, nv, C3_D)) return -1;
    if (dump) { snprintf(nm, sizeof nm, "s%d_%s_proj_in", step_tag, pn); c3_dump_bf(r, nm, r->gen_h, nv, C3_D); }
    { /* noisy frames (all but frame 0) get + time embedding */
      int fr = HP * WP, rows = nv - fr; CUdeviceptr base = r->gen_h + 2ull * fr * C3_D; int cols = C3_D;
      C3_LAUNCH(r->f_addrow, G1((long long)rows * cols, 256), 1, 256, 0, &base, &r->d_temb, &rows, &cols); }
    /* action tokens: DomainAwareLinear (bmm -> bf16, + bias), + modality embed, + time embedding */
    { long long tot = (long long)ch * C3_ADIM; C3_LAUNCH(r->f_f2bf, G1(tot, 256), 1, 256, 0, &r->d_act32, &r->d_actx, &tot);
      CUdeviceptr Wd = r->act_in_fc + 2ull * r->domain * C3_ADIM * C3_D, Bd = r->act_in_b + 2ull * r->domain * C3_D, Y = r->gen_h + 2ull * nv * C3_D;
      if (cublasew_gemm_ex_rm(r->cb, 1, ch, C3_D, C3_ADIM, Wd, C3_D, 1, r->d_actx, C3_ADIM, 1, Y, C3_D, 1, 0.f)) { fprintf(stderr, "cosmos3: action_proj_in gemm failed\n"); return -1; }
      int cols = C3_D;
      C3_LAUNCH(r->f_addbias, G1((long long)ch * cols, 256), 1, 256, 0, &Y, &Bd, &ch, &cols);
      C3_LAUNCH(r->f_addrow, G1((long long)ch * cols, 256), 1, 256, 0, &Y, &r->act_embed, &ch, &cols);
      C3_LAUNCH(r->f_addrow, G1((long long)ch * cols, 256), 1, 256, 0, &Y, &r->d_temb, &ch, &cols); }
    if (dump) { snprintf(nm, sizeof nm, "s%d_%s_layer_in_gen", step_tag, pn); c3_dump_bf(r, nm, r->gen_h, ng, C3_D);
                snprintf(nm, sizeof nm, "s%d_%s_in_vision_tokens", step_tag, pn); /* raw input latents are exported by the driver */ }
    for (int l = 0; l < C3_NL; l++) {
        c3_layer *L = &r->gen[l];
        CUdeviceptr kdst = P->kc[l] + 2ull * n * 1024, vdst = P->vc[l] + 2ull * n * 1024;
        if (c3_rms(r, r->gen_h, L->ln_in, r->bufX, ng)) return -1;
        if (c3_gemm(r, ng, C3_D, C3_D, L->q, r->bufX, r->bufQ)) return -1;
        if (c3_gemm(r, ng, 1024, C3_D, L->k, r->bufX, kdst)) return -1;
        if (c3_gemm(r, ng, 1024, C3_D, L->v, r->bufX, vdst)) return -1;
        if (c3_rmsh(r, r->bufQ, L->qn, ng, C3_NH, C3_D)) return -1;
        if (c3_rmsh(r, kdst, L->kn, ng, C3_NKV, 1024)) return -1;
        if (c3_rope_k(r, r->bufQ, P->cos_g, P->sin_g, ng, C3_NH, C3_D)) return -1;
        if (c3_rope_k(r, kdst, P->cos_g, P->sin_g, ng, C3_NKV, 1024)) return -1;
        if (c3_attn(r, r->bufQ, P->kc[l], P->vc[l], r->bufA, ng, nk, 0, P->iv_g)) return -1;
        if (c3_gemm(r, ng, C3_D, C3_D, L->out, r->bufA, r->bufX)) return -1;
        if (c3_addk(r, r->gen_h, r->bufX, (long long)ng * C3_D)) return -1;
        if (c3_rms(r, r->gen_h, L->ln_post, r->bufX, ng)) return -1;
        if (c3_gemm(r, ng, C3_FF, C3_D, L->up, r->bufX, r->bufM)) return -1;
        if (c3_relu2k(r, r->bufM, (long long)ng * C3_FF)) return -1;
        if (c3_gemm(r, ng, C3_D, C3_FF, L->down, r->bufM, r->bufX)) return -1;
        if (c3_addk(r, r->gen_h, r->bufX, (long long)ng * C3_D)) return -1;
        if (dump) { snprintf(nm, sizeof nm, "s%d_%s_layer%02d_gen", step_tag, pn, l); c3_dump_bf(r, nm, r->gen_h, ng, C3_D); }
    }
    if (c3_rms(r, r->gen_h, r->norm_gen, r->bufX, ng)) return -1;
    if (dump) { snprintf(nm, sizeof nm, "s%d_%s_norm_gen", step_tag, pn); c3_dump_bf(r, nm, r->bufX, ng, C3_D); }
    /* vision head: noisy frames only */
    { int fr = HP * WP, rows = nv - fr;
      if (c3_gemm_f32(r, rows, C3_PD, C3_D, r->proj_out_w, r->bufX + 2ull * fr * C3_D, r->bufF)) return -1;
      if (c3_bias_k(r, r->bufF, r->proj_out_b, r->d_pred, rows, C3_PD)) return -1;
      long long tot = latn;
      C3_LAUNCH(r->f_unpatch, G1(tot, 256), 1, 256, 0, &r->d_pred, &r->d_vel, &C, &T, &H, &W, &HP, &WP); }
    /* action head */
    { CUdeviceptr Wd = r->act_out_fc + 2ull * r->domain * C3_D * C3_ADIM, Bd = r->act_out_b + 2ull * r->domain * C3_ADIM, X = r->bufX + 2ull * nv * C3_D;
      if (cublasew_gemm_ex_rm(r->cb, 1, ch, C3_ADIM, C3_D, Wd, C3_ADIM, 1, X, C3_D, 1, r->d_actbf, C3_ADIM, 1, 0.f)) { fprintf(stderr, "cosmos3: action_proj_out gemm failed\n"); return -1; }
      int cols = C3_ADIM; long long tot = (long long)ch * C3_ADIM;
      C3_LAUNCH(r->f_addbias, G1(tot, 256), 1, 256, 0, &r->d_actbf, &Bd, &ch, &cols);
      C3_LAUNCH(r->f_bf2f, G1(tot, 256), 1, 256, 0, &r->d_actbf, &r->d_actf, &tot); }
    C3_CK(cuCtxSynchronize());
    if (vel_v) C3_CK(cuMemcpyDtoH(vel_v, r->d_vel, 4ull * latn));
    if (vel_a) C3_CK(cuMemcpyDtoH(vel_a, r->d_actf, 4ull * ch * C3_ADIM));
    if (dump && vel_v) { long long sh[5] = { 1, C, T, H, W }; snprintf(nm, sizeof nm, "s%d_%s_pred_vision", step_tag, pn); c3_dump_f(r, nm, vel_v, 5, sh); }
    if (dump && vel_a) { long long sh[2] = { ch, C3_ADIM }; snprintf(nm, sizeof nm, "s%d_%s_pred_action", step_tag, pn); c3_dump_f(r, nm, vel_a, 2, sh); }
    r->gen_ms = c3_now() - t0; c3_track_free(r);
    return 0;
}

/* ---- UniPC (diffusers UniPCMultistepScheduler: order 2, bh2, flow prediction, predict_x0, lower_order_final) ---- */
static void c3_sigmas(int steps, double shift, double *sig /*[steps+1]*/, int *ts) {
    for (int i = 0; i < steps; i++) {
        double s = (steps > 0) ? (1.0 - 1.0 / 1000.0) + (0.0 - (1.0 - 1.0 / 1000.0)) * (double)i / (double)steps : 0;
        s = shift * s / (1.0 + (shift - 1.0) * s);
        if (fabs(s - 1.0) < 1e-6) s -= 1e-6;
        ts[i] = (int)(s * 1000.0); sig[i] = (double)(float)s;
    }
    sig[steps] = 0.0;
}
int cosmos3_timestep_for(const cosmos3_runner *r, int steps, int i, double shift) {
    (void)r; double sig[256]; int ts[256]; c3_sigmas(steps, shift, sig, ts); return ts[i];
}
static double c3_lam(double sigma) { return log(1.0 - sigma) - log(sigma); }

/* one scheduler step for one modality: vel -> converted x0, optional corrector, predictor; sample updated in place */
static void c3_uni_step(const double *sig, int nsteps, c3_uni *u, int n, const float *vel, float *sample) {
    int si = u->idx;
    float *x0 = (float *)malloc(4 * (size_t)n);
    float sg = (float)sig[si];
    for (int i = 0; i < n; i++) x0[i] = sample[i] - sg * vel[i];
    int use_corr = si > 0 && u->last;
    if (use_corr) {
        int order = u->this_order;
        double sigma_t = sig[si], sigma_s0 = sig[si - 1], alpha_t = 1.0 - sigma_t, alpha_s0 = 1.0 - sigma_s0;
        double lam_t = log(alpha_t) - log(sigma_t), lam_s0 = log(alpha_s0) - log(sigma_s0), h = lam_t - lam_s0;
        double rk = 1.0; if (order == 2) { double sj = sig[si - 2]; rk = (c3_lam(sj) - lam_s0) / h; }
        double hh = -h, hphi1 = expm1(hh), hphik = hphi1 / hh - 1.0, Bh = expm1(hh);
        double b0 = hphik * 1.0 / Bh; hphik = hphik / hh - 1.0 / 2.0; double b1 = hphik * 2.0 / Bh;
        double r0 = 0, r1 = 0.5;
        if (order == 2) { r0 = (b0 - b1) / (1.0 - rk); r1 = b0 - r0; }
        float c_x = (float)(sigma_t / sigma_s0), c_m = (float)(alpha_t * hphi1), c_b = (float)(alpha_t * Bh);
        const float *m0 = u->mo[1], *mm = u->mo[0], *x = u->last;
        for (int i = 0; i < n; i++) {
            float d1t = x0[i] - m0[i];
            float corr = 0.f;
            if (order == 2) { float d1 = (mm[i] - m0[i]) / (float)rk; corr = (float)r0 * d1; }
            float xt_ = c_x * x[i] - c_m * m0[i];
            sample[i] = xt_ - c_b * (corr + (float)r1 * d1t);
        }
    }
    /* shift history */
    free(u->mo[0]); u->mo[0] = u->mo[1]; u->mo[1] = x0;
    int this_order = nsteps - si < 2 ? nsteps - si : 2;
    if (this_order > u->lower + 1) this_order = u->lower + 1;
    u->this_order = this_order;
    free(u->last); u->last = (float *)malloc(4 * (size_t)n); memcpy(u->last, sample, 4 * (size_t)n);
    /* predictor */
    double sigma_t = sig[si + 1], sigma_s0 = sig[si], alpha_t = 1.0 - sigma_t, alpha_s0 = 1.0 - sigma_s0;
    double lam_t = log(alpha_t) - log(sigma_t), lam_s0 = log(alpha_s0) - log(sigma_s0), h = lam_t - lam_s0;
    double hh = -h, hphi1 = expm1(hh), Bh = expm1(hh);
    double rk = 1.0; if (this_order == 2) { double sj = sig[si - 1]; rk = (c3_lam(sj) - lam_s0) / h; }
    float c_x = (float)(sigma_t / sigma_s0), c_m = (float)(alpha_t * hphi1), c_b = (float)(alpha_t * Bh);
    const float *m0 = u->mo[1], *mm = u->mo[0];
    for (int i = 0; i < n; i++) {
        float res = 0.f;
        if (this_order == 2) { float d1 = (mm[i] - m0[i]) / (float)rk; res = 0.5f * d1; }
        float xt_ = c_x * sample[i] - c_m * m0[i];
        sample[i] = xt_ - c_b * res;
    }
    if (u->lower < 2) u->lower++;
    u->idx++;
}
static void c3_uni_free(c3_uni *u) { free(u->mo[0]); free(u->mo[1]); free(u->last); memset(u, 0, sizeof *u); }

int cosmos3_sample(cosmos3_runner *r, const cosmos3_sample_cfg *c, const float *lat_init, const float *act_init, float *act_out, float *lat_out) {
    int T = r->T, H = r->H, W = r->W, ch = r->chunk, C = C3_LC;
    int nv = C * T * H * W, na = ch * C3_ADIM;
    double sig[256]; int ts[256]; if (c->steps > 250) return -1;
    c3_sigmas(c->steps, c->flow_shift, sig, ts);
    float *lat = (float *)malloc(4ull * nv), *act = (float *)malloc(4ull * na);
    float *cv = (float *)malloc(4ull * nv), *uv = (float *)malloc(4ull * nv), *ca = (float *)malloc(4ull * na), *ua = (float *)malloc(4ull * na);
    float *vv = (float *)malloc(4ull * nv), *va = (float *)malloc(4ull * na);
    memcpy(lat, lat_init, 4ull * nv); memcpy(act, act_init, 4ull * na);
    c3_uni uv_s, ua_s; memset(&uv_s, 0, sizeof uv_s); memset(&ua_s, 0, sizeof ua_s);
    for (int i = 0; i < c->steps; i++) {
        if (cosmos3_gen_pass(r, 0, i, lat, act, ts[i], cv, ca)) return -1;
        if (c->have_uncond) if (cosmos3_gen_pass(r, 1, i, lat, act, ts[i], uv, ua)) return -1;
        /* mask: action columns >= raw dim zero */
        for (int a = 0; a < ch; a++) for (int d = r->raw_adim; d < C3_ADIM; d++) { ca[a * C3_ADIM + d] = 0; if (c->have_uncond) ua[a * C3_ADIM + d] = 0; }
        for (int k = 0; k < nv; k++) vv[k] = c->have_uncond ? uv[k] + c->guidance * (cv[k] - uv[k]) : cv[k];
        for (int k = 0; k < na; k++) va[k] = c->have_uncond ? ua[k] + c->guidance * (ca[k] - ua[k]) : ca[k];
        if (r->dump_dir[0]) { char nm[96]; long long sh[5] = { 1, C, T, H, W }, sa[2] = { ch, C3_ADIM };
            snprintf(nm, sizeof nm, "step%d_vision_velocity_cfg", i); c3_dump_f(r, nm, vv, 5, sh);
            snprintf(nm, sizeof nm, "step%d_action_velocity_cfg", i); c3_dump_f(r, nm, va, 2, sa);
            snprintf(nm, sizeof nm, "step%d_vision_sample_in", i); c3_dump_f(r, nm, lat, 5, sh);
            snprintf(nm, sizeof nm, "step%d_action_sample_in", i); c3_dump_f(r, nm, act, 2, sa); }
        c3_uni_step(sig, c->steps, &uv_s, nv, vv, lat);
        c3_uni_step(sig, c->steps, &ua_s, na, va, act);
        for (int a = 0; a < ch; a++) for (int d = r->raw_adim; d < C3_ADIM; d++) act[a * C3_ADIM + d] = 0;
        if (r->dump_dir[0]) { char nm[96]; long long sh[5] = { 1, C, T, H, W }, sa[2] = { ch, C3_ADIM };
            snprintf(nm, sizeof nm, "step%d_vision_sample_out", i); c3_dump_f(r, nm, lat, 5, sh);
            snprintf(nm, sizeof nm, "step%d_action_sample_out", i); c3_dump_f(r, nm, act, 2, sa); }
    }
    memcpy(act_out, act, 4ull * na); if (lat_out) memcpy(lat_out, lat, 4ull * nv);
    c3_uni_free(&uv_s); c3_uni_free(&ua_s); free(lat); free(act); free(cv); free(uv); free(ca); free(ua); free(vv); free(va);
    return 0;
}

void cosmos3_get_stats(const cosmos3_runner *r, cosmos3_stats *s) {
    s->peak_alloc_mb = r->alloc_peak / 1048576.0; s->weights_mb = r->weights_bytes / 1048576.0; s->min_free_mb = r->min_free / 1048576.0;
    s->base_free_mb = r->base_free / 1048576.0; s->und_ms = r->und_ms; s->gen_ms_last = r->gen_ms; s->h2d_und_mb = r->h2d_und / 1048576.0;
}
#endif /* CUDA_COSMOS3_IMPLEMENTATION */
