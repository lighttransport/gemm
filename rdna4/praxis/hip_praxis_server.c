/*
 * hip_praxis_server.c - persistent praxis-1 policy server for RDNA4 / ROCm, same binary protocol as cuda/praxis/praxis_server.c (so
 * cuda/praxis/praxis_client.py works against it: PraxisClient(..., binary="rdna4/praxis/hip_praxis_server")).
 *
 *   rdna4/praxis/hip_praxis_server --policy /mnt/nvme01/models/praxis/praxis-1-droid [--stream 2] [--text-cache DIR]
 *        [--text FILE.npy --text-u FILE.npy] [--trim 0|1|2]
 *   env: PRAXIS_EXPERT_FMT=mixed|nvfp4a16|nvfp4 (W4A16 on RDNA4), PRAXIS_FP8_BANKS=..., PRAXIS_PACK_CACHE=<file>, PRAXIS_SLOT_MB=...
 *
 * Request  = u32 magic 'PRXQ' | u32 op | u64 len | payload.   Response = u32 'PRXR' | i32 status | u64 len | payload.
 * Every OK response payload starts with 4 f64: ms of the op, ms of load/aux work inside it, peak device GB over the idle context during the op, device GB
 * resident after the op (hipMemGetInfo, polled every 4 ms), then the op body.
 *  op 2 SAMPLE  : i32 K,h,w,steps,skip_last_video | f64 t_shift,cfg_video,cfg_action | i32 plen | prompt | i32 ulen | uncond prompt |
 *                 f32 anchor[C][1][h][w] | f32 clean[C][(K-1)*F][h][w] (K>1) | f32 proprio[K][10] | f32 noise_v[C][K*F][h][w] | f32 noise_a[K*F][A]
 *                 -> i32 C,F,h,w,A | f32 video[C][F][h][w] | f32 action[F][A]
 *  op 3 TEXT    : i32 n | n x (i32 len | bytes) -> empty (checks / warms the text cache)
 *  op 5 INFO    : -> text
 *  op 6 QUIT
 *  op 1 ENCODE  : i32 T,H,W | u8 [3][T][H][W]   (tile clip, 0..255) -> i32 C,Tl,h,w | f32 [C][Tl][h][w] policy latent        (hip_wan22_vae)
 *  op 4 DECODE  : i32 Tl,h,w,keep | f32 latent[C][Tl][h][w] -> i32 n,H,W | u8 [n][H][W][3]   (last `keep` frames)
 *  options: --vae DIR (wan2.2-vae) --qwen DIR (qwen3.5-2b) --vae-mode keep|reload (reload: VAE weights loaded per ENCODE/DECODE and freed after, which keeps
 *      a resident policy + VAE under 8 GB together with --trim 1/2) --text-cache DIR (disk cache of Qwen conditioning, key = fnv64 of the token ids, same
 *      files as the CUDA server's text_<hash>.npy)
 *  Qwen3.5-2B runs IN PROCESS (hip_qwen35_st): all cache misses of a request are computed in one load -> prefill(s) -> free cycle (the policy ring/workspace
 *  is released first with --trim), so Qwen's device memory is gone before the policy/VAE load.
 *
 * Text conditioning (Qwen3.5-2B hidden states [S][2048], praxis uses hidden_states[-3]) comes from the disk cache: <text-cache>/prompt_<fnv64(prompt)>.npy
 * (float32 [S][2048], any producer: the CUDA text stage, the RDNA4 Qwen port, HF), or from --text / --text-u (one fixed conditioning for every prompt / for the
 * blank caption), looked up in that order.  Cache misses without a fixed file fail the SAMPLE with status -3.
 */
#define SAFETENSORS_IMPLEMENTATION
#define HIP_PRAXIS_IMPLEMENTATION
#include "hip_praxis_runner.h"
#define HIP_WAN22_VAE_IMPLEMENTATION
#include "hip_wan22_vae.h"
#include "../llm/hip_qwen35_st.h"
#include "../../cuda/praxis/praxis_qwen.h"   /* tokenizer (praxis_tok.c) */
#include <pthread.h>
#include <unistd.h>

static int g_out_fd = 1;
static int rd_full(void *b, size_t n) { size_t o = 0; while (o < n) { ssize_t k = read(0, (char *)b + o, n - o); if (k <= 0) return -1; o += (size_t)k; } return 0; }
static int wr_full(const void *b, size_t n) { size_t o = 0; while (o < n) { ssize_t k = write(g_out_fd, (const char *)b + o, n - o); if (k <= 0) return -1; o += (size_t)k; } return 0; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double gb(size_t b) { return b / 1073741824.0; }
static size_t dev_used(void) { size_t fr = 0, tot = 0; hipMemGetInfo(&fr, &tot); return tot - fr; }

static volatile size_t g_peak_used = 0; static volatile int g_poll_run = 1; static hipCtx_t g_ctx;
static void *poller(void *arg) { (void)arg; hipCtxSetCurrent(g_ctx); while (g_poll_run) { size_t u = dev_used(); if (u > g_peak_used) g_peak_used = u; usleep(4000); } return NULL; }

typedef struct { unsigned char *p; size_t n, cap; } buf_t;
static void *bufw(buf_t *b, const void *d, size_t n) { if (b->n + n > b->cap) { b->cap = (b->n + n) * 2; b->p = (unsigned char *)realloc(b->p, b->cap); } memcpy(b->p + b->n, d, n); b->n += n; return b->p + b->n - n; }

/* float32 .npy reader: returns malloc'd data, rows = shape[0] */
static float *npy_f32(const char *path, long long *rows, long long *cols) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char m[10];
    if (fread(m, 1, 10, f) != 10 || memcmp(m, "\x93NUMPY", 6) || m[6] == 2) { fclose(f); return NULL; }
    int hl = m[8] | (m[9] << 8);
    char *h = (char *)calloc(1, hl + 1);
    if (fread(h, 1, hl, f) != (size_t)hl || !strstr(h, "<f4")) { fclose(f); free(h); return NULL; }
    char *p = strchr(strstr(h, "shape"), '(') + 1;
    long long sh[4] = { 1, 1, 1, 1 }; int nd = 0; size_t n = 1;
    while (*p && *p != ')' && nd < 4) { while (*p == ' ' || *p == ',') p++; if (*p == ')') break; sh[nd] = strtoll(p, &p, 10); n *= (size_t)sh[nd]; nd++; }
    free(h);
    float *d = (float *)malloc(n * 4);
    if (fread(d, 4, n, f) != n) { free(d); d = NULL; }
    fclose(f);
    if (nd == 3 && sh[0] == 1) { sh[0] = sh[1]; sh[1] = sh[2]; nd = 2; }   /* [1][S][C] -> [S][C] */
    if (rows) *rows = sh[0];
    if (cols) *cols = nd >= 2 ? sh[1] : 1;
    return d;
}
static unsigned long long fnv64(const char *s) { unsigned long long h = 1469598103934665603ULL; for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ULL; } return h; }

typedef struct { const char *qwen, *cache, *fixed, *fixed_u, *tokjson; int trim_for_qwen; } txt_cfg;
typedef struct { char *key; float *d; int S; } tc_ent;
static tc_ent g_tc[512]; static int g_ntc;
static int g_qwen_runs = 0; static double g_t_tok = 0, g_t_qload = 0, g_t_qrun = 0, g_t_qfree = 0, g_text_peak_gb = 0;
static praxis_tok *g_tok = NULL;
static void npy_save2d(const char *path, const float *d, long long rows, long long cols) {
    FILE *f = fopen(path, "wb"); if (!f) return;
    char hdr[128]; int n = snprintf(hdr, sizeof hdr, "{'descr': '<f4', 'fortran_order': False, 'shape': (%lld, %lld), }", rows, cols);
    int total = 10 + n + 1, pad = (64 - total % 64) % 64, hl = n + pad + 1;
    fwrite("\x93NUMPY\x01\x00", 1, 8, f); fputc(hl & 0xff, f); fputc(hl >> 8, f); fwrite(hdr, 1, n, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f); fwrite(d, 4, (size_t)rows * cols, f); fclose(f);
}
static tc_ent *tc_find(const char *k) { for (int i = 0; i < g_ntc; i++) if (!strcmp(g_tc[i].key, k)) return &g_tc[i]; return NULL; }
static void tc_add(const char *k, float *d, int S) { if (g_ntc < 512) { g_tc[g_ntc].key = strdup(k); g_tc[g_ntc].d = d; g_tc[g_ntc].S = S; g_ntc++; } }
static unsigned long long fnv_bytes(const void *p, size_t n) { unsigned long long h = 1469598103934665603ULL; const unsigned char *b = (const unsigned char *)p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; } return h; }
/* hook so SAMPLE/TEXT can release the policy workspace before Qwen takes the device */
typedef void (*pre_qwen_fn)(void *);
/* conditioning of nq prompts, [S][2048] each (cached, do not free); all misses are computed in ONE Qwen load/prefill/free cycle. is_u[i]: blank caption (use fixed_u).
 * Lookup order: memory (prompt string) -> disk (<cache>/text_<fnv(int32 ids)>.npy, CUDA-compatible; legacy prompt_<fnv(prompt)>.npy) -> --text/--text-u fixed file -> Qwen. */
static float **text_cond_multi(const txt_cfg *c, const char **prompts, const int *is_u, int nq, int *S, pre_qwen_fn pre, void *pre_arg) {
    static float *res[8]; int32_t *ids[8] = {0}; int nid[8] = {0}; unsigned long long h[8] = {0}; int pend[8] = {0}, npend = 0;
    if (nq > 8) return NULL;
    for (int i = 0; i < nq; i++) {
        char kb[4200]; snprintf(kb, sizeof kb, "p:%s", prompts[i]);
        tc_ent *e = tc_find(kb);
        if (e) { fprintf(stderr, "[text] memory cache hit (%.40s)\n", prompts[i]); res[i] = e->d; S[i] = e->S; continue; }
        char path[1400]; long long rows = 0, cols = 0; float *d = NULL;
        if (c->cache) { snprintf(path, sizeof path, "%s/prompt_%016llx.npy", c->cache, fnv64(prompts[i])); d = npy_f32(path, &rows, &cols); }
        if (!d && c->qwen) {
            double t0 = now();
            if (!g_tok) { g_tok = praxis_tok_load(c->tokjson); if (!g_tok) return NULL; }
            int cap = (int)strlen(prompts[i]) + 16; ids[i] = (int32_t *)malloc((size_t)cap * 4);
            nid[i] = praxis_tok_encode(g_tok, prompts[i], ids[i], cap); g_t_tok += now() - t0;
            if (nid[i] <= 0) { fprintf(stderr, "[text] tokenize failed for \"%.40s\"\n", prompts[i]); return NULL; }
            h[i] = fnv_bytes(ids[i], (size_t)nid[i] * 4);
            if (c->cache) { snprintf(path, sizeof path, "%s/text_%016llx.npy", c->cache, h[i]); d = npy_f32(path, &rows, &cols); }
        }
        if (!d) { const char *fx = is_u[i] && c->fixed_u ? c->fixed_u : c->fixed; if (fx) d = npy_f32(fx, &rows, &cols); }
        if (d) { res[i] = d; S[i] = (int)rows; tc_add(kb, d, S[i]); continue; }
        if (!c->qwen) { fprintf(stderr, "[server] no text conditioning for prompt \"%.60s\" (no cache hit, no --qwen)\n", prompts[i]); return NULL; }
        pend[i] = 1; npend++;
    }
    if (npend) {
        if (pre) pre(pre_arg);
        size_t base = dev_used(), peak = base; int mx = 2048;
        for (int i = 0; i < nq; i++) if (pend[i] && (nid[i] + 31) / 32 * 32 > mx) mx = (nid[i] + 31) / 32 * 32;
        double tq = now(); hip_qwen35 *q = hip_qwen35_init(0, 0);
        if (!q || hip_qwen35_load_safetensors(q, c->qwen, mx)) { fprintf(stderr, "[text] Qwen load failed\n"); return NULL; }
        g_t_qload += now() - tq; { size_t u = dev_used(); if (u > peak) peak = u; }
        for (int i = 0; i < nq; i++) if (pend[i]) {
            double tr = now(); int np = (nid[i] + 31) / 32 * 32;
            int32_t *p = (int32_t *)malloc((size_t)np * 4); memcpy(p, ids[i], (size_t)nid[i] * 4); for (int k = nid[i]; k < np; k++) p[k] = 151643;
            float *o = (float *)malloc((size_t)np * 2048 * 4);
            int rc = hip_qwen35_cond(q, p, np, o); free(p);
            if (rc) { fprintf(stderr, "[text] Qwen prefill failed\n"); free(o); return NULL; }
            hipDeviceSynchronize(); g_t_qrun += now() - tr; g_qwen_runs++;
            { size_t u = dev_used(); if (u > peak) peak = u; }
            res[i] = (float *)malloc((size_t)nid[i] * 2048 * 4); memcpy(res[i], o, (size_t)nid[i] * 2048 * 4); free(o); S[i] = nid[i];
        }
        double tf = now(); hip_qwen35_free(q); hipDeviceSynchronize(); g_t_qfree += now() - tf;
        g_text_peak_gb = gb(peak > base ? peak - base : 0);
        fprintf(stderr, "[text] Qwen3.5-2B in-process: load %.2f s, %d prefill(s) %.0f ms, free %.0f ms; device peak %.2f GB over the pre-load level; after free %+.3f GB vs before load\n",
                g_t_qload, npend, g_t_qrun * 1e3, g_t_qfree * 1e3, g_text_peak_gb, (double)((long long)dev_used() - (long long)base) / 1073741824.0);
        for (int i = 0; i < nq; i++) if (pend[i]) {
            char kb[4200]; snprintf(kb, sizeof kb, "p:%s", prompts[i]); tc_add(kb, res[i], S[i]);
            if (c->cache) { char cp[1400]; snprintf(cp, sizeof cp, "mkdir -p '%s'", c->cache); if (!strchr(c->cache, '\'')) { if (system(cp)) {} }
                snprintf(cp, sizeof cp, "%s/text_%016llx.npy", c->cache, h[i]); npy_save2d(cp, res[i], S[i], 2048); }
        }
    }
    for (int i = 0; i < nq; i++) free(ids[i]);
    return res;
}

static void pre_qwen(void *a) {   /* a resident policy gives its workspace back while Qwen is on the device */
    struct { praxis_runner **r; int *rel; int trim; } *pa = a;
    if (pa->trim && *pa->r) { praxis_trim(*pa->r); if (pa->trim >= 2 && !*pa->rel) { praxis_release_stream(*pa->r); *pa->rel = 1; } }
}

int main(int argc, char **argv) {
    const char *policy = NULL, *cache = NULL, *fixed = NULL, *fixed_u = NULL, *vae_dir = "/mnt/nvme01/models/praxis/wan2.2-vae", *qwen = "/mnt/nvme01/models/praxis/qwen3.5-2b", *vae_mode = "keep";
    int ring = 2, trim = 2, ring_released = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--policy") && i + 1 < argc) policy = argv[++i];
        else if (!strcmp(argv[i], "--text-cache") && i + 1 < argc) cache = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) fixed = argv[++i];
        else if (!strcmp(argv[i], "--text-u") && i + 1 < argc) fixed_u = argv[++i];
        else if (!strcmp(argv[i], "--stream") && i + 1 < argc) ring = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trim") && i + 1 < argc) trim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vae") && i + 1 < argc) vae_dir = argv[++i];
        else if (!strcmp(argv[i], "--qwen") && i + 1 < argc) qwen = argv[++i];
        else if (!strcmp(argv[i], "--vae-mode") && i + 1 < argc) vae_mode = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!policy) { fprintf(stderr, "usage: hip_praxis_server --policy DIR [--stream R] [--text-cache DIR] [--text F.npy --text-u F.npy] [--trim 0|1|2]\n"); return 2; }
    g_out_fd = dup(1); dup2(2, 1); setvbuf(stdout, NULL, _IOLBF, 0);
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS || hipInit(0) != 0) return 1;
    int dev = 0; hipCtx_t cctx; hipCtxCreate(&cctx, 0, dev); hipCtxSetCurrent(cctx); g_ctx = cctx;
    size_t idle = dev_used();
    pthread_t th; pthread_create(&th, NULL, poller, NULL);
    int vae_keep = strcmp(vae_mode, "reload") != 0;
    char tok_path[1300]; snprintf(tok_path, sizeof tok_path, "%s/tokenizer.json", qwen);
    txt_cfg tcfg = { (qwen && qwen[0] && strcmp(qwen, "none")) ? qwen : NULL, cache, fixed, fixed_u, tok_path, 0 };
    hip_wan22_vae *vae = NULL;
    praxis_runner *r = NULL; int C = 192, F = 6, A = 0;
    fprintf(stderr, "[server] ready: policy %s, vae-mode %s, trim %d, ring %d, idle ctx %.2f GB\n", policy, vae_mode, trim, ring, gb(idle));
    for (;;) {
        unsigned char h[16];
        if (rd_full(h, 16)) break;
        uint32_t magic, op; uint64_t len; memcpy(&magic, h, 4); memcpy(&op, h + 4, 4); memcpy(&len, h + 8, 8);
        if (magic != 0x51585250u) { fprintf(stderr, "[server] bad magic\n"); return 3; }
        unsigned char *pl = (unsigned char *)malloc(len + 1); if (len && rd_full(pl, len)) break;
        size_t pos = 0;
        #define RI(v) do { int32_t t_; memcpy(&t_, pl + pos, 4); pos += 4; v = t_; } while (0)
        #define RD(v) do { double t_; memcpy(&t_, pl + pos, 8); pos += 8; v = t_; } while (0)
        #define RF(n) (float *)(pl + (pos += (size_t)(n) * 4) - (size_t)(n) * 4)
        buf_t body = { 0 }; int status = 0; double aux_ms = 0; double t0 = now(); g_peak_used = dev_used();
        if (op == 6) { free(pl); break; }
        if (op == 1) {   /* ENCODE */
            int T, H, W; RI(T); RI(H); RI(W);
            const unsigned char *u8 = pl + pos;
            double ta = now();
            if (trim && r) { praxis_trim(r); if (trim >= 2 && !ring_released) { praxis_release_stream(r); ring_released = 1; } }
            if (!vae) { vae = hip_wan22_vae_load(vae_dir, 0, 0); if (!vae) status = -1; }
            aux_ms = (now() - ta) * 1e3;
            float *lat = NULL; int Tl = 0;
            if (!status && hip_wan22_vae_encode_u8(vae, u8, T, H, W, &lat, &Tl)) { fprintf(stderr, "[server] encode: %s\n", hip_wan22_vae_last_error(vae)); status = -2; }
            if (!status) { int hh = (H + 31) / 32, ww = (W + 31) / 32; int32_t hdr[4] = { C, Tl, hh, ww }; bufw(&body, hdr, 16); bufw(&body, lat, (size_t)C * Tl * hh * ww * 4); }
            free(lat);
            if (!vae_keep && vae) { hip_wan22_vae_free(vae); vae = NULL; }
        } else if (op == 4) {   /* DECODE */
            int Tl, h_, w_, keep; RI(Tl); RI(h_); RI(w_); RI(keep);
            const float *lat = (const float *)(pl + pos);
            double ta = now(); if (trim && r) { praxis_trim(r); if (trim >= 2 && !ring_released) { praxis_release_stream(r); ring_released = 1; } }
            if (!vae) { vae = hip_wan22_vae_load(vae_dir, 0, 0); if (!vae) status = -1; }
            aux_ms = (now() - ta) * 1e3;
            float *fr = NULL; int To = 0;
            if (!status && hip_wan22_vae_decode(vae, lat, Tl, h_, w_, &fr, &To)) { fprintf(stderr, "[server] decode: %s\n", hip_wan22_vae_last_error(vae)); status = -2; }
            if (!status) {
                int H = 32 * h_, W = 32 * w_, n = keep < To ? keep : To; int32_t hdr[3] = { n, H, W }; bufw(&body, hdr, 12);
                unsigned char *u = (unsigned char *)malloc((size_t)n * H * W * 3);
                for (int t = 0; t < n; t++) for (int c = 0; c < 3; c++) { const float *sp = fr + ((size_t)c * To + (To - n + t)) * H * W;
                    for (size_t q = 0; q < (size_t)H * W; q++) { float v = rintf((sp[q] + 1.0f) * 127.5f); u[((size_t)t * H * W + q) * 3 + c] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v); } }
                bufw(&body, u, (size_t)n * H * W * 3); free(u);
            }
            free(fr);
            if (!vae_keep && vae) { hip_wan22_vae_free(vae); vae = NULL; }
        } else if (op == 3) {   /* TEXT: warm the cache (Qwen load/free happens here) */
            int n; RI(n); if (n > 8) n = 8; static char pb[8][4096]; const char *pp[8]; int iu[8] = {0}, Ss[8];
            for (int i = 0; i < n; i++) { int l; RI(l); if (l > 4095) l = 4095; memcpy(pb[i], pl + pos, l); pb[i][l] = 0; pos += (l + 3) & ~3; pp[i] = pb[i]; }
            struct { praxis_runner **r; int *rel; int trim; } pa = { &r, &ring_released, trim };
            double ta = now();
            if (!text_cond_multi(&tcfg, pp, iu, n, Ss, pre_qwen, &pa)) status = -1;
            aux_ms = (now() - ta) * 1e3;
        } else if (op == 2) {   /* SAMPLE */
            int K, h_, w_, steps, skipv, pl_, ul_; double tsh, cv, ca;
            RI(K); RI(h_); RI(w_); RI(steps); RI(skipv); RD(tsh); RD(cv); RD(ca); RI(pl_);
            char *prompt = (char *)malloc(pl_ + 1); memcpy(prompt, pl + pos, pl_); prompt[pl_] = 0; pos += (pl_ + 3) & ~3;
            RI(ul_); char *uncond = (char *)malloc(ul_ + 1); memcpy(uncond, pl + pos, ul_); uncond[ul_] = 0; pos += (ul_ + 3) & ~3;
            int tpf = h_ * w_;
            float *anchor = RF((size_t)C * tpf), *clean = K > 1 ? RF((size_t)C * (K - 1) * F * tpf) : NULL;
            const float *prop = RF((size_t)K * 10);
            float *nz_v = RF((size_t)C * K * F * tpf);
            float *nz_a = NULL;
            double ta = now();
            const char *pp[2] = { prompt, uncond }; int iu[2] = { 0, 1 }, Ss[2] = { 0, 0 };
            struct { praxis_runner **r; int *rel; int trim; } pa = { &r, &ring_released, trim };
            float **tx = text_cond_multi(&tcfg, pp, iu, 2, Ss, pre_qwen, &pa);
            if (!tx) status = -3;
            int S0 = Ss[0], S1 = Ss[1]; float *tx0 = tx ? tx[0] : NULL, *tx1 = tx ? tx[1] : NULL;
            if (!status && !r) {
                r = praxis_load(policy, 0, 0);
                if (!r) status = -4;
                else {
                    if (praxis_set_stream(r, praxis_nvfp4_enabled(r) ? ring : 0)) status = -5;
                    A = praxis_action_dim(r); F = praxis_lat_per_chunk(r);
                    fprintf(stderr, "[server] policy loaded in %.1f s (A=%d F=%d)\n", now() - ta, A, F);
                }
            }
            aux_ms = (now() - ta) * 1e3;
            if (!status && ring_released) { if (praxis_set_stream(r, praxis_nvfp4_enabled(r) ? ring : 0)) status = -5; ring_released = 0; }
            if (!status) {
                nz_a = RF((size_t)K * F * A);
                float *vout = (float *)malloc((size_t)C * F * tpf * 4), *aout = (float *)malloc((size_t)F * A * 4);
                if (praxis_sample_window_joint_ex2(r, tx0, S0, tx1, S1, anchor, h_, w_, clean, prop, nz_v, nz_a, K, steps, tsh, cv, ca, vout, aout, 0, skipv)) status = -6;
                hipDeviceSynchronize();
                if (!status) { int32_t hdr[5] = { C, F, h_, w_, A }; bufw(&body, hdr, 20); bufw(&body, vout, (size_t)C * F * tpf * 4); bufw(&body, aout, (size_t)F * A * 4); }
                free(vout); free(aout);
            }
            free(prompt); free(uncond);
            if (!status && trim) { praxis_trim(r); if (trim >= 2) { praxis_release_stream(r); ring_released = 1; } }   /* the device is free for the VAE / Qwen between calls */
        } else if (op == 5) {
            char s[512]; praxis_stats st; if (r) praxis_stats_get(r, &st);
            int k = snprintf(s, sizeof s, "policy_loaded=%d vae_loaded=%d dev_used_gb=%.3f idle_gb=%.3f policy_own_peak_gb=%.3f nvfp4=%d qwen_runs=%d backend=rdna4", r != NULL, vae != NULL, gb(dev_used()), gb(idle),
                             r ? gb(st.dev_bytes_peak) : 0.0, r ? praxis_nvfp4_enabled(r) : 0, g_qwen_runs);
            bufw(&body, s, k);
        } else status = -99;
        hipDeviceSynchronize();
        size_t peak = g_peak_used, after = dev_used(); if (after > peak) peak = after;
        double hd4[4] = { (now() - t0) * 1e3, aux_ms, gb(peak > idle ? peak - idle : 0), gb(after > idle ? after - idle : 0) };
        uint32_t rm = 0x52585250u; int32_t st32 = status; uint64_t rl = status ? 0 : 32 + body.n;
        wr_full(&rm, 4); wr_full(&st32, 4); wr_full(&rl, 8);
        if (!status) { wr_full(hd4, 32); if (body.n) wr_full(body.p, body.n); }
        fprintf(stderr, "[server] op %u status %d %.0f ms (aux %.0f), device peak %.2f GB / after %.2f GB over the idle context\n", op, status, hd4[0], aux_ms, hd4[2], hd4[3]);
        free(body.p); free(pl);
    }
    g_poll_run = 0; pthread_join(th, NULL);
    if (r) praxis_free(r);
    if (vae) hip_wan22_vae_free(vae);
    return 0;
}
