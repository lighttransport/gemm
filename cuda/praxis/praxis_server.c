/*
 * praxis_server.c - persistent praxis-1 policy server (binary protocol over stdin/stdout), for closed-loop use from cuda/praxis/praxis_client.py.
 *
 *   cuda/praxis/praxis_server --policy $PRAXIS_ROOT/praxis-1-droid [--vae DIR] [--qwen DIR] [--stream 2] [--text-cache DIR]
 *        [--vae-mode keep|reload] [--trim 0|1|2]
 *   env: PRAXIS_NVFP4=1 | PRAXIS_EXPERT_FMT=mixed|nvfp4|nvfp4a16, PRAXIS_PACK_CACHE=<file> (as praxis_e2e)
 *
 * Run from the repo root (the Qwen runner loads cubins relative to it). The protocol stream is a private dup of the original stdout; stdout is redirected
 * to stderr so every printf of the shared praxis_e2e helpers (text stage, tiles, ...) is just log output.
 *
 * What is resident across calls: the policy (lazy-loaded on the first SAMPLE), the text-conditioning cache (prompt -> Qwen3.5-2B hidden states; the Qwen weights
 * themselves are loaded, used and freed per cache miss, so they never overlap the policy if the prompt was warmed with TEXT before the first SAMPLE), and the
 * VAE (--vae-mode keep: weights resident; reload: loaded per ENCODE/DECODE and freed after, which is what keeps a resident policy + VAE encode under 8 GB
 * together with --trim 1: the policy's device workspace is released before the VAE loads and regrown after).
 *
 * Request  = u32 magic 'PRXQ' | u32 op | u64 len | payload.   Response = u32 'PRXR' | i32 status | u64 len | payload.
 * Every OK response payload starts with 4 f64: ms of the op, ms of load/aux work inside it, peak device GB over the idle context during the op, device GB
 * resident after the op (cuMemGetInfo, this GPU; polled every 4 ms), then the op body.
 *  op 1 ENCODE  : i32 T,H,W | u8 [3][T][H][W]   (tile clip, 0..255) -> i32 C,Tl,h,w | f32 [C][Tl][h][w] policy latent
 *  op 2 SAMPLE  : i32 K,h,w,steps,skip_last_video | f64 t_shift,cfg_video,cfg_action | i32 plen | prompt | i32 ulen | uncond prompt |
 *                 f32 anchor[C][1][h][w] | f32 clean[C][(K-1)*F][h][w] (K>1) | f32 proprio[K][10] | f32 noise_v[C][K*F][h][w] | f32 noise_a[K*F][A]
 *                 -> i32 C,F,h,w,A | f32 video[C][F][h][w] | f32 action[F][A]
 *  op 3 TEXT    : i32 n | n x (i32 len | bytes) -> empty (warms the text cache; Qwen load/free happens here)
 *  op 4 DECODE  : i32 Tl,h,w,keep | f32 latent[C][Tl][h][w] -> i32 n,H,W | u8 [n][H][W][3]   (last `keep` frames)
 *  op 5 INFO    : -> text
 *  op 6 QUIT
 */
#define main praxis_e2e_main
#include "praxis_e2e.c"
#undef main
#include <pthread.h>
#include <stdint.h>
#include "../../common/praxis_paths.h"

static int g_out_fd = 1;
static int rd_full(void *b, size_t n) { size_t o = 0; while (o < n) { ssize_t k = read(0, (char *)b + o, n - o); if (k <= 0) return -1; o += (size_t)k; } return 0; }
static int wr_full(const void *b, size_t n) { size_t o = 0; while (o < n) { ssize_t k = write(g_out_fd, (const char *)b + o, n - o); if (k <= 0) return -1; o += (size_t)k; } return 0; }

/* ---- device memory poller ---- */
static volatile size_t g_peak_used = 0; static volatile int g_poll_run = 1; static CUcontext g_ctx;
static void *poller(void *arg) { (void)arg; cuCtxSetCurrent(g_ctx); while (g_poll_run) { size_t u = dev_used(); if (u > g_peak_used) g_peak_used = u; usleep(4000); } return NULL; }

typedef struct { unsigned char *p; size_t n, cap; } buf_t;
static void *bufw(buf_t *b, const void *d, size_t n) { if (b->n + n > b->cap) { b->cap = (b->n + n) * 2; b->p = (unsigned char *)realloc(b->p, b->cap); } memcpy(b->p + b->n, d, n); b->n += n; return b->p + b->n - n; }

int main(int argc, char **argv) {
    const char *policy = NULL, *vae_dir = px_root("wan2.2-vae"), *qwen = px_root("qwen3.5-2b"), *cache = NULL, *vae_mode = "keep";
    int ring = 2, trim = 2, ring_released = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--policy") && i + 1 < argc) policy = argv[++i];
        else if (!strcmp(argv[i], "--vae") && i + 1 < argc) vae_dir = argv[++i];
        else if (!strcmp(argv[i], "--qwen") && i + 1 < argc) qwen = argv[++i];
        else if (!strcmp(argv[i], "--text-cache") && i + 1 < argc) cache = argv[++i];
        else if (!strcmp(argv[i], "--stream") && i + 1 < argc) ring = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vae-mode") && i + 1 < argc) vae_mode = argv[++i];
        else if (!strcmp(argv[i], "--trim") && i + 1 < argc) trim = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!policy) { fprintf(stderr, "usage: praxis_server --policy DIR [--vae DIR] [--qwen DIR] [--stream R] [--text-cache DIR] [--vae-mode keep|reload] [--trim 0|1|2]\n"); return 2; }
    int vae_keep = strcmp(vae_mode, "reload") != 0;
    g_out_fd = dup(1); dup2(2, 1); setvbuf(stdout, NULL, _IOLBF, 0);
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS || cuInit(0) != CUDA_SUCCESS) return 1;
    CUdevice dev; CUcontext cctx; cuDeviceGet(&dev, 0); cuDevicePrimaryCtxRetain(&cctx, dev); cuCtxSetCurrent(cctx); g_ctx = cctx;
    size_t idle = dev_used();
    real_cfg rc; if (read_real_cfg(policy, &rc) || !rc.ok) { fprintf(stderr, "cannot read normalisation from %s/config.json\n", policy); return 1; }
    pthread_t th; pthread_create(&th, NULL, poller, NULL);
    char tok_path[1300], tmp[1300]; snprintf(tok_path, sizeof tok_path, "%s/tokenizer.json", qwen);
    snprintf(tmp, sizeof tmp, "%s", px_out("praxis_server_tmp")); mkdirs(tmp);
    txt_cfg tcfg = { qwen, "cuda/llm/test_qwen35_st", cache, tmp, tok_path, 0 };
    cuda_wan22_vae *vae = NULL; praxis_runner *r = NULL; int C = 192, F = 6, A = 0;
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
            double ta = now(); size_t freed = 0;
            if (trim && r) { freed = praxis_trim(r); if (trim >= 2 && !ring_released) { praxis_release_stream(r); ring_released = 1; } }
            if (!vae) { vae = cuda_wan22_vae_load(vae_dir, 0, 0); if (!vae) status = -1; }
            aux_ms = (now() - ta) * 1e3; (void)freed;
            float *lat = NULL; int Tl = 0;
            if (!status && cuda_wan22_vae_encode_u8(vae, u8, T, H, W, &lat, &Tl)) status = -2;
            if (!status) { int32_t hdr[4] = { C, Tl, H / 32, W / 32 }; bufw(&body, hdr, 16); bufw(&body, lat, (size_t)C * Tl * (H / 32) * (W / 32) * 4); }
            free(lat);
            if (!vae_keep && vae) { cuda_wan22_vae_free(vae); vae = NULL; }
        } else if (op == 3) {   /* TEXT warm */
            int n; RI(n); if (n > 8) n = 8; static char pb[8][4096]; txt_src qs[8]; int Ss[8];
            for (int i = 0; i < n; i++) { int l; RI(l); if (l > 4095) l = 4095; memcpy(pb[i], pl + pos, l); pb[i][l] = 0; pos += (l + 3) & ~3; qs[i].prompt = pb[i]; qs[i].ids_path = NULL; }
            if (trim && r) { praxis_trim(r); if (trim >= 2 && !ring_released) { praxis_release_stream(r); ring_released = 1; } }   /* a resident policy gives its workspace back while Qwen is on the device */
            if (!text_cond_multi(&tcfg, qs, n, Ss)) status = -1;
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
            /* the action noise size depends on A, known once the policy is loaded */
            double ta = now();
            txt_src qs[2] = { { prompt, NULL }, { uncond, NULL } }; int Ss[2];
            float **tx = text_cond_multi(&tcfg, qs, 2, Ss);   /* cache hit when TEXT warmed it; else Qwen load/use/free (overlaps a resident policy) */
            if (!tx) status = -3;
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
                if (praxis_sample_window_joint_ex2(r, tx[0], Ss[0], tx[1], Ss[1], anchor, h_, w_, clean, prop, nz_v, nz_a, K, steps, tsh, cv, ca, vout, aout, 0, skipv)) status = -6;
                cuCtxSynchronize();
                if (!status) { int32_t hdr[5] = { C, F, h_, w_, A }; bufw(&body, hdr, 20); bufw(&body, vout, (size_t)C * F * tpf * 4); bufw(&body, aout, (size_t)F * A * 4); }
                free(vout); free(aout);
            }
            free(prompt); free(uncond);
        } else if (op == 4) {   /* DECODE */
            int Tl, h_, w_, keep; RI(Tl); RI(h_); RI(w_); RI(keep);
            const float *lat = (const float *)(pl + pos);
            double ta = now(); if (trim && r) { praxis_trim(r); if (trim >= 2 && !ring_released) { praxis_release_stream(r); ring_released = 1; } }
            if (!vae) { vae = cuda_wan22_vae_load(vae_dir, 0, 0); if (!vae) status = -1; }
            aux_ms = (now() - ta) * 1e3;
            float *fr = NULL; int To = 0;
            if (!status && cuda_wan22_vae_decode(vae, lat, Tl, h_, w_, &fr, &To)) status = -2;
            if (!status) {
                int H = 32 * h_, W = 32 * w_, n = keep < To ? keep : To; int32_t hdr[3] = { n, H, W }; bufw(&body, hdr, 12);
                unsigned char *u = (unsigned char *)malloc((size_t)n * H * W * 3);
                for (int t = 0; t < n; t++) for (int c = 0; c < 3; c++) { const float *s = fr + ((size_t)c * To + (To - n + t)) * H * W;
                    for (size_t p = 0; p < (size_t)H * W; p++) { float v = rintf((s[p] + 1.0f) * 127.5f); u[((size_t)t * H * W + p) * 3 + c] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v); } }
                bufw(&body, u, (size_t)n * H * W * 3); free(u);
            }
            free(fr);
            if (!vae_keep && vae) { cuda_wan22_vae_free(vae); vae = NULL; }
        } else if (op == 5) {
            char s[512]; praxis_stats st; if (r) praxis_stats_get(r, &st);
            int k = snprintf(s, sizeof s, "policy_loaded=%d vae_loaded=%d dev_used_gb=%.3f idle_gb=%.3f policy_own_peak_gb=%.3f nvfp4=%d qwen_runs=%d", r != NULL, vae != NULL, gb(dev_used()), gb(idle),
                             r ? gb(st.dev_bytes_peak) : 0.0, r ? praxis_nvfp4_enabled(r) : 0, g_qwen_runs);
            bufw(&body, s, k);
        } else status = -99;
        cuCtxSynchronize();
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
    if (vae) cuda_wan22_vae_free(vae);
    return 0;
}
