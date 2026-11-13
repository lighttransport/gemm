/*
 * praxis_e2e.c - end-to-end praxis-1 policy call on one GPU: task ids -> Qwen3.5-2B text conditioning -> observation frames
 * (exterior + wrist) -> tile canvas -> Wan2.2 VAE encode -> policy latents -> window_joint sampler -> normalised action rows
 * (+ optional VAE decode of the predicted video).
 *
 *   cuda/praxis/praxis_e2e --policy DIR | --synthetic DIR  --ids ids.npy [--ids-uncond ids_u.npy]
 *        --ext ext.npy --wrist wrist.npy  [--proprio p.npy] [--K 2] [--steps 2] [--cfg-video 4 --cfg-action 1]
 *        [--qwen /mnt/disk1/praxis/qwen3.5-2b] [--qwen-bin cuda/llm/test_qwen35_st] [--text-cache DIR]
 *        [--vae /mnt/disk1/praxis/wan2.2-vae] [--tile-h 256 --tile-w 416] [--noise-v f.npy --noise-a f.npy] [--seed N]
 *        [--stream R] [--pair] [--skip-last-video] [--ctx-cache] [--decode] [--keep-vae] [--out DIR]
 *
 * Run from the repo root (the Qwen runner loads cubins relative to it). Stage sequence, VRAM freed between stages:
 *   1. text: Qwen3.5-2B runs in a CHILD process (cuda/llm/test_qwen35_st: cuda_llm_prefill_hidden_all, ids padded to a multiple of 32, hidden_states[-3],
 *      padding rows dropped); its result is cached per prompt (--text-cache) and the process exit frees all of its device memory.
 *   2. VAE: load, encode the K'-1 clean chunks (21 + 24 (K'-2) raw frames of the tile canvas), free (unless --keep-vae).
 *   3. policy: load (sharded safetensors / NVFP4 via PRAXIS_NVFP4=1 / pack cache via PRAXIS_PACK_CACHE), sample, free.
 *   4. optional decode: VAE reload, decode [clean chunks | predicted chunk], keep the last 24 frames.
 * Actions are written NORMALISED (q99 / proprio normalisation needs the checkpoint's NormStats, which are not available): action_rows.npy
 * holds the 24 unpacked rows of the 10-D model space; see README for the (documented) un-normalisation.
 */
#define SAFETENSORS_IMPLEMENTATION
#define CUDA_WAN22_VAE_IMPLEMENTATION
#define CUDA_PRAXIS_IMPLEMENTATION
#include "cuda_wan22_vae.h"
#include "cuda_praxis_runner.h"
#include "praxis_qwen.h"
#include <sys/wait.h>
#include <sys/time.h>

/* ---------------- npy ---------------- */
typedef struct { void *data; char dt[8]; int nd; long long shape[8]; size_t n; } npy_t;
static int npy_load(const char *path, npy_t *o) {
    memset(o, 0, sizeof *o);
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    unsigned char m[10];
    if (fread(m, 1, 10, f) != 10 || memcmp(m, "\x93NUMPY", 6)) { fclose(f); return -1; }
    int hl = m[6] == 1 ? (m[8] | (m[9] << 8)) : 0;
    if (m[6] != 1) { fclose(f); fprintf(stderr, "%s: npy v%d unsupported\n", path, m[6]); return -1; }
    char *h = (char *)calloc(1, hl + 1);
    if (fread(h, 1, hl, f) != (size_t)hl) { fclose(f); free(h); return -1; }
    char *d = strstr(h, "'descr'"); if (!d) { fclose(f); free(h); return -1; }
    d = strchr(d + 7, '\''); if (!d) { fclose(f); free(h); return -1; }
    sscanf(d + 1, "%7[^']", o->dt);
    char *p = strchr(strstr(h, "shape"), '(') + 1;
    o->n = 1;
    while (*p && *p != ')') { while (*p == ' ' || *p == ',') p++; if (*p == ')') break; o->shape[o->nd] = strtoll(p, &p, 10); o->n *= o->shape[o->nd]; o->nd++; }
    free(h);
    size_t es = !strcmp(o->dt, "<f4") ? 4 : !strcmp(o->dt, "<i8") ? 8 : !strcmp(o->dt, "<i4") ? 4 : !strcmp(o->dt, "|u1") ? 1 : 0;
    if (!es) { fprintf(stderr, "%s: dtype %s unsupported\n", path, o->dt); fclose(f); return -1; }
    o->data = malloc(o->n * es + 1);
    if (fread(o->data, es, o->n, f) != o->n) { fclose(f); free(o->data); return -1; }
    fclose(f);
    return 0;
}
static void npy_save(const char *path, const void *d, const char *dt, size_t es, int nd, const long long *shape) {
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    char hdr[256], sh[200] = ""; size_t n = 1;
    for (int i = 0; i < nd; i++) { char t[32]; snprintf(t, sizeof t, "%lld, ", shape[i]); strcat(sh, t); n *= shape[i]; }
    int hl = snprintf(hdr, sizeof hdr, "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", dt, sh);
    int pad = (64 - (10 + hl + 1) % 64) % 64; unsigned short hlen = (unsigned short)(hl + pad + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f); fwrite(&hlen, 2, 1, f); fwrite(hdr, 1, hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f); fwrite(d, es, n, f); fclose(f);
}
static void save_f32(const char *dir, const char *name, const float *d, int nd, const long long *sh) {
    char p[1300]; snprintf(p, sizeof p, "%s/%s.npy", dir, name); npy_save(p, d, "<f4", 4, nd, sh);
}
static float *npy_f32(const char *path, npy_t *o) {
    if (npy_load(path, o)) return NULL;
    if (strcmp(o->dt, "<f4")) { fprintf(stderr, "%s: expected float32\n", path); return NULL; }
    return (float *)o->data;
}
static void mkdirs(const char *path) {
    char b[1024]; snprintf(b, sizeof b, "%s", path);
    for (char *p = b + 1; *p; p++) if (*p == '/') { *p = 0; mkdir(b, 0755); *p = '/'; }
    mkdir(b, 0755);
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double gb(size_t b) { return b / 1073741824.0; }
/* device memory of THIS process only (nvidia-smi per-process accounting; cuMemGetInfo also sees other processes on a shared GPU). 0 if unavailable. */
static size_t proc_used(void) {
    char cmd[200], l[256]; snprintf(cmd, sizeof cmd, "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null");
    FILE *f = popen(cmd, "r"); size_t mb = 0; if (!f) return 0;
    while (fgets(l, sizeof l, f)) { int pid = 0; long m = 0; if (sscanf(l, "%d, %ld", &pid, &m) == 2 && pid == (int)getpid()) mb = (size_t)m; }
    pclose(f); return mb << 20;
}
static size_t dev_used(void) { size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot); return tot - fr; }

/* ---------------- tile canvas (single_arm.tile_clip, antialias off: aspect centre crop + one bilinear resize, round, uint8) ---------------- */
static void aspect_box(int sh, int sw, int th, int tw, double tol, int *y0, int *y1, int *x0, int *x1) {
    double sar = (double)sw / sh, tar = (double)tw / th;
    *y0 = 0; *y1 = sh; *x0 = 0; *x1 = sw;
    if (fabs(tar / sar - 1.0) <= tol) return;
    if (sar > tar) { int nw = (int)nearbyint(sh * tar); *x0 = (sw - nw) / 2; *x1 = *x0 + nw; }
    else { int nh = (int)nearbyint(sw / tar); *y0 = (sh - nh) / 2; *y1 = *y0 + nh; }
}
/* frames [T][H][W][3] u8 -> out [3][T][th][2*tw] float (canvas side offset xoff), values 0..255 */
static void tile_view(const unsigned char *fr, int T, int H, int W, int th, int tw, int xoff, int cw, float *out) {
    int y0, y1, x0, x1; aspect_box(H, W, th, tw, 0.03, &y0, &y1, &x0, &x1);
    int ch = y1 - y0, cwid = x1 - x0;
    float sy = (float)ch / th, sx = (float)cwid / tw;
    for (int t = 0; t < T; t++) {
        const unsigned char *f = fr + (size_t)t * H * W * 3;
        for (int y = 0; y < th; y++) {
            float fy = sy * (y + 0.5f) - 0.5f; if (fy < 0) fy = 0;
            int iy0 = (int)fy; if (iy0 > ch - 1) iy0 = ch - 1; int iy1 = iy0 + 1 < ch ? iy0 + 1 : ch - 1; float ly = fy - iy0;
            for (int x = 0; x < tw; x++) {
                float fx = sx * (x + 0.5f) - 0.5f; if (fx < 0) fx = 0;
                int ix0 = (int)fx; if (ix0 > cwid - 1) ix0 = cwid - 1; int ix1 = ix0 + 1 < cwid ? ix0 + 1 : cwid - 1; float lx = fx - ix0;
                for (int c = 0; c < 3; c++) {
                    #define PX(yy, xx) ((float)f[((size_t)(y0 + (yy)) * W + (x0 + (xx))) * 3 + c])
                    float top = PX(iy0, ix0) * (1 - lx) + PX(iy0, ix1) * lx, bot = PX(iy1, ix0) * (1 - lx) + PX(iy1, ix1) * lx;
                    float v = rintf(top * (1 - ly) + bot * ly);
                    v = v < 0 ? 0 : v > 255 ? 255 : v;
                    out[(((size_t)c * T + t) * th + y) * cw + xoff + x] = v;
                }
            }
        }
    }
}


/* ---- antialiased tiles (video.antialias, single_arm._fit_view): centre crop to the native aspect (720x1280), torch bilinear antialias resize
 * (align_corners=False, float32, horizontal pass then vertical) to fit_resize_size(tile_w), round/clamp, centre trim to the tile. ---- */
typedef struct { int xmin, n; float *w; } aa_tap;
static aa_tap *aa_weights(int in, int out) {
    double scale = (double)in / out, support = scale >= 1.0 ? scale : 1.0, invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    aa_tap *t = (aa_tap *)calloc(out, sizeof(aa_tap));
    for (int i = 0; i < out; i++) {
        double center = scale * (i + 0.5);
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > in) xmax = in;
        int n = xmax - xmin; t[i].xmin = xmin; t[i].n = n; t[i].w = (float *)malloc(4 * (n > 0 ? n : 1));
        double tot = 0;
        for (int j = 0; j < n; j++) { double x = (j + xmin - center + 0.5) * invscale; double w = 1.0 - fabs(x); if (w < 0) w = 0; t[i].w[j] = (float)w; tot += w; }
        for (int j = 0; j < n; j++) t[i].w[j] = (float)(t[i].w[j] / tot);
    }
    return t;
}
static void aa_free(aa_tap *t, int n) { for (int i = 0; i < n; i++) free(t[i].w); free(t); }
static void tile_view_fit(const unsigned char *fr, int T, int H, int W, int th, int tw, int xoff, int cw, float *out) {
    const int NH = 720, NW = 1280;
    int rh = NH < tw ? NH : tw; { int a = NH * tw / NW; if (a < rh) rh = a; } int rw = NW < tw ? NW : tw;
    int y0, y1, x0, x1; aspect_box(H, W, NH, NW, 0.03, &y0, &y1, &x0, &x1);
    int ch = y1 - y0, cwid = x1 - x0;
    int ty0 = (rh - th) / 2, tx0 = (rw - tw) / 2;
    aa_tap *wx = NULL, *wy = NULL; int need = !(ch == rh && cwid == rw);
    if (need) { wx = aa_weights(cwid, rw); wy = aa_weights(ch, rh); }
    float *tmp = need ? (float *)malloc((size_t)ch * rw * 4) : NULL;
    for (int t = 0; t < T; t++) for (int c = 0; c < 3; c++) {
        const unsigned char *f = fr + (size_t)t * H * W * 3;
        float *o = out + (((size_t)c * T + t) * th) * cw + xoff;
        if (!need) {
            for (int y = 0; y < th; y++) for (int x = 0; x < tw; x++) o[(size_t)y * cw + x] = f[((size_t)(y0 + ty0 + y) * W + (x0 + tx0 + x)) * 3 + c];
            continue;
        }
        for (int y = 0; y < ch; y++) for (int x = 0; x < rw; x++) {
            float a = 0; const aa_tap *tp = &wx[x];
            for (int j = 0; j < tp->n; j++) a += tp->w[j] * (float)f[((size_t)(y0 + y) * W + (x0 + tp->xmin + j)) * 3 + c];
            tmp[(size_t)y * rw + x] = a;
        }
        for (int y = 0; y < rh; y++) {
            if (y < ty0 || y >= ty0 + th) continue;
            const aa_tap *tp = &wy[y];
            for (int x = tx0; x < tx0 + tw; x++) {
                float a = 0; for (int j = 0; j < tp->n; j++) a += tp->w[j] * tmp[(size_t)(tp->xmin + j) * rw + x];
                float v = rintf(a); v = v < 0 ? 0 : v > 255 ? 255 : v;
                o[(size_t)(y - ty0) * cw + (x - tx0)] = v;
            }
        }
    }
    if (need) { aa_free(wx, rw); aa_free(wy, rh); free(tmp); }
}

/* ---- checkpoint config: normalisation, video, chunking (config.json) ---- */
typedef struct { double q01[10], q99[10], pmean[10], pstd[10]; int q99dims[10], nq99dims, ok, antialias, tile_h, tile_w, n_chunks, lat, nfut_rows; double t_shift; } real_cfg;
static int read_real_cfg(const char *dir, real_cfg *o) {
    memset(o, 0, sizeof *o); o->antialias = 0; o->tile_h = 0; o->tile_w = 0;
    char path[1300]; snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(n + 1); if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return -1; } fclose(f); buf[n] = 0;
    json_val *root = json_parse(buf, (int)n); free(buf); if (!root) return -1;
    json_val *no = json_obj_get(root, "normalization"), *vi = json_obj_get(root, "video"), *ch = json_obj_get(root, "chunking");
    if (vi) { json_val *a = json_obj_get(vi, "antialias"); o->antialias = a && a->type == JSON_TRUE;
              json_val *h = json_obj_get(vi, "tile_height"), *w = json_obj_get(vi, "tile_width"); if (h) o->tile_h = (int)h->num; if (w) o->tile_w = (int)w->num; }
    if (ch) { json_val *a = json_obj_get(ch, "n_chunks"), *b = json_obj_get(ch, "latents_per_chunk"); if (a) o->n_chunks = (int)a->num; if (b) o->lat = (int)b->num; }
    if (no) {
        const char *keys[4] = { "action_q01", "action_q99", "proprio_mean", "proprio_std" }; double *dst[4] = { o->q01, o->q99, o->pmean, o->pstd };
        int ok = 1;
        for (int k = 0; k < 4; k++) { json_val *a = json_obj_get(no, keys[k]); if (!a || a->arr.count != 10) { ok = 0; break; } for (int i = 0; i < 10; i++) dst[k][i] = a->arr.items[i].num; }
        json_val *d = json_obj_get(no, "q99_normed_dims"); if (d) { o->nq99dims = d->arr.count > 10 ? 10 : d->arr.count; for (int i = 0; i < o->nq99dims; i++) o->q99dims[i] = (int)d->arr.items[i].num; }
        o->ok = ok;
    }
    json_free(root);
    return 0;
}
/* pose8 [xyz3, quat_wxyz4, grip] -> normalised 10-D proprio (robot_io.pose8_to_10d + proprio_normalize) */
static void quat_to_mat(const double *q, double R[9]) {
    double n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) + 1e-8; double w = q[0]/n, x = q[1]/n, y = q[2]/n, z = q[3]/n;
    R[0] = 1 - 2*(y*y + z*z); R[1] = 2*(x*y - w*z); R[2] = 2*(x*z + w*y); R[3] = 2*(x*y + w*z); R[4] = 1 - 2*(x*x + z*z); R[5] = 2*(y*z - w*x);
    R[6] = 2*(x*z - w*y); R[7] = 2*(y*z + w*x); R[8] = 1 - 2*(x*x + y*y);
}
static void proprio_from_pose8(const real_cfg *c, const double *p8, float *out) {
    double R[9], v[10]; quat_to_mat(p8 + 3, R);
    v[0] = p8[0]; v[1] = p8[1]; v[2] = p8[2];
    v[3] = R[0]; v[4] = R[3]; v[5] = R[6]; v[6] = R[1]; v[7] = R[4]; v[8] = R[7]; v[9] = p8[7];   /* rot6d = first two columns */
    for (int i = 0; i < 10; i++) { float s = (float)c->pstd[i]; if (s < 1e-6f) s = 1e-6f; out[i] = ((float)v[i] - (float)c->pmean[i]) / s; }
}
/* normalised rows -> relative 10-D actions: clip q99 dims to [-1,1], (n+1)/2*(q99-q01)+q01, clip grip [0,1] (robot_io.denormalize_rows + clip_grip) */
static void unnormalise_rows(const real_cfg *c, const float *rows, int nrows, float *rel) {
    for (int r = 0; r < nrows; r++) {
        for (int i = 0; i < 10; i++) rel[r * 10 + i] = rows[r * 10 + i];
        for (int k = 0; k < c->nq99dims; k++) {
            int d = c->q99dims[k]; float n = rows[r * 10 + d]; n = n < -1 ? -1 : n > 1 ? 1 : n;
            rel[r * 10 + d] = (n + 1.0f) / 2.0f * ((float)c->q99[d] - (float)c->q01[d]) + (float)c->q01[d];
        }
        float g = rel[r * 10 + 9]; rel[r * 10 + 9] = g < 0 ? 0 : g > 1 ? 1 : g;
    }
}
/* relative rows + current pose8 -> absolute flange-frame poses [xyz3, R row-major 9, grip] (single_arm.rel10_to_wire without the client-frame conversion) */
static void rot6d_to_mat(const float *r6, double R[9]) {
    double a0[3], a1[3], c0[3], c1[3], c2[3]; for (int i = 0; i < 3; i++) { a0[i] = r6[i]; a1[i] = r6[3 + i]; }
    double n0 = sqrt(a0[0]*a0[0] + a0[1]*a0[1] + a0[2]*a0[2]) + 1e-8; for (int i = 0; i < 3; i++) c0[i] = a0[i] / n0;
    double d = c0[0]*a1[0] + c0[1]*a1[1] + c0[2]*a1[2]; for (int i = 0; i < 3; i++) a1[i] -= d * c0[i];
    double n1 = sqrt(a1[0]*a1[0] + a1[1]*a1[1] + a1[2]*a1[2]) + 1e-8; for (int i = 0; i < 3; i++) c1[i] = a1[i] / n1;
    c2[0] = c0[1]*c1[2] - c0[2]*c1[1]; c2[1] = c0[2]*c1[0] - c0[0]*c1[2]; c2[2] = c0[0]*c1[1] - c0[1]*c1[0];
    for (int i = 0; i < 3; i++) { R[i*3+0] = c0[i]; R[i*3+1] = c1[i]; R[i*3+2] = c2[i]; }
}
static void rel_to_abs(const float *rel, int nrows, const double *p8, float *abs12) {
    double q[4] = { p8[3], p8[4], p8[5], p8[6] }; double n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    double w = q[0]/n, x = q[1]/n, y = q[2]/n, z = q[3]/n, Ra[9] = { x*x - y*y - z*z + w*w, 2*(x*y - z*w), 2*(x*z + y*w), 2*(x*y + z*w), -x*x + y*y - z*z + w*w, 2*(y*z - x*w),
                                                                      2*(x*z - y*w), 2*(y*z + x*w), -x*x - y*y + z*z + w*w };
    for (int r = 0; r < nrows; r++) {
        double Rr[9]; rot6d_to_mat(rel + r * 10 + 3, Rr);
        for (int i = 0; i < 3; i++) abs12[r * 13 + i] = (float)((double)rel[r * 10 + i] + p8[i]);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { double a = 0; for (int k = 0; k < 3; k++) a += Rr[i*3+k] * Ra[k*3+j]; abs12[r * 13 + 3 + i * 3 + j] = (float)a; }
        abs12[r * 13 + 12] = rel[r * 10 + 9];
    }
}

/* ---------------- Qwen text stage ----------------
 * Default: IN-PROCESS (praxis_qwen.c: cuda/llm runner linked into this binary, same CUDA context). All cache misses of a call are computed in ONE
 * load -> prefill(s) -> free cycle, so Qwen's device memory (2.9 GB) is gone before the VAE/policy load (peak preserved). --qwen-spawn keeps the old
 * child process (cuda/llm/test_qwen35_st, full reload per prompt). Results are cached in memory (key = prompt string or ids hash, hit on repeated prompts
 * in a --session) and on disk (--text-cache, key = fnv(ids), shared by both paths). */
static unsigned long long fnv(const void *p, size_t n) { unsigned long long h = 1469598103934665603ULL; const unsigned char *b = (const unsigned char *)p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; } return h; }
typedef struct { char *key; float *d; int S; } tc_ent;
static tc_ent g_tc[512]; static int g_ntc;
static double g_text_peak_gb = 0, g_qwen_leak_gb = 0, g_t_tok = 0, g_t_qload = 0, g_t_qrun = 0, g_t_qfree = 0; static int g_qwen_runs = 0;
static praxis_tok *g_tok = NULL;
typedef struct { const char *prompt, *ids_path; } txt_src;   /* exactly one is set */
typedef struct { const char *qwen, *qbin, *cache, *tmp, *tokjson; int spawn; } txt_cfg;
static tc_ent *tc_find(const char *key) { for (int i = 0; i < g_ntc; i++) if (!strcmp(g_tc[i].key, key)) return &g_tc[i]; return NULL; }
/* resolve ids of a request (malloc'd int32), via the C tokenizer or an ids .npy */
static int32_t *src_ids(const txt_cfg *c, const txt_src *q, int *n) {
    if (q->prompt) {
        double t0 = now();
        if (!g_tok) { g_tok = praxis_tok_load(c->tokjson); if (!g_tok) return NULL; }
        int cap = (int)strlen(q->prompt) + 16; int32_t *ids = (int32_t *)malloc((size_t)cap * 4);
        *n = praxis_tok_encode(g_tok, q->prompt, ids, cap); g_t_tok += now() - t0;
        if (*n < 0) { free(ids); return NULL; }
        return ids;
    }
    npy_t t; if (npy_load(q->ids_path, &t)) return NULL;
    int32_t *ids = (int32_t *)malloc(t.n * 4); int is8 = !strcmp(t.dt, "<i8");
    for (size_t i = 0; i < t.n; i++) ids[i] = is8 ? (int32_t)((long long *)t.data)[i] : ((int32_t *)t.data)[i];
    *n = (int)t.n; free(t.data); return ids;
}
/* returns cond [S][2048] (cached, do not free) or NULL */
static float **text_cond_multi(const txt_cfg *c, const txt_src *qs, int nq, int *Sout) {
    static float *res[8]; int32_t *ids[8] = {0}; int nid[8] = {0}; unsigned long long h[8] = {0}; char *key[8] = {0}; int pend[8] = {0};
    int npend = 0;
    for (int i = 0; i < nq; i++) {
        char kb[1300];
        if (qs[i].prompt) snprintf(kb, sizeof kb, "p:%s", qs[i].prompt);
        else { npy_t t; if (npy_load(qs[i].ids_path, &t)) return NULL; snprintf(kb, sizeof kb, "f:%016llx", fnv(t.data, t.n * (!strcmp(t.dt, "<i8") ? 8 : 4))); free(t.data); }
        key[i] = strdup(kb);
        tc_ent *e = tc_find(kb);
        if (e) { printf("[text] memory cache hit (%.40s%s)\n", kb, strlen(kb) > 40 ? "..." : ""); res[i] = e->d; Sout[i] = e->S; continue; }
        ids[i] = src_ids(c, &qs[i], &nid[i]); if (!ids[i]) return NULL;
        h[i] = fnv(ids[i], (size_t)nid[i] * 4);
        /* disk cache is keyed like before: fnv over the ids as int64 when they came from an int64 .npy, else int32; use the int32 hash of the ids for prompts */
        char cp[1300]; snprintf(cp, sizeof cp, "%s/text_%016llx.npy", c->cache ? c->cache : c->tmp, h[i]);
        npy_t ct;
        if (access(cp, R_OK) == 0 && !npy_load(cp, &ct)) { printf("[text] disk cache hit %s\n", cp); res[i] = (float *)ct.data; Sout[i] = (int)ct.shape[0]; }
        else { pend[i] = 1; npend++; continue; }
        if (g_ntc < 512) { g_tc[g_ntc].key = strdup(kb); g_tc[g_ntc].d = res[i]; g_tc[g_ntc].S = Sout[i]; g_ntc++; }
    }
    if (npend) {
        size_t base = dev_used(); double t0 = now(); size_t peak = base;
        if (!c->spawn) {
            int mx = 2048; for (int i = 0; i < nq; i++) if (pend[i] && (nid[i] + 31) / 32 * 32 > mx) mx = (nid[i] + 31) / 32 * 32;
            size_t pbase = proc_used(); double tq = now(); praxis_qwen *q = praxis_qwen_open(c->qwen, mx); if (!q) { fprintf(stderr, "in-process Qwen load failed\n"); return NULL; }
            g_t_qload += now() - tq; { size_t u = dev_used(); if (u > peak) peak = u; }
            size_t ppeak = proc_used();
            for (int i = 0; i < nq; i++) if (pend[i]) {
                double tr = now(); float *o = praxis_qwen_cond(q, ids[i], nid[i]);
                if (!o) { fprintf(stderr, "Qwen prefill failed\n"); return NULL; }
                cuCtxSynchronize(); g_t_qrun += now() - tr; g_qwen_runs++;
                { size_t u = dev_used(); if (u > peak) peak = u; size_t pu = proc_used(); if (pu > ppeak) ppeak = pu; }
                res[i] = (float *)malloc((size_t)nid[i] * 2048 * 4); memcpy(res[i], o, (size_t)nid[i] * 2048 * 4); free(o); Sout[i] = nid[i];
            }
            double tf = now(); praxis_qwen_close(q); cuCtxSynchronize(); g_t_qfree += now() - tf;
            size_t after = dev_used(); (void)after; size_t pafter = proc_used(); g_qwen_leak_gb = pafter ? (pafter >= pbase ? gb(pafter - pbase) : -gb(pbase - pafter)) : 0;
            g_text_peak_gb = gb(peak - base); double pk = ppeak > pbase ? gb(ppeak - pbase) : 0;
            printf("[text] Qwen3.5-2B in-process: load %.2f s, %d prefill(s) %.0f ms, free %.0f ms; device peak %.2f GB over the context (this process only: %.2f GB); process memory after free %+.3f GB vs before load (nvidia-smi per-process)\n",
                   g_t_qload, npend, g_t_qrun * 1e3, g_t_qfree * 1e3, g_text_peak_gb, pk, g_qwen_leak_gb);
        } else for (int i = 0; i < nq; i++) if (pend[i]) {
            char od[1300], ip[1300]; snprintf(od, sizeof od, "%s/qwen_%016llx", c->tmp, h[i]); snprintf(ip, sizeof ip, "%s/ids_%016llx.npy", c->tmp, h[i]);
            { long long *l = (long long *)malloc((size_t)nid[i] * 8); for (int k = 0; k < nid[i]; k++) l[k] = ids[i][k]; long long sh[1] = { nid[i] }; npy_save(ip, l, "<i8", 8, 1, sh); free(l); }
            double tc0 = now(); pid_t pid = fork();
            if (pid == 0) { execl(c->qbin, c->qbin, c->qwen, "-i", ip, "-o", od, "-p", "32", "--cond-only", (char *)NULL); fprintf(stderr, "exec %s failed\n", c->qbin); _exit(127); }
            int st = 0; while (waitpid(pid, &st, WNOHANG) == 0) { size_t u = dev_used(); if (u > peak) peak = u; usleep(10000); }
            if (!WIFEXITED(st) || WEXITSTATUS(st)) { fprintf(stderr, "Qwen child failed (status %d)\n", st); return NULL; }
            char np[1300]; snprintf(np, sizeof np, "%s/cond.npy", od); npy_t ct; if (npy_load(np, &ct)) return NULL;
            res[i] = (float *)ct.data; Sout[i] = (int)ct.shape[0]; g_qwen_runs++;
            g_text_peak_gb = gb(peak - base);
            printf("[text] Qwen3.5-2B child: %.2f s (incl. load), %d tokens, device peak %.2f GB over the idle context (polled)\n", now() - tc0, Sout[i], g_text_peak_gb);
            g_t_qrun += now() - tc0;
        }
        for (int i = 0; i < nq; i++) if (pend[i]) {
            char cp[1300]; snprintf(cp, sizeof cp, "%s/text_%016llx.npy", c->cache ? c->cache : c->tmp, h[i]); mkdirs(c->cache ? c->cache : c->tmp);
            long long sh[2] = { Sout[i], 2048 }; npy_save(cp, res[i], "<f4", 4, 2, sh);
            if (g_ntc < 512) { g_tc[g_ntc].key = strdup(key[i]); g_tc[g_ntc].d = res[i]; g_tc[g_ntc].S = Sout[i]; g_ntc++; }
        }
        (void)t0;
    }
    for (int i = 0; i < nq; i++) { free(ids[i]); free(key[i]); }
    return res;
}

/* ---------------- noise ---------------- */
static unsigned long long rs_ = 88172645463325252ULL;
static float rnd_n(void) { double u1, u2; rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; u1 = ((rs_ >> 11) + 1) * (1.0 / 9007199254740993.0);
    rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; u2 = (rs_ >> 11) * (1.0 / 9007199254740992.0); return (float)(sqrt(-2 * log(u1)) * cos(6.283185307179586 * u2)); }

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(double *v, int n) { if (n <= 0) return 0; double *t = (double *)malloc(n * sizeof *t); memcpy(t, v, n * sizeof *t); qsort(t, n, sizeof *t, cmpd); double m = t[n / 2]; free(t); return m; }
typedef struct { double text, tiles, vae, pol_load, samp, post, total; } call_t;

int main(int argc, char **argv) {
    const char *policy = NULL, *synth = NULL, *ids_p = NULL, *ids_u_p = NULL, *ext_p = NULL, *wr_p = NULL, *prop_p = NULL, *noise_v_p = NULL, *noise_a_p = NULL, *pose8_p = NULL;
    const char *tok_test = NULL, *prompt = NULL, *prompt_u = " ", *tokjson = NULL, *session = NULL;
    const char *qwen = "/mnt/disk1/praxis/qwen3.5-2b", *qbin = "cuda/llm/test_qwen35_st", *cache = NULL, *vae_dir = "/mnt/disk1/praxis/wan2.2-vae", *out = "tmp/praxis_out/e2e";
    int repeat = 1, load_only = 0, aa_force = -1, cfg_tile = 1,  K = 2, steps = 2, th = 256, tw = 416, ring = 2, pair = 0, skipv = 0, ctx = 0, decode = 0, keep_vae = 0, qspawn = 0; unsigned long long seed = 1;
    double cfg_v = 4.0, cfg_a = 1.0, t_shift = 12.5;
    for (int i = 1; i < argc; i++) {
        #define ARG(n) (!strcmp(argv[i], n) && i + 1 < argc)
        if ARG("--policy") policy = argv[++i]; else if ARG("--synthetic") synth = argv[++i];
        else if ARG("--ids") ids_p = argv[++i]; else if ARG("--ids-uncond") ids_u_p = argv[++i];
        else if ARG("--prompt") prompt = argv[++i]; else if ARG("--prompt-uncond") prompt_u = argv[++i]; else if ARG("--tokenizer") tokjson = argv[++i]; else if ARG("--session") session = argv[++i];
        else if (!strcmp(argv[i], "--qwen-spawn")) qspawn = 1; else if ARG("--tok-test") tok_test = argv[++i];
        else if ARG("--ext") ext_p = argv[++i]; else if ARG("--wrist") wr_p = argv[++i]; else if ARG("--proprio") prop_p = argv[++i];
        else if ARG("--noise-v") noise_v_p = argv[++i]; else if ARG("--noise-a") noise_a_p = argv[++i];
        else if ARG("--qwen") qwen = argv[++i]; else if ARG("--qwen-bin") qbin = argv[++i]; else if ARG("--text-cache") cache = argv[++i];
        else if ARG("--vae") vae_dir = argv[++i]; else if ARG("--out") out = argv[++i];
        else if ARG("--K") K = atoi(argv[++i]); else if ARG("--steps") steps = atoi(argv[++i]); else if ARG("--tile-h") th = atoi(argv[++i]);
        else if ARG("--tile-w") tw = atoi(argv[++i]); else if ARG("--stream") ring = atoi(argv[++i]); else if ARG("--seed") seed = strtoull(argv[++i], 0, 10);
        else if ARG("--cfg-video") cfg_v = atof(argv[++i]); else if ARG("--cfg-action") cfg_a = atof(argv[++i]); else if ARG("--t-shift") t_shift = atof(argv[++i]);
        else if (!strcmp(argv[i], "--load-only")) load_only = 1; else if ARG("--repeat") repeat = atoi(argv[++i]); else if ARG("--pose8") pose8_p = argv[++i];
        else if ARG("--antialias") aa_force = atoi(argv[++i]); else if (!strcmp(argv[i], "--no-config-tile")) cfg_tile = 0; else if (!strcmp(argv[i], "--pair")) pair = 1; else if (!strcmp(argv[i], "--skip-last-video")) skipv = 1;
        else if (!strcmp(argv[i], "--ctx-cache")) ctx = 1; else if (!strcmp(argv[i], "--decode")) decode = 1; else if (!strcmp(argv[i], "--keep-vae")) keep_vae = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (tok_test) {   /* CPU only: one prompt per line (\\n, \\t, \\\\ escapes), prints "ids: i i i ..." per line for comparison against HF */
        char tj[1300]; snprintf(tj, sizeof tj, "%s/tokenizer.json", qwen); praxis_tok *tk = praxis_tok_load(tokjson ? tokjson : tj); if (!tk) return 1;
        FILE *f = fopen(tok_test, "r"); if (!f) return 1; char line[8192], un[8192]; double t0 = now(); int nl = 0;
        while (fgets(line, sizeof line, f)) {
            size_t n = strlen(line); while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
            size_t o = 0; for (size_t i = 0; i < n; i++) { if (line[i] == '\\' && i + 1 < n) { i++; un[o++] = line[i] == 'n' ? '\n' : line[i] == 't' ? '\t' : line[i] == 'r' ? '\r' : line[i]; } else un[o++] = line[i]; } un[o] = 0;
            int32_t ids[4096]; int k = praxis_tok_encode(tk, un, ids, 4096); printf("ids:"); for (int i = 0; i < k; i++) printf(" %d", ids[i]); printf("\n"); nl++; }
        fprintf(stderr, "tokenized %d lines in %.1f ms (after load)\n", nl, (now() - t0) * 1e3); return 0;
    }
    if (load_only && policy) {
        double t0 = now();
        if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS || cuInit(0) != CUDA_SUCCESS) return 1;
        praxis_runner *r0 = praxis_load(policy, 0, 1);
        if (!r0) return 1;
        printf("load-only: %.2f s, device in use %.2f GB\n", now() - t0, gb(dev_used()));
        { FILE *f = fopen("/proc/self/status", "r"); char l[256]; while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS", 5) || !strncmp(l, "VmHWM", 5)) fputs(l, stdout); if (f) fclose(f); }
        praxis_free(r0); return 0;
    }
    if ((!policy && !synth) || (!ids_p && !prompt && !session) || !ext_p || !wr_p) {
        fprintf(stderr, "usage: praxis_e2e (--policy DIR | --synthetic DIR) (--ids ids.npy [--ids-uncond u.npy] | --prompt \"text\" [--prompt-uncond \" \"] | --session FILE|-) --ext e.npy --wrist w.npy [...]\n"
                        "  text: in-process Qwen3.5-2B by default (--qwen-spawn = old child process per prompt); --prompt uses the C tokenizer (--tokenizer tokenizer.json, default <qwen>/tokenizer.json)\n"
                        "  --session FILE|-: one prompt per line (blank/# lines skipped); policy weights, VAE weights and the prompt cache stay resident across calls\n"); return 2; }
    if (K < 2) { fprintf(stderr, "K' >= 2 (the context needs at least one clean chunk)\n"); return 2; }
    if (ctx) setenv("PRAXIS_CTX_CACHE", "1", 1);
    mkdirs(out);
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS || cuInit(0) != CUDA_SUCCESS) return 1;
    CUdevice dev; CUcontext cctx; cuDeviceGet(&dev, 0); cuDevicePrimaryCtxRetain(&cctx, dev); cuCtxSetCurrent(cctx);
    real_cfg rc; int have_rc = 0;
    if (policy && cfg_tile) {
        have_rc = read_real_cfg(policy, &rc) == 0;
        if (have_rc) {
            if (aa_force < 0) aa_force = rc.antialias;
            if (rc.tile_h && th == 256) th = rc.tile_h;   /* the checkpoint's tile size unless --tile-h/--tile-w was given */
            if (rc.tile_w && tw == 416) tw = rc.tile_w;
        }
    }
    if (aa_force < 0) aa_force = 0;
    size_t idle = dev_used();
    printf("[cfg] tile %dx%d, antialias(fit rule) %d, normalisation %s\n", th, tw, aa_force, have_rc && rc.ok ? "from config.json" : "none (rows stay normalised)");
    printf("device in use at start (CUDA context only): %.2f GB\n", gb(idle));
    double T0 = now();

    /* ---- prompts of the calls ---- */
    char **plist = NULL; int ncalls = 1;
    if (session) {
        FILE *f = !strcmp(session, "-") ? stdin : fopen(session, "r"); if (!f) { fprintf(stderr, "cannot open session file %s\n", session); return 1; }
        char line[8192]; plist = (char **)calloc(1024, sizeof(char *)); ncalls = 0;
        while (ncalls < 1024 && fgets(line, sizeof line, f)) { size_t n = strlen(line); while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0; if (!n || line[0] == '#') continue; plist[ncalls++] = strdup(line); }
        if (f != stdin) fclose(f);
        if (!ncalls) { fprintf(stderr, "session has no prompts\n"); return 1; }
    }
    char tok_path[1300]; snprintf(tok_path, sizeof tok_path, "%s/tokenizer.json", qwen);
    char tmp[1300]; snprintf(tmp, sizeof tmp, "%s/tmp", out); mkdirs(tmp);
    txt_cfg tcfg = { qwen, qbin, cache, tmp, tokjson ? tokjson : tok_path, qspawn };

    /* ---- frames (loaded once; every call re-tiles them as a real stream would) ---- */
    int T_raw = 21 + 24 * (K - 2), F = 6;
    npy_t ne, nw;
    if (npy_load(ext_p, &ne) || npy_load(wr_p, &nw) || strcmp(ne.dt, "|u1") || strcmp(nw.dt, "|u1")) { fprintf(stderr, "frames must be uint8 [T][H][W][3] .npy\n"); return 1; }
    if (ne.nd != 4 || ne.shape[0] < T_raw || nw.shape[0] < T_raw) { fprintf(stderr, "need >= %d frames per view for K'=%d (got %lld / %lld)\n", T_raw, K, ne.shape[0], nw.shape[0]); return 1; }
    int CW = 2 * tw, H = th, W = CW, C = 192;
    int h = H / 32, w = W / 32, tpf = h * w;

    int resident = session != NULL;   /* session: policy + VAE weights stay loaded between calls */
    cuda_wan22_vae *vae = NULL; praxis_runner *r = NULL;
    float *nz_v = NULL, *nz_a = NULL, *proprio = NULL, *vout = NULL, *aout = NULL; double cur_pose8[8] = { 0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.4 };
    int A = 0, P = 0; size_t stage_peak[4] = { 0, 0, 0, 0 }; double t_text_first = 0;
    call_t *ct = (call_t *)calloc(ncalls, sizeof(call_t));
    char callout[1300];
    for (int call = 0; call < ncalls; call++) {
        double TC = now();
        if (session) { snprintf(callout, sizeof callout, "%s/call_%03d", out, call); mkdirs(callout); printf("\n=== call %d/%d: \"%.70s\"%s ===\n", call + 1, ncalls, plist[call], strlen(plist[call]) > 70 ? "..." : ""); }
        else snprintf(callout, sizeof callout, "%s", out);
        /* ---- 1. text ---- */
        int S = 0, Su = 0, Ss[2];
        txt_src qs[2]; int nq = 1;
        if (session) qs[0].prompt = plist[call], qs[0].ids_path = NULL; else if (prompt) qs[0].prompt = prompt, qs[0].ids_path = NULL; else qs[0].prompt = NULL, qs[0].ids_path = ids_p;
        int have_u = 1;
        if (!session && !prompt && !ids_u_p) have_u = 0;
        if (have_u) { nq = 2; if (!session && !prompt) qs[1].prompt = NULL, qs[1].ids_path = ids_u_p; else qs[1].prompt = prompt_u, qs[1].ids_path = NULL; }
        float **tx = text_cond_multi(&tcfg, qs, nq, Ss);
        if (!tx) return 1;
        float *text = tx[0], *text_u = nq == 2 ? tx[1] : NULL; S = Ss[0]; Su = nq == 2 ? Ss[1] : 0;
        if (!text_u) { text_u = text; Su = S; }
        if (pair && Su != S) { fprintf(stderr, "[warn] --pair needs equal text lengths (cond %d vs uncond %d): TIMING ONLY, the unguided branch reuses the cond text\n", S, Su); text_u = text; Su = S; }
        { long long sh[2] = { S, 2048 }; save_f32(callout, "text", text, 2, sh); }
        double t_text = now() - TC; if (call == 0) t_text_first = t_text;
        stage_peak[0] = (size_t)(g_text_peak_gb * 1073741824.0);

        /* ---- 2. frames -> canvas -> VAE ---- */
        float *canvas = (float *)malloc((size_t)3 * T_raw * H * W * 4);
        double tt_tiles = now();
        {   void (*tv)(const unsigned char *, int, int, int, int, int, int, int, float *) = aa_force ? tile_view_fit : tile_view;
            tv((unsigned char *)ne.data + (size_t)(ne.shape[0] - T_raw) * ne.shape[1] * ne.shape[2] * 3, T_raw, (int)ne.shape[1], (int)ne.shape[2], th, tw, 0, CW, canvas);
            tv((unsigned char *)nw.data + (size_t)(nw.shape[0] - T_raw) * nw.shape[1] * nw.shape[2] * 3, T_raw, (int)nw.shape[1], (int)nw.shape[2], th, tw, tw, CW, canvas);
            tt_tiles = now() - tt_tiles;
            printf("[tiles] %d frames x 2 views %lldx%lld -> %dx%d canvas (%s): %.2f s (CPU)\n", T_raw, ne.shape[2], ne.shape[1], CW, th, aa_force ? "fit rule, antialias" : "aspect crop + bilinear", tt_tiles); }
        { unsigned char *u = (unsigned char *)malloc((size_t)3 * T_raw * H * W); for (size_t i = 0; i < (size_t)3 * T_raw * H * W; i++) u[i] = (unsigned char)canvas[i];
          long long sh[4] = { 3, T_raw, H, W }; char p[1300]; snprintf(p, sizeof p, "%s/canvas_chtw.npy", callout); npy_save(p, u, "|u1", 1, 4, sh); free(u); }
        for (size_t i = 0; i < (size_t)3 * T_raw * H * W; i++) canvas[i] = canvas[i] / 255.0f * 2.0f - 1.0f;   /* vae._to_unit */
        double tv0 = now();
        if (!vae) { vae = cuda_wan22_vae_load(vae_dir, 0, 0); if (!vae) return 1; }
        float *lat = NULL; int Tl = 0;
        if (cuda_wan22_vae_encode(vae, canvas, T_raw, H, W, &lat, &Tl)) return 1;
        double t_vae = now() - tv0;
        if (cuda_wan22_vae_peak_bytes(vae) > stage_peak[1]) stage_peak[1] = cuda_wan22_vae_peak_bytes(vae);   /* the runner's own allocations (robust to other GPU users) */
        printf("[vae] encode %d frames %dx%d -> policy latent [192,%d,%d,%d]: %.1f ms (%s %.2f s), device peak %.2f GB over the context\n", T_raw, H, W, Tl, h, w,
               cuda_wan22_vae_last_ms(vae), call == 0 ? "load+encode" : "total", t_vae, gb(stage_peak[1]));
        { long long sh[4] = { 192, Tl, h, w }; save_f32(callout, "latent", lat, 4, sh); }
        if (Tl != (K - 1) * F) { fprintf(stderr, "encoded %d latent frames, expected %d\n", Tl, (K - 1) * F); return 1; }
        if (!keep_vae && !resident) { cuda_wan22_vae_free(vae); vae = NULL; }
        printf("[vae] %s; device in use after: %.2f GB\n", vae ? "kept resident" : "freed", gb(dev_used()));
        free(canvas);
        float *anchor = (float *)malloc((size_t)C * tpf * 4);
        for (int c = 0; c < C; c++) memcpy(anchor + (size_t)c * tpf, lat + (size_t)c * Tl * tpf, (size_t)tpf * 4);

        /* ---- 3. policy ---- */
        double tp0 = now();
        if (!r) {
            r = synth ? praxis_load_synthetic(synth, 0, 0, 4) : praxis_load(policy, 0, 0);
            if (!r) return 1;
            if (praxis_set_stream(r, synth || praxis_nvfp4_enabled(r) ? ring : 0)) return 1;
            if (praxis_video_c(r) != C || praxis_text_dim(r) != 2048) { fprintf(stderr, "policy dims (video %d, text %d) do not match the VAE/Qwen outputs (192, 2048)\n", praxis_video_c(r), praxis_text_dim(r)); return 1; }
            A = praxis_action_dim(r); P = praxis_proprio_dim(r);
            printf("[policy] loaded in %.1f s\n", now() - tp0);
            size_t nv = (size_t)C * K * F * tpf, na = (size_t)K * F * A;
            nz_v = (float *)malloc(nv * 4); nz_a = (float *)malloc(na * 4); proprio = (float *)calloc((size_t)K * P, 4);
            if (noise_v_p) { npy_t t; float *d = npy_f32(noise_v_p, &t); if (!d || t.n != nv) { fprintf(stderr, "noise-v size mismatch\n"); return 1; } memcpy(nz_v, d, nv * 4); }
            else { rs_ ^= seed * 0x9e3779b97f4a7c15ULL; for (size_t i = 0; i < nv; i++) nz_v[i] = rnd_n(); }
            if (noise_a_p) { npy_t t; float *d = npy_f32(noise_a_p, &t); if (!d || t.n != na) { fprintf(stderr, "noise-a size mismatch\n"); return 1; } memcpy(nz_a, d, na * 4); }
            else for (size_t i = 0; i < na; i++) nz_a[i] = rnd_n();
            if (pose8_p && have_rc && rc.ok) {
                npy_t t; float *d = npy_f32(pose8_p, &t); if (!d || t.n != (size_t)K * 8) { fprintf(stderr, "pose8 must be [K'=%d][8] float32 (xyz, quat wxyz, grip)\n", K); return 1; }
                for (int k = 0; k < K; k++) { double p8[8]; for (int i = 0; i < 8; i++) p8[i] = d[k * 8 + i]; proprio_from_pose8(&rc, p8, proprio + (size_t)k * P); }
                for (int i = 0; i < 8; i++) cur_pose8[i] = d[(size_t)(K - 1) * 8 + i];
            } else if (prop_p) { npy_t t; float *d = npy_f32(prop_p, &t); if (!d || t.n != (size_t)K * P) { fprintf(stderr, "proprio must be [K'=%d][%d] float32\n", K, P); return 1; } memcpy(proprio, d, (size_t)K * P * 4); }
            vout = (float *)malloc((size_t)C * F * tpf * 4); aout = (float *)malloc((size_t)F * A * 4);
        }
        ct[call].pol_load = now() - tp0;
        double ts0 = now();
        double t_first = 0, tmin = 1e30; double tl[64]; int ntl = 0;
        for (int it = 0; it < repeat; it++) {
            double tt = now();
            if (praxis_sample_window_joint_ex2(r, text, S, text_u, Su, anchor, h, w, lat, proprio, nz_v, nz_a, K, steps, t_shift, cfg_v, cfg_a, vout, aout, pair, skipv)) return 1;
            cuCtxSynchronize();
            tt = now() - tt; if (it == 0) t_first = tt; if (ntl < 64) tl[ntl++] = tt; if (tt < tmin) tmin = tt;
        }
        double t_samp = now() - ts0;
        if (repeat > 1) { double v[64]; int n = ntl - 1; for (int i = 0; i < n; i++) v[i] = tl[i + 1];
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (v[j] < v[i]) { double x = v[i]; v[i] = v[j]; v[j] = x; }
            printf("[bench] first call %.2f s; warm calls n=%d: median %.2f s, min %.2f s\n", t_first, n, n ? v[n / 2] : 0.0, tmin); t_samp = n ? v[n / 2] : t_first; }
        praxis_stats st; praxis_stats_get(r, &st);
        if (st.dev_bytes_peak > stage_peak[2]) stage_peak[2] = st.dev_bytes_peak;   /* own accounting; cuMemGetInfo peak is printed below (includes other processes on a shared GPU) */
        printf("[policy] window_joint K'=%d %d steps cfg (%.1f, %.1f)%s%s%s: %.2f s; own peak %.2f GB, cuMemGetInfo peak use %.2f GB (includes the VAE if kept)\n", K, steps, cfg_v, cfg_a,
               pair ? " paired" : "", skipv ? " skip-last-video" : "", ctx ? " ctx-cache" : "", t_samp, gb(st.dev_bytes_peak), gb(st.free_min_bytes_used));
        double tpost = now();
        { long long sh[4] = { C, F, h, w }; save_f32(callout, "pred_video", vout, 4, sh); }
        { long long sh[2] = { F, A }; save_f32(callout, "pred_action", aout, 2, sh); }
        { /* unpack the tokens (F x 4*10) to 24 rows x 10: action_rows at depth >= 1 (normalised, not clipped, not un-normalised) */
            int per = A / 4, rows = F * 4; float *ar = (float *)malloc((size_t)rows * per * 4);
            for (int k = 0; k < F; k++) for (int f = 0; f < 4; f++) memcpy(ar + ((size_t)k * 4 + f) * per, aout + (size_t)k * A + f * per, (size_t)per * 4);
            long long sh[2] = { rows, per }; save_f32(callout, "action_rows", ar, 2, sh);
            printf("[policy] action rows (normalised) [%d][%d], row0:", rows, per); for (int i = 0; i < per; i++) printf(" %.4f", ar[i]); printf("\n");
            if (have_rc && rc.ok && per == 10) {
                float *rel = (float *)malloc((size_t)rows * 10 * 4), *ab = (float *)malloc((size_t)rows * 13 * 4);
                unnormalise_rows(&rc, ar, rows, rel); rel_to_abs(rel, rows, cur_pose8, ab);
                long long sh2[2] = { rows, 10 }, sh3[2] = { rows, 13 }; save_f32(callout, "action_rel", rel, 2, sh2); save_f32(callout, "action_abs_flange", ab, 2, sh3);
                printf("[policy] action rows un-normalised (relative, q01/q99 on dims 0,1,2,9): row0:"); for (int i = 0; i < 10; i++) printf(" %.4f", rel[i]);
                printf("\n[policy] row%d:", rows - 1); for (int i = 0; i < 10; i++) printf(" %.4f", rel[(rows - 1) * 10 + i]); printf("\n");
                free(rel); free(ab);
            }
            free(ar); }
        tpost = now() - tpost;
        if (!resident) {
            praxis_free(r); r = NULL;
            printf("[policy] freed; device in use after: %.2f GB\n", gb(dev_used()));
        }

        /* ---- 4. optional decode of [clean chunks | predicted chunk] ---- */
        if (decode && !skipv) {
            if (!vae) { vae = cuda_wan22_vae_load(vae_dir, 0, 0); if (!vae) return 1; }
            int Tt = Tl + F; float *full = (float *)malloc((size_t)C * Tt * tpf * 4);
            for (int c = 0; c < C; c++) { memcpy(full + (size_t)c * Tt * tpf, lat + (size_t)c * Tl * tpf, (size_t)Tl * tpf * 4);
                                          memcpy(full + ((size_t)c * Tt + Tl) * tpf, vout + (size_t)c * F * tpf, (size_t)F * tpf * 4); }
            float *fr = NULL; int To = 0; double td0 = now();
            if (cuda_wan22_vae_decode(vae, full, Tt, h, w, &fr, &To)) return 1;
            stage_peak[3] = cuda_wan22_vae_peak_bytes(vae);
            int keep = To < 24 ? To : 24; float *last = (float *)malloc((size_t)3 * keep * H * W * 4);
            for (int c = 0; c < 3; c++) memcpy(last + (size_t)c * keep * H * W, fr + ((size_t)c * To + (To - keep)) * H * W, (size_t)keep * H * W * 4);
            long long sh[4] = { 3, keep, H, W }; save_f32(callout, "decoded", last, 4, sh);
            printf("[vae] decode [%d latent frames] -> %d frames, last %d kept: %.1f ms (load+decode %.2f s), device peak %.2f GB over the context\n", Tt, To, keep, cuda_wan22_vae_last_ms(vae), now() - td0, gb(stage_peak[3]));
            free(full); free(last); cuda_wan22_vae_free(vae); vae = NULL;   /* (frees the weights too; decode is an offline option, not the latency path) */
        }
        free(lat); free(anchor);
        ct[call] = (call_t){ t_text, tt_tiles, t_vae, ct[call].pol_load, t_samp, tpost, now() - TC };
        if (session) printf("[call %d] %.2f s total: text %.3f (qwen runs so far %d) | tiles %.3f | VAE enc %.3f | policy load %.2f | sampler %.2f | actions+io %.3f\n", call + 1, ct[call].total, t_text, g_qwen_runs, tt_tiles, t_vae, ct[call].pol_load, t_samp, tpost);
    }
    if (r) praxis_free(r);
    if (vae) cuda_wan22_vae_free(vae);
    size_t mx = 0; for (int i = 0; i < 4; i++) if (stage_peak[i] > mx) mx = stage_peak[i];
    printf("== device memory per stage (over the %.2f GB idle context): qwen %.2f | vae encode %.2f | policy %.2f (own accounting) | vae decode %.2f ; max instant %.2f GB (+ context %.2f = %.2f GB)\n",
           gb(idle), gb(stage_peak[0]), gb(stage_peak[1]), gb(stage_peak[2]), gb(stage_peak[3]), gb(mx), gb(idle), gb(mx + idle));
    if (!qspawn) printf("== qwen in-process: tokenize %.3f s total, load %.2f s, prefill %.0f ms (%d runs), free %.0f ms; device memory after Qwen free vs before load: %+.3f GB\n", g_t_tok, g_t_qload, g_t_qrun * 1e3, g_qwen_runs, g_t_qfree * 1e3, g_qwen_leak_gb);
    if (session && ncalls > 1) {
        double a[256], b[256], c2[256], d[256], e[256], f2[256], g[256]; int n = ncalls - 1 > 256 ? 256 : ncalls - 1;
        for (int i = 0; i < n; i++) { a[i] = ct[i + 1].text; b[i] = ct[i + 1].tiles; c2[i] = ct[i + 1].vae; d[i] = ct[i + 1].samp; e[i] = ct[i + 1].post; f2[i] = ct[i + 1].total; g[i] = ct[i + 1].pol_load; }
        printf("== session (%d calls): cold call 1: %.2f s (text %.2f, tiles %.2f, VAE %.2f [incl. VAE load], policy load %.2f, sampler %.2f, post %.3f)\n", ncalls, ct[0].total, ct[0].text, ct[0].tiles, ct[0].vae, ct[0].pol_load, ct[0].samp, ct[0].post);
        printf("== session warm calls 2..%d median: %.2f s (text %.3f, tiles %.3f, VAE %.3f, policy load %.3f, sampler %.2f, post %.3f)\n", ncalls, median(f2, n), median(a, n), median(b, n), median(c2, n), median(g, n), median(d, n), median(e, n));
    }
    printf("== wall: first-call text %.1f s, total %.1f s\n", t_text_first, now() - T0);
    return 0;
}
