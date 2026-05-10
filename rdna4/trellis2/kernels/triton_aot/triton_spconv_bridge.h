/* Triton spconv bridge: load AOT-extracted Triton hsacos at startup and
 * dispatch tex_dec spconv calls to them.
 *
 * Single-header, include into one .c file with TRITON_SPCONV_BRIDGE_IMPL
 * defined to emit the implementation.
 *
 * API:
 *   int  t2_triton_init(const char *kernels_dir);
 *        -- scans kernels_dir/N..._SPLITK.../kernel.hsaco + kernel.json,
 *          loads modules, populates the shape table.
 *
 *   int  t2_triton_spconv(int N, int Ci, int Co,
 *                         const void *d_input,  // f16, [N, Ci]
 *                         const void *d_weight, // f16, [Co, V=27, Ci]
 *                         const void *d_bias,   // f16, [Co]
 *                         const void *d_nmap,   // u32, [N, V]; -1 = empty
 *                         void       *d_output, // f16, [N, Co]
 *                         hipStream_t stream);
 *        -- returns 0 on launch success; -1 if no matching shape registered
 *          (caller falls back to legacy path).
 *          SPLITK>1 shapes return -1 for now (post-reduction not yet wired).
 *
 *   void t2_triton_release(void);
 */
#ifndef TRITON_SPCONV_BRIDGE_H
#define TRITON_SPCONV_BRIDGE_H

#ifdef TRITON_BRIDGE_USE_ROCEW
#include "../../rocew.h"
#else
#include <hip/hip_runtime.h>
#endif

int  t2_triton_init(const char *kernels_dir);
int  t2_triton_has_shape(int N, int Ci, int Co); /* 1 if registered, 0 otherwise */
/* Optional: set a directory used to persist (sorted, vk, vkseg) prep tensors
 * across processes. When set, _ts_prep_cache reads from
 *   <dir>/triton_prep_N{N}_B{B1}.bin
 * if present (skipping D2H + argsort + H2D), and writes the file on miss.
 * Pass NULL to disable. The caller owns dir lifetime. */
void t2_triton_set_prep_cache_dir(const char *dir);
/* Register a device-side reduction kernel for SPLITK>1: out_f16[total] =
 * sum_k partial_f32[k*total + i] -> f16. Bridge launches it after the
 * SPLITK kernel. Must be set before the first SPLITK call. */
void t2_triton_set_reduce_kernel(hipFunction_t kfn);
int  t2_triton_spconv(int N, int Ci, int Co,
                      const void *d_input, const void *d_weight, const void *d_bias,
                      const void *d_nmap, void *d_output, hipStream_t stream);
void t2_triton_release(void);

#ifdef TRITON_SPCONV_BRIDGE_IMPL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>
#include <dirent.h>
#include "triton_spconv_pp.h"

/* hipGetErrorString shim: rocew exposes the CUDA-driver-style
 * `(err, char**) -> err` form, while the runtime libamdhip64 provides the
 * `(err) -> const char*` form. Use a local helper to bridge both. */
static const char *_ts_err_str(hipError_t e) {
#ifdef TRITON_BRIDGE_USE_ROCEW
    static char buf[32];
    snprintf(buf, sizeof buf, "hipError=%d", (int)e);
    return buf;
#else
    return hipGetErrorString(e);
#endif
}

#define T2_V 27

typedef struct {
    int N, Ci, Co;
    int B1, B2, BK;
    int num_warps;
    int shared;
    int SPLITK;
    hipModule_t mod;
    hipFunction_t kfn;
    /* Cache for derived sorted_idx/vk/vkseg, keyed by nmap pointer addr.
     * Only one entry per shape -- assumes nmap is stable per shape per run. */
    const void *cached_nmap;
    void *d_sorted, *d_vk, *d_vkseg;
    int   vk_len, num_blocks;
    /* SPLITK>1: persistent f32 partial output buffer [SPLITK, N, Co]. */
    void *d_partial_f32;
} t2_tspconv_shape;

static hipFunction_t g_reduce_kfn = NULL;
void t2_triton_set_reduce_kernel(hipFunction_t kfn) { g_reduce_kfn = kfn; }

#define T2_TS_MAX_SHAPES 16
static t2_tspconv_shape g_shapes[T2_TS_MAX_SHAPES];
static int g_n_shapes = 0;

/* ---- minimal JSON helpers (parse the tiny key:value pairs we need) ---- */
static int _ts_json_int(const char *json, const char *key)
{
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(json, pat); if (!p) return -1;
    p += strlen(pat); while (*p == ' ') p++;
    return atoi(p);
}

static int _ts_load_shape(const char *kernels_dir, const char *tag, t2_tspconv_shape *out)
{
    /* tag: e.g. "N8452_Ci512_Co512_SPLITK1" */
    if (sscanf(tag, "N%d_Ci%d_Co%d_SPLITK%d", &out->N, &out->Ci, &out->Co, &out->SPLITK) != 4) {
        fprintf(stderr, "  skip bad tag: %s\n", tag); return -1;
    }
    /* SPLITK>1 kernels are loaded; partial-f32 buffer + reduction kernel
     * are wired at first dispatch. */
    /* hsaco */
    char p[1024];
    snprintf(p, sizeof p, "%s/%s/kernel.hsaco", kernels_dir, tag);
    FILE *f = fopen(p, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz); fread(buf, 1, sz, f); fclose(f);
    hipError_t e = hipModuleLoadData(&out->mod, buf); free(buf);
    if (e != hipSuccess) { fprintf(stderr, "  hipModuleLoadData fail %s: %s\n", tag, _ts_err_str(e)); return -1; }

    /* json (kernel name + num_warps + shared) */
    snprintf(p, sizeof p, "%s/%s/kernel.json", kernels_dir, tag);
    f = fopen(p, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long jsz = ftell(f); fseek(f, 0, SEEK_SET);
    char *js = malloc(jsz + 1); fread(js, 1, jsz, f); js[jsz] = 0; fclose(f);
    out->num_warps = _ts_json_int(js, "num_warps");
    out->shared    = _ts_json_int(js, "shared");
    /* Find kernel name. */
    const char *np = strstr(js, "\"name\":"); np = strchr(np, '"'); np = strchr(np+1, '"'); /* opening quote of value */
    np = strchr(np+1, '"');
    char kname[256]; const char *ks = strchr(strstr(js, "\"name\":"), ':'); ks = strchr(ks, '"') + 1;
    const char *ke = strchr(ks, '"');
    int klen = ke - ks; if (klen >= (int)sizeof kname) klen = sizeof kname - 1;
    memcpy(kname, ks, klen); kname[klen] = 0;
    free(js);
    e = hipModuleGetFunction(&out->kfn, out->mod, kname);
    if (e != hipSuccess) { fprintf(stderr, "  hipModuleGetFunction(%s) fail: %s\n", kname, _ts_err_str(e)); return -1; }

    /* Tile params: derived from autotune table baked at extract-time.
     * For our 9 shapes there are 3 distinct (B1,B2,BK,nw) configs -- encode
     * heuristic by (Ci, Co) thresholds. */
    if (out->SPLITK == 4 && out->N == 1905) { out->B1=64;  out->B2=128; out->BK=32; }
    else if (out->N == 1905 && out->Co == 4096) { out->B1=128; out->B2=128; out->BK=32; }
    else if (out->Ci == 64 && out->Co == 64) { out->B1=64; out->B2=64; out->BK=32; }
    else { out->B1=64; out->B2=128; out->BK=32; }

    out->cached_nmap = NULL;
    out->d_sorted = out->d_vk = out->d_vkseg = NULL;
    out->d_partial_f32 = NULL;
    return 0;
}

int t2_triton_init(const char *kernels_dir)
{
    DIR *d = opendir(kernels_dir);
    if (!d) { fprintf(stderr, "t2_triton_init: open %s failed\n", kernels_dir); return -1; }
    struct dirent *de;
    while ((de = readdir(d)) && g_n_shapes < T2_TS_MAX_SHAPES) {
        if (de->d_name[0] != 'N') continue;
        if (_ts_load_shape(kernels_dir, de->d_name, &g_shapes[g_n_shapes]) == 0) {
            t2_tspconv_shape *s = &g_shapes[g_n_shapes];
            fprintf(stderr, "t2_triton: registered N=%d Ci=%d Co=%d B1=%d B2=%d BK=%d nw=%d shared=%d\n",
                    s->N, s->Ci, s->Co, s->B1, s->B2, s->BK, s->num_warps, s->shared);
            g_n_shapes++;
        }
    }
    closedir(d);
    fprintf(stderr, "t2_triton: %d shapes registered\n", g_n_shapes);
    return g_n_shapes > 0 ? 0 : -1;
}

/* Matching is by (Ci, Co) only — kernel tile heuristic depends on Ci/Co, the
 * kernel itself takes runtime N. The N field in the manifest is the extraction
 * sample size; runtime N can differ (e.g. 822874 in manifest vs 822875 at run
 * time, off-by-one between the AOT-extraction graph and the live graph). */
int t2_triton_has_shape(int N, int Ci, int Co)
{
    (void)N;
    for (int i = 0; i < g_n_shapes; i++) {
        if (g_shapes[i].Ci == Ci && g_shapes[i].Co == Co) return 1;
    }
    return 0;
}

static t2_tspconv_shape *_ts_find(int N, int Ci, int Co)
{
    (void)N;
    for (int i = 0; i < g_n_shapes; i++) {
        if (g_shapes[i].Ci == Ci && g_shapes[i].Co == Co) return &g_shapes[i];
    }
    return NULL;
}

/* Disk-cache for (sorted, vk, vkseg). These are a pure function of nmap
 * content and B1 — see _ts_prep_cache. Format (little-endian, host-native):
 *   magic[8] = "T2PRP01\0"
 *   int32: N, B1, vk_len, num_blocks
 *   int64[N]: sorted
 *   int32[vk_len]: vk
 *   int32[num_blocks+1]: vkseg
 * Caller invalidates by clearing the cache_dir (same lifetime as the .npy
 * nmap cache that drives derivation). */
static const char *g_prep_cache_dir = NULL;
void t2_triton_set_prep_cache_dir(const char *dir) { g_prep_cache_dir = dir; }

static int _ts_prep_disk_load(t2_tspconv_shape *s, hipStream_t stream)
{
    if (!g_prep_cache_dir) return 0;
    char p[1024];
    snprintf(p, sizeof p, "%s/triton_prep_N%d_B%d.bin", g_prep_cache_dir, s->N, s->B1);
    FILE *f = fopen(p, "rb"); if (!f) return 0;
    char magic[8]; int32_t hdr[4];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "T2PRP01\0", 8) != 0) { fclose(f); return 0; }
    if (fread(hdr, 4, 4, f) != 4) { fclose(f); return 0; }
    int N = hdr[0], B1 = hdr[1], vk_len = hdr[2], num_blocks = hdr[3];
    if (N != s->N || B1 != s->B1) { fclose(f); return 0; }
    size_t N_pad = (size_t)((N + B1 - 1) / B1) * B1;
    size_t vk_pad = (size_t)vk_len + B1;
    int64_t *sorted = (int64_t *)malloc((size_t)N * 8);
    int32_t *vk     = (int32_t *)malloc((size_t)vk_len * 4);
    int32_t *seg    = (int32_t *)malloc((size_t)(num_blocks + 1) * 4);
    int ok = (fread(sorted, 8, N, f) == (size_t)N
           && fread(vk,     4, vk_len, f) == (size_t)vk_len
           && fread(seg,    4, num_blocks + 1, f) == (size_t)(num_blocks + 1));
    fclose(f);
    if (!ok) { free(sorted); free(vk); free(seg); return 0; }
    if (s->d_sorted) hipFree(s->d_sorted);
    if (s->d_vk)     hipFree(s->d_vk);
    if (s->d_vkseg)  hipFree(s->d_vkseg);
    hipMalloc(&s->d_sorted, N_pad * 8); hipMemsetAsync(s->d_sorted, 0, N_pad * 8, stream);
    hipMalloc(&s->d_vk,     vk_pad * 4); hipMemsetAsync(s->d_vk, 0, vk_pad * 4, stream);
    hipMalloc(&s->d_vkseg,  (size_t)(num_blocks + 1) * 4);
    hipMemcpyAsync(s->d_sorted, sorted, (size_t)N * 8, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(s->d_vk,     vk,     (size_t)vk_len * 4, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(s->d_vkseg,  seg,    (size_t)(num_blocks + 1) * 4, hipMemcpyHostToDevice, stream);
    /* Synchronize so the host-side `sorted/vk/seg` buffers can be freed safely. */
    hipStreamSynchronize(stream);
    s->vk_len = vk_len; s->num_blocks = num_blocks;
    free(sorted); free(vk); free(seg);
    return 1;
}

static void _ts_prep_disk_store(t2_tspconv_shape *s,
                                const int64_t *sorted, const int32_t *vk, const int32_t *seg)
{
    if (!g_prep_cache_dir) return;
    char p[1024];
    snprintf(p, sizeof p, "%s/triton_prep_N%d_B%d.bin.tmp", g_prep_cache_dir, s->N, s->B1);
    FILE *f = fopen(p, "wb"); if (!f) return;
    int32_t hdr[4] = { s->N, s->B1, s->vk_len, s->num_blocks };
    int ok = (fwrite("T2PRP01\0", 1, 8, f) == 8
           && fwrite(hdr, 4, 4, f) == 4
           && fwrite(sorted, 8, s->N, f) == (size_t)s->N
           && fwrite(vk,     4, s->vk_len, f) == (size_t)s->vk_len
           && fwrite(seg,    4, s->num_blocks + 1, f) == (size_t)(s->num_blocks + 1));
    fclose(f);
    if (ok) {
        char p2[1024];
        snprintf(p2, sizeof p2, "%s/triton_prep_N%d_B%d.bin", g_prep_cache_dir, s->N, s->B1);
        rename(p, p2);
    } else {
        remove(p);
    }
}

static void _ts_prep_cache(t2_tspconv_shape *s, const void *d_nmap, hipStream_t stream)
{
    if (s->cached_nmap == d_nmap && s->d_sorted) return;
    if (_ts_prep_disk_load(s, stream)) { s->cached_nmap = d_nmap; return; }
    /* Pull nmap host-side, derive, push back. */
    int N = s->N;
    int32_t *h_nmap = (int32_t *)malloc((size_t)N * T2_V * 4);
    hipMemcpyAsync(h_nmap, d_nmap, (size_t)N * T2_V * 4, hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    uint32_t *gray = (uint32_t *)malloc(N * 4);
    uint32_t *bin  = (uint32_t *)malloc(N * 4);
    int64_t *sorted = (int64_t *)malloc(N * 8);
    t2_neigh_to_gray_binary(N, T2_V, h_nmap, gray, bin);
    t2_argsort_binary(N, bin, sorted);
    int32_t *vk = NULL, *seg = NULL; int vk_len = 0;
    t2_build_valid_kernel(N, s->B1, gray, sorted, &vk, &seg, &vk_len);
    int num_blocks = (N + s->B1 - 1) / s->B1;
    if (s->d_sorted) hipFree(s->d_sorted);
    if (s->d_vk)     hipFree(s->d_vk);
    if (s->d_vkseg)  hipFree(s->d_vkseg);
    /* Triton kernels use masked tl.load but addresses are still computed up to
     * (block + B1 - 1). With small N (e.g. 1905) the trailing OOB read can
     * cross page boundary -> memory fault. Pad allocs to next B1 multiple. */
    size_t N_pad = (size_t)((N + s->B1 - 1) / s->B1) * s->B1;
    hipMalloc(&s->d_sorted, N_pad * 8);
    hipMemsetAsync(s->d_sorted, 0, N_pad * 8, stream);
    /* vk_len is bounded by num_blocks * V; pad by B1 entries for the same reason. */
    size_t vk_pad = (size_t)vk_len + s->B1;
    hipMalloc(&s->d_vk, vk_pad * 4);
    hipMemsetAsync(s->d_vk, 0, vk_pad * 4, stream);
    hipMalloc(&s->d_vkseg,  (size_t)(num_blocks + 1) * 4);
    hipMemcpyAsync(s->d_sorted, sorted, (size_t)N * 8, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(s->d_vk,     vk,     (size_t)vk_len * 4, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(s->d_vkseg,  seg,    (size_t)(num_blocks + 1) * 4, hipMemcpyHostToDevice, stream);
    hipStreamSynchronize(stream);
    s->vk_len = vk_len;
    s->num_blocks = num_blocks;
    s->cached_nmap = d_nmap;
    _ts_prep_disk_store(s, sorted, vk, seg);
    free(h_nmap); free(gray); free(bin); free(sorted); free(vk); free(seg);
}

int t2_triton_spconv(int N, int Ci, int Co,
                     const void *d_input, const void *d_weight, const void *d_bias,
                     const void *d_nmap, void *d_output, hipStream_t stream)
{
    t2_tspconv_shape *s = _ts_find(N, Ci, Co);
    if (!s) return -1;
    if (s->SPLITK > 1 && !g_reduce_kfn) return -1;  /* reduction kfn not registered */
    /* Adopt runtime N so prep-cache + partial buffer sizing are correct even
     * when the AOT manifest's sample N differs (e.g. 822874 vs 822875). */
    if (s->N != N) {
        if (s->d_sorted) { hipFree(s->d_sorted); s->d_sorted = NULL; }
        if (s->d_vk)     { hipFree(s->d_vk);     s->d_vk     = NULL; }
        if (s->d_vkseg)  { hipFree(s->d_vkseg);  s->d_vkseg  = NULL; }
        if (s->d_partial_f32) { hipFree(s->d_partial_f32); s->d_partial_f32 = NULL; }
        s->cached_nmap = NULL;
        s->N = N;
    }
    _ts_prep_cache(s, d_nmap, stream);
    int LOGN = (int)log2((double)N);
    void *null_ptr = NULL;
    int total = N * Co;

    /* Pick output ptr passed to the spconv kernel: f32 partial for SPLITK>1, else f16 final. */
    void *kern_out = d_output;
    if (s->SPLITK > 1) {
        size_t need = (size_t)s->SPLITK * total * sizeof(float);
        if (!s->d_partial_f32) {
            if (hipMalloc(&s->d_partial_f32, need) != hipSuccess) return -1;
        }
        kern_out = s->d_partial_f32;
    }

    void *args[] = {
        (void *)&d_input, (void *)&d_weight, (void *)&d_bias,
        (void *)&d_nmap,  (void *)&s->d_sorted, (void *)&kern_out,
        &N, &LOGN, &Ci, &Co,
        (void *)&s->d_vk, (void *)&s->d_vkseg,
        &null_ptr, &null_ptr,
    };
    int grid_x = ((Co + s->B2 - 1) / s->B2) * ((N + s->B1 - 1) / s->B1);
    int grid_y = (s->SPLITK > 1) ? s->SPLITK : 1;
    int block_x = s->num_warps * 32;
    hipError_t e = hipModuleLaunchKernel(s->kfn, grid_x, grid_y, 1, block_x, 1, 1,
                                         s->shared, stream, args, NULL);
    if (e != hipSuccess) {
        fprintf(stderr, "t2_triton_spconv launch fail (N=%d Ci=%d Co=%d): %s\n",
                N, Ci, Co, _ts_err_str(e));
        return -1;
    }
    if (s->SPLITK > 1) {
        int splitk = s->SPLITK;
        void *rargs[] = { &d_output, &s->d_partial_f32, &splitk, &total };
        e = hipModuleLaunchKernel(g_reduce_kfn, (total + 255)/256, 1, 1, 256, 1, 1,
                                  0, stream, rargs, NULL);
        if (e != hipSuccess) {
            fprintf(stderr, "t2_triton_spconv reduce fail: %s\n", _ts_err_str(e));
            return -1;
        }
    }
    return 0;
}

void t2_triton_release(void)
{
    for (int i = 0; i < g_n_shapes; i++) {
        t2_tspconv_shape *s = &g_shapes[i];
        if (s->d_sorted) hipFree(s->d_sorted);
        if (s->d_vk)     hipFree(s->d_vk);
        if (s->d_vkseg)  hipFree(s->d_vkseg);
        if (s->d_partial_f32) hipFree(s->d_partial_f32);
        if (s->mod) hipModuleUnload(s->mod);
    }
    g_n_shapes = 0;
}

#endif /* TRITON_SPCONV_BRIDGE_IMPL */
#endif /* TRITON_SPCONV_BRIDGE_H */
