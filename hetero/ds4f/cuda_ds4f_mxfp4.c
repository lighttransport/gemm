#define _GNU_SOURCE
#include "cuda_ds4f_mxfp4.h"
#include "../../cuda/cuew.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct { unsigned int x, y, z; } ds4f_u3;
/* ~1000 slots = ~12 GB of expert tensors (the whole owned bank is ~16.5 GB;
 * a resident-weight server keeps what fits and streams the rest). */
#define DS4F_CUDA_MXFP4_CACHE_SLOTS 32768
typedef struct {
    const void *wkey, *skey;
    CUdeviceptr d;
    size_t bytes;
    unsigned long long age;
    int rows, cols, valid;
} cuda_mxfp4_cache_entry;
typedef struct {
    CUdeviceptr x, q8, y, y2, tmpfix;
    float *hx, *hy, *hres;
    size_t xb, q8b, yb, hxb, hresb, hyb2, fixb;
} cuda_batch_buf;
struct cuda_ds4f_mxfp4 {
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction quant, quant_fp4, quant_rows, gemm, fixup, gemm64, fixup64, add, residual;
    CUstream stream; CUevent quant_done;
    CUdeviceptr w, x, q8, y, y2, tmpfix, ids; size_t xb, q8b, yb, y2b, fixb, wb, idsb;
    void *hx, *hy, *hw, *hres; size_t hxb, hyb, hwb, hresb;
    cuda_mxfp4_cache_entry cache[DS4F_CUDA_MXFP4_CACHE_SLOTS];
    size_t cache_bytes, cache_limit;
    unsigned long long cache_age;
    unsigned long long cache_hits, cache_misses, cache_evictions;
    /* Contiguous weight pool: one big cuMemAlloc (the ~2363 separate 4.46 MB
     * allocations fragmented the heap and topped out at ~10.8 GB; a single
     * allocation reaches ~13 GB on this driver).  Sub-sliced by bump + a free
     * list of evicted slots, so the cache can hold ~30 resident layers. */
    CUdeviceptr cache_pool;
    size_t cache_pool_size, cache_pool_used;
    struct { size_t off, size; } pool_free[128];
    int n_pool_free;
    int active_cache_slot;
    int cache_no_evict;
    int rows, cols, nsm; int verbose; int terms;
    cuda_batch_buf *b;
    int n_batch, cap_batch;
};
static ds4f_u3 fastdiv(unsigned long long d) {
    unsigned int L = 0, di = (unsigned int)d;
    while (L < 32 && ((unsigned int)1 << L) < di) ++L;
    unsigned int mp = (unsigned int)(((unsigned long long)1 << 32) *
        (((unsigned long long)1 << L) - di) / di + 1);
    ds4f_u3 r = { mp, L, di }; return r;
}
static int ck(CUresult r, const char *what) {
    if (r == CUDA_SUCCESS) return 0;
    const char *s = NULL; if (cuGetErrorString) cuGetErrorString(r, &s);
    fprintf(stderr, "cuda_ds4f_mxfp4: %s: %s (%d)\n", what, s ? s : "error", (int)r);
    return -1;
}
cuda_ds4f_mxfp4 *cuda_ds4f_mxfp4_create_ex(int device_id, int verbose,
                                            int cache_mb) {
    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS || cuInit(0) != CUDA_SUCCESS) return NULL;
    cuda_ds4f_mxfp4 *c = (cuda_ds4f_mxfp4 *)calloc(1, sizeof(*c)); if (!c) return NULL;
    c->verbose = verbose;
    c->cache_limit = (size_t)(cache_mb > 0 ? cache_mb : 12000) * 1024 * 1024;
    {   const char *e = getenv("DS4F_CUDA_MXFP4_CACHE_MB");
        if (e && *e) {
            long mb = atol(e);
            if (mb > 0) c->cache_limit = (size_t)mb * 1024 * 1024;
        }
    }
    c->active_cache_slot = -1;
    if (ck(cuDeviceGet(&c->dev, device_id), "device") ||
        ck(cuDevicePrimaryCtxRetain(&c->ctx, c->dev), "context") ||
        ck(cuCtxSetCurrent(c->ctx), "set context") ||
        ck(cuStreamCreate(&c->stream, CU_STREAM_NON_BLOCKING), "stream") ||
        ck(cuEventCreate(&c->quant_done, CU_EVENT_DISABLE_TIMING), "event")) goto fail;
    FILE *f = fopen("cuda/llm/mmq_kernels.cubin", "rb");
    if (!f) f = fopen("../../cuda/llm/mmq_kernels.cubin", "rb");
    if (!f) f = fopen("mmq_kernels.cubin", "rb");
    if (!f) goto fail;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *blob = n > 0 ? malloc((size_t)n) : NULL;
    int ok = blob && fread(blob, 1, (size_t)n, f) == (size_t)n; fclose(f);
    if (!ok || ck(cuModuleLoadDataEx(&c->mod, blob, 0, NULL, NULL), "module")) { free(blob); goto fail; }
    free(blob);
    if (ck(cuModuleGetFunction(&c->quant, c->mod, "mmqv_quant_q8_1_d4"), "quantizer") ||
        ck(cuModuleGetFunction(&c->quant_fp4, c->mod, "mmqv_quant_mxfp4"), "mxfp4 quantizer") ||
        ck(cuModuleGetFunction(&c->quant_rows, c->mod, "mmqv_quant_mxfp4_rows"), "row quantizer") ||
        ck(cuModuleGetFunction(&c->gemm, c->mod, "mmqv_mxfp4_x128_nc0"), "mxfp4") ||
        ck(cuModuleGetFunction(&c->gemm64, c->mod, "mmqv_mxfp4_x64_nc0"), "mxfp4 x64") ||
        ck(cuModuleGetFunction(&c->add, c->mod, "mmqv_add_f32"), "f32 add")) goto fail;
    cuModuleGetFunction(&c->residual, c->mod, "mmqv_mxfp4_residual");
    cuModuleGetFunction(&c->fixup, c->mod, "mmqv_fixup_mxfp4_x128_nc0");
    cuModuleGetFunction(&c->fixup64, c->mod, "mmqv_fixup_mxfp4_x64_nc0");
    cuDeviceGetAttribute(&c->nsm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, c->dev);
    int sh = 0; cuDeviceGetAttribute(&sh, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, c->dev);
    if (sh < 65536) sh = 65536;
    cuFuncSetAttribute(c->gemm, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sh);
    cuFuncSetAttribute(c->gemm64, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sh);
    /* Keep an operator-selected reserve free for display/compositor and CUDA
     * work buffers.  The configured limit is an upper bound, never permission
     * to consume memory already in use by the desktop. */
    size_t free_bytes = 0, total_bytes = 0;
    if (cuMemGetInfo(&free_bytes, &total_bytes) != CUDA_SUCCESS) goto fail;
    const size_t reserve = 4096ull * 1024 * 1024;
    size_t usable = free_bytes > reserve ? free_bytes - reserve : 0;
    if (c->cache_limit > usable) c->cache_limit = usable;
    if (c->cache_limit < 256ull * 1024 * 1024) goto fail;
    /* One contiguous pool avoids the small-allocation fragmentation ceiling. */
    if (cuMemAlloc(&c->cache_pool, c->cache_limit) != CUDA_SUCCESS)
        c->cache_pool = 0;
    c->cache_pool_size = c->cache_pool ? c->cache_limit : 0;
    return c;
fail: cuda_ds4f_mxfp4_destroy(c); return NULL;
}

cuda_ds4f_mxfp4 *cuda_ds4f_mxfp4_create(int device_id, int verbose) {
    return cuda_ds4f_mxfp4_create_ex(device_id, verbose, 12000);
}
void cuda_ds4f_mxfp4_destroy(cuda_ds4f_mxfp4 *c) {
    if (!c) return; if (c->stream) cuStreamSynchronize(c->stream);
    if (c->cache_pool) {
        cuMemFree(c->cache_pool);
    } else {
        for (int i = 0; i < DS4F_CUDA_MXFP4_CACHE_SLOTS; ++i)
            if (c->cache[i].valid) cuMemFree(c->cache[i].d);
    }
    if (c->verbose && (c->cache_hits || c->cache_misses))
        fprintf(stderr, "CUDA MXFP4 cache: hits=%llu misses=%llu evictions=%llu resident=%.1f MB\n",
                c->cache_hits, c->cache_misses, c->cache_evictions,
                (double)c->cache_bytes / (1024.0 * 1024.0));
    if (c->active_cache_slot < 0 && c->w) cuMemFree(c->w);
    if (c->x) cuMemFree(c->x); if (c->q8) cuMemFree(c->q8); if (c->y) cuMemFree(c->y); if (c->y2) cuMemFree(c->y2); if (c->tmpfix) cuMemFree(c->tmpfix); if (c->ids) cuMemFree(c->ids);
    if (c->hx && cuMemFreeHost) cuMemFreeHost(c->hx);
    if (c->hy && cuMemFreeHost) cuMemFreeHost(c->hy);
    if (c->hw && cuMemFreeHost) cuMemFreeHost(c->hw);
    if (c->hres && cuMemFreeHost) cuMemFreeHost(c->hres);
    for (int i = 0; i < c->n_batch; ++i) {
        if (c->b[i].x) cuMemFree(c->b[i].x);
        if (c->b[i].q8) cuMemFree(c->b[i].q8);
        if (c->b[i].y) cuMemFree(c->b[i].y);
        if (c->b[i].y2) cuMemFree(c->b[i].y2);
        if (c->b[i].tmpfix) cuMemFree(c->b[i].tmpfix);
        if (c->b[i].hx && cuMemFreeHost) cuMemFreeHost(c->b[i].hx);
        if (c->b[i].hy && cuMemFreeHost) cuMemFreeHost(c->b[i].hy);
        if (c->b[i].hres && cuMemFreeHost) cuMemFreeHost(c->b[i].hres);
    }
    free(c->b);
    if (c->quant_done) cuEventDestroy(c->quant_done); if (c->stream) cuStreamDestroy(c->stream); if (c->mod) cuModuleUnload(c->mod);
    if (c->ctx) cuDevicePrimaryCtxRelease(c->dev); free(c);
}
static int cuda_ds4f_mxfp4_load_any(cuda_ds4f_mxfp4 *c, const uint8_t *w,
                                    const uint8_t *s, const uint8_t *packed,
                                    int rows, int cols) {
    if (!c || (!packed && (!w || !s)) || rows <= 0 || cols <= 0 || (cols & 127)) return -1;
    const void *wkey=packed?packed:w,*skey=packed?packed:s;
    if (cuCtxSetCurrent(c->ctx) != CUDA_SUCCESS) return -1;
    size_t nb = (size_t)cols / 32, bytes = (size_t)rows * nb * 17, rb = (size_t)cols / 2;
    if (c->active_cache_slot < 0 && c->w &&
        !(c->cache_pool && c->w >= c->cache_pool &&
          c->w < c->cache_pool + c->cache_pool_used)) {
        cuMemFree(c->w); c->w = 0;
    }
    for (int i = 0; i < DS4F_CUDA_MXFP4_CACHE_SLOTS; ++i) {
        cuda_mxfp4_cache_entry *e = &c->cache[i];
        if (e->valid && e->wkey == wkey && e->skey == skey &&
            e->rows == rows && e->cols == cols) {
            c->cache_hits++;
            e->age = ++c->cache_age;
            c->active_cache_slot = i;
            c->w = e->d; c->wb = e->bytes;
            c->rows = rows; c->cols = cols;
            return 0;
        }
    }
    uint8_t *p=(uint8_t *)packed;
    if(!p){
      if (!cuMemHostAlloc || !cuMemFreeHost) return -1;
      if (c->hwb < bytes) {
          if (c->hw) cuMemFreeHost(c->hw);
          c->hw = NULL; c->hwb = 0;
          if (cuMemHostAlloc(&c->hw, bytes, 0) != CUDA_SUCCESS) return -1;
          c->hwb = bytes;
      }
      p = (uint8_t *)c->hw;
      for (int r = 0; r < rows; ++r) for (size_t b = 0; b < nb; ++b) {
        uint8_t *q = p + ((size_t)r * nb + b) * 17; uint8_t e = s[(size_t)r * nb + b];
        /* DS4F's on-disk LUT is 2x the native E2M1 LUT, so e-1 plus
         * the 2x LUT is numerically equivalent to native LUT with e. */
        q[0] = e;
        const uint8_t *src = w + (size_t)r * rb + b * 16;
        /* Native block_mxfp4 packs element j with element j+16 in one byte;
         * this is the layout consumed by the FP4 MMQ dequantizer. */
        for (int j = 0; j < 16; ++j) {
            uint8_t lo = (src[j / 2] >> ((j & 1) * 4)) & 0xf;
            uint8_t hi = (src[8 + j / 2] >> ((j & 1) * 4)) & 0xf;
            q[1 + j] = (uint8_t)(lo | (hi << 4));
        }
      }
    }
    c->active_cache_slot = -1;
    int slot = -1;
    if (bytes <= c->cache_limit) {
        if (c->cache_no_evict && c->cache_bytes + bytes > c->cache_limit)
            return -1;
        while (c->cache_bytes + bytes > c->cache_limit) {
            int victim = -1;
            for (int i = 0; i < DS4F_CUDA_MXFP4_CACHE_SLOTS; ++i)
                if (c->cache[i].valid && (victim < 0 || c->cache[i].age < c->cache[victim].age)) victim = i;
            if (victim < 0) break;
            if (c->cache_pool) {
                if (c->n_pool_free < 128) {
                    c->pool_free[c->n_pool_free].off =
                        (size_t)(c->cache[victim].d - c->cache_pool);
                    c->pool_free[c->n_pool_free].size = c->cache[victim].bytes;
                    c->n_pool_free++;
                }
            } else cuMemFree(c->cache[victim].d);
            c->cache_evictions++;
            c->cache_bytes -= c->cache[victim].bytes;
            c->cache[victim].valid = 0;
        }
        for (int i = 0; i < DS4F_CUDA_MXFP4_CACHE_SLOTS; ++i)
            if (!c->cache[i].valid) { slot = i; break; }
    }
    c->cache_misses++;
    CUdeviceptr d = 0;
    if (c->cache_pool) {
        size_t aligned = (bytes + 255) & ~255;
        size_t off = (size_t)-1;
        for (int i = 0; i < c->n_pool_free; ++i)
            if (c->pool_free[i].size >= bytes) {
                off = c->pool_free[i].off;
                c->n_pool_free--;
                c->pool_free[i] = c->pool_free[c->n_pool_free];
                break;
            }
        if (off == (size_t)-1) {
            if (c->cache_pool_used + aligned > c->cache_pool_size) return -1;
            off = c->cache_pool_used;
            c->cache_pool_used += aligned;
        }
        d = c->cache_pool + off;
        if (cuMemcpyHtoD(d, p, bytes) != CUDA_SUCCESS) return -1;
    } else {
        CUresult ar = cuMemAlloc(&d, bytes);
        if (ar != CUDA_SUCCESS || cuMemcpyHtoD(d, p, bytes) != CUDA_SUCCESS) {
            if (d) cuMemFree(d);
            return -1;
        }
    }
    if (slot >= 0) {
        cuda_mxfp4_cache_entry *e = &c->cache[slot];
        e->wkey = wkey; e->skey = skey; e->d = d; e->bytes = bytes;
        e->age = ++c->cache_age; e->rows = rows; e->cols = cols; e->valid = 1;
        c->cache_bytes += bytes; c->active_cache_slot = slot;
    }
    c->w = d; c->wb = bytes; c->rows = rows; c->cols = cols;
    return 0;
}
int cuda_ds4f_mxfp4_load(cuda_ds4f_mxfp4 *c,const uint8_t *w,const uint8_t *s,int rows,int cols){
    return cuda_ds4f_mxfp4_load_any(c,w,s,NULL,rows,cols);
}
int cuda_ds4f_mxfp4_load_packed(cuda_ds4f_mxfp4 *c,const uint8_t *p,int rows,int cols){
    return cuda_ds4f_mxfp4_load_any(c,NULL,NULL,p,rows,cols);
}
static int cuda_ds4f_mxfp4_gemm_once(cuda_ds4f_mxfp4 *c, float *dst,
                                     const float *x, int M, int N, int K,
                                     CUdeviceptr out, int copy_back) {
    if (!c || (copy_back && !dst) || !x || !c->w || M < 1 || N != c->rows || K != c->cols || N % 128 || K % 32) return -1;
    if (cuCtxSetCurrent(c->ctx) != CUDA_SUCCESS) return -1;
    /* The x64 MMQ specialization requires a full 64-row tile.  Pad tiny
     * batches to the proven x128 path; normal prefill batches use x64/x128. */
    const int use64 = M < 128;
    const int Mp = use64 ? (M < 64 ? 64 : M) : ((M + 127) & ~127);
    size_t xb = (size_t)Mp * K * sizeof(float), q8b = (size_t)Mp * ((K + 255) & ~255) / 256 * 144 + 256 * 144, yb = (size_t)Mp * N * sizeof(float) + 1024;
    if (!c->x || c->xb < xb) {
        if (c->x) cuMemFree(c->x);
        if (cuMemAlloc(&c->x, xb) != CUDA_SUCCESS) return -1;
        c->xb = xb;
        /* The padded rows (M..Mp) must read as zero; zero the device buffer
         * once and only upload the M real rows each call. */
        if (cuMemsetD8(c->x, 0, xb) != CUDA_SUCCESS) return -1;
    }
    if (!c->q8 || c->q8b < q8b) { if (c->q8) cuMemFree(c->q8); if (cuMemAlloc(&c->q8, q8b) != CUDA_SUCCESS) return -1; c->q8b = q8b; }
    if (!c->y || c->yb < yb) { if (c->y) cuMemFree(c->y); if (cuMemAlloc(&c->y, yb) != CUDA_SUCCESS) return -1; c->yb = yb; }
    if (!out) out = c->y;
    if (!cuMemHostAlloc || !cuMemFreeHost) return -1;
    if (c->hxb < xb) {
        if (c->hx) cuMemFreeHost(c->hx);
        c->hx = NULL; c->hxb = 0;
        if (cuMemHostAlloc(&c->hx, xb, 0) != CUDA_SUCCESS) return -1;
        c->hxb = xb;
    }
    if (c->hyb < yb) {
        if (c->hy) cuMemFreeHost(c->hy);
        c->hy = NULL; c->hyb = 0;
        if (cuMemHostAlloc(&c->hy, yb, 0) != CUDA_SUCCESS) return -1;
        c->hyb = yb;
    }
    memcpy(c->hx, x, (size_t)M * K * sizeof(float));
    CUresult xr = cuMemcpyHtoD(c->x, c->hx, (size_t)M * K * sizeof(float));
    if (xr != CUDA_SUCCESS) return -1;
    long long ne00 = K, s01 = K, ne0 = (K + 511) & ~511;
    int by = ((int)ne0 + 63) / 64;
    long long s02 = 0, s03 = 0;
    int ne1 = Mp, ne2 = 1;
    CUdeviceptr ids0 = 0;
    void *qa[] = { &c->x, &ids0, &c->q8, &ne00, &s01, &s02, &s03, &ne0, &ne1, &ne2 };
    /* Everything runs on one FIFO stream, so the quant -> gemm -> fixup chain
     * is ordered without per-step syncs.  One sync (before the host download)
     * is enough; this removes ~4 synchronizations per expert GEMM, which is
     * the dominant per-call cost for the tiny routed buckets. */
    if (cuLaunchKernel(c->quant_fp4, Mp, by, 1, 32, 1, 1, 0, c->stream, qa, NULL) != CUDA_SUCCESS)
        return -1;
    int nty = (N + 127) / 128, ntx = use64 ? (Mp + 63) / 64 : (Mp + 127) / 128, tiles = nty * ntx;
    int waves = (tiles + c->nsm - 1) / c->nsm;
    int eff = 100 * tiles / (c->nsm * waves);
    /* A small bucket must not launch idle scheduling blocks.  In particular,
     * x64 MMQ/fixup can write past its valid tile set when sk exceeds tiles
     * (common for M<64 expert buckets on a 36-SM device). */
    int sk = tiles < c->nsm ? tiles : (eff >= 90 ? tiles : c->nsm);
    if (sk < 1) sk = 1;
    if (tiles == 1) sk = 1;
    if (c->verbose) fprintf(stderr, "mmq M=%d Mp=%d tiles=%d nsm=%d sk=%d fix=%d\n", M, Mp, tiles, c->nsm, sk, (tiles % sk) != 0);
    int fix = (tiles % sk) != 0;
    if (fix && !c->fixup) return -1;
    size_t fixb = fix ? (size_t)sk * 128 * 128 * sizeof(float) : 0;
    if (fixb > c->fixb) {
        if (c->tmpfix) cuMemFree(c->tmpfix);
        if (cuMemAlloc(&c->tmpfix, fixb) != CUDA_SUCCESS) return -1;
        c->fixb = fixb;
    }
    ds4f_u3 bp = fastdiv((unsigned long long)K / 32), one = fastdiv(1), ntxfd = fastdiv((unsigned)ntx);
    int zero = 0, stride = K / 32, nrows = N, ncols = M, ny = Mp, stride_col = N;
    CUdeviceptr nullp = 0;
    CUdeviceptr tmp = fix ? c->tmpfix : 0;
    CUfunction gemmfn = use64 ? c->gemm64 : c->gemm;
    CUfunction fixfn = use64 ? c->fixup64 : c->fixup;
    if (fix && !fixfn) return -1;
    void *a[] = { &c->w, &c->q8, &nullp, &nullp, &out, &tmp, &bp, &nrows, &ncols, &stride, &ny, &stride_col,
                  &one, &one, &zero, &zero, &zero, &one, &one, &zero, &zero, &zero, &ntxfd };
    if (cuLaunchKernel(gemmfn, sk, 1, 1, 32, 8, 1, 57856, c->stream, a, NULL) != CUDA_SUCCESS)
        return -1;
    if (fix) {
        void *fa[] = { &nullp, &nullp, &out, &c->tmpfix, &bp, &nrows, &ny, &stride_col,
                       &one, &zero, &one, &zero, &ntxfd };
        if (cuLaunchKernel(fixfn, sk, 4, 1, 32, 4, 1, 0, c->stream, fa, NULL) != CUDA_SUCCESS)
            return -1;
    }
    if (copy_back) {
        if (cuStreamSynchronize(c->stream) != CUDA_SUCCESS) return -1;
        if (cuMemcpyDtoH(c->hy, out, (size_t)M * N * sizeof(float)) != CUDA_SUCCESS) return -1;
        for (int r = 0; r < M; ++r)
            memcpy(dst + (size_t)r * N, (float *)c->hy + (size_t)r * N,
                   (size_t)N * sizeof(float));
    }
    return 0;
}

/* Match mmqv_quant_mxfp4's nearest E2M1 representation on the host and form
 * the residual term for the optional two-term activation decomposition. */
static void make_mxfp4_residual(float *res, const float *x, int M, int Mp, int K) {
    static const float lut[8] = { 0.0f, .5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    memset(res, 0, (size_t)Mp * K * sizeof(float));
    for (int r = 0; r < M; ++r) {
        for (int b = 0; b < K / 32; ++b) {
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float a = fabsf(x[(size_t)r * K + b * 32 + j]);
                if (a > amax) amax = a;
            }
            int e = amax > 0.0f ? (int)lrintf(log2f(amax)) - 2 + 127 : 0;
            if (e < 0) e = 0;
            if (e > 254) e = 254;
            float scale = amax > 0.0f ? ldexpf(1.0f, e - 127) : 0.0f;
            for (int j = 0; j < 32; ++j) {
                float v = x[(size_t)r * K + b * 32 + j];
                float av = scale > 0.0f ? v / scale : 0.0f;
                int q = 0;
                float err = fabsf(av) - lut[0];
                for (int i = 1; i < 8; ++i) {
                    float d = fabsf(fabsf(av) - lut[i]);
                    if (d < err) { err = d; q = i; }
                }
                float qv = scale * lut[q];
                if (av < 0.0f) qv = -qv;
                res[(size_t)r * K + b * 32 + j] = v - qv;
            }
        }
    }
}

void cuda_ds4f_mxfp4_set_terms(cuda_ds4f_mxfp4 *c, int terms) {
    if (c) c->terms = terms < 2 ? 1 : 2;
}

void cuda_ds4f_mxfp4_set_no_evict(cuda_ds4f_mxfp4 *c, int enabled) {
    if (c) c->cache_no_evict = enabled != 0;
}

int cuda_ds4f_mxfp4_gemm(cuda_ds4f_mxfp4 *c, float *dst, const float *x,
                         int M, int N, int K) {
    /* Pass 0 as the output so gemm_once() grabs the current c->y: evaluating
     * c->y at the call site captures a pointer that a growth realloc inside
     * gemm_once() then frees, and the queued kernel writes into freed VRAM. */
    if (!c || c->terms < 2)
        return cuda_ds4f_mxfp4_gemm_once(c, dst, x, M, N, K, 0, 1);
    int Mp = M < 128 ? (M < 64 ? 64 : M) : ((M + 127) & ~127);
    size_t xb = (size_t)Mp * K * sizeof(float);
    size_t yb = (size_t)Mp * N * sizeof(float);
    if (!cuMemHostAlloc || !cuMemFreeHost) return -1;
    if (c->hresb < xb) {
        if (c->hres) cuMemFreeHost(c->hres);
        c->hres = NULL; c->hresb = 0;
        if (cuMemHostAlloc(&c->hres, xb, 0) != CUDA_SUCCESS) return -1;
        c->hresb = xb;
    }
    if (!c->y2 || c->y2b < yb) {
        if (c->y2) cuMemFree(c->y2);
        c->y2 = 0; c->y2b = 0;
        if (cuMemAlloc(&c->y2, yb) != CUDA_SUCCESS) return -1;
        c->y2b = yb;
    }
    make_mxfp4_residual((float *)c->hres, x, M, Mp, K);
    if (cuda_ds4f_mxfp4_gemm_once(c, NULL, x, M, N, K, 0, 0) != 0 ||
        cuda_ds4f_mxfp4_gemm_once(c, NULL, (const float *)c->hres,
                                  M, N, K, c->y2, 0) != 0)
        return -1;
    int total = Mp * N;
    void *aa[] = { &c->y, &c->y2, &total };
    int blocks = (total + 255) / 256;
    if (cuLaunchKernel(c->add, blocks, 1, 1, 256, 1, 1, 0,
                       c->stream, aa, NULL) != CUDA_SUCCESS ||
        cuStreamSynchronize(c->stream) != CUDA_SUCCESS)
        return -1;
    if (cuMemcpyDtoH(c->hy, c->y, (size_t)M * N * sizeof(float)) != CUDA_SUCCESS)
        return -1;
    for (int r = 0; r < M; ++r)
        memcpy(dst + (size_t)r * N, (float *)c->hy + (size_t)r * N,
               (size_t)N * sizeof(float));
    return 0;
}


/* Async batched expert GEMM.  All tasks' quant -> gemm -> fixup (x2 for the
 * two-term residual) are queued on the single stream so they overlap on the
 * GPU; the host then syncs once and downloads every result.  Every weight must
 * already be resident in the cache (preload): the batch records the cached
 * device pointers, and a temp (non-cached) weight would be freed underneath a
 * queued kernel, so it returns -1 and the caller falls back to the per-call
 * path. */
static int cuda_ds4f_mxfp4_gemm_batch_any(cuda_ds4f_mxfp4 *c, int n,
    float *const *dst, const uint8_t *const *w, const uint8_t *const *s,
    const float *const *x, const int *M, const int *rows, const int *cols,
    int packed) {
    static int dbg = -1, dbg_cnt = 0;
    if (dbg < 0) { const char *d = getenv("DS4F_DBG_BATCH"); dbg = d ? atoi(d) : 0; }
    struct timespec t0, t1;
    if (dbg) clock_gettime(CLOCK_MONOTONIC, &t0);
    if (!c || n < 1 || n > 32 || cuCtxSetCurrent(c->ctx) != CUDA_SUCCESS) return -1;
    if (!cuMemHostAlloc || !cuMemFreeHost || !cuMemcpyHtoDAsync) return -1;
    if (n > c->cap_batch) {
        int cap = c->cap_batch ? c->cap_batch * 2 : 16;
        if (cap < n) cap = n;
        cuda_batch_buf *nb = (cuda_batch_buf *)realloc(c->b, (size_t)cap * sizeof(*nb));
        if (!nb) return -1;
        memset(nb + c->cap_batch, 0, (size_t)(cap - c->cap_batch) * sizeof(*nb));
        c->b = nb; c->cap_batch = cap;
    }
    c->n_batch = n;
    int terms = c->terms < 2 ? 1 : 2;
    CUdeviceptr dw[32]; int use64[32], Mp[32], sk[32], ntx[32], fix[32];
    /* pass 1: load every weight; all must be resident (no temp). */
    for (int i = 0; i < n; ++i) {
        int lrc=packed?cuda_ds4f_mxfp4_load_packed(c,w[i],rows[i],cols[i]):
                       cuda_ds4f_mxfp4_load(c,w[i],s[i],rows[i],cols[i]);
        if(lrc!=0)return -1;
        if (c->active_cache_slot < 0) return -1;   /* temp: not batch-safe */
        dw[i] = c->w;
        int N = rows[i], mm = M[i];
        use64[i] = mm < 128;
        Mp[i] = use64[i] ? (mm < 64 ? 64 : mm) : ((mm + 127) & ~127);
        int nty = (N + 127) / 128, ntxv = use64[i] ? (Mp[i] + 63) / 64 : (Mp[i] + 127) / 128;
        int tiles = nty * ntxv, waves = (tiles + c->nsm - 1) / c->nsm;
        int eff = 100 * tiles / (c->nsm * waves);
        int skv = tiles < c->nsm ? tiles : (eff >= 90 ? tiles : c->nsm);
        if (skv < 1) skv = 1;
        if (tiles == 1) skv = 1;
        sk[i] = skv; ntx[i] = ntxv; fix[i] = (tiles % skv) != 0;
        if (fix[i] && !c->fixup) return -1;
    }
    /* per-task fixup scratch, sized for each task's sk. */
    for (int i = 0; i < n; ++i)
        if (fix[i]) {
            size_t fb = (size_t)sk[i] * 128 * 128 * sizeof(float);
            if (fb > c->b[i].fixb) {
                if (c->b[i].tmpfix) cuMemFree(c->b[i].tmpfix);
                if (cuMemAlloc(&c->b[i].tmpfix, fb) != CUDA_SUCCESS) return -1;
                c->b[i].fixb = fb;
            }
        }
    /* pass 2: per-task buffers (grow as needed). */
    for (int i = 0; i < n; ++i) {
        cuda_batch_buf *B = &c->b[i];
        int K = cols[i], N = rows[i], mm = M[i];
        size_t xb = (size_t)Mp[i] * K * sizeof(float);
        size_t q8b = (size_t)Mp[i] * ((K + 255) & ~255) / 256 * 144 + 256 * 144;
        size_t yb = (size_t)Mp[i] * N * sizeof(float) + 1024;
        if (xb > B->xb) {
            if (B->x) cuMemFree(B->x);
            CUresult arx = cuMemAlloc(&B->x, xb);
            if (arx != CUDA_SUCCESS) return -1;
            B->xb = xb;
            if (cuMemsetD8(B->x, 0, xb) != CUDA_SUCCESS) return -1;
        }
        if (q8b > B->q8b) { if (B->q8) cuMemFree(B->q8); if (cuMemAlloc(&B->q8, q8b) != CUDA_SUCCESS) return -1; B->q8b = q8b; }
        if (yb > B->yb) {
            if (B->y) cuMemFree(B->y);
            CUresult ary = cuMemAlloc(&B->y, yb);
            if (ary != CUDA_SUCCESS) return -1;
            B->yb = yb;
            if (terms == 2) {
                if (B->y2) cuMemFree(B->y2);
                if (cuMemAlloc(&B->y2, yb) != CUDA_SUCCESS) return -1;
            }
        }
        if (B->hxb < xb) {
            if (B->hx) cuMemFreeHost(B->hx);
            B->hx = NULL; B->hxb = 0;
            if (cuMemHostAlloc((void **)&B->hx, xb, 0) != CUDA_SUCCESS) return -1;
            B->hxb = xb;
        }
        if (B->hyb2 < (size_t)mm * N * sizeof(float)) {
            if (B->hy) cuMemFreeHost(B->hy);
            B->hy = NULL; B->hyb2 = 0;
            if (cuMemHostAlloc((void **)&B->hy, (size_t)mm * N * sizeof(float), 0) != CUDA_SUCCESS) return -1;
            B->hyb2 = (size_t)mm * N * sizeof(float);
        }
        if (terms == 2 && B->hresb < xb) {
            if (B->hres) cuMemFreeHost(B->hres);
            B->hres = NULL; B->hresb = 0;
            if (cuMemHostAlloc((void **)&B->hres, xb, 0) != CUDA_SUCCESS) return -1;
            B->hresb = xb;
        }
    }
    /* pass 3: queue everything on the stream (no syncs between tasks). */
    for (int i = 0; i < n; ++i) {
        cuda_batch_buf *B = &c->b[i];
        int K = cols[i], N = rows[i], mm = M[i];
        memcpy(B->hx, x[i], (size_t)mm * K * sizeof(float));
        if (cuMemcpyHtoDAsync(B->x, B->hx, (size_t)mm * K * sizeof(float), c->stream) != CUDA_SUCCESS) return -1;
        long long ne00 = K, s01 = K, ne0 = (K + 511) & ~511;
        int by = ((int)ne0 + 63) / 64;
        long long s02 = 0, s03 = 0;
        int ne1 = Mp[i], ne2 = 1;
        CUdeviceptr ids0 = 0;
        void *qa[] = { &B->x, &ids0, &B->q8, &ne00, &s01, &s02, &s03, &ne0, &ne1, &ne2 };
        if (cuLaunchKernel(c->quant_fp4, Mp[i], by, 1, 32, 1, 1, 0, c->stream, qa, NULL) != CUDA_SUCCESS) return -1;
        ds4f_u3 bp = fastdiv((unsigned long long)K / 32), one = fastdiv(1), ntxfd = fastdiv((unsigned)ntx[i]);
        int zero = 0, stride = K / 32, nrows = N, ncols = mm, ny = Mp[i], stride_col = N;
        CUdeviceptr nullp = 0;
        CUdeviceptr tmp = fix[i] ? c->b[i].tmpfix : 0;
        CUfunction gemmfn = use64[i] ? c->gemm64 : c->gemm;
        CUfunction fixfn = use64[i] ? c->fixup64 : c->fixup;
        void *a[] = { &dw[i], &B->q8, &nullp, &nullp, &B->y, &tmp, &bp, &nrows, &ncols, &stride, &ny, &stride_col,
                      &one, &one, &zero, &zero, &zero, &one, &one, &zero, &zero, &zero, &ntxfd };
        if (cuLaunchKernel(gemmfn, sk[i], 1, 1, 32, 8, 1, 57856, c->stream, a, NULL) != CUDA_SUCCESS) return -1;
        if (fix[i]) {
            void *fa[] = { &nullp, &nullp, &B->y, &B->tmpfix, &bp, &nrows, &ny, &stride_col,
                           &one, &zero, &one, &zero, &ntxfd };
            if (cuLaunchKernel(fixfn, sk[i], 4, 1, 32, 4, 1, 0, c->stream, fa, NULL) != CUDA_SUCCESS) return -1;
        }
        if (terms == 2) {
            if (!c->residual) return -1;
            long long r00 = K; int r1 = Mp[i];
            void *ra[] = { &B->x, &B->x, &r00, &r1 };
            if (cuLaunchKernel(c->residual, Mp[i], K / 64, 1, 32, 1, 1, 0, c->stream, ra, NULL) != CUDA_SUCCESS) return -1;
            if (cuLaunchKernel(c->quant_fp4, Mp[i], by, 1, 32, 1, 1, 0, c->stream, qa, NULL) != CUDA_SUCCESS) return -1;
            void *a2[] = { &dw[i], &B->q8, &nullp, &nullp, &B->y2, &tmp, &bp, &nrows, &ncols, &stride, &ny, &stride_col,
                           &one, &one, &zero, &zero, &zero, &one, &one, &zero, &zero, &zero, &ntxfd };
            if (cuLaunchKernel(gemmfn, sk[i], 1, 1, 32, 8, 1, 57856, c->stream, a2, NULL) != CUDA_SUCCESS) return -1;
            if (fix[i]) {
                void *fa2[] = { &nullp, &nullp, &B->y2, &B->tmpfix, &bp, &nrows, &ny, &stride_col,
                                &one, &zero, &one, &zero, &ntxfd };
                if (cuLaunchKernel(fixfn, sk[i], 4, 1, 32, 4, 1, 0, c->stream, fa2, NULL) != CUDA_SUCCESS) return -1;
            }
            int total = Mp[i] * N;
            void *aa[] = { &B->y, &B->y2, &total };
            if (cuLaunchKernel(c->add, (total + 255) / 256, 1, 1, 256, 1, 1, 0, c->stream, aa, NULL) != CUDA_SUCCESS) return -1;
        }
    }
    if (dbg) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        if (dbg_cnt++ < 40 || (dbg_cnt % 100) == 0)
            fprintf(stderr, "BATCH n=%d terms=%d %.3f ms\n", n, terms, ms);
    }
    /* pass 4: sync once, then download every result. */
    CUresult syncr = cuStreamSynchronize(c->stream);
    if (syncr != CUDA_SUCCESS) return -1;
    for (int i = 0; i < n; ++i) {
        cuda_batch_buf *B = &c->b[i];
        int N = rows[i], mm = M[i];
        if (cuMemcpyDtoH(B->hy, B->y, (size_t)mm * N * sizeof(float)) != CUDA_SUCCESS) return -1;
        for (int r = 0; r < mm; ++r)
            memcpy(dst[i] + (size_t)r * N, (float *)B->hy + (size_t)r * N, (size_t)N * sizeof(float));
    }
    return 0;
}
int cuda_ds4f_mxfp4_gemm_batch(cuda_ds4f_mxfp4 *c,int n,float *const *dst,const uint8_t *const *w,const uint8_t *const *s,const float *const *x,const int *M,const int *rows,const int *cols){
    return cuda_ds4f_mxfp4_gemm_batch_any(c,n,dst,w,s,x,M,rows,cols,0);
}
int cuda_ds4f_mxfp4_gemm_batch_packed(cuda_ds4f_mxfp4 *c,int n,float *const *dst,const uint8_t *const *p,const float *const *x,const int *M,const int *rows,const int *cols){
    return cuda_ds4f_mxfp4_gemm_batch_any(c,n,dst,p,NULL,x,M,rows,cols,1);
}

/* Preload a weight into the cache so the async batch's loads are hits.  The
 * repack + upload happen now (once), not during the measured prefill. */
int cuda_ds4f_mxfp4_warm(cuda_ds4f_mxfp4 *c, const uint8_t *w,
                         const uint8_t *scale, int rows, int cols) {
    if (!c || cuCtxSetCurrent(c->ctx) != CUDA_SUCCESS) return -1;
    return cuda_ds4f_mxfp4_load(c, w, scale, rows, cols);
}
