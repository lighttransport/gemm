/*
 * cuda_runner_common.h - Shared host-side utilities for CUDA runners
 *
 * Usage:
 *   #define CUDA_RUNNER_COMMON_IMPLEMENTATION
 *   #include "../cuda_runner_common.h"
 *
 * Provides: CU_CHECK/CU_CHECK_NULL macros, NVRTC compile, F32→F16/FP8
 * conversion, raw GPU upload.
 *
 * Requires: cuda.h, nvrtc.h (via cuew or system headers)
 */
#ifndef CUDA_RUNNER_COMMON_H
#define CUDA_RUNNER_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Error checking macros (return -1 or NULL) ---- */

#define CU_CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *es = "?"; cuGetErrorString(err, &es); \
        fprintf(stderr, "CUDA error %s:%d: %s (%d)\n", __FILE__, __LINE__, es, (int)err); \
        return -1; \
    } \
} while(0)

#define CU_CHECK_NULL(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *es = "?"; cuGetErrorString(err, &es); \
        fprintf(stderr, "CUDA error %s:%d: %s (%d)\n", __FILE__, __LINE__, es, (int)err); \
        return NULL; \
    } \
} while(0)

/* ---- Allocation helpers (rollback-friendly) ----
 *
 * CU_FREE / CU_CHECKED_ALLOC are for functions that allocate several
 * temporaries and need to roll back on any failure. Pair with a `fail:`
 * label that frees every locally-owned pointer (each pointer should be
 * initialized to 0 at declaration so unconditional `CU_FREE` is safe).
 *
 *   CUdeviceptr a = 0, b = 0;
 *   CU_CHECKED_ALLOC(a, n_bytes, "scratch_a", fail);
 *   CU_CHECKED_ALLOC(b, n_bytes, "scratch_b", fail);
 *   ...
 *   return success;
 *   fail:
 *       CU_FREE(a); CU_FREE(b);
 *       return error;
 */

#define CU_FREE(ptr) do { \
    if ((ptr)) { cuMemFree((ptr)); (ptr) = 0; } \
} while(0)

#define CU_CHECKED_ALLOC(dst, bytes, label, fail_label) do { \
    CUresult _ce = cuMemAlloc(&(dst), (size_t)(bytes)); \
    if (_ce != CUDA_SUCCESS) { \
        fprintf(stderr, "cuda: alloc failed for %s (%zu bytes, err=%d)\n", \
                (label), (size_t)(bytes), (int)_ce); \
        (dst) = 0; \
        goto fail_label; \
    } \
} while(0)

/* ---- Resizable device buffer slot ----
 *
 * Drop-in primitive for per-runner activation caches: keep an array of
 * cu_buf_slot indexed by a runner-local enum, then call ensure() on each
 * slot before use. Slots that need a larger capacity get freed and
 * re-allocated; smaller requests hit the cached buffer.
 *
 * Zero-init is valid (`cu_buf_slot s = {0};`), so a runner allocated with
 * calloc() needs no explicit init step.
 */

typedef struct {
    CUdeviceptr ptr;
    size_t      cap;
} cu_buf_slot;

/* Bytes that would be added if `bytes` is ensured. Useful for VRAM
 * accounting before committing to a working-set allocation. */
static inline size_t cu_buf_slot_growth(const cu_buf_slot *s, size_t bytes) {
    return (s && bytes > s->cap) ? bytes - s->cap : 0;
}

/* Ensure slot holds at least `bytes`. Returns 0 on success, -1 on alloc
 * failure (slot is cleared in that case). `bytes == 0` is a no-op. */
static inline int cu_buf_slot_ensure(cu_buf_slot *s, size_t bytes, const char *label) {
    if (!s || bytes == 0) return 0;
    if (s->ptr && s->cap >= bytes) return 0;
    if (s->ptr) { cuMemFree(s->ptr); s->ptr = 0; s->cap = 0; }
    CUresult err = cuMemAlloc(&s->ptr, bytes);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda: cache alloc failed for %s (%zu bytes, err=%d)\n",
                label ? label : "?", bytes, (int)err);
        s->ptr = 0; s->cap = 0;
        return -1;
    }
    s->cap = bytes;
    return 0;
}

static inline void cu_buf_slot_free(cu_buf_slot *s) {
    if (!s) return;
    if (s->ptr) { cuMemFree(s->ptr); s->ptr = 0; }
    s->cap = 0;
}

static inline void cu_buf_slots_free(cu_buf_slot *slots, int n) {
    if (!slots) return;
    for (int i = 0; i < n; i++) cu_buf_slot_free(&slots[i]);
}

/* Shared F32 split-key attention partial workspace:
 *   o: [n_splits, n_tok, n_heads * head_dim] float
 *   m/l: [n_splits, n_tok, n_heads] float
 */
typedef struct {
    cu_buf_slot o;
    cu_buf_slot m;
    cu_buf_slot l;
} cu_split_attn_f32_workspace;

static inline int cu_mul_size_checked(size_t a, size_t b, size_t *out) {
    if (!out) return -1;
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static inline int cu_split_attn_f32_workspace_bytes(int n_tok, int n_heads,
                                                    int head_dim, int n_splits,
                                                    size_t *o_bytes,
                                                    size_t *state_bytes) {
    size_t dim, elems;
    if (n_tok <= 0 || n_heads <= 0 || head_dim <= 0 || n_splits <= 0 ||
        !o_bytes || !state_bytes) {
        return -1;
    }
    if (cu_mul_size_checked((size_t)n_heads, (size_t)head_dim, &dim) != 0 ||
        cu_mul_size_checked((size_t)n_splits, (size_t)n_tok, &elems) != 0 ||
        cu_mul_size_checked(elems, dim, o_bytes) != 0 ||
        cu_mul_size_checked(*o_bytes, sizeof(float), o_bytes) != 0) {
        return -1;
    }
    if (cu_mul_size_checked(elems, (size_t)n_heads, state_bytes) != 0 ||
        cu_mul_size_checked(*state_bytes, sizeof(float), state_bytes) != 0) {
        return -1;
    }
    return 0;
}

static inline int cu_split_attn_f32_workspace_ensure(cu_split_attn_f32_workspace *w,
                                                     int n_tok, int n_heads,
                                                     int head_dim, int n_splits,
                                                     const char *label) {
    size_t o_bytes = 0, state_bytes = 0;
    char o_label[96], m_label[96], l_label[96];
    if (!w ||
        cu_split_attn_f32_workspace_bytes(n_tok, n_heads, head_dim, n_splits,
                                          &o_bytes, &state_bytes) != 0) {
        fprintf(stderr, "cuda: invalid split attention workspace shape for %s\n",
                label ? label : "?");
        return -1;
    }
    snprintf(o_label, sizeof(o_label), "%s.part_o", label ? label : "split_attn");
    snprintf(m_label, sizeof(m_label), "%s.part_m", label ? label : "split_attn");
    snprintf(l_label, sizeof(l_label), "%s.part_l", label ? label : "split_attn");
    if (cu_buf_slot_ensure(&w->o, o_bytes, o_label) != 0) return -1;
    if (cu_buf_slot_ensure(&w->m, state_bytes, m_label) != 0) return -1;
    if (cu_buf_slot_ensure(&w->l, state_bytes, l_label) != 0) return -1;
    return 0;
}

static inline void cu_split_attn_f32_workspace_free(cu_split_attn_f32_workspace *w) {
    if (!w) return;
    cu_buf_slot_free(&w->o);
    cu_buf_slot_free(&w->m);
    cu_buf_slot_free(&w->l);
}

/* Async zero of a device range on `stream`. Soft-failure: logs to stderr
 * but does not return an error (matches the cuda_llm reset helper). */
static inline void cu_async_zero(CUdeviceptr ptr, size_t bytes, CUstream stream,
                                 const char *label) {
    if (!ptr || bytes == 0) return;
    CUresult err = cuMemsetD8Async(ptr, 0, bytes, stream);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda: async zero failed for %s (err=%d)\n",
                label ? label : "?", (int)err);
    }
}

/* ======================================================================== */
#ifdef CUDA_RUNNER_COMMON_IMPLEMENTATION

/* ---- F32 → F16 conversion (truncation, no rounding) ---- */

static inline uint16_t cu_f32_to_f16(float f) {
    union { float f; uint32_t i; } u;
    u.f = f;
    uint32_t x = u.i;
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000);
    int32_t exp = ((x >> 23) & 0xFF) - 127;
    uint32_t mant = x & 0x7FFFFF;
    if (exp > 15) return sign | 0x7C00;
    if (exp < -14) {
        if (exp < -24) return sign;
        mant |= 0x800000;
        mant >>= (-1 - exp);
        return sign | (uint16_t)(mant >> 13);
    }
    return sign | (uint16_t)((exp + 15) << 10) | (uint16_t)(mant >> 13);
}

/* ---- F32 → FP8 E4M3 conversion (bias=7, range [-448,448]) ---- */

static inline uint8_t cu_f32_to_fp8_e4m3(float f) {
    /* Round-to-nearest-even, saturate-to-finite — matches the hardware
     * cvt.rn.satfinite.e4m3x2.f32 the device kernels use. E4M3 = 1 sign /
     * 4 exp (bias 7) / 3 mantissa; max finite 448 (exp=15,mant=6); (exp=15,
     * mant=7) is NaN; smallest normal 2^-6; subnormals are k*2^-9, k=1..7. */
    if (f != f) return 0x7F; /* NaN -> canonical E4M3 NaN */
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    uint8_t s8 = (uint8_t)(((bits >> 31) & 1u) << 7);
    float af = (f < 0.0f) ? -f : f;
    if (af == 0.0f) return s8;                       /* +/-0 keep sign */
    if (af >= 448.0f) return (uint8_t)(s8 | (0xFu << 3) | 0x6u); /* saturate (incl. Inf) */

    int32_t E = (int32_t)((bits >> 23) & 0xFF) - 127; /* unbiased f32 exponent */
    if (E < -6) {
        /* Subnormal range: nearest multiple of 2^-9 (k=0..8), RNE. k==8
         * carries up into the smallest normal (exp field 1, mant 0). */
        float scaled = af * 512.0f;                  /* exact power-of-two scale */
        uint32_t k = (uint32_t)scaled;               /* truncate; scaled >= 0 */
        float frac = scaled - (float)k;
        if (frac > 0.5f || (frac == 0.5f && (k & 1u))) k++;
        if (k == 0u) return s8;
        if (k < 8u) return (uint8_t)(s8 | (k & 0x7u));        /* subnormal */
        return (uint8_t)(s8 | (1u << 3));                     /* -> smallest normal */
    }
    /* Normal range: reduce the 23-bit mantissa to 3 bits with RNE. */
    uint32_t mant = (bits >> 20) & 0x7u;             /* top 3 mantissa bits */
    uint32_t rem  = bits & 0xFFFFFu;                 /* low 20 bits = round info */
    if (rem > 0x80000u || (rem == 0x80000u && (mant & 1u))) {
        if (++mant == 8u) { mant = 0u; E++; }        /* mantissa carry bumps exp */
    }
    int32_t exp_field = E + 7;
    if (exp_field > 15 || (exp_field == 15 && mant > 6u))
        return (uint8_t)(s8 | (0xFu << 3) | 0x6u);   /* saturate to 448 */
    return (uint8_t)(s8 | ((uint32_t)exp_field << 3) | mant);
}

/* ---- Raw GPU upload (synchronous) ---- */

static CUdeviceptr cu_upload_raw(const void *data, size_t bytes) {
    if (!data || bytes == 0) return 0;
    CUdeviceptr d;
    if (cuMemAlloc(&d, bytes) != CUDA_SUCCESS) return 0;
    cuMemcpyHtoD(d, data, bytes);
    return d;
}

/* ---- Persistent kernel cache (CUBIN/PTX on disk) ----------------------
 *
 * Skips NVRTC entirely on a hit. The key folds in the kernel source, target
 * sm, the compile-flag signature, and the NVRTC/driver versions, so a toolkit,
 * driver, or precision-flag change naturally lands on a fresh file. Best-effort:
 * any I/O or load failure falls back to a normal compile. Disable with
 * CUDA_RUNNER_NO_KERNEL_CACHE=1; relocate with CUDA_RUNNER_KERNEL_CACHE_DIR.
 */
#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

static uint64_t cu_fnv1a64(const void *data, size_t len, uint64_t h) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static int cu_mkdir_p(const char *path) {
    char tmp[1024];
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(tmp)) return -1;
    memcpy(tmp, path, len + 1);
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) return -1;
    return 0;
}

static int cu_kernel_cache_dir(char *buf, size_t buflen) {
    const char *env  = getenv("CUDA_RUNNER_KERNEL_CACHE_DIR");
    const char *xdg  = getenv("XDG_CACHE_HOME");
    const char *home = getenv("HOME");
    if (env && env[0])        snprintf(buf, buflen, "%s", env);
    else if (xdg && xdg[0])   snprintf(buf, buflen, "%s/cuda_runner", xdg);
    else if (home && home[0]) snprintf(buf, buflen, "%s/.cache/cuda_runner", home);
    else                      snprintf(buf, buflen, "/tmp/cuda_runner_cache");
    return cu_mkdir_p(buf);
}

/* Build the cache file path. Returns 0 on success, -1 if caching is disabled
 * or the directory could not be created. */
static int cu_kernel_cache_path(char *buf, size_t buflen, const char *source,
                                const char *prog_name, int sm, unsigned flagmask) {
    char dir[1024], tag[64], meta[160];
    int nvmaj = 0, nvmin = 0, drv = 0, mlen, j = 0;
    uint64_t key = 1469598103934665603ULL;
    if (getenv("CUDA_RUNNER_NO_KERNEL_CACHE")) return -1;
    if (cu_kernel_cache_dir(dir, sizeof(dir)) != 0) return -1;
    if (nvrtcVersion) nvrtcVersion(&nvmaj, &nvmin);
    if (cuDriverGetVersion) cuDriverGetVersion(&drv);
    key = cu_fnv1a64(source, strlen(source), key);
    mlen = snprintf(meta, sizeof(meta), "|sm=%d|fl=%u|nv=%d.%d|drv=%d",
                    sm, flagmask, nvmaj, nvmin, drv);
    if (mlen > 0) key = cu_fnv1a64(meta, (size_t)mlen, key);
    for (int i = 0; prog_name && prog_name[i] && j < (int)sizeof(tag) - 1; i++) {
        char c = prog_name[i];
        tag[j++] = (c == '/' || c == '\\' || c == ' ') ? '_' : c;
    }
    tag[j] = '\0';
    if (j == 0) snprintf(tag, sizeof(tag), "kernels");
    snprintf(buf, buflen, "%s/%s.sm%d.%016llx.crk", dir, tag, sm,
             (unsigned long long)key);
    return 0;
}

/* On-disk format: "CRK1" magic, 1-byte type (0=CUBIN, 1=PTX), 3 pad, blob. */
static int cu_try_load_cached(CUmodule *module, const char *path,
                              int verbose, const char *prefix) {
    FILE *fp = fopen(path, "rb");
    long fsz;
    unsigned char *data;
    size_t rd;
    int type;
    CUresult err;
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END); fsz = ftell(fp); fseek(fp, 0, SEEK_SET);
    if (fsz <= 8) { fclose(fp); return -1; }
    data = (unsigned char *)malloc((size_t)fsz + 1);
    if (!data) { fclose(fp); return -1; }
    rd = fread(data, 1, (size_t)fsz, fp);
    fclose(fp);
    if (rd != (size_t)fsz || memcmp(data, "CRK1", 4) != 0) { free(data); return -1; }
    data[fsz] = '\0';  /* NUL-terminate so the PTX path sees a valid string */
    type = data[4];
    err = (type == 0) ? cuModuleLoadData(module, data + 8)
                      : cuModuleLoadDataEx(module, data + 8, 0, NULL, NULL);
    free(data);
    if (err != CUDA_SUCCESS) {
        if (verbose >= 2)
            fprintf(stderr, "%s: cached module load failed (%d), recompiling\n",
                    prefix, (int)err);
        remove(path);
        return -1;
    }
    return 0;
}

static void cu_write_cache(const char *path, int type, const void *blob,
                           size_t blob_sz, int verbose, const char *prefix) {
    char tmp[1100];
    FILE *fp;
    unsigned char hdr[8] = { 'C', 'R', 'K', '1', (unsigned char)type, 0, 0, 0 };
    int ok;
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, (int)getpid());
    fp = fopen(tmp, "wb");
    if (!fp) { if (verbose >= 2) fprintf(stderr, "%s: cache open failed\n", prefix); return; }
    ok = (fwrite(hdr, 1, 8, fp) == 8) && (fwrite(blob, 1, blob_sz, fp) == blob_sz);
    fclose(fp);
    if (!ok) {
        remove(tmp);
        if (verbose >= 2) fprintf(stderr, "%s: cache write failed\n", prefix);
        return;
    }
    if (rename(tmp, path) != 0) {
        remove(tmp);
        if (verbose >= 2) fprintf(stderr, "%s: cache rename failed\n", prefix);
    } else if (verbose >= 2) {
        fprintf(stderr, "%s: wrote kernel cache %s\n", prefix, path);
    }
}
#endif /* !_WIN32 */

/* ---- NVRTC kernel compilation ---- */

static int cu_compile_kernels(CUmodule *module, CUdevice device,
                               const char *source, const char *prog_name,
                               int verbose, const char *prefix) {
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    int sm = major * 10 + minor;

    int use_fast_math = (getenv("CUDA_RUNNER_PRECISE_MATH") == NULL);
    int no_fmad = (getenv("CUDA_RUNNER_NO_FMAD") != NULL);

#ifndef _WIN32
    unsigned flagmask = (use_fast_math ? 1u : 0u) | (no_fmad ? 2u : 0u);
    char cache_path[1024];
    int have_cache = (cu_kernel_cache_path(cache_path, sizeof(cache_path),
                                           source, prog_name, sm, flagmask) == 0);
    if (have_cache && cu_try_load_cached(module, cache_path, verbose, prefix) == 0) {
        if (verbose >= 1)
            fprintf(stderr, "%s: loaded cached kernels for sm_%d\n", prefix, sm);
        return sm;
    }
#endif

    if (verbose >= 1)
        fprintf(stderr, "%s: compiling kernels for sm_%d ...\n", prefix, sm);

    nvrtcProgram prog;
    nvrtcResult cres = nvrtcCreateProgram(&prog, source, prog_name, 0, NULL, NULL);
    if (cres != NVRTC_SUCCESS) {
        fprintf(stderr, "%s: nvrtcCreateProgram failed: %d\n", prefix, (int)cres);
        return -1;
    }

    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d", sm);
    const char *opts[3] = { arch, NULL, NULL };
    int nopts = 1;
    if (use_fast_math) {
        opts[nopts++] = "--use_fast_math";
    }
    if (no_fmad) {
        opts[nopts++] = "--fmad=false";
    }
    nvrtcResult nres = nvrtcCompileProgram(prog, nopts, opts);

    if (nres != NVRTC_SUCCESS) {
        fprintf(stderr, "%s: NVRTC compile error %d\n", prefix, (int)nres);
        size_t log_sz;
        nvrtcGetProgramLogSize(prog, &log_sz);
        char *log = (char *)malloc(log_sz + 1);
        nvrtcGetProgramLog(prog, log);
        log[log_sz] = '\0';
        fprintf(stderr, "%s: NVRTC log (%zu bytes):\n%s\n", prefix, log_sz, log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return -1;
    }

    /* Prefer CUBIN (native binary) over PTX.  CUBIN avoids the
     * CUDA_ERROR_UNSUPPORTED_PTX_VERSION failure that occurs when the NVRTC
     * toolkit version is newer than the installed driver (e.g. NVRTC 13.1
     * emits PTX 9.1 which a CUDA 13.0 driver cannot JIT-compile). */
    CUresult err;
    size_t cubin_sz = 0;
    char *blob = NULL;   /* CUBIN/PTX blob, kept until after the cache write */
    size_t blob_sz = 0;
    int blob_type = 0;   /* 0 = CUBIN, 1 = PTX */
    if (nvrtcGetCUBINSize && nvrtcGetCUBIN &&
        nvrtcGetCUBINSize(prog, &cubin_sz) == NVRTC_SUCCESS && cubin_sz > 0) {
        blob = (char *)malloc(cubin_sz);
        blob_sz = cubin_sz;
        blob_type = 0;
        nvrtcGetCUBIN(prog, blob);
        nvrtcDestroyProgram(&prog);
        if (verbose >= 2)
            fprintf(stderr, "%s: loading CUBIN (%zu bytes)\n", prefix, cubin_sz);
        err = cuModuleLoadData(module, blob);
    } else {
        /* Fallback to PTX (works when toolkit <= driver) */
        size_t ptx_sz;
        nvrtcGetPTXSize(prog, &ptx_sz);
        blob = (char *)malloc(ptx_sz);
        blob_sz = ptx_sz;
        blob_type = 1;
        nvrtcGetPTX(prog, blob);
        nvrtcDestroyProgram(&prog);
        if (verbose >= 2)
            fprintf(stderr, "%s: loading PTX (%zu bytes)\n", prefix, ptx_sz);
        if (verbose >= 3) {
            char path[256];
            snprintf(path, sizeof(path), "/tmp/%s.ptx", prog_name);
            FILE *fp = fopen(path, "w");
            if (fp) { fwrite(blob, 1, ptx_sz, fp); fclose(fp);
                fprintf(stderr, "%s: PTX saved to %s\n", prefix, path); }
        }
        err = cuModuleLoadDataEx(module, blob, 0, NULL, NULL);
    }
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "%s: cuModuleLoad failed: %d\n", prefix, (int)err);
        free(blob);
        return -1;
    }

#ifndef _WIN32
    if (have_cache && blob)
        cu_write_cache(cache_path, blob_type, blob, blob_sz, verbose, prefix);
#endif
    free(blob);
    return sm;
}

#endif /* CUDA_RUNNER_COMMON_IMPLEMENTATION */
#endif /* CUDA_RUNNER_COMMON_H */
