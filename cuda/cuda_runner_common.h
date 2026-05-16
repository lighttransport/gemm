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
    if (f != f) return 0x7F; /* NaN */
    if (f == 0.0f) return 0x00;
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 7; /* rebias to E4M3 */
    uint32_t mant = (bits >> 20) & 0x7; /* top 3 mantissa bits */
    if (exp >= 15) { exp = 15; mant = 0x6; } /* clamp to max finite */
    if (exp <= 0) return (uint8_t)(sign << 7); /* flush subnormals to zero */
    return (uint8_t)((sign << 7) | ((exp & 0xF) << 3) | (mant & 0x7));
}

/* ---- Raw GPU upload (synchronous) ---- */

static CUdeviceptr cu_upload_raw(const void *data, size_t bytes) {
    if (!data || bytes == 0) return 0;
    CUdeviceptr d;
    if (cuMemAlloc(&d, bytes) != CUDA_SUCCESS) return 0;
    cuMemcpyHtoD(d, data, bytes);
    return d;
}

/* ---- NVRTC kernel compilation ---- */

static int cu_compile_kernels(CUmodule *module, CUdevice device,
                               const char *source, const char *prog_name,
                               int verbose, const char *prefix) {
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    int sm = major * 10 + minor;

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
    const char *opts[] = { arch, "--use_fast_math" };
    nvrtcResult nres = nvrtcCompileProgram(prog, 2, opts);

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
    if (nvrtcGetCUBINSize && nvrtcGetCUBIN &&
        nvrtcGetCUBINSize(prog, &cubin_sz) == NVRTC_SUCCESS && cubin_sz > 0) {
        char *cubin = (char *)malloc(cubin_sz);
        nvrtcGetCUBIN(prog, cubin);
        nvrtcDestroyProgram(&prog);
        if (verbose >= 2)
            fprintf(stderr, "%s: loading CUBIN (%zu bytes)\n", prefix, cubin_sz);
        err = cuModuleLoadData(module, cubin);
        free(cubin);
    } else {
        /* Fallback to PTX (works when toolkit <= driver) */
        size_t ptx_sz;
        nvrtcGetPTXSize(prog, &ptx_sz);
        char *ptx = (char *)malloc(ptx_sz);
        nvrtcGetPTX(prog, ptx);
        nvrtcDestroyProgram(&prog);
        if (verbose >= 2)
            fprintf(stderr, "%s: loading PTX (%zu bytes)\n", prefix, ptx_sz);
        if (verbose >= 3) {
            char path[256];
            snprintf(path, sizeof(path), "/tmp/%s.ptx", prog_name);
            FILE *fp = fopen(path, "w");
            if (fp) { fwrite(ptx, 1, ptx_sz, fp); fclose(fp);
                fprintf(stderr, "%s: PTX saved to %s\n", prefix, path); }
        }
        err = cuModuleLoadDataEx(module, ptx, 0, NULL, NULL);
        free(ptx);
    }
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "%s: cuModuleLoad failed: %d\n", prefix, (int)err);
        return -1;
    }

    return sm;
}

#endif /* CUDA_RUNNER_COMMON_IMPLEMENTATION */
#endif /* CUDA_RUNNER_COMMON_H */
