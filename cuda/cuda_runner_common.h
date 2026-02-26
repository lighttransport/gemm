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

/* ======================================================================== */
#ifdef CUDA_RUNNER_COMMON_IMPLEMENTATION

/* ---- F32 → F16 conversion (truncation, no rounding) ---- */

static uint16_t cu_f32_to_f16(float f) {
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

static uint8_t cu_f32_to_fp8_e4m3(float f) {
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
    if (nvrtcCreateProgram(&prog, source, prog_name, 0, NULL, NULL) != NVRTC_SUCCESS)
        return -1;

    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d", sm);
    const char *opts[] = { arch, "--use_fast_math" };
    nvrtcResult nres = nvrtcCompileProgram(prog, 2, opts);

    if (nres != NVRTC_SUCCESS) {
        size_t log_sz;
        nvrtcGetProgramLogSize(prog, &log_sz);
        if (log_sz > 1) {
            char *log = (char *)malloc(log_sz);
            nvrtcGetProgramLog(prog, log);
            fprintf(stderr, "%s: NVRTC log:\n%s\n", prefix, log);
            free(log);
        }
        nvrtcDestroyProgram(&prog);
        return -1;
    }

    size_t ptx_sz;
    nvrtcGetPTXSize(prog, &ptx_sz);
    char *ptx = (char *)malloc(ptx_sz);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    if (verbose >= 3) {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/%s.ptx", prog_name);
        FILE *fp = fopen(path, "w");
        if (fp) { fwrite(ptx, 1, ptx_sz, fp); fclose(fp);
            fprintf(stderr, "%s: PTX saved to %s\n", prefix, path); }
    }

    CUresult err = cuModuleLoadDataEx(module, ptx, 0, NULL, NULL);
    free(ptx);
    if (err != CUDA_SUCCESS) return -1;

    return sm;
}

#endif /* CUDA_RUNNER_COMMON_IMPLEMENTATION */
#endif /* CUDA_RUNNER_COMMON_H */
