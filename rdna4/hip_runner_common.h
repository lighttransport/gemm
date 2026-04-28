/*
 * hip_runner_common.h - Shared host-side utilities for HIP/ROCm runners
 *
 * Usage:
 *   #define HIP_RUNNER_COMMON_IMPLEMENTATION
 *   #include "../hip_runner_common.h"
 *
 * Provides: HIP_CHECK/HIP_CHECK_NULL macros, HIPRTC compile, F32→F16/FP8
 * conversion, raw GPU upload.
 *
 * Requires: rocew.h (dynamic loading of HIP/HIPRTC)
 */
#ifndef HIP_RUNNER_COMMON_H
#define HIP_RUNNER_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- Error checking macros (return -1 or NULL) ---- */

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        const char *es = "?"; \
        if (hipGetErrorString) hipGetErrorString(err, &es); \
        fprintf(stderr, "HIP error %s:%d: %s (%d)\n", __FILE__, __LINE__, es, (int)err); \
        return -1; \
    } \
} while(0)

#define HIP_CHECK_NULL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        const char *es = "?"; \
        if (hipGetErrorString) hipGetErrorString(err, &es); \
        fprintf(stderr, "HIP error %s:%d: %s (%d)\n", __FILE__, __LINE__, es, (int)err); \
        return NULL; \
    } \
} while(0)

/* ======================================================================== */
#ifdef HIP_RUNNER_COMMON_IMPLEMENTATION

/* ---- F32 → F16 conversion (truncation, no rounding) ---- */

static uint16_t hip_f32_to_f16(float f) {
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

static uint8_t hip_f32_to_fp8_e4m3(float f) {
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

static void *hip_upload_raw(const void *data, size_t bytes) {
    if (!data || bytes == 0) return NULL;
    void *d = NULL;
    if (hipMalloc(&d, bytes) != hipSuccess) return NULL;
    hipMemcpy(d, data, bytes, hipMemcpyHostToDevice);
    return d;
}

/* ---- HIPRTC kernel compilation ---- */

static int hip_compile_kernels(hipModule_t *module, int device_id,
                                const char *source, const char *prog_name,
                                int verbose, const char *prefix) {
    const char *arch = rocewGetRDNA4ArchString(device_id);
    if (!arch) {
        /* Fallback: get arch from device properties */
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, device_id) != hipSuccess) {
            fprintf(stderr, "%s: cannot query device properties\n", prefix);
            return -1;
        }
        arch = props.gcnArchName;
    }

    if (verbose >= 1)
        fprintf(stderr, "%s: compiling kernels for %s ...\n", prefix, arch);

    hiprtcProgram prog;
    hiprtcResult cres = hiprtcCreateProgram(&prog, source, prog_name, 0, NULL, NULL);
    if (cres != HIPRTC_SUCCESS) {
        fprintf(stderr, "%s: hiprtcCreateProgram failed: %d\n", prefix, (int)cres);
        return -1;
    }

    char arch_flag[64];
    snprintf(arch_flag, sizeof(arch_flag), "--gpu-architecture=%s", arch);
    const char *opts[] = { arch_flag, "-O3", "-ffast-math" };
    hiprtcResult nres = hiprtcCompileProgram(prog, 3, opts);

    if (nres != HIPRTC_SUCCESS) {
        fprintf(stderr, "%s: HIPRTC compile error %d\n", prefix, (int)nres);
        size_t log_sz;
        hiprtcGetProgramLogSize(prog, &log_sz);
        char *log = (char *)malloc(log_sz + 1);
        hiprtcGetProgramLog(prog, log);
        log[log_sz] = '\0';
        fprintf(stderr, "%s: HIPRTC log (%zu bytes):\n%s\n", prefix, log_sz, log);
        free(log);
        hiprtcDestroyProgram(&prog);
        return -1;
    }

    /* Get compiled code object */
    size_t code_sz = 0;
    hiprtcGetCodeSize(prog, &code_sz);
    char *code = (char *)malloc(code_sz);
    hiprtcGetCode(prog, code);
    hiprtcDestroyProgram(&prog);

    if (verbose >= 2)
        fprintf(stderr, "%s: loading code object (%zu bytes)\n", prefix, code_sz);

    if (verbose >= 3) {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/%s.co", prog_name);
        FILE *fp = fopen(path, "wb");
        if (fp) { fwrite(code, 1, code_sz, fp); fclose(fp);
            fprintf(stderr, "%s: code object saved to %s\n", prefix, path); }
    }

    hipError_t err = hipModuleLoadData(module, code);
    free(code);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: hipModuleLoadData failed: %d\n", prefix, (int)err);
        return -1;
    }

    /* Return a positive value to indicate success (arch-dependent) */
    return 1;
}

#endif /* HIP_RUNNER_COMMON_IMPLEMENTATION */
#endif /* HIP_RUNNER_COMMON_H */
