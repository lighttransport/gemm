#include "int4.h"
#include <limits.h>
#include <stdio.h>
#include <string.h>

const char *const int4_variant_names[INT4_VARIANTS] = {
    "gemm_ref", "gemm_s64_64", "gemm_a64_64", "gemm_s64_128", "gemm_a64_128",
    "gemm_s128_64", "gemm_a128_64", "gemm_s128_128", "gemm_a128_128"
};

int int4_padded_k(int k) { return k > 0 && k <= INT_MAX - 63 ? (k + 63) & ~63 : 0; }
size_t int4_codes_size(int rows, int k) {
    return rows > 0 ? (size_t)rows * int4_padded_k(k) / 2 : 0;
}
size_t int4_scales_size(int rows, int k) {
    return rows > 0 ? (size_t)rows * int4_padded_k(k) / 16 : 0;
}

int int4_quantize_cpu(const float *src, int rows, int k, uint8_t *codes,
                      uint8_t *scales, int format, int4_quant_stats *stats) {
    int kp = int4_padded_k(k);
    int4_quant_stats st = {0, 0};
    if (!src || !codes || !scales || rows <= 0 || !kp || (format != 0 && format != 1)) return -1;
    for (int r = 0; r < rows; r++) for (int g = 0; g < kp / 16; g++) {
        float amax = 0, limit = format ? 7.0f : 6.0f;
        for (int j = 0; j < 16 && g * 16 + j < k; j++) {
            float x = src[(size_t)r * k + g * 16 + j];
            if (!isfinite(x)) return -1;
            amax = fmaxf(amax, fabsf(x));
        }
        float desired = int4_div(amax, limit);
        unsigned sc = amax == 0 ? 0x38 : int4_scale_encode(desired);
        float scale = int4_scale_decode(sc);
        if (amax && (desired < 1.0f / 512.0f || desired > 448)) st.clamped_scales++;
        scales[(size_t)r * (kp / 16) + g] = (uint8_t)sc;
        for (int j = 0; j < 16; j += 2) {
            unsigned packed = 0;
            for (int q = 0; q < 2; q++) {
                int c = g * 16 + j + q;
                float x = c < k ? src[(size_t)r * k + c] : 0;
                if (int4_div(fabsf(x), scale) > limit) st.clipped_values++;
                packed |= int4_code_encode(x, scale, format) << (q * 4);
            }
            codes[(size_t)r * (kp / 2) + g * 8 + j / 2] = (uint8_t)packed;
        }
    }
    if (stats) *stats = st;
    return 0;
}

float int4_decode_cpu(const uint8_t *codes, const uint8_t *scales,
                      int row, int col, int kp, int format) {
    unsigned q = (codes[(size_t)row * (kp / 2) + col / 2] >> (4 * (col & 1))) & 15;
    return int4_code_decode(q, format) * int4_scale_decode(scales[(size_t)row * (kp / 16) + col / 16]);
}

void int4_destroy(int4_context *ctx) {
    if (!ctx) return;
    for (int i = 0; i < 4; i++) if (ctx->modules[i]) cuModuleUnload(ctx->modules[i]);
    memset(ctx, 0, sizeof(*ctx));
}

int int4_init(int4_context *ctx, const char *dir) {
    const char *names[] = {"e2m1_e2m1", "e0m3_e2m1", "e2m1_e0m3", "e0m3_e0m3"};
    CUdevice dev;
    int major = 0, minor = 0;
    if (!ctx || !dir) return -1;
    memset(ctx, 0, sizeof(*ctx));
    if (cuCtxGetDevice(&dev) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS ||
        major != 12 || minor != 0) {
        fprintf(stderr, "int4: this experimental CUBIN requires SM120\n"); return -1;
    }
    for (int f = 0; f < 4; f++) {
        char path[4096];
        int n = snprintf(path, sizeof(path), "%s/%s.cubin", dir, names[f]);
        if (n < 0 || n >= (int)sizeof(path) || cuModuleLoad(&ctx->modules[f], path) != CUDA_SUCCESS) goto fail;
        for (int v = 0; v < INT4_VARIANTS; v++)
            if (cuModuleGetFunction(&ctx->gemm[f][v], ctx->modules[f], int4_variant_names[v]) != CUDA_SUCCESS) goto fail;
        if (cuModuleGetFunction(&ctx->probe[f], ctx->modules[f], "probe_fp4") != CUDA_SUCCESS) goto fail;
    }
#define LOAD(field, name) if (cuModuleGetFunction(&ctx->field, ctx->modules[0], name) != CUDA_SUCCESS) goto fail
    LOAD(quant, "quantize_f32"); LOAD(dequant_bf16, "dequant_bf16"); LOAD(unpack_int8, "unpack_int8");
    LOAD(probe_bf16, "probe_bf16"); LOAD(probe_int8, "probe_int8"); LOAD(probe_s4, "probe_s4");
#undef LOAD
    return 0;
fail:
    fprintf(stderr, "int4: failed to load CUBINs from %s (run make)\n", dir);
    int4_destroy(ctx); return -1;
}

static int valid_matrix(int4_matrix a) {
    return a.rows > 0 && a.kp > 0 && a.kp == int4_padded_k(a.k) && a.codes && a.scales &&
        !(a.codes & 15) && !(a.scales & 3);
}

CUresult int4_quantize_f32(int4_context *ctx, CUdeviceptr src, int4_matrix dst,
                          int format, CUdeviceptr stats, CUstream stream) {
    if (!ctx || !ctx->quant || !valid_matrix(dst) || !src || (src & 3) || (stats & 7) ||
        (format != 0 && format != 1)) return CUDA_ERROR_INVALID_VALUE;
    size_t groups = (size_t)dst.rows * (dst.kp / 16);
    size_t blocks = (groups + 7) / 8;
    if (blocks > INT_MAX) return CUDA_ERROR_INVALID_VALUE;
    void *args[] = {&src, &dst.codes, &dst.scales, &dst.rows, &dst.k, &dst.kp, &format, &stats};
    return cuLaunchKernel(ctx->quant, (unsigned)blocks, 1, 1, 128, 1, 1, 0, stream, args, NULL);
}

CUresult int4_gemm(int4_context *ctx, int4_matrix a, int4_matrix b,
                   CUdeviceptr output, int formats, int variant, CUstream stream) {
    if (!ctx || !valid_matrix(a) || !valid_matrix(b) || a.k != b.k || !output || (output & 3) ||
        formats < 0 || formats > 3 || variant < 0 || variant >= INT4_VARIANTS) return CUDA_ERROR_INVALID_VALUE;
    if (!ctx->gemm[formats][variant]) return CUDA_ERROR_INVALID_VALUE;
    unsigned bx, by, threads;
    if (variant == 0) { bx = (b.rows + 7u) / 8u; by = (a.rows + 15u) / 16u; threads = 32; }
    else {
        int bn = variant <= 4 ? 64 : 128;
        bx = (b.rows + (unsigned)bn - 1) / (unsigned)bn; by = (a.rows + 63u) / 64u;
        threads = bn == 64 ? 128 : 256;
    }
    if (by > 65535 || bx > INT_MAX) return CUDA_ERROR_INVALID_VALUE;
    void *args[] = {&a.codes, &b.codes, &a.scales, &b.scales, &output, &a.rows, &b.rows, &a.kp};
    return cuLaunchKernel(ctx->gemm[formats][variant], bx, by, 1, threads, 1, 1, 0, stream, args, NULL);
}
