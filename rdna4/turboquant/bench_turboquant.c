/*
 * bench_turboquant.c - RDNA4 TurboQuant standalone kernels.
 *
 * HIPRTC kernels mirror cpu/turboquant's WHT/PolarQuant 3/4-bit block formats.
 * This is a standalone benchmark; it does not register GGUF types or touch the
 * existing LLM runner paths.
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../rocew.h"

#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include "../../cpu/turboquant/turboquant_cpu.h"

#define TQ3_THREADS 128

static const char *kernel_src =
"typedef unsigned char u8;\n"
"typedef unsigned short u16;\n"
"typedef unsigned int u32;\n"
"typedef unsigned long long u64;\n"
"#define TQ3_BLOCK_SIZE 128\n"
"#define TQ3_PACKED_BYTES 48\n"
"#define TQ4_PACKED_BYTES 64\n"
"__constant__ float tq3_centroids_dev[8] = {\n"
" -0.1900314689f, -0.1187858144f, -0.0668221228f, -0.0216554848f,\n"
"  0.0216554848f,  0.0668221228f,  0.1187858144f,  0.1900314689f };\n"
"__constant__ float tq3_thresholds_dev[7] = {\n"
" -0.1544086412f, -0.0928039686f, -0.0442388038f, 0.0f,\n"
"  0.0442388038f,  0.0928039686f,  0.1544086412f };\n"
"__constant__ float tq4_centroids_dev[16] = {\n"
" -0.2415290092f, -0.1828769682f, -0.1430164169f, -0.1110361778f,\n"
" -0.0832919158f, -0.0580498424f, -0.0342989423f, -0.0113486234f,\n"
"  0.0113486234f,  0.0342989423f,  0.0580498424f,  0.0832919158f,\n"
"  0.1110361778f,  0.1430164169f,  0.1828769682f,  0.2415290092f };\n"
"__constant__ float tq4_thresholds_dev[15] = {\n"
" -0.2122029887f, -0.1629466926f, -0.1270262974f, -0.0971640468f,\n"
" -0.0706708791f, -0.0461743923f, -0.0228237828f,  0.0000000000f,\n"
"  0.0228237828f,  0.0461743923f,  0.0706708791f,  0.0971640468f,\n"
"  0.1270262974f,  0.1629466926f,  0.2122029887f };\n"
"\n"
"__device__ __forceinline__ u64 splitmix64(u64 *x) {\n"
"    u64 z = (*x += 0x9e3779b97f4a7c15ull);\n"
"    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;\n"
"    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;\n"
"    return z ^ (z >> 31);\n"
"}\n"
"__device__ __forceinline__ int sign_bit(u64 seed, u64 block_index, int plane, int i) {\n"
"    u64 s = seed ^ (0xd1b54a32d192ed03ull * (block_index + 1));\n"
"    s ^= 0x9e3779b97f4a7c15ull * (u64)(plane + 17);\n"
"    s ^= (u64)i * 0xbf58476d1ce4e5b9ull;\n"
"    return (int)(splitmix64(&s) >> 63);\n"
"}\n"
"__device__ __forceinline__ float f16_to_f32(u16 h) {\n"
"    u32 sign = ((u32)h & 0x8000u) << 16;\n"
"    u32 exp = ((u32)h >> 10) & 0x1fu;\n"
"    u32 mant = (u32)h & 0x03ffu;\n"
"    u32 f;\n"
"    if (exp == 0) {\n"
"        if (mant == 0) f = sign;\n"
"        else {\n"
"            exp = 1;\n"
"            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }\n"
"            mant &= 0x03ffu;\n"
"            f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);\n"
"        }\n"
"    } else if (exp == 31) f = sign | 0x7f800000u | (mant << 13);\n"
"    else f = sign | ((exp + 127u - 15u) << 23) | (mant << 13);\n"
"    return __uint_as_float(f);\n"
"}\n"
"__device__ __forceinline__ u16 f32_to_f16(float f) {\n"
"    u32 x = __float_as_uint(f);\n"
"    u16 sign = (u16)((x >> 16) & 0x8000u);\n"
"    int exp = (int)((x >> 23) & 0xffu) - 127;\n"
"    u32 mant = x & 0x7fffffu;\n"
"    if (exp > 15) return (u16)(sign | 0x7c00u);\n"
"    if (exp < -14) {\n"
"        if (exp < -24) return sign;\n"
"        mant |= 0x800000u;\n"
"        mant >>= (u32)(-1 - exp);\n"
"        return (u16)(sign | (mant >> 13));\n"
"    }\n"
"    return (u16)(sign | ((u16)(exp + 15) << 10) | (u16)(mant >> 13));\n"
"}\n"
"__device__ __forceinline__ u16 f32_to_bf16(float f) {\n"
"    u32 u = __float_as_uint(f);\n"
"    u32 rounded = u + ((u >> 16) & 1u) + 0x7fffu;\n"
"    return (u16)(rounded >> 16);\n"
"}\n"
"__device__ __forceinline__ u8 f32_to_fp8_e4m3(float f) {\n"
"    if (f == 0.0f) return 0;\n"
"    u32 bits = __float_as_uint(f);\n"
"    u32 sign = (bits >> 31) & 1u;\n"
"    int exp = (int)((bits >> 23) & 0xffu) - 127 + 7;\n"
"    u32 mant = (bits >> 20) & 7u;\n"
"    if (exp >= 15) { exp = 15; mant = 6; }\n"
"    if (exp <= 0) return (u8)(sign << 7);\n"
"    return (u8)((sign << 7) | (((u32)exp & 15u) << 3) | (mant & 7u));\n"
"}\n"
"__device__ __forceinline__ float centroid(int bits, int idx) {\n"
"    return bits == 4 ? tq4_centroids_dev[idx] : tq3_centroids_dev[idx];\n"
"}\n"
"__device__ __forceinline__ int nearest_centroid(float v, int bits) {\n"
"    int idx = 0;\n"
"    if (bits == 4) { for (int i = 0; i < 15; i++) idx += v > tq4_thresholds_dev[i]; return idx; }\n"
"    idx += v > tq3_thresholds_dev[0]; idx += v > tq3_thresholds_dev[1];\n"
"    idx += v > tq3_thresholds_dev[2]; idx += v > tq3_thresholds_dev[3];\n"
"    idx += v > tq3_thresholds_dev[4]; idx += v > tq3_thresholds_dev[5];\n"
"    idx += v > tq3_thresholds_dev[6]; return idx;\n"
"}\n"
"__device__ __forceinline__ int block_bytes(int bits) { return bits == 4 ? 68 : 52; }\n"
"__device__ __forceinline__ int unpack_idx(const u8 *qs, int i, int bits) {\n"
"    if (bits == 4) { u8 q = qs[i >> 1]; return (i & 1) ? (q >> 4) : (q & 15); }\n"
"    int bit = i * 3, byte = bit >> 3, shift = bit & 7;\n"
"    u32 w = qs[byte]; if (byte + 1 < TQ3_PACKED_BYTES) w |= ((u32)qs[byte + 1] << 8);\n"
"    return (int)((w >> shift) & 7u);\n"
"}\n"
"__device__ __forceinline__ void pack_idx(u8 *qs, const u8 *idx, int bits) {\n"
"    int packed = bits == 4 ? TQ4_PACKED_BYTES : TQ3_PACKED_BYTES;\n"
"    for (int i = 0; i < packed; i++) qs[i] = 0;\n"
"    if (bits == 4) { for (int i = 0; i < TQ3_BLOCK_SIZE; i += 2) qs[i >> 1] = (idx[i] & 15u) | ((idx[i + 1] & 15u) << 4); return; }\n"
"    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {\n"
"        int bit = i * 3, byte = bit >> 3, shift = bit & 7;\n"
"        u32 w = ((u32)idx[i] & 7u) << shift;\n"
"        qs[byte] |= (u8)w;\n"
"        if (shift > 5) qs[byte + 1] |= (u8)(w >> 8);\n"
"    }\n"
"}\n"
"__device__ __forceinline__ void fwht128_shared(float *v, int tid) {\n"
"    for (int h = 1; h < TQ3_BLOCK_SIZE; h <<= 1) {\n"
"        if (tid < 64) {\n"
"            int j = tid & (h - 1); int base = (tid >> (__ffs(h) - 1)) * (h << 1);\n"
"            int i0 = base + j, i1 = i0 + h;\n"
"            float a = v[i0], b = v[i1];\n"
"            v[i0] = a + b; v[i1] = a - b;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    v[tid] *= 0.08838834764831845f;\n"
"    __syncthreads();\n"
"}\n"
"\n"
"extern \"C\" {\n"
"__global__ __launch_bounds__(128, 1)\n"
"void tq3_quantize_f32(u8 *dst, const float *src, int rows, int cols, u64 seed, int bits) {\n"
"    int nb = cols >> 7; int g = blockIdx.x; int row = g / nb; int cb = g - row * nb;\n"
"    int tid = threadIdx.x; if (row >= rows) return;\n"
"    __shared__ float v[128]; __shared__ float red[128]; __shared__ u8 idx[128];\n"
"    const float *x = src + (long)row * cols + cb * 128;\n"
"    float xv = x[tid]; red[tid] = xv * xv; __syncthreads();\n"
"    for (int s = 64; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }\n"
"    float norm2 = red[0];\n"
"    if (norm2 <= 0.0f) { if (tid == 0) { u8 *o = dst + (long)g * block_bytes(bits); for (int i=0;i<block_bytes(bits);i++) o[i]=0; } return; }\n"
"    float invn = 1.0f / sqrtf(norm2);\n"
"    v[tid] = (sign_bit(seed, (u64)cb, 0, tid) ? -xv : xv) * invn; __syncthreads();\n"
"    fwht128_shared(v, tid);\n"
"    float rv = sign_bit(seed, (u64)cb, 1, tid) ? -v[tid] : v[tid];\n"
"    int qi = nearest_centroid(rv, bits); idx[tid] = (u8)qi; red[tid] = centroid(bits, qi) * centroid(bits, qi); __syncthreads();\n"
"    for (int s = 64; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }\n"
"    if (tid == 0) { u8 *o = dst + (long)g * block_bytes(bits); float scale = sqrtf(norm2) / fmaxf(sqrtf(red[0]), 1.0e-20f); u16 h = f32_to_f16(scale); o[0]=(u8)h; o[1]=(u8)(h>>8); o[2]=0; o[3]=0; pack_idx(o+4, idx, bits); }\n"
"}\n"
"\n"
"__global__ __launch_bounds__(128, 1)\n"
"void tq3_dequant_f32(float *dst, const u8 *src, int rows, int cols, u64 seed, int bits) {\n"
"    int nb = cols >> 7; int g = blockIdx.x; int row = g / nb; int cb = g - row * nb;\n"
"    int tid = threadIdx.x; if (row >= rows) return;\n"
"    __shared__ float v[128]; const u8 *blk = src + (long)g * block_bytes(bits);\n"
"    u16 h = (u16)blk[0] | ((u16)blk[1] << 8); float scale = f16_to_f32(h);\n"
"    int qi = unpack_idx(blk + 4, tid, bits);\n"
"    float y = centroid(bits, qi) * scale;\n"
"    v[tid] = sign_bit(seed, (u64)cb, 1, tid) ? -y : y; __syncthreads();\n"
"    fwht128_shared(v, tid);\n"
"    float out = sign_bit(seed, (u64)cb, 0, tid) ? -v[tid] : v[tid];\n"
"    dst[(long)row * cols + cb * 128 + tid] = out;\n"
"}\n"
"\n"
"__global__ __launch_bounds__(128, 1)\n"
"void tq3_dot_partial(float *partial, const u8 *src, const float *x, int rows, int cols, u64 seed, int bits) {\n"
"    int nb = cols >> 7; int g = blockIdx.x; int row = g / nb; int cb = g - row * nb;\n"
"    int tid = threadIdx.x; if (row >= rows) return;\n"
"    __shared__ float v[128]; __shared__ float red[128]; const u8 *blk = src + (long)g * block_bytes(bits);\n"
"    float xv = x[cb * 128 + tid]; v[tid] = sign_bit(seed, (u64)cb, 0, tid) ? -xv : xv; __syncthreads();\n"
"    fwht128_shared(v, tid);\n"
"    float xrot = sign_bit(seed, (u64)cb, 1, tid) ? -v[tid] : v[tid];\n"
"    u16 h = (u16)blk[0] | ((u16)blk[1] << 8); float scale = f16_to_f32(h);\n"
"    int qi = unpack_idx(blk + 4, tid, bits); red[tid] = centroid(bits, qi) * scale * xrot; __syncthreads();\n"
"    for (int s = 64; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }\n"
"    if (tid == 0) partial[g] = red[0];\n"
"}\n"
"\n"
"__global__ __launch_bounds__(128, 1)\n"
"void tq3_dequant_bf16(u16 *dst, const u8 *src, int rows, int cols, u64 seed, int bits) {\n"
"    int nb = cols >> 7; int g = blockIdx.x; int row = g / nb; int cb = g - row * nb; int tid = threadIdx.x; if (row >= rows) return;\n"
"    __shared__ float v[128]; const u8 *blk = src + (long)g * block_bytes(bits); u16 h = (u16)blk[0] | ((u16)blk[1] << 8); float scale = f16_to_f32(h);\n"
"    int qi = unpack_idx(blk + 4, tid, bits); float y = centroid(bits, qi) * scale; v[tid] = sign_bit(seed, (u64)cb, 1, tid) ? -y : y; __syncthreads(); fwht128_shared(v, tid);\n"
"    float out = sign_bit(seed, (u64)cb, 0, tid) ? -v[tid] : v[tid]; dst[(long)row * cols + cb * 128 + tid] = f32_to_bf16(out);\n"
"}\n"
"__global__ __launch_bounds__(128, 1)\n"
"void tq3_dequant_f16(u16 *dst, const u8 *src, int rows, int cols, u64 seed, int bits) {\n"
"    int nb = cols >> 7; int g = blockIdx.x; int row = g / nb; int cb = g - row * nb; int tid = threadIdx.x; if (row >= rows) return;\n"
"    __shared__ float v[128]; const u8 *blk = src + (long)g * block_bytes(bits); u16 h = (u16)blk[0] | ((u16)blk[1] << 8); float scale = f16_to_f32(h);\n"
"    int qi = unpack_idx(blk + 4, tid, bits); float y = centroid(bits, qi) * scale; v[tid] = sign_bit(seed, (u64)cb, 1, tid) ? -y : y; __syncthreads(); fwht128_shared(v, tid);\n"
"    float out = sign_bit(seed, (u64)cb, 0, tid) ? -v[tid] : v[tid]; dst[(long)row * cols + cb * 128 + tid] = f32_to_f16(out);\n"
"}\n"
"__global__ __launch_bounds__(128, 1)\n"
"void tq3_dequant_fp8(u8 *dst, const u8 *src, int rows, int cols, u64 seed, int bits) {\n"
"    int nb = cols >> 7; int g = blockIdx.x; int row = g / nb; int cb = g - row * nb; int tid = threadIdx.x; if (row >= rows) return;\n"
"    __shared__ float v[128]; const u8 *blk = src + (long)g * block_bytes(bits); u16 h = (u16)blk[0] | ((u16)blk[1] << 8); float scale = f16_to_f32(h);\n"
"    int qi = unpack_idx(blk + 4, tid, bits); float y = centroid(bits, qi) * scale; v[tid] = sign_bit(seed, (u64)cb, 1, tid) ? -y : y; __syncthreads(); fwht128_shared(v, tid);\n"
"    float out = sign_bit(seed, (u64)cb, 0, tid) ? -v[tid] : v[tid]; dst[(long)row * cols + cb * 128 + tid] = f32_to_fp8_e4m3(out);\n"
"}\n"
"}\n";

static void die_hip(hipError_t err, const char *expr, const char *file, int line) {
    if (err == hipSuccess) return;
    const char *s = "?";
    if (hipGetErrorString) hipGetErrorString(err, &s);
    fprintf(stderr, "HIP error %s:%d: %s failed: %s (%d)\n", file, line, expr, s, err);
    exit(1);
}

#define CHECK_HIP(expr) die_hip((expr), #expr, __FILE__, __LINE__)

static uint64_t splitmix64_host(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static float rand_f32(uint64_t *s) {
    uint32_t v = (uint32_t)(splitmix64_host(s) >> 40);
    return ((float)v / 16777216.0f) * 2.0f - 1.0f;
}

static void fill(float *x, int n, uint64_t seed) {
    uint64_t s = seed;
    for (int i = 0; i < n; i++) x[i] = rand_f32(&s);
}

static float elapsed_kernel_ms(hipFunction_t fn, unsigned gx, unsigned bx,
                               void **args, int iters) {
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipEventRecord(start, NULL));
    for (int i = 0; i < iters; i++) {
        CHECK_HIP(hipModuleLaunchKernel(fn, gx, 1, 1, bx, 1, 1, 0, NULL, args, NULL));
    }
    CHECK_HIP(hipEventRecord(stop, NULL));
    CHECK_HIP(hipEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    return ms / (float)iters;
}

int main(int argc, char **argv) {
    int rows = 1024, cols = 128, iters = 200, device = 0, verify = 1, verbose = 0, bits = 3;
    uint64_t seed = 42;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bits") && i + 1 < argc) bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--bench-only")) verify = 0;
        else if (!strcmp(argv[i], "--verify")) verify = 1;
        else if (!strcmp(argv[i], "--verbose")) verbose++;
    }
    if (rows <= 0 || cols <= 0 || (cols % TQ3_BLOCK_SIZE) != 0) {
        fprintf(stderr, "rows must be positive and cols must be a positive multiple of %d\n", TQ3_BLOCK_SIZE);
        return 2;
    }
    if (bits != 3 && bits != 4) {
        fprintf(stderr, "--bits must be 3 or 4\n");
        return 2;
    }

    int nb = cols / TQ3_BLOCK_SIZE;
    size_t elems = (size_t)rows * (size_t)cols;
    size_t q_row_bytes = bits == 4 ? tq4_row_bytes(cols) : tq3_row_bytes(cols);
    size_t q_bytes = (size_t)rows * q_row_bytes;
    size_t partial_count = (size_t)rows * (size_t)nb;

    float *h_x = (float *)malloc(elems * sizeof(float));
    float *h_query = (float *)malloc((size_t)cols * sizeof(float));
    float *h_deq = (float *)malloc(elems * sizeof(float));
    float *h_gpu_deq = (float *)malloc(elems * sizeof(float));
    float *h_partial = (float *)malloc(partial_count * sizeof(float));
    uint8_t *h_q_cpu = (uint8_t *)malloc(q_bytes);
    uint8_t *h_q_gpu = (uint8_t *)malloc(q_bytes);
    if (!h_x || !h_query || !h_deq || !h_gpu_deq || !h_partial || !h_q_cpu || !h_q_gpu) return 2;
    fill(h_x, (int)elems, seed + 1);
    fill(h_query, cols, seed + 2);
    for (int r = 0; r < rows; r++) {
        if (bits == 4) {
            tq4_quantize_row_f32((tq4_block *)(h_q_cpu + (size_t)r * q_row_bytes),
                                 h_x + (size_t)r * cols, cols, seed);
            tq4_dequantize_row_f32(h_deq + (size_t)r * cols,
                                   (const tq4_block *)(h_q_cpu + (size_t)r * q_row_bytes),
                                   cols, seed);
        } else {
            tq3_quantize_row_f32((tq3_block *)(h_q_cpu + (size_t)r * q_row_bytes),
                                 h_x + (size_t)r * cols, cols, seed);
            tq3_dequantize_row_f32(h_deq + (size_t)r * cols,
                                   (const tq3_block *)(h_q_cpu + (size_t)r * q_row_bytes),
                                   cols, seed);
        }
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "rocewInit failed; HIP/HIPRTC not available\n");
        return 2;
    }
    CHECK_HIP(hipSetDevice(device));
    hipModule_t module;
    if (hip_compile_kernels(&module, device, kernel_src, "tq3_kernels", verbose, "tq3") < 0) return 2;
    hipFunction_t fn_quant, fn_deq, fn_dot, fn_bf16, fn_f16, fn_fp8;
    CHECK_HIP(hipModuleGetFunction(&fn_quant, module, "tq3_quantize_f32"));
    CHECK_HIP(hipModuleGetFunction(&fn_deq, module, "tq3_dequant_f32"));
    CHECK_HIP(hipModuleGetFunction(&fn_dot, module, "tq3_dot_partial"));
    CHECK_HIP(hipModuleGetFunction(&fn_bf16, module, "tq3_dequant_bf16"));
    CHECK_HIP(hipModuleGetFunction(&fn_f16, module, "tq3_dequant_f16"));
    CHECK_HIP(hipModuleGetFunction(&fn_fp8, module, "tq3_dequant_fp8"));

    void *d_x = NULL, *d_q = NULL, *d_deq = NULL, *d_query = NULL, *d_partial = NULL, *d_stage = NULL;
    CHECK_HIP(hipMalloc(&d_x, elems * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_q, q_bytes));
    CHECK_HIP(hipMalloc(&d_deq, elems * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_query, (size_t)cols * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_partial, partial_count * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_stage, elems * sizeof(uint16_t)));
    CHECK_HIP(hipMemcpy(d_x, h_x, elems * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_query, h_query, (size_t)cols * sizeof(float), hipMemcpyHostToDevice));

    unsigned gx = (unsigned)(rows * nb);
    void *qargs[] = { &d_q, &d_x, &rows, &cols, &seed, &bits };
    void *dargs[] = { &d_deq, &d_q, &rows, &cols, &seed, &bits };
    void *dotargs[] = { &d_partial, &d_q, &d_query, &rows, &cols, &seed, &bits };
    void *stageargs[] = { &d_stage, &d_q, &rows, &cols, &seed, &bits };

    CHECK_HIP(hipModuleLaunchKernel(fn_quant, gx, 1, 1, TQ3_THREADS, 1, 1, 0, NULL, qargs, NULL));
    CHECK_HIP(hipDeviceSynchronize());

    if (verify) {
        CHECK_HIP(hipMemcpy(h_q_gpu, d_q, q_bytes, hipMemcpyDeviceToHost));
        size_t mismatches = 0;
        for (size_t i = 0; i < q_bytes; i++) mismatches += h_q_cpu[i] != h_q_gpu[i];
        CHECK_HIP(hipModuleLaunchKernel(fn_deq, gx, 1, 1, TQ3_THREADS, 1, 1, 0, NULL, dargs, NULL));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_gpu_deq, d_deq, elems * sizeof(float), hipMemcpyDeviceToHost));
        double max_abs = 0.0, mean_abs = 0.0;
        for (size_t i = 0; i < elems; i++) {
            double e = fabs((double)h_deq[i] - (double)h_gpu_deq[i]);
            if (e > max_abs) max_abs = e;
            mean_abs += e;
        }
        mean_abs /= (double)elems;
        CHECK_HIP(hipModuleLaunchKernel(fn_dot, gx, 1, 1, TQ3_THREADS, 1, 1, 0, NULL, dotargs, NULL));
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_partial, d_partial, partial_count * sizeof(float), hipMemcpyDeviceToHost));
        double dot_max_abs = 0.0;
        for (int r = 0; r < rows; r++) {
            double gd = 0.0;
            for (int b = 0; b < nb; b++) gd += h_partial[(size_t)r * nb + b];
            float cd = bits == 4
                ? tq4_dot_row_f32((const tq4_block *)(h_q_cpu + (size_t)r * q_row_bytes),
                                  h_query, cols, seed)
                : tq3_dot_row_f32((const tq3_block *)(h_q_cpu + (size_t)r * q_row_bytes),
                                  h_query, cols, seed);
            double e = fabs(gd - (double)cd);
            if (e > dot_max_abs) dot_max_abs = e;
        }
        printf("verify q_mismatches=%zu deq_max_abs=%.6g deq_mean_abs=%.6g dot_max_abs=%.6g\n",
               mismatches, max_abs, mean_abs, dot_max_abs);
        if (max_abs > 1.0e-3 || dot_max_abs > 5.0e-3 * (double)nb) {
            fprintf(stderr, "verification failed\n");
            return 1;
        }
    }

    float ms_q = elapsed_kernel_ms(fn_quant, gx, TQ3_THREADS, qargs, iters);
    float ms_d = elapsed_kernel_ms(fn_deq, gx, TQ3_THREADS, dargs, iters);
    float ms_dot = elapsed_kernel_ms(fn_dot, gx, TQ3_THREADS, dotargs, iters);
    float ms_bf16 = elapsed_kernel_ms(fn_bf16, gx, TQ3_THREADS, stageargs, iters);
    float ms_f16 = elapsed_kernel_ms(fn_f16, gx, TQ3_THREADS, stageargs, iters);
    CHECK_HIP(hipFree(d_stage));
    CHECK_HIP(hipMalloc(&d_stage, elems * sizeof(uint8_t)));
    stageargs[0] = &d_stage;
    float ms_fp8 = elapsed_kernel_ms(fn_fp8, gx, TQ3_THREADS, stageargs, iters);

    double gb_f32 = (double)elems * sizeof(float) / 1.0e6;
    printf("tq%d rows=%d cols=%d blocks=%u bytes=%zu compression_vs_f16=%.3fx\n",
           bits, rows, cols, gx, q_bytes, ((double)elems * 2.0) / (double)q_bytes);
    printf("quant %.3f ms %.3f GB/s | dequant %.3f ms %.3f GB/s | dot %.3f ms %.3f GB/s\n",
           ms_q, gb_f32 / ms_q, ms_d, gb_f32 / ms_d, ms_dot, gb_f32 / ms_dot);
    printf("stage bf16 %.3f ms | f16 %.3f ms | fp8 %.3f ms\n", ms_bf16, ms_f16, ms_fp8);

    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_q));
    CHECK_HIP(hipFree(d_deq));
    CHECK_HIP(hipFree(d_query));
    CHECK_HIP(hipFree(d_partial));
    CHECK_HIP(hipFree(d_stage));
    hipModuleUnload(module);
    free(h_x); free(h_query); free(h_deq); free(h_gpu_deq); free(h_partial); free(h_q_cpu); free(h_q_gpu);
    return 0;
}
