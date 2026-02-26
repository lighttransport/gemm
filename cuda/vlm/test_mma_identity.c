/*
 * test_mma_identity.c - Test MMA with identity matrix to understand fragment mapping
 *
 * If W = I (identity), then Y = X * I^T = X.
 * Any difference reveals the fragment mapping on sm_120.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../cuew.h"

/* F32→F16 conversion */
static uint16_t f32_to_f16(float f) {
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

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { union { uint32_t i; float f; } u; u.i = sign; return u.f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        union { uint32_t i; float f; } u;
        u.i = sign | 0x7F800000 | (mant << 13);
        return u.f;
    }
    union { uint32_t i; float f; } u;
    u.i = sign | ((exp + 112) << 23) | (mant << 13);
    return u.f;
}

/* Minimal kernel: just one MMA m16n8k16 with hardcoded identity B */
static const char *test_kernel_src =
"typedef unsigned short half_raw;\n"
"\n"
"extern \"C\" {\n"
"\n"
"/* Single MMA: Y[16,8] = X[16,16] * W^T[8,16] */\n"
"/* W is identity-like: W[n][k] = (n==k) ? 1.0 : 0.0 for n<8, k<16 */\n"
"__global__ void mma_test_single(float *Y, const half_raw *W, const float *X,\n"
"                                  int n_out, int n_in) {\n"
"    /* Single warp, single MMA */\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4;\n"   /* groupID 0..7 */
"    int tid4 = lane % 4;\n"  /* threadID_in_group 0..3 */
"\n"
"    /* Load A fragment from X[16,16] via shared memory */\n"
"    __shared__ float smem[16 * 16];\n"
"    /* Each of 32 threads loads some elements */\n"
"    int tid = threadIdx.x;\n"
"    for (int i = tid; i < 16 * 16; i += blockDim.x)\n"
"        smem[i] = X[i];\n"
"    __syncthreads();\n"
"\n"
"    /* Convert X to F16 for A fragment */\n"
"    unsigned int a0, a1, a2, a3;\n"
"    { float f0 = smem[gid * 16 + tid4 * 2];\n"
"      float f1 = smem[gid * 16 + tid4 * 2 + 1];\n"
"      asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a0) : \"f\"(f0), \"f\"(f1)); }\n"
"    { float f0 = smem[gid * 16 + tid4 * 2 + 8];\n"
"      float f1 = smem[gid * 16 + tid4 * 2 + 9];\n"
"      asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a1) : \"f\"(f0), \"f\"(f1)); }\n"
"    { float f0 = smem[(gid + 8) * 16 + tid4 * 2];\n"
"      float f1 = smem[(gid + 8) * 16 + tid4 * 2 + 1];\n"
"      asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a2) : \"f\"(f0), \"f\"(f1)); }\n"
"    { float f0 = smem[(gid + 8) * 16 + tid4 * 2 + 8];\n"
"      float f1 = smem[(gid + 8) * 16 + tid4 * 2 + 9];\n"
"      asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a3) : \"f\"(f0), \"f\"(f1)); }\n"
"\n"
"    /* Load B fragment from W[8,16] */\n"
"    /* For identity: W[n][k] = (n==k) ? 1.0 : 0.0 */\n"
"    unsigned int b0, b1;\n"
"    int bc = gid; /* output channel 0..7 */\n"
"    const half_raw *wp = W + bc * n_in;\n"
"    b0 = *(const unsigned int *)(wp + tid4 * 2);\n"
"    b1 = *(const unsigned int *)(wp + tid4 * 2 + 8);\n"
"\n"
"    /* MMA */\n"
"    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;\n"
"    asm volatile(\n"
"        \"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"        \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"        : \"=f\"(d0), \"=f\"(d1), \"=f\"(d2), \"=f\"(d3)\n"
"        : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"          \"r\"(b0), \"r\"(b1),\n"
"          \"f\"(d0), \"f\"(d1), \"f\"(d2), \"f\"(d3)\n"
"    );\n"
"\n"
"    /* Store result */\n"
"    int yr0 = gid;      /* row 0..7 */\n"
"    int yr1 = gid + 8;  /* row 8..15 */\n"
"    int yc0 = tid4 * 2;\n"
"    int yc1 = yc0 + 1;\n"
"    if (yr0 < 16 && yc0 < n_out) Y[yr0 * n_out + yc0] = d0;\n"
"    if (yr0 < 16 && yc1 < n_out) Y[yr0 * n_out + yc1] = d1;\n"
"    if (yr1 < 16 && yc0 < n_out) Y[yr1 * n_out + yc0] = d2;\n"
"    if (yr1 < 16 && yc1 < n_out) Y[yr1 * n_out + yc1] = d3;\n"
"}\n"
"\n"
"} /* extern C */\n";

int main(void) {
    int M = 16, N = 8, K = 16;
    printf("=== MMA Identity Test (M=%d, N=%d, K=%d) ===\n", M, N, K);
    printf("Y[16,8] = X[16,16] * I^T[8,16]\n");
    printf("Expected: Y[i][j] = X[i][j] for j<8\n\n");

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuew init failed\n");
        return 1;
    }
    cuInit(0);

    CUdevice device;
    CUcontext ctx;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&ctx, 0, device);

    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    printf("GPU: sm_%d%d\n", major, minor);

    /* Compile */
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, test_kernel_src, "test_id.cu", 0, NULL, NULL);
    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
    const char *opts[] = { arch };
    nvrtcResult nres = nvrtcCompileProgram(prog, 1, opts);
    if (nres != NVRTC_SUCCESS) {
        size_t log_sz;
        nvrtcGetProgramLogSize(prog, &log_sz);
        char *log = (char *)malloc(log_sz);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "NVRTC error:\n%s\n", log);
        free(log);
        return 1;
    }

    size_t ptx_sz;
    nvrtcGetPTXSize(prog, &ptx_sz);
    char *ptx = (char *)malloc(ptx_sz);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUmodule module;
    cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
    free(ptx);

    CUfunction fn;
    cuModuleGetFunction(&fn, module, "mma_test_single");

    /* Create X[16,16]: X[i][j] = i * 16 + j + 1 (so each element is unique) */
    float h_X[16 * 16];
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            h_X[i * 16 + j] = (float)(i * 16 + j + 1);

    /* Create W[8,16] = identity: W[n][k] = (n==k) ? 1.0 : 0.0 */
    uint16_t h_W[8 * 16];
    memset(h_W, 0, sizeof(h_W));
    for (int n = 0; n < 8; n++)
        h_W[n * 16 + n] = f32_to_f16(1.0f);

    /* Expected: Y[i][j] = X[i][j] for j<8 */
    float h_Y_expected[16 * 8];
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++) {
            /* Y[i][j] = sum_k X_f16[i][k] * W_f16[j][k] */
            /* With identity W, Y[i][j] = X_f16[i][j] */
            h_Y_expected[i * 8 + j] = f16_to_f32(f32_to_f16(h_X[i * 16 + j]));
        }

    /* Upload */
    CUdeviceptr d_X, d_W, d_Y;
    cuMemAlloc(&d_X, 16 * 16 * sizeof(float));
    cuMemAlloc(&d_W, 8 * 16 * sizeof(uint16_t));
    cuMemAlloc(&d_Y, 16 * 8 * sizeof(float));
    cuMemcpyHtoD(d_X, h_X, 16 * 16 * sizeof(float));
    cuMemcpyHtoD(d_W, h_W, 8 * 16 * sizeof(uint16_t));
    cuMemsetD32(d_Y, 0, 16 * 8);

    /* Launch with 32 threads (1 warp) */
    int n_out = N, n_in = K;
    void *args[] = { &d_Y, &d_W, &d_X, &n_out, &n_in };
    cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1,
                   16 * 16 * sizeof(float), 0, args, NULL);
    cuCtxSynchronize();

    /* Download */
    float h_Y[16 * 8];
    cuMemcpyDtoH(h_Y, d_Y, 16 * 8 * sizeof(float));

    /* Print full output */
    printf("\nExpected Y = X[:, 0:8] (rounded to F16):\n");
    for (int i = 0; i < 16; i++) {
        printf("  row %2d: ", i);
        for (int j = 0; j < 8; j++)
            printf("%7.1f", h_Y_expected[i * 8 + j]);
        printf("\n");
    }

    printf("\nActual GPU Y:\n");
    for (int i = 0; i < 16; i++) {
        printf("  row %2d: ", i);
        for (int j = 0; j < 8; j++)
            printf("%7.1f", h_Y[i * 8 + j]);
        printf("\n");
    }

    /* Check */
    int errors = 0;
    for (int i = 0; i < 16 * 8; i++) {
        if (fabsf(h_Y[i] - h_Y_expected[i]) > 0.01f) {
            if (errors < 5)
                printf("MISMATCH at [%d][%d]: expected=%.1f got=%.1f\n",
                       i / 8, i % 8, h_Y_expected[i], h_Y[i]);
            errors++;
        }
    }
    printf("\nErrors: %d/%d\n", errors, 16 * 8);

    /* Cleanup */
    cuMemFree(d_X); cuMemFree(d_W); cuMemFree(d_Y);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    return errors > 0 ? 1 : 0;
}
