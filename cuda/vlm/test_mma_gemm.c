/*
 * test_mma_gemm.c - Standalone test for gemm_f16_f32 MMA kernel on Blackwell
 *
 * Tests a small GEMM: Y[M,N] = X[M,K] * W^T[N,K] using the shared
 * gemm_f16_f32 MMA m16n8k16 kernel, comparing against a simple CPU reference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../cuew.h"
#include "../cuda_kernels_common.h"

/* We need our own F32→F16 conversion (same as cuda_runner_common.h) */
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

/* F16→F32 */
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

/* Also need gemm_f32_f32 kernel for comparison */
static const char *test_kernel_src =
"\n"
"typedef unsigned short half_raw;\n"
"__device__ __forceinline__ float half_to_float(half_raw h) {\n"
"    float f; asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h)); return f;\n"
"}\n"
"\n"
"extern \"C\" {\n"
"\n"
"/* gemm_f16_f32: MMA m16n8k16 - from cuda_kernels_common.h */\n"
"#define GEMM_N_TILE 8\n"
"__global__ void gemm_f16_f32(float *Y, const half_raw *W, const float *X,\n"
"                              const float *bias,\n"
"                              int n_out, int n_in, int n_tok) {\n"
"    extern __shared__ float smem_x[];\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int out_base = blockIdx.x * 256 + warp_id * 64;\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4;\n"
"    int tid4 = lane % 4;\n"
"    int tid = threadIdx.x;\n"
"\n"
"    if (tok_base >= n_tok) return;\n"
"\n"
"    float d0[GEMM_N_TILE], d1[GEMM_N_TILE], d2[GEMM_N_TILE], d3[GEMM_N_TILE];\n"
"#pragma unroll\n"
"    for (int i = 0; i < GEMM_N_TILE; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"\n"
"    for (int k = 0; k < n_in; k += 16) {\n"
"        int srow = tid / 8, scol = (tid % 8) * 2;\n"
"        int grow = tok_base + srow;\n"
"        if (grow < n_tok) {\n"
"            smem_x[srow * 16 + scol] = X[grow * n_in + k + scol];\n"
"            smem_x[srow * 16 + scol + 1] = X[grow * n_in + k + scol + 1];\n"
"        } else {\n"
"            smem_x[srow * 16 + scol] = 0.0f;\n"
"            smem_x[srow * 16 + scol + 1] = 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        unsigned int a0, a1, a2, a3;\n"
"        { float f0 = smem_x[gid * 16 + tid4 * 2];\n"
"          float f1 = smem_x[gid * 16 + tid4 * 2 + 1];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a0) : \"f\"(f0), \"f\"(f1)); }\n"
"#if __CUDA_ARCH__ >= 1200\n"
"        { float f0 = smem_x[(gid + 8) * 16 + tid4 * 2];\n"
"          float f1 = smem_x[(gid + 8) * 16 + tid4 * 2 + 1];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a1) : \"f\"(f0), \"f\"(f1)); }\n"
"        { float f0 = smem_x[gid * 16 + tid4 * 2 + 8];\n"
"          float f1 = smem_x[gid * 16 + tid4 * 2 + 9];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a2) : \"f\"(f0), \"f\"(f1)); }\n"
"#else\n"
"        { float f0 = smem_x[gid * 16 + tid4 * 2 + 8];\n"
"          float f1 = smem_x[gid * 16 + tid4 * 2 + 9];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a1) : \"f\"(f0), \"f\"(f1)); }\n"
"        { float f0 = smem_x[(gid + 8) * 16 + tid4 * 2];\n"
"          float f1 = smem_x[(gid + 8) * 16 + tid4 * 2 + 1];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a2) : \"f\"(f0), \"f\"(f1)); }\n"
"#endif\n"
"        { float f0 = smem_x[(gid + 8) * 16 + tid4 * 2 + 8];\n"
"          float f1 = smem_x[(gid + 8) * 16 + tid4 * 2 + 9];\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\" : \"=r\"(a3) : \"f\"(f0), \"f\"(f1)); }\n"
"\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < GEMM_N_TILE; nt++) {\n"
"            int bc = out_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < n_out) {\n"
"                const half_raw *wp = W + (size_t)bc * n_in + k;\n"
"                b0 = *(const unsigned int *)(wp + tid4 * 2);\n"
"                b1 = *(const unsigned int *)(wp + tid4 * 2 + 8);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
"                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    int yr0 = tok_base + gid;\n"
"    int yr1 = tok_base + gid + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < GEMM_N_TILE; nt++) {\n"
"        int yc0 = out_base + nt * 8 + tid4 * 2;\n"
"        int yc1 = yc0 + 1;\n"
"        float bv0 = (bias && yc0 < n_out) ? bias[yc0] : 0.0f;\n"
"        float bv1 = (bias && yc1 < n_out) ? bias[yc1] : 0.0f;\n"
"        if (yr0 < n_tok && yc0 < n_out) Y[yr0 * n_out + yc0] = d0[nt] + bv0;\n"
"        if (yr0 < n_tok && yc1 < n_out) Y[yr0 * n_out + yc1] = d1[nt] + bv1;\n"
"        if (yr1 < n_tok && yc0 < n_out) Y[yr1 * n_out + yc0] = d2[nt] + bv0;\n"
"        if (yr1 < n_tok && yc1 < n_out) Y[yr1 * n_out + yc1] = d3[nt] + bv1;\n"
"    }\n"
"}\n"
"\n"
"} /* extern C */\n";

int main(void) {
    /* Test dimensions: must be multiples of 16 for K */
    int M = 16;   /* n_tok */
    int N = 256;  /* n_out (must be multiple of 256 for single grid block) */
    int K = 32;   /* n_in (must be multiple of 16) */

    printf("=== MMA GEMM Test (M=%d, N=%d, K=%d) ===\n", M, N, K);

    /* Initialize CUDA via cuew */
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

    /* Compile kernel */
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, test_kernel_src, "test_mma.cu", 0, NULL, NULL);

    char arch[32];
    /* Try compiling for sm_80 to let JIT handle forward compat to sm_120 */
    int target_sm = major * 10 + minor;
    const char *env_sm = getenv("TARGET_SM");
    if (env_sm) target_sm = atoi(env_sm);
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d", target_sm);
    printf("Compiling for %s\n", arch);
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

    /* Save PTX for inspection */
    FILE *fp = fopen("/tmp/test_mma.ptx", "w");
    if (fp) { fwrite(ptx, 1, ptx_sz, fp); fclose(fp); printf("PTX saved to /tmp/test_mma.ptx\n"); }

    CUmodule module;
    cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
    free(ptx);

    CUfunction fn_gemm;
    cuModuleGetFunction(&fn_gemm, module, "gemm_f16_f32");
    printf("gemm_f16_f32 kernel loaded\n");

    /* Generate test data */
    float *h_X = (float *)malloc(M * K * sizeof(float));
    float *h_W_f32 = (float *)malloc(N * K * sizeof(float));
    uint16_t *h_W_f16 = (uint16_t *)malloc(N * K * sizeof(uint16_t));
    float *h_Y_cpu = (float *)calloc(M * N, sizeof(float));
    float *h_Y_gpu = (float *)calloc(M * N, sizeof(float));

    /* Fill with small deterministic values */
    srand(42);
    for (int i = 0; i < M * K; i++)
        h_X[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;  /* [-1, 1] */
    for (int i = 0; i < N * K; i++) {
        h_W_f32[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;
        h_W_f16[i] = f32_to_f16(h_W_f32[i]);
    }

    /* CPU reference: Y[m][n] = sum_k(X[m][k] * W[n][k]) */
    /* Note: W is stored as [n_out, n_in] = [N, K] */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                /* CPU uses F16-rounded values to match GPU */
                float xf16 = f16_to_f32(f32_to_f16(h_X[m * K + k]));
                float wf16 = f16_to_f32(h_W_f16[n * K + k]);
                sum += xf16 * wf16;
            }
            h_Y_cpu[m * N + n] = sum;
        }
    }

    /* Upload to GPU */
    CUdeviceptr d_X, d_W, d_Y;
    cuMemAlloc(&d_X, M * K * sizeof(float));
    cuMemAlloc(&d_W, N * K * sizeof(uint16_t));
    cuMemAlloc(&d_Y, M * N * sizeof(float));
    cuMemcpyHtoD(d_X, h_X, M * K * sizeof(float));
    cuMemcpyHtoD(d_W, h_W_f16, N * K * sizeof(uint16_t));
    cuMemsetD32(d_Y, 0, M * N);

    /* Launch kernel */
    CUdeviceptr d_bias = 0;
    int n_out = N, n_in = K, n_tok = M;
    void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok };
    unsigned gx = (N + 255) / 256;
    unsigned gy = (M + 15) / 16;
    printf("Launching gemm_f16_f32: grid(%u,%u,1) block(128,1,1) smem=%zu\n",
           gx, gy, (size_t)(16 * 16 * sizeof(float)));
    cuLaunchKernel(fn_gemm, gx, gy, 1, 128, 1, 1,
                   16 * 16 * sizeof(float), 0, args, NULL);
    cuCtxSynchronize();

    /* Download result */
    cuMemcpyDtoH(h_Y_gpu, d_Y, M * N * sizeof(float));

    /* Compare */
    float max_diff = 0.0f;
    int max_idx = 0;
    float sum_diff2 = 0.0f, sum_ref2 = 0.0f;
    int nan_count = 0;
    for (int i = 0; i < M * N; i++) {
        if (isnan(h_Y_gpu[i])) { nan_count++; continue; }
        float diff = h_Y_gpu[i] - h_Y_cpu[i];
        sum_diff2 += diff * diff;
        sum_ref2 += h_Y_cpu[i] * h_Y_cpu[i];
        if (fabsf(diff) > max_diff) {
            max_diff = fabsf(diff);
            max_idx = i;
        }
    }
    float rel_l2 = (sum_ref2 > 0) ? sqrtf(sum_diff2 / sum_ref2) : sqrtf(sum_diff2);

    printf("\n=== Results ===\n");
    printf("Relative L2:  %.6e\n", rel_l2);
    printf("Max abs diff: %.6e at index %d\n", max_diff, max_idx);
    printf("NaN count:    %d\n", nan_count);

    printf("\nFirst 16 values (CPU vs GPU):\n");
    for (int i = 0; i < 16 && i < M * N; i++)
        printf("  [%d] cpu=%.6f gpu=%.6f diff=%.6e\n",
               i, h_Y_cpu[i], h_Y_gpu[i], h_Y_gpu[i] - h_Y_cpu[i]);

    printf("\nValues around max_diff [%d]:\n", max_idx);
    for (int i = (max_idx > 2 ? max_idx - 2 : 0); i <= max_idx + 2 && i < M * N; i++)
        printf("  [%d] cpu=%.6f gpu=%.6f diff=%.6e\n",
               i, h_Y_cpu[i], h_Y_gpu[i], h_Y_gpu[i] - h_Y_cpu[i]);

    /* Cleanup */
    cuMemFree(d_X);
    cuMemFree(d_W);
    cuMemFree(d_Y);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);
    free(h_X); free(h_W_f32); free(h_W_f16);
    free(h_Y_cpu); free(h_Y_gpu);

    printf("\n%s\n", rel_l2 < 0.01f ? "PASS" : "FAIL");
    return rel_l2 < 0.01f ? 0 : 1;
}
