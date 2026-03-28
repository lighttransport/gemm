/*
 * test_fp8_gemm.c - Standalone FP8 MMA GEMM correctness test
 *
 * Tests our gemm_fp8_f32 kernel against CPU F32 reference.
 * Isolates the GEMM from the full DiT pipeline to verify
 * fragment mapping and accumulation correctness.
 *
 * Build:
 *   cc -O2 -I../../common -I.. -o test_fp8_gemm test_fp8_gemm.c ../cuew.c -lm -ldl
 */

#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuew.h"
#include "../cuda_kernels_common.h"
#include "../cuda_runner_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- FP8 E4M3 conversion ---- */

static float fp8_to_f32(uint8_t b) {
    int s = (b >> 7) & 1;
    int e = (b >> 3) & 0xF;
    int m = b & 0x7;
    if (e == 0 && m == 0) return 0.0f;
    if (e == 15 && m == 7) return 0.0f;  /* NaN → 0 */
    if (e == 0) return (s ? -1.0f : 1.0f) * ((float)m / 8.0f) * powf(2.0f, -6.0f); /* subnormal */
    float val = (1.0f + (float)m / 8.0f) * powf(2.0f, (float)(e - 7));
    return s ? -val : val;
}

static uint8_t f32_to_fp8(float f) {
    /* Find closest FP8 E4M3 value via brute-force LUT comparison */
    uint8_t best = 0;
    float best_err = fabsf(f);
    for (int i = 1; i < 255; i++) {
        float v = fp8_to_f32((uint8_t)i);
        float err = fabsf(f - v);
        if (err < best_err) { best_err = err; best = (uint8_t)i; }
    }
    return best;
}

/* ---- Test helpers ---- */

static void cpu_gemm_f32(float *Y, const float *W, const float *X,
                          int n_out, int n_in, int n_tok) {
    /* Y[n_tok, n_out] = X[n_tok, n_in] @ W^T[n_in, n_out] */
    for (int t = 0; t < n_tok; t++)
        for (int o = 0; o < n_out; o++) {
            float sum = 0;
            for (int i = 0; i < n_in; i++)
                sum += X[t * n_in + i] * W[o * n_in + i];
            Y[t * n_out + o] = sum;
        }
}

static void cpu_gemm_fp8(float *Y, const uint8_t *W_fp8, const float *X,
                           int n_out, int n_in, int n_tok) {
    /* Same as above but dequant W from FP8 and quantize X to FP8 first */
    for (int t = 0; t < n_tok; t++)
        for (int o = 0; o < n_out; o++) {
            float sum = 0;
            for (int i = 0; i < n_in; i++) {
                float w = fp8_to_f32(W_fp8[o * n_in + i]);
                float x = fp8_to_f32(f32_to_fp8(X[t * n_in + i]));
                sum += x * w;
            }
            Y[t * n_out + o] = sum;
        }
}

static float correlation(const float *a, const float *b, int n) {
    double sa = 0, sb = 0, sab = 0, sa2 = 0, sb2 = 0;
    for (int i = 0; i < n; i++) {
        sa += a[i]; sb += b[i];
        sab += a[i] * b[i];
        sa2 += a[i] * a[i]; sb2 += b[i] * b[i];
    }
    double ma = sa / n, mb = sb / n;
    double cov = sab / n - ma * mb;
    double va = sa2 / n - ma * ma;
    double vb = sb2 / n - mb * mb;
    if (va < 1e-12 || vb < 1e-12) return 0;
    return (float)(cov / sqrt(va * vb));
}

static float max_abs_err(const float *a, const float *b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

static float rms(const float *a, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += a[i] * a[i];
    return (float)sqrt(s / n);
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    /* Init CUDA */
    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
        fprintf(stderr, "Failed to init CUEW CUDA\n"); return 1;
    }
    if (cuewInit(CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "Failed to init CUEW NVRTC\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    int sm_major, sm_minor;
    cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    int sm = sm_major * 10 + sm_minor;
    fprintf(stderr, "GPU: sm_%d\n", sm);

    if (sm < 89) {
        fprintf(stderr, "FP8 MMA requires sm_89+. Skipping.\n");
        return 0;
    }

    /* Compile kernel */
    CUmodule module;
    /* Use minimal source with just the FP8 kernel */
    static const char *fp8_test_src =
    "extern \"C\" {\n"
    "#define GEMM_N_TILE 8\n"
    /* Include the FP8 kernel from cuda_kernels_common.h line 191-285 */
    "#if __CUDA_ARCH__ >= 890\n"
    "__global__ void gemm_fp8_f32(float *Y, const unsigned char *W, const float *X,\n"
    "                              const float *bias, int n_out, int n_in, int n_tok) {\n"
    "    extern __shared__ float smem_x[];\n"
    "    int tok_base = blockIdx.y * 16;\n"
    "    int warp_id = threadIdx.x / 32;\n"
    "    int out_base = blockIdx.x * 256 + warp_id * 64;\n"
    "    int lane = threadIdx.x % 32;\n"
    "    int gid = lane / 4;\n"
    "    int tid4 = lane % 4;\n"
    "    int tid = threadIdx.x;\n"
    "    if (tok_base >= n_tok) return;\n"
    "    float d0[GEMM_N_TILE], d1[GEMM_N_TILE], d2[GEMM_N_TILE], d3[GEMM_N_TILE];\n"
    "#pragma unroll\n"
    "    for (int i = 0; i < GEMM_N_TILE; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
    "    for (int k = 0; k < n_in; k += 32) {\n"
    "        int srow = tid / 8, scol = (tid % 8) * 4;\n"
    "        int grow = tok_base + srow;\n"
    "        if (grow < n_tok) {\n"
    "            smem_x[srow * 32 + scol]     = X[grow * n_in + k + scol];\n"
    "            smem_x[srow * 32 + scol + 1] = X[grow * n_in + k + scol + 1];\n"
    "            smem_x[srow * 32 + scol + 2] = X[grow * n_in + k + scol + 2];\n"
    "            smem_x[srow * 32 + scol + 3] = X[grow * n_in + k + scol + 3];\n"
    "        } else {\n"
    "            smem_x[srow * 32 + scol] = 0.0f; smem_x[srow * 32 + scol + 1] = 0.0f;\n"
    "            smem_x[srow * 32 + scol + 2] = 0.0f; smem_x[srow * 32 + scol + 3] = 0.0f;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        unsigned int a0, a1, a2, a3;\n"
    "#define CVT_E4M3_PAIR(reg, r, c) \\\n"
    "        { unsigned short lo, hi; \\\n"
    "          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(lo) : \"f\"(smem_x[(r)*32+(c)]), \"f\"(smem_x[(r)*32+(c)+1])); \\\n"
    "          asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(hi) : \"f\"(smem_x[(r)*32+(c)+2]), \"f\"(smem_x[(r)*32+(c)+3])); \\\n"
    "          reg = (unsigned int)lo | ((unsigned int)hi << 16); }\n"
    "        CVT_E4M3_PAIR(a0, gid, tid4*4)\n"
    "#if __CUDA_ARCH__ >= 1200\n"
    "        CVT_E4M3_PAIR(a1, gid+8, tid4*4)\n"
    "        CVT_E4M3_PAIR(a2, gid,   tid4*4+16)\n"
    "#else\n"
    "        CVT_E4M3_PAIR(a1, gid,   tid4*4+16)\n"
    "        CVT_E4M3_PAIR(a2, gid+8, tid4*4)\n"
    "#endif\n"
    "        CVT_E4M3_PAIR(a3, gid+8, tid4*4+16)\n"
    "#undef CVT_E4M3_PAIR\n"
    "#pragma unroll\n"
    "        for (int nt = 0; nt < GEMM_N_TILE; nt++) {\n"
    "            int bc = out_base + nt * 8 + gid;\n"
    "            unsigned int b0 = 0, b1 = 0;\n"
    "            if (bc < n_out) {\n"
    "                const unsigned char *wp = W + (size_t)bc * n_in + k;\n"
    "                b0 = *(const unsigned int *)(wp + tid4 * 4);\n"
    "                b1 = *(const unsigned int *)(wp + tid4 * 4 + 16);\n"
    "            }\n"
    "            asm volatile(\n"
    "                \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
    "                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
    "                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
    "                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
    "                  \"r\"(b0), \"r\"(b1),\n"
    "                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
    "            );\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
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
    "#endif\n"
    "}\n";  /* close extern "C" */

    fprintf(stderr, "Kernel source length: %zu\n", strlen(fp8_test_src));
    int rc = cu_compile_kernels(&module, dev, fp8_test_src,
                                 "test_fp8.cu", 1, "test_fp8");
    if (rc < 0) { fprintf(stderr, "Kernel compilation failed\n"); return 1; }
    fprintf(stderr, "Kernels compiled OK\n");

    CUfunction gemm_fp8;
    CUresult frc = cuModuleGetFunction(&gemm_fp8, module, "gemm_fp8_f32");
    if (frc != CUDA_SUCCESS || !gemm_fp8) {
        fprintf(stderr, "gemm_fp8_f32 not found (sm_%d < 89?)\n", sm);
        return 1;
    }
    fprintf(stderr, "gemm_fp8_f32 loaded OK\n");

    CUstream stream;
    cuStreamCreate(&stream, 0);

    /* Test sizes */
    int tests[][3] = {
        /* {n_out, n_in, n_tok} */
        {8, 32, 16},       /* minimal: single MMA tile */
        {64, 64, 16},      /* one warp tile */
        {256, 256, 16},    /* multi-warp */
        {3072, 64, 256},   /* img_in projection size */
        {3072, 3072, 16},  /* attention QKV size */
        {0, 0, 0}          /* sentinel */
    };

    for (int ti = 0; tests[ti][0]; ti++) {
        int n_out = tests[ti][0], n_in = tests[ti][1], n_tok = tests[ti][2];
        int wn = n_out * n_in, xn = n_tok * n_in, yn = n_tok * n_out;

        fprintf(stderr, "\n=== Test %d: Y[%d,%d] = X[%d,%d] @ W[%d,%d]^T ===\n",
                ti, n_tok, n_out, n_tok, n_in, n_out, n_in);

        /* Generate random matrices */
        float *X = (float *)malloc(xn * sizeof(float));
        float *W_f32 = (float *)malloc(wn * sizeof(float));
        uint8_t *W_fp8 = (uint8_t *)malloc(wn);
        float *Y_ref = (float *)calloc(yn, sizeof(float));
        float *Y_fp8_ref = (float *)calloc(yn, sizeof(float));
        float *Y_gpu = (float *)calloc(yn, sizeof(float));

        srand(42 + ti);
        for (int i = 0; i < xn; i++) X[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        for (int i = 0; i < wn; i++) {
            W_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            W_fp8[i] = f32_to_fp8(W_f32[i]);
        }

        /* CPU references */
        cpu_gemm_f32(Y_ref, W_f32, X, n_out, n_in, n_tok);
        cpu_gemm_fp8(Y_fp8_ref, W_fp8, X, n_out, n_in, n_tok);

        fprintf(stderr, "  CPU F32 ref: rms=%.4f\n", rms(Y_ref, yn));
        fprintf(stderr, "  CPU FP8 ref: rms=%.4f  corr_vs_f32=%.6f  max_err=%.4f\n",
                rms(Y_fp8_ref, yn), correlation(Y_ref, Y_fp8_ref, yn),
                max_abs_err(Y_ref, Y_fp8_ref, yn));

        /* GPU FP8 GEMM */
        CUdeviceptr d_X, d_W, d_Y;
        cuMemAlloc(&d_X, xn * sizeof(float));
        cuMemAlloc(&d_W, wn);  /* FP8 = 1 byte per element */
        cuMemAlloc(&d_Y, yn * sizeof(float));
        cuMemcpyHtoD(d_X, X, xn * sizeof(float));
        cuMemcpyHtoD(d_W, W_fp8, wn);
        cuMemsetD32(d_Y, 0, yn);

        CUdeviceptr d_bias = 0;  /* no bias */
        void *args[] = {&d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok};
        unsigned gx = (unsigned)((n_out + 255) / 256);
        unsigned gy = (unsigned)((n_tok + 15) / 16);
        CUresult lr = cuLaunchKernel(gemm_fp8, gx, gy, 1, 128, 1, 1,
                                      16 * 32 * sizeof(float), stream, args, NULL);
        if (lr != CUDA_SUCCESS)
            fprintf(stderr, "  Launch FAILED: %d\n", (int)lr);
        cuStreamSynchronize(stream);

        cuMemcpyDtoH(Y_gpu, d_Y, yn * sizeof(float));

        /* Compare */
        float corr_gpu_f32 = correlation(Y_ref, Y_gpu, yn);
        float corr_gpu_fp8 = correlation(Y_fp8_ref, Y_gpu, yn);
        float mae_gpu_fp8 = max_abs_err(Y_fp8_ref, Y_gpu, yn);
        fprintf(stderr, "  GPU FP8:     rms=%.4f  corr_vs_f32=%.6f  corr_vs_fp8_ref=%.6f  max_err_vs_fp8=%.4f\n",
                rms(Y_gpu, yn), corr_gpu_f32, corr_gpu_fp8, mae_gpu_fp8);

        /* Check for NaN */
        int nan_count = 0;
        for (int i = 0; i < yn; i++) if (Y_gpu[i] != Y_gpu[i]) nan_count++;
        if (nan_count) fprintf(stderr, "  WARNING: %d NaN values!\n", nan_count);

        /* Print first few values for debugging */
        fprintf(stderr, "  Y_ref[0:4]:     ");
        for (int i = 0; i < 4 && i < yn; i++) fprintf(stderr, "%.4f ", Y_ref[i]);
        fprintf(stderr, "\n  Y_fp8_ref[0:4]: ");
        for (int i = 0; i < 4 && i < yn; i++) fprintf(stderr, "%.4f ", Y_fp8_ref[i]);
        fprintf(stderr, "\n  Y_gpu[0:4]:     ");
        for (int i = 0; i < 4 && i < yn; i++) fprintf(stderr, "%.4f ", Y_gpu[i]);
        fprintf(stderr, "\n");

        if (corr_gpu_fp8 < 0.99f)
            fprintf(stderr, "  *** FAIL: GPU-FP8 correlation too low (%.4f < 0.99) ***\n", corr_gpu_fp8);
        else
            fprintf(stderr, "  PASS\n");

        cuMemFree(d_X); cuMemFree(d_W); cuMemFree(d_Y);
        free(X); free(W_f32); free(W_fp8); free(Y_ref); free(Y_fp8_ref); free(Y_gpu);
    }

    cuStreamDestroy(stream);
    cuCtxDestroy(ctx);
    return 0;
}
