/*
 * test_fp8_gemm.c - Standalone FP8 MMA GEMM correctness test
 *
 * Tests our gemm_fp8_f32 kernel against multithreaded CPU FP8 reference.
 *
 * Build:
 *   cc -O2 -I../../common -I.. -o test_fp8_gemm test_fp8_gemm.c ../cuew.c -lm -ldl -lpthread
 */

#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuew.h"
#include "../cuda_kernels_common.h"
#include "../cuda_runner_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

/* ---- FP8 E4M3 conversion with pre-built LUTs ---- */

static float g_fp8_to_f32[256];
static uint8_t g_f32_to_fp8_pos[1024]; /* quantize positive values 0..447 → FP8 */

static void init_fp8_luts(void) {
    /* FP8 → F32 LUT */
    for (int i = 0; i < 256; i++) {
        int s = (i >> 7) & 1;
        int e = (i >> 3) & 0xF;
        int m = i & 0x7;
        float val;
        if (e == 0 && m == 0) val = 0.0f;
        else if (e == 15 && m == 7) val = 0.0f; /* NaN → 0 */
        else if (e == 0) val = ((float)m / 8.0f) * (1.0f / 64.0f); /* subnormal */
        else val = (1.0f + (float)m / 8.0f) * powf(2.0f, (float)(e - 7));
        g_fp8_to_f32[i] = s ? -val : val;
    }

    /* Build fast F32→FP8 quantization by sorting positive FP8 values */
    /* FP8 E4M3 positive values: 0, subnormals (e=0,m=1..7), normals (e=1..14, m=0..7), max (e=15,m=6) */
    /* Total positive: 127 values (indices 0..126). Find nearest for any float. */
    /* For speed: use the PTX cvt instruction semantics (round-to-nearest-even, saturate) */
}

static float fp8_to_f32(uint8_t b) { return g_fp8_to_f32[b]; }

static uint8_t f32_to_fp8_fast(float f) {
    /* Proper FP8 E4M3 quantization: extract sign, clamp, find nearest */
    int sign = 0;
    if (f < 0) { sign = 1; f = -f; }
    if (f > 448.0f) f = 448.0f; /* saturate */
    if (f < 0.001953125f) return sign ? 0x80 : 0x00; /* below smallest subnormal → ±0 */

    /* Find exponent */
    int e;
    float m_val;
    if (f < 0.015625f) { /* subnormal range: 2^-9 to 2^-6 */
        e = 0;
        m_val = f * 64.0f; /* f / 2^-6 */
    } else {
        /* log2(f) to find exponent */
        int raw_e = (int)floorf(log2f(f));
        e = raw_e + 7; /* bias = 7 */
        if (e < 1) e = 1;
        if (e > 15) e = 15;
        float base = powf(2.0f, (float)(e - 7));
        m_val = f / base - 1.0f;
    }

    /* Round mantissa to 3 bits */
    int m = (int)roundf(m_val * 8.0f);
    if (m > 7) { m = 0; e++; }
    if (e > 15) { e = 15; m = 6; } /* max normal = 448 */
    if (e == 15 && m > 6) m = 6;   /* avoid NaN (e=15,m=7) */

    uint8_t result = (uint8_t)((sign << 7) | (e << 3) | m);
    return result;
}

/* ---- Multithreaded CPU GEMM ---- */

typedef struct {
    float *Y;
    const float *W;  /* F32 weights (dequanted from FP8) */
    const float *X;  /* F32 inputs (dequanted from FP8) */
    int n_out, n_in, n_tok;
    int tok_start, tok_end;
} gemm_task;

static void *gemm_worker(void *arg) {
    gemm_task *t = (gemm_task *)arg;
    for (int tok = t->tok_start; tok < t->tok_end; tok++)
        for (int o = 0; o < t->n_out; o++) {
            float sum = 0;
            for (int i = 0; i < t->n_in; i++)
                sum += t->X[tok * t->n_in + i] * t->W[o * t->n_in + i];
            t->Y[tok * t->n_out + o] = sum;
        }
    return NULL;
}

static void cpu_gemm_mt(float *Y, const float *W, const float *X,
                         int n_out, int n_in, int n_tok, int n_threads) {
    pthread_t threads[64];
    gemm_task tasks[64];
    if (n_threads > 64) n_threads = 64;
    if (n_threads > n_tok) n_threads = n_tok;

    for (int i = 0; i < n_threads; i++) {
        tasks[i] = (gemm_task){Y, W, X, n_out, n_in, n_tok,
                               i * n_tok / n_threads, (i + 1) * n_tok / n_threads};
        pthread_create(&threads[i], NULL, gemm_worker, &tasks[i]);
    }
    for (int i = 0; i < n_threads; i++)
        pthread_join(threads[i], NULL);
}

/* ---- Stats ---- */

static float correlation(const float *a, const float *b, int n) {
    double sa = 0, sb = 0, sab = 0, sa2 = 0, sb2 = 0;
    for (int i = 0; i < n; i++) {
        sa += a[i]; sb += b[i]; sab += a[i] * b[i];
        sa2 += a[i] * a[i]; sb2 += b[i] * b[i];
    }
    double ma = sa / n, mb = sb / n;
    double cov = sab / n - ma * mb;
    double va = sa2 / n - ma * ma, vb = sb2 / n - mb * mb;
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
    int n_threads = 16;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--threads") == 0 && i+1 < argc)
            n_threads = atoi(argv[++i]);

    init_fp8_luts();

    /* Verify LUT roundtrip */
    {
        int mismatch = 0;
        for (int i = 0; i < 256; i++) {
            float v = fp8_to_f32((uint8_t)i);
            uint8_t q = f32_to_fp8_fast(v);
            float v2 = fp8_to_f32(q);
            if (fabsf(v - v2) > 1e-6f && v != 0.0f) mismatch++;
        }
        fprintf(stderr, "FP8 roundtrip: %d/256 mismatches\n", mismatch);
    }

    /* Init CUDA */
    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS ||
        cuewInit(CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "Failed to init CUEW\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    int sm_major, sm_minor;
    cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    int sm = sm_major * 10 + sm_minor;
    fprintf(stderr, "GPU: sm_%d, threads: %d\n", sm, n_threads);

    if (sm < 89) { fprintf(stderr, "FP8 MMA requires sm_89+\n"); return 0; }

    /* Compile kernel (inline source to avoid extern "C" issues) */
    static const char *src =
    "extern \"C\" {\n"
    "#define GEMM_N_TILE 8\n"
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
    "            smem_x[srow*32+scol] = X[grow*n_in+k+scol];\n"
    "            smem_x[srow*32+scol+1] = X[grow*n_in+k+scol+1];\n"
    "            smem_x[srow*32+scol+2] = X[grow*n_in+k+scol+2];\n"
    "            smem_x[srow*32+scol+3] = X[grow*n_in+k+scol+3];\n"
    "        } else {\n"
    "            smem_x[srow*32+scol]=0; smem_x[srow*32+scol+1]=0;\n"
    "            smem_x[srow*32+scol+2]=0; smem_x[srow*32+scol+3]=0;\n"
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
    "    int yr0 = tok_base + gid, yr1 = tok_base + gid + 8;\n"
    "#pragma unroll\n"
    "    for (int nt = 0; nt < GEMM_N_TILE; nt++) {\n"
    "        int yc0 = out_base + nt * 8 + tid4 * 2, yc1 = yc0 + 1;\n"
    "        float bv0 = (bias && yc0 < n_out) ? bias[yc0] : 0.0f;\n"
    "        float bv1 = (bias && yc1 < n_out) ? bias[yc1] : 0.0f;\n"
    "        if (yr0 < n_tok && yc0 < n_out) Y[yr0*n_out+yc0] = d0[nt]+bv0;\n"
    "        if (yr0 < n_tok && yc1 < n_out) Y[yr0*n_out+yc1] = d1[nt]+bv1;\n"
    "        if (yr1 < n_tok && yc0 < n_out) Y[yr1*n_out+yc0] = d2[nt]+bv0;\n"
    "        if (yr1 < n_tok && yc1 < n_out) Y[yr1*n_out+yc1] = d3[nt]+bv1;\n"
    "    }\n"
    "}\n"
    "#endif\n"
    "}\n";

    CUmodule module;
    int rc = cu_compile_kernels(&module, dev, src, "fp8.cu", 1, "fp8_test");
    if (rc < 0) return 1;

    CUfunction gemm_fp8;
    if (cuModuleGetFunction(&gemm_fp8, module, "gemm_fp8_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "gemm_fp8_f32 not found\n"); return 1;
    }
    fprintf(stderr, "gemm_fp8_f32 loaded OK\n\n");

    CUstream stream;
    cuStreamCreate(&stream, 0);

    /* Test sizes: {n_out, n_in, n_tok} */
    int tests[][3] = {
        {8, 32, 16},         /* minimal MMA tile */
        {64, 64, 16},        /* one warp tile */
        {256, 256, 16},      /* multi-warp */
        {3072, 64, 256},     /* img_in projection */
        {3072, 3072, 16},    /* attention QKV (16 tok) */
        {3072, 3072, 256},   /* attention QKV (256 tok) */
        {18432, 3072, 1},    /* modulation (1 tok) */
        {12288, 3072, 256},  /* MLP fc1 */
        {0, 0, 0}
    };

    int all_pass = 1;
    for (int ti = 0; tests[ti][0]; ti++) {
        int n_out = tests[ti][0], n_in = tests[ti][1], n_tok = tests[ti][2];
        size_t wn = (size_t)n_out * n_in, xn = (size_t)n_tok * n_in, yn = (size_t)n_tok * n_out;

        fprintf(stderr, "Test %d: Y[%d,%d] = X[%d,%d] @ W[%d,%d]^T ... ",
                ti, n_tok, n_out, n_tok, n_in, n_out, n_in);
        fflush(stderr);

        float *X = (float *)malloc(xn * sizeof(float));
        float *W_f32 = (float *)malloc(wn * sizeof(float));
        uint8_t *W_fp8 = (uint8_t *)malloc(wn);
        float *X_fp8_f32 = (float *)malloc(xn * sizeof(float)); /* X quantized to FP8, stored as F32 */
        float *W_fp8_f32 = (float *)malloc(wn * sizeof(float)); /* W dequanted to F32 */
        float *Y_ref = (float *)calloc(yn, sizeof(float));
        float *Y_fp8_ref = (float *)calloc(yn, sizeof(float));
        float *Y_gpu = (float *)calloc(yn, sizeof(float));

        srand(42 + ti);
        for (size_t i = 0; i < xn; i++) X[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        for (size_t i = 0; i < wn; i++) {
            W_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            W_fp8[i] = f32_to_fp8_fast(W_f32[i]);
            W_fp8_f32[i] = fp8_to_f32(W_fp8[i]);
        }
        /* Pre-quantize X to FP8 for CPU reference */
        for (size_t i = 0; i < xn; i++)
            X_fp8_f32[i] = fp8_to_f32(f32_to_fp8_fast(X[i]));

        /* CPU F32 reference */
        cpu_gemm_mt(Y_ref, W_f32, X, n_out, n_in, n_tok, n_threads);
        /* CPU FP8 reference (both inputs quantized to FP8, accumulated in F32) */
        cpu_gemm_mt(Y_fp8_ref, W_fp8_f32, X_fp8_f32, n_out, n_in, n_tok, n_threads);

        /* GPU FP8 GEMM */
        CUdeviceptr d_X, d_W, d_Y;
        cuMemAlloc(&d_X, xn * sizeof(float));
        cuMemAlloc(&d_W, wn);
        cuMemAlloc(&d_Y, yn * sizeof(float));
        cuMemcpyHtoD(d_X, X, xn * sizeof(float));
        cuMemcpyHtoD(d_W, W_fp8, wn);
        cuMemsetD32(d_Y, 0, (unsigned int)yn);

        CUdeviceptr d_bias = 0;
        void *args[] = {&d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok};
        unsigned gx = (unsigned)((n_out + 255) / 256);
        unsigned gy = (unsigned)((n_tok + 15) / 16);
        cuLaunchKernel(gemm_fp8, gx, gy, 1, 128, 1, 1,
                       16 * 32 * sizeof(float), stream, args, NULL);
        cuStreamSynchronize(stream);
        cuMemcpyDtoH(Y_gpu, d_Y, yn * sizeof(float));

        /* Check NaN */
        int nans = 0;
        for (size_t i = 0; i < yn; i++) if (Y_gpu[i] != Y_gpu[i]) nans++;

        float corr_fp8 = correlation(Y_fp8_ref, Y_gpu, (int)yn);
        float mae = max_abs_err(Y_fp8_ref, Y_gpu, (int)yn);
        float corr_f32 = correlation(Y_ref, Y_gpu, (int)yn);

        int pass = (corr_fp8 > 0.999f && nans == 0);
        fprintf(stderr, "%s  corr_fp8=%.6f  corr_f32=%.6f  mae=%.4f  rms=%.3f%s\n",
                pass ? "PASS" : "FAIL", corr_fp8, corr_f32, mae, rms(Y_gpu, (int)yn),
                nans ? "  NaN!" : "");
        if (!pass) {
            all_pass = 0;
            fprintf(stderr, "  Y_ref[0:4]:     %.4f %.4f %.4f %.4f\n",
                    Y_ref[0], yn>1?Y_ref[1]:0, yn>2?Y_ref[2]:0, yn>3?Y_ref[3]:0);
            fprintf(stderr, "  Y_fp8_ref[0:4]: %.4f %.4f %.4f %.4f\n",
                    Y_fp8_ref[0], yn>1?Y_fp8_ref[1]:0, yn>2?Y_fp8_ref[2]:0, yn>3?Y_fp8_ref[3]:0);
            fprintf(stderr, "  Y_gpu[0:4]:     %.4f %.4f %.4f %.4f\n",
                    Y_gpu[0], yn>1?Y_gpu[1]:0, yn>2?Y_gpu[2]:0, yn>3?Y_gpu[3]:0);
        }

        cuMemFree(d_X); cuMemFree(d_W); cuMemFree(d_Y);
        free(X); free(W_f32); free(W_fp8); free(X_fp8_f32); free(W_fp8_f32);
        free(Y_ref); free(Y_fp8_ref); free(Y_gpu);
    }

    fprintf(stderr, "\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    cuStreamDestroy(stream);
    cuCtxDestroy(ctx);
    return all_pass ? 0 : 1;
}
