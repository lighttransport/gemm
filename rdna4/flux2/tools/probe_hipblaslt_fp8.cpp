/* probe_hipblaslt_fp8.c — does ROCm 7.2.3 hipBLASLt successfully execute an FP8
 * E4M3 GEMM on gfx1201 (RX 9070 XT)? If yes, does it beat our hand-written pipe32?
 *
 * Probe shapes (klein-4b @512): M=4096 (img n_tok), K=3072 (H), N=9216 (qkv out).
 *
 * Build:  gcc -O2 -I/opt/rocm/include -o probe_hipblaslt_fp8 probe_hipblaslt_fp8.c \
 *             -L/opt/rocm/lib -lhipblaslt -lamdhip64
 *
 * Reports: status code per call, time/TF-s for the GEMM if it ran, and a coarse
 * correctness check (FP8 quantize the bf16 inputs, do FP32 GEMM as reference, cosine).
 */
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define HIP_CHECK(call) do { hipError_t e=(call); if(e!=hipSuccess){ fprintf(stderr,"HIP err %d (%s) at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); exit(1);} } while(0)
#define BL_CHECK(call, label) do { hipblasStatus_t s=(call); if(s!=HIPBLAS_STATUS_SUCCESS){ fprintf(stderr,"hipBLASLt %s -> status %d\n", label, (int)s); return (int)s; } } while(0)

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* F32 -> FP8 E4M3 (round-nearest, saturate at ±448). */
static unsigned char f32_to_fp8_e4m3(float f) {
    if (f >  448.0f) f =  448.0f;
    if (f < -448.0f) f = -448.0f;
    unsigned int b; memcpy(&b, &f, 4);
    unsigned int sign = b >> 31;
    int e = (int)((b >> 23) & 0xFFu) - 127;
    unsigned int mant = b & 0x7FFFFFu;
    int fp8_exp = e + 7;
    if (e < -9) return (unsigned char)(sign << 7);
    if (fp8_exp <= 0) {
        unsigned int full = mant | 0x800000u;
        int shift = 1 - fp8_exp + 20;
        if (shift >= 24) return (unsigned char)(sign << 7);
        unsigned int m8 = (full + (1u << (shift-1))) >> shift;
        if (m8 > 7) m8 = 7;
        return (unsigned char)((sign << 7) | (m8 & 7u));
    }
    if (fp8_exp >= 15) return (unsigned char)((sign<<7) | (15u<<3) | 6u);
    unsigned int m8 = (mant + (1u << 19)) >> 20;
    if (m8 > 7) { m8 = 0; fp8_exp++; }
    if (fp8_exp >= 15) return (unsigned char)((sign<<7) | (15u<<3) | 6u);
    return (unsigned char)((sign<<7) | ((unsigned)fp8_exp<<3) | (m8 & 7u));
}

static float fp8_e4m3_to_f32(unsigned char b) {
    unsigned int sign = (b >> 7) & 1;
    int e = (int)((b >> 3) & 0xF);
    unsigned int m = b & 0x7u;
    float v;
    if (e == 0 && m == 0) v = 0.0f;
    else if (e == 0)      v = ldexpf((float)m / 8.0f, -6);
    else if (e == 15 && m == 7) return 0.0f;
    else                  v = ldexpf(1.0f + (float)m/8.0f, e - 7);
    return sign ? -v : v;
}

static int probe_shape(hipblasLtHandle_t lt, hipStream_t stream, int M, int K, int N) {
    printf("\n=== shape M=%d K=%d N=%d ===\n", M, K, N);
    size_t Asz = (size_t)M*K, Bsz = (size_t)K*N, Csz = (size_t)M*N;
    unsigned char *hA = (unsigned char*)malloc(Asz);
    unsigned char *hB = (unsigned char*)malloc(Bsz);
    float *hC = (float*)malloc(Csz*sizeof(float));
    float *hRef = (float*)malloc(Csz*sizeof(float));
    /* synthetic fp8: random small values that decode cleanly */
    srand(1);
    for (size_t i=0;i<Asz;i++) hA[i] = f32_to_fp8_e4m3(((float)rand()/RAND_MAX - 0.5f) * 0.6f);
    for (size_t i=0;i<Bsz;i++) hB[i] = f32_to_fp8_e4m3(((float)rand()/RAND_MAX - 0.5f) * 0.6f);

    void *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, Asz));
    HIP_CHECK(hipMalloc(&dB, Bsz));
    HIP_CHECK(hipMalloc(&dC, Csz*sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA, Asz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB, Bsz, hipMemcpyHostToDevice));

    /* hipBLASLt FP8 only supports col-major. Use opT/opT on inputs (which interpret my
     * row-major bytes correctly as A_rm, B_rm) and read the C_cm output with col-major
     * indexing in host. The fast path empirically — no-transpose hung in algo lookup. */
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t lA, lB, lC;
    BL_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), "DescCreate");
    hipblasOperation_t opT = HIPBLAS_OP_T;
    BL_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)), "set transA=T");
    BL_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)), "set transB=T");
    /* A_cm = [K, M] ld=K (my row-major [M,K]); opT -> A_rm [M, K]. */
    BL_CHECK(hipblasLtMatrixLayoutCreate(&lA, HIP_R_8F_E4M3, K, M, K), "lA");
    /* B_cm = [N, K] ld=N (my row-major [K,N]); opT -> B_rm [K, N]. */
    BL_CHECK(hipblasLtMatrixLayoutCreate(&lB, HIP_R_8F_E4M3, N, K, N), "lB");
    /* C_cm = [M, N] ld=M  (col-major output). Host indexes hC[m + n*M]. */
    BL_CHECK(hipblasLtMatrixLayoutCreate(&lC, HIP_R_32F,     M, N, M), "lC");

    /* heuristic: find an algo */
    hipblasLtMatmulPreference_t pref;
    BL_CHECK(hipblasLtMatmulPreferenceCreate(&pref), "PrefCreate");
    size_t ws_max = 64ull * 1024 * 1024;
    BL_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_max, sizeof(ws_max)), "set ws_max");

    hipblasLtMatmulHeuristicResult_t hres[8] = {0};
    int n_algos = 0;
    BL_CHECK(hipblasLtMatmulAlgoGetHeuristic(lt, desc, lA, lB, lC, lC, pref, 8, hres, &n_algos), "Heuristic");
    if (n_algos == 0) {
        fprintf(stderr, "  NO ALGOS RETURNED for FP8 shape — vendor path unavailable.\n");
        hipFree(dA); hipFree(dB); hipFree(dC); free(hA); free(hB); free(hC); free(hRef);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtMatrixLayoutDestroy(lA);
        hipblasLtMatrixLayoutDestroy(lB);
        hipblasLtMatrixLayoutDestroy(lC);
        hipblasLtMatmulPreferenceDestroy(pref);
        return -2;
    }
    printf("  heuristic returned %d algos. workspace=%zu\n", n_algos, hres[0].workspaceSize);

    void *ws = NULL;
    if (hres[0].workspaceSize > 0) HIP_CHECK(hipMalloc(&ws, hres[0].workspaceSize));

    float alpha = 1.0f, beta = 0.0f;
    /* warm-up + 1 timed run */
    hipblasStatus_t s = hipblasLtMatmul(lt, desc, &alpha, dA, lA, dB, lB,
                                        &beta, dC, lC, dC, lC,
                                        &hres[0].algo, ws, hres[0].workspaceSize, stream);
    if (s != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "  hipblasLtMatmul (warm) -> status %d\n", (int)s);
        if (ws) hipFree(ws);
        return (int)s;
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    int iters = 5;
    double t0 = now_s();
    for (int it = 0; it < iters; it++) {
        BL_CHECK(hipblasLtMatmul(lt, desc, &alpha, dA, lA, dB, lB,
                                 &beta, dC, lC, dC, lC,
                                 &hres[0].algo, ws, hres[0].workspaceSize, stream), "matmul");
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    double dt = (now_s() - t0) / iters;
    double tflops = 2.0 * M * K * N / dt / 1e12;
    printf("  hipBLASLt FP8 GEMM: %.3f ms / %.1f TF/s\n", dt*1e3, tflops);

    /* coarse correctness — F32 reference via dequant + GEMM (slow). */
    HIP_CHECK(hipMemcpy(hC, dC, Csz*sizeof(float), hipMemcpyDeviceToHost));
    /* Run a tiny F32 ref over a SUBSET (first 64 rows × 64 cols) — full GEMM is too slow. */
    int Mc = 64, Nc = 64;
    double dot=0, nc=0, ng=0;
    for (int m=0;m<Mc;m++) {
        for (int n=0;n<Nc;n++) {
            float acc = 0.f;
            for (int k=0;k<K;k++) {
                float a = fp8_e4m3_to_f32(hA[(size_t)m*K + k]);
                float b = fp8_e4m3_to_f32(hB[(size_t)k*N + n]);
                acc += a*b;
            }
            float g = hC[(size_t)m + (size_t)n*M];   /* col-major C_cm[M,N] */
            dot += acc*g; nc += acc*acc; ng += g*g;
        }
    }
    double cos = dot / (sqrt(nc)*sqrt(ng) + 1e-30);
    printf("  correctness (first 64×64): cos(vendor, host-fp8-ref) = %.5f\n", cos);

    if (ws) hipFree(ws);
    hipFree(dA); hipFree(dB); hipFree(dC); free(hA); free(hB); free(hC); free(hRef);
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulPreferenceDestroy(pref);
    return 0;
}

int main(void) {
    hipblasLtHandle_t lt;
    BL_CHECK(hipblasLtCreate(&lt), "Create");
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    printf("Probe: ROCm + hipBLASLt FP8 E4M3 GEMM on klein-4b shapes (gfx1201 / RX 9070 XT)\n");

    /* klein-4b shapes @512 res: */
    probe_shape(lt, stream, 4096, 3072,  9216);  /* QKV       */
    probe_shape(lt, stream, 4096, 3072,  3072);  /* proj      */
    probe_shape(lt, stream, 4096, 3072, 18432);  /* MLP up    */
    probe_shape(lt, stream, 4096, 9216,  3072);  /* MLP down  */
    probe_shape(lt, stream, 4128, 3072, 27648);  /* single_blocks.linear1 */

    hipStreamDestroy(stream);
    hipblasLtDestroy(lt);
    return 0;
}
