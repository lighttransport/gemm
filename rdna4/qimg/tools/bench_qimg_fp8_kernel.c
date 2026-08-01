/* Standalone throughput microbench for qimg's hand-written FP8xFP8 WMMA GEMM
 * kernel (gemm_fp8_fp8w_perrow_pgr2), the kernel qimg's --fast fp8_matrix_mult
 * path actually dispatches (hipBLASLt's C++ FP8 path is broken on ROCm 7.2.2).
 *
 * Compiles the IDENTICAL HIPRTC source the runner builds (common + qimg
 * kernels) and times only the GEMM (activation cast excluded), so the numbers
 * are directly comparable to tools/bench_comfy_fast_fp8.py (torch._scaled_mm).
 *
 * Build (from rdna4/qimg):
 *   gcc -O3 -I.. -o tools/bench_qimg_fp8_kernel tools/bench_qimg_fp8_kernel.c \
 *       ../rocew.c -ldl -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../rocew.h"
#include "../../hip_kernels_common.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../../hip_runner_common.h"
#include "../hip_qimg_kernels.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP err %d at %s:%d\n", (int)e_, __FILE__, __LINE__); exit(1);} } while(0)

typedef struct { const char *label; int N, K; } gemm_t;

int main(int argc, char **argv) {
    int device_id = 0, iters = 100, warm = 10;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) iters = atoi(argv[++i]);
    }
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "rocewInit failed\n"); return 1;
    }
    CK(hipSetDevice(device_id));

    /* Compile the same source the runner compiles. */
    size_t l1 = strlen(hip_kernels_common_src), l2 = strlen(hip_qimg_specific_kernels);
    char *src = malloc(l1 + l2 + 1);
    memcpy(src, hip_kernels_common_src, l1);
    memcpy(src + l1, hip_qimg_specific_kernels, l2);
    src[l1 + l2] = '\0';
    hipModule_t mod;
    if (hip_compile_kernels(&mod, device_id, src, "qimg_bench.hip", 0, "bench") < 0) {
        fprintf(stderr, "compile failed\n"); return 1;
    }
    free(src);
    hipFunction_t fn;
    CK(hipModuleGetFunction(&fn, mod, "gemm_fp8_fp8w_perrow_pgr2"));

    hipDeviceProp_t props; hipGetDeviceProperties(&props, device_id);
    printf("gpu=%s  kernel=gemm_fp8_fp8w_perrow_pgr2 (qimg hand-written FP8xFP8 WMMA)\n", props.name);

    /* Qwen-Image MMDiT block GEMM shapes (hidden=3072, mlp_h=12288). */
    gemm_t gemms[] = {
        {"attn_q/k/v/out", 3072, 3072},
        {"mlp_fc1",       12288, 3072},
        {"mlp_fc2",        3072, 12288},
        {"mod",           18432, 3072},
    };
    int Ms[] = {256, 4096};
    const char *res[] = {"256x256", "1024x1024"};

    for (int mi = 0; mi < 2; mi++) {
        int M = Ms[mi], M_pad = ((M + 127) / 128) * 128, n_tok = M;
        printf("\n=== M=%d tokens (%s) ===\n", M, res[mi]);
        printf("%-16s %-22s %10s %10s\n", "gemm", "shape(MxNxK)", "TF/s", "ms");
        for (size_t g = 0; g < sizeof(gemms)/sizeof(gemms[0]); g++) {
            int N = gemms[g].N, K = gemms[g].K;
            void *W, *X, *Y, *scales;
            size_t wsz = (size_t)N * K, xsz = (size_t)M_pad * K;
            CK(hipMalloc(&W, wsz)); CK(hipMalloc(&X, xsz));
            CK(hipMalloc(&Y, (size_t)n_tok * N * sizeof(float)));
            CK(hipMalloc(&scales, (size_t)M_pad * sizeof(float)));
            /* random fp8 bytes; perf is value-independent. scales = 1. */
            unsigned char *hb = malloc(wsz > xsz ? wsz : xsz);
            for (size_t i = 0; i < (wsz > xsz ? wsz : xsz); i++) hb[i] = (unsigned char)(rand() & 0x77);
            CK(hipMemcpy(W, hb, wsz, hipMemcpyHostToDevice));
            CK(hipMemcpy(X, hb, xsz, hipMemcpyHostToDevice));
            float *hs = malloc((size_t)M_pad * sizeof(float));
            for (int i = 0; i < M_pad; i++) hs[i] = 1.0f;
            CK(hipMemcpy(scales, hs, (size_t)M_pad * sizeof(float), hipMemcpyHostToDevice));
            void *bias = NULL;
            void *args[] = {&Y, &W, &X, &bias, &scales, &N, &K, &n_tok, &M_pad};
            unsigned gx = N / 128, gy = M_pad / 128;

            for (int w = 0; w < warm; w++)
                CK(hipModuleLaunchKernel(fn, gx, gy, 1, 128, 1, 1, 0, NULL, args, NULL));
            CK(hipDeviceSynchronize());
            hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
            CK(hipEventRecord(t0, NULL));
            for (int it = 0; it < iters; it++)
                CK(hipModuleLaunchKernel(fn, gx, gy, 1, 128, 1, 1, 0, NULL, args, NULL));
            CK(hipEventRecord(t1, NULL)); CK(hipEventSynchronize(t1));
            float ms; CK(hipEventElapsedTime(&ms, t0, t1)); ms /= iters;
            double tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
            char shape[32]; snprintf(shape, sizeof(shape), "%dx%dx%d", M, N, K);
            printf("%-16s %-22s %10.1f %10.4f\n", gemms[g].label, shape, tf, ms);
            hipEventDestroy(t0); hipEventDestroy(t1);
            hipFree(W); hipFree(X); hipFree(Y); hipFree(scales); free(hb); free(hs);
        }
    }
    return 0;
}
