/*
 * test_pixel_shuffle_3d — Phase 4b.0 standalone microbench.
 *
 * Validates pixel_shuffle_3d_f32 (3D pixel shuffle, upscale=2)
 * on SS-VAE decoder geometry. Two shapes are exercised:
 *   16³ stage: src [1024, 16, 16, 16] -> dst [128, 32, 32, 32]
 *   32³ stage: src [256,  32, 32, 32] -> dst [32,  64, 64, 64]
 *
 * Pure data movement; device output is bit-exact vs host reference.
 *
 * Usage:
 *   ./test_pixel_shuffle_3d [-v]
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
#include "cuda_sam3d_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float urand(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return (float)((*s) >> 8) / (float)(1u << 24);
}

static void host_pixel_shuffle_3d(const float *src, float *dst,
                                  int C, int D, int H, int W) {
    int D2 = 2*D, H2 = 2*H, W2 = 2*W;
    long long sp_in  = (long long)D  * H  * W;
    long long sp_out = (long long)D2 * H2 * W2;
    for (int c = 0; c < C; c++) {
        for (int d = 0; d < D; d++) for (int h = 0; h < H; h++) for (int w = 0; w < W; w++) {
            for (int sd = 0; sd < 2; sd++) for (int sh = 0; sh < 2; sh++) for (int sw = 0; sw < 2; sw++) {
                int sub_ch = (sd * 2 + sh) * 2 + sw;
                int src_ch = c * 8 + sub_ch;
                int od = 2*d + sd, oh = 2*h + sh, ow = 2*w + sw;
                dst[(long long)c * sp_out + (long long)od * H2 * W2 + (long long)oh * W2 + ow] =
                    src[(long long)src_ch * sp_in + (long long)d * H * W + (long long)h * W + w];
            }
        }
    }
}

static int run_one(CUfunction fn, int C, int D, int H, int W, int verbose) {
    int Cin = C * 8;
    int D2 = 2*D, H2 = 2*H, W2 = 2*W;
    long long n_in  = (long long)Cin * D  * H  * W;
    long long n_out = (long long)C   * D2 * H2 * W2;
    float *h_src = (float *)malloc((size_t)n_in  * sizeof(float));
    float *h_ref = (float *)malloc((size_t)n_out * sizeof(float));
    float *h_dst = (float *)malloc((size_t)n_out * sizeof(float));
    if (!h_src || !h_ref || !h_dst) return 5;
    uint32_t rng = 0xC0FFEEu ^ (uint32_t)(C * 31 + D);
    for (long long i = 0; i < n_in; i++) h_src[i] = urand(&rng) * 8.0f - 4.0f;
    host_pixel_shuffle_3d(h_src, h_ref, C, D, H, W);

    CUdeviceptr d_src = cu_upload_raw(h_src, (size_t)n_in * sizeof(float));
    CUdeviceptr d_dst = 0;
    if (cuMemAlloc(&d_dst, (size_t)n_out * sizeof(float)) != CUDA_SUCCESS) return 6;

    int threads = 256;
    int blocks  = (int)((n_out + threads - 1) / threads);
    void *args[] = { &d_src, &d_dst, &C, &D, &H, &W };
    if (cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL) != CUDA_SUCCESS) {
        fprintf(stderr, "launch failed\n"); return 6;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(h_dst, d_dst, (size_t)n_out * sizeof(float));

    int ok = (memcmp(h_dst, h_ref, (size_t)n_out * sizeof(float)) == 0);
    float mx = 0.0f; double sum = 0.0;
    for (long long i = 0; i < n_out; i++) {
        float dif = fabsf(h_dst[i] - h_ref[i]);
        if (dif > mx) mx = dif;
        sum += dif;
    }
    fprintf(stderr,
        "[test_pixel_shuffle_3d] C=%d D=%d H=%d W=%d  in=%lld out=%lld  "
        "max_abs=%.4g mean_abs=%.4g  %s (BIT-EXACT %s)\n",
        C, D, H, W, n_in, n_out, (double)mx, sum / (double)(n_out > 0 ? n_out : 1),
        ok ? "OK" : "FAIL", ok ? "yes" : "no");
    (void)verbose;

    free(h_src); free(h_ref); free(h_dst);
    cuMemFree(d_src); cuMemFree(d_dst);
    return ok ? 0 : 7;
}

int main(int argc, char **argv) {
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 3;
    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 3;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 3;

    CUmodule mod = 0;
    int sm = cu_compile_kernels(&mod, dev, cuda_sam3d_kernel_src,
                                "sam3d_kernels.cu", verbose, "test_pixel_shuffle_3d");
    if (sm < 0) return 4;
    CUfunction fn = NULL;
    if (cuModuleGetFunction(&fn, mod, "pixel_shuffle_3d_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "lookup pixel_shuffle_3d_f32 failed\n"); return 4;
    }

    int rc = 0;
    /* SS-VAE 16³ -> 32³ stage: input [1024,16,16,16] -> [128,32,32,32] */
    rc |= run_one(fn, /*C=*/128, /*D=*/16, /*H=*/16, /*W=*/16, verbose);
    /* SS-VAE 32³ -> 64³ stage: input [256, 32,32,32] -> [32, 64,64,64] */
    rc |= run_one(fn, /*C=*/32,  /*D=*/32, /*H=*/32, /*W=*/32, verbose);

    cuModuleUnload(mod);
    cuCtxDestroy(cu_ctx);
    return rc;
}
