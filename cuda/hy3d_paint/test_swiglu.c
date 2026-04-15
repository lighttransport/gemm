/*
 * test_swiglu.c - Standalone NVRTC compile + launch test of
 * split_silu_gate_f32 from cuda_paint_nn_kernels.h.
 *
 * For an input H[rows, 2*half] the expected output is
 *   O[r, c] = silu(H[r, c]) * H[r, half + c]
 * where silu(x) = x / (1 + exp(-x)). We synthesise random inputs,
 * run the kernel, and compare against a host-side NumPy-ish reference.
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "cuda_paint_nn_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float urand(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return ((*s >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;  /* [-1, 1] */
}

int main(int argc, char **argv) {
    int rows = (argc >= 2) ? atoi(argv[1]) : 257;
    int half = (argc >= 3) ? atoi(argv[2]) : 4096;
    int full = 2 * half;
    size_t nin  = (size_t)rows * full;
    size_t nout = (size_t)rows * half;

    float *h_in  = (float *)malloc(nin  * sizeof(float));
    float *h_ref = (float *)malloc(nout * sizeof(float));
    float *h_got = (float *)malloc(nout * sizeof(float));
    uint32_t seed = 123;
    for (size_t i = 0; i < nin; i++) h_in[i] = urand(&seed);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < half; c++) {
            float x1 = h_in[(size_t)r * full + c];
            float x2 = h_in[(size_t)r * full + half + c];
            float silu = x1 / (1.0f + expf(-x1));
            h_ref[(size_t)r * half + c] = silu * x2;
        }
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    int sm = cu_compile_kernels(&mod, dev,
                                cuda_paint_nn_kernels_src,
                                "hy3d_paint_nn", 1, "HY3D-PAINT-NN");
    if (sm < 0) return 1;
    CUfunction fn;
    cuModuleGetFunction(&fn, mod, "split_silu_gate_f32");

    CUdeviceptr d_in, d_out;
    cuMemAlloc(&d_in,  nin  * sizeof(float));
    cuMemAlloc(&d_out, nout * sizeof(float));
    cuMemcpyHtoD(d_in, h_in, nin * sizeof(float));

    int rows_i = rows, half_i = half;
    void *args[] = { &d_in, &d_out, &rows_i, &half_i };
    unsigned grid = (unsigned)((nout + 255) / 256);
    cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL);
    cuCtxSynchronize();

    cuMemcpyDtoH(h_got, d_out, nout * sizeof(float));

    double max_err = 0.0, sum_err = 0.0;
    for (size_t i = 0; i < nout; i++) {
        double d = fabs((double)h_got[i] - (double)h_ref[i]);
        if (d > max_err) max_err = d;
        sum_err += d;
    }
    fprintf(stderr, "rows=%d half=%d  max=%.3e mean=%.3e\n",
            rows, half, max_err, sum_err / (double)nout);

    cuMemFree(d_in); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    free(h_in); free(h_ref); free(h_got);
    return 0;
}
