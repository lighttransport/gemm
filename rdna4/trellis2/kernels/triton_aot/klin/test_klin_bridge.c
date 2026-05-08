/* Standalone validation for triton_klin_bridge.h
 *
 * Loads all 8 klin shapes, runs each on random BF16 inputs vs an F32 host
 * reference, and benches the launch.
 *
 * Build:
 *   gcc -O2 -I../.. -o test_klin_bridge test_klin_bridge.c \
 *     -L/opt/rocm/lib -lamdhip64 -lstdc++ -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <hip/hip_runtime.h>

#define TRITON_KLIN_BRIDGE_IMPL
#include "triton_klin_bridge.h"

static uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}
static float bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f; memcpy(&f, &u, 4); return f;
}

typedef struct { int M, K, N; const char *tag; } Shape;
static Shape SH[] = {
    {1905, 1024, 4096, "stage0_klin_up"},
    {1905, 4096, 1024, "stage0_klin_dn"},
    {8452,  512, 2048, "stage1_klin_up"},
    {8452, 2048,  512, "stage1_klin_dn"},
    {16384, 256, 1024, "stage2_klin_up"},
    {16384,1024,  256, "stage2_klin_dn"},
    {16384, 128,  512, "stage3_klin_up"},
    {16384, 512,  128, "stage3_klin_dn"},
};

int main(int argc, char **argv)
{
    const char *kdir = (argc > 1) ? argv[1] : "kernels";
    if (t2_klin_init(kdir) != 0) { fprintf(stderr, "init fail\n"); return 1; }

    srand(0xC0FFEE);
    int worst_err = 0;
    for (int s = 0; s < (int)(sizeof SH / sizeof SH[0]); s++) {
        int M = SH[s].M, K = SH[s].K, N = SH[s].N;
        size_t Xb = (size_t)M*K, Wb = (size_t)N*K, Yb = (size_t)M*N;
        uint16_t *hX = (uint16_t*)malloc(Xb*2);
        uint16_t *hW = (uint16_t*)malloc(Wb*2);
        float *hB = (float*)malloc(N*4);
        float *hYref = (float*)calloc(Yb, 4);
        float *hY = (float*)malloc(Yb*4);
        for (size_t i=0;i<Xb;i++) hX[i] = f32_to_bf16(((rand()/(float)RAND_MAX)-0.5f)*0.5f);
        for (size_t i=0;i<Wb;i++) hW[i] = f32_to_bf16(((rand()/(float)RAND_MAX)-0.5f)*0.5f);
        for (int i=0;i<N;i++) hB[i] = ((rand()/(float)RAND_MAX)-0.5f)*0.1f;

        /* host ref: only check first 16 rows × first 16 cols to stay fast */
        int ROWS = M < 16 ? M : 16;
        int COLS = N < 16 ? N : 16;
        for (int m=0; m<ROWS; m++)
        for (int n=0; n<COLS; n++) {
            float acc = 0;
            for (int k=0; k<K; k++)
                acc += bf16_to_f32(hX[(size_t)m*K+k]) * bf16_to_f32(hW[(size_t)n*K+k]);
            hYref[(size_t)m*N+n] = acc + hB[n];
        }

        void *dX, *dW, *dB, *dY;
        hipMalloc(&dX, Xb*2); hipMalloc(&dW, Wb*2);
        hipMalloc(&dB, N*4);  hipMalloc(&dY, Yb*4);
        hipMemcpy(dX, hX, Xb*2, hipMemcpyHostToDevice);
        hipMemcpy(dW, hW, Wb*2, hipMemcpyHostToDevice);
        hipMemcpy(dB, hB, N*4,  hipMemcpyHostToDevice);

        if (t2_klin_run(M, K, N, dX, dW, dB, dY, 0) != 0) {
            fprintf(stderr, "%s: run fail\n", SH[s].tag);
            worst_err = 1;
            goto next;
        }
        hipDeviceSynchronize();
        hipMemcpy(hY, dY, Yb*4, hipMemcpyDeviceToHost);

        double max_abs = 0, max_rel = 0;
        for (int m=0; m<ROWS; m++)
        for (int n=0; n<COLS; n++) {
            float r = hYref[(size_t)m*N+n], g = hY[(size_t)m*N+n];
            float ad = fabsf(g - r);
            float rd = ad / (fabsf(r) + 1e-6f);
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) max_rel = rd;
        }

        /* bench */
        for (int w=0; w<5; w++) t2_klin_run(M, K, N, dX, dW, dB, dY, 0);
        hipDeviceSynchronize();
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0, 0);
        int iters = 50;
        for (int i=0; i<iters; i++) t2_klin_run(M, K, N, dX, dW, dB, dY, 0);
        hipEventRecord(e1, 0); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= iters;
        double tflops = 2.0 * M * K * N / (ms * 1e9);
        printf("%-22s M=%-5d K=%-4d N=%-4d  ms=%6.3f  TF/s=%5.1f  max_abs=%.3g max_rel=%.3g\n",
               SH[s].tag, M, K, N, ms, tflops, max_abs, max_rel);
        if (max_rel > 0.05) worst_err = 1;
        hipEventDestroy(e0); hipEventDestroy(e1);
    next:
        hipFree(dX); hipFree(dW); hipFree(dB); hipFree(dY);
        free(hX); free(hW); free(hB); free(hYref); free(hY);
    }
    t2_klin_release();
    return worst_err;
}
