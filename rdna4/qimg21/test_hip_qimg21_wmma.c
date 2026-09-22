/* Check the Qwen Image 2.1 gfx12 BF16 WMMA tile against an exact result. */
#include "hip_qimg21_runner.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    enum { M = 137, N = 153, K = 37 };
    uint16_t x[M * K], w[N * K];
    float y[M * N];
    for (int i = 0; i < M * K; ++i)
        x[i] = qimg_f32_to_bf16_rne((float)(i % 11 - 5) * 0.125f);
    for (int i = 0; i < N * K; ++i)
        w[i] = qimg_f32_to_bf16_rne((float)(i % 13 - 6) * 0.125f);
    cuda_qimg_runner *r = cuda_qimg_init(0, 1);
    if (!r || !r->gemm_bf16_wmma) {
        fprintf(stderr, "qimg21-wmma: WMMA kernel unavailable\n");
        cuda_qimg_free(r);
        return 1;
    }
    void *dx = checked_cuMemAlloc(sizeof(x));
    void *dw = checked_cuMemAlloc(sizeof(w));
    void *dy = checked_cuMemAlloc(sizeof(y));
    int rc = 1;
    if (!dx || !dw || !dy ||
        hipMemcpy(dx, x, sizeof(x), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(dw, w, sizeof(w), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemset(dy, 0, sizeof(y)) != hipSuccess) goto done;
    int m = M, n = N, k = K;
    void *args[] = {&dy, &dw, &dx, &n, &k, &m};
    hipError_t err = hipModuleLaunchKernel(r->gemm_bf16_wmma, (N + 127) / 128,
                                            (M + 127) / 128, 1, 256, 1, 1,
                                            0, r->stream, args, NULL);
    if (err != hipSuccess || hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(y, dy, sizeof(y), hipMemcpyDeviceToHost) != hipSuccess) goto done;
    int bad = 0;
    for (int row = 0; row < M; ++row) for (int col = 0; col < N; ++col) {
        float expected = 0.0f;
        for (int inner = 0; inner < K; ++inner)
            expected += (float)((row * K + inner) % 11 - 5) *
                        (float)((col * K + inner) % 13 - 6) * (0.125f * 0.125f);
        if (fabsf(y[row * N + col] - expected) > 1e-3f) ++bad;
    }
    fprintf(stderr, "qimg21-wmma: incorrect outputs=%d/%d\n", bad, M * N);
    rc = bad ? 1 : 0;
done:
    if (dx) hipFree(dx);
    if (dw) hipFree(dw);
    if (dy) hipFree(dy);
    cuda_qimg_free(r);
    return rc;
}
