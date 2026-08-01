#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void dp4a_gemm(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    int32_t acc = 0;
    const int *a4 = (const int *)(A + (size_t)row * K);
    const int *b4 = (const int *)(B + (size_t)col * K);
    for (int k = 0; k < K / 4; k++) {
        acc = __dp4a(a4[k], b4[k], acc);
    }
    C[(size_t)row * N + col] = acc;
}

static void ck(cudaError_t e, const char *what)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 2048;
    int N = argc > 2 ? atoi(argv[2]) : 2048;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    int iters = argc > 4 ? atoi(argv[4]) : 50;
    K = (K + 3) & ~3;
    size_t asz = (size_t)M * K, bsz = (size_t)N * K, csz = (size_t)M * N * sizeof(int32_t);
    int8_t *ha = (int8_t *)malloc(asz), *hb = (int8_t *)malloc(bsz);
    for (size_t i = 0; i < asz; i++) ha[i] = (int8_t)((int)(i * 13u + 7u) % 255 - 127);
    for (size_t i = 0; i < bsz; i++) hb[i] = (int8_t)((int)(i * 17u + 3u) % 255 - 127);
    int8_t *da, *db;
    int32_t *dc;
    ck(cudaMalloc(&da, asz), "cudaMalloc A");
    ck(cudaMalloc(&db, bsz), "cudaMalloc B");
    ck(cudaMalloc(&dc, csz), "cudaMalloc C");
    ck(cudaMemcpy(da, ha, asz, cudaMemcpyHostToDevice), "copy A");
    ck(cudaMemcpy(db, hb, bsz, cudaMemcpyHostToDevice), "copy B");
    dim3 block(16, 16), grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    cudaEvent_t t0, t1;
    ck(cudaEventCreate(&t0), "event0");
    ck(cudaEventCreate(&t1), "event1");
    dp4a_gemm<<<grid, block>>>(da, db, dc, M, N, K);
    ck(cudaGetLastError(), "kernel warmup");
    ck(cudaDeviceSynchronize(), "sync warmup");
    ck(cudaEventRecord(t0), "record0");
    for (int i = 0; i < iters; i++) dp4a_gemm<<<grid, block>>>(da, db, dc, M, N, K);
    ck(cudaEventRecord(t1), "record1");
    ck(cudaEventSynchronize(t1), "sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
    double tops = (2.0 * (double)M * N * K * iters) / ((double)ms * 1e-3) / 1e12;
    printf("M=%d N=%d K=%d iters=%d time=%.3f ms int8_ops=%.3f TOPS\n", M, N, K, iters, ms, tops);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(ha);
    free(hb);
    return 0;
}
