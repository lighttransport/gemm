#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void i16_gemm(const int16_t *A, const int16_t *B, int32_t *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    int32_t acc = 0;
    const int16_t *ar = A + (size_t)row * K;
    const int16_t *br = B + (size_t)col * K;
    for (int k = 0; k < K; k++) acc += (int32_t)ar[k] * (int32_t)br[k];
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
    int M = argc > 1 ? atoi(argv[1]) : 1024;
    int N = argc > 2 ? atoi(argv[2]) : 1024;
    int K = argc > 3 ? atoi(argv[3]) : 2048;
    int iters = argc > 4 ? atoi(argv[4]) : 20;
    size_t asz = (size_t)M * K * sizeof(int16_t);
    size_t bsz = (size_t)N * K * sizeof(int16_t);
    size_t csz = (size_t)M * N * sizeof(int32_t);
    int16_t *ha = (int16_t *)malloc(asz), *hb = (int16_t *)malloc(bsz);
    for (size_t i = 0; i < (size_t)M * K; i++) ha[i] = (int16_t)((int)(i * 13u + 7u) % 65535 - 32767);
    for (size_t i = 0; i < (size_t)N * K; i++) hb[i] = (int16_t)((int)(i * 17u + 3u) % 65535 - 32767);
    int16_t *da, *db;
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
    i16_gemm<<<grid, block>>>(da, db, dc, M, N, K);
    ck(cudaGetLastError(), "kernel warmup");
    ck(cudaDeviceSynchronize(), "sync warmup");
    ck(cudaEventRecord(t0), "record0");
    for (int i = 0; i < iters; i++) i16_gemm<<<grid, block>>>(da, db, dc, M, N, K);
    ck(cudaEventRecord(t1), "record1");
    ck(cudaEventSynchronize(t1), "sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
    double tops = (2.0 * (double)M * N * K * iters) / ((double)ms * 1e-3) / 1e12;
    printf("M=%d N=%d K=%d iters=%d time=%.3f ms int16_ops=%.3f TOPS\n", M, N, K, iters, ms, tops);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(ha);
    free(hb);
    return 0;
}
