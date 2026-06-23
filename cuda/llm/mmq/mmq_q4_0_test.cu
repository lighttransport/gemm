/* Standalone dev harness for the Q4_0 MMQ port (cuda/llm dense FFN/QKV).
 *
 * The gemma-4-12B "Q4_K_XL" QAT model is actually 100% Q4_0. This replaces the
 * "dequant Q4_0 -> F16 + cuBLAS GEMM" prefill path with a single int8 tensor-core
 * matmul consuming Q4_0 weights directly + int8 q8_1 activations.
 *
 * Q4_0 block layout (18 bytes, 32 elements):
 *   d/half   offset 0 : block scale
 *   qs[16]   offset 2 : 32x4-bit values, element j = qs[j]&0xF (j<16),
 *                       element j+16 = qs[j]>>4
 *   weight[j] = (nibble[j] - 8) * d
 *
 * Q4_0 is SIMPLER than Q4_K: nibble-8 in [-8,7] fits signed int8, so it folds
 * straight into the int8 weight -> SINGLE m16n8k32 MMA (no plus/minus split, no
 * sub-scale). d is per-32 block; result += d * d8 * MMA(nibble-8, x_int8).
 *
 * Build: make -C cuda/llm/mmq mmq_q4_0_test ; run: ./cuda/llm/mmq/mmq_q4_0_test
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define Q40_BYTES 18   /* d(2) + qs[16] */
#define CK 32

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static float half_to_float_h(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, out;
    if (e == 0) {
        if (m == 0) out = s << 31;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff;
               out = (s << 31) | (e << 23) | (m << 13); }
    } else if (e == 0x1f) { out = (s << 31) | (0xff << 23) | (m << 13);
    } else { out = (s << 31) | ((e - 15 + 127) << 23) | (m << 13); }
    float f; __builtin_memcpy(&f, &out, 4); return f;
}
static uint16_t float_to_half_h(float f) {
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t)s;
    if (e >= 0x1f) return (uint16_t)(s | 0x7bff);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

/* ---------------- host Q4_0 block decode (canonical order) ---------------- */
static void q40_decode_block_h(const uint8_t *bp, float *out) {
    float d = half_to_float_h(*(const uint16_t *)(bp + 0));
    const uint8_t *qs = bp + 2;
    for (int j = 0; j < 16; j++) {
        out[j]      = (float)((int)(qs[j] & 0x0F) - 8) * d;
        out[j + 16] = (float)((int)(qs[j] >> 4)   - 8) * d;
    }
}
static void q40_decode_row_h(const uint8_t *row, int K, float *out) {
    int nb = K / CK;
    for (int b = 0; b < nb; b++) q40_decode_block_h(row + b * Q40_BYTES, out + b * CK);
}

__device__ static float half_to_float_dev(uint16_t h) { return __half2float(*(const __half *)&h); }

/* ---------------- kernel A: quantize activations to q8_1 ---------------- */
__global__ void quantize_q8_1(const float *X, int8_t *xq8, float *xs, int M, int K) {
    int kb = blockIdx.x, m = blockIdx.y, t = threadIdx.x, k = kb * CK + t;
    float v = (m < M && k < K) ? X[(size_t)m * K + k] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
    float scale = a / 127.0f;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q = (int)rintf(v * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    xq8[(size_t)m * K + k] = (int8_t)q;
    if (t == 0) xs[(size_t)m * (K / CK) + kb] = scale;
}

/* ---------------- kernel B: dp4a MMQ (one thread per (m,n)) ---------------- */
/* result = sum_block d * d8 * sum((nibble-8)*x_int8) */
__global__ void mmq_q4_0_naive(float *dst, const uint8_t *W, const int8_t *xq8,
                                const float *xs, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;
    int nb = K / CK, row_bytes = nb * Q40_BYTES;
    const uint8_t *wrow = W + (size_t)n * row_bytes;
    const int8_t *xrow = xq8 + (size_t)m * K;
    const float *xsrow = xs + (size_t)m * (K / CK);
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *bp = wrow + b * Q40_BYTES;
        float d = half_to_float_dev(*(const uint16_t *)(bp + 0));
        const uint8_t *qs = bp + 2;
        float d8 = xsrow[b];
        int s = 0;
        for (int k = 0; k < 32; k++) {
            int nib = (k < 16) ? (qs[k] & 0xF) : (qs[k - 16] >> 4);
            s += (nib - 8) * (int)xrow[b * 32 + k];
        }
        acc += d * d8 * (float)s;
    }
    dst[(size_t)m * N + n] = acc;
}

static __device__ __forceinline__ int pack4(const int8_t *p) {
    return (p[0] & 0xff) | ((p[1] & 0xff) << 8) | ((p[2] & 0xff) << 16) | ((p[3] & 0xff) << 24);
}

/* ---------------- PHASE 2: single-warp tensor-core MMA ---------------- */
__global__ void mmq_q4_0_mma(float *dst, const uint8_t *W, const int8_t *xq8,
                              const float *xs, int M, int N, int K) {
    int lane = threadIdx.x, gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * 16, m0 = blockIdx.y * 8;
    int nb = K / CK, row_bytes = nb * Q40_BYTES, nsb = K / CK;

    __shared__ int8_t sW[16][CK];
    __shared__ float  sWs_d[16];
    __shared__ int8_t sX[8][CK];
    __shared__ float  sXs[8];

    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    for (int sb = 0; sb < nsb; sb++) {
        for (int i = lane; i < 8 * CK; i += 32) {
            int t = i / CK, kk = i % CK, m = m0 + t;
            sX[t][kk] = (m < M) ? xq8[(size_t)m * K + sb * CK + kk] : 0;
        }
        if (lane < 8) { int m = m0 + lane; sXs[lane] = (m < M) ? xs[(size_t)m * nsb + sb] : 0.0f; }
        if (lane < 16) {
            int r = lane, n = n0 + r;
            const uint8_t *bp = W + (size_t)n * row_bytes + (size_t)sb * Q40_BYTES;
            sWs_d[r] = half_to_float_dev(*(const uint16_t *)(bp + 0));
            const uint8_t *qs = bp + 2;
            for (int k = 0; k < 32; k++) {
                int nib = (k < 16) ? (qs[k] & 0xF) : (qs[k - 16] >> 4);
                sW[r][k] = (int8_t)(nib - 8);
            }
        }
        __syncwarp();
        int a0 = pack4(&sW[gid][tid*4]),   a1 = pack4(&sW[gid+8][tid*4]);
        int a2 = pack4(&sW[gid][tid*4+16]), a3 = pack4(&sW[gid+8][tid*4+16]);
        int b0 = pack4(&sX[gid][tid*4]), b1 = pack4(&sX[gid][tid*4+16]);
        int c0=0, c1=0, c2=0, c3=0;
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "r"(0), "r"(0), "r"(0), "r"(0));
        float d0 = sWs_d[gid], d8r = sWs_d[gid+8];
        float xc0 = sXs[tid*2], xc1 = sXs[tid*2+1];
        f0 += d0  * xc0 * (float)c0;
        f1 += d0  * xc1 * (float)c1;
        f2 += d8r * xc0 * (float)c2;
        f3 += d8r * xc1 * (float)c3;
        __syncwarp();
    }
    int n_a = n0 + gid, n_b = n0 + gid + 8;
    int m_a = m0 + tid*2, m_b = m0 + tid*2 + 1;
    if (m_a < M) { dst[(size_t)m_a * N + n_a] = f0; dst[(size_t)m_a * N + n_b] = f2; }
    if (m_b < M) { dst[(size_t)m_b * N + n_a] = f1; dst[(size_t)m_b * N + n_b] = f3; }
}

/* ---------------- PHASE 2.6: decode-amortized multi-warp MMA (production shape) ---- */
#define WN 4
#define TG 4
__global__ void mmq_q4_0_mma_v3(float *dst, const uint8_t *W, const int8_t *xq8,
                                  const float *xs, int M, int N, int K) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane >> 2, tid = lane & 3;
    int n0 = blockIdx.x * (16 * WN) + warp * 16;
    int m_base = blockIdx.y * (8 * TG);
    int ntg = (M - m_base + 7) / 8; if (ntg > TG) ntg = TG; if (ntg < 0) ntg = 0;
    int nb = K / CK, row_bytes = nb * Q40_BYTES, nsb = K / CK;

    __shared__ int8_t sW[16 * WN][CK];
    __shared__ float  sWs_d[16 * WN];
    __shared__ int8_t sX[8 * TG][CK];
    __shared__ float  sXs[8 * TG];

    float f[TG][4];
    for (int g = 0; g < TG; g++) { f[g][0]=f[g][1]=f[g][2]=f[g][3]=0; }

    for (int sb = 0; sb < nsb; sb++) {
        if (lane < 16) {
            int r = warp*16 + lane, n = n0 + lane;
            const uint8_t *bp = W + (size_t)n*row_bytes + (size_t)sb*Q40_BYTES;
            sWs_d[r] = half_to_float_dev(*(const uint16_t*)(bp + 0));
            const uint8_t *qs = bp + 2;
            for (int k = 0; k < 32; k++) {
                int nib = (k < 16) ? (qs[k] & 0xF) : (qs[k - 16] >> 4);
                sW[r][k] = (int8_t)(nib - 8);
            }
        }
        for (int i = threadIdx.x; i < ntg*8*CK; i += blockDim.x) {
            int t = i/CK, kk = i%CK, m = m_base + t;
            sX[t][kk] = (m < M) ? xq8[(size_t)m*K + sb*CK + kk] : 0;
        }
        for (int t = threadIdx.x; t < ntg*8; t += blockDim.x) {
            int m = m_base + t; sXs[t] = (m < M) ? xs[(size_t)m*nsb + sb] : 0.0f;
        }
        __syncthreads();
        int wr = warp*16;
        int a0 = pack4(&sW[wr+gid][tid*4]),   a1 = pack4(&sW[wr+gid+8][tid*4]);
        int a2 = pack4(&sW[wr+gid][tid*4+16]), a3 = pack4(&sW[wr+gid+8][tid*4+16]);
        float d0 = sWs_d[wr+gid], d8r = sWs_d[wr+gid+8];
        for (int g = 0; g < ntg; g++) {
            int b0 = pack4(&sX[g*8+gid][tid*4]), b1 = pack4(&sX[g*8+gid][tid*4+16]);
            int c0=0, c1=0, c2=0, c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                 "r"(0),"r"(0),"r"(0),"r"(0));
            float xc0 = sXs[g*8+tid*2], xc1 = sXs[g*8+tid*2+1];
            f[g][0] += d0  * xc0 * (float)c0;
            f[g][1] += d0  * xc1 * (float)c1;
            f[g][2] += d8r * xc0 * (float)c2;
            f[g][3] += d8r * xc1 * (float)c3;
        }
        __syncthreads();
    }
    int n_a = n0+gid, n_b = n0+gid+8;
    for (int g = 0; g < ntg; g++) {
        int m_a = m_base + g*8 + tid*2, m_b = m_a + 1;
        if (m_a < M) { dst[(size_t)m_a*N+n_a]=f[g][0]; dst[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b < M) { dst[(size_t)m_b*N+n_a]=f[g][1]; dst[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

static void cpu_oracle(const uint8_t *W, const float *X, float *dst, int M, int N, int K) {
    float *wf = (float *)malloc((size_t)K * sizeof(float));
    for (int n = 0; n < N; n++) {
        q40_decode_row_h(W + (size_t)n * (K / CK) * Q40_BYTES, K, wf);
        for (int m = 0; m < M; m++) {
            const float *x = X + (size_t)m * K;
            double s = 0.0;
            for (int k = 0; k < K; k++) s += (double)wf[k] * (double)x[k];
            dst[(size_t)m * N + n] = (float)s;
        }
    }
    free(wf);
}
static double rel_l2(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 8;
    int N = argc > 2 ? atoi(argv[2]) : 512;
    int K = argc > 3 ? atoi(argv[3]) : 2048;
    printf("MMQ Q4_0 test: M=%d N=%d K=%d  (K/32=%d blocks/row)\n", M, N, K, K / CK);
    if (K % CK) { fprintf(stderr, "K must be multiple of 32\n"); return 1; }

    srand(1234);
    int nb = K / CK, row_bytes = nb * Q40_BYTES;
    size_t wbytes = (size_t)N * row_bytes;
    uint8_t *W = (uint8_t *)malloc(wbytes);
    for (size_t i = 0; i < wbytes; i++) W[i] = rand() & 0xff;
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            float d = 0.005f + (rand() / (float)RAND_MAX) * 0.05f;
            *(uint16_t *)(W + (size_t)n * row_bytes + b * Q40_BYTES + 0) = float_to_half_h(d);
        }
    float *X = (float *)malloc((size_t)M * K * sizeof(float));
    for (int i = 0; i < M * K; i++) X[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 2.0f;

    float *ref = (float *)malloc((size_t)M * N * sizeof(float));
    cpu_oracle(W, X, ref, M, N, K);

    uint8_t *dW; int8_t *dXq; float *dX, *dXs, *dDst;
    CUDA_CHECK(cudaMalloc(&dW, wbytes));
    CUDA_CHECK(cudaMalloc(&dX, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dXq, (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&dXs, (size_t)M * (K / CK) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDst, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW, W, wbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX, X, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));

    dim3 qg(K / CK, M); quantize_q8_1<<<qg, 32>>>(dX, dXq, dXs, M, K);
    CUDA_CHECK(cudaGetLastError());
    dim3 bg((N + 127) / 128, M); mmq_q4_0_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    float *out = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double e = rel_l2(out, ref, M * N);
    printf("P1 dp4a   vs CPU-F32 oracle: rel_L2 = %.6f\n", e);

    if (N % 16 != 0) { fprintf(stderr, "P2 needs N %% 16 == 0\n"); return 1; }
    CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
    dim3 mg(N / 16, (M + 7) / 8); mmq_q4_0_mma<<<mg, 32>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    float *out2 = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out2, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    double e2 = rel_l2(out2, ref, M * N), e2v1 = rel_l2(out2, out, M * N);
    printf("P2 mma    vs CPU-F32 oracle: rel_L2 = %.6f\n", e2);
    printf("P2 mma    vs P1 dp4a       : rel_L2 = %.6f\n", e2v1);
    printf("  ref[0..3]=%.4f %.4f %.4f %.4f\n  mma[0..3]=%.4f %.4f %.4f %.4f\n",
           ref[0], ref[1], ref[2], ref[3], out2[0], out2[1], out2[2], out2[3]);

    double e4 = 1e9, e4v = 1e9;
    if (N % (16 * WN) == 0) {
        CUDA_CHECK(cudaMemset(dDst, 0, (size_t)M * N * sizeof(float)));
        dim3 v3g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
        mmq_q4_0_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *o4 = (float *)malloc((size_t)M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(o4, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
        e4 = rel_l2(o4, ref, M*N); e4v = rel_l2(o4, out2, M*N);
        printf("P2.6 mma_v3 vs CPU-F32 oracle: rel_L2 = %.6f\n", e4);
        printf("P2.6 mma_v3 vs P2 mma         : rel_L2 = %.6f\n", e4v);
        free(o4);
    } else printf("P2.6 skipped (N %% %d != 0)\n", 16 * WN);

    int ok = (e < 0.05) && (e2 < 0.05) && (e2v1 < 1e-4) && (e4 < 0.05) && (e4v < 1e-4);
    printf("%s\n", ok ? "PASS (P1..P2.6 data path correct)" : "FAIL");

    {
        int iters = 300;
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        dim3 bgp((N + 127) / 128, M);
        cudaEventRecord(t0);
        for (int i = 0; i < iters; i++) mmq_q4_0_naive<<<bgp, 128>>>(dDst, dW, dXq, dXs, M, N, K);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms_dp; cudaEventElapsedTime(&ms_dp, t0, t1); ms_dp /= iters;
        float ms_v3 = 0;
        if (N % (16 * WN) == 0) {
            dim3 v3g(N / (16 * WN), (M + 8*TG - 1) / (8*TG));
            cudaEventRecord(t0);
            for (int i = 0; i < iters; i++) mmq_q4_0_mma_v3<<<v3g, 32 * WN>>>(dDst, dW, dXq, dXs, M, N, K);
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms_v3, t0, t1); ms_v3 /= iters;
        }
        double flop = 2.0 * M * N * K;
        printf("\nperf (1 GEMM):\n  dp4a   : %.3f ms  %.1f GFLOP/s\n", ms_dp, flop / (ms_dp * 1e6));
        if (ms_v3 > 0)
            printf("  mma_v3 : %.3f ms  %.1f GFLOP/s  (%.2fx vs dp4a)\n",
                   ms_v3, flop / (ms_v3 * 1e6), ms_dp / ms_v3);
    }
    return ok ? 0 : 1;
}
