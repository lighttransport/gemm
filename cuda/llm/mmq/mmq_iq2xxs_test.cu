/* Standalone dev harness for the IQ2_XXS MMQ port (cuda/llm MoE expert FFN).
 *
 * Goal: replace the per-expert "dequant IQ2_XXS -> F16 weights + tiny cuBLAS GEMM"
 * loop with a single quantized matmul that consumes the IQ2_XXS weights directly
 * (no F16 weight materialization) and int8 q8_1 activations, the way llama.cpp's
 * mmq.cuh / mul_mat_id does. This file de-risks the *data path* against an exact
 * CPU oracle before any tiling/mma or runner integration.
 *
 * Phases (this file = Phase 0 + Phase 1):
 *   P0  CPU oracle: exact IQ2_XXS dequant + F32 matmul (ground truth).
 *   P1  GPU MMQ (correctness-first): on-GPU q8_1 activation quant + on-GPU
 *       IQ2_XXS weight decode-to-int8 + dp4a dot, one thread per output. Validate
 *       vs oracle (int8-activation path => expect ~1e-2 rel_L2, like real MMQ).
 *   P2  (later) tile + mma.sync.s8.s8.s32 for sm_120 tensor cores.
 *   P3  (later) mul_mat_id expert compaction + integration into cuda_llm_runner.c.
 *
 * Build: make -C cuda/llm/mmq ; run: ./cuda/llm/mmq/mmq_iq2xxs_test
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "iq2xxs_tables.h"

#define QK_K 256          /* IQ2_XXS super-block elements */
#define IQ2XXS_BYTES 66   /* 2 (half d) + 32*2 (qs) */
#define CK 32             /* q8_1 / per-scale granularity */

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

/* ---------------- host half -> float ---------------- */
static float half_to_float_h(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, out;
    if (e == 0) {
        if (m == 0) out = s << 31;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff;
               out = (s << 31) | (e << 23) | (m << 13); }
    } else if (e == 0x1f) {
        out = (s << 31) | (0xff << 23) | (m << 13);
    } else {
        out = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f; __builtin_memcpy(&f, &out, 4); return f;
}

/* host float -> half (round to nearest, finite inputs only) */
static uint16_t float_to_half_h(float f) {
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t)s;             /* underflow -> 0 (fine for test scales) */
    if (e >= 0x1f) return (uint16_t)(s | 0x7bff); /* clamp to max finite half */
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

/* ---------------- IQ2_XXS block decode (host, exact) ----------------
 * Matches the validated in-tree dequant_iq2_xxs_triplet_to_f16 element order:
 * out[ib32*32 + l*8 + j], l in 0..3 (group), j in 0..7 (within grid u64). */
static void iq2xxs_decode_block_h(const uint8_t *bp, float *out) {
    float d = half_to_float_h(*(const uint16_t *)bp);
    const uint16_t *qs = (const uint16_t *)(bp + 2);
    for (int ib = 0; ib < 8; ib++) {
        uint32_t aux0 = (uint32_t)qs[4*ib] | ((uint32_t)qs[4*ib+1] << 16);
        uint32_t aux1 = (uint32_t)qs[4*ib+2] | ((uint32_t)qs[4*ib+3] << 16);
        float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            uint8_t idx = (uint8_t)((aux0 >> (8*l)) & 255);
            const uint8_t *g = (const uint8_t *)&iq2xxs_grid[idx];
            uint8_t s = ksigns_iq2xs[(aux1 >> (7*l)) & 127];
            for (int j = 0; j < 8; j++)
                out[ib*32 + l*8 + j] = db * (float)g[j] * ((s & (1 << j)) ? -1.0f : 1.0f);
        }
    }
}

/* Decode a full quantized row [K] (K/256 blocks) -> float[K]. */
static void iq2xxs_decode_row_h(const uint8_t *row, int K, float *out) {
    int nb = K / QK_K;
    for (int b = 0; b < nb; b++) iq2xxs_decode_block_h(row + b * IQ2XXS_BYTES, out + b * QK_K);
}

/* ---------------- device tables ---------------- */
__constant__ uint64_t c_grid[256];
__constant__ uint8_t  c_ksigns[128];

/* device half->float (used by kernel B) */
__device__ static float half_to_float_dev(uint16_t h) {
    return __half2float(*(const __half *)&h);
}

/* ---------------- P1 kernel A: quantize activations to q8_1 (per-32 scale) ----------------
 * X[M][K] f32 -> xq8[M][K] int8, xs[M][K/32] f32 scale. One block per (m, kb). */
__global__ void quantize_q8_1(const float *X, int8_t *xq8, float *xs, int M, int K) {
    int kb = blockIdx.x;            /* which 32-block */
    int m  = blockIdx.y;            /* which row */
    int t  = threadIdx.x;          /* 0..31 */
    int k  = kb * CK + t;
    float v = (m < M && k < K) ? X[(size_t)m * K + k] : 0.0f;
    float a = fabsf(v);
    /* warp max-abs */
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
    float scale = a / 127.0f;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    int q = (int)rintf(v * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    xq8[(size_t)m * K + k] = (int8_t)q;
    if (t == 0) xs[(size_t)m * (K / CK) + kb] = scale;
}

/* ---------------- P1 kernel B: MMQ IQ2_XXS x q8_1, one thread per (m,n) ----------------
 * W: IQ2_XXS quantized [N][K] (row_bytes = K/256 * 66). dst[M][N] = sum_k W[n][k]*X[m][k].
 * Decodes each weight 32-group to int8 on the fly, dp4a against q8_1 activations. */
__global__ void mmq_iq2xxs_naive(float *dst, const uint8_t *W, const int8_t *xq8,
                                  const float *xs, int M, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;   /* output col (weight row) */
    int m = blockIdx.y;                              /* output row (token) */
    if (n >= N || m >= M) return;
    int nb = K / QK_K;
    int row_bytes = nb * IQ2XXS_BYTES;
    const uint8_t *wrow = W + (size_t)n * row_bytes;
    const int8_t *xrow = xq8 + (size_t)m * K;
    const float *xsrow = xs + (size_t)m * (K / CK);
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *bp = wrow + b * IQ2XXS_BYTES;
        float d = half_to_float_dev(*(const uint16_t *)bp);
        const uint16_t *qs = (const uint16_t *)(bp + 2);
        for (int ib = 0; ib < 8; ib++) {           /* eight 32-sub-blocks */
            uint32_t aux0 = (uint32_t)qs[4*ib] | ((uint32_t)qs[4*ib+1] << 16);
            uint32_t aux1 = (uint32_t)qs[4*ib+2] | ((uint32_t)qs[4*ib+3] << 16);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            int kbase = b * QK_K + ib * CK;          /* global k of this 32-block */
            int sumi = 0;
            for (int l = 0; l < 4; l++) {            /* four 8-element grid groups */
                uint8_t idx = (uint8_t)((aux0 >> (8*l)) & 255);
                const uint8_t *g = (const uint8_t *)&c_grid[idx];
                uint8_t s = c_ksigns[(aux1 >> (7*l)) & 127];
                /* pack 4+4 signed int8 weights and dp4a against activations */
                int w0 = 0, w1 = 0, x0, x1;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wv = (int)g[j] * ((s & (1 << j)) ? -1 : 1);
                    w0 |= (wv & 0xff) << (8 * j);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wv = (int)g[4 + j] * ((s & (1 << (4 + j))) ? -1 : 1);
                    w1 |= (wv & 0xff) << (8 * j);
                }
                const int8_t *xp = xrow + kbase + l * 8;
                x0 = *(const int *)(xp);
                x1 = *(const int *)(xp + 4);
                sumi = __dp4a(w0, x0, sumi);
                sumi = __dp4a(w1, x1, sumi);
            }
            acc += db * xsrow[(kbase) / CK] * (float)sumi;
        }
    }
    dst[(size_t)m * N + n] = acc;
}

/* ---------------- CPU oracle ---------------- */
static void cpu_oracle(const uint8_t *W, const float *X, float *dst, int M, int N, int K) {
    float *wf = (float *)malloc((size_t)K * sizeof(float));
    for (int n = 0; n < N; n++) {
        iq2xxs_decode_row_h(W + (size_t)n * (K / QK_K) * IQ2XXS_BYTES, K, wf);
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
    int M = argc > 1 ? atoi(argv[1]) : 16;     /* tokens for one expert */
    int N = argc > 2 ? atoi(argv[2]) : 512;    /* expert_ff (gate/up out_dim) */
    int K = argc > 3 ? atoi(argv[3]) : 2048;   /* n_embd */
    printf("MMQ IQ2_XXS test: M=%d N=%d K=%d  (K/256=%d blocks/row)\n", M, N, K, K / QK_K);
    if (K % QK_K) { fprintf(stderr, "K must be multiple of 256\n"); return 1; }

    srand(1234);
    int nb = K / QK_K, row_bytes = nb * IQ2XXS_BYTES;
    size_t wbytes = (size_t)N * row_bytes;
    uint8_t *W = (uint8_t *)malloc(wbytes);
    for (size_t i = 0; i < wbytes; i++) W[i] = rand() & 0xff;      /* random qs (indices/signs/ls) */
    /* Overwrite each block's half scale d with a finite, realistic value (random bytes
     * could encode half-Inf/NaN; real GGUF d ~ small positive). */
    for (int n = 0; n < N; n++)
        for (int b = 0; b < nb; b++) {
            float d = 0.01f + (rand() / (float)RAND_MAX) * 0.05f;
            *(uint16_t *)(W + (size_t)n * row_bytes + b * IQ2XXS_BYTES) = float_to_half_h(d);
        }
    float *X = (float *)malloc((size_t)M * K * sizeof(float));
    for (int i = 0; i < M * K; i++) X[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 2.0f;

    float *ref = (float *)malloc((size_t)M * N * sizeof(float));
    cpu_oracle(W, X, ref, M, N, K);

    /* device */
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid, iq2xxs_grid, sizeof(iq2xxs_grid)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ksigns, ksigns_iq2xs, sizeof(ksigns_iq2xs)));
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
    dim3 bg((N + 127) / 128, M); mmq_iq2xxs_naive<<<bg, 128>>>(dDst, dW, dXq, dXs, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *out = (float *)malloc((size_t)M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(out, dDst, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));

    double e = rel_l2(out, ref, M * N);
    printf("P1 GPU MMQ vs CPU-F32 oracle: rel_L2 = %.6f  (int8-activation path; expect ~1e-2)\n", e);
    printf("  ref[0..3]=%.4f %.4f %.4f %.4f\n  gpu[0..3]=%.4f %.4f %.4f %.4f\n",
           ref[0], ref[1], ref[2], ref[3], out[0], out[1], out[2], out[3]);
    printf("%s\n", e < 0.05 ? "PASS (data path correct)" : "FAIL");
    return e < 0.05 ? 0 : 1;
}
