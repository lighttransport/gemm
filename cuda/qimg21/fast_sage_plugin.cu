/* SageAttention2-style 8-bit attention for the fast Qwen-Image 2.1 denoiser
 * (pinned upstream SageAttention kernels, sm_89 code path, runs on sm_120):
 *
 * - K is smoothed by subtracting its per-channel mean over tokens (softmax is
 *   invariant to it), then Q (per 32-row warp) and K (per 64-row block) are
 *   quantized to INT8 per head;
 * - V is transposed, padded and quantized to FP8 E4M3 per channel;
 * - the kernel runs INT8 Q.K^T and FP8 P.V MMAs, applies the V scales in the
 *   epilogue, and writes BF16.
 *
 * GeForce Blackwell runs INT8 and FP8 MMAs at about four times the BF16 rate
 * with F32 accumulation, which bounds FlashAttention-2 here. Unmasked calls
 * only; causal prefix runs stay on FlashAttention-2. Not bit-comparable with
 * the reference: an opt-in speed backend for quantized presets. */
#include <cassert> /* cuda_fp8.hpp asserts; PyTorch headers normally include this */
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "qattn/qk_int_sv_f8_cuda_sm89.cuh"
#include "fused/fused_kernels.cuh"

namespace {

void *workspace;
size_t workspace_bytes;

size_t align256(size_t x) { return (x + 255) & ~(size_t)255; }

/* partial[chunk][c] = sum over a 128-row chunk of k[:, c]; the final pass
 * adds the chunks in order, so the mean (and the output) is deterministic. */
__global__ void kmean_partial(float *partial, const nv_bfloat16 *k, int nk, int cols, int stride) {
    int c = blockIdx.x * blockDim.x + threadIdx.x, r0 = blockIdx.y * 128;
    if (c >= cols) return;
    float s = 0.f;
    for (int r = r0; r < r0 + 128 && r < nk; r++) s += __bfloat162float(k[(size_t)r * stride + c]);
    partial[(size_t)blockIdx.y * cols + c] = s;
}

__global__ void kmean_final(nv_bfloat16 *mean, const float *partial, int chunks, int nk, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float s = 0.f;
    for (int i = 0; i < chunks; i++) s += partial[(size_t)i * cols + c];
    mean[c] = __float2bfloat16_rn(s / (float)nk);
}

template <bool Fp16Accum>
cudaError_t run(void *out, const void *q, const void *k, const void *v, int nq, int nk, int heads, int q_stride,
                int kv_stride, int o_stride, cudaStream_t stream) {
    constexpr int D = 128, CTA_Q = 128, CTA_K = 64, WARP_Q = 32, WARP_K = 64;
    const int cols = heads * D, pad = (nk + CTA_K - 1) / CTA_K * CTA_K;
    const int q_blocks = (nq + CTA_Q - 1) / CTA_Q * (CTA_Q / WARP_Q), k_blocks = (nk + CTA_K - 1) / CTA_K;
    const int chunks = (nk + 127) / 128;
    size_t off[9], total = 0;
    const size_t sizes[9] = {(size_t)nq * cols, (size_t)heads * q_blocks * 4, (size_t)nk * cols,
                             (size_t)heads * k_blocks * 4, (size_t)chunks * cols * 4, (size_t)cols * 2,
                             (size_t)cols * pad * 2, (size_t)cols * pad, (size_t)cols * 4};
    for (int i = 0; i < 9; i++) { off[i] = total; total += align256(sizes[i]); }
    if (total > workspace_bytes) {
        if (workspace) cudaFree(workspace);
        workspace = nullptr;
        workspace_bytes = 0;
        cudaError_t e = cudaMalloc(&workspace, total);
        if (e != cudaSuccess) return e;
        workspace_bytes = total;
    }
    char *w = static_cast<char *>(workspace);
    int8_t *q8 = reinterpret_cast<int8_t *>(w + off[0]), *k8 = reinterpret_cast<int8_t *>(w + off[2]);
    float *qs = reinterpret_cast<float *>(w + off[1]), *ks = reinterpret_cast<float *>(w + off[3]);
    float *ksum = reinterpret_cast<float *>(w + off[4]);
    nv_bfloat16 *kmean = reinterpret_cast<nv_bfloat16 *>(w + off[5]), *vt = reinterpret_cast<nv_bfloat16 *>(w + off[6]);
    int8_t *v8 = reinterpret_cast<int8_t *>(w + off[7]);
    float *vs = reinterpret_cast<float *>(w + off[8]);
    auto *qb = static_cast<nv_bfloat16 *>(const_cast<void *>(q));
    auto *kb = static_cast<nv_bfloat16 *>(const_cast<void *>(k));
    auto *vb = static_cast<nv_bfloat16 *>(const_cast<void *>(v));

    kmean_partial<<<dim3((cols + 255) / 256, chunks), 256, 0, stream>>>(ksum, kb, nk, cols, kv_stride);
    kmean_final<<<(cols + 255) / 256, 256, 0, stream>>>(kmean, ksum, chunks, nk, cols);
    QuantInt8Kernel<D, WARP_Q, 1, false, false, nv_bfloat16><<<dim3(q_blocks, heads, 1), WARP_Q * (D / 8), 0, stream>>>(
        qb, nullptr, q8, qs, 0.f, nq, 0, q_stride, D, 0, 0, 0, cols, D, 0, q_blocks);
    QuantInt8Kernel<D, CTA_K, 1, false, true, nv_bfloat16><<<dim3(k_blocks, heads, 1), CTA_K * (D / 8), 0, stream>>>(
        kb, kmean, k8, ks, 0.f, nk, 0, kv_stride, D, 0, D, 0, cols, D, 0, k_blocks);
    TransposePadPermuteKernel<D, CTA_K, true, nv_bfloat16><<<dim3(pad / CTA_K, heads, 1), CTA_K * (D / 8), 0, stream>>>(
        vb, vt, nk, 0, kv_stride, D, 0, heads * pad, pad);
    MeanScaleKernel<CTA_K, false, nv_bfloat16><<<dim3(heads, 1, D), 256, 0, stream>>>(
        vt, v8, nullptr, vs, Fp16Accum ? 2.25f : 448.0f, nk, 0, heads * pad, pad, 0, heads * pad, pad, 0, 0, 0, D);

    auto kernel = qk_int_sv_f8_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, D, DataType::kInt8, QuantGranularity::kPerWarp,
                                           QuantGranularity::kPerWarp, float, true, nv_bfloat16, ComputeUnit::kCudaCore,
                                           MaskMode::kNone, false, true, false, Fp16Accum>;
    const size_t smem = CTA_Q * D + CTA_K * D + CTA_K * D > CTA_Q * D * 2 ? CTA_Q * D + CTA_K * D + CTA_K * D
                                                                          : CTA_Q * D * 2;
    static bool configured;
    if (!configured) {
        cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return e;
        configured = true;
    }
    kernel<<<dim3((nq + CTA_Q - 1) / CTA_Q, heads, 1), dim3(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K)), smem, stream>>>(
        q8, k8, v8, static_cast<nv_bfloat16 *>(out), nullptr, qs, ks, vs, nullptr, nq, nk, 1, 0, cols, D, 0, cols, D,
        0, pad, heads * pad, 0, o_stride, D, 1.0f / sqrtf((float)D));
    return cudaGetLastError();
}

}  // namespace

/* Same contract as q21f_flash_attention (out[nq, heads*128], strides in
 * elements); mask must be 0. accum 1 (the runner default) uses the FP16
 * instruction accumulator upstream selects for sm_120 ("fp32+fp16", V scaled
 * to 2.25); accum 0 keeps P.V in F32 (upstream "fp32+fp32"). */
extern "C" int q21f_sage_attention(void *out, const void *q, const void *k, const void *v, int nq, int nk, int heads,
                                   int q_stride, int kv_stride, int o_stride, int mask, int accum,
                                   cudaStream_t stream) {
    if (!out || !q || !k || !v || nq <= 0 || nk <= 0 || heads <= 0 || mask != 0 || q_stride % 8 || kv_stride % 8 ||
        o_stride % 8 || (accum != 0 && accum != 1))
        return cudaErrorInvalidValue;
    return accum ? run<true>(out, q, k, v, nq, nk, heads, q_stride, kv_stride, o_stride, stream)
                 : run<false>(out, q, k, v, nq, nk, heads, q_stride, kv_stride, o_stride, stream);
}
