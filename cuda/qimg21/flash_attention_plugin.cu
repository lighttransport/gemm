/* Native wrapper for the FlashAttention specialization selected by PyTorch
 * SDPA for Qwen3-VL text attention on sm_120.  Q is [N,32,128], K/V are
 * [N,8,128], and the top-left causal mask is always enabled. */
#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_SOFTCAP
#define UNFUSE_FMA

#include <cstring>
#include "csrc/flash_attn/src/flash.h"
#include "csrc/flash_attn/src/kernel_traits.h"
#include "csrc/flash_attn/src/flash_fwd_kernel.h"

using q21_text_traits = Flash_fwd_kernel_traits<128, 128, 64, 4, false,
                                                  false, cutlass::bfloat16_t>;

__global__ void q21_text_flash_kernel(const FLASH_NAMESPACE::Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_attn<q21_text_traits, false, true, false, false,
                                  false, true, false, false>(params);
}

__global__ void q21_text_bf16_to_f32(float *out,
                                      const cutlass::bfloat16_t *in,
                                      int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = static_cast<float>(in[i]);
}

extern "C" int q21_flash_text_attention(float *out, const void *q,
                                          const void *k, const void *v,
                                          int tokens, cudaStream_t stream) {
    if (!out || !q || !k || !v || tokens <= 0) return cudaErrorInvalidValue;
    constexpr int query_heads = 32, kv_heads = 8, head_dim = 128;
    FLASH_NAMESPACE::Flash_fwd_params p;
    std::memset(&p, 0, sizeof(p));
    p.q_ptr = const_cast<void *>(q);
    p.k_ptr = const_cast<void *>(k);
    p.v_ptr = const_cast<void *>(v);
    cutlass::bfloat16_t *out_bf16 = nullptr;
    float *softmax_lse = nullptr;
    cudaError_t error = cudaMalloc(&out_bf16, static_cast<size_t>(tokens) *
                                   query_heads * head_dim * sizeof(*out_bf16));
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&softmax_lse, static_cast<size_t>(tokens) * query_heads * sizeof(float));
    if (error != cudaSuccess) {
        cudaFree(out_bf16);
        return error;
    }
    p.o_ptr = out_bf16;
    p.softmax_lse_ptr = softmax_lse;
    p.q_row_stride = query_heads * head_dim;
    p.k_row_stride = p.v_row_stride = kv_heads * head_dim;
    p.o_row_stride = query_heads * head_dim;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = head_dim;
    p.o_head_stride = head_dim;
    p.q_batch_stride = static_cast<int64_t>(tokens) * query_heads * head_dim;
    p.k_batch_stride = p.v_batch_stride = static_cast<int64_t>(tokens) * kv_heads * head_dim;
    p.o_batch_stride = p.q_batch_stride;
    p.b = 1;
    p.h = query_heads;
    p.h_k = kv_heads;
    p.h_h_k_ratio = query_heads / kv_heads;
    p.seqlen_q = p.seqlen_k = tokens;
    p.d = p.d_rounded = head_dim;
    p.seqlen_q_rounded = p.seqlen_k_rounded = 128;
    p.scale_softmax = static_cast<float>(1.0 / sqrt(static_cast<double>(head_dim)));
    p.scale_softmax_log2 = p.scale_softmax * static_cast<float>(M_LOG2E);
    p.p_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0f;
    p.window_size_left = -1;
    p.window_size_right = 0;
    auto kernel = &q21_text_flash_kernel;
    error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 q21_text_traits::kSmemSize);
    if (error == cudaSuccess) {
        dim3 grid((tokens + q21_text_traits::kBlockM - 1) /
                  q21_text_traits::kBlockM, 1, query_heads);
        kernel<<<grid, q21_text_traits::kNThreads, q21_text_traits::kSmemSize, stream>>>(p);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        int count = tokens * query_heads * head_dim;
        q21_text_bf16_to_f32<<<(count + 255) / 256, 256, 0, stream>>>(out, out_bf16, count);
        error = cudaGetLastError();
    }
    cudaError_t lse_error = cudaFree(softmax_lse);
    cudaError_t output_error = cudaFree(out_bf16);
    if (error != cudaSuccess) return error;
    return lse_error == cudaSuccess ? output_error : lse_error;
}
