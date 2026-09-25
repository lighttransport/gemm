/* BF16 attention for the fast Qwen-Image 2.1 denoiser. It uses the same
 * PyTorch 2.14 memory-efficient kernel specialization as
 * cutlass_attention_plugin.cu (pinned sources, no PyTorch runtime), but reads
 * strided BF16 rows and writes BF16 output directly, and exposes the causal
 * masks needed to prefill the text/condition prefix in one call per segment. */
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

#include <cmath>
#include <cuda_runtime.h>

using q21f_kernel = PyTorchMemEffAttention::AttentionKernel<
    cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>;

__global__ void __launch_bounds__(q21f_kernel::kNumThreads,
                                  q21f_kernel::kMinBlocksPerSm)
q21f_attention_kernel(typename q21f_kernel::Params params) {
    if (!params.advance_to_block()) return;
    q21f_kernel::attention_kernel(params);
}

/* out[nq, heads*128] = softmax(q k^T / sqrt(128)) v for each head.
 * Row strides are in elements and must be multiples of 8. mask: 0 none,
 * 1 causal from top-left, 2 causal from bottom-right (query i of nq attends
 * keys 0..nk-nq+i). */
extern "C" int q21f_attention(void *out, const void *q, const void *k, const void *v,
                              int nq, int nk, int heads, int q_stride, int kv_stride,
                              int o_stride, int mask, cudaStream_t stream) {
    static bool configured = false;
    if (!out || !q || !k || !v || nq <= 0 || nk <= 0 || heads <= 0 || mask < 0 || mask > 2 ||
        q_stride < heads * 128 || kv_stride < heads * 128 || o_stride < heads * 128 ||
        q_stride % 8 || kv_stride % 8 || o_stride % 8)
        return cudaErrorInvalidValue;
    typename q21f_kernel::Params params;
    params.query_ptr = static_cast<const cutlass::bfloat16_t *>(q);
    params.key_ptr = static_cast<const cutlass::bfloat16_t *>(k);
    params.value_ptr = static_cast<const cutlass::bfloat16_t *>(v);
    params.output_ptr = static_cast<cutlass::bfloat16_t *>(out);
    params.logsumexp_ptr = nullptr;
    params.scale = static_cast<float>(1.0 / std::sqrt(128.0));
    params.head_dim = params.head_dim_value = 128;
    params.num_queries = nq;
    params.num_keys = params.num_keys_absolute = nk;
    params.custom_mask_type = mask == 1 ? q21f_kernel::CausalFromTopLeft
                            : mask == 2 ? q21f_kernel::CausalFromBottomRight
                                        : q21f_kernel::NoCustomMask;
    params.q_strideM = q_stride;
    params.k_strideM = params.v_strideM = kv_stride;
    params.o_strideM = o_stride;
    params.q_strideH = params.k_strideH = params.v_strideH = 128;
    params.q_strideB = static_cast<int64_t>(nq) * q_stride;
    params.k_strideB = params.v_strideB = static_cast<int64_t>(nk) * kv_stride;
    params.num_batches = 1;
    params.num_heads = heads;
    params.q_heads_per_kv = 1;
    size_t shared_bytes = sizeof(typename q21f_kernel::SharedStorage);
    cudaError_t error = cudaSuccess;
    if (!configured) {
        error = cudaFuncSetAttribute(q21f_attention_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(shared_bytes));
        if (error != cudaSuccess) return error;
        configured = true;
    }
    dim3 grid((nq + q21f_kernel::kQueriesPerBlock - 1) / q21f_kernel::kQueriesPerBlock, heads, 1);
    q21f_attention_kernel<<<grid, params.getThreadsGrid(), shared_bytes, stream>>>(params);
    return cudaGetLastError();
}
