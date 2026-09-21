/* Exact native wrapper for the PyTorch 2.14 memory-efficient attention
 * specialization used by Qwen-Image 2.1 editing on sm_120.  The build pins
 * the matching PyTorch and CUTLASS source revisions; the resulting library
 * has no PyTorch runtime dependency. */
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

#include <cmath>
#include <cuda_runtime.h>

using q21_kernel = PyTorchMemEffAttention::AttentionKernel<
    cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>;

__global__ void __launch_bounds__(q21_kernel::kNumThreads,
                                  q21_kernel::kMinBlocksPerSm)
q21_cutlass_attention_kernel(typename q21_kernel::Params params) {
    if (!params.advance_to_block()) return;
    q21_kernel::attention_kernel(params);
}

__global__ void q21_cutlass_bf16_to_f32(float *out,
                                         const cutlass::bfloat16_t *in,
                                         int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = static_cast<float>(in[i]);
}

__device__ float q21_bf16_round(float value) {
    unsigned bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__global__ void q21_exact_qk_rope_kernel(float *q, float *k, const float *qw,
                                          const float *kw, const float *table,
                                          int tokens, int heads) {
    int token = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    if (token >= tokens || head >= heads || lane >= 32) return;
    int base = (token * heads + head) * 128;
    __shared__ float qsum[32], ksum[32];
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float x = q[base + lane * 4 + i];
        sq += x * x;
        x = k[base + lane * 4 + i];
        sk += x * x;
    }
    qsum[lane] = sq; ksum[lane] = sk;
    __syncthreads();
#pragma unroll
    for (int stride = 16; stride; stride >>= 1) {
        if (lane < stride) {
            qsum[lane] += qsum[lane + stride];
            ksum[lane] += ksum[lane + stride];
        }
        __syncthreads();
    }
    float qi = rsqrtf(qsum[0] / 128.0f + 1e-6f);
    float ki = rsqrtf(ksum[0] / 128.0f + 1e-6f);
#pragma unroll
    for (int pair_offset = 0; pair_offset < 4; pair_offset += 2) {
        int j = lane * 4 + pair_offset;
        float q0 = q21_bf16_round(q21_bf16_round(q[base+j] * qi) * qw[j]);
        float q1 = q21_bf16_round(q21_bf16_round(q[base+j+1] * qi) * qw[j+1]);
        float k0 = q21_bf16_round(q21_bf16_round(k[base+j] * ki) * kw[j]);
        float k1 = q21_bf16_round(q21_bf16_round(k[base+j+1] * ki) * kw[j+1]);
        float c = table[token * 128 + j], s = table[token * 128 + j + 1];
        q[base+j] = q21_bf16_round(__fmaf_rn(q0,c,-__fmul_rn(q1,s)));
        q[base+j+1] = q21_bf16_round(__fmaf_rn(q1,c,__fmul_rn(q0,s)));
        k[base+j] = q21_bf16_round(__fmaf_rn(k0,c,-__fmul_rn(k1,s)));
        k[base+j+1] = q21_bf16_round(__fmaf_rn(k1,c,__fmul_rn(k0,s)));
    }
}

extern "C" int q21_exact_qk_rope(float *q, float *k, const float *qw,
                                   const float *kw, const float *table,
                                   int tokens, int heads, cudaStream_t stream) {
    q21_exact_qk_rope_kernel<<<dim3(tokens, heads), 32, 0, stream>>>(
        q, k, qw, kw, table, tokens, heads);
    return cudaGetLastError();
}

extern "C" int q21_cutlass_attention(float *out, const void *q, const void *k,
                                      const void *v, int nq, int nk, int heads,
                                      int head_dim, cudaStream_t stream) {
    if (!out || !q || !k || !v || nq <= 0 || nk <= 0 || heads <= 0 ||
        head_dim != 128) return cudaErrorInvalidValue;
    typename q21_kernel::Params params;
    params.query_ptr = static_cast<const cutlass::bfloat16_t *>(q);
    params.key_ptr = static_cast<const cutlass::bfloat16_t *>(k);
    params.value_ptr = static_cast<const cutlass::bfloat16_t *>(v);
    cutlass::bfloat16_t *output = nullptr;
    float *lse = nullptr;
    int lse_queries = ((nq + q21_kernel::kAlignLSE - 1) /
                       q21_kernel::kAlignLSE) * q21_kernel::kAlignLSE;
    cudaError_t error = cudaMalloc(&output, static_cast<size_t>(nq) * heads *
                                   head_dim * sizeof(*output));
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&lse, static_cast<size_t>(lse_queries) * heads * sizeof(*lse));
    if (error != cudaSuccess) {
        cudaFree(output);
        return error;
    }
    params.output_ptr = output;
    params.logsumexp_ptr = lse;
    params.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    params.head_dim = params.head_dim_value = head_dim;
    params.num_queries = nq;
    params.num_keys = params.num_keys_absolute = nk;
    params.q_strideM = params.k_strideM = params.v_strideM = heads * head_dim;
    params.o_strideM = heads * head_dim;
    params.q_strideH = params.k_strideH = params.v_strideH = head_dim;
    params.q_strideB = static_cast<int64_t>(nq) * heads * head_dim;
    params.k_strideB = params.v_strideB = static_cast<int64_t>(nk) * heads * head_dim;
    params.num_batches = 1;
    params.num_heads = heads;
    params.q_heads_per_kv = 1;
    size_t shared_bytes = sizeof(typename q21_kernel::SharedStorage);
    error = cudaFuncSetAttribute(q21_cutlass_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    if (error == cudaSuccess) {
        dim3 grid((nq + q21_kernel::kQueriesPerBlock - 1) /
                  q21_kernel::kQueriesPerBlock, heads, 1);
        q21_cutlass_attention_kernel<<<grid, params.getThreadsGrid(), shared_bytes, stream>>>(params);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        int count = nq * heads * head_dim;
        q21_cutlass_bf16_to_f32<<<(count + 255) / 256, 256, 0, stream>>>(out, output, count);
        error = cudaGetLastError();
    }
    cudaError_t lse_error = cudaFree(lse);
    cudaError_t output_error = cudaFree(output);
    if (error != cudaSuccess) return error;
    return lse_error == cudaSuccess ? output_error : lse_error;
}

/* Text-only causal GQA entry point. Q is [N,32,128], K/V are [N,8,128]. */
extern "C" int q21_cutlass_text_attention(float *out, const void *q,
                                            const void *k, const void *v,
                                            int tokens, cudaStream_t stream) {
    if (!out || !q || !k || !v || tokens <= 0) return cudaErrorInvalidValue;
    constexpr int query_heads = 32, kv_heads = 8, head_dim = 128;
    typename q21_kernel::Params params;
    params.query_ptr = static_cast<const cutlass::bfloat16_t *>(q);
    params.key_ptr = static_cast<const cutlass::bfloat16_t *>(k);
    params.value_ptr = static_cast<const cutlass::bfloat16_t *>(v);
    cutlass::bfloat16_t *output = nullptr;
    float *lse = nullptr;
    int lse_queries = ((tokens + q21_kernel::kAlignLSE - 1) /
                       q21_kernel::kAlignLSE) * q21_kernel::kAlignLSE;
    cudaError_t error = cudaMalloc(&output, static_cast<size_t>(tokens) *
                                   query_heads * head_dim * sizeof(*output));
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&lse, static_cast<size_t>(lse_queries) * query_heads * sizeof(*lse));
    if (error != cudaSuccess) {
        cudaFree(output);
        return error;
    }
    params.output_ptr = output;
    params.logsumexp_ptr = lse;
    params.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    params.head_dim = params.head_dim_value = head_dim;
    params.num_queries = params.num_keys = params.num_keys_absolute = tokens;
    params.custom_mask_type = q21_kernel::CausalFromTopLeft;
    params.q_strideM = query_heads * head_dim;
    params.k_strideM = params.v_strideM = kv_heads * head_dim;
    params.o_strideM = query_heads * head_dim;
    params.q_strideH = params.k_strideH = params.v_strideH = head_dim;
    params.q_strideB = static_cast<int64_t>(tokens) * query_heads * head_dim;
    params.k_strideB = params.v_strideB = static_cast<int64_t>(tokens) * kv_heads * head_dim;
    params.num_batches = 1;
    params.num_heads = query_heads;
    params.q_heads_per_kv = query_heads / kv_heads;
    size_t shared_bytes = sizeof(typename q21_kernel::SharedStorage);
    error = cudaFuncSetAttribute(q21_cutlass_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    if (error == cudaSuccess) {
        dim3 grid((tokens + q21_kernel::kQueriesPerBlock - 1) /
                  q21_kernel::kQueriesPerBlock, query_heads, 1);
        q21_cutlass_attention_kernel<<<grid, params.getThreadsGrid(), shared_bytes, stream>>>(params);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        int count = tokens * query_heads * head_dim;
        q21_cutlass_bf16_to_f32<<<(count + 255) / 256, 256, 0, stream>>>(out, output, count);
        error = cudaGetLastError();
    }
    cudaError_t lse_error = cudaFree(lse);
    cudaError_t output_error = cudaFree(output);
    if (error != cudaSuccess) return error;
    return lse_error == cudaSuccess ? output_error : lse_error;
}
