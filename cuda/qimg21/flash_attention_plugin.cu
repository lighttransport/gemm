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
using q21_vision_traits = Flash_fwd_kernel_traits<96, 128, 64, 4, false,
                                                    false, cutlass::bfloat16_t>;

__global__ void q21_text_flash_kernel(const FLASH_NAMESPACE::Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_attn<q21_text_traits, false, true, false, false,
                                  false, true, false, false>(params);
}

__global__ void q21_vision_flash_kernel(const FLASH_NAMESPACE::Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_attn<q21_vision_traits, false, false, false, false,
                                  false, false, false, false>(params);
}

__global__ void q21_text_bf16_to_f32(float *out,
                                      const cutlass::bfloat16_t *in,
                                      int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = static_cast<float>(in[i]);
}

struct q21_ln_stat {
    float mean, var, count;
};

__device__ q21_ln_stat q21_ln_add(q21_ln_stat a, float value) {
    float delta = value - a.mean;
    float count = a.count + 1.0f;
    float mean = a.mean + delta * (1.0f / count);
    a.var = a.var + delta * (value - mean);
    a.mean = mean;
    a.count = count;
    return a;
}

__device__ q21_ln_stat q21_ln_combine(q21_ln_stat data_b, q21_ln_stat data_a) {
    float count = data_a.count + data_b.count;
    if (count <= 0.0f) return {0.0f, 0.0f, 0.0f};
    float coefficient = 1.0f / count;
    float n_a = data_a.count * coefficient;
    float n_b = data_b.count * coefficient;
    float delta = data_b.mean - data_a.mean;
    return {n_a * data_a.mean + n_b * data_b.mean,
            data_a.var + data_b.var + delta * delta * data_a.count * n_b,
            count};
}

__device__ float q21_round_bf16(float value) {
    unsigned bits = __float_as_uint(value);
    return __uint_as_float((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
}

__global__ void q21_vision_layer_norm_kernel(float *out, const float *input,
                                              const float *weight, const float *bias,
                                              int width) {
    int row = blockIdx.x, thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
    __shared__ float shared[12];
    q21_ln_stat stat{0.0f, 0.0f, 0.0f};
    int vectors = width / 4;
    for (int vector = thread; vector < vectors; vector += blockDim.x) {
        int column = vector * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) stat = q21_ln_add(stat, input[row * width + column + i]);
    }
    for (int offset = 16; offset; offset >>= 1) {
        q21_ln_stat other{__shfl_down_sync(0xffffffff, stat.mean, offset),
                          __shfl_down_sync(0xffffffff, stat.var, offset),
                          __shfl_down_sync(0xffffffff, stat.count, offset)};
        stat = q21_ln_combine(stat, other);
    }
    for (int offset = blockDim.x / 64; offset; offset >>= 1) {
        if (lane == 0 && warp >= offset && warp < 2 * offset) {
            int index = warp - offset;
            shared[2 * index] = stat.mean;
            shared[2 * index + 1] = stat.var;
            shared[8 + index] = stat.count;
        }
        __syncthreads();
        if (lane == 0 && warp < offset) {
            q21_ln_stat other{shared[2 * warp], shared[2 * warp + 1], shared[8 + warp]};
            stat = q21_ln_combine(stat, other);
        }
        __syncthreads();
    }
    if (thread == 0) {
        shared[0] = stat.mean;
        shared[1] = stat.var / static_cast<float>(width);
    }
    __syncthreads();
    float mean = shared[0], inverse = rsqrtf(shared[1] + 1.0e-6f);
    for (int column = thread; column < width; column += blockDim.x) {
        float value = weight[column] * (inverse * (input[row * width + column] - mean)) + bias[column];
        out[row * width + column] = q21_round_bf16(value);
    }
}

__global__ void q21_vision_layer_norm_stats_kernel(float *out, const float *input,
                                                    int width) {
    int row = blockIdx.x, thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
    __shared__ float shared[12];
    q21_ln_stat stat{0.0f, 0.0f, 0.0f};
    int vectors = width / 4;
    for (int vector = thread; vector < vectors; vector += blockDim.x) {
        int column = vector * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) stat = q21_ln_add(stat, input[row * width + column + i]);
    }
    for (int offset = 16; offset; offset >>= 1) {
        q21_ln_stat other{__shfl_down_sync(0xffffffff, stat.mean, offset),
                          __shfl_down_sync(0xffffffff, stat.var, offset),
                          __shfl_down_sync(0xffffffff, stat.count, offset)};
        stat = q21_ln_combine(stat, other);
    }
    for (int offset = blockDim.x / 64; offset; offset >>= 1) {
        if (lane == 0 && warp >= offset && warp < 2 * offset) {
            int index = warp - offset;
            shared[2 * index] = stat.mean;
            shared[2 * index + 1] = stat.var;
            shared[8 + index] = stat.count;
        }
        __syncthreads();
        if (lane == 0 && warp < offset) {
            q21_ln_stat other{shared[2 * warp], shared[2 * warp + 1], shared[8 + warp]};
            stat = q21_ln_combine(stat, other);
        }
        __syncthreads();
    }
    if (thread == 0) {
        out[2 * row] = stat.mean;
        out[2 * row + 1] = rsqrtf(stat.var / static_cast<float>(width) + 1.0e-6f);
    }
}

extern "C" int q21_flash_vision_layer_norm(float *out, const float *input,
                                             const float *weight, const float *bias,
                                             int rows, int width, cudaStream_t stream) {
    if (!out || !input || !weight || !bias || rows <= 0 || width <= 0 || width % 4)
        return cudaErrorInvalidValue;
    q21_vision_layer_norm_kernel<<<rows, 256, 0, stream>>>(out, input, weight, bias, width);
    return cudaGetLastError();
}

extern "C" int q21_flash_vision_layer_norm_pytorch(float *out, const float *input,
                                                     const float *weight, const float *bias,
                                                     int rows, int width,
                                                     cudaStream_t stream) {
    if (!out || !input || !weight || !bias || rows <= 0 || width <= 0 || width % 4)
        return cudaErrorInvalidValue;
    q21_vision_layer_norm_kernel<<<rows, 128, 0, stream>>>(out, input, weight, bias, width);
    return cudaGetLastError();
}

extern "C" int q21_flash_vision_layer_norm_stats(float *out, const float *input,
                                                   int rows, int width,
                                                   cudaStream_t stream) {
    if (!out || !input || rows <= 0 || width <= 0 || width % 4)
        return cudaErrorInvalidValue;
    q21_vision_layer_norm_stats_kernel<<<rows, 256, 0, stream>>>(out, input, width);
    return cudaGetLastError();
}

extern "C" int q21_flash_vision_layer_norm_stats_pytorch(float *out,
                                                           const float *input,
                                                           int rows, int width,
                                                           cudaStream_t stream) {
    if (!out || !input || rows <= 0 || width <= 0 || width % 4)
        return cudaErrorInvalidValue;
    q21_vision_layer_norm_stats_kernel<<<rows, 128, 0, stream>>>(out, input, width);
    return cudaGetLastError();
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

extern "C" int q21_flash_vision_attention(float *out, const void *qkv,
                                            int tokens, cudaStream_t stream) {
    if (!out || !qkv || tokens <= 0) return cudaErrorInvalidValue;
    constexpr int heads = 16, head_dim = 72, packed_dim = 3 * heads * head_dim;
    const auto *packed = static_cast<const cutlass::bfloat16_t *>(qkv);
    cutlass::bfloat16_t *out_bf16 = nullptr;
    float *softmax_lse = nullptr;
    cudaError_t error = cudaMalloc(&out_bf16, static_cast<size_t>(tokens) *
                                   heads * head_dim * sizeof(*out_bf16));
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&softmax_lse, static_cast<size_t>(tokens) * heads * sizeof(float));
    if (error != cudaSuccess) {
        cudaFree(out_bf16);
        return error;
    }

    FLASH_NAMESPACE::Flash_fwd_params p;
    std::memset(&p, 0, sizeof(p));
    p.q_ptr = const_cast<cutlass::bfloat16_t *>(packed);
    p.k_ptr = const_cast<cutlass::bfloat16_t *>(packed + heads * head_dim);
    p.v_ptr = const_cast<cutlass::bfloat16_t *>(packed + 2 * heads * head_dim);
    p.o_ptr = out_bf16;
    p.softmax_lse_ptr = softmax_lse;
    p.q_row_stride = p.k_row_stride = p.v_row_stride = packed_dim;
    p.o_row_stride = heads * head_dim;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = head_dim;
    p.o_head_stride = head_dim;
    p.q_batch_stride = p.k_batch_stride = p.v_batch_stride =
        static_cast<int64_t>(tokens) * packed_dim;
    p.o_batch_stride = static_cast<int64_t>(tokens) * heads * head_dim;
    p.b = 1;
    p.h = p.h_k = heads;
    p.h_h_k_ratio = 1;
    p.seqlen_q = p.seqlen_k = tokens;
    p.seqlen_q_rounded = ((tokens + 127) / 128) * 128;
    p.seqlen_k_rounded = ((tokens + 127) / 128) * 128;
    p.d = head_dim;
    p.d_rounded = 96;
    p.scale_softmax = static_cast<float>(1.0 / sqrt(static_cast<double>(head_dim)));
    p.scale_softmax_log2 = p.scale_softmax * static_cast<float>(M_LOG2E);
    p.p_dropout = p.rp_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.window_size_left = p.window_size_right = -1;
    p.is_bf16 = true;

    auto kernel = &q21_vision_flash_kernel;
    error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 q21_vision_traits::kSmemSize);
    if (error == cudaSuccess) {
        dim3 grid((tokens + q21_vision_traits::kBlockM - 1) /
                  q21_vision_traits::kBlockM, 1, heads);
        kernel<<<grid, q21_vision_traits::kNThreads,
                 q21_vision_traits::kSmemSize, stream>>>(p);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        int count = tokens * heads * head_dim;
        q21_text_bf16_to_f32<<<(count + 255) / 256, 256, 0, stream>>>(out, out_bf16, count);
        error = cudaGetLastError();
    }
    cudaError_t lse_error = cudaFree(softmax_lse);
    cudaError_t output_error = cudaFree(out_bf16);
    if (error != cudaSuccess) return error;
    return lse_error == cudaSuccess ? output_error : lse_error;
}
