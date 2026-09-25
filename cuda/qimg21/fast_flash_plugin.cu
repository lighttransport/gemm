/* FlashAttention-2 forward (pinned upstream source) for the fast Qwen-Image
 * 2.1 denoiser: strided BF16 rows in and out, seqlen_q != seqlen_k, and
 * bottom-right causal masking for text prefix runs. Faster than the
 * memory-efficient kernel but not bit-identical to the PyTorch reference, so
 * it is an opt-in backend (--attention flash). */
#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_SOFTCAP

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include "csrc/flash_attn/src/flash.h"
#include "csrc/flash_attn/src/kernel_traits.h"
#include "csrc/flash_attn/src/flash_fwd_kernel.h"

using q21f_flash_traits = Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>;

template <bool Causal>
__global__ void q21f_flash_kernel(const FLASH_NAMESPACE::Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_attn<q21f_flash_traits, false, Causal, false, false, false, true, false, false>(params);
}

static float *q21f_lse;
static size_t q21f_lse_bytes;

/* Same contract as q21f_attention: out[nq, heads*128], row strides in
 * elements, mask 0 none or 2 causal from bottom-right. */
extern "C" int q21f_flash_attention(void *out, const void *q, const void *k, const void *v,
                                    int nq, int nk, int heads, int q_stride, int kv_stride,
                                    int o_stride, int mask, cudaStream_t stream) {
    static bool configured[2];
    if (!out || !q || !k || !v || nq <= 0 || nk <= 0 || heads <= 0 || (mask != 0 && mask != 2) ||
        q_stride % 8 || kv_stride % 8 || o_stride % 8)
        return cudaErrorInvalidValue;
    size_t lse_bytes = (size_t)nq * heads * sizeof(float);
    if (lse_bytes > q21f_lse_bytes) {
        if (q21f_lse) cudaFree(q21f_lse);
        q21f_lse = nullptr;
        q21f_lse_bytes = 0;
        cudaError_t e = cudaMalloc(&q21f_lse, lse_bytes);
        if (e != cudaSuccess) return e;
        q21f_lse_bytes = lse_bytes;
    }
    FLASH_NAMESPACE::Flash_fwd_params p;
    std::memset(&p, 0, sizeof(p));
    p.q_ptr = const_cast<void *>(q);
    p.k_ptr = const_cast<void *>(k);
    p.v_ptr = const_cast<void *>(v);
    p.o_ptr = out;
    p.softmax_lse_ptr = q21f_lse;
    p.q_row_stride = q_stride;
    p.k_row_stride = p.v_row_stride = kv_stride;
    p.o_row_stride = o_stride;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128;
    p.q_batch_stride = (int64_t)nq * q_stride;
    p.k_batch_stride = p.v_batch_stride = (int64_t)nk * kv_stride;
    p.o_batch_stride = (int64_t)nq * o_stride;
    p.b = 1;
    p.h = p.h_k = heads;
    p.h_h_k_ratio = 1;
    p.seqlen_q = nq;
    p.seqlen_k = nk;
    p.d = p.d_rounded = 128;
    p.seqlen_q_rounded = ((nq + 127) / 128) * 128;
    p.seqlen_k_rounded = ((nk + 127) / 128) * 128;
    p.scale_softmax = (float)(1.0 / std::sqrt(128.0));
    p.scale_softmax_log2 = p.scale_softmax * (float)M_LOG2E;
    p.p_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0f;
    p.is_causal = mask == 2;
    p.window_size_left = -1;
    p.window_size_right = mask == 2 ? 0 : -1;
    void (*kernel)(const FLASH_NAMESPACE::Flash_fwd_params) =
        mask == 2 ? &q21f_flash_kernel<true> : &q21f_flash_kernel<false>;
    cudaError_t error = cudaSuccess;
    if (!configured[mask == 2]) {
        error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     q21f_flash_traits::kSmemSize);
        if (error != cudaSuccess) return error;
        configured[mask == 2] = true;
    }
    dim3 grid((nq + q21f_flash_traits::kBlockM - 1) / q21f_flash_traits::kBlockM, 1, heads);
    kernel<<<grid, q21f_flash_traits::kNThreads, q21f_flash_traits::kSmemSize, stream>>>(p);
    return cudaGetLastError();
}
