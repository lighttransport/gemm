// Vendored llama.cpp MMVQ (mul_mat_vec_q) device kernel exposed as a cubin for the
// cuda/llm runner's per-token DECODE matvec path.  Built to mmvq_kernels.cubin and
// loaded via the CUDA driver API (cuModuleGetFunction), mirroring mmq_kernels.cu.
//
// Why: our hand-written coalesced decode matvecs (matvec_iq2_xxs_q8_1_dp4a_coal et al.)
// top out at ~248 GB/s (1 warp/row).  llama.cpp's mul_mat_vec_q for ncols_dst=1 on
// sm_120 (MMVQ_PARAMETERS_GENERIC) uses nwarps=4 warps cooperating on ONE output row
// (K-split + shared reduction) -> ~347 GB/s, closing the 31B decode gap (22 vs 32.6 t/s).
//
// Upstream commit 5fd2dc2c41c342a75c26f9756ca6b1814ed05fb4 (MIT).  See mmq_vendor/NOTICE.
//
// build (see Makefile):
//   nvcc -cubin -arch=sm_120a -std=c++17 -O3 -I mmq_vendor \
//     -I <llama>/ggml/src/ggml-cuda -I <llama>/ggml/include -I <llama>/ggml/src \
//     -o mmvq_kernels.cubin mmvq_kernels.cu
//
// The vendored mmq_vendor/mmvq.cuh is resolved FIRST (-Immq_vendor); its mul_mat_vec_q
// is __device__ (not __global__) so the extern "C" trampolines below can call it.  Every
// other header (common.cuh, vecdotq.cuh, unary.cuh, ggml-common.h) comes from the
// upstream tree via the later -I dirs.
#include "mmvq.cuh"       // -> mmq_vendor/mmvq.cuh (via -Immq_vendor)

#include <cstdint>

// ---------------------------------------------------------------------------
// Activation quantizer: float -> block_q8_1 (regular 32-elem blocks, NOT the
// 144-byte mmq variant).  One warp per QK8_1 block.  Matches llama.cpp's
// quantize_q8_1 (MIT, same commit): d = amax/127, q = round(xi/d), ds=(d,sum).
// ---------------------------------------------------------------------------
extern "C" __global__ void mmvq_quant_q8_1(
        const float * __restrict__ x, void * __restrict__ vy, const int n) {
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int wid  = blockIdx.x * warps_per_block + threadIdx.x / WARP_SIZE; // q8_1 block index
    const int lane = threadIdx.x % WARP_SIZE;                                // quant index in block
    const int nblocks = n / QK8_1;
    if (wid >= nblocks) {
        return;
    }
    const float xi = x[wid*QK8_1 + lane];
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    block_q8_1 * y = (block_q8_1 *) vy;
    y[wid].qs[lane] = q;
    if (lane == 0) {
        y[wid].ds = make_half2(d, sum);
    }
}

// ---------------------------------------------------------------------------
// mul_mat_vec_q trampolines (ncols_dst=1 decode, has_fusion=false, ids=null).
// extern "C" -> stable unmangled cubin symbol names.  __launch_bounds__ matches
// upstream (warp_size * calc_nwarps).  Dense single-matrix matvec: the host fills
// the channel/sample strides with 0 and passes blockIdx.{y,z}=0, so the fastdiv
// uint3 args only ever see input 0 (-> 0) and need not be exact.
// ---------------------------------------------------------------------------
#define MMVQ_LB(TYPE) __launch_bounds__(calc_nwarps(TYPE, 1, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)

#define DEFINE_MMVQ(SUFFIX, TYPE)                                                     \
extern "C" __global__ void MMVQ_LB(TYPE) mmvq_##SUFFIX(                               \
        const void * vx, const void * vy, float * dst,                               \
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x,\
        const uint32_t stride_col_y, const uint32_t stride_col_dst,                  \
        const uint3 channel_ratio, const uint3 sample_ratio) {                       \
    ggml_cuda_mm_fusion_args_device fusion{};                                         \
    mul_mat_vec_q<TYPE, 1, false, false>(                                             \
        vx, vy, /*ids*/nullptr, fusion, dst,                                         \
        ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,            \
        channel_ratio, /*stride_channel_x*/0u, /*stride_channel_y*/0u,               \
        /*stride_channel_dst*/0u, sample_ratio, /*stride_sample_x*/0u,               \
        /*stride_sample_y*/0u, /*stride_sample_dst*/0u, /*ids_stride*/0u);           \
}

DEFINE_MMVQ(iq2xxs, GGML_TYPE_IQ2_XXS)  // 31B bulk (~58% of decode)
DEFINE_MMVQ(iq3xxs, GGML_TYPE_IQ3_XXS)  // 31B attn_v (every layer)
DEFINE_MMVQ(iq2s,   GGML_TYPE_IQ2_S)    // 31B UD-mix
DEFINE_MMVQ(iq3s,   GGML_TYPE_IQ3_S)    // 31B ffn_down
DEFINE_MMVQ(q2k,    GGML_TYPE_Q2_K)     // 31B ffn_down
DEFINE_MMVQ(q3k,    GGML_TYPE_Q3_K)     // 31B token_embd / lm_head
DEFINE_MMVQ(q6k,    GGML_TYPE_Q6_K)     // 12B Q6_K
DEFINE_MMVQ(q4_0,   GGML_TYPE_Q4_0)     // 12B QAT (qk=32)
