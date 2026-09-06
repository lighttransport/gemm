// Vendored llama.cpp MMQ (mul_mat_q) device kernels exposed as a cubin for the
// cuda/llm runner's dense int8 tensor-core prefill path.  Built to mmq_kernels.cubin
// and loaded via the CUDA driver API (cuModuleGetFunction), mirroring fa2_kernels.cu.
//
// Why: our hand-written grouped8 MMQ kernels top out at ~17k GFLOP/s (no stream-K);
// llama.cpp's mul_mat_q with stream-K work-partitioning reaches ~72k GFLOP/s on the
// same RTX 5060 Ti (sm_120a), closing the ~2x prefill gap vs llama-bench.  Phase 0
// (cuda/llm/mmq/mmq_iq2xxs_vendor_test.cu) proved bit-exact + 4x.  See plan + NOTICE.
//
// Upstream commit 5fd2dc2c41c342a75c26f9756ca6b1814ed05fb4 (MIT).
//
// build (see Makefile):
//   nvcc -cubin -arch=sm_120a -std=c++17 -O3 \
//     -I mmq_vendor \
//     -I <llama>/ggml/src/ggml-cuda -I <llama>/ggml/include -I <llama>/ggml/src \
//     -o mmq_kernels.cubin mmq_kernels.cu
//
// The vendored mmq_vendor/mmq.cuh is resolved FIRST (its mul_mat_q /
// mul_mat_q_stream_k_fixup are __device__, not __global__); every other header
// (common.cuh, mma.cuh, vecdotq.cuh, ggml-common.h, quantize.cuh, ...) comes from
// the upstream tree via the later -I dirs.

// IMPORTANT: include the VENDORED mmq.cuh directly (not quantize.cuh).  A quoted
// #include resolves relative to the including file's own directory FIRST, so
// quantize.cuh (upstream) would pull the upstream mmq.cuh (mul_mat_q == __global__)
// and bypass our vendored __device__ copy.  mmq.cuh transitively includes
// common.cuh (QK8_1, WARP_SIZE, ggml_cuda_pdl_sync) + defines block_q8_1_mmq and
// the MMQ_Q8_1_DS_LAYOUT_* enum, which is all the quantizer below needs.
#include "mmq.cuh"        // -> mmq_vendor/mmq.cuh (via -Immq_vendor)

#include <cstdint>
static_assert(sizeof(block_fp4_mmq) == 144, "unexpected MMQ FP4 block size");

static __device__ __forceinline__ uint8_t mmqv_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) return 0;
    int e = __float2int_rn(log2f(amax)) - 2 + 127;
    return (uint8_t) max(0, min(254, e));
}

static __device__ __forceinline__ uint8_t mmqv_fp4(float x, float inv_scale) {
    const float lut[8] = { 0.0f, .5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float ax = fabsf(x) * inv_scale;
    int best = 0; float err = ax;
#pragma unroll
    for (int i = 1; i < 8; ++i) { float d = fabsf(ax - lut[i]); if (d < err) { err = d; best = i; } }
    return (uint8_t)(best | ((x < 0.0f) ? 8 : 0));
}

/* SM120 MXFP4 MMQ consumes block_fp4_mmq activations, not block_q8_1_mmq.
 * Each warp quantizes two E8M0 blocks (64 values) and stores the interleaved
 * nibble order expected by load_tiles_mxfp4_fp4. */
extern "C" __global__ void mmqv_quant_mxfp4(
        const float *x, const int32_t *ids, void *vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int64_t start = ((int64_t)blockIdx.y * blockDim.y + warp) * 64;
    if (start >= ne0) return;
    const int64_t row = ids ? ids[blockIdx.x] : blockIdx.x;
    const int64_t base = (blockIdx.z / ne2) * s03 + (blockIdx.z % ne2) * s02 + row * s01;
    uint8_t scale[2], p0[2], p1[2];
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        int64_t p = start + b * 32 + lane;
        float v = p < ne00 ? x[base + p] : 0.0f;
        float a = fabsf(v);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, m));
        uint8_t raw_scale = mmqv_e8m0_scale(a);
        scale[b] = raw_scale;
        float inv = a > 0.0f ? exp2f(-((float)raw_scale - 127.0f)) : 0.0f;
        uint8_t qv = mmqv_fp4(v, inv);
        int base_lane = (lane / 4) * 2;
        uint8_t lo0 = __shfl_sync(0xffffffff, qv, base_lane);
        uint8_t lo1 = __shfl_sync(0xffffffff, qv, base_lane + 1);
        uint8_t hi0 = __shfl_sync(0xffffffff, qv, base_lane + 16);
        uint8_t hi1 = __shfl_sync(0xffffffff, qv, base_lane + 17);
        p0[b] = (uint8_t)(lo0 | (hi0 << 4));
        p1[b] = (uint8_t)(lo1 | (hi1 << 4));
    }
    const int64_t block_fp4 = start / 256;
    const int quad = (start % 256) / 64;
    block_fp4_mmq *out = (block_fp4_mmq *)vy + (blockIdx.z * (int64_t)ne1 * (ne0 / 256) + block_fp4 * ne1 + row);
    int group = lane / 4, in_group = lane & 3;
    if (in_group == 0) {
        char2 *dst = (char2 *)out->qs;
        dst[quad * 16 + group] = make_char2((char)p0[0], (char)p1[0]);
        dst[quad * 16 + 8 + group] = make_char2((char)p0[1], (char)p1[1]);
    }
    if (lane == 0) out->d4[quad] = (uint32_t)scale[0] | ((uint32_t)scale[1] << 8);
}

/* Conservative row/quad quantizer.  The FP4 MMA path consumes
 * block_fp4_mmq activations: four E8M0 pairs and 128 packed bytes per
 * 256-value tile, laid out block-major across rows. */
extern "C" __global__ void mmqv_quant_mxfp4_rows(
        const float *x, void *vy, const int64_t ne00, const int64_t s01,
        const int64_t ne0, const int ne1) {
    const int lane = threadIdx.x;
    const int row = (int)blockIdx.x;
    const int pair = (int)blockIdx.y;
    const int64_t start = (int64_t)pair * 64;
    if (row >= ne1 || start >= ne0) return;
    uint8_t scale[2], packed[2][16];
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t p = start + b * 32 + lane;
        const float v = p < ne00 ? x[(int64_t)row * s01 + p] : 0.0f;
        float a = fabsf(v);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, m));
        const uint8_t e = mmqv_e8m0_scale(a);
        scale[b] = e;
        const float inv = a > 0.0f ? exp2f(-((float)e - 127.0f)) : 0.0f;
        const uint8_t qv = mmqv_fp4(v, inv);
        /* Keep every shuffle source in-range.  Only lanes 0..15 consume the
         * high nibble, but all lanes must participate in the intrinsic. */
        const uint8_t hi = __shfl_sync(0xffffffff, qv, lane ^ 16);
        if (lane < 16) {
            const uint8_t lo = qv;
            packed[b][lane] = (uint8_t)(lo | (hi << 4));
        }
    }
    const int block = pair / 4;
    const int quad = pair % 4;
    block_fp4_mmq *out = (block_fp4_mmq *)vy + (int64_t)block * ne1 + row;
    if (lane < 16) {
        uint8_t *qs = (uint8_t *)out->qs;
        qs[quad * 32 + lane] = packed[0][lane];
        qs[quad * 32 + 16 + lane] = packed[1][lane];
    }
    if (lane == 0) out->d4[quad] = (uint32_t)scale[0] | ((uint32_t)scale[1] << 8);
}

/* Two-term activation decomposition: res = x - dequant(E2M1-quant(x)), using
 * the same block absmax / e8m0 scale / FP4 lut as mmqv_quant_mxfp4 so the
 * residual plus the quantized term reproduces x in FP32.  In-place safe: the
 * block absmax (shuffle reduction) completes before any lane writes. */
extern "C" __global__ void mmqv_mxfp4_residual(
        float *res, const float *x,
        const int64_t ne00, const int ne1) {
    const int lane = threadIdx.x;
    const int row = (int)blockIdx.x;
    const int64_t start = (int64_t)blockIdx.y * 64;
    if (row >= ne1 || start >= ne00) return;
    const float lut[8] = { 0.0f, .5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const int64_t base = (int64_t)row * ne00;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t p = start + b * 32 + lane;
        const float v = p < ne00 ? x[base + p] : 0.0f;
        float a = fabsf(v);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, m));
        const uint8_t e = mmqv_e8m0_scale(a);
        const float inv = a > 0.0f ? exp2f(-((float)e - 127.0f)) : 0.0f;
        const uint8_t qv = mmqv_fp4(v, inv);
        const float qval = lut[qv & 7] * exp2f((float)e - 127.0f) * ((qv & 8) ? -1.0f : 1.0f);
        if (p < ne00) res[base + p] = v - qval;
    }
}

extern "C" __global__ void mmqv_add_f32(float *dst, const float *src, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) dst[i] += src[i];
}

// ---------------------------------------------------------------------------
// Activation quantizer: float -> block_q8_1_mmq.  Copied verbatim from
// llama.cpp ggml/src/ggml-cuda/quantize.cu (MIT, same commit) so it lives in
// this cubin.  Only the D4 layout (used by IQ2_XXS et al.) is exposed for now;
// DS4 (Q4_0/K-quants) and D2S6 added in later phases.
// ---------------------------------------------------------------------------
template <mmq_q8_1_ds_layout ds_layout>
static __device__ void quantize_mmq_q8_1_body(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }
        y[ib].d2s6[2 + iqs/16] = sum;
        if (iqs % 64 != 0) {
            return;
        }
        const float d = 1.0f / d_inv;
        y[ib].d2s6[iqs/64] = d;
        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

extern "C" __global__ void mmqv_quant_q8_1_d4(
        const float * x, const int32_t * ids, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    quantize_mmq_q8_1_body<MMQ_Q8_1_DS_LAYOUT_D4>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

extern "C" __global__ void mmqv_quant_q8_1_ds4(
        const float * x, const int32_t * ids, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    quantize_mmq_q8_1_body<MMQ_Q8_1_DS_LAYOUT_DS4>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

extern "C" __global__ void mmqv_quant_q8_1_d2s6(
        const float * x, const int32_t * ids, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {
    quantize_mmq_q8_1_body<MMQ_Q8_1_DS_LAYOUT_D2S6>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}

// ---------------------------------------------------------------------------
// mul_mat_q trampolines.  extern "C" -> stable unmangled cubin symbol names.
// __launch_bounds__ matches upstream (warp_size * nwarps, 1 block/SM on Volta+).
// mmq_x = 128 (Blackwell default).  need_check templated on whether out_dim %
// mmq_y == 0 (false = fast path, no row bounds check).
// ---------------------------------------------------------------------------
#define MMQ_TRAMP_LB __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 1)

#define DEFINE_MMQ_TRAMPOLINE(SUFFIX, TYPE, NEEDCHK)                                                  \
extern "C" __global__ void MMQ_TRAMP_LB mmqv_##SUFFIX(                                                \
        const char * x, const int * y, const int32_t * ids_dst,                                       \
        const int32_t * expert_bounds, float * dst, float * tmp_fixup,                                \
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x,  \
        const int ncols_y, const int stride_col_dst,                                                  \
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x,               \
        const int stride_channel_y, const int stride_channel_dst,                                      \
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x,                  \
        const int stride_sample_y, const int stride_sample_dst,                                        \
        const uint3 ntx) {                                                                            \
    mul_mat_q<TYPE, 128, NEEDCHK>(                                                                     \
        x, y, ids_dst, expert_bounds, dst, tmp_fixup,                                                  \
        blocks_per_ne00, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst,                    \
        channel_ratio, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst,            \
        sample_ratio, nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, ntx);           \
}

#define DEFINE_MMQ_FIXUP(SUFFIX, TYPE, NEEDCHK)                                                       \
extern "C" __global__ void mmqv_fixup_##SUFFIX(                                                       \
        const int32_t * ids_dst, const int32_t * expert_bounds, float * dst,                          \
        float * tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,   \
        const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst,              \
        const uint3 nsamples_y, const int stride_sample_dst, const uint3 ntx) {                       \
    mul_mat_q_stream_k_fixup<TYPE, 128, NEEDCHK>(                                                      \
        ids_dst, expert_bounds, dst, tmp_last_tile, blocks_per_ne00, nrows_x, ncols_dst,              \
        stride_col_dst, nchannels_y, stride_channel_dst, nsamples_y, stride_sample_dst, ntx);         \
}

// Per-type: kernel (nc0/nc1) + stream-K fixup (nc0/nc1).
#define DEFINE_MMQ_TYPE(SUFFIX, TYPE)             \
    DEFINE_MMQ_TRAMPOLINE(SUFFIX##_x128_nc0, TYPE, false) \
    DEFINE_MMQ_TRAMPOLINE(SUFFIX##_x128_nc1, TYPE, true)  \
    DEFINE_MMQ_FIXUP(SUFFIX##_x128_nc0, TYPE, false)      \
    DEFINE_MMQ_FIXUP(SUFFIX##_x128_nc1, TYPE, true)

#define DEFINE_MMQ_X64(SUFFIX, TYPE) \
extern "C" __global__ void mmqv_##SUFFIX##_x64_nc0( \
        const char *x, const int *y, const int32_t *ids_dst, const int32_t *expert_bounds, float *dst, float *tmp_fixup, \
        const uint3 bp, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst, \
        const uint3 cr, const uint3 ny, const int scx, const int scy, const int scd, const uint3 sr, const uint3 sy, const int ssx, const int ssy, const int ssd, const uint3 ntx) { \
    mul_mat_q<TYPE, 64, false>(x,y,ids_dst,expert_bounds,dst,tmp_fixup,bp,nrows_x,ncols_dst,stride_row_x,ncols_y,stride_col_dst,cr,ny,scx,scy,scd,sr,sy,ssx,ssy,ssd,ntx); } \
extern "C" __global__ void mmqv_##SUFFIX##_x64_nc1( \
        const char *x, const int *y, const int32_t *ids_dst, const int32_t *expert_bounds, float *dst, float *tmp_fixup, \
        const uint3 bp, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst, \
        const uint3 cr, const uint3 ny, const int scx, const int scy, const int scd, const uint3 sr, const uint3 sy, const int ssx, const int ssy, const int ssd, const uint3 ntx) { \
    mul_mat_q<TYPE, 64, true>(x,y,ids_dst,expert_bounds,dst,tmp_fixup,bp,nrows_x,ncols_dst,stride_row_x,ncols_y,stride_col_dst,cr,ny,scx,scy,scd,sr,sy,ssx,ssy,ssd,ntx); } \
extern "C" __global__ void mmqv_fixup_##SUFFIX##_x64_nc0( \
        const int32_t *ids_dst, const int32_t *expert_bounds, float *dst, float *tmp, const uint3 bp, const int nr, const int nc, const int stride, const uint3 ny, const int scd, const uint3 sy, const int ssd, const uint3 ntx) { \
    mul_mat_q_stream_k_fixup<TYPE,64,false>(ids_dst,expert_bounds,dst,tmp,bp,nr,nc,stride,ny,scd,sy,ssd,ntx); }

DEFINE_MMQ_TYPE(iq2xxs, GGML_TYPE_IQ2_XXS)  // Phase 1 (31B, dominant)
DEFINE_MMQ_TYPE(iq3xxs, GGML_TYPE_IQ3_XXS)  // 31B attn_v
DEFINE_MMQ_TYPE(iq2s,   GGML_TYPE_IQ2_S)    // 31B UD-mix
DEFINE_MMQ_TYPE(iq3s,   GGML_TYPE_IQ3_S)    // 31B ffn_down
DEFINE_MMQ_TYPE(q2k,    GGML_TYPE_Q2_K)     // 31B ffn_down (D2S6 quant)
DEFINE_MMQ_TYPE(q4_0,   GGML_TYPE_Q4_0)     // 12B QAT (DS4 quant, qk=32)
DEFINE_MMQ_TYPE(q6k,    GGML_TYPE_Q6_K)     // 12B Q6_K
DEFINE_MMQ_TYPE(mxfp4,  GGML_TYPE_MXFP4)    // DS4F raw MXFP4 experts
DEFINE_MMQ_X64(mxfp4, GGML_TYPE_MXFP4)
