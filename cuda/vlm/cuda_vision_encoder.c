/*
 * cuda_vision_encoder.c - CUDA vision encoder for Qwen3-VL mmproj
 *
 * Compiles with plain gcc (no nvcc). Uses cuew for dynamic CUDA/NVRTC loading.
 * Supports F32 (verification) and F16 (performance) weight modes.
 * Single-stream sequential kernel launches.
 */

#include "../../common/ggml_dequant.h"

#include "cuda_vision_encoder.h"
#include "../cuew.h"
#include "../cublasew.h"
#include "../cuda_kernels_common.h"
#include "../cuda_fp8_mma_kernels.h"  /* fp8_mma_kernels_src: shared per-row FP8 MMA GEMM (gating experiment) */
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Aliases for shared macros */
#define CHECK_CU CU_CHECK
#define CHECK_CU_NULL CU_CHECK_NULL

/* ======================================================================== */
/* Vision-specific CUDA kernels (compiled at runtime via NVRTC)             */
/* Shared kernels (layernorm, GEMM, gelu, add, etc.) are in                 */
/* cuda_kernels_common.h. This string is concatenated after them.           */
/* ======================================================================== */

static const char *cuda_vlm_specific_kernels =
"\n"
"/* ---- gemm_f32_f32: Naive tiled F32 GEMM for verification ---- */\n"
"/* Y[tok][i] = sum_j(W[i][j] * X[tok][j]) + bias[i] */\n"
"/* Grid: (ceil(n_out/TILE), ceil(n_tok/TILE)), Block: (TILE, TILE) */\n"
"#define TILE_F32 16\n"
"__global__ void gemm_f32_f32(float *Y, const float *W, const float *X,\n"
"                              const float *bias,\n"
"                              int n_out, int n_in, int n_tok) {\n"
"    __shared__ float sW[TILE_F32][TILE_F32];\n"
"    __shared__ float sX[TILE_F32][TILE_F32];\n"
"    int bx = blockIdx.x * TILE_F32;\n"
"    int by = blockIdx.y * TILE_F32;\n"
"    int tx = threadIdx.x;\n"
"    int ty = threadIdx.y;\n"
"    int row = by + ty;  /* token index */\n"
"    int col = bx + tx;  /* output dim index */\n"
"    float sum = 0.0f;\n"
"    for (int t = 0; t < n_in; t += TILE_F32) {\n"
"        /* Load W tile: W[col][t+ty] — but col is the output row of W */\n"
"        if (col < n_out && (t + ty) < n_in)\n"
"            sW[tx][ty] = W[(size_t)col * n_in + t + ty];\n"
"        else\n"
"            sW[tx][ty] = 0.0f;\n"
"        /* Load X tile: X[row][t+tx] */\n"
"        if (row < n_tok && (t + tx) < n_in)\n"
"            sX[ty][tx] = X[(size_t)row * n_in + t + tx];\n"
"        else\n"
"            sX[ty][tx] = 0.0f;\n"
"        __syncthreads();\n"
"        for (int k = 0; k < TILE_F32; k++)\n"
"            sum += sW[tx][k] * sX[ty][k];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < n_tok && col < n_out) {\n"
"        float b = (bias) ? bias[col] : 0.0f;\n"
"        Y[(size_t)row * n_out + col] = sum + b;\n"
"    }\n"
"}\n"
"\n"
"/* ---- patch_embed_dual_f32: Dual Conv2D patch extraction ---- */\n"
"/* Grid: (n_patches), Block: (256) */\n"
"__global__ void patch_embed_dual_f32(float *out, const float *rgb,\n"
"                                      const float *w0, const float *w1,\n"
"                                      const float *bias,\n"
"                                      int gw, int dim, int ps, int img_w) {\n"
"    int patch = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    int ks = ps * ps * 3;\n"
"    for (int d = tid; d < dim; d += blockDim.x) {\n"
"        float sum = bias ? bias[d] : 0.0f;\n"
"        for (int c = 0; c < 3; c++) {\n"
"            for (int ky = 0; ky < ps; ky++) {\n"
"                for (int kx = 0; kx < ps; kx++) {\n"
"                    int iy = py * ps + ky;\n"
"                    int ix = px * ps + kx;\n"
"                    float pix = rgb[(iy * img_w + ix) * 3 + c];\n"
"                    int ki = c * ps * ps + ky * ps + kx;\n"
"                    sum += w0[d * ks + ki] * pix;\n"
"                    if (w1) sum += w1[d * ks + ki] * pix;\n"
"                }\n"
"            }\n"
"        }\n"
"        out[patch * dim + d] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ---- patch_im2col_f32: gather patches into [n_patches, ps*ps*3] for a GEMM ---- */\n"
"/* Replaces the uncoalesced patch_embed_dual_f32 kernel: build the im2col matrix here  */\n"
"/* (coalesced writes), then out = patch_pix . (w0+w1)^T via cuBLAS.                    */\n"
"/* Grid: (n_patches), Block: (256). ki = c*ps*ps + ky*ps + kx (matches conv weight).   */\n"
"__global__ void patch_im2col_f32(float *cols, const float *rgb,\n"
"                                 int gw, int ps, int img_w) {\n"
"    int patch = blockIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    int ks = ps * ps * 3;\n"
"    float *dst = cols + (size_t)patch * ks;\n"
"    for (int ki = threadIdx.x; ki < ks; ki += blockDim.x) {\n"
"        int c = ki / (ps * ps);\n"
"        int rem = ki - c * ps * ps;\n"
"        int ky = rem / ps, kx = rem % ps;\n"
"        int iy = py * ps + ky, ix = px * ps + kx;\n"
"        dst[ki] = rgb[((size_t)iy * img_w + ix) * 3 + c];\n"
"    }\n"
"}\n"
"\n"
"/* ---- add_pos_embd: add position embeddings via indirection map ---- */\n"
"/* Grid: (n_patches), Block: (256) */\n"
"__global__ void add_pos_embd(float *hidden, const float *pos_emb,\n"
"                               const int *pos_map, int dim) {\n"
"    int p = blockIdx.x;\n"
"    int orig_p = pos_map[p];\n"
"    int tid = threadIdx.x;\n"
"    for (int d = tid; d < dim; d += blockDim.x)\n"
"        hidden[p * dim + d] += pos_emb[orig_p * dim + d];\n"
"}\n"
"\n"
"/* ---- add_pos_embd_direct: add pre-interpolated position embeddings ---- */\n"
"/* Grid: (n_patches), Block: (256) */\n"
"__global__ void add_pos_embd_direct(float *hidden, const float *pos_emb,\n"
"                                      int dim, int n) {\n"
"    int p = blockIdx.x;\n"
"    if (p >= n) return;\n"
"    for (int d = threadIdx.x; d < dim; d += blockDim.x)\n"
"        hidden[p * dim + d] += pos_emb[p * dim + d];\n"
"}\n"
"\n"
"/* ---- rope_vision_f32: M-RoPE on Q and K ---- */\n"
"/* Grid-stride over (p,h,i) rotation pairs; block 256. (The old launch used   */\n"
"/* head_dim/2 == 36 threads/block over n_patches*n_heads blocks, which wastes  */\n"
"/* the 2nd warp and pins occupancy to the per-SM block limit -> it ran ~3x     */\n"
"/* slower than an equal-volume layernorm and scaled ~5x for 2.25x tokens.)     */\n"
"__global__ void rope_vision_f32(float *qkv, const float *rope_cos,\n"
"                                  const float *rope_sin,\n"
"                                  int n_patches, int n_heads,\n"
"                                  int dim, int head_dim, int half) {\n"
"    long total = (long)n_patches * n_heads * half;\n"
"    for (long gid = (long)blockIdx.x * blockDim.x + threadIdx.x; gid < total;\n"
"         gid += (long)gridDim.x * blockDim.x) {\n"
"        int i = (int)(gid % half);\n"
"        long t = gid / half;\n"
"        int h = (int)(t % n_heads);\n"
"        int p = (int)(t / n_heads);\n"
"        float cos_t = rope_cos[p * head_dim + 2 * i];\n"
"        float sin_t = rope_sin[p * head_dim + 2 * i];\n"
"        /* Q */\n"
"        float *q = qkv + (long)p * 3 * dim + h * head_dim;\n"
"        float q0 = q[i], q1 = q[i + half];\n"
"        q[i]        = q0 * cos_t - q1 * sin_t;\n"
"        q[i + half] = q0 * sin_t + q1 * cos_t;\n"
"        /* K */\n"
"        float *k = q + dim;\n"
"        float k0 = k[i], k1 = k[i + half];\n"
"        k[i]        = k0 * cos_t - k1 * sin_t;\n"
"        k[i + half] = k0 * sin_t + k1 * cos_t;\n"
"    }\n"
"}\n"
"\n"
"/* ---- attn_full_f32: Full NxN self-attention per head ---- */\n"
"/* Grid: (n_heads), Block: (256) */\n"
"/* Uses shared memory for softmax reduction. */\n"
"__global__ void attn_full_f32(float *out, const float *qkv,\n"
"                                int n_patches, int dim, int n_heads,\n"
"                                int head_dim, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int dim3 = 3 * dim;\n"
"    /* Process queries sequentially */\n"
"    for (int qi = 0; qi < n_patches; qi++) {\n"
"        const float *q_h = qkv + qi * dim3 + h * head_dim;\n"
"        /* Compute scores QK^T for this query */\n"
"        for (int ki = tid; ki < n_patches; ki += nt) {\n"
"            const float *k_h = qkv + ki * dim3 + dim + h * head_dim;\n"
"            float score = 0.0f;\n"
"            for (int d = 0; d < head_dim; d++)\n"
"                score += q_h[d] * k_h[d];\n"
"            smem[ki] = score * scale;\n"
"        }\n"
"        __syncthreads();\n"
"        /* Softmax: find max */\n"
"        float local_max = -1e30f;\n"
"        for (int ki = tid; ki < n_patches; ki += nt)\n"
"            if (smem[ki] > local_max) local_max = smem[ki];\n"
"        /* Reduce max in shared memory */\n"
"        smem[n_patches + tid] = local_max;\n"
"        __syncthreads();\n"
"        for (int r = nt/2; r > 0; r >>= 1) {\n"
"            if (tid < r && smem[n_patches + tid + r] > smem[n_patches + tid])\n"
"                smem[n_patches + tid] = smem[n_patches + tid + r];\n"
"            __syncthreads();\n"
"        }\n"
"        float max_val = smem[n_patches];\n"
"        /* Exp and sum */\n"
"        float local_sum = 0.0f;\n"
"        for (int ki = tid; ki < n_patches; ki += nt) {\n"
"            smem[ki] = expf(smem[ki] - max_val);\n"
"            local_sum += smem[ki];\n"
"        }\n"
"        smem[n_patches + tid] = local_sum;\n"
"        __syncthreads();\n"
"        for (int r = nt/2; r > 0; r >>= 1) {\n"
"            if (tid < r) smem[n_patches + tid] += smem[n_patches + tid + r];\n"
"            __syncthreads();\n"
"        }\n"
"        float inv_sum = 1.0f / smem[n_patches];\n"
"        /* Normalize */\n"
"        for (int ki = tid; ki < n_patches; ki += nt)\n"
"            smem[ki] *= inv_sum;\n"
"        __syncthreads();\n"
"        /* Weighted sum of V */\n"
"        for (int d = tid; d < head_dim; d += nt) {\n"
"            float sum = 0.0f;\n"
"            for (int vi = 0; vi < n_patches; vi++) {\n"
"                sum += smem[vi] * qkv[vi * dim3 + 2 * dim + h * head_dim + d];\n"
"            }\n"
"            out[qi * dim + h * head_dim + d] = sum;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"}\n"
"\n"
"/* ---- spatial_merge_f32: Gather 2x2 patches into merged tokens ---- */\n"
"/* Grid: (n_merged), Block: (256) */\n"
"__global__ void spatial_merge_f32(float *dst, const float *src,\n"
"                                    int gw, int sm, int dim) {\n"
"    int m = blockIdx.x;\n"
"    int mgw = gw / sm;\n"
"    int my = m / mgw, mx = m % mgw;\n"
"    int merged_dim = dim * sm * sm;\n"
"    int tid = threadIdx.x;\n"
"    for (int di = tid; di < merged_dim; di += blockDim.x) {\n"
"        int sub = di / dim;\n"
"        int d = di % dim;\n"
"        int sy = sub / sm, sx = sub % sm;\n"
"        int py = my * sm + sy;\n"
"        int px = mx * sm + sx;\n"
"        dst[m * merged_dim + di] = src[(py * gw + px) * dim + d];\n"
"    }\n"
"}\n"
"\n"
"/* ---- reorder_rows_f32: dst[row] = src[index[row]] ---- */\n"
"__global__ void reorder_rows_f32(float *dst, const float *src,\n"
"                                 const int *index, int row_dim) {\n"
"    int row = blockIdx.x;\n"
"    int src_row = index[row];\n"
"    for (int d = threadIdx.x; d < row_dim; d += blockDim.x) {\n"
"        dst[(size_t)row * row_dim + d] = src[(size_t)src_row * row_dim + d];\n"
"    }\n"
"}\n"
"\n"
"/* ---- attn_window_f32: local self-attention on contiguous windows ---- */\n"
"/* Grid: (n_heads, n_windows), Block: (256) */\n"
"__global__ void attn_window_f32(float *out, const float *qkv,\n"
"                                const int *win_start, const int *win_size,\n"
"                                int dim, int n_heads, int head_dim, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    int w = blockIdx.y;\n"
"    if (h >= n_heads) return;\n"
"    int start = win_start[w];\n"
"    int size = win_size[w];\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int dim3 = 3 * dim;\n"
"    if (size <= 0) return;\n"
"    for (int ql = 0; ql < size; ql++) {\n"
"        int qi = start + ql;\n"
"        const float *q_h = qkv + (size_t)qi * dim3 + h * head_dim;\n"
"        for (int kl = tid; kl < size; kl += nt) {\n"
"            int ki = start + kl;\n"
"            const float *k_h = qkv + (size_t)ki * dim3 + dim + h * head_dim;\n"
"            float score = 0.0f;\n"
"            for (int d = 0; d < head_dim; d++) score += q_h[d] * k_h[d];\n"
"            smem[kl] = score * scale;\n"
"        }\n"
"        __syncthreads();\n"
"        float local_max = -1e30f;\n"
"        for (int kl = tid; kl < size; kl += nt)\n"
"            if (smem[kl] > local_max) local_max = smem[kl];\n"
"        smem[size + tid] = local_max;\n"
"        __syncthreads();\n"
"        for (int r = nt/2; r > 0; r >>= 1) {\n"
"            if (tid < r && smem[size + tid + r] > smem[size + tid])\n"
"                smem[size + tid] = smem[size + tid + r];\n"
"            __syncthreads();\n"
"        }\n"
"        float max_val = smem[size];\n"
"        float local_sum = 0.0f;\n"
"        for (int kl = tid; kl < size; kl += nt) {\n"
"            smem[kl] = expf(smem[kl] - max_val);\n"
"            local_sum += smem[kl];\n"
"        }\n"
"        smem[size + tid] = local_sum;\n"
"        __syncthreads();\n"
"        for (int r = nt/2; r > 0; r >>= 1) {\n"
"            if (tid < r) smem[size + tid] += smem[size + tid + r];\n"
"            __syncthreads();\n"
"        }\n"
"        float inv_sum = 1.0f / smem[size];\n"
"        for (int kl = tid; kl < size; kl += nt) smem[kl] *= inv_sum;\n"
"        __syncthreads();\n"
"        for (int d = tid; d < head_dim; d += nt) {\n"
"            float sum = 0.0f;\n"
"            for (int vl = 0; vl < size; vl++) {\n"
"                int vi = start + vl;\n"
"                sum += smem[vl] * qkv[(size_t)vi * dim3 + 2 * dim + h * head_dim + d];\n"
"            }\n"
"            out[(size_t)qi * dim + h * head_dim + d] = sum;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"}\n"
"\n"
"/* ---- flash_attn_f32: online-softmax flash attention (full, non-windowed). ---- */\n"
"/* Replaces attn_full_f32's 16-block serial loop. One warp per query; K/V tiled into  */\n"
"/* shared memory and reused across the FA_WARPS queries in a block.                    */\n"
"/* Grid: (ceil(n_patches/FA_WARPS), n_heads).  Block: FA_WARPS*32 threads.             */\n"
"/* Dynamic shared mem: 2*FA_TILE_K*head_dim floats (K tile + V tile).                  */\n"
"#define FA_WARPS 16\n"
"#define FA_TILE_K 16\n"
"#define FA_MAXREG 4\n"
"__global__ void flash_attn_f32(float *out, const float *qkv,\n"
"                               int n_patches, int dim, int n_heads,\n"
"                               int head_dim, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    float *ksh = smem;\n"
"    float *vsh = smem + FA_TILE_K * head_dim;\n"
"    int warp = threadIdx.x >> 5;\n"
"    int lane = threadIdx.x & 31;\n"
"    int h = blockIdx.y;\n"
"    int qi = blockIdx.x * FA_WARPS + warp;\n"
"    int dim3 = 3 * dim;\n"
"    int nreg = (head_dim + 31) >> 5;\n"
"    int n_threads = FA_WARPS * 32;\n"
"    int valid = (qi < n_patches);\n"
"    float q[FA_MAXREG], acc[FA_MAXREG];\n"
"    #pragma unroll\n"
"    for (int r = 0; r < FA_MAXREG; r++) { q[r] = 0.0f; acc[r] = 0.0f; }\n"
"    if (valid) {\n"
"        const float *q_h = qkv + (size_t)qi * dim3 + h * head_dim;\n"
"        for (int r = 0; r < nreg; r++) { int d = lane + (r << 5); if (d < head_dim) q[r] = q_h[d]; }\n"
"    }\n"
"    float m = -1e30f, l = 0.0f;\n"
"    for (int kt = 0; kt < n_patches; kt += FA_TILE_K) {\n"
"        int tk = n_patches - kt; if (tk > FA_TILE_K) tk = FA_TILE_K;\n"
"        for (int idx = threadIdx.x; idx < tk * head_dim; idx += n_threads) {\n"
"            int j = idx / head_dim; int d = idx - j * head_dim;\n"
"            const float *base = qkv + (size_t)(kt + j) * dim3 + h * head_dim;\n"
"            ksh[idx] = base[dim + d];\n"
"            vsh[idx] = base[2 * dim + d];\n"
"        }\n"
"        __syncthreads();\n"
"        if (valid) {\n"
"            for (int j = 0; j < tk; j++) {\n"
"                const float *kj = ksh + j * head_dim;\n"
"                float partial = 0.0f;\n"
"                for (int r = 0; r < nreg; r++) { int d = lane + (r << 5); if (d < head_dim) partial += q[r] * kj[d]; }\n"
"                #pragma unroll\n"
"                for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);\n"
"                float score = partial * scale;\n"
"                float m_new = m > score ? m : score;\n"
"                float corr = __expf(m - m_new);\n"
"                float p = __expf(score - m_new);\n"
"                l = l * corr + p;\n"
"                const float *vj = vsh + j * head_dim;\n"
"                for (int r = 0; r < nreg; r++) { int d = lane + (r << 5); acc[r] = acc[r] * corr + (d < head_dim ? p * vj[d] : 0.0f); }\n"
"                m = m_new;\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (valid) {\n"
"        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;\n"
"        float *o = out + (size_t)qi * dim + h * head_dim;\n"
"        for (int r = 0; r < nreg; r++) { int d = lane + (r << 5); if (d < head_dim) o[d] = acc[r] * inv; }\n"
"    }\n"
"}\n"
"\n"
"/* ---- attn_window_warp_f32: windowed attention, ONE WARP PER QUERY.            */\n"
"/* Replaces attn_window_f32's serial-query + full-block-reduction design (which   */\n"
"/* did ~16 __syncthreads per query with most threads idle). Online softmax over   */\n"
"/* the window's keys; q/acc kept in regs (3/lane for head_dim=72); scores reduced */\n"
"/* with __shfl_xor_sync. The window's K/V are staged into shared memory once per   */\n"
"/* block so the WW_WARPS warps don't re-read them from global per query.           */\n"
"/* Grid: (n_windows, n_heads). Block: WW_WARPS*32. Dyn smem: 2*max_win*head_dim.   */\n"
"#define WW_WARPS 12  /* 36-token windows -> 3 queries/warp, balanced (swept) */\n"
"__global__ void attn_window_warp_f32(float *out, const float *qkv,\n"
"                                     const int *win_start, const int *win_size,\n"
"                                     int dim, int n_heads, int head_dim, float scale) {\n"
"    extern __shared__ float wsmem[];\n"
"    int w = blockIdx.x;\n"
"    int h = blockIdx.y;\n"
"    int start = win_start[w];\n"
"    int size = win_size[w];\n"
"    if (size <= 0) return;\n"
"    int warp = threadIdx.x >> 5;\n"
"    int lane = threadIdx.x & 31;\n"
"    int dim3 = 3 * dim;\n"
"    int nreg = (head_dim + 31) >> 5;\n"
"    float *ksh = wsmem;\n"
"    float *vsh = wsmem + size * head_dim;\n"
"    for (int idx = threadIdx.x; idx < size * head_dim; idx += blockDim.x) {\n"
"        int kl = idx / head_dim; int d = idx - kl * head_dim;\n"
"        const float *base = qkv + (size_t)(start + kl) * dim3 + h * head_dim;\n"
"        ksh[idx] = base[dim + d];\n"
"        vsh[idx] = base[2 * dim + d];\n"
"    }\n"
"    __syncthreads();\n"
"    for (int ql = warp; ql < size; ql += WW_WARPS) {\n"
"        int qi = start + ql;\n"
"        const float *q_h = qkv + (size_t)qi * dim3 + h * head_dim;\n"
"        float q[4], acc[4];\n"
"        for (int r = 0; r < nreg; r++) { int d = lane + (r << 5); q[r] = (d < head_dim) ? q_h[d] : 0.0f; acc[r] = 0.0f; }\n"
"        float m = -1e30f, l = 0.0f;\n"
"        for (int kl = 0; kl < size; kl++) {\n"
"            const float *kj = ksh + kl * head_dim;\n"
"            float partial = 0.0f;\n"
"            for (int r = 0; r < nreg; r++) { int d = lane + (r << 5); if (d < head_dim) partial += q[r] * kj[d]; }\n"
"            for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);\n"
"            float score = partial * scale;\n"
"            float m_new = m > score ? m : score;\n"
"            float corr = __expf(m - m_new);\n"
"            float p = __expf(score - m_new);\n"
"            l = l * corr + p;\n"
"            const float *vj = vsh + kl * head_dim;\n"
"            for (int r = 0; r < nreg; r++) { int d = lane + (r << 5); acc[r] = acc[r] * corr + (d < head_dim ? p * vj[d] : 0.0f); }\n"
"            m = m_new;\n"
"        }\n"
"        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;\n"
"        float *o = out + (size_t)qi * dim + h * head_dim;\n"
"        for (int r = 0; r < nreg; r++) { int d = lane + (r << 5); if (d < head_dim) o[d] = acc[r] * inv; }\n"
"    }\n"
"}\n"
"\n"
"/* ---- attn_window_tile_f32: windowed attention, MATERIALIZE-SCORES design.     */\n"
"/* The warp-per-query kernel above serializes the online softmax across keys     */\n"
"/* (one dependent __shfl_xor + __expf per key). For the tiny fixed windows here   */\n"
"/* (<=36 tokens) it is cheaper to stage Q/K/V in shared, compute the full         */\n"
"/* S[size,size] score matrix with NO reductions (each thread owns whole (i,j)     */\n"
"/* dot products), softmax each row once (warp-per-row, 2 reductions), then P*V.   */\n"
"/* This removes the per-key serial dependency entirely.                          */\n"
"/* Grid: (n_windows, n_heads). Block: WT_THREADS. Dyn smem:                       */\n"
"/* (3*max_win*head_dim + max_win*max_win) floats.                                 */\n"
"#define WT_THREADS 192\n"
"/* Q/K/V are staged in shared as F16 (half the bytes of F32). At head_dim=72,   */\n"
"/* win<=36: F32 staging needed 3*36*72*4 + 36*36*4 = 35.4 KB/block -> only 2     */\n"
"/* blocks/SM (smem-limited, ~25%% occ). F16 staging is 3*36*72*2 + 36*36*4 =     */\n"
"/* 20.3 KB -> 4 blocks/SM, doubling occupancy. Dot products still accumulate in  */\n"
"/* F32 and the score matrix / softmax stay F32; only the staged operands are     */\n"
"/* F16 -- consistent with the materialized full-attn path (attn_extract_heads    */\n"
"/* also stages Q/K/V as F16).                                                    */\n"
"__global__ void attn_window_tile_f32(float *out, const float *qkv,\n"
"                                     const int *win_start, const int *win_size,\n"
"                                     int dim, int n_heads, int head_dim, float scale) {\n"
"    extern __shared__ char tsmem[];\n"
"    int w = blockIdx.x;\n"
"    int h = blockIdx.y;\n"
"    int start = win_start[w];\n"
"    int size = win_size[w];\n"
"    if (size <= 0) return;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int dim3 = 3 * dim;\n"
"    half_raw *Qs = (half_raw *)tsmem;\n"
"    half_raw *Ks = Qs + size * head_dim;\n"
"    half_raw *Vs = Ks + size * head_dim;\n"
"    float *Ss = (float *)(Vs + size * head_dim);\n"  /* 3*size*head_dim halves is 4-byte aligned */
"    for (int idx = tid; idx < size * head_dim; idx += nt) {\n"
"        int kl = idx / head_dim; int d = idx - kl * head_dim;\n"
"        const float *base = qkv + (size_t)(start + kl) * dim3 + h * head_dim;\n"
"        half_raw q, k, v;\n"
"        asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(q) : \"f\"(base[d]));\n"
"        asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(k) : \"f\"(base[dim + d]));\n"
"        asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(v) : \"f\"(base[2 * dim + d]));\n"
"        Qs[idx] = q; Ks[idx] = k; Vs[idx] = v;\n"
"    }\n"
"    __syncthreads();\n"
"    int ns = size * size;\n"
"    for (int idx = tid; idx < ns; idx += nt) {\n"
"        int i = idx / size; int j = idx - i * size;\n"
"        const half_raw *qi = Qs + i * head_dim;\n"
"        const half_raw *kj = Ks + j * head_dim;\n"
"        float s = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) s += half_to_float(qi[d]) * half_to_float(kj[d]);\n"
"        Ss[idx] = s * scale;\n"
"    }\n"
"    __syncthreads();\n"
"    int warp = tid >> 5; int lane = tid & 31; int nwarps = nt >> 5;\n"
"    for (int i = warp; i < size; i += nwarps) {\n"
"        float *srow = Ss + i * size;\n"
"        float m = -1e30f;\n"
"        for (int j = lane; j < size; j += 32) { float v = srow[j]; if (v > m) m = v; }\n"
"        for (int off = 16; off > 0; off >>= 1) { float o = __shfl_xor_sync(0xffffffff, m, off); if (o > m) m = o; }\n"
"        float sum = 0.0f;\n"
"        for (int j = lane; j < size; j += 32) { float e = __expf(srow[j] - m); srow[j] = e; sum += e; }\n"
"        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);\n"
"        float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;\n"
"        for (int j = lane; j < size; j += 32) srow[j] *= inv;\n"
"    }\n"
"    __syncthreads();\n"
"    for (int idx = tid; idx < size * head_dim; idx += nt) {\n"
"        int i = idx / head_dim; int d = idx - i * head_dim;\n"
"        const float *prow = Ss + i * size;\n"
"        float acc = 0.0f;\n"
"        for (int j = 0; j < size; j++) acc += prow[j] * half_to_float(Vs[j * head_dim + d]);\n"
"        out[(size_t)(start + i) * dim + h * head_dim + d] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* ---- attn_extract_heads: deinterleave qkv[N,3*dim] (post-RoPE) into          */\n"
"/* head-contiguous F16 Q/K/V buffers [n_heads, n_patches, head_dim] for the      */\n"
"/* tensor-core (cuBLAS) attention path. One thread per qkv element.              */\n"
"__global__ void attn_extract_heads(half_raw *qh, half_raw *kh, half_raw *vh,\n"
"                                   const float *qkv, int n_patches, int dim,\n"
"                                   int n_heads, int head_dim) {\n"
"    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"    long total = (long)n_patches * 3 * dim;\n"
"    if (idx >= total) return;\n"
"    int d3 = 3 * dim;\n"
"    int i = (int)(idx / d3);\n"
"    int rem = (int)(idx - (long)i * d3);\n"
"    int third = rem / dim;\n"
"    int within = rem - third * dim;\n"
"    int h = within / head_dim;\n"
"    int dh = within - h * head_dim;\n"
"    long dst = ((long)h * n_patches + i) * head_dim + dh;\n"
"    float f = qkv[idx]; half_raw hr;\n"
"    asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(hr) : \"f\"(f));\n"
"    if (third == 0) qh[dst] = hr; else if (third == 1) kh[dst] = hr; else vh[dst] = hr;\n"
"}\n"
"\n"
"/* ---- attn_softmax_rows: row-wise softmax of scores[N,N] (scale folded in)     */\n"
"/* into F16 probs[N,N]. One block per row; block-reduction over the N columns.   */\n"
"/* Dyn smem: blockDim floats.                                                    */\n"
"__global__ void attn_softmax_rows(half_raw *probs, const float *scores,\n"
"                                  int n, float scale) {\n"
"    int row = blockIdx.x;\n"
"    const float *s = scores + (long)row * n;\n"
"    half_raw *p = probs + (long)row * n;\n"
"    extern __shared__ float red[];\n"
"    int t = threadIdx.x; int nt = blockDim.x;\n"
"    float m = -1e30f;\n"
"    for (int j = t; j < n; j += nt) { float v = s[j] * scale; if (v > m) m = v; }\n"
"    red[t] = m; __syncthreads();\n"
"    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o && red[t + o] > red[t]) red[t] = red[t + o]; __syncthreads(); }\n"
"    float row_max = red[0]; __syncthreads();\n"
"    float sum = 0.0f;\n"
"    for (int j = t; j < n; j += nt) sum += __expf(s[j] * scale - row_max);\n"
"    red[t] = sum; __syncthreads();\n"
"    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }\n"
"    float inv = (red[0] > 0.0f) ? 1.0f / red[0] : 0.0f;\n"
"    for (int j = t; j < n; j += nt) {\n"
"        float e = __expf(s[j] * scale - row_max) * inv;\n"
"        half_raw hr; asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(hr) : \"f\"(e)); p[j] = hr;\n"
"    }\n"
"}\n"
"\n"
"/* ---- cast_f32_bf16: F32 -> BF16, for the BF16 cuBLAS path ---- */\n"
"__global__ void cast_f32_bf16(unsigned short *dst, const float *src, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float f = src[i]; unsigned short h;\n"
"    asm(\"cvt.rn.bf16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(f));\n"
"    dst[i] = h;\n"
"}\n"
"\n"
"/* ---- gelu_f32_to_bf16: tanh-GELU fused with F32->BF16 cast (BF16 FFN path) ---- */\n"
"__global__ void gelu_f32_to_bf16(unsigned short *dst, const float *src, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float v = src[i];\n"
"    float g = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));\n"
"    unsigned short h;\n"
"    asm(\"cvt.rn.bf16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(g));\n"
"    dst[i] = h;\n"
"}\n"
"\n"
"/* ---- layernorm_f32_bf16: layernorm that emits BF16 directly (BF16 GEMM path) ---- */\n"
"__global__ void layernorm_f32_bf16(unsigned short *dst, const float *src,\n"
"                                    const float *w, const float *b,\n"
"                                    int dim, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int tok = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    const float *x = src + (size_t)tok * dim;\n"
"    unsigned short *y = dst + (size_t)tok * dim;\n"
"    float s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) s += x[i];\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)dim;\n"
"    __syncthreads();\n"
"    s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) { float d = x[i] - mean; s += d*d; }\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)dim + eps);\n"
"    for (int i = tid; i < dim; i += nt) {\n"
"        float v = (x[i] - mean) * inv * w[i] + b[i];\n"
"        unsigned short h; asm(\"cvt.rn.bf16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(v)); y[i] = h;\n"
"    }\n"
"}\n"
"\n"
"/* ---- attn_extract_heads_bf16: deinterleave qkv[N,3*dim] into BF16 head buffers ---- */\n"
"__global__ void attn_extract_heads_bf16(unsigned short *qh, unsigned short *kh,\n"
"                                         unsigned short *vh, const float *qkv,\n"
"                                         int n_patches, int dim, int n_heads, int head_dim) {\n"
"    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
"    long total = (long)n_patches * 3 * dim;\n"
"    if (idx >= total) return;\n"
"    int d3 = 3 * dim;\n"
"    int i = (int)(idx / d3);\n"
"    int rem = (int)(idx - (long)i * d3);\n"
"    int third = rem / dim;\n"
"    int within = rem - third * dim;\n"
"    int h = within / head_dim;\n"
"    int dh = within - h * head_dim;\n"
"    long dst = ((long)h * n_patches + i) * head_dim + dh;\n"
"    float f = qkv[idx]; unsigned short hr;\n"
"    asm(\"cvt.rn.bf16.f32 %0, %1;\" : \"=h\"(hr) : \"f\"(f));\n"
"    if (third == 0) qh[dst] = hr; else if (third == 1) kh[dst] = hr; else vh[dst] = hr;\n"
"}\n"
"\n"
"/* ---- attn_softmax_rows_bf16: row softmax -> BF16 probs (BF16 attention path) ---- */\n"
"__global__ void attn_softmax_rows_bf16(unsigned short *probs, const float *scores,\n"
"                                        int n, float scale) {\n"
"    int row = blockIdx.x;\n"
"    const float *s = scores + (long)row * n;\n"
"    unsigned short *p = probs + (long)row * n;\n"
"    extern __shared__ float red[];\n"
"    int t = threadIdx.x; int nt = blockDim.x;\n"
"    float m = -1e30f;\n"
"    for (int j = t; j < n; j += nt) { float v = s[j] * scale; if (v > m) m = v; }\n"
"    red[t] = m; __syncthreads();\n"
"    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o && red[t + o] > red[t]) red[t] = red[t + o]; __syncthreads(); }\n"
"    float row_max = red[0]; __syncthreads();\n"
"    float sum = 0.0f;\n"
"    for (int j = t; j < n; j += nt) sum += __expf(s[j] * scale - row_max);\n"
"    red[t] = sum; __syncthreads();\n"
"    for (int o = nt >> 1; o > 0; o >>= 1) { if (t < o) red[t] += red[t + o]; __syncthreads(); }\n"
"    float inv = (red[0] > 0.0f) ? 1.0f / red[0] : 0.0f;\n"
"    for (int j = t; j < n; j += nt) {\n"
"        float e = __expf(s[j] * scale - row_max) * inv;\n"
"        unsigned short hr; asm(\"cvt.rn.bf16.f32 %0, %1;\" : \"=h\"(hr) : \"f\"(e)); p[j] = hr;\n"
"    }\n"
"}\n"
"\n"
"/* ---- attn_prefill_vision_f32: TENSOR-CORE flash full attention. ----------------- */\n"
"/* Adapted from cuda_kernels_common.h's attn_prefill_f32 (proven LLM-prefill flash)  */\n"
"/* but (a) reads Q/K/V straight from the interleaved qkv[N,3*dim] buffer (no K_t/V_t  */\n"
"/* transpose -- same trick the window kernels use) and (b) handles head_dim=72        */\n"
"/* (nkf=ceil(72/16)=5 k16 frags, last padded d>=hd ->0; noc=ceil(72/8)=9 output       */\n"
"/* 8-col groups, 9*8=72 exact). mma.sync.m16n8k16.row.col.f32.f16.f16.f32 + online    */\n"
"/* softmax + O-rescale, O(N) memory; stages each 16-key K/V tile into shared so all   */\n"
"/* 4 warps reuse it. Used only as the O(N) fallback above flash_full_n -- it is        */\n"
"/* numerically identical to the materialized path (rel_L2 0.63948 vs 0.63945 @512²)    */\n"
"/* but 4-14%% SLOWER (cuBLAS tiles/pipelines better; head_dim=72 wastes half the 5th   */\n"
"/* k16 frag), far better than the retired CUDA-core flash (3.6-5.5x). Grid (n_heads,   */\n"
"/* ceil(N/64)), block 128 (4 warps, 16 q/warp). sm_120 a1/a2 frag swap from source.    */\n"
"#define VP_KF 5\n"
"#define VP_OC 9\n"
"#if __CUDA_ARCH__ >= 800\n"
"__global__ void attn_prefill_vision_f32(float *out, const float *qkv,\n"
"                                        int n_tok, int dim, int n_heads,\n"
"                                        int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    if (blockIdx.y * 64 >= n_tok) return;  /* whole block out -> all warps return */\n"
"    int warp_id = threadIdx.x / 32;\n"
"    int qb = blockIdx.y * 64 + warp_id * 16;  /* may exceed n_tok; per-query guards below */\n"
"    int lane = threadIdx.x % 32;\n"
"    int gid = lane / 4, tid4 = lane % 4;\n"
"    int dim3 = 3 * dim;\n"
"    int qi0 = qb + gid, qi1 = qb + gid + 8;\n"
"    /* Shared K/V tile (16 keys x head_dim), staged once per block so all 4 warps    */\n"
"    /* reuse it instead of each re-reading K/V from global (was the 4x-redundant      */\n"
"    /* bottleneck vs cuBLAS, which tiles K/V into shared).                            */\n"
"    extern __shared__ float smem[];\n"
"    float *Ks = smem;\n"
"    float *Vs = smem + 16 * head_dim;\n"
"    int nkf = (head_dim + 15) >> 4;\n"
"    int noc = (head_dim + 7) >> 3;\n"
"    int qoff = h * head_dim;\n"
"    /* Pre-load Q fragments (m16k16, 4 per k16 step). */\n"
"    unsigned int qa0[VP_KF], qa1[VP_KF], qa2[VP_KF], qa3[VP_KF];\n"
"    for (int ks = 0; ks < nkf; ks++) {\n"
"        int dc = ks * 16 + tid4 * 2;\n"
"        { float f0=(qi0<n_tok && dc  <head_dim)?qkv[(size_t)qi0*dim3+qoff+dc  ]:0.0f,\n"
"                f1=(qi0<n_tok && dc+1<head_dim)?qkv[(size_t)qi0*dim3+qoff+dc+1]:0.0f;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa0[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"#if __CUDA_ARCH__ >= 1200\n"
"        { float f0=(qi1<n_tok && dc  <head_dim)?qkv[(size_t)qi1*dim3+qoff+dc  ]:0.0f,\n"
"                f1=(qi1<n_tok && dc+1<head_dim)?qkv[(size_t)qi1*dim3+qoff+dc+1]:0.0f;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa1[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"        { float f0=(qi0<n_tok && dc+8<head_dim)?qkv[(size_t)qi0*dim3+qoff+dc+8]:0.0f,\n"
"                f1=(qi0<n_tok && dc+9<head_dim)?qkv[(size_t)qi0*dim3+qoff+dc+9]:0.0f;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa2[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"#else\n"
"        { float f0=(qi0<n_tok && dc+8<head_dim)?qkv[(size_t)qi0*dim3+qoff+dc+8]:0.0f,\n"
"                f1=(qi0<n_tok && dc+9<head_dim)?qkv[(size_t)qi0*dim3+qoff+dc+9]:0.0f;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa1[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"        { float f0=(qi1<n_tok && dc  <head_dim)?qkv[(size_t)qi1*dim3+qoff+dc  ]:0.0f,\n"
"                f1=(qi1<n_tok && dc+1<head_dim)?qkv[(size_t)qi1*dim3+qoff+dc+1]:0.0f;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa2[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"#endif\n"
"        { float f0=(qi1<n_tok && dc+8<head_dim)?qkv[(size_t)qi1*dim3+qoff+dc+8]:0.0f,\n"
"                f1=(qi1<n_tok && dc+9<head_dim)?qkv[(size_t)qi1*dim3+qoff+dc+9]:0.0f;\n"
"          asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(qa3[ks]):\"f\"(f0),\"f\"(f1)); }\n"
"    }\n"
"    float m0=-1e30f, l0=0.0f, m1=-1e30f, l1=0.0f;\n"
"    float oc0[VP_OC]={0}, oc1[VP_OC]={0}, oc2[VP_OC]={0}, oc3[VP_OC]={0};\n"
"    int koff = dim + h * head_dim, voff = 2 * dim + h * head_dim;\n"
"    for (int kv = 0; kv < n_tok; kv += 16) {\n"
"        /* Cooperatively stage the 16-key K/V tile (coalesced); zero-fill OOB keys. */\n"
"        for (int t = threadIdx.x; t < 16 * head_dim; t += blockDim.x) {\n"
"            int kl = t / head_dim, d = t - kl * head_dim;\n"
"            int gk = kv + kl;\n"
"            Ks[t] = (gk < n_tok) ? qkv[(size_t)gk*dim3 + koff + d] : 0.0f;\n"
"            Vs[t] = (gk < n_tok) ? qkv[(size_t)gk*dim3 + voff + d] : 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        float s0[2]={0,0}, s1[2]={0,0}, s2[2]={0,0}, s3[2]={0,0};\n"
"        for (int ks = 0; ks < nkf; ks++) {\n"
"            unsigned int a0=qa0[ks], a1=qa1[ks], a2=qa2[ks], a3=qa3[ks];\n"
"            int col = ks * 16 + tid4 * 2;\n"
"            for (int nh = 0; nh < 2; nh++) {\n"
"                int kl = nh*8 + gid;\n"
"                const float *kp = Ks + kl*head_dim;  /* shared; OOB keys zero-filled */\n"
"                unsigned int b0=0, b1=0;\n"
"                { float kf0=(col  <head_dim)?kp[col  ]:0.0f, kf1=(col+1<head_dim)?kp[col+1]:0.0f;\n"
"                  asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(b0):\"f\"(kf0),\"f\"(kf1));\n"
"                  float kf2=(col+8<head_dim)?kp[col+8]:0.0f, kf3=(col+9<head_dim)?kp[col+9]:0.0f;\n"
"                  asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(b1):\"f\"(kf2),\"f\"(kf3)); }\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    :\"=f\"(s0[nh]),\"=f\"(s1[nh]),\"=f\"(s2[nh]),\"=f\"(s3[nh])\n"
"                    :\"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1),\n"
"                     \"f\"(s0[nh]),\"f\"(s1[nh]),\"f\"(s2[nh]),\"f\"(s3[nh]));\n"
"            }\n"
"        }\n"
"        s0[0]*=scale; s1[0]*=scale; s2[0]*=scale; s3[0]*=scale;\n"
"        s0[1]*=scale; s1[1]*=scale; s2[1]*=scale; s3[1]*=scale;\n"
"        { int c0=kv+tid4*2, c1=c0+1;\n"
"          if (c0   >=n_tok){s0[0]=-1e30f;s2[0]=-1e30f;} if (c1   >=n_tok){s1[0]=-1e30f;s3[0]=-1e30f;}\n"
"          if (c0+8 >=n_tok){s0[1]=-1e30f;s2[1]=-1e30f;} if (c1+8 >=n_tok){s1[1]=-1e30f;s3[1]=-1e30f;} }\n"
"        if (qi0>=n_tok){s0[0]=-1e30f;s1[0]=-1e30f;s0[1]=-1e30f;s1[1]=-1e30f;}\n"
"        if (qi1>=n_tok){s2[0]=-1e30f;s3[0]=-1e30f;s2[1]=-1e30f;s3[1]=-1e30f;}\n"
"        /* row qi0 (gid): keys at cols tid4*2, tid4*2+1 over nh halves */\n"
"        float mx0 = fmaxf(fmaxf(s0[0],s1[0]), fmaxf(s0[1],s1[1]));\n"
"        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));\n"
"        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));\n"
"        float mn0 = fmaxf(m0, mx0);\n"
"        float al0 = __expf(m0 - mn0); l0 *= al0; m0 = mn0;\n"
"        for (int c=0;c<noc;c++){ oc0[c]*=al0; oc1[c]*=al0; }\n"
"        s0[0]=__expf(s0[0]-m0); s1[0]=__expf(s1[0]-m0); s0[1]=__expf(s0[1]-m0); s1[1]=__expf(s1[1]-m0);\n"
"        float rs0 = (s0[0]+s1[0])+(s0[1]+s1[1]);\n"
"        rs0 += __shfl_xor_sync(0xffffffff, rs0, 1);\n"
"        rs0 += __shfl_xor_sync(0xffffffff, rs0, 2);\n"
"        l0 += rs0;\n"
"        float mx1 = fmaxf(fmaxf(s2[0],s3[0]), fmaxf(s2[1],s3[1]));\n"
"        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));\n"
"        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));\n"
"        float mn1 = fmaxf(m1, mx1);\n"
"        float al1 = __expf(m1 - mn1); l1 *= al1; m1 = mn1;\n"
"        for (int c=0;c<noc;c++){ oc2[c]*=al1; oc3[c]*=al1; }\n"
"        s2[0]=__expf(s2[0]-m1); s3[0]=__expf(s3[0]-m1); s2[1]=__expf(s2[1]-m1); s3[1]=__expf(s3[1]-m1);\n"
"        float rs1 = (s2[0]+s3[0])+(s2[1]+s3[1]);\n"
"        rs1 += __shfl_xor_sync(0xffffffff, rs1, 1);\n"
"        rs1 += __shfl_xor_sync(0xffffffff, rs1, 2);\n"
"        l1 += rs1;\n"
"        /* Convert P (probs) to f16 fragments for the PV mma (A operand, m16k16).\n"
"         * Pair the two column elements of the same nh-half (s0,s1 / s2,s3), NOT\n"
"         * the two nh-halves -- matches attn_prefill_f32's C->A fragment remap.    */\n"
"        unsigned int pa0, pa1, pa2, pa3;\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa0):\"f\"(s0[0]),\"f\"(s1[0]));\n"
"#if __CUDA_ARCH__ >= 1200\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa1):\"f\"(s2[0]),\"f\"(s3[0]));\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa2):\"f\"(s0[1]),\"f\"(s1[1]));\n"
"#else\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa1):\"f\"(s0[1]),\"f\"(s1[1]));\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa2):\"f\"(s2[0]),\"f\"(s3[0]));\n"
"#endif\n"
"        asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(pa3):\"f\"(s2[1]),\"f\"(s3[1]));\n"
"        /* PV: O[16q, 8d] += P[16q,16k] V[16k,8d], 9 column groups (72=9*8). */\n"
"        for (int c = 0; c < noc; c++) {\n"
"            int vd = c*8+gid;\n"
"            int vl0=tid4*2, vl1=vl0+1, vl8=vl0+8, vl9=vl8+1;  /* local key indices 0..15 */\n"
"            unsigned int vb0=0, vb1=0;\n"
"            if (vd < head_dim) {\n"
"                float vf0=Vs[vl0*head_dim+vd], vf1=Vs[vl1*head_dim+vd];\n"
"                asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(vb0):\"f\"(vf0),\"f\"(vf1));\n"
"                float vf2=Vs[vl8*head_dim+vd], vf3=Vs[vl9*head_dim+vd];\n"
"                asm(\"cvt.rn.f16x2.f32 %0, %2, %1;\":\"=r\"(vb1):\"f\"(vf2),\"f\"(vf3));\n"
"            }\n"
"            asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                :\"=f\"(oc0[c]),\"=f\"(oc1[c]),\"=f\"(oc2[c]),\"=f\"(oc3[c])\n"
"                :\"r\"(pa0),\"r\"(pa1),\"r\"(pa2),\"r\"(pa3),\"r\"(vb0),\"r\"(vb1),\n"
"                 \"f\"(oc0[c]),\"f\"(oc1[c]),\"f\"(oc2[c]),\"f\"(oc3[c]));\n"
"        }\n"
"        __syncthreads();  /* all warps done with Ks/Vs before next tile overwrites */\n"
"    }\n"
"    float il0=(l0>0.0f)?1.0f/l0:0.0f, il1=(l1>0.0f)?1.0f/l1:0.0f;\n"
"    for (int c = 0; c < noc; c++) {\n"
"        int d0=c*8+tid4*2, d1=d0+1;\n"
"        if (qi0<n_tok && d0<head_dim) out[(size_t)qi0*dim+qoff+d0]=oc0[c]*il0;\n"
"        if (qi0<n_tok && d1<head_dim) out[(size_t)qi0*dim+qoff+d1]=oc1[c]*il0;\n"
"        if (qi1<n_tok && d0<head_dim) out[(size_t)qi1*dim+qoff+d0]=oc2[c]*il1;\n"
"        if (qi1<n_tok && d1<head_dim) out[(size_t)qi1*dim+qoff+d1]=oc3[c]*il1;\n"
"    }\n"
"}\n"
"#endif\n"
"\n"
"/* ---- attn_full_bq_f32: full attention, ONE BLOCK PER QUERY. ---- */\n"
"/* Grid: (n_patches, n_heads). Block: BQ_THREADS. Each block's threads cooperate on a   */\n"
"/* single query: each thread computes COMPLETE scores for its keys (no per-key warp     */\n"
"/* reduction), block-reduces softmax, then computes the output. K/V cached in L2 across  */\n"
"/* the query-blocks of a head. Dyn smem: (head_dim + n_patches + blockDim) floats.       */\n"
"__global__ void attn_full_bq_f32(float *out, const float *qkv,\n"
"                                 int n_patches, int dim, int n_heads,\n"
"                                 int head_dim, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int qi = blockIdx.x;\n"
"    int h  = blockIdx.y;\n"
"    int tid = threadIdx.x;\n"
"    int nt  = blockDim.x;\n"
"    int dim3 = 3 * dim;\n"
"    float *q_sh   = smem;\n"
"    float *scores = smem + head_dim;\n"
"    float *red    = scores + n_patches;\n"
"    const float *q_h = qkv + (size_t)qi * dim3 + h * head_dim;\n"
"    for (int d = tid; d < head_dim; d += nt) q_sh[d] = q_h[d];\n"
"    __syncthreads();\n"
"    for (int ki = tid; ki < n_patches; ki += nt) {\n"
"        const float *k_h = qkv + (size_t)ki * dim3 + dim + h * head_dim;\n"
"        float s = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) s += q_sh[d] * k_h[d];\n"
"        scores[ki] = s * scale;\n"
"    }\n"
"    __syncthreads();\n"
"    float lm = -1e30f;\n"
"    for (int ki = tid; ki < n_patches; ki += nt) lm = fmaxf(lm, scores[ki]);\n"
"    red[tid] = lm; __syncthreads();\n"
"    for (int r = nt >> 1; r > 0; r >>= 1) { if (tid < r) red[tid] = fmaxf(red[tid], red[tid + r]); __syncthreads(); }\n"
"    float mx = red[0]; __syncthreads();\n"
"    float ls = 0.0f;\n"
"    for (int ki = tid; ki < n_patches; ki += nt) { float e = __expf(scores[ki] - mx); scores[ki] = e; ls += e; }\n"
"    red[tid] = ls; __syncthreads();\n"
"    for (int r = nt >> 1; r > 0; r >>= 1) { if (tid < r) red[tid] += red[tid + r]; __syncthreads(); }\n"
"    float inv = 1.0f / red[0]; __syncthreads();\n"
"    for (int d = tid; d < head_dim; d += nt) {\n"
"        float acc = 0.0f;\n"
"        for (int vi = 0; vi < n_patches; vi++)\n"
"            acc += scores[vi] * qkv[(size_t)vi * dim3 + 2 * dim + h * head_dim + d];\n"
"        out[(size_t)qi * dim + h * head_dim + d] = acc * inv;\n"
"    }\n"
"}\n"
"\n"
"\n"
"/* ---- round_f32_to_f16: round F32 elements through F16 (in-place), matching ---- */\n"
"/* PyTorch's F16 precision after each GEMM. This is the key fix for operation     */\n"
"/* ordering parity: PyTorch's F16 model rounds every stored value to F16, but our */\n"
"/* layers keep F32 throughout. Rounding at the same points makes the numerics      */\n"
"/* match. Enabled by VLM_ROUND_F16=1 env var.                                     */\n"
"__global__ void round_f32_to_f16(float *buf, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    half_raw h;\n"
"    asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(buf[i]));\n"
"    asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(buf[i]) : \"h\"(h));\n"
"}\n"
"\n"
"/* ---- cast_f32_f16: F32 -> F16 (half_raw bits), for the cuBLAS F16xF16 path ---- */\n"
"/* Mixed F16(W)xF32(X) GemmEx is unsupported on Blackwell (sm_120); cast X to F16 */\n"
"/* and use cublasew_gemm_f16_f16_f32_rowmajor_nt instead. */\n"
"__global__ void cast_f32_f16(half_raw *dst, const float *src, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float f = src[i]; half_raw h;\n"
"    asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(f));\n"
"    dst[i] = h;\n"
"}\n"
"\n"
"/* ---- gelu_f32_to_f16: tanh-GELU (matches gelu_f32) fused with F32->F16 cast ---- */\n"
"/* Lets the FFN up-proj (bias fused in the GEMM) feed the down-proj as F16 with no */\n"
"/* separate cast: one pass reads F32, applies GELU, writes F16. */\n"
"__global__ void gelu_f32_to_f16(half_raw *dst, const float *src, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float v = src[i];\n"
"    float g = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));\n"
"    half_raw h;\n"
"    asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(g));\n"
"    dst[i] = h;\n"
"}\n"
"\n"
"/* ---- layernorm_f32_f16: layernorm that emits F16 directly (folds the F32->F16 ---- */\n"
"/* cast that the qkv / ffn-up GEMM would otherwise do on the Blackwell F16xF16 path). */\n"
"__global__ void layernorm_f32_f16(half_raw *dst, const float *src, const float *w,\n"
"                                   const float *b, int dim, float eps) {\n"
"    extern __shared__ float sdata[];\n"
"    int tok = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    const float *x = src + (size_t)tok * dim;\n"
"    half_raw *y = dst + (size_t)tok * dim;\n"
"    float s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) s += x[i];\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)dim;\n"
"    __syncthreads();\n"
"    s = 0.0f;\n"
"    for (int i = tid; i < dim; i += nt) { float d = x[i] - mean; s += d*d; }\n"
"    sdata[tid] = s;\n"
"    __syncthreads();\n"
"    for (int r = nt/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)dim + eps);\n"
"    for (int i = tid; i < dim; i += nt) {\n"
"        float v = (x[i] - mean) * inv * w[i] + b[i];\n"
"        half_raw h; asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h) : \"f\"(v)); y[i] = h;\n"
"    }\n"
"}\n"
"\n"
"} /* extern C */\n"
;

/* ======================================================================== */
/* FP8 ffn_up gating experiment (appended only when VLM_FFN_FP8=1).          */
/* cuda_vlm_specific_kernels closes its own extern "C", so this opens a fresh */
/* one. Defines the to_bf16 hook as identity -> gemm_fp8_pipe_perrow_f32      */
/* writes pure F32 (no bf16 rounding), and a weight prequant kernel.         */
/* fp8_mma_kernels_src (the shared per-row FP8 MMA GEMM + reduce) is          */
/* concatenated after this prefix, then cuda_vlm_fp8_suffix closes the block. */
/* ======================================================================== */
static const char *cuda_vlm_fp8_prefix =
"extern \"C\" {\n"
"__device__ __forceinline__ float to_bf16(float f) { return f; }\n"
"/* Referenced by the BF16-decode GEMM variants in fp8_mma_kernels_src (unused */\n"
"/* by VLM, which only calls gemm_fp8_pipe_perrow_f32); declared so the module  */\n"
"/* compiles. Left unpopulated -- those kernels are never launched here. */\n"
"__device__ __constant__ unsigned short d_fp8_to_bf16_lut[256];\n"
"/* Quantize F16 weight -> e4m3 with one inverse (per-tensor) scale. The low  */\n"
"/* byte of cvt.rn.satfinite.e4m3x2 holds operand b (= v), high holds a (=0). */\n"
"__global__ void vlm_quant_w_f16_to_e4m3(unsigned char *Wq,\n"
"                                        const unsigned short *Wf16,\n"
"                                        int n, float inv_w_scale) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float h;\n"
"    asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(h) : \"h\"(Wf16[i]));\n"
"    float v = h * inv_w_scale;\n"
"    unsigned short p;\n"
"    asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\" : \"=h\"(p) : \"f\"(v), \"f\"(0.0f));\n"
"    Wq[i] = (unsigned char)(p & 0xFF);\n"
"}\n";
static const char *cuda_vlm_fp8_suffix = "\n} /* extern C (fp8) */\n";

/* ======================================================================== */
/* Runner struct                                                            */
/* ======================================================================== */

typedef struct {
    CUdeviceptr w_f32;   /* F32 weight [n_out, n_in] */
    CUdeviceptr w_f16;   /* F16 weight [n_out, n_in] (if use_f16) */
    CUdeviceptr w_bf16;  /* BF16 weight [n_out, n_in] (if use_bf16) */
    CUdeviceptr bias;    /* F32 bias [n_out] (always F32) */
    CUdeviceptr w_fp8;   /* e4m3 weight [n_out_pad, n_in], n_out padded to *256 (FP8 expt) */
    float w_scale;       /* per-tensor weight scale: max|W|/448 (FP8 expt) */
} gpu_weight;

typedef struct {
    gpu_weight attn_qkv;    /* [3*dim, dim] */
    gpu_weight attn_out;    /* [dim, dim] */
    gpu_weight ffn_up;      /* [ffn_dim, dim] */
    gpu_weight ffn_down;    /* [dim, ffn_dim] */
    CUdeviceptr ln1_w, ln1_b;  /* F32 [dim] */
    CUdeviceptr ln2_w, ln2_b;  /* F32 [dim] */
} gpu_vit_block;

typedef struct {
    gpu_weight fc1;   /* [merged_dim, merged_dim] */
    gpu_weight fc2;   /* [proj_dim, merged_dim] */
    CUdeviceptr norm_w, norm_b;  /* F32 [merged_dim] */
} gpu_deepstack;

struct cuda_vision_runner {
    CUdevice device;
    CUcontext context;
    CUstream stream;
    cublasew_context *cublas;
    int verbose;
    int use_f16;
    int use_bf16;  /* BF16 native path (VLM_BF16=1); mutually exclusive with use_f16 */
    int use_cublas;
    int cublas_mixed_ok;   /* mixed F16(W)xF32(X) GemmEx works (false on Blackwell) */
    int ffn_fp8;           /* VLM_FFN_FP8=1: run ffn_up via per-row FP8 MMA (expt) */
    int f16_round;         /* VLM_ROUND_F16=1: round GEMM outputs to F16 to match PyTorch */

    CUmodule module;
    /* Shared kernels */
    CUfunction fn_layernorm_f32;
    CUfunction fn_layernorm_f32_f16; /* layernorm with F16 output (folds qkv/ffn-up cast) */
    CUfunction fn_gemm_f16_f32;
    CUfunction fn_gelu_f32;
    CUfunction fn_add_f32;
    CUfunction fn_add_bias_f32;
    /* Vision-specific kernels */
    CUfunction fn_gemm_f32_f32;
    CUfunction fn_patch_embed_dual_f32;
    CUfunction fn_patch_im2col_f32;
    CUfunction fn_add_pos_embd;
    CUfunction fn_add_pos_embd_direct;
    CUfunction fn_rope_vision_f32;
    CUfunction fn_attn_full_f32;
    CUfunction fn_flash_attn_f32;
    CUfunction fn_attn_prefill_vision_f32; /* TENSOR-CORE O(N)-mem full attention */
    CUfunction fn_attn_full_bq_f32;
    CUfunction fn_attn_extract_heads;   /* deinterleave qkv -> head-contiguous F16 */
    CUfunction fn_attn_softmax_rows;    /* row softmax scores[N,N] -> F16 probs */
    CUfunction fn_attn_window_f32;
    CUfunction fn_attn_window_warp_f32; /* warp-per-query windowed attention */
    CUfunction fn_attn_window_tile_f32; /* materialize-scores windowed attention */
    CUfunction fn_reorder_rows_f32;
    CUfunction fn_spatial_merge_f32;
    CUfunction fn_cast_f32_f16;    /* F32 -> F16 cast for cuBLAS F16xF16 path */
    CUfunction fn_gelu_f32_to_f16;  /* fused tanh-GELU + F32->F16 cast (FFN) */
    CUfunction fn_round_f32_to_f16; /* F32 -> F16 -> F32 (matches PyTorch F16 precision) */
    /* BF16 path kernels */
    CUfunction fn_cast_f32_bf16;    /* F32 -> BF16 cast */
    CUfunction fn_gelu_f32_to_bf16; /* fused tanh-GELU + F32->BF16 cast (FFN BF16) */
    CUfunction fn_layernorm_f32_bf16; /* layernorm with BF16 output */
    CUfunction fn_attn_extract_heads_bf16; /* deinterleave qkv -> BF16 head buffers */
    CUfunction fn_attn_softmax_rows_bf16;  /* row softmax -> BF16 probs */
    /* FP8 ffn_up experiment kernels (loaded only when ffn_fp8) */
    CUfunction fn_reduce_max_abs_per_row_f32;  /* per-row max|X| for activation scale */
    CUfunction fn_gemm_fp8_pipe_perrow_f32;    /* per-row FP8 MMA GEMM (F32 X, F32 Y) */
    CUfunction fn_vlm_quant_w_f16_to_e4m3;     /* F16 weight -> e4m3 prequant */

    /* Model hyperparams */
    int n_blocks;
    int dim;
    int n_heads;
    int head_dim;
    int ffn_dim;
    int patch_size;
    int image_size;
    int n_patches;
    int proj_dim;
    int spatial_merge;
    int n_merged;
    int n_wa_pattern;
    int attn_window_size;
    float ln_eps;
    float image_mean[3];
    float image_std[3];

    /* Dynamic resolution support */
    int max_patches;           /* max patches for buffer allocation (0 = use n_patches) */
    int max_merged;            /* max merged tokens */
    int max_pixels;            /* max pixel count for RGB buffer */
    float *h_pos_embd;         /* CPU copy of original pos embedding [n_patches * dim] */
    CUdeviceptr d_pos_interp;  /* GPU buffer for interpolated pos embedding [max_patches * dim] */
    int pos_interp_w, pos_interp_h; /* grid the cached d_pos_interp is valid for (-1 = none) */

    /* GPU weights: patch embeddings */
    CUdeviceptr d_patch_w0;     /* F32 [dim, ps*ps*3] (w1 folded in: w0 += w1 at load) */
    CUdeviceptr d_patch_w1;     /* F32 [dim, ps*ps*3] (second conv, may be 0) */
    CUdeviceptr d_patch_pix;    /* F32 [max_patches, ps*ps*3] im2col scratch */
    CUdeviceptr d_patch_bias;   /* F32 [dim] */

    /* Position embedding */
    CUdeviceptr d_pos_embd;     /* F32 [n_patches, dim] */

    /* Blocks */
    gpu_vit_block *blocks;

    /* DeepStack */
    int n_deepstack;
    int *deepstack_indices;
    gpu_deepstack *deepstack;

    /* Post LN */
    CUdeviceptr d_post_ln_w, d_post_ln_b;  /* F32 [dim] */

    /* MM projection */
    gpu_weight mm0;   /* [merged_dim, merged_dim] */
    gpu_weight mm2;   /* [proj_dim, merged_dim] */

    /* Scratch buffers (allocated on load) */
    CUdeviceptr d_hidden;     /* [max_patches * dim] */
    CUdeviceptr d_hidden2;    /* [max_patches * dim] */
    CUdeviceptr d_qkv;        /* [max_patches * 3 * dim] */
    CUdeviceptr d_attn_out;   /* [max_patches * dim] */
    CUdeviceptr d_ffn_buf;    /* [max_patches * ffn_dim] */
    CUdeviceptr d_ln_buf;     /* [max_patches * dim] */
    CUdeviceptr d_ln_buf_f16; /* [max_patches * dim] F16 layernorm out (Blackwell path) */
    CUdeviceptr d_merge_buf;  /* [n_merged * merged_dim] */
    CUdeviceptr d_mm_buf;     /* [n_merged * merged_dim] */
    CUdeviceptr d_mm_out;     /* [n_merged * proj_dim] */
    CUdeviceptr d_rgb;        /* [max_pixels * 3] */
    CUdeviceptr d_rope_cos;   /* [max_patches * head_dim] */
    CUdeviceptr d_rope_sin;   /* [max_patches * head_dim] */
    CUdeviceptr d_pos_map;    /* [max_patches] int */
    CUdeviceptr d_token_perm;     /* [max_patches] int */
    CUdeviceptr d_token_inv_perm; /* [max_patches] int */
    CUdeviceptr d_window_starts;  /* [max_merged] int */
    CUdeviceptr d_window_sizes;   /* [max_merged] int */
    CUdeviceptr d_ds_feats;   /* deepstack feature accumulation */
    CUdeviceptr d_x_f16;      /* [max_patches * max_in] F16 GEMM input (Blackwell path) */
    CUdeviceptr d_ffn_buf_f16; /* [max_patches * ffn_dim] F16 GELU(up) out, fed to ffn_down */
    CUdeviceptr d_ffn_row_max; /* [max_patches] per-row max|X| for FP8 ffn_up (expt) */
    /* Tensor-core full-attention scratch (cuBLAS QK^T -> softmax -> P*V) */
    int tc_attn;              /* 1 if cuBLAS available -> use tensor-core attention */
    int win_tile;             /* 1 -> materialize-scores windowed kernel (VLM_WINDOW_TILE) */
    int flash_full_n;         /* full-attn crossover: N > this -> fused flash kernel
                               * (materialized [N,N] scratch only sized to min(mp,this)) */
    int tc_flash;             /* VLM_TC_FLASH: 1 -> tensor-core attn_prefill_vision_f32
                               * for ALL full-attn layers (A/B vs materialized) */
    CUdeviceptr d_qh_f16;     /* [n_heads * max_patches * head_dim] F16 */
    CUdeviceptr d_kh_f16;     /* [n_heads * max_patches * head_dim] F16 */
    CUdeviceptr d_vh_f16;     /* [n_heads * max_patches * head_dim] F16 */
    CUdeviceptr d_attn_scores;/* [max_patches * max_patches] F32 (one head) */
    CUdeviceptr d_attn_probs; /* [max_patches * max_patches] F16 (one head) */

    /* CUDA graph capture of the ViT block loop (collapses per-layer host launches) */
    int use_graph;            /* VLM_CUDA_GRAPH (default 1) && driver graph API present */
    int capturing;            /* set while the loop is recorded under stream capture */
    CUgraph graph;
    CUgraphExec graph_exec;
    int graph_ready;          /* graph_exec valid for (graph_w, graph_h) */
    int graph_w, graph_h;     /* grid dims (gw, gh) the captured graph is valid for */
    int graph_warm;           /* same-size encodes before capture (>=1 => F16 path settled) */

    /* Host output */
    float *h_output;
    int loaded;
};

/* ======================================================================== */
/* NVRTC compilation                                                        */
/* ======================================================================== */

static int vlm_compile_kernels(cuda_vision_runner *r) {
    size_t len1 = strlen(cuda_kernels_common_src);
    size_t len2 = strlen(cuda_vlm_specific_kernels);
    /* FP8 ffn_up experiment: append the per-row FP8 MMA GEMM source only when
     * enabled, so the default-off build is byte-identical to baseline. */
    size_t len3 = 0, len4 = 0, len5 = 0;
    if (r->ffn_fp8) {
        len3 = strlen(cuda_vlm_fp8_prefix);
        len4 = strlen(fp8_mma_kernels_src);
        len5 = strlen(cuda_vlm_fp8_suffix);
    }
    char *full_src = (char *)malloc(len1 + len2 + len3 + len4 + len5 + 1);
    char *p = full_src;
    memcpy(p, cuda_kernels_common_src, len1); p += len1;
    memcpy(p, cuda_vlm_specific_kernels, len2); p += len2;
    if (r->ffn_fp8) {
        memcpy(p, cuda_vlm_fp8_prefix, len3); p += len3;
        memcpy(p, fp8_mma_kernels_src, len4); p += len4;
        memcpy(p, cuda_vlm_fp8_suffix, len5); p += len5;
    }
    *p = '\0';

    /* Use custom NVRTC compilation without --use_fast_math to ensure
     * correct MMA tensor core behavior on Blackwell (sm_120) */
    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, r->device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, r->device);
    int sm = major * 10 + minor;

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_vlm: compiling kernels for sm_%d ...\n", sm);

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, full_src, "vlm_kernels.cu", 0, NULL, NULL) != NVRTC_SUCCESS) {
        free(full_src);
        return -1;
    }

    char arch[32];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d", sm);
    /* No --use_fast_math: ensures correct F16 MMA behavior */
    const char *opts[] = { arch };
    nvrtcResult nres = nvrtcCompileProgram(prog, 1, opts);

    if (nres != NVRTC_SUCCESS) {
        size_t log_sz;
        nvrtcGetProgramLogSize(prog, &log_sz);
        if (log_sz > 1) {
            char *log = (char *)malloc(log_sz);
            nvrtcGetProgramLog(prog, log);
            fprintf(stderr, "cuda_vlm: NVRTC log:\n%s\n", log);
            free(log);
        }
        nvrtcDestroyProgram(&prog);
        free(full_src);
        return -1;
    }

    size_t ptx_sz;
    nvrtcGetPTXSize(prog, &ptx_sz);
    char *ptx = (char *)malloc(ptx_sz);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);
    free(full_src);

    if (r->verbose >= 3) {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/vlm_kernels.ptx");
        FILE *fp = fopen(path, "w");
        if (fp) { fwrite(ptx, 1, ptx_sz, fp); fclose(fp);
            fprintf(stderr, "cuda_vlm: PTX saved to %s\n", path); }
    }

    {
        CUresult lerr = cuModuleLoadDataEx(&r->module, ptx, 0, NULL, NULL);
        free(ptx);
        if (lerr != CUDA_SUCCESS) return -1;
    }

    CUresult err;
#define GET_FN(name) do { \
    err = cuModuleGetFunction(&r->fn_##name, r->module, #name); \
    if (err != CUDA_SUCCESS) { fprintf(stderr, "cuda_vlm: kernel '%s' not found\n", #name); return -1; } \
} while(0)

    /* Shared kernels */
    GET_FN(layernorm_f32);
    GET_FN(layernorm_f32_f16);
    GET_FN(gemm_f16_f32);
    GET_FN(gelu_f32);
    GET_FN(add_f32);
    GET_FN(add_bias_f32);

    /* Vision-specific kernels */
    GET_FN(gemm_f32_f32);
    GET_FN(patch_embed_dual_f32);
    GET_FN(patch_im2col_f32);
    GET_FN(add_pos_embd);
    GET_FN(add_pos_embd_direct);
    GET_FN(rope_vision_f32);
    GET_FN(attn_full_f32);
    GET_FN(flash_attn_f32);
    GET_FN(attn_prefill_vision_f32);
    GET_FN(attn_full_bq_f32);
    GET_FN(attn_extract_heads);
    GET_FN(attn_softmax_rows);
    GET_FN(attn_window_f32);
    GET_FN(attn_window_warp_f32);
    GET_FN(attn_window_tile_f32);
    GET_FN(reorder_rows_f32);
    GET_FN(spatial_merge_f32);
    GET_FN(cast_f32_f16);
    GET_FN(gelu_f32_to_f16);
    GET_FN(round_f32_to_f16);
    GET_FN(cast_f32_bf16);
    GET_FN(gelu_f32_to_bf16);
    GET_FN(layernorm_f32_bf16);
    GET_FN(attn_extract_heads_bf16);
    GET_FN(attn_softmax_rows_bf16);

    /* FP8 ffn_up experiment kernels (only present when source was appended).
     * Non-fatal: on miss, disable the experiment and fall back to F16 ffn_up. */
    if (r->ffn_fp8) {
        if (cuModuleGetFunction(&r->fn_reduce_max_abs_per_row_f32, r->module,
                                "reduce_max_abs_per_row_f32") != CUDA_SUCCESS ||
            cuModuleGetFunction(&r->fn_gemm_fp8_pipe_perrow_f32, r->module,
                                "gemm_fp8_pipe_perrow_f32") != CUDA_SUCCESS ||
            cuModuleGetFunction(&r->fn_vlm_quant_w_f16_to_e4m3, r->module,
                                "vlm_quant_w_f16_to_e4m3") != CUDA_SUCCESS) {
            fprintf(stderr, "cuda_vlm: FP8 ffn_up kernels not found; disabling VLM_FFN_FP8\n");
            r->ffn_fp8 = 0;
            r->fn_reduce_max_abs_per_row_f32 = 0;
            r->fn_gemm_fp8_pipe_perrow_f32 = 0;
            r->fn_vlm_quant_w_f16_to_e4m3 = 0;
        }
    }

#undef GET_FN

    if (r->verbose >= 1)
        fprintf(stderr, "cuda_vlm: %d kernels compiled (sm_%d)%s\n", 29, sm,
                r->ffn_fp8 ? " [+FP8 ffn_up]" : "");
    return 0;
}

static double vlm_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void vlm_swap_ptrs(CUdeviceptr *a, CUdeviceptr *b) {
    CUdeviceptr tmp = *a;
    *a = *b;
    *b = tmp;
}

static int vlm_build_qwen_window_maps(int gw, int gh, int sm, int patch_size,
                                      int attn_window_size, int use_window_attn,
                                      int *token_perm, int *token_inv_perm,
                                      int *window_starts, int *window_sizes,
                                      int *n_windows_out) {
    const int pw = gw / sm;
    const int ph = gh / sm;
    const int mpow = sm * sm;
    const int n_groups = pw * ph;
    int *group_perm = (int *)malloc((size_t)n_groups * sizeof(int));
    if (!group_perm) return -1;

    int n_windows = 0;
    if (use_window_attn) {
        int grid_window = attn_window_size / patch_size / sm;
        if (grid_window < 1) grid_window = 1;
        int dst = 0;
        for (int y = 0; y < ph; y += grid_window) {
            for (int x = 0; x < pw; x += grid_window) {
                int win_h = (y + grid_window <= ph) ? grid_window : (ph - y);
                int win_w = (x + grid_window <= pw) ? grid_window : (pw - x);
                int dst0 = dst;
                for (int dy = 0; dy < win_h; dy++) {
                    for (int dx = 0; dx < win_w; dx++) {
                        int src = (y + dy) * pw + (x + dx);
                        group_perm[src] = dst++;
                    }
                }
                window_starts[n_windows] = dst0 * mpow;
                window_sizes[n_windows] = (dst - dst0) * mpow;
                n_windows++;
            }
        }
    } else {
        for (int i = 0; i < n_groups; i++) group_perm[i] = i;
        window_starts[0] = 0;
        window_sizes[0] = gw * gh;
        n_windows = 1;
    }

    for (int gy = 0; gy < ph; gy++) {
        for (int gx = 0; gx < pw; gx++) {
            int src_group = gy * pw + gx;
            int dst_group = group_perm[src_group];
            for (int sy = 0; sy < sm; sy++) {
                for (int sx = 0; sx < sm; sx++) {
                    int sub = sy * sm + sx;
                    int src_token = (gy * sm + sy) * gw + (gx * sm + sx);
                    int dst_token = dst_group * mpow + sub;
                    token_perm[dst_token] = src_token;
                    token_inv_perm[src_token] = dst_token;
                }
            }
        }
    }

    free(group_perm);
    *n_windows_out = n_windows;
    return 0;
}

/* ======================================================================== */
/* Weight upload helpers                                                    */
/* ======================================================================== */

/* Helper to find a tensor in GGUF by name */
static int vlm_find_tensor(const gguf_context *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

/* Helper struct for tensor info */
typedef struct {
    const void *data;
    int type;
    int n_cols;
    int n_rows;
    int n_elem;
} vlm_tensor_info;

static vlm_tensor_info vlm_get_tensor(const gguf_context *g, const char *name, int req) {
    vlm_tensor_info t = {0};
    int idx = vlm_find_tensor(g, name);
    if (idx < 0) {
        if (req) fprintf(stderr, "cuda_vlm: missing tensor '%s'\n", name);
        return t;
    }
    t.data = gguf_tensor_data(g, idx);
    t.type = (int)g->tensors[idx].type;
    t.n_cols = (int)g->tensors[idx].dims[0];
    t.n_rows = (g->tensors[idx].n_dims >= 2) ? (int)g->tensors[idx].dims[1] : 1;
    /* Compute total elements as product of all dimensions */
    t.n_elem = 1;
    for (int d = 0; d < (int)g->tensors[idx].n_dims; d++)
        t.n_elem *= (int)g->tensors[idx].dims[d];
    return t;
}

static int vlm_get_int(const gguf_context *g, const char *key, int def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_UINT32) return (int)g->kv[idx].value.u32;
    if (g->kv[idx].type == GGUF_TYPE_INT32) return g->kv[idx].value.i32;
    return def;
}

static float vlm_get_float(const gguf_context *g, const char *key, float def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_FLOAT32) return g->kv[idx].value.f32;
    return def;
}

static const char *vlm_get_string(const gguf_context *g, const char *key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return NULL;
    if (g->kv[idx].type != GGUF_TYPE_STRING) return NULL;
    return g->kv[idx].value.str.str;
}

/* Dequantize full tensor to F32 and upload */
static CUdeviceptr vlm_upload_f32(const vlm_tensor_info *t) {
    if (!t->data) return 0;
    int n = t->n_elem;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        /* Dequantize row by row */
        size_t rb = dequant_row_size(t->type, t->n_cols);
        for (int row = 0; row < t->n_rows; row++) {
            const void *row_data = (const uint8_t *)t->data + row * rb;
            dequant_row(t->type, row_data, buf + row * t->n_cols, t->n_cols);
        }
    }
    CUdeviceptr d;
    if (cuMemAlloc(&d, (size_t)n * sizeof(float)) != CUDA_SUCCESS) { free(buf); return 0; }
    cuMemcpyHtoD(d, buf, (size_t)n * sizeof(float));
    free(buf);
    return d;
}

/* Upload tensor as F16 (converting from F32 if needed) */
static CUdeviceptr vlm_upload_f16(const vlm_tensor_info *t) {
    if (!t->data) return 0;
    int n = t->n_elem;
    if (t->type == GGML_TYPE_F16) {
        /* Direct copy */
        return cu_upload_raw(t->data, (size_t)n * 2);
    }
    /* Convert F32 -> F16 */
    float *f32_buf = NULL;
    if (t->type == GGML_TYPE_F32) {
        f32_buf = (float *)t->data;
    } else {
        /* Dequant to F32 first */
        f32_buf = (float *)malloc((size_t)n * sizeof(float));
        size_t rb = dequant_row_size(t->type, t->n_cols);
        for (int row = 0; row < t->n_rows; row++) {
            const void *row_data = (const uint8_t *)t->data + row * rb;
            dequant_row(t->type, row_data, f32_buf + row * t->n_cols, t->n_cols);
        }
    }
    uint16_t *h16 = (uint16_t *)malloc((size_t)n * 2);
    for (int i = 0; i < n; i++) h16[i] = cu_f32_to_f16(f32_buf[i]);
    if (f32_buf != (float *)t->data) free(f32_buf);
    CUdeviceptr d;
    if (cuMemAlloc(&d, (size_t)n * 2) != CUDA_SUCCESS) { free(h16); return 0; }
    cuMemcpyHtoD(d, h16, (size_t)n * 2);
    free(h16);
    return d;
}

/* Upload tensor as BF16 (converting from other formats if needed) */
static CUdeviceptr vlm_upload_bf16(const vlm_tensor_info *t) {
    if (!t->data) return 0;
    int n = t->n_elem;
    if (t->type == GGML_TYPE_BF16) {
        /* Direct copy */
        return cu_upload_raw(t->data, (size_t)n * 2);
    }
    /* Convert F32/F16 -> BF16 */
    float *f32_buf = NULL;
    if (t->type == GGML_TYPE_F32) {
        f32_buf = (float *)t->data;
    } else if (t->type == GGML_TYPE_F16) {
        f32_buf = (float *)malloc((size_t)n * sizeof(float));
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) f32_buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        /* Dequant to F32 first */
        f32_buf = (float *)malloc((size_t)n * sizeof(float));
        size_t rb = dequant_row_size(t->type, t->n_cols);
        for (int row = 0; row < t->n_rows; row++) {
            const void *row_data = (const uint8_t *)t->data + row * rb;
            dequant_row(t->type, row_data, f32_buf + row * t->n_cols, t->n_cols);
        }
    }
    uint16_t *hbf16 = (uint16_t *)malloc((size_t)n * 2);
    for (int i = 0; i < n; i++) {
        float f = f32_buf[i];
        uint16_t h = 0;
        /* Bit-cast F32 to BF16 via truncation (round-to-nearest-even). */
        uint32_t bits;
        memcpy(&bits, &f, 4);
        uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFFU;
        bits += rounding_bias;
        h = (uint16_t)(bits >> 16);
        hbf16[i] = h;
    }
    if (f32_buf != (float *)t->data) free(f32_buf);
    CUdeviceptr d;
    if (cuMemAlloc(&d, (size_t)n * 2) != CUDA_SUCCESS) { free(hbf16); return 0; }
    cuMemcpyHtoD(d, hbf16, (size_t)n * 2);
    free(hbf16);
    return d;
}

/* Upload a weight matrix (with optional F16/BF16 for performance mode) */
static gpu_weight vlm_upload_weight(const vlm_tensor_info *w, const vlm_tensor_info *b, int use_f16, int use_bf16) {
    gpu_weight gw = {0};
    if (use_bf16) {
        gw.w_bf16 = vlm_upload_bf16(w);
    } else if (use_f16) {
        gw.w_f16 = vlm_upload_f16(w);
    } else {
        gw.w_f32 = vlm_upload_f32(w);
    }
    if (b && b->data) {
        gw.bias = vlm_upload_f32(b);
    }
    return gw;
}

/* Host IEEE-754 half -> float (for computing max|W| at prequant time). */
static float vlm_half_to_float(unsigned short h) {
    unsigned int sign = (unsigned int)(h >> 15) & 1u;
    unsigned int exp  = (unsigned int)(h >> 10) & 0x1Fu;
    unsigned int mant = (unsigned int)h & 0x3FFu;
    unsigned int f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

/* FP8 ffn_up experiment: prequantize an F16 weight [n_out, n_in] to e4m3 with a
 * single per-tensor scale (max|W|/448), padding n_out up to a multiple of 256
 * (the perrow GEMM's W-load has no output-channel bound; pad rows are zero so
 * the read stays in-bounds while writeback stays bounded by the real n_out).
 * Quantization runs on-device so it bit-matches the kernel's cvt.rn.satfinite. */
static void vlm_prequant_ffn_up(cuda_vision_runner *r, gpu_weight *gw,
                                int n_out, int n_in) {
    if (!gw->w_f16 || !r->fn_vlm_quant_w_f16_to_e4m3) return;
    size_t ne = (size_t)n_out * (size_t)n_in;
    unsigned short *h = (unsigned short *)malloc(ne * sizeof(unsigned short));
    if (!h) return;
    if (cuMemcpyDtoH(h, gw->w_f16, ne * sizeof(unsigned short)) != CUDA_SUCCESS) {
        free(h); return;
    }
    float mx = 0.0f;
    for (size_t i = 0; i < ne; i++) {
        float a = fabsf(vlm_half_to_float(h[i]));
        if (a > mx) mx = a;
    }
    free(h);
    float w_scale  = (mx > 448.0f) ? (mx / 448.0f) : 1.0f;
    float inv_wsc  = (mx > 448.0f) ? (448.0f / mx) : 1.0f;
    int n_out_pad  = ((n_out + 255) / 256) * 256;
    CUdeviceptr w_fp8 = 0;
    if (cuMemAlloc(&w_fp8, (size_t)n_out_pad * (size_t)n_in) != CUDA_SUCCESS) return;
    cuMemsetD8(w_fp8, 0, (size_t)n_out_pad * (size_t)n_in);  /* zero pad rows */
    int n = (int)ne;
    void *args[] = { &w_fp8, &gw->w_f16, &n, &inv_wsc };
    cuLaunchKernel(r->fn_vlm_quant_w_f16_to_e4m3, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
    cuStreamSynchronize(r->stream);
    gw->w_fp8 = w_fp8;
    gw->w_scale = w_scale;
    if (r->verbose >= 2)
        fprintf(stderr, "cuda_vlm: ffn_up FP8 prequant n_out=%d(pad %d) n_in=%d max|W|=%.4f w_scale=%.6f\n",
                n_out, n_out_pad, n_in, mx, w_scale);
}

/* ======================================================================== */
/* Public API: init                                                         */
/* ======================================================================== */

cuda_vision_runner *cuda_vision_init(int device_id, int verbose, int use_f16) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuda_vlm: cuew init failed (no CUDA/NVRTC?)\n");
        return NULL;
    }
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_vlm: cuInit failed\n");
        return NULL;
    }

    cuda_vision_runner *r = (cuda_vision_runner *)calloc(1, sizeof(cuda_vision_runner));
    r->verbose = verbose;
    r->use_f16 = use_f16;
    /* BF16 native path: VLM_BF16=1 env var. Mutually exclusive with F16.
     * Loads and computes in BF16 (weights + intermediates) via BF16 cuBLAS. */
    {
        const char *e = getenv("VLM_BF16");
        r->use_bf16 = (e && atoi(e)) ? 1 : 0;
        if (r->use_bf16 && r->use_f16) {
            fprintf(stderr, "cuda_vlm: VLM_BF16 and --f16 are mutually exclusive; disabling BF16\n");
            r->use_bf16 = 0;
        }
    }
    /* FP8 ffn_up gating experiment: only on the F16 path (weights are F16). When
     * off, the FP8 kernel source is NOT appended, so the NVRTC module is
     * byte-identical to the baseline -> zero risk / fully reversible. */
    {
        const char *e = getenv("VLM_FFN_FP8");
        r->ffn_fp8 = (use_f16 && e && atoi(e)) ? 1 : 0;
    }
    /* F16 rounding: VLM_ROUND_F16=1. When active, round GEMM outputs to F16 to match
     * PyTorch's F16 operation ordering (PyTorch stores every tensor in F16). */
    {
        const char *e = getenv("VLM_ROUND_F16");
        r->f16_round = (e && atoi(e)) ? 1 : 0;
    }

    CU_CHECK_NULL(cuDeviceGet(&r->device, device_id));
    CU_CHECK_NULL(cuCtxCreate(&r->context, 0, r->device));
    CU_CHECK_NULL(cuStreamCreate(&r->stream, CU_STREAM_DEFAULT));

    if (cublasewCreate(&r->cublas, r->stream) == 0) {
        r->use_cublas = 1;
        r->cublas_mixed_ok = 1;   /* try mixed F16xF32 first; cleared on first failure */
        r->tc_attn = 1;           /* full attention via cuBLAS tensor cores */
        if (r->verbose >= 1) {
            fprintf(stderr, "cuda_vlm: cuBLAS GEMM fast path enabled\n");
        }
    } else if (r->verbose >= 1) {
        fprintf(stderr, "cuda_vlm: cuBLAS unavailable, using built-in GEMM kernels\n");
    }
    {
        /* Materialize-scores windowed kernel is faster than warp-per-query
         * (98 vs 102 ms @768², 38.7 vs 41.1 @512²) and numerically equivalent;
         * default ON, VLM_WINDOW_TILE=0 falls back to the warp kernel. */
        const char *wt = getenv("VLM_WINDOW_TILE");
        r->win_tile = (wt && wt[0] == '0') ? 0 : 1;
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_vlm: windowed attention = %s kernel\n",
                    r->win_tile ? "tile (materialize-scores)" : "warp-per-query");
    }
    {
        /* Full-attention crossover. The O(N)-memory tensor-core flash
         * (attn_prefill_vision_f32) kicks in for N > this. IMPORTANT: it is a
         * *memory* fallback, NOT a speedup -- benchmarking showed flash is
         * still 4-14%% SLOWER than the materialized tensor-core path (cuBLAS
         * QK^T -> softmax -> P*V) even with shared K/V staging, because cuBLAS's
         * larger tiles / pipelining beat a hand-rolled flash and head_dim=72
         * wastes ~half the 5th k16 fragment. So the default keeps every size
         * that fits VRAM on the fast materialized path; flash only takes over
         * above ~2900² where the [N,N] F32 scratch (>4 GB) would OOM the 16 GB
         * card. VLM_FLASH_FULL forces it on(1)/off(0); VLM_FLASH_FULL_N
         * overrides the crossover N directly (e.g. for A/B testing). */
        int fn = 32768;
        const char *fenv = getenv("VLM_FLASH_FULL_N");
        if (fenv) fn = atoi(fenv);
        const char *ff = getenv("VLM_FLASH_FULL");
        if (ff) fn = (ff[0] == '0') ? (1 << 30) : 0;  /* 0 -> never, else -> always */
        if (fn < 0) fn = 0;
        r->flash_full_n = fn;
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_vlm: full-attention flash crossover N = %d\n",
                    r->flash_full_n);
    }
    {
        /* Tensor-core flash: attn_prefill_vision_f32 (mma.sync, online softmax,
         * O(N) memory). VLM_TC_FLASH=1 routes ALL full-attn layers through it,
         * regardless of the crossover above -- so it can be A/B'd against the
         * materialized cuBLAS path at every size. Default off. */
        const char *tf = getenv("VLM_TC_FLASH");
        r->tc_flash = (tf && tf[0] != '0') ? 1 : 0;
        if (r->tc_flash && r->verbose >= 1)
            fprintf(stderr, "cuda_vlm: tensor-core flash full-attention ENABLED\n");
    }
    {
        /* CUDA-graph capture of the ViT block loop: default ON, VLM_CUDA_GRAPH=0
         * disables. Requires the driver graph API (loaded via cuew) -- guard the
         * pointers so an old driver falls back to per-layer launches. */
        const char *ge = getenv("VLM_CUDA_GRAPH");
        int want = !(ge && ge[0] == '0');
        int api = (cuStreamBeginCapture_v2 && cuStreamEndCapture &&
                   cuGraphInstantiateWithFlags && cuGraphLaunch &&
                   cuGraphExecDestroy && cuGraphDestroy);
        r->use_graph = (want && api) ? 1 : 0;
        r->graph_w = -1;
        r->graph_h = -1;
        if (r->verbose >= 1)
            fprintf(stderr, "cuda_vlm: CUDA graph capture = %s\n",
                    r->use_graph ? "on" : (want ? "off (driver API missing)" : "off"));
    }

    if (vlm_compile_kernels(r) != 0) {
        fprintf(stderr, "cuda_vlm: kernel compilation failed\n");
        if (r->cublas) cublasewDestroy(r->cublas);
        free(r);
        return NULL;
    }

    return r;
}

void cuda_vision_set_max_pixels(cuda_vision_runner *r, int max_pixels) {
    if (r) r->max_pixels = max_pixels;
}

/* ======================================================================== */
/* Public API: load_weights                                                 */
/* ======================================================================== */

int cuda_vision_load_weights(cuda_vision_runner *r, gguf_context *g) {
    if (!r || !g) return -1;
    const char *proj_type = vlm_get_string(g, "clip.projector_type");
    if (!proj_type) proj_type = vlm_get_string(g, "clip.vision.projector_type");

    /* Read hyperparameters */
    r->n_blocks    = vlm_get_int(g, "clip.vision.block_count", 24);
    r->dim         = vlm_get_int(g, "clip.vision.embedding_length", 1024);
    r->n_heads     = vlm_get_int(g, "clip.vision.attention.head_count", 16);
    r->ffn_dim     = vlm_get_int(g, "clip.vision.feed_forward_length", 4096);
    r->patch_size  = vlm_get_int(g, "clip.vision.patch_size", 16);
    r->image_size  = vlm_get_int(g, "clip.vision.image_size", 768);
    r->proj_dim    = vlm_get_int(g, "clip.vision.projection_dim", 2048);
    r->spatial_merge = vlm_get_int(g, "clip.vision.spatial_merge_size", 2);
    r->n_wa_pattern = vlm_get_int(g, "clip.vision.n_wa_pattern", 0);
    r->attn_window_size = vlm_get_int(g, "clip.vision.window_size", 112);
    if (r->n_wa_pattern == 0 && proj_type &&
        (strstr(proj_type, "qwen2.5vl") || strstr(proj_type, "qwen25vl") ||
         strstr(proj_type, "qwen3vl") || strstr(proj_type, "qwen3.5vl") ||
         strstr(proj_type, "qwen3-5"))) {
        r->n_wa_pattern = 4;
    }
    r->ln_eps      = vlm_get_float(g, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    r->head_dim    = r->dim / r->n_heads;

    int ps = r->patch_size;
    int gs = r->image_size / ps;
    r->n_patches = gs * gs;
    r->n_merged  = r->n_patches / (r->spatial_merge * r->spatial_merge);

    /* Compute max buffer sizes for dynamic resolution */
    {
        int mp = r->max_pixels > 0 ? r->max_pixels / (ps * ps) : r->n_patches;
        if (mp < r->n_patches) mp = r->n_patches;
        r->max_patches = mp;
        r->max_merged = mp / (r->spatial_merge * r->spatial_merge);
    }

    /* Image mean/std */
    int idx = gguf_find_key(g, "clip.vision.image_mean");
    if (idx >= 0) {
        float *d = (float *)g->kv[idx].value.arr.data;
        r->image_mean[0] = d[0]; r->image_mean[1] = d[1]; r->image_mean[2] = d[2];
    }
    idx = gguf_find_key(g, "clip.vision.image_std");
    if (idx >= 0) {
        float *d = (float *)g->kv[idx].value.arr.data;
        r->image_std[0] = d[0]; r->image_std[1] = d[1]; r->image_std[2] = d[2];
    }

    fprintf(stderr, "cuda_vlm: dim=%d heads=%d blocks=%d ffn=%d patch=%d image=%d patches=%d merged=%d proj=%d f16=%d bf16=%d max_patches=%d wa_pattern=%d window=%d proj_type=%s\n",
            r->dim, r->n_heads, r->n_blocks, r->ffn_dim,
            r->patch_size, r->image_size, r->n_patches, r->n_merged, r->proj_dim, r->use_f16, r->use_bf16,
            r->max_patches, r->n_wa_pattern, r->attn_window_size,
            proj_type ? proj_type : "(unknown)");

    int dim = r->dim;
    int mp = r->max_patches;
    int max_merged = r->max_merged;
    int sm = r->spatial_merge;
    int merged_dim = dim * sm * sm;

    /* Patch embeddings (always F32 — small, applied once) */
    vlm_tensor_info t_pw0 = vlm_get_tensor(g, "v.patch_embd.weight", 1);
    vlm_tensor_info t_pw1 = vlm_get_tensor(g, "v.patch_embd.weight.1", 0);
    vlm_tensor_info t_pb  = vlm_get_tensor(g, "v.patch_embd.bias", 0);
    r->d_patch_w0 = vlm_upload_f32(&t_pw0);
    r->d_patch_w1 = vlm_upload_f32(&t_pw1);
    r->d_patch_bias = vlm_upload_f32(&t_pb);
    if (t_pw1.data && r->d_patch_w1) {
        /* Fold the dual conv into one weight (both convs hit the same pixels):
         * w0 += w1, so patch embed reduces to a single im2col + GEMM. */
        int n = (int)t_pw0.n_elem;
        int grid = (n + 255) / 256;
        void *args[] = { &r->d_patch_w0, &r->d_patch_w1, &n };
        cuLaunchKernel(r->fn_add_f32, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
        cuStreamSynchronize(r->stream);
        fprintf(stderr, "cuda_vlm: loaded dual conv2d patch embeddings (folded w0+=w1)\n");
    }

    /* Position embedding (always F32) — keep CPU copy for interpolation */
    vlm_tensor_info t_pos = vlm_get_tensor(g, "v.position_embd.weight", 1);
    r->d_pos_embd = vlm_upload_f32(&t_pos);
    r->h_pos_embd = (float *)malloc(t_pos.n_elem * sizeof(float));
    if (t_pos.type == GGML_TYPE_F32) {
        memcpy(r->h_pos_embd, t_pos.data, t_pos.n_elem * sizeof(float));
    } else {
        dequant_row(t_pos.type, t_pos.data, r->h_pos_embd, t_pos.n_elem);
    }

    /* Blocks */
    r->blocks = (gpu_vit_block *)calloc(r->n_blocks, sizeof(gpu_vit_block));
    for (int l = 0; l < r->n_blocks; l++) {
        char name[128];
        gpu_vit_block *blk = &r->blocks[l];

        /* QKV */
        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.weight", l);
        vlm_tensor_info tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.bias", l);
        vlm_tensor_info tb = vlm_get_tensor(g, name, 1);
        blk->attn_qkv = vlm_upload_weight(&tw, &tb, r->use_f16, r->use_bf16);

        /* Attn out */
        snprintf(name, sizeof(name), "v.blk.%d.attn_out.weight", l);
        tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.attn_out.bias", l);
        tb = vlm_get_tensor(g, name, 1);
        blk->attn_out = vlm_upload_weight(&tw, &tb, r->use_f16, r->use_bf16);

        /* FFN up */
        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.weight", l);
        tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.bias", l);
        tb = vlm_get_tensor(g, name, 1);
        blk->ffn_up = vlm_upload_weight(&tw, &tb, r->use_f16, r->use_bf16);
        if (r->ffn_fp8)
            vlm_prequant_ffn_up(r, &blk->ffn_up, tw.n_rows, tw.n_cols);

        /* FFN down */
        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.weight", l);
        tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.bias", l);
        tb = vlm_get_tensor(g, name, 1);
        blk->ffn_down = vlm_upload_weight(&tw, &tb, r->use_f16, r->use_bf16);

        /* LayerNorms (always F32) */
        snprintf(name, sizeof(name), "v.blk.%d.ln1.weight", l);
        vlm_tensor_info tln = vlm_get_tensor(g, name, 1);
        blk->ln1_w = vlm_upload_f32(&tln);
        snprintf(name, sizeof(name), "v.blk.%d.ln1.bias", l);
        tln = vlm_get_tensor(g, name, 1);
        blk->ln1_b = vlm_upload_f32(&tln);

        snprintf(name, sizeof(name), "v.blk.%d.ln2.weight", l);
        tln = vlm_get_tensor(g, name, 1);
        blk->ln2_w = vlm_upload_f32(&tln);
        snprintf(name, sizeof(name), "v.blk.%d.ln2.bias", l);
        tln = vlm_get_tensor(g, name, 1);
        blk->ln2_b = vlm_upload_f32(&tln);
    }

    /* DeepStack */
    idx = gguf_find_key(g, "clip.vision.is_deepstack_layers");
    if (idx >= 0) {
        uint8_t *flags = (uint8_t *)g->kv[idx].value.arr.data;
        int n = (int)g->kv[idx].value.arr.n;
        int ns = 0;
        for (int i = 0; i < n; i++) if (flags[i]) ns++;
        r->n_deepstack = ns;
        r->deepstack_indices = (int *)malloc(ns * sizeof(int));
        r->deepstack = (gpu_deepstack *)calloc(ns, sizeof(gpu_deepstack));
        int si = 0;
        for (int i = 0; i < n; i++) {
            if (!flags[i]) continue;
            r->deepstack_indices[si] = i;
            char name[128];

            snprintf(name, sizeof(name), "v.deepstack.%d.fc1.weight", i);
            vlm_tensor_info tw = vlm_get_tensor(g, name, 1);
            snprintf(name, sizeof(name), "v.deepstack.%d.fc1.bias", i);
            vlm_tensor_info tb = vlm_get_tensor(g, name, 1);
            r->deepstack[si].fc1 = vlm_upload_weight(&tw, &tb, r->use_f16, r->use_bf16);

            snprintf(name, sizeof(name), "v.deepstack.%d.fc2.weight", i);
            tw = vlm_get_tensor(g, name, 1);
            snprintf(name, sizeof(name), "v.deepstack.%d.fc2.bias", i);
            tb = vlm_get_tensor(g, name, 1);
            r->deepstack[si].fc2 = vlm_upload_weight(&tw, &tb, r->use_f16, r->use_bf16);

            snprintf(name, sizeof(name), "v.deepstack.%d.norm.weight", i);
            vlm_tensor_info tln = vlm_get_tensor(g, name, 1);
            r->deepstack[si].norm_w = vlm_upload_f32(&tln);
            snprintf(name, sizeof(name), "v.deepstack.%d.norm.bias", i);
            tln = vlm_get_tensor(g, name, 1);
            r->deepstack[si].norm_b = vlm_upload_f32(&tln);

            si++;
        }
        fprintf(stderr, "cuda_vlm: %d deepstack layers at:", ns);
        for (int i = 0; i < ns; i++) fprintf(stderr, " %d", r->deepstack_indices[i]);
        fprintf(stderr, "\n");
    }

    /* Post LN */
    vlm_tensor_info tln = vlm_get_tensor(g, "v.post_ln.weight", 1);
    r->d_post_ln_w = vlm_upload_f32(&tln);
    tln = vlm_get_tensor(g, "v.post_ln.bias", 1);
    r->d_post_ln_b = vlm_upload_f32(&tln);

    /* MM projection */
    vlm_tensor_info tw, tb;
    tw = vlm_get_tensor(g, "mm.0.weight", 1);
    tb = vlm_get_tensor(g, "mm.0.bias", 1);
    r->mm0 = vlm_upload_weight(&tw, &tb, r->use_f16, r->use_bf16);

    tw = vlm_get_tensor(g, "mm.2.weight", 1);
    tb = vlm_get_tensor(g, "mm.2.bias", 1);
    r->mm2 = vlm_upload_weight(&tw, &tb, r->use_f16, r->use_bf16);

    /* Allocate scratch buffers (sized for max_patches, not n_patches) */
    {
        size_t rgb_pixels = r->max_pixels > 0 ? (size_t)r->max_pixels : (size_t)r->image_size * r->image_size;
        CHECK_CU(cuMemAlloc(&r->d_hidden,    (size_t)mp * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_hidden2,   (size_t)mp * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_qkv,       (size_t)mp * 3 * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_attn_out,  (size_t)mp * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ffn_buf,   (size_t)mp * r->ffn_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_ln_buf,    (size_t)mp * dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_merge_buf, (size_t)max_merged * merged_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_mm_buf,    (size_t)max_merged * merged_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_mm_out,    (size_t)max_merged * r->proj_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_rgb,       rgb_pixels * 3 * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_rope_cos,  (size_t)mp * r->head_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_rope_sin,  (size_t)mp * r->head_dim * sizeof(float)));
        CHECK_CU(cuMemAlloc(&r->d_pos_map,   (size_t)mp * sizeof(int)));
        CHECK_CU(cuMemAlloc(&r->d_token_perm, (size_t)mp * sizeof(int)));
        CHECK_CU(cuMemAlloc(&r->d_token_inv_perm, (size_t)mp * sizeof(int)));
        CHECK_CU(cuMemAlloc(&r->d_window_starts, (size_t)max_merged * sizeof(int)));
        CHECK_CU(cuMemAlloc(&r->d_window_sizes, (size_t)max_merged * sizeof(int)));
        CHECK_CU(cuMemAlloc(&r->d_pos_interp,(size_t)mp * dim * sizeof(float)));
        r->pos_interp_w = -1; r->pos_interp_h = -1; /* no interpolated pos cached yet */
        CHECK_CU(cuMemAlloc(&r->d_patch_pix, (size_t)mp * r->patch_size * r->patch_size * 3 * sizeof(float)));
        /* F16/BF16 GEMM-input scratch for the Blackwell cuBLAS path (cast X to lowp).
         * Largest GEMM input is max(mp*ffn_dim, max_merged*merged_dim) elements. */
        if (r->use_f16 || r->use_bf16) {
            size_t a = (size_t)mp * r->ffn_dim;
            size_t b = (size_t)max_merged * merged_dim;
            size_t n_xf16 = a > b ? a : b;
            CHECK_CU(cuMemAlloc(&r->d_x_f16, n_xf16 * sizeof(unsigned short)));
            /* GELU(ffn_up) output, consumed directly by ffn_down (no recast). */
            CHECK_CU(cuMemAlloc(&r->d_ffn_buf_f16, (size_t)mp * r->ffn_dim * sizeof(unsigned short)));
            /* Layernorm output, consumed directly by qkv/ffn-up GEMM (no recast). */
            CHECK_CU(cuMemAlloc(&r->d_ln_buf_f16, (size_t)mp * dim * sizeof(unsigned short)));
        }
        /* Per-row max|X| buffer for the FP8 ffn_up experiment (one float / token). */
        if (r->ffn_fp8) {
            CHECK_CU(cuMemAlloc(&r->d_ffn_row_max, (size_t)mp * sizeof(float)));
        }
        /* Tensor-core full-attention scratch (F16). Per-head Q/K/V + a single
         * head's [N,N] scores (F32) and probs (F16), reused across heads. Only
         * sized to tc_cap = min(mp, flash_full_n): above the crossover the
         * O(N)-memory tensor-core flash (attn_prefill_vision_f32) runs instead
         * (no [N,N] buffer, only a 9 KB shared K/V tile), so capping here avoids
         * e.g. a 1 GB scores alloc at 2048². Encodes with n_patches <= tc_cap
         * take the materialized path and fit this buffer. */
        if ((r->use_f16 || r->use_bf16) && r->tc_attn) {
            int tc_cap = (mp < r->flash_full_n) ? mp : r->flash_full_n;
            if (tc_cap < 1) tc_cap = 1;  /* forced-flash: keep a minimal valid alloc */
            size_t nhd = (size_t)r->n_heads * tc_cap * r->head_dim;
            CHECK_CU(cuMemAlloc(&r->d_qh_f16, nhd * sizeof(unsigned short)));
            CHECK_CU(cuMemAlloc(&r->d_kh_f16, nhd * sizeof(unsigned short)));
            CHECK_CU(cuMemAlloc(&r->d_vh_f16, nhd * sizeof(unsigned short)));
            CHECK_CU(cuMemAlloc(&r->d_attn_scores, (size_t)tc_cap * tc_cap * sizeof(float)));
            CHECK_CU(cuMemAlloc(&r->d_attn_probs,  (size_t)tc_cap * tc_cap * sizeof(unsigned short)));
        }
    }

    /* DeepStack feature buffer */
    if (r->n_deepstack > 0) {
        CHECK_CU(cuMemAlloc(&r->d_ds_feats,
            (size_t)max_merged * r->n_deepstack * r->proj_dim * sizeof(float)));
    }

    int total_embd = r->proj_dim * (1 + r->n_deepstack);
    r->h_output = (float *)malloc((size_t)max_merged * total_embd * sizeof(float));

    r->loaded = 1;
    fprintf(stderr, "cuda_vlm: weights loaded, VRAM for weights ~%.1f MB\n",
            ((r->use_f16 || r->use_bf16) ? 0.5f : 1.0f) * (float)(
                (size_t)r->n_blocks * (3*dim*dim + dim*dim + r->ffn_dim*dim + dim*r->ffn_dim) +
                merged_dim*merged_dim + r->proj_dim*merged_dim +
                r->n_deepstack * (merged_dim*merged_dim + r->proj_dim*merged_dim)
            ) * sizeof(float) / (1024.0f * 1024.0f));

    return 0;
}

/* ======================================================================== */
/* GEMM dispatch: F32 or F16                                                */
/* ======================================================================== */

/* Launch tanh-GELU in place over Y[n_tok*n_out] (fallback when not fused). */
static void vlm_launch_gelu(cuda_vision_runner *r, CUdeviceptr d_Y, int n) {
    int grid = (n + 255) / 256;
    void *args[] = { &d_Y, &n };
    cuLaunchKernel(r->fn_gelu_f32, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* Launch a GEMM: Y[n_tok, n_out] = X[n_tok, n_in] * W^T[n_out, n_in] + bias.
 * If do_gelu, a tanh-GELU is applied to (Y + bias). On the Blackwell F16xF16
 * cuBLAS path the bias (and gelu) are fused into a cuBLAS-LT epilogue when
 * available, eliminating the separate add_bias / gelu kernel launches. */
static void vlm_gemm_ex(cuda_vision_runner *r, CUdeviceptr d_Y, const gpu_weight *w,
                        CUdeviceptr d_X, int n_tok, int n_out, int n_in, int do_gelu) {
    /* Copy const fields to locals for void* args array */
    CUdeviceptr d_W, d_bias;
    d_bias = w->bias;
    if (r->use_cublas && r->cublas) {
        int ok = -1;
        if (r->use_f16 && w->w_f16) {
            /* Mixed F16(W)xF32(X) GemmEx works pre-Blackwell but is unsupported on
             * sm_120. Try it once; on failure switch permanently to the F16xF16
             * path (cast X to F16 here, weights are already F16). */
            if (r->cublas_mixed_ok) {
                ok = cublasew_gemm_f16_f32_rowmajor_nt(r->cublas, d_Y, w->w_f16, d_X,
                                                       n_tok, n_out, n_in);
                if (ok != 0) {
                    r->cublas_mixed_ok = 0;
                    if (r->verbose >= 1)
                        fprintf(stderr, "cuda_vlm: mixed F16xF32 GEMM unsupported "
                                "(Blackwell), using F16xF16->F32 cuBLAS path\n");
                }
            }
            if (ok != 0 && r->d_x_f16) {
                int total = n_tok * n_in;
                int grid = (total + 255) / 256;
                void *cargs[] = { &r->d_x_f16, &d_X, &total };
                cuLaunchKernel(r->fn_cast_f32_f16, grid, 1, 1, 256, 1, 1,
                               0, r->stream, cargs, NULL);
                /* Fuse bias (+gelu) into a cuBLAS-LT epilogue when possible. */
                if (d_bias && cublasew_lt_available(r->cublas) == 0) {
                    int lt = cublasew_gemm_f16_f16_f32_lt_bias_rowmajor_nt(
                                 r->cublas, d_Y, w->w_f16, r->d_x_f16, d_bias,
                                 do_gelu, /*y_f16=*/0, n_tok, n_out, n_in);
                    if (lt == 0) return;  /* GEMM + bias (+gelu) all fused */
                }
                ok = cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, d_Y, w->w_f16,
                                                           r->d_x_f16, n_tok, n_out, n_in);
            }
        } else if (r->use_bf16 && w->w_bf16 && r->d_x_f16) {
            /* BF16 cuBLAS path: cast X to BF16, then BF16xBF16->F32 GEMM. */
            int total = n_tok * n_in;
            int grid = (total + 255) / 256;
            void *cargs[] = { &r->d_x_f16, &d_X, &total };
            cuLaunchKernel(r->fn_cast_f32_bf16, grid, 1, 1, 256, 1, 1,
                           0, r->stream, cargs, NULL);
            /* Fuse bias (+gelu) into a cuBLAS-LT epilogue when possible. */
            if (d_bias && cublasew_lt_available(r->cublas) == 0) {
                int lt = cublasew_gemm_bf16_bf16_f32_lt_bias_rowmajor_nt(
                             r->cublas, d_Y, w->w_bf16, r->d_x_f16, d_bias,
                             do_gelu, /*y_f16=*/0, n_tok, n_out, n_in);
                if (lt == 0) return;  /* GEMM + bias (+gelu) all fused */
            }
            ok = cublasew_gemm_bf16_bf16_f32_rowmajor_nt(r->cublas, d_Y, w->w_bf16,
                                                         r->d_x_f16, n_tok, n_out, n_in);
        } else if (w->w_f32) {
            ok = cublasew_gemm_f32_rowmajor_nt(r->cublas, d_Y, w->w_f32, d_X,
                                               n_tok, n_out, n_in);
        }
        if (ok == 0) {
            if (d_bias) {
                int total = n_out * n_tok;
                int grid = (total + 255) / 256;
                void *args[] = { &d_Y, &d_bias, &n_out, &n_tok };
                cuLaunchKernel(r->fn_add_bias_f32,
                               grid, 1, 1,
                               256, 1, 1,
                               0, r->stream,
                               args, NULL);
            }
            if (do_gelu) vlm_launch_gelu(r, d_Y, n_tok * n_out);
            return;
        }
        if (r->verbose >= 1) {
            fprintf(stderr, "cuda_vlm: cuBLAS GEMM failed, falling back to built-in kernels\n");
        }
        r->use_cublas = 0;
    }
    if (r->use_f16 && w->w_f16) {
        /* F16 MMA tensor core path */
        d_W = w->w_f16;
        int grid_x = (n_out + 255) / 256;
        int grid_y = (n_tok + 15) / 16;
        size_t smem = 16 * 16 * sizeof(float);
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok };
        cuLaunchKernel(r->fn_gemm_f16_f32,
                       grid_x, grid_y, 1,
                       128, 1, 1,
                       smem, r->stream,
                       args, NULL);
    } else {
        /* F32 tiled path */
        d_W = w->w_f32;
        int grid_x = (n_out + 15) / 16;
        int grid_y = (n_tok + 15) / 16;
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok };
        cuLaunchKernel(r->fn_gemm_f32_f32,
                       grid_x, grid_y, 1,
                       16, 16, 1,
                       0, r->stream,
                       args, NULL);
    }
    if (do_gelu) vlm_launch_gelu(r, d_Y, n_tok * n_out);
}

static void vlm_gemm(cuda_vision_runner *r, CUdeviceptr d_Y, const gpu_weight *w,
                     CUdeviceptr d_X, int n_tok, int n_out, int n_in) {
    vlm_gemm_ex(r, d_Y, w, d_X, n_tok, n_out, n_in, 0);
}

/* Round F32 buffer through F16 precision (in-place): matches PyTorch's F16 model
 * where every stored value is F16. When VLM_ROUND_F16=1, this is called after */
/* each GEMM to introduce F16 rounding at the same points PyTorch does. */
static void vlm_round_f32_to_f16(cuda_vision_runner *r, CUdeviceptr d_buf, int n) {
    if (!r->f16_round || !r->fn_round_f32_to_f16) return;
    int grid = (n + 255) / 256;
    void *args[] = { &d_buf, &n };
    cuLaunchKernel(r->fn_round_f32_to_f16, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* Launch fused tanh-GELU + F32->F16 cast: dst_f16[n] = GELU(src_f32[n]). */
static void vlm_launch_gelu_f16(cuda_vision_runner *r, CUdeviceptr dst_f16,
                                CUdeviceptr src_f32, int n) {
    int grid = (n + 255) / 256;
    void *args[] = { &dst_f16, &src_f32, &n };
    cuLaunchKernel(r->fn_gelu_f32_to_f16, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* Same for BF16: dst_bf16[n] = GELU(src_f32[n]) with BF16 output. */
static void vlm_launch_gelu_bf16(cuda_vision_runner *r, CUdeviceptr dst_bf16,
                                 CUdeviceptr src_f32, int n) {
    int grid = (n + 255) / 256;
    void *args[] = { &dst_bf16, &src_f32, &n };
    cuLaunchKernel(r->fn_gelu_f32_to_bf16, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* GEMM whose input X is ALREADY F16 (skips the F32->F16 cast). Used for the FFN
 * down-projection when the up-projection produced F16 output. Bias is fused via
 * the cuBLAS-LT epilogue when available, else applied with add_bias_f32. */
static void vlm_gemm_x_f16(cuda_vision_runner *r, CUdeviceptr d_Y,
                           const gpu_weight *w, CUdeviceptr d_X_f16,
                           int n_tok, int n_out, int n_in) {
    if (w->bias && cublasew_lt_available(r->cublas) == 0) {
        int lt = cublasew_gemm_f16_f16_f32_lt_bias_rowmajor_nt(
                     r->cublas, d_Y, w->w_f16, d_X_f16, w->bias,
                     /*gelu=*/0, /*y_f16=*/0, n_tok, n_out, n_in);
        if (lt == 0) return;
    }
    int ok = cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, d_Y, w->w_f16,
                                                   d_X_f16, n_tok, n_out, n_in);
    if (ok == 0 && w->bias) {
        int total = n_out * n_tok;
        int grid = (total + 255) / 256;
        CUdeviceptr d_bias = w->bias;
        void *args[] = { &d_Y, &d_bias, &n_out, &n_tok };
        cuLaunchKernel(r->fn_add_bias_f32, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }
}

/* GEMM whose input X is ALREADY BF16 (skips the F32->BF16 cast). Used for the FFN
 * down-projection when the up-projection produced BF16 output. Bias is fused via
 * the cuBLAS-LT epilogue when available, else applied with add_bias_f32. */
static void vlm_gemm_x_bf16(cuda_vision_runner *r, CUdeviceptr d_Y,
                            const gpu_weight *w, CUdeviceptr d_X_bf16,
                            int n_tok, int n_out, int n_in) {
    if (w->bias && cublasew_lt_available(r->cublas) == 0) {
        int lt = cublasew_gemm_bf16_bf16_f32_lt_bias_rowmajor_nt(
                     r->cublas, d_Y, w->w_bf16, d_X_bf16, w->bias,
                     /*gelu=*/0, /*y_f16=*/0, n_tok, n_out, n_in);
        if (lt == 0) return;
    }
    int ok = cublasew_gemm_bf16_bf16_f32_rowmajor_nt(r->cublas, d_Y, w->w_bf16,
                                                     d_X_bf16, n_tok, n_out, n_in);
    if (ok == 0 && w->bias) {
        int total = n_out * n_tok;
        int grid = (total + 255) / 256;
        CUdeviceptr d_bias = w->bias;
        void *args[] = { &d_Y, &d_bias, &n_out, &n_tok };
        cuLaunchKernel(r->fn_add_bias_f32, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }
}

/* ======================================================================== */
/* Public API: encode                                                       */
/* ======================================================================== */

/* Destroy any captured ViT-block graph + its exec (called on size change/teardown). */
static void vlm_graph_reset(cuda_vision_runner *r) {
    if (r->graph_exec) { cuGraphExecDestroy(r->graph_exec); r->graph_exec = NULL; }
    if (r->graph)      { cuGraphDestroy(r->graph);          r->graph = NULL; }
    r->graph_ready = 0;
}

/* Debug: dump GPU buffer to file for comparison (VLM_DUMP_DIR env, only active
 * when set and block matches VLM_DUMP_BLOCK env). */
static void vlm_dump_hidden(cuda_vision_runner *r, CUdeviceptr d_buf,
                             int n_elem, const char *label, int block_idx) {
    const char *dump_dir = getenv("VLM_DUMP_DIR");
    if (!dump_dir || !dump_dir[0]) return;
    const char *blk_str = getenv("VLM_DUMP_BLOCK");
    int target_blk = blk_str ? atoi(blk_str) : -1;
    if (target_blk >= 0 && target_blk != block_idx) return;
    
    float *h = (float *)malloc((size_t)n_elem * sizeof(float));
    if (!h) return;
    cuMemcpyDtoH(h, d_buf, (size_t)n_elem * sizeof(float));
    
    char path[256];
    snprintf(path, sizeof(path), "%s/%s_%d.bin", dump_dir, label, block_idx);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(h, sizeof(float), (size_t)n_elem, f);
        fclose(f);
        fprintf(stderr, "  [DUMP] %s -> %s (%d elements)\n", label, path, n_elem);
    }
    free(h);
}

/* One ViT transformer block, repeated n_blocks times. Factored out of
 * cuda_vision_encode so the loop can be issued normally, recorded under a
 * CUDA-graph stream capture, or skipped in favor of graph replay. Touches only
 * persistent device buffers and r->stream and runs in place on r->d_hidden, so
 * a graph captured here replays correctly for any later image of the same size. */
static void vlm_run_vit_blocks(cuda_vision_runner *r,
                               int n_patches, int n_merged, int dim, int n_heads,
                               int head_dim, int half, int ffn_dim, int merged_dim,
                               int gw, int sm, int use_window_attn,
                               int n_windows, int max_window_tokens,
                               int *ds_count_out) {
    int ds_count = 0;
    for (int l = 0; l < r->n_blocks; l++) {
        if (l == 0 || l == r->n_blocks - 1 || (l + 1) % 6 == 0)
            fprintf(stderr, "  vit block %d/%d\n", l, r->n_blocks);

        gpu_vit_block *blk = &r->blocks[l];

        /* On the Blackwell F16xF16/BF16 cuBLAS path the qkv/ffn-up GEMMs cast their F32
         * activation input to F16/BF16 anyway, so have LayerNorm emit the low-precision
         * dtype directly and feed it via the x- dtype GEMM, folding away the separate
         * cast. BF16 and F16 are mutually exclusive. */
        int use_bf16_ln = (r->use_bf16 && r->use_cublas && r->cublas &&
                           r->d_ln_buf_f16);
        int use_f16_ln = (r->use_f16 && r->use_cublas && r->cublas &&
                          r->d_ln_buf_f16 && !r->cublas_mixed_ok && !use_bf16_ln);

        /* LayerNorm1 (emits BF16/F16 on the Blackwell path to fold the qkv-input cast). */
        {
            float eps = r->ln_eps;
            size_t smem = 256 * sizeof(float);
            CUdeviceptr ln_dst = (use_bf16_ln || use_f16_ln) ? r->d_ln_buf_f16 : r->d_ln_buf;
            CUfunction fn;
            if (use_bf16_ln)       fn = r->fn_layernorm_f32_bf16;
            else if (use_f16_ln)   fn = r->fn_layernorm_f32_f16;
            else                   fn = r->fn_layernorm_f32;
            void *args[] = { &ln_dst, &r->d_hidden, &blk->ln1_w, &blk->ln1_b, &dim, &eps };
            cuLaunchKernel(fn, n_patches, 1, 1, 256, 1, 1, smem, r->stream, args, NULL);
        }

        /* QKV projection */
        {
            int n_out = 3 * dim;
            if (use_bf16_ln)
                vlm_gemm_x_bf16(r, r->d_qkv, &blk->attn_qkv, r->d_ln_buf_f16, n_patches, n_out, dim);
            else if (use_f16_ln)
                vlm_gemm_x_f16(r, r->d_qkv, &blk->attn_qkv, r->d_ln_buf_f16, n_patches, n_out, dim);
            else
                vlm_gemm(r, r->d_qkv, &blk->attn_qkv, r->d_ln_buf, n_patches, n_out, dim);
            /* Round through F16 to match PyTorch's F16 stored precision after GEMM */
            vlm_round_f32_to_f16(r, r->d_qkv, n_patches * n_out);
        }

        /* Debug: check QKV right after GEMM (before RoPE) */
        if (l == 0 && r->verbose >= 2) {
            cuStreamSynchronize(r->stream);
            float dbg[8];
            cuMemcpyDtoH(dbg, r->d_qkv, 8 * sizeof(float));
            fprintf(stderr, "  [DBG] qkv after GEMM (pre-RoPE), block 0: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
            cuMemcpyDtoH(dbg, r->d_ln_buf, 8 * sizeof(float));
            fprintf(stderr, "  [DBG] ln_buf (QKV input), block 0: %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            /* Check weight values */
            if (r->use_f16 && blk->attn_qkv.w_f16) {
                uint16_t wdbg[8];
                cuMemcpyDtoH(wdbg, blk->attn_qkv.w_f16, 8 * sizeof(uint16_t));
                fprintf(stderr, "  [DBG] QKV weight F16[0..3]: %04x %04x %04x %04x (%.6f %.6f %.6f %.6f)\n",
                        wdbg[0], wdbg[1], wdbg[2], wdbg[3],
                        ggml_fp16_to_fp32(wdbg[0]), ggml_fp16_to_fp32(wdbg[1]),
                        ggml_fp16_to_fp32(wdbg[2]), ggml_fp16_to_fp32(wdbg[3]));
            } else if (blk->attn_qkv.w_f32) {
                float wdbg[4];
                cuMemcpyDtoH(wdbg, blk->attn_qkv.w_f32, 4 * sizeof(float));
                fprintf(stderr, "  [DBG] QKV weight F32[0..3]: %.6f %.6f %.6f %.6f\n",
                        wdbg[0], wdbg[1], wdbg[2], wdbg[3]);
            }
        }

        /* M-RoPE on Q and K */
        {
            void *args[] = {
                &r->d_qkv, &r->d_rope_cos, &r->d_rope_sin,
                &n_patches, &n_heads, &dim, &head_dim, &half
            };
            /* Grid-stride: 256 threads/block, enough blocks to cover all
             * (p,h,i) rotation pairs (capped so very large N stays grid-strided). */
            long rope_total = (long)n_patches * n_heads * half;
            unsigned int rope_blocks = (unsigned int)((rope_total + 255) / 256);
            if (rope_blocks > 65535u) rope_blocks = 65535u;
            if (rope_blocks < 1u) rope_blocks = 1u;
            cuLaunchKernel(r->fn_rope_vision_f32,
                           rope_blocks, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }

        /* Multi-head self-attention */
        {
            float scale = 1.0f / sqrtf((float)head_dim);
            int use_local_attn = use_window_attn && ((l + 1) % r->n_wa_pattern != 0);
            if (use_local_attn) {
                void *args[] = {
                    &r->d_attn_out, &r->d_qkv,
                    &r->d_window_starts, &r->d_window_sizes,
                    &dim, &n_heads, &head_dim, &scale
                };
                if (r->win_tile) {
                    /* Materialize-scores windowed attention. Grid (n_windows,
                     * n_heads), block WT_THREADS=192. Dyn smem stages the
                     * window's Q/K/V as F16 + the S[win,win] F32 score matrix
                     * (3*max_window_tokens*head_dim halves + max_window_tokens^2
                     * floats). F16 staging halves the dominant term -> 4 blocks/SM. */
                    size_t smem = (size_t)3 * max_window_tokens * head_dim * sizeof(unsigned short) +
                                  (size_t)max_window_tokens * max_window_tokens * sizeof(float);
                    cuLaunchKernel(r->fn_attn_window_tile_f32,
                                   n_windows, n_heads, 1,
                                   192, 1, 1,   /* WT_THREADS */
                                   smem, r->stream,
                                   args, NULL);
                } else {
                    /* Warp-per-query windowed attention. Grid (n_windows, n_heads),
                     * block WW_WARPS*32. Dyn smem stages the window's K/V
                     * (2 * max_window_tokens * head_dim floats). */
                    size_t smem = (size_t)2 * max_window_tokens * head_dim * sizeof(float);
                    cuLaunchKernel(r->fn_attn_window_warp_f32,
                                   n_windows, n_heads, 1,
                                   12 * 32, 1, 1,   /* must match WW_WARPS */
                                   smem, r->stream,
                                   args, NULL);
                }
            } else if (r->tc_flash ||
                       (r->tc_attn && (r->use_f16 || r->use_bf16) && n_patches > r->flash_full_n)) {
                /* Tensor-core flash full attention: attn_prefill_vision_f32.
                 * mma.sync m16n8k16 (F16 frags, F32 accum) + online softmax,
                 * O(N) memory; reads F32 qkv directly, stages each 16-key K/V
                 * tile into shared. Grid (n_heads, ceil(N/64)); block 128
                 * (4 warps, 16 queries each).
                 *
                 * This is the O(N)-memory fallback above the flash_full_n
                 * crossover (where the materialized path's [N,N] F32 scratch
                 * would OOM the 16 GB card). NOTE: benchmarking showed it is
                 * still 4-14% SLOWER than the materialized cuBLAS path even
                 * with shared K/V staging -- cuBLAS's larger tiles / cp.async
                 * pipelining win, and head_dim=72 wastes ~half the 5th k16
                 * fragment. So it does NOT replace materialized below the
                 * crossover; it is a much better fallback than the CUDA-core
                 * flash_attn_full_f32 (which was 3.6-5.5x slower). VLM_TC_FLASH=1
                 * forces it at every size for A/B testing. */
                int n_qtiles64 = (n_patches + 63) / 64;
                size_t smem = (size_t)2 * 16 * head_dim * sizeof(float);
                void *args[] = {
                    &r->d_attn_out, &r->d_qkv,
                    &n_patches, &dim, &n_heads, &head_dim, &scale
                };
                cuLaunchKernel(r->fn_attn_prefill_vision_f32,
                               n_heads, n_qtiles64, 1,
                               128, 1, 1,
                               smem, r->stream,
                               args, NULL);
            } else if (r->tc_attn && (r->use_f16 || r->use_bf16)) {
                /* Tensor-core full attention: cuBLAS QK^T (F16->F32) -> row
                 * softmax (->F16 probs) -> P*V (F16->F32), one head at a time.
                 * All ops share r->stream so the per-head scores/probs scratch
                 * is safely reused across heads. */
                long total = (long)n_patches * 3 * dim;
                int egrid = (int)((total + 255) / 256);
                void *eargs[] = { &r->d_qh_f16, &r->d_kh_f16, &r->d_vh_f16,
                                  &r->d_qkv, &n_patches, &dim, &n_heads, &head_dim };
                cuLaunchKernel(r->fn_attn_extract_heads, egrid, 1, 1,
                               256, 1, 1, 0, r->stream, eargs, NULL);
                size_t hbytes = (size_t)n_patches * head_dim * sizeof(unsigned short);
                size_t sm_red = 256 * sizeof(float);
                for (int h = 0; h < n_heads; h++) {
                    CUdeviceptr qh = r->d_qh_f16 + (size_t)h * hbytes;
                    CUdeviceptr kh = r->d_kh_f16 + (size_t)h * hbytes;
                    CUdeviceptr vh = r->d_vh_f16 + (size_t)h * hbytes;
                    /* S[N,N] = Q[N,d] * K[N,d]^T  (raw dot; scale folded in softmax) */
                    cublasew_gemm_f16_f16_f32_rowmajor_nt(r->cublas, r->d_attn_scores,
                                                          kh, qh, n_patches, n_patches, head_dim);
                    void *sargs[] = { &r->d_attn_probs, &r->d_attn_scores, &n_patches, &scale };
                    cuLaunchKernel(r->fn_attn_softmax_rows, n_patches, 1, 1,
                                   256, 1, 1, sm_red, r->stream, sargs, NULL);
                    /* O[N,d] = P[N,N] * V[N,d] written into interleaved attn_out (ld=dim) */
                    CUdeviceptr oh = r->d_attn_out + (size_t)h * head_dim * sizeof(float);
                    cublasew_gemm_f16_f16_f32_rowmajor_nn(r->cublas, oh, dim,
                                                          r->d_attn_probs, vh,
                                                          n_patches, head_dim, n_patches);
                }
            } else {
                /* Flash attention (online softmax). FA_WARPS/FA_TILE_K must match the
                 * kernel's #defines. One warp per query; K/V tiled into shared mem. */
                const int fa_warps = 16, fa_tile_k = 16;
                int n_qtiles = (n_patches + fa_warps - 1) / fa_warps;
                size_t smem = (size_t)2 * fa_tile_k * head_dim * sizeof(float);
                void *args[] = {
                    &r->d_attn_out, &r->d_qkv,
                    &n_patches, &dim, &n_heads, &head_dim, &scale
                };
                cuLaunchKernel(r->fn_flash_attn_f32,
                               n_qtiles, n_heads, 1,
                               fa_warps * 32, 1, 1,
                               smem, r->stream,
                               args, NULL);
            }
        }

        /* Round attention output to F16 before the attn_out projection */
        vlm_round_f32_to_f16(r, r->d_attn_out, n_patches * dim);

        /* Attn output projection */
        vlm_gemm(r, r->d_hidden2, &blk->attn_out, r->d_attn_out, n_patches, dim, dim);
        vlm_round_f32_to_f16(r, r->d_hidden2, n_patches * dim);

        /* Residual: hidden += hidden2 */
        {
            int n = n_patches * dim;
            int grid = (n + 255) / 256;
            void *args[] = { &r->d_hidden, &r->d_hidden2, &n };
            cuLaunchKernel(r->fn_add_f32,
                           grid, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }
        vlm_round_f32_to_f16(r, r->d_hidden, n_patches * dim);

        /* FP8 ffn_up experiment: run the up-projection via the per-row FP8 MMA
         * GEMM instead of cuBLAS F16. The kernel reads F32 X, so ln2 must emit
         * F32 here (only ffn_up reads ln2; qkv reads ln1, unaffected). */
        int ffn_up_fp8 = (r->ffn_fp8 && blk->ffn_up.w_fp8 &&
                          r->fn_gemm_fp8_pipe_perrow_f32 &&
                          r->fn_reduce_max_abs_per_row_f32 && n_patches >= 16);
        int use_bf16_ln2 = use_bf16_ln && !ffn_up_fp8;
        int use_f16_ln2 = use_f16_ln && !ffn_up_fp8 && !use_bf16_ln2;

        /* LayerNorm2 (emits BF16/F16 on the Blackwell path to fold the ffn-up-input cast). */
        {
            float eps = r->ln_eps;
            size_t smem = 256 * sizeof(float);
            CUdeviceptr ln_dst = (use_bf16_ln2 || use_f16_ln2) ? r->d_ln_buf_f16 : r->d_ln_buf;
            CUfunction fn;
            if (use_bf16_ln2)      fn = r->fn_layernorm_f32_bf16;
            else if (use_f16_ln2)  fn = r->fn_layernorm_f32_f16;
            else                   fn = r->fn_layernorm_f32;
            void *args[] = { &ln_dst, &r->d_hidden, &blk->ln2_w, &blk->ln2_b, &dim, &eps };
            cuLaunchKernel(fn, n_patches, 1, 1, 256, 1, 1, smem, r->stream, args, NULL);
        }

        /* FFN: up -> GELU -> down. On the Blackwell BF16/F16 cuBLAS path the up-proj uses
         * a BIAS-only epilogue (fused INLINE in the GEMM, no side kernel), then a
         * single gelu+cast kernel writes the low-precision dtype straight into
         * d_ffn_buf_f16 so the down-proj reads it with no separate F32->lowp recast.
         * GELU_BIAS-with-BF16/F16-output is unsupported by cuBLASLt here, hence this
         * split. The up-proj reads the LN output directly when present, folding its
         * input cast too. Otherwise: fused/separate F32. */
        int use_bf16_ffn = (r->use_bf16 && r->use_cublas && r->cublas &&
                            blk->ffn_down.w_bf16 && r->d_ffn_buf_f16 &&
                            cublasew_lt_available(r->cublas) == 0);
        int ffn_f16_path = (r->use_f16 && r->use_cublas && r->cublas &&
                            blk->ffn_down.w_f16 && r->d_ffn_buf_f16 &&
                            !r->cublas_mixed_ok && cublasew_lt_available(r->cublas) == 0 &&
                            !use_bf16_ffn);
        if (ffn_up_fp8) {
            /* up-proj via per-row FP8 MMA: reduce per-row max|X| (F32 ln2 output),
             * then the FP8 GEMM writes F32 into d_ffn_buf. n_out is passed as the
             * real ffn_dim while ffn_up.w_fp8 is zero-padded to a *256 row count,
             * so the W-load stays in-bounds and Y is naturally [n_patches,ffn_dim].
             * GELU+cast to F16 then the F16 down-proj are identical to the fast path. */
            CUdeviceptr d_X = r->d_ln_buf;
            void *rargs[] = { &r->d_ffn_row_max, &d_X, &n_patches, &dim };
            cuLaunchKernel(r->fn_reduce_max_abs_per_row_f32, (unsigned)n_patches, 1, 1,
                           256, 1, 1, 0, r->stream, rargs, NULL);
            CUdeviceptr d_Y = r->d_ffn_buf, d_W = blk->ffn_up.w_fp8, d_B = blk->ffn_up.bias;
            float ws = blk->ffn_up.w_scale;
            int n_out = ffn_dim, n_in = dim, n_tok = n_patches;
            void *gargs[] = { &d_Y, &d_W, &d_X, &d_B, &n_out, &n_in, &n_tok,
                              &ws, &r->d_ffn_row_max };
            unsigned gx = (unsigned)((ffn_dim + 255) / 256); gx = (gx + 3u) & ~3u;
            unsigned gy = (unsigned)((n_patches + 31) / 32); gy = (gy + 3u) & ~3u;
            size_t smem_pr = 1024 + 8192 * 2 + 256;   /* smX + 2 W stages + 64 scales */
            cuLaunchKernel(r->fn_gemm_fp8_pipe_perrow_f32, gx, gy, 1, 128, 1, 1,
                           smem_pr, r->stream, gargs, NULL);
            vlm_launch_gelu_f16(r, r->d_ffn_buf_f16, r->d_ffn_buf, n_patches * ffn_dim);
            vlm_gemm_x_f16(r, r->d_hidden2, &blk->ffn_down, r->d_ffn_buf_f16,
                           n_patches, dim, ffn_dim);
        } else if (use_bf16_ffn) {
            /* BF16 optimized path (mirrors the F16 path but with BF16 intermediates). */
            if (use_bf16_ln)
                vlm_gemm_x_bf16(r, r->d_ffn_buf, &blk->ffn_up, r->d_ln_buf_f16,
                                n_patches, ffn_dim, dim);
            else
                vlm_gemm_ex(r, r->d_ffn_buf, &blk->ffn_up, r->d_ln_buf,
                            n_patches, ffn_dim, dim, 0);
            vlm_launch_gelu_bf16(r, r->d_ffn_buf_f16, r->d_ffn_buf, n_patches * ffn_dim);
            vlm_gemm_x_bf16(r, r->d_hidden2, &blk->ffn_down, r->d_ffn_buf_f16,
                            n_patches, dim, ffn_dim);
        } else if (ffn_f16_path) {
            /* up-proj (bias fused inline, no gelu) reads the F16 LayerNorm output
             * directly when present (folds its input cast); then a single gelu+cast
             * kernel writes F16 into d_ffn_buf_f16 for the down-proj. (We cannot emit
             * F16 from the GEMM epilogue here: cuBLASLt has no F16-output BIAS/GELU
             * epilogue on sm_120, so this fused gelu+cast pass stays.) */
            if (use_f16_ln)
                vlm_gemm_x_f16(r, r->d_ffn_buf, &blk->ffn_up, r->d_ln_buf_f16, n_patches, ffn_dim, dim);
            else
                vlm_gemm_ex(r, r->d_ffn_buf, &blk->ffn_up, r->d_ln_buf, n_patches, ffn_dim, dim, 0);
            vlm_launch_gelu_f16(r, r->d_ffn_buf_f16, r->d_ffn_buf, n_patches * ffn_dim);
            vlm_gemm_x_f16(r, r->d_hidden2, &blk->ffn_down, r->d_ffn_buf_f16,
                           n_patches, dim, ffn_dim);
        } else {
            if (use_f16_ln) {
                /* F16 ln output, but no fused-gelu F16 down path: gelu separately. */
                vlm_gemm_x_f16(r, r->d_ffn_buf, &blk->ffn_up, r->d_ln_buf_f16, n_patches, ffn_dim, dim);
                vlm_launch_gelu(r, r->d_ffn_buf, n_patches * ffn_dim);
            } else {
                vlm_gemm_ex(r, r->d_ffn_buf, &blk->ffn_up, r->d_ln_buf, n_patches, ffn_dim, dim, 1);
            }
            vlm_gemm(r, r->d_hidden2, &blk->ffn_down, r->d_ffn_buf, n_patches, dim, ffn_dim);
        }

        /* Residual: hidden += hidden2 */
        {
            int n = n_patches * dim;
            int grid = (n + 255) / 256;
            void *args[] = { &r->d_hidden, &r->d_hidden2, &n };
            cuLaunchKernel(r->fn_add_f32,
                           grid, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }
        vlm_round_f32_to_f16(r, r->d_hidden, n_patches * dim);

        /* Debug dump: compare with PyTorch intermediates */
        if (!r->use_graph || !r->graph_ready)
            vlm_dump_hidden(r, r->d_hidden, n_patches * dim, "hidden", l);

        /* Debug: check hidden and qkv after first block */
        if (l == 0 && r->verbose >= 2) {
            cuStreamSynchronize(r->stream);
            float dbg[8];
            cuMemcpyDtoH(dbg, r->d_hidden, 8 * sizeof(float));
            fprintf(stderr, "  [DBG] hidden after block 0: %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            cuMemcpyDtoH(dbg, r->d_qkv, 8 * sizeof(float));
            fprintf(stderr, "  [DBG] qkv[0..3] after block 0: %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }

        /* DeepStack extraction */
        for (int ds = 0; ds < r->n_deepstack; ds++) {
            if (r->deepstack_indices[ds] != l) continue;

            fprintf(stderr, "  deepstack at layer %d\n", l);
            gpu_deepstack *dsl = &r->deepstack[ds];

            /* Spatial merge current hidden -> merge_buf */
            {
                void *args[] = { &r->d_merge_buf, &r->d_hidden, &gw, &sm, &dim };
                cuLaunchKernel(r->fn_spatial_merge_f32,
                               n_merged, 1, 1,
                               256, 1, 1,
                               0, r->stream,
                               args, NULL);
            }

            /* LayerNorm on merge_buf */
            {
                float eps = r->ln_eps;
                size_t smem = 256 * sizeof(float);
                void *args[] = { &r->d_merge_buf, &r->d_merge_buf,
                                 &dsl->norm_w, &dsl->norm_b, &merged_dim, &eps };
                cuLaunchKernel(r->fn_layernorm_f32,
                               n_merged, 1, 1,
                               256, 1, 1,
                               smem, r->stream,
                               args, NULL);
            }

            /* fc1: [merged_dim -> merged_dim], bias+GELU fused into epilogue */
            vlm_gemm_ex(r, r->d_mm_buf, &dsl->fc1, r->d_merge_buf, n_merged, merged_dim, merged_dim, 1);

            /* fc2: [merged_dim -> proj_dim] */
            vlm_gemm(r, r->d_mm_out, &dsl->fc2, r->d_mm_buf, n_merged, r->proj_dim, merged_dim);

            /* Copy to deepstack feature buffer at offset ds_count */
            {
                size_t offset = (size_t)ds_count * n_merged * r->proj_dim * sizeof(float);
                CUdeviceptr dst = r->d_ds_feats + offset;
                cuMemcpyDtoDAsync(dst, r->d_mm_out,
                                   (size_t)n_merged * r->proj_dim * sizeof(float),
                                   r->stream);
            }
            ds_count++;
        }
    }
    if (ds_count_out) *ds_count_out = ds_count;
}

float *cuda_vision_encode(cuda_vision_runner *r, const float *rgb_norm, int width, int height) {
    if (!r || !r->loaded) return NULL;

    int ps = r->patch_size;
    int dim = r->dim;
    int n_heads = r->n_heads;
    int head_dim = r->head_dim;
    int ffn_dim = r->ffn_dim;
    int gw = width / ps;
    int gh = height / ps;
    int n_patches = gw * gh;
    int sm = r->spatial_merge;
    int merged_dim = dim * sm * sm;
    int n_merged = n_patches / (sm * sm);
    int use_window_attn = (r->n_wa_pattern > 0);
    int n_windows = 0;
    int max_window_tokens = 0;
    double t_patch = 0.0, t_pos = 0.0, t_rope = 0.0, t_vit = 0.0;
    double t_postln = 0.0, t_merge = 0.0, t_mmproj = 0.0;
    double t0;

    if (n_patches > r->max_patches) {
        fprintf(stderr, "cuda_vlm: too many patches %d (max %d)\n", n_patches, r->max_patches);
        return NULL;
    }

    fprintf(stderr, "cuda_vlm: encoding %dx%d image (%d patches, %d merged tokens)\n",
            width, height, n_patches, n_merged);
    if (use_window_attn) {
        fprintf(stderr, "cuda_vlm: Qwen window attention enabled (pattern=%d window=%d)\n",
                r->n_wa_pattern, r->attn_window_size);
    }

    /* 1. Upload RGB to GPU */
    cuMemcpyHtoD(r->d_rgb, rgb_norm, (size_t)width * height * 3 * sizeof(float));

    /* 2. Patch embedding: im2col + GEMM (w1 already folded into w0 at load). */
    fprintf(stderr, "  patch embedding (im2col + GEMM)...\n");
    t0 = vlm_time_ms();
    {
        int img_w = width;
        int ks = ps * ps * 3;
        void *im_args[] = { &r->d_patch_pix, &r->d_rgb, &gw, &ps, &img_w };
        cuLaunchKernel(r->fn_patch_im2col_f32,
                       n_patches, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       im_args, NULL);
        gpu_weight pw = { .w_f32 = r->d_patch_w0, .w_f16 = 0, .bias = r->d_patch_bias };
        vlm_gemm(r, r->d_hidden, &pw, r->d_patch_pix, n_patches, dim, ks);
    }
    cuStreamSynchronize(r->stream);
    t_patch = vlm_time_ms() - t0;

    /* Round to F16 precision (matches PyTorch's F16 stored precision). */
    vlm_round_f32_to_f16(r, r->d_hidden, n_patches * dim);

    /* Debug dump: compare with PyTorch intermediates */
    if (!r->use_graph || !r->graph_ready)
        vlm_dump_hidden(r, r->d_hidden, n_patches * dim, "patch_embed", 0);

    /* Debug: check patch embedding output */
    if (r->verbose >= 2) {
        cuStreamSynchronize(r->stream);
        float dbg[8];
        cuMemcpyDtoH(dbg, r->d_hidden, 8 * sizeof(float));
        fprintf(stderr, "  [DBG] hidden after patch_embed: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
    }

    /* 3. Position embeddings (bilinear interpolation for dynamic resolution) */
    fprintf(stderr, "  position embeddings...\n");
    t0 = vlm_time_ms();
    {
        int orig_gw = r->image_size / ps;
        int orig_gh = orig_gw;  /* original grid is square */

        if (gw == orig_gw && gh == orig_gh) {
            /* Exact match: use direct indirection (no interpolation needed) */
            int *pos_map = (int *)malloc(n_patches * sizeof(int));
            for (int py = 0; py < gh; py++)
                for (int px = 0; px < gw; px++)
                    pos_map[py * gw + px] = py * orig_gw + px;
            cuMemcpyHtoD(r->d_pos_map, pos_map, n_patches * sizeof(int));
            free(pos_map);

            void *args[] = { &r->d_hidden, &r->d_pos_embd, &r->d_pos_map, &dim };
            cuLaunchKernel(r->fn_add_pos_embd,
                           n_patches, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        } else {
            /* Bilinear interpolation on CPU, upload to d_pos_interp. The result is
             * size-invariant, so cache it: d_pos_interp persists across encodes and
             * nothing else writes it, so a repeated (gw,gh) reuses the resident buffer
             * and skips the ~1M-elem CPU loop + the multi-MB HtoD (the dominant cost
             * at non-native sizes, e.g. ~2.9 ms/encode at 512²). */
            if (r->pos_interp_w != gw || r->pos_interp_h != gh) {
                fprintf(stderr, "  interpolating pos embedding: %dx%d -> %dx%d\n",
                        orig_gw, orig_gh, gw, gh);
                float *interp = (float *)malloc((size_t)n_patches * dim * sizeof(float));
                for (int py = 0; py < gh; py++) {
                    float sy = (float)py * (orig_gh - 1) / (gh > 1 ? gh - 1 : 1);
                    int y0 = (int)sy, y1 = (y0 + 1 < orig_gh) ? y0 + 1 : y0;
                    float wy = sy - y0;
                    for (int px = 0; px < gw; px++) {
                        float sx = (float)px * (orig_gw - 1) / (gw > 1 ? gw - 1 : 1);
                        int x0 = (int)sx, x1 = (x0 + 1 < orig_gw) ? x0 + 1 : x0;
                        float wx = sx - x0;
                        int dst_idx = (py * gw + px) * dim;
                        int s00 = (y0 * orig_gw + x0) * dim;
                        int s01 = (y0 * orig_gw + x1) * dim;
                        int s10 = (y1 * orig_gw + x0) * dim;
                        int s11 = (y1 * orig_gw + x1) * dim;
                        for (int d = 0; d < dim; d++) {
                            interp[dst_idx + d] =
                                r->h_pos_embd[s00+d] * (1-wy)*(1-wx) +
                                r->h_pos_embd[s01+d] * (1-wy)*wx +
                                r->h_pos_embd[s10+d] * wy*(1-wx) +
                                r->h_pos_embd[s11+d] * wy*wx;
                        }
                    }
                }
                cuMemcpyHtoD(r->d_pos_interp, interp, (size_t)n_patches * dim * sizeof(float));
                free(interp);
                r->pos_interp_w = gw; r->pos_interp_h = gh;
            }

            /* Add interpolated pos embedding directly (no pos_map needed) */
            void *args[] = { &r->d_hidden, &r->d_pos_interp, &dim, &n_patches };
            cuLaunchKernel(r->fn_add_pos_embd_direct,
                           n_patches, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }
    }
    cuStreamSynchronize(r->stream);
    t_pos = vlm_time_ms() - t0;
    /* Round to F16 after pos embed (matches PyTorch F16 precision) */
    vlm_round_f32_to_f16(r, r->d_hidden, n_patches * dim);

    /* 3b. Match llama.cpp token ordering for Qwen window attention. */
    {
        int *token_perm = (int *)malloc((size_t)n_patches * sizeof(int));
        int *token_inv_perm = (int *)malloc((size_t)n_patches * sizeof(int));
        int *window_starts = (int *)malloc((size_t)n_merged * sizeof(int));
        int *window_sizes = (int *)malloc((size_t)n_merged * sizeof(int));
        if (!token_perm || !token_inv_perm || !window_starts || !window_sizes) {
            free(token_perm);
            free(token_inv_perm);
            free(window_starts);
            free(window_sizes);
            fprintf(stderr, "cuda_vlm: failed to allocate window maps\n");
            return NULL;
        }

        if (vlm_build_qwen_window_maps(gw, gh, sm, ps, r->attn_window_size, use_window_attn,
                                       token_perm, token_inv_perm,
                                       window_starts, window_sizes, &n_windows) != 0) {
            free(token_perm);
            free(token_inv_perm);
            free(window_starts);
            free(window_sizes);
            fprintf(stderr, "cuda_vlm: failed to build window maps\n");
            return NULL;
        }

        cuMemcpyHtoD(r->d_token_perm, token_perm, (size_t)n_patches * sizeof(int));
        cuMemcpyHtoD(r->d_token_inv_perm, token_inv_perm, (size_t)n_patches * sizeof(int));
        cuMemcpyHtoD(r->d_window_starts, window_starts, (size_t)n_windows * sizeof(int));
        cuMemcpyHtoD(r->d_window_sizes, window_sizes, (size_t)n_windows * sizeof(int));
        for (int i = 0; i < n_windows; i++) {
            if (window_sizes[i] > max_window_tokens) max_window_tokens = window_sizes[i];
        }
        if (max_window_tokens == 0) max_window_tokens = n_patches;

        if (use_window_attn) {
            void *args[] = { &r->d_hidden2, &r->d_hidden, &r->d_token_perm, &dim };
            cuLaunchKernel(r->fn_reorder_rows_f32,
                           n_patches, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
            vlm_swap_ptrs(&r->d_hidden, &r->d_hidden2);
        }

        free(token_perm);
        free(token_inv_perm);
        free(window_starts);
        free(window_sizes);
    }

    /* 4. Precompute M-RoPE cos/sin on host, upload to GPU */
    t0 = vlm_time_ms();
    {
        int half = head_dim / 2;
        int sect_size = head_dim / 4;
        float freq_base = 10000.0f;
        float theta_scale = powf(freq_base, -2.0f / (float)half);
        float *rope_cos = (float *)malloc(n_patches * head_dim * sizeof(float));
        float *rope_sin = (float *)malloc(n_patches * head_dim * sizeof(float));
        int *token_perm = (int *)malloc((size_t)n_patches * sizeof(int));

        cuMemcpyDtoH(token_perm, r->d_token_perm, (size_t)n_patches * sizeof(int));

        for (int p = 0; p < n_patches; p++) {
            int src = use_window_attn ? token_perm[p] : p;
            int py = src / gw;
            int px = src % gw;
            float p_t = (float)py, p_h = (float)px, p_w = (float)py, p_e = (float)px;
            float cur_t = p_t, cur_h = p_h, cur_w = p_w, cur_e = p_e;

            for (int i0 = 0; i0 < head_dim; i0 += 2) {
                int sector = i0 / 2;
                if (sector == 0) cur_t = p_t;
                if (sector == sect_size) cur_h = p_h;
                if (sector == 2 * sect_size) cur_w = p_w;
                if (sector == 3 * sect_size) cur_e = p_e;

                float theta;
                if (sector < sect_size) theta = cur_t;
                else if (sector < 2 * sect_size) theta = cur_h;
                else if (sector < 3 * sect_size) theta = cur_w;
                else theta = cur_e;

                rope_cos[p * head_dim + i0] = cosf(theta);
                rope_sin[p * head_dim + i0] = sinf(theta);
                rope_cos[p * head_dim + i0 + 1] = cosf(theta);
                rope_sin[p * head_dim + i0 + 1] = sinf(theta);

                cur_t *= theta_scale;
                cur_h *= theta_scale;
                cur_w *= theta_scale;
                cur_e *= theta_scale;
            }
        }

        cuMemcpyHtoD(r->d_rope_cos, rope_cos, n_patches * head_dim * sizeof(float));
        cuMemcpyHtoD(r->d_rope_sin, rope_sin, n_patches * head_dim * sizeof(float));
        free(token_perm);
        free(rope_cos);
        free(rope_sin);
    }
    cuStreamSynchronize(r->stream);
    t_rope = vlm_time_ms() - t0;

    /* 5. ViT blocks */
    int half = head_dim / 2;
    int ds_count = 0;
    t0 = vlm_time_ms();

    /* The 27 ViT blocks issue ~288 host launches/encode (per-head full attention),
     * which left the GPU ~20 ms idle/encode (host-launch-bound). Capture the loop
     * once into a CUDA graph per image size, then replay it. Encode #1 runs normally
     * (settles cublas_mixed_ok, warms cuBLAS algos/workspace); encode #2 captures
     * (only when verbose<2 -- the in-loop debug syncs/DtoH would break capture -- and
     * no DeepStack, whose host-side ds_count cannot survive replay); #3+ replay. The
     * pre/post window reorder+swaps are net-identity per encode, so the loop sees the
     * same physical d_hidden/d_hidden2 every time and the baked graph pointers stay
     * valid. A capture or instantiate failure disables graphs and falls back to the
     * per-layer launches. */
    int gph = (r->use_graph && r->n_deepstack == 0);
    if (gph && (r->graph_w != gw || r->graph_h != gh)) {
        vlm_graph_reset(r);
        r->graph_w = gw; r->graph_h = gh; r->graph_warm = 0;
    }
    if (gph && r->graph_ready) {
        cuGraphLaunch(r->graph_exec, r->stream);
    } else if (gph && r->graph_warm >= 1 && r->verbose < 2 && r->use_cublas && r->cublas &&
               cuStreamBeginCapture_v2(r->stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) == CUDA_SUCCESS) {
        r->capturing = 1;
        vlm_run_vit_blocks(r, n_patches, n_merged, dim, n_heads, head_dim, half,
                           ffn_dim, merged_dim, gw, sm, use_window_attn,
                           n_windows, max_window_tokens, &ds_count);
        r->capturing = 0;
        CUgraph g = NULL;
        CUresult ec = cuStreamEndCapture(r->stream, &g);
        if (ec == CUDA_SUCCESS && g &&
            cuGraphInstantiateWithFlags(&r->graph_exec, g, 0) == CUDA_SUCCESS) {
            r->graph = g;
            r->graph_ready = 1;
            if (r->verbose >= 1)
                fprintf(stderr, "cuda_vlm: CUDA graph captured (%dx%d, %d blocks)\n",
                        gw * ps, gh * ps, r->n_blocks);
            cuGraphLaunch(r->graph_exec, r->stream);  /* capture only records; run it now */
        } else {
            if (g) cuGraphDestroy(g);
            r->use_graph = 0;
            fprintf(stderr, "cuda_vlm: CUDA graph capture failed (ec=%d), "
                    "using per-layer launches\n", (int)ec);
            vlm_run_vit_blocks(r, n_patches, n_merged, dim, n_heads, head_dim, half,
                               ffn_dim, merged_dim, gw, sm, use_window_attn,
                               n_windows, max_window_tokens, &ds_count);
        }
    } else {
        vlm_run_vit_blocks(r, n_patches, n_merged, dim, n_heads, head_dim, half,
                           ffn_dim, merged_dim, gw, sm, use_window_attn,
                           n_windows, max_window_tokens, &ds_count);
        if (gph) r->graph_warm++;
    }
    cuStreamSynchronize(r->stream);
    t_vit = vlm_time_ms() - t0;

    if (use_window_attn) {
        void *args[] = { &r->d_hidden2, &r->d_hidden, &r->d_token_inv_perm, &dim };
        cuLaunchKernel(r->fn_reorder_rows_f32,
                       n_patches, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
        vlm_swap_ptrs(&r->d_hidden, &r->d_hidden2);
        cuStreamSynchronize(r->stream);
    }

    /* 6. Post LayerNorm */
    fprintf(stderr, "  post layernorm...\n");
    t0 = vlm_time_ms();
    {
        float eps = r->ln_eps;
        size_t smem = 256 * sizeof(float);
        void *args[] = { &r->d_hidden, &r->d_hidden, &r->d_post_ln_w, &r->d_post_ln_b, &dim, &eps };
        cuLaunchKernel(r->fn_layernorm_f32,
                       n_patches, 1, 1,
                       256, 1, 1,
                       smem, r->stream,
                       args, NULL);
    }
    /* Debug dump: post-LN hidden state */
    if (!r->use_graph || !r->graph_ready)
        vlm_dump_hidden(r, r->d_hidden, n_patches * dim, "postln", 0);
    cuStreamSynchronize(r->stream);
    t_postln = vlm_time_ms() - t0;

    /* 7. Final spatial merge */
    fprintf(stderr, "  spatial merge...\n");
    t0 = vlm_time_ms();
    {
        void *args[] = { &r->d_merge_buf, &r->d_hidden, &gw, &sm, &dim };
        cuLaunchKernel(r->fn_spatial_merge_f32,
                       n_merged, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    }
    cuStreamSynchronize(r->stream);
    t_merge = vlm_time_ms() - t0;

    /* 8. MM projection: mm.0 -> GELU -> mm.2, bias+GELU fused into mm.0 epilogue */
    fprintf(stderr, "  mm projection...\n");
    t0 = vlm_time_ms();
    vlm_gemm_ex(r, r->d_mm_buf, &r->mm0, r->d_merge_buf, n_merged, merged_dim, merged_dim, 1);

    vlm_gemm(r, r->d_mm_out, &r->mm2, r->d_mm_buf, n_merged, r->proj_dim, merged_dim);
    cuStreamSynchronize(r->stream);
    t_mmproj = vlm_time_ms() - t0;

    /* Debug: check mm_out */
    if (r->verbose >= 2) {
        cuStreamSynchronize(r->stream);
        float dbg[8];
        cuMemcpyDtoH(dbg, r->d_mm_out, 8 * sizeof(float));
        fprintf(stderr, "  [DBG] mm_out: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
        cuMemcpyDtoH(dbg, r->d_merge_buf, 8 * sizeof(float));
        fprintf(stderr, "  [DBG] merge_buf: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
        cuMemcpyDtoH(dbg, r->d_mm_buf, 8 * sizeof(float));
        fprintf(stderr, "  [DBG] mm_buf (after gelu): %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
    }

    /* 9. Synchronize and copy results to host */
    cuStreamSynchronize(r->stream);

    int total_embd = r->proj_dim * (1 + r->n_deepstack);
    float *result = (float *)malloc((size_t)n_merged * total_embd * sizeof(float));

    if (ds_count == 0) {
        /* No DeepStack: the output IS d_mm_out, so DtoH straight into result --
         * skips the staging malloc, the 11.8 MB zero-fill, and the interleave copy
         * the general path below would do (this was ~half the untimed host tail). */
        cuMemcpyDtoH(result, r->d_mm_out, (size_t)n_merged * r->proj_dim * sizeof(float));
    } else {
        /* DeepStack: interleave [main, ds0, ds1, ...] per token from staged copies. */
        float *mm_host = (float *)malloc((size_t)n_merged * r->proj_dim * sizeof(float));
        float *ds_host = (float *)malloc((size_t)ds_count * n_merged * r->proj_dim * sizeof(float));
        cuMemcpyDtoH(mm_host, r->d_mm_out, (size_t)n_merged * r->proj_dim * sizeof(float));
        cuMemcpyDtoH(ds_host, r->d_ds_feats,
                     (size_t)ds_count * n_merged * r->proj_dim * sizeof(float));
        for (int t = 0; t < n_merged; t++) {
            float *dst = result + t * total_embd;
            memcpy(dst, mm_host + t * r->proj_dim, r->proj_dim * sizeof(float));
            for (int d = 0; d < ds_count; d++) {
                memcpy(dst + (1 + d) * r->proj_dim,
                       ds_host + d * n_merged * r->proj_dim + t * r->proj_dim,
                       r->proj_dim * sizeof(float));
            }
        }
        free(mm_host);
        free(ds_host);
    }

    if (r->verbose >= 1) {
        fprintf(stderr,
                "cuda_vlm: profile patch=%.1f ms pos=%.1f ms rope=%.1f ms vit=%.1f ms postln=%.1f ms merge=%.1f ms mm=%.1f ms\n",
                t_patch, t_pos, t_rope, t_vit, t_postln, t_merge, t_mmproj);
    }
    fprintf(stderr, "  vision encoding done: %d tokens of dim %d (main %d + %d deepstack)\n",
            n_merged, total_embd, r->proj_dim, ds_count);

    return result;
}

/* ======================================================================== */
/* Public API: accessors                                                    */
/* ======================================================================== */

int cuda_vision_n_merged(const cuda_vision_runner *r) {
    return r ? r->n_merged : 0;
}

int cuda_vision_proj_dim(const cuda_vision_runner *r) {
    return r ? r->proj_dim : 0;
}

int cuda_vision_total_embd(const cuda_vision_runner *r) {
    return r ? r->proj_dim * (1 + r->n_deepstack) : 0;
}

/* ======================================================================== */
/* Public API: free                                                         */
/* ======================================================================== */

static void vlm_free_weight(gpu_weight *w) {
    if (w->w_f32) cuMemFree(w->w_f32);
    if (w->w_f16) cuMemFree(w->w_f16);
    if (w->bias) cuMemFree(w->bias);
    if (w->w_fp8) cuMemFree(w->w_fp8);
    memset(w, 0, sizeof(*w));
}

void cuda_vision_free(cuda_vision_runner *r) {
    if (!r) return;

    /* Tear down any captured ViT-block graph before freeing its device buffers */
    vlm_graph_reset(r);

    /* Free weights */
    if (r->d_patch_w0) cuMemFree(r->d_patch_w0);
    if (r->d_patch_w1) cuMemFree(r->d_patch_w1);
    if (r->d_patch_bias) cuMemFree(r->d_patch_bias);
    if (r->d_pos_embd) cuMemFree(r->d_pos_embd);

    if (r->blocks) {
        for (int l = 0; l < r->n_blocks; l++) {
            gpu_vit_block *blk = &r->blocks[l];
            vlm_free_weight(&blk->attn_qkv);
            vlm_free_weight(&blk->attn_out);
            vlm_free_weight(&blk->ffn_up);
            vlm_free_weight(&blk->ffn_down);
            if (blk->ln1_w) cuMemFree(blk->ln1_w);
            if (blk->ln1_b) cuMemFree(blk->ln1_b);
            if (blk->ln2_w) cuMemFree(blk->ln2_w);
            if (blk->ln2_b) cuMemFree(blk->ln2_b);
        }
        free(r->blocks);
    }

    if (r->deepstack) {
        for (int i = 0; i < r->n_deepstack; i++) {
            vlm_free_weight(&r->deepstack[i].fc1);
            vlm_free_weight(&r->deepstack[i].fc2);
            if (r->deepstack[i].norm_w) cuMemFree(r->deepstack[i].norm_w);
            if (r->deepstack[i].norm_b) cuMemFree(r->deepstack[i].norm_b);
        }
        free(r->deepstack);
    }
    free(r->deepstack_indices);

    if (r->d_post_ln_w) cuMemFree(r->d_post_ln_w);
    if (r->d_post_ln_b) cuMemFree(r->d_post_ln_b);
    vlm_free_weight(&r->mm0);
    vlm_free_weight(&r->mm2);

    /* Free scratch buffers */
    if (r->d_hidden) cuMemFree(r->d_hidden);
    if (r->d_hidden2) cuMemFree(r->d_hidden2);
    if (r->d_qkv) cuMemFree(r->d_qkv);
    if (r->d_attn_out) cuMemFree(r->d_attn_out);
    if (r->d_ffn_buf) cuMemFree(r->d_ffn_buf);
    if (r->d_ffn_buf_f16) cuMemFree(r->d_ffn_buf_f16);
    if (r->d_ffn_row_max) cuMemFree(r->d_ffn_row_max);
    if (r->d_ln_buf) cuMemFree(r->d_ln_buf);
    if (r->d_ln_buf_f16) cuMemFree(r->d_ln_buf_f16);
    if (r->d_merge_buf) cuMemFree(r->d_merge_buf);
    if (r->d_mm_buf) cuMemFree(r->d_mm_buf);
    if (r->d_mm_out) cuMemFree(r->d_mm_out);
    if (r->d_rgb) cuMemFree(r->d_rgb);
    if (r->d_rope_cos) cuMemFree(r->d_rope_cos);
    if (r->d_rope_sin) cuMemFree(r->d_rope_sin);
    if (r->d_pos_map) cuMemFree(r->d_pos_map);
    if (r->d_token_perm) cuMemFree(r->d_token_perm);
    if (r->d_token_inv_perm) cuMemFree(r->d_token_inv_perm);
    if (r->d_window_starts) cuMemFree(r->d_window_starts);
    if (r->d_window_sizes) cuMemFree(r->d_window_sizes);
    if (r->d_pos_interp) cuMemFree(r->d_pos_interp);
    if (r->d_ds_feats) cuMemFree(r->d_ds_feats);
    if (r->d_x_f16) cuMemFree(r->d_x_f16);
    if (r->d_qh_f16) cuMemFree(r->d_qh_f16);
    if (r->d_kh_f16) cuMemFree(r->d_kh_f16);
    if (r->d_vh_f16) cuMemFree(r->d_vh_f16);
    if (r->d_attn_scores) cuMemFree(r->d_attn_scores);
    if (r->d_attn_probs) cuMemFree(r->d_attn_probs);
    if (r->d_patch_pix) cuMemFree(r->d_patch_pix);

    free(r->h_pos_embd);
    free(r->h_output);
    if (r->cublas) cublasewDestroy(r->cublas);

    /* Destroy CUDA objects */
    if (r->module) cuModuleUnload(r->module);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->context) cuCtxDestroy(r->context);

    free(r);
}
