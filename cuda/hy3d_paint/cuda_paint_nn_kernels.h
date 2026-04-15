/*
 * cuda_paint_nn_kernels.h - Extra NVRTC kernels for the Hunyuan3D-2.1
 * paint neural pipeline. Consumed alongside cuda/cuda_kernels_common.h
 * and cuda/hy3d/cuda_hy3d_kernels.h.
 *
 * First addition: split_silu_gate_f32, the SwiGLU FFN "gate" step used
 * by DINOv2-giant (and generally by SwiGLU-family feed-forwards).
 *
 * SwiGLU layout in transformers:
 *     h_in = weights_in(x)                 # [N, 2*H]
 *     x1, x2 = h_in.chunk(2, dim=-1)       # first half / second half
 *     h_mid = silu(x1) * x2                # [N, H]
 *     out   = weights_out(h_mid)           # [N, C]
 *
 * The two GEMMs (weights_in, weights_out) are already handled by the
 * shared gemm_f16_f32 / gemm_f32_f32 kernels. This header adds the
 * fused chunk-split + silu * gate step that sits between them.
 */
#ifndef CUDA_PAINT_NN_KERNELS_H
#define CUDA_PAINT_NN_KERNELS_H

static const char cuda_paint_nn_kernels_src[] =
"extern \"C\" {\n"
"\n"
"/* split_silu_gate_f32: SwiGLU gating.\n"
" *   in  [rows, 2*half_dim]  output of weights_in GEMM\n"
" *   out [rows,   half_dim]  silu(in[:, :half_dim]) * in[:, half_dim:]\n"
" * One thread per (row, col) pair. */\n"
"__global__ void split_silu_gate_f32(const float *in, float *out,\n"
"                                     int rows, int half_dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = rows * half_dim;\n"
"    if (idx >= total) return;\n"
"    int r = idx / half_dim;\n"
"    int c = idx % half_dim;\n"
"    size_t base = (size_t)r * 2 * half_dim;\n"
"    float x1 = in[base + c];\n"
"    float x2 = in[base + half_dim + c];\n"
"    /* silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) */\n"
"    float s = x1 / (1.0f + expf(-x1));\n"
"    out[idx] = s * x2;\n"
"}\n"
"\n"
"} /* extern C */\n"
;

#endif /* CUDA_PAINT_NN_KERNELS_H */
