/*
 * hip_sam3_kernels.h - HIP kernel sources for SAM 3 (RDNA4).
 *
 * Concatenated after hip_kernels_common_src. We close the extern "C" {
 * brace opened by the common preamble at the end of our string.
 *
 * Phase 1 kernels:
 *   patch_embed_sam3        Conv2d(3, 1024, k=14, s=14, bias=False)
 *                           Output layout: flat (grid*grid, D).
 *   pos_embed_tile_add      x[py,px,:] += pos[py%P, px%P, :] (P=24).
 *
 * Follow-up phases add: 2D axial RoPE, windowed/global ViT MHA,
 * FPN convs, CLIP text MHA, DETR enc/dec, pixel decoder, post-process.
 */
#ifndef HIP_SAM3_KERNELS_H_
#define HIP_SAM3_KERNELS_H_

static const char hip_sam3_kernels_src[] =
"\n"
"/* ===================== SAM 3 kernels ===================== */\n"
"\n"
"/* ---- patch_embed_sam3 ----\n"
" * Conv2d(3, D=1024, kernel_size=14, stride=14, bias=False), no padding.\n"
" * Output layout: flat (grid*grid, D) row-major (matches HF / CPU runner).\n"
" *\n"
" * Grid: (grid, grid), blockDim: 128 threads — each block computes one\n"
" * patch; threads stride across output channels.\n"
" */\n"
"__global__ void patch_embed_sam3(float *out,           /* [grid*grid, D] */\n"
"                                  const half_raw *w,    /* [D, 3, 14, 14] F16 */\n"
"                                  const float *img,     /* [3, img_size, img_size] CHW F32 */\n"
"                                  int grid, int img_size, int D) {\n"
"    int py = blockIdx.y;\n"
"    int px = blockIdx.x;\n"
"    if (py >= grid || px >= grid) return;\n"
"    const int K = 14;\n"
"    int iy0 = py * K;\n"
"    int ix0 = px * K;\n"
"    int HW = img_size * img_size;\n"
"    for (int co = threadIdx.x; co < D; co += blockDim.x) {\n"
"        float acc = 0.f;\n"
"        for (int ci = 0; ci < 3; ci++) {\n"
"            const float *ch = img + (size_t)ci * HW;\n"
"            for (int ky = 0; ky < K; ky++) {\n"
"                for (int kx = 0; kx < K; kx++) {\n"
"                    int widx = ((co * 3 + ci) * K + ky) * K + kx;\n"
"                    acc += half_to_float(w[widx])\n"
"                         * ch[(iy0 + ky) * img_size + (ix0 + kx)];\n"
"                }\n"
"            }\n"
"        }\n"
"        out[(py * grid + px) * D + co] = acc;\n"
"    }\n"
"}\n"
"\n"
"/* ---- pos_embed_tile_add ----\n"
" * In-place tokens += tile(pos[P, P, D], grid/P, grid/P) along the\n"
" * spatial axes. For SAM 3: P=24, grid=72, so pos is repeated 3x3.\n"
" */\n"
"__global__ void pos_embed_tile_add(float *tokens,         /* [grid*grid, D] */\n"
"                                    const float *pos,      /* [P, P, D] */\n"
"                                    int grid, int P, int D) {\n"
"    int py = blockIdx.y;\n"
"    int px = blockIdx.x;\n"
"    if (py >= grid || px >= grid) return;\n"
"    int sy = py % P, sx = px % P;\n"
"    float       *t = tokens + ((size_t)py * grid + px) * D;\n"
"    const float *s = pos    + ((size_t)sy * P   + sx) * D;\n"
"    for (int d = threadIdx.x; d < D; d += blockDim.x) t[d] += s[d];\n"
"}\n"
"\n"
"} /* extern \"C\" */\n"
;

#endif /* HIP_SAM3_KERNELS_H_ */
