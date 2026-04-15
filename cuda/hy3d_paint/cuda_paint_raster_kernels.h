/*
 * cuda_paint_raster_kernels.h - Native CUDA triangle rasterization for
 * Hunyuan3D-2.1 texture generation. Replacement for the upstream
 * custom_rasterizer PyTorch extension (torch::Tensor-free).
 *
 * Kernels (NVRTC-compiled source string):
 *
 *   rasterize_faces_f32(V, F, zbuffer, num_faces, W, H)
 *     One thread per face; for each pixel in the triangle's clip-space bbox,
 *     packs (depth_quantized, face_id+1) into an int64 and atomicMin's it
 *     into zbuffer. Uses the same encoding as upstream:
 *
 *       depth_q = int(z * (2<<17))
 *       token   = (int64)depth_q * INT32_MAX + (int64)(face_id + 1)
 *
 *     V is [num_vertices, 4] homogeneous clip-space. F is [num_faces, 3] i32.
 *
 *   resolve_bary_f32(V, F, zbuffer, findices, bary, num_faces, W, H)
 *     One thread per pixel; extracts face_id from zbuffer (mod INT32_MAX),
 *     recomputes 2D barycentric at the pixel center, then perspective-
 *     corrects by dividing each coord by the corresponding vertex w and
 *     renormalizing. Writes:
 *       findices[H,W]        int32 (1-indexed; 0 = empty)
 *       bary    [H,W,3]      float32 perspective-corrected barycentric
 *
 * SPDX-License-Identifier: MIT (wrapper); rasterization core derived from
 * Tencent Hunyuan3D-2.1 custom_rasterizer (non-commercial use only).
 */
#ifndef CUDA_PAINT_RASTER_KERNELS_H
#define CUDA_PAINT_RASTER_KERNELS_H

static const char cuda_paint_raster_kernels_src[] =
"#define MAXINT32 2147483647\n"
"\n"
"extern \"C\" {\n"
"\n"
"__device__ inline float signed_area2(const float *a, const float *b, const float *c) {\n"
"    return ((c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1]));\n"
"}\n"
"\n"
"__device__ inline void barycentric2d(const float *a, const float *b, const float *c,\n"
"                                      const float *p, float *out) {\n"
"    float area  = signed_area2(a, b, c);\n"
"    if (area == 0.f) { out[0]=-1.f; out[1]=-1.f; out[2]=-1.f; return; }\n"
"    float beta  = signed_area2(a, p, c) / area;\n"
"    float gamma = signed_area2(a, b, p) / area;\n"
"    out[0] = 1.0f - beta - gamma;\n"
"    out[1] = beta;\n"
"    out[2] = gamma;\n"
"}\n"
"\n"
"__device__ inline bool bary_in_bounds(const float *b) {\n"
"    return b[0] >= 0.f && b[0] <= 1.f &&\n"
"           b[1] >= 0.f && b[1] <= 1.f &&\n"
"           b[2] >= 0.f && b[2] <= 1.f;\n"
"}\n"
"\n"
"/* Main rasterization. One thread per face. Writes packed\n"
" * (depth_quant, face_id+1) tokens into int64 zbuffer via atomicMin. */\n"
"__global__ void rasterize_faces_f32(const float *V, const int *F,\n"
"                                      unsigned long long *zbuffer,\n"
"                                      int num_faces, int width, int height) {\n"
"    int f = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (f >= num_faces) return;\n"
"\n"
"    const float *vt0p = V + (size_t)F[f*3+0] * 4;\n"
"    const float *vt1p = V + (size_t)F[f*3+1] * 4;\n"
"    const float *vt2p = V + (size_t)F[f*3+2] * 4;\n"
"\n"
"    float vt0[3] = { (vt0p[0] / vt0p[3] * 0.5f + 0.5f) * (width - 1) + 0.5f,\n"
"                     (0.5f + 0.5f * vt0p[1] / vt0p[3]) * (height - 1) + 0.5f,\n"
"                     vt0p[2] / vt0p[3] * 0.49999f + 0.5f };\n"
"    float vt1[3] = { (vt1p[0] / vt1p[3] * 0.5f + 0.5f) * (width - 1) + 0.5f,\n"
"                     (0.5f + 0.5f * vt1p[1] / vt1p[3]) * (height - 1) + 0.5f,\n"
"                     vt1p[2] / vt1p[3] * 0.49999f + 0.5f };\n"
"    float vt2[3] = { (vt2p[0] / vt2p[3] * 0.5f + 0.5f) * (width - 1) + 0.5f,\n"
"                     (0.5f + 0.5f * vt2p[1] / vt2p[3]) * (height - 1) + 0.5f,\n"
"                     vt2p[2] / vt2p[3] * 0.49999f + 0.5f };\n"
"\n"
"    float xmin = fminf(vt0[0], fminf(vt1[0], vt2[0]));\n"
"    float xmax = fmaxf(vt0[0], fmaxf(vt1[0], vt2[0]));\n"
"    float ymin = fminf(vt0[1], fminf(vt1[1], vt2[1]));\n"
"    float ymax = fmaxf(vt0[1], fmaxf(vt1[1], vt2[1]));\n"
"\n"
"    int ix0 = (int)fmaxf(0.f, floorf(xmin));\n"
"    int ix1 = (int)fminf((float)(width-1), floorf(xmax + 1.f));\n"
"    int iy0 = (int)fmaxf(0.f, floorf(ymin));\n"
"    int iy1 = (int)fminf((float)(height-1), floorf(ymax + 1.f));\n"
"\n"
"    for (int py = iy0; py <= iy1; py++) {\n"
"        for (int px = ix0; px <= ix1; px++) {\n"
"            float p[2] = { (float)px + 0.5f, (float)py + 0.5f };\n"
"            float bary[3];\n"
"            barycentric2d(vt0, vt1, vt2, p, bary);\n"
"            if (!bary_in_bounds(bary)) continue;\n"
"\n"
"            float depth = bary[0]*vt0[2] + bary[1]*vt1[2] + bary[2]*vt2[2];\n"
"            /* Reject depths behind the near plane (matches upstream that\n"
"             * truncates at depth < 0). */\n"
"            if (depth < 0.f) continue;\n"
"            int depth_q = (int)(depth * (float)(2 << 17));\n"
"            unsigned long long token =\n"
"                (unsigned long long)depth_q * (unsigned long long)MAXINT32\n"
"                + (unsigned long long)(f + 1);\n"
"            atomicMin(&zbuffer[py * width + px], token);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* Resolve per-pixel barycentric from the packed zbuffer. Writes\n"
" *   findices [H*W]   int32  (1-indexed; 0 means empty)\n"
" *   bary     [H*W*3] float  (perspective-corrected) */\n"
"__global__ void resolve_bary_f32(const float *V, const int *F,\n"
"                                   const unsigned long long *zbuffer,\n"
"                                   int *findices, float *bary_out,\n"
"                                   int width, int height) {\n"
"    int pix = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = width * height;\n"
"    if (pix >= total) return;\n"
"\n"
"    unsigned long long zv = zbuffer[pix];\n"
"    long long f = (long long)(zv % (unsigned long long)MAXINT32);\n"
"    if (f == (long long)(MAXINT32 - 1) || f == 0) {\n"
"        findices[pix] = 0;\n"
"        bary_out[pix*3+0] = 0.f;\n"
"        bary_out[pix*3+1] = 0.f;\n"
"        bary_out[pix*3+2] = 0.f;\n"
"        return;\n"
"    }\n"
"    findices[pix] = (int)f;\n"
"    f -= 1;\n"
"\n"
"    const float *vt0p = V + (size_t)F[f*3+0] * 4;\n"
"    const float *vt1p = V + (size_t)F[f*3+1] * 4;\n"
"    const float *vt2p = V + (size_t)F[f*3+2] * 4;\n"
"\n"
"    float vt0[2] = { (vt0p[0] / vt0p[3] * 0.5f + 0.5f) * (width - 1) + 0.5f,\n"
"                     (0.5f + 0.5f * vt0p[1] / vt0p[3]) * (height - 1) + 0.5f };\n"
"    float vt1[2] = { (vt1p[0] / vt1p[3] * 0.5f + 0.5f) * (width - 1) + 0.5f,\n"
"                     (0.5f + 0.5f * vt1p[1] / vt1p[3]) * (height - 1) + 0.5f };\n"
"    float vt2[2] = { (vt2p[0] / vt2p[3] * 0.5f + 0.5f) * (width - 1) + 0.5f,\n"
"                     (0.5f + 0.5f * vt2p[1] / vt2p[3]) * (height - 1) + 0.5f };\n"
"    float p[2] = { (float)(pix % width) + 0.5f, (float)(pix / width) + 0.5f };\n"
"    float bary[3];\n"
"    barycentric2d(vt0, vt1, vt2, p, bary);\n"
"    /* Perspective-correct by dividing each barycentric coord by the\n"
"     * corresponding vertex w and renormalising so the three sum to 1. */\n"
"    bary[0] /= vt0p[3];\n"
"    bary[1] /= vt1p[3];\n"
"    bary[2] /= vt2p[3];\n"
"    float w = 1.f / (bary[0] + bary[1] + bary[2]);\n"
"    bary_out[pix*3+0] = bary[0] * w;\n"
"    bary_out[pix*3+1] = bary[1] * w;\n"
"    bary_out[pix*3+2] = bary[2] * w;\n"
"}\n"
"\n"
"} /* extern C */\n"
;

#endif /* CUDA_PAINT_RASTER_KERNELS_H */
