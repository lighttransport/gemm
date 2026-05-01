/*
 * cuda_gemm_ptx_kernels.h
 *
 * NVRTC kernel sources for the cuda/gemm benchmark.
 *
 * Layout for all kernels:
 *   Y[M, N] = X[M, K] * W[N, K]^T   (row-major; W is "transposed" weight)
 *   K must be a multiple of 16 (f16/bf16) or 32 (fp8 e4m3).
 *   Y is float32. X and W are in the native dtype.
 *
 * One CTA = 128 threads (4 warps). Each warp computes a 16xNTILE*8 output
 * tile via repeated mma.sync. Grid = (ceil(N/256), ceil(M/16)).
 *
 * Targeting sm_120 only: a1/a2 fragment layout follows the standard PTX
 * mapping (no pre-Hopper swap branch needed).
 */
#ifndef CUDA_GEMM_PTX_KERNELS_H
#define CUDA_GEMM_PTX_KERNELS_H

/* ---------- f16 / f16 -> f32 (mma.sync.aligned.m16n8k16) ---------- */
static const char k_gemm_f16_src[] =
"typedef unsigned short half_raw;\n"
"\n"
"extern \"C\" __global__ void gemm_f16(float *Y,\n"
"                                      const half_raw *X,\n"
"                                      const half_raw *W,\n"
"                                      int M, int N, int K) {\n"
"    extern __shared__ half_raw smem_x_f16[]; /* 16 x 16 = 256 halves */\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int warp_id  = threadIdx.x >> 5;\n"
"    int out_base = blockIdx.x * 256 + warp_id * 64;\n"
"    int lane     = threadIdx.x & 31;\n"
"    int gid      = lane >> 2;\n"
"    int tid4     = lane & 3;\n"
"    int tid      = threadIdx.x;\n"
"\n"
"    if (tok_base >= M) return;\n"
"\n"
"    float d0[8], d1[8], d2[8], d3[8];\n"
"#pragma unroll\n"
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"\n"
"    for (int k = 0; k < K; k += 16) {\n"
"        /* Cooperative load 16x16 X tile to smem (128 threads x 2 halves each) */\n"
"        int srow = tid >> 3;          /* 0..15 */\n"
"        int scol = (tid & 7) * 2;     /* 0,2,4,...,14 */\n"
"        int grow = tok_base + srow;\n"
"        if (grow < M) {\n"
"            *(unsigned int *)&smem_x_f16[srow * 16 + scol] =\n"
"                *(const unsigned int *)&X[(size_t)grow * K + k + scol];\n"
"        } else {\n"
"            *(unsigned int *)&smem_x_f16[srow * 16 + scol] = 0;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* Load A frags (mma.sync.aligned.m16n8k16.row.col fragment layout) */\n"
"        unsigned int a0, a1, a2, a3;\n"
"        a0 = *(unsigned int *)&smem_x_f16[gid * 16 + tid4 * 2];\n"
"        a1 = *(unsigned int *)&smem_x_f16[(gid + 8) * 16 + tid4 * 2];\n"
"        a2 = *(unsigned int *)&smem_x_f16[gid * 16 + tid4 * 2 + 8];\n"
"        a3 = *(unsigned int *)&smem_x_f16[(gid + 8) * 16 + tid4 * 2 + 8];\n"
"\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < 8; nt++) {\n"
"            int bc = out_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < N) {\n"
"                const half_raw *wp = W + (size_t)bc * K + k;\n"
"                b0 = *(const unsigned int *)(wp + tid4 * 2);\n"
"                b1 = *(const unsigned int *)(wp + tid4 * 2 + 8);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
"                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    int yr0 = tok_base + gid;\n"
"    int yr1 = tok_base + gid + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < 8; nt++) {\n"
"        int yc0 = out_base + nt * 8 + tid4 * 2;\n"
"        int yc1 = yc0 + 1;\n"
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"
"    }\n"
"}\n";

/* ---------- bf16 / bf16 -> f32 (mma.sync.aligned.m16n8k16) ---------- */
static const char k_gemm_bf16_src[] =
"typedef unsigned short bf16_raw;\n"
"\n"
"extern \"C\" __global__ void gemm_bf16(float *Y,\n"
"                                       const bf16_raw *X,\n"
"                                       const bf16_raw *W,\n"
"                                       int M, int N, int K) {\n"
"    extern __shared__ bf16_raw smem_x_bf16[]; /* 16 x 16 */\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int warp_id  = threadIdx.x >> 5;\n"
"    int out_base = blockIdx.x * 256 + warp_id * 64;\n"
"    int lane     = threadIdx.x & 31;\n"
"    int gid      = lane >> 2;\n"
"    int tid4     = lane & 3;\n"
"    int tid      = threadIdx.x;\n"
"\n"
"    if (tok_base >= M) return;\n"
"\n"
"    float d0[8], d1[8], d2[8], d3[8];\n"
"#pragma unroll\n"
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"\n"
"    for (int k = 0; k < K; k += 16) {\n"
"        int srow = tid >> 3;\n"
"        int scol = (tid & 7) * 2;\n"
"        int grow = tok_base + srow;\n"
"        if (grow < M) {\n"
"            *(unsigned int *)&smem_x_bf16[srow * 16 + scol] =\n"
"                *(const unsigned int *)&X[(size_t)grow * K + k + scol];\n"
"        } else {\n"
"            *(unsigned int *)&smem_x_bf16[srow * 16 + scol] = 0;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        unsigned int a0, a1, a2, a3;\n"
"        a0 = *(unsigned int *)&smem_x_bf16[gid * 16 + tid4 * 2];\n"
"        a1 = *(unsigned int *)&smem_x_bf16[(gid + 8) * 16 + tid4 * 2];\n"
"        a2 = *(unsigned int *)&smem_x_bf16[gid * 16 + tid4 * 2 + 8];\n"
"        a3 = *(unsigned int *)&smem_x_bf16[(gid + 8) * 16 + tid4 * 2 + 8];\n"
"\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < 8; nt++) {\n"
"            int bc = out_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < N) {\n"
"                const bf16_raw *wp = W + (size_t)bc * K + k;\n"
"                b0 = *(const unsigned int *)(wp + tid4 * 2);\n"
"                b1 = *(const unsigned int *)(wp + tid4 * 2 + 8);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
"                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    int yr0 = tok_base + gid;\n"
"    int yr1 = tok_base + gid + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < 8; nt++) {\n"
"        int yc0 = out_base + nt * 8 + tid4 * 2;\n"
"        int yc1 = yc0 + 1;\n"
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"
"    }\n"
"}\n";

/* ---------- fp8 e4m3 / e4m3 -> f32 (mma.sync.aligned.m16n8k32) ---------- */
/* m16n8k32 fragments (sm_120 standard layout):                                 */
/*   A 16x32: a0=A[g, c..c+3]  a1=A[g+8, c..c+3]                                */
/*            a2=A[g, c+16..c+19]  a3=A[g+8, c+16..c+19]   (g=lane/4, c=lane%4*4)*/
/*   B 32x8:  b0=B[c..c+3, g]  b1=B[c+16..c+19, g]                              */
/*   C 16x8:  d0..d3 same as f16 case                                           */
/* B = W^T means b is row-major over W: b0 = W[g, c..c+3], b1 = W[g, c+16..c+19] */
static const char k_gemm_fp8_src[] =
"typedef unsigned char fp8_raw;\n"
"\n"
"extern \"C\" __global__ void gemm_fp8(float *Y,\n"
"                                      const fp8_raw *X,\n"
"                                      const fp8_raw *W,\n"
"                                      int M, int N, int K) {\n"
"    extern __shared__ fp8_raw smem_x_fp8[]; /* 16 x 32 = 512 bytes */\n"
"    int tok_base = blockIdx.y * 16;\n"
"    int warp_id  = threadIdx.x >> 5;\n"
"    int out_base = blockIdx.x * 256 + warp_id * 64;\n"
"    int lane     = threadIdx.x & 31;\n"
"    int gid      = lane >> 2;\n"
"    int tid4     = lane & 3;\n"
"    int tid      = threadIdx.x;\n"
"\n"
"    if (tok_base >= M) return;\n"
"\n"
"    float d0[8], d1[8], d2[8], d3[8];\n"
"#pragma unroll\n"
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"\n"
"    for (int k = 0; k < K; k += 32) {\n"
"        /* Cooperative load 16x32 = 512 bytes; 128 threads x 4 bytes each */\n"
"        int srow = tid >> 3;          /* 0..15 */\n"
"        int scol = (tid & 7) * 4;     /* 0,4,...,28 */\n"
"        int grow = tok_base + srow;\n"
"        if (grow < M) {\n"
"            *(unsigned int *)&smem_x_fp8[srow * 32 + scol] =\n"
"                *(const unsigned int *)&X[(size_t)grow * K + k + scol];\n"
"        } else {\n"
"            *(unsigned int *)&smem_x_fp8[srow * 32 + scol] = 0;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        unsigned int a0, a1, a2, a3;\n"
"        a0 = *(unsigned int *)&smem_x_fp8[gid       * 32 + tid4 * 4];\n"
"        a1 = *(unsigned int *)&smem_x_fp8[(gid + 8) * 32 + tid4 * 4];\n"
"        a2 = *(unsigned int *)&smem_x_fp8[gid       * 32 + tid4 * 4 + 16];\n"
"        a3 = *(unsigned int *)&smem_x_fp8[(gid + 8) * 32 + tid4 * 4 + 16];\n"
"\n"
"#pragma unroll\n"
"        for (int nt = 0; nt < 8; nt++) {\n"
"            int bc = out_base + nt * 8 + gid;\n"
"            unsigned int b0 = 0, b1 = 0;\n"
"            if (bc < N) {\n"
"                const fp8_raw *wp = W + (size_t)bc * K + k;\n"
"                b0 = *(const unsigned int *)(wp + tid4 * 4);\n"
"                b1 = *(const unsigned int *)(wp + tid4 * 4 + 16);\n"
"            }\n"
"            asm volatile(\n"
"                \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"
"                : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                  \"r\"(b0), \"r\"(b1),\n"
"                  \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"
"            );\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    int yr0 = tok_base + gid;\n"
"    int yr1 = tok_base + gid + 8;\n"
"#pragma unroll\n"
"    for (int nt = 0; nt < 8; nt++) {\n"
"        int yc0 = out_base + nt * 8 + tid4 * 2;\n"
"        int yc1 = yc0 + 1;\n"
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"
"    }\n"
"}\n";

/* ==================================================================== *
 * v2 kernels — cp.async-staged operands, K-step 32, larger M tile
 * --------------------------------------------------------------------
 * Same row-major Y[M,N] = X[M,K] * W[N,K]^T contract as v1.
 *
 * CTA: 8 warps / 256 threads. Tile per CTA: 64 (M) × 128 (N), K-step 32.
 * Warp grid 4×2: warp_m = wid/2 (0..3), warp_n = wid%2.
 * Each warp owns 16 (M) × 64 (N) -> NTILE=8 mma.sync.m16n8k16 along N
 * × 2 along K = 16 mma.sync per K-iter.
 *
 * SMEM (single-buffer for v2.0):
 *   sA: 64 × 32 halves = 4 KiB
 *   sB: 128 × 32 halves = 8 KiB
 * Total 12 KiB/CTA — fits default 48 KiB dynamic-smem limit without opt-in.
 *
 * Loads: cp.async.cg 16-byte vectors. ldmatrix is *not* used yet (left for
 * a phase-3 v3 if v2 falls short of 98% cutile).
 *
 * Grid: (ceil(N/128), ceil(M/64), 1). Block: 256.
 * SMEM bytes: 12288 (12 KiB).
 * ==================================================================== */

#define CUDA_GEMM_V2_BODY(MMA_OP, ELEM_T)                                    \
"    extern __shared__ __align__(16) " ELEM_T " smem_v2[];\n"                \
"    " ELEM_T " *sA = smem_v2;          /* 64 x 32 */\n"                     \
"    " ELEM_T " *sB = smem_v2 + 64*32;  /* 128 x 32 */\n"                    \
"    int tid = threadIdx.x;\n"                                               \
"    int wid = tid >> 5;\n"                                                  \
"    int lane = tid & 31;\n"                                                 \
"    int gid  = lane >> 2;\n"                                                \
"    int tid4 = lane & 3;\n"                                                 \
"    int warp_m = wid >> 1;          /* 0..3 */\n"                           \
"    int warp_n = wid & 1;           /* 0..1 */\n"                           \
"    int cta_m  = blockIdx.y * 64;\n"                                        \
"    int cta_n  = blockIdx.x * 128;\n"                                       \
"    int wm_row = warp_m * 16;       /* row within sA, 0/16/32/48 */\n"      \
"    int wn_col = warp_n * 64;       /* col within sB, 0/64 */\n"            \
"\n"                                                                         \
"    if (cta_m >= M) return;\n"                                              \
"\n"                                                                         \
"    float d0[8], d1[8], d2[8], d3[8];\n"                                    \
"    #pragma unroll\n"                                                       \
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"  \
"\n"                                                                         \
"    /* Each thread loads N vecs of 16 bytes (8 halves) */\n"                \
"    /* sA: 64x32 = 2048 halves = 256 vecs ÷ 256 threads = 1 vec/thread */\n"\
"    /* sB: 128x32 = 4096 halves = 512 vecs ÷ 256 threads = 2 vecs/thread */\n"\
"\n"                                                                         \
"    for (int k = 0; k < K; k += 32) {\n"                                    \
"        /* Load A: each thread does 1 vec */\n"                             \
"        {\n"                                                                \
"            int vid = tid;            /* 0..255 */\n"                       \
"            int row = vid >> 2;       /* 0..63 */\n"                        \
"            int col = (vid & 3) * 8;  /* 0,8,16,24 */\n"                    \
"            int g_row = cta_m + row;\n"                                     \
"            unsigned int dst = __cvta_generic_to_shared(&sA[row*32 + col]);\n"\
"            if (g_row < M) {\n"                                             \
"                const " ELEM_T " *src = &X[(size_t)g_row * K + k + col];\n" \
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" \n"\
"                             :: \"r\"(dst), \"l\"(src));\n"                 \
"            } else {\n"                                                     \
"                asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dst));\n"\
"            }\n"                                                            \
"        }\n"                                                                \
"        /* Load B: each thread does 2 vecs */\n"                            \
"        #pragma unroll\n"                                                   \
"        for (int v = 0; v < 2; v++) {\n"                                    \
"            int vid = tid + v*256;     /* 0..511 */\n"                      \
"            int row = vid >> 2;        /* 0..127 */\n"                      \
"            int col = (vid & 3) * 8;\n"                                     \
"            int g_row = cta_n + row;\n"                                     \
"            unsigned int dst = __cvta_generic_to_shared(&sB[row*32 + col]);\n"\
"            if (g_row < N) {\n"                                             \
"                const " ELEM_T " *src = &W[(size_t)g_row * K + k + col];\n" \
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" \n"\
"                             :: \"r\"(dst), \"l\"(src));\n"                 \
"            } else {\n"                                                     \
"                asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dst));\n"\
"            }\n"                                                            \
"        }\n"                                                                \
"        asm volatile(\"cp.async.commit_group;\\n\");\n"                     \
"        asm volatile(\"cp.async.wait_group 0;\\n\");\n"                     \
"        __syncthreads();\n"                                                 \
"\n"                                                                         \
"        /* mma loop: 2 K-groups × 8 N-tiles per warp */\n"                  \
"        #pragma unroll\n"                                                   \
"        for (int kg = 0; kg < 2; kg++) {\n"                                 \
"            int k_off = kg * 16;\n"                                         \
"            int ar0 = wm_row + gid;\n"                                      \
"            int ar1 = wm_row + gid + 8;\n"                                  \
"            unsigned int a0 = *(unsigned int*)&sA[ar0*32 + k_off + tid4*2];\n"\
"            unsigned int a1 = *(unsigned int*)&sA[ar1*32 + k_off + tid4*2];\n"\
"            unsigned int a2 = *(unsigned int*)&sA[ar0*32 + k_off + tid4*2 + 8];\n"\
"            unsigned int a3 = *(unsigned int*)&sA[ar1*32 + k_off + tid4*2 + 8];\n"\
"            #pragma unroll\n"                                               \
"            for (int nt = 0; nt < 8; nt++) {\n"                             \
"                int bc = wn_col + nt*8 + gid;\n"                            \
"                unsigned int b0 = *(unsigned int*)&sB[bc*32 + k_off + tid4*2];\n"\
"                unsigned int b1 = *(unsigned int*)&sB[bc*32 + k_off + tid4*2 + 8];\n"\
"                asm volatile(\n"                                            \
"                    \"" MMA_OP "\\n\\t\"\n"                                 \
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"\
"                    : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"\
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"        \
"                      \"r\"(b0), \"r\"(b1),\n"                              \
"                      \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"\
"                );\n"                                                       \
"            }\n"                                                            \
"        }\n"                                                                \
"        __syncthreads();\n"                                                 \
"    }\n"                                                                    \
"\n"                                                                         \
"    int yr0 = cta_m + wm_row + gid;\n"                                      \
"    int yr1 = cta_m + wm_row + gid + 8;\n"                                  \
"    #pragma unroll\n"                                                       \
"    for (int nt = 0; nt < 8; nt++) {\n"                                     \
"        int yc0 = cta_n + wn_col + nt*8 + tid4*2;\n"                        \
"        int yc1 = yc0 + 1;\n"                                               \
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"       \
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"       \
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"       \
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"       \
"    }\n"                                                                    \
"}\n"

static const char k_gemm_f16_v2_src[] =
"typedef unsigned short half_raw;\n"
"extern \"C\" __global__ void gemm_f16_v2(float *Y,\n"
"                                          const half_raw *X,\n"
"                                          const half_raw *W,\n"
"                                          int M, int N, int K) {\n"
CUDA_GEMM_V2_BODY("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", "half_raw");

static const char k_gemm_bf16_v2_src[] =
"typedef unsigned short bf16_raw;\n"
"extern \"C\" __global__ void gemm_bf16_v2(float *Y,\n"
"                                           const bf16_raw *X,\n"
"                                           const bf16_raw *W,\n"
"                                           int M, int N, int K) {\n"
CUDA_GEMM_V2_BODY("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", "bf16_raw");

/* ==================================================================== *
 * v3 kernels — v2 + 2-stage cp.async software pipeline
 * --------------------------------------------------------------------
 * Same tile (64×128×32, 8 warps, 256 threads) and inner mma loop as v2,
 * but SMEM is double-buffered and the next K-tile's loads are issued
 * BEFORE computing the current tile, so the cp.async traffic overlaps
 * with mma compute. Per-CTA SMEM = 24 KiB (still under the 48 KiB
 * default cap; no MAX_DYN_SMEM opt-in required).
 *
 * Pattern (CUTLASS Mma pipelined, depth=2):
 *   prefetch(buf 0, k=0); commit;          // 1 group in flight
 *   for ki = 0 .. nk-1:
 *       wait_group 0; syncthreads;          // current buf ready
 *       if ki+1 < nk: prefetch(buf (ki+1)%2, k=(ki+1)*32); commit;
 *       compute on buf ki%2;                // overlaps with next prefetch
 *   wait_group 0;
 * ==================================================================== */

#define CUDA_GEMM_V3_BODY(MMA_OP, ELEM_T)                                    \
"    extern __shared__ __align__(16) " ELEM_T " smem_v3[];\n"                \
"    /* Stage layout: 24 KiB total = 2 × 12 KiB. Per stage:               */\n" \
"    /*   sA: 64 × 32 halves = 2048 halves                                */\n"\
"    /*   sB: 128 × 32 halves = 4096 halves                               */\n"\
"    " ELEM_T " *sA0 = smem_v3;\n"                                           \
"    " ELEM_T " *sB0 = smem_v3 + 2048;\n"                                    \
"    " ELEM_T " *sA1 = smem_v3 + 6144;\n"                                    \
"    " ELEM_T " *sB1 = smem_v3 + 8192;\n"                                    \
"    int tid = threadIdx.x;\n"                                               \
"    int wid = tid >> 5;\n"                                                  \
"    int lane = tid & 31;\n"                                                 \
"    int gid  = lane >> 2;\n"                                                \
"    int tid4 = lane & 3;\n"                                                 \
"    int warp_m = wid >> 1;\n"                                               \
"    int warp_n = wid & 1;\n"                                                \
"    int cta_m  = blockIdx.y * 64;\n"                                        \
"    int cta_n  = blockIdx.x * 128;\n"                                       \
"    int wm_row = warp_m * 16;\n"                                            \
"    int wn_col = warp_n * 64;\n"                                            \
"\n"                                                                         \
"    if (cta_m >= M) return;\n"                                              \
"\n"                                                                         \
"    float d0[8], d1[8], d2[8], d3[8];\n"                                    \
"    #pragma unroll\n"                                                       \
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"  \
"\n"                                                                         \
"    int vid_a   = tid;\n"                                                   \
"    int row_a   = vid_a >> 2;\n"                                            \
"    int col_a   = (vid_a & 3) * 8;\n"                                       \
"    int g_row_a = cta_m + row_a;\n"                                         \
"\n"                                                                         \
"    int vid_b0   = tid;\n"                                                  \
"    int row_b0   = vid_b0 >> 2;\n"                                          \
"    int col_b0   = (vid_b0 & 3) * 8;\n"                                     \
"    int g_row_b0 = cta_n + row_b0;\n"                                       \
"\n"                                                                         \
"    int vid_b1   = tid + 256;\n"                                            \
"    int row_b1   = vid_b1 >> 2;\n"                                          \
"    int col_b1   = (vid_b1 & 3) * 8;\n"                                     \
"    int g_row_b1 = cta_n + row_b1;\n"                                       \
"\n"                                                                         \
"    /* ---- Prologue: issue stage-0 loads, commit group 0 ---- */\n"        \
"    {\n"                                                                    \
"        unsigned int dA = __cvta_generic_to_shared(&sA0[row_a*32 + col_a]);\n"\
"        if (g_row_a < M) {\n"                                               \
"            const " ELEM_T " *src = &X[(size_t)g_row_a * K + 0 + col_a];\n" \
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"\
"        unsigned int dB0 = __cvta_generic_to_shared(&sB0[row_b0*32 + col_b0]);\n"\
"        if (g_row_b0 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b0 * K + 0 + col_b0];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"\
"        unsigned int dB1 = __cvta_generic_to_shared(&sB0[row_b1*32 + col_b1]);\n"\
"        if (g_row_b1 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b1 * K + 0 + col_b1];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"\
"    }\n"                                                                    \
"    asm volatile(\"cp.async.commit_group;\\n\");\n"                         \
"\n"                                                                         \
"    int num_k = K >> 5;   /* K/32 */\n"                                     \
"    for (int ki = 0; ki < num_k; ki++) {\n"                                 \
"        int stage = ki & 1;\n"                                              \
"        int next_stage = stage ^ 1;\n"                                      \
"        int next_k = (ki + 1) << 5;\n"                                      \
"\n"                                                                         \
"        /* Wait for current stage's loads to finish */\n"                   \
"        asm volatile(\"cp.async.wait_group 0;\\n\");\n"                     \
"        __syncthreads();\n"                                                 \
"\n"                                                                         \
"        /* Issue next stage's loads (overlap with compute below) */\n"      \
"        if (next_k < K) {\n"                                                \
"            " ELEM_T " *sA_next = (next_stage == 0) ? sA0 : sA1;\n"         \
"            " ELEM_T " *sB_next = (next_stage == 0) ? sB0 : sB1;\n"         \
"            unsigned int dA = __cvta_generic_to_shared(&sA_next[row_a*32 + col_a]);\n"\
"            if (g_row_a < M) {\n"                                           \
"                const " ELEM_T " *src = &X[(size_t)g_row_a * K + next_k + col_a];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"\
"            unsigned int dB0 = __cvta_generic_to_shared(&sB_next[row_b0*32 + col_b0]);\n"\
"            if (g_row_b0 < N) {\n"                                          \
"                const " ELEM_T " *src = &W[(size_t)g_row_b0 * K + next_k + col_b0];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"\
"            unsigned int dB1 = __cvta_generic_to_shared(&sB_next[row_b1*32 + col_b1]);\n"\
"            if (g_row_b1 < N) {\n"                                          \
"                const " ELEM_T " *src = &W[(size_t)g_row_b1 * K + next_k + col_b1];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"\
"        }\n"                                                                \
"        asm volatile(\"cp.async.commit_group;\\n\");\n"                     \
"\n"                                                                         \
"        /* Compute on current stage */\n"                                   \
"        " ELEM_T " *sAp = (stage == 0) ? sA0 : sA1;\n"                      \
"        " ELEM_T " *sBp = (stage == 0) ? sB0 : sB1;\n"                      \
"        #pragma unroll\n"                                                   \
"        for (int kg = 0; kg < 2; kg++) {\n"                                 \
"            int k_off = kg * 16;\n"                                         \
"            int ar0 = wm_row + gid;\n"                                      \
"            int ar1 = wm_row + gid + 8;\n"                                  \
"            unsigned int a0 = *(unsigned int*)&sAp[ar0*32 + k_off + tid4*2];\n"\
"            unsigned int a1 = *(unsigned int*)&sAp[ar1*32 + k_off + tid4*2];\n"\
"            unsigned int a2 = *(unsigned int*)&sAp[ar0*32 + k_off + tid4*2 + 8];\n"\
"            unsigned int a3 = *(unsigned int*)&sAp[ar1*32 + k_off + tid4*2 + 8];\n"\
"            #pragma unroll\n"                                               \
"            for (int nt = 0; nt < 8; nt++) {\n"                             \
"                int bc = wn_col + nt*8 + gid;\n"                            \
"                unsigned int b0 = *(unsigned int*)&sBp[bc*32 + k_off + tid4*2];\n"\
"                unsigned int b1 = *(unsigned int*)&sBp[bc*32 + k_off + tid4*2 + 8];\n"\
"                asm volatile(\n"                                            \
"                    \"" MMA_OP "\\n\\t\"\n"                                 \
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"\
"                    : \"=f\"(d0[nt]), \"=f\"(d1[nt]), \"=f\"(d2[nt]), \"=f\"(d3[nt])\n"\
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"        \
"                      \"r\"(b0), \"r\"(b1),\n"                              \
"                      \"f\"(d0[nt]), \"f\"(d1[nt]), \"f\"(d2[nt]), \"f\"(d3[nt])\n"\
"                );\n"                                                       \
"            }\n"                                                            \
"        }\n"                                                                \
"    }\n"                                                                    \
"    asm volatile(\"cp.async.wait_group 0;\\n\");\n"                         \
"\n"                                                                         \
"    int yr0 = cta_m + wm_row + gid;\n"                                      \
"    int yr1 = cta_m + wm_row + gid + 8;\n"                                  \
"    #pragma unroll\n"                                                       \
"    for (int nt = 0; nt < 8; nt++) {\n"                                     \
"        int yc0 = cta_n + wn_col + nt*8 + tid4*2;\n"                        \
"        int yc1 = yc0 + 1;\n"                                               \
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"       \
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"       \
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"       \
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"       \
"    }\n"                                                                    \
"}\n"

static const char k_gemm_f16_v3_src[] =
"typedef unsigned short half_raw;\n"
"extern \"C\" __global__ void gemm_f16_v3(float *Y,\n"
"                                          const half_raw *X,\n"
"                                          const half_raw *W,\n"
"                                          int M, int N, int K) {\n"
CUDA_GEMM_V3_BODY("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", "half_raw");

static const char k_gemm_bf16_v3_src[] =
"typedef unsigned short bf16_raw;\n"
"extern \"C\" __global__ void gemm_bf16_v3(float *Y,\n"
"                                           const bf16_raw *X,\n"
"                                           const bf16_raw *W,\n"
"                                           int M, int N, int K) {\n"
CUDA_GEMM_V3_BODY("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", "bf16_raw");

/* ==================================================================== *
 * v4 kernels — v3 + ldmatrix.x4 for A and B fragment loads
 * --------------------------------------------------------------------
 * Same CTA tile (64×128×32, 8 warps, 256 threads) and 2-stage cp.async
 * pipeline (24 KiB SMEM) as v3. The inner mma loop replaces 40 scalar
 * ld.shared.u32 per K-iter per warp with 10 ldmatrix.x4:
 *
 *   2 ldmatrix.x4 for A (one per K-group of 16, covers full 16×16 A-frag)
 *   8 ldmatrix.x4 for B (4 N-stripes × 2 K-groups; each ldmatrix produces
 *     4 regs that yield TWO mma N-tiles via {reg0, reg2} and {reg1, reg3})
 *
 * SMEM is NOT swizzled — accept ~2-way bank conflict on stride-32-half
 * layout in exchange for the 4× issue-throughput win. Adding the standard
 * (row & 3) << 3 XOR swizzle is a v5 follow-up.
 *
 * Layout invariants (same as v3):
 *   sA[stage]: 64 rows × 32 halves, sA[r][k] = X[cta_m+r, k_tile_base+k]
 *   sB[stage]: 128 rows × 32 halves, sB[n][k] = W[cta_n+n, k_tile_base+k]
 *
 * Per-lane ldmatrix row pointers:
 *   A: sAp[(wm_row + (lane & 15)) * 32 + k_off + ((lane >> 4) * 8)]
 *   B: sBp[(sn_base + (lane & 15)) * 32 + k_off + ((lane >> 4) * 8)]
 *
 * After ldmatrix.x4 on B with stripe base sn_base, lane (gid, tid4) gets:
 *   reg0 = B[K=2*tid4..+1,  N=sn_base+gid]
 *   reg1 = B[K=2*tid4..+1,  N=sn_base+gid+8]
 *   reg2 = B[K=2*tid4+8..+9, N=sn_base+gid]
 *   reg3 = B[K=2*tid4+8..+9, N=sn_base+gid+8]
 * → {reg0, reg2} = mma B-frag for N cols sn_base..+7   (nt = s*2)
 * → {reg1, reg3} = mma B-frag for N cols sn_base+8..+15 (nt = s*2+1)
 * ==================================================================== */

#define CUDA_GEMM_V4_BODY(MMA_OP, ELEM_T)                                    \
"    extern __shared__ __align__(16) " ELEM_T " smem_v4[];\n"                \
"    " ELEM_T " *sA0 = smem_v4;\n"                                           \
"    " ELEM_T " *sB0 = smem_v4 + 2048;\n"                                    \
"    " ELEM_T " *sA1 = smem_v4 + 6144;\n"                                    \
"    " ELEM_T " *sB1 = smem_v4 + 8192;\n"                                    \
"    int tid = threadIdx.x;\n"                                               \
"    int wid = tid >> 5;\n"                                                  \
"    int lane = tid & 31;\n"                                                 \
"    int gid  = lane >> 2;\n"                                                \
"    int tid4 = lane & 3;\n"                                                 \
"    int warp_m = wid >> 1;\n"                                               \
"    int warp_n = wid & 1;\n"                                                \
"    int cta_m  = blockIdx.y * 64;\n"                                        \
"    int cta_n  = blockIdx.x * 128;\n"                                       \
"    int wm_row = warp_m * 16;\n"                                            \
"    int wn_col = warp_n * 64;\n"                                            \
"\n"                                                                         \
"    if (cta_m >= M) return;\n"                                              \
"\n"                                                                         \
"    float d0[8], d1[8], d2[8], d3[8];\n"                                    \
"    #pragma unroll\n"                                                       \
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"  \
"\n"                                                                         \
"    int row_a   = tid >> 2;\n"                                              \
"    int col_a   = (tid & 3) * 8;\n"                                         \
"    int g_row_a = cta_m + row_a;\n"                                         \
"\n"                                                                         \
"    int row_b0   = tid >> 2;\n"                                             \
"    int col_b0   = (tid & 3) * 8;\n"                                        \
"    int g_row_b0 = cta_n + row_b0;\n"                                       \
"\n"                                                                         \
"    int vid_b1   = tid + 256;\n"                                            \
"    int row_b1   = vid_b1 >> 2;\n"                                          \
"    int col_b1   = (vid_b1 & 3) * 8;\n"                                     \
"    int g_row_b1 = cta_n + row_b1;\n"                                       \
"\n"                                                                         \
"    /* Prologue: stage 0 load, commit */\n"                                 \
"    {\n"                                                                    \
"        unsigned int dA = __cvta_generic_to_shared(&sA0[row_a*32 + col_a]);\n"\
"        if (g_row_a < M) {\n"                                               \
"            const " ELEM_T " *src = &X[(size_t)g_row_a * K + 0 + col_a];\n" \
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"\
"        unsigned int dB0 = __cvta_generic_to_shared(&sB0[row_b0*32 + col_b0]);\n"\
"        if (g_row_b0 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b0 * K + 0 + col_b0];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"\
"        unsigned int dB1 = __cvta_generic_to_shared(&sB0[row_b1*32 + col_b1]);\n"\
"        if (g_row_b1 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b1 * K + 0 + col_b1];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"\
"    }\n"                                                                    \
"    asm volatile(\"cp.async.commit_group;\\n\");\n"                         \
"\n"                                                                         \
"    int num_k = K >> 5;\n"                                                  \
"    int la_row = wm_row + (lane & 15);\n"                                   \
"    int la_col_off = (lane >> 4) * 8;\n"                                    \
"    int lb_row     = lane & 15;\n"                                          \
"    int lb_col_off = (lane >> 4) * 8;\n"                                    \
"\n"                                                                         \
"    for (int ki = 0; ki < num_k; ki++) {\n"                                 \
"        int stage = ki & 1;\n"                                              \
"        int next_stage = stage ^ 1;\n"                                      \
"        int next_k = (ki + 1) << 5;\n"                                      \
"\n"                                                                         \
"        asm volatile(\"cp.async.wait_group 0;\\n\");\n"                     \
"        __syncthreads();\n"                                                 \
"\n"                                                                         \
"        if (next_k < K) {\n"                                                \
"            " ELEM_T " *sA_next = (next_stage == 0) ? sA0 : sA1;\n"         \
"            " ELEM_T " *sB_next = (next_stage == 0) ? sB0 : sB1;\n"         \
"            unsigned int dA = __cvta_generic_to_shared(&sA_next[row_a*32 + col_a]);\n"\
"            if (g_row_a < M) {\n"                                           \
"                const " ELEM_T " *src = &X[(size_t)g_row_a * K + next_k + col_a];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"\
"            unsigned int dB0 = __cvta_generic_to_shared(&sB_next[row_b0*32 + col_b0]);\n"\
"            if (g_row_b0 < N) {\n"                                          \
"                const " ELEM_T " *src = &W[(size_t)g_row_b0 * K + next_k + col_b0];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"\
"            unsigned int dB1 = __cvta_generic_to_shared(&sB_next[row_b1*32 + col_b1]);\n"\
"            if (g_row_b1 < N) {\n"                                          \
"                const " ELEM_T " *src = &W[(size_t)g_row_b1 * K + next_k + col_b1];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"\
"        }\n"                                                                \
"        asm volatile(\"cp.async.commit_group;\\n\");\n"                     \
"\n"                                                                         \
"        " ELEM_T " *sAp = (stage == 0) ? sA0 : sA1;\n"                      \
"        " ELEM_T " *sBp = (stage == 0) ? sB0 : sB1;\n"                      \
"        #pragma unroll\n"                                                   \
"        for (int kg = 0; kg < 2; kg++) {\n"                                 \
"            int k_off = kg * 16;\n"                                         \
"            unsigned int a0,a1,a2,a3;\n"                                    \
"            unsigned int p_a = __cvta_generic_to_shared(&sAp[la_row*32 + k_off + la_col_off]);\n"\
"            asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"\
"                : \"=r\"(a0), \"=r\"(a1), \"=r\"(a2), \"=r\"(a3) : \"r\"(p_a));\n"\
"            #pragma unroll\n"                                               \
"            for (int s = 0; s < 4; s++) {\n"                                \
"                int sn_base = wn_col + s * 16;\n"                           \
"                unsigned int b0,b1,b2,b3;\n"                                \
"                unsigned int p_b = __cvta_generic_to_shared(&sBp[(sn_base + lb_row)*32 + k_off + lb_col_off]);\n"\
"                asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"\
"                    : \"=r\"(b0), \"=r\"(b1), \"=r\"(b2), \"=r\"(b3) : \"r\"(p_b));\n"\
"                int nt0 = s * 2 + 0;\n"                                     \
"                int nt1 = s * 2 + 1;\n"                                     \
"                asm volatile(\n"                                            \
"                    \"" MMA_OP "\\n\\t\"\n"                                 \
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"\
"                    : \"=f\"(d0[nt0]), \"=f\"(d1[nt0]), \"=f\"(d2[nt0]), \"=f\"(d3[nt0])\n"\
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"        \
"                      \"r\"(b0), \"r\"(b2),\n"                              \
"                      \"f\"(d0[nt0]), \"f\"(d1[nt0]), \"f\"(d2[nt0]), \"f\"(d3[nt0])\n"\
"                );\n"                                                       \
"                asm volatile(\n"                                            \
"                    \"" MMA_OP "\\n\\t\"\n"                                 \
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"\
"                    : \"=f\"(d0[nt1]), \"=f\"(d1[nt1]), \"=f\"(d2[nt1]), \"=f\"(d3[nt1])\n"\
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"        \
"                      \"r\"(b1), \"r\"(b3),\n"                              \
"                      \"f\"(d0[nt1]), \"f\"(d1[nt1]), \"f\"(d2[nt1]), \"f\"(d3[nt1])\n"\
"                );\n"                                                       \
"            }\n"                                                            \
"        }\n"                                                                \
"    }\n"                                                                    \
"    asm volatile(\"cp.async.wait_group 0;\\n\");\n"                         \
"\n"                                                                         \
"    int yr0 = cta_m + wm_row + gid;\n"                                      \
"    int yr1 = cta_m + wm_row + gid + 8;\n"                                  \
"    #pragma unroll\n"                                                       \
"    for (int nt = 0; nt < 8; nt++) {\n"                                     \
"        int yc0 = cta_n + wn_col + nt*8 + tid4*2;\n"                        \
"        int yc1 = yc0 + 1;\n"                                               \
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"       \
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"       \
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"       \
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"       \
"    }\n"                                                                    \
"}\n"

static const char k_gemm_f16_v4_src[] =
"typedef unsigned short half_raw;\n"
"extern \"C\" __global__ void gemm_f16_v4(float *Y,\n"
"                                          const half_raw *X,\n"
"                                          const half_raw *W,\n"
"                                          int M, int N, int K) {\n"
CUDA_GEMM_V4_BODY("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", "half_raw");

static const char k_gemm_bf16_v4_src[] =
"typedef unsigned short bf16_raw;\n"
"extern \"C\" __global__ void gemm_bf16_v4(float *Y,\n"
"                                           const bf16_raw *X,\n"
"                                           const bf16_raw *W,\n"
"                                           int M, int N, int K) {\n"
CUDA_GEMM_V4_BODY("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", "bf16_raw");

/* ==================================================================== *
 * v5 kernels — v4 + XOR-swizzled SMEM (RowMajor*Crosswise-style)
 * --------------------------------------------------------------------
 * Same CTA tile, 2-stage pipeline, ldmatrix.x4 strategy as v4. Only change:
 * SMEM is stored swizzled so that ldmatrix is conflict-free.
 *
 * Swizzle (in halves; row = M-row in sA or N-row in sB; stride = 32 halves):
 *   swz(row, col) = row * 32 + (col XOR ((row & 3) * 8))
 *
 * Why (row & 3) * 8:
 *   - 8 halves = 16 bytes = the cp.async chunk size and the ldmatrix row
 *     chunk size. XOR shift of 3 bits keeps each 8-half chunk contiguous
 *     (XOR by 0/8/16/24 in halves doesn't cross 8-half boundaries).
 *   - 4 distinct XOR values × 4 banks-per-byte-offset (4-byte banks across
 *     32 banks for 16-byte chunks) → 8 distinct bank quadrants for the 8
 *     rows in a tile group → no conflicts on ldmatrix.x4.
 *
 * Both `cp.async` store and `ldmatrix` load apply the same swizzle. col
 * passed through XOR is always a multiple of 8 (chunk-aligned) so the 8
 * halves of the chunk stay together.
 *
 * Per-thread store positions (cp.async, 8-halves vec):
 *   sA: row_a = tid>>2, col_a = (tid&3)*8.  swizzled col = col_a ^ ((row_a&3)*8)
 *   sB: row_b{0,1} = (tid+v*256)>>2, col_b{0,1} = ((tid+v*256)&3)*8.
 *
 * Per-lane ldmatrix row pointers:
 *   A-lane l: row = la_row = wm_row + (l&15), col = k_off + la_col_off
 *             swz_addr = la_row*32 + ((k_off+la_col_off) ^ ((la_row&3)*8))
 *   B-lane l: row = sn_base + (l&15), col = k_off + lb_col_off
 *
 * (wm_row, sn_base, k_off, la_col_off, lb_col_off all multiples of 8 →
 *  XOR with (row&3)*8 keeps result in [0..31].)
 * ==================================================================== */

#define CUDA_GEMM_V5_BODY(MMA_OP, ELEM_T)                                    \
"    extern __shared__ __align__(16) " ELEM_T " smem_v5[];\n"                \
"    " ELEM_T " *sA0 = smem_v5;\n"                                           \
"    " ELEM_T " *sB0 = smem_v5 + 2048;\n"                                    \
"    " ELEM_T " *sA1 = smem_v5 + 6144;\n"                                    \
"    " ELEM_T " *sB1 = smem_v5 + 8192;\n"                                    \
"    int tid = threadIdx.x;\n"                                               \
"    int wid = tid >> 5;\n"                                                  \
"    int lane = tid & 31;\n"                                                 \
"    int gid  = lane >> 2;\n"                                                \
"    int tid4 = lane & 3;\n"                                                 \
"    int warp_m = wid >> 1;\n"                                               \
"    int warp_n = wid & 1;\n"                                                \
"    int cta_m  = blockIdx.y * 64;\n"                                        \
"    int cta_n  = blockIdx.x * 128;\n"                                       \
"    int wm_row = warp_m * 16;\n"                                            \
"    int wn_col = warp_n * 64;\n"                                            \
"\n"                                                                         \
"    if (cta_m >= M) return;\n"                                              \
"\n"                                                                         \
"    float d0[8], d1[8], d2[8], d3[8];\n"                                    \
"    #pragma unroll\n"                                                       \
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"  \
"\n"                                                                         \
"    int row_a    = tid >> 2;\n"                                             \
"    int col_a    = (tid & 3) * 8;\n"                                        \
"    int g_row_a  = cta_m + row_a;\n"                                        \
"    int sw_a     = (row_a & 3) * 8;\n"                                      \
"    int swz_a    = row_a * 32 + (col_a ^ sw_a);\n"                          \
"\n"                                                                         \
"    int row_b0   = tid >> 2;\n"                                             \
"    int col_b0   = (tid & 3) * 8;\n"                                        \
"    int g_row_b0 = cta_n + row_b0;\n"                                       \
"    int sw_b0    = (row_b0 & 3) * 8;\n"                                     \
"    int swz_b0   = row_b0 * 32 + (col_b0 ^ sw_b0);\n"                       \
"\n"                                                                         \
"    int vid_b1   = tid + 256;\n"                                            \
"    int row_b1   = vid_b1 >> 2;\n"                                          \
"    int col_b1   = (vid_b1 & 3) * 8;\n"                                     \
"    int g_row_b1 = cta_n + row_b1;\n"                                       \
"    int sw_b1    = (row_b1 & 3) * 8;\n"                                     \
"    int swz_b1   = row_b1 * 32 + (col_b1 ^ sw_b1);\n"                       \
"\n"                                                                         \
"    /* Prologue */\n"                                                       \
"    {\n"                                                                    \
"        unsigned int dA = __cvta_generic_to_shared(&sA0[swz_a]);\n"         \
"        if (g_row_a < M) {\n"                                               \
"            const " ELEM_T " *src = &X[(size_t)g_row_a * K + 0 + col_a];\n" \
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"\
"        unsigned int dB0 = __cvta_generic_to_shared(&sB0[swz_b0]);\n"       \
"        if (g_row_b0 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b0 * K + 0 + col_b0];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"\
"        unsigned int dB1 = __cvta_generic_to_shared(&sB0[swz_b1]);\n"       \
"        if (g_row_b1 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b1 * K + 0 + col_b1];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"\
"    }\n"                                                                    \
"    asm volatile(\"cp.async.commit_group;\\n\");\n"                         \
"\n"                                                                         \
"    int num_k = K >> 5;\n"                                                  \
"    int la_row     = wm_row + (lane & 15);\n"                               \
"    int la_col_off = (lane >> 4) * 8;\n"                                    \
"    int la_sw      = (la_row & 3) * 8;\n"                                   \
"    int lb_row     = lane & 15;\n"                                          \
"    int lb_col_off = (lane >> 4) * 8;\n"                                    \
"\n"                                                                         \
"    for (int ki = 0; ki < num_k; ki++) {\n"                                 \
"        int stage = ki & 1;\n"                                              \
"        int next_stage = stage ^ 1;\n"                                      \
"        int next_k = (ki + 1) << 5;\n"                                      \
"\n"                                                                         \
"        asm volatile(\"cp.async.wait_group 0;\\n\");\n"                     \
"        __syncthreads();\n"                                                 \
"\n"                                                                         \
"        if (next_k < K) {\n"                                                \
"            " ELEM_T " *sA_next = (next_stage == 0) ? sA0 : sA1;\n"         \
"            " ELEM_T " *sB_next = (next_stage == 0) ? sB0 : sB1;\n"         \
"            unsigned int dA = __cvta_generic_to_shared(&sA_next[swz_a]);\n" \
"            if (g_row_a < M) {\n"                                           \
"                const " ELEM_T " *src = &X[(size_t)g_row_a * K + next_k + col_a];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"\
"            unsigned int dB0 = __cvta_generic_to_shared(&sB_next[swz_b0]);\n"\
"            if (g_row_b0 < N) {\n"                                          \
"                const " ELEM_T " *src = &W[(size_t)g_row_b0 * K + next_k + col_b0];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"\
"            unsigned int dB1 = __cvta_generic_to_shared(&sB_next[swz_b1]);\n"\
"            if (g_row_b1 < N) {\n"                                          \
"                const " ELEM_T " *src = &W[(size_t)g_row_b1 * K + next_k + col_b1];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"\
"        }\n"                                                                \
"        asm volatile(\"cp.async.commit_group;\\n\");\n"                     \
"\n"                                                                         \
"        " ELEM_T " *sAp = (stage == 0) ? sA0 : sA1;\n"                      \
"        " ELEM_T " *sBp = (stage == 0) ? sB0 : sB1;\n"                      \
"        #pragma unroll\n"                                                   \
"        for (int kg = 0; kg < 2; kg++) {\n"                                 \
"            int k_off = kg * 16;\n"                                         \
"            int a_off = la_row * 32 + ((k_off + la_col_off) ^ la_sw);\n"    \
"            unsigned int a0,a1,a2,a3;\n"                                    \
"            unsigned int p_a = __cvta_generic_to_shared(&sAp[a_off]);\n"    \
"            asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"\
"                : \"=r\"(a0), \"=r\"(a1), \"=r\"(a2), \"=r\"(a3) : \"r\"(p_a));\n"\
"            #pragma unroll\n"                                               \
"            for (int s = 0; s < 4; s++) {\n"                                \
"                int sn_base = wn_col + s * 16;\n"                           \
"                int b_brow  = sn_base + lb_row;\n"                          \
"                int b_sw    = (b_brow & 3) * 8;\n"                          \
"                int b_off   = b_brow * 32 + ((k_off + lb_col_off) ^ b_sw);\n"\
"                unsigned int b0,b1,b2,b3;\n"                                \
"                unsigned int p_b = __cvta_generic_to_shared(&sBp[b_off]);\n"\
"                asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"\
"                    : \"=r\"(b0), \"=r\"(b1), \"=r\"(b2), \"=r\"(b3) : \"r\"(p_b));\n"\
"                int nt0 = s * 2 + 0;\n"                                     \
"                int nt1 = s * 2 + 1;\n"                                     \
"                asm volatile(\n"                                            \
"                    \"" MMA_OP "\\n\\t\"\n"                                 \
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"\
"                    : \"=f\"(d0[nt0]), \"=f\"(d1[nt0]), \"=f\"(d2[nt0]), \"=f\"(d3[nt0])\n"\
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"        \
"                      \"r\"(b0), \"r\"(b2),\n"                              \
"                      \"f\"(d0[nt0]), \"f\"(d1[nt0]), \"f\"(d2[nt0]), \"f\"(d3[nt0])\n"\
"                );\n"                                                       \
"                asm volatile(\n"                                            \
"                    \"" MMA_OP "\\n\\t\"\n"                                 \
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"\
"                    : \"=f\"(d0[nt1]), \"=f\"(d1[nt1]), \"=f\"(d2[nt1]), \"=f\"(d3[nt1])\n"\
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"        \
"                      \"r\"(b1), \"r\"(b3),\n"                              \
"                      \"f\"(d0[nt1]), \"f\"(d1[nt1]), \"f\"(d2[nt1]), \"f\"(d3[nt1])\n"\
"                );\n"                                                       \
"            }\n"                                                            \
"        }\n"                                                                \
"    }\n"                                                                    \
"    asm volatile(\"cp.async.wait_group 0;\\n\");\n"                         \
"\n"                                                                         \
"    int yr0 = cta_m + wm_row + gid;\n"                                      \
"    int yr1 = cta_m + wm_row + gid + 8;\n"                                  \
"    #pragma unroll\n"                                                       \
"    for (int nt = 0; nt < 8; nt++) {\n"                                     \
"        int yc0 = cta_n + wn_col + nt*8 + tid4*2;\n"                        \
"        int yc1 = yc0 + 1;\n"                                               \
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"       \
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"       \
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"       \
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"       \
"    }\n"                                                                    \
"}\n"

static const char k_gemm_f16_v5_src[] =
"typedef unsigned short half_raw;\n"
"extern \"C\" __global__ void gemm_f16_v5(float *Y,\n"
"                                          const half_raw *X,\n"
"                                          const half_raw *W,\n"
"                                          int M, int N, int K) {\n"
CUDA_GEMM_V5_BODY("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", "half_raw");

static const char k_gemm_bf16_v5_src[] =
"typedef unsigned short bf16_raw;\n"
"extern \"C\" __global__ void gemm_bf16_v5(float *Y,\n"
"                                           const bf16_raw *X,\n"
"                                           const bf16_raw *W,\n"
"                                           int M, int N, int K) {\n"
CUDA_GEMM_V5_BODY("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", "bf16_raw");

/* ==================================================================== *
 * v6 kernels — v5 + 3-stage cp.async pipeline
 * --------------------------------------------------------------------
 * Same swizzled SMEM, ldmatrix.x4 mma loop as v5. Pipeline depth bumps
 * 2 → 3 stages so 1 cp.async group is always in flight while another
 * one is computing (vs v5 where compute-side and load-side are still
 * loosely serialized through wait_group 0).
 *
 * Pattern:
 *   prefetch stage 0 → commit_group #0
 *   prefetch stage 1 → commit_group #1     (now 2 in flight)
 *   for ki = 0 .. nk-1:
 *       wait_group 1                        (at most 1 in flight; stage ki done)
 *       __syncthreads()
 *       if ki + 2 < nk:
 *           prefetch stage (ki+2) → commit_group
 *       compute on stage ki
 *   wait_group 0
 *
 * SMEM = 3 × 12 KiB = 36 KiB (still under 48 KiB default cap).
 * Stage layout: stage s lives at smem_v6 + s * 6144 halves.
 *   sA[s]: smem_v6 + s*6144 .. + 2048  (64×32)
 *   sB[s]: smem_v6 + s*6144 + 2048 .. + 6144  (128×32)
 * ==================================================================== */

#define CUDA_GEMM_V6_BODY(MMA_OP, ELEM_T)                                    \
"    extern __shared__ __align__(16) " ELEM_T " smem_v6[];\n"                \
"    int tid = threadIdx.x;\n"                                               \
"    int wid = tid >> 5;\n"                                                  \
"    int lane = tid & 31;\n"                                                 \
"    int gid  = lane >> 2;\n"                                                \
"    int tid4 = lane & 3;\n"                                                 \
"    int warp_m = wid >> 1;\n"                                               \
"    int warp_n = wid & 1;\n"                                                \
"    int cta_m  = blockIdx.y * 64;\n"                                        \
"    int cta_n  = blockIdx.x * 128;\n"                                       \
"    int wm_row = warp_m * 16;\n"                                            \
"    int wn_col = warp_n * 64;\n"                                            \
"\n"                                                                         \
"    if (cta_m >= M) return;\n"                                              \
"\n"                                                                         \
"    float d0[8], d1[8], d2[8], d3[8];\n"                                    \
"    #pragma unroll\n"                                                       \
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"  \
"\n"                                                                         \
"    int row_a    = tid >> 2;\n"                                             \
"    int col_a    = (tid & 3) * 8;\n"                                        \
"    int g_row_a  = cta_m + row_a;\n"                                        \
"    int sw_a     = (row_a & 3) * 8;\n"                                      \
"    int swz_a    = row_a * 32 + (col_a ^ sw_a);\n"                          \
"\n"                                                                         \
"    int row_b0   = tid >> 2;\n"                                             \
"    int col_b0   = (tid & 3) * 8;\n"                                        \
"    int g_row_b0 = cta_n + row_b0;\n"                                       \
"    int sw_b0    = (row_b0 & 3) * 8;\n"                                     \
"    int swz_b0   = row_b0 * 32 + (col_b0 ^ sw_b0);\n"                       \
"\n"                                                                         \
"    int vid_b1   = tid + 256;\n"                                            \
"    int row_b1   = vid_b1 >> 2;\n"                                          \
"    int col_b1   = (vid_b1 & 3) * 8;\n"                                     \
"    int g_row_b1 = cta_n + row_b1;\n"                                       \
"    int sw_b1    = (row_b1 & 3) * 8;\n"                                     \
"    int swz_b1   = row_b1 * 32 + (col_b1 ^ sw_b1);\n"                       \
"\n"                                                                         \
"    int num_k = K >> 5;\n"                                                  \
"\n"                                                                         \
"    /* Lambda-style helper not portable in NVRTC C++ for inline-asm.   */\n" \
"    /* Inline the prefetch into a do-block keyed on stage and k-base.  */\n" \
"\n"                                                                         \
"    /* Prologue: issue stage 0 and stage 1 (if K > 32). */\n"               \
"    {\n"                                                                    \
"        " ELEM_T " *sA_st = smem_v6;\n"                                     \
"        " ELEM_T " *sB_st = smem_v6 + 2048;\n"                              \
"        unsigned int dA = __cvta_generic_to_shared(&sA_st[swz_a]);\n"       \
"        if (g_row_a < M) {\n"                                               \
"            const " ELEM_T " *src = &X[(size_t)g_row_a * K + 0 + col_a];\n" \
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"\
"        unsigned int dB0 = __cvta_generic_to_shared(&sB_st[swz_b0]);\n"     \
"        if (g_row_b0 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b0 * K + 0 + col_b0];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"\
"        unsigned int dB1 = __cvta_generic_to_shared(&sB_st[swz_b1]);\n"     \
"        if (g_row_b1 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b1 * K + 0 + col_b1];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"\
"    }\n"                                                                    \
"    asm volatile(\"cp.async.commit_group;\\n\");\n"                         \
"    if (num_k > 1) {\n"                                                     \
"        " ELEM_T " *sA_st = smem_v6 + 6144;\n"                              \
"        " ELEM_T " *sB_st = smem_v6 + 6144 + 2048;\n"                       \
"        unsigned int dA = __cvta_generic_to_shared(&sA_st[swz_a]);\n"       \
"        if (g_row_a < M) {\n"                                               \
"            const " ELEM_T " *src = &X[(size_t)g_row_a * K + 32 + col_a];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"\
"        unsigned int dB0 = __cvta_generic_to_shared(&sB_st[swz_b0]);\n"     \
"        if (g_row_b0 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b0 * K + 32 + col_b0];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"\
"        unsigned int dB1 = __cvta_generic_to_shared(&sB_st[swz_b1]);\n"     \
"        if (g_row_b1 < N) {\n"                                              \
"            const " ELEM_T " *src = &W[(size_t)g_row_b1 * K + 32 + col_b1];\n"\
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"\
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"\
"        asm volatile(\"cp.async.commit_group;\\n\");\n"                     \
"    } else {\n"                                                             \
"        /* Issue an empty group so wait_group 1 is well-defined */\n"       \
"        asm volatile(\"cp.async.commit_group;\\n\");\n"                     \
"    }\n"                                                                    \
"\n"                                                                         \
"    int la_row     = wm_row + (lane & 15);\n"                               \
"    int la_col_off = (lane >> 4) * 8;\n"                                    \
"    int la_sw      = (la_row & 3) * 8;\n"                                   \
"    int lb_row     = lane & 15;\n"                                          \
"    int lb_col_off = (lane >> 4) * 8;\n"                                    \
"\n"                                                                         \
"    for (int ki = 0; ki < num_k; ki++) {\n"                                 \
"        int stage = ki % 3;\n"                                              \
"        int next2_stage = (ki + 2) % 3;\n"                                  \
"        int next2_k = (ki + 2) << 5;\n"                                     \
"\n"                                                                         \
"        asm volatile(\"cp.async.wait_group 1;\\n\");\n"                     \
"        __syncthreads();\n"                                                 \
"\n"                                                                         \
"        if (next2_k < K) {\n"                                               \
"            " ELEM_T " *sA_next = smem_v6 + next2_stage * 6144;\n"          \
"            " ELEM_T " *sB_next = smem_v6 + next2_stage * 6144 + 2048;\n"   \
"            unsigned int dA = __cvta_generic_to_shared(&sA_next[swz_a]);\n" \
"            if (g_row_a < M) {\n"                                           \
"                const " ELEM_T " *src = &X[(size_t)g_row_a * K + next2_k + col_a];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"\
"            unsigned int dB0 = __cvta_generic_to_shared(&sB_next[swz_b0]);\n"\
"            if (g_row_b0 < N) {\n"                                          \
"                const " ELEM_T " *src = &W[(size_t)g_row_b0 * K + next2_k + col_b0];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"\
"            unsigned int dB1 = __cvta_generic_to_shared(&sB_next[swz_b1]);\n"\
"            if (g_row_b1 < N) {\n"                                          \
"                const " ELEM_T " *src = &W[(size_t)g_row_b1 * K + next2_k + col_b1];\n"\
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"\
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"\
"            asm volatile(\"cp.async.commit_group;\\n\");\n"                 \
"        } else {\n"                                                         \
"            /* Empty commit so wait_group 1 still has a group to wait on */\n"\
"            asm volatile(\"cp.async.commit_group;\\n\");\n"                 \
"        }\n"                                                                \
"\n"                                                                         \
"        " ELEM_T " *sAp = smem_v6 + stage * 6144;\n"                        \
"        " ELEM_T " *sBp = smem_v6 + stage * 6144 + 2048;\n"                 \
"        #pragma unroll\n"                                                   \
"        for (int kg = 0; kg < 2; kg++) {\n"                                 \
"            int k_off = kg * 16;\n"                                         \
"            int a_off = la_row * 32 + ((k_off + la_col_off) ^ la_sw);\n"    \
"            unsigned int a0,a1,a2,a3;\n"                                    \
"            unsigned int p_a = __cvta_generic_to_shared(&sAp[a_off]);\n"    \
"            asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"\
"                : \"=r\"(a0), \"=r\"(a1), \"=r\"(a2), \"=r\"(a3) : \"r\"(p_a));\n"\
"            #pragma unroll\n"                                               \
"            for (int s = 0; s < 4; s++) {\n"                                \
"                int sn_base = wn_col + s * 16;\n"                           \
"                int b_brow  = sn_base + lb_row;\n"                          \
"                int b_sw    = (b_brow & 3) * 8;\n"                          \
"                int b_off   = b_brow * 32 + ((k_off + lb_col_off) ^ b_sw);\n"\
"                unsigned int b0,b1,b2,b3;\n"                                \
"                unsigned int p_b = __cvta_generic_to_shared(&sBp[b_off]);\n"\
"                asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"\
"                    : \"=r\"(b0), \"=r\"(b1), \"=r\"(b2), \"=r\"(b3) : \"r\"(p_b));\n"\
"                int nt0 = s * 2 + 0;\n"                                     \
"                int nt1 = s * 2 + 1;\n"                                     \
"                asm volatile(\n"                                            \
"                    \"" MMA_OP "\\n\\t\"\n"                                 \
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"\
"                    : \"=f\"(d0[nt0]), \"=f\"(d1[nt0]), \"=f\"(d2[nt0]), \"=f\"(d3[nt0])\n"\
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"        \
"                      \"r\"(b0), \"r\"(b2),\n"                              \
"                      \"f\"(d0[nt0]), \"f\"(d1[nt0]), \"f\"(d2[nt0]), \"f\"(d3[nt0])\n"\
"                );\n"                                                       \
"                asm volatile(\n"                                            \
"                    \"" MMA_OP "\\n\\t\"\n"                                 \
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"\
"                    : \"=f\"(d0[nt1]), \"=f\"(d1[nt1]), \"=f\"(d2[nt1]), \"=f\"(d3[nt1])\n"\
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"        \
"                      \"r\"(b1), \"r\"(b3),\n"                              \
"                      \"f\"(d0[nt1]), \"f\"(d1[nt1]), \"f\"(d2[nt1]), \"f\"(d3[nt1])\n"\
"                );\n"                                                       \
"            }\n"                                                            \
"        }\n"                                                                \
"    }\n"                                                                    \
"    asm volatile(\"cp.async.wait_group 0;\\n\");\n"                         \
"\n"                                                                         \
"    int yr0 = cta_m + wm_row + gid;\n"                                      \
"    int yr1 = cta_m + wm_row + gid + 8;\n"                                  \
"    #pragma unroll\n"                                                       \
"    for (int nt = 0; nt < 8; nt++) {\n"                                     \
"        int yc0 = cta_n + wn_col + nt*8 + tid4*2;\n"                        \
"        int yc1 = yc0 + 1;\n"                                               \
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"       \
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"       \
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"       \
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"       \
"    }\n"                                                                    \
"}\n"

static const char k_gemm_f16_v6_src[] =
"typedef unsigned short half_raw;\n"
"extern \"C\" __global__ void gemm_f16_v6(float *Y,\n"
"                                          const half_raw *X,\n"
"                                          const half_raw *W,\n"
"                                          int M, int N, int K) {\n"
CUDA_GEMM_V6_BODY("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", "half_raw");

static const char k_gemm_bf16_v6_src[] =
"typedef unsigned short bf16_raw;\n"
"extern \"C\" __global__ void gemm_bf16_v6(float *Y,\n"
"                                           const bf16_raw *X,\n"
"                                           const bf16_raw *W,\n"
"                                           int M, int N, int K) {\n"
CUDA_GEMM_V6_BODY("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", "bf16_raw");

/* ==================================================================== *
 * fp8 v5 — same recipe as f16/bf16 v5 ported to e4m3 m16n8k32.
 * --------------------------------------------------------------------
 * SMEM layout is byte-equivalent to f16 v5 (24 KiB, 2 stages):
 *   sA: 64 rows × 32 b16 = 64 rows × 64 fp8 (per stage)
 *   sB: 128 rows × 32 b16 = 128 rows × 64 fp8 (per stage)
 * K-step = 64 fp8 = 32 b16. Inner kg loop (2 iters) covers 16 b16 each
 * = 32 fp8 per mma.m16n8k32.  ldmatrix.x4.b16 yields:
 *   A operand: 4 b32 = 16 fp8 per lane (= mma A frag for m16n8k32 .e4m3)
 *   B operand: 4 b32 covering 16 N × 16 K_b16 = 2 mma N-tiles per call,
 *              same {reg0,reg2}/{reg1,reg3} pairing as f16 v5.
 * XOR swizzle in b16 view is identical to v5 — `(row & 3) * 8` four-way.
 * ==================================================================== */

static const char k_gemm_fp8_v5_src[] =
"typedef unsigned char fp8_raw;\n"
"extern \"C\" __global__ void gemm_fp8_v5(float *Y,\n"
"                                          const fp8_raw *X,\n"
"                                          const fp8_raw *W,\n"
"                                          int M, int N, int K) {\n"
"    extern __shared__ __align__(16) unsigned short smem_v5[];\n"
"    unsigned short *sA0 = smem_v5;\n"
"    unsigned short *sB0 = smem_v5 + 2048;\n"
"    unsigned short *sA1 = smem_v5 + 6144;\n"
"    unsigned short *sB1 = smem_v5 + 8192;\n"
"    int tid = threadIdx.x;\n"
"    int wid = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int gid  = lane >> 2;\n"
"    int tid4 = lane & 3;\n"
"    int warp_m = wid >> 1;\n"
"    int warp_n = wid & 1;\n"
"    int cta_m  = blockIdx.y * 64;\n"
"    int cta_n  = blockIdx.x * 128;\n"
"    int wm_row = warp_m * 16;\n"
"    int wn_col = warp_n * 64;\n"
"\n"
"    if (cta_m >= M) return;\n"
"\n"
"    float d0[8], d1[8], d2[8], d3[8];\n"
"    #pragma unroll\n"
"    for (int i = 0; i < 8; i++) { d0[i]=0; d1[i]=0; d2[i]=0; d3[i]=0; }\n"
"\n"
"    /* SMEM addr in b16 units; global col in fp8 bytes (= 2 * b16 col) */\n"
"    int row_a    = tid >> 2;\n"
"    int col_a_b  = (tid & 3) * 8;        /* SMEM b16 col */\n"
"    int col_a_g  = (tid & 3) * 16;       /* global fp8 byte col */\n"
"    int g_row_a  = cta_m + row_a;\n"
"    int sw_a     = (row_a & 3) * 8;\n"
"    int swz_a    = row_a * 32 + (col_a_b ^ sw_a);\n"
"\n"
"    int row_b0   = tid >> 2;\n"
"    int col_b0_b = (tid & 3) * 8;\n"
"    int col_b0_g = (tid & 3) * 16;\n"
"    int g_row_b0 = cta_n + row_b0;\n"
"    int sw_b0    = (row_b0 & 3) * 8;\n"
"    int swz_b0   = row_b0 * 32 + (col_b0_b ^ sw_b0);\n"
"\n"
"    int vid_b1   = tid + 256;\n"
"    int row_b1   = vid_b1 >> 2;\n"
"    int col_b1_b = (vid_b1 & 3) * 8;\n"
"    int col_b1_g = (vid_b1 & 3) * 16;\n"
"    int g_row_b1 = cta_n + row_b1;\n"
"    int sw_b1    = (row_b1 & 3) * 8;\n"
"    int swz_b1   = row_b1 * 32 + (col_b1_b ^ sw_b1);\n"
"\n"
"    /* Prologue */\n"
"    {\n"
"        unsigned int dA = __cvta_generic_to_shared(&sA0[swz_a]);\n"
"        if (g_row_a < M) {\n"
"            const fp8_raw *src = &X[(size_t)g_row_a * K + 0 + col_a_g];\n"
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"
"        unsigned int dB0 = __cvta_generic_to_shared(&sB0[swz_b0]);\n"
"        if (g_row_b0 < N) {\n"
"            const fp8_raw *src = &W[(size_t)g_row_b0 * K + 0 + col_b0_g];\n"
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"
"        unsigned int dB1 = __cvta_generic_to_shared(&sB0[swz_b1]);\n"
"        if (g_row_b1 < N) {\n"
"            const fp8_raw *src = &W[(size_t)g_row_b1 * K + 0 + col_b1_g];\n"
"            asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"
"        } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"
"    }\n"
"    asm volatile(\"cp.async.commit_group;\\n\");\n"
"\n"
"    int num_k = K >> 6;                              /* K-step = 64 fp8 */\n"
"    int la_row     = wm_row + (lane & 15);\n"
"    int la_col_off = (lane >> 4) * 8;                /* b16 cols */\n"
"    int la_sw      = (la_row & 3) * 8;\n"
"    int lb_row     = lane & 15;\n"
"    int lb_col_off = (lane >> 4) * 8;\n"
"\n"
"    for (int ki = 0; ki < num_k; ki++) {\n"
"        int stage = ki & 1;\n"
"        int next_stage = stage ^ 1;\n"
"        int next_k = (ki + 1) << 6;                  /* fp8 byte stride */\n"
"\n"
"        asm volatile(\"cp.async.wait_group 0;\\n\");\n"
"        __syncthreads();\n"
"\n"
"        if (next_k < K) {\n"
"            unsigned short *sA_next = (next_stage == 0) ? sA0 : sA1;\n"
"            unsigned short *sB_next = (next_stage == 0) ? sB0 : sB1;\n"
"            unsigned int dA = __cvta_generic_to_shared(&sA_next[swz_a]);\n"
"            if (g_row_a < M) {\n"
"                const fp8_raw *src = &X[(size_t)g_row_a * K + next_k + col_a_g];\n"
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dA), \"l\"(src));\n"
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dA)); }\n"
"            unsigned int dB0 = __cvta_generic_to_shared(&sB_next[swz_b0]);\n"
"            if (g_row_b0 < N) {\n"
"                const fp8_raw *src = &W[(size_t)g_row_b0 * K + next_k + col_b0_g];\n"
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB0), \"l\"(src));\n"
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB0)); }\n"
"            unsigned int dB1 = __cvta_generic_to_shared(&sB_next[swz_b1]);\n"
"            if (g_row_b1 < N) {\n"
"                const fp8_raw *src = &W[(size_t)g_row_b1 * K + next_k + col_b1_g];\n"
"                asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(dB1), \"l\"(src));\n"
"            } else { asm volatile(\"st.shared.v4.b32 [%0], {0,0,0,0};\\n\" :: \"r\"(dB1)); }\n"
"        }\n"
"        asm volatile(\"cp.async.commit_group;\\n\");\n"
"\n"
"        unsigned short *sAp = (stage == 0) ? sA0 : sA1;\n"
"        unsigned short *sBp = (stage == 0) ? sB0 : sB1;\n"
"        #pragma unroll\n"
"        for (int kg = 0; kg < 2; kg++) {\n"
"            int k_off = kg * 16;                     /* b16 col within stage */\n"
"            int a_off = la_row * 32 + ((k_off + la_col_off) ^ la_sw);\n"
"            unsigned int a0,a1,a2,a3;\n"
"            unsigned int p_a = __cvta_generic_to_shared(&sAp[a_off]);\n"
"            asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"
"                : \"=r\"(a0), \"=r\"(a1), \"=r\"(a2), \"=r\"(a3) : \"r\"(p_a));\n"
"            #pragma unroll\n"
"            for (int s = 0; s < 4; s++) {\n"
"                int sn_base = wn_col + s * 16;\n"
"                int b_brow  = sn_base + lb_row;\n"
"                int b_sw    = (b_brow & 3) * 8;\n"
"                int b_off   = b_brow * 32 + ((k_off + lb_col_off) ^ b_sw);\n"
"                unsigned int b0,b1,b2,b3;\n"
"                unsigned int p_b = __cvta_generic_to_shared(&sBp[b_off]);\n"
"                asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\\n\"\n"
"                    : \"=r\"(b0), \"=r\"(b1), \"=r\"(b2), \"=r\"(b3) : \"r\"(p_b));\n"
"                int nt0 = s * 2 + 0;\n"
"                int nt1 = s * 2 + 1;\n"
"                asm volatile(\n"
"                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    : \"=f\"(d0[nt0]), \"=f\"(d1[nt0]), \"=f\"(d2[nt0]), \"=f\"(d3[nt0])\n"
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                      \"r\"(b0), \"r\"(b2),\n"
"                      \"f\"(d0[nt0]), \"f\"(d1[nt0]), \"f\"(d2[nt0]), \"f\"(d3[nt0])\n"
"                );\n"
"                asm volatile(\n"
"                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\\n\\t\"\n"
"                    \"    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\"\n"
"                    : \"=f\"(d0[nt1]), \"=f\"(d1[nt1]), \"=f\"(d2[nt1]), \"=f\"(d3[nt1])\n"
"                    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3),\n"
"                      \"r\"(b1), \"r\"(b3),\n"
"                      \"f\"(d0[nt1]), \"f\"(d1[nt1]), \"f\"(d2[nt1]), \"f\"(d3[nt1])\n"
"                );\n"
"            }\n"
"        }\n"
"    }\n"
"    asm volatile(\"cp.async.wait_group 0;\\n\");\n"
"\n"
"    int yr0 = cta_m + wm_row + gid;\n"
"    int yr1 = cta_m + wm_row + gid + 8;\n"
"    #pragma unroll\n"
"    for (int nt = 0; nt < 8; nt++) {\n"
"        int yc0 = cta_n + wn_col + nt*8 + tid4*2;\n"
"        int yc1 = yc0 + 1;\n"
"        if (yr0 < M && yc0 < N) Y[(size_t)yr0 * N + yc0] = d0[nt];\n"
"        if (yr0 < M && yc1 < N) Y[(size_t)yr0 * N + yc1] = d1[nt];\n"
"        if (yr1 < M && yc0 < N) Y[(size_t)yr1 * N + yc0] = d2[nt];\n"
"        if (yr1 < M && yc1 < N) Y[(size_t)yr1 * N + yc1] = d3[nt];\n"
"    }\n"
"}\n";

#endif /* CUDA_GEMM_PTX_KERNELS_H */
