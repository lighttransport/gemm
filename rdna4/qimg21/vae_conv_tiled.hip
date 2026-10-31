/* Tiled BF16 WMMA implicit-GEMM 3x3 convolution for the Qwen-Image VAE on gfx12 (RDNA4). SPDX-License-Identifier: MIT
 *
 * out[co][H*W] (f32) = conv3x3(in[ci][H*W] f32, w[co][ci][3][3] f32) + bias, zero or replicate padding.
 * GEMM view: M = co, N = pixels, K = 9 taps x ci (tap-major). Two cheap packs make every K tile contiguous:
 *   q21v_pack_hwc  : in  -> xb[pixel][ci]        BF16 (RNE)
 *   q21v_pack_w3x3 : w   -> wb[co][tap][ci]      BF16 (RNE)
 * q21v_conv3x3_tiled: 128x128 block tile, BK=32 (one tap, 32 channels), 8 waves (2 along M x 4 along N, 64x32 each),
 * 16-byte loads, padded double-buffered LDS, register prefetch of the next K tile. Requires ci % 32 == 0.
 * WMMA fragment mapping matches vae_conv2d_3x3_wmma_f32 (A row = lane&15 = oc, B column = lane&15 = pixel,
 * K half = lane>>4). Operands are rounded to nearest (the old kernel truncated). */
typedef unsigned short q21v_u16;
typedef short q21v_bf16x8 __attribute__((ext_vector_type(8)));
typedef float q21v_float8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ q21v_u16 q21v_bf16(float f) {
    unsigned u = __float_as_uint(f);
    return (q21v_u16)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

/* 32x32 tiled transpose: xb[p * c + ic] = bf16(in[ic * sp + p]). grid (ceil(sp/32), ceil(c/32)), block (32, 8). */
extern "C" __global__ void q21v_pack_hwc(q21v_u16 *__restrict__ xb, const float *__restrict__ in, int c, int sp) {
    __shared__ float t[32][33];
    int p0 = blockIdx.x * 32, c0 = blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
    for (int i = ty; i < 32; i += 8) {
        int ic = c0 + i, p = p0 + tx;
        t[i][tx] = (ic < c && p < sp) ? in[(long)ic * sp + p] : 0.f;
    }
    __syncthreads();
    for (int i = ty; i < 32; i += 8) {
        int p = p0 + i, ic = c0 + tx;
        if (p < sp && ic < c) xb[(long)p * c + ic] = q21v_bf16(t[tx][i]);
    }
}

/* wb[(oc * 9 + tap) * ci + ic] = bf16(w[((oc * ci + ic) * 3 + fy) * 3 + fx]), tap = fy * 3 + fx */
extern "C" __global__ void q21v_pack_w3x3(q21v_u16 *__restrict__ wb, const float *__restrict__ w, int co, int ci) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x, n = (long)co * ci * 9;
    if (i >= n) return;
    int ic = (int)(i % ci), tap = (int)((i / ci) % 9), oc = (int)(i / ((long)ci * 9));
    wb[i] = q21v_bf16(w[((long)oc * ci + ic) * 9 + tap]);
}

#define Q21V_BM 128
#define Q21V_BN 128
#define Q21V_BK 32
#define Q21V_LD 40   /* 32 + 8 bf16 pad per LDS row (80 bytes) */
extern "C" __global__ __launch_bounds__(256) void q21v_conv3x3_tiled(float *__restrict__ out, const q21v_u16 *__restrict__ xb,
        const q21v_u16 *__restrict__ wb, const float *__restrict__ bias, int ci, int h, int w, int co, int rep) {
    __shared__ __attribute__((aligned(16))) q21v_u16 As[2][Q21V_BM * Q21V_LD];
    __shared__ __attribute__((aligned(16))) q21v_u16 Bs[2][Q21V_BN * Q21V_LD];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, wM = wave & 1, wN = wave >> 1;
    const int half = lane >> 4, idx = lane & 15, sp = h * w;
    const int n0 = blockIdx.x * Q21V_BN, m0 = blockIdx.y * Q21V_BM;
    /* this thread's two load slots per operand: chunk c -> row c >> 2, 8-channel column (c & 3) * 8 */
    int arow[2], acol[2], py[2], px[2], pval[2];
    for (int i = 0; i < 2; i++) {
        int c = tid + i * 256; arow[i] = c >> 2; acol[i] = (c & 3) * 8;
        int p = n0 + arow[i]; pval[i] = p < sp; py[i] = pval[i] ? p / w : 0; px[i] = pval[i] ? p % w : 0;
    }
    const int ksteps = 9 * (ci / Q21V_BK);
    uint4 ra[2], rb[2];
    auto load = [&](int ks) {
        const int tap = ks / (ci / Q21V_BK), c0 = (ks % (ci / Q21V_BK)) * Q21V_BK;
        const int dy = tap / 3 - 1, dx = tap % 3 - 1;
        for (int i = 0; i < 2; i++) {
            int oc = m0 + arow[i];
            ra[i] = oc < co ? *(const uint4 *)(wb + ((long)oc * 9 + tap) * ci + c0 + acol[i]) : make_uint4(0, 0, 0, 0);
            int iy = py[i] + dy, ix = px[i] + dx, ok = pval[i];
            if (rep) { iy = iy < 0 ? 0 : (iy >= h ? h - 1 : iy); ix = ix < 0 ? 0 : (ix >= w ? w - 1 : ix); }
            else ok = ok && iy >= 0 && iy < h && ix >= 0 && ix < w;
            rb[i] = ok ? *(const uint4 *)(xb + ((long)iy * w + ix) * ci + c0 + acol[i]) : make_uint4(0, 0, 0, 0);
        }
    };
    auto store = [&](int buf) {
        for (int i = 0; i < 2; i++) {
            *(uint4 *)(&As[buf][arow[i] * Q21V_LD + acol[i]]) = ra[i];
            *(uint4 *)(&Bs[buf][arow[i] * Q21V_LD + acol[i]]) = rb[i];
        }
    };
    q21v_float8 acc[4][2];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) acc[i][j] = (q21v_float8){0, 0, 0, 0, 0, 0, 0, 0};
    load(0); store(0); __syncthreads();
    int buf = 0;
    for (int ks = 0; ks < ksteps; ks++) {
        const bool more = ks + 1 < ksteps;
        if (more) load(ks + 1);
        const q21v_u16 *A = As[buf], *B = Bs[buf];
#pragma unroll
        for (int kk = 0; kk < Q21V_BK; kk += 16) {
            q21v_bf16x8 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = *(const q21v_bf16x8 *)(A + (wM * 64 + i * 16 + idx) * Q21V_LD + kk + half * 8);
#pragma unroll
            for (int j = 0; j < 2; j++) b[j] = *(const q21v_bf16x8 *)(B + (wN * 32 + j * 16 + idx) * Q21V_LD + kk + half * 8);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
        if (more) { store(buf ^ 1); __syncthreads(); buf ^= 1; }
    }
    for (int j = 0; j < 2; j++) {
        int p = n0 + wN * 32 + j * 16 + idx; if (p >= sp) continue;
        for (int i = 0; i < 4; i++)
            for (int e = 0; e < 8; e++) {
                int oc = m0 + wM * 64 + i * 16 + half * 8 + e; if (oc >= co) continue;
                out[(long)oc * sp + p] = acc[i][j][e] + (bias ? bias[oc] : 0.f);
            }
    }
}
