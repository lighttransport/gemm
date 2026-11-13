/*
 * praxis_fp4_kernels.h - sm_120a NVRTC kernels of the NVFP4 routed-expert path (own module: block-scale mma needs sm_120a).
 *
 * Recipe (praxis/quantization/nvfp4.py): E2M1 values, 16-element blocks along K, e4m3 block scale, fp32 per-matrix global
 * scale amax/2688; weight = fp4 * block_scale * global. Activations are quantized dynamically with the same recipe (global
 * = amax of the rows of the call / 2688). The GEMM is the m16n8k64 kind::mxf4nvf4 block-scale mma (scale_vec::4X, ue4m3).
 * Packing: a 32-bit word holds 8 consecutive k (low nibble = lowest k); scales are [row][K/16] bytes.
 * (No '#' directives and no apostrophes inside: the source lives in a macro.)
 */
#ifndef PRAXIS_FP4_KERNELS_H
#define PRAXIS_FP4_KERNELS_H
#define PRK4_STR(...) #__VA_ARGS__
static const char *praxis_fp4_kernel_src = PRK4_STR(
extern "C" {

__device__ __forceinline__ unsigned pr4_e2m1(float q) {   /* round to nearest even on the e2m1 grid */
    unsigned s = q < 0.f ? 8u : 0u;
    float a = fabsf(q);
    unsigned c = a <= 0.25f ? 0u : a < 0.75f ? 1u : a <= 1.25f ? 2u : a < 1.75f ? 3u : a <= 2.5f ? 4u : a < 3.5f ? 5u : a <= 5.0f ? 6u : 7u;
    return s | c;
}
__device__ __forceinline__ unsigned char pr4_e4m3(float v) {
    unsigned short p;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(p) : "f"(v), "f"(v));
    return (unsigned char)(p & 0xff);
}
__device__ __forceinline__ float pr4_e4m3_dec(unsigned char b) {
    int e = (b >> 3) & 15, m = b & 7;
    if (e == 0) return ((float)m * 0.125f) * 0.015625f;
    return (1.0f + (float)m * 0.125f) * exp2f((float)(e - 7));
}

/* max |x| over n floats into a zero-initialised cell (positive floats order like their bit patterns) */
__global__ void pr4_amax_f32(const float *x, long long n, unsigned *cell)
{
    float m = 0.f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) m = fmaxf(m, fabsf(x[i]));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(cell, __float_as_uint(m));
}
__global__ void pr4_amax_bf16(const unsigned short *x, long long n, unsigned *cell)
{
    float m = 0.f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(__uint_as_float(((unsigned)x[i]) << 16)));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(cell, __float_as_uint(m));
}

__device__ __forceinline__ void pr4_quant16(const float *v, float g, unsigned *w2, unsigned char *sb)
{
    float bm = 0.f;
    for (int i = 0; i < 16; i++) bm = fmaxf(bm, fabsf(v[i]));
    float t = (bm * (1.0f / 6.0f)) / g;
    t = fminf(fmaxf(t, 0.015625f), 448.0f);
    unsigned char sbyte = pr4_e4m3(t);
    float sd = pr4_e4m3_dec(sbyte);
    float rcp = (1.0f / g) / sd;
    *sb = sbyte;
    for (int u = 0; u < 2; u++) {
        unsigned wd = 0;
        for (int i = 0; i < 8; i++) {
            float q = fminf(fmaxf(v[u * 8 + i] * rcp, -6.0f), 6.0f);
            wd |= pr4_e2m1(q) << (i * 4);
        }
        w2[u] = wd;
    }
}

/* Aq [R][K/8] words, As [R][K/16] bytes; source row of r = rowmap ? rowmap[r] : r; global scale from *amax */
__global__ void pr4_quant_f32(const float *x, long long ldx, const int *rowmap, const unsigned *amax, unsigned *Aq, unsigned char *As,
                              long long R, int K)
{
    int ng = K >> 4;
    long long gi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= R * ng) return;
    long long r = gi / ng;
    int grp = (int)(gi - r * ng);
    float g = fmaxf(__uint_as_float(*amax), 1e-12f) * (1.0f / 2688.0f);
    long long sr = rowmap ? (long long)rowmap[r] : r;
    const float *p = x + sr * ldx + (long long)grp * 16;
    float v[16];
    for (int i = 0; i < 16; i++) v[i] = p[i];
    unsigned w2[2]; unsigned char sb;
    pr4_quant16(v, g, w2, &sb);
    Aq[r * (K >> 3) + grp * 2] = w2[0]; Aq[r * (K >> 3) + grp * 2 + 1] = w2[1];
    As[r * ng + grp] = sb;
}
__global__ void pr4_quant_bf16(const unsigned short *x, long long ldx, const unsigned *amax, unsigned *Aq, unsigned char *As,
                               long long R, int K)
{
    int ng = K >> 4;
    long long gi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= R * ng) return;
    long long r = gi / ng;
    int grp = (int)(gi - r * ng);
    float g = fmaxf(__uint_as_float(*amax), 1e-12f) * (1.0f / 2688.0f);
    const unsigned short *p = x + r * ldx + (long long)grp * 16;
    float v[16];
    for (int i = 0; i < 16; i++) v[i] = __uint_as_float(((unsigned)p[i]) << 16);
    unsigned w2[2]; unsigned char sb;
    pr4_quant16(v, g, w2, &sb);
    Aq[r * (K >> 3) + grp * 2] = w2[0]; Aq[r * (K >> 3) + grp * 2 + 1] = w2[1];
    As[r * ng + grp] = sb;
}

/* tile prefix over experts: tpre[e] = sum_{i<e} ceil(n_i / bm) */
__global__ void pr4_tiles(const int *seg, int E, int bm, int *tpre)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int s = 0;
        for (int e = 0; e < E; e++) { tpre[e] = s; s += (seg[e + 1] - seg[e] + bm - 1) / bm; }
        tpre[E] = s;
    }
}

__device__ __forceinline__ unsigned pr4_pack_bf16(float lo, float hi)
{
    unsigned short a, b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(a) : "f"(lo));
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(hi));
    return ((unsigned)b << 16) | (unsigned)a;
}

/* Grouped W4A4 GEMM over the expert-sorted rows. Out[row][col] (bf16, ldo) = alpha_e * sum_k A[row][k] W_e[col][k].
 * Wq/Wsf/wglob: one expert after the other ([N][K/2] bytes, [N][K/16] bytes, fp32 scalar). alpha = act_global * wglob[e].
 * Tile BM=64 x BN=128, BK=128 per pipeline stage (two k64 mma steps), 2-stage cp.async double buffering, padded smem rows
 * (20 words: conflict-free fragment loads), 8 warps (2 x 4), warp tile 32 x 32. grid (N/128, tile-upper-bound).
 * K must be a multiple of 128. Dynamic smem: 2 * 16896 bytes. */
__global__ __launch_bounds__(256, 2) void pr4_gemm_grouped(
    const unsigned *__restrict__ A, const unsigned char *__restrict__ sA,
    const unsigned char *__restrict__ Wq, const unsigned char *__restrict__ Wsf, const float *__restrict__ wglob,
    const unsigned *__restrict__ amax, const int *__restrict__ seg, const int *__restrict__ tpre, int E,
    unsigned short *__restrict__ Out, long long ldo, int N, int K)
{
    extern __shared__ unsigned pr4_sm[];
    int ty = blockIdx.y;
    int total = tpre[E];
    if (ty >= total) return;
    int lo = 0, hi = E;                   /* largest e with tpre[e] <= ty */
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (tpre[mid] <= ty) lo = mid; else hi = mid; }
    int e = lo;
    int row0 = seg[e] + (ty - tpre[e]) * 64, mend = seg[e + 1];
    int bn0 = blockIdx.x * 128;
    const unsigned char *Bb = Wq + (size_t)e * N * (K >> 1);
    const unsigned char *sB = Wsf + (size_t)e * N * (K >> 4);
    float alpha = fmaxf(__uint_as_float(*amax), 1e-12f) * (1.0f / 2688.0f) * wglob[e];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int wm = warp >> 2, wn = warp & 3;
    int g = lane >> 2, t = lane & 3;
    /* stage layout (words): A [64][20], B [128][20], SA [64][2], SB [128][2]  = 1280 + 2560 + 128 + 256 = 4224 words */
    const int STW = 4224;
    long long Kb = K >> 1, Kg = K >> 4;
    float acc[2][4][4];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;
    int nkt = K >> 7;

    auto issue = [&](int kt, int st) {
        unsigned *sa = pr4_sm + st * STW, *sb = sa + 1280, *ssa = sb + 2560, *ssb = ssa + 128;
        long long kb0 = (long long)kt * 64, kg0 = (long long)kt * 8;
        {   /* A: 64 rows x 4 chunks of 16 B (one per thread) */
            int r = tid >> 2, c = tid & 3; int gr = row0 + r; int ok = gr < mend;
            const unsigned char *src = (const unsigned char *)A + (ok ? ((long long)gr * Kb + kb0 + c * 16) : 0);
            unsigned dst = __cvta_generic_to_shared(sa + r * 20 + c * 4);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
        }
        for (int idx = tid; idx < 128 * 4; idx += 256) {   /* B: 128 rows x 4 chunks */
            int r = idx >> 2, c = idx & 3; int gr = bn0 + r; int ok = gr < N;
            const unsigned char *src = Bb + (ok ? ((long long)gr * Kb + kb0 + c * 16) : 0);
            unsigned dst = __cvta_generic_to_shared(sb + r * 20 + c * 4);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
        }
        if (tid < 64) {   /* A scales: 8 bytes per row */
            int gr = row0 + tid; int ok = gr < mend;
            const unsigned char *src = sA + (ok ? ((long long)gr * Kg + kg0) : 0);
            unsigned dst = __cvta_generic_to_shared(ssa + tid * 2);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(dst), "l"(src), "r"(ok ? 8 : 0));
        }
        if (tid < 128) {
            int gr = bn0 + tid; int ok = gr < N;
            const unsigned char *src = sB + (ok ? ((long long)gr * Kg + kg0) : 0);
            unsigned dst = __cvta_generic_to_shared(ssb + tid * 2);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(dst), "l"(src), "r"(ok ? 8 : 0));
        }
    };
    issue(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    for (int kt = 0; kt < nkt; kt++) {
        int cur = kt & 1;
        if (kt + 1 < nkt) { issue(kt + 1, cur ^ 1); asm volatile("cp.async.commit_group;\n" ::); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const unsigned *sa = pr4_sm + cur * STW, *sb = sa + 1280, *ssa = sb + 2560, *ssb = ssa + 128;
        for (int ks = 0; ks < 2; ks++) {
            unsigned af[2][4], sfa[2], bf[4][2], sfb[4];
            for (int i = 0; i < 2; i++) {
                int mr = wm * 32 + i * 16;
                af[i][0] = sa[(mr + g) * 20 + ks * 8 + t]; af[i][1] = sa[(mr + g + 8) * 20 + ks * 8 + t];
                af[i][2] = sa[(mr + g) * 20 + ks * 8 + t + 4]; af[i][3] = sa[(mr + g + 8) * 20 + ks * 8 + t + 4];
                unsigned s = 0x38383838u;
                if (t == 0) s = ssa[(mr + g) * 2 + ks]; else if (t == 1) s = ssa[(mr + g + 8) * 2 + ks];
                sfa[i] = s;
            }
            for (int j = 0; j < 4; j++) {
                int nr = wn * 32 + j * 8;
                bf[j][0] = sb[(nr + g) * 20 + ks * 8 + t]; bf[j][1] = sb[(nr + g) * 20 + ks * 8 + t + 4];
                unsigned s = 0x38383838u;
                if (t == 0) s = ssb[(nr + g) * 2 + ks];
                sfb[j] = s;
            }
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++)
                    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
                                 : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                                 : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]), "r"(sfa[i]), "r"(sfb[j]));
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) {
            int mr = row0 + wm * 32 + i * 16, nc = bn0 + wn * 32 + j * 8;
            int r0 = mr + g, r1 = mr + g + 8, col = nc + 2 * t;
            if (col < N) {
                if (r0 < mend) *(unsigned *)(Out + (long long)r0 * ldo + col) = pr4_pack_bf16(acc[i][j][0] * alpha, acc[i][j][1] * alpha);
                if (r1 < mend) *(unsigned *)(Out + (long long)r1 * ldo + col) = pr4_pack_bf16(acc[i][j][2] * alpha, acc[i][j][3] * alpha);
            }
        }
}

/* ---- W4A16 / W8A16: f16 activations x NVFP4 / FP8 weights dequantised in registers, f16 m16n8k16 mma, fp32 accumulate ----
 * The dequantisation is exact in f16 (e2m1 x e4m3 block scale has <= 5 significant bits and lies in [2^-10, 2688]); the fp32 global weight scale
 * (NVFP4) or the fp32 per-output-channel scale (FP8) is applied in the epilogue. f16 activations keep 10 mantissa bits (bf16 only 7), values are
 * saturated to +-65504. A 16-block of K is one mma k16 step, so each B fragment column has ONE block scale per step.
 * k permutation inside every k16 step (A and B use the same one, the contraction is order independent): logical k pair t / 4+t of lane t maps to
 * physical k 4t..4t+1 / 4t+2..4t+3, so a lane reads 4 consecutive weights (one 16-bit chunk of nibbles / one 32-bit word of bytes) and its
 * A fragment pairs (a0,a2), (a1,a3) are adjacent words (one 64-bit LDS each). Same bank layout as pr4_gemm_grouped (Wq [N][K/2], Wsf [N][K/16]).
 * Tile BM64 x BN128 x BK64, 2-stage cp.async, 8 warps (2 x 4), warp tile 32 x 32. K multiple of 64; dynamic smem 2 * 16896 (NVFP4) / 2 * 20480 (FP8). */
__device__ __forceinline__ unsigned pr4_pack_f16(float lo, float hi)
{
    lo = fminf(fmaxf(lo, -65504.f), 65504.f); hi = fminf(fmaxf(hi, -65504.f), 65504.f);
    unsigned o;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(o) : "f"(hi), "f"(lo));
    return o;
}
__device__ __forceinline__ unsigned pr4_e2m1x2_f16(unsigned byte)
{
    unsigned o;
    asm("{ .reg .b8 tb; cvt.u8.u32 tb, %1; cvt.rn.f16x2.e2m1x2 %0, tb; }" : "=r"(o) : "r"(byte));
    return o;
}
__device__ __forceinline__ unsigned pr4_e4m3x2_f16(unsigned two)
{
    unsigned o;
    asm("{ .reg .b16 tw; cvt.u16.u32 tw, %1; cvt.rn.f16x2.e4m3x2 %0, tw; }" : "=r"(o) : "r"(two));
    return o;
}
__device__ __forceinline__ unsigned pr4_mulh2(unsigned a, unsigned b)
{
    unsigned o;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(b));
    return o;
}
__device__ __forceinline__ void pr4_mma_f16(float *acc, const unsigned *af, const unsigned *bf)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]) : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[0]), "r"(bf[1]));
}

/* fp32 rows -> f16 rows (saturating); ld = row stride of the source in floats */
__global__ void pr4_f32_to_f16(const float *x, long long ldx, unsigned short *y, long long ldy, long long rows, int cols)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * (cols >> 1)) return;
    long long r = i / (cols >> 1);
    int c = (int)(i - r * (cols >> 1)) * 2;
    *(unsigned *)(y + r * ldy + c) = pr4_pack_f16(x[r * ldx + c], x[r * ldx + c + 1]);
}
/* f16 SwiGLU: o = silu(a) * b, fp32 math, f16 in and out (the gate_up GEMM of the A16 path writes f16) */
__global__ void pr4_swiglu_f16(const unsigned short *a, long long lda, const unsigned short *b, long long ldb, unsigned short *o, long long ldo, int H, long long rows)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * (H >> 1)) return;
    long long r = i / (H >> 1);
    int c = (int)(i - r * (H >> 1)) * 2;
    unsigned ua = *(const unsigned *)(a + r * lda + c), ub = *(const unsigned *)(b + r * ldb + c);
    float2 fa, fb;
    asm("{ .reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }" : "=f"(fa.x), "=f"(fa.y) : "r"(ua));
    asm("{ .reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }" : "=f"(fb.x), "=f"(fb.y) : "r"(ub));
    *(unsigned *)(o + r * ldo + c) = pr4_pack_f16((fa.x / (1.0f + expf(-fa.x))) * fb.x, (fa.y / (1.0f + expf(-fa.y))) * fb.y);
}

__global__ void pr4_bf16_to_f16(unsigned short *p, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f = __uint_as_float(((unsigned)p[i]) << 16);
    unsigned u = pr4_pack_f16(f, 0.f);
    p[i] = (unsigned short)(u & 0xffffu);
}

__global__ __launch_bounds__(256, 2) void pr4_gemm_w4a16(
    const unsigned short *__restrict__ A, long long lda, const int *__restrict__ rowmap,
    const unsigned char *__restrict__ Wq, const unsigned char *__restrict__ Wsf, const float *__restrict__ wglob,
    const int *__restrict__ seg, const int *__restrict__ tpre, int E,
    unsigned short *__restrict__ Out, long long ldo, int N, int K, int out_f16)
{
    extern __shared__ unsigned pr4w_sm[];
    int ty = blockIdx.y;
    int total = tpre[E];
    if (ty >= total) return;
    int lo = 0, hi = E;
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (tpre[mid] <= ty) lo = mid; else hi = mid; }
    int e = lo;
    int row0 = seg[e] + (ty - tpre[e]) * 64, mend = seg[e + 1];
    int bn0 = blockIdx.x * 128;
    const unsigned char *Bb = Wq + (size_t)e * N * (K >> 1);
    const unsigned char *sB = Wsf + (size_t)e * N * (K >> 4);
    float alpha = wglob[e];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int wm = warp >> 2, wn = warp & 3;
    int g = lane >> 2, t = lane & 3;
    /* stage (words): A [64][40] = 2560, B [128][12] = 1536, SB [128] = 128  -> 4224 */
    const int STW = 4224;
    long long Kb = K >> 1, Kg = K >> 4;
    float acc[2][4][4];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;
    int nkt = K >> 6;
    auto issue = [&](int kt, int st) {
        unsigned *sa = pr4w_sm + st * STW, *sb = sa + 2560, *ssb = sb + 1536;
        for (int idx = tid; idx < 512; idx += 256) {   /* A: 64 rows x 8 chunks of 16 B */
            int r = idx >> 3, c = idx & 7; int gr = row0 + r; int ok = gr < mend;
            long long srow = ok ? (rowmap ? (long long)rowmap[gr] : (long long)gr) : 0;
            const unsigned char *src = (const unsigned char *)A + srow * lda * 2 + (long long)kt * 128 + c * 16;
            unsigned dst = __cvta_generic_to_shared(sa + r * 40 + c * 4);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
        }
        {   /* B: 128 rows x 2 chunks of 16 B */
            int r = tid >> 1, c = tid & 1; int gr = bn0 + r; int ok = gr < N;
            const unsigned char *src = Bb + (ok ? ((long long)gr * Kb + (long long)kt * 32 + c * 16) : 0);
            unsigned dst = __cvta_generic_to_shared(sb + r * 12 + c * 4);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
        }
        if (tid < 128) {   /* B scales: 4 bytes per row */
            int gr = bn0 + tid; int ok = gr < N;
            const unsigned char *src = sB + (ok ? ((long long)gr * Kg + (long long)kt * 4) : 0);
            unsigned dst = __cvta_generic_to_shared(ssb + tid);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(dst), "l"(src), "r"(ok ? 4 : 0));
        }
    };
    issue(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    for (int kt = 0; kt < nkt; kt++) {
        int cur = kt & 1;
        if (kt + 1 < nkt) { issue(kt + 1, cur ^ 1); asm volatile("cp.async.commit_group;\n" ::); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const unsigned *sa = pr4w_sm + cur * STW, *sb = sa + 2560, *ssb = sb + 1536;
        unsigned scw[4];
        for (int j = 0; j < 4; j++) scw[j] = ssb[wn * 32 + j * 8 + g];
        for (int ks = 0; ks < 4; ks++) {
            unsigned af[2][4], bfr[4][2];
            for (int i = 0; i < 2; i++) {
                int mr = wm * 32 + i * 16;
                uint2 p0 = *(const uint2 *)(sa + (mr + g) * 40 + ks * 8 + 2 * t), p1 = *(const uint2 *)(sa + (mr + g + 8) * 40 + ks * 8 + 2 * t);
                af[i][0] = p0.x; af[i][2] = p0.y; af[i][1] = p1.x; af[i][3] = p1.y;
            }
            for (int j = 0; j < 4; j++) {
                int nr = wn * 32 + j * 8 + g;
                unsigned w = sb[nr * 12 + ks * 2 + (t >> 1)];
                unsigned c16 = (w >> (16 * (t & 1))) & 0xffffu;
                unsigned sbyte = (scw[j] >> (8 * ks)) & 0xffu;
                unsigned s2 = pr4_e4m3x2_f16(sbyte | (sbyte << 8));
                bfr[j][0] = pr4_mulh2(pr4_e2m1x2_f16(c16 & 0xffu), s2);
                bfr[j][1] = pr4_mulh2(pr4_e2m1x2_f16(c16 >> 8), s2);
            }
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++) pr4_mma_f16(acc[i][j], af[i], bfr[j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) {
            int mr = row0 + wm * 32 + i * 16, nc = bn0 + wn * 32 + j * 8;
            int r0 = mr + g, r1 = mr + g + 8, col = nc + 2 * t;
            if (col < N) {
                if (r0 < mend) *(unsigned *)(Out + (long long)r0 * ldo + col) = out_f16 ? pr4_pack_f16(acc[i][j][0] * alpha, acc[i][j][1] * alpha) : pr4_pack_bf16(acc[i][j][0] * alpha, acc[i][j][1] * alpha);
                if (r1 < mend) *(unsigned *)(Out + (long long)r1 * ldo + col) = out_f16 ? pr4_pack_f16(acc[i][j][2] * alpha, acc[i][j][3] * alpha) : pr4_pack_bf16(acc[i][j][2] * alpha, acc[i][j][3] * alpha);
            }
        }
}

/* FP8: Wq [E][N][K] e4m3 bytes, rscale [E][N] floats (per output channel, applied in the epilogue) */
__global__ __launch_bounds__(256, 2) void pr8_gemm_w8a16(
    const unsigned short *__restrict__ A, long long lda, const int *__restrict__ rowmap,
    const unsigned char *__restrict__ Wq, const float *__restrict__ rscale,
    const int *__restrict__ seg, const int *__restrict__ tpre, int E,
    unsigned short *__restrict__ Out, long long ldo, int N, int K, int out_f16)
{
    extern __shared__ unsigned pr8_sm[];
    int ty = blockIdx.y;
    int total = tpre[E];
    if (ty >= total) return;
    int lo = 0, hi = E;
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (tpre[mid] <= ty) lo = mid; else hi = mid; }
    int e = lo;
    int row0 = seg[e] + (ty - tpre[e]) * 64, mend = seg[e + 1];
    int bn0 = blockIdx.x * 128;
    const unsigned char *Bb = Wq + (size_t)e * N * K;
    const float *rs = rscale + (size_t)e * N;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int wm = warp >> 2, wn = warp & 3;
    int g = lane >> 2, t = lane & 3;
    const int STW = 5120;   /* A [64][40] = 2560, B [128][20] = 2560 */
    float acc[2][4][4];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;
    int nkt = K >> 6;
    auto issue = [&](int kt, int st) {
        unsigned *sa = pr8_sm + st * STW, *sb = sa + 2560;
        for (int idx = tid; idx < 512; idx += 256) {
            int r = idx >> 3, c = idx & 7; int gr = row0 + r; int ok = gr < mend;
            long long srow = ok ? (rowmap ? (long long)rowmap[gr] : (long long)gr) : 0;
            const unsigned char *src = (const unsigned char *)A + srow * lda * 2 + (long long)kt * 128 + c * 16;
            unsigned dst = __cvta_generic_to_shared(sa + r * 40 + c * 4);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
        }
        for (int idx = tid; idx < 512; idx += 256) {   /* B: 128 rows x 4 chunks of 16 B */
            int r = idx >> 2, c = idx & 3; int gr = bn0 + r; int ok = gr < N;
            const unsigned char *src = Bb + (ok ? ((long long)gr * K + (long long)kt * 64 + c * 16) : 0);
            unsigned dst = __cvta_generic_to_shared(sb + r * 20 + c * 4);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
        }
    };
    issue(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    for (int kt = 0; kt < nkt; kt++) {
        int cur = kt & 1;
        if (kt + 1 < nkt) { issue(kt + 1, cur ^ 1); asm volatile("cp.async.commit_group;\n" ::); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const unsigned *sa = pr8_sm + cur * STW, *sb = sa + 2560;
        for (int ks = 0; ks < 4; ks++) {
            unsigned af[2][4], bfr[4][2];
            for (int i = 0; i < 2; i++) {
                int mr = wm * 32 + i * 16;
                uint2 p0 = *(const uint2 *)(sa + (mr + g) * 40 + ks * 8 + 2 * t), p1 = *(const uint2 *)(sa + (mr + g + 8) * 40 + ks * 8 + 2 * t);
                af[i][0] = p0.x; af[i][2] = p0.y; af[i][1] = p1.x; af[i][3] = p1.y;
            }
            for (int j = 0; j < 4; j++) {
                int nr = wn * 32 + j * 8 + g;
                unsigned w = sb[nr * 20 + ks * 4 + t];
                bfr[j][0] = pr4_e4m3x2_f16(w & 0xffffu);
                bfr[j][1] = pr4_e4m3x2_f16(w >> 16);
            }
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++) pr4_mma_f16(acc[i][j], af[i], bfr[j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) {
            int mr = row0 + wm * 32 + i * 16, nc = bn0 + wn * 32 + j * 8;
            int r0 = mr + g, r1 = mr + g + 8, col = nc + 2 * t;
            if (col < N) {
                float s0 = rs[col], s1 = rs[col + 1];
                if (r0 < mend) *(unsigned *)(Out + (long long)r0 * ldo + col) = out_f16 ? pr4_pack_f16(acc[i][j][0] * s0, acc[i][j][1] * s1) : pr4_pack_bf16(acc[i][j][0] * s0, acc[i][j][1] * s1);
                if (r1 < mend) *(unsigned *)(Out + (long long)r1 * ldo + col) = out_f16 ? pr4_pack_f16(acc[i][j][2] * s0, acc[i][j][3] * s1) : pr4_pack_bf16(acc[i][j][2] * s0, acc[i][j][3] * s1);
            }
        }
}

}
);
#endif
