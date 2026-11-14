/*
 * hip_wan22_vae.h - standalone RDNA4/ROCm (HIPRTC, no hipBLAS) runner for the
 * Wan2.2 VAE (praxis/vae.py is the oracle).
 *
 *   #define HIP_WAN22_VAE_IMPLEMENTATION   in exactly one .c file
 *   #include "hip_wan22_vae.h"
 *
 * Design
 *  - Every convolution is an IMPLICIT GEMM on tensor cores (mma.sync m16n8k16,
 *    fp16 operands, fp32 accumulate, cp.async 3-stage pipeline, optional
 *    split-K): the im2col is done on the fly while the A tile is staged into
 *    shared memory, so there is no column buffer. Causal temporal handling,
 *    stride-2 down-sampling, the virtual nearest 2x up-sample, bias, residual
 *    add and fp32/fp16 output (also transposed fp16) are fused in the kernel.
 *  - Activations that feed a conv are fp16 channels-last [T][H][W][C] (this is
 *    what the im2col of the old fp16 path fed to the GEMM anyway); the
 *    residual stream and the conv outputs stay fp32. RMSNorm(+SiLU) writes the
 *    fp16 conv input directly (one fused kernel).
 *  - Self-attention (single head, C = 640 / 1024) uses the same tensor-core
 *    GEMM for QK^T and PV (V is produced transposed by the qkv epilogue) with
 *    an fp32 row softmax in between; scores are materialised per frame.
 *  - Chunked causal encode/decode with CACHE_T=2 feature caches follow vae.py
 *    (encode_eager / decode_unit) exactly. Optional spatial tiling with
 *    overlap + linear blending (WAN22_VAE_TILE / hip_wan22_vae_set_tiling).
 *
 * Numerics: weights fp32->fp16->bf16 rounded (as the oracle) and stored fp16,
 * convs fp16 x fp16 -> fp32 accumulate, everything else fp32.
 *   WAN22_VAE_RAW_WEIGHTS=1  skip the fp16->bf16 weight rounding
 *   WAN22_VAE_CACHE=<path|0> pre-converted fp16 weight cache (default
 *                            <dir>/model.w22v2.cache, 0 disables)
 *   WAN22_VAE_TILE=<px>[,<overlap px>]  spatial tiling (0 = off)
 */
#include <stddef.h>
#ifndef HIP_WAN22_VAE_H
#define HIP_WAN22_VAE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_wan22_vae hip_wan22_vae;

/* dir contains model.safetensors (see ref/praxis/convert_vae.py). */
hip_wan22_vae *hip_wan22_vae_load(const char *dir, int device_id, int verbose);
void hip_wan22_vae_free(hip_wan22_vae *r);

/* video: float [3][T][H][W] in [-1,1] (or [0,255] after
 * hip_wan22_vae_set_input_range255(r,1)). Any H,W >= 1: the frame is padded
 * to the next multiple of 32 by edge replication (bottom/right) and the
 * returned latent covers the PADDED frame. Frames beyond 1+4*((T-1)/4) are
 * ignored. Returns the POLICY latent [192][Tl][ceil(H/32)][ceil(W/32)]
 * (normalised, bf16-rounded, as encode_eager) in a malloc'd buffer (free()).
 * Returns 0 on success, <0 on error (see hip_wan22_vae_last_error). */
int hip_wan22_vae_encode(hip_wan22_vae *r, const float *video, int T, int H, int W,
                          float **latent, int *Tl);
/* Same, with uint8 input [3][T][H][W] (0..255 -> v/127.5-1). */
int hip_wan22_vae_encode_u8(hip_wan22_vae *r, const unsigned char *video, int T, int H, int W,
                             float **latent, int *Tl);
/* latent: policy latent [192][Tl][h][w]. Returns frames float [3][To][H][W] in
 * [-1,1] (decode_unit), To = 1+4*(Tl-1), H = 32h, W = 32w, in a malloc'd buffer. */
int hip_wan22_vae_decode(hip_wan22_vae *r, const float *latent, int Tl, int h, int w,
                          float **frames, int *To);
/* As decode, but crops the (padded) output to the top-left Hc x Wc pixels
 * (Hc <= 32h, Wc <= 32w): the inverse of the encode padding. */
int hip_wan22_vae_decode_crop(hip_wan22_vae *r, const float *latent, int Tl, int h, int w,
                               int Hc, int Wc, float **frames, int *To);

/* Options. */
void hip_wan22_vae_set_input_range255(hip_wan22_vae *r, int on); /* float input in [0,255] */
/* Spatial tiling: tile_px / overlap_px are pixel sizes (rounded to multiples
 * of 32; tile_px 0 = untiled, -1 = automatic from the VRAM budget). Tiles are
 * blended with linear ramps over the overlap. Default: automatic. */
void hip_wan22_vae_set_tiling(hip_wan22_vae *r, int tile_px, int overlap_px);
/* Keep the device scratch pool between calls (faster repeated calls, but the
 * peak stays resident). Default 0 = release after each call. */
void hip_wan22_vae_set_persistent(hip_wan22_vae *r, int on);
/* Pre-allocate/warm the kernels for HxW so the first real call is not slower. */
int hip_wan22_vae_warmup(hip_wan22_vae *r, int H, int W);

/* Debug: dump intermediate tensors as <dir>/<name>.npy ([1,C,T,H,W]) on the
 * next encode/decode (names as in ref/praxis/gen_vae_ref.py). NULL disables. */
void hip_wan22_vae_set_tap_dir(hip_wan22_vae *r, const char *dir);
/* Peak device memory allocated by this runner since load (bytes; weights +
 * scratch; excludes the HIP context), and the wall time (ms) of the last
 * encode/decode. */
size_t hip_wan22_vae_peak_bytes(const hip_wan22_vae *r);
void hip_wan22_vae_reset_peak(hip_wan22_vae *r); /* peak := current */
double hip_wan22_vae_last_ms(const hip_wan22_vae *r);
/* Text of the last error (empty if none). */
const char *hip_wan22_vae_last_error(const hip_wan22_vae *r);

#ifdef __cplusplus
}
#endif

#ifdef HIP_WAN22_VAE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "../../common/safetensors.h"

/* ------------------------------------------------------------------ */
/* Device kernels                                                       */
/* ------------------------------------------------------------------ */
#define W22_STR(...) #__VA_ARGS__
static const char *w22_kernel_src = W22_STR(
typedef _Float16 w22_f16x8 __attribute__((ext_vector_type(8)));
typedef float w22_f32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned short w22_f2h(float x) {
    _Float16 h = (_Float16)x;
    return *(unsigned short *)&h;
}
__device__ __forceinline__ float w22_h2f(unsigned short u) {
    return (float)(*(_Float16 *)&u);
}

struct W22ConvArgs {
    const unsigned short *X;   /* fp16 activations, pixel stride lda */
    const unsigned short *Wt;  /* fp16 weights [N][K], row stride ldb */
    const float *bias;
    const float *res;          /* fp32 [M][N] */
    float *Y;                  /* fp32 out (may be null) */
    unsigned short *Y16;       /* fp16 out (may be null) */
    float *ws;                 /* split-K partials [z][M][N] */
    int M, N, K, lda, ldb, ldy, trans;
    int Cin, KH, KW, st, sh, p0, up2, Hi, Wi, Ho, Wo;
    int ktiles, kper;
};

__device__ __forceinline__ void w22_store1(const W22ConvArgs &a, int m, int n, float v) {
    if (a.bias) v += a.bias[n];
    if (a.res) v += a.res[(size_t)m * a.N + n];
    if (!a.trans) {
        if (a.Y) a.Y[(size_t)m * a.ldy + n] = v;
        if (a.Y16) a.Y16[(size_t)m * a.ldy + n] = w22_f2h(v);
    } else {
        a.Y16[(size_t)n * a.ldy + m] = w22_f2h(v);
    }
}

template <int LA, int LB> struct W22AS {
    int kcur, cc, kx, ky, kz; bool dirty;
    int tb[LA], yb[LA], xb[LA]; bool rv[LA];
    const unsigned short *rp[LA]; const unsigned short *bp[LB];
};

/* fetch the next BK=32 slice of the A (im2col gather) and B (weights) tiles into registers */
template <int NT, int LA, int LB>
__device__ __forceinline__ void w22_fetch(const W22ConvArgs &a, W22AS<LA, LB> &s, uint4 *ra, uint4 *rb,
                                          int n0, int rl, int Hv, int Wv)
{
    const bool kvalid = s.kcur < a.K;
    if (s.dirty) {
        _Pragma("unroll")
        for (int i = 0; i < LA; i++) {
            int iy = s.yb[i] + s.ky, ix = s.xb[i] + s.kx;
            bool ok = s.rv[i] && iy >= 0 && iy < Hv && ix >= 0 && ix < Wv;
            const unsigned short *p = 0;
            if (ok) {
                int jy = a.up2 ? (iy >> 1) : iy, jx = a.up2 ? (ix >> 1) : ix;
                long long pix = ((long long)(s.tb[i] + s.kz) * a.Hi + jy) * a.Wi + jx;
                p = a.X + pix * a.lda;
            }
            s.rp[i] = p;
        }
        s.dirty = false;
    }
    _Pragma("unroll")
    for (int i = 0; i < LA; i++) {
        bool ok = kvalid && s.rp[i] != 0;
        ra[i] = ok ? *(const uint4 *)(s.rp[i] + s.cc) : make_uint4(0, 0, 0, 0);
    }
    _Pragma("unroll")
    for (int i = 0; i < LB; i++) {
        int row = rl + (NT / 4) * i;
        bool bok = kvalid && (n0 + row) < a.N;
        rb[i] = bok ? *(const uint4 *)s.bp[i] : make_uint4(0, 0, 0, 0);
        s.bp[i] += 32;
    }
    s.kcur += 32; s.cc += 32;
    while (s.cc >= a.Cin) { s.cc -= a.Cin; s.dirty = true; if (++s.kx == a.KW) { s.kx = 0; if (++s.ky == a.KH) { s.ky = 0; s.kz++; } } }
}

/* Implicit-GEMM convolution on RDNA4 WMMA (f16 x f16 -> f32 accumulate):
 * Y[m][n] = sum_k A[m][k] * Wt[n][k]; row m is output pixel (t,y,x); k = (kt,ky,kx,cin), cin fastest.
 * CTA tile BM x BN x 32, WM x WN waves (wave32, 8 waves), register-prefetched global->LDS staging. */
template <int BM, int BN, int WM, int WN>
__device__ __forceinline__ void w22_conv_impl(const W22ConvArgs &a)
{
    constexpr int NT = WM * WN * 32;
    constexpr int TM = BM / WM, TN = BN / WN;
    constexpr int MI = TM / 16, NI = TN / 16;
    constexpr int LA = BM * 4 / NT, LB = BN * 4 / NT;
    constexpr int LDS = 40;
    __shared__ __align__(16) unsigned short sA[BM * LDS];
    __shared__ __align__(16) unsigned short sB[BN * LDS];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int half = lane >> 4, idx = lane & 15;
    const int wm = wave % WM, wn = wave / WM;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int kt0 = blockIdx.z * a.kper;
    int kt1 = kt0 + a.kper; if (kt1 > a.ktiles) kt1 = a.ktiles;
    const int nk = kt1 - kt0;
    const int chunk = tid & 3, rl = tid >> 2;
    const int Cin = a.Cin, KW = a.KW, KH = a.KH;
    const int HoWo = a.Ho * a.Wo;
    const int Hv = a.up2 ? a.Hi * 2 : a.Hi, Wv = a.up2 ? a.Wi * 2 : a.Wi;

    W22AS<LA, LB> as;
    _Pragma("unroll")
    for (int i = 0; i < LA; i++) {
        int m = m0 + rl + (NT / 4) * i;
        as.rv[i] = m < a.M;
        int mm = as.rv[i] ? m : 0;
        int t = mm / HoWo; int rem = mm - t * HoWo;
        int y = rem / a.Wo; int x = rem - y * a.Wo;
        as.tb[i] = t * a.st; as.yb[i] = y * a.sh - a.p0; as.xb[i] = x * a.sh - a.p0;
    }
    as.kcur = kt0 * 32 + chunk * 8;
    {
        int tap = as.kcur / Cin;
        as.cc = as.kcur - tap * Cin;
        as.kx = tap % KW; as.ky = (tap / KW) % KH; as.kz = tap / (KW * KH);
    }
    as.dirty = true;
    _Pragma("unroll")
    for (int i = 0; i < LB; i++) as.bp[i] = a.Wt + (size_t)(n0 + rl + (NT / 4) * i) * a.ldb + as.kcur;

    w22_f32x8 acc[MI][NI];
    _Pragma("unroll")
    for (int i = 0; i < MI; i++)
        _Pragma("unroll")
        for (int j = 0; j < NI; j++)
            _Pragma("unroll")
            for (int q = 0; q < 8; q++) acc[i][j][q] = 0.f;

    uint4 ra[LA], rb[LB];
    if (nk > 0) w22_fetch<NT, LA, LB>(a, as, ra, rb, n0, rl, Hv, Wv);
    for (int it = 0; it < nk; it++) {
        _Pragma("unroll")
        for (int i = 0; i < LA; i++) *(uint4 *)&sA[(rl + (NT / 4) * i) * LDS + chunk * 8] = ra[i];
        _Pragma("unroll")
        for (int i = 0; i < LB; i++) *(uint4 *)&sB[(rl + (NT / 4) * i) * LDS + chunk * 8] = rb[i];
        __syncthreads();
        if (it + 1 < nk) w22_fetch<NT, LA, LB>(a, as, ra, rb, n0, rl, Hv, Wv);
        _Pragma("unroll")
        for (int kk = 0; kk < 32; kk += 16) {
            w22_f16x8 af[MI], bf[NI];
            _Pragma("unroll")
            for (int mi = 0; mi < MI; mi++)
                af[mi] = *(const w22_f16x8 *)&sA[(wm * TM + 16 * mi + idx) * LDS + kk + half * 8];
            _Pragma("unroll")
            for (int ni = 0; ni < NI; ni++)
                bf[ni] = *(const w22_f16x8 *)&sB[(wn * TN + 16 * ni + idx) * LDS + kk + half * 8];
            _Pragma("unroll")
            for (int mi = 0; mi < MI; mi++)
                _Pragma("unroll")
                for (int ni = 0; ni < NI; ni++)
                    acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[mi], bf[ni], acc[mi][ni]);
        }
        __syncthreads();
    }
    const bool split = gridDim.z > 1;
    _Pragma("unroll")
    for (int mi = 0; mi < MI; mi++)
        _Pragma("unroll")
        for (int ni = 0; ni < NI; ni++) {
            int n = n0 + wn * TN + 16 * ni + idx;
            if (n >= a.N) continue;
            _Pragma("unroll")
            for (int i = 0; i < 8; i++) {
                int m = m0 + wm * TM + 16 * mi + half * 8 + i;
                if (m >= a.M) continue;
                float v = acc[mi][ni][i];
                if (split) a.ws[((size_t)blockIdx.z * a.M + m) * a.N + n] = v;
                else w22_store1(a, m, n, v);
            }
        }
}

extern "C" {

__global__ void __launch_bounds__(256) w22_conv_128x128(const W22ConvArgs a) { w22_conv_impl<128, 128, 2, 4>(a); }
__global__ void __launch_bounds__(256) w22_conv_64x128(const W22ConvArgs a) { w22_conv_impl<64, 128, 2, 4>(a); }
__global__ void __launch_bounds__(256) w22_conv_64x64(const W22ConvArgs a) { w22_conv_impl<64, 64, 4, 2>(a); }

/* split-K reduction + epilogue: one thread per (m, n). */
__global__ void w22_splitk_reduce(const W22ConvArgs a, int nsplit)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)a.M * a.N;
    if (i >= total) return;
    int m = (int)(i / a.N), n = (int)(i - (long long)m * a.N);
    float v = 0.f;
    for (int z = 0; z < nsplit; z++) v += a.ws[((size_t)z * a.M + m) * a.N + n];
    w22_store1(a, m, n, v);
}

/* fp32 -> fp16 copy (n4 = number of float4). */
__global__ void w22_f32_to_f16(const float *src, unsigned short *dst, long long n4)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = ((const float4 *)src)[i];
    ushort4 h;
    h.x = w22_f2h(v.x); h.y = w22_f2h(v.y); h.z = w22_f2h(v.z); h.w = w22_f2h(v.w);
    ((ushort4 *)dst)[i] = h;
}

/* channel RMS norm per pixel: x / max(||x||,1e-12) * sqrt(C) * gamma, optional
 * SiLU, fp16 output. One wave32 per pixel; C <= 1024. */
__global__ void w22_rmsnorm(const float *x, const float *gamma, unsigned short *y,
                            long long npix, int C, float scale, int silu)
{
    long long pix = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (pix >= npix) return;
    const float4 *xp = (const float4 *)(x + pix * C);
    int C4 = C >> 2;
    float4 v[8];
    float ss = 0.f;
    _Pragma("unroll")
    for (int j = 0; j < 8; j++) {
        int c = lane + j * 32;
        if (c < C4) {
            v[j] = xp[c];
            ss += v[j].x * v[j].x + v[j].y * v[j].y + v[j].z * v[j].z + v[j].w * v[j].w;
        }
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
    float nrm = sqrtf(ss);
    if (nrm < 1e-12f) nrm = 1e-12f;
    ushort4 *yp = (ushort4 *)(y + pix * C);
    _Pragma("unroll")
    for (int j = 0; j < 8; j++) {
        int c = lane + j * 32;
        if (c < C4) {
            float4 g = ((const float4 *)gamma)[c];
            float a0 = v[j].x / nrm * scale * g.x;
            float a1 = v[j].y / nrm * scale * g.y;
            float a2 = v[j].z / nrm * scale * g.z;
            float a3 = v[j].w / nrm * scale * g.w;
            if (silu) {
                a0 = a0 / (1.f + expf(-a0));
                a1 = a1 / (1.f + expf(-a1));
                a2 = a2 / (1.f + expf(-a2));
                a3 = a3 / (1.f + expf(-a3));
            }
            ushort4 h;
            h.x = w22_f2h(a0); h.y = w22_f2h(a1); h.z = w22_f2h(a2); h.w = w22_f2h(a3);
            yp[c] = h;
        }
    }
}

/* acc += AvgDown3D(x). x: [T][H][W][C]; acc: [To][Ho][Wo][Co]. */
__global__ void w22_avg_down_add(const float *x, float *acc, int T, int H, int W, int C,
                                 int To, int Ho, int Wo, int Co,
                                 int ft, int fs, int gs, int padt)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)To * Ho * Wo * Co;
    if (idx >= total) return;
    int o = (int)(idx % Co);
    long long pix = idx / Co;
    int wo = (int)(pix % Wo);
    int ho = (int)((pix / Wo) % Ho);
    int to = (int)(pix / ((long long)Wo * Ho));
    int factor = ft * fs * fs;
    float s = 0.f;
    for (int g = 0; g < gs; g++) {
        int ci = o * gs + g;
        int c = ci / factor;
        int rr = ci - c * factor;
        int i = rr / (fs * fs);
        int j = (rr / fs) % fs;
        int k = rr % fs;
        int tt = to * ft + i - padt;
        if (tt >= 0 && tt < T)
            s += x[(((long long)tt * H + ho * fs + j) * W + wo * fs + k) * C + c];
    }
    acc[idx] += s / (float)gs;
}

/* acc += DupUp3D(x). x: [T][H][W][Cin]; acc: [To][H*fs][W*fs][Co]. */
__global__ void w22_dup_up_add(const float *x, float *acc, int T, int H, int W, int Cin,
                               int Co, int ft, int fs, int repeats, int drop)
{
    int To = T * ft - drop, Ho = H * fs, Wo = W * fs;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)To * Ho * Wo * Co;
    if (idx >= total) return;
    int o = (int)(idx % Co);
    long long pix = idx / Co;
    int wo = (int)(pix % Wo);
    int ho = (int)((pix / Wo) % Ho);
    int to = (int)(pix / ((long long)Wo * Ho));
    int t = to + drop;
    int tin = t / ft, i = t % ft;
    int j = ho % fs, k = wo % fs;
    int n = ((o * ft + i) * fs + j) * fs + k;
    int cin = n / repeats;
    acc[idx] += x[(((long long)tin * H + ho / fs) * W + wo / fs) * Cin + cin];
}

/* in: [T][HW][2C] fp32 -> out: [2T][HW][C] fp16; frame 2t+j takes channels [j*C, j*C+C). */
__global__ void w22_tconv_interleave(const float *in, unsigned short *out, int T, long long HW, int C)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = 2LL * T * HW * C;
    if (idx >= total) return;
    int c = (int)(idx % C);
    long long q = idx / C;
    long long pix = q % HW;
    int tf = (int)(q / HW);
    int t = tf >> 1, j = tf & 1;
    out[idx] = w22_f2h(in[((long long)t * HW + pix) * 2 * C + j * C + c]);
}

/* video [3][Tv][H][W] -> dst fp16 [nt][H/2][W/2][16], channel = c*4 + pw*2 + ph (12..15 = 0). */
__global__ void w22_patchify(const float *video, unsigned short *dst, int Tv, int H, int W,
                             int t0, int nt)
{
    int H2 = H >> 1, W2 = W >> 1;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nt * H2 * W2 * 16;
    if (idx >= total) return;
    int ch = (int)(idx & 15);
    long long pix = idx >> 4;
    int x = (int)(pix % W2);
    int y = (int)((pix / W2) % H2);
    int t = (int)(pix / ((long long)W2 * H2));
    unsigned short o = 0;
    if (ch < 12) {
        int c = ch >> 2, pw = (ch >> 1) & 1, ph = ch & 1;
        o = w22_f2h(video[(((long long)c * Tv + t0 + t) * H + 2 * y + ph) * W + 2 * x + pw]);
    }
    dst[idx] = o;
}

/* src [nt][h2][w2][12] -> dst [3][To][H][W] (clamped to [-1,1]) at frame tbase. */
__global__ void w22_unpatchify(const float *src, float *dst, int nt, int h2, int w2,
                               int To, int tbase)
{
    int H = h2 * 2, W = w2 * 2;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nt * h2 * w2 * 12;
    if (idx >= total) return;
    int ch = (int)(idx % 12);
    long long pix = idx / 12;
    int x = (int)(pix % w2);
    int y = (int)((pix / w2) % h2);
    int t = (int)(pix / ((long long)w2 * h2));
    int c = ch >> 2, pw = (ch >> 1) & 1, ph = ch & 1;
    float v = src[idx];
    v = fminf(fmaxf(v, -1.f), 1.f);
    dst[(((long long)c * To + tbase + t) * H + 2 * y + ph) * W + 2 * x + pw] = v;
}

/* row softmax of scale*S (fp32, row stride ld) -> P fp16 (same stride); columns
 * [N, ld) of P are zero. One block (256 thr) per row. */
__global__ void w22_softmax_rows(const float *S, unsigned short *P, int N, int ld, float scale)
{
    __shared__ float red[32];
    const float *row = S + (long long)blockIdx.x * ld;
    unsigned short *prow = P + (long long)blockIdx.x * ld;
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    float m = -3.4e38f;
    for (int i = tid; i < N; i += blockDim.x) m = fmaxf(m, row[i] * scale);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, red[i]);
    __syncthreads();
    float s = 0.f;
    for (int i = tid; i < N; i += blockDim.x) s += expf(row[i] * scale - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o);
    if (lane == 0) red[wid] = s;
    __syncthreads();
    s = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) s += red[i];
    float inv = 1.f / s;
    for (int i = tid; i < ld; i += blockDim.x)
        prow[i] = i < N ? w22_f2h(expf(row[i] * scale - m) * inv) : (unsigned short)0;
}

}
);


/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* Host structures                                                      */
/* ------------------------------------------------------------------ */
typedef unsigned long long w22_dp;

static void w22_h2d(w22_dp dst, const void *src, size_t n) { hipMemcpy((void *)(uintptr_t)dst, src, n, hipMemcpyHostToDevice); }
static void w22_d2h(void *dst, w22_dp src, size_t n) { hipMemcpy(dst, (void *)(uintptr_t)src, n, hipMemcpyDeviceToHost); }
static void w22_d2d(w22_dp dst, w22_dp src, size_t n) { hipMemcpy((void *)(uintptr_t)dst, (void *)(uintptr_t)src, n, hipMemcpyDeviceToDevice); }
static void w22_set8(w22_dp dst, int v, size_t n) { hipMemset((void *)(uintptr_t)dst, v, n); }
static void w22_dfree(w22_dp p) { if (p) hipFree((void *)(uintptr_t)p); }

typedef struct {
    int Cin, Cout, kt, kh, kw;   /* Cin is the (8-aligned) stored channel count */
    int st, sh;        /* temporal / spatial stride */
    int p0, pt;        /* spatial leading pad / total pad */
    int up2;           /* virtual 2x nearest upsample before the conv */
    int cf;            /* causal cache frames (0 = none) */
    w22_dp w;     /* [Cout][kt][kh][kw][Cin] fp16 */
    w22_dp bias;  /* fp32 [Cout] */
    w22_dp cache; /* fp16 [cf][H][W][Cin], (re)allocated per spatial size */
    size_t cache_bytes;
    int cache_valid;
} w22_conv;

typedef struct { w22_dp gamma; int C; } w22_norm;
typedef struct { w22_norm n1, n2; w22_conv c1, c2, sc; int has_sc, in, out; } w22_res;
typedef struct { w22_norm norm; w22_conv qkv, proj; int C; } w22_attn;
typedef struct { w22_conv sp, tc; int has_tc; int dim; } w22_rs;
typedef struct { w22_res res[3]; int nres; w22_rs rs; int has_rs; int ft, fs, in, out; } w22_blk;

typedef struct { char name[112]; const unsigned char *p; size_t n; } w22_crec;
typedef struct {
    int mode;                 /* 0 off, 1 read (mmap), 2 write */
    unsigned char *map; size_t maplen;
    FILE *fw; size_t wpos;
    w22_crec *recs; int nrec, caprec;
    char path[1100], tmp[1200];
    uint64_t src_size, src_mtime;
    uint32_t flags;
} w22_wcache;

struct hip_wan22_vae {
    int dev;
    hipModule_t mod;
    hipFunction_t f_conv[3], f_splitk, f_f2h, f_norm, f_avg, f_dup, f_tint, f_patch, f_unpatch, f_softmax;
    int verbose, raw, nsm;

    w22_conv e_conv1, e_head, e_mu;
    w22_blk e_blk[4];
    w22_res e_mid[2];
    w22_attn e_attn;
    w22_norm e_hnorm;

    w22_conv d_conv2, d_conv1, d_head;
    w22_res d_mid[2];
    w22_attn d_attn;
    w22_blk d_blk[4];
    w22_norm d_hnorm;

    w22_conv *all[512];
    int nall;
    w22_dp gam[128];
    int ngam;

    /* memory pool */
    struct { w22_dp p; size_t sz; int used; } pool[1024];
    int npool;
    size_t cur_bytes, peak_bytes;
    int failed;
    char err[256];
    double last_ms;
    int persistent;
    double flops;
    size_t wbytes;

    /* options */
    int range255, tile_px, overlap_px;

    /* taps */
    char tap_dir[512];
    struct { char name[32]; float *buf; size_t n, cap; int T, H, W, C; } taps[40];
    int ntaps;
};

static double w22_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void w22_seterr(hip_wan22_vae *r, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (!r->failed || !r->err[0]) vsnprintf(r->err, sizeof r->err, fmt, ap);
    va_end(ap);
    r->failed = 1;
    fprintf(stderr, "wan22_vae: %s\n", r->err);
}

/* ---- memory (all device allocations are accounted for) ---- */
static w22_dp w22_malloc(hip_wan22_vae *r, size_t bytes) {
    w22_dp p = 0;
    void *vp = NULL;
    if (hipMalloc(&vp, bytes) != hipSuccess) return 0;
    p = (w22_dp)(uintptr_t)vp;
    r->cur_bytes += bytes;
    if (r->cur_bytes > r->peak_bytes) r->peak_bytes = r->cur_bytes;
    return p;
}
/* free with known size (we track sizes in the pool or by caller) */
static void w22_mfree(hip_wan22_vae *r, w22_dp p, size_t bytes) {
    if (!p) return;
    w22_dfree(p);
    r->cur_bytes = r->cur_bytes > bytes ? r->cur_bytes - bytes : 0;
}

static void w22_pool_release(hip_wan22_vae *r) {
    for (int i = 0; i < r->npool; i++)
        if (r->pool[i].p && !r->pool[i].used) { w22_mfree(r, r->pool[i].p, r->pool[i].sz); r->pool[i].p = 0; r->pool[i].sz = 0; }
}

static w22_dp w22_alloc(hip_wan22_vae *r, size_t bytes) {
    if (r->failed) return 0;
    bytes = (bytes + 511) & ~(size_t)511;
    int best = -1;
    for (int i = 0; i < r->npool; i++)
        if (r->pool[i].p && !r->pool[i].used && r->pool[i].sz >= bytes && r->pool[i].sz <= bytes + bytes / 4 &&
            (best < 0 || r->pool[i].sz < r->pool[best].sz))
            best = i;
    if (best >= 0) { r->pool[best].used = 1; return r->pool[best].p; }
    w22_dp p = w22_malloc(r, bytes);
    if (!p) {
        w22_pool_release(r);  /* flush free blocks and retry */
        p = w22_malloc(r, bytes);
        if (!p) { w22_seterr(r, "out of device memory (%.1f MB requested, %.2f GB in use)", bytes / 1048576.0, r->cur_bytes / 1073741824.0); return 0; }
    }
    int slot = -1;
    for (int i = 0; i < r->npool; i++) if (!r->pool[i].p) { slot = i; break; }
    if (slot < 0) {
        if (r->npool >= 1024) { w22_mfree(r, p, bytes); w22_seterr(r, "scratch pool full"); return 0; }
        slot = r->npool++;
    }
    r->pool[slot].p = p; r->pool[slot].sz = bytes; r->pool[slot].used = 1;
    return p;
}

static void w22_free(hip_wan22_vae *r, w22_dp p) {
    if (!p) return;
    for (int i = 0; i < r->npool; i++)
        if (r->pool[i].p == p) { r->pool[i].used = 0; return; }
}

static void w22_pool_reset(hip_wan22_vae *r) {  /* error recovery: everything unused */
    for (int i = 0; i < r->npool; i++) r->pool[i].used = 0;
}

#define F4(x) ((size_t)(x) * sizeof(float))
#define F2(x) ((size_t)(x) * 2)

/* ---- launch helpers ---- */
static int w22_lk(hip_wan22_vae *r, hipFunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, void **args) {
    if (r->failed) return -1;
    hipError_t e = hipModuleLaunchKernel(f, gx, gy, gz, bx, 1, 1, 0, NULL, args, NULL);
    if (e != hipSuccess) { w22_seterr(r, "kernel launch failed (%d)", (int)e); return -1; }
    return 0;
}
static int w22_launch1d(hip_wan22_vae *r, hipFunction_t f, long long n, void **args) {
    long long blocks = (n + 255) / 256;
    return w22_lk(r, f, (unsigned)blocks, 1, 1, 256, args);
}

/* fp32 -> fp16 conversion of n elements (n % 4 == 0). */
static int w22_to_f16(hip_wan22_vae *r, w22_dp src, w22_dp dst, long long n) {
    long long n4 = n / 4;
    void *args[] = { &src, &dst, &n4 };
    return w22_launch1d(r, r->f_f2h, n4, args);
}

/* ---- taps ---- */
static void w22_tap(hip_wan22_vae *r, const char *name, w22_dp d, int T, int H, int W, int C) {
    if (!r->tap_dir[0]) return;
    int i;
    for (i = 0; i < r->ntaps; i++) if (!strcmp(r->taps[i].name, name)) break;
    if (i == r->ntaps) {
        if (i >= 40) return;
        memset(&r->taps[i], 0, sizeof(r->taps[i]));
        snprintf(r->taps[i].name, 32, "%s", name);
        r->ntaps++;
    }
    size_t n = (size_t)T * H * W * C;
    if (r->taps[i].n + n > r->taps[i].cap) {
        r->taps[i].cap = (r->taps[i].n + n) * 2;
        r->taps[i].buf = (float *)realloc(r->taps[i].buf, F4(r->taps[i].cap));
    }
    hipDeviceSynchronize();
    w22_d2h(r->taps[i].buf + r->taps[i].n, d, F4(n));
    r->taps[i].n += n; r->taps[i].T += T; r->taps[i].H = H; r->taps[i].W = W; r->taps[i].C = C;
}

static void w22_write_npy5(const char *path, const float *d, int n0, int n1, int n2, int n3, int n4) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "wan22_vae: cannot write %s\n", path); return; }
    char hdr[256];
    int hl = snprintf(hdr, sizeof hdr,
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d, %d, %d), }", n0, n1, n2, n3, n4);
    int tot = 10 + hl + 1;
    int pad = (64 - tot % 64) % 64;
    unsigned short hlen = (unsigned short)(hl + pad + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);
    fwrite(hdr, 1, hl, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(d, sizeof(float), (size_t)n0 * n1 * n2 * n3 * n4, f);
    fclose(f);
}

static void w22_tap_flush(hip_wan22_vae *r) {
    for (int i = 0; i < r->ntaps; i++) {
        int T = r->taps[i].T, H = r->taps[i].H, W = r->taps[i].W, C = r->taps[i].C;
        size_t HW = (size_t)H * W;
        float *o = (float *)malloc(F4((size_t)C * T * HW));
        const float *s = r->taps[i].buf;
        for (int t = 0; t < T; t++)
            for (size_t p = 0; p < HW; p++)
                for (int c = 0; c < C; c++)
                    o[((size_t)c * T + t) * HW + p] = s[((size_t)t * HW + p) * C + c];
        char path[640];
        snprintf(path, sizeof path, "%s/%s.npy", r->tap_dir, r->taps[i].name);
        w22_write_npy5(path, o, 1, C, T, H, W);
        free(o);
        free(r->taps[i].buf);
    }
    r->ntaps = 0;
}

/* ---- conv core (implicit GEMM) ---- */
typedef struct {
    w22_dp X, Wt, bias, res, Y, Y16;
    int M, N, K, lda, ldb, ldy, trans;
    int Cin, KH, KW, st, sh, p0, up2, Hi, Wi, Ho, Wo;
} w22_gj;

static int w22_pick_cfg(hip_wan22_vae *r, int M, int N) {
    const char *e = getenv("W22_CFG");
    if (e) return atoi(e);
    (void)M;
    return N % 128 != 0 ? 2 : 0;  /* 128x128 for the wide convs, 64x64 for narrow ones (measured fastest on gfx1201) */
}

static int w22_gemm_launch(hip_wan22_vae *r, const w22_gj *g) {
    typedef struct {
        w22_dp X, Wt, bias, res, Y, Y16, ws;
        int M, N, K, lda, ldb, ldy, trans;
        int Cin, KH, KW, st, sh, p0, up2, Hi, Wi, Ho, Wo;
        int ktiles, kper;
    } args_t;
    args_t a;
    memset(&a, 0, sizeof a);
    a.X = g->X; a.Wt = g->Wt; a.bias = g->bias; a.res = g->res; a.Y = g->Y; a.Y16 = g->Y16;
    a.M = g->M; a.N = g->N; a.K = g->K; a.lda = g->lda; a.ldb = g->ldb; a.ldy = g->ldy; a.trans = g->trans;
    a.Cin = g->Cin; a.KH = g->KH; a.KW = g->KW; a.st = g->st; a.sh = g->sh; a.p0 = g->p0; a.up2 = g->up2;
    a.Hi = g->Hi; a.Wi = g->Wi; a.Ho = g->Ho; a.Wo = g->Wo;
    r->flops += 2.0 * g->M * (double)g->N * g->K;
    int ktiles = (g->K + 31) / 32;
    static const int cbm[3] = { 128, 64, 64 }, cbn[3] = { 128, 128, 64 }, cth[3] = { 256, 256, 256 };
    int cfg = w22_pick_cfg(r, g->M, g->N);
    int bm = cbm[cfg], bn = cbn[cfg];
    int mt = (g->M + bm - 1) / bm, nt = (g->N + bn - 1) / bn;
    long long tiles = (long long)mt * nt;
    int split = 1;
    if (tiles < 2LL * r->nsm) {
        split = (int)((2LL * r->nsm + tiles - 1) / tiles);
        if (split > ktiles / 6) split = ktiles / 6;
        if (split > 32) split = 32;
        if (split < 1) split = 1;
    }
    a.kper = (ktiles + split - 1) / split;
    split = (ktiles + a.kper - 1) / a.kper;
    a.ktiles = ktiles;
    w22_dp ws = 0;
    if (split > 1) {
        ws = w22_alloc(r, F4((size_t)split * g->M * g->N));
        if (!ws) return -1;
        a.ws = ws;
    }
    void *args[] = { &a };
    if (r->failed) return -1;
    hipError_t e = hipModuleLaunchKernel(r->f_conv[cfg], mt, nt, split, cth[cfg], 1, 1, 0, NULL, args, NULL);
    if (e != hipSuccess) { w22_seterr(r, "conv launch failed (%d)", (int)e); return -1; }
    if (split > 1) {
        void *args2[] = { &a, &split };
        if (w22_launch1d(r, r->f_splitk, (long long)g->M * g->N, args2)) return -1;
        w22_free(r, ws);
    }
    return 0;
}

/* Convolve fp16 `in` ([Tin][Hi][Wi][Cin], includes any causal halo) into fresh
 * [Tout][Ho][Wo][Cout] fp32 (and/or fp16) buffers, adding bias and optional residual. */
static int w22_conv_run(hip_wan22_vae *r, w22_conv *c, w22_dp in, int Tin, int Hi, int Wi,
                        w22_dp res, w22_dp *o32, w22_dp *o16, int *oT, int *oH, int *oW)
{
    int Hv = c->up2 ? Hi * 2 : Hi, Wv = c->up2 ? Wi * 2 : Wi;
    int Ho = (Hv + c->pt - c->kh) / c->sh + 1;
    int Wo = (Wv + c->pt - c->kw) / c->sh + 1;
    int To = (Tin - c->kt) / c->st + 1;
    long long rows = (long long)To * Ho * Wo;
    w22_gj g;
    memset(&g, 0, sizeof g);
    w22_dp y32 = 0, y16 = 0;
    if (o32) { y32 = w22_alloc(r, F4(rows * c->Cout)); if (!y32) return -1; }
    if (o16) { y16 = w22_alloc(r, F2(rows * c->Cout)); if (!y16) return -1; }
    g.X = in; g.Wt = c->w; g.bias = c->bias; g.res = res; g.Y = y32; g.Y16 = y16;
    g.M = (int)rows; g.N = c->Cout; g.K = c->kt * c->kh * c->kw * c->Cin;
    g.lda = c->Cin; g.ldb = g.K; g.ldy = c->Cout;
    g.Cin = c->Cin; g.KH = c->kh; g.KW = c->kw; g.st = c->st; g.sh = c->sh; g.p0 = c->p0; g.up2 = c->up2;
    g.Hi = Hi; g.Wi = Wi; g.Ho = Ho; g.Wo = Wo;
    if (w22_gemm_launch(r, &g)) return -1;
    if (o32) *o32 = y32;
    if (o16) *o16 = y16;
    *oT = To; *oH = Ho; *oW = Wo;
    return 0;
}

/* Non-causal (no cache) conv on fp16 input [T][H][W][Cin] -> fp32. */
static w22_dp w22_conv_plain(hip_wan22_vae *r, w22_conv *c, w22_dp in16, int T, int *H, int *W, w22_dp res) {
    int oT, oH, oW;
    w22_dp o = 0;
    if (w22_conv_run(r, c, in16, T, *H, *W, res, &o, NULL, &oT, &oH, &oW)) return 0;
    *H = oH; *W = oW;
    return o;
}

/* Same, but the input is the fp32 tensor x (converted on the fly into a temp). */
static w22_dp w22_conv_plain32(hip_wan22_vae *r, w22_conv *c, w22_dp x, int T, int *H, int *W, w22_dp res) {
    w22_dp x16 = w22_alloc(r, F2((size_t)T * (*H) * (*W) * c->Cin));
    if (!x16) return 0;
    if (w22_to_f16(r, x, x16, (long long)T * (*H) * (*W) * c->Cin)) return 0;
    w22_dp o = w22_conv_plain(r, c, x16, T, H, W, res);
    w22_free(r, x16);
    return o;
}

/* Causal conv with halo: allocate fp16 [cf+T][H][W][Cin], preload cache (or zeros). */
static w22_dp w22_halo_begin(hip_wan22_vae *r, w22_conv *c, int T, int H, int W) {
    size_t fe = (size_t)H * W * c->Cin;
    w22_dp halo = w22_alloc(r, F2((size_t)(c->cf + T) * fe));
    if (!halo) return 0;
    size_t need = F2((size_t)c->cf * fe);
    if (c->cache && c->cache_bytes != need) { w22_mfree(r, c->cache, c->cache_bytes); c->cache = 0; }
    if (!c->cache) {
        c->cache = w22_malloc(r, need);
        if (!c->cache) { w22_seterr(r, "out of device memory for a causal cache (%.1f MB)", need / 1048576.0); return 0; }
        c->cache_bytes = need;
        w22_set8(c->cache, 0, need);
        c->cache_valid = 0;
    }
    w22_d2d(halo, c->cache, c->cache_bytes);
    return halo;
}

/* Run the conv on a prepared halo (frames cf..cf+T-1 already written), refresh
 * the cache with its last cf frames, release the halo. */
static w22_dp w22_conv_halo(hip_wan22_vae *r, w22_conv *c, w22_dp halo, int T, int *H, int *W, w22_dp res) {
    int oT, oH, oW;
    size_t fe = (size_t)(*H) * (*W) * c->Cin;
    w22_dp o = 0;
    if (w22_conv_run(r, c, halo, c->cf + T, *H, *W, res, &o, NULL, &oT, &oH, &oW)) return 0;
    w22_d2d(c->cache, halo + F2((size_t)T * fe), c->cache_bytes);
    c->cache_valid = 1;
    w22_free(r, halo);
    *H = oH; *W = oW;
    return o;
}

static int w22_norm_run(hip_wan22_vae *r, w22_norm *n, w22_dp x, w22_dp y16, long long npix, int silu) {
    float scale = sqrtf((float)n->C);
    int C = n->C;
    void *args[] = { &x, &n->gamma, &y16, &npix, &C, &scale, &silu };
    return w22_lk(r, r->f_norm, (unsigned)((npix + 7) / 8), 1, 1, 256, args);
}

/* y = silu(rmsnorm(x)) written (fp16) into the halo of conv c, then conv. */
static w22_dp w22_norm_conv(hip_wan22_vae *r, w22_norm *n, w22_conv *c, w22_dp x, int T, int *H, int *W, w22_dp res) {
    w22_dp halo = w22_halo_begin(r, c, T, *H, *W);
    if (!halo) return 0;
    size_t fe = (size_t)(*H) * (*W) * c->Cin;
    if (w22_norm_run(r, n, x, halo + F2((size_t)c->cf * fe), (long long)T * (*H) * (*W), 1)) return 0;
    return w22_conv_halo(r, c, halo, T, H, W, res);
}

static w22_dp w22_resblock(hip_wan22_vae *r, w22_res *b, w22_dp x, int T, int *H, int *W) {
    w22_dp h = x;
    int h1 = *H, w1 = *W;
    if (b->has_sc) { h = w22_conv_plain32(r, &b->sc, x, T, &h1, &w1, 0); if (!h) return 0; }
    w22_dp a = w22_norm_conv(r, &b->n1, &b->c1, x, T, H, W, 0);
    if (!a) return 0;
    w22_dp o = w22_norm_conv(r, &b->n2, &b->c2, a, T, H, W, h);
    w22_free(r, a);
    if (b->has_sc) w22_free(r, h);
    return o;
}

/* Single-head self attention per frame (vae.py AttentionBlock): out = x + proj(softmax(q k^T / sqrt(C)) v).
 * q,k,v are fp16 (qkv epilogue; v written transposed), S = qk^T fp32, P fp16, PV fp32-accumulated. */
static w22_dp w22_attnblock(hip_wan22_vae *r, w22_attn *a, w22_dp x, int T, int H, int W) {
    int C = a->C, N = H * W;
    int ldP = (N + 7) & ~7;
    w22_dp xn = w22_alloc(r, F2((size_t)T * N * C));
    w22_dp q = w22_alloc(r, F2((size_t)T * N * C));
    w22_dp k = w22_alloc(r, F2((size_t)T * N * C));
    w22_dp o = w22_alloc(r, F2((size_t)T * N * C));
    w22_dp vt = w22_alloc(r, F2((size_t)C * ldP));
    w22_dp S = w22_alloc(r, F4((size_t)N * ldP));
    w22_dp P = w22_alloc(r, F2((size_t)N * ldP));
    if (!xn || !q || !k || !o || !vt || !S || !P) return 0;
    if (w22_norm_run(r, &a->norm, x, xn, (long long)T * N, 0)) return 0;
    w22_set8(vt, 0, F2((size_t)C * ldP));
    float sc = 1.0f / sqrtf((float)C);
    for (int f = 0; f < T; f++) {
        w22_gj g;
        for (int which = 0; which < 3; which++) {
            memset(&g, 0, sizeof g);
            g.X = xn + F2((size_t)f * N * C);
            g.Wt = a->qkv.w + F2((size_t)which * C * C);
            g.bias = a->qkv.bias + F4((size_t)which * C);
            g.M = N; g.N = C; g.K = C; g.lda = C; g.ldb = C; g.Cin = C;
            g.KH = g.KW = 1; g.st = g.sh = 1; g.Hi = 1; g.Wi = N; g.Ho = 1; g.Wo = N;
            if (which == 0) { g.Y16 = q + F2((size_t)f * N * C); g.ldy = C; }
            else if (which == 1) { g.Y16 = k + F2((size_t)f * N * C); g.ldy = C; }
            else { g.Y16 = vt; g.ldy = ldP; g.trans = 1; }
            if (w22_gemm_launch(r, &g)) return 0;
        }
        /* S = q k^T */
        memset(&g, 0, sizeof g);
        g.X = q + F2((size_t)f * N * C); g.Wt = k + F2((size_t)f * N * C); g.Y = S;
        g.M = N; g.N = N; g.K = C; g.lda = C; g.ldb = C; g.ldy = ldP; g.Cin = C;
        g.KH = g.KW = 1; g.st = g.sh = 1; g.Hi = 1; g.Wi = N; g.Ho = 1; g.Wo = N;
        if (w22_gemm_launch(r, &g)) return 0;
        {
            void *args[] = { &S, &P, &N, &ldP, &sc };
            if (w22_lk(r, r->f_softmax, N, 1, 1, 256, args)) return 0;
        }
        /* O = P v : A = P [N][ldP], W = v^T [C][ldP] */
        memset(&g, 0, sizeof g);
        g.X = P; g.Wt = vt; g.Y16 = o + F2((size_t)f * N * C);
        g.M = N; g.N = C; g.K = ldP; g.lda = ldP; g.ldb = ldP; g.ldy = C; g.Cin = ldP;
        g.KH = g.KW = 1; g.st = g.sh = 1; g.Hi = 1; g.Wi = N; g.Ho = 1; g.Wo = N;
        if (w22_gemm_launch(r, &g)) return 0;
    }
    w22_free(r, xn); w22_free(r, q); w22_free(r, k); w22_free(r, vt); w22_free(r, S); w22_free(r, P);
    int h1 = H, w1 = W;
    w22_dp out = w22_conv_plain(r, &a->proj, o, T, &h1, &w1, x);
    w22_free(r, o);
    return out;
}

/* ---- persistent fp16 weight cache ---- */
static void w22_cache_close(w22_wcache *c) {
    if (c->map) munmap(c->map, c->maplen);
    if (c->fw) fclose(c->fw);
    free(c->recs);
    memset(c, 0, sizeof *c);
}

static void w22_cache_add(w22_wcache *c, const char *name, const unsigned char *p, size_t n) {
    if (c->nrec == c->caprec) {
        c->caprec = c->caprec ? c->caprec * 2 : 512;
        c->recs = (w22_crec *)realloc(c->recs, c->caprec * sizeof(w22_crec));
    }
    snprintf(c->recs[c->nrec].name, sizeof c->recs[0].name, "%s", name);
    c->recs[c->nrec].p = p; c->recs[c->nrec].n = n;
    c->nrec++;
}

static void w22_cache_open(w22_wcache *c, const char *dir, int raw) {
    memset(c, 0, sizeof *c);
    const char *env = getenv("WAN22_VAE_CACHE");
    if (env && !strcmp(env, "0")) return;
    char src[1100];
    snprintf(src, sizeof src, "%s/model.safetensors", dir);
    struct stat st;
    if (stat(src, &st) != 0) return;
    c->src_size = (uint64_t)st.st_size;
    c->src_mtime = (uint64_t)st.st_mtime;
    c->flags = (raw ? 1u : 0u) | 0x100u;  /* 0x100 = layout version 2 */
    if (env && env[0]) snprintf(c->path, sizeof c->path, "%s", env);
    else snprintf(c->path, sizeof c->path, "%s/model.w22v2%s.cache", dir, raw ? ".raw" : "");
    int fd = open(c->path, O_RDONLY);
    if (fd >= 0) {
        struct stat cs;
        if (fstat(fd, &cs) == 0 && (size_t)cs.st_size > 64) {
            void *m = mmap(NULL, (size_t)cs.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m != MAP_FAILED) {
                const unsigned char *h = (const unsigned char *)m;
                uint64_t ssz, smt; uint32_t fl, nrec;
                memcpy(&ssz, h + 8, 8); memcpy(&smt, h + 16, 8); memcpy(&fl, h + 24, 4); memcpy(&nrec, h + 28, 4);
                if (!memcmp(h, "W22CACHE", 8) && ssz == c->src_size && smt == c->src_mtime && fl == c->flags) {
                    size_t pos = 64;
                    int ok = 1;
                    for (uint32_t i = 0; i < nrec && ok; i++) {
                        if (pos + 128 > (size_t)cs.st_size) { ok = 0; break; }
                        char nm[112]; uint64_t nb;
                        memcpy(nm, h + pos, 112); memcpy(&nb, h + pos + 112, 8);
                        nm[111] = 0;
                        size_t dpos = pos + 128;
                        if (dpos + nb > (size_t)cs.st_size) { ok = 0; break; }
                        w22_cache_add(c, nm, h + dpos, (size_t)nb);
                        pos = dpos + ((nb + 63) & ~(uint64_t)63);
                    }
                    if (ok && (int)nrec == c->nrec) {
                        c->map = (unsigned char *)m; c->maplen = (size_t)cs.st_size; c->mode = 1;
                        close(fd);
                        return;
                    }
                    free(c->recs); c->recs = NULL; c->nrec = c->caprec = 0;
                }
                munmap(m, (size_t)cs.st_size);
            }
        }
        close(fd);
    }
    /* write mode: header (magic written last), then records */
    snprintf(c->tmp, sizeof c->tmp, "%s.%d.tmp", c->path, (int)getpid());
    c->fw = fopen(c->tmp, "wb");
    if (!c->fw) return;  /* read-only dir: no cache */
    unsigned char z[64] = {0};
    fwrite(z, 1, 64, c->fw);
    c->wpos = 64;
    c->mode = 2;
}

static const unsigned char *w22_cache_get(w22_wcache *c, const char *name, size_t n) {
    if (c->mode != 1) return NULL;
    for (int i = 0; i < c->nrec; i++)
        if (!strcmp(c->recs[i].name, name)) return c->recs[i].n == n ? c->recs[i].p : NULL;
    return NULL;
}

static void w22_cache_put(w22_wcache *c, const char *name, const void *p, size_t n) {
    if (c->mode != 2 || !c->fw) return;
    unsigned char rh[128] = {0};
    snprintf((char *)rh, 112, "%s", name);
    uint64_t nb = n;
    memcpy(rh + 112, &nb, 8);
    fwrite(rh, 1, 128, c->fw);
    fwrite(p, 1, n, c->fw);
    size_t pad = ((n + 63) & ~(size_t)63) - n;
    static const unsigned char z[64] = {0};
    fwrite(z, 1, pad, c->fw);
    c->nrec++;
}

static void w22_cache_finish(w22_wcache *c, int ok) {
    if (c->mode != 2 || !c->fw) { w22_cache_close(c); return; }
    if (ok) {
        unsigned char h[64] = {0};
        memcpy(h + 8, &c->src_size, 8); memcpy(h + 16, &c->src_mtime, 8); memcpy(h + 24, &c->flags, 4);
        uint32_t nr = (uint32_t)c->nrec; memcpy(h + 28, &nr, 4);
        memcpy(h, "W22CACHE", 8);
        fseek(c->fw, 0, SEEK_SET);
        fwrite(h, 1, 64, c->fw);
        fclose(c->fw); c->fw = NULL;
        if (rename(c->tmp, c->path) != 0) remove(c->tmp);
    } else {
        fclose(c->fw); c->fw = NULL;
        remove(c->tmp);
    }
    w22_cache_close(c);
}

/* ---- weight loading ---- */
static float w22_bf16_round(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return f;
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float o;
    memcpy(&o, &u, 4);
    return o;
}
static float w22_wround(float f, int raw) {
    if (raw) return f;
    _Float16 h = (_Float16)f;
    return w22_bf16_round((float)h);
}

typedef struct { st_context *st; hip_wan22_vae *r; w22_wcache *wc; int err; } w22_ld;

static const float *w22_tensor(w22_ld *L, const char *name, int nd_expect, const uint64_t **shape) {
    int i = safetensors_find(L->st, name);
    if (i < 0) { fprintf(stderr, "wan22_vae: missing tensor %s\n", name); L->err = 1; return NULL; }
    if (strcmp(safetensors_dtype(L->st, i), "F32") != 0 || (nd_expect && safetensors_ndims(L->st, i) != nd_expect)) {
        fprintf(stderr, "wan22_vae: bad dtype/rank for %s\n", name); L->err = 1; return NULL;
    }
    *shape = safetensors_shape(L->st, i);
    return (const float *)safetensors_data(L->st, i);
}

static void w22_reg(hip_wan22_vae *r, w22_conv *c) { r->all[r->nall++] = c; }

/* prefix.weight / prefix.bias. cout_use limits to the first rows. Cin is padded to a multiple of 8
 * with zero weights. */
static void w22_load_conv(w22_ld *L, w22_conv *c, const char *prefix, int Cin, int Cout,
                          int kt, int kh, int kw, int st, int sh, int p0, int pt, int up2, int cf, int cout_use)
{
    hip_wan22_vae *r = L->r;
    char nm[256], cn[256];
    memset(c, 0, sizeof *c);
    int Cp = (Cin + 7) & ~7;
    c->Cin = Cp; c->Cout = cout_use ? cout_use : Cout; c->kt = kt; c->kh = kh; c->kw = kw;
    c->st = st; c->sh = sh; c->p0 = p0; c->pt = pt; c->up2 = up2; c->cf = cf;
    const uint64_t *sh_;
    snprintf(nm, sizeof nm, "%s.weight", prefix);
    const float *w = w22_tensor(L, nm, 0, &sh_);
    if (!w) return;
    if ((int)sh_[0] != Cout || (int)sh_[1] != Cin) { fprintf(stderr, "wan22_vae: shape mismatch %s\n", nm); L->err = 1; return; }
    int Co = c->Cout;
    size_t K = (size_t)kt * kh * kw * Cp;
    size_t nbytes = K * Co * 2;
    snprintf(cn, sizeof cn, "%s.w", prefix);
    c->w = w22_malloc(r, nbytes);
    if (!c->w) { L->err = 1; return; }
    const unsigned char *cached = w22_cache_get(L->wc, cn, nbytes);
    if (cached) {
        w22_h2d(c->w, cached, nbytes);
    } else {
        _Float16 *wh = (_Float16 *)malloc(nbytes);
        /* fp32 -> (fp16 -> bf16 rounded) -> fp16, permuted [o][t][y][x][ci] (zero-padded ci) */
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < Co; o++)
            for (int t = 0; t < kt; t++)
                for (int y = 0; y < kh; y++)
                    for (int x = 0; x < kw; x++) {
                        size_t dst = (size_t)o * K + (((size_t)t * kh + y) * kw + x) * Cp;
                        for (int ci = 0; ci < Cp; ci++) {
                            if (ci >= Cin) { wh[dst + ci] = (_Float16)0; continue; }
                            size_t src = ((((size_t)o * Cin + ci) * kt + t) * kh + y) * kw + x;
                            wh[dst + ci] = (_Float16)w22_wround(w[src], r->raw);
                        }
                    }
        w22_h2d(c->w, wh, nbytes);
        w22_cache_put(L->wc, cn, wh, nbytes);
        free(wh);
    }
    snprintf(nm, sizeof nm, "%s.bias", prefix);
    const float *b = w22_tensor(L, nm, 1, &sh_);
    if (!b) return;
    snprintf(cn, sizeof cn, "%s.b", prefix);
    c->bias = w22_malloc(r, F4(Co));
    const unsigned char *cb = w22_cache_get(L->wc, cn, F4(Co));
    if (cb) w22_h2d(c->bias, cb, F4(Co));
    else {
        float *bf = (float *)malloc(F4(Co));
        for (int o = 0; o < Co; o++) bf[o] = w22_wround(b[o], r->raw);
        w22_h2d(c->bias, bf, F4(Co));
        w22_cache_put(L->wc, cn, bf, F4(Co));
        free(bf);
    }
    w22_reg(r, c);
}

static void w22_load_norm(w22_ld *L, w22_norm *n, const char *prefix, int C) {
    char nm[256], cn[256];
    snprintf(nm, sizeof nm, "%s.gamma", prefix);
    snprintf(cn, sizeof cn, "%s.g", prefix);
    const uint64_t *sh_;
    const float *g = w22_tensor(L, nm, 0, &sh_);
    n->C = C;
    if (!g) return;
    n->gamma = w22_malloc(L->r, F4(C));
    L->r->gam[L->r->ngam++] = n->gamma;
    const unsigned char *cg = w22_cache_get(L->wc, cn, F4(C));
    if (cg) w22_h2d(n->gamma, cg, F4(C));
    else {
        float *gf = (float *)malloc(F4(C));
        for (int i = 0; i < C; i++) gf[i] = w22_wround(g[i], L->r->raw);
        w22_h2d(n->gamma, gf, F4(C));
        w22_cache_put(L->wc, cn, gf, F4(C));
        free(gf);
    }
}

static void w22_load_res(w22_ld *L, w22_res *b, const char *p, int in, int out) {
    char nm[256];
    b->in = in; b->out = out;
    snprintf(nm, sizeof nm, "%s.residual.0", p); w22_load_norm(L, &b->n1, nm, in);
    snprintf(nm, sizeof nm, "%s.residual.2", p); w22_load_conv(L, &b->c1, nm, in, out, 3, 3, 3, 1, 1, 1, 2, 0, 2, 0);
    snprintf(nm, sizeof nm, "%s.residual.3", p); w22_load_norm(L, &b->n2, nm, out);
    snprintf(nm, sizeof nm, "%s.residual.6", p); w22_load_conv(L, &b->c2, nm, out, out, 3, 3, 3, 1, 1, 1, 2, 0, 2, 0);
    b->has_sc = in != out;
    if (b->has_sc) { snprintf(nm, sizeof nm, "%s.shortcut", p); w22_load_conv(L, &b->sc, nm, in, out, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0); }
}

static void w22_load_attn(w22_ld *L, w22_attn *a, const char *p, int C) {
    char nm[256];
    a->C = C;
    snprintf(nm, sizeof nm, "%s.norm", p); w22_load_norm(L, &a->norm, nm, C);
    snprintf(nm, sizeof nm, "%s.to_qkv", p); w22_load_conv(L, &a->qkv, nm, C, 3 * C, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
    snprintf(nm, sizeof nm, "%s.proj", p); w22_load_conv(L, &a->proj, nm, C, C, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
}

static void w22_reset_caches(hip_wan22_vae *r) {
    for (int i = 0; i < r->nall; i++) {
        w22_conv *c = r->all[i];
        if (c->cache) w22_set8(c->cache, 0, c->cache_bytes);
        c->cache_valid = 0;
    }
}
static void w22_free_caches(hip_wan22_vae *r) {
    for (int i = 0; i < r->nall; i++) {
        w22_conv *c = r->all[i];
        if (c->cache) { w22_mfree(r, c->cache, c->cache_bytes); c->cache = 0; c->cache_bytes = 0; }
        c->cache_valid = 0;
    }
}

size_t hip_wan22_vae_peak_bytes(const hip_wan22_vae *r) { return r->peak_bytes; }
void hip_wan22_vae_reset_peak(hip_wan22_vae *r) { r->peak_bytes = r->cur_bytes; }
double hip_wan22_vae_last_ms(const hip_wan22_vae *r) { return r->last_ms; }
const char *hip_wan22_vae_last_error(const hip_wan22_vae *r) { return r->err; }
void hip_wan22_vae_set_tap_dir(hip_wan22_vae *r, const char *dir) {
    if (dir) snprintf(r->tap_dir, sizeof r->tap_dir, "%s", dir); else r->tap_dir[0] = 0;
}
void hip_wan22_vae_set_input_range255(hip_wan22_vae *r, int on) { r->range255 = on; }
void hip_wan22_vae_set_tiling(hip_wan22_vae *r, int tile_px, int overlap_px) {
    r->tile_px = tile_px > 0 ? (tile_px + 31) & ~31 : tile_px;
    r->overlap_px = overlap_px > 0 ? (overlap_px + 31) & ~31 : 0;
}
void hip_wan22_vae_set_persistent(hip_wan22_vae *r, int on) { r->persistent = on; }

static void w22_free_weights(hip_wan22_vae *r) {
    for (int i = 0; i < r->nall; i++) {
        w22_conv *c = r->all[i];
        if (c->w) { w22_dfree(c->w); c->w = 0; }
        if (c->bias) { w22_dfree(c->bias); c->bias = 0; }
        if (c->cache) { w22_dfree(c->cache); c->cache = 0; }
    }
    for (int i = 0; i < r->ngam; i++) if (r->gam[i]) w22_dfree(r->gam[i]);
    r->ngam = 0;
    for (int i = 0; i < r->npool; i++) if (r->pool[i].p) { w22_dfree(r->pool[i].p); r->pool[i].p = 0; }
    r->npool = 0;
}

hip_wan22_vae *hip_wan22_vae_load(const char *dir, int device_id, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) { fprintf(stderr, "wan22_vae: rocewInit failed\n"); return NULL; }
    if (hipInit(0) != hipSuccess) { fprintf(stderr, "wan22_vae: hipInit failed\n"); return NULL; }
    hip_wan22_vae *r = (hip_wan22_vae *)calloc(1, sizeof *r);
    r->verbose = verbose;
    r->tile_px = -1;
    r->dev = device_id;
    if (hipSetDevice(device_id) != hipSuccess) { fprintf(stderr, "wan22_vae: no HIP device %d\n", device_id); free(r); return NULL; }
    {
        r->nsm = 0;
        { hipDeviceProp_t pr; memset(&pr, 0, sizeof pr); if (hipGetDeviceProperties(&pr, device_id) == hipSuccess) r->nsm = pr.multiProcessorCount; }
        if (verbose) fprintf(stderr, "wan22_vae: device %d, %d CUs\n", device_id, r->nsm);
        if (r->nsm < 1) r->nsm = 32;
    }
    r->raw = getenv("WAN22_VAE_RAW_WEIGHTS") && atoi(getenv("WAN22_VAE_RAW_WEIGHTS"));
    {
        const char *te = getenv("WAN22_VAE_TILE");
        if (te) {
            int t = 0, o = -1;
            sscanf(te, "%d,%d", &t, &o);
            r->tile_px = t > 0 ? (t + 31) & ~31 : t;
            if (o >= 0) r->overlap_px = (o + 31) & ~31;
        }
    }

    if (hip_compile_kernels_ex(&r->mod, device_id, w22_kernel_src, "wan22_vae", verbose, "wan22_vae", 1) < 0) { free(r); return NULL; }
    struct { hipFunction_t *f; const char *n; } fns[] = {
        { &r->f_conv[0], "w22_conv_128x128" }, { &r->f_conv[1], "w22_conv_64x128" }, { &r->f_conv[2], "w22_conv_64x64" },
        { &r->f_splitk, "w22_splitk_reduce" }, { &r->f_f2h, "w22_f32_to_f16" },
        { &r->f_norm, "w22_rmsnorm" },
        { &r->f_avg, "w22_avg_down_add" }, { &r->f_dup, "w22_dup_up_add" }, { &r->f_tint, "w22_tconv_interleave" },
        { &r->f_patch, "w22_patchify" }, { &r->f_unpatch, "w22_unpatchify" }, { &r->f_softmax, "w22_softmax_rows" } };
    for (size_t i = 0; i < sizeof fns / sizeof fns[0]; i++)
        if (hipModuleGetFunction(fns[i].f, r->mod, fns[i].n) != hipSuccess) { fprintf(stderr, "wan22_vae: missing kernel %s\n", fns[i].n); free(r); return NULL; }

    char path[1024];
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    w22_wcache wc;
    w22_cache_open(&wc, dir, r->raw);
    w22_ld L = { NULL, r, &wc, 0 };
    L.st = safetensors_open(path);
    if (!L.st) { fprintf(stderr, "wan22_vae: cannot open %s\n", path); w22_cache_finish(&wc, 0); free(r); return NULL; }
    double t0 = w22_now_ms();
    const char *cmode = wc.mode == 1 ? "fp16 cache hit" : wc.mode == 2 ? "converted, cache written" : "converted";

    char nm[256];
    /* encoder */
    {
        const int dims[5] = { 160, 160, 320, 640, 640 };
        const int down_t[4] = { 0, 1, 1, 0 };
        w22_load_conv(&L, &r->e_conv1, "encoder.conv1", 12, 160, 3, 3, 3, 1, 1, 1, 2, 0, 2, 0);
        for (int i = 0; i < 4; i++) {
            w22_blk *b = &r->e_blk[i];
            b->in = dims[i]; b->out = dims[i + 1]; b->nres = 2;
            b->has_rs = i != 3;
            b->ft = down_t[i] ? 2 : 1; b->fs = b->has_rs ? 2 : 1;
            for (int j = 0; j < 2; j++) {
                snprintf(nm, sizeof nm, "encoder.downsamples.%d.downsamples.%d", i, j);
                w22_load_res(&L, &b->res[j], nm, j == 0 ? dims[i] : dims[i + 1], dims[i + 1]);
            }
            if (b->has_rs) {
                int d = dims[i + 1];
                b->rs.dim = d;
                snprintf(nm, sizeof nm, "encoder.downsamples.%d.downsamples.2.resample.1", i);
                w22_load_conv(&L, &b->rs.sp, nm, d, d, 1, 3, 3, 1, 2, 0, 1, 0, 0, 0);
                b->rs.has_tc = down_t[i];
                if (b->rs.has_tc) {
                    snprintf(nm, sizeof nm, "encoder.downsamples.%d.downsamples.2.time_conv", i);
                    w22_load_conv(&L, &b->rs.tc, nm, d, d, 3, 1, 1, 2, 1, 0, 0, 0, 1, 0);
                }
            }
        }
        w22_load_res(&L, &r->e_mid[0], "encoder.middle.0", 640, 640);
        w22_load_attn(&L, &r->e_attn, "encoder.middle.1", 640);
        w22_load_res(&L, &r->e_mid[1], "encoder.middle.2", 640, 640);
        w22_load_norm(&L, &r->e_hnorm, "encoder.head.0", 640);
        w22_load_conv(&L, &r->e_head, "encoder.head.2", 640, 96, 3, 3, 3, 1, 1, 1, 2, 0, 2, 0);
        w22_load_conv(&L, &r->e_mu, "conv1", 96, 96, 1, 1, 1, 1, 1, 0, 0, 0, 0, 48);
    }
    /* decoder */
    {
        const int dims[5] = { 1024, 1024, 1024, 512, 256 };
        const int up_t[4] = { 1, 1, 0, 0 };
        w22_load_conv(&L, &r->d_conv2, "conv2", 48, 48, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0);
        w22_load_conv(&L, &r->d_conv1, "decoder.conv1", 48, 1024, 3, 3, 3, 1, 1, 1, 2, 0, 2, 0);
        w22_load_res(&L, &r->d_mid[0], "decoder.middle.0", 1024, 1024);
        w22_load_attn(&L, &r->d_attn, "decoder.middle.1", 1024);
        w22_load_res(&L, &r->d_mid[1], "decoder.middle.2", 1024, 1024);
        for (int i = 0; i < 4; i++) {
            w22_blk *b = &r->d_blk[i];
            b->in = dims[i]; b->out = dims[i + 1]; b->nres = 3;
            b->has_rs = i != 3;
            b->ft = up_t[i] ? 2 : 1; b->fs = 2;
            for (int j = 0; j < 3; j++) {
                snprintf(nm, sizeof nm, "decoder.upsamples.%d.upsamples.%d", i, j);
                w22_load_res(&L, &b->res[j], nm, j == 0 ? dims[i] : dims[i + 1], dims[i + 1]);
            }
            if (b->has_rs) {
                int d = dims[i + 1];
                b->rs.dim = d;
                snprintf(nm, sizeof nm, "decoder.upsamples.%d.upsamples.3.resample.1", i);
                w22_load_conv(&L, &b->rs.sp, nm, d, d, 1, 3, 3, 1, 1, 1, 2, 1, 0, 0);
                b->rs.has_tc = up_t[i];
                if (b->rs.has_tc) {
                    snprintf(nm, sizeof nm, "decoder.upsamples.%d.upsamples.3.time_conv", i);
                    w22_load_conv(&L, &b->rs.tc, nm, d, 2 * d, 3, 1, 1, 1, 1, 0, 0, 0, 2, 0);
                }
            }
        }
        w22_load_norm(&L, &r->d_hnorm, "decoder.head.0", 256);
        w22_load_conv(&L, &r->d_head, "decoder.head.2", 256, 12, 3, 3, 3, 1, 1, 1, 2, 0, 2, 0);
    }
    safetensors_close(L.st);
    w22_cache_finish(&wc, !L.err);
    if (L.err) { fprintf(stderr, "wan22_vae: weight load failed\n"); w22_free_weights(r); free(r); return NULL; }
    hipDeviceSynchronize();
    if (verbose)
        fprintf(stderr, "wan22_vae: weights loaded in %.0f ms (%s; fp16 tensor-op convs%s), %.2f GB on device\n", w22_now_ms() - t0,
                cmode, r->raw ? ", raw weights" : ", fp16->bf16 weights", r->cur_bytes / 1073741824.0);
    r->peak_bytes = r->cur_bytes;
    r->wbytes = r->cur_bytes;
    return r;
}

/* ------------------------------------------------------------------ */
/* Encoder                                                              */
/* ------------------------------------------------------------------ */
static w22_dp w22_resample_down(hip_wan22_vae *r, w22_rs *rs, w22_dp x, int T, int *H, int *W, int *oT) {
    int h = *H, w = *W;
    w22_dp y = w22_conv_plain32(r, &rs->sp, x, T, &h, &w, 0);
    if (!y) return 0;
    *oT = T;
    if (!rs->has_tc) { *H = h; *W = w; return y; }
    size_t fe = (size_t)h * w * rs->dim;
    size_t need = F2(fe);
    if (rs->tc.cache && rs->tc.cache_bytes != need) { w22_mfree(r, rs->tc.cache, rs->tc.cache_bytes); rs->tc.cache = 0; rs->tc.cache_valid = 0; }
    if (!rs->tc.cache) {
        rs->tc.cache = w22_malloc(r, need);
        if (!rs->tc.cache) { w22_seterr(r, "out of device memory for a causal cache (%.1f MB)", need / 1048576.0); return 0; }
        rs->tc.cache_bytes = need;
        w22_set8(rs->tc.cache, 0, need);
        rs->tc.cache_valid = 0;
    }
    if (!rs->tc.cache_valid) {
        /* first chunk: pass through, keep the last frame (fp16) */
        if (w22_to_f16(r, y + F4((size_t)(T - 1) * fe), rs->tc.cache, (long long)fe)) return 0;
        rs->tc.cache_valid = 1;
        *H = h; *W = w;
        return y;
    }
    w22_dp halo = w22_halo_begin(r, &rs->tc, T, h, w);
    if (!halo) return 0;
    if (w22_to_f16(r, y, halo + F2(fe), (long long)T * fe)) return 0;
    w22_free(r, y);
    int hh = h, ww = w;
    w22_dp o = w22_conv_halo(r, &rs->tc, halo, T, &hh, &ww, 0);
    if (!o) return 0;
    *oT = (1 + T - 3) / 2 + 1;
    *H = h; *W = w;
    return o;
}

static int w22_launch_avg(hip_wan22_vae *r, w22_dp x, w22_dp acc, int T, int H, int W, int C,
                          int To, int Ho, int Wo, int Co, int ft, int fs, int gs, int padt) {
    void *args[] = { &x, &acc, &T, &H, &W, &C, &To, &Ho, &Wo, &Co, &ft, &fs, &gs, &padt };
    return w22_launch1d(r, r->f_avg, (long long)To * Ho * Wo * Co, args);
}

static const char *w22_tapn(char *b, const char *pfx, int i) { snprintf(b, 32, "%s%d", pfx, i); return b; }

/* One encoder chunk: frames [t0,t0+nt) of the device video. Returns mu
 * [nt_out][h][w][48]; *To, h, w set. */
static w22_dp w22_enc_chunk(hip_wan22_vae *r, w22_dp vid, int H, int W, int nt, int *oT, int *oh, int *ow) {
    int Tv = nt, t0 = 0;
    char tn[32];
    int h = H / 2, w = W / 2, T = nt;
    /* conv1 on patchified chunk (16 channels, 12 used) */
    w22_dp halo = w22_halo_begin(r, &r->e_conv1, T, h, w);
    if (!halo) return 0;
    {
        w22_dp dst = halo + F2((size_t)r->e_conv1.cf * h * w * 16);
        void *args[] = { &vid, &dst, &Tv, &H, &W, &t0, &nt };
        if (w22_launch1d(r, r->f_patch, (long long)nt * h * w * 16, args)) return 0;
    }
    w22_dp x = w22_conv_halo(r, &r->e_conv1, halo, T, &h, &w, 0);
    if (!x) return 0;
    w22_tap(r, "enc_conv1", x, T, h, w, 160);
    for (int i = 0; i < 4; i++) {
        w22_blk *b = &r->e_blk[i];
        w22_dp main = x;
        for (int j = 0; j < b->nres; j++) {
            w22_dp n = w22_resblock(r, &b->res[j], main, T, &h, &w);
            if (!n) return 0;
            if (main != x) w22_free(r, main);
            main = n;
        }
        int Hin = h, Win = w, Tin = T;
        if (b->has_rs) {
            int oT2;
            w22_dp n = w22_resample_down(r, &b->rs, main, T, &h, &w, &oT2);
            if (!n) return 0;
            w22_free(r, main);
            main = n; T = oT2;
        }
        /* avg shortcut of the block input */
        int ft = b->ft, fs = b->fs, factor = ft * fs * fs, gs = b->in * factor / b->out;
        int padt = (ft - Tin % ft) % ft;
        if (w22_launch_avg(r, x, main, Tin, Hin, Win, b->in, T, h, w, b->out, ft, fs, gs, padt)) return 0;
        w22_free(r, x);
        x = main;
        w22_tap(r, w22_tapn(tn, "enc_down", i), x, T, h, w, b->out);
    }
    {
        w22_dp a = w22_resblock(r, &r->e_mid[0], x, T, &h, &w);
        if (!a) return 0;
        w22_free(r, x);
        w22_dp b2 = w22_attnblock(r, &r->e_attn, a, T, h, w);
        if (!b2) return 0;
        w22_free(r, a);
        x = w22_resblock(r, &r->e_mid[1], b2, T, &h, &w);
        if (!x) return 0;
        w22_free(r, b2);
        w22_tap(r, "enc_mid", x, T, h, w, 640);
    }
    w22_dp hd = w22_norm_conv(r, &r->e_hnorm, &r->e_head, x, T, &h, &w, 0);
    if (!hd) return 0;
    w22_free(r, x);
    w22_tap(r, "enc_head", hd, T, h, w, 96);
    int h1 = h, w1 = w;
    w22_dp mu = w22_conv_plain32(r, &r->e_mu, hd, T, &h1, &w1, 0);
    w22_free(r, hd);
    *oT = T; *oh = h; *ow = w;
    return mu;
}

static const float w22_mean[48] = {
    -0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f,
    -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.1570f, -0.0098f, 0.0375f, -0.1825f,
    -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f,
    -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.1230f,
    -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.0520f, 0.3748f,
    0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f };
static const float w22_stdv[48] = {
    0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f,
    0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f,
    0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f,
    0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f,
    0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f,
    0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f };

/* Encode one tile: host video [3][T][H][W] fp32 in [-1,1], H,W multiples of 32.
 * Returns policy latent [192][nlat][H/32][W/32] (malloc'd). */
static float *w22_encode_tile(hip_wan22_vae *r, const float *video, int T, int H, int W, int *Tl) {
    w22_reset_caches(r);
    float *mu = NULL, *lat = NULL;
    int teff = 1 + 4 * ((T - 1) / 4);
    int nlat = 1 + (teff - 1) / 4;
    int h = H / 16, w = W / 16;  /* encoder output spatial */
    size_t HW = (size_t)H * W;
    mu = (float *)malloc(F4((size_t)nlat * h * w * 48));
    int got = 0;
    for (int cut_prev = 0, ci = 0; cut_prev < teff; ci++) {
        int cut = ci == 0 ? 1 : cut_prev + 4;
        int nt = cut - cut_prev;
        int oT, oh, ow;
        /* upload just this chunk of frames: [3][nt][H][W] */
        w22_dp vid = w22_alloc(r, F4(3 * (size_t)nt * HW));
        if (!vid) goto fail;
        for (int c = 0; c < 3; c++)
            w22_h2d(vid + F4((size_t)c * nt * HW), video + ((size_t)c * T + cut_prev) * HW, F4((size_t)nt * HW));
        w22_dp m = w22_enc_chunk(r, vid, H, W, nt, &oT, &oh, &ow);
        w22_free(r, vid);
        if (!m || r->failed || oh != h || ow != w) {
            if (!r->failed) w22_seterr(r, "encoder chunk failed");
            goto fail;
        }
        hipDeviceSynchronize();
        w22_d2h(mu + (size_t)got * h * w * 48, m, F4((size_t)oT * h * w * 48));
        w22_free(r, m);
        got += oT;
        cut_prev = cut;
    }
    if (got != nlat) { w22_seterr(r, "got %d latent frames, expected %d", got, nlat); goto fail; }
    {
        /* normalise, bf16-round, policy patchify: out [192][nlat][h/2][w/2], ch = c*4 + ph*2 + pw */
        int hp = h / 2, wp = w / 2;
        lat = (float *)malloc(F4((size_t)192 * nlat * hp * wp));
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < nlat; t++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int c = 0; c < 48; c++) {
                        float v = mu[(((size_t)t * h + y) * w + x) * 48 + c];
                        v = (v - w22_mean[c]) * (1.0f / w22_stdv[c]);
                        v = w22_bf16_round(v);
                        int ch = c * 4 + (y & 1) * 2 + (x & 1);
                        lat[((((size_t)ch * nlat) + t) * hp + (y >> 1)) * wp + (x >> 1)] = v;
                    }
    }
    free(mu);
    *Tl = nlat;
    return lat;
fail:
    free(mu);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Decoder                                                              */
/* ------------------------------------------------------------------ */
/* Decode one tile: policy latent [192][Tl][hp][wp] (contiguous) -> frames [3][To][32hp][32wp]. */
static float *w22_decode_tile(hip_wan22_vae *r, const float *latent, int Tl, int hp, int wp, int *To_out) {
    w22_reset_caches(r);
    int h = hp * 2, w = wp * 2;
    w22_dp out = 0, x48 = 0;
    size_t out_n = 0;
    int To = 1 + 4 * (Tl - 1);
    int H = h * 16, W = w * 16;
    float *fr = NULL;
    /* from_policy_latent + z/inv_std + mean -> [Tl][h][w][48] */
    float *z = (float *)malloc(F4((size_t)Tl * h * w * 48));
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < Tl; t++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int c = 0; c < 48; c++) {
                    int ch = c * 4 + (y & 1) * 2 + (x & 1);
                    float v = latent[((((size_t)ch * Tl) + t) * hp + (y >> 1)) * wp + (x >> 1)];
                    v = w22_bf16_round(v);
                    float inv = 1.0f / w22_stdv[c];
                    z[(((size_t)t * h + y) * w + x) * 48 + c] = v / inv + w22_mean[c];
                }
    w22_dp dz = w22_alloc(r, F4((size_t)Tl * h * w * 48));
    w22_dp dz16 = w22_alloc(r, F2((size_t)Tl * h * w * 48));
    if (!dz || !dz16) { free(z); return NULL; }
    w22_h2d(dz, z, F4((size_t)Tl * h * w * 48));
    free(z);
    if (w22_to_f16(r, dz, dz16, (long long)Tl * h * w * 48)) return NULL;
    w22_free(r, dz);
    int hh = h, ww = w;
    x48 = w22_conv_plain(r, &r->d_conv2, dz16, Tl, &hh, &ww, 0);
    w22_free(r, dz16);
    if (!x48) return NULL;
    w22_tap(r, "dec_conv2", x48, Tl, h, w, 48);

    out_n = (size_t)3 * 4 * H * W;  /* one chunk (<= 4 frames) at a time, copied to the host after each chunk */
    out = w22_malloc(r, F4(out_n));
    if (!out) { w22_seterr(r, "out of device memory for the decoded frames (%.1f MB)", F4(out_n) / 1048576.0); return NULL; }
    fr = (float *)malloc(F4((size_t)3 * To * H * W));
    if (!fr) { w22_seterr(r, "out of host memory for the decoded frames"); goto fail; }
    char tn[32];
    int tbase = 0;
    double t0 = w22_now_ms();
    for (int ci = 0; ci < Tl; ci++) {
        int T = 1, ch = h, cw = w;
        int first = ci == 0;
        w22_dp halo = w22_halo_begin(r, &r->d_conv1, 1, ch, cw);
        if (!halo) goto fail;
        if (w22_to_f16(r, x48 + F4((size_t)ci * ch * cw * 48), halo + F2((size_t)r->d_conv1.cf * ch * cw * 48), (long long)ch * cw * 48)) goto fail;
        w22_dp x = w22_conv_halo(r, &r->d_conv1, halo, 1, &ch, &cw, 0);
        if (!x) goto fail;
        w22_tap(r, "dec_conv1", x, T, ch, cw, 1024);
        {
            w22_dp a = w22_resblock(r, &r->d_mid[0], x, T, &ch, &cw);
            if (!a) goto fail;
            w22_free(r, x);
            w22_dp b2 = w22_attnblock(r, &r->d_attn, a, T, ch, cw);
            if (!b2) goto fail;
            w22_free(r, a);
            x = w22_resblock(r, &r->d_mid[1], b2, T, &ch, &cw);
            if (!x) goto fail;
            w22_free(r, b2);
            w22_tap(r, "dec_mid", x, T, ch, cw, 1024);
        }
        for (int i = 0; i < 4; i++) {
            w22_blk *b = &r->d_blk[i];
            w22_dp main = x;
            int Tin = T, Hin = ch, Win = cw;
            for (int j = 0; j < b->nres; j++) {
                w22_dp n = w22_resblock(r, &b->res[j], main, T, &ch, &cw);
                if (!n) goto fail;
                if (main != x) w22_free(r, main);
                main = n;
            }
            if (b->has_rs) {
                w22_rs *rs = &b->rs;
                int d = rs->dim;
                w22_dp y16 = 0;
                if (rs->has_tc && !first) {
                    w22_dp halo2 = w22_halo_begin(r, &rs->tc, T, ch, cw);
                    if (!halo2) goto fail;
                    if (w22_to_f16(r, main, halo2 + F2((size_t)rs->tc.cf * ch * cw * d), (long long)T * ch * cw * d)) goto fail;
                    w22_free(r, main);
                    int h2 = ch, w2 = cw;
                    w22_dp y2 = w22_conv_halo(r, &rs->tc, halo2, T, &h2, &w2, 0);
                    if (!y2) goto fail;
                    y16 = w22_alloc(r, F2((size_t)2 * T * ch * cw * d));
                    if (!y16) goto fail;
                    long long HWl = (long long)ch * cw;
                    void *args[] = { &y2, &y16, &T, &HWl, &d };
                    w22_launch1d(r, r->f_tint, 2LL * T * HWl * d, args);
                    w22_free(r, y2);
                    T *= 2;
                } else {
                    y16 = w22_alloc(r, F2((size_t)T * ch * cw * d));
                    if (!y16) goto fail;
                    if (w22_to_f16(r, main, y16, (long long)T * ch * cw * d)) goto fail;
                    w22_free(r, main);
                }
                int h2 = ch, w2 = cw;
                w22_dp y = w22_conv_plain(r, &rs->sp, y16, T, &h2, &w2, 0);
                w22_free(r, y16);
                if (!y) goto fail;
                main = y;
                ch = h2; cw = w2;
                /* DupUp3D shortcut of the block input */
                int ft = b->ft, fs = 2;
                int repeats = b->out * ft * fs * fs / b->in;
                int drop = first ? ft - 1 : 0;
                void *args[] = { &x, &main, &Tin, &Hin, &Win, &b->in, &b->out, &ft, &fs, &repeats, &drop };
                w22_launch1d(r, r->f_dup, (long long)(Tin * ft - drop) * (Hin * fs) * (Win * fs) * b->out, args);
            }
            w22_free(r, x);
            x = main;
            w22_tap(r, w22_tapn(tn, "dec_up", i), x, T, ch, cw, b->out);
        }
        w22_dp hd = w22_norm_conv(r, &r->d_hnorm, &r->d_head, x, T, &ch, &cw, 0);
        if (!hd) goto fail;
        w22_free(r, x);
        w22_tap(r, "dec_head", hd, T, ch, cw, 12);
        {
            int four = 4, zero = 0;
            void *args[] = { &hd, &out, &T, &ch, &cw, &four, &zero };
            w22_launch1d(r, r->f_unpatch, (long long)T * ch * cw * 12, args);
        }
        w22_free(r, hd);
        if (r->failed) goto fail;
        hipDeviceSynchronize();
        for (int c = 0; c < 3; c++)
            w22_d2h(fr + ((size_t)c * To + tbase) * H * W, out + F4((size_t)c * 4 * H * W), F4((size_t)T * H * W));
        tbase += T;
        if (r->verbose >= 2) { hipDeviceSynchronize(); fprintf(stderr, "wan22_vae: decoded chunk %d/%d (%.0f ms)\n", ci + 1, Tl, w22_now_ms() - t0); }
    }
    w22_free(r, x48); x48 = 0;
    if (tbase != To) { w22_seterr(r, "decode frame count mismatch %d vs %d", tbase, To); goto fail; }
    hipDeviceSynchronize();
    w22_mfree(r, out, F4(out_n));
    *To_out = To;
    return fr;
fail:
    if (out) w22_mfree(r, out, F4(out_n));
    free(fr);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Tiling, padding, public API                                          */
/* ------------------------------------------------------------------ */
typedef struct { int n, start[64]; } w22_axis;

/* tile starts along an axis of L units with tile size S and minimum overlap O */
static void w22_axis_plan(w22_axis *a, int L, int S, int O) {
    a->n = 0;
    if (S <= 0 || L <= S) { a->n = 1; a->start[0] = 0; return; }
    int stride = S - O;
    if (stride < 1) stride = 1;
    int s = 0;
    while (a->n < 63) {
        a->start[a->n++] = s;
        if (s + S >= L) break;
        s += stride;
        if (s + S > L) s = L - S;
    }
}

/* 1-D blend weight of tile i at offset x (units of the tile), ramp over the overlap with its neighbours. */
static float w22_axis_w(const w22_axis *a, int i, int S, int x) {
    float w = 1.f;
    if (i > 0) {
        int ov = a->start[i - 1] + S - a->start[i];
        if (x < ov) w *= (x + 0.5f) / ov;
    }
    if (i + 1 < a->n) {
        int ov = a->start[i] + S - a->start[i + 1];
        if (x >= S - ov) w *= (S - x - 0.5f) / ov;
    }
    return w;
}

/* Device scratch bytes per tile pixel (measured, see README) and the automatic budget. */
#define W22_ENC_BYTES_PER_PX 6700.0
#define W22_DEC_BYTES_PER_PX 14500.0

/* Pick the tile rectangle (multiples of 32). tile_px: 0 untiled, >0 max tile edge (area tile_px^2), <0 automatic
 * from WAN22_VAE_BUDGET_GB (default 5.0 GB total incl. weights). Returns 1 if tiled. */
static int w22_tile_dims(hip_wan22_vae *r, int Hp, int Wp, int decode, int *tH, int *tW) {
    *tH = Hp; *tW = Wp;
    long long area;
    if (r->tile_px == 0) return 0;
    if (r->tile_px > 0) area = (long long)r->tile_px * r->tile_px;
    else {
        const char *be = getenv("WAN22_VAE_BUDGET_GB");
        double budget = be ? atof(be) : 5.0;
        double scratch = budget * 1073741824.0 - (double)r->wbytes;
        if (scratch < 256.0 * 1048576.0) scratch = 256.0 * 1048576.0;
        area = (long long)(scratch / (decode ? W22_DEC_BYTES_PER_PX : W22_ENC_BYTES_PER_PX));
    }
    if ((long long)Hp * Wp <= area) return 0;
    int e = (int)sqrt((double)area) & ~31;
    if (e < 128) e = 128;
    int th = Hp < e ? Hp : e;
    int tw = (int)(area / th) & ~31; if (tw < 128) tw = 128; if (tw > Wp) tw = Wp;
    if (tw == Wp) { th = (int)(area / tw) & ~31; if (th < 128) th = 128; if (th > Hp) th = Hp; }
    *tH = th; *tW = tw;
    return th < Hp || tw < Wp;
}

static void w22_cleanup_after(hip_wan22_vae *r, int ok) {
    w22_tap_flush(r);
    if (!ok) { w22_pool_reset(r); w22_free_caches(r); }
    if (!r->persistent || !ok) w22_pool_release(r);
    if (!r->persistent) w22_free_caches(r);
    hipDeviceSynchronize();
}

static int w22_encode_impl(hip_wan22_vae *r, const float *vf, const unsigned char *vu, int T, int H, int W,
                           float **latent, int *Tl_out) {
    if (!r) return -1;
    double t0 = w22_now_ms();
    r->failed = 0; r->err[0] = 0;
    if (!latent || !Tl_out || T < 1 || H < 1 || W < 1 || (!vf && !vu)) { snprintf(r->err, sizeof r->err, "bad arguments"); return -1; }
    int Hp = (H + 31) & ~31, Wp = (W + 31) & ~31;
    int tH, tW;
    w22_tile_dims(r, Hp, Wp, 0, &tH, &tW);
    int ovr = r->overlap_px > 0 ? r->overlap_px : 128;
    w22_axis ay, ax;
    w22_axis_plan(&ay, Hp, tH, (ovr > tH / 2 ? tH / 2 : ovr) & ~31);
    w22_axis_plan(&ax, Wp, tW, (ovr > tW / 2 ? tW / 2 : ovr) & ~31);
    int ntiles = ay.n * ax.n;
    int hl = Hp / 32, wl = Wp / 32;
    float in_scale = vu ? 1.f / 127.5f : (r->range255 ? 1.f / 127.5f : 1.f);
    float in_bias = (vu || r->range255) ? -1.f : 0.f;
    int zero_copy = !vu && !r->range255 && H == Hp && W == Wp && ntiles == 1;
    size_t tvn = (size_t)3 * T * tH * tW;
    float *tv = zero_copy ? NULL : (float *)malloc(F4(tvn));
    float *acc = NULL, *wacc = NULL;
    int nlat = 0;
    float *res = NULL;
    for (int iy = 0; iy < ay.n; iy++)
        for (int ix = 0; ix < ax.n; ix++) {
            int y0 = ay.start[iy], x0 = ax.start[ix];
            const float *src = vf;
            if (!zero_copy) {
                #pragma omp parallel for collapse(2) schedule(static)
                for (int c = 0; c < 3; c++)
                    for (int t = 0; t < T; t++)
                        for (int y = 0; y < tH; y++) {
                            int sy = y0 + y; if (sy > H - 1) sy = H - 1;
                            float *d = tv + (((size_t)c * T + t) * tH + y) * tW;
                            size_t sb = (((size_t)c * T + t) * H + sy) * W;
                            for (int x = 0; x < tW; x++) {
                                int sx = x0 + x; if (sx > W - 1) sx = W - 1;
                                float v = vu ? (float)vu[sb + sx] : vf[sb + sx];
                                d[x] = v * in_scale + in_bias;
                            }
                        }
                src = tv;
            }
            int Tl = 0;
            float *lt = w22_encode_tile(r, src, T, tH, tW, &Tl);
            if (!lt) { free(tv); free(acc); free(wacc); w22_cleanup_after(r, 0); return -1; }
            nlat = Tl;
            int th = tH / 32, tw = tW / 32;
            if (ntiles == 1) { res = lt; break; }
            if (!acc) {
                acc = (float *)calloc((size_t)192 * Tl * hl * wl, sizeof(float));
                wacc = (float *)calloc((size_t)hl * wl, sizeof(float));
            }
            int ly0 = y0 / 32, lx0 = x0 / 32;
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < th; y++) {
                float wy = w22_axis_w(&ay, iy, th, y);
                for (int x = 0; x < tw; x++) {
                    float wv = wy * w22_axis_w(&ax, ix, tw, x);
                    wacc[(size_t)(ly0 + y) * wl + lx0 + x] += wv;
                    for (int c = 0; c < 192; c++)
                        for (int t = 0; t < Tl; t++)
                            acc[((((size_t)c * Tl) + t) * hl + ly0 + y) * wl + lx0 + x] +=
                                wv * lt[((((size_t)c * Tl) + t) * th + y) * tw + x];
                }
            }
            free(lt);
        }
    free(tv);
    if (!res) {
        res = acc;
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < 192; c++)
            for (int t = 0; t < nlat; t++)
                for (int p = 0; p < hl * wl; p++) {
                    size_t i = (((size_t)c * nlat) + t) * hl * wl + p;
                    res[i] = w22_bf16_round(res[i] / wacc[p]);
                }
        free(wacc);
    }
    w22_cleanup_after(r, 1);
    r->last_ms = w22_now_ms() - t0;
    if (r->verbose >= 2) { fprintf(stderr, "wan22_vae: encode %.1f GFLOP -> %.1f TFLOPS\n", r->flops * 1e-9, r->flops / (r->last_ms * 1e9)); }
    r->flops = 0;
    *latent = res; *Tl_out = nlat;
    return 0;
}

int hip_wan22_vae_encode(hip_wan22_vae *r, const float *video, int T, int H, int W, float **latent, int *Tl) {
    return w22_encode_impl(r, video, NULL, T, H, W, latent, Tl);
}
int hip_wan22_vae_encode_u8(hip_wan22_vae *r, const unsigned char *video, int T, int H, int W, float **latent, int *Tl) {
    return w22_encode_impl(r, NULL, video, T, H, W, latent, Tl);
}

int hip_wan22_vae_decode_crop(hip_wan22_vae *r, const float *latent, int Tl, int hp, int wp,
                               int Hc, int Wc, float **frames, int *To_out) {
    if (!r) return -1;
    double t0 = w22_now_ms();
    r->failed = 0; r->err[0] = 0;
    if (!latent || !frames || !To_out || Tl < 1 || hp < 1 || wp < 1) { snprintf(r->err, sizeof r->err, "bad arguments"); return -1; }
    int H = hp * 32, W = wp * 32;
    if (Hc <= 0 || Hc > H) Hc = H;
    if (Wc <= 0 || Wc > W) Wc = W;
    int tHp, tWp;
    w22_tile_dims(r, H, W, 1, &tHp, &tWp);
    int th = tHp / 32, tw = tWp / 32;
    int ovl = (r->overlap_px > 0 ? r->overlap_px : 128) / 32;
    w22_axis ay, ax;
    w22_axis_plan(&ay, hp, th, ovl > th / 2 ? th / 2 : ovl);
    w22_axis_plan(&ax, wp, tw, ovl > tw / 2 ? tw / 2 : ovl);
    int To = 1 + 4 * (Tl - 1);
    int ntiles = ay.n * ax.n;
    float *res = NULL, *acc = NULL, *wacc = NULL;
    for (int iy = 0; iy < ay.n; iy++)
        for (int ix = 0; ix < ax.n; ix++) {
            int y0 = ay.start[iy], x0 = ax.start[ix];
            float *tl_buf = NULL;
            const float *src = latent;
            if (ntiles > 1) {
                tl_buf = (float *)malloc(F4((size_t)192 * Tl * th * tw));
                for (int c = 0; c < 192; c++)
                    for (int t = 0; t < Tl; t++)
                        for (int y = 0; y < th; y++)
                            memcpy(tl_buf + ((((size_t)c * Tl) + t) * th + y) * tw,
                                   latent + ((((size_t)c * Tl) + t) * hp + y0 + y) * wp + x0, F4(tw));
                src = tl_buf;
            }
            int to = 0;
            float *fr = w22_decode_tile(r, src, Tl, th, tw, &to);
            free(tl_buf);
            if (!fr) { free(acc); free(wacc); w22_cleanup_after(r, 0); return -1; }
            if (ntiles == 1) { res = fr; break; }
            int TH = th * 32, TW = tw * 32;
            if (!acc) {
                acc = (float *)calloc((size_t)3 * To * H * W, sizeof(float));
                wacc = (float *)calloc((size_t)H * W, sizeof(float));
            }
            int Y0 = y0 * 32, X0 = x0 * 32;
            /* weights in pixel units: ramp over the (pixel) overlap */
            w22_axis ayp = ay, axp = ax;
            for (int i = 0; i < ayp.n; i++) ayp.start[i] *= 32;
            for (int i = 0; i < axp.n; i++) axp.start[i] *= 32;
            float *wy = (float *)malloc(F4(TH)), *wx = (float *)malloc(F4(TW));
            for (int y = 0; y < TH; y++) wy[y] = w22_axis_w(&ayp, iy, TH, y);
            for (int x = 0; x < TW; x++) wx[x] = w22_axis_w(&axp, ix, TW, x);
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < TH; y++) {
                for (int x = 0; x < TW; x++) wacc[(size_t)(Y0 + y) * W + X0 + x] += wy[y] * wx[x];
                for (int c = 0; c < 3; c++)
                    for (int t = 0; t < To; t++) {
                        float *d = acc + (((size_t)c * To + t) * H + Y0 + y) * W + X0;
                        const float *s = fr + (((size_t)c * To + t) * TH + y) * TW;
                        for (int x = 0; x < TW; x++) d[x] += wy[y] * wx[x] * s[x];
                    }
            }
            free(wy); free(wx); free(fr);
        }
    if (!res) {
        res = acc;
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < 3; c++)
            for (int t = 0; t < To; t++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        res[(((size_t)c * To + t) * H + y) * W + x] /= wacc[(size_t)y * W + x];
        free(wacc);
    }
    if (Hc != H || Wc != W) {
        float *cr = (float *)malloc(F4((size_t)3 * To * Hc * Wc));
        for (int c = 0; c < 3; c++)
            for (int t = 0; t < To; t++)
                for (int y = 0; y < Hc; y++)
                    memcpy(cr + (((size_t)c * To + t) * Hc + y) * Wc, res + (((size_t)c * To + t) * H + y) * W, F4(Wc));
        free(res);
        res = cr;
    }
    w22_cleanup_after(r, 1);
    r->last_ms = w22_now_ms() - t0;
    if (r->verbose >= 2) { fprintf(stderr, "wan22_vae: decode %.1f GFLOP -> %.1f TFLOPS\n", r->flops * 1e-9, r->flops / (r->last_ms * 1e9)); }
    r->flops = 0;
    *frames = res; *To_out = To;
    return 0;
}

int hip_wan22_vae_decode(hip_wan22_vae *r, const float *latent, int Tl, int hp, int wp, float **frames, int *To_out) {
    return hip_wan22_vae_decode_crop(r, latent, Tl, hp, wp, 0, 0, frames, To_out);
}

int hip_wan22_vae_warmup(hip_wan22_vae *r, int H, int W) {
    int Hp = (H + 31) & ~31, Wp = (W + 31) & ~31;
    size_t n = (size_t)3 * 1 * Hp * Wp;
    float *v = (float *)calloc(n, sizeof(float));
    float *lat = NULL, *fr = NULL;
    int Tl = 0, To = 0, rc = 0;
    int keep = r->persistent;
    r->persistent = 1;
    if (hip_wan22_vae_encode(r, v, 1, Hp, Wp, &lat, &Tl)) rc = -1;
    else if (hip_wan22_vae_decode(r, lat, Tl, Hp / 32, Wp / 32, &fr, &To)) rc = -1;
    r->persistent = keep;
    if (!keep) { w22_pool_release(r); w22_free_caches(r); }
    free(v); free(lat); free(fr);
    return rc;
}

void hip_wan22_vae_free(hip_wan22_vae *r) {
    if (!r) return;
    w22_free_weights(r);
    free(r);
}

#endif /* HIP_WAN22_VAE_IMPLEMENTATION */
#endif /* HIP_WAN22_VAE_H */
