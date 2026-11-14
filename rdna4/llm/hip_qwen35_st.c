/*
 * hip_qwen35_st.c - Qwen3.5 text tower (hybrid DeltaNet + gated attention) on RDNA4, direct safetensors.
 * See hip_qwen35_st.h.  All kernels are compiled at run time with HIPRTC (no hipcc at build).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#ifndef HQ35_EXTERNAL_SAFETENSORS
#define SAFETENSORS_IMPLEMENTATION
#endif
#include "../../common/safetensors.h"
#include "hip_qwen35_st.h"

/* ------------------------------------------------------------------------ */
/* Kernel source (stringified; no preprocessor directives / char literals).  */
/* ------------------------------------------------------------------------ */
#define HQ_SRC(...) #__VA_ARGS__

static const char hq35_src[] = HQ_SRC(
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned short u16;

__device__ __forceinline__ float hq_block_sum(float v, float *sm) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sm[w] = v;
    __syncthreads();
    float r = 0.0f;
    int nw = blockDim.x >> 5;
    for (int i = 0; i < nw; i++) r += sm[i];
    return r;
}

/* ---- WMMA f16 GEMM: Y[T,N] (+)= X[T,K](f32->f16) * W[N,K]^T(f16), fp32 accumulate ---- */
template <int BM, int WM, int WN>
__device__ __forceinline__ void hq_gemm_body(float *__restrict__ Y, const u16 *__restrict__ W,
                                             const float *__restrict__ X, int N, int K, int T,
                                             int ldy, int accum) {
    constexpr int FM = BM / WM / 16;
    constexpr int FN = 128 / WN / 16;
    constexpr int AC = BM * 8 / 256;
    __shared__ _Float16 sA[BM * 40];
    __shared__ _Float16 sB[128 * 40];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int half = lane >> 4, idx = lane & 15;
    const int wm = wave % WM, wn = wave / WM;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * 128;
    f32x8 acc[FM][FN];
    for (int f = 0; f < FM; f++)
        for (int g = 0; g < FN; g++)
            for (int i = 0; i < 8; i++) acc[f][g][i] = 0.0f;
    float4 ra[AC];
    uint4 rb[2];
    for (int i = 0; i < AC; i++) {
        int c = tid + i * 256, row = c >> 3, kq = (c & 7) * 4, gm = m0 + row;
        ra[i] = gm < T ? *(const float4 *)(X + (size_t)gm * K + kq) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (int i = 0; i < 2; i++) {
        int c = tid + i * 256, row = c >> 2, kq = (c & 3) * 8, gn = n0 + row;
        rb[i] = gn < N ? *(const uint4 *)(W + (size_t)gn * K + kq) : make_uint4(0, 0, 0, 0);
    }
    for (int k = 0; k < K; k += 32) {
        for (int i = 0; i < AC; i++) {
            int c = tid + i * 256, row = c >> 3, kq = (c & 7) * 4;
            f16x4 h4 = {(_Float16)ra[i].x, (_Float16)ra[i].y, (_Float16)ra[i].z, (_Float16)ra[i].w};
            *(f16x4 *)&sA[row * 40 + kq] = h4;
        }
        for (int i = 0; i < 2; i++) {
            int c = tid + i * 256, row = c >> 2, kq = (c & 3) * 8;
            *(uint4 *)&sB[row * 40 + kq] = rb[i];
        }
        __syncthreads();
        if (k + 32 < K) {
            for (int i = 0; i < AC; i++) {
                int c = tid + i * 256, row = c >> 3, kq = (c & 7) * 4, gm = m0 + row;
                ra[i] = gm < T ? *(const float4 *)(X + (size_t)gm * K + k + 32 + kq) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
            for (int i = 0; i < 2; i++) {
                int c = tid + i * 256, row = c >> 2, kq = (c & 3) * 8, gn = n0 + row;
                rb[i] = gn < N ? *(const uint4 *)(W + (size_t)gn * K + k + 32 + kq) : make_uint4(0, 0, 0, 0);
            }
        }
        for (int kk = 0; kk < 32; kk += 16) {
            f16x8 af[FM], bf[FN];
            for (int f = 0; f < FM; f++)
                af[f] = *(const f16x8 *)&sA[(wm * (BM / WM) + 16 * f + idx) * 40 + kk + half * 8];
            for (int g = 0; g < FN; g++)
                bf[g] = *(const f16x8 *)&sB[(wn * (128 / WN) + 16 * g + idx) * 40 + kk + half * 8];
            for (int f = 0; f < FM; f++)
                for (int g = 0; g < FN; g++)
                    acc[f][g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af[f], bf[g], acc[f][g]);
        }
        __syncthreads();
    }
    for (int f = 0; f < FM; f++)
        for (int g = 0; g < FN; g++) {
            int col = n0 + wn * (128 / WN) + 16 * g + idx;
            if (col >= N) continue;
            for (int i = 0; i < 8; i++) {
                int row = m0 + wm * (BM / WM) + 16 * f + half * 8 + i;
                if (row < T) {
                    float *p = Y + (size_t)row * ldy + col;
                    *p = accum ? (*p + acc[f][g][i]) : acc[f][g][i];
                }
            }
        }
}

extern "C" {

__global__ void __launch_bounds__(256) hq_gemm_128(float *Y, const u16 *W, const float *X, int N, int K, int T, int ldy, int accum) {
    hq_gemm_body<128, 2, 4>(Y, W, X, N, K, T, ldy, accum);
}
__global__ void __launch_bounds__(256) hq_gemm_64(float *Y, const u16 *W, const float *X, int N, int K, int T, int ldy, int accum) {
    hq_gemm_body<64, 2, 4>(Y, W, X, N, K, T, ldy, accum);
}
__global__ void __launch_bounds__(256) hq_gemm_32(float *Y, const u16 *W, const float *X, int N, int K, int T, int ldy, int accum) {
    hq_gemm_body<32, 1, 8>(Y, W, X, N, K, T, ldy, accum);
}

/* weight conversions (device side) */
__global__ void hq_bf16_to_f16(u16 *dst, const u16 *src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int b = ((unsigned int)src[i]) << 16;
    float f = __uint_as_float(b);
    _Float16 h = (_Float16)f;
    dst[i] = *(u16 *)&h;
}
__global__ void hq_f32_to_f16(u16 *dst, const float *src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    _Float16 h = (_Float16)src[i];
    dst[i] = *(u16 *)&h;
}

/* out = rmsnorm(x) * w (w already includes the +1) */
__global__ void hq_rmsnorm(float *out, const float *x, const float *w, int D, float eps) {
    __shared__ float sm[32];
    const float *xr = x + (size_t)blockIdx.x * D;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i] * xr[i];
    s = hq_block_sum(s, sm);
    float inv = rsqrtf(s / (float)D + eps);
    float *o = out + (size_t)blockIdx.x * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) o[i] = xr[i] * inv * w[i];
}

__global__ void hq_silu_mul(float *out, const float *gu, int F, int T) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)F * T) return;
    size_t t = i / F, f = i % F;
    float g = gu[t * 2 * F + f], u = gu[t * 2 * F + F + f];
    out[i] = g / (1.0f + expf(-g)) * u;
}

/* causal depthwise conv1d (k=4) + SiLU, input = qkv slice of proj (row stride ldin), state [3][C] */
__global__ void hq_conv_silu(float *out, const float *state, const float *in, const float *w,
                             int T, int C, int ldin) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= C) return;
    float acc = 0.0f;
    for (int j = 0; j < 4; j++) {
        int s = t - 3 + j;
        float x = s >= 0 ? in[(size_t)s * ldin + c] : state[(3 + s) * C + c];
        acc += w[c * 4 + j] * x;
    }
    out[(size_t)t * C + c] = acc / (1.0f + expf(-acc));
}
__global__ void hq_conv_state(float *state, const float *in, int T, int C, int ldin) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float old[3];
    for (int j = 0; j < 3; j++) old[j] = state[j * C + c];
    for (int j = 0; j < 3; j++) {
        int s = T - 3 + j;
        state[j * C + c] = s >= 0 ? in[(size_t)s * ldin + c] : old[3 + s];
    }
}

/* per (t,head): l2norm q,k (q scaled by 1/sqrt(dk)), beta=sigmoid(b), decay=exp(A*softplus(a+dtb)) */
__global__ void hq_ssm_prep(float *qn, float *kn, float *dec, float *beta, const float *conv,
                            const float *proj, const float *A, const float *dtb,
                            int T, int H, int ldproj, int a_off, int b_off, int key_dim) {
    __shared__ float sm[32];
    int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    int C = 2 * key_dim + H * 128;
    float q = conv[(size_t)t * C + h * 128 + d];
    float k = conv[(size_t)t * C + key_dim + h * 128 + d];
    float sq = hq_block_sum(q * q, sm);
    float sk = hq_block_sum(k * k, sm);
    float iq = rsqrtf(sq + 1e-6f) * 0.08838834764831845f;
    float ik = rsqrtf(sk + 1e-6f);
    qn[((size_t)t * H + h) * 128 + d] = q * iq;
    kn[((size_t)t * H + h) * 128 + d] = k * ik;
    if (d == 0) {
        float a = proj[(size_t)t * ldproj + a_off + h] + dtb[h];
        float b = proj[(size_t)t * ldproj + b_off + h];
        float sp = a > 20.0f ? a : log1pf(expf(a));
        dec[t * H + h] = expf(A[h] * sp);
        beta[t * H + h] = 1.0f / (1.0f + expf(-b));
    }
}

/* gated delta rule recurrence.  one wave per (head, 32-column slice of dv); lane owns a column,
 * state column S[:, j] (128 dk) lives in registers.  S layout [H][dk][dv]. */
__global__ void __launch_bounds__(32) hq_ssm_scan(float *out, float *S, const float *qn, const float *kn,
                                                  const float *dec, const float *beta, const float *conv,
                                                  int T, int H, int key_dim) {
    int h = blockIdx.x >> 2, sl = blockIdx.x & 3, lane = threadIdx.x;
    int j = sl * 32 + lane;
    int C = 2 * key_dim + H * 128;
    float s[128];
    _Pragma("unroll")
    for (int i = 0; i < 128; i++) s[i] = S[((size_t)h * 128 + i) * 128 + j];
    for (int t = 0; t < T; t++) {
        const float4 *k4 = (const float4 *)(kn + ((size_t)t * H + h) * 128);
        const float4 *q4 = (const float4 *)(qn + ((size_t)t * H + h) * 128);
        float d = dec[t * H + h], b = beta[t * H + h];
        float v = conv[(size_t)t * C + 2 * key_dim + h * 128 + j];
        float kv = 0.0f;
        _Pragma("unroll")
        for (int i4 = 0; i4 < 32; i4++) {
            float4 k = k4[i4];
            s[i4 * 4 + 0] *= d; kv += s[i4 * 4 + 0] * k.x;
            s[i4 * 4 + 1] *= d; kv += s[i4 * 4 + 1] * k.y;
            s[i4 * 4 + 2] *= d; kv += s[i4 * 4 + 2] * k.z;
            s[i4 * 4 + 3] *= d; kv += s[i4 * 4 + 3] * k.w;
        }
        float dl = (v - kv) * b;
        float o = 0.0f;
        _Pragma("unroll")
        for (int i4 = 0; i4 < 32; i4++) {
            float4 k = k4[i4], q = q4[i4];
            s[i4 * 4 + 0] += k.x * dl; o += s[i4 * 4 + 0] * q.x;
            s[i4 * 4 + 1] += k.y * dl; o += s[i4 * 4 + 1] * q.y;
            s[i4 * 4 + 2] += k.z * dl; o += s[i4 * 4 + 2] * q.z;
            s[i4 * 4 + 3] += k.w * dl; o += s[i4 * 4 + 3] * q.w;
        }
        out[(size_t)t * (H * 128) + h * 128 + j] = o;
    }
    _Pragma("unroll")
    for (int i = 0; i < 128; i++) S[((size_t)h * 128 + i) * 128 + j] = s[i];
}

/* gated RMSNorm: out = rms(o)*w * silu(z) per (t, head) */
__global__ void hq_ssm_gnorm(float *out, const float *o, const float *proj, const float *w,
                             int ldproj, int z_off, int H, float eps) {
    __shared__ float sm[32];
    int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float v = o[(size_t)t * (H * 128) + h * 128 + d];
    float ss = hq_block_sum(v * v, sm);
    float inv = rsqrtf(ss / 128.0f + eps);
    float z = proj[(size_t)t * ldproj + z_off + h * 128 + d];
    out[(size_t)t * (H * 128) + h * 128 + d] = v * inv * w[d] * (z / (1.0f + expf(-z)));
}

/* full-attention prep: proj row = [q|gate interleaved per head (nh*2*hd) | k (nkv*hd) | v (nkv*hd)] */
__global__ void __launch_bounds__(256) hq_attn_prep(float *qbuf, float *gbuf, float *kc, float *vc,
                            const float *proj, const float *qnw, const float *knw,
                            int T, int start, float eps, float theta, int nh, int nkv, int rot) {
    __shared__ float sm[32];
    __shared__ float sv[256];
    int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    int pos = start + t;
    int ld = nh * 512 + 2 * nkv * 256;
    const float *row = proj + (size_t)t * ld;
    if (hh >= nh + nkv) {                       /* v copy */
        int kvh = hh - nh - nkv;
        vc[(size_t)pos * (nkv * 256) + kvh * 256 + d] = row[nh * 512 + nkv * 256 + kvh * 256 + d];
        return;
    }
    float x;
    const float *nw;
    if (hh < nh) {
        x = row[hh * 512 + d];
        gbuf[(size_t)t * (nh * 256) + hh * 256 + d] = row[hh * 512 + 256 + d];
        nw = qnw;
    } else {
        x = row[nh * 512 + (hh - nh) * 256 + d];
        nw = knw;
    }
    float ss = hq_block_sum(x * x, sm);
    x = x * rsqrtf(ss / 256.0f + eps) * nw[d];
    sv[d] = x;
    __syncthreads();
    int half = rot / 2;
    float r = x;
    if (d < rot) {
        int j = d % half;
        float freq = powf(theta, -(float)(2 * j) / (float)rot);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        if (d < half) r = sv[j] * c - sv[j + half] * s;
        else r = sv[j] * s + sv[j + half] * c;
    }
    if (hh < nh) qbuf[(size_t)t * (nh * 256) + hh * 256 + d] = r;
    else kc[(size_t)pos * (nkv * 256) + (hh - nh) * 256 + d] = r;
}

/* causal GQA attention, head_dim 256, one block per (token, q head), online softmax */
__global__ void __launch_bounds__(256) hq_attn_fwd(float *out, const float *qbuf, const float *gbuf,
                           const float *kc, const float *vc, int T, int start, int nh, int nkv) {
    __shared__ float sc[8];
    int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    int wave = tid >> 5, lane = tid & 31;
    int kvh = h / (nh / nkv);
    int pos = start + t;
    int kvd = nkv * 256;
    float qr[8];
    for (int e = 0; e < 8; e++) qr[e] = qbuf[(size_t)t * (nh * 256) + h * 256 + lane + 32 * e] * 0.0625f;
    float m = -1e30f, l = 0.0f, acc = 0.0f;
    for (int j0 = 0; j0 <= pos; j0 += 8) {
        int j = j0 + wave;
        float dot = 0.0f;
        if (j <= pos) {
            const float *kr = kc + (size_t)j * kvd + kvh * 256;
            for (int e = 0; e < 8; e++) dot += qr[e] * kr[lane + 32 * e];
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor(dot, o);
        } else dot = -1e30f;
        if (lane == 0) sc[wave] = dot;
        __syncthreads();
        float tm = sc[0];
        for (int k = 1; k < 8; k++) tm = fmaxf(tm, sc[k]);
        float mn = fmaxf(m, tm);
        float c = expf(m - mn);
        float lsum = 0.0f, a2 = 0.0f;
        for (int k = 0; k < 8; k++) {
            if (j0 + k <= pos) {
                float p = expf(sc[k] - mn);
                lsum += p;
                a2 += p * vc[(size_t)(j0 + k) * kvd + kvh * 256 + tid];
            }
        }
        l = l * c + lsum;
        acc = acc * c + a2;
        m = mn;
        __syncthreads();
    }
    float g = gbuf[(size_t)t * (nh * 256) + h * 256 + tid];
    out[(size_t)t * (nh * 256) + h * 256 + tid] = acc / l * (1.0f / (1.0f + expf(-g)));
}

}
);

/* ------------------------------------------------------------------------ */
typedef struct {
    int is_ssm;
    void *w_in, *w_out, *w_gu, *w_down;          /* f16 */
    float *norm_in, *norm_post;                  /* f32, +1 applied */
    float *conv_w, *A, *dtb, *gnorm_w;           /* ssm */
    float *conv_state, *rec_state;
    float *qn_w, *kn_w;                          /* attn */
    float *kc, *vc;
    int n_in;                                    /* rows of w_in */
} hq_layer;

struct hip_qwen35 {
    int verbose, device;
    hipStream_t stream;
    hipModule_t module;
    hipFunction_t f_gemm[3], f_bf16_f16, f_f32_f16, f_rms, f_silumul, f_conv, f_convst, f_prep, f_scan,
                  f_gnorm, f_aprep, f_afwd;
    int loaded;
    int n_embd, n_layers, n_ff, n_heads, n_kv, head_dim, max_seq, n_vocab, interval;
    int lk_heads, lk_dim, lv_dim, conv_k, rot;
    float eps, theta;
    int key_dim, val_dim, qkv_dim, n_in_ssm;
    hq_layer *layers;
    float *d_final_norm;
    st_context *embd_shard;
    const uint16_t *host_embd;                    /* bf16 [vocab][n_embd] in the mmap */
    int M;                                        /* max chunk rows */
    float *x, *xn, *proj, *conv_out, *qn, *kn, *dec, *beta, *o, *gn, *gu, *act, *qbuf, *gbuf;
    void *tmp; size_t tmp_bytes;
};

#define CK(call) do { hipError_t _e = (call); if (_e != hipSuccess) { const char *_s = "?"; \
    if (hipGetErrorString) hipGetErrorString(_e, &_s); \
    fprintf(stderr, "hip_qwen35: HIP error %s:%d: %s\n", __FILE__, __LINE__, _s); return -1; } } while (0)

static double hq_now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

hip_qwen35 *hip_qwen35_init(int device_id, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_qwen35: rocew init failed\n"); return NULL;
    }
    if (hipInit(0) != hipSuccess) { fprintf(stderr, "hip_qwen35: hipInit failed\n"); return NULL; }
    hip_qwen35 *r = (hip_qwen35 *)calloc(1, sizeof(*r));
    r->verbose = verbose; r->device = device_id;
    if (hipSetDevice(device_id) != hipSuccess) { free(r); return NULL; }
    if (hipStreamCreate(&r->stream) != hipSuccess) { free(r); return NULL; }
    if (hip_compile_kernels_ex(&r->module, device_id, hq35_src, "hip_qwen35_st", verbose, "hq35", 1) < 0) {
        free(r); return NULL;
    }
#define GF(f, n) do { if (hipModuleGetFunction(&r->f, r->module, n) != hipSuccess) { \
        fprintf(stderr, "hip_qwen35: missing kernel %s\n", n); free(r); return NULL; } } while (0)
    GF(f_gemm[0], "hq_gemm_128"); GF(f_gemm[1], "hq_gemm_64"); GF(f_gemm[2], "hq_gemm_32");
    GF(f_bf16_f16, "hq_bf16_to_f16"); GF(f_f32_f16, "hq_f32_to_f16");
    GF(f_rms, "hq_rmsnorm"); GF(f_silumul, "hq_silu_mul"); GF(f_conv, "hq_conv_silu");
    GF(f_convst, "hq_conv_state"); GF(f_prep, "hq_ssm_prep"); GF(f_scan, "hq_ssm_scan");
    GF(f_gnorm, "hq_ssm_gnorm"); GF(f_aprep, "hq_attn_prep"); GF(f_afwd, "hq_attn_fwd");
#undef GF
    return r;
}

#define LAUNCH(fn, gx, gy, gz, bx, args) \
    hipModuleLaunchKernel(fn, gx, gy, gz, bx, 1, 1, 0, r->stream, args, NULL)

/* ---- safetensors helpers (shards) ---- */
typedef struct { const void *data; const char *dtype; int nd; uint64_t sh[8]; size_t nbytes; } hq_t;

static int hq_open_shards(const char *path, st_context **sh, int maxs) {
    struct stat sb;
    char p[600];
    int n = 0;
    if (stat(path, &sb) != 0) return -1;
    if (!S_ISDIR(sb.st_mode)) { sh[0] = safetensors_open(path); return sh[0] ? 1 : -1; }
    for (int total = 1; total <= maxs && !n; total++) {
        for (int pat = 0; pat < 2 && !n; pat++) {
            snprintf(p, sizeof(p), pat ? "%s/model.safetensors-%05d-of-%05d.safetensors"
                                       : "%s/model-%05d-of-%05d.safetensors", path, 1, total);
            if (stat(p, &sb) != 0) continue;
            for (int j = 1; j <= total; j++) {
                snprintf(p, sizeof(p), pat ? "%s/model.safetensors-%05d-of-%05d.safetensors"
                                           : "%s/model-%05d-of-%05d.safetensors", path, j, total);
                st_context *s = safetensors_open(p);
                if (!s) break;
                sh[n++] = s;
            }
        }
    }
    if (!n) {
        snprintf(p, sizeof(p), "%s/model.safetensors", path);
        if (stat(p, &sb) == 0) { st_context *s = safetensors_open(p); if (s) sh[n++] = s; }
    }
    return n;
}

static int hq_get(st_context **sh, int ns, const char *name, hq_t *t) {
    for (int s = 0; s < ns; s++) {
        int i = safetensors_find(sh[s], name);
        if (i < 0) continue;
        t->data = safetensors_data(sh[s], i);
        t->dtype = safetensors_dtype(sh[s], i);
        t->nd = safetensors_ndims(sh[s], i);
        const uint64_t *shp = safetensors_shape(sh[s], i);
        for (int d = 0; d < t->nd && d < 8; d++) t->sh[d] = shp[d];
        t->nbytes = safetensors_nbytes(sh[s], i);
        return 0;
    }
    fprintf(stderr, "hip_qwen35: missing tensor %s\n", name);
    return -1;
}

static float hq_bf16(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static float hq_f16(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float f;
    if (e == 0) f = ldexpf((float)m, -24);
    else if (e == 31) f = m ? NAN : INFINITY;
    else f = ldexpf((float)(m + 1024), (int)e - 25);
    return s ? -f : f;
}

static size_t hq_numel(const hq_t *t) { size_t n = 1; for (int d = 0; d < t->nd; d++) n *= t->sh[d]; return n; }

/* host -> float vector with mode 0 plain, 1 (+1), 2 (-exp) */
static int hq_upload_vec(float **dst, st_context **sh, int ns, const char *name, int n, int mode) {
    hq_t t;
    if (hq_get(sh, ns, name, &t)) return -1;
    if ((int)hq_numel(&t) != n) { fprintf(stderr, "hip_qwen35: %s numel %zu != %d\n", name, hq_numel(&t), n); return -1; }
    float *f = (float *)malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) {
        float v;
        if (!strcmp(t.dtype, "BF16")) v = hq_bf16(((const uint16_t *)t.data)[i]);
        else if (!strcmp(t.dtype, "F16")) v = hq_f16(((const uint16_t *)t.data)[i]);
        else v = ((const float *)t.data)[i];
        f[i] = mode == 1 ? v + 1.0f : mode == 2 ? -expf(v) : v;
    }
    hipError_t e = hipMalloc((void **)dst, (size_t)n * 4);
    if (e == hipSuccess) e = hipMemcpy(*dst, f, (size_t)n * 4, hipMemcpyHostToDevice);
    free(f);
    return e == hipSuccess ? 0 : -1;
}

/* upload tensor rows into f16 device buffer dst at element offset off; returns rows via *rows */
static int hq_upload_mat(hip_qwen35 *r, void *dst, size_t off_elems, st_context **sh, int ns,
                         const char *name, int expect_cols, int *rows) {
    hq_t t;
    if (hq_get(sh, ns, name, &t)) return -1;
    if (t.nd != 2 || (expect_cols && (int)t.sh[1] != expect_cols)) {
        fprintf(stderr, "hip_qwen35: %s bad shape\n", name); return -1;
    }
    size_t n = hq_numel(&t);
    *rows = (int)t.sh[0];
    char *d = (char *)dst + off_elems * 2;
    int is_bf = !strcmp(t.dtype, "BF16"), is_h = !strcmp(t.dtype, "F16");
    if (is_h) { CK(hipMemcpy(d, t.data, n * 2, hipMemcpyHostToDevice)); return 0; }
    size_t eb = is_bf ? 2 : 4;
    if (r->tmp_bytes < n * eb) {
        if (r->tmp) hipFree(r->tmp);
        r->tmp_bytes = n * eb;
        CK(hipMalloc(&r->tmp, r->tmp_bytes));
    }
    CK(hipMemcpy(r->tmp, t.data, n * eb, hipMemcpyHostToDevice));
    unsigned grid = (unsigned)((n + 255) / 256);
    void *a[] = { &d, &r->tmp, &n };
    hipFunction_t f = is_bf ? r->f_bf16_f16 : r->f_f32_f16;
    hipModuleLaunchKernel(f, grid, 1, 1, 256, 1, 1, 0, r->stream, a, NULL);
    CK(hipStreamSynchronize(r->stream));
    return 0;
}

static double hq_cfg_num(const json_val *o, const char *k, double d) {
    if (!o || o->type != JSON_OBJECT) return d;
    const json_val *v = json_obj_get(o, k);
    return (v && v->type == JSON_NUMBER) ? v->num : d;
}

int hip_qwen35_load_safetensors(hip_qwen35 *r, const char *model_path, int max_seq_len) {
    if (!r || !model_path) return -1;
    st_context *sh[16] = {0};
    int ns = hq_open_shards(model_path, sh, 16);
    if (ns <= 0) { fprintf(stderr, "hip_qwen35: cannot open safetensors in %s\n", model_path); return -1; }
    const char *LM = "model.language_model.";
    char name[300];
    float rms_eps = 1e-6f, theta = 1e7f, partial = 0.25f;
    int head_dim = 256, n_heads = 8, n_kv = 2, interval = 4, conv_k = 4;
    int lk_heads = 16, lv_heads = 16, lk_dim = 128, lv_dim = 128;
    {
        char cp[600]; snprintf(cp, sizeof(cp), "%s/config.json", model_path);
        FILE *f = fopen(cp, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            char *buf = (char *)malloc(sz > 0 ? sz : 1);
            if (buf && sz > 0 && fread(buf, 1, sz, f) == (size_t)sz) {
                json_val *root = json_parse(buf, (int)sz);
                json_val *tc = root ? json_obj_get(root, "text_config") : NULL;
                if (!tc) tc = root;
                if (tc) {
                    rms_eps = (float)hq_cfg_num(tc, "rms_norm_eps", rms_eps);
                    head_dim = (int)hq_cfg_num(tc, "head_dim", head_dim);
                    n_heads = (int)hq_cfg_num(tc, "num_attention_heads", n_heads);
                    n_kv = (int)hq_cfg_num(tc, "num_key_value_heads", n_kv);
                    interval = (int)hq_cfg_num(tc, "full_attention_interval", interval);
                    conv_k = (int)hq_cfg_num(tc, "linear_conv_kernel_dim", conv_k);
                    lk_heads = (int)hq_cfg_num(tc, "linear_num_key_heads", lk_heads);
                    lv_heads = (int)hq_cfg_num(tc, "linear_num_value_heads", lv_heads);
                    lk_dim = (int)hq_cfg_num(tc, "linear_key_head_dim", lk_dim);
                    lv_dim = (int)hq_cfg_num(tc, "linear_value_head_dim", lv_dim);
                    json_val *rp = json_obj_get(tc, "rope_parameters");
                    if (rp && rp->type == JSON_OBJECT) {
                        theta = (float)hq_cfg_num(rp, "rope_theta", theta);
                        partial = (float)hq_cfg_num(rp, "partial_rotary_factor", partial);
                    }
                }
                if (root) json_free(root);
            }
            free(buf); fclose(f);
        }
    }
    if (lk_heads != lv_heads || lk_dim != 128 || lv_dim != 128 || head_dim != 256 || conv_k != 4 || n_heads / n_kv * n_kv != n_heads) {
        fprintf(stderr, "hip_qwen35: unsupported config (need k_heads==v_heads, dk=dv=128, head_dim=256, conv_k=4)\n");
        goto fail;
    }
    int n_layers = 0;
    for (int s = 0; s < ns; s++)
        for (int i = 0; i < sh[s]->n_tensors; i++) {
            const char *nm = safetensors_name(sh[s], i);
            if (strncmp(nm, "model.language_model.layers.", 28) == 0) { int l = atoi(nm + 28); if (l + 1 > n_layers) n_layers = l + 1; }
        }
    if (n_layers <= 0) { fprintf(stderr, "hip_qwen35: no layers\n"); goto fail; }
    hq_t embd, ff0;
    snprintf(name, sizeof(name), "%sembed_tokens.weight", LM);
    if (hq_get(sh, ns, name, &embd)) goto fail;
    snprintf(name, sizeof(name), "%slayers.0.mlp.gate_proj.weight", LM);
    if (hq_get(sh, ns, name, &ff0)) goto fail;
    if (strcmp(embd.dtype, "BF16") != 0) { fprintf(stderr, "hip_qwen35: embed must be bf16\n"); goto fail; }

    r->n_embd = (int)embd.sh[1]; r->n_vocab = (int)embd.sh[0]; r->n_layers = n_layers;
    r->n_ff = (int)ff0.sh[0]; r->n_heads = n_heads; r->n_kv = n_kv; r->head_dim = head_dim;
    r->eps = rms_eps; r->theta = theta; r->rot = (int)(partial * head_dim); r->interval = interval;
    r->lk_heads = lk_heads; r->lk_dim = lk_dim; r->lv_dim = lv_dim; r->conv_k = conv_k;
    r->key_dim = lk_heads * lk_dim;
    r->val_dim = lv_heads * lv_dim;
    r->qkv_dim = 2 * r->key_dim + r->val_dim;
    r->n_in_ssm = r->qkv_dim + r->val_dim + 2 * lv_heads;
    if (max_seq_len <= 0) max_seq_len = 2048;
    r->max_seq = max_seq_len;
    r->M = 1024;
    if (r->verbose)
        fprintf(stderr, "hip_qwen35: n_embd=%d layers=%d heads=%d/%d hd=%d ff=%d rot=%d theta=%.0f ssm qkv=%d\n",
                r->n_embd, n_layers, n_heads, n_kv, head_dim, r->n_ff, r->rot, theta, r->qkv_dim);

    /* host-gathered embeddings: keep the first shard mapped */
    {
        snprintf(name, sizeof(name), "%sembed_tokens.weight", LM);
        for (int s = 0; s < ns; s++) if (safetensors_find(sh[s], name) >= 0) {
            r->host_embd = (const uint16_t *)safetensors_data(sh[s], safetensors_find(sh[s], name));
            r->embd_shard = sh[s]; break;
        }
    }
    snprintf(name, sizeof(name), "%snorm.weight", LM);
    if (hq_upload_vec(&r->d_final_norm, sh, ns, name, r->n_embd, 1)) goto fail;

    r->layers = (hq_layer *)calloc(n_layers, sizeof(hq_layer));
    int D = r->n_embd, F = r->n_ff;
    int n_in_attn = n_heads * 2 * head_dim + 2 * n_kv * head_dim;
    for (int l = 0; l < n_layers; l++) {
        hq_layer *L = &r->layers[l];
        L->is_ssm = (interval > 0 && (l + 1) % interval != 0);
        int rows;
#define VEC(field, suffix, n, mode) do { snprintf(name, sizeof(name), "%slayers.%d." suffix, LM, l); \
            if (hq_upload_vec(&L->field, sh, ns, name, n, mode)) goto fail; } while (0)
#define MAT(dst, off, suffix, cols) do { snprintf(name, sizeof(name), "%slayers.%d." suffix, LM, l); \
            if (hq_upload_mat(r, dst, off, sh, ns, name, cols, &rows)) goto fail; } while (0)
        VEC(norm_in, "input_layernorm.weight", D, 1);
        VEC(norm_post, "post_attention_layernorm.weight", D, 1);
        if (L->is_ssm) {
            hipMalloc(&L->w_in, (size_t)r->n_in_ssm * D * 2);
            MAT(L->w_in, 0, "linear_attn.in_proj_qkv.weight", D);
            if (rows != r->qkv_dim) { fprintf(stderr, "hip_qwen35: qkv rows %d != %d\n", rows, r->qkv_dim); goto fail; }
            MAT(L->w_in, (size_t)r->qkv_dim * D, "linear_attn.in_proj_z.weight", D);
            MAT(L->w_in, (size_t)(r->qkv_dim + r->val_dim) * D, "linear_attn.in_proj_a.weight", D);
            MAT(L->w_in, (size_t)(r->qkv_dim + r->val_dim + lv_heads) * D, "linear_attn.in_proj_b.weight", D);
            hipMalloc(&L->w_out, (size_t)D * r->val_dim * 2);
            MAT(L->w_out, 0, "linear_attn.out_proj.weight", r->val_dim);
            VEC(A, "linear_attn.A_log", lv_heads, 2);
            VEC(dtb, "linear_attn.dt_bias", lv_heads, 0);
            VEC(conv_w, "linear_attn.conv1d.weight", r->qkv_dim * conv_k, 0);
            VEC(gnorm_w, "linear_attn.norm.weight", lv_dim, 0);
            hipMalloc((void **)&L->conv_state, (size_t)(conv_k - 1) * r->qkv_dim * 4);
            hipMalloc((void **)&L->rec_state, (size_t)lv_heads * lk_dim * lv_dim * 4);
        } else {
            hipMalloc(&L->w_in, (size_t)n_in_attn * D * 2);
            MAT(L->w_in, 0, "self_attn.q_proj.weight", D);
            MAT(L->w_in, (size_t)n_heads * 2 * head_dim * D, "self_attn.k_proj.weight", D);
            MAT(L->w_in, (size_t)(n_heads * 2 * head_dim + n_kv * head_dim) * D, "self_attn.v_proj.weight", D);
            hipMalloc(&L->w_out, (size_t)D * n_heads * head_dim * 2);
            MAT(L->w_out, 0, "self_attn.o_proj.weight", n_heads * head_dim);
            VEC(qn_w, "self_attn.q_norm.weight", head_dim, 1);
            VEC(kn_w, "self_attn.k_norm.weight", head_dim, 1);
            size_t kvb = (size_t)r->max_seq * n_kv * head_dim * 4;
            hipMalloc((void **)&L->kc, kvb);
            hipMalloc((void **)&L->vc, kvb);
        }
        hipMalloc(&L->w_gu, (size_t)2 * F * D * 2);
        MAT(L->w_gu, 0, "mlp.gate_proj.weight", D);
        MAT(L->w_gu, (size_t)F * D, "mlp.up_proj.weight", D);
        hipMalloc(&L->w_down, (size_t)D * F * 2);
        MAT(L->w_down, 0, "mlp.down_proj.weight", F);
#undef VEC
#undef MAT
    }
    /* activation buffers */
    {
        size_t M = r->M;
        int proj_n = r->n_in_ssm > n_in_attn ? r->n_in_ssm : n_in_attn;
#define AL(p, n) do { if (hipMalloc((void **)&r->p, (size_t)(n) * 4) != hipSuccess) goto fail; } while (0)
        AL(x, M * D); AL(xn, M * D); AL(proj, M * proj_n); AL(conv_out, M * r->qkv_dim);
        AL(qn, M * r->key_dim); AL(kn, M * r->key_dim); AL(dec, M * lv_heads); AL(beta, M * lv_heads);
        AL(o, M * r->val_dim); AL(gn, M * (r->val_dim > n_heads * head_dim ? r->val_dim : n_heads * head_dim));
        AL(gu, M * 2 * F); AL(act, M * F); AL(qbuf, M * n_heads * head_dim); AL(gbuf, M * n_heads * head_dim);
#undef AL
    }
    for (int s = 0; s < ns; s++) if (sh[s] && sh[s] != r->embd_shard) safetensors_close(sh[s]);
    r->loaded = 1;
    if (r->tmp) { hipFree(r->tmp); r->tmp = NULL; r->tmp_bytes = 0; }
    return 0;
fail:
    for (int s = 0; s < ns; s++) if (sh[s]) safetensors_close(sh[s]);
    return -1;
}

static int hq_gemm(hip_qwen35 *r, float *Y, const void *W, const float *X, int N, int K, int T, int accum) {
    int gx = (N + 127) / 128;
    int bm = 128, v = 0;
    const int tiles[3] = {128, 64, 32};
    for (int i = 0; i < 3; i++) {
        int blocks = gx * ((T + tiles[i] - 1) / tiles[i]);
        bm = tiles[i]; v = i;
        if (blocks >= 128) break;
    }
    int gy = (T + bm - 1) / bm;
    int ldy = N;
    void *a[] = { &Y, &W, &X, &N, &K, &T, &ldy, &accum };
    hipError_t e = hipModuleLaunchKernel(r->f_gemm[v], gx, gy, 1, 256, 1, 1, 0, r->stream, a, NULL);
    return e == hipSuccess ? 0 : -1;
}

static int hq_run_chunk(hip_qwen35 *r, const int32_t *ids, int n, int n_run, int start_pos, int dump_only,
                        float *out_layers, int n_total, int c0, float *out_normed) {
    int D = r->n_embd, F = r->n_ff, H = r->lv_dim ? r->lk_heads : 16;
    int nh = r->n_heads, nkv = r->n_kv, hd = r->head_dim;
    float eps = r->eps;
    /* embeddings */
    float *emb = (float *)malloc((size_t)n * D * 4);
    for (int t = 0; t < n; t++) {
        int id = ids[t];
        if (id < 0 || id >= r->n_vocab) { free(emb); fprintf(stderr, "hip_qwen35: token id %d out of range\n", id); return -1; }
        const uint16_t *row = r->host_embd + (size_t)id * D;
        uint32_t *o = (uint32_t *)(emb + (size_t)t * D);
        for (int i = 0; i < D; i++) o[i] = (uint32_t)row[i] << 16;
    }
    hipError_t he = hipMemcpyAsync(r->x, emb, (size_t)n * D * 4, hipMemcpyHostToDevice, r->stream);
    if (he == hipSuccess) he = hipStreamSynchronize(r->stream);
    free(emb);
    if (he != hipSuccess) return -1;
    int proj_ld;
#define DUMP(slot) do { \
        if (out_layers && (dump_only < 0 || dump_only == (slot))) { \
            float *dst = out_layers + (dump_only < 0 ? (size_t)(slot) * n_total * D : 0) + (size_t)c0 * D; \
            if (hipMemcpy(dst, r->x, (size_t)n * D * 4, hipMemcpyDeviceToHost) != hipSuccess) return -1; } } while (0)
    for (int l = 0; l < n_run; l++) {
        hq_layer *L = &r->layers[l];
        if (out_layers && (dump_only < 0 || dump_only == l)) { hipStreamSynchronize(r->stream); DUMP(l); }
        { void *a[] = { &r->xn, &r->x, &L->norm_in, &D, &eps }; LAUNCH(r->f_rms, n, 1, 1, 256, a); }
        if (L->is_ssm) {
            proj_ld = r->n_in_ssm;
            if (hq_gemm(r, r->proj, L->w_in, r->xn, proj_ld, D, n, 0)) return -1;
            int C = r->qkv_dim;
            { void *a[] = { &r->conv_out, &L->conv_state, &r->proj, &L->conv_w, &n, &C, &proj_ld };
              LAUNCH(r->f_conv, (C + 255) / 256, n, 1, 256, a); }
            { void *a[] = { &L->conv_state, &r->proj, &n, &C, &proj_ld };
              LAUNCH(r->f_convst, (C + 255) / 256, 1, 1, 256, a); }
            int a_off = r->qkv_dim + r->val_dim, b_off = a_off + H, z_off = r->qkv_dim, kd = r->key_dim;
            { void *a[] = { &r->qn, &r->kn, &r->dec, &r->beta, &r->conv_out, &r->proj, &L->A, &L->dtb,
                            &n, &H, &proj_ld, &a_off, &b_off, &kd };
              LAUNCH(r->f_prep, H, n, 1, 128, a); }
            { void *a[] = { &r->o, &L->rec_state, &r->qn, &r->kn, &r->dec, &r->beta, &r->conv_out, &n, &H, &kd };
              LAUNCH(r->f_scan, H * 4, 1, 1, 32, a); }
            { void *a[] = { &r->gn, &r->o, &r->proj, &L->gnorm_w, &proj_ld, &z_off, &H, &eps };
              LAUNCH(r->f_gnorm, H, n, 1, 128, a); }
            if (hq_gemm(r, r->x, L->w_out, r->gn, D, r->val_dim, n, 1)) return -1;
        } else {
            proj_ld = nh * 2 * hd + 2 * nkv * hd;
            if (hq_gemm(r, r->proj, L->w_in, r->xn, proj_ld, D, n, 0)) return -1;
            int rot = r->rot; float th = r->theta;
            { void *a[] = { &r->qbuf, &r->gbuf, &L->kc, &L->vc, &r->proj, &L->qn_w, &L->kn_w,
                            &n, &start_pos, &eps, &th, &nh, &nkv, &rot };
              LAUNCH(r->f_aprep, n, nh + 2 * nkv, 1, 256, a); }
            { void *a[] = { &r->gn, &r->qbuf, &r->gbuf, &L->kc, &L->vc, &n, &start_pos, &nh, &nkv };
              LAUNCH(r->f_afwd, n, nh, 1, 256, a); }
            if (hq_gemm(r, r->x, L->w_out, r->gn, D, nh * hd, n, 1)) return -1;
        }
        { void *a[] = { &r->xn, &r->x, &L->norm_post, &D, &eps }; LAUNCH(r->f_rms, n, 1, 1, 256, a); }
        if (hq_gemm(r, r->gu, L->w_gu, r->xn, 2 * F, D, n, 0)) return -1;
        { size_t tot = (size_t)n * F; unsigned g = (unsigned)((tot + 255) / 256);
          void *a[] = { &r->act, &r->gu, &F, &n }; LAUNCH(r->f_silumul, g, 1, 1, 256, a); }
        if (hq_gemm(r, r->x, L->w_down, r->act, D, F, n, 1)) return -1;
    }
    hipStreamSynchronize(r->stream);
    DUMP(n_run);
    if (out_normed) {
        { void *a[] = { &r->xn, &r->x, &r->d_final_norm, &D, &eps }; LAUNCH(r->f_rms, n, 1, 1, 256, a); }
        if (hipMemcpy(out_normed + (size_t)c0 * D, r->xn, (size_t)n * D * 4, hipMemcpyDeviceToHost) != hipSuccess) return -1;
    }
    hipError_t e = hipStreamSynchronize(r->stream);
    if (e != hipSuccess || hipGetLastError() != hipSuccess) { fprintf(stderr, "hip_qwen35: kernel error\n"); return -1; }
    return 0;
#undef DUMP
}

int hip_qwen35_prefill_hidden_ex(hip_qwen35 *r, const int32_t *ids, int n, int n_run_layers, int start_pos,
                                 int dump_only_slot, float *out_per_layer, float *out_normed) {
    if (!r || !r->loaded || !ids || n <= 0 || start_pos < 0) return -1;
    if (start_pos + n > r->max_seq) {
        fprintf(stderr, "hip_qwen35: %d tokens at pos %d exceeds max_seq_len %d\n", n, start_pos, r->max_seq);
        return -1;
    }
    int n_run = (n_run_layers > 0 && n_run_layers < r->n_layers) ? n_run_layers : r->n_layers;
    if (start_pos == 0) {
        for (int l = 0; l < r->n_layers; l++) {
            hq_layer *L = &r->layers[l];
            if (!L->is_ssm) continue;
            hipMemsetAsync(L->conv_state, 0, (size_t)(r->conv_k - 1) * r->qkv_dim * 4, r->stream);
            hipMemsetAsync(L->rec_state, 0, (size_t)r->lv_dim * r->lk_dim * r->lk_heads * 4, r->stream);
        }
    }
    for (int c0 = 0; c0 < n; c0 += r->M) {
        int m = n - c0 < r->M ? n - c0 : r->M;
        if (hq_run_chunk(r, ids + c0, m, n_run, start_pos + c0, dump_only_slot, out_per_layer, n, c0, out_normed))
            return -1;
    }
    return 0;
}

int hip_qwen35_cond(hip_qwen35 *r, const int32_t *ids, int n, float *out) {
    return hip_qwen35_prefill_hidden_ex(r, ids, n, 22, 0, 22, out, NULL);
}

int hip_qwen35_n_embd(const hip_qwen35 *r) { return r ? r->n_embd : 0; }
int hip_qwen35_n_layers(const hip_qwen35 *r) { return r ? r->n_layers : 0; }
int hip_qwen35_max_seq_len(const hip_qwen35 *r) { return r ? r->max_seq : 0; }

void hip_qwen35_free(hip_qwen35 *r) {
    if (!r) return;
    for (int l = 0; l < r->n_layers && r->layers; l++) {
        hq_layer *L = &r->layers[l];
        void *p[] = { L->w_in, L->w_out, L->w_gu, L->w_down, L->norm_in, L->norm_post, L->conv_w, L->A, L->dtb,
                      L->gnorm_w, L->conv_state, L->rec_state, L->qn_w, L->kn_w, L->kc, L->vc };
        for (unsigned i = 0; i < sizeof(p) / sizeof(p[0]); i++) if (p[i]) hipFree(p[i]);
    }
    free(r->layers);
    float *b[] = { r->d_final_norm, r->x, r->xn, r->proj, r->conv_out, r->qn, r->kn, r->dec, r->beta, r->o, r->gn,
                   r->gu, r->act, r->qbuf, r->gbuf };
    for (unsigned i = 0; i < sizeof(b) / sizeof(b[0]); i++) if (b[i]) hipFree(b[i]);
    if (r->tmp) hipFree(r->tmp);
    if (r->embd_shard) safetensors_close(r->embd_shard);
    if (r->stream) hipStreamDestroy(r->stream);
    free(r);
}
