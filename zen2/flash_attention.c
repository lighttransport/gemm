/*
 * FlashAttention-1 with online softmax — Zen2 AVX2 implementation.
 *
 * Uses the existing 6×16 GEMM microkernels and packing routines.
 * Tile sizes: Br=48 (8×MR=6), Bc=64 (4×NR=16).
 *
 * Working set per Q-tile (~108 KB for d=128) fits in L2 (512 KB).
 */

#include "flash_attention.h"
#include "gemm.h"
#include "gemm_pack.h"

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/mman.h>

#define BR 48   /* 8 * MR */
#define BC 64   /* 4 * NR */

/* ------------------------------------------------------------------ */
/*  AVX2 helpers                                                       */
/* ------------------------------------------------------------------ */

static inline float hmax_avx2(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m  = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

static inline float hsum_avx2(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/*
 * Fast vectorized exp(x) using 2^(x * log2e) decomposition.
 * 4th-order polynomial for 2^f, bit manipulation for 2^n.
 * Relative error ~1e-6. Input clamped to [-88, 88].
 */
static inline __m256 fast_exp_avx2(__m256 x)
{
    const __m256 lo  = _mm256_set1_ps(-88.0f);
    const __m256 hi  = _mm256_set1_ps(88.0f);
    x = _mm256_max_ps(x, lo);
    x = _mm256_min_ps(x, hi);

    const __m256 log2e = _mm256_set1_ps(1.4426950409f);
    __m256 t = _mm256_mul_ps(x, log2e);

    __m256 n = _mm256_floor_ps(t);
    __m256 f = _mm256_sub_ps(t, n);

    /* 2^f via 4th-order minimax polynomial on [0, 1) */
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931472f);
    const __m256 c2 = _mm256_set1_ps(0.2402265f);
    const __m256 c3 = _mm256_set1_ps(0.05550411f);
    const __m256 c4 = _mm256_set1_ps(0.009618129f);

    __m256 p = _mm256_fmadd_ps(c4, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);

    /* 2^n via bit manipulation: float(2^n) = (n+127) << 23 */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);

    return _mm256_mul_ps(pow2n, p);
}

/* ------------------------------------------------------------------ */
/*  Custom V packing for GEMM 2                                        */
/* ------------------------------------------------------------------ */

/*
 * Pack a column block of V for GEMM 2: O_tile += P × V_tile.
 *
 * Produces B_pack format: V_pack[k * NR + n] = V[(jc+k)*d + col + n]
 * for k in [0, bc), n in [0, nc).  Zero-pads if nc < NR.
 *
 * For full tiles (nc=16): 2 AVX2 loads + stores per k-step (no transpose).
 */
static void pack_v(const float *V, int d, float *V_pack,
                   int bc, int nc, int jc, int col)
{
    if (nc == NR) {
        for (int k = 0; k < bc; k++) {
            const float *src = V + (jc + k) * d + col;
            float *dst = V_pack + k * NR;
            _mm256_store_ps(dst,     _mm256_loadu_ps(src));
            _mm256_store_ps(dst + 8, _mm256_loadu_ps(src + 8));
        }
    } else {
        for (int k = 0; k < bc; k++) {
            const float *src = V + (jc + k) * d + col;
            float *dst = V_pack + k * NR;
            int n;
            for (n = 0; n < nc; n++)
                dst[n] = src[n];
            for (; n < NR; n++)
                dst[n] = 0.0f;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Scalar edge kernels                                                */
/* ------------------------------------------------------------------ */

/* C = A_pack × B_pack (overwrite), for edge tiles where mr/nr don't
 * match any assembly kernel. */
static void gemm_edge_overwrite(const float *A_pack, const float *B_pack,
                                float *C, int ldc, int mr, int nr, int K)
{
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            C[i * ldc + j] = 0.0f;
    for (int k = 0; k < K; k++)
        for (int i = 0; i < mr; i++) {
            float a = A_pack[k * MR + i];
            for (int j = 0; j < nr; j++)
                C[i * ldc + j] += a * B_pack[k * NR + j];
        }
}

/* C += A_pack × B_pack (accumulate), for GEMM 2 edge tiles. */
static void gemm_edge_accum(const float *A_pack, const float *B_pack,
                            float *C, int ldc, int mr, int nr, int K)
{
    for (int k = 0; k < K; k++)
        for (int i = 0; i < mr; i++) {
            float a = A_pack[k * MR + i];
            for (int j = 0; j < nr; j++)
                C[i * ldc + j] += a * B_pack[k * NR + j];
        }
}

/* ------------------------------------------------------------------ */
/*  Online softmax with rescaling                                      */
/* ------------------------------------------------------------------ */

/*
 * For each row i of S[br × BC]:
 *   1. Find m_new = max(S[i, 0..bc-1] * scale)
 *   2. m_global = max(m_prev[i], m_new)
 *   3. correction = exp(m_prev[i] - m_global)
 *   4. Rescale O_tile row: O_tile[i,:] *= correction
 *   5. l[i] *= correction
 *   6. P[i,j] = exp(S[i,j]*scale - m_global), overwrite S in-place
 *   7. l[i] += sum(P[i,:])
 *   8. m_prev[i] = m_global
 */
static void online_softmax_rescale(
    float *S, float *O_tile, float *m_prev, float *l_arr,
    int br, int bc, int d, float scale)
{
    __m256 vscale = _mm256_set1_ps(scale);

    /* bc_vec: largest multiple of 8 <= bc (for AVX2 main loop) */
    int bc_vec = bc & ~7;

    for (int i = 0; i < br; i++) {
        float *Si = S + i * BC;   /* stride is always BC */
        float *Oi = O_tile + i * d;

        /* --- Step 1: row max of S*scale --- */
        __m256 vmax;
        if (bc_vec > 0) {
            vmax = _mm256_mul_ps(vscale, _mm256_load_ps(Si));
            for (int j = 8; j < bc_vec; j += 8)
                vmax = _mm256_max_ps(vmax,
                    _mm256_mul_ps(vscale, _mm256_load_ps(Si + j)));
        } else {
            vmax = _mm256_set1_ps(-FLT_MAX);
        }
        float m_new = hmax_avx2(vmax);
        /* Scalar tail for bc not multiple of 8 */
        for (int j = bc_vec; j < bc; j++) {
            float v = Si[j] * scale;
            if (v > m_new) m_new = v;
        }

        /* --- Step 2: update running state --- */
        float m_global = fmaxf(m_prev[i], m_new);
        float correction;
        if (m_prev[i] == -FLT_MAX)
            correction = 0.0f;   /* first iteration: O_tile is 0, skip rescale */
        else
            correction = expf(m_prev[i] - m_global);

        /* --- Step 3: rescale O row (AVX2, d always multiple of 8) --- */
        __m256 vcorr = _mm256_set1_ps(correction);
        for (int j = 0; j < d; j += 8)
            _mm256_store_ps(Oi + j,
                _mm256_mul_ps(vcorr, _mm256_load_ps(Oi + j)));
        l_arr[i] *= correction;

        /* --- Step 4: P = exp(S*scale - m_global), accumulate sum --- */
        __m256 vmg  = _mm256_set1_ps(m_global);
        __m256 vsum = _mm256_setzero_ps();
        for (int j = 0; j < bc_vec; j += 8) {
            __m256 vs = _mm256_load_ps(Si + j);
            __m256 vp = fast_exp_avx2(
                _mm256_sub_ps(_mm256_mul_ps(vscale, vs), vmg));
            _mm256_store_ps(Si + j, vp);
            vsum = _mm256_add_ps(vsum, vp);
        }
        float row_sum = hsum_avx2(vsum);
        /* Scalar tail */
        for (int j = bc_vec; j < bc; j++) {
            float p = expf(Si[j] * scale - m_global);
            Si[j] = p;
            row_sum += p;
        }
        l_arr[i] += row_sum;
        m_prev[i] = m_global;
    }
}

/* ------------------------------------------------------------------ */
/*  GEMM 1 macro-kernel: S[br,bc] = Q_pack × K_pack^T                 */
/* ------------------------------------------------------------------ */

static void macrokernel_gemm1(
    const float *Q_pack, const float *K_pack,
    float *S, int br, int bc, int K_gemm)
{
    int64_t ldc_bytes = (int64_t)BC * (int64_t)sizeof(float);

    for (int jr = 0; jr < bc; jr += NR) {
        int nr = (jr + NR <= bc) ? NR : (bc - jr);
        for (int ir = 0; ir < br; ir += MR) {
            int mr = (ir + MR <= br) ? MR : (br - ir);

            const float *A_tile = Q_pack + ir * K_gemm;
            const float *B_tile = K_pack + jr * K_gemm;
            float *C_tile = S + ir * BC + jr;

            if (mr == MR && nr == NR) {
                /* Full tile with A_next prefetch */
                const float *A_next;
                if (ir + MR < br)
                    A_next = Q_pack + (ir + MR) * K_gemm;
                else
                    A_next = Q_pack + ir * K_gemm; /* self */
                gemm_kernel_6x16_pf(A_tile, B_tile, C_tile,
                                    K_gemm, ldc_bytes, A_next);
            } else if (mr == 4 && nr == NR) {
                gemm_kernel_4x16(A_tile, B_tile, C_tile,
                                 K_gemm, ldc_bytes);
            } else if (mr == 2 && nr == NR) {
                gemm_kernel_2x16(A_tile, B_tile, C_tile,
                                 K_gemm, ldc_bytes);
            } else {
                gemm_edge_overwrite(A_tile, B_tile, C_tile,
                                    BC, mr, nr, K_gemm);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  GEMM 2 macro-kernel: O_tile[br,d] += P_pack × V_pack              */
/* ------------------------------------------------------------------ */

static void macrokernel_gemm2(
    const float *P_pack, const float *V, int d,
    float *O_tile, int br, int bc, int jc)
{
    int64_t ldc_bytes = (int64_t)d * (int64_t)sizeof(float);

    /* Temporary V_pack buffer — one NR-wide column panel at a time.
     * Size: bc * NR floats = 64 * 16 * 4 = 4 KB. */
    float V_pack[BC * NR] __attribute__((aligned(64)));

    for (int col = 0; col < d; col += NR) {
        int nc = (col + NR <= d) ? NR : (d - col);

        pack_v(V, d, V_pack, bc, nc, jc, col);

        for (int ir = 0; ir < br; ir += MR) {
            int mr = (ir + MR <= br) ? MR : (br - ir);

            const float *A_tile = P_pack + ir * bc;
            float *C_tile = O_tile + ir * d + col;

            if (mr == MR && nc == NR) {
                gemm_kernel_6x16_accum(A_tile, V_pack, C_tile,
                                       bc, ldc_bytes);
            } else {
                gemm_edge_accum(A_tile, V_pack, C_tile,
                                d, mr, nc, bc);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Persistent working buffers                                         */
/* ------------------------------------------------------------------ */

static float *g_Q_pack  = NULL;
static float *g_K_pack  = NULL;
static float *g_S_buf   = NULL;
static float *g_P_pack  = NULL;
static float *g_O_tile  = NULL;
static size_t g_buf_cap = 0;  /* tracks d for which buffers were sized */

static void *alloc_aligned(size_t bytes)
{
    void *p = NULL;
    posix_memalign(&p, 64, bytes);
    madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

static void ensure_buffers(int d)
{
    if ((size_t)d <= g_buf_cap)
        return;

    free(g_Q_pack);
    free(g_K_pack);
    free(g_S_buf);
    free(g_P_pack);
    free(g_O_tile);

    g_Q_pack = (float *)alloc_aligned((size_t)BR * d * sizeof(float));
    g_K_pack = (float *)alloc_aligned((size_t)BC * d * sizeof(float));
    g_S_buf  = (float *)alloc_aligned((size_t)BR * BC * sizeof(float));
    g_P_pack = (float *)alloc_aligned((size_t)BR * BC * sizeof(float));
    g_O_tile = (float *)alloc_aligned((size_t)BR * d * sizeof(float));
    g_buf_cap = (size_t)d;
}

void flash_attention_cleanup(void)
{
    free(g_Q_pack);  g_Q_pack = NULL;
    free(g_K_pack);  g_K_pack = NULL;
    free(g_S_buf);   g_S_buf  = NULL;
    free(g_P_pack);  g_P_pack = NULL;
    free(g_O_tile);  g_O_tile = NULL;
    g_buf_cap = 0;
}

/* ------------------------------------------------------------------ */
/*  Main FlashAttention driver                                         */
/* ------------------------------------------------------------------ */

void flash_attention_fp32(
    const float *Q, const float *K, const float *V,
    float *O,
    int L, int d, float scale)
{
    ensure_buffers(d);

    float *Q_pack = g_Q_pack;
    float *K_pack = g_K_pack;
    float *S_buf  = g_S_buf;
    float *P_pack = g_P_pack;
    float *O_tile = g_O_tile;

    float m_prev[BR];
    float l_arr[BR];

    /* Outer loop: Q tiles (rows ic..ic+br-1) */
    for (int ic = 0; ic < L; ic += BR) {
        int br = (ic + BR <= L) ? BR : (L - ic);

        /* Pack Q tile: br/MR panels, each MR×d */
        for (int ir = 0; ir < br; ir += MR) {
            int mr = (ir + MR <= br) ? MR : (br - ir);
            pack_a(Q + (ic + ir) * d, d, Q_pack + ir * d, mr, d);
        }

        /* Initialize running softmax state */
        memset(O_tile, 0, (size_t)br * d * sizeof(float));
        for (int i = 0; i < br; i++) {
            m_prev[i] = -FLT_MAX;
            l_arr[i]  = 0.0f;
        }

        /* Inner loop: K/V tiles (rows jc..jc+bc-1) */
        for (int jc = 0; jc < L; jc += BC) {
            int bc = (jc + BC <= L) ? BC : (L - jc);

            /* --- GEMM 1: S[br,bc] = Q_tile × K_tile^T --- */

            /* Pack K tile: bc/NR panels, each NR×d */
            for (int jr = 0; jr < bc; jr += NR) {
                int nr = (jr + NR <= bc) ? NR : (bc - jr);
                pack_b(K + (jc + jr) * d, d, K_pack + jr * d, nr, d);
            }

            macrokernel_gemm1(Q_pack, K_pack, S_buf, br, bc, d);

            /* --- Online softmax: rescale O, compute P --- */
            online_softmax_rescale(S_buf, O_tile, m_prev, l_arr,
                                   br, bc, d, scale);

            /* --- GEMM 2: O_tile += P × V_tile --- */

            /* Pack P (S_buf now contains P values): br/MR panels, each MR×bc */
            for (int ir = 0; ir < br; ir += MR) {
                int mr = (ir + MR <= br) ? MR : (br - ir);
                pack_a(S_buf + ir * BC, BC, P_pack + ir * bc, mr, bc);
            }

            macrokernel_gemm2(P_pack, V, d, O_tile, br, bc, jc);
        }

        /* --- Final normalization: O[ic:ic+br, :] = O_tile / l --- */
        for (int i = 0; i < br; i++) {
            float inv_l = 1.0f / l_arr[i];
            __m256 vinv = _mm256_set1_ps(inv_l);
            float *src = O_tile + i * d;
            float *dst = O + (ic + i) * d;
            int j;
            for (j = 0; j + 8 <= d; j += 8)
                _mm256_storeu_ps(dst + j,
                    _mm256_mul_ps(_mm256_load_ps(src + j), vinv));
            for (; j < d; j++)
                dst[j] = src[j] * inv_l;
        }
    }
}
