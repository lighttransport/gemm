/*
 * svdquant_cpu.h — portable C reference for the SVDQuant forward.
 *
 * Reproduces, in plain C (double accumulation), the forward that
 * ref/svdquant/gen_svdquant_ref.py dumps as `*_y_svdq`:
 *
 *   y = act(x/lam) @ R_dec^T + (x @ lora_down_emit^T) @ lora_up^T + bias
 *
 * - residual term uses the SMOOTHED activation x/lam; low-rank uses RAW x.
 * - lora_down_emit is dumped already divided by lam.
 * - R_dec is decoded from the 4-bit residual: INT4 (signed [-8,7] group-64) or
 *   NVFP4 (e2m1 group-16 codes x e4m3 micro-scales x per-row wcwt).
 * - act(.) = identity (W4A16) or 4-bit per-token-group quant->dequant (W4A4).
 *
 * Single header, all `static inline`, no deps beyond libm. Sizes here are tiny
 * (unit-test scale) so the naive triple loops in f64 are the reference, not a
 * perf path.
 */
#ifndef SVDQUANT_CPU_H
#define SVDQUANT_CPU_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* E2M1 value LUT: code 0..7 positive magnitudes, 8..15 negatives (sign in bit 3).
 * Matches cuda/fp4/fp4_w4a4_opt.c:24 and the kernel's e2m1 decode. */
static const float SQ_E2M1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6,
                                  -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};

static inline float sq_e2m1_decode(int code) { return SQ_E2M1[code & 15]; }

/* Unsigned-E4M3 decode (1+m/8)*2^(e-7); subnormal (m/8)*2^-6.
 * Bit-identical to qf_e4m3_dec in cuda/fp4_w4a4.h and ue4m3_decode in fp4_w4a4_opt.c. */
static inline float sq_ue4m3_decode(unsigned char b) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, 1 - 7);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

/* INT4 residual decode: signed nibble [-8,7] x per-group scale -> R[out,in] f32.
 * Even col -> low nibble, odd col -> high nibble (matches pack_nibbles). */
static inline void sq_unpack_int4_residual(const uint8_t *qint4, const float *wscale,
                                           float *R, int out, int in, int group) {
    int npk = in / 2, ng = in / group;
    for (int o = 0; o < out; o++) {
        const uint8_t *q = qint4 + (size_t)o * npk;
        const float *ws = wscale + (size_t)o * ng;
        float *Ro = R + (size_t)o * in;
        for (int c = 0; c < npk; c++) {
            int lo = q[c] & 0xF;        if (lo >= 8) lo -= 16;
            int hi = (q[c] >> 4) & 0xF; if (hi >= 8) hi -= 16;
            int k = 2 * c;
            Ro[k]     = (float)lo * ws[k / group];
            Ro[k + 1] = (float)hi * ws[(k + 1) / group];
        }
    }
}

/* NVFP4 residual decode: 8 e2m1 codes / uint32, e4m3 micro-scale per group-16,
 * per-output wcwt. R[o,k] = E2M1[code] * e4m3(ws[o,k/16]) * wcwt[o]. */
static inline void sq_unpack_nvfp4_residual(const int32_t *qw, const uint8_t *ws,
                                            const float *wcwt, float *R,
                                            int out, int in, int gs) {
    int nu = in / 8, ng = in / gs;
    for (int o = 0; o < out; o++) {
        const int32_t *qrow = qw + (size_t)o * nu;
        const uint8_t *wsr = ws + (size_t)o * ng;
        float wc = wcwt[o];
        float *Ro = R + (size_t)o * in;
        for (int u = 0; u < nu; u++) {
            uint32_t v = (uint32_t)qrow[u];
            for (int i = 0; i < 8; i++) {
                int code = (v >> (i * 4)) & 0xF;
                int k = u * 8 + i;
                Ro[k] = sq_e2m1_decode(code) * sq_ue4m3_decode(wsr[k / gs]) * wc;
            }
        }
    }
}

/* Per-token per-group-64 symmetric INT4 activation quant -> dequant in place.
 * rintf uses the current (round-to-nearest-even) mode, matching torch.round. */
static inline void sq_quant_act_int4_g64(const float *xr, float *xdq,
                                         int tok, int in, int group) {
    int ng = in / group;
    for (int t = 0; t < tok; t++) {
        for (int g = 0; g < ng; g++) {
            int k0 = g * group;
            float amax = 0.f;
            for (int i = 0; i < group; i++) {
                float v = fabsf(xr[(size_t)t * in + k0 + i]);
                if (v > amax) amax = v;
            }
            float scale = amax / 7.0f;
            if (scale < 1e-12f) scale = 1e-12f;
            for (int i = 0; i < group; i++) {
                float q = rintf(xr[(size_t)t * in + k0 + i] / scale);
                if (q > 7.f) q = 7.f;
                if (q < -7.f) q = -7.f;
                xdq[(size_t)t * in + k0 + i] = q * scale;
            }
        }
    }
}

/* xr[t,i] = x[t,i] / lam[i] */
static inline void sq_smooth_div(const float *x, const float *lam, float *xr,
                                 int tok, int in) {
    for (int t = 0; t < tok; t++)
        for (int i = 0; i < in; i++)
            xr[(size_t)t * in + i] = x[(size_t)t * in + i] / lam[i];
}

/* The unified SVDQuant forward (f64 accumulation):
 *   y = x_act @ R^T + (x_raw @ ld^T) @ lu^T + bias
 * x_act,x_raw,R: [.,in]; ld=lora_down_emit [rank,in]; lu=lora_up [out,rank]; bias[out]. */
static inline void sq_forward(const float *x_act, const float *x_raw,
                              const float *R, const float *lu, const float *ld,
                              const float *bias, float *y,
                              int tok, int out, int in, int rank) {
    double *la = (double *)malloc((size_t)tok * rank * sizeof(double));
    for (int t = 0; t < tok; t++) {
        const float *xr = x_raw + (size_t)t * in;
        for (int r = 0; r < rank; r++) {
            const float *ldr = ld + (size_t)r * in;
            double s = 0.0;
            for (int i = 0; i < in; i++) s += (double)xr[i] * ldr[i];
            la[(size_t)t * rank + r] = s;
        }
    }
    for (int t = 0; t < tok; t++) {
        const float *xa = x_act + (size_t)t * in;
        const double *lar = la + (size_t)t * rank;
        for (int o = 0; o < out; o++) {
            const float *Ro = R + (size_t)o * in;
            double s = 0.0;
            for (int i = 0; i < in; i++) s += (double)xa[i] * Ro[i];
            const float *luo = lu + (size_t)o * rank;
            for (int r = 0; r < rank; r++) s += lar[r] * (double)luo[r];
            y[(size_t)t * out + o] = (float)(s + (double)bias[o]);
        }
    }
    free(la);
}

#endif /* SVDQUANT_CPU_H */
