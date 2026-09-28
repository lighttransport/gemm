/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * ja_dsp.h - signal processing for ja_align, written from first principles:
 *   ja_resample   band-limited resampling by windowed-sinc interpolation
 *                 (Blackman-windowed ideal low-pass, cutoff 0.95 * min Nyquist)
 *   ja_rms_db     frame RMS energy in dBFS
 *   ja_yin        F0 via the YIN algorithm (de Cheveigne & Kawahara, JASA 2002):
 *                 difference function, cumulative-mean normalization, absolute
 *                 threshold, parabolic interpolation.
 */
#ifndef JA_DSP_H
#define JA_DSP_H

float *ja_resample(const float *x, int n, int sr_in, int sr_out, int *n_out);
/* n_frames = 1 + (n - 1) / hop (centered frames, zero padded) */
void   ja_rms_db(const float *x, int n, int win, int hop, float *out, int n_frames);
/* f0[i] = Hz (0 if unvoiced), aperiodicity[i] = min CMNDF value (0 = periodic, 1 = noise) */
void   ja_yin(const float *x, int n, int sr, int win, int hop, float fmin, float fmax, float threshold,
              float *f0, float *aperiodicity, int n_frames);

#endif /* JA_DSP_H */

#if defined(JA_DSP_IMPLEMENTATION) && !defined(JA_DSP_IMPL_DONE)
#define JA_DSP_IMPL_DONE

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float *ja_resample(const float *x, int n, int sr_in, int sr_out, int *n_out) {
    if (sr_in == sr_out) {
        float *y = (float *)malloc(sizeof(float) * (size_t)n);
        memcpy(y, x, sizeof(float) * (size_t)n);
        *n_out = n;
        return y;
    }
    long long no = ((long long)n * sr_out + sr_in - 1) / sr_in;
    float *y = (float *)malloc(sizeof(float) * (size_t)no);
    double ratio = (double)sr_out / sr_in;
    double fc = 0.95 * 0.5 * (ratio < 1.0 ? ratio : 1.0); /* cutoff in cycles per input sample */
    const int half = 32;                                   /* zero crossings of the output-rate kernel */
    double width = half / (2.0 * fc) ;                     /* half kernel length in input samples */
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < no; i++) {
        double t = i / ratio;                              /* position in input samples */
        long long j0 = (long long)ceil(t - width), j1 = (long long)floor(t + width);
        double acc = 0.0, wsum = 0.0;
        for (long long j = j0; j <= j1; j++) {
            double d = t - j;
            double arg = 2.0 * fc * d;
            double s = fabs(arg) < 1e-12 ? 1.0 : sin(M_PI * arg) / (M_PI * arg);
            double u = (d + width) / (2.0 * width);        /* 0..1 over the window */
            double w = 0.42 - 0.5 * cos(2 * M_PI * u) + 0.08 * cos(4 * M_PI * u);
            double k = 2.0 * fc * s * w;
            wsum += k;
            if (j >= 0 && j < n) acc += k * x[j];
        }
        y[i] = (float)(wsum > 0 ? acc / wsum : 0.0);
    }
    *n_out = (int)no;
    return y;
}

void ja_rms_db(const float *x, int n, int win, int hop, float *out, int n_frames) {
    for (int f = 0; f < n_frames; f++) {
        int c = f * hop, a = c - win / 2;
        double s = 0.0;
        for (int i = 0; i < win; i++) {
            int k = a + i;
            if (k >= 0 && k < n) s += (double)x[k] * x[k];
        }
        out[f] = (float)(10.0 * log10(s / win + 1e-10));
    }
}

void ja_yin(const float *x, int n, int sr, int win, int hop, float fmin, float fmax, float threshold,
            float *f0, float *aper, int n_frames) {
    int tmax = (int)(sr / fmin) + 1, tmin = (int)(sr / fmax);
    if (tmax >= win) tmax = win - 1;
    #pragma omp parallel for schedule(dynamic, 8)
    for (int f = 0; f < n_frames; f++) {
        double *d = (double *)calloc((size_t)tmax + 2, sizeof(double));
        int a = f * hop - win / 2;
        /* difference function over an integration window of win - tmax samples */
        int W = win - tmax;
        for (int tau = 1; tau <= tmax; tau++) {
            double s = 0.0;
            for (int j = 0; j < W; j++) {
                int k0 = a + j, k1 = a + j + tau;
                float v0 = (k0 >= 0 && k0 < n) ? x[k0] : 0.0f;
                float v1 = (k1 >= 0 && k1 < n) ? x[k1] : 0.0f;
                double e = v0 - v1;
                s += e * e;
            }
            d[tau] = s;
        }
        /* cumulative mean normalized difference */
        double run = 0.0;
        d[0] = 1.0;
        for (int tau = 1; tau <= tmax; tau++) {
            run += d[tau];
            d[tau] = run > 0 ? d[tau] * tau / run : 1.0;
        }
        int best = -1;
        for (int tau = tmin > 1 ? tmin : 2; tau < tmax; tau++) {
            if (d[tau] < threshold) {
                while (tau + 1 < tmax && d[tau + 1] < d[tau]) tau++;
                best = tau;
                break;
            }
        }
        double mn = 1.0;
        int arg = -1;
        for (int tau = tmin > 1 ? tmin : 2; tau < tmax; tau++) if (d[tau] < mn) { mn = d[tau]; arg = tau; }
        aper[f] = (float)mn;
        if (best < 0) { f0[f] = 0.0f; free(d); continue; }
        /* parabolic interpolation of the dip */
        double p = best;
        if (best > 1 && best < tmax) {
            double y0 = d[best - 1], y1 = d[best], y2 = d[best + 1];
            double den = y0 - 2 * y1 + y2;
            if (fabs(den) > 1e-12) p = best + 0.5 * (y0 - y2) / den;
        }
        f0[f] = (float)(sr / p);
        (void)arg;
        free(d);
    }
}

#endif /* JA_DSP_IMPLEMENTATION */
