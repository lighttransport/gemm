/* SPDX-License-Identifier: MIT
 * test_ja_dsp: resampler SNR on band-limited tones, YIN F0 accuracy, RMS level. */
#define JA_DSP_IMPLEMENTATION
#include "ja_dsp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int fails = 0;
    /* 24 kHz -> 16 kHz: sum of tones below the new Nyquist must survive (SNR in the interior) */
    int n = 24000;
    float *x = malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) x[i] = 0.5f * sinf(2 * M_PI * 440 * i / 24000.0) + 0.25f * sinf(2 * M_PI * 3100 * i / 24000.0);
    int no = 0;
    float *y = ja_resample(x, n, 24000, 16000, &no);
    double se = 0, ss = 0;
    for (int i = 400; i < no - 400; i++) {
        double r = 0.5 * sin(2 * M_PI * 440 * i / 16000.0) + 0.25 * sin(2 * M_PI * 3100 * i / 16000.0);
        se += (y[i] - r) * (y[i] - r); ss += r * r;
    }
    double snr = 10 * log10(ss / se);
    printf("resample 24k->16k: %d samples, tone SNR %.1f dB\n", no, snr);
    if (no != 16000 || snr < 60) fails++;
    /* a tone above the new Nyquist must be rejected */
    for (int i = 0; i < n; i++) x[i] = sinf(2 * M_PI * 9000 * i / 24000.0);
    free(y);
    y = ja_resample(x, n, 24000, 16000, &no);
    double e = 0;
    for (int i = 400; i < no - 400; i++) e += y[i] * y[i];
    double att = 10 * log10(e / (no - 800) / 0.5);
    printf("resample: 9 kHz tone attenuation %.1f dB\n", att);
    if (att > -40) fails++;
    free(x); free(y);

    /* YIN on a harmonic-rich voiced signal with F0 gliding 120 -> 300 Hz */
    int sr = 16000, len = sr;
    float *s = malloc(sizeof(float) * len);
    double ph = 0;
    for (int i = 0; i < len; i++) {
        double f = 120 + 180.0 * i / len;
        ph += 2 * M_PI * f / sr;
        s[i] = (float)(0.3 * sin(ph) + 0.2 * sin(2 * ph) + 0.1 * sin(3 * ph));
    }
    int nf = 1 + (len - 1) / 160;
    float *f0 = malloc(sizeof(float) * nf), *ap = malloc(sizeof(float) * nf), *db = malloc(sizeof(float) * nf);
    ja_yin(s, len, sr, 640, 160, 60, 600, 0.15f, f0, ap, nf);
    double maxerr = 0;
    for (int i = 5; i < nf - 5; i++) {
        double want = 120 + 180.0 * (i * 160) / len;
        double err = fabs(f0[i] - want) / want;
        if (err > maxerr) maxerr = err;
    }
    printf("yin: max relative F0 error %.3f%%\n", 100 * maxerr);
    if (maxerr > 0.01) fails++;
    ja_rms_db(s, len, 400, 160, db, nf);
    printf("rms: %.2f dBFS (expected %.2f)\n", db[nf / 2], 10 * log10((0.09 + 0.04 + 0.01) / 2));
    if (fabs(db[nf / 2] - 10 * log10(0.07)) > 0.3) fails++;
    free(s); free(f0); free(ap); free(db);
    printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
