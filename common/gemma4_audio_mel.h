/* gemma4_audio_mel.h — CPU log-mel front-end matching HF Gemma4AudioFeatureExtractor.
 *
 * 16 kHz mono, frame 320 / hop 160, FFT 512 (257 bins), 128 HTK-mel filters,
 * periodic Hann window, mel_floor 1e-3, preemphasis 0, semicausal pad (160 zeros).
 *
 *   g4a_mel *fe = g4a_mel_init();
 *   float *pcm = g4a_wav_load("a.wav", &n);          // 16-bit PCM mono -> [-1,1]
 *   float *mel = g4a_mel_extract(fe, pcm, n, &n_frames);  // [n_frames*128]
 *   ... free(mel); free(pcm); g4a_mel_free(fe);
 */
#ifndef GEMMA4_AUDIO_MEL_H
#define GEMMA4_AUDIO_MEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct g4a_mel g4a_mel;

g4a_mel *g4a_mel_init(void);
void     g4a_mel_free(g4a_mel *fe);
float   *g4a_wav_load(const char *path, int *n_samples);  /* malloc'd, caller frees */
float   *g4a_mel_extract(g4a_mel *fe, const float *pcm, int n, int *n_frames); /* malloc'd */

#ifdef __cplusplus
}
#endif

#ifdef GEMMA4_AUDIO_MEL_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define G4M_FRAME  320
#define G4M_HOP    160
#define G4M_FFT    512
#define G4M_BINS   257   /* fft/2 + 1 */
#define G4M_MEL    128
#define G4M_FLOOR  0.001
#define G4M_SR     16000
#define G4M_FMIN   0.0
#define G4M_FMAX   8000.0

struct g4a_mel {
    float window[G4M_FRAME];        /* periodic Hann */
    float mel_filters[G4M_BINS * G4M_MEL];   /* [257,128] */
    float cs[G4M_BINS * G4M_FFT];   /* cos table cs[k*512+n] */
    float sn[G4M_BINS * G4M_FFT];
};

static double g4m_h2m(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double g4m_m2h(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

g4a_mel *g4a_mel_init(void) {
    g4a_mel *fe = (g4a_mel *)calloc(1, sizeof(g4a_mel));
    /* periodic Hann: hanning(frame+1)[:-1] = 0.5-0.5cos(2pi n/frame) */
    for (int n = 0; n < G4M_FRAME; n++) fe->window[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * n / G4M_FRAME));
    /* HTK mel filter bank, norm=None, triangularize in freq space */
    double mel_min = g4m_h2m(G4M_FMIN), mel_max = g4m_h2m(G4M_FMAX);
    double ff[G4M_MEL + 2];           /* filter_freqs (Hz) */
    for (int i = 0; i < G4M_MEL + 2; i++) { double mel = mel_min + (mel_max - mel_min) * i / (G4M_MEL + 1); ff[i] = g4m_m2h(mel); }
    for (int b = 0; b < G4M_BINS; b++) {
        double fft_freq = (double)(G4M_SR / 2) * b / (G4M_BINS - 1);   /* linspace(0,8000,257) */
        for (int m = 0; m < G4M_MEL; m++) {
            double down = -(ff[m] - fft_freq) / (ff[m + 1] - ff[m]);
            double up = (ff[m + 2] - fft_freq) / (ff[m + 2] - ff[m + 1]);
            double v = down < up ? down : up; if (v < 0) v = 0;
            fe->mel_filters[b * G4M_MEL + m] = (float)v;
        }
    }
    /* DFT twiddles for n=0..511, k=0..256 */
    for (int k = 0; k < G4M_BINS; k++) for (int n = 0; n < G4M_FFT; n++) {
        double a = -2.0 * M_PI * k * n / G4M_FFT;
        fe->cs[k * G4M_FFT + n] = (float)cos(a);
        fe->sn[k * G4M_FFT + n] = (float)sin(a);
    }
    return fe;
}
void g4a_mel_free(g4a_mel *fe) { free(fe); }

float *g4a_wav_load(const char *path, int *n_samples) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "g4a_wav: cannot open %s\n", path); return NULL; }
    char riff[4], wave[4]; uint32_t sz;
    if (fread(riff, 1, 4, fp) != 4 || fread(&sz, 4, 1, fp) != 1 || fread(wave, 1, 4, fp) != 4 ||
        memcmp(riff, "RIFF", 4) || memcmp(wave, "WAVE", 4)) { fprintf(stderr, "g4a_wav: not a WAV\n"); fclose(fp); return NULL; }
    uint16_t fmt = 0, ch = 0, bits = 0; uint32_t rate = 0;
    float *out = NULL; int nout = 0;
    char id[4]; uint32_t csz;
    while (fread(id, 1, 4, fp) == 4 && fread(&csz, 4, 1, fp) == 1) {
        if (!memcmp(id, "fmt ", 4)) {
            uint8_t buf[64]; uint32_t r = csz < 64 ? csz : 64; fread(buf, 1, r, fp);
            memcpy(&fmt, buf, 2); memcpy(&ch, buf + 2, 2); memcpy(&rate, buf + 4, 4); memcpy(&bits, buf + 14, 2);
            if (csz > r) fseek(fp, csz - r, SEEK_CUR);
        } else if (!memcmp(id, "data", 4)) {
            int bytes_per = bits / 8;
            int total = csz / bytes_per;            /* total samples across channels */
            int frames = ch ? total / ch : total;
            out = (float *)malloc((size_t)frames * sizeof(float));
            nout = frames;
            for (int i = 0; i < frames; i++) {
                double acc = 0;
                for (int c = 0; c < ch; c++) {
                    if (bits == 16) { int16_t s; fread(&s, 2, 1, fp); acc += s / 32768.0; }
                    else if (bits == 32 && fmt == 3) { float s; fread(&s, 4, 1, fp); acc += s; }
                    else if (bits == 32) { int32_t s; fread(&s, 4, 1, fp); acc += s / 2147483648.0; }
                    else if (bits == 8) { uint8_t s; fread(&s, 1, 1, fp); acc += (s - 128) / 128.0; }
                }
                out[i] = (float)(acc / (ch ? ch : 1));   /* downmix to mono */
            }
        } else { fseek(fp, csz + (csz & 1), SEEK_CUR); continue; }
        if (csz & 1) fseek(fp, 1, SEEK_CUR);
    }
    fclose(fp);
    if (rate != G4M_SR) fprintf(stderr, "g4a_wav: WARNING sample rate %u != 16000 (resample first)\n", rate);
    if (n_samples) *n_samples = nout;
    return out;
}

float *g4a_mel_extract(g4a_mel *fe, const float *pcm, int n, int *n_frames) {
    int pad = G4M_FRAME / 2;                  /* 160 semicausal pad */
    int plen = n + pad;
    float *w = (float *)calloc(plen, sizeof(float));
    memcpy(w + pad, pcm, (size_t)n * sizeof(float));
    int frame_unfold = G4M_FRAME + 1;         /* 321 */
    int nf = (plen - frame_unfold) / G4M_HOP + 1;
    if (nf < 0) nf = 0;
    float *mel = (float *)malloc((size_t)nf * G4M_MEL * sizeof(float));
    float frame[G4M_FRAME], mag[G4M_BINS];
    for (int f = 0; f < nf; f++) {
        int off = f * G4M_HOP;
        for (int i = 0; i < G4M_FRAME; i++) frame[i] = w[off + i] * fe->window[i];   /* drop last of 321 */
        for (int k = 0; k < G4M_BINS; k++) {
            const float *cs = fe->cs + k * G4M_FFT, *sn = fe->sn + k * G4M_FFT;
            double re = 0, im = 0;
            for (int i = 0; i < G4M_FRAME; i++) { re += frame[i] * cs[i]; im += frame[i] * sn[i]; }
            mag[k] = (float)sqrt(re * re + im * im);
        }
        float *mrow = mel + (size_t)f * G4M_MEL;
        for (int m = 0; m < G4M_MEL; m++) {
            double acc = 0;
            for (int k = 0; k < G4M_BINS; k++) acc += (double)mag[k] * fe->mel_filters[k * G4M_MEL + m];
            mrow[m] = (float)log(acc + G4M_FLOOR);
        }
    }
    free(w);
    if (n_frames) *n_frames = nf;
    return mel;
}

#endif /* GEMMA4_AUDIO_MEL_IMPLEMENTATION */
#endif /* GEMMA4_AUDIO_MEL_H */
