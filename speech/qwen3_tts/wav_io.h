/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * wav_io.h - minimal RIFF/WAVE reader/writer (PCM16 / IEEE float32, mono or
 * interleaved multi-channel downmixed to mono on read), written from the RIFF
 * specification. Header-only, static functions.
 */
#ifndef QTTS_WAV_IO_H
#define QTTS_WAV_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void wav__u32(FILE *f, uint32_t v) { uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) }; fwrite(b, 1, 4, f); }
static void wav__u16(FILE *f, uint16_t v) { uint8_t b[2] = { (uint8_t)v, (uint8_t)(v >> 8) }; fwrite(b, 1, 2, f); }

/* Write mono PCM16 (float input clamped to [-1, 1]). */
static inline int wav_write_pcm16(const char *path, const float *x, int n, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t data = (uint32_t)n * 2;
    fwrite("RIFF", 1, 4, f); wav__u32(f, 36 + data); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); wav__u32(f, 16); wav__u16(f, 1); wav__u16(f, 1);
    wav__u32(f, (uint32_t)sr); wav__u32(f, (uint32_t)sr * 2); wav__u16(f, 2); wav__u16(f, 16);
    fwrite("data", 1, 4, f); wav__u32(f, data);
    for (int i = 0; i < n; i++) {
        float v = x[i] < -1.0f ? -1.0f : (x[i] > 1.0f ? 1.0f : x[i]);
        int s = (int)(v * 32767.0f + (v >= 0 ? 0.5f : -0.5f));
        wav__u16(f, (uint16_t)(int16_t)s);
    }
    fclose(f);
    return 0;
}

static uint32_t wav__rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint16_t wav__rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

/* Read PCM16/PCM24/PCM32/float32 WAV, downmixed to mono float. Returns malloc'd samples. */
static inline float *wav_read(const char *path, int *n_out, int *sr_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc((size_t)sz);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    if (sz < 12 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) { free(b); return NULL; }
    int fmt = 0, ch = 0, bits = 0, sr = 0;
    const uint8_t *data = NULL;
    uint32_t dlen = 0;
    for (long p = 12; p + 8 <= sz; ) {
        uint32_t cl = wav__rd32(b + p + 4);
        if (!memcmp(b + p, "fmt ", 4) && cl >= 16) {
            fmt = wav__rd16(b + p + 8); ch = wav__rd16(b + p + 10);
            sr = (int)wav__rd32(b + p + 12); bits = wav__rd16(b + p + 22);
            if (fmt == 0xFFFE && cl >= 40) fmt = wav__rd16(b + p + 32); /* extensible: subformat */
        } else if (!memcmp(b + p, "data", 4)) {
            data = b + p + 8;
            dlen = cl;
            if ((long)(p + 8 + dlen) > sz) dlen = (uint32_t)(sz - p - 8);
        }
        p += 8 + cl + (cl & 1);
    }
    if (!data || ch <= 0 || (fmt != 1 && fmt != 3)) { free(b); return NULL; }
    int bps = bits / 8, frame = bps * ch;
    int n = (int)(dlen / (uint32_t)frame);
    float *x = (float *)malloc(sizeof(float) * (size_t)(n ? n : 1));
    for (int i = 0; i < n; i++) {
        double acc = 0.0;
        for (int c = 0; c < ch; c++) {
            const uint8_t *s = data + (size_t)i * frame + (size_t)c * bps;
            double v = 0.0;
            if (fmt == 3 && bits == 32) { float fv; memcpy(&fv, s, 4); v = fv; }
            else if (bits == 16) v = (int16_t)wav__rd16(s) / 32768.0;
            else if (bits == 24) v = (double)((int32_t)((uint32_t)s[0] << 8 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 24) >> 8) / 8388608.0;
            else if (bits == 32) v = (int32_t)wav__rd32(s) / 2147483648.0;
            else if (bits == 8) v = (s[0] - 128) / 128.0;
            acc += v;
        }
        x[i] = (float)(acc / ch);
    }
    free(b);
    *n_out = n;
    if (sr_out) *sr_out = sr;
    return x;
}

#endif /* QTTS_WAV_IO_H */
