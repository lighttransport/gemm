/* SPDX-License-Identifier: MIT
 * GNR1: little-endian generic sparse-policy replay. No game/rules dependency.
 * Plane storage: 0 = packed bits, 1 = broadcast F32, 2/3 = fixed x/y coordinate.
 * action = spatial_index * actions_per_square + action_plane.
 */
#ifndef GEMM_GN_REPLAY_H
#define GEMM_GN_REPLAY_H
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    uint32_t side, channels, actions;
    uint8_t planes[1024];
} gn_replay_schema;
typedef struct {
    uint64_t game, generation;
    uint32_t ply, count, label;
} gn_replay_record;
static inline int gnr_put32(FILE *f, uint32_t v) {
    uint8_t b[4];
    for (int i = 0; i < 4; i++)
        b[i] = (uint8_t)(v >> (8 * i));
    return fwrite(b, 1, 4, f) == 4;
}
static inline int gnr_get32(FILE *f, uint32_t *v) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4)
        return 0;
    *v = 0;
    for (int i = 0; i < 4; i++)
        *v |= (uint32_t)b[i] << (8 * i);
    return 1;
}
static inline int gnr_put64(FILE *f, uint64_t v) {
    return gnr_put32(f, (uint32_t)v) && gnr_put32(f, (uint32_t)(v >> 32));
}
static inline int gnr_get64(FILE *f, uint64_t *v) {
    uint32_t a, b;
    if (!gnr_get32(f, &a) || !gnr_get32(f, &b))
        return 0;
    *v = (uint64_t)a | ((uint64_t)b << 32);
    return 1;
}
static inline int gnr_putfloat(FILE *f, float x) {
    uint32_t v;
    memcpy(&v, &x, 4);
    return isfinite(x) && gnr_put32(f, v);
}
static inline int gnr_getfloat(FILE *f, float *x) {
    uint32_t v;
    if (!gnr_get32(f, &v))
        return 0;
    memcpy(x, &v, 4);
    return isfinite(*x);
}
static inline int gnr_valid(const gn_replay_schema *s) {
    if (s->side < 2 || s->side > 19 || !s->channels || s->channels > 1024 || !s->actions ||
        s->actions > 1024)
        return 0;
    for (unsigned i = 0; i < s->channels; i++)
        if (s->planes[i] > 3)
            return 0;
    return 1;
}
static inline int gnr_write_header(FILE *f, const gn_replay_schema *s) {
    return gnr_valid(s) && fwrite("GNR1", 1, 4, f) == 4 && gnr_put32(f, s->side) &&
           gnr_put32(f, s->channels) && gnr_put32(f, s->actions) &&
           fwrite(s->planes, 1, s->channels, f) == s->channels;
}
static inline int gnr_read_header(FILE *f, gn_replay_schema *s) {
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GNR1", 4) || !gnr_get32(f, &s->side) ||
        !gnr_get32(f, &s->channels) || !gnr_get32(f, &s->actions) || s->channels > 1024)
        return 0;
    return fread(s->planes, 1, s->channels, f) == s->channels && gnr_valid(s);
}
static inline int gnr_write(FILE *f, const gn_replay_schema *s, const gn_replay_record *r,
                            const float *x, const uint32_t *ids, const uint32_t *visits) {
    unsigned S = s->side * s->side, A = S * s->actions;
    if (!r->count || r->count > A || r->label > 2)
        return 0;
    if (!gnr_put64(f, r->game) || !gnr_put64(f, r->generation) || !gnr_put32(f, r->ply) ||
        !gnr_put32(f, r->count) || !gnr_put32(f, r->label))
        return 0;
    for (unsigned c = 0; c < s->channels; c++) {
        if (s->planes[c] == 0) {
            uint8_t bits[46] = {0};
            for (unsigned i = 0; i < S; i++) {
                float v = x[i * s->channels + c];
                if (v != 0 && v != 1)
                    return 0;
                if (v == 1)
                    bits[i / 8] |= (uint8_t)(1U << (i % 8));
            }
            if (fwrite(bits, 1, (S + 7) / 8, f) != (S + 7) / 8)
                return 0;
        } else if (s->planes[c] == 1) {
            if (!gnr_putfloat(f, x[c]))
                return 0;
        }
    }
    uint64_t total = 0;
    for (unsigned i = 0; i < r->count; i++) {
        if (ids[i] >= A)
            return 0;
        total += visits[i];
        if (!gnr_put32(f, ids[i]) || !gnr_put32(f, visits[i]))
            return 0;
    }
    return total != 0;
}
/* 1 record, 0 clean EOF, -1 malformed/truncated. Target is dense, -1 illegal. */
static inline int gnr_read(FILE *f, const gn_replay_schema *s, gn_replay_record *r, float *x,
                           float *target) {
    int first = fgetc(f);
    if (first == EOF)
        return ferror(f) ? -1 : 0;
    ungetc(first, f);
    unsigned S = s->side * s->side, A = S * s->actions;
    if (!gnr_get64(f, &r->game) || !gnr_get64(f, &r->generation) || !gnr_get32(f, &r->ply) ||
        !gnr_get32(f, &r->count) || !gnr_get32(f, &r->label) || !r->count || r->count > A ||
        r->label > 2)
        return -1;
    for (unsigned c = 0; c < s->channels; c++) {
        if (s->planes[c] == 0) {
            uint8_t bits[46];
            if (fread(bits, 1, (S + 7) / 8, f) != (S + 7) / 8)
                return -1;
            for (unsigned i = 0; i < S; i++)
                x[i * s->channels + c] = (bits[i / 8] >> (i % 8)) & 1;
        } else if (s->planes[c] == 1) {
            float v;
            if (!gnr_getfloat(f, &v))
                return -1;
            for (unsigned i = 0; i < S; i++)
                x[i * s->channels + c] = v;
        } else
            for (unsigned i = 0; i < S; i++)
                x[i * s->channels + c] =
                    2.0f * (float)(s->planes[c] == 2 ? i % s->side : i / s->side) / (s->side - 1) -
                    1;
    }
    for (unsigned i = 0; i < A; i++)
        target[i] = -1;
    uint64_t total = 0;
    for (unsigned i = 0; i < r->count; i++) {
        uint32_t id, n;
        if (!gnr_get32(f, &id) || !gnr_get32(f, &n) || id >= A || target[id] >= 0)
            return -1;
        target[id] = (float)n;
        total += n;
    }
    if (!total)
        return -1;
    for (unsigned i = 0; i < A; i++)
        if (target[i] >= 0)
            target[i] /= (float)total;
    return 1;
}
#endif
