#include "vhuman_deformer.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_avx2.h"
#include "lightrig_mlp2.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../common/safetensors.h"

#define ATTR 6

struct vh_deformer {
    st_context *st;
    size_t C, P, I, J, B, M, V, K, H, corr_k, ml_in;
    const float *range;        /* C x 2 */
    const int32_t *corr_in;    /* P x corr_k (-1 padded) */
    const float *corr_w;       /* P */
    const float *jm;           /* (J*6) x I */
    const int32_t *parent;     /* J */
    const float *rest_t;       /* J x 3 */
    const float *rest_R;       /* J x 3 x 3 */
    const float *inv_bind;     /* J x 4 x 4 */
    const int32_t *src;        /* B */
    const float *morph;        /* M x V x 3 */
    const float *rest;         /* V x 3 */
    const int32_t *sj;         /* V x 4 */
    const float *sw;           /* V x 4 */
    /* ML */
    const float *w1, *b1, *w2, *b2, *imean, *iscale, *omean, *oscale;
    /* scratch */
    float *inp, *delta, *skin, *hid, *coef, *wts, *pos;
};

static const void *tensor(st_context *s, const char *name, const char *dtype, int nd, size_t *shape) {
    int i = safetensors_find(s, name);
    if (i < 0 || strcmp(safetensors_dtype(s, i), dtype) || safetensors_ndims(s, i) != nd) return NULL;
    const uint64_t *sh = safetensors_shape(s, i);
    for (int k = 0; k < nd; ++k) shape[k] = (size_t)sh[k];
    return safetensors_data(s, i);
}

vh_deformer *vh_deformer_load(const char *path) {
    st_context *s = safetensors_open(path);
    if (!s) return NULL;
    vh_deformer *d = calloc(1, sizeof(*d));
    if (!d) { safetensors_close(s); return NULL; }
    d->st = s;
    size_t sh[3];
    int ok = 1;
#define GET(field, name, dt, nd) do { d->field = tensor(s, name, dt, nd, sh); ok = ok && d->field; } while (0)
    GET(range, "controls.range", "F32", 2); d->C = sh[0];
    GET(jm, "joints.matrix", "F32", 2); d->I = sh[1]; d->J = sh[0] / ATTR;
    GET(parent, "joints.parent", "I32", 1);
    GET(rest_t, "joints.rest_t", "F32", 2);
    GET(rest_R, "joints.rest_R", "F32", 3);
    GET(inv_bind, "joints.inv_bind", "F32", 3);
    GET(src, "shapes.src", "I32", 1); d->B = sh[0];
    GET(morph, "morph", "F32", 3); d->M = sh[0]; d->V = sh[1];
    GET(rest, "rest", "F32", 2);
    GET(sj, "skin.joints", "I32", 2);
    GET(sw, "skin.weights", "F32", 2);
    d->P = d->I - d->C;
    if (d->P) {
        GET(corr_in, "corr.inputs", "I32", 2); d->corr_k = sh[1];
        GET(corr_w, "corr.weight", "F32", 1);
    }
#undef GET
    d->w1 = tensor(s, "ml.fc1.weight", "F32", 2, sh);
    if (d->w1) {
        d->H = sh[0];
        d->ml_in = sh[1];                            /* controls, or controls + correctives */
        ok = ok && d->ml_in <= d->I;
        d->b1 = tensor(s, "ml.fc1.bias", "F32", 1, sh);
        d->w2 = tensor(s, "ml.fc2.weight", "F32", 2, sh); d->K = sh[0];
        d->b2 = tensor(s, "ml.fc2.bias", "F32", 1, sh);
        d->imean = tensor(s, "ml.input.mean", "F32", 1, sh);
        d->iscale = tensor(s, "ml.input.scale", "F32", 1, sh);
        d->omean = tensor(s, "ml.output.mean", "F32", 1, sh);
        d->oscale = tensor(s, "ml.output.scale", "F32", 1, sh);
        ok = ok && d->b1 && d->w2 && d->b2 && d->imean && d->iscale && d->omean && d->oscale
             && d->M == d->B + 1 + d->K;
    } else {
        ok = ok && d->M == d->B;
    }
    if (!ok || d->J > 64 || d->I > 512) { vh_deformer_free(d); return NULL; }
    d->inp = malloc(sizeof(float) * d->I);
    d->delta = malloc(sizeof(float) * d->J * ATTR);
    d->skin = malloc(sizeof(float) * d->J * 16);
    d->hid = malloc(sizeof(float) * (d->H ? d->H : 1));
    d->coef = malloc(sizeof(float) * (d->K ? d->K : 1));
    d->wts = malloc(sizeof(float) * d->M);
    d->pos = malloc(sizeof(float) * d->V * 3);
    if (!d->inp || !d->delta || !d->skin || !d->hid || !d->coef || !d->wts || !d->pos) {
        vh_deformer_free(d);
        return NULL;
    }
    return d;
}

void vh_deformer_free(vh_deformer *d) {
    if (!d) return;
    free(d->inp); free(d->delta); free(d->skin); free(d->hid); free(d->coef); free(d->wts); free(d->pos);
    if (d->st) safetensors_close(d->st);
    free(d);
}

size_t vh_deformer_controls(const vh_deformer *d) { return d->C; }
size_t vh_deformer_vertices(const vh_deformer *d) { return d->V; }
size_t vh_deformer_morphs(const vh_deformer *d) { return d->M; }
int vh_deformer_has_ml(const vh_deformer *d) { return d->w1 != NULL; }

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

static void inputs(vh_deformer *d, const float *controls) {
    for (size_t c = 0; c < d->C; ++c) d->inp[c] = clampf(controls[c], d->range[2 * c], d->range[2 * c + 1]);
    for (size_t p = 0; p < d->P; ++p) {
        float v = d->corr_w[p];
        for (size_t k = 0; k < d->corr_k; ++k) {
            int32_t i = d->corr_in[p * d->corr_k + k];
            if (i >= 0) v *= clampf(d->inp[i], 0.f, 1.f);
        }
        d->inp[d->C + p] = v < 1.f ? v : 1.f;
    }
}

/* R = rest_R @ Rz @ Ry @ Rx, local = [R | rest_t + dt], world = parent world @ local, skin = world @ inv_bind */
static void skinning(vh_deformer *d) {
    memset(d->delta, 0, sizeof(float) * d->J * ATTR);
    for (size_t r = 0; r < d->J * ATTR; ++r) {
        const float *row = d->jm + r * d->I;
        float v = 0;
        for (size_t i = 0; i < d->I; ++i) v += row[i] * d->inp[i];
        d->delta[r] = v;
    }
    float world[64][16];
    for (size_t j = 0; j < d->J && j < 64; ++j) {
        const float *dl = d->delta + j * ATTR;
        float cx = cosf(dl[3]), sx = sinf(dl[3]), cy = cosf(dl[4]), sy = sinf(dl[4]), cz = cosf(dl[5]), sz = sinf(dl[5]);
        /* Rz Ry Rx */
        float E[9] = {cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
                      sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
                      -sy, cy * sx, cy * cx};
        const float *R0 = d->rest_R + j * 9;
        float L[16] = {0};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                L[a * 4 + b] = R0[a * 3 + 0] * E[0 * 3 + b] + R0[a * 3 + 1] * E[1 * 3 + b] + R0[a * 3 + 2] * E[2 * 3 + b];
        for (int a = 0; a < 3; ++a) L[a * 4 + 3] = d->rest_t[j * 3 + a] + dl[a];
        L[15] = 1;
        int32_t p = d->parent[j];
        if (p < 0) memcpy(world[j], L, sizeof(L));
        else
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    world[j][a * 4 + b] = world[p][a * 4 + 0] * L[0 * 4 + b] + world[p][a * 4 + 1] * L[1 * 4 + b]
                                        + world[p][a * 4 + 2] * L[2 * 4 + b] + world[p][a * 4 + 3] * L[3 * 4 + b];
        const float *IB = d->inv_bind + j * 16;
        float *S = d->skin + j * 16;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                S[a * 4 + b] = world[j][a * 4 + 0] * IB[0 * 4 + b] + world[j][a * 4 + 1] * IB[1 * 4 + b]
                             + world[j][a * 4 + 2] * IB[2 * 4 + b] + world[j][a * 4 + 3] * IB[3 * 4 + b];
    }
}

static void weights(vh_deformer *d, int use_ml, float *w) {
    for (size_t b = 0; b < d->B; ++b) w[b] = d->inp[d->src[b]];
    if (!d->w1) return;
    if (!use_ml) {
        memset(w + d->B, 0, sizeof(float) * (1 + d->K));
        return;
    }
    float x[512];
    for (size_t c = 0; c < d->ml_in && c < 512; ++c) x[c] = (d->inp[c] - d->imean[c]) * d->iscale[c];
    lt_mlp2_f32(x, d->w1, d->b1, d->w2, d->b2, d->hid, d->coef, d->ml_in, d->H, d->K);
    w[d->B] = 1.f;                                   /* ml_mean */
    for (size_t k = 0; k < d->K; ++k)                /* c_k / output.scale_k */
        w[d->B + 1 + k] = (d->coef[k] * d->oscale[k] + d->omean[k]) / d->oscale[k];
}

void vh_deformer_weights(vh_deformer *d, const float *controls, float *w) {
    inputs(d, controls);
    weights(d, 1, w);
}

static void lbs(const vh_deformer *d, const float *p, float *out) {
    for (size_t v = 0; v < d->V; ++v) {
        float m[12] = {0};
        for (int k = 0; k < 4; ++k) {
            float wk = d->sw[v * 4 + k];
            if (wk == 0.f) continue;
            const float *S = d->skin + (size_t)d->sj[v * 4 + k] * 16;
            for (int e = 0; e < 12; ++e) m[e] += wk * S[e];
        }
        const float *q = p + v * 3;
        for (int a = 0; a < 3; ++a) out[v * 3 + a] = m[a * 4] * q[0] + m[a * 4 + 1] * q[1] + m[a * 4 + 2] * q[2] + m[a * 4 + 3];
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
__attribute__((target("avx2,fma"))) static void axpy(float a, const float *x, float *y, size_t n) {
    __m256 va = _mm256_set1_ps(a);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; ++i) y[i] += a * x[i];
}
#else
static void axpy(float a, const float *x, float *y, size_t n) { for (size_t i = 0; i < n; ++i) y[i] += a * x[i]; }
#endif

void vh_deformer_eval(vh_deformer *d, const float *controls, int use_ml, float *out) {
    inputs(d, controls);
    skinning(d);
    weights(d, use_ml, d->wts);
    size_t n = d->V * 3;
    memcpy(d->pos, d->rest, sizeof(float) * n);
    for (size_t m = 0; m < d->M; ++m)
        if (d->wts[m] != 0.f) axpy(d->wts[m], d->morph + m * n, d->pos, n);
    lbs(d, d->pos, out);
}

size_t vh_deformer_batch_scratch(const vh_deformer *d, size_t frames) {
    return frames * d->M + frames * d->V * 3;
}

void vh_deformer_eval_batch(vh_deformer *d, const float *controls, size_t frames, int use_ml, float *scratch,
                            float *out) {
    float *W = scratch, *P = scratch + frames * d->M;
    size_t n = d->V * 3;
    for (size_t f = 0; f < frames; ++f) {
        inputs(d, controls + f * d->C);
        weights(d, use_ml, W + f * d->M);
        memcpy(P + f * n, d->rest, sizeof(float) * n);
    }
    /* P (frames x 3V) += W (frames x M) . morph (M x 3V) */
    sgemm_avx2(frames, n, d->M, 1.f, W, d->M, d->morph, n, 1.f, P, n);
    for (size_t f = 0; f < frames; ++f) {
        inputs(d, controls + f * d->C);
        skinning(d);
        lbs(d, P + f * n, out + f * n);
    }
}

size_t vh_deformer_joints(const vh_deformer *d) { return d->J; }
const float *vh_deformer_rest(const vh_deformer *d) { return d->rest; }
const float *vh_deformer_morph(const vh_deformer *d) { return d->morph; }
const int *vh_deformer_skin_joints(const vh_deformer *d) { return (const int *)d->sj; }
const float *vh_deformer_skin_weights(const vh_deformer *d) { return d->sw; }

void vh_deformer_prepare(vh_deformer *d, const float *controls, int use_ml, float *w, float *skin12) {
    inputs(d, controls);
    skinning(d);
    weights(d, use_ml, w);
    for (size_t j = 0; j < d->J; ++j) memcpy(skin12 + j * 12, d->skin + j * 16, sizeof(float) * 12);
}
