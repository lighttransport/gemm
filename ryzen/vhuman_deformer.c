#include "vhuman_deformer.h"

#include <math.h>
#include <limits.h>
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
    vh_contacts ct;
    int has_ct, ct_iters;
    /* scratch */
    float *inp, *delta, *skin, *hid, *coef, *wts, *pos;
    float *w1t, *w2t;          /* transposed MLP weights for batched GEMMs */
    float *ct_x0, *ct_d;       /* contact smoothing scratch: nc x 3 each */
};

static const void *tensor(st_context *s, const char *name, const char *dtype, int nd, size_t *shape) {
    memset(shape, 0, (size_t)nd * sizeof(*shape));
    int i = safetensors_find(s, name);
    if (i < 0 || strcmp(safetensors_dtype(s, i), dtype) || safetensors_ndims(s, i) != nd) return NULL;
    const uint64_t *sh = safetensors_shape(s, i);
    size_t count = 1;
    for (int k = 0; k < nd; ++k) {
        if (sh[k] > INT_MAX || (sh[k] && count > SIZE_MAX / sh[k])) return NULL;
        shape[k] = (size_t)sh[k];
        count *= shape[k];
    }
    const st_tensor_info *t = &s->tensors[i];
    if (s->data_offset > s->map_size || t->offset % 4 || count > SIZE_MAX / 4 || t->nbytes != count * 4 ||
        t->offset > s->map_size - s->data_offset || t->nbytes > s->map_size - s->data_offset - t->offset) return NULL;
    const void *p = safetensors_data(s, i);
    /* Integer classification remains effective in the CPU fast-math build. */
    if (!strcmp(dtype, "F32"))
        for (size_t k = 0; k < count; ++k) {
            uint32_t bits;
            memcpy(&bits, (const char *)p + k*4, 4);
            if ((bits & 0x7f800000u) == 0x7f800000u) return NULL;
        }
    return p;
}

static const void *tensor_exact(st_context *s, const char *name, const char *dtype, int nd, const size_t *want) {
    size_t sh[3];
    const void *p = tensor(s, name, dtype, nd, sh);
    for (int k = 0; k < nd; ++k) if (sh[k] != want[k]) return NULL;
    return p;
}

static int indices_valid(const int32_t *ids, size_t n, size_t bound, int unique) {
    unsigned char *seen = unique ? calloc(bound, 1) : NULL;
    if (unique && !seen) return 0;
    int ok = 1;
    for (size_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || (size_t)ids[i] >= bound || (seen && seen[ids[i]])) { ok = 0; break; }
        if (seen) seen[ids[i]] = 1;
    }
    free(seen);
    return ok;
}

vh_deformer *vh_deformer_load(const char *path) {
    st_context *s = safetensors_open(path);
    if (!s) return NULL;
    vh_deformer *d = calloc(1, sizeof(*d));
    if (!d) { safetensors_close(s); return NULL; }
    d->st = s;
    size_t sh[3];
#define GET(field, name, dt, nd) do { d->field = tensor(s, name, dt, nd, sh); if (!d->field) goto invalid; } while (0)
#define EXACT(field, name, dt, nd, ...) do { \
    d->field = tensor_exact(s, name, dt, nd, (size_t[]){__VA_ARGS__}); if (!d->field) goto invalid; \
} while (0)
    GET(range, "controls.range", "F32", 2); d->C = sh[0];
    if (!d->C || d->C > 512 || sh[1] != 2) goto invalid;
    GET(jm, "joints.matrix", "F32", 2); d->I = sh[1]; d->J = sh[0] / ATTR;
    if (!d->J || d->J > 64 || sh[0] % ATTR || d->I < d->C || d->I > 512) goto invalid;
    EXACT(parent, "joints.parent", "I32", 1, d->J);
    EXACT(rest_t, "joints.rest_t", "F32", 2, d->J, 3);
    EXACT(rest_R, "joints.rest_R", "F32", 3, d->J, 3, 3);
    EXACT(inv_bind, "joints.inv_bind", "F32", 3, d->J, 4, 4);
    GET(src, "shapes.src", "I32", 1); d->B = sh[0];
    GET(morph, "morph", "F32", 3); d->M = sh[0]; d->V = sh[1];
    if (!d->M || !d->V || d->V > INT_MAX / 3 || sh[2] != 3) goto invalid;
    EXACT(rest, "rest", "F32", 2, d->V, 3);
    EXACT(sj, "skin.joints", "I32", 2, d->V, 4);
    EXACT(sw, "skin.weights", "F32", 2, d->V, 4);
    for (size_t j = 0; j < d->J; ++j) if (d->parent[j] < -1 || d->parent[j] >= (int32_t)j) goto invalid;
    for (size_t i = 0; i < d->C; ++i) if (d->range[2*i] > d->range[2*i+1]) goto invalid;
    if (!indices_valid(d->src, d->B, d->I, 0) || !indices_valid(d->sj, d->V*4, d->J, 0)) goto invalid;
    for (size_t v = 0; v < d->V; ++v) {
        float sum = 0;
        for (int k = 0; k < 4; ++k) {
            float w = d->sw[v*4+k];
            if (w < 0 || w > 1.0001f) goto invalid;
            sum += w;
        }
        if (fabsf(sum-1) > 1e-4f) goto invalid;
    }
    d->P = d->I - d->C;
    if (d->P) {
        GET(corr_in, "corr.inputs", "I32", 2); d->corr_k = sh[1];
        if (sh[0] != d->P || !d->corr_k || d->corr_k > d->I) goto invalid;
        EXACT(corr_w, "corr.weight", "F32", 1, d->P);
        for (size_t p = 0; p < d->P; ++p)
            for (size_t k = 0; k < d->corr_k; ++k) {
                int32_t i = d->corr_in[p*d->corr_k+k];
                if (i < -1 || i >= (int32_t)(d->C+p)) goto invalid;
            }
    }
    if (safetensors_find(s, "ml.fc1.weight") >= 0) {
        GET(w1, "ml.fc1.weight", "F32", 2);
        d->H = sh[0];
        d->ml_in = sh[1];
        if (!d->H || !d->ml_in || d->ml_in > d->I) goto invalid;
        EXACT(b1, "ml.fc1.bias", "F32", 1, d->H);
        GET(w2, "ml.fc2.weight", "F32", 2); d->K = sh[0];
        if (!d->K || sh[1] != d->H || d->M != d->B+1+d->K) goto invalid;
        EXACT(b2, "ml.fc2.bias", "F32", 1, d->K);
        EXACT(imean, "ml.input.mean", "F32", 1, d->ml_in);
        EXACT(iscale, "ml.input.scale", "F32", 1, d->ml_in);
        EXACT(omean, "ml.output.mean", "F32", 1, d->K);
        EXACT(oscale, "ml.output.scale", "F32", 1, d->K);
        for (size_t k = 0; k < d->K; ++k) if (d->oscale[k] == 0) goto invalid;
    } else {
        if (d->M != d->B) goto invalid;
    }
    for (int i = 0; i < s->n_tensors; ++i) {
        const char *name = safetensors_name(s, i);
        if ((!strncmp(name, "ml.", 3) && !d->w1) ||
            (!strncmp(name, "contact.", 8) && safetensors_find(s, "contact.eye_ids") < 0)) goto invalid;
    }
    {
        size_t a[3];
        vh_contacts *c = &d->ct;
#define CT(field, name, dt, nd, ...) do { \
    c->field = tensor_exact(s, "contact." name, dt, nd, (size_t[]){__VA_ARGS__}); if (!c->field) goto invalid; \
} while (0)
        if (safetensors_find(s, "contact.eye_ids") >= 0) {
            c->eye_ids = tensor(s, "contact.eye_ids", "I32", 1, a);
            if (!c->eye_ids) goto invalid;
            c->ne = a[0];
            CT(eye_joint, "eye_joint", "I32", 1, c->ne);
            CT(eye_center, "eye_center", "F32", 2, c->ne, 3);
            CT(eye_thr, "eye_thr", "F32", 1, c->ne);
            c->lip_ids = tensor(s, "contact.lip_ids", "I32", 1, a); if (!c->lip_ids) goto invalid; c->nl = a[0];
            c->sph_center = tensor(s, "contact.sph_center", "F32", 2, a); if (!c->sph_center || a[1] != 3) goto invalid; c->ns = a[0];
            if (c->ns > 512) goto invalid;
            CT(sph_joint, "sph_joint", "I32", 2, c->ns, 2);
            CT(sph_weight, "sph_weight", "F32", 2, c->ns, 2);
            CT(sph_thr, "sph_thr", "F32", 2, c->nl, c->ns);
            c->pair_u = tensor(s, "contact.pair_u", "I32", 1, a); if (!c->pair_u) goto invalid; c->np = a[0];
            CT(pair_l, "pair_l", "I32", 1, c->np);
            CT(pair_floor, "pair_floor", "F32", 1, c->np);
            CT(up_joints, "up_joints", "I32", 1, 2);
            if (!indices_valid(c->eye_ids, c->ne, d->V, 1) || !indices_valid(c->eye_joint, c->ne, d->J, 0) ||
                !indices_valid(c->lip_ids, c->nl, d->V, 1) || !indices_valid(c->sph_joint, c->ns*2, d->J, 0) ||
                !indices_valid(c->pair_u, c->np, d->V, 0) || !indices_valid(c->pair_l, c->np, d->V, 0) ||
                !indices_valid(c->up_joints, 2, d->J, 0)) goto invalid;
            for (size_t i = 0; i < c->ne; ++i) if (c->eye_thr[i] < 0) goto invalid;
            for (size_t i = 0; i < c->nl*c->ns; ++i) if (c->sph_thr[i] < 0) goto invalid;
            for (size_t i = 0; i < c->ns; ++i)
                if (c->sph_weight[2*i] < 0 || c->sph_weight[2*i+1] < 0 ||
                    fabsf(c->sph_weight[2*i]+c->sph_weight[2*i+1]-1) > 1e-4f) goto invalid;
            if (safetensors_find(s, "contact.verts") >= 0) {
                c->verts = tensor(s, "contact.verts", "I32", 1, a); if (!c->verts) goto invalid;
                c->nc = a[0];
                CT(nbr_ptr, "nbr_ptr", "I32", 1, c->nc+1);
                c->nbr_idx = tensor(s, "contact.nbr_idx", "I32", 1, a);
                if (!c->nbr_idx || c->nbr_ptr[0] != 0 || c->nbr_ptr[c->nc] != (int32_t)a[0] ||
                    !indices_valid(c->verts, c->nc, d->V, 1) || !indices_valid(c->nbr_idx, a[0], c->nc, 0)) goto invalid;
                for (size_t i = 0; i < c->nc; ++i) if (c->nbr_ptr[i] < 0 || c->nbr_ptr[i] > c->nbr_ptr[i+1]) goto invalid;
            } else if (safetensors_find(s, "contact.nbr_ptr") >= 0 || safetensors_find(s, "contact.nbr_idx") >= 0) {
                goto invalid;
            }
            d->has_ct = 1;
            d->ct_iters = 4;
        }
#undef CT
    }
#undef GET
#undef EXACT
    if (d->w1) {
        d->w1t = malloc(sizeof(float) * d->H * d->ml_in);
        d->w2t = malloc(sizeof(float) * d->K * d->H);
        if (!d->w1t || !d->w2t) { vh_deformer_free(d); return NULL; }
        for (size_t h = 0; h < d->H; ++h)
            for (size_t i = 0; i < d->ml_in; ++i) d->w1t[i * d->H + h] = d->w1[h * d->ml_in + i];
        for (size_t k = 0; k < d->K; ++k)
            for (size_t h = 0; h < d->H; ++h) d->w2t[h * d->K + k] = d->w2[k * d->H + h];
    }
    if (d->has_ct && d->ct.nc) {
        d->ct_x0 = malloc(sizeof(float) * d->ct.nc * 3);
        d->ct_d = malloc(sizeof(float) * d->ct.nc * 3);
        if (!d->ct_x0 || !d->ct_d) { vh_deformer_free(d); return NULL; }
    }
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
invalid:
    vh_deformer_free(d);
    return NULL;
}

void vh_deformer_free(vh_deformer *d) {
    if (!d) return;
    free(d->w1t); free(d->w2t); free(d->ct_x0); free(d->ct_d);
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

/* skin: J matrices of `stride` floats (16: 4x4, 12: top 3 rows), row-major */
static void xform(const float *S, const float *p, float *o) {
    for (int a = 0; a < 3; ++a) o[a] = S[a * 4] * p[0] + S[a * 4 + 1] * p[1] + S[a * 4 + 2] * p[2] + S[a * 4 + 3];
}

static int push_out(float *x, const float *c, float thr) {
    float v[3] = {x[0] - c[0], x[1] - c[1], x[2] - c[2]};
    float dd = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (dd < thr) {
        float k = thr / (dd > 1e-12f ? dd : 1e-12f);
        for (int a = 0; a < 3; ++a) x[a] = c[a] + v[a] * k;
        return 1;
    }
    return 0;
}

#define VH_SPHERE_PASSES 8

static void solve(const vh_deformer *d, const float *skin, int stride, float *x) {
    const vh_contacts *c = &d->ct;
    float sc[512 * 3];
    size_t ns = c->ns < 512 ? c->ns : 512;
    for (size_t t = 0; t < ns; ++t) {             /* sphere centres: blended skinning of two joints */
        float M[12] = {0};
        for (int k = 0; k < 2; ++k) {
            float w = c->sph_weight[t * 2 + k];
            if (w == 0.f) continue;
            const float *S = skin + (size_t)c->sph_joint[t * 2 + k] * stride;
            for (int e = 0; e < 12; ++e) M[e] += w * S[e];
        }
        xform(M, c->sph_center + t * 3, sc + t * 3);
    }
    const float *H = skin + (size_t)c->up_joints[0] * stride, *Jw = skin + (size_t)c->up_joints[1] * stride;
    float up[3] = {H[1] + Jw[1], H[5] + Jw[5], H[9] + Jw[9]};
    float un = sqrtf(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
    for (int a = 0; a < 3; ++a) up[a] /= un;
    for (int it = 0; it < d->ct_iters; ++it) {    /* stop early: an iteration that moves nothing is final */
        int any = 0;
        for (size_t i = 0; i < c->ne; ++i) {
            float ec[3];
            xform(skin + (size_t)c->eye_joint[i] * stride, c->eye_center + i * 3, ec);
            any |= push_out(x + (size_t)c->eye_ids[i] * 3, ec, c->eye_thr[i]);
        }
        for (size_t p = 0; p < c->np; ++p) {
            float *u = x + (size_t)c->pair_u[p] * 3, *l = x + (size_t)c->pair_l[p] * 3;
            float sep = (u[0] - l[0]) * up[0] + (u[1] - l[1]) * up[1] + (u[2] - l[2]) * up[2];
            float dl = c->pair_floor[p] - sep;
            if (dl > 0.f) {
                dl *= 0.5f;
                any = 1;
                for (int a = 0; a < 3; ++a) { u[a] += dl * up[a]; l[a] -= dl * up[a]; }
            }
        }
        for (size_t l = 0; l < c->nl; ++l) {       /* passes until one pushes nothing (overlapping spheres) */
            float *q = x + (size_t)c->lip_ids[l] * 3;
            for (int pass = 0; pass < VH_SPHERE_PASSES; ++pass) {
                int moved = 0;
                for (size_t t = 0; t < ns; ++t) moved |= push_out(q, sc + t * 3, c->sph_thr[l * c->ns + t]);
                any |= moved;
                if (!moved) break;
            }
        }
        if (!any) break;
    }
}

/* Project; smooth the contact vertices' displacement over their graph
 * (Jacobi: d <- (d + mean of the neighbours' d) / 2); project again. */
static void project(const vh_deformer *d, const float *skin, int stride, float *x) {
    const vh_contacts *c = &d->ct;
    if (c->nc)
        for (size_t i = 0; i < c->nc; ++i) memcpy(d->ct_x0 + i * 3, x + (size_t)c->verts[i] * 3, sizeof(float) * 3);
    solve(d, skin, stride, x);
    if (!c->nc) return;
    for (int step = 0; step < VH_CONTACT_SMOOTH_STEPS; ++step) {
        for (size_t i = 0; i < c->nc; ++i)
            for (int a = 0; a < 3; ++a) d->ct_d[i * 3 + a] = x[(size_t)c->verts[i] * 3 + a] - d->ct_x0[i * 3 + a];
        for (size_t i = 0; i < c->nc; ++i) {
            int b = c->nbr_ptr[i], e = c->nbr_ptr[i + 1];
            float m[3] = {0, 0, 0};
            for (int k = b; k < e; ++k)
                for (int a = 0; a < 3; ++a) m[a] += d->ct_d[(size_t)c->nbr_idx[k] * 3 + a];
            for (int a = 0; a < 3; ++a) {
                float di = d->ct_d[i * 3 + a], mean = e > b ? m[a] / (float)(e - b) : di;
                x[(size_t)c->verts[i] * 3 + a] = d->ct_x0[i * 3 + a] + 0.5f * (di + mean);
            }
        }
    }
    solve(d, skin, stride, x);
}

int vh_deformer_has_contacts(const vh_deformer *d) { return d->has_ct; }
void vh_deformer_set_contact_iterations(vh_deformer *d, int n) { d->ct_iters = d->has_ct && n > 0 ? n : 0; }
int vh_deformer_contact_iterations(const vh_deformer *d) { return d->ct_iters; }
void vh_deformer_project(const vh_deformer *d, const float *skin12, float *pos) {
    if (d->ct_iters) project(d, skin12, 12, pos);
}
const vh_contacts *vh_deformer_contacts(const vh_deformer *d) { return d->has_ct ? &d->ct : NULL; }

void vh_deformer_eval(vh_deformer *d, const float *controls, int use_ml, float *out) {
    inputs(d, controls);
    skinning(d);
    weights(d, use_ml, d->wts);
    size_t n = d->V * 3;
    memcpy(d->pos, d->rest, sizeof(float) * n);
    for (size_t m = 0; m < d->M; ++m)
        if (d->wts[m] != 0.f) axpy(d->wts[m], d->morph + m * n, d->pos, n);
    lbs(d, d->pos, out);
    if (d->ct_iters) project(d, d->skin, 16, out);
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
        if (d->ct_iters) project(d, d->skin, 16, out + f * n);
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

size_t vh_deformer_prepare_scratch(const vh_deformer *d, size_t frames) {
    return frames * (d->ml_in + d->H + d->K);
}

/* A batch of frames: the MLP as two GEMMs (frames x in) . (in x H), then
 * (frames x H) . (H x K); the rest per frame. Same results as
 * vh_deformer_prepare up to float summation order. */
void vh_deformer_prepare_batch(vh_deformer *d, const float *controls, size_t frames, int use_ml, float *wout,
                               float *skin12, float *scratch) {
    int ml = use_ml && d->w1;
    float *X = scratch, *Hh = X + frames * d->ml_in, *Y = Hh + frames * d->H;
    for (size_t f = 0; f < frames; ++f) {
        inputs(d, controls + f * d->C);
        skinning(d);
        weights(d, 0, wout + f * d->M);           /* blendshapes; the ML part below */
        for (size_t j = 0; j < d->J; ++j) memcpy(skin12 + (f * d->J + j) * 12, d->skin + j * 16, sizeof(float) * 12);
        if (ml)
            for (size_t c = 0; c < d->ml_in; ++c) X[f * d->ml_in + c] = (d->inp[c] - d->imean[c]) * d->iscale[c];
    }
    if (!ml) return;
    for (size_t f = 0; f < frames; ++f) memcpy(Hh + f * d->H, d->b1, sizeof(float) * d->H);
    sgemm_avx2(frames, d->H, d->ml_in, 1.f, X, d->ml_in, d->w1t, d->H, 1.f, Hh, d->H);
    for (size_t i = 0; i < frames * d->H; ++i) Hh[i] = Hh[i] > 0.f ? Hh[i] : 0.f;
    for (size_t f = 0; f < frames; ++f) memcpy(Y + f * d->K, d->b2, sizeof(float) * d->K);
    sgemm_avx2(frames, d->K, d->H, 1.f, Hh, d->H, d->w2t, d->K, 1.f, Y, d->K);
    for (size_t f = 0; f < frames; ++f) {
        float *w = wout + f * d->M;
        w[d->B] = 1.f;
        for (size_t k = 0; k < d->K; ++k) w[d->B + 1 + k] = (Y[f * d->K + k] * d->oscale[k] + d->omean[k]) / d->oscale[k];
    }
}
