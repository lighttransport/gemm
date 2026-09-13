/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include "gn_cpu.h"
#include "gn_internal.h"
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define SAFETENSORS_IMPLEMENTATION
#include "../common/safetensors.h"
#define SAFETENSORS_WRITER_IMPLEMENTATION
#include "../common/safetensors_writer.h"
static _Thread_local char error_text[256];
int gn_fail(const char *s) {
    snprintf(error_text, sizeof(error_text), "%s", s);
    return -1;
}
#define fail gn_fail
const char *gn_error(void) { return error_text; }
static void *alloc(gn_model *m, size_t n) {
    if (n > SIZE_MAX / sizeof(float) || n * sizeof(float) > m->cfg.memory_limit - m->bytes) {
        fail("network memory limit exceeded");
        return NULL;
    }
    void *p = calloc(n, sizeof(float));
    if (!p) {
        fail("network allocation failed");
        return NULL;
    }
    m->bytes += n * sizeof(float);
    return p;
}
static void release(gn_model *m, float *p, size_t n) {
    if (p) {
        free(p);
        m->bytes -= n * sizeof(float);
    }
}
static uint64_t random_u64(gn_model *m) {
    uint64_t z = (m->rng += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
uint64_t gn_random(gn_model *m) { return random_u64(m); }
static Param *parameter(gn_model *m, const char *name, size_t r, size_t c, int init, int learned) {
    for (size_t i = 0; i < m->np; i++)
        if (!strcmp(m->p[i].name, name))
            return &m->p[i];
    if (m->np == GN_PARAMS || r == 0 || c == 0 || r > SIZE_MAX / c) {
        fail("invalid parameter shape/count");
        return NULL;
    }
    Param *p = &m->p[m->np++];
    snprintf(p->name, sizeof(p->name), "%s", name);
    p->r = r;
    p->c = c;
    p->learned = learned;
    size_t n = r * c;
    p->x = alloc(m, n);
    if (learned) {
        p->g = alloc(m, n);
        p->m = alloc(m, n);
        p->v = alloc(m, n);
        m->parameters += n;
    }
    if (!p->x || (learned && (!p->g || !p->m || !p->v)))
        return NULL;
    float scale = sqrtf(6.0f / (float)c);
    for (size_t i = 0; i < n; i++)
        p->x[i] = init == 1 ? 1.0f
                  : init == 2
                      ? ((float)(random_u64(m) >> 40) * (1.0f / 16777216.0f) * 2 - 1) * scale
                      : 0;
    return p;
}
static size_t aux_count(const gn_model *m, const Node *n) {
    if (n->kind == BN)
        return 2 * n->c;
    if (n->kind == LN)
        return 2 * n->r;
    if (n->kind == ATTENTION)
        return n->r * (m->cfg.side * m->cfg.side) * (n->c / m->cfg.head_dim);
    return 0;
}
static void clear_graph(gn_model *m) {
    for (size_t i = 0; i < m->nn; i++) {
        Node *n = &m->n[i];
        release(m, n->x, n->r * n->c);
        release(m, n->g, n->r * n->c);
        release(m, n->aux, aux_count(m, n));
        memset(n, 0, sizeof(*n));
    }
    m->nn = 0;
}
static int node(gn_model *m, Kind kind, int a, int b, size_t r, size_t c, int k, const char *name) {
    if (m->nn == GN_NODES || !r || !c || r > SIZE_MAX / c) {
        fail("invalid graph shape/count");
        return -1;
    }
    int id = (int)m->nn++;
    Node *n = &m->n[id];
    n->kind = kind;
    n->a = a;
    n->b = b;
    n->r = r;
    n->c = c;
    n->k = k;
    n->x = alloc(m, r * c);
    if (m->training)
        n->g = alloc(m, r * c);
    size_t na = aux_count(m, n);
    if (na)
        n->aux = alloc(m, na);
    if (!n->x || (m->training && !n->g) || (na && !n->aux))
        return -1;
    char key[96];
    if (kind == LINEAR || kind == CONV) {
        size_t in = kind == LINEAR ? (size_t)k : m->n[a].c * (size_t)k * k;
        snprintf(key, sizeof(key), "%s.weight", name);
        n->w = parameter(m, key, c, in, 2, 1);
        snprintf(key, sizeof(key), "%s.bias", name);
        n->bias = parameter(m, key, 1, c, 0, 1);
    } else if (kind == BN || kind == LN) {
        snprintf(key, sizeof(key), "%s.weight", name);
        n->w = parameter(m, key, 1, c, 1, 1);
        snprintf(key, sizeof(key), "%s.bias", name);
        n->bias = parameter(m, key, 1, c, 0, 1);
        if (kind == BN) {
            snprintf(key, sizeof(key), "%s.mean", name);
            n->mean = parameter(m, key, 1, c, 0, 0);
            snprintf(key, sizeof(key), "%s.variance", name);
            n->variance = parameter(m, key, 1, c, 1, 0);
        }
    } else if (kind == ATTENTION) {
        size_t side = 2 * m->cfg.side - 1;
        snprintf(key, sizeof(key), "%s.relative_bias", name);
        n->w = parameter(m, key, side * side, c / m->cfg.head_dim, 0, 1);
    }
    if ((kind == LINEAR || kind == CONV || kind == BN || kind == LN) && (!n->w || !n->bias))
        return -1;
    if (kind == BN && (!n->mean || !n->variance))
        return -1;
    if (kind == ATTENTION && !n->w)
        return -1;
    return id;
}
#define N(kind, a, b, r, c, k, name)                                                               \
    do {                                                                                           \
        t = node(m, kind, a, b, r, c, k, name);                                                    \
        if (t < 0)                                                                                 \
            goto bad;                                                                              \
    } while (0)
static int build(gn_model *m, size_t batch, int training) {
    if (m->nn && m->batch == batch && m->training == training) {
        /* Keep the tape and buffers for steady-state training/inference. */
        /* GPU node gradients are resident and cleared by prepare(); their
         * unused host mirrors must not add hundreds of MB of CPU writes. */
        if (training && !m->gpu)
            for (size_t i = 0; i < m->nn; i++)
                memset(m->n[i].g, 0, m->n[i].r * m->n[i].c * sizeof(float));
        return 0;
    }
    clear_graph(m);
    m->batch = batch;
    m->training = training;
    size_t S = m->cfg.side * m->cfg.side, C = m->cfg.channels, R = batch * S;
    int t, activation = m->cfg.version >= 2 ? SILU : RELU;
    if (!batch || batch > 4096)
        return fail("batch must be in 1..4096");
    N(INPUT, -1, -1, R, m->cfg.inputs, 0, "");
    N(CONV, t, -1, R, C, 5, "stem");
    N(BN, t, -1, R, C, 0, "stem_norm");
    N(activation, t, -1, R, C, 0, "");
    for (unsigned block = 0; block < m->cfg.blocks; block++) {
        int skip = t;
        char key[80];
#define KEY(suffix) snprintf(key, sizeof(key), "blocks.%u.%s", block, suffix)
        if (m->cfg.attention_every && (block + 1) % m->cfg.attention_every == 0) {
            KEY("norm1");
            N(LN, t, -1, R, C, 0, key);
            KEY("qkv");
            N(LINEAR, t, -1, R, 3 * C, (int)C, key);
            KEY("attention");
            N(ATTENTION, t, -1, R, C, 0, key);
            KEY("proj");
            N(LINEAR, t, -1, R, C, (int)C, key);
            N(ADD, t, skip, R, C, 0, "");
            skip = t;
            KEY("norm2");
            N(LN, t, -1, R, C, 0, key);
            int normalized = t;
            KEY("gate");
            N(LINEAR, t, -1, R, 2 * C, (int)C, key);
            N(SILU, t, -1, R, 2 * C, 0, "");
            int gate = t;
            KEY("up");
            N(LINEAR, normalized, -1, R, 2 * C, (int)C, key);
            N(MUL, t, gate, R, 2 * C, 0, "");
            KEY("down");
            N(LINEAR, t, -1, R, C, (int)(2 * C), key);
            N(ADD, t, skip, R, C, 0, "");
        } else {
            KEY("conv1");
            N(CONV, t, -1, R, C, 3, key);
            KEY("norm1");
            N(BN, t, -1, R, C, 0, key);
            N(activation, t, -1, R, C, 0, "");
            KEY("conv2");
            N(CONV, t, -1, R, C, 3, key);
            KEY("norm2");
            N(BN, t, -1, R, C, 0, key);
            N(ADD, t, skip, R, C, 0, "");
            N(activation, t, -1, R, C, 0, "");
        }
#undef KEY
    }
    int trunk = t;
    N(LINEAR, trunk, -1, R, m->cfg.actions, (int)C, "policy");
    m->policy = t;
    N(LINEAR, trunk, -1, R, m->cfg.value_channels, (int)C, "value.project");
    N(activation, t, -1, R, m->cfg.value_channels, 0, "");
    N(LINEAR, t, -1, batch, m->cfg.value_hidden, (int)(S * m->cfg.value_channels), "value.hidden");
    N(activation, t, -1, batch, m->cfg.value_hidden, 0, "");
    N(LINEAR, t, -1, batch, 3, (int)m->cfg.value_hidden, "value.output");
    m->value = t;
    return 0;
bad:
    clear_graph(m);
    return -1;
}
#undef N
gn_config gn_default_config(void) {
    gn_config c = {1, 9, 80, 139, 256, 20, 5, 32, 32, 256, 1, (size_t)6 * 1024 * 1024 * 1024};
    return c;
}
gn_model *gn_create(const gn_config *c, const char *backend, int device) {
    (void)device;
    error_text[0] = 0;
    if (!backend ||
        (strcmp(backend, "cpu") && strcmp(backend, "cuda") && strcmp(backend, "hip") &&
         strcmp(backend, "cuda-fp32") && strcmp(backend, "hip-fp32") &&
         strcmp(backend, "cuda-legacy") && strcmp(backend, "hip-legacy") &&
         strcmp(backend, "hip-blaslt") && strcmp(backend, "cuda-int8") &&
         strcmp(backend, "cuda-int16") && strcmp(backend, "hip-int8") &&
         strcmp(backend, "hip-int8-i64") && strcmp(backend, "hip-int16") &&
         strcmp(backend, "hip-bf16") && strcmp(backend, "hip-bf16-acc") &&
         strcmp(backend, "hip-bf16-acc128") && strcmp(backend, "hip-bf16-blaslt") &&
         strcmp(backend, "hip-bf16x3") && strcmp(backend, "hip-bf16x3-blaslt") &&
         strcmp(backend, "hip-bf16x3-blaslt-tuned") && strcmp(backend, "hip-bf16x3-dx-blaslt") &&
         strcmp(backend, "hip-bf16x3-dw-blaslt") && strcmp(backend, "hip-bf16x3-forward-blaslt") &&
         strcmp(backend, "hip-bf16-mixed") && strcmp(backend, "hip-bf16-mixed-blaslt") &&
         strcmp(backend, "hip-fp16-blaslt"))) {
        fail("unknown backend; expected cpu, cuda or hip");
        return NULL;
    }
    if (!c || (c->version != 1 && c->version != 2) || c->side < 1 || c->side > 19 || !c->inputs ||
        c->inputs > 1024 || !c->actions || c->actions > 1024 || c->channels < 1 ||
        c->channels > 1024 || c->blocks > 80 || !c->head_dim || c->channels % c->head_dim ||
        !c->value_channels || c->value_channels > 1024 || !c->value_hidden ||
        c->value_hidden > 4096 || !c->memory_limit) {
        fail("invalid network configuration");
        return NULL;
    }
    gn_model *m = calloc(1, sizeof(*m));
    if (!m) {
        fail("model allocation failed");
        return NULL;
    }
    m->cfg = *c;
    m->rng = c->seed;
    if (build(m, 1, 0)) {
        gn_destroy(m);
        return NULL;
    }
    clear_graph(m);
    if (strcmp(backend, "cpu")) {
        m->gpu = gn_gpu_open(backend, device, c->memory_limit);
        if (!m->gpu) {
            gn_destroy(m);
            return NULL;
        }
    }
    return m;
}
void gn_destroy(gn_model *m) {
    if (!m)
        return;
    clear_graph(m);
    gn_gpu_close(m->gpu);
    gn_cpu_close(m->cpu);
    for (size_t i = 0; i < m->np; i++) {
        free(m->p[i].x);
        free(m->p[i].g);
        free(m->p[i].m);
        free(m->p[i].v);
    }
    free(m);
}
const gn_config *gn_configuration(const gn_model *m) { return m ? &m->cfg : NULL; }
size_t gn_parameter_count(const gn_model *m) { return m ? m->parameters : 0; }
size_t gn_memory_used(const gn_model *m) { return m ? m->bytes : 0; }
double gn_gemm_flops(const gn_model *m) {
    if (!m)
        return 0;
    double operations = 0;
    for (size_t i = 0; i < m->nn; i++) {
        const Node *n = &m->n[i];
        if (n->kind == LINEAR || n->kind == CONV) {
            double K = n->kind == LINEAR ? n->k : (double)m->n[n->a].c * n->k * n->k;
            operations += (m->training ? 6 : 2) * (double)n->r * n->c * K;
        }
    }
    return operations;
}
double gn_matrix_flops(const gn_model *m) {
    double operations = gn_gemm_flops(m);
    if (m)
        for (size_t i = 0; i < m->nn; i++) {
            const Node *n = &m->n[i];
            if (n->kind == ATTENTION)
                operations +=
                    (m->training ? 12 : 4) * (double)n->r * n->c * m->cfg.side * m->cfg.side;
        }
    return operations;
}
uint64_t gn_step(const gn_model *m) { return m ? m->step : 0; }
size_t gn_tensor_count(const gn_model *m) { return m ? m->np : 0; }
const char *gn_tensor(gn_model *m, size_t i, size_t *r, size_t *c, float **x, float **g) {
    if (!m || i >= m->np)
        return NULL;
    if (m->gpu && gn_gpu_sync(m))
        return NULL;
    Param *p = &m->p[i];
    if (r)
        *r = p->r;
    if (c)
        *c = p->c;
    if (x)
        *x = p->x;
    if (g)
        *g = p->g;
    return p->name;
}
static void softmax(float *p, size_t n) {
    float top = -FLT_MAX;
    for (size_t i = 0; i < n; i++)
        if (p[i] > top)
            top = p[i];
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        p[i] = expf(p[i] - top);
        sum += p[i];
    }
    for (size_t i = 0; i < n; i++)
        p[i] /= (float)sum;
}
static int forward(gn_model *m, const float *input) {
    for (size_t i = 0; i < m->n[0].r * m->n[0].c; i++)
        if (!isfinite(input[i]))
            return fail("non-finite network input");
    if (m->gpu)
        return gn_gpu_forward(m, input);
    memcpy(m->n[0].x, input, m->n[0].r * m->n[0].c * sizeof(float));
    size_t side = m->cfg.side, S = side * side;
    for (size_t z = 1; z < m->nn; z++) {
        Node *n = &m->n[z], *a = &m->n[n->a];
        float *x = a->x, *y = n->x;
        size_t R = n->r, C = n->c;
        switch (n->kind) {
        case LINEAR:
        case CONV:
            if (gn_cpu_projection(m, n))
                return -1;
            break;
        case ADD:
        case MUL:
        case RELU:
        case SILU:
            for (size_t i = 0; i < R * C; i++)
                y[i] = n->kind == ADD    ? x[i] + m->n[n->b].x[i]
                       : n->kind == MUL  ? x[i] * m->n[n->b].x[i]
                       : n->kind == RELU ? fmaxf(0, x[i])
                                         : x[i] / (1 + expf(-x[i]));
            break;
        case BN:
            for (size_t c = 0; c < C; c++) {
                double mean = 0, var = 0;
                if (m->training) {
                    for (size_t r = 0; r < R; r++)
                        mean += x[r * C + c];
                    mean /= R;
                    for (size_t r = 0; r < R; r++) {
                        double d = x[r * C + c] - mean;
                        var += d * d;
                    }
                    var /= R;
                    n->mean->x[c] = 0.9f * n->mean->x[c] + 0.1f * (float)mean;
                    n->variance->x[c] =
                        0.9f * n->variance->x[c] + 0.1f * (float)(R > 1 ? var * R / (R - 1) : var);
                } else {
                    mean = n->mean->x[c];
                    var = n->variance->x[c];
                }
                float inv = 1 / sqrtf((float)var + 1e-5f);
                n->aux[c] = (float)mean;
                n->aux[C + c] = inv;
                for (size_t r = 0; r < R; r++)
                    y[r * C + c] = (x[r * C + c] - (float)mean) * inv * n->w->x[c] + n->bias->x[c];
            }
            break;
        case LN:
            for (size_t r = 0; r < R; r++) {
                double mean = 0, var = 0;
                for (size_t c = 0; c < C; c++)
                    mean += x[r * C + c];
                mean /= C;
                for (size_t c = 0; c < C; c++) {
                    double d = x[r * C + c] - mean;
                    var += d * d;
                }
                var /= C;
                float inv = 1 / sqrtf((float)var + 1e-5f);
                n->aux[r] = (float)mean;
                n->aux[R + r] = inv;
                for (size_t c = 0; c < C; c++)
                    y[r * C + c] = (x[r * C + c] - (float)mean) * inv * n->w->x[c] + n->bias->x[c];
            }
            break;
        case ATTENTION: {
            size_t D = m->cfg.head_dim, H = C / D, span = 2 * side - 1;
            float scale = 1 / sqrtf((float)D);
            for (size_t b = 0; b < R / S; b++)
                for (size_t h = 0; h < H; h++)
                    for (size_t i = 0; i < S; i++) {
                        float *prob = n->aux + ((b * H + h) * S + i) * S;
                        for (size_t j = 0; j < S; j++) {
                            size_t rel = (i / side + side - 1 - j / side) * span +
                                         (i % side + side - 1 - j % side);
                            float v = 0;
                            for (size_t d = 0; d < D; d++)
                                v += x[(b * S + i) * 3 * C + h * D + d] *
                                     x[(b * S + j) * 3 * C + C + h * D + d];
                            prob[j] = v * scale + n->w->x[rel * H + h];
                        }
                        softmax(prob, S);
                        for (size_t d = 0; d < D; d++) {
                            float v = 0;
                            for (size_t j = 0; j < S; j++)
                                v += prob[j] * x[(b * S + j) * 3 * C + 2 * C + h * D + d];
                            y[(b * S + i) * C + h * D + d] = v;
                        }
                    }
            break;
        }
        default:
            return fail("unknown operation");
        }
        for (size_t i = 0; i < R * C; i++)
            if (!isfinite(y[i]))
                return fail("non-finite network activation");
    }
    return 0;
}
static void backward_graph(gn_model *m) {
    size_t side = m->cfg.side, S = side * side;
    for (size_t z = m->nn; z-- > 1;) {
        Node *n = &m->n[z], *a = &m->n[n->a];
        float *x = a->x, *g = n->g;
        size_t R = n->r, C = n->c;
        switch (n->kind) {
        case LINEAR:
            for (size_t r = 0; r < R; r++)
                for (size_t c = 0; c < C; c++) {
                    float v = g[r * C + c];
                    n->bias->g[c] += v;
                    gn_cpu_accumulate(a->g + r * n->k, n->w->g + c * n->k, x + r * n->k,
                                      n->w->x + c * n->k, v, (size_t)n->k);
                }
            break;
        case CONV:
            for (size_t r = 0; r < R; r++)
                for (size_t c = 0; c < C; c++) {
                    float v = g[r * C + c];
                    n->bias->g[c] += v;
                    int yy = (int)((r % S) / side), xx = (int)(r % side), K = n->k;
                    for (int dy = 0; dy < K; dy++)
                        for (int dx = 0; dx < K; dx++) {
                            int sy = yy + dy - K / 2, sx = xx + dx - K / 2;
                            if (sy < 0 || sx < 0 || sy >= (int)side || sx >= (int)side)
                                continue;
                            size_t base = ((r / S) * S + (size_t)sy * side + sx) * a->c,
                                   wb = (c * K * K + dy * K + dx) * a->c;
                            gn_cpu_accumulate(a->g + base, n->w->g + wb, x + base, n->w->x + wb, v,
                                              a->c);
                        }
                }
            break;
        case ADD:
        case MUL:
        case RELU:
        case SILU:
            for (size_t i = 0; i < R * C; i++) {
                if (n->kind == ADD) {
                    a->g[i] += g[i];
                    m->n[n->b].g[i] += g[i];
                } else if (n->kind == MUL) {
                    a->g[i] += g[i] * m->n[n->b].x[i];
                    m->n[n->b].g[i] += g[i] * x[i];
                } else if (n->kind == RELU)
                    a->g[i] += x[i] > 0 ? g[i] : 0;
                else {
                    float s = 1 / (1 + expf(-x[i]));
                    a->g[i] += g[i] * s * (1 + x[i] * (1 - s));
                }
            }
            break;
        case BN:
            for (size_t c = 0; c < C; c++) {
                double sum = 0, prod = 0;
                float mean = n->aux[c], inv = n->aux[C + c];
                for (size_t r = 0; r < R; r++) {
                    sum += g[r * C + c];
                    prod += g[r * C + c] * (x[r * C + c] - mean) * inv;
                }
                n->w->g[c] += (float)prod;
                n->bias->g[c] += (float)sum;
                for (size_t r = 0; r < R; r++)
                    a->g[r * C + c] += n->w->x[c] * inv *
                                       (g[r * C + c] - (float)(sum / R) -
                                        (x[r * C + c] - mean) * inv * (float)(prod / R));
            }
            break;
        case LN:
            for (size_t r = 0; r < R; r++) {
                double sum = 0, prod = 0;
                float mean = n->aux[r], inv = n->aux[R + r];
                for (size_t c = 0; c < C; c++) {
                    float t = (x[r * C + c] - mean) * inv, dy = g[r * C + c] * n->w->x[c];
                    sum += dy;
                    prod += dy * t;
                    n->w->g[c] += g[r * C + c] * t;
                    n->bias->g[c] += g[r * C + c];
                }
                for (size_t c = 0; c < C; c++)
                    a->g[r * C + c] += inv * (g[r * C + c] * n->w->x[c] - (float)(sum / C) -
                                              (x[r * C + c] - mean) * inv * (float)(prod / C));
            }
            break;
        case ATTENTION: {
            size_t D = m->cfg.head_dim, H = C / D, span = 2 * side - 1;
            float scale = 1 / sqrtf((float)D);
            for (size_t b = 0; b < R / S; b++)
                for (size_t h = 0; h < H; h++)
                    for (size_t i = 0; i < S; i++) {
                        float dp[361], *prob = n->aux + ((b * H + h) * S + i) * S;
                        double dot = 0;
                        for (size_t j = 0; j < S; j++) {
                            float v = 0;
                            for (size_t d = 0; d < D; d++) {
                                float dy = g[(b * S + i) * C + h * D + d];
                                v += dy * x[(b * S + j) * 3 * C + 2 * C + h * D + d];
                                a->g[(b * S + j) * 3 * C + 2 * C + h * D + d] += prob[j] * dy;
                            }
                            dp[j] = v;
                            dot += v * prob[j];
                        }
                        for (size_t j = 0; j < S; j++) {
                            float ds = prob[j] * (dp[j] - (float)dot);
                            size_t rel = (i / side + side - 1 - j / side) * span +
                                         (i % side + side - 1 - j % side);
                            n->w->g[rel * H + h] += ds;
                            for (size_t d = 0; d < D; d++) {
                                size_t qi = (b * S + i) * 3 * C + h * D + d,
                                       ki = (b * S + j) * 3 * C + C + h * D + d;
                                a->g[qi] += ds * scale * x[ki];
                                a->g[ki] += ds * scale * x[qi];
                            }
                        }
                    }
            break;
        }
        default:
            break;
        }
    }
}
int gn_infer(gn_model *m, size_t batch, const float *input, float *policy, float *wdl) {
    if (!m || !input || !policy || !wdl)
        return fail("null inference argument");
    if (build(m, batch, 0) || forward(m, input))
        return -1;
    Node *p = &m->n[m->policy], *v = &m->n[m->value];
    memcpy(policy, p->x, p->r * p->c * sizeof(float));
    memcpy(wdl, v->x, batch * 3 * sizeof(float));
    for (size_t i = 0; i < batch; i++)
        softmax(wdl + i * 3, 3);
    return 0;
}
void gn_zero_grad(gn_model *m) {
    if (!m)
        return;
    if (m->gpu)
        gn_gpu_zero(m);
    for (size_t i = 0; i < m->np; i++)
        if (m->p[i].g)
            memset(m->p[i].g, 0, m->p[i].r * m->p[i].c * sizeof(float));
    m->accumulated = 0;
}
int gn_backward(gn_model *m, size_t batch, const float *input, const float *target,
                const uint32_t *label, gn_metrics *metrics) {
    if (!m || !input || !target || !label || !metrics)
        return fail("null training argument");
    size_t A = m->cfg.side * m->cfg.side * m->cfg.actions;
    if (!batch || batch > 4096)
        return fail("invalid training batch");
    for (size_t b = 0; b < batch; b++) {
        double sum = 0;
        size_t legal = 0;
        if (label[b] > 2)
            return fail("WDL label out of range");
        for (size_t i = 0; i < A; i++) {
            float t = target[b * A + i];
            if (!isfinite(t) || (t < 0 && t != -1) || t > 1)
                return fail("invalid policy target");
            if (t >= 0) {
                sum += t;
                legal++;
            }
        }
        if (!legal || fabs(sum - 1) > 1e-4)
            return fail("policy targets must sum to one over legal actions");
    }
    if (build(m, batch, 1) || forward(m, input))
        return -1;
    if (m->gpu) {
        if (gn_gpu_backward(m, target, label, metrics))
            return -1;
        m->accumulated += batch;
        return 0;
    }
    Node *p = &m->n[m->policy], *v = &m->n[m->value];
    double lp = 0, lv = 0;
    for (size_t b = 0; b < batch; b++) {
        float top = -FLT_MAX;
        for (size_t i = 0; i < A; i++)
            if (target[b * A + i] >= 0 && p->x[b * A + i] > top)
                top = p->x[b * A + i];
        double sum = 0;
        for (size_t i = 0; i < A; i++)
            if (target[b * A + i] >= 0)
                sum += exp((double)p->x[b * A + i] - top);
        double logz = log(sum) + top;
        for (size_t i = 0; i < A; i++)
            if (target[b * A + i] >= 0) {
                p->g[b * A + i] = (float)exp(p->x[b * A + i] - logz) - target[b * A + i];
                lp += target[b * A + i] * (logz - p->x[b * A + i]);
            }
        float probs[3];
        memcpy(probs, v->x + 3 * b, sizeof(probs));
        softmax(probs, 3);
        top = fmaxf(v->x[3 * b], fmaxf(v->x[3 * b + 1], v->x[3 * b + 2]));
        sum = 0;
        for (size_t i = 0; i < 3; i++) {
            sum += exp((double)v->x[3 * b + i] - top);
            v->g[3 * b + i] = probs[i] - (i == label[b]);
        }
        lv += log(sum) + top - v->x[3 * b + label[b]];
    }
    backward_graph(m);
    m->accumulated += batch;
    metrics->policy = (float)(lp / batch);
    metrics->value = (float)(lv / batch);
    metrics->step = m->step;
    metrics->grad_norm = 0;
    return 0;
}
int gn_update(gn_model *m, float lr, float decay, float clip, gn_metrics *metrics) {
    if (!m || !m->accumulated || !isfinite(lr) || lr <= 0 || !isfinite(decay) || decay < 0 ||
        !isfinite(clip) || clip <= 0)
        return fail("invalid optimizer arguments");
    if (m->gpu)
        return gn_gpu_update(m, lr, decay, clip, metrics);
    double norm = 0;
    for (size_t i = 0; i < m->np; i++)
        if (m->p[i].learned)
            for (size_t j = 0; j < m->p[i].r * m->p[i].c; j++) {
                double v = m->p[i].g[j] / m->accumulated;
                if (!isfinite(v))
                    return fail("non-finite gradient; optimizer not advanced");
                norm += v * v;
            }
    norm = sqrt(norm);
    float scale = (float)((norm > clip ? clip / norm : 1) / m->accumulated);
    m->step++;
    float b1 = 1 - (float)pow(0.9, (double)m->step), b2 = 1 - (float)pow(0.999, (double)m->step);
    for (size_t i = 0; i < m->np; i++) {
        Param *p = &m->p[i];
        if (!p->learned)
            continue;
        for (size_t j = 0; j < p->r * p->c; j++) {
            float g = p->g[j] * scale;
            p->m[j] = 0.9f * p->m[j] + 0.1f * g;
            p->v[j] = 0.999f * p->v[j] + 0.001f * g * g;
            p->x[j] -= lr * ((p->m[j] / b1) / (sqrtf(p->v[j] / b2) + 1e-8f) + decay * p->x[j]);
        }
    }
    if (metrics) {
        metrics->grad_norm = (float)norm;
        metrics->step = m->step;
    }
    gn_zero_grad(m);
    return 0;
}
static void config_words(const gn_config *c, uint64_t *v) {
    v[0] = c->version;
    v[1] = c->side;
    v[2] = c->inputs;
    v[3] = c->actions;
    v[4] = c->channels;
    v[5] = c->blocks;
    v[6] = c->attention_every;
    v[7] = c->head_dim;
    v[8] = c->value_channels;
    v[9] = c->value_hidden;
    v[10] = c->seed;
    v[11] = c->memory_limit;
}
int gn_save(const gn_model *m, const char *path) {
    if (!m || !path || strlen(path) > 3500)
        return fail("invalid checkpoint path");
    if (m->accumulated)
        return fail("checkpoint requires an optimizer boundary");
    if (m->gpu && gn_gpu_sync((gn_model *)m))
        return -1;
    for (size_t i = 0; i < m->np; i++)
        for (size_t j = 0; j < m->p[i].r * m->p[i].c; j++)
            if (!isfinite(m->p[i].x[j]) ||
                (m->p[i].learned &&
                 (!isfinite(m->p[i].m[j]) || !isfinite(m->p[i].v[j]) || m->p[i].v[j] < 0)))
                return fail("refusing to checkpoint non-finite/invalid optimizer state");
    stw_writer *w = stw_create();
    if (!w)
        return fail("checkpoint writer allocation failed");
    uint64_t cfg[12], shape[2] = {12, 0}, state[2] = {m->step, m->rng};
    config_words(&m->cfg, cfg);
    int rc = stw_add(w, "__config", "U64", shape, 1, cfg, sizeof(cfg));
    shape[0] = 2;
    rc |= stw_add(w, "__state", "U64", shape, 1, state, sizeof(state));
    for (size_t i = 0; i < m->np; i++) {
        const Param *p = &m->p[i];
        shape[0] = p->r;
        shape[1] = p->c;
        char key[128];
        rc |= stw_add(w, p->name, "F32", shape, 2, p->x, p->r * p->c * 4);
        if (p->learned) {
            snprintf(key, sizeof(key), "adam.m.%s", p->name);
            rc |= stw_add(w, key, "F32", shape, 2, p->m, p->r * p->c * 4);
            snprintf(key, sizeof(key), "adam.v.%s", p->name);
            rc |= stw_add(w, key, "F32", shape, 2, p->v, p->r * p->c * 4);
        }
    }
    char temp[4096];
    snprintf(temp, sizeof(temp), "%s.partial.%ld", path, (long)getpid());
    if (!rc)
        rc = stw_save(w, temp);
    stw_destroy(w);
    if (!rc) {
        int fd = open(temp, O_RDONLY);
        if (fd < 0)
            rc = -1;
        else {
            rc = fsync(fd);
            if (close(fd))
                rc = -1;
        }
    }
    if (rc || rename(temp, path)) {
        unlink(temp);
        return fail("checkpoint write/rename failed");
    }
    return 0;
}
static int load_array(st_context *s, const char *name, float *out, size_t r, size_t c) {
    int id = safetensors_find(s, name);
    if (id < 0 || strcmp(safetensors_dtype(s, id), "F32") || safetensors_ndims(s, id) != 2 ||
        safetensors_shape(s, id)[0] != r || safetensors_shape(s, id)[1] != c ||
        safetensors_nbytes(s, id) != r * c * 4)
        return -1;
    const float *x = safetensors_data(s, id);
    for (size_t i = 0; i < r * c; i++)
        if (!isfinite(x[i]))
            return -1;
    memcpy(out, x, r * c * 4);
    return 0;
}
gn_model *gn_load(const char *path, const char *backend, int device) {
    st_context *s = safetensors_open(path);
    if (!s) {
        fail("cannot open safetensors checkpoint");
        return NULL;
    }
    int id = safetensors_find(s, "__config"), st = safetensors_find(s, "__state");
    gn_model *m = NULL;
    if (id < 0 || st < 0 || strcmp(safetensors_dtype(s, id), "U64") ||
        strcmp(safetensors_dtype(s, st), "U64") || safetensors_nbytes(s, id) != 96 ||
        safetensors_nbytes(s, st) != 16 || safetensors_ndims(s, id) != 1 ||
        safetensors_ndims(s, st) != 1 || safetensors_shape(s, id)[0] != 12 ||
        safetensors_shape(s, st)[0] != 2) {
        fail("invalid checkpoint metadata");
        goto done;
    }
    uint64_t v[12], state[2];
    memcpy(v, safetensors_data(s, id), 96);
    memcpy(state, safetensors_data(s, st), 16);
    for (int i = 0; i < 10; i++)
        if (v[i] > UINT32_MAX) {
            fail("checkpoint configuration overflow");
            goto done;
        }
    gn_config c = {(uint32_t)v[0], (uint32_t)v[1], (uint32_t)v[2], (uint32_t)v[3],
                   (uint32_t)v[4], (uint32_t)v[5], (uint32_t)v[6], (uint32_t)v[7],
                   (uint32_t)v[8], (uint32_t)v[9], v[10],          (size_t)v[11]};
    if (c.memory_limit > (size_t)6 * 1024 * 1024 * 1024)
        c.memory_limit = (size_t)6 * 1024 * 1024 * 1024;
    m = gn_create(&c, backend, device);
    if (!m)
        goto done;
    for (size_t i = 0; i < m->np; i++) {
        Param *p = &m->p[i];
        char key[128];
        int rc = load_array(s, p->name, p->x, p->r, p->c);
        if (p->learned) {
            snprintf(key, sizeof(key), "adam.m.%s", p->name);
            rc |= load_array(s, key, p->m, p->r, p->c);
            snprintf(key, sizeof(key), "adam.v.%s", p->name);
            rc |= load_array(s, key, p->v, p->r, p->c);
        }
        if (rc) {
            gn_destroy(m);
            m = NULL;
            fail("checkpoint tensor shape, dtype or value mismatch");
            goto done;
        }
    }
    m->step = state[0];
    m->rng = state[1];
done:
    safetensors_close(s);
    return m;
}
