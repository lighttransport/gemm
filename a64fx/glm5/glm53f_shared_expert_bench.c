/* Real-weight GLM-5.3F shared-expert decode stream. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <arm_sve.h>
#include <omp.h>
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_arch.h"
#include "glm53f_expert_kern.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint8_t *gate, *up, *down;
    float *gate_scale, *up_scale, *down_scale;
    int inter;
    size_t bytes;
} shared_layer;

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *read_tensor(glm53f_st_context *ctx, const char *name,
                         const st_tensor_info **info) {
    const st_tensor_info *t = glm53f_st_find(ctx, name, NULL);
    void *p = NULL;
    if (!t || posix_memalign(&p, 256, t->nbytes) ||
        glm53f_st_read(ctx, name, 0, p, t->nbytes)) {
        free(p);
        return NULL;
    }
    *info = t;
    return p;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : getenv("GLM53F_MODEL_DIR");
    int reps = getenv("REPS") ? atoi(getenv("REPS")) : 20;
    int shard_part = getenv("SHARD_PART") ? atoi(getenv("SHARD_PART")) : -1;
    glm53f_st_context *ctx;
    shared_layer layer[42] = {{0}};
    float *x, *gate, *up, *act, *y;
    size_t total = 0;
    double best = 1e30, sum = 0;
    int bad = 0, max_inter = 0;
    if (!model || reps < 1) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    ctx = glm53f_st_open(model);
    if (!ctx) return 2;
    for (int l = 3; l < 45; ++l) {
        char name[512];
        const st_tensor_info *g, *u, *d, *gs, *us, *ds;
        shared_layer *s = &layer[l - 3];
#define READ(FIELD, SUFFIX, INFO) do { \
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.shared_experts.%s", l, SUFFIX); \
        s->FIELD = read_tensor(ctx, name, &INFO); \
        if (!s->FIELD) { fprintf(stderr, "failed: %s\n", name); return 2; } \
        s->bytes += INFO->nbytes; \
    } while (0)
        READ(gate, "gate_proj.weight", g);
        READ(gate_scale, "gate_proj.weight_scale_inv", gs);
        READ(up, "up_proj.weight", u);
        READ(up_scale, "up_proj.weight_scale_inv", us);
        READ(down, "down_proj.weight", d);
        READ(down_scale, "down_proj.weight_scale_inv", ds);
#undef READ
        if (g->n_dims != 2 || u->n_dims != 2 || d->n_dims != 2 ||
            g->shape[0] != u->shape[0] || g->shape[1] != 4096 ||
            u->shape[1] != 4096 || d->shape[0] != 4096 ||
            d->shape[1] != g->shape[0]) return 2;
        s->inter = (int)g->shape[0];
        if (shard_part >= 0) {
            int begin, count, sb = 4096 / 128;
            uint8_t *ng, *nu, *nd;
            float *ngs, *nus, *nds;
            if (shard_part >= 12 || s->inter != 2048 ||
                glm53f_block_aligned_slice(2048, 128, shard_part, 12, &begin, &count)) return 2;
            posix_memalign((void **)&ng, 256, (size_t)count * 4096);
            posix_memalign((void **)&nu, 256, (size_t)count * 4096);
            posix_memalign((void **)&nd, 256, (size_t)4096 * count);
            posix_memalign((void **)&ngs, 256, (size_t)(count / 128) * sb * 4);
            posix_memalign((void **)&nus, 256, (size_t)(count / 128) * sb * 4);
            posix_memalign((void **)&nds, 256, (size_t)32 * (count / 128) * 4);
            if (!ng || !nu || !nd || !ngs || !nus || !nds) return 2;
            memcpy(ng, s->gate + (size_t)begin * 4096, (size_t)count * 4096);
            memcpy(nu, s->up + (size_t)begin * 4096, (size_t)count * 4096);
            memcpy(ngs, s->gate_scale + (size_t)(begin / 128) * sb,
                   (size_t)(count / 128) * sb * 4);
            memcpy(nus, s->up_scale + (size_t)(begin / 128) * sb,
                   (size_t)(count / 128) * sb * 4);
            for (int r = 0; r < 4096; ++r) {
                memcpy(nd + (size_t)r * count,
                       s->down + (size_t)r * 2048 + begin, (size_t)count);
                memcpy(nds + (size_t)r / 128 * (count / 128),
                       s->down_scale + (size_t)r / 128 * 16 + begin / 128,
                       (size_t)(count / 128) * 4);
            }
            free(s->down_scale); free(s->down); free(s->up_scale);
            free(s->up); free(s->gate_scale); free(s->gate);
            s->gate=ng;s->up=nu;s->down=nd;s->gate_scale=ngs;s->up_scale=nus;s->down_scale=nds;
            s->inter=count;
            s->bytes=(size_t)count*4096*3 + ((size_t)2*(count/128)*32 + (size_t)32*(count/128))*4;
        }
        if (s->inter > max_inter) max_inter = s->inter;
        total += s->bytes;
    }
    glm53f_st_close(ctx);
    posix_memalign((void **)&x, 256, 4096 * 4);
    posix_memalign((void **)&gate, 256, (size_t)max_inter * 4);
    posix_memalign((void **)&up, 256, (size_t)max_inter * 4);
    posix_memalign((void **)&act, 256, (size_t)max_inter * 4);
    posix_memalign((void **)&y, 256, 4096 * 4);
    if (!x || !gate || !up || !act || !y) return 2;
    for (int i = 0; i < 4096; ++i) x[i] = (float)((i % 29) - 14) * .001f;
    double token_start = 0;
#pragma omp parallel shared(token_start, best, sum, x, gate, up, act, y, layer)
    for (int q = 0; q < reps; ++q) {
#pragma omp single
        token_start = now_sec();
        for (int l = 0; l < 42; ++l) {
            shared_layer *s = &layer[l];
            int n8 = s->inter / 8;
#pragma omp for schedule(static)
            for (int bi = 0; bi < 2 * n8; ++bi) {
                int second = bi >= n8, r = (second ? bi - n8 : bi) * 8;
                const uint8_t *w = second ? s->up : s->gate;
                const float *scale = second ? s->up_scale : s->gate_scale;
                float *dst = second ? up : gate;
                glm53f_matvec_fp8_bits_8(
                    dst + r, w + (size_t)r * 4096,
                    scale + (size_t)(r / 128) * 32, x, 4096);
            }
#pragma omp single
            {
                for (int r = n8 * 8; r < s->inter; ++r) {
                    gate[r] = glm53f_dot_fp8_block128(
                        s->gate + (size_t)r * 4096,
                        s->gate_scale + (size_t)(r / 128) * 32, x, 4096);
                    up[r] = glm53f_dot_fp8_block128(
                        s->up + (size_t)r * 4096,
                        s->up_scale + (size_t)(r / 128) * 32, x, 4096);
                }
            }
#pragma omp for schedule(static)
            for (int i = 0; i < s->inter; ++i) {
                float g = gate[i];
                act[i] = (g / (1.0f + expf(-g))) * up[i];
            }
#pragma omp for schedule(static)
            for (int bi = 0; bi < 4096 / 8; ++bi) {
                int r = bi * 8, blocks = (s->inter + 127) / 128;
                glm53f_matvec_fp8_bits_8(
                    y + r, s->down + (size_t)r * s->inter,
                    s->down_scale + (size_t)(r / 128) * blocks,
                    act, s->inter);
            }
#pragma omp single
            x[(l * 97 + q) & 4095] += y[(l * 131 + q) & 4095] * 1e-5f;
        }
#pragma omp single
        {
            double dt = now_sec() - token_start;
            if (dt < best) best = dt;
            sum += dt;
        }
    }
    double checksum = 0;
    for (int i = 0; i < 4096; ++i) { bad += !isfinite(y[i]); checksum += y[i]; }
    printf("GLM53F_SHARED_EXPERT layers=42 shard_part=%d inter=%d weight_MiB=%.3f reps=%d best_ms_tok=%.3f mean_ms_tok=%.3f tok_s=%.2f bad=%d checksum=%.9g\n",
           shard_part, max_inter, total / 1048576.0, reps, best * 1e3, sum / reps * 1e3,
           1.0 / best, bad, checksum);
    free(y); free(act); free(up); free(gate); free(x);
    for (int l = 0; l < 42; ++l) {
        free(layer[l].down_scale); free(layer[l].down);
        free(layer[l].up_scale); free(layer[l].up);
        free(layer[l].gate_scale); free(layer[l].gate);
    }
    return bad ? 1 : 0;
}
