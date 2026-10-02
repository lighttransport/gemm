#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_iq_scale_words.h"
#include "glm53f_team.h"
enum { MAX_ROWS = 4096, MAX_BLOCKS = 16, GUARD = 64 };
static uint32_t rng = 1;
static uint32_t random_value(void) { rng = rng * 1664525u + 1013904223u; return rng; }
static void *allocate(size_t bytes) { void *p = NULL; if (posix_memalign(&p,256,bytes)) abort(); return p; }
typedef struct { int rows, blocks, q5, mode; const uint8_t *w; const iqf_act *a; float *out; } call;
static void worker(void *context) {
    call *c = context; size_t rb = (size_t)c->blocks * (c->q5 ? 176 : 144);
#pragma omp for schedule(static)
    for (int r = 0; r < c->rows; r += 8) {
        int n = c->rows - r < 8 ? c->rows - r : 8;
        if (c->mode) iqfw_rows(c->out + r, c->w + (size_t)r * rb, rb, n,
            c->a, c->blocks, c->q5);
        else iqf_rows(c->out + r, c->w + (size_t)r * rb, rb, n, c->a, c->blocks, c->q5);
    }
}
static void run(call *c) {
    if (glm53f_team_available()) glm53f_team_dispatch(worker,c);
    else {
#pragma omp parallel
        worker(c);
    }
}
static void fill(uint8_t *w, iqf_act *a, int rows, int blocks, int q5, int mode) {
    const int bytes = q5 ? 176 : 144; rng = 1;
    for (int i = 0; i < rows * blocks; ++i) {
        uint8_t *p = w + (size_t)i * bytes;
        for (int j = 0; j < bytes; ++j) p[j] = (uint8_t)(random_value() >> 24);
        uint16_t d = (uint16_t)(0x1800 + (random_value() & 1023)), dm = 0x1400;
        if (mode == 4) d |= 0x8000;
        if (mode == 5) d = dm = 0;
        if (mode == 6) d = 1;
        memcpy(p,&d,2); memcpy(p+2,&dm,2);
    }
    iqf_src_block src[MAX_BLOCKS];
    for (int b = 0; b < blocks; ++b) {
        src[b].d = mode == 5 ? 0.f : (float)(1 + (random_value() & 1023)) * 0x1p-17f;
        if (mode == 6) src[b].d = 0x1p-100f;
        for (int k = 0; k < 256; ++k) {
            int x = (int)(random_value() >> 24) - 128;
            if (mode == 0) x = 0;
            if (mode == 2) x = x < 0 ? -127 : 127;
            if (mode == 3 && (k & 7)) x = 0;
            src[b].q[k] = (int8_t)x;
        }
    }
    iqf_prepare(a,src,blocks);
}
typedef struct {
    uint8_t *w;
    iqf_act *a;
    float *ref, *out;
    int cases, failed;
} unit_context;
static void unit_cases(void *context) {
    unit_context *u = context;
    uint8_t *w = u->w;
    iqf_act *a = u->a;
    float *ref = u->ref, *out = u->out;
    const int rows[] = {1,3,4,8,17,31,64,512,1024,4096}, blocks[] = {1,2,16};
    for (int q5 = 0; q5 < 2; ++q5)
        for (unsigned r = 0; r < sizeof(rows) / sizeof(*rows); ++r)
            for (unsigned b = 0; b < sizeof(blocks) / sizeof(*blocks); ++b)
                for (int input = 0; input < 7; ++input) {
                    fill(w,a,rows[r],blocks[b],q5,input);
                    call c = {rows[r],blocks[b],q5,0,w,a,ref+GUARD};
                    run(&c);
                    for (int g = 0; g < GUARD; ++g) {
                        out[g] = -71.25f;
                        out[GUARD+rows[r]+g] = -72.25f;
                    }
                    c.mode = 1; c.out = out+GUARD;
                    run(&c);
                    for (int i = 0; i < rows[r]; ++i) {
                        uint32_t rb, ob;
                        memcpy(&rb,ref+GUARD+i,4); memcpy(&ob,out+GUARD+i,4);
                        if (rb != ob || (ob & 0x7f800000u) == 0x7f800000u) {
                            printf("GLM53F_IQ_SCALE_WORDS_FAIL q5=%d rows=%d blocks=%d input=%d row=%d reference=%08x actual=%08x\n",
                                   q5,rows[r],blocks[b],input,i,rb,ob);
                            u->failed = 1;
                            break;
                        }
                    }
                    for (int g = 0; g < GUARD; ++g)
                        u->failed |= out[g] != -71.25f || out[GUARD+rows[r]+g] != -72.25f;
                    ++u->cases;
                }
}
int main(void) {
    if (svcntw() != 16) {
        fprintf(stderr,"GLM53F_IQ_SCALE_WORDS requires512-bit SVE\n");
        return 2;
    }
    unit_context u = {
        allocate((size_t)MAX_ROWS*MAX_BLOCKS*176),
        allocate(MAX_BLOCKS*sizeof(iqf_act)),
        allocate((MAX_ROWS+2*GUARD)*sizeof(float)),
        allocate((MAX_ROWS+2*GUARD)*sizeof(float)),
        0, 0
    };
    iqf_init();
    unit_cases(&u);
    glm53f_team_run(unit_cases,&u);
    printf("GLM53F_IQ_SCALE_WORDS_UNIT threads=%d cases=%d rows_and_guards=BIT_EXACT %s\n",
           omp_get_max_threads(),u.cases,u.failed ? "FAIL" : "PASS");
    free(u.w); free(u.a); free(u.ref); free(u.out);
    return u.failed ? 1 : 0;
}
