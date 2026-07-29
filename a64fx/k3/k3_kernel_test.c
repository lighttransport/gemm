#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3_kernels.h"
#include "ggml_dequant.h"

static uint64_t rng_state = UINT64_C(0x4b33413634465801);
static uint64_t rng_next(void) {
    uint64_t z = (rng_state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static float rnd(void) { return ((rng_next() >> 40) / 8388608.0f) - 1.0f; }
static void fill(float *x, size_t n, float scale) { for (size_t i = 0; i < n; ++i) x[i] = rnd() * scale; }
static double now_sec(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static float max_abs(const float *a, const float *b, size_t n) {
    float e = 0.0f; for (size_t i = 0; i < n; ++i) e = fmaxf(e, fabsf(a[i] - b[i])); return e;
}
static int check(const char *name, const float *a, const float *b, size_t n, float tol) {
    float e = max_abs(a, b, n); printf("[%-12s] max_abs=%9.3e %s\n", name, e, e <= tol ? "OK" : "FAIL"); return e > tol;
}

static float mxfp4_ref(const uint8_t *w, const uint8_t *scale, const float *x, int n) {
    double sum = 0.0;
    for (int b = 0; b < n / 32; ++b) {
        float s = ggml_e8m0_to_fp32(scale[b]);
        for (int j = 0; j < 16; ++j) {
            uint8_t p = w[b * 16 + j];
            sum += (double)ds4f_kvalues_mxfp4_f32[p & 15] * s * x[b * 32 + j];
            sum += (double)ds4f_kvalues_mxfp4_f32[p >> 4] * s * x[b * 32 + j + 16];
        }
    }
    return (float)sum;
}

int main(void) {
    int fail = 0;
    enum { N = 257, H = 2, K = 32, V = 24, T = 37 };
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *r = malloc(N * sizeof(float));
    float *o = malloc(N * sizeof(float));
    fill(a, N, 3.0f); fill(b, N, 1.0f);
    k3_rmsnorm_ref(r, a, b, N, 1e-6f); k3_rmsnorm_sve(o, a, b, N, 1e-6f);
    fail |= check("rmsnorm", r, o, N, 3e-5f);
    k3_gated_rmsnorm_ref(r, a, b, b, N, 1e-6f);
    k3_gated_rmsnorm_sve(o, a, b, b, N, 1e-6f);
    fail |= check("gated-rms", r, o, N, 3e-5f);
    k3_situ_ref(r, a, b, N); k3_situ_sve(o, a, b, N);
    fail |= check("situ", r, o, N, 0.0f);
    fill(a, N, 12.0f); fill(b, N, 40.0f);
    k3_situ_ref(r, a, b, N); k3_situ_fast_sve(o, a, b, N);
    fail |= check("situ-fexpa", r, o, N, 2e-3f);

    float conv_x[64], conv_w[64 * 4], conv_b[64], conv_s0[64 * 3] = {0}, conv_s1[64 * 3] = {0};
    float conv_r[64], conv_o[64]; fill(conv_w, 64 * 4, .3f); fill(conv_b, 64, .1f);
    for (int t = 0; t < 9; ++t) {
        fill(conv_x, 64, 1.0f);
        k3_conv_step_ref(conv_r, conv_x, conv_s0, conv_w, conv_b, 64, 4);
        k3_conv_step_sve(conv_o, conv_x, conv_s1, conv_w, conv_b, 64, 4);
    }
    fail |= check("conv", conv_r, conv_o, 64, 0.0f);

    size_t qn = H * K, vn = H * V, sn = (size_t)H * V * K;
    float *q = malloc(qn * 4), *key = malloc(qn * 4), *val = malloc(vn * 4);
    float *gate = malloc(qn * 4), beta[H], *s0 = calloc(sn, 4), *s1 = calloc(sn, 4);
    float *kr = malloc(vn * 4), *ko = malloc(vn * 4);
    for (int step = 0; step < 5; ++step) {
        fill(q, qn, 1); fill(key, qn, 1); fill(val, vn, 1); fill(gate, qn, .2f);
        float alog[K], draw[H*K];
        memset(draw, 0, sizeof(draw));
        for (int d=0;d<K;++d) alog[d]=-1.0f+.01f*d;
        k3_kda_log_decay(gate, gate, alog, draw, H, K);
        for (int h = 0; h < H; ++h) { k3_l2_normalize_ref(q + h*K, K, 1e-6f); k3_l2_normalize_ref(key+h*K,K,1e-6f); beta[h] = .2f + .6f * (rnd()+1)*.5f; }
        k3_kda_step_ref(kr, q, key, val, gate, beta, s0, H, K, V);
        k3_kda_step_sve(ko, q, key, val, gate, beta, s1, H, K, V);
    }
    fail |= check("kda-output", kr, ko, vn, 3e-5f);
    fail |= check("kda-state", s0, s1, sn, 3e-5f);

    float *keys = malloc((size_t)T * K * 4), *values = malloc((size_t)T * V * 4);
    fill(q, K, 1); fill(keys, (size_t)T*K, 1); fill(values, (size_t)T*V, 1);
    k3_attention_ref(kr, q, keys, values, T, K, V); k3_attention_sve(ko, q, keys, values, T, K, V);
    fail |= check("mla-online", kr, ko, V, 2e-5f);

    float candidates[13 * 64], scores[13], ar[64], ao[64]; fill(candidates, 13*64, 1); fill(scores,13,2);
    k3_attnres_ref(ar,candidates,scores,13,64); k3_attnres_sve(ao,candidates,scores,13,64);
    fail |= check("attnres", ar, ao, 64, 3e-7f);

    float logits[32], biasv[32], weights[4]; int idx[4]; fill(logits,32,3); fill(biasv,32,.1f);
    k3_router_topk(logits,biasv,32,4,idx,weights); float wsum=weights[0]+weights[1]+weights[2]+weights[3];
    int unique = idx[0]!=idx[1] && idx[0]!=idx[2] && idx[0]!=idx[3] && idx[1]!=idx[2] && idx[1]!=idx[3] && idx[2]!=idx[3];
    printf("[router      ] indices=%d,%d,%d,%d sum=%.8f %s\n",idx[0],idx[1],idx[2],idx[3],wsum,(unique&&fabsf(wsum-1)<1e-6f)?"OK":"FAIL");
    fail |= !unique || fabsf(wsum-1) >= 1e-6f;

    enum { MK = 3584, MR = 8, MI = 2000 };
    uint8_t *mw = malloc((size_t)MR * MK / 2), *ms = malloc((size_t)MR * MK / 32);
    float *mx = malloc(MK * sizeof(float)), mout[MR], mref[MR];
    fill(mx, MK, 1.0f);
    for (size_t i = 0; i < (size_t)MR * MK / 2; ++i) mw[i] = (uint8_t)rng_next();
    for (size_t i = 0; i < (size_t)MR * MK / 32; ++i) ms[i] = 127;
    const uint8_t *wr[MR], *sr[MR];
    for (int j = 0; j < MR; ++j) {
        wr[j] = mw + (size_t)j * MK / 2; sr[j] = ms + (size_t)j * MK / 32;
        mref[j] = mxfp4_ref(wr[j], sr[j], mx, MK);
    }
    matvec_mxfp4_8row(mout,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],
                      sr[0],sr[1],sr[2],sr[3],sr[4],sr[5],sr[6],sr[7],mx,MK);
    fail |= check("mxfp4", mref, mout, MR, 2e-4f);
    volatile float msink = 0.0f;
    double mt0=now_sec();
    for (int i=0;i<MI;++i) {
        matvec_mxfp4_8row(mout,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],
                          sr[0],sr[1],sr[2],sr[3],sr[4],sr[5],sr[6],sr[7],mx,MK);
        msink += mout[i & 7];
    }
    double mdt=now_sec()-mt0;
    double mbytes=(double)MI*MR*(MK/2+MK/32);
    printf("CALIBRATION mxfp4_gbps=%.3f mxfp4_us=%.3f sink=%g\n",
           mbytes/mdt/1e9,mdt/MI*1e6,(double)msink);

    enum { BK = 128, BV = 128, BI = 250 };
    float *bq=aligned_alloc(256,BK*4), *bk=aligned_alloc(256,BK*4), *bv=aligned_alloc(256,BV*4);
    float *bg=aligned_alloc(256,BK*4), *bs=aligned_alloc(256,(size_t)BK*BV*4), *bo=aligned_alloc(256,BV*4);
    fill(bq,BK,1); fill(bk,BK,1); fill(bv,BV,1); fill(bg,BK,.02f); memset(bs,0,(size_t)BK*BV*4);
    k3_l2_normalize_sve(bq,BK,1e-6f); k3_l2_normalize_sve(bk,BK,1e-6f); float bb=.5f;
    for(int i=0;i<20;++i) k3_kda_step_sve(bo,bq,bk,bv,bg,&bb,bs,1,BK,BV);
    double t0=now_sec(); for(int i=0;i<BI;++i) k3_kda_step_sve(bo,bq,bk,bv,bg,&bb,bs,1,BK,BV); double dt=now_sec()-t0;
    double ops=(double)BI*BV*BK*6.0;
    printf("CALIBRATION kda_gops=%.3f kda_us=%.3f\n",ops/dt/1e9,dt/BI*1e6);
#if defined(__ARM_FEATURE_SVE)
    printf("CALIBRATION sve_bits=%d\n",(int)svcntb()*8);
#else
    printf("CALIBRATION sve_bits=0\n");
#endif

    free(a);free(b);free(r);free(o);free(q);free(key);free(val);free(gate);free(s0);free(s1);free(kr);free(ko);
    free(keys);free(values);free(bq);free(bk);free(bv);free(bg);free(bs);free(bo);
    free(mw);free(ms);free(mx);
    printf("K3 kernel tests: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
