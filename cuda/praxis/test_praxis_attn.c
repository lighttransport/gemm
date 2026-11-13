/*
 * test_praxis_attn.c - the block-sparse bf16 flash kernel vs the dense-masked fp32 reference kernel at REAL token counts
 * (encoder K'=2: ~4.1k tokens; decoder: ~10k tokens; 24 Q / 8 KV heads, head_dim 128, teacher-forcing codes, gate).
 *   cuda/praxis/test_praxis_attn [--hd 128 --heads 24 --kv 8] [--tpf 208 --F 6 --K 2 --S 64]
 */
#define SAFETENSORS_IMPLEMENTATION
#define CUDA_PRAXIS_IMPLEMENTATION
#include "cuda_praxis_runner.h"

static unsigned long long rs = 88172645463325252ULL;
static float rnd(void) { float a = 0; for (int i = 0; i < 4; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; a += (float)(rs >> 40) / 16777216.0f; } return (a - 2.0f) * 1.732f; }
static float rbf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); u &= 0xffff0000u; memcpy(&f, &u, 4); return f; }

static int run_case(praxis_runner *r, const char *name, const int *code, int T, int H, int KVH, int hd) {
    size_t nq = (size_t)T * H * hd, nk = (size_t)T * KVH * hd;
    float *q = (float *)malloc(nq * 4), *k = (float *)malloc(nk * 4), *v = (float *)malloc(nk * 4), *g = (float *)malloc((size_t)T * H * 4);
    unsigned short *qb = (unsigned short *)malloc(nq * 2), *kb = (unsigned short *)malloc(nk * 2), *vb = (unsigned short *)malloc(nk * 2);
    for (size_t i = 0; i < nq; i++) { q[i] = rbf(rnd()); uint32_t u; memcpy(&u, &q[i], 4); qb[i] = (unsigned short)(u >> 16); }
    for (size_t i = 0; i < nk; i++) { k[i] = rbf(rnd()); uint32_t u; memcpy(&u, &k[i], 4); kb[i] = (unsigned short)(u >> 16); v[i] = rbf(rnd()); memcpy(&u, &v[i], 4); vb[i] = (unsigned short)(u >> 16); }
    for (size_t i = 0; i < (size_t)T * H; i++) g[i] = rnd();
    CUdeviceptr dq = pr_malloc(r, nq * 4), dk = pr_malloc(r, nk * 4), dv = pr_malloc(r, nk * 4), dg = pr_malloc(r, (size_t)T * H * 4), dref = pr_malloc(r, nq * 4);
    CUdeviceptr dqb = pr_malloc(r, nq * 2), dkb = pr_malloc(r, nk * 2), dvb = pr_malloc(r, nk * 2), dob = pr_malloc(r, nq * 2);
    CUdeviceptr dcode = pr_malloc(r, (size_t)T * 4), dtile = pr_malloc(r, 3 * (size_t)((T + 31) / 32) * 4 + 16);
    cuMemcpyHtoD(dq, q, nq * 4); cuMemcpyHtoD(dk, k, nk * 4); cuMemcpyHtoD(dv, v, nk * 4); cuMemcpyHtoD(dg, g, (size_t)T * H * 4);
    cuMemcpyHtoD(dqb, qb, nq * 2); cuMemcpyHtoD(dkb, kb, nk * 2); cuMemcpyHtoD(dvb, vb, nk * 2); cuMemcpyHtoD(dcode, code, (size_t)T * 4);
    cuLaunchKernel(r->f_ctiles, ((T + 31) / 32 + 63) / 64, 1, 1, 64, 1, 1, 0, 0, (void *[]) { PD(dcode), PI(T), PI(32), PD(dtile) }, NULL);
    CUevent e0, e1; cuEventCreate(&e0, 0); cuEventCreate(&e1, 0);
    if (pr_attn(r, dq, (long long)H * hd, dk, (long long)KVH * hd, dv, (long long)KVH * hd, dref, (long long)H * hd, dg, H, KVH, hd, T, T, 1, 0, dcode, dcode)) return 1;
    cuEventRecord(e0, 0);
    if (pr_flash(r, dqb, (long long)H * hd, dkb, (long long)KVH * hd, dvb, (long long)KVH * hd, dob, (long long)H * hd, dg, H, KVH, hd, T, T, dcode, dcode, dtile)) return 1;
    cuEventRecord(e1, 0); cuCtxSynchronize();
    float ms = 0; cuEventElapsedTime(&ms, e0, e1);
    float *ref = (float *)malloc(nq * 4); unsigned short *ob = (unsigned short *)malloc(nq * 2);
    cuMemcpyDtoH(ref, dref, nq * 4); cuMemcpyDtoH(ob, dob, nq * 2);
    double d = 0, s = 0, mx = 0;
    for (size_t i = 0; i < nq; i++) { uint32_t u = (uint32_t)ob[i] << 16; float f; memcpy(&f, &u, 4); d += (double)(f - ref[i]) * (f - ref[i]); s += (double)ref[i] * ref[i]; if (fabs(f - ref[i]) > mx) mx = fabs(f - ref[i]); }
    double rel = sqrt(d / s);
    printf("%-28s T=%5d H=%d KV=%d hd=%d : flash vs dense-masked fp32 rel-L2 %.3e (max abs %.3e), flash kernel %.2f ms %s\n", name, T, H, KVH, hd, rel, mx, ms, rel < 1e-2 ? "OK" : "FAIL");
    return rel < 1e-2 ? 0 : 1;
}


/* action-expert attention: Q = K + K*nf + K*F queries (proprio | future | actions), cross keys [globals | clean frames | noised frames] with the
 * window / own-chunk rule, self keys = own chunk. Interval-flash (bf16) vs the dense byte-masked fp32 kernel. */
static int run_expert_case(praxis_runner *r, const char *name, int K, int F, int nf, int tpf, int G, int NH, int hd, int cross) {
    int cs = F, n_c = (K - 1) * F, n_n = K * F, T = K * F, Q = K + K * nf + T, KV = cross ? G + (n_c + n_n) * tpf : Q;
    int off = 0, window = n_c + n_n, inner = NH * hd;
    int *iv = (int *)malloc((size_t)Q * 24);
    unsigned char *mask = (unsigned char *)calloc((size_t)Q * KV, 1);
    for (int q = 0; q < Q; q++) {
        int qch = q < K ? q : q < K * (1 + nf) ? (q - K) / (nf > 0 ? nf : 1) : (q - K * (1 + nf)) / cs;
        int qst = qch * cs + off;
        int *v = iv + q * 6;
        if (cross) {
            int clo = qst - window < 0 ? 0 : qst - window, chi = qst < n_c ? qst : n_c;
            v[0] = 0; v[1] = G;
            if (chi > clo) { v[2] = G + clo * tpf; v[3] = G + chi * tpf; } else { v[2] = v[3] = 0; }
            int n0 = qch * cs, n1 = (qch + 1) * cs < n_n ? (qch + 1) * cs : n_n;
            v[4] = G + (n_c + n0) * tpf; v[5] = G + (n_c + n1) * tpf;
            for (int j = 0; j < KV; j++) {   /* the rule of action_expert.trunk_mask_mods, written independently of the intervals */
                int ok;
                if (j < G) ok = 1; else { int fr = (j - G) / tpf; ok = fr < n_c ? (fr < qst && fr >= qst - window) : ((fr - n_c) / cs == qch); }
                mask[(size_t)q * KV + j] = (unsigned char)ok;
            }
        } else {
            v[0] = qch; v[1] = qch + 1; v[2] = K + qch * nf; v[3] = K + (qch + 1) * nf; v[4] = K * (1 + nf) + qch * cs; v[5] = K * (1 + nf) + (qch + 1) * cs;
            for (int j = 0; j < Q; j++) {
                int jch = j < K ? j : j < K * (1 + nf) ? (j - K) / (nf > 0 ? nf : 1) : (j - K * (1 + nf)) / cs;
                mask[(size_t)q * Q + j] = (unsigned char)(jch == qch);
            }
        }
    }
    size_t nq = (size_t)Q * inner, nk = (size_t)KV * inner;
    float *q = (float *)malloc(nq * 4), *kk = (float *)malloc(nk * 4), *vv = (float *)malloc(nk * 4);
    unsigned short *qb = (unsigned short *)malloc(nq * 2), *kb = (unsigned short *)malloc(nk * 2), *vb = (unsigned short *)malloc(nk * 2);
    for (size_t i = 0; i < nq; i++) { q[i] = rbf(rnd()); uint32_t u; memcpy(&u, &q[i], 4); qb[i] = (unsigned short)(u >> 16); }
    for (size_t i = 0; i < nk; i++) { kk[i] = rbf(rnd()); uint32_t u; memcpy(&u, &kk[i], 4); kb[i] = (unsigned short)(u >> 16); vv[i] = rbf(rnd()); memcpy(&u, &vv[i], 4); vb[i] = (unsigned short)(u >> 16); }
    CUdeviceptr dq = pr_malloc(r, nq * 4), dk = pr_malloc(r, nk * 4), dv = pr_malloc(r, nk * 4), dref = pr_malloc(r, nq * 4);
    CUdeviceptr dqb = pr_malloc(r, nq * 2), dkb = pr_malloc(r, nk * 2), dvb = pr_malloc(r, nk * 2), dob = pr_malloc(r, nq * 2);
    CUdeviceptr dmask = pr_malloc(r, (size_t)Q * KV), div_ = pr_malloc(r, (size_t)Q * 24);
    cuMemcpyHtoD(dq, q, nq * 4); cuMemcpyHtoD(dk, kk, nk * 4); cuMemcpyHtoD(dv, vv, nk * 4);
    cuMemcpyHtoD(dqb, qb, nq * 2); cuMemcpyHtoD(dkb, kb, nk * 2); cuMemcpyHtoD(dvb, vb, nk * 2);
    cuMemcpyHtoD(dmask, mask, (size_t)Q * KV); cuMemcpyHtoD(div_, iv, (size_t)Q * 24);
    CUevent e0, e1, e2, e3; cuEventCreate(&e0, 0); cuEventCreate(&e1, 0); cuEventCreate(&e2, 0); cuEventCreate(&e3, 0);
    for (int w = 0; w < 2; w++) {   /* warm-up + timed */
        cuEventRecord(e0, 0);
        if (pr_attn(r, dq, inner, dk, inner, dv, inner, dref, inner, 0, NH, NH, hd, Q, KV, 0, dmask, 0, 0)) return 1;
        cuEventRecord(e1, 0);
        if (pr_flash_ex(r, dqb, inner, dkb, inner, dvb, inner, dob, inner, 0, NH, NH, hd, Q, KV, div_, 0, 0, 1)) return 1;
        cuEventRecord(e2, 0); cuEventRecord(e3, 0);
        cuCtxSynchronize();
    }
    float md = 0, mf = 0; cuEventElapsedTime(&md, e0, e1); cuEventElapsedTime(&mf, e1, e2);
    float *ref = (float *)malloc(nq * 4); unsigned short *ob = (unsigned short *)malloc(nq * 2);
    cuMemcpyDtoH(ref, dref, nq * 4); cuMemcpyDtoH(ob, dob, nq * 2);
    double d = 0, sm = 0;
    for (size_t i = 0; i < nq; i++) { uint32_t u = (uint32_t)ob[i] << 16; float f; memcpy(&f, &u, 4); d += (double)(f - ref[i]) * (f - ref[i]); sm += (double)ref[i] * ref[i]; }
    double rel = sqrt(d / sm);
    printf("%-26s Q=%3d KV=%5d NH=%d hd=%d : interval-flash vs dense fp32 rel-L2 %.3e | dense %.3f ms -> flash %.3f ms (%.1fx) %s\n", name, Q, KV, NH, hd, rel, md, mf, md / mf, rel < 1e-2 ? "OK" : "FAIL");
    free(iv); free(mask); free(q); free(kk); free(vv); free(qb); free(kb); free(vb); free(ref); free(ob);
    cuMemFree(dq); cuMemFree(dk); cuMemFree(dv); cuMemFree(dref); cuMemFree(dqb); cuMemFree(dkb); cuMemFree(dvb); cuMemFree(dob); cuMemFree(dmask); cuMemFree(div_);
    return rel < 1e-2 ? 0 : 1;
}

int main(int argc, char **argv) {
    int hd = 128, H = 24, KVH = 8, tpf = 208, F = 6, K = 2, S = 64, nreg = 4;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--hd") && i + 1 < argc) hd = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) KVH = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tpf") && i + 1 < argc) tpf = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--F") && i + 1 < argc) F = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--S") && i + 1 < argc) S = atoi(argv[++i]);
    }
    praxis_runner *r = praxis_load_synthetic("/mnt/disk1/praxis/mid", 0, 0, 1);   /* only the compiled kernels are used */
    if (!r) return 1;
    int bad = 0;
    {   /* encoder: [registers, anchor, text (code 0)] [noised frames (2c+2)] [clean frames (2c+1)] */
        int Ng = nreg + tpf + S, Nn = K * F * tpf, Nc = (K - 1) * F * tpf, T = Ng + Nn + Nc;
        int *code = (int *)calloc(T, 4);
        for (int n = 0; n < Nn; n++) code[Ng + n] = 2 * ((n / tpf) / F) + 2;
        for (int n = 0; n < Nc; n++) code[Ng + Nn + n] = 2 * ((n / tpf) / F) + 1;
        bad += run_case(r, "encoder (K'=2 window)", code, T, H, KVH, hd);
    }
    {   /* decoder: registers + noised frames at 2x2 upsampled resolution */
        int Nd = K * F * tpf * 4, T = nreg + Nd;
        int *code = (int *)calloc(T, 4);
        for (int n = 0; n < Nd; n++) code[nreg + n] = 2 * ((n / (tpf * 4)) / F) + 2;
        bad += run_case(r, "decoder", code, T, H, KVH, hd);
    }
    {   /* K'=3 window for a deeper clean-context pattern */
        int Kk = 3, Ng = nreg + tpf + S, Nn = Kk * F * tpf, Nc = (Kk - 1) * F * tpf, T = Ng + Nn + Nc;
        int *code = (int *)calloc(T, 4);
        for (int n = 0; n < Nn; n++) code[Ng + n] = 2 * ((n / tpf) / F) + 2;
        for (int n = 0; n < Nc; n++) code[Ng + Nn + n] = 2 * ((n / tpf) / F) + 1;
        bad += run_case(r, "encoder (K'=3 window)", code, T, H, KVH, hd);
    }
    for (int KK = 2; KK <= 4; KK++) {   /* action expert (16 heads, hd 64): cross over KV = 4.0k (K'=2) .. 9.0k (K'=4), and self */
        char nm[64]; int Gx = nreg + tpf + S;
        snprintf(nm, sizeof nm, "expert cross K'=%d", KK); bad += run_expert_case(r, nm, KK, F, 8, tpf, Gx, 16, 64, 1);
        snprintf(nm, sizeof nm, "expert self  K'=%d", KK); bad += run_expert_case(r, nm, KK, F, 8, tpf, Gx, 16, 64, 0);
    }
    printf(bad ? "FAILED\n" : "ALL OK\n");
    return bad;
}
