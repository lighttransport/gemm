/*
 * bench_cuda_fa.c
 *
 * FlashAttention-forward benchmark for sm_120 (RTX 5060 Ti).
 * Mirrors cuda/gemm/bench_cuda_gemm.c structure.
 *
 * Math: O[b,h,i,d] = sum_j softmax(Q[b,h,i,:] · K[b,h,j,:] * scale) * V[b,h,j,d]
 * scale = 1/sqrt(D). Non-causal self-attention. Q,K,V,O all [B*H, S, D] f16.
 *
 * Modes:
 *   ptx : our kernels (cuda_fa_kernels.h, NVRTC at runtime)
 *   ref : naive CUDA reference (also in cuda_fa_kernels.h) — correctness baseline
 *
 * FLOP count: 4 * B * H * S * S * D (two matmuls of S×D × D×S and S×S × S×D).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>

#include "../cuew.h"

#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

#include "cuda_fa_kernels.h"

typedef enum { DT_F16 } dtype_t;
typedef enum { MD_PTX_V1, MD_PTX_V2, MD_PTX_V3, MD_PTX_V4, MD_REF, MD_ALL } bmode_t;

typedef struct {
    const char *name;
    int B, H, S, D;
} shape_t;

static const shape_t g_shapes[] = {
    {"qwen3_512", 1, 16, 512,  128},
    {"qwen3_2k",  1, 16, 2048, 128},
    {"qwen3_4k",  1, 16, 4096, 128},
    {"dit_1k",    1, 24, 1024, 64 },
    {"sd_8k_d64", 1,  8, 8192, 64 },
};
#define N_SHAPES ((int)(sizeof(g_shapes)/sizeof(g_shapes[0])))

/* ---------- F16 conversion (host) ---------- */
static uint16_t f32_to_f16(float f) { return cu_f32_to_f16(f); }
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    uint32_t f_exp, f_mant;
    if (exp == 0 && mant == 0) { uint32_t b = sign << 31; float v; memcpy(&v, &b, 4); return v; }
    if (exp == 0) { float v = (float)mant / (float)(1<<24); return sign?-v:v; }
    if (exp == 31) { f_exp = 255; f_mant = mant << 13; }
    else { f_exp = exp + (127 - 15); f_mant = mant << 13; }
    uint32_t b = (sign << 31) | (f_exp << 23) | f_mant;
    float v; memcpy(&v, &b, 4); return v;
}

/* ---------- CPU FP32 reference ---------- */
static void cpu_fa_ref(float *O, const float *Q, const float *K, const float *V,
                       int B, int H, int S, int D, int q_rows_to_check) {
    float scale = 1.0f / sqrtf((float)D);
    for (int bh = 0; bh < B*H; bh++) {
        for (int i = 0; i < q_rows_to_check; i++) {
            const float *q = Q + ((size_t)bh*S + i) * D;
            float *scores = (float *)malloc(sizeof(float)*S);
            float mx = -1e30f;
            for (int j = 0; j < S; j++) {
                float s = 0;
                const float *k = K + ((size_t)bh*S + j) * D;
                for (int d = 0; d < D; d++) s += q[d]*k[d];
                s *= scale;
                scores[j] = s;
                if (s > mx) mx = s;
            }
            float sum = 0;
            for (int j = 0; j < S; j++) { scores[j] = expf(scores[j]-mx); sum += scores[j]; }
            float inv = 1.0f / sum;
            float *o = O + ((size_t)bh*q_rows_to_check + i) * D;
            for (int d = 0; d < D; d++) {
                float acc = 0;
                for (int j = 0; j < S; j++) acc += scores[j] * V[((size_t)bh*S + j) * D + d];
                o[d] = acc * inv;
            }
            free(scores);
        }
    }
}

typedef struct {
    int ok; double cos_sim; float max_abs_err;
} acc_result_t;

static acc_result_t validate(const float *got, const float *ref, int n) {
    acc_result_t r = {0,0,0};
    double dot=0, ng=0, nr=0;
    float me=0;
    for (int i = 0; i < n; i++) {
        float g=got[i], rr=ref[i], e=fabsf(g-rr);
        if (e>me) me=e;
        dot += (double)g*rr; ng += (double)g*g; nr += (double)rr*rr;
    }
    r.cos_sim = (ng>0 && nr>0) ? dot/(sqrt(ng)*sqrt(nr)) : 0;
    r.max_abs_err = me;
    r.ok = (r.cos_sim >= 0.999);
    return r;
}

/* ---------- Kernel cache ---------- */
typedef struct {
    CUmodule mod_v1, mod_v2, mod_v3, mod_v4, mod_ref;
    CUfunction f_v1, f_v2, f_v3, f_v4, f_ref;
    int built_v1, built_v2, built_v3, built_v4, built_ref;
} kc_t;

static int build_one(CUmodule *mod, CUfunction *fn, int *built,
                     const char *src, const char *name, CUdevice dev, int verbose) {
    if (*built) return 0;
    if (cu_compile_kernels(mod, dev, src, name, verbose, "bench_cuda_fa") < 0) return -1;
    if (cuModuleGetFunction(fn, *mod, name) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuModuleGetFunction(%s) failed\n", name);
        return -1;
    }
    *built = 1;
    return 0;
}

/* ---------- Run one mode ---------- */
typedef struct {
    int B, H, S, D;
    int warmup, iters, verify, verify_qrows, verbose;
    kc_t *kc;
    CUdevice dev;
    CUstream stream;
    CUdeviceptr d_Q, d_K, d_V, d_O;
} ctx_t;

static int run_mode(ctx_t *c, bmode_t md, float *avg_ms_out) {
    int S=c->S, D=c->D, BH=c->B*c->H;
    CUfunction fn;
    size_t smem;
    int FA_BKV = 64;
    int FA_W = 4;
    int block, gx;
    if (md == MD_PTX_V1) {
        if (build_one(&c->kc->mod_v1, &c->kc->f_v1, &c->kc->built_v1,
                      k_fa_f16_v1_src, "fa_f16_v1", c->dev, c->verbose) != 0) return -1;
        fn = c->kc->f_v1;
        smem = (size_t)2 * FA_BKV * D * sizeof(float) + (size_t)FA_BKV * sizeof(float);
        block = D; gx = S;
    } else if (md == MD_PTX_V2) {
        if (build_one(&c->kc->mod_v2, &c->kc->f_v2, &c->kc->built_v2,
                      k_fa_f16_v2_src, "fa_f16_v2", c->dev, c->verbose) != 0) return -1;
        fn = c->kc->f_v2;
        smem = (size_t)2 * FA_BKV * D * sizeof(uint16_t);  /* sK + sV in f16 */
        block = 256; FA_W = 8; gx = (S + FA_W - 1) / FA_W;
    } else if (md == MD_PTX_V3) {
        if (D != 128) { fprintf(stderr, "v3 requires D=128 (got %d)\n", D); return -1; }
        if (build_one(&c->kc->mod_v3, &c->kc->f_v3, &c->kc->built_v3,
                      k_fa_f16_v3_src, "fa_f16_v3", c->dev, c->verbose) != 0) return -1;
        fn = c->kc->f_v3;
        /* sK[BC,DP] + sVT[D,BCP], BC=32, DP=136, D=128, BCP=40 -> (32*136 + 128*40) halves */
        smem = (size_t)((32 * 136) + (128 * 40)) * sizeof(uint16_t);
        block = 128; gx = (S + 63) / 64;
    } else if (md == MD_PTX_V4) {
        if (D != 128) { fprintf(stderr, "v4 requires D=128 (got %d)\n", D); return -1; }
        if (build_one(&c->kc->mod_v4, &c->kc->f_v4, &c->kc->built_v4,
                      k_fa_f16_v4_src, "fa_f16_v4", c->dev, c->verbose) != 0) return -1;
        fn = c->kc->f_v4;
        /* sK[2][BC,DP] + sVR[2][BC,DP] halves (no sVT — ldmatrix.trans replaces transpose) */
        smem = (size_t)(2*32*136 + 2*32*136) * sizeof(uint16_t);
        block = 128; gx = (S + 63) / 64;
    } else {
        if (build_one(&c->kc->mod_ref, &c->kc->f_ref, &c->kc->built_ref,
                      k_fa_f16_ref_src, "fa_f16_ref", c->dev, c->verbose) != 0) return -1;
        fn = c->kc->f_ref;
        smem = (size_t)S * sizeof(float);
        block = D; gx = S;
    }

    /* For dynamic SMEM > 48 KiB on Blackwell, opt in. */
    if (smem > 48 * 1024) {
        cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem);
    }

    if (block < 32) block = 32;
    float scale = 1.0f / sqrtf((float)D);
    void *args[] = { &c->d_O, &c->d_Q, &c->d_K, &c->d_V, &S, &D, &scale };

    for (int i = 0; i < c->warmup; i++) {
        if (cuLaunchKernel(fn, gx, BH, 1, block, 1, 1, smem, c->stream, args, NULL) != CUDA_SUCCESS) {
            fprintf(stderr, "  launch failed\n"); return -1;
        }
    }
    cuStreamSynchronize(c->stream);

    CUevent t0, t1;
    cuEventCreate(&t0, CU_EVENT_DEFAULT);
    cuEventCreate(&t1, CU_EVENT_DEFAULT);
    cuEventRecord(t0, c->stream);
    for (int i = 0; i < c->iters; i++) {
        cuLaunchKernel(fn, gx, BH, 1, block, 1, 1, smem, c->stream, args, NULL);
    }
    cuEventRecord(t1, c->stream);
    cuEventSynchronize(t1);
    float ms;
    cuEventElapsedTime(&ms, t0, t1);
    *avg_ms_out = ms / c->iters;
    cuEventDestroy(t0);
    cuEventDestroy(t1);
    return 0;
}

/* ---------- Per-shape orchestration ---------- */
static int run_shape(bmode_t md, const shape_t *sh,
                     int warmup, int iters, int verify, int verify_qrows, int verbose,
                     kc_t *kc, CUdevice dev, CUstream stream) {
    int B=sh->B, H=sh->H, S=sh->S, D=sh->D, BH=B*H;
    if (D > 128) { fprintf(stderr, "skip %s: D=%d > 128 (v1 limit)\n", sh->name, D); return 0; }
    if (D & 31) { fprintf(stderr, "skip %s: D=%d not multiple of 32\n", sh->name, D); return 0; }

    size_t qkv_n = (size_t)BH * S * D;
    float *Qf = (float*)malloc(qkv_n*sizeof(float));
    float *Kf = (float*)malloc(qkv_n*sizeof(float));
    float *Vf = (float*)malloc(qkv_n*sizeof(float));
    if (!Qf || !Kf || !Vf) { fprintf(stderr, "OOM\n"); return -1; }
    for (size_t i = 0; i < qkv_n; i++) {
        Qf[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
        Kf[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
        Vf[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    }

    /* Quantize-then-dequantize so CPU ref sees same f16 rounding as GPU. */
    uint16_t *Qh = (uint16_t*)malloc(qkv_n*2);
    uint16_t *Kh = (uint16_t*)malloc(qkv_n*2);
    uint16_t *Vh = (uint16_t*)malloc(qkv_n*2);
    for (size_t i = 0; i < qkv_n; i++) { Qh[i]=f32_to_f16(Qf[i]); Kh[i]=f32_to_f16(Kf[i]); Vh[i]=f32_to_f16(Vf[i]); }
    if (verify) {
        for (size_t i = 0; i < qkv_n; i++) { Qf[i]=f16_to_f32(Qh[i]); Kf[i]=f16_to_f32(Kh[i]); Vf[i]=f16_to_f32(Vh[i]); }
    }

    /* CPU reference for first verify_qrows query rows of every (B*H). */
    float *Oref = NULL;
    if (verify) {
        if (verify_qrows > S) verify_qrows = S;
        Oref = (float*)malloc((size_t)BH*verify_qrows*D*sizeof(float));
        cpu_fa_ref(Oref, Qf, Kf, Vf, B, H, S, D, verify_qrows);
    }

    CUdeviceptr d_Q=0, d_K=0, d_V=0, d_O=0;
    cuMemAlloc(&d_Q, qkv_n*2);
    cuMemAlloc(&d_K, qkv_n*2);
    cuMemAlloc(&d_V, qkv_n*2);
    cuMemAlloc(&d_O, qkv_n*2);
    cuMemcpyHtoD(d_Q, Qh, qkv_n*2);
    cuMemcpyHtoD(d_K, Kh, qkv_n*2);
    cuMemcpyHtoD(d_V, Vh, qkv_n*2);
    cuMemsetD8(d_O, 0, qkv_n*2);

    ctx_t c = { B, H, S, D, warmup, iters, verify, verify_qrows, verbose,
                kc, dev, stream, d_Q, d_K, d_V, d_O };

    uint16_t *Oh = (uint16_t*)malloc(qkv_n*2);
    float *Ogot = (float*)malloc((size_t)BH*verify_qrows*D*sizeof(float));

    double flops = 4.0 * (double)BH * (double)S * (double)S * (double)D;

    bmode_t modes[5]; int nmodes=0;
    if (md == MD_PTX_V1 || md == MD_ALL) modes[nmodes++] = MD_PTX_V1;
    if (md == MD_PTX_V2 || md == MD_ALL) modes[nmodes++] = MD_PTX_V2;
    if ((md == MD_PTX_V3 || md == MD_ALL) && D == 128) modes[nmodes++] = MD_PTX_V3;
    if ((md == MD_PTX_V4 || md == MD_ALL) && D == 128) modes[nmodes++] = MD_PTX_V4;
    if (md == MD_REF    || md == MD_ALL) modes[nmodes++] = MD_REF;

    for (int mi = 0; mi < nmodes; mi++) {
        bmode_t m = modes[mi];
        cuMemsetD8(d_O, 0, qkv_n*2);
        float ms = 0;
        if (run_mode(&c, m, &ms) != 0) continue;
        double tflops = flops / (ms * 1e9);
        acc_result_t acc = {1, 1.0, 0};
        if (verify) {
            cuMemcpyDtoH(Oh, d_O, qkv_n*2);
            /* Extract first verify_qrows rows per (B*H) and convert to f32. */
            for (int bh = 0; bh < BH; bh++) {
                for (int i = 0; i < verify_qrows; i++) {
                    for (int d = 0; d < D; d++) {
                        Ogot[((size_t)bh*verify_qrows+i)*D + d] =
                            f16_to_f32(Oh[((size_t)bh*S+i)*D + d]);
                    }
                }
            }
            acc = validate(Ogot, Oref, BH*verify_qrows*D);
        }
        const char *ms_str = (m == MD_PTX_V1) ? "ptx_v1" :
                             (m == MD_PTX_V2) ? "ptx_v2" :
                             (m == MD_PTX_V3) ? "ptx_v3" :
                             (m == MD_PTX_V4) ? "ptx_v4" : "ref   ";
        const char *status = acc.ok ? "ACC_OK" : "ACC_FAIL";
        printf("dtype=f16  mode=%s  shape=%-9s  B=%d H=%2d S=%5d D=%3d  ms=%7.3f  TFLOP/s=%6.2f  %s",
               ms_str, sh->name, B, H, S, D, ms, tflops, status);
        if (verify) printf("  cos=%.5f  max_err=%.4g", acc.cos_sim, acc.max_abs_err);
        printf("\n");
    }

    cuMemFree(d_Q); cuMemFree(d_K); cuMemFree(d_V); cuMemFree(d_O);
    free(Qf); free(Kf); free(Vf); free(Qh); free(Kh); free(Vh);
    free(Oh); free(Ogot);
    if (Oref) free(Oref);
    return 0;
}

static void usage(const char *p) {
    printf("Usage: %s [options]\n", p);
    printf("  --dtype  f16              dtype (default f16)\n");
    printf("  --mode   ptx|ref|all      backend (default all)\n");
    printf("  --shape  <name>|all       shape (default all)\n");
    printf("  --batch B --heads H --seqlen S --head-dim D   ad-hoc\n");
    printf("  --iters N                 (default 50)\n");
    printf("  --warmup N                (default 5)\n");
    printf("  --verify 0|1              (default 1)\n");
    printf("  --verify-qrows N          CPU ref rows per head (default 8)\n");
    printf("  --verbose N               (default 0)\n");
    for (int i = 0; i < N_SHAPES; i++)
        printf("  shape: %-10s  B=%d H=%2d S=%5d D=%3d\n",
               g_shapes[i].name, g_shapes[i].B, g_shapes[i].H, g_shapes[i].S, g_shapes[i].D);
}

int main(int argc, char **argv) {
    bmode_t md = MD_ALL;
    const char *shape_name = "all";
    int B=0,H=0,S=0,D=0;
    int iters=50, warmup=5, verify=1, verify_qrows=8, verbose=0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dtype") && i+1<argc) i++; /* only f16 */
        else if (!strcmp(argv[i], "--mode") && i+1<argc) {
            const char *s = argv[++i];
            if (!strcmp(s,"ptx") || !strcmp(s,"v4")) md=MD_PTX_V4;
            else if (!strcmp(s,"v3")) md=MD_PTX_V3;
            else if (!strcmp(s,"v2")) md=MD_PTX_V2;
            else if (!strcmp(s,"v1")) md=MD_PTX_V1;
            else if (!strcmp(s,"ref")) md=MD_REF;
            else if (!strcmp(s,"all")) md=MD_ALL;
            else { fprintf(stderr,"unknown mode %s\n", s); return 1; }
        }
        else if (!strcmp(argv[i],"--shape")&&i+1<argc) shape_name=argv[++i];
        else if (!strcmp(argv[i],"--batch")&&i+1<argc) B=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--heads")&&i+1<argc) H=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seqlen")&&i+1<argc) S=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--head-dim")&&i+1<argc) D=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--iters")&&i+1<argc) iters=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--warmup")&&i+1<argc) warmup=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--verify")&&i+1<argc) verify=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--verify-qrows")&&i+1<argc) verify_qrows=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--verbose")&&i+1<argc) verbose=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) { fprintf(stderr,"cuewInit failed\n"); return 1; }
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr,"cuInit failed\n"); return 1; }
    CUdevice dev; CUcontext ctx;
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);
    char dn[256]; int sm_maj=0,sm_min=0,sm_count=0,clk=0;
    cuDeviceGetName(dn, sizeof(dn), dev);
    cuDeviceGetAttribute(&sm_maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&sm_min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&clk, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    printf("Device: %s sm_%d%d SMs=%d clock=%d MHz\n", dn, sm_maj, sm_min, sm_count, clk/1000);
    CUstream stream=NULL; cuStreamCreate(&stream, 0);

    kc_t kc = {0};
    srand(42);

    if (B>0 && H>0 && S>0 && D>0) {
        shape_t a = {"adhoc", B, H, S, D};
        run_shape(md, &a, warmup, iters, verify, verify_qrows, verbose, &kc, dev, stream);
    } else if (!strcmp(shape_name,"all")) {
        for (int i = 0; i < N_SHAPES; i++)
            run_shape(md, &g_shapes[i], warmup, iters, verify, verify_qrows, verbose, &kc, dev, stream);
    } else {
        const shape_t *sh=NULL;
        for (int i = 0; i < N_SHAPES; i++) if (!strcmp(g_shapes[i].name, shape_name)) { sh=&g_shapes[i]; break; }
        if (!sh) { fprintf(stderr,"unknown shape %s\n", shape_name); return 1; }
        run_shape(md, sh, warmup, iters, verify, verify_qrows, verbose, &kc, dev, stream);
    }

    if (kc.built_v1) cuModuleUnload(kc.mod_v1);
    if (kc.built_v2) cuModuleUnload(kc.mod_v2);
    if (kc.built_v3) cuModuleUnload(kc.mod_v3);
    if (kc.built_v4) cuModuleUnload(kc.mod_v4);
    if (kc.built_ref) cuModuleUnload(kc.mod_ref);
    cuStreamDestroy(stream);
    cuCtxDestroy(ctx);
    return 0;
}
