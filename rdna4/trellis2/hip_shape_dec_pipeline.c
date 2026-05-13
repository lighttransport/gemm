/*
 * hip_shape_dec_pipeline.c - HIP shape_dec forward pipeline (RDNA4)
 *
 * Factored out of test_hip_tex_dec.c (Milestone C). Holds:
 *   - kernel function lookup (K)
 *   - hipBLASLt + Triton bridge state
 *   - per-weight device caches (F32/F16/BF16)
 *   - persistent C2S transient scratch
 *   - run_convnext / run_c2s / synthesize_unguided_subdiv
 *   - the per-stage forward loop (input_layer + 4× {convnext + c2s} +
 *     output_layer)
 *
 * State is file-scope so the existing per-process design is preserved
 * bit-for-bit (single shape_dec instance per process).
 *
 * IMPLEMENTATION DEFINES (placed here exactly once per binary):
 *   T2_SHAPE_DEC_IMPLEMENTATION
 *   T2_FDG_MESH_IMPLEMENTATION
 *   SPARSE3D_IMPLEMENTATION
 *   TRITON_SPCONV_BRIDGE_IMPL
 *   TRITON_KLIN_BRIDGE_IMPL
 *
 * SAFETENSORS_IMPLEMENTATION + GGML_DEQUANT_IMPLEMENTATION are NOT defined
 * here; they are owned by the binary's own .c (test_hip_tex_dec.c,
 * test_hip_trellis2.c) so we don't double-define.
 *
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"
#define T2_FDG_MESH_IMPLEMENTATION
#include "../../common/trellis2_fdg_mesh.h"

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "hip_tex_dec_kernels.h"
#include "../llm/mm_blaslt_bridge.h"
#define TRITON_BRIDGE_USE_ROCEW
#define TRITON_SPCONV_BRIDGE_IMPL
#include "triton_aot/triton_spconv_bridge.h"
#define TRITON_KLIN_BRIDGE_IMPL
#include "triton_aot/klin/triton_klin_bridge.h"

#include "hip_shape_dec_pipeline.h"

/* ======================================================================== *
 * Globals (preserved verbatim from test_hip_tex_dec.c)
 * ======================================================================== */

static int g_use_triton_klin = 0;
static hipStream_t g_stream = NULL;
static int g_use_blaslt = 0;
static void *g_d_act_bf16 = NULL;
static size_t g_act_bf16_floats = 0;
#define BF16_CACHE_N 1024
static const float *g_bf16_keys[BF16_CACHE_N];
static size_t g_bf16_lens[BF16_CACHE_N];
static void *g_bf16_devs[BF16_CACHE_N];
static int g_bf16_count = 0;

static int g_use_triton = 0;
static void *g_d_in_f16  = NULL;
static void *g_d_out_f16 = NULL;
static size_t g_in_f16_bytes = 0, g_out_f16_bytes = 0;
#define F16_CACHE_N 1024
static const float *g_f16_keys[F16_CACHE_N];
static size_t g_f16_lens[F16_CACHE_N];
static void *g_f16_devs[F16_CACHE_N];
static int g_f16_count = 0;

#define F32_CACHE_N 2048
static const void *g_f32_keys[F32_CACHE_N];
static size_t g_f32_lens[F32_CACHE_N];
static void *g_f32_devs[F32_CACHE_N];
static int g_f32_count = 0;

static void *get_f32_dev(const void *host_ptr, size_t n_bytes) {
    if (!host_ptr) return NULL;
    for (int i = 0; i < g_f32_count; i++) {
        if (g_f32_keys[i] == host_ptr && g_f32_lens[i] == n_bytes)
            return g_f32_devs[i];
    }
    if (g_f32_count >= F32_CACHE_N) {
        fprintf(stderr, "T2-TEX: f32 cache full\n"); return NULL;
    }
    void *d = NULL;
    if (hipMalloc(&d, n_bytes) != hipSuccess) return NULL;
    hipMemcpy(d, host_ptr, n_bytes, hipMemcpyHostToDevice);
    int slot = g_f32_count++;
    g_f32_keys[slot] = host_ptr;
    g_f32_lens[slot] = n_bytes;
    g_f32_devs[slot] = d;
    return d;
}

#define SYNC() hipDeviceSynchronize()

/* ======================================================================== *
 * Kernel function table
 * ======================================================================== */

typedef struct { hipFunction_t ins, conv, conv_tiled, conv_nmap, conv_nmap_tiled, conv_nmap_bf16, conv_nmap_bf16_x2, conv_nmap_bf16_x4, conv_nmap_bf16_x8, conv_nmap_bf16_x8_db, gather27, gather27_v2, ln, silu, silu_bf16, gelu, add, lin, lin_naive, gather, resrep, pack_bf16, pack_f16, unpack_f16, splitk_reduce, dense_x8_db, dense_bf16in_x8_db; } K;

static K g_K = {0};

static void hash_build(K *k, void *keys, void *vals, int cap_mask, void *coords, int N) {
    void *a[] = {&keys, &vals, &cap_mask, &coords, &N};
    hipModuleLaunchKernel(k->ins, (N+255)/256, 1, 1, 256, 1, 1, 0, g_stream, a, NULL);
}
static void kspconv(K *k, void *out, void *in, void *co, void *w, void *b,
                    void *keys, void *vals, int cap_mask, int N, int inC, int outC) {
    void *a[] = {&out, &in, &co, &w, &b, &keys, &vals, &cap_mask, &inC, &outC};
    if ((outC % 64 == 0) && (inC % 32 == 0)) {
        hipModuleLaunchKernel(k->conv_tiled, N, outC / 64, 1, 64, 1, 1, 0, g_stream, a, NULL);
    } else {
        hipModuleLaunchKernel(k->conv, N, 1, 1, 256, 1, 1, 0, g_stream, a, NULL);
    }
}

static int g_use_wmma_spconv = 1;
static int g_use_blaslt_spconv = 1;
static void *get_bf16_weight(const float *host_w_f32, size_t n_elems);
static int kspconv_nmap_triton(K *k, void *out_f32, void *in_f32, void *nmap,
                               const float *host_w, void *bias_f32,
                               int N, int inC, int outC);
static void kspconv_nmap_blaslt(K *k, void *out_f32, void *feats_f32, void *nmap,
                                const float *host_w, void *bias_f32,
                                int N, int inC, int outC);

static int g_prof_cn = -1;
static double g_prof_ms[8] = {0};
static long long g_prof_n[8] = {0};
static hipEvent_t g_prof_sp[3] = {0};

static void kspconv_nmap_h(K *k, void *out, void *in, void *nmap, const float *host_w,
                           void *w, void *b,
                           void *co, void *keys, void *vals, int cap_mask,
                           int N, int inC, int outC) {
    static int s_path_dbg = -1;
    if (s_path_dbg < 0) {
        const char *e = getenv("T2_TEX_PATH_DBG");
        s_path_dbg = (e && atoi(e)) ? 1 : 0;
    }
    #define PATH(tag) do { if (s_path_dbg) fprintf(stderr, "    spc N=%d inC=%d outC=%d path=%s nmap=%p\n", N, inC, outC, tag, nmap); } while (0)
    int triton_nmax = 0;
    {
        const char *e = getenv("T2_TRITON_NMAX");
        if (e) triton_nmax = atoi(e);
    }
    int triton_skip = 0;
    if (inC == 64 && outC == 64) {
        const char *e = getenv("T2_TRITON_C64");
        if (!e || !atoi(e)) triton_skip = 1;
    }
    if (!triton_skip && nmap && g_use_triton && host_w && k->pack_f16 &&
        (triton_nmax <= 0 || N <= triton_nmax) &&
        t2_triton_has_shape(N, inC, outC)) {
        if (kspconv_nmap_triton(k, out, in, nmap, host_w, b, N, inC, outC) == 0) { PATH("triton"); return; }
    }
    if (nmap && g_use_blaslt_spconv && g_use_blaslt && host_w && k->gather27 &&
        (inC % 16 == 0) && (outC % 16 == 0) && inC <= 512 && N >= 512) {
        PATH("blaslt");
        kspconv_nmap_blaslt(k, out, in, nmap, host_w, b, N, inC, outC);
        return;
    }
    if (nmap) {
        if (g_use_wmma_spconv && k->conv_nmap_bf16_x8 &&
            (inC % 16 == 0) && (outC % 64 == 0) && (N >= 32)) {
            PATH("wmma_x8");
            void *a[] = {&out, &in, &nmap, &w, &b, &N, &inC, &outC};
            int gx = (N + 31) / 32, gy = (outC + 63) / 64;
            hipFunction_t f = k->conv_nmap_bf16_x8;
            const char *e = getenv("T2_TEX_WMMA_DB");
            if (k->conv_nmap_bf16_x8_db && (e == NULL || atoi(e))) f = k->conv_nmap_bf16_x8_db;
            hipModuleLaunchKernel(f, gx, gy, 1, 256, 1, 1, 0, g_stream, a, NULL);
            return;
        }
        if (g_use_wmma_spconv && k->conv_nmap_bf16_x4 &&
            (inC % 16 == 0) && (outC % 32 == 0) && (N >= 32)) {
            PATH("wmma_x4");
            void *a[] = {&out, &in, &nmap, &w, &b, &N, &inC, &outC};
            int gx = (N + 31) / 32, gy = (outC + 31) / 32;
            hipModuleLaunchKernel(k->conv_nmap_bf16_x4, gx, gy, 1, 128, 1, 1, 0, g_stream, a, NULL);
            return;
        }
        if (g_use_wmma_spconv && k->conv_nmap_bf16_x2 &&
            (inC % 16 == 0) && (outC % 32 == 0)) {
            PATH("wmma_x2");
            void *a[] = {&out, &in, &nmap, &w, &b, &N, &inC, &outC};
            int gx = (N + 15) / 16, gy = (outC + 31) / 32;
            hipModuleLaunchKernel(k->conv_nmap_bf16_x2, gx, gy, 1, 64, 1, 1, 0, g_stream, a, NULL);
            return;
        }
        if (g_use_wmma_spconv && k->conv_nmap_bf16 &&
            (inC % 16 == 0) && (outC % 16 == 0)) {
            PATH("wmma_x1");
            void *a[] = {&out, &in, &nmap, &w, &b, &N, &inC, &outC};
            int gx = (N + 15) / 16, gy = (outC + 15) / 16;
            hipModuleLaunchKernel(k->conv_nmap_bf16, gx, gy, 1, 32, 1, 1, 0, g_stream, a, NULL);
            return;
        }
        void *a[] = {&out, &in, &nmap, &w, &b, &inC, &outC};
        if ((outC % 64 == 0) && (inC % 32 == 0)) {
            PATH("nmap_tiled_f32");
            hipModuleLaunchKernel(k->conv_nmap_tiled, N, outC / 64, 1, 64, 1, 1, 0, g_stream, a, NULL);
        } else {
            PATH("nmap_scalar_f32");
            hipModuleLaunchKernel(k->conv_nmap, N, 1, 1, 256, 1, 1, 0, g_stream, a, NULL);
        }
        } else {
        PATH("hash_kspconv");
        kspconv(k, out, in, co, w, b, keys, vals, cap_mask, N, inC, outC);
    }
    #undef PATH
}

static void kspconv_nmap_blaslt(K *k, void *out_f32, void *feats_f32, void *nmap,
                                const float *host_w, void *bias_f32,
                                int N, int inC, int outC) {
    void *d_w_bf16 = get_bf16_weight(host_w, (size_t)outC * 27 * inC);
    if (!d_w_bf16) return;
    int K_total = 27 * inC;
    long long max_chunk = (long long)g_act_bf16_floats / K_total;
    int CAP = (max_chunk > 16384) ? 16384 : (int)max_chunk;
    if (CAP < 1) CAP = 1;
    int nchunks = (N + CAP - 1) / CAP;
    int M_CHUNK = (N + nchunks - 1) / nchunks;
    for (int m0 = 0; m0 < N; m0 += M_CHUNK) {
        int M = (N - m0 < M_CHUNK) ? (N - m0) : M_CHUNK;
        void *act = g_d_act_bf16;
        void *args[] = {&act, &feats_f32, &nmap, &m0, &M, &inC};
        if (g_prof_cn) hipEventRecord(g_prof_sp[0], g_stream);
        if (k->gather27_v2 && (inC % 2 == 0)) {
            int MB = 4;
            int gx = (M + MB - 1) / MB;
            hipModuleLaunchKernel(k->gather27_v2, gx, 1, 1, 256, 1, 1, 0, g_stream, args, NULL);
        } else {
            hipModuleLaunchKernel(k->gather27, M, 27, 1, 256, 1, 1, 0, g_stream, args, NULL);
        }
        if (g_prof_cn) hipEventRecord(g_prof_sp[1], g_stream);
        float *out_chunk = (float *)out_f32 + (size_t)m0 * outC;
        static int s_use_bf16in = -1;
        if (s_use_bf16in < 0) {
            const char *e = getenv("T2_TEX_BF16IN_GEMM");
            s_use_bf16in = (e && atoi(e)) ? 1 : 0;
        }
        if (s_use_bf16in && k->dense_bf16in_x8_db &&
            (outC % 16 == 0) && (K_total % 16 == 0)) {
            void *gemm_args[] = {&out_chunk, &act, &d_w_bf16, &bias_f32,
                                 &M, &K_total, &outC};
            int gx = (M + 31) / 32, gy = (outC + 63) / 64;
            hipModuleLaunchKernel(k->dense_bf16in_x8_db, gx, gy, 1,
                                  256, 1, 1, 0, g_stream, gemm_args, NULL);
        } else {
            mm_blaslt_run_bf16_bias(out_chunk, d_w_bf16, act, bias_f32,
                                     M, outC, K_total, g_stream);
        }
        if (g_prof_cn) {
            hipEventRecord(g_prof_sp[2], g_stream);
            hipEventSynchronize(g_prof_sp[2]);
            float ms=0;
            hipEventElapsedTime(&ms, g_prof_sp[0], g_prof_sp[1]); g_prof_ms[6]+=ms; g_prof_n[6]++;
            hipEventElapsedTime(&ms, g_prof_sp[1], g_prof_sp[2]); g_prof_ms[7]+=ms; g_prof_n[7]++;
        }
    }
}

static void kln(K *k, void *o, void *i, void *w, void *b, int N, int C, int hw, int hb) {
    float eps = 1e-6f;
    void *a[] = {&o, &i, &w, &b, &C, &eps, &hw, &hb};
    int bs = C >= 256 ? 256 : (C >= 128 ? 128 : 64);
    hipModuleLaunchKernel(k->ln, N, 1, 1, bs, 1, 1, 0, g_stream, a, NULL);
}
static void ksilu(K *k, void *o, void *i, int n) {
    void *a[] = {&o, &i, &n};
    hipModuleLaunchKernel(k->silu, (n+255)/256, 1, 1, 256, 1, 1, 0, g_stream, a, NULL);
}
static void ksilu_bf16(K *k, void *o, void *i, int n) {
    void *a[] = {&o, &i, &n};
    int blocks = ((n + 3) / 4 + 255) / 256;
    hipModuleLaunchKernel(k->silu_bf16, blocks, 1, 1, 256, 1, 1, 0, g_stream, a, NULL);
}
static void kadd(K *k, void *x, void *y, int n) {
    void *a[] = {&x, &y, &n};
    hipModuleLaunchKernel(k->add, (n+255)/256, 1, 1, 256, 1, 1, 0, g_stream, a, NULL);
}
static void klin(K *k, void *o, void *i, void *w, void *b, int N, int inC, int outC) {
    int gx = (outC+15)/16, gy = (N+15)/16;
    void *a[] = {&o, &i, &w, &b, &N, &inC, &outC};
    static int use_naive = -1;
    if (use_naive < 0) use_naive = (getenv("T2_LIN_NAIVE") && atoi(getenv("T2_LIN_NAIVE"))) ? 1 : 0;
    hipFunction_t fn = use_naive ? k->lin_naive : k->lin;
    hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, g_stream, a, NULL);
}

static void *get_bf16_weight(const float *host_w_f32, size_t n_elems) {
    for (int i = 0; i < g_bf16_count; i++) {
        if (g_bf16_keys[i] == host_w_f32 && g_bf16_lens[i] == n_elems)
            return g_bf16_devs[i];
    }
    if (g_bf16_count >= BF16_CACHE_N) {
        fprintf(stderr, "T2-TEX: bf16 cache full\n"); return NULL;
    }
    unsigned short *h_bf16 = (unsigned short *)malloc(n_elems * 2);
    for (size_t i = 0; i < n_elems; i++) {
        unsigned int u; memcpy(&u, &host_w_f32[i], 4);
        u = u + ((u >> 16) & 1u) + 0x7FFFu;
        h_bf16[i] = (unsigned short)(u >> 16);
    }
    void *d_bf16 = NULL;
    if (hipMalloc(&d_bf16, n_elems * 2) != hipSuccess) {
        free(h_bf16); return NULL;
    }
    hipMemcpy(d_bf16, h_bf16, n_elems * 2, hipMemcpyHostToDevice);
    free(h_bf16);
    int slot = g_bf16_count++;
    g_bf16_keys[slot] = host_w_f32;
    g_bf16_lens[slot] = n_elems;
    g_bf16_devs[slot] = d_bf16;
    return d_bf16;
}

static unsigned short f32_to_f16_rne(float f) {
    unsigned int u; memcpy(&u, &f, 4);
    unsigned int sign = (u >> 16) & 0x8000u;
    unsigned int exp  = (u >> 23) & 0xFF;
    unsigned int mant = u & 0x7FFFFFu;
    if (exp == 0xFF) {
        return (unsigned short)(sign | 0x7C00u | (mant ? (mant >> 13) | 0x200u : 0));
    }
    int e = (int)exp - 127 + 15;
    if (e >= 31) return (unsigned short)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (unsigned short)sign;
        mant |= 0x800000u;
        unsigned int shift = 14 - e;
        unsigned int hm = mant >> shift;
        unsigned int rb = (mant >> (shift - 1)) & 1u;
        unsigned int sticky = (mant & ((1u << (shift - 1)) - 1u)) ? 1u : 0u;
        unsigned int round = (rb && (sticky || (hm & 1u))) ? 1u : 0u;
        return (unsigned short)(sign | (hm + round));
    }
    unsigned int hm = mant >> 13;
    unsigned int rb = (mant >> 12) & 1u;
    unsigned int sticky = (mant & 0xFFFu) ? 1u : 0u;
    unsigned int round = (rb && (sticky || (hm & 1u))) ? 1u : 0u;
    unsigned int packed = sign | ((unsigned int)e << 10) | hm;
    packed += round;
    return (unsigned short)packed;
}

static void *get_f16_weight_k(K *k, const float *host_w_f32, size_t n_elems) {
    for (int i = 0; i < g_f16_count; i++) {
        if (g_f16_keys[i] == host_w_f32 && g_f16_lens[i] == n_elems)
            return g_f16_devs[i];
    }
    if (g_f16_count >= F16_CACHE_N) {
        fprintf(stderr, "T2-TEX: f16 cache full\n"); return NULL;
    }
    void *d_f16 = NULL;
    if (hipMalloc(&d_f16, n_elems * 2) != hipSuccess) return NULL;
    if (k && k->pack_f16) {
        void *d_f32 = get_f32_dev(host_w_f32, n_elems * sizeof(float));
        if (!d_f32) { hipFree(d_f16); return NULL; }
        int n = (int)n_elems;
        void *args[] = { &d_f16, &d_f32, &n };
        hipModuleLaunchKernel(k->pack_f16, (n + 255)/256, 1, 1, 256, 1, 1, 0, g_stream, args, NULL);
    } else {
        unsigned short *h = (unsigned short *)malloc(n_elems * 2);
        for (size_t i = 0; i < n_elems; i++) h[i] = f32_to_f16_rne(host_w_f32[i]);
        hipMemcpy(d_f16, h, n_elems * 2, hipMemcpyHostToDevice);
        free(h);
    }
    int slot = g_f16_count++;
    g_f16_keys[slot] = host_w_f32;
    g_f16_lens[slot] = n_elems;
    g_f16_devs[slot] = d_f16;
    return d_f16;
}

static int kspconv_nmap_triton(K *k, void *out_f32, void *in_f32, void *nmap,
                               const float *host_w, void *bias_f32,
                               int N, int inC, int outC) {
    if (!g_use_triton || !host_w || !k->pack_f16 || !k->unpack_f16) return -1;
    size_t need_in  = (size_t)N * inC  * 2;
    size_t need_out = (size_t)N * outC * 2;
    if (need_in > g_in_f16_bytes) {
        if (g_d_in_f16) hipFree(g_d_in_f16);
        if (hipMalloc(&g_d_in_f16, need_in) != hipSuccess) { g_d_in_f16=NULL; g_in_f16_bytes=0; return -1; }
        g_in_f16_bytes = need_in;
    }
    if (need_out > g_out_f16_bytes) {
        if (g_d_out_f16) hipFree(g_d_out_f16);
        if (hipMalloc(&g_d_out_f16, need_out) != hipSuccess) { g_d_out_f16=NULL; g_out_f16_bytes=0; return -1; }
        g_out_f16_bytes = need_out;
    }
    void *d_w_f16 = get_f16_weight_k(k, host_w, (size_t)outC * 27 * inC);
    if (!d_w_f16) return -1;
    void *d_b_f16 = NULL;
    {
        const float *key = (const float *)bias_f32;
        for (int i = 0; i < g_f16_count; i++) {
            if (g_f16_keys[i] == key && g_f16_lens[i] == (size_t)outC) { d_b_f16 = g_f16_devs[i]; break; }
        }
        if (!d_b_f16) {
            if (g_f16_count >= F16_CACHE_N) return -1;
            if (hipMalloc(&d_b_f16, (size_t)outC * 2) != hipSuccess) return -1;
            void *args[] = { &d_b_f16, &bias_f32, &outC };
            hipModuleLaunchKernel(k->pack_f16, (outC + 255)/256, 1, 1, 256, 1, 1, 0, g_stream, args, NULL);
            int slot = g_f16_count++;
            g_f16_keys[slot] = key;
            g_f16_lens[slot] = (size_t)outC;
            g_f16_devs[slot] = d_b_f16;
        }
    }
    int n_in = N * inC;
    {
        void *args[] = { &g_d_in_f16, &in_f32, &n_in };
        hipModuleLaunchKernel(k->pack_f16, (n_in + 255)/256, 1, 1, 256, 1, 1, 0, g_stream, args, NULL);
    }
    int rc = t2_triton_spconv(N, inC, outC, g_d_in_f16, d_w_f16, d_b_f16,
                              nmap, g_d_out_f16, g_stream);
    if (rc != 0) return -1;
    int n_out = N * outC;
    {
        void *args[] = { &out_f32, &g_d_out_f16, &n_out };
        hipModuleLaunchKernel(k->unpack_f16, (n_out + 255)/256, 1, 1, 256, 1, 1, 0, g_stream, args, NULL);
    }
    return 0;
}

static int klin_dense_wmma(K *k, void *o_f32, void *x_f32, void *d_w_f32,
                           void *bias_f32, int N, int inC, int outC) {
    int gx = (N + 31) / 32;
    int gy = (outC + 63) / 64;
    void *args[] = { &o_f32, &x_f32, &d_w_f32, &bias_f32, &N, &inC, &outC };
    if (hipModuleLaunchKernel(k->dense_x8_db, gx, gy, 1, 256, 1, 1, 0, g_stream, args, NULL) != hipSuccess)
        return -1;
    return 0;
}

static int g_prof_klin = -1;
static double g_klin_pack_ms = 0, g_klin_blaslt_ms = 0;
static long long g_klin_pack_n = 0, g_klin_blaslt_n = 0;
static hipEvent_t g_klin_ev[3] = {0};
static void klin_prof_init(void) {
    if (g_prof_klin != -1) return;
    g_prof_klin = (getenv("T2_PROF_KLIN") && atoi(getenv("T2_PROF_KLIN"))) ? 1 : 0;
    if (g_prof_klin) for (int i=0;i<3;i++) hipEventCreate(&g_klin_ev[i]);
}

static void klin_bl(K *k, void *o_f32, void *x_f32, const float *host_w_f32,
                    void *dev_w_f32_fallback, void *bias_f32,
                    int N, int inC, int outC) {
    int eligible = (g_use_blaslt && N >= 8 && (inC % 16) == 0 && (outC % 16) == 0);
    if (!eligible || !host_w_f32) {
        klin(k, o_f32, x_f32, dev_w_f32_fallback, bias_f32, N, inC, outC);
        return;
    }
    {
        static int s_dense_wmma = -1;
        if (s_dense_wmma < 0) {
            const char *e = getenv("T2_TEX_DENSE_WMMA");
            s_dense_wmma = (e && atoi(e)) ? 1 : 0;
        }
        static int s_dense_cap = -1;
        if (s_dense_cap < 0) {
            const char *e = getenv("T2_TEX_DENSE_WMMA_NMAX");
            s_dense_cap = e ? atoi(e) : 4096;
        }
        if (s_dense_wmma && dev_w_f32_fallback && bias_f32 &&
            (inC % 16) == 0 && (outC % 16) == 0 && N > 0 &&
            (s_dense_cap == 0 || N <= s_dense_cap)) {
            if (klin_dense_wmma(k, o_f32, x_f32, dev_w_f32_fallback, bias_f32, N, inC, outC) == 0)
                return;
        }
    }
    void *d_w_bf16 = get_bf16_weight(host_w_f32, (size_t)outC * inC);
    if (!d_w_bf16) {
        klin(k, o_f32, x_f32, dev_w_f32_fallback, bias_f32, N, inC, outC);
        return;
    }
    if (g_use_triton_klin && t2_klin_has_shape(inC, outC)) {
        const int CAP_KL_T = 16384;
        int nchunks_t = (N + CAP_KL_T - 1) / CAP_KL_T;
        int M_CHUNK_T = (N + nchunks_t - 1) / nchunks_t;
        int ok = 1;
        for (int m0 = 0; m0 < N && ok; m0 += M_CHUNK_T) {
            int M = (N - m0 < M_CHUNK_T) ? (N - m0) : M_CHUNK_T;
            int n_act = M * inC;
            void *dst = g_d_act_bf16;
            float *src_chunk = (float *)x_f32 + (size_t)m0 * inC;
            void *src = src_chunk;
            void *args0[] = { &dst, &src, &n_act };
            int n4 = (n_act + 3) / 4;
            hipModuleLaunchKernel(k->pack_bf16, (n4 + 255) / 256, 1, 1, 256, 1, 1, 0, g_stream, args0, NULL);
            float *dst_chunk = (float *)o_f32 + (size_t)m0 * outC;
            int rc = t2_klin_run(M, inC, outC, g_d_act_bf16, d_w_bf16, bias_f32, dst_chunk, g_stream);
            if (rc != 0) { ok = 0; break; }
        }
        if (ok) return;
    }
    klin_prof_init();
    static int s_cap_kl = -1;
    if (s_cap_kl < 0) {
        const char *e = getenv("T2_TEX_KLIN_CAP");
        s_cap_kl = (e && atoi(e) > 0) ? atoi(e) : 16384;
    }
    const int CAP_KL = s_cap_kl;
    int nchunks_kl = (N + CAP_KL - 1) / CAP_KL;
    int M_CHUNK = (N + nchunks_kl - 1) / nchunks_kl;
    for (int m0 = 0; m0 < N; m0 += M_CHUNK) {
        int M = (N - m0 < M_CHUNK) ? (N - m0) : M_CHUNK;
        int n_act = M * inC;
        void *dst = g_d_act_bf16;
        float *src_chunk = (float *)x_f32 + (size_t)m0 * inC;
        void *src = src_chunk;
        void *args0[] = { &dst, &src, &n_act };
        int n4 = (n_act + 3) / 4;
        if (g_prof_klin) hipEventRecord(g_klin_ev[0], g_stream);
        hipModuleLaunchKernel(k->pack_bf16, (n4 + 255) / 256, 1, 1, 256, 1, 1, 0, g_stream, args0, NULL);
        if (g_prof_klin) hipEventRecord(g_klin_ev[1], g_stream);
        float *dst_chunk = (float *)o_f32 + (size_t)m0 * outC;
        if (mm_blaslt_run_bf16_bias(dst_chunk, d_w_bf16, g_d_act_bf16, bias_f32,
                                    M, outC, inC, g_stream) != 0) {
            fprintf(stderr, "T2-TEX: blaslt failed (M=%d N=%d K=%d), falling back to F32\n",
                    M, outC, inC);
            klin(k, o_f32, x_f32, dev_w_f32_fallback, bias_f32, N, inC, outC);
            return;
        }
        if (g_prof_klin) {
            hipEventRecord(g_klin_ev[2], g_stream);
            hipEventSynchronize(g_klin_ev[2]);
            float ms;
            hipEventElapsedTime(&ms, g_klin_ev[0], g_klin_ev[1]); g_klin_pack_ms += ms; g_klin_pack_n++;
            hipEventElapsedTime(&ms, g_klin_ev[1], g_klin_ev[2]); g_klin_blaslt_ms += ms; g_klin_blaslt_n++;
        }
    }
}

static void klin_bl_silu(K *k, void *o_f32, void *x_f32, const float *host_w_f32,
                         void *dev_w_f32_fallback, void *bias_f32,
                         int N, int inC, int outC) {
    int eligible = (g_use_blaslt && N >= 8 && (inC % 16) == 0 && (outC % 16) == 0
                    && host_w_f32 && bias_f32);
    if (eligible && g_use_triton_klin && t2_klin_has_shape_silu(inC, outC)) {
        void *d_w_bf16 = get_bf16_weight(host_w_f32, (size_t)outC * inC);
        if (d_w_bf16) {
            const int CAP_KL_T = 16384;
            int nchunks_t = (N + CAP_KL_T - 1) / CAP_KL_T;
            int M_CHUNK_T = (N + nchunks_t - 1) / nchunks_t;
            int ok = 1;
            for (int m0 = 0; m0 < N && ok; m0 += M_CHUNK_T) {
                int M = (N - m0 < M_CHUNK_T) ? (N - m0) : M_CHUNK_T;
                int n_act = M * inC;
                void *dst = g_d_act_bf16;
                float *src_chunk = (float *)x_f32 + (size_t)m0 * inC;
                void *src = src_chunk;
                void *args0[] = { &dst, &src, &n_act };
                int n4 = (n_act + 3) / 4;
                hipModuleLaunchKernel(k->pack_bf16, (n4 + 255) / 256, 1, 1, 256, 1, 1, 0, g_stream, args0, NULL);
                float *dst_chunk = (float *)o_f32 + (size_t)m0 * outC;
                if (t2_klin_run_silu(M, inC, outC, g_d_act_bf16, d_w_bf16, bias_f32, dst_chunk, g_stream) != 0) {
                    ok = 0; break;
                }
            }
            if (ok) return;
        }
    }
    klin_bl(k, o_f32, x_f32, host_w_f32, dev_w_f32_fallback, bias_f32, N, inC, outC);
    ksilu(k, o_f32, o_f32, N * outC);
}

static int klin_bl_bf16d(K *k, void *o_bf16, void *x_f32, const float *host_w_f32,
                         void *bias_f32, int N, int inC, int outC) {
    int eligible = (g_use_blaslt && N >= 8 && (inC % 16) == 0 && (outC % 16) == 0 && bias_f32 && host_w_f32);
    if (!eligible) return -1;
    void *d_w_bf16 = get_bf16_weight(host_w_f32, (size_t)outC * inC);
    if (!d_w_bf16) return -1;
    const int CAP_KL = 16384;
    int nchunks = (N + CAP_KL - 1) / CAP_KL;
    int M_CHUNK = (N + nchunks - 1) / nchunks;
    for (int m0 = 0; m0 < N; m0 += M_CHUNK) {
        int M = (N - m0 < M_CHUNK) ? (N - m0) : M_CHUNK;
        int n_act = M * inC;
        void *dst = g_d_act_bf16;
        float *src_chunk = (float *)x_f32 + (size_t)m0 * inC;
        void *src = src_chunk;
        void *args0[] = { &dst, &src, &n_act };
        int n4 = (n_act + 3) / 4;
        hipModuleLaunchKernel(k->pack_bf16, (n4 + 255) / 256, 1, 1, 256, 1, 1, 0, g_stream, args0, NULL);
        unsigned short *dst_chunk = (unsigned short *)o_bf16 + (size_t)m0 * outC;
        if (mm_blaslt_run_bf16_bias_bf16d(dst_chunk, d_w_bf16, g_d_act_bf16, bias_f32,
                                          M, outC, inC, g_stream) != 0) {
            return -1;
        }
    }
    return 0;
}

static int klin_bl_bf16in(K *k, void *o_f32, void *x_bf16, const float *host_w_f32,
                          void *bias_f32, int N, int inC, int outC) {
    (void)k;
    int eligible = (g_use_blaslt && N >= 8 && (inC % 16) == 0 && (outC % 16) == 0 && host_w_f32);
    if (!eligible) return -1;
    void *d_w_bf16 = get_bf16_weight(host_w_f32, (size_t)outC * inC);
    if (!d_w_bf16) return -1;
    const int CAP_KL = 16384;
    int nchunks = (N + CAP_KL - 1) / CAP_KL;
    int M_CHUNK = (N + nchunks - 1) / nchunks;
    for (int m0 = 0; m0 < N; m0 += M_CHUNK) {
        int M = (N - m0 < M_CHUNK) ? (N - m0) : M_CHUNK;
        unsigned short *src_chunk = (unsigned short *)x_bf16 + (size_t)m0 * inC;
        float *dst_chunk = (float *)o_f32 + (size_t)m0 * outC;
        if (mm_blaslt_run_bf16_bias(dst_chunk, d_w_bf16, src_chunk, bias_f32,
                                    M, outC, inC, g_stream) != 0) {
            return -1;
        }
    }
    return 0;
}

static void kgather(K *k, void *hf, void *xf, void *hc, void *xc,
                    void *idx, void *si, int Nf, int Co, int Ci8) {
    int mx = Co > Ci8 ? Co : Ci8; int gy = (mx+255)/256;
    void *a[] = {&hf, &xf, &hc, &xc, &idx, &si, &Co, &Ci8};
    hipModuleLaunchKernel(k->gather, Nf, gy, 1, 256, 1, 1, 0, g_stream, a, NULL);
}
static void kresrep(K *k, void *h, void *x, int N, int Co, int Ci8) {
    void *a[] = {&h, &x, &N, &Co, &Ci8};
    hipModuleLaunchKernel(k->resrep, N, 1, 1, 256, 1, 1, 0, g_stream, a, NULL);
}

static void prof_init(void) {
    if (g_prof_cn != -1) return;
    g_prof_cn = (getenv("T2_PROF_CN") && atoi(getenv("T2_PROF_CN"))) ? 1 : 0;
    if (g_prof_cn) {
        for (int i=0;i<3;i++) hipEventCreate(&g_prof_sp[i]);
    }
}

static int g_prof_c2s = -1;
static double g_c2s_ms[10] = {0};
static long long g_c2s_n[10] = {0};
static hipEvent_t g_c2s_ev[10] = {0};
static double g_c2s_alloc_ms = 0;
static int g_c2s_alloc_calls = 0;
static void c2s_prof_init(void) {
    if (g_prof_c2s != -1) return;
    g_prof_c2s = (getenv("T2_PROF_C2S") && atoi(getenv("T2_PROF_C2S"))) ? 1 : 0;
    if (g_prof_c2s) for (int i=0;i<10;i++) hipEventCreate(&g_c2s_ev[i]);
}

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void run_convnext(K *k, const t2sd_convnext *blk, int C, int N,
    void *d_feats, void *d_coords, void *d_keys, void *d_vals, int cap_mask,
    void *d_tmp, void *d_mlp, void *d_nmap) {
    static int s_block_time = -1;
    if (s_block_time < 0) {
        const char *e = getenv("T2_TEX_BLOCK_TIME");
        s_block_time = (e && atoi(e)) ? 1 : 0;
    }
    double t_get0 = s_block_time ? now_ms() : 0;
    void *dwc = get_f32_dev(blk->conv_w, (size_t)C * 27 * C * sizeof(float));
    void *dwb = get_f32_dev(blk->conv_b, (size_t)C * sizeof(float));
    void *dnw = get_f32_dev(blk->norm_w, (size_t)C * sizeof(float));
    void *dnb = get_f32_dev(blk->norm_b, (size_t)C * sizeof(float));
    void *dm0 = get_f32_dev(blk->mlp0_w, (size_t)4*C*C*sizeof(float));
    void *dm0b= get_f32_dev(blk->mlp0_b, (size_t)4*C*sizeof(float));
    void *dm2 = get_f32_dev(blk->mlp2_w, (size_t)C*4*C*sizeof(float));
    void *dm2b= get_f32_dev(blk->mlp2_b, (size_t)C*sizeof(float));
    double t_get1 = 0;
    if (s_block_time) { if (g_stream) hipStreamSynchronize(g_stream); t_get1 = now_ms(); }
    prof_init();
    kspconv_nmap_h(k, d_tmp, d_feats, d_nmap, blk->conv_w, dwc, dwb, d_coords, d_keys, d_vals, cap_mask, N, C, C);
    double t_spc = 0;
    if (s_block_time) { if (g_stream) hipStreamSynchronize(g_stream); t_spc = now_ms(); }
    kln(k, d_tmp, d_tmp, dnw, dnb, N, C, 1, 1);
    double t_ln = 0;
    if (s_block_time) { if (g_stream) hipStreamSynchronize(g_stream); t_ln = now_ms(); }
    static int s_klin_bf16 = -1;
    if (s_klin_bf16 < 0) {
        const char *e = getenv("T2_TEX_KLIN_BF16");
        s_klin_bf16 = (e && atoi(e)) ? 1 : 0;
    }
    int used_bf16 = 0;
    if (s_klin_bf16 && N >= 20000) {
        if (klin_bl_bf16d(k, d_mlp, d_tmp, blk->mlp0_w, dm0b, N, C, 4*C) == 0) {
            ksilu_bf16(k, d_mlp, d_mlp, N * 4 * C);
            if (klin_bl_bf16in(k, d_tmp, d_mlp, blk->mlp2_w, dm2b, N, 4*C, C) == 0) {
                used_bf16 = 1;
            }
        }
    }
    if (!used_bf16) {
        static int s_dbg_cn = -1;
        if (s_dbg_cn < 0) { const char *e=getenv("T2_TEX_DBG_CN"); s_dbg_cn = (e && atoi(e)) ? 1 : 0; }
        if (s_dbg_cn) {
            fprintf(stderr, "  cn mlp: N=%d C=%d d_tmp=%p d_mlp=%p mlp0_w=%p dm0=%p dm0b=%p mlp2_w=%p dm2=%p dm2b=%p\n",
                    N, C, d_tmp, d_mlp, (void*)blk->mlp0_w, dm0, dm0b, (void*)blk->mlp2_w, dm2, dm2b);
        }
        klin_bl_silu(k, d_mlp, d_tmp, blk->mlp0_w, dm0, dm0b, N, C, 4*C);
        klin_bl(k, d_tmp, d_mlp, blk->mlp2_w, dm2, dm2b, N, 4*C, C);
    }
    double t_mlp = 0;
    if (s_block_time) { if (g_stream) hipStreamSynchronize(g_stream); t_mlp = now_ms(); }
    kadd(k, d_feats, d_tmp, N * C);
    if (s_block_time) {
        if (g_stream) hipStreamSynchronize(g_stream);
        double t_end = now_ms();
        fprintf(stderr, "    cn N=%d C=%d: get=%.1f spc=%.1f ln=%.1f mlp=%.1f(bf16=%d) add=%.1f tot=%.1f\n",
                N, C, t_get1 - t_get0, t_spc - t_get1, t_ln - t_spc, t_mlp - t_ln, used_bf16, t_end - t_mlp, t_end - t_get0);
    }
}

typedef struct { void *feats, *coords, *keys, *vals; int N, C, cap_mask; } DevSparse;

typedef struct {
    void *dn, *de, *dhf, *dxf, *dhn;
    void *didx, *dsi;
    void *dout;
    void *fk, *fv;
    size_t dn_b, de_b, dhf_b, dxf_b, dhn_b;
    size_t didx_b, dsi_b;
    size_t dout_b, fk_b, fv_b;
} C2SScratch;
static C2SScratch g_c2s = {0};
static void c2s_grow(void **p, size_t *cap, size_t need) {
    if (need <= *cap) return;
    if (*p) hipFree(*p);
    hipMalloc(p, need);
    *cap = need;
}

static DevSparse run_c2s(K *k, const t2sd_c2s *blk, int Nc,
    void *d_feats, void *d_coords, void *d_keys, void *d_vals, int cap_mask,
    void *d_idx_pre, void *d_si_pre, void *d_xc_pre, int Nf,
    void *d_nmap_coarse, void *d_nmap_fine) {
    int Ci = blk->C_in, Co = blk->C_out, Ci8 = Ci/8, Cexp = Co*8;
    void *dn1w = get_f32_dev(blk->norm1_w, (size_t)Ci*sizeof(float));
    void *dn1b = get_f32_dev(blk->norm1_b, (size_t)Ci*sizeof(float));
    void *dc1w = get_f32_dev(blk->conv1_w, (size_t)Cexp*27*Ci*sizeof(float));
    void *dc1b = get_f32_dev(blk->conv1_b, (size_t)Cexp*sizeof(float));
    void *dc2w = get_f32_dev(blk->conv2_w, (size_t)Co*27*Co*sizeof(float));
    void *dc2b = get_f32_dev(blk->conv2_b, (size_t)Co*sizeof(float));
    void *dout=NULL;
    c2s_prof_init();
    double a0 = g_prof_c2s ? now_ms() : 0;
    c2s_grow(&g_c2s.dn,  &g_c2s.dn_b,  (size_t)Nc*Ci*sizeof(float));
    c2s_grow(&g_c2s.de,  &g_c2s.de_b,  (size_t)Nc*Cexp*sizeof(float));
    c2s_grow(&g_c2s.dhf, &g_c2s.dhf_b, (size_t)Nf*Co*sizeof(float));
    c2s_grow(&g_c2s.dxf, &g_c2s.dxf_b, (size_t)Nf*Ci8*sizeof(float));
    c2s_grow(&g_c2s.dhn, &g_c2s.dhn_b, (size_t)Nf*Co*sizeof(float));
    void *dn=g_c2s.dn, *de=g_c2s.de, *dhf=g_c2s.dhf, *dxf=g_c2s.dxf, *dhn=g_c2s.dhn;
    c2s_grow(&g_c2s.dout, &g_c2s.dout_b, (size_t)Nf*Co*sizeof(float));
    dout = g_c2s.dout;
    if (g_prof_c2s) { g_c2s_alloc_ms += now_ms()-a0; g_c2s_alloc_calls++; }
    kln(k, dn, d_feats, dn1w, dn1b, Nc, Ci, 1, 1);
    ksilu(k, dn, dn, Nc*Ci);
    kspconv_nmap_h(k, de, dn, d_nmap_coarse, blk->conv1_w, dc1w, dc1b, d_coords, d_keys, d_vals, cap_mask, Nc, Ci, Cexp);
    void *didx = d_idx_pre; void *dsi = d_si_pre;
    void *dxc  = d_xc_pre;
    kgather(k, dhf, dxf, de, d_feats, didx, dsi, Nf, Co, Ci8);
    kln(k, dhn, dhf, NULL, NULL, Nf, Co, 0, 0);
    ksilu(k, dhn, dhn, Nf*Co);
    int cap = 1; while (cap < Nf*2) cap <<= 1; int cm = cap - 1;
    c2s_grow(&g_c2s.fk, &g_c2s.fk_b, (size_t)cap*sizeof(uint64_t));
    c2s_grow(&g_c2s.fv, &g_c2s.fv_b, (size_t)cap*sizeof(int32_t));
    void *fk = g_c2s.fk, *fv = g_c2s.fv;
    hipMemsetAsync(fk, 0,    (size_t)cap*sizeof(uint64_t), g_stream);
    hipMemsetAsync(fv, 0xff, (size_t)cap*sizeof(int32_t),  g_stream);
    hash_build(k, fk, fv, cm, dxc, Nf);
    kspconv_nmap_h(k, dout, dhn, d_nmap_fine, blk->conv2_w, dc2w, dc2b, dxc, fk, fv, cm, Nf, Co, Co);
    kresrep(k, dout, dxf, Nf, Co, Ci8);
    DevSparse ds = { dout, dxc, fk, fv, Nf, Co, cm };
    return ds;
}

/* When capture_h_* are non-NULL, retain host copies (ownership transferred
 * to caller); otherwise host arrays are freed after upload. */
static void synthesize_unguided_subdiv(
    const t2_shape_dec *dec, int stage_idx, void *d_feats, void *d_coords, int Nc,
    void **out_d_idx, void **out_d_si, void **out_d_xc, int *out_Nf,
    int64_t **capture_h_idx, int64_t **capture_h_si, int32_t **capture_h_xc)
{
    int Ci = dec->c2s[stage_idx].C_in;
    float *h_feats = (float *)malloc((size_t)Nc * Ci * sizeof(float));
    int32_t *h_coords = (int32_t *)malloc((size_t)Nc * 4 * sizeof(int32_t));
    if (g_stream) hipStreamSynchronize(g_stream);
    hipMemcpy(h_feats, d_feats, (size_t)Nc * Ci * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_coords, d_coords, (size_t)Nc * 4 * sizeof(int32_t), hipMemcpyDeviceToHost);

    int64_t *idx = NULL, *si = NULL; int32_t *xc = NULL; int Nf = 0;
    int rc = t2_shape_dec_unguided_synth_host(
        dec, stage_idx, h_feats, h_coords, Nc, &idx, &si, &xc, &Nf);
    free(h_feats); free(h_coords);
    if (rc != 0) {
        fprintf(stderr, "synthesize_unguided_subdiv: host helper failed rc=%d\n", rc);
        *out_d_idx = NULL; *out_d_si = NULL; *out_d_xc = NULL; *out_Nf = 0;
        if (capture_h_idx) *capture_h_idx = NULL;
        if (capture_h_si)  *capture_h_si  = NULL;
        if (capture_h_xc)  *capture_h_xc  = NULL;
        return;
    }

    *out_d_idx = hip_upload_raw(idx, (size_t)Nf * sizeof(int64_t));
    *out_d_si  = hip_upload_raw(si,  (size_t)Nf * sizeof(int64_t));
    *out_d_xc  = hip_upload_raw(xc,  (size_t)Nf * 4 * sizeof(int32_t));
    *out_Nf = Nf;
    if (capture_h_idx) { *capture_h_idx = idx; } else { free(idx); }
    if (capture_h_si)  { *capture_h_si  = si;  } else { free(si);  }
    if (capture_h_xc)  { *capture_h_xc  = xc;  } else { free(xc);  }
}

/* ======================================================================== *
 * Public ctx + forward
 * ======================================================================== */

struct hip_shape_dec_ctx {
    hipModule_t module;
    hipStream_t stream;
    const t2_shape_dec *dec;
    int verbose;
    int initialized;
    /* Optional capture of per-stage subdiv arrays (host copies) so a
     * downstream decoder (e.g. tex_dec) can be guided by this run. */
    int capture_enabled;
    hip_shape_dec_cache pending_cache;
};

static int g_ctx_init_done = 0;  /* file-scope globals init guard */

hip_shape_dec_ctx *hip_shape_dec_ctx_create(hipModule_t module,
                                            hipStream_t stream,
                                            const t2_shape_dec *dec,
                                            int verbose)
{
    if (!module || !dec) return NULL;
    hip_shape_dec_ctx *ctx = (hip_shape_dec_ctx *)calloc(1, sizeof(*ctx));
    ctx->module = module;
    ctx->stream = stream;
    ctx->dec = dec;
    ctx->verbose = verbose;

    /* Bind global stream — every kernel launch + bridge call routes here. */
    g_stream = stream;

    /* Look up kernel functions (idempotent across multiple ctx creates in a
     * single process — only runs once because file-scope state persists). */
    if (!g_ctx_init_done) {
        K *k = &g_K;
        hipModuleGetFunction(&k->ins, module, "hash_insert_kernel");
        hipModuleGetFunction(&k->conv, module, "sparse_conv3d_f32");
        hipModuleGetFunction(&k->conv_tiled, module, "sparse_conv3d_tiled_f32");
        hipModuleGetFunction(&k->conv_nmap, module, "sparse_conv3d_nmap_f32");
        hipModuleGetFunction(&k->conv_nmap_tiled, module, "sparse_conv3d_nmap_tiled_f32");
        hipModuleGetFunction(&k->conv_nmap_bf16, module, "sparse_conv3d_nmap_tiled_bf16");
        hipModuleGetFunction(&k->conv_nmap_bf16_x2, module, "sparse_conv3d_nmap_tiled_bf16_x2");
        hipModuleGetFunction(&k->conv_nmap_bf16_x4, module, "sparse_conv3d_nmap_tiled_bf16_x4");
        hipModuleGetFunction(&k->conv_nmap_bf16_x8, module, "sparse_conv3d_nmap_tiled_bf16_x8");
        hipModuleGetFunction(&k->conv_nmap_bf16_x8_db, module, "sparse_conv3d_nmap_tiled_bf16_x8_db");
        hipModuleGetFunction(&k->gather27, module, "t2_gather27_pack_bf16");
        hipModuleGetFunction(&k->gather27_v2, module, "t2_gather27_pack_bf16_v2");
        hipModuleGetFunction(&k->ln, module, "t2_layernorm_f32");
        hipModuleGetFunction(&k->silu, module, "t2_silu_f32");
        hipModuleGetFunction(&k->silu_bf16, module, "t2_silu_bf16");
        hipModuleGetFunction(&k->dense_x8_db, module, "t2_dense_gemm_bf16_x8_db");
        hipModuleGetFunction(&k->dense_bf16in_x8_db, module, "t2_dense_gemm_bf16in_x8_db");
        hipModuleGetFunction(&k->gelu, module, "t2_gelu_f32");
        hipModuleGetFunction(&k->add, module, "t2_add_f32");
        hipModuleGetFunction(&k->lin, module, "t2_linear_f32");
        hipModuleGetFunction(&k->lin_naive, module, "t2_linear_f32_naive");
        hipModuleGetFunction(&k->gather, module, "t2_c2s_gather_f32");
        hipModuleGetFunction(&k->resrep, module, "t2_residual_repeat_f32");
        hipModuleGetFunction(&k->pack_bf16, module, "t2_pack_bf16_from_f32");
        hipModuleGetFunction(&k->pack_f16, module, "t2_pack_f16_from_f32");
        hipModuleGetFunction(&k->unpack_f16, module, "t2_unpack_f32_from_f16");
        hipModuleGetFunction(&k->splitk_reduce, module, "t2_splitk_reduce_to_f16");

        /* Triton AOT spconv bridge (default ON; T2_TEX_TRITON=0 to disable). */
        {
            const char *e = getenv("T2_TEX_TRITON");
            if (!e || atoi(e)) {
                const char *kd = getenv("T2_TEX_TRITON_KERNELS");
                if (!kd) kd = "triton_aot/kernels";
                if (t2_triton_init(kd) == 0) {
                    g_use_triton = 1;
                    t2_triton_set_reduce_kernel(k->splitk_reduce);
                    const char *pd = getenv("T2_TEX_PREP_CACHE_DIR");
                    if (pd && pd[0]) t2_triton_set_prep_cache_dir(pd);
                    fprintf(stderr, "T2-TEX: Triton AOT spconv bridge enabled (kernels=%s)\n", kd);
                } else {
                    fprintf(stderr, "T2-TEX: t2_triton_init failed; bridge disabled\n");
                }
            }
        }

        /* Triton AOT klin bridge. */
        {
            const char *e = getenv("T2_TEX_KLIN_TRITON");
            if (!e || atoi(e)) {
                const char *kd = getenv("T2_TEX_KLIN_TRITON_KERNELS");
                if (!kd) kd = "triton_aot/klin/kernels";
                if (t2_klin_init(kd) == 0) {
                    g_use_triton_klin = 1;
                    fprintf(stderr, "T2-TEX: Triton AOT klin bridge enabled (kernels=%s)\n", kd);
                } else {
                    fprintf(stderr, "T2-TEX: t2_klin_init failed; bridge disabled\n");
                }
            }
        }

        {
            const char *e = getenv("T2_TEX_WMMA_SPCONV");
            if (e && strcmp(e, "0") == 0) g_use_wmma_spconv = 0;
            fprintf(stderr, "T2-TEX: WMMA spconv %s\n", g_use_wmma_spconv ? "enabled" : "disabled");
        }
        {
            /* Disabling JUST the blaslt-gather27 spconv path (keeps the
             * BF16 GEMM bridge alive for everything else). Useful for
             * isolating whether the blaslt or WMMA x8 path is faster
             * per-stage; stage 1 ConvNeXt at C=512 currently routes to
             * blaslt and dominates tex_dec wall (~1.1 s of 3.3 s). */
            const char *e = getenv("T2_TEX_BLASLT_SPCONV");
            if (e && strcmp(e, "0") == 0) g_use_blaslt_spconv = 0;
            fprintf(stderr, "T2-TEX: blaslt spconv %s\n", g_use_blaslt_spconv ? "enabled" : "disabled");
        }

        /* hipBLASLt init (default ON; T2_TEX_BLASLT=0 to disable). */
        {
            const char *env = getenv("T2_TEX_BLASLT");
            int want = (!env || strcmp(env, "0") != 0);
            if (want) {
                if (!getenv("HIPBLASLT_PRELOAD_KERNELS"))
                    setenv("HIPBLASLT_PRELOAD_KERNELS", "1", 0);
                if (mm_blaslt_init() == 0) {
                    size_t scratch_floats = 64ULL * 1024 * 1024;
                    if (hipMalloc(&g_d_act_bf16, scratch_floats * 2) == hipSuccess) {
                        g_use_blaslt = 1;
                        g_act_bf16_floats = scratch_floats;
                        fprintf(stderr, "T2-TEX: hipBLASLt BF16 GEMM enabled (act scratch %zu MB)\n",
                                scratch_floats * 2 / (1024 * 1024));
                    } else {
                        fprintf(stderr, "T2-TEX: act scratch alloc failed; disabling blaslt\n");
                        mm_blaslt_destroy();
                    }
                } else {
                    fprintf(stderr, "T2-TEX: mm_blaslt_init failed; using F32 scalar Linear\n");
                }
            }
        }

        g_ctx_init_done = 1;
    }

    ctx->initialized = 1;
    return ctx;
}

void hip_shape_dec_ctx_free(hip_shape_dec_ctx *ctx) {
    /* File-scope state (kernel handles, weight caches, scratch) is intentionally
     * leaked at process end — matches the original test's behaviour. The ctx
     * struct itself is what we own. */
    if (!ctx) return;
    /* Drop any unclaimed captured cache. */
    hip_shape_dec_cache_free(&ctx->pending_cache);
    free(ctx);
}

void hip_shape_dec_set_capture(hip_shape_dec_ctx *ctx, int enable) {
    if (!ctx) return;
    ctx->capture_enabled = enable ? 1 : 0;
    if (!enable) hip_shape_dec_cache_free(&ctx->pending_cache);
}

int hip_shape_dec_take_cache(hip_shape_dec_ctx *ctx, hip_shape_dec_cache *out) {
    if (!ctx || !out) return -1;
    memset(out, 0, sizeof(*out));
    int any = 0;
    out->n_stages = ctx->pending_cache.n_stages;
    for (int s = 0; s < 8; s++) {
        out->gi[s]  = ctx->pending_cache.gi[s];
        out->gs[s]  = ctx->pending_cache.gs[s];
        out->gxc[s] = ctx->pending_cache.gxc[s];
        out->gN[s]  = ctx->pending_cache.gN[s];
        if (out->gN[s] > 0) any = 1;
        ctx->pending_cache.gi[s]  = NULL;
        ctx->pending_cache.gs[s]  = NULL;
        ctx->pending_cache.gxc[s] = NULL;
        ctx->pending_cache.gN[s]  = 0;
    }
    ctx->pending_cache.n_stages = 0;
    return any ? 0 : -1;
}

void hip_shape_dec_cache_free(hip_shape_dec_cache *c) {
    if (!c) return;
    for (int s = 0; s < 8; s++) {
        if (c->gi[s])  { free(c->gi[s]);  c->gi[s]  = NULL; }
        if (c->gs[s])  { free(c->gs[s]);  c->gs[s]  = NULL; }
        if (c->gxc[s]) { free(c->gxc[s]); c->gxc[s] = NULL; }
        if (c->nmap_cn[s]) { free(c->nmap_cn[s]); c->nmap_cn[s] = NULL; }
        if (c->nmap_pc[s]) { free(c->nmap_pc[s]); c->nmap_pc[s] = NULL; }
        c->gN[s] = 0; c->nmap_cn_N[s] = 0; c->nmap_pc_N[s] = 0;
    }
    c->n_stages = 0;
}

int hip_shape_dec_forward_ex(hip_shape_dec_ctx *ctx,
                             const float *slat_feats,
                             const int32_t *coords,
                             int N, int slat_C,
                             const hip_shape_dec_cache *cache,
                             float **out_d_feats,
                             int32_t **out_d_coords,
                             int *out_Nf)
{
    if (!ctx || !ctx->initialized) return -1;
    g_stream = ctx->stream;
    const t2_shape_dec *dec = ctx->dec;
    K *k = &g_K;

    int C0 = dec->channels[0];
    void *d_slat = hip_upload_raw(slat_feats, (size_t)N*slat_C*sizeof(float));
    void *d_flw = get_f32_dev(dec->from_latent_w, (size_t)C0*slat_C*sizeof(float));
    void *d_flb = get_f32_dev(dec->from_latent_b, (size_t)C0*sizeof(float));
    void *d_feats = NULL;
    hipError_t _e_df = hipMalloc(&d_feats, (size_t)N*C0*sizeof(float));
    if (_e_df != hipSuccess || !d_feats) {
        fprintf(stderr, "T2-TEX: hipMalloc d_feats(%zu) failed: %d\n",
                (size_t)N*C0*sizeof(float), (int)_e_df);
        return -1;
    }
    if (ctx->verbose)
        fprintf(stderr, "T2-TEX: forward N=%d C0=%d d_slat=%p d_flw=%p d_flb=%p d_feats=%p\n",
                N, C0, d_slat, d_flw, d_flb, d_feats);
    klin_bl(k, d_feats, d_slat, dec->from_latent_w, d_flw, d_flb, N, slat_C, C0);

    void *d_coords = hip_upload_raw(coords, (size_t)N*4*sizeof(int32_t));
    int cap = 1; while (cap < N*2) cap <<= 1; int cap_mask = cap - 1;
    void *d_keys=NULL, *d_vals=NULL;
    if (hipMalloc(&d_keys, (size_t)cap*sizeof(uint64_t)) != hipSuccess) {
        fprintf(stderr, "T2-TEX: hipMalloc d_keys failed\n"); return -1;
    }
    if (hipMalloc(&d_vals, (size_t)cap*sizeof(int32_t)) != hipSuccess) {
        fprintf(stderr, "T2-TEX: hipMalloc d_vals failed\n"); return -1;
    }
    hipMemset(d_keys, 0, (size_t)cap*sizeof(uint64_t));
    hipMemset(d_vals, 0xff, (size_t)cap*sizeof(int32_t));
    hash_build(k, d_keys, d_vals, cap_mask, d_coords, N);

    /* est_Nf for scratch sizing. */
    int est_Nf[T2SD_MAX_STAGES] = {0};
    int gN_cache[T2SD_MAX_STAGES] = {0};
    if (cache) {
        for (int s = 0; s < dec->n_stages && s < cache->n_stages; s++) gN_cache[s] = cache->gN[s];
    }
    for (int s = 0; s < dec->n_stages && s < T2SD_MAX_STAGES; s++) {
        int prev_n = (s == 0) ? N : (gN_cache[s-1] > 0 ? gN_cache[s-1] : est_Nf[s-1]);
        if (gN_cache[s] > 0) est_Nf[s] = gN_cache[s];
        else if (dec->c2s[s].to_subdiv_w) est_Nf[s] = prev_n * 5;
        else est_Nf[s] = prev_n * 8;
    }
    size_t max_tmp_floats = 0, max_mlp_floats = 0;
    for (int s = 0; s < dec->n_stages; s++) {
        if (dec->n_convnext[s] <= 0) continue;
        int nstage_i = (s == 0) ? N : (gN_cache[s-1] > 0 ? gN_cache[s-1] : est_Nf[s-1]);
        size_t nstage = (size_t)nstage_i;
        size_t cstage = (size_t)dec->channels[s];
        size_t tmp_f = nstage * cstage;
        size_t mlp_f = nstage * 4 * cstage;
        if (tmp_f > max_tmp_floats) max_tmp_floats = tmp_f;
        if (mlp_f > max_mlp_floats) max_mlp_floats = mlp_f;
    }
    void *d_tmp=NULL, *d_mlp=NULL;
    hipError_t _e_tmp = hipMalloc(&d_tmp, max_tmp_floats * sizeof(float));
    hipError_t _e_mlp = hipMalloc(&d_mlp, max_mlp_floats * sizeof(float));
    if (_e_tmp != hipSuccess || !d_tmp || _e_mlp != hipSuccess || !d_mlp) {
        fprintf(stderr, "T2-TEX: hipMalloc scratch failed (tmp=%zuMB rc=%d, mlp=%zuMB rc=%d)\n",
                max_tmp_floats * sizeof(float) / (1024*1024), (int)_e_tmp,
                max_mlp_floats * sizeof(float) / (1024*1024), (int)_e_mlp);
        return -1;
    }
    if (ctx->verbose)
        fprintf(stderr, "T2-TEX: scratch d_tmp=%p (%zuMB) d_mlp=%p (%zuMB)\n",
                d_tmp, max_tmp_floats * sizeof(float) / (1024*1024),
                d_mlp, max_mlp_floats * sizeof(float) / (1024*1024));

    /* Optional cache uploads. */
    void *d_gi[T2SD_MAX_STAGES]={0}, *d_gs[T2SD_MAX_STAGES]={0}, *d_gxc[T2SD_MAX_STAGES]={0};
    void *d_nmap_cn[T2SD_MAX_STAGES]={0}, *d_nmap_pc[T2SD_MAX_STAGES]={0};
    if (cache) {
        for (int s = 0; s < dec->n_stages && s < cache->n_stages; s++) {
            if (cache->gi[s])  d_gi[s]  = hip_upload_raw(cache->gi[s],  (size_t)cache->gN[s] * sizeof(int64_t));
            if (cache->gs[s])  d_gs[s]  = hip_upload_raw(cache->gs[s],  (size_t)cache->gN[s] * sizeof(int64_t));
            if (cache->gxc[s]) d_gxc[s] = hip_upload_raw(cache->gxc[s], (size_t)cache->gN[s] * 4 * sizeof(int32_t));
            if (cache->nmap_cn[s]) d_nmap_cn[s] = hip_upload_raw(cache->nmap_cn[s], (size_t)cache->nmap_cn_N[s]*27*sizeof(uint32_t));
            if (cache->nmap_pc[s]) d_nmap_pc[s] = hip_upload_raw(cache->nmap_pc[s], (size_t)cache->nmap_pc_N[s]*27*sizeof(uint32_t));
        }
    }

    /* Pre-size all C2S persistent scratch to worst-case. */
    {
        size_t max_dn = 0, max_de = 0, max_dhf = 0, max_dxf = 0, max_dhn = 0;
        size_t max_dout = 0, max_didx = 0, max_dsi = 0;
        size_t max_fk = 0, max_fv = 0;
        for (int s = 0; s < dec->n_stages; s++) {
            int Nc = (s == 0) ? N : est_Nf[s - 1];
            int Nf = est_Nf[s];
            int Ci = dec->c2s[s].C_in, Co = dec->c2s[s].C_out;
            int Ci8 = Ci/8, Cexp = Co*8;
            size_t b;
            b = (size_t)Nc * Ci   * sizeof(float); if (b > max_dn)   max_dn = b;
            b = (size_t)Nc * Cexp * sizeof(float); if (b > max_de)   max_de = b;
            b = (size_t)Nf * Co   * sizeof(float); if (b > max_dhf)  max_dhf = b;
            b = (size_t)Nf * Ci8  * sizeof(float); if (b > max_dxf)  max_dxf = b;
            b = (size_t)Nf * Co   * sizeof(float); if (b > max_dhn)  max_dhn = b;
            b = (size_t)Nf * Co   * sizeof(float); if (b > max_dout) max_dout = b;
            b = (size_t)Nf * sizeof(int64_t);      if (b > max_didx) max_didx = b;
            if (b > max_dsi) max_dsi = b;
            int cap2 = 1; while (cap2 < Nf*2) cap2 <<= 1;
            b = (size_t)cap2 * sizeof(uint64_t);   if (b > max_fk)   max_fk = b;
            b = (size_t)cap2 * sizeof(int32_t);    if (b > max_fv)   max_fv = b;
        }
        c2s_grow(&g_c2s.dn,   &g_c2s.dn_b,   max_dn);
        c2s_grow(&g_c2s.de,   &g_c2s.de_b,   max_de);
        c2s_grow(&g_c2s.dhf,  &g_c2s.dhf_b,  max_dhf);
        c2s_grow(&g_c2s.dxf,  &g_c2s.dxf_b,  max_dxf);
        c2s_grow(&g_c2s.dhn,  &g_c2s.dhn_b,  max_dhn);
        c2s_grow(&g_c2s.dout, &g_c2s.dout_b, max_dout);
        c2s_grow(&g_c2s.didx, &g_c2s.didx_b, max_didx);
        c2s_grow(&g_c2s.dsi,  &g_c2s.dsi_b,  max_dsi);
        c2s_grow(&g_c2s.fk,   &g_c2s.fk_b,   max_fk);
        c2s_grow(&g_c2s.fv,   &g_c2s.fv_b,   max_fv);
    }

    /* Persistent output buffer. */
    void *d_out_persist = NULL;
    size_t d_out_persist_b = 0;
    if (dec->n_stages > 0 && dec->out_channels > 0) {
        size_t fin_n = (size_t)est_Nf[dec->n_stages - 1];
        if (fin_n > 0) {
            d_out_persist_b = fin_n * dec->out_channels * sizeof(float);
            hipMalloc(&d_out_persist, d_out_persist_b);
        }
    }

    static int s_stage_time = -1;
    if (s_stage_time < 0) {
        const char *e = getenv("T2_TEX_STAGE_TIME");
        s_stage_time = (e && atoi(e)) ? 1 : 0;
    }
    int cur_N = N;
    /* Forward through all stages. */
    for (int s = 0; s < dec->n_stages; s++) {
        int nc = dec->n_convnext[s]; int ch = dec->channels[s];
        if (ctx->verbose)
            fprintf(stderr, "stage %d: %d ConvNeXt(C=%d), N=%d\n", s, nc, ch, cur_N);
        if (s_stage_time && g_stream) hipStreamSynchronize(g_stream);
        double cn_t0 = s_stage_time ? now_ms() : 0;
        for (int b = 0; b < nc; b++) {
            run_convnext(k, &dec->convnext[s][b], ch, cur_N,
                d_feats, d_coords, d_keys, d_vals, cap_mask, d_tmp, d_mlp, d_nmap_cn[s]);
        }
        if (s_stage_time) {
            if (g_stream) hipStreamSynchronize(g_stream);
            fprintf(stderr, "  stage %d ConvNeXt(%d blks, C=%d, N=%d): %.1f ms\n",
                    s, nc, ch, cur_N, now_ms() - cn_t0);
        }
        if (dec->c2s[s].conv1_w) {
            void *d_idx_use = d_gi[s];
            void *d_si_use  = d_gs[s];
            void *d_xc_use  = d_gxc[s];
            int   Nf_use    = gN_cache[s];
            void *d_idx_owned = NULL, *d_si_owned = NULL, *d_xc_owned = NULL;
            void *d_feats_owned = NULL;
            if (Nf_use == 0 && dec->c2s[s].to_subdiv_w) {
                int64_t **cap_idx = NULL, **cap_si = NULL;
                int32_t **cap_xc = NULL;
                if (ctx->capture_enabled && s < 8) {
                    /* Drop any prior captured stage (re-running forward). */
                    if (ctx->pending_cache.gi[s])  { free(ctx->pending_cache.gi[s]);  ctx->pending_cache.gi[s]  = NULL; }
                    if (ctx->pending_cache.gs[s])  { free(ctx->pending_cache.gs[s]);  ctx->pending_cache.gs[s]  = NULL; }
                    if (ctx->pending_cache.gxc[s]) { free(ctx->pending_cache.gxc[s]); ctx->pending_cache.gxc[s] = NULL; }
                    ctx->pending_cache.gN[s] = 0;
                    cap_idx = &ctx->pending_cache.gi[s];
                    cap_si  = &ctx->pending_cache.gs[s];
                    cap_xc  = &ctx->pending_cache.gxc[s];
                }
                synthesize_unguided_subdiv(dec, s, d_feats, d_coords, cur_N,
                    &d_idx_owned, &d_si_owned, &d_xc_owned, &Nf_use,
                    cap_idx, cap_si, cap_xc);
                if (ctx->capture_enabled && s < 8) {
                    ctx->pending_cache.gN[s] = Nf_use;
                    if (s + 1 > ctx->pending_cache.n_stages)
                        ctx->pending_cache.n_stages = s + 1;
                }
                d_idx_use = d_idx_owned;
                d_si_use  = d_si_owned;
                d_xc_use  = d_xc_owned;
                if (ctx->verbose)
                    fprintf(stderr, "  unguided c2s synth: Nc=%d -> Nf=%d\n", cur_N, Nf_use);
                int Ci = dec->c2s[s].C_in, Co = dec->c2s[s].C_out;
                size_t need_dout = (size_t)Nf_use * Co * sizeof(float);
                if (need_dout > g_c2s.dout_b) {
                    size_t feats_b = (size_t)cur_N * Ci * sizeof(float);
                    hipMalloc(&d_feats_owned, feats_b);
                    hipMemcpyAsync(d_feats_owned, d_feats, feats_b,
                                   hipMemcpyDeviceToDevice, g_stream);
                    if (g_stream) hipStreamSynchronize(g_stream);
                    if (g_c2s.dout) hipFree(g_c2s.dout);
                    hipMalloc(&g_c2s.dout, need_dout);
                    g_c2s.dout_b = need_dout;
                    d_feats = d_feats_owned;
                }
            }
            if (ctx->verbose)
                fprintf(stderr, "  c2s %d->%d, N_fine=%d\n",
                    dec->c2s[s].C_in, dec->c2s[s].C_out, Nf_use);
            void *d_fine = d_nmap_pc[s] ? d_nmap_pc[s]
                                        : (s+1 < T2SD_MAX_STAGES ? d_nmap_cn[s+1] : NULL);
            if (s_stage_time && g_stream) hipStreamSynchronize(g_stream);
            double c2s_t0 = s_stage_time ? now_ms() : 0;
            DevSparse ds = run_c2s(k, &dec->c2s[s], cur_N,
                d_feats, d_coords, d_keys, d_vals, cap_mask,
                d_idx_use, d_si_use, d_xc_use, Nf_use,
                d_nmap_cn[s], d_fine);
            if (s_stage_time) {
                if (g_stream) hipStreamSynchronize(g_stream);
                fprintf(stderr, "  stage %d C2S(%d->%d, Nc=%d->Nf=%d): %.1f ms\n",
                        s, dec->c2s[s].C_in, dec->c2s[s].C_out, cur_N, Nf_use,
                        now_ms() - c2s_t0);
            }
            if (d_idx_owned) hipFree(d_idx_owned);
            if (d_si_owned)  hipFree(d_si_owned);
            if (d_feats_owned) hipFree(d_feats_owned);
            d_feats = ds.feats; d_coords = ds.coords;
            d_keys = ds.keys; d_vals = ds.vals; cap_mask = ds.cap_mask; cur_N = ds.N;
        }
    }

    /* output_layer (LN + linear). */
    int Cf = dec->c2s[dec->n_stages-1].C_out;
    int out_ch = dec->out_channels;
    size_t need_out = (size_t)cur_N * out_ch * sizeof(float);
    if (need_out > d_out_persist_b) {
        if (d_out_persist) hipFree(d_out_persist);
        hipMalloc(&d_out_persist, need_out);
        d_out_persist_b = need_out;
    }
    void *d_out = d_out_persist;
    kln(k, d_feats, d_feats, NULL, NULL, cur_N, Cf, 0, 0);
    void *d_ow = get_f32_dev(dec->output_w, (size_t)out_ch*Cf*sizeof(float));
    void *d_ob = get_f32_dev(dec->output_b, (size_t)out_ch*sizeof(float));
    klin(k, d_out, d_feats, d_ow, d_ob, cur_N, Cf, out_ch);

    /* Persist d_coords too — promote to caller. d_coords by this point is
     * either the original or ds.coords, both heap-owned somewhere in our
     * scratch chain. We need a fresh buffer the caller can hipFree. */
    void *d_coords_out = NULL;
    hipMalloc(&d_coords_out, (size_t)cur_N * 4 * sizeof(int32_t));
    hipMemcpyAsync(d_coords_out, d_coords, (size_t)cur_N * 4 * sizeof(int32_t),
                   hipMemcpyDeviceToDevice, g_stream);

    /* Ditto for d_out: callers expect to own this; transfer ownership of
     * d_out_persist directly. */
    *out_d_feats = (float *)d_out;
    *out_d_coords = (int32_t *)d_coords_out;
    *out_Nf = cur_N;

    /* Drain g_stream BEFORE freeing transient buffers: the d_coords_out
     * memcpy above is async on g_stream and its source is d_gxc[final],
     * which we hipFree below. Without this sync, hipFree can unmap the
     * source pages while the memcpy is still pending → caller reads zeros.
     * This was the root cause of nondeterministic tex_coord collapse to
     * (0,0,0) observed across e2e --tex runs. */
    if (g_stream) hipStreamSynchronize(g_stream);

    /* Free transient buffers we own here. The persistent C2S scratch + cached
     * weights are intentionally kept across calls (file-scope state). */
    if (d_slat) hipFree(d_slat);
    if (d_tmp) hipFree(d_tmp);
    if (d_mlp) hipFree(d_mlp);
    if (d_keys) hipFree(d_keys);
    if (d_vals) hipFree(d_vals);
    /* Cache uploads. */
    for (int s = 0; s < T2SD_MAX_STAGES; s++) {
        if (d_gi[s]) hipFree(d_gi[s]);
        if (d_gs[s]) hipFree(d_gs[s]);
        if (d_gxc[s]) hipFree(d_gxc[s]);
        if (d_nmap_cn[s]) hipFree(d_nmap_cn[s]);
        if (d_nmap_pc[s]) hipFree(d_nmap_pc[s]);
    }
    /* d_feats from stage 0 was a fresh hipMalloc; later stages reassign to
     * persistent C2S scratch. We can't safely free that without double-freeing
     * the persistent buffer on next call — leak the stage-0 fresh allocation
     * as before. d_coords stage-0 was a fresh upload; same story. */

    return 0;
}

int hip_shape_dec_forward(hip_shape_dec_ctx *ctx,
                          const float *slat_feats,
                          const int32_t *coords,
                          int N, int slat_C,
                          float **out_d_feats,
                          int32_t **out_d_coords,
                          int *out_Nf)
{
    return hip_shape_dec_forward_ex(ctx, slat_feats, coords, N, slat_C, NULL,
                                    out_d_feats, out_d_coords, out_Nf);
}
