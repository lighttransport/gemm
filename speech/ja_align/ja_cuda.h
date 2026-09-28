/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * ja_cuda.h - optional CUDA encoder for ja_align (same math as w2v2.h, same output struct).
 *
 * Self-contained: libcuda.so.1 and libnvrtc.so are opened with dlopen at run time (no CUDA
 * toolkit headers or link-time dependency); the few driver / NVRTC entry points used are
 * declared here from the public CUDA driver API. Kernels are compiled once with NVRTC for
 * the device's architecture. Weights are F32 on the device; kernels:
 *   gemm      64x64-tiled SGEMM with a row-stride A operand (strided convs = GEMM with
 *             lda = stride * C, K = k * C) and fused bias / accumulate
 *   posconv   grouped positional conv as a gathered GEMM (one grid slice per group)
 *   attention bidirectional multi-head attention, 4 warps with online softmax per query
 *   conv0 / groupnorm+GELU / layernorm / gelu / pos-add elementwise kernels
 *
 * Define JA_CUDA_IMPLEMENTATION in one TU (after ja_align.h / w2v2.h). Link with -ldl.
 */
#ifndef JA_CUDA_H
#define JA_CUDA_H

#include "w2v2.h"

typedef struct w2v2_cuda w2v2_cuda;

w2v2_cuda  *w2v2_cuda_create(const char *safetensors_path, int device, int verbose);
void        w2v2_cuda_free(w2v2_cuda *g);
int         w2v2_cuda_run(w2v2_cuda *g, const float *wav, int n, w2v2_output *out, const char *dump_dir);
const char *w2v2_cuda_name(const w2v2_cuda *g);

#endif /* JA_CUDA_H */

#if defined(JA_CUDA_IMPLEMENTATION) && !defined(JA_CUDA_IMPL_DONE)
#define JA_CUDA_IMPL_DONE

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- minimal driver / NVRTC API (public ABI) ---- */
typedef int jcu_result;
typedef int jcu_device;
typedef void *jcu_context;
typedef void *jcu_module;
typedef void *jcu_function;
typedef void *jcu_stream;
typedef unsigned long long jcu_ptr;
typedef void *jnvrtc_prog;

static struct {
    int ok;
    jcu_result (*Init)(unsigned);
    jcu_result (*DeviceGet)(jcu_device *, int);
    jcu_result (*DeviceGetAttribute)(int *, int, jcu_device);
    jcu_result (*DeviceGetName)(char *, int, jcu_device);
    jcu_result (*CtxCreate)(jcu_context *, unsigned, jcu_device);
    jcu_result (*CtxDestroy)(jcu_context);
    jcu_result (*CtxSetCurrent)(jcu_context);
    jcu_result (*CtxSynchronize)(void);
    jcu_result (*ModuleLoadData)(jcu_module *, const void *);
    jcu_result (*ModuleUnload)(jcu_module);
    jcu_result (*ModuleGetFunction)(jcu_function *, jcu_module, const char *);
    jcu_result (*MemAlloc)(jcu_ptr *, size_t);
    jcu_result (*MemFree)(jcu_ptr);
    jcu_result (*MemcpyHtoD)(jcu_ptr, const void *, size_t);
    jcu_result (*MemcpyDtoH)(void *, jcu_ptr, size_t);
    jcu_result (*LaunchKernel)(jcu_function, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, jcu_stream, void **, void **);
    int (*nvCreate)(jnvrtc_prog *, const char *, const char *, int, const char *const *, const char *const *);
    int (*nvCompile)(jnvrtc_prog, int, const char *const *);
    int (*nvLogSize)(jnvrtc_prog, size_t *);
    int (*nvLog)(jnvrtc_prog, char *);
    int (*nvCubinSize)(jnvrtc_prog, size_t *);
    int (*nvCubin)(jnvrtc_prog, char *);
    int (*nvPtxSize)(jnvrtc_prog, size_t *);
    int (*nvPtx)(jnvrtc_prog, char *);
    int (*nvDestroy)(jnvrtc_prog *);
} jcu;

static int jcu_load(void) {
    if (jcu.ok) return 0;
    void *cu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!cu) cu = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    static const char *nv_names[] = { "libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12",
                                      "/usr/local/cuda/lib64/libnvrtc.so", NULL };
    void *nv = NULL;
    for (int i = 0; nv_names[i] && !nv; i++) nv = dlopen(nv_names[i], RTLD_NOW | RTLD_LOCAL);
    if (!cu || !nv) { fprintf(stderr, "ja_cuda: libcuda / libnvrtc not found\n"); return -1; }
#define JL(lib, f, s) if (!(*(void **)&jcu.f = dlsym(lib, s))) { fprintf(stderr, "ja_cuda: missing %s\n", s); return -1; }
    JL(cu, Init, "cuInit"); JL(cu, DeviceGet, "cuDeviceGet"); JL(cu, DeviceGetAttribute, "cuDeviceGetAttribute");
    JL(cu, DeviceGetName, "cuDeviceGetName"); JL(cu, CtxCreate, "cuCtxCreate_v2"); JL(cu, CtxDestroy, "cuCtxDestroy_v2");
    JL(cu, CtxSetCurrent, "cuCtxSetCurrent"); JL(cu, CtxSynchronize, "cuCtxSynchronize");
    JL(cu, ModuleLoadData, "cuModuleLoadData"); JL(cu, ModuleUnload, "cuModuleUnload");
    JL(cu, ModuleGetFunction, "cuModuleGetFunction"); JL(cu, MemAlloc, "cuMemAlloc_v2"); JL(cu, MemFree, "cuMemFree_v2");
    JL(cu, MemcpyHtoD, "cuMemcpyHtoD_v2"); JL(cu, MemcpyDtoH, "cuMemcpyDtoH_v2"); JL(cu, LaunchKernel, "cuLaunchKernel");
    JL(nv, nvCreate, "nvrtcCreateProgram"); JL(nv, nvCompile, "nvrtcCompileProgram");
    JL(nv, nvLogSize, "nvrtcGetProgramLogSize"); JL(nv, nvLog, "nvrtcGetProgramLog");
    JL(nv, nvPtxSize, "nvrtcGetPTXSize"); JL(nv, nvPtx, "nvrtcGetPTX"); JL(nv, nvDestroy, "nvrtcDestroyProgram");
#undef JL
    *(void **)&jcu.nvCubinSize = dlsym(nv, "nvrtcGetCUBINSize");
    *(void **)&jcu.nvCubin = dlsym(nv, "nvrtcGetCUBIN");
    if (jcu.Init(0)) return -1;
    jcu.ok = 1;
    return 0;
}

static const char *ja_cuda_src =
"#define INFINITY __int_as_float(0x7f800000)\n"
"__device__ __forceinline__ float warp_sum(float v) {\n"
"  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);\n"
"  return v;\n"
"}\n"
"__device__ __forceinline__ float block_sum(float v, float *sh) {\n"
"  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;\n"
"  v = warp_sum(v);\n"
"  if (lane == 0) sh[w] = v;\n"
"  __syncthreads();\n"
"  float t = 0.f;\n"
"  for (int i = 0; i < nw; ++i) t += sh[i];\n"
"  __syncthreads();\n"
"  return t;\n"
"}\n"
"__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }\n"
"\n"
"/* C[m][n] (+)= bias[n] + sum_k A(m, k) W[n][k].  mode 0: A(m,k) = A[m*lda + k].\n"
" * mode 1 (grouped pos conv, group = blockIdx.z, gi = N, K = ks*gi): A(m, j*gi + i) =\n"
" * A[(m + j - pad) * lda + group*gi + i] (0 outside [0, M)); W and C are offset per group. */\n"
"extern \"C\" __global__ void gemm(const float *__restrict__ A, int M, int lda, int K, const float *__restrict__ W,\n"
"    int N, const float *__restrict__ bias, float *__restrict__ C, int ldc, int accum, int mode, int gi, int pad) {\n"
"  __shared__ float As[16][68];\n"
"  __shared__ float Bs[16][68];\n"
"  int g = blockIdx.z;\n"
"  if (mode == 1) { W += (size_t)g * N * K; C += g * gi; }\n"
"  int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;\n"
"  int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;\n"
"  float acc[4][4];\n"
"  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;\n"
"  for (int k0 = 0; k0 < K; k0 += 16) {\n"
"    for (int i = threadIdx.x; i < 1024; i += 256) {\n"
"      int r = i >> 4, kk = i & 15, m = m0 + r, k = k0 + kk;\n"
"      float v = 0.f;\n"
"      if (m < M && k < K) {\n"
"        if (mode == 0) v = A[(size_t)m * lda + k];\n"
"        else { int j = k / gi, ii = k - j * gi, src = m + j - pad;\n"
"          if (src >= 0 && src < M) v = A[(size_t)src * lda + g * gi + ii]; }\n"
"      }\n"
"      As[kk][r] = v;\n"
"      int n = n0 + r;\n"
"      Bs[kk][r] = (n < N && k < K) ? W[(size_t)n * K + k] : 0.f;\n"
"    }\n"
"    __syncthreads();\n"
"#pragma unroll\n"
"    for (int kk = 0; kk < 16; ++kk) {\n"
"      float a[4], b[4];\n"
"#pragma unroll\n"
"      for (int i = 0; i < 4; ++i) { a[i] = As[kk][ty * 4 + i]; b[i] = Bs[kk][tx * 4 + i]; }\n"
"#pragma unroll\n"
"      for (int i = 0; i < 4; ++i)\n"
"#pragma unroll\n"
"        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];\n"
"    }\n"
"    __syncthreads();\n"
"  }\n"
"  for (int i = 0; i < 4; ++i) {\n"
"    int m = m0 + ty * 4 + i;\n"
"    if (m >= M) continue;\n"
"    for (int j = 0; j < 4; ++j) {\n"
"      int n = n0 + tx * 4 + j;\n"
"      if (n >= N) continue;\n"
"      float v = acc[i][j] + (bias ? bias[(mode == 1 ? g * gi : 0) + n] : 0.f);\n"
"      float *o = C + (size_t)m * ldc + n;\n"
"      *o = accum ? *o + v : v;\n"
"    }\n"
"  }\n"
"}\n"
"\n"
"extern \"C\" __global__ void conv0(const float *x, const float *w, float *y, int T, int C, int k, int s) {\n"
"  size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (e >= (size_t)T * C) return;\n"
"  int t = (int)(e / C), c = (int)(e % C);\n"
"  float a = 0.f;\n"
"  for (int j = 0; j < k; ++j) a += w[c * k + j] * x[(size_t)t * s + j];\n"
"  y[e] = a;\n"
"}\n"
"\n"
"/* per-channel GroupNorm over time (groups == channels) + GELU; one block per channel */\n"
"extern \"C\" __global__ void groupnorm_gelu(float *x, const float *w, const float *b, int T, int C) {\n"
"  __shared__ float sh[32];\n"
"  int c = blockIdx.x;\n"
"  float s = 0.f;\n"
"  for (int t = threadIdx.x; t < T; t += blockDim.x) s += x[(size_t)t * C + c];\n"
"  float mu = block_sum(s, sh) / T;\n"
"  float v = 0.f;\n"
"  for (int t = threadIdx.x; t < T; t += blockDim.x) { float d = x[(size_t)t * C + c] - mu; v += d * d; }\n"
"  float r = rsqrtf(block_sum(v, sh) / T + 1e-5f);\n"
"  for (int t = threadIdx.x; t < T; t += blockDim.x) {\n"
"    float *p = &x[(size_t)t * C + c];\n"
"    *p = gelu((*p - mu) * r * w[c] + b[c]);\n"
"  }\n"
"}\n"
"\n"
"extern \"C\" __global__ void gelu_inplace(float *x, size_t n) {\n"
"  size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (e < n) x[e] = gelu(x[e]);\n"
"}\n"
"\n"
"extern \"C\" __global__ void layernorm(const float *x, float *y, const float *w, const float *b, int n, float eps) {\n"
"  __shared__ float sh[32];\n"
"  const float *xr = x + (size_t)blockIdx.x * n;\n"
"  float *yr = y + (size_t)blockIdx.x * n;\n"
"  float s = 0.f;\n"
"  for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];\n"
"  float mu = block_sum(s, sh) / n;\n"
"  float v = 0.f;\n"
"  for (int i = threadIdx.x; i < n; i += blockDim.x) { float d = xr[i] - mu; v += d * d; }\n"
"  float r = rsqrtf(block_sum(v, sh) / n + eps);\n"
"  for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mu) * r * w[i] + b[i];\n"
"}\n"
"\n"
"/* h += gelu(pc) (pc already includes the conv bias) */\n"
"extern \"C\" __global__ void add_gelu(float *h, const float *pc, size_t n) {\n"
"  size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (e < n) h[e] += gelu(pc[e]);\n"
"}\n"
"\n"
"/* bidirectional MHA over qkv rows [T][3H]; one block (4 warps) per (query, head); hd <= 128 */\n"
"extern \"C\" __global__ void attention(const float *qkv, float *out, int T, int nh, int hd) {\n"
"  __shared__ float sm[4], sl[4], sacc[4][128];\n"
"  int i = blockIdx.x, h = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;\n"
"  int H = nh * hd, S = 3 * H, per = hd / 32;\n"
"  float scale = rsqrtf((float)hd);\n"
"  float qv[4], acc[4];\n"
"  for (int e = 0; e < 4; ++e) { qv[e] = e < per ? qkv[(size_t)i * S + h * hd + lane + 32 * e] : 0.f; acc[e] = 0.f; }\n"
"  float m = -INFINITY, l = 0.f;\n"
"  for (int j = w; j < T; j += 4) {\n"
"    const float *kr = qkv + (size_t)j * S + H + h * hd, *vr = qkv + (size_t)j * S + 2 * H + h * hd;\n"
"    float d = 0.f;\n"
"    for (int e = 0; e < per; ++e) d += qv[e] * kr[lane + 32 * e];\n"
"    d = warp_sum(d) * scale;\n"
"    float mn = fmaxf(m, d), c = expf(m - mn), p = expf(d - mn);\n"
"    l = l * c + p;\n"
"    for (int e = 0; e < per; ++e) acc[e] = acc[e] * c + p * vr[lane + 32 * e];\n"
"    m = mn;\n"
"  }\n"
"  if (lane == 0) { sm[w] = m; sl[w] = l; }\n"
"  for (int e = 0; e < per; ++e) sacc[w][lane + 32 * e] = acc[e];\n"
"  __syncthreads();\n"
"  if (w == 0) {\n"
"    float M = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3])), L = 0.f, c[4];\n"
"    for (int k = 0; k < 4; ++k) { c[k] = sm[k] == -INFINITY ? 0.f : expf(sm[k] - M); L += sl[k] * c[k]; }\n"
"    for (int e = 0; e < per; ++e) {\n"
"      float s = 0.f;\n"
"      for (int k = 0; k < 4; ++k) s += sacc[k][lane + 32 * e] * c[k];\n"
"      out[(size_t)i * H + h * hd + lane + 32 * e] = s / L;\n"
"    }\n"
"  }\n"
"}\n";

typedef struct { jcu_ptr ln1w, ln1b, ln2w, ln2b, qkv, bqkv, o, bo, fc1, b1, fc2, b2; } jg_layer;

struct w2v2_cuda {
    char name[128];
    jcu_context ctx;
    jcu_module mod;
    jcu_function f_gemm, f_conv0, f_gn, f_gelu, f_ln, f_addgelu, f_att;
    int n_conv, conv_k[8], conv_s[8], C, H, nh, I, n_layers, inter, pos_k, pos_groups, n_phon, n_kana;
    jcu_ptr conv0_w, gn_w, gn_b, conv[8], fp_lnw, fp_lnb, fp_w, fp_b, pos_w, pos_b, fln_w, fln_b;
    jcu_ptr phon_w, phon_b, kana_w, kana_b;
    jg_layer *L;
};

static jcu_ptr jg__up(const float *h, size_t n) {
    jcu_ptr d = 0;
    if (jcu.MemAlloc(&d, (n ? n : 1) * 4)) { fprintf(stderr, "ja_cuda: alloc failed\n"); exit(1); }
    if (n) jcu.MemcpyHtoD(d, h, n * 4);
    return d;
}
static jcu_ptr jg__alloc(size_t n) {
    jcu_ptr d = 0;
    if (jcu.MemAlloc(&d, (n ? n : 1) * 4)) { fprintf(stderr, "ja_cuda: alloc of %zu floats failed\n", n); exit(1); }
    return d;
}
static jcu_ptr jg__tensor(const ja_st *st, const char *name, size_t expect) {
    size_t n = 0;
    float *d = ja_st_f32(st, name, &n);
    if (!d || (expect && n != expect)) { fprintf(stderr, "ja_cuda: bad tensor %s\n", name); exit(1); }
    jcu_ptr p = jg__up(d, n);
    free(d);
    return p;
}

static int jg__compile(w2v2_cuda *g, jcu_device dev, int verbose) {
    int major = 0, minor = 0;
    jcu.DeviceGetAttribute(&major, 75, dev);  /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */
    jcu.DeviceGetAttribute(&minor, 76, dev);  /* ..._MINOR */
    char arch[48];
    snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
    const char *opts[] = { arch };
    jnvrtc_prog prog;
    if (jcu.nvCreate(&prog, ja_cuda_src, "ja_cuda", 0, NULL, NULL)) return -1;
    if (verbose) fprintf(stderr, "ja_cuda: compiling kernels for sm_%d%d\n", major, minor);
    if (jcu.nvCompile(prog, 1, opts)) {
        size_t ls = 0;
        jcu.nvLogSize(prog, &ls);
        char *log = (char *)malloc(ls + 1);
        jcu.nvLog(prog, log);
        log[ls] = 0;
        fprintf(stderr, "ja_cuda: NVRTC error:\n%s\n", log);
        free(log);
        return -1;
    }
    size_t sz = 0;
    char *blob = NULL;
    if (jcu.nvCubinSize && jcu.nvCubin && !jcu.nvCubinSize(prog, &sz) && sz) {
        blob = (char *)malloc(sz);
        jcu.nvCubin(prog, blob);
    } else {
        jcu.nvPtxSize(prog, &sz);
        blob = (char *)malloc(sz);
        jcu.nvPtx(prog, blob);
    }
    jcu.nvDestroy(&prog);
    int rc = jcu.ModuleLoadData(&g->mod, blob) ? -1 : 0;
    free(blob);
    return rc;
}

w2v2_cuda *w2v2_cuda_create(const char *path, int device, int verbose) {
    if (jcu_load()) return NULL;
    ja_st st;
    if (ja_st_open(&st, path)) { fprintf(stderr, "ja_cuda: cannot open %s\n", path); return NULL; }
    w2v2_cuda *g = (w2v2_cuda *)calloc(1, sizeof(*g));
    jcu_device dev;
    if (jcu.DeviceGet(&dev, device) || jcu.CtxCreate(&g->ctx, 0, dev)) { free(g); ja_st_close(&st); return NULL; }
    jcu.DeviceGetName(g->name, sizeof(g->name), dev);
    if (jg__compile(g, dev, verbose)) { w2v2_cuda_free(g); ja_st_close(&st); return NULL; }
#define JF(f, n) if (jcu.ModuleGetFunction(&g->f, g->mod, n)) { fprintf(stderr, "ja_cuda: kernel %s missing\n", n); return NULL; }
    JF(f_gemm, "gemm"); JF(f_conv0, "conv0"); JF(f_gn, "groupnorm_gelu"); JF(f_gelu, "gelu_inplace");
    JF(f_ln, "layernorm"); JF(f_addgelu, "add_gelu"); JF(f_att, "attention");
#undef JF
    char nm[256];
    for (g->n_conv = 0; g->n_conv < 8; g->n_conv++) {
        snprintf(nm, sizeof(nm), "feature_extractor.conv_layers.%d.conv.weight", g->n_conv);
        const ja_st_tensor *t = ja_st_find(&st, nm);
        if (!t) break;
        g->conv_k[g->n_conv] = (int)t->shape[2];
        g->conv_s[g->n_conv] = g->n_conv == 0 ? 5 : 2;
        g->C = (int)t->shape[0];
    }
    int C = g->C;
    g->conv0_w = jg__tensor(&st, "feature_extractor.conv_layers.0.conv.weight", (size_t)C * g->conv_k[0]);
    g->gn_w = jg__tensor(&st, "feature_extractor.conv_layers.0.layer_norm.weight", (size_t)C);
    g->gn_b = jg__tensor(&st, "feature_extractor.conv_layers.0.layer_norm.bias", (size_t)C);
    for (int i = 1; i < g->n_conv; i++) {
        int k = g->conv_k[i];
        snprintf(nm, sizeof(nm), "feature_extractor.conv_layers.%d.conv.weight", i);
        float *w = ja_st_f32(&st, nm, NULL);
        float *r = (float *)malloc(sizeof(float) * (size_t)C * C * k);
        for (int o = 0; o < C; o++)
            for (int ci = 0; ci < C; ci++)
                for (int j = 0; j < k; j++) r[(size_t)o * C * k + (size_t)j * C + ci] = w[((size_t)o * C + ci) * k + j];
        g->conv[i] = jg__up(r, (size_t)C * C * k);
        free(w); free(r);
    }
    g->fp_lnw = jg__tensor(&st, "feature_projection.layer_norm.weight", (size_t)C);
    g->fp_lnb = jg__tensor(&st, "feature_projection.layer_norm.bias", (size_t)C);
    g->H = (int)ja_st_find(&st, "feature_projection.projection.weight")->shape[0];
    int H = g->H;
    g->fp_w = jg__tensor(&st, "feature_projection.projection.weight", (size_t)H * C);
    g->fp_b = jg__tensor(&st, "feature_projection.projection.bias", (size_t)H);
    const ja_st_tensor *tpc = ja_st_find(&st, "encoder.pos_conv_embed.conv.weight");
    int gi = (int)tpc->shape[1];
    g->pos_k = (int)tpc->shape[2];
    g->pos_groups = H / gi;
    {   /* [H][gi][k] -> per group [gi out][k*gi] with kk = j*gi + i */
        float *w = ja_st_f32(&st, "encoder.pos_conv_embed.conv.weight", NULL);
        float *r = (float *)malloc(sizeof(float) * (size_t)H * gi * g->pos_k);
        for (int gg = 0; gg < g->pos_groups; gg++)
            for (int o = 0; o < gi; o++)
                for (int i = 0; i < gi; i++)
                    for (int j = 0; j < g->pos_k; j++)
                        r[((size_t)gg * gi + o) * gi * g->pos_k + (size_t)j * gi + i] =
                            w[(((size_t)(gg * gi + o)) * gi + i) * g->pos_k + j];
        g->pos_w = jg__up(r, (size_t)H * gi * g->pos_k);
        free(w); free(r);
    }
    g->pos_b = jg__tensor(&st, "encoder.pos_conv_embed.conv.bias", (size_t)H);
    for (g->n_layers = 0;; g->n_layers++) {
        snprintf(nm, sizeof(nm), "encoder.layers.%d.attention.q_proj.weight", g->n_layers);
        if (!ja_st_find(&st, nm)) break;
    }
    g->I = (int)ja_st_find(&st, "encoder.layers.0.feed_forward.intermediate_dense.weight")->shape[0];
    g->nh = 16;
    int I = g->I;
    g->L = (jg_layer *)calloc((size_t)g->n_layers, sizeof(jg_layer));
    for (int l = 0; l < g->n_layers; l++) {
        jg_layer *L = &g->L[l];
#define JP(s) (snprintf(nm, sizeof(nm), "encoder.layers.%d.%s", l, s), nm)
        L->ln1w = jg__tensor(&st, JP("layer_norm.weight"), (size_t)H);
        L->ln1b = jg__tensor(&st, JP("layer_norm.bias"), (size_t)H);
        L->ln2w = jg__tensor(&st, JP("final_layer_norm.weight"), (size_t)H);
        L->ln2b = jg__tensor(&st, JP("final_layer_norm.bias"), (size_t)H);
        float *cat = (float *)malloc(sizeof(float) * 3 * (size_t)H * H), *bc = (float *)malloc(sizeof(float) * 3 * H);
        static const char *qkvn[3] = { "q_proj", "k_proj", "v_proj" };
        for (int q = 0; q < 3; q++) {
            char t[64];
            snprintf(t, sizeof(t), "attention.%s.weight", qkvn[q]);
            float *w = ja_st_f32(&st, JP(t), NULL);
            memcpy(cat + (size_t)q * H * H, w, sizeof(float) * (size_t)H * H);
            free(w);
            snprintf(t, sizeof(t), "attention.%s.bias", qkvn[q]);
            w = ja_st_f32(&st, JP(t), NULL);
            memcpy(bc + (size_t)q * H, w, sizeof(float) * H);
            free(w);
        }
        L->qkv = jg__up(cat, 3 * (size_t)H * H);
        L->bqkv = jg__up(bc, 3 * (size_t)H);
        free(cat); free(bc);
        L->o = jg__tensor(&st, JP("attention.out_proj.weight"), (size_t)H * H);
        L->bo = jg__tensor(&st, JP("attention.out_proj.bias"), (size_t)H);
        L->fc1 = jg__tensor(&st, JP("feed_forward.intermediate_dense.weight"), (size_t)I * H);
        L->b1 = jg__tensor(&st, JP("feed_forward.intermediate_dense.bias"), (size_t)I);
        L->fc2 = jg__tensor(&st, JP("feed_forward.output_dense.weight"), (size_t)H * I);
        L->b2 = jg__tensor(&st, JP("feed_forward.output_dense.bias"), (size_t)H);
#undef JP
    }
    g->fln_w = jg__tensor(&st, "encoder.layer_norm.weight", (size_t)H);
    g->fln_b = jg__tensor(&st, "encoder.layer_norm.bias", (size_t)H);
    g->n_phon = (int)ja_st_find(&st, "phoneme_head.weight")->shape[0];
    g->n_kana = (int)ja_st_find(&st, "kana_head.weight")->shape[0];
    g->phon_w = jg__tensor(&st, "phoneme_head.weight", (size_t)g->n_phon * H);
    g->phon_b = jg__tensor(&st, "phoneme_head.bias", (size_t)g->n_phon);
    g->kana_w = jg__tensor(&st, "kana_head.weight", (size_t)g->n_kana * H);
    g->kana_b = jg__tensor(&st, "kana_head.bias", (size_t)g->n_kana);
    g->inter = st.meta_inter[0] ? atoi(st.meta_inter) : g->n_layers / 2;
    ja_st_close(&st);
    if (verbose) fprintf(stderr, "ja_cuda: %s, %d layers uploaded\n", g->name, g->n_layers);
    return g;
}

const char *w2v2_cuda_name(const w2v2_cuda *g) { return g->name; }

void w2v2_cuda_free(w2v2_cuda *g) {
    if (!g) return;
    if (g->ctx) {
        jcu.CtxSetCurrent(g->ctx);
        if (g->mod) jcu.ModuleUnload(g->mod);
        jcu.CtxDestroy(g->ctx);   /* releases all device memory */
    }
    free(g->L);
    free(g);
}

#define JG_LAUNCH(fn, gx, gy, gz, bx, ...) do { \
    void *a_[] = { __VA_ARGS__ }; \
    if (jcu.LaunchKernel((fn), (gx), (gy), (gz), (bx), 1, 1, 0, NULL, a_, NULL)) { \
        fprintf(stderr, "ja_cuda: launch failed at line %d\n", __LINE__); goto fail; } \
} while (0)

static unsigned jg__bl(size_t n) { return (unsigned)((n + 255) / 256); }

static void jg__dump(const char *dir, const char *name, jcu_ptr x, int T, int C) {
    if (!dir) return;
    float *h = (float *)malloc(sizeof(float) * (size_t)T * C);
    jcu.MemcpyDtoH(h, x, sizeof(float) * (size_t)T * C);
    char p[1024];
    snprintf(p, sizeof(p), "%s/%s.npy", dir, name);
    int dims[2] = { T, C };
    ja_npy_save_f32(p, h, 2, dims);
    free(h);
}

int w2v2_cuda_run(w2v2_cuda *g, const float *wav, int n, w2v2_output *out, const char *dump) {
    memset(out, 0, sizeof(*out));
    jcu.CtxSetCurrent(g->ctx);
    int C = g->C, H = g->H, I = g->I, nh = g->nh, hd = H / nh, zero = 0, one = 1;
    double mu = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mu += wav[i];
    mu /= n;
    for (int i = 0; i < n; i++) { double d = wav[i] - mu; var += d * d; }
    var /= n;
    float inv = (float)(1.0 / sqrt(var + 1e-7));
    float *xh = (float *)malloc(sizeof(float) * (size_t)n);
    for (int i = 0; i < n; i++) xh[i] = (float)(wav[i] - mu) * inv;
    int T = (n - g->conv_k[0]) / g->conv_s[0] + 1;
    if (T < 1) { free(xh); return -1; }
    size_t T0 = (size_t)T;
    jcu_ptr x = jg__up(xh, (size_t)n);
    free(xh);
    size_t big = T0 * C;
    jcu_ptr a = jg__alloc(big), b = jg__alloc(big);
    jcu_ptr hs = 0, xn = 0, qkv = 0, att = 0, ff = 0, tmp = 0, pl = 0, kl = 0;
    int rc = -1, s0 = g->conv_s[0], k0 = g->conv_k[0];
    JG_LAUNCH(g->f_conv0, jg__bl(T0 * C), 1, 1, 256, &x, &g->conv0_w, &a, &T, &C, &k0, &s0);
    JG_LAUNCH(g->f_gn, (unsigned)C, 1, 1, 256, &a, &g->gn_w, &g->gn_b, &T, &C);
    jcu_ptr nul = 0;
    for (int i = 1; i < g->n_conv; i++) {
        int k = g->conv_k[i], s = g->conv_s[i], To = (T - k) / s + 1, lda = s * C, K = k * C;
        JG_LAUNCH(g->f_gemm, (unsigned)((C + 63) / 64), (unsigned)((To + 63) / 64), 1, 256,
                  &a, &To, &lda, &K, &g->conv[i], &C, &nul, &b, &C, &zero, &zero, &zero, &zero);
        size_t ne = (size_t)To * C;
        JG_LAUNCH(g->f_gelu, jg__bl(ne), 1, 1, 256, &b, &ne);
        jcu_ptr t = a; a = b; b = t;
        T = To;
    }
    jg__dump(dump, "w2v_feat", a, T, C);
    float eps = 1e-5f;
    JG_LAUNCH(g->f_ln, (unsigned)T, 1, 1, 256, &a, &b, &g->fp_lnw, &g->fp_lnb, &C, &eps);
    hs = jg__alloc((size_t)T * H); xn = jg__alloc((size_t)T * H); qkv = jg__alloc((size_t)T * 3 * H);
    att = jg__alloc((size_t)T * H); ff = jg__alloc((size_t)T * I); tmp = jg__alloc((size_t)T * H);
    JG_LAUNCH(g->f_gemm, (unsigned)((H + 63) / 64), (unsigned)((T + 63) / 64), 1, 256,
              &b, &T, &C, &C, &g->fp_w, &H, &g->fp_b, &hs, &H, &zero, &zero, &zero, &zero);
    {
        int gi = H / g->pos_groups, K = g->pos_k * gi, pad = g->pos_k / 2;
        JG_LAUNCH(g->f_gemm, (unsigned)((gi + 63) / 64), (unsigned)((T + 63) / 64), (unsigned)g->pos_groups, 256,
                  &hs, &T, &H, &K, &g->pos_w, &gi, &g->pos_b, &tmp, &H, &zero, &one, &gi, &pad);
        size_t ne = (size_t)T * H;
        JG_LAUNCH(g->f_addgelu, jg__bl(ne), 1, 1, 256, &hs, &tmp, &ne);
    }
    jg__dump(dump, "w2v_h0", hs, T, H);
    out->T = T; out->n_phon = g->n_phon; out->n_kana = g->n_kana;
    out->phon_logp = (float *)malloc(sizeof(float) * (size_t)T * g->n_phon);
    out->kana_logp = (float *)malloc(sizeof(float) * (size_t)T * g->n_kana);
    pl = jg__alloc((size_t)T * g->n_phon);
    kl = jg__alloc((size_t)T * g->n_kana);
    int H3 = 3 * H;
    for (int l = 0; l < g->n_layers; l++) {
        jg_layer *L = &g->L[l];
        JG_LAUNCH(g->f_ln, (unsigned)T, 1, 1, 256, &hs, &xn, &L->ln1w, &L->ln1b, &H, &eps);
        JG_LAUNCH(g->f_gemm, (unsigned)((H3 + 63) / 64), (unsigned)((T + 63) / 64), 1, 256,
                  &xn, &T, &H, &H, &L->qkv, &H3, &L->bqkv, &qkv, &H3, &zero, &zero, &zero, &zero);
        JG_LAUNCH(g->f_att, (unsigned)T, (unsigned)nh, 1, 128, &qkv, &att, &T, &nh, &hd);
        JG_LAUNCH(g->f_gemm, (unsigned)((H + 63) / 64), (unsigned)((T + 63) / 64), 1, 256,
                  &att, &T, &H, &H, &L->o, &H, &L->bo, &hs, &H, &one, &zero, &zero, &zero);
        JG_LAUNCH(g->f_ln, (unsigned)T, 1, 1, 256, &hs, &xn, &L->ln2w, &L->ln2b, &H, &eps);
        JG_LAUNCH(g->f_gemm, (unsigned)((I + 63) / 64), (unsigned)((T + 63) / 64), 1, 256,
                  &xn, &T, &H, &H, &L->fc1, &I, &L->b1, &ff, &I, &zero, &zero, &zero, &zero);
        size_t ne = (size_t)T * I;
        JG_LAUNCH(g->f_gelu, jg__bl(ne), 1, 1, 256, &ff, &ne);
        JG_LAUNCH(g->f_gemm, (unsigned)((H + 63) / 64), (unsigned)((T + 63) / 64), 1, 256,
                  &ff, &T, &I, &I, &L->fc2, &H, &L->b2, &hs, &H, &one, &zero, &zero, &zero);
        if (l + 1 == g->inter) {
            char nm[32]; snprintf(nm, sizeof(nm), "w2v_h%d", g->inter);
            jg__dump(dump, nm, hs, T, H);
            JG_LAUNCH(g->f_gemm, (unsigned)((g->n_phon + 63) / 64), (unsigned)((T + 63) / 64), 1, 256,
                      &hs, &T, &H, &H, &g->phon_w, &g->n_phon, &g->phon_b, &pl, &g->n_phon, &zero, &zero, &zero, &zero);
        }
    }
    JG_LAUNCH(g->f_ln, (unsigned)T, 1, 1, 256, &hs, &xn, &g->fln_w, &g->fln_b, &H, &eps);
    jg__dump(dump, "w2v_final", xn, T, H);
    JG_LAUNCH(g->f_gemm, (unsigned)((g->n_kana + 63) / 64), (unsigned)((T + 63) / 64), 1, 256,
              &xn, &T, &H, &H, &g->kana_w, &g->n_kana, &g->kana_b, &kl, &g->n_kana, &zero, &zero, &zero, &zero);
    if (jcu.CtxSynchronize()) goto fail;
    jcu.MemcpyDtoH(out->phon_logp, pl, sizeof(float) * (size_t)T * g->n_phon);
    jcu.MemcpyDtoH(out->kana_logp, kl, sizeof(float) * (size_t)T * g->n_kana);
    for (int t = 0; t < T; t++) {
        ja_log_softmax(out->phon_logp + (size_t)t * g->n_phon, g->n_phon);
        ja_log_softmax(out->kana_logp + (size_t)t * g->n_kana, g->n_kana);
    }
    if (dump) {
        char p[1024];
        int dp[2] = { T, g->n_phon }, dk[2] = { T, g->n_kana };
        snprintf(p, sizeof(p), "%s/phoneme_logp.npy", dump); ja_npy_save_f32(p, out->phon_logp, 2, dp);
        snprintf(p, sizeof(p), "%s/kana_logp.npy", dump); ja_npy_save_f32(p, out->kana_logp, 2, dk);
    }
    rc = 0;
fail:
    jcu.MemFree(x); jcu.MemFree(a); jcu.MemFree(b);
    if (hs) jcu.MemFree(hs);
    if (xn) jcu.MemFree(xn);
    if (qkv) jcu.MemFree(qkv);
    if (att) jcu.MemFree(att);
    if (ff) jcu.MemFree(ff);
    if (tmp) jcu.MemFree(tmp);
    if (pl) jcu.MemFree(pl);
    if (kl) jcu.MemFree(kl);
    if (rc) w2v2_output_free(out);
    return rc;
}

#endif /* JA_CUDA_IMPLEMENTATION */
