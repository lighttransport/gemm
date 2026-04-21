/*
 * verify_mask_decoder_cuda.c — CUDA port of the SAM2 mask decoder.
 *
 * Mirrors verify_mask_decoder.c (CPU reference). Runs the full mask decoder
 * pipeline on GPU and compares against /tmp/sam2_trace/md_*.npy.
 *
 * Kernels:
 *   linear      — generic (N, din) @ (din, dout)^T + bias
 *   ln_last     — LayerNorm over last dim
 *   ln_chw      — LayerNorm channels_first on BCHW
 *   gelu, relu  — elementwise activations
 *   add_vec     — y += x (elementwise)
 *   add3        — out = a + b + c (broadcast over spatial for dense+embed)
 *   conv_t2x    — ConvTranspose2d k=s=2
 *   attn_pre    — Q/K/V projections (reuses linear)
 *   attn_scores — computes softmax attention scores + output
 *   flat_bchw_to_hwc, flat_hwc_to_bchw — layout converts
 */
#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_npy_f32(const char *path, int dims[6], int *ndims) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    uint8_t h10[10]; if (fread(h10, 1, 10, f) != 10) { fclose(f); return NULL; }
    if (memcmp(h10, "\x93NUMPY", 6)) { fclose(f); return NULL; }
    uint16_t hlen = (uint16_t)(h10[8] | (h10[9] << 8));
    char *hdr = (char *)malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = '\0';
    if (!strstr(hdr, "'descr': '<f4'")) { free(hdr); fclose(f); return NULL; }
    char *p = strchr(hdr, '('), *q = strchr(hdr, ')');
    p++; int n = 0;
    while (p < q && n < 6) {
        while (p < q && (*p < '0' || *p > '9')) p++;
        if (p >= q) break;
        dims[n++] = (int)strtol(p, &p, 10);
    }
    free(hdr);
    size_t cnt = 1; for (int i = 0; i < n; i++) cnt *= (size_t)dims[i];
    float *x = (float *)malloc(cnt * sizeof(float));
    if (fread(x, sizeof(float), cnt, f) != cnt) { free(x); fclose(f); return NULL; }
    fclose(f); *ndims = n; return x;
}

static st_context *ST;
static CUfunction fn_lin, fn_add_vec, fn_relu, fn_gelu, fn_sigmoid;
static CUfunction fn_ln_last, fn_ln_chw, fn_convt2x;
static CUfunction fn_bchw_to_hwc, fn_scores_q, fn_scores_softmax, fn_scores_out;
static CUfunction fn_matmul_cb, fn_axpy;

static CUdeviceptr upf(const float *h, size_t n) {
    CUdeviceptr d; cuMemAlloc(&d, n*4); cuMemcpyHtoD(d, h, n*4); return d;
}
static CUdeviceptr dalloc(size_t n) { CUdeviceptr d; cuMemAlloc(&d, n*4); return d; }
static void dzero(CUdeviceptr d, size_t n) { cuMemsetD8(d, 0, n*4); }

static CUdeviceptr TD(const char *name, size_t *out_n) {
    int i = safetensors_find(ST, name);
    if (i < 0) { fprintf(stderr, "missing %s\n", name); exit(4); }
    if (strcmp(safetensors_dtype(ST, i), "F32")) exit(4);
    float *h = (float *)safetensors_data(ST, i);
    size_t nbytes = safetensors_nbytes(ST, i);
    size_t n = nbytes / 4;
    if (out_n) *out_n = n;
    return upf(h, n);
}

static void diff_dev_host(const char *name, CUdeviceptr d, const float *ref, size_t n) {
    float *tmp = (float *)malloc(n*4);
    cuMemcpyDtoH(tmp, d, n*4);
    double mad = 0.0; float mxd = 0.f;
    for (size_t i = 0; i < n; i++) { float x = fabsf(tmp[i]-ref[i]); if (x > mxd) mxd = x; mad += x; }
    mad /= (double)n;
    fprintf(stderr, "  %-22s: max_abs=%.6g mean_abs=%.6g\n", name, mxd, mad);
    free(tmp);
}

/* ---------- Kernel source ---------- */
static const char *kern_src =
"extern \"C\" {\n"
"__global__ void linear(float *y, const float *x, const float *W, const float *b, int N, int din, int dout) {\n"
"  int o = blockIdx.y * blockDim.x + threadIdx.x; int t = blockIdx.x;\n"
"  if (t >= N || o >= dout) return;\n"
"  float acc = b ? b[o] : 0.f;\n"
"  const float *xt = x + (size_t)t*din;\n"
"  const float *wo = W + (size_t)o*din;\n"
"  for (int i = 0; i < din; i++) acc += wo[i] * xt[i];\n"
"  y[(size_t)t*dout + o] = acc;\n"
"}\n"
"__global__ void add_vec(float *y, const float *x, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += x[i];\n"
"}\n"
"__global__ void axpy(float *y, const float *a, const float *b, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a[i] + b[i];\n"
"}\n"
"__global__ void relu(float *x, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n && x[i] < 0.f) x[i] = 0.f;\n"
"}\n"
"__global__ void gelu(float *x, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (i < n) { float v = x[i]; x[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752440f)); }\n"
"}\n"
"__global__ void sigmoid(float *x, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = 1.f / (1.f + expf(-x[i]));\n"
"}\n"
/* LayerNorm on last dim; one block per token. blockDim.x = 256 threads, C <= 256. */
"__global__ void ln_last(float *x, const float *w, const float *b, int N, int C, float eps) {\n"
"  int t = blockIdx.x; if (t >= N) return;\n"
"  extern __shared__ float smem[];\n"
"  float *xs = smem; float *tmp = smem + C;\n"
"  int d = threadIdx.x;\n"
"  if (d < C) xs[d] = x[(size_t)t*C + d];\n"
"  __syncthreads();\n"
"  float sum = 0.f;\n"
"  if (d == 0) { for (int i = 0; i < C; i++) sum += xs[i]; tmp[0] = sum / C; }\n"
"  __syncthreads();\n"
"  float mean = tmp[0];\n"
"  if (d == 0) { float ss = 0.f; for (int i = 0; i < C; i++) { float v = xs[i]-mean; ss += v*v; } tmp[1] = rsqrtf(ss/C + eps); }\n"
"  __syncthreads();\n"
"  float inv = tmp[1];\n"
"  if (d < C) x[(size_t)t*C + d] = (xs[d] - mean) * inv * w[d] + b[d];\n"
"}\n"
/* LayerNorm channels_first on (B,C,H,W). One block per spatial pixel. */
"__global__ void ln_chw(float *x, const float *w, const float *b, int B, int C, int HW) {\n"
"  int bi = blockIdx.y; int p = blockIdx.x; if (bi >= B || p >= HW) return;\n"
"  extern __shared__ float smem[];\n"
"  int c = threadIdx.x;\n"
"  float *xs = smem;\n"
"  size_t base = (size_t)bi*C*HW + p;\n"
"  if (c < C) xs[c] = x[base + (size_t)c*HW];\n"
"  __syncthreads();\n"
"  __shared__ float mean, inv;\n"
"  if (c == 0) {\n"
"    float s = 0.f; for (int i = 0; i < C; i++) s += xs[i]; mean = s/C;\n"
"    float ss = 0.f; for (int i = 0; i < C; i++) { float v = xs[i]-mean; ss += v*v; }\n"
"    inv = rsqrtf(ss/C + 1e-6f);\n"
"  }\n"
"  __syncthreads();\n"
"  if (c < C) x[base + (size_t)c*HW] = (xs[c] - mean) * inv * w[c] + b[c];\n"
"}\n"
/* ConvTranspose2d with kernel=stride=2, non-overlapping.
 * W shape: (Ci, Co, 2, 2); bias: (Co,). Input BCHW (Ci,H,W); output (Co, 2H, 2W). B=1. */
"__global__ void conv_t2x(float *out, const float *in, const float *W, const float *b,\n"
"                          int Ci, int Co, int H, int Wdim) {\n"
"  int Ho = 2*H, Wo = 2*Wdim;\n"
"  int co = blockIdx.y; int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int total = Ho*Wo; if (co >= Co || idx >= total) return;\n"
"  int oi = idx / Wo, oj = idx % Wo;\n"
"  int i = oi>>1, j = oj>>1, dy = oi&1, dx = oj&1;\n"
"  float acc = b[co];\n"
"  for (int ci = 0; ci < Ci; ci++) {\n"
"    float wv = W[(((size_t)ci*Co + co)*2 + dy)*2 + dx];\n"
"    acc += in[((size_t)ci*H + i)*Wdim + j] * wv;\n"
"  }\n"
"  out[((size_t)co*Ho + oi)*Wo + oj] = acc;\n"
"}\n"
/* Flatten BCHW (1,C,H,W) into (HW, C). For stacking as tokens. */
"__global__ void bchw_to_hwc(float *y, const float *x, int C, int HW) {\n"
"  int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= HW) return;\n"
"  int c = blockIdx.y; if (c >= C) return;\n"
"  y[(size_t)p*C + c] = x[(size_t)c*HW + p];\n"
"}\n"
/* Add dense prompt (BCHW) + image_embed (BCHW) into a flattened HWC tokens buffer. */
"__global__ void merge_bchw_to_hwc(float *y, const float *a, const float *bb, int C, int HW) {\n"
"  int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= HW) return;\n"
"  int c = blockIdx.y; if (c >= C) return;\n"
"  y[(size_t)p*C + c] = a[(size_t)c*HW + p] + bb[(size_t)c*HW + p];\n"
"}\n"
/* Compute attention scores matrix: scores[h, i, j] = dot(Q[i, h*hd..], K[j, h*hd..]) * scale.
 * Grid: (nq, nk, nh). Block: (hd) threads cooperative dot using shared mem. */
"__global__ void attn_scores(float *scores, const float *Q, const float *K,\n"
"                             int nq, int nk, int intD, int nh, int hd, float scale) {\n"
"  int i = blockIdx.x, j = blockIdx.y, h = blockIdx.z;\n"
"  if (i >= nq || j >= nk || h >= nh) return;\n"
"  extern __shared__ float s[];\n"
"  int d = threadIdx.x;\n"
"  float prod = (d < hd) ? Q[(size_t)i*intD + h*hd + d] * K[(size_t)j*intD + h*hd + d] : 0.f;\n"
"  s[d] = prod; __syncthreads();\n"
"  for (int off = blockDim.x/2; off > 0; off >>= 1) { if (d < off) s[d] += s[d+off]; __syncthreads(); }\n"
"  if (d == 0) scores[((size_t)h*nq + i)*nk + j] = s[0] * scale;\n"
"}\n"
/* Softmax each row (h,i) over j=0..nk. One block per row, blockDim.x threads cooperate. */
"__global__ void attn_softmax(float *scores, int nq, int nk) {\n"
"  int row = blockIdx.x; int i = row % nq; int h = row / nq;\n"
"  float *sr = scores + ((size_t)h*nq + i)*nk;\n"
"  int tid = threadIdx.x, T = blockDim.x;\n"
"  extern __shared__ float sh[];\n"
"  float mx = -1e30f;\n"
"  for (int j = tid; j < nk; j += T) if (sr[j] > mx) mx = sr[j];\n"
"  sh[tid] = mx; __syncthreads();\n"
"  for (int off = T/2; off > 0; off >>= 1) { if (tid < off) sh[tid] = fmaxf(sh[tid], sh[tid+off]); __syncthreads(); }\n"
"  float M = sh[0];\n"
"  float sum = 0.f;\n"
"  for (int j = tid; j < nk; j += T) { float v = expf(sr[j]-M); sr[j] = v; sum += v; }\n"
"  sh[tid] = sum; __syncthreads();\n"
"  for (int off = T/2; off > 0; off >>= 1) { if (tid < off) sh[tid] += sh[tid+off]; __syncthreads(); }\n"
"  float inv = 1.f / sh[0];\n"
"  for (int j = tid; j < nk; j += T) sr[j] *= inv;\n"
"}\n"
/* Compute attn_out[i, h*hd+d] = sum_j scores[h,i,j] * V[j, h*hd+d]. Grid (nq, nh). Block hd. */
"__global__ void attn_apply(float *out, const float *scores, const float *V,\n"
"                            int nq, int nk, int intD, int nh, int hd) {\n"
"  int i = blockIdx.x, h = blockIdx.y; int d = threadIdx.x;\n"
"  if (i >= nq || h >= nh || d >= hd) return;\n"
"  float acc = 0.f;\n"
"  const float *sr = scores + ((size_t)h*nq + i)*nk;\n"
"  for (int j = 0; j < nk; j++) acc += sr[j] * V[(size_t)j*intD + h*hd + d];\n"
"  out[(size_t)i*intD + h*hd + d] = acc;\n"
"}\n"
/* y[m, p] = sum_c A[m,c] * B[c,p]. Grid (M, P/THR), block THR. */
"__global__ void matmul_cb(float *y, const float *A, const float *B, int M, int C, int P) {\n"
"  int m = blockIdx.x; int p = blockIdx.y * blockDim.x + threadIdx.x;\n"
"  if (m >= M || p >= P) return;\n"
"  float acc = 0.f;\n"
"  for (int c = 0; c < C; c++) acc += A[(size_t)m*C + c] * B[(size_t)c*P + p];\n"
"  y[(size_t)m*P + p] = acc;\n"
"}\n"
"}\n";

/* ---- Host helpers ---- */

static void k_linear(CUdeviceptr y, CUdeviceptr x, CUdeviceptr W, CUdeviceptr b,
                     int N, int din, int dout) {
    void *args[] = { &y, &x, &W, &b, &N, &din, &dout };
    int THR = 256;
    cuLaunchKernel(fn_lin, N, (dout + THR - 1)/THR, 1, THR, 1, 1, 0, 0, args, 0);
}
static void k_add_vec(CUdeviceptr y, CUdeviceptr x, int n) {
    void *a[] = { &y, &x, &n };
    cuLaunchKernel(fn_add_vec, (n+255)/256, 1, 1, 256, 1, 1, 0, 0, a, 0);
}
static void k_axpy(CUdeviceptr y, CUdeviceptr a, CUdeviceptr b, int n) {
    void *aa[] = { &y, &a, &b, &n };
    cuLaunchKernel(fn_axpy, (n+255)/256, 1, 1, 256, 1, 1, 0, 0, aa, 0);
}
static void k_relu(CUdeviceptr x, int n) {
    void *a[] = { &x, &n };
    cuLaunchKernel(fn_relu, (n+255)/256, 1, 1, 256, 1, 1, 0, 0, a, 0);
}
static void k_gelu(CUdeviceptr x, int n) {
    void *a[] = { &x, &n };
    cuLaunchKernel(fn_gelu, (n+255)/256, 1, 1, 256, 1, 1, 0, 0, a, 0);
}
static void k_sig(CUdeviceptr x, int n) {
    void *a[] = { &x, &n };
    cuLaunchKernel(fn_sigmoid, (n+255)/256, 1, 1, 256, 1, 1, 0, 0, a, 0);
}
static void k_ln_last(CUdeviceptr x, CUdeviceptr w, CUdeviceptr b, int N, int C, float eps) {
    void *a[] = { &x, &w, &b, &N, &C, &eps };
    size_t sh = (C + 8) * sizeof(float);
    cuLaunchKernel(fn_ln_last, N, 1, 1, 256, 1, 1, sh, 0, a, 0);
}
static void k_ln_chw(CUdeviceptr x, CUdeviceptr w, CUdeviceptr b, int B, int C, int HW) {
    void *a[] = { &x, &w, &b, &B, &C, &HW };
    size_t sh = C*sizeof(float);
    cuLaunchKernel(fn_ln_chw, HW, B, 1, 256, 1, 1, sh, 0, a, 0);
}
static void k_conv_t2x(CUdeviceptr out, CUdeviceptr in, CUdeviceptr W, CUdeviceptr b,
                       int Ci, int Co, int H, int Wdim) {
    int total = 4*H*Wdim;
    void *a[] = { &out, &in, &W, &b, &Ci, &Co, &H, &Wdim };
    cuLaunchKernel(fn_convt2x, (total+255)/256, Co, 1, 256, 1, 1, 0, 0, a, 0);
}
static void k_bchw_to_hwc(CUdeviceptr y, CUdeviceptr x, int C, int HW) {
    void *a[] = { &y, &x, &C, &HW };
    cuLaunchKernel(fn_bchw_to_hwc, (HW+255)/256, C, 1, 256, 1, 1, 0, 0, a, 0);
}
static void k_merge_bchw_to_hwc(CUdeviceptr y, CUdeviceptr a, CUdeviceptr b, int C, int HW) {
    void *aa[] = { &y, &a, &b, &C, &HW };
    cuLaunchKernel(fn_bchw_to_hwc, (HW+255)/256, C, 1, 256, 1, 1, 0, 0, aa, 0);
    /* NOTE: need merge kernel; but we'll use a separate path — see main. */
    (void)aa;
}

/* Power-of-two >= x, clamp to [32, 1024]. */
static int po2(int x) { int p = 32; while (p < x) p <<= 1; if (p > 1024) p = 1024; return p; }

/* Generic attention: returns out (nq, D). Allocates+frees Qp/Kp/Vp/scores/aout internally. */
static void run_attention(CUdeviceptr out, CUdeviceptr q_in, CUdeviceptr k_in, CUdeviceptr v_in,
                          int nq, int nk, int D, int intD, int nh,
                          CUdeviceptr Wq, CUdeviceptr Bq,
                          CUdeviceptr Wk, CUdeviceptr Bk,
                          CUdeviceptr Wv, CUdeviceptr Bv,
                          CUdeviceptr Wo, CUdeviceptr Bo) {
    int hd = intD / nh;
    float scale = 1.f / sqrtf((float)hd);
    CUdeviceptr Qp = dalloc((size_t)nq*intD);
    CUdeviceptr Kp = dalloc((size_t)nk*intD);
    CUdeviceptr Vp = dalloc((size_t)nk*intD);
    CUdeviceptr sc = dalloc((size_t)nh*nq*nk);
    CUdeviceptr ao = dalloc((size_t)nq*intD);

    k_linear(Qp, q_in, Wq, Bq, nq, D, intD);
    k_linear(Kp, k_in, Wk, Bk, nk, D, intD);
    k_linear(Vp, v_in, Wv, Bv, nk, D, intD);

    /* scores */
    {
        int P = po2(hd);
        void *a[] = { &sc, &Qp, &Kp, &nq, &nk, &intD, &nh, &hd, &scale };
        cuLaunchKernel(fn_scores_q, nq, nk, nh, P, 1, 1, P*sizeof(float), 0, a, 0);
    }
    /* softmax per row */
    {
        int rows = nh*nq;
        int T = po2(nk < 256 ? nk : 256); if (T < 32) T = 32;
        void *a[] = { &sc, &nq, &nk };
        cuLaunchKernel(fn_scores_softmax, rows, 1, 1, T, 1, 1, T*sizeof(float), 0, a, 0);
    }
    /* apply */
    {
        void *a[] = { &ao, &sc, &Vp, &nq, &nk, &intD, &nh, &hd };
        cuLaunchKernel(fn_scores_out, nq, nh, 1, hd, 1, 1, 0, 0, a, 0);
    }
    k_linear(out, ao, Wo, Bo, nq, intD, D);

    cuMemFree(Qp); cuMemFree(Kp); cuMemFree(Vp); cuMemFree(sc); cuMemFree(ao);
}

/* Constants */
#define HD 256
#define NH 8
#define SA_INT 256
#define CA_INT 128
#define NL 2
#define MD 2048
#define NMASK 4
#define H_IM 64
#define W_IM 64
#define N_IM (H_IM*W_IM)

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <model.safetensors> <refdir>\n", argv[0]); return 1; }
    const char *ckpt = argv[1]; const char *refdir = argv[2]; char path[1024];

    ST = safetensors_open(ckpt);
    if (!ST) return 3;
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 5;
    if (cuInit(0) != CUDA_SUCCESS) return 5;
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    if (cu_compile_kernels(&mod, dev, kern_src, "sam2_md", 1, "sam2") < 0) return 6;
    cuModuleGetFunction(&fn_lin, mod, "linear");
    cuModuleGetFunction(&fn_add_vec, mod, "add_vec");
    cuModuleGetFunction(&fn_axpy, mod, "axpy");
    cuModuleGetFunction(&fn_relu, mod, "relu");
    cuModuleGetFunction(&fn_gelu, mod, "gelu");
    cuModuleGetFunction(&fn_sigmoid, mod, "sigmoid");
    cuModuleGetFunction(&fn_ln_last, mod, "ln_last");
    cuModuleGetFunction(&fn_ln_chw, mod, "ln_chw");
    cuModuleGetFunction(&fn_convt2x, mod, "conv_t2x");
    cuModuleGetFunction(&fn_bchw_to_hwc, mod, "bchw_to_hwc");
    cuModuleGetFunction(&fn_scores_q, mod, "attn_scores");
    cuModuleGetFunction(&fn_scores_softmax, mod, "attn_softmax");
    cuModuleGetFunction(&fn_scores_out, mod, "attn_apply");
    cuModuleGetFunction(&fn_matmul_cb, mod, "matmul_cb");

    /* ---------- Load trace inputs ---------- */
    int d[6], nd;
    #define LOAD(v,name) \
        snprintf(path, sizeof(path), "%s/%s.npy", refdir, name); \
        float *v = read_npy_f32(path, d, &nd); \
        if (!v) { fprintf(stderr, "missing %s\n", name); return 2; }
    LOAD(h_md_image_embed, "md_image_embed");
    LOAD(h_md_image_pe,    "md_image_pe");
    LOAD(h_md_high_res_0,  "md_high_res_0");
    LOAD(h_md_high_res_1,  "md_high_res_1");
    LOAD(h_prompt_sparse,  "prompt_sparse");
    LOAD(h_prompt_dense,   "prompt_dense");
    #undef LOAD

    /* Upload */
    size_t N_SPA = (size_t)HD*N_IM;
    CUdeviceptr d_img_embed = upf(h_md_image_embed, N_SPA);
    CUdeviceptr d_img_pe    = upf(h_md_image_pe,    N_SPA);
    CUdeviceptr d_hr0       = upf(h_md_high_res_0, (size_t)32*256*256);
    CUdeviceptr d_hr1       = upf(h_md_high_res_1, (size_t)64*128*128);
    CUdeviceptr d_sparse    = upf(h_prompt_sparse,  2*HD);
    CUdeviceptr d_dense     = upf(h_prompt_dense,   N_SPA);

    /* ---------- Build tokens (8, 256) ---------- */
    size_t dummy;
    CUdeviceptr d_obj  = TD("mask_decoder.obj_score_token.weight", &dummy);
    CUdeviceptr d_iou  = TD("mask_decoder.iou_token.weight", &dummy);
    CUdeviceptr d_mt   = TD("mask_decoder.mask_tokens.weight", &dummy);

    int N_TOK = 8;
    CUdeviceptr d_queries = dalloc((size_t)N_TOK*HD);
    cuMemcpyDtoD(d_queries + 0*HD*4, d_obj, HD*4);
    cuMemcpyDtoD(d_queries + 1*HD*4, d_iou, HD*4);
    cuMemcpyDtoD(d_queries + 2*HD*4, d_mt,  4*HD*4);
    cuMemcpyDtoD(d_queries + 6*HD*4, d_sparse, 2*HD*4);

    /* Save point_pe = initial queries */
    CUdeviceptr d_point_pe = dalloc((size_t)N_TOK*HD);
    cuMemcpyDtoD(d_point_pe, d_queries, (size_t)N_TOK*HD*4);

    /* ---------- Build keys (N_IM, 256) = (image_embed + dense) transposed BCHW→HWC ---------- */
    CUdeviceptr d_keys = dalloc(N_SPA);
    CUdeviceptr d_pe_k = dalloc(N_SPA);
    /* axpy into BCHW buffer then transpose */
    {
        CUdeviceptr tmp = dalloc(N_SPA);
        k_axpy(tmp, d_img_embed, d_dense, (int)N_SPA);
        k_bchw_to_hwc(d_keys, tmp, HD, N_IM);
        cuMemFree(tmp);
        k_bchw_to_hwc(d_pe_k, d_img_pe, HD, N_IM);
    }

    /* Scratch */
    CUdeviceptr d_qbuf = dalloc((size_t)N_TOK*HD);
    CUdeviceptr d_abuf = dalloc((size_t)N_IM*HD);
    CUdeviceptr d_mlpH = dalloc((size_t)N_TOK*MD);
    CUdeviceptr d_qq_tok = dalloc((size_t)N_TOK*HD);
    CUdeviceptr d_qq_img = dalloc((size_t)N_IM*HD);

    const float LN_EPS = 1e-5f;

    /* ---------- Two-way transformer ---------- */
    for (int layer = 0; layer < NL; layer++) {
        char base[160]; snprintf(base, sizeof(base), "mask_decoder.transformer.layers.%d", layer);
        char n[256];
        #define WT(s) (snprintf(n,sizeof(n),"%s.%s",base,s), TD(n,&dummy))

        /* Self-attn */
        CUdeviceptr Wq = WT("self_attn.q_proj.weight"), Bq = WT("self_attn.q_proj.bias");
        CUdeviceptr Wk = WT("self_attn.k_proj.weight"), Bk = WT("self_attn.k_proj.bias");
        CUdeviceptr Wv = WT("self_attn.v_proj.weight"), Bv = WT("self_attn.v_proj.bias");
        CUdeviceptr Wo = WT("self_attn.o_proj.weight"), Bo = WT("self_attn.o_proj.bias");
        if (layer == 0) {
            run_attention(d_qbuf, d_queries, d_queries, d_queries,
                          N_TOK, N_TOK, HD, SA_INT, NH, Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo);
            cuMemcpyDtoD(d_queries, d_qbuf, (size_t)N_TOK*HD*4);
        } else {
            k_axpy(d_qq_tok, d_queries, d_point_pe, N_TOK*HD);
            run_attention(d_qbuf, d_qq_tok, d_qq_tok, d_queries,
                          N_TOK, N_TOK, HD, SA_INT, NH, Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo);
            k_add_vec(d_queries, d_qbuf, N_TOK*HD);
        }
        cuMemFree(Wq); cuMemFree(Bq); cuMemFree(Wk); cuMemFree(Bk);
        cuMemFree(Wv); cuMemFree(Bv); cuMemFree(Wo); cuMemFree(Bo);
        /* LN1 */
        { CUdeviceptr W = WT("layer_norm1.weight"), B = WT("layer_norm1.bias");
          k_ln_last(d_queries, W, B, N_TOK, HD, LN_EPS); cuMemFree(W); cuMemFree(B); }

        /* Cross-attn token → image */
        Wq = WT("cross_attn_token_to_image.q_proj.weight"); Bq = WT("cross_attn_token_to_image.q_proj.bias");
        Wk = WT("cross_attn_token_to_image.k_proj.weight"); Bk = WT("cross_attn_token_to_image.k_proj.bias");
        Wv = WT("cross_attn_token_to_image.v_proj.weight"); Bv = WT("cross_attn_token_to_image.v_proj.bias");
        Wo = WT("cross_attn_token_to_image.o_proj.weight"); Bo = WT("cross_attn_token_to_image.o_proj.bias");
        k_axpy(d_qq_tok, d_queries, d_point_pe, N_TOK*HD);
        k_axpy(d_qq_img, d_keys,    d_pe_k,     N_IM*HD);
        run_attention(d_qbuf, d_qq_tok, d_qq_img, d_keys,
                      N_TOK, N_IM, HD, CA_INT, NH, Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo);
        k_add_vec(d_queries, d_qbuf, N_TOK*HD);
        cuMemFree(Wq); cuMemFree(Bq); cuMemFree(Wk); cuMemFree(Bk);
        cuMemFree(Wv); cuMemFree(Bv); cuMemFree(Wo); cuMemFree(Bo);
        /* LN2 */
        { CUdeviceptr W = WT("layer_norm2.weight"), B = WT("layer_norm2.bias");
          k_ln_last(d_queries, W, B, N_TOK, HD, LN_EPS); cuMemFree(W); cuMemFree(B); }

        /* MLP: proj_in + relu + proj_out */
        { CUdeviceptr Wi = WT("mlp.proj_in.weight"), Bi = WT("mlp.proj_in.bias");
          CUdeviceptr Wmo = WT("mlp.proj_out.weight"), Bmo = WT("mlp.proj_out.bias");
          k_linear(d_mlpH, d_queries, Wi, Bi, N_TOK, HD, MD);
          k_relu(d_mlpH, N_TOK*MD);
          k_linear(d_qbuf, d_mlpH, Wmo, Bmo, N_TOK, MD, HD);
          k_add_vec(d_queries, d_qbuf, N_TOK*HD);
          cuMemFree(Wi); cuMemFree(Bi); cuMemFree(Wmo); cuMemFree(Bmo); }
        /* LN3 */
        { CUdeviceptr W = WT("layer_norm3.weight"), B = WT("layer_norm3.bias");
          k_ln_last(d_queries, W, B, N_TOK, HD, LN_EPS); cuMemFree(W); cuMemFree(B); }

        /* Cross-attn image → token */
        Wq = WT("cross_attn_image_to_token.q_proj.weight"); Bq = WT("cross_attn_image_to_token.q_proj.bias");
        Wk = WT("cross_attn_image_to_token.k_proj.weight"); Bk = WT("cross_attn_image_to_token.k_proj.bias");
        Wv = WT("cross_attn_image_to_token.v_proj.weight"); Bv = WT("cross_attn_image_to_token.v_proj.bias");
        Wo = WT("cross_attn_image_to_token.o_proj.weight"); Bo = WT("cross_attn_image_to_token.o_proj.bias");
        k_axpy(d_qq_tok, d_queries, d_point_pe, N_TOK*HD);
        k_axpy(d_qq_img, d_keys,    d_pe_k,     N_IM*HD);
        run_attention(d_abuf, d_qq_img, d_qq_tok, d_queries,
                      N_IM, N_TOK, HD, CA_INT, NH, Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo);
        k_add_vec(d_keys, d_abuf, N_IM*HD);
        cuMemFree(Wq); cuMemFree(Bq); cuMemFree(Wk); cuMemFree(Bk);
        cuMemFree(Wv); cuMemFree(Bv); cuMemFree(Wo); cuMemFree(Bo);
        /* LN4 on keys */
        { CUdeviceptr W = WT("layer_norm4.weight"), B = WT("layer_norm4.bias");
          k_ln_last(d_keys, W, B, N_IM, HD, LN_EPS); cuMemFree(W); cuMemFree(B); }

        #undef WT
    }

    /* ---- Final attn ---- */
    {
        char n[256]; size_t dum;
        #define WT(s) (snprintf(n,sizeof(n),"mask_decoder.transformer.final_attn_token_to_image.%s", s), TD(n,&dum))
        CUdeviceptr Wq = WT("q_proj.weight"), Bq = WT("q_proj.bias");
        CUdeviceptr Wk = WT("k_proj.weight"), Bk = WT("k_proj.bias");
        CUdeviceptr Wv = WT("v_proj.weight"), Bv = WT("v_proj.bias");
        CUdeviceptr Wo = WT("o_proj.weight"), Bo = WT("o_proj.bias");
        k_axpy(d_qq_tok, d_queries, d_point_pe, N_TOK*HD);
        k_axpy(d_qq_img, d_keys,    d_pe_k,     N_IM*HD);
        run_attention(d_qbuf, d_qq_tok, d_qq_img, d_keys,
                      N_TOK, N_IM, HD, CA_INT, NH, Wq, Bq, Wk, Bk, Wv, Bv, Wo, Bo);
        k_add_vec(d_queries, d_qbuf, N_TOK*HD);
        cuMemFree(Wq); cuMemFree(Bq); cuMemFree(Wk); cuMemFree(Bk);
        cuMemFree(Wv); cuMemFree(Bv); cuMemFree(Wo); cuMemFree(Bo);
        CUdeviceptr lnw = TD("mask_decoder.transformer.layer_norm_final_attn.weight", &dum);
        CUdeviceptr lnb = TD("mask_decoder.transformer.layer_norm_final_attn.bias",   &dum);
        k_ln_last(d_queries, lnw, lnb, N_TOK, HD, LN_EPS);
        cuMemFree(lnw); cuMemFree(lnb);
        #undef WT
    }

    /* ---- IoU head ---- */
    size_t dum;
    CUdeviceptr iou_Wi = TD("mask_decoder.iou_prediction_head.proj_in.weight", &dum);
    CUdeviceptr iou_Bi = TD("mask_decoder.iou_prediction_head.proj_in.bias",   &dum);
    CUdeviceptr iou_Wm = TD("mask_decoder.iou_prediction_head.layers.0.weight", &dum);
    CUdeviceptr iou_Bm = TD("mask_decoder.iou_prediction_head.layers.0.bias",   &dum);
    CUdeviceptr iou_Wo = TD("mask_decoder.iou_prediction_head.proj_out.weight", &dum);
    CUdeviceptr iou_Bo = TD("mask_decoder.iou_prediction_head.proj_out.bias",   &dum);
    CUdeviceptr d_iou_h = dalloc(HD);
    CUdeviceptr d_iou_pred = dalloc(NMASK);
    /* iou token at index 1 — slice from queries */
    CUdeviceptr d_iou_tok = d_queries + (size_t)1*HD*4;
    k_linear(d_iou_h, d_iou_tok, iou_Wi, iou_Bi, 1, HD, HD); k_relu(d_iou_h, HD);
    CUdeviceptr d_iou_h2 = dalloc(HD);
    k_linear(d_iou_h2, d_iou_h, iou_Wm, iou_Bm, 1, HD, HD); k_relu(d_iou_h2, HD);
    k_linear(d_iou_pred, d_iou_h2, iou_Wo, iou_Bo, 1, HD, NMASK); k_sig(d_iou_pred, NMASK);
    cuMemFree(d_iou_h); cuMemFree(d_iou_h2);

    /* ---- Upscale ---- */
    /* keys (N_IM, HD) → BCHW (HD, 64, 64) by inverse transpose */
    CUdeviceptr d_key_chw = dalloc(N_SPA);
    /* Simple: use host round-trip (small). */
    {
        float *tmp = (float *)malloc(N_SPA*4);
        cuMemcpyDtoH(tmp, d_keys, N_SPA*4);
        float *tmp2 = (float *)malloc(N_SPA*4);
        for (int p = 0; p < N_IM; p++)
            for (int c = 0; c < HD; c++)
                tmp2[(size_t)c*N_IM + p] = tmp[(size_t)p*HD + c];
        cuMemcpyHtoD(d_key_chw, tmp2, N_SPA*4);
        free(tmp); free(tmp2);
    }

    CUdeviceptr Wc1 = TD("mask_decoder.upscale_conv1.weight", &dum);
    CUdeviceptr Bc1 = TD("mask_decoder.upscale_conv1.bias",   &dum);
    CUdeviceptr d_up1 = dalloc((size_t)64*128*128);
    k_conv_t2x(d_up1, d_key_chw, Wc1, Bc1, 256, 64, 64, 64);
    k_add_vec(d_up1, d_hr1, 64*128*128);
    CUdeviceptr lnw = TD("mask_decoder.upscale_layer_norm.weight", &dum);
    CUdeviceptr lnb = TD("mask_decoder.upscale_layer_norm.bias",   &dum);
    k_ln_chw(d_up1, lnw, lnb, 1, 64, 128*128);
    k_gelu(d_up1, 64*128*128);
    cuMemFree(lnw); cuMemFree(lnb); cuMemFree(Wc1); cuMemFree(Bc1);

    CUdeviceptr Wc2 = TD("mask_decoder.upscale_conv2.weight", &dum);
    CUdeviceptr Bc2 = TD("mask_decoder.upscale_conv2.bias",   &dum);
    CUdeviceptr d_up2 = dalloc((size_t)32*256*256);
    k_conv_t2x(d_up2, d_up1, Wc2, Bc2, 64, 32, 128, 128);
    k_add_vec(d_up2, d_hr0, 32*256*256);
    k_gelu(d_up2, 32*256*256);
    cuMemFree(Wc2); cuMemFree(Bc2); cuMemFree(d_key_chw); cuMemFree(d_up1);

    /* ---- Hypernetwork MLPs on mask_tokens[2..5] ---- */
    CUdeviceptr d_hyper = dalloc((size_t)NMASK*32);
    CUdeviceptr d_ht1 = dalloc(HD);
    CUdeviceptr d_ht2 = dalloc(HD);
    for (int m = 0; m < NMASK; m++) {
        char b[128]; snprintf(b, sizeof(b), "mask_decoder.output_hypernetworks_mlps.%d", m);
        char n[256];
        #define WT(s) (snprintf(n,sizeof(n),"%s.%s", b, s), TD(n,&dum))
        CUdeviceptr Wi = WT("proj_in.weight"), Bi = WT("proj_in.bias");
        CUdeviceptr Wm = WT("layers.0.weight"), Bm = WT("layers.0.bias");
        CUdeviceptr Wo = WT("proj_out.weight"), Bo = WT("proj_out.bias");
        CUdeviceptr d_src = d_queries + (size_t)(2+m)*HD*4;
        k_linear(d_ht1, d_src, Wi, Bi, 1, HD, HD); k_relu(d_ht1, HD);
        k_linear(d_ht2, d_ht1, Wm, Bm, 1, HD, HD); k_relu(d_ht2, HD);
        CUdeviceptr d_out = d_hyper + (size_t)m*32*4;
        k_linear(d_out, d_ht2, Wo, Bo, 1, HD, 32);
        cuMemFree(Wi); cuMemFree(Bi); cuMemFree(Wm); cuMemFree(Bm); cuMemFree(Wo); cuMemFree(Bo);
        #undef WT
    }
    cuMemFree(d_ht1); cuMemFree(d_ht2);

    /* ---- masks = hyper @ upscaled ---- */
    /* hyper: (NMASK, 32). upscaled: (32, 256*256) (CHW). masks: (NMASK, 256*256). */
    int HW = 256*256;
    CUdeviceptr d_masks = dalloc((size_t)NMASK*HW);
    /* Treat as linear: y[m, p] = sum_c hyper[m,c] * up2[c,p]. This is N=NMASK, din=32, dout=HW.
     * linear expects W shape (dout, din); our "W" is up2 with shape (32, HW) i.e. (din, dout).
     * Use transposed matmul by swapping: y = hyper @ W where W is (32, HW). We'll write an inline
     * variant: iterate (m,p) and accumulate. Reuse linear if we pre-transpose, but easier: one-off kernel.
     * For now — use linear where dout=HW, din=32, requires W shape (HW, 32). We have W^T shape (32, HW).
     * Workaround: use axpy-style approach via looping. Simpler: just cuMemcpyDtoH + host compute this one op. */
    {
        int M = NMASK, C = 32, P = HW;
        void *a[] = { &d_masks, &d_hyper, &d_up2, &M, &C, &P };
        cuLaunchKernel(fn_matmul_cb, M, (P+255)/256, 1, 256, 1, 1, 0, 0, a, 0);
    }
    cuMemFree(d_up2); cuMemFree(d_hyper);

    /* ---- Compare ---- */
    snprintf(path, sizeof(path), "%s/md_iou_scores.npy", refdir);
    float *ref_iou = read_npy_f32(path, d, &nd);
    snprintf(path, sizeof(path), "%s/md_low_res_masks.npy", refdir);
    float *ref_masks = read_npy_f32(path, d, &nd);

    /* iou: slice [1..] (NMASK-1=3) */
    float h_iou[NMASK]; cuMemcpyDtoH(h_iou, d_iou_pred, NMASK*4);
    {
        double mad=0; float mxd=0;
        for (int i = 0; i < 3; i++) { float x = fabsf(h_iou[1+i]-ref_iou[i]); if (x>mxd) mxd=x; mad+=x; }
        fprintf(stderr, "  %-22s: max_abs=%.6g mean_abs=%.6g\n", "md_iou_scores", mxd, mad/3);
    }
    /* masks: slice [1..] (3 masks) */
    diff_dev_host("md_low_res_masks", d_masks + (size_t)1*HW*4, ref_masks, (size_t)3*HW);

    free(ref_iou); free(ref_masks);
    cuCtxSynchronize();
    cuCtxDestroy(ctx); safetensors_close(ST);
    return 0;
}
