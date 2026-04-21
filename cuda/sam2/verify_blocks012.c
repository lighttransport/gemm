/*
 * verify_blocks012.c — Verify CUDA Hiera blocks 0, 1, 2 against PyTorch trace.
 *
 * Block 0: stage 0, ws=8, heads=1, dim 96→96, no pool
 * Block 1: stage 0→1 transition, ws=8, heads=2, dim 96→192, Q-pool 2×2 (+skip-proj+pool)
 * Block 2: stage 1, ws=4, heads=2, dim 192→192, no pool
 *
 * Feed block0_input.npy, compare block{0,1,2}_output.npy.
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

static float *read_npy_f32(const char *path, int dims[5], int *ndims) {
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
    while (p < q && n < 5) {
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

/* ------------------- kernels ------------------- */

static const char *kern_src =
"extern \"C\" {\n"
"__global__ void ln_last(float *y, const float *x, const float *w, const float *b, int n_tok, int dim, float eps) {\n"
"  int t = blockIdx.x; int d = threadIdx.x;\n"
"  if (t >= n_tok || d >= dim) return;\n"
"  const float *xt = x + (size_t)t * dim;\n"
"  float mean = 0.f; for (int i = 0; i < dim; i++) mean += xt[i]; mean /= dim;\n"
"  float var  = 0.f; for (int i = 0; i < dim; i++) { float v = xt[i]-mean; var += v*v; } var /= dim;\n"
"  y[(size_t)t*dim + d] = ((xt[d]-mean) * rsqrtf(var+eps)) * w[d] + b[d];\n"
"}\n"
"__global__ void linear2d(float *y, const float *x, const float *w, const float *b, int n_tok, int din, int dout) {\n"
"  int o = blockIdx.y * blockDim.x + threadIdx.x; int t = blockIdx.x;\n"
"  if (t >= n_tok || o >= dout) return;\n"
"  float acc = b ? b[o] : 0.f;\n"
"  const float *xt = x + (size_t)t*din;\n"
"  const float *wo = w + (size_t)o*din;\n"
"  for (int i = 0; i < din; i++) acc += wo[i] * xt[i];\n"
"  y[(size_t)t*dout + o] = acc;\n"
"}\n"
"__global__ void gelu_f(float *x, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (i < n) { float v = x[i]; x[i] = 0.5f*v*(1.f+erff(v*0.7071067811865475f)); }\n"
"}\n"
"__global__ void add_inplace(float *x, const float *y, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (i < n) x[i] += y[i];\n"
"}\n"
"__global__ void window_partition(float *y, const float *x, int H, int W, int C, int ws) {\n"
"  int t = blockIdx.x; int c = blockIdx.y * blockDim.x + threadIdx.x;\n"
"  if (c >= C) return;\n"
"  int nww = W / ws;\n"
"  int tok_per_win = ws*ws;\n"
"  int nwin = (H/ws) * nww;\n"
"  if (t >= nwin * tok_per_win) return;\n"
"  int w_idx = t / tok_per_win;\n"
"  int i = t % tok_per_win;\n"
"  int wy = w_idx / nww, wx = w_idx % nww;\n"
"  int yy = wy*ws + i/ws, xx = wx*ws + i%ws;\n"
"  y[(size_t)t*C + c] = x[((size_t)yy*W + xx)*C + c];\n"
"}\n"
"__global__ void window_unpartition(float *y, const float *x, int nwy, int nwx, int ws, int C) {\n"
"  int t = blockIdx.x; int c = blockIdx.y * blockDim.x + threadIdx.x;\n"
"  if (c >= C) return;\n"
"  int tok_per_win = ws*ws;\n"
"  int nwin = nwy*nwx;\n"
"  if (t >= nwin*tok_per_win) return;\n"
"  int w_idx = t / tok_per_win;\n"
"  int i = t % tok_per_win;\n"
"  int wy = w_idx / nwx, wx = w_idx % nwx;\n"
"  int yy = wy*ws + i/ws, xx = wx*ws + i%ws;\n"
"  int W = nwx*ws;\n"
"  y[((size_t)yy*W + xx)*C + c] = x[(size_t)t*C + c];\n"
"}\n"
"__global__ void maxpool2d_k2s2(float *y, const float *x, int B, int H, int W, int C) {\n"
"  int Ho = H/2, Wo = W/2;\n"
"  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int total = B*Ho*Wo*C;\n"
"  if (idx >= total) return;\n"
"  int c  = idx % C;\n"
"  int wo = (idx / C) % Wo;\n"
"  int ho = (idx / (C*Wo)) % Ho;\n"
"  int b  = idx / (C*Wo*Ho);\n"
"  const float *xb = x + ((size_t)b*H*W)*C;\n"
"  float m = -1e30f;\n"
"  for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {\n"
"    int yy = ho*2+dy, xx = wo*2+dx;\n"
"    float v = xb[((size_t)yy*W + xx)*C + c];\n"
"    if (v > m) m = v;\n"
"  }\n"
"  y[idx] = m;\n"
"}\n"
"__global__ void pool_q_from_qkv(float *qp, const float *qkv, int nwin, int ws, int D) {\n"
"  int wsout = ws/2;\n"
"  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int total = nwin * wsout*wsout * D;\n"
"  if (idx >= total) return;\n"
"  int d = idx % D;\n"
"  int p = (idx / D) % (wsout*wsout);\n"
"  int w = idx / (D*wsout*wsout);\n"
"  int py = p / wsout, px = p % wsout;\n"
"  const float *base = qkv + ((size_t)w*(ws*ws))*(3*D);\n"
"  float m = -1e30f;\n"
"  for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {\n"
"    int yy = py*2+dy, xx = px*2+dx;\n"
"    int tok = yy*ws + xx;\n"
"    float v = base[(size_t)tok*(3*D) + d];\n"
"    if (v > m) m = v;\n"
"  }\n"
"  qp[idx] = m;\n"
"}\n"
"__global__ void mh_window_attn(float *out, const float *Q, const float *qkv,\n"
"                                int nwin, int Nq, int Nk, int nh, int hd, float scale) {\n"
"  int w  = blockIdx.x;\n"
"  int qi = blockIdx.y;\n"
"  int h  = threadIdx.y;\n"
"  int d  = threadIdx.x;\n"
"  if (w >= nwin || qi >= Nq || h >= nh || d >= hd) return;\n"
"  int D = nh*hd;\n"
"  extern __shared__ float sh[];\n"
"  float *s_scores = sh;\n"
"  float *s_out    = sh + nh*Nk;\n"
"  const float *Qw = Q + ((size_t)w*Nq + qi)*D + h*hd;\n"
"  const float *KVw = qkv + (size_t)w*Nk*(3*D);\n"
"  for (int kj = d; kj < Nk; kj += hd) {\n"
"    const float *K = KVw + (size_t)kj*(3*D) + D + h*hd;\n"
"    float s = 0.f;\n"
"    for (int i = 0; i < hd; i++) s += Qw[i] * K[i];\n"
"    s_scores[h*Nk + kj] = s * scale;\n"
"  }\n"
"  __syncthreads();\n"
"  if (d == 0) {\n"
"    float m = -1e30f;\n"
"    for (int kj = 0; kj < Nk; kj++) { float v = s_scores[h*Nk+kj]; if (v > m) m = v; }\n"
"    float z = 0.f;\n"
"    for (int kj = 0; kj < Nk; kj++) { float e = expf(s_scores[h*Nk+kj]-m); s_scores[h*Nk+kj] = e; z += e; }\n"
"    float invz = 1.f/z;\n"
"    for (int kj = 0; kj < Nk; kj++) s_scores[h*Nk+kj] *= invz;\n"
"  }\n"
"  __syncthreads();\n"
"  float acc = 0.f;\n"
"  for (int kj = 0; kj < Nk; kj++) {\n"
"    const float *V = KVw + (size_t)kj*(3*D) + 2*D + h*hd;\n"
"    acc += s_scores[h*Nk+kj] * V[d];\n"
"  }\n"
"  s_out[h*hd + d] = acc;\n"
"  __syncthreads();\n"
"  out[((size_t)w*Nq + qi)*D + h*hd + d] = s_out[h*hd + d];\n"
"}\n"
"}\n";

/* ------------------- host helpers ------------------- */

static int load(st_context *st, const char *name, float **out) {
    int i = safetensors_find(st, name);
    if (i < 0) { fprintf(stderr, "missing %s\n", name); return -1; }
    if (strcmp(safetensors_dtype(st, i), "F32")) { fprintf(stderr, "bad dtype %s\n", name); return -1; }
    *out = (float *)safetensors_data(st, i);
    return 0;
}

static void diff(const char *name, const float *a, const float *b, size_t n) {
    double mad = 0.0; float mxd = 0.f;
    for (size_t i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d > mxd) mxd = d; mad += d; }
    mad /= (double)n;
    fprintf(stderr, "  %s: max_abs=%.6g mean_abs=%.6g\n", name, mxd, mad);
}

static CUdeviceptr upf(const float *h, size_t n) {
    CUdeviceptr d; cuMemAlloc(&d, n*4); cuMemcpyHtoD(d, h, n*4); return d;
}

/* Forward one Hiera block on device. Allocates and frees scratch internally.
 *
 * Inputs:
 *   d_x: input (1, H, W, dim), device
 * Outputs:
 *   d_out: output (1, H_out, W_out, dim_out), device-allocated by caller
 *
 * Weights passed as device pointers (caller uploads).
 */
typedef struct {
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr qkv_w, qkv_b;   /* (3*dim_out, dim), (3*dim_out) */
    CUdeviceptr proj_w, proj_b; /* (dim_out, dim_out), (dim_out) */
    CUdeviceptr skip_w, skip_b; /* (dim_out, dim), (dim_out) — only if dim != dim_out */
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr fc1_w, fc1_b;   /* (mlp, dim_out) */
    CUdeviceptr fc2_w, fc2_b;   /* (dim_out, mlp) */
} block_wts;

typedef struct {
    CUfunction ln, lin, gelu, add, wpart, wunpart, mpool, qpool, attn;
} block_fns;

static int forward_block(const block_fns *fn, const block_wts *W,
                         CUdeviceptr d_x, int H, int W_, int dim,
                         CUdeviceptr d_out, int dim_out, int nh,
                         int ws, int pool /* 1 if q_stride=(2,2), else 0 */) {
    int hd = dim_out / nh;
    float eps = 1e-6f;
    int H_out = pool ? H/2 : H;
    int W_out = pool ? W_/2 : W_;
    int n_tok_in = H * W_;
    int n_tok_out = H_out * W_out;
    int mlp = dim_out * 4;

    CUdeviceptr d_ln1, d_skip_pre, d_residual, d_part, d_qkv, d_qpool, d_attn, d_proj, d_unpart, d_ln2, d_mlp, d_fc2;
    cuMemAlloc(&d_ln1, (size_t)n_tok_in*dim*4);
    cuMemAlloc(&d_residual, (size_t)n_tok_out*dim_out*4);
    cuMemAlloc(&d_part, (size_t)n_tok_in*dim*4);
    cuMemAlloc(&d_qkv,  (size_t)n_tok_in*3*dim_out*4);
    cuMemAlloc(&d_attn, (size_t)n_tok_out*dim_out*4);
    cuMemAlloc(&d_proj, (size_t)n_tok_out*dim_out*4);
    cuMemAlloc(&d_unpart, (size_t)n_tok_out*dim_out*4);
    cuMemAlloc(&d_ln2, (size_t)n_tok_out*dim_out*4);
    cuMemAlloc(&d_mlp, (size_t)n_tok_out*mlp*4);
    cuMemAlloc(&d_fc2, (size_t)n_tok_out*dim_out*4);
    if (dim != dim_out) cuMemAlloc(&d_skip_pre, (size_t)n_tok_in*dim_out*4); else d_skip_pre = 0;

    /* ln1 */
    void *a_ln1[] = { &d_ln1, &d_x, &W->ln1_w, &W->ln1_b, &n_tok_in, &dim, &eps };
    cuLaunchKernel(fn->ln, n_tok_in, 1, 1, dim, 1, 1, 0, 0, a_ln1, 0);

    /* residual path: if dim!=dim_out, proj then pool; else copy */
    if (dim != dim_out) {
        int one = 1;
        void *a_p[] = { &d_skip_pre, &d_ln1, &W->skip_w, &W->skip_b, &n_tok_in, &dim, &dim_out };
        cuLaunchKernel(fn->lin, n_tok_in, (unsigned)((dim_out+255)/256), 1, 256, 1, 1, 0, 0, a_p, 0);
        if (pool) {
            void *a_m[] = { &d_residual, &d_skip_pre, &one, &H, &W_, &dim_out };
            int total = n_tok_out*dim_out;
            cuLaunchKernel(fn->mpool, (unsigned)((total+255)/256),1,1,256,1,1,0,0,a_m,0);
        } else {
            cuMemcpyDtoD(d_residual, d_skip_pre, (size_t)n_tok_in*dim_out*4);
        }
    } else {
        cuMemcpyDtoD(d_residual, d_x, (size_t)n_tok_out*dim_out*4);
    }

    /* window_partition ln1 → d_part */
    int nwy = H / ws, nwx = W_ / ws;
    int nwin = nwy * nwx;
    int Nk = ws * ws;
    {
        void *a_wp[] = { &d_part, &d_ln1, &H, &W_, &dim, &ws };
        cuLaunchKernel(fn->wpart, nwin*Nk, (unsigned)((dim+255)/256), 1, 256, 1, 1, 0, 0, a_wp, 0);
    }

    /* qkv = linear(d_part, qkv_w, qkv_b) with din=dim, dout=3*dim_out */
    {
        int din = dim, dout = 3*dim_out;
        int n = nwin * Nk;
        void *a_q[] = { &d_qkv, &d_part, &W->qkv_w, &W->qkv_b, &n, &din, &dout };
        cuLaunchKernel(fn->lin, n, (unsigned)((dout+255)/256), 1, 256, 1, 1, 0, 0, a_q, 0);
    }

    /* Q-pool (if needed) → d_qpool (nwin, (ws/2)^2, dim_out). Else use Q slice of qkv. */
    int Nq = pool ? (ws/2)*(ws/2) : Nk;
    int ws_out = pool ? ws/2 : ws;
    CUdeviceptr d_Q;
    if (pool) {
        cuMemAlloc(&d_qpool, (size_t)nwin*Nq*dim_out*4);
        int total = nwin*Nq*dim_out;
        void *a_pq[] = { &d_qpool, &d_qkv, &nwin, &ws, &dim_out };
        cuLaunchKernel(fn->qpool, (unsigned)((total+255)/256),1,1,256,1,1,0,0,a_pq,0);
        d_Q = d_qpool;
    } else {
        /* Extract Q from qkv directly is awkward; since Nq==Nk we can just pass qkv ptr:
           mh_window_attn reads Q at offset 0 for each token. But the kernel expects Q
           packed as (nwin*Nq*D) contiguously. So we must pack Q. Easiest: always use
           pool path with kernel_size=1 — or write a small pack kernel. We'll reuse
           pool_q with ws=ws and explicit NO_POOL flag. Instead, just allocate and pack
           via a trivial kernel: pool_q_from_qkv with... Different shape. Simpler:
           when pool==0, run pool_q_from_qkv on ws_effective=ws but with 1x1 kernel.
           Actually we need a different extract. Write inline small kernel? Alternative:
           keep stride=2 invariant, require pool==1. For now handle pool==0 via host pack. */
        cuMemAlloc(&d_qpool, (size_t)nwin*Nq*dim_out*4);
        /* Host-side pack: DtoH qkv, pack Q, HtoD. */
        size_t nb_qkv = (size_t)nwin*Nk*3*dim_out*4;
        float *hqkv = (float *)malloc(nb_qkv);
        cuMemcpyDtoH(hqkv, d_qkv, nb_qkv);
        float *hq = (float *)malloc((size_t)nwin*Nk*dim_out*4);
        for (int w = 0; w < nwin; w++)
          for (int k = 0; k < Nk; k++)
            memcpy(hq + ((size_t)w*Nk + k)*dim_out,
                   hqkv + ((size_t)w*Nk + k)*3*dim_out,
                   dim_out*4);
        cuMemcpyHtoD(d_qpool, hq, (size_t)nwin*Nk*dim_out*4);
        free(hq); free(hqkv);
        d_Q = d_qpool;
    }

    /* Attention */
    float scale = 1.f / sqrtf((float)hd);
    {
        size_t smem = (size_t)(nh*Nk + nh*hd) * 4;
        void *a_at[] = { &d_attn, &d_Q, &d_qkv, &nwin, &Nq, &Nk, &nh, &hd, &scale };
        cuLaunchKernel(fn->attn, nwin, Nq, 1, hd, nh, 1, smem, 0, a_at, 0);
    }

    /* proj linear */
    {
        int din = dim_out, dout = dim_out;
        int n = nwin * Nq;
        void *a_p[] = { &d_proj, &d_attn, &W->proj_w, &W->proj_b, &n, &din, &dout };
        cuLaunchKernel(fn->lin, n, (unsigned)((dout+255)/256), 1, 256, 1, 1, 0, 0, a_p, 0);
    }

    /* window_unpartition */
    int nwy_out = H_out / ws_out, nwx_out = W_out / ws_out;
    {
        void *a_wu[] = { &d_unpart, &d_proj, &nwy_out, &nwx_out, &ws_out, &dim_out };
        cuLaunchKernel(fn->wunpart, nwy_out*nwx_out*ws_out*ws_out, (unsigned)((dim_out+255)/256), 1, 256, 1, 1, 0, 0, a_wu, 0);
    }

    /* residual + unpart */
    {
        int n = n_tok_out*dim_out;
        void *a[] = { &d_residual, &d_unpart, &n };
        cuLaunchKernel(fn->add, (unsigned)((n+255)/256),1,1,256,1,1,0,0,a,0);
    }

    /* ln2 */
    {
        void *a[] = { &d_ln2, &d_residual, &W->ln2_w, &W->ln2_b, &n_tok_out, &dim_out, &eps };
        cuLaunchKernel(fn->ln, n_tok_out, 1, 1, dim_out, 1, 1, 0, 0, a, 0);
    }

    /* fc1 + gelu */
    {
        int din = dim_out, dout = mlp;
        void *a[] = { &d_mlp, &d_ln2, &W->fc1_w, &W->fc1_b, &n_tok_out, &din, &dout };
        cuLaunchKernel(fn->lin, n_tok_out, (unsigned)((dout+255)/256), 1, 256, 1, 1, 0, 0, a, 0);
        int n = n_tok_out*mlp;
        void *ag[] = { &d_mlp, &n };
        cuLaunchKernel(fn->gelu, (unsigned)((n+255)/256),1,1,256,1,1,0,0,ag,0);
    }

    /* fc2 */
    {
        int din = mlp, dout = dim_out;
        void *a[] = { &d_fc2, &d_mlp, &W->fc2_w, &W->fc2_b, &n_tok_out, &din, &dout };
        cuLaunchKernel(fn->lin, n_tok_out, (unsigned)((dout+255)/256), 1, 256, 1, 1, 0, 0, a, 0);
    }

    /* out = residual + fc2 */
    cuMemcpyDtoD(d_out, d_residual, (size_t)n_tok_out*dim_out*4);
    {
        int n = n_tok_out*dim_out;
        void *a[] = { &d_out, &d_fc2, &n };
        cuLaunchKernel(fn->add, (unsigned)((n+255)/256),1,1,256,1,1,0,0,a,0);
    }

    cuCtxSynchronize();
    cuMemFree(d_ln1); cuMemFree(d_residual); cuMemFree(d_part); cuMemFree(d_qkv);
    cuMemFree(d_attn); cuMemFree(d_proj); cuMemFree(d_unpart); cuMemFree(d_ln2);
    cuMemFree(d_mlp); cuMemFree(d_fc2); cuMemFree(d_qpool);
    if (d_skip_pre) cuMemFree(d_skip_pre);
    return 0;
}

static int load_block_wts(st_context *st, int idx, int has_proj, block_wts *W,
                           int dim, int dim_out, int nh /* unused */) {
    (void)nh;
    char k[256];
    float *h;
    #define LOAD(SUF, OUT, N) do { \
        snprintf(k, sizeof(k), "vision_encoder.backbone.blocks.%d.%s", idx, SUF); \
        if (load(st, k, &h)) return -1; \
        OUT = upf(h, (size_t)(N)); \
    } while(0)
    LOAD("layer_norm1.weight", W->ln1_w, dim);
    LOAD("layer_norm1.bias",   W->ln1_b, dim);
    LOAD("attn.qkv.weight",    W->qkv_w, 3*dim_out*dim);
    LOAD("attn.qkv.bias",      W->qkv_b, 3*dim_out);
    LOAD("attn.proj.weight",   W->proj_w, dim_out*dim_out);
    LOAD("attn.proj.bias",     W->proj_b, dim_out);
    LOAD("layer_norm2.weight", W->ln2_w, dim_out);
    LOAD("layer_norm2.bias",   W->ln2_b, dim_out);
    LOAD("mlp.proj_in.weight", W->fc1_w, (dim_out*4)*dim_out);
    LOAD("mlp.proj_in.bias",   W->fc1_b, dim_out*4);
    LOAD("mlp.proj_out.weight",W->fc2_w, dim_out*(dim_out*4));
    LOAD("mlp.proj_out.bias",  W->fc2_b, dim_out);
    if (has_proj) {
        LOAD("proj.weight", W->skip_w, dim_out*dim);
        LOAD("proj.bias",   W->skip_b, dim_out);
    } else { W->skip_w = 0; W->skip_b = 0; }
    #undef LOAD
    return 0;
}

static void free_block_wts(block_wts *W) {
    CUdeviceptr *ps[] = { &W->ln1_w,&W->ln1_b,&W->qkv_w,&W->qkv_b,&W->proj_w,&W->proj_b,
                          &W->ln2_w,&W->ln2_b,&W->fc1_w,&W->fc1_b,&W->fc2_w,&W->fc2_b,
                          &W->skip_w,&W->skip_b };
    for (int i = 0; i < 14; i++) if (*ps[i]) cuMemFree(*ps[i]);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <sam2 model.safetensors> <refdir>\n", argv[0]);
        return 1;
    }
    const char *ckpt = argv[1]; const char *refdir = argv[2];
    char path[1024];

    st_context *st = safetensors_open(ckpt);
    if (!st) { fprintf(stderr, "safetensors_open failed\n"); return 3; }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 5;
    if (cuInit(0) != CUDA_SUCCESS) return 5;
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    if (cu_compile_kernels(&mod, dev, kern_src, "sam2_b012", 1, "sam2") < 0) return 6;
    block_fns fn;
    cuModuleGetFunction(&fn.ln, mod, "ln_last");
    cuModuleGetFunction(&fn.lin, mod, "linear2d");
    cuModuleGetFunction(&fn.gelu, mod, "gelu_f");
    cuModuleGetFunction(&fn.add, mod, "add_inplace");
    cuModuleGetFunction(&fn.wpart, mod, "window_partition");
    cuModuleGetFunction(&fn.wunpart, mod, "window_unpartition");
    cuModuleGetFunction(&fn.mpool, mod, "maxpool2d_k2s2");
    cuModuleGetFunction(&fn.qpool, mod, "pool_q_from_qkv");
    cuModuleGetFunction(&fn.attn, mod, "mh_window_attn");

    block_wts W0 = {0}, W1 = {0}, W2 = {0};
    if (load_block_wts(st, 0, 0, &W0, 96, 96, 1)) return 4;
    if (load_block_wts(st, 1, 1, &W1, 96, 192, 2)) return 4;
    if (load_block_wts(st, 2, 0, &W2, 192, 192, 2)) return 4;

    /* Load block0 input */
    snprintf(path, sizeof(path), "%s/block0_input.npy", refdir);
    int d[5], nd; float *xin = read_npy_f32(path, d, &nd);
    if (!xin || d[0]!=1 || d[1]!=256 || d[2]!=256 || d[3]!=96) { fprintf(stderr,"bad input\n"); return 2; }

    CUdeviceptr d_x, d_y1, d_y2;
    cuMemAlloc(&d_x, (size_t)256*256*96*4);
    cuMemcpyHtoD(d_x, xin, (size_t)256*256*96*4);
    cuMemAlloc(&d_y1, (size_t)128*128*192*4);
    cuMemAlloc(&d_y2, (size_t)128*128*192*4);

    /* ---- Block 0 ---- */
    fprintf(stderr, "block 0:\n");
    CUdeviceptr d_b0out; cuMemAlloc(&d_b0out, (size_t)256*256*96*4);
    forward_block(&fn, &W0, d_x, 256, 256, 96, d_b0out, 96, 1, 8, 0);
    float *ref; int rd[5], rnd;
    snprintf(path,sizeof(path),"%s/block0_output.npy",refdir);
    ref = read_npy_f32(path, rd, &rnd);
    float *tmp = (float *)malloc((size_t)256*256*96*4);
    cuMemcpyDtoH(tmp, d_b0out, (size_t)256*256*96*4);
    diff("block0", tmp, ref, (size_t)256*256*96);
    free(ref); free(tmp);

    /* ---- Block 1 (pool) ---- */
    fprintf(stderr, "block 1:\n");
    forward_block(&fn, &W1, d_b0out, 256, 256, 96, d_y1, 192, 2, 8, 1);
    snprintf(path,sizeof(path),"%s/block1_output.npy",refdir);
    ref = read_npy_f32(path, rd, &rnd);
    tmp = (float *)malloc((size_t)128*128*192*4);
    cuMemcpyDtoH(tmp, d_y1, (size_t)128*128*192*4);
    diff("block1", tmp, ref, (size_t)128*128*192);
    free(ref); free(tmp);

    /* ---- Block 2 ---- */
    fprintf(stderr, "block 2:\n");
    forward_block(&fn, &W2, d_y1, 128, 128, 192, d_y2, 192, 2, 4, 0);
    snprintf(path,sizeof(path),"%s/block2_output.npy",refdir);
    ref = read_npy_f32(path, rd, &rnd);
    tmp = (float *)malloc((size_t)128*128*192*4);
    cuMemcpyDtoH(tmp, d_y2, (size_t)128*128*192*4);
    diff("block2", tmp, ref, (size_t)128*128*192);
    free(ref); free(tmp);

    free(xin);
    cuMemFree(d_x); cuMemFree(d_y1); cuMemFree(d_y2); cuMemFree(d_b0out);
    free_block_wts(&W0); free_block_wts(&W1); free_block_wts(&W2);
    cuModuleUnload(mod); cuCtxDestroy(ctx); safetensors_close(st);
    return 0;
}
