/*
 * verify_backbone.c — Verify every Hiera block (0..11) independently against
 * the PyTorch reference trace. Each block is fed block{N}_input.npy and the
 * output is compared to block{N}_output.npy — so failures are localised.
 *
 * Handles:
 *   - Multi-head attention (1, 2, 4, 8 heads)
 *   - Q-pool (stride 2x2) + skip-proj+pool for stage transitions (blocks 1, 3, 10)
 *   - Padded window partition (ws=14 on 64^2, ws=7 on 32^2)
 *   - Global attention (blocks 5, 7, 9; ws=0)
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

static const char *kern_src =
"extern \"C\" {\n"
"__global__ void ln_last(float *y, const float *x, const float *w, const float *b, int n_tok, int dim, float eps) {\n"
"  int t = blockIdx.x; int d = threadIdx.x;\n"
"  if (t >= n_tok || d >= dim) return;\n"
"  const float *xt = x + (size_t)t*dim;\n"
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
/* Padded window partition: (1,H,W,C) → (nwin, ws*ws, C), with nwin = (Hp/ws)*(Wp/ws) */
"__global__ void wpart_pad(float *y, const float *x, int H, int W, int Hp, int Wp, int C, int ws) {\n"
"  int t = blockIdx.x; int c = blockIdx.y * blockDim.x + threadIdx.x;\n"
"  if (c >= C) return;\n"
"  int nww = Wp/ws; int tok_per_win = ws*ws;\n"
"  int nwin = (Hp/ws)*nww;\n"
"  if (t >= nwin*tok_per_win) return;\n"
"  int w_idx = t/tok_per_win; int i = t%tok_per_win;\n"
"  int wy = w_idx/nww, wx = w_idx%nww;\n"
"  int yy = wy*ws + i/ws, xx = wx*ws + i%ws;\n"
"  float v = 0.f;\n"
"  if (yy < H && xx < W) v = x[((size_t)yy*W + xx)*C + c];\n"
"  y[(size_t)t*C + c] = v;\n"
"}\n"
/* Padded window unpartition + crop: (nwin, ws*ws, C) → (1, H, W, C) (crops pad) */
"__global__ void wunpart_crop(float *y, const float *x, int H, int W, int Hp, int Wp, int C, int ws) {\n"
"  int yy = blockIdx.x; int xx = blockIdx.y; int c = threadIdx.x;\n"
"  if (yy >= H || xx >= W || c >= C) return;\n"
"  int nww = Wp/ws;\n"
"  int wy = yy/ws, wx = xx/ws;\n"
"  int iy = yy%ws, ix = xx%ws;\n"
"  int w_idx = wy*nww + wx;\n"
"  int tok = w_idx*(ws*ws) + iy*ws + ix;\n"
"  y[((size_t)yy*W + xx)*C + c] = x[(size_t)tok*C + c];\n"
"}\n"
"__global__ void maxpool2d_k2s2(float *y, const float *x, int B, int H, int W, int C) {\n"
"  int Ho = H/2, Wo = W/2;\n"
"  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int total = B*Ho*Wo*C;\n"
"  if (idx >= total) return;\n"
"  int c  = idx % C; int wo = (idx/C)%Wo; int ho = (idx/(C*Wo))%Ho; int b = idx/(C*Wo*Ho);\n"
"  const float *xb = x + ((size_t)b*H*W)*C;\n"
"  float m = -1e30f;\n"
"  for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {\n"
"    float v = xb[((size_t)(ho*2+dy)*W + (wo*2+dx))*C + c]; if (v > m) m = v;\n"
"  }\n"
"  y[idx] = m;\n"
"}\n"
/* Pool Q from qkv. When pool=1: ws→ws/2 per window. When pool=0: copy Q slice only.
 * Output layout (nwin, Nq, D) where Nq = (ws/stride)^2. */
"__global__ void q_extract(float *qp, const float *qkv, int nwin, int ws, int D, int pool) {\n"
"  int wsout = pool ? ws/2 : ws;\n"
"  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int total = nwin * wsout*wsout * D;\n"
"  if (idx >= total) return;\n"
"  int d = idx % D; int p = (idx/D) % (wsout*wsout); int w = idx/(D*wsout*wsout);\n"
"  int py = p/wsout, px = p%wsout;\n"
"  const float *base = qkv + ((size_t)w*(ws*ws))*(3*D);\n"
"  if (pool) {\n"
"    float m = -1e30f;\n"
"    for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {\n"
"      int tok = (py*2+dy)*ws + (px*2+dx);\n"
"      float v = base[(size_t)tok*(3*D) + d]; if (v > m) m = v;\n"
"    }\n"
"    qp[idx] = m;\n"
"  } else {\n"
"    int tok = py*ws + px;\n"
"    qp[idx] = base[(size_t)tok*(3*D) + d];\n"
"  }\n"
"}\n"
/* Multi-head attention with flash-style online softmax.
 * Grid: (nwin, Nq, nh). Block: (hd) threads.
 * Q at (nwin, Nq, nh, hd); KV at (nwin, Nk, 3*D) with D=nh*hd.
 * Output: (nwin, Nq, D). */
"__global__ void mh_attn_flash(float *out, const float *Q, const float *qkv,\n"
"                               int nwin, int Nq, int Nk, int nh, int hd, float scale) {\n"
"  int w = blockIdx.x, qi = blockIdx.y, h = blockIdx.z;\n"
"  int d = threadIdx.x;\n"
"  int D = nh*hd;\n"
"  const float *Qv = Q + ((size_t)w*Nq + qi)*D + h*hd;\n"
"  const float *KVw = qkv + (size_t)w*Nk*(3*D);\n"
"  extern __shared__ float sh[];\n"
"  float *sQ = sh;                /* hd */\n"
"  float *sMS = sh + hd;          /* 2 floats: m, l (running max and sum) */\n"
"  float *sO  = sh + hd + 2;      /* hd: running output */\n"
"  if (d < hd) sQ[d] = Qv[d];\n"
"  if (d == 0) { sMS[0] = -1e30f; sMS[1] = 0.f; }\n"
"  if (d < hd) sO[d] = 0.f;\n"
"  __syncthreads();\n"
"  for (int k = 0; k < Nk; k++) {\n"
"    const float *K = KVw + (size_t)k*(3*D) + D + h*hd;\n"
"    const float *V = KVw + (size_t)k*(3*D) + 2*D + h*hd;\n"
"    /* Compute dot(Q, K) cooperatively via shared mem reduce (hd <= 128). */\n"
"    float prod = (d < hd) ? sQ[d]*K[d] : 0.f;\n"
"    /* warp/block reduction (simple: write partials to shmem slot [hd+2+hd .. +hd]) */\n"
"    float *tmp = sh + hd + 2 + hd;\n"
"    tmp[d] = prod;\n"
"    __syncthreads();\n"
"    if (d == 0) {\n"
"      float s = 0.f;\n"
"      for (int i = 0; i < hd; i++) s += tmp[i];\n"
"      s *= scale;\n"
"      float m_old = sMS[0];\n"
"      float m_new = fmaxf(m_old, s);\n"
"      float alpha = expf(m_old - m_new);\n"
"      float beta  = expf(s - m_new);\n"
"      sMS[0] = m_new;\n"
"      sMS[1] = sMS[1]*alpha + beta;\n"
"      tmp[0] = alpha; tmp[1] = beta;\n"
"    }\n"
"    __syncthreads();\n"
"    float alpha = tmp[0], beta = tmp[1];\n"
"    if (d < hd) sO[d] = sO[d]*alpha + beta*V[d];\n"
"    __syncthreads();\n"
"  }\n"
"  if (d < hd) out[((size_t)w*Nq + qi)*D + h*hd + d] = sO[d] / sMS[1];\n"
"}\n"
"}\n";

static int load_tensor(st_context *st, const char *name, float **out) {
    int i = safetensors_find(st, name);
    if (i < 0) { fprintf(stderr, "missing %s\n", name); return -1; }
    if (strcmp(safetensors_dtype(st, i), "F32")) return -1;
    *out = (float *)safetensors_data(st, i);
    return 0;
}

static void diff(const char *name, const float *a, const float *b, size_t n) {
    double mad = 0.0; float mxd = 0.f;
    for (size_t i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d > mxd) mxd = d; mad += d; }
    mad /= (double)n;
    fprintf(stderr, "  %-8s: max_abs=%.6g mean_abs=%.6g\n", name, mxd, mad);
}

static CUdeviceptr upf(const float *h, size_t n) {
    CUdeviceptr d; cuMemAlloc(&d, n*4); cuMemcpyHtoD(d, h, n*4); return d;
}

typedef struct {
    CUdeviceptr ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b;
    CUdeviceptr skip_w, skip_b;
    CUdeviceptr ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b;
} block_wts;

typedef struct {
    CUfunction ln, lin, gelu, add, wpart, wunpart, mpool, qextract, attn;
} block_fns;

/* Forward one block. dim, dim_out, nh, ws, pool, global, H, W are inputs.
 * For global, ws is ignored (treated as 1 big window of H*W tokens). */
static int forward_block(const block_fns *fn, const block_wts *W,
                         CUdeviceptr d_x, CUdeviceptr d_out,
                         int H, int W_, int dim, int dim_out, int nh,
                         int ws, int pool, int global_) {
    float eps = 1e-6f;
    int hd = dim_out / nh;
    int H_out = pool ? H/2 : H;
    int W_out = pool ? W_/2 : W_;
    int n_tok_in = H*W_;
    int n_tok_out = H_out*W_out;
    int mlp = dim_out*4;
    int ws_attn = global_ ? H_out : ws;
    int ws_in   = global_ ? H : ws;
    int pad_h_in = 0, pad_w_in = 0, Hp_in = H, Wp_in = W_;
    int pad_h_out = 0, pad_w_out = 0, Hp_out = H_out, Wp_out = W_out;

    if (!global_) {
        if (H % ws != 0)   pad_h_in = ws - H % ws;
        if (W_ % ws != 0)  pad_w_in = ws - W_ % ws;
        Hp_in = H + pad_h_in; Wp_in = W_ + pad_w_in;
        int ws_out = pool ? ws/2 : ws;
        if (H_out % ws_out != 0)  pad_h_out = ws_out - H_out % ws_out;
        if (W_out % ws_out != 0)  pad_w_out = ws_out - W_out % ws_out;
        Hp_out = H_out + pad_h_out; Wp_out = W_out + pad_w_out;
    }

    int nwin = global_ ? 1 : (Hp_in/ws_in) * (Wp_in/ws_in);
    int Nk = global_ ? H*W_ : ws_in*ws_in;
    int ws_out_eff = global_ ? H_out : (pool ? ws/2 : ws);
    int Nq = global_ ? H_out*W_out : ws_out_eff*ws_out_eff;

    CUdeviceptr d_ln1, d_residual, d_part, d_qkv, d_q, d_attn, d_proj, d_unpart, d_ln2, d_mlp, d_fc2, d_skip_pre=0;
    cuMemAlloc(&d_ln1, (size_t)n_tok_in*dim*4);
    cuMemAlloc(&d_residual, (size_t)n_tok_out*dim_out*4);
    cuMemAlloc(&d_qkv,  (size_t)nwin*Nk*3*dim_out*4);
    cuMemAlloc(&d_q,    (size_t)nwin*Nq*dim_out*4);
    cuMemAlloc(&d_attn, (size_t)nwin*Nq*dim_out*4);
    cuMemAlloc(&d_proj, (size_t)nwin*Nq*dim_out*4);
    cuMemAlloc(&d_unpart, (size_t)n_tok_out*dim_out*4);
    cuMemAlloc(&d_ln2, (size_t)n_tok_out*dim_out*4);
    cuMemAlloc(&d_mlp, (size_t)n_tok_out*mlp*4);
    cuMemAlloc(&d_fc2, (size_t)n_tok_out*dim_out*4);
    if (dim != dim_out) cuMemAlloc(&d_skip_pre, (size_t)n_tok_in*dim_out*4);
    if (!global_) cuMemAlloc(&d_part, (size_t)nwin*Nk*dim*4); else d_part = d_ln1;

    /* ln1 */
    {
        int nt = n_tok_in;
        void *a[] = { &d_ln1, &d_x, (void*)&W->ln1_w, (void*)&W->ln1_b, &nt, &dim, &eps };
        cuLaunchKernel(fn->ln, nt, 1, 1, dim, 1, 1, 0, 0, a, 0);
    }

    /* residual */
    if (dim != dim_out) {
        int nt = n_tok_in, dw = dim, dow = dim_out;
        void *a[] = { &d_skip_pre, &d_ln1, (void*)&W->skip_w, (void*)&W->skip_b, &nt, &dw, &dow };
        cuLaunchKernel(fn->lin, nt, (unsigned)((dow+255)/256), 1, 256, 1, 1, 0, 0, a, 0);
        if (pool) {
            int one=1; int total=n_tok_out*dim_out;
            void *am[] = { &d_residual, &d_skip_pre, &one, &H, &W_, &dim_out };
            cuLaunchKernel(fn->mpool, (unsigned)((total+255)/256),1,1,256,1,1,0,0,am,0);
        } else {
            cuMemcpyDtoD(d_residual, d_skip_pre, (size_t)n_tok_in*dim_out*4);
        }
    } else {
        cuMemcpyDtoD(d_residual, d_x, (size_t)n_tok_in*dim_out*4);
    }

    /* partition (if not global) */
    if (!global_) {
        int n = nwin*Nk;
        void *a[] = { &d_part, &d_ln1, &H, &W_, &Hp_in, &Wp_in, &dim, &ws };
        cuLaunchKernel(fn->wpart, n, (unsigned)((dim+255)/256),1, 256,1,1,0,0,a,0);
    }

    /* qkv */
    {
        int din = dim, dout = 3*dim_out, n = nwin*Nk;
        void *a[] = { &d_qkv, &d_part, (void*)&W->qkv_w, (void*)&W->qkv_b, &n, &din, &dout };
        cuLaunchKernel(fn->lin, n, (unsigned)((dout+255)/256), 1, 256, 1, 1, 0, 0, a, 0);
    }

    /* Extract / pool Q */
    {
        int total = nwin*Nq*dim_out;
        int poolflag = pool ? 1 : 0;
        int ws_e = ws_attn;
        void *a[] = { &d_q, &d_qkv, &nwin, &ws_e, &dim_out, &poolflag };
        cuLaunchKernel(fn->qextract, (unsigned)((total+255)/256),1,1,256,1,1,0,0,a,0);
    }

    /* attention (flash) */
    {
        float scale = 1.f/sqrtf((float)hd);
        size_t smem = (size_t)(hd + 2 + hd + hd) * 4;  /* sQ, mMS, sO, tmp */
        void *a[] = { &d_attn, &d_q, &d_qkv, &nwin, &Nq, &Nk, &nh, &hd, &scale };
        cuLaunchKernel(fn->attn, nwin, Nq, nh, hd, 1, 1, smem, 0, a, 0);
    }

    /* proj */
    {
        int din = dim_out, dout = dim_out, n = nwin*Nq;
        void *a[] = { &d_proj, &d_attn, (void*)&W->proj_w, (void*)&W->proj_b, &n, &din, &dout };
        cuLaunchKernel(fn->lin, n, (unsigned)((dout+255)/256), 1, 256, 1, 1, 0, 0, a, 0);
    }

    /* unpartition + crop (or just copy for global) */
    if (global_) {
        cuMemcpyDtoD(d_unpart, d_proj, (size_t)n_tok_out*dim_out*4);
    } else {
        int ws_e = ws_out_eff;
        void *a[] = { &d_unpart, &d_proj, &H_out, &W_out, &Hp_out, &Wp_out, &dim_out, &ws_e };
        cuLaunchKernel(fn->wunpart, H_out, W_out, 1, dim_out, 1, 1, 0, 0, a, 0);
    }

    /* residual += unpart */
    {
        int n = n_tok_out*dim_out;
        void *a[] = { &d_residual, &d_unpart, &n };
        cuLaunchKernel(fn->add, (unsigned)((n+255)/256),1,1,256,1,1,0,0,a,0);
    }

    /* ln2 */
    {
        int nt = n_tok_out;
        void *a[] = { &d_ln2, &d_residual, (void*)&W->ln2_w, (void*)&W->ln2_b, &nt, &dim_out, &eps };
        cuLaunchKernel(fn->ln, nt, 1, 1, dim_out, 1, 1, 0, 0, a, 0);
    }

    /* fc1 + gelu + fc2 */
    {
        int nt=n_tok_out, din=dim_out, dout=mlp;
        void *a[] = { &d_mlp, &d_ln2, (void*)&W->fc1_w, (void*)&W->fc1_b, &nt, &din, &dout };
        cuLaunchKernel(fn->lin, nt, (unsigned)((dout+255)/256), 1, 256, 1, 1, 0, 0, a, 0);
        int n = n_tok_out*mlp;
        void *ag[] = { &d_mlp, &n };
        cuLaunchKernel(fn->gelu, (unsigned)((n+255)/256),1,1,256,1,1,0,0,ag,0);
    }
    {
        int nt=n_tok_out, din=mlp, dout=dim_out;
        void *a[] = { &d_fc2, &d_mlp, (void*)&W->fc2_w, (void*)&W->fc2_b, &nt, &din, &dout };
        cuLaunchKernel(fn->lin, nt, (unsigned)((dout+255)/256), 1, 256, 1, 1, 0, 0, a, 0);
    }

    cuMemcpyDtoD(d_out, d_residual, (size_t)n_tok_out*dim_out*4);
    {
        int n = n_tok_out*dim_out;
        void *a[] = { &d_out, &d_fc2, &n };
        cuLaunchKernel(fn->add, (unsigned)((n+255)/256),1,1,256,1,1,0,0,a,0);
    }

    cuCtxSynchronize();
    cuMemFree(d_ln1); cuMemFree(d_residual); cuMemFree(d_qkv); cuMemFree(d_q);
    cuMemFree(d_attn); cuMemFree(d_proj); cuMemFree(d_unpart); cuMemFree(d_ln2);
    cuMemFree(d_mlp); cuMemFree(d_fc2);
    if (d_skip_pre) cuMemFree(d_skip_pre);
    if (!global_) cuMemFree(d_part);
    return 0;
}

static int load_block(st_context *st, int idx, int has_proj, block_wts *W, int dim, int dim_out) {
    char k[256]; float *h;
    #define LD(SUF, OUT, N) do { snprintf(k,sizeof(k),"vision_encoder.backbone.blocks.%d.%s",idx,SUF); \
        if (load_tensor(st,k,&h)) return -1; OUT = upf(h,(size_t)(N)); } while(0)
    LD("layer_norm1.weight", W->ln1_w, dim);
    LD("layer_norm1.bias",   W->ln1_b, dim);
    LD("attn.qkv.weight",    W->qkv_w, 3*dim_out*dim);
    LD("attn.qkv.bias",      W->qkv_b, 3*dim_out);
    LD("attn.proj.weight",   W->proj_w, dim_out*dim_out);
    LD("attn.proj.bias",     W->proj_b, dim_out);
    LD("layer_norm2.weight", W->ln2_w, dim_out);
    LD("layer_norm2.bias",   W->ln2_b, dim_out);
    LD("mlp.proj_in.weight", W->fc1_w, (dim_out*4)*dim_out);
    LD("mlp.proj_in.bias",   W->fc1_b, dim_out*4);
    LD("mlp.proj_out.weight",W->fc2_w, dim_out*(dim_out*4));
    LD("mlp.proj_out.bias",  W->fc2_b, dim_out);
    if (has_proj) {
        LD("proj.weight", W->skip_w, dim_out*dim);
        LD("proj.bias",   W->skip_b, dim_out);
    } else { W->skip_w = 0; W->skip_b = 0; }
    #undef LD
    return 0;
}

static void free_block(block_wts *W) {
    CUdeviceptr ps[] = { W->ln1_w,W->ln1_b,W->qkv_w,W->qkv_b,W->proj_w,W->proj_b,
                          W->ln2_w,W->ln2_b,W->fc1_w,W->fc1_b,W->fc2_w,W->fc2_b,
                          W->skip_w,W->skip_b };
    for (int i = 0; i < 14; i++) if (ps[i]) cuMemFree(ps[i]);
}

typedef struct { int dim_in, dim_out, nh, ws, pool, has_proj, global_, H, W; } blk_cfg;

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <model.safetensors> <refdir>\n", argv[0]); return 1; }
    const char *ckpt = argv[1]; const char *refdir = argv[2]; char path[1024];

    st_context *st = safetensors_open(ckpt);
    if (!st) { fprintf(stderr, "safetensors_open failed\n"); return 3; }
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 5;
    if (cuInit(0) != CUDA_SUCCESS) return 5;
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    if (cu_compile_kernels(&mod, dev, kern_src, "sam2_bb", 1, "sam2") < 0) return 6;
    block_fns fn;
    cuModuleGetFunction(&fn.ln,     mod, "ln_last");
    cuModuleGetFunction(&fn.lin,    mod, "linear2d");
    cuModuleGetFunction(&fn.gelu,   mod, "gelu_f");
    cuModuleGetFunction(&fn.add,    mod, "add_inplace");
    cuModuleGetFunction(&fn.wpart,  mod, "wpart_pad");
    cuModuleGetFunction(&fn.wunpart,mod, "wunpart_crop");
    cuModuleGetFunction(&fn.mpool,  mod, "maxpool2d_k2s2");
    cuModuleGetFunction(&fn.qextract,mod,"q_extract");
    cuModuleGetFunction(&fn.attn,   mod, "mh_attn_flash");

    /* Block configs for tiny model. {dim_in, dim_out, nh, ws, pool, has_proj, global, H, W} */
    blk_cfg cfg[12] = {
        {96,  96,  1, 8,  0, 0, 0, 256, 256},  /* 0 */
        {96,  192, 2, 8,  1, 1, 0, 256, 256},  /* 1 */
        {192, 192, 2, 4,  0, 0, 0, 128, 128},  /* 2 */
        {192, 384, 4, 4,  1, 1, 0, 128, 128},  /* 3 */
        {384, 384, 4, 14, 0, 0, 0, 64,  64 },  /* 4 */
        {384, 384, 4, 0,  0, 0, 1, 64,  64 },  /* 5 global */
        {384, 384, 4, 14, 0, 0, 0, 64,  64 },  /* 6 */
        {384, 384, 4, 0,  0, 0, 1, 64,  64 },  /* 7 global */
        {384, 384, 4, 14, 0, 0, 0, 64,  64 },  /* 8 */
        {384, 384, 4, 0,  0, 0, 1, 64,  64 },  /* 9 global */
        {384, 768, 8, 14, 1, 1, 0, 64,  64 },  /* 10 */
        {768, 768, 8, 7,  0, 0, 0, 32,  32 },  /* 11 */
    };
    block_wts W[12] = {0};
    for (int i = 0; i < 12; i++) {
        if (load_block(st, i, cfg[i].has_proj, &W[i], cfg[i].dim_in, cfg[i].dim_out)) {
            fprintf(stderr, "load block %d failed\n", i); return 4;
        }
    }

    int cascade = (argc > 3 && !strcmp(argv[3], "--cascade"));

    if (!cascade) {
        for (int i = 0; i < 12; i++) {
            fprintf(stderr, "block %d:\n", i);
            snprintf(path, sizeof(path), "%s/block%d_input.npy", refdir, i);
            int d[5], nd; float *xin = read_npy_f32(path, d, &nd);
            if (!xin) { fprintf(stderr, "  missing %s\n", path); continue; }
            int H = cfg[i].H, Wv = cfg[i].W, din = cfg[i].dim_in, dout = cfg[i].dim_out;
            int H_out = cfg[i].pool ? H/2 : H;
            int W_out = cfg[i].pool ? Wv/2 : Wv;
            CUdeviceptr d_x, d_out;
            cuMemAlloc(&d_x, (size_t)H*Wv*din*4);
            cuMemcpyHtoD(d_x, xin, (size_t)H*Wv*din*4);
            cuMemAlloc(&d_out, (size_t)H_out*W_out*dout*4);
            forward_block(&fn, &W[i], d_x, d_out, H, Wv, din, dout, cfg[i].nh,
                          cfg[i].ws, cfg[i].pool, cfg[i].global_);

            snprintf(path, sizeof(path), "%s/block%d_output.npy", refdir, i);
            float *ref = read_npy_f32(path, d, &nd);
            if (ref) {
                float *tmp = (float *)malloc((size_t)H_out*W_out*dout*4);
                cuMemcpyDtoH(tmp, d_out, (size_t)H_out*W_out*dout*4);
                char tag[16]; snprintf(tag,sizeof(tag),"block%d", i);
                diff(tag, tmp, ref, (size_t)H_out*W_out*dout);
                free(tmp); free(ref);
            }
            cuMemFree(d_x); cuMemFree(d_out); free(xin);
        }
    } else {
        /* Cascade: seed from block0_input, forward all 12 blocks chained, compare
         * stage-end outputs to intermediate_{0..3}.npy. */
        fprintf(stderr, "cascade:\n");
        snprintf(path, sizeof(path), "%s/block0_input.npy", refdir);
        int d[5], nd; float *xin = read_npy_f32(path, d, &nd);
        if (!xin) { fprintf(stderr, "  missing %s\n", path); return 2; }
        CUdeviceptr d_a, d_b;
        cuMemAlloc(&d_a, (size_t)256*256*96*4);
        cuMemcpyHtoD(d_a, xin, (size_t)256*256*96*4);
        free(xin);
        int cur_H = 256, cur_W = 256, cur_C = 96;
        int stage_end[4] = {0, 2, 9, 11};
        int se = 0;
        for (int i = 0; i < 12; i++) {
            int H = cur_H, Wv = cur_W;
            int H_out = cfg[i].pool ? H/2 : H;
            int W_out = cfg[i].pool ? Wv/2 : Wv;
            cuMemAlloc(&d_b, (size_t)H_out*W_out*cfg[i].dim_out*4);
            forward_block(&fn, &W[i], d_a, d_b, H, Wv, cfg[i].dim_in, cfg[i].dim_out,
                          cfg[i].nh, cfg[i].ws, cfg[i].pool, cfg[i].global_);
            cuMemFree(d_a); d_a = d_b;
            cur_H = H_out; cur_W = W_out; cur_C = cfg[i].dim_out;

            if (i == stage_end[se]) {
                snprintf(path, sizeof(path), "%s/intermediate_%d.npy", refdir, se);
                float *ref = read_npy_f32(path, d, &nd);
                if (ref) {
                    float *tmp = (float *)malloc((size_t)cur_H*cur_W*cur_C*4);
                    cuMemcpyDtoH(tmp, d_a, (size_t)cur_H*cur_W*cur_C*4);
                    char tag[24]; snprintf(tag,sizeof(tag),"intermed_%d", se);
                    diff(tag, tmp, ref, (size_t)cur_H*cur_W*cur_C);
                    free(tmp); free(ref);
                }
                se++;
            }
        }
        /* Final vs backbone_last */
        snprintf(path, sizeof(path), "%s/backbone_last.npy", refdir);
        float *ref = read_npy_f32(path, d, &nd);
        if (ref) {
            float *tmp = (float *)malloc((size_t)cur_H*cur_W*cur_C*4);
            cuMemcpyDtoH(tmp, d_a, (size_t)cur_H*cur_W*cur_C*4);
            diff("backbone", tmp, ref, (size_t)cur_H*cur_W*cur_C);
            free(tmp); free(ref);
        }
        cuMemFree(d_a);
    }

    for (int i = 0; i < 12; i++) free_block(&W[i]);
    cuModuleUnload(mod); cuCtxDestroy(ctx); safetensors_close(st);
    return 0;
}
