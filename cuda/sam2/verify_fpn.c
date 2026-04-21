/*
 * verify_fpn.c — Verify CUDA FPN neck against /tmp/sam2_trace/fpn_*.npy.
 *
 * FPN (Sam2VisionNeck) for tiny:
 *   4 conv 1x1: in=[768,384,192,96], out=256
 *   Loop i=n..0 (n=3): lateral = convs[n-i](intermediate_i)
 *     if i==n or i not in {2,3}: prev = lateral
 *     else: prev = lateral + up2x(prev)
 *   With scalp=1: HF drops lowest-res head, returning 3 outputs at 256²,128²,64².
 *
 * Our convention: work in BHWC (matches backbone outputs). Reference fpn_k.npy
 * is BCHW — we permute before comparison.
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
"__global__ void linear2d(float *y, const float *x, const float *w, const float *b, int n_tok, int din, int dout) {\n"
"  int o = blockIdx.y * blockDim.x + threadIdx.x; int t = blockIdx.x;\n"
"  if (t >= n_tok || o >= dout) return;\n"
"  float acc = b ? b[o] : 0.f;\n"
"  const float *xt = x + (size_t)t*din;\n"
"  const float *wo = w + (size_t)o*din;\n"
"  for (int i = 0; i < din; i++) acc += wo[i] * xt[i];\n"
"  y[(size_t)t*dout + o] = acc;\n"
"}\n"
"__global__ void add_inplace(float *x, const float *y, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (i < n) x[i] += y[i];\n"
"}\n"
/* Nearest 2x upsample, (1,H,W,C) -> (1,2H,2W,C). */
"__global__ void upsample2x_hwc(float *y, const float *x, int H, int W, int C) {\n"
"  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int total = 2*H*2*W*C;\n"
"  if (idx >= total) return;\n"
"  int Wo = 2*W;\n"
"  int c = idx % C;\n"
"  int xo = (idx/C) % Wo;\n"
"  int yo = idx / (C*Wo);\n"
"  int xi = xo/2, yi = yo/2;\n"
"  y[idx] = x[((size_t)yi*W + xi)*C + c];\n"
"}\n"
"}\n";

static int load_tensor(st_context *st, const char *name, float **out) {
    int i = safetensors_find(st, name);
    if (i < 0) { fprintf(stderr, "missing %s\n", name); return -1; }
    if (strcmp(safetensors_dtype(st, i), "F32")) return -1;
    *out = (float *)safetensors_data(st, i);
    return 0;
}

static CUdeviceptr upf(const float *h, size_t n) {
    CUdeviceptr d; cuMemAlloc(&d, n*4); cuMemcpyHtoD(d, h, n*4); return d;
}

static void diff(const char *name, const float *a, const float *b, size_t n) {
    double mad = 0.0; float mxd = 0.f;
    for (size_t i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d > mxd) mxd = d; mad += d; }
    mad /= (double)n;
    fprintf(stderr, "  %-8s: max_abs=%.6g mean_abs=%.6g\n", name, mxd, mad);
}

/* PyTorch Conv2d.weight shape is [C_out, C_in, 1, 1] — stored same as [C_out, C_in] flat
 * which is also the Linear.weight layout expected by linear2d. */

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
    if (cu_compile_kernels(&mod, dev, kern_src, "sam2_fpn", 1, "sam2") < 0) return 6;
    CUfunction fn_lin, fn_add, fn_up;
    cuModuleGetFunction(&fn_lin, mod, "linear2d");
    cuModuleGetFunction(&fn_add, mod, "add_inplace");
    cuModuleGetFunction(&fn_up,  mod, "upsample2x_hwc");

    /* Load intermediates from trace */
    int stage_res[4] = {256, 128, 64, 32};
    int stage_ch[4]  = {96,  192, 384, 768};
    float *im[4]; int d[5], nd;
    for (int i = 0; i < 4; i++) {
        snprintf(path, sizeof(path), "%s/intermediate_%d.npy", refdir, i);
        im[i] = read_npy_f32(path, d, &nd);
        if (!im[i]) { fprintf(stderr, "missing intermediate_%d.npy\n", i); return 2; }
    }

    /* Load conv weights: convs[k] for k=0..3, maps stage (3-k) -> 256 ch */
    CUdeviceptr cw[4], cb[4];
    for (int k = 0; k < 4; k++) {
        char nw[128], nb[128];
        snprintf(nw, sizeof(nw), "vision_encoder.neck.convs.%d.weight", k);
        snprintf(nb, sizeof(nb), "vision_encoder.neck.convs.%d.bias", k);
        float *h;
        if (load_tensor(st, nw, &h)) return 4;
        cw[k] = upf(h, (size_t)256 * stage_ch[3-k]);
        if (load_tensor(st, nb, &h)) return 4;
        cb[k] = upf(h, 256);
    }

    /* Upload intermediates to device */
    CUdeviceptr d_im[4];
    for (int i = 0; i < 4; i++) {
        size_t n = (size_t)stage_res[i]*stage_res[i]*stage_ch[i];
        cuMemAlloc(&d_im[i], n*4);
        cuMemcpyHtoD(d_im[i], im[i], n*4);
    }

    /* Run FPN loop (i = 3 .. 0) */
    CUdeviceptr d_prev = 0;
    int prev_res = 0;
    int top_down[2] = {2, 3};
    CUdeviceptr outs[4] = {0}; int out_res[4];

    for (int i = 3; i >= 0; i--) {
        int k = 3 - i;
        int res = stage_res[i];
        int n_tok = res*res;
        /* lateral = conv_{k}(im[i]) : (res², 256) */
        CUdeviceptr d_lat; cuMemAlloc(&d_lat, (size_t)n_tok*256*4);
        {
            int din = stage_ch[i], dout = 256;
            void *a[] = { &d_lat, &d_im[i], &cw[k], &cb[k], &n_tok, &din, &dout };
            cuLaunchKernel(fn_lin, n_tok, (unsigned)((dout+255)/256), 1, 256, 1, 1, 0, 0, a, 0);
        }
        int use_td = (i == 3) ? 0 :
                     (i == top_down[0] || i == top_down[1]) ? 1 : 0;
        if (!use_td) {
            cuMemFree(d_prev);
            d_prev = d_lat; prev_res = res;
        } else {
            /* upsample prev 2x → res² */
            CUdeviceptr d_up; cuMemAlloc(&d_up, (size_t)n_tok*256*4);
            int C = 256;
            void *a[] = { &d_up, &d_prev, &prev_res, &prev_res, &C };
            int total = n_tok*256;
            cuLaunchKernel(fn_up, (unsigned)((total+255)/256),1,1,256,1,1,0,0,a,0);
            cuMemFree(d_prev);
            /* d_lat += d_up */
            int n = n_tok*256;
            void *aa[] = { &d_lat, &d_up, &n };
            cuLaunchKernel(fn_add, (unsigned)((n+255)/256),1,1,256,1,1,0,0,aa,0);
            cuMemFree(d_up);
            d_prev = d_lat; prev_res = res;
        }
        outs[3-i] = d_prev; out_res[3-i] = res; /* outs[0]=res 32, outs[1]=64, outs[2]=128, outs[3]=256 */
        if (i != 0) {
            CUdeviceptr keep; cuMemAlloc(&keep, (size_t)n_tok*256*4);
            cuMemcpyDtoD(keep, d_prev, (size_t)n_tok*256*4);
            outs[3-i] = keep;
        }
    }
    cuCtxSynchronize();
    /* scalp=1: drop outs[0] (32²), keep outs[1],outs[2],outs[3] as fpn_2,fpn_1,fpn_0
     * But trace fpn_0=256²,fpn_1=128²,fpn_2=64². So:
     *   fpn_0 ↔ outs[3] (256²)
     *   fpn_1 ↔ outs[2] (128²)
     *   fpn_2 ↔ outs[1] (64²)
     */
    int fpn_map[3] = { 3, 2, 1 };
    int fpn_res[3] = { 256, 128, 64 };
    for (int k = 0; k < 3; k++) {
        snprintf(path, sizeof(path), "%s/fpn_%d.npy", refdir, k);
        float *ref = read_npy_f32(path, d, &nd);
        if (!ref) continue;
        int res = fpn_res[k];
        size_t n = (size_t)res*res*256;
        float *tmp = (float *)malloc(n*4);
        cuMemcpyDtoH(tmp, outs[fpn_map[k]], n*4);
        /* tmp is BHWC. Ref is BCHW. Permute tmp to BCHW for comparison. */
        float *tmp_bchw = (float *)malloc(n*4);
        for (int y = 0; y < res; y++)
            for (int x = 0; x < res; x++)
                for (int c = 0; c < 256; c++)
                    tmp_bchw[((size_t)c*res + y)*res + x] = tmp[((size_t)y*res + x)*256 + c];
        char tag[16]; snprintf(tag, sizeof(tag), "fpn_%d", k);
        diff(tag, tmp_bchw, ref, n);
        free(tmp); free(tmp_bchw); free(ref);
    }

    for (int i = 0; i < 4; i++) { if (outs[i]) cuMemFree(outs[i]); cuMemFree(d_im[i]); free(im[i]); }
    for (int k = 0; k < 4; k++) { cuMemFree(cw[k]); cuMemFree(cb[k]); }
    cuModuleUnload(mod); cuCtxDestroy(ctx); safetensors_close(st);
    return 0;
}
