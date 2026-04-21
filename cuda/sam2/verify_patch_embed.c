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

static float *read_npy_f32(const char *path, int dims[4], int *ndims) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t h10[10];
    if (fread(h10, 1, 10, f) != 10) { fclose(f); return NULL; }
    if (memcmp(h10, "\x93NUMPY", 6)) { fclose(f); return NULL; }
    uint16_t hlen = (uint16_t)(h10[8] | (h10[9] << 8));
    char *hdr = (char *)malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = '\0';
    if (!strstr(hdr, "'descr': '<f4'")) { free(hdr); fclose(f); return NULL; }
    char *p = strchr(hdr, '('), *q = strchr(hdr, ')');
    if (!p || !q) { free(hdr); fclose(f); return NULL; }
    p++;
    int n = 0;
    while (p < q && n < 4) {
        while (p < q && (*p < '0' || *p > '9')) p++;
        if (p >= q) break;
        dims[n++] = (int)strtol(p, &p, 10);
    }
    free(hdr);
    size_t cnt = 1;
    for (int i = 0; i < n; i++) cnt *= (size_t)dims[i];
    float *x = (float *)malloc(cnt * sizeof(float));
    if (!x) { fclose(f); return NULL; }
    if (fread(x, sizeof(float), cnt, f) != cnt) { free(x); fclose(f); return NULL; }
    fclose(f);
    *ndims = n;
    return x;
}

static const char *kern_src =
"extern \"C\" __global__ void sam2_patch_conv(float *out, const float *in, const float *w, const float *b, int H, int W, int Co, int Ci, int K, int S, int P) {\n"
"  int ox = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  int oy = blockIdx.y * blockDim.y + threadIdx.y;\n"
"  int co = blockIdx.z;\n"
"  int OH = (H + 2 * P - K) / S + 1;\n"
"  int OW = (W + 2 * P - K) / S + 1;\n"
"  if (ox >= OW || oy >= OH || co >= Co) return;\n"
"  float acc = b ? b[co] : 0.0f;\n"
"  for (int ci = 0; ci < Ci; ci++) {\n"
"    for (int ky = 0; ky < K; ky++) {\n"
"      int iy = oy * S + ky - P;\n"
"      if (iy < 0 || iy >= H) continue;\n"
"      for (int kx = 0; kx < K; kx++) {\n"
"        int ix = ox * S + kx - P;\n"
"        if (ix < 0 || ix >= W) continue;\n"
"        float v = in[(ci * H + iy) * W + ix];\n"
"        float ww = w[((co * Ci + ci) * K + ky) * K + kx];\n"
"        acc += ww * v;\n"
"      }\n"
"    }\n"
"  }\n"
"  out[(co * OH + oy) * OW + ox] = acc;\n"
"}\n";

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <sam2 model.safetensors> <refdir>\n", argv[0]);
        return 1;
    }
    const char *ckpt = argv[1];
    const char *refdir = argv[2];

    char p_in[1024], p_ref[1024];
    snprintf(p_in, sizeof(p_in), "%s/input_pixel_values.npy", refdir);
    snprintf(p_ref, sizeof(p_ref), "%s/patch_conv.npy", refdir);

    int id[4]={0}, rd[4]={0}, in_nd=0, ref_nd=0;
    float *in = read_npy_f32(p_in, id, &in_nd);
    float *ref = read_npy_f32(p_ref, rd, &ref_nd);
    if (!in || !ref) { fprintf(stderr, "failed to load npy\n"); return 2; }

    int Ci = id[0], H = id[1], W = id[2];
    int Co = rd[0], OH = rd[1], OW = rd[2];

    st_context *st = safetensors_open(ckpt);
    if (!st) { fprintf(stderr, "safetensors_open failed\n"); return 3; }
    int iw = safetensors_find(st, "vision_encoder.backbone.patch_embed.projection.weight");
    int ib = safetensors_find(st, "vision_encoder.backbone.patch_embed.projection.bias");
    if (iw < 0 || ib < 0) { fprintf(stderr, "missing patch weights\n"); return 4; }
    const float *w = (const float *)safetensors_data(st, iw);
    const float *b = (const float *)safetensors_data(st, ib);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 5;
    if (cuInit(0) != CUDA_SUCCESS) return 5;
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    if (cu_compile_kernels(&mod, dev, kern_src, "sam2_patch", 1, "sam2") < 0) return 6;
    CUfunction fn; if (cuModuleGetFunction(&fn, mod, "sam2_patch_conv") != CUDA_SUCCESS) return 6;

    CUdeviceptr d_in=0,d_w=0,d_b=0,d_out=0;
    size_t in_b = (size_t)Ci*H*W*sizeof(float);
    size_t w_b  = (size_t)Co*Ci*7*7*sizeof(float);
    size_t b_b  = (size_t)Co*sizeof(float);
    size_t out_b= (size_t)Co*OH*OW*sizeof(float);
    cuMemAlloc(&d_in, in_b); cuMemcpyHtoD(d_in, in, in_b);
    cuMemAlloc(&d_w, w_b); cuMemcpyHtoD(d_w, w, w_b);
    cuMemAlloc(&d_b, b_b); cuMemcpyHtoD(d_b, b, b_b);
    cuMemAlloc(&d_out, out_b);

    int K=7,S=4,P=3;
    void *args[] = { &d_out, &d_in, &d_w, &d_b, &H, &W, &Co, &Ci, &K, &S, &P };
    unsigned bx=16, by=16;
    unsigned gx=(unsigned)((OW + (int)bx - 1)/(int)bx);
    unsigned gy=(unsigned)((OH + (int)by - 1)/(int)by);
    cuLaunchKernel(fn, gx, gy, (unsigned)Co, bx, by, 1, 0, 0, args, 0);
    cuCtxSynchronize();

    float *out = (float *)malloc(out_b);
    cuMemcpyDtoH(out, d_out, out_b);

    double mad = 0.0; float mxd = 0.0f;
    size_t n = (size_t)Co*OH*OW;
    for (size_t i=0;i<n;i++) {
        float d = fabsf(out[i]-ref[i]);
        if (d > mxd) mxd = d;
        mad += d;
    }
    mad /= (double)n;
    fprintf(stderr, "patch_conv: max_abs=%.6g mean_abs=%.6g (C=%d H=%d W=%d)\n", mxd, mad, Co, OH, OW);

    free(in); free(ref); free(out);
    cuMemFree(d_in); cuMemFree(d_w); cuMemFree(d_b); cuMemFree(d_out);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    safetensors_close(st);
    return 0;
}
