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

#define H 256
#define W 256
#define D 96
#define N (H*W)
#define WS 8
#define NW ((H/WS)*(W/WS))
#define T (WS*WS)
#define MLP 384

static float *read_npy_f32(const char *path, int dims[4], int *ndims) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t h10[10];
    if (fread(h10, 1, 10, f) != 10) { fclose(f); return NULL; }
    if (memcmp(h10, "\x93NUMPY", 6)) { fclose(f); return NULL; }
    uint16_t hlen = (uint16_t)(h10[8] | (h10[9] << 8));
    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr) { fclose(f); return NULL; }
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
"extern \"C\" {\n"
"__global__ void ln_last(float *y, const float *x, const float *w, const float *b, int n_tok, int dim, float eps) {\n"
"  int t = blockIdx.x;\n"
"  int d = threadIdx.x;\n"
"  if (t >= n_tok || d >= dim) return;\n"
"  const float *xt = x + (size_t)t * dim;\n"
"  float mean = 0.0f;\n"
"  for (int i = 0; i < dim; i++) mean += xt[i];\n"
"  mean /= dim;\n"
"  float var = 0.0f;\n"
"  for (int i = 0; i < dim; i++) { float v = xt[i] - mean; var += v * v; }\n"
"  var /= dim;\n"
"  y[(size_t)t * dim + d] = ((xt[d] - mean) * rsqrtf(var + eps)) * w[d] + b[d];\n"
"}\n"
"__global__ void linear2d(float *y, const float *x, const float *w, const float *b, int n_tok, int din, int dout) {\n"
"  int o = blockIdx.y * blockDim.x + threadIdx.x;\n"
"  int t = blockIdx.x;\n"
"  if (t >= n_tok || o >= dout) return;\n"
"  float acc = b ? b[o] : 0.0f;\n"
"  const float *xt = x + (size_t)t * din;\n"
"  const float *wo = w + (size_t)o * din;\n"
"  for (int i = 0; i < din; i++) acc += wo[i] * xt[i];\n"
"  y[(size_t)t * dout + o] = acc;\n"
"}\n"
"__global__ void gelu_f(float *x, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (i < n) { float v = x[i]; x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f)); }\n"
"}\n"
"__global__ void add_inplace(float *x, const float *y, int n) {\n"
"  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (i < n) x[i] += y[i];\n"
"}\n"
"__global__ void block0_attn(float *out, const float *qkv, const float *pw, const float *pb, int H, int W, int D, int ws, float scale) {\n"
"  int win = blockIdx.x;\n"
"  int qi = threadIdx.x;\n"
"  if (qi >= ws*ws) return;\n"
"  int nww = W / ws;\n"
"  int wy = win / nww, wx = win % nww;\n"
"  int qy = wy * ws + qi / ws;\n"
"  int qx = wx * ws + qi % ws;\n"
"  int qg = qy * W + qx;\n"
"  const float *Q = qkv + (size_t)qg * (3*D);\n"
"  float scores[64];\n"
"  float m = -1e30f;\n"
"  for (int kj = 0; kj < ws*ws; kj++) {\n"
"    int ky = wy * ws + kj / ws;\n"
"    int kx = wx * ws + kj % ws;\n"
"    int kg = ky * W + kx;\n"
"    const float *K = qkv + (size_t)kg * (3*D) + D;\n"
"    float s = 0.0f;\n"
"    for (int d = 0; d < D; d++) s += Q[d] * K[d];\n"
"    s *= scale;\n"
"    scores[kj] = s;\n"
"    if (s > m) m = s;\n"
"  }\n"
"  float z = 0.0f;\n"
"  for (int kj = 0; kj < ws*ws; kj++) { scores[kj] = expf(scores[kj] - m); z += scores[kj]; }\n"
"  float attn[96];\n"
"  for (int d = 0; d < D; d++) attn[d] = 0.0f;\n"
"  for (int kj = 0; kj < ws*ws; kj++) {\n"
"    int ky = wy * ws + kj / ws;\n"
"    int kx = wx * ws + kj % ws;\n"
"    int kg = ky * W + kx;\n"
"    const float *V = qkv + (size_t)kg * (3*D) + 2*D;\n"
"    float a = scores[kj] / z;\n"
"    for (int d = 0; d < D; d++) attn[d] += a * V[d];\n"
"  }\n"
"  for (int o = 0; o < D; o++) {\n"
"    const float *wo = pw + (size_t)o * D;\n"
"    float v = pb[o];\n"
"    for (int d = 0; d < D; d++) v += wo[d] * attn[d];\n"
"    out[(size_t)qg * D + o] = v;\n"
"  }\n"
"}\n"
"}\n";

static int load_tensor(st_context *st, const char *name, float **out, size_t *count) {
    int i = safetensors_find(st, name);
    if (i < 0) { fprintf(stderr, "missing %s\n", name); return -1; }
    if (strcmp(safetensors_dtype(st, i), "F32")) { fprintf(stderr, "dtype mismatch %s\n", name); return -1; }
    size_t nb = safetensors_nbytes(st, i);
    *count = nb / sizeof(float);
    *out = (float *)safetensors_data(st, i);
    return 0;
}

static void diff_stats(const char *name, const float *a, const float *b, size_t n) {
    double mad = 0.0;
    float mxd = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mxd) mxd = d;
        mad += d;
    }
    mad /= (double)n;
    fprintf(stderr, "%s: max_abs=%.6g mean_abs=%.6g\n", name, mxd, mad);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <sam2 model.safetensors> <refdir>\n", argv[0]);
        return 1;
    }
    const char *ckpt = argv[1];
    const char *refdir = argv[2];

    char p_in[1024], p_ref[1024];
    snprintf(p_in, sizeof(p_in), "%s/block0_input.npy", refdir);
    snprintf(p_ref, sizeof(p_ref), "%s/block0_output.npy", refdir);

    int id[4]={0}, od[4]={0}, in_nd=0, out_nd=0;
    float *xin = read_npy_f32(p_in, id, &in_nd);
    float *yref = read_npy_f32(p_ref, od, &out_nd);
    if (!xin || !yref) { fprintf(stderr, "failed to load ref npy\n"); return 2; }

    char p_ln1[1024], p_attn[1024], p_res1[1024], p_ln2[1024], p_mlp[1024];
    snprintf(p_ln1,  sizeof(p_ln1),  "%s/block0_ln1.npy", refdir);
    snprintf(p_attn, sizeof(p_attn), "%s/block0_attn.npy", refdir);
    snprintf(p_res1, sizeof(p_res1), "%s/block0_res1.npy", refdir);
    snprintf(p_ln2,  sizeof(p_ln2),  "%s/block0_ln2.npy", refdir);
    snprintf(p_mlp,  sizeof(p_mlp),  "%s/block0_mlp.npy", refdir);
    int td[4], tnd;
    float *ref_ln1  = read_npy_f32(p_ln1, td, &tnd);
    float *ref_attn = read_npy_f32(p_attn, td, &tnd);
    float *ref_res1 = read_npy_f32(p_res1, td, &tnd);
    float *ref_ln2  = read_npy_f32(p_ln2, td, &tnd);
    float *ref_mlp  = read_npy_f32(p_mlp, td, &tnd);

    st_context *st = safetensors_open(ckpt);
    if (!st) { fprintf(stderr, "safetensors_open failed\n"); return 3; }

    float *ln1_w,*ln1_b,*qkv_w,*qkv_b,*proj_w,*proj_b,*ln2_w,*ln2_b,*fc1_w,*fc1_b,*fc2_w,*fc2_b;
    size_t n;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.layer_norm1.weight", &ln1_w, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.layer_norm1.bias", &ln1_b, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.attn.qkv.weight", &qkv_w, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.attn.qkv.bias", &qkv_b, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.attn.proj.weight", &proj_w, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.attn.proj.bias", &proj_b, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.layer_norm2.weight", &ln2_w, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.layer_norm2.bias", &ln2_b, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.mlp.proj_in.weight", &fc1_w, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.mlp.proj_in.bias", &fc1_b, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.mlp.proj_out.weight", &fc2_w, &n)) return 4;
    if (load_tensor(st, "vision_encoder.backbone.blocks.0.mlp.proj_out.bias", &fc2_b, &n)) return 4;

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 5;
    if (cuInit(0) != CUDA_SUCCESS) return 5;
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuCtxCreate(&ctx, 0, dev);

    CUmodule mod;
    if (cu_compile_kernels(&mod, dev, kern_src, "sam2_b0", 1, "sam2") < 0) return 6;
    CUfunction fn_ln, fn_lin, fn_gelu, fn_add, fn_attn;
    cuModuleGetFunction(&fn_ln, mod, "ln_last");
    cuModuleGetFunction(&fn_lin, mod, "linear2d");
    cuModuleGetFunction(&fn_gelu, mod, "gelu_f");
    cuModuleGetFunction(&fn_add, mod, "add_inplace");
    cuModuleGetFunction(&fn_attn, mod, "block0_attn");

    CUdeviceptr d_x,d_ln,d_qkv,d_attn,d_tmp,d_mlp,d_w,d_b;
    cuMemAlloc(&d_x, (size_t)N*D*4);
    cuMemcpyHtoD(d_x, xin, (size_t)N*D*4);
    cuMemAlloc(&d_ln, (size_t)N*D*4);
    cuMemAlloc(&d_qkv, (size_t)N*3*D*4);
    cuMemAlloc(&d_attn, (size_t)N*D*4);
    cuMemAlloc(&d_tmp, (size_t)N*D*4);
    cuMemAlloc(&d_mlp, (size_t)N*MLP*4);

    cuMemAlloc(&d_w, (size_t)D*4); cuMemcpyHtoD(d_w, ln1_w, (size_t)D*4);
    cuMemAlloc(&d_b, (size_t)D*4); cuMemcpyHtoD(d_b, ln1_b, (size_t)D*4);
    float eps = 1e-6f;
    int n_tok = N, dim = D;
    void *args_ln1[] = { &d_ln, &d_x, &d_w, &d_b, &n_tok, &dim, &eps };
    cuLaunchKernel(fn_ln, (unsigned)N, 1, 1, (unsigned)D, 1, 1, 0, 0, args_ln1, 0);
    cuCtxSynchronize();
    if (ref_ln1) {
        float *tmp = (float *)malloc((size_t)N * D * 4);
        cuMemcpyDtoH(tmp, d_ln, (size_t)N * D * 4);
        diff_stats("ln1", tmp, ref_ln1, (size_t)N * D);
        free(tmp);
    }

    CUdeviceptr d_qkv_w,d_qkv_b;
    cuMemAlloc(&d_qkv_w, (size_t)3*D*D*4); cuMemcpyHtoD(d_qkv_w, qkv_w, (size_t)3*D*D*4);
    cuMemAlloc(&d_qkv_b, (size_t)3*D*4); cuMemcpyHtoD(d_qkv_b, qkv_b, (size_t)3*D*4);
    int dout = 3*D, din = D;
    void *args_qkv[] = { &d_qkv, &d_ln, &d_qkv_w, &d_qkv_b, &n_tok, &din, &dout };
    cuLaunchKernel(fn_lin, (unsigned)N, (unsigned)((dout + 255)/256), 1, 256,1,1,0,0,args_qkv,0);

    CUdeviceptr d_proj_w,d_proj_b;
    cuMemAlloc(&d_proj_w, (size_t)D*D*4); cuMemcpyHtoD(d_proj_w, proj_w, (size_t)D*D*4);
    cuMemAlloc(&d_proj_b, (size_t)D*4); cuMemcpyHtoD(d_proj_b, proj_b, (size_t)D*4);
    float scale = 0.1020620726f;
    void *args_attn[] = { &d_attn, &d_qkv, &d_proj_w, &d_proj_b, &(int){H}, &(int){W}, &(int){D}, &(int){WS}, &scale };
    cuLaunchKernel(fn_attn, (unsigned)NW, 1, 1, (unsigned)T, 1, 1, 0, 0, args_attn, 0);
    cuCtxSynchronize();
    if (ref_attn) {
        float *tmp = (float *)malloc((size_t)N * D * 4);
        cuMemcpyDtoH(tmp, d_attn, (size_t)N * D * 4);
        diff_stats("attn", tmp, ref_attn, (size_t)N * D);
        free(tmp);
    }

    int n_all = N*D;
    void *args_add1[] = { &d_x, &d_attn, &n_all };
    cuLaunchKernel(fn_add, (unsigned)((n_all+255)/256),1,1,256,1,1,0,0,args_add1,0);
    cuCtxSynchronize();
    if (ref_res1) {
        float *tmp = (float *)malloc((size_t)N * D * 4);
        cuMemcpyDtoH(tmp, d_x, (size_t)N * D * 4);
        diff_stats("res1", tmp, ref_res1, (size_t)N * D);
        free(tmp);
    }

    cuMemFree(d_w); cuMemFree(d_b);
    cuMemAlloc(&d_w, (size_t)D*4); cuMemcpyHtoD(d_w, ln2_w, (size_t)D*4);
    cuMemAlloc(&d_b, (size_t)D*4); cuMemcpyHtoD(d_b, ln2_b, (size_t)D*4);
    void *args_ln2[] = { &d_ln, &d_x, &d_w, &d_b, &n_tok, &dim, &eps };
    cuLaunchKernel(fn_ln, (unsigned)N, 1, 1, (unsigned)D, 1, 1, 0, 0, args_ln2, 0);
    cuCtxSynchronize();
    if (ref_ln2) {
        float *tmp = (float *)malloc((size_t)N * D * 4);
        cuMemcpyDtoH(tmp, d_ln, (size_t)N * D * 4);
        diff_stats("ln2", tmp, ref_ln2, (size_t)N * D);
        free(tmp);
    }

    CUdeviceptr d_fc1_w,d_fc1_b,d_fc2_w,d_fc2_b;
    cuMemAlloc(&d_fc1_w, (size_t)MLP*D*4); cuMemcpyHtoD(d_fc1_w, fc1_w, (size_t)MLP*D*4);
    cuMemAlloc(&d_fc1_b, (size_t)MLP*4); cuMemcpyHtoD(d_fc1_b, fc1_b, (size_t)MLP*4);
    int dout1=MLP;
    void *args_fc1[] = { &d_mlp, &d_ln, &d_fc1_w, &d_fc1_b, &n_tok, &din, &dout1 };
    cuLaunchKernel(fn_lin, (unsigned)N, (unsigned)((dout1+255)/256), 1, 256,1,1,0,0,args_fc1,0);
    int n_mlp=N*MLP;
    void *args_g[] = { &d_mlp, &n_mlp };
    cuLaunchKernel(fn_gelu, (unsigned)((n_mlp+255)/256),1,1,256,1,1,0,0,args_g,0);

    cuMemAlloc(&d_fc2_w, (size_t)D*MLP*4); cuMemcpyHtoD(d_fc2_w, fc2_w, (size_t)D*MLP*4);
    cuMemAlloc(&d_fc2_b, (size_t)D*4); cuMemcpyHtoD(d_fc2_b, fc2_b, (size_t)D*4);
    int din2=MLP, dout2=D;
    void *args_fc2[] = { &d_tmp, &d_mlp, &d_fc2_w, &d_fc2_b, &n_tok, &din2, &dout2 };
    cuLaunchKernel(fn_lin, (unsigned)N, (unsigned)((dout2+255)/256), 1, 256,1,1,0,0,args_fc2,0);
    cuCtxSynchronize();
    if (ref_mlp) {
        float *tmp = (float *)malloc((size_t)N * D * 4);
        cuMemcpyDtoH(tmp, d_tmp, (size_t)N * D * 4);
        diff_stats("mlp", tmp, ref_mlp, (size_t)N * D);
        free(tmp);
    }

    void *args_add2[] = { &d_x, &d_tmp, &n_all };
    cuLaunchKernel(fn_add, (unsigned)((n_all+255)/256),1,1,256,1,1,0,0,args_add2,0);
    cuCtxSynchronize();

    float *out = (float *)malloc((size_t)N*D*4);
    cuMemcpyDtoH(out, d_x, (size_t)N*D*4);

    diff_stats("block0", out, yref, (size_t)N * D);

    free(xin); free(yref); free(out);
    free(ref_ln1); free(ref_attn); free(ref_res1); free(ref_ln2); free(ref_mlp);
    cuMemFree(d_x); cuMemFree(d_ln); cuMemFree(d_qkv); cuMemFree(d_attn); cuMemFree(d_tmp); cuMemFree(d_mlp);
    cuMemFree(d_w); cuMemFree(d_b); cuMemFree(d_qkv_w); cuMemFree(d_qkv_b); cuMemFree(d_proj_w); cuMemFree(d_proj_b);
    cuMemFree(d_fc1_w); cuMemFree(d_fc1_b); cuMemFree(d_fc2_w); cuMemFree(d_fc2_b);
    cuModuleUnload(mod); cuCtxDestroy(ctx); safetensors_close(st);
    return 0;
}
