#include "vhuman_deformer_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

#define FRAMES_PER_THREAD 8
#define MAX_MORPHS 256
#define MAX_JOINTS 64

static const char *kSource =
"#define FT 8\n"
"extern \"C\" __global__ void vh_deform(const float *__restrict__ rest, const float *__restrict__ morph,\n"
"    const int *__restrict__ sj, const float *__restrict__ sw, const float *__restrict__ weights,\n"
"    const float *__restrict__ skin, float *__restrict__ out, int V, int M, int J, int frames) {\n"
"  extern __shared__ float sh[];            /* FT*M weights, then FT*J*12 skin */\n"
"  int f0 = blockIdx.y * FT;\n"
"  int nf = min(FT, frames - f0);\n"
"  float *sw_ = sh, *sk = sh + FT * M;\n"
"  for (int i = threadIdx.x; i < FT * M; i += blockDim.x) { int f = i / M; sw_[i] = f < nf ? weights[(size_t)(f0 + f) * M + i % M] : 0.f; }\n"
"  for (int i = threadIdx.x; i < FT * J * 12; i += blockDim.x) { int f = i / (J * 12); sk[i] = f < nf ? skin[(size_t)(f0 + f) * J * 12 + i % (J * 12)] : 0.f; }\n"
"  __syncthreads();\n"
"  int v = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (v >= V) return;\n"
"  float px[FT], py[FT], pz[FT];\n"
"  float rx = rest[3 * v], ry = rest[3 * v + 1], rz = rest[3 * v + 2];\n"
"#pragma unroll\n"
"  for (int f = 0; f < FT; ++f) { px[f] = rx; py[f] = ry; pz[f] = rz; }\n"
"  size_t stride = (size_t)V * 3;\n"
"  for (int m = 0; m < M; ++m) {\n"
"    const float *d = morph + m * stride + 3 * v;\n"
"    float dx = d[0], dy = d[1], dz = d[2];\n"
"    if (dx == 0.f && dy == 0.f && dz == 0.f) continue;\n"
"#pragma unroll\n"
"    for (int f = 0; f < FT; ++f) { float w = sw_[f * M + m]; px[f] += w * dx; py[f] += w * dy; pz[f] += w * dz; }\n"
"  }\n"
"  int j[4]; float wt[4];\n"
"#pragma unroll\n"
"  for (int k = 0; k < 4; ++k) { j[k] = sj[4 * v + k]; wt[k] = sw[4 * v + k]; }\n"
"  for (int f = 0; f < nf; ++f) {\n"
"    float mm[12];\n"
"#pragma unroll\n"
"    for (int e = 0; e < 12; ++e) mm[e] = 0.f;\n"
"#pragma unroll\n"
"    for (int k = 0; k < 4; ++k) { const float *S = sk + (f * J + j[k]) * 12;\n"
"#pragma unroll\n"
"      for (int e = 0; e < 12; ++e) mm[e] += wt[k] * S[e]; }\n"
"    float *o = out + ((size_t)(f0 + f) * V + v) * 3;\n"
"    o[0] = mm[0] * px[f] + mm[1] * py[f] + mm[2] * pz[f] + mm[3];\n"
"    o[1] = mm[4] * px[f] + mm[5] * py[f] + mm[6] * pz[f] + mm[7];\n"
"    o[2] = mm[8] * px[f] + mm[9] * py[f] + mm[10] * pz[f] + mm[11];\n"
"  }\n"
"}\n";

struct vh_gpu {
    vh_deformer *d;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUstream stream;
    CUdeviceptr rest, morph, sj, sw, w, skin, out;
    size_t cap_frames, V, M, J, C;
    float *h_w, *h_skin;
    char name[128];
};

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

vh_gpu *vh_gpu_create(vh_deformer *d, int device, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return NULL;
    if (cuInit(0) != CUDA_SUCCESS) return NULL;
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS || device >= count) return NULL;
    vh_gpu *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->d = d;
    g->V = vh_deformer_vertices(d);
    g->M = vh_deformer_morphs(d);
    g->J = vh_deformer_joints(d);
    g->C = vh_deformer_controls(d);
    if (g->M > MAX_MORPHS || g->J > MAX_JOINTS) { free(g); return NULL; }
    CUdevice dev;
    cuDeviceGet(&dev, device);
    cuDeviceGetName(g->name, sizeof(g->name), dev);
    if (cuCtxCreate(&g->ctx, 0, dev) != CUDA_SUCCESS) { free(g); return NULL; }
    if (cu_compile_kernels(&g->mod, dev, kSource, "vhuman_deformer", verbose, "vhuman_deformer") < 0 ||
        cuModuleGetFunction(&g->fn, g->mod, "vh_deform") != CUDA_SUCCESS) {
        vh_gpu_free(g);
        return NULL;
    }
    cuStreamCreate(&g->stream, CU_STREAM_NON_BLOCKING);
    size_t n3 = g->V * 3;
    g->rest = cu_upload_raw(vh_deformer_rest(d), n3 * 4);
    g->morph = cu_upload_raw(vh_deformer_morph(d), g->M * n3 * 4);
    g->sj = cu_upload_raw(vh_deformer_skin_joints(d), g->V * 16);
    g->sw = cu_upload_raw(vh_deformer_skin_weights(d), g->V * 16);
    if (!g->rest || !g->morph || !g->sj || !g->sw) { vh_gpu_free(g); return NULL; }
    return g;
}

const char *vh_gpu_name(const vh_gpu *g) { return g->name; }

static int ensure(vh_gpu *g, size_t frames) {
    if (frames <= g->cap_frames) return 0;
    CU_FREE(g->w); CU_FREE(g->skin); CU_FREE(g->out);
    free(g->h_w); free(g->h_skin);
    size_t cap = frames < 64 ? 64 : frames;
    if (cuMemAlloc(&g->w, cap * g->M * 4) != CUDA_SUCCESS || cuMemAlloc(&g->skin, cap * g->J * 48) != CUDA_SUCCESS ||
        cuMemAlloc(&g->out, cap * g->V * 12) != CUDA_SUCCESS) return -1;
    g->h_w = malloc(cap * g->M * 4);
    g->h_skin = malloc(cap * g->J * 48);
    if (!g->h_w || !g->h_skin) return -1;
    g->cap_frames = cap;
    return 0;
}

int vh_gpu_eval_batch(vh_gpu *g, const float *controls, size_t frames, int use_ml, float *out, double *ms4) {
    if (!frames) return 0;
    cuCtxSetCurrent(g->ctx);
    if (ensure(g, frames)) return -1;
    double t0 = now_ms();
    for (size_t f = 0; f < frames; ++f)
        vh_deformer_prepare(g->d, controls + f * g->C, use_ml, g->h_w + f * g->M, g->h_skin + f * g->J * 12);
    double t1 = now_ms();
    cuMemcpyHtoDAsync(g->w, g->h_w, frames * g->M * 4, g->stream);
    cuMemcpyHtoDAsync(g->skin, g->h_skin, frames * g->J * 48, g->stream);
    CUevent e0, e1;
    cuEventCreate(&e0, 0);
    cuEventCreate(&e1, 0);
    cuStreamSynchronize(g->stream);
    double t2 = now_ms();
    int V = (int)g->V, M = (int)g->M, J = (int)g->J, F = (int)frames;
    void *args[] = {&g->rest, &g->morph, &g->sj, &g->sw, &g->w, &g->skin, &g->out, &V, &M, &J, &F};
    unsigned block = 128;
    unsigned gx = (unsigned)((g->V + block - 1) / block), gy = (unsigned)((frames + FRAMES_PER_THREAD - 1) / FRAMES_PER_THREAD);
    unsigned smem = (unsigned)(FRAMES_PER_THREAD * (g->M + g->J * 12) * 4);
    cuEventRecord(e0, g->stream);
    CUresult r = cuLaunchKernel(g->fn, gx, gy, 1, block, 1, 1, smem, g->stream, args, NULL);
    cuEventRecord(e1, g->stream);
    cuEventSynchronize(e1);
    float kms = 0;
    cuEventElapsedTime(&kms, e0, e1);
    cuEventDestroy(e0);
    cuEventDestroy(e1);
    if (r != CUDA_SUCCESS) return -2;
    double t3 = now_ms();
    if (out) cuMemcpyDtoH(out, g->out, frames * g->V * 12);
    double t4 = now_ms();
    if (ms4) {
        ms4[0] = t1 - t0;
        ms4[1] = t2 - t1;
        ms4[2] = kms;
        ms4[3] = t4 - t3;
    }
    return 0;
}

void vh_gpu_free(vh_gpu *g) {
    if (!g) return;
    if (g->ctx) cuCtxSetCurrent(g->ctx);
    CU_FREE(g->rest); CU_FREE(g->morph); CU_FREE(g->sj); CU_FREE(g->sw);
    CU_FREE(g->w); CU_FREE(g->skin); CU_FREE(g->out);
    free(g->h_w); free(g->h_skin);
    if (g->stream) cuStreamDestroy(g->stream);
    if (g->mod) cuModuleUnload(g->mod);
    if (g->ctx) cuCtxDestroy(g->ctx);
    free(g);
}
