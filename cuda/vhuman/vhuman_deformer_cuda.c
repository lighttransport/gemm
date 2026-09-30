#include "vhuman_deformer_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef VHUMAN_WITH_HIP
#include "../../rdna4/cuda_driver_compat.h"
#else
#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#endif

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
"}\n"
"__device__ int vh_push(float *x, const float *c, float thr) {\n"
"  float vx = x[0] - c[0], vy = x[1] - c[1], vz = x[2] - c[2];\n"
"  float dd = sqrtf(vx * vx + vy * vy + vz * vz);\n"
"  if (dd < thr) { float k = thr / (dd > 1e-12f ? dd : 1e-12f); x[0] = c[0] + vx * k; x[1] = c[1] + vy * k; x[2] = c[2] + vz * k; return 1; }\n"
"  return 0;\n"
"}\n"
"__device__ void vh_xf(const float *S, const float *p, float *o) {\n"
"  for (int a = 0; a < 3; ++a) o[a] = S[a * 4] * p[0] + S[a * 4 + 1] * p[1] + S[a * 4 + 2] * p[2] + S[a * 4 + 3];\n"
"}\n"
"/* post-skinning contact projection: one block per frame (server/vhuman/rig/contacts.py) */\n"
"struct VhCt { const int *eye_ids, *eye_joint, *lip_ids, *pair_u, *pair_l; const float *eye_center, *eye_thr,\n"
"  *sph_thr, *pair_floor; int ne, nl, ns, np, iters; };\n"
"__device__ void vh_solve(float *x, const float *S, const VhCt &c, const float *sc, const float *up) {\n"
"  for (int it = 0; it < c.iters; ++it) {\n"
"    int any = 0;\n"
"    for (int i = threadIdx.x; i < c.ne; i += blockDim.x) {\n"
"      float ec[3]; vh_xf(S + c.eye_joint[i] * 12, c.eye_center + i * 3, ec);\n"
"      any |= vh_push(x + c.eye_ids[i] * 3, ec, c.eye_thr[i]);\n"
"    }\n"
"    __syncthreads();\n"
"    for (int p = threadIdx.x; p < c.np; p += blockDim.x) {\n"
"      float *u = x + c.pair_u[p] * 3, *l = x + c.pair_l[p] * 3;\n"
"      float sep = (u[0] - l[0]) * up[0] + (u[1] - l[1]) * up[1] + (u[2] - l[2]) * up[2];\n"
"      float dl = c.pair_floor[p] - sep;\n"
"      if (dl > 0.f) { dl *= 0.5f; any = 1; for (int a = 0; a < 3; ++a) { u[a] += dl * up[a]; l[a] -= dl * up[a]; } }\n"
"    }\n"
"    __syncthreads();\n"
"    for (int l = threadIdx.x; l < c.nl; l += blockDim.x) {\n"
"      float *q = x + c.lip_ids[l] * 3;\n"
"      for (int pass = 0; pass < 8; ++pass) {\n"
"        int moved = 0;\n"
"        for (int t = 0; t < c.ns; ++t) moved |= vh_push(q, sc + t * 3, c.sph_thr[(size_t)t * c.nl + l]);  /* (T, L): coalesced */\n"
"        any |= moved;\n"
"        if (!moved) break;\n"
"      }\n"
"    }\n"
"    if (!__syncthreads_or(any)) break;   /* an iteration that moves nothing is final */\n"
"  }\n"
"}\n"
"extern \"C\" __global__ void vh_contacts(float *out, const float *__restrict__ skin, int V, int J,\n"
"    const int *eye_ids, const int *eye_joint, const float *eye_center, const float *eye_thr, int ne,\n"
"    const int *lip_ids, const float *sph_center, const int *sph_joint, const float *sph_weight,\n"
"    const float *sph_thr, int nl, int ns, const int *pair_u, const int *pair_l, const float *pair_floor,\n"
"    int np, int head, int jaw, int iters, const int *verts, const int *nbr_ptr, const int *nbr_idx, int nc,\n"
"    int smooth, float *scratch) {\n"
"  __shared__ float sc[512 * 3];\n"
"  __shared__ float up[3];\n"
"  const float *S = skin + (size_t)blockIdx.x * J * 12;\n"
"  float *x = out + (size_t)blockIdx.x * V * 3;\n"
"  float *x0 = scratch + (size_t)blockIdx.x * nc * 6, *dd = x0 + (size_t)nc * 3;\n"
"  for (int t = threadIdx.x; t < ns; t += blockDim.x) {\n"
"    float M[12];\n"
"    for (int e = 0; e < 12; ++e) M[e] = 0.f;\n"
"    for (int k = 0; k < 2; ++k) { float w = sph_weight[t * 2 + k]; if (w == 0.f) continue;\n"
"      const float *Sj = S + sph_joint[t * 2 + k] * 12; for (int e = 0; e < 12; ++e) M[e] += w * Sj[e]; }\n"
"    vh_xf(M, sph_center + t * 3, sc + t * 3);\n"
"  }\n"
"  if (threadIdx.x == 0) {\n"
"    const float *H = S + head * 12, *Jw = S + jaw * 12;\n"
"    float u0 = H[1] + Jw[1], u1 = H[5] + Jw[5], u2 = H[9] + Jw[9], n = sqrtf(u0 * u0 + u1 * u1 + u2 * u2);\n"
"    up[0] = u0 / n; up[1] = u1 / n; up[2] = u2 / n;\n"
"  }\n"
"  for (int i = threadIdx.x; i < nc; i += blockDim.x)\n"
"    for (int a = 0; a < 3; ++a) x0[i * 3 + a] = x[verts[i] * 3 + a];\n"
"  __syncthreads();\n"
"  VhCt c = {eye_ids, eye_joint, lip_ids, pair_u, pair_l, eye_center, eye_thr, sph_thr, pair_floor, ne, nl, ns, np, iters};\n"
"  vh_solve(x, S, c, sc, up);\n"
"  if (!nc) return;\n"
"  for (int step = 0; step < smooth; ++step) {\n"
"    __syncthreads();\n"
"    for (int i = threadIdx.x; i < nc; i += blockDim.x)\n"
"      for (int a = 0; a < 3; ++a) dd[i * 3 + a] = x[verts[i] * 3 + a] - x0[i * 3 + a];\n"
"    __syncthreads();\n"
"    for (int i = threadIdx.x; i < nc; i += blockDim.x) {\n"
"      int b = nbr_ptr[i], e = nbr_ptr[i + 1];\n"
"      float m[3] = {0.f, 0.f, 0.f};\n"
"      for (int k = b; k < e; ++k) for (int a = 0; a < 3; ++a) m[a] += dd[nbr_idx[k] * 3 + a];\n"
"      for (int a = 0; a < 3; ++a) { float di = dd[i * 3 + a], mean = e > b ? m[a] / (float)(e - b) : di;\n"
"        x[verts[i] * 3 + a] = x0[i * 3 + a] + 0.5f * (di + mean); }\n"
"    }\n"
"  }\n"
"  __syncthreads();\n"
"  vh_solve(x, S, c, sc, up);\n"
"}\n";

struct vh_gpu {
    vh_deformer *d;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn, fn_ct;
    int has_ct;
    CUdeviceptr c_eye_ids, c_eye_joint, c_eye_center, c_eye_thr, c_lip_ids, c_sph_center, c_sph_joint, c_sph_weight,
        c_sph_thr, c_pair_u, c_pair_l, c_pair_floor, c_verts, c_nbr_ptr, c_nbr_idx, c_scratch;
    int nc;
    CUstream stream;
    CUdeviceptr rest, morph, sj, sw, w, skin, out;
    size_t cap_frames, V, M, J, C;
    float *h_w, *h_skin, *h_scratch;
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
    if (cuDeviceGet(&dev, device) != CUDA_SUCCESS) { free(g); return NULL; }
    cuDeviceGetName(g->name, sizeof(g->name), dev);
    if (cuCtxCreate(&g->ctx, 0, dev) != CUDA_SUCCESS) { free(g); return NULL; }
    if (cu_compile_kernels(&g->mod, dev, kSource, "vhuman_deformer", verbose, "vhuman_deformer") < 0 ||
        cuModuleGetFunction(&g->fn, g->mod, "vh_deform") != CUDA_SUCCESS) {
        vh_gpu_free(g);
        return NULL;
    }
    cuStreamCreate(&g->stream, CU_STREAM_NON_BLOCKING);
    const vh_contacts *c = vh_deformer_contacts(d);
    if (c && c->ns <= 512 && cuModuleGetFunction(&g->fn_ct, g->mod, "vh_contacts") == CUDA_SUCCESS) {
        g->c_eye_ids = cu_upload_raw(c->eye_ids, c->ne * 4);
        g->c_eye_joint = cu_upload_raw(c->eye_joint, c->ne * 4);
        g->c_eye_center = cu_upload_raw(c->eye_center, c->ne * 12);
        g->c_eye_thr = cu_upload_raw(c->eye_thr, c->ne * 4);
        g->c_lip_ids = cu_upload_raw(c->lip_ids, c->nl * 4);
        g->c_sph_center = cu_upload_raw(c->sph_center, c->ns * 12);
        g->c_sph_joint = cu_upload_raw(c->sph_joint, c->ns * 8);
        g->c_sph_weight = cu_upload_raw(c->sph_weight, c->ns * 8);
        float *thr_t = malloc(c->nl * c->ns * 4);          /* transposed to (T, L) */
        if (thr_t) {
            for (size_t l = 0; l < c->nl; ++l)
                for (size_t t = 0; t < c->ns; ++t) thr_t[t * c->nl + l] = c->sph_thr[l * c->ns + t];
            g->c_sph_thr = cu_upload_raw(thr_t, c->nl * c->ns * 4);
            free(thr_t);
        }
        g->c_pair_u = cu_upload_raw(c->pair_u, c->np * 4);
        g->c_pair_l = cu_upload_raw(c->pair_l, c->np * 4);
        g->c_pair_floor = cu_upload_raw(c->pair_floor, c->np * 4);
        if (c->nc) {
            g->nc = (int)c->nc;
            g->c_verts = cu_upload_raw(c->verts, c->nc * 4);
            g->c_nbr_ptr = cu_upload_raw(c->nbr_ptr, (c->nc + 1) * 4);
            g->c_nbr_idx = cu_upload_raw(c->nbr_idx, (size_t)c->nbr_ptr[c->nc] * 4 + 4);
        }
        g->has_ct = 1;
    }
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
    CU_FREE(g->w); CU_FREE(g->skin); CU_FREE(g->out); CU_FREE(g->c_scratch);
    free(g->h_w); free(g->h_skin);
    size_t cap = frames < 64 ? 64 : frames;
    if (cuMemAlloc(&g->w, cap * g->M * 4) != CUDA_SUCCESS || cuMemAlloc(&g->skin, cap * g->J * 48) != CUDA_SUCCESS ||
        cuMemAlloc(&g->out, cap * g->V * 12) != CUDA_SUCCESS) return -1;
    if (g->nc && cuMemAlloc(&g->c_scratch, cap * (size_t)g->nc * 24) != CUDA_SUCCESS) return -1;
    g->h_w = malloc(cap * g->M * 4);
    g->h_skin = malloc(cap * g->J * 48);
    free(g->h_scratch);
    g->h_scratch = malloc(vh_deformer_prepare_scratch(g->d, cap) * 4 + 4);

    if (!g->h_w || !g->h_skin || !g->h_scratch) return -1;
    g->cap_frames = cap;
    return 0;
}

int vh_gpu_eval_batch(vh_gpu *g, const float *controls, size_t frames, int use_ml, float *out, double *ms4) {
    if (!frames) return 0;
    cuCtxSetCurrent(g->ctx);
    if (ensure(g, frames)) return -1;
    double t0 = now_ms();
    vh_deformer_prepare_batch(g->d, controls, frames, use_ml, g->h_w, g->h_skin, g->h_scratch);
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
    int iters = vh_deformer_contact_iterations(g->d);
    if (g->has_ct && iters > 0) {
        const vh_contacts *c = vh_deformer_contacts(g->d);
        int ne = (int)c->ne, nl = (int)c->nl, ns = (int)c->ns, np_ = (int)c->np, head = c->up_joints[0],
            jaw = c->up_joints[1], smooth = VH_CONTACT_SMOOTH_STEPS;
        void *cargs[] = {&g->out, &g->skin, &V, &J, &g->c_eye_ids, &g->c_eye_joint, &g->c_eye_center, &g->c_eye_thr, &ne,
                         &g->c_lip_ids, &g->c_sph_center, &g->c_sph_joint, &g->c_sph_weight, &g->c_sph_thr, &nl, &ns,
                         &g->c_pair_u, &g->c_pair_l, &g->c_pair_floor, &np_, &head, &jaw, &iters,
                         &g->c_verts, &g->c_nbr_ptr, &g->c_nbr_idx, &g->nc, &smooth, &g->c_scratch};
        if (r == CUDA_SUCCESS) r = cuLaunchKernel(g->fn_ct, (unsigned)frames, 1, 1, 256, 1, 1, 0, g->stream, cargs, NULL);
    }
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
    CU_FREE(g->c_eye_ids); CU_FREE(g->c_eye_joint); CU_FREE(g->c_eye_center); CU_FREE(g->c_eye_thr);
    CU_FREE(g->c_lip_ids); CU_FREE(g->c_sph_center); CU_FREE(g->c_sph_joint); CU_FREE(g->c_sph_weight);
    CU_FREE(g->c_sph_thr); CU_FREE(g->c_pair_u); CU_FREE(g->c_pair_l); CU_FREE(g->c_pair_floor);
    CU_FREE(g->c_verts); CU_FREE(g->c_nbr_ptr); CU_FREE(g->c_nbr_idx); CU_FREE(g->c_scratch);
    free(g->h_w); free(g->h_skin); free(g->h_scratch);
    if (g->stream) cuStreamDestroy(g->stream);
    if (g->mod) cuModuleUnload(g->mod);
    if (g->ctx) cuCtxDestroy(g->ctx);
    free(g);
}
