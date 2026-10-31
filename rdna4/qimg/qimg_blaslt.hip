// SPDX-License-Identifier: MIT
// hipBLASLt plugin for the native Edit-2511 runner (opt-in QIMG_BLASLT=1): Y[M,N] f32 = X[M,K] bf16 . dequant(W)^T,
// W an INT4-g64 (nibbles, even k low) or INT8-g64 weight with a BF16 scale per (row, 64-group). The weight is
// dequantized to a BF16 scratch (the same RNE values the tiled loaders produce) and multiplied by hipBLASLt,
// which reaches 100-150 TF on the DiT shapes against 65-82 for the tiled WMMA kernel.
//
// Algorithm choice is tuned once per (N, K, M): each of the top heuristic candidates runs twice, candidates whose
// two results differ bit-wise are dropped (split-K with atomics), and the fastest survivor's heuristic rank is
// stored in the cache file (QIMG_BLASLT_CACHE, default <repo>/tmp/qimg-blaslt-algos.txt), so later runs use the
// same algorithm and outputs stay reproducible across runs.
//
// The runner dlopens this library; both must resolve to one HIP runtime instance (the runner checks).
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

__device__ __forceinline__ unsigned short qb_bf16(float f) {
    unsigned u = __float_as_uint(f);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

/* Wb[n, k] = bf16(q[n, k] * scale[n, k / 64]); 8 elements per thread. */
extern "C" __global__ void qb_dequant_i4(unsigned short *__restrict__ Wb, const unsigned char *__restrict__ Wq,
                                         const unsigned short *__restrict__ S, int K, long n8) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    long e = i * 8; long n = e / K; int k = (int)(e - n * K);
    unsigned word = *(const unsigned *)(Wq + (e >> 1));
    float s = __uint_as_float((unsigned)S[n * (K / 64) + k / 64] << 16);
    unsigned short o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) { int nb = (word >> (4 * j)) & 0xF; o[j] = qb_bf16((float)(nb >= 8 ? nb - 16 : nb) * s); }
    *(uint4 *)(Wb + e) = *(const uint4 *)o;
}
extern "C" __global__ void qb_dequant_i8(unsigned short *__restrict__ Wb, const unsigned char *__restrict__ Wq,
                                         const unsigned short *__restrict__ S, int K, long n8) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    long e = i * 8; long n = e / K; int k = (int)(e - n * K);
    uint2 v = *(const uint2 *)(Wq + e);
    float s = __uint_as_float((unsigned)S[n * (K / 64) + k / 64] << 16);
    unsigned short o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = qb_bf16((float)(signed char)(((j < 4 ? v.x : v.y) >> (8 * (j & 3))) & 0xFF) * s);
    *(uint4 *)(Wb + e) = *(const uint4 *)o;
}

namespace {
struct Plan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    size_t ws = 0;
};
struct Ctx {
    hipblasLtHandle_t h = nullptr;
    std::map<std::tuple<int, int, int>, Plan> plans;
    std::map<std::tuple<int, int, int>, int> ranks;   // from the cache file
    std::string cache;
    void *wb = nullptr; size_t wb_cap = 0;
    void *ws = nullptr; size_t ws_cap = 0;
    float *probe = nullptr; size_t probe_cap = 0;
};
const size_t kMaxWs = 64u << 20;

std::string default_cache() {
    if (const char *p = getenv("QIMG_BLASLT_CACHE")) return p;
    Dl_info info;   // <repo>/rdna4/qimg/libqimg_blaslt.so -> <repo>/tmp/qimg-blaslt-algos.txt
    if (dladdr((void *)default_cache, &info) && info.dli_fname) {
        std::string f = info.dli_fname; size_t s = f.rfind('/');
        if (s != std::string::npos) return f.substr(0, s) + "/../../tmp/qimg-blaslt-algos.txt";
    }
    return "qimg-blaslt-algos.txt";
}
int grow(void **p, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    if (*p) { hipDeviceSynchronize(); hipFree(*p); *p = nullptr; *cap = 0; }
    if (hipMalloc(p, need) != hipSuccess) return -1;
    *cap = need; return 0;
}
}  // namespace

#define QB_CHECK(x) do { hipblasStatus_t st_ = (x); if (st_ != HIPBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "qimg_blaslt: %s -> %d\n", #x, (int)st_); return 1000 + (int)st_; } } while (0)

extern "C" int qimg_blaslt_create(void **out) {
    Ctx *c = new Ctx();
    if (hipblasLtCreate(&c->h) != HIPBLAS_STATUS_SUCCESS) { delete c; return -1; }
    c->cache = default_cache();
    if (FILE *f = fopen(c->cache.c_str(), "r")) {
        int n, k, m, r;
        while (fscanf(f, "%d %d %d %d", &n, &k, &m, &r) == 4) c->ranks[{n, k, m}] = r;
        fclose(f);
    }
    *out = c; return 0;
}

extern "C" void qimg_blaslt_destroy(void *ctx) {
    Ctx *c = (Ctx *)ctx; if (!c) return;
    hipDeviceSynchronize();
    for (auto &kv : c->plans) {
        Plan &p = kv.second;
        if (p.c) hipblasLtMatrixLayoutDestroy(p.c);
        if (p.b) hipblasLtMatrixLayoutDestroy(p.b);
        if (p.a) hipblasLtMatrixLayoutDestroy(p.a);
        if (p.desc) hipblasLtMatmulDescDestroy(p.desc);
    }
    hipFree(c->wb); hipFree(c->ws); hipFree(c->probe);
    if (c->h) hipblasLtDestroy(c->h);
    delete c;
}

/* Builds (and on first sight tunes) the plan for Y[M,N] f32 = X[M,K] bf16 . W[N,K]^T bf16. */
static int qb_plan(Ctx *c, int N, int K, int M, const void *W, const void *X, float *Y, Plan **out) {
    auto key = std::make_tuple(N, K, M);
    auto it = c->plans.find(key);
    if (it != c->plans.end()) { *out = &it->second; return 0; }
    Plan p;
    QB_CHECK(hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;   // column-major: Y^T[N,M] = W[K,N]^T . X^T[K,M]
    QB_CHECK(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    QB_CHECK(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    QB_CHECK(hipblasLtMatrixLayoutCreate(&p.a, HIP_R_16BF, K, N, K));
    QB_CHECK(hipblasLtMatrixLayoutCreate(&p.b, HIP_R_16BF, K, M, K));
    QB_CHECK(hipblasLtMatrixLayoutCreate(&p.c, HIP_R_32F, N, M, N));
    hipblasLtMatmulPreference_t pref;
    QB_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t maxws = kMaxWs;
    QB_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxws, sizeof maxws));
    hipblasLtMatmulHeuristicResult_t res[16]; int n = 0;
    QB_CHECK(hipblasLtMatmulAlgoGetHeuristic(c->h, p.desc, p.a, p.b, p.c, p.c, pref, 16, res, &n));
    hipblasLtMatmulPreferenceDestroy(pref);
    if (n <= 0) return -2;
    if (grow(&c->ws, &c->ws_cap, kMaxWs)) return -3;
    const float one = 1.f, zero = 0.f;
    int rank = -1;
    auto cached = c->ranks.find(key);
    if (cached != c->ranks.end() && cached->second < n) rank = cached->second;
    else {
        /* tune: two runs per candidate, keep bit-identical ones, fastest wins */
        size_t ybytes = (size_t)M * N * 4;
        if (grow((void **)&c->probe, &c->probe_cap, ybytes)) return -3;
        std::vector<float> h0(ybytes / 4), h1(ybytes / 4);
        float best = 1e30f;
        for (int i = 0; i < n; i++) {
            if (res[i].workspaceSize > kMaxWs) continue;
            if (hipblasLtMatmul(c->h, p.desc, &one, W, p.a, X, p.b, &zero, Y, p.c, Y, p.c, &res[i].algo, c->ws, kMaxWs, 0)) continue;
            if (hipblasLtMatmul(c->h, p.desc, &one, W, p.a, X, p.b, &zero, c->probe, p.c, c->probe, p.c, &res[i].algo, c->ws, kMaxWs, 0)) continue;
            hipDeviceSynchronize();
            hipMemcpy(h0.data(), Y, ybytes, hipMemcpyDeviceToHost); hipMemcpy(h1.data(), c->probe, ybytes, hipMemcpyDeviceToHost);
            if (memcmp(h0.data(), h1.data(), ybytes)) continue;   // non-deterministic candidate
            hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0);
            for (int r = 0; r < 3; r++) hipblasLtMatmul(c->h, p.desc, &one, W, p.a, X, p.b, &zero, Y, p.c, Y, p.c, &res[i].algo, c->ws, kMaxWs, 0);
            hipEventRecord(e1); hipEventSynchronize(e1); float ms = 0; hipEventElapsedTime(&ms, e0, e1);
            hipEventDestroy(e0); hipEventDestroy(e1);
            if (ms < best) { best = ms; rank = i; }
        }
        if (rank < 0) return -4;
        c->ranks[key] = rank;
        if (FILE *f = fopen(c->cache.c_str(), "a")) { fprintf(f, "%d %d %d %d\n", N, K, M, rank); fclose(f); }
        if (getenv("QIMG_BLASLT_VERBOSE"))
            fprintf(stderr, "qimg_blaslt: tuned N=%d K=%d M=%d -> heuristic #%d of %d (%.3f ms)\n", N, K, M, rank, n, best / 3);
    }
    p.algo = res[rank].algo; p.ws = res[rank].workspaceSize;
    *out = &c->plans.emplace(key, p).first->second;
    return 0;
}

/* Y[M,N] f32 = X[M,K] bf16 . dequant(Wq)^T. bits 4 or 8. Same (null) stream as the runner. */
extern "C" int qimg_blaslt_qgemm(void *ctx, float *Y, const void *X, const unsigned char *Wq, const unsigned short *S,
                                 int bits, int N, int K, int M) {
    Ctx *c = (Ctx *)ctx;
    if (!c || !Y || !X || !Wq || !S || (bits != 4 && bits != 8) || N < 1 || M < 1 || K < 64 || K % 64) return -1;
    if (grow(&c->wb, &c->wb_cap, (size_t)N * K * 2)) return -3;
    long n8 = (long)N * K / 8;
    if (bits == 4) qb_dequant_i4<<<(unsigned)((n8 + 255) / 256), 256>>>((unsigned short *)c->wb, Wq, S, K, n8);
    else           qb_dequant_i8<<<(unsigned)((n8 + 255) / 256), 256>>>((unsigned short *)c->wb, Wq, S, K, n8);
    Plan *p = nullptr;
    int rc = qb_plan(c, N, K, M, c->wb, X, Y, &p);
    if (rc) return rc;
    const float one = 1.f, zero = 0.f;
    QB_CHECK(hipblasLtMatmul(c->h, p->desc, &one, c->wb, p->a, X, p->b, &zero, Y, p->c, Y, p->c, &p->algo, c->ws, p->ws, 0));
    return hipGetLastError() == hipSuccess ? 0 : -5;
}
