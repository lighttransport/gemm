/* Verified cuBLASLt NVFP4 GEMM on SM120. Packed data and the documented
 * 128x4 block-scale layout are prepared outside CUDA-event timing.
 * Uses the dynamically loaded cuBLASLt ABI; CUDA SDK headers are not required. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

#include "cuew.h"

/* ---- cudaDataType_t (library_types.h) ---- */
typedef enum {
    CUDA_R_32F      = 0,
    CUDA_R_16F      = 2,
    CUDA_R_16BF     = 14,
    CUDA_R_8F_E4M3  = 28,
    CUDA_R_8F_E5M2  = 29,
    CUDA_R_8F_UE8M0 = 30,   /* scale type for MXFP4 */
    CUDA_R_4F_E2M1  = 33,   /* FP4 data */
} cudaDataType_t;

typedef enum {
    CUBLAS_COMPUTE_16F            = 64,
    CUBLAS_COMPUTE_32F            = 68,
    CUBLAS_COMPUTE_32F_FAST_TF32  = 77,
} cublasComputeType_t;

/* block-scale modes (cublasLt.h cublasLtMatmulMatrixScale_t) */
typedef enum {
    CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F   = 0,
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3  = 1,  /* NVFP4 */
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0  = 2,  /* MXFP4 */
} cublasLtMatmulMatrixScale_t;

typedef enum {
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
} cublasLtMatmulPreferenceAttributes_t;

typedef enum {
    CUBLASLT_MATMUL_DESC_TRANSA          = 3,
    CUBLASLT_MATMUL_DESC_TRANSB          = 4,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_A_SCALE_MODE    = 31,
    CUBLASLT_MATMUL_DESC_B_SCALE_MODE    = 32,
} cublasLtMatmulDescAttributes_t;

typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 } cublasOperation_t;

typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;

typedef struct {
    uint64_t algo[8];        /* real cublasLtMatmulAlgo_t is uint64_t[8] (64 B), not a pointer */
    size_t   workspaceSize;
    int32_t  status;
    float    wavesCount;
    int      reserved[4];
} cublasLtMatmulHeuristicResult_t;

typedef int (*cublasLtCreate_t)(cublasLtHandle_t*);
typedef int (*cublasLtDestroy_t)(cublasLtHandle_t);
typedef int (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t);
typedef int (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t);
typedef int (*cublasLtMatmulDescSetAttribute_t)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void*, size_t);
typedef int (*cublasLtMatrixLayoutCreate_t)(cublasLtMatrixLayout_t*, cudaDataType_t, uint64_t, uint64_t, int64_t);
typedef int (*cublasLtMatrixLayoutDestroy_t)(cublasLtMatrixLayout_t);
typedef int (*cublasLtMatmulPreferenceCreate_t)(cublasLtMatmulPreference_t*);
typedef int (*cublasLtMatmulPreferenceDestroy_t)(cublasLtMatmulPreference_t);
typedef int (*cublasLtMatmulPreferenceSetAttribute_t)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void*, size_t);
typedef int (*cublasLtMatmulAlgoGetHeuristic_t)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t*, int*);
typedef int (*cublasLtMatmul_t)(cublasLtHandle_t, cublasLtMatmulDesc_t,
    const void*, const void*, cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t,
    const void*, const void*, cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t,
    const void*, void*, size_t, CUstream);

static cublasLtCreate_t                      p_cublasLtCreate;
static cublasLtDestroy_t                     p_cublasLtDestroy;
static cublasLtMatmulDescCreate_t            p_cublasLtMatmulDescCreate;
static cublasLtMatmulDescDestroy_t           p_cublasLtMatmulDescDestroy;
static cublasLtMatmulDescSetAttribute_t      p_cublasLtMatmulDescSetAttribute;
static cublasLtMatrixLayoutCreate_t          p_cublasLtMatrixLayoutCreate;
static cublasLtMatrixLayoutDestroy_t         p_cublasLtMatrixLayoutDestroy;
static cublasLtMatmulPreferenceCreate_t      p_cublasLtMatmulPreferenceCreate;
static cublasLtMatmulPreferenceDestroy_t     p_cublasLtMatmulPreferenceDestroy;
static cublasLtMatmulPreferenceSetAttribute_t p_cublasLtMatmulPreferenceSetAttribute;
static cublasLtMatmulAlgoGetHeuristic_t      p_cublasLtMatmulAlgoGetHeuristic;
static cublasLtMatmul_t                      p_cublasLtMatmul;

/* Lower-level search ABI, matching the installed cublasLt.h. */
typedef int (*algo_ids_fn)(void *, int, int, int, int, int, int, int, int *, int *);
typedef int (*algo_init_fn)(void *, int, int, int, int, int, int, int, void *);
typedef int (*algo_attr_fn)(const void *, int, void *, size_t, size_t *);
typedef int (*algo_set_fn)(void *, int, const void *, size_t);
typedef int (*algo_check_fn)(void *, void *, void *, void *, void *, void *, const void *,
                             cublasLtMatmulHeuristicResult_t *);
static algo_ids_fn algo_ids;
static algo_init_fn algo_init;
static algo_attr_fn algo_cap, algo_get;
static algo_set_fn algo_set;
static algo_check_fn algo_check;

static void* cublas_lib = NULL;

static int load_cublas(void) {
    const char* names[] = { "libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so", NULL };
    for (int i = 0; names[i]; i++) {
        cublas_lib = dlopen(names[i], RTLD_NOW);
        if (cublas_lib) { printf("Loaded cuBLAS: %s\n", names[i]); break; }
    }
    if (!cublas_lib) { fprintf(stderr, "Failed to load cuBLASLt\n"); return -1; }
    #define LOAD_FUNC(n) p_##n = (n##_t)dlsym(cublas_lib, #n); \
        if (!p_##n) { fprintf(stderr, "missing %s\n", #n); return -1; }
    LOAD_FUNC(cublasLtCreate); LOAD_FUNC(cublasLtDestroy);
    LOAD_FUNC(cublasLtMatmulDescCreate); LOAD_FUNC(cublasLtMatmulDescDestroy);
    LOAD_FUNC(cublasLtMatmulDescSetAttribute);
    LOAD_FUNC(cublasLtMatrixLayoutCreate); LOAD_FUNC(cublasLtMatrixLayoutDestroy);
    LOAD_FUNC(cublasLtMatmulPreferenceCreate); LOAD_FUNC(cublasLtMatmulPreferenceDestroy);
    LOAD_FUNC(cublasLtMatmulPreferenceSetAttribute);
    LOAD_FUNC(cublasLtMatmulAlgoGetHeuristic); LOAD_FUNC(cublasLtMatmul);
    #define LOAD_SEARCH(dst, name) do { void *s = dlsym(cublas_lib,name); \
        memcpy(&dst,&s,sizeof(dst)); if (!dst) return -1; } while (0)
    LOAD_SEARCH(algo_ids,"cublasLtMatmulAlgoGetIds");
    LOAD_SEARCH(algo_init,"cublasLtMatmulAlgoInit");
    LOAD_SEARCH(algo_cap,"cublasLtMatmulAlgoCapGetAttribute");
    LOAD_SEARCH(algo_get,"cublasLtMatmulAlgoConfigGetAttribute");
    LOAD_SEARCH(algo_set,"cublasLtMatmulAlgoConfigSetAttribute");
    LOAD_SEARCH(algo_check,"cublasLtMatmulAlgoCheck");
    #undef LOAD_SEARCH
    #undef LOAD_FUNC
    return 0;
}

#include "../int4/bench_common.h"
#include <errno.h>
#include <limits.h>

typedef struct {
    int m, n, k, warmup, iters, rounds, algos, workspace_mb, full_check, skip_verify;
    const char *format, *output, *data, *csv;
    int exhaustive, candidate;
    const char *cutlass;
    int raster, swizzle;
    int compare_cublas, graph, flush_mb, pilot_iters;
} options;
typedef struct {
    const char *name;
    cudaDataType_t type;
    int block, scale_mode, codebook;
} config;
static const config configs[] = {
    {"bf16",CUDA_R_16BF,0,-1,0}, {"fp8",CUDA_R_8F_E4M3,0,0,0},
    {"nvfp4",CUDA_R_4F_E2M1,16,1,0}, {"mxfp4",CUDA_R_4F_E2M1,32,2,0},
    {"e0m3",CUDA_R_4F_E2M1,16,1,1}
};

static int round_up(int x, int q) { return (x + q - 1) / q * q; }
static size_t data_bytes(size_t n, int type) {
    return type == CUDA_R_4F_E2M1 ? n / 2 : type == CUDA_R_8F_E4M3 ? n : n * 2;
}
static size_t scale_offset(int row, int g, int groups) {
    return (size_t)(row / 128) * 128 * round_up(groups,4) + (size_t)(g / 4) * 512 +
        (row % 32) * 16 + ((row % 128) / 32) * 4 + g % 4;
}
static float read_input(const uint8_t *p, const uint8_t *scales, const config *c,
                        int row, int k, int depth) {
    size_t i = (size_t)row * depth + k;
    if (c->type == CUDA_R_16BF) {
        uint32_t bits = (uint32_t)((const uint16_t *)p)[i] << 16;
        float v; memcpy(&v,&bits,4); return v;
    }
    if (c->type == CUDA_R_8F_E4M3) {
        unsigned x = p[i]; float v = int4_scale_decode(x & 127);
        return x & 128 ? -v : v;
    }
    unsigned code = (p[i / 2] >> (4 * (i & 1))) & 15;
    unsigned sc = scales[scale_offset(row,k / c->block,depth / c->block)];
    float scale = c->block == 16 ? int4_scale_decode(sc) : ldexpf(1.0f,(int)sc - 127);
    return int4_code_decode(code,c->codebook) * scale;
}
static void make_input(const config *c, int rows, int depth, int pattern,
                       uint32_t *rng, CUdeviceptr *device, CUdeviceptr *scale_device,
                       uint8_t **host, uint8_t **scales) {
    size_t count = (size_t)rows * depth, bytes = data_bytes(count,c->type);
    int random = pattern == 1;
    *host = host_alloc(bytes); *scales = NULL;
    for (size_t i = 0; i < count; i++) {
        uint32_t r = random ? random_u32(rng) : 0;
        if (c->type == CUDA_R_4F_E2M1) {
            unsigned code = pattern == 2 ? (i / depth) % 16 : random ? r & 15 : c->codebook ? 1 : 2;
            if (!(i & 1)) (*host)[i / 2] = code;
            else (*host)[i / 2] |= code << 4;
        } else if (c->type == CUDA_R_8F_E4M3) {
            (*host)[i] = random ? (0x20 + r % 40) | ((r >> 8) & 128) : 0x38;
        } else {
            float v = random ? ((int)(r % 15) - 7) * 0.25f : 1.0f;
            uint32_t bits; memcpy(&bits,&v,4); ((uint16_t *)*host)[i] = bits >> 16;
        }
    }
    *device = device_alloc(bytes); CUDA(cuMemcpyHtoD(*device,*host,bytes));
    *scale_device = 0;
    if (c->block) {
        int groups = depth / c->block;
        size_t ns = (size_t)round_up(rows,128) * round_up(groups,4);
        *scales = host_alloc(ns); memset(*scales,0,ns);
        static const uint8_t e4[] = {0x28,0x30,0x34,0x38,0x3c,0x40};
        for (int row = 0; row < rows; row++) for (int g = 0; g < groups; g++)
            (*scales)[scale_offset(row,g,groups)] = c->block == 16 ?
                (pattern == 2 ? e4[(row + g) % 6] : random ? e4[random_u32(rng) % 6] : 0x38) :
                (random ? 125 + random_u32(rng) % 5 : 127);
        *scale_device = device_alloc(ns); CUDA(cuMemcpyHtoD(*scale_device,*scales,ns));
    }
}
/* E0M3 x UE4M3 products are exactly representable in BF16. This independent
 * decoded-input reference is prepared outside every GEMM timing interval. */
static CUdeviceptr decoded_bf16(const uint8_t *codes, const uint8_t *scales,
                                const config *c, int rows, int depth) {
    size_t count = (size_t)rows * depth;
    uint16_t *host = host_alloc(count * 2);
    for (int row = 0; row < rows; row++) for (int k = 0; k < depth; k++) {
        float v = read_input(codes,scales,c,row,k,depth);
        uint32_t bits; memcpy(&bits,&v,4);
        REQUIRE(!(bits & 65535)); host[(size_t)row * depth + k] = bits >> 16;
    }
    CUdeviceptr device = device_alloc(count * 2);
    CUDA(cuMemcpyHtoD(device,host,count * 2)); free(host); return device;
}
static float read_output(const void *raw, size_t i, int type) {
    if (type == CUDA_R_32F) { float v; memcpy(&v,(const uint8_t *)raw + i * 4,4); return v; }
    uint16_t half; memcpy(&half,(const uint8_t *)raw + i * 2,2);
    unsigned h = half;
    if (type == CUDA_R_16BF) { uint32_t u = h << 16; float v; memcpy(&v,&u,4); return v; }
    unsigned exp = (h >> 10) & 31, mant = h & 1023;
    float v = exp == 31 ? (mant ? NAN : INFINITY) : exp ?
        ldexpf(1.0f + mant / 1024.0f,(int)exp - 15) : ldexpf((float)mant,-24);
    return h & 0x8000 ? -v : v;
}
typedef struct {
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t a_layout, b_layout, out_layout;
    CUdeviceptr a, b, output, workspace;
    CUdeviceptr decoded_a, decoded_b;
    size_t workspace_bytes;
    cublasLtMatmulHeuristicResult_t *algorithm;
    void *custom;
    int (*custom_run)(void *);
    int (*custom_run_stream)(void *, void *);
    CUstream stream;
    CUdeviceptr flush;
    int graph, flush_mb;
} job;

static int algo_value(const void *a, int attribute) {
    int value = -1; size_t written = 0;
    if (algo_get(a,attribute,&value,sizeof(value),&written)) return -1;
    return value;
}
static void print_algo(FILE *f, const void *a) {
    fprintf(f,"id=%d tile=%d splitk=%d reduction=%d swizzle=%d custom=%d stages=%d",
        algo_value(a,0),algo_value(a,1),algo_value(a,2),algo_value(a,3),
        algo_value(a,4),algo_value(a,5),algo_value(a,6));
}
static void add_algo(job *j, cublasLtMatmulHeuristicResult_t *pool, int *count,
                      const cublasLtMatmulHeuristicResult_t *a) {
    if (a->status || a->workspaceSize > j->workspace_bytes) return;
    for (int i = 0; i < *count; i++) if (!memcmp(pool[i].algo,a->algo,sizeof(a->algo))) return;
    REQUIRE(*count < 16384); pool[(*count)++] = *a;
}
static int cap_values(const void *a, int attr, unsigned *values, int capacity) {
    size_t written = 0;
    REQUIRE(!algo_cap(a,attr,NULL,0,&written));
    REQUIRE(written <= (size_t)capacity * sizeof(*values));
    if (!written) { values[0] = 0; return 1; }
    REQUIRE(!algo_cap(a,attr,values,written,&written));
    return written / sizeof(*values);
}
static void exhaustive_algos(job *j, const config *c, int output,
                              cublasLtMatmulHeuristicResult_t *pool, int *count) {
    int heuristics = *count;
    int ids[128], num_ids = 0;
    REQUIRE(!algo_ids(j->handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,c->type,c->type,
        output,output,128,ids,&num_ids));
    for (int id = 0; id < num_ids; id++) {
        cublasLtMatmulHeuristicResult_t base = {0};
        if (algo_init(j->handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,c->type,c->type,
            output,output,ids[id],base.algo)) continue;
        unsigned tiles[1024], stages[256]; int custom = 0, swizzle = 0; size_t written;
        int nt = cap_values(base.algo,6,tiles,1024), ns = cap_values(base.algo,13,stages,256);
        REQUIRE(!algo_cap(base.algo,7,&custom,sizeof(custom),&written));
        REQUIRE(!algo_cap(base.algo,2,&swizzle,sizeof(swizzle),&written));
        printf("Search algorithm %d: %d tiles, %d stages, custom=0..%d swizzle=0..%d\n",
            ids[id],nt,ns,custom,swizzle);
        REQUIRE(custom < 256 && swizzle < 256);
        for (int t = 0; t < nt; t++) for (int s = 0; s < ns; s++)
            for (int x = 0; x <= custom; x++) for (int w = 0; w <= swizzle; w++) {
                cublasLtMatmulHeuristicResult_t a = base;
                if (algo_set(a.algo,1,&tiles[t],4) || algo_set(a.algo,6,&stages[s],4) ||
                    algo_set(a.algo,5,&x,4) || algo_set(a.algo,4,&w,4)) continue;
                if (!algo_check(j->handle,j->desc,j->a_layout,j->b_layout,j->out_layout,
                    j->out_layout,a.algo,&a)) add_algo(j,pool,count,&a);
            }
    }
    /* Some families expose only UNDEFINED in their capability lists, while
     * their heuristic descriptors carry explicit tiles/stages. Check common
     * documented configurations as well; only AlgoCheck successes can run. */
    unsigned tiles[] = {17,18,20,23,24};
    unsigned stages[] = {20,21,22,23,24,35,36,37};
    for (int h = 0; h < heuristics; h++)
        for (int t = 0; t < 5; t++) for (int s = 0; s < 8; s++) for (unsigned w = 0; w < 2; w++) {
            cublasLtMatmulHeuristicResult_t a = pool[h];
            if (algo_set(a.algo,1,&tiles[t],4) || algo_set(a.algo,6,&stages[s],4) ||
                algo_set(a.algo,4,&w,4)) continue;
            if (!algo_check(j->handle,j->desc,j->a_layout,j->b_layout,j->out_layout,
                j->out_layout,a.algo,&a)) add_algo(j,pool,count,&a);
        }
}
static int launch(job *j) {
    if (j->custom_run) return j->stream ? j->custom_run_stream(j->custom,j->stream) : j->custom_run(j->custom);
    float alpha = 1, beta = 0;
    return p_cublasLtMatmul(j->handle,j->desc,&alpha,
        (void *)(uintptr_t)j->b,j->a_layout,(void *)(uintptr_t)j->a,j->b_layout,&beta,
        (void *)(uintptr_t)j->output,j->out_layout,(void *)(uintptr_t)j->output,j->out_layout,
        &j->algorithm->algo,(void *)(uintptr_t)j->workspace,j->workspace_bytes,j->stream);
}
static float time_job(job *j, int iters) {
    CUevent begin, end; CUDA(cuEventCreate(&begin,0)); CUDA(cuEventCreate(&end,0));
    CUgraph graph = NULL; CUgraphExec executable = NULL;
    if (j->graph) {
        REQUIRE(!j->custom_run || j->custom_run_stream);
        CUDA(cuStreamCreate(&j->stream,CU_STREAM_NON_BLOCKING));
        CUDA(cuStreamBeginCapture(j->stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        if (j->flush_mb) {
            CUDA(cuMemsetD8Async(j->flush,0xa5,(size_t)j->flush_mb << 20,j->stream));
            CUDA(cuEventRecordWithFlags(begin,j->stream,CU_EVENT_RECORD_EXTERNAL));
        }
        int nodes = j->flush_mb ? 1 : iters;
        for (int i = 0; i < nodes; i++) REQUIRE(!launch(j));
        if (j->flush_mb) CUDA(cuEventRecordWithFlags(end,j->stream,CU_EVENT_RECORD_EXTERNAL));
        CUDA(cuStreamEndCapture(j->stream,&graph));
        CUDA(cuGraphInstantiate(&executable,graph,0));
        /* Upload and warm the graph outside timing, including first-use setup. */
        CUDA(cuGraphLaunch(executable,j->stream)); CUDA(cuStreamSynchronize(j->stream));
    }
    float ms = 0;
    int trials = j->flush_mb ? iters : 1;
    for (int t = 0; t < trials; t++) {
        if (j->flush_mb && !executable)
            CUDA(cuMemsetD8Async(j->flush,(unsigned char)t,(size_t)j->flush_mb << 20,j->stream));
        if (!(j->flush_mb && executable)) CUDA(cuEventRecord(begin,j->stream));
        if (executable) CUDA(cuGraphLaunch(executable,j->stream));
        else for (int i = 0; i < (j->flush_mb ? 1 : iters); i++) REQUIRE(!launch(j));
        if (!(j->flush_mb && executable)) CUDA(cuEventRecord(end,j->stream));
        CUDA(cuEventSynchronize(end));
        float elapsed; CUDA(cuEventElapsedTime(&elapsed,begin,end)); ms += elapsed;
    }
    if (executable) {
        CUDA(cuGraphExecDestroy(executable)); CUDA(cuGraphDestroy(graph));
        CUDA(cuStreamDestroy(j->stream)); j->stream = NULL;
    }
    CUDA(cuEventDestroy(begin)); CUDA(cuEventDestroy(end)); return ms / iters;
}
static int compare_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y);
}
static int verify(job *j, const size_t *indices, const double *refs, int samples,
                   int type, int full, size_t count, double *rel, double *max_error) {
    size_t stride = type == CUDA_R_32F ? 4 : 2;
    void *raw = full ? host_alloc(count * stride) : NULL;
    if (full) CUDA(cuMemcpyDtoH(raw,j->output,count * stride));
    double num = 0, den = 0; *max_error = 0;
    for (int i = 0; i < samples; i++) {
        uint32_t bits = 0;
        if (!full) CUDA(cuMemcpyDtoH(&bits,j->output + indices[i] * stride,stride));
        float got = read_output(full ? raw : &bits,full ? indices[i] : 0,type);
        if (!isfinite(got)) { free(raw); return 0; }
        double e = got - refs[i]; num += e * e; den += refs[i] * refs[i];
        *max_error = fmax(*max_error,fabs(e));
    }
    free(raw); *rel = sqrt(num / fmax(den,1e-30));
    return *rel <= (type == CUDA_R_32F ? 1e-4 : type == CUDA_R_16BF ? 0.004 : 0.0005);
}
static void gpu_info(FILE *f) {
    fputs("GPU: name, driver, pstate, SM MHz, max SM MHz, memory MHz, max memory MHz, power limit W, temperature C\n",f);
    FILE *p = popen("nvidia-smi --query-gpu=name,driver_version,pstate,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory,power.limit,temperature.gpu --format=csv,noheader","r");
    if (p) { char line[1024]; while (fgets(line,sizeof(line),p)) fputs(line,f); pclose(p); }
}
static void *cutlass_lib;
static void compare_cublas(job *j, const config *c, int out_type, const options *o,
                            CUdeviceptr sa, CUdeviceptr sb, FILE *meta) {
    job ref = {0}; ref.handle = j->handle; ref.a = j->a; ref.b = j->b;
    int input_type = c->codebook ? CUDA_R_16BF : CUDA_R_4F_E2M1;
    if (c->codebook) { REQUIRE(j->decoded_a && j->decoded_b); ref.a = j->decoded_a; ref.b = j->decoded_b; }
    size_t count = (size_t)o->m * o->n, stride = out_type == CUDA_R_32F ? 4 : 2;
    ref.output = device_alloc(count * stride);
    REQUIRE(!p_cublasLtMatmulDescCreate(&ref.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    int ta = CUBLAS_OP_T, tb = CUBLAS_OP_N, mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
#define REF_ATTR(attr, value) REQUIRE(!p_cublasLtMatmulDescSetAttribute(ref.desc,attr,&value,sizeof(value)))
    REF_ATTR(CUBLASLT_MATMUL_DESC_TRANSA,ta); REF_ATTR(CUBLASLT_MATMUL_DESC_TRANSB,tb);
    if (!c->codebook) {
        REF_ATTR(CUBLASLT_MATMUL_DESC_A_SCALE_MODE,mode); REF_ATTR(CUBLASLT_MATMUL_DESC_B_SCALE_MODE,mode);
        REF_ATTR(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,sb); REF_ATTR(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,sa);
    }
#undef REF_ATTR
    REQUIRE(!p_cublasLtMatrixLayoutCreate(&ref.a_layout,input_type,o->k,o->n,o->k));
    REQUIRE(!p_cublasLtMatrixLayoutCreate(&ref.b_layout,input_type,o->k,o->m,o->k));
    REQUIRE(!p_cublasLtMatrixLayoutCreate(&ref.out_layout,out_type,o->n,o->m,o->n));
    cublasLtMatmulPreference_t pref; REQUIRE(!p_cublasLtMatmulPreferenceCreate(&pref));
    cublasLtMatmulHeuristicResult_t algorithm; int returned = 0;
    REQUIRE(!p_cublasLtMatmulAlgoGetHeuristic(ref.handle,ref.desc,ref.a_layout,ref.b_layout,
        ref.out_layout,ref.out_layout,pref,1,&algorithm,&returned));
    REQUIRE(returned == 1 && !algorithm.status && !algorithm.workspaceSize);
    ref.algorithm = &algorithm;
    CUDA(cuMemsetD8(ref.output,0xff,count * stride)); REQUIRE(!launch(&ref)); CUDA(cuCtxSynchronize());
    void *expected = host_alloc(count * stride), *actual = host_alloc(count * stride);
    CUDA(cuMemcpyDtoH(expected,ref.output,count * stride));
    CUDA(cuMemcpyDtoH(actual,j->output,count * stride));
    size_t mismatches = 0; double num = 0, den = 0, max_error = 0;
    for (size_t i = 0; i < count; i++) {
        float got = read_output(actual,i,out_type), want = read_output(expected,i,out_type);
        REQUIRE(isfinite(got) && isfinite(want));
        double error = (double)got - want;
        num += error * error; den += (double)want * want; max_error = fmax(max_error,fabs(error));
        mismatches += memcmp((uint8_t *)actual + i * stride,(uint8_t *)expected + i * stride,stride) != 0;
    }
    double rel = sqrt(num / fmax(den,1e-30));
    REQUIRE(rel <= (out_type == CUDA_R_32F ? 1e-4 : out_type == CUDA_R_16BF ? 0.004 : 0.0005));
    printf("Full cuBLAS %s comparison: %zu elements, bit_mismatches=%zu rel_L2=%.9g max_error=%.9g PASS\n",
        c->codebook ? "decoded BF16" : "NVFP4",count,mismatches,rel,max_error);
    if (meta) fprintf(meta,"full_cublas_compare input=%s elements=%zu bit_mismatches=%zu rel_l2=%.9g max_error=%.9g PASS\n",
        c->codebook ? "decoded_bf16" : "nvfp4",count,mismatches,rel,max_error);
    free(expected); free(actual);
    REQUIRE(!p_cublasLtMatmulPreferenceDestroy(pref));
    REQUIRE(!p_cublasLtMatrixLayoutDestroy(ref.a_layout));
    REQUIRE(!p_cublasLtMatrixLayoutDestroy(ref.b_layout));
    REQUIRE(!p_cublasLtMatrixLayoutDestroy(ref.out_layout));
    REQUIRE(!p_cublasLtMatmulDescDestroy(ref.desc)); CUDA(cuMemFree(ref.output));
}
static int run_cutlass(job *j, const config *c, int out_type, const char *out_name,
                        const options *o, FILE *csv, FILE *meta,
                        const size_t *indices, const double *refs, int samples,
                        CUdeviceptr sa, CUdeviceptr sb) {
    if (strcmp(c->name,"nvfp4") && !c->codebook) return 0;
    if (!cutlass_lib) cutlass_lib = dlopen(o->cutlass,RTLD_NOW);
    if (!cutlass_lib) { fprintf(stderr,"%s\n",dlerror()); exit(1); }
    int (*count)(void); const char *(*name)(int);
    void *(*create)(int,int,int,int,int,int,void *,void *,void *,void *,void *,int);
    void (*destroy)(void *);
#define LOAD_PLUGIN(dst, symbol) do { void *s = dlsym(cutlass_lib,symbol); \
    memcpy(&dst,&s,sizeof(dst)); REQUIRE(dst); } while (0)
    LOAD_PLUGIN(count,"fp4_kernel_count"); LOAD_PLUGIN(name,"fp4_kernel_name");
    LOAD_PLUGIN(create,"fp4_kernel_create"); LOAD_PLUGIN(destroy,"fp4_kernel_destroy");
    LOAD_PLUGIN(j->custom_run,"fp4_kernel_run");
    void *stream_symbol = dlsym(cutlass_lib,"fp4_kernel_run_stream");
    memcpy(&j->custom_run_stream,&stream_symbol,sizeof(j->custom_run_stream));
    REQUIRE(!o->graph || j->custom_run_stream);
#undef LOAD_PLUGIN
    const char *(*build_info)(void); void *build_symbol = dlsym(cutlass_lib,"fp4_kernel_build");
    memcpy(&build_info,&build_symbol,sizeof(build_info));
    if (build_info) { printf("%s\n",build_info()); if (meta) fprintf(meta,"%s\n",build_info()); }
    void (*set_format)(int); void *format_symbol = dlsym(cutlass_lib,"fp4_kernel_set_format");
    memcpy(&set_format,&format_symbol,sizeof(set_format));
    if (set_format) set_format(c->codebook);
    int num = count(), best = -1; float fastest = INFINITY;
    REQUIRE(o->candidate < num);
    size_t elements = (size_t)o->m * o->n, bytes = elements * (out_type == CUDA_R_32F ? 4 : 2);
    double rel = 0, max_error = 0;
    for (int i = 0; i < num; i++) {
        if (o->candidate >= 0 && i != o->candidate) continue;
        j->custom = create(i,o->raster,o->swizzle,o->m,o->n,o->k,
            (void *)(uintptr_t)j->a,(void *)(uintptr_t)j->b,
            (void *)(uintptr_t)sa,(void *)(uintptr_t)sb,
            (void *)(uintptr_t)j->output,out_type);
        if (!j->custom) continue;
        CUDA(cuMemsetD8(j->output,0xff,bytes)); REQUIRE(!launch(j)); CUDA(cuCtxSynchronize());
        if (!o->skip_verify) REQUIRE(verify(j,indices,refs,samples,out_type,o->full_check,elements,&rel,&max_error));
        float pilot[3]; for (int r = 0; r < 3; r++)
            pilot[r] = time_job(j,o->pilot_iters ? o->pilot_iters : o->graph ? 32 : 3);
        qsort(pilot,3,sizeof(float),compare_float);
        printf("CUTLASS candidate=%d %s ms=%.7f %.2f TOp/s %s\n",i,name(i),pilot[1],
            2.0 * o->m * o->n * o->k / (pilot[1] * 1e9),o->skip_verify ? "UNVERIFIED" : "PASS");
        if (meta) fprintf(meta,"cutlass candidate=%d %s pilot_ms=%.7f\n",i,name(i),pilot[1]);
        if (pilot[1] < fastest) { fastest = pilot[1]; best = i; }
        destroy(j->custom); j->custom = NULL;
    }
    if (best < 0) {
        printf("CUTLASS %s SKIP: adapter has no supported candidate for this output/shape\n",out_name);
        j->custom_run = NULL; return 0;
    }
    j->custom = create(best,o->raster,o->swizzle,o->m,o->n,o->k,
        (void *)(uintptr_t)j->a,(void *)(uintptr_t)j->b,
        (void *)(uintptr_t)sa,(void *)(uintptr_t)sb,
        (void *)(uintptr_t)j->output,out_type); REQUIRE(j->custom);
    for (int i = 0; i < o->warmup; i++) REQUIRE(!launch(j));
    CUDA(cuCtxSynchronize());
    /* Verify the actual timed path, including graph stream dispatch. */
    CUDA(cuMemsetD8(j->output,0xff,bytes));
    float times[101]; for (int i = 0; i < o->rounds; i++) times[i] = time_job(j,o->iters);
    if (!o->skip_verify) REQUIRE(verify(j,indices,refs,samples,out_type,o->full_check,elements,&rel,&max_error));
    qsort(times,o->rounds,sizeof(float),compare_float);
    float ms = times[o->rounds/2]; double tops = 2.0 * o->m * o->n * o->k / (ms * 1e9);
    printf("CUTLASS %s %s %s raster=%d swizzle=%d %.7f ms %.2f TOp/s %s rel_L2=%.6g max_error=%.6g\n",
        c->name,out_name,name(best),o->raster,o->swizzle,ms,tops,o->skip_verify ? "UNVERIFIED" : "PASS",rel,max_error);
    if (csv) {
        fprintf(csv,"%d,%d,%d,cutlass_%s,%s,%s,%d,%d,%d,%d,%d,0,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%s,%s,%d\n",
            o->m,o->n,o->k,c->name,out_name,o->data,o->warmup,o->iters,o->rounds,best,num,ms,times[0],
            times[o->rounds-1],tops,rel,max_error,samples,o->skip_verify ? "UNVERIFIED" : "PASS",o->graph ? "graph" : "direct",o->flush_mb); fflush(csv);
    }
    if (meta) { fprintf(meta,"selected=%d %s raster=%d swizzle=%d\n",best,name(best),o->raster,o->swizzle);
        gpu_info(meta); fflush(meta); }
    if (o->compare_cublas) compare_cublas(j,c,out_type,o,sa,sb,meta);
    destroy(j->custom); j->custom = NULL; j->custom_run = NULL;
    return 1;
}
static int run_config(cublasLtHandle_t handle, const config *c, int out_type,
                       const char *out_name, const options *o, FILE *csv, FILE *meta) {
    job j = {0}; j.handle = handle; j.graph = o->graph; j.flush_mb = o->flush_mb;
    if (j.flush_mb) j.flush = device_alloc((size_t)j.flush_mb << 20);
    cublasLtMatmulPreference_t pref = NULL;
    CUdeviceptr sa = 0, sb = 0, one = 0;
    uint8_t *a, *b, *as, *bs;
    uint32_t rng = 42; int pattern = !strcmp(o->data,"codebook") ? 2 : !strcmp(o->data,"random");
    make_input(c,o->m,o->k,pattern,&rng,&j.a,&sa,&a,&as);
    make_input(c,o->n,o->k,pattern,&rng,&j.b,&sb,&b,&bs);
    size_t count = (size_t)o->m * o->n, out_bytes = count * (out_type == CUDA_R_32F ? 4 : 2);
    j.output = device_alloc(out_bytes); CUDA(cuMemsetD8(j.output,0xff,out_bytes));
    int samples = o->full_check ? (int)count : (count < 64 ? (int)count : 64);
    size_t *indices = host_alloc((size_t)samples * sizeof(size_t));
    double *refs = host_alloc((size_t)samples * sizeof(double));
    for (int i = 0; i < samples; i++) {
        size_t index = samples == 1 ? 0 : (size_t)i * (count - 1) / (samples - 1);
        indices[i] = index; refs[i] = o->k;
        if (pattern) {
            double ref = 0;
            for (int k = 0; k < o->k; k++) ref +=
                (double)read_input(a,as,c,index / o->n,k,o->k) * read_input(b,bs,c,index % o->n,k,o->k);
            refs[i] = ref;
        }
    }
    if (o->compare_cublas && c->codebook) {
        j.decoded_a = decoded_bf16(a,as,c,o->m,o->k);
        j.decoded_b = decoded_bf16(b,bs,c,o->n,o->k);
    }
    free(a); free(b); free(as); free(bs);
    int status = 0, measured = 0, best = -1, returned = 0;
    cublasLtMatmulHeuristicResult_t *algorithms = calloc(16384,sizeof(*algorithms)); REQUIRE(algorithms);
    const char *stage = "descriptor";
    double rel = 0, max_error = 0;
    if (o->cutlass) {
        measured = run_cutlass(&j,c,out_type,out_name,o,csv,meta,indices,refs,samples,sa,sb);
        goto cleanup;
    }
#define TRY_LT(call) do { stage = #call; status = (call); if (status) goto unsupported; } while (0)
    TRY_LT(p_cublasLtMatmulDescCreate(&j.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    int trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
    TRY_LT(p_cublasLtMatmulDescSetAttribute(j.desc,CUBLASLT_MATMUL_DESC_TRANSA,&trans_a,sizeof(trans_a)));
    TRY_LT(p_cublasLtMatmulDescSetAttribute(j.desc,CUBLASLT_MATMUL_DESC_TRANSB,&trans_b,sizeof(trans_b)));
    if (c->block) {
        int mode = c->scale_mode;
        TRY_LT(p_cublasLtMatmulDescSetAttribute(j.desc,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
        TRY_LT(p_cublasLtMatmulDescSetAttribute(j.desc,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
        TRY_LT(p_cublasLtMatmulDescSetAttribute(j.desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sb,sizeof(sb)));
        TRY_LT(p_cublasLtMatmulDescSetAttribute(j.desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sa,sizeof(sa)));
    } else if (c->scale_mode == 0) {
        float v = 1; one = device_alloc(4); CUDA(cuMemcpyHtoD(one,&v,4));
        TRY_LT(p_cublasLtMatmulDescSetAttribute(j.desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&one,sizeof(one)));
        TRY_LT(p_cublasLtMatmulDescSetAttribute(j.desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&one,sizeof(one)));
    }
    /* Row-major C[M,N] = X[M,K] W[N,K]^T, viewed as column-major C[N,M]. */
    TRY_LT(p_cublasLtMatrixLayoutCreate(&j.a_layout,c->type,o->k,o->n,o->k));
    TRY_LT(p_cublasLtMatrixLayoutCreate(&j.b_layout,c->type,o->k,o->m,o->k));
    TRY_LT(p_cublasLtMatrixLayoutCreate(&j.out_layout,out_type,o->n,o->m,o->n));
    TRY_LT(p_cublasLtMatmulPreferenceCreate(&pref));
    j.workspace_bytes = (size_t)o->workspace_mb << 20;
    j.workspace = device_alloc(j.workspace_bytes);
    TRY_LT(p_cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &j.workspace_bytes,sizeof(j.workspace_bytes)));
    TRY_LT(p_cublasLtMatmulAlgoGetHeuristic(handle,j.desc,j.a_layout,j.b_layout,j.out_layout,j.out_layout,
        pref,o->algos,algorithms,&returned));
    if (o->exhaustive) exhaustive_algos(&j,c,out_type,algorithms,&returned);
    printf("Tuning %d configurations\n",returned);
    if (!returned) { stage = "heuristic returned no algorithm"; goto unsupported; }
    float fastest = INFINITY;
    for (int i = 0; i < returned; i++) {
        if (o->candidate >= 0 && i != o->candidate) continue;
        if (algorithms[i].status) continue;
        j.algorithm = &algorithms[i];
        CUDA(cuMemsetD8(j.output,0xff,out_bytes));
        if (launch(&j)) continue;
        CUDA(cuCtxSynchronize());
        if (!o->skip_verify) REQUIRE(verify(&j,indices,refs,samples,out_type,o->full_check,count,&rel,&max_error));
        float pilot[3]; for (int r = 0; r < 3; r++)
            pilot[r] = time_job(&j,o->pilot_iters ? o->pilot_iters : o->graph ? 32 : 3);
        qsort(pilot,3,sizeof(float),compare_float);
        if (o->exhaustive || o->candidate >= 0) {
            printf("  candidate=%d ms=%.7f ",i,pilot[1]); print_algo(stdout,algorithms[i].algo); puts("");
        }
        if (meta) {
            fprintf(meta,"candidate=%d ms=%.7f ",i,pilot[1]); print_algo(meta,algorithms[i].algo); fputc('\n',meta);
        }
        if (pilot[1] < fastest) { fastest = pilot[1]; best = i; }
    }
    REQUIRE(best >= 0); j.algorithm = &algorithms[best];
    for (int i = 0; i < o->warmup; i++) REQUIRE(!launch(&j));
    CUDA(cuCtxSynchronize());
    CUDA(cuMemsetD8(j.output,0xff,out_bytes));
    float times[101]; for (int i = 0; i < o->rounds; i++) times[i] = time_job(&j,o->iters);
    if (!o->skip_verify) REQUIRE(verify(&j,indices,refs,samples,out_type,o->full_check,count,&rel,&max_error));
    qsort(times,o->rounds,sizeof(float),compare_float);
    float ms = times[o->rounds / 2]; double tops = 2.0 * o->m * o->n * o->k / (ms * 1e9);
    printf("%-6s %-4s %-6s %.7f ms  %.2f TOp/s range=[%.7f,%.7f] algo=%d/%d workspace=%zu %s rel_L2=%.3g max_error=%.6g\n",
        c->name,out_name,o->data,ms,tops,times[0],times[o->rounds-1],best,returned,
        algorithms[best].workspaceSize,o->skip_verify ? "UNVERIFIED" : "PASS",rel,max_error);
    print_algo(stdout,algorithms[best].algo); puts("");
    if (csv) {
        fprintf(csv,"%d,%d,%d,%s,%s,%s,%d,%d,%d,%d,%d,%zu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%s,%s,%d\n",
            o->m,o->n,o->k,c->name,out_name,o->data,o->warmup,o->iters,o->rounds,best,returned,
            algorithms[best].workspaceSize,ms,times[0],times[o->rounds-1],tops,rel,max_error,samples,
            o->skip_verify ? "UNVERIFIED" : "PASS",o->graph ? "graph" : "direct",o->flush_mb); fflush(csv);
    }
    if (meta) {
        fprintf(meta,"%s/%s/%s selected=%d returned=%d algo=",c->name,out_name,o->data,best,returned);
        for (int i = 0; i < 8; i++) fprintf(meta,"%016llx",(unsigned long long)algorithms[best].algo[i]);
        fputc('\n',meta); gpu_info(meta); fflush(meta);
    }
    measured = 1; goto cleanup;
unsupported:
    printf("%-6s %-4s SKIP status=%d stage=%s\n",c->name,out_name,status,stage);
    if (meta) fprintf(meta,"%s/%s SKIP status=%d stage=%s\n",c->name,out_name,status,stage);
cleanup:
#undef TRY_LT
    if (pref) p_cublasLtMatmulPreferenceDestroy(pref);
    if (j.out_layout) p_cublasLtMatrixLayoutDestroy(j.out_layout);
    if (j.b_layout) p_cublasLtMatrixLayoutDestroy(j.b_layout);
    if (j.a_layout) p_cublasLtMatrixLayoutDestroy(j.a_layout);
    if (j.desc) p_cublasLtMatmulDescDestroy(j.desc);
    free(indices); free(refs);
    free(algorithms);
    if (j.flush) CUDA(cuMemFree(j.flush));
    if (one) CUDA(cuMemFree(one));
    if (j.decoded_a) CUDA(cuMemFree(j.decoded_a));
    if (j.decoded_b) CUDA(cuMemFree(j.decoded_b));
    if (sa) CUDA(cuMemFree(sa));
    if (sb) CUDA(cuMemFree(sb));
    if (j.workspace) CUDA(cuMemFree(j.workspace));
    CUDA(cuMemFree(j.a)); CUDA(cuMemFree(j.b)); CUDA(cuMemFree(j.output)); return measured;
}
static int positive_int(const char *s) {
    char *end; errno = 0; long v = strtol(s,&end,10);
    REQUIRE(!errno && *s && !*end && v > 0 && v <= INT_MAX); return (int)v;
}
int main(int argc, char **argv) {
    setvbuf(stdout,NULL,_IOLBF,0);
    options o = {8192,8192,8192,5,100,5,32,256,0,0,"all","f32","random",NULL,0,-1,NULL,0,1,0,0,0,0};
    for (int i = 1; i < argc; i++) {
        const char *key = argv[i];
        if (!strcmp(key,"--help") || !strcmp(key,"-h")) {
            puts("cublas_fp4_gemm [-m M -n N -k K] [--format all|nvfp4|fp8|bf16|mxfp4|e0m3]\n"
                 "  [--output f32|bf16|f16|all] [--data random|ones|codebook] [--csv PATH]\n"
                 "  [--warmup N --iters N --rounds N --algos N --workspace-mb N]\n"
                 "  [--graph] [--flush-mb N] [--pilot-iters N]\n"
                 "  [--full-check] [--skip-verify] [--exhaustive] [--candidate N]\n"
                 "  [--cutlass PATH --raster 0|1|2 --swizzle 1|2|4|8 --compare-cublas]\n"
                 "Defaults: 8192 cubed, random signed codes/varying block scales, FP32 output,\n"
                 "5 warmups, 100 launches x 5 rounds, fastest of up to 32 heuristics, 256 MiB workspace.\n"
                 "The default CUTLASS adapter requires --format nvfp4 --output bf16.\n"
                 "E0M3 requires its verified patched --cutlass adapter and verification;\n"
                 "--compare-cublas then uses exactly decoded BF16 inputs outside timing.\n"
                 "Packed GEMM only; CPU packing/scale swizzle and algorithm search excluded."); return 0;
        }
        if (!strcmp(key,"--graph")) { o.graph = 1; continue; }
        if (!strcmp(key,"--full-check")) { o.full_check = 1; continue; }
        if (!strcmp(key,"--exhaustive")) { o.exhaustive = 1; continue; }
        if (!strcmp(key,"--compare-cublas")) { o.compare_cublas = 1; continue; }
        if (!strcmp(key,"--skip-verify")) { o.skip_verify = 1; continue; }
        REQUIRE(i + 1 < argc); const char *v = argv[++i];
        if (!strcmp(key,"-m") || !strcmp(key,"--m")) o.m = positive_int(v);
        else if (!strcmp(key,"-n") || !strcmp(key,"--n")) o.n = positive_int(v);
        else if (!strcmp(key,"-k") || !strcmp(key,"--k")) o.k = positive_int(v);
        else if (!strcmp(key,"--warmup")) o.warmup = positive_int(v);
        else if (!strcmp(key,"--iters")) o.iters = positive_int(v);
        else if (!strcmp(key,"--pilot-iters")) o.pilot_iters = positive_int(v);
        else if (!strcmp(key,"--rounds")) o.rounds = positive_int(v);
        else if (!strcmp(key,"--algos")) o.algos = positive_int(v);
        else if (!strcmp(key,"--workspace-mb")) o.workspace_mb = positive_int(v);
        else if (!strcmp(key,"--candidate")) o.candidate = !strcmp(v,"0") ? 0 : positive_int(v);
        else if (!strcmp(key,"--flush-mb")) o.flush_mb = positive_int(v);
        else if (!strcmp(key,"--cutlass")) o.cutlass = v;
        else if (!strcmp(key,"--raster")) o.raster = !strcmp(v,"0") ? 0 : positive_int(v);
        else if (!strcmp(key,"--swizzle")) o.swizzle = positive_int(v);
        else if (!strcmp(key,"--format")) o.format = v;
        else if (!strcmp(key,"--output")) o.output = v;
        else if (!strcmp(key,"--data")) o.data = v;
        else if (!strcmp(key,"--csv")) o.csv = v;
        else { fprintf(stderr,"unknown option %s\n",key); return 1; }
    }
    REQUIRE(o.m <= 16384 && o.n <= 262144 && o.k <= 32768 && !(o.n % 8) && !(o.k % 64));
    /* Large vocabulary projections retain a bounded decode allocation. */
    REQUIRE(o.n <= 32768 || (o.m <= 64 && o.k <= 4096));
    REQUIRE(o.flush_mb <= 1024);
    REQUIRE(o.rounds <= 101 && (o.rounds & 1) && o.algos <= 64 && o.workspace_mb <= 1024);
    REQUIRE(o.raster <= 2 && (o.swizzle == 1 || o.swizzle == 2 || o.swizzle == 4 || o.swizzle == 8));
    REQUIRE(!o.compare_cublas || (o.cutlass && !o.skip_verify));
    REQUIRE(strcmp(o.format,"e0m3") || (o.cutlass && !o.skip_verify));
    REQUIRE(!o.full_check || (size_t)o.m * o.n <= 65536);
    REQUIRE(!strcmp(o.data,"ones") || !strcmp(o.data,"random") || !strcmp(o.data,"codebook"));
    REQUIRE(strcmp(o.data,"codebook") || !strcmp(o.format,"e0m3") || !strcmp(o.format,"nvfp4"));
    int found = !strcmp(o.format,"all");
    for (int i = 0; i < 5; i++) found |= !strcmp(o.format,configs[i].name);
    REQUIRE(found);
    REQUIRE(!strcmp(o.output,"all") || !strcmp(o.output,"f32") || !strcmp(o.output,"bf16") || !strcmp(o.output,"f16"));
    CUcontext context = start_cuda(); CUdevice dev; CUDA(cuCtxGetDevice(&dev));
    char name[256]; CUDA(cuDeviceGetName(name,sizeof(name),dev)); int driver; CUDA(cuDriverGetVersion(&driver));
    REQUIRE(!load_cublas()); cublasLtHandle_t handle; REQUIRE(!p_cublasLtCreate(&handle));
    size_t (*version)(void); void *symbol = dlsym(cublas_lib,"cublasLtGetVersion"); memcpy(&version,&symbol,sizeof(symbol));
    printf("%s; driver API=%d cuBLASLt=%zu; M=%d N=%d K=%d; FP32 compute\n",name,driver,version ? version() : 0,o.m,o.n,o.k);
    printf("timing=%s cache=%s flush_mb=%d\n",o.graph ? "graph" : "direct",o.flush_mb ? "evicted" : "warm",o.flush_mb);
    FILE *csv = NULL, *meta = NULL;
    if (o.csv) {
        csv = fopen(o.csv,"w"); REQUIRE(csv);
        fputs("m,n,k,format,output,data,warmup,iters,rounds,algo_index,algos_returned,workspace_bytes,median_ms,min_ms,max_ms,tops,rel_l2,max_error,samples,status,timing,flush_mb\n",csv);
        char path[4096]; REQUIRE(snprintf(path,sizeof(path),"%s.metadata.txt",o.csv) < (int)sizeof(path));
        meta = fopen(path,"w"); REQUIRE(meta);
        for (int i = 0; i < argc; i++) fprintf(meta,"%s%s",i ? " " : "",argv[i]);
        fprintf(meta,"\ndriver_api=%d cuBLASLt=%zu\n",driver,version ? version() : 0); gpu_info(meta);
    }
    int types[] = {CUDA_R_32F,CUDA_R_16BF,CUDA_R_16F}; const char *names[] = {"f32","bf16","f16"};
    int measured = 0;
    for (int c = 0; c < 5; c++) if ((!strcmp(o.format,"all") && !configs[c].codebook) || !strcmp(o.format,configs[c].name))
        for (int t = 0; t < 3; t++) if (!strcmp(o.output,"all") || !strcmp(o.output,names[t]))
            measured += run_config(handle,&configs[c],types[t],names[t],&o,csv,meta);
    if (csv) REQUIRE(!fclose(csv));
    if (meta) REQUIRE(!fclose(meta));
    REQUIRE(!p_cublasLtDestroy(handle));
    /* CUDA context teardown invokes module callbacks owned by cuBLASLt. */
    CUDA(cuCtxDestroy(context));
    if (cutlass_lib) dlclose(cutlass_lib);
    dlclose(cublas_lib);
    return measured ? 0 : 2;
}
