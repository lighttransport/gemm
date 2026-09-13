/* SPDX-License-Identifier: MIT */
#include "../cuda/cuew.h"
#include "../rdna4/rocew.h"
#include "gn_internal.h"
#include "gn_kernels.inc"
#ifdef GN_HIPBLASLT
#include "gn_hipblaslt.h"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NODE_BASE (4 * GN_PARAMS)
#define SCRATCH_BASE (NODE_BASE + 3 * GN_NODES)
#define SLOT_COUNT (SCRATCH_BASE + 15)
typedef struct {
    uint64_t ptr;
    size_t bytes;
} Buffer;
typedef struct {
    int hip, device, active, fp32, fp16, hybrid16, precise, synced, legacy, integer, reduced, chunk;
    int wide, mixed;
    int x3_backward;
    size_t used, limit;
    CUcontext context;
    CUmodule cuda_module;
    hipModule_t hip_module;
    Buffer b[SLOT_COUNT];
    void *functions[46];
    void *lt;
} Gpu;
static const char *names[] = {"gn_mm",
                              "gn_columns",
                              "gn_uncolumns",
                              "gn_bias",
                              "gn_bias_back",
                              "gn_point",
                              "gn_norm",
                              "gn_norm_back",
                              "gn_layer_param",
                              "gn_attention",
                              "gn_attention_back",
                              "gn_loss",
                              "gn_grad_norm",
                              "gn_adam",
                              "gn_mm_fp32",
                              "gn_pack_bf16",
                              "gn_mm_tiled",
                              "gn_mm_tiled_fast",
                              "gn_norm_parallel",
                              "gn_norm_back_parallel",
                              "gn_layer_param_parallel",
                              "gn_loss_parallel",
                              "gn_pack_integer",
                              "gn_mm_integer",
                              "gn_attention_parallel",
                              "gn_attention_scores_back",
                              "gn_attention_qkv_back",
                              "gn_attention_bias_back",
                              "gn_lt_combine",
                              "gn_bias_back_parallel",
                              "gn_grad_norm_parallel",
                              "gn_mm_bf16_acc",
                              "gn_columns_bf16",
                              "gn_attention_81",
                              "gn_transpose_rows",
                              "gn_mm_bf16x3",
                              "gn_bn_channels",
                              "gn_bn_back_channels",
                              "gn_columns_bf16_back",
                              "gn_attention_scores_back_81",
                              "gn_pack_fp16",
                              "gn_columns_fp16",
                              "gn_columns_fp16_back",
                              "gn_point_pair",
                              "gn_bn_silu_channels",
                              "gn_bn_silu_back_channels"};
static int current(Gpu *g) {
    int rc = g->hip ? (int)hipSetDevice(g->device) : (int)cuCtxSetCurrent(g->context);
    return rc ? gn_fail("cannot activate GPU context") : 0;
}
static int copy_to(Gpu *g, uint64_t d, const void *s, size_t bytes) {
    int rc = g->hip ? (int)hipMemcpy((void *)(uintptr_t)d, s, bytes, hipMemcpyHostToDevice)
                    : (int)cuMemcpyHtoD(d, s, bytes);
    return rc ? gn_fail("GPU upload failed") : 0;
}
static int copy_from(Gpu *g, void *d, uint64_t s, size_t bytes) {
    int rc = g->hip ? (int)hipMemcpy(d, (void *)(uintptr_t)s, bytes, hipMemcpyDeviceToHost)
                    : (int)cuMemcpyDtoH(d, s, bytes);
    return rc ? gn_fail("GPU download/execution failed") : 0;
}
static int zero(Gpu *g, uint64_t p, size_t bytes) {
    int rc = g->hip ? (int)hipMemset((void *)(uintptr_t)p, 0, bytes) : (int)cuMemsetD8(p, 0, bytes);
    return rc ? gn_fail("GPU clear failed") : 0;
}
static uint64_t buffer(Gpu *g, int slot, size_t bytes, const void *initial) {
    Buffer *b = &g->b[slot];
    if (b->bytes >= bytes)
        return b->ptr;
    if (b->ptr) {
        if (g->hip)
            hipFree((void *)(uintptr_t)b->ptr);
        else
            cuMemFree(b->ptr);
        g->used -= b->bytes;
        b->ptr = 0;
        b->bytes = 0;
    }
    if (bytes > g->limit - g->used) {
        gn_fail("GPU workspace exceeds memory limit");
        return 0;
    }
    int rc;
    if (g->hip) {
        void *p = NULL;
        rc = (int)hipMalloc(&p, bytes);
        b->ptr = (uint64_t)(uintptr_t)p;
    } else {
        CUdeviceptr pointer = 0;
        rc = (int)cuMemAlloc(&pointer, bytes);
        b->ptr = (uint64_t)pointer;
    }
    if (rc) {
        gn_fail("GPU allocation failed");
        return 0;
    }
    b->bytes = bytes;
    g->used += bytes;
    if (initial) {
        if (copy_to(g, b->ptr, initial, bytes))
            return 0;
    } else if (zero(g, b->ptr, bytes))
        return 0;
    return b->ptr;
}
static int launch(Gpu *g, int fn, unsigned x, unsigned y, unsigned threads, void **args) {
    int rc = g->hip ? (int)hipModuleLaunchKernel((hipFunction_t)g->functions[fn], x, y, 1, threads,
                                                 1, 1, 0, 0, args, NULL)
                    : (int)cuLaunchKernel((CUfunction)g->functions[fn], x, y, 1, threads, 1, 1, 0,
                                          0, args, NULL);
    if (rc) {
        char text[256];
        snprintf(text, sizeof(text), "GPU launch %s failed (%d)", names[fn], rc);
        return gn_fail(text);
    }
    return 0;
}
static int flat(Gpu *g, int fn, size_t count, void **args) {
    return launch(g, fn, (unsigned)((count + 255) / 256), 1, 256, args);
}
#define CALL(expr)                                                                                 \
    do {                                                                                           \
        if ((expr))                                                                                \
            return -1;                                                                             \
    } while (0)
#define PTR_NODE(m, id, part) (((Gpu *)(m)->gpu)->b[NODE_BASE + 3 * (id) + (part)].ptr)
static uint64_t param(gn_model *m, Param *p, int part) {
    return p ? ((Gpu *)m->gpu)->b[4 * (p - m->p) + part].ptr : 0;
}
/* ci>0 packs forward im2col into A; ci<0 packs transposed im2col into B
 * for dW, using -ci input channels. ci=0 consumes ordinary FP32 matrices. */
static int mm_columns(Gpu *g, uint64_t y, uint64_t a, uint64_t b, int M, int N, int K, int ta,
                      int tb, int add, int ci, int side, int kernel) {
    void *args[] = {&y, &a, &b, &M, &N, &K, &ta, &tb, &add, &g->precise};
    if (g->fp32)
        return flat(g, 14, (size_t)M * N, args);
    if (g->integer) {
        size_t stride = ((size_t)K + 31) & ~(size_t)31;
        uint64_t pa = buffer(g, SCRATCH_BASE + 8, M * stride * (g->integer / 8), NULL);
        uint64_t pb = buffer(g, SCRATCH_BASE + 9, N * stride * (g->integer / 8), NULL);
        uint64_t sa = buffer(g, SCRATCH_BASE + 10, (size_t)M * 4, NULL);
        uint64_t sb = buffer(g, SCRATCH_BASE + 11, (size_t)N * 4, NULL);
        if (!pa || !pb || !sa || !sb)
            return -1;
        int bt = !tb;
        uint64_t ax = a, bx = b;
        int at = ta;
        uint64_t transpose = 0;
        if (g->hip && K >= 32 && ((ta && M >= 32) || (bt && N >= 32))) {
            size_t rows = M > N ? M : N;
            transpose = buffer(g, SCRATCH_BASE + 7, rows * K * 4, NULL);
            if (!transpose)
                return -1;
        }
        if (transpose && ta && M >= 32) {
            void *tx[] = {&transpose, &a, &M, &K};
            CALL(launch(g, 34, (K + 31) / 32, (M + 31) / 32, 256, tx));
            ax = transpose;
            at = 0;
        }
        void *ap[] = {&pa, &sa, &ax, &M, &K, &at, &g->integer};
        CALL(launch(g, 22, M, 1, 256, ap));
        if (transpose && bt && N >= 32) {
            void *tx[] = {&transpose, &b, &N, &K};
            CALL(launch(g, 34, (K + 31) / 32, (N + 31) / 32, 256, tx));
            bx = transpose;
            bt = 0;
        }
        void *bp[] = {&pb, &sb, &bx, &N, &K, &bt, &g->integer};
        CALL(launch(g, 22, N, 1, 256, bp));
        uint64_t exact = 0;
        void *packed[] = {&y, &pa, &pb, &sa, &sb, &M, &N, &K, &add, &g->integer, &g->wide, &exact};
        if (g->hip) {
            int tile_n = g->integer == 16 ? 32 : 64;
            return launch(g, 23, (N + tile_n - 1) / tile_n, (M + 63) / 64, 128, packed);
        }
        return launch(g, 23, (N + 7) / 8, (M + 15) / 16, 32, packed);
    }
    if (g->reduced || (!g->legacy && M >= 32 && N >= 32 && K >= 32)) {
        size_t stride = ((size_t)K + 31) & ~(size_t)31;
        int planes = g->precise == 2 ? 2 : g->precise ? 3 : 1;
        uint64_t pa = buffer(g, SCRATCH_BASE + 8, M * stride * 2 * planes, NULL);
        uint64_t pb = buffer(g, SCRATCH_BASE + 9, N * stride * 2 * planes, NULL);
        if (!pa || !pb)
            return -1;
        int bt = !tb;
        void *ap[] = {&pa, &a, &M, &K, &ta, &g->precise};
        void *bp[] = {&pb, &b, &N, &K, &bt, &g->precise};
        if (ci > 0) {
            void *cp[] = {&pa, &a, &M, &ci, &side, &kernel, &g->precise};
            CALL(flat(g, g->fp16 ? 41 : 32, M * stride, cp));
        } else
            CALL(launch(g, g->fp16 ? 40 : 15, (K + 31) / 32, (M + 31) / 32, 256, ap));
        if (ci < 0) {
            int channels = -ci;
            void *cp[] = {&pb, &b, &K, &channels, &side, &kernel, &g->precise};
            CALL(launch(g, g->fp16 ? 42 : 38, (N + 31) / 32, (K + 31) / 32, 256, cp));
        } else
            CALL(launch(g, g->fp16 ? 40 : 15, (K + 31) / 32, (N + 31) / 32, 256, bp));
#ifdef GN_HIPBLASLT
        /* BLASLt wins the long-K convolutions; native WMMA avoids six vendor
         * launches and intermediate writes for short-K/small training GEMMs. */
        if (g->lt && (!g->precise || (K >= 512 && (size_t)M * N >= 262144))) {
            size_t workspace_bytes = 64u * 1024 * 1024 + (((size_t)M * N * 4 + 255) & ~(size_t)255);
            uint64_t workspace = buffer(g, SCRATCH_BASE + 12, workspace_bytes, NULL);
            if (!workspace)
                return -1;
            if (!g->precise) {
                if (gn_lt_run(g->lt, (void *)(uintptr_t)y, (void *)(uintptr_t)pa,
                              (void *)(uintptr_t)pb, M, N, K, add ? 1 : 0,
                              (void *)(uintptr_t)workspace, workspace_bytes, g->fp16))
                    return gn_fail("hipBLASLt inference matmul failed");
                return 0;
            }
            uint64_t high = buffer(g, SCRATCH_BASE + 13, (size_t)M * N * 4, NULL);
            uint64_t low = buffer(g, SCRATCH_BASE + 14, (size_t)M * N * 4, NULL);
            if (!high || !low)
                return -1;
            const int ai[] = {0, 1, 0, 1, 2, 0}, bi[] = {0, 0, 1, 1, 0, 2};
            for (int i = 0; i < (g->precise == 2 ? 3 : 6); i++)
                if (gn_lt_run(g->lt, (void *)(uintptr_t)(i ? low : high),
                              (void *)(uintptr_t)(pa + ai[i] * M * stride * 2),
                              (void *)(uintptr_t)(pb + bi[i] * N * stride * 2), M, N, K,
                              i > 1 ? 1 : 0, (void *)(uintptr_t)workspace, workspace_bytes, 0))
                    return gn_fail("hipBLASLt compensated training matmul failed");
            int count = M * N;
            void *combine[] = {&y, &high, &low, &count, &add};
            return flat(g, 28, count, combine);
        }
#endif
        void *packed[] = {&y, &pa, &pb, &M, &N, &K, &add, &g->chunk};
        if (g->reduced == 2)
            return launch(g, 31, (N + 63) / 64, (M + 63) / 64, 128, packed);
        if (g->precise == 2)
            return launch(g, 35, (N + 31) / 32, (M + 31) / 32, 128, packed);
        int tile_n = g->hip && g->precise ? 32 : 64, tile_m = g->precise ? 32 : 64;
        return launch(g, g->precise ? 16 : 17, (N + tile_n - 1) / tile_n, (M + tile_m - 1) / tile_m,
                      128, packed);
    }
    return launch(g, 0, (unsigned)((N + (g->hip ? 15 : 7)) / (g->hip ? 16 : 8)),
                  (unsigned)((M + 15) / 16), 32, args);
}
static int mm(Gpu *g, uint64_t y, uint64_t a, uint64_t b, int M, int N, int K, int ta, int tb,
              int add) {
    return mm_columns(g, y, a, b, M, N, K, ta, tb, add, 0, 0, 0);
}
static int columns(Gpu *g, uint64_t col, uint64_t x, int R, int C, int side, int kernel, int back) {
    void *args[] = {&col, &x, &R, &C, &side, &kernel};
    return flat(g, back ? 2 : 1, (size_t)R * C * (back ? 1 : kernel * kernel), args);
}
void *gn_gpu_open(const char *backend, int device, size_t limit) {
    Gpu *g = calloc(1, sizeof(*g));
    if (!g) {
        gn_fail("GPU context allocation failed");
        return NULL;
    }
    g->hip = !strncmp(backend, "hip", 3);
    g->fp32 = strstr(backend, "fp32") != NULL;
    g->hybrid16 = strstr(backend, "fp16back") != NULL;
    g->fp16 = strstr(backend, "fp16") != NULL && !g->hybrid16;
    g->legacy = strstr(backend, "legacy") != NULL;
    g->integer = strstr(backend, "int16") ? 16 : strstr(backend, "int8") ? 8 : 0;
    g->wide = strstr(backend, "i64") != NULL;
    g->mixed = strstr(backend, "bf16-mixed") != NULL;
    g->x3_backward = strstr(backend, "bf16x3-dx")        ? 1
                     : strstr(backend, "bf16x3-dw")      ? 2
                     : strstr(backend, "bf16x3-forward") ? 3
                                                         : 0;
    g->reduced = g->fp16                       ? 1
                 : strstr(backend, "bf16x3")  ? 3
                 : strstr(backend, "bf16-acc") ? 2
                 : strstr(backend, "bf16")     ? 1
                                               : 0;
    g->chunk = strstr(backend, "acc128") ? 128 : 0;
    if (g->integer || g->reduced)
        fprintf(
            stderr,
            "EXPERIMENTAL %s: reduced-precision matrices, FP32 master/optimizer/nonmatrix state; "
            "qualification is model/batch specific; see RDNA4.md\n",
            backend);
    g->device = device;
    g->limit = limit;
    int rc = 0;
    if (g->hip) {
        int loader = rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC);
        int init = loader == ROCEW_SUCCESS && rocewHiprtcAvailable() ? (int)hipInit(0) : -1;
        int selected = init == hipSuccess ? (int)hipSetDevice(device) : -1;
        if (loader != ROCEW_SUCCESS || init != hipSuccess || selected != hipSuccess) {
            char message[256];
            snprintf(message, sizeof(message),
                     "HIP unavailable: loader=%d hiprtc=%d init=%d device=%d; set ROCEW_ROCM_LIB "
                     "to ROCm library directory",
                     loader, rocewHiprtcAvailable(), init, selected);
            gn_fail(message);
            goto bad;
        }
        g->active = 1;
        if (strstr(backend, "blaslt")) {
#ifdef GN_HIPBLASLT
            fprintf(stderr, "hip-blaslt: consult RDNA4.md for per-backend gradient results\n");
            g->lt = gn_lt_open(strstr(backend, "blaslt-tuned") != NULL);
            if (!g->lt) {
                gn_fail("hipBLASLt initialization failed");
                goto bad;
            }
#else
            gn_fail("hip-blaslt requires an opt-in HIPBLASLT=1 / GN_HIPBLASLT=ON build");
            goto bad;
#endif
        }
        hiprtcProgram program = NULL;
        if (hiprtcCreateProgram(&program, gn_kernel_source, "gn_kernels.cu", 0, NULL, NULL) !=
            HIPRTC_SUCCESS) {
            gn_fail("HIPRTC program creation failed");
            goto bad;
        }
        const char *options[] = {"--gpu-architecture=gfx1201", "--std=c++17", "-DGN_HIP=1", "-O3"};
        hiprtcResult result = hiprtcCompileProgram(program, 4, options);
        if (result != HIPRTC_SUCCESS) {
            size_t n = 0;
            hiprtcGetProgramLogSize(program, &n);
            char *log = malloc(n + 1);
            if (log) {
                hiprtcGetProgramLog(program, log);
                log[n] = 0;
                fprintf(stderr, "%s\n", log);
                free(log);
            }
            hiprtcDestroyProgram(&program);
            gn_fail("HIPRTC kernel compilation failed");
            goto bad;
        }
        size_t bytes = 0;
        hiprtcGetCodeSize(program, &bytes);
        char *code = malloc(bytes);
        if (!code) {
            hiprtcDestroyProgram(&program);
            gn_fail("HIPRTC output allocation failed");
            goto bad;
        }
        rc = (int)hiprtcGetCode(program, code);
        hiprtcDestroyProgram(&program);
        if (!rc)
            rc = (int)hipModuleLoadData(&g->hip_module, code);
        free(code);
    } else {
        if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS ||
            cuInit(0) != CUDA_SUCCESS) {
            gn_fail("CUDA driver/NVRTC unavailable (requires sm120)");
            goto bad;
        }
        CUdevice dev;
        int major = 0, minor = 0;
        if (cuDeviceGet(&dev, device) != CUDA_SUCCESS ||
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) !=
                CUDA_SUCCESS ||
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) !=
                CUDA_SUCCESS ||
            major != 12 || minor != 0) {
            gn_fail("CUDA backend requires compute capability 12.0");
            goto bad;
        }
        if (cuCtxCreate(&g->context, 0, dev) != CUDA_SUCCESS) {
            gn_fail("CUDA context creation failed");
            goto bad;
        }
        g->active = 1;
        nvrtcProgram program = NULL;
        if (nvrtcCreateProgram(&program, gn_kernel_source, "gn_kernels.cu", 0, NULL, NULL) !=
            NVRTC_SUCCESS) {
            gn_fail("NVRTC program creation failed");
            goto bad;
        }
        const char *options[] = {"--gpu-architecture=compute_120", "--std=c++17"};
        nvrtcResult result = nvrtcCompileProgram(program, 2, options);
        if (result != NVRTC_SUCCESS) {
            size_t n = 0;
            nvrtcGetProgramLogSize(program, &n);
            char *log = malloc(n + 1);
            if (log) {
                nvrtcGetProgramLog(program, log);
                log[n] = 0;
                fprintf(stderr, "%s\n", log);
                free(log);
            }
            nvrtcDestroyProgram(&program);
            gn_fail("NVRTC kernel compilation failed");
            goto bad;
        }
        size_t bytes = 0;
        nvrtcGetPTXSize(program, &bytes);
        char *code = malloc(bytes);
        if (!code) {
            nvrtcDestroyProgram(&program);
            gn_fail("NVRTC output allocation failed");
            goto bad;
        }
        rc = (int)nvrtcGetPTX(program, code);
        nvrtcDestroyProgram(&program);
        if (!rc)
            rc = (int)cuModuleLoadData(&g->cuda_module, code);
        free(code);
    }
    if (rc) {
        gn_fail("GPU module load failed");
        goto bad;
    }
    for (int i = 0; i < (g->hip ? 46 : 28); i++) {
        if (g->hip) {
            hipFunction_t f;
            rc = (int)hipModuleGetFunction(&f, g->hip_module, names[i]);
            g->functions[i] = (void *)f;
        } else {
            CUfunction f;
            rc = (int)cuModuleGetFunction(&f, g->cuda_module, names[i]);
            g->functions[i] = (void *)f;
        }
        if (rc) {
            gn_fail("missing GPU kernel");
            goto bad;
        }
    }
    return g;
bad:
    gn_gpu_close(g);
    return NULL;
}
void gn_gpu_close(void *opaque) {
    Gpu *g = opaque;
    if (!g)
        return;
    if (g->active) {
        current(g);
#ifdef GN_HIPBLASLT
        gn_lt_close(g->lt);
#endif
        for (int i = 0; i < SLOT_COUNT; i++)
            if (g->b[i].ptr) {
                if (g->hip)
                    hipFree((void *)(uintptr_t)g->b[i].ptr);
                else
                    cuMemFree(g->b[i].ptr);
            }
    }
    if (g->hip_module)
        hipModuleUnload(g->hip_module);
    if (g->cuda_module)
        cuModuleUnload(g->cuda_module);
    if (g->context)
        cuCtxDestroy(g->context);
    free(g);
}
static int prepare(gn_model *m) {
    Gpu *g = m->gpu;
    CALL(current(g));
    for (size_t i = 0; i < m->np; i++) {
        Param *p = &m->p[i];
        float *initial[] = {p->x, p->g, p->m, p->v};
        for (int j = 0; j < (p->learned ? 4 : 1); j++)
            if (!buffer(g, (int)i * 4 + j, p->r * p->c * 4, initial[j]))
                return -1;
    }
    for (size_t i = 0; i < m->nn; i++) {
        Node *n = &m->n[i];
        int slot = NODE_BASE + 3 * (int)i;
        if (!buffer(g, slot, n->r * n->c * 4, NULL))
            return -1;
        if (m->training) {
            uint64_t d = buffer(g, slot + 1, n->r * n->c * 4, NULL);
            if (!d)
                return -1;
            CALL(zero(g, d, n->r * n->c * 4));
        }
        size_t aux = n->kind == BN   ? 2 * n->c
                     : n->kind == LN ? 2 * n->r
                     : n->kind == ATTENTION
                         ? n->r * m->cfg.side * m->cfg.side * (n->c / m->cfg.head_dim)
                         : 0;
        if (aux && !buffer(g, slot + 2, aux * 4, NULL))
            return -1;
    }
    return 0;
}
int gn_gpu_forward(gn_model *m, const float *input) {
    Gpu *g = m->gpu;
    if (g->hybrid16)
        g->fp16 = 0;
    g->precise = g->mixed ? 1 : g->reduced == 3 ? 2 : m->training && !g->reduced;
    if (m->training)
        g->synced = 0;
    CALL(prepare(m));
    CALL(copy_to(g, PTR_NODE(m, 0, 0), input, m->n[0].r * m->n[0].c * 4));
    int side = (int)m->cfg.side, B = (int)m->batch, D = (int)m->cfg.head_dim,
        training = m->training;
    for (size_t i = 1; i < m->nn; i++) {
        Node *n = &m->n[i];
        int R = (int)n->r, C = (int)n->c, K = n->k, layer = n->kind == LN;
        uint64_t x = PTR_NODE(m, n->a, 0), y = PTR_NODE(m, i, 0), w = param(m, n->w, 0),
                 bias = param(m, n->bias, 0), aux = PTR_NODE(m, i, 2);
        if (g->hip && !g->legacy && n->kind == BN && n->a > 0 &&
            m->n[n->a].kind == CONV && i + 1 < m->nn &&
            m->n[i + 1].kind == SILU && m->n[i + 1].a == (int)i) {
            uint64_t out = PTR_NODE(m, i + 1, 0), mean = param(m, n->mean, 0),
                     var = param(m, n->variance, 0), prebias = param(m, m->n[n->a].bias, 0);
            void *args[] = {&y, &out, &aux, &mean, &var, &x, &prebias,
                            &w, &bias, &R,   &C,    &training};
            CALL(launch(g, 44, (C + 7) / 8, 1, 256, args));
            i++;
            continue;
        }
        if (g->hip && !g->legacy && i + 1 < m->nn) {
            Node *next = &m->n[i + 1];
            int pair = n->kind == ADD && next->kind == SILU && next->a == (int)i   ? 0
                       : n->kind == SILU && next->kind == MUL && next->a == (int)i ? 1
                                                                                   : -1;
            if (pair >= 0) {
                uint64_t out = PTR_NODE(m, i + 1, 0);
                uint64_t other = pair ? PTR_NODE(m, next->b, 0) : PTR_NODE(m, n->b, 0);
                uint64_t dummy = 0;
                int count = R * C, back = 0;
                void *args[] = {&y, &out, &dummy, &dummy, &x, &other,
                                &dummy, &count, &pair,  &back};
                CALL(flat(g, 43, (size_t)count, args));
                i++;
                continue;
            }
        }
        if (n->kind == LINEAR || n->kind == CONV) {
            int fused_ci = 0;
            if (n->kind == CONV) {
                int ci = (int)m->n[n->a].c;
                K = ci * n->k * n->k;
                if (g->hip && !g->legacy && !g->fp32 && !g->integer &&
                    (g->reduced || (R >= 32 && C >= 32 && K >= 32))) {
                    fused_ci = ci;
                } else {
                    uint64_t col = buffer(g, SCRATCH_BASE, (size_t)R * K * 4, NULL);
                    if (!col)
                        return -1;
                    CALL(columns(g, col, x, R, ci, side, n->k, 0));
                    x = col;
                }
            }
            CALL(mm_columns(g, y, x, w, R, C, K, 0, 1, 0, fused_ci, side, n->k));
            int fused_bias = g->hip && !g->legacy && i + 2 < m->nn &&
                             m->n[i + 1].kind == BN && m->n[i + 1].a == (int)i &&
                             m->n[i + 2].kind == SILU && m->n[i + 2].a == (int)i + 1;
            if (!fused_bias) {
                void *args[] = {&y, &bias, &R, &C};
                CALL(flat(g, 3, (size_t)R * C, args));
            }
        } else if (n->kind == BN || n->kind == LN) {
            uint64_t mean = param(m, n->mean, 0), var = param(m, n->variance, 0);
            void *args[] = {&y, &aux, &mean, &var, &x, &w, &bias, &R, &C, &layer, &training};
            if (g->hip && !g->legacy && !layer)
                CALL(launch(g, 36, (C + 7) / 8, 1, 256, args));
            else if (!g->legacy)
                CALL(launch(g, 18, layer ? R : C, 1, 256, args));
            else
                CALL(flat(g, 6, layer ? R : C, args));
        } else if (n->kind == ATTENTION) {
            void *args[] = {&y, &aux, &x, &w, &B, &side, &C, &D};
            if (!g->legacy)
                if (g->hip && side == 9 && D == 32)
                    CALL(launch(g, 33, B * (C / D) * 11, 1, 256, args));
                else
                    CALL(launch(g, 24, R * (C / D), 1, 256, args));
            else
                CALL(flat(g, 9, (size_t)R * (C / D), args));
        } else {
            uint64_t other = n->b >= 0 ? PTR_NODE(m, n->b, 0) : 0, dummy = 0;
            int count = R * C, op = n->kind, back = 0;
            void *args[] = {&y, &dummy, &dummy, &x, &other, &dummy, &count, &op, &back};
            CALL(flat(g, 5, (size_t)count, args));
        }
    }
    if (m->training)
        return 0; /* loss/backward consume resident heads */
    Node *p = &m->n[m->policy], *v = &m->n[m->value];
    CALL(copy_from(g, p->x, PTR_NODE(m, m->policy, 0), p->r * p->c * 4));
    CALL(copy_from(g, v->x, PTR_NODE(m, m->value, 0), v->r * v->c * 4));
    for (size_t i = 0; i < p->r * p->c; i++)
        if (!isfinite(p->x[i]))
            return gn_fail("non-finite GPU policy output");
    for (size_t i = 0; i < v->r * v->c; i++)
        if (!isfinite(v->x[i]))
            return gn_fail("non-finite GPU value output");
    return 0;
}
int gn_gpu_backward(gn_model *m, const float *target, const uint32_t *labels, gn_metrics *metrics) {
    Gpu *g = m->gpu;
    if (g->hybrid16) {
        g->fp16 = 1;
        g->precise = 0;
    }
    if (g->mixed)
        g->precise = 2;
    g->synced = 0;
    int B = (int)m->batch, side = (int)m->cfg.side, D = (int)m->cfg.head_dim,
        A = side * side * m->cfg.actions;
    uint64_t t = buffer(g, SCRATCH_BASE + 2, (size_t)B * A * 4, NULL),
             l = buffer(g, SCRATCH_BASE + 3, (size_t)B * 4, NULL),
             loss = buffer(g, SCRATCH_BASE + 4, (size_t)B * 8, NULL);
    if (!t || !l || !loss)
        return -1;
    CALL(copy_to(g, t, target, (size_t)B * A * 4));
    CALL(copy_to(g, l, labels, (size_t)B * 4));
    uint64_t p = PTR_NODE(m, m->policy, 0), v = PTR_NODE(m, m->value, 0),
             dp = PTR_NODE(m, m->policy, 1), dv = PTR_NODE(m, m->value, 1);
    void *lossargs[] = {&dp, &dv, &loss, &p, &v, &t, &l, &B, &A};
    if (!g->legacy)
        CALL(launch(g, 21, B, 1, 256, lossargs));
    else
        CALL(flat(g, 11, B, lossargs));
    for (size_t i = m->nn; i-- > 1;) {
        Node *n = &m->n[i];
        int R = (int)n->r, C = (int)n->c, K = n->k, layer = n->kind == LN;
        uint64_t x = PTR_NODE(m, n->a, 0), dx = PTR_NODE(m, n->a, 1), dy = PTR_NODE(m, i, 1),
                 w = param(m, n->w, 0), dw = param(m, n->w, 1), db = param(m, n->bias, 1),
                 aux = PTR_NODE(m, i, 2);
        if (g->hip && !g->legacy && n->kind == SILU && i > 1) {
            Node *bn = &m->n[i - 1];
            if (bn->kind == BN && n->a == (int)i - 1) {
                uint64_t source = PTR_NODE(m, bn->a, 0), dsource = PTR_NODE(m, bn->a, 1),
                         mid = PTR_NODE(m, i - 1, 0), bn_w = param(m, bn->w, 0),
                         bn_dw = param(m, bn->w, 1), bn_db = param(m, bn->bias, 1),
                         bn_aux = PTR_NODE(m, i - 1, 2);
                void *args[] = {&dsource, &bn_dw, &bn_db, &source, &mid,
                                &dy,      &bn_w,  &bn_aux, &R,      &C};
                CALL(launch(g, 45, (C + 7) / 8, 1, 256, args));
                i--;
                continue;
            }
        }
        if (g->hip && !g->legacy && i > 1) {
            Node *first = &m->n[i - 1];
            int pair = n->kind == SILU && first->kind == ADD && n->a == (int)i - 1   ? 0
                       : n->kind == MUL && first->kind == SILU && n->a == (int)i - 1 ? 1
                                                                                     : -1;
            if (pair >= 0) {
                uint64_t mid = PTR_NODE(m, i - 1, 0), source = PTR_NODE(m, first->a, 0);
                uint64_t other = pair ? PTR_NODE(m, n->b, 0) : PTR_NODE(m, first->b, 0);
                uint64_t dsource = PTR_NODE(m, first->a, 1);
                uint64_t dother = pair ? PTR_NODE(m, n->b, 1) : PTR_NODE(m, first->b, 1);
                uint64_t dummy = 0;
                int count = R * C, back = 1;
                void *args[] = {&mid, &dummy, &dsource, &dother, &source,
                                &other, &dy,   &count,   &pair,    &back};
                CALL(flat(g, 43, (size_t)count, args));
                i--;
                continue;
            }
        }
        if (n->kind == LINEAR || n->kind == CONV) {
            uint64_t input = x, gradient = dx;
            int fused_ci = 0;
            if (n->kind == CONV) {
                int ci = (int)m->n[n->a].c;
                K = ci * n->k * n->k;
                gradient = buffer(g, SCRATCH_BASE + 1, (size_t)R * K * 4, NULL);
                if (!gradient)
                    return -1;
                if (g->hip && !g->legacy && !g->fp32 && !g->integer &&
                    (g->reduced || (R >= 32 && C >= 32 && K >= 32)))
                    fused_ci = -ci;
                else {
                    input = buffer(g, SCRATCH_BASE, (size_t)R * K * 4, NULL);
                    if (!input)
                        return -1;
                    CALL(columns(g, input, x, R, ci, side, n->k, 0));
                }
            }
            if (g->x3_backward)
                g->precise = g->x3_backward == 1 || g->x3_backward == 3 ? 0 : 2;
            CALL(mm_columns(g, dw, dy, input, C, K, R, 1, 0, 1, fused_ci, side, n->k));
            if (g->x3_backward)
                g->precise = g->x3_backward == 2 || g->x3_backward == 3 ? 0 : 2;
            /* FP16 backward is close to the full-gradient gate. Preserve its
             * fast convolution path, but compensate the less numerous linear
             * dX propagations that feed attention and policy/value heads. */
            if (g->hybrid16 && n->kind == LINEAR) {
                g->fp16 = 0;
                g->precise = 2;
            }
            CALL(mm(g, gradient, dy, w, R, K, C, 0, 0, n->kind == LINEAR));
            if (g->hybrid16 && n->kind == LINEAR) {
                g->fp16 = 1;
                g->precise = 0;
            }
            if (n->kind == CONV)
                CALL(columns(g, dx, gradient, R, (int)m->n[n->a].c, side, n->k, 1));
            void *args[] = {&db, &dy, &R, &C};
            if (g->hip && !g->legacy)
                CALL(launch(g, 29, (C + 7) / 8, 1, 256, args));
            else
                CALL(flat(g, 4, C, args));
        } else if (n->kind == BN || n->kind == LN) {
            void *args[] = {&dx, &dw, &db, &x, &dy, &w, &aux, &R, &C, &layer};
            if (g->hip && !g->legacy && !layer)
                CALL(launch(g, 37, (C + 7) / 8, 1, 256, args));
            else if (!g->legacy)
                CALL(launch(g, 19, layer ? R : C, 1, 256, args));
            else
                CALL(flat(g, 7, layer ? R : C, args));
            if (layer) {
                void *pa[] = {&dw, &db, &x, &dy, &aux, &R, &C};
                if (!g->legacy)
                    CALL(launch(g, 20, C, 1, 256, pa));
                else
                    CALL(flat(g, 8, C, pa));
            }
        } else if (n->kind == ATTENTION) {
            if (!g->legacy) {
                uint64_t ds =
                    buffer(g, SCRATCH_BASE + 6, (size_t)R * (C / D) * side * side * 4, NULL);
                if (!ds)
                    return -1;
                void *score[] = {&ds, &x, &dy, &aux, &B, &side, &C, &D};
                if (g->hip && side == 9 && D == 32)
                    CALL(launch(g, 39, B * (C / D) * 11, 1, 256, score));
                else
                    CALL(launch(g, 25, R * (C / D), 1, 256, score));
                void *qkv[] = {&dx, &x, &dy, &aux, &ds, &B, &side, &C, &D};
                CALL(flat(g, 26, (size_t)R * C, qkv));
                void *relative[] = {&dw, &ds, &B, &side, &C, &D};
                CALL(launch(g, 27, (2 * side - 1) * (2 * side - 1) * (C / D), 1, 256, relative));
            } else {
                void *args[] = {&dx, &dw, &x, &dy, &aux, &B, &side, &C, &D};
                CALL(flat(g, 10, (size_t)R * (C / D), args));
            }
        } else {
            uint64_t other = n->b >= 0 ? PTR_NODE(m, n->b, 0) : 0,
                     dother = n->b >= 0 ? PTR_NODE(m, n->b, 1) : 0, dummy = 0;
            int count = R * C, op = n->kind, back = 1;
            void *args[] = {&dummy, &dx, &dother, &x, &other, &dy, &count, &op, &back};
            CALL(flat(g, 5, (size_t)count, args));
        }
    }
    float *values = malloc((size_t)B * 8);
    if (!values)
        return gn_fail("GPU metrics allocation failed");
    int rc = copy_from(g, values, loss, (size_t)B * 8);
    if (!rc) {
        double lp = 0, lv = 0;
        for (int i = 0; i < B; i++) {
            lp += values[2 * i];
            lv += values[2 * i + 1];
        }
        if (!isfinite(lp) || !isfinite(lv)) {
            free(values);
            return gn_fail("non-finite GPU training loss");
        }
        metrics->policy = (float)(lp / B);
        metrics->value = (float)(lv / B);
        metrics->step = m->step;
        metrics->grad_norm = 0;
    }
    free(values);
    return rc;
}
int gn_gpu_update(gn_model *m, float lr, float decay, float clip, gn_metrics *metrics) {
    Gpu *g = m->gpu;
    g->synced = 0;
    CALL(current(g));
    uint64_t norm = buffer(g, SCRATCH_BASE + 5, 8, NULL);
    if (!norm)
        return -1;
    CALL(zero(g, norm, 8));
    float inv = 1.0f / m->accumulated;
    for (size_t i = 0; i < m->np; i++)
        if (m->p[i].learned) {
            int count = (int)(m->p[i].r * m->p[i].c);
            uint64_t grad = param(m, &m->p[i], 1);
            void *args[] = {&norm, &grad, &count, &inv};
            if (g->hip && !g->legacy) {
                unsigned blocks = (count + 255) / 256;
                CALL(launch(g, 30, blocks > 64 ? 64 : blocks, 1, 256, args));
            } else
                CALL(flat(g, 12, (size_t)count, args));
        }
    float result[2];
    CALL(copy_from(g, result, norm, 8));
    if (result[1] || !isfinite(result[0]))
        return gn_fail("non-finite GPU gradient; optimizer not advanced");
    float length = sqrtf(result[0]), scale = inv * (length > clip ? clip / length : 1);
    uint64_t step = m->step + 1;
    float b1 = 1 - (float)pow(.9, (double)step), b2 = 1 - (float)pow(.999, (double)step);
    for (size_t i = 0; i < m->np; i++)
        if (m->p[i].learned) {
            int count = (int)(m->p[i].r * m->p[i].c);
            uint64_t x = param(m, &m->p[i], 0), grad = param(m, &m->p[i], 1),
                     mom = param(m, &m->p[i], 2), var = param(m, &m->p[i], 3);
            void *args[] = {&x, &mom, &var, &grad, &count, &scale, &lr, &decay, &b1, &b2};
            CALL(flat(g, 13, (size_t)count, args));
        }
    int rc = g->hip ? (int)hipDeviceSynchronize() : (int)cuCtxSynchronize();
    if (rc)
        return gn_fail("GPU optimizer execution failed");
    m->step = step;
    m->accumulated = 0;
    if (metrics) {
        metrics->grad_norm = length;
        metrics->step = step;
    }
    return 0;
}
int gn_gpu_sync(gn_model *m) {
    Gpu *g = m->gpu;
    if (g->synced)
        return 0;
    CALL(current(g));
    for (size_t i = 0; i < m->np; i++) {
        Param *p = &m->p[i];
        float *dest[] = {p->x, p->g, p->m, p->v};
        for (int j = 0; j < (p->learned ? 4 : 1); j++)
            if (g->b[4 * i + j].ptr)
                CALL(copy_from(g, dest[j], g->b[4 * i + j].ptr, p->r * p->c * 4));
    }
    g->synced = 1;
    return 0;
}
int gn_gpu_debug_nodes(gn_model *m) {
    Gpu *g = m->gpu;
    if (!g || !m->training)
        return gn_fail("node snapshot requires a GPU training graph");
    CALL(current(g));
    for (size_t i = 0; i < m->nn; i++) {
        Node *n = &m->n[i];
        CALL(copy_from(g, n->x, PTR_NODE(m, i, 0), n->r * n->c * 4));
        CALL(copy_from(g, n->g, PTR_NODE(m, i, 1), n->r * n->c * 4));
    }
    return 0;
}
void gn_gpu_zero(gn_model *m) {
    Gpu *g = m->gpu;
    g->synced = 0;
    if (current(g))
        return;
    for (size_t i = 0; i < m->np; i++)
        if (m->p[i].learned && g->b[4 * i + 1].ptr)
            zero(g, g->b[4 * i + 1].ptr, m->p[i].r * m->p[i].c * 4);
}
