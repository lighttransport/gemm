/* SPDX-License-Identifier: MIT */
#include "../cuda/cuew.h"
#include "../rdna4/rocew.h"
#include "gn_internal.h"
#include "gn_kernels.inc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NODE_BASE (4 * GN_PARAMS)
#define SCRATCH_BASE (NODE_BASE + 3 * GN_NODES)
#define SLOT_COUNT (SCRATCH_BASE + 8)
typedef struct {
    uint64_t ptr;
    size_t bytes;
} Buffer;
typedef struct {
    int hip, device, active, fp32;
    size_t used, limit;
    CUcontext context;
    CUmodule cuda_module;
    hipModule_t hip_module;
    Buffer b[SLOT_COUNT];
    void *functions[16];
} Gpu;
static const char *names[] = {"gn_mm",          "gn_columns",   "gn_uncolumns",      "gn_bias",
                              "gn_bias_back",   "gn_point",     "gn_norm",           "gn_norm_back",
                              "gn_layer_param", "gn_attention", "gn_attention_back", "gn_loss",
                              "gn_grad_norm",   "gn_adam",      "gn_mm_fp32"};
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
static int mm(Gpu *g, uint64_t y, uint64_t a, uint64_t b, int M, int N, int K, int ta, int tb,
              int add) {
    void *args[] = {&y, &a, &b, &M, &N, &K, &ta, &tb, &add};
    if (g->fp32)
        return flat(g, 14, (size_t)M * N, args);
    return launch(g, 0, (unsigned)((N + (g->hip ? 15 : 7)) / (g->hip ? 16 : 8)),
                  (unsigned)((M + 15) / 16), 32, args);
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
    g->device = device;
    g->limit = limit;
    int rc = 0;
    if (g->hip) {
        if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS ||
            hipInit(0) != hipSuccess || hipSetDevice(device) != hipSuccess) {
            gn_fail("HIP/HIPRTC or GPU unavailable (requires gfx1201)");
            goto bad;
        }
        g->active = 1;
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
    for (int i = 0; i < 15; i++) {
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
    CALL(prepare(m));
    CALL(copy_to(g, PTR_NODE(m, 0, 0), input, m->n[0].r * m->n[0].c * 4));
    int side = (int)m->cfg.side, B = (int)m->batch, D = (int)m->cfg.head_dim,
        training = m->training;
    for (size_t i = 1; i < m->nn; i++) {
        Node *n = &m->n[i];
        int R = (int)n->r, C = (int)n->c, K = n->k, layer = n->kind == LN;
        uint64_t x = PTR_NODE(m, n->a, 0), y = PTR_NODE(m, i, 0), w = param(m, n->w, 0),
                 bias = param(m, n->bias, 0), aux = PTR_NODE(m, i, 2);
        if (n->kind == LINEAR || n->kind == CONV) {
            if (n->kind == CONV) {
                int ci = (int)m->n[n->a].c;
                K = ci * n->k * n->k;
                uint64_t col = buffer(g, SCRATCH_BASE, (size_t)R * K * 4, NULL);
                if (!col)
                    return -1;
                CALL(columns(g, col, x, R, ci, side, n->k, 0));
                x = col;
            }
            CALL(mm(g, y, x, w, R, C, K, 0, 1, 0));
            void *args[] = {&y, &bias, &R, &C};
            CALL(flat(g, 3, (size_t)R * C, args));
        } else if (n->kind == BN || n->kind == LN) {
            uint64_t mean = param(m, n->mean, 0), var = param(m, n->variance, 0);
            void *args[] = {&y, &aux, &mean, &var, &x, &w, &bias, &R, &C, &layer, &training};
            CALL(flat(g, 6, layer ? R : C, args));
        } else if (n->kind == ATTENTION) {
            void *args[] = {&y, &aux, &x, &w, &B, &side, &C, &D};
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
    CALL(flat(g, 11, B, lossargs));
    for (size_t i = m->nn; i-- > 1;) {
        Node *n = &m->n[i];
        int R = (int)n->r, C = (int)n->c, K = n->k, layer = n->kind == LN;
        uint64_t x = PTR_NODE(m, n->a, 0), dx = PTR_NODE(m, n->a, 1), dy = PTR_NODE(m, i, 1),
                 w = param(m, n->w, 0), dw = param(m, n->w, 1), db = param(m, n->bias, 1),
                 aux = PTR_NODE(m, i, 2);
        if (n->kind == LINEAR || n->kind == CONV) {
            uint64_t input = x, gradient = dx;
            if (n->kind == CONV) {
                int ci = (int)m->n[n->a].c;
                K = ci * n->k * n->k;
                input = buffer(g, SCRATCH_BASE, (size_t)R * K * 4, NULL);
                gradient = buffer(g, SCRATCH_BASE + 1, (size_t)R * K * 4, NULL);
                if (!input || !gradient)
                    return -1;
                CALL(columns(g, input, x, R, ci, side, n->k, 0));
            }
            CALL(mm(g, dw, dy, input, C, K, R, 1, 0, 1));
            CALL(mm(g, gradient, dy, w, R, K, C, 0, 0, n->kind == LINEAR));
            if (n->kind == CONV)
                CALL(columns(g, dx, gradient, R, (int)m->n[n->a].c, side, n->k, 1));
            void *args[] = {&db, &dy, &R, &C};
            CALL(flat(g, 4, C, args));
        } else if (n->kind == BN || n->kind == LN) {
            void *args[] = {&dx, &dw, &db, &x, &dy, &w, &aux, &R, &C, &layer};
            CALL(flat(g, 7, layer ? R : C, args));
            if (layer) {
                void *pa[] = {&dw, &db, &x, &dy, &aux, &R, &C};
                CALL(flat(g, 8, C, pa));
            }
        } else if (n->kind == ATTENTION) {
            void *args[] = {&dx, &dw, &x, &dy, &aux, &B, &side, &C, &D};
            CALL(flat(g, 10, (size_t)R * (C / D), args));
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
    CALL(current(g));
    for (size_t i = 0; i < m->np; i++) {
        Param *p = &m->p[i];
        float *dest[] = {p->x, p->g, p->m, p->v};
        for (int j = 0; j < (p->learned ? 4 : 1); j++)
            if (g->b[4 * i + j].ptr)
                CALL(copy_from(g, dest[j], g->b[4 * i + j].ptr, p->r * p->c * 4));
    }
    return 0;
}
void gn_gpu_zero(gn_model *m) {
    Gpu *g = m->gpu;
    if (current(g))
        return;
    for (size_t i = 0; i < m->np; i++)
        if (m->p[i].learned && g->b[4 * i + 1].ptr)
            zero(g, g->b[4 * i + 1].ptr, m->p[i].r * m->p[i].c * 4);
}
