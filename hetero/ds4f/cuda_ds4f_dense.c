#include "cuda_ds4f_dense.h"
#include "../../cuda/cuew.h"
#include "../../cuda/cublasew.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    CUdeviceptr w, s, sf;
    const void *hw;
    int rows, cols, scale_cols, kind;
} cuda_ds4f_matrix;

struct cuda_ds4f_dense {
    CUdevice dev;
    CUcontext context;
    CUmodule module;
    CUfunction gemm;
    CUfunction quant;
    CUfunction swiglu;
    CUstream stream;
    cublasew_context *blas;
    cuda_ds4f_matrix *mat;
    int nmat, cap;
    CUdeviceptr dx, dy;
    float *hx, *hy;
    size_t hxcap, hycap;
    CUdeviceptr dxq, dxs;
    size_t xqcap, xscap;
    CUdeviceptr ffn_x, ffn_g, ffn_u, ffn_y;
    size_t ffn_xcap, ffn_gcap, ffn_ucap, ffn_ycap;
    size_t xcap, ycap, resident;
    int verbose;
};

static int cu_ok(CUresult rc, const char *what) {
    if (rc == CUDA_SUCCESS) return 0;
    const char *s = NULL; if (cuGetErrorString) cuGetErrorString(rc, &s);
    fprintf(stderr, "cuda_ds4f_dense: %s: %s (%d)\n", what, s ? s : "error", (int)rc);
    return -1;
}

cuda_ds4f_dense *cuda_ds4f_dense_create(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS || cuInit(0) != CUDA_SUCCESS) return NULL;
    cuda_ds4f_dense *c = (cuda_ds4f_dense *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->verbose = verbose;
    if (cu_ok(cuDeviceGet(&c->dev, device_id), "device") ||
        cu_ok(cuDevicePrimaryCtxRetain(&c->context, c->dev), "context") ||
        cu_ok(cuCtxSetCurrent(c->context), "set context") ||
        cu_ok(cuStreamCreate(&c->stream, CU_STREAM_NON_BLOCKING), "stream")) goto fail;
    CUresult rc = cuModuleLoad(&c->module, "cuda/llm/ds4f_dense_kernels.cubin");
    if (rc != CUDA_SUCCESS) rc = cuModuleLoad(&c->module, "../../cuda/llm/ds4f_dense_kernels.cubin");
    if (cu_ok(rc, "load ds4f_dense_kernels.cubin") ||
        cu_ok(cuModuleGetFunction(&c->gemm, c->module, "ds4f_cuda_dense_gemm"), "dense gemm") ||
        cu_ok(cuModuleGetFunction(&c->quant, c->module, "ds4f_cuda_quant_fp8_vec128"), "fp8 quant") ||
        cu_ok(cuModuleGetFunction(&c->swiglu, c->module, "ds4f_cuda_swiglu"), "swiglu")) goto fail;
    if (cublasewInit() != 0 || cublasewCreate(&c->blas, c->stream) != 0) goto fail;
    return c;
fail:
    cuda_ds4f_dense_destroy(c); return NULL;
}

void cuda_ds4f_dense_destroy(cuda_ds4f_dense *c) {
    if (!c) return;
    if (c->context) cuCtxSetCurrent(c->context);
    if (c->stream) cuStreamSynchronize(c->stream);
    for (int i = 0; i < c->nmat; ++i) { if (c->mat[i].sf) cuMemFree(c->mat[i].sf); if (c->mat[i].s) cuMemFree(c->mat[i].s); if (c->mat[i].w) cuMemFree(c->mat[i].w); }
    if (c->blas) cublasewDestroy(c->blas);
    if (c->dxs) cuMemFree(c->dxs);
    if (c->dxq) cuMemFree(c->dxq);
    if (c->dy) cuMemFree(c->dy);
    if (c->dx) cuMemFree(c->dx);
    if (c->ffn_y) cuMemFree(c->ffn_y);
    if (c->ffn_u) cuMemFree(c->ffn_u);
    if (c->ffn_g) cuMemFree(c->ffn_g);
    if (c->ffn_x) cuMemFree(c->ffn_x);
    if (c->module) cuModuleUnload(c->module);
    if (c->stream) cuStreamDestroy(c->stream);
    if (c->context) cuDevicePrimaryCtxRelease(c->dev);
    free(c->hy); free(c->hx); free(c->mat); free(c);
}

int cuda_ds4f_dense_bind_tensor(cuda_ds4f_dense *c, ds4f_tensor *t) {
    if (!c || !t || !t->w || (t->type != DS4F_FP8 && t->type != DS4F_BF16)) return -1;
    if (cuCtxSetCurrent(c->context) != CUDA_SUCCESS) return -1;
    int kind = t->type == DS4F_FP8 ? 0 : 1;
    size_t wb = (size_t)t->rows * t->cols * (kind ? 2u : 1u);
    int sc = kind ? 0 : (t->cols + 127) / 128;
    size_t sb = kind ? 0 : (size_t)((t->rows + 127) / 128) * sc;
    size_t nsf = 0;
    CUdeviceptr dw = 0, ds = 0, dsf = 0;
    if (cuMemAlloc(&dw, wb) != CUDA_SUCCESS || cuMemcpyHtoD(dw, t->w, wb) != CUDA_SUCCESS ||
        (sb && (cuMemAlloc(&ds, sb) != CUDA_SUCCESS || cuMemcpyHtoD(ds, t->scale, sb) != CUDA_SUCCESS))) {
        if (ds) cuMemFree(ds);
        if (dw) cuMemFree(dw);
        return -1;
    }
    if (sb) {
        int nb=(t->cols+31)/32, nti=(nb+3)/4;
        nsf=(size_t)((t->rows+127)/128)*nti*512;
        uint8_t *hsf = (uint8_t *)calloc(nsf,1);
        if (!hsf) { cuMemFree(ds); cuMemFree(dw); return -1; }
        for (int r=0;r<t->rows;++r) for(int b=0;b<nb;++b){
            size_t off=(size_t)((b>>2)+(r>>7)*nti)*512+
                       (size_t)(r&31)*16+(size_t)((r&127)>>5)*4+(b&3);
            hsf[off]=t->scale[(size_t)(r/128)*sc+b/4];
        }
        if (cuMemAlloc(&dsf, nsf) != CUDA_SUCCESS || cuMemcpyHtoD(dsf, hsf, nsf) != CUDA_SUCCESS) {
            free(hsf); if (dsf) cuMemFree(dsf); cuMemFree(ds); cuMemFree(dw); return -1;
        }
        free(hsf);
    }
    if (c->nmat == c->cap) {
        int cap = c->cap ? c->cap * 2 : 16;
        cuda_ds4f_matrix *p = (cuda_ds4f_matrix *)realloc(c->mat, (size_t)cap * sizeof(*p));
        if (!p) { if (dsf) cuMemFree(dsf); if (ds) cuMemFree(ds); cuMemFree(dw); return -1; }
        c->mat = p; c->cap = cap;
    }
    int id = c->nmat++;
    c->mat[id] = (cuda_ds4f_matrix){dw, ds, dsf, t->w, t->rows, t->cols, sc, kind};
    c->resident += wb + sb + (kind ? 0 : nsf); t->gpu_id = id;
    if (c->verbose) fprintf(stderr, "cuda_ds4f_dense: bind id=%d %dx%d kind=%d resident=%.3f GB\n", id, t->rows, t->cols, kind, c->resident/1e9);
    return id;
}

static int grow(CUdeviceptr *p, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    if (*p) cuMemFree(*p);
    *p = 0; *cap = 0;
    if (cuMemAlloc(p, need) != CUDA_SUCCESS) return -1;
    *cap = need; return 0;
}

static int grow_host(float **p, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    void *q = realloc(*p, need);
    if (!q) return -1;
    *p = (float *)q;
    *cap = need;
    return 0;
}

static int native_gemm_device(cuda_ds4f_dense *c, CUdeviceptr dy,
                              const cuda_ds4f_matrix *a, CUdeviceptr dx,
                              int M) {
    int N=a->rows,K=a->cols;
    if(a->kind!=0)return -1;
    int nb=(K+31)/32,nti=(nb+3)/4;
    size_t qb=(size_t)M*K,sb=(size_t)((M+127)/128)*nti*512;
    if(grow(&c->dxq,&c->xqcap,qb)||grow(&c->dxs,&c->xscap,sb))return -1;
    cuMemsetD8Async(c->dxs,0,sb,c->stream);
    void *qa[]={&c->dxq,&c->dxs,&dx,&M,&K};
    if(cuLaunchKernel(c->quant,(K+31)/32,M,1,32,1,1,0,c->stream,qa,NULL)!=CUDA_SUCCESS)return -1;
    return cublasew_gemm_fp8_scaled_rowmajor_nt(c->blas,dy,a->w,a->sf,
                                                 c->dxq,c->dxs,M,N,K);
}

int cuda_ds4f_dense_gemm_tensor(void *opaque, float *dst,
                                const ds4f_tensor *t, const float *x,
                                int M, int Ys, int Xs) {
    cuda_ds4f_dense *c = (cuda_ds4f_dense *)opaque;
    if (!c || !dst || !t || !x || t->gpu_id < 0 || t->gpu_id >= c->nmat ||
        M < 1 || Ys < t->rows || Xs < t->cols) return -1;
    cuda_ds4f_matrix view = c->mat[t->gpu_id], *a = &view;
    if (a->cols != t->cols) return -1;
    ptrdiff_t bytes=(const uint8_t *)t->w-(const uint8_t *)a->hw;
    size_t es=a->kind?2u:1u,rowb=(size_t)a->cols*es;
    if(bytes<0||(size_t)bytes%rowb)return -1;
    int r0=(int)((size_t)bytes/rowb);
    if(r0<0||r0+t->rows>a->rows||(!a->kind&&(r0&127)))return -1;
    a->w+=(size_t)r0*rowb;a->rows=t->rows;a->hw=t->w;
    if(!a->kind){int nti=(((a->cols+31)/32)+3)/4;a->s+=(size_t)(r0/128)*a->scale_cols;a->sf+=(size_t)(r0/128)*nti*512;}
    size_t xb=(size_t)M*t->cols*4, yb=(size_t)M*t->rows*4;
    if (cuCtxSetCurrent(c->context) != CUDA_SUCCESS || grow(&c->dx,&c->xcap,xb) || grow(&c->dy,&c->ycap,yb)) return -1;
    const float *xin=x;
    if(Xs!=t->cols){
        if(grow_host(&c->hx,&c->hxcap,xb))return -1;
        for(int mm=0;mm<M;++mm)
            memcpy(c->hx+(size_t)mm*t->cols,x+(size_t)mm*Xs,(size_t)t->cols*4);
        xin=c->hx;
    }
    if (cuMemcpyHtoDAsync(c->dx,xin,xb,c->stream) != CUDA_SUCCESS) return -1;
    int N=t->rows,K=t->cols,kind=a->kind,sc=a->scale_cols;
    if (kind == 0) {
        int brc=native_gemm_device(c,c->dy,a,c->dx,M);
        static int reported=0;
        if(c->verbose&&!reported++){fprintf(stderr,"cuda_ds4f_dense: native scaled FP8 %s\n",brc==0?"enabled":"unavailable; using WMMA fallback");}
        if (brc==0) goto download;
    }
    void *args[]={&c->dy,&a->w,&a->s,&c->dx,&N,&K,&M,&sc,&kind};
    if (cuLaunchKernel(c->gemm,(N+31)/32,(M+63)/64,1,256,1,1,0,c->stream,args,NULL)!=CUDA_SUCCESS) return -1;
download:;
    float *yout=dst;
    if(Ys!=t->rows){if(grow_host(&c->hy,&c->hycap,yb))return -1;yout=c->hy;}
    if(cuMemcpyDtoHAsync(yout,c->dy,yb,c->stream)!=CUDA_SUCCESS||cuStreamSynchronize(c->stream)!=CUDA_SUCCESS)return -1;
    if(yout!=dst)for(int mm=0;mm<M;++mm)
        memcpy(dst+(size_t)mm*Ys,yout+(size_t)mm*t->rows,(size_t)t->rows*4);
    return 0;
}

int cuda_ds4f_dense_shared_ffn(void *opaque, float *dst,
                               const ds4f_tensor *w1,
                               const ds4f_tensor *w3,
                               const ds4f_tensor *w2,
                               const float *x, int M, int inter, int C,
                               float lim) {
    cuda_ds4f_dense *c=(cuda_ds4f_dense *)opaque;
    if(!c||!dst||!x||!w1||!w3||!w2||w1->gpu_id<0||w3->gpu_id<0||w2->gpu_id<0||
       w1->gpu_id>=c->nmat||w3->gpu_id>=c->nmat||w2->gpu_id>=c->nmat)return -1;
    cuda_ds4f_matrix *a=&c->mat[w1->gpu_id],*b=&c->mat[w3->gpu_id],*d=&c->mat[w2->gpu_id];
    if(a->rows!=inter||a->cols!=C||b->rows!=inter||b->cols!=C||d->rows!=C||d->cols!=inter)return -1;
    size_t xb=(size_t)M*C*4,ib=(size_t)M*inter*4;
    if(cuCtxSetCurrent(c->context)!=CUDA_SUCCESS||grow(&c->ffn_x,&c->ffn_xcap,xb)||
       grow(&c->ffn_g,&c->ffn_gcap,ib)||grow(&c->ffn_u,&c->ffn_ucap,ib)||
       grow(&c->ffn_y,&c->ffn_ycap,xb))return -1;
    if(cuMemcpyHtoDAsync(c->ffn_x,x,xb,c->stream)!=CUDA_SUCCESS||
       native_gemm_device(c,c->ffn_g,a,c->ffn_x,M)||
       native_gemm_device(c,c->ffn_u,b,c->ffn_x,M))return -1;
    size_t ne=(size_t)M*inter;void *sa[]={&c->ffn_g,&c->ffn_u,&ne,&lim};
    if(cuLaunchKernel(c->swiglu,(unsigned)((ne+255)/256),1,1,256,1,1,0,c->stream,sa,NULL)!=CUDA_SUCCESS||
       native_gemm_device(c,c->ffn_y,d,c->ffn_g,M)||
       cuMemcpyDtoHAsync(dst,c->ffn_y,xb,c->stream)!=CUDA_SUCCESS||
       cuStreamSynchronize(c->stream)!=CUDA_SUCCESS)return -1;
    return 0;
}
