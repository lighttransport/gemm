#ifndef K3_DENSE_H
#define K3_DENSE_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "ggml_dequant.h"

typedef struct {
    const uint16_t *weight;
    int rows;
    int cols;
} k3_bf16_matrix;

static inline void k3_pack_bf16_pv(uint16_t*dst,const uint16_t*src,int rows,int cols){
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(int g=0;g<rows/8;++g){uint16_t*d=dst+(size_t)g*8*cols;
        for(int p=0;p<4;++p){const uint16_t*a=src+(size_t)(g*8+2*p)*cols,*b=a+cols;uint16_t*q=d+(size_t)p*2*cols;
            for(int j=0;j<cols;++j){q[2*j]=a[j];q[2*j+1]=b[j];}}}
}
static inline void k3_matvec_bf16_pv(float*out,const uint16_t*pv,int rows,int cols,const float*x,int threads){
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for(int g=0;g<rows/8;++g){const uint16_t*p=pv+(size_t)g*8*cols;
        matvec_bf16_8row_pv(out+g*8,p,p+2*cols,p+4*cols,p+6*cols,x,cols);}
}

/* Row-wise symmetric W8.  A single dynamic activation scale is deliberately
 * used for decode: both routed_down and routed_up consume RMS-normalized
 * vectors, and the full K reduction remains safely inside int32. */
typedef struct {
    const int8_t *weight;
    const float *scale;
    int rows;
    int cols;
} k3_q8_matrix;

/* Eight-row, group-64 Q8 layout shared with the proven DS4F sdot kernel:
 * each block is [8 fp16 row scales][8 x 64 int8 weights]. */
typedef struct {
    const uint8_t *data;
    int rows;
    int cols;
} k3_q8pv_matrix;

static inline size_t k3_q8pv_matrix_bytes(int rows, int cols) {
    return (size_t)(rows / 8) * (cols / 64) * 528;
}

static inline void k3_q8pv_quantize_bf16(uint8_t *dst, const uint16_t *src,
                                          int rows, int cols) {
    int groups=rows/8,blocks=cols/64;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(int g=0;g<groups;++g)for(int b=0;b<blocks;++b){
        uint8_t*blk=dst+((size_t)g*blocks+b)*528;
        uint16_t*scale=(uint16_t*)blk;int8_t*q=(int8_t*)(blk+16);
        for(int r=0;r<8;++r){const uint16_t*w=src+(size_t)(g*8+r)*cols+b*64;
            float amax=0;for(int j=0;j<64;++j)amax=fmaxf(amax,fabsf(bf16_to_f32_scalar(w[j])));
            float s=amax>0?amax/127.0f:1.0f,inv=1.0f/s;scale[r]=ggml_fp32_to_fp16(s);
            for(int j=0;j<64;++j){long v=lrintf(bf16_to_f32_scalar(w[j])*inv);
                q[r*64+j]=(int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);}}
    }
}

static inline void k3_q8pv_quantize_vector(int8_t*xq,float*xs,const float*x,int n){
    for(int b=0;b<n/64;++b){const float*p=x+(size_t)b*64;float amax=0;
        for(int j=0;j<64;++j)amax=fmaxf(amax,fabsf(p[j]));float s=amax>0?amax/127.0f:1.0f,inv=1.0f/s;xs[b]=s;
        for(int j=0;j<64;++j){long v=lrintf(p[j]*inv);xq[(size_t)b*64+j]=(int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);}}
}

static inline void k3_matvec_q8pv(float*out,const k3_q8pv_matrix*m,
                                   const float*x,int8_t*xq,float*xs,int threads){
    int blocks=m->cols/64;size_t group_bytes=(size_t)blocks*528;
    k3_q8pv_quantize_vector(xq,xs,x,m->cols);
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for(int g=0;g<m->rows/8;++g)matvec_sdot_8row(out+g*8,
        m->data+(size_t)g*group_bytes,xq,xs,m->cols);
}

/* Quality-gated variant: group-32 with FP32 scales.  Per eight rows, each
 * block is [8 f32 scales][8 x 32 int8]. */
static inline size_t k3_q8pv32_matrix_bytes(int rows,int cols){
    return(size_t)(rows/8)*(cols/32)*288;
}
static inline void k3_q8pv32_quantize_bf16(uint8_t*dst,const uint16_t*src,int rows,int cols){
    int groups=rows/8,blocks=cols/32;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(int g=0;g<groups;++g)for(int b=0;b<blocks;++b){uint8_t*blk=dst+((size_t)g*blocks+b)*288;float*sc=(float*)blk;int8_t*q=(int8_t*)(blk+32);
        for(int r=0;r<8;++r){const uint16_t*w=src+(size_t)(g*8+r)*cols+b*32;float amax=0;
            for(int j=0;j<32;++j)amax=fmaxf(amax,fabsf(bf16_to_f32_scalar(w[j])));float s=amax>0?amax/127.0f:1.0f,inv=1.0f/s;sc[r]=s;
            for(int j=0;j<32;++j){long v=lrintf(bf16_to_f32_scalar(w[j])*inv);q[r*32+j]=(int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);}}}
}
static inline void k3_q8pv32_quantize_vector(int8_t*q,float*sc,const float*x,int n){
    for(int b=0;b<n/32;++b){float amax=0;for(int j=0;j<32;++j)amax=fmaxf(amax,fabsf(x[b*32+j]));float s=amax>0?amax/127.0f:1.0f,inv=1.0f/s;sc[b]=s;
        for(int j=0;j<32;++j){long v=lrintf(x[b*32+j]*inv);q[b*32+j]=(int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);}}
}
static inline void k3_matvec_q8pv32_group(float*out,const uint8_t*group,const int8_t*xq,const float*xs,int k){
#if defined(__ARM_FEATURE_SVE)
    svbool_t ps=svptrue_b32(),pb=svwhilelt_b8(0,32);svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    for(int b=0;b<k/32;++b){const uint8_t*blk=group+(size_t)b*288;const float*sc=(const float*)blk;const int8_t*q=(const int8_t*)(blk+32);svint8_t xv=svld1_s8(pb,xq+(size_t)b*32);
#define K3_Q8PV32_ROW(R,A) do{svint32_t d=svdot_s32(svdup_s32(0),svld1_s8(pb,q+(size_t)(R)*32),xv);A=svmla_n_f32_x(ps,A,svcvt_f32_s32_x(ps,d),sc[R]*xs[b]);}while(0)
        K3_Q8PV32_ROW(0,a0);K3_Q8PV32_ROW(1,a1);K3_Q8PV32_ROW(2,a2);K3_Q8PV32_ROW(3,a3);K3_Q8PV32_ROW(4,a4);K3_Q8PV32_ROW(5,a5);K3_Q8PV32_ROW(6,a6);K3_Q8PV32_ROW(7,a7);
#undef K3_Q8PV32_ROW
    }out[0]=svaddv(ps,a0);out[1]=svaddv(ps,a1);out[2]=svaddv(ps,a2);out[3]=svaddv(ps,a3);out[4]=svaddv(ps,a4);out[5]=svaddv(ps,a5);out[6]=svaddv(ps,a6);out[7]=svaddv(ps,a7);
#else
    (void)out;(void)group;(void)xq;(void)xs;(void)k;
#endif
}
static inline void k3_matvec_q8pv32(float*out,const k3_q8pv_matrix*m,const float*x,int8_t*xq,float*xs,int threads){
    int blocks=m->cols/32;size_t gb=(size_t)blocks*288;k3_q8pv32_quantize_vector(xq,xs,x,m->cols);
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for(int g=0;g<m->rows/8;++g)k3_matvec_q8pv32_group(out+g*8,m->data+(size_t)g*gb,xq,xs,m->cols);
}

static inline float k3_q8_quantize_vector(int8_t *q, const float *x, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = fmaxf(amax, fabsf(x[i]));
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        long v = lrintf(x[i] * inv);
        q[i] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
    }
    return scale;
}

static inline void k3_q8_quantize_bf16_rows(int8_t *q, float *scale,
                                             const uint16_t *w,
                                             int rows, int cols) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const uint16_t *src = w + (size_t)r * cols;
        int8_t *dst = q + (size_t)r * cols;
        float amax = 0.0f;
        for (int c = 0; c < cols; ++c)
            amax = fmaxf(amax, fabsf(bf16_to_f32_scalar(src[c])));
        float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inv = 1.0f / s;
        scale[r] = s;
        for (int c = 0; c < cols; ++c) {
            long v = lrintf(bf16_to_f32_scalar(src[c]) * inv);
            dst[c] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
        }
    }
}

static inline int32_t k3_q8_dot(const int8_t *a, const int8_t *b, int n) {
#if defined(__ARM_FEATURE_SVE)
    svint32_t a0 = svdup_s32(0), a1 = a0, a2 = a0, a3 = a0;
    int i = 0, vl = (int)svcntb();
    svbool_t pg = svptrue_b8();
    for (; i + 4 * vl <= n; i += 4 * vl) {
        a0 = svdot_s32(a0, svld1_s8(pg, a + i),
                       svld1_s8(pg, b + i));
        a1 = svdot_s32(a1, svld1_s8(pg, a + i + vl),
                       svld1_s8(pg, b + i + vl));
        a2 = svdot_s32(a2, svld1_s8(pg, a + i + 2 * vl),
                       svld1_s8(pg, b + i + 2 * vl));
        a3 = svdot_s32(a3, svld1_s8(pg, a + i + 3 * vl),
                       svld1_s8(pg, b + i + 3 * vl));
    }
    for (; i + vl <= n; i += vl)
        a0 = svdot_s32(a0, svld1_s8(pg, a + i), svld1_s8(pg, b + i));
    svbool_t pg32 = svptrue_b32();
    a0 = svadd_s32_x(pg32, svadd_s32_x(pg32, a0, a1),
                     svadd_s32_x(pg32, a2, a3));
    int32_t sum = svaddv_s32(pg32, a0);
    for (; i < n; ++i) sum += (int32_t)a[i] * b[i];
    return sum;
#else
    int32_t sum = 0;
    for (int i = 0; i < n; ++i) sum += (int32_t)a[i] * b[i];
    return sum;
#endif
}

static inline void k3_q8_dot8(int32_t out[8], const int8_t *w,
                               const int8_t *x, int n) {
#if defined(__ARM_FEATURE_SVE)
    svint32_t a0=svdup_s32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    svbool_t pg=svptrue_b8(),pg32=svptrue_b32();int vl=(int)svcntb();
    for(int i=0;i<n;i+=vl){svint8_t xv=svld1_s8(pg,x+i);
        a0=svdot_s32(a0,svld1_s8(pg,w+(size_t)0*n+i),xv);
        a1=svdot_s32(a1,svld1_s8(pg,w+(size_t)1*n+i),xv);
        a2=svdot_s32(a2,svld1_s8(pg,w+(size_t)2*n+i),xv);
        a3=svdot_s32(a3,svld1_s8(pg,w+(size_t)3*n+i),xv);
        a4=svdot_s32(a4,svld1_s8(pg,w+(size_t)4*n+i),xv);
        a5=svdot_s32(a5,svld1_s8(pg,w+(size_t)5*n+i),xv);
        a6=svdot_s32(a6,svld1_s8(pg,w+(size_t)6*n+i),xv);
        a7=svdot_s32(a7,svld1_s8(pg,w+(size_t)7*n+i),xv);}
    out[0]=svaddv_s32(pg32,a0);out[1]=svaddv_s32(pg32,a1);
    out[2]=svaddv_s32(pg32,a2);out[3]=svaddv_s32(pg32,a3);
    out[4]=svaddv_s32(pg32,a4);out[5]=svaddv_s32(pg32,a5);
    out[6]=svaddv_s32(pg32,a6);out[7]=svaddv_s32(pg32,a7);
#else
    for(int r=0;r<8;++r)out[r]=k3_q8_dot(w+(size_t)r*n,x,n);
#endif
}

static inline void k3_q8_dot16(int32_t out[16], const int8_t *w,
                                const int8_t *x, int n) {
#if defined(__ARM_FEATURE_SVE)
    svint32_t a0=svdup_s32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    svint32_t a8=a0,a9=a0,a10=a0,a11=a0,a12=a0,a13=a0,a14=a0,a15=a0;
    svbool_t pg=svptrue_b8(),p32=svptrue_b32();int vl=(int)svcntb();
    for(int i=0;i<n;i+=vl){svint8_t v=svld1_s8(pg,x+i);
#define K3_Q8_DOT16(R) a##R=svdot_s32(a##R,svld1_s8(pg,w+(size_t)(R)*n+i),v)
        K3_Q8_DOT16(0);K3_Q8_DOT16(1);K3_Q8_DOT16(2);K3_Q8_DOT16(3);
        K3_Q8_DOT16(4);K3_Q8_DOT16(5);K3_Q8_DOT16(6);K3_Q8_DOT16(7);
        K3_Q8_DOT16(8);K3_Q8_DOT16(9);K3_Q8_DOT16(10);K3_Q8_DOT16(11);
        K3_Q8_DOT16(12);K3_Q8_DOT16(13);K3_Q8_DOT16(14);K3_Q8_DOT16(15);
#undef K3_Q8_DOT16
    }
#define K3_Q8_SUM16(R) out[R]=svaddv_s32(p32,a##R)
    K3_Q8_SUM16(0);K3_Q8_SUM16(1);K3_Q8_SUM16(2);K3_Q8_SUM16(3);
    K3_Q8_SUM16(4);K3_Q8_SUM16(5);K3_Q8_SUM16(6);K3_Q8_SUM16(7);
    K3_Q8_SUM16(8);K3_Q8_SUM16(9);K3_Q8_SUM16(10);K3_Q8_SUM16(11);
    K3_Q8_SUM16(12);K3_Q8_SUM16(13);K3_Q8_SUM16(14);K3_Q8_SUM16(15);
#undef K3_Q8_SUM16
#else
    k3_q8_dot8(out,w,x,n);k3_q8_dot8(out+8,w+(size_t)8*n,x,n);
#endif
}

static inline void k3_q8_dot24(int32_t out[24], const int8_t *w,
                                const int8_t *x, int n) {
#if defined(__ARM_FEATURE_SVE)
    svint32_t a0=svdup_s32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    svint32_t a8=a0,a9=a0,a10=a0,a11=a0,a12=a0,a13=a0,a14=a0,a15=a0;
    svint32_t a16=a0,a17=a0,a18=a0,a19=a0,a20=a0,a21=a0,a22=a0,a23=a0;
    svbool_t pg=svptrue_b8(),p32=svptrue_b32();int vl=(int)svcntb();
    for(int i=0;i<n;i+=vl){svint8_t v=svld1_s8(pg,x+i);
#define K3_Q8_DOT24(R) a##R=svdot_s32(a##R,svld1_s8(pg,w+(size_t)(R)*n+i),v)
        K3_Q8_DOT24(0);K3_Q8_DOT24(1);K3_Q8_DOT24(2);K3_Q8_DOT24(3);
        K3_Q8_DOT24(4);K3_Q8_DOT24(5);K3_Q8_DOT24(6);K3_Q8_DOT24(7);
        K3_Q8_DOT24(8);K3_Q8_DOT24(9);K3_Q8_DOT24(10);K3_Q8_DOT24(11);
        K3_Q8_DOT24(12);K3_Q8_DOT24(13);K3_Q8_DOT24(14);K3_Q8_DOT24(15);
        K3_Q8_DOT24(16);K3_Q8_DOT24(17);K3_Q8_DOT24(18);K3_Q8_DOT24(19);
        K3_Q8_DOT24(20);K3_Q8_DOT24(21);K3_Q8_DOT24(22);K3_Q8_DOT24(23);
#undef K3_Q8_DOT24
    }
#define K3_Q8_SUM24(R) out[R]=svaddv_s32(p32,a##R)
    K3_Q8_SUM24(0);K3_Q8_SUM24(1);K3_Q8_SUM24(2);K3_Q8_SUM24(3);
    K3_Q8_SUM24(4);K3_Q8_SUM24(5);K3_Q8_SUM24(6);K3_Q8_SUM24(7);
    K3_Q8_SUM24(8);K3_Q8_SUM24(9);K3_Q8_SUM24(10);K3_Q8_SUM24(11);
    K3_Q8_SUM24(12);K3_Q8_SUM24(13);K3_Q8_SUM24(14);K3_Q8_SUM24(15);
    K3_Q8_SUM24(16);K3_Q8_SUM24(17);K3_Q8_SUM24(18);K3_Q8_SUM24(19);
    K3_Q8_SUM24(20);K3_Q8_SUM24(21);K3_Q8_SUM24(22);K3_Q8_SUM24(23);
#undef K3_Q8_SUM24
#else
    k3_q8_dot16(out,w,x,n);k3_q8_dot8(out+16,w+(size_t)16*n,x,n);
#endif
}

static inline void k3_matvec_q8(float *out, const k3_q8_matrix *m,
                                 const float *x, int8_t *qx, int threads) {
    float xs = k3_q8_quantize_vector(qx, x, m->cols);
    int groups = (m->rows + 23) / 24;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for (int group = 0; group < groups; ++group) {
        int r=group*24,left=m->rows-r;int32_t dot[24];
        if(left>=24)k3_q8_dot24(dot,m->weight+(size_t)r*m->cols,qx,m->cols);
        else if(left>=16)k3_q8_dot16(dot,m->weight+(size_t)r*m->cols,qx,m->cols);
        else k3_q8_dot8(dot,m->weight+(size_t)r*m->cols,qx,m->cols);
        int nr=left<24?left:24;
        for (int j = 0; j < nr; ++j)
            out[r+j] = (float)dot[j] * (m->scale[r+j] * xs);
    }
}

static inline size_t k3_q8_matrix_bytes(int rows, int cols) {
    return (size_t)rows * cols + (size_t)rows * sizeof(float);
}

/* A64FX loses bandwidth at 48 workers for these two matrices.  Reserving the
 * highest cpuset core for uTofu progress also makes 47 the runner default. */
#define K3_DENSE_THREADS 47

static inline void k3_dense_pair_bf16(float *router_out, float *latent_out,
                                      const k3_bf16_matrix *router,
                                      const k3_bf16_matrix *latent_down,
                                      const float *hidden, int threads) {
    int router_groups = router->rows / 8;
    int down_groups = latent_down->rows / 8;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for (int task = 0; task < router_groups + down_groups; ++task) {
        const k3_bf16_matrix *m = task < router_groups ? router : latent_down;
        float *out = task < router_groups ? router_out : latent_out;
        int group = task < router_groups ? task : task - router_groups;
        int row = group * 8;
        const uint16_t *w = m->weight + (size_t)row * m->cols;
        matvec_bf16_8row(out + row, w, w + m->cols, w + 2 * m->cols,
                         w + 3 * m->cols, w + 4 * m->cols, w + 5 * m->cols,
                         w + 6 * m->cols, w + 7 * m->cols, hidden, m->cols);
    }
}

/* Fuse independent small-output projections (KDA q/k/v/g/f_a) into one team.
 * Four-row tasks give enough work items to balance all CMGs when each matrix
 * has only 128 rows. */
static inline void k3_dense_many_bf16_row4(float **out,
        const k3_bf16_matrix *matrix, int count, const float *x, int threads) {
    int groups=matrix[0].rows/4;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for(int task=0;task<count*groups;++task){int m=task/groups,r=(task%groups)*4;
        const uint16_t*w=matrix[m].weight+(size_t)r*matrix[m].cols;
        matvec_bf16_4row(out[m]+r,w,w+matrix[m].cols,w+2*matrix[m].cols,
            w+3*matrix[m].cols,x,matrix[m].cols);}
}

/* Keep the quality-sensitive router in BF16 while streaming the larger routed
 * down projection in Q8.  Both task classes share one persistent OpenMP team. */
static inline void k3_dense_router_bf16_down_q8(
        float *router_out, float *latent_out,
        const k3_bf16_matrix *router, const k3_q8_matrix *latent_down,
        const float *hidden, int8_t *qx, int threads) {
    float xs = k3_q8_quantize_vector(qx, hidden, latent_down->cols);
    int router_groups = router->rows / 8;
    int down_groups = latent_down->rows / 8;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for (int task = 0; task < router_groups + down_groups; ++task) {
        if (task < router_groups) {
            int row = task * 8;
            const uint16_t *w = router->weight + (size_t)row * router->cols;
            matvec_bf16_8row(router_out + row, w, w + router->cols,
                w + 2 * router->cols, w + 3 * router->cols,
                w + 4 * router->cols, w + 5 * router->cols,
                w + 6 * router->cols, w + 7 * router->cols,
                hidden, router->cols);
        } else {
            int r = (task - router_groups) * 8; int32_t dot[8];
            k3_q8_dot8(dot, latent_down->weight + (size_t)r * latent_down->cols,
                       qx, latent_down->cols);
            for (int j = 0; j < 8; ++j)
                latent_out[r+j] = (float)dot[j] *
                    (latent_down->scale[r+j] * xs);
        }
    }
}

#endif
