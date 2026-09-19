/* Benchmark the compact decode kernels on real Qwen3.8 tensor layouts/data.
 * The requested tensors are copied out of the lazy GGUF mapping so timed runs
 * measure HBM traffic rather than filesystem page faults. */
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

typedef struct {
    uint16_t d, dmin;
    uint8_t scales[12];
    int8_t q[256];
} packed_q5_block;

_Static_assert(sizeof(packed_q5_block) == 272, "packed Q5 block size");

static void pack_q5(packed_q5_block *dst, const block_q5_K *src, size_t blocks) {
#pragma omp parallel for schedule(static)
    for (size_t b = 0; b < blocks; b++) {
        dst[b].d = src[b].d; dst[b].dmin = src[b].dmin;
        memcpy(dst[b].scales, src[b].scales, 12);
        for (int g = 0; g < 4; g++) for (int k = 0; k < 32; k++) {
            uint8_t v = src[b].qs[g * 32 + k];
            dst[b].q[g * 64 + k] = (int8_t)((v & 15) |
                (((src[b].qh[k] >> (2 * g)) & 1) << 4));
            dst[b].q[g * 64 + 32 + k] = (int8_t)((v >> 4) |
                (((src[b].qh[k] >> (2 * g + 1)) & 1) << 4));
        }
    }
}

static inline float packed_q5_dot(const packed_q5_block *w,
                                  const tf_kquant_a8_block *x, int nb) {
    const svbool_t p8 = svptrue_b8(), pg = svptrue_b32();
    const svbool_t first8 = svwhilelt_b32(0, 8);
    svfloat32_t acc = svdup_f32(0); float corr = 0;
    for (int b = 0; b < nb; b++) {
        svint32_t ia = svdup_s32(0);
        for (int g = 0; g < 4; g++) {
            uint8_t s0, m0, s1, m1;
            get_scale_min_k4(2*g, w[b].scales, &s0, &m0);
            get_scale_min_k4(2*g+1, w[b].scales, &s1, &m1);
            svint32_t dot = svdot_s32(svdup_s32(0),
                svld1_s8(p8, w[b].q + g*64), svld1_s8(p8, x[b].q + g*64));
            ia = svmla_s32_x(pg, ia, dot,
                svsel_s32(first8, svdup_s32(s0), svdup_s32(s1)));
            corr -= ggml_fp16_to_fp32(w[b].dmin) * x[b].d[0] *
                ((float)m0*x[b].sum[2*g] + (float)m1*x[b].sum[2*g+1]);
        }
        acc = svmla_n_f32_x(pg, acc, svcvt_f32_s32_x(pg, ia),
                            ggml_fp16_to_fp32(w[b].d)*x[b].d[0]);
    }
    return svaddv_f32(pg, acc) + corr;
}

static inline void packed_q5_dot8(float out[8], const packed_q5_block *const w[8],
                                  const tf_kquant_a8_block *x, int nb) {
    const svbool_t p8=svptrue_b8(),pg=svptrue_b32(),first8=svwhilelt_b32(0,8);
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    float c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
    for(int b=0;b<nb;b++){
        svint32_t i0=svdup_s32(0),i1=i0,i2=i0,i3=i0,i4=i0,i5=i0,i6=i0,i7=i0;
        for(int g=0;g<4;g++){
            svint8_t xv=svld1_s8(p8,x[b].q+g*64);
#define PACKED_Q5_ROW(R,IA,C) do { const packed_q5_block *wb=&w[R][b]; \
    uint8_t s0,m0,s1,m1; get_scale_min_k4(2*g,wb->scales,&s0,&m0); \
    get_scale_min_k4(2*g+1,wb->scales,&s1,&m1); \
    svint32_t dot=svdot_s32(svdup_s32(0),svld1_s8(p8,wb->q+g*64),xv); \
    IA=svmla_s32_x(pg,IA,dot,svsel_s32(first8,svdup_s32(s0),svdup_s32(s1))); \
    C-=ggml_fp16_to_fp32(wb->dmin)*x[b].d[0]*((float)m0*x[b].sum[2*g]+(float)m1*x[b].sum[2*g+1]); \
} while(0)
            PACKED_Q5_ROW(0,i0,c0);PACKED_Q5_ROW(1,i1,c1);
            PACKED_Q5_ROW(2,i2,c2);PACKED_Q5_ROW(3,i3,c3);
            PACKED_Q5_ROW(4,i4,c4);PACKED_Q5_ROW(5,i5,c5);
            PACKED_Q5_ROW(6,i6,c6);PACKED_Q5_ROW(7,i7,c7);
#undef PACKED_Q5_ROW
        }
#define PACKED_Q5_SCALE(R,A,I) A=svmla_n_f32_x(pg,A,svcvt_f32_s32_x(pg,I),ggml_fp16_to_fp32(w[R][b].d)*x[b].d[0])
        PACKED_Q5_SCALE(0,a0,i0);PACKED_Q5_SCALE(1,a1,i1);
        PACKED_Q5_SCALE(2,a2,i2);PACKED_Q5_SCALE(3,a3,i3);
        PACKED_Q5_SCALE(4,a4,i4);PACKED_Q5_SCALE(5,a5,i5);
        PACKED_Q5_SCALE(6,a6,i6);PACKED_Q5_SCALE(7,a7,i7);
#undef PACKED_Q5_SCALE
    }
    out[0]=svaddv_f32(pg,a0)+c0;out[1]=svaddv_f32(pg,a1)+c1;
    out[2]=svaddv_f32(pg,a2)+c2;out[3]=svaddv_f32(pg,a3)+c3;
    out[4]=svaddv_f32(pg,a4)+c4;out[5]=svaddv_f32(pg,a5)+c5;
    out[6]=svaddv_f32(pg,a6)+c6;out[7]=svaddv_f32(pg,a7)+c7;
}

static void run_packed_q5(float *y, const packed_q5_block *weights,
                          const float *x, int rows, int cols) {
    int nb = cols/256;
#pragma omp parallel
    {
        int tid=omp_get_thread_num(), nt=omp_get_num_threads();
        int r0=rows*tid/nt, r1=rows*(tid+1)/nt;
        tf_kquant_a8_block *qx=alloca((size_t)nb*sizeof(*qx));
        tf_kquant_quant_a8(qx,x,cols);
        int r=r0;
        for(;r+7<r1;r+=8){
            const packed_q5_block *wr[8];
            for(int j=0;j<8;j++)wr[j]=weights+(size_t)(r+j)*nb;
            packed_q5_dot8(y+r,wr,qx,nb);
        }
        for(;r<r1;r++) y[r]=packed_q5_dot(weights+(size_t)r*nb,qx,nb);
    }
}

typedef struct {
    float d, dmin;
    uint8_t scales[8];
    uint8_t mins[8];
} packed_q5r_header;

_Static_assert(sizeof(packed_q5r_header) == 24, "packed Q5R header size");

static size_t packed_q5r_block_bytes(void) {
    return 8 * sizeof(packed_q5r_header) + 8 * 256;
}

static void pack_q5r(uint8_t *dst, const block_q5_K *src, int rows, int cols) {
    int nb = cols / 256;
    size_t bb = packed_q5r_block_bytes();
#pragma omp parallel for schedule(static)
    for (int rg = 0; rg < rows / 8; rg++) {
        for (int b = 0; b < nb; b++) {
            uint8_t *block = dst + ((size_t)rg * nb + b) * bb;
            packed_q5r_header *headers = (packed_q5r_header *)block;
            int8_t *q = (int8_t *)(block + 8 * sizeof(*headers));
            for (int rr = 0; rr < 8; rr++) {
                const block_q5_K *wb = src + (size_t)(rg * 8 + rr) * nb + b;
                headers[rr].d = ggml_fp16_to_fp32(wb->d);
                headers[rr].dmin = ggml_fp16_to_fp32(wb->dmin);
                for (int i = 0; i < 8; i++)
                    get_scale_min_k4(i, wb->scales,
                                     &headers[rr].scales[i], &headers[rr].mins[i]);
                for (int g = 0; g < 4; g++) {
                    int8_t *qrow = q + (g * 8 + rr) * 64;
                    for (int k = 0; k < 32; k++) {
                        uint8_t v = wb->qs[g * 32 + k];
                        qrow[k] = (int8_t)((v & 15) |
                            (((wb->qh[k] >> (2 * g)) & 1) << 4));
                        qrow[32 + k] = (int8_t)((v >> 4) |
                            (((wb->qh[k] >> (2 * g + 1)) & 1) << 4));
                    }
                }
            }
        }
    }
}

static inline void packed_q5r_dot8(float out[8], const uint8_t *weights,
                                   const tf_kquant_a8_block *x, int nb) {
    const svbool_t p8 = svptrue_b8(), pg = svptrue_b32();
    const svbool_t first8 = svwhilelt_b32(0, 8);
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    svfloat32_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    float c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    size_t bb = packed_q5r_block_bytes();
    for (int b = 0; b < nb; b++) {
        const uint8_t *block = weights + (size_t)b * bb;
        const packed_q5r_header *headers = (const packed_q5r_header *)block;
        const int8_t *q = (const int8_t *)(block + 8 * sizeof(*headers));
        svint32_t i0 = svdup_s32(0), i1 = i0, i2 = i0, i3 = i0;
        svint32_t i4 = i0, i5 = i0, i6 = i0, i7 = i0;
        for (int g = 0; g < 4; g++) {
            svint8_t xv = svld1_s8(p8, x[b].q + g * 64);
#define PACKED_Q5R_ROW(R, IA, C) do { \
    const packed_q5r_header *h = &headers[R]; \
    uint8_t s0 = h->scales[2 * g], s1 = h->scales[2 * g + 1]; \
    uint8_t m0 = h->mins[2 * g], m1 = h->mins[2 * g + 1]; \
    svint32_t dot = svdot_s32(svdup_s32(0), \
        svld1_s8(p8, q + (g * 8 + (R)) * 64), xv); \
    IA = svmla_s32_x(pg, IA, dot, \
        svsel_s32(first8, svdup_s32(s0), svdup_s32(s1))); \
    C -= h->dmin * x[b].d[0] * \
        ((float)m0 * x[b].sum[2 * g] + (float)m1 * x[b].sum[2 * g + 1]); \
} while (0)
            PACKED_Q5R_ROW(0, i0, c0); PACKED_Q5R_ROW(1, i1, c1);
            PACKED_Q5R_ROW(2, i2, c2); PACKED_Q5R_ROW(3, i3, c3);
            PACKED_Q5R_ROW(4, i4, c4); PACKED_Q5R_ROW(5, i5, c5);
            PACKED_Q5R_ROW(6, i6, c6); PACKED_Q5R_ROW(7, i7, c7);
#undef PACKED_Q5R_ROW
        }
#define PACKED_Q5R_SCALE(R, A, I) \
    A = svmla_n_f32_x(pg, A, svcvt_f32_s32_x(pg, I), \
        headers[R].d * x[b].d[0])
        PACKED_Q5R_SCALE(0, a0, i0); PACKED_Q5R_SCALE(1, a1, i1);
        PACKED_Q5R_SCALE(2, a2, i2); PACKED_Q5R_SCALE(3, a3, i3);
        PACKED_Q5R_SCALE(4, a4, i4); PACKED_Q5R_SCALE(5, a5, i5);
        PACKED_Q5R_SCALE(6, a6, i6); PACKED_Q5R_SCALE(7, a7, i7);
#undef PACKED_Q5R_SCALE
    }
    out[0] = svaddv_f32(pg, a0) + c0; out[1] = svaddv_f32(pg, a1) + c1;
    out[2] = svaddv_f32(pg, a2) + c2; out[3] = svaddv_f32(pg, a3) + c3;
    out[4] = svaddv_f32(pg, a4) + c4; out[5] = svaddv_f32(pg, a5) + c5;
    out[6] = svaddv_f32(pg, a6) + c6; out[7] = svaddv_f32(pg, a7) + c7;
}

static void run_packed_q5r(float *y, const uint8_t *weights,
                           const float *x, int rows, int cols) {
    int nb = cols / 256;
    size_t row_group_bytes = (size_t)nb * packed_q5r_block_bytes();
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        int g0 = (rows / 8) * tid / nt;
        int g1 = (rows / 8) * (tid + 1) / nt;
        tf_kquant_a8_block *qx = alloca((size_t)nb * sizeof(*qx));
        tf_kquant_quant_a8(qx, x, cols);
        for (int g = g0; g < g1; g++)
            packed_q5r_dot8(y + g * 8, weights + (size_t)g * row_group_bytes,
                            qx, nb);
    }
}

static size_t q8r_group_bytes(int cols) { return 32 + (size_t)8*cols; }

static void pack_q5_q8r(uint8_t *dst, const block_q5_K *src, int rows, int cols) {
    int nb=cols/256; size_t gb=q8r_group_bytes(cols);
#pragma omp parallel
    {
        float *tmp=aligned_alloc(64,(size_t)cols*sizeof(float));
#pragma omp for schedule(static)
        for(int r=0;r<rows;r++){
            dequantize_row_q5_K(src+(size_t)r*nb,tmp,cols);
            float mx=0;for(int k=0;k<cols;k++){float a=fabsf(tmp[k]);if(a>mx)mx=a;}
            float d=mx>0?mx/127.f:0,inv=mx>0?127.f/mx:0;
            uint8_t *group=dst+(size_t)(r/8)*gb;
            ((float*)group)[r&7]=d;
            int rr=r&7;
            for(int k=0;k<cols;k++){int v=(int)lrintf(tmp[k]*inv);
                ((int8_t*)(group+32))[(size_t)(k/64)*8*64+(size_t)rr*64+(k&63)]=
                    (int8_t)(v<-127?-127:v>127?127:v);}
        }
        free(tmp);
    }
}

static void quant_x64(const float *x,int cols,int8_t *q,float *d){
    for(int b=0;b<cols/64;b++){float mx=0;for(int k=0;k<64;k++){float a=fabsf(x[b*64+k]);if(a>mx)mx=a;}
        d[b]=mx>0?mx/127.f:0;float inv=mx>0?127.f/mx:0;
        for(int k=0;k<64;k++){int v=(int)lrintf(x[b*64+k]*inv);q[b*64+k]=(int8_t)(v<-127?-127:v>127?127:v);}}
}

static inline void q8r_dot8(float *out,const uint8_t *group,const int8_t *xq,const float *xd,int cols){
    const float *ws=(const float*)group;const int8_t *q=(const int8_t*)(group+32);
    svbool_t p8=svptrue_b8(),pg=svptrue_b32();
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    for(int b=0;b<cols/64;b++){int k=b*64;svint8_t xv=svld1_s8(p8,xq+k);svfloat32_t ds=svdup_f32(xd[b]);
#define Q8R_ROW(R,A) A=svmla_f32_x(pg,A,svcvt_f32_s32_x(pg,svdot_s32(svdup_s32(0),svld1_s8(p8,q+(size_t)b*8*64+(R)*64),xv)),ds)
        Q8R_ROW(0,a0);Q8R_ROW(1,a1);Q8R_ROW(2,a2);Q8R_ROW(3,a3);
        Q8R_ROW(4,a4);Q8R_ROW(5,a5);Q8R_ROW(6,a6);Q8R_ROW(7,a7);
#undef Q8R_ROW
    }
    out[0]=svaddv_f32(pg,a0)*ws[0];out[1]=svaddv_f32(pg,a1)*ws[1];
    out[2]=svaddv_f32(pg,a2)*ws[2];out[3]=svaddv_f32(pg,a3)*ws[3];
    out[4]=svaddv_f32(pg,a4)*ws[4];out[5]=svaddv_f32(pg,a5)*ws[5];
    out[6]=svaddv_f32(pg,a6)*ws[6];out[7]=svaddv_f32(pg,a7)*ws[7];
}

static void run_q8r(float *y,const uint8_t *weights,const float *x,int rows,int cols){
    int8_t *xq=aligned_alloc(64,(size_t)cols);float *xd=aligned_alloc(64,(size_t)(cols/64)*sizeof(float));
    quant_x64(x,cols,xq,xd);size_t gb=q8r_group_bytes(cols);
#pragma omp parallel for schedule(static)
    for(int g=0;g<rows/8;g++)q8r_dot8(y+g*8,weights+(size_t)g*gb,xq,xd,cols);
    free(xd);free(xq);
}

static size_t q8r64_group_bytes(int cols) {
    return (size_t)(cols / 64) * (32 + 8 * 64);
}

static void pack_q5_q8r64(uint8_t *dst, const block_q5_K *src,
                          int rows, int cols) {
    int nb = cols / 256;
    size_t gb = q8r64_group_bytes(cols);
#pragma omp parallel
    {
        float *tmp = aligned_alloc(64, (size_t)cols * sizeof(float));
#pragma omp for schedule(static)
        for (int r = 0; r < rows; r++) {
            dequantize_row_q5_K(src + (size_t)r * nb, tmp, cols);
            uint8_t *group = dst + (size_t)(r / 8) * gb;
            int rr = r & 7;
            for (int b = 0; b < cols / 64; b++) {
                const float *xb = tmp + b * 64;
                float mx = 0.0f;
                for (int k = 0; k < 64; k++) {
                    float a = fabsf(xb[k]);
                    if (a > mx) mx = a;
                }
                float d = mx > 0.0f ? mx / 127.0f : 0.0f;
                float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
                uint8_t *block = group + (size_t)b * (32 + 8 * 64);
                ((float *)block)[rr] = d;
                int8_t *q = (int8_t *)(block + 32) + rr * 64;
                for (int k = 0; k < 64; k++) {
                    int v = (int)lrintf(xb[k] * inv);
                    q[k] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
                }
            }
        }
        free(tmp);
    }
}

static inline void q8r64_dot8(float *out, const uint8_t *group,
                              const int8_t *xq, const float *xd, int cols) {
    svbool_t p8 = svptrue_b8(), pg = svptrue_b32();
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    svfloat32_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (int b = 0; b < cols / 64; b++) {
        const uint8_t *block = group + (size_t)b * (32 + 8 * 64);
        const float *ws = (const float *)block;
        const int8_t *q = (const int8_t *)(block + 32);
        svint8_t xv = svld1_s8(p8, xq + b * 64);
#define Q8R64_ROW(R, A) do { \
    svint32_t dot = svdot_s32(svdup_s32(0), svld1_s8(p8, q + (R) * 64), xv); \
    A = svmla_n_f32_x(pg, A, svcvt_f32_s32_x(pg, dot), xd[b] * ws[R]); \
} while (0)
        Q8R64_ROW(0, a0); Q8R64_ROW(1, a1);
        Q8R64_ROW(2, a2); Q8R64_ROW(3, a3);
        Q8R64_ROW(4, a4); Q8R64_ROW(5, a5);
        Q8R64_ROW(6, a6); Q8R64_ROW(7, a7);
#undef Q8R64_ROW
    }
    out[0] = svaddv_f32(pg, a0); out[1] = svaddv_f32(pg, a1);
    out[2] = svaddv_f32(pg, a2); out[3] = svaddv_f32(pg, a3);
    out[4] = svaddv_f32(pg, a4); out[5] = svaddv_f32(pg, a5);
    out[6] = svaddv_f32(pg, a6); out[7] = svaddv_f32(pg, a7);
}

static void run_q8r64(float *y, const uint8_t *weights, const float *x,
                      int rows, int cols) {
    int8_t *xq = aligned_alloc(64, (size_t)cols);
    float *xd = aligned_alloc(64, (size_t)(cols / 64) * sizeof(float));
    quant_x64(x, cols, xq, xd);
    size_t gb = q8r64_group_bytes(cols);
#pragma omp parallel for schedule(static)
    for (int g = 0; g < rows / 8; g++)
        q8r64_dot8(y + g * 8, weights + (size_t)g * gb, xq, xd, cols);
    free(xd);
    free(xq);
}

static float quant_x_global(const float*x,int cols,int8_t*q){
    float mx=0;for(int k=0;k<cols;k++){float a=fabsf(x[k]);if(a>mx)mx=a;}
    float d=mx>0?mx/127.f:0,inv=mx>0?127.f/mx:0;
    for(int k=0;k<cols;k++){int v=(int)lrintf(x[k]*inv);q[k]=(int8_t)(v<-127?-127:v>127?127:v);}return d;
}

static inline void q8r_i32_dot8(float*out,const uint8_t*group,const int8_t*xq,float xd,int cols){
    const float*ws=(const float*)group;const int8_t*q=(const int8_t*)(group+32);
    svbool_t p8=svptrue_b8(),pg=svptrue_b32();
    svint32_t a0=svdup_s32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    for(int k=0;k<cols;k+=64){svint8_t xv=svld1_s8(p8,xq+k);
        const int8_t *qb=q+(size_t)(k/64)*8*64;
        a0=svdot_s32(a0,svld1_s8(p8,qb+0*64),xv);
        a1=svdot_s32(a1,svld1_s8(p8,qb+1*64),xv);
        a2=svdot_s32(a2,svld1_s8(p8,qb+2*64),xv);
        a3=svdot_s32(a3,svld1_s8(p8,qb+3*64),xv);
        a4=svdot_s32(a4,svld1_s8(p8,qb+4*64),xv);
        a5=svdot_s32(a5,svld1_s8(p8,qb+5*64),xv);
        a6=svdot_s32(a6,svld1_s8(p8,qb+6*64),xv);
        a7=svdot_s32(a7,svld1_s8(p8,qb+7*64),xv);
    }
    out[0]=(float)svaddv_s32(pg,a0)*ws[0]*xd;out[1]=(float)svaddv_s32(pg,a1)*ws[1]*xd;
    out[2]=(float)svaddv_s32(pg,a2)*ws[2]*xd;out[3]=(float)svaddv_s32(pg,a3)*ws[3]*xd;
    out[4]=(float)svaddv_s32(pg,a4)*ws[4]*xd;out[5]=(float)svaddv_s32(pg,a5)*ws[5]*xd;
    out[6]=(float)svaddv_s32(pg,a6)*ws[6]*xd;out[7]=(float)svaddv_s32(pg,a7)*ws[7]*xd;
}

static void run_q8r_i32(float*y,const uint8_t*weights,const float*x,int rows,int cols){
    int8_t*xq=aligned_alloc(64,(size_t)cols);float xd=quant_x_global(x,cols,xq);size_t gb=q8r_group_bytes(cols);
#pragma omp parallel for schedule(static)
    for(int g=0;g<rows/8;g++)q8r_i32_dot8(y+g*8,weights+(size_t)g*gb,xq,xd,cols);
    free(xq);
}

static int find_tensor(const gguf_context *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(gguf_tensor_name(g, (int)i), name)) return (int)i;
    return -1;
}

static void print_model_summary(const gguf_context *g) {
    size_t total_bytes = 0, q5_bytes = 0, q5r_bytes = 0;
    uint64_t q5_count = 0, q5r_count = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor_info *info = &g->tensors[i];
        size_t bytes = gguf_tensor_size(g, (int)i);
        total_bytes += bytes;
        if (info->type != GGML_TYPE_Q5_K) continue;
        q5_count++;
        q5_bytes += bytes;
        if (info->n_dims == 2 && info->dims[0] % 256 == 0 &&
            info->dims[1] % 8 == 0) {
            size_t cols = (size_t)info->dims[0];
            size_t rows = (size_t)info->dims[1];
            q5r_count++;
            q5r_bytes += (rows / 8) * (cols / 256) * packed_q5r_block_bytes();
        }
    }
    size_t projected = total_bytes - q5_bytes + q5r_bytes;
    printf("model tensors=%llu tensor_bytes=%.3fGB Q5_K=%llu/%.3fGB "
           "Q5R_eligible=%llu/%.3fGB Q5R_delta=%.3fGB projected=%.3fGB\n",
           (unsigned long long)g->n_tensors, total_bytes / 1e9,
           (unsigned long long)q5_count, q5_bytes / 1e9,
           (unsigned long long)q5r_count, q5r_bytes / 1e9,
           ((double)q5r_bytes - (double)q5_bytes) / 1e9, projected / 1e9);
}

static void run_rows(float *y, const void *weights, uint32_t type,
                     const float *x, int rows, int cols, int use_a8) {
    const int nb = cols / 256;
    const size_t rb = tf_row_bytes(type, cols);
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        int r0 = rows * tid / nt, r1 = rows * (tid + 1) / nt;
        tf_kquant_a8_block *qx = NULL;
        if (use_a8) {
            qx = alloca((size_t)nb * sizeof(*qx));
            tf_kquant_quant_a8(qx, x, cols);
        }
        int r = r0;
        if (use_a8 && type == GGML_TYPE_Q5_K) {
            for (; r + 3 < r1; r += 4) {
                const uint8_t *w = (const uint8_t *)weights + (size_t)r * rb;
                tf_q5_k_a8_dot4_sve(y + r, (const block_q5_K *)w,
                    (const block_q5_K *)(w + rb), (const block_q5_K *)(w + 2 * rb),
                    (const block_q5_K *)(w + 3 * rb), qx, nb);
            }
        }
        for (; r < r1; r++) {
            const uint8_t *w = (const uint8_t *)weights + (size_t)r * rb;
            if (type == GGML_TYPE_Q5_K)
                y[r] = use_a8 ? tf_q5_k_a8_dot_sve((const block_q5_K *)w, qx, nb) :
                                tf_q5_k_dot_sve((const block_q5_K *)w, x, cols);
            else
                y[r] = use_a8 ? tf_iq4_xs_a8_dot_sve((const block_iq4_xs *)w, qx, nb) :
                                tf_iq4_xs_dot_sve((const block_iq4_xs *)w, x, cols);
        }
    }
}

static int bench_tensor(const gguf_context *g, const char *name, int reps) {
    int ti = find_tensor(g, name);
    if (ti < 0) { fprintf(stderr, "missing tensor: %s\n", name); return -1; }
    const gguf_tensor_info *info = &g->tensors[ti];
    if (info->n_dims != 2 || (info->type != GGML_TYPE_Q5_K &&
                             info->type != GGML_TYPE_IQ4_XS)) {
        fprintf(stderr, "unsupported tensor: %s type=%s dims=%u\n", name,
                ggml_type_name(info->type), info->n_dims);
        return -1;
    }
    int cols = (int)info->dims[0], rows = (int)info->dims[1];
    size_t bytes = gguf_tensor_size(g, ti);
    size_t alloc_bytes = (bytes + 255) & ~(size_t)255;
    void *weights = aligned_alloc(256, alloc_bytes);
    float *x = aligned_alloc(256, ((size_t)cols * sizeof(*x) + 255) & ~(size_t)255);
    float *ref = aligned_alloc(256, ((size_t)rows * sizeof(*ref) + 255) & ~(size_t)255);
    float *a8 = aligned_alloc(256, ((size_t)rows * sizeof(*a8) + 255) & ~(size_t)255);
    if (!weights || !x || !ref || !a8) { fprintf(stderr, "allocation failed\n"); return -1; }
    memcpy(weights, gguf_tensor_data(g, ti), bytes);
    for (int i = 0; i < cols; i++)
        x[i] = 0.75f * sinf((float)i * 0.0137f) + 0.2f * cosf((float)i * 0.071f);

    run_rows(ref, weights, info->type, x, rows, cols, 0);
    run_rows(a8, weights, info->type, x, rows, cols, 1);
    double err2 = 0.0, ref2 = 0.0, max_rel = 0.0;
    for (int r = 0; r < rows; r++) {
        double e = (double)a8[r] - ref[r];
        double rel = fabs(e) / (fabs((double)ref[r]) + 1e-9);
        err2 += e * e; ref2 += (double)ref[r] * ref[r];
        if (rel > max_rel) max_rel = rel;
    }

    double best_ref = 1e9, best_a8 = 1e9;
    volatile float checksum = 0.0f;
    for (int mode = 0; mode < 2; mode++) for (int rep = 0; rep < reps; rep++) {
        double t0 = seconds();
        run_rows(mode ? a8 : ref, weights, info->type, x, rows, cols, mode);
        double dt = seconds() - t0;
        if (mode ? dt < best_a8 : dt < best_ref) {
            if (mode) best_a8 = dt; else best_ref = dt;
        }
        checksum += mode ? a8[rep % rows] : ref[rep % rows];
    }
    printf("%s type=%s shape=%dx%d bytes=%.3fMB threads=%d "
           "fp32=%.3fms/%.1fGB/s a8=%.3fms/%.1fGB/s speedup=%.3fx "
           "nrmse=%.3g max_rel=%.3g checksum=%g\n",
           name, ggml_type_name(info->type), rows, cols, bytes / 1e6,
           omp_get_max_threads(), best_ref * 1e3, bytes / best_ref / 1e9,
           best_a8 * 1e3, bytes / best_a8 / 1e9, best_ref / best_a8,
           sqrt(err2 / (ref2 + 1e-30)), max_rel, checksum);
    if (info->type == GGML_TYPE_Q5_K) {
        size_t output_bytes = (size_t)rows * sizeof(*a8);
        float *native_a8 = aligned_alloc(256, (output_bytes + 255) & ~(size_t)255);
        if (!native_a8) { fprintf(stderr, "native A8 copy allocation failed\n"); return -1; }
        memcpy(native_a8, a8, output_bytes);
        size_t blocks=(size_t)rows*(cols/256), pbytes=blocks*sizeof(packed_q5_block);
        double pe=0,pn=0,best=1e9;
        if (!getenv("QWEN_KQUANT_SKIP_PACKED")) {
            packed_q5_block *packed=aligned_alloc(256,(pbytes+255)&~(size_t)255);
            if (!packed) { fprintf(stderr,"packed allocation failed\n"); return -1; }
            pack_q5(packed,(const block_q5_K*)weights,blocks);
            run_packed_q5(a8,packed,x,rows,cols);
            for(int r=0;r<rows;r++){double e=a8[r]-ref[r];pe+=e*e;pn+=(double)ref[r]*ref[r];}
            for(int rep=0;rep<reps;rep++){double t0=seconds();run_packed_q5(a8,packed,x,rows,cols);double dt=seconds()-t0;if(dt<best)best=dt;}
            printf("  packed_q5 storage=%.3fMB expansion=%.3fx time=%.3fms physical=%.1fGB/s "
                   "effective=%.1fGB/s nrmse=%.3g\n",pbytes/1e6,(double)pbytes/bytes,
                   best*1e3,pbytes/best/1e9,bytes/best/1e9,sqrt(pe/(pn+1e-30)));
            free(packed);
        }
        size_t q5rbytes = (size_t)(rows / 8) * (cols / 256) *
                          packed_q5r_block_bytes();
        uint8_t *q5r = aligned_alloc(256, (q5rbytes + 255) & ~(size_t)255);
        if (!q5r) { fprintf(stderr, "packed Q5R allocation failed\n"); return -1; }
        pack_q5r(q5r, (const block_q5_K *)weights, rows, cols);
        run_packed_q5r(a8, q5r, x, rows, cols);
        pe = pn = 0;
        double a8_err2 = 0.0, a8_norm2 = 0.0, a8_max_abs = 0.0;
        for (int r = 0; r < rows; r++) {
            double e = a8[r] - ref[r];
            pe += e * e;
            pn += (double)ref[r] * ref[r];
            double ae = (double)a8[r] - native_a8[r];
            double aa = fabs(ae);
            a8_err2 += ae * ae;
            a8_norm2 += (double)native_a8[r] * native_a8[r];
            if (aa > a8_max_abs) a8_max_abs = aa;
        }
        best = 1e9;
        for (int rep = 0; rep < reps; rep++) {
            double t0 = seconds();
            run_packed_q5r(a8, q5r, x, rows, cols);
            double dt = seconds() - t0;
            if (dt < best) best = dt;
        }
        printf("  packed_q5r storage=%.3fMB expansion=%.3fx time=%.3fms physical=%.1fGB/s "
               "effective=%.1fGB/s nrmse=%.3g a8_nrmse=%.3g a8_max_abs=%.3g\n",
               q5rbytes / 1e6,
               (double)q5rbytes / bytes, best * 1e3, q5rbytes / best / 1e9,
               bytes / best / 1e9, sqrt(pe / (pn + 1e-30)),
               sqrt(a8_err2 / (a8_norm2 + 1e-30)), a8_max_abs);
        fflush(stdout);
        free(q5r);
        size_t q8bytes=(size_t)(rows/8)*q8r_group_bytes(cols);
        uint8_t *q8r=aligned_alloc(256,(q8bytes+255)&~(size_t)255);
        if(!q8r){fprintf(stderr,"q8r allocation failed\n");return -1;}
        pack_q5_q8r(q8r,(const block_q5_K*)weights,rows,cols);
        run_q8r(a8,q8r,x,rows,cols);
        pe=pn=0;for(int r=0;r<rows;r++){double e=a8[r]-ref[r];pe+=e*e;pn+=(double)ref[r]*ref[r];}
        best=1e9;for(int rep=0;rep<reps;rep++){double t0=seconds();run_q8r(a8,q8r,x,rows,cols);double dt=seconds()-t0;if(dt<best)best=dt;}
        printf("  q8r_q5 storage=%.3fMB expansion=%.3fx time=%.3fms physical=%.1fGB/s "
               "effective=%.1fGB/s nrmse=%.3g\n",q8bytes/1e6,(double)q8bytes/bytes,
               best*1e3,q8bytes/best/1e9,bytes/best/1e9,sqrt(pe/(pn+1e-30)));
        run_q8r_i32(a8,q8r,x,rows,cols);pe=pn=0;
        for(int r=0;r<rows;r++){double e=a8[r]-ref[r];pe+=e*e;pn+=(double)ref[r]*ref[r];}
        best=1e9;for(int rep=0;rep<reps;rep++){double t0=seconds();run_q8r_i32(a8,q8r,x,rows,cols);double dt=seconds()-t0;if(dt<best)best=dt;}
        printf("  q8r_i32 storage=%.3fMB expansion=%.3fx time=%.3fms physical=%.1fGB/s "
               "effective=%.1fGB/s nrmse=%.3g\n",q8bytes/1e6,(double)q8bytes/bytes,
               best*1e3,q8bytes/best/1e9,bytes/best/1e9,sqrt(pe/(pn+1e-30)));
        fflush(stdout);
        free(q8r);
        size_t q8r64bytes = (size_t)(rows / 8) * q8r64_group_bytes(cols);
        uint8_t *q8r64 = aligned_alloc(256, (q8r64bytes + 255) & ~(size_t)255);
        if (!q8r64) { fprintf(stderr, "q8r64 allocation failed\n"); return -1; }
        pack_q5_q8r64(q8r64, (const block_q5_K *)weights, rows, cols);
        run_q8r64(a8, q8r64, x, rows, cols);
        pe = pn = 0;
        for (int r = 0; r < rows; r++) {
            double e = a8[r] - ref[r];
            pe += e * e;
            pn += (double)ref[r] * ref[r];
        }
        best = 1e9;
        for (int rep = 0; rep < reps; rep++) {
            double t0 = seconds();
            run_q8r64(a8, q8r64, x, rows, cols);
            double dt = seconds() - t0;
            if (dt < best) best = dt;
        }
        printf("  q8r64_q5 storage=%.3fMB expansion=%.3fx time=%.3fms physical=%.1fGB/s "
               "effective=%.1fGB/s nrmse=%.3g\n", q8r64bytes / 1e6,
               (double)q8r64bytes / bytes, best * 1e3, q8r64bytes / best / 1e9,
               bytes / best / 1e9, sqrt(pe / (pn + 1e-30)));
        fflush(stdout);
        free(q8r64);
        free(native_a8);
    }
    free(a8); free(ref); free(x); free(weights);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL [REPS] [TENSOR]\n"
                        "       %s MODEL --summary\n", argv[0], argv[0]);
        return 2;
    }
    setenv("GGUF_LAZY_MMAP", "1", 1);
    gguf_context *g = gguf_open(argv[1], 1);
    if (!g) return 1;
    if (argc > 2 && !strcmp(argv[2], "--summary")) {
        print_model_summary(g);
        gguf_close(g);
        return 0;
    }
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    if (reps < 1) reps = 1;
    int rc = 0;
    if (argc > 3) {
        rc = bench_tensor(g, argv[3], reps);
        gguf_close(g);
        return rc != 0;
    }
    rc |= bench_tensor(g, "blk.0.ffn_gate.weight", reps);
    rc |= bench_tensor(g, "blk.0.ffn_up.weight", reps);
    rc |= bench_tensor(g, "blk.0.ffn_down.weight", reps);
    gguf_close(g);
    return rc != 0;
}
