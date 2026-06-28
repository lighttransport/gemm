/*
 * glm5_int8.h - INT8 (compressed-tensors w8a16, pack-quantized) matvec for A64FX.
 *
 * GLM-5.2-INT8 stores each quantized weight as `weight_packed` (I32, 4 int8 per int32, LE -> the
 * raw bytes are the int8 row-major weight [rows,cols]) plus a BF16 `weight_scale` [rows, ngroups].
 * The pack is OFFSET-BINARY: the stored byte b encodes q = b - 128 (verified vs bf16, cosine
 * 0.99998). Dequant:  w[r,c] = (b[r,c]-128) * scale[r, c/gs],  gs = cols/ngroups
 *   - group-128 (attention / dense MLP / shared experts): gs = 128
 *   - per-channel (routed experts):                        gs = cols (one group)
 *
 * Like the FP8 path, activations stay f32 (w8a16): decode int8 -> f32 in-register, apply the
 * per-row group scale, accumulate in f32. SVE: svld1ub widens the bytes, subtract 128, convert
 * s32->f32, svmla against x. The scale is per actual row (finer than FP8's 128x128 blocks).
 */
#ifndef GLM5_INT8_H
#define GLM5_INT8_H
#include <stdint.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/* scalar reference: y = sum_c (b[c]-128) * s[c/gs] * x[c] for one row. */
static inline float glm5_dot_int8_row(const uint8_t*w,const float*s,int gs,const float*x,int cols){
    double a=0;
    for(int b=0;b<cols;b+=gs){
        float sc=s[b/gs]; int e=b+gs<cols?b+gs:cols;
        for(int c=b;c<e;c++) a+=(double)((int)w[c]-128)*sc*(double)x[c];
    }
    return (float)a;
}

#if defined(__ARM_FEATURE_SVE)
/* SVE: per group of gs cols, per row, decode int8 (byte-128) -> f32, * row group scale, svmla. */
static inline void glm5_matvec_int8_8row(float*restrict dst,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
        const float*s0,const float*s1,const float*s2,const float*s3,
        const float*s4,const float*s5,const float*s6,const float*s7,
        int gs,const float*x,int cols){
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    int vl=(int)svcntw();
    #define GLM5_I8_ROW(WP,SC,ACC) do{ \
        svuint32_t bz=svld1ub_u32(pg,&(WP)[c]); \
        svfloat32_t wv=svcvt_f32_s32_x(pg, svsub_n_s32_x(pg, svreinterpret_s32_u32(bz), 128)); \
        ACC=svmla_x(pg,ACC,wv,svmul_n_f32_x(pg,xv,(SC))); \
    }while(0)
    for(int b=0;b<cols;b+=gs){
        int bend=b+gs<cols?b+gs:cols, blk=b/gs;
        float c0=s0[blk],c1=s1[blk],c2=s2[blk],c3=s3[blk];
        float c4=s4[blk],c5=s5[blk],c6=s6[blk],c7=s7[blk];
        for(int c=b;c<bend;c+=vl){
            svbool_t pg=svwhilelt_b32(c,bend);
            svfloat32_t xv=svld1(pg,&x[c]);
            GLM5_I8_ROW(w0,c0,a0); GLM5_I8_ROW(w1,c1,a1);
            GLM5_I8_ROW(w2,c2,a2); GLM5_I8_ROW(w3,c3,a3);
            GLM5_I8_ROW(w4,c4,a4); GLM5_I8_ROW(w5,c5,a5);
            GLM5_I8_ROW(w6,c6,a6); GLM5_I8_ROW(w7,c7,a7);
        }
    }
    #undef GLM5_I8_ROW
    svbool_t pt=svptrue_b32();
    dst[0]=svaddv_f32(pt,a0); dst[1]=svaddv_f32(pt,a1); dst[2]=svaddv_f32(pt,a2); dst[3]=svaddv_f32(pt,a3);
    dst[4]=svaddv_f32(pt,a4); dst[5]=svaddv_f32(pt,a5); dst[6]=svaddv_f32(pt,a6); dst[7]=svaddv_f32(pt,a7);
}
#endif

#endif /* GLM5_INT8_H */
