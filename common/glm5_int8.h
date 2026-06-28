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
/* SVE group-factored: per group accumulate the UNSIGNED byte dot (no per-element -128, no
 * per-element scale), fold the scale once per group (A += groupdot*sc), and track the
 * correction term Σ sc*Σx. Final y = Σ_g sc_g*Σ(byte*x) - 128*Σ_g sc_g*Σx = Σ(byte-128)*sc*x.
 * Drops 2 of 5 inner ops (the subtract + the per-element scale-multiply) vs the naive form. */
static inline void glm5_matvec_int8_8row(float*restrict dst,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
        const float*s0,const float*s1,const float*s2,const float*s3,
        const float*s4,const float*s5,const float*s6,const float*s7,
        int gs,const float*x,int cols){
    svbool_t pt=svptrue_b32(); int vl=(int)svcntw();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    float cr0=0,cr1=0,cr2=0,cr3=0,cr4=0,cr5=0,cr6=0,cr7=0;
    for(int b=0;b<cols;b+=gs){
        int bend=b+gs<cols?b+gs:cols, blk=b/gs;
        svfloat32_t g0=svdup_f32(0.f),g1=svdup_f32(0.f),g2=svdup_f32(0.f),g3=svdup_f32(0.f);
        svfloat32_t g4=svdup_f32(0.f),g5=svdup_f32(0.f),g6=svdup_f32(0.f),g7=svdup_f32(0.f);
        svfloat32_t sx=svdup_f32(0.f);
        #define GLM5_I8_R(WP,G) do{ svfloat32_t wv=svcvt_f32_u32_x(pg,svld1ub_u32(pg,&(WP)[c])); \
            G=svmla_m(pg,G,wv,xv); }while(0)
        for(int c=b;c<bend;c+=vl){
            svbool_t pg=svwhilelt_b32(c,bend);
            svfloat32_t xv=svld1(pg,&x[c]);
            sx=svadd_f32_m(pg,sx,xv);
            GLM5_I8_R(w0,g0); GLM5_I8_R(w1,g1); GLM5_I8_R(w2,g2); GLM5_I8_R(w3,g3);
            GLM5_I8_R(w4,g4); GLM5_I8_R(w5,g5); GLM5_I8_R(w6,g6); GLM5_I8_R(w7,g7);
        }
        #undef GLM5_I8_R
        float Sx=svaddv_f32(pt,sx);
        float v0=s0[blk],v1=s1[blk],v2=s2[blk],v3=s3[blk],v4=s4[blk],v5=s5[blk],v6=s6[blk],v7=s7[blk];
        a0=svmla_n_f32_x(pt,a0,g0,v0); a1=svmla_n_f32_x(pt,a1,g1,v1);
        a2=svmla_n_f32_x(pt,a2,g2,v2); a3=svmla_n_f32_x(pt,a3,g3,v3);
        a4=svmla_n_f32_x(pt,a4,g4,v4); a5=svmla_n_f32_x(pt,a5,g5,v5);
        a6=svmla_n_f32_x(pt,a6,g6,v6); a7=svmla_n_f32_x(pt,a7,g7,v7);
        cr0+=v0*Sx; cr1+=v1*Sx; cr2+=v2*Sx; cr3+=v3*Sx;
        cr4+=v4*Sx; cr5+=v5*Sx; cr6+=v6*Sx; cr7+=v7*Sx;
    }
    dst[0]=svaddv_f32(pt,a0)-128.f*cr0; dst[1]=svaddv_f32(pt,a1)-128.f*cr1;
    dst[2]=svaddv_f32(pt,a2)-128.f*cr2; dst[3]=svaddv_f32(pt,a3)-128.f*cr3;
    dst[4]=svaddv_f32(pt,a4)-128.f*cr4; dst[5]=svaddv_f32(pt,a5)-128.f*cr5;
    dst[6]=svaddv_f32(pt,a6)-128.f*cr6; dst[7]=svaddv_f32(pt,a7)-128.f*cr7;
}
#endif

#endif /* GLM5_INT8_H */
