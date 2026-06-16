/*
 * m3_mxfp8.h - MXFP8 (FP8 E4M3 + E8M0 per-[1,32]-block scale) matvec for A64FX.
 *
 * MiniMax-M3-MXFP8 stores each quantized weight as F8_E4M3 [rows,cols] (1 byte/elem) plus
 * a companion E8M0 (uint8) scale [rows, cols/32] -- one power-of-2 block scale per 32
 * contiguous columns per row. Dequant:  w[r,c] = fp8_e4m3(W[r,c]) * 2^(scale[r,c/32]-127).
 *
 * Precision path (best on A64FX): decode FP8 -> f32 (exact, via a 256-entry LUT), apply the
 * block scale, accumulate in f32. The weights stay FP8 in HBM (the memory win); only the
 * in-register values are f32. SVE: svld1ub widens 16 bytes -> uint32 indices, a gather pulls
 * the exact f32 bits from the LUT, then svmla. A scalar reference (m3_*_ref) validates it.
 *
 * NOTE: the E8M0 bias/"scale_inv" sign convention is validated numerically against a Python
 * reference (a64fx/m3/m3_mxfp8_test.c) before wiring into the forward.
 */
#ifndef M3_MXFP8_H
#define M3_MXFP8_H
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>

#define M3_MX_BLK 32   /* E8M0 scale block size (weight_block_size = [1,32]) */

/* FP8 E4M3 (fn: one NaN code, no inf) -> f32 bits. Exact. */
static inline uint32_t m3_fp8_e4m3_bits(uint8_t x){
    uint8_t sign=(x>>7)&1, exp=(x>>3)&0xF, mant=x&0x7;
    if(exp==0){ if(mant==0) return (uint32_t)sign<<31;
        int sh=0; while((mant&0x4)==0){ mant<<=1; sh++; } mant&=0x3;
        uint32_t e=(uint32_t)(127-7-sh); return ((uint32_t)sign<<31)|(e<<23)|((uint32_t)mant<<20); }
    if(exp==15 && mant==7) return ((uint32_t)sign<<31)|(0xFFu<<23)|(1u<<22);
    uint32_t e=(uint32_t)exp+(127-7); return ((uint32_t)sign<<31)|(e<<23)|((uint32_t)mant<<20);
}
static inline void m3_init_fp8_lut(uint32_t *lut){ for(int i=0;i<256;i++) lut[i]=m3_fp8_e4m3_bits((uint8_t)i); }
/* E8M0 byte -> f32 power-of-2 scale = 2^(b-127). b=0 -> 0, b=255 -> NaN (OCP). */
static inline float m3_e8m0(uint8_t b){ uint32_t bits=(uint32_t)b<<23; float f; memcpy(&f,&bits,4); return f; }

/* scalar reference: dst[8] = sum_c lut(w_j[c])*2^(s_j[c/32]-127) * x[c], for 8 rows. */
static inline void m3_matvec_mxfp8_8row_ref(float*dst,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
        const uint8_t*s0,const uint8_t*s1,const uint8_t*s2,const uint8_t*s3,
        const uint8_t*s4,const uint8_t*s5,const uint8_t*s6,const uint8_t*s7,
        const float*x,int n,const uint32_t*lut){
    const uint8_t*w[8]={w0,w1,w2,w3,w4,w5,w6,w7}; const uint8_t*s[8]={s0,s1,s2,s3,s4,s5,s6,s7};
    for(int j=0;j<8;j++){ double a=0;
        for(int b=0;b<n;b+=M3_MX_BLK){ float sc=m3_e8m0(s[j][b/M3_MX_BLK]); int e=b+M3_MX_BLK<n?b+M3_MX_BLK:n;
            for(int c=b;c<e;c++){ float wf; uint32_t u=lut[w[j][c]]; memcpy(&wf,&u,4); a+=(double)wf*sc*x[c]; } }
        dst[j]=(float)a; }
}

#if defined(__ARM_FEATURE_SVE)
/* SVE: per 32-col block, per row, gather-decode FP8->f32 via LUT, * block scale, svmla. */
static inline void m3_matvec_mxfp8_8row(float*dst,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
        const uint8_t*s0,const uint8_t*s1,const uint8_t*s2,const uint8_t*s3,
        const uint8_t*s4,const uint8_t*s5,const uint8_t*s6,const uint8_t*s7,
        const float*x,int n,const uint32_t*lut){
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    int vl=(int)svcntw();
    #define M3MX_ROW(WP,SC,ACC) do{ svuint32_t idx=svld1ub_u32(pg,&(WP)[c]); \
        svfloat32_t wv=svreinterpret_f32_u32(svld1_gather_u32index_u32(pg,lut,idx)); \
        ACC=svmla_x(pg,ACC,svmul_n_f32_x(pg,wv,(SC)),xv); }while(0)
    for(int b=0;b<n;b+=M3_MX_BLK){
        int bend=b+M3_MX_BLK<n?b+M3_MX_BLK:n; int blk=b/M3_MX_BLK;
        float c0=m3_e8m0(s0[blk]),c1=m3_e8m0(s1[blk]),c2=m3_e8m0(s2[blk]),c3=m3_e8m0(s3[blk]);
        float c4=m3_e8m0(s4[blk]),c5=m3_e8m0(s5[blk]),c6=m3_e8m0(s6[blk]),c7=m3_e8m0(s7[blk]);
        for(int c=b;c<bend;c+=vl){
            svbool_t pg=svwhilelt_b32(c,bend); svfloat32_t xv=svld1(pg,&x[c]);
            M3MX_ROW(w0,c0,a0); M3MX_ROW(w1,c1,a1); M3MX_ROW(w2,c2,a2); M3MX_ROW(w3,c3,a3);
            M3MX_ROW(w4,c4,a4); M3MX_ROW(w5,c5,a5); M3MX_ROW(w6,c6,a6); M3MX_ROW(w7,c7,a7);
        }
    }
    #undef M3MX_ROW
    svbool_t pt=svptrue_b32();
    dst[0]=svaddv_f32(pt,a0); dst[1]=svaddv_f32(pt,a1); dst[2]=svaddv_f32(pt,a2); dst[3]=svaddv_f32(pt,a3);
    dst[4]=svaddv_f32(pt,a4); dst[5]=svaddv_f32(pt,a5); dst[6]=svaddv_f32(pt,a6); dst[7]=svaddv_f32(pt,a7);
}
#else
#define m3_matvec_mxfp8_8row m3_matvec_mxfp8_8row_ref
#endif

#endif /* M3_MXFP8_H */
