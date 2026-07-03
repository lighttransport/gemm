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

/* w8a8 SDOT 8-row matvec (M=1 decode): activations pre-quantized to int8 (xq); caller multiplies
 * the result by the activation scale xsc. Offset-binary weights -> signed int8 via eor 0x80. Per
 * group g: svdot_s32 accumulates int8*int8 into 16 int32 lanes; defer to a per-row f32 accumulator
 * folding the group scale sc[g] (svmla_n), one svaddv per row at the end (svaddv is linear). This
 * replaces the w8a16 per-byte u8->f32 convert (the convert-throughput bottleneck) with SDOT
 * (64 int8 MACs/instr on the int pipe), so the FP pipes only do 1 convert+fma per group per row. */
static inline void glm5_matvec_int8_sdot_8row(float*restrict dst,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
        const float*s0,const float*s1,const float*s2,const float*s3,
        const float*s4,const float*s5,const float*s6,const float*s7,
        int gs,const int8_t*xq,int cols){
    svbool_t pf=svptrue_b32(); int vb=(int)svcntb();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    for(int b=0;b<cols;b+=gs){
        int bend=b+gs<cols?b+gs:cols, blk=b/gs, c=b;
        svint32_t d0=svdup_s32(0),d1=svdup_s32(0),d2=svdup_s32(0),d3=svdup_s32(0);
        svint32_t d4=svdup_s32(0),d5=svdup_s32(0),d6=svdup_s32(0),d7=svdup_s32(0);
        #define GLM5_SD(WP,D) do{ svint8_t wv=svreinterpret_s8_u8(sveor_n_u8_x(pg,svld1_u8(pg,&(WP)[c]),0x80)); \
            D=svdot_s32(D,wv,xv); }while(0)
        for(;c<bend;c+=vb){
            svbool_t pg=svwhilelt_b8((uint32_t)c,(uint32_t)bend);
            svint8_t xv=svld1_s8(pg,&xq[c]);
            GLM5_SD(w0,d0); GLM5_SD(w1,d1); GLM5_SD(w2,d2); GLM5_SD(w3,d3);
            GLM5_SD(w4,d4); GLM5_SD(w5,d5); GLM5_SD(w6,d6); GLM5_SD(w7,d7);
        }
        #undef GLM5_SD
        a0=svmla_n_f32_x(pf,a0,svcvt_f32_s32_x(pf,d0),s0[blk]);
        a1=svmla_n_f32_x(pf,a1,svcvt_f32_s32_x(pf,d1),s1[blk]);
        a2=svmla_n_f32_x(pf,a2,svcvt_f32_s32_x(pf,d2),s2[blk]);
        a3=svmla_n_f32_x(pf,a3,svcvt_f32_s32_x(pf,d3),s3[blk]);
        a4=svmla_n_f32_x(pf,a4,svcvt_f32_s32_x(pf,d4),s4[blk]);
        a5=svmla_n_f32_x(pf,a5,svcvt_f32_s32_x(pf,d5),s5[blk]);
        a6=svmla_n_f32_x(pf,a6,svcvt_f32_s32_x(pf,d6),s6[blk]);
        a7=svmla_n_f32_x(pf,a7,svcvt_f32_s32_x(pf,d7),s7[blk]);
    }
    dst[0]=svaddv_f32(pf,a0); dst[1]=svaddv_f32(pf,a1);
    dst[2]=svaddv_f32(pf,a2); dst[3]=svaddv_f32(pf,a3);
    dst[4]=svaddv_f32(pf,a4); dst[5]=svaddv_f32(pf,a5);
    dst[6]=svaddv_f32(pf,a6); dst[7]=svaddv_f32(pf,a7);
}

/* scalar w8a8 reference (pre-quantized xq), y = xsc * sum_g sc[g] * sum_c (b[c]-128)*xq[c]. */
static inline float glm5_dot_int8_sdot_row(const uint8_t*w,const float*s,int gs,const int8_t*xq,int cols){
    double a=0;
    for(int b=0;b<cols;b+=gs){
        float sc=s[b/gs]; int e=b+gs<cols?b+gs:cols; long acc=0;
        for(int c=b;c<e;c++) acc+=(long)((int)w[c]-128)*(int)xq[c];
        a+=(double)acc*sc;
    }
    return (float)a;
}

/* w8a16-MIMIC via int16 SDOT: the QuantTrio GLM-5.2-Int8 checkpoint is w8a16 (int8 weight x 16-bit
 * activation) — quantizing activations to int8 (glm5_matvec_int8_sdot_8row) is more aggressive than
 * the model expects and loses accuracy on outlier-heavy activations. Instead quantize the activation
 * to INT16 (per-vector symmetric, ~3e-5/elem = near-lossless, finer than bf16's 8-bit mantissa) and
 * contract with svdot_s64 (int16xint16->int64, 32 MACs/instr). Offset-binary weight byte -> int16 via
 * (byte-128). Deferred per-group f64 scale fold + one svaddv/row. ~2x denser than the convert-bound
 * w8a16 f32 kernel (half of the int8-SDOT density) but with w8a16 accuracy. Caller multiplies by xsc. */
static inline void glm5_matvec_int16sdot_8row(float*restrict dst,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
        const float*s0,const float*s1,const float*s2,const float*s3,
        const float*s4,const float*s5,const float*s6,const float*s7,
        int gs,const int16_t*xq,int cols){
    svbool_t pf=svptrue_b64(); int vh=(int)svcnth();
    svfloat64_t a0=svdup_f64(0),a1=svdup_f64(0),a2=svdup_f64(0),a3=svdup_f64(0);
    svfloat64_t a4=svdup_f64(0),a5=svdup_f64(0),a6=svdup_f64(0),a7=svdup_f64(0);
    for(int b=0;b<cols;b+=gs){
        int bend=b+gs<cols?b+gs:cols, blk=b/gs, c=b;
        svint64_t d0=svdup_s64(0),d1=svdup_s64(0),d2=svdup_s64(0),d3=svdup_s64(0);
        svint64_t d4=svdup_s64(0),d5=svdup_s64(0),d6=svdup_s64(0),d7=svdup_s64(0);
        #define GLM5_SD16(WP,D) do{ svint16_t wv=svsub_n_s16_x(pg,svreinterpret_s16_u16(svld1ub_u16(pg,&(WP)[c])),128); \
            D=svdot_s64(D,wv,xv); }while(0)
        for(;c<bend;c+=vh){
            svbool_t pg=svwhilelt_b16((uint32_t)c,(uint32_t)bend);
            svint16_t xv=svld1_s16(pg,&xq[c]);
            GLM5_SD16(w0,d0); GLM5_SD16(w1,d1); GLM5_SD16(w2,d2); GLM5_SD16(w3,d3);
            GLM5_SD16(w4,d4); GLM5_SD16(w5,d5); GLM5_SD16(w6,d6); GLM5_SD16(w7,d7);
        }
        #undef GLM5_SD16
        a0=svmla_n_f64_x(pf,a0,svcvt_f64_s64_x(pf,d0),(double)s0[blk]);
        a1=svmla_n_f64_x(pf,a1,svcvt_f64_s64_x(pf,d1),(double)s1[blk]);
        a2=svmla_n_f64_x(pf,a2,svcvt_f64_s64_x(pf,d2),(double)s2[blk]);
        a3=svmla_n_f64_x(pf,a3,svcvt_f64_s64_x(pf,d3),(double)s3[blk]);
        a4=svmla_n_f64_x(pf,a4,svcvt_f64_s64_x(pf,d4),(double)s4[blk]);
        a5=svmla_n_f64_x(pf,a5,svcvt_f64_s64_x(pf,d5),(double)s5[blk]);
        a6=svmla_n_f64_x(pf,a6,svcvt_f64_s64_x(pf,d6),(double)s6[blk]);
        a7=svmla_n_f64_x(pf,a7,svcvt_f64_s64_x(pf,d7),(double)s7[blk]);
    }
    dst[0]=(float)svaddv_f64(pf,a0); dst[1]=(float)svaddv_f64(pf,a1);
    dst[2]=(float)svaddv_f64(pf,a2); dst[3]=(float)svaddv_f64(pf,a3);
    dst[4]=(float)svaddv_f64(pf,a4); dst[5]=(float)svaddv_f64(pf,a5);
    dst[6]=(float)svaddv_f64(pf,a6); dst[7]=(float)svaddv_f64(pf,a7);
}
static inline float glm5_dot_int16sdot_row(const uint8_t*w,const float*s,int gs,const int16_t*xq,int cols){
    double a=0;
    for(int b=0;b<cols;b+=gs){
        float sc=s[b/gs]; int e=b+gs<cols?b+gs:cols; long acc=0;
        for(int c=b;c<e;c++) acc+=(long)((int)w[c]-128)*(int)xq[c];
        a+=(double)acc*sc;
    }
    return (float)a;
}

#if defined(__ARM_FEATURE_SVE)
/* PREFILL int16-SDOT GEMM micro-kernel: 4 weight rows x 5 token-streams register block. Reads int8
 * weights directly (offset-binary), widens each K-vector to int16 ONCE (byte-128) and reuses it across
 * all 5 tokens (amortizes the widen — the prefill analog of the bf16 tile-dequant). Per group: svdot_s64
 * into 20 int64 accumulators, then fold the per-row group scale into 20 f64 accumulators. Accumulates
 * into a0..a4[4] (the 4 rows for each of the 5 tokens). Caller multiplies each token's output by xsc[t].
 * cols0 = absolute start column (for scale group indexing); processes [0,kl) of the given row pointers. */
static inline void glm5_int16sdot_4row_5x(float*restrict a0,float*restrict a1,float*restrict a2,float*restrict a3,float*restrict a4,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const float*s0,const float*s1,const float*s2,const float*s3,int gs,int col0,
        const int16_t*xq0,const int16_t*xq1,const int16_t*xq2,const int16_t*xq3,const int16_t*xq4,int kl){
    svbool_t pf=svptrue_b64(); int vh=(int)svcnth();
    svfloat64_t f00=svdup_f64(0),f01=svdup_f64(0),f02=svdup_f64(0),f03=svdup_f64(0);
    svfloat64_t f10=svdup_f64(0),f11=svdup_f64(0),f12=svdup_f64(0),f13=svdup_f64(0);
    svfloat64_t f20=svdup_f64(0),f21=svdup_f64(0),f22=svdup_f64(0),f23=svdup_f64(0);
    svfloat64_t f30=svdup_f64(0),f31=svdup_f64(0),f32=svdup_f64(0),f33=svdup_f64(0);
    svfloat64_t f40=svdup_f64(0),f41=svdup_f64(0),f42=svdup_f64(0),f43=svdup_f64(0);
    for(int b=0;b<kl;b+=gs){
        int bend=b+gs<kl?b+gs:kl, blk=(col0+b)/gs, c=b;
        svint64_t d00=svdup_s64(0),d01=svdup_s64(0),d02=svdup_s64(0),d03=svdup_s64(0);
        svint64_t d10=svdup_s64(0),d11=svdup_s64(0),d12=svdup_s64(0),d13=svdup_s64(0);
        svint64_t d20=svdup_s64(0),d21=svdup_s64(0),d22=svdup_s64(0),d23=svdup_s64(0);
        svint64_t d30=svdup_s64(0),d31=svdup_s64(0),d32=svdup_s64(0),d33=svdup_s64(0);
        svint64_t d40=svdup_s64(0),d41=svdup_s64(0),d42=svdup_s64(0),d43=svdup_s64(0);
        for(;c<bend;c+=vh){
            svbool_t pg=svwhilelt_b16((uint32_t)c,(uint32_t)bend);
            svint16_t v0=svsub_n_s16_x(pg,svreinterpret_s16_u16(svld1ub_u16(pg,&w0[c])),128);
            svint16_t v1=svsub_n_s16_x(pg,svreinterpret_s16_u16(svld1ub_u16(pg,&w1[c])),128);
            svint16_t v2=svsub_n_s16_x(pg,svreinterpret_s16_u16(svld1ub_u16(pg,&w2[c])),128);
            svint16_t v3=svsub_n_s16_x(pg,svreinterpret_s16_u16(svld1ub_u16(pg,&w3[c])),128);
            svint16_t xv;
            xv=svld1_s16(pg,&xq0[c]); d00=svdot_s64(d00,v0,xv); d01=svdot_s64(d01,v1,xv); d02=svdot_s64(d02,v2,xv); d03=svdot_s64(d03,v3,xv);
            xv=svld1_s16(pg,&xq1[c]); d10=svdot_s64(d10,v0,xv); d11=svdot_s64(d11,v1,xv); d12=svdot_s64(d12,v2,xv); d13=svdot_s64(d13,v3,xv);
            xv=svld1_s16(pg,&xq2[c]); d20=svdot_s64(d20,v0,xv); d21=svdot_s64(d21,v1,xv); d22=svdot_s64(d22,v2,xv); d23=svdot_s64(d23,v3,xv);
            xv=svld1_s16(pg,&xq3[c]); d30=svdot_s64(d30,v0,xv); d31=svdot_s64(d31,v1,xv); d32=svdot_s64(d32,v2,xv); d33=svdot_s64(d33,v3,xv);
            xv=svld1_s16(pg,&xq4[c]); d40=svdot_s64(d40,v0,xv); d41=svdot_s64(d41,v1,xv); d42=svdot_s64(d42,v2,xv); d43=svdot_s64(d43,v3,xv);
        }
        double c0=s0[blk],c1=s1[blk],c2=s2[blk],c3=s3[blk];
        f00=svmla_n_f64_x(pf,f00,svcvt_f64_s64_x(pf,d00),c0); f01=svmla_n_f64_x(pf,f01,svcvt_f64_s64_x(pf,d01),c1); f02=svmla_n_f64_x(pf,f02,svcvt_f64_s64_x(pf,d02),c2); f03=svmla_n_f64_x(pf,f03,svcvt_f64_s64_x(pf,d03),c3);
        f10=svmla_n_f64_x(pf,f10,svcvt_f64_s64_x(pf,d10),c0); f11=svmla_n_f64_x(pf,f11,svcvt_f64_s64_x(pf,d11),c1); f12=svmla_n_f64_x(pf,f12,svcvt_f64_s64_x(pf,d12),c2); f13=svmla_n_f64_x(pf,f13,svcvt_f64_s64_x(pf,d13),c3);
        f20=svmla_n_f64_x(pf,f20,svcvt_f64_s64_x(pf,d20),c0); f21=svmla_n_f64_x(pf,f21,svcvt_f64_s64_x(pf,d21),c1); f22=svmla_n_f64_x(pf,f22,svcvt_f64_s64_x(pf,d22),c2); f23=svmla_n_f64_x(pf,f23,svcvt_f64_s64_x(pf,d23),c3);
        f30=svmla_n_f64_x(pf,f30,svcvt_f64_s64_x(pf,d30),c0); f31=svmla_n_f64_x(pf,f31,svcvt_f64_s64_x(pf,d31),c1); f32=svmla_n_f64_x(pf,f32,svcvt_f64_s64_x(pf,d32),c2); f33=svmla_n_f64_x(pf,f33,svcvt_f64_s64_x(pf,d33),c3);
        f40=svmla_n_f64_x(pf,f40,svcvt_f64_s64_x(pf,d40),c0); f41=svmla_n_f64_x(pf,f41,svcvt_f64_s64_x(pf,d41),c1); f42=svmla_n_f64_x(pf,f42,svcvt_f64_s64_x(pf,d42),c2); f43=svmla_n_f64_x(pf,f43,svcvt_f64_s64_x(pf,d43),c3);
    }
    a0[0]+=svaddv_f64(pf,f00); a0[1]+=svaddv_f64(pf,f01); a0[2]+=svaddv_f64(pf,f02); a0[3]+=svaddv_f64(pf,f03);
    a1[0]+=svaddv_f64(pf,f10); a1[1]+=svaddv_f64(pf,f11); a1[2]+=svaddv_f64(pf,f12); a1[3]+=svaddv_f64(pf,f13);
    a2[0]+=svaddv_f64(pf,f20); a2[1]+=svaddv_f64(pf,f21); a2[2]+=svaddv_f64(pf,f22); a2[3]+=svaddv_f64(pf,f23);
    a3[0]+=svaddv_f64(pf,f30); a3[1]+=svaddv_f64(pf,f31); a3[2]+=svaddv_f64(pf,f32); a3[3]+=svaddv_f64(pf,f33);
    a4[0]+=svaddv_f64(pf,f40); a4[1]+=svaddv_f64(pf,f41); a4[2]+=svaddv_f64(pf,f42); a4[3]+=svaddv_f64(pf,f43);
}
#endif
#endif

#endif /* GLM5_INT8_H */
