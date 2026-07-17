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

/* scalar reference for a possibly column-sliced row. qg0 is the original column offset
 * relative to the first copied scale group. */
static inline float glm5_dot_int8_row(const uint8_t*w,const float*s,int gs,int qg0,const float*x,int cols){
    double a=0;
    for(int b=0;b<cols;){
        int blk=(qg0+b)/gs, e=(blk+1)*gs-qg0; if(e>cols)e=cols;
        float sc=s[blk];
        for(int c=b;c<e;c++) a+=(double)((int)w[c]-128)*sc*(double)x[c];
        b=e;
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
        int gs,int qg0,const float*x,int cols){
    svbool_t pt=svptrue_b32(); int vl=(int)svcntw();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    float cr0=0,cr1=0,cr2=0,cr3=0,cr4=0,cr5=0,cr6=0,cr7=0;
    for(int b=0;b<cols;){
        int blk=(qg0+b)/gs, bend=(blk+1)*gs-qg0; if(bend>cols)bend=cols;
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
        b=bend;
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
        int gs,int qg0,const int8_t*xq,int cols){
    svbool_t pf=svptrue_b32(); int vb=(int)svcntb();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    for(int b=0;b<cols;){
        int blk=(qg0+b)/gs, bend=(blk+1)*gs-qg0; if(bend>cols)bend=cols; int c=b;
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
        b=bend;
    }
    dst[0]=svaddv_f32(pf,a0); dst[1]=svaddv_f32(pf,a1);
    dst[2]=svaddv_f32(pf,a2); dst[3]=svaddv_f32(pf,a3);
    dst[4]=svaddv_f32(pf,a4); dst[5]=svaddv_f32(pf,a5);
    dst[6]=svaddv_f32(pf,a6); dst[7]=svaddv_f32(pf,a7);
}

/* scalar w8a8 reference (pre-quantized xq), y = xsc * sum_g sc[g] * sum_c (b[c]-128)*xq[c]. */
static inline float glm5_dot_int8_sdot_row(const uint8_t*w,const float*s,int gs,int qg0,const int8_t*xq,int cols){
    double a=0;
    for(int b=0;b<cols;){
        int blk=(qg0+b)/gs, e=(blk+1)*gs-qg0; if(e>cols)e=cols;
        float sc=s[blk]; long acc=0;
        for(int c=b;c<e;c++) acc+=(long)((int)w[c]-128)*(int)xq[c];
        a+=(double)acc*sc;
        b=e;
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
        int gs,int qg0,const int16_t*xq,const int64_t*restrict xgsum,int cols){
    /* BIAS-FOLD widen (integer analog of the bf16 p_odd "no-shift" trick): the offset-binary
     * weight byte is zero-extended to a positive s16 (0..255) with svld1ub_u16 ONLY — the
     * per-vector svsub_n_s16(-128) that competed with svdot_s64 on FLA/FLB is dropped. The
     * -128 offset is corrected per group as -128*sum(xq[group]) via the precomputed xgsum
     * (built once per activation vector, shared across every weight row). Bit-exact. */
    svbool_t pf=svptrue_b64(); int vh=(int)svcnth();
    svfloat64_t a0=svdup_f64(0),a1=svdup_f64(0),a2=svdup_f64(0),a3=svdup_f64(0);
    svfloat64_t a4=svdup_f64(0),a5=svdup_f64(0),a6=svdup_f64(0),a7=svdup_f64(0);
    double bc0=0,bc1=0,bc2=0,bc3=0,bc4=0,bc5=0,bc6=0,bc7=0;
    const int fast_g64=gs==64 && (qg0&63)==0 && (cols&63)==0;
    for(int b=0;b<cols;){
        int blk=fast_g64?(qg0+b)>>6:(qg0+b)/gs;
        int bend=fast_g64?b+64:(blk+1)*gs-qg0; if(bend>cols)bend=cols; int c=b;
        /* A64FX cache lines are 256 B.  On the production group-64 path, issue one
         * hint per cache line, 512 B ahead, instead of four redundant per-group hints.
         * The generic path retains the established schedule for unusual group layouts. */
        int pdist=fast_g64?512:1024;
        if(b+pdist<cols && (!fast_g64 || (b&255)==0)){
            __builtin_prefetch(&w0[b+pdist]); __builtin_prefetch(&w1[b+pdist]);
            __builtin_prefetch(&w2[b+pdist]); __builtin_prefetch(&w3[b+pdist]);
            __builtin_prefetch(&w4[b+pdist]); __builtin_prefetch(&w5[b+pdist]);
            __builtin_prefetch(&w6[b+pdist]); __builtin_prefetch(&w7[b+pdist]);
        }
        svint64_t d0=svdup_s64(0),d1=svdup_s64(0),d2=svdup_s64(0),d3=svdup_s64(0);
        svint64_t d4=svdup_s64(0),d5=svdup_s64(0),d6=svdup_s64(0),d7=svdup_s64(0);
        #define GLM5_SD16(WP,D) do{ svint16_t wv=svreinterpret_s16_u16(svld1ub_u16(pg,&(WP)[c])); \
            D=svdot_s64(D,wv,xv); }while(0)
        for(;c<bend;c+=vh){
            svbool_t pg=svwhilelt_b16((uint32_t)c,(uint32_t)bend);
            svint16_t xv=svld1_s16(pg,&xq[c]);
            GLM5_SD16(w0,d0); GLM5_SD16(w1,d1); GLM5_SD16(w2,d2); GLM5_SD16(w3,d3);
            GLM5_SD16(w4,d4); GLM5_SD16(w5,d5); GLM5_SD16(w6,d6); GLM5_SD16(w7,d7);
        }
        #undef GLM5_SD16
        double sc0=s0[blk],sc1=s1[blk],sc2=s2[blk],sc3=s3[blk];
        double sc4=s4[blk],sc5=s5[blk],sc6=s6[blk],sc7=s7[blk],xg=(double)xgsum[blk];
        a0=svmla_n_f64_x(pf,a0,svcvt_f64_s64_x(pf,d0),sc0);
        a1=svmla_n_f64_x(pf,a1,svcvt_f64_s64_x(pf,d1),sc1);
        a2=svmla_n_f64_x(pf,a2,svcvt_f64_s64_x(pf,d2),sc2);
        a3=svmla_n_f64_x(pf,a3,svcvt_f64_s64_x(pf,d3),sc3);
        a4=svmla_n_f64_x(pf,a4,svcvt_f64_s64_x(pf,d4),sc4);
        a5=svmla_n_f64_x(pf,a5,svcvt_f64_s64_x(pf,d5),sc5);
        a6=svmla_n_f64_x(pf,a6,svcvt_f64_s64_x(pf,d6),sc6);
        a7=svmla_n_f64_x(pf,a7,svcvt_f64_s64_x(pf,d7),sc7);
        bc0+=sc0*xg; bc1+=sc1*xg; bc2+=sc2*xg; bc3+=sc3*xg;
        bc4+=sc4*xg; bc5+=sc5*xg; bc6+=sc6*xg; bc7+=sc7*xg;
        b=bend;
    }
    dst[0]=(float)(svaddv_f64(pf,a0)-128.0*bc0); dst[1]=(float)(svaddv_f64(pf,a1)-128.0*bc1);
    dst[2]=(float)(svaddv_f64(pf,a2)-128.0*bc2); dst[3]=(float)(svaddv_f64(pf,a3)-128.0*bc3);
    dst[4]=(float)(svaddv_f64(pf,a4)-128.0*bc4); dst[5]=(float)(svaddv_f64(pf,a5)-128.0*bc5);
    dst[6]=(float)(svaddv_f64(pf,a6)-128.0*bc6); dst[7]=(float)(svaddv_f64(pf,a7)-128.0*bc7);
}
static inline float glm5_dot_int16sdot_row(const uint8_t*w,const float*s,int gs,int qg0,const int16_t*xq,int cols){
    double a=0;
    for(int b=0;b<cols;){
        int blk=(qg0+b)/gs, e=(blk+1)*gs-qg0; if(e>cols)e=cols;
        float sc=s[blk]; long acc=0;
        for(int c=b;c<e;c++) acc+=(long)((int)w[c]-128)*(int)xq[c];
        a+=(double)acc*sc;
        b=e;
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
        const int16_t*xq0,const int16_t*xq1,const int16_t*xq2,const int16_t*xq3,const int16_t*xq4,
        const int64_t*restrict xg0,const int64_t*restrict xg1,const int64_t*restrict xg2,
        const int64_t*restrict xg3,const int64_t*restrict xg4,int kl){
    /* BIAS-FOLD widen: drop the per-vector svsub_n_s16(-128); weight byte -> positive s16 via
     * svld1ub_u16 only, correct per group with -128*xgsum[token][blk] (precomputed once per
     * token activation, reused across all weight rows). See glm5_matvec_int16sdot_8row. */
    svbool_t pf=svptrue_b64(); int vh=(int)svcnth();
    svfloat64_t f00=svdup_f64(0),f01=svdup_f64(0),f02=svdup_f64(0),f03=svdup_f64(0);
    svfloat64_t f10=svdup_f64(0),f11=svdup_f64(0),f12=svdup_f64(0),f13=svdup_f64(0);
    svfloat64_t f20=svdup_f64(0),f21=svdup_f64(0),f22=svdup_f64(0),f23=svdup_f64(0);
    svfloat64_t f30=svdup_f64(0),f31=svdup_f64(0),f32=svdup_f64(0),f33=svdup_f64(0);
    svfloat64_t f40=svdup_f64(0),f41=svdup_f64(0),f42=svdup_f64(0),f43=svdup_f64(0);
    double b00=0,b01=0,b02=0,b03=0, b10=0,b11=0,b12=0,b13=0, b20=0,b21=0,b22=0,b23=0;
    double b30=0,b31=0,b32=0,b33=0, b40=0,b41=0,b42=0,b43=0;
    for(int b=0;b<kl;){
        int blk=(col0+b)/gs, bend=(blk+1)*gs-col0; if(bend>kl)bend=kl; int c=b;
        svint64_t d00=svdup_s64(0),d01=svdup_s64(0),d02=svdup_s64(0),d03=svdup_s64(0);
        svint64_t d10=svdup_s64(0),d11=svdup_s64(0),d12=svdup_s64(0),d13=svdup_s64(0);
        svint64_t d20=svdup_s64(0),d21=svdup_s64(0),d22=svdup_s64(0),d23=svdup_s64(0);
        svint64_t d30=svdup_s64(0),d31=svdup_s64(0),d32=svdup_s64(0),d33=svdup_s64(0);
        svint64_t d40=svdup_s64(0),d41=svdup_s64(0),d42=svdup_s64(0),d43=svdup_s64(0);
        for(;c<bend;c+=vh){
            svbool_t pg=svwhilelt_b16((uint32_t)c,(uint32_t)bend);
            svint16_t v0=svreinterpret_s16_u16(svld1ub_u16(pg,&w0[c]));
            svint16_t v1=svreinterpret_s16_u16(svld1ub_u16(pg,&w1[c]));
            svint16_t v2=svreinterpret_s16_u16(svld1ub_u16(pg,&w2[c]));
            svint16_t v3=svreinterpret_s16_u16(svld1ub_u16(pg,&w3[c]));
            svint16_t xv;
            xv=svld1_s16(pg,&xq0[c]); d00=svdot_s64(d00,v0,xv); d01=svdot_s64(d01,v1,xv); d02=svdot_s64(d02,v2,xv); d03=svdot_s64(d03,v3,xv);
            xv=svld1_s16(pg,&xq1[c]); d10=svdot_s64(d10,v0,xv); d11=svdot_s64(d11,v1,xv); d12=svdot_s64(d12,v2,xv); d13=svdot_s64(d13,v3,xv);
            xv=svld1_s16(pg,&xq2[c]); d20=svdot_s64(d20,v0,xv); d21=svdot_s64(d21,v1,xv); d22=svdot_s64(d22,v2,xv); d23=svdot_s64(d23,v3,xv);
            xv=svld1_s16(pg,&xq3[c]); d30=svdot_s64(d30,v0,xv); d31=svdot_s64(d31,v1,xv); d32=svdot_s64(d32,v2,xv); d33=svdot_s64(d33,v3,xv);
            xv=svld1_s16(pg,&xq4[c]); d40=svdot_s64(d40,v0,xv); d41=svdot_s64(d41,v1,xv); d42=svdot_s64(d42,v2,xv); d43=svdot_s64(d43,v3,xv);
        }
        double c0=s0[blk],c1=s1[blk],c2=s2[blk],c3=s3[blk];
        double g0=(double)xg0[blk],g1=(double)xg1[blk],g2=(double)xg2[blk],g3=(double)xg3[blk],g4=(double)xg4[blk];
        f00=svmla_n_f64_x(pf,f00,svcvt_f64_s64_x(pf,d00),c0); f01=svmla_n_f64_x(pf,f01,svcvt_f64_s64_x(pf,d01),c1); f02=svmla_n_f64_x(pf,f02,svcvt_f64_s64_x(pf,d02),c2); f03=svmla_n_f64_x(pf,f03,svcvt_f64_s64_x(pf,d03),c3);
        f10=svmla_n_f64_x(pf,f10,svcvt_f64_s64_x(pf,d10),c0); f11=svmla_n_f64_x(pf,f11,svcvt_f64_s64_x(pf,d11),c1); f12=svmla_n_f64_x(pf,f12,svcvt_f64_s64_x(pf,d12),c2); f13=svmla_n_f64_x(pf,f13,svcvt_f64_s64_x(pf,d13),c3);
        f20=svmla_n_f64_x(pf,f20,svcvt_f64_s64_x(pf,d20),c0); f21=svmla_n_f64_x(pf,f21,svcvt_f64_s64_x(pf,d21),c1); f22=svmla_n_f64_x(pf,f22,svcvt_f64_s64_x(pf,d22),c2); f23=svmla_n_f64_x(pf,f23,svcvt_f64_s64_x(pf,d23),c3);
        f30=svmla_n_f64_x(pf,f30,svcvt_f64_s64_x(pf,d30),c0); f31=svmla_n_f64_x(pf,f31,svcvt_f64_s64_x(pf,d31),c1); f32=svmla_n_f64_x(pf,f32,svcvt_f64_s64_x(pf,d32),c2); f33=svmla_n_f64_x(pf,f33,svcvt_f64_s64_x(pf,d33),c3);
        f40=svmla_n_f64_x(pf,f40,svcvt_f64_s64_x(pf,d40),c0); f41=svmla_n_f64_x(pf,f41,svcvt_f64_s64_x(pf,d41),c1); f42=svmla_n_f64_x(pf,f42,svcvt_f64_s64_x(pf,d42),c2); f43=svmla_n_f64_x(pf,f43,svcvt_f64_s64_x(pf,d43),c3);
        b00+=c0*g0; b01+=c1*g0; b02+=c2*g0; b03+=c3*g0;
        b10+=c0*g1; b11+=c1*g1; b12+=c2*g1; b13+=c3*g1;
        b20+=c0*g2; b21+=c1*g2; b22+=c2*g2; b23+=c3*g2;
        b30+=c0*g3; b31+=c1*g3; b32+=c2*g3; b33+=c3*g3;
        b40+=c0*g4; b41+=c1*g4; b42+=c2*g4; b43+=c3*g4;
        b=bend;
    }
    a0[0]+=svaddv_f64(pf,f00)-128.0*b00; a0[1]+=svaddv_f64(pf,f01)-128.0*b01; a0[2]+=svaddv_f64(pf,f02)-128.0*b02; a0[3]+=svaddv_f64(pf,f03)-128.0*b03;
    a1[0]+=svaddv_f64(pf,f10)-128.0*b10; a1[1]+=svaddv_f64(pf,f11)-128.0*b11; a1[2]+=svaddv_f64(pf,f12)-128.0*b12; a1[3]+=svaddv_f64(pf,f13)-128.0*b13;
    a2[0]+=svaddv_f64(pf,f20)-128.0*b20; a2[1]+=svaddv_f64(pf,f21)-128.0*b21; a2[2]+=svaddv_f64(pf,f22)-128.0*b22; a2[3]+=svaddv_f64(pf,f23)-128.0*b23;
    a3[0]+=svaddv_f64(pf,f30)-128.0*b30; a3[1]+=svaddv_f64(pf,f31)-128.0*b31; a3[2]+=svaddv_f64(pf,f32)-128.0*b32; a3[3]+=svaddv_f64(pf,f33)-128.0*b33;
    a4[0]+=svaddv_f64(pf,f40)-128.0*b40; a4[1]+=svaddv_f64(pf,f41)-128.0*b41; a4[2]+=svaddv_f64(pf,f42)-128.0*b42; a4[3]+=svaddv_f64(pf,f43)-128.0*b43;
}

static inline void glm5_int16sdot_4row_4x(float*restrict a0,float*restrict a1,float*restrict a2,float*restrict a3,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const float*s0,const float*s1,const float*s2,const float*s3,int gs,int col0,
        const int16_t*xq0,const int16_t*xq1,const int16_t*xq2,const int16_t*xq3,
        const int64_t*restrict xg0,const int64_t*restrict xg1,const int64_t*restrict xg2,const int64_t*restrict xg3,int kl){
    /* BIAS-FOLD widen (see glm5_int16sdot_4row_5x): svld1ub_u16 only + per-group -128*xgsum. */
    svbool_t pf=svptrue_b64(); int vh=(int)svcnth();
    svfloat64_t f00=svdup_f64(0),f01=svdup_f64(0),f02=svdup_f64(0),f03=svdup_f64(0);
    svfloat64_t f10=svdup_f64(0),f11=svdup_f64(0),f12=svdup_f64(0),f13=svdup_f64(0);
    svfloat64_t f20=svdup_f64(0),f21=svdup_f64(0),f22=svdup_f64(0),f23=svdup_f64(0);
    svfloat64_t f30=svdup_f64(0),f31=svdup_f64(0),f32=svdup_f64(0),f33=svdup_f64(0);
    double b00=0,b01=0,b02=0,b03=0, b10=0,b11=0,b12=0,b13=0;
    double b20=0,b21=0,b22=0,b23=0, b30=0,b31=0,b32=0,b33=0;
    for(int b=0;b<kl;){
        int blk=(col0+b)/gs, bend=(blk+1)*gs-col0; if(bend>kl)bend=kl; int c=b;
        svint64_t d00=svdup_s64(0),d01=svdup_s64(0),d02=svdup_s64(0),d03=svdup_s64(0);
        svint64_t d10=svdup_s64(0),d11=svdup_s64(0),d12=svdup_s64(0),d13=svdup_s64(0);
        svint64_t d20=svdup_s64(0),d21=svdup_s64(0),d22=svdup_s64(0),d23=svdup_s64(0);
        svint64_t d30=svdup_s64(0),d31=svdup_s64(0),d32=svdup_s64(0),d33=svdup_s64(0);
        for(;c<bend;c+=vh){
            svbool_t pg=svwhilelt_b16((uint32_t)c,(uint32_t)bend);
            svint16_t v0=svreinterpret_s16_u16(svld1ub_u16(pg,&w0[c]));
            svint16_t v1=svreinterpret_s16_u16(svld1ub_u16(pg,&w1[c]));
            svint16_t v2=svreinterpret_s16_u16(svld1ub_u16(pg,&w2[c]));
            svint16_t v3=svreinterpret_s16_u16(svld1ub_u16(pg,&w3[c]));
            svint16_t xv;
            xv=svld1_s16(pg,&xq0[c]); d00=svdot_s64(d00,v0,xv); d01=svdot_s64(d01,v1,xv); d02=svdot_s64(d02,v2,xv); d03=svdot_s64(d03,v3,xv);
            xv=svld1_s16(pg,&xq1[c]); d10=svdot_s64(d10,v0,xv); d11=svdot_s64(d11,v1,xv); d12=svdot_s64(d12,v2,xv); d13=svdot_s64(d13,v3,xv);
            xv=svld1_s16(pg,&xq2[c]); d20=svdot_s64(d20,v0,xv); d21=svdot_s64(d21,v1,xv); d22=svdot_s64(d22,v2,xv); d23=svdot_s64(d23,v3,xv);
            xv=svld1_s16(pg,&xq3[c]); d30=svdot_s64(d30,v0,xv); d31=svdot_s64(d31,v1,xv); d32=svdot_s64(d32,v2,xv); d33=svdot_s64(d33,v3,xv);
        }
        double c0=s0[blk],c1=s1[blk],c2=s2[blk],c3=s3[blk];
        double g0=(double)xg0[blk],g1=(double)xg1[blk],g2=(double)xg2[blk],g3=(double)xg3[blk];
        b00+=c0*g0; b01+=c1*g0; b02+=c2*g0; b03+=c3*g0;
        b10+=c0*g1; b11+=c1*g1; b12+=c2*g1; b13+=c3*g1;
        b20+=c0*g2; b21+=c1*g2; b22+=c2*g2; b23+=c3*g2;
        b30+=c0*g3; b31+=c1*g3; b32+=c2*g3; b33+=c3*g3;
        f00=svmla_n_f64_x(pf,f00,svcvt_f64_s64_x(pf,d00),c0); f01=svmla_n_f64_x(pf,f01,svcvt_f64_s64_x(pf,d01),c1); f02=svmla_n_f64_x(pf,f02,svcvt_f64_s64_x(pf,d02),c2); f03=svmla_n_f64_x(pf,f03,svcvt_f64_s64_x(pf,d03),c3);
        f10=svmla_n_f64_x(pf,f10,svcvt_f64_s64_x(pf,d10),c0); f11=svmla_n_f64_x(pf,f11,svcvt_f64_s64_x(pf,d11),c1); f12=svmla_n_f64_x(pf,f12,svcvt_f64_s64_x(pf,d12),c2); f13=svmla_n_f64_x(pf,f13,svcvt_f64_s64_x(pf,d13),c3);
        f20=svmla_n_f64_x(pf,f20,svcvt_f64_s64_x(pf,d20),c0); f21=svmla_n_f64_x(pf,f21,svcvt_f64_s64_x(pf,d21),c1); f22=svmla_n_f64_x(pf,f22,svcvt_f64_s64_x(pf,d22),c2); f23=svmla_n_f64_x(pf,f23,svcvt_f64_s64_x(pf,d23),c3);
        f30=svmla_n_f64_x(pf,f30,svcvt_f64_s64_x(pf,d30),c0); f31=svmla_n_f64_x(pf,f31,svcvt_f64_s64_x(pf,d31),c1); f32=svmla_n_f64_x(pf,f32,svcvt_f64_s64_x(pf,d32),c2); f33=svmla_n_f64_x(pf,f33,svcvt_f64_s64_x(pf,d33),c3);
        b=bend;
    }
    a0[0]+=svaddv_f64(pf,f00)-128.0*b00; a0[1]+=svaddv_f64(pf,f01)-128.0*b01; a0[2]+=svaddv_f64(pf,f02)-128.0*b02; a0[3]+=svaddv_f64(pf,f03)-128.0*b03;
    a1[0]+=svaddv_f64(pf,f10)-128.0*b10; a1[1]+=svaddv_f64(pf,f11)-128.0*b11; a1[2]+=svaddv_f64(pf,f12)-128.0*b12; a1[3]+=svaddv_f64(pf,f13)-128.0*b13;
    a2[0]+=svaddv_f64(pf,f20)-128.0*b20; a2[1]+=svaddv_f64(pf,f21)-128.0*b21; a2[2]+=svaddv_f64(pf,f22)-128.0*b22; a2[3]+=svaddv_f64(pf,f23)-128.0*b23;
    a3[0]+=svaddv_f64(pf,f30)-128.0*b30; a3[1]+=svaddv_f64(pf,f31)-128.0*b31; a3[2]+=svaddv_f64(pf,f32)-128.0*b32; a3[3]+=svaddv_f64(pf,f33)-128.0*b33;
}

/* 4 weight rows x 2 tokens: the M=2 spec-verify block. One weight widen feeds both tokens'
 * sdot lanes (two per-token matvecs widen twice; the dup-lane 4x wastes half its sdots). */
static inline void glm5_int16sdot_4row_2x(float*restrict a0,float*restrict a1,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const float*s0,const float*s1,const float*s2,const float*s3,int gs,int col0,
        const int16_t*xq0,const int16_t*xq1,
        const int64_t*restrict xg0,const int64_t*restrict xg1,int kl){
    svbool_t pf=svptrue_b64(); int vh=(int)svcnth();
    svfloat64_t f00=svdup_f64(0),f01=svdup_f64(0),f02=svdup_f64(0),f03=svdup_f64(0);
    svfloat64_t f10=svdup_f64(0),f11=svdup_f64(0),f12=svdup_f64(0),f13=svdup_f64(0);
    double b00=0,b01=0,b02=0,b03=0, b10=0,b11=0,b12=0,b13=0;
    for(int b=0;b<kl;){
        int blk=(col0+b)/gs, bend=(blk+1)*gs-col0; if(bend>kl)bend=kl; int c=b;
        svint64_t d00=svdup_s64(0),d01=svdup_s64(0),d02=svdup_s64(0),d03=svdup_s64(0);
        svint64_t d10=svdup_s64(0),d11=svdup_s64(0),d12=svdup_s64(0),d13=svdup_s64(0);
        for(;c<bend;c+=vh){
            svbool_t pg=svwhilelt_b16((uint32_t)c,(uint32_t)bend);
            svint16_t v0=svreinterpret_s16_u16(svld1ub_u16(pg,&w0[c]));
            svint16_t v1=svreinterpret_s16_u16(svld1ub_u16(pg,&w1[c]));
            svint16_t v2=svreinterpret_s16_u16(svld1ub_u16(pg,&w2[c]));
            svint16_t v3=svreinterpret_s16_u16(svld1ub_u16(pg,&w3[c]));
            svint16_t xv;
            xv=svld1_s16(pg,&xq0[c]); d00=svdot_s64(d00,v0,xv); d01=svdot_s64(d01,v1,xv); d02=svdot_s64(d02,v2,xv); d03=svdot_s64(d03,v3,xv);
            xv=svld1_s16(pg,&xq1[c]); d10=svdot_s64(d10,v0,xv); d11=svdot_s64(d11,v1,xv); d12=svdot_s64(d12,v2,xv); d13=svdot_s64(d13,v3,xv);
        }
        double c0=s0[blk],c1=s1[blk],c2=s2[blk],c3=s3[blk];
        double g0=(double)xg0[blk],g1=(double)xg1[blk];
        b00+=c0*g0; b01+=c1*g0; b02+=c2*g0; b03+=c3*g0;
        b10+=c0*g1; b11+=c1*g1; b12+=c2*g1; b13+=c3*g1;
        f00=svmla_n_f64_x(pf,f00,svcvt_f64_s64_x(pf,d00),c0); f01=svmla_n_f64_x(pf,f01,svcvt_f64_s64_x(pf,d01),c1); f02=svmla_n_f64_x(pf,f02,svcvt_f64_s64_x(pf,d02),c2); f03=svmla_n_f64_x(pf,f03,svcvt_f64_s64_x(pf,d03),c3);
        f10=svmla_n_f64_x(pf,f10,svcvt_f64_s64_x(pf,d10),c0); f11=svmla_n_f64_x(pf,f11,svcvt_f64_s64_x(pf,d11),c1); f12=svmla_n_f64_x(pf,f12,svcvt_f64_s64_x(pf,d12),c2); f13=svmla_n_f64_x(pf,f13,svcvt_f64_s64_x(pf,d13),c3);
        b=bend;
    }
    a0[0]+=svaddv_f64(pf,f00)-128.0*b00; a0[1]+=svaddv_f64(pf,f01)-128.0*b01; a0[2]+=svaddv_f64(pf,f02)-128.0*b02; a0[3]+=svaddv_f64(pf,f03)-128.0*b03;
    a1[0]+=svaddv_f64(pf,f10)-128.0*b10; a1[1]+=svaddv_f64(pf,f11)-128.0*b11; a1[2]+=svaddv_f64(pf,f12)-128.0*b12; a1[3]+=svaddv_f64(pf,f13)-128.0*b13;
}

/* PREFILL int8 w8a8-SDOT GEMM micro-kernel: 4 weight rows x 5 token register block. Like the int16
 * kernel but int8xint8 svdot_s32 (64 MACs/instr = 2x the int16, 4x the w8a16 bf16-FMA) — the FASTEST
 * but LOSSY (activations rounded to int8). Weight offset-binary -> signed via eor 0x80 (no widen).
 * Per group: svdot_s32 into 20 int32 accumulators, fold per-row group scale into 20 f32. This is the
 * register-blocked version the shipped glm5_gemm_int8_sdot lacks (it re-reads weights per token). */
static inline void glm5_int8sdot_4row_5x(float*restrict a0,float*restrict a1,float*restrict a2,float*restrict a3,float*restrict a4,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const float*s0,const float*s1,const float*s2,const float*s3,int gs,int col0,
        const int8_t*xq0,const int8_t*xq1,const int8_t*xq2,const int8_t*xq3,const int8_t*xq4,int kl){
    svbool_t pf=svptrue_b32(); int vb=(int)svcntb();
    svfloat32_t f00=svdup_f32(0),f01=svdup_f32(0),f02=svdup_f32(0),f03=svdup_f32(0);
    svfloat32_t f10=svdup_f32(0),f11=svdup_f32(0),f12=svdup_f32(0),f13=svdup_f32(0);
    svfloat32_t f20=svdup_f32(0),f21=svdup_f32(0),f22=svdup_f32(0),f23=svdup_f32(0);
    svfloat32_t f30=svdup_f32(0),f31=svdup_f32(0),f32=svdup_f32(0),f33=svdup_f32(0);
    svfloat32_t f40=svdup_f32(0),f41=svdup_f32(0),f42=svdup_f32(0),f43=svdup_f32(0);
    for(int b=0;b<kl;){
        int blk=(col0+b)/gs, bend=(blk+1)*gs-col0; if(bend>kl)bend=kl; int c=b;
        svint32_t d00=svdup_s32(0),d01=svdup_s32(0),d02=svdup_s32(0),d03=svdup_s32(0);
        svint32_t d10=svdup_s32(0),d11=svdup_s32(0),d12=svdup_s32(0),d13=svdup_s32(0);
        svint32_t d20=svdup_s32(0),d21=svdup_s32(0),d22=svdup_s32(0),d23=svdup_s32(0);
        svint32_t d30=svdup_s32(0),d31=svdup_s32(0),d32=svdup_s32(0),d33=svdup_s32(0);
        svint32_t d40=svdup_s32(0),d41=svdup_s32(0),d42=svdup_s32(0),d43=svdup_s32(0);
        for(;c<bend;c+=vb){
            svbool_t pg=svwhilelt_b8((uint32_t)c,(uint32_t)bend);
            svint8_t v0=svreinterpret_s8_u8(sveor_n_u8_x(pg,svld1_u8(pg,&w0[c]),0x80));
            svint8_t v1=svreinterpret_s8_u8(sveor_n_u8_x(pg,svld1_u8(pg,&w1[c]),0x80));
            svint8_t v2=svreinterpret_s8_u8(sveor_n_u8_x(pg,svld1_u8(pg,&w2[c]),0x80));
            svint8_t v3=svreinterpret_s8_u8(sveor_n_u8_x(pg,svld1_u8(pg,&w3[c]),0x80));
            svint8_t xv;
            xv=svld1_s8(pg,&xq0[c]); d00=svdot_s32(d00,v0,xv); d01=svdot_s32(d01,v1,xv); d02=svdot_s32(d02,v2,xv); d03=svdot_s32(d03,v3,xv);
            xv=svld1_s8(pg,&xq1[c]); d10=svdot_s32(d10,v0,xv); d11=svdot_s32(d11,v1,xv); d12=svdot_s32(d12,v2,xv); d13=svdot_s32(d13,v3,xv);
            xv=svld1_s8(pg,&xq2[c]); d20=svdot_s32(d20,v0,xv); d21=svdot_s32(d21,v1,xv); d22=svdot_s32(d22,v2,xv); d23=svdot_s32(d23,v3,xv);
            xv=svld1_s8(pg,&xq3[c]); d30=svdot_s32(d30,v0,xv); d31=svdot_s32(d31,v1,xv); d32=svdot_s32(d32,v2,xv); d33=svdot_s32(d33,v3,xv);
            xv=svld1_s8(pg,&xq4[c]); d40=svdot_s32(d40,v0,xv); d41=svdot_s32(d41,v1,xv); d42=svdot_s32(d42,v2,xv); d43=svdot_s32(d43,v3,xv);
        }
        float c0=s0[blk],c1=s1[blk],c2=s2[blk],c3=s3[blk];
        f00=svmla_n_f32_x(pf,f00,svcvt_f32_s32_x(pf,d00),c0); f01=svmla_n_f32_x(pf,f01,svcvt_f32_s32_x(pf,d01),c1); f02=svmla_n_f32_x(pf,f02,svcvt_f32_s32_x(pf,d02),c2); f03=svmla_n_f32_x(pf,f03,svcvt_f32_s32_x(pf,d03),c3);
        f10=svmla_n_f32_x(pf,f10,svcvt_f32_s32_x(pf,d10),c0); f11=svmla_n_f32_x(pf,f11,svcvt_f32_s32_x(pf,d11),c1); f12=svmla_n_f32_x(pf,f12,svcvt_f32_s32_x(pf,d12),c2); f13=svmla_n_f32_x(pf,f13,svcvt_f32_s32_x(pf,d13),c3);
        f20=svmla_n_f32_x(pf,f20,svcvt_f32_s32_x(pf,d20),c0); f21=svmla_n_f32_x(pf,f21,svcvt_f32_s32_x(pf,d21),c1); f22=svmla_n_f32_x(pf,f22,svcvt_f32_s32_x(pf,d22),c2); f23=svmla_n_f32_x(pf,f23,svcvt_f32_s32_x(pf,d23),c3);
        f30=svmla_n_f32_x(pf,f30,svcvt_f32_s32_x(pf,d30),c0); f31=svmla_n_f32_x(pf,f31,svcvt_f32_s32_x(pf,d31),c1); f32=svmla_n_f32_x(pf,f32,svcvt_f32_s32_x(pf,d32),c2); f33=svmla_n_f32_x(pf,f33,svcvt_f32_s32_x(pf,d33),c3);
        f40=svmla_n_f32_x(pf,f40,svcvt_f32_s32_x(pf,d40),c0); f41=svmla_n_f32_x(pf,f41,svcvt_f32_s32_x(pf,d41),c1); f42=svmla_n_f32_x(pf,f42,svcvt_f32_s32_x(pf,d42),c2); f43=svmla_n_f32_x(pf,f43,svcvt_f32_s32_x(pf,d43),c3);
        b=bend;
    }
    a0[0]+=svaddv_f32(pf,f00); a0[1]+=svaddv_f32(pf,f01); a0[2]+=svaddv_f32(pf,f02); a0[3]+=svaddv_f32(pf,f03);
    a1[0]+=svaddv_f32(pf,f10); a1[1]+=svaddv_f32(pf,f11); a1[2]+=svaddv_f32(pf,f12); a1[3]+=svaddv_f32(pf,f13);
    a2[0]+=svaddv_f32(pf,f20); a2[1]+=svaddv_f32(pf,f21); a2[2]+=svaddv_f32(pf,f22); a2[3]+=svaddv_f32(pf,f23);
    a3[0]+=svaddv_f32(pf,f30); a3[1]+=svaddv_f32(pf,f31); a3[2]+=svaddv_f32(pf,f32); a3[3]+=svaddv_f32(pf,f33);
    a4[0]+=svaddv_f32(pf,f40); a4[1]+=svaddv_f32(pf,f41); a4[2]+=svaddv_f32(pf,f42); a4[3]+=svaddv_f32(pf,f43);
}
#endif
#endif

#endif /* GLM5_INT8_H */
