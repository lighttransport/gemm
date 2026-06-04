/*
 * DS4F FP8 dense matvec — fast-decode prototyping bench.
 *
 * The Stage-0 bench (ds4f_kernels_bench.c) is L1-resident and single-thread, so
 * it measures the *decode ALU ceiling* (gather-bound, 16.5 GB/s). The MODEL runs
 * the FP8 dense matvecs cold from HBM across 48 threads. This bench reproduces
 * that regime: a large (>L2) FP8 weight matrix streamed once per "token" by 48
 * OpenMP threads, so the measured GB/s is what the model actually sees — and it
 * tells us whether a faster FP8 decode helps at 48T-HBM or whether we're already
 * memory-bound (in which case BF16-predequant, not faster decode, is the lever).
 *
 * Candidates (all bit-validated vs the LUT reference):
 *   A gather   : current matvec_fp8e4m3_8row (svld1ub_u32 strided load + 256-LUT gather)
 *   B pk_gather: packed svld1_u8 (1 load/64B) + svunpk -> 4 u32 + 4 gathers
 *   C pk_arith : packed svld1_u8 + svunpk + branchless E4M3->f32 arithmetic (no gather)
 *
 * Build (native A64FX, OpenMP):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *       -I../../common -o /local/ds4f_fp8fast ds4f_fp8_fast_bench.c -lm
 *   OMP_NUM_THREADS=48 /local/ds4f_fp8fast
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <arm_sve.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "ggml_dequant.h"   /* matvec_fp8e4m3_8row (candidate A) + helpers */

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ __volatile__("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ __volatile__("mrs %0, cntfrq_el0":"=r"(v)); return v; }
/* Enable flush-to-zero (FPCR.FZ bit24) on THIS thread: denormal operands/results
 * are flushed to 0, dodging A64FX's denormal microcode penalty. Acceptable for
 * the harness (values meaningless; the magic-decode's tiny subnormals -> 0). */
static inline void set_ftz(void){
    uint64_t fpcr; __asm__ __volatile__("mrs %0, fpcr":"=r"(fpcr));
    fpcr |= (1u<<24); __asm__ __volatile__("msr fpcr, %0"::"r"(fpcr));
}

static uint64_t sm_state;
static inline uint64_t sm_next(void){
    uint64_t z=(sm_state+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31);
}
static inline uint8_t  rand_byte(void){ return (uint8_t)(sm_next()&0xff); }
static inline float    rand_unit(void){ return (float)((int64_t)(sm_next()&0xffffff)-0x800000)/(float)0x800000; }

/* ============================ candidate B ============================ *
 * Packed byte load + svunpk -> four u32 lanes + 4 gathers. Replaces the 4
 * per-16 strided svld1ub_u32 loads of candidate A with ONE contiguous 64B
 * svld1_u8 + cheap unpacks, keeping the (expensive) 256-entry LUT gather. */
static inline void matvec_fp8e4m3_8row_pkg(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const uint32_t *lut, const float *x, int K) {
    svbool_t pg = svptrue_b32();
    svbool_t pb = svptrue_b8();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    int vlb = (int)svcntb();   /* 64 bytes / iter */
    for (int c0 = 0; c0 < K; c0 += 128) {
        float s = ggml_e8m0_to_fp32(escale[c0 >> 7]);
        svfloat32_t vs = svdup_f32(s);
        for (int c = c0; c < c0 + 128; c += vlb) {
            svfloat32_t vx0 = svmul_x(pg, svld1(pg,&x[c]),    vs);
            svfloat32_t vx1 = svmul_x(pg, svld1(pg,&x[c+16]), vs);
            svfloat32_t vx2 = svmul_x(pg, svld1(pg,&x[c+32]), vs);
            svfloat32_t vx3 = svmul_x(pg, svld1(pg,&x[c+48]), vs);
            #define ROW_PKG(W, ACC) do {                                        \
                svuint8_t  b   = svld1_u8(pb, (W) + c);                          \
                svuint16_t l16 = svunpklo_u16(b), h16 = svunpkhi_u16(b);         \
                svuint32_t i0=svunpklo_u32(l16), i1=svunpkhi_u32(l16);           \
                svuint32_t i2=svunpklo_u32(h16), i3=svunpkhi_u32(h16);           \
                svfloat32_t f0=svreinterpret_f32_u32(svld1_gather_u32index_u32(pg,lut,i0)); \
                svfloat32_t f1=svreinterpret_f32_u32(svld1_gather_u32index_u32(pg,lut,i1)); \
                svfloat32_t f2=svreinterpret_f32_u32(svld1_gather_u32index_u32(pg,lut,i2)); \
                svfloat32_t f3=svreinterpret_f32_u32(svld1_gather_u32index_u32(pg,lut,i3)); \
                ACC = svmla_x(pg, ACC, f0, vx0);                                 \
                ACC = svmla_x(pg, ACC, f1, vx1);                                 \
                ACC = svmla_x(pg, ACC, f2, vx2);                                 \
                ACC = svmla_x(pg, ACC, f3, vx3);                                 \
            } while (0)
            ROW_PKG(w0,a0); ROW_PKG(w1,a1); ROW_PKG(w2,a2); ROW_PKG(w3,a3);
            ROW_PKG(w4,a4); ROW_PKG(w5,a5); ROW_PKG(w6,a6); ROW_PKG(w7,a7);
            #undef ROW_PKG
        }
    }
    dst[0]=svaddv(pg,a0); dst[1]=svaddv(pg,a1); dst[2]=svaddv(pg,a2); dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4); dst[5]=svaddv(pg,a5); dst[6]=svaddv(pg,a6); dst[7]=svaddv(pg,a7);
}

/* ============================ candidate C ============================ *
 * Packed byte load + svunpk + BRANCHLESS E4M3->f32 arithmetic, no gather.
 * E4M3 (bias 7, no inf; exp==15 we map to a large normal, harmless for the
 * harness and never produced by the NaN-free synthetic fill). Decode of byte b:
 *   normal  (exp!=0): bits = sign | (exp+120)<<23 | mant<<20
 *   subnorm (exp==0): mant * 2^-9  (with sign), 0 if mant==0
 * Inline helper decodes one u32 lane-vector of bytes to f32. */
static inline svfloat32_t fp8_decode_u32(svbool_t pg, svuint32_t b) {
    svuint32_t sign = svlsl_n_u32_x(pg, svand_n_u32_x(pg, b, 0x80u), 24);
    svuint32_t exp  = svand_n_u32_x(pg, svlsr_n_u32_x(pg, b, 3), 0xFu);
    svuint32_t mant = svand_n_u32_x(pg, b, 0x7u);
    /* normal path */
    svuint32_t nb = svorr_u32_x(pg,
                        svorr_u32_x(pg, sign,
                            svlsl_n_u32_x(pg, svadd_n_u32_x(pg, exp, 120u), 23)),
                        svlsl_n_u32_x(pg, mant, 20));
    svfloat32_t nrm = svreinterpret_f32_u32(nb);
    /* subnormal path: mant * 2^-9, re-sign */
    svfloat32_t sub = svmul_n_f32_x(pg, svcvt_f32_u32_x(pg, mant), 1.0f/512.0f);
    sub = svreinterpret_f32_u32(svorr_u32_x(pg, sign, svreinterpret_u32_f32(sub)));
    svbool_t is_sub = svcmpeq_n_u32(pg, exp, 0);
    return svsel_f32(is_sub, sub, nrm);
}
static inline void matvec_fp8e4m3_8row_pka(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const float *x, int K) {
    svbool_t pg = svptrue_b32();
    svbool_t pb = svptrue_b8();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    int vlb = (int)svcntb();
    for (int c0 = 0; c0 < K; c0 += 128) {
        float s = ggml_e8m0_to_fp32(escale[c0 >> 7]);
        svfloat32_t vs = svdup_f32(s);
        for (int c = c0; c < c0 + 128; c += vlb) {
            svfloat32_t vx0 = svmul_x(pg, svld1(pg,&x[c]),    vs);
            svfloat32_t vx1 = svmul_x(pg, svld1(pg,&x[c+16]), vs);
            svfloat32_t vx2 = svmul_x(pg, svld1(pg,&x[c+32]), vs);
            svfloat32_t vx3 = svmul_x(pg, svld1(pg,&x[c+48]), vs);
            #define ROW_PKA(W, ACC) do {                                        \
                svuint8_t  b   = svld1_u8(pb, (W) + c);                          \
                svuint16_t l16 = svunpklo_u16(b), h16 = svunpkhi_u16(b);         \
                svfloat32_t f0=fp8_decode_u32(pg, svunpklo_u32(l16));            \
                svfloat32_t f1=fp8_decode_u32(pg, svunpkhi_u32(l16));            \
                svfloat32_t f2=fp8_decode_u32(pg, svunpklo_u32(h16));            \
                svfloat32_t f3=fp8_decode_u32(pg, svunpkhi_u32(h16));            \
                ACC = svmla_x(pg, ACC, f0, vx0);                                 \
                ACC = svmla_x(pg, ACC, f1, vx1);                                 \
                ACC = svmla_x(pg, ACC, f2, vx2);                                 \
                ACC = svmla_x(pg, ACC, f3, vx3);                                 \
            } while (0)
            ROW_PKA(w0,a0); ROW_PKA(w1,a1); ROW_PKA(w2,a2); ROW_PKA(w3,a3);
            ROW_PKA(w4,a4); ROW_PKA(w5,a5); ROW_PKA(w6,a6); ROW_PKA(w7,a7);
            #undef ROW_PKA
        }
    }
    dst[0]=svaddv(pg,a0); dst[1]=svaddv(pg,a1); dst[2]=svaddv(pg,a2); dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4); dst[5]=svaddv(pg,a5); dst[6]=svaddv(pg,a6); dst[7]=svaddv(pg,a7);
}

/* ============================ candidate D ============================ *
 * Packed byte load + DENORMAL MAGIC-MULTIPLY decode. The cheapest faithful
 * E4M3->f32: build a tiny (possibly subnormal) f32 from the 7 low bits, then
 * one fmul by 2^120 renormalizes BOTH normals and subnormals in hardware (A64FX
 * has no denormal penalty) — no gather, no select, no cvt. ~6 ops/lane.
 *   u = (b&0x7F)<<20  -> exp@[26:23], mant@[22:20]; reinterpret*2^120; re-sign.
 * exp==15 maps to a finite large normal (<=480) not NaN — desirable here. */
static inline svfloat32_t fp8_decode_magic_u32(svbool_t pg, svuint32_t b) {
    svuint32_t sign = svlsl_n_u32_x(pg, svand_n_u32_x(pg, b, 0x80u), 24);
    svuint32_t mag  = svlsl_n_u32_x(pg, svand_n_u32_x(pg, b, 0x7Fu), 20);
    svfloat32_t f   = svmul_n_f32_x(pg, svreinterpret_f32_u32(mag), 0x1.0p+120f);
    return svreinterpret_f32_u32(svorr_u32_x(pg, sign, svreinterpret_u32_f32(f)));
}
static inline void matvec_fp8e4m3_8row_pkm(float *dst,
        const uint8_t *w0, const uint8_t *w1, const uint8_t *w2, const uint8_t *w3,
        const uint8_t *w4, const uint8_t *w5, const uint8_t *w6, const uint8_t *w7,
        const uint8_t *escale, const float *x, int K) {
    svbool_t pg = svptrue_b32();
    svbool_t pb = svptrue_b8();
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    int vlb = (int)svcntb();
    for (int c0 = 0; c0 < K; c0 += 128) {
        float s = ggml_e8m0_to_fp32(escale[c0 >> 7]);
        svfloat32_t vs = svdup_f32(s);
        for (int c = c0; c < c0 + 128; c += vlb) {
            svfloat32_t vx0 = svmul_x(pg, svld1(pg,&x[c]),    vs);
            svfloat32_t vx1 = svmul_x(pg, svld1(pg,&x[c+16]), vs);
            svfloat32_t vx2 = svmul_x(pg, svld1(pg,&x[c+32]), vs);
            svfloat32_t vx3 = svmul_x(pg, svld1(pg,&x[c+48]), vs);
            #define ROW_PKM(W, ACC) do {                                        \
                svuint8_t  b   = svld1_u8(pb, (W) + c);                          \
                svuint16_t l16 = svunpklo_u16(b), h16 = svunpkhi_u16(b);         \
                svfloat32_t f0=fp8_decode_magic_u32(pg, svunpklo_u32(l16));      \
                svfloat32_t f1=fp8_decode_magic_u32(pg, svunpkhi_u32(l16));      \
                svfloat32_t f2=fp8_decode_magic_u32(pg, svunpklo_u32(h16));      \
                svfloat32_t f3=fp8_decode_magic_u32(pg, svunpkhi_u32(h16));      \
                ACC = svmla_x(pg, ACC, f0, vx0);                                 \
                ACC = svmla_x(pg, ACC, f1, vx1);                                 \
                ACC = svmla_x(pg, ACC, f2, vx2);                                 \
                ACC = svmla_x(pg, ACC, f3, vx3);                                 \
            } while (0)
            ROW_PKM(w0,a0); ROW_PKM(w1,a1); ROW_PKM(w2,a2); ROW_PKM(w3,a3);
            ROW_PKM(w4,a4); ROW_PKM(w5,a5); ROW_PKM(w6,a6); ROW_PKM(w7,a7);
            #undef ROW_PKM
        }
    }
    dst[0]=svaddv(pg,a0); dst[1]=svaddv(pg,a1); dst[2]=svaddv(pg,a2); dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4); dst[5]=svaddv(pg,a5); dst[6]=svaddv(pg,a6); dst[7]=svaddv(pg,a7);
}

/* scalar f32 reference dot (lane-order agnostic; double acc) */
static float fp8_dot_dbl(const uint8_t *w, const uint8_t *escale, const float *x, int K){
    double acc=0.0;
    for (int c=0;c<K;c++){ uint32_t bb=ds4f_fp8_e4m3_to_fp32_bits(w[c]); float f; memcpy(&f,&bb,4);
        acc += (double)(f*ggml_e8m0_to_fp32(escale[c>>7]))*(double)x[c]; }
    return (float)acc;
}

int main(int argc, char **argv){
    int K     = (argc>1)?atoi(argv[1]):4096;     /* contraction dim */
    int NROW  = (argc>2)?atoi(argv[2]):16384;    /* rows (must be mult of 8) */
    int ITERS = (argc>3)?atoi(argv[3]):60;       /* "tokens" */
    if (K%128||NROW%8){ fprintf(stderr,"K%%128 and NROW%%8 required\n"); return 1; }
    sm_state = 0xD5F0F00D00000001ULL;
    double freq=(double)rdfreq();
    int nthr=1;
#ifdef _OPENMP
    #pragma omp parallel
    { nthr = omp_get_num_threads(); }
#endif
    int nblk = K/128, ngrp = NROW/8;
    int ftz = getenv("DS4F_FTZ") ? atoi(getenv("DS4F_FTZ")) : 0;
    if (ftz) {
#ifdef _OPENMP
        #pragma omp parallel
#endif
        set_ftz();
    }
    double wbytes = (double)NROW*K;                 /* FP8 weight bytes / token */
    printf("DS4F FP8 fast-decode bench  K=%d NROW=%d (%.0f MB weights) iters=%d threads=%d ftz=%d\n",
           K, NROW, wbytes/1e6, ITERS, nthr, ftz);

    uint32_t lut[256]; ds4f_init_fp8_e4m3_lut(lut);
    uint8_t  *W  = (uint8_t*)aligned_alloc(256,(size_t)NROW*K);
    uint16_t *WB = (uint16_t*)aligned_alloc(256,(size_t)NROW*K*sizeof(uint16_t)); /* bf16 predequant control */
    uint8_t  *ES = (uint8_t*)aligned_alloc(256,(size_t)ngrp*nblk);
    float    *X  = (float*)aligned_alloc(256,(size_t)K*sizeof(float));
    float    *Da = (float*)aligned_alloc(256,(size_t)NROW*sizeof(float));
    float    *Db = (float*)aligned_alloc(256,(size_t)NROW*sizeof(float));
    float    *Dc = (float*)aligned_alloc(256,(size_t)NROW*sizeof(float));
    if(!W||!WB||!ES||!X||!Da||!Db||!Dc){ fprintf(stderr,"alloc fail\n"); return 1; }
    for (int i=0;i<K;i++) X[i]=rand_unit();
    for (size_t i=0;i<(size_t)ngrp*nblk;i++) ES[i]=(uint8_t)(125+(rand_byte()%5));
    for (size_t i=0;i<(size_t)NROW*K;i++){ uint8_t b=rand_byte(); if((b&0x78)==0x78) b&=~0x08; W[i]=b; }
    /* bf16 predequant: fold the block e8m0 scale into the weight (as the model's
     * DS4F_FP8_BF16=1 path does), so matvec_bf16_8row needs no per-block scale. */
    for (int g=0; g<ngrp; g++) for (int r=0;r<8;r++){
        const uint8_t *wr = W + ((size_t)g*8+r)*K; uint16_t *wb = WB + ((size_t)g*8+r)*K;
        for (int c=0;c<K;c++){ uint32_t bb=ds4f_fp8_e4m3_to_fp32_bits(wr[c]); float f; memcpy(&f,&bb,4);
            f *= ggml_e8m0_to_fp32(ES[(size_t)g*nblk + (c>>7)]);
            uint32_t u; memcpy(&u,&f,4); wb[c]=(uint16_t)(u>>16); }
    }

    /* ---- correctness: candidates B,C vs A vs scalar (first group) ---- */
    {
        const uint8_t *wr[8]; for(int r=0;r<8;r++) wr[r]=W+(size_t)r*K;
        const uint8_t *es=ES;
        float da[8],db[8],dc[8],dd[8],ref[8];
        for(int r=0;r<8;r++) ref[r]=fp8_dot_dbl(wr[r],es,X,K);
        matvec_fp8e4m3_8row    (da,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],es,lut,X,K);
        matvec_fp8e4m3_8row_pkg(db,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],es,lut,X,K);
        matvec_fp8e4m3_8row_pka(dc,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],es,X,K);
        matvec_fp8e4m3_8row_pkm(dd,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],es,X,K);
        double ea=0,eb=0,ec=0,ed=0;
        for(int r=0;r<8;r++){
            double den=fabs(ref[r])+1e-6;
            ea=fmax(ea,fabs(da[r]-ref[r])/den);
            eb=fmax(eb,fabs(db[r]-ref[r])/den);
            ec=fmax(ec,fabs(dc[r]-ref[r])/den);
            ed=fmax(ed,fabs(dd[r]-ref[r])/den);
        }
        printf("[correct] A(gather)=%.2e B(pk_gather)=%.2e C(pk_arith)=%.2e D(pk_magic)=%.2e vs scalar -> %s\n",
               ea,eb,ec,ed,(ea<2e-5&&eb<2e-5&&ec<2e-5&&ed<2e-5)?"OK":"FAIL");
    }

    /* ---- streaming throughput (48T HBM, cold each token) ---- */
    #define BENCH(NAME, DST, CALL) do {                                            \
        volatile double sink=0;                                                    \
        for(int w=0;w<3;w++){ _Pragma("omp parallel for schedule(static)")         \
            for(int g=0;g<ngrp;g++){ const uint8_t *wr[8]; for(int r=0;r<8;r++) wr[r]=W+((size_t)g*8+r)*K; \
                const uint8_t *es=ES+(size_t)g*nblk; CALL; } }                      \
        uint64_t t0=rdcyc();                                                       \
        for(int it=0;it<ITERS;it++){ _Pragma("omp parallel for schedule(static)")  \
            for(int g=0;g<ngrp;g++){ const uint8_t *wr[8]; for(int r=0;r<8;r++) wr[r]=W+((size_t)g*8+r)*K; \
                const uint8_t *es=ES+(size_t)g*nblk; CALL; } }                      \
        uint64_t t1=rdcyc(); double sec=(double)(t1-t0)/freq;                       \
        for(int g=0;g<8;g++) sink+=DST[g];                                         \
        printf("[%-9s] %6.1f GB/s  %7.3f ms/token  (%.0f Mmac/s, sink=%.3g)\n",    \
            NAME, wbytes*ITERS/sec/1e9, sec/ITERS*1e3,                             \
            (double)NROW*K*ITERS/sec/1e6, (double)sink);                           \
    } while(0)

    BENCH("A gather",    Da, (matvec_fp8e4m3_8row    (Da+g*8,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],es,lut,X,K)));
    BENCH("B pk_gather", Db, (matvec_fp8e4m3_8row_pkg(Db+g*8,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],es,lut,X,K)));
    BENCH("C pk_arith",  Dc, (matvec_fp8e4m3_8row_pka(Dc+g*8,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],es,X,K)));
    BENCH("D pk_magic",  Dc, (matvec_fp8e4m3_8row_pkm(Dc+g*8,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],es,X,K)));

    /* bf16 control: streams 2x the bytes but no decode — the predequant path.
     * Compare ms/token (wall-clock) directly with the FP8 candidates above.
     * GB/s here is over bf16 bytes (2*NROW*K); ms/token is the apples-to-apples. */
    {
        volatile double sink=0;
        for(int w=0;w<3;w++){ _Pragma("omp parallel for schedule(static)")
            for(int g=0;g<ngrp;g++){ const uint16_t *wr[8]; for(int r=0;r<8;r++) wr[r]=WB+((size_t)g*8+r)*K;
                matvec_bf16_8row(Db+g*8,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],X,K); } }
        uint64_t t0=rdcyc();
        for(int it=0;it<ITERS;it++){ _Pragma("omp parallel for schedule(static)")
            for(int g=0;g<ngrp;g++){ const uint16_t *wr[8]; for(int r=0;r<8;r++) wr[r]=WB+((size_t)g*8+r)*K;
                matvec_bf16_8row(Db+g*8,wr[0],wr[1],wr[2],wr[3],wr[4],wr[5],wr[6],wr[7],X,K); } }
        uint64_t t1=rdcyc(); double sec=(double)(t1-t0)/freq;
        for(int g=0;g<8;g++) sink+=Db[g];
        printf("[%-9s] %6.1f GB/s  %7.3f ms/token  (%.0f Mmac/s, sink=%.3g)  [2 B/elem]\n",
            "E bf16pre", 2.0*wbytes*ITERS/sec/1e9, sec/ITERS*1e3,
            (double)NROW*K*ITERS/sec/1e6, (double)sink);
    }

    free(W);free(WB);free(ES);free(X);free(Da);free(Db);free(Dc);
    return 0;
}
