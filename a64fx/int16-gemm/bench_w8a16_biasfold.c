/* bench_w8a16_biasfold.c — prove the "bias-fold" widening trick for the GLM-5.2 w8a16
 * int16-SDOT matvec on A64FX. Offset-binary weight byte w_u8 in [0,255] (= signed+128),
 * activation quantized to int16, per-group (gs) f32 scale, per-row.
 *
 * The dot for one group is  sum_c (w_u8[c]-128)*xq[c].  Three widening strategies:
 *
 *   sub      : svld1ub_u16 (u8->u16 zero-extend) + svsub_n_s16(...,128) -> signed s16,
 *              then svdot_s64.  == the CURRENT kernel. 2 SVE-ALU ops/vec (SUB + SDOT).
 *
 *   biasfold : svld1ub_u16 ONLY (values 0..255 are positive s16), svdot_s64 gives
 *              sum(w_u8*xq); subtract 128*sum(xq) per group ONCE at reduce time. The
 *              per-group sum(xq) is precomputed once per activation vector and reused
 *              across every weight row. 1 SVE-ALU op/vec (SDOT). Integer analog of the
 *              bf16->fp32 p_odd "no-shift" widen.
 *
 *   unpk     : full 64B svld1_u8 + uunpklo/uunpkhi -> two s16 vectors from ONE load
 *              (halves the load count), also bias-folded. x contiguous (lo=xq[0:32],
 *              hi=xq[32:64]).
 *
 * All validated against an f64 reference (relL2 must be ~0 for all three).
 *
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          bench_w8a16_biasfold.c -lm -o bench_w8a16_biasfold
 * run:   OMP_NUM_THREADS=48 ./bench_w8a16_biasfold [cols] [rows] [gs] [reps]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <arm_sve.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

/* ---- current kernel: ld1ub + sub #128 per vector, per-group f64 scale fold ---- */
static inline float row_sub(const uint8_t*w,const float*s,int gs,const int16_t*xq,int cols){
    svbool_t pf=svptrue_b64(); int vh=(int)svcnth(); double acc=0;
    for(int b=0;b<cols;){
        int blk=b/gs, bend=(blk+1)*gs; if(bend>cols)bend=cols; int c=b;
        svint64_t d=svdup_s64(0);
        for(;c<bend;c+=vh){
            svbool_t pg=svwhilelt_b16((uint32_t)c,(uint32_t)bend);
            svint16_t wv=svsub_n_s16_x(pg,svreinterpret_s16_u16(svld1ub_u16(pg,&w[c])),128);
            svint16_t xv=svld1_s16(pg,&xq[c]);
            d=svdot_s64(d,wv,xv);
        }
        acc+=(double)svaddv_s64(pf,d)*(double)s[blk];
        b=bend;
    }
    return (float)acc;
}

/* ---- bias-fold: ld1ub only, correct with precomputed per-group sum(xq) ---- */
static inline float row_biasfold(const uint8_t*w,const float*s,int gs,const int16_t*xq,
                                 const int64_t*xgsum,int cols){
    svbool_t pf=svptrue_b64(); int vh=(int)svcnth(); double acc=0;
    for(int b=0;b<cols;){
        int blk=b/gs, bend=(blk+1)*gs; if(bend>cols)bend=cols; int c=b;
        svint64_t d=svdup_s64(0);
        for(;c<bend;c+=vh){
            svbool_t pg=svwhilelt_b16((uint32_t)c,(uint32_t)bend);
            svint16_t wv=svreinterpret_s16_u16(svld1ub_u16(pg,&w[c])); /* 0..255, positive s16 */
            svint16_t xv=svld1_s16(pg,&xq[c]);
            d=svdot_s64(d,wv,xv);
        }
        long raw=svaddv_s64(pf,d);
        acc+=(double)(raw - 128*xgsum[blk])*(double)s[blk];
        b=bend;
    }
    return (float)acc;
}

/* ---- unpk: one 64B load per 64 elems, uunpklo/hi widen, bias-folded ---- */
static inline float row_unpk(const uint8_t*w,const float*s,int gs,const int16_t*xq,
                             const int64_t*xgsum,int cols){
    svbool_t pf=svptrue_b64(); int vb=(int)svcntb(),vh=(int)svcnth(); double acc=0;
    for(int b=0;b<cols;){
        int blk=b/gs, bend=(blk+1)*gs; if(bend>cols)bend=cols; int c=b;
        svint64_t d=svdup_s64(0);
        /* full 64-byte chunks: one ld1b, uunpklo+uunpkhi -> two s16 vecs */
        for(;c+vb<=bend;c+=vb){
            svuint8_t raw=svld1_u8(svptrue_b8(),&w[c]);
            svint16_t wlo=svreinterpret_s16_u16(svunpklo_u16(raw));
            svint16_t whi=svreinterpret_s16_u16(svunpkhi_u16(raw));
            svint16_t xlo=svld1_s16(svptrue_b16(),&xq[c]);
            svint16_t xhi=svld1_s16(svptrue_b16(),&xq[c+vh]);
            d=svdot_s64(d,wlo,xlo);
            d=svdot_s64(d,whi,xhi);
        }
        /* half/tail with ld1ub */
        for(;c<bend;c+=vh){
            svbool_t pg=svwhilelt_b16((uint32_t)c,(uint32_t)bend);
            svint16_t wv=svreinterpret_s16_u16(svld1ub_u16(pg,&w[c]));
            svint16_t xv=svld1_s16(pg,&xq[c]);
            d=svdot_s64(d,wv,xv);
        }
        long raw=svaddv_s64(pf,d);
        acc+=(double)(raw - 128*xgsum[blk])*(double)s[blk];
        b=bend;
    }
    return (float)acc;
}

#define MV(NAME,CALL) \
static void NAME(float*y,const uint8_t*W,const float*S,int gs,const int16_t*xq,\
                 const int64_t*xgsum,int rows,int cols,int sb){ \
    (void)xgsum; \
    _Pragma("omp parallel for schedule(static)") \
    for(int r=0;r<rows;r++){ const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb; y[r]=CALL; } }
MV(mv_sub,      row_sub(w,s,gs,xq,cols))
MV(mv_biasfold, row_biasfold(w,s,gs,xq,xgsum,cols))
MV(mv_unpk,     row_unpk(w,s,gs,xq,xgsum,cols))

static double rel(const float*a,const float*b,int n){ double nu=0,de=0;
    for(int i=0;i<n;i++){ double e=a[i]-b[i]; nu+=e*e; de+=(double)b[i]*b[i]; } return sqrt(nu/(de+1e-30)); }

int main(int argc,char**argv){
    int cols=argc>1?atoi(argv[1]):5120, rows=argc>2?atoi(argv[2]):8192,
        gs=argc>3?atoi(argv[3]):128, reps=argc>4?atoi(argv[4]):50;
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    int sb=(cols+gs-1)/gs;
    printf("w8a16 int16-SDOT matvec  cols=%d rows=%d gs=%d threads=%d (W=%.1fMB)\n",
           cols,rows,gs,nt,(double)rows*cols/1e6);

    uint8_t*W=(uint8_t*)malloc((size_t)rows*cols);
    float  *S=(float*)malloc((size_t)rows*sb*4);
    int16_t*xq=(int16_t*)malloc((size_t)cols*2);
    int64_t*xgsum=(int64_t*)malloc((size_t)sb*8);
    float  *Yr=(float*)malloc((size_t)rows*4),*Y=(float*)malloc((size_t)rows*4);

    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++){
        for(int k=0;k<cols;k++){ size_t i=(size_t)r*cols+k; W[i]=(uint8_t)((i*131u+7u)&0xff); }
        for(int g=0;g<sb;g++) S[(size_t)r*sb+g]=0.0008f*(float)(((r*7+g)%13)-6);
    }
    for(int k=0;k<cols;k++) xq[k]=(int16_t)(((k*97+11)%2001)-1000);
    /* precompute per-group activation sums (once per activation vector) */
    for(int g=0;g<sb;g++){ long s=0; int e=(g+1)*gs; if(e>cols)e=cols; for(int c=g*gs;c<e;c++) s+=xq[c]; xgsum[g]=s; }

    /* f64 reference */
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++){ double acc=0;
        for(int b=0;b<cols;){ int blk=b/gs,e=(blk+1)*gs; if(e>cols)e=cols; long a=0;
            for(int c=b;c<e;c++) a+=(long)((int)W[(size_t)r*cols+c]-128)*(int)xq[c];
            acc+=(double)a*(double)S[(size_t)r*sb+blk]; b=e; }
        Yr[r]=(float)acc; }

    double flop=2.0*rows*cols, bytes=(double)rows*cols + cols*2; /* W u8 + xq */
    #define RUN(NAME,FN) do{ FN(Y,W,S,gs,xq,xgsum,rows,cols,sb); double e=rel(Y,Yr,rows); \
        double t=wall(); for(int r=0;r<reps;r++) FN(Y,W,S,gs,xq,xgsum,rows,cols,sb); t=(wall()-t)/reps; \
        printf("  %-9s %8.3f ms  %8.1f GOPS  %7.1f GB/s  relL2=%.2e\n",NAME,t*1e3,flop/t/1e9,bytes/t/1e9,e); }while(0)
    RUN("sub",     mv_sub);
    RUN("biasfold",mv_biasfold);
    RUN("unpk",    mv_unpk);
    free(W);free(S);free(xq);free(xgsum);free(Yr);free(Y);
    return 0;
}
