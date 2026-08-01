/* Microbenchmark + numeric check for fp8-e4m3 dequant strategies in the
 * block-scaled fp8 matvec (decode) and matmat (prefill) kernels.
 *   v0  gather  : current kernel -- svld1_gather from a 256-entry f32 LUT
 *   v1  bitadd  : (b&0x7f)<<20 + (120<<23), sign OR'd in.  Exact for normals,
 *                 denormals approximated as (1+m/8)*2^-7 (<2e-5 of block max).
 *   v2  bitmul  : (b&0x7f)<<20 reinterpreted * 2^120.  Exact incl. denormals.
 * Run: OMP_NUM_THREADS=1 ./fp8_dq_bench   (single core = kernel issue rate)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <arm_sve.h>

enum { BLK = 128, ROWS = 1024, COLS = 3072 };
static float lut[256];

static float ref_dequant(int b) {
    int s=(b>>7)&1, e=(b>>3)&0xf, m=b&0x7; float v;
    if (e==0)             v = (float)m * 0.001953125f;
    else if (e==15&&m==7) v = 0.0f;
    else                  v = ldexpf(1.0f + (float)m*0.125f, e-7);
    return s ? -v : v;
}
static float bf16f(uint16_t x){ uint32_t u=(uint32_t)x<<16; float f; memcpy(&f,&u,sizeof f); return f; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ---- v0: gather LUT (current) ---- */
static void mv_gather(float *y, const uint8_t *W, const uint16_t *sc, const float *x,
                      int rows, int cols) {
    int cblk = cols/BLK, VL=(int)svcntw(); svbool_t pt=svptrue_b32();
    #pragma omp parallel for schedule(static)
    for (int r=0;r<rows;r+=8) {
        const uint16_t *sr = sc + (size_t)(r/BLK)*cblk;
        const uint8_t *w0=W+(size_t)r*cols,*w1=w0+cols,*w2=w1+cols,*w3=w2+cols;
        const uint8_t *w4=w3+cols,*w5=w4+cols,*w6=w5+cols,*w7=w6+cols;
        float t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t7=0;
        for (int cb=0;cb<cblk;++cb) {
            int c0=cb*BLK;
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
            svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
            for (int c=c0;c<c0+BLK;c+=VL) {
                svfloat32_t xf=svld1_f32(pt,x+c);
                a0=svmla_f32_x(pt,a0,svld1_gather_u32index_f32(pt,lut,svld1ub_u32(pt,w0+c)),xf);
                a1=svmla_f32_x(pt,a1,svld1_gather_u32index_f32(pt,lut,svld1ub_u32(pt,w1+c)),xf);
                a2=svmla_f32_x(pt,a2,svld1_gather_u32index_f32(pt,lut,svld1ub_u32(pt,w2+c)),xf);
                a3=svmla_f32_x(pt,a3,svld1_gather_u32index_f32(pt,lut,svld1ub_u32(pt,w3+c)),xf);
                a4=svmla_f32_x(pt,a4,svld1_gather_u32index_f32(pt,lut,svld1ub_u32(pt,w4+c)),xf);
                a5=svmla_f32_x(pt,a5,svld1_gather_u32index_f32(pt,lut,svld1ub_u32(pt,w5+c)),xf);
                a6=svmla_f32_x(pt,a6,svld1_gather_u32index_f32(pt,lut,svld1ub_u32(pt,w6+c)),xf);
                a7=svmla_f32_x(pt,a7,svld1_gather_u32index_f32(pt,lut,svld1ub_u32(pt,w7+c)),xf);
            }
            float s=bf16f(sr[cb]);
            t0+=s*svaddv_f32(pt,a0); t1+=s*svaddv_f32(pt,a1); t2+=s*svaddv_f32(pt,a2); t3+=s*svaddv_f32(pt,a3);
            t4+=s*svaddv_f32(pt,a4); t5+=s*svaddv_f32(pt,a5); t6+=s*svaddv_f32(pt,a6); t7+=s*svaddv_f32(pt,a7);
        }
        y[r]=t0;y[r+1]=t1;y[r+2]=t2;y[r+3]=t3;y[r+4]=t4;y[r+5]=t5;y[r+6]=t6;y[r+7]=t7;
    }
}

/* ---- v1: integer bias add ---- */
#define DQ_ADD(p, ptr) \
    ({ svuint32_t _b = svld1ub_u32(p, (ptr)); \
       svuint32_t _n = svadd_n_u32_x(p, svlsl_n_u32_x(p, svand_n_u32_x(p,_b,0x7fu), 20), 0x3C000000u); \
       svreinterpret_f32_u32(svorr_u32_x(p, _n, svand_n_u32_x(p, svlsl_n_u32_x(p,_b,24), 0x80000000u))); })

/* ---- v2: reinterpret + multiply by 2^120 (exact, incl. subnormals) ---- */
#define DQ_MUL(p, ptr, k) \
    ({ svuint32_t _b = svld1ub_u32(p, (ptr)); \
       svfloat32_t _v = svreinterpret_f32_u32(svlsl_n_u32_x(p, svand_n_u32_x(p,_b,0x7fu), 20)); \
       svreinterpret_f32_u32(svorr_u32_x(p, svreinterpret_u32_f32(svmul_f32_x(p,_v,(k))), \
                                         svand_n_u32_x(p, svlsl_n_u32_x(p,_b,24), 0x80000000u))); })

#define MV_BIT(name, DQ)                                                              \
static void name(float *y, const uint8_t *W, const uint16_t *sc, const float *x,      \
                 int rows, int cols) {                                                \
    int cblk=cols/BLK, VL=(int)svcntw(); svbool_t pt=svptrue_b32();                    \
    svfloat32_t k=svdup_f32(ldexpf(1.0f,120)); (void)k;                                \
    _Pragma("omp parallel for schedule(static)")                                       \
    for (int r=0;r<rows;r+=8) {                                                        \
        const uint16_t *sr=sc+(size_t)(r/BLK)*cblk;                                    \
        const uint8_t *w0=W+(size_t)r*cols,*w1=w0+cols,*w2=w1+cols,*w3=w2+cols;        \
        const uint8_t *w4=w3+cols,*w5=w4+cols,*w6=w5+cols,*w7=w6+cols;                 \
        float t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t7=0;                                 \
        for (int cb=0;cb<cblk;++cb) {                                                  \
            int c0=cb*BLK;                                                             \
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);\
            svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);\
            for (int c=c0;c<c0+BLK;c+=VL) {                                            \
                svfloat32_t xf=svld1_f32(pt,x+c);                                       \
                a0=svmla_f32_x(pt,a0,DQ,xf); a1=svmla_f32_x(pt,a1,DQ,xf);               \
                a2=svmla_f32_x(pt,a2,DQ,xf); a3=svmla_f32_x(pt,a3,DQ,xf);               \
                a4=svmla_f32_x(pt,a4,DQ,xf); a5=svmla_f32_x(pt,a5,DQ,xf);               \
                a6=svmla_f32_x(pt,a6,DQ,xf); a7=svmla_f32_x(pt,a7,DQ,xf);               \
            }                                                                          \
            float s=bf16f(sr[cb]);                                                     \
            t0+=s*svaddv_f32(pt,a0); t1+=s*svaddv_f32(pt,a1);                           \
            t2+=s*svaddv_f32(pt,a2); t3+=s*svaddv_f32(pt,a3);                           \
            t4+=s*svaddv_f32(pt,a4); t5+=s*svaddv_f32(pt,a5);                           \
            t6+=s*svaddv_f32(pt,a6); t7+=s*svaddv_f32(pt,a7);                           \
        }                                                                              \
        y[r]=t0;y[r+1]=t1;y[r+2]=t2;y[r+3]=t3;y[r+4]=t4;y[r+5]=t5;y[r+6]=t6;y[r+7]=t7;  \
    }                                                                                  \
}
/* the DQ macros expand w0..w7 via a per-use pointer; instantiate manually below */
#undef MV_BIT

#define MVBODY(DQ8)                                                                    \
    int cblk=cols/BLK, VL=(int)svcntw(); svbool_t pt=svptrue_b32();                     \
    const svfloat32_t k=svdup_f32(ldexpf(1.0f,120)); (void)k;                           \
    _Pragma("omp parallel for schedule(static)")                                        \
    for (int r=0;r<rows;r+=8) {                                                         \
        const uint16_t *sr=sc+(size_t)(r/BLK)*cblk;                                     \
        const uint8_t *w0=W+(size_t)r*cols,*w1=w0+cols,*w2=w1+cols,*w3=w2+cols;         \
        const uint8_t *w4=w3+cols,*w5=w4+cols,*w6=w5+cols,*w7=w6+cols;                  \
        float t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t7=0;                                  \
        for (int cb=0;cb<cblk;++cb) {                                                   \
            int c0=cb*BLK;                                                              \
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);\
            svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);\
            for (int c=c0;c<c0+BLK;c+=VL) {                                             \
                svfloat32_t xf=svld1_f32(pt,x+c);                                        \
                a0=svmla_f32_x(pt,a0,DQ8(w0+c),xf); a1=svmla_f32_x(pt,a1,DQ8(w1+c),xf);  \
                a2=svmla_f32_x(pt,a2,DQ8(w2+c),xf); a3=svmla_f32_x(pt,a3,DQ8(w3+c),xf);  \
                a4=svmla_f32_x(pt,a4,DQ8(w4+c),xf); a5=svmla_f32_x(pt,a5,DQ8(w5+c),xf);  \
                a6=svmla_f32_x(pt,a6,DQ8(w6+c),xf); a7=svmla_f32_x(pt,a7,DQ8(w7+c),xf);  \
            }                                                                           \
            float s=bf16f(sr[cb]);                                                      \
            t0+=s*svaddv_f32(pt,a0); t1+=s*svaddv_f32(pt,a1);                            \
            t2+=s*svaddv_f32(pt,a2); t3+=s*svaddv_f32(pt,a3);                            \
            t4+=s*svaddv_f32(pt,a4); t5+=s*svaddv_f32(pt,a5);                            \
            t6+=s*svaddv_f32(pt,a6); t7+=s*svaddv_f32(pt,a7);                            \
        }                                                                               \
        y[r]=t0;y[r+1]=t1;y[r+2]=t2;y[r+3]=t3;y[r+4]=t4;y[r+5]=t5;y[r+6]=t6;y[r+7]=t7;   \
    }

#define DQA(ptr) DQ_ADD(pt,(ptr))
#define DQM(ptr) DQ_MUL(pt,(ptr),k)

static void mv_bitadd(float *y, const uint8_t *W, const uint16_t *sc, const float *x,
                      int rows, int cols) { MVBODY(DQA) }
static void mv_bitmul(float *y, const uint8_t *W, const uint16_t *sc, const float *x,
                      int rows, int cols) { MVBODY(DQM) }

/* ---- v3: fp8 -> int8 per-128-block at load, then the existing int8 kernel ----
 * Each 128-col block gets scale2 = (block bf16 scale)*max|w|/127; weights become
 * int8.  Decode then costs ld1sb+cvt+fmla (3 ops) on half the bytes. */
static void quant_i8blk(int8_t *Q, float *S, const uint8_t *W, const uint16_t *sc,
                        int rows, int cols) {
    int cblk=cols/BLK;
    #pragma omp parallel for schedule(static)
    for (int r=0;r<rows;++r) {
        const uint8_t *wr=W+(size_t)r*cols; const uint16_t *sr=sc+(size_t)(r/BLK)*cblk;
        for (int cb=0;cb<cblk;++cb) {
            float bs=bf16f(sr[cb]); const uint8_t *wb=wr+cb*BLK;
            float mx=0; for (int j=0;j<BLK;++j){ float a=fabsf(lut[wb[j]]); if(a>mx)mx=a; }
            float s = mx>0 ? bs*mx/127.0f : bs;
            float inv = mx>0 ? 127.0f/mx : 0.0f;
            S[(size_t)r*cblk+cb]=s;
            int8_t *qb=Q+(size_t)r*cols+cb*BLK;
            for (int j=0;j<BLK;++j){ int v=(int)lrintf(lut[wb[j]]*inv);
                qb[j]=(int8_t)(v<-127?-127:v>127?127:v); }
        }
    }
}
static void mv_i8blk(float *y, const int8_t *Q, const float *S, const float *x,
                     int rows, int cols) {
    int cblk=cols/BLK, VL=(int)svcntw(); svbool_t pt=svptrue_b32();
    #pragma omp parallel for schedule(static)
    for (int r=0;r<rows;r+=8) {
        const float *sr=S+(size_t)r*cblk;
        const int8_t *w0=Q+(size_t)r*cols,*w1=w0+cols,*w2=w1+cols,*w3=w2+cols;
        const int8_t *w4=w3+cols,*w5=w4+cols,*w6=w5+cols,*w7=w6+cols;
        float t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0,t7=0;
        for (int cb=0;cb<cblk;++cb) {
            int c0=cb*BLK;
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
            svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
            for (int c=c0;c<c0+BLK;c+=VL) {
                svfloat32_t xf=svld1_f32(pt,x+c);
                a0=svmla_f32_x(pt,a0,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w0+c)),xf);
                a1=svmla_f32_x(pt,a1,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w1+c)),xf);
                a2=svmla_f32_x(pt,a2,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w2+c)),xf);
                a3=svmla_f32_x(pt,a3,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w3+c)),xf);
                a4=svmla_f32_x(pt,a4,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w4+c)),xf);
                a5=svmla_f32_x(pt,a5,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w5+c)),xf);
                a6=svmla_f32_x(pt,a6,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w6+c)),xf);
                a7=svmla_f32_x(pt,a7,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w7+c)),xf);
            }
            t0+=sr[cb]*svaddv_f32(pt,a0);        t1+=sr[cblk+cb]*svaddv_f32(pt,a1);
            t2+=sr[2*cblk+cb]*svaddv_f32(pt,a2); t3+=sr[3*cblk+cb]*svaddv_f32(pt,a3);
            t4+=sr[4*cblk+cb]*svaddv_f32(pt,a4); t5+=sr[5*cblk+cb]*svaddv_f32(pt,a5);
            t6+=sr[6*cblk+cb]*svaddv_f32(pt,a6); t7+=sr[7*cblk+cb]*svaddv_f32(pt,a7);
        }
        y[r]=t0;y[r+1]=t1;y[r+2]=t2;y[r+3]=t3;y[r+4]=t4;y[r+5]=t5;y[r+6]=t6;y[r+7]=t7;
    }
}

/* ---- v4: i8blk with lane-wise block-scale accumulation ----
 * sum_cb s_cb * addv(a_cb) == addv( sum_cb s_cb * a_cb ), so keep a running
 * VECTOR accumulator and fold each block's scale with a vector*scalar FMLA,
 * doing ONE cross-lane reduction per row instead of one per 128-col block. */
static void mv_i8blk2(float *y, const int8_t *Q, const float *S, const float *x,
                      int rows, int cols) {
    int cblk = cols/BLK, VL=(int)svcntw(); svbool_t pt=svptrue_b32();
    #pragma omp parallel for schedule(static)
    for (int r=0;r<rows;r+=8) {
        const float *sr=S+(size_t)r*cblk;
        const int8_t *w0=Q+(size_t)r*cols,*w1=w0+cols,*w2=w1+cols,*w3=w2+cols;
        const int8_t *w4=w3+cols,*w5=w4+cols,*w6=w5+cols,*w7=w6+cols;
        svfloat32_t T0=svdup_f32(0),T1=svdup_f32(0),T2=svdup_f32(0),T3=svdup_f32(0);
        svfloat32_t T4=svdup_f32(0),T5=svdup_f32(0),T6=svdup_f32(0),T7=svdup_f32(0);
        for (int cb=0;cb<cblk;++cb) {
            int c0=cb*BLK;
            svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
            svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
            for (int c=c0;c<c0+BLK;c+=VL) {
                svfloat32_t xf=svld1_f32(pt,x+c);
                a0=svmla_f32_x(pt,a0,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w0+c)),xf);
                a1=svmla_f32_x(pt,a1,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w1+c)),xf);
                a2=svmla_f32_x(pt,a2,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w2+c)),xf);
                a3=svmla_f32_x(pt,a3,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w3+c)),xf);
                a4=svmla_f32_x(pt,a4,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w4+c)),xf);
                a5=svmla_f32_x(pt,a5,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w5+c)),xf);
                a6=svmla_f32_x(pt,a6,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w6+c)),xf);
                a7=svmla_f32_x(pt,a7,svcvt_f32_s32_x(pt,svld1sb_s32(pt,w7+c)),xf);
            }
            T0=svmla_n_f32_x(pt,T0,a0,sr[cb]);          T1=svmla_n_f32_x(pt,T1,a1,sr[cblk+cb]);
            T2=svmla_n_f32_x(pt,T2,a2,sr[2*cblk+cb]);   T3=svmla_n_f32_x(pt,T3,a3,sr[3*cblk+cb]);
            T4=svmla_n_f32_x(pt,T4,a4,sr[4*cblk+cb]);   T5=svmla_n_f32_x(pt,T5,a5,sr[5*cblk+cb]);
            T6=svmla_n_f32_x(pt,T6,a6,sr[6*cblk+cb]);   T7=svmla_n_f32_x(pt,T7,a7,sr[7*cblk+cb]);
        }
        y[r]=svaddv_f32(pt,T0);y[r+1]=svaddv_f32(pt,T1);y[r+2]=svaddv_f32(pt,T2);y[r+3]=svaddv_f32(pt,T3);
        y[r+4]=svaddv_f32(pt,T4);y[r+5]=svaddv_f32(pt,T5);y[r+6]=svaddv_f32(pt,T6);y[r+7]=svaddv_f32(pt,T7);
    }
}

/* scalar reference */
static void mv_ref(float *y, const uint8_t *W, const uint16_t *sc, const float *x,
                   int rows, int cols) {
    int cblk=cols/BLK;
    for (int r=0;r<rows;++r) {
        const uint8_t *wr=W+(size_t)r*cols; const uint16_t *sr=sc+(size_t)(r/BLK)*cblk;
        double acc=0;
        for (int cb=0;cb<cblk;++cb) { double part=0;
            for (int j=0;j<BLK;++j) part += (double)ref_dequant(wr[cb*BLK+j])*x[cb*BLK+j];
            acc += (double)bf16f(sr[cb])*part; }
        y[r]=(float)acc;
    }
}

static double relerr(const float *a, const float *b, int n) {
    double num=0, den=0;
    for (int i=0;i<n;++i){ double d=(double)a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; }
    return sqrt(num/(den>0?den:1));
}

int main(int argc, char **argv) {
    int iters = argc>1 ? atoi(argv[1]) : 30;
    for (int b=0;b<256;++b) lut[b]=ref_dequant(b);

    size_t nw=(size_t)ROWS*COLS;
    uint8_t *W = aligned_alloc(256, nw);
    uint16_t *sc = aligned_alloc(256, (size_t)(ROWS/BLK)*(COLS/BLK)*sizeof(uint16_t));
    float *x = aligned_alloc(256, COLS*sizeof(float));
    float *y0=aligned_alloc(256,ROWS*4), *y1=aligned_alloc(256,ROWS*4),
          *y2=aligned_alloc(256,ROWS*4), *yr=aligned_alloc(256,ROWS*4);
    srand(1234);
    /* Realistic weights: Gaussian per 128x128 block, block-scaled so max -> 448,
     * then encoded to the nearest e4m3 code (what the checkpoint actually holds).
     * (Uniform-over-code-space bytes would give each block a 2^17 dynamic range,
     * which no real weight matrix has, and would unfairly penalize int8.) */
    int cbn=COLS/BLK;
    float *Worig = aligned_alloc(256, nw*sizeof(float));
    for (int rb=0; rb<ROWS/BLK; ++rb) for (int cb=0; cb<cbn; ++cb) {
        float blk[BLK*BLK]; float mx=0;
        for (int i=0;i<BLK*BLK;++i){
            double u1=(rand()+1.0)/((double)RAND_MAX+2), u2=(rand()+1.0)/((double)RAND_MAX+2);
            float g=(float)(sqrt(-2*log(u1))*cos(2*M_PI*u2))*0.02f;
            blk[i]=g; if (fabsf(g)>mx) mx=fabsf(g);
        }
        float bs = mx/448.0f;                       /* the checkpoint's block scale */
        uint32_t bu; memcpy(&bu,&bs,4); uint16_t bsb=(uint16_t)(bu>>16);
        sc[rb*cbn+cb]=bsb; float bsq=bf16f(bsb);
        for (int i=0;i<BLK*BLK;++i) {
            float v = blk[i]/bsq;                   /* to e4m3 domain */
            int best=0; float bd=INFINITY;          /* nearest-code encode */
            for (int b=0;b<256;++b){ if((b&0x7f)==0x7f) continue;
                float d=fabsf(lut[b]-v); if(d<bd){bd=d;best=b;} }
            W[(size_t)(rb*BLK+i/BLK)*COLS + cb*BLK + (i%BLK)] = (uint8_t)best;
            Worig[(size_t)(rb*BLK+i/BLK)*COLS + cb*BLK + (i%BLK)] = blk[i];
        }
    }
    for (int i=0;i<COLS;++i) x[i]=(float)((rand()%2000)-1000)/1000.0f;

    /* ground truth: dot with the ORIGINAL f32 weights (pre-fp8-quantization) */
    float *ytrue = aligned_alloc(256, ROWS*4);
    for (int r=0;r<ROWS;++r){ double a=0; const float *wr=Worig+(size_t)r*COLS;
        for (int c=0;c<COLS;++c) a += (double)wr[c]*x[c]; ytrue[r]=(float)a; }
    mv_ref(yr,W,sc,x,ROWS,COLS);
    mv_gather(y0,W,sc,x,ROWS,COLS);
    mv_bitadd(y1,W,sc,x,ROWS,COLS);
    mv_bitmul(y2,W,sc,x,ROWS,COLS);
    printf("relerr vs f64 scalar ref:  gather=%.3e  bitadd=%.3e  bitmul=%.3e\n",
           relerr(y0,yr,ROWS), relerr(y1,yr,ROWS), relerr(y2,yr,ROWS));
    printf("relerr vs gather kernel :  bitadd=%.3e  bitmul=%.3e\n",
           relerr(y1,y0,ROWS), relerr(y2,y0,ROWS));

    /* int8-per-block variant */
    int8_t *Q = aligned_alloc(256, nw);
    float  *S = aligned_alloc(256, (size_t)ROWS*(COLS/BLK)*sizeof(float));
    float  *y3 = aligned_alloc(256, ROWS*4);
    quant_i8blk(Q,S,W,sc,ROWS,COLS);
    mv_i8blk(y3,Q,S,x,ROWS,COLS);
    { float *y4=aligned_alloc(256,ROWS*4); mv_i8blk2(y4,Q,S,x,ROWS,COLS);
      printf("relerr i8blk2 vs i8blk = %.3e\n", relerr(y4,y3,ROWS)); free(y4); }
    printf("relerr i8blk vs f64 ref = %.3e   (vs gather kernel = %.3e)\n",
           relerr(y3,yr,ROWS), relerr(y3,y0,ROWS));
    printf("--- vs ORIGINAL f32 weights (what actually matters) ---\n");
    printf("  e4m3 (exact kernel) = %.3e\n  e4m3 (bitadd)       = %.3e\n  fp8->int8 per-blk   = %.3e\n",
           relerr(yr,ytrue,ROWS), relerr(y1,ytrue,ROWS), relerr(y3,ytrue,ROWS));

    double macs = (double)ROWS*COLS*iters;
    struct { const char *n; void (*f)(float*,const uint8_t*,const uint16_t*,const float*,int,int); float *y; } v[] = {
        {"gather", mv_gather, y0}, {"bitadd", mv_bitadd, y1}, {"bitmul", mv_bitmul, y2} };
    for (int i=0;i<3;++i) {
        v[i].f(v[i].y,W,sc,x,ROWS,COLS);            /* warm */
        double t0=now();
        for (int it=0;it<iters;++it) v[i].f(v[i].y,W,sc,x,ROWS,COLS);
        double dt=now()-t0;
        printf("%-7s  %8.2f ms/iter   %6.2f GMAC/s   %5.2f MAC/cycle@2.0GHz\n",
               v[i].n, dt/iters*1e3, macs/dt/1e9, macs/dt/2.0e9);
    }
    mv_i8blk(y3,Q,S,x,ROWS,COLS);
    { double t0=now();
      for (int it=0;it<iters;++it) mv_i8blk(y3,Q,S,x,ROWS,COLS);
      double dt=now()-t0;
      printf("%-7s  %8.2f ms/iter   %6.2f GMAC/s   %5.2f MAC/cycle@2.0GHz\n",
             "i8blk", dt/iters*1e3, macs/dt/1e9, macs/dt/2.0e9); }
    mv_i8blk2(y3,Q,S,x,ROWS,COLS);
    { double t0=now();
      for (int it=0;it<iters;++it) mv_i8blk2(y3,Q,S,x,ROWS,COLS);
      double dt=now()-t0;
      printf("%-7s  %8.2f ms/iter   %6.2f GMAC/s   %5.2f MAC/cycle@2.0GHz\n",
             "i8blk2", dt/iters*1e3, macs/dt/1e9, macs/dt/2.0e9); }
    return 0;
}
