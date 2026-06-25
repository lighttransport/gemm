/* gemm_q8v2_prefill.c - per-block (Q8v2) int8 prefill GEMM for Gemma-4 Q4_0 FFN.
 *
 * Unlike gemm_q4_prefill.c (per-TENSOR int8 scale, ~9% err, raw-int32 accumulate),
 * this keeps Q4_0's per-32-block weight scale EXACTLY: the int8 weight is the
 * centered nibble (q-8) in [-8,7] (lossless), with a per-block fp32 scale d_w[n][b].
 * Activations are quantized per-32-block per-row to int8 (the only error source).
 *
 *   C[m,n] = sum_b  d_a[m][b] * d_w[n][b] * ( sum_{k in block b} qa[m,k] * (qw[n,k]-8) )
 *
 * The microkernel accumulates int32 over each 32-wide K block, then scvtf + scales
 * by d_w (per-col vector) and d_a (per-row scalar) into fp32 accumulators (Q8v2).
 * Expected err ~1% vs fp32-dequant reference (activation int8 only).
 *
 * This is the MODEL-COUPLED bench (real Q4_0 tensor accuracy + multi-thread node
 * throughput). The standalone, profileable microbench is ../gemma4-kernels/q8v2_profile.c.
 * The kernel lives canonically in ../gemma4-kernels/kernel_q8v2_3x4.S.
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *       -I../../common gemm_q8v2_prefill.c ../gemma4-kernels/kernel_q8v2_3x4.S -lm -o gemm_q8v2_prefill
 *   NUMA_DISTRIBUTE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
 *       ./gemm_q8v2_prefill ~/models/gemma4/31b-qat/gemma-4-31B_q4_0-it.gguf [tensor]
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <arm_sve.h>
#include <omp.h>
#include <sys/mman.h>
#include <time.h>
#include <math.h>

/* cntvct integer timer (FP timers get clobbered by SVE-heavy code paths) */
static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
/* hand-asm Q8v2 microkernel (kernel_q8v2_3x4.S) */
void kernel_q8v2_3x4(const int8_t*aq,const float*ad,const int8_t*bq,const float*bd,long nb,float*C,long ldc);
void kernel_q8v2_3x4_arow(const int8_t*aq,const float*ad,const int8_t*bq,const float*bd,long nb,float*C,long ldc);
static int find_tensor(gguf_context*g,const char*n){ for(uint64_t i=0;i<g->n_tensors;i++){ const char*s=gguf_tensor_name(g,(int)i); if(s&&!strcmp(s,n))return (int)i;} return -1; }
static void*xmap(size_t n){ void*p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); if(p==MAP_FAILED){perror("mmap");exit(1);} return p; }

#define MR 3      /* token rows per microkernel tile (3x4 fits 32 SVE regs: 12 fp32 acc + 12 int32 partial) */
#define NR 64     /* weight output cols per tile (4 SVE vectors of 16) */
#define BLK 32    /* Q4_0 block size (K granularity of scales) */

/* Q8v2 microkernel: C[3 rows, 64 cols] += sum_b da[r,b]*dw[c,b]*<qa[r,b],qw[c,b]>.
 * n-vec inner: 3 A-row broadcasts reused across all 4 B-vecs (4x fewer ld1rw than
 * v-outer). 12 fp32 accumulators persist across K; 12 int32 partials per block. */
static inline void q8v2_tile_3x4(const int8_t*aq,const float*ad,
                                 const int8_t*bq,const float*bd,int nb,
                                 float*C,int N,int mb,int Mlim){
    svbool_t pg=svptrue_b8(), pw=svptrue_b32();
    svfloat32_t f00=svdup_n_f32(0),f01=svdup_n_f32(0),f02=svdup_n_f32(0),f03=svdup_n_f32(0);
    svfloat32_t f10=svdup_n_f32(0),f11=svdup_n_f32(0),f12=svdup_n_f32(0),f13=svdup_n_f32(0);
    svfloat32_t f20=svdup_n_f32(0),f21=svdup_n_f32(0),f22=svdup_n_f32(0),f23=svdup_n_f32(0);
    for(int b=0;b<nb;b++){
        svint32_t p00=svdup_n_s32(0),p01=svdup_n_s32(0),p02=svdup_n_s32(0),p03=svdup_n_s32(0);
        svint32_t p10=svdup_n_s32(0),p11=svdup_n_s32(0),p12=svdup_n_s32(0),p13=svdup_n_s32(0);
        svint32_t p20=svdup_n_s32(0),p21=svdup_n_s32(0),p22=svdup_n_s32(0),p23=svdup_n_s32(0);
        const int8_t*bp=bq+(size_t)b*8*4*64;       /* 8 kgroups x 4 vecs x 64B */
        const int8_t*ap=aq+(size_t)(b*MR)*BLK;
        for(int g3=0;g3<8;g3++){
            const int8_t*a=ap+g3*4;
            svint8_t a0=svreinterpret_s8_s32(svdup_n_s32(*(const int32_t*)(a+0*BLK)));
            svint8_t a1=svreinterpret_s8_s32(svdup_n_s32(*(const int32_t*)(a+1*BLK)));
            svint8_t a2=svreinterpret_s8_s32(svdup_n_s32(*(const int32_t*)(a+2*BLK)));
            const int8_t*bg=bp+(size_t)g3*4*64;
            svint8_t b0=svld1_s8(pg,bg+0*64); svint8_t b1=svld1_s8(pg,bg+1*64);
            svint8_t b2=svld1_s8(pg,bg+2*64); svint8_t b3=svld1_s8(pg,bg+3*64);
            p00=svdot_s32(p00,a0,b0); p01=svdot_s32(p01,a0,b1); p02=svdot_s32(p02,a0,b2); p03=svdot_s32(p03,a0,b3);
            p10=svdot_s32(p10,a1,b0); p11=svdot_s32(p11,a1,b1); p12=svdot_s32(p12,a1,b2); p13=svdot_s32(p13,a1,b3);
            p20=svdot_s32(p20,a2,b0); p21=svdot_s32(p21,a2,b1); p22=svdot_s32(p22,a2,b2); p23=svdot_s32(p23,a2,b3);
        }
        svfloat32_t w0=svld1_f32(pw,bd+b*NR+0*16),w1=svld1_f32(pw,bd+b*NR+1*16);
        svfloat32_t w2=svld1_f32(pw,bd+b*NR+2*16),w3=svld1_f32(pw,bd+b*NR+3*16);
        float da0=ad[b*MR+0],da1=ad[b*MR+1],da2=ad[b*MR+2];
        f00=svmla_n_f32_x(pw,f00,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p00),w0),da0);
        f01=svmla_n_f32_x(pw,f01,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p01),w1),da0);
        f02=svmla_n_f32_x(pw,f02,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p02),w2),da0);
        f03=svmla_n_f32_x(pw,f03,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p03),w3),da0);
        f10=svmla_n_f32_x(pw,f10,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p10),w0),da1);
        f11=svmla_n_f32_x(pw,f11,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p11),w1),da1);
        f12=svmla_n_f32_x(pw,f12,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p12),w2),da1);
        f13=svmla_n_f32_x(pw,f13,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p13),w3),da1);
        f20=svmla_n_f32_x(pw,f20,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p20),w0),da2);
        f21=svmla_n_f32_x(pw,f21,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p21),w1),da2);
        f22=svmla_n_f32_x(pw,f22,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p22),w2),da2);
        f23=svmla_n_f32_x(pw,f23,svmul_f32_x(pw,svcvt_f32_s32_x(pw,p23),w3),da2);
    }
    if(mb+0<Mlim){ float*c=C+(size_t)(mb+0)*N; svst1_f32(pw,c+0*16,f00);svst1_f32(pw,c+1*16,f01);svst1_f32(pw,c+2*16,f02);svst1_f32(pw,c+3*16,f03);}
    if(mb+1<Mlim){ float*c=C+(size_t)(mb+1)*N; svst1_f32(pw,c+0*16,f10);svst1_f32(pw,c+1*16,f11);svst1_f32(pw,c+2*16,f12);svst1_f32(pw,c+3*16,f13);}
    if(mb+2<Mlim){ float*c=C+(size_t)(mb+2)*N; svst1_f32(pw,c+0*16,f20);svst1_f32(pw,c+1*16,f21);svst1_f32(pw,c+2*16,f22);svst1_f32(pw,c+3*16,f23);}
}

/* ---- Packed layouts ----
 * Aq:  [MTn][nb][MR][BLK]   int8  centered/quantized activations, row-major per tile
 * Ad:  [MTn][nb][MR]        f32   per-row per-block activation scale
 * Bq:  [NTn][nb][8][4][16][4] int8  weights, pack_B-style (kgroup,vec,col,byte) per block
 * Bd:  [NTn][nb][4][16]     f32   per-col per-block weight scale (4 vecs x 16 cols)
 */

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [tensor]\n",argv[0]); return 1; }
    const char*tname=argc>2?argv[2]:"blk.0.ffn_gate.weight";
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    double peak=512.0*nt;  /* int8 SDOT GIOPS */

    gguf_context*g=gguf_open(argv[1],1); if(!g){fprintf(stderr,"open fail\n");return 1;}
    int idx=find_tensor(g,tname); if(idx<0){fprintf(stderr,"no tensor %s\n",tname);return 1;}
    gguf_tensor_info*ti=&g->tensors[idx];
    int K=(int)ti->dims[0], N=(int)ti->dims[1], nb=K/BLK;
    size_t rb=(size_t)nb*sizeof(block_q4_0);
    if(K%256||N%NR){ fprintf(stderr,"need K%%256==0,N%%64==0 (K=%d N=%d)\n",K,N); return 1; }
    int NTn=N/NR;
    fprintf(stderr,"[q8v2-gemm] %s N=%d K=%d nb=%d Ntiles=%d\n",tname,N,K,nb,NTn);
    block_q4_0*W=(block_q4_0*)xmap((size_t)N*rb); memcpy(W,gguf_tensor_data(g,idx),(size_t)N*rb); gguf_close(g);

    /* Pack weights: centered nibbles (q-8) -> int8 in pack_B layout, per block.
     * Bq tile stride = nb*8*4*16*4 = nb*NR*BLK bytes.  Bd tile stride = nb*NR floats. */
    size_t bq_tile=(size_t)nb*NR*BLK; int8_t*Bq=(int8_t*)xmap((size_t)NTn*bq_tile);
    size_t bd_tile=(size_t)nb*NR;     float*Bd=(float*)xmap((size_t)NTn*bd_tile*sizeof(float));
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int n0=0;n0<NTn;n0++){
        int8_t*bq=Bq+(size_t)n0*bq_tile; float*bd=Bd+(size_t)n0*bd_tile;
        for(int b=0;b<nb;b++){
            /* scales: 4 vecs x 16 cols */
            for(int vec=0;vec<4;vec++) for(int col=0;col<16;col++){
                int n=n0*NR+vec*16+col;
                const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)n*rb);
                bd[b*NR+vec*16+col]=ggml_fp16_to_fp32(row[b].d);
            }
            /* quants: kgroup g(0..7) x vec(0..3) x col(0..15) x byte(0..3) */
            for(int g3=0;g3<8;g3++) for(int vec=0;vec<4;vec++) for(int col=0;col<16;col++){
                int n=n0*NR+vec*16+col;
                const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)n*rb);
                for(int kk=0;kk<4;kk++){
                    int k=g3*4+kk;                 /* 0..31 within block */
                    int q = (k<16)? (row[b].qs[k]&0xf) : (row[b].qs[k-16]>>4);
                    *bq++ = (int8_t)(q-8);
                }
            }
        }
    }
    munmap(W,(size_t)N*rb);

    int Ms[]={6,48,96,192,384}; int nM=5;
    printf("\n=== %s N=%d K=%d  Q8v2 per-block int8 GEMM @ %d thr (peak %.0f GIOPS) ===\n",tname,N,K,nt,peak);
    printf("  %-6s %9s %10s %8s %10s\n","M(tok)","ms","GIOPS","%peak","relL2");
    for(int mi=0;mi<nM;mi++){ int M=Ms[mi]; int MTn=(M+MR-1)/MR;
        /* synthetic activations A[M,K] f32 (escaping, deterministic) */
        float*Af=(float*)xmap((size_t)M*K*sizeof(float));
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int i=0;i<M;i++) for(int k=0;k<K;k++) Af[(size_t)i*K+k]=0.05f*(float)(((i*7+k*3)&0x3f)-32);
        /* pack/quant activations per block per row */
        size_t aq_tile=(size_t)nb*MR*BLK; int8_t*Aq=(int8_t*)xmap((size_t)MTn*aq_tile);
        size_t ad_tile=(size_t)nb*MR;     float*Ad=(float*)xmap((size_t)MTn*ad_tile*sizeof(float));
        int arow = getenv("Q8V2_AROW")?1:0;  /* per-row act scale (factors da out of block loop) */
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int m0=0;m0<MTn;m0++){
            int8_t*aq=Aq+(size_t)m0*aq_tile; float*ad=Ad+(size_t)m0*ad_tile;
            for(int r=0;r<MR;r++){
                int m=m0*MR+r;
                float rmax=0; if(arow&&m<M) for(int k=0;k<K;k++){ float a=fabsf(Af[(size_t)m*K+k]); if(a>rmax)rmax=a; }
                for(int b=0;b<nb;b++){
                    float amax=rmax;
                    if(!arow){ amax=0; if(m<M) for(int k=0;k<BLK;k++){ float a=fabsf(Af[(size_t)m*K+b*BLK+k]); if(a>amax)amax=a; } }
                    float d=amax>0?amax/127.0f:0.0f, inv=d>0?1.0f/d:0.0f;
                    ad[b*MR+r]=d;
                    for(int k=0;k<BLK;k++){ int q=0; if(m<M){ q=(int)lrintf(Af[(size_t)m*K+b*BLK+k]*inv); if(q>127)q=127; if(q<-127)q=-127; } aq[(b*MR+r)*BLK+k]=(int8_t)q; }
                }
            }
        }
        int use_asm = getenv("Q8V2_INTRIN")?0:1;
        float*C=(float*)xmap((size_t)MTn*MR*N*sizeof(float));  /* padded rows for asm 3-row store */

        #define GEMM_PASS() \
          _Pragma("omp parallel for num_threads(nt) schedule(static) collapse(2)") \
          for(int n0=0;n0<NTn;n0++) for(int m0=0;m0<MTn;m0++){ \
            if(use_asm&&arow) kernel_q8v2_3x4_arow(Aq+(size_t)m0*aq_tile, Ad+(size_t)m0*ad_tile, \
                          Bq+(size_t)n0*bq_tile, Bd+(size_t)n0*bd_tile, nb, \
                          C+(size_t)(m0*MR)*N+n0*NR, (long)N*4); \
            else if(use_asm) kernel_q8v2_3x4(Aq+(size_t)m0*aq_tile, Ad+(size_t)m0*ad_tile, \
                          Bq+(size_t)n0*bq_tile, Bd+(size_t)n0*bd_tile, nb, \
                          C+(size_t)(m0*MR)*N+n0*NR, (long)N*4); \
            else q8v2_tile_3x4(Aq+(size_t)m0*aq_tile, Ad+(size_t)m0*ad_tile, \
                          Bq+(size_t)n0*bq_tile, Bd+(size_t)n0*bd_tile, nb, \
                          C+n0*NR, N, m0*MR, M); \
          }
        GEMM_PASS();  /* warm */

        /* accuracy vs fp32 dequant reference (a few cols of a few rows) */
        double num=0,den=0; int rows_chk=(M<8?M:8);
        for(int m=0;m<rows_chk;m++) for(int c=0;c<N;c+=257){
            const block_q4_0*row=(const block_q4_0*)0; (void)row; double ref=0;
            for(int b=0;b<nb;b++){ float dw=Bd[ (size_t)(c/NR)*bd_tile + b*NR + ((c%NR)/16)*16 + (c%16) ]; (void)dw; }
            /* recompute ref directly from Af and weight nibble scale via Bd/Bq tiles */
            for(int b=0;b<nb;b++){
                int n0=c/NR, vcol=c%NR, vec=vcol/16, col=vcol%16;
                float dw=Bd[(size_t)n0*bd_tile + b*NR + vec*16 + col];
                const int8_t*bp=Bq+(size_t)n0*bq_tile + ((size_t)b*8*4 + 0*4 + vec)*64; (void)bp;
                for(int k=0;k<BLK;k++){
                    int g3=k/4, kk=k%4;
                    int qw=Bq[(size_t)n0*bq_tile + ((size_t)b*8*4 + (size_t)g3*4 + vec)*64 + col*4 + kk];
                    ref += (double)Af[(size_t)m*K+b*BLK+k] * (double)qw * (double)dw;
                }
            }
            double got=C[(size_t)m*N+c]; double e=got-ref; num+=e*e; den+=ref*ref;
        }
        double rel=den>0?sqrt(num/den):0;

        int reps=(M*(long)N*K>4L<<30)?3:8;
        volatile double sink=0; double freq=(double)rdfreq();
        volatile uint64_t c0=rdcyc();
        for(int rep=0;rep<reps;rep++){ GEMM_PASS(); sink+=C[((size_t)rep*48271)%((size_t)M*N)]; }
        volatile uint64_t c1=rdcyc();
        double per=(double)(c1-c0)/freq/reps; (void)sink;
        double gi=2.0*(double)M*N*K/per/1e9;
        printf("  %-6d %9.3f %10.1f %7.0f%% %10.4f\n",M,per*1000,gi,gi/peak*100,rel);
        #undef GEMM_PASS
        munmap(Af,(size_t)M*K*sizeof(float)); munmap(Aq,(size_t)MTn*aq_tile);
        munmap(Ad,(size_t)MTn*ad_tile*sizeof(float)); munmap(C,(size_t)MTn*MR*N*sizeof(float));
    }
    return 0;
}
