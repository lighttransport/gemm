/* layer_opt.c - single-tensor kernel optimization + accuracy on REAL weights.
 *
 * Memory-bounded: extracts ONE weight matrix from the GGUF (~65 MB Q4_0 /
 * ~115 MB int8), never the whole model. Dequants it to /local int8, then:
 *   - reference  = exact fp32 matvec from the Q4_0 weights (the true values)
 *   - int8 SDOT  = the fast path; measure perf AND error vs the fp32 reference
 * Use NUMA_DISTRIBUTE=1 so gguf_open is a LAZY mmap (touch only this tensor;
 * no MAP_POPULATE OOM).
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -D_GNU_SOURCE -I../../common layer_opt.c -lm -o layer_opt
 * Run:   NUMA_DISTRIBUTE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
 *          ./layer_opt model.gguf [tensor] [/local/w.i8]
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int find_tensor(gguf_context*g,const char*n){ for(uint64_t i=0;i<g->n_tensors;i++){ const char*s=gguf_tensor_name(g,(int)i); if(s&&!strcmp(s,n))return (int)i;} return -1; }

/* int8 8-row matvec (8 independent accumulators). */
static inline void mv8(int32_t acc[8], const int8_t *w0,int nbp,const int8_t *xi8){
    svbool_t pg=svptrue_b8();
    svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0),a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
    size_t rs=(size_t)nbp*64; const int8_t*w1=w0+rs,*w2=w0+2*rs,*w3=w0+3*rs,*w4=w0+4*rs,*w5=w0+5*rs,*w6=w0+6*rs,*w7=w0+7*rs;
    for(int p=0;p<nbp;p++){ svint8_t xv=svld1_s8(pg,xi8+(size_t)p*64);
        a0=svdot_s32(a0,svld1_s8(pg,w0+(size_t)p*64),xv); a1=svdot_s32(a1,svld1_s8(pg,w1+(size_t)p*64),xv);
        a2=svdot_s32(a2,svld1_s8(pg,w2+(size_t)p*64),xv); a3=svdot_s32(a3,svld1_s8(pg,w3+(size_t)p*64),xv);
        a4=svdot_s32(a4,svld1_s8(pg,w4+(size_t)p*64),xv); a5=svdot_s32(a5,svld1_s8(pg,w5+(size_t)p*64),xv);
        a6=svdot_s32(a6,svld1_s8(pg,w6+(size_t)p*64),xv); a7=svdot_s32(a7,svld1_s8(pg,w7+(size_t)p*64),xv); }
    svbool_t p3=svptrue_b32();
    acc[0]=svaddv_s32(p3,a0);acc[1]=svaddv_s32(p3,a1);acc[2]=svaddv_s32(p3,a2);acc[3]=svaddv_s32(p3,a3);
    acc[4]=svaddv_s32(p3,a4);acc[5]=svaddv_s32(p3,a5);acc[6]=svaddv_s32(p3,a6);acc[7]=svaddv_s32(p3,a7);
}
/* Q8_0-style per-block matvec: int8 weights (v-8, lossless) + per-block fp d.
 * 8 rows, per 32-block: SDOT(32) -> int32 -> svaddv -> *d_b -> fp accumulate.
 * Accurate (~1%) but pays a per-block reduction. wq8[r] at base+r*cols int8;
 * d8[r] at d8base+r*nb. */
static inline void q8_8row(float out[8], const int8_t*wq8,const float*d8,int nb,int cols,const int8_t*xi8){
    svbool_t pg32=svwhilelt_b8((uint32_t)0,(uint32_t)32), p3=svptrue_b32();
    float ac[8]={0,0,0,0,0,0,0,0};
    size_t rs=(size_t)cols; const int8_t*w[8]; const float*d[8];
    for(int r=0;r<8;r++){ w[r]=wq8+(size_t)r*rs; d[r]=d8+(size_t)r*nb; }
    for(int b=0;b<nb;b++){ svint8_t xb=svld1_s8(pg32,xi8+(size_t)b*32);
        for(int r=0;r<8;r++){ svint32_t s=svdot_s32(svdup_s32(0),svld1_s8(pg32,w[r]+(size_t)b*32),xb);
            ac[r]+=d[r][b]*(float)svaddv_s32(p3,s); } }
    for(int r=0;r<8;r++) out[r]=ac[r];
}
/* Q8_0 OPTIMIZED: per-block SDOT -> fp-scale-accumulate into a per-row vector
 * (svmla by d_b), single svaddv per ROW (not per block). Removes the 168
 * per-block horizontal reductions that bottleneck q8_8row. Accurate (~1%). */
static inline void q8v_8row(float out[8], const int8_t*wq8,const float*d8,int nb,int cols,const int8_t*xi8){
    svbool_t pg32=svwhilelt_b8((uint32_t)0,(uint32_t)32), p3=svptrue_b32();
    svfloat32_t f0=svdup_f32(0),f1=svdup_f32(0),f2=svdup_f32(0),f3=svdup_f32(0),f4=svdup_f32(0),f5=svdup_f32(0),f6=svdup_f32(0),f7=svdup_f32(0);
    size_t rs=(size_t)cols; const int8_t*w0=wq8,*w1=wq8+rs,*w2=wq8+2*rs,*w3=wq8+3*rs,*w4=wq8+4*rs,*w5=wq8+5*rs,*w6=wq8+6*rs,*w7=wq8+7*rs;
    const float*d0=d8,*d1=d8+nb,*d2=d8+2*nb,*d3=d8+3*nb,*d4=d8+4*nb,*d5=d8+5*nb,*d6=d8+6*nb,*d7=d8+7*nb;
    for(int b=0;b<nb;b++){ svint8_t xb=svld1_s8(pg32,xi8+(size_t)b*32);
        f0=svmla_f32_x(p3,f0,svcvt_f32_s32_x(p3,svdot_s32(svdup_s32(0),svld1_s8(pg32,w0+(size_t)b*32),xb)),svdup_f32(d0[b]));
        f1=svmla_f32_x(p3,f1,svcvt_f32_s32_x(p3,svdot_s32(svdup_s32(0),svld1_s8(pg32,w1+(size_t)b*32),xb)),svdup_f32(d1[b]));
        f2=svmla_f32_x(p3,f2,svcvt_f32_s32_x(p3,svdot_s32(svdup_s32(0),svld1_s8(pg32,w2+(size_t)b*32),xb)),svdup_f32(d2[b]));
        f3=svmla_f32_x(p3,f3,svcvt_f32_s32_x(p3,svdot_s32(svdup_s32(0),svld1_s8(pg32,w3+(size_t)b*32),xb)),svdup_f32(d3[b]));
        f4=svmla_f32_x(p3,f4,svcvt_f32_s32_x(p3,svdot_s32(svdup_s32(0),svld1_s8(pg32,w4+(size_t)b*32),xb)),svdup_f32(d4[b]));
        f5=svmla_f32_x(p3,f5,svcvt_f32_s32_x(p3,svdot_s32(svdup_s32(0),svld1_s8(pg32,w5+(size_t)b*32),xb)),svdup_f32(d5[b]));
        f6=svmla_f32_x(p3,f6,svcvt_f32_s32_x(p3,svdot_s32(svdup_s32(0),svld1_s8(pg32,w6+(size_t)b*32),xb)),svdup_f32(d6[b]));
        f7=svmla_f32_x(p3,f7,svcvt_f32_s32_x(p3,svdot_s32(svdup_s32(0),svld1_s8(pg32,w7+(size_t)b*32),xb)),svdup_f32(d7[b]));
    }
    out[0]=svaddv_f32(p3,f0);out[1]=svaddv_f32(p3,f1);out[2]=svaddv_f32(p3,f2);out[3]=svaddv_f32(p3,f3);
    out[4]=svaddv_f32(p3,f4);out[5]=svaddv_f32(p3,f5);out[6]=svaddv_f32(p3,f6);out[7]=svaddv_f32(p3,f7);
}
/* Q8v2: full 64-wide SDOT over a PAIR (2 blocks). svdot lanes 0-7 = block0
 * partials, 8-15 = block1 -> scale by per-lane dvec=[d0x8,d1x8], fp-accumulate,
 * one svaddv/row. Full SDOT utilization + accurate (~1%). Requires even nb. */
static inline void q8v2_8row(float out[8], const int8_t*wq8,const float*d8,int nb,int cols,const int8_t*xi8){
    svbool_t pg=svptrue_b8(), p3=svptrue_b32(), lo8=svwhilelt_b32((uint32_t)0,(uint32_t)8);
    svfloat32_t f0=svdup_f32(0),f1=svdup_f32(0),f2=svdup_f32(0),f3=svdup_f32(0),f4=svdup_f32(0),f5=svdup_f32(0),f6=svdup_f32(0),f7=svdup_f32(0);
    size_t rs=(size_t)cols; const int8_t*w[8]; const float*d[8];
    for(int r=0;r<8;r++){ w[r]=wq8+(size_t)r*rs; d[r]=d8+(size_t)r*nb; }
    int pairs=nb/2;
    for(int p=0;p<pairs;p++){ svint8_t xb=svld1_s8(pg,xi8+(size_t)p*64);
        #define R(F,R_) { svint32_t sv=svdot_s32(svdup_s32(0),svld1_s8(pg,w[R_]+(size_t)p*64),xb); \
            svfloat32_t dv=svsel_f32(lo8,svdup_f32(d[R_][2*p]),svdup_f32(d[R_][2*p+1])); \
            F=svmla_f32_x(p3,F,svcvt_f32_s32_x(p3,sv),dv); }
        R(f0,0)R(f1,1)R(f2,2)R(f3,3)R(f4,4)R(f5,5)R(f6,6)R(f7,7)
        #undef R
    }
    out[0]=svaddv_f32(p3,f0);out[1]=svaddv_f32(p3,f1);out[2]=svaddv_f32(p3,f2);out[3]=svaddv_f32(p3,f3);
    out[4]=svaddv_f32(p3,f4);out[5]=svaddv_f32(p3,f5);out[6]=svaddv_f32(p3,f6);out[7]=svaddv_f32(p3,f7);
}
/* Q8v3: like Q8v2 but 2 fp accumulators/row (16 total, even/odd pairs) to hide
 * the svmla dependency latency (ILP 8 -> 16). Even nb (pairs), pairs even-ish. */
static inline void q8v3_8row(float out[8], const int8_t*wq8,const float*d8,int nb,int cols,const int8_t*xi8){
    svbool_t pg=svptrue_b8(), p3=svptrue_b32(), lo8=svwhilelt_b32((uint32_t)0,(uint32_t)8);
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0),a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    svfloat32_t b0=svdup_f32(0),b1=svdup_f32(0),b2=svdup_f32(0),b3=svdup_f32(0),b4=svdup_f32(0),b5=svdup_f32(0),b6=svdup_f32(0),b7=svdup_f32(0);
    size_t rs=(size_t)cols; const int8_t*w[8]; const float*d[8];
    for(int r=0;r<8;r++){ w[r]=wq8+(size_t)r*rs; d[r]=d8+(size_t)r*nb; }
    int pairs=nb/2, p=0;
#define MAC(F,RR,PP) { svint32_t sv=svdot_s32(svdup_s32(0),svld1_s8(pg,w[RR]+(size_t)(PP)*64),xb); \
    F=svmla_f32_x(p3,F,svcvt_f32_s32_x(p3,sv),svsel_f32(lo8,svdup_f32(d[RR][2*(PP)]),svdup_f32(d[RR][2*(PP)+1]))); }
    for(;p+2<=pairs;p+=2){
        { svint8_t xb=svld1_s8(pg,xi8+(size_t)p*64);      MAC(a0,0,p)MAC(a1,1,p)MAC(a2,2,p)MAC(a3,3,p)MAC(a4,4,p)MAC(a5,5,p)MAC(a6,6,p)MAC(a7,7,p) }
        { svint8_t xb=svld1_s8(pg,xi8+(size_t)(p+1)*64);  MAC(b0,0,p+1)MAC(b1,1,p+1)MAC(b2,2,p+1)MAC(b3,3,p+1)MAC(b4,4,p+1)MAC(b5,5,p+1)MAC(b6,6,p+1)MAC(b7,7,p+1) }
    }
    for(;p<pairs;p++){ svint8_t xb=svld1_s8(pg,xi8+(size_t)p*64); MAC(a0,0,p)MAC(a1,1,p)MAC(a2,2,p)MAC(a3,3,p)MAC(a4,4,p)MAC(a5,5,p)MAC(a6,6,p)MAC(a7,7,p) }
#undef MAC
    out[0]=svaddv_f32(p3,svadd_f32_x(p3,a0,b0));out[1]=svaddv_f32(p3,svadd_f32_x(p3,a1,b1));
    out[2]=svaddv_f32(p3,svadd_f32_x(p3,a2,b2));out[3]=svaddv_f32(p3,svadd_f32_x(p3,a3,b3));
    out[4]=svaddv_f32(p3,svadd_f32_x(p3,a4,b4));out[5]=svaddv_f32(p3,svadd_f32_x(p3,a5,b5));
    out[6]=svaddv_f32(p3,svadd_f32_x(p3,a6,b6));out[7]=svaddv_f32(p3,svadd_f32_x(p3,a7,b7));
}
/* Q8v4: interleaved-by-4 pair layout so svdot even lanes=block0 partials,
 * odd=block1. Per-lane d-vector is svld1rq([d0,d1,d0,d1]) (1 replicated load,
 * no svsel/dup). wi/xi pre-interleaved; dp = 4 floats [d0,d1,d0,d1] per pair. */
static inline void q8v4_8row(float out[8], const int8_t*wq8i,const float*dp8,int pairs,size_t rs,const int8_t*xi8i){
    svbool_t pg=svptrue_b8(), p3=svptrue_b32();
    svfloat32_t f0=svdup_f32(0),f1=svdup_f32(0),f2=svdup_f32(0),f3=svdup_f32(0),f4=svdup_f32(0),f5=svdup_f32(0),f6=svdup_f32(0),f7=svdup_f32(0);
    const int8_t*w[8]; const float*d[8];
    for(int r=0;r<8;r++){ w[r]=wq8i+(size_t)r*rs; d[r]=dp8+(size_t)r*pairs*4; }
    for(int p=0;p<pairs;p++){ svint8_t xb=svld1_s8(pg,xi8i+(size_t)p*64);
#define M4(F,RR) { svfloat32_t dv=svld1rq_f32(p3,d[RR]+(size_t)p*4); \
    F=svmla_f32_x(p3,F,svcvt_f32_s32_x(p3,svdot_s32(svdup_s32(0),svld1_s8(pg,w[RR]+(size_t)p*64),xb)),dv); }
        M4(f0,0)M4(f1,1)M4(f2,2)M4(f3,3)M4(f4,4)M4(f5,5)M4(f6,6)M4(f7,7)
#undef M4
    }
    out[0]=svaddv_f32(p3,f0);out[1]=svaddv_f32(p3,f1);out[2]=svaddv_f32(p3,f2);out[3]=svaddv_f32(p3,f3);
    out[4]=svaddv_f32(p3,f4);out[5]=svaddv_f32(p3,f5);out[6]=svaddv_f32(p3,f6);out[7]=svaddv_f32(p3,f7);
}
/* exact fp32 reference matvec from Q4_0 (true dequantized weights). */
static void ref_mv(float*dst,const block_q4_0*W,size_t rb,const float*x,int rows,int cols){
    int nb=cols/32;
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)r*rb);
        float s=0; for(int b=0;b<nb;b++){ float d=ggml_fp16_to_fp32(row[b].d);
            for(int j=0;j<16;j++){ s+=d*((row[b].qs[j]&0xf)-8)*x[b*32+j]; s+=d*((row[b].qs[j]>>4)-8)*x[b*32+j+16]; } }
        dst[r]=s; }
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [tensor] [/local/out.i8]\n",argv[0]); return 1; }
    const char*tname=argc>2?argv[2]:"blk.0.ffn_gate.weight";
    const char*outp =argc>3?argv[3]:"/local/u14346/w.i8";
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    double freq; { uint64_t f; __asm__ volatile("mrs %0,cntfrq_el0":"=r"(f)); freq=(double)f; }

    gguf_context*g=gguf_open(argv[1],1); if(!g){ fprintf(stderr,"open failed\n"); return 1; }
    int idx=find_tensor(g,tname); if(idx<0){ fprintf(stderr,"tensor %s not found\n",tname); return 1; }
    gguf_tensor_info*t=&g->tensors[idx];
    int cols=(int)t->dims[0], rows=(int)t->dims[1], nb=cols/32, nbp=(cols+63)/64;
    size_t rb=(size_t)nb*sizeof(block_q4_0), q4sz=(size_t)rows*rb, i8sz=(size_t)rows*nbp*64;
    fprintf(stderr,"[opt] %s  rows=%d cols=%d  Q4_0=%.0f MB int8=%.0f MB\n",tname,rows,cols,q4sz/1e6,i8sz/1e6);

    /* extract ONLY this tensor's Q4_0 into a small buffer, then drop the model */
    block_q4_0*W=(block_q4_0*)aligned_alloc(256,q4sz);
    memcpy(W, gguf_tensor_data(g,idx), q4sz);
    gguf_close(g);

    float*x=(float*)aligned_alloc(256,(size_t)cols*4); srand(137); for(int i=0;i<cols;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*2.0f;
    float*dref=(float*)aligned_alloc(256,(size_t)rows*4), *di8=(float*)aligned_alloc(256,(size_t)rows*4);
    ref_mv(dref,W,rb,x,rows,cols);

    /* dequant -> int8 (per-tensor scale_w), write to /local, then mmap it back */
    float max_d=0; for(size_t i=0;i<(size_t)rows*nb;i++){ float a=fabsf(ggml_fp16_to_fp32(W[i].d)); if(a>max_d)max_d=a; }
    float sw=max_d>0?127.0f/(8.0f*max_d):1.0f;
    /* dequant into an HBM buffer, first-touched NUMA-local by the SAME parallel
     * split the matvec uses (so the kernel perf is real, not cross-CMG). Also
     * write it to /local as the staging artifact (the no-OOM path for the full
     * model); but the kernel runs on this NUMA-local HBM buffer. */
    /* mmap fresh anon pages (NOT aligned_alloc — glibc reuses faulted arena
     * pages, defeating the per-thread first-touch and forcing cross-CMG reads). */
    int8_t*WI8=(int8_t*)mmap(NULL,i8sz,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r0=0;r0<rows;r0+=8) tf_dequant_q4_0_8row_strided_to_int8((const uint8_t*)W+(size_t)r0*rb,rb,WI8+(size_t)r0*nbp*64,cols,sw);
    { int fd2=open(outp,O_WRONLY|O_CREAT|O_TRUNC,0644); if(fd2>=0){ ssize_t wr=0; while(wr<(ssize_t)i8sz){ssize_t n=write(fd2,WI8+wr,i8sz-wr); if(n<=0)break; wr+=n;} close(fd2);
      fprintf(stderr,"[opt] staged int8 -> %s (%.0f MB)\n",outp,i8sz/1e6); } }
    int8_t*xi8=(int8_t*)aligned_alloc(256,(size_t)nbp*64); float x_inv; tf_quantize_f32_to_int8(x,xi8,cols,&x_inv);
    float inv=1.0f/(sw*x_inv); float inv_x0=1.0f/x_inv;

    /* int8 matvec (warm: first touch pages in from /local), best-of-N */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r0=0;r0<rows;r0+=8){ int32_t a[8]; mv8(a,WI8+(size_t)r0*nbp*64,nbp,xi8); for(int i=0;i<8;i++) di8[r0+i]=(float)a[i]*inv; }
    double bi=1e30; for(int rep=0;rep<20;rep++){ double s=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r0=0;r0<rows;r0+=8){ int32_t a[8]; mv8(a,WI8+(size_t)r0*nbp*64,nbp,xi8); for(int i=0;i<8;i++) di8[r0+i]=(float)a[i]*inv; }
        double e=now()-s; if(e<bi)bi=e; }
    /* fp32 reference perf (the current decode path) */
    double bf=1e30; for(int rep=0;rep<10;rep++){ double s=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r0=0;r0<rows;r0+=8) tf_matvec_q4_0_rows(di8,(const uint8_t*)W,rb,x,cols,r0,(r0+8<=rows)?r0+8:rows);
        double e=now()-s; if(e<bf)bf=e; }
    /* Q8_0 per-block path: build int8(v-8) [cols/row] + fp d [nb/row], NUMA-local */
    size_t q8sz=(size_t)rows*cols; int8_t*WQ8=(int8_t*)mmap(NULL,q8sz,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    float*D8=(float*)mmap(NULL,(size_t)rows*nb*4,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)r*rb); int8_t*o=WQ8+(size_t)r*cols;
        for(int b=0;b<nb;b++){ D8[(size_t)r*nb+b]=ggml_fp16_to_fp32(row[b].d);
            for(int j=0;j<16;j++){ o[b*32+j]=(int8_t)((row[b].qs[j]&0xf)-8); o[b*32+16+j]=(int8_t)((row[b].qs[j]>>4)-8); } } }
    float fout8[8];
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r0=0;r0<rows;r0+=8){ float o[8]; q8_8row(o,WQ8+(size_t)r0*cols,D8+(size_t)r0*nb,nb,cols,xi8); for(int i=0;i<8;i++) di8[r0+i]=o[i]*inv_x0; }
    double bq=1e30; for(int rep=0;rep<20;rep++){ double s=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r0=0;r0<rows;r0+=8){ float o[8]; q8_8row(o,WQ8+(size_t)r0*cols,D8+(size_t)r0*nb,nb,cols,xi8); for(int i=0;i<8;i++) di8[r0+i]=o[i]*inv_x0; }
        double e=now()-s; if(e<bq)bq=e; }
    double bqv=1e30; for(int rep=0;rep<20;rep++){ double s=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r0=0;r0<rows;r0+=8){ float o[8]; q8v_8row(o,WQ8+(size_t)r0*cols,D8+(size_t)r0*nb,nb,cols,xi8); for(int i=0;i<8;i++) di8[r0+i]=o[i]*inv_x0; }
        double e=now()-s; if(e<bqv)bqv=e; }
    double bqv2=1e30; for(int rep=0;rep<20;rep++){ double s=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r0=0;r0<rows;r0+=8){ float o[8]; q8v2_8row(o,WQ8+(size_t)r0*cols,D8+(size_t)r0*nb,nb,cols,xi8); for(int i=0;i<8;i++) di8[r0+i]=o[i]*inv_x0; }
        double e=now()-s; if(e<bqv2)bqv2=e; }
    double bqv3=1e30; for(int rep=0;rep<20;rep++){ double s=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r0=0;r0<rows;r0+=8){ float o[8]; q8v3_8row(o,WQ8+(size_t)r0*cols,D8+(size_t)r0*nb,nb,cols,xi8); for(int i=0;i<8;i++) di8[r0+i]=o[i]*inv_x0; }
        double e=now()-s; if(e<bqv3)bqv3=e; }
    /* Q8v4: build interleaved-by-4 weights + [d0,d1,d0,d1]/pair + interleaved x */
    int pairs=nb/2; size_t dp8sz=(size_t)rows*pairs*4*4;
    int8_t*WQ8i=(int8_t*)mmap(NULL,q8sz,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    float *DP8=(float *)mmap(NULL,dp8sz,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    int8_t*xi8i=(int8_t*)aligned_alloc(256,(size_t)pairs*64);
    for(int p=0;p<pairs;p++) for(int gg=0;gg<8;gg++){ const int8_t*sx=xi8+(size_t)p*64; int8_t*dx=xi8i+(size_t)p*64+gg*8;
        for(int k=0;k<4;k++){ dx[k]=sx[gg*4+k]; dx[4+k]=sx[32+gg*4+k]; } }
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r=0;r<rows;r++){ const int8_t*sw=WQ8+(size_t)r*cols; int8_t*dw=WQ8i+(size_t)r*cols; const float*dd=D8+(size_t)r*nb; float*dpp=DP8+(size_t)r*pairs*4;
        for(int p=0;p<pairs;p++){ const int8_t*spw=sw+(size_t)p*64; int8_t*dpw=dw+(size_t)p*64;
            for(int gg=0;gg<8;gg++) for(int k=0;k<4;k++){ dpw[gg*8+k]=spw[gg*4+k]; dpw[gg*8+4+k]=spw[32+gg*4+k]; }
            dpp[p*4+0]=dd[2*p]; dpp[p*4+1]=dd[2*p+1]; dpp[p*4+2]=dd[2*p]; dpp[p*4+3]=dd[2*p+1]; } }
    double bqv4=1e30; for(int rep=0;rep<20;rep++){ double s=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r0=0;r0<rows;r0+=8){ float o[8]; q8v4_8row(o,WQ8i+(size_t)r0*cols,DP8+(size_t)r0*pairs*4,pairs,(size_t)cols,xi8i); for(int i=0;i<8;i++) di8[r0+i]=o[i]*inv_x0; }
        double e=now()-s; if(e<bqv4)bqv4=e; }
    double e2v4=0; { float*tmp=(float*)aligned_alloc(256,(size_t)rows*4);
      #pragma omp parallel for num_threads(nt) schedule(static)
      for(int r0=0;r0<rows;r0+=8){ float o[8]; q8v4_8row(o,WQ8i+(size_t)r0*cols,DP8+(size_t)r0*pairs*4,pairs,(size_t)cols,xi8i); for(int i=0;i<8;i++) tmp[r0+i]=o[i]*inv_x0; }
      for(int i=0;i<rows;i++){ double d=tmp[i]-dref[i]; e2v4+=d*d; } free(tmp); }
    munmap(WQ8i,q8sz); munmap(DP8,dp8sz); free(xi8i);
    /* accuracy of Q8v2 (should match per-block ~1%) */
    double e2v=0; { float*tmp=(float*)aligned_alloc(256,(size_t)rows*4);
      #pragma omp parallel for num_threads(nt) schedule(static)
      for(int r0=0;r0<rows;r0+=8){ float o[8]; q8v2_8row(o,WQ8+(size_t)r0*cols,D8+(size_t)r0*nb,nb,cols,xi8); for(int i=0;i<8;i++) tmp[r0+i]=o[i]*inv_x0; }
      for(int i=0;i<rows;i++){ double d=tmp[i]-dref[i]; e2v+=d*d; } free(tmp); }
    (void)fout8;

    /* error: recompute the int8 output (di8 was overwritten by the fp32 timing
     * loop above), then compare to the exact fp32-Q4_0 reference. */
    double e2=0,s2=0,mx=0;
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r0=0;r0<rows;r0+=8){ int32_t a[8]; mv8(a,WI8+(size_t)r0*nbp*64,nbp,xi8); for(int i=0;i<8;i++) di8[r0+i]=(float)a[i]*inv; }
    for(int i=0;i<rows;i++){ double d=di8[i]-dref[i]; e2+=d*d; s2+=(double)dref[i]*dref[i]; double ad=fabs(d); if(ad>mx)mx=ad; }
    /* per-block (Q8_0-style) accuracy: each block keeps its OWN fp d (lossless
     * from Q4_0); only x is int8-quantized. Isolates weight-quant vs x-quant. */
    double e2b=0; float inv_x=1.0f/x_inv;
    #pragma omp parallel for schedule(static) reduction(+:e2b)
    for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)r*rb);
        double s=0; for(int b=0;b<nb;b++){ float d=ggml_fp16_to_fp32(row[b].d); int bs=0;
            for(int j=0;j<16;j++){ bs+=((row[b].qs[j]&0xf)-8)*xi8[b*32+j]; bs+=((row[b].qs[j]>>4)-8)*xi8[b*32+j+16]; }
            s+=(double)d*bs; }
        double v=s*inv_x - dref[r]; e2b+=v*v; }
    /* per-ROW scale accuracy: same BW-bound kernel as per-tensor (1 SDOT/pair),
     * but each row uses its own scale_w_r = 127/(8*max_d_in_row). Output scale
     * is per-row (free). Does row-to-row d variation explain the 8.9%? */
    double e2r=0;
    #pragma omp parallel for schedule(static) reduction(+:e2r)
    for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)r*rb);
        float md=0; for(int b=0;b<nb;b++){ float a=fabsf(ggml_fp16_to_fp32(row[b].d)); if(a>md)md=a; }
        float swr=md>0?127.0f/(8.0f*md):1.0f; double acc=0;
        for(int b=0;b<nb;b++){ float d=ggml_fp16_to_fp32(row[b].d); int di=(int)lrintf(d*swr); if(di>127)di=127; if(di<-128)di=-128;
            for(int j=0;j<16;j++){ acc+=(double)((row[b].qs[j]&0xf)-8)*di*xi8[b*32+j]; acc+=(double)((row[b].qs[j]>>4)-8)*di*xi8[b*32+16+j]; } }
        double v=acc/((double)swr*x_inv)-dref[r]; e2r+=v*v; }
    double opsf=2.0*rows*cols;
    printf("\n=== %s (rows=%d cols=%d) @ %d threads ===\n",tname,rows,cols,nt);
    printf("  PERF @ %d threads (NUMA-local HBM, best-of-N):\n",nt);
    printf("    fp32 fused (current) : %.3f ms  %.1f GFLOPS  1.0x\n",bf*1000,opsf/bf/1e9);
    printf("    int8 per-TENSOR SDOT : %.3f ms  %.1f GIOPS  %.1fx\n",bi*1000,opsf/bi/1e9,bf/bi);
    printf("    int8 per-BLOCK (Q8_0): %.3f ms  %.1f GIOPS  %.1fx  (svaddv/block)\n",bq*1000,opsf/bq/1e9,bf/bq);
    printf("    int8 per-BLOCK (Q8v) : %.3f ms  %.1f GIOPS  %.1fx  (32-wide, fp-accum/row)\n",bqv*1000,opsf/bqv/1e9,bf/bqv);
    printf("    int8 per-BLOCK (Q8v2): %.3f ms  %.1f GIOPS  %.1fx  %.0f GB/s (64-wide pair)\n",bqv2*1000,opsf/bqv2/1e9,bf/bqv2,(double)(q8sz+(size_t)rows*nb*4)/bqv2/1e9);
    printf("    int8 per-BLOCK (Q8v3): %.3f ms  %.1f GIOPS  %.1fx  %.0f GB/s (16-acc ILP)\n",bqv3*1000,opsf/bqv3/1e9,bf/bqv3,(double)(q8sz+(size_t)rows*nb*4)/bqv3/1e9);
    printf("    int8 per-BLOCK (Q8v4): %.3f ms  %.1f GIOPS  %.1fx  %.0f GB/s (interleave+ld1rq)\n",bqv4*1000,opsf/bqv4/1e9,bf/bqv4,(double)(q8sz+(size_t)rows*pairs*16)/bqv4/1e9);
    printf("  ACCURACY vs exact fp32-Q4_0 (rms=%.4g):\n", sqrt(s2/rows));
    printf("    per-TENSOR int8 : relL2=%.4f  maxabs=%.4g  <- fast but lossy\n", sqrt(e2/(s2+1e-12)), mx);
    printf("    per-ROW    int8 : relL2=%.4f             <- SAME fast kernel, per-row scale\n", sqrt(e2r/(s2+1e-12)));
    printf("    per-BLOCK  Q8_0 : relL2=%.4f             <- accurate\n", sqrt(e2b/(s2+1e-12)));
    printf("    per-BLOCK  Q8v2 : relL2=%.4f (SVE kernel, real output)\n", sqrt(e2v/(s2+1e-12)));
    printf("    per-BLOCK  Q8v4 : relL2=%.4f (interleaved SVE kernel)\n", sqrt(e2v4/(s2+1e-12)));
    munmap(WI8,i8sz); munmap(WQ8,q8sz); munmap(D8,(size_t)rows*nb*4); free(W); free(x); free(dref); free(di8); free(xi8);
    return 0;
}
