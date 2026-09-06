/* lmhead_opt.c - optimize the OTHER decode weight kernel: the Q6_K tied
 * lm-head (token_embd.weight, 5376x262144), run every token for all logits.
 * Same manner: extract the one tensor, build int8, measure int8-SDOT perf
 * (BW%) + accuracy vs the exact Q6_K-fp32 reference. int8 SDOT kernel is mv8
 * (already optimized); the new questions are Q6_K->int8 error + BW on a huge
 * shape. Memory: Q6_K ~1.1GB + fp32 ~5.6GB + int8 ~1.4GB (one tensor, no OOM).
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -D_GNU_SOURCE -I../../common lmhead_opt.c -lm -o lmhead_opt
 * Run:   NUMA_DISTRIBUTE=1 OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./lmhead_opt model.gguf
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <omp.h>
#include <sys/mman.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int find_tensor(gguf_context*g,const char*n){ for(uint64_t i=0;i<g->n_tensors;i++){ const char*s=gguf_tensor_name(g,(int)i); if(s&&!strcmp(s,n))return (int)i;} return -1; }
static void*xmap(size_t n){ void*p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); if(p==MAP_FAILED){perror("mmap");exit(1);} return p; }

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

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf\n",argv[0]); return 1; }
    const char*tname="token_embd.weight";
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    gguf_context*g=gguf_open(argv[1],1); if(!g){fprintf(stderr,"open\n");return 1;}
    int idx=find_tensor(g,tname); if(idx<0){fprintf(stderr,"no %s\n",tname);return 1;}
    gguf_tensor_info*t=&g->tensors[idx];
    int cols=(int)t->dims[0], rows=(int)t->dims[1], nbp=(cols+63)/64;
    size_t qsz=gguf_tensor_size(g,idx);
    fprintf(stderr,"[lm] %s rows=%d cols=%d Q6_K=%.2f GB int8=%.2f GB\n",tname,rows,cols,qsz/1e9,(double)rows*nbp*64/1e9);
    uint8_t*Wq=(uint8_t*)xmap(qsz); memcpy(Wq,gguf_tensor_data(g,idx),qsz); gguf_close(g);
    size_t rowq=qsz/rows;  /* Q6_K bytes per row */

    /* dequant Q6_K -> fp32 (NUMA-local), parallel per row */
    float*Wf=(float*)xmap((size_t)rows*cols*4);
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r=0;r<rows;r++) dequantize_row_q6_K(Wq+(size_t)r*rowq, Wf+(size_t)r*cols, cols);
    float*x=(float*)aligned_alloc(256,(size_t)cols*4); srand(137); for(int i=0;i<cols;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*2.0f;
    float*dref=(float*)aligned_alloc(256,(size_t)rows*4), *dout=(float*)aligned_alloc(256,(size_t)rows*4);
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r=0;r<rows;r++){ const float*w=Wf+(size_t)r*cols; double s=0; for(int j=0;j<cols;j++) s+=(double)w[j]*x[j]; dref[r]=s; }

    /* global max|w| for per-tensor scale */
    float gmax=0;
    #pragma omp parallel for num_threads(nt) schedule(static) reduction(max:gmax)
    for(int r=0;r<rows;r++){ const float*w=Wf+(size_t)r*cols; for(int j=0;j<cols;j++){ float a=fabsf(w[j]); if(a>gmax)gmax=a; } }
    float gsc=gmax>0?127.0f/gmax:1.0f;
    /* int8 weights (per-tensor scale), NUMA-local */
    int8_t*WI8=(int8_t*)xmap((size_t)rows*nbp*64);
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r=0;r<rows;r++){ const float*w=Wf+(size_t)r*cols; int8_t*o=WI8+(size_t)r*nbp*64;
        for(int j=0;j<cols;j++){ int v=(int)lrintf(w[j]*gsc); if(v>127)v=127; if(v<-128)v=-128; o[j]=(int8_t)v; } }
    int8_t*xi8=(int8_t*)aligned_alloc(256,(size_t)nbp*64); float x_inv; tf_quantize_f32_to_int8(x,xi8,cols,&x_inv);
    float inv=1.0f/(gsc*x_inv), inv_x=1.0f/x_inv;

    /* perf: int8 SDOT (mv8), best-of-N */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r0=0;r0<rows;r0+=8){ int32_t a[8]; mv8(a,WI8+(size_t)r0*nbp*64,nbp,xi8); for(int i=0;i<8;i++) dout[r0+i]=(float)a[i]*inv; }
    double bi=1e30; for(int rep=0;rep<20;rep++){ double s=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r0=0;r0<rows;r0+=8){ int32_t a[8]; mv8(a,WI8+(size_t)r0*nbp*64,nbp,xi8); for(int i=0;i<8;i++) dout[r0+i]=(float)a[i]*inv; }
        double e=now()-s; if(e<bi)bi=e; }

    /* fp32 baseline: dot the resident fp32 weights (BW-bound on 4 B/wt; the
     * production on-the-fly Q6_K dequant+dot is compute-bound, slower still). */
    double bf=1e30; for(int rep=0;rep<6;rep++){ double s=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r=0;r<rows;r++){ const float*w=Wf+(size_t)r*cols; float acc=0; for(int j=0;j<cols;j++) acc+=w[j]*x[j]; dout[r]=acc; }
        double e=now()-s; if(e<bf)bf=e; }

    /* accuracy: per-tensor (dout) + per-row (recompute scalar) */
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int r0=0;r0<rows;r0+=8){ int32_t a[8]; mv8(a,WI8+(size_t)r0*nbp*64,nbp,xi8); for(int i=0;i<8;i++) dout[r0+i]=(float)a[i]*inv; }
    double e2=0,s2=0; for(int i=0;i<rows;i++){ double d=dout[i]-dref[i]; e2+=d*d; s2+=(double)dref[i]*dref[i]; }
    double e2r=0;
    #pragma omp parallel for num_threads(nt) schedule(static) reduction(+:e2r)
    for(int r=0;r<rows;r++){ const float*w=Wf+(size_t)r*cols; float mr=0; for(int j=0;j<cols;j++){ float a=fabsf(w[j]); if(a>mr)mr=a; }
        float sr=mr>0?127.0f/mr:1.0f; double acc=0;
        for(int j=0;j<cols;j++){ int v=(int)lrintf(w[j]*sr); if(v>127)v=127; if(v<-128)v=-128; acc+=(double)v*xi8[j]; }
        double vv=acc/((double)sr*x_inv)-dref[r]; e2r+=vv*vv; }

    double opsf=2.0*rows*cols, bytes=(double)rows*nbp*64;
    double ceil_gbps=633.0;
    printf("\n=== lm_head %s (rows=%d cols=%d) @ %d threads ===\n",tname,rows,cols,nt);
    printf("  fp32 scalar dot (NOT representative; compiler didn't vectorize): %.3f ms\n",bf*1000);
    printf("  int8 SDOT (per-tensor) : %.3f ms  %.1f GIOPS  %.1f GB/s  (%.0f%% of %.0f GB/s ceiling)\n",
           bi*1000,opsf/bi/1e9,bytes/bi/1e9,bytes/bi/1e9/ceil_gbps*100,ceil_gbps);
    printf("    int8 reads 1 B/wt vs fp32 4 B/wt -> ~4x over BW-bound fp32; more vs Q6_K-dequant decode path\n");
    printf("  ACCURACY vs exact Q6_K-fp32 (rms=%.4g):\n",sqrt(s2/rows));
    printf("    per-TENSOR int8 : relL2=%.4f\n",sqrt(e2/(s2+1e-12)));
    printf("    per-ROW    int8 : relL2=%.4f\n",sqrt(e2r/(s2+1e-12)));
    (void)inv_x;
    return 0;
}
