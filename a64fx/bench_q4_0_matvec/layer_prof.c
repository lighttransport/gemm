/* layer_prof.c - per-op decode-matvec profiling on REAL Gemma-4 31B weights.
 *
 * The production llm_runner uses the on-the-fly fp32 dequant+FMA path
 * (tf_matvec_q4_0_rows). This loads each decode weight matrix from the real
 * GGUF and times the M=1 matvec two ways at 1 CMG (NUMA-clean, best-of-N):
 *   fp32  = tf_matvec_q4_0_rows           (what decode does today)
 *   int8  = dequant->int8 + SDOT (mv8)    (the faster, BW-bound path)
 * Reports per-op ms/GFLOPS and the per-token decode-matvec total for both,
 * to find the dominant op and quantify the int8 win.
 *
 * Run: NUMA_DISTRIBUTE=1 OMP_PROC_BIND=close OMP_PLACES=cores \
 *      ./layer_prof ~/models/gemma4/31b-qat/gemma-4-31B_q4_0-it.gguf [nthreads]
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <omp.h>
#include <sys/mman.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }

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
    for(int i=0;i<8;i++){} acc[0]=svaddv_s32(p3,a0);acc[1]=svaddv_s32(p3,a1);acc[2]=svaddv_s32(p3,a2);acc[3]=svaddv_s32(p3,a3);
    acc[4]=svaddv_s32(p3,a4);acc[5]=svaddv_s32(p3,a5);acc[6]=svaddv_s32(p3,a6);acc[7]=svaddv_s32(p3,a7);
}

static int find_tensor(gguf_context*g,const char*name){ for(uint64_t i=0;i<g->n_tensors;i++){ const char*n=gguf_tensor_name(g,(int)i); if(n&&!strcmp(n,name)) return (int)i; } return -1; }

typedef struct { const char*op; int count; } opspec;

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [nthreads]\n",argv[0]); return 1; }
    int nt=argc>2?atoi(argv[2]):12;
    double freq=(double)rdfreq();
    gguf_context*g=gguf_open(argv[1],1); if(!g){ fprintf(stderr,"open failed\n"); return 1; }

    /* decode matmuls per layer; counts = how many of the 60 layers have this op
     * (Gemma-4: 50 "global" layers + 10 "local"-attn layers w/ wider q/o). We
     * profile layer 0 (global) shapes and multiply by per-op layer counts. */
    const char*ops[]={"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"};
    int counts[]={50,50,50,50,60,60,60};   /* attn q/k/v/o on the 50 global layers; ffn on all 60 */
    printf("Gemma-4 31B decode-matvec profile @ %d threads (best-of-N, NUMA-local)\n",nt);
    printf("  %-12s %-12s %8s %9s %8s %9s %7s\n","op(blk.0)","shape r x c","fp32 ms","fp32 GF","int8 ms","int8 GIOP","x");
    double tot_fp32=0, tot_int8=0, tot_otf=0;
    for(int o=0;o<7;o++){
        char nm[64]; snprintf(nm,sizeof nm,"blk.0.%s.weight",ops[o]);
        int idx=find_tensor(g,nm); if(idx<0){ printf("  %-12s MISSING\n",ops[o]); continue; }
        gguf_tensor_info*t=&g->tensors[idx];
        int cols=(int)t->dims[0], rows=(int)t->dims[1];           /* ne0=in, ne1=out */
        int nb=cols/32, nbp=(cols+63)/64; size_t rb=(size_t)nb*sizeof(block_q4_0);
        const uint8_t*src=(const uint8_t*)gguf_tensor_data(g,idx);
        /* copy real Q4_0 into a fresh NUMA-local buffer (first-touch by measurement split) */
        uint8_t*Wq=(uint8_t*)mmap(NULL,(size_t)rows*rb,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r=0;r<rows;r++) memcpy(Wq+(size_t)r*rb, src+(size_t)r*rb, rb);
        /* activation */
        float*x=(float*)aligned_alloc(256,(size_t)cols*4); for(int i=0;i<cols;i++) x[i]=((i*1103515245+12345)&0xffff)/32768.0f-1.0f;
        float*dst=(float*)aligned_alloc(256,(size_t)rows*4);
        double opsf=2.0*rows*cols;
        /* fp32 path (what decode uses today) */
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int g0=0;g0<rows;g0+=8) tf_matvec_q4_0_rows(dst,Wq,rb,x,cols,g0,(g0+8<=rows)?g0+8:rows);
        double bf=1e30; for(int rep=0;rep<10;rep++){ uint64_t t0=rdcyc();
            #pragma omp parallel for num_threads(nt) schedule(static)
            for(int g0=0;g0<rows;g0+=8) tf_matvec_q4_0_rows(dst,Wq,rb,x,cols,g0,(g0+8<=rows)?g0+8:rows);
            double s=(double)(rdcyc()-t0)/freq; if(s<bf)bf=s; }
        /* int8 path */
        float max_d=0; { const block_q4_0*bb=(const block_q4_0*)Wq; for(size_t i=0;i<(size_t)rows*nb;i++){ float a=fabsf(ggml_fp16_to_fp32(bb[i].d)); if(a>max_d)max_d=a; } }
        float sw=max_d>0?127.0f/(8.0f*max_d):1.0f;
        int8_t*WI8=(int8_t*)mmap(NULL,(size_t)rows*nbp*64,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int g0=0;g0<rows;g0+=8) tf_dequant_q4_0_8row_strided_to_int8(Wq+(size_t)g0*rb,rb,WI8+(size_t)g0*nbp*64,cols,sw);
        int8_t*xi8=(int8_t*)aligned_alloc(256,(size_t)nbp*64); float xinv; tf_quantize_f32_to_int8(x,xi8,cols,&xinv);
        float i8inv=1.0f/(sw*xinv);
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int g0=0;g0<rows;g0+=8){ int32_t a[8]; mv8(a,WI8+(size_t)g0*nbp*64,nbp,xi8); for(int i=0;i<8;i++) dst[g0+i]=(float)a[i]*i8inv; }
        double bi=1e30; for(int rep=0;rep<10;rep++){ uint64_t t0=rdcyc();
            #pragma omp parallel for num_threads(nt) schedule(static)
            for(int g0=0;g0<rows;g0+=8){ int32_t a[8]; mv8(a,WI8+(size_t)g0*nbp*64,nbp,xi8); for(int i=0;i<8;i++) dst[g0+i]=(float)a[i]*i8inv; }
            double s=(double)(rdcyc()-t0)/freq; if(s<bi)bi=s; }
        /* on-the-fly int8 (dequant Q4_0->int8 in the timed loop; fits in 17GB) */
        int8_t*scratch[64]={0};
        #pragma omp parallel num_threads(nt)
        { scratch[omp_get_thread_num()]=(int8_t*)aligned_alloc(256,(size_t)8*nbp*64); }
        double bo=1e30; for(int rep=0;rep<10;rep++){ uint64_t t0=rdcyc();
            #pragma omp parallel for num_threads(nt) schedule(static)
            for(int g0=0;g0<rows;g0+=8){ int8_t*ws=scratch[omp_get_thread_num()];
                tf_dequant_q4_0_8row_strided_to_int8(Wq+(size_t)g0*rb,rb,ws,cols,sw);
                int32_t a[8]; mv8(a,ws,nbp,xi8); for(int i=0;i<8;i++) dst[g0+i]=(float)a[i]*i8inv; }
            double s=(double)(rdcyc()-t0)/freq; if(s<bo)bo=s; }
        for(int tt=0;tt<nt;tt++) free(scratch[tt]);
        printf("  %-12s %5dx%-6d %7.3f %8.1f %7.3f %9.1f %7.3f %5.1fx/%4.1fx\n",ops[o],rows,cols,
               bf*1000,opsf/bf/1e9, bi*1000,opsf/bi/1e9, bo*1000, bf/bi, bf/bo);
        tot_fp32+=bf*counts[o]; tot_int8+=bi*counts[o]; tot_otf+=bo*counts[o];
        munmap(Wq,(size_t)rows*rb); munmap(WI8,(size_t)rows*nbp*64); free(x); free(dst); free(xi8);
    }
    printf("\n  per-token matvec total: fp32 %.1f ms (%.1f tok/s)  int8-preq %.1f ms (%.1f)  int8-otf %.1f ms (%.1f)\n", tot_fp32*1000,1.0/tot_fp32, tot_int8*1000,1.0/tot_int8, tot_otf*1000,1.0/tot_otf);
    printf("  preq fits? ~30GB int8 (tight on 32GB); otf fits in 17GB Q4_0\n");
    gguf_close(g);
    return 0;
}
