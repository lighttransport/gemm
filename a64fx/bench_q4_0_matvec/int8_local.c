/* int8_local.c - file-backed int8 decode weights on /local (no-OOM).
 *
 * The full int8 cache (~30 GB) OOMs as an anonymous malloc (30 GB int8 + 17 GB
 * Q4_0 + KV + NUMA panels > 32 GB). Instead: stream-dequant Q4_0 -> int8 to a
 * /local file (peak HBM = one tensor), then mmap it (PROT_READ). The pages are
 * CLEAN/file-backed = reclaimable page-cache, so they never OOM-kill; the page
 * cache holds the working set in HBM (~30 GB < 31 GB avail) => HBM-speed matvec
 * after a one-time warm-up, cold pages re-faulting from /local (~0.65 GB/s).
 *
 * STAGE: dequant all decode weights -> /local/g4_int8.bin + index.
 * PROFILE: decode-pass (one M=1 matvec per tensor, layer order), cold vs warm,
 * per-layer-type breakdown.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -D_GNU_SOURCE -I../../common int8_local.c -lm -o int8_local
 * Run:   OMP_PROC_BIND=close OMP_PLACES=cores ./int8_local <model.gguf> [/local/g4_int8.bin]
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
static int find_tensor(gguf_context*g,const char*name){ for(uint64_t i=0;i<g->n_tensors;i++){ const char*n=gguf_tensor_name(g,(int)i); if(n&&!strcmp(n,name))return (int)i;} return -1; }

#define MAXT 1024
typedef struct { char op[20]; int layer; size_t off; int rows,cols,nbp; } tinfo;
static tinfo idx[MAXT]; static int nidx=0;

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [/local/g4_int8.bin]\n",argv[0]); return 1; }
    const char*outp=argc>2?argv[2]:"/local/u14346/g4_int8.bin";
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    gguf_context*g=gguf_open(argv[1],1); if(!g){ fprintf(stderr,"open failed\n"); return 1; }
    const char*ops[]={"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"};
    int nlayer=0; { int kc=gguf_find_key_internal(g,"gemma4.block_count"); (void)kc; }
    /* discover layer count by probing */
    while(nlayer<200){ char nm[64]; snprintf(nm,sizeof nm,"blk.%d.ffn_gate.weight",nlayer); if(find_tensor(g,nm)<0) break; nlayer++; }
    fprintf(stderr,"[stage] %d layers; staging int8 to %s\n",nlayer,outp);

    /* ---- STAGE (skip if file exists) ---- */
    int fd; size_t total=0;
    if(access(outp,R_OK)==0){
        fprintf(stderr,"[stage] %s exists, skipping dequant\n",outp);
        /* rebuild index */
        for(int L=0;L<nlayer;L++) for(int o=0;o<7;o++){ char nm[64]; snprintf(nm,sizeof nm,"blk.%d.%s.weight",L,ops[o]); int ti=find_tensor(g,nm); if(ti<0)continue;
            gguf_tensor_info*t=&g->tensors[ti]; int cols=(int)t->dims[0],rows=(int)t->dims[1],nbp=(cols+63)/64;
            strcpy(idx[nidx].op,ops[o]); idx[nidx].layer=L; idx[nidx].off=total; idx[nidx].rows=rows; idx[nidx].cols=cols; idx[nidx].nbp=nbp;
            total+=(size_t)rows*nbp*64; nidx++; }
    } else {
        fd=open(outp,O_WRONLY|O_CREAT|O_TRUNC,0644); if(fd<0){ perror("open out"); return 1; }
        double ts=now();
        for(int L=0;L<nlayer;L++) for(int o=0;o<7;o++){ char nm[64]; snprintf(nm,sizeof nm,"blk.%d.%s.weight",L,ops[o]); int ti=find_tensor(g,nm); if(ti<0)continue;
            gguf_tensor_info*t=&g->tensors[ti]; int cols=(int)t->dims[0],rows=(int)t->dims[1],nb=cols/32,nbp=(cols+63)/64; size_t rb=(size_t)nb*sizeof(block_q4_0);
            const uint8_t*src=(const uint8_t*)gguf_tensor_data(g,ti);
            float max_d=0; { const block_q4_0*bb=(const block_q4_0*)src; for(size_t i=0;i<(size_t)rows*nb;i++){ float a=fabsf(ggml_fp16_to_fp32(bb[i].d)); if(a>max_d)max_d=a; } }
            float sw=max_d>0?127.0f/(8.0f*max_d):1.0f;
            size_t isz=(size_t)rows*nbp*64; int8_t*buf=(int8_t*)aligned_alloc(256,isz);
            #pragma omp parallel for num_threads(nt) schedule(static)
            for(int r0=0;r0<rows;r0+=8) tf_dequant_q4_0_8row_strided_to_int8(src+(size_t)r0*rb,rb,buf+(size_t)r0*nbp*64,cols,sw);
            ssize_t wr=0; while(wr<(ssize_t)isz){ ssize_t n=write(fd,buf+wr,isz-wr); if(n<=0){perror("write");return 1;} wr+=n; }
            free(buf);
            strcpy(idx[nidx].op,ops[o]); idx[nidx].layer=L; idx[nidx].off=total; idx[nidx].rows=rows; idx[nidx].cols=cols; idx[nidx].nbp=nbp; total+=isz; nidx++;
        }
        close(fd);
        fprintf(stderr,"[stage] wrote %.1f GB in %.1fs (%.0f MB/s)\n",total/1e9,now()-ts,total/1e6/(now()-ts));
    }
    fprintf(stderr,"[stage] int8 total %.1f GB across %d tensors\n",total/1e9,nidx);

    /* ---- PROFILE: mmap + decode-pass ---- */
    fd=open(outp,O_RDONLY); if(fd<0){perror("open ro");return 1;}
    int8_t*base=(int8_t*)mmap(NULL,total,PROT_READ,MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){ perror("mmap"); return 1; }
    int maxcols=21504; int8_t*xi8=(int8_t*)aligned_alloc(256,(size_t)((maxcols+63)&~63)); for(int i=0;i<maxcols;i++) xi8[i]=(int8_t)((i*7+1)&0x7f);
    int32_t*dst=(int32_t*)aligned_alloc(256,(size_t)24576*4);  /* consume mv8 output (defeat DCE) */
    fprintf(stderr,"[prof] mmap ok; running decode passes (cold then warm)\n");
    printf("  pass %-6s %9s %9s %9s\n","kind","ms/tok","GB/s","tok/s");
    double per_op_ms[7]={0};
    for(int pass=0;pass<4;pass++){
        for(int k=0;k<7;k++) per_op_ms[k]=0;
        double t0=now();
        for(int j=0;j<nidx;j++){ tinfo*ti=&idx[j];
            double s=now();
            #pragma omp parallel for num_threads(nt) schedule(static)
            for(int r0=0;r0<ti->rows;r0+=8){ int32_t a[8]; mv8(a,base+ti->off+(size_t)r0*ti->nbp*64,ti->nbp,xi8); for(int i=0;i<8;i++) dst[r0+i]=a[i]; }
            int oi=0; for(int o=0;o<7;o++) if(!strcmp(ti->op,ops[o])) oi=o; per_op_ms[oi]+=(now()-s)*1000;
        }
        double dt=now()-t0;
        printf("  %-4d %-6s %9.1f %9.1f %9.2f\n",pass,pass==0?"COLD":"warm",dt*1000,total/dt/1e9,1.0/dt);
        if(pass==3){ printf("  per-op-type ms/tok (warm): "); for(int o=0;o<7;o++) printf("%s=%.1f ",ops[o],per_op_ms[o]); printf("\n"); }
    }
    munmap(base,total); close(fd); gguf_close(g);
    return 0;
}
