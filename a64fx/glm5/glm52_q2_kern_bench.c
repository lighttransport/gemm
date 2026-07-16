#define _GNU_SOURCE
/* GLM-5.2 Q2 (mixed-IQ) decode kernel bench — native A64FX, 1 node, no MPI.
 *
 * Measures the three decode walls found on the 12n Q2 run (1.24-2.12 tok/s):
 *   1. dense BF16 GEMV thread scaling (glm5_mv_bf16: OMP-12 vs OMP-47 vs pinned pool)
 *   2. mixed-IQ q8 row kernels: v1 (stack tiles + per-tile svaddv) vs v2 (gather tiles)
 *   3. absorbed-MLA decode attention: serial glm5_prefill_absorb_token vs
 *      glm5_absorb_token_par (head x pos-chunk tiles on all threads)
 *
 * Build (native):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *       -I../../common -o glm52_q2_kern_bench glm52_q2_kern_bench.c -lm
 * Run:  OMP_PROC_BIND=close OMP_PLACES=cores ./glm52_q2_kern_bench
 */
#define GLM5_IMPL
#include "glm5.h"
#include <sys/syscall.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint32_t lcg(uint32_t*s){ *s=*s*1664525u+1013904223u; return *s; }
static float frnd(uint32_t*s){ return ((int)(lcg(s)%2001)-1000)*1e-3f; }

static void fill_bf16(uint16_t*w,size_t n,uint32_t seed){
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;i++){ uint32_t s=seed+(uint32_t)i*2654435761u; w[i]=glm5_f2bf(frnd(&s)*0.05f); }
}

static double bench_mv_bf16(const uint16_t*W,const float*x,float*y,int rows,int cols,int reps){
    glm5_mv_bf16(y,W,x,rows,cols);            /* warm */
    double t0=wall();
    for(int r=0;r<reps;r++) glm5_mv_bf16(y,W,x,rows,cols);
    double dt=(wall()-t0)/reps;
    return dt;
}
/* row-block variant: rb rows per inner call (rb=8 -> matvec_bf16_8row, rb=1 -> vec_dot).
 * Fewer rows per thread slice = fewer concurrent HBM streams. */
static void mv_bf16_rb(float*y,const uint16_t*W,const float*x,int rows,int cols,int rb){
    if(rb>=8){
        int nb=rows/8;
        #pragma omp parallel for schedule(static)
        for(int bi=0;bi<nb;bi++){ int r=bi*8; const uint16_t*b=W+(size_t)r*cols;
            matvec_bf16_8row(y+r,b,b+cols,b+2*(size_t)cols,b+3*(size_t)cols,
                             b+4*(size_t)cols,b+5*(size_t)cols,b+6*(size_t)cols,b+7*(size_t)cols,x,cols); }
        for(int r=nb*8;r<rows;r++) y[r]=vec_dot_bf16_f32(W+(size_t)r*cols,x,cols);
    } else {
        #pragma omp parallel for schedule(static)
        for(int r=0;r<rows;r++) y[r]=vec_dot_bf16_f32(W+(size_t)r*cols,x,cols);
    }
}
static double bench_mv_bf16_rb(const uint16_t*W,const float*x,float*y,int rows,int cols,int reps,int rb){
    mv_bf16_rb(y,W,x,rows,cols,rb);
    double t0=wall();
    for(int r=0;r<reps;r++) mv_bf16_rb(y,W,x,rows,cols,rb);
    return (wall()-t0)/reps;
}

int main(void){
    if(glm5_envi("BN_NUMA",1)){
        /* interleave over the CMG nodes only (Fugaku: nodes 4-7 hold the compute cores;
         * nodes 0-3 are the tiny assistant-core nodes). BN_MASK overrides. */
        unsigned long nodemask=(unsigned long)glm5_envi("BN_MASK",0xf0);
        syscall(SYS_set_mempolicy,MPOL_INTERLEAVE,&nodemask,(unsigned long)(8*sizeof nodemask));
    }
    int maxthr=1;
#ifdef _OPENMP
    maxthr=omp_get_max_threads();
#endif
    printf("== glm52_q2_kern_bench: maxthr=%d numa=%d sve=%d bytes\n",maxthr,glm5_envi("BN_NUMA",1),(int)svcntb());

    /* ---------- 1. dense BF16 GEMV scaling ---------- */
    struct { const char*name; int rows,cols; } SH[]={
        {"head ",12907,6144},{"q_a  ",2048,6144},{"o_prj",6144,1536},{"sh_w1",171,6144}
    };
    for(int s=0;s<4;s++){
        int rows=SH[s].rows, cols=SH[s].cols;
        uint16_t*W=glm5_amalloc((size_t)rows*cols*2);
        float*x=glm5_amalloc((size_t)cols*4),*y=glm5_amalloc((size_t)rows*4);
        fill_bf16(W,(size_t)rows*cols,7u+s);
        { uint32_t sd=99; for(int i=0;i<cols;i++) x[i]=frnd(&sd); }
        double bytes=(double)rows*cols*2;
        int reps=(int)(0.35e9/bytes)+3;
        int thrs[3]={12,24,maxthr};
        printf("bf16 %s [%5d x %4d]",SH[s].name,rows,cols);
        for(int ti=0;ti<3;ti++){
#ifdef _OPENMP
            omp_set_num_threads(thrs[ti]);
#endif
            double dt=bench_mv_bf16(W,x,y,rows,cols,reps);
            printf("  omp%-2d %7.1f us %6.1f GB/s",thrs[ti],dt*1e6,bytes/dt/1e9);
        }
#ifdef _OPENMP
        omp_set_num_threads(maxthr);
#endif
        if(glm5_envi("BN_RB",0)){
            double dt=bench_mv_bf16_rb(W,x,y,rows,cols,reps,1);
            printf("  rb1    %7.1f us %6.1f GB/s",dt*1e6,bytes/dt/1e9);
        }
        /* pinned pool */
        if(glm5_envi("BN_POOL",1)){
            glm5_pool*p=glm5_pool_create(maxthr); glm5_g_pool=p;
            double dt=bench_mv_bf16(W,x,y,rows,cols,reps);
            printf("  pool%-2d %7.1f us %6.1f GB/s",p->nthr,dt*1e6,bytes/dt/1e9);
            glm5_g_pool=NULL; glm5_pool_destroy(p);
        }
        printf("\n");
        glm5_afree(W);glm5_afree(x);glm5_afree(y);
    }

    /* ---------- 1b. decode-pattern qkv sequence: region overhead probe ----------
     * Mimics one layer's qkv: norm(6144) serial -> mv q_a [2048x6144] -> norm(2048)
     * serial -> mv q_b [1536x2048] -> mv kv_a [576x6144].  Compares the in-pattern
     * per-layer time against the sum of tight-loop GEMV times (=> per-region cost). */
    if(glm5_envi("BN_QKV",1)){
        uint16_t*Wqa=glm5_amalloc((size_t)2048*6144*2),*Wqb=glm5_amalloc((size_t)1536*2048*2),*Wka=glm5_amalloc((size_t)576*6144*2);
        float*xn=glm5_amalloc(6144*4),*ql=glm5_amalloc(2048*4),*qv=glm5_amalloc(1536*4),*kv=glm5_amalloc(576*4);
        fill_bf16(Wqa,(size_t)2048*6144,41); fill_bf16(Wqb,(size_t)1536*2048,42); fill_bf16(Wka,(size_t)576*6144,43);
        { uint32_t sd=5; for(int i=0;i<6144;i++) xn[i]=frnd(&sd); }
        double t_tight=bench_mv_bf16(Wqa,xn,ql,2048,6144,50)
                      +bench_mv_bf16(Wqb,ql,qv,1536,2048,50)
                      +bench_mv_bf16(Wka,xn,kv,576,6144,50);
        int reps=200; double t0=wall();
        for(int r=0;r<reps;r++){
            volatile float s=0; for(int i=0;i<6144;i++) s+=xn[i];          /* serial norm-ish */
            glm5_mv_bf16(ql,Wqa,xn,2048,6144);
            for(int i=0;i<2048;i++) ql[i]*=1.0000001f;                      /* serial norm-ish */
            glm5_mv_bf16(qv,Wqb,ql,1536,2048);
            glm5_mv_bf16(kv,Wka,xn,576,6144);
        }
        double t_pat=(wall()-t0)/reps;
        printf("qkv pattern: tight-sum %.1f us | in-pattern %.1f us  (overhead %.1f us/layer, ~%.1f us/region)\n",
               t_tight*1e6,t_pat*1e6,(t_pat-t_tight)*1e6,(t_pat-t_tight)*1e6/3);
        glm5_afree(Wqa);glm5_afree(Wqb);glm5_afree(Wka);glm5_afree(xn);glm5_afree(ql);glm5_afree(qv);glm5_afree(kv);
    }

    /* ---------- 2. mixed-IQ q8 kernels ---------- */
    pthread_once(&glm5_iq_lut_once,glm5_iq_init_luts);
    int rows=2048, cols=6144, nb=cols/256;
    glm5_iq_q8_block*xq=glm5_amalloc((size_t)nb*sizeof(*xq));
    float*x=glm5_amalloc((size_t)cols*4);
    { uint32_t sd=1234; for(int i=0;i<cols;i++) x[i]=frnd(&sd); }
    glm5_iq_quant_q8(xq,x,cols);
    { double t0=wall(); for(int r=0;r<1000;r++) glm5_iq_quant_q8(xq,x,cols);
      printf("iq_quant_q8 cols=%d: %.2f us\n",cols,(wall()-t0)/1000*1e6); }

    float*yref=glm5_amalloc((size_t)rows*4),*y1=glm5_amalloc((size_t)rows*4),*y2=glm5_amalloc((size_t)rows*4);
    for(int typ=0;typ<3;typ++){
        int qtype = typ==0?GLM5_IQ2_XS: typ==1?GLM5_IQ3_XXS:GLM5_IQ4_XS;
        const char*tn = typ==0?"IQ2_XS ":typ==1?"IQ3_XXS":"IQ4_XS ";
        size_t rb=dequant_row_size((uint32_t)qtype,cols);
        uint8_t*W=glm5_amalloc((size_t)rows*rb);
        #pragma omp parallel for schedule(static)
        for(int r=0;r<rows;r++){ uint32_t s=17u+(uint32_t)r*2654435761u+typ;
            uint8_t*row=W+(size_t)r*rb;
            for(size_t i=0;i<rb;i++) row[i]=(uint8_t)(lcg(&s)&0xff);
            /* keep fp16 d sane: overwrite each block's d with a small positive value */
            for(int b=0;b<nb;b++){ uint16_t d=0x2e66; /* fp16 0.1 */ memcpy(row+(rb/nb)*b,&d,2); }
        }
        glm5_tensor T={W,NULL,qtype,rows,cols,0,0};
        glm5_mv_ggml(yref,&T,x,rows,cols);
        /* v1 / v2 rows under OMP (same parallel shape as glm5_mv_iq_q8) */
        double t1=0,t2=0; int reps=40;
        for(int pass=0;pass<2;pass++){
            float*y = pass?y2:y1;
            double t0=wall();
            for(int rep=0;rep<reps;rep++){
                #pragma omp parallel for schedule(static)
                for(int r=0;r<rows;r++){
                    const uint8_t*w=W+(size_t)r*rb;
#if defined(__ARM_FEATURE_SVE)
                    if(pass){
                        if(qtype==GLM5_IQ2_XS) y[r]=glm5_iq2_xs_q8_row_v2((const block_iq2_xs*)w,xq,nb);
                        else if(qtype==GLM5_IQ3_XXS) y[r]=glm5_iq3_xxs_q8_row_v2((const block_iq3_xxs*)w,xq,nb);
                        else y[r]=glm5_iq4_xs_q8_row_v2((const block_iq4_xs*)w,xq,nb);
                    } else
#endif
                    {
                        if(qtype==GLM5_IQ2_XS) y[r]=glm5_iq2_xs_q8_row((const block_iq2_xs*)w,xq,nb);
                        else if(qtype==GLM5_IQ3_XXS) y[r]=glm5_iq3_xxs_q8_row((const block_iq3_xxs*)w,xq,nb);
                        else y[r]=glm5_iq4_xs_q8_row((const block_iq4_xs*)w,xq,nb);
                    }
                }
            }
            double dt=(wall()-t0)/reps;
            if(pass) t2=dt; else t1=dt;
        }
        double e1=0,e2=0,rn=0,mx1=0,mx2=0;
        for(int r=0;r<rows;r++){
            double d1=(double)y1[r]-yref[r], d2=(double)y2[r]-yref[r];
            e1+=d1*d1; e2+=d2*d2; rn+=(double)yref[r]*yref[r];
            if(fabs(d1)>mx1)mx1=fabs(d1); if(fabs(d2)>mx2)mx2=fabs(d2);
        }
        double gw=(double)rows*cols;
        printf("%s [%d x %d] v1 %8.1f us (%6.1f Gw/s) rel %.2e | v2 %8.1f us (%6.1f Gw/s) rel %.2e  speedup %.2fx\n",
               tn,rows,cols,t1*1e6,gw/t1/1e9,sqrt(e1/(rn+1e-30)),
               t2*1e6,gw/t2/1e9,sqrt(e2/(rn+1e-30)),t1/t2);
        glm5_afree(W);
    }
    glm5_afree(yref);glm5_afree(y1);glm5_afree(y2);glm5_afree(xq);glm5_afree(x);

    /* ---------- 3. absorbed-MLA decode attention ---------- */
    {
        glm5_model*m=glm5_acalloc(1,sizeof *m);
        m->cfg=glm5_default_config(); m->cfg.max_pos=2304; m->n_threads=maxthr;
        const glm5_config*c=&m->cfg;
        int KVC=glm5_kv_cache_dim(c), nown=6, kvb_stride=c->qk_nope_dim+c->v_head_dim;
        glm5_alloc_scratch(m,c->vocab);
        glm5_layer L; memset(&L,0,sizeof L);
        L.kv_cache=glm5_amalloc((size_t)c->max_pos*KVC*2);
        fill_bf16(L.kv_cache,(size_t)c->max_pos*KVC,555u);
        int wrows=nown*kvb_stride;
        L.wkv_b.w=glm5_amalloc((size_t)wrows*c->kv_lora*2);
        fill_bf16((uint16_t*)L.wkv_b.w,(size_t)wrows*c->kv_lora,777u);
        L.wkv_b.type=GLM5_BF16; L.wkv_b.rows=wrows; L.wkv_b.cols=c->kv_lora;
        float*q=glm5_amalloc((size_t)nown*c->qk_head_dim*4);
        { uint32_t sd=31; for(int i=0;i<nown*c->qk_head_dim;i++) q[i]=frnd(&sd); }
        float*kvtmp=glm5_amalloc((size_t)KVC*4);
        float*ab1=glm5_amalloc((size_t)nown*c->v_head_dim*4),*ab2=glm5_amalloc((size_t)nown*c->v_head_dim*4);
        float hmx1[64],hse1[64],hmx2[64],hse2[64];
        int NSns[2]={640,2048};
        for(int nsx=0;nsx<2;nsx++){
            int ns=NSns[nsx];
            int*sel=glm5_amalloc((size_t)ns*sizeof(int));
            for(int i=0;i<ns;i++) sel[i]=i;
            glm5_prefill_absorb_token(m,&L,q,ab1,sel,ns,hmx1,hse1,m->s_qabs,m->s_ctx,kvtmp,nown,0,0,1);
            glm5_absorb_token_par(m,&L,q,ab2,sel,ns,hmx2,hse2,nown);
            double emax=0;
            for(int hh=0;hh<nown;hh++){
                float i1=1.f/(hse1[hh]>0?hse1[hh]:1), i2=1.f/(hse2[hh]>0?hse2[hh]:1);
                for(int i=0;i<c->v_head_dim;i++){
                    double a=ab1[hh*c->v_head_dim+i]*i1, b=ab2[hh*c->v_head_dim+i]*i2;
                    double d=fabs(a-b)/(fabs(a)+1e-6); if(d>emax)emax=d;
                }
            }
            int r1=ns>1024?10:20, r2=200;
            double t0=wall(); for(int r=0;r<r1;r++) glm5_prefill_absorb_token(m,&L,q,ab1,sel,ns,hmx1,hse1,m->s_qabs,m->s_ctx,kvtmp,nown,0,0,1);
            double dt1=(wall()-t0)/r1;
            t0=wall(); for(int r=0;r<r2;r++) glm5_absorb_token_par(m,&L,q,ab2,sel,ns,hmx2,hse2,nown);
            double dt2=(wall()-t0)/r2;
            printf("absorb ns=%4d nown=%d: serial %8.1f us | par %7.1f us  speedup %5.1fx  relmax %.2e  (x78L: %.1f -> %.1f ms/tok)\n",
                   ns,nown,dt1*1e6,dt2*1e6,dt1/dt2,emax,dt1*78e3,dt2*78e3);
            glm5_afree(sel);
        }
    }
    printf("BENCH_DONE\n");
    return 0;
}
