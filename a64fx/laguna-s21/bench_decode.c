/* Single-node synthetic decode benchmark for the Laguna forward pass.
 * Reuses the runner's forward_token/attention/mlp on malloc'd synthetic weights
 * (ep_size=1 => all 256 experts present, i.e. worst-case per-rank expert load).
 * No uTofu, no staging -> fast kernel-tuning iteration.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *            -DLAGUNA_BENCH -I../../common -I../utofu-tests -o build/bench_decode \
 *            bench_decode.c -lm
 * Run:   OMP_NUM_THREADS=47 ./build/bench_decode [n_layers] [n_active] [iters]
 */
#define LAGUNA_BENCH
#include "laguna_s21_ep_runner.c"
#include <time.h>

/* Parallel first-touch fills so synthetic-weight pages distribute across CMGs,
 * matching the real runner's parallel arena copy (else bandwidth is under-counted). */
static uint16_t *bf16_buf(size_t n){ uint16_t*p=NULL; posix_memalign((void**)&p,256,n*2);
    #pragma omp parallel for schedule(static)
    for(long i=0;i<(long)n;i++){ uint32_t h=(uint32_t)i*2654435761u; p[i]=laguna_f32_to_bf16(((int)(h&0xffff)-32768)/32768.0f*0.05f);} return p; }
static uint32_t *u32_buf(size_t n){ uint32_t*p=NULL; posix_memalign((void**)&p,256,n*4);
    #pragma omp parallel for schedule(static)
    for(long i=0;i<(long)n;i++) p[i]=(uint32_t)i*2654435761u; return p; }
static uint16_t *scale_buf(size_t n){ uint16_t*p=NULL; posix_memalign((void**)&p,256,n*2);
    uint16_t s=laguna_f32_to_bf16(0.001f);
    #pragma omp parallel for schedule(static)
    for(long i=0;i<(long)n;i++) p[i]=s; return p; }
static float *f32_buf(size_t n){ float*p=NULL; posix_memalign((void**)&p,256,n*4);
    #pragma omp parallel for schedule(static)
    for(long i=0;i<(long)n;i++) p[i]=0.0f; return p; }
/* synthetic int8 W8 weight (parallel first-touch) */
static laguna_w8 w8_buf(int rows,int cols){ laguna_w8 w;
    posix_memalign((void**)&w.q,256,(size_t)rows*cols);
    posix_memalign((void**)&w.s,256,(size_t)rows*sizeof(float));
    #pragma omp parallel for schedule(static)
    for(long r=0;r<rows;r++){ w.s[r]=0.001f; int8_t*q=w.q+(size_t)r*cols;
        for(int c=0;c<cols;c++)q[c]=(int8_t)(((r*131+c)%255)-127); }
    return w; }

int main(int argc, char**argv){
    int n_layers = argc>1?atoi(argv[1]):48;
    int n_active = argc>2?atoi(argv[2]):1;   /* experts computed per MoE layer on this rank */
    int iters    = argc>3?atoi(argv[3]):20;
    int maxpos=256;
    laguna_model m; memset(&m,0,sizeof m);
    m.n_layers=n_layers; m.max_pos=maxpos; m.ep_rank=0; m.ep_size=1;
    m.embed=bf16_buf((size_t)LAGUNA_VOCAB*LAGUNA_HIDDEN);
    m.lm_head=w8_buf(LAGUNA_VOCAB, LAGUNA_HIDDEN);
    m.final_norm=bf16_buf(LAGUNA_HIDDEN);
    int H=LAGUNA_HIDDEN, hd=LAGUNA_HEAD_DIM;
    for(int L=0;L<n_layers;L++){
        laguna_layer*ly=&m.layers[L]; int full=(L%4==0);
        ly->is_sliding=!full; ly->num_heads=full?LAGUNA_FULL_HEADS:LAGUNA_SLIDING_HEADS; ly->is_moe=(L!=0);
        int nh=ly->num_heads;
        ly->q_proj=w8_buf(nh*hd, H);
        ly->k_proj=w8_buf(LAGUNA_KV_HEADS*hd, H);
        ly->v_proj=w8_buf(LAGUNA_KV_HEADS*hd, H);
        ly->o_proj=w8_buf(H, nh*hd);
        ly->g_proj=w8_buf(nh, H);
        ly->q_norm=bf16_buf(LAGUNA_HEAD_DIM); ly->k_norm=bf16_buf(LAGUNA_HEAD_DIM);
        ly->in_ln=bf16_buf(LAGUNA_HIDDEN); ly->post_ln=bf16_buf(LAGUNA_HIDDEN);
        if(!ly->is_moe){
            ly->dense_gate=w8_buf(LAGUNA_DENSE_INTER, H);
            ly->dense_up  =w8_buf(LAGUNA_DENSE_INTER, H);
            ly->dense_down=w8_buf(H, LAGUNA_DENSE_INTER);
        } else {
            ly->shared_gate=w8_buf(LAGUNA_SHARED_INTER, H);
            ly->shared_up  =w8_buf(LAGUNA_SHARED_INTER, H);
            ly->shared_down=w8_buf(H, LAGUNA_SHARED_INTER);
            ly->router_w=w8_buf(LAGUNA_EXPERTS, H);
            ly->router_bias=f32_buf(LAGUNA_EXPERTS);
            size_t gp=(size_t)LAGUNA_EXPERT_INTER*(LAGUNA_HIDDEN/8), gs=(size_t)LAGUNA_EXPERT_INTER*(LAGUNA_HIDDEN/32);
            size_t dp=(size_t)LAGUNA_HIDDEN*(LAGUNA_EXPERT_INTER/8), ds=(size_t)LAGUNA_HIDDEN*(LAGUNA_EXPERT_INTER/32);
            /* share one synthetic expert's buffers across all owned experts (timing only) */
            uint32_t*GP=u32_buf(gp),*UP=u32_buf(gp),*DP=u32_buf(dp);
            uint16_t*GS=scale_buf(gs),*US=scale_buf(gs),*DS=scale_buf(ds);
            for(int e=0;e<n_active && e<LAGUNA_EXPERTS;e++){
                ly->experts[e]=(laguna_expert){GP,UP,DP,GS,US,DS,1};
            }
        }
    }
    int hf=LAGUNA_ROPE_FULL_DIM/2, hs=LAGUNA_ROPE_SLIDING_DIM/2;
    m.full_cos=f32_buf((size_t)maxpos*hf); m.full_sin=f32_buf((size_t)maxpos*hf);
    m.swa_cos=f32_buf((size_t)maxpos*hs); m.swa_sin=f32_buf((size_t)maxpos*hs);
    laguna_build_rope_tables(&m);
    m.kv_layer_stride=(size_t)maxpos*LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM;
    m.kcache=calloc((size_t)n_layers*m.kv_layer_stride,2); m.vcache=calloc((size_t)n_layers*m.kv_layer_stride,2);

    laguna_scratch sc; scratch_alloc(&sc,maxpos);
    float*x=f32_buf(LAGUNA_HIDDEN);

    struct timespec t0,t1;
    /* warmup (touches weights) */
    for(int i=0;i<3;i++){ for(int j=0;j<LAGUNA_HIDDEN;j++)x[j]=0.01f; forward_token(&m,&sc,x,i,NULL,1); }
    extern double g_t_attn,g_t_mlp,g_t_norm; extern int g_prof; g_prof=1;
    g_t_attn=g_t_mlp=g_t_norm=0;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int i=0;i<iters;i++){ for(int j=0;j<LAGUNA_HIDDEN;j++)x[j]=0.01f; forward_token(&m,&sc,x,10+i,NULL,1); }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    printf("  phases/token: norm=%.2f attn=%.2f mlp=%.2f ms\n",g_t_norm*1e3/iters,g_t_attn*1e3/iters,g_t_mlp*1e3/iters);
    g_prof=0;
    /* layers only (no lm_head) */
    struct timespec t2,t3; clock_gettime(CLOCK_MONOTONIC,&t2);
    for(int i=0;i<iters;i++){ for(int j=0;j<LAGUNA_HIDDEN;j++)x[j]=0.01f; forward_token(&m,&sc,x,10+i,NULL,0); }
    clock_gettime(CLOCK_MONOTONIC,&t3);
    double dl=((t3.tv_sec-t2.tv_sec)+(t3.tv_nsec-t2.tv_nsec)*1e-9)/iters;
    /* lm_head only */
    clock_gettime(CLOCK_MONOTONIC,&t2);
    for(int i=0;i<iters;i++) laguna_matvec_i8(sc.logits,&m.lm_head,sc.n1,LAGUNA_VOCAB,LAGUNA_HIDDEN);
    clock_gettime(CLOCK_MONOTONIC,&t3);
    double dh=((t3.tv_sec-t2.tv_sec)+(t3.tv_nsec-t2.tv_nsec)*1e-9)/iters;
    printf("  breakdown: layers=%.2f ms  lm_head=%.2f ms\n",dl*1e3,dh*1e3);
    printf("layers=%d active=%d iters=%d: %.2f ms/token  %.2f tok/s  (SVE=%d threads=%d)\n",
           n_layers,n_active,iters,dt*1e3/iters,iters/dt,
#if defined(__ARM_FEATURE_SVE)
           1,
#else
           0,
#endif
#ifdef _OPENMP
           omp_get_max_threads());
#else
           1);
#endif
    return 0;
}
