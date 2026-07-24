/* Long-context prefill attention benchmark: times attention_full_flash (the
 * O(N^2) term that dominates long-context prefill) at a range of context depths,
 * so LAGUNA_KB can be tuned without 25-minute 12-node runs.
 *
 * Build one binary per KB, e.g.
 *   for kb in 32 64 128 256; do fcc ... -DLAGUNA_KB=$kb -o attn_bench_$kb attn_bench.c; done
 */
#define LAGUNA_BENCH
#include "laguna_s21_ep_runner.c"

static uint64_t rs=0x9E3779B97F4A7C15ull;
static float frand(void){ rs=rs*6364136223846793005ull+1442695040888963407ull;
    return (float)((int)((rs>>33)%2000)-1000)/1000.0f; }
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char **argv) {
    int C = argc>1?atoi(argv[1]):256;
    int hd=LAGUNA_HEAD_DIM, nh=LAGUNA_FULL_HEADS, kvs=LAGUNA_KV_HEADS*hd;
    int depths[] = {2048, 8192, 32768, 65536};
    printf("LAGUNA_KB=%d  C=%d  (full-attention layer, nh=%d)\n", LAGUNA_KB, C, nh);
    for (size_t di=0; di<sizeof depths/sizeof *depths; ++di) {
        int pos0=depths[di], maxpos=pos0+C+8;
        laguna_model m; memset(&m,0,sizeof m);
        m.n_layers=1; m.max_pos=maxpos;
        laguna_layer *ly=&m.layers[0];
        ly->is_sliding=0; ly->num_heads=nh;
        uint16_t *qn=malloc(hd*2),*kn=malloc(hd*2);
        for(int i=0;i<hd;i++){qn[i]=laguna_f32_to_bf16(1.0f);kn[i]=laguna_f32_to_bf16(1.0f);}
        ly->q_norm=qn; ly->k_norm=kn;
        int hf=LAGUNA_ROPE_FULL_DIM/2,hs=LAGUNA_ROPE_SLIDING_DIM/2;
        m.full_cos=malloc((size_t)maxpos*hf*4);m.full_sin=malloc((size_t)maxpos*hf*4);
        m.swa_cos=malloc((size_t)maxpos*hs*4);m.swa_sin=malloc((size_t)maxpos*hs*4);
        laguna_build_rope_tables(&m);
        m.kv_cap[0]=maxpos; m.kv_off[0]=0;
        size_t kvel=(size_t)maxpos*kvs;
        m.kcache=malloc(kvel*2); m.vcache=malloc(kvel*2);
        for(size_t i=0;i<kvel;i++){m.kcache[i]=laguna_f32_to_bf16(frand()*0.3f);
                                   m.vcache[i]=laguna_f32_to_bf16(frand()*0.3f);}
        laguna_scratch sc; scratch_alloc(&sc,maxpos);
        size_t qe=(size_t)C*nh*hd, kve=(size_t)C*kvs;
        float *Q=malloc(qe*4),*K=malloc(kve*4),*V=malloc(kve*4),*G=malloc((size_t)C*nh*4),*AO=malloc(qe*4);
        for(size_t i=0;i<qe;i++)Q[i]=frand();
        for(size_t i=0;i<kve;i++){K[i]=frand();V[i]=frand();}
        for(size_t i=0;i<(size_t)C*nh;i++)G[i]=frand()*0.5f;

        attention_full_flash(&m,ly,&sc,0,pos0,C,nh,Q,K,V,G,AO);   /* warm */
        int it = pos0>=32768?2:4;
        double t0=now_s();
        for(int i=0;i<it;i++) attention_full_flash(&m,ly,&sc,0,pos0,C,nh,Q,K,V,G,AO);
        double dt=(now_s()-t0)/it;
        /* qk dots + av axpys, both ~pos0+C/2 keys per (query, head) */
        double keys=(double)pos0+C/2.0;
        double gflop=2.0*2.0*C*nh*keys*hd/1e9;   /* qk + av, 2 flop each */
        printf("  pos0=%6d   %8.1f ms   %6.1f GFLOP/s\n", pos0, dt*1e3, gflop/dt);
        free(Q);free(K);free(V);free(G);free(AO);free(m.kcache);free(m.vcache);
    }
    return 0;
}
