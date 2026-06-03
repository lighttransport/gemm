/* Workstream C — C0 de-risk: is the 27B decode attention position/memory-bound
 * (so sharding KV positions across ranks cuts per-rank wall-clock ~N×), or
 * head-bound (so it doesn't)?
 *
 * 27B TP=11 decode does NOT use the qpkd K_DP kernel (that needs n_heads==16;
 * 27B shards 24 heads to 2-3/rank), so attention runs through tf_attn_worker:
 * threads parallelise over HEADS (h_start/h_count per tid), each thread serial
 * over all seq_len positions. Today a rank runs its ~3 local heads on 3 of 48
 * threads, each streaming the full S positions. Position-sharding (Workstream C)
 * would instead run ALL 24 heads (24 of 48 threads, still no contention) over
 * only S/N positions. If tf_attn_worker is ~linear in S, the per-rank attention
 * wall-clock drops ~N× — the whole premise of C.
 *
 * This bench calls the REAL tf_attn_worker over a 48-thread pool, partitioning
 * heads with the SAME h_start/h_count formula as the decode loop (transformer.h
 * ~8255), on a synthetic F16 KV cache (the 27B production kv_dtype). No model /
 * staging / multinode needed — single A64FX node.
 *
 * Decisive comparison at ctx=8192, N=11:
 *   A) n_heads=3,  S=8192  (today's rank work)
 *   B) n_heads=24, S=745   (pos-shard rank work)   expect B ~ A/11  -> GO
 *   C) n_heads=24, S=8192  (24-head dispatch at same S; ~C? vs A isolates head cost)
 *   D) n_heads=3,  S=745   (linear-in-S check; A/D ~ 11 if linear)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + (double)t.tv_nsec*1e-9; }

/* 27B attention shape */
#define HD   256
#define NKV  4
#define GQA  6      /* 24 global q-heads / 4 kv-heads */
#define NTH  48
#define SMAX 8192

static float  *g_q, *g_att, *g_xb2;
static uint16_t *g_kc, *g_vc;
static size_t g_kv_dim;

static uint16_t f2h(float v){ __fp16 h = (__fp16)v; uint16_t u; memcpy(&u,&h,2); return u; }

/* One config: n_heads partitioned across NTH threads, each thread runs
 * tf_attn_worker for its heads over seq_len positions. Returns median per-iter ms. */
static double run_cfg(int n_heads, int seq_len, int iters, int warm){
    double out_ms = 0.0;
    int max_seq = SMAX + 64;
    #pragma omp parallel num_threads(NTH)
    {
        int tid = omp_get_thread_num();
        int h_per = n_heads / NTH;
        int h_extra = n_heads % NTH;
        int h_start = tid*h_per + (tid < h_extra ? tid : h_extra);
        int h_count = h_per + (tid < h_extra ? 1 : 0);
        tf_attn_task at = {
            .q = g_q, .att = g_att, .xb2 = g_xb2,
            .key_cache = g_kc, .value_cache = g_vc,
            .key_scales = NULL, .value_scales = NULL,
            .head_start = h_start, .head_end = h_start + h_count,
            .head_dim = HD, .kv_dim = (int)g_kv_dim, .gqa_ratio = GQA,
            .seq_len = seq_len, .max_seq_len = max_seq,
            .qhead_base = 0, .kv_head_base = 0, .n_kv_heads = NKV,
            .kv_dtype = TF_KV_DTYPE_F16, .k_transposed = 0, .k_dp = 0,
            .skip_qk = 0, .skip_av = 0, .scale = 1.0f/sqrtf((float)HD),
        };
        for (int it=0; it<warm; it++){
            if (h_count>0){ memset(g_xb2 + (size_t)h_start*HD, 0, (size_t)h_count*HD*sizeof(float));
                            tf_attn_worker(&at); }
            #pragma omp barrier
        }
        #pragma omp barrier
        double t0=0;
        #pragma omp master
        t0 = now_s();
        #pragma omp barrier
        for (int it=0; it<iters; it++){
            if (h_count>0){ memset(g_xb2 + (size_t)h_start*HD, 0, (size_t)h_count*HD*sizeof(float));
                            tf_attn_worker(&at); }
            #pragma omp barrier
        }
        #pragma omp master
        { double t1=now_s(); out_ms = (t1-t0)/iters*1000.0; }
    }
    return out_ms;
}

/* Correctness: compare the attn_pp kernels + online-softmax merge (the exact
 * sequence the persistent worker runs) against the monolithic tf_attn_worker,
 * for the 27B sharded shape. Emulates the nt-way position partition serially. */
static int check_pp(int n_heads, int seq_len, int qhead_base, int kv_head_base){
    int nt = NTH, max_seq = SMAX + 64;
    float scale = 1.0f/sqrtf((float)HD);
    float *att = aligned_alloc(256,(size_t)n_heads*max_seq*sizeof(float));
    float *pmax= aligned_alloc(256,(size_t)nt*n_heads*sizeof(float));
    float *psum= aligned_alloc(256,(size_t)nt*n_heads*sizeof(float));
    float *avt = aligned_alloc(256,(size_t)nt*n_heads*HD*sizeof(float));
    float *xref= aligned_alloc(256,(size_t)n_heads*HD*sizeof(float));
    float *xpp = aligned_alloc(256,(size_t)n_heads*HD*sizeof(float));

    /* reference: EXACT scalar monolithic softmax (exact expf, sequential sum).
     * The PP path also uses exact expf (transformer.h:8791), so this isolates
     * kernel math (reduction-order only) — unlike tf_attn_worker which uses the
     * fast-exp SVE polynomial (~1e-4 error) and would muddy the comparison. */
    for(int h=0;h<n_heads;h++){
        int kv_h=(qhead_base+h)/GQA - kv_head_base;
        const float *q_h=g_q+(size_t)h*HD;
        float *sc=(float*)att;  /* reuse att as score scratch for this head */
        float mx=-INFINITY;
        for(int p=0;p<seq_len;p++){
            const uint16_t *k=g_kc+(size_t)p*g_kv_dim+(size_t)kv_h*HD;
            float s=0; for(int d=0;d<HD;d++) s+=q_h[d]*tf_f16_to_f32(k[d]);
            s*=scale; sc[p]=s; if(s>mx)mx=s;
        }
        float sum=0; for(int p=0;p<seq_len;p++){ sc[p]=expf(sc[p]-mx); sum+=sc[p]; }
        float *o=xref+(size_t)h*HD; for(int d=0;d<HD;d++)o[d]=0;
        for(int p=0;p<seq_len;p++){ const uint16_t *v=g_vc+(size_t)p*g_kv_dim+(size_t)kv_h*HD;
            for(int d=0;d<HD;d++) o[d]+=sc[p]*tf_f16_to_f32(v[d]); }
        float inv=1.0f/sum; for(int d=0;d<HD;d++) o[d]*=inv;
    }

    /* PP path: QK + partial-max per thread */
    for(int tid=0;tid<nt;tid++){
        long p_lo=(long)tid*seq_len/nt, p_hi=(long)(tid+1)*seq_len/nt;
        if(p_hi>p_lo)
            tf_qk_chunk_pp_f16(g_kc,g_q,att,(int)p_lo,(int)p_hi,n_heads,
                               qhead_base,GQA,kv_head_base,HD,(int)g_kv_dim,max_seq,scale);
        for(int h=0;h<n_heads;h++){ float mx=-INFINITY;
            for(long p=p_lo;p<p_hi;p++){ float v=att[(size_t)h*max_seq+p]; if(v>mx)mx=v; }
            pmax[(size_t)tid*n_heads+h]=mx; }
    }
    float gmax[64];
    for(int h=0;h<n_heads;h++){ float g=-INFINITY;
        for(int t=0;t<nt;t++){ float v=pmax[(size_t)t*n_heads+h]; if(v>g)g=v; } gmax[h]=g; }
    /* exp+sum + AV per thread */
    for(int tid=0;tid<nt;tid++){
        long p_lo=(long)tid*seq_len/nt, p_hi=(long)(tid+1)*seq_len/nt;
        for(int h=0;h<n_heads;h++){ float s=0;
            for(long p=p_lo;p<p_hi;p++){ float e=expf(att[(size_t)h*max_seq+p]-gmax[h]);
                att[(size_t)h*max_seq+p]=e; s+=e; }
            psum[(size_t)tid*n_heads+h]=s; }
        float *mt=avt+(size_t)tid*n_heads*HD;
        memset(mt,0,(size_t)n_heads*HD*sizeof(float));
        if(p_hi>p_lo)
            tf_av_chunk_pp_f16(g_vc,att,mt,(int)p_lo,(int)p_hi,n_heads,
                               qhead_base,GQA,kv_head_base,HD,(int)g_kv_dim,max_seq);
    }
    float ginv[64];
    for(int h=0;h<n_heads;h++){ float s=0;
        for(int t=0;t<nt;t++) s+=psum[(size_t)t*n_heads+h]; ginv[h]= s>0?1.0f/s:0.0f; }
    for(int h=0;h<n_heads;h++) for(int d=0;d<HD;d++){ float acc=0;
        for(int t=0;t<nt;t++) acc+=avt[(size_t)t*n_heads*HD+(size_t)h*HD+d];
        xpp[(size_t)h*HD+d]=acc*ginv[h]; }
    double maxabs=0,maxrel=0;
    for(int i=0;i<n_heads*HD;i++){ double a=xref[i],b=xpp[i],e=fabs(a-b);
        if(e>maxabs)maxabs=e; double r=fabs(a)>1e-6?e/fabs(a):0; if(r>maxrel)maxrel=r; }
    int pass = maxrel < 1e-3;
    printf("check_pp nh=%d S=%-5d qhb=%d kvb=%d: max|d|=%.3e maxrel=%.3e  %s\n",
           n_heads,seq_len,qhead_base,kv_head_base,maxabs,maxrel,pass?"PASS":"FAIL");
    free(att);free(pmax);free(psum);free(avt);free(xref);free(xpp);
    return pass;
}

int main(void){
    g_kv_dim = (size_t)NKV*HD;
    int max_seq = SMAX + 64;
    g_q   = aligned_alloc(256, (size_t)24*HD*sizeof(float));
    g_att = aligned_alloc(256, (size_t)24*max_seq*sizeof(float));
    g_xb2 = aligned_alloc(256, (size_t)24*HD*sizeof(float));
    g_kc  = aligned_alloc(256, (size_t)SMAX*g_kv_dim*sizeof(uint16_t));
    g_vc  = aligned_alloc(256, (size_t)SMAX*g_kv_dim*sizeof(uint16_t));
    for (int i=0;i<24*HD;i++) g_q[i] = 0.02f*((i%11)-5);
    for (size_t i=0;i<(size_t)SMAX*g_kv_dim;i++){
        g_kc[i] = f2h(0.015f*((int)(i%13)-6));
        g_vc[i] = f2h(0.010f*((int)(i%17)-8));
    }
    printf("# C0 attn-bench  HD=%d NKV=%d GQA=%d NTH=%d kv=F16\n", HD,NKV,GQA,NTH);
    printf("# --- attn_pp kernel correctness vs monolithic tf_attn_worker ---\n");
    int ok = 1;
    ok &= check_pp(3, 8192, 0, 0);   /* rank0: heads 0-2 -> kv_h 0,0,0 */
    ok &= check_pp(2, 8192, 6, 0);   /* a rank: heads 6,7 -> kv_h 1,1 (nonzero kv map) */
    ok &= check_pp(3, 4096, 0, 0);
    ok &= check_pp(2, 1, 6, 0);      /* first decode token (seq_len=1) */
    ok &= check_pp(3, 745, 0, 0);
    printf("# attn_pp correctness: %s\n\n", ok?"ALL PASS":"FAILED");
    printf("# cfg            n_heads  S      per_iter_ms   (16 layers => *16 ms/token attn)\n");

    struct { const char*name; int nh, S; } cfgs[] = {
        {"A today@8192   ", 3, 8192},
        {"B posN11@8192  ", 24, 745},
        {"C 24h@8192     ", 24, 8192},
        {"D today@745    ", 3, 745},
        {"A2 2h@8192     ", 2, 8192},
        {"E today@4096   ", 3, 4096},
        {"F posN11@4096  ", 24, 373},
        {"G today@2048   ", 3, 2048},
        {"H 16h@8192     ", 16, 8192},
    };
    int nc = (int)(sizeof(cfgs)/sizeof(cfgs[0]));
    for (int i=0;i<nc;i++){
        int iters = cfgs[i].S >= 4096 ? 120 : 300;
        double ms = run_cfg(cfgs[i].nh, cfgs[i].S, iters, 20);
        printf("%-16s  %3d   %5d   %9.4f    %8.2f\n",
               cfgs[i].name, cfgs[i].nh, cfgs[i].S, ms, ms*16);
        fflush(stdout);
    }
    printf("\n# verdict: B/A ratio = per-rank attn speedup from pos-sharding @8192 (N=11)\n");
    return 0;
}
