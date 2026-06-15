/*
 * m3_selftest.c - single-process native validation of the M3 forward graph's
 * TP + MSA code paths (no MPI/uTofu, so it runs on any native A64FX node even
 * while the multi-rank topo helper is down).
 *
 * Checks:
 *   1. ep_size=1 full model, MSA ON with small block/topk so selection ENGAGES
 *      cheaply (sparse layers attend a subset) -> runs NaN-free, prints how many
 *      positions were selected vs full at the last pos.
 *   2. ep_size=4 with M3_TP=1 (TP_ATTN/SHARED/FFN/HEAD shards active): exercises
 *      the sharded buffer sizing + partial-output paths without crashing/NaN.
 *      (Numerical reconstruction needs the real cross-rank all-reduce -> uTofu run;
 *       here ar_cb is NULL so partials are used as-is — this is a SAFETY check only.)
 *
 * Build (native): fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *                     -I <repo> -o m3_selftest a64fx/m3/m3_selftest.c -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define M3_IMPL
#include "m3.h"
#include "m3_impl.h"

static int run_case(const char*name,int ep_rank,int ep_size,int tp,int small_msa){
    setenv("M3_MSA","1",1);
    setenv("M3_TP", tp?"1":"0",1);
    m3_config c=m3_default_config();
    c.n_layers=6;                 /* 3 dense + 3 MoE+MSA */
    c.n_experts=8; c.n_active=4;
    c.max_pos=512;
    if(small_msa){ c.msa_block_size=32; c.msa_topk_blocks=2; c.msa_local_block=1; c.msa_init_block=0; }
    int prefill=200, decode=8;    /* pos up to 207 > keep*block (3*32=96) -> MSA engages */
    m3_model*m=m3_alloc_synth(c,ep_rank,ep_size,1,1);
    if(!m){ printf("[%s] alloc failed\n",name); return 1; }
    m3_sm=0xD3F00D;
    float*x=malloc((size_t)c.hidden*4);
    int nan=0, argmax=0;
    for(int p=0;p<prefill;p++){ for(int i=0;i<c.hidden;i++) x[i]=(float)(m3_sm_next()*2.0-1.0);
        argmax=m3_forward_token(m,x,p); for(int i=0;i<c.hidden;i++) if(!(x[i]==x[i])) nan++; }
    /* probe MSA selection at the last prefill position on a sparse layer */
    int last=prefill-1, keep=c.msa_topk_blocks+c.msa_local_block+c.msa_init_block;
    int nblk=last/c.msa_block_size+1;
    int sel_probe=-1;
    if(c.n_layers>c.n_dense_layers){ /* re-run select on sparse layer 3 to read nsel */
        m3_layer*L=&m->layers[c.n_dense_layers];
        float*xn=malloc((size_t)c.hidden*4); for(int i=0;i<c.hidden;i++) xn[i]=0.02f;
        sel_probe=m3_msa_select(m,L,xn,last,1,m->s_blk_sel); free(xn);
    }
    for(int g=0;g<decode;g++){ int pos=prefill+g; for(int i=0;i<c.hidden;i++) x[i]=(float)(m3_sm_next()*2.0-1.0);
        argmax=m3_forward_token(m,x,pos); for(int i=0;i<c.hidden;i++) if(!(x[i]==x[i])) nan++; }
    printf("[%s] ep=%d/%d tp=%d  nblk@%d=%d keep=%d  MSA_selected=%d (full=%d, engaged=%s)  argmax=%d  NaNs=%d  %s\n",
           name, ep_rank, ep_size, tp, last, nblk, keep, sel_probe, last+1,
           (sel_probe>=0 && sel_probe<last+1)?"YES":"no", argmax, nan, nan==0?"OK":"FAIL");
    free(x); m3_free(m);
    return nan==0?0:1;
}

int main(void){
    int rc=0;
    rc |= run_case("full+MSA",      0,1, 0, 1);   /* ep1, MSA engages */
    rc |= run_case("TP+EP+MSA",     1,4, 1, 1);   /* ep4 rank1, TP shards active (safety) */
    rc |= run_case("realMSAparams", 0,1, 0, 0);   /* ep1, real topk16/block128 (no engage at maxpos512) */
    printf("%s\n", rc==0?"ALL OK (NaN-free)":"FAIL");
    return rc;
}
