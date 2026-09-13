/* Real-weight, decode-only GLM-5.3F layer-45 sparse-attention/cache oracle. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include <arm_sve.h>
#include <omp.h>
#include "glm53f_expert_kern.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { H=4096, Q_A=1536, HEADS=64, KD=256, VD=256, LATENT=512,
       IHEADS=32, IDIM=128, KPOOL=4, INDEX_TOPK=2048 };

static double now_sec(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void *alloc256(size_t n) { void *p=NULL; if(posix_memalign(&p,256,n))return NULL; return p; }
static void *tensor(glm53f_st_context *st,const char *name,size_t *bytes) {
    const st_tensor_info *t=glm53f_st_find(st,name,NULL); void *p;
    if(!t){fprintf(stderr,"missing %s\n",name);return NULL;} p=alloc256(t->nbytes);
    if(!p||glm53f_st_read(st,name,0,p,t->nbytes)){fprintf(stderr,"read failed %s\n",name);free(p);return NULL;}
    if(bytes)*bytes=t->nbytes; return p;
}
static void mv_bf16(float*y,const uint16_t*w,const float*x,int rows,int cols){
#pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++) y[r]=glm53f_dot_bf16_sve(w+(size_t)r*cols,x,cols);
}
static void mv_fp8(float*y,const uint8_t*w,const float*s,const float*x,int rows,int cols){
    glm53f_mv_fp8_block128_bits(y,w,s,x,rows,cols);
}
static int same_bits(const float*a,const float*b,int n){return !memcmp(a,b,(size_t)n*sizeof(float));}

int main(int argc,char**argv){
    int tokens=argc>2?atoi(argv[2]):9, selected_n[2], stable=1;
    int index_only=getenv("GLM53F_INDEX_ONLY")!=NULL;
    int skip_direct=getenv("GLM53F_SKIP_DIRECT")!=NULL;
    char n[256]; size_t z;
    glm53f_st_context*st; uint8_t *qa,*qb,*kva,*op; float *qas,*qbs,*kvas,*ops;
    uint16_t *qan,*kvan,*kvb,*wk,*knw,*knb,*gate,*ape,*wqb,*wp;
    float *hidden,*qres,*query,*latent,*key,*gcache,*iq,*iw,*pool,*attn,*ref_out,*out[2]; int *selected[2];
    if(argc<2||tokens<1||tokens>((index_only||skip_direct)?4096:64)){fprintf(stderr,"usage: %s MODEL_DIR [tokens=9]\n",argv[0]);return 2;}
    st=glm53f_st_open(argv[1]);if(!st)return 2;
#define L(S) do{snprintf(n,sizeof n,"model.language_model.layers.45.self_attn.%s",S);}while(0)
#define GET(V,S,T) do{L(S);V=(T*)tensor(st,n,&z);if(!(V))return 2;}while(0)
    GET(qa,"q_a_proj.weight",uint8_t); GET(qas,"q_a_proj.weight_scale_inv",float);
    GET(qan,"q_a_layernorm.weight",uint16_t); GET(qb,"q_b_proj.weight",uint8_t);
    GET(qbs,"q_b_proj.weight_scale_inv",float); GET(kva,"kv_a_proj_with_mqa.weight",uint8_t);
    GET(kvas,"kv_a_proj_with_mqa.weight_scale_inv",float); GET(kvan,"kv_a_layernorm.weight",uint16_t);
    GET(kvb,"kv_b_proj.weight",uint16_t);
    GET(op,"o_proj.weight",uint8_t); GET(ops,"o_proj.weight_scale_inv",float);
    GET(wk,"indexer.wk.weight",uint16_t); GET(knw,"indexer.k_norm.weight",uint16_t);
    GET(knb,"indexer.k_norm.bias",uint16_t); GET(gate,"indexer.index_kpool_compress_gate",uint16_t);
    GET(ape,"indexer.index_kpool_compress_ape",uint16_t); GET(wqb,"indexer.wq_b.weight",uint16_t);
    GET(wp,"indexer.weights_proj.weight",uint16_t);
#undef GET
#undef L
    glm53f_st_close(st);
    hidden=alloc256((size_t)tokens*H*4);qres=alloc256(Q_A*4);query=alloc256((size_t)HEADS*KD*4);
    latent=alloc256((size_t)tokens*LATENT*4);key=alloc256((size_t)tokens*IDIM*4);
    gcache=alloc256((size_t)tokens*IDIM*4);iq=alloc256((size_t)IHEADS*IDIM*4);iw=alloc256(IHEADS*4);
    pool=alloc256((size_t)((tokens/KPOOL)+1)*IDIM*4);attn=alloc256((size_t)HEADS*VD*4);ref_out=alloc256(H*4);
    out[0]=alloc256(H*4);out[1]=alloc256(H*4);selected[0]=alloc256((INDEX_TOPK+KPOOL)*sizeof(int));selected[1]=alloc256((INDEX_TOPK+KPOOL)*sizeof(int));
    if(!hidden||!qres||!query||!latent||!key||!gcache||!iq||!iw||!pool||!attn||!ref_out||!out[0]||!out[1]||!selected[0]||!selected[1])return 2;
    for(int t=0;t<tokens;t++)for(int i=0;i<H;i++) hidden[(size_t)t*H+i]=(float)(((i*17+t*29+3)%251)-125)/125.0f;
    /* Cache exactly the compressed latent and indexer K/gate values. */
    for(int t=0;t<tokens;t++){
        float rawk[IDIM]; const float*x=hidden+(size_t)t*H;
        mv_fp8(latent+(size_t)t*LATENT,kva,kvas,x,LATENT,H);
        glm53f_rmsnorm_bf16(latent+(size_t)t*LATENT,latent+(size_t)t*LATENT,kvan,LATENT,1e-5f);
        mv_bf16(rawk,wk,x,IDIM,H); glm53f_layernorm_bf16(key+(size_t)t*IDIM,rawk,knw,knb,IDIM,1e-5f);
        mv_bf16(gcache+(size_t)t*IDIM,gate,x,IDIM,H);
    }
    double elapsed[2];
    for(int pass=0;pass<2;pass++){
        const float*x=hidden+(size_t)(tokens-1)*H; double t0=now_sec();
        mv_fp8(qres,qa,qas,x,Q_A,H); glm53f_rmsnorm_bf16(qres,qres,qan,Q_A,1e-5f);
        mv_fp8(query,qb,qbs,qres,HEADS*KD,Q_A); mv_bf16(iq,wqb,qres,IHEADS*IDIM,Q_A); mv_bf16(iw,wp,x,IHEADS,H);
        /* APE is BF16 in the checkpoint; convert it before selection. */
        float apef[KPOOL*IDIM]; for(int i=0;i<KPOOL*IDIM;i++)apef[i]=glm53f_bf16_to_f32(ape[i]);
        selected_n[pass]=glm53f_index_select_decode(pool,selected[pass],iq,iw,key,gcache,
                apef,tokens,KPOOL,INDEX_TOPK,IHEADS,IDIM);
        if(!index_only){
            if(pass==0&&!skip_direct){
                glm53f_mla_selected_bf16(attn,query,latent,kvb,selected[pass],selected_n[pass],HEADS,KD,VD,LATENT);
                mv_fp8(ref_out,op,ops,attn,H,HEADS*VD);
                t0=now_sec();
            }
            if(glm53f_mla_absorbed_sve(attn,query,latent,kvb,selected[pass],selected_n[pass],HEADS,KD,VD,LATENT))return 2;
            mv_fp8(out[pass],op,ops,attn,H,HEADS*VD);
        } else memset(out[pass],0,H*sizeof(float));
        elapsed[pass]=now_sec()-t0;
    }
    stable=selected_n[0]==selected_n[1]&&!memcmp(selected[0],selected[1],(size_t)selected_n[0]*sizeof(int))&&same_bits(out[0],out[1],H);
    double ss=0,se=0,sr=0;int finite=1;for(int i=0;i<H;i++){double d=(index_only||skip_direct)?0.0:(double)out[0][i]-ref_out[i];ss+=(double)out[0][i]*out[0][i];se+=d*d;if(!index_only&&!skip_direct)sr+=(double)ref_out[i]*ref_out[i];finite&=isfinite(out[0][i]);}
    double rel=(index_only||skip_direct)?0:sqrt(se/(sr+1e-30));stable&=index_only||skip_direct||rel<2e-4;
    printf("GLM53F_SPARSE_ATTN mode=%s tokens=%d selected=%d excluded=%d first=%d last=%d repeat=%s finite=%s direct_rel_l2=%.9g rms=%.9g sec=%.6f %s\n",
           index_only?"INDEX":(skip_direct?"SVE_BENCH":"SVE_ABSORBED"),tokens,selected_n[0],tokens-selected_n[0],selected[0][0],selected[0][selected_n[0]-1],stable?"BIT_EXACT":"FAIL",finite?"YES":"NO",rel,sqrt(ss/H),elapsed[1],stable&&finite?"PASS":"FAIL");
    return stable&&finite?0:1;
}
