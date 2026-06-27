/* Gemma4-assistant MTP draft + speculative decode (12B BF16, A64FX).
 *
 * Reference: ggml-org/llama.cpp PR #23398 (build_gemma4_assistant). The MTP is a
 * 4-block mini-transformer @1024-dim that predicts future tokens from the TARGET's
 * last hidden state. Each block has q-only attention that READS the target's KV
 * cache (no own k/v): SWA blocks (0,1,2) share target layer n-2=46, the full block
 * (3) shares target layer n-1=47. Fusion: concat(scale(main_embd,sqrt(3840)),
 * hidden) -> pre_projection -> 1024. Recurse: h_next = post_projection(cur).
 *
 * Draft K tokens, then ONE batched target verify (greedy-exact). Measures acceptance
 * alpha vs the target's greedy and end-to-end tok/s. Drop-in for mini_spec's PLD. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static int argmax(const float*v,int n){ int b=0; float m=v[0]; for(int i=1;i<n;i++) if(v[i]>m){m=v[i];b=i;} return b; }
static inline float bf2f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,4); return f; }

/* ---- MTP weights ---- */
#define MTP_NL 4
typedef struct {
    const uint16_t *attn_q, *attn_output, *ffn_gate, *ffn_up, *ffn_down; /* bf16 */
    const float *attn_norm, *attn_q_norm, *post_attn_norm, *ffn_norm, *post_ffw_norm, *out_scale; /* f32 */
    int hd, qdim;            /* per-block head_dim (256 swa / 512 full), qdim=16*hd */
    int is_swa, share_layer; /* target KV layer */
} mtp_block;
typedef struct {
    mtp_block blk[MTP_NL];
    const uint16_t *pre_proj;   /* [7680,1024] bf16  (in=7680, out=1024) */
    const uint16_t *post_proj;  /* [1024,3840] bf16 */
    const uint16_t *tok_embd;   /* [1024,262144] bf16 (tied head) */
    const float    *out_norm;   /* [1024] f32 */
    int D, V, NH;               /* D=1024 working dim, V=vocab, NH=16 heads */
} mtp_model;

static const void* find_t(gguf_context*g,const char*name){
    for(uint64_t i=0;i<g->n_tensors;i++){ const char*nm=gguf_tensor_name(g,(int)i);
        if(nm&&!strcmp(nm,name)) return gguf_tensor_data(g,(int)i); }
    fprintf(stderr,"MTP: missing tensor %s\n",name); return NULL;
}
static int load_mtp(mtp_model*M,const char*path,int n_layers_target){
    gguf_context*g=gguf_open(path,0); if(!g){fprintf(stderr,"MTP open fail\n");return 0;} /* eager RAM (small, avoids slow /home mmap faults) */
    M->D=1024; M->V=262144; M->NH=16;
    M->pre_proj =find_t(g,"nextn.pre_projection.weight");
    M->post_proj=find_t(g,"nextn.post_projection.weight");
    M->tok_embd =find_t(g,"token_embd.weight");
    M->out_norm =find_t(g,"output_norm.weight");
    char nm[96];
    for(int l=0;l<MTP_NL;l++){ mtp_block*b=&M->blk[l];
        #define T(field,suf) (sprintf(nm,"blk.%d." suf,l), b->field=find_t(g,nm))
        T(attn_norm,"attn_norm.weight"); T(attn_q,"attn_q.weight"); T(attn_q_norm,"attn_q_norm.weight");
        T(attn_output,"attn_output.weight"); T(post_attn_norm,"post_attention_norm.weight");
        T(ffn_norm,"ffn_norm.weight"); T(ffn_gate,"ffn_gate.weight"); T(ffn_up,"ffn_up.weight");
        T(ffn_down,"ffn_down.weight"); T(post_ffw_norm,"post_ffw_norm.weight"); T(out_scale,"layer_output_scale.weight");
        #undef T
        b->is_swa = (l!=3);                 /* blocks 0,1,2 SWA (256), block 3 full (512) */
        b->hd = b->is_swa ? 256 : 512;
        b->qdim = M->NH * b->hd;
        b->share_layer = b->is_swa ? (n_layers_target-2) : (n_layers_target-1);
        if(!b->attn_q||!b->ffn_down) return 0;
    }
    return M->pre_proj&&M->post_proj&&M->tok_embd&&M->out_norm;
}

/* out[o] = sum_i W[o*in+i]*x[i],  W bf16 [out,in]. SVE zip-widen + omp over rows. */
static void bf16_mv(float*out,const uint16_t*W,const float*x,int out_dim,int in_dim){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<out_dim;o++){ const uint16_t*w=W+(size_t)o*in_dim;
#if defined(__ARM_FEATURE_SVE)
        svbool_t pt=svptrue_b32(),pth=svptrue_b16(); svuint16_t zero=svdup_u16(0);
        int vlh=(int)svcnth(),vl=(int)svcntw(); svfloat32_t a=svdup_f32(0); int i=0;
        for(;i+vlh<=in_dim;i+=vlh){ svuint16_t raw=svld1_u16(pth,w+i);
            a=svmla_f32_x(pt,a,svreinterpret_f32_u16(svzip1_u16(zero,raw)),svld1_f32(pt,x+i));
            a=svmla_f32_x(pt,a,svreinterpret_f32_u16(svzip2_u16(zero,raw)),svld1_f32(pt,x+i+vl)); }
        float s=svaddv_f32(pt,a); for(;i<in_dim;i++) s+=bf2f(w[i])*x[i]; out[o]=s;
#else
        float s=0; for(int i=0;i<in_dim;i++) s+=bf2f(w[i])*x[i]; out[o]=s;
#endif
    }
}
static void rmsnorm_w(float*o,const float*x,const float*w,int n,float eps){
    float ss=0; for(int i=0;i<n;i++) ss+=x[i]*x[i]; ss=1.0f/sqrtf(ss/n+eps);
    for(int i=0;i<n;i++) o[i]=x[i]*ss*w[i];
}
static void rmsnorm_head(float*v,int nh,int hd,const float*w,float eps){
    for(int h=0;h<nh;h++){ float*p=v+h*hd; float ss=0; for(int i=0;i<hd;i++) ss+=p[i]*p[i];
        ss=1.0f/sqrtf(ss/hd+eps); for(int i=0;i<hd;i++) p[i]=p[i]*ss*w[i]; }
}
static inline float gelu_t(float x){ return 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x))); }

/* MTP forward: 4-block stack on fused (embd,hidden) -> draft token + h_next[3840].
 * scratch buffers sized for the working dim. attn reads target KV (m->key/value_cache). */
/* qpos = RoPE position of the consumed token; kv_len = # valid target KV positions
 * (attend target KV [start..kv_len-1]). */
static int mtp_step(transformer_model*m, mtp_model*M, const float*hidden /*3840*/, int prev_tok,
                    int qpos, int kv_len, float attn_scale, float embd_scale, float*h_next_out /*3840*/){
    int D=M->D, NE=m->n_embd /*3840*/, NH=M->NH;
    static float *xh=NULL,*cur=NULL,*xb=NULL,*q=NULL,*att=NULL,*ao=NULL,*g1=NULL,*g2=NULL,*g3=NULL,*emb=NULL,*lg=NULL;
    if(!xh){ xh=malloc(2*NE*4); cur=malloc(D*4); xb=malloc(D*4); q=malloc(8192*4); att=malloc((size_t)262144*4);
             ao=malloc(8192*4); g1=malloc(8192*4); g2=malloc(8192*4); g3=malloc(8192*4); emb=malloc(NE*4); lg=malloc((size_t)M->V*4); }
    /* fusion: xh = [ scale*main_embd(prev_tok) , hidden ] (3840+3840) */
    static int swap=-1,nohid=-1,hnorm=-1; static float hscale=1;
    if(swap<0){ swap=getenv("MTP_SWAP")?atoi(getenv("MTP_SWAP")):0; nohid=getenv("MTP_NOHID")?atoi(getenv("MTP_NOHID")):0;
                hnorm=getenv("MTP_HNORM")?atoi(getenv("MTP_HNORM")):0; hscale=getenv("MTP_HSCALE")?atof(getenv("MTP_HSCALE")):1.0f; }
    tf_dequant_row(&m->token_embd, prev_tok, emb);
    float *pe = swap ? xh+NE : xh;       /* embd half */
    float *ph = swap ? xh : xh+NE;       /* hidden half */
    for(int i=0;i<NE;i++) pe[i]=emb[i]*embd_scale;
    if(nohid){ for(int i=0;i<NE;i++) ph[i]=0; }
    else if(hnorm){ double ss=0; for(int i=0;i<NE;i++) ss+=hidden[i]*hidden[i]; float r=(float)(1.0/sqrt(ss/NE+1e-6)); for(int i=0;i<NE;i++) ph[i]=hidden[i]*r; }
    else for(int i=0;i<NE;i++) ph[i]=hidden[i]*hscale;
    bf16_mv(cur, M->pre_proj, xh, D, 2*NE);     /* pre_projection -> 1024 */

    for(int l=0;l<MTP_NL;l++){ mtp_block*b=&M->blk[l];
        int hd=b->hd, qdim=b->qdim, L=b->share_layer;
        int l_kvh = m->layers[L].n_kv_heads>0?m->layers[L].n_kv_heads:m->n_kv_heads;
        int kv_dim = l_kvh*hd, gqa=NH/l_kvh;
        int win=m->swa_window_size, is_swaL=m->layers[L].is_swa;
        /* attn_norm */
        rmsnorm_w(xb,cur,b->attn_norm,D,1e-6f);
        /* q proj + per-head qnorm + RoPE */
        bf16_mv(q, b->attn_q, xb, qdim, D);
        rmsnorm_head(q, NH, hd, b->attn_q_norm, 1e-6f);
        { float*inv=b->is_swa?m->rope_inv_freq_swa:m->rope_inv_freq; int half=hd/2;
          for(int h=0;h<NH;h++){ float*vh=q+h*hd;
            for(int j=0;j<half;j++){ float f=(float)qpos*inv[j], c=cosf(f),s=sinf(f);
                float r0=vh[j],r1=vh[j+half]; vh[j]=r0*c-r1*s; vh[j+half]=r0*s+r1*c; } } }
        /* attention over TARGET KV at layer L, valid positions [start..kv_len-1] */
        int start = (is_swaL && qpos>=win)?(qpos-win+1):0;
        if(start>kv_len-1) start=kv_len-1; if(start<0) start=0;
        int seq=kv_len-start;
        float ascale = attn_scale<0 ? 1.0f/sqrtf((float)hd) : attn_scale;
        memset(ao,0,qdim*4);
        for(int h=0;h<NH;h++){ float*qh=q+h*hd; int kvh=h/gqa; float*a=att;
            for(int p=0;p<seq;p++){ int ap=start+p; int slot=is_swaL?(ap%win):ap;
                size_t kb=(size_t)slot*kv_dim+(size_t)kvh*hd; float sc=0;
                for(int d=0;d<hd;d++) sc+=qh[d]*tf_kv_load_key(m,L,kb+d); a[p]=sc*ascale; }
            float mx=a[0]; for(int p=1;p<seq;p++) if(a[p]>mx)mx=a[p];
            float se=0; for(int p=0;p<seq;p++){ a[p]=expf(a[p]-mx); se+=a[p]; }
            float inv=1.0f/se; float*oh=ao+h*hd;
            for(int p=0;p<seq;p++){ int ap=start+p; int slot=is_swaL?(ap%win):ap; float w=a[p]*inv;
                size_t vb=(size_t)slot*kv_dim+(size_t)kvh*hd;
                for(int d=0;d<hd;d++) oh[d]+=w*tf_kv_load_value(m,L,vb+d); } }
        /* out proj -> D, post-attn norm, residual */
        { static int na=-1; if(na<0)na=getenv("MTP_NOATTN")?atoi(getenv("MTP_NOATTN")):0; if(na) memset(ao,0,qdim*4); }
        bf16_mv(xb, b->attn_output, ao, D, qdim);
        rmsnorm_w(xb,xb,b->post_attn_norm,D,1e-6f);
        for(int i=0;i<D;i++) cur[i]+=xb[i];
        /* ffn: gelu(gate)*up -> down, post-ffw norm, residual, out_scale */
        rmsnorm_w(xb,cur,b->ffn_norm,D,1e-6f);
        bf16_mv(g1,b->ffn_gate,xb,8192,D); bf16_mv(g2,b->ffn_up,xb,8192,D);
        for(int i=0;i<8192;i++) g3[i]=gelu_t(g1[i])*g2[i];
        bf16_mv(xb,b->ffn_down,g3,D,8192);
        rmsnorm_w(xb,xb,b->post_ffw_norm,D,1e-6f);
        float os=b->out_scale[0];
        for(int i=0;i<D;i++) cur[i]=(cur[i]+xb[i])*os;
    }
    /* head: output_norm -> tied tok_embd[1024] -> argmax */
    rmsnorm_w(xb,cur,M->out_norm,D,1e-6f);
    bf16_mv(lg, M->tok_embd, xb, M->V, D);
    int tok=argmax(lg,M->V);
    /* h_next = post_projection(.) -> 3840 for the next step. PNORM: feed the
     * output-normed hidden (xb) so the recurrence stays in post-norm space. */
    static int pnorm=-1; if(pnorm<0) pnorm=getenv("MTP_PNORM")?atoi(getenv("MTP_PNORM")):1; /* default: recurrence in post-norm space */
    bf16_mv(h_next_out, M->post_proj, pnorm?xb:cur, NE, D);
    return tok;
}

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s target.gguf mtp.gguf [G] [K]\n",argv[0]); return 1; }
    int G=argc>3?atoi(argv[3]):48, K=argc>4?atoi(argv[4]):4; if(K>MTP_NL) K=MTP_NL;
    float attn_scale=getenv("MTP_SCALE")?atof(getenv("MTP_SCALE")):1.0f;
    float embd_scale=getenv("MTP_ESCALE")?atof(getenv("MTP_ESCALE")):sqrtf(3840.0f);
    int use_mmap=getenv("NO_MMAP")&&atoi(getenv("NO_MMAP"))?0:1;
    gguf_context*g=gguf_open(argv[1],use_mmap); if(!g){fprintf(stderr,"open fail\n");return 1;}
    transformer_model*m=transformer_load(g,8192); if(!m){fprintf(stderr,"load fail\n");return 1;}
    int nthr=getenv("LLM_THREADS")?atoi(getenv("LLM_THREADS")):48;
    transformer_set_threads(m,nthr); transformer_numa_setup(m,g); transformer_build_panels(m);
    if(getenv("TF_PODD")&&atoi(getenv("TF_PODD"))){ extern void transformer_prepack_podd(transformer_model*); transformer_prepack_podd(m); }
    mtp_model M; if(!load_mtp(&M,argv[2],m->n_layers)){ fprintf(stderr,"MTP load fail\n"); return 1; }
    fprintf(stderr,"MTP loaded: %d blocks, share layers %d/%d/%d/%d, scale=%.4f escale=%.3f\n",
        MTP_NL,M.blk[0].share_layer,M.blk[1].share_layer,M.blk[2].share_layer,M.blk[3].share_layer,attn_scale,embd_scale);

    extern float *tf_batch_all_logits; extern float **tf_batch_hidden_out; extern int tf_batch_keep_pool, tf_batch_quiet;
    tf_batch_keep_pool=1; tf_batch_quiet=1;
    int V=m->n_vocab;
    float*all=malloc((size_t)(K+1)*V*sizeof(float));

    int prompt[]={2,651,6037,576,6081,603,476,6892,576};
    int P=sizeof(prompt)/sizeof(prompt[0]);
    int32_t pp[64]; for(int i=0;i<P;i++) pp[i]=prompt[i];

    /* reference: single-token greedy via GEMM path */
    int ref[1024];
    transformer_prefill_gemm(m,pp,P,0); ref[0]=argmax(m->logits,V);
    for(int i=1;i<G;i++){ int32_t t=ref[i-1]; float*lg=transformer_prefill_gemm(m,&t,1,P+i-1); ref[i]=argmax(lg,V); }

    /* spec decode with MTP draft */
    float*hid=NULL; tf_batch_hidden_out=&hid;
    transformer_prefill_gemm(m,pp,P,0);          /* reset KV [0,P), hid -> [P x 3840] */
    int seq[2048]; for(int i=0;i<P;i++) seq[i]=prompt[i];
    int pos=P; seq[pos]=argmax(m->logits,V);
    if(!hid){ fprintf(stderr,"BUG: hid is NULL after prefill\n"); return 1; }
    /* inp_h = target POST-output-norm hidden (the LM-head input feature), per PR #23398 */
    float thid[3840]; tf_rmsnorm(thid, hid+(size_t)(P-1)*m->n_embd, &m->output_norm, m->n_embd, m->rms_norm_eps, m->matvec_tmp);
    int gen=0,rounds=0,accepted=0,fwd=0,mtp_tries=0,mtp_hit=0;
    double t0=now();
    while(gen<G){
        /* draft K tokens from the target hidden at the current pos */
        int draft[8]; float hh[3840]; memcpy(hh,thid,sizeof(hh));
        int prev=seq[pos];
        static int rnorm=-1,qoff=0; if(rnorm<0){ rnorm=getenv("MTP_RNORM")?atoi(getenv("MTP_RNORM")):0; qoff=getenv("MTP_QOFF")?atoi(getenv("MTP_QOFF")):0; }
        for(int j=0;j<K;j++){ float hn[3840];
            int dt=mtp_step(m,&M,hh,prev,pos+j+qoff,pos,attn_scale,embd_scale,hn);
            draft[j]=dt; prev=dt; memcpy(hh,hn,sizeof(hh));
            if(rnorm){ double ss=0; for(int i=0;i<3840;i++) ss+=hh[i]*hh[i]; float r=(float)(1.0/sqrt(ss/3840+1e-6)); for(int i=0;i<3840;i++) hh[i]*=r; } }
        /* verify: batched forward over [seq[pos], draft...] */
        int32_t batch[16]; int N=K+1; batch[0]=seq[pos];
        for(int j=0;j<K;j++) batch[1+j]=draft[j];
        tf_batch_all_logits=all; tf_batch_hidden_out=&hid;
        transformer_prefill_gemm(m,batch,N,pos); tf_batch_all_logits=NULL; fwd++;
        int gg[16]; for(int j=0;j<N;j++) gg[j]=argmax(all+(size_t)j*V,V);
        int a=0; while(a<K && draft[a]==gg[a]) a++;
        for(int j=0;j<K;j++){ mtp_tries++; if(j<a) mtp_hit++; }
        for(int j=0;j<a;j++) seq[pos+1+j]=draft[j];
        seq[pos+1+a]=gg[a];
        /* target hidden at the new pos = post-norm hid[a] (position pos+a produced token gg[a]) */
        tf_rmsnorm(thid, hid+(size_t)a*m->n_embd, &m->output_norm, m->n_embd, m->rms_norm_eps, m->matvec_tmp);
        gen+=1+a; accepted+=a; rounds++; pos+=1+a;
        if(gen>=G) break;
    }
    double dt=now()-t0;
    int match=0; for(int i=0;i<G;i++) if(seq[P+i]==ref[i]) match++;
    fprintf(stderr,"\nMTP SPEC G=%d K=%d: %d/%d match greedy; %d rounds %d fwd, %d accepted -> %.2f tok/fwd\n",
        G,G,match,G,rounds,fwd,accepted,(double)gen/fwd);
    fprintf(stderr,"  MTP alpha (draft==greedy, per-slot) = %.1f%% (%d/%d); spec %.3fs = %.2f tok/s\n",
        100.0*mtp_hit/mtp_tries,mtp_hit,mtp_tries,dt,gen/dt);
    printf("MTP G=%d K=%d match=%d/%d tok_fwd=%.2f alpha=%.3f tok_s=%.2f\n",
        G,G,match,G,(double)gen/fwd,(double)mtp_hit/mtp_tries,gen/dt);
    return match==G?0:2;
}
