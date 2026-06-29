/* gemma4_mtp.h - Gemma4-assistant MTP draft head (4-block @1024-dim), extracted from
 * a64fx/bench_q4_0_matvec/mini_mtp.c for reuse in the TP spec-decode runner.
 *
 * The MTP predicts future tokens from the TARGET's last hidden state. Each block has
 * q-only attention that READS the target's KV cache (no own k/v): SWA blocks 0,1,2 share
 * target layer n-2, full block 3 shares n-1. Fusion: concat(scale(embd,sqrt(n_embd)),
 * hidden) -> pre_proj -> 1024; recurse h_next = post_proj(post-normed cur). Working
 * config (single-node alpha ~0.88): PNORM=1, escale=sqrt(n_embd), defaults elsewhere.
 *
 * In TP every rank replicates the MTP weights AND reads its replicated target KV, so the
 * draft is identical on all ranks (no comm). Requires transformer.h already included.
 */
#ifndef GEMMA4_MTP_H
#define GEMMA4_MTP_H
#include <math.h>

#define MTP_NL 4
typedef struct {
    qtensor attn_q, attn_output, ffn_gate, ffn_up, ffn_down;     /* any type (Q8_0 here) */
    const float *attn_norm, *attn_q_norm, *post_attn_norm, *ffn_norm, *post_ffw_norm, *out_scale;
    int hd, qdim, is_swa, share_layer;
} mtp_block;
typedef struct {
    mtp_block blk[MTP_NL];
    qtensor pre_proj, post_proj, tok_embd;
    const float    *out_norm;
    int D, V, NH, FF;
} mtp_model;

static int mtp_argmax(const float*v,int n){ int b=0; float m=v[0]; for(int i=1;i<n;i++) if(v[i]>m){m=v[i];b=i;} return b; }
static const void* mtp_find(gguf_context*g,const char*name){
    for(uint64_t i=0;i<g->n_tensors;i++){ const char*nm=gguf_tensor_name(g,(int)i);
        if(nm&&!strcmp(nm,name)) return gguf_tensor_data(g,(int)i); }
    fprintf(stderr,"MTP: missing tensor %s\n",name); return NULL;
}
/* load the MTP gguf (eager RAM, small). MTP weights may be Q8_0 -> store qtensors and
 * dequant via tf_dequant_row (type-generic). norms are f32 raw pointers. */
static int mtp_load(mtp_model*M,const char*path,int n_layers_target,int n_embd){
    gguf_context*g=gguf_open(path,0); if(!g){fprintf(stderr,"MTP open fail %s\n",path);return 0;}
    M->D=1024; M->V=262144; M->NH=16; M->FF=8192; (void)n_embd;
    M->pre_proj =tf_load_tensor(g,"nextn.pre_projection.weight",1);
    M->post_proj=tf_load_tensor(g,"nextn.post_projection.weight",1);
    M->tok_embd =tf_load_tensor(g,"token_embd.weight",1);
    M->out_norm =mtp_find(g,"output_norm.weight");
    char nm[96];
    for(int l=0;l<MTP_NL;l++){ mtp_block*b=&M->blk[l];
        #define MTQ(field,suf) (sprintf(nm,"blk.%d." suf,l), b->field=tf_load_tensor(g,nm,1))
        #define MTF(field,suf) (sprintf(nm,"blk.%d." suf,l), b->field=mtp_find(g,nm))
        MTQ(attn_q,"attn_q.weight"); MTQ(attn_output,"attn_output.weight");
        MTQ(ffn_gate,"ffn_gate.weight"); MTQ(ffn_up,"ffn_up.weight"); MTQ(ffn_down,"ffn_down.weight");
        MTF(attn_norm,"attn_norm.weight"); MTF(attn_q_norm,"attn_q_norm.weight");
        MTF(post_attn_norm,"post_attention_norm.weight"); MTF(ffn_norm,"ffn_norm.weight");
        MTF(post_ffw_norm,"post_ffw_norm.weight"); MTF(out_scale,"layer_output_scale.weight");
        #undef MTQ
        #undef MTF
        b->is_swa=(l!=3); b->hd=b->is_swa?256:512; b->qdim=M->NH*b->hd;
        b->share_layer=b->is_swa?(n_layers_target-2):(n_layers_target-1);
        if(!b->attn_q.data||!b->ffn_down.data||!b->attn_norm||!b->ffn_gate.data||!b->out_scale){
            fprintf(stderr,"MTP: block %d has a NULL weight\n",l); return 0; }
    }
    return M->pre_proj.data&&M->post_proj.data&&M->tok_embd.data&&M->out_norm;
}

/* generic dequant matvec: out[o]=dot(dequant(W row o), x), OMP over rows, per-thread
 * dequant scratch. out_dim==W->n_rows, in_dim==W->n_cols. Handles Q8_0/BF16/F32. */
static void mtp_mv(float*out,const qtensor*W,const float*x,int out_dim,int in_dim){
    (void)out_dim; (void)in_dim;
    int OD=W->n_rows, ID=W->n_cols;
    #pragma omp parallel
    {
        float *row=(float*)malloc((size_t)ID*sizeof(float));
        #pragma omp for schedule(static)
        for(int o=0;o<OD;o++){
            tf_dequant_row(W,o,row);
            float s=0;
#if defined(__ARM_FEATURE_SVE)
            svbool_t pt=svptrue_b32(); svfloat32_t a=svdup_f32(0); int i=0,vl=(int)svcntw();
            for(;i+vl<=ID;i+=vl) a=svmla_f32_x(pt,a,svld1_f32(pt,row+i),svld1_f32(pt,x+i));
            s=svaddv_f32(pt,a); for(;i<ID;i++) s+=row[i]*x[i];
#else
            for(int i=0;i<ID;i++) s+=row[i]*x[i];
#endif
            out[o]=s;
        }
        free(row);
    }
}
static void mtp_rms(float*o,const float*x,const float*w,int n,float eps){
    float ss=0; for(int i=0;i<n;i++) ss+=x[i]*x[i]; ss=1.0f/sqrtf(ss/n+eps);
    for(int i=0;i<n;i++) o[i]=x[i]*ss*w[i];
}
static void mtp_rms_head(float*v,int nh,int hd,const float*w,float eps){
    for(int h=0;h<nh;h++){ float*p=v+h*hd; float ss=0; for(int i=0;i<hd;i++) ss+=p[i]*p[i];
        ss=1.0f/sqrtf(ss/hd+eps); for(int i=0;i<hd;i++) p[i]=p[i]*ss*w[i]; }
}
static inline float mtp_gelu(float x){ return 0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x))); }

/* one MTP draft step: fuse (embd(prev_tok), target hidden) -> 4 blocks reading target KV
 * -> draft token; writes h_next_out[n_embd] (post_proj output) for the next step. */
static int mtp_step(transformer_model*m, mtp_model*M, const float*hidden, int prev_tok,
                    int qpos, int kv_len, float embd_scale, float*h_next_out){
    int D=M->D, NE=m->n_embd, NH=M->NH, FF=M->FF;
    static float *xh=NULL,*cur=NULL,*xb=NULL,*q=NULL,*att=NULL,*ao=NULL,*g1=NULL,*g2=NULL,*g3=NULL,*emb=NULL,*lg=NULL;
    if(!xh){ xh=malloc(2*NE*4); cur=malloc(D*4); xb=malloc(D*4); q=malloc(8192*4); att=malloc((size_t)262144*4);
             ao=malloc(8192*4); g1=malloc((size_t)FF*4); g2=malloc((size_t)FF*4); g3=malloc((size_t)FF*4); emb=malloc(NE*4); lg=malloc((size_t)M->V*4); }
    tf_dequant_row(&m->token_embd, prev_tok, emb);
    for(int i=0;i<NE;i++) xh[i]=emb[i]*embd_scale;     /* embd half (scaled) */
    for(int i=0;i<NE;i++) xh[NE+i]=hidden[i];          /* target hidden half */
    mtp_mv(cur, &M->pre_proj, xh, D, 2*NE);
    for(int l=0;l<MTP_NL;l++){ mtp_block*b=&M->blk[l];
        int hd=b->hd, qdim=b->qdim, L=b->share_layer;
        int l_kvh=m->layers[L].n_kv_heads>0?m->layers[L].n_kv_heads:m->n_kv_heads;
        int kv_dim=l_kvh*hd, gqa=NH/l_kvh, win=m->swa_window_size, is_swaL=m->layers[L].is_swa;
        mtp_rms(xb,cur,b->attn_norm,D,1e-6f);
        mtp_mv(q,&b->attn_q,xb,qdim,D);
        mtp_rms_head(q,NH,hd,b->attn_q_norm,1e-6f);
        { float*inv=b->is_swa?m->rope_inv_freq_swa:m->rope_inv_freq; int half=hd/2;
          for(int h=0;h<NH;h++){ float*vh=q+h*hd;
            for(int j=0;j<half;j++){ float f=(float)qpos*inv[j],c=cosf(f),s=sinf(f);
              float r0=vh[j],r1=vh[j+half]; vh[j]=r0*c-r1*s; vh[j+half]=r0*s+r1*c; } } }
        int start=(is_swaL && qpos>=win)?(qpos-win+1):0;
        if(start>kv_len-1) start=kv_len-1; if(start<0) start=0;
        int seq=kv_len-start; float ascale=1.0f;   /* QK-norm handles scaling (mini_mtp default) */
        memset(ao,0,qdim*4);
        for(int h=0;h<NH;h++){ float*qh=q+h*hd; int kvh=h/gqa; float*a=att;
            for(int p=0;p<seq;p++){ int ap=start+p,slot=is_swaL?(ap%win):ap;
                size_t kb=(size_t)slot*kv_dim+(size_t)kvh*hd; float sc=0;
                for(int d=0;d<hd;d++) sc+=qh[d]*tf_kv_load_key(m,L,kb+d); a[p]=sc*ascale; }
            float mx=a[0]; for(int p=1;p<seq;p++) if(a[p]>mx)mx=a[p];
            float se=0; for(int p=0;p<seq;p++){ a[p]=expf(a[p]-mx); se+=a[p]; }
            float iv=1.0f/se; float*oh=ao+h*hd;
            for(int p=0;p<seq;p++){ int ap=start+p,slot=is_swaL?(ap%win):ap; float w=a[p]*iv;
                size_t vb=(size_t)slot*kv_dim+(size_t)kvh*hd;
                for(int d=0;d<hd;d++) oh[d]+=w*tf_kv_load_value(m,L,vb+d); } }
        mtp_mv(xb,&b->attn_output,ao,D,qdim);
        mtp_rms(xb,xb,b->post_attn_norm,D,1e-6f);
        for(int i=0;i<D;i++) cur[i]+=xb[i];
        mtp_rms(xb,cur,b->ffn_norm,D,1e-6f);
        mtp_mv(g1,&b->ffn_gate,xb,FF,D); mtp_mv(g2,&b->ffn_up,xb,FF,D);
        for(int i=0;i<FF;i++) g3[i]=mtp_gelu(g1[i])*g2[i];
        mtp_mv(xb,&b->ffn_down,g3,D,FF);
        mtp_rms(xb,xb,b->post_ffw_norm,D,1e-6f);
        float os=b->out_scale[0];
        for(int i=0;i<D;i++) cur[i]=(cur[i]+xb[i])*os;
    }
    mtp_rms(xb,cur,M->out_norm,D,1e-6f);
    mtp_mv(lg,&M->tok_embd,xb,M->V,D);
    int tok=mtp_argmax(lg,M->V);
    mtp_mv(h_next_out,&M->post_proj,xb,NE,D);     /* PNORM: recurse in post-norm space */
    return tok;
}
#endif
