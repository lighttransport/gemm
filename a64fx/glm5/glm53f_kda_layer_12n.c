/* 12-way head/tensor-parallel real-weight GLM-5.3F KDA decode layer. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include "../../common/glm53f_arch.h"
#include "glm53f_kda_12n.h"
#include "glm53f_q8_projection.h"
#include "glm53f_collective_12n.h"
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
#include "glm53f_expert_kern.h"
#include "glm53f_kda_queue.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { H=4096, NH=64, D=128, QKV=8192, KERNEL=4 };
typedef struct { uint16_t *q,*k,*v,*qc,*kc,*vc,*fa,*fb,*b,*ga,*gb,*on,*op; float *al,*dt; } weights;

static void *a256(size_t n){void*p=NULL;if(posix_memalign(&p,256,n))p=NULL;if(!p)MPI_Abort(MPI_COMM_WORLD,2);return p;}
static inline void dot8(float*y,const uint16_t*w,const float*x,int n){
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svfloat32_t xv=svld1(p,x+i);
#define R(N,A) do{svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+(size_t)(N)*n+i),16);A=svmla_x(p,A,svreinterpret_f32_u32(z),xv);}while(0)
        R(0,a0);R(1,a1);R(2,a2);R(3,a3);R(4,a4);R(5,a5);R(6,a6);R(7,a7);
#undef R
    }svbool_t p=svptrue_b32();y[0]=svaddv_f32(p,a0);y[1]=svaddv_f32(p,a1);y[2]=svaddv_f32(p,a2);y[3]=svaddv_f32(p,a3);y[4]=svaddv_f32(p,a4);y[5]=svaddv_f32(p,a5);y[6]=svaddv_f32(p,a6);y[7]=svaddv_f32(p,a7);
}
static inline float dot1(const uint16_t*w,const float*x,int n){svfloat32_t a=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+i),16);a=svmla_x(p,a,svreinterpret_f32_u32(z),svld1(p,x+i));}return svaddv_f32(svptrue_b32(),a);}
#ifndef GLM53F_KDA_NO_MAIN
static void mv(float*y,const uint16_t*w,const float*x,int rows,int cols){int nb=rows/8;
#pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++)dot8(y+b*8,w+(size_t)b*8*cols,x,cols);
    if(rows&7){
#pragma omp parallel for schedule(static)
        for(int r=nb*8;r<rows;r++)y[r]=dot1(w+(size_t)r*cols,x,cols);
    }
}
#endif
/* Orphaned work-sharing variants let one team execute the complete scalar
 * decode graph.  Every loop has its implicit barrier because the following
 * projection consumes the preceding loop's output. Skip remainder worksharing
 * uniformly when rows are a multiple of eight: an empty omp for still pays
 * an implicit team barrier. */
static void mv_team(float*y,const uint16_t*w,const float*x,int rows,int cols){int nb=rows/8;
#pragma omp for schedule(static)
    for(int b=0;b<nb;b++)dot8(y+b*8,w+(size_t)b*8*cols,x,cols);
    if(rows&7){
#pragma omp for schedule(static)
    for(int r=nb*8;r<rows;r++)y[r]=dot1(w+(size_t)r*cols,x,cols);
    }
}
static void mv3_team(float*y0,float*y1,float*y2,const uint16_t*w0,
        const uint16_t*w1,const uint16_t*w2,const float*x,int rows,int cols){int nb=rows/8;
#pragma omp for schedule(static)
    for(int b=0;b<nb;b++){size_t off=(size_t)b*8*cols;dot8(y0+b*8,w0+off,x,cols);dot8(y1+b*8,w1+off,x,cols);dot8(y2+b*8,w2+off,x,cols);}
    if(rows&7){
#pragma omp for schedule(static)
    for(int r=nb*8;r<rows;r++){size_t off=(size_t)r*cols;y0[r]=dot1(w0+off,x,cols);y1[r]=dot1(w1+off,x,cols);y2[r]=dot1(w2+off,x,cols);}
    }
}
static void conv3_team(float*q,float*k,float*v,float*state,const uint16_t*qw,
        const uint16_t*kw,const uint16_t*vw,int channels){
#pragma omp for schedule(static)
    for(int j=0;j<3*channels;j++){int which=j/channels,c=j-which*channels;float*out=which==0?q:(which==1?k:v);const uint16_t*w=which==0?qw:(which==1?kw:vw);float*s=state+(size_t)j*KERNEL,y=0.0f;memmove(s,s+1,(KERNEL-1)*sizeof(*s));s[KERNEL-1]=out[c];for(int z=0;z<KERNEL;z++)y+=s[z]*glm53f_bf16_to_f32(w[(size_t)c*KERNEL+z]);out[c]=y/(1.0f+expf(-y));}
}
static void read_part(glm53f_st_context*st,const char*n,size_t off,void*p,size_t z,int rank){if(glm53f_st_read(st,n,off,p,z)){fprintf(stderr,"rank=%d read %s failed\n",rank,n);MPI_Abort(MPI_COMM_WORLD,2);}}
static void read_cols(glm53f_st_context*st,const char*n,uint16_t*p,int rows,int cols,int c0,int cn,int rank){(void)rows;if(glm53f_st_read_columns(st,n,(size_t)cols*sizeof(uint16_t),(size_t)c0*sizeof(uint16_t),(size_t)cn*sizeof(uint16_t),p)){fprintf(stderr,"rank=%d read columns %s failed\n",rank,n);MPI_Abort(MPI_COMM_WORLD,2);}}
static void name(char*out,int l,const char*s){snprintf(out,256,"model.language_model.layers.%d.self_attn.%s",l,s);}

struct glm53f_kda_context_12n {
    int rank, h0, hn, qd, detail_profile;
    weights w;
    int int8_enabled;
    glm53f_q8_projection quant[4];
    int8_t quant_input[4*H];
    float quant_scale[4*H/128];
    float *qkv,*small,*gate,*decay,*beta,*core,*normed,*work;
    float *conv,*state,*partial,*batch_partial;
    float *bq,*bk,*bv,*bsmall_f,*bsmall_g,*bgate_f,*bgate_g,*bbeta,*bnormed;
    double phase[3], detail[5];
};

glm53f_kda_context_12n *glm53f_kda_create_12n(const char *model,int layer){int rank,nr,h0,hn,qd;char n[256];glm53f_st_context*st;glm53f_kda_context_12n*c;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(nr!=12)return NULL;glm53f_balanced_slice(NH,rank,nr,&h0,&hn);qd=hn*D;st=glm53f_st_open(model);if(!st)return NULL;c=calloc(1,sizeof(*c));if(!c)MPI_Abort(MPI_COMM_WORLD,2);c->rank=rank;c->h0=h0;c->hn=hn;c->qd=qd;c->detail_profile=getenv("GLM53F_KDA_DETAIL")!=NULL;
#define PART(F,S,T,OFF,N) do{name(n,layer,S);c->w.F=(T*)a256((size_t)(N)*sizeof(T));read_part(st,n,(size_t)(OFF)*sizeof(T),c->w.F,(size_t)(N)*sizeof(T),rank);}while(0)
    PART(al,"A_log",float,h0,hn);PART(dt,"dt_bias",float,h0*D,qd);PART(q,"q_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(k,"k_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(v,"v_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(qc,"q_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(kc,"k_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(vc,"v_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(fb,"f_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);PART(b,"b_proj.weight",uint16_t,(size_t)h0*H,(size_t)hn*H);PART(gb,"g_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);PART(fa,"f_a_proj.weight",uint16_t,0,(size_t)D*H);PART(ga,"g_a_proj.weight",uint16_t,0,(size_t)D*H);PART(on,"o_norm.weight",uint16_t,0,D);name(n,layer,"o_proj.weight");c->w.op=a256((size_t)H*qd*sizeof(uint16_t));read_cols(st,n,c->w.op,H,QKV,h0*D,qd,rank);
#undef PART
    glm53f_st_close(st);c->qkv=a256((size_t)3*qd*4);c->small=a256(D*4);c->gate=a256(qd*4);c->decay=a256(qd*4);c->beta=a256(hn*4);c->core=a256(qd*4);c->normed=a256(qd*4);c->work=a256(qd*4);c->conv=a256((size_t)3*qd*KERNEL*4);c->state=a256((size_t)hn*D*D*4);c->partial=a256(H*4);c->batch_partial=a256((size_t)4*H*4);c->bq=a256((size_t)4*qd*4);c->bk=a256((size_t)4*qd*4);c->bv=a256((size_t)4*qd*4);c->bsmall_f=a256((size_t)4*D*4);c->bsmall_g=a256((size_t)4*D*4);c->bgate_f=a256((size_t)4*qd*4);c->bgate_g=a256((size_t)4*qd*4);c->bbeta=a256((size_t)4*hn*4);c->bnormed=a256((size_t)4*qd*4);glm53f_kda_reset_12n(c);return c;}

void glm53f_kda_reset_12n(glm53f_kda_context_12n*c){if(!c)return;memset(c->conv,0,(size_t)3*c->qd*KERNEL*4);memset(c->state,0,(size_t)c->hn*D*D*4);}
int glm53f_kda_set_int8_12n(glm53f_kda_context_12n*c,int enabled){
    if(!c||(enabled!=0&&enabled!=1))return -1;
    if(enabled){
        const uint16_t *w[4]={c->w.q,c->w.k,c->w.v,c->w.op};
        for(int i=0;i<4;i++)if(glm53f_q8_projection_init(c->quant+i,w[i],i==3?H:c->qd,i==3?c->qd:H))return -1;
    }
    c->int8_enabled=enabled;return 0;
}
static void kda_qkv_int8_team(glm53f_kda_context_12n*c,float*q,float*k,float*v,const float*x,int tokens){
    glm53f_q8_inputs_team(c->quant_input,c->quant_scale,x,tokens,H);
    float *out[3]={q,k,v};
#pragma omp for schedule(static)
    for(int r=0;r<3*c->qd;r++)for(int t=0;t<tokens;t++)
        out[r/c->qd][(size_t)t*c->qd+r%c->qd]=glm53f_q8_projection_dot(c->quant+r/c->qd,r%c->qd,
            c->quant_input+(size_t)t*H,c->quant_scale+(size_t)t*(H/128));
}
static void kda_op_int8_team(glm53f_kda_context_12n*c,float*out,const float*x,int tokens){
    glm53f_q8_inputs_team(c->quant_input,c->quant_scale,x,tokens,c->qd);
    glm53f_q8_projection_team(out,c->quant+3,c->quant_input,c->quant_scale,tokens);
}
static int kda_local(glm53f_kda_context_12n*c,float*out,const float*x){
    weights*w=&c->w;int qd=c->qd,hn=c->hn;double td=c->detail_profile?MPI_Wtime():0;
    int fused_op=getenv("GLM53F_KDA_FUSED_OP")&&atoi(getenv("GLM53F_KDA_FUSED_OP"));
    double t0=MPI_Wtime();float*q=c->qkv,*k=q+qd,*v=k+qd;
#pragma omp parallel shared(td)
    {
        /* detail_profile is uniform across this team. Keep timing singles
         * inside the condition so disabled instrumentation adds no barriers. */
        if(c->int8_enabled)kda_qkv_int8_team(c,q,k,v,x,1);
        else mv3_team(q,k,v,w->q,w->k,w->v,x,qd,H);
        if(c->detail_profile){
#pragma omp single
            {double t=MPI_Wtime();c->detail[0]=t-td;td=t;}
        }
        conv3_team(q,k,v,c->conv,w->qc,w->kc,w->vc,qd);
        if(c->detail_profile){
#pragma omp single
            {double t=MPI_Wtime();c->detail[1]=t-td;td=t;}
        }
        mv_team(c->small,w->fa,x,D,H);
        mv_team(c->gate,w->fb,c->small,qd,D);
        mv_team(c->beta,w->b,x,hn,H);
#pragma omp for schedule(static)
        for(int h=0;h<hn;h++){glm53f_l2norm(q+(size_t)h*D,D,1e-6f);glm53f_l2norm(k+(size_t)h*D,D,1e-6f);glm53f_kda_safe_log_decay(c->decay+(size_t)h*D,c->gate+(size_t)h*D,w->dt+(size_t)h*D,w->al[h],-5.0f,D);c->beta[h]=glm53f_sigmoid(c->beta[h]);}
        if(c->detail_profile){
#pragma omp single
            {double t=MPI_Wtime();c->detail[2]=t-td;td=t;}
        }
#pragma omp for schedule(static)
        for(int h=0;h<hn;h++)glm53f_kda_step_vec_streamed(c->state+(size_t)h*D*D,q+(size_t)h*D,k+(size_t)h*D,v+(size_t)h*D,c->decay+(size_t)h*D,c->beta[h],D,D,c->core+(size_t)h*D,c->work+(size_t)h*D);
        if(c->detail_profile){
#pragma omp single
            {double t=MPI_Wtime();c->detail[3]=t-td;td=t;}
        }
        mv_team(c->small,w->ga,x,D,H);
        mv_team(c->gate,w->gb,c->small,qd,D);
#pragma omp for schedule(static)
        for(int h=0;h<hn;h++)glm53f_rmsnorm_gated_bf16(c->normed+(size_t)h*D,c->core+(size_t)h*D,c->gate+(size_t)h*D,w->on,1,D,1e-5f);
        if(c->detail_profile){
#pragma omp single
            {c->detail[4]=MPI_Wtime()-td;}
        }
        if(c->int8_enabled)kda_op_int8_team(c,out,c->normed,1);
        else if(fused_op)mv_team(out,w->op,c->normed,H,qd);
    }
    double t1=MPI_Wtime();
    if(!fused_op&&!c->int8_enabled){
#pragma omp parallel for schedule(static)
    for(int r=0;r<H;r++)out[r]=dot1(w->op+(size_t)r*qd,c->normed,qd);
    }
    double t2=MPI_Wtime();c->phase[0]=t1-t0;c->phase[1]=t2-t1;c->phase[2]=0;return 0;
}
int glm53f_kda_sublayer_12n(void*context,float*out,const float*x){glm53f_kda_context_12n*c=context;if(!c||kda_local(c,c->partial,x))return-1;double t=MPI_Wtime();int rc=glm53f_sum_allreduce_12n(c->partial,out,H);c->phase[2]=MPI_Wtime()-t;return rc;}
static int kda_batch_legacy(glm53f_kda_context_12n*c,float*out,const float*x,int tokens,void*states,size_t stride){size_t bytes=glm53f_kda_state_bytes_12n(c);if(!c||!out||!x||tokens<1||tokens>5||(states&&stride<bytes))return-1;if(tokens==5){if(glm53f_kda_sublayer_batch_capture_12n(c,out,x,4,states,stride))return-1;if(glm53f_kda_sublayer_12n(c,out+(size_t)4*H,x+(size_t)4*H))return-1;return !states||!glm53f_kda_save_state_12n(c,(unsigned char*)states+(size_t)4*stride,stride)?0:-1;}if(tokens==1){if(kda_local(c,c->batch_partial,x))return-1;if(states&&glm53f_kda_save_state_12n(c,states,stride))return-1;}else{weights*w=&c->w;int qd=c->qd,hn=c->hn;double t0=MPI_Wtime();glm53f_mv_bf16_batch(c->bq,w->q,x,tokens,qd,H);glm53f_mv_bf16_batch(c->bk,w->k,x,tokens,qd,H);glm53f_mv_bf16_batch(c->bv,w->v,x,tokens,qd,H);glm53f_mv_bf16_batch(c->bsmall_f,w->fa,x,tokens,D,H);glm53f_mv_bf16_batch(c->bgate_f,w->fb,c->bsmall_f,tokens,qd,D);glm53f_mv_bf16_batch(c->bbeta,w->b,x,tokens,hn,H);glm53f_mv_bf16_batch(c->bsmall_g,w->ga,x,tokens,D,H);glm53f_mv_bf16_batch(c->bgate_g,w->gb,c->bsmall_g,tokens,qd,D);for(int t=0;t<tokens;t++){float*q=c->bq+(size_t)t*qd,*k=c->bk+(size_t)t*qd,*v=c->bv+(size_t)t*qd,*gate=c->bgate_f+(size_t)t*qd,*beta=c->bbeta+(size_t)t*hn;glm53f_causal_conv1d_silu_bf16(q,c->conv,q,w->qc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(k,c->conv+(size_t)qd*KERNEL,k,w->kc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(v,c->conv+(size_t)2*qd*KERNEL,v,w->vc,qd,KERNEL);for(int h=0;h<hn;h++){glm53f_l2norm(q+(size_t)h*D,D,1e-6f);glm53f_l2norm(k+(size_t)h*D,D,1e-6f);glm53f_kda_safe_log_decay(c->decay+(size_t)h*D,gate+(size_t)h*D,w->dt+(size_t)h*D,w->al[h],-5.0f,D);beta[h]=glm53f_sigmoid(beta[h]);}
#pragma omp parallel for schedule(static)
            for(int h=0;h<hn;h++)glm53f_kda_step_vec_streamed(c->state+(size_t)h*D*D,q+(size_t)h*D,k+(size_t)h*D,v+(size_t)h*D,c->decay+(size_t)h*D,beta[h],D,D,c->core+(size_t)h*D,c->work+(size_t)h*D);glm53f_rmsnorm_gated_bf16(c->bnormed+(size_t)t*qd,c->core,c->bgate_g+(size_t)t*qd,w->on,hn,D,1e-5f);if(states&&glm53f_kda_save_state_12n(c,(unsigned char*)states+(size_t)t*stride,stride))return-1;}double t1=MPI_Wtime();glm53f_mv_bf16_batch(c->batch_partial,w->op,c->bnormed,tokens,H,qd);c->phase[0]=t1-t0;c->phase[1]=MPI_Wtime()-t1;}double t=MPI_Wtime();int rc=glm53f_sum_allreduce_12n(c->batch_partial,out,tokens*H);c->phase[2]=MPI_Wtime()-t;return rc;}
static void mv_batch_exact_team(float *y, const uint16_t *w, const float *x,
                                int tokens, int rows, int cols) {
    int nb = rows / 8;
#pragma omp for schedule(static)
    for (int z = 0; z < tokens * nb; ++z) {
        int t = z / nb, b = z - t * nb;
        dot8(y + (size_t)t * rows + b * 8,
             w + (size_t)b * 8 * cols,
             x + (size_t)t * cols, cols);
    }
    if (rows & 7) {
#pragma omp for schedule(static)
        for (int z = 0; z < tokens * (rows & 7); ++z) {
            int t = z / (rows & 7), r = z - t * (rows & 7) + nb * 8;
            y[(size_t)t * rows + r] = dot1(
                w + (size_t)r * cols, x + (size_t)t * cols, cols);
        }
    }
}

static void mv_batch_team(float *y, const uint16_t *w, const float *x,
                          int tokens, int rows, int cols) {
    /* The 4x4 prefill kernel is faster but does not preserve the scalar
     * eight-row accumulation order.  Long-context greedy decoding can amplify
     * that tiny rounding difference into a token change.  Keep an opt-in
     * exact path for verifier diagnostics and correctness-sensitive serving. */
    if (getenv("GLM53F_KDA_BATCH_EXACT") &&
        atoi(getenv("GLM53F_KDA_BATCH_EXACT"))) {
        mv_batch_exact_team(y, w, x, tokens, rows, cols);
        return;
    }
    int n4 = rows / 4;
#pragma omp for schedule(static)
    for (int b = 0; b < n4; b++)
        glm53f_matvec_bf16_4x4(y + b * 4, rows,
            w + (size_t)b * 4 * cols, x, tokens, cols);
    if (rows % 4) {
#pragma omp for schedule(static)
        for (int t = 0; t < tokens; t++)
            for (int r = n4 * 4; r < rows; r++)
                y[(size_t)t * rows + r] = glm53f_dot_bf16_sve(
                    w + (size_t)r * cols, x + (size_t)t * cols, cols);
    }
}

static int kda_batch_capture(glm53f_kda_context_12n *c,
        float *out, const float *x, int tokens, void *states, size_t stride,
        void *const *destinations) {
    size_t bytes = glm53f_kda_state_bytes_12n(c);
    if (!c || !out || !x || tokens < 1 || tokens > 5 || ((states||destinations) && stride < bytes))
        return -1;
    if ((!c->int8_enabled&&(!getenv("GLM53F_KDA_BATCH_TEAM") ||
        !atoi(getenv("GLM53F_KDA_BATCH_TEAM")))) || tokens == 1) {
        if(destinations) {
            for(int t=0;t<tokens;t++)
                if(kda_batch_legacy(c,out+(size_t)t*H,x+(size_t)t*H,1,destinations[t],stride))return -1;
            return 0;
        }
        return kda_batch_legacy(c, out, x, tokens, states, stride);
    }
    if (tokens == 5) {
        if (kda_batch_capture(c, out, x, 4, states, stride, destinations) ||
            glm53f_kda_sublayer_12n(c, out + (size_t)4 * H, x + (size_t)4 * H))
            return -1;
        if (states || destinations) {
            void *dst = destinations ? destinations[4] :
                (unsigned char *)states + (size_t)4 * stride;
            if (dst && glm53f_kda_save_state_12n(c,dst,stride)) return -1;
        }
        return 0;
    }
    weights *w = &c->w;
    int qd = c->qd, hn = c->hn;
    double start = MPI_Wtime(), front_end = start;
    int projection_queue = getenv("GLM53F_KDA_PROJECTION_QUEUE") &&
        atoi(getenv("GLM53F_KDA_PROJECTION_QUEUE"));
#pragma omp parallel shared(front_end)
    {
        if (projection_queue) {
            glm53f_kda_projection_job front[] = {
                {c->bq, w->q, x, qd, H},
                {c->bk, w->k, x, qd, H},
                {c->bv, w->v, x, qd, H},
                {c->bsmall_f, w->fa, x, D, H},
                {c->bsmall_g, w->ga, x, D, H},
                {c->bbeta, w->b, x, hn, H}
            };
            if (c->int8_enabled) {
                kda_qkv_int8_team(c,c->bq,c->bk,c->bv,x,tokens);
                glm53f_kda_projection_queue_team(front + 3, 3, tokens);
            } else {
                glm53f_kda_projection_queue_team(front, 6, tokens);
            }
            glm53f_kda_projection_job gates[] = {
                {c->bgate_f, w->fb, c->bsmall_f, qd, D},
                {c->bgate_g, w->gb, c->bsmall_g, qd, D}
            };
            glm53f_kda_projection_queue_team(gates, 2, tokens);
        } else {
            if(c->int8_enabled)kda_qkv_int8_team(c,c->bq,c->bk,c->bv,x,tokens);
            else {
                mv_batch_team(c->bq, w->q, x, tokens, qd, H);
                mv_batch_team(c->bk, w->k, x, tokens, qd, H);
                mv_batch_team(c->bv, w->v, x, tokens, qd, H);
            }
            mv_batch_team(c->bsmall_f, w->fa, x, tokens, D, H);
            mv_batch_team(c->bgate_f, w->fb, c->bsmall_f, tokens, qd, D);
            mv_batch_team(c->bbeta, w->b, x, tokens, hn, H);
            mv_batch_team(c->bsmall_g, w->ga, x, tokens, D, H);
            mv_batch_team(c->bgate_g, w->gb, c->bsmall_g, tokens, qd, D);
        }
        /* Tokens remain causal; parallelize independent channels/heads within
         * each position and retain every snapshot before advancing state. */
        for (int t = 0; t < tokens; t++) {
            float *q = c->bq + (size_t)t * qd, *k = c->bk + (size_t)t * qd;
            float *v = c->bv + (size_t)t * qd, *gate = c->bgate_f + (size_t)t * qd;
            float *beta = c->bbeta + (size_t)t * hn;
            conv3_team(q, k, v, c->conv, w->qc, w->kc, w->vc, qd);
#pragma omp for schedule(static)
            for (int h = 0; h < hn; h++) {
                glm53f_l2norm(q + (size_t)h * D, D, 1e-6f);
                glm53f_l2norm(k + (size_t)h * D, D, 1e-6f);
                glm53f_kda_safe_log_decay(c->decay + (size_t)h * D,
                    gate + (size_t)h * D, w->dt + (size_t)h * D, w->al[h], -5.0f, D);
                beta[h] = glm53f_sigmoid(beta[h]);
                glm53f_kda_step_vec_streamed(c->state + (size_t)h * D * D,
                    q + (size_t)h * D, k + (size_t)h * D, v + (size_t)h * D,
                    c->decay + (size_t)h * D, beta[h], D, D,
                    c->core + (size_t)h * D, c->work + (size_t)h * D);
                glm53f_rmsnorm_gated_bf16(c->bnormed + (size_t)t * qd + h * D,
                    c->core + (size_t)h * D, c->bgate_g + (size_t)t * qd + h * D,
                    w->on, 1, D, 1e-5f);
            }
            if (states || destinations) {
                unsigned char *dst = destinations ? destinations[t] : (unsigned char *)states + (size_t)t * stride;
                if (!dst) continue;
                size_t state_bytes = (size_t)hn * D * D * sizeof(float);
#pragma omp for schedule(static)
                for (size_t offset = 0; offset < bytes; offset += 256) {
                    size_t end = offset < state_bytes ? state_bytes : bytes;
                    size_t n = end - offset < 256 ? end - offset : 256;
                    const unsigned char *src = offset < state_bytes ?
                        (const unsigned char *)c->state + offset :
                        (const unsigned char *)c->conv + offset - state_bytes;
                    memcpy(dst + offset, src, n);
                }
            }
        }
#pragma omp master
        front_end = MPI_Wtime();
        if(c->int8_enabled)kda_op_int8_team(c,c->batch_partial,c->bnormed,tokens);
        else mv_batch_team(c->batch_partial, w->op, c->bnormed, tokens, H, qd);
    }
    double projection_end = MPI_Wtime();
    c->phase[0] = front_end - start;
    c->phase[1] = projection_end - front_end;
    int rc = glm53f_sum_allreduce_12n(c->batch_partial, out, tokens * H);
    c->phase[2] = MPI_Wtime() - projection_end;
    return rc;
}
int glm53f_kda_sublayer_batch_capture_12n(glm53f_kda_context_12n *c,
        float *out,const float *x,int tokens,void *states,size_t stride) {
    return kda_batch_capture(c,out,x,tokens,states,stride,NULL);
}
int glm53f_kda_sublayer_batch_capture_to_12n(glm53f_kda_context_12n *c,
        float *out,const float *x,int tokens,void *const *states,size_t bytes) {
    return kda_batch_capture(c,out,x,tokens,NULL,bytes,states);
}
int glm53f_kda_sublayer_batch_12n(glm53f_kda_context_12n*c,float*out,const float*x,int tokens){return glm53f_kda_sublayer_batch_capture_12n(c,out,x,tokens,NULL,0);}
void glm53f_kda_last_phase_12n(const glm53f_kda_context_12n*c,double p[3]){memcpy(p,c->phase,sizeof(c->phase));}
void glm53f_kda_last_detail_12n(const glm53f_kda_context_12n*c,double p[5]){memcpy(p,c->detail,sizeof(c->detail));}
size_t glm53f_kda_state_bytes_12n(const glm53f_kda_context_12n*c){return c?((size_t)c->hn*D*D+(size_t)3*c->qd*KERNEL)*sizeof(float):0;}
int glm53f_kda_save_state_12n(const glm53f_kda_context_12n*c,void*dst,size_t bytes){size_t sb=c?(size_t)c->hn*D*D*sizeof(float):0,need=glm53f_kda_state_bytes_12n(c);if(!c||!dst||bytes<need)return-1;memcpy(dst,c->state,sb);memcpy((unsigned char*)dst+sb,c->conv,need-sb);return 0;}
int glm53f_kda_restore_state_12n(glm53f_kda_context_12n*c,const void*src,size_t bytes){size_t sb=c?(size_t)c->hn*D*D*sizeof(float):0,need=glm53f_kda_state_bytes_12n(c);if(!c||!src||bytes<need)return-1;memcpy(c->state,src,sb);memcpy(c->conv,(const unsigned char*)src+sb,need-sb);return 0;}
void glm53f_kda_free_12n(glm53f_kda_context_12n*c){if(!c)return;for(int i=0;i<4;i++)glm53f_q8_projection_free(c->quant+i);free(c->bnormed);free(c->bbeta);free(c->bgate_g);free(c->bgate_f);free(c->bsmall_g);free(c->bsmall_f);free(c->bv);free(c->bk);free(c->bq);free(c->batch_partial);free(c->partial);free(c->state);free(c->conv);free(c->work);free(c->normed);free(c->core);free(c->beta);free(c->decay);free(c->gate);free(c->small);free(c->qkv);free(c->w.op);free(c->w.on);free(c->w.ga);free(c->w.fa);free(c->w.gb);free(c->w.b);free(c->w.fb);free(c->w.vc);free(c->w.kc);free(c->w.qc);free(c->w.v);free(c->w.k);free(c->w.q);free(c->w.dt);free(c->w.al);free(c);}

#ifndef GLM53F_KDA_NO_MAIN
int main(int argc,char**argv){
    int rank,nr,layer=argc>2?atoi(argv[2]):44,h0,hn,qd;char n[256];glm53f_st_context*st;weights w={0};
    float *x,*qkv,*small,*gate,*decay,*beta,*core,*normed,*work,*conv,*state,*partial,*out[2];
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);
    if(argc<2||nr!=12){if(!rank)fprintf(stderr,"usage: mpiexec -np 12 %s MODEL_DIR [layer=44]\n",argv[0]);MPI_Finalize();return 2;}
    glm53f_balanced_slice(NH,rank,nr,&h0,&hn);qd=hn*D;st=glm53f_st_open(argv[1]);if(!st)MPI_Abort(MPI_COMM_WORLD,2);
#define PART(F,S,T,OFF,N) do{name(n,layer,S);w.F=(T*)a256((size_t)(N)*sizeof(T));read_part(st,n,(size_t)(OFF)*sizeof(T),w.F,(size_t)(N)*sizeof(T),rank);}while(0)
    PART(al,"A_log",float,h0,hn);PART(dt,"dt_bias",float,h0*D,qd);
    PART(q,"q_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(k,"k_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(v,"v_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);
    PART(qc,"q_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(kc,"k_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(vc,"v_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);
    PART(fb,"f_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);PART(b,"b_proj.weight",uint16_t,(size_t)h0*H,(size_t)hn*H);PART(gb,"g_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);
    PART(fa,"f_a_proj.weight",uint16_t,0,(size_t)D*H);PART(ga,"g_a_proj.weight",uint16_t,0,(size_t)D*H);PART(on,"o_norm.weight",uint16_t,0,D);name(n,layer,"o_proj.weight");w.op=a256((size_t)H*qd*sizeof(uint16_t));read_cols(st,n,w.op,H,QKV,h0*D,qd,rank);
#undef PART
    glm53f_st_close(st);
    x=a256(H*4);qkv=a256((size_t)3*qd*4);small=a256(D*4);gate=a256(qd*4);decay=a256(qd*4);beta=a256(hn*4);core=a256(qd*4);normed=a256(qd*4);work=a256(qd*4);conv=a256((size_t)3*qd*KERNEL*4);state=a256((size_t)hn*D*D*4);partial=a256(H*4);out[0]=a256(H*4);out[1]=a256(H*4);memset(conv,0,(size_t)3*qd*KERNEL*4);memset(state,0,(size_t)hn*D*D*4);
    for(int i=0;i<H;i++)x[i]=(float)(((i*17+3)%251)-125)/125.0f;
    float phase[2][3],elapsed[2];
    for(int pass=0;pass<2;pass++){
        double t0=MPI_Wtime();float*q=qkv,*k=q+qd,*v=k+qd;mv(q,w.q,x,qd,H);mv(k,w.k,x,qd,H);mv(v,w.v,x,qd,H);glm53f_causal_conv1d_silu_bf16(q,conv,q,w.qc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(k,conv+(size_t)qd*KERNEL,k,w.kc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(v,conv+(size_t)2*qd*KERNEL,v,w.vc,qd,KERNEL);mv(small,w.fa,x,D,H);mv(gate,w.fb,small,qd,D);mv(beta,w.b,x,hn,H);
        for(int h=0;h<hn;h++){glm53f_l2norm(q+(size_t)h*D,D,1e-6f);glm53f_l2norm(k+(size_t)h*D,D,1e-6f);glm53f_kda_safe_log_decay(decay+(size_t)h*D,gate+(size_t)h*D,w.dt+(size_t)h*D,w.al[h],-5.0f,D);beta[h]=glm53f_sigmoid(beta[h]);}
#pragma omp parallel for schedule(static)
        for(int h=0;h<hn;h++)glm53f_kda_step_vec_streamed(state+(size_t)h*D*D,q+(size_t)h*D,k+(size_t)h*D,v+(size_t)h*D,decay+(size_t)h*D,beta[h],D,D,core+(size_t)h*D,work+(size_t)h*D);
        mv(small,w.ga,x,D,H);mv(gate,w.gb,small,qd,D);glm53f_rmsnorm_gated_bf16(normed,core,gate,w.on,hn,D,1e-5f);double t1=MPI_Wtime();
#pragma omp parallel for schedule(static)
        for(int r=0;r<H;r++)partial[r]=dot1(w.op+(size_t)r*qd,normed,qd);
        double t2=MPI_Wtime();MPI_Allreduce(partial,out[pass],H,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);double t3=MPI_Wtime();phase[pass][0]=t1-t0;phase[pass][1]=t2-t1;phase[pass][2]=t3-t2;elapsed[pass]=t3-t0;
        for(int i=0;i<H;i++)x[i]=(float)(((i*29+7)%257)-128)/128.0f;
    }
    int stable=1,allstable;
    for(int i=0;i<H;i++)stable&=isfinite(out[0][i])&&isfinite(out[1][i]);
    MPI_Allreduce(&stable,&allstable,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    float maxe,maxp[3],rank_phase[12][3];MPI_Allreduce(&elapsed[1],&maxe,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);MPI_Allreduce(phase[1],maxp,3,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);MPI_Gather(phase[1],3,MPI_FLOAT,rank_phase,3,MPI_FLOAT,0,MPI_COMM_WORLD);
    double ss=0.0,sum=0.0;for(int i=0;i<H;i++){ss+=(double)out[1][i]*out[1][i];sum+=out[1][i];}
    if(!rank)printf("GLM53F_KDA_12N layer=%d heads=64 local_heads=%d..%d max_ms=%.3f local_graph_ms=%.3f oproj_ms=%.3f allreduce_ms=%.3f rms=%.9g sum=%.9g finite=%s %s\n",layer,h0,h0+hn,maxe*1e3f,maxp[0]*1e3f,maxp[1]*1e3f,maxp[2]*1e3f,sqrt(ss/H),sum,allstable?"YES":"NO",allstable?"PASS":"FAIL");
    if(!rank){printf("GLM53F_KDA_12N_RANK_MS");for(int r=0;r<12;r++)printf(" r%d=%.3f/%.3f/%.3f",r,rank_phase[r][0]*1e3f,rank_phase[r][1]*1e3f,rank_phase[r][2]*1e3f);putchar('\n');}
    if(!rank){const char*dump=getenv("GLM53F_KDA_OUTPUT");if(dump&&*dump){FILE*f=fopen(dump,"wb");if(!f||fwrite(out[1],sizeof(float),H,f)!=(size_t)H)MPI_Abort(MPI_COMM_WORLD,2);fclose(f);}}
    MPI_Finalize();return allstable?0:1;
}
#endif
