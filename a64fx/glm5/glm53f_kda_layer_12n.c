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
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
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
static void mv(float*y,const uint16_t*w,const float*x,int rows,int cols){int nb=rows/8;
#pragma omp parallel for schedule(static)
    for(int b=0;b<nb;b++)dot8(y+b*8,w+(size_t)b*8*cols,x,cols);
#pragma omp parallel for schedule(static)
    for(int r=nb*8;r<rows;r++)y[r]=dot1(w+(size_t)r*cols,x,cols);
}
static void read_part(glm53f_st_context*st,const char*n,size_t off,void*p,size_t z,int rank){if(glm53f_st_read(st,n,off,p,z)){fprintf(stderr,"rank=%d read %s failed\n",rank,n);MPI_Abort(MPI_COMM_WORLD,2);}}
static void read_cols(glm53f_st_context*st,const char*n,uint16_t*p,int rows,int cols,int c0,int cn,int rank){(void)rows;if(glm53f_st_read_columns(st,n,(size_t)cols*sizeof(uint16_t),(size_t)c0*sizeof(uint16_t),(size_t)cn*sizeof(uint16_t),p)){fprintf(stderr,"rank=%d read columns %s failed\n",rank,n);MPI_Abort(MPI_COMM_WORLD,2);}}
static void name(char*out,int l,const char*s){snprintf(out,256,"model.language_model.layers.%d.self_attn.%s",l,s);}

struct glm53f_kda_context_12n {
    int rank, h0, hn, qd;
    weights w;
    float *qkv,*small,*gate,*decay,*beta,*core,*normed,*work;
    float *conv,*state,*partial;
    double phase[3];
};

glm53f_kda_context_12n *glm53f_kda_create_12n(const char *model,int layer){int rank,nr,h0,hn,qd;char n[256];glm53f_st_context*st;glm53f_kda_context_12n*c;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(nr!=12)return NULL;glm53f_balanced_slice(NH,rank,nr,&h0,&hn);qd=hn*D;st=glm53f_st_open(model);if(!st)return NULL;c=calloc(1,sizeof(*c));if(!c)MPI_Abort(MPI_COMM_WORLD,2);c->rank=rank;c->h0=h0;c->hn=hn;c->qd=qd;
#define PART(F,S,T,OFF,N) do{name(n,layer,S);c->w.F=(T*)a256((size_t)(N)*sizeof(T));read_part(st,n,(size_t)(OFF)*sizeof(T),c->w.F,(size_t)(N)*sizeof(T),rank);}while(0)
    PART(al,"A_log",float,h0,hn);PART(dt,"dt_bias",float,h0*D,qd);PART(q,"q_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(k,"k_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(v,"v_proj.weight",uint16_t,(size_t)h0*D*H,(size_t)qd*H);PART(qc,"q_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(kc,"k_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(vc,"v_conv1d.weight",uint16_t,(size_t)h0*D*KERNEL,(size_t)qd*KERNEL);PART(fb,"f_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);PART(b,"b_proj.weight",uint16_t,(size_t)h0*H,(size_t)hn*H);PART(gb,"g_b_proj.weight",uint16_t,(size_t)h0*D*D,(size_t)qd*D);PART(fa,"f_a_proj.weight",uint16_t,0,(size_t)D*H);PART(ga,"g_a_proj.weight",uint16_t,0,(size_t)D*H);PART(on,"o_norm.weight",uint16_t,0,D);name(n,layer,"o_proj.weight");c->w.op=a256((size_t)H*qd*sizeof(uint16_t));read_cols(st,n,c->w.op,H,QKV,h0*D,qd,rank);
#undef PART
    glm53f_st_close(st);c->qkv=a256((size_t)3*qd*4);c->small=a256(D*4);c->gate=a256(qd*4);c->decay=a256(qd*4);c->beta=a256(hn*4);c->core=a256(qd*4);c->normed=a256(qd*4);c->work=a256(qd*4);c->conv=a256((size_t)3*qd*KERNEL*4);c->state=a256((size_t)hn*D*D*4);c->partial=a256(H*4);glm53f_kda_reset_12n(c);return c;}

void glm53f_kda_reset_12n(glm53f_kda_context_12n*c){if(!c)return;memset(c->conv,0,(size_t)3*c->qd*KERNEL*4);memset(c->state,0,(size_t)c->hn*D*D*4);}
int glm53f_kda_sublayer_12n(void*context,float*out,const float*x){glm53f_kda_context_12n*c=context;weights*w=&c->w;int qd=c->qd,hn=c->hn;double t0=MPI_Wtime();float*q=c->qkv,*k=q+qd,*v=k+qd;mv(q,w->q,x,qd,H);mv(k,w->k,x,qd,H);mv(v,w->v,x,qd,H);glm53f_causal_conv1d_silu_bf16(q,c->conv,q,w->qc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(k,c->conv+(size_t)qd*KERNEL,k,w->kc,qd,KERNEL);glm53f_causal_conv1d_silu_bf16(v,c->conv+(size_t)2*qd*KERNEL,v,w->vc,qd,KERNEL);mv(c->small,w->fa,x,D,H);mv(c->gate,w->fb,c->small,qd,D);mv(c->beta,w->b,x,hn,H);for(int h=0;h<hn;h++){glm53f_l2norm(q+(size_t)h*D,D,1e-6f);glm53f_l2norm(k+(size_t)h*D,D,1e-6f);glm53f_kda_safe_log_decay(c->decay+(size_t)h*D,c->gate+(size_t)h*D,w->dt+(size_t)h*D,w->al[h],-5.0f,D);c->beta[h]=glm53f_sigmoid(c->beta[h]);}
#pragma omp parallel for schedule(static)
    for(int h=0;h<hn;h++)glm53f_kda_step_vec_streamed(c->state+(size_t)h*D*D,q+(size_t)h*D,k+(size_t)h*D,v+(size_t)h*D,c->decay+(size_t)h*D,c->beta[h],D,D,c->core+(size_t)h*D,c->work+(size_t)h*D);mv(c->small,w->ga,x,D,H);mv(c->gate,w->gb,c->small,qd,D);glm53f_rmsnorm_gated_bf16(c->normed,c->core,c->gate,w->on,hn,D,1e-5f);double t1=MPI_Wtime();
#pragma omp parallel for schedule(static)
    for(int r=0;r<H;r++)c->partial[r]=dot1(w->op+(size_t)r*qd,c->normed,qd);double t2=MPI_Wtime();int rc=MPI_Allreduce(c->partial,out,H,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);double t3=MPI_Wtime();c->phase[0]=t1-t0;c->phase[1]=t2-t1;c->phase[2]=t3-t2;return rc==MPI_SUCCESS?0:-1;}
void glm53f_kda_last_phase_12n(const glm53f_kda_context_12n*c,double p[3]){memcpy(p,c->phase,sizeof(c->phase));}
size_t glm53f_kda_state_bytes_12n(const glm53f_kda_context_12n*c){return c?((size_t)c->hn*D*D+(size_t)3*c->qd*KERNEL)*sizeof(float):0;}
int glm53f_kda_save_state_12n(const glm53f_kda_context_12n*c,void*dst,size_t bytes){size_t sb=c?(size_t)c->hn*D*D*sizeof(float):0,need=glm53f_kda_state_bytes_12n(c);if(!c||!dst||bytes<need)return-1;memcpy(dst,c->state,sb);memcpy((unsigned char*)dst+sb,c->conv,need-sb);return 0;}
int glm53f_kda_restore_state_12n(glm53f_kda_context_12n*c,const void*src,size_t bytes){size_t sb=c?(size_t)c->hn*D*D*sizeof(float):0,need=glm53f_kda_state_bytes_12n(c);if(!c||!src||bytes<need)return-1;memcpy(c->state,src,sb);memcpy(c->conv,(const unsigned char*)src+sb,need-sb);return 0;}
void glm53f_kda_free_12n(glm53f_kda_context_12n*c){if(!c)return;free(c->partial);free(c->state);free(c->conv);free(c->work);free(c->normed);free(c->core);free(c->beta);free(c->decay);free(c->gate);free(c->small);free(c->qkv);free(c->w.op);free(c->w.on);free(c->w.ga);free(c->w.fa);free(c->w.gb);free(c->w.b);free(c->w.fb);free(c->w.vc);free(c->w.kc);free(c->w.qc);free(c->w.v);free(c->w.k);free(c->w.q);free(c->w.dt);free(c->w.al);free(c);}

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
