/* GLM-5.3F target hyper-head mean, final norm, and sharded vocab argmax. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef GLM53F_EXTERNAL_ST_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#endif
#include "glm53f_clock.h"
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include "glm53f_target_head_12n.h"
#include "glm53f_pp_f32.h"
#include "glm53f_pp_core.h"
#include "glm53f_team.h"
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_expert_kern.h"

enum{HC=4,H=4096,V=154880};
static void*a256(size_t n){void*p=NULL;if(posix_memalign(&p,256,n))p=NULL;if(!p)MPI_Abort(MPI_COMM_WORLD,2);return p;}
static inline float dot_bf16(const uint16_t*w,const float*x,int n){svfloat32_t a=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+i),16);a=svmla_x(p,a,svreinterpret_f32_u32(z),svld1(p,x+i));}return svaddv_f32(svptrue_b32(),a);}
static double head_acc[3];static long head_calls;static int head_atexit;
static void head_report(void){if(head_calls)fprintf(stderr,"GLM53F_HEAD_DETAIL us per call: mean+norm=%.1f logits+argmax=%.1f allreduce=%.1f (calls=%ld)\n",head_acc[0]*1e6/head_calls,head_acc[1]*1e6/head_calls,head_acc[2]*1e6/head_calls,head_calls);}
static inline float dot_f32(const float*w,const float*x,int n){svfloat32_t a=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);a=svmla_x(p,a,svld1(p,w+i),svld1(p,x+i));}return svaddv_f32(svptrue_b32(),a);}
/* Four weight rows per pass: x is loaded once per 4 rows and each row keeps two independent accumulators, so the
 * FMA latency chain that made dot_f32 ~2300 cycles per 4096-long row disappears and the loop is bandwidth-bound. */
static inline void dot_f32_rows4(float*out,const float*w,const float*x,int n){
    const svbool_t p=svptrue_b32();
    svfloat32_t a00=svdup_f32(0),a01=a00,a10=a00,a11=a00,a20=a00,a21=a00,a30=a00,a31=a00;
    const float*w0=w,*w1=w+(size_t)n,*w2=w+2*(size_t)n,*w3=w+3*(size_t)n;
    for(int i=0;i<n;i+=32){
        const svfloat32_t xa=svld1(p,x+i),xb=svld1(p,x+i+16);
        a00=svmla_x(p,a00,svld1(p,w0+i),xa);a01=svmla_x(p,a01,svld1(p,w0+i+16),xb);
        a10=svmla_x(p,a10,svld1(p,w1+i),xa);a11=svmla_x(p,a11,svld1(p,w1+i+16),xb);
        a20=svmla_x(p,a20,svld1(p,w2+i),xa);a21=svmla_x(p,a21,svld1(p,w2+i+16),xb);
        a30=svmla_x(p,a30,svld1(p,w3+i),xa);a31=svmla_x(p,a31,svld1(p,w3+i+16),xb);
    }
    out[0]=svaddv_f32(p,svadd_f32_x(p,a00,a01));out[1]=svaddv_f32(p,svadd_f32_x(p,a10,a11));
    out[2]=svaddv_f32(p,svadd_f32_x(p,a20,a21));out[3]=svaddv_f32(p,svadd_f32_x(p,a30,a31));
}
struct glm53f_target_head_context_12n{int rank,r0,rn,hidden_tokens;uint16_t*norm,*head;float*q2_head,*hidden,*x,*logits;double phase[3];const glm53f_dist *dist;};
int glm53f_target_head_normalize_12n(const glm53f_target_head_context_12n *c,
        float *hidden, int tokens) {
    if (!c || !hidden || tokens < 1 || tokens > 4096) return -1;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; ++t) {
        float *h = hidden + (size_t)t * H;
        double ss = 0;
        for (int i = 0; i < H; ++i) ss += (double)h[i] * h[i];
        float inv = 1 / sqrtf((float)(ss / H) + 1e-5f);
        for (int i = 0; i < H; ++i) h[i] = h[i] * inv * glm53f_bf16_to_f32(c->norm[i]);
    }
    return 0;
}
int glm53f_target_head_hidden_12n(const glm53f_target_head_context_12n *c,
        float *hidden, int tokens) {
    if (!c || !hidden || tokens < 1 || tokens > c->hidden_tokens) return -1;
    memcpy(hidden, c->x, (size_t)tokens * H * sizeof(float));
    return 0;
}
const float *glm53f_target_head_logits_12n(const glm53f_target_head_context_12n *c,
                                          int *first, int *count) {
    if (!c || !first || !count) return NULL;
    *first = c->r0; *count = c->rn; return c->logits;
}
static glm53f_target_head_context_12n *head_create(const glm53f_dist *dist,const char *model,const char *native_stage,const char *norm_name){int rank,nr;glm53f_st_context*st=NULL;glm53f_target_head_context_12n*c;
    if(dist){if(!dist->initialized||dist->config.layout!=GLM53F_PP3_TP4||dist->map.stage!=2||!native_stage)return NULL;rank=dist->map.tp_rank;nr=dist->map.tp_size;}
    else{MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(nr!=12)return NULL;}c=calloc(1,sizeof(*c));if(!c)return NULL;c->dist=dist;c->rank=rank;c->r0=(int)((long long)V*rank/nr);c->rn=(int)((long long)V*(rank+1)/nr)-c->r0;st=dist?glm53f_pp_core_open(dist,model):glm53f_st_open(model);if(!st)goto fail;c->norm=a256(H*2);if(glm53f_st_read(st,norm_name,0,c->norm,H*2))goto fail;const char*stage=dist?native_stage:getenv("GLM53F_Q2_HEAD_STAGE");if(stage&&*stage){
if(dist){if(glm53f_pp_f32_load(dist,stage,"HEAD","output.weight",c->r0,c->rn,H,&c->q2_head))goto fail;}else{char path[4096];FILE*f;snprintf(path,sizeof(path),"%s/rank%02d.f32",stage,rank);f=fopen(path,"rb");if(!f)goto fail;c->q2_head=a256((size_t)c->rn*H*4);size_t count=(size_t)c->rn*H;if(fread(c->q2_head,sizeof(float),count,f)!=count||fgetc(f)!=EOF){fclose(f);goto fail;}fclose(f);}
/* fread() first-touches every page from this one thread, putting the whole table in a single CMG.  Re-touch it
 * with the same static partition the logits loop uses so each CMG streams its own rows. */
{const int nb=c->rn/4;
/* local first touch for this table even when the process default is interleave */
syscall(SYS_set_mempolicy,0L /* MPOL_DEFAULT */,(void*)0,0UL);float*spread=a256((size_t)c->rn*H*4);
#pragma omp parallel for schedule(static)
for(int b=0;b<nb;b++)memcpy(spread+(size_t)b*4*H,c->q2_head+(size_t)b*4*H,(size_t)4*H*4);
memcpy(spread+(size_t)nb*4*H,c->q2_head+(size_t)nb*4*H,(size_t)(c->rn-nb*4)*H*4);free(c->q2_head);c->q2_head=spread;
if(!getenv("GLM53F_NUMA_INTERLEAVE")||atoi(getenv("GLM53F_NUMA_INTERLEAVE"))){unsigned long mask=0xF0UL;syscall(SYS_set_mempolicy,3L,&mask,8UL);}}
}else{c->head=a256((size_t)c->rn*H*2);if(glm53f_st_read(st,"lm_head.weight",(size_t)c->r0*H*2,c->head,(size_t)c->rn*H*2))goto fail;}glm53f_st_close(st);st=NULL;c->hidden=a256((size_t)5*H*4);c->x=a256((size_t)5*H*4);c->logits=a256((size_t)5*c->rn*4);return c;fail:if(st)glm53f_st_close(st);glm53f_target_head_free_12n(c);return NULL;}
glm53f_target_head_context_12n *glm53f_target_head_create_with_norm_12n(const char *model,const char *norm_name){return head_create(NULL,model,NULL,norm_name);}
glm53f_target_head_context_12n *glm53f_target_head_create_dist(const glm53f_dist *dist,const char *model,const char *native_stage){return dist?head_create(dist,model,native_stage,"model.language_model.norm.weight"):NULL;}
glm53f_target_head_context_12n*glm53f_target_head_create_12n(const char*model){return glm53f_target_head_create_with_norm_12n(model,"model.language_model.norm.weight");}
void glm53f_target_head_free_12n(glm53f_target_head_context_12n*c){if(!c)return;free(c->logits);free(c->x);free(c->hidden);free(c->q2_head);free(c->head);free(c->norm);free(c);}
struct head_call {
    glm53f_target_head_context_12n *c;
    const float *streams;
    _Alignas(256) double partial[128][32];
    float inv;
    double projection_begin;
};
static void head_worker(void *context) {
    struct head_call *a = context;
    glm53f_target_head_context_12n *c = a->c;
    const int tid = omp_get_thread_num(), threads = omp_get_num_threads();
    if (threads > 128) abort();
    double ss = 0;
#pragma omp for schedule(static) nowait
    for (int i = 0; i < H; ++i) {
        float z = 0;
        for (int h = 0; h < HC; ++h) z += a->streams[(size_t)h * H + i];
        c->hidden[i] = z / HC;
        ss += (double)c->hidden[i] * c->hidden[i];
    }
    a->partial[tid][0] = ss;
#pragma omp barrier
#pragma omp single
    {
        double total = 0;
        for (int t = 0; t < threads; ++t) total += a->partial[t][0];
        a->inv = 1 / sqrtf((float)(total / H) + 1e-5f);
    }
#pragma omp for schedule(static)
    for (int i = 0; i < H; ++i) c->x[i] = c->hidden[i] * a->inv * glm53f_bf16_to_f32(c->norm[i]);
#pragma omp master
    a->projection_begin = glm53f_clock();
    if (c->q2_head && !(H % 32)) {
#pragma omp for schedule(static)
        for (int b = 0; b < c->rn / 4; ++b)
            dot_f32_rows4(c->logits + b * 4, c->q2_head + (size_t)b * 4 * H, c->x, H);
#pragma omp single
        for (int r = c->rn / 4 * 4; r < c->rn; ++r)
            c->logits[r] = dot_f32(c->q2_head + (size_t)r * H, c->x, H);
    } else {
#pragma omp for schedule(static)
        for (int r = 0; r < c->rn; ++r) c->logits[r] = c->q2_head ?
            dot_f32(c->q2_head + (size_t)r * H, c->x, H) : dot_bf16(c->head + (size_t)r * H, c->x, H);
    }
}
int glm53f_target_head_argmax_12n(glm53f_target_head_context_12n*c,const float*streams,int*token,float*value){struct{float value;int index;}in,best;double t0=glm53f_clock();double ss=0;
    double t1;
    if (glm53f_team_available()) {
        struct head_call call;
        call.c = c; call.streams = streams;
        glm53f_team_dispatch(head_worker, &call);
        t1 = call.projection_begin;
    } else {
#pragma omp parallel for reduction(+:ss)
    for(int i=0;i<H;i++){float z=0;for(int h=0;h<HC;h++)z+=streams[(size_t)h*H+i];c->hidden[i]=z/HC;ss+=(double)c->hidden[i]*c->hidden[i];}float inv=1/sqrtf((float)(ss/H)+1e-5f);
#pragma omp parallel for schedule(static)
    for(int i=0;i<H;i++)c->x[i]=c->hidden[i]*inv*glm53f_bf16_to_f32(c->norm[i]);t1=glm53f_clock();
    if(c->q2_head&&(H%32)==0){
        const int nb=c->rn/4;
#pragma omp parallel for schedule(static)
        for(int b=0;b<nb;b++)dot_f32_rows4(c->logits+(size_t)b*4,c->q2_head+(size_t)b*4*H,c->x,H);
        for(int r=nb*4;r<c->rn;r++)c->logits[r]=dot_f32(c->q2_head+(size_t)r*H,c->x,H);
    }else
#pragma omp parallel for schedule(static)
    for(int r=0;r<c->rn;r++)c->logits[r]=c->q2_head?dot_f32(c->q2_head+(size_t)r*H,c->x,H):dot_bf16(c->head+(size_t)r*H,c->x,H);
    }
    in.value=-INFINITY;in.index=-1;for(int r=0;r<c->rn;r++){int id=c->r0+r;if(c->logits[r]>in.value||(c->logits[r]==in.value&&id<in.index)){in.value=c->logits[r];in.index=id;}}double t2=glm53f_clock();int rc=MPI_Allreduce(&in,&best,1,MPI_FLOAT_INT,MPI_MAXLOC,c->dist?c->dist->tp:MPI_COMM_WORLD);double t3=glm53f_clock();c->phase[0]=t1-t0;c->phase[1]=t2-t1;c->phase[2]=t3-t2;head_acc[0]+=c->phase[0];head_acc[1]+=c->phase[1];head_acc[2]+=c->phase[2];head_calls++;if(!head_atexit&&getenv("GLM53F_HEAD_DETAIL")){head_atexit=1;atexit(head_report);}*token=best.index;*value=best.value;c->hidden_tokens=rc==MPI_SUCCESS?1:0;return rc==MPI_SUCCESS?0:-1;}
int glm53f_target_head_argmax_batch_12n(glm53f_target_head_context_12n*c,const float*streams,int tokens,int*token,float*value){if(!c||!streams||!token||!value||tokens<1||tokens>5)return-1;struct pair{float value;int index;}in[5],best[5];double t0=glm53f_clock();float invs[5];for(int t=0;t<tokens;t++){double ss=0;float*h=c->hidden+(size_t)t*H;const float*s=streams+(size_t)t*HC*H;
#pragma omp parallel for reduction(+:ss)
        for(int i=0;i<H;i++){float v=0;for(int q=0;q<HC;q++)v+=s[(size_t)q*H+i];h[i]=v/HC;ss+=(double)h[i]*h[i];}invs[t]=1/sqrtf((float)(ss/H)+1e-5f);}
    /* Projection is independent across verification positions; flatten it
     * into one team to avoid a second fork/join per token. */
#pragma omp parallel for collapse(2) schedule(static)
    for(int t=0;t<tokens;t++)
        for(int i=0;i<H;i++){float*h=c->hidden+(size_t)t*H,*z=c->x+(size_t)t*H;z[i]=h[i]*invs[t]*glm53f_bf16_to_f32(c->norm[i]);}
    double t1=glm53f_clock();if(c->q2_head){
#pragma omp parallel for collapse(2) schedule(static)
        for(int t=0;t<tokens;t++)for(int r=0;r<c->rn;r++)c->logits[(size_t)t*c->rn+r]=dot_f32(c->q2_head+(size_t)r*H,c->x+(size_t)t*H,H);
    }else{int n=tokens<5?tokens:4;glm53f_mv_bf16_batch(c->logits,c->head,c->x,n,c->rn,H);if(tokens==5){float*z=c->x+(size_t)4*H,*l=c->logits+(size_t)4*c->rn;
#pragma omp parallel for schedule(static)
        for(int r=0;r<c->rn;r++)l[r]=dot_bf16(c->head+(size_t)r*H,z,H);}}for(int t=0;t<tokens;t++){float*l=c->logits+(size_t)t*c->rn;in[t].value=-INFINITY;in[t].index=-1;for(int r=0;r<c->rn;r++){int id=c->r0+r;if(l[r]>in[t].value||(l[r]==in[t].value&&id<in[t].index)){in[t].value=l[r];in[t].index=id;}}}double t2=glm53f_clock();int rc=MPI_Allreduce(in,best,tokens,MPI_FLOAT_INT,MPI_MAXLOC,c->dist?c->dist->tp:MPI_COMM_WORLD);double t3=glm53f_clock();for(int t=0;t<tokens;t++){token[t]=best[t].index;value[t]=best[t].value;}c->phase[0]=t1-t0;c->phase[1]=t2-t1;c->phase[2]=t3-t2;c->hidden_tokens=rc==MPI_SUCCESS?tokens:0;return rc==MPI_SUCCESS?0:-1;}
void glm53f_target_head_last_phase_12n(const glm53f_target_head_context_12n*c,double p[3]){memcpy(p,c->phase,sizeof(c->phase));}
#ifndef GLM53F_TARGET_HEAD_NO_MAIN
int main(int argc,char**argv){int rank,nr,token[2],ok,all;float value[2],*streams;double phase[3],max_phase[3];MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(argc<2||nr!=12){if(!rank)fprintf(stderr,"usage: mpiexec -np 12 %s MODEL_DIR\n",argv[0]);MPI_Finalize();return 2;}glm53f_target_head_context_12n*c=glm53f_target_head_create_12n(argv[1]);if(!c)MPI_Abort(MPI_COMM_WORLD,2);streams=a256((size_t)HC*H*4);for(int h=0;h<HC;h++)for(int i=0;i<H;i++)streams[(size_t)h*H+i]=(float)((((h+1)*31+i*17+3)%251)-125)/125.0f;for(int p=0;p<2;p++)if(glm53f_target_head_argmax_12n(c,streams,&token[p],&value[p]))MPI_Abort(MPI_COMM_WORLD,2);ok=token[0]==token[1]&&value[0]==value[1];MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);glm53f_target_head_last_phase_12n(c,phase);MPI_Allreduce(phase,max_phase,3,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);if(!rank)printf("GLM53F_TARGET_HEAD token=%d logit=%.9g max_ms=%.3f collapse_norm_ms=%.3f vocab_ms=%.3f argmax_ms=%.3f repeat=%s %s\n",token[0],value[0],(max_phase[0]+max_phase[1]+max_phase[2])*1e3,max_phase[0]*1e3,max_phase[1]*1e3,max_phase[2]*1e3,all?"BIT_EXACT":"FAIL",all?"PASS":"FAIL");glm53f_target_head_free_12n(c);MPI_Finalize();return all?0:1;}
#endif
