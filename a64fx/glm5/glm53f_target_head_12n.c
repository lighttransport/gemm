/* GLM-5.3F target hyper-head mean, final norm, and sharded vocab argmax. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef GLM53F_EXTERNAL_ST_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#endif
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include "glm53f_target_head_12n.h"
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "glm53f_expert_kern.h"

enum{HC=4,H=4096,V=154880};
static void*a256(size_t n){void*p=NULL;if(posix_memalign(&p,256,n))p=NULL;if(!p)MPI_Abort(MPI_COMM_WORLD,2);return p;}
static inline float dot(const uint16_t*w,const float*x,int n){svfloat32_t a=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+i),16);a=svmla_x(p,a,svreinterpret_f32_u32(z),svld1(p,x+i));}return svaddv_f32(svptrue_b32(),a);}
struct glm53f_target_head_context_12n{int rank,r0,rn;uint16_t*norm,*head;float*hidden,*x,*logits;double phase[3];};
glm53f_target_head_context_12n*glm53f_target_head_create_with_norm_12n(const char*model,const char*norm_name){int rank,nr;glm53f_st_context*st;glm53f_target_head_context_12n*c;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(nr!=12)return NULL;c=calloc(1,sizeof(*c));if(!c)return NULL;c->rank=rank;c->r0=(int)((long long)V*rank/nr);c->rn=(int)((long long)V*(rank+1)/nr)-c->r0;st=glm53f_st_open(model);if(!st)goto fail;c->norm=a256(H*2);if(glm53f_st_read(st,norm_name,0,c->norm,H*2))goto fail;c->head=a256((size_t)c->rn*H*2);if(glm53f_st_read(st,"lm_head.weight",(size_t)c->r0*H*2,c->head,(size_t)c->rn*H*2))goto fail;glm53f_st_close(st);c->hidden=a256((size_t)5*H*4);c->x=a256((size_t)5*H*4);c->logits=a256((size_t)5*c->rn*4);return c;fail:if(st)glm53f_st_close(st);glm53f_target_head_free_12n(c);return NULL;}
glm53f_target_head_context_12n*glm53f_target_head_create_12n(const char*model){return glm53f_target_head_create_with_norm_12n(model,"model.language_model.norm.weight");}
void glm53f_target_head_free_12n(glm53f_target_head_context_12n*c){if(!c)return;free(c->logits);free(c->x);free(c->hidden);free(c->head);free(c->norm);free(c);}
int glm53f_target_head_argmax_12n(glm53f_target_head_context_12n*c,const float*streams,int*token,float*value){struct{float value;int index;}in,best;double t0=MPI_Wtime();double ss=0;
#pragma omp parallel for reduction(+:ss)
    for(int i=0;i<H;i++){float z=0;for(int h=0;h<HC;h++)z+=streams[(size_t)h*H+i];c->hidden[i]=z/HC;ss+=(double)c->hidden[i]*c->hidden[i];}float inv=1/sqrtf((float)(ss/H)+1e-5f);
#pragma omp parallel for schedule(static)
    for(int i=0;i<H;i++)c->x[i]=c->hidden[i]*inv*glm53f_bf16_to_f32(c->norm[i]);double t1=MPI_Wtime();
#pragma omp parallel for schedule(static)
    for(int r=0;r<c->rn;r++)c->logits[r]=dot(c->head+(size_t)r*H,c->x,H);in.value=-INFINITY;in.index=-1;for(int r=0;r<c->rn;r++){int id=c->r0+r;if(c->logits[r]>in.value||(c->logits[r]==in.value&&id<in.index)){in.value=c->logits[r];in.index=id;}}double t2=MPI_Wtime();int rc=MPI_Allreduce(&in,&best,1,MPI_FLOAT_INT,MPI_MAXLOC,MPI_COMM_WORLD);double t3=MPI_Wtime();c->phase[0]=t1-t0;c->phase[1]=t2-t1;c->phase[2]=t3-t2;*token=best.index;*value=best.value;return rc==MPI_SUCCESS?0:-1;}
int glm53f_target_head_argmax_batch_12n(glm53f_target_head_context_12n*c,const float*streams,int tokens,int*token,float*value){if(!c||!streams||!token||!value||tokens<1||tokens>5)return-1;struct pair{float value;int index;}in[5],best[5];double t0=MPI_Wtime();float invs[5];for(int t=0;t<tokens;t++){double ss=0;float*h=c->hidden+(size_t)t*H,*z=c->x+(size_t)t*H;const float*s=streams+(size_t)t*HC*H;
#pragma omp parallel for reduction(+:ss)
        for(int i=0;i<H;i++){float v=0;for(int q=0;q<HC;q++)v+=s[(size_t)q*H+i];h[i]=v/HC;ss+=(double)h[i]*h[i];}invs[t]=1/sqrtf((float)(ss/H)+1e-5f);}
    /* Projection is independent across verification positions; flatten it
     * into one team to avoid a second fork/join per token. */
#pragma omp parallel for schedule(static)
    for(int k=0;k<tokens*H;k++){int t=k/H,i=k%H;float*h=c->hidden+(size_t)t*H,*z=c->x+(size_t)t*H;z[i]=h[i]*invs[t]*glm53f_bf16_to_f32(c->norm[i]);}
    double t1=MPI_Wtime();int n=tokens<5?tokens:4;glm53f_mv_bf16_batch(c->logits,c->head,c->x,n,c->rn,H);if(tokens==5){float*z=c->x+(size_t)4*H,*l=c->logits+(size_t)4*c->rn;
#pragma omp parallel for schedule(static)
        for(int r=0;r<c->rn;r++)l[r]=dot(c->head+(size_t)r*H,z,H);}for(int t=0;t<tokens;t++){float*l=c->logits+(size_t)t*c->rn;in[t].value=-INFINITY;in[t].index=-1;for(int r=0;r<c->rn;r++){int id=c->r0+r;if(l[r]>in[t].value||(l[r]==in[t].value&&id<in[t].index)){in[t].value=l[r];in[t].index=id;}}}double t2=MPI_Wtime();int rc=MPI_Allreduce(in,best,tokens,MPI_FLOAT_INT,MPI_MAXLOC,MPI_COMM_WORLD);double t3=MPI_Wtime();for(int t=0;t<tokens;t++){token[t]=best[t].index;value[t]=best[t].value;}c->phase[0]=t1-t0;c->phase[1]=t2-t1;c->phase[2]=t3-t2;return rc==MPI_SUCCESS?0:-1;}
void glm53f_target_head_last_phase_12n(const glm53f_target_head_context_12n*c,double p[3]){memcpy(p,c->phase,sizeof(c->phase));}
#ifndef GLM53F_TARGET_HEAD_NO_MAIN
int main(int argc,char**argv){int rank,nr,token[2],ok,all;float value[2],*streams;double phase[3],max_phase[3];MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(argc<2||nr!=12){if(!rank)fprintf(stderr,"usage: mpiexec -np 12 %s MODEL_DIR\n",argv[0]);MPI_Finalize();return 2;}glm53f_target_head_context_12n*c=glm53f_target_head_create_12n(argv[1]);if(!c)MPI_Abort(MPI_COMM_WORLD,2);streams=a256((size_t)HC*H*4);for(int h=0;h<HC;h++)for(int i=0;i<H;i++)streams[(size_t)h*H+i]=(float)((((h+1)*31+i*17+3)%251)-125)/125.0f;for(int p=0;p<2;p++)if(glm53f_target_head_argmax_12n(c,streams,&token[p],&value[p]))MPI_Abort(MPI_COMM_WORLD,2);ok=token[0]==token[1]&&value[0]==value[1];MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);glm53f_target_head_last_phase_12n(c,phase);MPI_Allreduce(phase,max_phase,3,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);if(!rank)printf("GLM53F_TARGET_HEAD token=%d logit=%.9g max_ms=%.3f collapse_norm_ms=%.3f vocab_ms=%.3f argmax_ms=%.3f repeat=%s %s\n",token[0],value[0],(max_phase[0]+max_phase[1]+max_phase[2])*1e3,max_phase[0]*1e3,max_phase[1]*1e3,max_phase[2]*1e3,all?"BIT_EXACT":"FAIL",all?"PASS":"FAIL");glm53f_target_head_free_12n(c);MPI_Finalize();return all?0:1;}
#endif
