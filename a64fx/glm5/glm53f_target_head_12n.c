/* GLM-5.3F target hyper-head mean, final norm, and sharded vocab argmax. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum{HC=4,H=4096,V=154880};
static void*a256(size_t n){void*p=NULL;if(posix_memalign(&p,256,n))p=NULL;if(!p)MPI_Abort(MPI_COMM_WORLD,2);return p;}
static inline float dot(const uint16_t*w,const float*x,int n){svfloat32_t a=svdup_f32(0);int vl=(int)svcntw();for(int i=0;i<n;i+=vl){svbool_t p=svwhilelt_b32(i,n);svuint32_t z=svlsl_n_u32_x(p,svld1uh_u32(p,w+i),16);a=svmla_x(p,a,svreinterpret_f32_u32(z),svld1(p,x+i));}return svaddv_f32(svptrue_b32(),a);}
int main(int argc,char**argv){int rank,nr,r0,rn;glm53f_st_context*st;uint16_t*norm,*head;float*streams,*hidden,*x,*logit;struct{float value;int index;}in,best[2];MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(argc<2||nr!=12){if(!rank)fprintf(stderr,"usage: mpiexec -np 12 %s MODEL_DIR\n",argv[0]);MPI_Finalize();return 2;}r0=(int)((long long)V*rank/nr);rn=(int)((long long)V*(rank+1)/nr)-r0;st=glm53f_st_open(argv[1]);if(!st)MPI_Abort(MPI_COMM_WORLD,2);norm=a256(H*2);if(glm53f_st_read(st,"model.language_model.norm.weight",0,norm,H*2))MPI_Abort(MPI_COMM_WORLD,2);head=a256((size_t)rn*H*2);if(glm53f_st_read(st,"lm_head.weight",(size_t)r0*H*2,head,(size_t)rn*H*2))MPI_Abort(MPI_COMM_WORLD,2);glm53f_st_close(st);streams=a256((size_t)HC*H*4);hidden=a256(H*4);x=a256(H*4);logit=a256((size_t)rn*4);for(int h=0;h<HC;h++)for(int i=0;i<H;i++)streams[(size_t)h*H+i]=(float)((((h+1)*31+i*17+3)%251)-125)/125.0f;float elapsed[2],phase[2][2];for(int p=0;p<2;p++){double t0=MPI_Wtime();double ss=0;
#pragma omp parallel for reduction(+:ss)
        for(int i=0;i<H;i++){float z=0;for(int h=0;h<HC;h++)z+=streams[(size_t)h*H+i];hidden[i]=z/HC;ss+=(double)hidden[i]*hidden[i];}float inv=1/sqrtf((float)(ss/H)+1e-5f);
#pragma omp parallel for schedule(static)
        for(int i=0;i<H;i++)x[i]=hidden[i]*inv*glm53f_bf16_to_f32(norm[i]);double t1=MPI_Wtime();
#pragma omp parallel for schedule(static)
        for(int r=0;r<rn;r++)logit[r]=dot(head+(size_t)r*H,x,H);in.value=-INFINITY;in.index=-1;for(int r=0;r<rn;r++){int id=r0+r;if(logit[r]>in.value||(logit[r]==in.value&&id<in.index)){in.value=logit[r];in.index=id;}}double t2=MPI_Wtime();MPI_Allreduce(&in,&best[p],1,MPI_FLOAT_INT,MPI_MAXLOC,MPI_COMM_WORLD);double t3=MPI_Wtime();phase[p][0]=t1-t0;phase[p][1]=t2-t1;elapsed[p]=t3-t0;}int ok=best[0].index==best[1].index&&best[0].value==best[1].value,all;MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);float me,mp[2];MPI_Allreduce(&elapsed[1],&me,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);MPI_Allreduce(phase[1],mp,2,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);if(!rank)printf("GLM53F_TARGET_HEAD token=%d logit=%.9g max_ms=%.3f collapse_norm_ms=%.3f vocab_ms=%.3f repeat=%s %s\n",best[0].index,best[0].value,me*1e3f,mp[0]*1e3f,mp[1]*1e3f,all?"BIT_EXACT":"FAIL",all?"PASS":"FAIL");MPI_Finalize();return all?0:1;}
