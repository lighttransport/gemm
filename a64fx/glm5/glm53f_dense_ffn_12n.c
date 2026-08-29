/* Real-weight 12-way tensor-parallel GLM-5.3F dense FFN (layers 0--2). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_arch.h"
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
#ifndef __ARM_FEATURE_SVE
#define __ARM_FEATURE_SVE 1
#endif
#include "glm53f_expert_kern.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum{H=4096,I=12288,B=128};
static void*a256(size_t n){void*p=NULL;if(posix_memalign(&p,256,n))p=NULL;if(!p)MPI_Abort(MPI_COMM_WORLD,2);return p;}
static void rd(glm53f_st_context*s,const char*n,size_t o,void*p,size_t z,int r){if(glm53f_st_read(s,n,o,p,z)){fprintf(stderr,"rank=%d read %s failed\n",r,n);MPI_Abort(MPI_COMM_WORLD,2);}}
static inline float dot(const uint8_t*w,const float*s,const float*x,int n){svfloat32_t a=svdup_f32(0);int vl=(int)svcntw();for(int b=0;b<n;b+=B){for(int i=b;i<b+B;i+=vl){svbool_t p=svwhilelt_b32(i,b+B);a=svmla_x(p,a,glm53f_fp8_e4m3_bits(p,w,i),svmul_n_f32_x(p,svld1(p,x+i),s[b/B]));}}return svaddv_f32(svptrue_b32(),a);}
static void mv(float*y,const uint8_t*w,const float*s,const float*x,int rows,int cols){int nb=cols/B;
#pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++)y[r]=dot(w+(size_t)r*cols,s+(size_t)(r/B)*nb,x,cols);}
int main(int argc,char**argv){int rank,nr,layer=argc>2?atoi(argv[2]):0,i0,in;char n[256];glm53f_st_context*st;uint8_t*g,*u,*d;float*gs,*us,*ds,*x,*gv,*uv,*act,*part,*out[2];MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(argc<2||nr!=12||layer<0||layer>2){if(!rank)fprintf(stderr,"usage: mpiexec -np 12 %s MODEL_DIR [layer=0]\n",argv[0]);MPI_Finalize();return 2;}if(glm53f_block_aligned_slice(I,B,rank,nr,&i0,&in))MPI_Abort(MPI_COMM_WORLD,2);st=glm53f_st_open(argv[1]);if(!st)MPI_Abort(MPI_COMM_WORLD,2);
#define N(S) snprintf(n,sizeof n,"model.language_model.layers.%d.mlp.%s",layer,S)
    g=a256((size_t)in*H);N("gate_proj.weight");rd(st,n,(size_t)i0*H,g,(size_t)in*H,rank);u=a256((size_t)in*H);N("up_proj.weight");rd(st,n,(size_t)i0*H,u,(size_t)in*H,rank);int ib=I/B,hb=H/B,lb=in/B;gs=a256((size_t)lb*hb*4);N("gate_proj.weight_scale_inv");rd(st,n,(size_t)(i0/B)*hb*4,gs,(size_t)lb*hb*4,rank);us=a256((size_t)lb*hb*4);N("up_proj.weight_scale_inv");rd(st,n,(size_t)(i0/B)*hb*4,us,(size_t)lb*hb*4,rank);d=a256((size_t)H*I);N("down_proj.weight");rd(st,n,0,d,(size_t)H*I,rank);ds=a256((size_t)(H/B)*ib*4);N("down_proj.weight_scale_inv");rd(st,n,0,ds,(size_t)(H/B)*ib*4,rank);
#undef N
    glm53f_st_close(st);x=a256(H*4);gv=a256(in*4);uv=a256(in*4);act=a256(in*4);part=a256(H*4);out[0]=a256(H*4);out[1]=a256(H*4);for(int j=0;j<H;j++)x[j]=(float)(((j*17+3)%251)-125)/125.0f;float ph[2][3],el[2];for(int p=0;p<2;p++){double t0=MPI_Wtime();mv(gv,g,gs,x,in,H);mv(uv,u,us,x,in,H);
#pragma omp parallel for schedule(static)
        for(int j=0;j<in;j++){float a=gv[j]>10?10:gv[j],b=uv[j]>10?10:uv[j]<-10?-10:uv[j];act[j]=a/(1+expf(-a))*b;}double t1=MPI_Wtime();
#pragma omp parallel for schedule(static)
        for(int r=0;r<H;r++)part[r]=dot(d+(size_t)r*I+i0,ds+(size_t)(r/B)*ib+i0/B,act,in);double t2=MPI_Wtime();MPI_Allreduce(part,out[p],H,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);double t3=MPI_Wtime();ph[p][0]=t1-t0;ph[p][1]=t2-t1;ph[p][2]=t3-t2;el[p]=t3-t0;}int ok=!memcmp(out[0],out[1],H*4),all;for(int j=0;j<H;j++)ok&=isfinite(out[0][j]);MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);float me,mp[3];MPI_Allreduce(&el[1],&me,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);MPI_Allreduce(ph[1],mp,3,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);double ss=0;for(int j=0;j<H;j++)ss+=(double)out[0][j]*out[0][j];if(!rank)printf("GLM53F_DENSE_FFN_12N layer=%d slice=%d+%d max_ms=%.3f gate_up_ms=%.3f down_ms=%.3f ar_ms=%.3f rms=%.9g repeat=%s %s\n",layer,i0,in,me*1e3f,mp[0]*1e3f,mp[1]*1e3f,mp[2]*1e3f,sqrt(ss/H),all?"BIT_EXACT":"FAIL",all?"PASS":"FAIL");MPI_Finalize();return all?0:1;}
