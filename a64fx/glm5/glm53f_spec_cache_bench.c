/* Snapshot/rollback journal for one rank's GLM-5.3F target decode state. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

enum { LINEAR=34,LOCAL_HEADS=6,D=128,CONV=4,SPARSE=11,MAX_STEPS=4 };
typedef struct{float*recurrent,*conv;int sparse_length[SPARSE];} cache;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void*a256(size_t n){void*p=NULL;return posix_memalign(&p,256,n)?NULL:p;}
static size_t rbytes(void){return(size_t)LINEAR*LOCAL_HEADS*D*D*4;}
static size_t cbytes(void){return(size_t)LINEAR*3*LOCAL_HEADS*D*CONV*4;}
static int alloc_cache(cache*c){c->recurrent=a256(rbytes());c->conv=a256(cbytes());return c->recurrent&&c->conv?0:-1;}
static void save(cache*d,const cache*s){
    memcpy(d->recurrent,s->recurrent,rbytes());
    memcpy(d->conv,s->conv,cbytes());
    memcpy(d->sparse_length,s->sparse_length,sizeof(s->sparse_length));}
static void advance(cache*c,int step){size_t rn=rbytes()/4,cn=cbytes()/4;for(size_t i=step;i<rn;i+=4093)c->recurrent[i]=c->recurrent[i]*.999f+(float)(step+1)*1e-4f;for(size_t i=step;i<cn;i+=509)c->conv[i]=(float)(step+1)+c->conv[i]*.5f;for(int l=0;l<SPARSE;l++)c->sparse_length[l]++;}
int main(void){cache live={0},snap[MAX_STEPS+1]={{0}};if(alloc_cache(&live))return 2;for(int i=0;i<=MAX_STEPS;i++)if(alloc_cache(&snap[i]))return 2;size_t rn=rbytes()/4,cn=cbytes()/4;for(size_t i=0;i<rn;i++)live.recurrent[i]=(float)((int)(i%31)-15)*1e-5f;for(size_t i=0;i<cn;i++)live.conv[i]=(float)((int)(i%17)-8)*1e-3f;for(int l=0;l<SPARSE;l++)live.sparse_length[l]=2048+l;double t=now();save(&snap[0],&live);for(int s=0;s<MAX_STEPS;s++){advance(&live,s);save(&snap[s+1],&live);}double save_ms=(now()-t)*1e3;int ok=1;for(int commit=1;commit<=MAX_STEPS;commit++){save(&live,&snap[MAX_STEPS]);save(&live,&snap[commit]);ok&=!memcmp(live.recurrent,snap[commit].recurrent,rbytes())&&!memcmp(live.conv,snap[commit].conv,cbytes())&&!memcmp(live.sparse_length,snap[commit].sparse_length,sizeof(live.sparse_length));}printf("GLM53F_SPEC_CACHE local_heads=%d state_MiB=%.3f snapshots=%d save_ms=%.3f rollback=%s %s\n",LOCAL_HEADS,(rbytes()+cbytes())/1048576.0,MAX_STEPS+1,save_ms,ok?"BIT_EXACT":"FAIL",ok?"PASS":"FAIL");return ok?0:1;}
