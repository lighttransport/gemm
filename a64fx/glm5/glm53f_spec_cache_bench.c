/* Snapshot/rollback journal for one rank's GLM-5.3F target decode state. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../common/glm53f_decode_state.h"

enum { MAX_STEPS=4 };
typedef glm53f_decode_state_12n cache;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void advance(cache*c,int step){size_t rn=glm53f_recurrent_bytes_12n()/4,cn=glm53f_conv_bytes_12n()/4;for(size_t i=step;i<rn;i+=4093)c->recurrent[i]=c->recurrent[i]*.999f+(float)(step+1)*1e-4f;for(size_t i=step;i<cn;i+=509)c->conv[i]=(float)(step+1)+c->conv[i]*.5f;for(int l=0;l<GLM53F_SPARSE_LAYERS;l++)c->sparse_length[l]++;}
int main(void){cache live={0},snap[MAX_STEPS+1]={{0}};if(glm53f_decode_state_alloc_12n(&live))return 2;for(int i=0;i<=MAX_STEPS;i++)if(glm53f_decode_state_alloc_12n(&snap[i]))return 2;size_t rn=glm53f_recurrent_bytes_12n()/4,cn=glm53f_conv_bytes_12n()/4;for(size_t i=0;i<rn;i++)live.recurrent[i]=(float)((int)(i%31)-15)*1e-5f;for(size_t i=0;i<cn;i++)live.conv[i]=(float)((int)(i%17)-8)*1e-3f;for(int l=0;l<GLM53F_SPARSE_LAYERS;l++)live.sparse_length[l]=2048+l;double t=now();glm53f_decode_state_save_12n(&snap[0],&live);for(int s=0;s<MAX_STEPS;s++){advance(&live,s);glm53f_decode_state_save_12n(&snap[s+1],&live);}double save_ms=(now()-t)*1e3;int ok=1;for(int commit=1;commit<=MAX_STEPS;commit++){glm53f_decode_state_save_12n(&live,&snap[MAX_STEPS]);glm53f_decode_state_save_12n(&live,&snap[commit]);ok&=!memcmp(live.recurrent,snap[commit].recurrent,glm53f_recurrent_bytes_12n())&&!memcmp(live.conv,snap[commit].conv,glm53f_conv_bytes_12n())&&!memcmp(live.sparse_length,snap[commit].sparse_length,sizeof(live.sparse_length));}printf("GLM53F_SPEC_CACHE local_heads=%d state_MiB=%.3f snapshots=%d save_ms=%.3f rollback=%s %s\n",GLM53F_LOCAL_HEADS_12N,(glm53f_recurrent_bytes_12n()+glm53f_conv_bytes_12n())/1048576.0,MAX_STEPS+1,save_ms,ok?"BIT_EXACT":"FAIL",ok?"PASS":"FAIL");return ok?0:1;}
