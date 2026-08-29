/* Real-weight two-site mHC/RMSNorm decoder-layer state transition. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "glm53f_mhc_sve.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HC=GLM53F_MHC_STREAMS,W=GLM53F_MHC_WIDTH,FLAT=GLM53F_MHC_FLAT,MIX=GLM53F_MHC_MIX };
typedef struct{uint16_t*fn;float*base,*scale;} site;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void*load(glm53f_st_context*s,const char*n,size_t z){const st_tensor_info*t=glm53f_st_find(s,n,NULL);void*p;if(!t||t->nbytes!=z||!(p=malloc(z))||glm53f_st_read(s,n,0,p,z)){fprintf(stderr,"bad %s\n",n);return NULL;}return p;}
static int get_site(glm53f_st_context*st,int layer,const char*which,site*s){char n[256];snprintf(n,sizeof n,"model.language_model.layers.%d.hc_%s_fn",layer,which);s->fn=load(st,n,(size_t)MIX*FLAT*2);snprintf(n,sizeof n,"model.language_model.layers.%d.hc_%s_base",layer,which);s->base=load(st,n,MIX*4);snprintf(n,sizeof n,"model.language_model.layers.%d.hc_%s_scale",layer,which);s->scale=load(st,n,3*4);return s->fn&&s->base&&s->scale?0:-1;}
static void apply_site(float*streams,const site*s,const uint16_t*norm,
                       const float*sublayer,float*collapsed,float*normalized,
                       float*post,float*comb,float*residual){if(getenv("GLM53F_MHC_SCALAR")){memcpy(residual,streams,(size_t)FLAT*4);glm53f_mhc_pre(collapsed,post,comb,streams,s->fn,s->base,s->scale,HC,W,20,1e-5f,1e-6f);glm53f_rmsnorm_bf16(normalized,collapsed,norm,W,1e-5f);glm53f_mhc_post(streams,residual,sublayer,post,comb,HC,W);}else{glm53f_mhc_site ws={s->fn,s->base,s->scale};glm53f_mhc_scratch scratch;glm53f_mhc_pre_sve(&scratch,streams,&ws,norm);memcpy(collapsed,scratch.collapsed,W*4);memcpy(normalized,scratch.normalized,W*4);memcpy(post,scratch.post,HC*4);memcpy(comb,scratch.combine,HC*HC*4);memcpy(residual,scratch.residual,(size_t)FLAT*4);glm53f_mhc_post_sve(streams,sublayer,&scratch);}}
int main(int argc,char**argv){int layer=argc>2?atoi(argv[2]):44;char n[256];glm53f_st_context*st;if(argc<2)return 2;st=glm53f_st_open(argv[1]);if(!st)return 2;site a={0},f={0};if(get_site(st,layer,"attn",&a)||get_site(st,layer,"ffn",&f))return 2;snprintf(n,sizeof n,"model.language_model.layers.%d.input_layernorm.weight",layer);uint16_t*an=load(st,n,W*2);snprintf(n,sizeof n,"model.language_model.layers.%d.post_attention_layernorm.weight",layer);uint16_t*fn=load(st,n,W*2);glm53f_st_close(st);float *initial=malloc((size_t)FLAT*4),*streams[2],*collapsed=malloc(W*4),*normalized=malloc(W*4),*post=malloc(HC*4),*comb=malloc(HC*HC*4),*resid=malloc((size_t)FLAT*4),*attn=malloc(W*4),*mlp=malloc(W*4);if(!initial||!an||!fn||!collapsed||!normalized||!post||!comb||!resid||!attn||!mlp)return 2;for(int i=0;i<FLAT;i++)initial[i]=(float)(((i*17+3)%251)-125)/125.0f;for(int i=0;i<W;i++){attn[i]=(float)(((i*11+7)%127)-63)/63.0f;mlp[i]=(float)(((i*23+9)%131)-65)/65.0f;}double sec[2];for(int p=0;p<2;p++){streams[p]=malloc((size_t)FLAT*4);memcpy(streams[p],initial,(size_t)FLAT*4);double t=now();apply_site(streams[p],&a,an,attn,collapsed,normalized,post,comb,resid);apply_site(streams[p],&f,fn,mlp,collapsed,normalized,post,comb,resid);sec[p]=now()-t;}int ok=!memcmp(streams[0],streams[1],(size_t)FLAT*4);double ss=0,mean_ss=0;for(int i=0;i<FLAT;i++){ss+=(double)streams[0][i]*streams[0][i];ok&=isfinite(streams[0][i]);}for(int d=0;d<W;d++){double z=0;for(int h=0;h<HC;h++)z+=streams[0][(size_t)h*W+d];z/=HC;mean_ss+=z*z;}printf("GLM53F_LAYER_STATE layer=%d two_site_ms=%.3f repeat=%s streams_rms=%.9g collapsed_mean_rms=%.9g %s\n",layer,sec[1]*1e3,ok?"BIT_EXACT":"FAIL",sqrt(ss/FLAT),sqrt(mean_ss/W),ok?"PASS":"FAIL");const char*d=getenv("GLM53F_MHC_OUTPUT");if(d&&*d){FILE*fout=fopen(d,"wb");if(!fout||fwrite(streams[0],4,FLAT,fout)!=(size_t)FLAT)return 2;fclose(fout);}return ok?0:1;}
