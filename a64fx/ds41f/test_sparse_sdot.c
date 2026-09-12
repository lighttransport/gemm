#define _POSIX_C_SOURCE 200809L
#include "ds41f_ops.h"
#include "ds41f_cache.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static void require(int ok,const char *s){if(!ok){fprintf(stderr,"SPARSE_SDOT FAIL %s\n",s);exit(1);}}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void check(size_t raw,size_t extra,size_t heads,int mask,int bench)
{
    size_t count=raw+extra;float *q=malloc(heads*512*4),*kv=malloc((count+1)*512*4),*fp=malloc(heads*512*4);
    float *got=malloc((heads*512+1)*4),*ref=malloc(heads*512*4);uint8_t *packed=malloc((extra+1)*288);
    const uint8_t *source[512];float sink[64];int ids[640];require(q&&kv&&fp&&got&&ref&&packed,"allocation");
    for(size_t i=0;i<heads*512;++i)q[i]=sinf(i*.13f)*.75f;
    for(size_t i=0;i<count*512;++i)kv[i]=sinf(i*.23f)*.7f;
    for(size_t h=0;h<heads;++h)sink[h]=(float)((int)(h%7)-3)*.125f;
    for(size_t i=0;i<raw;++i)ids[i]=(int)i;
    for(size_t i=0;i<extra;++i){require(!ds41f_fp4_pack(packed+i*288,kv+(raw+i)*512,512,16,1),"pack key");
        require(!ds41f_fp4_unpack(kv+(raw+i)*512,packed+i*288,512,16,1),"decode PV");
        source[i]=mask==2||(mask==1&&i%3==0)?NULL:packed+i*288;ids[raw+i]=source[i]?(int)(raw+i):-1;}
    got[heads*512]=12345;
    require(!ds41f_sparse_attention_sdot(got,q,kv,sink,source,raw,extra,heads,0,0),"SDOT operator");
    require(!ds41f_sparse_attention_sdot(ref,q,kv,sink,source,raw,extra,heads,0,1),"integer reference");
    require(got[heads*512]==12345,"output canary");
    require(!ds41f_sparse_attention_tiled(fp,q,kv,sink,ids,count,count,heads,512,4),"FP32 control");
    double error=0,norm=0,ferr=0,fnorm=0;
    for(size_t i=0;i<heads*512;++i){double d=got[i]-ref[i];error+=d*d;norm+=(double)ref[i]*ref[i];d=got[i]-fp[i];ferr+=d*d;fnorm+=(double)fp[i]*fp[i];}
    if(error>1e-10*fmax(norm,1e-20)||ferr>1e-3*fmax(fnorm,1e-20))fprintf(stderr,"raw=%zu extra=%zu heads=%zu mask=%d integer_rms=%g fp_rms=%g\n",raw,extra,heads,mask,sqrt(error/fmax(norm,1e-20)),sqrt(ferr/fmax(fnorm,1e-20)));
    require(error<=1e-10*fmax(norm,1e-20),"SDOT vs integer scores");require(ferr<=1e-3*fmax(fnorm,1e-20),"bounded local FP32 error");
    if(bench){double elapsed[2]={0,0};
        for(int mode=0;mode<2;++mode)for(int it=0;it<31;++it){double start=now();
            require(!(mode?ds41f_sparse_attention_sdot(got,q,kv,sink,source,raw,extra,heads,0,0):ds41f_sparse_attention_tiled(got,q,kv,sink,ids,count,count,heads,512,4)),"bench");
            double dt=now()-start;if(it)elapsed[mode]+=dt;}
        printf("SPARSE_SDOT raw=%zu extra=%zu heads=%zu fp32_us=%.3f sdot_with_pack_us=%.3f speedup=%.3f fp_relative_rms=%.6g\n",raw,extra,heads,elapsed[0]/30*1e6,elapsed[1]/30*1e6,elapsed[0]/elapsed[1],sqrt(ferr/fnorm));
    }
    if(count){q[0]=NAN;require(ds41f_sparse_attention_sdot(got,q,kv,sink,source,raw,extra,heads,0,0)==EDOM,"nonfinite query");}
    free(q);free(kv);free(fp);free(got);free(ref);free(packed);
}
int main(int argc,char **argv)
{
    (void)argv;size_t heads[]={1,3,4,16,64};
    for(size_t h=0;h<5;++h){check(0,0,heads[h],0,0);check(1,0,heads[h],0,0);check(0,7,heads[h],2,0);check(17,13,heads[h],1,0);check(128,512,heads[h],0,0);}
    puts("SPARSE_SDOT PASS cases=25 integer_reference masks sink_only tails canaries nonfinite");
    if(argc>1){check(128,512,16,0,1);check(128,512,64,0,1);}return 0;
}
