#define _POSIX_C_SOURCE 200809L
#include "ds41f_team.h"
#include "ds41f_ops.h"
#include "ds41f_sve.h"
#include "ds41f_kernels.h"
#include "ds41f_int8.h"
#include "ds41f_fp4_sdot.h"
#include "ds41f_cache.h"
#include "ds41f_attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
typedef struct {
    float *out,*x,*hc,*kv;uint16_t *bf;
    uint8_t *w,*scale,*fp4_scale,*packed,*packed_scale,*cache;
    ds41f_int8 q;size_t used;int failed;
} test_job;
static void check(test_job *j,int rc){if(rc)j->failed=1;}
static void compute(void *context)
{
    test_job *j=context;float *out=j->out;size_t n=0;
    check(j,ds41f_act_quant(out,j->x,5120));n+=5120;
    check(j,ds41f_int8_matvec(out+n,&j->q,j->x,64,0));n+=64;
    check(j,ds41f_int8_matvec(out+n,&j->q,j->x,32,0));n+=64;
    ds41f_bf16_f32_matvec(out+n,j->bf,j->x,64,5120);n+=64;
    check(j,ds41f_fp8_matvec(out+n,j->w,j->scale,j->x,64,5120));n+=64;
    check(j,ds41f_mxfp4_matvec(out+n,j->w,j->fp4_scale,j->x,64,5120));n+=64;
    check(j,ds41f_mxfp4_matvec_pair(out+n,out+n+64,j->w,j->fp4_scale,j->w,j->fp4_scale,j->x,64,5120,1));n+=128;
    check(j,ds41f_mxfp4_sdot(out+n,j->packed,j->packed_scale,j->x,64,5120,0));n+=64;
    ds41f_swiglu(out+n,j->x,j->x+5120,2304,10);n+=2304;
    for(int mode=0;mode<3;++mode){check(j,ds41f_hc_matvec(out+n,j->hc,j->x,.02f,mode));n+=24;}
    float post[4]={.3f,.5f,-.2f,.1f},comb[16];for(int i=0;i<16;++i)comb[i]=(i%7-3)*.125f;
    ds41f_hc_post(out+n,j->x,j->x,post,comb,5120);n+=20480;
    float sink[16],iw[32];int ids[64];for(int i=0;i<16;++i)sink[i]=i*.125f;
    for(int i=0;i<64;++i)ids[i]=i%7==0?-1:i;
    check(j,ds41f_sparse_attention_tiled(out+n,j->x,j->kv,sink,ids,64,64,16,512,4));n+=16*512;
    for(int i=0;i<32;++i)iw[i]=(i%5)*.03125f;
    check(j,ds41f_index_scores(out+n,j->x,iw,j->cache,64,NULL,1));n+=64;
    int selected[6];float bias[384]={0};check(j,ds41f_gate(j->x,bias,384,6,1.25f,2.5f,selected,out+n));n+=6;
    for(int i=0;i<6;++i)out[n++]=(float)selected[i];
    j->used=n;
    float saved=j->x[0];j->x[0]=NAN;
    if(ds41f_act_quant(out+n,j->x,5120)!=EDOM)j->failed=1;
    j->x[0]=saved;
}
int main(void)
{
    test_job j={.used=0};size_t values=64*5120;
    j.x=malloc(20480*4);j.hc=malloc(24*20480*4);j.kv=malloc(64*512*4);j.bf=malloc(values*2);
    j.w=malloc(values);j.scale=malloc(320);j.fp4_scale=malloc(values/32);j.packed=malloc(values/2);j.packed_scale=malloc(values/32);j.cache=malloc(64*356);
    float *ref=calloc(65536,4),*got=calloc(65536,4);
    if(!j.x||!j.hc||!j.kv||!j.bf||!j.w||!j.scale||!j.fp4_scale||!j.packed||!j.packed_scale||!j.cache||!ref||!got)return 1;
    for(size_t i=0;i<20480;++i)j.x[i]=sinf(i*.13f)*.75f;
    for(size_t i=0;i<24*20480;++i)j.hc[i]=(float)((int)(i%31)-15)*.00390625f;
    for(size_t i=0;i<64*512;++i)j.kv[i]=sinf(i*.23f)*.7f;
    for(size_t i=0;i<values;++i){j.w[i]=(uint8_t)((i*17+i/5120)%112);j.bf[i]=ds41f_f32_to_bf16((float)((int)(i%13)-6)*.03125f);}
    for(size_t i=0;i<320;++i)j.scale[i]=(uint8_t)(120+i%12);
    for(size_t i=0;i<values/32;++i)j.fp4_scale[i]=(uint8_t)(120+i%12);
    for(size_t r=0;r<64;++r){check(&j,ds41f_fp4_pack(j.cache+r*356,j.kv+r*512,512,16,1));check(&j,ds41f_fp4_pack(j.cache+r*356+288,j.kv+r*512,128,32,0));}
    check(&j,ds41f_int8_from_fp8(&j.q,j.w,j.scale,64,5120,32));
    check(&j,ds41f_mxfp4_pack_sdot(j.packed,j.packed_scale,j.w,j.fp4_scale,64,5120));
    ds41f_set_quant_parallel(1);j.out=ref;compute(&j);size_t expected=j.used;
    j.out=got;check(&j,ds41f_team_run(compute,&j));
    if(j.used!=expected||memcmp(ref,got,expected*4)){for(size_t i=0;i<expected;++i)if(memcmp(ref+i,got+i,4)){fprintf(stderr,"TEAM_KERNEL mismatch offset=%zu reference=%g actual=%g\n",i,ref[i],got[i]);break;}j.failed=1;}
    ds41f_int8_free(&j.q);free(j.x);free(j.hc);free(j.kv);free(j.bf);free(j.w);free(j.scale);free(j.fp4_scale);free(j.packed);free(j.packed_scale);free(j.cache);free(ref);free(got);
    if(j.failed){puts("TEAM_KERNEL FAIL");return 1;}
    printf("TEAM_KERNEL PASS bit_exact floats=%zu FP8 INT8 FP4 BF16 quantization mHC gate index attention nonfinite\n",expected);return 0;
}
