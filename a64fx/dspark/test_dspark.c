#include "dspark_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int ok,const char*name){printf("%-30s %s\n",name,ok?"PASS":"FAIL");failures+=!ok;}

static void test_formats(void){
    check(ds_bf16_to_f32(ds_f32_to_bf16(1.5f))==1.5f,"bf16 round trip");
    check(ds_decode_e2m1(7)==6.0f&&ds_decode_e2m1(15)==-6.0f,"e2m1 decode");
    check(ds_decode_e4m3(0x38)==1.0f&&ds_decode_e4m3(1)==0x1p-9f,"e4m3 decode");
}

static void test_gemm(void){
    dspark_model m={0};m.backend=DSPARK_BACKEND_SCALAR;m.threads=2;
    uint16_t w[3*5];float x[7*5],y[7*3],ref[7*3];
    for(int i=0;i<15;i++)w[i]=ds_f32_to_bf16((float)(i-7)/8.0f);
    for(int i=0;i<35;i++)x[i]=(float)(i%9-4)/7.0f;
    ds_gemm_bf16(&m,w,3,5,x,7,y);
    for(int b=0;b<7;b++)for(int r=0;r<3;r++){float z=0;for(int k=0;k<5;k++)z=fmaf(x[b*5+k],ds_bf16_to_f32(w[r*5+k]),z);ref[b*3+r]=z;}
    float mx=0;for(int i=0;i<21;i++){float e=fabsf(y[i]-ref[i]);if(e>mx)mx=e;}
    check(mx<1e-6f,"bf16 GEMM width seven");
#if defined(__ARM_FEATURE_SVE)
    m.backend=DSPARK_BACKEND_SVE;memset(y,0,sizeof(y));ds_gemm_bf16(&m,w,3,5,x,7,y);mx=0;
    for(int i=0;i<21;i++){float e=fabsf(y[i]-ref[i]);if(e>mx)mx=e;}
    check(mx<1e-5f,"BF16 GEMM scalar/SVE");
#endif
}

static void test_norm_rope(void){
    uint16_t w[8];float x[16],y[16];for(int i=0;i<8;i++)w[i]=ds_f32_to_bf16(1.0f);for(int i=0;i<16;i++)x[i]=(float)(i+1);
    ds_rmsnorm(w,x,y,2,8,1e-6f,2);double ss=0;for(int i=0;i<8;i++)ss+=(double)y[i]*y[i];check(fabs(ss/8.0-1.0)<1e-5,"RMSNorm unit RMS");
    dspark_model m={0};m.threads=1;m.rope_attention_factor=1;for(int i=0;i<64;i++)m.rope_inv_freq[i]=i?0:1;
    float v[128]={0};v[0]=1;ds_apply_rope(&m,v,1,1,0,1);check(v[0]==1&&v[64]==0,"RoPE position zero");
    memset(v,0,sizeof(v));v[0]=1;ds_apply_rope(&m,v,1,1,1,1);check(fabsf(v[0]-cosf(1))<1e-6f&&fabsf(v[64]-sinf(1))<1e-6f,"RoPE rotate-half layout");
}

static void test_nvfp4(void){
    dspark_model m={0};m.backend=DSPARK_BACKEND_SCALAR;m.threads=2;
    ds_nvfp4_matrix w={0};w.n=16;w.k=16;w.groups=1;w.global_scale=1;w.code_bytes=128;w.scale_bytes=16;
    w.codes=calloc(1,w.code_bytes);w.scales=malloc(w.scale_bytes);memset(w.scales,0x38,w.scale_bytes);
    for(int q=0;q<8;q++)for(int lane=0;lane<16;lane++)w.codes[(q*16)+lane]=(uint8_t)(2|(uint8_t)(10<<4));
    float x[7*16],y[7*16];for(int i=0;i<7*16;i++)x[i]=(float)(i%5-2);
    ds_nvfp4_gemm(&m,&w,x,7,y);int ok=1;
    for(int r=0;r<7;r++)for(int n=0;n<16;n++){float z=0;for(int k=0;k<16;k++)z+=x[r*16+k]*(k&1?-1.0f:1.0f);if(fabsf(y[r*16+n]-z)>1e-5f)ok=0;}
    check(ok,"NVFP4 panel width seven");free(w.scales);free(w.codes);
#if defined(__ARM_FEATURE_SVE)
    w.codes=calloc(1,w.code_bytes);w.scales=malloc(w.scale_bytes);memset(w.scales,0x38,w.scale_bytes);
    for(int q=0;q<8;q++)for(int lane=0;lane<16;lane++)w.codes[(q*16)+lane]=(uint8_t)(2|(uint8_t)(10<<4));
    float ref[7*16];m.backend=DSPARK_BACKEND_SCALAR;ds_nvfp4_gemm(&m,&w,x,7,ref);m.backend=DSPARK_BACKEND_SVE;ds_nvfp4_gemm(&m,&w,x,7,y);ok=1;
    for(int i=0;i<7*16;i++)if(fabsf(y[i]-ref[i])>1e-4f)ok=0;
    check(ok,"NVFP4 scalar/SVE");free(w.scales);free(w.codes);
#endif
}

static void test_state(void){
    dspark_model m={0};m.threads=2;dspark_state*s=NULL;dspark_state_options opt={4};char err[128]={0};
    int rc=dspark_state_create(&s,&m,&opt,err,sizeof(err));check(rc==0&&s&&dspark_state_context_tokens(s)==0,"state create/reset");
    if(s){s->cursor=3;check(dspark_state_truncate(s,2)==0&&dspark_state_context_tokens(s)==2,"state truncate");check(dspark_state_truncate(s,3)==DSPARK_EINVAL,"truncate rejects growth");dspark_state_reset(s);check(dspark_state_context_tokens(s)==0,"state reset");dspark_state_free(s);}
}

#if defined(__ARM_FEATURE_SVE)
static uint32_t long_rng=7;
static float long_value(void){long_rng=long_rng*1664525u+1013904223u;return (float)((int)(long_rng>>16)-32768)/65536.0f;}

static int close_arrays(const float*a,const float*b,size_t n,double limit,double*rel_out,double*max_out){
    double se=0,sr=0,mx=0,mr=0;for(size_t i=0;i<n;i++){double e=(double)a[i]-b[i];se+=e*e;sr+=(double)b[i]*b[i];if(fabs(e)>mx)mx=fabs(e);if(fabs(b[i])>mr)mr=fabs(b[i]);}
    *rel_out=sqrt(se/(sr+1e-300));*max_out=mx/fmax(1.0,mr);return *rel_out<=limit&&*max_out<=limit*5;
}

static void test_long_k(void){
    const size_t rows=9,k=5120,mrows=7;dspark_model m={0};m.threads=4;
    uint16_t*w=malloc(rows*k*2);float*x=malloc(mrows*k*4),*scalar=malloc(mrows*rows*4),*sve=malloc(mrows*rows*4);
    for(size_t i=0;i<rows*k;i++)w[i]=ds_f32_to_bf16(long_value());for(size_t i=0;i<mrows*k;i++)x[i]=long_value();
    m.backend=DSPARK_BACKEND_SCALAR;ds_gemm_bf16(&m,w,rows,k,x,mrows,scalar);m.backend=DSPARK_BACKEND_SVE;ds_gemm_bf16(&m,w,rows,k,x,mrows,sve);
    double rel,mx;int ok=close_arrays(sve,scalar,mrows*rows,2e-5,&rel,&mx);printf("long-K GEMM rel_l2=%g scaled_max=%g\n",rel,mx);check(ok,"long-K BF16 scalar/SVE");
    float ds=ds_dot_bf16(x,w,k,DSPARK_BACKEND_SCALAR),dv=ds_dot_bf16(x,w,k,DSPARK_BACKEND_SVE);check(fabsf(ds-dv)<=2e-5f*fmaxf(1.0f,fabsf(ds)),"long-K dot scalar/SVE");
    free(sve);free(scalar);free(x);free(w);
}
#endif

int main(void){test_formats();test_gemm();test_norm_rope();test_nvfp4();test_state();
#if defined(__ARM_FEATURE_SVE)
    test_long_k();
#endif
    printf("dspark tests: %s\n",failures?"FAIL":"PASS");return failures?1:0;}
