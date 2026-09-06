/* Validate real GLM-5.3F target-layer mHC weights and Sinkhorn stability. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HC=4, WIDTH=4096, FLAT=HC*WIDTH, MIX=(2+HC)*HC };
static void *load(glm53f_st_context*st,const char*n,const char*dtype,size_t want){
    const st_tensor_info*t=glm53f_st_find(st,n,NULL);void*p;
    if(!t||strcmp(t->dtype_str,dtype)||t->nbytes!=want){fprintf(stderr,"bad tensor %s\n",n);return NULL;}
    p=malloc(want);if(!p||glm53f_st_read(st,n,0,p,want)){free(p);return NULL;}return p;
}
int main(int argc,char**argv){
    glm53f_st_context*st;uint16_t *fn;float *base,*scale,*x,*resid,*collapsed[2],*post[2],*comb[2],*out[2],sub[WIDTH];
    int layer=argc>2?atoi(argv[2]):44,ok=1;char n[256];
    if(argc<2||layer<0||layer>44){fprintf(stderr,"usage: %s MODEL_DIR [target_layer=44]\n",argv[0]);return 2;}
    st=glm53f_st_open(argv[1]);if(!st)return 2;
#define GET(V,S,D,Z) do{snprintf(n,sizeof n,"model.language_model.layers.%d.hc_attn_%s",layer,S);V=load(st,n,D,Z);if(!(V))return 2;}while(0)
    GET(fn,"fn","BF16",(size_t)MIX*FLAT*2);GET(base,"base","F32",MIX*4);GET(scale,"scale","F32",3*4);
#undef GET
    glm53f_st_close(st);x=malloc((size_t)FLAT*4);resid=malloc((size_t)FLAT*4);
    for(int p=0;p<2;p++){collapsed[p]=malloc(WIDTH*4);post[p]=malloc(HC*4);comb[p]=malloc(HC*HC*4);out[p]=malloc((size_t)FLAT*4);}
    if(!x||!resid||!collapsed[0]||!collapsed[1]||!post[0]||!post[1]||!comb[0]||!comb[1]||!out[0]||!out[1])return 2;
    for(int i=0;i<FLAT;i++)x[i]=(float)(((i*17+3)%251)-125)/125.0f;
    for(int i=0;i<WIDTH;i++)sub[i]=(float)(((i*11+7)%127)-63)/63.0f;
    memcpy(resid,x,(size_t)FLAT*4);
    for(int p=0;p<2;p++){glm53f_mhc_pre(collapsed[p],post[p],comb[p],x,fn,base,scale,HC,WIDTH,20,1e-5f,1e-6f);glm53f_mhc_post(out[p],resid,sub,post[p],comb[p],HC,WIDTH);}
    ok&=!memcmp(collapsed[0],collapsed[1],WIDTH*4)&&!memcmp(post[0],post[1],HC*4)&&!memcmp(comb[0],comb[1],HC*HC*4)&&!memcmp(out[0],out[1],(size_t)FLAT*4);
    float row_err=0,col_err=0;for(int i=0;i<HC;i++){float r=0,c=0;for(int j=0;j<HC;j++){r+=comb[0][i*HC+j];c+=comb[0][j*HC+i];}if(fabsf(r-1)>row_err)row_err=fabsf(r-1);if(fabsf(c-1)>col_err)col_err=fabsf(c-1);}
    double ss=0;for(int i=0;i<FLAT;i++){ss+=(double)out[0][i]*out[0][i];ok&=isfinite(out[0][i]);}
    printf("GLM53F_MHC layer=%d repeat=%s sinkhorn_row_err=%.9g col_err=%.9g out_rms=%.9g %s\n",layer,ok?"BIT_EXACT":"FAIL",row_err,col_err,sqrt(ss/FLAT),ok&&row_err<5e-3f&&col_err<2e-5f?"PASS":"FAIL");
    return ok&&row_err<5e-3f&&col_err<2e-5f?0:1;
}
