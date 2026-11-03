#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "glm53f_mhc_sve.h"

int main(void) {
    uint16_t *weights=malloc((size_t)GLM53F_MHC_MIX*GLM53F_MHC_FLAT*2);
    uint16_t norm[GLM53F_MHC_WIDTH];
    float base[24]={0},scale[3]={0.1f,0.1f,0.1f};
    float *input=malloc(5UL*GLM53F_MHC_FLAT*sizeof(float));
    float *a=malloc(5UL*GLM53F_MHC_WIDTH*sizeof(float));
    float *b=malloc(5UL*GLM53F_MHC_WIDTH*sizeof(float));
    glm53f_mhc_scratch *sa=calloc(5,sizeof(*sa)),*sb=calloc(5,sizeof(*sb));
    if(!weights||!input||!a||!b||!sa||!sb)return 2;
    for(int i=0;i<GLM53F_MHC_MIX*GLM53F_MHC_FLAT;i++){
        float x=(float)((i*17+31)%251-125)/8192;uint32_t u;
        memcpy(&u,&x,4);weights[i]=(uint16_t)(u>>16);
    }
    for(int i=0;i<GLM53F_MHC_WIDTH;i++){
        float x=0.5f+(float)(i%127)/127;uint32_t u;
        memcpy(&u,&x,4);norm[i]=(uint16_t)(u>>16);
    }
    for(int i=0;i<24;i++)base[i]=(float)(i-12)/32;
    scale[1]=0.2f;scale[2]=0.3f;
    for(int i=0;i<5*GLM53F_MHC_FLAT;i++)input[i]=sinf((float)i*0.017f);
    glm53f_mhc_site site={weights,base,scale};
    int pass=1;
    for(int n=1;n<=5;n++){
        setenv("GLM53F_MHC_BATCH_TEAM","0",1);
        glm53f_mhc_pre_batch_sve(sa,input,&site,norm,n,sizeof(*sa),a);
        setenv("GLM53F_MHC_BATCH_TEAM","1",1);
        glm53f_mhc_pre_batch_sve(sb,input,&site,norm,n,sizeof(*sb),b);
        double maxerr=0;
        for(size_t i=0;i<(size_t)n*sizeof(*sa)/sizeof(float);i++){
            double e=fabs((double)((float*)sa)[i]-((float*)sb)[i]);
            if(!isfinite(e)){pass=0;break;}
            if(e>maxerr)maxerr=e;
        }
        for(int i=0;i<n*GLM53F_MHC_WIDTH;i++)if(a[i]!=b[i])pass=0;
        if(maxerr>1e-6)pass=0;
        printf("MHC_TEAM tokens=%d scratch_max_abs=%.9g normalized_exact=%s\n",n,maxerr,pass?"PASS":"FAIL");
        float *pa=malloc((size_t)n*GLM53F_MHC_FLAT*sizeof(float));
        float *pb=malloc((size_t)n*GLM53F_MHC_FLAT*sizeof(float));
        if(!pa||!pb)return 2;
        setenv("GLM53F_MHC_POST_TILES","0",1);
        glm53f_mhc_post_batch_sve(pa,a,sa,n,sizeof(*sa));
        setenv("GLM53F_MHC_POST_TILES","1",1);
        glm53f_mhc_post_batch_sve(pb,a,sa,n,sizeof(*sa));
        if(memcmp(pa,pb,(size_t)n*GLM53F_MHC_FLAT*sizeof(float)))pass=0;
        printf("MHC_POST_TILES tokens=%d %s\n",n,pass?"PASS":"FAIL");
        free(pb);free(pa);
    }
    free(sb);free(sa);free(b);free(a);free(input);free(weights);
    return pass?0:1;
}
