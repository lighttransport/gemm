#include "ds41f_ops.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float product(float a,float b)
{volatile float rounded=a*b;return rounded;}
static void exact(const float *a,const float *b,size_t n,const char *label)
{
    if(memcmp(a,b,n*sizeof(float))){fprintf(stderr,"FAIL %s\n",label);exit(1);}
}
int main(void)
{
    const size_t sizes[]={1,511,512,513,2304,5120,5123};
    for(size_t test=0;test<sizeof sizes/sizeof sizes[0];++test){size_t n=sizes[test];
        float *x=malloc((4*n+1)*sizeof(float)),*gate=malloc(n*sizeof(float));
        float *up=malloc(n*sizeof(float)),*out=malloc((4*n+1)*sizeof(float));
        float *expected=malloc(4*n*sizeof(float));
        if(!x||!gate||!up||!out||!expected)return 2;
        for(size_t i=0;i<n;++i){gate[i]=sinf((float)i*.17f)*20;up[i]=cosf((float)i*.31f)*20;}
        for(int limited=0;limited<2;++limited){float limit=limited?10:0;
            for(size_t i=0;i<n;++i){float g=gate[i],u=up[i];
                if(limit>0){g=fminf(g,limit);u=fminf(limit,fmaxf(-limit,u));}
                float sigmoid=g>=0?1/(1+expf(-g)):expf(g)/(1+expf(g));
                expected[i]=g*sigmoid*u;}
            out[n]=12345;ds41f_swiglu(out,gate,up,n,limit);
            exact(out,expected,n,"parallel SwiGLU vs scalar");if(out[n]!=12345)return 1;
        }
        float post[4]={.25f,-.3f,1.3f,.75f},comb[16];
        for(int i=0;i<16;++i)comb[i]=sinf((float)i*.713f);
        for(size_t i=0;i<4*n;++i)x[i]=cosf((float)i*.191f)*100;
        for(size_t j=0;j<n;++j)for(int o=0;o<4;++o){float sum=0;
            for(int i=0;i<4;++i)sum+=product(comb[i*4+o],x[i*n+j]);
            expected[o*n+j]=product(post[o],gate[j])+sum;}
        out[4*n]=12345;ds41f_hc_post(out,gate,x,post,comb,n);
        exact(out,expected,4*n,"parallel mHC post separate products");if(out[4*n]!=12345)return 1;
        x[4*n]=12345;ds41f_hc_post(x,gate,x,post,comb,n);
        exact(x,expected,4*n,"parallel mHC post in-place");if(x[4*n]!=12345)return 1;
        free(x);free(gate);free(up);free(out);free(expected);
    }
    puts("POINTWISE PASS bit_exact SwiGLU mHC_post in_place sizes=1,511,512,513,2304,5120,5123");
    return 0;
}
