#include "ds41f_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void reference(float *x,size_t heads,size_t dim,size_t rd,size_t pos,
                      double theta,double factor,int original,int inverse)
{
    const double pi=3.14159265358979323846;double low=0,high=0;
    if(original){low=fmax(floor(rd*log(original/(32*2*pi))/(2*log(theta))),0);
        high=fmin(ceil(rd*log(original/(2*pi))/(2*log(theta))),rd-1);}
    for(size_t j=0;j<rd/2;++j){double freq=pow(theta,-2.0*j/rd);
        if(original){double ramp=fmax(0,fmin(1,(j-low)/fmax(high-low,1e-3)));freq*=1-ramp+ramp/factor;}
        double angle=pos*freq*(inverse?-1:1),c=cos(angle),s=sin(angle);
        for(size_t h=0;h<heads;++h){size_t i=h*dim+dim-rd+2*j;float a=x[i],b=x[i+1];
            x[i]=(float)(a*c-b*s);x[i+1]=(float)(a*s+b*c);}
    }
}
int main(void)
{
    const size_t heads[]={1,32,64},dims[]={128,512},rotary[]={4,64,128},positions[]={0,1,127,1000,1048575};
    float *x=malloc((64*512+1)*sizeof(float)),*y=malloc((64*512+1)*sizeof(float));
    if(!x||!y)return 2;
    size_t count=0;
    for(int cached=0;cached<2;++cached)
    for(size_t h=0;h<3;++h)for(size_t d=0;d<2;++d)for(size_t r=0;r<3;++r)
    for(size_t p=0;p<5;++p)for(int cfg=0;cfg<2;++cfg)for(int inverse=0;inverse<2;++inverse){
        ds41f_set_rope_cache(cached);
        size_t n=heads[h]*dims[d];for(size_t i=0;i<n;++i)x[i]=sinf((float)i*.013f);
        x[n]=12345;memcpy(y,x,(n+1)*sizeof(float));double theta=cfg?160000:10000;int original=cfg?65536:0;
        reference(x,heads[h],dims[d],rotary[r],positions[p],theta,16,original,inverse);
        ds41f_rope(y,heads[h],dims[d],rotary[r],positions[p],theta,16,original,inverse);
        if(memcmp(x,y,(n+1)*sizeof(float))||y[n]!=12345){fprintf(stderr,"ROPE_LAYOUT FAIL heads=%zu dim=%zu rd=%zu pos=%zu cfg=%d inverse=%d\n",heads[h],dims[d],rotary[r],positions[p],cfg,inverse);return 1;}
        if(cached){for(size_t i=0;i<n;++i)y[i]=sinf((float)i*.013f);
            ds41f_rope(y,heads[h],dims[d],rotary[r],positions[p],theta,16,original,inverse);
            if(memcmp(x,y,(n+1)*sizeof(float)))return 1;}
        ++count;
    }
    free(x);free(y);printf("ROPE_LAYOUT PASS bit_exact=%zu canaries inverse long_positions\n",count);return 0;
}
