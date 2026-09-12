#include "ds41f_cache.h"
#include "ds41f_kernels.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
static const float fp4_values[8]={0,.5f,1,1.5f,2,3,4,6};
static unsigned encode(float x)
{
    float a=fabsf(x);unsigned best=0;
    for(unsigned i=1;i<8;++i){float d=fabsf(a-fp4_values[i]),old=fabsf(a-fp4_values[best]);
        if(d<old||(d==old&&!(i&1)))best=i;}
    return best|(signbit(x)?8:0);
}
int ds41f_fp4_pack(uint8_t *out,const float *x,size_t dim,size_t group,int e4)
{
    if(!out||!x||!dim||(group!=16&&group!=32)||dim%group)return EINVAL;
    memset(out,0,dim/2+dim/group);
    for(size_t b=0;b<dim;b+=group){float a=e4?6.f/512:6.f*0x1p-126f;
        float v[32];
        for(size_t j=0;j<group;++j){v[j]=ds41f_bf16_to_f32(ds41f_f32_to_bf16(x[b+j]));
            if(!isfinite(v[j]))return EDOM;
            a=fmaxf(a,fabsf(v[j]));}
        float s;uint8_t code;
        if(e4){code=ds41f_f32_to_fp8(a/6);s=ds41f_fp8_e4m3_to_f32(code);}
        else{int exp=(int)ceilf(log2f(a/6));if(exp>127||exp< -126)return ERANGE;
            code=(uint8_t)(exp+127);s=ldexpf(1,exp);}
        out[dim/2+b/group]=code;
        for(size_t j=0;j<group;++j){size_t i=b+j;unsigned q=encode(v[j]/s);
            out[i/2]|=(uint8_t)(q<<((i&1)*4));}
    }
    return 0;
}
int ds41f_fp4_unpack(float *out,const uint8_t *row,size_t dim,size_t group,int e4)
{
    if(!out||!row||!dim||(group!=16&&group!=32)||dim%group)return EINVAL;
    static const float values[16]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
    for(size_t base=0;base<dim;base+=group){uint8_t sc=row[dim/2+base/group];
        float scale=e4?ds41f_fp8_e4m3_to_f32(sc):ds41f_e8m0_to_f32(sc);
        /* E2M1 times E4M3 (or E8M0) is exactly representable in BF16.
         * Decode the shared scale once, not once for every value. */
        for(size_t j=0;j<group;++j){size_t i=base+j;unsigned code=(row[i/2]>>((i&1)*4))&15;
            out[i]=values[code]*scale;}}
    return 0;
}
/* Heap root is the least desirable selected element. */
static int worse(const float *s,int a,int b)
{return s[a]<s[b]||(s[a]==s[b]&&a>b);}
static int ascending(const void *a,const void *b)
{int x=*(const int *)a,y=*(const int *)b;return (x>y)-(x<y);}
size_t ds41f_select_topk(const float *s,size_t n,size_t k,int *ids)
{
    size_t used=0;if(!s||!ids||!k)return 0;
    if(n<=k){for(size_t i=0;i<n;++i)if(!isnan(s[i])&&s[i]!=-INFINITY)ids[used++]=(int)i;return used;}
    for(size_t i=0;i<n;++i){if(isnan(s[i])||s[i]==-INFINITY)continue;
        if(used<k){size_t c=used++;ids[c]=(int)i;
            while(c){size_t p=(c-1)/2;if(!worse(s,ids[c],ids[p]))break;
                int t=ids[p];ids[p]=ids[c];ids[c]=t;c=p;}}
        else if(worse(s,ids[0],(int)i)){ids[0]=(int)i;size_t p=0;
            while(2*p+1<used){size_t c=2*p+1;if(c+1<used&&worse(s,ids[c+1],ids[c]))++c;
                if(!worse(s,ids[c],ids[p]))break;
                int t=ids[p];ids[p]=ids[c];ids[c]=t;p=c;}}
    }
    if(n<=4096){uint8_t selected[4096]={0};
        for(size_t i=0;i<used;++i)selected[ids[i]]=1;
        size_t out=0;for(size_t i=0;i<n;++i)if(selected[i])ids[out++]=(int)i;
    }else qsort(ids,used,sizeof *ids,ascending);
    return used;
}
