#include "ds41f_ops.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void)
{
    const size_t lengths[]={1,7,8,9,15,31,32,33,65,5120,20480};
    float x[20480];size_t norms=0;
    uint32_t rng=19;
    for(int trial=0;trial<32;++trial){
        for(size_t i=0;i<20480;++i){rng=rng*1664525u+1013904223u;
            uint32_t bits=(rng&0x80000000u)|((uint32_t)(112+(rng%24))<<23)|(rng&0x007f0000u);
            memcpy(x+i,&bits,4);}
        for(size_t shape=0;shape<sizeof lengths/sizeof lengths[0];++shape){size_t n=lengths[shape];
            double ss=0;for(size_t i=0;i<n;++i)ss+=(double)x[i]*x[i];
            float ref=(float)(1/sqrt(ss/n+1e-20)),got=ds41f_hc_inverse_rms(x,n);
            if(got!=ref){fprintf(stderr,"MHC_NORM FAIL n=%zu ref=%g got=%g\n",n,ref,got);return 1;}
            ++norms;
        }
    }
    if(ds41f_hc_inverse_rms(NULL,0)!=0)return 1;
    printf("MHC_MIX PASS norm_bit_exact=%zu tails\n",norms);
    return 0;
}
