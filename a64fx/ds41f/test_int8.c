#include "ds41f_int8.h"
#include "ds41f_kernels.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(int ok,const char *message)
{if(!ok){fprintf(stderr,"INT8 FAIL %s\n",message);exit(1);}}
int main(void)
{
    const size_t row_counts[]={1,3,4,5,31,32,33,64};
    const size_t lengths[]={32,64,128,256,512,4096};
    size_t cases=0;
    for(size_t ri=0;ri<sizeof row_counts/sizeof row_counts[0];++ri)
    for(size_t ci=0;ci<sizeof lengths/sizeof lengths[0];++ci)
    for(size_t block=32;block<=256;block*=2){
        size_t rows=row_counts[ri],cols=lengths[ci];if(cols%block)continue;
        uint8_t *w=malloc(rows*cols),*s=malloc(((rows+31)/32)*(cols/32));
        float *x=malloc(cols*2*sizeof(float)),*got=malloc((rows+1)*sizeof(float));
        float *ref=malloc(rows*sizeof(float));
        require(w&&s&&x&&got&&ref,"allocation");
        for(size_t i=0;i<rows*cols;++i)w[i]=(uint8_t)((i*17+i/cols*3)%112)|((i%3)?0:128);
        for(size_t i=0;i<((rows+31)/32)*(cols/32);++i)s[i]=(uint8_t)(122+i%7);
        for(size_t i=0;i<cols*2;++i)x[i]=sinf((float)i*.071f)*.75f;
        ds41f_int8 q;require(!ds41f_int8_from_fp8(&q,w,s,rows,cols,block),"convert");
        for(int grouped=0;grouped<2;++grouped){
            if(grouped&&rows!=64)continue;
            size_t group=grouped?32:rows;got[rows]=12345;
            require(!ds41f_int8_matvec(got,&q,x,group,0),"kernel");
            require(!ds41f_int8_matvec(ref,&q,x,group,1),"integer reference");
            double error=0,norm=0;
            for(size_t r=0;r<rows;++r){double d=got[r]-ref[r];error+=d*d;norm+=(double)ref[r]*ref[r];}
            require(error<=1e-10*fmax(norm,1e-20),"SDOT vs integer/double reference");
            require(got[rows]==12345,"output canary");
            ds41f_int8_input prepared;
            require(!ds41f_int8_prepare_input(&prepared,x,(rows/group)*cols,block),"prepare shared input");
            require(!ds41f_int8_matvec_prepared(ref,&q,&prepared,group,0),"prepared kernel");
            require(!memcmp(got,ref,rows*sizeof(float)),"prepared kernel bit exact");
            ds41f_int8_input_free(&prepared);++cases;
        }
        memset(x,0,cols*sizeof(float));require(!ds41f_int8_matvec(got,&q,x,rows,0),"zero input");
        for(size_t r=0;r<rows;++r)require(got[r]==0,"zero result");
        x[0]=NAN;require(ds41f_int8_matvec(got,&q,x,rows,0)==EDOM,"nonfinite input");
        ds41f_int8_free(&q);
        w[0]=127;require(ds41f_int8_from_fp8(&q,w,s,rows,cols,block)==EDOM,"NaN weight");
        require(!q.weight&&!q.scale,"failed conversion cleanup");
        w[0]=0;s[0]=255;require(ds41f_int8_from_fp8(&q,w,s,rows,cols,block)==EDOM,"NaN scale");
        free(w);free(s);free(x);free(got);free(ref);
    }
    for(int tiny_input=0;tiny_input<2;++tiny_input){
        uint8_t w[32],sc=(uint8_t)(tiny_input?254:0);float x[32],out;
        memset(w,56,sizeof w);for(int i=0;i<32;++i)x[i]=tiny_input?1e-38f:1e38f;
        ds41f_int8 q;require(!ds41f_int8_from_fp8(&q,w,&sc,1,32,32),"subnormal conversion");
        require(!ds41f_int8_matvec(&out,&q,x,1,0),"subnormal scale kernel");
        double expected=32.*ldexp(1.,(int)sc-127)*x[0];
        require(isfinite(out)&&fabs(out-expected)/expected<1e-3,"subnormal scale accuracy");
        ds41f_int8_free(&q);
    }
    printf("INT8 PASS cases=%zu blocks=32,64,128,256 padded_rows grouped canaries zero nonfinite subnormal_scales\n",cases);
    return 0;
}
