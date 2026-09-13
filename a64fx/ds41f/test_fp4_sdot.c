#define _POSIX_C_SOURCE 200809L
#include "ds41f_fp4_sdot.h"
#include "ds41f_sve.h"
#include "ds41f_kernels.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static void require(int ok,const char *s){if(!ok){fprintf(stderr,"FP4_SDOT FAIL %s\n",s);exit(1);}}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void check(size_t rows,size_t cols,int bench)
{
    size_t bytes=rows*cols/2,blocks=cols/32;
    uint8_t *w=malloc(bytes),*p=malloc(bytes),*s=malloc(rows*blocks),*ps=malloc(rows*blocks);
    float *x=malloc(cols*4),*out=malloc((rows+1)*4),*ref=malloc(rows*4),*fp=malloc(rows*4);
    require(w&&p&&s&&ps&&x&&out&&ref&&fp,"allocation");
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<bytes;++i)w[i]=(uint8_t)(i*73+i/17);
    for(size_t i=0;i<rows*blocks;++i)s[i]=(uint8_t)(120+i%12);
    for(size_t i=0;i<cols;++i)x[i]=sinf(i*.13f)*.75f;
    require(!ds41f_act_quant(x,x,cols),"FP8 input boundary");
    double start=now();require(!ds41f_mxfp4_pack_sdot(p,ps,w,s,rows,cols),"pack");double pack=now()-start;
    for(size_t r=0;r<rows;r+=4)for(size_t b=0;b<blocks;++b)for(size_t j=0;j<4;++j){
        require(!memcmp(p+r*(cols/2)+b*64+j*16,w+(r+j)*(cols/2)+b*16,16),"lossless packing");
        require(ps[r*blocks+b*4+j]==s[(r+j)*blocks+b],"lossless scales");}
    out[rows]=12345;
    require(!ds41f_mxfp4_sdot(out,p,ps,x,rows,cols,0),"SDOT");
    require(!ds41f_mxfp4_sdot(ref,p,ps,x,rows,cols,1),"integer oracle");
    require(!ds41f_mxfp4_matvec(fp,w,s,x,rows,cols),"FP32 control");
    double err=0,norm=0,fp_err=0,fp_norm=0;
    for(size_t r=0;r<rows;++r){double d=out[r]-ref[r];err+=d*d;norm+=(double)ref[r]*ref[r];d=out[r]-fp[r];fp_err+=d*d;fp_norm+=(double)fp[r]*fp[r];}
    require(err<=1e-10*fmax(norm,1e-20),"SDOT vs integer oracle");require(out[rows]==12345,"canary");
    if(bench){size_t n=16*1024*1024;float *flush=malloc(n*4);require(flush!=NULL,"flush");
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i)flush[i]=(float)(i%7);
        volatile float guard=0;double elapsed[2]={0,0};
        for(int mode=0;mode<2;++mode)for(int it=0;it<31;++it){float value=0;
            #pragma omp parallel for reduction(+:value) schedule(static)
            for(size_t i=0;i<n;i+=16)value+=flush[i];guard=value;start=now();
            require(!(mode?ds41f_mxfp4_sdot(out,p,ps,x,rows,cols,0):ds41f_mxfp4_matvec(out,w,s,x,rows,cols)),"benchmark");
            double dt=now()-start;if(it)elapsed[mode]+=dt;
        }
        printf("FP4_SDOT rows=%zu cols=%zu pack_ms=%.3f fp32_us=%.3f sdot_us=%.3f speedup=%.3f rel_rms=%.6g guard=%g\n",rows,cols,pack*1e3,elapsed[0]/30*1e6,elapsed[1]/30*1e6,elapsed[0]/elapsed[1],sqrt(fp_err/fp_norm),guard);free(flush);
    }
    float *pair=malloc((rows*2+1)*4);uint8_t *second_scale=malloc(rows*blocks),*second=malloc(bytes);
    require(pair&&second_scale&&second,"pair allocation");
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<bytes;++i)second[i]=p[i];
    for(size_t i=0;i<rows*blocks;++i)second_scale[i]=(uint8_t)(ps[i]+1);
    ds41f_int8_input input;require(!ds41f_int8_prepare_input(&input,x,cols,32),"pair input");
    require(!ds41f_mxfp4_sdot_prepared(out,p,ps,&input,rows,cols,0),"pair first control");
    require(!ds41f_mxfp4_sdot_prepared(ref,second,second_scale,&input,rows,cols,0),"pair second control");
    pair[rows*2]=12345;
    require(!ds41f_mxfp4_sdot_pair_prepared(pair,pair+rows,p,ps,second,second_scale,&input,rows,cols),"paired kernel");
    require(!memcmp(pair,out,rows*4)&&!memcmp(pair+rows,ref,rows*4)&&pair[rows*2]==12345,"pair bit exact and canary");
    if(bench){double elapsed[2]={0,0};size_t count=16*1024*1024;float *flush=malloc(count*4);require(flush!=NULL,"pair flush");
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<count;++i)flush[i]=(float)(i%7);
        volatile float guard=0;
        for(int mode=0;mode<2;++mode)for(int it=0;it<31;++it){float value=0;
            #pragma omp parallel for reduction(+:value) schedule(static)
            for(size_t i=0;i<count;i+=16)value+=flush[i];guard=value;
            double start=now();
            if(mode)require(!ds41f_mxfp4_sdot_pair_prepared(pair,pair+rows,p,ps,second,second_scale,&input,rows,cols),"pair bench");
            else{require(!ds41f_mxfp4_sdot_prepared(pair,p,ps,&input,rows,cols,0),"pair bench first");require(!ds41f_mxfp4_sdot_prepared(pair+rows,second,second_scale,&input,rows,cols,0),"pair bench second");}
            double dt=now()-start;if(it)elapsed[mode]+=dt;}
        printf("FP4_SDOT_PAIR rows=%zu cols=%zu cold_two_gemv_us=%.3f cold_pair_us=%.3f speedup=%.3f guard=%g\n",rows,cols,elapsed[0]/30*1e6,elapsed[1]/30*1e6,elapsed[0]/elapsed[1],guard);free(flush);}
    ds41f_int8_input_free(&input);free(pair);free(second_scale);free(second);
    memset(x,0,cols*4);require(!ds41f_mxfp4_sdot(out,p,ps,x,rows,cols,0),"zeros");for(size_t r=0;r<rows;++r)require(out[r]==0,"zero result");
    x[0]=NAN;require(ds41f_mxfp4_sdot(out,p,ps,x,rows,cols,0)==EDOM,"NaN input");
    s[0]=255;require(ds41f_mxfp4_pack_sdot(p,ps,w,s,rows,cols)==EDOM,"NaN scale");
    free(w);free(p);free(s);free(ps);free(x);free(out);free(ref);free(fp);
}
int main(int argc,char **argv){(void)argv;size_t rows[]={4,8,32,128},cols[]={32,64,128,5120};for(size_t r=0;r<4;++r)for(size_t c=0;c<4;++c)check(rows[r],cols[c],0);
    puts("FP4_SDOT PASS cases=16 lossless_packing integer_oracle zeros canaries nonfinite");if(argc>1){check(2304,5120,1);check(5120,2304,1);}return 0;}
