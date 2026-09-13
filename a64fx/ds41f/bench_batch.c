#define _POSIX_C_SOURCE 200809L
#include "ds41f_batch.h"
#include "ds41f_kernels.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int quantization(void)
{
    float x[32],ref[32],fast[32];uint16_t bits[32];uint32_t rng=1;
    for(int mode=0;mode<3;++mode)for(int trial=0;trial<4096;++trial){
        for(int j=0;j<32;++j){
            if(mode==0){unsigned code=(unsigned)(trial*32+j)%0x7f80;
                x[j]=ds41f_bf16_to_f32((uint16_t)(code|((trial&1)?0x8000:0)));}
            else if(mode==1){rng=rng*1664525u+1013904223u;union {uint32_t u;float f;}v={rng&0xfeffffffu};x[j]=v.f;}
            else {int code=(trial+j)%126;float a=ds41f_fp8_e4m3_to_f32((uint8_t)code),b=ds41f_fp8_e4m3_to_f32((uint8_t)(code+1));
                x[j]=ldexpf((a+b)*.5f,(trial%201)-100);if(j&1)x[j]=-x[j];}
        }
        if(mode==2)x[31]=ldexpf(448.f,(trial%201)-100);
        int expected=ds41f_act_quant_ref(ref,x,32),actual=ds41f_batch_quantize32(bits,x);
        if(expected!=actual)return 1;
        if(ds41f_act_quant(fast,x,32)!=expected||(!expected&&memcmp(ref,fast,sizeof ref)))return 1;
        if(!expected)for(int j=0;j<32;++j)if(bits[j]!=ds41f_f32_to_bf16(ref[j])){
            fprintf(stderr,"QUANT_PANEL FAIL mode=%d trial=%d index=%d x=%g got=%04x ref=%04x\n",mode,trial,j,x[j],bits[j],ds41f_f32_to_bf16(ref[j]));return 1;}
    }
    x[0]=NAN;if(!ds41f_batch_quantize32(bits,x))return 1;
    x[0]=INFINITY;if(!ds41f_batch_quantize32(bits,x))return 1;
    puts("QUANT_PANEL PASS groups=12288 bf16_codes random_f32 scaled_fp8_midpoints nonfinite");fflush(stdout);return 0;
}
static int run(size_t rows,size_t cols,size_t batch,int fp4)
{
    size_t bytes=rows*cols/(fp4?2:1),scales=fp4?rows*cols/32:rows*cols/1024;
    uint8_t *w=malloc(bytes),*s=malloc(scales);float *x=malloc(batch*cols*4),*y=malloc(batch*rows*4),*q=malloc(cols*4),*ref=malloc(rows*4);
    if(!w||!s||!x||!y||!q||!ref)return 1;
    #pragma omp parallel for
    for(size_t i=0;i<bytes;++i)w[i]=fp4?(uint8_t)(i*37+11):(uint8_t)(0x28+i%32);
    for(size_t i=0;i<scales;++i)s[i]=125+(i%3);
    #pragma omp parallel for
    for(size_t i=0;i<batch*cols;++i)x[i]=(float)((int)(i%23)-11)/16;
    if(ds41f_quant_batch(y,w,s,x,rows,cols,batch,fp4))return 1;
    for(size_t token=0;token<batch;token+=(batch<20?1:batch/2)){
        if(ds41f_act_quant(q,x+token*cols,cols))return 1;
        if(fp4)ds41f_mxfp4_matvec_ref(ref,w,s,q,rows,cols);else ds41f_fp8_matvec_ref(ref,w,s,q,rows,cols,32);
        for(size_t r=0;r<rows;++r){float expected=ds41f_bf16_to_f32(ds41f_f32_to_bf16(ref[r]));
            if(!isfinite(y[token*rows+r])||fabsf(y[token*rows+r]-expected)>1e-4f){fprintf(stderr,"BATCH FAIL fp4=%d token=%zu row=%zu got=%g ref=%g\n",fp4,token,r,y[token*rows+r],expected);return 1;}}
    }
    double best=1e9;for(int it=0;it<3;++it){double t=now();if(ds41f_quant_batch(y,w,s,x,rows,cols,batch,fp4))return 1;t=now()-t;if(t<best)best=t;}
    printf("BATCH PASS format=%s M=%zu K=%zu N=%zu ms=%.3f GFLOPs=%.3f nominal_fraction=%.4f\n",fp4?"MXFP4":"FP8",rows,cols,batch,best*1e3,2.*rows*cols*batch/best/1e9,2.*rows*cols*batch/best/1e9/6144);fflush(stdout);
    double times[4];if(ds41f_quant_batch_profile(y,w,s,x,rows,cols,batch,fp4,times))return 1;
    printf("BATCH_PHASES B_pack_ms=%.3f A_pack_ms=%.3f GEMM_ms=%.3f round_ms=%.3f\n",times[0]*1e3,times[1]*1e3,times[2]*1e3,times[3]*1e3);fflush(stdout);
    free(w);free(s);free(x);free(y);free(q);free(ref);return 0;
}
int main(void)
{if(quantization())return 1;for(int format=0;format<2;++format){if(run(64,64,13,format)||run(2304,5120,768,format)||run(32768,1280,768,format))return 1;}return 0;}
