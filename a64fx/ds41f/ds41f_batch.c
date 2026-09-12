#define _POSIX_C_SOURCE 200809L
#include "ds41f_batch.h"
#include "ds41f_kernels.h"
#include "ds41f_quant_sve.h"
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static inline void store_interleaved(uint16_t *out,svuint32_t low,svuint32_t high)
{svbool_t pg=svptrue_b32();svst1h_u32(pg,out,svzip1_u32(low,high));svst1h_u32(pg,out+16,svzip2_u32(low,high));}
#endif
extern void sgemm_bf16_2x12(int64_t k,const void *a,const void *b,float *c,int64_t ldc);
static const float f4[16]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
static inline uint16_t bf_bits(float x)
{union {float f;uint32_t u;}v={x};if((v.u&0x7fffffffu)>0x7f800000u)return (uint16_t)((v.u>>16)|64);return (uint16_t)((v.u+0x7fff+((v.u>>16)&1))>>16);}
static inline void round_tile(float *out,size_t stride,size_t tokens)
{
    for(size_t t=0;t<tokens;++t){
        #if defined(__ARM_FEATURE_SVE)
        svbool_t pg=svptrue_b32();
        for(size_t r=0;r<32;r+=16){
            svuint32_t bits=svreinterpret_u32_f32(svld1(pg,out+t*stride+r));
            svuint32_t odd=svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),1);
            svuint32_t rounded=svand_n_u32_x(pg,svadd_u32_x(pg,bits,svadd_n_u32_x(pg,odd,0x7fff)),0xffff0000u);
            svbool_t nan=svcmpgt_n_u32(pg,svand_n_u32_x(pg,bits,0x7fffffffu),0x7f800000u);
            rounded=svsel_u32(nan,svand_n_u32_x(pg,svorr_n_u32_x(pg,bits,0x00400000u),0xffff0000u),rounded);
            svst1(pg,out+t*stride+r,svreinterpret_f32_u32(rounded));
        }
        #else
        for(size_t r=0;r<32;++r){union {uint32_t u;float f;}v={(uint32_t)bf_bits(out[t*stride+r])<<16};out[t*stride+r]=v.f;}
        #endif
    }
}
static inline float scale8(uint8_t x)
{union {uint32_t u;float f;}v={x==0?0x00400000u:x==255?0x7fc00000u:(uint32_t)x<<23};return v.f;}
static inline float value8(uint8_t x)
{unsigned mag=x&127;if(mag<8)return (x&128?-(float)mag:(float)mag)/512;
    union {uint32_t u;float f;}v={mag==127?0x7fc00000u:(mag<<20)+0x3c000000u};v.u|=(uint32_t)(x&128)<<24;return v.f;}

/* Independent batch packing implementation. The benchmark checks it against
 * ds41f_act_quant, including exact FP8 midpoint and BF16 boundary cases. */
static int quantize_panel(uint16_t out[32],const float *x)
{return ds41f_quantize32_bf16(out,x);}

int ds41f_batch_quantize32(uint16_t out[32],const float x[32])
{if(!out||!x)return EINVAL;return quantize_panel(out,x);}

int ds41f_quant_batch_profile(float *out,const uint8_t *w,const uint8_t *s,const float *x,
                      size_t rows,size_t cols,size_t batch,int fp4,double times[4])
{
    if(!out||!w||!s||!x||!rows||rows%32||!cols||cols%32||!batch||batch>4096||rows>131072||cols>32768)return EINVAL;
    size_t tiles=(batch+11)/12,kblocks=cols/32,rtiles=rows/32;
    uint16_t *a=NULL,*b=NULL;
    if(posix_memalign((void **)&a,256,rows*cols*sizeof *a))return ENOMEM;
    if(posix_memalign((void **)&b,256,tiles*cols*12*sizeof *b)){free(a);return ENOMEM;}
    int rc=0;double start=omp_get_wtime();
    /* Transpose L1-sized blocks instead of repeatedly touching a full strided panel. */
    #pragma omp parallel for schedule(static) reduction(|:rc)
    for(size_t block=0;block<tiles*kblocks;++block){size_t tile=block/kblocks,k=(block%kblocks)*32;uint16_t quant[32];
        for(size_t t=0;t<12;++t){size_t token=tile*12+t;
            if(token<batch){int error=quantize_panel(quant,x+token*cols+k);if(error){rc=error;continue;}}
            for(size_t j=0;j<32;++j)b[tile*cols*12+(k+j)*12+t]=token<batch?quant[j]:0;}
    }
    if(rc){free(a);free(b);return rc;}
    if(times)times[0]=omp_get_wtime()-start;
    start=omp_get_wtime();
    size_t codes=fp4?16:256;
    uint32_t *lookup=malloc(256*codes*sizeof *lookup);
    if(!lookup){free(a);free(b);return ENOMEM;}
    #pragma omp parallel for schedule(static)
    for(size_t index=0;index<256*codes;++index){unsigned code=index%codes;
        lookup[index]=bf_bits((fp4?f4[code]:value8((uint8_t)code))*scale8((uint8_t)(index/codes)));}
    #pragma omp parallel for schedule(static)
    for(size_t rt=0;rt<rtiles;++rt){size_t row=rt*32;uint16_t *panel=a+rt*cols*32;
        #if defined(__ARM_FEATURE_SVE)
        svbool_t pg=svptrue_b32();
        svuint32_t offsets=svmul_n_u32_x(pg,svindex_u32(0,1),(uint32_t)(fp4?cols/2:cols));
        svuint32_t scale_offsets=svmul_n_u32_x(pg,svindex_u32(0,1),(uint32_t)kblocks);
        for(size_t k=0;k<cols;k+=32){
            if(fp4){
                svuint32_t sl=svlsl_n_u32_x(pg,svld1ub_gather_u32offset_u32(pg,s+row*kblocks+k/32,scale_offsets),4);
                svuint32_t sh=svlsl_n_u32_x(pg,svld1ub_gather_u32offset_u32(pg,s+(row+16)*kblocks+k/32,scale_offsets),4);
                for(size_t j=0;j<32;j+=2){
                    svuint32_t rl=svld1ub_gather_u32offset_u32(pg,w+row*(cols/2)+(k+j)/2,offsets);
                    svuint32_t rh=svld1ub_gather_u32offset_u32(pg,w+(row+16)*(cols/2)+(k+j)/2,offsets);
                    svuint32_t lo=svld1_gather_u32index_u32(pg,lookup,svorr_u32_x(pg,sl,svand_n_u32_x(pg,rl,15)));
                    svuint32_t hi=svld1_gather_u32index_u32(pg,lookup,svorr_u32_x(pg,sh,svand_n_u32_x(pg,rh,15)));
                    store_interleaved(panel+(k+j)*32,lo,hi);
                    lo=svld1_gather_u32index_u32(pg,lookup,svorr_u32_x(pg,sl,svlsr_n_u32_x(pg,rl,4)));
                    hi=svld1_gather_u32index_u32(pg,lookup,svorr_u32_x(pg,sh,svlsr_n_u32_x(pg,rh,4)));
                    store_interleaved(panel+(k+j+1)*32,lo,hi);
                }
            }else{const uint32_t *table=lookup+(size_t)s[rt*kblocks+k/32]*256;
                for(size_t j=0;j<32;++j){
                    svuint32_t lo=svld1ub_gather_u32offset_u32(pg,w+row*cols+k+j,offsets);
                    svuint32_t hi=svld1ub_gather_u32offset_u32(pg,w+(row+16)*cols+k+j,offsets);
                    lo=svld1_gather_u32index_u32(pg,table,lo);hi=svld1_gather_u32index_u32(pg,table,hi);
                    store_interleaved(panel+(k+j)*32,lo,hi);
                }
            }
        }
        #else
        for(size_t k=0;k<cols;k+=32){float common=fp4?0:scale8(s[rt*kblocks+k/32]);
            for(size_t r=0;r<32;++r){float sc=fp4?scale8(s[(row+r)*kblocks+k/32]):common;
                for(size_t j=0;j<32;++j){float value;
                    if(fp4){unsigned code=(w[(row+r)*(cols/2)+(k+j)/2]>>((j&1)*4))&15;value=f4[code]*sc;}
                    else value=value8(w[(row+r)*cols+k+j])*sc;
                    panel[(k+j)*32+2*(r%16)+r/16]=bf_bits(value);}}}
        #endif
    }
    free(lookup);
    if(times)times[1]=omp_get_wtime()-start;
    start=omp_get_wtime();
    /* Flatten both tile dimensions: 72 row tiles alone leave 48 threads imbalanced. */
    #pragma omp parallel for schedule(static)
    for(size_t work=0;work<rtiles*tiles;++work){size_t rt=work/tiles,tile=work%tiles,row=rt*32,first=tile*12;
        if(first+12<=batch){sgemm_bf16_2x12((int64_t)cols,a+rt*cols*32,b+tile*cols*12,out+first*rows+row,(int64_t)rows);
            round_tile(out+first*rows+row,rows,12);}
        else{float tail[12*32];sgemm_bf16_2x12((int64_t)cols,a+rt*cols*32,b+tile*cols*12,tail,32);
            round_tile(tail,32,batch-first);
            for(size_t t=first;t<batch;++t)for(size_t r=0;r<32;++r)out[t*rows+row+r]=tail[(t-first)*32+r];}
    }
    if(times)times[2]=omp_get_wtime()-start;
    if(times)times[3]=0; /* Rounding fused with each thread's hot output tile. */
    free(a);free(b);return 0;
}
int ds41f_quant_batch(float *out,const uint8_t *w,const uint8_t *s,const float *x,
                      size_t rows,size_t cols,size_t batch,int fp4)
{return ds41f_quant_batch_profile(out,w,s,x,rows,cols,batch,fp4,NULL);}
