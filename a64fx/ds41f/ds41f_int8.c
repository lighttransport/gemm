#define _GNU_SOURCE
#include "ds41f_int8.h"
#include "ds41f_alloc.h"
#include "ds41f_kernels.h"
#include "ds41f_profile.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

static float e8m0(uint8_t s)
{
    uint32_t bits=s==0?0x00400000u:(uint32_t)s<<23;
    float value;memcpy(&value,&bits,4);return value;
}
void ds41f_int8_free(ds41f_int8 *q)
{
    if(!q)return;
    size_t n=((q->rows+3)/4*4)*q->cols;
    ds41f_free_resident(q->weight,n,1);
    if(q->scale)ds41f_free_resident(q->scale,n/q->block*sizeof(float),1);
    memset(q,0,sizeof *q);
}
int ds41f_int8_from_fp8(ds41f_int8 *out,const uint8_t *weight,
                      const uint8_t *scale,size_t rows,size_t cols,size_t block)
{
    if(!out||!weight||!scale||!rows||!cols||block<32||block>256||block%32||cols%block||
       rows>SIZE_MAX-3||((rows+3)/4)>SIZE_MAX/4/cols)return EINVAL;
    memset(out,0,sizeof *out);
    size_t padded=(rows+3)/4*4,blocks=cols/block,n=padded*cols;
    if(n>SIZE_MAX-n/block*sizeof(float))return EINVAL;
    out->rows=rows;out->cols=cols;out->block=block;out->bytes=n+n/block*sizeof(float);
    int rc=ds41f_alloc_resident((void **)&out->weight,n,1);
    if(rc)return rc;
    rc=ds41f_alloc_resident((void **)&out->scale,n/block*sizeof(float),1);
    if(rc){ds41f_int8_free(out);return rc;}
    float lut[256];for(int i=0;i<256;++i)lut[i]=ds41f_fp8_e4m3_to_f32((uint8_t)i);
    int invalid=0;
    #pragma omp parallel for schedule(static) reduction(|:invalid)
    for(size_t r=0;r<padded;r+=4){
        for(size_t b=0;b<blocks;++b)for(size_t row=0;row<4;++row){
            float values[256],maximum=0;
            for(size_t c=0;c<block;++c){
                float value=0;
                if(r+row<rows){size_t k=b*block+c;
                    uint8_t sc=scale[((r+row)/32)*(cols/32)+k/32];
                    value=lut[weight[(r+row)*cols+k]]*e8m0(sc);
                    if(sc==255||!isfinite(value)){invalid=1;value=0;}}
                values[c]=value;maximum=fmaxf(maximum,fabsf(value));
            }
            float step=maximum?fmaxf(maximum/127.f,0x1p-149f):1.f;
            out->scale[((r/4)*blocks+b)*4+row]=step;
            for(size_t c=0;c<block;++c){
                size_t offset=r*cols+b*4*block+(c/32)*128+row*32+c%32;
                out->weight[offset]=(int8_t)fmaxf(-127.f,fminf(127.f,nearbyintf(values[c]/step)));
            }
        }
    }
    if(invalid){ds41f_int8_free(out);return EDOM;}
    return 0;
}

static int quantize_input_block(int8_t *input,float *scale,const float *values,size_t block)
{
    float maximum=0;
    #if defined(__ARM_FEATURE_SVE)
    if(svcntw()==16){
        svbool_t pg=svptrue_b32();svfloat32_t maxv=svdup_f32(0);
        for(size_t c=0;c<block;c+=16){svfloat32_t v=svabs_f32_x(pg,svld1_f32(pg,values+c));
            if(svptest_any(pg,svcmpge_n_f32(pg,v,INFINITY))||
               svptest_any(pg,svcmpuo_f32(pg,v,v))){return EDOM;}
            maxv=svmax_f32_x(pg,maxv,v);}
        maximum=svmaxv_f32(pg,maxv);
        float step=maximum?fmaxf(maximum/127.f,0x1p-149f):1.f;*scale=step;
        for(size_t c=0;c<block;c+=16){
            svfloat32_t v=svld1_f32(pg,values+c);
            v=step<FLT_MIN?svdiv_n_f32_x(pg,v,step):svmul_n_f32_x(pg,v,1.f/step);
            v=svmax_n_f32_x(pg,svmin_n_f32_x(pg,svrintn_f32_x(pg,v),127.f),-127.f);
            svint32_t quant=svcvt_s32_f32_x(pg,v);
            size_t k=(c/32)*64+c%32;
            svst1b_s32(pg,input+k,quant);svst1b_s32(pg,input+k+32,quant);
        }
        return 0;
    }
    #endif
    for(size_t c=0;c<block;++c){
        if(!isfinite(values[c])){return EDOM;}
        maximum=fmaxf(maximum,fabsf(values[c]));}
    float step=maximum?fmaxf(maximum/127.f,0x1p-149f):1.f;*scale=step;
    for(size_t c=0;c<block;++c){
        int8_t v=(int8_t)fmaxf(-127.f,fminf(127.f,nearbyintf(values[c]/step)));
        size_t k=(c/32)*64+c%32;
        input[k]=v;input[k+32]=v;
    }
    return 0;
}

int ds41f_int8_matvec(float *out,const ds41f_int8 *q,const float *x,
                     size_t group_rows,int reference)
{
    if(!out||!q||!x||!q->weight||!q->scale||!q->rows||!q->cols||
       q->block<32||q->block>256||q->block%32||q->cols%q->block||!group_rows||q->rows%group_rows||
       (group_rows!=q->rows&&group_rows%4))return EINVAL;
    double pt=P_BEGIN();
    size_t blocks=q->cols/q->block,groups=q->rows/group_rows;
    if(groups>SIZE_MAX/q->cols/2)return EINVAL;
    int8_t *input=malloc(groups*q->cols*2);
    float *scales=malloc(groups*blocks*sizeof(float));
    if(!input||!scales){free(input);free(scales);return ENOMEM;}
    int invalid=0;
    if(groups*q->cols>=8192){
        #pragma omp parallel for schedule(static) reduction(|:invalid)
        for(size_t b=0;b<groups*blocks;++b)
            invalid|=quantize_input_block(input+b*q->block*2,scales+b,x+b*q->block,q->block);
    }else for(size_t b=0;b<groups*blocks;++b)
        invalid|=quantize_input_block(input+b*q->block*2,scales+b,x+b*q->block,q->block);
    if(invalid){free(input);free(scales);return invalid;}
    P_END(INT8_INPUT_QUANT,pt);
    #pragma omp parallel for schedule(static)
    for(size_t r=0;r<q->rows;r+=4){
        size_t g=r/group_rows;
        #if defined(__ARM_FEATURE_SVE)
        if(!reference&&svcntw()==16){
            svbool_t pg=svptrue_b32(),lo=svwhilelt_b32((uint64_t)0,(uint64_t)8);
            svbool_t hi=sveor_b_z(pg,lo,pg);
            svfloat32_t sum01=svdup_f32(0),sum23=sum01;
            for(size_t b=0;b<blocks;++b){
                svint32_t dot01=svdup_s32(0),dot23=dot01;
                for(size_t c=0;c<q->block;c+=32){
                    const int8_t *w=q->weight+r*q->cols+b*4*q->block+c*4;
                    svint8_t v=svld1_s8(svptrue_b8(),input+g*q->cols*2+b*q->block*2+c*2);
                    dot01=svdot_s32(dot01,svld1_s8(svptrue_b8(),w),v);
                    dot23=svdot_s32(dot23,svld1_s8(svptrue_b8(),w+64),v);
                }
                const float *ws=q->scale+((r/4)*blocks+b)*4;
                float xs=scales[g*blocks+b];
                svfloat32_t sc01=svsel_f32(lo,svdup_f32(ws[0]*xs),svdup_f32(ws[1]*xs));
                svfloat32_t sc23=svsel_f32(lo,svdup_f32(ws[2]*xs),svdup_f32(ws[3]*xs));
                sum01=svmla_f32_x(pg,sum01,svcvt_f32_s32_x(pg,dot01),sc01);
                sum23=svmla_f32_x(pg,sum23,svcvt_f32_s32_x(pg,dot23),sc23);
            }
            out[r]=svaddv_f32(lo,sum01);
            if(r+1<q->rows)out[r+1]=svaddv_f32(hi,sum01);
            if(r+2<q->rows)out[r+2]=svaddv_f32(lo,sum23);
            if(r+3<q->rows)out[r+3]=svaddv_f32(hi,sum23);
            continue;
        }
        #else
        (void)reference;
        #endif
        for(size_t row=0;row<4&&r+row<q->rows;++row){double sum=0;
            for(size_t b=0;b<blocks;++b){int dot=0;
                for(size_t c=0;c<q->block;++c){
                    size_t wi=r*q->cols+b*4*q->block+(c/32)*128+row*32+c%32;
                    size_t xi=g*q->cols*2+b*q->block*2+(c/32)*64+c%32;
                    dot+=(int)q->weight[wi]*input[xi];}
                sum+=(double)dot*q->scale[((r/4)*blocks+b)*4+row]*scales[g*blocks+b];
            }
            out[r+row]=(float)sum;
        }
    }
    free(input);free(scales);return 0;
}
