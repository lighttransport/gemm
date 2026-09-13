#ifndef DS41F_QUANT_SVE_H
#define DS41F_QUANT_SVE_H
#include "ds41f_kernels.h"
#include <errno.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
/* Group-32 FP8 activation quantization, returned as exact BF16 values. */
static inline int ds41f_quantize32_bf16(uint16_t out[32],const float *x)
{
    #if defined(__ARM_FEATURE_SVE)
    if(svcntw()!=16){float values[32];int rc=ds41f_act_quant_ref(values,x,32);if(rc)return rc;
        for(int j=0;j<32;++j)out[j]=ds41f_f32_to_bf16(values[j]);
        return 0;}
    svbool_t pg=svptrue_b32();float maximum=1e-4f;
    float rounded[32];
    for(size_t j=0;j<32;j+=16){
        svuint32_t bits=svreinterpret_u32_f32(svld1(pg,x+j));
        if(svptest_any(pg,svcmpge_n_u32(pg,svand_n_u32_x(pg,bits,0x7fffffffu),0x7f800000u)))return EDOM;
        svuint32_t odd=svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),1);
        bits=svand_n_u32_x(pg,svadd_u32_x(pg,bits,svadd_n_u32_x(pg,odd,0x7fff)),0xffff0000u);
        svfloat32_t value=svreinterpret_f32_u32(bits);
        svuint32_t mag=svand_n_u32_x(pg,bits,0x7fffffffu);
        if(svptest_any(pg,svcmpge_n_u32(pg,mag,0x7f800000u)))return EDOM;
        float block=svmaxv_f32(pg,svreinterpret_f32_u32(mag));
        if(block>maximum)maximum=block;
        svst1(pg,rounded+j,value);
    }
    union {float f;uint32_t u;} scale={maximum/448.f};
    scale.u=(scale.u+0x007fffffu)&0x7f800000u;
    for(size_t j=0;j<32;j+=16){
        svfloat32_t value=svmul_n_f32_x(pg,svld1(pg,rounded+j),1.f/scale.f);
        svuint32_t bits=svreinterpret_u32_f32(value),sign=svand_n_u32_x(pg,bits,0x80000000u);
        svfloat32_t mag=svabs_f32_x(pg,value);
        bits=svreinterpret_u32_f32(mag);
        svuint32_t odd=svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,20),1);
        svuint32_t normal=svand_n_u32_x(pg,svadd_u32_x(pg,bits,svadd_n_u32_x(pg,odd,0x7ffff)),0xfff00000u);
        svfloat32_t sub=svmul_n_f32_x(pg,svrintn_f32_x(pg,svmul_n_f32_x(pg,mag,512.f)),1.f/512);
        svfloat32_t quant=svsel_f32(svcmplt_n_f32(pg,mag,0x1p-6f),sub,svreinterpret_f32_u32(normal));
        quant=svmin_n_f32_x(pg,quant,448.f);
        bits=svorr_u32_x(pg,svreinterpret_u32_f32(quant),sign);
        bits=svreinterpret_u32_f32(svmul_n_f32_x(pg,svreinterpret_f32_u32(bits),scale.f));
        svst1h_u32(pg,out+j,svlsr_n_u32_x(pg,bits,16));
    }
    return 0;
    #else
    float quant[32];int rc=ds41f_act_quant_ref(quant,x,32);if(rc)return rc;
    for(size_t j=0;j<32;++j)out[j]=ds41f_f32_to_bf16(quant[j]);
    return 0;
    #endif
}
#endif
