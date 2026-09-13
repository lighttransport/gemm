#ifndef DS41F_CACHE_SVE_H
#define DS41F_CACHE_SVE_H
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static svfloat32_t cache_bf16(svfloat32_t value)
{
    svbool_t pg=svptrue_b32();svuint32_t bits=svreinterpret_u32_f32(value);
    svuint32_t odd=svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),1);
    svuint32_t rounded=svadd_u32_x(pg,bits,svadd_n_u32_x(pg,odd,0x7fff));
    rounded=svsel_u32(svcmpgt_n_u32(pg,svand_n_u32_x(pg,bits,0x7fffffffu),0x7f800000u),
        svorr_n_u32_x(pg,bits,0x00400000u),rounded);
    return svreinterpret_f32_u32(svand_n_u32_x(pg,rounded,0xffff0000u));
}
static svuint32_t cache_encode(svfloat32_t x)
{
    svbool_t pg=svptrue_b32();svfloat32_t a=svabs_f32_x(pg,x),best_value=svdup_f32(0);
    svuint32_t best=svdup_u32(0);
    /* Keep the reference's rounded distance comparisons, including extreme
     * finite values where subtraction can tie for several grid points. */
    for(unsigned i=1;i<8;++i){
        svfloat32_t distance=svabs_f32_x(pg,svsub_n_f32_x(pg,a,fp4_values[i]));
        svfloat32_t old=svabs_f32_x(pg,svsub_f32_x(pg,a,best_value));
        svbool_t choose=svcmplt_f32(pg,distance,old);
        if(!(i&1))choose=svorr_b_z(pg,choose,svcmpeq_f32(pg,distance,old));
        best=svsel_u32(choose,svdup_u32(i),best);best_value=svsel_f32(choose,svdup_f32(fp4_values[i]),best_value);
    }
    svuint32_t sign=svand_n_u32_x(pg,svlsr_n_u32_x(pg,svreinterpret_u32_f32(x),28),8);
    return svorr_u32_x(pg,best,sign);
}
static int cache_pack_sve(uint8_t *out,const float *x,size_t dim,size_t group,int e4)
{
    svbool_t pg=svptrue_b32();memset(out,0,dim/2+dim/group);
    for(size_t base=0;base<dim;base+=group){
        svfloat32_t a=cache_bf16(svld1_f32(pg,x+base)),b=svdup_f32(0);
        if(group==32)b=cache_bf16(svld1_f32(pg,x+base+16));
        svfloat32_t maximum=svmax_f32_x(pg,svabs_f32_x(pg,a),svabs_f32_x(pg,b));
        if(svptest_any(pg,svcmpge_n_u32(pg,svand_n_u32_x(pg,svreinterpret_u32_f32(a),0x7fffffffu),0x7f800000u))||
           svptest_any(pg,svcmpge_n_u32(pg,svand_n_u32_x(pg,svreinterpret_u32_f32(b),0x7fffffffu),0x7f800000u)))return EDOM;
        float peak=fmaxf(e4?6.f/512:6.f*0x1p-126f,svmaxv_f32(pg,maximum)),scale;uint8_t code;
        if(e4){code=ds41f_f32_to_fp8(peak/6);scale=ds41f_fp8_e4m3_to_f32(code);}
        else{int exponent=(int)ceilf(log2f(peak/6));if(exponent>127||exponent< -126)return ERANGE;
            code=(uint8_t)(exponent+127);scale=ldexpf(1,exponent);}
        out[dim/2+base/group]=code;
        svuint32_t lo=cache_encode(svdiv_n_f32_x(pg,a,scale)),hi=cache_encode(svdiv_n_f32_x(pg,b,scale));
        svuint32_t even=svuzp1_u32(lo,hi),odd=svuzp2_u32(lo,hi);
        svst1b_u32(svwhilelt_b32((uint64_t)0,(uint64_t)(group/2)),out+base/2,
            svorr_u32_x(pg,even,svlsl_n_u32_x(pg,odd,4)));
    }
    return 0;
}
static int cache_unpack_sve(float *out,const uint8_t *row,size_t dim,size_t group,int e4)
{
    static const float values[16]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
    svbool_t pg=svptrue_b32();svfloat32_t table=svld1_f32(pg,values);
    for(size_t base=0;base<dim;base+=group){uint8_t sc=row[dim/2+base/group];
        float scale=e4?ds41f_fp8_e4m3_to_f32(sc):ds41f_e8m0_to_f32(sc);
        svuint32_t bytes=svld1ub_u32(svwhilelt_b32((uint64_t)0,(uint64_t)(group/2)),row+base/2);
        svfloat32_t lo=svtbl_f32(table,svand_n_u32_x(pg,bytes,15)),hi=svtbl_f32(table,svlsr_n_u32_x(pg,bytes,4));
        svst1_f32(pg,out+base,svmul_n_f32_x(pg,svzip1_f32(lo,hi),scale));
        if(group==32)svst1_f32(pg,out+base+16,svmul_n_f32_x(pg,svzip2_f32(lo,hi),scale));
    }
    return 0;
}
#endif
#endif
