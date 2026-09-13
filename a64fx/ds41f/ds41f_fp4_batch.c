#include "ds41f_sve.h"
#include "ds41f_fp4_sdot.h"
#include "ds41f_team.h"
#include <errno.h>
#include <stdint.h>
#include <string.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static float scale_e8(uint8_t scale)
{uint32_t bits=scale==255?0x7fc00000u:scale?((uint32_t)scale<<23):0x00400000u;float value;memcpy(&value,&bits,4);return value;}
static const float grid_float[16]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};
static const int8_t grid_int[64]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static void fp4_batch_2(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const float *x,size_t xs,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32();svfloat32_t table=svld1_f32(pg,grid_float);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    for(size_t b=0;b<cols/32;++b){
        svuint32_t raw=svld1ub_u32(pg,w+r*(cols/2)+b*16);
        svfloat32_t lo=svtbl_f32(table,svand_n_u32_x(pg,raw,15)),hi=svtbl_f32(table,svlsr_n_u32_x(pg,raw,4));
        float scale=scale_e8(sc[r*(cols/32)+b]);
        {svfloat32_t a=svld1_f32(pg,x+0*xs+b*32),z=svld1_f32(pg,x+0*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a0=svmla_n_f32_x(pg,a0,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+1*xs+b*32),z=svld1_f32(pg,x+1*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a1=svmla_n_f32_x(pg,a1,prod,scale);}
    }
    out[0*stride+r]=svaddv_f32(pg,a0);
    out[1*stride+r]=svaddv_f32(pg,a1);
}
static void fp4_batch_3(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const float *x,size_t xs,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32();svfloat32_t table=svld1_f32(pg,grid_float);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    svfloat32_t a2=svdup_f32(0);
    for(size_t b=0;b<cols/32;++b){
        svuint32_t raw=svld1ub_u32(pg,w+r*(cols/2)+b*16);
        svfloat32_t lo=svtbl_f32(table,svand_n_u32_x(pg,raw,15)),hi=svtbl_f32(table,svlsr_n_u32_x(pg,raw,4));
        float scale=scale_e8(sc[r*(cols/32)+b]);
        {svfloat32_t a=svld1_f32(pg,x+0*xs+b*32),z=svld1_f32(pg,x+0*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a0=svmla_n_f32_x(pg,a0,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+1*xs+b*32),z=svld1_f32(pg,x+1*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a1=svmla_n_f32_x(pg,a1,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+2*xs+b*32),z=svld1_f32(pg,x+2*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a2=svmla_n_f32_x(pg,a2,prod,scale);}
    }
    out[0*stride+r]=svaddv_f32(pg,a0);
    out[1*stride+r]=svaddv_f32(pg,a1);
    out[2*stride+r]=svaddv_f32(pg,a2);
}
static void fp4_batch_4(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const float *x,size_t xs,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32();svfloat32_t table=svld1_f32(pg,grid_float);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    svfloat32_t a2=svdup_f32(0);
    svfloat32_t a3=svdup_f32(0);
    for(size_t b=0;b<cols/32;++b){
        svuint32_t raw=svld1ub_u32(pg,w+r*(cols/2)+b*16);
        svfloat32_t lo=svtbl_f32(table,svand_n_u32_x(pg,raw,15)),hi=svtbl_f32(table,svlsr_n_u32_x(pg,raw,4));
        float scale=scale_e8(sc[r*(cols/32)+b]);
        {svfloat32_t a=svld1_f32(pg,x+0*xs+b*32),z=svld1_f32(pg,x+0*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a0=svmla_n_f32_x(pg,a0,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+1*xs+b*32),z=svld1_f32(pg,x+1*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a1=svmla_n_f32_x(pg,a1,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+2*xs+b*32),z=svld1_f32(pg,x+2*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a2=svmla_n_f32_x(pg,a2,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+3*xs+b*32),z=svld1_f32(pg,x+3*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a3=svmla_n_f32_x(pg,a3,prod,scale);}
    }
    out[0*stride+r]=svaddv_f32(pg,a0);
    out[1*stride+r]=svaddv_f32(pg,a1);
    out[2*stride+r]=svaddv_f32(pg,a2);
    out[3*stride+r]=svaddv_f32(pg,a3);
}
static void fp4_batch_5(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const float *x,size_t xs,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32();svfloat32_t table=svld1_f32(pg,grid_float);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    svfloat32_t a2=svdup_f32(0);
    svfloat32_t a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0);
    for(size_t b=0;b<cols/32;++b){
        svuint32_t raw=svld1ub_u32(pg,w+r*(cols/2)+b*16);
        svfloat32_t lo=svtbl_f32(table,svand_n_u32_x(pg,raw,15)),hi=svtbl_f32(table,svlsr_n_u32_x(pg,raw,4));
        float scale=scale_e8(sc[r*(cols/32)+b]);
        {svfloat32_t a=svld1_f32(pg,x+0*xs+b*32),z=svld1_f32(pg,x+0*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a0=svmla_n_f32_x(pg,a0,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+1*xs+b*32),z=svld1_f32(pg,x+1*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a1=svmla_n_f32_x(pg,a1,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+2*xs+b*32),z=svld1_f32(pg,x+2*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a2=svmla_n_f32_x(pg,a2,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+3*xs+b*32),z=svld1_f32(pg,x+3*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a3=svmla_n_f32_x(pg,a3,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+4*xs+b*32),z=svld1_f32(pg,x+4*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a4=svmla_n_f32_x(pg,a4,prod,scale);}
    }
    out[0*stride+r]=svaddv_f32(pg,a0);
    out[1*stride+r]=svaddv_f32(pg,a1);
    out[2*stride+r]=svaddv_f32(pg,a2);
    out[3*stride+r]=svaddv_f32(pg,a3);
    out[4*stride+r]=svaddv_f32(pg,a4);
}
static void fp4_batch_6(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const float *x,size_t xs,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32();svfloat32_t table=svld1_f32(pg,grid_float);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    svfloat32_t a2=svdup_f32(0);
    svfloat32_t a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0);
    svfloat32_t a5=svdup_f32(0);
    for(size_t b=0;b<cols/32;++b){
        svuint32_t raw=svld1ub_u32(pg,w+r*(cols/2)+b*16);
        svfloat32_t lo=svtbl_f32(table,svand_n_u32_x(pg,raw,15)),hi=svtbl_f32(table,svlsr_n_u32_x(pg,raw,4));
        float scale=scale_e8(sc[r*(cols/32)+b]);
        {svfloat32_t a=svld1_f32(pg,x+0*xs+b*32),z=svld1_f32(pg,x+0*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a0=svmla_n_f32_x(pg,a0,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+1*xs+b*32),z=svld1_f32(pg,x+1*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a1=svmla_n_f32_x(pg,a1,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+2*xs+b*32),z=svld1_f32(pg,x+2*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a2=svmla_n_f32_x(pg,a2,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+3*xs+b*32),z=svld1_f32(pg,x+3*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a3=svmla_n_f32_x(pg,a3,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+4*xs+b*32),z=svld1_f32(pg,x+4*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a4=svmla_n_f32_x(pg,a4,prod,scale);}
        {svfloat32_t a=svld1_f32(pg,x+5*xs+b*32),z=svld1_f32(pg,x+5*xs+b*32+16);
        svfloat32_t prod=svmul_f32_x(pg,lo,svuzp1_f32(a,z));
        prod=svmla_f32_x(pg,prod,hi,svuzp2_f32(a,z));a5=svmla_n_f32_x(pg,a5,prod,scale);}
    }
    out[0*stride+r]=svaddv_f32(pg,a0);
    out[1*stride+r]=svaddv_f32(pg,a1);
    out[2*stride+r]=svaddv_f32(pg,a2);
    out[3*stride+r]=svaddv_f32(pg,a3);
    out[4*stride+r]=svaddv_f32(pg,a4);
    out[5*stride+r]=svaddv_f32(pg,a5);
}
static void sdot_batch_2(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const ds41f_int8_input *input,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32(),p4=svptrue_pat_b32(SV_VL4);size_t blocks=cols/32;
    svuint32_t expand=svlsr_n_u32_x(pg,svindex_u32(0,1),2);svint8_t table=svld1_s8(svptrue_b8(),grid_int);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    for(size_t b=0;b<blocks;++b){
        svuint8_t raw=svld1_u8(svptrue_b8(),w+r*(cols/2)+b*64);
        svint8_t lo=svtbl_s8(table,svand_n_u8_x(svptrue_b8(),raw,15)),hi=svtbl_s8(table,svlsr_n_u8_x(svptrue_b8(),raw,4));
        svuint32_t codes=svld1ub_u32(p4,sc+r*blocks+b*4),bits=svlsl_n_u32_x(p4,codes,23);
        bits=svsel_u32(svcmpeq_n_u32(p4,codes,0),svdup_u32(0x00400000u),bits);
        svfloat32_t scale=svtbl_f32(svreinterpret_f32_u32(bits),expand);
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[0].scale[b]*.5f);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[1].scale[b]*.5f);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,dot),scaled);}
    }
    for(size_t j=0;j<4;++j){svuint32_t idx=svindex_u32(0,1);
        svbool_t mask=svand_b_z(pg,svcmpge_n_u32(pg,idx,j*4),svcmplt_n_u32(pg,idx,j*4+4));
        out[0*stride+r+j]=svaddv_f32(mask,a0);
        out[1*stride+r+j]=svaddv_f32(mask,a1);
    }
}
static void sdot_batch_3(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const ds41f_int8_input *input,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32(),p4=svptrue_pat_b32(SV_VL4);size_t blocks=cols/32;
    svuint32_t expand=svlsr_n_u32_x(pg,svindex_u32(0,1),2);svint8_t table=svld1_s8(svptrue_b8(),grid_int);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    svfloat32_t a2=svdup_f32(0);
    for(size_t b=0;b<blocks;++b){
        svuint8_t raw=svld1_u8(svptrue_b8(),w+r*(cols/2)+b*64);
        svint8_t lo=svtbl_s8(table,svand_n_u8_x(svptrue_b8(),raw,15)),hi=svtbl_s8(table,svlsr_n_u8_x(svptrue_b8(),raw,4));
        svuint32_t codes=svld1ub_u32(p4,sc+r*blocks+b*4),bits=svlsl_n_u32_x(p4,codes,23);
        bits=svsel_u32(svcmpeq_n_u32(p4,codes,0),svdup_u32(0x00400000u),bits);
        svfloat32_t scale=svtbl_f32(svreinterpret_f32_u32(bits),expand);
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[0].scale[b]*.5f);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[1].scale[b]*.5f);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[2].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[2].scale[b]*.5f);
        a2=svmla_f32_x(pg,a2,svcvt_f32_s32_x(pg,dot),scaled);}
    }
    for(size_t j=0;j<4;++j){svuint32_t idx=svindex_u32(0,1);
        svbool_t mask=svand_b_z(pg,svcmpge_n_u32(pg,idx,j*4),svcmplt_n_u32(pg,idx,j*4+4));
        out[0*stride+r+j]=svaddv_f32(mask,a0);
        out[1*stride+r+j]=svaddv_f32(mask,a1);
        out[2*stride+r+j]=svaddv_f32(mask,a2);
    }
}
static void sdot_batch_4(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const ds41f_int8_input *input,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32(),p4=svptrue_pat_b32(SV_VL4);size_t blocks=cols/32;
    svuint32_t expand=svlsr_n_u32_x(pg,svindex_u32(0,1),2);svint8_t table=svld1_s8(svptrue_b8(),grid_int);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    svfloat32_t a2=svdup_f32(0);
    svfloat32_t a3=svdup_f32(0);
    for(size_t b=0;b<blocks;++b){
        svuint8_t raw=svld1_u8(svptrue_b8(),w+r*(cols/2)+b*64);
        svint8_t lo=svtbl_s8(table,svand_n_u8_x(svptrue_b8(),raw,15)),hi=svtbl_s8(table,svlsr_n_u8_x(svptrue_b8(),raw,4));
        svuint32_t codes=svld1ub_u32(p4,sc+r*blocks+b*4),bits=svlsl_n_u32_x(p4,codes,23);
        bits=svsel_u32(svcmpeq_n_u32(p4,codes,0),svdup_u32(0x00400000u),bits);
        svfloat32_t scale=svtbl_f32(svreinterpret_f32_u32(bits),expand);
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[0].scale[b]*.5f);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[1].scale[b]*.5f);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[2].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[2].scale[b]*.5f);
        a2=svmla_f32_x(pg,a2,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[3].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[3].scale[b]*.5f);
        a3=svmla_f32_x(pg,a3,svcvt_f32_s32_x(pg,dot),scaled);}
    }
    for(size_t j=0;j<4;++j){svuint32_t idx=svindex_u32(0,1);
        svbool_t mask=svand_b_z(pg,svcmpge_n_u32(pg,idx,j*4),svcmplt_n_u32(pg,idx,j*4+4));
        out[0*stride+r+j]=svaddv_f32(mask,a0);
        out[1*stride+r+j]=svaddv_f32(mask,a1);
        out[2*stride+r+j]=svaddv_f32(mask,a2);
        out[3*stride+r+j]=svaddv_f32(mask,a3);
    }
}
static void sdot_batch_5(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const ds41f_int8_input *input,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32(),p4=svptrue_pat_b32(SV_VL4);size_t blocks=cols/32;
    svuint32_t expand=svlsr_n_u32_x(pg,svindex_u32(0,1),2);svint8_t table=svld1_s8(svptrue_b8(),grid_int);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    svfloat32_t a2=svdup_f32(0);
    svfloat32_t a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0);
    for(size_t b=0;b<blocks;++b){
        svuint8_t raw=svld1_u8(svptrue_b8(),w+r*(cols/2)+b*64);
        svint8_t lo=svtbl_s8(table,svand_n_u8_x(svptrue_b8(),raw,15)),hi=svtbl_s8(table,svlsr_n_u8_x(svptrue_b8(),raw,4));
        svuint32_t codes=svld1ub_u32(p4,sc+r*blocks+b*4),bits=svlsl_n_u32_x(p4,codes,23);
        bits=svsel_u32(svcmpeq_n_u32(p4,codes,0),svdup_u32(0x00400000u),bits);
        svfloat32_t scale=svtbl_f32(svreinterpret_f32_u32(bits),expand);
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[0].scale[b]*.5f);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[1].scale[b]*.5f);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[2].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[2].scale[b]*.5f);
        a2=svmla_f32_x(pg,a2,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[3].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[3].scale[b]*.5f);
        a3=svmla_f32_x(pg,a3,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[4].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[4].scale[b]*.5f);
        a4=svmla_f32_x(pg,a4,svcvt_f32_s32_x(pg,dot),scaled);}
    }
    for(size_t j=0;j<4;++j){svuint32_t idx=svindex_u32(0,1);
        svbool_t mask=svand_b_z(pg,svcmpge_n_u32(pg,idx,j*4),svcmplt_n_u32(pg,idx,j*4+4));
        out[0*stride+r+j]=svaddv_f32(mask,a0);
        out[1*stride+r+j]=svaddv_f32(mask,a1);
        out[2*stride+r+j]=svaddv_f32(mask,a2);
        out[3*stride+r+j]=svaddv_f32(mask,a3);
        out[4*stride+r+j]=svaddv_f32(mask,a4);
    }
}
static void sdot_batch_6(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,const ds41f_int8_input *input,size_t cols,size_t r)
{
    svbool_t pg=svptrue_b32(),p4=svptrue_pat_b32(SV_VL4);size_t blocks=cols/32;
    svuint32_t expand=svlsr_n_u32_x(pg,svindex_u32(0,1),2);svint8_t table=svld1_s8(svptrue_b8(),grid_int);
    svfloat32_t a0=svdup_f32(0);
    svfloat32_t a1=svdup_f32(0);
    svfloat32_t a2=svdup_f32(0);
    svfloat32_t a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0);
    svfloat32_t a5=svdup_f32(0);
    for(size_t b=0;b<blocks;++b){
        svuint8_t raw=svld1_u8(svptrue_b8(),w+r*(cols/2)+b*64);
        svint8_t lo=svtbl_s8(table,svand_n_u8_x(svptrue_b8(),raw,15)),hi=svtbl_s8(table,svlsr_n_u8_x(svptrue_b8(),raw,4));
        svuint32_t codes=svld1ub_u32(p4,sc+r*blocks+b*4),bits=svlsl_n_u32_x(p4,codes,23);
        bits=svsel_u32(svcmpeq_n_u32(p4,codes,0),svdup_u32(0x00400000u),bits);
        svfloat32_t scale=svtbl_f32(svreinterpret_f32_u32(bits),expand);
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[0].scale[b]*.5f);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[1].scale[b]*.5f);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[2].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[2].scale[b]*.5f);
        a2=svmla_f32_x(pg,a2,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[3].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[3].scale[b]*.5f);
        a3=svmla_f32_x(pg,a3,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[4].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[4].scale[b]*.5f);
        a4=svmla_f32_x(pg,a4,svcvt_f32_s32_x(pg,dot),scaled);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[5].data+b*64);
        svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
        svfloat32_t scaled=svmul_n_f32_x(pg,scale,input[5].scale[b]*.5f);
        a5=svmla_f32_x(pg,a5,svcvt_f32_s32_x(pg,dot),scaled);}
    }
    for(size_t j=0;j<4;++j){svuint32_t idx=svindex_u32(0,1);
        svbool_t mask=svand_b_z(pg,svcmpge_n_u32(pg,idx,j*4),svcmplt_n_u32(pg,idx,j*4+4));
        out[0*stride+r+j]=svaddv_f32(mask,a0);
        out[1*stride+r+j]=svaddv_f32(mask,a1);
        out[2*stride+r+j]=svaddv_f32(mask,a2);
        out[3*stride+r+j]=svaddv_f32(mask,a3);
        out[4*stride+r+j]=svaddv_f32(mask,a4);
        out[5*stride+r+j]=svaddv_f32(mask,a5);
    }
}
typedef struct {float *out;size_t stride,cols,xs,batch;const uint8_t *w,*sc;const float *x;const ds41f_int8_input *input;} fp4_batch_job;
static void float_work(void *context,size_t first,size_t last)
{
    fp4_batch_job *j=context;
    void (*kernel)(float *,size_t,const uint8_t *,const uint8_t *,const float *,size_t,size_t,size_t)=
        j->batch==2?fp4_batch_2:j->batch==3?fp4_batch_3:j->batch==4?fp4_batch_4:j->batch==5?fp4_batch_5:fp4_batch_6;
    for(size_t r=first;r<last;++r){
        if(j->batch==6){/* Two three-token tiles keep the input working set smaller. */
            fp4_batch_3(j->out,j->stride,j->w,j->sc,j->x,j->xs,j->cols,r);
            fp4_batch_3(j->out+3*j->stride,j->stride,j->w,j->sc,j->x+3*j->xs,j->xs,j->cols,r);
        }else kernel(j->out,j->stride,j->w,j->sc,j->x,j->xs,j->cols,r);
    }
}
static void sdot_work(void *context,size_t first,size_t last)
{
    fp4_batch_job *j=context;
    void (*kernel)(float *,size_t,const uint8_t *,const uint8_t *,const ds41f_int8_input *,size_t,size_t)=
        j->batch==2?sdot_batch_2:j->batch==3?sdot_batch_3:j->batch==4?sdot_batch_4:j->batch==5?sdot_batch_5:sdot_batch_6;
    for(size_t r=first;r<last;++r){
        if(j->batch==6){
            sdot_batch_3(j->out,j->stride,j->w,j->sc,j->input,j->cols,r*4);
            sdot_batch_3(j->out+3*j->stride,j->stride,j->w,j->sc,j->input+3,j->cols,r*4);
        }else kernel(j->out,j->stride,j->w,j->sc,j->input,j->cols,r*4);
    }
}
#endif
int ds41f_mxfp4_matmul(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,
                       const float *x,size_t xs,size_t rows,size_t cols,size_t batch)
{
    if(!out||!w||!sc||!x||!rows||!cols||cols%32||!batch||batch>6||stride<rows||xs<cols||
       stride>SIZE_MAX/sizeof(float)/batch||xs>SIZE_MAX/sizeof(float)/batch||rows>SIZE_MAX/cols)return EINVAL;
#if defined(__ARM_FEATURE_SVE)
    if(svcntw()==16&&batch>1){fp4_batch_job j={out,stride,cols,xs,batch,w,sc,x,NULL};
        return ds41f_team_for(rows,float_work,&j);}
#endif
    for(size_t i=0;i<batch;++i){int rc=ds41f_mxfp4_matvec(out+i*stride,w,sc,x+i*xs,rows,cols);if(rc)return rc;}
    return 0;
}
int ds41f_mxfp4_sdot_matmul(float *out,size_t stride,const uint8_t *w,const uint8_t *sc,
                           const ds41f_int8_input *input,size_t rows,size_t cols,size_t batch)
{
    if(!out||!w||!sc||!input||!rows||rows%4||!cols||cols%32||!batch||batch>6||stride<rows||
       stride>SIZE_MAX/sizeof(float)/batch||rows>SIZE_MAX/cols)return EINVAL;
    for(size_t i=0;i<batch;++i)if(!input[i].data||!input[i].scale||input[i].elements!=cols||input[i].block!=32)return EINVAL;
#if defined(__ARM_FEATURE_SVE)
    if(svcntw()==16&&batch>1){fp4_batch_job j={out,stride,cols,0,batch,w,sc,NULL,input};
        return ds41f_team_for(rows/4,sdot_work,&j);}
#endif
    for(size_t i=0;i<batch;++i){int rc=ds41f_mxfp4_sdot_prepared(out+i*stride,w,sc,input+i,rows,cols,0);if(rc)return rc;}
    return 0;
}
