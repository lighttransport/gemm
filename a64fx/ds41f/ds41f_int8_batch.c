#include "ds41f_int8.h"
#include "ds41f_team.h"
#include <errno.h>
#include <stdint.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
/* Four rows by 2 tokens. Each output retains the GEMV lane/reduction
 * order; weights are loaded once for all tokens in each K block. */
static void int8_batch_2(float *out,size_t stride,const ds41f_int8 *q,
                          const ds41f_int8_input *input,size_t group_rows,size_t r)
{
    svbool_t pg=svptrue_b32(),lo=svwhilelt_b32((uint64_t)0,(uint64_t)8);
    svbool_t hi=sveor_b_z(pg,lo,pg);size_t blocks=q->cols/32,g=r/group_rows;
    svfloat32_t a0=svdup_f32(0),b0=a0;
    svfloat32_t a1=svdup_f32(0),b1=a1;
    for(size_t block=0;block<blocks;++block){
        const int8_t *w=q->weight+r*q->cols+block*128;
        svint8_t w01=svld1_s8(svptrue_b8(),w),w23=svld1_s8(svptrue_b8(),w+64);
        const float *sc=q->scale+((r/4)*blocks+block)*4;
        svuint32_t indices=svlsr_n_u32_x(pg,svindex_u32(0,1),3);
        svfloat32_t scales=svld1_f32(svptrue_pat_b32(SV_VL4),sc);
        svfloat32_t base01=svtbl_f32(scales,indices),base23=svtbl_f32(scales,svadd_n_u32_x(pg,indices,2));
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[0].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,d01),s01);
        b0=svmla_f32_x(pg,b0,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[1].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,d01),s01);
        b1=svmla_f32_x(pg,b1,svcvt_f32_s32_x(pg,d23),s23);}
    }
    out[0*stride+r]=svaddv_f32(lo,a0);
    if(r+1<q->rows)out[0*stride+r+1]=svaddv_f32(hi,a0);
    if(r+2<q->rows)out[0*stride+r+2]=svaddv_f32(lo,b0);
    if(r+3<q->rows)out[0*stride+r+3]=svaddv_f32(hi,b0);
    out[1*stride+r]=svaddv_f32(lo,a1);
    if(r+1<q->rows)out[1*stride+r+1]=svaddv_f32(hi,a1);
    if(r+2<q->rows)out[1*stride+r+2]=svaddv_f32(lo,b1);
    if(r+3<q->rows)out[1*stride+r+3]=svaddv_f32(hi,b1);
}
/* Four rows by 3 tokens. Each output retains the GEMV lane/reduction
 * order; weights are loaded once for all tokens in each K block. */
static void int8_batch_3(float *out,size_t stride,const ds41f_int8 *q,
                          const ds41f_int8_input *input,size_t group_rows,size_t r)
{
    svbool_t pg=svptrue_b32(),lo=svwhilelt_b32((uint64_t)0,(uint64_t)8);
    svbool_t hi=sveor_b_z(pg,lo,pg);size_t blocks=q->cols/32,g=r/group_rows;
    svfloat32_t a0=svdup_f32(0),b0=a0;
    svfloat32_t a1=svdup_f32(0),b1=a1;
    svfloat32_t a2=svdup_f32(0),b2=a2;
    for(size_t block=0;block<blocks;++block){
        const int8_t *w=q->weight+r*q->cols+block*128;
        svint8_t w01=svld1_s8(svptrue_b8(),w),w23=svld1_s8(svptrue_b8(),w+64);
        const float *sc=q->scale+((r/4)*blocks+block)*4;
        svuint32_t indices=svlsr_n_u32_x(pg,svindex_u32(0,1),3);
        svfloat32_t scales=svld1_f32(svptrue_pat_b32(SV_VL4),sc);
        svfloat32_t base01=svtbl_f32(scales,indices),base23=svtbl_f32(scales,svadd_n_u32_x(pg,indices,2));
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[0].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,d01),s01);
        b0=svmla_f32_x(pg,b0,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[1].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,d01),s01);
        b1=svmla_f32_x(pg,b1,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[2].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[2].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a2=svmla_f32_x(pg,a2,svcvt_f32_s32_x(pg,d01),s01);
        b2=svmla_f32_x(pg,b2,svcvt_f32_s32_x(pg,d23),s23);}
    }
    out[0*stride+r]=svaddv_f32(lo,a0);
    if(r+1<q->rows)out[0*stride+r+1]=svaddv_f32(hi,a0);
    if(r+2<q->rows)out[0*stride+r+2]=svaddv_f32(lo,b0);
    if(r+3<q->rows)out[0*stride+r+3]=svaddv_f32(hi,b0);
    out[1*stride+r]=svaddv_f32(lo,a1);
    if(r+1<q->rows)out[1*stride+r+1]=svaddv_f32(hi,a1);
    if(r+2<q->rows)out[1*stride+r+2]=svaddv_f32(lo,b1);
    if(r+3<q->rows)out[1*stride+r+3]=svaddv_f32(hi,b1);
    out[2*stride+r]=svaddv_f32(lo,a2);
    if(r+1<q->rows)out[2*stride+r+1]=svaddv_f32(hi,a2);
    if(r+2<q->rows)out[2*stride+r+2]=svaddv_f32(lo,b2);
    if(r+3<q->rows)out[2*stride+r+3]=svaddv_f32(hi,b2);
}
/* Four rows by 4 tokens. Each output retains the GEMV lane/reduction
 * order; weights are loaded once for all tokens in each K block. */
static void int8_batch_4(float *out,size_t stride,const ds41f_int8 *q,
                          const ds41f_int8_input *input,size_t group_rows,size_t r)
{
    svbool_t pg=svptrue_b32(),lo=svwhilelt_b32((uint64_t)0,(uint64_t)8);
    svbool_t hi=sveor_b_z(pg,lo,pg);size_t blocks=q->cols/32,g=r/group_rows;
    svfloat32_t a0=svdup_f32(0),b0=a0;
    svfloat32_t a1=svdup_f32(0),b1=a1;
    svfloat32_t a2=svdup_f32(0),b2=a2;
    svfloat32_t a3=svdup_f32(0),b3=a3;
    for(size_t block=0;block<blocks;++block){
        const int8_t *w=q->weight+r*q->cols+block*128;
        svint8_t w01=svld1_s8(svptrue_b8(),w),w23=svld1_s8(svptrue_b8(),w+64);
        const float *sc=q->scale+((r/4)*blocks+block)*4;
        svuint32_t indices=svlsr_n_u32_x(pg,svindex_u32(0,1),3);
        svfloat32_t scales=svld1_f32(svptrue_pat_b32(SV_VL4),sc);
        svfloat32_t base01=svtbl_f32(scales,indices),base23=svtbl_f32(scales,svadd_n_u32_x(pg,indices,2));
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[0].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,d01),s01);
        b0=svmla_f32_x(pg,b0,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[1].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,d01),s01);
        b1=svmla_f32_x(pg,b1,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[2].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[2].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a2=svmla_f32_x(pg,a2,svcvt_f32_s32_x(pg,d01),s01);
        b2=svmla_f32_x(pg,b2,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[3].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[3].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a3=svmla_f32_x(pg,a3,svcvt_f32_s32_x(pg,d01),s01);
        b3=svmla_f32_x(pg,b3,svcvt_f32_s32_x(pg,d23),s23);}
    }
    out[0*stride+r]=svaddv_f32(lo,a0);
    if(r+1<q->rows)out[0*stride+r+1]=svaddv_f32(hi,a0);
    if(r+2<q->rows)out[0*stride+r+2]=svaddv_f32(lo,b0);
    if(r+3<q->rows)out[0*stride+r+3]=svaddv_f32(hi,b0);
    out[1*stride+r]=svaddv_f32(lo,a1);
    if(r+1<q->rows)out[1*stride+r+1]=svaddv_f32(hi,a1);
    if(r+2<q->rows)out[1*stride+r+2]=svaddv_f32(lo,b1);
    if(r+3<q->rows)out[1*stride+r+3]=svaddv_f32(hi,b1);
    out[2*stride+r]=svaddv_f32(lo,a2);
    if(r+1<q->rows)out[2*stride+r+1]=svaddv_f32(hi,a2);
    if(r+2<q->rows)out[2*stride+r+2]=svaddv_f32(lo,b2);
    if(r+3<q->rows)out[2*stride+r+3]=svaddv_f32(hi,b2);
    out[3*stride+r]=svaddv_f32(lo,a3);
    if(r+1<q->rows)out[3*stride+r+1]=svaddv_f32(hi,a3);
    if(r+2<q->rows)out[3*stride+r+2]=svaddv_f32(lo,b3);
    if(r+3<q->rows)out[3*stride+r+3]=svaddv_f32(hi,b3);
}
/* Four rows by 5 tokens. Each output retains the GEMV lane/reduction
 * order; weights are loaded once for all tokens in each K block. */
static void int8_batch_5(float *out,size_t stride,const ds41f_int8 *q,
                          const ds41f_int8_input *input,size_t group_rows,size_t r)
{
    svbool_t pg=svptrue_b32(),lo=svwhilelt_b32((uint64_t)0,(uint64_t)8);
    svbool_t hi=sveor_b_z(pg,lo,pg);size_t blocks=q->cols/32,g=r/group_rows;
    svfloat32_t a0=svdup_f32(0),b0=a0;
    svfloat32_t a1=svdup_f32(0),b1=a1;
    svfloat32_t a2=svdup_f32(0),b2=a2;
    svfloat32_t a3=svdup_f32(0),b3=a3;
    svfloat32_t a4=svdup_f32(0),b4=a4;
    for(size_t block=0;block<blocks;++block){
        const int8_t *w=q->weight+r*q->cols+block*128;
        svint8_t w01=svld1_s8(svptrue_b8(),w),w23=svld1_s8(svptrue_b8(),w+64);
        const float *sc=q->scale+((r/4)*blocks+block)*4;
        svuint32_t indices=svlsr_n_u32_x(pg,svindex_u32(0,1),3);
        svfloat32_t scales=svld1_f32(svptrue_pat_b32(SV_VL4),sc);
        svfloat32_t base01=svtbl_f32(scales,indices),base23=svtbl_f32(scales,svadd_n_u32_x(pg,indices,2));
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[0].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,d01),s01);
        b0=svmla_f32_x(pg,b0,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[1].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,d01),s01);
        b1=svmla_f32_x(pg,b1,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[2].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[2].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a2=svmla_f32_x(pg,a2,svcvt_f32_s32_x(pg,d01),s01);
        b2=svmla_f32_x(pg,b2,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[3].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[3].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a3=svmla_f32_x(pg,a3,svcvt_f32_s32_x(pg,d01),s01);
        b3=svmla_f32_x(pg,b3,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[4].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[4].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a4=svmla_f32_x(pg,a4,svcvt_f32_s32_x(pg,d01),s01);
        b4=svmla_f32_x(pg,b4,svcvt_f32_s32_x(pg,d23),s23);}
    }
    out[0*stride+r]=svaddv_f32(lo,a0);
    if(r+1<q->rows)out[0*stride+r+1]=svaddv_f32(hi,a0);
    if(r+2<q->rows)out[0*stride+r+2]=svaddv_f32(lo,b0);
    if(r+3<q->rows)out[0*stride+r+3]=svaddv_f32(hi,b0);
    out[1*stride+r]=svaddv_f32(lo,a1);
    if(r+1<q->rows)out[1*stride+r+1]=svaddv_f32(hi,a1);
    if(r+2<q->rows)out[1*stride+r+2]=svaddv_f32(lo,b1);
    if(r+3<q->rows)out[1*stride+r+3]=svaddv_f32(hi,b1);
    out[2*stride+r]=svaddv_f32(lo,a2);
    if(r+1<q->rows)out[2*stride+r+1]=svaddv_f32(hi,a2);
    if(r+2<q->rows)out[2*stride+r+2]=svaddv_f32(lo,b2);
    if(r+3<q->rows)out[2*stride+r+3]=svaddv_f32(hi,b2);
    out[3*stride+r]=svaddv_f32(lo,a3);
    if(r+1<q->rows)out[3*stride+r+1]=svaddv_f32(hi,a3);
    if(r+2<q->rows)out[3*stride+r+2]=svaddv_f32(lo,b3);
    if(r+3<q->rows)out[3*stride+r+3]=svaddv_f32(hi,b3);
    out[4*stride+r]=svaddv_f32(lo,a4);
    if(r+1<q->rows)out[4*stride+r+1]=svaddv_f32(hi,a4);
    if(r+2<q->rows)out[4*stride+r+2]=svaddv_f32(lo,b4);
    if(r+3<q->rows)out[4*stride+r+3]=svaddv_f32(hi,b4);
}
/* Four rows by 6 tokens. Each output retains the GEMV lane/reduction
 * order; weights are loaded once for all tokens in each K block. */
static void int8_batch_6(float *out,size_t stride,const ds41f_int8 *q,
                          const ds41f_int8_input *input,size_t group_rows,size_t r)
{
    svbool_t pg=svptrue_b32(),lo=svwhilelt_b32((uint64_t)0,(uint64_t)8);
    svbool_t hi=sveor_b_z(pg,lo,pg);size_t blocks=q->cols/32,g=r/group_rows;
    svfloat32_t a0=svdup_f32(0),b0=a0;
    svfloat32_t a1=svdup_f32(0),b1=a1;
    svfloat32_t a2=svdup_f32(0),b2=a2;
    svfloat32_t a3=svdup_f32(0),b3=a3;
    svfloat32_t a4=svdup_f32(0),b4=a4;
    svfloat32_t a5=svdup_f32(0),b5=a5;
    for(size_t block=0;block<blocks;++block){
        const int8_t *w=q->weight+r*q->cols+block*128;
        svint8_t w01=svld1_s8(svptrue_b8(),w),w23=svld1_s8(svptrue_b8(),w+64);
        const float *sc=q->scale+((r/4)*blocks+block)*4;
        svuint32_t indices=svlsr_n_u32_x(pg,svindex_u32(0,1),3);
        svfloat32_t scales=svld1_f32(svptrue_pat_b32(SV_VL4),sc);
        svfloat32_t base01=svtbl_f32(scales,indices),base23=svtbl_f32(scales,svadd_n_u32_x(pg,indices,2));
        {svint8_t x=svld1_s8(svptrue_b8(),input[0].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[0].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a0=svmla_f32_x(pg,a0,svcvt_f32_s32_x(pg,d01),s01);
        b0=svmla_f32_x(pg,b0,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[1].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[1].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a1=svmla_f32_x(pg,a1,svcvt_f32_s32_x(pg,d01),s01);
        b1=svmla_f32_x(pg,b1,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[2].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[2].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a2=svmla_f32_x(pg,a2,svcvt_f32_s32_x(pg,d01),s01);
        b2=svmla_f32_x(pg,b2,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[3].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[3].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a3=svmla_f32_x(pg,a3,svcvt_f32_s32_x(pg,d01),s01);
        b3=svmla_f32_x(pg,b3,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[4].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[4].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a4=svmla_f32_x(pg,a4,svcvt_f32_s32_x(pg,d01),s01);
        b4=svmla_f32_x(pg,b4,svcvt_f32_s32_x(pg,d23),s23);}
        {svint8_t x=svld1_s8(svptrue_b8(),input[5].data+g*q->cols*2+block*64);
        svint32_t d01=svdot_s32(svdup_s32(0),w01,x),d23=svdot_s32(svdup_s32(0),w23,x);
        float scale=input[5].scale[g*blocks+block];
        svfloat32_t s01=svmul_n_f32_x(pg,base01,scale);
        svfloat32_t s23=svmul_n_f32_x(pg,base23,scale);
        a5=svmla_f32_x(pg,a5,svcvt_f32_s32_x(pg,d01),s01);
        b5=svmla_f32_x(pg,b5,svcvt_f32_s32_x(pg,d23),s23);}
    }
    out[0*stride+r]=svaddv_f32(lo,a0);
    if(r+1<q->rows)out[0*stride+r+1]=svaddv_f32(hi,a0);
    if(r+2<q->rows)out[0*stride+r+2]=svaddv_f32(lo,b0);
    if(r+3<q->rows)out[0*stride+r+3]=svaddv_f32(hi,b0);
    out[1*stride+r]=svaddv_f32(lo,a1);
    if(r+1<q->rows)out[1*stride+r+1]=svaddv_f32(hi,a1);
    if(r+2<q->rows)out[1*stride+r+2]=svaddv_f32(lo,b1);
    if(r+3<q->rows)out[1*stride+r+3]=svaddv_f32(hi,b1);
    out[2*stride+r]=svaddv_f32(lo,a2);
    if(r+1<q->rows)out[2*stride+r+1]=svaddv_f32(hi,a2);
    if(r+2<q->rows)out[2*stride+r+2]=svaddv_f32(lo,b2);
    if(r+3<q->rows)out[2*stride+r+3]=svaddv_f32(hi,b2);
    out[3*stride+r]=svaddv_f32(lo,a3);
    if(r+1<q->rows)out[3*stride+r+1]=svaddv_f32(hi,a3);
    if(r+2<q->rows)out[3*stride+r+2]=svaddv_f32(lo,b3);
    if(r+3<q->rows)out[3*stride+r+3]=svaddv_f32(hi,b3);
    out[4*stride+r]=svaddv_f32(lo,a4);
    if(r+1<q->rows)out[4*stride+r+1]=svaddv_f32(hi,a4);
    if(r+2<q->rows)out[4*stride+r+2]=svaddv_f32(lo,b4);
    if(r+3<q->rows)out[4*stride+r+3]=svaddv_f32(hi,b4);
    out[5*stride+r]=svaddv_f32(lo,a5);
    if(r+1<q->rows)out[5*stride+r+1]=svaddv_f32(hi,a5);
    if(r+2<q->rows)out[5*stride+r+2]=svaddv_f32(lo,b5);
    if(r+3<q->rows)out[5*stride+r+3]=svaddv_f32(hi,b5);
}
typedef struct {
    float *out;size_t stride,group;
    const ds41f_int8 *q;const ds41f_int8_input *input;
    void (*kernel)(float *,size_t,const ds41f_int8 *,const ds41f_int8_input *,size_t,size_t);
} batch_job;
static void batch_work(void *context,size_t first,size_t last)
{batch_job *j=context;for(size_t i=first;i<last;++i)j->kernel(j->out,j->stride,j->q,j->input,j->group,i*4);}

#endif
int ds41f_int8_matmul_prepared(float *out,size_t output_stride,const ds41f_int8 *q,
                              const ds41f_int8_input *input,size_t batch,
                              size_t group_rows,int reference)
{
    if(!out||!q||!input||!q->weight||!q->scale||!q->rows||!q->cols||!batch||batch>6||
       output_stride<q->rows||batch>SIZE_MAX/sizeof(float)/output_stride||!group_rows||q->rows%group_rows||
       (group_rows!=q->rows&&group_rows%4)||q->block<32||q->block>256||q->block%32||
       q->cols%q->block||(q->rows/group_rows)>SIZE_MAX/q->cols)return EINVAL;
    size_t elements=(q->rows/group_rows)*q->cols;
    for(size_t t=0;t<batch;++t)if(input[t].elements!=elements||input[t].block!=q->block||
        !input[t].data||!input[t].scale)return EINVAL;
#if defined(__ARM_FEATURE_SVE)
    if(!reference&&q->block==32&&svcntw()==16&&batch>1){
        void (*kernel)(float *,size_t,const ds41f_int8 *,const ds41f_int8_input *,size_t,size_t)=
            batch==2?int8_batch_2:batch==3?int8_batch_3:batch==4?int8_batch_4:batch==5?int8_batch_5:int8_batch_6;
        if(ds41f_team_active()){
            batch_job job={out,output_stride,group_rows,q,input,kernel};
            return ds41f_team_for((q->rows+3)/4,batch_work,&job);
        }
        #pragma omp parallel for schedule(static)
        for(size_t r=0;r<q->rows;r+=4)kernel(out,output_stride,q,input,group_rows,r);
        return 0;
    }
#endif
    for(size_t t=0;t<batch;++t){int rc=ds41f_int8_matvec_prepared(out+t*output_stride,q,input+t,group_rows,reference);if(rc)return rc;}
    return 0;
}
int ds41f_int8_matmul(float *out,size_t output_stride,const ds41f_int8 *q,const float *x,
                      size_t input_stride,size_t batch,size_t group_rows,int reference)
{
    if(!out||!x||!q||!q->rows||!q->cols||!group_rows||q->rows%group_rows||!batch||batch>6||
       q->rows/group_rows>SIZE_MAX/q->cols||input_stride<(q->rows/group_rows)*q->cols||
       !input_stride||batch>SIZE_MAX/sizeof(float)/input_stride)return EINVAL;
    ds41f_int8_input input[6]={{0}};int rc=0;
    for(size_t t=0;t<batch;++t){rc=ds41f_int8_prepare_input(input+t,x+t*input_stride,
        (q->rows/group_rows)*q->cols,q->block);if(rc)break;}
    if(!rc)rc=ds41f_int8_matmul_prepared(out,output_stride,q,input,batch,group_rows,reference);
    for(size_t t=0;t<batch;++t)ds41f_int8_input_free(input+t);
    return rc;
}
