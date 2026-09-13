#include "ds41f_team.h"
#include "ds41f_fp4_sdot.h"
#include <errno.h>
#include <math.h>
#include <string.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
static const int8_t grid[64]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static float e8(uint8_t s)
{uint32_t u=s==0?0x00400000u:(uint32_t)s<<23;float f;memcpy(&f,&u,4);return f;}
int ds41f_mxfp4_pack_sdot(uint8_t *packed,uint8_t *scales,const uint8_t *w,
                   const uint8_t *s,size_t rows,size_t cols)
{
    if(!packed||!scales||!w||!s||!rows||rows%4||!cols||cols%32||rows>SIZE_MAX/cols)return EINVAL;
    size_t blocks=cols/32;int invalid=0;
    #pragma omp parallel for schedule(static) reduction(|:invalid)
    for(size_t r=0;r<rows;r+=4)for(size_t b=0;b<blocks;++b)for(size_t j=0;j<4;++j){
        memcpy(packed+r*(cols/2)+b*64+j*16,w+(r+j)*(cols/2)+b*16,16);
        uint8_t sc=s[(r+j)*blocks+b];scales[(r*blocks)+b*4+j]=sc;invalid|=sc==255;
    }
    return invalid?EDOM:0;
}
typedef struct {
    float * out;
    const uint8_t * w;
    const uint8_t * s;
    const ds41f_int8_input * input;
    size_t rows;
    size_t cols;
    size_t blocks;
    int reference;
} mxfp4_sdot_prepared_team_job;
static void mxfp4_sdot_prepared_team_work(void *context,size_t first,size_t last)
{
    mxfp4_sdot_prepared_team_job *job=context;
    float * out=job->out;
    const uint8_t * w=job->w;
    const uint8_t * s=job->s;
    const ds41f_int8_input * input=job->input;
    size_t rows=job->rows;
    size_t cols=job->cols;
    size_t blocks=job->blocks;
    int reference=job->reference;
    (void)out;
    (void)w;
    (void)s;
    (void)input;
    (void)rows;
    (void)cols;
    (void)blocks;
    (void)reference;
    for(size_t task=first;task<last;++task){size_t r=task*(4);
        #if defined(__ARM_FEATURE_SVE)
        if(!reference&&svcntw()==16){
            svbool_t pg=svptrue_b32(),p4=svwhilelt_b32((uint64_t)0,(uint64_t)4);
            svuint32_t expand=svlsr_n_u32_x(pg,svindex_u32(0,1),2);
            svint8_t lut=svld1_s8(svptrue_b8(),grid);svfloat32_t sum=svdup_f32(0);
            for(size_t b=0;b<blocks;++b){
                svuint8_t raw=svld1_u8(svptrue_b8(),w+r*(cols/2)+b*64);
                svint8_t lo=svtbl_s8(lut,svand_n_u8_x(svptrue_b8(),raw,15));
                svint8_t hi=svtbl_s8(lut,svlsr_n_u8_x(svptrue_b8(),raw,4));
                svint8_t x=svld1_s8(svptrue_b8(),input->data+b*64);
                svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));
                dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
                svuint32_t sc=svld1ub_u32(p4,s+r*blocks+b*4);
                svuint32_t bits=svlsl_n_u32_x(p4,sc,23);
                bits=svsel_u32(svcmpeq_n_u32(p4,sc,0),svdup_u32(0x00400000u),bits);
                svfloat32_t scale=svmul_n_f32_x(pg,svtbl_f32(svreinterpret_f32_u32(bits),expand),input->scale[b]*.5f);
                sum=svmla_f32_x(pg,sum,svcvt_f32_s32_x(pg,dot),scale);
            }
            for(size_t j=0;j<4;++j){svuint32_t idx=svindex_u32(0,1);
                svbool_t mask=svand_b_z(pg,svcmpge_n_u32(pg,idx,j*4),svcmplt_n_u32(pg,idx,j*4+4));
                out[r+j]=svaddv_f32(mask,sum);}
            continue;
        }
        #else
        (void)reference;
        #endif
        for(size_t j=0;j<4;++j){double sum=0;
            for(size_t b=0;b<blocks;++b){int dot=0;
                for(size_t c=0;c<32;++c){uint8_t raw=w[r*(cols/2)+b*64+j*16+c/2];
                    dot+=grid[(raw>>((c%2)*4))&15]*input->data[b*64+c];}
                sum+=(double)dot*e8(s[r*blocks+b*4+j])*(input->scale[b]*.5f);
            }out[r+j]=(float)sum;
        }

    }
}
int ds41f_mxfp4_sdot_prepared(float *out,const uint8_t *w,const uint8_t *s,
                           const ds41f_int8_input *input,size_t rows,size_t cols,int reference)
{
    if(!out||!w||!s||!input||!input->data||!input->scale||!rows||rows%4||!cols||cols%32||
       rows>SIZE_MAX/cols||input->elements!=cols||input->block!=32)return EINVAL;
    size_t blocks=cols/32;
    if(ds41f_team_active()){
        mxfp4_sdot_prepared_team_job job={out, w, s, input, rows, cols, blocks, reference};
        (void)ds41f_team_for(rows/4,mxfp4_sdot_prepared_team_work,&job);
    }else
    #pragma omp parallel for schedule(static)
    for(size_t r=0;r<rows;r+=4){
        #if defined(__ARM_FEATURE_SVE)
        if(!reference&&svcntw()==16){
            svbool_t pg=svptrue_b32(),p4=svwhilelt_b32((uint64_t)0,(uint64_t)4);
            svuint32_t expand=svlsr_n_u32_x(pg,svindex_u32(0,1),2);
            svint8_t lut=svld1_s8(svptrue_b8(),grid);svfloat32_t sum=svdup_f32(0);
            for(size_t b=0;b<blocks;++b){
                svuint8_t raw=svld1_u8(svptrue_b8(),w+r*(cols/2)+b*64);
                svint8_t lo=svtbl_s8(lut,svand_n_u8_x(svptrue_b8(),raw,15));
                svint8_t hi=svtbl_s8(lut,svlsr_n_u8_x(svptrue_b8(),raw,4));
                svint8_t x=svld1_s8(svptrue_b8(),input->data+b*64);
                svint32_t dot=svdot_s32(svdup_s32(0),lo,svuzp1_s8(x,x));
                dot=svdot_s32(dot,hi,svuzp2_s8(x,x));
                svuint32_t sc=svld1ub_u32(p4,s+r*blocks+b*4);
                svuint32_t bits=svlsl_n_u32_x(p4,sc,23);
                bits=svsel_u32(svcmpeq_n_u32(p4,sc,0),svdup_u32(0x00400000u),bits);
                svfloat32_t scale=svmul_n_f32_x(pg,svtbl_f32(svreinterpret_f32_u32(bits),expand),input->scale[b]*.5f);
                sum=svmla_f32_x(pg,sum,svcvt_f32_s32_x(pg,dot),scale);
            }
            for(size_t j=0;j<4;++j){svuint32_t idx=svindex_u32(0,1);
                svbool_t mask=svand_b_z(pg,svcmpge_n_u32(pg,idx,j*4),svcmplt_n_u32(pg,idx,j*4+4));
                out[r+j]=svaddv_f32(mask,sum);}
            continue;
        }
        #else
        (void)reference;
        #endif
        for(size_t j=0;j<4;++j){double sum=0;
            for(size_t b=0;b<blocks;++b){int dot=0;
                for(size_t c=0;c<32;++c){uint8_t raw=w[r*(cols/2)+b*64+j*16+c/2];
                    dot+=grid[(raw>>((c%2)*4))&15]*input->data[b*64+c];}
                sum+=(double)dot*e8(s[r*blocks+b*4+j])*(input->scale[b]*.5f);
            }out[r+j]=(float)sum;
        }
    }
    return 0;
}
int ds41f_mxfp4_sdot(float *out,const uint8_t *w,const uint8_t *s,const float *x,
                   size_t rows,size_t cols,int reference)
{
    ds41f_int8_input input;int rc=ds41f_int8_prepare_input(&input,x,cols,32);
    if(rc)return rc;
    rc=ds41f_mxfp4_sdot_prepared(out,w,s,&input,rows,cols,reference);
    ds41f_int8_input_free(&input);return rc;
}
#if defined(__ARM_FEATURE_SVE)
typedef struct {float *gate,*up;const uint8_t *wg,*sg,*wu,*su;const ds41f_int8_input *input;size_t cols;} fp4_pair_job;
static void fp4_pair_work(void *context,size_t first,size_t last)
{
    fp4_pair_job *job=context;size_t cols=job->cols,blocks=cols/32;
    svbool_t pg=svptrue_b32(),p4=svwhilelt_b32((uint64_t)0,(uint64_t)4);
    svuint32_t expand=svlsr_n_u32_x(pg,svindex_u32(0,1),2);
    svint8_t lut=svld1_s8(svptrue_b8(),grid);
    for(size_t tile=first;tile<last;++tile){size_t r=tile*4;
        svfloat32_t gate=svdup_f32(0),up=gate;
        for(size_t b=0;b<blocks;++b){
            svint8_t x=svld1_s8(svptrue_b8(),job->input->data+b*64),even=svuzp1_s8(x,x),odd=svuzp2_s8(x,x);
            float xs=job->input->scale[b]*.5f;
            #define FP4_PAIR(acc,weight,scales) { \
                svuint8_t raw=svld1_u8(svptrue_b8(),(weight)+r*(cols/2)+b*64); \
                svint8_t lo=svtbl_s8(lut,svand_n_u8_x(svptrue_b8(),raw,15)); \
                svint8_t hi=svtbl_s8(lut,svlsr_n_u8_x(svptrue_b8(),raw,4)); \
                svint32_t dot=svdot_s32(svdup_s32(0),lo,even);dot=svdot_s32(dot,hi,odd); \
                svuint32_t sc=svld1ub_u32(p4,(scales)+r*blocks+b*4); \
                svuint32_t bits=svlsl_n_u32_x(p4,sc,23); \
                bits=svsel_u32(svcmpeq_n_u32(p4,sc,0),svdup_u32(0x00400000u),bits); \
                svfloat32_t scale=svmul_n_f32_x(pg,svtbl_f32(svreinterpret_f32_u32(bits),expand),xs); \
                acc=svmla_f32_x(pg,acc,svcvt_f32_s32_x(pg,dot),scale); }
            FP4_PAIR(gate,job->wg,job->sg);FP4_PAIR(up,job->wu,job->su);
            #undef FP4_PAIR
        }
        for(size_t j=0;j<4;++j){svuint32_t ix=svindex_u32(0,1);
            svbool_t mask=svand_b_z(pg,svcmpge_n_u32(pg,ix,j*4),svcmplt_n_u32(pg,ix,j*4+4));
            job->gate[r+j]=svaddv_f32(mask,gate);job->up[r+j]=svaddv_f32(mask,up);}
    }
}
#endif
int ds41f_mxfp4_sdot_pair_prepared(float *gate,float *up,const uint8_t *wg,const uint8_t *sg,
                                  const uint8_t *wu,const uint8_t *su,const ds41f_int8_input *input,
                                  size_t rows,size_t cols)
{
    if(!gate||!up||!wg||!sg||!wu||!su||!input||!input->data||!input->scale||!rows||rows%4||
       !cols||cols%32||rows>SIZE_MAX/cols||input->elements!=cols||input->block!=32)return EINVAL;
#if defined(__ARM_FEATURE_SVE)
    if(svcntw()==16){fp4_pair_job job={gate,up,wg,sg,wu,su,input,cols};
        if(ds41f_team_active())return ds41f_team_for(rows/4,fp4_pair_work,&job);
        #pragma omp parallel for schedule(static)
        for(size_t r=0;r<rows/4;++r)fp4_pair_work(&job,r,r+1);
        return 0;
    }
#endif
    int rc=ds41f_mxfp4_sdot_prepared(gate,wg,sg,input,rows,cols,0);
    return rc?rc:ds41f_mxfp4_sdot_prepared(up,wu,su,input,rows,cols,0);
}
