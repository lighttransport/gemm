#include "ds41f_sve.h"
#include "ds41f_kernels.h"
#include <errno.h>
#include <string.h>

static inline float scale_e8m0(uint8_t s)
{
    uint32_t bits = s == 0 ? 0x00400000u : s == 255 ? 0x7fc00000u : (uint32_t)s << 23;
    float f; memcpy(&f,&bits,4); return f;
}

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

static inline svfloat32_t load_bf16(svbool_t pg, const uint16_t *p)
{
    return svreinterpret_f32(svlsl_n_u32_x(pg, svld1uh_u32(pg, p), 16));
}

#if defined(DS41F_FP8_BITS)
static inline svfloat32_t decode_fp8_bits(svbool_t pg,svuint32_t raw)
{
    svuint32_t mag=svand_n_u32_x(pg,raw,127);
    svfloat32_t value=svreinterpret_f32(svadd_n_u32_x(pg,svlsl_n_u32_x(pg,mag,20),0x3c000000u));
    value=svmad_m(svcmplt_n_u32(pg,mag,8),value,svdup_f32(2),svdup_f32(-0x1p-6f));
    value=svsel_f32(svcmpeq_n_u32(pg,mag,127),svreinterpret_f32(svdup_u32(0x7fc00000u)),value);
    return svneg_m(value,svcmpge_n_u32(pg,raw,128),value);
}
#endif

void ds41f_bf16_matvec(float *out, const uint16_t *w, const uint16_t *x,
                       size_t rows, size_t cols)
{
    svbool_t pg = svptrue_b32();
    size_t vl = svcntw();
    for (size_t r = 0; r < rows; ++r) {
        svfloat32_t acc = svdup_f32(0.0f);
        size_t c = 0;
        for (; c + vl <= cols; c += vl)
            acc = svmla_x(pg, acc, load_bf16(pg, w + r * cols + c),
                          load_bf16(pg, x + c));
        if (c < cols) {
            svbool_t tail = svwhilelt_b32(c, cols);
            acc = svmla_m(tail, acc, load_bf16(tail, w + r * cols + c),
                          load_bf16(tail, x + c));
        }
        out[r] = svaddv(pg, acc);
    }
}

void ds41f_rmsnorm_fast(float *out, const float *x, const uint16_t *weight,
                        size_t n, float eps)
{
    svbool_t pg = svptrue_b32();
    size_t vl = svcntw();
    svfloat32_t ss = svdup_f32(0.0f);
    size_t i = 0;
    for (; i + vl <= n; i += vl) {
        svfloat32_t v = svld1(pg, x + i);
        ss = svmla_x(pg, ss, v, v);
    }
    if (i < n) {
        svbool_t tail = svwhilelt_b32(i, n);
        svfloat32_t v = svld1(tail, x + i);
        ss = svmla_m(tail, ss, v, v);
    }
    float inv = 1.0f / __builtin_sqrtf(svaddv(pg, ss) / (float)n + eps);
    for (i = 0; i < n; i += vl) {
        svbool_t q = svwhilelt_b32(i, n);
        svfloat32_t v = svmul_x(q, svld1(q, x + i), inv);
        v = svmul_x(q, v, load_bf16(q, weight + i));
        svst1(q, out + i, v);
    }
}

#else

void ds41f_bf16_matvec(float *out, const uint16_t *w, const uint16_t *x,
                       size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c)
            sum += ds41f_bf16_to_f32(w[r * cols + c]) * ds41f_bf16_to_f32(x[c]);
        out[r] = sum;
    }
}

void ds41f_rmsnorm_fast(float *out, const float *x, const uint16_t *weight,
                        size_t n, float eps)
{ ds41f_rmsnorm_ref(out, x, weight, n, eps); }

#endif

void ds41f_bf16_f32_matvec(float *out,const uint16_t *w,const float *x,size_t rows,size_t cols)
{
    #pragma omp parallel for schedule(static)
    for(size_t r=0;r<rows;++r){
        #if defined(__ARM_FEATURE_SVE)
        svbool_t pg=svptrue_b32();size_t vl=svcntw(),c=0;
        svfloat32_t a=svdup_f32(0),b=svdup_f32(0);
        for(;c+2*vl<=cols;c+=2*vl){
            a=svmla_x(pg,a,load_bf16(pg,w+r*cols+c),svld1(pg,x+c));
            b=svmla_x(pg,b,load_bf16(pg,w+r*cols+c+vl),svld1(pg,x+c+vl));}
        for(;c<cols;c+=vl){svbool_t tail=svwhilelt_b32(c,cols);
            a=svmla_m(tail,a,load_bf16(tail,w+r*cols+c),svld1(tail,x+c));}
        out[r]=svaddv(pg,svadd_x(pg,a,b));
        #else
        float sum=0;
        for(size_t c=0;c<cols;++c){union {uint32_t u;float f;}v={(uint32_t)w[r*cols+c]<<16};sum+=v.f*x[c];}
        out[r]=sum;
        #endif
    }
}

int ds41f_mxfp4_matvec(float *out, const uint8_t *w, const uint8_t *scale,
                        const float *x, size_t rows, size_t cols)
{
    if (!out || !w || !scale || !x || !cols || cols%32) return EINVAL;
#if defined(__ARM_FEATURE_SVE)
    if (svcntw()!=16) return ds41f_mxfp4_matvec_ref(out,w,scale,x,rows,cols);
    static const float lut[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
    #pragma omp parallel for schedule(static)
    for (size_t r=0;r<rows;++r) {
        svbool_t pg=svptrue_b32();
        svfloat32_t table=svld1(pg,lut), acc=svdup_f32(0);
        for(size_t b=0;b<cols/32;++b) {
            svuint32_t raw=svld1ub_u32(pg,w+r*(cols/2)+b*16);
            svfloat32_t lo=svtbl_f32(table,svand_n_u32_x(pg,raw,15));
            svfloat32_t hi=svtbl_f32(table,svlsr_n_u32_x(pg,raw,4));
            svfloat32_t a=svld1(pg,x+b*32), z=svld1(pg,x+b*32+16);
            svfloat32_t even=svuzp1_f32(a,z), odd=svuzp2_f32(a,z);
            svfloat32_t prod=svmul_x(pg,lo,even);
            prod=svmla_x(pg,prod,hi,odd);
            acc=svmla_n_f32_x(pg,acc,prod,scale_e8m0(scale[r*(cols/32)+b]));
        }
        out[r]=svaddv(pg,acc);
    }
    return 0;
#else
    return ds41f_mxfp4_matvec_ref(out,w,scale,x,rows,cols);
#endif
}

static int fp8_matvec_impl(float *out, const uint8_t *w, const uint8_t *scale,
                      const float *x, size_t rows, size_t cols,size_t group_rows)
{
    if (!out || !w || !scale || !x || !cols) return EINVAL;
#if defined(__ARM_FEATURE_SVE)
    #if !defined(DS41F_FP8_BITS)
    float lut[256];
    for(int i=0;i<256;++i) lut[i]=ds41f_fp8_e4m3_to_f32((uint8_t)i);
    #endif
    /* Small matrices benefit from row interleaving. Larger working sets
     * regressed in measured HBM-streaming cases; retain contiguous rows. */
    const size_t tile_rows=rows<=((size_t)16*1024*1024)/cols?4:1;
    #pragma omp parallel for schedule(static)
    for(size_t r=0;r<rows;r+=tile_rows) {
        const float *input=x+(r/group_rows)*cols;
        svbool_t pg=svptrue_b32();
        svfloat32_t acc=svdup_f32(0),a1=acc,a2=acc,a3=acc;
        for(size_t b=0;b<cols;b+=32) {
            size_t end=b+32<cols?b+32:cols;
            float sc=scale_e8m0(scale[(r/32)*((cols+31)/32)+b/32]);
            for(size_t c=b;c<end;c+=svcntw()) {
                svbool_t q=svwhilelt_b32(c,end);
                svuint32_t codes=svld1ub_u32(q,w+r*cols+c);
                #if defined(DS41F_FP8_BITS)
                svfloat32_t weights=decode_fp8_bits(q,codes);
                #else
                svfloat32_t weights=svld1_gather_u32index_f32(q,lut,codes);
                #endif
                weights=svmul_n_f32_x(q,weights,sc);
                svfloat32_t xv=svld1(q,input+c);
                acc=svmla_m(q,acc,weights,xv);
                /* Four independent rows preserve each row's accumulation
                 * order while sharing input loads and the 32-row scale. */
                if(tile_rows==4&&r+1<rows){codes=svld1ub_u32(q,w+(r+1)*cols+c);
                    #if defined(DS41F_FP8_BITS)
                    weights=decode_fp8_bits(q,codes);
                    #else
                    weights=svld1_gather_u32index_f32(q,lut,codes);
                    #endif
                    a1=svmla_m(q,a1,svmul_n_f32_x(q,weights,sc),xv);}
                if(tile_rows==4&&r+2<rows){codes=svld1ub_u32(q,w+(r+2)*cols+c);
                    #if defined(DS41F_FP8_BITS)
                    weights=decode_fp8_bits(q,codes);
                    #else
                    weights=svld1_gather_u32index_f32(q,lut,codes);
                    #endif
                    a2=svmla_m(q,a2,svmul_n_f32_x(q,weights,sc),xv);}
                if(tile_rows==4&&r+3<rows){codes=svld1ub_u32(q,w+(r+3)*cols+c);
                    #if defined(DS41F_FP8_BITS)
                    weights=decode_fp8_bits(q,codes);
                    #else
                    weights=svld1_gather_u32index_f32(q,lut,codes);
                    #endif
                    a3=svmla_m(q,a3,svmul_n_f32_x(q,weights,sc),xv);}
            }
        }
        out[r]=svaddv(pg,acc);
        if(tile_rows==4&&r+1<rows)out[r+1]=svaddv(pg,a1);
        if(tile_rows==4&&r+2<rows)out[r+2]=svaddv(pg,a2);
        if(tile_rows==4&&r+3<rows)out[r+3]=svaddv(pg,a3);
    }
#else
    for(size_t r=0;r<rows;r+=group_rows)
        ds41f_fp8_matvec_ref(out+r,w+r*cols,scale+(r/32)*((cols+31)/32),
                            x+(r/group_rows)*cols,group_rows,cols,32);
#endif
    return 0;
}
int ds41f_fp8_matvec(float *out,const uint8_t *w,const uint8_t *scale,
                    const float *x,size_t rows,size_t cols)
{return fp8_matvec_impl(out,w,scale,x,rows,cols,rows);}

int ds41f_fp8_grouped_matvec(float *out,const uint8_t *w,const uint8_t *scale,
                            const float *x,size_t groups,size_t group_rows,size_t cols)
{
    if(!groups||!group_rows||group_rows%32||groups>SIZE_MAX/group_rows)return EINVAL;
    return fp8_matvec_impl(out,w,scale,x,groups*group_rows,cols,group_rows);
}

int ds41f_mxfp4_matvec_pair(float *gate,float *up,const uint8_t *wg,const uint8_t *sg,
                           const uint8_t *wu,const uint8_t *su,const float *x,
                           size_t rows,size_t cols,int tile)
{
    if(!gate||!up||!wg||!sg||!wu||!su||!x||!cols||cols%32||(tile!=1&&tile!=2))return EINVAL;
#if defined(__ARM_FEATURE_SVE)
    if(svcntw()==16){
        static const float lut[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
        #pragma omp parallel for schedule(static)
        for(size_t r=0;r<rows;r+=(size_t)tile){
            svbool_t pg=svptrue_b32();svfloat32_t table=svld1_f32(pg,lut);
            svfloat32_t g0=svdup_f32(0),u0=g0,g1=g0,u1=g0;
            for(size_t b=0;b<cols/32;++b){
                svfloat32_t a=svld1_f32(pg,x+b*32),z=svld1_f32(pg,x+b*32+16);
                svfloat32_t even=svuzp1_f32(a,z),odd=svuzp2_f32(a,z);
                #define PAIR_DOT(acc,weight,scales,row) { \
                    svuint32_t raw=svld1ub_u32(pg,(weight)+(row)*(cols/2)+b*16); \
                    svfloat32_t lo=svtbl_f32(table,svand_n_u32_x(pg,raw,15)); \
                    svfloat32_t hi=svtbl_f32(table,svlsr_n_u32_x(pg,raw,4)); \
                    svfloat32_t prod=svmul_f32_x(pg,lo,even); \
                    prod=svmla_f32_x(pg,prod,hi,odd); \
                    acc=svmla_n_f32_x(pg,acc,prod,scale_e8m0((scales)[(row)*(cols/32)+b])); }
                PAIR_DOT(g0,wg,sg,r);PAIR_DOT(u0,wu,su,r);
                if(tile==2&&r+1<rows){PAIR_DOT(g1,wg,sg,r+1);PAIR_DOT(u1,wu,su,r+1);}
                #undef PAIR_DOT
            }
            gate[r]=svaddv_f32(pg,g0);up[r]=svaddv_f32(pg,u0);
            if(tile==2&&r+1<rows){gate[r+1]=svaddv_f32(pg,g1);up[r+1]=svaddv_f32(pg,u1);}
        }
        return 0;
    }
#endif
    int rc=ds41f_mxfp4_matvec(gate,wg,sg,x,rows,cols);
    return rc?rc:ds41f_mxfp4_matvec(up,wu,su,x,rows,cols);
}
