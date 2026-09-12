#include "ds41f_kernels.h"
#include "ds41f_team.h"
#include "ds41f_quant_sve.h"
#include <math.h>
#include <errno.h>

static int quant_parallel;
void ds41f_set_quant_parallel(int enabled){quant_parallel=enabled!=0;}
int ds41f_get_quant_parallel(void){return quant_parallel;}

uint8_t ds41f_f32_to_fp8(float x)
{
    union {float f;uint32_t u;} v={x};
    unsigned sign=(v.u>>24)&128;v.u&=0x7fffffffu;
    if(v.u>0x7f800000u)return 127;
    if(v.f>=448)return (uint8_t)(sign|126);
    unsigned code;
    if(v.f<0x1p-6f){float scaled=v.f*512;code=(unsigned)scaled;
        float remainder=scaled-code;if(remainder>.5f||(remainder==.5f&&(code&1)))++code;
    }else code=((v.u+0x7ffffu+((v.u>>20)&1))>>20)-960;
    return (uint8_t)(sign|code);
}

int ds41f_act_quant_ref(float *out,const float *x,size_t n)
{
    if (!out || !x || !n || n%32) return EINVAL;
    for (size_t b=0;b<n;b+=32) {
        float v[32],max=1e-4f;
        for (int i=0;i<32;++i) {
            v[i]=ds41f_bf16_to_f32(ds41f_f32_to_bf16(x[b+i]));
            if (!__builtin_isfinite(v[i])) return EDOM;
            if (__builtin_fabsf(v[i])>max) max=__builtin_fabsf(v[i]);
        }
        float scale=__builtin_exp2f(__builtin_ceilf(__builtin_log2f(max/448.f)));
        for (int i=0;i<32;++i)
            out[b+i]=ds41f_fp8_e4m3_to_f32(ds41f_f32_to_fp8(v[i]/scale))*scale;
    }
    return 0;
}

#if defined(__ARM_FEATURE_SVE)
typedef struct {float *out;const float *x;int *errors;} act_quant_team_job;
static void act_quant_team_work(void *context,size_t first,size_t last)
{
    act_quant_team_job *job=context;int invalid=0;
    for(size_t b=first*32;b<last*32;b+=32){uint16_t quant[32];int rc=ds41f_quantize32_bf16(quant,job->x+b);
        if(rc){invalid|=rc;continue;}
        for(size_t j=0;j<32;j+=svcntw()){
            svbool_t pg=svwhilelt_b32(j,(size_t)32);
            svst1(pg,job->out+b+j,svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,quant+j),16)));}}
    if(invalid)job->errors[ds41f_team_thread_id()]=invalid;
}
#endif
int ds41f_act_quant(float *out,const float *x,size_t n)
{
    #if defined(__ARM_FEATURE_SVE)
    if(!out||!x||!n||n%32)return EINVAL;
    int invalid=0;
    if(ds41f_team_active()&&quant_parallel&&n>=1024){
        int errors[48]={0};act_quant_team_job job={out,x,errors};
        (void)ds41f_team_for(n/32,act_quant_team_work,&job);
        for(int i=0;i<48;++i)invalid|=errors[i];
    }else if(quant_parallel&&n>=5120){
        #pragma omp parallel for schedule(static) reduction(|:invalid)
        for(size_t b=0;b<n;b+=32){uint16_t quant[32];int rc=ds41f_quantize32_bf16(quant,x+b);
            if(rc){invalid|=rc;continue;}
            for(size_t j=0;j<32;j+=svcntw()){
                svbool_t pg=svwhilelt_b32(j,(size_t)32);
                svst1(pg,out+b+j,svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,quant+j),16)));}}
    }else{
        for(size_t b=0;b<n;b+=32){uint16_t quant[32];int rc=ds41f_quantize32_bf16(quant,x+b);if(rc)return rc;
            for(size_t j=0;j<32;j+=svcntw()){
                svbool_t pg=svwhilelt_b32(j,(size_t)32);
                svst1(pg,out+b+j,svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,quant+j),16)));}}
    }
    return invalid;
    #else
    return ds41f_act_quant_ref(out,x,n);
    #endif
}

int ds41f_mxfp4_matvec_ref(float *out, const uint8_t *packed,
                          const uint8_t *scales, const float *x,
                          size_t rows, size_t cols)
{
    static const float values[16] = {
        0, .5f, 1, 1.5f, 2, 3, 4, 6,
        -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6
    };
    if (!out || !packed || !scales || !x || !cols || cols % 32)
        return EINVAL;
    for (size_t r=0; r<rows; ++r) {
        double sum=0;
        for (size_t c=0; c<cols; ++c) {
            unsigned byte=packed[r*(cols/2)+c/2];
            unsigned code=(byte >> ((c%2)*4)) & 15;
            sum += (double)values[code] * ds41f_e8m0_to_f32(scales[r*(cols/32)+c/32]) * x[c];
        }
        out[r]=(float)sum;
    }
    return 0;
}

float ds41f_fp8_e4m3_to_f32(uint8_t x)
{
    unsigned magnitude=x&127;
    if(magnitude<8)return (x&128?-(float)magnitude:(float)magnitude)*(1.f/512);
    union {uint32_t u;float f;} v={magnitude==127?0x7fc00000u:(magnitude<<20)+0x3c000000u};
    v.u|=(uint32_t)(x&128)<<24;return v.f;
}

float ds41f_e8m0_to_f32(uint8_t x)
{
    union {uint32_t u;float f;} v={x==0?0x00400000u:x==255?0x7fc00000u:(uint32_t)x<<23};return v.f;
}

uint16_t ds41f_f32_to_bf16(float x)
{
    union { float f; uint32_t u; } v = { x };
    if (isnan(x)) return (uint16_t)((v.u >> 16) | 0x40u);
    uint32_t round = ((v.u >> 16) & 1u) + 0x7fffu;
    return (uint16_t)((v.u + round) >> 16);
}

float ds41f_bf16_to_f32(uint16_t x)
{
    union { uint32_t u; float f; } v = { (uint32_t)x << 16 };
    return v.f;
}

void ds41f_round_bf16(float *x,size_t n)
{
    #if defined(__ARM_FEATURE_SVE)
    for(size_t i=0;i<n;i+=svcntw()){
        svbool_t pg=svwhilelt_b32(i,n);
        svuint32_t bits=svreinterpret_u32_f32(svld1(pg,x+i));
        svuint32_t odd=svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),1);
        svuint32_t rounded=svadd_u32_x(pg,bits,svadd_n_u32_x(pg,odd,0x7fff));
        /* Match the scalar NaN payload/sign rule, including signaling NaNs. */
        svbool_t nan=svcmpgt_n_u32(pg,svand_n_u32_x(pg,bits,0x7fffffffu),0x7f800000u);
        rounded=svsel_u32(nan,svorr_n_u32_x(pg,bits,0x00400000u),rounded);
        svst1(pg,x+i,svreinterpret_f32_u32(svand_n_u32_x(pg,rounded,0xffff0000u)));
    }
    #else
    for(size_t i=0;i<n;++i)x[i]=ds41f_bf16_to_f32(ds41f_f32_to_bf16(x[i]));
    #endif
}

void ds41f_fp8_matvec_ref(float *out, const uint8_t *w, const uint8_t *scale,
                          const float *x, size_t rows, size_t cols,
                          size_t block_cols)
{
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c)
            sum += ds41f_fp8_e4m3_to_f32(w[r * cols + c]) *
                   ds41f_e8m0_to_f32(scale[(r / 32) * ((cols + block_cols - 1) / block_cols) + c / block_cols]) * x[c];
        out[r] = sum;
    }
}

void ds41f_i8_matvec_ref(float *out, const int8_t *w, const uint8_t *scale,
                         const float *x, size_t rows, size_t cols,
                         size_t block_cols)
{
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c)
            sum += (float)w[r * cols + c] *
                   ds41f_e8m0_to_f32(scale[(r * cols + c) / block_cols]) * x[c];
        out[r] = sum;
    }
}

void ds41f_rmsnorm_ref(float *out, const float *x, const uint16_t *weight,
                       size_t n, float eps)
{
    float ss = 0.0f;
    for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; ++i)
        out[i] = x[i] * inv * ds41f_bf16_to_f32(weight[i]);
}
