#define _POSIX_C_SOURCE 200112L
#include "fp4_gemm.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

static const float e2m1_values[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f,
};

const char *fp4_format_name(fp4_format f) {
    static const char *names[] = {"mxfp4", "nvfp4-1d", "nvfp4-2d"};
    return (unsigned)f < 3 ? names[f] : "unknown";
}

float fp4_e2m1_decode(uint8_t v) { return e2m1_values[v & 15]; }

uint8_t fp4_e2m1_encode(float x) {
    if (isnan(x)) return 0;
    int sign = signbit(x) ? 8 : 0;
    float a = fabsf(x);
    int best = a <= .25f ? 0 : a < .75f ? 1 : a <= 1.25f ? 2 :
               a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 :
               a <= 5.0f ? 6 : 7;
    return (uint8_t)(sign | best);
}

float fp4_e8m0_decode(uint8_t e) {
    if (e == 255) return INFINITY;
    return ldexpf(1.0f, (int)e - 127);
}

uint8_t fp4_e8m0_encode_ceil(float x) {
    if (!(x > 0.0f)) return 0;
    if (!isfinite(x)) return 254;
    int exp;
    float m = frexpf(x, &exp);
    int unbiased = exp - 1 + (m != 0.5f);
    if (unbiased < -127) return 0;
    if (unbiased > 127) return 254;
    return (uint8_t)(unbiased + 127);
}

float fp4_e4m3_decode_positive(uint8_t b) {
    b &= 0x7f;
    int e = (b >> 3) & 15, m = b & 7;
    if (e == 0) return ldexpf((float)m, -9);
    if (e == 15 && m == 7) return 448.0f;
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

uint8_t fp4_e4m3_encode_positive(float x) {
    if (!(x > 0.0f)) return 0;
    if (!isfinite(x) || x >= 448.0f) return 0x7e;
    uint8_t best = 0;
    float bd = FLT_MAX;
    for (int b = 0; b <= 0x7e; ++b) {
        float v = fp4_e4m3_decode_positive((uint8_t)b);
        float d = fabsf(x - v);
        if (d < bd || (d == bd && (b & 1) == 0 && (best & 1))) {
            best = (uint8_t)b; bd = d;
        }
    }
    return best;
}

static size_t scale_count(fp4_format f, int n, int k) {
    if (f == FP4_MX) return (size_t)n * (k / 32);
    if (f == FP4_NV_1D) return (size_t)n * (k / 16);
    return (size_t)(n / 16) * (k / 16);
}

int fp4_matrix_alloc(fp4_matrix *p, fp4_format f, int n, int k) {
    if (!p || n <= 0 || k <= 0 || n % 32 || k % 32 || (unsigned)f >= 3)
        return -1;
    memset(p, 0, sizeof(*p));
    p->format = f; p->n = n; p->k = k; p->global_scale = 1.0f;
    p->code_bytes = (size_t)n * k / 2;
    p->scale_bytes = scale_count(f, n, k);
    if (posix_memalign((void **)&p->codes, 256, p->code_bytes) ||
        posix_memalign((void **)&p->scales, 256, p->scale_bytes)) {
        fp4_matrix_free(p); return -1;
    }
    return 0;
}

void fp4_matrix_free(fp4_matrix *p) {
    if (!p) return;
    free(p->codes); free(p->scales); memset(p, 0, sizeof(*p));
}

static float block_amax_1d(const float *w, int k, int row, int col, int bs) {
    float a = 0;
    for (int i = 0; i < bs; ++i) a = fmaxf(a, fabsf(w[(size_t)row*k+col+i]));
    return a;
}

int fp4_quantize_f32(fp4_matrix *p, const float *w) {
    if (!p || !w || !p->codes || !p->scales) return -1;
    const int n=p->n,k=p->k;
    if (p->format != FP4_MX) {
        float amax=0;
        for (size_t i=0;i<(size_t)n*k;++i) amax=fmaxf(amax,fabsf(w[i]));
        p->global_scale = amax > 0 ? amax/(448.0f*6.0f) : 1.0f;
    }
    memset(p->codes,0,p->code_bytes);
    if (p->format == FP4_NV_2D) {
        int kb=k/16;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for(int rt=0;rt<n/16;++rt) for(int b=0;b<kb;++b){
            float amax=0;
            for(int rr=0;rr<16;++rr) for(int j=0;j<16;++j)
                amax=fmaxf(amax,fabsf(w[(size_t)(rt*16+rr)*k+b*16+j]));
            float raw=amax>0?amax/(6.0f*p->global_scale):0;
            uint8_t se=fp4_e4m3_encode_positive(raw);
            p->scales[(size_t)rt*kb+b]=se;
            float s=fp4_e4m3_decode_positive(se)*p->global_scale;
            if (!(s>0)) s=1;
            for(int rr=0;rr<16;++rr) for(int j=0;j<16;++j){
                int r=rt*16+rr,c=b*16+j;
                uint8_t q=fp4_e2m1_encode(w[(size_t)r*k+c]/s);
                size_t z=(size_t)r*(k/2)+c/2;
                p->codes[z]|=(uint8_t)(q<<((c&1)*4));
            }
        }
        return 0;
    }
    int bs=p->format==FP4_MX?32:16, nb=k/bs;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<n;++r) for(int b=0;b<nb;++b){
        float amax=block_amax_1d(w,k,r,b*bs,bs), s;
        uint8_t se;
        if(p->format==FP4_MX){ se=fp4_e8m0_encode_ceil(amax/6.0f); s=fp4_e8m0_decode(se); }
        else { se=fp4_e4m3_encode_positive(amax>0?amax/(6.0f*p->global_scale):0);
               s=fp4_e4m3_decode_positive(se)*p->global_scale; }
        p->scales[(size_t)r*nb+b]=se;
        if(!(s>0)&&amax>0) s=amax/6.0f; else if(!(s>0)) s=1;
        for(int j=0;j<bs;++j){ int c=b*bs+j; uint8_t q=fp4_e2m1_encode(w[(size_t)r*k+c]/s);
            size_t z=(size_t)r*(k/2)+c/2; p->codes[z]|=(uint8_t)(q<<((c&1)*4)); }
    }
    return 0;
}

static float row_scale(const fp4_matrix *p,int r,int c){
    if(p->format==FP4_MX) return fp4_e8m0_decode(p->scales[(size_t)r*(p->k/32)+c/32]);
    if(p->format==FP4_NV_1D) return p->global_scale*fp4_e4m3_decode_positive(p->scales[(size_t)r*(p->k/16)+c/16]);
    return p->global_scale*fp4_e4m3_decode_positive(p->scales[(size_t)(r/16)*(p->k/16)+c/16]);
}

float fp4_dequant_value(const fp4_matrix*p,int r,int c){
    uint8_t b=p->codes[(size_t)r*(p->k/2)+c/2];
    return fp4_e2m1_decode((uint8_t)((b>>((c&1)*4))&15))*row_scale(p,r,c);
}

int fp4_gemm_reference(float*c,const _Float16*a,const fp4_matrix*w,int m,int fp16_products){
    if(!c||!a||!w||m<1)return -1;
    for(int i=0;i<m;++i)for(int r=0;r<w->n;++r){double s=0;
        for(int k=0;k<w->k;++k){float q=fp4_dequant_value(w,r,k);
            if(fp16_products)q=(float)(_Float16)q;
            s+=(double)(float)a[(size_t)i*w->k+k]*q;}
        c[(size_t)i*w->n+r]=(float)s;}
    return 0;
}

#if defined(__ARM_FEATURE_SVE)
static inline void dot_rows6_sve(float*out,const _Float16*a,int astride,int rows,
        const fp4_matrix*w,int r,int promotion_k){
    static const __fp16 table_data[32] __attribute__((aligned(64)))={
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6,
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
    svbool_t ph=svptrue_b16(), p16=svwhilelt_b16(0,16);
    svfloat16_t tab=svld1_f16(ph,table_data);
    svfloat16_t a0=svdup_f16(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0;
    float shadow[6]={0};
    int span=promotion_k?promotion_k:w->k, since=0;
    __fp16 lane[32] __attribute__((aligned(64)));
    __fp16 scale_lane[32] __attribute__((aligned(64)));
    for(int k0=0;k0<w->k;k0+=32){
        const uint8_t*q=w->codes+(size_t)r*(w->k/2)+k0/2;
        svuint16_t bytes=svld1ub_u16(p16,q),lo=svand_n_u16_x(p16,bytes,15);
        svuint16_t hi=svlsr_n_u16_x(p16,bytes,4);
        svuint16_t idx=svzip1_u16(lo,hi);
        svfloat16_t weights=svtbl_f16(tab,idx);
        __fp16 s0=(__fp16)row_scale(w,r,k0),s1=(__fp16)row_scale(w,r,k0+16);
        for(int j=0;j<16;++j)scale_lane[j]=s0;
        for(int j=16;j<32;++j)scale_lane[j]=s1;
        weights=svmul_f16_x(ph,weights,svld1_f16(ph,scale_lane));
#define FP4_FMA_ROW(I,A) do { if(rows>(I)){svfloat16_t x=svld1_f16(ph, \
            (const __fp16 *)(a+(size_t)(I)*astride+k0)); \
            (A)=svmla_f16_x(ph,(A),x,weights);} } while(0)
        FP4_FMA_ROW(0,a0);FP4_FMA_ROW(1,a1);FP4_FMA_ROW(2,a2);
        FP4_FMA_ROW(3,a3);FP4_FMA_ROW(4,a4);FP4_FMA_ROW(5,a5);
#undef FP4_FMA_ROW
        since+=32;
        if(since==span||k0+32==w->k){
#define FP4_PROMOTE(I,A) do { if(rows>(I)){svst1_f16(ph,lane,(A));float z=0; \
                for(int j=0;j<32;++j)z+=(float)lane[j];shadow[I]+=z;(A)=svdup_f16(0);} }while(0)
            FP4_PROMOTE(0,a0);FP4_PROMOTE(1,a1);FP4_PROMOTE(2,a2);
            FP4_PROMOTE(3,a3);FP4_PROMOTE(4,a4);FP4_PROMOTE(5,a5);
#undef FP4_PROMOTE
            since=0;}
    }
    for(int i=0;i<rows;++i)out[(size_t)i*w->n+r]=shadow[i];
}
#endif

int fp4_gemm_f16(float*c,const _Float16*a,const fp4_matrix*w,int m,int promotion_k,int threads){
    if(!c||!a||!w||m<1||promotion_k<0||(promotion_k&&((promotion_k%32)||promotion_k>w->k)))return -1;
    if(threads<1)threads=1;
    for(int m0=0;m0<m;m0+=6){int mr=m-m0<6?m-m0:6;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for(int r=0;r<w->n;++r){
#if defined(__ARM_FEATURE_SVE)
            dot_rows6_sve(c+(size_t)m0*w->n,a+(size_t)m0*w->k,w->k,mr,w,r,promotion_k);
#else
            for(int i=0;i<mr;++i){_Float16 h=0;float s=0;int span=promotion_k?promotion_k:w->k,used=0;
                for(int k=0;k<w->k;++k){_Float16 q=(_Float16)fp4_dequant_value(w,r,k);
                    h=(_Float16)(h+a[(size_t)(m0+i)*w->k+k]*q);if(++used==span){s+=(float)h;h=0;used=0;}}
                if(used)s+=(float)h;c[(size_t)(m0+i)*w->n+r]=s;}
#endif
        }}return 0;
}
