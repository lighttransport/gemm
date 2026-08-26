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

typedef struct {const int8_t*q,*a;const _Float16*ws;const float*as;float*out;
    int k,weight_group,act_group;} fp4_sdot_args;
extern void fp4_i8_sdot_m1_asm(const fp4_sdot_args*);
typedef struct {const uint8_t*q;const int8_t*a;const float*ws;const float*as;
    float*out;int ng,g,nb,pg;} fp4_sdot4_args;
extern void fp4_i8_sdot4_m1_asm(const fp4_sdot4_args*);
typedef struct {const uint8_t*q;const _Float16*ws;const int16_t*tab;
    const float*as;float*out;int ng,g0,gcount,nb,pairs,act_group;} fp4_pair_args;
extern void fp4_pair_lut_m1_asm(const fp4_pair_args*);
extern void fp4_pair_tbl_m1_asm(const fp4_pair_args*);

/* Scalar AArch64 producer LUT: packed FP4 byte -> two FP16 bit patterns. */
uint32_t fp4_pair_lut[256] __attribute__((aligned(1024)));
static int fp4_pair_lut_ready;
static void prepare_pair_lut(void){
    if(fp4_pair_lut_ready)return;
    for(int b=0;b<256;++b){_Float16 lo=(_Float16)e2m1_values[b&15];
        _Float16 hi=(_Float16)e2m1_values[b>>4];uint16_t l,h;
        memcpy(&l,&lo,2);memcpy(&h,&hi,2);fp4_pair_lut[b]=(uint32_t)l|((uint32_t)h<<16);}
    fp4_pair_lut_ready=1;
}

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
    free(p->codes); free(p->scales); free(p->codes_n32); free(p->codes_u8);
    free(p->codes_sdot);free(p->codes_sdot4);free(p->scales_sdot);free(p->codes_pair);free(p->scales_pair);
    free(p->codes_t8);free(p->codes_t8_half);free(p->codes_t12);free(p->codes_t8_affine);
    free(p->scales_t8);free(p->scales_t12);free(p->scales_sdot4);
    free(p->codes_bitplane);
    free(p->scales_n32); memset(p, 0, sizeof(*p));
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

int fp4_matrix_prepare_n32(fp4_matrix *p){
    if(!p||!p->codes||!p->scales||p->n%32||p->k%32)return -1;
    prepare_pair_lut();
    free(p->codes_n32);free(p->codes_u8);free(p->codes_sdot);free(p->codes_sdot4);free(p->codes_pair);free(p->codes_bitplane);free(p->codes_t8);free(p->codes_t8_half);free(p->codes_t12);free(p->codes_t8_affine);
    free(p->scales_n32);free(p->scales_sdot);free(p->scales_pair);free(p->scales_t8);free(p->scales_t12);free(p->scales_sdot4);
    p->codes_n32=NULL;p->codes_u8=NULL;p->codes_sdot=NULL;p->codes_sdot4=NULL;p->codes_pair=NULL;p->codes_bitplane=NULL;
    p->scales_n32=NULL;p->scales_sdot=NULL;p->scales_pair=NULL;p->codes_t8=NULL;p->codes_t8_half=NULL;p->codes_t12=NULL;p->codes_t8_affine=NULL;
    p->scales_t8=NULL;p->scales_t12=NULL;p->scales_sdot4=NULL;
    int nt=p->n/32,bs=p->format==FP4_MX?32:16,nb=p->k/bs;
    p->scales_n32_count=(size_t)nt*nb*32;
    if(posix_memalign((void**)&p->codes_n32,256,p->code_bytes)||
       posix_memalign((void**)&p->scales_n32,256,p->scales_n32_count*sizeof(_Float16))){
        free(p->codes_n32);free(p->codes_u8);free(p->scales_n32);p->codes_n32=NULL;p->codes_u8=NULL;p->scales_n32=NULL;return-1;}
    for(int t=0;t<nt;++t){
        for(int k=0;k<p->k;++k){uint8_t*q=p->codes_n32+((size_t)t*p->k+k)*16;
            for(int j=0;j<16;++j){int r=t*32+2*j;
                uint8_t a=p->codes[(size_t)r*(p->k/2)+k/2];
                uint8_t b=p->codes[(size_t)(r+1)*(p->k/2)+k/2];
                uint8_t qa=(a>>((k&1)*4))&15,qb=(b>>((k&1)*4))&15;
                q[j]=(uint8_t)(qa|(qb<<4));}}
        for(int b=0;b<nb;++b){_Float16*s=p->scales_n32+((size_t)t*nb+b)*32;
            for(int j=0;j<32;++j)s[j]=(_Float16)row_scale(p,t*32+j,b*bs);}
    }
    if(p->n%256==0){
      if(posix_memalign((void**)&p->codes_t8,256,p->code_bytes)||
         posix_memalign((void**)&p->scales_t8,256,p->scales_n32_count*sizeof(_Float16))){
        free(p->codes_t8);free(p->scales_t8);p->codes_t8=NULL;p->scales_t8=NULL;return-1;}
      int groups=p->n/256;
      for(int g=0;g<groups;++g)for(int b=0;b<nb;++b){
        uint8_t*dq=p->codes_t8+(((size_t)g*nb+b)*bs)*128;
        _Float16*ds=p->scales_t8+((size_t)g*nb+b)*256;
        for(int t=0;t<8;++t){const _Float16*ss=p->scales_n32+
            ((size_t)(g*8+t)*nb+b)*32;memcpy(ds+t*32,ss,64);}
        for(int kk=0;kk<bs;++kk)for(int t=0;t<8;++t){const uint8_t*sq=p->codes_n32+
            ((size_t)(g*8+t)*p->k+b*bs+kk)*16;
          memcpy(dq+((size_t)kk*8+t)*16,sq,16);}
      }
    }
    return 0;
}

int fp4_matrix_prepare_half(fp4_matrix*p){
    if(!p||!p->codes||!p->codes_n32||!p->scales_t8||p->n%256||p->k%32)return-1;
    free(p->codes_t8_half);p->codes_t8_half=NULL;
    if(posix_memalign((void**)&p->codes_t8_half,256,p->code_bytes))return-1;
    int nb=p->k/32;
    for(int g=0;g<p->n/256;++g)for(int b=0;b<nb;++b)for(int kk=0;kk<32;++kk)
      for(int t=0;t<8;++t){uint8_t*hd=p->codes_t8_half+
        ((((size_t)g*nb+b)*32+kk)*8+t)*16;int k=b*32+kk;
        for(int j=0;j<16;++j){int r0=(g*8+t)*32+j,r1=r0+16;
          uint8_t a=p->codes[(size_t)r0*(p->k/2)+k/2];
          uint8_t c=p->codes[(size_t)r1*(p->k/2)+k/2];
          hd[j]=(uint8_t)(((a>>((k&1)*4))&15)|(((c>>((k&1)*4))&15)<<4));}}
    return 0;
}

int fp4_matrix_prepare_affine(fp4_matrix*p){
    static const uint8_t mr[8]={1,0,2,3,4,5,6,7};
    if(!p||!p->codes_t8||p->n%256||p->k%32)return-1;
    free(p->codes_t8_affine);p->codes_t8_affine=NULL;
    if(posix_memalign((void**)&p->codes_t8_affine,256,p->code_bytes))return-1;
    for(size_t i=0;i<p->code_bytes;++i){uint8_t lo=p->codes_t8[i]&15,hi=p->codes_t8[i]>>4;
      uint8_t al=(uint8_t)((mr[lo&7]<<1)|(lo>>3));
      uint8_t ah=(uint8_t)((mr[hi&7]<<1)|(hi>>3));p->codes_t8_affine[i]=(uint8_t)(al|(ah<<4));}
    return 0;
}

int fp4_matrix_prepare_t12(fp4_matrix*p){
    if(!p||!p->codes_n32||!p->scales_n32||p->n%384||p->k%32)return-1;
    free(p->codes_t12);free(p->scales_t12);p->codes_t12=NULL;p->scales_t12=NULL;
    if(posix_memalign((void**)&p->codes_t12,256,p->code_bytes)||
       posix_memalign((void**)&p->scales_t12,256,p->scales_n32_count*sizeof(_Float16))){
      free(p->codes_t12);free(p->scales_t12);p->codes_t12=NULL;p->scales_t12=NULL;return-1;}
    int nb=p->k/32;
    for(int g=0;g<p->n/384;++g)for(int b=0;b<nb;++b){
      uint8_t*dq=p->codes_t12+(((size_t)g*nb+b)*32)*192;
      _Float16*ds=p->scales_t12+((size_t)g*nb+b)*384;
      for(int t=0;t<12;++t){const _Float16*ss=p->scales_n32+
          ((size_t)(g*12+t)*nb+b)*32;memcpy(ds+t*32,ss,64);}
      for(int kk=0;kk<32;++kk)for(int t=0;t<12;++t){const uint8_t*sq=p->codes_n32+
          ((size_t)(g*12+t)*p->k+b*32+kk)*16;
        memcpy(dq+((size_t)kk*12+t)*16,sq,16);}}
    return 0;
}

int fp4_matrix_prepare_u8(fp4_matrix*p){
    if(!p||!p->codes_n32||p->n%128||p->k%32)return-1;
    free(p->codes_u8);p->codes_u8=NULL;
    if(posix_memalign((void**)&p->codes_u8,256,(size_t)p->n*p->k))return-1;
    for(int g=0;g<p->n/32;g+=4)for(int k=0;k<p->k;++k)for(int i=0;i<4;++i){
      const uint8_t*q=p->codes_n32+((size_t)(g+i)*p->k+k)*16;
      uint8_t*d=p->codes_u8+(((size_t)(g/4)*p->k+k)*4+i)*32;
      for(int j=0;j<16;++j){d[2*j]=q[j]&15;d[2*j+1]=q[j]>>4;}
    }return 0;
}

int fp4_matrix_prepare_bitplane(fp4_matrix*p){
    if(!p||!p->codes_n32||p->n%32||p->k%32)return-1;
    free(p->codes_bitplane);p->codes_bitplane=NULL;
    if(posix_memalign((void**)&p->codes_bitplane,256,p->code_bytes))return-1;
    for(int t=0;t<p->n/32;++t)for(int k=0;k<p->k;++k){
      const uint8_t*q=p->codes_n32+((size_t)t*p->k+k)*16;
      uint32_t*d=p->codes_bitplane+((size_t)t*p->k+k)*4;
      d[0]=d[1]=d[2]=d[3]=0;
      for(int j=0;j<16;++j)for(int b=0;b<4;++b){
        d[b]|=(uint32_t)((q[j]>>b)&1)<<j;
        d[b]|=(uint32_t)((q[j]>>(b+4))&1)<<(16+j);
      }
    }return 0;
}

int fp4_matrix_prepare_sdot(fp4_matrix*p){
    if(!p||!p->codes_u8||!p->scales_n32||p->n%128||p->k%32)return-1;
    int bs=p->format==FP4_MX?32:16,nb=p->k/bs;
    free(p->codes_sdot);free(p->scales_sdot);p->codes_sdot=NULL;p->scales_sdot=NULL;
    if(posix_memalign((void**)&p->codes_sdot,256,(size_t)p->n*p->k)||
       posix_memalign((void**)&p->scales_sdot,256,
                      (size_t)(p->n/128)*nb*128*sizeof(_Float16))){
        free(p->codes_sdot);free(p->scales_sdot);p->codes_sdot=NULL;p->scales_sdot=NULL;return-1;
    }
    static const int8_t values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    for(int g=0;g<p->n/128;++g)for(int b=0;b<nb;++b){
      for(int pg=0;pg<bs/4;++pg)for(int v=0;v<8;++v)for(int lane=0;lane<16;++lane)
        for(int j=0;j<4;++j){int k=b*bs+pg*4+j,nlane=v*16+lane;
          size_t src=((size_t)g*p->k+k)*128+nlane;
          size_t dst=((((size_t)g*nb+b)*(bs/4)+pg)*8+v)*64+lane*4+j;
          p->codes_sdot[dst]=values[p->codes_u8[src]&15];}
      _Float16*d=p->scales_sdot+((size_t)g*nb+b)*128;
      for(int t=0;t<4;++t){const _Float16*s=p->scales_n32+
          ((size_t)(g*4+t)*nb+b)*32;memcpy(d+t*32,s,32*sizeof(_Float16));}
    }return 0;
}

int fp4_matrix_prepare_pair(fp4_matrix*p){
    if(!p||!p->codes||!p->scales_n32||p->n%128||p->k%32)return-1;
    int bs=p->format==FP4_MX?32:16,nb=p->k/bs,ng=p->n/128,pairs=bs/2;
    free(p->codes_pair);free(p->scales_pair);p->codes_pair=NULL;p->scales_pair=NULL;
    if(posix_memalign((void**)&p->codes_pair,256,p->code_bytes)||
       posix_memalign((void**)&p->scales_pair,256,(size_t)nb*ng*128*sizeof(_Float16))){
        free(p->codes_pair);free(p->scales_pair);p->codes_pair=NULL;p->scales_pair=NULL;return-1;}
    for(int b=0;b<nb;++b)for(int g=0;g<ng;++g){
      uint8_t*d=p->codes_pair+(((size_t)b*ng+g)*pairs)*128;
      for(int pair=0;pair<pairs;++pair)for(int v=0;v<8;++v)for(int lane=0;lane<16;++lane){
        int r=g*128+v*16+lane,k=b*bs+pair*2;
        uint8_t lo=p->codes[(size_t)r*(p->k/2)+k/2]&15;
        uint8_t hi=p->codes[(size_t)r*(p->k/2)+(k+1)/2]>>4;
        d[((size_t)pair*8+v)*16+lane]=(uint8_t)(lo|(hi<<4));}
      _Float16*s=p->scales_pair+((size_t)b*ng+g)*128;
      for(int t=0;t<4;++t)memcpy(s+t*32,p->scales_n32+
        ((size_t)(g*4+t)*nb+b)*32,32*sizeof(_Float16));
    }return 0;
}

int fp4_matrix_prepare_sdot4(fp4_matrix*p){
    if(!p||!p->codes||!p->scales_pair||p->n%128||p->k%32)return-1;
    int bs=p->format==FP4_MX?32:16,nb=p->k/bs,ng=p->n/128;
    free(p->codes_sdot4);free(p->scales_sdot4);p->codes_sdot4=NULL;p->scales_sdot4=NULL;
    if(posix_memalign((void**)&p->codes_sdot4,256,p->code_bytes)||
       posix_memalign((void**)&p->scales_sdot4,256,(size_t)nb*ng*128*sizeof(float))){
      free(p->codes_sdot4);free(p->scales_sdot4);p->codes_sdot4=NULL;p->scales_sdot4=NULL;return-1;}
    for(size_t i=0;i<(size_t)nb*ng*128;++i)p->scales_sdot4[i]=(float)p->scales_pair[i];
    for(int b=0;b<nb;++b)for(int g=0;g<ng;++g)for(int pg=0;pg<bs/4;++pg)
      for(int chunk=0;chunk<4;++chunk)for(int row=0;row<32;++row){
        int r=g*128+chunk*32+row,k=b*bs+pg*4;uint8_t q[4];
        for(int j=0;j<4;++j){uint8_t x=p->codes[(size_t)r*(p->k/2)+(k+j)/2];
          q[j]=(x>>(((k+j)&1)*4))&15;}
        size_t d=(((((size_t)b*ng+g)*(bs/4)+pg)*4+chunk)*64)+(size_t)row*2;
        p->codes_sdot4[d]=(uint8_t)(q[0]|(q[1]<<4));
        p->codes_sdot4[d+1]=(uint8_t)(q[2]|(q[3]<<4));
      }
    return 0;
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

#if defined(__ARM_FEATURE_SVE)
static inline void gemm_n32_m6_full_segment(float*c,const _Float16*a,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    static const __fp16 table_data[32] __attribute__((aligned(64)))={
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6,
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
    svbool_t ph=svptrue_b16(),ps=svptrue_b32(),p16=svwhilelt_b16(0,16);
    svfloat16_t tab=svld1_f16(ph,table_data),scale=svdup_f16(0);
    svfloat16_t h0=svdup_f16(0),h1=h0,h2=h0,h3=h0,h4=h0,h5=h0;
    int bs=w->format==FP4_MX?32:16,nb=w->k/bs;
    const uint8_t*cp=w->codes_n32+(size_t)tile*w->k*16;
    const _Float16*sp=w->scales_n32+(size_t)tile*nb*32;
    for(int b=kbegin/bs;b<kend/bs;++b){
      scale=svld1_f16(ph,(const __fp16*)(sp+(size_t)b*32));
      for(int k=b*bs;k<(b+1)*bs;++k){
        const uint8_t*q=cp+(size_t)k*16;
        svuint16_t z=svld1ub_u16(p16,q),lo=svand_n_u16_x(p16,z,15);
        svuint16_t hi=svlsr_n_u16_x(p16,z,4),idx=svzip1_u16(lo,hi);
        svfloat16_t weight=svmul_f16_x(ph,svtbl_f16(tab,idx),scale);
        h0=svmla_n_f16_x(ph,h0,weight,(__fp16)a[k]);
        h1=svmla_n_f16_x(ph,h1,weight,(__fp16)a[(size_t)w->k+k]);
        h2=svmla_n_f16_x(ph,h2,weight,(__fp16)a[(size_t)2*w->k+k]);
        h3=svmla_n_f16_x(ph,h3,weight,(__fp16)a[(size_t)3*w->k+k]);
        h4=svmla_n_f16_x(ph,h4,weight,(__fp16)a[(size_t)4*w->k+k]);
        h5=svmla_n_f16_x(ph,h5,weight,(__fp16)a[(size_t)5*w->k+k]);
      }
    }
#define N32_STORE6(I,H) do {svuint16_t hb=svreinterpret_u16_f16(H); \
        svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb))); \
        svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb))); \
        float*d=c+(size_t)(I)*w->n+tile*32; \
        if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d));hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));} \
        svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}while(0)
    N32_STORE6(0,h0);N32_STORE6(1,h1);N32_STORE6(2,h2);
    N32_STORE6(3,h3);N32_STORE6(4,h4);N32_STORE6(5,h5);
#undef N32_STORE6
}

static inline void gemm_n32_m12_segment(float*c,const _Float16*a,int m,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    static const __fp16 table_data[32] __attribute__((aligned(64)))={
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6,
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
    svbool_t ph=svptrue_b16(),ps=svptrue_b32(),p16=svwhilelt_b16(0,16);
    svfloat16_t tab=svld1_f16(ph,table_data),scale=svdup_f16(0);
    svfloat16_t h0=svdup_f16(0),h1=h0,h2=h0,h3=h0,h4=h0,h5=h0;
    svfloat16_t h6=h0,h7=h0,h8=h0,h9=h0,h10=h0,h11=h0;
    int bs=w->format==FP4_MX?32:16,nb=w->k/bs;
    const uint8_t*cp=w->codes_n32+(size_t)tile*w->k*16;
    const _Float16*sp=w->scales_n32+(size_t)tile*nb*32;
    if(m==12){
      for(int b=kbegin/bs;b<kend/bs;++b){
       scale=svld1_f16(ph,(const __fp16*)(sp+(size_t)b*32));
       for(int k=b*bs;k<(b+1)*bs;++k){
        const uint8_t*q=cp+(size_t)k*16;
        svuint16_t z=svld1ub_u16(p16,q),lo=svand_n_u16_x(p16,z,15);
        svuint16_t hi=svlsr_n_u16_x(p16,z,4),idx=svzip1_u16(lo,hi);
        svfloat16_t weight=svmul_f16_x(ph,svtbl_f16(tab,idx),scale);
        h0=svmla_n_f16_x(ph,h0,weight,(__fp16)a[k]);
        h1=svmla_n_f16_x(ph,h1,weight,(__fp16)a[(size_t)w->k+k]);
        h2=svmla_n_f16_x(ph,h2,weight,(__fp16)a[(size_t)2*w->k+k]);
        h3=svmla_n_f16_x(ph,h3,weight,(__fp16)a[(size_t)3*w->k+k]);
        h4=svmla_n_f16_x(ph,h4,weight,(__fp16)a[(size_t)4*w->k+k]);
        h5=svmla_n_f16_x(ph,h5,weight,(__fp16)a[(size_t)5*w->k+k]);
        h6=svmla_n_f16_x(ph,h6,weight,(__fp16)a[(size_t)6*w->k+k]);
        h7=svmla_n_f16_x(ph,h7,weight,(__fp16)a[(size_t)7*w->k+k]);
        h8=svmla_n_f16_x(ph,h8,weight,(__fp16)a[(size_t)8*w->k+k]);
        h9=svmla_n_f16_x(ph,h9,weight,(__fp16)a[(size_t)9*w->k+k]);
        h10=svmla_n_f16_x(ph,h10,weight,(__fp16)a[(size_t)10*w->k+k]);
        h11=svmla_n_f16_x(ph,h11,weight,(__fp16)a[(size_t)11*w->k+k]);
      }}
    }else for(int k=kbegin;k<kend;++k){
        if(k%bs==0)scale=svld1_f16(ph,(const __fp16*)(sp+(size_t)(k/bs)*32));
        const uint8_t*q=cp+(size_t)k*16;
        svuint16_t z=svld1ub_u16(p16,q),lo=svand_n_u16_x(p16,z,15);
        svuint16_t hi=svlsr_n_u16_x(p16,z,4),idx=svzip1_u16(lo,hi);
        svfloat16_t weight=svmul_f16_x(ph,svtbl_f16(tab,idx),scale);
#define N32_FMA12(I,H) do {if(m>(I))(H)=svmla_n_f16_x(ph,(H),weight,(__fp16)a[(size_t)(I)*w->k+k]);}while(0)
        N32_FMA12(0,h0);N32_FMA12(1,h1);N32_FMA12(2,h2);N32_FMA12(3,h3);
        N32_FMA12(4,h4);N32_FMA12(5,h5);N32_FMA12(6,h6);N32_FMA12(7,h7);
        N32_FMA12(8,h8);N32_FMA12(9,h9);N32_FMA12(10,h10);N32_FMA12(11,h11);
#undef N32_FMA12
    }
#define N32_STORE12(I,H) do {if(m>(I)){ \
        svuint16_t hb=svreinterpret_u16_f16(H); \
        svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb))); \
        svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb))); \
        float*d=c+(size_t)(I)*w->n+tile*32; \
        if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d));hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));} \
        svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}}while(0)
    N32_STORE12(0,h0);N32_STORE12(1,h1);N32_STORE12(2,h2);N32_STORE12(3,h3);
    N32_STORE12(4,h4);N32_STORE12(5,h5);N32_STORE12(6,h6);N32_STORE12(7,h7);
    N32_STORE12(8,h8);N32_STORE12(9,h9);N32_STORE12(10,h10);N32_STORE12(11,h11);
#undef N32_STORE12
}

#endif

int fp4_gemm_f16_n32(float*c,const _Float16*a,const fp4_matrix*w,int m,int promotion_k){
    if(!c||!a||!w||!w->codes_n32||!w->scales_n32||m<1||promotion_k<0||
       (promotion_k&&((promotion_k%32)||promotion_k>w->k)))return-1;
    int span=promotion_k?promotion_k:w->k;
    for(int m0=0;m0<m;m0+=6){int mr=m-m0<6?m-m0:6;
        for(int kb=0;kb<w->k;kb+=span){int ke=kb+span<w->k?kb+span:w->k;
          for(int t=0;t<w->n/32;++t){
#if defined(__ARM_FEATURE_SVE)
            if(mr==6)gemm_n32_m6_full_segment(c+(size_t)m0*w->n,a+(size_t)m0*w->k,w,t,kb,ke,kb!=0);
            else gemm_n32_m12_segment(c+(size_t)m0*w->n,a+(size_t)m0*w->k,mr,w,t,kb,ke,kb!=0);
#else
            (void)t;return-1;
#endif
          }
        }}return 0;
}

#if defined(__ARM_FEATURE_SVE)
static inline void gemm_n32_m1_t4_segment(float*c,const _Float16*a,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    static const __fp16 table_data[32] __attribute__((aligned(64)))={
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6,
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
    svbool_t ph=svptrue_b16(),ps=svptrue_b32(),p16=svwhilelt_b16(0,16);
    svfloat16_t tab=svld1_f16(ph,table_data),h0=svdup_f16(0),h1=h0,h2=h0,h3=h0;
    svuint16_t nibble_mask=svdup_u16(15);
    int bs=w->format==FP4_MX?32:16,nb=w->k/bs;
    const uint8_t*q0=w->codes_n32+(size_t)(tile+0)*w->k*16;
    const uint8_t*q1=w->codes_n32+(size_t)(tile+1)*w->k*16;
    const uint8_t*q2=w->codes_n32+(size_t)(tile+2)*w->k*16;
    const uint8_t*q3=w->codes_n32+(size_t)(tile+3)*w->k*16;
    const _Float16*s0=w->scales_n32+(size_t)(tile+0)*nb*32;
    const _Float16*s1=w->scales_n32+(size_t)(tile+1)*nb*32;
    const _Float16*s2=w->scales_n32+(size_t)(tile+2)*nb*32;
    const _Float16*s3=w->scales_n32+(size_t)(tile+3)*nb*32;
    for(int b=kbegin/bs;b<kend/bs;++b){svfloat16_t sc0=svld1_f16(ph,(const __fp16*)(s0+(size_t)b*32));
      svfloat16_t sc1=svld1_f16(ph,(const __fp16*)(s1+(size_t)b*32));
      svfloat16_t sc2=svld1_f16(ph,(const __fp16*)(s2+(size_t)b*32));
      svfloat16_t sc3=svld1_f16(ph,(const __fp16*)(s3+(size_t)b*32));
      for(int k=b*bs;k<(b+1)*bs;++k){svfloat16_t x=svdup_f16((__fp16)a[k]);
#define DECODE_T4(Q,SC,H) do{const uint8_t*q=(Q)+(size_t)k*16; \
        svuint16_t z=svld1ub_u16(p16,q),lo=svand_u16_x(p16,z,nibble_mask); \
        svuint16_t hi=svlsr_n_u16_x(p16,z,4),idx=svzip1_u16(lo,hi); \
        svfloat16_t v=svmul_f16_x(ph,svtbl_f16(tab,idx),(SC)); \
        (H)=svmla_f16_x(ph,(H),v,x);}while(0)
        DECODE_T4(q0,sc0,h0);DECODE_T4(q1,sc1,h1);
        DECODE_T4(q2,sc2,h2);DECODE_T4(q3,sc3,h3);
#undef DECODE_T4
      }
    }
#define STORE_T4(I,H) do{svuint16_t hb=svreinterpret_u16_f16(H); \
      svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb))); \
      svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb))); \
      float*d=c+(tile+(I))*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d)); \
      hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}while(0)
    STORE_T4(0,h0);STORE_T4(1,h1);STORE_T4(2,h2);STORE_T4(3,h3);
#undef STORE_T4
}

static inline void gemm_u8tbl_m1_t4_segment(float*c,const _Float16*a,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    static const __fp16 table_data[32] __attribute__((aligned(64)))={
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6,
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
    svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    svfloat16_t tab=svld1_f16(ph,table_data),h0=svdup_f16(0),h1=h0,h2=h0,h3=h0;
    int bs=w->format==FP4_MX?32:16,nb=w->k/bs;
    const uint8_t*q0=w->codes_u8+(size_t)(tile/4)*w->k*128+0*32;
    const uint8_t*q1=w->codes_u8+(size_t)(tile/4)*w->k*128+1*32;
    const uint8_t*q2=w->codes_u8+(size_t)(tile/4)*w->k*128+2*32;
    const uint8_t*q3=w->codes_u8+(size_t)(tile/4)*w->k*128+3*32;
    const _Float16*s0=w->scales_n32+(size_t)(tile+0)*nb*32;
    const _Float16*s1=w->scales_n32+(size_t)(tile+1)*nb*32;
    const _Float16*s2=w->scales_n32+(size_t)(tile+2)*nb*32;
    const _Float16*s3=w->scales_n32+(size_t)(tile+3)*nb*32;
    for(int b=kbegin/bs;b<kend/bs;++b){svfloat16_t sc0=svld1_f16(ph,(const __fp16*)(s0+(size_t)b*32));
      svfloat16_t sc1=svld1_f16(ph,(const __fp16*)(s1+(size_t)b*32));
      svfloat16_t sc2=svld1_f16(ph,(const __fp16*)(s2+(size_t)b*32));
      svfloat16_t sc3=svld1_f16(ph,(const __fp16*)(s3+(size_t)b*32));
      for(int k=b*bs;k<(b+1)*bs;++k){svfloat16_t x=svdup_f16((__fp16)a[k]);
#define DECODE_U8(Q,SC,H) do{svuint16_t q=svld1ub_u16(ph,(Q)+(size_t)k*128); \
        svfloat16_t v=svmul_f16_x(ph,svtbl_f16(tab,q),(SC)); \
        (H)=svmla_f16_x(ph,(H),v,x);}while(0)
        DECODE_U8(q0,sc0,h0);DECODE_U8(q1,sc1,h1);
        DECODE_U8(q2,sc2,h2);DECODE_U8(q3,sc3,h3);
#undef DECODE_U8
      }
    }
#define STORE_U8(I,H) do{svuint16_t hb=svreinterpret_u16_f16(H); \
      svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb))); \
      svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb))); \
      float*d=c+(tile+(I))*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d)); \
      hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}while(0)
    STORE_U8(0,h0);STORE_U8(1,h1);STORE_U8(2,h2);STORE_U8(3,h3);
#undef STORE_U8
}

static inline svfloat16_t decode_bitplane(const uint32_t*q,svuint16_t shifts,
        svuint16_t one){
    svbool_t ph=svptrue_b16();
#define BP_BIT(I) svand_u16_x(ph,svlsr_u16_x(ph,svreinterpret_u16_u32( \
        svdup_u32(q[(I)])),shifts),one)
    svuint16_t b0=BP_BIT(0),b1=BP_BIT(1),b2=BP_BIT(2),sgn=BP_BIT(3);
#undef BP_BIT
    svuint16_t upper=svorr_u16_x(ph,b1,b2),nz=svorr_u16_x(ph,b0,upper);
    svuint16_t bits=svmul_n_u16_x(ph,nz,0x3800);
    bits=svadd_u16_x(ph,bits,svlsl_n_u16_x(ph,b1,10));
    bits=svadd_u16_x(ph,bits,svlsl_n_u16_x(ph,b2,11));
    bits=svadd_u16_x(ph,bits,svlsl_n_u16_x(ph,svand_u16_x(ph,b0,upper),9));
    bits=svorr_u16_x(ph,bits,svlsl_n_u16_x(ph,sgn,15));
    return svreinterpret_f16_u16(bits);
}

static inline void gemm_bitplane_m1_t4_segment(float*c,const _Float16*a,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    static const uint16_t shift_data[32] __attribute__((aligned(64)))={
        0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,
        8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15};
    svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    svuint16_t shifts=svld1_u16(ph,shift_data),one=svdup_u16(1);
    svfloat16_t h0=svdup_f16(0),h1=h0,h2=h0,h3=h0;
    int bs=w->format==FP4_MX?32:16,nb=w->k/bs;
    const uint32_t*q0=w->codes_bitplane+(size_t)(tile+0)*w->k*4;
    const uint32_t*q1=w->codes_bitplane+(size_t)(tile+1)*w->k*4;
    const uint32_t*q2=w->codes_bitplane+(size_t)(tile+2)*w->k*4;
    const uint32_t*q3=w->codes_bitplane+(size_t)(tile+3)*w->k*4;
    const _Float16*s0=w->scales_n32+(size_t)(tile+0)*nb*32;
    const _Float16*s1=w->scales_n32+(size_t)(tile+1)*nb*32;
    const _Float16*s2=w->scales_n32+(size_t)(tile+2)*nb*32;
    const _Float16*s3=w->scales_n32+(size_t)(tile+3)*nb*32;
    for(int b=kbegin/bs;b<kend/bs;++b){
      svfloat16_t sc0=svld1_f16(ph,(const __fp16*)(s0+(size_t)b*32));
      svfloat16_t sc1=svld1_f16(ph,(const __fp16*)(s1+(size_t)b*32));
      svfloat16_t sc2=svld1_f16(ph,(const __fp16*)(s2+(size_t)b*32));
      svfloat16_t sc3=svld1_f16(ph,(const __fp16*)(s3+(size_t)b*32));
      for(int k=b*bs;k<(b+1)*bs;++k){svfloat16_t x=svdup_f16((__fp16)a[k]);
#define BP_FMA(Q,SC,H) do{svfloat16_t v=svmul_f16_x(ph,decode_bitplane( \
          (Q)+(size_t)k*4,shifts,one),(SC));(H)=svmla_f16_x(ph,(H),v,x);}while(0)
        BP_FMA(q0,sc0,h0);BP_FMA(q1,sc1,h1);
        BP_FMA(q2,sc2,h2);BP_FMA(q3,sc3,h3);
#undef BP_FMA
      }
    }
#define BP_STORE(I,H) do{svuint16_t hb=svreinterpret_u16_f16(H); \
      svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb))); \
      svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb))); \
      float*d=c+(tile+(I))*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d)); \
      hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo); \
      svst1_f32(ps,d+16,hi);}while(0)
    BP_STORE(0,h0);BP_STORE(1,h1);BP_STORE(2,h2);BP_STORE(3,h3);
#undef BP_STORE
}

typedef struct {
    const uint8_t*q[4];const _Float16*s[4];const _Float16*a;_Float16*out;
    int k_count,bs;
} fp4_m1_t4_args;
extern void fp4_mx_m1_t4_asm(const fp4_m1_t4_args*);
extern void fp4_mx_u8_m1_t4_asm(const fp4_m1_t4_args*);
typedef struct {const uint8_t*q;const _Float16*s,*a;_Float16*out;int k_count;} fp4_m1_t8_args;
extern void fp4_mx_m1_t8_asm(const fp4_m1_t8_args*);
extern void fp4_mx_m1_t8_half_asm(const fp4_m1_t8_args*);
extern void fp4_mx_m1_t12_asm(const fp4_m1_t8_args*);
extern void fp4_mx_m1_t8_affine_asm(const fp4_m1_t8_args*);

static inline void gemm_mx_m1_t8_asm_segment(float*c,const _Float16*tables,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    _Float16 tmp[8*32] __attribute__((aligned(256)));int nb=w->k/32,g=tile/8,b=kbegin/32;
    fp4_m1_t8_args x={w->codes_t8+(((size_t)g*nb+b)*32)*128,
      w->scales_t8+((size_t)g*nb+b)*256,tables+(size_t)kbegin*32,tmp,kend-kbegin};
    fp4_mx_m1_t8_asm(&x);svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    for(int i=0;i<8;++i){svuint16_t hb=svreinterpret_u16_f16(
        svld1_f16(ph,(const __fp16*)(tmp+i*32)));
      svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb)));
      svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb)));
      float*d=c+(tile+i)*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d));
        hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}
}

static inline void gemm_mx_m1_t8_half_segment(float*c,const _Float16*tables,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    _Float16 tmp[8*32] __attribute__((aligned(256)));int nb=w->k/32,g=tile/8,b=kbegin/32;
    fp4_m1_t8_args x={w->codes_t8_half+(((size_t)g*nb+b)*32)*128,
      w->scales_t8+((size_t)g*nb+b)*256,tables+(size_t)kbegin*32,tmp,kend-kbegin};
    fp4_mx_m1_t8_half_asm(&x);svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    for(int i=0;i<8;++i){svuint16_t hb=svreinterpret_u16_f16(
        svld1_f16(ph,(const __fp16*)(tmp+i*32)));
      svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb)));
      svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb)));
      float*d=c+(tile+i)*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d));
        hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}
}

static inline void gemm_mx_m1_t12_segment(float*c,const _Float16*tables,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    _Float16 tmp[12*32] __attribute__((aligned(256)));int nb=w->k/32,g=tile/12,b=kbegin/32;
    fp4_m1_t8_args x={w->codes_t12+(((size_t)g*nb+b)*32)*192,
      w->scales_t12+((size_t)g*nb+b)*384,tables+(size_t)kbegin*32,tmp,kend-kbegin};
    fp4_mx_m1_t12_asm(&x);svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    for(int i=0;i<12;++i){svuint16_t hb=svreinterpret_u16_f16(
        svld1_f16(ph,(const __fp16*)(tmp+i*32)));
      svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb)));
      svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb)));
      float*d=c+(tile+i)*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d));
        hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}
}

static inline void gemm_mx_m1_t8_affine_segment(float*c,const _Float16*a,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    _Float16 tmp[8*32] __attribute__((aligned(256)));int nb=w->k/32,g=tile/8,b=kbegin/32;
    fp4_m1_t8_args x={w->codes_t8_affine+(((size_t)g*nb+b)*32)*128,
      w->scales_t8+((size_t)g*nb+b)*256,a+kbegin,tmp,kend-kbegin};
    fp4_mx_m1_t8_affine_asm(&x);svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    for(int i=0;i<8;++i){svuint16_t hb=svreinterpret_u16_f16(
        svld1_f16(ph,(const __fp16*)(tmp+i*32)));
      svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb)));
      svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb)));
      float*d=c+(tile+i)*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d));
        hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}
}

static inline void gemm_mx_m1_t4_asm_segment(float*c,const _Float16*a,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    _Float16 tmp[4*32] __attribute__((aligned(256)));
    int nb=w->k/32;fp4_m1_t4_args x;
    for(int i=0;i<4;++i){x.q[i]=w->codes_n32+(size_t)(tile+i)*w->k*16+(size_t)kbegin*16;
        x.s[i]=w->scales_n32+(size_t)(tile+i)*nb*32+(size_t)(kbegin/32)*32;}
    x.a=a+kbegin;x.out=tmp;x.k_count=kend-kbegin;x.bs=32;
    fp4_mx_m1_t4_asm(&x);
    svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    for(int i=0;i<4;++i){svuint16_t hb=svreinterpret_u16_f16(svld1_f16(ph,(const __fp16*)(tmp+i*32)));
      svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb)));
      svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb)));
      float*d=c+(tile+i)*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d));
        hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}
}

static inline void gemm_mx_u8_m1_t4_asm_segment(float*c,const _Float16*a,
        const fp4_matrix*w,int tile,int kbegin,int kend,int add){
    _Float16 tmp[4*32] __attribute__((aligned(256)));
    int nb=w->k/32;fp4_m1_t4_args x;
    for(int i=0;i<4;++i){x.q[i]=w->codes_u8+(size_t)(tile/4)*w->k*128+(size_t)kbegin*128+i*32;
        x.s[i]=w->scales_n32+(size_t)(tile+i)*nb*32+(size_t)(kbegin/32)*32;}
    x.a=a+kbegin;x.out=tmp;x.k_count=kend-kbegin;x.bs=32;
    fp4_mx_u8_m1_t4_asm(&x);
    svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    for(int i=0;i<4;++i){svuint16_t hb=svreinterpret_u16_f16(svld1_f16(ph,(const __fp16*)(tmp+i*32)));
      svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb)));
      svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb)));
      float*d=c+(tile+i)*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d));
        hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}
}
#endif

int fp4_gemm_f16_n32_omp(float*c,const _Float16*a,const fp4_matrix*w,int m,
        int promotion_k,int threads){
    if(!c||!a||!w||!w->codes_n32||!w->scales_n32||m<1||threads<1||promotion_k<0||
       (promotion_k&&((promotion_k%32)||promotion_k>w->k)))return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    int span=promotion_k?promotion_k:w->k;
    if(m==1&&w->format==FP4_MX&&w->codes_t8&&w->scales_t8&&w->n%256==0){
      static const __fp16 td[32] __attribute__((aligned(64)))={
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6,
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
      _Float16*tables=NULL;if(posix_memalign((void**)&tables,256,(size_t)w->k*64))return-1;
      svbool_t ph=svptrue_b16();svfloat16_t tab=svld1_f16(ph,td);
      for(int k=0;k<w->k;++k)svst1_f16(ph,(__fp16*)(tables+(size_t)k*32),
          svmul_n_f16_x(ph,tab,(__fp16)a[k]));
#pragma omp parallel for num_threads(threads) schedule(static)
      for(int t=0;t<w->n/32;t+=8)for(int kb=0;kb<w->k;kb+=span){
        int ke=kb+span<w->k?kb+span:w->k;
        gemm_mx_m1_t8_asm_segment(c,tables,w,t,kb,ke,kb!=0);
      }free(tables);return 0;
    }
    if(m==1&&w->n%128==0){
#pragma omp parallel for num_threads(threads) schedule(static)
      for(int t=0;t<w->n/32;t+=4)for(int kb=0;kb<w->k;kb+=span){
        int ke=kb+span<w->k?kb+span:w->k;
        if(w->format==FP4_MX)gemm_mx_m1_t4_asm_segment(c,a,w,t,kb,ke,kb!=0);
        else gemm_n32_m1_t4_segment(c,a,w,t,kb,ke,kb!=0);
      }
      return 0;
    }
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int t=0;t<w->n/32;++t){
      for(int m0=0;m0<m;m0+=6){int mr=m-m0<6?m-m0:6;
        for(int kb=0;kb<w->k;kb+=span){int ke=kb+span<w->k?kb+span:w->k;
          if(mr==6)gemm_n32_m6_full_segment(c+(size_t)m0*w->n,a+(size_t)m0*w->k,w,t,kb,ke,kb!=0);
          else gemm_n32_m12_segment(c+(size_t)m0*w->n,a+(size_t)m0*w->k,mr,w,t,kb,ke,kb!=0);
        }
      }
    }return 0;
#else
    (void)threads;return-1;
#endif
}

int fp4_gemm_f16_half_omp(float*c,const _Float16*a,const fp4_matrix*w,int m,
        int promotion_k,int threads){
    if(!c||!a||!w||!w->codes_t8_half||!w->scales_t8||m!=1||threads<1||
       w->format!=FP4_MX||w->n%256||promotion_k<=0||promotion_k%32||
       promotion_k>w->k)return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    static const __fp16 td[32] __attribute__((aligned(64)))={
      0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6,
      0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
    _Float16*tables=NULL;if(posix_memalign((void**)&tables,256,(size_t)w->k*64))return-1;
    svbool_t ph=svptrue_b16();svfloat16_t tab=svld1_f16(ph,td);
    for(int k=0;k<w->k;++k)svst1_f16(ph,(__fp16*)(tables+(size_t)k*32),
        svmul_n_f16_x(ph,tab,(__fp16)a[k]));
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int t=0;t<w->n/32;t+=8)for(int kb=0;kb<w->k;kb+=promotion_k){
      int ke=kb+promotion_k<w->k?kb+promotion_k:w->k;
      gemm_mx_m1_t8_half_segment(c,tables,w,t,kb,ke,kb!=0);
    }
    free(tables);return 0;
#else
    return-1;
#endif
}

int fp4_gemm_f16_t12_omp(float*c,const _Float16*a,const fp4_matrix*w,int m,
        int promotion_k,int threads){
    if(!c||!a||!w||!w->codes_t12||!w->scales_t12||m!=1||threads<1||
       w->format!=FP4_MX||w->n%384||promotion_k<=0||promotion_k%32||
       promotion_k>w->k)return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    static const __fp16 td[32] __attribute__((aligned(64)))={
      0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6,
      0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
    _Float16*tables=NULL;if(posix_memalign((void**)&tables,256,(size_t)w->k*64))return-1;
    svbool_t ph=svptrue_b16();svfloat16_t tab=svld1_f16(ph,td);
    for(int k=0;k<w->k;++k)svst1_f16(ph,(__fp16*)(tables+(size_t)k*32),
        svmul_n_f16_x(ph,tab,(__fp16)a[k]));
#pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for(int t=0;t<w->n/32;t+=12)for(int kb=0;kb<w->k;kb+=promotion_k){
      int ke=kb+promotion_k<w->k?kb+promotion_k:w->k;
      gemm_mx_m1_t12_segment(c,tables,w,t,kb,ke,kb!=0);
    }
    free(tables);return 0;
#else
    return-1;
#endif
}

int fp4_gemm_f16_affine_omp(float*c,const _Float16*a,const fp4_matrix*w,int m,
        int promotion_k,int threads){
    if(!c||!a||!w||!w->codes_t8_affine||!w->scales_t8||m!=1||threads<1||
       w->format!=FP4_MX||w->n%256||promotion_k<=0||promotion_k%32||
       promotion_k>w->k)return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int t=0;t<w->n/32;t+=8)for(int kb=0;kb<w->k;kb+=promotion_k){
      int ke=kb+promotion_k<w->k?kb+promotion_k:w->k;
      gemm_mx_m1_t8_affine_segment(c,a,w,t,kb,ke,kb!=0);
    }
    return 0;
#else
    return-1;
#endif
}

int fp4_gemm_f16_u8tbl_omp(float*c,const _Float16*a,const fp4_matrix*w,int m,
        int promotion_k,int threads){
    if(!c||!a||!w||!w->codes_u8||!w->scales_n32||m!=1||threads<1||w->n%128||
       promotion_k<0||(promotion_k&&((promotion_k%32)||promotion_k>w->k)))return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    int span=promotion_k?promotion_k:w->k;
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int t=0;t<w->n/32;t+=4)for(int kb=0;kb<w->k;kb+=span){
      int ke=kb+span<w->k?kb+span:w->k;
      if(w->format==FP4_MX)gemm_mx_u8_m1_t4_asm_segment(c,a,w,t,kb,ke,kb!=0);
      else gemm_u8tbl_m1_t4_segment(c,a,w,t,kb,ke,kb!=0);
    }
    return 0;
#else
    (void)threads;return-1;
#endif
}

int fp4_gemm_f16_bitplane_omp(float*c,const _Float16*a,const fp4_matrix*w,int m,
        int promotion_k,int threads){
    if(!c||!a||!w||!w->codes_bitplane||!w->scales_n32||m!=1||threads<1||
       w->n%128||promotion_k<0||(promotion_k&&((promotion_k%32)||promotion_k>w->k)))return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    int span=promotion_k?promotion_k:w->k;
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int t=0;t<w->n/32;t+=4)for(int kb=0;kb<w->k;kb+=span){
      int ke=kb+span<w->k?kb+span:w->k;
      gemm_bitplane_m1_t4_segment(c,a,w,t,kb,ke,kb!=0);
    }
    return 0;
#else
    (void)threads;return-1;
#endif
}

#if defined(__aarch64__)
typedef struct {
    const uint8_t *cp;
    const _Float16 *sp;
    const _Float16 *a;
    _Float16 *out;
    int k, kbegin, kend, bs;
} fp4_l1_args;
extern void fp4_n32_m6_l1_asm(const fp4_l1_args *args);

static inline void l1_store6(float*c,const _Float16*tmp,const fp4_matrix*w,
        int m0,int tile,int add){
#if defined(__ARM_FEATURE_SVE)
    svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    for(int i=0;i<6;++i){
        svuint16_t hb=svreinterpret_u16_f16(svld1_f16(ph,(const __fp16*)(tmp+i*32)));
        svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb)));
        svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb)));
        float*d=c+(size_t)(m0+i)*w->n+tile*32;
        if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d));hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}
        svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);
    }
#else
    (void)c;(void)tmp;(void)w;(void)m0;(void)tile;(void)add;
#endif
}
#endif

int fp4_gemm_f16_l1(float*c,const _Float16*a,const fp4_matrix*w,int m,int promotion_k){
    if(!c||!a||!w||!w->codes_n32||!w->scales_n32||m<1||m%6||promotion_k<0||
       (promotion_k&&((promotion_k%32)||promotion_k>w->k)))return-1;
#if defined(__aarch64__)
    int bs=w->format==FP4_MX?32:16,nb=w->k/bs,span=promotion_k?promotion_k:w->k;
    _Float16 tmp[6*32] __attribute__((aligned(256)));
    for(int m0=0;m0<m;m0+=6)for(int kb=0;kb<w->k;kb+=span){
      int ke=kb+span<w->k?kb+span:w->k;
      for(int t=0;t<w->n/32;++t){
        fp4_l1_args x={w->codes_n32+(size_t)t*w->k*16,
            w->scales_n32+(size_t)t*nb*32,a+(size_t)m0*w->k,tmp,
            w->k,kb,ke,bs};
        fp4_n32_m6_l1_asm(&x);
        l1_store6(c,tmp,w,m0,t,kb!=0);
      }
    }return 0;
#else
    return -1;
#endif
}

#if defined(__ARM_FEATURE_SVE)
extern void fp4_dequant_mx_n32_asm(_Float16*,const uint8_t*,const _Float16*,int);
static inline void dequant_n32_panel(_Float16*panel,const fp4_matrix*w,int tile,
        int kbegin,int kend){
    static const __fp16 table_data[32] __attribute__((aligned(64)))={
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6,
        0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
    svbool_t ph=svptrue_b16(),p16=svwhilelt_b16(0,16);
    svfloat16_t tab=svld1_f16(ph,table_data);
    int bs=w->format==FP4_MX?32:16,nb=w->k/bs;
    const uint8_t*cp=w->codes_n32+(size_t)tile*w->k*16;
    const _Float16*sp=w->scales_n32+(size_t)tile*nb*32;
    if(w->format==FP4_MX){fp4_dequant_mx_n32_asm(panel,cp+(size_t)kbegin*16,
        sp+(size_t)(kbegin/32)*32,kend-kbegin);return;}
    for(int b=kbegin/bs;b<kend/bs;++b){svfloat16_t scale=svld1_f16(ph,(const __fp16*)(sp+(size_t)b*32));
      for(int k=b*bs;k<(b+1)*bs;++k){const uint8_t*q=cp+(size_t)k*16;
        svuint16_t z=svld1ub_u16(p16,q),lo=svand_n_u16_x(p16,z,15);
        svuint16_t hi=svlsr_n_u16_x(p16,z,4),idx=svzip1_u16(lo,hi);
        svst1_f16(ph,(__fp16*)(panel+(size_t)(k-kbegin)*32),
            svmul_f16_x(ph,svtbl_f16(tab,idx),scale));
      }
    }
}

static inline void dense_panel_m6(float*c,const _Float16*a,const _Float16*p,
        int k,int n,int tile,int kbegin,int kend,int panel_k0,int rows,int add){
    svbool_t ph=svptrue_b16(),ps=svptrue_b32();
    svfloat16_t h0=svdup_f16(0),h1=h0,h2=h0,h3=h0,h4=h0,h5=h0;
    for(int x=kbegin;x<kend;++x){svfloat16_t wv=svld1_f16(ph,(const __fp16*)(p+(size_t)(x-panel_k0)*32));
#define PANEL_FMA(I,H) do{if(rows>(I))(H)=svmla_n_f16_x(ph,(H),wv,(__fp16)a[(size_t)(I)*k+x]);}while(0)
        PANEL_FMA(0,h0);PANEL_FMA(1,h1);PANEL_FMA(2,h2);
        PANEL_FMA(3,h3);PANEL_FMA(4,h4);PANEL_FMA(5,h5);
#undef PANEL_FMA
    }
#define PANEL_STORE(I,H) do{if(rows>(I)){svuint16_t hb=svreinterpret_u16_f16(H); \
        svfloat32_t lo=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpklo_u32(hb))); \
        svfloat32_t hi=svcvt_f32_f16_x(ps,svreinterpret_f16_u32(svunpkhi_u32(hb))); \
        float*d=c+(size_t)(I)*n+tile*32;if(add){lo=svadd_f32_x(ps,lo,svld1_f32(ps,d)); \
        hi=svadd_f32_x(ps,hi,svld1_f32(ps,d+16));}svst1_f32(ps,d,lo);svst1_f32(ps,d+16,hi);}}while(0)
    PANEL_STORE(0,h0);PANEL_STORE(1,h1);PANEL_STORE(2,h2);
    PANEL_STORE(3,h3);PANEL_STORE(4,h4);PANEL_STORE(5,h5);
#undef PANEL_STORE
}
#endif

int fp4_gemm_f16_l2(float*c,const _Float16*a,const fp4_matrix*w,int m,int promotion_k){
    if(!c||!a||!w||!w->codes_n32||!w->scales_n32||m<1||promotion_k<0||
       (promotion_k&&((promotion_k%32)||promotion_k>w->k)))return-1;
#if defined(__ARM_FEATURE_SVE)
    _Float16*panel=NULL;size_t bytes=(size_t)w->k*32*sizeof(*panel);
    if(posix_memalign((void**)&panel,256,bytes))return-1;
    int span=promotion_k?promotion_k:w->k;
    for(int t=0;t<w->n/32;++t){dequant_n32_panel(panel,w,t,0,w->k);
      for(int m0=0;m0<m;m0+=6){int mr=m-m0<6?m-m0:6;
        for(int kb=0;kb<w->k;kb+=span){int ke=kb+span<w->k?kb+span:w->k;
          dense_panel_m6(c+(size_t)m0*w->n,a+(size_t)m0*w->k,panel,w->k,w->n,t,kb,ke,0,mr,kb!=0);
        }}
    }free(panel);return 0;
#else
    return-1;
#endif
}

int fp4_gemm_f16_l1panel(float*c,const _Float16*a,const fp4_matrix*w,int m,int promotion_k){
    if(!c||!a||!w||!w->codes_n32||!w->scales_n32||m<1||promotion_k<32||
       promotion_k>w->k||promotion_k%32)return-1;
#if defined(__ARM_FEATURE_SVE)
    _Float16*panel=NULL;size_t bytes=(size_t)promotion_k*32*sizeof(*panel);
    if(posix_memalign((void**)&panel,256,bytes))return-1;
    for(int t=0;t<w->n/32;++t)for(int kb=0;kb<w->k;kb+=promotion_k){
      int ke=kb+promotion_k<w->k?kb+promotion_k:w->k;
      dequant_n32_panel(panel,w,t,kb,ke);
      for(int m0=0;m0<m;m0+=6){int mr=m-m0<6?m-m0:6;
        dense_panel_m6(c+(size_t)m0*w->n,a+(size_t)m0*w->k,panel,w->k,w->n,t,kb,ke,kb,mr,kb!=0);
      }
    }free(panel);return 0;
#else
    return-1;
#endif
}

int fp4_gemm_f16_l2_omp(float*c,const _Float16*a,const fp4_matrix*w,int m,
        int promotion_k,int threads){
    if(!c||!a||!w||!w->codes_n32||!w->scales_n32||m<1||threads<1||promotion_k<0||
       (promotion_k&&((promotion_k%32)||promotion_k>w->k)))return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    size_t panel_elems=(size_t)w->k*32,panel_bytes=panel_elems*sizeof(_Float16);
    _Float16*work=NULL;if(posix_memalign((void**)&work,256,panel_bytes*(size_t)threads))return-1;
    int span=promotion_k?promotion_k:w->k;
#pragma omp parallel num_threads(threads)
    {
      int tid=omp_get_thread_num();_Float16*panel=work+(size_t)tid*panel_elems;
#pragma omp for schedule(static)
      for(int t=0;t<w->n/32;++t){
        dequant_n32_panel(panel,w,t,0,w->k);
        for(int m0=0;m0<m;m0+=6){int mr=m-m0<6?m-m0:6;
          for(int kb=0;kb<w->k;kb+=span){int ke=kb+span<w->k?kb+span:w->k;
            dense_panel_m6(c+(size_t)m0*w->n,a+(size_t)m0*w->k,panel,w->k,w->n,t,kb,ke,0,mr,kb!=0);
          }
        }
      }
    }
    free(work);return 0;
#else
    (void)threads;return-1;
#endif
}

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

static int fp4_quant_i8(float x,float inverse){
    long q=lrintf(x*inverse);if(q>127)q=127;else if(q< -127)q=-127;return(int)q;
}

int fp4_i8_activation_prepare(fp4_i8_activation*q,const float*a,int k,int group){
    if(!q||!a||k<=0||k%32||group<4||group>32||(group&(group-1))||k%group)return-1;
    if(!q->codes||!q->scales||q->k!=k){fp4_i8_activation_free(q);q->k=k;
      if(posix_memalign((void**)&q->codes,256,(size_t)k)||
         posix_memalign((void**)&q->scales,256,(size_t)(k/4)*sizeof(float))){
          fp4_i8_activation_free(q);return-1;}}
    q->scale_group=group;
    for(int b=0;b<k/group;++b){float m=0.0f;
      for(int j=0;j<group;++j){float x=fabsf(a[b*group+j]);if(x>m)m=x;}
      float s=m>0.0f?m/127.0f:1.0f,inv=1.0f/s;q->scales[b]=s*0.5f;
      for(int j=0;j<group;++j)q->codes[b*group+j]=
          (int8_t)fp4_quant_i8(a[b*group+j],inv);
    }return 0;
}

void fp4_i8_activation_free(fp4_i8_activation*q){
    if(!q)return;free(q->codes);free(q->scales);memset(q,0,sizeof(*q));
}

int fp4_pair_activation_prepare(fp4_pair_activation*q,const float*a,int k,int group){
    static const int16_t values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    if(!q||!a||k<=0||k%32||group<4||group>32||(group&(group-1))||k%group)return-1;
    if(!q->codes||!q->scales||!q->tables||!q->tables16||q->k!=k){
      fp4_pair_activation_free(q);q->k=k;
      if(posix_memalign((void**)&q->codes,256,(size_t)k)||
         posix_memalign((void**)&q->scales,256,(size_t)(k/4)*sizeof(float))||
         posix_memalign((void**)&q->tables,256,(size_t)k*256)||
         posix_memalign((void**)&q->tables16,256,(size_t)k*64)){
          fp4_pair_activation_free(q);return-1;}}
    q->scale_group=group;
    for(int b=0;b<k/group;++b){float m=0.0f;
      for(int j=0;j<group;++j){float x=fabsf(a[b*group+j]);if(x>m)m=x;}
      float s=m>0.0f?m/127.0f:1.0f,inv=1.0f/s;q->scales[b]=s*0.5f;
      for(int j=0;j<group;++j)q->codes[b*group+j]=
          (int8_t)fp4_quant_i8(a[b*group+j],inv);
    }
    for(int p=0;p<k/2;++p){int a0=q->codes[p*2],a1=q->codes[p*2+1];
      int16_t*t=q->tables+(size_t)p*256;
      for(int c=0;c<256;++c)t[c]=(int16_t)(values[c&15]*a0+values[c>>4]*a1);
    }
    for(int x=0;x<k;++x)for(int c=0;c<32;++c)
      q->tables16[(size_t)x*32+c]=(int16_t)(values[c&15]*q->codes[x]);
    return 0;
}

void fp4_pair_activation_free(fp4_pair_activation*q){
    if(!q)return;free(q->codes);free(q->scales);free(q->tables);free(q->tables16);memset(q,0,sizeof(*q));
}

int fp4_gemv_pair_tbl_omp(float*c,const fp4_pair_activation*a,
                           const fp4_matrix*w,int threads){
    if(!c||!a||!w||!a->scales||!a->tables16||!w->codes_pair||!w->scales_pair||
       a->k!=w->k||threads<1||w->n%128||a->scale_group>16)return-1;
    int wg=w->format==FP4_MX?32:16;if(wg%a->scale_group)return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    int ng=w->n/128,nb=w->k/wg,pairs=wg/2;
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int g=0;g<ng;++g){
      fp4_pair_args x={w->codes_pair,w->scales_pair,a->tables16,a->scales,c,
                       ng,g,1,nb,pairs,a->scale_group};
      fp4_pair_tbl_m1_asm(&x);
    }return 0;
#else
    (void)threads;return-1;
#endif
}

int fp4_gemv_pair_lut_omp(float*c,const fp4_pair_activation*a,
                           const fp4_matrix*w,int threads){
    if(!c||!a||!w||!a->scales||!a->tables||!w->codes_pair||!w->scales_pair||
       a->k!=w->k||threads<1||w->n%128)return-1;
    int wg=w->format==FP4_MX?32:16;if(wg%a->scale_group)return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    int ng=w->n/128,nb=w->k/wg,pairs=wg/2;
#pragma omp parallel num_threads(threads)
    {
      int tid=omp_get_thread_num(),nt=omp_get_num_threads();
      int g0=ng*tid/nt,g1=ng*(tid+1)/nt;
      fp4_pair_args x={w->codes_pair,w->scales_pair,a->tables,a->scales,c,
                       ng,g0,g1-g0,nb,pairs,a->scale_group};
      if(g1>g0)fp4_pair_lut_m1_asm(&x);
    }return 0;
#else
    (void)threads;return-1;
#endif
}

int fp4_gemv_i8_sdot_omp(float*c,const fp4_i8_activation*a,
                          const fp4_matrix*w,int threads){
    if(!c||!a||!w||!a->codes||!a->scales||!w->codes_sdot||!w->scales_sdot||
       a->k!=w->k||threads<1)return-1;
    int wg=w->format==FP4_MX?32:16;if(wg%a->scale_group)return-1;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
    int nb=w->k/wg;
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int g=0;g<w->n/128;++g){fp4_sdot_args x={
      w->codes_sdot+(size_t)g*w->k*128,a->codes,
      w->scales_sdot+(size_t)g*nb*128,a->scales,c+g*128,w->k,wg,a->scale_group};
      fp4_i8_sdot_m1_asm(&x);}
    return 0;
#else
    (void)threads;return-1;
#endif
}

int fp4_gemv_i8_sdot4_omp(float*c,const fp4_i8_activation*a,
                           const fp4_matrix*w,int threads){
    if(!c||!a||!w||!a->codes||!a->scales||!w->codes_sdot4||!w->scales_sdot4||
       a->k!=w->k||a->scale_group!=4||threads<1||w->n%128)return-1;
    int wg=w->format==FP4_MX?32:16,ng=w->n/128,nb=w->k/wg,pg=wg/4;
#if defined(__ARM_FEATURE_SVE) && defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int g=0;g<ng;++g){fp4_sdot4_args x={w->codes_sdot4,a->codes,
      w->scales_sdot4,a->scales,c,ng,g,nb,pg};fp4_i8_sdot4_m1_asm(&x);}
    return 0;
#else
    (void)threads;return-1;
#endif
}
