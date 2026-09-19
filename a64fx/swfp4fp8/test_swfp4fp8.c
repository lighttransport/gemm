#include "swfp4fp8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_state = 1;
static uint32_t rng_u32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static float rng_f32(void) { return ((int)(rng_u32() >> 8) % 2001 - 1000) / 4096.0f; }

static int check_close(const char *name, const float *got, const double *ref,
                       size_t count, double limit) {
    double se = 0.0, sr = 0.0, maxe = 0.0, maxr = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double e = fabs((double)got[i] - ref[i]);
        se += e * e;
        sr += ref[i] * ref[i];
        if (e > maxe) maxe = e;
        if (fabs(ref[i]) > maxr) maxr = fabs(ref[i]);
    }
    double rel = sqrt(se / (sr + 1e-300));
    double scaled = maxe / fmax(1.0, maxr);
    if (rel > limit || scaled > limit * 5.0) {
        fprintf(stderr, "%s FAIL rel_l2=%g scaled_max=%g\n", name, rel, scaled);
        return 1;
    }
    printf("%-18s PASS rel_l2=%g scaled_max=%g\n", name, rel, scaled);
    return 0;
}

static uint8_t fp4_at(const uint8_t *codes, size_t k, size_t n, size_t kk,
                      int split) {
    const uint8_t *row = codes + n * (k / 2);
    if (!split) {
        uint8_t b = row[kk / 2];
        return (b >> (4 * (kk & 1))) & 15;
    }
    size_t g = kk / 32, q = kk % 32;
    uint8_t b = row[g * 16 + (q & 15)];
    return q < 16 ? b & 15 : b >> 4;
}

static void ref_fp4(double *dst, const float *a, size_t m, size_t n, size_t k,
                    const uint8_t *codes, const uint8_t *scales,
                    size_t group, float global, int split) {
    for (size_t im = 0; im < m; ++im)
        for (size_t in = 0; in < n; ++in) {
            double sum = 0.0;
            for (size_t kk = 0; kk < k; ++kk) {
                float s = split ? swfp4fp8_decode_e8m0(scales[in * (k/group) + kk/group])
                                : swfp4fp8_decode_e4m3(scales[in * (k/group) + kk/group]);
                sum += (double)a[im*k+kk] * swfp4fp8_decode_e2m1(fp4_at(codes,k,in,kk,split)) * s * global;
            }
            dst[im*n+in] = sum;
        }
}

static void ref_fp8(double *dst, const float *a, size_t m, size_t n, size_t k,
                    const uint8_t *codes, const void *scales, int block128) {
    for (size_t im = 0; im < m; ++im)
        for (size_t in = 0; in < n; ++in) {
            double sum = 0.0;
            for (size_t kk = 0; kk < k; ++kk) {
                float s;
                if (block128) {
                    const uint8_t *bs = scales;
                    s = swfp4fp8_decode_e8m0(bs[(in/128)*(k/128)+kk/128]);
                } else {
                    const float *ts = scales;
                    s = ts[in/32] / 256.0f;
                }
                sum += (double)a[im*k+kk] * swfp4fp8_decode_e4m3(codes[in*k+kk]) * s;
            }
            dst[im*n+in] = sum;
        }
}

static const uint8_t korder[16] = {0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15};
static const uint8_t lane_col[32] = {0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,
                                     4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31};

static void make_qpn2(uint8_t *qc, uint8_t *qs, const uint8_t *codes,
                      const uint8_t *scales, size_t n, size_t k) {
    size_t groups = k / 16;
    for (size_t t = 0; t < n/32; ++t)
        for (size_t g = 0; g < groups; ++g)
            for (size_t lane = 0; lane < 32; ++lane) {
                size_t row = t*32 + lane_col[lane];
                qs[(t*groups+g)*32+lane] = scales[row*groups+g];
                for (size_t q = 0; q < 8; ++q) {
                    uint8_t c0 = fp4_at(codes,k,row,g*16+korder[q*2],0);
                    uint8_t c1 = fp4_at(codes,k,row,g*16+korder[q*2+1],0);
                    qc[((t*groups+g)*32+lane)*8+q] = c0 | (c1<<4);
                }
            }
}

static void make_qpn8(uint8_t *q, const uint8_t *codes, size_t n, size_t k) {
    size_t groups = k/16;
    for (size_t t=0;t<n/32;++t) for(size_t g=0;g<groups;++g)
        for(size_t lane=0;lane<32;++lane) for(size_t j=0;j<16;++j)
            q[((t*groups+g)*32+lane)*16+j] =
                codes[(t*32+lane_col[lane])*k+g*16+korder[j]];
}

static int test_decoders(void) {
    int fail = 0;
    fail |= swfp4fp8_decode_e2m1(7) != 6.0f;
    fail |= swfp4fp8_decode_e2m1(15) != -6.0f;
    fail |= swfp4fp8_decode_e4m3(1) != 0x1p-9f;
    fail |= swfp4fp8_decode_e4m3(0x38) != 1.0f;
    fail |= swfp4fp8_decode_e4m3(0x7e) != 448.0f;
    fail |= !isnan(swfp4fp8_decode_e4m3(0x7f));
    fail |= !isnan(swfp4fp8_decode_e4m3(0xff));
    fail |= swfp4fp8_decode_e8m0(127) != 1.0f;
    fail |= !isnan(swfp4fp8_decode_e8m0(255));
    for (int i=0;i<256;++i) {
        float x=swfp4fp8_decode_e4m3((uint8_t)i);
        if ((i&0x7f)==0x7f) fail |= !isnan(x); else fail |= !isfinite(x);
    }
    printf("decoders           %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

static int run_tests(void) {
    const size_t m=17,n=35,k=128;
    int fail=test_decoders(), rc;
    swfp4fp8_context *ctx=NULL;
    rc=swfp4fp8_context_create(&ctx,4,NULL); if(rc) return 1;
    float *a=malloc(m*k*sizeof(*a)), *got=malloc(m*n*sizeof(*got));
    double *ref=malloc(m*n*sizeof(*ref));
    uint8_t *fp4=malloc(n*k/2), *fp8=malloc(n*k);
    uint8_t *s16=malloc(n*(k/16)), *s32=malloc(n*(k/32));
    uint8_t *s128=malloc(((n+127)/128)*(k/128));
    for(size_t i=0;i<m*k;++i)a[i]=rng_f32();
    for(size_t i=0;i<n*k/2;++i)fp4[i]=(uint8_t)rng_u32();
    for(size_t i=0;i<n*k;++i){ uint8_t q=(uint8_t)rng_u32(); fp8[i]=((q&0x7f)==0x7f)?0x38:q; }
    memset(s16,0x38,n*(k/16)); memset(s32,127,n*(k/32)); memset(s128,127,((n+127)/128)*(k/128));

    swfp4fp8_matrix *w=NULL;
    rc=swfp4fp8_pack_nvfp4(ctx,&w,n,k,fp4,s16,0.25f,SWFP4FP8_LAYOUT_CANONICAL);
    fail|=rc; if(!rc){ uint8_t *u=malloc(n*k/2); swfp4fp8_unpack_codes(w,u); fail|=memcmp(u,fp4,n*k/2)!=0; free(u);
        swfp4fp8_gemm_f32(ctx,w,a,k,got,n,m,SWFP4FP8_KERNEL_AUTO); ref_fp4(ref,a,m,n,k,fp4,s16,16,.25f,0); fail|=check_close("nvfp4 canonical",got,ref,m*n,2e-5);
        swfp4fp8_gemm_f32(ctx,w,a,k,got,n,m,SWFP4FP8_KERNEL_ROW); fail|=check_close("nvfp4 row",got,ref,m*n,2e-5);
        swfp4fp8_gemm_f32(ctx,w,a,k,got,n,m,SWFP4FP8_KERNEL_FP4_SDOT); fail|=check_close("nvfp4 sdot lossy",got,ref,m*n,3e-2); swfp4fp8_matrix_destroy(w); }

    /* V100 QPN2 import, including its lane/K permutation. */
    const size_t nq=64; uint8_t *f4q=malloc(nq*k/2),*sq=malloc(nq*(k/16));
    uint8_t *qc=malloc(nq*k/2),*qsc=malloc(nq*(k/16));
    for(size_t i=0;i<nq*k/2;++i)f4q[i]=(uint8_t)rng_u32(); memset(sq,0x38,nq*(k/16)); make_qpn2(qc,qsc,f4q,sq,nq,k);
    rc=swfp4fp8_pack_nvfp4(ctx,&w,nq,k,qc,qsc,1.f,SWFP4FP8_LAYOUT_V100_QPN); fail|=rc;
    if(!rc){ uint8_t *u=malloc(nq*k/2); swfp4fp8_unpack_codes(w,u); fail|=memcmp(u,f4q,nq*k/2)!=0; free(u); swfp4fp8_matrix_destroy(w); }
    free(f4q);free(sq);free(qc);free(qsc);

    rc=swfp4fp8_pack_mxfp4(ctx,&w,n,k,fp4,s32); fail|=rc;
    if(!rc){ uint8_t *u=malloc(n*k/2); swfp4fp8_unpack_codes(w,u); fail|=memcmp(u,fp4,n*k/2)!=0; free(u);
        swfp4fp8_gemm_f32(ctx,w,a,k,got,n,m,SWFP4FP8_KERNEL_PANEL); ref_fp4(ref,a,m,n,k,fp4,s32,32,1.f,1); fail|=check_close("mxfp4 native",got,ref,m*n,2e-5); swfp4fp8_matrix_destroy(w); }
    rc=swfp4fp8_pack_mxfp4(ctx,&w,32,k,fp4,s32); fail|=rc;
    if(!rc){ swfp4fp8_gemm_f32(ctx,w,a,k,got,32,1,SWFP4FP8_KERNEL_MXFP4_FUSED_SDOT);
        ref_fp4(ref,a,1,32,k,fp4,s32,32,1.f,1);
        fail|=check_close("mxfp4 fused sdot",got,ref,32,3e-2); swfp4fp8_matrix_destroy(w); }

    float ts[2]={256.f,128.f};
    uint8_t *f8q=malloc(nq*k),*q8=malloc(nq*k); for(size_t i=0;i<nq*k;++i){uint8_t q=(uint8_t)rng_u32();f8q[i]=((q&0x7f)==0x7f)?0x38:q;} make_qpn8(q8,f8q,nq,k);
    rc=swfp4fp8_pack_qpn8(ctx,&w,nq,k,q8,ts,SWFP4FP8_LAYOUT_V100_QPN); fail|=rc;
    if(!rc){uint8_t*u=malloc(nq*k);swfp4fp8_unpack_codes(w,u);fail|=memcmp(u,f8q,nq*k)!=0;free(u);swfp4fp8_matrix_destroy(w);} free(f8q);free(q8);

    /* Canonical QPN8 requires N multiple 32; use N=32. */
    float t1[1]={256.f};
    rc=swfp4fp8_pack_qpn8(ctx,&w,32,k,fp8,t1,SWFP4FP8_LAYOUT_CANONICAL); fail|=rc;
    if(!rc){ swfp4fp8_gemm_f32(ctx,w,a,k,got,32,m,SWFP4FP8_KERNEL_AUTO); ref_fp8(ref,a,m,32,k,fp8,t1,0); fail|=check_close("qpn8 canonical",got,ref,m*32,2e-5);
        swfp4fp8_gemm_f32(ctx,w,a,k,got,32,m,SWFP4FP8_KERNEL_ROW); fail|=check_close("qpn8 row",got,ref,m*32,2e-5); swfp4fp8_matrix_destroy(w); }

    rc=swfp4fp8_pack_fp8_block128(ctx,&w,n,k,fp8,s128); fail|=rc;
    if(!rc){ uint8_t *u=malloc(n*k); swfp4fp8_unpack_codes(w,u); fail|=memcmp(u,fp8,n*k)!=0; free(u);
        swfp4fp8_gemm_f32(ctx,w,a,k,got,n,m,SWFP4FP8_KERNEL_AUTO); ref_fp8(ref,a,m,n,k,fp8,s128,1); fail|=check_close("fp8 block128",got,ref,m*n,2e-5);
        uint16_t *ah=malloc(m*k*2),*ch=malloc(m*n*2); for(size_t i=0;i<m*k;++i){_Float16 h=(_Float16)a[i];memcpy(ah+i,&h,2);} rc=swfp4fp8_gemm_f16(ctx,w,ah,k,ch,n,m,SWFP4FP8_KERNEL_AUTO); fail|=rc;
        for(size_t i=0;i<m*n;++i){_Float16 h;memcpy(&h,ch+i,2);got[i]=(float)h;} fail|=check_close("fp16 adapter",got,ref,m*n,3e-3);free(ah);free(ch);swfp4fp8_matrix_destroy(w); }

    free(a);free(got);free(ref);free(fp4);free(fp8);free(s16);free(s32);free(s128);
    swfp4fp8_context_destroy(ctx);
    return fail;
}

int main(void) {
    int fail=run_tests();
    printf("swfp4fp8 tests: %s\n",fail?"FAIL":"PASS");
    return fail?1:0;
}
