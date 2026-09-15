#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "glm53f_prefill_gemm.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int test(int rows, int cols, int tokens, int fp8, int repeats) {
    float *arena = malloc((size_t)((rows+47)/48)*48*cols*sizeof(float));
    uint16_t *bf = malloc((size_t)rows*cols*sizeof(uint16_t));
    uint8_t *eight = malloc((size_t)rows*cols);
    float *scale = malloc((size_t)((rows+127)/128)*((cols+127)/128)*sizeof(float));
    float *x = malloc((size_t)tokens*cols*sizeof(float));
    float *out = malloc((size_t)tokens*rows*sizeof(float));
    if (!arena || !bf || !eight || !scale || !x || !out) return 2;
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < cols; ++k) {
            size_t i = (size_t)r*cols+k;
            float v = sinf((r*13+k*17)*.011f);
            uint32_t bits; memcpy(&bits,&v,4); bf[i] = (uint16_t)(bits>>16);
            int q = (r*37+k*13)%256;
            eight[i] = (q&127)==127 ? 0 : (uint8_t)q;
        }
    for (int i=0; i<((rows+127)/128)*((cols+127)/128); ++i) scale[i] = .001f*(1+i%13);
    for (int i=0; i<tokens*cols; ++i) x[i] = i<cols ? 0 : cosf(i*.013f);
    double begin=0;
    for (int rep=-1; rep<repeats; ++rep) {
        if (!rep) begin=now();
#pragma omp parallel
        glm53f_prefill_gemm_team(out,fp8?(const void*)eight:bf,scale,x,tokens,rows,cols,fp8,arena);
    }
    double elapsed=(now()-begin)/repeats, err=0, norm=0, maxabs=0, maxref=0;
    double legacy=0;
#if defined(__ARM_FEATURE_SVE)
    if (rows%4==0 && cols%128==0) {
        float *old=malloc((size_t)tokens*rows*sizeof(float));
        if (!old) return 2;
        for (int rep=-1;rep<repeats;++rep) {
            if (!rep) begin=now();
#pragma omp parallel for collapse(2) schedule(static)
            for (int r=0;r<rows;r+=4)
                for (int t=0;t<tokens;t+=4) {
                    int n=tokens-t<4?tokens-t:4;
                    if (fp8) glm53f_matvec_fp8_bits_4x4(old+(size_t)t*rows+r,rows,
                        eight+(size_t)r*cols,scale+(size_t)(r/128)*(cols/128),x+(size_t)t*cols,n,cols);
                    else glm53f_matvec_bf16_4x4(old+(size_t)t*rows+r,rows,bf+(size_t)r*cols,x+(size_t)t*cols,n,cols);
                }
        }
        legacy=(now()-begin)/repeats;
        free(old);
    }
#endif
    int fail=0;
#pragma omp parallel for collapse(2) reduction(+:err,norm) reduction(max:maxabs,maxref) reduction(|:fail)
    for (int t=0; t<tokens; ++t)
        for (int r=0; r<rows; ++r) {
            double sum=0;
            for (int k=0; k<cols; ++k) {
                double w=fp8 ? (double)glm53f_fp8_e4m3_scalar(eight[(size_t)r*cols+k]) *
                    scale[(size_t)(r/128)*((cols+127)/128)+k/128] : glm53f_bf16_to_f32(bf[(size_t)r*cols+k]);
                sum += w*x[(size_t)t*cols+k];
            }
            double d=out[(size_t)t*rows+r]-sum;
            if (!isfinite(d)) fail=1;
            err+=d*d; norm+=sum*sum;
            if (fabs(d)>maxabs) maxabs=fabs(d);
            if (fabs(sum)>maxref) maxref=fabs(sum);
        }
    double rel=sqrt(err/fmax(norm,1e-30));
    fail |= rel>1e-4 || maxabs>1e-6+5e-4*maxref;
    printf("PREFILL_GEMM type=%s rows=%d cols=%d tokens=%d pack_compute_ms=%.6f legacy_ms=%.6f speedup=%.3f rel_l2=%.9g max_abs=%.9g %s\n",
        fp8?"fp8":"bf16",rows,cols,tokens,elapsed*1e3,legacy*1e3,legacy/elapsed,rel,maxabs,fail?"FAIL":"PASS");
    free(arena); free(bf); free(eight); free(scale); free(x); free(out);
    return fail;
}

int main(int argc,char **argv) {
    for (int i=0;i<256;++i) {
        float a=glm53f_prefill_fp8(i), b=glm53f_fp8_e4m3_scalar(i);
        if (!(isnan(a)&&isnan(b)) && memcmp(&a,&b,4)) return 1;
    }
    if (argc==5) {
        int rows=atoi(argv[1]),cols=atoi(argv[2]),tokens=atoi(argv[3]),reps=atoi(argv[4]);
        if (rows<1||rows>4096||cols<1||cols>4096||tokens<1||tokens>32||reps<1) return 2;
        return test(rows,cols,tokens,0,reps)|test(rows,cols,tokens,1,reps);
    }
    const int rows[]={1,15,16,17,47,48,49,127,128,129,192};
    const int tokens[]={1,5,8,9,31,32};
    int fail=0;
    for (size_t r=0;r<sizeof(rows)/sizeof(rows[0]);++r)
        for (size_t t=0;t<sizeof(tokens)/sizeof(tokens[0]);++t)
            for (int fp8=0;fp8<2;++fp8) fail |= test(rows[r],257,tokens[t],fp8,1);
    return fail;
}
