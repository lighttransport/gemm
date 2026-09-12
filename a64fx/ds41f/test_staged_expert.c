#include "ds41f_tensor.h"
#include "ds41f_expert.h"
#include "ds41f_sve.h"
#include "ds41f_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr,"usage: %s stage_dir expert_id\n",argv[0]); return 2; }
    int expert = atoi(argv[2]);
    if (expert < 0 || expert >= 384) return 2;
    size_t rows = 2304, cols = 5120;
    char name[256]; void *w = NULL, *scale = NULL;
    snprintf(name,sizeof name,"layers.0.ffn.experts.%d.w1.weight",expert);
    int rc = ds41f_tensor_load(argv[1],name,rows*cols/2,&w);
    if (rc) { fprintf(stderr,"load %s error=%d\n",name,rc); return 1; }
    snprintf(name,sizeof name,"layers.0.ffn.experts.%d.w1.scale",expert);
    rc = ds41f_tensor_load(argv[1],name,rows*cols/32,&scale);
    if (rc) { fprintf(stderr,"load %s error=%d\n",name,rc); free(w); return 1; }
    float *x = malloc(cols*sizeof *x), *y = malloc(rows*sizeof *y), *ref = malloc(rows*sizeof *ref);
    if (!x || !y || !ref) return 1;
    for (size_t i = 0; i < cols; ++i) x[i] = sinf((float)i*.013f);
    rc = ds41f_mxfp4_matvec(y,w,scale,x,rows,cols);
    rc |= ds41f_mxfp4_matvec_ref(ref,w,scale,x,rows,cols);
    float max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < rows; ++i) {
        float err = fabsf(y[i]-ref[i]);
        if (!isfinite(y[i]) || !isfinite(ref[i]) || err > 2e-4f*(1+fabsf(ref[i]))) rc = 1;
        max_abs = fmaxf(max_abs,err);
        max_rel = fmaxf(max_rel,err/(1+fabsf(ref[i])));
    }
    printf("STAGED_EXPERT %s expert=%d max_abs=%g normalized_error=%g\n",rc?"FAIL":"PASS",expert,max_abs,max_rel);
    free(x); free(y); free(ref); free(w); free(scale);
    ds41f_expert e;
    if (ds41f_expert_load(&e,argv[1],0,expert)) return 1;
    x=malloc(5120*sizeof *x);y=malloc(5120*sizeof *y);ref=malloc(5120*sizeof *ref);
    float *scratch=malloc((3*2304+5120)*sizeof *scratch);
    if (!x || !y || !ref || !scratch) return 1;
    for (size_t i=0;i<5120;++i) x[i]=sinf((float)i*.013f);
    rc|=ds41f_expert_forward(&e,y,x,.25f,scratch,0);
    rc|=ds41f_expert_forward(&e,ref,x,.25f,scratch,1);
    max_abs=0;
    for (size_t i=0;i<5120;++i) {
        float err=fabsf(y[i]-ref[i]);
        if (!isfinite(y[i]) || !isfinite(ref[i]) || err>2e-4f*(1+fabsf(ref[i]))) rc=1;
        max_abs=fmaxf(max_abs,err);
    }
    printf("STAGED_EXPERT_CHAIN %s expert=%d max_abs=%g activation=dynamic_FP8_BF16_boundaries\n",rc?"FAIL":"PASS",expert,max_abs);
    ds41f_expert_free(&e);free(x);free(y);free(ref);free(scratch);
    return !!rc;
}
