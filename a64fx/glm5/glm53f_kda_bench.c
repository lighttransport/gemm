#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../common/glm53f_ref.h"

static double now_sec(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
int main(void) {
    enum { L=34, D=128 };
    int H=getenv("HEADS_PER_RANK")?atoi(getenv("HEADS_PER_RANK")):64;
    int NH=L*H;
    size_t nelem=(size_t)NH*D*D, bytes=nelem*sizeof(float);
    int reps=getenv("REPS")?atoi(getenv("REPS")):20, r, h;
    float *state=NULL, *q=NULL, *k=NULL, *v=NULL, *out=NULL, *work=NULL;
    double best=1e30, sum=0; volatile double sink=0;
    if(H<1||H>64||posix_memalign((void**)&state,256,bytes)||posix_memalign((void**)&q,256,(size_t)NH*D*4)||
       posix_memalign((void**)&k,256,(size_t)NH*D*4)||posix_memalign((void**)&v,256,(size_t)NH*D*4)||
       posix_memalign((void**)&out,256,(size_t)NH*D*4)||posix_memalign((void**)&work,256,(size_t)NH*D*4)) return 2;
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<nelem;i++) state[i]=(float)((int)(i%31)-15)*1e-5f;
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<(size_t)NH*D;i++) q[i]=k[i]=v[i]=(float)((int)(i%17)-8)*0.01f;
    for(r=0;r<reps;r++) { double t=now_sec();
#pragma omp parallel for schedule(static)
        for(h=0;h<NH;h++) glm53f_kda_step_streamed(state+(size_t)h*D*D,q+(size_t)h*D,k+(size_t)h*D,
            v+(size_t)h*D,-0.01f,0.5f,D,D,out+(size_t)h*D,work+(size_t)h*D);
        t=now_sec()-t; if(t<best)best=t; sum+=t;
    }
    for(h=0;h<NH;h++) sink+=out[(size_t)h*D];
    printf("GLM53F_KDA layers=%d heads=%d state=%.3fGiB reps=%d best_ms=%.3f mean_ms=%.3f tok_s=%.2f traffic_GBs=%.1f sink=%.6g\n",
           L,H,bytes/1073741824.0,reps,best*1e3,sum/reps*1e3,1.0/best,(4.0*bytes)/best/1e9,sink);
    free(work);free(out);free(v);free(k);free(q);free(state); return 0;
}
