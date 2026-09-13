#include "../../common/ds4f.h"
#include "dual_ds4f_prefill.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 64, N = 128, K = 4096, nb = K / 32;
    uint8_t *mw = malloc((size_t)N*K/2), *ms = malloc((size_t)N*nb);
    uint16_t *bw = malloc((size_t)N*K*2);
    float *x = malloc((size_t)M*K*4), *ym = malloc((size_t)M*N*4),
          *yb = malloc((size_t)M*N*4);
    if (!mw || !ms || !bw || !x || !ym || !yb) return 2;
    for (int r=0;r<N;r++) for (int b=0;b<nb;b++) {
        ms[(size_t)r*nb+b]=(uint8_t)(120+(r*13+b*7)%15);
        for (int j=0;j<16;j++) mw[(size_t)r*K/2+b*16+j]=(uint8_t)(r*29+b*17+j*7+3);
    }
    for (int i=0;i<N*K;i++) bw[i]=0x3f80; /* BF16 1.0 */
    for (int r=0;r<M;r++) for (int k=0;k<K;k++) x[(size_t)r*K+k]=((r*31+k*17)%1009-504)/173.f;
    ds4f_tensor tm={mw,ms,DS4F_MXFP4,N,K,-1};
    ds4f_tensor tb={bw,NULL,DS4F_BF16,N,K,-1};
    dual_ds4f_prefill *d=dual_ds4f_prefill_create(0,0,1);
    int bm=d ? dual_ds4f_prefill_bind_tensor(d,&tm) : -1;
    int bb=d ? dual_ds4f_prefill_bind_tensor(d,&tb) : -1;
    fprintf(stderr, "dual init=%p bind_mx=%d bind_bf=%d\n", (void *)d, bm, bb);
    int rc=d && bm>=0 && bb>=0;
    float *dst[2]={ym,yb}; const ds4f_tensor *ts[2]={&tm,&tb};
    const float *xs[2]={x,x}; int mm[2]={M,M}, ys[2]={N,N}, ks[2]={K,K};
    if (rc) rc=dual_ds4f_prefill_gemm_multi(d,dst,ts,xs,mm,ys,ks,2);
    float maxb=0; for (int r=0;r<M;r++) for (int n=0;n<N;n++) {
        float ref=0; for (int k=0;k<K;k++) ref += x[(size_t)r*K+k];
        float e=fabsf(yb[(size_t)r*N+n]-ref); if(e>maxb)maxb=e;
    }
    int pass = rc==0 && isfinite(ym[0]) && maxb < 5.0f;
    printf("dual prefill %s rc=%d mx0=%g bf16_max_abs=%g\n",pass?"PASS":"FAIL",rc,ym[0],maxb);
    dual_ds4f_prefill_destroy(d); free(mw); free(ms); free(bw); free(x); free(ym); free(yb);
    return pass ? 0 : 1;
}
