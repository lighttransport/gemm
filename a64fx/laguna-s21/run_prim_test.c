#define LAGUNA_BENCH
#include "laguna_s21_ep_runner.c"
static uint64_t rs=0xABCDEF0123456789ull;
static float fr(void){rs=rs*6364136223846793005ull+1442695040888963407ull;
    return (float)((int)((rs>>33)%2000)-1000)/1000.0f;}
int main(void){
    int hd=LAGUNA_HEAD_DIM, kvstride=LAGUNA_KV_HEADS*hd;
    int bad=0;
    for (int n=1;n<=257;n+= (n<8?1:37)) {
        float *q=aligned_alloc(256,hd*4);
        uint16_t *k=aligned_alloc(256,(size_t)n*kvstride*2);
        for(int i=0;i<hd;i++)q[i]=fr();
        for(size_t i=0;i<(size_t)n*kvstride;i++)k[i]=laguna_f32_to_bf16(fr()*0.5f);
        float scale=0.0883883f;
        float *r1=malloc(n*4),*r2=malloc(n*4);
        /* qk: run kernel vs per-key laguna_qkdot */
        float run_max=laguna_qk_run(r1,q,k,kvstride,n,scale,hd);
        for(int i=0;i<n;i++) r2[i]=laguna_qkdot(q,k+(size_t)i*kvstride,hd)*scale;
        double mq=0; float check_max=-INFINITY;
        for(int i=0;i<n;i++){double d=fabs((double)r1[i]-r2[i]);if(d>mq)mq=d;
            if(r1[i]>check_max)check_max=r1[i];}
        int max_ok=run_max==check_max;
        /* av: run kernel vs per-key laguna_vaxpy, with a nonzero incoming acc+corr */
        float *w=malloc(n*4); for(int i=0;i<n;i++)w[i]=fabsf(fr())+0.01f;
        float corr=0.37f;
        float a1[LAGUNA_HEAD_DIM],a2[LAGUNA_HEAD_DIM];
        for(int d=0;d<hd;d++){a1[d]=a2[d]=fr();}
        laguna_av_run(a1,w,k,kvstride,n,corr,hd);
        for(int i=0;i<n;i++) laguna_vaxpy(a2,k+(size_t)i*kvstride,w[i],i==0?corr:1.0f,hd);
        double ma=0,den=0; for(int d=0;d<hd;d++){double e=fabs((double)a1[d]-a2[d]);
            if(e>ma)ma=e; den+=fabs(a2[d]);}
        int ok = mq<2e-4 && ma < 1e-4*(1+den/hd) && max_ok;
        if(!ok){bad=1;printf("n=%3d  qk max|d|=%.3e max=%s  av max|d|=%.3e  ** FAIL **\n",
                            n,mq,max_ok?"exact":"BAD",ma);}
        else if(n<8||n>200)printf("n=%3d  qk max|d|=%.3e max=exact  av max|d|=%.3e  OK\n",n,mq,ma);
        free(q);free(k);free(r1);free(r2);free(w);
    }
    puts(bad?"run-primitive equivalence: FAIL":"run-primitive equivalence: PASS");
    return bad;
}
