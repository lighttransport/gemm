/* Statistical tests for laguna_sample: greedy, temperature, top-k, top-p, min-p.
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -DLAGUNA_FP8 -DLAGUNA_BENCH -I../../common -I../utofu-tests \
 *          -o sampler_test sampler_test.c -lm
 */
#define LAGUNA_BENCH
#include "laguna_s21_ep_runner.c"

#define NV 64
#define DRAWS 400000

static int fails = 0;
static void check(int ok, const char *what, const char *detail) {
    printf("  %-52s %s%s%s\n", what, ok?"OK":"** FAIL **", detail?"  ":"", detail?detail:"");
    if(!ok) fails=1;
}

/* reference softmax over the top-k of `lg`, after temperature */
static void ref_topk_probs(const float *lg, int n, int k, float temp, double *p, int *idx) {
    int ord[NV]; for(int i=0;i<n;i++) ord[i]=i;
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) if(lg[ord[j]]>lg[ord[i]]){int t=ord[i];ord[i]=ord[j];ord[j]=t;}
    double s=0;
    for(int i=0;i<k;i++){ idx[i]=ord[i]; p[i]=exp((lg[ord[i]]-lg[ord[0]])/temp); s+=p[i]; }
    for(int i=0;i<k;i++) p[i]/=s;
}

int main(void) {
    static float lg[NV];
    uint64_t sd=99;
    for (int i=0;i<NV;i++){ sd=sd*6364136223846793005ull+1; lg[i]=(float)((int)((sd>>33)%1000))/100.0f; }
    int top = argmax(lg,NV);

    puts("greedy / degenerate settings:");
    { laguna_sampler s={.do_sample=0,.temp=1,.top_p=1,.min_p=0,.top_k=20,.rng=1};
      int ok=1; for(int i=0;i<100;i++) if(laguna_sample(lg,NV,&s)!=top) ok=0;
      check(ok,"do_sample=0 always returns argmax",NULL); }
    { laguna_sampler s={.do_sample=1,.temp=1,.top_p=1,.min_p=0,.top_k=1,.rng=7};
      int ok=1; for(int i=0;i<100;i++) if(laguna_sample(lg,NV,&s)!=top) ok=0;
      check(ok,"top_k=1 always returns argmax",NULL); }
    { laguna_sampler s={.do_sample=1,.temp=0.0f,.top_p=1,.min_p=0,.top_k=20,.rng=7};
      int ok=1; for(int i=0;i<50;i++) if(laguna_sample(lg,NV,&s)!=top) ok=0;
      check(ok,"temp=0 falls back to greedy",NULL); }

    puts("top-k support:");
    for (int k=1;k<=20;k+=19) {
        laguna_sampler s={.do_sample=1,.temp=1,.top_p=1,.min_p=0,.top_k=k,.rng=42};
        double pr[NV]; int id[NV]; ref_topk_probs(lg,NV,k,1.0f,pr,id);
        int allowed[NV]={0}; for(int i=0;i<k;i++) allowed[id[i]]=1;
        int bad=0; static int cnt[NV];
        memset(cnt,0,sizeof cnt);
        for(int d=0;d<DRAWS;d++){ int t=laguna_sample(lg,NV,&s); if(!allowed[t]) bad++; cnt[t]++; }
        char buf[64]; snprintf(buf,sizeof buf,"(k=%d, %d out-of-set)",k,bad);
        check(bad==0,"only top-k tokens are ever drawn",buf);
        /* frequencies match the reference softmax */
        double maxerr=0;
        for(int i=0;i<k;i++){ double f=(double)cnt[id[i]]/DRAWS; double e=fabs(f-pr[i]);
            if(pr[i]>0.01 && e/pr[i]>maxerr) maxerr=e/pr[i]; }
        snprintf(buf,sizeof buf,"(k=%d, max rel err %.3f)",k,maxerr);
        check(maxerr<0.05,"draw frequencies match softmax probabilities",buf);
    }

    puts("temperature:");
    /* The property is monotonicity of P(top) in temperature, not any particular
     * value -- how peaked a given temperature looks depends entirely on the gaps
     * between the top logits, which are ~0.16 apart for this test distribution. */
    { const float T[] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
      double f[5]; int mono=1; char buf[128]; int off=0;
      for (int i=0;i<5;i++) {
        laguna_sampler s={.do_sample=1,.temp=T[i],.top_p=1,.min_p=0,.top_k=20,.rng=5};
        int c=0; for(int d=0;d<DRAWS;d++) if(laguna_sample(lg,NV,&s)==top) c++;
        f[i]=(double)c/DRAWS;
        off+=snprintf(buf+off,sizeof buf-off,"%s%.3f",i?" > ":"(P(top): ",f[i]);
        if(i && f[i] >= f[i-1]) mono=0;
      }
      snprintf(buf+off,sizeof buf-off,")");
      check(mono,"P(top) decreases monotonically with temperature",buf); }
    /* With a clearly separated top logit, a low temperature must be ~deterministic. */
    { static float sep[NV]; for(int i=0;i<NV;i++) sep[i]=(float)(i%7);
      sep[13]=20.0f;   /* unique, far above the rest */
      laguna_sampler s={.do_sample=1,.temp=0.1f,.top_p=1,.min_p=0,.top_k=20,.rng=6};
      int c=0; for(int d=0;d<DRAWS/10;d++) if(laguna_sample(sep,NV,&s)==13) c++;
      double f=(double)c/(DRAWS/10); char buf[64];
      snprintf(buf,sizeof buf,"(P=%.4f)",f);
      check(f>0.999,"temp=0.1 on a separated peak is ~deterministic",buf); }

    puts("top-p:");
    { laguna_sampler s={.do_sample=1,.temp=1,.top_p=0.5f,.min_p=0,.top_k=20,.rng=11};
      double pr[NV]; int id[NV]; ref_topk_probs(lg,NV,20,1.0f,pr,id);
      double c=0; int m=0; for(int i=0;i<20;i++){ c+=pr[i]; m=i+1; if(c>=0.5) break; }
      int allowed[NV]={0}; for(int i=0;i<m;i++) allowed[id[i]]=1;
      int bad=0; for(int d=0;d<DRAWS;d++) if(!allowed[laguna_sample(lg,NV,&s)]) bad++;
      char buf[64]; snprintf(buf,sizeof buf,"(nucleus=%d tokens, %d outside)",m,bad);
      check(bad==0,"top_p=0.5 draws only from the nucleus",buf); }

    puts("min-p:");
    { float mp=0.3f;
      laguna_sampler s={.do_sample=1,.temp=1,.top_p=1,.min_p=mp,.top_k=20,.rng=13};
      double pr[NV]; int id[NV]; ref_topk_probs(lg,NV,20,1.0f,pr,id);
      int allowed[NV]={0}, m=0;
      for(int i=0;i<20;i++){ if(pr[i] < mp*pr[0]) break; allowed[id[i]]=1; m++; }
      int bad=0; for(int d=0;d<DRAWS;d++) if(!allowed[laguna_sample(lg,NV,&s)]) bad++;
      char buf[64]; snprintf(buf,sizeof buf,"(kept=%d tokens, %d outside)",m,bad);
      check(bad==0,"min_p=0.3 drops tokens below 0.3*p_max",buf); }

    puts("determinism:");
    { laguna_sampler a={.do_sample=1,.temp=1,.top_p=1,.min_p=0,.top_k=20,.rng=2024};
      laguna_sampler b={.do_sample=1,.temp=1,.top_p=1,.min_p=0,.top_k=20,.rng=2024};
      int ok=1; for(int d=0;d<1000;d++) if(laguna_sample(lg,NV,&a)!=laguna_sample(lg,NV,&b)) ok=0;
      check(ok,"same seed gives the same sequence",NULL); }
    { laguna_sampler a={.do_sample=1,.temp=1,.top_p=1,.min_p=0,.top_k=20,.rng=1};
      laguna_sampler b={.do_sample=1,.temp=1,.top_p=1,.min_p=0,.top_k=20,.rng=2};
      int diff=0; for(int d=0;d<1000;d++) if(laguna_sample(lg,NV,&a)!=laguna_sample(lg,NV,&b)) diff++;
      check(diff>100,"different seeds diverge",NULL); }

    puts(fails ? "sampler: FAIL" : "sampler: PASS");
    return fails;
}
