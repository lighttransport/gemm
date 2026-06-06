/* int8 KV-latent quantization probe (single-process, no MPI) — de-risk the int8 KV
 * lever before wiring it into ds4f_impl.h. The kv_cache latent is [kv_lora=512] per
 * (layer,pos). Per the bf16-not-fp16 finding, a few latent CHANNELS carry massive
 * activations (~1e3..1e5) while the rest are O(1). int8 has only 256 levels, so the
 * scale choice decides whether the O(1) dims survive. Compare schemes on rel-error
 * vs the bf16 reference (the committed kv storage), and on bytes/elem.
 *
 * Schemes:
 *   S1 per-token  absmax/127           (1 scale per row)       1.00 B/elem + scales
 *   S2 per-channel absmax/127          (1 scale per col)       1.00 B/elem + scales
 *   S3 per-token int8 + bf16 sidecar for the K largest-|.| dims (hybrid)
 *   S4 per-token, scale from a high percentile (robust), outliers clamped
 *
 * Build:  fcc -Nclang -O3 -march=armv8.2-a+sve -o /tmp/int8kv_probe /tmp/int8kv_probe.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define KV   512          /* kv_lora */
#define NTOK 4096         /* synthetic positions */
#define NCH_MASSIVE 6     /* # channels that are activation sinks */

static uint64_t rng = 0x9e3779b97f4a7c15ULL;
static double urand(void){ rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; return (rng>>11)*(1.0/9007199254740992.0); }
static double nrand(void){ double u1=urand(), u2=urand(); if(u1<1e-12)u1=1e-12; return sqrt(-2*log(u1))*cos(6.283185307*u2); }

/* bf16 round-to-nearest-even-ish (truncate+round bit) — matches ds4f_f32bf/bf16f intent */
static uint16_t f32bf(float f){ uint32_t u; memcpy(&u,&f,4); uint32_t r=(u>>16)&1; u+=0x7fff+r; return (uint16_t)(u>>16); }
static float bf16f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,4); return f; }

/* a representative latent matrix: base N(0,1)*0.8, plus NCH_MASSIVE sink channels
 * whose magnitude is large and token-varying (1e3..1e5), sign-stable per channel. */
static float *make_latents(void){
    float *X = malloc((size_t)NTOK*KV*sizeof(float));
    int sink[NCH_MASSIVE]; for(int s=0;s<NCH_MASSIVE;s++) sink[s]=(int)(urand()*KV);
    float sinkbase[NCH_MASSIVE], sinksign[NCH_MASSIVE];
    for(int s=0;s<NCH_MASSIVE;s++){ sinkbase[s]=powf(10.f, 3.0f+3.0f*(float)urand()); sinksign[s]=urand()<0.5?-1.f:1.f; }
    for(int t=0;t<NTOK;t++) for(int d=0;d<KV;d++) X[(size_t)t*KV+d]=(float)(nrand()*0.8);
    for(int s=0;s<NCH_MASSIVE;s++) for(int t=0;t<NTOK;t++)
        X[(size_t)t*KV+sink[s]] = sinksign[s]*sinkbase[s]*(0.5f+(float)urand());
    return X;
}

/* L1-relative error: sum|q-a| / sum|a|. This is what an attention dot-product over the
 * latent actually sees (it sums element contributions), and it is immune to the
 * tiny-value threshold artifact that inflates max-of-pointwise-rel to ~1.0. Also report
 * the fraction of "real" dims (|a|>0.1) whose pointwise rel exceeds 5%. */
typedef struct { double l1rel, badfrac; long n; double bytes_per_elem; } Res;
static Res score(const float *ref, const float *q, double bpe){
    Res r={0,0,0,bpe}; double num=0,den=0; long bad=0,n=0;
    for(size_t i=0;i<(size_t)NTOK*KV;i++){ float a=ref[i]; float aa=a<0?-a:a;
        num+=fabs((double)q[i]-a); den+=aa;
        if(aa>0.1f){ n++; if(fabs((double)q[i]-a)/(aa+1e-9)>0.05) bad++; } }
    r.l1rel=den>0?num/den:0; r.badfrac=n?(double)bad/n:0; r.n=n; return r;
}

int main(void){
    float *X = make_latents();
    float *bref = malloc((size_t)NTOK*KV*sizeof(float));   /* bf16 reference (committed storage) */
    for(size_t i=0;i<(size_t)NTOK*KV;i++) bref[i]=bf16f(f32bf(X[i]));
    float *Q = malloc((size_t)NTOK*KV*sizeof(float));

    /* ---- S1 per-token absmax ---- */
    for(int t=0;t<NTOK;t++){ const float*x=X+(size_t)t*KV; float*q=Q+(size_t)t*KV;
        float mx=0; for(int d=0;d<KV;d++){float a=x[d]<0?-x[d]:x[d]; if(a>mx)mx=a;}
        float s=mx/127.f, inv=s>0?1.f/s:0; for(int d=0;d<KV;d++){ int v=(int)lrintf(x[d]*inv); if(v>127)v=127; if(v<-127)v=-127; q[d]=v*s; } }
    Res s1=score(bref,Q, 1.0 + 4.0/KV);   /* 1B/elem + 1 f32 scale per 512-row */

    /* ---- S2 per-channel absmax ---- */
    for(int d=0;d<KV;d++){ float mx=0; for(int t=0;t<NTOK;t++){float a=X[(size_t)t*KV+d]; a=a<0?-a:a; if(a>mx)mx=a;}
        float s=mx/127.f, inv=s>0?1.f/s:0; for(int t=0;t<NTOK;t++){ int v=(int)lrintf(X[(size_t)t*KV+d]*inv); if(v>127)v=127; if(v<-127)v=-127; Q[(size_t)t*KV+d]=v*s; } }
    Res s2=score(bref,Q, 1.0 + 4.0/NTOK); /* 1B/elem + 1 f32 scale per channel (amortized over all tokens) */

    /* ---- S3 per-token int8 + bf16 sidecar for K largest-|.| dims of EACH token ---- */
    const int K=NCH_MASSIVE;
    for(int t=0;t<NTOK;t++){ const float*x=X+(size_t)t*KV; float*q=Q+(size_t)t*KV;
        /* find K largest-|.| indices (simple selection, K tiny) */
        int big[16]; for(int k=0;k<K;k++)big[k]=-1;
        for(int d=0;d<KV;d++){ float a=x[d]<0?-x[d]:x[d];
            for(int k=0;k<K;k++){ if(big[k]<0||a>fabsf(x[big[k]])){ for(int j=K-1;j>k;j--)big[j]=big[j-1]; big[k]=d; break; } } }
        char isbig[KV]; memset(isbig,0,sizeof isbig); float mx=0;
        for(int k=0;k<K;k++) if(big[k]>=0) isbig[big[k]]=1;
        for(int d=0;d<KV;d++) if(!isbig[d]){float a=x[d]<0?-x[d]:x[d]; if(a>mx)mx=a;}
        float s=mx/127.f, inv=s>0?1.f/s:0;
        for(int d=0;d<KV;d++){
            if(isbig[d]){ q[d]=bf16f(f32bf(x[d])); }                    /* sidecar: bf16-exact */
            else { int v=(int)lrintf(x[d]*inv); if(v>127)v=127; if(v<-127)v=-127; q[d]=v*s; }
        } }
    Res s3=score(bref,Q, 1.0 + 4.0/KV + (double)K*(2.0+2.0)/KV); /* int8 + scale + K*(bf16 val+int16 idx) */

    /* ---- S5 STATIC per-channel scale, calibrated on first 256 tokens (streaming-real) ----
     * Decode appends tokens one at a time => cannot absmax over future tokens. Calibrate a
     * fixed per-channel scale from an early window and apply it to ALL positions. This is
     * the implementable form of S2. */
    { const int CAL=256; float cs[KV];
      for(int d=0;d<KV;d++){ float mx=0; for(int t=0;t<CAL;t++){float a=X[(size_t)t*KV+d]; a=a<0?-a:a; if(a>mx)mx=a;}
          cs[d]=mx/127.f; }
      for(int d=0;d<KV;d++){ float s=cs[d], inv=s>0?1.f/s:0;
          for(int t=0;t<NTOK;t++){ int v=(int)lrintf(X[(size_t)t*KV+d]*inv); if(v>127)v=127; if(v<-127)v=-127; Q[(size_t)t*KV+d]=v*s; } } }
    Res s5=score(bref,Q, 1.0 + 4.0/NTOK);

    /* ---- S4 per-token robust (99.5th pct scale), outliers clamp ---- */
    for(int t=0;t<NTOK;t++){ const float*x=X+(size_t)t*KV; float*q=Q+(size_t)t*KV;
        float tmp[KV]; for(int d=0;d<KV;d++){tmp[d]=x[d]<0?-x[d]:x[d];}
        /* partial: pick the ~0.5% largest as the clip point => index KV*0.995 */
        for(int a=0;a<KV;a++)for(int b=a+1;b<KV;b++) if(tmp[b]<tmp[a]){float u=tmp[a];tmp[a]=tmp[b];tmp[b]=u;}
        float clip=tmp[(int)(KV*0.995f)]; float s=clip/127.f, inv=s>0?1.f/s:0;
        for(int d=0;d<KV;d++){ int v=(int)lrintf(x[d]*inv); if(v>127)v=127; if(v<-127)v=-127; q[d]=v*s; } }
    Res s4=score(bref,Q, 1.0 + 4.0/KV);

    printf("int8 KV-latent quant probe  (KV=%d, NTOK=%d, %d massive sink channels)\n", KV, NTOK, NCH_MASSIVE);
    printf("reference = bf16 (current kv_cache storage). rel-err over |x|>1e-3 dims.\n\n");
    printf("%-44s %10s %10s %8s\n","scheme","L1_rel","bad>5%%","B/elem");
    printf("%-44s %10.4f %9.2f%% %8.3f\n","S1 per-token absmax",          s1.l1rel,100*s1.badfrac,s1.bytes_per_elem);
    printf("%-44s %10.4f %9.2f%% %8.3f\n","S2 per-channel absmax (oracle)",s2.l1rel,100*s2.badfrac,s2.bytes_per_elem);
    printf("%-44s %10.4f %9.2f%% %8.3f\n","S3 per-token int8 + bf16 sidecar (K)",s3.l1rel,100*s3.badfrac,s3.bytes_per_elem);
    printf("%-44s %10.4f %9.2f%% %8.3f\n","S5 STATIC per-channel (calib 256, streaming)",s5.l1rel,100*s5.badfrac,s5.bytes_per_elem);
    printf("%-44s %10.4f %9.2f%% %8.3f\n","S4 per-token 99.5pct + clamp", s4.l1rel,100*s4.badfrac,s4.bytes_per_elem);
    printf("\nL1_rel = sum|q-a|/sum|a| vs bf16 ref. bad = %% of |a|>0.1 dims with >5%% pointwise err.\n");
    printf("bf16 ref itself is ~2e-3 L1_rel vs f32. int8 is fundamentally ~1/127=0.8%% per step;\n");
    printf("a scheme in the low-single-digit %% L1 range is the realistic int8 KV target.\n");
    free(X);free(bref);free(Q);
    return 0;
}
