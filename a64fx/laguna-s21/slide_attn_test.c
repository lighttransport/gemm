/* Isolated equivalence test: attention_slide_flash (query-blocked, used by chunked
 * prefill) must match attention_core (per-token reference) on a sliding layer.
 *
 * Both write K/V into the ring as they go, so the two runs are given identical
 * initial ring state by snapshotting and restoring the cache between them.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -DLAGUNA_FP8 -DLAGUNA_BENCH -I../../common -I../utofu-tests \
 *          -o slide_attn_test slide_attn_test.c -lm
 */
#define LAGUNA_BENCH
#include "laguna_s21_ep_runner.c"

static uint64_t rs_state = 0x243F6A8885A308D3ull;
static float frand(void){ rs_state = rs_state*6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((rs_state>>33)%2000)-1000)/1000.0f; }

/* one sliding layer, synthetic weights/inputs */
static int run_case(int pos0, int C, int nh, double *max_abs, double *rel) {
    int hd=LAGUNA_HEAD_DIM, kvs=LAGUNA_KV_HEADS*hd;
    int maxpos = pos0+C+16;

    laguna_model m; memset(&m,0,sizeof m);
    m.n_layers=1; m.max_pos=maxpos; m.ep_rank=0; m.ep_size=1;
    laguna_layer *ly=&m.layers[0];
    ly->is_sliding=1; ly->is_moe=0; ly->num_heads=nh;
    uint16_t *qn=malloc(hd*2), *kn=malloc(hd*2);
    for(int i=0;i<hd;i++){ qn[i]=laguna_f32_to_bf16(0.9f+0.2f*(i%7)/7.0f);
                           kn[i]=laguna_f32_to_bf16(0.8f+0.3f*(i%5)/5.0f); }
    ly->q_norm=qn; ly->k_norm=kn;

    int hf=LAGUNA_ROPE_FULL_DIM/2, hs=LAGUNA_ROPE_SLIDING_DIM/2;
    m.full_cos=malloc((size_t)maxpos*hf*4); m.full_sin=malloc((size_t)maxpos*hf*4);
    m.swa_cos =malloc((size_t)maxpos*hs*4); m.swa_sin =malloc((size_t)maxpos*hs*4);
    laguna_build_rope_tables(&m);

    int cap=LAGUNA_SLIDING_CAP;
    m.kv_cap[0]=cap; m.kv_off[0]=0;
    size_t kvel=(size_t)cap*kvs;
    m.kcache=malloc(kvel*2); m.vcache=malloc(kvel*2);
    for(size_t i=0;i<kvel;i++){ m.kcache[i]=laguna_f32_to_bf16(frand()*0.5f);
                                m.vcache[i]=laguna_f32_to_bf16(frand()*0.5f); }
    uint16_t *ksnap=malloc(kvel*2), *vsnap=malloc(kvel*2);
    memcpy(ksnap,m.kcache,kvel*2); memcpy(vsnap,m.vcache,kvel*2);

    laguna_scratch sc; scratch_alloc(&sc, maxpos);

    /* identical inputs for both paths */
    size_t qn_el=(size_t)C*nh*hd, kv_el=(size_t)C*kvs, g_el=(size_t)C*nh;
    float *Q0=malloc(qn_el*4),*K0=malloc(kv_el*4),*V0=malloc(kv_el*4),*G0=malloc(g_el*4);
    for(size_t i=0;i<qn_el;i++) Q0[i]=frand();
    for(size_t i=0;i<kv_el;i++) K0[i]=frand();
    for(size_t i=0;i<kv_el;i++) V0[i]=frand();
    for(size_t i=0;i<g_el;i++)  G0[i]=frand()*0.5f;
    float *Qa=malloc(qn_el*4),*Ka=malloc(kv_el*4),*Va=malloc(kv_el*4),*Ga=malloc(g_el*4);
    float *AOa=malloc(qn_el*4), *AOb=malloc(qn_el*4);

    /* A: per-token reference */
    memcpy(Qa,Q0,qn_el*4); memcpy(Ka,K0,kv_el*4); memcpy(Va,V0,kv_el*4); memcpy(Ga,G0,g_el*4);
    for (int c=0;c<C;++c)
        attention_core(&m,ly,&sc,0,pos0+c,nh, Qa+(size_t)c*nh*hd, Ka+(size_t)c*kvs,
                       Va+(size_t)c*kvs, Ga+(size_t)c*nh, AOa+(size_t)c*nh*hd);

    /* B: query-blocked, from the same initial ring */
    memcpy(m.kcache,ksnap,kvel*2); memcpy(m.vcache,vsnap,kvel*2);
    memcpy(Qa,Q0,qn_el*4); memcpy(Ka,K0,kv_el*4); memcpy(Va,V0,kv_el*4); memcpy(Ga,G0,g_el*4);
    attention_slide_flash(&m,ly,&sc,0,pos0,C,nh, Qa,Ka,Va,Ga, AOb);

    double mx=0, num=0, den=0;
    for (size_t i=0;i<qn_el;i++){ double d=fabs((double)AOa[i]-AOb[i]);
        if(d>mx)mx=d; num+=d*d; den+=(double)AOa[i]*AOa[i]; }
    *max_abs=mx; *rel=sqrt(num/(den>0?den:1));
    return 0;
}

/* Control: the same comparison with the window deliberately shifted by one
 * position, to show the tolerance is tight enough to catch a real indexing error
 * rather than merely loose enough to pass. */
static void run_bugged(int pos0, int C, int nh, double *max_abs, double *rel) {
    int hd=LAGUNA_HEAD_DIM, kvs=LAGUNA_KV_HEADS*hd, kv_groups=nh/LAGUNA_KV_HEADS;
    int maxpos=pos0+C+16, cap=LAGUNA_SLIDING_CAP, W=LAGUNA_SLIDING_WINDOW;
    laguna_model m; memset(&m,0,sizeof m);
    m.n_layers=1; m.max_pos=maxpos; laguna_layer *ly=&m.layers[0];
    ly->is_sliding=1; ly->num_heads=nh;
    uint16_t *qn=malloc(hd*2),*kn=malloc(hd*2);
    for(int i=0;i<hd;i++){qn[i]=laguna_f32_to_bf16(1.0f);kn[i]=laguna_f32_to_bf16(1.0f);}
    ly->q_norm=qn; ly->k_norm=kn;
    int hf=LAGUNA_ROPE_FULL_DIM/2,hs=LAGUNA_ROPE_SLIDING_DIM/2;
    m.full_cos=malloc((size_t)maxpos*hf*4);m.full_sin=malloc((size_t)maxpos*hf*4);
    m.swa_cos=malloc((size_t)maxpos*hs*4);m.swa_sin=malloc((size_t)maxpos*hs*4);
    laguna_build_rope_tables(&m);
    m.kv_cap[0]=cap; m.kv_off[0]=0; size_t kvel=(size_t)cap*kvs;
    m.kcache=malloc(kvel*2);m.vcache=malloc(kvel*2);
    for(size_t i=0;i<kvel;i++){m.kcache[i]=laguna_f32_to_bf16(frand()*0.5f);
                               m.vcache[i]=laguna_f32_to_bf16(frand()*0.5f);}
    laguna_scratch sc; scratch_alloc(&sc,maxpos);
    size_t qe=(size_t)C*nh*hd;
    float *Q=malloc(qe*4),*AO=malloc(qe*4),*AOref=malloc(qe*4);
    for(size_t i=0;i<qe;i++) Q[i]=frand();
    float scale=1.0f/sqrtf((float)hd);
    /* correct vs off-by-one window, scored directly (no KV writes involved) */
    for (int shift=0; shift<2; ++shift) {
        float *out = shift? AO : AOref;
        for (int h=0;h<nh;++h) for (int c=0;c<C;++c) {
            int kvh=h/kv_groups, lo=pos0+c-(W-1)+shift, hi=pos0+c; if(lo<0)lo=0;
            const float *q=Q+(size_t)c*nh*hd+(size_t)h*hd;
            float mx=-INFINITY, *sco=sc.scores; int n=hi-lo+1;
            for(int i=0;i<n;i++){ float d=laguna_qkdot(q,m.kcache+(size_t)((lo+i)%cap)*kvs+(size_t)kvh*hd,hd)*scale;
                sco[i]=d; if(d>mx)mx=d; }
            float l=laguna_exp_shift_sum(sco,n,mx);
            float acc[LAGUNA_HEAD_DIM]; for(int d=0;d<hd;d++)acc[d]=0;
            for(int i=0;i<n;i++) laguna_vaxpy(acc,m.vcache+(size_t)((lo+i)%cap)*kvs+(size_t)kvh*hd,sco[i],1.0f,hd);
            float *o=out+(size_t)c*nh*hd+(size_t)h*hd; for(int d=0;d<hd;d++)o[d]=acc[d]/l;
        }
    }
    double mxd=0,num=0,den=0;
    for(size_t i=0;i<qe;i++){double d=fabs((double)AOref[i]-AO[i]); if(d>mxd)mxd=d;
        num+=d*d; den+=(double)AOref[i]*AOref[i];}
    *max_abs=mxd; *rel=sqrt(num/(den>0?den:1));
}

int main(void) {
    struct { int pos0, C; const char *what; } cases[] = {
        {   0,  16, "chunk at position 0 (window clamps at 0)" },
        {   0, 256, "full chunk from 0" },
        { 100, 256, "window still clamped for early queries" },
        { 512, 256, "window exactly full, no ring wrap yet" },
        { 700, 256, "ring wraps mid-chunk" },
        { 768, 256, "chunk starts exactly at ring wrap" },
        {1023, 256, "wrap straddles chunk boundary" },
        {4096, 256, "steady state deep in context" },
        {4096,   7, "short tail chunk" },
        {9999, 129, "odd offsets, odd chunk" },
    };
    int bad=0;
    for (size_t i=0;i<sizeof cases/sizeof *cases;i++) {
        for (int nh_i=0; nh_i<2; ++nh_i) {
            int nh = nh_i? LAGUNA_SLIDING_HEADS : LAGUNA_FULL_HEADS;
            double mx,rel; run_case(cases[i].pos0,cases[i].C,nh,&mx,&rel);
            /* Tolerance: the two paths sum the same 512 terms in different orders
             * (block-wise online softmax vs a single max-then-sum pass), so the
             * floor is fp32 eps * sqrt(512) ~ 2.7e-6 -- which is what we measure
             * (3.1e-6).  A real indexing error is O(1e-1); see the control below. */
            int ok = (rel < 1e-5) && (mx < 2e-4);
            printf("%-45s nh=%2d pos0=%5d C=%3d  max|d|=%.3e rel=%.3e  %s\n",
                   cases[i].what, nh, cases[i].pos0, cases[i].C, mx, rel, ok?"OK":"** FAIL **");
            if(!ok) bad=1;
        }
    }
    { double mx,rel; run_bugged(4096,256,LAGUNA_SLIDING_HEADS,&mx,&rel);
      int detects = (rel >= 1e-5);
      printf("\ncontrol: window shifted by ONE position -> max|d|=%.3e rel=%.3e  "
             "(tolerance %s detect this)\n", mx, rel, detects?"DOES":"** DOES NOT **");
      if(!detects){ puts("tolerance is too loose to be meaningful"); bad=1; } }
    puts(bad ? "slide-attn equivalence: FAIL" : "slide-attn equivalence: PASS");
    return bad;
}
