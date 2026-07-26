/* Equivalence test for attention_full_flash (chunked prefill, full-attention
 * layers) against attention_core, the per-token reference.
 *
 * The sliding path has slide_attn_test; this covers the full path, in particular
 * the causal diagonal block, which is the part that is easy to get wrong: a query
 * must see keys [pos0, pos0+c] and no further, and must see them in increasing
 * order for the online softmax to be valid.
 *
 * Both paths write K/V into the cache as they go, so the two runs are given
 * identical initial state by snapshotting and restoring it.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *          -DLAGUNA_FP8 -I../../common -I../utofu-tests -o full_attn_test full_attn_test.c -lm
 */
#define LAGUNA_BENCH
#include "laguna_s21_ep_runner.c"

static uint64_t rs_state = 0x9E3779B97F4A7C15ull;
static float frand(void){ rs_state = rs_state*6364136223846793005ull + 1442695040888963407ull;
    return (float)((int)((rs_state>>33)%2000)-1000)/1000.0f; }

static void run_case(int pos0, int C, int nh, double *max_abs, double *rel, int shift) {
    int hd=LAGUNA_HEAD_DIM, kvs=LAGUNA_KV_HEADS*hd;
    int maxpos = pos0+C+16;

    laguna_model m; memset(&m,0,sizeof m);
    m.n_layers=1; m.max_pos=maxpos; m.ep_rank=0; m.ep_size=1;
    laguna_layer *ly=&m.layers[0];
    ly->is_sliding=0; ly->is_moe=0; ly->num_heads=nh;   /* FULL attention layer */
    uint16_t *qn=malloc(hd*2), *kn=malloc(hd*2);
    for(int i=0;i<hd;i++){ qn[i]=laguna_f32_to_bf16(0.9f+0.2f*(i%7)/7.0f);
                           kn[i]=laguna_f32_to_bf16(0.8f+0.3f*(i%5)/5.0f); }
    ly->q_norm=qn; ly->k_norm=kn;

    int hf=LAGUNA_ROPE_FULL_DIM/2, hs=LAGUNA_ROPE_SLIDING_DIM/2;
    m.full_cos=malloc((size_t)maxpos*hf*4); m.full_sin=malloc((size_t)maxpos*hf*4);
    m.swa_cos =malloc((size_t)maxpos*hs*4); m.swa_sin =malloc((size_t)maxpos*hs*4);
    laguna_build_rope_tables(&m);

    m.kv_cap[0]=maxpos; m.kv_off[0]=0;              /* full layer: no ring */
    size_t kvel=(size_t)maxpos*kvs;
    m.kcache=malloc(kvel*2); m.vcache=malloc(kvel*2);
    for(size_t i=0;i<kvel;i++){ m.kcache[i]=laguna_f32_to_bf16(frand()*0.5f);
                                m.vcache[i]=laguna_f32_to_bf16(frand()*0.5f); }
    uint16_t *ksnap=malloc(kvel*2), *vsnap=malloc(kvel*2);
    memcpy(ksnap,m.kcache,kvel*2); memcpy(vsnap,m.vcache,kvel*2);

    laguna_scratch sc; scratch_alloc(&sc, maxpos);

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

    /* B: query-blocked flash, from the same initial cache */
    memcpy(m.kcache,ksnap,kvel*2); memcpy(m.vcache,vsnap,kvel*2);
    memcpy(Qa,Q0,qn_el*4); memcpy(Ka,K0,kv_el*4); memcpy(Va,V0,kv_el*4); memcpy(Ga,G0,g_el*4);
    attention_full_flash(&m,ly,&sc,0,pos0,C,nh, Qa,Ka,Va,Ga, AOb);

    /* optional deliberate corruption, to prove the tolerance can detect a bug:
     * drop the last diagonal key of every query (an off-by-one in the causal cut) */
    if (shift) for (size_t i=0;i<qn_el;i+=nh*hd) AOb[i] *= 1.0f + 1e-3f;

    double mx=0, num=0, den=0;
    for (size_t i=0;i<qn_el;i++){ double d=fabs((double)AOa[i]-AOb[i]);
        if(d>mx)mx=d; num+=d*d; den+=(double)AOa[i]*AOa[i]; }
    *max_abs=mx; *rel=sqrt(num/(den>0?den:1));
}

/* control: causal cut moved by one key, scored directly against the reference */
static void run_offbyone(int pos0, int C, int nh, double *rel) {
    int hd=LAGUNA_HEAD_DIM, kvs=LAGUNA_KV_HEADS*hd, kv_groups=nh/LAGUNA_KV_HEADS;
    int maxpos=pos0+C+16;
    laguna_model m; memset(&m,0,sizeof m);
    m.n_layers=1; m.max_pos=maxpos; laguna_layer *ly=&m.layers[0];
    ly->is_sliding=0; ly->num_heads=nh;
    uint16_t *qn=malloc(hd*2),*kn=malloc(hd*2);
    for(int i=0;i<hd;i++){qn[i]=laguna_f32_to_bf16(1.0f);kn[i]=laguna_f32_to_bf16(1.0f);}
    ly->q_norm=qn; ly->k_norm=kn;
    int hf=LAGUNA_ROPE_FULL_DIM/2,hs=LAGUNA_ROPE_SLIDING_DIM/2;
    m.full_cos=malloc((size_t)maxpos*hf*4);m.full_sin=malloc((size_t)maxpos*hf*4);
    m.swa_cos=malloc((size_t)maxpos*hs*4);m.swa_sin=malloc((size_t)maxpos*hs*4);
    laguna_build_rope_tables(&m);
    m.kv_cap[0]=maxpos; m.kv_off[0]=0; size_t kvel=(size_t)maxpos*kvs;
    m.kcache=malloc(kvel*2);m.vcache=malloc(kvel*2);
    for(size_t i=0;i<kvel;i++){m.kcache[i]=laguna_f32_to_bf16(frand()*0.5f);
                               m.vcache[i]=laguna_f32_to_bf16(frand()*0.5f);}
    laguna_scratch sc; scratch_alloc(&sc,maxpos);
    size_t qe=(size_t)C*nh*hd;
    float *Q=malloc(qe*4),*AO=malloc(qe*4),*AOref=malloc(qe*4);
    for(size_t i=0;i<qe;i++) Q[i]=frand();
    float scale=1.0f/sqrtf((float)hd);
    for (int drop=0; drop<2; ++drop) {
        float *out = drop? AO : AOref;
        for (int h=0;h<nh;++h) for (int c=0;c<C;++c) {
            int kvh=h/kv_groups, hi=pos0+c-drop;   /* drop=1: one key short */
            const float *q=Q+(size_t)c*nh*hd+(size_t)h*hd;
            float mx=-INFINITY, *sco=sc.scores; int n=hi+1;
            if (n<1) n=1;
            for(int i=0;i<n;i++){ float d=laguna_qkdot(q,m.kcache+(size_t)i*kvs+(size_t)kvh*hd,hd)*scale;
                sco[i]=d; if(d>mx)mx=d; }
            float l=laguna_exp_shift_sum(sco,n,mx);
            float acc[LAGUNA_HEAD_DIM]; for(int d=0;d<hd;d++)acc[d]=0;
            for(int i=0;i<n;i++) laguna_vaxpy(acc,m.vcache+(size_t)i*kvs+(size_t)kvh*hd,sco[i],1.0f,hd);
            float *o=out+(size_t)c*nh*hd+(size_t)h*hd; for(int d=0;d<hd;d++)o[d]=acc[d]/l;
        }
    }
    double num=0,den=0;
    for(size_t i=0;i<qe;i++){double d=(double)AOref[i]-AO[i];num+=d*d;den+=(double)AOref[i]*AOref[i];}
    *rel=sqrt(num/(den>0?den:1));
}

int main(void) {
    struct { int pos0, C; const char *what; } cases[] = {
        {    0,   1, "single token at position 0" },
        {    0,   8, "tiny chunk from 0 (all diagonal)" },
        {    0, 256, "full chunk from 0 (all diagonal, spans KB blocks)" },
        {    0, 129, "odd chunk from 0" },
        {  128, 256, "prefix shorter than one key block" },
        {  256, 256, "prefix exactly one block" },
        { 1000, 256, "prefix not a block multiple" },
        { 4096,  17, "short tail chunk deep in context" },
        { 4096, 256, "steady state" },
        { 9999, 129, "odd offsets, odd chunk" },
    };
    int bad=0;
    for (size_t i=0;i<sizeof cases/sizeof *cases;i++) {
        for (int nh_i=0; nh_i<2; ++nh_i) {
            int nh = nh_i? LAGUNA_SLIDING_HEADS : LAGUNA_FULL_HEADS;
            double mx,rel; run_case(cases[i].pos0,cases[i].C,nh,&mx,&rel,0);
            /* the two paths sum the same keys in different orders (block-wise
             * online softmax vs one max-then-sum pass), so the floor is fp32
             * reassociation; an off-by-one in the causal cut is O(1e-2), below. */
            int ok = (rel < 1e-5) && (mx < 2e-4);
            printf("%-50s nh=%2d pos0=%5d C=%3d  max|d|=%.3e rel=%.3e  %s\n",
                   cases[i].what, nh, cases[i].pos0, cases[i].C, mx, rel, ok?"OK":"** FAIL **");
            if(!ok) bad=1;
        }
    }
    { double rel; run_offbyone(4096,256,LAGUNA_SLIDING_HEADS,&rel);
      int detects = rel >= 1e-5;
      printf("\ncontrol: causal cut one key short -> rel=%.3e  (tolerance %s detect it)\n",
             rel, detects?"DOES":"** DOES NOT **");
      if(!detects){ puts("tolerance too loose to be meaningful"); bad=1; } }
    puts(bad ? "full-attn equivalence: FAIL" : "full-attn equivalence: PASS");
    return bad;
}
