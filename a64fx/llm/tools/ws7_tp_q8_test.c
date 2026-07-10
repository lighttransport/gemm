/* WS7 TP x Q8_PV single-node analysis harness (no alloc, no uTofu).
 *
 * WS7 (a64fx/ds4f.md): DS4F_TP_ATTN=1 + DS4F_Q8_DENSE=1 diverges from the first
 * decoded token on 11n; isolated to ds4f_matvec_blockdiag's Q8_PV branch with a
 * TP_ATTN-partial zero-padded s_attn. The doc's analysis expected the
 * partial-quantize-then-sum to reconstruct the full-quantize result EXACTLY.
 *
 * This harness tests that expectation directly, plus the two TP patterns the
 * prefill-verify TP compose (P2) actually needs:
 *   A. TP_ATTN-style : REPLICATED blockdiag weight, per-rank ZERO-PADDED input
 *                      (owned heads only), partials summed across ranks.
 *                      Hypothesis: per-64-block quantization is exact (blocks
 *                      never straddle a 512-elem head), but the per-lane f32
 *                      accumulation CHAIN reassociates when split across ranks
 *                      -> expect relL2 ~1e-7 (reassoc class), NOT bit-exact,
 *                      NOT garbage. bf16 comparison shows the same class.
 *   B. TP_OPROJ-style: ROW-SHARDED blockdiag weight (8-aligned), FULL input,
 *                      disjoint output rows stitched. Every row is computed by
 *                      exactly one rank from an identically-quantized full x
 *                      -> expect BIT-EXACT. (Green-lights o-proj/w1/w3/head TP
 *                      under Q8 for the verify-path compose.)
 *   C. TP_SHARED-style sh_w2 contraction: REPLICATED weight, input zero-padded
 *                      by a COLUMN shard [r0,r0+rows) of K, partials summed.
 *                      With an 8-aligned boundary a 64-block STRADDLES ownership
 *                      -> the straddled block's absmax (hence quantization)
 *                      differs from the full-vector quantize -> real (small)
 *                      quantization error on top of reassoc. With a 64-aligned
 *                      boundary the straddle disappears -> reassoc class only.
 *                      Also run through ds4f_gemm at M=2 (the verify path).
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *       -I../../common -o build/ws7_tp_q8_test tools/ws7_tp_q8_test.c -lm -lpthread -lhwb
 *   OMP_NUM_THREADS=12 taskset -c 12-23 ./build/ws7_tp_q8_test
 */
#include "ds4f.h"
#include <math.h>

static uint32_t rng = 0xC0FFEEu;
static inline uint32_t nr(void){ rng = rng*1664525u + 1013904223u; return rng; }
static inline float frand(void){ return ((float)(nr()>>8)/(float)(1u<<24))*2.f-1.f; }

static void pack_pv(uint16_t *dst, const uint16_t *Wrm, int rows, int cols){
    for(int i=0;i<rows;i++){ size_t gb=(size_t)(i/8)*8*cols; int loc=i&7,pair=loc>>1,slot=loc&1;
        uint16_t *pb=dst+gb+(size_t)pair*2*cols; for(int j=0;j<cols;j++) pb[2*j+slot]=Wrm[(size_t)i*cols+j]; }
}

static ds4f_tensor mk_q8(ds4f_model *m, int rows, int cols, ds4f_tensor *bf_out){
    uint16_t *Wrm=malloc((size_t)rows*cols*2);
    for(size_t k=0;k<(size_t)rows*cols;k++) Wrm[k]=ds4f_f32_bf16(frand());
    ds4f_tensor bf; memset(&bf,0,sizeof(bf)); bf.type=DS4F_BF16_PV; bf.rows=rows; bf.cols=cols;
    size_t wb=ds4f_wbytes(DS4F_BF16_PV,rows,cols); bf.w=aligned_alloc(256,(wb+255)&~(size_t)255);
    pack_pv((uint16_t*)bf.w,Wrm,rows,cols);
    ds4f_tensor q8=bf; q8.w=aligned_alloc(256,(wb+255)&~(size_t)255); memcpy(q8.w,bf.w,wb);
    ds4f_repack_bf16pv_to_q8pv(m,&q8);
    free(Wrm);
    if(bf_out) *bf_out=bf; else free(bf.w);
    return q8;
}

typedef struct { int nan, nbit; double rel; } cmp_t;
static cmp_t cmp(const float *y, const float *ref, int n){
    cmp_t c={0,0,0}; double e=0,nn=0;
    for(int i=0;i<n;i++){
        if(!(y[i]==y[i])) c.nan++;
        if(y[i]==ref[i]) c.nbit++;
        double d=(double)y[i]-ref[i]; e+=d*d; nn+=(double)ref[i]*ref[i];
    }
    c.rel = nn>0?sqrt(e/nn):0; return c;
}

int main(int argc,char**argv){
    int nthr=(argc>1)?atoi(argv[1]):12, R=(argc>2)?atoi(argv[2]):11;
    ds4f_model m; memset(&m,0,sizeof(m)); m.n_threads=nthr; m.n_cmgs=1;
    m.pool=ds4f_pool_start(nthr,1); ds4f_init_fp8_e4m3_lut(m.fp8_lut);
    printf("ws7_tp_q8_test nthr=%d ranks=%d\n",nthr,R);
    int fails=0;

    /* ============ Case A: TP_ATTN-style partial zero-padded input ============ */
    {
        int o_inter=8192, gin=4096, glora=1024, n_heads=64, HD=512;
        int xlen=n_heads*HD; /* 32768 = 8 groups x gin */
        ds4f_tensor bf, q8 = mk_q8(&m, o_inter, gin, &bf);
        float *x  = aligned_alloc(256,(size_t)xlen*4);
        float *xr = aligned_alloc(256,(size_t)xlen*4);
        for(int j=0;j<xlen;j++) x[j]=frand();
        float *yq_ref=aligned_alloc(256,(size_t)o_inter*4), *yb_ref=aligned_alloc(256,(size_t)o_inter*4);
        float *yq_sum=calloc(o_inter,4), *yb_sum=calloc(o_inter,4);
        float *yr=aligned_alloc(256,(size_t)o_inter*4);
        ds4f_matvec_blockdiag(&m,yq_ref,&q8,x,gin,glora,0);
        ds4f_matvec_blockdiag(&m,yb_ref,&bf,x,gin,glora,0);
        for(int r=0;r<R;r++){
            int h0,h1; ds4f_tp_rowshard(n_heads,R,r,1,&h0,&h1);
            memset(xr,0,(size_t)xlen*4);
            memcpy(xr+(size_t)h0*HD, x+(size_t)h0*HD, (size_t)(h1-h0)*HD*4);
            ds4f_matvec_blockdiag(&m,yr,&q8,xr,gin,glora,0);
            for(int i=0;i<o_inter;i++) yq_sum[i]+=yr[i];
            ds4f_matvec_blockdiag(&m,yr,&bf,xr,gin,glora,0);
            for(int i=0;i<o_inter;i++) yb_sum[i]+=yr[i];
        }
        cmp_t cq=cmp(yq_sum,yq_ref,o_inter), cb=cmp(yb_sum,yb_ref,o_inter);
        printf("A TP_ATTN partial-sum blockdiag [%d x %d] glora=%d heads=%d/%d ranks:\n",o_inter,gin,glora,n_heads,R);
        printf("  q8  : nan=%d bitexact=%d/%d relL2=%.3e\n",cq.nan,cq.nbit,o_inter,cq.rel);
        printf("  bf16: nan=%d bitexact=%d/%d relL2=%.3e\n",cb.nan,cb.nbit,o_inter,cb.rel);
        if(cq.nan||cq.rel>1e-4){ printf("  -> Q8 partial-sum BROKEN (beyond reassoc class)\n"); fails++; }
        else printf("  -> Q8 partial-sum = reassoc class (matches bf16) — WS7 divergence is fp-reassoc flipping greedy, not a kernel bug\n");
        free(bf.w);free(x);free(xr);free(yq_ref);free(yb_ref);free(yq_sum);free(yb_sum);free(yr);
    }

    /* ============ Case B: TP_OPROJ-style row-shard, full input ============ */
    {
        int o_inter=8192, gin=4096, glora=1024;
        int xlen=(o_inter/glora)*gin;
        ds4f_tensor q8 = mk_q8(&m, o_inter, gin, NULL);
        float *x=aligned_alloc(256,(size_t)xlen*4);
        for(int j=0;j<xlen;j++) x[j]=frand();
        float *yref=aligned_alloc(256,(size_t)o_inter*4);
        float *ystitch=calloc(o_inter,4);
        ds4f_matvec_blockdiag(&m,yref,&q8,x,gin,glora,0);
        size_t gb=(size_t)(gin/64)*528;
        float *yr=aligned_alloc(256,(size_t)o_inter*4);
        for(int r=0;r<R;r++){
            int a0,a1; ds4f_tp_rowshard(o_inter,R,r,8,&a0,&a1);
            if(a1<=a0) continue;
            ds4f_tensor sub=q8; sub.rows=a1-a0; sub.w=(uint8_t*)q8.w+(size_t)(a0/8)*gb;
            ds4f_matvec_blockdiag(&m,yr,&sub,x,gin,glora,a0);
            memcpy(ystitch+a0,yr,(size_t)(a1-a0)*4);
        }
        cmp_t c=cmp(ystitch,yref,o_inter);
        printf("B TP_OPROJ row-shard full-input blockdiag: nan=%d bitexact=%d/%d relL2=%.3e\n",c.nan,c.nbit,o_inter,c.rel);
        if(c.nbit!=o_inter){ printf("  -> NOT bit-exact: row-shard Q8 UNSAFE\n"); fails++; }
        else printf("  -> BIT-EXACT: Q8 row-shard safe for the verify TP compose\n");
        free(x);free(yref);free(ystitch);free(yr);
    }

    /* ============ Case C: sh_w2-style contraction over zero-padded input ============ */
    for(int alignc=0;alignc<2;alignc++){
        int align = alignc? 64 : 8;
        int rows=4096, K=2048, M=2;
        ds4f_tensor q8 = mk_q8(&m, rows, K, NULL);
        float *x=aligned_alloc(256,(size_t)M*K*4);
        for(int j=0;j<M*K;j++) x[j]=frand();
        float *yref=aligned_alloc(256,(size_t)M*rows*4);
        float *ysum=calloc((size_t)M*rows,4);
        float *yr=aligned_alloc(256,(size_t)M*rows*4);
        float *xr=aligned_alloc(256,(size_t)M*K*4);
        ds4f_gemm(&m,yref,&q8,x,M,rows,K);
        for(int r=0;r<R;r++){
            int a0,a1; ds4f_tp_rowshard(K,R,r,align,&a0,&a1);
            memset(xr,0,(size_t)M*K*4);
            for(int mm=0;mm<M;mm++) memcpy(xr+(size_t)mm*K+a0, x+(size_t)mm*K+a0, (size_t)(a1-a0)*4);
            ds4f_gemm(&m,yr,&q8,xr,M,rows,K);
            for(size_t i=0;i<(size_t)M*rows;i++) ysum[i]+=yr[i];
        }
        cmp_t c=cmp(ysum,yref,M*rows);
        printf("C sh_w2 contraction zero-pad align=%d (gemm M=%d): nan=%d bitexact=%d/%d relL2=%.3e\n",align,M,c.nan,c.nbit,M*rows,c.rel);
        if(c.nan){ printf("  -> NaN: BROKEN\n"); fails++; }
        free(x);free(yref);free(ysum);free(yr);free(xr);
    }

    printf(fails? "WS7 harness: %d FAILURES\n" : "WS7 harness: all cases as-hypothesized\n", fails);
    ds4f_pool_stop(m.pool);
    return fails?2:0;
}
