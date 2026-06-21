/* ds4f_gemm_test.c — validate the batched (M>1) prefill GEMM in common/ds4f.h
 * (ds4f_gemm) against the single-token matvec (ds4f_matvec) it must reproduce.
 *
 * For each dtype under test we:
 *   1. build a [rows,cols] tensor with random bf16-representable weights,
 *   2. build M random token vectors X[M][cols],
 *   3. compute the reference Y_ref[M][rows] = M independent ds4f_matvec calls,
 *   4. compute Y_gemm[M][rows] = ONE ds4f_gemm call (token-major),
 *   5. report max-abs / max-rel diff.
 * The K-tile reassociation in the GEMM makes the result bit-SIMILAR, not
 * bit-identical, so the gate is max-abs < 1e-3 (values are O(sqrt(K)) here).
 *
 * No cluster, no weights on disk — runs on this native A64FX node.
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *       -D_GNU_SOURCE -I../../common -o build/ds4f_gemm_test \
 *       ds4f_gemm_test.c -lm -lpthread -lhwb
 */
#include "ds4f.h"
#include <math.h>
#ifdef DS4F_FAPP
#include "fj_tool/fapp.h"
#else
#define fapp_start(n,a,b) ((void)0)
#define fapp_stop(n,a,b)  ((void)0)
#endif

/* deterministic LCG so runs are reproducible (Date/rand-free) */
static uint32_t rng = 0x12345678u;
static inline float frand(void) {        /* ~U(-1,1) */
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}
/* one exp15/NaN-free E4M3 byte (so the magic decode == LUT decode bit-exactly,
 * mirroring rand_fp8 in ds4f_decode_bw_bench.c): if exp bits are all-ones, clear
 * one so exp<=14. Lets the magic GEMM dequant be validated against the LUT gather. */
static inline uint8_t rand_fp8_byte(void) {
    rng = rng * 1664525u + 1013904223u;
    uint8_t b = (uint8_t)(rng >> 13);
    if ((b & 0x78) == 0x78) b &= ~0x08;
    return b;
}
static inline uint64_t rdcyc(void){ uint64_t v; __asm__ __volatile__("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ __volatile__("mrs %0, cntfrq_el0":"=r"(v)); return v; }

/* pack logical row-major W[rows][cols] of bf16 halfwords into a DS4F_BF16_PV
 * tensor's pair-interleaved buffer (mirrors ds4f_synth_worker / ds4f_promote). */
static void pack_pv(uint16_t *dst, const uint16_t *Wrm, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        size_t gbase = (size_t)(i / 8) * 8 * cols;
        int local = i & 7, pair = local >> 1, slot = local & 1;
        uint16_t *pb = dst + gbase + (size_t)pair * 2 * cols;
        for (int j = 0; j < cols; j++) pb[2*j + slot] = Wrm[(size_t)i*cols + j];
    }
}

static int run_case(ds4f_model *m, ds4f_qtype type, int rows, int cols, int M) {
    /* logical bf16 weights (random, bf16-representable) */
    uint16_t *Wrm = (uint16_t *)malloc((size_t)rows * cols * 2);
    for (size_t k = 0; k < (size_t)rows * cols; k++) Wrm[k] = ds4f_f32_bf16(frand());

    int is_q8 = (type == DS4F_Q8_PV);
    ds4f_tensor t; memset(&t, 0, sizeof(t));
    /* Q8 builds bf16-pv first (for the high-precision reference) then repacks. */
    t.type = is_q8 ? DS4F_BF16_PV : type; t.rows = rows; t.cols = cols; t.scale = NULL;
    size_t wb = ds4f_wbytes(t.type, rows, cols);
    void *worig = aligned_alloc(256, (wb + 255) & ~(size_t)255);
    t.w = worig;
    if (t.type == DS4F_BF16) memcpy(t.w, Wrm, (size_t)rows * cols * 2);
    else if (t.type == DS4F_BF16_PV) pack_pv((uint16_t *)t.w, Wrm, rows, cols);
    else { fprintf(stderr, "unsupported test dtype %d\n", type); return 1; }

    /* M random token vectors */
    float *X = (float *)aligned_alloc(256, (size_t)M * cols * 4);
    for (size_t k = 0; k < (size_t)M * cols; k++) X[k] = frand();

    /* reference: M independent single-token matvecs (bf16-pv high precision) */
    float *Yref = (float *)aligned_alloc(256, (size_t)M * rows * 4);
    for (int mm = 0; mm < M; mm++)
        ds4f_matvec(m, Yref + (size_t)mm * rows, &t, X + (size_t)mm * cols);

    /* Q8: repack the (bf16-pv) weights to int8 W8A8 in place before the GEMM. */
    if (is_q8) ds4f_repack_bf16pv_to_q8pv(m, &t);

    /* batched GEMM, token-major (Ystride=rows, Xstride=cols) — warm then timed */
    float *Ygemm = (float *)aligned_alloc(256, (size_t)M * rows * 4);
    double freq = (double)rdfreq();
    ds4f_gemm(m, Ygemm, &t, X, M, rows, cols);                                        /* warm */
    int niter = (getenv("DS4F_GEMM_NITER") ? atoi(getenv("DS4F_GEMM_NITER")) : 20); double best = 1e30;  /* min over iters */
    fapp_start("gemm", 1, 0);
    for (int it = 0; it < niter; it++) {
        uint64_t tc0 = rdcyc(); ds4f_gemm(m, Ygemm, &t, X, M, rows, cols); uint64_t tc1 = rdcyc();
        double ms = (double)(tc1 - tc0) / freq * 1e3; if (ms < best) best = ms;
    }
    fapp_stop("gemm", 1, 0);
    double ms_g = best;
    double gmac = (double)M * rows * cols / 1e9;

    double maxabs = 0.0, maxrel = 0.0; int nbad = 0;
    for (size_t k = 0; k < (size_t)M * rows; k++) {
        double a = Yref[k], b = Ygemm[k], d = fabs(a - b);
        if (d > maxabs) maxabs = d;
        double rel = d / (fabs(a) + 1e-6);
        if (rel > maxrel) maxrel = rel;
        if (!isfinite(b)) nbad++;
    }
    /* per-token relative L2 error — the robust quality metric for a quantized
     * path (max-rel above is contaminated by near-zero individual entries on
     * random data). Worst token reported. */
    double maxrelL2 = 0.0;
    for (int mm = 0; mm < M; mm++) {
        const float *r = Yref + (size_t)mm*rows, *g = Ygemm + (size_t)mm*rows;
        double num = 0.0, den = 0.0;
        for (int i = 0; i < rows; i++) { double e = (double)r[i]-g[i]; num += e*e; den += (double)r[i]*r[i]; }
        double rl2 = sqrt(num) / (sqrt(den) + 1e-12);
        if (rl2 > maxrelL2) maxrelL2 = rl2;
    }
    /* argmax agreement (informational on random data — no logit margin, so a few
     * flips are expected noise; the real argmax check is in the full model). */
    int amatch = 0;
    for (int mm = 0; mm < M; mm++) {
        const float *r = Yref + (size_t)mm*rows, *g = Ygemm + (size_t)mm*rows;
        int ar = 0, ag = 0;
        for (int i = 1; i < rows; i++) { if (r[i] > r[ar]) ar = i; if (g[i] > g[ag]) ag = i; }
        if (ar == ag) amatch++;
    }
    const char *tn = is_q8 ? "Q8_PV" : (type == DS4F_BF16_PV ? "BF16_PV" : "BF16");
    int ok = is_q8 ? (maxrelL2 < 0.08 && nbad == 0)    /* int8 rel-L2 gate  */
                   : (maxabs < 1e-3 && nbad == 0);      /* bit-similar gate  */
    printf("  %-8s rows=%-6d cols=%-5d M=%-3d  %7.3fms  %6.1f Gmac/s  max-abs=%.3e  relL2=%.3e  argmax=%d/%d  nonfinite=%d  %s\n",
           tn, rows, cols, M, ms_g, ms_g > 0 ? gmac*1e3/ms_g : 0.0, maxabs, maxrelL2, amatch, M, nbad, ok ? "OK" : "FAIL");

    free(Wrm); free(worig);
    if (is_q8) munmap(t.w, ds4f_wbytes(DS4F_Q8_PV, rows, cols));
    free(X); free(Yref); free(Ygemm);
    return ok ? 0 : 1;
}

/* FP8 (DS4F_FP8) prefill GEMM correctness + throughput. This is the on-demand
 * dequant path used under DS4F_FP8_BF16=0 (no resident bf16 copy, -6 GB): the
 * GEMM reads each 8-row FP8 group once, dequants (LUT gather) into a small bf16
 * L1 tile, and reuses it across all M tokens via the register-blocked bf16-pv
 * kernel -- so the dequant amortizes over M and prefill stays compute-bound
 * WITHOUT a second resident bf16 rep. Reference = single-token gather matvecs
 * (f32, no bf16 truncation), so the GEMM is NOT bit-identical to the ref (it
 * truncates weights to bf16); the gate is argmax-exact + a bf16-magnitude relL2
 * bound, never bit-equality. NOTE: the GEMM uses gather, NOT the register magic
 * decode -- magic+FTZ flushes E4M3 subnormals (lossy, breaks bf16-predequant
 * parity) and the GEMM is compute-bound so it would be speed-neutral anyway
 * (measured: gather vs magic 1.00-1.06x). Magic's win is the M=1 decode matvec. */
static int run_fp8_case(ds4f_model *m, int rows, int cols, int M) {
    int K = cols;
    int ngrp128 = (rows + 127) / 128, sbc = (K + 127) / 128;
    uint8_t *W  = (uint8_t *)aligned_alloc(256, ((size_t)rows*K + 255) & ~(size_t)255);
    uint8_t *ES = (uint8_t *)aligned_alloc(256, ((size_t)ngrp128*sbc + 255) & ~(size_t)255);
    for (size_t i = 0; i < (size_t)rows*K; i++) W[i] = rand_fp8_byte();
    for (size_t i = 0; i < (size_t)ngrp128*sbc; i++) {
        rng = rng*1664525u + 1013904223u; ES[i] = (uint8_t)(125 + (rng>>20)%5);  /* E8M0 ~ 1.0 */
    }
    ds4f_tensor t; memset(&t, 0, sizeof(t));
    t.type = DS4F_FP8; t.rows = rows; t.cols = cols; t.w = W; t.scale = ES;

    float *X = (float *)aligned_alloc(256, (size_t)M*cols*4);
    for (size_t k = 0; k < (size_t)M*cols; k++) X[k] = frand();

    /* single-token f32 gather matvec reference (argmax + relL2 base) */
    float *Yref = (float *)aligned_alloc(256, (size_t)M*rows*4);
    for (int mm = 0; mm < M; mm++) ds4f_matvec(m, Yref + (size_t)mm*rows, &t, X + (size_t)mm*cols);

    float *Yg = (float *)aligned_alloc(256, (size_t)M*rows*4);
    double freq = (double)rdfreq();
    ds4f_gemm(m, Yg, &t, X, M, rows, cols);           /* warm */
    uint64_t t0 = rdcyc(); ds4f_gemm(m, Yg, &t, X, M, rows, cols); uint64_t t1 = rdcyc();

    double ms_g = (double)(t1-t0)/freq*1e3;
    double sd = 0.0, sr = 0.0; int nbad = 0, amatch = 0;   /* relL2 = ||Yg-Yref|| / ||Yref|| */
    for (size_t k = 0; k < (size_t)M*rows; k++) {
        double d = (double)Yg[k]-(double)Yref[k]; sd += d*d; sr += (double)Yref[k]*(double)Yref[k];
        if (!isfinite(Yg[k])) nbad++;
    }
    for (int mm = 0; mm < M; mm++) {
        const float *r = Yref + (size_t)mm*rows, *a = Yg + (size_t)mm*rows;
        int ar=0, aa=0; for (int i=1;i<rows;i++){ if(r[i]>r[ar])ar=i; if(a[i]>a[aa])aa=i; }
        if (ar==aa) amatch++;
    }
    double relL2 = sr>0 ? sqrt(sd/sr) : 0.0;
    double gmac = (double)rows*(double)K*(double)M;
    int ok = (amatch == M && nbad == 0 && relL2 < 5e-2);   /* argmax-exact; relL2 < bf16-trunc bound */
    printf("  FP8      rows=%-6d cols=%-5d M=%-3d  %7.2fms  %6.1f Gmac/s  relL2=%.2e argmax=%d/%d  %s\n",
           rows, cols, M, ms_g, ms_g>0?gmac/(ms_g*1e6):0.0, relL2, amatch, M, ok?"OK":"FAIL");
    free(W); free(ES); free(X); free(Yref); free(Yg);
    return ok ? 0 : 1;
}

/* MXFP4 (DS4F_MXFP4) expert GEMM correctness + throughput. The expert weights are
 * 2 fp4 nibbles/byte + per-32 E8M0 scale; the kernel (matvec_mxfp4_8row[_2x]) dequants
 * the nibbles in-register via svtbl (NO gather, NO LUT, NO resident bf16) -- already
 * the on-register SVE-ld+SVE-insn dequant the user asked for. GEMM is group-outer,
 * token-inner, 2-token register-blocked so each 8-row group's nibbles are HBM-read
 * once and the svtbl dequant amortizes across the M tokens. We feed random nibbles +
 * E8M0~=1.0; both the single-token matvec ref and the GEMM read the same bytes, so the
 * comparison is self-consistent (gate = argmax-exact + relL2; reassoc only). */
static int run_mxfp4_case(ds4f_model *m, int rows, int cols, int M) {
    int K = cols;
    size_t wb = (size_t)rows * K / 2, sb = (size_t)rows * K / 32;
    uint8_t *W  = (uint8_t *)aligned_alloc(256, (wb + 255) & ~(size_t)255);
    uint8_t *S  = (uint8_t *)aligned_alloc(256, (sb + 255) & ~(size_t)255);
    for (size_t i = 0; i < wb; i++) { rng = rng*1664525u + 1013904223u; W[i] = (uint8_t)(rng >> 24); }
    for (size_t i = 0; i < sb; i++) { rng = rng*1664525u + 1013904223u; S[i] = (uint8_t)(126 + (rng>>20)%3); } /* E8M0 ~ 1.0 */
    ds4f_tensor t; memset(&t, 0, sizeof(t));
    t.type = DS4F_MXFP4; t.rows = rows; t.cols = cols; t.w = W; t.scale = S;

    float *X = (float *)aligned_alloc(256, (size_t)M*cols*4);
    for (size_t k = 0; k < (size_t)M*cols; k++) X[k] = frand();

    float *Yref = (float *)aligned_alloc(256, (size_t)M*rows*4);
    for (int mm = 0; mm < M; mm++) ds4f_matvec(m, Yref + (size_t)mm*rows, &t, X + (size_t)mm*cols);

    float *Ys = (float *)aligned_alloc(256, (size_t)M*rows*4);   /* svtbl 2x path */
    float *Yt = (float *)aligned_alloc(256, (size_t)M*rows*4);   /* tile-dequant path */
    double freq = (double)rdfreq();
    m->mxfp4_gemm_tile = 0;                            /* svtbl per-token-pair */
    ds4f_gemm(m, Ys, &t, X, M, rows, cols);            /* warm */
    uint64_t t0 = rdcyc(); ds4f_gemm(m, Ys, &t, X, M, rows, cols); uint64_t t1 = rdcyc();
    m->mxfp4_gemm_tile = 1;                            /* tile-dequant (thr=1 => always on) */
    ds4f_gemm(m, Yt, &t, X, M, rows, cols);            /* warm */
    uint64_t t2 = rdcyc(); ds4f_gemm(m, Yt, &t, X, M, rows, cols); uint64_t t3 = rdcyc();
    m->mxfp4_gemm_tile = 0;

    double ms_s = (double)(t1-t0)/freq*1e3, ms_t = (double)(t3-t2)/freq*1e3;
    double sd = 0.0, sr = 0.0; int nbad = 0, amatch = 0;   /* tile-dequant vs f32 ref */
    for (size_t k = 0; k < (size_t)M*rows; k++) {
        double d = (double)Yt[k]-(double)Yref[k]; sd += d*d; sr += (double)Yref[k]*(double)Yref[k];
        if (!isfinite(Yt[k])) nbad++;
    }
    for (int mm = 0; mm < M; mm++) {
        const float *r = Yref + (size_t)mm*rows, *a = Yt + (size_t)mm*rows;
        int ar=0, aa=0; for (int i=1;i<rows;i++){ if(r[i]>r[ar])ar=i; if(a[i]>a[aa])aa=i; }
        if (ar==aa) amatch++;
    }
    double relL2 = sr>0 ? sqrt(sd/sr) : 0.0;
    double gmac = (double)rows*(double)K*(double)M;
    int ok = (amatch == M && nbad == 0 && relL2 < 5e-2);
    printf("  MXFP4    rows=%-6d cols=%-5d M=%-3d  svtbl=%7.2fms tile=%7.2fms %4.2fx  tile=%6.1f Gmac/s  relL2=%.2e argmax=%d/%d  %s\n",
           rows, cols, M, ms_s, ms_t, ms_t>0?ms_s/ms_t:0.0, ms_t>0?gmac/(ms_t*1e6):0.0, relL2, amatch, M, ok?"OK":"FAIL");
    free(W); free(S); free(X); free(Yref); free(Ys); free(Yt);
    return ok ? 0 : 1;
}

/* ===== Lever-2 fix(b) prototype: fp32 OUTER-PRODUCT GEMM =====================
 * Vectorize 16 rows per SVE f32 vector; keep OP_NT tokens' accumulators in registers
 * and broadcast the X scalars -> each weight vector W[16rows][k] is loaded ONCE per
 * token-block (M/OP_NT times) instead of the dot-product kernel's once-per-token-triple
 * (M/3) -> ~weight L1 traffic cut, targeting the ~30% LD_COMP_WAIT. Weights packed
 * [rowblk][K][16] bf16 (widened to f32 by svld1uh+lsl16). fp32 accumulate (lossless,
 * no fp16 overflow). OP_NT divides the bench M values {16,32,64}. */
#define OP_RB 16
#define OP_NT 8
typedef struct { float *Y; const uint16_t *W; const float *X; int rows, K, M, Ys, Xs; } op_task;
static void op_worker(void *arg, int tid, int nthr) {
    op_task *T = (op_task *)arg;
    int K = T->K, M = T->M, Xs = T->Xs, Ys = T->Ys, nrb = T->rows / OP_RB;
    int per = nrb/nthr, ex = nrb%nthr, b0 = per*tid + (tid<ex?tid:ex), b1 = b0 + per + (tid<ex?1:0);
    svbool_t pg = svptrue_b32();
    for (int rb = b0; rb < b1; rb++) {
        const uint16_t *Wb = T->W + (size_t)rb*K*OP_RB;
        int row = rb*OP_RB;
        for (int m0 = 0; m0 + OP_NT <= M; m0 += OP_NT) {   /* bench M in {16,32,64} -> exact multiple of OP_NT=8 */
            svfloat32_t c0=svdup_f32(0),c1=svdup_f32(0),c2=svdup_f32(0),c3=svdup_f32(0),
                        c4=svdup_f32(0),c5=svdup_f32(0),c6=svdup_f32(0),c7=svdup_f32(0);
            const float *xb = T->X + m0;             /* X is TRANSPOSED [K][M]: 8 tokens at k are contiguous */
            for (int k = 0; k < K; k++) {
                svfloat32_t wv = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, Wb + (size_t)k*OP_RB), 16));
                const float *xk = xb + (size_t)k*Xs;  /* Xs = M (transposed row stride) */
                c0=svmla_n_f32_x(pg,c0,wv,xk[0]); c1=svmla_n_f32_x(pg,c1,wv,xk[1]);
                c2=svmla_n_f32_x(pg,c2,wv,xk[2]); c3=svmla_n_f32_x(pg,c3,wv,xk[3]);
                c4=svmla_n_f32_x(pg,c4,wv,xk[4]); c5=svmla_n_f32_x(pg,c5,wv,xk[5]);
                c6=svmla_n_f32_x(pg,c6,wv,xk[6]); c7=svmla_n_f32_x(pg,c7,wv,xk[7]);
            }
            svst1_f32(pg,T->Y+(size_t)(m0+0)*Ys+row,c0); svst1_f32(pg,T->Y+(size_t)(m0+1)*Ys+row,c1);
            svst1_f32(pg,T->Y+(size_t)(m0+2)*Ys+row,c2); svst1_f32(pg,T->Y+(size_t)(m0+3)*Ys+row,c3);
            svst1_f32(pg,T->Y+(size_t)(m0+4)*Ys+row,c4); svst1_f32(pg,T->Y+(size_t)(m0+5)*Ys+row,c5);
            svst1_f32(pg,T->Y+(size_t)(m0+6)*Ys+row,c6); svst1_f32(pg,T->Y+(size_t)(m0+7)*Ys+row,c7);
        }
    }
}
static int run_op_case(ds4f_model *m, int rows, int cols, int M) {
    uint16_t *Wrm = (uint16_t *)malloc((size_t)rows*cols*2);
    for (size_t k=0;k<(size_t)rows*cols;k++) Wrm[k]=ds4f_f32_bf16(frand());
    int nrb = rows/OP_RB;
    uint16_t *Wop = (uint16_t *)aligned_alloc(256,(size_t)rows*cols*2);
    for (int rb=0;rb<nrb;rb++) for(int k=0;k<cols;k++) for(int r=0;r<OP_RB;r++)
        Wop[((size_t)rb*cols+k)*OP_RB+r]=Wrm[((size_t)(rb*OP_RB+r))*cols+k];
    float *X=(float*)aligned_alloc(256,(size_t)M*cols*4);
    for(size_t k=0;k<(size_t)M*cols;k++) X[k]=frand();
    /* reference: bf16-pv single-token matvec */
    ds4f_tensor t; memset(&t,0,sizeof(t)); t.type=DS4F_BF16_PV; t.rows=rows; t.cols=cols;
    t.w=aligned_alloc(256,(ds4f_wbytes(DS4F_BF16_PV,rows,cols)+255)&~(size_t)255);
    pack_pv((uint16_t*)t.w,Wrm,rows,cols);
    float *Yref=(float*)aligned_alloc(256,(size_t)M*rows*4);
    for(int mm=0;mm<M;mm++) ds4f_matvec(m,Yref+(size_t)mm*rows,&t,X+(size_t)mm*cols);
    float *Xt=(float*)aligned_alloc(256,(size_t)cols*M*4);   /* transpose X[M][cols] -> Xt[cols][M] */
    for(int mm=0;mm<M;mm++) for(int k=0;k<cols;k++) Xt[(size_t)k*M+mm]=X[(size_t)mm*cols+k];
    float *Yop=(float*)aligned_alloc(256,(size_t)M*rows*4);
    op_task T={Yop,Wop,Xt,rows,cols,M,rows,M};            /* Xs=M (transposed stride) */
    double freq=(double)rdfreq();
    ds4f_pool_run(m->pool,op_worker,&T);  /* warm */
    int niter=(getenv("DS4F_GEMM_NITER")?atoi(getenv("DS4F_GEMM_NITER")):20); double best=1e30;
    for(int it=0;it<niter;it++){uint64_t a=rdcyc();ds4f_pool_run(m->pool,op_worker,&T);uint64_t b=rdcyc();
        double ms=(double)(b-a)/freq*1e3; if(ms<best)best=ms;}
    double gmac=(double)M*rows*cols/1e9, maxrelL2=0;
    for(int mm=0;mm<M;mm++){const float*r=Yref+(size_t)mm*rows,*g=Yop+(size_t)mm*rows;
        double num=0,den=0;for(int i=0;i<rows;i++){double e=(double)r[i]-g[i];num+=e*e;den+=(double)r[i]*r[i];}
        double rl=sqrt(num)/(sqrt(den)+1e-12);if(rl>maxrelL2)maxrelL2=rl;}
    printf("  OP_fp32  rows=%-6d cols=%-5d M=%-3d  %7.3fms  %6.1f Gmac/s  relL2=%.3e  %s\n",
           rows,cols,M,best,best>0?gmac*1e3/best:0,maxrelL2, maxrelL2<1e-3?"OK":"FAIL");
    free(Wrm);free(Wop);free(X);free(Xt);free(t.w);free(Yref);free(Yop);
    return maxrelL2<1e-3?0:1;
}

int main(int argc, char **argv) {
    int nthr = (argc > 1) ? atoi(argv[1]) : 12;
    int ncmg = (argc > 2) ? atoi(argv[2]) : 1;

    ds4f_model m; memset(&m, 0, sizeof(m));
    m.n_threads = nthr; m.n_cmgs = ncmg;
    m.pool = ds4f_pool_start(nthr, ncmg);
    ds4f_init_fp8_e4m3_lut(m.fp8_lut);    /* gather path needs the LUT */

    printf("ds4f_gemm_test  nthr=%d n_cmgs=%d\n", nthr, ncmg);
    if (getenv("DS4F_OP_PROTO")) {   /* fp32 outer-product GEMM prototype vs the dot-product pv kernel */
        printf("--- fp32 outer-product GEMM prototype (OP_NT=%d, vs BF16_PV dot-product) ---\n", OP_NT);
        struct { int r, c; } sh[] = {{32768,1024},{8192,4096},{4096,8192},{2048,4096},{4096,2048},{1024,4096}};
        int M2[] = {16,32,64}, f = 0;
        for (int s=0;s<6;s++) for (int mi=0;mi<3;mi++) f += run_op_case(&m, sh[s].r, sh[s].c, M2[mi]);
        ds4f_pool_stop(m.pool); return f?1:0;
    }
    if (argc > 5) {   /* single bf16-pv shape for fapp profiling: nthr ncmg rows cols M (DS4F_GEMM_NITER loops) */
        int rows = atoi(argv[3]), cols = atoi(argv[4]), MM = atoi(argv[5]);
        int rc = run_case(&m, DS4F_BF16_PV, rows, cols, MM);
        ds4f_pool_stop(m.pool); return rc;
    }
    int fails = 0;
    /* shapes mirror real DS4F dense tensors (all K multiple of 512; rows %8==0) */
    struct { int rows, cols; } shapes[] = {
        {1024, 4096},     /* wq_a   */
        {32768, 1024},    /* wq_b   */
        {512, 4096},      /* wkv    */
        {8192, 4096},     /* o_inter (wo_b view-ish) */
        {4096, 8192},     /* wo_b   */
        {2048, 4096},     /* shared w1/w3 */
        {4096, 2048},     /* shared w2 */
        {256, 4096},      /* router (small rows) */
    };
    int ns = (int)(sizeof(shapes)/sizeof(shapes[0]));
    int Ms[] = {1, 2, 8, 16, 32, 64};
    int nM = (int)(sizeof(Ms)/sizeof(Ms[0]));

    ds4f_qtype tys[] = { DS4F_BF16_PV, DS4F_BF16, DS4F_Q8_PV };
    int nty = (int)(sizeof(tys)/sizeof(tys[0]));
    for (int ti = 0; ti < nty; ti++)
        for (int s = 0; s < ns; s++)
            for (int mi = 0; mi < nM; mi++)
                fails += run_case(&m, tys[ti], shapes[s].rows, shapes[s].cols, Ms[mi]);

    /* FP8 on-demand prefill GEMM (DS4F_FP8_BF16=0, -6 GB): tile-dequant correctness
     * (vs single-token f32 ref, argmax-exact + relL2) and throughput at the M values
     * the prefill batches over. */
    printf("\n--- FP8 on-demand prefill GEMM (tile-dequant, no resident bf16) ---\n");
    for (int s = 0; s < ns; s++)
        for (int mi = 0; mi < nM; mi++)
            fails += run_fp8_case(&m, shapes[s].rows, shapes[s].cols, Ms[mi]);

    /* MXFP4 expert GEMM: svtbl register dequant (no gather/LUT/resident bf16),
     * group-outer 2-token-blocked. Real expert shapes: w1/w3 [moe_inter,hidden],
     * w2 [hidden,moe_inter] with moe_inter=2048, hidden=4096. */
    printf("\n--- MXFP4 expert GEMM (svtbl register dequant) ---\n");
    struct { int rows, cols; } mxsh[] = { {2048, 4096}, {4096, 2048} };
    for (int s = 0; s < (int)(sizeof(mxsh)/sizeof(mxsh[0])); s++)
        for (int mi = 0; mi < nM; mi++)
            fails += run_mxfp4_case(&m, mxsh[s].rows, mxsh[s].cols, Ms[mi]);

    ds4f_pool_stop(m.pool);
    printf("%s  (%d failure%s)\n", fails ? "FAIL" : "ALL OK", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
