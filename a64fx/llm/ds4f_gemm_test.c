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

/* deterministic LCG so runs are reproducible (Date/rand-free) */
static uint32_t rng = 0x12345678u;
static inline float frand(void) {        /* ~U(-1,1) */
    rng = rng * 1664525u + 1013904223u;
    return ((float)(rng >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

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

    /* batched GEMM, token-major (Ystride=rows, Xstride=cols) */
    float *Ygemm = (float *)aligned_alloc(256, (size_t)M * rows * 4);
    ds4f_gemm(m, Ygemm, &t, X, M, rows, cols);

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
    printf("  %-8s rows=%-6d cols=%-5d M=%-3d  max-abs=%.3e  relL2=%.3e  argmax=%d/%d  nonfinite=%d  %s\n",
           tn, rows, cols, M, maxabs, maxrelL2, amatch, M, nbad, ok ? "OK" : "FAIL");

    free(Wrm); free(worig);
    if (is_q8) munmap(t.w, ds4f_wbytes(DS4F_Q8_PV, rows, cols));
    free(X); free(Yref); free(Ygemm);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    int nthr = (argc > 1) ? atoi(argv[1]) : 12;
    int ncmg = (argc > 2) ? atoi(argv[2]) : 1;

    ds4f_model m; memset(&m, 0, sizeof(m));
    m.n_threads = nthr; m.n_cmgs = ncmg;
    m.pool = ds4f_pool_start(nthr, ncmg);

    printf("ds4f_gemm_test  nthr=%d n_cmgs=%d\n", nthr, ncmg);
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

    ds4f_pool_stop(m.pool);
    printf("%s  (%d failure%s)\n", fails ? "FAIL" : "ALL OK", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
