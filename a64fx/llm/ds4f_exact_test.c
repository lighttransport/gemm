/* ds4f_exact_test.c — validate the exact DS4F forward math (DS4F_EXACT path) in
 * common/ds4f.h against the pure-Python reference (tools/ds4f_exact_ref.py):
 *   ds4f_rope_table / ds4f_rope_apply (RoPE+YaRN, fwd+inverse) and
 *   ds4f_topk_exact (sqrtsoftplus gate, bias-select, unbiased weights).
 * Uses the SAME deterministic test vectors and writes exact_c.txt with the SAME
 * line order as exact_py.txt so a paste/awk diff reads column 2 vs column 4.
 *
 * Build:  fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *             -D_GNU_SOURCE -I../../common -o build/ds4f_exact_test \
 *             ds4f_exact_test.c -lm -lpthread
 */
#include "ds4f.h"

#define DIM    64
#define HALF   (DIM/2)      /* 32 */
#define MAXP   32
#define FACTOR 16
#define BETA_F 32
#define BETA_S 1
#define THETA_D 10000.0
#define THETA_C 160000.0
#define ORIG   65536
#define N_EXP  8
#define TOPK   3
#define ROUTE  1.5f

int main(void) {
    static float cd[MAXP*HALF], sd[MAXP*HALF], cc[MAXP*HALF], sc[MAXP*HALF];
    ds4f_rope_table(cd, sd, DIM, MAXP, THETA_D, FACTOR, BETA_F, BETA_S, 0);     /* dense */
    ds4f_rope_table(cc, sc, DIM, MAXP, THETA_C, FACTOR, BETA_F, BETA_S, ORIG);  /* comp  */

    /* same hashes as ds4f_exact_ref.py */
    float v[DIM];
    for (int i = 0; i < DIM; i++) v[i] = (float)((((i*37 + 11) % 97) - 48)) / 48.0f;
    float logits[N_EXP], bias[N_EXP];
    for (int e = 0; e < N_EXP; e++) {
        logits[e] = (float)((((e*53 + 7) % 29) - 14)) / 7.0f;
        bias[e]   = (float)((((e*19 + 3) % 11) - 5)) / 10.0f;
    }

    int POS[3] = {1, 7, 31}, KS[4] = {0, 1, 15, 31};
    FILE *fp = fopen("exact_c.txt", "w");

    for (int pi = 0; pi < 3; pi++) for (int ki = 0; ki < 4; ki++) {
        int p = POS[pi], k = KS[ki];
        fprintf(fp, "dcos%d_%-7d %.7e\n", p, k, cd[p*HALF + k]);
        fprintf(fp, "dsin%d_%-7d %.7e\n", p, k, sd[p*HALF + k]);
    }
    for (int pi = 0; pi < 3; pi++) for (int ki = 0; ki < 4; ki++) {
        int p = POS[pi], k = KS[ki];
        fprintf(fp, "ccos%d_%-7d %.7e\n", p, k, cc[p*HALF + k]);
        fprintf(fp, "csin%d_%-7d %.7e\n", p, k, sc[p*HALF + k]);
    }

    float fwd[DIM], inv[DIM];
    memcpy(fwd, v, sizeof v); ds4f_rope_apply(fwd, cc, sc, 7, HALF, 0);
    memcpy(inv, v, sizeof v); ds4f_rope_apply(inv, cc, sc, 7, HALF, 1);
    for (int i = 0; i < DIM; i++) fprintf(fp, "fwd%-9d %.7e\n", i, fwd[i]);
    for (int i = 0; i < DIM; i++) fprintf(fp, "inv%-9d %.7e\n", i, inv[i]);

    /* gate with bias, then without (== plain topk). sort (idx,wt) by eid so the
     * output is independent of selection order. */
    int   idx[TOPK]; float wt[TOPK];
    ds4f_topk_exact(logits, bias, N_EXP, TOPK, idx, wt, ROUTE);
    for (int a = 0; a < TOPK; a++) for (int b = a+1; b < TOPK; b++)
        if (idx[b] < idx[a]) { int t=idx[a];idx[a]=idx[b];idx[b]=t; float u=wt[a];wt[a]=wt[b];wt[b]=u; }
    for (int i = 0; i < TOPK; i++) {
        fprintf(fp, "gb%d_eid%-5d %.7e\n", i, 0, (double)idx[i]);
        fprintf(fp, "gb%d_wt%-6d %.7e\n", i, 0, wt[i]);
    }
    ds4f_topk_exact(logits, NULL, N_EXP, TOPK, idx, wt, ROUTE);
    for (int a = 0; a < TOPK; a++) for (int b = a+1; b < TOPK; b++)
        if (idx[b] < idx[a]) { int t=idx[a];idx[a]=idx[b];idx[b]=t; float u=wt[a];wt[a]=wt[b];wt[b]=u; }
    for (int i = 0; i < TOPK; i++) {
        fprintf(fp, "g0%d_eid%-5d %.7e\n", i, 0, (double)idx[i]);
        fprintf(fp, "g0%d_wt%-6d %.7e\n", i, 0, wt[i]);
    }
    fclose(fp);
    printf("wrote exact_c.txt\n");
    return 0;
}
