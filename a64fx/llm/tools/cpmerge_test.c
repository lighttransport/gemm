/* CP top-k merge exactness: does sharded local-top-k + merge == global top-k?
 * Claim: a globally-top-k slot has <k slots above it globally, hence <k on its own shard,
 * so it's within its shard's local top-k -> merging N local top-k lists recovers the global
 * top-k. Tie rule: higher score wins; ties -> lower global index. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { float s; int idx; } cand;
static int cmp_desc(const void *a, const void *b) {        /* desc score, ties -> lower idx */
    const cand *x = a, *y = b;
    if (x->s > y->s) return -1; if (x->s < y->s) return 1;
    return x->idx - y->idx;
}
static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
static void topk(const cand *in, int n, int k, int *out, int *nout) {
    cand *c = malloc((size_t)n*sizeof(cand)); memcpy(c, in, (size_t)n*sizeof(cand));
    qsort(c, n, sizeof(cand), cmp_desc);
    int m = k < n ? k : n;
    for (int i = 0; i < m; i++) out[i] = c[i].idx;
    qsort(out, m, sizeof(int), cmp_int);                   /* ascending -> set compare */
    *nout = m; free(c);
}
int main(int argc, char **argv) {
    unsigned seed = argc>1?atoi(argv[1]):1;
    int S = 50000, K = 512, N = 11, fails = 0, trials = 200;
    for (int t = 0; t < trials; t++) {
        cand *all = malloc((size_t)S*sizeof(cand));
        for (int i = 0; i < S; i++) { seed=seed*1103515245+12345;
            all[i].s = (float)((seed>>16) % 64); all[i].idx = i; }   /* coarse -> many ties */
        int ref[512], nref; topk(all, S, K, ref, &nref);            /* global reference */
        cand merged[11*512]; int mc = 0;
        for (int r = 0; r < N; r++) {                                /* 8-aligned shards (ds4f_tp_rowshard) */
            int blk=(S+7)/8, per=blk/N, ex=blk%N;
            int a0=(per*r+(r<ex?r:ex))*8, a1=(per*(r+1)+((r+1)<ex?(r+1):ex))*8; if(a1>S)a1=S;
            int loc[512], nloc; topk(all+a0, a1-a0, K, loc, &nloc);   /* loc[] already GLOBAL idx */
            for (int j=0;j<nloc;j++){ merged[mc].idx=loc[j]; merged[mc].s=all[loc[j]].s; mc++; }
        }
        int got[512], ngot; topk(merged, mc, K, got, &ngot);
        int bad = (ngot!=nref); for(int i=0;i<nref && !bad;i++) if(ref[i]!=got[i]) bad=1;
        if (bad) { fails++; if(fails<=3) printf("trial %d MISMATCH\n", t); }
        free(all);
    }
    printf("CP top-k merge: %d/%d trials EXACT (S=%d K=%d N=%d, coarse scores w/ heavy ties)\n",
           trials-fails, trials, S, K, N);
    return fails?1:0;
}
