/* Equivalence test for ds4f_index_topk: fast (default) vs naive (DS4F_TOPK_NAIVE=1).
 * Runs the REAL static function from ds4f.h on many adversarial score arrays (large T,
 * heavy ties, thr<T, all-equal, k>=thr) and dumps every sel[] so two runs can be diffed:
 *
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -D_GNU_SOURCE \
 *       -I../../common -o build/topk_equiv_test tools/topk_equiv_test.c -lm
 *   DS4F_TOPK_NAIVE=1 ./build/topk_equiv_test > /tmp/topk_naive.txt
 *   ./build/topk_equiv_test > /tmp/topk_fast.txt
 *   diff /tmp/topk_naive.txt /tmp/topk_fast.txt && echo EQUIV_PASS
 *
 * Determinism: fixed splitmix64 seed per trial => identical scores across both runs, so a
 * clean diff proves the fast heap+merge selects the identical set & order as the naive scan.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ds4f.h"

static uint64_t sm_state;
static uint64_t sm_next(void) {        /* splitmix64 */
    uint64_t z = (sm_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static float frand(void) { return (float)((sm_next() >> 11) * (1.0 / 9007199254740992.0)); }

/* one trial: fill T scores with `mode` (0=continuous,1=small-int ties,2=all-equal,
 * 3=many-zero-relu-like), select top-k within thr, dump sel[0..k). */
static void trial(int id, int T, int thr, int k, int offset, int mode) {
    float *score = (float *)malloc((size_t)T * sizeof(float));
    int   *sel   = (int   *)malloc((size_t)(k > 0 ? k : 1) * sizeof(int));
    for (int t = 0; t < T; t++) {
        switch (mode) {
            case 0: score[t] = frand() * 100.f - 50.f; break;            /* continuous */
            case 1: score[t] = (float)(int)(frand() * 5.f); break;        /* {0..4}, heavy ties */
            case 2: score[t] = 3.14159f; break;                          /* all equal */
            default: { float r = frand(); score[t] = r < 0.7f ? 0.f : r * 10.f; } break; /* relu-sparse */
        }
    }
    ds4f_index_topk(score, T, thr, k, offset, sel);
    printf("trial %d T=%d thr=%d k=%d off=%d mode=%d:", id, T, thr, k, offset, mode);
    for (int n = 0; n < k; n++) printf(" %d", sel[n]);
    printf("\n");
    free(score); free(sel);
}

int main(void) {
    int Ts[]   = { 1, 2, 7, 50, 511, 512, 513, 2560, 4096 };
    int ks[]   = { 1, 8, 64, 512 };
    int id = 0;
    for (unsigned ti = 0; ti < sizeof(Ts)/sizeof(Ts[0]); ti++)
      for (unsigned ki = 0; ki < sizeof(ks)/sizeof(ks[0]); ki++)
        for (int mode = 0; mode < 4; mode++)
          for (int tcase = 0; tcase < 3; tcase++) {
              int T = Ts[ti], k = ks[ki];
              int thr = (tcase == 0) ? T : (tcase == 1) ? T/2 : (k < T ? k : T); /* T, T/2, ~k */
              if (thr < 1) thr = 1;
              sm_state = 0xD5F0000ull + id;        /* fixed per-trial seed => reproducible */
              trial(id++, T, thr, k, (ti*7)%1000, mode);
          }
    printf("TOTAL trials=%d\n", id);
    return 0;
}
