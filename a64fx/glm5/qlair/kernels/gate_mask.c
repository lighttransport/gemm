/* GLM5.2 router gate masking with short-circuit conditions: keep an expert only
 * if it is unmasked AND its gate exceeds a threshold, OR it is a forced/shared
 * expert. Stresses && / || short-circuit evaluation (incl. side-effect ordering
 * via a call-counting helper). Compile with -O0 (see README). */
#include <stdio.h>
#define E 16
static float gate[E];
static int   masked[E];
static int   forced[E];
static int   keep[E];
volatile int SEED = 7;
static int probe_calls = 0;
static int probe(int e){ probe_calls++; return forced[e]; }
int main(void){
  for (int e=0;e<E;e++){
    gate[e]   = (float)((e*SEED)%21-10)*0.1f;
    masked[e] = ((e%5)==0);
    forced[e] = ((e%8)==3);
  }
  float thr = 0.3f;
  int kept=0;
  for (int e=0;e<E;e++){
    /* short-circuit: probe() (forced) only evaluated when the first clause fails */
    if ((!masked[e] && gate[e] > thr) || probe(e)) { keep[e]=1; kept++; }
    else keep[e]=0;
  }
  /* verify recompute (without the probe side effect) */
  int ok=1;
  for (int e=0;e<E && ok;e++){
    int want = ((!masked[e] && gate[e] > thr) || forced[e]) ? 1 : 0;
    if (keep[e]!=want) ok=0;
  }
  /* probe must be called exactly for slots where the first clause was false */
  int expect_probes=0;
  for (int e=0;e<E;e++) if (!(!masked[e] && gate[e] > thr)) expect_probes++;
  int probes_ok = (probe_calls==expect_probes);
  printf("gate_mask E=%d kept=%d probe_calls=%d probes_ok=%d ok=%d\n",
         E, kept, probe_calls, probes_ok, ok);
  return 0;
}
