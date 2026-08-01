/* GLM5.2 router top-k with an expert mask + early exit: pick the K highest
 * gate values among UNMASKED experts (masking = capacity/routing drop). Stresses
 * nested loops with continue (skip masked) and break (early stop). -O0. */
#include <stdio.h>
#define E 32
#define K 4
static float gate[E];
static int masked[E];
static int idx[K];
static float val[K];
volatile int SEED = 5;
int main(void){
  for (int e=0;e<E;e++){ gate[e]=(float)((e*SEED)%23-11)*0.1f; masked[e]=((e%7)==0); }
  for (int k=0;k<K;k++){ idx[k]=-1; val[k]=-1e30f; }
  /* repeated argmax over unmasked, not-yet-picked experts */
  for (int k=0;k<K;k++){
    int best=-1; float bv=-1e30f;
    for (int e=0;e<E;e++){
      if (masked[e]) continue;               /* skip masked */
      int taken=0;
      for (int j=0;j<k;j++) if (idx[j]==e){ taken=1; break; }  /* skip picked */
      if (taken) continue;
      if (gate[e]>bv){ bv=gate[e]; best=e; }
    }
    if (best<0) break;                        /* no candidate left */
    idx[k]=best; val[k]=bv;
  }
  /* verify: strictly descending values, distinct indices, none masked */
  int ok=1;
  for (int k=0;k<K;k++){
    if (idx[k]<0){ ok=0; break; }
    if (masked[idx[k]]){ ok=0; break; }
    if (k>0 && val[k]>val[k-1]){ ok=0; break; }
    for (int j=0;j<k;j++) if (idx[j]==idx[k]){ ok=0; break; }
  }
  printf("masked_topk E=%d K=%d i0=%d i1=%d i2=%d i3=%d v0_x1e2=%d ok=%d\n",
         E, K, idx[0], idx[1], idx[2], idx[3], (int)(val[0]*100.0f), ok);
  return 0;
}
