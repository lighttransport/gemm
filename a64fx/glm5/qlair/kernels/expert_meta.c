/* GLM5.2 MoE expert metadata + dispatch: per-expert id/scale/count kept as
 * parallel arrays (clair -O0 has no working struct support — see README), a
 * routing index selects an expert, bumps its count, and scales its id.
 * Stresses indirect gather/scatter (experts[route[e]]) + mixed int/float. -O0. */
#include <stdio.h>
#define E 8
static int   ex_id[E];
static float ex_scale[E];
static int   ex_count[E];
static int   route[E];
static float acc[E];
volatile int SEED = 5;
int main(void){
  for (int e=0;e<E;e++){
    ex_id[e]    = e*3 + 1;
    ex_scale[e] = 0.1f*(float)(e+1);
    ex_count[e] = 0;
    route[e]    = (e*SEED) % E;
  }
  /* dispatch: each slot routes to expert route[e], bump its count, scale its id */
  for (int e=0;e<E;e++){
    int r = route[e];
    ex_count[r] = ex_count[r] + 1;
    acc[e] = ex_scale[r] * (float)ex_id[r];
  }
  /* verify by recompute */
  int ok=1;
  for (int e=0;e<E && ok;e++){
    int r = route[e];
    if (acc[e] != ex_scale[r] * (float)ex_id[r]) ok=0;
  }
  int csum=0, idsum=0;
  for (int e=0;e<E;e++){ csum += ex_count[e]; idsum += ex_id[e]; }
  printf("expert_meta E=%d acc0_x1e2=%d countsum=%d idsum=%d ok=%d\n",
         E, (int)(acc[0]*100.0f), csum, idsum, ok);
  return 0;
}
