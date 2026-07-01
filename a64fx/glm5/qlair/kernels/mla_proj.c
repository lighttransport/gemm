/* GLM5.2 MLA low-rank projection: x(hidden) --Wd--> c(kv_lora) --Wu--> o(out).
 * The two-stage down/up matvec chain that MLA uses to compress KV (kv_lora=512
 * in the real model). Down-scaled dims here. Stresses a matvec feeding another
 * matvec (intermediate buffer reuse). f32, -O0. */
#include <stdio.h>
#define H 32       /* hidden   */
#define L 8        /* kv_lora  */
#define O 24       /* out      */
static float x[H], c[L], o[O];
static float Wd[L*H], Wu[O*L];
volatile int SEED = 4;
int main(void){
  for (int i=0;i<H;i++) x[i]=(float)((i*SEED)%11-5)*0.2f;
  for (int i=0;i<L*H;i++) Wd[i]=(float)((i*3)%7-3)*0.1f;
  for (int i=0;i<O*L;i++) Wu[i]=(float)((i*5)%9-4)*0.1f;
  /* down: c[l] = sum_h Wd[l*H+h]*x[h] */
  for (int l=0;l<L;l++){
    float acc=0.0f;
    for (int h=0;h<H;h++) acc += Wd[l*H+h]*x[h];
    c[l]=acc;
  }
  /* up: o[r] = sum_l Wu[r*L+l]*c[l] */
  for (int r=0;r<O;r++){
    float acc=0.0f;
    for (int l=0;l<L;l++) acc += Wu[r*L+l]*c[l];
    o[r]=acc;
  }
  /* verify: recompute the chain */
  int ok=1;
  for (int r=0;r<O && ok;r++){
    float acc=0.0f;
    for (int l=0;l<L;l++){
      float cl=0.0f; for (int h=0;h<H;h++) cl += Wd[l*H+h]*x[h];
      acc += Wu[r*L+l]*cl;
    }
    if (o[r]!=acc) ok=0;
  }
  int chk=0; for (int r=0;r<O;r++) chk += (int)(o[r]*1000.0f);
  printf("mla_proj H=%d L=%d O=%d o0_x1e3=%d chk=%d ok=%d\n",
         H, L, O, (int)(o[0]*1000.0f), chk, ok);
  return 0;
}
