/* GLM5.2 SwiGLU via a helper function: SiLU(x)=x*sigmoid(x)=x/(1+exp(-x)), then
 * out = SiLU(gate)*up. Stresses user function calls with float args + float
 * return inside a loop (calling convention / arg passing / return marshalling).
 * Compile with -O0 (see README). */
#include <stdio.h>
extern float expf(float);
#define M 64
static float gate[M], up[M], out[M];
volatile int SEED = 3;
static float silu(float x){ return x / (1.0f + expf(-x)); }
static float mul(float a, float b){ return a*b; }
int main(void){
  for (int i=0;i<M;i++){
    gate[i]=(float)((i*SEED)%15-7)*0.25f;
    up[i]  =(float)((i*5)%9-4)*0.5f;
  }
  for (int i=0;i<M;i++) out[i]=mul(silu(gate[i]), up[i]);
  /* verify recompute through the same helpers */
  int ok=1;
  for (int i=0;i<M && ok;i++) if (out[i]!=mul(silu(gate[i]),up[i])) ok=0;
  int chk=0; for (int i=0;i<M;i++) chk += (int)(out[i]*1000.0f);
  printf("silu_helper M=%d out0_x1e3=%d chk=%d ok=%d\n",
         M, (int)(out[0]*1000.0f), chk, ok);
  return 0;
}
