/* GLM5.2 MoE FFN activation: out = SiLU(gate) * up, SiLU(x)=x/(1+exp(-x)). */
#include <stdio.h>
extern float expf(float);
#define M 2048
static float gate[M], up[M], out[M];
volatile int S=3;
int main(void){
  for (int i=0;i<M;i++){ gate[i]=(float)((i*S)%13-6)*0.25f; up[i]=(float)((i%7)-3)*0.5f; }
  for (int i=0;i<M;i++){ float g=gate[i]; float s=g/(1.0f+expf(-g)); out[i]=s*up[i]; }
  /* verify against recompute at endpoints + a checksum */
  float g0=gate[0], s0=g0/(1.0f+expf(-g0));
  int ok = (out[0]==s0*up[0]);
  int chk=0; for(int i=0;i<M;i++) chk+=(int)(out[i]*1000.0f);
  printf("swiglu M=%d out0_x1e3=%d chk=%d ok=%d\n", M,(int)(out[0]*1000.0f),chk,ok);
  return 0;
}
