/* GLM5.2 RMSNorm: y = x/sqrt(mean(x^2)+eps)*w  (hidden=6144, f32).
 * Compile: clair -t arm64 -f bin rmsnorm.c -o r.elf   (use -O0; clair -O float
 * codegen is broken — see README). Run: qlair -P -p r.elf */
#include <stdio.h>
extern float sqrtf(float);
#define H 6144
static float x[H], w[H], y[H];
volatile int SEED = 7;
int main(void){
  for (int i=0;i<H;i++){ x[i]=(float)((i*SEED)%17-8)*0.125f; w[i]=1.0f+(float)(i%5)*0.1f; }
  float ss=0.0f; for (int i=0;i<H;i++) ss+=x[i]*x[i];
  float inv=1.0f/sqrtf(ss/(float)H+1e-6f);
  for (int i=0;i<H;i++) y[i]=x[i]*inv*w[i];
  int ok=1; for (int i=0;i<H;i++) if (y[i]!=x[i]*inv*w[i]) { ok=0; break; }
  int csum=0; for (int i=0;i<H;i++) csum += (int)(y[i]*1000.0f);
  printf("rmsnorm H=%d inv_x1e6=%d csum=%d ok=%d\n", H,(int)(inv*1e6f),csum,ok);
  return 0;
}
