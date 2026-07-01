/* GLM5.2 int8 w8a16 matvec (the core MoE/attn GEMV): y[r] = scale[r] *
 * sum_c (int8)W[r*C+c] * x[c].  Per-channel scale (routed-expert layout).
 * Rtot rows x C cols, f32 activations. Compile with -O0 (see README). */
#include <stdio.h>
#define R 16
#define C 64
static signed char W[R*C];   /* symmetric int8 weights */
static float x[C], scale[R], y[R];
volatile int SEED = 3;
int main(void){
  for (int c=0;c<C;c++) x[c]=(float)((c*SEED)%11-5)*0.2f;
  for (int r=0;r<R;r++){
    scale[r]=0.02f*(float)(r+1);
    for (int c=0;c<C;c++) W[r*C+c]=(signed char)(((r*7+c*SEED)%255)-127);
  }
  for (int r=0;r<R;r++){
    float acc=0.0f;
    for (int c=0;c<C;c++) acc += (float)W[r*C+c]*x[c];
    y[r]=scale[r]*acc;
  }
  int ok=1;
  for (int r=0;r<R;r++){
    float acc=0.0f;
    for (int c=0;c<C;c++) acc += (float)W[r*C+c]*x[c];
    if (y[r]!=scale[r]*acc){ ok=0; break; }
  }
  int chk=0; for (int r=0;r<R;r++) chk += (int)(y[r]*100.0f);
  printf("int8_matvec R=%d C=%d y0_x1e2=%d chk=%d ok=%d\n",
         R, C, (int)(y[0]*100.0f), chk, ok);
  return 0;
}
