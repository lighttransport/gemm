/* GLM5.2 weight streaming via pointer walking: a matvec that advances a running
 * weight pointer (*wp++) across a contiguous row-major buffer instead of index
 * arithmetic — mirrors how the runner streams staged weights. Stresses pointer
 * increment/dereference and pointer parameters. Compile with -O0 (see README). */
#include <stdio.h>
#define R 8
#define C 16
static float W[R*C];
static float x[C], y[R];
volatile int SEED = 4;
/* matvec through a walking pointer */
static void matvec(const float* w, const float* xv, float* yv){
  for (int r=0;r<R;r++){
    const float* wp = w + (long)r*C;   /* row base */
    float acc = 0.0f;
    for (int c=0;c<C;c++) acc += (*wp++) * xv[c];   /* walk the row */
    yv[r] = acc;
  }
}
int main(void){
  for (int i=0;i<R*C;i++) W[i]=(float)((i*SEED)%13-6)*0.1f;
  for (int c=0;c<C;c++) x[c]=(float)((c*3)%7-3)*0.5f;
  matvec(W, x, y);
  /* verify with index arithmetic */
  int ok=1;
  for (int r=0;r<R && ok;r++){
    float acc=0.0f;
    for (int c=0;c<C;c++) acc += W[r*C+c]*x[c];
    if (y[r]!=acc) ok=0;
  }
  int chk=0; for (int r=0;r<R;r++) chk += (int)(y[r]*1000.0f);
  printf("ptr_stream R=%d C=%d y0_x1e3=%d chk=%d ok=%d\n",
         R, C, (int)(y[0]*1000.0f), chk, ok);
  return 0;
}
