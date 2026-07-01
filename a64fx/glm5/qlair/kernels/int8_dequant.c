/* GLM5.2 int8 (compressed-tensors w8a16) dequant: w = (int8)(q-128)*scale[group].
 * offset-binary bytes, group size 128 (attn/dense/shared experts). */
#include <stdio.h>
#define C 512
#define GS 128
static unsigned char q[C];       /* offset-binary bytes */
static float scale[C/GS];        /* per-group scale */
static float w[C];
volatile int S=5;
int main(void){
  for (int i=0;i<C;i++) q[i]=(unsigned char)((i*S)&0xff);
  for (int g=0;g<C/GS;g++) scale[g]=0.01f*(float)(g+1);
  for (int c=0;c<C;c++){ int v=(int)q[c]-128; w[c]=(float)v*scale[c/GS]; }
  int ok = (w[0]==(float)((int)q[0]-128)*scale[0]) && (w[C-1]==(float)((int)q[C-1]-128)*scale[(C-1)/GS]);
  int chk=0; for(int i=0;i<C;i++) chk+=(int)(w[i]*10000.0f);
  printf("deq C=%d GS=%d w0_x1e4=%d chk=%d ok=%d\n", C,GS,(int)(w[0]*10000.0f),chk,ok);
  return 0;
}
