/* GLM5.2 int8 checkpoint unpack: compressed-tensors 'weight_packed' stores 4
 * signed int8 per int32 (little-endian). Unpack + sign-extend each byte, then
 * dequant against a per-row scale. Stresses shifts/masks + int8 sign-extend
 * from a packed int32. Compile with -O0 (see README). */
#include <stdio.h>
#include <stdint.h>
#define R 8
#define C 16            /* cols; packed as C/4 int32 per row */
static int32_t packed[R*(C/4)];
static float scale[R], x[C], y[R];
volatile int SEED = 5;
int main(void){
  /* build packed int32s from known signed int8 lanes */
  for (int r=0;r<R;r++){
    for (int g=0; g<C/4; g++){
      uint32_t w=0;
      for (int l=0;l<4;l++){
        int c = g*4+l;
        int8_t b = (int8_t)(((r*13 + c*SEED) % 251) - 125);   /* in [-125,125] */
        w |= ((uint32_t)(uint8_t)b) << (8*l);
      }
      packed[r*(C/4)+g] = (int32_t)w;
    }
    scale[r] = 0.01f*(float)(r+1);
  }
  for (int c=0;c<C;c++) x[c]=(float)((c*7)%9-4)*0.25f;
  /* unpack + matvec */
  for (int r=0;r<R;r++){
    float acc=0.0f;
    for (int g=0; g<C/4; g++){
      uint32_t w=(uint32_t)packed[r*(C/4)+g];
      for (int l=0;l<4;l++){
        int8_t b=(int8_t)((w >> (8*l)) & 0xFF);   /* extract + sign-extend */
        acc += (float)b * x[g*4+l];
      }
    }
    y[r]=scale[r]*acc;
  }
  /* verify: recompute against the ORIGINAL int8 lanes directly */
  int ok=1;
  for (int r=0;r<R && ok;r++){
    float acc=0.0f;
    for (int c=0;c<C;c++){
      int8_t b=(int8_t)(((r*13 + c*SEED) % 251) - 125);
      acc += (float)b * x[c];
    }
    if (y[r]!=scale[r]*acc) ok=0;
  }
  int chk=0; for (int r=0;r<R;r++) chk += (int)(y[r]*1000.0f);
  printf("int8_unpack R=%d C=%d y0_x1e3=%d chk=%d ok=%d\n",
         R, C, (int)(y[0]*1000.0f), chk, ok);
  return 0;
}
