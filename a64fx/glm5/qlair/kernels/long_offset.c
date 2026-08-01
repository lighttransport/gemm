/* GLM5.2 tensor offset math in 64-bit: a weight byte offset in a big model
 * (layer*layer_stride + row*cols + col) overflows int32, so it must be computed
 * in long/int64. Stresses 64-bit multiply/add/compare and long->int narrowing.
 * Compile with -O0 (see README). */
#include <stdio.h>
#include <stdint.h>
#define N 64
static int buf[N];
volatile int SEED = 3;
int main(void){
  for (int i=0;i<N;i++) buf[i]=i;
  /* realistic-ish sizes: 78 layers x (6144*2048) f32 weights ~ 9.8e8 elems */
  long cols   = 2048;
  long rows   = 6144;
  long lstride = rows*cols;              /* 12.58M — fits 32b, but products below don't */
  long chk = 0;
  int hits = 0;
  for (int layer=0; layer<78; layer++){
    long base = (long)layer * lstride;   /* up to ~9.8e8 > 2^31/2, needs 64b */
    long off  = base + (long)(layer*SEED % rows)*cols + (long)(layer % cols);
    chk += off;
    if (off > 2147483647L) hits++;       /* count offsets that exceeded int32 max */
    buf[layer % N] = (int)(off % (long)N);  /* long->int narrowing + modulo index */
  }
  /* also verify a 64-bit product that clearly overflows 32-bit */
  long big = (long)100000 * (long)100000; /* 1e10 */
  int big_ok = (big == 10000000000L);
  int buf_sum=0; for (int i=0;i<N;i++) buf_sum += buf[i];
  printf("long_offset chk=%ld hits=%d big=%ld big_ok=%d buf_sum=%d\n",
         chk, hits, big, big_ok, buf_sum);
  return 0;
}
