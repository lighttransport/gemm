/* GLM5.2 RoPE (rotary position embedding) on the MLA qk_rope portion.
 * For each dim pair i in [0,D/2): angle = pos * base^(-2i/D), then
 *   x'[2i]   = x[2i]*cos(a) - x[2i+1]*sin(a)
 *   x'[2i+1] = x[2i]*sin(a) + x[2i+1]*cos(a)
 * D=64 (qk_rope_head_dim), f32. Compile with -O0 (see README). */
#include <stdio.h>
extern float sinf(float);
extern float cosf(float);
extern float powf(float, float);
#define D 64
static float x[D], y[D];
volatile int POS = 5;
int main(void){
  for (int i=0;i<D;i++) x[i]=(float)((i*7)%13-6)*0.25f;
  float pos=(float)POS;
  for (int i=0;i<D/2;i++){
    float freq = powf(10000.0f, -2.0f*(float)i/(float)D);
    float a = pos*freq;
    float c = cosf(a), s = sinf(a);
    float e = x[2*i], o = x[2*i+1];
    y[2*i]   = e*c - o*s;
    y[2*i+1] = e*s + o*c;
  }
  /* verify: recompute and check exact-equality; rotation preserves pair norm */
  int ok=1;
  for (int i=0;i<D/2;i++){
    float freq = powf(10000.0f, -2.0f*(float)i/(float)D);
    float a = pos*freq;
    float c = cosf(a), s = sinf(a);
    float e = x[2*i], o = x[2*i+1];
    if (y[2*i]!=e*c-o*s || y[2*i+1]!=e*s+o*c) { ok=0; break; }
  }
  /* pair-norm preservation: sum |y|^2 == sum |x|^2 (to int scale) */
  int nx=0, ny=0;
  for (int i=0;i<D;i++){ nx+=(int)(x[i]*x[i]*1000.0f); ny+=(int)(y[i]*y[i]*1000.0f); }
  printf("rope D=%d pos=%d y0_x1e3=%d normx=%d normy=%d ok=%d\n",
         D, POS, (int)(y[0]*1000.0f), nx, ny, ok);
  return 0;
}
