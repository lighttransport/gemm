/* GLM5.2 attention online (streaming) softmax: one pass tracking running max m
 * and running sum l with rescale, as in flash-attention. N scores, f32.
 *   m_new = max(m, s[i]); l = l*exp(m-m_new) + exp(s[i]-m_new); m = m_new
 * then p[i] = exp(s[i]-m)/l. Compile with -O0 (see README). */
#include <stdio.h>
extern float expf(float);
#define N 128
static float s[N], p[N];
volatile int SEED = 9;
int main(void){
  for (int i=0;i<N;i++) s[i]=(float)((i*SEED)%19-9)*0.3f;
  /* streaming max + normaliser */
  float m=-1e30f, l=0.0f;
  for (int i=0;i<N;i++){
    float mn = s[i]>m ? s[i] : m;
    l = l*expf(m-mn) + expf(s[i]-mn);
    m = mn;
  }
  for (int i=0;i<N;i++) p[i]=expf(s[i]-m)/l;
  /* verify vs a naive two-pass softmax with the same max */
  float mx=s[0]; for (int i=1;i<N;i++) if (s[i]>mx) mx=s[i];
  float sum=0.0f; for (int i=0;i<N;i++) sum+=expf(s[i]-mx);
  int ok=1;
  for (int i=0;i<N;i++){ float ref=expf(s[i]-mx)/sum; float d=p[i]-ref; if (d<-1e-4f||d>1e-4f){ ok=0; break; } }
  int psum=0; for (int i=0;i<N;i++) psum+=(int)(p[i]*10000.0f);
  printf("online_softmax N=%d p0_x1e4=%d psum_x1e4=%d ok=%d\n",
         N, (int)(p[0]*10000.0f), psum, ok);
  return 0;
}
