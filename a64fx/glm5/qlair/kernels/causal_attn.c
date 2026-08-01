/* GLM5.2 causal attention scores: for query position q, score[k] = (Q.K_k)/sqrt(d)
 * for k<=q (causal mask), then softmax over the unmasked prefix, then the context
 * o = sum_k p[k]*V_k. Stresses 2D indexing + a data-dependent (k<=q) loop bound.
 * T positions, d head dim, single head, f32. Compile with -O0 (see README). */
#include <stdio.h>
extern float expf(float);
extern float sqrtf(float);
#define T 12
#define Dh 8
static float Q[T*Dh], Kc[T*Dh], Vc[T*Dh];
static float ctx[T*Dh];
volatile int SEED = 6;
int main(void){
  for (int i=0;i<T*Dh;i++){
    Q[i]=(float)((i*SEED)%13-6)*0.2f;
    Kc[i]=(float)((i*5)%11-5)*0.2f;
    Vc[i]=(float)((i*3)%7-3)*0.3f;
  }
  float inv=1.0f/sqrtf((float)Dh);
  for (int q=0;q<T;q++){
    /* scores over causal prefix + streaming softmax */
    float m=-1e30f, l=0.0f;
    float sc[T];
    for (int k=0;k<=q;k++){
      float dot=0.0f;
      for (int d=0;d<Dh;d++) dot += Q[q*Dh+d]*Kc[k*Dh+d];
      sc[k]=dot*inv;
      float mn = sc[k]>m ? sc[k] : m;
      l = l*expf(m-mn) + expf(sc[k]-mn);
      m = mn;
    }
    for (int d=0;d<Dh;d++){
      float acc=0.0f;
      for (int k=0;k<=q;k++) acc += (expf(sc[k]-m)/l)*Vc[k*Dh+d];
      ctx[q*Dh+d]=acc;
    }
  }
  /* verify row0: q=0 attends only to k=0 => ctx[0]==Vc[0..] */
  int ok=1;
  for (int d=0;d<Dh;d++) if (ctx[d]!=Vc[d]) { ok=0; break; }
  int chk=0; for (int i=0;i<T*Dh;i++) chk += (int)(ctx[i]*1000.0f);
  printf("causal_attn T=%d Dh=%d ctx0_x1e3=%d chk=%d ok=%d\n",
         T, Dh, (int)(ctx[0]*1000.0f), chk, ok);
  return 0;
}
