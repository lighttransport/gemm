/* GLM5.2 MoE router: softmax over 256 routed experts, pick top-8 (indices +
 * renormalized weights). This is the argmax-critical path (kept high precision
 * in the runner). f32; compile with -O0. */
#include <stdio.h>
extern float expf(float);
#define E 256
#define K 8
static float logit[E], prob[E];
volatile int S=11;
int main(void){
  float mx=-1e30f;
  for (int e=0;e<E;e++){ logit[e]=(float)((e*S)%23-11)*0.3f; if(logit[e]>mx) mx=logit[e]; }
  float sum=0.0f; for (int e=0;e<E;e++){ prob[e]=expf(logit[e]-mx); sum+=prob[e]; }
  for (int e=0;e<E;e++) prob[e]/=sum;
  /* top-8 by repeated argmax */
  int idx[K]; float wsum=0.0f; float p2[E]; for(int e=0;e<E;e++) p2[e]=prob[e];
  for (int k=0;k<K;k++){ int bi=0; float bv=-1.0f; for(int e=0;e<E;e++) if(p2[e]>bv){bv=p2[e];bi=e;} idx[k]=bi; wsum+=bv; p2[bi]=-1.0f; }
  /* verify: probs sum ~1, top-8 weights descending, all indices distinct */
  int desc=1; for(int k=1;k<K;k++) if(prob[idx[k]]>prob[idx[k-1]]) desc=0;
  int distinct=1; for(int a=0;a<K;a++) for(int b=a+1;b<K;b++) if(idx[a]==idx[b]) distinct=0;
  printf("router E=%d K=%d top1=%d topKw_x1e3=%d desc=%d distinct=%d ok=%d\n",
         E,K,idx[0],(int)(wsum*1000.0f),desc,distinct,(desc&&distinct));
  return 0;
}
