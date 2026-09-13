#include "lightrig_mlp2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float sample(unsigned *state){*state=*state*1664525u+1013904223u;return ((float)(*state>>8)/8388608.0f)-1.0f;}
int main(void){const size_t in=111,hidden=64,out=32;float *x=malloc(in*4),*w1=malloc(in*hidden*4),*b1=malloc(hidden*4),*w2=malloc(hidden*out*4),*b2=malloc(out*4),*hs=malloc(hidden*4),*hf=malloc(hidden*4),*ys=malloc(out*4),*yf=malloc(out*4);unsigned state=7;float maximum=0;if(!x||!w1||!b1||!w2||!b2||!hs||!hf||!ys||!yf)return 2;for(size_t n=0;n<in;++n)x[n]=sample(&state);for(size_t n=0;n<in*hidden;++n)w1[n]=sample(&state);for(size_t n=0;n<hidden;++n)b1[n]=sample(&state);for(size_t n=0;n<hidden*out;++n)w2[n]=sample(&state);for(size_t n=0;n<out;++n)b2[n]=sample(&state);lt_mlp2_f32_scalar(x,w1,b1,w2,b2,hs,ys,in,hidden,out);lt_mlp2_f32(x,w1,b1,w2,b2,hf,yf,in,hidden,out);for(size_t n=0;n<out;++n){float d=fabsf(ys[n]-yf[n]);if(d>maximum)maximum=d;}printf("backend=%s max_abs_error=%g\n",lt_mlp2_f32_backend(),maximum);return maximum<=1e-4f?0:1;}
