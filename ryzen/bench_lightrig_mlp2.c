#include "lightrig_mlp2.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(int argc,char **argv){size_t n=argc==2?strtoull(argv[1],0,10):10000,in=111,h=64,out=52;float*x=calloc(in,4),*w1=calloc(in*h,4),*b1=calloc(h,4),*w2=calloc(h*out,4),*b2=calloc(out,4),*s=calloc(h,4),*y=calloc(out,4);struct timespec a,b;if(!n||!x||!w1||!b1||!w2||!b2||!s||!y)return 2;for(size_t i=0;i<in*h;++i)w1[i]=(float)(i%17)*.001f;for(size_t i=0;i<h*out;++i)w2[i]=(float)(i%13)*.001f;for(size_t i=0;i<100;++i)lt_mlp2_f32(x,w1,b1,w2,b2,s,y,in,h,out);clock_gettime(CLOCK_MONOTONIC,&a);for(size_t i=0;i<n;++i)lt_mlp2_f32(x,w1,b1,w2,b2,s,y,in,h,out);clock_gettime(CLOCK_MONOTONIC,&b);double ms=((b.tv_sec-a.tv_sec)*1e9+b.tv_nsec-a.tv_nsec)/1e6/n;printf("backend=%s iterations=%zu latency_ms=%.6f\n",lt_mlp2_f32_backend(),n,ms);free(x);free(w1);free(b1);free(w2);free(b2);free(s);free(y);return 0;}
