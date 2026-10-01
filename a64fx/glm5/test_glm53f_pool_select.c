#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "glm53f_pool_select.h"
typedef glm53f_pool_score pool_score;
static int cmp(const void *a,const void *b){const pool_score*x=a,*y=b;if(x->score>y->score)return -1;if(x->score<y->score)return 1;return x->id<y->id?-1:x->id>y->id;}
static void down(pool_score*h,int n,int p){for(;;){int w=p,l=2*p+1,r=l+1;if(l<n&&cmp(h+l,h+w)>0)w=l;if(r<n&&cmp(h+r,h+w)>0)w=r;if(w==p)return;pool_score t=h[p];h[p]=h[w];h[w]=t;p=w;}}
static void heap(pool_score*h,const float*s,int n,int k){for(int p=0;p<k;p++)h[p]=(pool_score){s[p],p};for(int p=k/2;p-->0;)down(h,k,p);for(int p=k;p<n;p++){pool_score x={s[p],p};if(cmp(&x,h)<0){h[0]=x;down(h,k,0);}}qsort(h,k,sizeof(*h),cmp);}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){
    enum {MAX=8193,K=512};float *s=malloc(MAX*sizeof(*s));pool_score *a=malloc(MAX*sizeof(*a)),*b=malloc(MAX*sizeof(*b)),*r=malloc(MAX*sizeof(*r));
    int *ids=malloc((MAX+4)*sizeof(*ids));
    if(!s||!a||!b||!r||!ids)return 2;
    int sizes[]={0,1,7,128,512,513,1024,2012,2053,4096,4097,8193},cases=0,bad=0;uint32_t seed=47;
    for(unsigned nn=0;nn<sizeof(sizes)/sizeof(*sizes);nn++)for(int shape=0;shape<10;shape++){
        int n=sizes[nn],k=n<K?n:K;
        for(int p=0;p<n;p++){
            seed=seed*1664525u+1013904223u;
            float v=(int)(seed>>8)*.0001f;
            if(shape==1)v=0;
            if(shape==2)v=p;
            if(shape==3)v=n-p;
            if(shape==4)v=p%13;
            if(shape==5)v=p<n/2?p:n-p;
            if(shape==6)v=(p%3)?0:-0.f;
            if(shape==7)v=(p%5)?INFINITY:-INFINITY;
            if(shape>=8 && (shape==9 || p%7==0)){uint32_t bits=0x7fc00000u+(uint32_t)p;memcpy(&v,&bits,4);}
            s[p]=v;r[p]=(pool_score){v,p};
        }
        qsort(r,n,sizeof(*r),cmp);heap(a,s,n,k);
        for(int p=k;p<k+4;p++)b[p]=(pool_score){123.25f,98765};
        for(int p=n;p<n+4;p++)ids[p]=98765;
        glm53f_pool_top_partition((glm53f_pool_score*)b,s,n,k,ids);
        bad|=memcmp(a,b,k*sizeof(*a))!=0;
        if(shape<8)bad|=memcmp(r,b,k*sizeof(*b))!=0;
        for(int p=k;p<k+4;p++)bad|=b[p].score!=123.25f||b[p].id!=98765;
        for(int p=n;p<n+4;p++)bad|=ids[p]!=98765;
        cases++;
    }
    for(int p=0;p<2012;p++){seed=seed*1664525u+1013904223u;s[p]=(int)(seed>>8)*.0001f;}
    double ht=1e9,st=1e9;
    for(int rep=-1;rep<5;rep++){
        double t=now();for(int i=0;i<1000;i++)heap(a,s,2012,K);double h=now()-t;
        t=now();for(int i=0;i<1000;i++)glm53f_pool_top_partition((glm53f_pool_score*)b,s,2012,K,ids);double v=now()-t;
        if(rep>=0){if(h<ht)ht=h;if(v<st)st=v;}
    }
    printf("POOL_SELECT cases=%d ties_infinities_nan_fallback_bounds=BIT_EXACT heap_us=%.3f select_us=%.3f ratio=%.3f %s\n",cases,ht*1000,st*1000,ht/st,bad?"FAIL":"PASS");
    free(ids);free(r);free(b);free(a);free(s);return bad;
}
