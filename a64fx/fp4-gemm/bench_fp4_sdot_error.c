#define _POSIX_C_SOURCE 200809L
#include "fp4_gemm.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {off_t off;size_t bytes;int n,k;char dtype[16],name[256];} rec;
static unsigned state=19;static unsigned rnd(void){state=state*1664525u+1013904223u;return state;}
static float uni(void){return((rnd()>>8)*(1.0f/8388608.0f))-1.0f;}
static int find_rec(const char*manifest,const char*name,rec*r){FILE*f=fopen(manifest,"r");char line[640];if(!f)return-1;
 while(fgets(line,sizeof(line),f)){long long o,b;int nd,n,k;char dt[16],nm[256];
  if(sscanf(line,"%lld %lld %15s %d %d %d %255s",&o,&b,dt,&nd,&n,&k,nm)==7&&nd==2&&!strcmp(nm,name)){
   r->off=(off_t)o;r->bytes=(size_t)b;r->n=n;r->k=k;snprintf(r->dtype,sizeof(r->dtype),"%s",dt);snprintf(r->name,sizeof(r->name),"%s",nm);fclose(f);return 0;}}
 fclose(f);return-1;}
static int read_full(int fd,void*p,size_t n,off_t o){size_t z=0;while(z<n){ssize_t q=pread(fd,(char*)p+z,n-z,o+(off_t)z);if(q<=0)return-1;z+=(size_t)q;}return 0;}
static void fill(float*a,int k,int dist,int seed){state=(unsigned)(19+seed*977+dist*65537);for(int i=0;i<k;++i){
 if(dist==0)a[i]=uni();else if(dist==1)a[i]=(rnd()&1)?1.0f:-1.0f;else{float u=fmaxf((uni()+1)*.5f,1e-7f),v=(uni()+1)*.5f;
  float x=sqrtf(-2*logf(u))*cosf(6.28318530718f*v);if(dist==3&&(rnd()&255)==0)x*=16;a[i]=x;}}}
static void err(const float*r,const float*g,int n,double*rel,double*cosine){double ne=0,rr=0,gg=0,rg=0;for(int i=0;i<n;++i){double d=g[i]-r[i];ne+=d*d;rr+=(double)r[i]*r[i];gg+=(double)g[i]*g[i];rg+=(double)r[i]*g[i];}*rel=sqrt(ne/(rr+1e-300));*cosine=rg/sqrt(rr*gg+1e-300);}
int main(int ac,char**av){const char*raw=ac>1?av[1]:"/local/u14346/ds4f-fp4-gemm/layer0.raw";
 const char*man=ac>2?av[2]:"/local/u14346/ds4f-fp4-gemm/layer0.manifest";int ag=ac>3?atoi(av[3]):4;
 int fd=open(raw,O_RDONLY);if(fd<0){perror(raw);return 1;}double worst=0,mincos=1;
 for(int wi=1;wi<=3;++wi){char wn[256],sn[256];snprintf(wn,sizeof(wn),"layers.0.ffn.experts.0.w%d.weight",wi);snprintf(sn,sizeof(sn),"layers.0.ffn.experts.0.w%d.scale",wi);rec wr,sr;
  if(find_rec(man,wn,&wr)||find_rec(man,sn,&sr)||strcmp(wr.dtype,"I8")||strcmp(sr.dtype,"F8_E8M0"))return 1;
  int n=wr.n,k=wr.k*2;fp4_matrix w;if(fp4_matrix_alloc(&w,FP4_MX,n,k)||read_full(fd,w.codes,wr.bytes,wr.off)||read_full(fd,w.scales,sr.bytes,sr.off)||fp4_matrix_prepare_n32(&w)||fp4_matrix_prepare_u8(&w)||fp4_matrix_prepare_sdot(&w)||fp4_matrix_prepare_pair(&w))return 1;
  float*a=aligned_alloc(256,(size_t)k*4),*ref=aligned_alloc(256,(size_t)n*4),*got=aligned_alloc(256,(size_t)n*4),*pair=aligned_alloc(256,(size_t)n*4);fp4_i8_activation qa={0};fp4_pair_activation pa={0};if(!a||!ref||!got||!pair)return 1;
  double tw=0,tc=1;for(int d=0;d<4;++d)for(int seed=0;seed<3;++seed){fill(a,k,d,seed);
   for(int r=0;r<n;++r){double z=0;for(int x=0;x<k;++x)z+=(double)a[x]*fp4_dequant_value(&w,r,x);ref[r]=(float)z;}
   if(fp4_i8_activation_prepare(&qa,a,k,ag)||fp4_gemv_i8_sdot_omp(got,&qa,&w,12)||
      fp4_pair_activation_prepare(&pa,a,k,ag)||fp4_gemv_pair_lut_omp(pair,&pa,&w,12))return 1;
   double re,co,pdiff,pcos;err(ref,got,n,&re,&co);err(got,pair,n,&pdiff,&pcos);
   if(pdiff>2e-6){fprintf(stderr,"pair/sdot mismatch %.6g\n",pdiff);return 1;}
   if(re>tw)tw=re;if(co<tc)tc=co;}
  printf("tensor=w%d N=%d K=%d A_G=%d worst_rel_l2=%.6g min_cos=%.9f\n",wi,n,k,ag,tw,tc);if(tw>worst)worst=tw;if(tc<mincos)mincos=tc;
  fp4_i8_activation_free(&qa);fp4_pair_activation_free(&pa);fp4_matrix_free(&w);free(a);free(ref);free(got);free(pair);}
 close(fd);printf("summary A_G=%d worst_rel_l2=%.6g min_cos=%.9f\n",ag,worst,mincos);return 0;}
