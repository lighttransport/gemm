#define _GNU_SOURCE
#include "fp4_gemm.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {size_t off,bytes;char dtype[16],name[192];int ndim;long shape[3];} record;
static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static unsigned rng=7;static float rnd(void){rng=rng*1664525u+1013904223u;return ((rng>>8)/8388608.0f)-1.0f;}
static size_t mem_available(void){FILE*f=fopen("/proc/meminfo","r");char line[256];size_t kb=0;while(f&&fgets(line,sizeof(line),f))if(sscanf(line,"MemAvailable: %zu kB",&kb)==1)break;if(f)fclose(f);return kb<<10;}

static int load_manifest(const char*path,record**out){FILE*f=fopen(path,"r");if(!f)return-1;int cap=2048,n=0;record*r=calloc(cap,sizeof(*r));char line[512];
    while(fgets(line,sizeof(line),f)){if(line[0]=='#')continue;record x={0};long d0=0,d1=0;int got=sscanf(line,"%zu %zu %15s %d %ld %ld %191s",&x.off,&x.bytes,x.dtype,&x.ndim,&d0,&d1,x.name);if(got!=7||x.ndim!=2)continue;x.shape[0]=d0;x.shape[1]=d1;if(n==cap){cap*=2;r=realloc(r,(size_t)cap*sizeof(*r));if(!r){fclose(f);return-1;}}r[n++]=x;}
    fclose(f);*out=r;return n;}
static record*find_rec(record*r,int n,const char*name){for(int i=0;i<n;++i)if(!strcmp(r[i].name,name))return&r[i];return NULL;}

static void*load_anon(const char*path,size_t*bytes){int fd=open(path,O_RDONLY);if(fd<0)return NULL;struct stat st;if(fstat(fd,&st)){close(fd);return NULL;}if(mem_available()<(size_t)st.st_size+(12ULL<<30)){fprintf(stderr,"memory guard: need blob + 12 GiB free\n");close(fd);return NULL;}
    uint8_t*p=mmap(NULL,(size_t)st.st_size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(p==MAP_FAILED){close(fd);return NULL;}size_t done=0;while(done<(size_t)st.st_size){size_t z=(size_t)st.st_size-done;if(z>(8<<20))z=8<<20;ssize_t nr=pread(fd,p+done,z,(off_t)done);if(nr!=(ssize_t)z){munmap(p,(size_t)st.st_size);close(fd);return NULL;}posix_fadvise(fd,(off_t)done,(off_t)z,POSIX_FADV_DONTNEED);done+=z;}close(fd);*bytes=(size_t)st.st_size;return p;}

static float fp8e4m3(uint8_t b){int sign=b>>7,e=(b>>3)&15,m=b&7;float v;if(e==0)v=ldexpf((float)m,-9);else if(e==15&&m==7)v=NAN;else v=ldexpf(1.0f+m/8.0f,e-7);return sign?-v:v;}
static float e8m0(uint8_t b){return b==255?INFINITY:ldexpf(1.0f,(int)b-127);}

static float*dequant_fp8(const uint8_t*blob,const record*w,const record*s){int n=(int)w->shape[0],k=(int)w->shape[1];if(strcmp(w->dtype,"F8_E4M3")||strcmp(s->dtype,"F8_E8M0"))return NULL;float*out=malloc((size_t)n*k*4);if(!out)return NULL;const uint8_t*wp=blob+w->off,*sp=blob+s->off;int sk=(int)s->shape[1];
#pragma omp parallel for schedule(static)
    for(int r=0;r<n;++r)for(int c=0;c<k;++c)out[(size_t)r*k+c]=fp8e4m3(wp[(size_t)r*k+c])*e8m0(sp[(size_t)(r/128)*sk+c/128]);return out;}
static float*dequant_native_mx(const uint8_t*blob,const record*w,const record*s){int n=(int)w->shape[0],pk=(int)w->shape[1],k=pk*2;float*out=malloc((size_t)n*k*4);if(!out)return NULL;const uint8_t*wp=blob+w->off,*sp=blob+s->off;int nb=k/32;
#pragma omp parallel for schedule(static)
    for(int r=0;r<n;++r)for(int c=0;c<k;++c){uint8_t b=wp[(size_t)r*pk+c/2],q=(b>>((c&1)*4))&15;out[(size_t)r*k+c]=fp4_e2m1_decode(q)*e8m0(sp[(size_t)r*nb+c/32]);}return out;}
static int native_mx(fp4_matrix*p,const uint8_t*blob,const record*w,const record*s){int n=(int)w->shape[0],k=(int)w->shape[1]*2;if(fp4_matrix_alloc(p,FP4_MX,n,k))return-1;memcpy(p->codes,blob+w->off,p->code_bytes);memcpy(p->scales,blob+s->off,p->scale_bytes);return 0;}

static void sampled_error(const float*c,const _Float16*a,const fp4_matrix*w,int m,double*rel,double*mx){double num=0,den=0,ma=0;int step=w->n/64;if(step<1)step=1;for(int i=0;i<m&&i<2;++i)for(int r=0;r<w->n;r+=step){double ref=0;for(int k=0;k<w->k;++k)ref+=(double)(float)a[(size_t)i*w->k+k]*(double)(float)(_Float16)fp4_dequant_value(w,r,k);double d=c[(size_t)i*w->n+r]-ref;num+=d*d;den+=ref*ref;if(fabs(d)>ma)ma=fabs(d);}*rel=sqrt(num/(den+1e-30));*mx=ma;}

static double bench_one(const char*name,const fp4_matrix*w,int m,int kc,int threads,int reps,double*rel,double*mx){_Float16*a=aligned_alloc(256,((size_t)m*w->k*2+255)&~255ULL);float*c=aligned_alloc(256,((size_t)m*w->n*4+255)&~255ULL);for(size_t i=0;i<(size_t)m*w->k;++i)a[i]=(_Float16)(rnd()*0.25f);fp4_gemm_f16(c,a,w,m,kc,threads);double best=1e30,sum=0;for(int q=0;q<reps;++q){double t=now_sec();fp4_gemm_f16(c,a,w,m,kc,threads);double d=now_sec()-t;sum+=d;if(d<best)best=d;}sampled_error(c,a,w,m,rel,mx);double med=sum/reps;double gf=2.0*m*w->n*w->k/med/1e9;printf("tensor=%s format=%s M=%d N=%d K=%d kc=%d thr=%d ms=%.3f best_ms=%.3f gflops=%.2f rel_l2=%.3e max_abs=%.3e\n",name,fp4_format_name(w->format),m,w->n,w->k,kc,threads,med*1e3,best*1e3,gf,*rel,*mx);free(a);free(c);return med;}

static void quant_error(const float*base,const fp4_matrix*p,double*rel,double*mx){size_t total=(size_t)p->n*p->k,step=total/(1<<20);if(step<1)step=1;double num=0,den=0,ma=0;for(size_t z=0;z<total;z+=step){int r=(int)(z/p->k),c=(int)(z%p->k);double d=fp4_dequant_value(p,r,c)-base[z];num+=d*d;den+=(double)base[z]*base[z];if(fabs(d)>ma)ma=fabs(d);}*rel=sqrt(num/(den+1e-30));*mx=ma;}
static int run_weight(const char*name,const uint8_t*blob,record*wr,record*sr,int native,int threads,const int*ms,int nm,int reps){float*base=native?dequant_native_mx(blob,wr,sr):dequant_fp8(blob,wr,sr);if(!base){fprintf(stderr,"dequant failed: %s\n",name);return 1;}int n=(int)wr->shape[0],k=(int)wr->shape[1]*(native?2:1),kcs[]={32,64,128,256,0};for(int f=0;f<3;++f){fp4_matrix p;if(f==0&&native){if(native_mx(&p,blob,wr,sr))return 1;}else{if(fp4_matrix_alloc(&p,(fp4_format)f,n,k)||fp4_quantize_f32(&p,base))return 1;}double qrel,qmax;quant_error(base,&p,&qrel,&qmax);printf("quant tensor=%s format=%s source=%s rel_l2=%.3e max_abs=%.3e\n",name,fp4_format_name((fp4_format)f),native?"native-mxfp4":"checkpoint-fp8",qrel,qmax);for(int mi=0;mi<nm;++mi)for(int ci=0;ci<5;++ci)if(!kcs[ci]||kcs[ci]<=k){double rel,mx;bench_one(name,&p,ms[mi],kcs[ci],threads,reps,&rel,&mx);}fp4_matrix_free(&p);}free(base);return 0;}

static int scan_weight(const uint8_t*blob,record*wr,record*sr,int threads,double sec[3],double flop[3]){float*base=dequant_native_mx(blob,wr,sr);if(!base)return 1;int n=(int)wr->shape[0],k=(int)wr->shape[1]*2;_Float16*a=aligned_alloc(256,((size_t)k*2+255)&~255ULL);float*c=aligned_alloc(256,((size_t)n*4+255)&~255ULL);if(!a||!c)return 1;for(int i=0;i<k;++i)a[i]=(_Float16)(rnd()*.25f);for(int f=0;f<3;++f){fp4_matrix p;if(f==0){if(native_mx(&p,blob,wr,sr))return 1;}else if(fp4_matrix_alloc(&p,(fp4_format)f,n,k)||fp4_quantize_f32(&p,base))return 1;double t=now_sec();fp4_gemm_f16(c,a,&p,1,32,threads);sec[f]+=now_sec()-t;flop[f]+=2.0*n*k;fp4_matrix_free(&p);}free(a);free(c);free(base);return 0;}

int main(int argc,char**argv){const char*dir="/local/u14346/ds4f-fp4-gemm";int threads=48,experts=256,reps=3,full=0,skip_attn=0,skip_experts=0;for(int i=1;i<argc;++i){if(!strcmp(argv[i],"--data")&&i+1<argc)dir=argv[++i];else if(!strcmp(argv[i],"--threads")&&i+1<argc)threads=atoi(argv[++i]);else if(!strcmp(argv[i],"--experts")&&i+1<argc)experts=atoi(argv[++i]);else if(!strcmp(argv[i],"--reps")&&i+1<argc)reps=atoi(argv[++i]);else if(!strcmp(argv[i],"--full"))full=1;else if(!strcmp(argv[i],"--skip-attention"))skip_attn=1;else if(!strcmp(argv[i],"--skip-experts"))skip_experts=1;else{fprintf(stderr,"usage: %s [--data DIR] [--threads N] [--experts N] [--reps N] [--full] [--skip-attention|--skip-experts]\n",argv[0]);return 2;}}
    char mp[512],bp[512];snprintf(mp,sizeof(mp),"%s/layer0.manifest",dir);snprintf(bp,sizeof(bp),"%s/layer0.raw",dir);record*r=NULL;int nr=load_manifest(mp,&r);if(nr<1){perror(mp);return 2;}size_t blob_bytes=0;uint8_t*blob=load_anon(bp,&blob_bytes);if(!blob){perror(bp);free(r);return 2;}printf("loaded %.3f GiB into anonymous HBM; MemAvailable %.3f GiB\n",blob_bytes/(double)(1ULL<<30),mem_available()/(double)(1ULL<<30));
    const char*att[]={"wkv","wq_a","wq_b","wo_a","wo_b"};int ms_quick[]={1,8},ms_full[]={1,8,32,128,512};if(!skip_attn)for(int i=0;i<5;++i){char wn[192],sn[192];snprintf(wn,sizeof(wn),"layers.0.attn.%s.weight",att[i]);snprintf(sn,sizeof(sn),"layers.0.attn.%s.scale",att[i]);record*wr=find_rec(r,nr,wn),*sr=find_rec(r,nr,sn);if(!wr||!sr){fprintf(stderr,"missing %s\n",wn);return 2;}if(run_weight(wn,blob,wr,sr,0,threads,full?ms_full:ms_quick,full?5:2,reps))return 1;}
    int ems_quick[]={1,8},ems_full[]={1,8,32,128};if(experts>256)experts=256;
    if(!skip_experts&&experts>0)for(int wi=1;wi<=3;++wi){char wn[192],sn[192];snprintf(wn,sizeof(wn),"layers.0.ffn.experts.0.w%d.weight",wi);snprintf(sn,sizeof(sn),"layers.0.ffn.experts.0.w%d.scale",wi);record*wr=find_rec(r,nr,wn),*sr=find_rec(r,nr,sn);if(!wr||!sr)return 2;if(run_weight(wn,blob,wr,sr,1,threads,full?ems_full:ems_quick,full?4:2,reps))return 1;}
    if(!skip_experts&&experts>1){double sec[3]={0},flop[3]={0};for(int e=1;e<experts;++e)for(int wi=1;wi<=3;++wi){char wn[192],sn[192];snprintf(wn,sizeof(wn),"layers.0.ffn.experts.%d.w%d.weight",e,wi);snprintf(sn,sizeof(sn),"layers.0.ffn.experts.%d.w%d.scale",e,wi);record*wr=find_rec(r,nr,wn),*sr=find_rec(r,nr,sn);if(!wr||!sr||scan_weight(blob,wr,sr,threads,sec,flop))return 2;}for(int f=0;f<3;++f)printf("expert_scan format=%s experts=%d projections=%d M=1 kc=32 thr=%d total_ms=%.3f effective_gflops=%.2f\n",fp4_format_name((fp4_format)f),experts-1,(experts-1)*3,threads,sec[f]*1e3,flop[f]/sec[f]/1e9);}
    munmap(blob,blob_bytes);free(r);return 0;}
