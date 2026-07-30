#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3_prefill.h"
#include "k3_runtime.h"

typedef struct{uint64_t offset,nbytes;int rows,cols;char name[512];}entry;
static k3_pool pool;
static uint32_t rng=1;
static float rf(void){rng=rng*1664525u+1013904223u;return((rng>>8)&0xffff)/65536.0f-.5f;}
static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void *pa(size_t n){void*p=k3_pool_alloc(&pool,n);if(!p)fprintf(stderr,"%s\n",k3_pool_error(&pool));return p;}
static void pf(void*p){if(p&&k3_pool_free(&pool,p))fprintf(stderr,"%s\n",k3_pool_error(&pool));}
static int read_manifest(const char*path,entry*out,int cap){FILE*f=fopen(path,"r");if(!f)return-1;
    char line[1024];int n=0;while(fgets(line,sizeof line,f)){if(line[0]=='#')continue;
        unsigned long long o,b;char dt[16];int nd,r,c;if(n>=cap||sscanf(line,"%llu %llu %15s %d %d %d %511s",&o,&b,dt,&nd,&r,&c,out[n].name)!=7||strcmp(dt,"BF16")||nd!=2){fclose(f);return-1;}
        out[n++]=(entry){o,b,r,c,""};strncpy(out[n-1].name,strrchr(line,' ')+1,sizeof out[n-1].name-1);out[n-1].name[strcspn(out[n-1].name,"\r\n")]=0;}
    fclose(f);return n;}
static int load_file(const char*path,void**data,size_t*bytes){FILE*f=fopen(path,"rb");if(!f)return-1;
    if(fseek(f,0,SEEK_END)||(*bytes=(size_t)ftell(f),fseek(f,0,SEEK_SET))){fclose(f);return-1;}
    *data=pa(*bytes);if(!*data){fclose(f);return-1;}size_t off=0;while(off<*bytes){size_t n=fread((uint8_t*)*data+off,1,*bytes-off>8388608?8388608:*bytes-off,f);if(!n){fclose(f);return-1;}off+=n;}
    fclose(f);return 0;}
static int bench(const char*label,const k3_bf16_matrix*m,int threads){
    size_t pb=k3_prefill_bf16_packed_bytes(m->rows,m->cols),sb=k3_prefill_bf16_scratch_bytes(1024,m->cols);
    uint16_t*p=pa(pb);float*x=pa((size_t)1024*m->cols*4),*y=pa((size_t)1024*m->rows*4),*scratch=pa(sb);
    if(!p||!x||!y||!scratch)return 1;for(size_t i=0;i<(size_t)1024*m->cols;++i)x[i]=rf()*.125f;
    double pt=now_sec();int fail=k3_prefill_pack_bf16_pv(p,pb,m,threads);pt=now_sec()-pt;
    /* M=3 exercises both the 8-token and output-row tail copies. */
    if(!fail)fail=k3_prefill_gemm_bf16_pv(y,x,3,m,p,scratch,sb,threads);
    double maxe=0,se=0,sr=0;for(int t=0;t<2&&!fail;++t)for(int r=0;r<m->rows;r+=m->rows/31+1){double ref=0;
        for(int k=0;k<m->cols;++k)ref+=(double)x[(size_t)t*m->cols+k]*bf16_to_f32_scalar(m->weight[(size_t)r*m->cols+k]);
        double d=y[(size_t)t*m->rows+r]-ref;if(fabs(d)>maxe)maxe=fabs(d);se+=d*d;sr+=ref*ref;}
    double rel=sqrt(se/(sr+1e-30));printf("[prefill-bf16-%s] sample_max_abs=%.3e sample_rel_l2=%.3e %s pack_ms=%.3f packed_MiB=%.2f\n",label,maxe,rel,maxe<2e-3&&rel<2e-4?"OK":"FAIL",pt*1e3,pb/1048576.0);fail|=maxe>=2e-3||rel>=2e-4;
    const int batches[]={64,256,1024};for(int bi=0;bi<3&&!fail;++bi){int b=batches[bi],reps=b==1024?3:5;double sum=0,best=1e9;
        for(int it=0;it<reps;++it){double t=now_sec();fail|=k3_prefill_gemm_bf16_pv(y,x,b,m,p,scratch,sb,threads);double dt=now_sec()-t;sum+=dt;if(dt<best)best=dt;}
        double mean=sum/reps,flops=2.0*b*m->rows*(double)m->cols;printf("PROBE prefill-bf16 mode=%s M=%d threads=%d mean_ms=%.3f best_ms=%.3f TFLOP/s=%.3f best_TFLOP/s=%.3f\n",label,b,threads,mean*1e3,best*1e3,flops/mean/1e12,flops/best/1e12);}
    pf(p);pf(x);pf(y);pf(scratch);return fail;}
int main(int argc,char**argv){int threads=48;if(argc!=3&&argc!=5){fprintf(stderr,"usage: %s [--threads N] BLOB MANIFEST\n",argv[0]);return 2;}
    int base=1;if(argc==5){if(strcmp(argv[1],"--threads")){fprintf(stderr,"unknown option: %s\n",argv[1]);return 2;}char*e;long v=strtol(argv[2],&e,10);if(*e||v<1||v>48)return 2;threads=(int)v;base=3;}
    k3_pool_init(&pool,"prefill-probe");void*blob=NULL;size_t bytes=0;entry es[8];int n=read_manifest(argv[base+1],es,8);if(n<1||load_file(argv[base],&blob,&bytes)){fprintf(stderr,"prefill probe load failed: %s\n",strerror(errno));k3_pool_destroy(&pool);return 2;}
    int fail=0;for(int i=0;i<n;++i){if(es[i].offset>bytes||es[i].nbytes>bytes-es[i].offset||es[i].nbytes!=(uint64_t)es[i].rows*es[i].cols*2){fail=1;break;}
        const char*label=strstr(es[i].name,"gate.weight")?"router":strstr(es[i].name,"down_proj")?"down":"up";
        k3_bf16_matrix m={(const uint16_t*)((uint8_t*)blob+es[i].offset),es[i].rows,es[i].cols};fail|=bench(label,&m,threads);}
    pf(blob);fprintf(stderr,"k3 prefill pool: peak=%.2f MiB reserved=%.2f MiB\n",pool.peak_active_bytes/1048576.0,pool.reserved_bytes/1048576.0);k3_pool_destroy(&pool);printf("K3 dense prefill probe: %s\n",fail?"FAIL":"PASS");return fail?1:0;}
