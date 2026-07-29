#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "k3_moe.h"
#include "k3_runtime.h"

typedef struct { uint64_t offset,nbytes,rows,cols; char dtype[16],name[512]; } entry;
typedef struct { uint8_t *blob; size_t size; k3_mxfp4_matrix w1,w2,w3; } loaded_expert;
static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int load_manifest(const char*path,entry*es,int cap){
    FILE*f=fopen(path,"r");if(!f)return-1;char line[1024];int n=0;
    while(fgets(line,sizeof(line),f)){if(line[0]=='#')continue;entry e;int nd;unsigned long long o,b,r,c;
        if(sscanf(line,"%llu %llu %15s %d %llu %llu %511s",&o,&b,e.dtype,&nd,&r,&c,e.name)!=7||nd!=2||n>=cap){fclose(f);return-1;}
        e.offset=o;e.nbytes=b;e.rows=r;e.cols=c;es[n++]=e;}
    fclose(f);return n;
}
static entry*find_entry(entry*es,int n,const char*suf){size_t sl=strlen(suf);for(int i=0;i<n;++i){size_t nl=strlen(es[i].name);if(nl>=sl&&!strcmp(es[i].name+nl-sl,suf))return&es[i];}return NULL;}
static uint64_t rs=UINT64_C(0x4b334d4f45000001);
static uint64_t rn(void){uint64_t z=(rs+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
static float rf(void){return((rn()>>40)/8388608.0f)-1.0f;}
static float max_abs(const float*a,const float*b,size_t n){float e=0;for(size_t i=0;i<n;++i)e=fmaxf(e,fabsf(a[i]-b[i]));return e;}
static void evict(float*b,size_t n,int threads){omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;i+=16)b[i]+=1.0f;
}

static int dispatch_unit(void){
    enum{B=8,TK=16,W=64};int routes[B*TK],local[1]={0},counts[1],tok[B];float rw[B*TK],tw[B];
    float src[B*W],gather[B*W],expert[B*W],got[B*W],ref[B*W];memset(got,0,sizeof(got));memset(ref,0,sizeof(ref));
    for(int t=0;t<B;++t){float sum=0;for(int k=0;k<TK;++k){routes[t*TK+k]=(k==t%TK)?0:100+t*TK+k;rw[t*TK+k]=.1f+.01f*k;sum+=rw[t*TK+k];}for(int k=0;k<TK;++k)rw[t*TK+k]/=sum;}
    for(int i=0;i<B*W;++i)src[i]=rf();
    int n=k3_moe_build_dispatch(routes,rw,B,TK,local,1,counts,tok,tw);
    if(n!=B||counts[0]!=B)return 1;k3_moe_gather(gather,src,tok,B,W);
    for(int p=0;p<B;++p)for(int i=0;i<W;++i)expert[p*W+i]=gather[p*W+i]*(1.f+.01f*i);
    k3_moe_scatter_add(got,expert,tok,tw,B,W);
    for(int p=0;p<B;++p)for(int i=0;i<W;++i)ref[tok[p]*W+i]+=tw[p]*expert[p*W+i];
    float e=max_abs(got,ref,B*W);printf("[dispatch] assignments=%d max_abs=%.3e %s\n",n,e,e==0?"OK":"FAIL");return e!=0;
}

static int batched_correctness(const k3_mxfp4_matrix*w1,const k3_mxfp4_matrix*w2,const k3_mxfp4_matrix*w3){
    int m=2;float*x=malloc((size_t)m*K3_LATENT*4),*yb=malloc((size_t)m*K3_LATENT*4),*yr=malloc((size_t)m*K3_LATENT*4);
    float*g=malloc((size_t)m*K3_EXPERT_INTER*4),*u=malloc((size_t)m*K3_EXPERT_INTER*4);
    float*g1=malloc((size_t)K3_EXPERT_INTER*4),*u1=malloc((size_t)K3_EXPERT_INTER*4);
    for(int i=0;i<m*K3_LATENT;++i)x[i]=rf()*.125f;
    k3_expert_forward_mxfp4(yb,w1,w2,w3,x,m,g,u,24);
    for(int t=0;t<m;++t)k3_expert_forward_mxfp4(yr+(size_t)t*K3_LATENT,w1,w2,w3,x+(size_t)t*K3_LATENT,1,g1,u1,24);
    float e=max_abs(yb,yr,(size_t)m*K3_LATENT);printf("[batch-2x] max_abs=%.3e %s\n",e,e<2e-6?"OK":"FAIL");
    free(x);free(yb);free(yr);free(g);free(u);free(g1);free(u1);return e>=2e-6;
}

static int tiled_correctness(const k3_mxfp4_matrix*w1,const k3_mxfp4_matrix*w2,const k3_mxfp4_matrix*w3){
    enum{M=32};float*x=malloc((size_t)M*K3_LATENT*4),*yt=malloc((size_t)M*K3_LATENT*4),*yr=malloc((size_t)M*K3_LATENT*4);
    float*gt=malloc((size_t)M*K3_EXPERT_INTER*4),*ut=malloc((size_t)M*K3_EXPERT_INTER*4);
    float*gr=malloc((size_t)M*K3_EXPERT_INTER*4),*ur=malloc((size_t)M*K3_EXPERT_INTER*4);
    if(!x||!yt||!yr||!gt||!ut||!gr||!ur)return 1;
    for(int i=0;i<M*K3_LATENT;++i)x[i]=rf()*.125f;
    k3_expert_forward_mxfp4_mode(yr,w1,w2,w3,x,M,gr,ur,48,0);
    k3_expert_forward_mxfp4_mode(yt,w1,w2,w3,x,M,gt,ut,48,8);
    float e=max_abs(yt,yr,(size_t)M*K3_LATENT);printf("[tile-vs-svtbl] M=32 max_abs=%.3e %s\n",e,e<2e-4?"OK":"FAIL");
    free(x);free(yt);free(yr);free(gt);free(ut);free(gr);free(ur);return e>=2e-4;
}

static int situ_fast_correctness(const k3_mxfp4_matrix*w1,
                                 const k3_mxfp4_matrix*w2,
                                 const k3_mxfp4_matrix*w3){
    enum{M=2};int n=M*K3_EXPERT_INTER;
    float*x=malloc((size_t)M*K3_LATENT*4),*u=malloc((size_t)n*4);
    float*gf=malloc((size_t)n*4),*ge=malloc((size_t)n*4);
    float*yf=malloc((size_t)M*K3_LATENT*4),*ye=malloc((size_t)M*K3_LATENT*4);
    if(!x||!u||!gf||!ge||!yf||!ye)return 1;
    for(int i=0;i<M*K3_LATENT;++i)x[i]=rf()*.125f;
    k3_mxfp4_gemm2_mode(gf,w1,u,w3,x,M,24,8);memcpy(ge,gf,(size_t)n*4);
    k3_moe_situ(gf,u,n,24);
    for(int i=0;i<n;++i)ge[i]=4.0f*tanhf(ge[i]*.25f)*k3_sigmoidf(ge[i])
        *25.0f*tanhf(u[i]*.04f);
    k3_mxfp4_gemm_mode(yf,w2,gf,M,24,8);k3_mxfp4_gemm_mode(ye,w2,ge,M,24,8);
    float e=max_abs(yf,ye,(size_t)M*K3_LATENT);double se=0,sr=0;
    for(int i=0;i<M*K3_LATENT;++i){double d=yf[i]-ye[i];se+=d*d;sr+=(double)ye[i]*ye[i];}
    double rel=sqrt(se/(sr+1e-30));printf("[situ-fexpa-real] max_abs=%.3e rel_l2=%.3e %s\n",e,rel,e<2e-4&&rel<5e-4?"OK":"FAIL");
    free(x);free(u);free(gf);free(ge);free(yf);free(ye);return e>=2e-4||rel>=5e-4;
}

static int local_scheduler_correctness(const k3_mxfp4_matrix*w1,const k3_mxfp4_matrix*w2,const k3_mxfp4_matrix*w3){
    enum{B=8,E=4,TK=2};int routes[B*TK],ids[E]={0,1,2,3},counts[E],tok[E*B];float rw[B*TK],tw[E*B];
    k3_mxfp4_matrix a1[E],a2[E],a3[E];for(int e=0;e<E;++e){a1[e]=*w1;a2[e]=*w2;a3[e]=*w3;}
    float*x=malloc((size_t)B*K3_LATENT*4),*gx=malloc((size_t)E*B*K3_LATENT*4);
    float*gate=malloc((size_t)E*B*K3_EXPERT_INTER*4),*up=malloc((size_t)E*B*K3_EXPERT_INTER*4);
    float*eo=malloc((size_t)E*B*K3_LATENT*4),*got=malloc((size_t)B*K3_LATENT*4),*ref=calloc((size_t)B*K3_LATENT,4);
    float*rg=malloc((size_t)B*K3_EXPERT_INTER*4),*ru=malloc((size_t)B*K3_EXPERT_INTER*4),*ro=malloc((size_t)B*K3_LATENT*4);
    if(!x||!gx||!gate||!up||!eo||!got||!ref||!rg||!ru||!ro)return 1;
    for(int t=0;t<B;++t){routes[t*TK]=t%E;routes[t*TK+1]=(t+1)%E;rw[t*TK]=.6f;rw[t*TK+1]=.4f;for(int i=0;i<K3_LATENT;++i)x[(size_t)t*K3_LATENT+i]=rf()*.125f;}
    int n=k3_moe_build_dispatch(routes,rw,B,TK,ids,E,counts,tok,tw);
    for(int e=0;e<E;++e)k3_moe_gather(gx+(size_t)e*B*K3_LATENT,x,tok+e*B,counts[e],K3_LATENT);
    k3_moe_forward_local_mxfp4(got,a1,a2,a3,E,counts,tok,tw,gx,B,gate,up,eo,48,8);
    for(int e=0;e<E;++e){k3_expert_forward_mxfp4(ro,w1,w2,w3,gx+(size_t)e*B*K3_LATENT,counts[e],rg,ru,48);k3_moe_scatter_add(ref,ro,tok+e*B,tw+e*B,counts[e],K3_LATENT);}
    float err=max_abs(got,ref,(size_t)B*K3_LATENT);printf("[local-scheduler] assignments=%d max_abs=%.3e %s\n",n,err,err<2e-6?"OK":"FAIL");
    free(x);free(gx);free(gate);free(up);free(eo);free(got);free(ref);free(rg);free(ru);free(ro);return err>=2e-6;
}

static void perf(const k3_mxfp4_matrix*w1,const k3_mxfp4_matrix*w2,const k3_mxfp4_matrix*w3,int batch,int threads,int tile_threshold){
    float*x=malloc((size_t)batch*K3_LATENT*4),*y=malloc((size_t)batch*K3_LATENT*4);
    float*g=malloc((size_t)batch*K3_EXPERT_INTER*4),*u=malloc((size_t)batch*K3_EXPERT_INTER*4);
    size_t en=(size_t)128*1024*1024/4;float*eb=calloc(en,4);for(int i=0;i<batch*K3_LATENT;++i)x[i]=rf()*.125f;
    int iters=batch<=2?10:batch<=8?6:3;double sec=0;
    for(int it=0;it<iters;++it){evict(eb,en,threads);double t0=now_sec();k3_expert_forward_mxfp4_mode(y,w1,w2,w3,x,batch,g,u,threads,tile_threshold);sec+=now_sec()-t0;}
    double bytes=(double)(w1->rows*(w1->cols/2+w1->cols/32)+w3->rows*(w3->cols/2+w3->cols/32)+w2->rows*(w2->cols/2+w2->cols/32));
    double sum=0;for(int i=0;i<batch*K3_LATENT;++i)sum+=y[i];
    printf("PROBE expert kernel=%s batch=%2d threads=%2d ms=%.3f tok/s=%.1f distinct_GB/s=%.2f checksum=%+.6e\n",
           tile_threshold?"tile":"svtbl",batch,threads,sec/iters*1e3,batch/(sec/iters),bytes/(sec/iters)/1e9,sum);
    free(x);free(y);free(g);free(u);free(eb);
}

static void profile_stages(const k3_mxfp4_matrix *w1,
                           const k3_mxfp4_matrix *w2,
                           const k3_mxfp4_matrix *w3, int batch,
                           int threads, int tile_threshold) {
    float *x=malloc((size_t)batch*K3_LATENT*4);
    float *y=malloc((size_t)batch*K3_LATENT*4);
    float *g=malloc((size_t)batch*K3_EXPERT_INTER*4);
    float *u=malloc((size_t)batch*K3_EXPERT_INTER*4);
    size_t en=(size_t)128*1024*1024/4;
    float *eb=calloc(en,4);
    if(!x||!y||!g||!u||!eb)return;
    for(int i=0;i<batch*K3_LATENT;++i)x[i]=rf()*.125f;
    double t13=0,tsitu=0,t2=0;
    int iters=5;
    for(int it=0;it<iters;++it){
        evict(eb,en,threads);
        double t=now_sec();
        k3_mxfp4_gemm2_mode(g,w1,u,w3,x,batch,threads,tile_threshold);
        t13+=now_sec()-t;
        t=now_sec();
        omp_set_num_threads(threads);
        int vl=(int)svcntw();
#pragma omp parallel for schedule(static)
        for(int i=0;i<batch*K3_EXPERT_INTER;i+=vl)
            k3_situ_fast_sve(g+i,g+i,u+i,
                batch*K3_EXPERT_INTER-i<vl?batch*K3_EXPERT_INTER-i:vl);
        tsitu+=now_sec()-t;
        t=now_sec();
        k3_mxfp4_gemm_mode(y,w2,g,batch,threads,tile_threshold);
        t2+=now_sec()-t;
    }
    printf("PROFILE expert situ=fexpa batch=%d threads=%d w13_ms=%.3f situ_ms=%.3f w2_ms=%.3f total_ms=%.3f\n",
           batch,threads,t13/iters*1e3,tsitu/iters*1e3,t2/iters*1e3,
           (t13+tsitu+t2)/iters*1e3);
    free(x);free(y);free(g);free(u);free(eb);
}

static int load_expert(const char *blob_path,const char *manifest_path,loaded_expert *out){
    entry es[8];int ne=load_manifest(manifest_path,es,8);if(ne!=6)return-1;
    size_t size=0;uint8_t*blob=k3_load_blob_anon(blob_path,&size);if(!blob)return-1;
    entry*e1=find_entry(es,ne,"w1.weight_packed"),*s1=find_entry(es,ne,"w1.weight_scale");
    entry*e2=find_entry(es,ne,"w2.weight_packed"),*s2=find_entry(es,ne,"w2.weight_scale");
    entry*e3=find_entry(es,ne,"w3.weight_packed"),*s3=find_entry(es,ne,"w3.weight_scale");
    if(!e1||!s1||!e2||!s2||!e3||!s3){free(blob);return-1;}
    out->blob=blob;out->size=size;
    out->w1=(k3_mxfp4_matrix){blob+e1->offset,blob+s1->offset,(int)e1->rows,(int)e1->cols*2};
    out->w2=(k3_mxfp4_matrix){blob+e2->offset,blob+s2->offset,(int)e2->rows,(int)e2->cols*2};
    out->w3=(k3_mxfp4_matrix){blob+e3->offset,blob+s3->offset,(int)e3->rows,(int)e3->cols*2};
    return 0;
}

static void multi_expert_perf(const loaded_expert *le,int nlocal,int threads){
    int*counts=malloc((size_t)nlocal*4),*tok=malloc((size_t)nlocal*nlocal*4);float*tw=malloc((size_t)nlocal*nlocal*4);
    k3_mxfp4_matrix*w1=malloc((size_t)nlocal*sizeof(*w1)),*w2=malloc((size_t)nlocal*sizeof(*w2)),*w3=malloc((size_t)nlocal*sizeof(*w3));
    float*x=malloc((size_t)nlocal*nlocal*K3_LATENT*4),*gate=malloc((size_t)nlocal*nlocal*K3_EXPERT_INTER*4);
    float*up=malloc((size_t)nlocal*nlocal*K3_EXPERT_INTER*4),*eo=malloc((size_t)nlocal*nlocal*K3_LATENT*4),*dst=malloc((size_t)nlocal*K3_LATENT*4);
    size_t en=(size_t)128*1024*1024/4;float*eb=calloc(en,4);if(!counts||!tok||!tw||!w1||!w2||!w3||!x||!gate||!up||!eo||!dst||!eb)return;
    for(int e=0;e<nlocal;++e){counts[e]=1;tok[e*nlocal]=e;tw[e*nlocal]=1;w1[e]=le[e].w1;w2[e]=le[e].w2;w3[e]=le[e].w3;for(int i=0;i<K3_LATENT;++i)x[((size_t)e*nlocal)*K3_LATENT+i]=rf()*.125f;}
    double sec=0;int iters=8;
    for(int it=0;it<iters;++it){evict(eb,en,threads);double t0=now_sec();k3_moe_forward_local_mxfp4(dst,w1,w2,w3,nlocal,counts,tok,tw,x,nlocal,gate,up,eo,threads,8);sec+=now_sec()-t0;}
    double sum=0;for(int i=0;i<nlocal*K3_LATENT;++i)sum+=dst[i];
    printf("PROBE local-scheduler experts=%d assignments=%d threads=%d ms=%.3f assignments/s=%.1f checksum=%+.6e\n",nlocal,nlocal,threads,sec/iters*1e3,nlocal/(sec/iters),sum);
    free(counts);free(tok);free(tw);free(w1);free(w2);free(w3);free(x);free(gate);free(up);free(eo);free(dst);free(eb);
}

int main(int argc,char**argv){
    if(argc<3||!(argc&1)){fprintf(stderr,"usage: %s BLOB MANIFEST [BLOB MANIFEST ...]\n",argv[0]);return 2;}
    int nlocal=(argc-1)/2;if(nlocal>16)return 2;loaded_expert le[16];memset(le,0,sizeof(le));k3_apply_numa_interleave();
    for(int e=0;e<nlocal;++e)if(load_expert(argv[1+2*e],argv[2+2*e],&le[e])){fprintf(stderr,"load expert %d failed\n",e);return 2;}
    k3_mxfp4_matrix w1=le[0].w1,w2=le[0].w2,w3=le[0].w3;
    int fail=dispatch_unit()|batched_correctness(&w1,&w2,&w3)|tiled_correctness(&w1,&w2,&w3)|situ_fast_correctness(&w1,&w2,&w3)|local_scheduler_correctness(&w1,&w2,&w3);
    int batches[]={1,2,4,8,16,32},threads[]={24,48};
    int tile_threshold=getenv("K3_MXFP4_TILE")?atoi(getenv("K3_MXFP4_TILE")):8;
    for(int ti=0;ti<2;++ti)for(int bi=0;bi<6;++bi)perf(&w1,&w2,&w3,batches[bi],threads[ti],tile_threshold);
    profile_stages(&w1,&w2,&w3,32,48,tile_threshold);
    for(int e=1;e<=nlocal;e*=2)multi_expert_perf(le,e,48);
    if(nlocal>2&&(nlocal&(nlocal-1)))multi_expert_perf(le,nlocal,48);
    for(int e=0;e<nlocal;++e)free(le[e].blob);printf("K3 MoE probe: %s\n",fail?"FAIL":"PASS");return fail?1:0;
}
