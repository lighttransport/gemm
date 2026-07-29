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
static k3_pool probe_pool;
static int probe_alloc_failed;
static void *probe_alloc(size_t bytes){void*p=k3_pool_alloc(&probe_pool,bytes);if(!p){probe_alloc_failed=1;fprintf(stderr,"%s\n",k3_pool_error(&probe_pool));}return p;}
static void *probe_calloc(size_t count,size_t size){void*p=k3_pool_calloc(&probe_pool,count,size);if(!p){probe_alloc_failed=1;fprintf(stderr,"%s\n",k3_pool_error(&probe_pool));}return p;}
static void probe_free(void*ptr){if(k3_pool_free(&probe_pool,ptr))fprintf(stderr,"%s\n",k3_pool_error(&probe_pool));}
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
    int m=2;float*x=probe_alloc((size_t)m*K3_LATENT*4),*yb=probe_alloc((size_t)m*K3_LATENT*4),*yr=probe_alloc((size_t)m*K3_LATENT*4);
    float*g=probe_alloc((size_t)m*K3_EXPERT_INTER*4),*u=probe_alloc((size_t)m*K3_EXPERT_INTER*4);
    float*g1=probe_alloc((size_t)K3_EXPERT_INTER*4),*u1=probe_alloc((size_t)K3_EXPERT_INTER*4);
    if(!x||!yb||!yr||!g||!u||!g1||!u1)return 1;
    for(int i=0;i<m*K3_LATENT;++i)x[i]=rf()*.125f;
    k3_expert_forward_mxfp4(yb,w1,w2,w3,x,m,g,u,24);
    for(int t=0;t<m;++t)k3_expert_forward_mxfp4(yr+(size_t)t*K3_LATENT,w1,w2,w3,x+(size_t)t*K3_LATENT,1,g1,u1,24);
    float e=max_abs(yb,yr,(size_t)m*K3_LATENT);printf("[batch-2x] max_abs=%.3e %s\n",e,e<2e-6?"OK":"FAIL");
    probe_free(x);probe_free(yb);probe_free(yr);probe_free(g);probe_free(u);probe_free(g1);probe_free(u1);return e>=2e-6;
}

static int tiled_correctness(const k3_mxfp4_matrix*w1,const k3_mxfp4_matrix*w2,const k3_mxfp4_matrix*w3){
    enum{M=32};float*x=probe_alloc((size_t)M*K3_LATENT*4),*yt=probe_alloc((size_t)M*K3_LATENT*4),*yr=probe_alloc((size_t)M*K3_LATENT*4);
    float*gt=probe_alloc((size_t)M*K3_EXPERT_INTER*4),*ut=probe_alloc((size_t)M*K3_EXPERT_INTER*4);
    float*gr=probe_alloc((size_t)M*K3_EXPERT_INTER*4),*ur=probe_alloc((size_t)M*K3_EXPERT_INTER*4);
    if(!x||!yt||!yr||!gt||!ut||!gr||!ur)return 1;
    for(int i=0;i<M*K3_LATENT;++i)x[i]=rf()*.125f;
    k3_expert_forward_mxfp4_mode(yr,w1,w2,w3,x,M,gr,ur,48,0);
    k3_expert_forward_mxfp4_mode(yt,w1,w2,w3,x,M,gt,ut,48,8);
    float e=max_abs(yt,yr,(size_t)M*K3_LATENT);printf("[tile-vs-svtbl] M=32 max_abs=%.3e %s\n",e,e<2e-4?"OK":"FAIL");
    probe_free(x);probe_free(yt);probe_free(yr);probe_free(gt);probe_free(ut);probe_free(gr);probe_free(ur);return e>=2e-4;
}

static int situ_fast_correctness(const k3_mxfp4_matrix*w1,
                                 const k3_mxfp4_matrix*w2,
                                 const k3_mxfp4_matrix*w3){
    enum{M=2};int n=M*K3_EXPERT_INTER;
    float*x=probe_alloc((size_t)M*K3_LATENT*4),*u=probe_alloc((size_t)n*4);
    float*gf=probe_alloc((size_t)n*4),*ge=probe_alloc((size_t)n*4);
    float*yf=probe_alloc((size_t)M*K3_LATENT*4),*ye=probe_alloc((size_t)M*K3_LATENT*4);
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
    probe_free(x);probe_free(u);probe_free(gf);probe_free(ge);probe_free(yf);probe_free(ye);return e>=2e-4||rel>=5e-4;
}

static int local_scheduler_correctness(const k3_mxfp4_matrix*w1,const k3_mxfp4_matrix*w2,const k3_mxfp4_matrix*w3){
    enum{B=8,E=4,TK=2};int routes[B*TK],ids[E]={0,1,2,3},counts[E],tok[E*B];float rw[B*TK],tw[E*B];
    k3_mxfp4_matrix a1[E],a2[E],a3[E];for(int e=0;e<E;++e){a1[e]=*w1;a2[e]=*w2;a3[e]=*w3;}
    float*x=probe_alloc((size_t)B*K3_LATENT*4),*gx=probe_alloc((size_t)E*B*K3_LATENT*4);
    float*gate=probe_alloc((size_t)E*B*K3_EXPERT_INTER*4),*up=probe_alloc((size_t)E*B*K3_EXPERT_INTER*4);
    float*eo=probe_alloc((size_t)E*B*K3_LATENT*4),*got=probe_alloc((size_t)B*K3_LATENT*4),*ref=probe_calloc((size_t)B*K3_LATENT,4);
    float*rg=probe_alloc((size_t)B*K3_EXPERT_INTER*4),*ru=probe_alloc((size_t)B*K3_EXPERT_INTER*4),*ro=probe_alloc((size_t)B*K3_LATENT*4);
    if(!x||!gx||!gate||!up||!eo||!got||!ref||!rg||!ru||!ro)return 1;
    for(int t=0;t<B;++t){routes[t*TK]=t%E;routes[t*TK+1]=(t+1)%E;rw[t*TK]=.6f;rw[t*TK+1]=.4f;for(int i=0;i<K3_LATENT;++i)x[(size_t)t*K3_LATENT+i]=rf()*.125f;}
    int n=k3_moe_build_dispatch(routes,rw,B,TK,ids,E,counts,tok,tw);
    for(int e=0;e<E;++e)k3_moe_gather(gx+(size_t)e*B*K3_LATENT,x,tok+e*B,counts[e],K3_LATENT);
    k3_moe_forward_local_mxfp4(got,a1,a2,a3,E,counts,tok,tw,gx,B,gate,up,eo,48,8);
    for(int e=0;e<E;++e){k3_expert_forward_mxfp4(ro,w1,w2,w3,gx+(size_t)e*B*K3_LATENT,counts[e],rg,ru,48);k3_moe_scatter_add(ref,ro,tok+e*B,tw+e*B,counts[e],K3_LATENT);}
    float err=max_abs(got,ref,(size_t)B*K3_LATENT);printf("[local-scheduler] assignments=%d max_abs=%.3e %s\n",n,err,err<2e-6?"OK":"FAIL");
    probe_free(x);probe_free(gx);probe_free(gate);probe_free(up);probe_free(eo);probe_free(got);probe_free(ref);probe_free(rg);probe_free(ru);probe_free(ro);return err>=2e-6;
}

static void perf(const k3_mxfp4_matrix*w1,const k3_mxfp4_matrix*w2,const k3_mxfp4_matrix*w3,int batch,int threads,int tile_threshold){
    float*x=probe_alloc((size_t)batch*K3_LATENT*4),*y=probe_alloc((size_t)batch*K3_LATENT*4);
    float*g=probe_alloc((size_t)batch*K3_EXPERT_INTER*4),*u=probe_alloc((size_t)batch*K3_EXPERT_INTER*4);
    size_t en=(size_t)128*1024*1024/4;float*eb=probe_calloc(en,4);
    if(!x||!y||!g||!u||!eb){fprintf(stderr,"k3_moe_probe: skipping perf after allocation failure\n");return;}
    for(int i=0;i<batch*K3_LATENT;++i)x[i]=rf()*.125f;
    int iters=batch<=2?10:batch<=8?6:3;double sec=0;
    for(int it=0;it<iters;++it){evict(eb,en,threads);double t0=now_sec();k3_expert_forward_mxfp4_mode(y,w1,w2,w3,x,batch,g,u,threads,tile_threshold);sec+=now_sec()-t0;}
    double bytes=(double)(w1->rows*(w1->cols/2+w1->cols/32)+w3->rows*(w3->cols/2+w3->cols/32)+w2->rows*(w2->cols/2+w2->cols/32));
    double sum=0;for(int i=0;i<batch*K3_LATENT;++i)sum+=y[i];
    printf("PROBE expert kernel=%s batch=%2d threads=%2d ms=%.3f tok/s=%.1f distinct_GB/s=%.2f checksum=%+.6e\n",
           tile_threshold?"tile":"svtbl",batch,threads,sec/iters*1e3,batch/(sec/iters),bytes/(sec/iters)/1e9,sum);
    probe_free(x);probe_free(y);probe_free(g);probe_free(u);probe_free(eb);
}

static void profile_stages(const k3_mxfp4_matrix *w1,
                           const k3_mxfp4_matrix *w2,
                           const k3_mxfp4_matrix *w3, int batch,
                           int threads, int tile_threshold) {
    float *x=probe_alloc((size_t)batch*K3_LATENT*4);
    float *y=probe_alloc((size_t)batch*K3_LATENT*4);
    float *g=probe_alloc((size_t)batch*K3_EXPERT_INTER*4);
    float *u=probe_alloc((size_t)batch*K3_EXPERT_INTER*4);
    size_t en=(size_t)128*1024*1024/4;
    float *eb=probe_calloc(en,4);
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
    probe_free(x);probe_free(y);probe_free(g);probe_free(u);probe_free(eb);
}

static int load_expert(const char *blob_path,const char *manifest_path,loaded_expert *out){
    entry es[8];int ne=load_manifest(manifest_path,es,8);if(ne!=6){fprintf(stderr,"k3_moe_probe: invalid manifest '%s': expected 6 tensors, got %d\n",manifest_path,ne);return-1;}
    size_t size=0;uint8_t*blob=k3_pool_load_blob(&probe_pool,blob_path,&size);if(!blob){fprintf(stderr,"%s\n",k3_pool_error(&probe_pool));return-1;}
    for(int i=0;i<ne;++i)if(es[i].offset>size||es[i].nbytes>size-es[i].offset){fprintf(stderr,"k3_moe_probe: tensor '%s' exceeds blob '%s'\n",es[i].name,blob_path);probe_free(blob);return-1;}
    entry*e1=find_entry(es,ne,"w1.weight_packed"),*s1=find_entry(es,ne,"w1.weight_scale");
    entry*e2=find_entry(es,ne,"w2.weight_packed"),*s2=find_entry(es,ne,"w2.weight_scale");
    entry*e3=find_entry(es,ne,"w3.weight_packed"),*s3=find_entry(es,ne,"w3.weight_scale");
    if(!e1||!s1||!e2||!s2||!e3||!s3){fprintf(stderr,"k3_moe_probe: manifest '%s' lacks an expert tensor\n",manifest_path);probe_free(blob);return-1;}
    out->blob=blob;out->size=size;
    out->w1=(k3_mxfp4_matrix){blob+e1->offset,blob+s1->offset,(int)e1->rows,(int)e1->cols*2};
    out->w2=(k3_mxfp4_matrix){blob+e2->offset,blob+s2->offset,(int)e2->rows,(int)e2->cols*2};
    out->w3=(k3_mxfp4_matrix){blob+e3->offset,blob+s3->offset,(int)e3->rows,(int)e3->cols*2};
    return 0;
}

static void multi_expert_perf(const loaded_expert *le,int nlocal,int threads){
    int*counts=probe_alloc((size_t)nlocal*4),*tok=probe_alloc((size_t)nlocal*nlocal*4);float*tw=probe_alloc((size_t)nlocal*nlocal*4);
    k3_mxfp4_matrix*w1=probe_alloc((size_t)nlocal*sizeof(*w1)),*w2=probe_alloc((size_t)nlocal*sizeof(*w2)),*w3=probe_alloc((size_t)nlocal*sizeof(*w3));
    float*x=probe_alloc((size_t)nlocal*nlocal*K3_LATENT*4),*gate=probe_alloc((size_t)nlocal*nlocal*K3_EXPERT_INTER*4);
    float*up=probe_alloc((size_t)nlocal*nlocal*K3_EXPERT_INTER*4),*eo=probe_alloc((size_t)nlocal*nlocal*K3_LATENT*4),*dst=probe_alloc((size_t)nlocal*K3_LATENT*4);
    size_t en=(size_t)128*1024*1024/4;float*eb=probe_calloc(en,4);if(!counts||!tok||!tw||!w1||!w2||!w3||!x||!gate||!up||!eo||!dst||!eb)return;
    for(int e=0;e<nlocal;++e){counts[e]=1;tok[e*nlocal]=e;tw[e*nlocal]=1;w1[e]=le[e].w1;w2[e]=le[e].w2;w3[e]=le[e].w3;for(int i=0;i<K3_LATENT;++i)x[((size_t)e*nlocal)*K3_LATENT+i]=rf()*.125f;}
    double sec=0;int iters=8;
    for(int it=0;it<iters;++it){evict(eb,en,threads);double t0=now_sec();k3_moe_forward_local_mxfp4(dst,w1,w2,w3,nlocal,counts,tok,tw,x,nlocal,gate,up,eo,threads,8);sec+=now_sec()-t0;}
    double sum=0;for(int i=0;i<nlocal*K3_LATENT;++i)sum+=dst[i];
    printf("PROBE local-scheduler experts=%d assignments=%d threads=%d ms=%.3f assignments/s=%.1f checksum=%+.6e\n",nlocal,nlocal,threads,sec/iters*1e3,nlocal/(sec/iters),sum);
    probe_free(counts);probe_free(tok);probe_free(tw);probe_free(w1);probe_free(w2);probe_free(w3);probe_free(x);probe_free(gate);probe_free(up);probe_free(eo);probe_free(dst);probe_free(eb);
}

static int tp_slices_correctness(const loaded_expert *le, int nslice, int threads,
                                 int selected) {
    int local=le[0].w1.rows,total=local*nslice;
    if(selected){
        k3_mxfp4_matrix*w1=probe_alloc((size_t)nslice*sizeof(*w1)),*w2=probe_alloc((size_t)nslice*sizeof(*w2)),*w3=probe_alloc((size_t)nslice*sizeof(*w3));
        float*rw=probe_alloc((size_t)nslice*4),*x=probe_alloc((size_t)K3_LATENT*4),*got=probe_alloc((size_t)K3_LATENT*4),*ref=probe_calloc(K3_LATENT,4),*part=probe_alloc((size_t)K3_LATENT*4);
        float*g=probe_alloc((size_t)nslice*local*4),*u=probe_alloc((size_t)nslice*local*4),*eo=probe_alloc((size_t)nslice*K3_LATENT*4),*gl=probe_alloc((size_t)local*4),*ul=probe_alloc((size_t)local*4);
        if(!w1||!w2||!w3||!rw||!x||!got||!ref||!part||!g||!u||!eo||!gl||!ul)return 1;
        for(int e=0;e<nslice;++e){w1[e]=le[e].w1;w2[e]=le[e].w2;w3[e]=le[e].w3;rw[e]=1.0f/nslice;}for(int i=0;i<K3_LATENT;++i)x[i]=rf()*.125f;
        for(int e=0;e<nslice;++e){k3_expert_tp_forward_mxfp4(part,&w1[e],&w2[e],&w3[e],x,1,gl,ul,threads,8);for(int i=0;i<K3_LATENT;++i)ref[i]+=rw[e]*part[i];}
        k3_expert_tp_forward_selected_mxfp4(got,w1,w2,w3,rw,nslice,x,g,u,eo,threads);
        float err=max_abs(got,ref,K3_LATENT);double mean=0,best=1e9;for(int it=0;it<8;++it){double t=now_sec();k3_expert_tp_forward_selected_mxfp4(got,w1,w2,w3,rw,nslice,x,g,u,eo,threads);double ms=(now_sec()-t)*1e3;mean+=ms;if(ms<best)best=ms;}mean/=8;
        printf("[expert-tp-selected] experts=%d channels/rank=%d max_abs=%.3e %s\n",nslice,local,err,err<2e-6?"OK":"FAIL");
        printf("PROBE expert-tp-selected experts=%d channels=%d mean_ms=%.3f best_ms=%.3f projected_stack_ms=%.3f\n",nslice,local,mean,best,mean*K3_MOE_LAYERS);
        probe_free(w1);probe_free(w2);probe_free(w3);probe_free(rw);probe_free(x);probe_free(got);probe_free(ref);probe_free(part);probe_free(g);probe_free(u);probe_free(eo);probe_free(gl);probe_free(ul);return err>=2e-6;
    }
    if(total!=K3_EXPERT_INTER){
        float*x=probe_alloc((size_t)K3_LATENT*4),*y=probe_alloc((size_t)K3_LATENT*4);
        float*g=probe_alloc((size_t)local*4),*u=probe_alloc((size_t)local*4);if(!x||!y||!g||!u)return 1;
        for(int i=0;i<K3_LATENT;++i)x[i]=rf()*.125f;double mean=0,best=1e9,checksum=0;
        for(int q=0;q<nslice;++q){if(k3_expert_tp_forward_mxfp4(y,&le[q].w1,&le[q].w2,&le[q].w3,x,1,g,u,threads,8))return 1;
            for(int it=0;it<8;++it){double t=now_sec();if(k3_expert_tp_forward_mxfp4(y,&le[q].w1,&le[q].w2,&le[q].w3,x,1,g,u,threads,8))return 1;
                double ms=(now_sec()-t)*1e3;mean+=ms;if(ms<best)best=ms;}
            for(int i=0;i<K3_LATENT;++i){if(!isfinite(y[i]))return 1;checksum+=y[i];}}
        mean/=8*nslice;
        printf("[expert-tp-slice] slices=%d channels=%d finite=yes PASS\n",nslice,total);
        printf("PROBE expert-tp local_slices=%d channels=%d mean_ms=%.3f best_ms=%.3f checksum=%+.6e\n",nslice,total,mean,best,checksum);
        probe_free(x);probe_free(y);probe_free(g);probe_free(u);return 0;
    }
    size_t w13=(size_t)K3_EXPERT_INTER*K3_LATENT/2,s13=(size_t)K3_EXPERT_INTER*K3_LATENT/32;
    size_t w2n=(size_t)K3_LATENT*K3_EXPERT_INTER/2,s2n=(size_t)K3_LATENT*K3_EXPERT_INTER/32;
    uint8_t *p1=probe_alloc(w13),*s1=probe_alloc(s13),*p2=probe_alloc(w2n),*s2=probe_alloc(s2n),*p3=probe_alloc(w13),*s3=probe_alloc(s13);
    float*x=probe_alloc((size_t)K3_LATENT*4),*ref=probe_alloc((size_t)K3_LATENT*4),*sum=probe_calloc(K3_LATENT,4),*part=probe_alloc((size_t)K3_LATENT*4);
    float*g=probe_alloc((size_t)K3_EXPERT_INTER*4),*u=probe_alloc((size_t)K3_EXPERT_INTER*4),*gl=probe_alloc((size_t)local*4),*ul=probe_alloc((size_t)local*4);
    if(!p1||!s1||!p2||!s2||!p3||!s3||!x||!ref||!sum||!part||!g||!u||!gl||!ul)return 1;
    for(int q=0;q<nslice;++q){
        size_t pw=(size_t)local*K3_LATENT/2,ps=(size_t)local*K3_LATENT/32;
        memcpy(p1+(size_t)q*pw,le[q].w1.packed,pw);memcpy(s1+(size_t)q*ps,le[q].w1.scale,ps);
        memcpy(p3+(size_t)q*pw,le[q].w3.packed,pw);memcpy(s3+(size_t)q*ps,le[q].w3.scale,ps);
        for(int r=0;r<K3_LATENT;++r){
            memcpy(p2+(size_t)r*K3_EXPERT_INTER/2+(size_t)q*local/2,
                   le[q].w2.packed+(size_t)r*local/2,(size_t)local/2);
            memcpy(s2+(size_t)r*K3_EXPERT_INTER/32+(size_t)q*local/32,
                   le[q].w2.scale+(size_t)r*local/32,(size_t)local/32);
        }
    }
    for(int i=0;i<K3_LATENT;++i)x[i]=rf()*.125f;
    k3_mxfp4_matrix f1={p1,s1,K3_EXPERT_INTER,K3_LATENT},f2={p2,s2,K3_LATENT,K3_EXPERT_INTER},f3={p3,s3,K3_EXPERT_INTER,K3_LATENT};
    k3_expert_forward_mxfp4(ref,&f1,&f2,&f3,x,1,g,u,threads);
    double max_ms=0,total_ms=0;
    for(int q=0;q<nslice;++q){double t=now_sec();int bad=k3_expert_tp_forward_mxfp4(part,&le[q].w1,&le[q].w2,&le[q].w3,x,1,gl,ul,threads,8);double ms=(now_sec()-t)*1e3;
        if(bad)return 1;if(ms>max_ms)max_ms=ms;total_ms+=ms;for(int i=0;i<K3_LATENT;++i)sum[i]+=part[i];}
    float e=max_abs(sum,ref,K3_LATENT);double se=0,sr=0;for(int i=0;i<K3_LATENT;++i){double d=sum[i]-ref[i];se+=d*d;sr+=(double)ref[i]*ref[i];}
    double rel=sqrt(se/(sr+1e-30));printf("[expert-tp] slices=%d channels/rank=%d max_abs=%.3e rel_l2=%.3e %s\n",nslice,local,e,rel,e<2e-4&&rel<5e-4?"OK":"FAIL");
    printf("PROBE expert-tp emulated_ranks=%d serial_ms=%.3f critical_rank_ms=%.3f projected_layers_ms=%.3f\n",nslice,total_ms,max_ms,max_ms*K3_MOE_LAYERS);
    probe_free(p1);probe_free(s1);probe_free(p2);probe_free(s2);probe_free(p3);probe_free(s3);probe_free(x);probe_free(ref);probe_free(sum);probe_free(part);probe_free(g);probe_free(u);probe_free(gl);probe_free(ul);return e>=2e-4||rel>=5e-4;
}

static int parse_int_arg(const char*flag,const char*text,int lo,int hi,int*out){
    char*end=NULL;errno=0;long v=strtol(text,&end,10);if(errno||!end||*end||v<lo||v>hi){fprintf(stderr,"k3_moe_probe: %s expects integer in [%d,%d], got '%s'\n",flag,lo,hi,text);return-1;}*out=(int)v;return 0;
}

int main(int argc,char**argv){
    const char*paths[32];int npath=0,threads=48,tile_threshold=8,selected=0;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--threads")){if(++i>=argc||parse_int_arg("--threads",argv[i],1,48,&threads))return 2;}
        else if(!strcmp(argv[i],"--tile-threshold")){if(++i>=argc||parse_int_arg("--tile-threshold",argv[i],0,256,&tile_threshold))return 2;}
        else if(!strcmp(argv[i],"--tp-selected"))selected=1;
        else if(!strcmp(argv[i],"--help")){printf("usage: %s [--threads N] [--tile-threshold N] [--tp-selected] BLOB MANIFEST [BLOB MANIFEST ...]\n",argv[0]);return 0;}
        else if(argv[i][0]=='-'){fprintf(stderr,"k3_moe_probe: unknown option '%s'\n",argv[i]);return 2;}
        else if(npath<32)paths[npath++]=argv[i];else{fprintf(stderr,"k3_moe_probe: at most 16 expert pairs are supported\n");return 2;}
    }
    if(npath<2||(npath&1)){fprintf(stderr,"usage: %s [--threads N] [--tile-threshold N] [--tp-selected] BLOB MANIFEST [BLOB MANIFEST ...]\n",argv[0]);return 2;}
    int nlocal=npath/2;loaded_expert le[16];memset(le,0,sizeof(le));k3_pool_init(&probe_pool,"moe-probe");
    for(int e=0;e<nlocal;++e)if(load_expert(paths[2*e],paths[2*e+1],&le[e])){fprintf(stderr,"k3_moe_probe: load expert pair %d failed (%s, %s)\n",e,paths[2*e],paths[2*e+1]);for(int q=0;q<e;++q)probe_free(le[q].blob);k3_pool_destroy(&probe_pool);return 2;}
    k3_mxfp4_matrix w1=le[0].w1,w2=le[0].w2,w3=le[0].w3;
    if(w1.rows!=K3_EXPERT_INTER){int fail=tp_slices_correctness(le,nlocal,threads,selected);for(int e=0;e<nlocal;++e)probe_free(le[e].blob);printf("K3 expert-TP probe: %s\n",fail?"FAIL":"PASS");int status=probe_alloc_failed?2:fail?1:0;k3_pool_destroy(&probe_pool);return status;}
    int fail=dispatch_unit()|batched_correctness(&w1,&w2,&w3)|tiled_correctness(&w1,&w2,&w3)|situ_fast_correctness(&w1,&w2,&w3)|local_scheduler_correctness(&w1,&w2,&w3);
    int batches[]={1,2,4,8,16,32},thread_sweep[]={24,threads};
    if(!probe_alloc_failed)for(int ti=0;ti<2;++ti)for(int bi=0;bi<6;++bi)perf(&w1,&w2,&w3,batches[bi],thread_sweep[ti],tile_threshold);
    if(!probe_alloc_failed)profile_stages(&w1,&w2,&w3,32,threads,tile_threshold);
    if(!probe_alloc_failed)for(int e=1;e<=nlocal;e*=2)multi_expert_perf(le,e,threads);
    if(!probe_alloc_failed&&nlocal>2&&(nlocal&(nlocal-1)))multi_expert_perf(le,nlocal,threads);
    for(int e=0;e<nlocal;++e)probe_free(le[e].blob);printf("K3 MoE probe: %s\n",fail?"FAIL":"PASS");int status=probe_alloc_failed?2:fail?1:0;fprintf(stderr,"k3 MoE pool: peak=%.2f MiB reserved=%.2f MiB\n",probe_pool.peak_active_bytes/1048576.0,probe_pool.reserved_bytes/1048576.0);k3_pool_destroy(&probe_pool);return status;
}
