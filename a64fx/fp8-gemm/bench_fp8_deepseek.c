#define _POSIX_C_SOURCE 200809L
#include "fp8_gemm.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef struct {
    off_t offset;
    size_t bytes;
    int n, k;
    char dtype[16], name[256];
} tensor_record;

static const char *tensor_names[]={
    "layers.0.attn.wkv", "layers.0.attn.wo_a", "layers.0.attn.wo_b",
    "layers.0.attn.wq_a", "layers.0.attn.wq_b"
};
static unsigned rng_state=17;
static unsigned rng_u32(void){rng_state=rng_state*1664525u+1013904223u;return rng_state;}
static float uniform_signed(void){return((rng_u32()>>8)*(1.0f/8388608.0f))-1.0f;}

static int find_record(const char*manifest,const char*name,tensor_record*r){
    FILE*f=fopen(manifest,"r");char line[640];if(!f)return-1;
    while(fgets(line,sizeof(line),f)){
        long long off,bytes;int ndim,n,k;char dtype[16],found[256];
        if(sscanf(line,"%lld %lld %15s %d %d %d %255s",&off,&bytes,dtype,
                  &ndim,&n,&k,found)==7&&ndim==2&&!strcmp(found,name)){
            r->offset=(off_t)off;r->bytes=(size_t)bytes;r->n=n;r->k=k;
            memcpy(r->dtype,dtype,sizeof(r->dtype));
            snprintf(r->name,sizeof(r->name),"%s",found);fclose(f);return 0;
        }
    }fclose(f);return-1;
}

static int read_full(int fd,void*dst,size_t bytes,off_t offset){
    unsigned char*p=dst;size_t done=0;
    while(done<bytes){ssize_t n=pread(fd,p+done,bytes-done,offset+(off_t)done);
        if(n<=0)return-1;done+=(size_t)n;}
    return 0;
}

static int load_tensor(int fd,const tensor_record*wr,const tensor_record*sr,
                       fp8_matrix*w){
    if(strcmp(wr->dtype,"F8_E4M3")||strcmp(sr->dtype,"F8_E8M0")||
       wr->bytes!=(size_t)wr->n*wr->k||sr->n!=wr->n/128||sr->k!=wr->k/128||
       fp8_matrix_alloc(w,wr->n,wr->k))return-1;
    uint8_t*rowmajor=malloc(wr->bytes),*scale=malloc(sr->bytes);
    if(!rowmajor||!scale||read_full(fd,rowmajor,wr->bytes,wr->offset)||
       read_full(fd,scale,sr->bytes,sr->offset)){free(rowmajor);free(scale);return-1;}
    for(int r=0;r<wr->n;++r)for(int k=0;k<wr->k;++k)
        w->codes[((size_t)(r/128)*wr->k+k)*128+r%128]=rowmajor[(size_t)r*wr->k+k];
    for(size_t i=0;i<sr->bytes;++i)w->scales[i]=ldexpf(1.0f,(int)scale[i]-127);
    free(rowmajor);free(scale);
#if defined(POSIX_FADV_DONTNEED)
    posix_fadvise(fd,wr->offset,(off_t)wr->bytes,POSIX_FADV_DONTNEED);
    posix_fadvise(fd,sr->offset,(off_t)sr->bytes,POSIX_FADV_DONTNEED);
#endif
    return 0;
}

static void fill_activation(float*a,int k,int distribution,int seed){
    rng_state=(unsigned)(17+seed*977+distribution*65537);
    for(int i=0;i<k;++i){
        if(distribution==0)a[i]=uniform_signed();
        else if(distribution==1)a[i]=(rng_u32()&1)?1.0f:-1.0f;
        else {float u1=fmaxf((uniform_signed()+1.0f)*0.5f,1e-7f);
            float u2=(uniform_signed()+1.0f)*0.5f;
            float z=sqrtf(-2.0f*logf(u1))*cosf(6.28318530718f*u2);
            if(distribution==3&&(rng_u32()&255)==0)z*=16.0f;a[i]=z;}
    }
}

static void errors(const float*ref,const float*got,int n,double*rel,double*cosine,
                   double*max_abs){
    double ne=0.0,rr=0.0,gg=0.0,rg=0.0,ma=0.0;
    for(int i=0;i<n;++i){double d=(double)got[i]-ref[i];ne+=d*d;rr+=(double)ref[i]*ref[i];
        gg+=(double)got[i]*got[i];rg+=(double)ref[i]*got[i];if(fabs(d)>ma)ma=fabs(d);}
    *rel=sqrt(ne/(rr+1e-300));*cosine=rg/sqrt(rr*gg+1e-300);*max_abs=ma;
}

static double weight_rel_l2(const fp8_matrix*w,const fp8_i8_matrix*q){
    double ne=0.0,de=0.0;int kb=w->k/128,qkb=w->k/q->scale_group,tiles=128/q->lane_group;
    for(int g=0;g<w->n/128;++g)for(int k=0;k<w->k;++k)for(int lane=0;lane<128;++lane){
        size_t off=((size_t)g*w->k+k)*128+lane;
        double exact=fp8_e4m3fn_decode(w->codes[off])*w->scales[(size_t)g*kb+k/128];
        double approx=q->codes[off]*q->scales[((size_t)g*qkb+k/q->scale_group)*tiles+
                                              lane/q->lane_group],d=approx-exact;
        ne+=d*d;de+=exact*exact;
    }return sqrt(ne/(de+1e-300));
}

static int write_encoded(const char*dir,const char*base,const char*policy,
                         const fp8_i8_matrix*w){
    if(!dir)return 0;char path[768];
    snprintf(path,sizeof(path),"%s/%s.%s.i8",dir,base,policy);FILE*f=fopen(path,"wb");
    if(!f)return-1;size_t codes=(size_t)w->n*w->k;
    size_t scales=(size_t)(w->n/128)*(w->k/w->scale_group)*(128/w->lane_group);
    int bad=fwrite(w->codes,1,codes,f)!=codes||fwrite(w->scales,sizeof(float),scales,f)!=scales||
        fflush(f)||fsync(fileno(f));fclose(f);return bad?-1:0;
}

int main(int argc,char**argv){
    const char*raw=argc>1?argv[1]:"/local/u14346/ds4f-fp4-gemm/layer0.raw";
    const char*manifest=argc>2?argv[2]:"/local/u14346/ds4f-fp4-gemm/layer0.manifest";
    const char*outdir=argc>3?argv[3]:NULL;const char*policy_name=getenv("FP8_I8_POLICY");
    fp8_i8_policy policy=policy_name&&!strcmp(policy_name,"mse")?FP8_I8_MSE:FP8_I8_ABSMAX;
    policy_name=policy==FP8_I8_MSE?"mse":"absmax";
    int group=getenv("FP8_I8_GROUP")?atoi(getenv("FP8_I8_GROUP")):2;
    int lane_group=getenv("FP8_I8_LANES")?atoi(getenv("FP8_I8_LANES")):32;
    int act_group=getenv("FP8_I8_ACT_GROUP")?atoi(getenv("FP8_I8_ACT_GROUP")):16;
    int fd=open(raw,O_RDONLY);if(fd<0){perror(raw);return 1;}
    double worst=0.0,min_cos=1.0,sdot_exact_worst=0.0,sdot_fma_worst=0.0;
    for(size_t ti=0;ti<sizeof(tensor_names)/sizeof(tensor_names[0]);++ti){
        char wn[320],sn[320];snprintf(wn,sizeof(wn),"%s.weight",tensor_names[ti]);
        snprintf(sn,sizeof(sn),"%s.scale",tensor_names[ti]);tensor_record wr,sr;
        if(find_record(manifest,wn,&wr)||find_record(manifest,sn,&sr)){fprintf(stderr,"missing %s\n",tensor_names[ti]);return 1;}
        fp8_matrix w;fp8_i8_matrix q={0},qdot={0};fp8_i8_activation aq={0};
        if(load_tensor(fd,&wr,&sr,&w)||fp8_matrix_prepare_i8_tile(&w,&q,policy,group,lane_group))return 1;
        if(fp8_matrix_prepare_i8(&w,&qdot,FP8_I8_ABSMAX)||fp8_i8_matrix_prepare_sdot(&qdot))return 1;
        double wrq=weight_rel_l2(&w,&q);
        if(write_encoded(outdir,tensor_names[ti],policy_name,&q)){perror("write encoded");return 1;}
        if(fp8_matrix_prepare_fast(&w))return 1;
        float*a=aligned_alloc(256,(size_t)w.k*sizeof(float));
        float*ref=aligned_alloc(256,(size_t)w.n*sizeof(float));
        float*got=aligned_alloc(256,(size_t)w.n*sizeof(float));
        float*fma=aligned_alloc(256,(size_t)w.n*sizeof(float));if(!a||!ref||!got||!fma)return 1;
        double tensor_worst=0.0,tensor_cos=1.0,max_abs=0.0;
        double tensor_sdot_exact=0.0,tensor_sdot_fma=0.0,tensor_sdot_cos=1.0;
        for(int dist=0;dist<4;++dist)for(int seed=0;seed<3;++seed){
            fill_activation(a,w.k,dist,seed);double rel,cosine,ma,re,rf,cs,unused;
            fp8_gemv_f32_omp(ref,a,&w,12,FP8_DECODE_SPARSE_EXACT);
            fp8_i8_gemv_f32_omp(got,a,&q,12);errors(ref,got,w.n,&rel,&cosine,&ma);
            if(rel>tensor_worst)tensor_worst=rel;if(cosine<tensor_cos)tensor_cos=cosine;
            if(ma>max_abs)max_abs=ma;
            fp8_i8_gemv_f32_omp(fma,a,&qdot,12);
            if(fp8_i8_activation_prepare_group(&aq,a,w.k,act_group)||
               fp8_i8_gemv_sdot_omp(got,&aq,&qdot,12))return 1;
            errors(ref,got,w.n,&re,&cs,&unused);errors(fma,got,w.n,&rf,&cosine,&unused);
            if(re>tensor_sdot_exact)tensor_sdot_exact=re;
            if(rf>tensor_sdot_fma)tensor_sdot_fma=rf;if(cs<tensor_sdot_cos)tensor_sdot_cos=cs;
        }
        printf("tensor=%s N=%d K=%d policy=%s G=%dx%d weight_rel_l2=%.6g "
               "gemv_worst_rel_l2=%.6g min_cos=%.9f max_abs=%.6g "
               "sdot_A%d_vs_exact=%.6g sdot_vs_fma=%.6g sdot_cos=%.9f\n",tensor_names[ti],
               w.n,w.k,policy_name,lane_group,group,wrq,tensor_worst,tensor_cos,max_abs,
               act_group,tensor_sdot_exact,tensor_sdot_fma,tensor_sdot_cos);fflush(stdout);
        if(tensor_worst>worst)worst=tensor_worst;if(tensor_cos<min_cos)min_cos=tensor_cos;
        if(tensor_sdot_exact>sdot_exact_worst)sdot_exact_worst=tensor_sdot_exact;
        if(tensor_sdot_fma>sdot_fma_worst)sdot_fma_worst=tensor_sdot_fma;
        free(a);free(ref);free(got);free(fma);fp8_i8_activation_free(&aq);
        fp8_i8_matrix_free(&q);fp8_i8_matrix_free(&qdot);fp8_matrix_free(&w);
    }
    close(fd);printf("summary policy=%s G=%dx%d worst_rel=%.6g min_cos=%.9f "
        "gate_rel=0.005 gate_cos=0.9999 result=%s sdot128_vs_exact=%.6g "
        "sdot_A%d_activation_delta=%.6g\n",policy_name,lane_group,group,worst,min_cos,
        worst<=.005&&min_cos>=.9999?"PASS":"FAIL",sdot_exact_worst,act_group,sdot_fma_worst);
    return 0;
}
