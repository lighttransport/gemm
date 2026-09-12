#include "ds41f_team.h"
#include "ds41f_ops.h"
#include "ds41f_profile.h"
#include <math.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
static float sigmoid(float x) {return x>=0?1/(1+expf(-x)):expf(x)/(1+expf(x));}
int ds41f_argmax_finite(const float *x,size_t n,size_t *index)
{
    if(!x||!n||!index)return EINVAL;
    #if defined(__ARM_FEATURE_SVE)
    svfloat32_t maximum=svdup_f32(-INFINITY);
    for(size_t i=0;i<n;i+=svcntw()){
        svbool_t pg=svwhilelt_b32(i,n);svfloat32_t v=svld1_f32(pg,x+i);
        if(svptest_any(pg,svcmpge_n_f32(pg,svabs_f32_x(pg,v),INFINITY))||
           svptest_any(pg,svcmpuo_f32(pg,v,v)))return EDOM;
        maximum=svmax_f32_m(pg,maximum,v);
    }
    float best=svmaxv_f32(svptrue_b32(),maximum);
    for(size_t i=0;i<n;i+=svcntw()){
        svbool_t pg=svwhilelt_b32(i,n);
        if(svptest_any(pg,svcmpeq_n_f32(pg,svld1_f32(pg,x+i),best))){
            for(size_t j=i;j<n&&j<i+svcntw();++j)if(x[j]==best){*index=j;return 0;}
        }
    }
    return EDOM;
    #else
    size_t best=0;
    for(size_t i=0;i<n;++i){if(!isfinite(x[i]))return EDOM;if(x[i]>x[best])best=i;}
    *index=best;return 0;
    #endif
}
float ds41f_hc_inverse_rms(const float *x,size_t n)
{
    if(!n)return 0;
    double ss=0;size_t i=0;
    #if defined(__ARM_FEATURE_SVE)
    if(svcntw()==16){
        svbool_t pg=svptrue_b64(),p8=svptrue_pat_b32(SV_VL8);
        svfloat64_t a=svdup_f64(0),b=a,c=a,d=a;
        #define HC_LOAD8(offset) svcvt_f64_f32_x(pg,svreinterpret_f32_u64( \
            svunpklo_u64(svreinterpret_u32_f32(svld1_f32(p8,x+i+(offset))))))
        for(;i+32<=n;i+=32){
            svfloat64_t x0=HC_LOAD8(0),x1=HC_LOAD8(8),x2=HC_LOAD8(16),x3=HC_LOAD8(24);
            a=svmla_f64_x(pg,a,x0,x0);b=svmla_f64_x(pg,b,x1,x1);
            c=svmla_f64_x(pg,c,x2,x2);d=svmla_f64_x(pg,d,x3,x3);
        }
        for(;i+8<=n;i+=8){svfloat64_t v=HC_LOAD8(0);a=svmla_f64_x(pg,a,v,v);}
        #undef HC_LOAD8
        ss=svaddv_f64(pg,svadd_f64_x(pg,svadd_f64_x(pg,a,b),svadd_f64_x(pg,c,d)));
    }
    #endif
    for(;i<n;++i)ss+=(double)x[i]*x[i];
    return (float)(1/sqrt(ss/n+1e-20));
}

typedef struct {
    float * out;
    const float * gate;
    const float * up;
    float limit;
} swiglu_team_job;
static void swiglu_team_work(void *context,size_t first,size_t last)
{
    swiglu_team_job *job=context;
    float * out=job->out;
    const float * gate=job->gate;
    const float * up=job->up;
    float limit=job->limit;
    (void)out;
    (void)gate;
    (void)up;
    (void)limit;
    for(size_t task=first;task<last;++task){size_t i=task*(1);float g=gate[i],u=up[i];
        if(limit>0){g=fminf(g,limit);u=fminf(limit,fmaxf(-limit,u));}
        out[i]=g*sigmoid(g)*u;

    }
}
void ds41f_swiglu(float *out,const float *gate,const float *up,size_t n,float limit)
{
    if(ds41f_team_active()){
        swiglu_team_job job={out, gate, up, limit};
        (void)ds41f_team_for(n,swiglu_team_work,&job);
    }else
    #pragma omp parallel for schedule(static) if(n>=512)
    for(size_t i=0;i<n;++i){float g=gate[i],u=up[i];
        if(limit>0){g=fminf(g,limit);u=fminf(limit,fmaxf(-limit,u));}
        out[i]=g*sigmoid(g)*u;
    }
}
typedef struct {const float *logits,*bias;float *scores;float temperature;int *errors;} gate_team_job;
static void gate_team_work(void *context,size_t first,size_t last)
{
    gate_team_job *job=context;int invalid=0;
    for(size_t i=first;i<last;++i){float x=job->logits[i]/job->temperature;
        if(!isfinite(x)||!isfinite(job->bias[i])){invalid=1;job->scores[i]=0;continue;}
        job->scores[i]=sqrtf(fmaxf(x,0)+log1pf(expf(-fabsf(x))));}
    if(invalid)job->errors[ds41f_team_thread_id()]=invalid;
}
int ds41f_gate(const float *logits,const float *bias,int experts,int k,
               float temperature,float route_scale,int *ids,float *weights)
{
    if(!logits||!bias||!ids||!weights||experts<1||k<1||k>experts||
       !isfinite(temperature)||temperature<=0||!isfinite(route_scale))return EINVAL;
    float *scores=malloc((size_t)experts*sizeof(float));
    if(!scores)return ENOMEM;
    int invalid=0;
    if(ds41f_team_active()){
        int errors[48]={0};gate_team_job job={logits,bias,scores,temperature,errors};
        (void)ds41f_team_for((size_t)experts,gate_team_work,&job);
        for(int i=0;i<48;++i)invalid|=errors[i];
    }else
    #pragma omp parallel for schedule(static) if(experts>=384) reduction(|:invalid)
    for(int i=0;i<experts;++i){float x=logits[i]/temperature;
        if(!isfinite(x)||!isfinite(bias[i])){invalid=1;scores[i]=0;continue;}
        scores[i]=sqrtf(fmaxf(x,0)+log1pf(expf(-fabsf(x))));
    }
    if(invalid){free(scores);return EDOM;}
    /* Scan IDs in ascending order. Strict insertion keeps the original
     * smallest-ID tie rule without rescanning already selected experts. */
    int count=0;
    for(int i=0;i<experts;++i){float value=scores[i]+bias[i];
        if(count==k&&value<=scores[ids[k-1]]+bias[ids[k-1]])continue;
        int j=count<k?count:k-1;
        while(j>0&&value>scores[ids[j-1]]+bias[ids[j-1]]){ids[j]=ids[j-1];--j;}
        ids[j]=i;if(count<k)++count;
    }
    float sum=0;
    for(int j=0;j<k;++j){weights[j]=scores[ids[j]];sum+=weights[j];}
    for(int j=0;j<k;++j)weights[j]*=route_scale/(k>1?sum+1e-20f:1);
    free(scores);return 0;
}
void ds41f_hc_split(const float mix[24],const float scale[3],const float base[24],
                    int iters,float eps,float pre[4],float post[4],float comb[16])
{
    for(int i=0;i<4;++i){pre[i]=sigmoid(mix[i]*scale[0]+base[i])+eps;
        post[i]=2*sigmoid(mix[i+4]*scale[1]+base[i+4]);}
    for(int i=0;i<4;++i){float mx=-INFINITY,sum=0;
        for(int j=0;j<4;++j){int k=i*4+j;comb[k]=mix[k+8]*scale[2]+base[k+8];mx=fmaxf(mx,comb[k]);}
        for(int j=0;j<4;++j){int k=i*4+j;comb[k]=expf(comb[k]-mx);sum+=comb[k];}
        for(int j=0;j<4;++j)comb[i*4+j]=comb[i*4+j]/sum+eps;
    }
    for(int it=0;it<iters;++it){
        if(it)for(int i=0;i<4;++i){float sum=eps;for(int j=0;j<4;++j)sum+=comb[i*4+j];for(int j=0;j<4;++j)comb[i*4+j]/=sum;}
        for(int j=0;j<4;++j){float sum=eps;for(int i=0;i<4;++i)sum+=comb[i*4+j];for(int i=0;i<4;++i)comb[i*4+j]/=sum;}
    }
}
void ds41f_hc_pre(float *out,const float *x,const float pre[4],size_t dim)
{for(size_t j=0;j<dim;++j){float v=0;for(int i=0;i<4;++i)v+=pre[i]*x[i*dim+j];out[j]=v;}}
typedef struct {
    float * out;
    const float * x;
    const float * residual;
    const float * post;
    const float * comb;
    size_t dim;
} hc_post_team_job;
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
static void hc_post_team_work(void *context,size_t first,size_t last)
{
    #if defined(__clang__)
    #pragma STDC FP_CONTRACT OFF
    #endif
    hc_post_team_job *job=context;
    float * out=job->out;
    const float * x=job->x;
    const float * residual=job->residual;
    const float * post=job->post;
    const float * comb=job->comb;
    size_t dim=job->dim;
    (void)out;
    (void)x;
    (void)residual;
    (void)post;
    (void)comb;
    (void)dim;
    for(size_t task=first;task<last;++task){size_t j=task*(1);float v[4];
        for(int o=0;o<4;++o){float sum=0;
            for(int i=0;i<4;++i)sum+=comb[i*4+o]*residual[i*dim+j];
            v[o]=post[o]*x[j]+sum;}
        for(int o=0;o<4;++o)out[o*dim+j]=v[o];

    }
}
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void ds41f_hc_post(float *out,const float *x,const float *residual,
                   const float post[4],const float comb[16],size_t dim)
{
    /* model.py forms the residual sum separately, then adds post*x. Keeping
     * post*x in the reduction loses residual terms when those paths cancel.
     * Products in the upstream eager FP32 expression are rounded before sum. */
    #if defined(__clang__)
    #pragma STDC FP_CONTRACT OFF
    #endif
    if(ds41f_team_active()){
        hc_post_team_job job={out, x, residual, post, comb, dim};
        (void)ds41f_team_for(dim,hc_post_team_work,&job);
    }else
    #pragma omp parallel for schedule(static) if(dim>=512)
    for(size_t j=0;j<dim;++j){float v[4];
        for(int o=0;o<4;++o){float sum=0;
            for(int i=0;i<4;++i)sum+=comb[i*4+o]*residual[i*dim+j];
            v[o]=post[o]*x[j]+sum;}
        for(int o=0;o<4;++o)out[o*dim+j]=v[o];
    }
}
void ds41f_engram_fuse(float *x,const float *key,const float *value,
                       const float *qw,const float *kw,size_t dim,float eps)
{for(int h=0;h<4;++h){double xx=0,kk=0,dot=0;
    for(size_t j=0;j<dim;++j){size_t i=h*dim+j;xx+=x[i]*x[i];kk+=key[i]*key[i];dot+=(double)x[i]*key[i]*qw[i]*kw[i];}
    double d=dot/sqrt((xx/dim+eps)*(kk/dim+eps)*dim);
    float gate=sigmoid((float)copysign(sqrt(fmax(fabs(d),1e-6)),d));
    for(size_t j=0;j<dim;++j)x[h*dim+j]+=gate*value[j];}}
static int rope_cache_enabled;
void ds41f_set_rope_cache(int enabled){rope_cache_enabled=enabled!=0;}
typedef struct {double theta,factor,cosine[32],sine[32];size_t pos,stamp;int original,inverse;} rope_entry;
static _Thread_local rope_entry rope_entries[8];
static _Thread_local size_t rope_clock;
void ds41f_rope(float *x,size_t heads,size_t dim,size_t rd,size_t pos,
                 double theta,double factor,int original,int inverse)
{
    const double pi=3.14159265358979323846;
    double low=0,high=0;
    if(original){low=fmax(floor(rd*log(original/(32*2*pi))/(2*log(theta))),0);
        high=fmin(ceil(rd*log(original/(2*pi))/(2*log(theta))),rd-1);}
    if(rd==64){double local_cosine[32],local_sine[32];
        double *cosine=local_cosine,*sine=local_sine;rope_entry *entry=NULL;int hit=0;
        if(rope_cache_enabled){
            for(int i=0;i<8;++i){rope_entry *candidate=rope_entries+i;
                if(candidate->stamp&&candidate->theta==theta&&candidate->factor==factor&&
                   candidate->pos==pos&&candidate->original==original&&candidate->inverse==inverse){entry=candidate;hit=1;break;}
                if(!entry||candidate->stamp<entry->stamp)entry=candidate;}
            cosine=entry->cosine;sine=entry->sine;
        }
        if(!hit)for(size_t j=0;j<32;++j){double freq=pow(theta,-2.0*j/rd);
            if(original){double ramp=fmax(0,fmin(1,(j-low)/fmax(high-low,1e-3)));freq*=1-ramp+ramp/factor;}
            double angle=pos*freq*(inverse?-1:1);cosine[j]=cos(angle);sine[j]=sin(angle);}
        if(entry){entry->theta=theta;entry->factor=factor;entry->pos=pos;entry->original=original;entry->inverse=inverse;entry->stamp=++rope_clock;}
        /* Complete each head's contiguous cache line before moving to the
         * next head; every element keeps the original double arithmetic. */
        for(size_t h=0;h<heads;++h)for(size_t j=0;j<32;++j){
            size_t i=h*dim+dim-rd+2*j;float a=x[i],b=x[i+1];double c=cosine[j],s=sine[j];
            x[i]=(float)(a*c-b*s);x[i+1]=(float)(a*s+b*c);}
        return;
    }
    for(size_t j=0;j<rd/2;++j){double freq=pow(theta,-2.0*j/rd);
        if(original){double ramp=fmax(0,fmin(1,(j-low)/fmax(high-low,1e-3)));freq*=1-ramp+ramp/factor;}
        double angle=pos*freq*(inverse?-1:1),c=cos(angle),s=sin(angle);
        for(size_t h=0;h<heads;++h){size_t i=h*dim+dim-rd+2*j;float a=x[i],b=x[i+1];x[i]=(float)(a*c-b*s);x[i+1]=(float)(a*s+b*c);}
    }
}
int ds41f_sparse_attention_ref(float *out,const float *q,const float *kv,
                            const float *sink,const int *ids,size_t selected,
                            size_t tokens,size_t heads,size_t dim)
{
    if(!dim||!heads||!out||!q||!kv||!sink||(!ids&&selected))return EINVAL;
    for(size_t i=0;i<selected;++i)if(ids[i]>=0&&(size_t)ids[i]>=tokens)return ERANGE;
    #pragma omp parallel for
    for(size_t h=0;h<heads;++h){double mx=sink[h],sum=1;float *o=out+h*dim;
        for(size_t j=0;j<dim;++j)o[j]=0;
        for(size_t i=0;i<selected;++i){if(ids[i]<0)continue;const float *v=kv+(size_t)ids[i]*dim;double score=0;
            for(size_t j=0;j<dim;++j)score+=(double)q[h*dim+j]*v[j];
            score/=sqrt((double)dim);
            double next=fmax(mx,score),a=exp(mx-next),b=exp(score-next);
            for(size_t j=0;j<dim;++j)o[j]=(float)(o[j]*a+v[j]*b);
            sum=sum*a+b;mx=next;
        }
        for(size_t j=0;j<dim;++j)o[j]/=(float)sum;
    }
    return 0;
}
int ds41f_sparse_attention(float *out,const float *q,const float *kv,
                            const float *sink,const int *ids,size_t selected,
                            size_t tokens,size_t heads,size_t dim)
{
#if defined(__ARM_FEATURE_SVE)
    if(!dim||!heads||!out||!q||!kv||!sink||(!ids&&selected))return EINVAL;
    for(size_t i=0;i<selected;++i)if(ids[i]>=0&&(size_t)ids[i]>=tokens)return ERANGE;
    if(!selected){for(size_t i=0;i<heads*dim;++i)out[i]=0;return 0;}
    float *scores=malloc(heads*selected*sizeof(float));if(!scores)return ENOMEM;
    float scale=1/sqrtf((float)dim);
    int timed=P_BEGIN()!=0;double qk_max=0,softmax_max=0,pv_max=0;
    #pragma omp parallel reduction(max:qk_max,softmax_max,pv_max)
    {
    double qk_time=0,softmax_time=0,pv_time=0;
    #pragma omp for schedule(static)
    for(size_t h=0;h<heads;++h){float *s=scores+h*selected;float mx=sink[h];
        double timer=ds41f_profile_worker_clock(timed);
        svbool_t pg=svptrue_b32();size_t vl=svcntw();
        for(size_t i=0;i<selected;++i){
            if(ids[i]<0){s[i]=-INFINITY;continue;}const float *v=kv+(size_t)ids[i]*dim;
            svfloat32_t a=svdup_f32(0),b=svdup_f32(0);size_t j=0;
            for(;j+2*vl<=dim;j+=2*vl){a=svmla_x(pg,a,svld1(pg,q+h*dim+j),svld1(pg,v+j));
                b=svmla_x(pg,b,svld1(pg,q+h*dim+j+vl),svld1(pg,v+j+vl));}
            for(;j<dim;j+=vl){svbool_t tail=svwhilelt_b32(j,dim);a=svmla_m(tail,a,svld1(tail,q+h*dim+j),svld1(tail,v+j));}
            s[i]=svaddv(pg,svadd_x(pg,a,b))*scale;mx=fmaxf(mx,s[i]);
        }
        qk_time+=ds41f_profile_worker_clock(timed)-timer;timer=ds41f_profile_worker_clock(timed);
        float sum=expf(sink[h]-mx);
        for(size_t i=0;i<selected;++i){s[i]=expf(s[i]-mx);sum+=s[i];}
        for(size_t i=0;i<selected;++i)s[i]/=sum;
        softmax_time+=ds41f_profile_worker_clock(timed)-timer;timer=ds41f_profile_worker_clock(timed);
        /* Consume a full A64FX cache line from each selected row. Four
         * independent output vectors share the score/row lookup and hide
         * FMA latency without changing any output lane's reduction order. */
        size_t j=0;
        for(;j+4*vl<=dim;j+=4*vl){svfloat32_t a=svdup_f32(0),b=a,c=a,d=a;
            for(size_t i=0;i<selected;++i)if(ids[i]>=0){const float *v=kv+(size_t)ids[i]*dim+j;float weight=s[i];
                a=svmla_n_f32_x(pg,a,svld1(pg,v),weight);
                b=svmla_n_f32_x(pg,b,svld1(pg,v+vl),weight);
                c=svmla_n_f32_x(pg,c,svld1(pg,v+2*vl),weight);
                d=svmla_n_f32_x(pg,d,svld1(pg,v+3*vl),weight);}
            svst1(pg,out+h*dim+j,a);svst1(pg,out+h*dim+j+vl,b);
            svst1(pg,out+h*dim+j+2*vl,c);svst1(pg,out+h*dim+j+3*vl,d);}
        for(;j<dim;j+=vl){svbool_t tail=svwhilelt_b32(j,dim);svfloat32_t acc=svdup_f32(0);
            for(size_t i=0;i<selected;++i)if(ids[i]>=0)
                acc=svmla_n_f32_m(tail,acc,svld1(tail,kv+(size_t)ids[i]*dim+j),s[i]);
            svst1(tail,out+h*dim+j,acc);}
        pv_time+=ds41f_profile_worker_clock(timed)-timer;
    }
    qk_max=qk_time;softmax_max=softmax_time;pv_max=pv_time;
    }
    P_VALUE(SPARSE_QK,qk_max);P_VALUE(SPARSE_SOFTMAX,softmax_max);P_VALUE(SPARSE_PV,pv_max);
    free(scores);return 0;
#else
    return ds41f_sparse_attention_ref(out,q,kv,sink,ids,selected,tokens,heads,dim);
#endif
}
void ds41f_pool_pair(float *out,const float *a,const float *b,
                      const float *sa,const float *sb,size_t dim)
{for(size_t j=0;j<dim;++j){float w=sigmoid(sa[j]-sb[j]);out[j]=a[j]*w+b[j]*(1-w);}}

#include "ds41f_exp2.h"
#include "ds41f_sparse_sve.h"

typedef struct {
    float * out;
    const float * w;
    const float * x;
    float inv;
    int mode;
} hc_matvec_team_job;
static void hc_matvec_team_work(void *context,size_t first,size_t last)
{
    hc_matvec_team_job *job=context;
    float * out=job->out;
    const float * w=job->w;
    const float * x=job->x;
    float inv=job->inv;
    int mode=job->mode;
    (void)out;
    (void)w;
    (void)x;
    (void)inv;
    (void)mode;
    for(size_t task=first;task<last;++task){size_t r=task*(1);const float *row=w+(size_t)r*20480;float sum=0;
        #if defined(__ARM_FEATURE_SVE)
        if(mode==1&&svcntw()==16){svbool_t pg=svptrue_b32();
            svfloat32_t a=svdup_f32(0),b=a,c=a,d=a;
            for(size_t i=0;i<20480;i+=4*svcntw()){
                a=svmla_f32_x(pg,a,svld1_f32(pg,row+i),svld1_f32(pg,x+i));
                b=svmla_f32_x(pg,b,svld1_f32(pg,row+i+svcntw()),svld1_f32(pg,x+i+svcntw()));
                c=svmla_f32_x(pg,c,svld1_f32(pg,row+i+2*svcntw()),svld1_f32(pg,x+i+2*svcntw()));
                d=svmla_f32_x(pg,d,svld1_f32(pg,row+i+3*svcntw()),svld1_f32(pg,x+i+3*svcntw()));}
            sum=svaddv_f32(pg,svadd_f32_x(pg,svadd_f32_x(pg,a,b),svadd_f32_x(pg,c,d)));
        }else if(mode==2&&svcntw()==16){
            svbool_t pg=svptrue_b64(),p8=svptrue_pat_b32(SV_VL8);
            svfloat64_t a=svdup_f64(0),b=a,c=a,d=a;
            /* Preserve FP32 products, then accumulate them in FP64. This is
             * an explicit alternative to the ordered FP32 checkpoint path. */
            #define HC_PRODUCT(offset) svcvt_f64_f32_x(pg,svreinterpret_f32_u64(svunpklo_u64( \
                svreinterpret_u32_f32(svmul_f32_x(p8,svld1_f32(p8,row+i+(offset)),svld1_f32(p8,x+i+(offset)))))))
            for(size_t i=0;i<20480;i+=32){
                a=svadd_f64_x(pg,a,HC_PRODUCT(0));b=svadd_f64_x(pg,b,HC_PRODUCT(8));
                c=svadd_f64_x(pg,c,HC_PRODUCT(16));d=svadd_f64_x(pg,d,HC_PRODUCT(24));}
            #undef HC_PRODUCT
            sum=(float)svaddv_f64(pg,svadd_f64_x(pg,svadd_f64_x(pg,a,b),svadd_f64_x(pg,c,d)));
        }else
        #endif
        if(mode==2){double total=0;for(size_t i=0;i<20480;++i)total+=(float)(row[i]*x[i]);sum=(float)total;}
        else{
            #pragma omp simd reduction(+:sum)
            for(size_t i=0;i<20480;++i)sum+=row[i]*x[i];}
        out[r]=sum*inv;

    }
}
int ds41f_hc_matvec(float out[24],const float *w,const float *x,float inv,int mode)
{
    if(!out||!w||!x||mode<0||mode>2)return EINVAL;
    if(ds41f_team_active()){
        hc_matvec_team_job job={out, w, x, inv, mode};
        (void)ds41f_team_for(24,hc_matvec_team_work,&job);
    }else
    #pragma omp parallel for schedule(static)
    for(int r=0;r<24;++r){const float *row=w+(size_t)r*20480;float sum=0;
        #if defined(__ARM_FEATURE_SVE)
        if(mode==1&&svcntw()==16){svbool_t pg=svptrue_b32();
            svfloat32_t a=svdup_f32(0),b=a,c=a,d=a;
            for(size_t i=0;i<20480;i+=4*svcntw()){
                a=svmla_f32_x(pg,a,svld1_f32(pg,row+i),svld1_f32(pg,x+i));
                b=svmla_f32_x(pg,b,svld1_f32(pg,row+i+svcntw()),svld1_f32(pg,x+i+svcntw()));
                c=svmla_f32_x(pg,c,svld1_f32(pg,row+i+2*svcntw()),svld1_f32(pg,x+i+2*svcntw()));
                d=svmla_f32_x(pg,d,svld1_f32(pg,row+i+3*svcntw()),svld1_f32(pg,x+i+3*svcntw()));}
            sum=svaddv_f32(pg,svadd_f32_x(pg,svadd_f32_x(pg,a,b),svadd_f32_x(pg,c,d)));
        }else if(mode==2&&svcntw()==16){
            svbool_t pg=svptrue_b64(),p8=svptrue_pat_b32(SV_VL8);
            svfloat64_t a=svdup_f64(0),b=a,c=a,d=a;
            /* Preserve FP32 products, then accumulate them in FP64. This is
             * an explicit alternative to the ordered FP32 checkpoint path. */
            #define HC_PRODUCT(offset) svcvt_f64_f32_x(pg,svreinterpret_f32_u64(svunpklo_u64( \
                svreinterpret_u32_f32(svmul_f32_x(p8,svld1_f32(p8,row+i+(offset)),svld1_f32(p8,x+i+(offset)))))))
            for(size_t i=0;i<20480;i+=32){
                a=svadd_f64_x(pg,a,HC_PRODUCT(0));b=svadd_f64_x(pg,b,HC_PRODUCT(8));
                c=svadd_f64_x(pg,c,HC_PRODUCT(16));d=svadd_f64_x(pg,d,HC_PRODUCT(24));}
            #undef HC_PRODUCT
            sum=(float)svaddv_f64(pg,svadd_f64_x(pg,svadd_f64_x(pg,a,b),svadd_f64_x(pg,c,d)));
        }else
        #endif
        if(mode==2){double total=0;for(size_t i=0;i<20480;++i)total+=(float)(row[i]*x[i]);sum=(float)total;}
        else{
            #pragma omp simd reduction(+:sum)
            for(size_t i=0;i<20480;++i)sum+=row[i]*x[i];}
        out[r]=sum*inv;
    }
    return 0;
}

#include "ds41f_sparse_sdot.h"
