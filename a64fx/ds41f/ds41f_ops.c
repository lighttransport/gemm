#include "ds41f_ops.h"
#include <math.h>
#include <errno.h>
#include <stdlib.h>
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

void ds41f_swiglu(float *out,const float *gate,const float *up,size_t n,float limit)
{
    #pragma omp parallel for schedule(static) if(n>=512)
    for(size_t i=0;i<n;++i){float g=gate[i],u=up[i];
        if(limit>0){g=fminf(g,limit);u=fminf(limit,fmaxf(-limit,u));}
        out[i]=g*sigmoid(g)*u;
    }
}
int ds41f_gate(const float *logits,const float *bias,int experts,int k,
               float temperature,float route_scale,int *ids,float *weights)
{
    if(!logits||!bias||!ids||!weights||experts<1||k<1||k>experts||
       !isfinite(temperature)||temperature<=0||!isfinite(route_scale))return EINVAL;
    float *scores=malloc((size_t)experts*sizeof(float));
    if(!scores)return ENOMEM;
    int invalid=0;
    #pragma omp parallel for schedule(static) if(experts>=384) reduction(|:invalid)
    for(int i=0;i<experts;++i){float x=logits[i]/temperature;
        if(!isfinite(x)||!isfinite(bias[i])){invalid=1;scores[i]=0;continue;}
        scores[i]=sqrtf(fmaxf(x,0)+log1pf(expf(-fabsf(x))));
    }
    if(invalid){free(scores);return EDOM;}
    float sum=0;
    for(int j=0;j<k;++j){int best=-1;
        for(int i=0;i<experts;++i){int used=0;for(int p=0;p<j;++p)if(ids[p]==i)used=1;
            if(!used&&(best<0||scores[i]+bias[i]>scores[best]+bias[best]))best=i;}
        ids[j]=best;weights[j]=scores[best];sum+=weights[j];
    }
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
void ds41f_rope(float *x,size_t heads,size_t dim,size_t rd,size_t pos,
                 double theta,double factor,int original,int inverse)
{
    const double pi=3.14159265358979323846;
    double low=0,high=0;
    if(original){low=fmax(floor(rd*log(original/(32*2*pi))/(2*log(theta))),0);
        high=fmin(ceil(rd*log(original/(2*pi))/(2*log(theta))),rd-1);}
    if(rd==64){double cosine[32],sine[32];
        for(size_t j=0;j<32;++j){double freq=pow(theta,-2.0*j/rd);
            if(original){double ramp=fmax(0,fmin(1,(j-low)/fmax(high-low,1e-3)));freq*=1-ramp+ramp/factor;}
            double angle=pos*freq*(inverse?-1:1);cosine[j]=cos(angle);sine[j]=sin(angle);}
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
    #pragma omp parallel for schedule(static)
    for(size_t h=0;h<heads;++h){float *s=scores+h*selected;float mx=sink[h];
        svbool_t pg=svptrue_b32();size_t vl=svcntw();
        for(size_t i=0;i<selected;++i){
            if(ids[i]<0){s[i]=-INFINITY;continue;}const float *v=kv+(size_t)ids[i]*dim;
            svfloat32_t a=svdup_f32(0),b=svdup_f32(0);size_t j=0;
            for(;j+2*vl<=dim;j+=2*vl){a=svmla_x(pg,a,svld1(pg,q+h*dim+j),svld1(pg,v+j));
                b=svmla_x(pg,b,svld1(pg,q+h*dim+j+vl),svld1(pg,v+j+vl));}
            for(;j<dim;j+=vl){svbool_t tail=svwhilelt_b32(j,dim);a=svmla_m(tail,a,svld1(tail,q+h*dim+j),svld1(tail,v+j));}
            s[i]=svaddv(pg,svadd_x(pg,a,b))*scale;mx=fmaxf(mx,s[i]);
        }
        float sum=expf(sink[h]-mx);
        for(size_t i=0;i<selected;++i){s[i]=expf(s[i]-mx);sum+=s[i];}
        for(size_t i=0;i<selected;++i)s[i]/=sum;
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
    }
    free(scores);return 0;
#else
    return ds41f_sparse_attention_ref(out,q,kv,sink,ids,selected,tokens,heads,dim);
#endif
}
void ds41f_pool_pair(float *out,const float *a,const float *b,
                      const float *sa,const float *sb,size_t dim)
{for(size_t j=0;j<dim;++j){float w=sigmoid(sa[j]-sb[j]);out[j]=a[j]*w+b[j]*(1-w);}}
