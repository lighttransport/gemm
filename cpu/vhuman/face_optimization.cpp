/* Included by face.cpp. Scalar objective blocks and native optimizers. */
namespace {
template<class T> void adam_step(T *x,const T *g,T *m,T *v,const double *lr,size_t n,int step,double clip) {
    finite(x,n);finite(g,n);finite(m,n);finite(v,n);finite(lr,n);
    require(step>0 && std::isfinite(clip) && clip>=0,"invalid Adam state");
    double norm=0;for(size_t i=0;i<n;++i){require(v[i]>=0 && lr[i]>=0,"invalid Adam moments/rate");norm+=double(g[i])*g[i];}
    double scale=clip>0?std::min(1.,clip/(std::sqrt(norm)+1e-6)):1;
    double b1=1-std::pow(.9,step),b2=std::sqrt(1-std::pow(.999,step));
    for(size_t i=0;i<n;++i) {
        T grad=T(g[i]*scale);m[i]=T(.9*m[i]+.1*grad);v[i]=T(.999*v[i]+.001*grad*grad);
        x[i]=T(x[i]-lr[i]/b1*m[i]/(std::sqrt(double(v[i]))/b2+1e-8));
    }
}
double dot(const std::vector<float>&a,const std::vector<float>&b) {double s=0;for(size_t i=0;i<a.size();++i)s+=double(a[i])*b[i];return s;}
double maximum(const std::vector<float>&a) {double x=0;for(float y:a)x=std::max(x,double(std::abs(y)));return x;}
}
extern "C" int vhf_adam(float *x,const float *g,float *m,float *v,const double *lr,int n,int step,double clip) {
    try { require(n>0 && n<=10000000,"invalid optimizer dimension");adam_step(x,g,m,v,lr,n,step,clip);return 0; }
    catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int vhf_adam64(double *x,const double *g,double *m,double *v,const double *lr,int n,int step,double clip) {
    try { require(n>0 && n<=10000000,"invalid optimizer dimension");adam_step(x,g,m,v,lr,n,step,clip);return 0; }
    catch(const std::exception &e){error=e.what();return -1;}
}
/* Callback returns zero on success; cancellation/errors propagate immediately.
 * No exception may cross the callback ABI. Strong Wolfe c1=1e-4,c2=.9. */
extern "C" int vhf_lbfgs(float *x,int n,int iterations,int max_evals,
    int (*objective)(void*,const float*,float*,double*),void *user) {
    try {
        require(x && objective && n>0 && n<=1000000 && iterations>0 && iterations<=2000 && max_evals>=2,"invalid L-BFGS arguments");
        finite(x,n);std::vector<float> point(x,x+n),g(n),d(n),trial(n),tg(n),old_g(n);
        std::vector<std::vector<float>> ss,yy;std::vector<double> rho;int evals=0;
        auto eval=[&](const std::vector<float>&p,std::vector<float>&grad) {
            double f=0;require(objective(user,p.data(),grad.data(),&f)==0,"objective cancelled or failed");
            require(std::isfinite(f),"nonfinite objective");finite(grad.data(),grad.size());++evals;return f;
        };
        double f=eval(point,g),last_step=0;std::vector<float> previous_direction;
        for(int it=0;it<iterations;++it) {
            if(maximum(g)<=1e-7)break;
            if(it) {
                std::vector<float>s(n),y(n);for(int j=0;j<n;++j){s[j]=float(previous_direction[j]*last_step);y[j]=g[j]-old_g[j];}
                double sy=dot(s,y);
                if(sy>1e-10) { if(ss.size()==100){ss.erase(ss.begin());yy.erase(yy.begin());rho.erase(rho.begin());}ss.push_back(s);yy.push_back(y);rho.push_back(1/sy); }
            }
            d=g;for(float &v:d)v=-v;std::vector<double> alpha(ss.size());
            for(int k=int(ss.size())-1;k>=0;--k) { alpha[k]=dot(ss[k],d)*rho[k];for(int j=0;j<n;++j)d[j]-=float(alpha[k]*yy[k][j]); }
            double scale=ss.empty()?1:dot(ss.back(),yy.back())/dot(yy.back(),yy.back());for(float &v:d)v=float(v*scale);
            for(size_t k=0;k<ss.size();++k) { double beta=dot(yy[k],d)*rho[k];for(int j=0;j<n;++j)d[j]+=float((alpha[k]-beta)*ss[k][j]); }
            double gd=dot(g,d);if(gd>-1e-9)break;
            double rate=.5;if(!it){double sum=0;for(float v:g)sum+=std::abs(v);rate*=std::min(1.,1/sum);}
            old_g=g;double old_f=f;
            struct End {double t,f,gd;std::vector<float> g;};
            End low{0,f,gd,g},high=low,prev=low,chosen=low;bool bracket=false,accepted=false;
            // Interpolation is safeguarded; the bracket always contains the
            // last Armijo step. At budget exhaustion retain that finite state.
            auto interpolate=[](const End&a,const End&b,double lo,double hi) {
                if(a.t==b.t)return (lo+hi)/2;
                double d1=a.gd+b.gd-3*(a.f-b.f)/(a.t-b.t),rad=d1*d1-a.gd*b.gd;
                if(rad>=0) {double d2=std::sqrt(rad);if(b.t<a.t)d2=-d2;double den=b.gd-a.gd+2*d2;
                    if(std::abs(den)>1e-30){double t=b.t-(b.t-a.t)*(b.gd+d2-d1)/den;if(std::isfinite(t))return std::clamp(t,lo,hi);}}
                return (lo+hi)/2;
            };
            int ls=0;
            for(;ls<25;++ls) {
                for(int j=0;j<n;++j)trial[j]=point[j]+float(rate*d[j]);
                End cur{rate,eval(trial,tg),dot(tg,d),tg};
                if(cur.f>f+1e-4*rate*gd || (ls>0 && cur.f>=prev.f)) {low=prev;high=cur;bracket=true;break;}
                if(std::abs(cur.gd)<=-.9*gd) {chosen=cur;accepted=true;break;}
                if(cur.gd>=0) {low=cur;high=prev;bracket=true;break;}
                double next=interpolate(prev,cur,rate+.01*(rate-prev.t),rate*10);prev=cur;rate=next;
            }
            if(!accepted && !bracket){chosen=prev;accepted=true;}
            for(;bracket && !accepted && ls<25;++ls) {
                double a=std::min(low.t,high.t),b=std::max(low.t,high.t);
                if((b-a)*maximum(d)<1e-9){chosen=low;accepted=true;break;}
                rate=interpolate(low,high,a+.1*(b-a),b-.1*(b-a));
                for(int j=0;j<n;++j)trial[j]=point[j]+float(rate*d[j]);
                End cur{rate,eval(trial,tg),dot(tg,d),tg};
                if(cur.f>f+1e-4*rate*gd || cur.f>=low.f)high=cur;
                else { if(std::abs(cur.gd)<=-.9*gd){chosen=cur;accepted=true;break;}
                    if(cur.gd*(high.t-low.t)>=0)high=low;
                    low=cur; }
            }
            if(!accepted)chosen=low;
            last_step=chosen.t;previous_direction=d;f=chosen.f;g=chosen.g;
            for(int j=0;j<n;++j)point[j]+=float(last_step*d[j]);
            if(evals>=max_evals || maximum(d)*last_step<=1e-9 || std::abs(f-old_f)<1e-9)break;
        }
        std::copy(point.begin(),point.end(),x);return 0;
    } catch(const std::exception &e){error=e.what();return -1;}
}
extern "C" int vhf_registration_loss(const double *x,int vertices,const int32_t *ids,int count,
    const double *target,const double *normal,const int32_t *edges,int edge_count,const double *length,
    double *gradient,double *energy) {
    try {
        require(vertices>0 && vertices<=1000000 && count>0 && count<=vertices && edge_count>0 && edge_count<=6000000 && ids && edges && gradient && energy,"invalid registration dimensions");
        finite(x,size_t(vertices)*3);finite(target,size_t(count)*3);finite(normal,size_t(count)*3);finite(length,edge_count);
        std::fill(gradient,gradient+size_t(vertices)*3,0.);std::fill(energy,energy+4,0.);
        std::vector<int> degree(vertices);std::vector<double> lap(size_t(vertices)*3),dl(size_t(vertices)*3);
        for(int e=0;e<edge_count;++e) {
            int a=edges[2*e],b=edges[2*e+1];require(a>=0 && a<vertices && b>=0 && b<vertices,"invalid registration edge");++degree[a];++degree[b];double l=0;
            for(int c=0;c<3;++c){double d=x[a*3+c]-x[b*3+c];l+=d*d;lap[a*3+c]+=x[b*3+c];lap[b*3+c]+=x[a*3+c];}
            l=std::sqrt(l);double delta=l-length[e];energy[2]+=delta*delta/edge_count;
            double g=l>0?2*.3e6*delta/(edge_count*l):0;
            for(int c=0;c<3;++c){double z=g*(x[a*3+c]-x[b*3+c]);gradient[a*3+c]+=z;gradient[b*3+c]-=z;}
        }
        for(int k=0;k<count;++k) {
            int id=ids[k];require(id>=0 && id<vertices,"invalid movable vertex");double plane=0;
            for(int c=0;c<3;++c)plane+=(x[id*3+c]-target[k*3+c])*normal[k*3+c];
            energy[0]+=plane*plane/count;
            for(int c=0;c<3;++c) {
                double d=x[id*3+c]-target[k*3+c],l=lap[id*3+c]/std::max(1,degree[id])-x[id*3+c];
                energy[1]+=d*d/count;energy[3]+=l*l/count;dl[id*3+c]=2*.05e6*l/count;
                gradient[id*3+c]+=2e6*(plane*normal[k*3+c]+.05*d)/count-dl[id*3+c];
            }
        }
        for(int e=0;e<edge_count;++e)for(int c=0;c<3;++c) {
            int a=edges[e*2],b=edges[e*2+1];gradient[a*3+c]+=dl[b*3+c]/std::max(1,degree[b]);gradient[b*3+c]+=dl[a*3+c]/std::max(1,degree[a]);
        }
        finite(gradient,size_t(vertices)*3);return 0;
    } catch(const std::exception &e){error=e.what();return -1;}
}
