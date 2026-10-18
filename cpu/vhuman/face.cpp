/* Full GNM equations and their explicit reverse pass. Apache-2.0 model
 * equations: google/GNM. No inference or differentiation runtime. */
#include "../../common/vhuman_face.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#ifdef VHF_HAVE_AVX2
#include "../../ryzen/gemm_avx2.h"
#endif

namespace {
thread_local std::string error;
void require(bool ok,const char *s) { if(!ok)throw std::invalid_argument(s); }
template<class T> void finite(const T *p,size_t n) {
    require(p,"missing native face array");
    for(size_t i=0;i<n;++i)require(std::isfinite(p[i]),"nonfinite native face array");
}
void gemm(float *out,const float *a,const float *b,int m,int n,int k,bool ta,bool tb,int threads) {
    std::vector<float> aa,bb;
    if(ta) { aa.resize(size_t(m)*k);for(int i=0;i<m;++i)for(int j=0;j<k;++j)aa[size_t(i)*k+j]=a[size_t(j)*m+i];a=aa.data(); }
    if(tb) { bb.resize(size_t(k)*n);for(int i=0;i<k;++i)for(int j=0;j<n;++j)bb[size_t(i)*n+j]=b[size_t(j)*k+i];b=bb.data(); }
#ifdef VHF_HAVE_AVX2
    if(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        // Split rows for large batches, retaining cache-friendly GEMM blocks.
        const int jobs=m>=32?std::min(threads,m):1;
        #pragma omp parallel for num_threads(jobs) if(jobs>1)
        for(int j=0;j<jobs;++j) {
            int lo=int(int64_t(m)*j/jobs),hi=int(int64_t(m)*(j+1)/jobs);
            sgemm_avx2(hi-lo,n,k,1,a+size_t(lo)*k,k,b,n,0,out+size_t(lo)*n,n);
        }
        return;
    }
#endif
    #pragma omp parallel for num_threads(threads) if(m>8)
    for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
        double s=0;for(int q=0;q<k;++q)s+=double(a[size_t(i)*k+q])*b[size_t(q)*n+j];
        out[size_t(i)*n+j]=float(s);
    }
}
using M=std::array<double,9>;
using V=std::array<double,3>;
M mul(const M &a,const M &b) { M c{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)c[i*3+j]+=a[i*3+k]*b[k*3+j];return c; }
M transpose(const M &a) { M b;for(int i=0;i<3;++i)for(int j=0;j<3;++j)b[i*3+j]=a[j*3+i];return b; }
V mv(const M &a,const V &v) { V b{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)b[i]+=a[i*3+j]*v[j];return b; }
M outer(const V &a,const V &b) { M c;for(int i=0;i<3;++i)for(int j=0;j<3;++j)c[i*3+j]=a[i]*b[j];return c; }
template<size_t N> void add(std::array<double,N> &a,const std::array<double,N> &b,double scale=1) { for(size_t i=0;i<N;++i)a[i]+=scale*b[i]; }
void rotation(const double *r,M &out,std::array<M,3> *derivative=nullptr) {
    double s=r[0]*r[0]+r[1]*r[1]+r[2]*r[2],a,b,da,db;
    // Derivatives da/db are with respect to theta squared; no singular axis.
    if(s<1e-6) {
        a=1-s/6+s*s/120-s*s*s/5040;b=.5-s/24+s*s/720-s*s*s/40320;
        da=-1./6+s/60-s*s/1680;db=-1./24+s/360-s*s/13440;
    } else {
        double t=std::sqrt(s);a=std::sin(t)/t;b=(1-std::cos(t))/s;
        da=(t*std::cos(t)-std::sin(t))/(2*s*t);db=(t*std::sin(t)/2-(1-std::cos(t)))/(s*s);
    }
    M k{0,-r[2],r[1],r[2],0,-r[0],-r[1],r[0],0},kk=mul(k,k);
    for(int i=0;i<9;++i)out[i]=(i%4==0?1.:0.)+a*k[i]+b*kk[i];
    if(derivative) {
        const M dk[3]={{0,0,0,0,0,-1,0,1,0},{0,0,1,0,0,0,-1,0,0},{0,-1,0,1,0,0,0,0,0}};
        for(int c=0;c<3;++c) {
            M dkk=mul(dk[c],k);add(dkk,mul(k,dk[c]));
            for(int i=0;i<9;++i)(*derivative)[c][i]=2*r[c]*(da*k[i]+db*kk[i])+a*dk[c][i]+b*dkk[i];
        }
    }
}
}
struct vhf_gnm {
    int v,i,e,threads;std::array<int,4> parent;
    std::vector<float> rest,joints,identity,joint_identity,expression,correctives,weights;
};
extern "C" int vhf_abi_version() { return 1; }
extern "C" const char *vhf_error() { return error.c_str(); }
extern "C" void vhf_gnm_close(vhf_gnm *m) { delete m; }
extern "C" int vhf_gemm(float *out,const float *a,const float *b,int m,int n,int k,int ta,int tb,int threads) {
    try {
        require(out && m>0 && n>0 && k>0 && m<=65536 && n<=65536 && k<=65536 &&
                size_t(m)*n<=134217728 && size_t(m)*k<=134217728 && size_t(n)*k<=134217728 &&
                (ta==0||ta==1) && (tb==0||tb==1) && threads>0 && threads<=256,"invalid GEMM dimensions");
        finite(a,size_t(m)*k);finite(b,size_t(n)*k);gemm(out,a,b,m,n,k,ta,tb,threads);finite(out,size_t(m)*n);return 0;
    } catch(const std::exception &e) { error=e.what();return -1; }
}
extern "C" int vhf_rodrigues(const double *r,double *out,double *deriv,int count) {
    try {
        require(out && count>0 && count<=65536,"invalid rotation count");finite(r,size_t(count)*3);
        for(int i=0;i<count;++i) { M m;std::array<M,3>d;rotation(r+i*3,m,deriv?&d:nullptr);std::copy(m.begin(),m.end(),out+i*9);if(deriv)for(int j=0;j<3;++j)std::copy(d[j].begin(),d[j].end(),deriv+i*27+j*9); }
        return 0;
    } catch(const std::exception &e) { error=e.what();return -1; }
}
extern "C" vhf_gnm *vhf_gnm_open(int v,int i,int e,int threads,const int32_t *parents,
    const float *rest,const float *joints,const float *identity,const float *ji,
    const float *expression,const float *correctives,const float *weights) {
    try {
        require(v>0 && v<=65536 && i>0 && i<=512 && e>0 && e<=512 && threads>0 && threads<=256 && parents,"invalid GNM dimensions");
        require(size_t(v)*3*(i+e+36)<=134217728,"GNM model exceeds allocation limit");
        auto m=std::make_unique<vhf_gnm>();m->v=v;m->i=i;m->e=e;m->threads=threads;
        for(int j=0;j<4;++j) { require((j==0 && parents[j]==-1)||(j>0 && parents[j]>=0 && parents[j]<j),"invalid joint hierarchy");m->parent[j]=parents[j]; }
        auto copy=[](std::vector<float> &out,const float *p,size_t n) { finite(p,n);out.assign(p,p+n); };
        copy(m->rest,rest,size_t(v)*3);copy(m->joints,joints,12);copy(m->identity,identity,size_t(i)*v*3);
        copy(m->joint_identity,ji,size_t(i)*12);copy(m->expression,expression,size_t(e)*v*3);
        copy(m->correctives,correctives,size_t(36)*v*3);copy(m->weights,weights,size_t(4)*v);
        for(int p=0;p<v;++p) { double sum=0;for(int j=0;j<4;++j){require(weights[j*v+p]>=0,"negative skin weight");sum+=weights[j*v+p];}require(std::abs(sum-1)<1e-4,"skin weights must sum to one"); }
        return m.release();
    } catch(const std::exception &e) { error=e.what();return nullptr; }
}
extern "C" int vhf_gnm_eval(vhf_gnm *m,int frames,const float *beta,const float *expr,const float *rot,
    const float *trans,const float *residual,float *positions,float *joints,const float *dp,const float *dj,
    float *dbeta,float *dexpr,float *drot,float *dtrans) {
    try {
        require(m && frames>0 && frames<=1024 && positions && joints && size_t(frames)*m->v*3<=67108864,"invalid GNM batch");
        int v=m->v,i=m->i,e=m->e,d=v*3;bool backward=dp||dj;
        finite(beta,i);finite(expr,size_t(frames)*e);finite(rot,size_t(frames)*12);finite(trans,size_t(frames)*3);
        if(residual)finite(residual,d);
        if(backward) { require(dbeta && dexpr && drot && dtrans,"missing GNM gradient buffers");if(dp)finite(dp,size_t(frames)*d);if(dj)finite(dj,size_t(frames)*12);std::fill(dbeta,dbeta+i,0); }
        std::vector<float> bind(d),jj(12),local(size_t(frames)*d),pose(size_t(frames)*36),corrective(size_t(frames)*d);
        gemm(bind.data(),beta,m->identity.data(),1,d,i,false,false,m->threads);
        gemm(jj.data(),beta,m->joint_identity.data(),1,12,i,false,false,1);
        for(int p=0;p<d;++p)bind[p]+=m->rest[p]+(residual?residual[p]:0);
        for(int p=0;p<12;++p)jj[p]+=m->joints[p];
        gemm(local.data(),expr,m->expression.data(),frames,d,e,false,false,m->threads);
        std::vector<std::array<M,4>> lr(frames),wr(frames);
        std::vector<std::array<V,4>> wt(frames);
        std::vector<std::array<std::array<M,3>,4>> dr(backward?frames:0);
        for(int f=0;f<frames;++f)for(int j=0;j<4;++j) {
            double r[3];for(int a=0;a<3;++a)r[a]=rot[f*12+j*3+a];rotation(r,lr[f][j],backward?&dr[f][j]:nullptr);
            for(int a=0;a<9;++a)pose[f*36+j*9+a]=float(lr[f][j][a]-(a%4==0?1:0));
            if(j==0) { wr[f][j]=lr[f][j];for(int a=0;a<3;++a)wt[f][j][a]=jj[a]+trans[f*3+a]; }
            else { int p=m->parent[j];wr[f][j]=mul(wr[f][p],lr[f][j]);V delta;for(int a=0;a<3;++a)delta[a]=jj[j*3+a]-jj[p*3+a];wt[f][j]=mv(wr[f][p],delta);add(wt[f][j],wt[f][p]); }
            for(int a=0;a<3;++a)joints[f*12+j*3+a]=float(wt[f][j][a]);
        }
        gemm(corrective.data(),pose.data(),m->correctives.data(),frames,d,36,false,false,m->threads);
        for(size_t p=0;p<local.size();++p)local[p]+=bind[p%d]+corrective[p];
        std::vector<float> dl(backward?local.size():0),db(backward?size_t(frames)*i:0),dc(backward?size_t(frames)*36:0);
        std::vector<std::array<M,4>> gwr(backward?frames:0);
        std::vector<std::array<V,4>> gwt(backward?frames:0),gj(backward?frames:0);
        #pragma omp parallel for num_threads(m->threads) if(frames>1)
        for(int f=0;f<frames;++f) {
            if(backward && dj)for(int j=0;j<4;++j)for(int a=0;a<3;++a)gwt[f][j][a]=dj[f*12+j*3+a];
            for(int p=0;p<v;++p) {
                V out{},gradient{},dx{};if(dp)for(int a=0;a<3;++a)gradient[a]=dp[size_t(f)*d+p*3+a];
                for(int j=0;j<4;++j) {
                    double w=m->weights[j*v+p];if(w==0)continue;
                    V x;for(int a=0;a<3;++a)x[a]=local[size_t(f)*d+p*3+a]-jj[j*3+a];
                    V q=mv(wr[f][j],x);add(q,wt[f][j]);add(out,q,w);
                    if(backward) {
                        V dg=mv(transpose(wr[f][j]),gradient);add(dx,dg,w);add(gj[f][j],dg,-w);
                        add(gwr[f][j],outer(gradient,x),w);add(gwt[f][j],gradient,w);
                    }
                }
                for(int a=0;a<3;++a) { positions[size_t(f)*d+p*3+a]=float(out[a]);if(backward)dl[size_t(f)*d+p*3+a]=float(dx[a]); }
            }
        }
        if(backward) {
            gemm(dexpr,dl.data(),m->expression.data(),frames,e,d,false,true,m->threads);
            gemm(db.data(),dl.data(),m->identity.data(),frames,i,d,false,true,m->threads);
            gemm(dc.data(),dl.data(),m->correctives.data(),frames,36,d,false,true,m->threads);
            for(int f=0;f<frames;++f) {
                std::array<M,4> gl{};
                for(int j=3;j>0;--j) {
                    int p=m->parent[j];V delta;for(int a=0;a<3;++a)delta[a]=jj[j*3+a]-jj[p*3+a];
                    gl[j]=mul(transpose(wr[f][p]),gwr[f][j]);
                    add(gwr[f][p],mul(gwr[f][j],transpose(lr[f][j])));
                    add(gwr[f][p],outer(gwt[f][j],delta));add(gwt[f][p],gwt[f][j]);
                    V dg=mv(transpose(wr[f][p]),gwt[f][j]);add(gj[f][j],dg);add(gj[f][p],dg,-1);
                }
                gl[0]=gwr[f][0];add(gj[f][0],gwt[f][0]);
                for(int a=0;a<3;++a)dtrans[f*3+a]=float(gwt[f][0][a]);
                for(int j=0;j<4;++j) {
                    for(int a=0;a<9;++a)gl[j][a]+=dc[f*36+j*9+a];
                    for(int a=0;a<3;++a) { double s=0;for(int k=0;k<9;++k)s+=gl[j][k]*dr[f][j][a][k];drot[f*12+j*3+a]=float(s); }
                }
                for(int k=0;k<i;++k) { double s=db[f*i+k];for(int j=0;j<4;++j)for(int a=0;a<3;++a)s+=gj[f][j][a]*m->joint_identity[k*12+j*3+a];dbeta[k]+=float(s); }
            }
            finite(dbeta,i);finite(dexpr,size_t(frames)*e);finite(drot,size_t(frames)*12);finite(dtrans,size_t(frames)*3);
        }
        finite(positions,size_t(frames)*d);finite(joints,size_t(frames)*12);return 0;
    } catch(const std::exception &e) { error=e.what();return -1; }
}

#include "face_image.cpp"
#include "face_optimization.cpp"
#include "face_losses.cpp"
