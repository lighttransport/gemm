/* Included by training.cpp: CPU trace-v1 Gaussian fitting. Native projection
 * uses four local dual derivatives; compositing uses an analytic reverse pass.
 * Visibility/order/tile membership are piecewise constant, as in splat fitting. */
#include <numeric>
struct appearance_dual {
    double value,derivative[4]{};
    appearance_dual(double v=0):value(v) {}
    static appearance_dual variable(double v,int i) { appearance_dual x(v);x.derivative[i]=1;return x; }
};
static appearance_dual operator+(const appearance_dual &a,const appearance_dual &b) {
    appearance_dual x(a.value+b.value);for(int i=0;i<4;++i)x.derivative[i]=a.derivative[i]+b.derivative[i];return x;
}
static appearance_dual operator-(const appearance_dual &a,const appearance_dual &b) {
    appearance_dual x(a.value-b.value);for(int i=0;i<4;++i)x.derivative[i]=a.derivative[i]-b.derivative[i];return x;
}
static appearance_dual operator*(const appearance_dual &a,const appearance_dual &b) {
    appearance_dual x(a.value*b.value);for(int i=0;i<4;++i)x.derivative[i]=a.derivative[i]*b.value+a.value*b.derivative[i];return x;
}
static appearance_dual operator/(const appearance_dual &a,const appearance_dual &b) {
    appearance_dual x(a.value/b.value);for(int i=0;i<4;++i)x.derivative[i]=(a.derivative[i]-x.value*b.derivative[i])/b.value;return x;
}
static appearance_dual appearance_exp(const appearance_dual &a) {
    appearance_dual x(std::exp(a.value));for(int i=0;i<4;++i)x.derivative[i]=x.value*a.derivative[i];return x;
}
static appearance_dual appearance_tanh(const appearance_dual &a) {
    appearance_dual x(std::tanh(a.value));for(int i=0;i<4;++i)x.derivative[i]=(1-x.value*x.value)*a.derivative[i];return x;
}
static appearance_dual appearance_bound(const appearance_dual &a,double lo,double hi) {
    return a.value<lo?appearance_dual(lo):a.value>hi?appearance_dual(hi):a;
}
struct appearance_projection {
    double p[12]{},jacobian[5][4]{};
    bool valid=false;
};
static double appearance_sigmoid(double x) { return x>=0?1/(1+std::exp(-x)):std::exp(x)/(1+std::exp(x)); }
static appearance_projection appearance_project(const float *parameters,const float *vertices,const int32_t *ids,
    const float *bary,const float *camera,const float *coefficient,int width,int height)
{
    appearance_projection result;double points[9],basis[9],normal[3];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)points[i*3+j]=vertices[ids[i]*3+j];
    for(int j=0;j<3;++j) { basis[j*3]=points[3+j]-points[j];basis[j*3+1]=points[6+j]-points[j]; }
    normal[0]=basis[3]*basis[7]-basis[6]*basis[4];normal[1]=basis[6]*basis[1]-basis[0]*basis[7];normal[2]=basis[0]*basis[4]-basis[3]*basis[1];
    double length=std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    if(length<=1e-10)return result;
    for(int j=0;j<3;++j)basis[j*3+2]=normal[j]/std::max(length,1e-12);
    using d=appearance_dual;d center[3],scale2[3],covariance[9],cam_point[3],cam_cov[9];
    d offset=appearance_tanh(d::variable(parameters[7],0))*.005;
    double limits[]={1,1,.002};
    for(int i=0;i<3;++i) { d scale=appearance_bound(appearance_exp(d::variable(parameters[4+i],1+i)),1e-5,limits[i]);scale2[i]=scale*scale; }
    for(int i=0;i<3;++i) {
        center[i]=points[i]*bary[0]+points[3+i]*bary[1]+points[6+i]*bary[2]+offset*basis[i*3+2];
        for(int j=0;j<3;++j) { covariance[i*3+j]=i==j?1e-10:0;for(int k=0;k<3;++k)covariance[i*3+j]=covariance[i*3+j]+scale2[k]*basis[i*3+k]*basis[j*3+k]; }
    }
    d trace=covariance[0]+covariance[4]+covariance[8];d bound=trace.value>1e-4?d(1e-4)/trace:d(1);
    for(auto &v:covariance)v=v*bound;
    for(int i=0;i<3;++i) {
        cam_point[i]=camera[i*4+3];for(int k=0;k<3;++k)cam_point[i]=cam_point[i]+center[k]*camera[i*4+k];
        for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)cam_cov[i*3+j]=cam_cov[i*3+j]+covariance[k*3+l]*camera[i*4+k]*camera[j*4+l];
    }
    double opacity=appearance_sigmoid(parameters[3]);d z=cam_point[2];
    if(z.value<.01||z.value>1e10||opacity<1./255)return result;
    double fx=camera[16],fy=camera[20],cx=camera[18],cy=camera[21];
    d tx=appearance_bound(cam_point[0]/z,-(cx+.15*width)/fx,(width-cx+.15*width)/fx);
    d ty=appearance_bound(cam_point[1]/z,-(cy+.15*height)/fy,(height-cy+.15*height)/fy);
    d j[6]={d(fx)/z,0,d(-fx)*tx/z,0,d(fy)/z,d(-fy)*ty/z},s[4];
    for(int u=0;u<2;++u)for(int v=0;v<2;++v)for(int k=0;k<3;++k)for(int l=0;l<3;++l)s[u*2+v]=s[u*2+v]+j[u*3+k]*cam_cov[k*3+l]*j[v*3+l];
    s[0]=s[0]+.3;s[3]=s[3]+.3;d determinant=s[0]*s[3]-s[1]*s[2];
    if(determinant.value<=0||!std::isfinite(determinant.value))return result;
    double extent=std::min(3.33,std::sqrt(2*std::log(opacity*255)));
    double rx=std::ceil(extent*std::sqrt(s[0].value)),ry=std::ceil(extent*std::sqrt(s[3].value));
    d values[]={cam_point[0]*fx/z+cx,cam_point[1]*fy/z+cy,s[3]/determinant,d(-1)*s[1]/determinant,s[0]/determinant};
    if(values[0].value+rx<=0||values[0].value-rx>=width||values[1].value+ry<=0||values[1].value-ry>=height)return result;
    for(int i=0;i<5;++i) { result.p[i<2?i:i+1]=values[i].value;for(int k=0;k<4;++k)result.jacobian[i][k]=values[i].derivative[k]; }
    result.p[2]=z.value;result.p[6]=opacity;result.p[10]=rx;result.p[11]=ry;
    for(int c=0;c<3;++c) { double color=appearance_sigmoid(parameters[c]);for(int k=0;k<8;++k)color+=parameters[8+k*3+c]*coefficient[k];result.p[7+c]=std::clamp(color,0.,1.); }
    for(double value:result.p)require(std::isfinite(value),"nonfinite Gaussian projection");
    result.valid=true;return result;
}
extern "C" int vh_train_appearance(const float *parameters,const float *vertices,const int32_t *attachments,
    const float *bary,const float *controls,const float *camera,int n,int v,int c,int width,int height,
    const float *truth,const float *mask,float *rgba,float *gradient,double *loss)
{
    try {
        require(n>0&&n<=200000&&v>=3&&v<=2000000&&c>0&&c<=512&&width>0&&width<=4096&&height>0&&height<=4096&&rgba&&loss,"invalid appearance dimensions");
        size_t count=size_t(n)*32+size_t(c)*8,pixels=size_t(width)*height;
        finite(parameters,count);finite(vertices,size_t(v)*3);indices(attachments,size_t(n)*3,v);
        finite(bary,size_t(n)*3);finite(controls,c);finite(camera,25);
        require(camera[16]>0&&camera[20]>0,"invalid appearance focal length");
        if(gradient) { finite(truth,pixels*3);finite(mask,pixels);std::fill(gradient,gradient+count,0.f); }
        float coefficient[8];gemm(coefficient,controls,parameters+size_t(n)*32,1,8,c);
        for(float &x:coefficient)x=std::tanh(x);
        std::vector<appearance_projection> projected(n);
        for(int i=0;i<n;++i)projected[i]=appearance_project(parameters+size_t(i)*32,vertices,attachments+i*3,bary+i*3,camera,coefficient,width,height);
        int tw=(width+15)/16,th=(height+15)/16,tiles=tw*th;
        std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return projected[a].p[2]<projected[b].p[2];});
        std::vector<std::vector<int>> tile_ids(tiles);size_t overlaps=0;
        for(int i:order)if(projected[i].valid) {
            const double *p=projected[i].p;
            int x0=int(std::clamp(std::floor((p[0]-p[10])/16),0.,double(tw))),x1=int(std::clamp(std::ceil((p[0]+p[10])/16),0.,double(tw)));
            int y0=int(std::clamp(std::floor((p[1]-p[11])/16),0.,double(th))),y1=int(std::clamp(std::ceil((p[1]+p[11])/16),0.,double(th)));
            for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x) { require(++overlaps<=8000000,"Gaussian tile overlap budget exceeded");tile_ids[y*tw+x].push_back(i); }
        }
        std::vector<double> projected_gradient(gradient?size_t(n)*9:0,0.);
        struct hit { int id;double alpha,trans,dx,dy,gaussian;bool capped; };
        std::vector<hit> hits;double l1=0,total=0;
        for(int y=0;y<height;++y)for(int x=0;x<width;++x) {
            size_t pixel=size_t(y)*width+x;double trans=1,color[3]={};hits.clear();
            for(int id:tile_ids[(y/16)*tw+x/16]) {
                const double *p=projected[id].p;double dx=p[0]-(x+.5),dy=p[1]-(y+.5);
                double sigma=.5*(p[3]*dx*dx+p[5]*dy*dy)+p[4]*dx*dy,g=std::exp(-sigma),raw_alpha=p[6]*g,a=std::min(.99,raw_alpha);
                if(sigma<0||a<1./255)continue;
                double next=trans*(1-a);if(next<=1e-4)break;
                if(gradient)hits.push_back({id,a,trans,dx,dy,g,raw_alpha>=.99});
                for(int k=0;k<3;++k)color[k]+=a*trans*p[7+k];
                trans=next;
            }
            for(int k=0;k<3;++k)rgba[pixel*4+k]=float(color[k]);
            rgba[pixel*4+3]=float(1-trans);
            if(gradient) {
                double dc[3];for(int k=0;k<3;++k) { double error=color[k]-truth[pixel*3+k];l1+=std::abs(error)/(pixels*3);dc[k]=((error>0)-(error<0))/double(pixels*3); }
                double da=1-trans-mask[pixel];require(mask[pixel]>=0&&mask[pixel]<=1,"invalid appearance mask");
                total+=.05*std::abs(da)/pixels;double dt=-.05*((da>0)-(da<0))/pixels;
                for(auto it=hits.rbegin();it!=hits.rend();++it) {
                    const hit &h=*it;const double *p=projected[h.id].p;double *g=projected_gradient.data()+size_t(h.id)*9,dot=0;
                    for(int k=0;k<3;++k) { dot+=dc[k]*p[7+k];g[6+k]+=dc[k]*h.alpha*h.trans; }
                    double d_alpha=h.trans*(dot-dt);dt=h.alpha*dot+(1-h.alpha)*dt;
                    if(!h.capped) {
                        g[5]+=d_alpha*h.gaussian;double dsigma=-d_alpha*h.alpha;
                        g[0]+=dsigma*(p[3]*h.dx+p[4]*h.dy);g[1]+=dsigma*(p[5]*h.dy+p[4]*h.dx);
                        g[2]+=dsigma*.5*h.dx*h.dx;g[3]+=dsigma*h.dx*h.dy;g[4]+=dsigma*.5*h.dy*h.dy;
                    }
                }
            }
        }
        if(gradient) {
            double dcoefficient[8]={};
            for(int i=0;i<n;++i) {
                const float *p=parameters+size_t(i)*32;float *g=gradient+size_t(i)*32;const double *pg=projected_gradient.data()+size_t(i)*9;
                for(int k=0;k<4;++k) { double value=0;for(int j=0;j<5;++j)value+=pg[j]*projected[i].jacobian[j][k];g[k?3+k:7]=float(value); }
                double opacity=appearance_sigmoid(p[3]);g[3]=float(pg[5]*opacity*(1-opacity));
                for(int channel=0;channel<3;++channel) {
                    double base=appearance_sigmoid(p[channel]),color=base;
                    for(int k=0;k<8;++k)color+=p[8+k*3+channel]*coefficient[k];
                    double dc=color>=0&&color<=1?pg[6+channel]:0;g[channel]=float(dc*base*(1-base));
                    for(int k=0;k<8;++k) { g[8+k*3+channel]=float(dc*coefficient[k]);dcoefficient[k]+=dc*p[8+k*3+channel]; }
                }
                for(int j=8;j<32;++j) { total+=1e-4*double(p[j])*p[j]/(size_t(n)*24);g[j]+=float(2e-4*p[j]/(size_t(n)*24)); }
            }
            float dc[8];for(int k=0;k<8;++k)dc[k]=float(dcoefficient[k]*(1-coefficient[k]*coefficient[k]));
            gemm(gradient+size_t(n)*32,controls,dc,c,8,1);
            finite(gradient,count);
        }
        finite(rgba,pixels*4);loss[0]=total+l1;loss[1]=l1;return 0;
    } catch(const std::exception &e) { train_error=e.what();return -1; }
}
