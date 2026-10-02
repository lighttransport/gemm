#ifndef VHUMAN_TRAINING_KERNELS_H
#define VHUMAN_TRAINING_KERNELS_H
/* Native CUDA training; projection dual derivatives share CPU trace-v1 math. */
static const char *vh_training_source = R"CUDA(
typedef int int32_t;
struct appearance_dual {
    double value,derivative[4]{};
    __device__ appearance_dual(double v=0):value(v) {}
    __device__ static appearance_dual variable(double v,int i) { appearance_dual x(v);x.derivative[i]=1;return x; }
};
__device__ static appearance_dual operator+(const appearance_dual &a,const appearance_dual &b) {
    appearance_dual x(a.value+b.value);for(int i=0;i<4;++i)x.derivative[i]=a.derivative[i]+b.derivative[i];return x;
}
__device__ static appearance_dual operator-(const appearance_dual &a,const appearance_dual &b) {
    appearance_dual x(a.value-b.value);for(int i=0;i<4;++i)x.derivative[i]=a.derivative[i]-b.derivative[i];return x;
}
__device__ static appearance_dual operator*(const appearance_dual &a,const appearance_dual &b) {
    appearance_dual x(a.value*b.value);for(int i=0;i<4;++i)x.derivative[i]=a.derivative[i]*b.value+a.value*b.derivative[i];return x;
}
__device__ static appearance_dual operator/(const appearance_dual &a,const appearance_dual &b) {
    appearance_dual x(a.value/b.value);for(int i=0;i<4;++i)x.derivative[i]=(a.derivative[i]-x.value*b.derivative[i])/b.value;return x;
}
__device__ static appearance_dual appearance_exp(const appearance_dual &a) {
    appearance_dual x(exp(a.value));for(int i=0;i<4;++i)x.derivative[i]=x.value*a.derivative[i];return x;
}
__device__ static appearance_dual appearance_tanh(const appearance_dual &a) {
    appearance_dual x(tanh(a.value));for(int i=0;i<4;++i)x.derivative[i]=(1-x.value*x.value)*a.derivative[i];return x;
}
__device__ static appearance_dual appearance_bound(const appearance_dual &a,double lo,double hi) {
    return a.value<lo?appearance_dual(lo):a.value>hi?appearance_dual(hi):a;
}
struct appearance_projection {
    double p[12]{},jacobian[5][4]{};
    bool valid=false;
};
__device__ static double appearance_sigmoid(double x) { return x>=0?1/(1+exp(-x)):exp(x)/(1+exp(x)); }
__device__ static appearance_projection appearance_project(const float *parameters,const float *vertices,const int32_t *ids,
    const float *bary,const float *camera,const float *coefficient,int width,int height)
{
    appearance_projection result;double points[9],basis[9],normal[3];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)points[i*3+j]=vertices[ids[i]*3+j];
    for(int j=0;j<3;++j) { basis[j*3]=points[3+j]-points[j];basis[j*3+1]=points[6+j]-points[j]; }
    normal[0]=basis[3]*basis[7]-basis[6]*basis[4];normal[1]=basis[6]*basis[1]-basis[0]*basis[7];normal[2]=basis[0]*basis[4]-basis[3]*basis[1];
    double length=sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    if(length<=1e-10)return result;
    for(int j=0;j<3;++j)basis[j*3+2]=normal[j]/fmax(length,1e-12);
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
    if(determinant.value<=0||!isfinite(determinant.value))return result;
    double extent=fmin(3.33,sqrt(2*log(opacity*255)));
    double rx=ceil(extent*sqrt(s[0].value)),ry=ceil(extent*sqrt(s[3].value));
    d values[]={cam_point[0]*fx/z+cx,cam_point[1]*fy/z+cy,s[3]/determinant,d(-1)*s[1]/determinant,s[0]/determinant};
    if(values[0].value+rx<=0||values[0].value-rx>=width||values[1].value+ry<=0||values[1].value-ry>=height)return result;
    for(int i=0;i<5;++i) { result.p[i<2?i:i+1]=values[i].value;for(int k=0;k<4;++k)result.jacobian[i][k]=values[i].derivative[k]; }
    result.p[2]=z.value;result.p[6]=opacity;result.p[10]=rx;result.p[11]=ry;
    for(int c=0;c<3;++c) { double color=appearance_sigmoid(parameters[c]);for(int k=0;k<8;++k)color+=parameters[8+k*3+c]*coefficient[k];result.p[7+c]=fmin(fmax(color,0.),1.); }
    result.valid=true;return result;
}

extern "C" {
__global__ void check_finite(const float *p,int n,int *invalid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&!isfinite(p[i]))atomicExch(invalid,1);
}
__global__ void adam(float *p,const float *g,float *m,float *v,int n,double lr,double wd,double b1,double b2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double grad=g[i],mm=.9*double(m[i])+.1*grad,vv=.999*double(v[i])+.001*grad*grad;
    m[i]=mm;v[i]=vv;
    p[i]=double(p[i])*(1-lr*wd)-lr*(mm/b1)/(sqrt(vv/b2)+1e-8);
}
__global__ void tanh8(float *x) { int i=threadIdx.x;if(i<8)x[i]=tanhf(x[i]); }
__global__ void project_train(const float *p,const float *verts,const int *ids,const float *bary,
    const float *camera,const float *coeff,int n,int w,int h,double *projection,double *jac) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    auto q=appearance_project(p+i*32,verts,ids+i*3,bary+i*3,camera,coeff,w,h);
    for(int j=0;j<12;++j)projection[i*12+j]=q.p[j];
    for(int j=0;j<5;++j)for(int k=0;k<4;++k)jac[i*20+j*4+k]=q.jacobian[j][k];
}
__global__ void raster_train(const double *p,const int *offsets,const int *ids,int w,int h,
    const float *truth,const float *mask,float *out,double *pg,double *loss,int backward) {
    int tw=(w+15)/16,tile=blockIdx.x,x=(tile%tw)*16+threadIdx.x%16,y=(tile/tw)*16+threadIdx.x/16;
    if(x>=w||y>=h)return;
    int pixel=y*w+x,start=offsets[tile],end=offsets[tile+1],last=start-1;
    double trans=1,color[3]={};
    for(int j=start;j<end;++j) {
        const double *q=p+ids[j]*12;double dx=q[0]-(x+.5),dy=q[1]-(y+.5);
        double sigma=.5*(q[3]*dx*dx+q[5]*dy*dy)+q[4]*dx*dy,a=fmin(.99,q[6]*exp(-sigma));
        if(sigma<0||a<1./255)continue;
        double next=trans*(1-a);if(next<=1e-4)break;
        for(int k=0;k<3;++k)color[k]+=a*trans*q[7+k];
        trans=next;last=j;
    }
    for(int k=0;k<3;++k)out[pixel*4+k]=color[k];out[pixel*4+3]=1-trans;
    if(!backward)return;
    double dc[3],l1=0;
    for(int k=0;k<3;++k) {double e=color[k]-truth[pixel*3+k];l1+=fabs(e)/(double(w)*h*3);dc[k]=((e>0)-(e<0))/(double(w)*h*3);}
    double da=1-trans-mask[pixel],dt=-.05*((da>0)-(da<0))/(double(w)*h);
    atomicAdd(loss,l1+.05*fabs(da)/(double(w)*h));atomicAdd(loss+1,l1);
    // Replay only accepted prefix, then reconstruct T_before backwards. This
    // avoids an unbounded per-pixel hit tape even for 200k splats.
    for(int j=last;j>=start;--j) {
        int id=ids[j];const double *q=p+id*12;double dx=q[0]-(x+.5),dy=q[1]-(y+.5);
        double sigma=.5*(q[3]*dx*dx+q[5]*dy*dy)+q[4]*dx*dy,gauss=exp(-sigma),raw=q[6]*gauss,a=fmin(.99,raw);
        if(sigma<0||a<1./255)continue;
        double before=trans/(1-a),dot=0,*g=pg+id*9;
        for(int k=0;k<3;++k) {dot+=dc[k]*q[7+k];atomicAdd(g+6+k,dc[k]*a*before);}
        double d_alpha=before*(dot-dt);dt=a*dot+(1-a)*dt;trans=before;
        if(raw<.99) {
            atomicAdd(g+5,d_alpha*gauss);double ds=-d_alpha*a;
            atomicAdd(g,ds*(q[3]*dx+q[4]*dy));atomicAdd(g+1,ds*(q[5]*dy+q[4]*dx));
            atomicAdd(g+2,ds*.5*dx*dx);atomicAdd(g+3,ds*dx*dy);atomicAdd(g+4,ds*.5*dy*dy);
        }
    }
}
__global__ void chain_train(const float *p,const double *pg,const double *jac,const float *coeff,
    int n,float *grad,double *dcoeff,double *loss) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    p+=i*32;float *g=grad+i*32;pg+=i*9;jac+=i*20;
    for(int k=0;k<4;++k) {double v=0;for(int j=0;j<5;++j)v+=pg[j]*jac[j*4+k];g[k?3+k:7]=v;}
    double opacity=appearance_sigmoid(p[3]);g[3]=pg[5]*opacity*(1-opacity);
    for(int c=0;c<3;++c) {
        double base=appearance_sigmoid(p[c]),color=base;
        for(int k=0;k<8;++k)color+=double(p[8+k*3+c])*coeff[k];
        double dc=(color>=0&&color<=1)?pg[6+c]:0;g[c]=dc*base*(1-base);
        for(int k=0;k<8;++k) {g[8+k*3+c]=dc*coeff[k];atomicAdd(dcoeff+k,dc*p[8+k*3+c]);}
    }
    double regularizer=0;
    for(int j=8;j<32;++j) {regularizer+=1e-4*double(p[j])*p[j]/(double(n)*24);g[j]+=2e-4*p[j]/(double(n)*24);}
    atomicAdd(loss,regularizer);
}
__global__ void coeff_reverse(const double *dc,const float *coeff,float *out) {
    int k=threadIdx.x;if(k<8)out[k]=dc[k]*(1-double(coeff[k])*coeff[k]);
}
__global__ void layout_input(const float *in,float *out,int pixels,int channels,int area) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=pixels*channels)return;
    int c=i%channels,p=i/channels;out[i]=in[(p/area*channels+c)*area+p%area];
}
__global__ void columns(const float *in,float *out,int n,int c,int ih,int iw,int oh,int ow,int kernel,int stride) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,k=c*kernel*kernel;if(i>=n*oh*ow*k)return;
    int j=i%k,row=i/k,ch=j/(kernel*kernel),ky=j/kernel%kernel,kx=j%kernel;
    int x=row%ow*stride+kx-kernel/2,y=row/ow%oh*stride+ky-kernel/2,b=row/(oh*ow);
    out[i]=(x>=0&&x<iw&&y>=0&&y<ih)?in[((b*ih+y)*iw+x)*c+ch]:0;
}
__global__ void conv_finish(float *z,float *out,const float *bias,int elements,int c,int active) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=elements)return;
    float v=z[i]+bias[i%c];z[i]=v;out[i]=active?v/(1+expf(-v)):v;
}
__global__ void conv_reverse(float *dy,const float *z,int elements,int active) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=elements||!active)return;
    float s=1/(1+expf(-z[i]));dy[i]*=s*(1+z[i]*(1-s));
}
__global__ void bias_reverse(const float *dy,float *g,int pixels,int c) {
    __shared__ float sum[256];int ch=blockIdx.x,t=threadIdx.x;float acc=0;
    for(int i=t;i<pixels;i+=256)acc+=dy[i*c+ch];sum[t]=acc;__syncthreads();
    for(int s=128;s;s>>=1){if(t<s)sum[t]+=sum[t+s];__syncthreads();}if(!t)g[ch]=sum[0];
}
__global__ void col_reverse(const float *col,float *out,int n,int c,int ih,int iw,int oh,int ow,int kernel,int stride) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*ih*iw*c)return;
    int ch=i%c,x=i/c%iw,y=i/c/iw%ih,b=i/c/(ih*iw),k=c*kernel*kernel;float sum=0;
    for(int ky=0;ky<kernel;++ky)for(int kx=0;kx<kernel;++kx) {
        int yy=y+kernel/2-ky,xx=x+kernel/2-kx;
        if(yy<0||xx<0||yy%stride||xx%stride)continue;yy/=stride;xx/=stride;
        if(yy<oh&&xx<ow)sum+=col[((b*oh+yy)*ow+xx)*k+(ch*kernel+ky)*kernel+kx];
    }out[i]=sum;
}
__device__ void coordinate(int i,int input,int output,int &a,int &b,float &weight) {
    float p=fmaxf(0,(i+.5f)*input/output-.5f);a=min(int(p),input-1);b=min(int(p)+1,input-1);weight=p-int(p);
}
__global__ void resize_train(const float *in,float *out,int n,int c,int ih,int iw,int oh,int ow,int reverse) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*oh*ow*c)return;
    int ch=i%c,x=i/c%ow,y=i/c/ow%oh,b=i/c/(oh*ow),ya,yb,xa,xb;float u,v;
    coordinate(y,ih,oh,ya,yb,u);coordinate(x,iw,ow,xa,xb,v);
    int aa=((b*ih+ya)*iw+xa)*c+ch,ab=((b*ih+ya)*iw+xb)*c+ch,ba=((b*ih+yb)*iw+xa)*c+ch,bb=((b*ih+yb)*iw+xb)*c+ch;
    if(!reverse)out[i]=(1-u)*((1-v)*in[aa]+v*in[ab])+u*((1-v)*in[ba]+v*in[bb]);
    else {float d=in[i];atomicAdd(out+aa,d*(1-u)*(1-v));atomicAdd(out+ab,d*(1-u)*v);atomicAdd(out+ba,d*u*(1-v));atomicAdd(out+bb,d*u*v);}
}
__global__ void cue_loss(const float *raw,const float *inputs,const float *truth,const float *mask,
    float *out,float *draw,int pixels,int area,double denominator,double *loss,int backward) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=pixels)return;
    float t[3],v[3],normal[3],norm2=0;
    for(int c=0;c<3;++c){t[c]=tanhf(raw[p*4+c]);v[c]=inputs[p*6+3+c]+.15f*t[c];norm2+=v[c]*v[c];}
    float norm=sqrtf(norm2),divisor=fmaxf(norm,1e-6f);
    for(int c=0;c<3;++c){normal[c]=v[c]/divisor;out[(p/area*4+c)*area+p%area]=normal[c];}
    float logit=raw[p*4+3];out[(p/area*4+3)*area+p%area]=logit;
    if(!backward)return;
    float m=mask[p],dn[3],dot=0,cosine=0;double total=.15*(fmaxf(logit,0)-logit*m+log1pf(expf(-fabsf(logit))))/pixels;
    for(int c=0;c<3;++c){float target=truth[(p/area*3+c)*area+p%area],d=normal[c]-inputs[p*6+3+c];
        cosine+=normal[c]*target;total+=.02*double(d)*d*m/denominator;dn[c]=(-target+.04*d)*m/denominator;dot+=dn[c]*normal[c];}
    total+=(1-cosine)*m/denominator;
    for(int c=0;c<3;++c)draw[p*4+c]=(dn[c]-(norm>=1e-6f?normal[c]*dot:0))/divisor*.15f*(1-t[c]*t[c]);
    draw[p*4+3]=.15f*(1/(1+expf(-logit))-m)/pixels;atomicAdd(loss,total);
}
}

)CUDA";
#endif
